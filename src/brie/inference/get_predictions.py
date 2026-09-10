'''
Prediction script
This script will take in the following:
1. Question file: question_id contains {patient_id}_{question number}, question
2. Notes Directory: Json files labeled with {patient_id}_subsetrecords.json (ordered most recent to least recent)
3. Output path: file to write all answers

For each question in the question file, send an api call for claude, gemini, and gpt using the question and notes input.

Outfile will have the following format:
-question_id
-response
-model (claude|gemini|gpt)
'''

import argparse
import asyncio
import json
import logging
import os

import pandas as pd

from .utils import send_single_message, send_batch_messages, VERTEX_BATCH_MODELS, MODEL_CONTEXT_LIMITS, ALL_MODELS, count_tokens, VLLM_MODELS, CsvWriter, load_completed_pairs, load_force_ids, drop_rows_for_id_models, log_token_stats, set_tpm_limit, filter_records_as_of

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_DELAY = 25  # seconds; set --delay 0 when running vLLM-only models

# Some providers count tokens differently, so a budget that looks
# under the limit can still 500 with "prompt is too long". On each 500, shave this many tokens
# off the budget, re-truncate, and retry (claude_opus only — see run_inference).
OPUS_PROMPT_BACKOFF = 100_000

PROMPT_TEMPLATE = """\
You are a clinical assistant. Using only the patient notes provided, answer the \
following clinical question as accurately and concisely as possible. If the notes \
do not contain enough information to answer the question, say so explicitly.

The question is asked as of {QUERY_DATE}.

Question:
{QUESTION}

Patient Notes:
{NOTES}

Answer:"""


def format_notes(records: list[dict], token_budget: int) -> tuple[str, int, int, int]:
    """
    Format notes most-recent to least-recent, stopping before exceeding token_budget.
    Returns (formatted_notes_string, notes_included, notes_total, tokens_used).
    """
    parts = []
    used = 0
    for i, item in enumerate(records):
        chunk = (
            f"Note Title: {item['note_title']}\n"
            f"Note Date: {item['note_date']}\n"
            f"Text: {item['text']}\n"
        )
        chunk_tokens = count_tokens(chunk)
        if used + chunk_tokens > token_budget:
            logger.debug(f"Token limit reached: including {i}/{len(records)} notes ({used:,} tokens)")
            return "\n".join(parts), i, len(records), used
        parts.append(chunk)
        used += chunk_tokens
    return "\n".join(parts), len(records), len(records), used


def load_patient_records(notes_dir: str, patient_id: str) -> list[dict] | None:
    """Load raw records list for a patient. Returns None on failure."""
    notes_path = os.path.join(notes_dir, f"{patient_id}_subsetrecords.json")
    try:
        with open(notes_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Notes file not found for patient '{patient_id}': {notes_path}")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse notes for patient '{patient_id}': {e}")
        return None


async def run_inference(question_id: str, question: str, query_date: str, records: list[dict], model: str, backend: str, context_limit: int | None = None) -> dict:
    """Format notes for this model's context window and run inference.

    For claude_opus, a 500 "prompt is too long" can occur even after our tiktoken-based
    truncation (Bedrock counts more tokens than tiktoken). On each 500, shave
    OPUS_PROMPT_BACKOFF tokens off the budget, re-truncate, and retry until it fits.
    """
    records = filter_records_as_of(records, query_date)
    token_budget = min(MODEL_CONTEXT_LIMITS[model], context_limit) if context_limit else MODEL_CONTEXT_LIMITS[model]

    while True:
        notes_str, included, total, tokens_used = format_notes(records, token_budget)

        if included < total:
            logger.warning(
                f"[{model}] question_id={question_id}: truncated to {included}/{total} notes "
                f"({tokens_used:,}/{token_budget:,} tokens)"
            )
        else:
            logger.info(
                f"[{model}] question_id={question_id}: {included}/{total} notes "
                f"({tokens_used:,}/{token_budget:,} tokens)"
            )

        try:
            response = await send_single_message(
                user_prompt=PROMPT_TEMPLATE.format(QUESTION=question, QUERY_DATE=query_date, NOTES=notes_str),
                model_id=model,
                backend=backend,
            )
            logger.info(f"Completed: question_id={question_id}, model={model}")
            return {"question_id": question_id, "model": model, "response": response, "error": None}
        except Exception as e:
            # claude_opus only: "prompt is too long" (a 500 from Bedrock's higher token
            # count) — drop the budget by 100K and retry until it fits.
            too_long = "prompt is too long" in str(e).lower() or "500" in str(e)
            if model == "claude_opus" and too_long and token_budget > OPUS_PROMPT_BACKOFF:
                token_budget -= OPUS_PROMPT_BACKOFF
                logger.warning(
                    f"[claude_opus] question_id={question_id}: 500 error; reducing token budget "
                    f"to {token_budget:,} and retrying. ({e})"
                )
                continue
            logger.error(f"Inference failed: question_id={question_id}, model={model}, error={e}")
            return {"question_id": question_id, "model": model, "response": None, "error": str(e)}


async def main(args):
    if args.tpm_limit:
        set_tpm_limit(args.tpm_limit)
        logger.info(f"TPM rate limiter enabled: {args.tpm_limit:,} tokens/min")

    df = pd.read_csv(args.questions)
    df = df[["question_id", args.question_column, "timestamp"]].drop_duplicates()
    logger.info(f"Loaded {len(df)} questions from {args.questions}")

    force_ids = load_force_ids(args.force_ids, args.force_ids_file)
    if force_ids:
        df = df[df["question_id"].astype(str).isin(force_ids)]
        removed = drop_rows_for_id_models(args.output, "question_id", force_ids, args.models)
        logger.info(
            f"--force-ids: {len(df)} question(s) kept for {len(force_ids)} forced id(s); "
            f"dropped {removed} stale row(s) from {args.output}"
        )

    completed = load_completed_pairs(args.output, ["question_id", "model"], nonempty_col="response")
    if completed:
        logger.info(f"Resuming: {len(completed)} (question_id, model) pairs already done")

    records_cache: dict[str, list[dict] | None] = {}
    for _, row in df.iterrows():
        patient_id = str(row["question_id"]).split("_")[0]
        if patient_id not in records_cache:
            records_cache[patient_id] = load_patient_records(args.notes, patient_id)

    tasks = []
    for _, row in df.iterrows():
        patient_id = str(row["question_id"]).split("_")[0]
        records = records_cache.get(patient_id)
        if records is None:
            logger.warning(f"Skipping question_id={row['question_id']} — notes unavailable.")
            continue
        for model in args.models:
            if (str(row["question_id"]), model.removesuffix("_batch")) in completed:
                logger.debug(f"Already done: question_id={row['question_id']} model={model} — skipping")
                continue
            safe_records = filter_records_as_of(records, row["timestamp"])
            tasks.append((row["question_id"], row[args.question_column], str(row["timestamp"]), safe_records, model, args.context_limit))

    if not tasks:
        logger.info("All tasks already complete.")
        return

    # Split: Vertex batch models → batch API (50% discount); everything else → staggered concurrent
    batch_tasks  = [(qid, q, ts, rec, m, ctx) for qid, q, ts, rec, m, ctx in tasks
                    if m in VERTEX_BATCH_MODELS and args.backend == "vertex"]
    other_tasks  = [(qid, q, ts, rec, m, ctx) for qid, q, ts, rec, m, ctx in tasks
                    if m not in VERTEX_BATCH_MODELS or args.backend != "vertex"]

    logger.info(
        f"{len(tasks)} tasks total: {len(batch_tasks)} via Vertex batch API, "
        f"{len(other_tasks)} via concurrent calls"
    )

    if args.dry_run:
        from collections import Counter
        per_model_reqs   = Counter()
        per_model_tokens = Counter()
        for qid, question, query_date, records, model, context_limit in tasks:
            token_budget = min(MODEL_CONTEXT_LIMITS[model], context_limit) if context_limit else MODEL_CONTEXT_LIMITS[model]
            notes_str, *_ = format_notes(records, token_budget)
            prompt = PROMPT_TEMPLATE.format(QUESTION=question, QUERY_DATE=query_date, NOTES=notes_str)
            per_model_reqs[model]   += 1
            per_model_tokens[model] += count_tokens(prompt)
        total_tokens = sum(per_model_tokens.values())
        logger.info("── Dry-run estimate ──────────────────────────────")
        logger.info(f"  Tasks total:    {len(tasks):>10,}  ({len(batch_tasks)} batch, {len(other_tasks)} concurrent)")
        for model in sorted(per_model_reqs):
            pathway = "batch" if (model in VERTEX_BATCH_MODELS and args.backend == "vertex") else "concurrent"
            logger.info(
                f"    {model:<22} {per_model_reqs[model]:>7,} reqs  "
                f"{per_model_tokens[model]:>12,} input tok  [{pathway}]"
            )
        logger.info(f"  Input tokens:   {total_tokens:>10,}  (post-truncation; excludes model output)")
        if len(batch_tasks) > 200_000:
            logger.warning(f"  *** {len(batch_tasks):,} batch requests exceeds the 200K-per-job limit — split into multiple runs ***")
        elif batch_tasks:
            logger.info(f"  Batch job limit: OK ({len(batch_tasks):,} / 200,000)")
        logger.info("  No requests submitted (--dry-run).")
        return

    writer = CsvWriter(args.output, ["question_id", "model", "response"])

    # ── Vertex batch: single batch submission ─────────────────────────────────
    if batch_tasks:
        batch_requests = []
        task_meta: dict[str, tuple] = {}
        for qid, question, query_date, records, model, context_limit in batch_tasks:
            token_budget = min(MODEL_CONTEXT_LIMITS[model], context_limit) if context_limit else MODEL_CONTEXT_LIMITS[model]
            notes_str, included, total, _ = format_notes(records, token_budget)
            if included < total:
                logger.warning(f"[{model}] {qid}: truncated to {included}/{total} notes")
            custom_id = f"{qid}||{model}"
            task_meta[custom_id] = (qid, model)
            batch_requests.append({
                "custom_id": custom_id,
                "user_prompt": PROMPT_TEMPLATE.format(QUESTION=question, QUERY_DATE=query_date, NOTES=notes_str),
                "model_id": model,
            })

        logger.info(f"Submitting {len(batch_requests)} tasks to Vertex batch API...")
        batch_results = await send_batch_messages(batch_requests)

        for custom_id, response in batch_results.items():
            qid, model = task_meta[custom_id]
            if response is None:
                logger.error(f"Batch failed: question_id={qid}, model={model}")
            display_model = model.removesuffix("_batch")
            await writer.write({"question_id": qid, "model": display_model, "response": response})
            logger.info(f"Batch result written: question_id={qid}, model={display_model}")

    # ── Other models: staggered concurrent calls ──────────────────────────────
    if other_tasks:
        delay = args.delay
        rpm_str = f"~{60 // delay} RPM" if delay > 0 else "unlimited (vLLM)"
        logger.info(f"{len(other_tasks)} tasks at {delay}s intervals ({rpm_str})")

        async def delayed(i, question_id, question, query_date, records, model, context_limit):
            task_delay = 0 if model in VLLM_MODELS else delay
            await asyncio.sleep(i * task_delay)
            result = await run_inference(question_id, question, query_date, records, model, args.backend, context_limit)
            if result["error"]:
                logger.warning(f"Writing failed result for question_id={question_id} model={model}")
            await writer.write(result)
            return result

        results = await asyncio.gather(*[
            delayed(i, qid, q, ts, rec, model, ctx)
            for i, (qid, q, ts, rec, model, ctx) in enumerate(other_tasks)
        ])

        failed = sum(1 for r in results if r["error"])
        if failed:
            logger.warning(f"{failed} task(s) failed — see errors above.")

    logger.info(f"Done. Wrote {len(tasks)} rows to {args.output}")
    log_token_stats(logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QA Generation Pipeline")
    parser.add_argument("-q", "--questions", type=str, required=True,
                    help="CSV file containing questions")
    parser.add_argument("-n", "--notes", type=str, required=True,
                    help="Directory containing patient notes")
    parser.add_argument("-o", "--output", type=str, required=True,
                    help="Path to output CSV")
    parser.add_argument("-c", "--question-column", type=str, default="natural_query",
                    help="Column name in the questions CSV containing the question text (default: natural_query)")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
                    metavar="MODEL",
                    help=f"Models to run (default: all). Choices: {', '.join(ALL_MODELS)}")
    parser.add_argument("--backend", choices=["vertex", "openai"], default="vertex",
                    help="Backend for multi-backend models (default: vertex). Secure GPT and deepseek are unaffected.")
    parser.add_argument("--delay", type=int, default=DEFAULT_DELAY,
                    help=f"Seconds between API requests (default: {DEFAULT_DELAY}). Use 0 for vLLM-only runs.")
    parser.add_argument("--tpm-limit", type=int, default=None,
                    help="Enable a sliding-window tokens-per-minute rate limiter.")
    parser.add_argument("--context-limit", type=int, default=None,
                    help="Cap token budget per model at this value (e.g. 128000). Defaults to each model's full context.")
    parser.add_argument("--force-ids", nargs="+", default=None, metavar="QUESTION_ID",
                    help="Re-run only these question_ids, dropping their stale output rows first.")
    parser.add_argument("--force-ids-file", type=str, default=None,
                    help="File with one question_id per line to re-run (unioned with --force-ids).")
    parser.add_argument("--dry-run", action="store_true",
                    help="Report per-model request and input-token counts without submitting any requests.")
    args = parser.parse_args()
    asyncio.run(main(args))
