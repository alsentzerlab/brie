'''
Rolling inference script.

Processes notes oldest-first in batches. The first batch uses the standard
prompt; each subsequent batch presents the current answer alongside the next
batch of notes and asks the model to revise only if the new context warrants
it.

Output CSV columns:
  question_id, model, approach, batch_num, total_batches,
  notes_start_idx, notes_end_idx, response
'''

import argparse
import asyncio
import json
import logging
import os

import pandas as pd

from .utils import send_single_message, MODEL_CONTEXT_LIMITS, ALL_MODELS, count_tokens, VLLM_MODELS, CsvWriter, load_rolling_progress, filter_records_as_of

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_DELAY = 30  # seconds; set --delay 0 when running vLLM-only models

# ── Prompts ────────────────────────────────────────────────────────────────────
INITIAL_PROMPT = """\
You are a clinical assistant. Using only the patient notes provided, answer the \
following clinical question as accurately and concisely as possible. If the notes \
do not contain enough information to answer the question, say so explicitly.

The question is asked as of {QUERY_DATE}. The most recent patient note is dated \
{LATEST_NOTE_DATE}. Notes are presented in chronological order, oldest first.

Question:
{QUESTION}

Patient Notes:
{NOTES}

Answer:"""

UPDATE_PROMPT = """\
You are a clinical assistant reviewing additional patient notes. Your current \
answer to the question is shown below. Using the new notes provided, revise your \
answer if—and only if—the new information warrants a change. If the new notes do \
not add relevant information, return your current answer unchanged.

The question is asked as of {QUERY_DATE}. The most recent patient note is dated \
{LATEST_NOTE_DATE}. Notes are presented in chronological order, oldest first.

Question:
{QUESTION}

Current Answer:
{PREVIOUS_ANSWER}

New Patient Notes:
{NOTES}

Revised Answer:"""


def format_notes_batch(records: list[dict]) -> str:
    """Format a list of note records into a prompt string."""
    parts = []
    for item in records:
        parts.append(
            f"Note Title: {item['note_title']}\n"
            f"Note Date: {item['note_date']}\n"
            f"Text: {item['text']}\n"
        )
    return "\n".join(parts)


def build_batches(records: list[dict], token_budget: int) -> list[list[dict]]:
    """
    Split records into non-overlapping batches, each fitting within token_budget.
    Records are assumed to be in oldest-first order already.
    """
    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_tokens = 0

    for record in records:
        chunk = (
            f"Note Title: {record['note_title']}\n"
            f"Note Date: {record['note_date']}\n"
            f"Text: {record['text']}\n\n"
        )
        chunk_tokens = count_tokens(chunk)
        if current_tokens + chunk_tokens > token_budget and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(record)
        current_tokens += chunk_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


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


async def run_rolling_inference(
    question_id: str,
    question: str,
    query_date: str,
    records: list[dict],
    model: str,
    latest_note_date: str,
    backend: str = "vertex",
    delay: int = DEFAULT_DELAY,
    writer: CsvWriter | None = None,
    start_batch: int = 0,
    initial_answer: str | None = None,
) -> list[dict]:
    """
    Run rolling inference for one (question, model) pair.

    start_batch / initial_answer allow resuming a partial run: batches before
    start_batch are skipped, and initial_answer is used as the seeded current answer.

    Returns one result dict per *new* batch written.
    """
    records = filter_records_as_of(records, query_date)
    token_budget = MODEL_CONTEXT_LIMITS[model]

    # Records arrive newest-first; reverse so batches are processed oldest-first
    oldest_first = list(reversed(records))
    batches = build_batches(oldest_first, token_budget)
    total_batches = len(batches)

    results: list[dict] = []
    current_answer: str | None = initial_answer
    # Advance cursor past already-completed batches
    note_cursor = sum(len(batches[i]) for i in range(start_batch))

    for batch_num, batch in enumerate(batches):
        if batch_num < start_batch:
            continue
        notes_str = format_notes_batch(batch)
        batch_tokens = count_tokens(notes_str)
        notes_start = note_cursor
        notes_end = note_cursor + len(batch) - 1
        note_cursor += len(batch)
        logger.info(
            f"[{model}] question_id={question_id}: batch {batch_num + 1}/{total_batches}, "
            f"{len(batch)} notes, {batch_tokens:,}/{token_budget:,} tokens"
        )

        if batch_num == 0:
            prompt = INITIAL_PROMPT.format(
                QUESTION=question,
                NOTES=notes_str,
                QUERY_DATE=query_date,
                LATEST_NOTE_DATE=latest_note_date,
            )
        else:
            prompt = UPDATE_PROMPT.format(
                QUESTION=question,
                PREVIOUS_ANSWER=current_answer,
                NOTES=notes_str,
                QUERY_DATE=query_date,
                LATEST_NOTE_DATE=latest_note_date,
            )

        batch_delay = 0 if model in VLLM_MODELS else delay
        if batch_num > 0:
            await asyncio.sleep(batch_delay)

        try:
            response = await send_single_message(user_prompt=prompt, model_id=model, backend=backend)
            logger.info(
                f"Completed: question_id={question_id}, model={model}, "
                f"batch={batch_num + 1}/{total_batches}"
            )
        except Exception as e:
            logger.error(
                f"Inference failed: question_id={question_id}, model={model}, "
                f"batch={batch_num + 1}/{total_batches}, error={e}"
            )
            response = None

        current_answer = response

        row = {
            "question_id":    question_id,
            "model":          model,
            "approach":       "rolling",
            "batch_num":      batch_num,
            "total_batches":  total_batches,
            "notes_start_idx": notes_start,
            "notes_end_idx":   notes_end,
            "response":       response,
        }
        results.append(row)
        if writer is not None:
            await writer.write(row)

        # If inference failed, stop rolling for this pair
        if response is None:
            break

    return results


_ROLLING_FIELDS = [
    "question_id", "model", "approach", "batch_num", "total_batches",
    "notes_start_idx", "notes_end_idx", "response",
]


async def main(args):
    df = pd.read_csv(args.questions)
    df = df[["question_id", args.question_column, "timestamp"]].drop_duplicates()
    logger.info(f"Loaded {len(df)} questions from {args.questions}")

    completed, partial = load_rolling_progress(args.output)
    if completed:
        logger.info(f"Resuming: {len(completed)} (question_id, model) pairs already done")
    if partial:
        logger.info(f"Resuming: {len(partial)} (question_id, model) pairs partially done")

    records_cache: dict[str, list[dict] | None] = {}
    for _, row in df.iterrows():
        patient_id = row["question_id"].split("_")[0]
        if patient_id not in records_cache:
            records_cache[patient_id] = load_patient_records(args.notes, patient_id)

    tasks = []
    for _, row in df.iterrows():
        patient_id = row["question_id"].split("_")[0]
        records = records_cache.get(patient_id)
        if records is None:
            logger.warning(f"Skipping question_id={row['question_id']} — notes unavailable.")
            continue
        # Records are newest-first; records[0] is the most recent note
        latest_note_date = records[0]["note_date"]
        for model in args.models:
            key = (str(row["question_id"]), str(model))
            if key in completed:
                logger.debug(f"Already done: question_id={row['question_id']} model={model} — skipping")
                continue
            resume_batch, resume_answer = partial[key] if key in partial else (0, None)
            if resume_batch > 0:
                logger.info(
                    f"Resuming from batch {resume_batch + 1}: "
                    f"question_id={row['question_id']} model={model}"
                )
            safe_records = filter_records_as_of(records, row["timestamp"])
            tasks.append((
                row["question_id"], row[args.question_column], str(row["timestamp"]), safe_records, model,
                latest_note_date, resume_batch + 1 if resume_batch > 0 else 0, resume_answer,
            ))

    if not tasks:
        logger.info("All tasks already complete.")
        return

    writer = CsvWriter(args.output, _ROLLING_FIELDS)

    # Pre-compute remaining batch counts (excluding already-done batches) for rate-limit offsets
    batch_counts = [
        len(build_batches(list(reversed(rec)), MODEL_CONTEXT_LIMITS[model])) - start_b
        for _, _, _, rec, model, _, start_b, _ in tasks
    ]
    delay = args.delay
    total_api_calls = sum(batch_counts)
    eta_minutes = (total_api_calls * delay) / 60
    rpm_str = f"~{60 // delay} RPM" if delay > 0 else "unlimited (vLLM)"
    logger.info(
        f"{len(tasks)} tasks, {total_api_calls} total API calls at "
        f"{delay}s intervals ({rpm_str}) — ETA ≥{eta_minutes:.0f} min"
    )

    cumulative_offsets = [sum(batch_counts[:i]) for i in range(len(tasks))]

    async def delayed(offset, question_id, question, query_date, records, model, latest_note_date, start_b, resume_ans):
        task_delay = 0 if model in VLLM_MODELS else delay
        await asyncio.sleep(offset * task_delay)
        return await run_rolling_inference(
            question_id, question, query_date, records, model, latest_note_date,
            args.backend, delay, writer,
            start_batch=start_b, initial_answer=resume_ans,
        )

    nested_results = await asyncio.gather(*[
        delayed(cumulative_offsets[i], qid, q, ts, rec, model, lnd, start_b, resume_ans)
        for i, (qid, q, ts, rec, model, lnd, start_b, resume_ans) in enumerate(tasks)
    ])

    total_rows = sum(len(batch_rows) for batch_rows in nested_results)
    logger.info(f"Done. Wrote {total_rows} new rows to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rolling inference: oldest-first batched context updates"
    )
    parser.add_argument("-q", "--questions", type=str, required=True,
                        help="CSV file containing questions")
    parser.add_argument("-n", "--notes", type=str, required=True,
                        help="Directory containing patient notes JSON files")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to output CSV")
    parser.add_argument("-c", "--question-column", type=str, default="natural_query",
                        help="Column name for the question text (default: natural_query)")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
                        metavar="MODEL",
                        help=f"Models to run (default: all). Choices: {', '.join(ALL_MODELS)}")
    parser.add_argument("--backend", choices=["vertex", "openai"], default="vertex",
                        help="Backend for multi-backend models (default: vertex). Secure GPT and deepseek are unaffected.")
    parser.add_argument("--delay", type=int, default=DEFAULT_DELAY,
                        help=f"Seconds between API requests (default: {DEFAULT_DELAY}). Use 0 for vLLM-only runs.")
    args = parser.parse_args()
    asyncio.run(main(args))
