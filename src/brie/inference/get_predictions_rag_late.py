'''
RAG inference script — LateOn (ColBERT late-interaction) retrieval.

For each question, retrieves the top-K most relevant note chunks using
a per-patient PLAID index (ColBERT late-interaction), then runs
inference with the specified LLM(s).

PLAID indexes are built on first run and cached in --embeddings-dir.
Each patient gets its own index folder ({patient_id}_plaid/) alongside
a chunk metadata pickle ({patient_id}_late_chunks.pkl).

Output CSV columns:
  question_id, model, approach, response, retrieved_chunks
  (retrieved_chunks is a JSON string of chunk metadata + scores)
'''

import argparse
import asyncio
import json
import logging
import os

import pandas as pd

from .rag_utils import (
    load_cached_late_embeddings,
    build_late_embeddings_batch,
    encode_queries_late,
    retrieve_late_precomputed,
    format_retrieved_notes,
    serialise_retrieved,
)
from .utils import send_single_message, send_batch_messages, VERTEX_BATCH_MODELS, ALL_MODELS, VLLM_MODELS, CsvWriter, load_completed_pairs, load_force_ids, drop_rows_for_id_models, log_token_stats, set_tpm_limit, parse_timestamp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_DELAY = 10  # seconds; set --delay 0 when running vLLM-only models

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


def load_patient_records(notes_dir: str, patient_id: str) -> list[dict] | None:
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


async def run_inference(
    question_id: str,
    question: str,
    query_date: str,
    query_emb,
    chunks,
    index,
    model: str,
    top_k: int,
    backend: str = "vertex",
) -> dict:
    cutoff = parse_timestamp(query_date)
    if cutoff is None:
        raise ValueError(f"invalid query timestamp: {query_date!r}")
    valid = [position for position, chunk in enumerate(chunks) if (stamp := parse_timestamp(chunk.get("note_date"))) is not None and stamp <= cutoff]
    retrieved = retrieve_late_precomputed(query_emb, [chunks[position] for position in valid], [index[position] for position in valid], top_k=top_k)
    notes_str = format_retrieved_notes(retrieved)

    try:
        response = await send_single_message(
            user_prompt=PROMPT_TEMPLATE.format(QUESTION=question, QUERY_DATE=query_date, NOTES=notes_str),
            model_id=model,
            backend=backend,
        )
        logger.info(f"Completed: question_id={question_id}, model={model}")
        return {
            "question_id":      question_id,
            "model":            model,
            "approach":         "rag_late",
            "response":         response,
            "retrieved_chunks": serialise_retrieved(retrieved),
        }
    except Exception as e:
        logger.error(f"Inference failed: question_id={question_id}, model={model}, error={e}")
        return {
            "question_id":      question_id,
            "model":            model,
            "approach":         "rag_late",
            "response":         None,
            "retrieved_chunks": serialise_retrieved(retrieved),
        }


_LATE_FIELDS = ["question_id", "model", "approach", "response", "retrieved_chunks"]


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

    os.makedirs(args.embeddings_dir, exist_ok=True)

    # Pass 1: load notes and serve cache hits immediately
    records_cache:   dict[str, list[dict] | None] = {}
    embedding_cache: dict[str, tuple] = {}   # patient_id -> (chunks, doc_embeddings)
    uncached:        dict[str, list[dict]] = {}

    for _, row in df.iterrows():
        patient_id = str(row["question_id"]).split("_")[0]
        if patient_id in records_cache:
            continue
        records = load_patient_records(args.notes, patient_id)
        records_cache[patient_id] = records
        if records is None:
            continue
        cached = load_cached_late_embeddings(patient_id, args.embeddings_dir)
        if cached is not None:
            embedding_cache[patient_id] = cached
            logger.info(f"Loaded cached LateOn embeddings for patient {patient_id}")
        else:
            uncached[patient_id] = records

    # Pass 2: encode all uncached patients in one batched model.encode() call
    if uncached:
        logger.info(f"Building LateOn embeddings for {len(uncached)} patients …")
        embedding_cache.update(
            build_late_embeddings_batch(uncached, args.embeddings_dir)
        )

    tasks = []
    for _, row in df.iterrows():
        patient_id = str(row["question_id"]).split("_")[0]
        if records_cache.get(patient_id) is None:
            logger.warning(f"Skipping question_id={row['question_id']} — notes unavailable.")
            continue
        chunks, index = embedding_cache[patient_id]
        for model in args.models:
            if (str(row["question_id"]), model.removesuffix("_batch")) in completed:
                logger.debug(f"Already done: question_id={row['question_id']} model={model} — skipping")
                continue
            tasks.append((row["question_id"], row[args.question_column], str(row["timestamp"]), chunks, index, model))

    if not tasks:
        logger.info("All tasks already complete.")
        return

    writer = CsvWriter(args.output, _LATE_FIELDS)

    # Encode all queries in one batched call before the async inference loop
    logger.info(f"Encoding {len(tasks)} queries …")
    query_embeddings = encode_queries_late([q for _, q, _, _, _, _ in tasks])

    # Split: Vertex batch models → batch API (50% discount); everything else → staggered concurrent
    batch_indices = [i for i, (_, _, _, _, _, m) in enumerate(tasks)
                     if m in VERTEX_BATCH_MODELS and args.backend == "vertex"]
    other_indices = [i for i in range(len(tasks)) if i not in set(batch_indices)]

    logger.info(
        f"{len(tasks)} tasks total: {len(batch_indices)} via Vertex batch API, "
        f"{len(other_indices)} via concurrent calls"
    )

    # ── Vertex batch: pre-retrieve then batch ─────────────────────────────────
    if batch_indices:
        batch_requests = []
        task_meta: dict[str, tuple] = {}
        retrieved_lookup: dict[str, list[dict]] = {}
        for i in batch_indices:
            qid, question, query_date, chunks, index, model = tasks[i]
            cutoff = parse_timestamp(query_date)
            if cutoff is None:
                raise ValueError(f"invalid query timestamp: {query_date!r}")
            valid = [position for position, chunk in enumerate(chunks) if (stamp := parse_timestamp(chunk.get("note_date"))) is not None and stamp <= cutoff]
            retrieved = retrieve_late_precomputed(query_embeddings[i], [chunks[position] for position in valid], [index[position] for position in valid], top_k=args.top_k)
            custom_id = f"{qid}||{model}"
            task_meta[custom_id] = (qid, model)
            retrieved_lookup[custom_id] = retrieved
            batch_requests.append({
                "custom_id": custom_id,
                "user_prompt": PROMPT_TEMPLATE.format(QUESTION=question, QUERY_DATE=query_date, NOTES=format_retrieved_notes(retrieved)),
                "model_id": model,
            })

        logger.info(f"Submitting {len(batch_requests)} tasks to Vertex batch API...")
        batch_results = await send_batch_messages(batch_requests)

        for custom_id, response in batch_results.items():
            qid, model = task_meta[custom_id]
            if response is None:
                logger.error(f"Batch failed: question_id={qid}, model={model}")
            await writer.write({
                "question_id":      qid,
                "model":            model,
                "approach":         "rag_late",
                "response":         response,
                "retrieved_chunks": serialise_retrieved(retrieved_lookup[custom_id]),
            })

    # ── Other models: staggered concurrent calls ──────────────────────────────
    if other_indices:
        delay = args.delay
        rpm_str = f"~{60 // delay} RPM" if delay > 0 else "unlimited (vLLM)"
        logger.info(f"{len(other_indices)} tasks at {delay}s intervals ({rpm_str})")

        async def delayed(i, question_id, question, query_date, query_emb, chunks, index, model):
            task_delay = 0 if model in VLLM_MODELS else delay
            await asyncio.sleep(i * task_delay)
            result = await run_inference(
                question_id, question, query_date, query_emb, chunks, index,
                model, args.top_k, args.backend,
            )
            await writer.write(result)
            return result

        results = await asyncio.gather(*[
            delayed(i, tasks[i][0], tasks[i][1], tasks[i][2], query_embeddings[i], tasks[i][3], tasks[i][4], tasks[i][5])
            for i in other_indices
        ])

        failed = [r for r in results if r.get("response") is None]
        if failed:
            logger.warning(f"{len(failed)} task(s) failed — see errors above.")

    logger.info(f"Done. Wrote {len(tasks)} rows to {args.output}")
    log_token_stats(logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG inference with LateOn ColBERT late-interaction retrieval"
    )
    parser.add_argument("-q", "--questions", type=str, required=True,
                        help="CSV file containing questions")
    parser.add_argument("-n", "--notes", type=str, required=True,
                        help="Directory containing patient notes JSON files")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to output CSV")
    parser.add_argument("-e", "--embeddings-dir", type=str, required=True,
                        help="Directory for per-patient LateOn embedding cache (pickle files)")
    parser.add_argument("-c", "--question-column", type=str, default="natural_query",
                        help="Column name for the question text (default: natural_query)")
    parser.add_argument("-k", "--top-k", type=int, default=50,
                        help="Number of chunks to retrieve (default: 50)")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
                        metavar="MODEL",
                        help=f"Models to run (default: all). Choices: {', '.join(ALL_MODELS)}")
    parser.add_argument("--backend", choices=["vertex", "openai"], default="vertex",
                        help="Backend for multi-backend models (default: vertex). Secure GPT and deepseek are unaffected.")
    parser.add_argument("--delay", type=int, default=DEFAULT_DELAY,
                        help=f"Seconds between API requests (default: {DEFAULT_DELAY}). Use 0 for vLLM-only runs.")
    parser.add_argument("--tpm-limit", type=int, default=None,
                        help="Enable a sliding-window tokens-per-minute rate limiter.")
    parser.add_argument("--force-ids", nargs="+", default=None, metavar="QUESTION_ID",
                        help="Re-run only these question_ids, dropping their stale output rows first.")
    parser.add_argument("--force-ids-file", type=str, default=None,
                        help="File with one question_id per line to re-run (unioned with --force-ids).")
    args = parser.parse_args()
    asyncio.run(main(args))
