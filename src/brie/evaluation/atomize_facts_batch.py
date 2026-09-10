'''
Batch atomic fact extraction via Vertex AI Gemini batch prediction.

Reads a YAML config listing source CSVs, builds a single JSONL batch job,
uploads it to GCS, waits for the Vertex AI batch job to complete, then
downloads and parses results into one combined output CSV.

Already-completed (source_name, <id_column>) pairs are skipped (checkpointing).

YAML format:
    gcs_location: gs://bucket/prefix
    sources:
      - source_name: questions
        path: /path/to/questions.csv
        text_column: annotation_sub_answer
        facts_column: facts_edited        # optional
      - source_name: predictions_gemini
        path: /path/to/predictions.csv
        text_column: response

Output CSV columns: source_name, <id_column>, facts_atomic
'''

import argparse
import ast
import csv
import json
import logging
import os
import re
import sys
import uuid
import time
from pathlib import Path

import fsspec
import pandas as pd
import yaml
from google import genai
from google.genai.types import CreateBatchJobConfig

from .utils import (
    VERTEX_LOCATION,
    VERTEX_GEMINI_PROJECT,
    _VERTEX_GEMINI_MODELS,
    count_tokens,
    load_completed_pairs,
    load_force_ids,
    drop_rows_for_source_models,
    drop_rows_for_ids,
    safe_json_parse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

MODEL_ID = _VERTEX_GEMINI_MODELS["gemini_flash_juror"]

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a clinician performing chart review. "
    "Respond only with a valid JSON object — no markdown, no explanation outside the JSON."
)

ATOMIZE_PROMPT = """\
You will be given a list of clinical facts. Decompose any non-atomic facts into \
truly atomic claims.

## Atomic Claim Definition
An atomic claim makes exactly one assertion: a single subject, predicate, and \
object. It cannot be split into more fundamental independent claims.

## Rules
1. For each input fact, produce one or more output facts.
2. If a fact mentions a date, produce one atomic fact that states the date and \
   the high-level event (e.g., "A TSH measurement was performed on <DATE>."). \
   All other sub-facts MUST NOT include the date — even if the original fact \
   contained it.
3. Each output fact must assert only one thing — do not bundle multiple \
   measurements, percentages, or findings into a single claim.
4. Do not add information not present in the source fact.
5. Do not merge information from different source facts.
6. If a fact is already atomic and contains no date, output it unchanged. \
   A fact that bundles a date with a measurement or finding is NOT atomic — \
   split it into a date anchor and one or more date-free sub-facts.
7. Always refer to the subject as "patient".

## Example
Input: "The patient's TSH was elevated on <DATE>."
Output:
- "A TSH measurement was performed on <DATE>."
- "The patient's TSH was 42."

## Output Format
{{
    "facts": [
        {{"text": "Atomic claim one.", "source_idx": 0}},
        {{"text": "Atomic claim two.", "source_idx": 0}},
        ...
    ]
}}

source_idx is the 0-based index of the input fact this atomic claim was derived from.

## Input Facts
{FACTS}
"""

EXTRACT_PROMPT = """\
Extract every atomic clinical claim from the answer below.

## Atomic Claim Definition
An atomic claim makes a single assertion with a subject, predicate, and object. \
It must stand alone without ambiguity and cannot be decomposed into more \
fundamental claims.

## DO
1. Extract discrete atomic claims. Each must include a subject, predicate, and object.
2. Include only clinically relevant claims (symptoms, procedures, tests, \
   medications, diagnoses, clinical locations).
3. Use only the provided text. Do not add outside knowledge or assumptions.
4. Write each claim in the shortest unambiguous form. Avoid pronouns or vague references.
5. Always refer to the subject as "patient", even if the text uses a name or identifier.
6. When a date is mentioned in the text, create one atomic fact that captures the \
   date and the high-level event type (e.g., "A TSH measurement was performed on \
   <DATE>."). All other sub-facts from that same event MUST NOT include the date.
7. If there are no valid clinically relevant claims, return "facts" as an empty list [].

## DO NOT
1. Do not include claims unrelated to the patient's clinical care (e.g., provider \
   names, administrative details).
2. Do not invent or infer claims beyond what is explicitly stated.
3. Do not combine multiple events into a single claim.
4. Do not append dates to sub-facts — a measurement or finding is a separate \
   atomic fact from the date on which it occurred.

## Examples

Input: "5/1: EGD with stricturoplasty by endoscopic submucosal dissection, steroid \
injection, and placement of a 20 mm Axios stent (sutured/anchored) at the pylorus"
Output:
- "A procedure was performed on 5/1."
- "The procedure was an EGD (esophagogastroduodenoscopy)."
- "A stricturoplasty was performed during the procedure."
- "The stricturoplasty was performed by endoscopic submucosal dissection."
- "A steroid injection was administered during the procedure."
- "An Axios stent was placed during the procedure."
- "The Axios stent was 20 mm in size."
- "The Axios stent was placed at the pylorus."
- "The Axios stent was sutured/anchored in place."

Input: "TSH was elevated on <DATE_A> — treatment began on <DATE_B>."
Output:
- "A TSH measurement was performed on <DATE_A>."
- "The patient's TSH was 42."
- "Treatment was started on <DATE_B>."
- "The patient was started on Levothyroxine 50 mcg daily."

## Output Format
{{
    "facts": ["Claim one.", "Claim two.", ...]
}}

### Answer
{ANSWER}
"""

OUTPUT_FIELDS = ["source_name", "question_id", "model", "facts_atomic", "fact_source_indices"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _parse_facts(raw) -> list[str] | None:
    if raw is None or isinstance(raw, float):
        return None
    if isinstance(raw, list):
        return raw or None
    if isinstance(raw, str) and raw.strip():
        for loader in (ast.literal_eval, json.loads):
            try:
                result = loader(raw)
                if isinstance(result, list):
                    return result or None
            except (ValueError, SyntaxError, json.JSONDecodeError):
                pass
    return None


def _build_prompt(row: pd.Series, text_col: str, facts_col: str | None) -> str | None:
    facts = _parse_facts(row.get(facts_col)) if facts_col else None
    if facts:
        return ATOMIZE_PROMPT.format(FACTS="\n".join(f"{i}. {f}" for i, f in enumerate(facts)))
    text = str(row.get(text_col) or "").strip()
    return EXTRACT_PROMPT.format(ANSWER=text) if text else None


def _build_request_line(prompt: str, idx: int) -> str:
    return json.dumps({
        "key": str(idx),
        "request": {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        }
    })


def _parse_result_line(line: dict) -> tuple[list[str] | None, list[int] | None]:
    """Return (facts, source_indices). source_indices is None for EXTRACT_PROMPT results."""
    if line.get("status") and line["status"] != "":
        return None, None
    try:
        text  = line["response"]["candidates"][0]["content"]["parts"][0]["text"]
        items = safe_json_parse(text)["facts"]
        if not items:
            return [], []
        # ATOMIZE_PROMPT: list of {"text": ..., "source_idx": ...}
        if isinstance(items[0], dict):
            # Provenance is required for every atomic claim. Do not write a
            # superficially successful row when the model omits one mapping:
            # downstream extraction cannot construct stable hierarchical IDs.
            if any("text" not in item or "source_idx" not in item for item in items):
                return None, None
            facts = [str(item["text"]) for item in items]
            indices = [int(item["source_idx"]) for item in items]
            return facts, indices
        # EXTRACT_PROMPT: flat list of strings — no source provenance
        return [str(f) for f in items], None
    except Exception:
        return None, None


def _csv_writer(path: str, fieldnames: list[str]):
    """Return an open csv.DictWriter, writing header only if the file is new/empty."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
    if write_header:
        writer.writeheader()
    return f, writer


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    config = yaml.safe_load(Path(args.config).read_text())
    gcs_location = config["gcs_location"].rstrip("/")
    sources      = config["sources"]
    id_col       = args.id_column
    output_fields = ["source_name", id_col, "model", "facts_atomic", "fact_source_indices"]

    force_ids = load_force_ids(args.force_ids, args.force_ids_file)
    force_models = {re.sub(r"_(batch|sandbox)$", "", model) for model in args.force_models}
    if force_ids:
        removed = drop_rows_for_ids(args.output, id_col, force_ids)
        log.info(f"--force-ids: dropped {removed} stale row(s) for {len(force_ids)} question_id(s) from {args.output}")

    if args.force_models:
        removed = drop_rows_for_source_models(args.output, args.force_source, args.force_models)
        log.info(
            f"--force-source/--force-models: dropped {removed} stale row(s) "
            f"for {args.force_source} x {args.force_models} from {args.output}"
        )
    elif args.force_source:
        removed = drop_rows_for_ids(args.output, "source_name", set(args.force_source))
        log.info(f"--force-source: dropped {removed} stale row(s) for {args.force_source} from {args.output}")

    # Checkpointing key: (source_name, question_id, model) — model is "" for gold
    completed = load_completed_pairs(
        args.output, ["source_name", id_col, "model"], nonempty_col="facts_atomic"
    )
    if completed:
        log.info(f"Resuming: {len(completed)} rows already done")

    # Build JSONL and positional ID list: (source_name, question_id, model_or_empty)
    jsonl_lines: list[str]               = []
    id_list:     list[tuple[str, str, str]] = []

    for source in sources:
        name       = source["source_name"]
        if args.force_source and name not in args.force_source:
            continue
        text_col   = source["text_column"]
        facts_col  = source.get("facts_column")
        model_col  = source.get("model_column")
        df         = pd.read_csv(source["path"], dtype=str)

        if id_col not in df.columns:
            log.error(f"id column '{id_col}' not found in {source['path']}")
            return

        skipped = included = 0
        for _, row in df.iterrows():
            qid   = str(row[id_col])
            if force_ids and qid not in force_ids:
                skipped += 1
                continue
            model = re.sub(r"_(batch|sandbox)$", "", str(row[model_col])) if model_col and model_col in df.columns else ""
            if force_models and model not in force_models:
                skipped += 1
                continue
            if (name, qid, model) in completed:
                skipped += 1
                continue
            prompt = _build_prompt(row, text_col, facts_col)
            if prompt is None:
                log.warning(f"{name}/{qid}: empty text, skipping")
                skipped += 1
                continue
            jsonl_lines.append(_build_request_line(prompt, len(jsonl_lines)))
            id_list.append((name, qid, model))
            included += 1

        log.info(f"  {name}: {included} to submit, {skipped} skipped")

    if not jsonl_lines:
        log.info("All rows already processed.")
        return

    total_requests = len(jsonl_lines)
    log.info(f"Total: {total_requests} requests across {len(sources)} sources")

    if args.dry_run:
        prompts = [json.loads(line)["request"]["contents"][0]["parts"][0]["text"] for line in jsonl_lines]
        prompt_tokens = sum(count_tokens(p) for p in prompts)
        # gemini-3.1-flash-lite batch pricing (Global): $0.125/1M input, $0.75/1M output (est. ~100 tokens out)
        est_output_tokens = total_requests * 100
        est_cost = (prompt_tokens / 1_000_000) * 0.125 + (est_output_tokens / 1_000_000) * 0.75
        log.info("── Dry-run estimate ──────────────────────────────")
        log.info(f"  Requests:       {total_requests:>10,}")
        log.info(f"  Prompt tokens:  {prompt_tokens:>10,}")
        log.info(f"  Est. cost:      ${est_cost:>9.2f}  (flash-lite batch rates)")
        if total_requests > 200_000:
            log.warning(f"  *** {total_requests:,} requests exceeds the 200K-per-job limit — split into multiple runs ***")
        else:
            log.info(f"  Job limit:      OK ({total_requests:,} / 200,000)")
        return

    if args.gcs_output_uri:
        # Recovery mode: results already in GCS from a prior completed job
        log.info(f"Recovery mode — reading results from {args.gcs_output_uri}")
        dest_uri = args.gcs_output_uri.rstrip("/")
    else:
        # Upload input JSONL to GCS
        run_id = uuid.uuid4().hex[:12]
        gcs_input_uri = f"{gcs_location}/input_{run_id}.jsonl"
        gcs_output_prefix = f"{gcs_location}/output/{run_id}"
        log.info(f"Uploading to {gcs_input_uri} ...")
        with fsspec.open(gcs_input_uri, "w") as f:
            f.write("\n".join(jsonl_lines))  # type: ignore[union-attr]

        # Submit batch prediction job
        client = genai.Client(vertexai=True, project=VERTEX_GEMINI_PROJECT, location=VERTEX_LOCATION)

        log.info(f"Submitting batch job (model={MODEL_ID}) ...")
        batch_job = client.batches.create(
            model=MODEL_ID,
            src=gcs_input_uri,
            config=CreateBatchJobConfig(dest=gcs_output_prefix),
        )
        job_name = batch_job.name
        if not job_name:
            log.error("Batch job returned no name")
            sys.exit(1)
        log.info(f"Batch job: {job_name}  state={batch_job.state}")

        # Poll until terminal
        while batch_job.state in ("JOB_STATE_RUNNING", "JOB_STATE_PENDING", "JOB_STATE_QUEUED"):
            time.sleep(args.poll_interval)
            batch_job = client.batches.get(name=job_name)
            log.info(f"  state={batch_job.state}")

        if batch_job.state != "JOB_STATE_SUCCEEDED":
            log.error(f"Batch job failed: {getattr(batch_job, 'error', batch_job.state)}")
            sys.exit(1)

        log.info("Batch job succeeded. Fetching results ...")

        if batch_job.dest is None:
            log.error("Batch job has no dest")
            sys.exit(1)
        dest_uri = batch_job.dest.gcs_uri
    fs = fsspec.filesystem("gcs")
    result_files = fs.glob(f"{dest_uri}/*/predictions.jsonl")

    if not result_files:
        log.error(f"No predictions.jsonl found under {dest_uri}")
        sys.exit(1)

    all_lines: list[dict] = []
    for fp in result_files:
        with fsspec.open(f"gs://{fp}", "r") as f:
            all_lines.extend(_read_jsonl_records(f))

    log.info(f"Downloaded {len(all_lines)} result lines")

    key_to_line: dict[int, dict] = {}
    for line in all_lines:
        raw_key = line.get("key")
        if raw_key is not None:
            try:
                key_to_line[int(raw_key)] = line
            except (ValueError, TypeError):
                pass

    if len(key_to_line) != len(id_list):
        log.warning(f"Count mismatch: {len(key_to_line)} keyed results for {len(id_list)} requests")

    # Write results to output CSV
    f_out, writer = _csv_writer(args.output, output_fields)
    succeeded = errors = 0
    try:
        for i, (source_name, qid, model) in enumerate(id_list):
            line = key_to_line.get(i)
            if line is None:
                errors += 1
                log.warning(f"{source_name}/{qid}/{model}: no result found for key {i}")
                continue
            facts, source_indices = _parse_result_line(line)
            writer.writerow({
                "source_name":        source_name,
                id_col:               qid,
                "model":              model,
                "facts_atomic":       json.dumps(facts) if facts is not None else None,
                "fact_source_indices": json.dumps(source_indices) if source_indices is not None else None,
            })
            if facts is not None:
                succeeded += 1
            else:
                errors += 1
                log.warning(f"{source_name}/{qid}/{model}: failed to parse result (status={line.get('status')!r})")
    finally:
        f_out.close()

    log.info(f"Done. {succeeded} succeeded, {errors} errors — wrote to {args.output}")

    # Token stats from usageMetadata
    prompt_tokens = sum(
        line.get("response", {}).get("usageMetadata", {}).get("promptTokenCount", 0)
        for line in all_lines if line.get("response")
    )
    output_tokens = sum(
        line.get("response", {}).get("usageMetadata", {}).get("candidatesTokenCount", 0)
        for line in all_lines if line.get("response")
    )
    log.info(
        f"Token usage: {prompt_tokens:,} prompt + {output_tokens:,} completion "
        f"= {prompt_tokens + output_tokens:,} total ({len(all_lines)} calls)"
    )


def _read_jsonl_records(lines) -> list[dict]:
    """Parse JSONL without pandas dtype inference changing request keys."""
    records: list[dict] = []
    for raw_line in lines:
        if isinstance(raw_line, bytes):
            raw_line = raw_line.decode("utf-8")
        if raw_line.strip():
            records.append(json.loads(raw_line))
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch atomic fact extraction via Vertex AI Gemini batch prediction"
    )
    parser.add_argument("-c", "--config",        required=True,
                        help="YAML config file (gcs_location + sources list)")
    parser.add_argument("-o", "--output",        required=True,
                        help="Output CSV (source_name, <id-column>, facts_atomic)")
    parser.add_argument("--id-column",           default="question_id",
                        help="Row key column in source CSVs (default: question_id)")
    parser.add_argument("--poll-interval",       type=int, default=60,
                        help="Seconds between batch job status checks (default: 60)")
    parser.add_argument("--dry-run",             action="store_true",
                        help="Estimate request count and cost without submitting a job")
    parser.add_argument("--gcs-output-uri",      default=None,
                        help="Skip job submission and pull results from this GCS URI (recovery mode)")
    parser.add_argument("--force-source",         nargs="+", default=[],
                        help="Re-process these source_name(s), dropping their stale output rows first "
                             "(e.g. gemini_recent gemini_recent200 gemini_bm25 gemini_dense).")
    parser.add_argument("--force-models",         nargs="+", default=[], metavar="MODEL",
                        help="With --force-source, re-process only these models in the selected sources.")
    parser.add_argument("--force-ids",            nargs="+", default=None, metavar="QUESTION_ID",
                        help="Re-process only these question_ids (across all sources), dropping their stale output rows first.")
    parser.add_argument("--force-ids-file",       type=str, default=None,
                        help="File with one question_id per line to re-process (unioned with --force-ids).")
    args = parser.parse_args()
    if args.force_models and not args.force_source:
        parser.error("--force-models requires --force-source")
    main(args)
