'''
Two-phase clinical rubric evaluation.

Phase 1 — Rubric generation (Gemini Flash batch):
  For each question_id, generates a rubric from the gold atomic facts and question text.
  Each rubric item is tied to one or more gold facts and represents a clinically important
  point whose omission would change clinical practice or increase patient harm risk.
  Rubrics are saved to a CSV checkpoint and can be reused or hand-edited before scoring.

Phase 2 — Rubric scoring via 3-model jury:
  Scores each candidate prediction (raw answer text) against the rubric.
  Jurors: gemini_flash (Vertex GCS batch), claude_haiku (Vertex Claude batch),
          gpt5_nano (concurrent OpenAI-compatible calls).
  An item is marked "met" in the consensus if ≥2/3 jurors agree.
  Reports per-juror and consensus rubric scores (recall only — items only add points).

Inputs (via YAML config -c/--config):
  gcs_location    GCS prefix for all batch I/O
  questions       Questions CSV (question_id + question text column)
  question_column Column name for question text (default: question)
  facts           Atomic facts CSV — output of atomize_facts_batch.py
  rubrics         Path to save/load rubrics CSV (Phase 1 checkpoint)
  output          Path to save scoring results (Phase 2 checkpoint)
  gold_source     source_name for gold facts (default: reference)
  sources         List of {source_name, path} prediction CSVs

Resumes from existing output — completed (question_id, source_name, model) triples
in the scoring output and completed question_ids in the rubrics CSV are skipped.
'''

import argparse
import ast
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

import fsspec
import pandas as pd
from google import genai
from google.genai.types import CreateBatchJobConfig

from .utils import (  # type: ignore[reportAttributeAccessIssue]
    _TPMRateLimiter,
    _VERTEX_GEMINI_MODELS,
    count_tokens,
    CsvWriter,
    load_completed_pairs,
    log_token_stats,
    safe_json_parse,
    send_batch_messages,
    send_single_message,
    VERTEX_LOCATION,
    VERTEX_GEMINI_PROJECT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

GEMINI_MODEL_ID = _VERTEX_GEMINI_MODELS["gemini_flash_juror"]
JURORS = ["gemini", "claude", "gpt"]

SYSTEM_PROMPT = (
    "You are a clinician performing chart review. "
    "Respond only with a valid JSON object — no markdown, no explanation outside the JSON."
)

# ── Prompts ───────────────────────────────────────────────────────────────────

RUBRIC_GEN_PROMPT = """\
Given a clinical question and a list of atomic facts from the reference answer, \
generate a scoring rubric for evaluating candidate responses.

Each rubric item must:
1. Be tied to one or more of the provided facts.
2. Represent a clinically important point — if omitted from a candidate response, \
   it would change clinical practice or increase the risk of patient harm.
3. Be phrased as a self-contained criterion that can be evaluated without access to \
   the reference answer. The description must embed the specific expected value, date, \
   finding, or detail so a juror can verify it directly against the candidate response. \
   Do NOT use vague words like "correctly", "accurately", or "appropriately" — instead \
   state the exact expected content. For example, write \
   "Response states the dated hemoglobin value" not \
   "Response correctly identifies the hemoglobin value."
4. Correspond to exactly one independently scorable clinical claim — one that a juror \
   can mark met or not met without reference to any other rubric item. Apply this as follows: \
   SPLIT facts that express distinct verifiable claims a response could capture separately \
   (e.g. furosemide 40 mg daily and spironolactone 100 mg daily → two items, because a \
   response could mention one drug without the other). GROUP facts that express the same \
   clinical point from different angles (e.g. two facts both documenting the same diagnosis \
   → one item). The result should be a condensed, non-redundant set of checkpoints — \
   not a reformatted copy of the fact list.

### Question
{QUESTION}

### Reference facts
{FACTS}

### Output Format
{{
  "items": [
    {{
      "id": "r1",
      "description": "Response states that ...",
      "rationale": "Omitting this would ... because ...",
      "facts": ["supporting fact verbatim", ...]
    }}
  ]
}}
"""

SCORING_PROMPT = """\
You are a clinician evaluating whether a candidate response satisfies each criterion \
in a clinical rubric.

An item is "met" if the candidate response contains enough information to satisfy \
the criterion, even if phrased differently from the rubric wording. An item is "not met" \
if the information is absent, vague, or incorrect.

### Question
{QUESTION}

### Candidate response
{RESPONSE}

### Rubric items
{RUBRIC_ITEMS}

### Output Format
{{
  "scores": [
    {{"id": "r1", "met": true, "explanation": "one sentence"}},
    ...
  ]
}}
"""

# Per-juror (input $/1M, output $/1M)
# Gemini 3.1 Flash-Lite Vertex batch (Global)
# Claude Haiku 4.5 Vertex batch (50% discount applied)
# GPT-compatible online juror
_JUROR_COSTS = {
    "gemini": (0.125, 0.75),
    "claude": (0.40, 2.00),
    "gpt":    (0.15, 0.60),
}

RUBRIC_FIELDS = ["question_id", "rubric"]

OUTPUT_FIELDS = [
    "question_id", "source_name", "model",
    "gemini_items_met", "gemini_total_items", "gemini_rubric_score", "gemini_per_item_scores",
    "claude_items_met",  "claude_total_items",  "claude_rubric_score",  "claude_per_item_scores",
    "gpt_items_met",    "gpt_total_items",    "gpt_rubric_score",    "gpt_per_item_scores",
    "avg_rubric_score",
    "consensus_items_met", "consensus_total_items", "consensus_rubric_score", "consensus_per_item_scores",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_facts(raw) -> list[str]:
    if not raw or isinstance(raw, float):
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw.strip():
        for loader in (json.loads, ast.literal_eval):
            try:
                result = loader(raw)
                if isinstance(result, list):
                    return result
            except (ValueError, SyntaxError, json.JSONDecodeError):
                pass
    return []


def _parse_rubric(raw) -> dict | None:
    if not raw or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        s = raw if isinstance(raw, str) else json.dumps(raw)
        parsed = safe_json_parse(s)
        if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
            return parsed
    except Exception:
        pass
    return None


def _parse_scores(raw: str | None) -> list[dict] | None:
    if not raw:
        return None
    try:
        parsed = safe_json_parse(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("scores"), list):
            return parsed["scores"]
    except Exception:
        pass
    return None


def _rubric_items_text(items: list[dict]) -> str:
    return "\n".join(
        f"- id: {item['id']}\n  description: {item['description']}"
        for item in items
    )


def _score_juror(
    items: list[dict],
    scores: list[dict] | None,
) -> tuple[int, int, float, list[dict]]:
    """Return (items_met, total_items, rubric_score, per_item_scores)."""
    total = len(items)
    if not scores:
        return 0, total, 0.0, [{"id": item["id"], "met": False, "explanation": ""} for item in items]
    id_to_score = {s["id"]: s for s in scores if isinstance(s, dict) and "id" in s}
    per_item = []
    met = 0
    for item in items:
        s = id_to_score.get(item["id"], {})
        is_met = bool(s.get("met", False))
        if is_met:
            met += 1
        per_item.append({"id": item["id"], "met": is_met, "explanation": s.get("explanation", "")})
    score = met / total if total else 0.0
    return met, total, score, per_item


def _consensus_score(
    items: list[dict],
    all_per_item: dict[str, list[dict]],
    threshold: int = 2,
) -> tuple[int, int, float, list[dict]]:
    """≥threshold jurors must agree item is met."""
    total = len(items)
    met_count = 0
    per_item = []
    for item in items:
        iid = item["id"]
        votes = sum(
            1 for juror_items in all_per_item.values()
            if any(s.get("id") == iid and bool(s.get("met")) for s in juror_items)
        )
        is_met = votes >= threshold
        if is_met:
            met_count += 1
        per_item.append({"id": iid, "met": is_met, "votes": votes})
    score = met_count / total if total else 0.0
    return met_count, total, score, per_item


def log_token_estimate(prompts: list[dict]) -> None:
    input_tokens  = sum(count_tokens(SYSTEM_PROMPT + p["user_prompt"]) for p in prompts)
    # ~200 output tokens per response: JSON object with 5-10 rubric items,
    # each scored with a one-sentence explanation (~30 tokens/item)
    output_tokens = len(prompts) * 200
    per_juror = {
        name: input_tokens / 1e6 * c_in + output_tokens / 1e6 * c_out
        for name, (c_in, c_out) in _JUROR_COSTS.items()
    }
    total_cost = sum(per_juror.values())
    breakdown  = ", ".join(f"{name} ${cost:.2f}" for name, cost in per_juror.items())
    log.info(
        f"Token estimate per juror: {input_tokens:,} input + {output_tokens:,} output "
        f"({len(prompts):,} prompts)\n"
        f"  Estimated cost — {breakdown} — TOTAL: ${total_cost:.2f}"
    )


# ── Phase 1: Rubric generation (Gemini Flash batch, sync + async wrapper) ─────

def _run_rubric_gen_sync(
    tasks: list[tuple[str, str, list[str]]],
    gcs_location: str,
    poll_interval: int,
) -> dict[str, dict]:
    """Upload rubric-generation prompts as a Gemini batch job and return {qid: rubric}."""
    qid_list = [qid for qid, _, _ in tasks]
    jsonl_lines = [
        json.dumps({
            "key": str(i),
            "request": {
                "contents": [{"role": "user", "parts": [{"text": RUBRIC_GEN_PROMPT.format(
                    QUESTION=question,
                    FACTS="\n".join(f"- {f}" for f in facts),
                )}]}],
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            },
        })
        for i, (qid, question, facts) in enumerate(tasks)
    ]

    run_id = uuid.uuid4().hex[:12]
    gcs_input_uri     = f"{gcs_location}/rubric_gen_input_{run_id}.jsonl"
    gcs_output_prefix = f"{gcs_location}/rubric_gen_output/{run_id}"

    log.info(f"[rubric] Uploading {len(jsonl_lines)} prompts to {gcs_input_uri} ...")
    with fsspec.open(gcs_input_uri, "w") as f:
        f.write("\n".join(jsonl_lines))  # type: ignore[union-attr]

    client    = genai.Client(vertexai=True, project=VERTEX_GEMINI_PROJECT, location=VERTEX_LOCATION)
    batch_job = client.batches.create(
        model=GEMINI_MODEL_ID,
        src=gcs_input_uri,
        config=CreateBatchJobConfig(dest=gcs_output_prefix),
    )
    job_name = batch_job.name
    if not job_name:
        log.error("[rubric] Batch job returned no name")
        return {}
    log.info(f"[rubric] Batch job: {job_name}  state={batch_job.state}")

    while batch_job.state in ("JOB_STATE_RUNNING", "JOB_STATE_PENDING", "JOB_STATE_QUEUED"):
        time.sleep(poll_interval)
        batch_job = client.batches.get(name=job_name)
        log.info(f"[rubric] state={batch_job.state}")

    if batch_job.state != "JOB_STATE_SUCCEEDED":
        log.error(f"[rubric] Batch job failed: {getattr(batch_job, 'error', batch_job.state)}")
        return {}

    if batch_job.dest is None:
        log.error("[rubric] Batch job has no dest")
        return {}
    dest_uri     = batch_job.dest.gcs_uri
    fs           = fsspec.filesystem("gcs")
    result_files = fs.glob(f"{dest_uri}/*/predictions.jsonl")
    if not result_files:
        log.error(f"[rubric] No predictions.jsonl found under {dest_uri}")
        return {}

    all_lines: list[dict] = []
    for fp in result_files:
        shard = pd.read_json(f"gs://{fp}", lines=True)  # type: ignore[call-overload]
        all_lines.extend(shard.to_dict("records"))

    prompt_tok = sum(
        line.get("response", {}).get("usageMetadata", {}).get("promptTokenCount", 0)
        for line in all_lines if line.get("response")
    )
    output_tok = sum(
        line.get("response", {}).get("usageMetadata", {}).get("candidatesTokenCount", 0)
        for line in all_lines if line.get("response")
    )
    log.info(
        f"[rubric] {len(all_lines)} results downloaded — "
        f"tokens: {prompt_tok:,} prompt + {output_tok:,} completion"
    )

    key_to_line: dict[int, dict] = {}
    for line in all_lines:
        raw_key = line.get("key")
        if raw_key is None:
            continue
        try:
            key_to_line[int(raw_key)] = line
        except (ValueError, TypeError):
            pass

    rubrics: dict[str, dict] = {}
    for i, qid in enumerate(qid_list):
        line = key_to_line.get(i)
        if line is None:
            log.warning(f"[rubric] {qid}: no result found for key {i}")
            continue
        try:
            if line.get("status") and line["status"] != "":
                log.warning(f"[rubric] {qid}: status={line['status']!r}")
                continue
            text = line["response"]["candidates"][0]["content"]["parts"][0]["text"]
            parsed = safe_json_parse(text)
            if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
                rubrics[qid] = parsed
            else:
                log.warning(f"[rubric] {qid}: unexpected format")
        except Exception as e:
            log.warning(f"[rubric] {qid}: parse error: {e}")
    return rubrics


async def generate_rubrics(
    tasks: list[tuple[str, str, list[str]]],
    gcs_location: str,
    poll_interval: int,
) -> dict[str, dict]:
    if not tasks:
        return {}
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return await loop.run_in_executor(
            executor, _run_rubric_gen_sync, tasks, gcs_location, poll_interval
        )


# ── Phase 2: Gemini juror (sync + async wrapper) ──────────────────────────────

def _run_gemini_scoring_sync(
    prompts: list[dict],
    gcs_location: str,
    poll_interval: int,
) -> dict[int, list[dict] | None]:
    jsonl_lines = [
        json.dumps({
            "key": str(p["idx"]),
            "request": {
                "contents": [{"role": "user", "parts": [{"text": p["user_prompt"]}]}],
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            },
        })
        for p in prompts
    ]

    run_id = uuid.uuid4().hex[:12]
    gcs_input_uri     = f"{gcs_location}/scoring_input_{run_id}.jsonl"
    gcs_output_prefix = f"{gcs_location}/scoring_output/{run_id}"

    log.info(f"[gemini] Uploading {len(jsonl_lines):,} prompts to {gcs_input_uri} ...")
    with fsspec.open(gcs_input_uri, "w") as f:
        f.write("\n".join(jsonl_lines))  # type: ignore[union-attr]

    client    = genai.Client(vertexai=True, project=VERTEX_GEMINI_PROJECT, location=VERTEX_LOCATION)
    batch_job = client.batches.create(
        model=GEMINI_MODEL_ID,
        src=gcs_input_uri,
        config=CreateBatchJobConfig(dest=gcs_output_prefix),
    )
    job_name = batch_job.name
    if not job_name:
        log.error("[gemini] Batch job returned no name")
        return {}
    log.info(f"[gemini] Batch job: {job_name}  state={batch_job.state}")

    while batch_job.state in ("JOB_STATE_RUNNING", "JOB_STATE_PENDING", "JOB_STATE_QUEUED"):
        time.sleep(poll_interval)
        batch_job = client.batches.get(name=job_name)
        log.info(f"[gemini] state={batch_job.state}")

    if batch_job.state != "JOB_STATE_SUCCEEDED":
        log.error(f"[gemini] Batch job failed: {getattr(batch_job, 'error', batch_job.state)}")
        return {}

    if batch_job.dest is None:
        log.error("[gemini] Batch job has no dest")
        return {}
    dest_uri     = batch_job.dest.gcs_uri
    fs           = fsspec.filesystem("gcs")
    result_files = fs.glob(f"{dest_uri}/*/predictions.jsonl")
    if not result_files:
        log.error(f"[gemini] No predictions.jsonl found under {dest_uri}")
        return {}

    all_lines: list[dict] = []
    for fp in result_files:
        shard = pd.read_json(f"gs://{fp}", lines=True)  # type: ignore[call-overload]
        all_lines.extend(shard.to_dict("records"))

    prompt_tok = sum(
        line.get("response", {}).get("usageMetadata", {}).get("promptTokenCount", 0)
        for line in all_lines if line.get("response")
    )
    output_tok = sum(
        line.get("response", {}).get("usageMetadata", {}).get("candidatesTokenCount", 0)
        for line in all_lines if line.get("response")
    )
    log.info(
        f"[gemini] {len(all_lines)} results downloaded — "
        f"tokens: {prompt_tok:,} prompt + {output_tok:,} completion"
    )

    results: dict[int, list[dict] | None] = {}
    for line in all_lines:
        raw_key = line.get("key")
        if raw_key is None:
            continue
        try:
            idx = int(raw_key)
        except (ValueError, TypeError):
            continue
        try:
            if line.get("status") and line["status"] != "":
                results[idx] = None
                continue
            text = line["response"]["candidates"][0]["content"]["parts"][0]["text"]
            results[idx] = _parse_scores(text)
        except Exception:
            results[idx] = None
    return results


async def run_gemini_scoring(
    prompts: list[dict],
    gcs_location: str,
    poll_interval: int,
) -> dict[int, list[dict] | None]:
    if not prompts:
        return {}
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return await loop.run_in_executor(
            executor, _run_gemini_scoring_sync, prompts, gcs_location, poll_interval
        )


# ── Phase 2: Claude juror (Vertex Claude batch) ───────────────────────────────

async def run_claude_scoring(
    prompts: list[dict],
    poll_interval: int,
) -> dict[int, list[dict] | None]:
    requests = [
        {
            "custom_id":          str(p["idx"]),
            "user_prompt":        p["user_prompt"],
            "model_id":           "claude_haiku_batch",
            "system_instructions": SYSTEM_PROMPT,
        }
        for p in prompts
    ]
    log.info(f"[claude] Submitting {len(requests):,} requests via Vertex batch ...")
    batch_results = await send_batch_messages(requests, poll_interval=poll_interval)
    results: dict[int, list[dict] | None] = {}
    for p in prompts:
        text = batch_results.get(str(p["idx"]))
        results[p["idx"]] = _parse_scores(text) if text is not None else None
    succeeded = sum(1 for v in results.values() if v is not None)
    log.info(f"[claude] {succeeded}/{len(prompts)} succeeded")
    return results


# ── Phase 2: GPT juror (concurrent online calls) ──────────────────────────────

async def run_gpt_scoring(
    prompts: list[dict],
    tpm: int,
    rate: int,
) -> dict[int, list[dict] | None]:
    tpm_limiter = _TPMRateLimiter(tpm)
    semaphore   = asyncio.Semaphore(rate)

    async def _call(p: dict) -> list[dict] | None:
        estimated = count_tokens(SYSTEM_PROMPT + p["user_prompt"])
        await tpm_limiter.wait(estimated)
        async with semaphore:
            try:
                text = await send_single_message(
                    user_prompt=p["user_prompt"],
                    system_instructions=SYSTEM_PROMPT,
                    model_id="gpt5_nano_sandbox",
                    backend="openai",
                )
                return _parse_scores(text)
            except Exception as e:
                log.warning(f"[gpt] idx={p['idx']} failed: {e}")
                return None

    log.info(f"[gpt] Sending {len(prompts):,} requests via online backend (tpm={tpm:,}, rate={rate}) ...")
    responses = await asyncio.gather(*[_call(p) for p in prompts])
    results   = {p["idx"]: r for p, r in zip(prompts, responses)}
    succeeded = sum(1 for v in results.values() if v is not None)
    log.info(f"[gpt] {succeeded}/{len(prompts)} succeeded")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args):
    config          = yaml.safe_load(Path(args.config).read_text())
    gcs_location    = config["gcs_location"]
    questions_path  = config["questions"]
    question_column = config.get("question_column", "question")
    facts_path      = config["facts"]
    rubrics_path    = config["rubrics"]
    output_path     = config["output"]
    gold_source     = config.get("gold_source", "reference")
    sources         = config["sources"]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Load questions
    q_df = pd.read_csv(questions_path, dtype=str)
    if question_column not in q_df.columns:
        log.error(f"Column '{question_column}' not found in {questions_path}")
        sys.exit(1)
    question_lookup: dict[str, str] = {
        str(row["question_id"]): str(row[question_column])
        for _, row in q_df.iterrows()
    }
    log.info(f"Loaded {len(question_lookup)} questions from {questions_path}")

    # Load gold facts
    facts_df = pd.read_csv(facts_path, dtype=str)
    gold_df  = facts_df[facts_df["source_name"] == gold_source]
    gold_facts: dict[str, list[str]] = {}
    for _, row in gold_df.iterrows():
        facts = _parse_facts(row.get("facts_atomic"))
        if facts:
            gold_facts[str(row["question_id"])] = facts
    log.info(f"Loaded gold facts for {len(gold_facts)} questions")

    # ── Phase 1: Rubric generation ────────────────────────────────────────────
    completed_rubrics = load_completed_pairs(
        rubrics_path, ["question_id"], nonempty_col="rubric"
    )
    rubric_tasks: list[tuple[str, str, list[str]]] = []
    for qid, facts in gold_facts.items():
        if (qid,) in completed_rubrics:
            continue
        question = question_lookup.get(qid)
        if not question:
            log.warning(f"[rubric] No question text for {qid}, skipping")
            continue
        rubric_tasks.append((qid, question, facts))

    if rubric_tasks and not args.skip_rubric_gen:
        log.info(
            f"Phase 1: generating {len(rubric_tasks)} rubrics "
            f"({len(completed_rubrics)} already done)"
        )
        new_rubrics = await generate_rubrics(rubric_tasks, gcs_location, args.poll_interval)
        rubric_writer = CsvWriter(rubrics_path, RUBRIC_FIELDS)
        for qid, rubric_dict in new_rubrics.items():
            await rubric_writer.write({"question_id": qid, "rubric": json.dumps(rubric_dict)})
        log.info(f"Phase 1 complete: {len(new_rubrics)} rubrics written to {rubrics_path}")
    elif rubric_tasks and args.skip_rubric_gen:
        log.info(
            f"--skip-rubric-gen set; {len(rubric_tasks)} question(s) have no rubric and will be skipped in scoring"
        )
    else:
        log.info("Phase 1: all rubrics already generated")

    # Load all rubrics — keyed by question_id only; shared across all sources and models
    rubric_lookup: dict[str, dict] = {}
    if os.path.exists(rubrics_path) and os.path.getsize(rubrics_path) > 0:
        rdf = pd.read_csv(rubrics_path, dtype=str)
        for _, row in rdf.iterrows():
            r = _parse_rubric(row.get("rubric"))
            if r:
                rubric_lookup[str(row["question_id"])] = r
    log.info(f"Loaded {len(rubric_lookup)} rubrics for scoring (matched by question_id)")

    # ── Phase 2: Scoring ──────────────────────────────────────────────────────
    frames = []
    for source in sources:
        if source["source_name"] == gold_source:
            continue
        df = pd.read_csv(source["path"], dtype=str)
        if "source_name" not in df.columns:
            df["source_name"] = source["source_name"]
        frames.append(df)
    pred_df = pd.concat(frames, ignore_index=True)
    log.info(
        f"Loaded {len(pred_df)} prediction rows from {len(frames)} source(s): "
        + ", ".join(pred_df["source_name"].unique().tolist())
    )

    completed_scores = load_completed_pairs(
        output_path, ["question_id", "source_name", "model"],
        nonempty_col="consensus_rubric_score",
    )
    if completed_scores:
        log.info(f"Resuming: {len(completed_scores)} scoring rows already done")

    scoring_tasks: list[tuple] = []
    skipped_done = skipped_missing = 0
    for _, row in pred_df.iterrows():
        qid    = str(row["question_id"])
        source = str(row["source_name"])
        model  = str(row.get("model", ""))
        if (qid, source, model) in completed_scores:
            skipped_done += 1
            continue
        rubric = rubric_lookup.get(qid)
        if not rubric:
            skipped_missing += 1
            continue
        question = question_lookup.get(qid)
        if not question:
            skipped_missing += 1
            continue
        response = str(row.get("response", "")).strip()
        if not response:
            skipped_missing += 1
            continue
        scoring_tasks.append((qid, source, model, question, response, rubric))

    log.info(
        f"Phase 2: {len(scoring_tasks)} predictions to score "
        f"({skipped_done} already done, {skipped_missing} missing rubric/question/response)"
    )
    if not scoring_tasks:
        log.info("Nothing to score.")
        return

    scoring_prompts = []
    for idx, (qid, source, model, question, response, rubric) in enumerate(scoring_tasks):
        scoring_prompts.append({
            "idx": idx,
            "user_prompt": SCORING_PROMPT.format(
                QUESTION=question,
                RESPONSE=response,
                RUBRIC_ITEMS=_rubric_items_text(rubric["items"]),
            ),
        })

    log.info(f"Built {len(scoring_prompts):,} scoring prompts")
    log_token_estimate(scoring_prompts)

    if args.dry_run:
        log.info("[DRY RUN] Stopping before submission. No batch jobs submitted.")
        return

    log.info("Submitting all three jurors concurrently ...")
    gemini_raw, claude_raw, gpt_raw = await asyncio.gather(
        run_gemini_scoring(scoring_prompts, gcs_location, args.poll_interval),
        run_claude_scoring(scoring_prompts, args.poll_interval),
        run_gpt_scoring(scoring_prompts, args.gpt_tpm, args.rate),
    )

    writer = CsvWriter(output_path, OUTPUT_FIELDS)
    succeeded = 0
    for idx, (qid, source, model, _question, _response, rubric) in enumerate(scoring_tasks):
        items = rubric["items"]
        g_met, g_total, g_score, g_per = _score_juror(items, gemini_raw.get(idx))
        c_met, c_total, c_score, c_per = _score_juror(items, claude_raw.get(idx))
        p_met, p_total, p_score, p_per = _score_juror(items, gpt_raw.get(idx))

        avg_score = round((g_score + c_score + p_score) / 3, 4)

        cons_met, cons_total, cons_score, cons_per = _consensus_score(
            items, {"gemini": g_per, "claude": c_per, "gpt": p_per}
        )

        await writer.write({
            "question_id":               qid,
            "source_name":               source,
            "model":                     model,
            "gemini_items_met":          g_met,
            "gemini_total_items":        g_total,
            "gemini_rubric_score":       round(g_score, 4),
            "gemini_per_item_scores":    json.dumps(g_per),
            "claude_items_met":          c_met,
            "claude_total_items":        c_total,
            "claude_rubric_score":       round(c_score, 4),
            "claude_per_item_scores":    json.dumps(c_per),
            "gpt_items_met":             p_met,
            "gpt_total_items":           p_total,
            "gpt_rubric_score":          round(p_score, 4),
            "gpt_per_item_scores":       json.dumps(p_per),
            "avg_rubric_score":          avg_score,
            "consensus_items_met":       cons_met,
            "consensus_total_items":     cons_total,
            "consensus_rubric_score":    round(cons_score, 4),
            "consensus_per_item_scores": json.dumps(cons_per),
        })
        succeeded += 1

    log.info(f"Done. {succeeded}/{len(scoring_tasks)} rows written to {output_path}")
    log_token_stats(log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-phase clinical rubric evaluation: generate rubrics from facts, then score predictions"
    )
    parser.add_argument("-c", "--config",      required=True,
                        help="YAML config file (gcs_location, questions, facts, rubrics, output, sources)")
    parser.add_argument("--poll-interval",     type=int, default=60,
                        help="Seconds between batch job status checks (default: 60)")
    parser.add_argument("--gpt-tpm",           type=int, default=5_000_000,
                        help="Online juror token-per-minute limit")
    parser.add_argument("--rate",              type=int, default=20,
                        help="Max concurrent GPT nano requests (default: 20)")
    parser.add_argument("--skip-rubric-gen",   action="store_true",
                        help="Skip Phase 1 and use existing rubrics CSV as-is")
    parser.add_argument("--dry-run",           action="store_true",
                        help="Estimate token count and cost without submitting any batch jobs")
    args = parser.parse_args()
    asyncio.run(main(args))
