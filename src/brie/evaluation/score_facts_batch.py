'''
Fact-based precision/recall evaluation via 3-model entailment jury.

For each (question_id, source_name, model) comparison vs. gold facts:
  - Runs fact entailment in both directions (recall + precision)
  - Jurors: gemini_flash (Vertex GCS batch), claude_haiku (Vertex Claude batch),
            gpt5_nano (concurrent OpenAI-compatible calls)
  - Reports per-juror scores, macro average, and ≥2/3 consensus scores

Logs a token/cost estimate before submitting, and actual usage after.
Resumes from an existing output CSV — completed (question_id, source_name, model)
triples are skipped.

Input:  atomic facts CSV (source_name, question_id, model, facts_atomic)
Output: one row per (question_id, source_name, model)
'''

import argparse
import ast
import asyncio
import json
import logging
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import fsspec
import pandas as pd
from google import genai
from google.genai.types import CreateBatchJobConfig

from .utils import (  # type: ignore[reportAttributeAccessIssue]
    VERTEX_LOCATION,
    VERTEX_GEMINI_PROJECT,
    _TPMRateLimiter,
    _VERTEX_GEMINI_MODELS,
    _example_gemini_contents,
    count_tokens,
    CsvWriter,
    load_completed_pairs,
    load_force_ids,
    drop_rows_for_source_models,
    drop_rows_for_ids,
    log_token_stats,
    safe_json_parse,
    send_batch_messages,
    send_single_message,
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
    "Respond only with a valid JSON array — no markdown, no explanation."
)

RECALL_PROMPT = """\
Given a list of REFERENCE facts and a list of CANDIDATE facts, identify which \
REFERENCE facts are semantically entailed by any of the CANDIDATE facts.
Respond with a JSON array of 0-based indices of the REFERENCE facts that are entailed. \
Return [] if none are entailed.

Judge as a clinician reviewing the chart would, not as a literal string matcher. The \
usual error is being too conservative — marking a REFERENCE fact absent because no \
CANDIDATE fact restates it in the same words, when the CANDIDATE facts plainly cover it.

## Rules
1. A REFERENCE fact is entailed if the CANDIDATE facts, taken together, assert it — it \
   need not be restated by a single CANDIDATE fact.
2. Both lists are atomized, so a dated event is split into a bare event anchor \
   ("A hemodynamic measurement was performed on <DATE>.") plus separate date-free \
   facts giving that event's details, findings, or results. Judge an anchor by its \
   details, not by its date: the anchor is entailed whenever the CANDIDATE facts assert \
   those details, even if no CANDIDATE fact mentions the date or names the event type.
3. Two events belong to the same episode of care when their dates match, fall within \
   about two weeks of each other, or one is given only as a month or an approximate \
   date. Do not require an exact date match: when one date gives only a month and year, \
   and they match the other date's month and year, treat them as the same event. Treat \
   events as separate only when the dates clearly indicate different encounters.
4. Within one episode of care, a REFERENCE fact describing a component, step, or \
   routine part of a larger event is entailed by a CANDIDATE fact describing that \
   larger event: an encounter entails the medications, fluids, and assessments given \
   during it, and a procedure entails the measurements and specimens it ordinarily \
   involves.
5. Wording and granularity need not match. A fact may be more specific in one respect \
   (naming the drug, device, or site) and less specific in another (a month rather than \
   a day); neither difference blocks entailment, in either direction.
6. Entail on semantic equivalence or logical implication, not only on restatement. If a \
   CANDIDATE fact means the same thing in different words, or logically implies the \
   REFERENCE fact, mark it entailed — a stated consequence of a symptom implies the \
   symptom, resuming or restarting a treatment implies that it was initiated, and a \
   documented trial of a treatment implies that it was given.
7. One supporting CANDIDATE fact is enough. Judge each REFERENCE fact on its own, and \
   do not withhold entailment because other CANDIDATE facts describe related events \
   pointing a different way: a separate or later event involving the same medication, \
   problem, or procedure does not cancel an earlier one.
8. Only two things block entailment: the CANDIDATE facts describe no related event or \
   encounter at all, or a CANDIDATE fact directly denies the REFERENCE fact itself. A \
   treatment tried without benefit still entails that it was given, but does not entail \
   that it helped.

### Reference facts
{REFERENCE_FACTS}

### Candidate facts
{CANDIDATE_FACTS}
"""

PRECISION_PROMPT = """\
Given a list of REFERENCE facts and a list of CANDIDATE facts, identify which \
CANDIDATE facts are semantically entailed by any of the REFERENCE facts.
Respond with a JSON array of 0-based indices of the CANDIDATE facts that are entailed. \
Return [] if none are entailed.

Judge as a clinician reviewing the chart would, not as a literal string matcher. The \
usual error is being too conservative — marking a CANDIDATE fact absent because no \
REFERENCE fact restates it in the same words, when the REFERENCE facts plainly cover it.

## Rules
1. A CANDIDATE fact is entailed if the REFERENCE facts, taken together, assert it — it \
   need not be restated by a single REFERENCE fact.
2. Both lists are atomized, so a dated event is split into a bare event anchor \
   ("An Emergency Department visit occurred on <DATE>.") plus separate \
   date-free facts giving that event's details, findings, or results. Judge an anchor by \
   its details, not by its date: the anchor is entailed whenever the REFERENCE facts \
   assert those details, even if no REFERENCE fact mentions the date or names the event \
   type.
3. Two events belong to the same episode of care when their dates match, fall within \
   about two weeks of each other, or one is given only as a month or an approximate \
   date. Do not require an exact date match: when one date gives only a month and year, \
   and they match the other date's month and year, treat them as the same event. Treat \
   events as separate only when the dates clearly indicate different encounters.
4. Within one episode of care, a CANDIDATE fact describing a component, step, or \
   routine part of a larger event is entailed by a REFERENCE fact describing that \
   larger event: an encounter entails the medications, fluids, and assessments given \
   during it, and a procedure entails the measurements and specimens it ordinarily \
   involves.
5. Wording and granularity need not match. A fact may be more specific in one respect \
   (naming the drug, device, or site) and less specific in another (a month rather than \
   a day); neither difference blocks entailment, in either direction.
6. Entail on semantic equivalence or logical implication, not only on restatement. If a \
   REFERENCE fact means the same thing in different words, or logically implies the \
   CANDIDATE fact, mark it entailed — a stated consequence of a symptom implies the \
   symptom, resuming or restarting a treatment implies that it was initiated, and a \
   documented trial of a treatment implies that it was given.
7. One supporting REFERENCE fact is enough. Judge each CANDIDATE fact on its own, and \
   do not withhold entailment because other REFERENCE facts describe related events \
   pointing a different way: a separate or later event involving the same medication, \
   problem, or procedure does not cancel an earlier one.
8. Only two things block entailment: the REFERENCE facts describe no related event or \
   encounter at all, or a REFERENCE fact directly denies the CANDIDATE fact itself. A \
   treatment tried without benefit still entails that it was given, but does not entail \
   that it helped.

### Reference facts
{REFERENCE_FACTS}

### Candidate facts
{CANDIDATE_FACTS}
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

OUTPUT_FIELDS = [
    "question_id", "source_name", "model",
    "gemini_precision", "gemini_recall",
    "gemini_entailed_ref_facts", "gemini_entailed_cand_facts",
    "claude_precision", "claude_recall",
    "claude_entailed_ref_facts", "claude_entailed_cand_facts",
    "gpt_precision", "gpt_recall",
    "gpt_entailed_ref_facts", "gpt_entailed_cand_facts",
    "avg_precision", "avg_recall",
    "consensus_precision", "consensus_recall",
    "consensus_entailed_ref_facts", "consensus_entailed_cand_facts",
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


def _parse_entailment_response(raw: str) -> list[int] | None:
    try:
        result = safe_json_parse(raw)
        if isinstance(result, list):
            return [int(x) for x in result]
    except Exception:
        pass
    return None


# ── Few-shot examples ─────────────────────────────────────────────────────────
def load_examples(path: str | None) -> dict[str, list[dict]]:
    """Load few-shot examples (from build_fact_examples.py) grouped by direction.

    Returns {"r": [{role, content}, ...], "p": [...]}. Empty dict when path is None.
    """
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        turns = json.load(f)
    by_dir: dict[str, list[dict]] = {"r": [], "p": []}
    for t in turns:
        by_dir.setdefault(t.get("direction", "r"), []).append(
            {"role": t["role"], "content": t["content"]})
    log.info(f"Loaded few-shot examples: {len(by_dir.get('r', []))//2} recall + "
             f"{len(by_dir.get('p', []))//2} precision pairs from {path}")
    return by_dir


# ── Prompt building ───────────────────────────────────────────────────────────
def build_prompts(comparisons: list[tuple], examples_by_dir: dict[str, list[dict]] | None = None) -> list[dict]:
    """Two prompts per comparison: recall (which ref facts are covered by cand)
    and precision (which cand facts are covered by ref). Facts are matched
    by question_id — each comparison tuple already carries the paired lists.

    When examples_by_dir is given, each prompt carries direction-matching few-shot
    turns under "example_messages" so the recall/precision request is preceded by
    human-validated demonstrations of the same task."""
    examples_by_dir = examples_by_dir or {}
    prompts = []
    for qid, source, model, ref_facts, cand_facts in comparisons:
        key           = (qid, source, model)
        ref_numbered  = "\n".join(f"{i}. {f}" for i, f in enumerate(ref_facts))
        cand_numbered = "\n".join(f"{i}. {f}" for i, f in enumerate(cand_facts))
        prompts.append({
            "direction":      "r",
            "comparison_key": key,
            "user_prompt":    RECALL_PROMPT.format(
                REFERENCE_FACTS=ref_numbered, CANDIDATE_FACTS=cand_numbered),
            "example_messages": examples_by_dir.get("r", []),
        })
        prompts.append({
            "direction":      "p",
            "comparison_key": key,
            "user_prompt":    PRECISION_PROMPT.format(
                REFERENCE_FACTS=ref_numbered, CANDIDATE_FACTS=cand_numbered),
            "example_messages": examples_by_dir.get("p", []),
        })
    for i, p in enumerate(prompts):
        p["idx"] = i
    return prompts


# ── Token estimate ────────────────────────────────────────────────────────────
def _example_tokens(p: dict) -> int:
    return sum(count_tokens(m["content"]) for m in p.get("example_messages", []))


def log_token_estimate(prompts: list[dict], jurors: list[str] | None = None) -> None:
    jurors = jurors or JURORS
    input_tokens  = sum(count_tokens(SYSTEM_PROMPT + p["user_prompt"]) + _example_tokens(p) for p in prompts)
    output_tokens = len(prompts) * 5   # short JSON array per response
    total_cost = sum(
        input_tokens / 1e6 * c_in + output_tokens / 1e6 * c_out
        for name, (c_in, c_out) in _JUROR_COSTS.items() if name in jurors
    )
    log.info(
        f"Token estimate per juror: {input_tokens:,} input + {output_tokens:,} output "
        f"({len(prompts):,} calls) — {', '.join(jurors)} estimated cost: ${total_cost:.2f}"
    )


# ── Gemini batch (sync, run in thread executor) ───────────────────────────────
def _run_gemini_sync(
    prompts: list[dict],
    gcs_location: str,
    poll_interval: int,
) -> dict[int, list[int] | None]:
    jsonl_lines = [
        json.dumps({
            "key": str(p["idx"]),
            "request": {
                "contents": _example_gemini_contents(p.get("example_messages"))
                + [{"role": "user", "parts": [{"text": p["user_prompt"]}]}],
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            }
        })
        for p in prompts
    ]

    run_id = uuid.uuid4().hex[:12]
    gcs_input_uri    = f"{gcs_location}/entailment_input_{run_id}.jsonl"
    gcs_output_prefix = f"{gcs_location}/entailment_output/{run_id}"

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

    results: dict[int, list[int] | None] = {}
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
            results[idx] = _parse_entailment_response(text)
        except Exception:
            results[idx] = None
    return results


def _download_gemini_job(job_name: str) -> dict[int, list[int] | None]:
    """Re-download results from an already-completed Gemini batch job by job name."""
    client    = genai.Client(vertexai=True, project=VERTEX_GEMINI_PROJECT, location=VERTEX_LOCATION)
    batch_job = client.batches.get(name=job_name)
    if batch_job.state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"[gemini] Job {job_name} is not succeeded: {batch_job.state}")
    if batch_job.dest is None:
        raise RuntimeError(f"[gemini] Job {job_name} has no dest")
    dest_uri     = batch_job.dest.gcs_uri
    fs           = fsspec.filesystem("gcs")
    result_files = fs.glob(f"{dest_uri}/*/predictions.jsonl")
    if not result_files:
        raise RuntimeError(f"[gemini] No predictions.jsonl found under {dest_uri}")
    all_lines: list[dict] = []
    for fp in result_files:
        shard = pd.read_json(f"gs://{fp}", lines=True)  # type: ignore[call-overload]
        all_lines.extend(shard.to_dict("records"))
    log.info(f"[gemini] Recovery: {len(all_lines)} results from {dest_uri}")
    results: dict[int, list[int] | None] = {}
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
            results[idx] = _parse_entailment_response(text)
        except Exception:
            results[idx] = None
    return results


async def run_gemini_batch(
    prompts: list[dict],
    gcs_location: str,
    poll_interval: int,
) -> dict[int, list[int] | None]:
    if not prompts:
        return {}
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return await loop.run_in_executor(
            executor, _run_gemini_sync, prompts, gcs_location, poll_interval
        )


def _download_claude_job(job_name: str) -> dict[int, list[int] | None]:
    """Re-download results from an already-completed Vertex Claude batch job by job name."""
    # Parse project/location from job_name: projects/<proj>/locations/<loc>/batchPredictionJobs/<id>
    parts = job_name.split("/")
    project  = parts[1] if len(parts) > 1 else VERTEX_GEMINI_PROJECT
    location = parts[3] if len(parts) > 3 else "us-east5"
    client    = genai.Client(vertexai=True, project=project, location=location)
    batch_job = client.batches.get(name=job_name)
    if batch_job.state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"[claude] Job {job_name} is not succeeded: {batch_job.state}")
    if batch_job.dest is None:
        raise RuntimeError(f"[claude] Job {job_name} has no dest")
    dest_uri     = batch_job.dest.gcs_uri
    fs           = fsspec.filesystem("gcs")
    result_files = fs.glob(f"{dest_uri}/*/predictions.jsonl")
    if not result_files:
        raise RuntimeError(f"[claude] No predictions.jsonl found under {dest_uri}")
    all_lines: list[dict] = []
    for fp in result_files:
        shard = pd.read_json(f"gs://{fp}", lines=True)  # type: ignore[call-overload]
        all_lines.extend(shard.to_dict("records"))
    log.info(f"[claude] Recovery: {len(all_lines)} results from {dest_uri}")
    results: dict[int, list[int] | None] = {}
    for line in all_lines:
        raw_key = line.get("custom_id")
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
            blocks = line["response"]["content"]
            text: str | None = next((b["text"] for b in blocks if b.get("type") == "text"), None)
            results[idx] = _parse_entailment_response(text) if text is not None else None
        except Exception:
            results[idx] = None
    return results


# ── Claude batch (Vertex AI) ──────────────────────────────────────────────────
async def run_claude_batch(
    prompts: list[dict],
    poll_interval: int,
) -> dict[int, list[int] | None]:
    if not prompts:
        return {}

    requests = [
        {
            "custom_id":          str(p["idx"]),
            "user_prompt":        p["user_prompt"],
            "model_id":           "claude_haiku_batch",
            "system_instructions": SYSTEM_PROMPT,
            "example_messages":   p.get("example_messages"),
        }
        for p in prompts
    ]

    log.info(f"[claude] Submitting {len(requests):,} requests via Vertex batch ...")
    batch_results = await send_batch_messages(requests, poll_interval=poll_interval)

    results: dict[int, list[int] | None] = {}
    for cid, text in batch_results.items():
        try:
            idx = int(cid)
        except (ValueError, TypeError):
            continue
        results[idx] = _parse_entailment_response(text) if text is not None else None

    succeeded = sum(1 for v in results.values() if v is not None)
    log.info(f"[claude] {succeeded:,}/{len(prompts):,} succeeded")
    return results


# ── Concurrent online juror ───────────────────────────────────────────────────
async def run_claude_online(
    prompts: list[dict],
    tpm: int,
    rate: int,
) -> dict[int, list[int] | None]:
    """Claude juror via concurrent online calls instead of cloud batch.

    Same model as the batch path (claude-haiku-4-5), reached through Bedrock on the
    An OpenAI-compatible endpoint may be used when cloud batch is
    unavailable -- e.g. the Gemini/Claude GCS bucket-location mismatch. No batch
    discount applies, so prefer the batch path when it works.
    """
    if not prompts:
        return {}

    tpm_limiter = _TPMRateLimiter(tpm)
    semaphore   = asyncio.Semaphore(rate)

    async def _call(p: dict) -> list[int] | None:
        estimated = count_tokens(SYSTEM_PROMPT + p["user_prompt"]) + _example_tokens(p)
        await tpm_limiter.wait(estimated)
        async with semaphore:
            try:
                text = await send_single_message(
                    user_prompt=p["user_prompt"],
                    system_instructions=SYSTEM_PROMPT,
                    model_id="claude_haiku_sandbox",
                    backend="openai",
                    example_messages=p.get("example_messages"),
                )
                return _parse_entailment_response(text)
            except Exception as e:
                log.warning(f"[claude] idx={p['idx']} failed: {e}")
                return None

    log.info(f"[claude] Sending {len(prompts):,} requests via online backend "
             f"(tpm={tpm:,}, rate={rate}) ...")
    responses = await asyncio.gather(*[_call(p) for p in prompts])
    results = {p["idx"]: r for p, r in zip(prompts, responses)}
    succeeded = sum(1 for v in results.values() if v is not None)
    log.info(f"[claude] {succeeded:,}/{len(prompts):,} succeeded")
    return results


async def run_gpt_batch(
    prompts: list[dict],
    tpm: int,
    rate: int,
) -> dict[int, list[int] | None]:
    if not prompts:
        return {}

    tpm_limiter = _TPMRateLimiter(tpm)
    semaphore   = asyncio.Semaphore(rate)

    async def _call(p: dict) -> list[int] | None:
        estimated = count_tokens(SYSTEM_PROMPT + p["user_prompt"]) + _example_tokens(p)
        await tpm_limiter.wait(estimated)
        async with semaphore:
            try:
                text = await send_single_message(
                    user_prompt=p["user_prompt"],
                    system_instructions=SYSTEM_PROMPT,
                    model_id="gpt5_nano_sandbox",
                    backend="openai",
                    example_messages=p.get("example_messages"),
                )
                return _parse_entailment_response(text)
            except Exception as e:
                log.warning(f"[gpt] idx={p['idx']} failed: {e}")
                return None

    log.info(f"[gpt] Sending {len(prompts):,} requests via online backend (tpm={tpm:,}, rate={rate}) ...")
    responses = await asyncio.gather(*[_call(p) for p in prompts])
    results = {p["idx"]: r for p, r in zip(prompts, responses)}
    succeeded = sum(1 for v in results.values() if v is not None)
    log.info(f"[gpt] {succeeded:,}/{len(prompts):,} succeeded")
    return results


# ── Result organisation and scoring ──────────────────────────────────────────
def _organize(
    prompts: list[dict],
    results: dict[int, list[int] | None],
) -> dict[tuple, dict[str, list[int] | None]]:
    """Group results by comparison_key → direction (r/p) → entailed index list."""
    org: dict = {}
    for p in prompts:
        key = p["comparison_key"]
        org.setdefault(key, {"r": None, "p": None})
        org[key][p["direction"]] = results.get(p["idx"])
    return org


def _score(
    ref_facts: list[str],
    cand_facts: list[str],
    r_indices: list[int] | None,
    p_indices: list[int] | None,
) -> tuple[float, float, list[str], list[str]]:
    """Return (precision, recall, entailed_cand_facts, entailed_ref_facts)."""
    entailed_ref  = [ref_facts[i]  for i in (r_indices or []) if i < len(ref_facts)]
    entailed_cand = [cand_facts[i] for i in (p_indices or []) if i < len(cand_facts)]
    recall    = len(entailed_ref)  / len(ref_facts)  if ref_facts  else 0.0
    precision = len(entailed_cand) / len(cand_facts) if cand_facts else 0.0
    return precision, recall, entailed_cand, entailed_ref


def _consensus(
    ref_facts: list[str],
    cand_facts: list[str],
    all_r: dict[str, list[int] | None],
    all_p: dict[str, list[int] | None],
    threshold: int = 2,
) -> tuple[float, float, list[str], list[str]]:
    """≥threshold jurors must agree for a fact index to be considered entailed."""
    cons_ref = [
        ref_facts[i] for i in range(len(ref_facts))
        if sum(1 for r in all_r.values() if r and i in r) >= threshold
    ]
    cons_cand = [
        cand_facts[i] for i in range(len(cand_facts))
        if sum(1 for p in all_p.values() if p and i in p) >= threshold
    ]
    recall    = len(cons_ref)  / len(ref_facts)  if ref_facts  else 0.0
    precision = len(cons_cand) / len(cand_facts) if cand_facts else 0.0
    return precision, recall, cons_cand, cons_ref


# ── Main ──────────────────────────────────────────────────────────────────────
async def main(args):
    rerun_juror: str | None = getattr(args, "rerun_juror", None)
    force_ids = load_force_ids(args.force_ids, args.force_ids_file)
    force_models = {re.sub(r"_(batch|sandbox)$", "", model) for model in args.force_models}

    df = pd.read_csv(args.facts, dtype=str)
    log.info(f"Loaded {len(df)} rows from {args.facts}")

    # Predictions always come from --facts (everything that isn't the gold source).
    pred_df = df[df["source_name"] != args.gold_source]
    # Gold facts come from --gold-facts when given, else from --facts itself.
    if args.gold_facts:
        gold_src_df = pd.read_csv(args.gold_facts, dtype=str)
        log.info(f"Loaded {len(gold_src_df)} gold rows from {args.gold_facts}")
        gold_df = gold_src_df[gold_src_df["source_name"] == args.gold_source]
    else:
        gold_df = df[df["source_name"] == args.gold_source]
    log.info(f"Gold rows: {len(gold_df)}, Prediction rows: {len(pred_df)}")

    gold_lookup: dict[str, list[str]] = {}
    for _, row in gold_df.iterrows():
        facts = _parse_facts(row.get("facts_atomic"))
        if facts:
            gold_lookup[str(row["question_id"])] = facts
    log.info(f"Gold: {len(gold_lookup)} questions with facts")

    # When rerunnning a single juror, load existing rows so other jurors' results
    # can be preserved, and skip the completed-pairs check so every row is rerun.
    existing_rows: dict[tuple, dict] = {}
    if rerun_juror:
        if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
            for _, row in pd.read_csv(args.output, dtype=str).iterrows():
                k = (str(row["question_id"]), str(row["source_name"]), str(row.get("model", "")))
                existing_rows[k] = row.to_dict()
        log.info(f"Rerun mode: juror={rerun_juror}, loaded {len(existing_rows)} existing rows")
        completed: set = set()
    else:
        if force_ids:
            removed = drop_rows_for_ids(args.output, "question_id", force_ids)
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
        completed = load_completed_pairs(
            args.output, ["question_id", "source_name", "model"],
            nonempty_col="consensus_recall",
        )
        if completed:
            log.info(f"Resuming: {len(completed)} comparisons already done")

    comparisons: list[tuple] = []
    skipped_missing = skipped_done = 0
    for _, row in pred_df.iterrows():
        qid    = str(row["question_id"])
        source = str(row["source_name"])
        model  = str(row.get("model", ""))
        if args.force_source and source not in args.force_source:
            continue
        if force_models and re.sub(r"_(batch|sandbox)$", "", model) not in force_models:
            continue
        if force_ids and qid not in force_ids:
            continue
        if (qid, source, model) in completed:
            skipped_done += 1
            continue
        ref_facts  = gold_lookup.get(qid, [])
        cand_facts = _parse_facts(row.get("facts_atomic"))
        if not ref_facts or not cand_facts:
            skipped_missing += 1
            continue
        comparisons.append((qid, source, model, ref_facts, cand_facts))

    log.info(
        f"{len(comparisons)} comparisons to run "
        f"({skipped_done} already done, {skipped_missing} missing facts)"
    )
    if not comparisons:
        log.info("Nothing to do.")
        return

    examples_by_dir = load_examples(args.examples)
    prompts = build_prompts(comparisons, examples_by_dir)
    log.info(f"Built {len(prompts):,} entailment prompts")
    log_token_estimate(prompts, ["gemini"] if args.gemini_only else None)

    if args.dry_run:
        log.info("[DRY RUN] Stopping before submission. No batch jobs submitted.")
        return

    if args.gemini_only:
        if args.gemini_job_name:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                gemini_raw = await loop.run_in_executor(
                    executor, _download_gemini_job, args.gemini_job_name
                )
        else:
            gemini_raw = await run_gemini_batch(prompts, args.gcs_location, args.poll_interval)

        valid_results = sum(value is not None for value in gemini_raw.values())
        if len(gemini_raw) != len(prompts) or valid_results != len(prompts):
            raise RuntimeError(
                f"gemini juror returned {valid_results}/{len(prompts)} valid results; "
                "refusing to write incomplete scores"
            )

        gemini_org = _organize(prompts, gemini_raw)
        writer = CsvWriter(args.output, OUTPUT_FIELDS)
        succeeded = 0
        for qid, source, model, ref_facts, cand_facts in comparisons:
            dirs = gemini_org[(qid, source, model)]
            precision, recall, ent_cand, ent_ref = _score(
                ref_facts, cand_facts, dirs["r"], dirs["p"]
            )
            entry = {
                "question_id": qid,
                "source_name": source,
                "model": model,
                "gemini_precision": round(precision, 4),
                "gemini_recall": round(recall, 4),
                "gemini_entailed_ref_facts": json.dumps(ent_ref),
                "gemini_entailed_cand_facts": json.dumps(ent_cand),
                "claude_precision": "",
                "claude_recall": "",
                "claude_entailed_ref_facts": "",
                "claude_entailed_cand_facts": "",
                "gpt_precision": "",
                "gpt_recall": "",
                "gpt_entailed_ref_facts": "",
                "gpt_entailed_cand_facts": "",
                # These aliases keep existing single-score consumers working.
                "avg_precision": round(precision, 4),
                "avg_recall": round(recall, 4),
                "consensus_precision": round(precision, 4),
                "consensus_recall": round(recall, 4),
                "consensus_entailed_ref_facts": json.dumps(ent_ref),
                "consensus_entailed_cand_facts": json.dumps(ent_cand),
            }
            await writer.write(entry)
            succeeded += 1
        log.info(f"Done. {succeeded}/{len(comparisons)} Gemini-only rows written to {args.output}")
        log_token_stats(log, prefix="[gemini] ")
        return

    if rerun_juror:
        log.info(f"Running single juror: {rerun_juror} ...")
        if rerun_juror == "gemini":
            target_raw = await run_gemini_batch(prompts, args.gcs_location, args.poll_interval)
        elif rerun_juror == "claude":
            target_raw = await (
                run_claude_online(prompts, args.claude_tpm, args.rate)
                if args.claude_backend == "online"
                else run_claude_batch(prompts, args.poll_interval)
            )
        else:
            target_raw = await run_gpt_batch(prompts, args.gpt_tpm, args.rate)
        target_org = _organize(prompts, target_raw)

        writer = CsvWriter(args.output, OUTPUT_FIELDS, overwrite=True)
        succeeded = 0
        for qid, source, model, ref_facts, cand_facts in comparisons:
            key = (qid, source, model)
            ex  = existing_rows.get(key, {})
            entry = {"question_id": qid, "source_name": source, "model": model}

            juror_prec:     dict[str, float]     = {}
            juror_rec:      dict[str, float]     = {}
            juror_ent_ref:  dict[str, list[str]] = {}
            juror_ent_cand: dict[str, list[str]] = {}

            for j in JURORS:
                if j == rerun_juror:
                    dirs = target_org.get(key, {"r": None, "p": None})
                    prec, rec, ent_cand, ent_ref = _score(
                        ref_facts, cand_facts, dirs["r"], dirs["p"]
                    )
                else:
                    prec     = float(ex.get(f"{j}_precision", 0) or 0)
                    rec      = float(ex.get(f"{j}_recall",    0) or 0)
                    ent_ref  = _parse_facts(ex.get(f"{j}_entailed_ref_facts",  "[]"))
                    ent_cand = _parse_facts(ex.get(f"{j}_entailed_cand_facts", "[]"))

                entry.update({
                    f"{j}_precision":           round(prec, 4),
                    f"{j}_recall":              round(rec,  4),
                    f"{j}_entailed_ref_facts":  json.dumps(ent_ref),
                    f"{j}_entailed_cand_facts": json.dumps(ent_cand),
                })
                juror_prec[j]     = prec
                juror_rec[j]      = rec
                juror_ent_ref[j]  = ent_ref
                juror_ent_cand[j] = ent_cand

            # avg/consensus are only meaningful when the OTHER jurors actually have
            # verdicts for this row. Rerunning one juror into a fresh output file
            # leaves them empty, and averaging over absent jurors would report a
            # third of the real score while consensus (needs ≥2 jurors) would be
            # uniformly 0 — silently wrong for the seven scripts that read these
            # columns. Blank them instead, which also stops a later resume from
            # treating the row as complete (it tests consensus_recall non-empty).
            others = [j for j in JURORS
                      if j != rerun_juror and str(ex.get(f"{j}_recall", "")).strip() != ""]
            if not others:
                entry.update({
                    "avg_precision": "", "avg_recall": "",
                    "consensus_precision": "", "consensus_recall": "",
                    "consensus_entailed_ref_facts": "", "consensus_entailed_cand_facts": "",
                })
            else:
                entry["avg_precision"] = round(sum(juror_prec.values()) / len(juror_prec), 4)
                entry["avg_recall"]    = round(sum(juror_rec.values())  / len(juror_rec),  4)

                # Consensus: ≥2 jurors must have entailed the fact (string-based match)
                cons_ref = [
                    f for f in ref_facts
                    if sum(1 for ent in juror_ent_ref.values()  if f in ent) >= 2
                ]
                cons_cand = [
                    f for f in cand_facts
                    if sum(1 for ent in juror_ent_cand.values() if f in ent) >= 2
                ]
                c_rec  = len(cons_ref)  / len(ref_facts)  if ref_facts  else 0.0
                c_prec = len(cons_cand) / len(cand_facts) if cand_facts else 0.0
                entry.update({
                    "consensus_precision":           round(c_prec, 4),
                    "consensus_recall":              round(c_rec,  4),
                    "consensus_entailed_ref_facts":  json.dumps(cons_ref),
                    "consensus_entailed_cand_facts": json.dumps(cons_cand),
                })

            await writer.write(entry)
            succeeded += 1

        log.info(f"Done. {succeeded}/{len(comparisons)} rows written to {args.output}")
        log_token_stats(log, prefix=f"[{rerun_juror}] ")
        return

    # Recovery mode: re-download Gemini/Claude from completed Vertex jobs, run GPT fresh
    gemini_job_name = getattr(args, "gemini_job_name", None)
    claude_job_name = getattr(args, "claude_job_name", None)
    if gemini_job_name or claude_job_name:
        log.info("Recovery mode: re-downloading completed juror results ...")

        async def _empty() -> dict[int, list[int] | None]:
            return {}

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=2) as executor:
            gemini_fut = loop.run_in_executor(executor, _download_gemini_job, gemini_job_name) if gemini_job_name else _empty()
            if claude_job_name:
                claude_fut = loop.run_in_executor(executor, _download_claude_job, claude_job_name)
            elif args.claude_backend == "online":
                claude_fut = run_claude_online(prompts, args.claude_tpm, args.rate)
            else:
                claude_fut = _empty()
            gemini_raw, claude_raw, gpt_raw = await asyncio.gather(
                gemini_fut,
                claude_fut,
                run_gpt_batch(prompts, args.gpt_tpm, args.rate),
            )
        log.info(f"[recovery] gemini={len(gemini_raw)} claude={len(claude_raw)} gpt={len(gpt_raw)}")
    else:
        log.info(f"Submitting all three jurors concurrently "
                 f"(claude via {args.claude_backend}) ...")
        claude_task = (run_claude_online(prompts, args.claude_tpm, args.rate)
                       if args.claude_backend == "online"
                       else run_claude_batch(prompts, args.poll_interval))
        gemini_raw, claude_raw, gpt_raw = await asyncio.gather(
            run_gemini_batch(prompts, args.gcs_location, args.poll_interval),
            claude_task,
            run_gpt_batch(prompts, args.gpt_tpm, args.rate),
        )

    for juror, results in (("gemini", gemini_raw), ("claude", claude_raw), ("gpt", gpt_raw)):
        if len(results) != len(prompts):
            raise RuntimeError(
                f"{juror} juror returned {len(results)}/{len(prompts)} results; "
                "refusing to write incomplete scores"
            )

    gemini_org = _organize(prompts, gemini_raw)
    claude_org  = _organize(prompts, claude_raw)
    gpt_org     = _organize(prompts, gpt_raw)

    writer = CsvWriter(args.output, OUTPUT_FIELDS)
    succeeded = 0
    for qid, source, model, ref_facts, cand_facts in comparisons:
        key  = (qid, source, model)
        by_j = {"gemini": gemini_org.get(key, {"r": None, "p": None}),
                "claude": claude_org.get(key,  {"r": None, "p": None}),
                "gpt":    gpt_org.get(key,     {"r": None, "p": None})}

        entry = {"question_id": qid, "source_name": source, "model": model}
        precisions, recalls = [], []
        all_r = {j: by_j[j]["r"] for j in JURORS}
        all_p = {j: by_j[j]["p"] for j in JURORS}

        for j in JURORS:
            prec, rec, ent_cand, ent_ref = _score(
                ref_facts, cand_facts, by_j[j]["r"], by_j[j]["p"]
            )
            entry.update({
                f"{j}_precision":           round(prec, 4),
                f"{j}_recall":              round(rec,  4),
                f"{j}_entailed_ref_facts":  json.dumps(ent_ref),
                f"{j}_entailed_cand_facts": json.dumps(ent_cand),
            })
            precisions.append(prec)
            recalls.append(rec)

        entry["avg_precision"] = round(sum(precisions) / len(precisions), 4)
        entry["avg_recall"]    = round(sum(recalls)    / len(recalls),    4)

        c_prec, c_rec, c_cand, c_ref = _consensus(ref_facts, cand_facts, all_r, all_p)
        entry.update({
            "consensus_precision":           round(c_prec, 4),
            "consensus_recall":              round(c_rec,  4),
            "consensus_entailed_ref_facts":  json.dumps(c_ref),
            "consensus_entailed_cand_facts": json.dumps(c_cand),
        })

        await writer.write(entry)
        succeeded += 1

    log.info(f"Done. {succeeded}/{len(comparisons)} comparisons written to {args.output}")
    log_token_stats(log, prefix="[claude+gpt] ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fact-based precision/recall evaluation via 3-model entailment jury"
    )
    parser.add_argument("--facts",         required=True,
                        help="Atomic facts CSV (source_name, question_id, model, facts_atomic)")
    parser.add_argument("--gcs-location",  required=True,
                        help="GCS prefix for Gemini batch I/O (e.g. gs://bucket/entailment)")
    parser.add_argument("-o", "--output",  required=True,
                        help="Output CSV")
    parser.add_argument("--gold-source",   default="gold",
                        help="source_name value for gold/reference facts (default: gold)")
    parser.add_argument("--gold-facts",     default=None,
                        help="Separate CSV holding the gold/reference facts. When given, gold "
                             "rows (source_name == --gold-source) are read from here instead of "
                             "--facts; predictions still come from --facts. Matched by question_id.")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between batch job status checks (default: 60)")
    parser.add_argument("--gpt-tpm",       type=int, default=5_000_000,
                        help="Online juror token-per-minute limit")
    parser.add_argument("--rate",          type=int, default=20,
                        help="Max concurrent GPT nano requests (default: 20)")
    parser.add_argument("--claude-backend",   choices=["batch", "online"], default="batch",
                        help="How to reach the claude juror: 'batch' = Vertex Claude batch "
                             "(default); 'online' = concurrent provider calls "
                             "(same claude-haiku-4-5 model, use when Vertex batch is failing).")
    parser.add_argument("--claude-tpm",       type=int, default=1_000_000,
                        help="Token/min budget for --claude-backend online")
    parser.add_argument("--rerun-juror",      choices=["gemini", "claude", "gpt"], default=None,
                        help="Rerun a single juror and recalculate avg/consensus (preserves other jurors)")
    parser.add_argument("--gemini-only",      action="store_true",
                        help="Use only Gemini 3.1 Flash-Lite for entailment; aggregate fields mirror Gemini and other juror fields are blank.")
    parser.add_argument("--dry-run",          action="store_true",
                        help="Estimate token count and cost without submitting any batch jobs")
    parser.add_argument("--examples",         default=None,
                        help="Few-shot examples JSON from build_fact_examples.py; prepended "
                             "as multi-turn human-validated demonstrations (per direction).")
    parser.add_argument("--force-ids",        nargs="+", default=None, metavar="QUESTION_ID",
                        help="Re-score only these question_ids, dropping their stale output rows first.")
    parser.add_argument("--force-ids-file",   type=str, default=None,
                        help="File with one question_id per line to re-score (unioned with --force-ids).")
    parser.add_argument("--force-source",     nargs="+", default=[], metavar="SOURCE_NAME",
                        help="Re-score these source_name(s), dropping their stale output rows first "
                             "(e.g. gemini_recent gemini_recent200 gemini_bm25 gemini_dense).")
    parser.add_argument("--force-models",     nargs="+", default=[], metavar="MODEL",
                        help="With --force-source, re-score only these models in the selected sources.")
    parser.add_argument("--gemini-job-name",  default=None,
                        help="Recovery: re-download Gemini results from this completed Vertex job name "
                             "(e.g. projects/PROJECT/locations/LOCATION/batchPredictionJobs/ID)")
    parser.add_argument("--claude-job-name",  default=None,
                        help="Recovery: re-download Claude results from this completed Vertex job name "
                             "(e.g. projects/PROJECT/locations/LOCATION/batchPredictionJobs/ID)")
    args = parser.parse_args()
    if args.force_models and not args.force_source:
        parser.error("--force-models requires --force-source")
    asyncio.run(main(args))
