"""Vertex AI Gemini batch-inference engine for the BRIE generation pipeline.

This is a *synchronous* port of the batch helper in ``src/evaluation/utils.py``
(the generation pipeline is thread-based, not async, so the asyncio wrapper is
dropped). Every pipeline stage builds a list of request dicts and calls
``send_gemini_batch``; results come back keyed by ``custom_id``.

Design guarantees that matter for this pipeline:

* **Strict key-based attribution.** Vertex batch does *not* preserve input
  order and may shard output across several ``predictions.jsonl`` files. Results
  are matched to inputs *only* through the per-request ``custom_id`` (sent as the
  batch ``key``), never positionally. Duplicate ``custom_id``s are rejected
  before submission, and any requested id missing from the output is logged as a
  failure rather than silently dropped or misattributed.

* **Two-level resume.**
    - *Item level* is the caller's job: stages skip ``custom_id``s already
      present in their output files, so only outstanding work is passed in.
    - *Job level* lives here: each submitted Vertex job name is written to a
      manifest. If the process dies mid-poll and is restarted with the *same*
      request set, the manifest is re-attached (``client.batches.get``) instead
      of resubmitting — an in-flight or finished job is never paid for twice.

* **Partial completion.** Successful results are returned; failed/missing ones
  are returned as ``None`` and appended to ``{stage}_failed.jsonl`` with their
  reason, so a later rerun retries exactly them.

* **200K-per-job limit.** Requests are auto-split into sequential sub-jobs of
  ``MAX_REQUESTS_PER_JOB`` and tracked individually in the manifest.

* **Dry-run.** ``log_cost_estimate`` reports request count, token counts, the
  estimated batch cost, and the job-limit check without submitting anything.
"""

import json
import os
import time
import uuid

import fsspec
import pandas as pd
from google import genai
from google.genai.types import CreateBatchJobConfig

from .utils import (  # noqa: F401  (safe_json_parse re-exported for stages)
    count_tokens,
    safe_json_parse,
    VERTEX_PROJECT,
    VERTEX_LOCATION,
    GEMINI_MODEL_ID,
)

# ── GCS config (batch-specific) ───────────────────────────────────────────────
VERTEX_GCS_BUCKET = os.environ.get("BRIE_VERTEX_GCS_BUCKET", "")

# Default generation config applied to every request unless the request (or the
# send_gemini_batch call) overrides it. The online pipeline used 65535 tokens.
DEFAULT_GENERATION_CONFIG: dict = {"maxOutputTokens": 65535}

# Vertex batch prediction caps each job at 200K requests.
MAX_REQUESTS_PER_JOB = 200_000

_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PAUSED",
}

# ── Pricing (Gemini 2.5 Pro, Vertex batch = 50% of online) ────────────────────
# Online list price for prompts <=200K tokens: $1.25 / 1M input, $10 / 1M output.
# Batch prediction applies a 50% discount. Override via env if pricing changes.
GEMINI_BATCH_INPUT_COST_PER_1M  = float(os.environ.get("GEMINI_BATCH_INPUT_COST_PER_1M",  "0.625"))
GEMINI_BATCH_OUTPUT_COST_PER_1M = float(os.environ.get("GEMINI_BATCH_OUTPUT_COST_PER_1M", "5.0"))


# ── Request construction ──────────────────────────────────────────────────────
def build_request_body(req: dict, default_generation_config: dict | None) -> dict:
    """Build the Vertex Gemini request body for one request dict.

    A request dict contains:
      - "custom_id":            str, unique
      - "user_prompt":          str  (single-turn convenience), OR
      - "contents":             list[dict]  (full multi-turn turns; takes priority)
      - "system_instructions":  str | None   (optional)
      - "generation_config":    dict | None  (optional per-request override)
    """
    if req.get("contents") is not None:
        contents = req["contents"]
    else:
        contents = [{"role": "user", "parts": [{"text": req["user_prompt"]}]}]

    body: dict = {"contents": contents}

    if req.get("system_instructions"):
        body["systemInstruction"] = {"parts": [{"text": req["system_instructions"]}]}

    gen_cfg = req.get("generation_config", default_generation_config)
    if gen_cfg:
        body["generationConfig"] = gen_cfg

    return body


def _build_jsonl_lines(requests: list[dict], default_generation_config: dict | None) -> list[str]:
    return [
        json.dumps({"key": req["custom_id"], "request": build_request_body(req, default_generation_config)})
        for req in requests
    ]


def _check_unique_custom_ids(requests: list[dict]) -> None:
    seen: set[str] = set()
    dups: set[str] = set()
    for req in requests:
        cid = req["custom_id"]
        if cid in seen:
            dups.add(cid)
        seen.add(cid)
    if dups:
        raise ValueError(
            f"Duplicate custom_id(s) in batch request set ({len(dups)}): "
            f"{sorted(dups)[:10]}{' ...' if len(dups) > 10 else ''}. "
            "custom_id is the only key used to map results back, so it must be unique."
        )


# ── Manifest (job-level resume) ───────────────────────────────────────────────
def _manifest_path(state_dir: str, stage: str) -> str:
    return os.path.join(state_dir, f"{stage}_manifest.json")


def _failed_path(state_dir: str, stage: str) -> str:
    return os.path.join(state_dir, f"{stage}_failed.jsonl")


def _load_manifest(state_dir: str, stage: str) -> dict | None:
    path = _manifest_path(state_dir, stage)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _save_manifest(state_dir: str, stage: str, manifest: dict) -> None:
    os.makedirs(state_dir, exist_ok=True)
    path = _manifest_path(state_dir, stage)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)


def _record_failures(state_dir: str, stage: str, failures: list[dict]) -> None:
    if not failures:
        return
    os.makedirs(state_dir, exist_ok=True)
    with open(_failed_path(state_dir, stage), "a") as f:
        for item in failures:
            f.write(json.dumps(item) + "\n")


# ── Core: submit / re-attach / poll / parse ───────────────────────────────────
def _chunk(seq: list, size: int) -> list[list]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def _parse_predictions(dest_uri: str, expected_ids: set[str]) -> tuple[dict[str, str | None], list[dict]]:
    """Download predictions.jsonl shards and map them back strictly by key.

    Returns (results, failures) where results covers exactly ``expected_ids``.
    """
    fs = fsspec.filesystem("gcs")
    result_files = fs.glob(f"{dest_uri}/*/predictions.jsonl")

    results: dict[str, str | None] = {cid: None for cid in expected_ids}
    failures: list[dict] = []

    if not result_files:
        for cid in expected_ids:
            failures.append({"custom_id": cid, "reason": "no predictions.jsonl produced"})
        return results, failures

    # Parse JSONL by hand (not pd.read_json): pandas infers the ``key`` column as
    # Numeric coercion can alter opaque identifiers, so preserve keys as strings.
    # strings and must round-trip verbatim.
    all_lines: list[dict] = []
    for fp in result_files:
        with fs.open(fp, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    line_by_key = {str(line.get("key")): line for line in all_lines if line.get("key") is not None}

    for cid in expected_ids:
        line = line_by_key.get(cid)
        if line is None:
            failures.append({"custom_id": cid, "reason": "key missing from output"})
            continue
        status = line.get("status")
        if status and status != "":
            failures.append({"custom_id": cid, "reason": f"status={status!r}"})
            continue
        try:
            results[cid] = line["response"]["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as exc:
            failures.append({"custom_id": cid, "reason": f"parse error: {exc}"})

    return results, failures


def send_gemini_batch(
    requests: list[dict],
    stage: str,
    state_dir: str,
    poll_interval: int = 60,
    generation_config: dict | None = DEFAULT_GENERATION_CONFIG,
    model_id: str = GEMINI_MODEL_ID,
    max_per_job: int = MAX_REQUESTS_PER_JOB,
) -> dict[str, str | None]:
    """Run ``requests`` through Vertex Gemini batch prediction.

    See the module docstring for the resume / attribution guarantees. Returns
    ``{custom_id: response_text}``; ``None`` marks a failed/missing request (also
    appended to ``{stage}_failed.jsonl``).
    """
    if not requests:
        return {}
    if not VERTEX_PROJECT:
        raise RuntimeError("Set BRIE_VERTEX_PROJECT before using Vertex batch")
    if not VERTEX_GCS_BUCKET:
        raise RuntimeError("Set BRIE_VERTEX_GCS_BUCKET before using Vertex batch")

    _check_unique_custom_ids(requests)

    requested_ids = [r["custom_id"] for r in requests]
    requested_set = set(requested_ids)
    req_by_id = {r["custom_id"]: r for r in requests}

    client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)

    # ── Job-level resume: re-attach if a manifest for the *same* request set exists.
    manifest = _load_manifest(state_dir, stage)
    reattach = False
    if manifest is not None:
        manifest_ids = set()
        for sj in manifest.get("subjobs", []):
            manifest_ids.update(sj.get("custom_ids", []))
        if manifest_ids == requested_set:
            reattach = True
            print(f"[batch:{stage}] Re-attaching to {len(manifest['subjobs'])} job(s) from manifest "
                  f"(same {len(requested_set):,} request set).")
        else:
            print(f"[batch:{stage}] Existing manifest covers a different request set "
                  f"({len(manifest_ids):,} vs {len(requested_set):,}); starting fresh.")

    if not reattach:
        # ── Phase 1: upload + submit one sub-job per <=max_per_job chunk.
        subjobs = []
        for idx, chunk in enumerate(_chunk(requested_ids, max_per_job)):
            chunk_reqs = [req_by_id[cid] for cid in chunk]
            job_uuid      = uuid.uuid4().hex[:12]
            gcs_prefix    = f"gs://{VERTEX_GCS_BUCKET}/predictions/{stage}/{job_uuid}"
            input_uri     = f"{gcs_prefix}/input.jsonl"
            output_prefix = f"{gcs_prefix}/output"

            lines = _build_jsonl_lines(chunk_reqs, generation_config)
            print(f"[batch:{stage}] sub-job {idx}: uploading {len(lines):,} requests -> {input_uri}")
            with fsspec.open(input_uri, "w") as f:
                f.write("\n".join(lines))

            batch_job = client.batches.create(
                model=model_id,
                src=input_uri,
                config=CreateBatchJobConfig(dest=output_prefix),
            )
            print(f"[batch:{stage}] sub-job {idx} submitted: {batch_job.name}  state={batch_job.state}")
            subjobs.append({
                "index": idx,
                "job_name": batch_job.name,
                "output_prefix": output_prefix,
                "custom_ids": chunk,
                "state": str(batch_job.state),
            })
        manifest = {"stage": stage, "model": model_id, "subjobs": subjobs}
        _save_manifest(state_dir, stage, manifest)

    # ── Phase 2: poll all sub-jobs together until terminal. Job objects are kept
    # out of the manifest (not JSON-serializable); only the state string is saved.
    assert manifest is not None  # set by either the reattach or submit branch above
    sj_by_job = {sj["job_name"]: sj for sj in manifest["subjobs"]}
    pending = dict(sj_by_job)
    finished_jobs = {}
    while pending:
        time.sleep(poll_interval)
        still_pending: dict[str, dict] = {}
        for job_name, sj in pending.items():
            batch_job = client.batches.get(name=job_name)
            sj["state"] = str(batch_job.state)
            print(f"[batch:{stage}] sub-job {sj['index']} ({job_name}): {batch_job.state}")
            if batch_job.state in _TERMINAL_STATES:
                finished_jobs[job_name] = batch_job
            else:
                still_pending[job_name] = sj
        pending = still_pending
        _save_manifest(state_dir, stage, manifest)

    # ── Phase 3: download + parse strictly by key; collect failures.
    results: dict[str, str | None] = {}
    all_failures: list[dict] = []
    for job_name, batch_job in finished_jobs.items():
        sj       = sj_by_job[job_name]
        expected = set(sj["custom_ids"])
        if batch_job.state != "JOB_STATE_SUCCEEDED":
            reason = f"job {sj['state']}: {getattr(batch_job, 'error', '')}"
            for cid in expected:
                results[cid] = None
                all_failures.append({"custom_id": cid, "reason": reason})
            print(f"[batch:{stage}] sub-job {sj['index']} did not succeed: {reason}")
            continue
        sub_results, sub_failures = _parse_predictions(batch_job.dest.gcs_uri, expected)
        results.update(sub_results)
        all_failures.extend(sub_failures)

    _record_failures(state_dir, stage, all_failures)

    n_ok = sum(1 for v in results.values() if v is not None)
    print(f"[batch:{stage}] {n_ok:,}/{len(requested_set):,} succeeded; "
          f"{len(all_failures):,} failed (logged to {_failed_path(state_dir, stage)})")
    return results


# ── Dry-run / cost estimate ───────────────────────────────────────────────────
def log_cost_estimate(
    requests: list[dict],
    stage: str,
    generation_config: dict | None = DEFAULT_GENERATION_CONFIG,
    est_output_tokens_per_req: int = 800,
    max_per_job: int = MAX_REQUESTS_PER_JOB,
) -> dict:
    """Print a token/cost estimate for ``requests`` without submitting anything.

    Mirrors the ``--dry-run`` output of the evaluation batch scripts. Returns the
    computed figures so callers can assert/test on them.
    """
    n = len(requests)
    if n == 0:
        print(f"[batch:{stage}] dry-run: nothing to submit (0 requests).")
        return {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0}

    jsonl_lines = _build_jsonl_lines(requests, generation_config)
    input_tokens  = sum(count_tokens(line) for line in jsonl_lines)
    output_tokens = n * est_output_tokens_per_req
    cost = (input_tokens  / 1_000_000) * GEMINI_BATCH_INPUT_COST_PER_1M \
         + (output_tokens / 1_000_000) * GEMINI_BATCH_OUTPUT_COST_PER_1M
    n_subjobs = (n + max_per_job - 1) // max_per_job

    print(f"── Dry-run estimate [{stage}] ─────────────────────────")
    print(f"  Requests:        {n:>12,}")
    print(f"  Input tokens:    {input_tokens:>12,}   (tiktoken cl100k_base)")
    print(f"  Est. output:     {output_tokens:>12,}   (~{est_output_tokens_per_req}/req)")
    print(f"  Est. cost:       ${cost:>11.2f}   ({GEMINI_MODEL_ID} batch rates)")
    if n > max_per_job:
        print(f"  Job split:       {n:,} -> {n_subjobs} sub-jobs of <= {max_per_job:,}")
    else:
        print(f"  Job limit:       OK ({n:,} / {max_per_job:,})")
    return {"requests": n, "input_tokens": input_tokens, "output_tokens": output_tokens, "cost": cost}


# ── Item-level resume helper ──────────────────────────────────────────────────
def load_completed_ids(path: str, id_col: str, nonempty_col: str | None = None) -> set[str]:
    """Return the set of ``id_col`` values already present in a CSV output.

    If ``nonempty_col`` is given, rows where that column is blank/NaN are treated
    as incomplete so failed rows get retried. Mirrors evaluation's
    ``load_completed_pairs`` for single-key outputs.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    try:
        read_cols = [id_col] if nonempty_col is None else list(dict.fromkeys([id_col, nonempty_col]))
        df = pd.read_csv(path, usecols=read_cols, dtype=str)
        if df.empty:
            return set()
        if nonempty_col is not None:
            df = df[df[nonempty_col].notna() & (df[nonempty_col].str.strip() != "")]
        return set(df[id_col].fillna("").astype(str))
    except Exception:
        return set()
