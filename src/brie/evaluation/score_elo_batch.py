"""
score_elo_batch.py

Pairwise ELO evaluation via LLM judge.

For each question_id, generates all model pairs and asks a judge to pick the
better response per dimension (completeness, relevancy, concision). Each pair
is submitted in both orders (A-vs-B and B-vs-A) as separate rows to mitigate
position bias. ELO ratings are computed from win/loss/tie outcomes.

Judge options (--judge):
  gemini   Gemini Flash-Lite via Vertex GCS batch job  [default]
  claude   Claude Haiku via Vertex Claude batch
  gpt      GPT-compatible concurrent calls
  jury     All three; majority vote determines winner

Checkpointing: completed (question_id, model_a, model_b, position) triples
are skipped on re-run. ELO is recomputed over all results at the end.

Input:
  --questions  CSV with question_id, question (or natural_query), answer (or annotation_sub_answer)
  --responses  One or more CSVs with question_id, model, response
               If a file contains multiple models, use --select-model to filter.

Output:
  --output     Pairwise results CSV (one row per ordered pair × question)
  --elo        ELO summary CSV (one row per model)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import fsspec
import pandas as pd
from google import genai
from google.genai.types import CreateBatchJobConfig

from .utils import (
    VERTEX_LOCATION,
    VERTEX_GEMINI_PROJECT,
    _VERTEX_GEMINI_MODELS,
    _TPMRateLimiter,
    _example_gemini_contents,
    count_tokens,
    CsvWriter,
    load_completed_pairs,
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

DIMS = ("completeness", "relevancy", "concision")

# Default judge model (Vertex GCS batch)
DEFAULT_JUDGE = "gemini"
GEMINI_MODEL_ID = _VERTEX_GEMINI_MODELS["gemini_flash_juror"]

# Per-judge (input $/1M, output $/1M) — batch rates where applicable
_JUDGE_COSTS = {
    "gemini": (0.125, 0.75),   # Gemini 3.1 Flash-Lite Vertex batch (Global)
    "claude": (0.40,  2.00),   # Claude Haiku Vertex batch (50% discount)
    "gpt":    (0.15,  0.60),
}

# Estimated output tokens per pairwise judgment (3 dims × ~40 tokens each)
_EST_OUTPUT_TOKENS = 150

OUTPUT_FIELDS = [
    "question_id", "model_a", "model_b", "position",   # position: "ab" or "ba"
    "judge",
    "completeness_winner", "completeness_winner_model", "completeness_explanation",
    "relevancy_winner",    "relevancy_winner_model",    "relevancy_explanation",
    "concision_winner",    "concision_winner_model",    "concision_explanation",
    "overall_winner",      "overall_winner_model",      # majority across dims; tie if no majority
]

ELO_FIELDS = [
    "model",
    "elo_completeness", "elo_relevancy", "elo_concision", "elo_overall",
    "wins", "losses", "ties", "total", "win_rate", "non_tie_win_rate",
]

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a medical expert evaluating clinical information retrieval responses. "
    "Respond only with a valid JSON object — no markdown, no explanation outside the JSON."
)

PAIRWISE_PROMPT = """\
You are a medical expert comparing two responses to a clinical information retrieval query.
Given a reference answer (gold standard) and two candidate responses (A and B), decide which
response is better on each of the following dimensions, or declare a tie.

Question:
<question>{QUESTION}</question>

Reference answer:
<reference>{REFERENCE}</reference>

Response A:
<response_a>{RESPONSE_A}</response_a>

Response B:
<response_b>{RESPONSE_B}</response_b>

Evaluate on these three dimensions:

Completeness: Which response includes more of the important clinical details present in the \
reference answer? Prefer the response that omits fewer key facts.

Relevancy: Which response stays closer to what the question asks and the reference answer \
covers, without introducing unnecessary or tangential details?

Concision: Which response communicates the necessary information more concisely, without \
excessive verbosity or redundant phrasing?

For each dimension, output "A", "B", or "tie".

Output Format:
{{
    "completeness": {{"winner": "A" | "B" | "tie", "explanation": "..."}},
    "relevancy":    {{"winner": "A" | "B" | "tie", "explanation": "..."}},
    "concision":    {{"winner": "A" | "B" | "tie", "explanation": "..."}}
}}

Ensure the output is valid JSON with double quotes for all keys and string values.\
"""


# ── Pair generation ───────────────────────────────────────────────────────────
def build_pairwise_rows(
    responses: pd.DataFrame,
    questions: pd.DataFrame,
) -> list[dict]:
    """Generate all ordered model pairs per question_id.

    Each unordered pair (A, B) produces two rows — position "ab" and "ba" —
    to mitigate position bias. Returns dicts with question, reference, and
    response texts pre-filled.
    """
    q_col = next((c for c in ("question", "natural_query") if c in questions.columns), None)
    a_col = next((c for c in ("answer", "annotation_sub_answer") if c in questions.columns), None)
    if a_col is None:
        raise ValueError("questions CSV must have an 'answer' or 'annotation_sub_answer' column")

    q_lookup: dict[str, tuple[str, str]] = {}
    for _, row in questions.iterrows():
        qid = str(row["question_id"])
        question  = str(row[q_col]) if q_col else ""
        reference = str(row[a_col])
        q_lookup[qid] = (question, reference)

    # Group responses by question_id → list of (model, response)
    by_qid: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for _, row in responses.iterrows():
        qid   = str(row["question_id"])
        model = str(row["model"])
        resp  = str(row["response"])
        if qid in q_lookup:
            by_qid[qid].append((model, resp))
        else:
            log.warning(f"question_id={qid} model={model} has no matching question — skipping")

    rows: list[dict] = []
    for qid, entries in by_qid.items():
        question, reference = q_lookup[qid]
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                model_a, resp_a = entries[i]
                model_b, resp_b = entries[j]
                base = {
                    "question_id": qid,
                    "question":    question,
                    "reference":   reference,
                }
                # position "ab": A=model_a, B=model_b
                rows.append({**base,
                    "model_a": model_a, "model_b": model_b, "position": "ab",
                    "response_a": resp_a, "response_b": resp_b,
                })
                # position "ba": A=model_b, B=model_a (reversed)
                rows.append({**base,
                    "model_a": model_b, "model_b": model_a, "position": "ba",
                    "response_a": resp_b, "response_b": resp_a,
                })

    log.info(
        f"Generated {len(rows)} pairwise rows "
        f"({len(rows) // 2} unique pairs, 2 positions each) "
        f"across {len(by_qid)} questions"
    )
    return rows


# ── Response parsing ──────────────────────────────────────────────────────────
def _parse_judgment(raw: str | None, model_a: str = "", model_b: str = "") -> dict | None:
    """Parse LLM pairwise judgment JSON. Returns None on failure.

    model_a/model_b are used to populate *_winner_model columns so callers
    don't need to re-resolve A/B labels.
    """
    if raw is None:
        return None
    try:
        parsed = safe_json_parse(raw)
        result = {}
        for dim in DIMS:
            entry = parsed.get(dim, {})
            winner = str(entry.get("winner", "")).strip().upper()
            if winner not in ("A", "B", "TIE"):
                winner = "TIE"
            result[f"{dim}_winner"]       = winner
            result[f"{dim}_winner_model"] = model_a if winner == "A" else (model_b if winner == "B" else "tie")
            result[f"{dim}_explanation"]  = str(entry.get("explanation", ""))
        # Overall: majority across dims; tie if no single winner has majority
        counts: dict[str, int] = {"A": 0, "B": 0, "TIE": 0}
        for dim in DIMS:
            counts[result[f"{dim}_winner"]] += 1
        if counts["A"] >= 2:
            result["overall_winner"]       = "A"
            result["overall_winner_model"] = model_a
        elif counts["B"] >= 2:
            result["overall_winner"]       = "B"
            result["overall_winner_model"] = model_b
        else:
            result["overall_winner"]       = "TIE"
            result["overall_winner_model"] = "tie"
        return result
    except Exception:
        return None


# ── Token / cost estimation ───────────────────────────────────────────────────
def log_cost_estimate(rows: list[dict], judges: list[str], example_tokens: int = 0) -> None:
    total_input = sum(
        count_tokens(
            SYSTEM_PROMPT
            + PAIRWISE_PROMPT.format(
                QUESTION=r["question"],
                REFERENCE=r["reference"],
                RESPONSE_A=r["response_a"],
                RESPONSE_B=r["response_b"],
            )
        )
        + example_tokens
        for r in rows
    )
    total_output = len(rows) * _EST_OUTPUT_TOKENS
    log.info(
        f"Cost estimate — {len(rows):,} requests × {len(judges)} judge(s): "
        f"{total_input:,} input tokens + {total_output:,} output tokens (est.)"
    )
    total_cost = 0.0
    for j in judges:
        in_rate, out_rate = _JUDGE_COSTS[j]
        cost = (total_input / 1e6) * in_rate + (total_output / 1e6) * out_rate
        total_cost += cost
        log.info(f"  [{j}] ${cost:.4f}")
    log.info(f"  Total: ${total_cost:.4f}")
    if len(rows) > 200_000:
        log.warning(
            f"Request count {len(rows):,} exceeds 200K Vertex batch limit — "
            "consider splitting by question_id range."
        )


# ── Gemini batch judge ────────────────────────────────────────────────────────
def _run_gemini_judge_sync(
    prompts: list[dict],
    gcs_location: str,
    poll_interval: int,
) -> dict[str, str | None]:
    """Submit prompts to Gemini Flash-Lite batch, return {idx_str: raw_text}."""
    jsonl_lines = [
        json.dumps({
            "key": str(p["idx"]),
            "request": {
                "contents": _example_gemini_contents(p.get("example_messages"))
                + [{"role": "user", "parts": [{"text": p["prompt"]}]}],
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            },
        })
        for p in prompts
    ]
    run_id           = uuid.uuid4().hex[:12]
    gcs_input_uri    = f"{gcs_location}/elo_input_{run_id}.jsonl"
    gcs_output_prefix = f"{gcs_location}/elo_output/{run_id}"

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

    dest_uri     = batch_job.dest.gcs_uri  # type: ignore[union-attr]
    fs           = fsspec.filesystem("gcs")
    result_files = fs.glob(f"{dest_uri}/*/predictions.jsonl")
    if not result_files:
        log.error(f"[gemini] No predictions.jsonl found under {dest_uri}")
        return {}

    all_lines: list[dict] = []
    for fp in result_files:
        shard = pd.read_json(f"gs://{fp}", lines=True)  # type: ignore[call-overload]
        all_lines.extend(shard.to_dict("records"))

    log.info(f"[gemini] {len(all_lines)} results downloaded")

    results: dict[str, str | None] = {}
    for line in all_lines:
        key = str(line.get("key", ""))
        if not key:
            continue
        try:
            if line.get("status") and line["status"] != "":
                results[key] = None
                continue
            results[key] = line["response"]["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            results[key] = None
    return results


async def _run_gemini_judge(
    prompts: list[dict],
    gcs_location: str,
    poll_interval: int,
) -> dict[str, str | None]:
    if not prompts:
        return {}
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return await loop.run_in_executor(
            executor, _run_gemini_judge_sync, prompts, gcs_location, poll_interval
        )


# ── Claude batch judge ────────────────────────────────────────────────────────
async def _run_claude_judge(prompts: list[dict], poll_interval: int) -> dict[str, str | None]:
    if not prompts:
        return {}
    requests = [
        {
            "custom_id":           str(p["idx"]),
            "user_prompt":         p["prompt"],
            "model_id":            "claude_haiku_batch",
            "system_instructions": SYSTEM_PROMPT,
            "example_messages":    p.get("example_messages"),
        }
        for p in prompts
    ]
    log.info(f"[claude] Submitting {len(requests):,} requests via Vertex batch ...")
    batch_results = await send_batch_messages(requests, poll_interval=poll_interval)
    return {k: v for k, v in batch_results.items()}


# ── GPT concurrent judge ──────────────────────────────────────────────────────
async def _run_gpt_judge(
    prompts: list[dict],
    tpm: int,
    rate: int,
) -> dict[str, str | None]:
    if not prompts:
        return {}
    tpm_limiter = _TPMRateLimiter(tpm)
    semaphore   = asyncio.Semaphore(rate)

    async def _call(p: dict) -> tuple[str, str | None]:
        estimated = count_tokens(SYSTEM_PROMPT + p["prompt"]) + sum(
            count_tokens(m["content"]) for m in (p.get("example_messages") or []))
        await tpm_limiter.wait(estimated)
        async with semaphore:
            try:
                text = await send_single_message(
                    user_prompt=p["prompt"],
                    system_instructions=SYSTEM_PROMPT,
                    model_id="gpt5_nano_sandbox",
                    backend="openai",
                    example_messages=p.get("example_messages"),
                )
                return str(p["idx"]), text
            except Exception as e:
                log.warning(f"[gpt] idx={p['idx']} failed: {e}")
                return str(p["idx"]), None

    log.info(f"[gpt] Sending {len(prompts):,} requests (tpm={tpm:,}, rate={rate}) ...")
    pairs = await asyncio.gather(*[_call(p) for p in prompts])
    results = dict(pairs)
    succeeded = sum(1 for v in results.values() if v is not None)
    log.info(f"[gpt] {succeeded:,}/{len(prompts):,} succeeded")
    return results


# ── Jury majority vote ────────────────────────────────────────────────────────
def _jury_vote(judgments: list[dict | None], model_a: str = "", model_b: str = "") -> dict | None:
    """Majority vote across up to 3 juror judgments. Returns None if all failed."""
    valid = [j for j in judgments if j is not None]
    if not valid:
        return None
    result: dict = {}
    for dim in DIMS:
        counts: dict[str, int] = {"A": 0, "B": 0, "TIE": 0}
        for j in valid:
            counts[j.get(f"{dim}_winner", "TIE")] += 1
        winner = max(counts, key=lambda k: counts[k])
        top = max(counts.values())
        if sum(1 for v in counts.values() if v == top) > 1:
            winner = "TIE"
        result[f"{dim}_winner"]       = winner
        result[f"{dim}_winner_model"] = model_a if winner == "A" else (model_b if winner == "B" else "tie")
        result[f"{dim}_explanation"]  = " | ".join(
            j.get(f"{dim}_explanation", "") for j in valid
        )
    counts_ov: dict[str, int] = {"A": 0, "B": 0, "TIE": 0}
    for dim in DIMS:
        counts_ov[result[f"{dim}_winner"]] += 1
    if counts_ov["A"] >= 2:
        result["overall_winner"]       = "A"
        result["overall_winner_model"] = model_a
    elif counts_ov["B"] >= 2:
        result["overall_winner"]       = "B"
        result["overall_winner_model"] = model_b
    else:
        result["overall_winner"]       = "TIE"
        result["overall_winner_model"] = "tie"
    return result


# ── ELO calculation ───────────────────────────────────────────────────────────
def compute_elo(
    results_df: pd.DataFrame,
    k: float = 32.0,
    initial: float = 1000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute ELO ratings from pairwise results.

    Uses both display orders. In every stored row, model_a/model_b already match
    response A/B, so winner labels must not be flipped a second time. Rows are
    shuffled before processing to reduce Elo ordering effects.
    """
    models: set[str] = set(results_df["model_a"]) | set(results_df["model_b"])
    ratings: dict[str, dict[str, float]] = {
        m: {d: initial for d in (*DIMS, "overall")} for m in models
    }
    wins   = defaultdict(int)
    losses = defaultdict(int)
    ties   = defaultdict(int)

    rows = results_df.sample(frac=1, random_state=seed).to_dict("records")
    for row in rows:
        ma = str(row["model_a"])
        mb = str(row["model_b"])
        for dim in (*DIMS, "overall"):
            col = f"{dim}_winner" if dim != "overall" else "overall_winner"
            raw_winner = str(row.get(col, "TIE")).upper()

            ra = ratings[ma][dim]
            rb = ratings[mb][dim]
            ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
            eb = 1.0 - ea

            if raw_winner == "A":
                sa, sb = 1.0, 0.0
            elif raw_winner == "B":
                sa, sb = 0.0, 1.0
            else:
                sa, sb = 0.5, 0.5

            ratings[ma][dim] += k * (sa - ea)
            ratings[mb][dim] += k * (sb - eb)

        # Count win/loss/tie using the model identities stored for this display order.
        ov = str(row.get("overall_winner", "TIE")).upper()
        if ov == "A":
            wins[ma] += 1
            losses[mb] += 1
        elif ov == "B":
            wins[mb] += 1
            losses[ma] += 1
        else:
            ties[ma] += 1
            ties[mb] += 1

    records = []
    for m in sorted(models):
        total = wins[m] + losses[m] + ties[m]
        records.append({
            "model":             m,
            "elo_completeness":  round(ratings[m]["completeness"], 1),
            "elo_relevancy":     round(ratings[m]["relevancy"],    1),
            "elo_concision":     round(ratings[m]["concision"],    1),
            "elo_overall":       round(ratings[m]["overall"],      1),
            "wins":              wins[m],
            "losses":            losses[m],
            "ties":              ties[m],
            "total":             total,
            "win_rate":          round(wins[m] / total, 6) if total else 0.0,
            "non_tie_win_rate":  round(wins[m] / (wins[m] + losses[m]), 6)
                                   if wins[m] + losses[m] else 0.0,
        })
    return pd.DataFrame(records).sort_values("elo_overall", ascending=False)


# ── Selective rerun ───────────────────────────────────────────────────────────
def drop_pairs_for_models(path: str, force_models: set[str]) -> int:
    """Atomically rewrite the pairwise CSV, removing rows where either side's model
    is in force_models. Models are matched on their base name (the part before the
    ``__tag`` suffix added by --response-tags), so e.g. force_models={"gemini_pro"}
    drops both raw ``gemini_pro`` rows (per-method ELO) and ``gemini_pro__recent``
    rows (per-model ELO). Returns the number of rows removed.
    """
    if not force_models or not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    df = pd.read_csv(path, dtype=str)
    if df.empty or not {"model_a", "model_b"}.issubset(df.columns):
        return 0
    base_a = df["model_a"].astype(str).str.split("__").str[0]
    base_b = df["model_b"].astype(str).str.split("__").str[0]
    mask = base_a.isin(force_models) | base_b.isin(force_models)
    removed = int(mask.sum())
    if removed:
        tmp = path + ".tmp"
        df[~mask].to_csv(tmp, index=False)
        os.replace(tmp, path)  # atomic
    return removed


# ── Few-shot examples ─────────────────────────────────────────────────────────
def load_examples(path: str | None) -> list[dict]:
    """Load ELO few-shot examples (from build_elo_examples.py): a flat list of
    {role, content} turns prepended to every pairwise request. [] when no path."""
    if not path:
        return []
    with open(path, encoding="utf-8") as f:
        turns = json.load(f)
    turns = [{"role": t["role"], "content": t["content"]} for t in turns]
    log.info(f"Loaded {len(turns)//2} few-shot example pairs from {path}")
    return turns


# ── Main ──────────────────────────────────────────────────────────────────────
async def main(args):
    # ── Load inputs ───────────────────────────────────────────────────────────
    questions = pd.read_csv(args.questions, dtype=str)
    log.info(f"Loaded {len(questions)} questions from {args.questions}")

    tags = args.response_tags or []
    if tags and len(tags) != len(args.responses):
        log.error(
            f"--response-tags has {len(tags)} values but --responses has {len(args.responses)} files"
        )
        return

    frames = []
    for i, path in enumerate(args.responses):
        df = pd.read_csv(path, dtype=str, on_bad_lines="warn")
        df["model"] = df["model"].str.replace(r"_(batch|sandbox)$", "", regex=True)
        if args.select_model:
            df = df[df["model"].isin(args.select_model)]
            log.info(f"{path}: kept {len(df)} rows for models {args.select_model}")
        if tags:
            df["model"] = df["model"] + "__" + tags[i]
            log.info(f"{path}: tagged models as *__{tags[i]}")
        frames.append(df)
    responses = pd.concat(frames, ignore_index=True)

    required = {"question_id", "model", "response"}
    missing  = required - set(responses.columns)
    if missing:
        log.error(f"Response CSV(s) missing columns: {missing}")
        return

    log.info(
        f"Loaded {len(responses)} response rows across "
        f"{responses['model'].nunique()} models: {sorted(responses['model'].unique())}"
    )

    # ── Generate pairs ─────────────────────────────────────────────────────────
    all_rows = build_pairwise_rows(responses, questions)
    if not all_rows:
        log.error("No pairwise rows generated — check that question_ids match.")
        return

    judges = ["gemini", "claude", "gpt"] if args.judge == "jury" else [args.judge]

    examples = load_examples(args.examples)
    example_tokens = sum(count_tokens(m["content"]) for m in examples)

    # ── Cost estimate ──────────────────────────────────────────────────────────
    if args.force_model:
        removed = drop_pairs_for_models(args.output, set(args.force_model))
        log.info(f"--force-model: dropped {removed} stale pair row(s) involving {args.force_model} from {args.output}")
    completed = load_completed_pairs(
        args.output, ["question_id", "model_a", "model_b", "position"],
        nonempty_col="overall_winner",
    )
    pending_rows = [
        r for r in all_rows
        if (r["question_id"], r["model_a"], r["model_b"], r["position"]) not in completed
    ]
    log.info(
        f"{len(pending_rows)} rows to judge "
        f"({len(all_rows) - len(pending_rows)} already done)"
    )
    log_cost_estimate(pending_rows, judges, example_tokens)

    if args.dry_run:
        log.info("[DRY RUN] Stopping before submission.")
        return

    if not pending_rows:
        log.info("Nothing to submit — recomputing ELO over existing results.")
    else:
        # ── Build prompt list ──────────────────────────────────────────────────
        prompts = [
            {
                "idx": i,
                "prompt": PAIRWISE_PROMPT.format(
                    QUESTION=r["question"],
                    REFERENCE=r["reference"],
                    RESPONSE_A=r["response_a"],
                    RESPONSE_B=r["response_b"],
                ),
                "example_messages": examples,
            }
            for i, r in enumerate(pending_rows)
        ]

        # ── Run judge(s) ───────────────────────────────────────────────────────
        if args.judge == "jury":
            log.info("Running jury (gemini + claude + gpt) concurrently ...")
            gemini_raw, claude_raw, gpt_raw = await asyncio.gather(
                _run_gemini_judge(prompts, args.gcs_location, args.poll_interval),
                _run_claude_judge(prompts, args.poll_interval),
                _run_gpt_judge(prompts, args.gpt_tpm, args.rate),
            )
            raw_by_judge = {"gemini": gemini_raw, "claude": claude_raw, "gpt": gpt_raw}
        elif args.judge == "gemini":
            raw = await _run_gemini_judge(prompts, args.gcs_location, args.poll_interval)
            raw_by_judge = {"gemini": raw}
        elif args.judge == "claude":
            raw = await _run_claude_judge(prompts, args.poll_interval)
            raw_by_judge = {"claude": raw}
        else:  # gpt
            raw = await _run_gpt_judge(prompts, args.gpt_tpm, args.rate)
            raw_by_judge = {"gpt": raw}

        # ── Parse and write results ────────────────────────────────────────────
        writer    = CsvWriter(args.output, OUTPUT_FIELDS)
        succeeded = 0
        for i, row in enumerate(pending_rows):
            idx_str = str(i)
            ma, mb = row["model_a"], row["model_b"]
            if args.judge == "jury":
                judgments = [
                    _parse_judgment(raw_by_judge[j].get(idx_str), ma, mb)
                    for j in ["gemini", "claude", "gpt"]
                ]
                judgment = _jury_vote(judgments, ma, mb)
                judge_label = "jury"
            else:
                judgment    = _parse_judgment(raw_by_judge[args.judge].get(idx_str), ma, mb)
                judge_label = args.judge

            if judgment is None:
                log.warning(
                    f"Failed to parse judgment for question_id={row['question_id']} "
                    f"model_a={row['model_a']} model_b={row['model_b']} pos={row['position']}"
                )
                continue

            entry = {
                "question_id": row["question_id"],
                "model_a":     row["model_a"],
                "model_b":     row["model_b"],
                "position":    row["position"],
                "judge":       judge_label,
                **judgment,
            }
            await writer.write(entry)
            succeeded += 1

        log.info(f"Written {succeeded}/{len(pending_rows)} pairwise judgments to {args.output}")

    # ── Compute ELO ───────────────────────────────────────────────────────────
    if not os.path.exists(args.output) or os.path.getsize(args.output) == 0:
        log.warning("No results to compute ELO from.")
        return

    results_df = pd.read_csv(args.output, dtype=str)
    elo_df     = compute_elo(results_df)
    log.info("\n" + elo_df.to_string(index=False))

    if args.elo:
        elo_df.to_csv(args.elo, index=False)
        log.info(f"ELO summary written to {args.elo}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairwise ELO evaluation via LLM judge")
    parser.add_argument("-q", "--questions",    required=True,
                        help="Questions CSV (question_id, question/natural_query, answer/annotation_sub_answer)")
    parser.add_argument("-r", "--responses",    required=True, nargs="+",
                        help="Response CSV(s) (question_id, model, response). Multiple files accepted.")
    parser.add_argument("-o", "--output",       required=True,
                        help="Pairwise results CSV (one row per ordered pair × question)")
    parser.add_argument("--elo",                default=None,
                        help="ELO summary CSV (one row per model)")
    parser.add_argument("--select-model",       nargs="+", default=None,
                        help="Filter response CSV(s) to these model names (after stripping _batch/_sandbox)")
    parser.add_argument("--response-tags",      nargs="+", default=None,
                        help="One tag per --responses file; appended to model name as model__tag "
                             "(e.g. claude_haiku__recent). Must match number of --responses files.")
    parser.add_argument("--judge",              choices=["gemini", "claude", "gpt", "jury"],
                        default=DEFAULT_JUDGE,
                        help="Judge model (default: gemini). 'jury' = all three with majority vote.")
    parser.add_argument("--gcs-location",       default=None,
                        help="GCS prefix for Gemini batch I/O (required unless --judge is claude or gpt)")
    parser.add_argument("--poll-interval",      type=int, default=60,
                        help="Seconds between batch job status polls (default: 60)")
    parser.add_argument("--gpt-tpm",            type=int, default=5_000_000,
                        help="GPT nano token-per-minute limit (default: 5000000)")
    parser.add_argument("--rate",               type=int, default=20,
                        help="Max concurrent GPT nano requests (default: 20)")
    parser.add_argument("--elo-k",              type=float, default=32.0,
                        help="ELO K-factor (default: 32)")
    parser.add_argument("--elo-seed",           type=int, default=42,
                        help="Random seed for ELO row shuffle (default: 42)")
    parser.add_argument("--dry-run",            action="store_true",
                        help="Estimate request count and cost without submitting any jobs")
    parser.add_argument("--examples",           default=None,
                        help="Few-shot examples JSON from build_elo_examples.py; prepended as "
                             "multi-turn human-validated demonstrations to every pairwise request.")
    parser.add_argument("--force-model",        nargs="+", default=None, metavar="MODEL",
                        help="Drop existing pairwise rows where either model (base name before the "
                             "__tag suffix) is in this list, forcing those pairs to be re-judged. Use "
                             "when a model's responses changed (e.g. gemini_pro gemini_flash). "
                             "May also be supplied as a whitespace-separated ELO_FORCE_MODELS "
                             "environment variable for Slurm wrapper submissions.")
    args = parser.parse_args()

    env_force_models = os.environ.get("ELO_FORCE_MODELS", "").split()
    if env_force_models:
        args.force_model = sorted(set((args.force_model or []) + env_force_models))

    if args.judge in ("gemini", "jury") and not args.gcs_location and not args.dry_run:
        parser.error("--gcs-location is required when --judge is gemini or jury")

    asyncio.run(main(args))
