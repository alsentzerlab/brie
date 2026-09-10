'''
Sub-question fact-based precision/recall evaluation via entailment jury.

Like score_facts_batch.py, but the gold/reference facts are keyed by
sub_question_id ("{question_id}_{idx}") while the candidate facts are keyed by
question_id. Each sub-question reference is scored against the candidate facts of
its *parent* question_id, so one candidate (question-level) answer is evaluated
once per constituent sub_question_id — i.e. every (sub_question_id reference,
question_id candidate) combination.

Jurors are selectable (--jurors): any subset of gemini (Vertex GCS batch,
gemini-3.1-flash-lite), claude (Vertex Claude batch, claude_haiku), gpt
(concurrent OpenAI-compatible calls). avg/consensus are computed over the selected
jurors only.

NOTE on precision: a candidate answers the whole question, so candidate facts
that belong to *sibling* sub-questions are (correctly) counted as not entailed
by a single sub-question's reference. Per-sub-question precision therefore reads
low by design — recall is the primary signal. This is intentional, not a bug.

Inputs:
  --gold-facts  atomic facts CSV keyed by sub_question_id; gold rows have
                source_name == --gold-source.
  --pred-facts  atomic facts CSV keyed by question_id; candidate rows have
                source_name != --gold-source.
Output: one row per (sub_question_id, source_name, model).
'''

import argparse
import asyncio
import json
import logging
import sys

import pandas as pd

from .utils import (  # type: ignore[reportAttributeAccessIssue]
    CsvWriter,
    count_tokens,
    drop_rows_for_ids,
    load_completed_pairs,
    load_force_ids,
    log_token_stats,
)
from .score_facts_batch import (  # reuse prompts, runners, and scoring helpers
    SYSTEM_PROMPT,
    RECALL_PROMPT,
    PRECISION_PROMPT,
    JURORS,
    _JUROR_COSTS,
    _parse_facts,
    _organize,
    _score,
    _consensus,
    load_examples,
    _example_tokens,
    run_gemini_batch,
    run_claude_batch,
    run_gpt_batch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── Prompt building (sub_question_id-aware comparison key) ────────────────────
def build_prompts(comparisons: list[tuple], examples_by_dir: dict | None = None) -> list[dict]:
    """Two prompts per comparison (recall + precision). comparison_key carries the
    sub_question_id so each sub-question is scored independently against its
    parent question_id's candidate facts."""
    examples_by_dir = examples_by_dir or {}
    prompts: list[dict] = []
    for sub_id, _qid, source, model, ref_facts, cand_facts in comparisons:
        key           = (sub_id, source, model)
        ref_numbered  = "\n".join(f"{i}. {f}" for i, f in enumerate(ref_facts))
        cand_numbered = "\n".join(f"{i}. {f}" for i, f in enumerate(cand_facts))
        prompts.append({
            "direction":        "r",
            "comparison_key":   key,
            "user_prompt":      RECALL_PROMPT.format(
                REFERENCE_FACTS=ref_numbered, CANDIDATE_FACTS=cand_numbered),
            "example_messages": examples_by_dir.get("r", []),
        })
        prompts.append({
            "direction":        "p",
            "comparison_key":   key,
            "user_prompt":      PRECISION_PROMPT.format(
                REFERENCE_FACTS=ref_numbered, CANDIDATE_FACTS=cand_numbered),
            "example_messages": examples_by_dir.get("p", []),
        })
    for i, p in enumerate(prompts):
        p["idx"] = i
    return prompts


def log_token_estimate(prompts: list[dict], jurors: list[str]) -> None:
    input_tokens  = sum(count_tokens(SYSTEM_PROMPT + p["user_prompt"]) + _example_tokens(p) for p in prompts)
    output_tokens = len(prompts) * 5
    total_cost = sum(
        input_tokens / 1e6 * _JUROR_COSTS[j][0] + output_tokens / 1e6 * _JUROR_COSTS[j][1]
        for j in jurors
    )
    log.info(
        f"Token estimate per juror: {input_tokens:,} input + {output_tokens:,} output "
        f"({len(prompts):,} calls) — jurors {jurors} estimated cost: ${total_cost:.2f}"
    )


def _output_fields(jurors: list[str]) -> list[str]:
    fields = ["sub_question_id", "question_id", "source_name", "model"]
    for j in jurors:
        fields += [f"{j}_precision", f"{j}_recall",
                   f"{j}_entailed_ref_facts", f"{j}_entailed_cand_facts"]
    fields += ["avg_precision", "avg_recall",
               "consensus_precision", "consensus_recall",
               "consensus_entailed_ref_facts", "consensus_entailed_cand_facts"]
    return fields


# ── Main ──────────────────────────────────────────────────────────────────────
async def main(args):
    jurors = args.jurors
    bad = [j for j in jurors if j not in JURORS]
    if bad:
        log.error(f"Unknown juror(s) {bad}; choose from {JURORS}")
        return
    threshold = args.consensus_threshold or (len(jurors) // 2 + 1)
    output_fields = _output_fields(jurors)

    force_ids = load_force_ids(args.force_ids, args.force_ids_file)

    # ── Gold: sub_question_id → facts ─────────────────────────────────────────
    gold_df = pd.read_csv(args.gold_facts, dtype=str)
    gold_df = gold_df[gold_df["source_name"] == args.gold_source]
    gold_lookup: dict[str, list[str]] = {}
    for _, row in gold_df.iterrows():
        facts = _parse_facts(row.get("facts_atomic"))
        if facts:
            gold_lookup[str(row["sub_question_id"])] = facts
    log.info(f"Gold: {len(gold_lookup)} sub-questions with facts from {args.gold_facts}")

    # ── Candidates: question_id → list of (source, model, facts) ──────────────
    pred_df = pd.read_csv(args.pred_facts, dtype=str)
    pred_df = pred_df[pred_df["source_name"] != args.gold_source]
    if args.source:
        pred_df = pred_df[pred_df["source_name"] == args.source]
    if args.model:
        if "model" in pred_df.columns:
            pred_df = pred_df[pred_df["model"].fillna("") == args.model]
        else:
            log.warning("--model given but pred-facts has no 'model' column; no candidates match")
            pred_df = pred_df.iloc[0:0]
    log.info(
        f"Candidates: {len(pred_df)} rows from {args.pred_facts}"
        + (f" | source={args.source}" if args.source else "")
        + (f" | model={args.model}" if args.model else "")
    )

    cand_by_qid: dict[str, list[tuple[str, str, list[str]]]] = {}
    for _, row in pred_df.iterrows():
        qid    = str(row["question_id"])
        source = str(row["source_name"])
        model  = str(row.get("model", "") or "")
        facts  = _parse_facts(row.get("facts_atomic"))
        if facts:
            cand_by_qid.setdefault(qid, []).append((source, model, facts))

    # ── Resume / force ────────────────────────────────────────────────────────
    if force_ids:
        removed = drop_rows_for_ids(args.output, "sub_question_id", force_ids)
        log.info(f"--force-ids: dropped {removed} stale row(s) for {len(force_ids)} sub_question_id(s)")
    if args.force_source:
        removed = drop_rows_for_ids(args.output, "source_name", set(args.force_source))
        log.info(f"--force-source: dropped {removed} stale row(s) for {args.force_source}")
    completed = load_completed_pairs(
        args.output, ["sub_question_id", "source_name", "model"], nonempty_col="avg_recall"
    )
    if completed:
        log.info(f"Resuming: {len(completed)} comparisons already done")

    # ── Build comparisons: each sub_question_id vs its parent question_id ──────
    comparisons: list[tuple] = []
    skipped_done = skipped_missing = 0
    for sub_id, ref_facts in gold_lookup.items():
        if force_ids and sub_id not in force_ids:
            continue
        qid = sub_id.rsplit("_", 1)[0]
        candidates = cand_by_qid.get(qid, [])
        if not candidates:
            skipped_missing += 1
            continue
        for source, model, cand_facts in candidates:
            if (sub_id, source, model) in completed:
                skipped_done += 1
                continue
            comparisons.append((sub_id, qid, source, model, ref_facts, cand_facts))

    log.info(
        f"{len(comparisons)} comparisons to run "
        f"({skipped_done} already done, {skipped_missing} sub-questions with no candidate)"
    )
    if not comparisons:
        log.info("Nothing to do.")
        return

    examples_by_dir = load_examples(args.examples)
    prompts = build_prompts(comparisons, examples_by_dir)
    log.info(f"Built {len(prompts):,} entailment prompts")
    log_token_estimate(prompts, jurors)

    if args.dry_run:
        log.info("[DRY RUN] Stopping before submission. No batch jobs submitted.")
        return

    # ── Run selected jurors concurrently ──────────────────────────────────────
    async def _empty() -> dict:
        return {}

    log.info(f"Submitting jurors {jurors} ...")
    gemini_raw, claude_raw, gpt_raw = await asyncio.gather(
        run_gemini_batch(prompts, args.gcs_location, args.poll_interval) if "gemini" in jurors else _empty(),
        run_claude_batch(prompts, args.poll_interval)                    if "claude" in jurors else _empty(),
        run_gpt_batch(prompts, args.gpt_tpm, args.rate)                  if "gpt"    in jurors else _empty(),
    )
    org = {
        "gemini": _organize(prompts, gemini_raw),
        "claude": _organize(prompts, claude_raw),
        "gpt":    _organize(prompts, gpt_raw),
    }

    # ── Score and write ───────────────────────────────────────────────────────
    writer = CsvWriter(args.output, output_fields)
    succeeded = 0
    for sub_id, qid, source, model, ref_facts, cand_facts in comparisons:
        key   = (sub_id, source, model)
        entry = {"sub_question_id": sub_id, "question_id": qid,
                 "source_name": source, "model": model}
        precisions, recalls = [], []
        all_r = {j: org[j].get(key, {"r": None, "p": None})["r"] for j in jurors}
        all_p = {j: org[j].get(key, {"r": None, "p": None})["p"] for j in jurors}

        for j in jurors:
            dirs = org[j].get(key, {"r": None, "p": None})
            prec, rec, ent_cand, ent_ref = _score(ref_facts, cand_facts, dirs["r"], dirs["p"])
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

        c_prec, c_rec, c_cand, c_ref = _consensus(ref_facts, cand_facts, all_r, all_p, threshold)
        entry.update({
            "consensus_precision":           round(c_prec, 4),
            "consensus_recall":              round(c_rec,  4),
            "consensus_entailed_ref_facts":  json.dumps(c_ref),
            "consensus_entailed_cand_facts": json.dumps(c_cand),
        })

        await writer.write(entry)
        succeeded += 1

    log.info(f"Done. {succeeded}/{len(comparisons)} comparisons written to {args.output}")
    log_token_stats(log, prefix=f"[{'+'.join(jurors)}] ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sub-question fact precision/recall via entailment jury "
                    "(sub_question_id reference vs question_id candidate)"
    )
    parser.add_argument("--gold-facts", required=True,
                        help="Atomic facts CSV keyed by sub_question_id (gold = --gold-source rows)")
    parser.add_argument("--pred-facts", required=True,
                        help="Atomic facts CSV keyed by question_id (candidate rows)")
    parser.add_argument("--gcs-location", required=True,
                        help="GCS prefix for Gemini batch I/O (e.g. gs://bucket/entailment)")
    parser.add_argument("-o", "--output", required=True, help="Output CSV")
    parser.add_argument("--gold-source", default="reference",
                        help="source_name value for gold/reference facts (default: reference)")
    parser.add_argument("--jurors", nargs="+", default=list(JURORS),
                        choices=list(JURORS),
                        help=f"Subset of jurors to run (default: all {JURORS})")
    parser.add_argument("--source", default=None,
                        help="Restrict candidates to this source_name")
    parser.add_argument("--model", default=None,
                        help="Restrict candidates to this model")
    parser.add_argument("--consensus-threshold", type=int, default=None,
                        help="Jurors that must agree for consensus (default: majority of selected)")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between batch job status checks (default: 60)")
    parser.add_argument("--gpt-tpm", type=int, default=5_000_000,
                        help="Online juror tokens-per-minute limit")
    parser.add_argument("--rate", type=int, default=20,
                        help="Max concurrent GPT nano requests (default: 20)")
    parser.add_argument("--examples", default=None,
                        help="Few-shot examples JSON from build_fact_examples.py")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate token count and cost without submitting any batch jobs")
    parser.add_argument("--force-ids", nargs="+", default=None, metavar="SUB_QUESTION_ID",
                        help="Re-score only these sub_question_ids, dropping their stale output rows first")
    parser.add_argument("--force-ids-file", type=str, default=None,
                        help="File with one sub_question_id per line to re-score (unioned with --force-ids)")
    parser.add_argument("--force-source", nargs="+", default=[], metavar="SOURCE_NAME",
                        help="Re-score these source_name(s), dropping their stale output rows first")
    args = parser.parse_args()
    asyncio.run(main(args))
