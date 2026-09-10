#!/usr/bin/env python
"""Re-judge human-annotated ELO pairs with multiple LLM judges (batch inference).

Takes the CSV produced by ``extract_elo_responses.py`` (one row per
human-reviewed pair, with the responses shown in a fixed A/B order) and asks
each of several judge models to pick the winner per dimension. Because the A/B
order is exactly what the human saw, the judge's A/B labels line up with the
human ground-truth labels, so no position flipping is done here.

Judge aliases and provider configuration are selected at runtime.

Outputs (under --outdir):
  predictions/{judge}.csv   exact schema: question_id, model_a, model_b,
                            position, source, completeness, relevancy, concision
                            (values A / B / TIE)
  responses/{judge}.jsonl   raw judge text + parsed winners/explanations per pair

Resumable: pairs already present in predictions/{judge}.csv are skipped.

Usage
-----
    python judge_elo_pairs.py --pairs extracted.csv --outdir judge_eval
    python judge_elo_pairs.py --pairs extracted.csv --judges gemini-3.5-flash \
        claude-opus-4-7 --dry-run

Resume an already-submitted job (no re-submission / re-billing) — pass the GCS
output prefix, the full job resource name, or the bare job id printed in the
"Uploading ... predictions/<id>/input.jsonl" log line:

    python judge_elo_pairs.py --pairs extracted.csv --outdir judge_eval \
        --resume claude-haiku-4-5=<job-id> claude-sonnet-4-6=<job-id>
    # later, once opus finishes:
    python judge_elo_pairs.py --pairs extracted.csv --outdir judge_eval \
        --resume claude-opus-4-7=<job-id>

--resume and --judges combine: resumed judges are collected from their existing
jobs while any --judges not being resumed are submitted fresh, in one call:

    python judge_elo_pairs.py --pairs extracted.csv --outdir judge_eval \
        --judges gemini-2.5-flash-lite gemini-3.1-flash-lite gemini-3.5-flash \
        --resume claude-haiku-4-5=<job-id> claude-sonnet-4-6=<job-id>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from ..score_elo_batch import (
    DIMS,
    PAIRWISE_PROMPT,
    SYSTEM_PROMPT,
    _parse_judgment,
)
from ..utils import (
    VERTEX_CLAUDE_GCS_BUCKET,
    count_tokens,
    send_batch_messages,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Display name -> utils.py batch alias.
JUDGE_ALIASES: dict[str, str] = {
    "gemini-2.5-pro":        "gemini_pro_batch",
    "gemini-2.5-flash-lite": "gemini_flash_batch",
    "gemini-3.1-flash-lite": "gemini_flash_juror_batch",
    "gemini-3.5-flash":      "gemini_flash35_batch",
    "claude-opus-4-7":       "claude_opus_batch",
    "claude-sonnet-4-6":     "claude_sonnet_batch",
    "claude-haiku-4-5":      "claude_haiku_batch",
}

# Columns that identify a pair — the join key against the ground-truth file.
KEY_COLS = ["question_id", "model_a", "model_b", "position", "source"]
PRED_COLS = KEY_COLS + ["completeness", "relevancy", "concision"]

# custom_id = "{judge}|||{row_index}". '|||' cannot appear in a judge name.
_CID_SEP = "|||"


def _pair_key(row: dict | pd.Series) -> tuple:
    return tuple(str(row[c]) for c in KEY_COLS)


def _question_text(row: pd.Series) -> str:
    nq = str(row.get("natural_query", "") or "")
    return nq if nq else str(row.get("question", "") or "")


def load_done_keys(path: Path) -> set[tuple]:
    """Pair keys already judged (present with a non-empty completeness label)."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path, dtype=str).fillna("")
    if not set(KEY_COLS).issubset(df.columns):
        return set()
    if "completeness" in df.columns:
        df = df[df["completeness"].astype(str).str.strip() != ""]
    return {_pair_key(r) for _, r in df.iterrows()}


def build_prompt(row: pd.Series) -> str:
    return PAIRWISE_PROMPT.format(
        QUESTION=_question_text(row),
        REFERENCE=str(row.get("reference_answer", "") or ""),
        RESPONSE_A=str(row.get("model_a_response", "") or ""),
        RESPONSE_B=str(row.get("model_b_response", "") or ""),
    )


def append_rows(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Append rows to a CSV, writing a header only when the file is new/empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    pd.DataFrame(rows, columns=columns).to_csv(
        path, mode="a", header=write_header, index=False
    )


def append_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── Resume from an already-submitted batch job ────────────────────────────────
_TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
             "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED"}


def _extract_text(line: dict) -> str | None:
    """Pull the response text from one batch result line (Claude or Gemini)."""
    if line.get("status"):  # non-empty status = per-request failure
        return None
    resp = line.get("response") or {}
    content = resp.get("content")  # Anthropic format: list of content blocks
    if isinstance(content, list):
        return next((b.get("text") for b in content if b.get("type") == "text"), None)
    try:  # Gemini format
        return resp["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return None


def _download_predictions(dest_uri: str) -> dict[str, str | None]:
    """Read predictions.jsonl shard(s) under a finished job's GCS output prefix."""
    import fsspec
    import pandas as pd

    fs = fsspec.filesystem("gcs")
    files = fs.glob(f"{dest_uri.rstrip('/')}/*/predictions.jsonl")
    if not files:
        # Some jobs write predictions.jsonl directly under the prefix.
        files = fs.glob(f"{dest_uri.rstrip('/')}/predictions.jsonl")
    if not files:
        raise FileNotFoundError(f"No predictions.jsonl found under {dest_uri}")
    out: dict[str, str | None] = {}
    for fp in files:
        shard = pd.read_json(f"gs://{fp}", lines=True)
        for line in shard.to_dict("records"):
            cid = line.get("custom_id")
            if cid is not None:
                out[str(cid)] = _extract_text(line)
    return out


def resume_job(spec: str, poll_interval: int) -> dict[str, str | None]:
    """Attach to an already-submitted batch job and return {custom_id: text}.

    ``spec`` may be:
      - a GCS output prefix:  gs://bucket/predictions/<id>/output
      - a full job resource:  projects/<proj>/locations/<loc>/batchPredictionJobs/<id>
      - a bare job id:        <hex>  (resolved to the Claude predictions bucket)
    GCS/bare-id forms download directly (job must be finished); the resource-name
    form polls to completion first.
    """
    import time

    from google import genai

    if spec.startswith("gs://"):
        log.info(f"Reading results from GCS output {spec}")
        return _download_predictions(spec)

    if spec.startswith("projects/"):
        parts = spec.split("/")
        project, location = parts[1], parts[3]
        client = genai.Client(vertexai=True, project=project, location=location)
        job = client.batches.get(name=spec)
        log.info(f"Attached to {spec}  state={job.state}")
        while job.state not in _TERMINAL:
            time.sleep(poll_interval)
            job = client.batches.get(name=spec)
            log.info(f"  state={job.state}")
        if job.state != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Job {spec} ended in {job.state}")
        return _download_predictions(job.dest.gcs_uri)  # type: ignore[union-attr]

    # Bare job id -> Claude predictions bucket layout used by send_batch_messages.
    dest = f"gs://{VERTEX_CLAUDE_GCS_BUCKET}/predictions/{spec}/output"
    log.info(f"Resolving bare job id {spec!r} -> {dest}")
    return _download_predictions(dest)


async def main(args: argparse.Namespace) -> None:
    pairs = pd.read_csv(args.pairs, dtype=str).fillna("")
    missing = set(KEY_COLS) - set(pairs.columns)
    if missing:
        sys.exit(f"--pairs is missing required columns: {sorted(missing)}")
    log.info(f"Loaded {len(pairs)} pairs from {args.pairs}")

    # Parse --resume "judge=SPEC" pairs, if any.
    resume_specs: dict[str, str] = {}
    for item in (args.resume or []):
        if "=" not in item:
            sys.exit(f"--resume expects judge=SPEC, got {item!r}")
        j, spec = item.split("=", 1)
        resume_specs[j.strip()] = spec.strip()

    # Judges to resume (attach to existing jobs) vs. submit (fresh batch jobs).
    # --judges selects which to submit; default is all when nothing is resumed,
    # else only what's explicitly requested. Resumed judges are always processed.
    if args.judges is not None:
        requested = args.judges
    elif resume_specs:
        requested = []
    else:
        requested = list(JUDGE_ALIASES)
    submit_judges = [j for j in requested if j not in resume_specs]
    resume_judges = list(resume_specs)
    judges = list(dict.fromkeys(resume_judges + submit_judges))  # unique, ordered

    unknown = [j for j in judges if j not in JUDGE_ALIASES]
    if unknown:
        sys.exit(f"Unknown judge(s): {unknown}. Choose from {list(JUDGE_ALIASES)}")
    if not judges:
        sys.exit("No judges to run — pass --judges and/or --resume.")

    outdir = Path(args.outdir)
    pred_dir = outdir / "predictions"
    resp_dir = outdir / "responses"

    # Build the pending request list per judge, demuxed by custom_id. The same
    # deterministic ordering is used to submit AND to map resumed results back,
    # so custom_id "{judge}|||{i}" indexes pending[judge][i] either way.
    requests: list[dict] = []  # only for submit_judges
    pending: dict[str, list] = {}  # judge -> row indices, aligned to custom_id index
    for judge in judges:
        alias = JUDGE_ALIASES[judge]
        done = load_done_keys(pred_dir / f"{judge}.csv")
        rows_idx = [i for i, r in pairs.iterrows() if _pair_key(r) not in done]
        pending[judge] = rows_idx
        mode = "resume" if judge in resume_specs else "submit"
        log.info(f"[{judge}] {len(rows_idx)} to judge ({len(pairs) - len(rows_idx)} already done) [{mode}]")
        if judge in resume_specs:
            continue
        for local_i, row_i in enumerate(rows_idx):
            requests.append({
                "custom_id":           f"{judge}{_CID_SEP}{local_i}",
                "user_prompt":         build_prompt(pairs.loc[row_i]),
                "model_id":            alias,
                "system_instructions": SYSTEM_PROMPT,
            })

    if not requests and not resume_specs:
        log.info("Nothing to judge — all pairs already done for these judges.")
        return

    raw: dict[str, str | None] = {}

    # Submit fresh batch jobs for the non-resumed judges.
    if requests:
        in_tokens = sum(count_tokens(SYSTEM_PROMPT + r["user_prompt"]) for r in requests)
        log.info(
            f"{len(requests):,} requests to submit across {len(submit_judges)} judge(s): "
            f"{submit_judges}; ~{in_tokens:,} input tokens "
            f"(+~{len(requests) * 150:,} output tokens est.)"
        )
        if args.dry_run:
            log.info("[DRY RUN] Not submitting.")
        else:
            # send_batch_messages groups by model_id and runs the Gemini (ehr
            # project) and Claude (bmds218 project) jobs concurrently.
            log.info("Submitting batch jobs ...")
            raw.update(await send_batch_messages(requests, poll_interval=args.poll_interval))

    if args.dry_run:
        return

    # Attach to already-submitted jobs and download results — no new submission.
    for judge, spec in resume_specs.items():
        try:
            raw.update(resume_job(spec, args.poll_interval))
        except Exception as e:
            log.error(f"[{judge}] resume failed for {spec!r}: {e}")

    # Demux results back per judge and write outputs.
    for judge in judges:
        alias = JUDGE_ALIASES[judge]
        rows_idx = pending[judge]
        pred_rows: list[dict] = []
        resp_records: list[dict] = []
        parsed_ok = 0
        for local_i, row_i in enumerate(rows_idx):
            cid = f"{judge}{_CID_SEP}{local_i}"
            row = pairs.loc[row_i]
            text = raw.get(cid)
            judgment = _parse_judgment(text, str(row["model_a"]), str(row["model_b"]))
            key_fields = {c: str(row[c]) for c in KEY_COLS}

            if judgment is None:
                log.warning(f"[{judge}] failed to parse judgment for {_pair_key(row)}")
                resp_records.append({**key_fields, "judge": judge, "model_alias": alias,
                                     "raw": text, "parsed": None})
                continue
            parsed_ok += 1
            pred_rows.append({
                **key_fields,
                "completeness": judgment["completeness_winner"],
                "relevancy":    judgment["relevancy_winner"],
                "concision":    judgment["concision_winner"],
            })
            resp_records.append({
                **key_fields, "judge": judge, "model_alias": alias, "raw": text,
                "parsed": {d: {"winner": judgment[f"{d}_winner"],
                               "explanation": judgment[f"{d}_explanation"]} for d in DIMS},
            })

        if pred_rows:
            append_rows(pred_dir / f"{judge}.csv", pred_rows, PRED_COLS)
        if resp_records:
            append_jsonl(resp_dir / f"{judge}.jsonl", resp_records)
        log.info(f"[{judge}] wrote {len(pred_rows)} predictions "
                 f"({parsed_ok}/{len(rows_idx)} parsed) -> {pred_dir / f'{judge}.csv'}")

    log.info(f"Done. Predictions in {pred_dir}, raw responses in {resp_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True,
                    help="Extracted pairs CSV from extract_elo_responses.py")
    ap.add_argument("--judges", nargs="+", default=None,
                    help=f"Subset of judges to run (default: all). Choices: {list(JUDGE_ALIASES)}")
    ap.add_argument("--outdir", default="judge_eval",
                    help="Output directory (default: judge_eval)")
    ap.add_argument("--poll-interval", type=int, default=60,
                    help="Seconds between batch status polls (default: 60)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Estimate request count / tokens without submitting")
    ap.add_argument("--resume", nargs="+", default=None, metavar="JUDGE=SPEC",
                    help="Resume from already-submitted batch job(s) instead of submitting. "
                         "One or more JUDGE=SPEC, where SPEC is a GCS output prefix "
                         "(gs://bucket/predictions/<id>/output), a full job resource name "
                         "(projects/<p>/locations/<l>/batchPredictionJobs/<id>), or a bare "
                         "job id (the hex in the 'Uploading ... predictions/<id>/input.jsonl' "
                         "log line). Only the named judges are processed.")
    asyncio.run(main(ap.parse_args()))
