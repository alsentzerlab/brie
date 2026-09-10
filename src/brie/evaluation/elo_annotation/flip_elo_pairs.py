#!/usr/bin/env python
"""Emit the missing presentation order(s) for each extracted ELO pair.

To measure (and collapse) judge position bias, every pair must be scored in BOTH
slot arrangements: the smaller model in slot A, and the larger model in slot A.
``extract_elo_responses.py`` usually emits just the single order the human saw.

This script reads an extracted pairs CSV and, for each canonical pair
(question_id + source + the unordered model pair), checks which arrangements are
already present and writes ONLY the missing complement(s) — created by swapping
an existing row's two sides:

    model_a            <-> model_b
    model_a_response   <-> model_b_response
    position           ab <-> ba   (any other value gets a '_flip' suffix)

Existing responses are reused verbatim (nothing is regenerated); only the slot
each response sits in changes. A pair that already has both orders is skipped,
so you never double-cover. Everything else (question_id, source, question text,
reference_answer) is left untouched, so (question_id, model_a, model_b, position,
source) stays a unique key distinct from the original row's key.

Run the output into the SAME --outdir as the original judging: the already-
scored rows are skipped as done and only the new order is submitted.

Usage
-----
    python flip_elo_pairs.py pairs_filtered.csv -o pairs_missing_order.csv

    # judge only the missing order into the same outdir (existing rows skipped):
    python judge_elo_pairs.py --pairs pairs_missing_order.csv --outdir elo_eval \
        --judges gemini-2.5-flash-lite ...

    # then compare with position collapse (orders now combine per pair):
    python compare_judge_accuracy.py --truth human.csv --pred-dir elo_eval

    # --all-flip mirrors EVERY row instead (old blanket behaviour)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SWAP_PAIRS = [("model_a", "model_b"), ("model_a_response", "model_b_response")]


def _flip_position(pos: str) -> str:
    p = pos.strip()
    low = p.lower()
    if low == "ab":
        return "ba"
    if low == "ba":
        return "ab"
    return f"{p}_flip" if p else "flip"


def _flip_row(row: pd.Series) -> dict:
    """Swap the two sides of one row, returning a new record dict."""
    rec = row.to_dict()
    for left, right in SWAP_PAIRS:
        if left in rec and right in rec:
            rec[left], rec[right] = rec[right], rec[left]
    if "position" in rec:
        rec["position"] = _flip_position(str(rec["position"]))
    return rec


def _canon(row: pd.Series) -> tuple:
    """Canonical pair key, independent of which model is shown in slot A."""
    src = (str(row["source"]),) if "source" in row.index else ()
    a, b = str(row["model_a"]), str(row["model_b"])
    return (str(row["question_id"]), *src, *tuple(sorted((a, b))))


def _orientation(row: pd.Series) -> str:
    a, b = str(row["model_a"]), str(row["model_b"])
    lo, _ = sorted((a, b))
    return "lo_in_a" if a == lo else "hi_in_a"


def complete(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the rows needed to give every pair both slot arrangements."""
    df = df.reset_index(drop=True)
    groups: dict[tuple, dict[str, int]] = {}
    for i in range(len(df)):
        row = df.iloc[i]
        groups.setdefault(_canon(row), {}).setdefault(_orientation(row), i)

    new_rows: list[dict] = []
    already_complete = 0
    for present in groups.values():
        if len(present) >= 2:           # both arrangements already judged
            already_complete += 1
            continue
        src_idx = next(iter(present.values()))
        new_rows.append(_flip_row(df.iloc[src_idx]))

    print(f"{len(groups)} canonical pairs: {already_complete} already had both "
          f"orders, {len(new_rows)} missing order(s) generated", file=sys.stderr)
    return pd.DataFrame(new_rows, columns=list(df.columns))


def flip_all(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror every row (blanket flip, ignores what's already present)."""
    return pd.DataFrame([_flip_row(r) for _, r in df.iterrows()],
                        columns=list(df.columns))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", help="Extracted pairs CSV from extract_elo_responses.py")
    ap.add_argument("-o", "--output", default="pairs_missing_order.csv",
                    help="Output CSV path (default: pairs_missing_order.csv)")
    ap.add_argument("--all-flip", action="store_true",
                    help="Mirror every row instead of only the missing complement.")
    args = ap.parse_args()

    df = pd.read_csv(args.pairs, dtype=str).fillna("")
    missing = {"question_id", "model_a", "model_b"} - set(df.columns)
    if missing:
        sys.exit(f"--pairs is missing required columns: {sorted(missing)}")

    out_df = flip_all(df) if args.all_flip else complete(df)
    out = Path(args.output)
    out_df.to_csv(out, index=False)
    print(f"Wrote {len(out_df)} rows to {out}")


if __name__ == "__main__":
    main()
