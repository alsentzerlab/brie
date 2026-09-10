#!/usr/bin/env python
"""Build a judge --pairs CSV from the unblinding map the humans were actually shown.

The map written by ``sample_elo_pairs.py`` is the authoritative record of what went
into a blinded annotation HTML: one record per shown pair, carrying identity
(question_id, model_a/model_b, position, source_file) AND the exact response text
(response_a / response_b, plus the question and reference answer).

Judging must run on those same pairs. Re-using an unrelated pool (e.g. a
``pairs_filtered.csv`` sampled independently) yields predictions that share no join
key with the human truth CSV, so every rater-vs-human cell comes out empty. This
script converts the map into the pairs schema that ``flip_elo_pairs.py`` and
``judge_elo_pairs.py`` consume, so alignment holds by construction:

    map record                 -> pairs column
    ---------------------------------------------
    Path(source_file).stem     -> source
    question_id                -> question_id
    model_a / model_b          -> model_a / model_b
    position                   -> position
    natural_query / question   -> natural_query / question
    reference_answer           -> reference_answer
    response_a / response_b    -> model_a_response / model_b_response

``source`` is the *stem* of source_file, matching what
``merge_elo_annotations.write_truth`` puts in the human truth CSV -- the two must
agree or the five-column join key (question_id, model_a, model_b, position, source)
will never match.

Usage
-----
    python pairs_from_map.py $EVAL_DIR/elo/annotate_pooled.map.json \
        -o $EVAL_DIR/elo/pairs_human_pool.csv

    # diagnose why an existing pairs CSV doesn't line up with the map
    python pairs_from_map.py $EVAL_DIR/elo/annotate_pooled.map.json \
        --compare $EVAL_DIR/elo/pairs_filtered.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

OUT_COLS = ["source", "question_id", "model_a", "model_b", "position",
            "natural_query", "question", "reference_answer",
            "model_a_response", "model_b_response"]
KEY_COLS = ["question_id", "model_a", "model_b", "position", "source"]


def map_to_pairs(amap: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for rec in amap.values():
        rows.append({
            "source": Path(str(rec.get("source_file", ""))).stem,
            "question_id": str(rec.get("question_id", "")),
            "model_a": str(rec.get("model_a", "")),
            "model_b": str(rec.get("model_b", "")),
            "position": str(rec.get("position", "")),
            "natural_query": str(rec.get("natural_query", "") or ""),
            "question": str(rec.get("question", "") or ""),
            "reference_answer": str(rec.get("reference_answer", "") or ""),
            "model_a_response": str(rec.get("response_a", "") or ""),
            "model_b_response": str(rec.get("response_b", "") or ""),
        })
    return pd.DataFrame(rows, columns=OUT_COLS)


def report_blanks(df: pd.DataFrame) -> None:
    """A pair with no response text produces a meaningless judgment."""
    for col in ("model_a_response", "model_b_response", "reference_answer"):
        blank = int((df[col].astype(str).str.strip() == "").sum())
        if blank:
            print(f"  WARNING: {blank}/{len(df)} rows have an empty {col}", file=sys.stderr)


def compare(df: pd.DataFrame, other_path: Path) -> None:
    """Locate which key field makes an existing pairs CSV diverge from the map."""
    other = pd.read_csv(other_path, dtype=str).fillna("")
    print(f"\n=== Map vs {other_path.name} ({len(df)} vs {len(other)} rows) ===")
    missing = [c for c in KEY_COLS if c not in other.columns]
    if missing:
        print(f"  {other_path.name} is missing key column(s): {missing}")

    levels = [
        ("question_id only", ["question_id"]),
        ("+ model pair (unordered)", ["question_id", "_lo", "_hi"]),
        ("+ source", ["question_id", "_lo", "_hi", "source"]),
        ("full key (+ position)", KEY_COLS),
    ]
    for frame in (df, other):
        if {"model_a", "model_b"}.issubset(frame.columns):
            frame["_lo"] = frame[["model_a", "model_b"]].min(axis=1)
            frame["_hi"] = frame[["model_a", "model_b"]].max(axis=1)

    for name, cols in levels:
        if not all(c in df.columns and c in other.columns for c in cols):
            print(f"  {name:<26} n/a (column absent)")
            continue
        a = {tuple(str(v) for v in row) for row in df[cols].itertuples(index=False)}
        b = {tuple(str(v) for v in row) for row in other[cols].itertuples(index=False)}
        print(f"  {name:<26} {len(a & b):>5} shared   "
              f"{len(a - b):>5} map-only   {len(b - a):>5} other-only")

    for col in ("source", "position"):
        if col in other.columns:
            print(f"  distinct {col}: map={sorted(set(df[col]))[:6]} "
                  f"other={sorted(set(other[col].astype(str)))[:6]}")
    print("\n  Read the first level where 'shared' collapses to 0 -- that is the field "
          "that diverges.\n  0 shared at 'question_id only' means the two pools are "
          "disjoint samples, and the\n  map is the one to judge on.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map", help="annotate_*.map.json from sample_elo_pairs.py")
    ap.add_argument("-o", "--output", help="Pairs CSV to write (omit to only diagnose)")
    ap.add_argument("--compare", help="Existing pairs CSV to diff against the map")
    args = ap.parse_args()

    with open(args.map, encoding="utf-8") as f:
        amap = json.load(f)
    if not isinstance(amap, dict):
        sys.exit("map must be a JSON object keyed by record_id.")
    print(f"Loaded map with {len(amap)} records from {args.map}")

    df = map_to_pairs(amap)
    dupes = int(df.duplicated(subset=KEY_COLS).sum())
    if dupes:
        print(f"  WARNING: {dupes} rows share a join key with another row", file=sys.stderr)
    report_blanks(df)

    if args.compare:
        compare(df, Path(args.compare))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nWrote {len(df)} pairs -> {out}")
        print(f"Next: python flip_elo_pairs.py {out} -o {out.with_name(out.stem + '_flipped.csv')}")
    elif not args.compare:
        sys.exit("Nothing to do: pass -o to write pairs or --compare to diagnose.")


if __name__ == "__main__":
    main()
