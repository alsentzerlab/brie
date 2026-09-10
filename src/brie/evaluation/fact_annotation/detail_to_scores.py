#!/usr/bin/env python
"""
detail_to_scores.py

Turn the per-fact detail CSV written by ``eval_entailment_prompt.py --out`` into a
``score_facts_batch.py``-shaped scores CSV, so ``rater_agreement_matrix.py --scores``
can plot the judge you just measured without waiting on a full batch rescore.

Use this when you have tuned a prompt with the online eval loop and want the heatmap
now; run the real ``score_facts_batch.py`` when you need all three jurors, the
consensus columns, or scores for records nobody annotated.

The detail CSV carries one row per GRADED fact:
    record_id, question_id, source_name, direction, idx, fact, human, judge, agree
Rows exist only for facts the annotator labelled, so any unlabelled fact is absent
here and is emitted as "not entailed" — harmless for the agreement matrix, which
only scores facts the human marked, but it makes the precision/recall columns in
the output a lower bound. They are written for schema completeness; the matrix does
not read them.

The map supplies `model`, which the detail CSV does not carry, so the output keys on
the same (question_id, source_name, model) triple the matrix aligns by.

Usage:
  python detail_to_scores.py \
      --detail $FS/detail_50_jordan_v7.csv \
      --map    $FS/annotate.map.json \
      --output $FS/scores_50_v7_from_detail.csv

  python rater_agreement_matrix.py --map $FS/annotate.map.json \
      --scores $FS/scores_50_v7_from_detail.csv --jurors gemini \
      --human bridget=... --human jordan=... --no-combined --prefix ...
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

DIR_KEY = {"recall": "ref", "precision": "cand"}


def _truthy(v) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detail", nargs="+", required=True,
                    help="detail CSV(s) from eval_entailment_prompt.py --out. With several, "
                         "the first file wins on any unit they disagree about (separate runs "
                         "re-query the model, so verdicts can differ).")
    ap.add_argument("--map", required=True,
                    help="annotate.map.json — supplies `model` and the full fact lists")
    ap.add_argument("--juror", default="gemini",
                    help="Juror name to write the columns under (default: gemini, which is "
                         "what `rater_agreement_matrix.py --jurors gemini` expects)")
    ap.add_argument("--output", required=True, help="Output scores CSV")
    args = ap.parse_args()

    amap = json.loads(Path(args.map).read_text(encoding="utf-8"))
    if not isinstance(amap, dict):
        sys.exit("--map must be a JSON object keyed by record_id")

    # record_id -> direction -> {fact: entailed}
    verdicts: dict[str, dict[str, dict[str, bool]]] = defaultdict(lambda: defaultdict(dict))
    split_dupes = cross_file = 0
    for path in args.detail:
        df = pd.read_csv(path, dtype=str).fillna("")
        need = {"record_id", "direction", "fact", "judge"}
        missing = need - set(df.columns)
        if missing:
            sys.exit(f"{path} is missing columns: {sorted(missing)}")

        # Within one file, the same fact STRING can occur at several idx (atomization
        # emits repeated date anchors), and the judge may entail some occurrences and
        # not others. Collapse with OR, because that is exactly what a real
        # score_facts_batch CSV encodes: it stores entailed fact strings, so one
        # entailed occurrence puts the string in the list and membership reads as
        # entailed for every occurrence. Anything else would disagree with the
        # production pipeline.
        this_file: dict[tuple, bool] = {}
        for _, r in df.iterrows():
            k = (r["record_id"], r["direction"], r["fact"])
            v = _truthy(r["judge"])
            if k in this_file:
                split_dupes += (this_file[k] != v)
            this_file[k] = this_file.get(k, False) or v

        for (rid, direction, fact), v in this_file.items():
            d = verdicts[rid][direction]
            if fact in d:
                cross_file += (d[fact] != v)          # first file wins
            else:
                d[fact] = v
        print(f"  {path}: {len(df)} graded facts")

    if split_dupes:
        print(f"  NOTE: {split_dupes} repeated fact string(s) were entailed at one index but "
              f"not another; collapsed to entailed (matches how the scores CSV is read)")
    if cross_file:
        print(f"  NOTE: {cross_file} fact(s) judged differently across the detail files; "
              f"kept the first file's verdict")

    juror = args.juror
    fields = ["question_id", "source_name", "model",
              f"{juror}_precision", f"{juror}_recall",
              f"{juror}_entailed_ref_facts", f"{juror}_entailed_cand_facts"]

    rows, missing_records, ungraded = [], 0, 0
    for rid, rec in amap.items():
        if rid not in verdicts:
            missing_records += 1
            continue
        entailed = {}
        totals = {}
        for direction, item_key in (("recall", "recall_items"), ("precision", "precision_items")):
            items = sorted((rec.get(item_key) or []), key=lambda x: x.get("idx", 0))
            marks = verdicts[rid].get(direction, {})
            ungraded += sum(1 for it in items if it.get("fact") not in marks)
            # Fact STRINGS, matching score_facts_batch's schema — the matrix tests
            # membership by exact string, not by index.
            entailed[direction] = [it["fact"] for it in items if marks.get(it.get("fact"), False)]
            totals[direction] = len(items)

        rows.append({
            "question_id": rec.get("question_id", ""),
            "source_name": rec.get("source_name", ""),
            "model":       rec.get("model", ""),
            f"{juror}_recall":    round(len(entailed["recall"])    / totals["recall"], 4)
                                  if totals["recall"] else 0.0,
            f"{juror}_precision": round(len(entailed["precision"]) / totals["precision"], 4)
                                  if totals["precision"] else 0.0,
            f"{juror}_entailed_ref_facts":  json.dumps(entailed["recall"]),
            f"{juror}_entailed_cand_facts": json.dumps(entailed["precision"]),
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {out}  ({len(rows)} rows, juror='{juror}')")
    if missing_records:
        print(f"  {missing_records} map record(s) had no detail rows — omitted "
              f"(the matrix will report them as unmatched)")
    if ungraded:
        print(f"  {ungraded} fact(s) in the map were never graded, so no judge verdict "
              f"exists for them; emitted as not-entailed")


if __name__ == "__main__":
    main()
