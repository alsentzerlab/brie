#!/usr/bin/env python
"""Is a rater lenient or strict? Fact-entailment leniency profile per rater.

Companion to ``rater_agreement_matrix.py`` (same inputs, same rater set, same
(record_id, direction, idx) unit). The matrix says *how much* raters agree; this says
*which direction* they lean when they don't.

For fact entailment "lenient" is well defined: marking a fact **Present** (entailed) is
the permissive call. So a rater's Present rate per direction IS its score --

    recall    = share of REFERENCE facts it judged entailed by the answer
    precision = share of ANSWER facts it judged supported by the reference

Reported per rater:
  recall / precision, micro (over all facts) and macro (mean of per-record rates,
  matching how score_facts_batch computes them)

Stratified by how contested the fact is:
  unanimous  facts every rater labelled the same way -- the easy ones
  contested  facts at least one rater disagreed on -- where leniency actually shows

Pairwise on disagreements only:
  for each pair, of the facts they disagreed on, what share did X call Present?
  >0.5 means X is the more lenient of the two. Aggregated per rater into a single
  leniency index = (times this rater said Present while the other said Absent) /
  (all disagreements it took part in).

Outputs {prefix}_rater_profile.csv, {prefix}_stratified.csv, {prefix}_pairwise.csv.

Usage
-----
    python leniency_report.py \
        --map    $EVAL_DIR/facts_score/annotate.map.json \
        --scores $EVAL_DIR/facts_score/scores_phase2_31flashlite_50.csv \
        --human bridget=.../fact_annotations_bridget.json \
        --human jordan=.../fact_annotations_jordan.json \
        --include-consensus --pool-jurors --prefix figures/fact_leniency
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from .rater_agreement_matrix import (
    DIRECTIONS, load_human, load_map_raters, load_scores_raters, majority_rater,
    _parse_named,
)

UNITS = DIRECTIONS + ["pooled"]


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:.3f}")


def rater_profile(raters: dict[str, dict[tuple, bool]]) -> pd.DataFrame:
    """Present rate per direction, micro and macro-by-record."""
    rows = []
    for name, d in raters.items():
        rec = {"rater": name}
        for unit in UNITS:
            vals = [(k[0], v) for k, v in d.items() if unit == "pooled" or k[1] == unit]
            rec[f"n_{unit}"] = len(vals)
            if not vals:
                rec[f"{unit}_micro"] = rec[f"{unit}_macro"] = float("nan")
                continue
            rec[f"{unit}_micro"] = sum(v for _, v in vals) / len(vals)
            per_record: dict[str, list[bool]] = defaultdict(list)
            for rid, v in vals:
                per_record[rid].append(v)
            rec[f"{unit}_macro"] = sum(sum(v) / len(v) for v in per_record.values()) / len(per_record)
        rows.append(rec)
    return pd.DataFrame(rows)


def stratified(raters: dict[str, dict[tuple, bool]]) -> pd.DataFrame:
    """Present rate split by whether every rater agreed on that fact."""
    shared = set.intersection(*(set(d) for d in raters.values()))
    unanimous = {u for u in shared if len({d[u] for d in raters.values()}) == 1}
    contested = shared - unanimous
    print(f"\n  {len(shared)} facts labelled by every rater: "
          f"{len(unanimous)} unanimous, {len(contested)} contested")

    rows = []
    for name, d in raters.items():
        for stratum, units in (("unanimous", unanimous), ("contested", contested)):
            rec = {"rater": name, "stratum": stratum, "n": len(units)}
            for unit in UNITS:
                sel = [d[u] for u in units if unit == "pooled" or u[1] == unit]
                rec[f"{unit}_present_rate"] = (sum(sel) / len(sel)) if sel else float("nan")
            rows.append(rec)
    return pd.DataFrame(rows)


def pairwise(raters: dict[str, dict[tuple, bool]]) -> pd.DataFrame:
    """Of the facts a pair disagreed on, who called Present?"""
    names = list(raters)
    rows = []
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            a, b = raters[na], raters[nb]
            for unit in UNITS:
                shared = [u for u in (set(a) & set(b))
                          if unit == "pooled" or u[1] == unit]
                a_pres = sum(1 for u in shared if a[u] and not b[u])
                b_pres = sum(1 for u in shared if b[u] and not a[u])
                n_dis = a_pres + b_pres
                rows.append({
                    "rater_a": na, "rater_b": nb, "direction": unit,
                    "n_shared": len(shared), "n_disagree": n_dis,
                    "a_present_b_absent": a_pres, "b_present_a_absent": b_pres,
                    "a_leniency": (a_pres / n_dis) if n_dis else float("nan"),
                    "more_lenient": (na if a_pres > b_pres else
                                     nb if b_pres > a_pres else "tie") if n_dis else "",
                })
    return pd.DataFrame(rows)


def leniency_index(pw: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Per rater, share of all its disagreements where IT was the permissive one."""
    rows = []
    for name in names:
        pres = dis = 0
        for _, r in pw[pw["direction"] == "pooled"].iterrows():
            if r["rater_a"] == name:
                pres += r["a_present_b_absent"]
                dis += r["n_disagree"]
            elif r["rater_b"] == name:
                pres += r["b_present_a_absent"]
                dis += r["n_disagree"]
        rows.append({"rater": name, "n_disagreements": dis,
                     "times_more_lenient": pres,
                     "leniency_index": (pres / dis) if dis else float("nan")})
    return pd.DataFrame(rows).sort_values("leniency_index", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--scores", default=None,
                    help="Read jurors from this scores CSV instead of the map's `judges`")
    ap.add_argument("--human", action="append", metavar="NAME=JSON", required=True)
    ap.add_argument("--jurors", nargs="*", default=None)
    ap.add_argument("--include-consensus", action="store_true")
    ap.add_argument("--pool-jurors", action="store_true")
    ap.add_argument("--prefix", default="fact_leniency")
    args = ap.parse_args()

    raters: dict[str, dict[tuple, bool]] = {}
    for spec in args.human:
        name, path = _parse_named(spec)
        raters[name] = load_human(path, name)

    if args.scores:
        with open(args.map, encoding="utf-8") as f:
            amap = json.load(f)
        print(f"Loaded map with {len(amap)} records from {args.map}")
        map_raters = load_scores_raters(args.scores, amap, args.include_consensus)
    else:
        map_raters, _ = load_map_raters(args.map, args.include_consensus)

    if args.jurors:
        map_raters = {k: v for k, v in map_raters.items()
                      if k in args.jurors or k == "consensus"}
    juror_names = [n for n in map_raters if n != "consensus"]
    raters.update(map_raters)
    if args.pool_jurors and len(juror_names) >= 2:
        raters["jurors_majority"] = majority_rater({n: raters[n] for n in juror_names})

    names = list(raters)
    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    prof = rater_profile(raters)
    strat = stratified(raters)
    pw = pairwise(raters)
    idx = leniency_index(pw, names)

    print("\n=== Recall / precision per rater (Present rate; higher = more lenient) ===")
    print(_fmt(prof[["rater", "n_recall", "recall_micro", "recall_macro",
                     "n_precision", "precision_micro", "precision_macro"]]))

    print("\n=== Present rate by stratum (unanimous vs contested facts) ===")
    print(_fmt(strat[["rater", "stratum", "n", "recall_present_rate",
                      "precision_present_rate", "pooled_present_rate"]]))

    print("\n=== Pairwise, on disagreements only (pooled) ===")
    print(_fmt(pw[pw["direction"] == "pooled"][
        ["rater_a", "rater_b", "n_disagree", "a_present_b_absent",
         "b_present_a_absent", "a_leniency", "more_lenient"]]))

    print("\n=== Leniency index (share of own disagreements where it said Present) ===")
    print(_fmt(idx))
    print("  1.0 = always the permissive one when raters differ; 0.0 = always the strict one.")

    prof.to_csv(prefix.with_name(prefix.name + "_rater_profile.csv"), index=False)
    strat.to_csv(prefix.with_name(prefix.name + "_stratified.csv"), index=False)
    pw.to_csv(prefix.with_name(prefix.name + "_pairwise.csv"), index=False)
    idx.to_csv(prefix.with_name(prefix.name + "_leniency_index.csv"), index=False)
    print(f"\nWrote {prefix.name}_{{rater_profile,stratified,pairwise,leniency_index}}.csv")


if __name__ == "__main__":
    main()
