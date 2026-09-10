#!/usr/bin/env python
"""A/B/TIE profile per rater: who hedges, who commits, and which way they lean.

Companion to ``rater_agreement_matrix.py`` (same inputs, same rater set, same
position-collapsed units). The matrix says *how much* raters agree; this says *how they
behave* when they don't.

Note what "lenient" means here. After the position collapse, A and B are model
IDENTITIES (A = the lexicographically smaller model), so an A/B skew is a preference
between the two systems, not severity. The severity-like axis in ELO is **hedging**:
how readily a rater declines to name a winner. So:

  tie_rate     share of judgments called TIE -- the hedging/"easy" direction
  decisiveness 1 - tie_rate

Reported per rater, per dimension + overall (majority vote) + all pooled:
  pct_A, pct_B, pct_TIE, decisiveness, and a_minus_b (net lean toward model A)

Stratified by how contested the judgment is:
  unanimous  units every rater labelled identically -- the easy calls
  contested  units at least one rater differed on -- where behaviour separates

Pairwise on disagreements only, split into the two kinds that mean different things:
  flip           X said A, Y said B (or vice versa) -- a real directional conflict,
                 neither party is "more lenient"
  adjacent       exactly one said TIE -- the TIE-sayer hedged where the other committed

Aggregated into a hedging index per rater = (times it said TIE while the counterpart
committed) / (all its disagreements). High = hedges when others commit.

Outputs {prefix}_rater_profile.csv, {prefix}_stratified.csv, {prefix}_pairwise.csv,
{prefix}_hedging_index.csv.

Usage
-----
    python leniency_report.py \
        --human chloe=$EVAL_DIR/elo/raters/human_chloe.csv \
        --human sulaiman=$EVAL_DIR/elo/raters/human_sulaiman.csv \
        --pred-dir $EVAL_DIR/elo_eval_human_pool \
        --pool-judges --prefix figures/elo_leniency
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from .rater_agreement_matrix import (
    DIMS, LABELS, add_overall, load_raters, majority_rater, prepare,
)

UNITS = list(DIMS) + ["overall", "pooled"]


def _fmt(df) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:.3f}")


def to_units(prepped: dict[str, pd.DataFrame], collapse: bool) -> dict[str, dict[tuple, str]]:
    """{rater: {(pair_key, dimension): label}} -- the shape the stats below need."""
    frames = list(prepped.values())
    key = [c for c in (["question_id", "model_a", "model_b", "source"] if collapse
                       else ["question_id", "model_a", "model_b", "position", "source"])
           if all(c in f.columns for f in frames)]
    out: dict[str, dict[tuple, str]] = {}
    for name, df in prepped.items():
        d: dict[tuple, str] = {}
        for _, row in df.iterrows():
            pk = tuple(str(row[c]) for c in key)
            for dim in list(DIMS) + ["overall"]:
                lab = str(row.get(dim, ""))
                if lab in LABELS:
                    d[(pk, dim)] = lab
        out[name] = d
    return out


def _sel(d: dict[tuple, str], unit: str, keys=None):
    src = d if keys is None else {k: d[k] for k in keys if k in d}
    return [v for k, v in src.items() if unit == "pooled" or k[1] == unit]


def rater_profile(units: dict[str, dict[tuple, str]]) -> pd.DataFrame:
    rows = []
    for name, d in units.items():
        for unit in UNITS:
            labs = _sel(d, unit)
            n = len(labs)
            c = Counter(labs)
            rows.append({
                "rater": name, "unit": unit, "n": n,
                "pct_A": c["A"] / n if n else float("nan"),
                "pct_B": c["B"] / n if n else float("nan"),
                "pct_TIE": c["TIE"] / n if n else float("nan"),
                "decisiveness": 1 - c["TIE"] / n if n else float("nan"),
                "a_minus_b": (c["A"] - c["B"]) / n if n else float("nan"),
            })
    return pd.DataFrame(rows)


def stratified(units: dict[str, dict[tuple, str]]) -> pd.DataFrame:
    shared = set.intersection(*(set(d) for d in units.values()))
    unanimous = {u for u in shared if len({d[u] for d in units.values()}) == 1}
    contested = shared - unanimous
    print(f"\n  {len(shared)} units labelled by every rater: "
          f"{len(unanimous)} unanimous, {len(contested)} contested")

    rows = []
    for name, d in units.items():
        for stratum, keys in (("unanimous", unanimous), ("contested", contested)):
            for unit in ("overall", "pooled"):
                labs = _sel(d, unit, keys)
                n = len(labs)
                c = Counter(labs)
                rows.append({
                    "rater": name, "stratum": stratum, "unit": unit, "n": n,
                    "pct_A": c["A"] / n if n else float("nan"),
                    "pct_B": c["B"] / n if n else float("nan"),
                    "pct_TIE": c["TIE"] / n if n else float("nan"),
                    "decisiveness": 1 - c["TIE"] / n if n else float("nan"),
                })
    return pd.DataFrame(rows)


def pairwise(units: dict[str, dict[tuple, str]]) -> pd.DataFrame:
    names = list(units)
    rows = []
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            a, b = units[na], units[nb]
            for unit in UNITS:
                shared = [k for k in (set(a) & set(b)) if unit == "pooled" or k[1] == unit]
                flip = sum(1 for k in shared if {a[k], b[k]} == {"A", "B"})
                a_tie = sum(1 for k in shared if a[k] == "TIE" and b[k] in ("A", "B"))
                b_tie = sum(1 for k in shared if b[k] == "TIE" and a[k] in ("A", "B"))
                n_dis = flip + a_tie + b_tie
                rows.append({
                    "rater_a": na, "rater_b": nb, "unit": unit,
                    "n_shared": len(shared), "n_disagree": n_dis, "n_flip": flip,
                    "a_tie_b_commit": a_tie, "b_tie_a_commit": b_tie,
                    "a_hedge_share": (a_tie / (a_tie + b_tie)) if (a_tie + b_tie) else float("nan"),
                    "more_hedging": (na if a_tie > b_tie else nb if b_tie > a_tie else "tie")
                                    if (a_tie + b_tie) else "",
                })
    return pd.DataFrame(rows)


def hedging_index(pw: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        hedged = dis = 0
        for _, r in pw[pw["unit"] == "pooled"].iterrows():
            if r["rater_a"] == name:
                hedged += r["a_tie_b_commit"]
                dis += r["n_disagree"]
            elif r["rater_b"] == name:
                hedged += r["b_tie_a_commit"]
                dis += r["n_disagree"]
        rows.append({"rater": name, "n_disagreements": dis, "times_hedged": hedged,
                     "hedging_index": (hedged / dis) if dis else float("nan")})
    return pd.DataFrame(rows).sort_values("hedging_index", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", action="append", metavar="NAME=CSV")
    ap.add_argument("--judge", action="append", metavar="NAME=CSV")
    ap.add_argument("--pred-dir")
    ap.add_argument("--no-position-collapse", action="store_true")
    ap.add_argument("--pool-judges", action="store_true")
    ap.add_argument("--prefix", default="elo_leniency")
    args = ap.parse_args()

    collapse = not args.no_position_collapse
    raters, kinds = load_raters(args)
    prepped = prepare(raters, collapse)

    if args.pool_judges:
        members = {n: prepped[n] for n in prepped if kinds.get(n) == "judge"}
        if len(members) >= 2:
            prepped["judges_majority"] = add_overall(majority_rater(members, collapse))
            kinds["judges_majority"] = "judge"

    units = to_units(prepped, collapse)
    names = list(units)

    prof = rater_profile(units)
    strat = stratified(units)
    pw = pairwise(units)
    idx = hedging_index(pw, names)

    print("\n=== A/B/TIE per rater (A = lexicographically smaller model, post-collapse) ===")
    print(_fmt(prof[prof["unit"].isin(["overall", "pooled"])]))

    print("\n=== Per dimension ===")
    print(_fmt(prof[prof["unit"].isin(DIMS)]))

    print("\n=== Stratified: unanimous vs contested units ===")
    print(_fmt(strat[strat["unit"] == "pooled"]))

    print("\n=== Pairwise on disagreements (pooled): flips vs who hedged ===")
    print(_fmt(pw[pw["unit"] == "pooled"][
        ["rater_a", "rater_b", "n_disagree", "n_flip", "a_tie_b_commit",
         "b_tie_a_commit", "a_hedge_share", "more_hedging"]]))

    print("\n=== Hedging index (share of own disagreements where IT said TIE) ===")
    print(_fmt(idx))
    print("  High = declines to pick a winner where others commit (the 'easy' rater).")
    print("  Flips are excluded from the numerator: they are directional conflicts,")
    print("  not leniency.")

    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prof.to_csv(prefix.with_name(prefix.name + "_rater_profile.csv"), index=False)
    strat.to_csv(prefix.with_name(prefix.name + "_stratified.csv"), index=False)
    pw.to_csv(prefix.with_name(prefix.name + "_pairwise.csv"), index=False)
    idx.to_csv(prefix.with_name(prefix.name + "_hedging_index.csv"), index=False)
    print(f"\nWrote {prefix.name}_{{rater_profile,stratified,pairwise,hedging_index}}.csv")


if __name__ == "__main__":
    main()
