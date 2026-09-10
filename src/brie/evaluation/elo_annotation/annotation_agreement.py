#!/usr/bin/env python
"""Preliminary human-vs-judge agreement stats for legacy ELO annotations.

Reads the *legacy* (pre-blinding) ELO annotation exports — the ones where each
record carries the LLM judge's verdict (``llm_{dim}_winner``) and the human's
per-dimension decision as ``{agrees, override, reason}``. From each record two
labels are reconstructed per dimension:

    gemini (judge) label = llm_{dim}_winner
    human label          = llm_{dim}_winner if agrees else override

Because the annotator saw the judge's verdict and chose only to *agree* or
*override* it, the gemini judge here is gemini-2.5-flash-lite (``judge`` field
== "gemini"). The two labels share the same positional A/B frame (A == model_a
as shown), so they are compared directly — no position flipping.

CAVEAT: this annotation was NOT blinded — the human saw gemini-flash-lite's
answer before deciding. Agreement here is therefore an upper bound (anchoring
inflates it, most visibly on ties). The blinded pipeline
(sample_elo_pairs -> merge_elo_annotations -> compare_judge_accuracy) is the
unbiased successor; this script is for a quick read on the legacy exports.

Reports, per dimension and for the overall (majority-vote) winner:
  n                  pairs with both labels present
  agreement          raw exact-match rate (== accuracy of judge vs human)
  cohen_kappa        Cohen's kappa on nominal {A, TIE, B}
  quad_kappa         quadratic-weighted kappa on ordinal A < TIE < B
  flip_rate          A<->B directional disagreements (the errors that matter)
  adjacent_rate      exactly one side said TIE
  tolerant_acc       1 - flip_rate (TIE treated as a wildcard)
  committed_dir_acc  agreement where BOTH picked a side (no TIE)
  override_rate      how often the human overrode the judge (human != judge)
Plus a 3x3 confusion matrix (rows human, cols judge) per dimension.

Optionally (--pred-out) writes a gemini-2.5-flash-lite prediction CSV and
(--truth-out) the human truth CSV in the schema consumed by
compare_judge_accuracy.py, so the legacy set can also be fed through the
blinded-era tooling.

Usage
-----
    python annotation_agreement.py elo_annotate.json elo_baseline*.json
    python annotation_agreement.py DIR/                 # all *.json in DIR
    python annotation_agreement.py *.json --csv stats.csv \
        --pred-out gemini-2.5-flash-lite.csv --truth-out human.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

DIMS = ["completeness", "relevancy", "concision"]
LABELS = ["A", "TIE", "B"]
ORD = {"A": 0, "TIE": 1, "B": 2}  # ordinal scale for weighted kappa


def norm(w) -> str:
    """Normalize a winner label to A / TIE / B (folds T/TIED -> TIE)."""
    w = str(w or "").strip().upper()
    return {"T": "TIE", "TIED": "TIE"}.get(w, w) if w in ("T", "TIED") else w


def _expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pat in patterns:
        p = Path(pat)
        matches = sorted(p.glob("*.json")) if p.is_dir() else [Path(m) for m in glob.glob(pat)]
        if not matches and p.exists():
            matches = [p]
        if not matches:
            print(f"WARNING: no files matched {pat!r}", file=sys.stderr)
        for m in matches:
            rp = m.resolve()
            if rp not in seen:
                seen.add(rp)
                paths.append(m)
    return paths


def overall(labels: dict) -> str:
    """Majority vote across dimensions (mirrors score_elo_batch/_parse_judgment)."""
    c = Counter(labels[d] for d in DIMS)
    if c["A"] >= 2:
        return "A"
    if c["B"] >= 2:
        return "B"
    return "TIE"


def load_pairs(paths: list[Path]) -> list[dict]:
    """Reconstruct (human, gemini) labels per pair from legacy annotation JSON."""
    rows: list[dict] = []
    for path in paths:
        data = json.load(open(path, encoding="utf-8"))
        if not isinstance(data, list):
            print(f"WARNING: {path} is not a JSON list — skipping", file=sys.stderr)
            continue
        n_override = 0
        for r in data:
            if "llm_completeness_winner" not in r:
                print(f"WARNING: {path.name}: record lacks llm_* fields "
                      "(not a legacy annotation export?) — skipping file", file=sys.stderr)
                rows = [x for x in rows if x["source"] != path.stem]
                break
            human, gem = {}, {}
            for d in DIMS:
                llm = norm(r.get(f"llm_{d}_winner"))
                cell = r.get(d) or {}
                if cell.get("agrees"):
                    h = llm
                else:
                    h = norm(cell.get("override"))
                    if not h:
                        h = ""  # disagreed but no override recorded -> drop this cell
                    else:
                        n_override += 1
                human[d], gem[d] = h, llm
            rows.append({
                "question_id": r.get("question_id", ""),
                "model_a": r.get("model_a", ""), "model_b": r.get("model_b", ""),
                "position": r.get("position", ""), "source": path.stem,
                "judge": r.get("judge", ""),
                "human": human, "gem": gem,
            })
        print(f"  {path.name}: {sum(1 for x in rows if x['source']==path.stem)} pairs "
              f"({n_override} dim-level human overrides)")
    print(f"Loaded {len(rows)} annotated pairs from {len(paths)} file(s)")
    judges = {x["judge"] for x in rows if x["judge"]}
    if judges:
        print(f"Embedded judge(s): {sorted(judges)}  (treated as gemini-2.5-flash-lite)")
    return rows


# ── Metrics ───────────────────────────────────────────────────────────────────

def cohen_kappa(true: list[str], pred: list[str]) -> float:
    n = len(true)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(true, pred)) / n
    ct, cp = Counter(true), Counter(pred)
    pe = sum((ct[label] / n) * (cp[label] / n) for label in LABELS)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def quad_kappa(true: list[str], pred: list[str]) -> float:
    """Quadratic-weighted Cohen's kappa over the ordinal scale A < TIE < B."""
    n = len(true)
    if n == 0:
        return float("nan")
    k = len(LABELS)
    observed = [[0.0] * k for _ in range(k)]
    for t, p in zip(true, pred):
        observed[ORD[t]][ORD[p]] += 1
    row = [sum(observed[i]) for i in range(k)]
    col = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    dw = (k - 1) ** 2
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / dw
            num += w * observed[i][j]
            den += w * row[i] * col[j] / n
    return float("nan") if den == 0 else 1.0 - num / den


def metrics(human: list[str], gem: list[str]) -> dict:
    """Tie-aware agreement metrics for one dimension (human vs gemini)."""
    pairs = [(h, g) for h, g in zip(human, gem) if h and g]
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    t = [h for h, _ in pairs]
    p = [g for _, g in pairs]
    flips = {("A", "B"), ("B", "A")}
    agree = sum(a == b for a, b in pairs)
    flip = sum((a, b) in flips for a, b in pairs)
    adjacent = n - agree - flip
    committed = [(a, b) for a, b in pairs if a != "TIE" and b != "TIE"]
    committed_agree = sum(a == b for a, b in committed)
    return {
        "n": n,
        "agreement": round(agree / n, 4),
        "cohen_kappa": round(cohen_kappa(t, p), 4),
        "quad_kappa": round(quad_kappa(t, p), 4),
        "flip_rate": round(flip / n, 4),
        "adjacent_rate": round(adjacent / n, 4),
        "tolerant_acc": round((agree + adjacent) / n, 4),
        "committed_n": len(committed),
        "committed_dir_acc": round(committed_agree / len(committed), 4) if committed else float("nan"),
        "override_rate": round(sum(a != b for a, b in pairs) / n, 4),
    }


def confusion(human: list[str], gem: list[str]) -> dict:
    c = Counter((h, g) for h, g in zip(human, gem) if h and g)
    return {(h, g): c.get((h, g), 0) for h in LABELS for g in LABELS}


def _fmt(v) -> str:
    if isinstance(v, float):
        return "  nan" if v != v else f"{v:6.3f}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="Legacy annotation JSON file(s), globs, or a directory")
    ap.add_argument("--csv", default=None, help="Write the per-dimension metrics table to this CSV")
    ap.add_argument("--pred-out", default=None,
                    help="Write a gemini-2.5-flash-lite prediction CSV (compare_judge_accuracy schema)")
    ap.add_argument("--truth-out", default=None,
                    help="Write the human truth CSV (compare_judge_accuracy schema)")
    args = ap.parse_args()

    paths = _expand(args.inputs)
    if not paths:
        sys.exit("No input files found.")
    rows = load_pairs(paths)
    if not rows:
        sys.exit("No usable annotation records.")

    # Assemble label vectors per dimension + overall.
    keys = DIMS + ["overall"]
    human_v: dict[str, list[str]] = {k: [] for k in keys}
    gem_v: dict[str, list[str]] = {k: [] for k in keys}
    for r in rows:
        for d in DIMS:
            human_v[d].append(r["human"][d])
            gem_v[d].append(r["gem"][d])
        human_v["overall"].append(overall(r["human"]))
        gem_v["overall"].append(overall(r["gem"]))

    metric_rows = [{"dimension": k, **metrics(human_v[k], gem_v[k])} for k in keys]

    cols = ["dimension", "n", "agreement", "cohen_kappa", "quad_kappa", "flip_rate",
            "adjacent_rate", "tolerant_acc", "committed_dir_acc", "override_rate"]
    widths = {c: max(len(c), max(len(_fmt(m.get(c, ""))) for m in metric_rows)) for c in cols}
    print("\n=== Human vs gemini-2.5-flash-lite agreement (per dimension + overall) ===")
    print("  ".join(c.rjust(widths[c]) for c in cols))
    for m in metric_rows:
        print("  ".join(_fmt(m.get(c, "")).rjust(widths[c]) for c in cols))

    print("\n=== Confusion matrices (rows = human, cols = gemini-flash-lite) ===")
    for k in keys:
        conf = confusion(human_v[k], gem_v[k])
        print(f"\n{k}:")
        print("         " + "".join(f"g={g:>5}" for g in LABELS))
        for h in LABELS:
            print(f"  h={h:>3}  " + "".join(f"{conf[(h, g)]:6d}" for g in LABELS))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows({c: m.get(c, "") for c in cols} for m in metric_rows)
        print(f"\nWrote metrics -> {args.csv}")

    schema = ["question_id", "model_a", "model_b", "position", "source"] + DIMS
    if args.pred_out:
        with open(args.pred_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=schema)
            w.writeheader()
            for r in rows:
                w.writerow({**{c: r[c] for c in schema[:5]}, **r["gem"]})
        print(f"Wrote gemini-2.5-flash-lite predictions -> {args.pred_out}")
    if args.truth_out:
        with open(args.truth_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=schema)
            w.writeheader()
            for r in rows:
                w.writerow({**{c: r[c] for c in schema[:5]}, **r["human"]})
        print(f"Wrote human truth -> {args.truth_out}")


if __name__ == "__main__":
    main()
