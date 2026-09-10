#!/usr/bin/env python
"""All-pairs agreement matrix + heatmaps for fact-entailment annotation.

The fact analogue of ``elo_annotation/rater_agreement_matrix.py``. Every human
annotator and every LLM juror becomes one rater in a single N x N matrix, with one
panel per direction (recall, precision) plus both pooled.

Unlike the ELO study, nothing needs re-judging: ``sample_fact_entailment.py`` already
writes per-juror verdicts into the sidecar map --

    recall_items / precision_items: [{idx, fact, consensus_entailed, votes,
                                      judges: {gemini: bool, claude: bool, gpt: bool}}]

-- which are deliberately kept out of the blinded HTML. This script reads the jurors
straight from the map and the humans from their blinded exports

    recall_items / precision_items: [{idx, fact, entailed, note}]

and compares them on the unit **(record_id, direction, idx)** -- one fact, in one
direction, of one record. Labels are binary: entailed (Present) vs not (Absent);
items either side left unmarked are excluded from that cell.

Because the label is binary and usually skewed (most reference facts are entailed),
kappa is prone to the prevalence paradox: two raters can agree on 95% of facts and
still score kappa near 0. Three metrics are therefore emitted side by side --

  agreement     raw exact-match rate
  cohen_kappa   chance-corrected, but depressed by skewed margins
  pabak         prevalence-adjusted (2*po - 1); robust to skew

-- plus a per-rater Present-rate table, which is what makes a low kappa interpretable.
Read agreement and kappa together; if they diverge sharply, trust pabak and the
Present-rate table.

Outputs
-------
  {prefix}_agreement.png / _cohen_kappa.png / _pabak.png
  {prefix}_long.csv       every rater pair x direction with all metrics + n
  {prefix}_prevalence.csv per-rater Present rate per direction

Usage
-----
    python rater_agreement_matrix.py \
        --map $EVAL_DIR/facts_score/annotate.map.json \
        --human bridget=$EVAL_DIR/facts_score/human/fact_annotations_bridget.json \
        --human jordan=$EVAL_DIR/facts_score/human/fact_annotations_jordan.json \
        --include-consensus --prefix figures/fact_agreement
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

DIRECTIONS = ["recall", "precision"]
ITEM_KEYS = {"recall": "recall_items", "precision": "precision_items"}
UNITS = DIRECTIONS + ["pooled"]

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)


# -- Loading -------------------------------------------------------------------

def _expand(pattern: str) -> list[Path]:
    pattern = os.path.expanduser(os.path.expandvars(pattern))
    p = Path(pattern)
    if p.is_dir():
        return sorted(p.glob("*.json"))
    matched = [Path(m) for m in glob.glob(pattern)]
    return sorted(matched) if matched else [p]


def _parse_named(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        sys.exit(f"Expected NAME=path.json, got {spec!r}")
    name, path = spec.split("=", 1)
    return name.strip(), path.strip()


def load_human(pattern: str, label: str) -> dict[tuple, bool]:
    """{(record_id, direction, idx): entailed} from one annotator's blinded export(s)."""
    out: dict[tuple, bool] = {}
    files = _expand(pattern)
    seen_records: set[str] = set()
    for path in files:
        if not path.exists():
            sys.exit(f"[{label}] file not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            sys.exit(f"[{label}] {path} is not a JSON list")
        for rec in data:
            rid = rec.get("record_id")
            if not rid:
                print(f"  [{label}] WARNING: record without record_id in {path.name} — skipped")
                continue
            seen_records.add(rid)
            for direction, key in ITEM_KEYS.items():
                for it in (rec.get(key) or []):
                    ent = it.get("entailed")
                    if ent is None:            # left unmarked by the annotator
                        continue
                    out[(rid, direction, it.get("idx"))] = bool(ent)
    print(f"[{label}] {len(out)} marked facts across {len(seen_records)} records "
          f"from {len(files)} file(s)")
    return out


def load_map_raters(map_path: str, include_consensus: bool
                    ) -> tuple[dict[str, dict[tuple, bool]], int]:
    """{juror: {(record_id, direction, idx): entailed}} from the sidecar map."""
    with open(map_path, encoding="utf-8") as f:
        amap = json.load(f)
    if not isinstance(amap, dict):
        sys.exit("--map must be a JSON object keyed by record_id.")

    raters: dict[str, dict[tuple, bool]] = {}
    for rid, rec in amap.items():
        for direction, key in ITEM_KEYS.items():
            for it in (rec.get(key) or []):
                unit = (rid, direction, it.get("idx"))
                for juror, verdict in (it.get("judges") or {}).items():
                    if verdict is None:
                        continue
                    raters.setdefault(juror, {})[unit] = bool(verdict)
                if include_consensus and it.get("consensus_entailed") is not None:
                    raters.setdefault("consensus", {})[unit] = bool(it["consensus_entailed"])

    print(f"Loaded map with {len(amap)} records from {map_path}")
    for name, d in raters.items():
        print(f"[{name}] {len(d)} facts (from map)")
    return raters, len(amap)


def _parse_facts(raw) -> list:
    """Mirror of score_facts_batch._parse_facts: tolerate JSON, repr, blank, NaN."""
    if raw is None or isinstance(raw, float):
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw.strip():
        for loader in (json.loads, ast.literal_eval):
            try:
                result = loader(raw)
                if isinstance(result, list):
                    return result
            except Exception:
                pass
    return []


def load_scores_raters(scores_path: str, amap: dict, include_consensus: bool
                       ) -> dict[str, dict[tuple, bool]]:
    """Juror labels read from a scores CSV, aligned to an EXISTING map.

    Avoids regenerating the map (which re-samples, and can land on a different 50
    records with different record_ids). The map supplies record_id, the fact `idx`
    ordering, and the fact text the annotators actually saw; the CSV supplies the
    verdicts. A fact counts as entailed when its exact string appears in that juror's
    entailed list -- the same rule sample_fact_entailment._build_fact_items applies.
    """
    df = pd.read_csv(scores_path, dtype=str).fillna("")
    jurors = sorted({c[: -len("_entailed_ref_facts")] for c in df.columns
                     if c.endswith("_entailed_ref_facts")} - {"consensus"})
    if not jurors:
        sys.exit(f"{scores_path} has no *_entailed_ref_facts columns")
    print(f"Loaded scores CSV with {len(df)} rows from {scores_path}")
    print(f"  jurors found: {', '.join(jurors)}")

    by_triple, by_pair = {}, {}
    for _, row in df.iterrows():
        qid, src, mdl = row.get("question_id", ""), row.get("source_name", ""), row.get("model", "")
        by_triple[(qid, src, mdl)] = row
        by_pair.setdefault((qid, src), row)

    col = {"recall": "_entailed_ref_facts", "precision": "_entailed_cand_facts"}
    raters: dict[str, dict[tuple, bool]] = {}
    matched = fell_back = unmatched = 0

    for rid, rec in amap.items():
        qid = str(rec.get("question_id", ""))
        src = str(rec.get("source_name", ""))
        mdl = str(rec.get("model", ""))
        row = by_triple.get((qid, src, mdl))
        if row is None:
            row = by_pair.get((qid, src))
            if row is None:
                unmatched += 1
                continue
            fell_back += 1
        else:
            matched += 1

        for direction, key in ITEM_KEYS.items():
            for it in (rec.get(key) or []):
                unit = (rid, direction, it.get("idx"))
                fact = it.get("fact", "")
                for j in jurors:
                    ent = _parse_facts(row.get(f"{j}{col[direction]}", ""))
                    raters.setdefault(j, {})[unit] = fact in ent
                if include_consensus:
                    cons = _parse_facts(row.get(f"consensus{col[direction]}", ""))
                    raters.setdefault("consensus", {})[unit] = fact in cons

    print(f"  matched {matched}/{len(amap)} map records on (question_id, source_name, model)"
          + (f", {fell_back} via (question_id, source_name) fallback" if fell_back else ""))
    if unmatched:
        print(f"  WARNING: {unmatched} map records had no row in the scores CSV — "
              "their facts are excluded from every juror", file=sys.stderr)
    for name, d in raters.items():
        print(f"[{name}] {len(d)} facts (from scores CSV)")
    return raters


# -- Metrics -------------------------------------------------------------------

def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    labels = {True, False}
    pe = sum((ca[label] / n) * (cb[label] / n) for label in labels)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def pair_metrics(pairs: list[tuple[bool, bool]]) -> dict:
    n = len(pairs)
    if n == 0:
        return {"n": 0, "agreement": float("nan"), "cohen_kappa": float("nan"),
                "pabak": float("nan"), "present_rate_a": float("nan"),
                "present_rate_b": float("nan")}
    po = sum(a == b for a, b in pairs) / n
    return {
        "n": n,
        "agreement": po,
        "cohen_kappa": cohen_kappa(pairs),
        "pabak": 2 * po - 1,          # k = 2
        "present_rate_a": sum(a for a, _ in pairs) / n,
        "present_rate_b": sum(b for _, b in pairs) / n,
    }


def build_long(raters: dict[str, dict[tuple, bool]]) -> pd.DataFrame:
    names = list(raters)
    rows: list[dict] = []
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            a, b = raters[na], raters[nb]
            shared = set(a) & set(b)
            if not shared:
                print(f"WARNING: {na} vs {nb} share 0 facts", file=sys.stderr)
            by_dir: dict[str, list[tuple[bool, bool]]] = {d: [] for d in DIRECTIONS}
            for unit in shared:
                by_dir[unit[1]].append((a[unit], b[unit]))
            pooled: list[tuple[bool, bool]] = []
            for d in DIRECTIONS:
                pooled += by_dir[d]
                rows.append({"rater_a": na, "rater_b": nb, "direction": d,
                             **pair_metrics(by_dir[d])})
            rows.append({"rater_a": na, "rater_b": nb, "direction": "pooled",
                         **pair_metrics(pooled)})
    return pd.DataFrame(rows)


def majority_rater(members: dict[str, dict[tuple, bool]]) -> dict[tuple, bool]:
    """Per-fact majority vote across the given raters.

    A unit split exactly evenly (possible only with an even number of voters) has no
    majority and is omitted, rather than broken toward either label.
    """
    votes: dict[tuple, list[bool]] = {}
    for d in members.values():
        for unit, v in d.items():
            votes.setdefault(unit, []).append(v)
    out: dict[tuple, bool] = {}
    for unit, vs in votes.items():
        yes = sum(vs)
        if yes * 2 == len(vs):
            continue
        out[unit] = yes * 2 > len(vs)
    return out


def prevalence_table(raters: dict[str, dict[tuple, bool]]) -> pd.DataFrame:
    rows = []
    for name, d in raters.items():
        rec = {"rater": name}
        for unit_name in UNITS:
            vals = [v for k, v in d.items()
                    if unit_name == "pooled" or k[1] == unit_name]
            rec[f"n_{unit_name}"] = len(vals)
            rec[f"present_rate_{unit_name}"] = (sum(vals) / len(vals)) if vals else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def to_matrix(long: pd.DataFrame, unit: str, metric: str, names: list[str]) -> np.ndarray:
    m = np.full((len(names), len(names)), np.nan)
    idx = {n: i for i, n in enumerate(names)}
    for _, r in long[long["direction"] == unit].iterrows():
        i, j = idx[r["rater_a"]], idx[r["rater_b"]]
        m[i, j] = m[j, i] = r[metric]
    return m


# -- Plotting ------------------------------------------------------------------

def _text_color(rgba) -> str:
    r, g, b = rgba[:3]
    return "#ffffff" if (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.5 else INK


def display_labels(names: list[str], human_names: list[str]) -> list[str]:
    """Publication tick labels: humans are anonymized in the order given, and any
    Gemini juror variant collapses to the model family."""
    order = {n: i for i, n in enumerate(human_names)}
    out = []
    for n in names:
        if n in order:
            out.append(f"Annotator {order[n] + 1}")
        elif "gemini" in n.lower():
            out.append("Gemini")
        else:
            out.append(n)
    return out


def plot_metric(long: pd.DataFrame, names: list[str], labels: list[str], metric: str,
                out_path: Path, units: list[str] | None = None) -> None:
    units = units or UNITS
    vals = long[metric].to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        print(f"WARNING: no finite values for {metric} — skipping", file=sys.stderr)
        return

    # One fixed 0-1 ramp for every metric so panels and figures are comparable.
    # A sub-chance kappa clamps to the lightest blue; its printed value still shows
    # the sign.
    cmap, norm = SEQ_CMAP, Normalize(vmin=0.0, vmax=1.0)

    n = len(names)
    side = max(3.0, 0.62 * n + 1.9)
    ncols = len(units)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * side, side * 1.08),
                             facecolor=SURFACE, constrained_layout=True, squeeze=False)

    for ax, unit in zip(axes.ravel(), units):
        m = to_matrix(long, unit, metric, names)
        ax.set_facecolor(SURFACE)
        ax.pcolormesh(np.ma.masked_invalid(m), cmap=cmap, norm=norm,
                      edgecolors=SURFACE, linewidth=2)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color=INK)
        ax.set_yticklabels(labels, fontsize=8, color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        label = "both directions pooled" if unit == "pooled" else unit
        ax.set_title(label, fontsize=11, color=INK, pad=8)
        for i in range(n):
            for j in range(n):
                v = m[i, j]
                if np.isfinite(v):
                    ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center",
                            fontsize=7.5, color=_text_color(cmap(norm(v))))

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, color=MUTED, labelcolor=MUTED, length=0)

    # No figure title/subtitle: these go into papers where the caption carries them.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True,
                    help="annotate.map.json from sample_fact_entailment.py (carries the jurors)")
    ap.add_argument("--human", action="append", metavar="NAME=JSON", required=True,
                    help="Blinded annotator export. Repeatable.")
    ap.add_argument("--scores", default=None,
                    help="Read juror verdicts from this score_facts_batch.py CSV instead of "
                         "the map's stored `judges`, aligning to the map by "
                         "(question_id, source_name, model). Use after backfilling a juror, "
                         "so the map (and its record_ids) never has to be regenerated.")
    ap.add_argument("--jurors", nargs="*", default=None,
                    help="Subset of jurors to include (default: all found)")
    ap.add_argument("--include-consensus", action="store_true",
                    help="Add the map's stored >=2/3 juror consensus as its own rater")
    ap.add_argument("--pool-jurors", action="store_true",
                    help="Add 'jurors_majority': majority vote across the INCLUDED jurors "
                         "(equals --include-consensus only when all three are in play)")
    ap.add_argument("--pool-humans", action="store_true",
                    help="Add 'humans_majority': majority vote across the --human raters "
                         "(needs 3+ humans; with 2 there is no majority)")
    ap.add_argument("--no-combined", action="store_true",
                    help="Drop the pooled-directions panel, leaving just recall and "
                         "precision. The pooled rows stay in the long CSV.")
    ap.add_argument("--prefix", default="fact_agreement", help="Output path prefix")
    args = ap.parse_args()

    raters: dict[str, dict[tuple, bool]] = {}
    human_names: list[str] = []
    for spec in args.human:
        name, path = _parse_named(spec)
        if name in raters:
            sys.exit(f"Duplicate rater name {name!r}")
        raters[name] = load_human(path, name)
        human_names.append(name)

    if args.scores:
        with open(args.map, encoding="utf-8") as f:
            amap = json.load(f)
        if not isinstance(amap, dict):
            sys.exit("--map must be a JSON object keyed by record_id.")
        print(f"Loaded map with {len(amap)} records from {args.map}")
        map_raters = load_scores_raters(args.scores, amap, args.include_consensus)
        n_map = len(amap)
    else:
        map_raters, n_map = load_map_raters(args.map, args.include_consensus)
    if args.jurors:
        unknown = set(args.jurors) - set(map_raters)
        if unknown:
            sys.exit(f"--jurors not present in the map: {sorted(unknown)}")
        map_raters = {k: v for k, v in map_raters.items()
                      if k in args.jurors or k == "consensus"}
    juror_names = [n for n in map_raters if n != "consensus"]
    for name, d in map_raters.items():
        if name in raters:
            sys.exit(f"Juror {name!r} collides with a --human name; rename one")
        raters[name] = d

    if args.pool_jurors:
        if len(juror_names) < 2:
            sys.exit("--pool-jurors needs at least two jurors in the map")
        raters["jurors_majority"] = majority_rater({n: raters[n] for n in juror_names})
        print(f"[jurors_majority] {len(raters['jurors_majority'])} facts "
              f"(majority of {', '.join(juror_names)})")
    if args.pool_humans:
        if len(human_names) < 3:
            sys.exit(f"--pool-humans needs 3+ humans to have a majority; got {len(human_names)}")
        raters["humans_majority"] = majority_rater({n: raters[n] for n in human_names})
        print(f"[humans_majority] {len(raters['humans_majority'])} facts "
              f"(majority of {', '.join(human_names)})")

    if len(raters) < 2:
        sys.exit("Need at least two raters.")

    # A juror whose batch failed reads as "entailed nothing" rather than "no label":
    # _parse_facts turns an empty column into [], so every fact scores Absent. That is
    # indistinguishable from a real verdict downstream, so flag it here.
    degenerate = [n for n, d in raters.items() if d and len(set(d.values())) == 1]
    if degenerate:
        print("\n*** WARNING: constant raters (every fact given the same label) ***",
              file=sys.stderr)
        for n in degenerate:
            only = "Present" if next(iter(raters[n].values())) else "Absent"
            print(f"      {n}: all {len(raters[n])} facts marked {only}", file=sys.stderr)
        print("      An all-Absent juror usually means its scoring batch failed and left\n"
              "      empty *_entailed_*_facts columns. Kappa is undefined against a constant\n"
              "      rater (NaN), and any consensus/majority including it is contaminated.\n"
              "      Backfill with: score_facts_batch.py --rerun-juror <name>, regenerate the\n"
              "      map, and re-run this script.", file=sys.stderr)

    # Coverage: humans annotate a sample of the map, so the shared unit set is the
    # binding constraint on every human-containing cell.
    print("\n=== Coverage ===")
    all_units = set.intersection(*(set(d) for d in raters.values()))
    print(f"  map records: {n_map}")
    print(f"  facts marked by EVERY rater: {len(all_units)}")
    for name, d in raters.items():
        print(f"  {name:<14} {len(d):>6} facts")

    names = list(raters)
    long = build_long(raters)
    if long.empty:
        sys.exit("No rater pairs produced metrics.")

    plot_units = DIRECTIONS if args.no_combined else UNITS
    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(prefix.with_name(prefix.name + "_long.csv"), index=False)
    prev = prevalence_table(raters)
    prev.to_csv(prefix.with_name(prefix.name + "_prevalence.csv"), index=False)
    print(f"\nWrote {prefix.name}_long.csv and {prefix.name}_prevalence.csv")

    print("\n=== Present rate per rater (the prevalence kappa is sensitive to) ===")
    print(prev.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== Pooled agreement, sorted by kappa ===")
    pooled = long[long["direction"] == "pooled"].sort_values("cohen_kappa", ascending=False)
    print(pooled[["rater_a", "rater_b", "n", "agreement", "cohen_kappa", "pabak"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    labels = display_labels(names, human_names)
    for metric in ("cohen_kappa", "pabak", "agreement"):
        plot_metric(long, names, labels, metric,
                    prefix.with_name(f"{prefix.name}_{metric}.png"),
                    units=plot_units)


if __name__ == "__main__":
    main()
