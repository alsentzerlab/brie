#!/usr/bin/env python
"""All-pairs agreement matrix + heatmaps across human annotators and LLM judges.

``compare_judge_accuracy.py`` scores many judges against ONE human truth, and
``../fact_annotation/get_discrepencies.py`` compares exactly TWO annotators. This
script generalises both: every input is just a *rater*, and it computes the
symmetric rater x rater agreement matrix for each dimension plus the combined
(majority-vote) verdict.

A rater is any CSV in the shared ELO schema

    question_id, model_a, model_b, position, source, completeness, relevancy, concision

with labels in {A, B, TIE} -- the schema written by ``merge_elo_annotations.py
--truth`` (one human annotator) and by ``judge_elo_pairs.py`` at
``predictions/{judge}.csv`` (one LLM judge). Humans and judges are therefore
interchangeable here; the only difference is bookkeeping (see --human/--judge).

Position collapse
-----------------
Every rater is collapsed with the same rule used by ``compare_judge_accuracy``:
each positional A/B is resolved to a model NAME, then all rows sharing a canonical
pair (question_id + the unordered model pair [+ source]) combine --

    all orders name the same model  -> that model wins
    orders disagree                 -> TIE  (position-flipping names no winner)

and the result is re-expressed with the lexicographically smaller model in slot A.
This makes the matrix a comparison of *model identity*, not display slot.

Judges are expected to carry BOTH orders (run ``flip_elo_pairs.py`` and re-judge
into the same outdir first); a human who annotated only one order passes through
unchanged, which is correct under the assumption that human labels are not
order-biased. ``--check-orders`` reports, per rater, how many canonical pairs were
covered in only one order -- use it to confirm the judge backfill actually landed.

Outputs
-------
  {prefix}_quad_kappa.png    quadratic-weighted kappa, ordinal A < TIE < B
  {prefix}_cohen_kappa.png   nominal Cohen's kappa on {A, TIE, B}
  {prefix}_flip_rate.png     A<->B disagreement rate (ties ignored; lower is better)
  {prefix}_long.csv          every rater pair x dimension with all metrics + n

Each figure has four panels: the three dimensions and `overall`, the per-record
majority vote across dimensions (the label the ELO ranking consumes).

Usage
-----
    python rater_agreement_matrix.py \
        --human alice=raters/human_alice.csv --human bob=raters/human_bob.csv \
        --pred-dir $EVAL_DIR/elo_eval \
        --prefix figures/rater_agreement --check-orders
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from .compare_judge_accuracy import (
    DIMS, LABELS, _norm, _resolve_collapse, quadratic_weighted_kappa,
)

UNITS = DIMS + ["overall"]

# Palette (see dataviz reference): one sequential blue ramp fixed to 0-1 for every
# metric, so panels and figures are read on the same scale.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"

SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)


# -- Loading -------------------------------------------------------------------

def _parse_named(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        sys.exit(f"Expected NAME=path.csv, got {spec!r}")
    name, path = spec.split("=", 1)
    return name.strip(), Path(path.strip())


def load_raters(args) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """({rater: dataframe}, {rater: 'human'|'judge'}). Humans first, then judges."""
    specs: list[tuple[str, Path, str]] = []
    for spec in args.human or []:
        name, path = _parse_named(spec)
        specs.append((name, path, "human"))
    for spec in args.judge or []:
        name, path = _parse_named(spec)
        specs.append((name, path, "judge"))
    if args.pred_dir:
        pred_dir = Path(args.pred_dir)
        search = pred_dir / "predictions" if (pred_dir / "predictions").is_dir() else pred_dir
        found = sorted(search.glob("*.csv"))
        if not found:
            print(f"WARNING: no prediction CSVs under {search}", file=sys.stderr)
        specs += [(p.stem, p, "judge") for p in found]

    raters: dict[str, pd.DataFrame] = {}
    kinds: dict[str, str] = {}
    for name, path, kind in specs:
        if not path.exists():
            sys.exit(f"[{name}] file not found: {path}")
        df = _norm(pd.read_csv(path, dtype=str).fillna(""))
        need = {"question_id", "model_a", "model_b"}
        missing = need - set(df.columns)
        if missing:
            sys.exit(f"[{name}] {path} is missing required columns: {sorted(missing)}")
        if not any(d in df.columns for d in DIMS):
            sys.exit(f"[{name}] {path} has no dimension columns {DIMS}")
        if name in raters:
            sys.exit(f"Duplicate rater name {name!r} -- give one an explicit NAME=path")
        raters[name] = df
        kinds[name] = kind
        print(f"[{name}] {len(df)} rows ({kind}) from {path}")
    if len(raters) < 2:
        sys.exit("Need at least two raters to build an agreement matrix.")
    return raters, kinds


def check_orders(name: str, df: pd.DataFrame, kind: str) -> None:
    """Report canonical pairs covered in only one presentation order.

    Only a concern for judges: every judge should have been run over both orders
    (via flip_elo_pairs.py) so the collapse can detect position flips. A human
    annotating one order is expected and passes through unchanged.
    """
    if "position" not in df.columns:
        print(f"  [{name}] no position column -- cannot check order coverage")
        return
    lo = df[["model_a", "model_b"]].min(axis=1)
    hi = df[["model_a", "model_b"]].max(axis=1)
    orders: dict[tuple, set] = {}
    for k, pos in zip(zip(df["question_id"], lo, hi), df["position"]):
        orders.setdefault(k, set()).add(str(pos).strip().lower())
    single = sum(1 for v in orders.values() if len(v) < 2)
    if single == 0:
        flag = "  (both orders everywhere)"
    elif kind == "human":
        flag = "  (expected -- humans are assumed order-unbiased)"
    else:
        flag = "   <-- run flip_elo_pairs.py and re-judge into the same outdir"
    print(f"  [{name}] {len(orders)} canonical pairs, {single} with a single order{flag}")


# -- Labels --------------------------------------------------------------------

def add_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Majority vote across dimensions (mirrors merge_elo_annotations._overall).

    Empty unless all three dimensions carry a label -- a partial record has no
    well-defined majority.
    """
    df = df.copy()
    have = [d for d in DIMS if d in df.columns]

    def vote(row) -> str:
        vals = [row[d] for d in have]
        if len(have) < len(DIMS) or any(v == "" for v in vals):
            return ""
        c = Counter(vals)
        if c["A"] >= 2:
            return "A"
        if c["B"] >= 2:
            return "B"
        return "TIE"

    df["overall"] = df.apply(vote, axis=1) if len(df) else pd.Series(dtype=str)
    return df


def majority_rater(members: dict[str, pd.DataFrame], collapse: bool) -> pd.DataFrame:
    """Per-pair, per-dimension majority vote across the given (already collapsed) raters.

    A dimension with no strict majority -- a 3-way A/B/TIE split, or any tie for the
    top label -- resolves to TIE, the same "no winner was named" semantics the position
    collapse uses. Only pairs every member covers are emitted, so the pooled rater is
    never a mix of different underlying samples.
    """
    frames = list(members.values())
    key = [c for c in (["question_id", "model_a", "model_b", "source"] if collapse
                       else ["question_id", "model_a", "model_b", "position", "source"])
           if all(c in f.columns for f in frames)]

    merged = frames[0][key + [d for d in DIMS if d in frames[0].columns]].copy()
    merged.columns = key + [f"{d}__0" for d in DIMS if d in frames[0].columns]
    for n, f in enumerate(frames[1:], start=1):
        cols = [d for d in DIMS if d in f.columns]
        right = f[key + cols].copy()
        right.columns = key + [f"{d}__{n}" for d in cols]
        merged = merged.merge(right, on=key, how="inner")

    out = merged[key].copy()
    for d in DIMS:
        vote_cols = [c for c in merged.columns if c.startswith(f"{d}__")]
        if not vote_cols:
            out[d] = ""
            continue

        def vote(row) -> str:
            labs = [str(v) for v in row if str(v) in LABELS]
            if not labs:
                return ""
            counts = Counter(labs).most_common()
            top = counts[0][1]
            winners = [label for label, count in counts if count == top]
            return winners[0] if len(winners) == 1 else "TIE"

        out[d] = merged[vote_cols].apply(vote, axis=1) if len(merged) else ""
    return out


def prepare(raters: dict[str, pd.DataFrame], collapse: bool) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, df in raters.items():
        prepped = _resolve_collapse(df) if collapse else df.copy()
        out[name] = add_overall(prepped)
    return out


# -- Metrics -------------------------------------------------------------------

def cohen_kappa(t: list[str], p: list[str]) -> float:
    """Nominal Cohen's kappa over {A, TIE, B}."""
    n = len(t)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(t, p)) / n
    ct, cp = Counter(t), Counter(p)
    pe = sum((ct[label] / n) * (cp[label] / n) for label in LABELS)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def pair_metrics(t: list[str], p: list[str]) -> dict:
    n = len(t)
    if n == 0:
        return {"n": 0, "agreement": float("nan"), "cohen_kappa": float("nan"),
                "quad_kappa": float("nan"), "flip_rate": float("nan")}
    flips = {("A", "B"), ("B", "A")}
    agree = sum(a == b for a, b in zip(t, p))
    flip = sum((a, b) in flips for a, b in zip(t, p))
    return {
        "n": n,
        "agreement": agree / n,
        "cohen_kappa": cohen_kappa(t, p),
        "quad_kappa": quadratic_weighted_kappa(t, p),
        "flip_rate": flip / n,
    }


def join_key(a: pd.DataFrame, b: pd.DataFrame, collapse: bool) -> list[str]:
    cols = (["question_id", "model_a", "model_b", "source"] if collapse
            else ["question_id", "model_a", "model_b", "position", "source"])
    return [c for c in cols if c in a.columns and c in b.columns]


def build_long(prepped: dict[str, pd.DataFrame], collapse: bool) -> pd.DataFrame:
    names = list(prepped)
    rows: list[dict] = []
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            a, b = prepped[na], prepped[nb]
            key = join_key(a, b, collapse)
            merged = a.merge(b, on=key, how="inner", suffixes=("_x", "_y"))
            if merged.empty:
                print(f"WARNING: {na} vs {nb} matched 0 pairs on {key}", file=sys.stderr)
            for unit in UNITS:
                ca, cb = f"{unit}_x", f"{unit}_y"
                if ca not in merged or cb not in merged:
                    continue
                mask = (merged[ca] != "") & (merged[cb] != "")
                t = list(merged[ca][mask])
                p = list(merged[cb][mask])
                rows.append({"rater_a": na, "rater_b": nb, "dimension": unit,
                             **pair_metrics(t, p)})
    return pd.DataFrame(rows)


def to_matrix(long: pd.DataFrame, unit: str, metric: str, names: list[str]) -> np.ndarray:
    """Symmetric N x N matrix; diagonal is NaN (self-agreement is trivially perfect)."""
    m = np.full((len(names), len(names)), np.nan)
    idx = {n: i for i, n in enumerate(names)}
    sub = long[long["dimension"] == unit]
    for _, r in sub.iterrows():
        i, j = idx[r["rater_a"]], idx[r["rater_b"]]
        m[i, j] = m[j, i] = r[metric]
    return m


# -- Plotting ------------------------------------------------------------------

def _text_color(rgba) -> str:
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if lum < 0.5 else INK


def display_labels(names: list[str], kinds: dict[str, str]) -> list[str]:
    """Publication tick labels: humans are anonymized in the order given, and any
    Gemini judge variant collapses to the model family."""
    human_order = [n for n in names if kinds.get(n) == "human"]
    out = []
    for n in names:
        if n in human_order:
            out.append(f"Annotator {human_order.index(n) + 1}")
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
        print(f"WARNING: no finite values for {metric} -- skipping figure", file=sys.stderr)
        return

    # One fixed 0-1 ramp for every metric so panels and figures are comparable.
    # A sub-chance kappa clamps to the lightest blue; its printed value still shows
    # the sign.
    cmap = SEQ_CMAP
    norm = Normalize(vmin=0.0, vmax=1.0)

    n = len(names)
    side = max(3.1, 0.62 * n + 1.9)
    # Panels wrap at 2 columns (the 4-unit default is a 2x2); three or fewer sit
    # in a single row so dropping `overall` gives a 1x3 strip, not a ragged 2x2.
    ncols = len(units) if len(units) <= 3 else 2
    nrows = -(-len(units) // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * side, nrows * side * 1.04),
                             facecolor=SURFACE, constrained_layout=True,
                             squeeze=False)
    for ax in axes.ravel()[len(units):]:
        ax.set_visible(False)

    for ax, unit in zip(axes.ravel(), units):
        m = to_matrix(long, unit, metric, names)
        ax.set_facecolor(SURFACE)
        # 2px surface gap between cells (dataviz mark spec)
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
        label = "combined (majority vote)" if unit == "overall" else unit
        ax.set_title(label, fontsize=11, color=INK, pad=8)

        for i in range(n):
            for j in range(n):
                v = m[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color=_text_color(cmap(norm(v))))

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes.ravel().tolist(), shrink=0.55, pad=0.02)
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
    ap.add_argument("--human", action="append", metavar="NAME=CSV",
                    help="Human annotator truth CSV (merge_elo_annotations --truth). Repeatable.")
    ap.add_argument("--judge", action="append", metavar="NAME=CSV",
                    help="Explicit judge prediction CSV. Repeatable.")
    ap.add_argument("--pred-dir", help="Judge outdir; adds every predictions/*.csv as a rater")
    ap.add_argument("--no-combined", action="store_true",
                    help="Drop the `overall` (majority-vote) panel, leaving a 1x3 strip of "
                         "the three dimensions. The overall rows stay in the long CSV.")
    ap.add_argument("--prefix", default="rater_agreement",
                    help="Output path prefix (default: rater_agreement)")
    ap.add_argument("--no-position-collapse", action="store_true",
                    help="Compare raw positional rows instead of collapsing ab/ba per pair")
    ap.add_argument("--check-orders", action="store_true",
                    help="Report canonical pairs covered in only one presentation order")
    ap.add_argument("--pool-judges", action="store_true",
                    help="Add 'judges_majority': per-dimension majority vote across all "
                         "judges (no strict majority -> TIE)")
    ap.add_argument("--pool-humans", action="store_true",
                    help="Add 'humans_majority': majority vote across the --human raters "
                         "(needs 3+ humans; with 2 there is no majority)")
    args = ap.parse_args()

    collapse = not args.no_position_collapse
    raters, kinds = load_raters(args)

    if args.check_orders:
        print("\n=== Presentation-order coverage ===")
        for name, df in raters.items():
            check_orders(name, df, kinds[name])

    prepped = prepare(raters, collapse)

    for flag, kind, pooled_name in ((args.pool_judges, "judge", "judges_majority"),
                                    (args.pool_humans, "human", "humans_majority")):
        if not flag:
            continue
        members = {n: prepped[n] for n in prepped if kinds.get(n) == kind}
        need = 2 if kind == "judge" else 3
        if len(members) < need:
            sys.exit(f"--pool-{kind}s needs at least {need} {kind}s; got {len(members)}")
        pooled = add_overall(majority_rater(members, collapse))
        prepped[pooled_name] = pooled
        kinds[pooled_name] = kind
        print(f"[{pooled_name}] {len(pooled)} pairs (majority of {', '.join(members)})")

    names = list(prepped)
    long = build_long(prepped, collapse)
    if long.empty:
        sys.exit("No rater pairs produced metrics.")

    prefix = Path(args.prefix)
    csv_path = prefix.with_name(prefix.name + "_long.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}  ({len(long)} rater-pair x dimension rows)")

    print("\n=== Combined (majority vote) agreement, sorted by quadratic kappa ===")
    overall = long[long["dimension"] == "overall"].sort_values("quad_kappa", ascending=False)
    if not overall.empty:
        print(overall[["rater_a", "rater_b", "n", "agreement",
                       "cohen_kappa", "quad_kappa", "flip_rate"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    plot_units = DIMS if args.no_combined else UNITS
    labels = display_labels(names, kinds)
    for metric in ("quad_kappa", "cohen_kappa", "flip_rate"):
        plot_metric(long, names, labels, metric,
                    prefix.with_name(f"{prefix.name}_{metric}.png"),
                    units=plot_units)


if __name__ == "__main__":
    main()
