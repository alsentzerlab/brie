#!/usr/bin/env python
"""Score each LLM judge against the human ELO annotations, accounting for ties.

Compares the per-judge predictions written by ``judge_elo_pairs.py`` against a
ground-truth CSV of human labels (A / B / TIE per dimension) and reports, per
judge x dimension, a set of tie-aware agreement metrics plus the full 3x3
confusion matrix.

Why tie-aware: the human annotation was unblinded to gemini-2.5-flash-lite, so
the annotator's labels are anchored to that model — most visibly on ties (a
true toss-up nudged toward whatever side the model named). A flat exact-match
accuracy charges every tie disagreement as a full miss; these metrics separate
*directional* errors (A<->B flips) from *adjacent* tie disagreements.

Position collapse (default on): every pair is shown in both presentation orders
(position ab and ba). Each side's positional A/B winner is resolved to the
actual model name and the two orders are combined per pair — if they disagree
(an A<->B flip, or one order commits while the other ties) the verdict becomes
TIE, since a position-flipping judgment names no real winner. The collapsed
verdict is re-expressed in a fixed orientation (lexicographically smaller model
= slot A) so the metrics below compare *model identity*, not display position.
This is applied symmetrically to the human truth and to every judge. Pass
--no-position-collapse to score the raw positional rows instead.

Per judge x dimension (over pairs where both sides have a non-empty label):
  strict_acc          exact match  (TIE mismatch counts as wrong)
  adjacent_rate       exactly one side said TIE
  flip_rate (DER)     A<->B directional error rate  (the errors that matter)
  tolerant_acc        1 - flip_rate  (TIE treated as a wildcard)
  committed_dir_acc   agreement among rows where BOTH picked a side (no TIE)
  quad_kappa          quadratic-weighted Cohen's kappa on ordinal A < TIE < B
By construction: strict_acc + adjacent_rate + flip_rate = 1.

Outputs:
  --output (judge_accuracy.csv)   one row per judge x dimension, metrics above
  --confusion (judge_confusion.csv)  3x3 cell counts + flip split + tie asymmetry
  --transitions (judge_transitions.csv)  per judge x dimension label-transition
        matrix P(judge = Y | human = X), arrow read human -> judge: probability
        columns 'A→A','A→TIE','A→B','TIE→A',...,'B→B' (each from-label's three
        sum to 1), the matching 'n_A→B' counts, and n_from_{A,TIE,B}
        denominators. Feed this to a perturbation pipeline as the judge's noise
        model. --transition-smoothing ALPHA applies Laplace smoothing.

Ground-truth CSV columns:
    question_id[, model_a, model_b, position, source], completeness, relevancy, concision
Join uses whichever key columns are present in BOTH truth and predictions
(always question_id).

Usage
-----
    python compare_judge_accuracy.py --truth human.csv --pred-dir elo_eval
    python compare_judge_accuracy.py --truth human.csv \
        --predictions elo_eval/predictions/claude-opus-4-7.csv
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

KEY_COLS = ["question_id", "model_a", "model_b", "position", "source"]
DIMS = ["completeness", "relevancy", "concision"]

# Ordinal positions for weighted kappa: a flip (A<->B) is distance 2, a
# tie-adjacent disagreement is distance 1.
ORD = {"A": 0, "TIE": 1, "B": 2}
LABELS = ["A", "TIE", "B"]


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize key + label columns (str, stripped; labels upper-cased).

    Only touches columns that are present, so a file missing an optional key
    column (e.g. 'source') is handled gracefully. 'T' is folded to 'TIE'.
    """
    df = df.copy()
    for c in KEY_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "source" in df.columns:
        df["source"] = df["source"].str.replace(r"\.html?$", "", regex=True)
    for d in DIMS:
        if d in df.columns:
            lab = df[d].astype(str).str.strip().str.upper()
            df[d] = lab.replace({"T": "TIE", "TIED": "TIE"})
    return df


def _winning_model(label: pd.Series, model_a: pd.Series, model_b: pd.Series) -> pd.Series:
    """Map a positional label (A/B/TIE/'') to the winning model name.

    A -> model_a, B -> model_b; TIE and '' (no judgment) pass through unchanged.
    """
    out = label.mask(label == "A", model_a)
    return out.mask(label == "B", model_b)


def _resolve_collapse(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the two presentation orders (ab/ba) of each pair into one verdict.

    For every dimension the positional A/B winner is resolved to the actual model
    name, then all rows sharing a canonical pair key (question_id + the unordered
    model pair [+ source]) are combined:

      * all orders name the same model    -> that model wins
      * orders disagree (A<->B flip, or one
        order commits while the other ties) -> TIE  (position bias => no winner)

    Output rows carry one verdict per pair, re-expressed in a fixed orientation:
    the lexicographically smaller model is slot 'A', the larger is slot 'B'.
    """
    df = df.copy()
    df["_lo"] = df[["model_a", "model_b"]].min(axis=1)
    df["_hi"] = df[["model_a", "model_b"]].max(axis=1)
    for d in DIMS:
        if d in df.columns:
            df[f"_win_{d}"] = _winning_model(
                pd.Series(df[d]), pd.Series(df["model_a"]), pd.Series(df["model_b"]))

    key = ["question_id", "_lo", "_hi"] + (["source"] if "source" in df.columns else [])
    records: list[dict] = []
    for keyvals, grp in df.groupby(key, sort=False):
        keyvals = keyvals if isinstance(keyvals, tuple) else (keyvals,)
        rec = dict(zip(key, keyvals))
        model_a, model_b = rec.pop("_lo"), rec.pop("_hi")
        rec["model_a"], rec["model_b"] = model_a, model_b
        for d in DIMS:
            rec[d] = ""
            wcol = f"_win_{d}"
            if wcol not in grp.columns:
                continue
            votes = {w for w in grp[wcol].tolist() if w != ""}
            if not votes:
                lab = ""
            elif len(votes) == 1:
                lab = next(iter(votes))           # unanimous: a model name or TIE
            else:
                lab = "TIE"                         # any disagreement -> TIE
            rec[d] = "A" if lab == model_a else "B" if lab == model_b else lab
        records.append(rec)

    cols = (["question_id", "model_a", "model_b"]
            + (["source"] if "source" in df.columns else []) + DIMS)
    return pd.DataFrame(records, columns=cols).fillna("")


def quadratic_weighted_kappa(true: list[str], pred: list[str]) -> float:
    """Quadratic-weighted Cohen's kappa over the ordinal scale A < TIE < B."""
    n = len(true)
    if n == 0:
        return float("nan")
    cats = LABELS
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    observed = [[0.0] * k for _ in range(k)]
    for t, p in zip(true, pred):
        observed[idx[t]][idx[p]] += 1
    row_tot = [sum(observed[i]) for i in range(k)]
    col_tot = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    denom_w = (k - 1) ** 2
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / denom_w
            e = row_tot[i] * col_tot[j] / n
            num += w * observed[i][j]
            den += w * e
    if den == 0:  # one side is constant -> no expected disagreement to correct
        return float("nan")
    return 1.0 - num / den


def _dim_metrics(true: pd.Series, pred: pd.Series) -> dict:
    """Tie-aware agreement metrics for one judge x dimension."""
    mask = (true != "") & (pred != "")
    t = list(true[mask])
    p = list(pred[mask])
    n = len(t)
    if n == 0:
        return {"n": 0}

    flips = {("A", "B"), ("B", "A")}
    agree = sum(a == b for a, b in zip(t, p))
    flip = sum((a, b) in flips for a, b in zip(t, p))
    adjacent = n - agree - flip

    committed = [(a, b) for a, b in zip(t, p) if a != "TIE" and b != "TIE"]
    committed_agree = sum(a == b for a, b in committed)

    return {
        "n":                 n,
        "strict_acc":        round(agree / n, 4),
        "adjacent_rate":     round(adjacent / n, 4),
        "flip_rate":         round(flip / n, 4),          # DER
        "tolerant_acc":      round((agree + adjacent) / n, 4),
        "committed_n":       len(committed),
        "committed_dir_acc": round(committed_agree / len(committed), 4) if committed else float("nan"),
        "quad_kappa":        round(quadratic_weighted_kappa(t, p), 4),
    }


def _dim_confusion(true: pd.Series, pred: pd.Series) -> dict:
    """3x3 cell counts plus directional-flip split and tie asymmetry."""
    mask = (true != "") & (pred != "")
    pairs = Counter(zip(true[mask], pred[mask]))
    cells = {f"annot{a}_model{b}": pairs.get((a, b), 0) for a in LABELS for b in LABELS}
    cells["flip_annotA_modelB"] = pairs.get(("A", "B"), 0)
    cells["flip_annotB_modelA"] = pairs.get(("B", "A"), 0)
    # Tie asymmetry: who 'breaks' the tie. Anchoring -> annotator follows the
    # model, so annot_commit_model_tie tends to be small for the anchor model.
    cells["annot_commit_model_tie"] = (
        pairs.get(("A", "TIE"), 0) + pairs.get(("B", "TIE"), 0))
    cells["annot_tie_model_commit"] = (
        pairs.get(("TIE", "A"), 0) + pairs.get(("TIE", "B"), 0))
    return cells


def _dim_transitions(true: pd.Series, pred: pd.Series, alpha: float = 0.0) -> dict:
    """Label-transition (noise) model for one judge x dimension.

    Estimates P(judge = Y | human = X), arrow read human -> judge, so each
    from-label's three probabilities form a row of a row-stochastic matrix that a
    perturbation pipeline can apply to the human labels.

    With alpha > 0, additive (Laplace) smoothing is applied:
        P(X->Y) = (count(X,Y) + alpha) / (count(X,.) + alpha * 3)
    so unobserved transitions get nonzero mass and a from-label never seen yields
    a uniform row instead of NaN. With alpha == 0 a never-seen from-label gives
    NaN for that row (no empirical evidence).
    """
    mask = (true != "") & (pred != "")
    counts = Counter(zip(true[mask], pred[mask]))
    rec: dict = {"n": int(mask.sum())}
    k = len(LABELS)
    for a in LABELS:
        from_n = sum(counts.get((a, b), 0) for b in LABELS)
        rec[f"n_from_{a}"] = from_n
        denom = from_n + alpha * k
        for b in LABELS:
            c = counts.get((a, b), 0)
            rec[f"n_{a}→{b}"] = c
            rec[f"{a}→{b}"] = round((c + alpha) / denom, 6) if denom > 0 else float("nan")
    return rec


# Column order for the transitions file: metadata, then the 3x3 matrix read
# row-major over the from-label (A row, TIE row, B row), probs then counts.
_TRANS_PROB_COLS = [f"{a}→{b}" for a in LABELS for b in LABELS]
_TRANS_COUNT_COLS = [f"n_{a}→{b}" for a in LABELS for b in LABELS]
_TRANS_COLS = (["judge", "dimension", "n", "n_from_A", "n_from_TIE", "n_from_B"]
               + _TRANS_PROB_COLS + _TRANS_COUNT_COLS)


def main(args: argparse.Namespace) -> None:
    collapse = not args.no_position_collapse
    truth = _norm(pd.read_csv(args.truth, dtype=str).fillna(""))
    required = {"question_id", *DIMS}
    if collapse:
        required |= {"model_a", "model_b"}  # needed to resolve positional winners
    missing = required - set(truth.columns)
    if missing:
        sys.exit(f"--truth is missing required columns: {sorted(missing)}")

    # Position collapse maps the ab/ba orders of each pair to a single verdict and
    # drops 'position'; the remaining canonical key joins truth to predictions.
    join_cols = ["question_id", "model_a", "model_b", "source"] if collapse else KEY_COLS
    if collapse:
        truth = _resolve_collapse(truth)
    dropped = [c for c in join_cols if c not in truth.columns]
    if dropped:
        print(f"NOTE: --truth has no {dropped} column(s); joining on "
              f"{[c for c in join_cols if c in truth.columns]} only. Make sure "
              "that uniquely identifies each pair.", file=sys.stderr)

    if args.predictions:
        pred_files = [Path(p) for p in args.predictions]
    else:
        pred_dir = Path(args.pred_dir)
        search = pred_dir / "predictions" if (pred_dir / "predictions").is_dir() else pred_dir
        pred_files = sorted(search.glob("*.csv"))
    if not pred_files:
        sys.exit("No prediction CSVs found.")

    metric_rows: list[dict] = []
    confusion_rows: list[dict] = []
    transition_rows: list[dict] = []
    for pf in pred_files:
        pred = _norm(pd.read_csv(pf, dtype=str).fillna(""))
        if "question_id" not in pred.columns:
            print(f"WARNING: {pf.name} has no question_id — skipping", file=sys.stderr)
            continue
        judge = pf.stem
        if collapse:
            if not {"model_a", "model_b"}.issubset(pred.columns):
                print(f"WARNING: {judge} lacks model_a/model_b for position collapse "
                      "— skipping", file=sys.stderr)
                continue
            pred = _resolve_collapse(pred)
        key = [c for c in join_cols if c in truth.columns and c in pred.columns]
        merged = truth.merge(pred, on=key, how="inner", suffixes=("_true", "_pred"))
        if merged.empty:
            print(f"WARNING: {judge} had 0 matched pairs on key {key} — skipping",
                  file=sys.stderr)
            continue
        for d in DIMS:
            if f"{d}_true" not in merged or f"{d}_pred" not in merged:
                continue
            t = pd.Series(merged[f"{d}_true"]).astype(str)
            p = pd.Series(merged[f"{d}_pred"]).astype(str)
            metric_rows.append({"judge": judge, "dimension": d, "n_matched": len(merged),
                                **_dim_metrics(t, p)})
            confusion_rows.append({"judge": judge, "dimension": d,
                                   **_dim_confusion(t, p)})
            transition_rows.append({"judge": judge, "dimension": d,
                                    **_dim_transitions(t, p, args.transition_smoothing)})

    if not metric_rows:
        sys.exit("No comparable prediction files.")

    metrics = pd.DataFrame(metric_rows)
    confusion = pd.DataFrame(confusion_rows)

    # Per-judge macro average across dimensions (for ranking / quick read).
    macro_cols = ["strict_acc", "flip_rate", "tolerant_acc",
                  "committed_dir_acc", "quad_kappa"]
    macro_records = []
    for judge, grp in metrics.groupby("judge"):
        rec = {"judge": judge}
        for c in macro_cols:
            rec[c] = round(float(pd.Series(grp[c]).mean()), 4)
        macro_records.append(rec)
    macro = pd.DataFrame(macro_records).sort_values("flip_rate")  # fewest flips first

    show = ["judge", "dimension", "n", "strict_acc", "adjacent_rate",
            "flip_rate", "tolerant_acc", "committed_dir_acc", "quad_kappa"]
    print("\n=== Per judge x dimension (tie-aware) ===")
    print(metrics[show].to_string(index=False))
    print("\n=== Per judge (macro over dimensions, sorted by flip_rate) ===")
    print(macro.to_string(index=False))
    print("\n=== Directional flips & tie asymmetry (per judge x dimension) ===")
    cshow = ["judge", "dimension", "flip_annotA_modelB", "flip_annotB_modelA",
             "annot_commit_model_tie", "annot_tie_model_commit"]
    print(confusion[cshow].to_string(index=False))

    out = Path(args.output)
    metrics.to_csv(out, index=False)
    conf_out = Path(args.confusion)
    confusion.to_csv(conf_out, index=False)

    transitions = pd.DataFrame(transition_rows).reindex(columns=_TRANS_COLS)
    trans_out = Path(args.transitions)
    transitions.to_csv(trans_out, index=False, encoding="utf-8")
    print(f"\nWrote metrics -> {out}\nWrote confusion -> {conf_out}"
          f"\nWrote transitions -> {trans_out} "
          f"(P(judge|human), smoothing alpha={args.transition_smoothing})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True, help="Human ground-truth CSV")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pred-dir", default="elo_eval",
                   help="Dir containing predictions/*.csv (default: elo_eval)")
    g.add_argument("--predictions", nargs="+", default=None,
                   help="Explicit prediction CSV paths")
    ap.add_argument("--output", default="judge_accuracy.csv",
                    help="Per judge x dimension metrics CSV (default: judge_accuracy.csv)")
    ap.add_argument("--confusion", default="judge_confusion.csv",
                    help="Confusion-matrix CSV (default: judge_confusion.csv)")
    ap.add_argument("--transitions", default="judge_transitions.csv",
                    help="Per judge x dimension label-transition matrix P(judge|human) "
                         "CSV: prob cols 'A→B' ... plus 'n_A→B' counts and n_from_* "
                         "denominators (default: judge_transitions.csv)")
    ap.add_argument("--transition-smoothing", type=float, default=0.0, metavar="ALPHA",
                    help="Additive (Laplace) smoothing for the transition matrix so "
                         "unobserved transitions get nonzero mass (default: 0.0 = raw "
                         "empirical; NaN row for a from-label never seen)")
    ap.add_argument("--no-position-collapse", action="store_true",
                    help="Score raw positional rows instead of collapsing the ab/ba "
                         "orders of each pair to a single (model-identity) verdict.")
    main(ap.parse_args())
