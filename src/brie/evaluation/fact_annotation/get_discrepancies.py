"""
compare_annotations.py

Compare two annotators' exported annotation files and surface every disagreement.

Auto-detects the platform from the record schema:
  - ELO pairwise   : records with completeness/relevancy/concision = {winner, note}
                     → compared per dimension (A / B / TIE)
  - Fact entailment: records with recall_items / precision_items = [{idx, fact, entailed}]
                     → compared per fact (by idx) in each direction
                       (recall = reference facts, precision = answer facts), Present/Absent

Records are matched by `record_id`. Free-text notes are ignored — only the labels are
compared. Reports: coverage (records only one annotator completed), per-unit agreement
rate and Cohen's kappa, exact full-record agreement, and the full list of discrepancies.
`--csv` dumps every discrepancy as one row for review.

Each side may be a single .json export or a directory/glob of them (merged by record_id;
on a duplicate record_id with conflicting labels within one side, the last one wins and a
warning is printed).

Usage:
  python compare_annotations.py annotator1.json annotator2.json
  python compare_annotations.py alice/ bob/ --labels alice,bob --csv discrepancies.csv
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

ELO_DIMS = ("completeness", "relevancy", "concision")


# ── Loading ───────────────────────────────────────────────────────────────────

def _expand(pattern: str) -> list[Path]:
    pattern = os.path.expanduser(os.path.expandvars(pattern))  # resolve literal ~ and $VARS
    p = Path(pattern)
    if p.is_dir():
        return sorted(p.glob("*.json"))
    matched = [Path(m) for m in glob.glob(pattern)]
    return sorted(matched) if matched else [p]


def load_side(pattern: str, label: str) -> dict:
    """Merge one annotator's export(s) into {record_id: record}."""
    records: dict[str, dict] = {}
    files = _expand(pattern)
    if not files:
        sys.exit(f"[{label}] no files matched {pattern!r}")
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
                print(f"  [{label}] WARNING: a record in {path.name} has no record_id — skipped")
                continue
            if rid in records and records[rid] != rec:
                print(f"  [{label}] WARNING: duplicate record_id {rid} with differing content — keeping last")
            records[rid] = rec
    print(f"[{label}] loaded {len(records)} records from {len(files)} file(s)")
    return records


def detect_type(records: dict) -> str:
    for rec in records.values():
        if "recall_items" in rec or "precision_items" in rec:
            return "fact"
        if any(d in rec for d in ELO_DIMS):
            return "elo"
    sys.exit("Could not detect annotation type (no ELO dimensions or fact item lists found).")


# ── Cohen's kappa ─────────────────────────────────────────────────────────────

def cohen_kappa(pairs: list[tuple]) -> float:
    """Cohen's kappa over a list of (label_a, label_b). NaN if undefined."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    labels = sorted({label for pair in pairs for label in pair})
    obs = sum(a == b for a, b in pairs) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    exp = sum((ca[label] / n) * (cb[label] / n) for label in labels)
    if exp == 1.0:
        return float("nan")  # everyone used one label → kappa undefined
    return (obs - exp) / (1 - exp)


def _agree_line(name: str, pairs: list[tuple]) -> str:
    n = len(pairs)
    if n == 0:
        return f"  {name:<26} (no overlapping items)"
    agree = sum(a == b for a, b in pairs)
    k = cohen_kappa(pairs)
    kstr = "n/a " if k != k else f"{k:.3f}"
    return f"  {name:<26} {agree}/{n} agree ({agree/n*100:5.1f}%)   kappa={kstr}"


# ── ELO comparison ────────────────────────────────────────────────────────────

def _winner(rec: dict, dim: str):
    d = rec.get(dim) or {}
    w = str(d.get("winner") or "").strip().upper()
    return w if w in ("A", "B", "TIE") else None


def compare_elo(a: dict, b: dict, la: str, lb: str) -> list[dict]:
    common = sorted(set(a) & set(b))
    discrepancies: list[dict] = []
    per_dim: dict[str, list[tuple]] = {d: [] for d in ELO_DIMS}
    exact_records = 0

    for rid in common:
        ra, rb = a[rid], b[rid]
        rec_agree = True
        for dim in ELO_DIMS:
            wa, wb = _winner(ra, dim), _winner(rb, dim)
            if wa is not None and wb is not None:
                per_dim[dim].append((wa, wb))
            if wa != wb:
                rec_agree = False
                discrepancies.append({
                    "type": "elo", "record_id": rid,
                    "question_id": ra.get("question_id", ""),
                    "context": f'{ra.get("model_a","?")} vs {ra.get("model_b","?")}',
                    "position": ra.get("position", ""),
                    "unit": dim, "idx": "", "fact": "",
                    la: wa or "(unset)", lb: wb or "(unset)",
                })
        exact_records += rec_agree

    print("\n=== ELO agreement (per dimension, A/B/TIE) ===")
    all_pairs = []
    for dim in ELO_DIMS:
        print(_agree_line(dim, per_dim[dim]))
        all_pairs += per_dim[dim]
    print(_agree_line("ALL dimensions pooled", all_pairs))
    if common:
        print(f"  exact full-record agreement: {exact_records}/{len(common)} "
              f"({exact_records/len(common)*100:.1f}%)")
    return discrepancies


# ── Fact comparison ───────────────────────────────────────────────────────────

def _entailed_map(rec: dict, key: str) -> dict:
    """idx -> (entailed, fact) for one direction."""
    out = {}
    for it in rec.get(key) or []:
        out[it.get("idx")] = (it.get("entailed"), it.get("fact", ""))
    return out


def _lab(v):
    return "Present" if v is True else "Absent" if v is False else "(unmarked)"


def compare_fact(a: dict, b: dict, la: str, lb: str) -> list[dict]:
    common = sorted(set(a) & set(b))
    discrepancies: list[dict] = []
    per_dir = {"recall_items": [], "precision_items": []}
    exact_records = 0

    for rid in common:
        ra, rb = a[rid], b[rid]
        rec_agree = True
        for key, dname in (("recall_items", "recall"), ("precision_items", "precision")):
            ma, mb = _entailed_map(ra, key), _entailed_map(rb, key)
            for idx in sorted(set(ma) & set(mb), key=lambda x: (x is None, x)):
                va, fact = ma[idx]
                vb, _ = mb[idx]
                if va is not None and vb is not None:
                    per_dir[key].append((bool(va), bool(vb)))
                if va != vb:
                    rec_agree = False
                    discrepancies.append({
                        "type": "fact", "record_id": rid,
                        "question_id": ra.get("question_id", ""),
                        "context": ra.get("source_name", ""),
                        "position": "", "unit": dname, "idx": idx, "fact": fact,
                        la: _lab(va), lb: _lab(vb),
                    })
            # facts one side marked but the other lacks entirely (rare; schema mismatch)
            for idx in set(ma) ^ set(mb):
                rec_agree = False
                src = la if idx in ma else lb
                fact = (ma.get(idx) or mb.get(idx) or (None, ""))[1]
                discrepancies.append({
                    "type": "fact", "record_id": rid,
                    "question_id": ra.get("question_id", ""),
                    "context": ra.get("source_name", ""),
                    "position": "", "unit": f"{dname} (only in {src})",
                    "idx": idx, "fact": fact,
                    la: _lab(ma.get(idx, (None,))[0]) if idx in ma else "(missing)",
                    lb: _lab(mb.get(idx, (None,))[0]) if idx in mb else "(missing)",
                })
        exact_records += rec_agree

    print("\n=== Fact agreement (per direction, Present/Absent) ===")
    pooled = []
    for key, dname in (("recall_items", "recall"), ("precision_items", "precision")):
        print(_agree_line(dname, per_dir[key]))
        pooled += per_dir[key]
    print(_agree_line("BOTH directions pooled", pooled))
    if common:
        print(f"  exact full-card agreement: {exact_records}/{len(common)} "
              f"({exact_records/len(common)*100:.1f}%)")
    return discrepancies


# ── Report ────────────────────────────────────────────────────────────────────

def coverage(a: dict, b: dict, la: str, lb: str):
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    print("\n=== Coverage ===")
    print(f"  in both: {len(set(a) & set(b))}")
    if only_a:
        print(f"  only in {la} ({len(only_a)}): {', '.join(only_a[:8])}{' …' if len(only_a) > 8 else ''}")
    if only_b:
        print(f"  only in {lb} ({len(only_b)}): {', '.join(only_b[:8])}{' …' if len(only_b) > 8 else ''}")
    if not only_a and not only_b:
        print("  both annotators covered the same record set ✓")


def print_discrepancies(discs: list[dict], la: str, lb: str):
    print(f"\n=== Discrepancies ({len(discs)}) ===")
    if not discs:
        print("  none — the annotators agree on every compared label 🎉")
        return
    last_rid = None
    for d in discs:
        if d["record_id"] != last_rid:
            ctx = d["context"] + (f'  [{d["position"]}]' if d["position"] else "")
            print(f'\n  • {d["record_id"]}  (qid {d["question_id"]})  {ctx}')
            last_rid = d["record_id"]
        loc = d["unit"] + (f' #{d["idx"]}' if d["idx"] != "" else "")
        line = f'      {loc:<22} {la}={d[la]:<11} {lb}={d[lb]}'
        print(line)
        if d["fact"]:
            print(f'          ↳ {d["fact"]}')


def write_csv(discs: list[dict], path: Path, la: str, lb: str):
    cols = ["type", "record_id", "question_id", "context", "position", "unit", "idx", "fact", la, lb]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for d in discs:
            w.writerow({c: d.get(c, "") for c in cols})
    print(f"\nWrote {len(discs)} discrepancies → {path}")


def main():
    ap = argparse.ArgumentParser(description="Compare two annotators' exports and surface disagreements.")
    ap.add_argument("file_a", help="Annotator A export (.json), directory, or glob")
    ap.add_argument("file_b", help="Annotator B export (.json), directory, or glob")
    ap.add_argument("--labels", help="Comma-separated names for the two annotators (default: file stems)")
    ap.add_argument("--csv", help="Write all discrepancies to this CSV")
    args = ap.parse_args()

    if args.labels:
        la, lb = (s.strip() for s in args.labels.split(",", 1))
    else:
        la, lb = Path(args.file_a).stem, Path(args.file_b).stem
        if la == lb:
            la, lb = la + "_A", lb + "_B"

    a = load_side(args.file_a, la)
    b = load_side(args.file_b, lb)

    ta, tb = detect_type(a), detect_type(b)
    if ta != tb:
        sys.exit(f"Type mismatch: {la} looks like '{ta}' but {lb} looks like '{tb}'.")
    print(f"Detected annotation type: {ta}")

    coverage(a, b, la, lb)
    discs = compare_elo(a, b, la, lb) if ta == "elo" else compare_fact(a, b, la, lb)

    print_discrepancies(discs, la, lb)
    if args.csv:
        write_csv(discs, Path(args.csv), la, lb)


if __name__ == "__main__":
    main()
