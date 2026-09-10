"""
merge_elo_annotations.py

Rejoin a blinded ELO annotation export (keyed only by record_id) with the
unblinding map written by sample_elo_pairs.py to recover identity (question_id,
model_a/model_b, inference type, position) and the LLM verdicts.

Produces two outputs:
  --output  full annotation JSON — one record per annotated pair, merging the map's
            identity/LLM fields with the human winner+note per dimension and a derived
            overall_winner. Consumed by build_elo_examples.py.
  --truth   ground-truth CSV with columns
              question_id, model_a, model_b, position, source,
              completeness, relevancy, concision
            (positional A/B/TIE labels) — the exact --truth schema expected by
            compare_judge_accuracy.py.

The annotation export carries no model identity by design (blinded study); this
merge is the only place identity is re-attached, via the private map.

Usage:
  python merge_elo_annotations.py \
      --annotations ANNOTATIONS.json [more.json ...] \
      --map         annotate.map.json \
      --output      elo_annotations_full.json \
      --truth       elo_truth.csv
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

DIMS = ("completeness", "relevancy", "concision")
DEFAULT_OUTPUT_DIR = Path("evaluation_output")


def _norm_winner(w) -> str:
    w = str(w or "").strip().upper()
    if w in ("T", "TIED"):
        w = "TIE"
    return w if w in ("A", "B", "TIE") else ""


def _overall(verdict: dict) -> str:
    """Majority vote across dimensions (mirrors score_elo_batch._parse_judgment)."""
    counts = {"A": 0, "B": 0, "TIE": 0}
    for dim in DIMS:
        counts[verdict[dim]["winner"] or "TIE"] += 1
    if counts["A"] >= 2:
        return "A"
    if counts["B"] >= 2:
        return "B"
    return "TIE"


def _expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pat in patterns:
        p = Path(pat)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.json")))
        else:
            matched = [Path(m) for m in glob.glob(pat)]
            paths.extend(sorted(matched) if matched else [p])
    return paths


def load_annotations(patterns: list[str]) -> list[dict]:
    records: list[dict] = []
    for path in _expand(patterns):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"  WARNING: {path} is not a JSON list — skipping")
            continue
        records.extend(data)
        print(f"  {path}: {len(data)} annotations")
    print(f"Loaded {len(records)} annotations")
    return records


def merge(annotations: list[dict], amap: dict) -> list[dict]:
    out: list[dict] = []
    missing = 0
    seen: set[str] = set()
    for ann in annotations:
        rid = ann.get("record_id")
        if not rid or rid in seen:
            continue
        rec = amap.get(rid)
        if rec is None:
            missing += 1
            continue
        seen.add(rid)

        verdict = {}
        for dim in DIMS:
            d = ann.get(dim) or {}
            verdict[dim] = {"winner": _norm_winner(d.get("winner")), "note": d.get("note") or ""}

        merged = dict(rec)  # identity + LLM fields + response text from the map
        for dim in DIMS:
            merged[dim] = verdict[dim]
        merged["overall_winner_human"] = _overall(verdict)
        out.append(merged)

    if missing:
        print(f"WARNING: {missing} annotations had no matching record_id in the map (skipped)")
    print(f"Merged {len(out)} annotated pairs")
    return out


def write_truth(merged: list[dict], path: Path) -> None:
    cols = ["question_id", "model_a", "model_b", "position", "source"] + list(DIMS)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in merged:
            source = Path(str(r.get("source_file", ""))).stem
            w.writerow({
                "question_id": r.get("question_id", ""),
                "model_a": r.get("model_a", ""),
                "model_b": r.get("model_b", ""),
                "position": r.get("position", ""),
                "source": source,
                **{dim: r[dim]["winner"] for dim in DIMS},
            })
    print(f"Wrote truth CSV → {path}  ({len(merged)} rows)")


def main():
    ap = argparse.ArgumentParser(description="Merge blinded ELO annotations with the unblinding map.")
    ap.add_argument("--annotations", nargs="+", required=True,
                    help="elo_annotations_*.json export(s), globs, or directories")
    ap.add_argument("--map", required=True, help="annotate.map.json from sample_elo_pairs.py")
    ap.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR / "elo_annotations_full.json",
        help="Full merged annotation JSON (default: protected evaluation data directory)",
    )
    ap.add_argument(
        "--truth",
        default=DEFAULT_OUTPUT_DIR / "elo_truth.csv",
        help="Ground-truth CSV (default: protected evaluation data directory)",
    )
    args = ap.parse_args()

    with open(args.map, encoding="utf-8") as f:
        amap = json.load(f)
    if not isinstance(amap, dict):
        sys.exit("--map must be a JSON object keyed by record_id.")
    print(f"Loaded map with {len(amap)} records from {args.map}")

    annotations = load_annotations(args.annotations)
    merged = merge(annotations, amap)
    if not merged:
        sys.exit("No annotations matched the map. Nothing written.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote merged annotations → {out}")
    truth = Path(args.truth)
    truth.parent.mkdir(parents=True, exist_ok=True)
    write_truth(merged, truth)
    print(f"\nNext: python compare_judge_accuracy.py --truth {args.truth} --pred-dir <elo_eval>")
    print(f"      python build_elo_examples.py --annotations {out} --records <sample.html|records.json>")


if __name__ == "__main__":
    main()
