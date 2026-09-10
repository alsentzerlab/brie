"""
merge_fact_annotations.py

Rejoin a blinded fact-entailment annotation export (keyed only by record_id) with
the unblinding map written by sample_fact_entailment.py to recover identity
(question_id, source_name, model, inference type) and the per-fact LLM consensus.

The merged output mirrors the pre-blinding annotation schema, so downstream consumers
(build_fact_examples.py, any human-vs-LLM scoring comparison) work unchanged:
  {record_id, question_id, source_name, model, inference_type,
   recall_items:    [{idx, fact, entailed, llm_consensus_entailed, explanation}],
   precision_items: [{idx, fact, entailed, llm_consensus_entailed, explanation}]}

Usage:
  python merge_fact_annotations.py \
      --annotations ANNOTATIONS.json [more.json ...] \
      --map         annotate.map.json \
      --output      fact_annotations_full.json
"""

import argparse
import glob
import json
import sys
from pathlib import Path


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


def _merge_items(ann_items: list | None, map_items: list | None) -> list:
    """Join annotator marks (idx, entailed, note) with the map's LLM consensus by idx."""
    consensus = {it["idx"]: bool(it.get("consensus_entailed")) for it in (map_items or [])}
    out = []
    for it in (ann_items or []):
        idx = it.get("idx")
        out.append({
            "idx": idx,
            "fact": it.get("fact", ""),
            "entailed": it.get("entailed"),
            "llm_consensus_entailed": consensus.get(idx),
            "explanation": it.get("note") or None,
        })
    return out


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
        out.append({
            "record_id": rid,
            "question_id": rec.get("question_id", ""),
            "source_name": rec.get("source_name", ""),
            "model": rec.get("model", ""),
            "inference_type": rec.get("inference_type", ""),
            "recall_items":    _merge_items(ann.get("recall_items"),    rec.get("recall_items")),
            "precision_items": _merge_items(ann.get("precision_items"), rec.get("precision_items")),
        })
    if missing:
        print(f"WARNING: {missing} annotations had no matching record_id in the map (skipped)")
    print(f"Merged {len(out)} annotated comparisons")
    return out


def main():
    ap = argparse.ArgumentParser(description="Merge blinded fact annotations with the unblinding map.")
    ap.add_argument("--annotations", nargs="+", required=True,
                    help="fact_annotations_*.json export(s), globs, or directories")
    ap.add_argument("--map", required=True, help="annotate.map.json from sample_fact_entailment.py")
    ap.add_argument("--output", default="fact_annotations_full.json",
                    help="Full merged annotation JSON (default: fact_annotations_full.json)")
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
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote merged annotations → {out}")
    print(f"\nNext: python build_fact_examples.py --annotations {out}")


if __name__ == "__main__":
    main()
