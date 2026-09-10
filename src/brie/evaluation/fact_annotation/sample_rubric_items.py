"""
sample_rubric_items.py

Sample scored rubric responses, join in question text and candidate responses,
and generate a self-contained HTML annotation file with pre-embedded data.

Usage:
  python sample_rubric_items.py \
      --scores    $EVAL/rubric/rubric_scores.csv \
      --rubrics   $EVAL/rubric/rubric_items.csv \
      --questions $EVAL/phase2_qa.csv \
      --responses $EVAL/answer_cleaned/ \
      --n 50 --seed 42 \
      --output    $EVAL/rubric/annotate.html

Each output record contains one (question, source_name, model) scored response
with all rubric items and per-judge verdicts pre-joined.

Sampling is stratified by avg_rubric_score tercile (low/mid/high) so the
annotation set covers the full performance range.
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

_KNOWN_METHODS = {"bm25", "dense", "late", "recent", "recent200"}


# ── Load scores ───────────────────────────────────────────────────────────────

def load_scores(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["question_id"] = df["question_id"].astype(str)
    print(f"Loaded {len(df)} score rows from {path}")
    return df


# ── Load rubrics ──────────────────────────────────────────────────────────────

def load_rubrics(path: str) -> dict:
    """Returns {question_id: [item_dict, ...]}"""
    df = pd.read_csv(path)
    df["question_id"] = df["question_id"].astype(str)
    lookup = {}
    for _, row in df.iterrows():
        qid = str(row["question_id"])
        try:
            rubric = json.loads(row["rubric"]) if isinstance(row["rubric"], str) else row["rubric"]
            lookup[qid] = rubric.get("items", [])
        except (json.JSONDecodeError, AttributeError):
            lookup[qid] = []
    print(f"Loaded rubrics for {len(lookup)} questions from {path}")
    return lookup


# ── Load questions ────────────────────────────────────────────────────────────

def load_questions(path: str) -> dict:
    df = pd.read_csv(path)
    df["question_id"] = df["question_id"].astype(str)
    q_col = next((c for c in ("question", "natural_query") if c in df.columns), None)
    lookup = {}
    for _, row in df.iterrows():
        lookup[str(row["question_id"])] = str(row[q_col]) if q_col and pd.notna(row.get(q_col)) else ""
    print(f"Loaded {len(lookup)} questions from {path}")
    return lookup


# ── Load responses ────────────────────────────────────────────────────────────

def _infer_tag(path: Path) -> str:
    parts = path.stem.split("_")
    if parts and parts[-1] == "sample":
        parts = parts[:-1]
    return parts[-1] if parts else path.stem


def load_responses(responses_arg: str, response_tags: list) -> dict:
    p = Path(responses_arg)
    if p.is_dir():
        files = sorted(f for f in p.glob("*.csv") if not f.stem.endswith("_sample"))
        if not files:
            sys.exit(f"No (non-sample) CSV files found in {p}")
    else:
        files = [p]

    if response_tags and len(response_tags) != len(files):
        sys.exit(f"--response-tags has {len(response_tags)} values but found {len(files)} files")

    lookup = {}
    for i, f in enumerate(files):
        tag = response_tags[i] if response_tags else _infer_tag(f)
        df = pd.read_csv(f)
        if not all(c in df.columns for c in ("question_id", "model", "response")):
            print(f"  WARNING: {f.name} missing required columns — skipping")
            continue
        df["question_id"] = df["question_id"].astype(str)
        for _, row in df.iterrows():
            qid = str(row["question_id"])
            base_model = str(row["model"])
            compound = base_model if "__" in base_model else f"{base_model}__{tag}"
            resp = str(row["response"]) if pd.notna(row.get("response")) else ""
            lookup[(qid, compound)] = resp
        print(f"  {f.name}: tag='{tag}', {len(df)} rows")

    print(f"Response lookup built: {len(lookup)} entries")
    return lookup


def _lookup_response(r_lookup: dict, qid: str, source_name: str) -> Optional[str]:
    val = r_lookup.get((qid, source_name))
    if val is not None:
        return val
    # If source_name has no tag, try any compound match
    if "__" not in source_name:
        candidates = sorted(m for (q, m) in r_lookup if q == qid and m.split("__")[0] == source_name)
        if candidates:
            return r_lookup[(qid, candidates[0])]
    return None


# ── Parse per-item scores ─────────────────────────────────────────────────────

def _parse_scores(raw) -> list:
    if pd.isna(raw) if not isinstance(raw, str) else False:
        return []
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []
    return raw if isinstance(raw, list) else []


def _build_item_records(row: pd.Series, rubric_items: list) -> list:
    """Merge rubric item definitions with per-judge scores into one list."""
    judge_cols = {
        "gemini": "gemini_per_item_scores",
        "claude": "claude_per_item_scores",
        "gpt":    "gpt_per_item_scores",
    }
    consensus_col = "consensus_per_item_scores"

    # Index per-judge scores by item id
    judge_scores: dict[str, dict] = {j: {} for j in judge_cols}
    for judge, col in judge_cols.items():
        if col in row.index:
            for s in _parse_scores(row[col]):
                judge_scores[judge][s["id"]] = s

    # Index consensus scores
    consensus: dict[str, dict] = {}
    if consensus_col in row.index:
        for s in _parse_scores(row[consensus_col]):
            consensus[s["id"]] = s

    result = []
    for item in rubric_items:
        iid = item["id"]
        cons = consensus.get(iid, {})
        judges_out = {}
        for judge in judge_cols:
            jd = judge_scores[judge].get(iid)
            if jd:
                judges_out[judge] = {"met": jd.get("met"), "explanation": jd.get("explanation", "")}

        result.append({
            "id": iid,
            "description": item.get("description", ""),
            "rationale": item.get("rationale", ""),
            "facts": item.get("facts", []),
            "consensus_met": cons.get("met"),
            "consensus_votes": cons.get("votes"),
            "judges": judges_out,
        })
    return result


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_scores(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    before = len(df)
    df = df[df["avg_rubric_score"].notna()].copy()
    print(f"Dropped {before - len(df)} rows missing avg_rubric_score; {len(df)} remain")

    if len(df) <= n:
        print(f"Fewer rows ({len(df)}) than n={n} — using all")
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Stratify by score tercile
    df["_tercile"] = pd.cut(df["avg_rubric_score"], bins=[0, 0.34, 0.67, 1.01],
                            labels=["low", "mid", "high"], include_lowest=True)
    total = len(df)
    parts = []
    for label, group in df.groupby("_tercile", observed=True):
        k = max(1, round(len(group) / total * n))
        k = min(k, len(group))
        parts.append(group.sample(n=k, random_state=seed))

    sampled = pd.concat(parts)
    if len(sampled) < n:
        rest = df[~df.index.isin(sampled.index)]
        sampled = pd.concat([sampled, rest.sample(n=min(n - len(sampled), len(rest)), random_state=seed)])
    sampled = sampled.head(n).sample(frac=1, random_state=seed).drop(columns=["_tercile"]).reset_index(drop=True)
    print(f"Sampled {len(sampled)} rows (stratified by avg_rubric_score tercile)")
    return sampled


# ── Enrich ────────────────────────────────────────────────────────────────────

def enrich(df: pd.DataFrame, rubrics: dict, q_lookup: dict, r_lookup: dict) -> list:
    records = []
    missing_q = missing_r = missing_rubric = 0

    for _, row in df.iterrows():
        qid = str(row["question_id"])
        source_name = str(row.get("source_name", ""))
        model = str(row.get("model", ""))

        question = q_lookup.get(qid, "")
        if not question:
            missing_q += 1

        rubric_items_raw = rubrics.get(qid, [])
        if not rubric_items_raw:
            missing_rubric += 1

        response = _lookup_response(r_lookup, qid, source_name)
        if response is None:
            missing_r += 1
            print(f"  WARNING: no response for ({qid}, {source_name})")
            response = ""

        rubric_items = _build_item_records(row, rubric_items_raw)

        records.append({
            "question_id": qid,
            "source_name": source_name,
            "model": model,
            "question": question,
            "response": response,
            "avg_rubric_score": float(row["avg_rubric_score"]) if pd.notna(row.get("avg_rubric_score")) else None,
            "rubric_items": rubric_items,
        })

    if missing_q:
        print(f"WARNING: {missing_q} records had no matching question")
    if missing_rubric:
        print(f"WARNING: {missing_rubric} records had no rubric items")
    if missing_r:
        print(f"WARNING: {missing_r} records had no response text")
    return records


# ── HTML generation ───────────────────────────────────────────────────────────

def generate_html(records: list, output_path: Path):
    template_path = Path(__file__).parent / "rubric_annotation.html"
    if not template_path.exists():
        sys.exit(f"Template not found: {template_path}")

    html = template_path.read_text(encoding="utf-8")

    data_json = json.dumps(records, ensure_ascii=False)
    data_b64 = base64.b64encode(data_json.encode("utf-8")).decode("ascii")

    pattern = r"const PRELOADED_DATA\s*=\s*\[\s*\];"
    replacement = f'const PRELOADED_DATA = JSON.parse(atob("{data_b64}"));'
    new_html, count = re.subn(pattern, replacement, html)
    if count == 0:
        sys.exit("Could not find PRELOADED_DATA placeholder in rubric_annotation.html")

    output_path.write_text(new_html, encoding="utf-8")
    print(f"\nGenerated: {output_path}  ({len(records)} records)")
    print("Open this file in a browser to begin annotation.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sample rubric scores and generate a self-contained annotation HTML file."
    )
    parser.add_argument("--scores",    required=True, help="Rubric scoring output CSV")
    parser.add_argument("--rubrics",   required=True, help="Rubric items checkpoint CSV")
    parser.add_argument("--questions", required=True, help="Questions CSV (question_id + question text)")
    parser.add_argument("--responses", required=True, help="Response CSV file or directory")
    parser.add_argument("--response-tags", nargs="+", default=None,
                        help="One tag per response file (overrides filename-based inference)")
    parser.add_argument("--n",    type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else Path(args.scores).parent / "rubric_annotation_sample.html"

    print("=== Loading scores ===")
    scores_df = load_scores(args.scores)

    print("\n=== Loading rubrics ===")
    rubrics = load_rubrics(args.rubrics)

    print("\n=== Loading questions ===")
    q_lookup = load_questions(args.questions)

    print("\n=== Loading responses ===")
    r_lookup = load_responses(args.responses, args.response_tags or [])

    print("\n=== Sampling ===")
    sampled = sample_scores(scores_df, args.n, args.seed)

    print("\n=== Enriching records ===")
    records = enrich(sampled, rubrics, q_lookup, r_lookup)

    print("\n=== Generating HTML ===")
    generate_html(records, output_path)


if __name__ == "__main__":
    main()
