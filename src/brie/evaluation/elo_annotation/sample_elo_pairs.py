"""
sample_elo_pairs.py

Sample pairwise ELO comparisons, join in question text and model responses,
and generate a self-contained HTML annotation file with pre-embedded data.

BLINDED STUDY: the generated HTML embeds ONLY the question and the two responses
(plus an opaque record_id) — no model names, judge, position, or LLM verdicts. All
identifying/LLM fields are written to a sidecar `<output>.map.json` keyed by
record_id, which is NOT given to annotators. After annotation, join the export back
to the map with merge_elo_annotations.py to recover identity and build the truth CSV.

Usage:
  # Single pairs file
  python sample_elo_pairs.py \
      --pairs     $EVAL/elo/elo_haiku_pairs.csv \
      --questions $EVAL/phase2_qa.csv \
      --responses $EVAL/answer_cleaned/ \
      --n 50 --seed 42 \
      --output    $EVAL/elo/annotate_haiku.html

  # Pool all *_pairs.csv from a directory
  python sample_elo_pairs.py \
      --pairs     $EVAL/elo/ \
      --questions $EVAL/phase2_qa.csv \
      --responses $EVAL/answer_cleaned/ \
      --n 50 --seed 42 \
      --output    $EVAL/elo/annotate_pooled.html

Response matching:
  Model names in the pairs CSV are compound identifiers: {base_model}__{inference_type}
  (e.g. claude_haiku__recent). Responses are joined by splitting on '__' and matching
  against base_model + inference_type columns, or against the full compound name if
  the response CSV already contains compound model names.

  If --response-tags is supplied (one tag per --responses file), tags are applied the
  same way as score_elo_batch.py. Otherwise the tag is inferred from the filename
  suffix (last '_'-separated component before .csv), skipping any trailing '_sample'.

  For per-retrieval-method ELO runs (elo_bm25, elo_recent, etc.) the model names in
  the pairs CSV are bare (e.g. 'gpt5', 'gemini_pro'). The fallback resolver uses the
  source pairs filename to determine which retrieval method was used and constructs
  the compound lookup key automatically.
"""

import argparse
import base64
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Retrieval method names used in per-method ELO scripts (elo_bm25.sh, elo_recent.sh, …)
# When a pairs file is named elo_{METHOD}_pairs.csv and model names are bare,
# we use METHOD to build the compound lookup key.
_KNOWN_METHODS = {"bm25", "dense", "late", "recent", "recent200"}


# ── Load pairs ────────────────────────────────────────────────────────────────

def load_pairs(pairs_arg: str) -> pd.DataFrame:
    p = Path(pairs_arg)
    if p.is_dir():
        files = sorted(p.glob("*_pairs.csv"))
        if not files:
            sys.exit(f"No *_pairs.csv files found in {p}")
        frames = []
        for f in files:
            df = pd.read_csv(f)
            df["source_file"] = f.name
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        print(f"Loaded {len(combined)} rows from {len(files)} files in {p}")
        return combined
    else:
        df = pd.read_csv(p)
        df["source_file"] = p.name
        print(f"Loaded {len(df)} rows from {p}")
        return df


# ── Load questions ────────────────────────────────────────────────────────────

def load_questions(questions_path: str) -> dict:
    df = pd.read_csv(questions_path)
    df["question_id"] = df["question_id"].astype(str)

    q_col = "question" if "question" in df.columns else None
    nq_col = "natural_query" if "natural_query" in df.columns else None
    ans_col = "answer" if "answer" in df.columns else (
        "annotation_sub_answer" if "annotation_sub_answer" in df.columns else None
    )

    lookup = {}
    for _, row in df.iterrows():
        qid = str(row["question_id"])
        lookup[qid] = {
            "question": str(row[q_col]) if q_col and pd.notna(row.get(q_col)) else "",
            "natural_query": str(row[nq_col]) if nq_col and pd.notna(row.get(nq_col)) else "",
            "reference_answer": str(row[ans_col]) if ans_col and pd.notna(row.get(ans_col)) else "",
        }
        # Fall back: use whichever question field is populated
        if not lookup[qid]["natural_query"] and lookup[qid]["question"]:
            lookup[qid]["natural_query"] = lookup[qid]["question"]
        if not lookup[qid]["question"] and lookup[qid]["natural_query"]:
            lookup[qid]["question"] = lookup[qid]["natural_query"]

    print(f"Loaded {len(lookup)} questions from {questions_path}")
    return lookup


# ── Load responses ────────────────────────────────────────────────────────────

def _infer_tag(path: Path) -> str:
    """
    Infer the retrieval tag from the filename.
    answer_508_claude_recent.csv        → "recent"
    answer_508_claude_recent_sample.csv → "recent"  (strip trailing _sample)
    """
    stem = path.stem  # e.g. "answer_508_claude_recent" or "answer_508_claude_recent_sample"
    parts = stem.split("_")
    # Strip trailing "sample" if present
    if parts and parts[-1] == "sample":
        parts = parts[:-1]
    return parts[-1] if parts else stem


def load_responses(responses_arg: str, response_tags: list) -> dict:
    """
    Returns lookup: {(question_id_str, compound_model_name): response_text}
    compound_model_name = base_model + "__" + tag  (e.g. "claude_haiku__recent")

    When loading from a directory, *_sample.csv files are skipped (they are
    subsampled duplicates of the non-sample files).
    """
    p = Path(responses_arg)
    if p.is_dir():
        files = sorted(f for f in p.glob("*.csv") if not f.stem.endswith("_sample"))
        if not files:
            sys.exit(f"No (non-sample) CSV files found in {p}")
    else:
        files = [p]

    if response_tags and len(response_tags) != len(files):
        sys.exit(
            f"--response-tags has {len(response_tags)} values but found {len(files)} response files"
        )

    lookup = {}
    for i, f in enumerate(files):
        tag = response_tags[i] if response_tags else _infer_tag(f)
        df = pd.read_csv(f)
        if "question_id" not in df.columns or "model" not in df.columns or "response" not in df.columns:
            print(f"  WARNING: {f.name} missing required columns (question_id, model, response) — skipping")
            continue

        df["question_id"] = df["question_id"].astype(str)
        for _, row in df.iterrows():
            qid = str(row["question_id"])
            base_model = str(row["model"])

            # Support two compound-name patterns:
            # 1. The CSV already has compound names (base_model already contains __)
            # 2. The CSV has bare model names and we append the tag
            if "__" in base_model:
                compound = base_model
            else:
                compound = f"{base_model}__{tag}"

            resp = str(row["response"]) if pd.notna(row.get("response")) else ""
            lookup[(qid, compound)] = resp

        print(f"  {f.name}: tag='{tag}', {len(df)} rows")

    print(f"Response lookup built: {len(lookup)} entries")
    return lookup


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_pairs(df: pd.DataFrame, n: int, seed: int, judge_filter: Optional[str]) -> pd.DataFrame:
    # Drop incomplete judgments
    before = len(df)
    df = df[df["overall_winner"].notna() & (df["overall_winner"] != "")].copy()
    print(f"Dropped {before - len(df)} incomplete rows; {len(df)} remain")

    # Filter by judge
    if judge_filter:
        df = df[df["judge"] == judge_filter].copy()
        print(f"Filtered to judge='{judge_filter}': {len(df)} rows")
        if df.empty:
            sys.exit(f"No rows remain after filtering to judge='{judge_filter}'")

    # Deduplicate by canonical (unordered) pair — prefer "ab" position
    df["_pair_key"] = df.apply(
        lambda r: (str(r["question_id"]), tuple(sorted([str(r["model_a"]), str(r["model_b"])]))),
        axis=1,
    )
    # Sort so "ab" rows come first, then keep first occurrence per key
    df["_pos_rank"] = (df["position"] != "ab").astype(int)
    df = df.sort_values("_pos_rank").drop_duplicates(subset=["_pair_key"]).drop(
        columns=["_pair_key", "_pos_rank"]
    )
    print(f"After dedup: {len(df)} unique pairs")

    if len(df) <= n:
        print(f"Fewer pairs ({len(df)}) than requested n={n} — using all")
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Stratified sample proportional to overall_winner distribution
    sampled_parts = []
    groups = df.groupby("overall_winner")
    total = len(df)
    winners = list(groups.groups.keys())

    for winner in winners:
        group = groups.get_group(winner)
        k = max(1, round(len(group) / total * n))
        k = min(k, len(group))
        sampled_parts.append(group.sample(n=k, random_state=seed))

    sampled = pd.concat(sampled_parts)

    # Top up or trim to exactly n
    if len(sampled) < n:
        rest = df[~df.index.isin(sampled.index)]
        extra = rest.sample(n=min(n - len(sampled), len(rest)), random_state=seed)
        sampled = pd.concat([sampled, extra])
    sampled = sampled.head(n).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Sampled {len(sampled)} pairs (stratified by overall_winner)")
    return sampled


# ── Join and enrich ───────────────────────────────────────────────────────────

def _method_from_source_file(source_file: str) -> Optional[str]:
    """
    Extract the retrieval method from a per-method ELO filename.
    "elo_bm25_pairs.csv" → "bm25",  "elo_recent200_pairs.csv" → "recent200"
    Returns None if the filename encodes a model name rather than a method.
    """
    m = re.match(r"elo_(.+)_pairs\.csv", source_file)
    if m:
        candidate = m.group(1)
        if candidate in _KNOWN_METHODS:
            return candidate
    return None


def _lookup_response(r_lookup: dict, qid: str, model: str, source_file: str = "") -> Optional[str]:
    """
    Resolve a (question_id, model_name) pair to a response string.

    Resolution order:
    1. Exact match — handles per-model ELO runs where model names are already
       compound (e.g. "claude_haiku__recent").
    2. Source-file hint — for per-method ELO runs (elo_bm25_pairs.csv, etc.) the
       pairs CSV has bare model names. Extract the method from the source filename
       and try (qid, "{model}__{method}").
    3. Unique-candidate fallback — if exactly one compound key matches the base
       model, use it.
    4. Alphabetically-first fallback — deterministic tie-break when multiple
       retrieval variants exist.
    """
    # 1. Exact match
    val = r_lookup.get((qid, model))
    if val is not None:
        return val

    if "__" not in model:
        # 2. Source-file hint
        method = _method_from_source_file(source_file)
        if method:
            val = r_lookup.get((qid, f"{model}__{method}"))
            if val is not None:
                return val

        # 3 & 4. Fallback: find all compound keys for this base model
        candidate_keys = sorted(
            m for (q, m) in r_lookup if q == qid and m.split("__")[0] == model
        )
        if candidate_keys:
            return r_lookup[(qid, candidate_keys[0])]

    return None


def _record_id(question_id: str, model_a: str, model_b: str, position: str) -> str:
    """Deterministic, opaque key for one pair. Stable across reruns (resume-safe)
    and not reversible by the annotator — identity lives only in the sidecar map."""
    raw = f"{question_id}|{model_a}|{model_b}|{position}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _inference_type(model: str) -> str:
    """Retrieval/inference tag from a compound model name (claude_haiku__recent → recent)."""
    return model.split("__", 1)[1] if "__" in model else ""


# Fields shown to the annotator in the UI.
DISPLAY_FIELDS = (
    "record_id", "natural_query", "question", "reference_answer",
    "response_a", "response_b",
)

# Identity carried THROUGH to the annotator's export so it is self-sufficient.
# These are embedded but never rendered in the UI (blinded at the UI layer only).
# Note: being embedded, they are base64-decodable from the HTML — but the LLM
# verdicts (*_winner / *_explanation / judge / overall_winner) are NOT embedded;
# they stay in the sidecar map so they can never leak into the human labels.
EXPORT_META_FIELDS = (
    "question_id", "model_a", "model_b",
    "inference_type_a", "inference_type_b", "position",
)

EMBED_FIELDS = DISPLAY_FIELDS + EXPORT_META_FIELDS


def split_records(records: list) -> tuple[list, dict]:
    """Return (display_records, map). display_records carry the UI display fields plus
    identity (EXPORT_META_FIELDS), but NOT the LLM verdicts; they are embedded in the
    HTML. map is record_id → full enriched record (incl. LLM verdicts) for later
    unblinding / human-vs-LLM comparison."""
    display = [{k: r.get(k, "") for k in EMBED_FIELDS} for r in records]
    mapping = {r["record_id"]: r for r in records}
    return display, mapping


def enrich(df: pd.DataFrame, q_lookup: dict, r_lookup: dict) -> list:
    records = []
    missing_q = 0
    missing_r = 0

    str_cols = [
        "completeness_winner", "completeness_winner_model", "completeness_explanation",
        "relevancy_winner",    "relevancy_winner_model",    "relevancy_explanation",
        "concision_winner",    "concision_winner_model",    "concision_explanation",
        "overall_winner",      "overall_winner_model",
    ]

    for _, row in df.iterrows():
        qid = str(row["question_id"])
        model_a = str(row["model_a"])
        model_b = str(row["model_b"])

        q = q_lookup.get(qid)
        if not q:
            missing_q += 1
            q = {"question": "", "natural_query": "", "reference_answer": ""}

        src = str(row.get("source_file", ""))
        resp_a = _lookup_response(r_lookup, qid, model_a, src)
        resp_b = _lookup_response(r_lookup, qid, model_b, src)
        if resp_a is None or resp_b is None:
            missing_r += 1
            if resp_a is None:
                print(f"  WARNING: no response for ({qid}, {model_a})")
            if resp_b is None:
                print(f"  WARNING: no response for ({qid}, {model_b})")

        position = str(row.get("position", ""))
        rec = {
            "record_id": _record_id(qid, model_a, model_b, position),
            "question_id": qid,
            "model_a": model_a,
            "model_b": model_b,
            "inference_type_a": _inference_type(model_a),
            "inference_type_b": _inference_type(model_b),
            "position": position,
            "judge": str(row.get("judge", "")),
            "source_file": str(row.get("source_file", "")),
            # LLM judgment fields
            **{c: str(row[c]) if pd.notna(row.get(c)) else "" for c in str_cols if c in row},
            # Joined fields
            "natural_query": q["natural_query"],
            "question": q["question"],
            "reference_answer": q["reference_answer"],
            "response_a": resp_a if resp_a is not None else "",
            "response_b": resp_b if resp_b is not None else "",
        }
        records.append(rec)

    if missing_q:
        print(f"WARNING: {missing_q} records had no matching question")
    if missing_r:
        print(f"WARNING: {missing_r} records had missing responses (one or both sides)")

    return records


# ── HTML generation ───────────────────────────────────────────────────────────

def map_path_for(output_path: Path) -> Path:
    """Sidecar unblinding map path: annotate.html → annotate.map.json."""
    return output_path.with_suffix(".map.json")


def write_map(mapping: dict, output_path: Path) -> Path:
    """Write the record_id → full-record unblinding map next to the HTML.
    This file is NOT given to annotators — it is joined back in by
    merge_elo_annotations.py to recover identity and the LLM verdicts."""
    map_path = map_path_for(output_path)
    map_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote unblinding map: {map_path}  ({len(mapping)} records)  [keep private]")
    return map_path


def generate_html(records: list, output_path: Path):
    """Embed the blinded display records (DISPLAY_FIELDS only) into the template."""
    template_path = Path(__file__).parent / "elo_annotation.html"
    if not template_path.exists():
        sys.exit(f"Template not found: {template_path}")

    html = template_path.read_text(encoding="utf-8")

    # Base64-encode the JSON so no clinical note content can ever break the
    # <script> block (base64 alphabet contains no '<', '>', or '-').
    data_json = json.dumps(records, ensure_ascii=False)
    data_b64 = base64.b64encode(data_json.encode("utf-8")).decode("ascii")

    # Replace the empty PRELOADED_DATA array with a base64 decode expression
    pattern = r"const PRELOADED_DATA\s*=\s*\[\s*\];"
    replacement = (
        f'const PRELOADED_DATA = JSON.parse(atob("{data_b64}"));'
    )
    new_html, count = re.subn(pattern, replacement, html)
    if count == 0:
        sys.exit("Could not find PRELOADED_DATA placeholder in elo_annotation.html")

    output_path.write_text(new_html, encoding="utf-8")
    print(f"\nGenerated: {output_path}  ({len(records)} records)")
    print("Open this file in a browser to begin annotation.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sample ELO pairs and generate a self-contained annotation HTML file."
    )
    parser.add_argument("--pairs", required=True,
                        help="Path to a *_pairs.csv file or a directory containing them")
    parser.add_argument("--questions", required=True,
                        help="Path to phase2_qa.csv (or equivalent questions CSV)")
    parser.add_argument("--responses", required=True,
                        help="Path to a response CSV file or directory of response CSVs")
    parser.add_argument("--response-tags", nargs="+", default=None,
                        help="One tag per response file (overrides filename-based tag inference)")
    parser.add_argument("--n", type=int, default=50, help="Number of pairs to sample (default: 50)")
    # PROVENANCE: the BRIE ELO annotation sample was generated with --seed 42.
    # (Fact entailment used --seed 43; see sample_fact_entailment.py.)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42; BRIE ELO used 42)")
    parser.add_argument("--judge", default=None, help="Filter to a specific judge (e.g. gemini)")
    parser.add_argument("--output", default=None,
                        help="Output HTML path (default: elo_annotation_sample.html next to --pairs)")
    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        pairs_p = Path(args.pairs)
        base_dir = pairs_p if pairs_p.is_dir() else pairs_p.parent
        output_path = base_dir / "elo_annotation_sample.html"

    print("=== Loading pairs ===")
    pairs_df = load_pairs(args.pairs)

    print("\n=== Loading questions ===")
    q_lookup = load_questions(args.questions)

    print("\n=== Loading responses ===")
    r_lookup = load_responses(args.responses, args.response_tags or [])

    print("\n=== Sampling ===")
    sampled = sample_pairs(pairs_df, args.n, args.seed, args.judge)

    print("\n=== Enriching records ===")
    records = enrich(sampled, q_lookup, r_lookup)

    print("\n=== Splitting blinded display from unblinding map ===")
    display, mapping = split_records(records)

    print("\n=== Generating HTML ===")
    generate_html(display, output_path)
    write_map(mapping, output_path)


if __name__ == "__main__":
    main()
