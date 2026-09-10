"""
sample_fact_entailment.py

Sample fact-entailment scoring results, join in per-fact judge verdicts, question
text, reference answer, and candidate responses, then generate a self-contained HTML
annotation file.

BLINDED STUDY: the generated HTML embeds ONLY the question, reference answer,
candidate response, and bare {idx, fact} lists — no model/source identity and no LLM
consensus/votes/per-juror verdicts. Those are written to a sidecar
`<output>.map.json` keyed by record_id (NOT given to annotators). After annotation,
join the export back to the map with merge_fact_annotations.py.

Usage:
  python sample_fact_entailment.py \
      --scores    $EVAL/entailment/scores.csv \
      --facts     $EVAL/entailment/facts_atomic.csv \
      --questions $EVAL/phase2_qa.csv \
      --responses $EVAL/answer_cleaned/ \
      --n 50 --seed 42 \
      --output    $EVAL/entailment/annotate.html

Input schemas
  scores CSV   (output of score_facts_batch.py):
    question_id, source_name, model,
    gemini_precision, gemini_recall, gemini_entailed_ref_facts, gemini_entailed_cand_facts,
    claude_*, gpt_*, avg_precision, avg_recall,
    consensus_precision, consensus_recall,
    consensus_entailed_ref_facts, consensus_entailed_cand_facts

  facts CSV    (output of atomize_facts_batch.py):
    source_name, question_id, model, facts_atomic  (JSON list of strings)

  questions CSV:
    question_id, question (or natural_query)

Sampling is stratified by consensus_recall tercile (low/mid/high).
"""

import argparse
import ast
import base64
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


_KNOWN_METHODS = {"bm25", "dense", "late", "recent", "recent200", "agent"}
JURORS = ["gemini", "claude", "gpt"]

# Provider names in source_name that don't map to model names by prefix matching.
# Keys are normalized provider strings (no underscores, lowercase).
_PROVIDER_ALIASES: dict[str, list[str]] = {
    "openai": ["gpt5", "gpt5_nano"],
}


def _normalize(s: str) -> str:
    """Remove underscores and lowercase — used for fuzzy provider matching."""
    return s.replace("_", "").lower()


def _parse_source_name(source_name: str) -> tuple[str, str]:
    """
    Split 'provider_method' into ('provider', 'method').
    Tries each known method as a suffix (longest first) so that
    'qwensmall_recent200' → ('qwensmall', 'recent200') rather than
    ('qwensmall_recent', '200').
    Returns ('', '') if no known method suffix is found.
    """
    for m in sorted(_KNOWN_METHODS, key=len, reverse=True):
        suffix = "_" + m
        if source_name.endswith(suffix):
            return source_name[: -len(suffix)], m
    return "", ""


# ── Fact parsing ──────────────────────────────────────────────────────────────

def _parse_facts(raw) -> list:
    if not raw or (isinstance(raw, float)):
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


# ── Load scores ───────────────────────────────────────────────────────────────

def load_scores(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["question_id"] = df["question_id"].astype(str)
    print(f"Loaded {len(df)} score rows from {path}")
    return df


# ── Load facts ────────────────────────────────────────────────────────────────

def load_facts(path: str, gold_source: str) -> tuple[dict, dict]:
    """
    Returns:
      gold_lookup:  {question_id: [fact_str, ...]}
      cand_lookup:  {(question_id, source_name): [fact_str, ...]}
    """
    df = pd.read_csv(path)
    df["question_id"] = df["question_id"].astype(str)

    gold_lookup: dict[str, list] = {}
    cand_lookup: dict[tuple, list] = {}

    all_sources = set()
    for _, row in df.iterrows():
        qid = str(row["question_id"])
        src = str(row["source_name"])
        all_sources.add(src)
        facts = _parse_facts(row.get("facts_atomic"))
        if src == gold_source:
            gold_lookup[qid] = facts
        else:
            cand_lookup[(qid, src)] = facts

    if not gold_lookup:
        print(f"  WARNING: no rows matched gold_source='{gold_source}'")
        print(f"  Available source_name values: {sorted(all_sources)}")
        print("  Pass --gold-source <name> to override.")
    print(f"Gold facts: {len(gold_lookup)} questions")
    print(f"Candidate facts: {len(cand_lookup)} (question, source) pairs")
    return gold_lookup, cand_lookup


# ── Load questions ────────────────────────────────────────────────────────────

def load_questions(path: str) -> dict:
    """Returns {question_id: {"question": str, "reference_answer": str}}."""
    df = pd.read_csv(path)
    df["question_id"] = df["question_id"].astype(str)
    q_col = next((c for c in ("question", "natural_query") if c in df.columns), None)
    ans_col = next((c for c in ("answer", "annotation_sub_answer") if c in df.columns), None)
    lookup = {}
    for _, row in df.iterrows():
        lookup[str(row["question_id"])] = {
            "question": str(row[q_col]) if q_col and pd.notna(row.get(q_col)) else "",
            "reference_answer": str(row[ans_col]) if ans_col and pd.notna(row.get(ans_col)) else "",
        }
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
            sys.exit(f"No CSV files found in {p}")
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


def _resolve_response(r_lookup: dict, qid: str, source_name: str, model: str) -> tuple[Optional[str], bool]:
    """Resolve the candidate answer, preferring the EXACT `model` identity the facts were
    generated under (the `model` column is the same `display_model` written into the answer
    files by get_predictions.py and carried through atomize/score). This guarantees the
    facts and the answer come from the same model — unlike the source_name fuzzy resolver,
    which, for providers mapping to several models (openai→gpt5/gpt5_nano, gemini→flash/pro,
    qwen→small/large), can return a different model's answer.

    Returns (response_or_None, used_fuzzy_fallback).
    """
    if model:
        v = r_lookup.get((qid, model))                       # model already compound (base__method)
        if v is not None:
            return v, False
        if "__" not in model:                                # bare model: qualify with the method
            method = _parse_source_name(source_name)[1]
            if method:
                v = r_lookup.get((qid, f"{model}__{method}"))
                if v is not None:
                    return v, False
    return _lookup_response(r_lookup, qid, source_name), True  # last resort: lossy source_name match


def _lookup_response(r_lookup: dict, qid: str, source_name: str) -> Optional[str]:
    """
    Fuzzy fallback resolver: (question_id, source_name) → response text.
    Prefer _resolve_response(), which joins on the exact `model` identity; this is only
    used when the row has no usable `model` (it can mis-resolve multi-model providers).

    source_name format in facts/scores CSV: '{provider}_{method}'
      e.g. 'gemini_bm25', 'claude_agent', 'local_recent', 'openai_dense'

    Response lookup keys: (qid, '{base_model}__{method}')
      e.g. (qid, 'gemini_flash__bm25'), (qid, 'claude_haiku__agent'),
           (qid, 'qwen_small__recent200'), (qid, 'gpt5__dense')

    Resolution:
    1. Exact key match (handles pre-compound source names).
    2. Parse source_name → (provider, method); collect all lookup keys for this
       qid + method; filter by normalized provider prefix or explicit alias;
       return the first match alphabetically for determinism.
    """
    # 1. Exact match
    val = r_lookup.get((qid, source_name))
    if val is not None:
        return val

    # 2. Parse provider + method
    provider, method = _parse_source_name(source_name)
    if not method:
        return None

    norm_provider = _normalize(provider)

    # Explicit alias check (e.g. openai → gpt5, gpt5_nano)
    alias_prefixes = _PROVIDER_ALIASES.get(norm_provider)

    candidates = []
    for (q, compound) in r_lookup:
        if q != qid:
            continue
        parts = compound.split("__")
        if len(parts) != 2 or parts[1] != method:
            continue
        base_model = parts[0]
        norm_model = _normalize(base_model)

        if alias_prefixes:
            # Match against explicit alias list
            if any(_normalize(alias) == norm_model or norm_model.startswith(_normalize(alias))
                   for alias in alias_prefixes):
                candidates.append(compound)
        else:
            # Normalized prefix match: 'qwensmall' matches 'qwen_small' → 'qwensmall'
            if norm_model == norm_provider or norm_model.startswith(norm_provider) or norm_provider.startswith(norm_model):
                candidates.append(compound)

    if candidates:
        return r_lookup[(qid, sorted(candidates)[0])]
    return None


# ── Blinding: record_id + display/map split ───────────────────────────────────

def _record_id(question_id: str, source_name: str) -> str:
    """Deterministic, opaque key for one (question, candidate source). Stable across
    reruns (resume-safe) and not reversible by the annotator — identity lives only in
    the sidecar map."""
    raw = f"{question_id}|{source_name}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _inference_type(model: str) -> str:
    """Retrieval/inference tag from a compound model name (claude_haiku__recent → recent)."""
    return model.split("__", 1)[1] if "__" in model else ""


def _display_items(items: list) -> list:
    """Strip every item to just {idx, fact} — no consensus/votes/judges reach the HTML."""
    return [{"idx": it["idx"], "fact": it["fact"]} for it in items]


def split_records(records: list) -> tuple[list, dict]:
    """Return (display_records, map). display_records carry the question, reference answer,
    candidate response, bare {idx, fact} lists, and identity (question_id / source_name /
    model / inference_type) so the annotator's export is self-sufficient. The identity is
    embedded but never rendered in the UI. The per-fact LLM consensus/votes/judges are NOT
    embedded — they stay in the map so they can never leak into the human labels. map is
    record_id → full enriched record for later unblinding / human-vs-LLM comparison."""
    display = []
    for r in records:
        display.append({
            "record_id":        r["record_id"],
            # Identity — embedded for the export, not shown in the UI.
            "question_id":      r.get("question_id", ""),
            "source_name":      r.get("source_name", ""),
            "model":            r.get("model", ""),
            "inference_type":   r.get("inference_type", ""),
            # Display.
            "question":         r.get("question", ""),
            "reference_answer": r.get("reference_answer", ""),
            "response":         r.get("response", ""),
            "recall_items":     _display_items(r.get("recall_items", [])),
            "precision_items":  _display_items(r.get("precision_items", [])),
        })
    mapping = {r["record_id"]: r for r in records}
    return display, mapping


# ── Build per-fact verdict records ────────────────────────────────────────────

def _build_fact_items(facts: list, entailed_by_juror: dict[str, list], consensus_entailed: list) -> list:
    """
    facts            : list of fact strings
    entailed_by_juror: {juror: [entailed_fact_str, ...]}
    consensus_entailed: [entailed_fact_str, ...]

    Returns list of {idx, fact, consensus_entailed, votes, judges: {juror: bool}}.
    The LLM fields (consensus_entailed/votes/judges) are kept ONLY in the sidecar
    map; they are stripped from the blinded display (see split_records).
    """
    items = []
    for idx, fact in enumerate(facts):
        judges = {j: fact in entailed_by_juror.get(j, []) for j in JURORS}
        votes = sum(judges.values())
        items.append({
            "idx": idx,
            "fact": fact,
            "consensus_entailed": fact in consensus_entailed,
            "votes": votes,
            "judges": judges,
        })
    return items


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_scores(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    before = len(df)
    df = df[df["consensus_recall"].notna()].copy()
    print(f"Dropped {before - len(df)} rows missing consensus_recall; {len(df)} remain")

    if len(df) <= n:
        print(f"Fewer rows ({len(df)}) than n={n} — using all")
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df["_tercile"] = pd.cut(df["consensus_recall"], bins=[-.001, 0.34, 0.67, 1.01],
                            labels=["low", "mid", "high"])
    total = len(df)
    parts = []
    for _, group in df.groupby("_tercile", observed=True):
        k = max(1, round(len(group) / total * n))
        k = min(k, len(group))
        parts.append(group.sample(n=k, random_state=seed))

    sampled = pd.concat(parts)
    if len(sampled) < n:
        rest = df[~df.index.isin(sampled.index)]
        sampled = pd.concat([sampled, rest.sample(n=min(n - len(sampled), len(rest)), random_state=seed)])
    sampled = sampled.head(n).sample(frac=1, random_state=seed).drop(columns=["_tercile"]).reset_index(drop=True)
    print(f"Sampled {len(sampled)} rows (stratified by consensus_recall tercile)")
    return sampled


# ── Enrich ────────────────────────────────────────────────────────────────────

def enrich(df: pd.DataFrame, gold_lookup: dict, cand_lookup: dict,
           q_lookup: dict, r_lookup: dict) -> list:
    records = []
    missing_q = missing_resp = missing_gold = missing_cand = fuzzy_resp = 0

    for _, row in df.iterrows():
        qid    = str(row["question_id"])
        source = str(row.get("source_name", ""))
        model  = str(row["model"]) if pd.notna(row.get("model")) else ""

        q = q_lookup.get(qid, {})
        question = q.get("question", "")
        reference_answer = q.get("reference_answer", "")
        if not question:
            missing_q += 1

        ref_facts  = gold_lookup.get(qid, [])
        cand_facts = cand_lookup.get((qid, source), [])
        if not ref_facts:
            missing_gold += 1
        if not cand_facts:
            missing_cand += 1

        response, used_fuzzy = _resolve_response(r_lookup, qid, source, model)
        if response is None:
            missing_resp += 1
            print(f"  WARNING: no response for ({qid}, {source}, model={model!r})")
            response = ""
        elif used_fuzzy:
            fuzzy_resp += 1
            print(f"  NOTE: ({qid}, {source}) answer resolved by source_name fuzzy match "
                  f"(no exact model join; model={model!r}) — verify facts↔answer alignment")

        # Per-juror entailed fact lists (recall direction: which ref facts are covered)
        recall_by_juror = {
            j: _parse_facts(row.get(f"{j}_entailed_ref_facts", "[]"))
            for j in JURORS
        }
        consensus_ref = _parse_facts(row.get("consensus_entailed_ref_facts", "[]"))

        # Per-juror entailed fact lists (precision direction: which cand facts are supported)
        precision_by_juror = {
            j: _parse_facts(row.get(f"{j}_entailed_cand_facts", "[]"))
            for j in JURORS
        }
        consensus_cand = _parse_facts(row.get("consensus_entailed_cand_facts", "[]"))

        records.append({
            "record_id":            _record_id(qid, source),
            "question_id":          qid,
            "source_name":          source,
            "model":                model,
            "inference_type":       _inference_type(model),
            "question":             question,
            "reference_answer":     reference_answer,
            "response":             response,
            "avg_recall":           float(row["avg_recall"])           if pd.notna(row.get("avg_recall"))           else None,
            "avg_precision":        float(row["avg_precision"])        if pd.notna(row.get("avg_precision"))        else None,
            "consensus_recall":     float(row["consensus_recall"])     if pd.notna(row.get("consensus_recall"))     else None,
            "consensus_precision":  float(row["consensus_precision"])  if pd.notna(row.get("consensus_precision"))  else None,
            "recall_items":    _build_fact_items(ref_facts,  recall_by_juror,    consensus_ref),
            "precision_items": _build_fact_items(cand_facts, precision_by_juror, consensus_cand),
        })

    if missing_q:
        print(f"WARNING: {missing_q} records had no question text")
    if missing_gold:
        print(f"WARNING: {missing_gold} records had no reference facts")
    if missing_cand:
        print(f"WARNING: {missing_cand} records had no candidate facts")
    if missing_resp:
        print(f"WARNING: {missing_resp} records had no response text")
    if fuzzy_resp:
        print(f"WARNING: {fuzzy_resp} records used the source_name fuzzy fallback for the "
              f"answer (exact `model` join failed) — these are the alignment-risk rows")
    return records


# ── HTML generation ───────────────────────────────────────────────────────────

def map_path_for(output_path: Path) -> Path:
    """Sidecar unblinding map path: annotate.html → annotate.map.json."""
    return output_path.with_suffix(".map.json")


def write_map(mapping: dict, output_path: Path) -> Path:
    """Write the record_id → full-record unblinding map next to the HTML. NOT given to
    annotators — joined back in by merge_fact_annotations.py to recover identity and the
    LLM consensus."""
    map_path = map_path_for(output_path)
    map_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote unblinding map: {map_path}  ({len(mapping)} records)  [keep private]")
    return map_path


def generate_html(records: list, output_path: Path):
    """Embed the blinded display records (question + answers + bare facts) into the template."""
    template_path = Path(__file__).parent / "fact_annotation.html"
    if not template_path.exists():
        sys.exit(f"Template not found: {template_path}")

    html = template_path.read_text(encoding="utf-8")
    data_b64 = base64.b64encode(json.dumps(records, ensure_ascii=False).encode("utf-8")).decode("ascii")

    pattern = r"const PRELOADED_DATA\s*=\s*\[\s*\];"
    replacement = f'const PRELOADED_DATA = JSON.parse(atob("{data_b64}"));'
    new_html, count = re.subn(pattern, replacement, html)
    if count == 0:
        sys.exit("Could not find PRELOADED_DATA placeholder in fact_annotation.html")

    output_path.write_text(new_html, encoding="utf-8")
    print(f"\nGenerated: {output_path}  ({len(records)} records)")
    print("Open this file in a browser to begin annotation.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sample fact entailment scores and generate an annotation HTML file."
    )
    parser.add_argument("--scores",       required=True, help="score_facts_batch.py output CSV")
    parser.add_argument("--facts",        required=True, help="Atomic facts CSV (source_name, question_id, model, facts_atomic)")
    parser.add_argument("--questions",    required=True, help="Questions CSV")
    parser.add_argument("--responses",    required=True, help="Response CSV file or directory")
    parser.add_argument("--gold-source",  default="reference", help="source_name for gold facts (default: reference)")
    parser.add_argument("--response-tags", nargs="+", default=None)
    parser.add_argument("--n",    type=int, default=50)
    # PROVENANCE: the BRIE fact-entailment annotation sample was generated with --seed 43.
    # (ELO used --seed 42; see sample_elo_pairs.py.) Default is 43 so a re-run reproduces it.
    parser.add_argument("--seed", type=int, default=43, help="Random seed (default: 43; BRIE facts used 43)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else Path(args.scores).parent / "fact_annotation_sample.html"

    print("=== Loading scores ===")
    scores_df = load_scores(args.scores)

    print("\n=== Loading facts ===")
    gold_lookup, cand_lookup = load_facts(args.facts, args.gold_source)

    print("\n=== Loading questions ===")
    q_lookup = load_questions(args.questions)

    print("\n=== Loading responses ===")
    r_lookup = load_responses(args.responses, args.response_tags or [])

    print("\n=== Sampling ===")
    sampled = sample_scores(scores_df, args.n, args.seed)

    print("\n=== Enriching records ===")
    records = enrich(sampled, gold_lookup, cand_lookup, q_lookup, r_lookup)

    print("\n=== Splitting blinded display from unblinding map ===")
    display, mapping = split_records(records)

    print("\n=== Generating HTML ===")
    generate_html(display, output_path)
    write_map(mapping, output_path)


if __name__ == "__main__":
    main()
