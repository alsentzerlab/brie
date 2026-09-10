"""
build_elo_examples.py

Turn merged human ELO annotations (from merge_elo_annotations.py) into few-shot,
multi-turn in-context examples for score_elo_batch.py.

Input: the FULL merged annotation JSON produced by merge_elo_annotations.py, which
carries both the human verdict per dimension ({winner, note}) and the question /
reference / response text (re-attached from the unblinding map). Because the merged
record is self-contained, --records is optional; if given (a sample HTML's
PRELOADED_DATA or a records JSON) it is used as a fallback for any text the merged
record is missing. Join key:
  question_id__model_a__model_b__position

Each annotated pair yields one example turn-pair:
  user      = PAIRWISE_PROMPT (Q / reference / response A / response B)
  assistant = {dim: {winner, explanation}}  with the human winner and the annotator's
              optional note as the explanation.

Every annotated pair is used (subject to --max-examples). Deduplicated by unordered
pair (question_id, {model_a, model_b}); the position=="ab" row is preferred.

Usage:
  python build_elo_examples.py \
      --annotations elo_annotations_full.json \
      --output      elo_examples.json
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path

# Prompt template + dimensions — kept byte-for-byte in sync with score_elo_batch.py
# (PAIRWISE_PROMPT / DIMS). Copied rather than imported so this builder stays
# dependency-free (the scorer pulls in tiktoken/google at import time).
DIMS = ("completeness", "relevancy", "concision")

PAIRWISE_PROMPT = """\
You are a medical expert comparing two responses to a clinical information retrieval query.
Given a reference answer (gold standard) and two candidate responses (A and B), decide which
response is better on each of the following dimensions, or declare a tie.

Question:
<question>{QUESTION}</question>

Reference answer:
<reference>{REFERENCE}</reference>

Response A:
<response_a>{RESPONSE_A}</response_a>

Response B:
<response_b>{RESPONSE_B}</response_b>

Evaluate on these three dimensions:

Completeness: Which response includes more of the important clinical details present in the \
reference answer? Prefer the response that omits fewer key facts.

Relevancy: Which response stays closer to what the question asks and the reference answer \
covers, without introducing unnecessary or tangential details?

Concision: Which response communicates the necessary information more concisely, without \
excessive verbosity or redundant phrasing?

For each dimension, output "A", "B", or "tie".

Output Format:
{{
    "completeness": {{"winner": "A" | "B" | "tie", "explanation": "..."}},
    "relevancy":    {{"winner": "A" | "B" | "tie", "explanation": "..."}},
    "concision":    {{"winner": "A" | "B" | "tie", "explanation": "..."}}
}}

Ensure the output is valid JSON with double quotes for all keys and string values.\
"""

_B64_RE = re.compile(r'JSON\.parse\(atob\("([A-Za-z0-9+/=]+)"\)\)')


def load_records(path: str) -> dict:
    """Return {recordKey: record} from a sample HTML (PRELOADED_DATA) or records JSON."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        m = _B64_RE.search(text)
        if not m:
            sys.exit(f"Could not find embedded PRELOADED_DATA in {path}")
        data = json.loads(base64.b64decode(m.group(1)).decode("utf-8"))
    lookup = {_key(r): r for r in data}
    print(f"Loaded {len(lookup)} context records from {path}")
    return lookup


def _key(r: dict) -> str:
    return f"{r.get('question_id')}__{r.get('model_a')}__{r.get('model_b')}__{r.get('position')}"


def _winner_out(w: str | None) -> str:
    """Normalize a stored winner (A/B/TIE) to the prompt's output vocabulary."""
    w = (w or "").strip().upper()
    return "tie" if w not in ("A", "B") else w


def _overall_winner(verdict: dict) -> str:
    """Majority vote across dimensions (mirrors score_elo_batch._parse_judgment)."""
    counts = {"A": 0, "B": 0, "tie": 0}
    for dim in DIMS:
        counts[verdict[dim]["winner"]] += 1
    if counts["A"] >= 2:
        return "A"
    if counts["B"] >= 2:
        return "B"
    return "tie"


def build_examples(annotations: list[dict], ctx: dict) -> list[dict]:
    """Return a list of per-example records, each:
        {overall: 'A'|'B'|'tie', turns: [user, assistant], preview: {...}}
    Tagged by overall verdict so they can be sampled across the winner axis."""
    examples: list[dict] = []
    seen: set[tuple] = set()
    missing = 0

    # Prefer the ab position when both orientations were annotated.
    annotations = sorted(annotations, key=lambda a: 0 if a.get("position") == "ab" else 1)

    for ann in annotations:
        pair_key = (ann.get("question_id"), frozenset({ann.get("model_a"), ann.get("model_b")}))
        if pair_key in seen:
            continue

        # Merged annotation records are self-contained; fall back to --records / ann itself.
        rec = ctx.get(_key(ann)) or ann
        if not (rec.get("response_a") or rec.get("response_b")):
            missing += 1
            continue
        seen.add(pair_key)

        verdict = {}
        dim_views = []
        for dim in DIMS:
            d = ann.get(dim) or {}
            winner = _winner_out(d.get("winner"))
            explanation = d.get("note") or ""
            verdict[dim] = {"winner": winner, "explanation": explanation}
            dim_views.append({"dim": dim, "winner": winner, "explanation": explanation})

        question = rec.get("question") or rec.get("natural_query") or ""
        user = PAIRWISE_PROMPT.format(
            QUESTION=question,
            REFERENCE=rec.get("reference_answer", ""),
            RESPONSE_A=rec.get("response_a", ""),
            RESPONSE_B=rec.get("response_b", ""),
        )
        overall = _overall_winner(verdict)
        examples.append({
            "overall": overall,
            "turns": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": json.dumps(verdict, ensure_ascii=False)},
            ],
            "preview": {
                "question_id": ann.get("question_id"),
                "model_a": ann.get("model_a"), "model_b": ann.get("model_b"),
                "question": question,
                "reference": rec.get("reference_answer", ""),
                "response_a": rec.get("response_a", ""),
                "response_b": rec.get("response_b", ""),
                "overall": overall,
                "dims": dim_views,
            },
        })

    if missing:
        print(f"WARNING: {missing} annotations had no response text (skipped)")
    print(f"Built {len(examples)} ELO example pairs")
    print("  by overall winner: " + _stratum_summary(examples))
    return examples


_STRATA = ("A", "B", "tie")


def _stratum_summary(examples: list[dict]) -> str:
    from collections import Counter
    c = Counter(e["overall"] for e in examples)
    return ", ".join(f"{w}={c[w]}" for w in _STRATA if c[w])


def select_examples(examples: list[dict], max_examples: int, seed: int) -> list[dict]:
    """Round-robin sample up to max_examples across overall winner (A/B/tie) so the
    few-shot set is balanced. Keeps all if there are fewer than the cap. Result is
    shuffled. Deterministic by seed."""
    import random
    rng = random.Random(seed)

    buckets = {w: [e for e in examples if e["overall"] == w] for w in _STRATA}
    for w in buckets:
        rng.shuffle(buckets[w])
    order = [w for w in _STRATA if buckets[w]]
    kept: list[dict] = []
    while len(kept) < max_examples and any(buckets[w] for w in order):
        for w in order:
            if buckets[w]:
                kept.append(buckets[w].pop())
                if len(kept) >= max_examples:
                    break
    rng.shuffle(kept)

    print(f"Selected {len(kept)} examples")
    print("  by overall winner: " + _stratum_summary(kept))
    return kept


def _wlabel(w: str, model_a: str, model_b: str) -> str:
    """Human label for a winner code, e.g. 'A (claude_haiku__recent)'."""
    if w == "A":
        return f"A ({model_a})"
    if w == "B":
        return f"B ({model_b})"
    return "tie"


def write_preview(previews: list[dict], path: Path) -> None:
    out = []
    for i, p in enumerate(previews, 1):
        ma, mb = p["model_a"], p["model_b"]
        ov = p.get("overall", "tie")
        ov_label = "tie" if ov == "tie" else f"winner {_wlabel(ov, ma, mb)}"
        out.append("=" * 78)
        out.append(f"EXAMPLE {i}   [overall {ov_label}]   question_id: {p['question_id']}")
        out.append("=" * 78)
        out.append(f"QUESTION:\n  {p['question']}")
        out.append(f"\nREFERENCE ANSWER:\n  {p['reference']}")
        out.append("\n--- RESPONSES BEING COMPARED ---")
        out.append(f"\nRESPONSE A  ({ma}):\n  {p['response_a']}")
        out.append(f"\nRESPONSE B  ({mb}):\n  {p['response_b']}")
        out.append("\n--- HUMAN VERDICT ---")
        for d in p["dims"]:
            out.append(f"\n  {d['dim'].capitalize()}: {_wlabel(d['winner'], ma, mb)}")
            if d["explanation"]:
                out.append(f"    Note: {d['explanation']}")
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Build few-shot examples from merged ELO annotations.")
    ap.add_argument("--annotations", required=True, help="Merged annotation JSON from merge_elo_annotations.py")
    ap.add_argument("--records", default=None,
                    help="Optional fallback: sample HTML (PRELOADED_DATA) or records JSON for missing text")
    ap.add_argument("--output", default=None, help="Output JSON path (default: elo_examples.json next to annotations)")
    ap.add_argument("--max-examples", type=int, default=11,
                    help="Cap on total examples (default: 11, i.e. <12), balanced across winner A/B/tie.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    args = ap.parse_args()

    with open(args.annotations, encoding="utf-8") as f:
        annotations = json.load(f)
    if not isinstance(annotations, list):
        sys.exit("Annotations file must be a JSON list.")
    print(f"Loaded {len(annotations)} annotations from {args.annotations}")

    ctx = load_records(args.records) if args.records else {}
    examples = build_examples(annotations, ctx)
    if not examples:
        sys.exit("No usable examples (no annotations carried response text). Nothing written.")

    examples = select_examples(examples, args.max_examples, args.seed)
    turns = [t for e in examples for t in e["turns"]]
    previews = [e["preview"] for e in examples]

    out = Path(args.output) if args.output else Path(args.annotations).parent / "elo_examples.json"
    out.write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")
    preview = out.with_suffix(".txt")
    write_preview(previews, preview)
    print(f"\nWrote {len(turns)} turns ({len(examples)} examples) → {out}")
    print(f"Preview → {preview}")
    print(f"Use with: python score_elo_batch.py ... --examples {out}")


if __name__ == "__main__":
    main()
