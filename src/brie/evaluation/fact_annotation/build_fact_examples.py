"""
build_fact_examples.py

Turn human fact-entailment annotations (exported by fact_annotation.html) into
few-shot, multi-turn in-context examples for score_facts_batch.py.

Each annotated comparison yields up to two example turn-pairs — one per direction:
  - recall    ("r"): user = RECALL_PROMPT,    assistant = JSON array of REFERENCE indices present
  - precision ("p"): user = PRECISION_PROMPT, assistant = JSON array of CANDIDATE indices present
The user turns reuse the exact prompt templates from score_facts_batch.py, and the
assistant turns are the human-validated gold (the indices the annotator marked Present).

Every completed annotation is used (no sampling). A direction is emitted only when all
of its facts were marked (no nulls), so the gold index set is unambiguous. Records are
deduplicated by (question_id, source_name).

INPUT: the merged annotation JSON from merge_fact_annotations.py — it restores
question_id/source_name and the per-fact llm_consensus_entailed that the blinded export
omits. (Raw blinded exports still parse, but lack identity for dedup/preview.)

Usage:
  python merge_fact_annotations.py --annotations fact_annotations_*.json --map annotate.map.json \
      --output fact_annotations_full.json
  python build_fact_examples.py --annotations fact_annotations_full.json --output fact_examples.json
"""

import argparse
import glob
import json
import sys
from pathlib import Path

# Prompt templates — kept byte-for-byte in sync with score_facts_batch.py
# (RECALL_PROMPT / PRECISION_PROMPT). Copied rather than imported so this builder
# stays dependency-free (the scorer pulls in tiktoken/google at import time).
RECALL_PROMPT = """\
Given a list of REFERENCE facts and a list of CANDIDATE facts, identify which \
REFERENCE facts are semantically entailed by any of the CANDIDATE facts.
Respond with a JSON array of 0-based indices of the REFERENCE facts that are entailed. \
Return [] if none are entailed.

Judge as a clinician reviewing the chart would, not as a literal string matcher. The \
usual error is being too conservative — marking a REFERENCE fact absent because no \
CANDIDATE fact restates it in the same words, when the CANDIDATE facts plainly cover it.

## Rules
1. A REFERENCE fact is entailed if the CANDIDATE facts, taken together, assert it — it \
   need not be restated by a single CANDIDATE fact.
2. Both lists are atomized, so a dated event is split into a bare event anchor \
   ("A hemodynamic measurement was performed on <DATE>.") plus separate date-free \
   facts giving that event's details, findings, or results. Judge an anchor by its \
   details, not by its date: the anchor is entailed whenever the CANDIDATE facts assert \
   those details, even if no CANDIDATE fact mentions the date or names the event type.
3. Two events belong to the same episode of care when their dates match, fall within \
   about two weeks of each other, or one is given only as a month or an approximate \
   date. Do not require an exact date match: when one date gives only a month and year, \
   and they match the other date's month and year, treat them as the same event. Treat \
   events as separate only when the dates clearly indicate different encounters.
4. Within one episode of care, a REFERENCE fact describing a component, step, or \
   routine part of a larger event is entailed by a CANDIDATE fact describing that \
   larger event: an encounter entails the medications, fluids, and assessments given \
   during it, and a procedure entails the measurements and specimens it ordinarily \
   involves.
5. Wording and granularity need not match. A fact may be more specific in one respect \
   (naming the drug, device, or site) and less specific in another (a month rather than \
   a day); neither difference blocks entailment, in either direction.
6. Entail on semantic equivalence or logical implication, not only on restatement. If a \
   CANDIDATE fact means the same thing in different words, or logically implies the \
   REFERENCE fact, mark it entailed — a stated consequence of a symptom implies the \
   symptom, resuming or restarting a treatment implies that it was initiated, and a \
   documented trial of a treatment implies that it was given.
7. One supporting CANDIDATE fact is enough. Judge each REFERENCE fact on its own, and \
   do not withhold entailment because other CANDIDATE facts describe related events \
   pointing a different way: a separate or later event involving the same medication, \
   problem, or procedure does not cancel an earlier one.
8. Only two things block entailment: the CANDIDATE facts describe no related event or \
   encounter at all, or a CANDIDATE fact directly denies the REFERENCE fact itself. A \
   treatment tried without benefit still entails that it was given, but does not entail \
   that it helped.

### Reference facts
{REFERENCE_FACTS}

### Candidate facts
{CANDIDATE_FACTS}
"""

PRECISION_PROMPT = """\
Given a list of REFERENCE facts and a list of CANDIDATE facts, identify which \
CANDIDATE facts are semantically entailed by any of the REFERENCE facts.
Respond with a JSON array of 0-based indices of the CANDIDATE facts that are entailed. \
Return [] if none are entailed.

Judge as a clinician reviewing the chart would, not as a literal string matcher. The \
usual error is being too conservative — marking a CANDIDATE fact absent because no \
REFERENCE fact restates it in the same words, when the REFERENCE facts plainly cover it.

## Rules
1. A CANDIDATE fact is entailed if the REFERENCE facts, taken together, assert it — it \
   need not be restated by a single REFERENCE fact.
2. Both lists are atomized, so a dated event is split into a bare event anchor \
   ("An Emergency Department visit occurred on <DATE>.") plus separate \
   date-free facts giving that event's details, findings, or results. Judge an anchor by \
   its details, not by its date: the anchor is entailed whenever the REFERENCE facts \
   assert those details, even if no REFERENCE fact mentions the date or names the event \
   type.
3. Two events belong to the same episode of care when their dates match, fall within \
   about two weeks of each other, or one is given only as a month or an approximate \
   date. Do not require an exact date match: when one date gives only a month and year, \
   and they match the other date's month and year, treat them as the same event. Treat \
   events as separate only when the dates clearly indicate different encounters.
4. Within one episode of care, a CANDIDATE fact describing a component, step, or \
   routine part of a larger event is entailed by a REFERENCE fact describing that \
   larger event: an encounter entails the medications, fluids, and assessments given \
   during it, and a procedure entails the measurements and specimens it ordinarily \
   involves.
5. Wording and granularity need not match. A fact may be more specific in one respect \
   (naming the drug, device, or site) and less specific in another (a month rather than \
   a day); neither difference blocks entailment, in either direction.
6. Entail on semantic equivalence or logical implication, not only on restatement. If a \
   REFERENCE fact means the same thing in different words, or logically implies the \
   CANDIDATE fact, mark it entailed — a stated consequence of a symptom implies the \
   symptom, resuming or restarting a treatment implies that it was initiated, and a \
   documented trial of a treatment implies that it was given.
7. One supporting REFERENCE fact is enough. Judge each CANDIDATE fact on its own, and \
   do not withhold entailment because other REFERENCE facts describe related events \
   pointing a different way: a separate or later event involving the same medication, \
   problem, or procedure does not cancel an earlier one.
8. Only two things block entailment: the REFERENCE facts describe no related event or \
   encounter at all, or a REFERENCE fact directly denies the CANDIDATE fact itself. A \
   treatment tried without benefit still entails that it was given, but does not entail \
   that it helped.

### Reference facts
{REFERENCE_FACTS}

### Candidate facts
{CANDIDATE_FACTS}
"""


def _expand_paths(patterns: list[str]) -> list[Path]:
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
    for path in _expand_paths(patterns):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"  WARNING: {path} is not a JSON list — skipping")
            continue
        records.extend(data)
        print(f"  {path}: {len(data)} records")
    print(f"Loaded {len(records)} annotation records")
    return records


def _numbered(facts: list[str]) -> str:
    return "\n".join(f"{i}. {f}" for i, f in enumerate(facts))


def _view_for_direction(items: list[dict]):
    """Return (ordered facts, gold indices, per-item views) if the direction is fully
    annotated, else None (some fact left unmarked → ambiguous gold).

    Each view: {idx, fact, correct(bool), llm(bool|None), note(str|None)}.
    """
    if not items:
        return None
    ordered = sorted(items, key=lambda it: it["idx"])
    if any(it.get("entailed") is None for it in ordered):
        return None
    facts = [it["fact"] for it in ordered]
    gold = sorted(it["idx"] for it in ordered if it.get("entailed") is True)
    views = [{
        "idx": it["idx"], "fact": it["fact"],
        "correct": bool(it.get("entailed")),
        "llm": it.get("llm_consensus_entailed"),
        "note": (it.get("explanation") or None),
    } for it in ordered]
    return facts, gold, views


def build_examples(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (turns, previews).

    turns    — flat {direction, role, content} list for ICL (consecutive user→assistant).
    previews — structured, human-readable view of each example for review.
    """
    turns: list[dict] = []
    previews: list[dict] = []
    seen: set[tuple] = set()
    n_pairs = {"r": 0, "p": 0}

    for rec in records:
        key = (rec.get("question_id") or rec.get("record_id"), rec.get("source_name"))
        if key in seen:
            continue
        seen.add(key)

        recall = _view_for_direction(rec.get("recall_items") or [])
        prec = _view_for_direction(rec.get("precision_items") or [])
        if not recall and not prec:
            continue

        # Both directions share the same numbered ref/cand lists. Reference facts are
        # the recall_items; candidate facts are the precision_items.
        ref_facts = recall[0] if recall else [it["fact"] for it in sorted(rec.get("recall_items") or [], key=lambda x: x["idx"])]
        cand_facts = prec[0] if prec else [it["fact"] for it in sorted(rec.get("precision_items") or [], key=lambda x: x["idx"])]
        ref_numbered = _numbered(ref_facts)
        cand_numbered = _numbered(cand_facts)
        qid, src = rec.get("question_id"), rec.get("source_name")

        if recall:
            turns.append({"direction": "r", "role": "user",
                          "content": RECALL_PROMPT.format(REFERENCE_FACTS=ref_numbered, CANDIDATE_FACTS=cand_numbered)})
            turns.append({"direction": "r", "role": "assistant", "content": json.dumps(recall[1])})
            previews.append({
                "direction": "r", "question_id": qid, "source_name": src,
                "judged_label": "REFERENCE facts — is each present in the candidate answer?",
                "context_label": "Candidate answer facts (what we check against)",
                "context_facts": cand_facts,
                "items": recall[2], "gold": recall[1],
            })
            n_pairs["r"] += 1
        if prec:
            turns.append({"direction": "p", "role": "user",
                          "content": PRECISION_PROMPT.format(REFERENCE_FACTS=ref_numbered, CANDIDATE_FACTS=cand_numbered)})
            turns.append({"direction": "p", "role": "assistant", "content": json.dumps(prec[1])})
            previews.append({
                "direction": "p", "question_id": qid, "source_name": src,
                "judged_label": "CANDIDATE answer facts — is each present in the reference?",
                "context_label": "Reference facts (what we check against)",
                "context_facts": ref_facts,
                "items": prec[2], "gold": prec[1],
            })
            n_pairs["p"] += 1

    print(f"Built {n_pairs['r']} recall + {n_pairs['p']} precision example pairs "
          f"from {len(seen)} unique comparisons")
    return turns, previews


def _verdict(present: bool) -> str:
    return "PRESENT" if present else "ABSENT "


def write_preview(previews: list[dict], path: Path) -> None:
    out = []
    for i, p in enumerate(previews, 1):
        out.append("=" * 78)
        out.append(f"EXAMPLE {i}   ({'recall' if p['direction']=='r' else 'precision'})   "
                   f"question_id: {p['question_id']}  ·  source: {p['source_name']}")
        out.append("=" * 78)
        out.append(f"{p['context_label']}:")
        for j, f in enumerate(p["context_facts"]):
            out.append(f"  {j}. {f}")
        out.append(f"\n{p['judged_label']}")
        for it in p["items"]:
            llm = it["llm"]
            if llm is None:
                tag = "LLM n/a"
            elif bool(llm) == it["correct"]:
                tag = f"LLM {_verdict(bool(llm)).strip()} ✓"
            else:
                tag = f"CORRECTED — LLM said {_verdict(bool(llm)).strip()} ✗"
            out.append(f"  [{_verdict(it['correct'])} | {tag}]  {it['idx']}. {it['fact']}")
            if it["note"]:
                out.append(f"        note: {it['note']}")
        out.append(f"\nCorrect answer (indices marked PRESENT): {json.dumps(p['gold'])}")
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Build few-shot examples from fact-entailment annotations.")
    ap.add_argument("--annotations", nargs="+", required=True,
                    help="One or more fact_annotations_*.json files, globs, or directories")
    ap.add_argument("--output", default=None, help="Output JSON path (default: fact_examples.json next to first input)")
    args = ap.parse_args()

    records = load_annotations(args.annotations)
    turns, previews = build_examples(records)
    if not turns:
        sys.exit("No usable examples (need fully-annotated directions). Nothing written.")

    out = Path(args.output) if args.output else _expand_paths(args.annotations)[0].parent / "fact_examples.json"
    out.write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")
    preview = out.with_suffix(".txt")
    write_preview(previews, preview)
    print(f"\nWrote {len(turns)} turns → {out}")
    print(f"Preview → {preview}")
    print(f"Use with: python score_facts_batch.py ... --examples {out}")


if __name__ == "__main__":
    main()
