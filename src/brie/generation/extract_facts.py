"""Atomic fact extraction + deduplication via Vertex AI Gemini batch prediction.

Extraction is pooled into a single cross-patient batch (one request per note
chunk). Deduplication preserves the original two-phase algorithm (within-batch
then iterative cross-batch passes) and the exact index bookkeeping, but each
pass is executed as one cross-patient batch instead of a thread pool.

Per-patient outputs are unchanged: ``{id}_raw.json`` (raw facts + note
provenance) and ``{id}.tsv`` (deduplicated). Both are skipped if already
present, so a rerun resumes.
"""

import argparse
import os
import re
import random
import json

import pandas as pd

# custom
from .prompts.extract_facts import EXTRACT_SYS, DEDUP_SYS, format_extract, format_dedup
from .utils import load_notes, safe_json_parse
from .batch_utils import send_gemini_batch, log_cost_estimate

# Regex to check if a fact already ends with (YYYY-MM-DD)
PATTERN = r"\(\d{4}-\d{2}-\d{2}\)$"
BATCH_SIZE = 500
CHUNK_SIZE = 20000
MAX_ITER = 3
THRESHOLD = 5

random.seed(42)


# ── Chunking helpers ──────────────────────────────────────────────────────────
def chunk_note(text: str, chunk_size: int):
    """Split note text into fixed-size chunks."""
    if len(text) < chunk_size:
        return [text]
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_facts(facts_list, batch_size=BATCH_SIZE):
    """Batch a fact list into (start_index, sublist) pairs."""
    return [(i, facts_list[i:i + batch_size]) for i in range(0, len(facts_list), batch_size)]


def shuffle_with_mapping(original_list):
    """Shuffle a list; return the shuffled list and {new_idx: old_idx} mapping."""
    n = len(original_list)
    indices = list(range(n))
    random.shuffle(indices)
    shuffled_list = [original_list[i] for i in indices]
    mapping = {new_idx: old_idx for new_idx, old_idx in enumerate(indices)}
    return shuffled_list, mapping


# ── Phase A: extraction ───────────────────────────────────────────────────────
def build_extraction_requests(id, notes_df):
    """Build one request per note chunk for a patient.

    Returns (requests, registry) where registry[custom_id] holds the chunk
    metadata needed to reconstruct {id}_raw.json.
    """
    requests = []
    registry = {}
    for _, row in notes_df.iterrows():
        for chunk_idx, chunk in enumerate(chunk_note(row["text"], CHUNK_SIZE)):
            cid = f"{id}:{row.name}:{chunk_idx}"
            registry[cid] = {
                "note_number": row.name,
                "note_date": str(row["note_date"]),
                "note_title": row["note_title"],
                "chunk_num": chunk_idx,
                "text": chunk,
                "note_date_obj": row["note_date"],
            }
            requests.append({
                "custom_id": cid,
                "user_prompt": format_extract(note_date=str(row["note_date"]), text=chunk),
                "system_instructions": EXTRACT_SYS,
            })
    return requests, registry


def parse_extraction(text, note_date_obj):
    """Parse a claims response and append a date suffix where missing."""
    claims = safe_json_parse(text)["claims"]
    facts = []
    for f in claims:
        facts.append(f if re.search(PATTERN, f) else f"{f} ({note_date_obj.date()})")
    return facts


def assemble_raw(id, registry, results, output_dir):
    """Build facts_list + write {id}_raw.json from batch results (ordered by chunk)."""
    facts_list = []
    chunk_list = []
    idx = 0
    for cid in sorted(registry, key=lambda c: (registry[c]["note_number"], registry[c]["chunk_num"])):
        text = results.get(cid)
        if text is None:
            print(f"[extract_facts] {cid}: no/failed extraction response")
            continue
        try:
            facts = parse_extraction(text, registry[cid]["note_date_obj"])
        except Exception as e:
            print(f"[extract_facts] {cid}: failed to parse claims: {e}")
            continue
        meta = registry[cid]
        note_text = (f"Note Title: {meta['note_title']}\n"
                     f"Note Date: {meta['note_date']}\n"
                     f"Note Excerpt: {meta['text']}\n")
        for fact in facts:
            facts_list.append(fact)
            chunk_list.append({"index": idx, "fact": fact, "note_text": note_text})
            idx += 1

    # Only checkpoint when at least one fact was extracted, so a fully-failed
    # patient (e.g. a batch error) re-extracts on the next run instead of being
    # frozen behind an empty {id}_raw.json.
    if chunk_list:
        with open(f"{output_dir}/{id}_raw.json", "w+") as ofile:
            json.dump(chunk_list, ofile)
    else:
        print(f"[extract_facts] {id}: no facts extracted; not writing raw.json")
    return facts_list


def load_raw_facts(raw_path):
    """Read facts from an existing {id}_raw.json, or None if empty/invalid."""
    try:
        df = pd.read_json(raw_path)
    except ValueError:
        return None
    if df.empty or 'fact' not in df.columns:
        return None
    return df['fact'].to_list()


# ── Phase B: deduplication (cross-patient batched passes) ─────────────────────
def _redundancy_requests(groups):
    """groups: list of {custom_id, fact_map} -> batch request dicts."""
    return [
        {"custom_id": g["custom_id"],
         "user_prompt": format_dedup(input_fact_list=g["fact_map"]),
         "system_instructions": DEDUP_SYS}
        for g in groups
    ]


def _parse_redundant(text):
    if text is None:
        return []
    try:
        return safe_json_parse(text)["redundant_fact_indices"]
    except Exception:
        return []


def deduplicate_all(patients_facts, state_dir, poll_interval,
                    max_iter=MAX_ITER, threshold=THRESHOLD):
    """Run within-batch + iterative cross-batch dedup across all patients.

    Each pass is one cross-patient Gemini batch. Per-patient index math is
    identical to the original thread-pool implementation.
    """
    to_remove = {id: set() for id in patients_facts}

    # ── Within-batch pass (one batch for all patients).
    groups, owner = [], {}
    for id, facts in patients_facts.items():
        for start_idx, batch in chunk_facts(facts):
            cid = f"{id}:within:{start_idx}"
            fact_map = {start_idx + i: fact for i, fact in enumerate(batch)}
            groups.append({"custom_id": cid, "fact_map": fact_map})
            owner[cid] = id
    if groups:
        results = send_gemini_batch(_redundancy_requests(groups), "extract_facts_dedup_within",
                                    state_dir, poll_interval=poll_interval)
        for cid, text in results.items():
            to_remove[owner[cid]].update(_parse_redundant(text))

    # ── Cross-batch passes (one batch per pass).
    active = set(patients_facts)
    for k in range(max_iter):
        if not active:
            break
        groups, owner, mapping_by_id = [], {}, {}
        for id in active:
            facts = patients_facts[id]
            keep_indices = [i for i in range(len(facts)) if i not in to_remove[id]]
            keep_facts_shuffled, keep_indices_mapping = shuffle_with_mapping(
                [facts[i] for i in keep_indices]
            )
            mapping_by_id[id] = keep_indices_mapping
            for start_idx, group in chunk_facts(keep_facts_shuffled):
                cid = f"{id}:cross{k}:{start_idx}"
                fact_map = {start_idx + i: fact for i, fact in enumerate(group)}
                groups.append({"custom_id": cid, "fact_map": fact_map})
                owner[cid] = id
        if not groups:
            break

        prev_counts = {id: len(to_remove[id]) for id in active}
        results = send_gemini_batch(_redundancy_requests(groups), f"extract_facts_dedup_cross{k}",
                                    state_dir, poll_interval=poll_interval)
        for cid, text in results.items():
            id = owner[cid]
            redundant = _parse_redundant(text)
            mapping = mapping_by_id[id]
            global_redundant = [mapping[i] for i in redundant if i in mapping]
            to_remove[id].update(global_redundant)

        for id in list(active):
            if len(to_remove[id]) - prev_counts[id] < threshold:
                active.discard(id)

    out = {}
    for id, facts in patients_facts.items():
        deduped = [facts[i] for i in range(len(facts)) if i not in to_remove[id]]
        out[id] = (deduped, to_remove[id])
    return out


# ── Orchestration ─────────────────────────────────────────────────────────────
def parse_args():
    """
    Example usage: python extract_facts.py --id 12,34,56 --input IN_DIR --output OUT_DIR
    """
    parser = argparse.ArgumentParser(description="Fact extraction (Vertex batch)")
    parser.add_argument('-i', '--id', type=str, required=True,
                        help="List of IDs, separated by commas")
    parser.add_argument('-f', '--input', type=str, required=True, help="Input notes directory")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output fact directory")
    parser.add_argument('--poll-interval', type=int, default=60,
                        help="Seconds between batch-job status polls")
    parser.add_argument('--dry-run', action='store_true',
                        help="Estimate request count and cost without submitting a job")
    return parser.parse_args()


def main(args):
    id_list = args.id.split(',')
    os.makedirs(args.output, exist_ok=True)
    state_dir = f"{args.output}/_batch_state"

    # ── Phase A: build extraction requests for patients without a raw fact file.
    extract_requests = []
    registry_by_id = {}
    raw_facts = {}  # id -> facts_list (from existing raw.json)
    for id in id_list:
        raw_path = f"{args.output}/{id}_raw.json"
        if os.path.exists(raw_path):
            facts = load_raw_facts(raw_path)
            if facts is not None:
                raw_facts[id] = facts
                continue
            # empty/invalid (e.g. from a failed earlier run) — re-extract.
            print(f"[extract_facts] {id}: existing raw.json is empty/invalid; re-extracting")
        notes_df = load_notes(f"{args.input}/{id}_subsetrecords.json")
        reqs, registry = build_extraction_requests(id, notes_df)
        registry_by_id[id] = registry
        extract_requests.extend(reqs)

    if args.dry_run:
        log_cost_estimate(extract_requests, "extract_facts_extract", est_output_tokens_per_req=1500)
        # Estimate the within-batch dedup pass for patients whose facts are known.
        groups = []
        for id, facts in raw_facts.items():
            for start_idx, batch in chunk_facts(facts):
                groups.append({"custom_id": f"{id}:within:{start_idx}",
                               "fact_map": {start_idx + i: f for i, f in enumerate(batch)}})
        log_cost_estimate(_redundancy_requests(groups), "extract_facts_dedup_within (1st pass)",
                          est_output_tokens_per_req=200)
        return

    # ── Run extraction batch, assemble raw.json per patient.
    if extract_requests:
        results = send_gemini_batch(extract_requests, "extract_facts_extract",
                                    state_dir, poll_interval=args.poll_interval)
        for id, registry in registry_by_id.items():
            facts_list = assemble_raw(id, registry, results, args.output)
            print(f"[extract_facts] {id}: extracted {len(facts_list)} facts")
            if facts_list:
                raw_facts[id] = facts_list

    # ── Phase B: deduplicate patients that lack a {id}.tsv.
    to_dedup = {id: raw_facts[id] for id in id_list
                if not os.path.exists(f"{args.output}/{id}.tsv") and id in raw_facts}
    if to_dedup:
        deduped = deduplicate_all(to_dedup, state_dir, args.poll_interval)
        for id, (deduped_list, removed) in deduped.items():
            with open(f"{args.output}/{id}.tsv", "w+") as ofile:
                ofile.write("index\tfact\n")
                for idx, fact in enumerate(deduped_list):
                    ofile.write(f"{idx}\t{fact}\n")
            print(f"[extract_facts] {id}: kept {len(deduped_list)}, removed {len(removed)}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
