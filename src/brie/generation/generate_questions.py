"""Multi-turn question generation via Vertex AI Gemini batch prediction.

The original pipeline ran a 5-turn conversation per (patient, part) where each
turn sees the questions produced so far (so iterations don't duplicate). Batch
inference is for independent requests, so we **pipeline by turn across the whole
population**: every (patient, part) sits at the same turn `i` simultaneously, so
all turn-`i` requests go in one batch. After collecting responses we append each
to its own conversation and submit turn `i+1`. This yields NUM_ITERS batches
total while preserving exact multi-turn semantics per (patient, part).

Each turn is a distinct batch stage (``generate_questions_iter{i}``), so a crash
mid-run re-attaches finished turn jobs from their manifests on restart instead
of resubmitting.
"""

import argparse
import os

import pandas as pd

from .prompts.generate_questions import (
    GENERATE_HP_SYS, GENERATE_SYS, GENERATE_USER, FACT_SYS, HP_SYS, TIMESTAMP_SYS,
)
from .utils import safe_json_parse
from .batch_utils import send_gemini_batch, log_cost_estimate

NUM_ITERS = 5
PARTS = ["Both"]            # original conditions: ["Both", "Fact", "H&P"]
FACT_LIMIT = 40000         # above this, use the pre-filtered fact list


def system_prompt_builder(timestamp, fact_list=None, note=None):
    """Build the system prompt for an experimental condition (unchanged logic)."""
    if fact_list is not None:
        if note is not None:
            return GENERATE_SYS + TIMESTAMP_SYS.format(TIMESTAMP=timestamp) + HP_SYS.format(NOTE=note) + FACT_SYS.format(FACTS=fact_list)
        return GENERATE_SYS + TIMESTAMP_SYS.format(TIMESTAMP=timestamp) + FACT_SYS.format(FACTS=fact_list)
    return GENERATE_HP_SYS + TIMESTAMP_SYS.format(TIMESTAMP=timestamp) + HP_SYS.format(NOTE=note)


def build_system(part, fact_list, hp):
    timestamp = hp['reference_timestamp']
    if part == "Both":
        return system_prompt_builder(timestamp=timestamp, fact_list=fact_list, note=hp)
    elif part == "Fact":
        return system_prompt_builder(timestamp=timestamp, fact_list=fact_list)
    else:  # H&P
        return system_prompt_builder(timestamp=timestamp, note=hp)


def parse_args():
    """
    Example usage: python generate_questions.py --id 12,34 --input FACT_DIR --note NOTE_DIR --output OUT_DIR
    """
    parser = argparse.ArgumentParser(description="Question generation (Vertex batch)")
    parser.add_argument('-i', '--id', type=str, required=True,
                        help="List of IDs, separated by commas")
    parser.add_argument('-f', '--input', type=str, required=True, help="Input fact directory")
    parser.add_argument('-n', '--note', type=str, required=True, help="H&P input directory")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output directory")
    parser.add_argument('--poll-interval', type=int, default=60,
                        help="Seconds between batch-job status polls")
    parser.add_argument('--dry-run', action='store_true',
                        help="Estimate the first-turn request count and cost without submitting")
    return parser.parse_args()


def load_patient_inputs(id, input_dir, note_dir):
    """Load (fact_list, hp) for one patient, mirroring the original selection."""
    fact_list = pd.read_csv(f'{input_dir}/{id}.tsv', delimiter='\t', index_col=0)['fact'].tolist()
    hp_info = pd.read_json(f'{note_dir}/{id}_hp.json')
    hp = hp_info[hp_info['type'] == 'Full Note'].text.item()
    hp.pop('original_timestamp', None)
    if len(fact_list) > FACT_LIMIT:
        # Oversized patients first fall back to the visit-aware filtered list
        # ({id}_filtered.tsv from filter_facts.py; the phase2 method). A patient
        # still over FACT_LIMIT after filtering is an outpatient-heavy outlier
        # beyond what fits one prompt — phase2 excluded these at cohort
        # construction (note-count cap), so we exclude them here too (main()).
        filtered_path = f'{input_dir}/{id}_filtered.tsv'
        if os.path.exists(filtered_path):
            fact_list = pd.read_csv(filtered_path, delimiter='\t', index_col=0)['fact'].tolist()
    return fact_list, hp


def main(args):
    id_list = args.id.split(',')
    os.makedirs(args.output, exist_ok=True)
    state_dir = f"{args.output}/_batch_state"

    # Skip patients whose question file already exists (item-level resume).
    pending_ids = [id for id in id_list if not os.path.exists(f"{args.output}/{id}.csv")]
    if len(pending_ids) < len(id_list):
        print(f"[generate_questions] {len(id_list) - len(pending_ids)} already done; "
              f"{len(pending_ids)} pending.")

    # Build per-(patient, part) conversation state.
    convos = {}  # key (id, part) -> {"system": str, "contents": [...], "results": [...], "active": bool}
    for id in pending_ids:
        try:
            fact_list, hp = load_patient_inputs(id, args.input, args.note)
        except Exception as e:
            print(f"[generate_questions] {id}: failed to load inputs: {e}")
            continue
        # Exclude oversized outliers: still over FACT_LIMIT even after the
        # visit-aware filter. Consistent with phase2's cohort-level note cap.
        if len(fact_list) > FACT_LIMIT:
            print(f"[generate_questions] {id}: {len(fact_list)} facts exceed "
                  f"FACT_LIMIT={FACT_LIMIT} after filtering; excluding oversized patient")
            continue
        for part in PARTS:
            convos[(id, part)] = {
                "system": build_system(part, fact_list, hp),
                "reference_timestamp": hp["reference_timestamp"],
                "contents": [],
                "results": [],
                "active": True,
            }

    if not convos:
        print("[generate_questions] Nothing to do.")
        return

    if args.dry_run:
        # Estimate the first turn (every subsequent turn is the same count, growing in size).
        requests = []
        for (id, part), c in convos.items():
            contents = c["contents"] + [{"role": "user", "parts": [{"text": GENERATE_USER}]}]
            requests.append({"custom_id": f"{id}:{part}:0", "system_instructions": c["system"],
                             "contents": contents})
        print(f"[generate_questions] Dry-run estimates turn 0 only; "
              f"{NUM_ITERS} turns total (each grows as the conversation accumulates).")
        log_cost_estimate(requests, "generate_questions_iter0", est_output_tokens_per_req=4000)
        return

    # ── Turn pipeline: one batch per iteration across all active conversations.
    for i in range(NUM_ITERS):
        requests = []
        for (id, part), c in convos.items():
            if not c["active"]:
                continue
            # Append this turn's user request, then snapshot contents for the request.
            c["contents"].append({"role": "user", "parts": [{"text": GENERATE_USER}]})
            requests.append({
                "custom_id": f"{id}:{part}:{i}",
                "system_instructions": c["system"],
                "contents": list(c["contents"]),
            })
        if not requests:
            break

        results = send_gemini_batch(requests, f"generate_questions_iter{i}",
                                    state_dir, poll_interval=args.poll_interval)

        for (id, part), c in convos.items():
            if not c["active"]:
                continue
            text = results.get(f"{id}:{part}:{i}")
            if text is None:
                print(f"[generate_questions] {id}/{part} iter {i}: failed response; dropping conversation")
                c["active"] = False
                continue
            # Append the assistant turn so the next iteration sees it.
            c["contents"].append({"role": "model", "parts": [{"text": text}]})
            try:
                items = safe_json_parse(text)
                if isinstance(items, dict):
                    items = [items]
                for item in items:
                    item["part"] = part
                    item["iteration"] = i
                    item["reference_timestamp"] = c["reference_timestamp"]
                    c["results"].append(item)
            except Exception as e:
                print(f"[generate_questions] {id}/{part} iter {i}: failed to parse: {e}")

    # ── Write one CSV per patient.
    for id in pending_ids:
        rows = []
        for part in PARTS:
            c = convos.get((id, part))
            if c:
                rows.extend(c["results"])
        if not rows:
            print(f"[generate_questions] {id}: no questions produced")
            continue
        pd.DataFrame(rows).to_csv(f"{args.output}/{id}.csv", index=False)
        print(f"[generate_questions] {id}: wrote {len(rows)} questions")


if __name__ == '__main__':
    args = parse_args()
    main(args)
