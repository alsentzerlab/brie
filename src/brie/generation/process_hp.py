# default
import argparse
import os

# pip
import pandas as pd

# custom
from .utils import safe_json_parse
from .batch_utils import send_gemini_batch, log_cost_estimate, load_completed_ids
from .prompts.process_hp import HP_SYS

STAGE = "process_hp"

def parse_args():
    """
    Example usage: python process_hp.py --id 12,34,56 --notes NOTE_DIR --phase2 OUT_DIR
    """
    parser = argparse.ArgumentParser(description="H&P processing (Vertex batch)")
    parser.add_argument('-i', '--id', type=str, required=True,
                        help="List of IDs, separated by commas")
    parser.add_argument('-n', '--notes', type=str, required=True, help="H&P input directory")
    parser.add_argument('-p', '--phase2', type=str, required=True, help="Output directory")
    parser.add_argument('--poll-interval', type=int, default=60,
                        help="Seconds between batch-job status polls")
    parser.add_argument('--dry-run', action='store_true',
                        help="Estimate request count and cost without submitting a job")
    return parser.parse_args()


def load_hp_row(notes_dir, id):
    """Load the 'Full Note' H&P dict for one person_id and tag it with person_id."""
    hp_info = pd.read_json(f'{notes_dir}/{id}_hp.json')
    hp = hp_info[hp_info['type'] == 'Full Note'].text.item()
    hp['person_id'] = id
    return hp


def main(args):
    id_list = args.id.split(',')
    os.makedirs(args.phase2, exist_ok=True)
    out_path = f'{args.phase2}/hp_combined.csv'
    state_dir = f'{args.phase2}/_batch_state'

    # Build the per-patient H&P rows (preserves all original H&P dict fields).
    rows = {id: load_hp_row(args.notes, id) for id in id_list}

    # Item-level resume: skip person_ids already present with a clinical_summary.
    done = load_completed_ids(out_path, "person_id", nonempty_col="clinical_summary")
    pending_ids = [id for id in id_list if str(id) not in done]
    if done:
        print(f"[{STAGE}] {len(done)} already processed; {len(pending_ids)} pending.")

    requests = [
        {"custom_id": str(id), "user_prompt": str(rows[id]['text']), "system_instructions": HP_SYS}
        for id in pending_ids
    ]

    if args.dry_run:
        log_cost_estimate(requests, STAGE, est_output_tokens_per_req=400)
        return

    if not requests:
        print(f"[{STAGE}] Nothing to do.")
        return

    results = send_gemini_batch(requests, STAGE, state_dir, poll_interval=args.poll_interval)

    # Scatter strictly by custom_id (== person_id); never positional.
    new_rows = []
    for id in pending_ids:
        text = results.get(str(id))
        if text is None:
            print(f"[{STAGE}] {id}: no/failed response (see {STAGE}_failed.jsonl)")
            continue
        try:
            parsed = safe_json_parse(text)
            row = dict(rows[id])
            row['reason_for_admission'] = parsed['reason_for_admission']
            row['clinical_summary'] = parsed['clinical_summary']
            new_rows.append(row)
        except Exception as e:
            print(f"[{STAGE}] {id}: failed to parse response: {e}")

    if not new_rows:
        print(f"[{STAGE}] No new rows produced.")
        return

    df = pd.DataFrame(new_rows)
    df['person_id'] = df['person_id'].astype(str)

    # Merge with any previously-written rows (resume) and persist.
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        prev = pd.read_csv(out_path)
        prev['person_id'] = prev['person_id'].astype(str)
        df = pd.concat([prev[~prev['person_id'].isin(df['person_id'])], df], ignore_index=True)

    df.to_csv(out_path, index=False)
    print(f"[{STAGE}] Wrote {len(new_rows)} new row(s) -> {out_path}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
