"""Select the best questions per patient via Vertex AI Gemini batch prediction.

Faithful batch port of the original ``export_annotations.filter_dataframe``
selection step: the LLM is shown the admission H&P plus all of a patient's
generated questions and returns the ~10 best (``selected_questions``). Because
the selection is per-patient (every question for a patient is judged together
for relevance / definition / diversity / no-leakage), each patient is **one**
batch request keyed by ``person_id``.

Output: one ``{id}.csv`` per patient containing only the selected questions, with
``question_rewrite`` exposed as ``natural_query`` (plus a combined
``questions_filtered.csv`` rebuilt across all patients). Resume is item-level
(skip patients whose ``{id}.csv`` already exists) plus the job-level re-attach in
``send_gemini_batch``.
"""

import argparse
import json
import os

import pandas as pd

from .utils import safe_json_parse
from .batch_utils import send_gemini_batch, log_cost_estimate
# The original "10 best of 15" selector prompt (returns {"selected_questions": [...]}).
from .prompts.export_annotations import FILTER_SYS as SELECT_SYS

STAGE = "filter_questions"

# Columns shown to the selector for each question (matches the original filter_input).
SELECT_FIELDS = ["question_id", "question", "answer", "clinical_relevance_rationale"]
# Columns carried into the final per-patient output.
OUTPUT_COLS = ["person_id", "question_id", "natural_query", "question", "answer",
               "facts", "question_type", "clinical_relevance_rationale",
               "reference_timestamp", "part", "iteration"]


def parse_args():
    """
    Example usage:
        python filter_questions.py --id 12,34 --input QDIR --note NOTEDIR --output OUTDIR
    """
    parser = argparse.ArgumentParser(description="Question selection (Vertex batch)")
    parser.add_argument('-i', '--id', type=str, required=True,
                        help="List of IDs, separated by commas")
    parser.add_argument('-n', '--note', type=str, required=True,
                        help="Note directory (holds {id}_hp.json for H&P context)")
    parser.add_argument('-f', '--input', type=str, required=True,
                        help="Input (questions) directory holding {id}.csv from generate_questions")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Output directory for selected {id}.csv files")
    parser.add_argument('--poll-interval', type=int, default=60,
                        help="Seconds between batch-job status polls")
    parser.add_argument('--dry-run', action='store_true',
                        help="Estimate request count and cost without submitting a job")
    return parser.parse_args()


def load_hp_text(note_dir, id):
    """Return the H&P note text used as selection context (mirrors generate_questions)."""
    hp_info = pd.read_json(f'{note_dir}/{id}_hp.json')
    hp = hp_info[hp_info['type'] == 'Full Note'].text.item()
    # hp is {original_timestamp, reference_timestamp, note_title, text}
    return hp.get('text', '')


def build_question_frame(input_dir, id):
    """Load one patient's generated questions, assign question_id, expose natural_query."""
    df = pd.read_csv(f'{input_dir}/{id}.csv')
    df['person_id'] = str(id)
    df['question_id'] = df.index
    # question_rewrite is the natural-clinician phrasing -> natural_query.
    if 'question_rewrite' in df.columns:
        df['natural_query'] = df['question_rewrite']
    else:
        df['natural_query'] = df['question']
    if 'fact_subset' in df.columns and 'facts' not in df.columns:
        df['facts'] = df['fact_subset']
    return df


def build_request(id, df, hp_text):
    """One selection request per patient: H&P + all questions -> select the best."""
    # Show the model the natural-query phrasing as "question" (as the original did).
    questions = []
    for _, row in df.iterrows():
        questions.append({
            "question_id": int(row['question_id']),
            "question": row.get('natural_query', row.get('question', '')),
            "answer": row.get('answer', ''),
            "clinical_relevance_rationale": row.get('clinical_relevance_rationale', ''),
        })
    filter_input = {"H&P_note": hp_text, "questions": questions}
    return {
        "custom_id": str(id),
        "system_instructions": SELECT_SYS,
        "user_prompt": json.dumps(filter_input, indent=2, ensure_ascii=False),
    }


def write_combined(output_dir, id_list):
    """Rebuild the combined questions file from whatever per-patient files exist."""
    frames = []
    for id in id_list:
        path = f'{output_dir}/{id}.csv'
        if os.path.exists(path) and os.path.getsize(path) > 0:
            frames.append(pd.read_csv(path))
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    combined['within_subject_question_id'] = combined['question_id']
    combined['question_id'] = (
        combined['person_id'].astype(str)
        + '_'
        + combined['within_subject_question_id'].astype(str)
    )
    combined['timestamp'] = combined['reference_timestamp']
    out_path = f'{output_dir}/questions_filtered.csv'
    combined.to_csv(out_path, index=False)
    print(f"[{STAGE}] wrote {len(combined)} selected questions across "
          f"{combined['person_id'].nunique()} patients -> {out_path}")


def main(args):
    id_list = args.id.split(',')
    state_dir = f'{args.output}/_batch_state'
    os.makedirs(args.output, exist_ok=True)

    # Item-level resume: skip patients already selected.
    pending = [id for id in id_list if not os.path.exists(f'{args.output}/{id}.csv')]
    if len(pending) < len(id_list):
        print(f"[{STAGE}] {len(id_list) - len(pending)} already done; {len(pending)} pending.")

    requests, frame_by_id = [], {}
    for id in pending:
        try:
            df = build_question_frame(args.input, id)
            hp_text = load_hp_text(args.note, id)
        except Exception as e:
            print(f"[{STAGE}] {id}: failed to load inputs: {e}")
            continue
        if df.empty:
            print(f"[{STAGE}] {id}: no questions to select from")
            continue
        frame_by_id[id] = df
        requests.append(build_request(id, df, hp_text))

    if args.dry_run:
        log_cost_estimate(requests, STAGE, est_output_tokens_per_req=600)
        return

    if not requests:
        print(f"[{STAGE}] Nothing to do.")
        write_combined(args.output, id_list)
        return

    results = send_gemini_batch(requests, STAGE, state_dir, poll_interval=args.poll_interval)

    for id, df in frame_by_id.items():
        text = results.get(str(id))
        if text is None:
            print(f"[{STAGE}] {id}: no/failed response (see {STAGE}_failed.jsonl)")
            continue
        try:
            parsed = safe_json_parse(text)
            selected = parsed['selected_questions'] if isinstance(parsed, dict) else parsed
            keep_ids = {int(item['question_id']) for item in selected}
        except Exception as e:
            print(f"[{STAGE}] {id}: failed to parse selection: {e}")
            continue

        sel = df[df['question_id'].isin(keep_ids)].copy()
        if 'justification' not in sel.columns:
            just = {int(item['question_id']): item.get('justification', '')
                    for item in selected if isinstance(item, dict)}
            sel['selection_justification'] = sel['question_id'].map(just)
        cols = [c for c in OUTPUT_COLS if c in sel.columns]
        if 'selection_justification' in sel.columns:
            cols.append('selection_justification')
        sel[cols].to_csv(f'{args.output}/{id}.csv', index=False)
        print(f"[{STAGE}] {id}: selected {len(sel)}/{len(df)} questions")

    write_combined(args.output, id_list)


if __name__ == '__main__':
    args = parse_args()
    main(args)
