"""Tag each question with clinical topics via Vertex AI Gemini batch prediction.

Batch port of ``export_annotations.apply_topic_module``: each question is shown
to the model with ``TOPIC_SYS``/``TOPIC_USER`` and gets up to 3 topics from the
fixed category list (returned under ``topics``). The original ran one online call
per question; here every question across the input file goes in **one** batch,
keyed by the question's unique id.

Input is a single questions CSV (e.g. the ``questions_filtered.csv`` produced by
filter_questions.py). Output is the same rows with an added ``question_topics``
column (list of topic strings). Resume is item-level: questions whose id already
carries a non-empty ``question_topics`` in the output are skipped.
"""

import argparse
import os

import pandas as pd

from .utils import safe_json_parse
from .batch_utils import send_gemini_batch, log_cost_estimate, load_completed_ids
from .prompts.export_annotations import TOPIC_SYS, TOPIC_USER

STAGE = "question_topics"


def parse_args():
    """
    Example usage:
        python get_topics.py --input questions_filtered.csv --output questions_topics.csv
    """
    parser = argparse.ArgumentParser(description="Question topic tagging (Vertex batch)")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Input questions CSV (e.g. questions_filtered.csv)")
    parser.add_argument('-o', '--output', type=str,
                        help="Output CSV (defaults to the input path, written in place)")
    parser.add_argument('--question-column', type=str, default="natural_query",
                        help="Column holding the question text (default: natural_query)")
    parser.add_argument('--id-column', type=str, default="question_id",
                        help="Column with a unique id per question (default: id)")
    parser.add_argument('--poll-interval', type=int, default=60,
                        help="Seconds between batch-job status polls")
    parser.add_argument('--dry-run', action='store_true',
                        help="Estimate request count and cost without submitting a job")
    return parser.parse_args()


def resolve_id_column(df, id_column):
    """Return a Series of unique row ids, deriving one if id_column is absent."""
    if id_column in df.columns:
        return df[id_column].astype(str)
    if {'person_id', 'question_id'}.issubset(df.columns):
        return df['person_id'].astype(str) + '_' + df['question_id'].astype(str)
    return df.index.astype(str)


def resolve_question_column(df, question_column):
    """Pick the question text column, falling back like the eval scripts do."""
    for col in (question_column, 'natural_query', 'question'):
        if col in df.columns:
            return col
    raise ValueError(f"No question column found (tried {question_column}, natural_query, question)")


def main(args):
    out_path = args.output or args.input
    state_dir = f"{os.path.dirname(os.path.abspath(out_path))}/_batch_state"

    df = pd.read_csv(args.input)
    df['_cid'] = resolve_id_column(df, args.id_column)
    if df['_cid'].duplicated().any():
        dups = df.loc[df['_cid'].duplicated(), '_cid'].drop_duplicates().head(5).tolist()
        raise ValueError(f"Non-unique ids in {args.id_column!r} (e.g. {dups}); "
                         "topics are matched by id, so it must be unique.")
    q_col = resolve_question_column(df, args.question_column)

    done = load_completed_ids(out_path, args.id_column, nonempty_col="question_topics")
    requests = []
    for _, row in df.iterrows():
        cid = row['_cid']
        if cid in done:
            continue
        requests.append({
            "custom_id": cid,
            "system_instructions": TOPIC_SYS,
            "user_prompt": TOPIC_USER.format(QUESTION=row[q_col]),
        })

    if args.dry_run:
        log_cost_estimate(requests, STAGE, est_output_tokens_per_req=60)
        return

    if not requests:
        print(f"[{STAGE}] Nothing to do.")
        return

    results = send_gemini_batch(requests, STAGE, state_dir, poll_interval=args.poll_interval)

    topics_by_cid = {}
    for cid in [r["custom_id"] for r in requests]:
        text = results.get(cid)
        if text is None:
            print(f"[{STAGE}] {cid}: no/failed response (see {STAGE}_failed.jsonl)")
            continue
        try:
            parsed = safe_json_parse(text)
            topics_by_cid[cid] = parsed['topics'] if isinstance(parsed, dict) else parsed
        except Exception as e:
            print(f"[{STAGE}] {cid}: failed to parse topics: {e}")

    # Merge into any previously-completed output so resumed runs accumulate.
    prev_topics = {}
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0 and out_path != args.input:
        prev = pd.read_csv(out_path)
        if args.id_column in prev.columns and 'question_topics' in prev.columns:
            prev_topics = dict(zip(prev[args.id_column].astype(str), prev['question_topics']))

    def topics_for(cid):
        if cid in topics_by_cid:
            return topics_by_cid[cid]
        return prev_topics.get(cid)

    df['question_topics'] = df['_cid'].map(topics_for)
    df = df.drop(columns=['_cid'])
    df.to_csv(out_path, index=False)
    tagged = df['question_topics'].notna().sum()
    print(f"[{STAGE}] tagged {tagged}/{len(df)} questions -> {out_path}")


if __name__ == '__main__':
    main(parse_args())
