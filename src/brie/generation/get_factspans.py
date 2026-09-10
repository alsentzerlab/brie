# default
import argparse
import json
import ast
import concurrent.futures
import logging
import os
import re
import traceback

# pip
import pandas as pd
import tqdm

# custom
from .utils import send_single_message
from .prompts.export_annotations import EVIDENCE_SYS, ANSWER_SPAN_USER, ANSWER_SPAN_SYS


def parse_args():
    """
    Description: Parse the arguments
    Example usage:
        python -m brie.generation.get_factspans \
            --input PATH_TO_INPUT_CSV \
            --fact PATH_TO_FACT_DIR \
            --output PATH_TO_OUTPUT_JSON \
            --log PATH_TO_LOG_DIR \
            --workers 4
    """
    parser = argparse.ArgumentParser(description="Export span annotations linking answers to clinical note evidence via facts")
    parser.add_argument('-i', '--input',   type=str, required=True,
                        help="Input CSV: must have columns question_id, subquestion_id, answer, facts")
    parser.add_argument('-f', '--fact',    type=str, required=True,
                        help="Directory containing per-patient fact JSON files (<patient_id>_raw.json)")
    parser.add_argument('-o', '--output',  type=str, required=True,
                        help="Output JSON file path")
    parser.add_argument('-l', '--log',     type=str, default='logs',
                        help="Directory to write per-patient log files (default: ./logs)")
    parser.add_argument('-w', '--workers', type=int, default=10,
                        help="Number of parallel workers for patient-level processing")
    return parser.parse_args()


def get_patient_logger(name: str, log_dir: str) -> logging.Logger:
    """
    Create (or retrieve) a logger that writes exclusively to
    <log_dir>/<name>.log. Each run overwrites the previous file (mode='w')
    so re-runs don't accumulate stale entries.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f'patient.{name}')
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if the logger is somehow reused in the same process
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, f'{name}.log'), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s  %(levelname)-8s  %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(fh)

    # Don't propagate to the root logger so stdout stays clean
    logger.propagate = False
    return logger


def _parse_json_response(response: str) -> dict:
    """
    Robustly parse a JSON object from a model response that may be wrapped
    in markdown fences (```json ... ```) or returned as plain JSON.
    """
    cleaned = re.sub(r'^```(?:json)?\s*', '', response.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned.strip())
    return json.loads(cleaned)


def get_fact_text_module(fact: str, text: str, logger: logging.Logger):
    """
    Ask the model to identify substrings in `text` that support `fact`.
    Returns (fact, text, spans) where spans is a list of [start, end] pairs.
    """
    logger.info(f'[fact→note ] START  | fact: {fact!r}')
    try:
        response = send_single_message(
            system_instructions=EVIDENCE_SYS,
            user_prompt=json.dumps({'fact': fact, 'note': text})
        )
        parsed = _parse_json_response(response)

        spans = []
        for evidence in parsed.get('evidence', []):
            start = text.find(evidence)
            if start != -1:
                spans.append([start, start + len(evidence)])

        logger.info(f'[fact→note ] DONE   | fact: {fact!r} | {len(spans)} span(s) found')
        return (fact, text, spans)
    except Exception as e:
        logger.error(f'[fact→note ] ERROR  | fact: {fact!r} | {e}')
        raise

def get_answer_text_module(fact_list: list, answer: str, subquestion_id, logger: logging.Logger):
    """
    Ask the model to map each fact in `fact_list` to a verbatim substring within `answer`,
    then resolve each substring to a character span via str.find().
    Returns (mappings, subquestion_id).
    """
    preview = answer[:80].replace('\n', ' ')
    logger.info(f'[fact→answer] START  | subquestion_id: {subquestion_id!r} | '
                f'{len(fact_list)} fact(s) | answer preview: {preview!r}...')
    try:
        response = send_single_message(
            system_instructions=ANSWER_SPAN_SYS,
            user_prompt=ANSWER_SPAN_USER.format(
                FACTS_LIST=json.dumps(fact_list),
                ANSWER=answer
            )
        )
        parsed = _parse_json_response(response)

        mappings = []
        for m in parsed['mappings']:
            substring = m.get('substring')
            if substring is not None:
                start = answer.find(substring)
                span = [start, start + len(substring)] if start != -1 else None
            else:
                span = None
            mappings.append({'fact': m['fact'], 'span': span})

        matched   = sum(1 for m in mappings if m.get('span') is not None)
        unmatched = len(mappings) - matched
        logger.info(f'[fact→answer] DONE   | subquestion_id: {subquestion_id!r} | '
                    f'{matched} matched, {unmatched} unmatched')
        return mappings, subquestion_id
    except Exception as e:
        logger.error(f'[fact→answer] ERROR  | subquestion_id: {subquestion_id!r} | {e}')
        raise

def process_patient(args, name: str, group: pd.DataFrame):
    """
    For a single patient:
      1. Load their fact→note_text mapping from <fact_dir>/<name>_raw.json
      2. For every unique fact referenced across all sub-questions, find its
         supporting span(s) in the source note (fact2note).
      3. For every sub-question row, map each fact to a span in the free-text answer (facts2answer).
      4. Combine: for facts that are grounded in both the note AND the answer,
         record both spans together.

    Returns a list of dicts, one per sub-question.
    """
    logger = get_patient_logger(name, args.log)
    logger.info(f'=== START patient {name!r} | {len(group)} sub-question row(s) ===')

    # ------------------------------------------------------------------
    # 1. Load the raw fact file for this patient
    # ------------------------------------------------------------------
    fact_file = os.path.join(args.fact, f'{name}_raw.json')
    logger.info(f'Loading fact file: {fact_file}')
    facts_df = pd.read_json(fact_file)
    facts2text: dict = dict(zip(facts_df['fact'], facts_df['note_text']))
    logger.info(f'Loaded {len(facts2text)} fact→note_text entries')

    # ------------------------------------------------------------------
    # 2. Collect unique facts referenced across all rows for this patient
    # ------------------------------------------------------------------
    fact_set = set(group['facts'].apply(ast.literal_eval).explode())
    fact_set = fact_set.union(set(group['reference_facts'].apply(ast.literal_eval).explode()))
    logger.info(f'Unique facts across all sub-questions (generated + reference): {len(fact_set)}')

    # ------------------------------------------------------------------
    # 3. For each fact that maps to a note, find the evidence spans
    # ------------------------------------------------------------------
    fact2note: dict = {}

    def _strip_note_prefix(note_text: str) -> str:
        prefix = 'Note Excerpt: '
        if prefix in note_text:
            return note_text.split(prefix, 1)[1]
        return note_text

    mappable = [f for f in fact_set if f in facts2text]
    logger.info(f'Facts with a matching note entry (will call API): {len(mappable)} / {len(fact_set)}')

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for fact in mappable:
            raw_text = _strip_note_prefix(facts2text[fact])
            futures.append(
                executor.submit(get_fact_text_module, fact, raw_text, logger)
            )
        for future in concurrent.futures.as_completed(futures):
            try:
                fact, text, spans = future.result()
                fact2note[fact] = (text, spans)
            except Exception:
                traceback.print_exc()

    logger.info(f'fact→note mapping complete | {len(fact2note)} / {len(mappable)} facts resolved')

    # ------------------------------------------------------------------
    # 4. For each sub-question row, map facts -> answer spans
    # ------------------------------------------------------------------
    facts2answer: dict = {}

    ref_rows = group.drop_duplicates(subset=['question_id'])
    logger.info(f'Starting fact→answer mapping | {len(group)} generated answer(s), '
                f'{len(ref_rows)} unique reference answer(s)')
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        # Submit reference answers (one per unique question_id)
        for _, row in ref_rows.iterrows():
            fact_list = ast.literal_eval(row['reference_facts'])
            futures.append(
                executor.submit(
                    get_answer_text_module,
                    fact_list,
                    row['reference_answer'],
                    row['question_id'] + '_r',
                    logger
                )
            )
        # Submit generated answers
        for _, row in group.iterrows():
            fact_list = ast.literal_eval(row['facts'])
            futures.append(
                executor.submit(
                    get_answer_text_module,
                    fact_list,
                    row['answer'],
                    row['subquestion_id'],
                    logger
                )
            )
        for future in concurrent.futures.as_completed(futures):
            try:
                mappings, subquestion_id = future.result()
                facts2answer[subquestion_id] = {
                    m['fact']: m['span'] for m in mappings
                }
            except Exception:
                traceback.print_exc()

    expected = len(group) + len(ref_rows)
    logger.info(f'fact→answer mapping complete | {len(facts2answer)} / {expected} answer(s) resolved '
                f'({len(ref_rows)} reference + {len(group)} generated)')

    # ------------------------------------------------------------------
    # 5. Build the final output
    # ------------------------------------------------------------------
    ret = []
    for subquestion_id, fact_span_map in facts2answer.items():
        temp = {
            'sub_question_id': subquestion_id,
            'sub_answer_spans': []
        }
        for fact, answer_span in fact_span_map.items():
            if fact in fact2note:
                note_text, note_spans = fact2note[fact]
                temp['sub_answer_spans'].append({
                    'fact': fact,
                    'answer_span': answer_span,
                    'note_text': note_text,
                    'note_spans': note_spans
                })
        ret.append(temp)

    total_spans = sum(len(r['sub_answer_spans']) for r in ret)
    logger.info(f'=== DONE patient {name!r} | {len(ret)} sub-question(s) | {total_spans} total span annotation(s) ===')
    return ret


def main(args):
    df = pd.read_csv(args.input)
    df['patient_id'] = df['question_id'].str.split('_').str[0]

    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_patient, args, patient_id, group): patient_id
            for patient_id, group in df.groupby('patient_id')
        }
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc='Processing patients'
        ):
            patient_id = futures[future]
            try:
                result = future.result()
                all_results.extend(result)
            except Exception:
                print(f'Error processing patient {patient_id}:')
                traceback.print_exc()

    with open(args.output, 'w+') as f:
        json.dump(all_results, f, indent=2)

    print(f'Done. Wrote {len(all_results)} sub-question records to {args.output}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
