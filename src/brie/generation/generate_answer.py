# default
import argparse
import json
import random
import concurrent.futures
import logging
import os
from datetime import date
import re

# pip
import tqdm
import pandas as pd

from .prompts.generate_answer import clinical_seed_prompt, clinical_qa_prompt, clinical_qa_update_prompt, clinical_dedup_prompt, clinical_filter_prompt
from .utils import send_single_message

random.seed(42)

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def send_with_retry(user_prompt: str, logger=None, context: str = '') -> str:
    """
    Call send_single_message with simple retry logic for empty or failed responses.
    Raises RuntimeError if all retries are exhausted.
    """
    import time
    for attempt in range(1, MAX_RETRIES + 1):
        result = send_single_message(user_prompt=user_prompt)
        if result and result.strip():
            return result
        msg = f'{context} — attempt {attempt}/{MAX_RETRIES} returned empty response'
        if logger:
            logger.warning(msg)
        else:
            print(f'WARNING: {msg}')
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
    raise RuntimeError(f'{context} — all {MAX_RETRIES} attempts returned empty response')


PERSONAS = [
    'ED physician pre-disposition: specify exact presentation timing, triage vitals, prior ED visits for same complaint, and current mental/functional status',
    'Admitting resident writing H&P: specify full chronology of disease progression, prior workup results with values and dates, treatment responses, and current active problems',
    'Admitting attending validating plan: specify prior admissions for same diagnosis, highest-acuity events, key inflection points in disease course, and current clinical trajectory',
    'Pharmacist reconciling medications: specify drug name, formulation, dose, route, frequency, start/stop dates, indication, and any documented holds, reactions, or substitutions',
    'Bedside RN receiving patient: specify allergies with reaction type, fall and pressure injury history, isolation requirements with pathogen, code status with date, and prior nursing care needs',
    'Subspecialty consultant pre-consult: specify all findings, values, and events directly relevant to the organ system in question, with dates and clinical context',
    'Social worker at admission: specify living situation, caregiver support, prior placement, insurance coverage, prior social work involvement, and known barriers to discharge',
]

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def checkpoint_path(output_dir: str, question_id: str, stage: str) -> str:
    """Return the path for a checkpoint file."""
    return os.path.join(output_dir, f"{question_id}.{stage}.json")


def load_checkpoint(output_dir: str, question_id: str, stage: str):
    """Load a checkpoint if it exists, otherwise return None."""
    path = checkpoint_path(output_dir, question_id, stage)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def save_checkpoint(output_dir: str, question_id: str, stage: str, data) -> None:
    """Persist a checkpoint to disk."""
    path = checkpoint_path(output_dir, question_id, stage)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage1_seed(natural_query, reference_question, reference_answer, logger=None, context=''):
    """Generate 8-10 question seeds from the natural query and reference QA pair."""
    user_prompt = clinical_seed_prompt.format(
        natural_query=natural_query,
        reference_question=reference_question,
        reference_answer=reference_answer,
    )
    return send_with_retry(user_prompt, logger=logger, context=context)


def _filter_chunk(seed, chunk):
    """
    Run the filter prompt against a single chunk of facts.
    Returns the list of filtered facts from that chunk (may be empty).
    """
    user_prompt = clinical_filter_prompt.format(
        seed_query_focus=seed['query_focus'],
        seed_timeframe=seed['timeframe'],
        fact_list="\n".join(chunk),
    )
    result = send_with_retry(user_prompt, logger=None, context=f'filter seed={seed["seed_id"]}')
    parsed = parse_result(result, context=f'filter seed={seed["seed_id"]}')
    return parsed.get('filtered_facts', [])


def stage2_filter(seed, fact_list, logger, question_id):
    DATE_RE = re.compile(r'\((\d{4}-\d{2}-\d{2})\)\s*$')
    seed_id = seed['seed_id']

    if 'full history' in seed['timeframe'].lower():
        return fact_list

    # If fact list is large, chunk it and collect date-resolved results across all chunks,
    # then take the earliest start and latest end to form a single bounding window.
    CHUNK_SIZE = 30000
    if len(fact_list) > CHUNK_SIZE:
        chunks = chunk_fact_list(fact_list, chunk_size=CHUNK_SIZE)
        logger.info(
            f'[{question_id}] Seed {seed_id} — '
            f'fact list chunked into {len(chunks)} chunks for timeframe resolution'
        )
        start_dates, end_dates = [], []
        for chunk_id, chunk in enumerate(chunks):
            user_prompt = clinical_filter_prompt.format(
                seed_query_focus=seed['query_focus'],
                seed_timeframe=seed['timeframe'],
                seed_topic=seed['topic'],
                fact_list="\n".join(chunk),
            )
            result = send_with_retry(
                user_prompt, logger=logger,
                context=f'[{question_id}] filter seed={seed_id} chunk={chunk_id + 1}'
            )
            parsed = parse_result(result, context=f'[{question_id}] filter seed={seed_id} chunk={chunk_id + 1}')
            s, e = parsed.get('start_date'), parsed.get('end_date')
            logger.info(
                f'[{question_id}] Seed {seed_id} chunk {chunk_id + 1}/{len(chunks)} — '
                f'{parsed.get("resolved_timeframe", "n/a")} ({s} → {e})'
            )
            if s and e:
                try:
                    start_dates.append(date.fromisoformat(s))
                    end_dates.append(date.fromisoformat(e))
                except ValueError:
                    pass

        if not start_dates:
            logger.info(f'[{question_id}] Seed {seed_id} — no chunk resolved a date range, dropping seed')
            return None

        start_date = min(start_dates)
        end_date   = max(end_dates)
        logger.info(
            f'[{question_id}] Seed {seed_id} — '
            f'merged date range across {len(chunks)} chunks: {start_date} → {end_date}'
        )

    else:
        user_prompt = clinical_filter_prompt.format(
            seed_query_focus=seed['query_focus'],
            seed_timeframe=seed['timeframe'],
            seed_topic=seed['topic'],
            fact_list="\n".join(fact_list),
        )
        result = send_with_retry(
            user_prompt, logger=logger,
            context=f'[{question_id}] filter seed={seed_id}'
        )
        parsed = parse_result(result, context=f'[{question_id}] filter seed={seed_id}')

        start_str = parsed.get('start_date')
        end_str   = parsed.get('end_date')
        logger.info(
            f'[{question_id}] Seed {seed_id} — resolved timeframe: '
            f'{parsed.get("resolved_timeframe", "n/a")} ({start_str} → {end_str})'
        )

        if not start_str or not end_str:
            logger.info(f'[{question_id}] Seed {seed_id} — LLM could not resolve date range, dropping seed')
            return None

        try:
            start_date = date.fromisoformat(start_str)
            end_date   = date.fromisoformat(end_str)
        except ValueError as e:
            logger.error(f'[{question_id}] Seed {seed_id} — invalid date format ({e}), dropping seed')
            return None

    filtered = []
    for fact in fact_list:
        m = DATE_RE.search(fact)
        if m:
            try:
                fact_date = date.fromisoformat(m.group(1))
                if start_date <= fact_date and fact_date <= end_date:
                    filtered.append(fact)
            except ValueError:
                pass

    logger.info(
        f'[{question_id}] Seed {seed_id} — '
        f'{len(filtered)}/{len(fact_list)} facts retained ({start_date} → {end_date})'
    )
    return filtered if filtered else None


def stage3_qa(natural_query, seed, persona, fact_list):
    """
    Generate an initial QA pair for a single seed + persona against a fact list chunk.
    Returns parsed dict with question/answer/supporting_facts, or None if unfulfillable.
    """
    user_prompt = clinical_qa_prompt.format(
        natural_query=natural_query,
        seed_query_focus=seed['query_focus'],
        seed_timeframe=seed['timeframe'],
        seed_topic=seed['topic'],
        seed_detail_level=seed['detail_level'],
        persona=persona,
        fact_list="\n".join(fact_list),
    )
    result = send_with_retry(user_prompt, logger=None, context=f'qa seed={seed["seed_id"]}')
    parsed = parse_result(result, context=f'qa seed={seed["seed_id"]}')
    # Treat as unfulfillable if question is null or empty
    if not parsed.get('question'):
        return None
    return parsed


def stage3_qa_update(natural_query, seed, persona, current_qa, new_facts):
    """
    Update an existing QA pair with additional facts from a subsequent chunk.
    Returns an updated parsed dict, or the original current_qa if no new facts are relevant.
    """
    user_prompt = clinical_qa_update_prompt.format(
        natural_query=natural_query,
        seed_query_focus=seed['query_focus'],
        seed_timeframe=seed['timeframe'],
        seed_topic=seed['topic'],
        seed_detail_level=seed['detail_level'],
        persona=persona,
        current_question=current_qa['question'],
        current_answer=current_qa['answer'],
        current_supporting_facts=json.dumps(current_qa.get('supporting_facts', []), indent=2),
        new_facts="\n".join(new_facts),
    )
    result = send_with_retry(user_prompt, logger=None, context=f'qa-update seed={seed["seed_id"]}')
    parsed = parse_result(result, context=f'qa-update seed={seed["seed_id"]}')
    if not parsed.get('question'):
        return current_qa
    return parsed


def stage4_dedup(natural_query, reference_question, reference_answer, qa_pairs):
    """
    Remove redundant or incoherent QA pairs from the generated set.
    Returns list of retained QA dicts.
    """
    user_prompt = clinical_dedup_prompt.format(
        natural_query=natural_query,
        reference_question=reference_question,
        reference_answer=reference_answer,
        qa_pairs=json.dumps(qa_pairs, indent=2),
    )
    tries=0
    while tries < 5:
        tries += 1
        try:
            result = send_with_retry(user_prompt, logger=None, context='dedup')
            parsed = parse_result(result, context='dedup')
            return parsed.get('retained', [])
        except (ValueError, RuntimeError):
            continue
    return []


def parse_result(result: str, context: str = '') -> dict:
    """
    Safely parse JSON from model output, stripping markdown code fences if present.
    Raises ValueError with the raw response included if parsing fails, so callers
    can log which seed and stage produced the bad output.
    """
    raw = result
    result = result.strip()
    if result.startswith("```"):
        lines = result.splitlines()
        result = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(result)
    except json.JSONDecodeError as e:
        preview = repr(raw[:300]) if raw else '<empty response>'
        suffix = f' [{context}]' if context else ''
        raise ValueError(f"JSON parse failed{suffix} ({e}); raw response: {preview}") from e


def chunk_fact_list(fact_list, chunk_size=30000):
    """Split fact list into chunks of specified size."""
    chunks = []
    for i in range(0, len(fact_list), chunk_size):
        chunks.append(fact_list[i:i + chunk_size])
    return chunks


def run_stage3_for_seed(natural_query, seed, persona, fact_list, logger, question_id):
    """
    Run Stage 3 (QA generation) for a single seed, chunking the fact list if necessary.

    For long fact lists, generates an initial QA pair from the first fulfillable chunk,
    then updates it with each subsequent chunk so that evidence is accumulated across
    the entire fact list rather than stopping at the first hit.

    Returns the final QA dict, or None if no chunk yields an initial result.
    """
    seed_id = seed["seed_id"]

    if len(fact_list) > 30000:
        chunks = chunk_fact_list(fact_list, chunk_size=30000)
        logger.info(
            f'[{question_id}] Seed {seed_id} — fact list chunked into {len(chunks)} chunks'
        )
        current_qa = None
        for chunk_id, chunk in enumerate(chunks):
            logger.info(
                f'[{question_id}] Seed {seed_id} — '
                f'processing chunk {chunk_id + 1}/{len(chunks)} ({len(chunk)} facts)'
            )
            if current_qa is None:
                # No QA yet — try to generate the initial one from this chunk
                current_qa = stage3_qa(natural_query, seed, persona, chunk)
                if current_qa is not None:
                    logger.info(
                        f'[{question_id}] Seed {seed_id} — '
                        f'initial QA generated from chunk {chunk_id + 1}'
                    )
                else:
                    logger.info(
                        f'[{question_id}] Seed {seed_id} — '
                        f'chunk {chunk_id + 1} unfulfillable, continuing to next chunk'
                    )
            else:
                # QA exists — update it with any additional relevant facts from this chunk
                updated_qa = stage3_qa_update(natural_query, seed, persona, current_qa, chunk)
                added = len(updated_qa.get('supporting_facts', [])) - len(current_qa.get('supporting_facts', []))
                logger.info(
                    f'[{question_id}] Seed {seed_id} — '
                    f'chunk {chunk_id + 1} update: {added:+d} supporting facts'
                )
                current_qa = updated_qa

        if current_qa is None:
            logger.info(f'[{question_id}] Seed {seed_id} — unfulfillable across all chunks')
        return current_qa
    else:
        result = stage3_qa(natural_query, seed, persona, fact_list)
        if result is None:
            logger.info(f'[{question_id}] Seed {seed_id} — unfulfillable')
        return result


def process_single_seed(seed, natural_query, row, stage2_checkpoint, stage3_checkpoint,
                         checkpoint_lock, args, logger, id):
    """
    Process a single seed for Stage 3, skipping if already checkpointed.
    Thread-safe: uses checkpoint_lock when reading/writing stage3_checkpoint.
    Returns (seed_id, result) tuple.
    """
    seed_id = seed['seed_id']

    with checkpoint_lock:
        if seed_id in stage3_checkpoint:
            logger.info(f'[{id}] Seed {seed_id} — skipping (loaded from checkpoint)')
            return seed_id, stage3_checkpoint[seed_id]

    persona = random.choice(PERSONAS)
    filtered_facts = stage2_checkpoint[seed_id]
    logger.info(f'[{id}] Running Stage 3 for seed {seed_id} with persona "{persona}"')

    try:
        result = run_stage3_for_seed(
            natural_query=natural_query,
            seed=seed,
            persona=persona,
            fact_list=filtered_facts,
            logger=logger,
            question_id=id,
        )
    except (ValueError, RuntimeError) as e:
        logger.error(f'[{id}] Seed {seed_id} — Stage 3 failed, skipping seed: {e}')
        result = None

    if result is not None:
        result['seed_id'] = seed_id
        result['persona'] = persona

    with checkpoint_lock:
        stage3_checkpoint[seed_id] = result
        save_checkpoint(args.checkpoint, id, 'stage3', stage3_checkpoint)

    return seed_id, result


def parse_args():
    parser = argparse.ArgumentParser(description="QA Generation Pipeline")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="CSV containing qa pairs, comments, question_id, and timestamp")
    parser.add_argument('-f', '--fact', type=str, required=True,
                        help="Fact directory (TSV files named by patient ID)")
    parser.add_argument('-l', '--log', type=str, required=True,
                        help="Log directory")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Output directory")
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                        help="Checkpoint directory")
    return parser.parse_args()


def make_run_row(args):
    def run_row(index_row_tuple):
        import threading
        _, row = index_row_tuple
        id = row['question_id']
        patient_id = row['question_id'].split('_')[0]
        log_path = f"{args.log}/{id}.log"

        logger = logging.getLogger(id)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            fh = logging.FileHandler(log_path, mode='w')
            fh.setFormatter(logging.Formatter('[%(asctime)s]\t%(message)s'))
            logger.addHandler(fh)
        logging.getLogger("LiteLLM").setLevel(logging.CRITICAL + 1)
        logging.getLogger("httpx").setLevel(logging.WARNING + 1)

        logger.info(f'[{id}] Logfile: {log_path}')

        # Load fact list
        fact_list = pd.read_csv(
            f'{args.fact}/{patient_id}.tsv', delimiter='\t', index_col=0
        )['fact'].tolist()

        # --- Stage 1: Generate seeds (with checkpointing) ---
        seeds = load_checkpoint(args.checkpoint, id, 'stage1')
        if seeds is not None:
            logger.info(f'[{id}] Stage 1 loaded from checkpoint ({len(seeds)} seeds)')
        else:
            logger.info(f'[{id}] Running Stage 1 seed generation for question {row["question_id"]}')
            seeds_raw = stage1_seed(
                natural_query=row['question'],
                reference_question=row['original_question'],
                reference_answer=row['answer'],
                logger=logger,
                context=f'[{id}] stage1',
            )
            seeds = parse_result(seeds_raw, context=f'[{id}] stage1')
            save_checkpoint(args.checkpoint, id, 'stage1', seeds)
            logger.info(f'[{id}] Generated {len(seeds)} seeds — checkpoint saved')

        # --- Stage 2: Filter fact list per seed (with checkpointing) ---
        stage2_checkpoint = load_checkpoint(args.checkpoint, id, 'stage2') or {}
        stage2_checkpoint = {int(k): v for k, v in stage2_checkpoint.items()}
        if stage2_checkpoint:
            logger.info(
                f'[{id}] Stage 2 checkpoint loaded '
                f'({len(stage2_checkpoint)}/{len(seeds)} seeds already filtered)'
            )

        for seed in seeds:
            seed_id = seed['seed_id']
            if seed_id in stage2_checkpoint.keys():
                logger.info(f'[{id}] Seed {seed_id} — filtering skipping (loaded from checkpoint)')
                continue

            is_full_history = 'full history' in seed['timeframe'].lower()
            if is_full_history:
                logger.info(f'[{id}] Seed {seed_id} — timeframe is full history, skipping filter')
            else:
                logger.info(f'[{id}] Running Stage 2 fact filtering for seed {seed_id} (timeframe: "{seed["timeframe"]}")')

            try:
                filtered = stage2_filter(seed, fact_list, logger, id)
            except (ValueError, RuntimeError) as e:
                logger.error(f'[{id}] Seed {seed_id} — Stage 2 failed, dropping seed: {e}')
                filtered = None

            if filtered is None:
                logger.info(f'[{id}] Seed {seed_id} — filtered fact list is empty, seed will be dropped')
            else:
                logger.info(f'[{id}] Seed {seed_id} — {len(filtered)} facts retained after filtering')

            stage2_checkpoint[seed_id] = filtered
            save_checkpoint(args.checkpoint, id, 'stage2', stage2_checkpoint)

        # Drop seeds whose filtered fact list came back empty
        seeds_after_filter = [s for s in seeds if stage2_checkpoint.get(s['seed_id']) is not None]
        # seeds_after_filter = [s for s in seeds]
        logger.info(
            f'[{id}] {len(seeds_after_filter)}/{len(seeds)} seeds remain after Stage 2 filtering'
        )

        # --- Stage 3: Generate QA pairs for each seed (with checkpointing) ---
        # Seeds are processed in parallel using 3 inner workers per row.
        # A threading.Lock protects concurrent reads/writes to stage3_checkpoint.
        stage3_checkpoint = load_checkpoint(args.checkpoint, id, 'stage3') or {}
        stage3_checkpoint = {int(k): v for k, v in stage3_checkpoint.items()}
        if stage3_checkpoint:
            logger.info(
                f'[{id}] Stage 3 partial checkpoint loaded '
                f'({len(stage3_checkpoint)}/{len(seeds_after_filter)} seeds already done)'
            )

        checkpoint_lock = threading.Lock()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as seed_executor:
            seed_futures = {
                seed_executor.submit(
                    process_single_seed,
                    seed, row['question'], row, stage2_checkpoint, stage3_checkpoint,
                    checkpoint_lock, args, logger, id
                ): seed
                for seed in seeds_after_filter
            }
            for future in concurrent.futures.as_completed(seed_futures):
                seed = seed_futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(
                        f'[{id}] Seed {seed["seed_id"]} — unhandled Stage 3 exception: {e}'
                    )

        qa_pairs = [v for v in stage3_checkpoint.values() if v is not None]
        logger.info(f'[{id}] {len(qa_pairs)}/{len(seeds_after_filter)} seeds fulfilled after Stage 3')

        # --- Stage 4: Deduplication and coherence filtering ---
        logger.info(f'[{id}] Running Stage 4 deduplication for question {row["question_id"]}')
        qa_lookup = {qa['question']: qa for qa in qa_pairs}

        try:
            retained = stage4_dedup(
                natural_query=row['question'],
                reference_question=row['original_question'],
                reference_answer=row['answer'],
                qa_pairs=qa_pairs,
            )
        except (ValueError, RuntimeError) as e:
            logger.error(f'[{id}] Stage 4 failed: {e} — returning empty result for this row')
            retained = []

        logger.info(f'[{id}] {len(retained)} QA pairs retained after Stage 4')

        # --- Build output records ---
        seed_lookup = {s['seed_id']: s for s in seeds}

        records = []
        for i, qa in enumerate(retained):
            original = qa_lookup.get(qa['question'], {})
            qa.setdefault('persona', original.get('persona', ''))
            qa.setdefault('seed_id', original.get('seed_id', ''))
            seed = seed_lookup.get(qa['seed_id'], {})
            records.append({
                'question_id': row['question_id'],
                'subquestion_id': f"{row['question_id']}_{i}",
                'question': qa['question'],
                'answer': qa['answer'],
                'facts': json.dumps(qa.get('supporting_facts', [])),
                'persona': qa.get('persona', ''),
                'seed_id': qa.get('seed_id', ''),
                'seed_query_focus': seed.get('query_focus', ''),
                'seed_timeframe': seed.get('timeframe', ''),
                'seed_topic': seed.get('topic', ''),
                'seed_detail_level': seed.get('detail_level', ''),
                'natural_query': row['question'],
                'reference_question': row['original_question'],
                'reference_answer': row['answer'],
                'reference_facts': row['facts']
            })

        # Clean up logger handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        return pd.DataFrame(records)

    return run_row


def main(args):
    ret = []
    os.makedirs(args.log, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.checkpoint, exist_ok=True)
    df = pd.read_csv(args.input)

    run_row = make_run_row(args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(run_row, item): item for item in df.iterrows()}
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(df)):
            try:
                result = future.result()
                if result is not None and not result.empty:
                    ret.append(result)
            except Exception as e:
                row_item = futures[future]
                _, row_data = row_item
                print(f"Error processing row {row_data.get('question_id', '?')}: {e}")

    if not ret:
        raise RuntimeError("No answer-generation rows were produced; inspect the external logs")
    pd.concat(ret).to_csv(f"{args.output}/answers.csv", index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)
