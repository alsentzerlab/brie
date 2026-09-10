# default
import argparse
import json
import concurrent.futures
import logging

# pip
import tqdm
import pandas as pd

from .prompts.revise_answer import ANSWER_SYS, ANSWER_USER
from .utils import send_single_message


def revise(timestamp, question, answer, comment, facts):
    user_prompt = ANSWER_USER.format(
        TIMESTAMP=timestamp,
        QUESTION=question,
        ANSWER=answer,
        COMMENT=comment,
        FACTS=facts
    )
    return send_single_message(system_instructions=ANSWER_SYS, user_prompt=user_prompt)


def parse_result(result: str) -> dict:
    """
    Safely parse JSON from model output, stripping markdown code fences if present.
    """
    result = result.strip()
    if result.startswith("```"):
        lines = result.splitlines()
        result = "\n".join(lines[1:-1]).strip()
    return json.loads(result)


def chunk_fact_list(fact_list, chunk_size=30000):
    """
    Split fact list into chunks of specified size.
    """
    chunks = []
    for i in range(0, len(fact_list), chunk_size):
        chunks.append(fact_list[i:i + chunk_size])
    return chunks


def parse_args():
    """
    Parse the arguments.
    Example usage:
        python revise_answers.py --input PATH_TO_QA_CSV --fact FACT_DIR --output OUTPUT_DIR
    """
    parser = argparse.ArgumentParser(description="Answer Revision")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="CSV containing qa pairs, comments, question_id, and timestamp")
    parser.add_argument('-f', '--fact', type=str, required=True,
                        help="Fact directory (TSV files named by patient ID)")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Output directory")
    return parser.parse_args()


def make_run_row(args):
    """
    Factory that closes over `args` so run_row can be used safely in threads.
    """
    def run_row(index_row_tuple):
        """
        Process a single row: revise the answer using the patient's fact list.
        """
        # FIX: iterrows() yields (index, row) tuples
        _, row = index_row_tuple

        # FIX: define `id` before any logging that references it
        id = row['question_id'].split('_')[0]

        log_path = f"{args.output}/{id}.log"
        csv_path = f"{args.output}/{id}.csv"

        # Set up a per-row named logger to avoid cross-thread root logger pollution
        logger = logging.getLogger(id)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            fh = logging.FileHandler(log_path, mode='w')
            fh.setFormatter(logging.Formatter('[%(asctime)s]\t%(message)s'))
            logger.addHandler(fh)
        logging.getLogger("LiteLLM").setLevel(logging.CRITICAL + 1)
        logging.getLogger("httpx").setLevel(logging.WARNING + 1)

        logger.info(f'[{id}] Output: {csv_path}')
        logger.info(f'[{id}] Logfile: {log_path}')

        # FIX: use args.fact for the fact directory, not args.input
        fact_list = pd.read_csv(
            f'{args.fact}/{id}.tsv', delimiter='\t', index_col=0
        )['fact'].tolist()

        revised_answer = row['answer']

        if len(fact_list) > 30000:
            logger.info(f'[{id}] Fact list has {len(fact_list)} facts, chunking into 30K chunks')
            chunks = chunk_fact_list(fact_list, chunk_size=30000)

            for chunk_id, chunk in enumerate(chunks):
                # FIX: pass revised_answer (updated each iteration) not always the raw original
                logger.info(
                    f'[{id}] Revising question {row["question_id"]} — '
                    f'chunk {chunk_id + 1}/{len(chunks)} ({len(chunk)} facts)'
                )
                result = revise(
                    row['timestamp'],
                    row['question'],
                    revised_answer,
                    row['comment'],
                    chunk
                )
                parsed = parse_result(result)
                # If any chunk returns empty, propagate failure immediately
                if parsed.get('answer') == '':
                    revised_answer = ''
                    logger.info(f'[{id}] Chunk {chunk_id + 1} returned empty answer; stopping early')
                    break
                revised_answer = parsed['answer']

        else:
            # FIX: log uses len(fact_list), not undefined `chunk`
            logger.info(
                f'[{id}] Starting answer revision for question {row["question_id"]} ({len(fact_list)} facts)'
            )
            result = revise(
                row['timestamp'],
                row['question'],
                row['answer'],
                row['comment'],
                fact_list
            )
            parsed = parse_result(result)
            revised_answer = parsed.get('answer', '')

        # Clean up logger handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        return {
            'question_id': row['question_id'],
            'timestamp': row['timestamp'],
            'question': row['question'],
            'answer': row['answer'],
            'revised_answer': revised_answer
        }

    return run_row


def main(args):
    import os

    os.makedirs(args.output, exist_ok=True)
    ret = []
    df = pd.read_csv(args.input).iloc[:10]

    run_row = make_run_row(args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # FIX: pass (index, row) tuples directly from iterrows()
        futures = {executor.submit(run_row, item): item for item in df.iterrows()}
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(df)):
            try:
                ret.append(future.result())
            except Exception as e:
                print(f"Error revising answer: {e}")

    # FIX: use args.output (not args.o)
    pd.DataFrame(ret).to_csv(f"{args.output}/revised_answers.csv", index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)
