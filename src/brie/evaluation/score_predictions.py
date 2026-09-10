'''
Score prediction script
This script scores LLM responses using:
  1. ROUGE score
  2. BERTScore
  3. BLEU score (BLEU-1 through BLEU-4, with smoothing)
  4. Fact-based precision & recall using a single model (via LLM fact extraction)
  5. Fact-based precision & recall using a jury of models, with per-model entailed fact lists
  6. LLM Jury evaluation (completeness, relevancy, faithfulness, clarity)
     — evaluated independently by Claude, GPT, and Gemini;
       per-juror scores/explanations + macro-averaged scores are reported.

Input CSVs:
  Questions: question_id, sub_question_id, annotation_sub_answer, facts_edited
  Responses: question_id, response, model

Output CSV:
  sub_question_id, model, candidate_facts,
  bert_score_precision, bert_score_recall, bert_score_f1,
  rouge1, rouge2, rougeL,
  bleu1, bleu2, bleu3, bleu4,
  fact_precision, fact_recall,
  fact_jury_claude_precision,      fact_jury_claude_recall,
  fact_jury_claude_entailed_ref_facts, fact_jury_claude_entailed_cand_facts,
  fact_jury_gpt_precision,         fact_jury_gpt_recall,
  fact_jury_gpt_entailed_ref_facts,    fact_jury_gpt_entailed_cand_facts,
  fact_jury_gemini_precision,      fact_jury_gemini_recall,
  fact_jury_gemini_entailed_ref_facts, fact_jury_gemini_entailed_cand_facts,
  fact_jury_avg_precision,         fact_jury_avg_recall,
  jury_claude_completeness_score,  jury_claude_completeness_explanation,
  jury_claude_relevancy_score,     jury_claude_relevancy_explanation,
  jury_claude_faithfulness_score,  jury_claude_faithfulness_explanation,
  jury_gpt_completeness_score,     jury_gpt_completeness_explanation,
  jury_gpt_relevancy_score,        jury_gpt_relevancy_explanation,
  jury_gpt_faithfulness_score,     jury_gpt_faithfulness_explanation,
  jury_gemini_completeness_score,  jury_gemini_completeness_explanation,
  jury_gemini_relevancy_score,     jury_gemini_relevancy_explanation,
  jury_gemini_faithfulness_score,  jury_gemini_faithfulness_explanation,
  jury_avg_completeness_score, jury_avg_relevancy_score, jury_avg_faithfulness_score
'''
import asyncio
import argparse
import json
import logging
import os
import sys
import ast
import re

import pandas as pd

from .utils import safe_json_parse, send_single_message, send_batch_messages, CsvWriter, load_completed_pairs, log_token_stats

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a clinician performing chart review. "
    "Respond only with a valid JSON object — no markdown, no explanation outside the JSON."
)

EXTRACT_PROMPT = """\
Extract every atomic clinical claim from the answer below.

## Atomic Claim Definition
An atomic claim makes a single assertion with a subject, predicate, and object. It must stand \
alone without ambiguity and cannot be decomposed into more fundamental claims.

## DO
1. Extract discrete atomic claims. Each must include a subject, predicate, and object.
2. Include only clinically relevant claims (symptoms, procedures, tests, medications, diagnoses, \
clinical locations).
3. Use only the provided text. Do not add outside knowledge or assumptions.
4. Write each claim in the shortest unambiguous form. Avoid pronouns or vague references.
5. Always refer to the subject as "patient", even if the text uses a name or identifier.
6. Append a date (YYYY-MM-DD) to every claim:
   a. If the text specifies an absolute date, use that date.
   b. If the text uses a relative reference (e.g., "last week"), resolve it to the best \
specific date inferable from context.
   c. If no date can be determined, append (0000-00-00).
7. If there are no valid clinically relevant claims, return "claims" as an empty list [].

## DO NOT
1. Do not include claims unrelated to the patient's clinical care (e.g., provider names, \
administrative details).
2. Do not invent or infer claims beyond what is explicitly stated.
3. Do not combine multiple events into a single claim.

## Output Format
{{
    "claims": ["Claim one (YYYY-MM-DD)", "Claim two (YYYY-MM-DD)", ...]
}}

### Answer
{ANSWER}\
"""

ENTAILMENT_PROMPT = """\
Given a reference fact and a list of candidate facts, determine whether the reference fact \
is semantically entailed by any candidate fact.
Respond with a JSON array containing only the 0-based indices of candidate facts that entail \
the reference fact. Return an empty array if none do: []

### Reference fact
{REFERENCE_FACT}

### Candidate facts
{CANDIDATE_FACTS}\
"""

JURY_SYSTEM_PROMPT = (
    "You are a medical expert evaluating the quality of clinical information retrieval responses. "
    "Respond only with a valid JSON object — no markdown, no explanation outside the JSON."
)

JURY_PROMPT = """\
You are a medical expert tasked with evaluating the quality of a response to a clinical \
information retrieval query. Your goal is to assess how well the response retrieves and \
presents relevant clinical information compared to a reference answer (gold standard).

Question:
<question>{QUESTION}</question>

Generated Response:
<response>{RESPONSE}</response>

Reference Response:
<gold_response>{GOLD_RESPONSE}</gold_response>

Carefully review the <response> based on the <question> and compare it to the <gold_response>. \
For each of the following criteria, rate the response on a scale of 0 to 4 and provide a short \
justification for your score.

Scoring Scale (applies to all axes):
- 4: Maximum possible marks; fulfills all aspects of this axis.
- 3: Missing details, but not enough to impact overall interpretation.
- 2: Missing details that impact interpretation but do not impact care.
- 1: Missing details that impact interpretation and could negatively impact patient care.
- 0: Complete refusal (e.g., the response states that the notes are not in context).

Guidance on Refusals:
If the response states that the notes are not in context (i.e., a refusal to answer), assign a \
score of 0 across all axes. Apply partial credit where applicable if the response partially \
addresses the question before refusing.

Evaluation Criteria:
Completeness (0-4) - Does the response include all facts present in the reference answer, \
without omitting important clinical details?

Relevancy (0-4) - Does the response contain only information that is present in the reference \
answer, without introducing unnecessary or tangential details? Lower marks on relevancy reflect \
added details that are *implied by or consistent with* the reference answer but provide more \
detail than necessary.

Faithfulness (0-4) - Does the response accurately reflect the reference answer without \
contradicting or distorting any clinical facts (i.e., no hallucination)? Lower marks on \
faithfulness reflect added details that *contradict* the reference answer.

Output Format:
Output the evaluation as a single valid JSON object matching the following structure:
{{
    "completeness":
        {{"score":0,
        "explanation":"Explain why this score was given."}},
    "relevancy":
        {{"score":0,
        "explanation":"Explain why this score was given."}},
    "faithfulness":
        {{"score":0,
        "explanation":"Explain why this score was given."}}
}}

Ensure the output is valid JSON:
- Use **double quotes** (") for all keys and string values.
- When quoting text or sections inside the explanations, use escaped double quotes (\\") to maintain valid JSON formatting.
- Do not include any additional information in the output.\
"""

RUBRIC_JURY_PROMPT_SYS = """\
You are a medical expert tasked with evaluating the quality of a response to a clinical \
information retrieval query. Your goal is to assess how well the response retrieves and \
presents relevant clinical information compared to a reference answer (gold standard).

## Rubric Schema
Each criterion in the rubric has:
- `id`: unique identifier (e.g., "c1")
- `description`: what the criterion evaluates

## Evaluation Instructions

For EACH criterion in the rubric, perform this reasoning:
1. **Read the criterion carefully.** Understand its `description`
2. **Locate relevant spans** in the prediction that relate to this criterion.
3. **Decide:** Does the prediction satisfy this criterion? When in doubt, do not mark it as applying.
4. **Write a concise rationale** (1 sentence) citing specific evidence from the prediction. Reference the supporting facts or explain the specific gap/error.

## Output Format
Return ONLY a JSON object with this exact shape — no preamble, no markdown fences, no extra keys:

 [
    {"id": "<criterion_id>", "rationale": "<your concise reasoning>"},
    ...
  ]

**Important rules:**
- Include ONLY criteria that the prediction satisfies (i.e., that "apply"). Omit criteria the prediction fails.
- Preserve the original criterion `id` values exactly.
- The `rationale` must be grounded in the prediction's actual content — quote or paraphrase specific parts.
- Do NOT include scores, points, booleans, or any fields other than `id` and `rationale`.
- Output must be valid JSON parseable by a standard JSON parser.

"""
RUBRIC_JURY_PROMPT = """\
Question:
<question>{QUESTION}</question>

Generated Response:
<response>{RESPONSE}</response>

Reference Response:
<gold_response>{GOLD_RESPONSE}</gold_response>

Rubric:
<rubric>{RUBRIC}</rubric>\

"""

# ── Rate limiter ──────────────────────────────────────────────────────────────
class RateLimiter:
    """Sliding-window: allows at most `rate` calls per `period` seconds."""
    def __init__(self, rate: int, period: float = 60.0):
        self._semaphore = asyncio.Semaphore(rate)
        self._period = period

    async def acquire(self):
        await self._semaphore.acquire()
        asyncio.get_event_loop().call_later(self._period, self._semaphore.release)


# ── Constants ─────────────────────────────────────────────────────────────────
FACT_BERT_THRESHOLD = 0.90
_FACT_DATE_RE = re.compile(r'\((\d{4}-\d{2}-\d{2})\)\s*$')
FACT_EXTRACTION_MODEL = "gemini_flash"

BERT_MODEL = os.environ.get("BRIE_BERTSCORE_MODEL", "thomas-sounack/BioClinical-ModernBERT-base")
BERT_NUM_LAYERS = int(os.environ.get("BRIE_BERTSCORE_LAYERS", "22"))


def _bert_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _parse_fact(fact: str) -> tuple[str, str]:
    """Return (text, date) from a fact string 'Text (YYYY-MM-DD)'.
    If no date is found, date is returned as an empty string."""
    m = _FACT_DATE_RE.search(fact)
    if m:
        return fact[:m.start()].strip(), m.group(1)
    return fact, ''


# ── Scoring helpers ───────────────────────────────────────────────────────────
def compute_bert_score(candidates: list[str], references: list[str]) -> tuple:
    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(
        candidates, references,
        model_type=BERT_MODEL,
        num_layers=BERT_NUM_LAYERS, lang="en",
        batch_size=64,
        device=_bert_device(),
        verbose=False,
    )
    return P.tolist(), R.tolist(), F1.tolist()


def compute_rouge_score(candidate: str, reference: str) -> dict:
    from rouge_score import rouge_scorer as rouge_scorer_lib

    scorer = rouge_scorer_lib.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return {k: v.fmeasure for k, v in scorer.score(reference, candidate).items()}


def compute_bleu_score(candidate: str, reference: str) -> dict:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    smoothie = SmoothingFunction().method1
    ref_tokens = reference.split()
    hyp_tokens = candidate.split()
    return {
        'bleu1': sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0),       smoothing_function=smoothie),
        'bleu2': sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0),   smoothing_function=smoothie),
        'bleu3': sentence_bleu([ref_tokens], hyp_tokens, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothie),
        'bleu4': sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25,)*4,           smoothing_function=smoothie),
    }


async def extract_facts(text: str, model_id: str, limiter: RateLimiter) -> list[str]:
    await limiter.acquire()
    raw = await send_single_message(
        user_prompt=EXTRACT_PROMPT.format(ANSWER=text),
        system_instructions=SYSTEM_PROMPT,
        model_id=model_id,
    )
    return safe_json_parse(raw)['claims']


async def check_entailment(reference_fact: str, candidate_facts: list[str], model_id: str, limiter: RateLimiter, backend: str = "vertex") -> list[int]:
    if not candidate_facts:
        return []
    numbered = "\n".join(f"{i}. {f}" for i, f in enumerate(candidate_facts))
    await limiter.acquire()
    raw = await send_single_message(
        user_prompt=ENTAILMENT_PROMPT.format(
            REFERENCE_FACT=reference_fact,
            CANDIDATE_FACTS=numbered,
        ),
        system_instructions=SYSTEM_PROMPT,
        model_id=model_id,
        backend=backend,
    )
    return safe_json_parse(raw)


async def compute_fact_score(
    candidate_facts: list[str],
    reference_facts: list[str],
    model_id: str,
    limiter: RateLimiter,
) -> tuple[float, float]:
    if not candidate_facts or not reference_facts:
        return 0.0, 0.0

    recall_results, prec_results = await asyncio.gather(
        asyncio.gather(*[check_entailment(rf, candidate_facts, model_id, limiter) for rf in reference_facts]),
        asyncio.gather(*[check_entailment(cf, reference_facts, model_id, limiter) for cf in candidate_facts]),
    )
    recall    = sum(1 for m in recall_results if m) / len(reference_facts)
    precision = sum(1 for m in prec_results  if m) / len(candidate_facts)
    return precision, recall


async def compute_fact_score_jury(
    candidate_facts: list[str],
    reference_facts: list[str],
    limiter: RateLimiter,
) -> dict:
    """Run fact entailment scoring across all three jury models.

    For each model, records which reference facts are recalled (entailed by any
    candidate fact) and which candidate facts have precision (entailed by any
    reference fact), along with per-model precision/recall and macro averages.
    """
    if not candidate_facts or not reference_facts:
        combined = {}
        for juror in ["claude", "gpt", "gemini"]:
            combined.update({
                f'fact_jury_{juror}_precision':            0.0,
                f'fact_jury_{juror}_recall':               0.0,
                f'fact_jury_{juror}_entailed_ref_facts':   json.dumps([]),
                f'fact_jury_{juror}_entailed_cand_facts':  json.dumps([]),
            })
        combined['fact_jury_avg_precision'] = 0.0
        combined['fact_jury_avg_recall']    = 0.0
        return combined

    combined = {}
    precisions, recalls = [], []

    for juror in ["claude", "gpt", "gemini"]:
        recall_results, prec_results = await asyncio.gather(
            asyncio.gather(*[check_entailment(rf, candidate_facts, _JURY_MODEL_IDS[juror], limiter, _JURY_BACKENDS[juror]) for rf in reference_facts]),
            asyncio.gather(*[check_entailment(cf, reference_facts, _JURY_MODEL_IDS[juror], limiter, _JURY_BACKENDS[juror]) for cf in candidate_facts]),
        )
        entailed_ref  = [reference_facts[i] for i, m in enumerate(recall_results) if m]
        entailed_cand = [candidate_facts[i]  for i, m in enumerate(prec_results)  if m]
        precision = len(entailed_cand) / len(candidate_facts)
        recall    = len(entailed_ref)  / len(reference_facts)

        precisions.append(precision)
        recalls.append(recall)
        combined.update({
            f'fact_jury_{juror}_precision':           round(precision, 4),
            f'fact_jury_{juror}_recall':              round(recall, 4),
            f'fact_jury_{juror}_entailed_ref_facts':  json.dumps(entailed_ref),
            f'fact_jury_{juror}_entailed_cand_facts': json.dumps(entailed_cand),
        })

    combined['fact_jury_avg_precision'] = round(sum(precisions) / len(precisions), 4)
    combined['fact_jury_avg_recall']    = round(sum(recalls)    / len(recalls),    4)
    return combined


def _fact_bert_pairs(
    candidate_facts: list[str],
    reference_facts: list[str],
) -> tuple[list[tuple[int, int]], list[str], list[str]]:
    """Return same-date (cand_idx, ref_idx) pairs and their text strings for BERTScore.

    No model calls — pure pre-filtering used to build the deferred scoring bucket.
    """
    cand_parsed = [_parse_fact(f) for f in candidate_facts]
    ref_parsed  = [_parse_fact(f) for f in reference_facts]

    ref_by_date: dict[str, list[tuple[int, str]]] = {}
    for ri, (text, date) in enumerate(ref_parsed):
        ref_by_date.setdefault(date, []).append((ri, text))

    pairs:     list[tuple[int, int]] = []
    hyp_texts: list[str]             = []
    ref_texts: list[str]             = []
    for ci, (cand_text, cand_date) in enumerate(cand_parsed):
        for ri, ref_text in ref_by_date.get(cand_date, []):
            pairs.append((ci, ri))
            hyp_texts.append(cand_text)
            ref_texts.append(ref_text)

    return pairs, hyp_texts, ref_texts


JURY_MODELS = ["claude", "gpt", "gemini"]

# Maps jury shorthand names to valid utils.send_single_message model IDs
_JURY_MODEL_IDS: dict[str, str] = {
    "claude": "claude_opus",
    "gpt":    "gpt5",
    "gemini": "gemini_pro",
}

# Configure each jury backend independently.
_JURY_BACKENDS: dict[str, str] = {
    "claude": "vertex",
    "gpt":    "openai",
    "gemini": "vertex",
}


async def _call_single_juror(
    question: str,
    candidate: str,
    reference: str,
    juror: str,
    limiter: RateLimiter,
    max_retries: int = 3,
    prefetched: str | None = None,
) -> dict:
    """Call one juror model and return its parsed scores/explanations (prefixed jury_<juror>_).
    If prefetched is provided (Claude batch result), parse it directly instead of calling the API.
    Retries up to max_retries times on parse or network failure.
    Returns None values per key if all attempts fail."""
    null_result = {
        f'jury_{juror}_completeness_score':       None,
        f'jury_{juror}_completeness_explanation': None,
        f'jury_{juror}_relevancy_score':          None,
        f'jury_{juror}_relevancy_explanation':    None,
        f'jury_{juror}_faithfulness_score':       None,
        f'jury_{juror}_faithfulness_explanation': None,
    }

    def _parse(raw: str) -> dict:
        parsed = safe_json_parse(raw)
        return {
            f'jury_{juror}_completeness_score':       parsed['completeness']['score'],
            f'jury_{juror}_completeness_explanation': parsed['completeness']['explanation'],
            f'jury_{juror}_relevancy_score':          parsed['relevancy']['score'],
            f'jury_{juror}_relevancy_explanation':    parsed['relevancy']['explanation'],
            f'jury_{juror}_faithfulness_score':       parsed['faithfulness']['score'],
            f'jury_{juror}_faithfulness_explanation': parsed['faithfulness']['explanation'],
        }

    if juror == "claude" and prefetched is not None:
        try:
            return _parse(prefetched)
        except Exception as e:
            log.warning(f"Failed to parse prefetched Claude jury result: {e!r} — falling back to API call")

    last_error = None
    raw = None
    for attempt in range(1, max_retries + 1):
        await limiter.acquire()
        try:
            raw = await send_single_message(
                user_prompt=JURY_PROMPT.format(
                    QUESTION=question,
                    RESPONSE=candidate,
                    GOLD_RESPONSE=reference,
                ),
                system_instructions=JURY_SYSTEM_PROMPT,
                model_id=_JURY_MODEL_IDS[juror],
                backend=_JURY_BACKENDS[juror],
            )
            return _parse(raw)
        except Exception as e:
            last_error = e
            log.warning(
                f"Juror {juror} attempt {attempt}/{max_retries} failed: {e}\n"
                f"Raw response: {raw!r}"
            )

    log.error(f"Juror {juror} failed all {max_retries} attempts. Last error: {last_error}")
    return null_result


async def compute_jury_score(
    question: str,
    candidate: str,
    reference: str,
    limiter: RateLimiter,
    claude_prefetch: str | None = None,
) -> dict:
    """Run all three jurors (Claude, GPT, Gemini) concurrently and return their
    individual scores/explanations plus macro-averaged scores per dimension.
    claude_prefetch, if provided, is the raw Claude response from a pre-submitted batch."""
    results = await asyncio.gather(*[
        _call_single_juror(
            question, candidate, reference, juror, limiter,
            prefetched=claude_prefetch if juror == "claude" else None,
        )
        for juror in JURY_MODELS
    ])

    combined = {}
    for r in results:
        combined.update(r)

    # Macro-average over jurors that succeeded (non-None scores)
    for dim in ('completeness', 'relevancy', 'faithfulness'):
        scores = [
            combined[f'jury_{j}_{dim}_score']
            for j in JURY_MODELS
            if combined.get(f'jury_{j}_{dim}_score') is not None
        ]
        combined[f'jury_avg_{dim}_score'] = round(sum(scores) / len(scores), 4) if scores else None

    return combined


def _compute_rubric_total(rubric_str: str, grade_results: list[dict]) -> tuple[int, int, int]:
    """Return (total_score, num_criteria, max_points).

    grade_results is the list of criteria the juror found to apply; presence means the
    criterion applies. Positive-point criteria reward the response; negative-point
    criteria penalise it. max_points sums only positive-point values.
    """
    try:
        criteria = json.loads(rubric_str).get("criteria", [])
    except (json.JSONDecodeError, AttributeError):
        criteria = []
    points_by_id = {c["id"]: c["points"] for c in criteria}
    num_criteria = len(criteria)
    max_points = sum(p for p in points_by_id.values() if p > 0)
    total = sum(points_by_id.get(entry["id"], 0) for entry in grade_results)
    return total, num_criteria, max_points


async def _call_rubric_juror(
    question: str,
    candidate: str,
    reference: str,
    rubric: str,
    juror: str,
    limiter: RateLimiter,
    max_retries: int = 3,
    prefetched: str | None = None,
) -> dict:
    """Call one juror with a question-specific rubric.

    Returns the numeric score (sum of applied criteria points) and the raw LLM
    response as the explanation. If prefetched is provided for Claude, parses it directly.
    """
    def _parse(raw: str) -> dict:
        grade_results = safe_json_parse(raw)
        total, _, _ = _compute_rubric_total(rubric, grade_results)
        return {
            f'jury_rubric_{juror}_score':       total,
            f'jury_rubric_{juror}_explanation': grade_results,
        }

    if juror == "claude" and prefetched is not None:
        try:
            return _parse(prefetched)
        except Exception as e:
            log.warning(f"Failed to parse prefetched Claude rubric jury result: {e!r} — falling back to API call")

    last_error = None
    raw = None
    for attempt in range(1, max_retries + 1):
        await limiter.acquire()
        try:
            raw = await send_single_message(
                user_prompt=RUBRIC_JURY_PROMPT.format(
                    QUESTION=question,
                    RESPONSE=candidate,
                    GOLD_RESPONSE=reference,
                    RUBRIC=rubric,
                ),
                system_instructions=RUBRIC_JURY_PROMPT_SYS,
                model_id=_JURY_MODEL_IDS[juror],
                backend=_JURY_BACKENDS[juror],
            )
            return _parse(raw)
        except Exception as e:
            last_error = e
            log.warning(
                f"Rubric juror {juror} attempt {attempt}/{max_retries} failed: {e}\n"
                f"Raw response: {raw!r}"
            )

    log.error(f"Rubric juror {juror} failed all {max_retries} attempts. Last error: {last_error}")
    return {
        f'jury_rubric_{juror}_score':       None,
        f'jury_rubric_{juror}_explanation': None,
    }


async def compute_rubric_jury_score(
    question: str,
    candidate: str,
    reference: str,
    rubric: str,
    limiter: RateLimiter,
    claude_prefetch: str | None = None,
) -> dict:
    """Run all three jurors with a question-specific rubric concurrently.
    claude_prefetch, if provided, is the raw Claude response from a pre-submitted batch."""
    results = await asyncio.gather(*[
        _call_rubric_juror(
            question, candidate, reference, rubric, juror, limiter,
            prefetched=claude_prefetch if juror == "claude" else None,
        )
        for juror in JURY_MODELS
    ])
    combined = {}
    for r in results:
        combined.update(r)
    scores = [
        combined[f'jury_rubric_{j}_score']
        for j in JURY_MODELS
        if combined.get(f'jury_rubric_{j}_score') is not None
    ]
    combined['jury_rubric_avg_score'] = round(sum(scores) / len(scores), 4) if scores else None

    # num_criteria and max_points are rubric properties, same for all jurors
    _, num_criteria, max_points = _compute_rubric_total(rubric, [])
    combined['jury_rubric_num_criteria'] = num_criteria
    combined['jury_rubric_max_points']   = max_points
    return combined


# ── Workers ───────────────────────────────────────────────────────────────────
async def process_response(resp: pd.Series, model_id: str, limiter: RateLimiter) -> list[str] | None:
    """Extract candidate facts once per response, shared across all its sub-question pairs."""
    try:
        return await extract_facts(resp['response'], model_id, limiter)
    except Exception as e:
        log.error(f"Fact extraction failed for question_id={resp['question_id']} model={resp['model']}: {e}")
        return None


async def process_pair(
    sub_question_id: str,
    response_row: pd.Series,
    question_row: pd.Series,
    candidate_facts: list[str],
    model_id: str,
    bert_bucket: list,
    fact_bert_bucket: list,
    limiter: RateLimiter,
    metrics: set[str],
    rubrics_lookup: dict[str, str] | None = None,
    claude_jury_cache: dict[str, str | None] | None = None,
) -> dict | None:
    model     = response_row['model']
    candidate = response_row['response']
    reference = question_row.get('annotation_sub_answer') or question_row.get('answer', '')
    question  = question_row.get('question', '')          # used by jury; empty string if column absent
    reference_facts = (
        ast.literal_eval(question_row['facts_edited'])
        if isinstance(question_row['facts_edited'], str)
        else question_row['facts_edited']
    ) if ({'fact', 'fact_jury', 'fact_bert'} & metrics) else []

    log.info(f"Scoring sub_question_id={sub_question_id} model={model} metrics={metrics}")
    try:
        entry = {
            'sub_question_id': sub_question_id,
            'model':           model,
            'batch_num':       response_row.get('batch_num', ''),
            'candidate_facts': json.dumps(candidate_facts),
        }

        # ── ROUGE ────────────────────────────────────────────────────────────
        if 'rouge' in metrics:
            rouge = compute_rouge_score(candidate, reference)
            entry.update({
                'rouge1': rouge['rouge1'],
                'rouge2': rouge['rouge2'],
                'rougeL': rouge['rougeL'],
            })

        # ── BLEU ─────────────────────────────────────────────────────────────
        if 'bleu' in metrics:
            bleu = compute_bleu_score(candidate, reference)
            entry.update(bleu)

        # ── Fact score ───────────────────────────────────────────────────────
        if 'fact' in metrics:
            precision, recall = await compute_fact_score(candidate_facts, reference_facts, model_id, limiter)
            entry.update({
                'fact_precision': precision,
                'fact_recall':    recall,
            })

        # ── Fact BERTScore: defer to batch step; register in bucket ─────────
        if 'fact_bert' in metrics:
            if candidate_facts and reference_facts:
                fb_pairs, fb_hyps, fb_refs = _fact_bert_pairs(candidate_facts, reference_facts)
            else:
                fb_pairs, fb_hyps, fb_refs = [], [], []
            fact_bert_bucket.append((entry, candidate_facts, reference_facts, fb_pairs, fb_hyps, fb_refs))

        # ── Fact jury ────────────────────────────────────────────────────────
        if 'fact_jury' in metrics:
            fact_jury = await compute_fact_score_jury(candidate_facts, reference_facts, limiter)
            entry.update(fact_jury)

        # ── Jury ─────────────────────────────────────────────────────────────
        if 'jury' in metrics:
            cache_key = f"jury||{sub_question_id}||{response_row['model']}||{response_row.get('batch_num', '')}"
            claude_prefetch = (claude_jury_cache or {}).get(cache_key)
            jury = await compute_jury_score(question, candidate, reference, limiter, claude_prefetch)
            entry.update(jury)

        # ── Rubric jury ──────────────────────────────────────────────────────
        if 'jury_rubric' in metrics:
            rubric = (rubrics_lookup or {}).get(str(sub_question_id))
            if rubric:
                cache_key = f"jury_rubric||{sub_question_id}||{response_row['model']}||{response_row.get('batch_num', '')}"
                claude_prefetch = (claude_jury_cache or {}).get(cache_key)
                rubric_jury = await compute_rubric_jury_score(question, candidate, reference, rubric, limiter, claude_prefetch)
                entry.update(rubric_jury)
            else:
                log.warning(f"No rubric found for sub_question_id={sub_question_id} — skipping jury_rubric")

        # ── BERT: defer to batch step; register in bucket ────────────────────
        if 'bert' in metrics:
            bert_bucket.append((entry, candidate, reference))
        return entry

    except Exception as e:
        log.error(f"Failed sub_question_id={sub_question_id} model={model}: {e}")
        return None


# ── Join helper ───────────────────────────────────────────────────────────────
def build_pairs(responses: pd.DataFrame, questions: pd.DataFrame) -> list[tuple[str, pd.Series, pd.Series]]:
    """
    Join each response against all sub-questions sharing its question_id.
    One response row → N output pairs, one per matching sub_question_id in questions.
    Returns a list of (sub_question_id, response_row, question_row) tuples.
    """
    q_grouped = questions.groupby('question_id')
    pairs = []
    for _, resp in responses.iterrows():
        qid = resp['question_id']
        if len(qid.split('_')) > 1:
            qid = '_'.join(qid.split('_')[:2])
        if qid not in q_grouped.groups:
            log.warning(f"No sub-questions found for question_id={qid} — skipping.")
            continue
        for _, q_row in q_grouped.get_group(qid).iterrows():
            pairs.append((q_row['sub_question_id'], resp, q_row))

    log.info(f"Built {len(pairs)} pairs from {len(responses)} responses")
    return pairs


# ── Column ordering ───────────────────────────────────────────────────────────
def output_columns(metrics: set[str]) -> list[str]:
    cols = ['sub_question_id', 'model', 'batch_num', 'candidate_facts']
    if 'bert' in metrics:
        cols += ['bert_score_precision', 'bert_score_recall', 'bert_score_f1']
    if 'rouge' in metrics:
        cols += ['rouge1', 'rouge2', 'rougeL']
    if 'bleu' in metrics:
        cols += ['bleu1', 'bleu2', 'bleu3', 'bleu4']
    if 'fact' in metrics:
        cols += ['fact_precision', 'fact_recall']
    if 'fact_bert' in metrics:
        cols += [
            'fact_bert_precision', 'fact_bert_recall',
            'fact_bert_entailed_ref_facts', 'fact_bert_entailed_cand_facts',
        ]
    if 'fact_jury' in metrics:
        for juror in JURY_MODELS:
            cols += [
                f'fact_jury_{juror}_precision',
                f'fact_jury_{juror}_recall',
                f'fact_jury_{juror}_entailed_ref_facts',
                f'fact_jury_{juror}_entailed_cand_facts',
            ]
        cols += ['fact_jury_avg_precision', 'fact_jury_avg_recall']
    if 'jury' in metrics:
        for juror in JURY_MODELS:
            cols += [
                f'jury_{juror}_completeness_score',       f'jury_{juror}_completeness_explanation',
                f'jury_{juror}_relevancy_score',          f'jury_{juror}_relevancy_explanation',
                f'jury_{juror}_faithfulness_score',       f'jury_{juror}_faithfulness_explanation',
            ]
        cols += [
            'jury_avg_completeness_score', 'jury_avg_relevancy_score',
            'jury_avg_faithfulness_score',
        ]
    if 'jury_rubric' in metrics:
        for juror in JURY_MODELS:
            cols += [f'jury_rubric_{juror}_score', f'jury_rubric_{juror}_explanation']
        cols += ['jury_rubric_avg_score', 'jury_rubric_num_criteria', 'jury_rubric_max_points']
    return cols


# ── Main ──────────────────────────────────────────────────────────────────────
async def main(args):
    metrics: set[str] = set(args.metrics)
    log.info(f"Active metrics: {metrics}")

    questions = pd.read_csv(args.questions)
    responses = pd.read_csv(args.responses)
    log.info(f"Loaded {len(questions)} questions, {len(responses)} responses")

    if 'sub_question_id' not in questions.columns:
        questions = questions.copy()
        questions['sub_question_id'] = questions['question_id'] + '_r'

    if 'jury_rubric' in metrics and not args.rubrics:
        log.error("--rubrics is required when jury_rubric metric is selected.")
        return

    rubrics_lookup: dict[str, str] = {}
    if args.rubrics:
        rubrics_df = pd.read_csv(args.rubrics, dtype=str)
        rubrics_lookup = dict(zip(rubrics_df['sub_question_id'], rubrics_df['rubric']))
        log.info(f"Loaded {len(rubrics_lookup)} rubrics from {args.rubrics}")

    all_pairs = build_pairs(responses, questions)
    if not all_pairs:
        log.error("No valid pairs to score — exiting.")
        return

    # Skip already-scored pairs; include batch_num in the key for rolling responses
    has_batch = 'batch_num' in responses.columns
    key_cols = ["sub_question_id", "model", "batch_num"] if has_batch else ["sub_question_id", "model"]
    completed = load_completed_pairs(args.output, key_cols)
    if completed:
        log.info(f"Resuming: {len(completed)} {tuple(key_cols)} pairs already scored")

    def _key(sub_id, resp):
        if has_batch:
            return (str(sub_id), str(resp['model']), str(resp.get('batch_num', '')))
        return (str(sub_id), str(resp['model']))

    pairs = [
        (sub_id, resp, q_row)
        for sub_id, resp, q_row in all_pairs
        if _key(sub_id, resp) not in completed
    ]
    log.info(f"{len(pairs)} pairs to score ({len(all_pairs) - len(pairs)} skipped)")

    if not pairs:
        log.info("All pairs already scored.")
        return

    limiter = RateLimiter(rate=100)

    # Extract candidate facts when any fact-based metric is requested
    facts_lookup: dict[tuple, list[str]] = {}
    if {'fact', 'fact_jury', 'fact_bert'} & metrics:
        # Build sub_question_id → question_id mapping for cache lookups
        sub_to_question_id: dict[str, str] = {
            str(sub_id): resp['question_id']
            for sub_id, resp, _ in all_pairs
        }

        # Load cached facts from already-scored rows in the output file
        if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
            try:
                needed_cols = {'sub_question_id', 'model', 'candidate_facts'}
                cached_df = pd.read_csv(args.output, usecols=lambda c: c in needed_cols, dtype=str)
                for _, row in cached_df.iterrows():
                    sub_id = str(row.get('sub_question_id', ''))
                    model  = str(row.get('model', ''))
                    cf_str = row.get('candidate_facts', '')
                    if not isinstance(cf_str, str) or not cf_str.strip():
                        continue
                    qid = sub_to_question_id.get(sub_id)
                    if qid is None:
                        continue
                    key = (qid, model)
                    if key not in facts_lookup:
                        try:
                            facts_lookup[key] = json.loads(cf_str)
                        except (json.JSONDecodeError, ValueError):
                            pass
                log.info(f"Loaded {len(facts_lookup)} cached fact sets from {args.output}")
            except Exception as e:
                log.warning(f"Could not load cached facts from {args.output}: {e}")

        # Extract facts only for responses not already in the cache; always uses gemini_flash
        needed_keys = {(resp['question_id'], resp['model']) for _, resp, _ in pairs}
        to_extract_keys = needed_keys - facts_lookup.keys()
        unique_responses = responses.drop_duplicates(subset=['question_id', 'model'])
        unique_responses = unique_responses[
            unique_responses.apply(lambda r: (r['question_id'], r['model']) in to_extract_keys, axis=1)
        ]
        log.info(f"Extracting facts for {len(unique_responses)} unique responses (gemini_flash)...")
        if len(unique_responses) > 0:
            extracted = await asyncio.gather(*(
                process_response(resp, "gemini_flash", limiter)
                for _, resp in unique_responses.iterrows()
            ))
            new_facts = {
                (resp['question_id'], resp['model']): facts
                for (_, resp), facts in zip(unique_responses.iterrows(), extracted)
                if facts is not None
            }
            facts_lookup.update(new_facts)
        log.info(f"Fact extraction complete: {len(facts_lookup)} total ({len(unique_responses)} newly extracted)")

    # ── Pre-batch Claude jury calls ───────────────────────────────────────────
    # Submit all Claude jury prompts as a single Anthropic batch (50% discount)
    # before starting the per-pair scoring loop; results are injected via cache.
    claude_jury_cache: dict[str, str | None] = {}
    if {'jury', 'jury_rubric'} & metrics:
        jury_requests = []
        for sub_id, resp, q_row in pairs:
            if not isinstance(resp.get('response'), str) or not resp['response'].strip():
                continue
            question  = str(q_row.get('question', ''))
            candidate = str(resp['response'])
            reference = str(q_row.get('annotation_sub_answer') or q_row.get('answer', ''))
            key_base  = f"{sub_id}||{resp['model']}||{resp.get('batch_num', '')}"

            if 'jury' in metrics:
                jury_requests.append({
                    "custom_id": f"jury||{key_base}",
                    "user_prompt": JURY_PROMPT.format(
                        QUESTION=question, RESPONSE=candidate, GOLD_RESPONSE=reference
                    ),
                    "system_instructions": JURY_SYSTEM_PROMPT,
                    "model_id": "claude_opus",
                })

            if 'jury_rubric' in metrics:
                rubric = (rubrics_lookup or {}).get(str(sub_id))
                if rubric:
                    jury_requests.append({
                        "custom_id": f"jury_rubric||{key_base}",
                        "user_prompt": RUBRIC_JURY_PROMPT.format(
                            QUESTION=question, RESPONSE=candidate,
                            GOLD_RESPONSE=reference, RUBRIC=rubric
                        ),
                        "system_instructions": RUBRIC_JURY_PROMPT_SYS,
                        "model_id": "claude_opus",
                    })

        if jury_requests:
            log.info(f"Pre-batching {len(jury_requests)} Claude jury calls via Anthropic batch API...")
            claude_jury_cache = await send_batch_messages(jury_requests)
            succeeded = sum(1 for v in claude_jury_cache.values() if v is not None)
            log.info(f"Claude jury batch complete: {succeeded}/{len(jury_requests)} succeeded")

    # Rows that need a post-batch BERTScore pass can't be written until the batch completes.
    # All other rows are written to the CSV immediately as each pair finishes.
    bert_metrics = {'bert', 'fact_bert'} & metrics
    cols = output_columns(metrics)
    writer = CsvWriter(args.output, cols)

    bert_bucket:      list = []
    fact_bert_bucket: list = []

    async def score_and_write(sub_id, resp, q_row):
        entry = await process_pair(
            sub_id, resp, q_row,
            facts_lookup.get((resp['question_id'], resp['model']), []),
            args.model, bert_bucket, fact_bert_bucket, limiter, metrics,
            rubrics_lookup=rubrics_lookup,
            claude_jury_cache=claude_jury_cache,
        )
        if entry is None:
            return None
        if not bert_metrics:
            await writer.write(entry)
        return entry

    results = await asyncio.gather(*(
        score_and_write(sub_id, resp, q_row)
        for sub_id, resp, q_row in pairs
        if not ({'fact', 'fact_jury', 'fact_bert'} & metrics) or (resp['question_id'], resp['model']) in facts_lookup
    ))

    successes = [r for r in results if r is not None]
    log.info(f"{len(successes)}/{len(pairs)} pairs scored successfully")

    if not successes:
        log.error("No results to write.")
        return

    # Batch fact BERTScore over all pairs (single model load + inference pass)
    if 'fact_bert' in metrics and fact_bert_bucket:
        all_hyps = [h for *_, hyps, _ in fact_bert_bucket for h in hyps]
        all_refs = [r for *_, _, refs  in fact_bert_bucket for r in refs]
        log.info(f"Running fact BERTScore over {len(all_hyps)} fact pairs across {len(fact_bert_bucket)} entries...")
        f1_all = []
        if all_hyps:
            from bert_score import score as bert_score_fn

            _, _, F1 = bert_score_fn(
                all_hyps, all_refs,
                model_type=BERT_MODEL,
                num_layers=BERT_NUM_LAYERS, lang="en",
                batch_size=64,
                device=_bert_device(),
                verbose=False,
            )
            f1_all = F1.tolist()
        offset = 0
        for entry, cand_facts, ref_facts, fb_pairs, _, _ in fact_bert_bucket:
            n = len(fb_pairs)
            entailed_cand: set[int] = set()
            entailed_ref:  set[int] = set()
            for (ci, ri), f1 in zip(fb_pairs, f1_all[offset:offset + n]):
                if f1 >= FACT_BERT_THRESHOLD:
                    entailed_cand.add(ci)
                    entailed_ref.add(ri)
            offset += n
            precision = len(entailed_cand) / len(cand_facts) if cand_facts else 0.0
            recall    = len(entailed_ref)  / len(ref_facts)  if ref_facts  else 0.0
            entry.update({
                'fact_bert_precision':           round(precision, 4),
                'fact_bert_recall':              round(recall, 4),
                'fact_bert_entailed_ref_facts':  json.dumps([ref_facts[i]  for i in sorted(entailed_ref)]),
                'fact_bert_entailed_cand_facts': json.dumps([cand_facts[i] for i in sorted(entailed_cand)]),
            })

    # Batch BERTScore over all successful pairs
    if 'bert' in metrics and bert_bucket:
        log.info(f"Running BERTScore over {len(bert_bucket)} pairs...")
        P, R, F1 = compute_bert_score(
            [c for _, c, _ in bert_bucket],
            [r for _, _, r in bert_bucket],
        )
        for i, (entry, _, _) in enumerate(bert_bucket):
            entry.update({
                'bert_score_precision': P[i],
                'bert_score_recall':    R[i],
                'bert_score_f1':        F1[i],
            })

    # Write all rows now (entries already have bert columns filled in above)
    if bert_metrics:
        for entry in successes:
            await writer.write(entry)

    log.info(f"Done. Wrote {len(successes)} new rows to {args.output}")
    log_token_stats(log)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    def parse_metrics(value: str) -> list[str]:
        valid = {'bert', 'rouge', 'bleu', 'fact', 'fact_bert', 'fact_jury', 'jury', 'jury_rubric'}
        chosen = [m.strip().lower() for m in value.split(',')]
        invalid = set(chosen) - valid
        if invalid:
            raise argparse.ArgumentTypeError(
                f"Invalid metric(s): {invalid}. Choose from: {valid}"
            )
        return chosen

    parser = argparse.ArgumentParser(description="Score Prediction Pipeline")
    parser.add_argument("-q", "--questions", type=str, required=True, help="Questions CSV")
    parser.add_argument("-r", "--responses", type=str, required=True, help="Responses CSV")
    parser.add_argument("-o", "--output",    type=str, required=True, help="Output CSV")
    parser.add_argument("--rubrics",          type=str, default=None,
                        help="Rubrics CSV (question_id, sub_question_id, rubric); required for jury_rubric metric")
    parser.add_argument("-m", "--model",     type=str, default="gemini",
                        choices=["gemini", "gpt", "claude"],
                        help="Model used for LLM-based fact extraction and single-model entailment (default: gemini)")
    parser.add_argument("--metrics",         type=parse_metrics,
                        default=["bert", "rouge", "fact", "jury"],
                        metavar="METRICS",
                        help=(
                            "Comma-separated list of metrics to compute (default: bert,rouge,fact,jury).\n"
                            "  bert      — BERTScore precision, recall, F1 (BioClinical-ModernBERT)\n"
                            "  rouge     — ROUGE-1, ROUGE-2, ROUGE-L F-measure\n"
                            "  bleu      — BLEU-1 through BLEU-4 with smoothing\n"
                            "  fact      — Fact precision & recall via LLM entailment (uses --model)\n"
                            "  fact_bert — Fact precision & recall via BERTScore; facts must share a date\n"
                            "              and have F1 >= 0.95 to be considered entailed (no API calls)\n"
                            "  fact_jury — Fact precision & recall via LLM entailment scored independently\n"
                            "              by claude, gpt, and gemini; records entailed fact lists per model\n"
                            "  jury      — LLM-as-a-judge scored independently by claude, gpt, and gemini\n"
                            "              on completeness, relevancy, and faithfulness (0-4 each)\n"
                            "  jury_rubric — LLM-as-a-judge using a question-specific rubric (requires --rubrics);\n"
                            "              returns a single score + explanation per juror\n"
                            "Example: --metrics bert,rouge,bleu,fact_bert,jury"
                        ))

    args = parser.parse_args()
    asyncio.run(main(args))
