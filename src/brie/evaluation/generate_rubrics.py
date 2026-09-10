'''
Rubric generation script.

For each question (or sub-question), calls an LLM to generate a question-specific
scoring rubric grounded in the reference answer and patient facts.

Input CSV columns (required): question_id, question text, reference answer, facts
  - question text column:   --question-column  (default: natural_query)
  - reference answer column: --answer-column   (default: annotation_sub_answer)
  - facts column:            --facts-column    (default: facts_edited)
  - sub_question_id is used as the scoring key when present; otherwise question_id + '_r'

Output CSV columns:
  question_id, sub_question_id, rubric
  (rubric is a JSON string matching the rubric generation schema)
'''

import argparse
import ast
import asyncio
import json
import logging
import sys

import pandas as pd

from .utils import send_single_message, CsvWriter, load_completed_pairs, safe_json_parse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────
RUBRIC_GENERATION_SYSTEM_PROMPT = """\
You are a medical expert designing a rubric to evaluate the quality of an AI-generated \
answer to a clinical question about a patient's longitudinal notes. You will be given:
1. A clinical question about the patient.
2. A reference answer (gold standard) to that question.
3. A list of patient facts, each with an associated date in the format "Patient fact (YYYY-MM-DD)".

Your task is to generate a question-specific rubric composed of discrete, objectively \
gradable criteria. Each criterion describes one attribute of a response that should be \
rewarded (positive points) or penalized (negative points). A downstream grader will decide, \
for each criterion independently, whether a candidate response meets it (binary: met / not met).

=== Rubric Design Principles ===

1. GROUND EVERY CRITERION IN THE PROVIDED FACTS AND QUESTION.
   - Each positive criterion must correspond to a specific fact (or set of related facts) \
from the patient fact list that a correct answer to the question should include.
   - Criteria must be relevant to what the question is actually asking. Do not generate \
criteria for facts that, while present in the patient record, are not responsive to the question.
   - Preserve date precision. If a fact's date is clinically relevant (e.g., onset of a \
symptom, date of a diagnosis, medication start), the criterion should require the correct \
date or an equivalent relative-time reference.
   - Do not invent facts that are not in the provided list.
2. TAG EACH CRITERION WITH ONE AXIS:
   - "completeness": the response includes a specific fact or claim present in the \
reference answer.
   - "faithfulness": the response does NOT contradict or distort a fact from the reference \
answer or patient fact list. Use this axis for NEGATIVE criteria that penalize \
contradictions, fabricated details, or distortions.
   - "relevancy": the response does NOT include extraneous details that go beyond what \
the reference answer requires, even if those details are consistent with the facts. \
Use this axis for NEGATIVE criteria that penalize unnecessary scope expansion.

3. ASSIGN POINT VALUES REFLECTING CLINICAL IMPACT.
   Use the following scale, calibrated to patient-care impact:
   - +4: Critical fact. Omission would impact interpretation AND could negatively impact \
patient care (e.g., an active diagnosis, current medication, key allergy, critical lab \
trend relevant to the question).
   - +2: Important fact. Omission would impact interpretation but not directly impact care \
(e.g., a relevant historical event or secondary finding).
   - +1: Supporting fact. Omission would not meaningfully impact interpretation.
   - -4: Dangerous contradiction or hallucination. A claim that contradicts the reference \
answer or patient facts AND could negatively impact care (faithfulness axis).
   - -2: Minor contradiction or clinically irrelevant hallucination (faithfulness axis).
   - -2: Significant irrelevant detail that expands scope beyond what the reference \
answer requires (relevancy axis).
   - -1: Minor extraneous detail (relevancy axis).

5. INCLUDE A REFUSAL CRITERION.
   Always include one negative criterion on the completeness axis:
   - Description: "The response states or implies that the requested information is not \
available in the provided notes, when in fact the reference answer shows it is available."
   - Points: -4 (a full refusal effectively zeroes out the response, consistent with the \
intent that refusals receive no credit).

6. CRITERIA MUST BE SELF-CONTAINED AND OBJECTIVELY GRADABLE.
   - A grader should be able to decide "met" or "not met" from the response alone plus the \
criterion text, without re-reading the full reference.
   - Avoid vague language ("discusses the condition adequately"). Prefer specific claims \
("States that the patient was diagnosed with type 2 diabetes on the documented date or equivalent \
date reference").
   - Allow for paraphrase and equivalent date expressions; the criterion should describe \
the CLAIM, not require verbatim wording.

7. COVERAGE.
   - Typical rubrics contain between 4 and 15 criteria depending on the complexity of \
the question and reference answer.
   - Include at least one criterion per atomic clinical claim in the reference answer.
   - Include at least one faithfulness criterion targeting the most plausible \
hallucination for this question (e.g., a wrong date, wrong medication, wrong diagnosis).
   - Include the refusal criterion (see #5).

=== Output Format ===

Return a single valid JSON object with this structure:

{
    "criteria": [
        {
            "id": "c1",
            "description": "A specific, self-contained statement of what the response must (or must not) do.",
            "axis": "completeness" | "faithfulness" | "relevancy",
            "points": <integer in [-4, -3, -2, -1, 1, 2, 3, 4]>,
            "supporting_facts": ["Patient fact text (YYYY-MM-DD)", ...],
            "rationale": "Brief justification for the point value, anchored in clinical impact."
        },
        ...
    ]
}

JSON formatting requirements:
- Use double quotes (") for all keys and string values.
- Escape any internal double quotes as \\".
- Do not include any text outside the JSON object.
- "supporting_facts" should be an empty list [] ONLY for negative criteria targeting \
hallucinations that are not tied to a specific fact (e.g., the refusal criterion).\
"""

RUBRIC_GENERATION_USER_PROMPT = """\
Generate a question-specific rubric for the following clinical QA item.

<question>
{QUESTION}
</question>

<reference_answer>
{GOLD_RESPONSE}
</reference_answer>

<patient_facts>
{PATIENT_FACTS}
</patient_facts>

Produce the rubric as JSON following the specification in your instructions. Ensure every \
positive criterion is traceable to one or more entries in <patient_facts> and that the rubric \
fully covers the claims in <reference_answer>.\
"""

_VALID_MODELS = {"gemini_pro", "gpt5", "claude_opus", "claude_haiku", "gemini_flash"}

_OUTPUT_FIELDS = ["question_id", "sub_question_id", "rubric"]


def _format_facts(raw) -> str:
    """Convert a facts_edited cell (Python list literal or list) to a newline-joined string."""
    if not raw or (isinstance(raw, float)):
        return ""
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw
    if isinstance(raw, list):
        return "\n".join(f"- {f}" for f in raw)
    return str(raw)


async def generate_rubric(
    row: pd.Series,
    question_col: str,
    answer_col: str,
    facts_col: str,
    model_id: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> str | None:
    question     = str(row.get(question_col) or "")
    gold_response = str(row.get(answer_col) or "")
    patient_facts = _format_facts(row.get(facts_col))

    last_error = None
    for attempt in range(1, max_retries + 1):
        async with semaphore:
            try:
                raw = await send_single_message(
                    user_prompt=RUBRIC_GENERATION_USER_PROMPT.format(
                        QUESTION=question,
                        GOLD_RESPONSE=gold_response,
                        PATIENT_FACTS=patient_facts,
                    ),
                    system_instructions=RUBRIC_GENERATION_SYSTEM_PROMPT,
                    model_id=model_id,
                )
                parsed = safe_json_parse(raw)
                return json.dumps(parsed)
            except Exception as e:
                last_error = e
                log.warning(f"Attempt {attempt}/{max_retries} failed for sub_question_id={row.get('sub_question_id')}: {e}")

    log.error(f"All {max_retries} attempts failed for sub_question_id={row.get('sub_question_id')}: {last_error}")
    return None


async def main(args):
    df = pd.read_csv(args.questions, dtype=str)
    log.info(f"Loaded {len(df)} rows from {args.questions}")

    if 'sub_question_id' not in df.columns:
        df = df.copy()
        df['sub_question_id'] = df['question_id'] + '_r'

    # Resolve answer column with fallback
    answer_col = args.answer_column
    if answer_col not in df.columns:
        fallback = 'answer'
        if fallback in df.columns:
            log.info(f"Column '{answer_col}' not found — falling back to '{fallback}'")
            answer_col = fallback
        else:
            log.error(f"Neither '{answer_col}' nor 'answer' found in questions CSV.")
            return

    completed = load_completed_pairs(args.output, ["sub_question_id"], nonempty_col="rubric")
    if completed:
        log.info(f"Resuming: {len(completed)} sub_question_ids already done")

    rows = [
        row for _, row in df.iterrows()
        if (str(row['sub_question_id']),) not in completed
    ]
    log.info(f"{len(rows)} rubrics to generate ({len(df) - len(rows)} skipped)")

    if not rows:
        log.info("All rubrics already generated.")
        return

    writer   = CsvWriter(args.output, _OUTPUT_FIELDS)
    semaphore = asyncio.Semaphore(args.rate)

    async def process(row):
        rubric = await generate_rubric(
            row, args.question_column, answer_col, args.facts_column,
            args.model, semaphore,
        )
        entry = {
            "question_id":    row['question_id'],
            "sub_question_id": row['sub_question_id'],
            "rubric":         rubric,
        }
        await writer.write(entry)
        if rubric:
            log.info(f"Generated rubric for sub_question_id={row['sub_question_id']}")
        return entry

    results = await asyncio.gather(*[process(row) for row in rows])
    succeeded = sum(1 for r in results if r['rubric'] is not None)
    log.info(f"Done. {succeeded}/{len(rows)} rubrics written to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate question-specific scoring rubrics")
    parser.add_argument("-q", "--questions",        required=True,
                        help="Questions CSV")
    parser.add_argument("-o", "--output",           required=True,
                        help="Output CSV (question_id, sub_question_id, rubric)")
    parser.add_argument("-m", "--model",            default="gemini_pro",
                        choices=sorted(_VALID_MODELS),
                        help="LLM for rubric generation (default: gemini_pro)")
    parser.add_argument("--question-column",        default="natural_query",
                        help="Column containing the question text (default: natural_query)")
    parser.add_argument("--answer-column",          default="annotation_sub_answer",
                        help="Column containing the reference answer (default: annotation_sub_answer)")
    parser.add_argument("--facts-column",           default="facts_edited",
                        help="Column containing the patient facts list (default: facts_edited)")
    parser.add_argument("--rate",                   type=int, default=10,
                        help="Max concurrent LLM requests (default: 10)")
    args = parser.parse_args()
    asyncio.run(main(args))
