"""Prompts for topic, evidence, answer-span, and question selection stages."""

TOPIC_SYS = """Assign up to three broad clinical topics. Return JSON with
`topics`, a list of strings. Omit names, identifiers, and locations."""
TOPIC_USER = "<question>{QUESTION}</question>"

EVIDENCE_SYS = """The user supplies JSON containing a fact and a note. Return
JSON with `evidence`, a list of minimal verbatim note excerpts supporting the
fact. Return an empty list when the note does not support it."""

ANSWER_SPAN_SYS = """Map each supplied fact to the minimal verbatim substring
of the answer that expresses it. Return JSON with `mappings`; each item contains
`fact` and either `substring` or null. Do not paraphrase substrings."""
ANSWER_SPAN_USER = """<facts>{FACTS_LIST}</facts>
<answer>{ANSWER}</answer>"""

FILTER_SYS = """Select the strongest non-duplicate questions from the supplied
JSON object. Prefer clinical usefulness, clarity, diversity, grounded answers,
and no future leakage. Return JSON with `selected_questions`; every item must
contain the integer `question_id` and a short `justification`."""
