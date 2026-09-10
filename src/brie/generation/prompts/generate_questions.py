"""De-identified prompts used by the longitudinal question generator."""

GENERATE_HP_SYS = """Generate one clinically useful question-answer item from
the supplied admission summary. Return one JSON object, not a wrapper or
markdown. Never expose names or identifiers."""

GENERATE_SYS = """Generate one diverse longitudinal clinical question-answer
item from the supplied dated facts. Return one JSON object, not a wrapper or
markdown. Use only supplied evidence and preserve uncertainty and negation."""

GENERATE_USER = """Generate the next non-duplicate item. Include `question`,
`question_rewrite`, `answer`, `fact_subset`, `question_type`, and
`clinical_relevance_rationale`. `fact_subset` must contain verbatim supplied
facts. `question_type` must be `recent`, `past`, or `multi`."""

FACT_SYS = """

<dated_facts>
{FACTS}
</dated_facts>
"""

HP_SYS = """

<admission_summary>
{NOTE}
</admission_summary>
"""

TIMESTAMP_SYS = """

The question is asked as of {TIMESTAMP}. Do not use evidence after this time.
"""
