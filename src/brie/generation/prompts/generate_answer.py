"""De-identified prompt templates for derived question-answer generation."""

clinical_seed_prompt = """Create a JSON array of candidate reasoning seeds for
the natural query. Each item must contain integer `seed_id` and strings
`query_focus`, `timeframe`, `topic`, and `detail_level`. Use the reference only
to understand the task; do not copy identifiers.

Natural query: {natural_query}
Reference question: {reference_question}
Reference answer: {reference_answer}
"""

clinical_filter_prompt = """Resolve the seed timeframe against the dated fact
list. Return JSON with `resolved_timeframe`, `start_date`, and `end_date` using
ISO dates. Return null dates if the timeframe cannot be grounded.

Focus: {seed_query_focus}
Timeframe: {seed_timeframe}
Facts:
{fact_list}
"""

clinical_qa_prompt = """Using only the supplied dated facts, generate one
clinically useful, de-identified QA pair. Return JSON with `question`, `answer`,
and `supporting_facts` (verbatim facts from the list). Return a null question if
the request cannot be answered. Preserve negation, uncertainty, and timing.

Natural query: {natural_query}
Focus: {seed_query_focus}
Timeframe: {seed_timeframe}
Topic: {seed_topic}
Detail level: {seed_detail_level}
Perspective: {persona}
Facts:
{fact_list}
"""

clinical_qa_update_prompt = """Update the existing QA pair only when the new
facts add relevant supported information. Return JSON with `question`, `answer`,
and `supporting_facts`; otherwise return the existing pair unchanged.

Natural query: {natural_query}
Focus: {seed_query_focus}
Timeframe: {seed_timeframe}
Topic: {seed_topic}
Detail level: {seed_detail_level}
Perspective: {persona}
Current question: {current_question}
Current answer: {current_answer}
Current supporting facts: {current_supporting_facts}
New facts:
{new_facts}
"""

clinical_dedup_prompt = """Remove duplicate, unsupported, or incoherent QA
pairs. Return JSON with `retained`, a list of retained QA objects. Do not add
facts or identifiers.

Natural query: {natural_query}
Reference question: {reference_question}
Reference answer: {reference_answer}
Candidate QA pairs: {qa_pairs}
"""
