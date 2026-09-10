from brie.generation.prompts.export_annotations import ANSWER_SPAN_USER, TOPIC_USER
from brie.generation.prompts.generate_answer import (
    clinical_dedup_prompt,
    clinical_filter_prompt,
    clinical_qa_prompt,
    clinical_qa_update_prompt,
    clinical_seed_prompt,
)
from brie.generation.prompts.generate_questions import FACT_SYS, HP_SYS, TIMESTAMP_SYS
from brie.generation.prompts.revise_answer import ANSWER_USER


def test_prompt_templates_match_callers() -> None:
    clinical_seed_prompt.format(
        natural_query="q", reference_question="q", reference_answer="a"
    )
    clinical_filter_prompt.format(seed_query_focus="f", seed_timeframe="t", fact_list="x")
    clinical_qa_prompt.format(
        natural_query="q", seed_query_focus="f", seed_timeframe="t", seed_topic="p",
        seed_detail_level="d", persona="r", fact_list="x"
    )
    clinical_qa_update_prompt.format(
        natural_query="q", seed_query_focus="f", seed_timeframe="t", seed_topic="p",
        seed_detail_level="d", persona="r", current_question="q", current_answer="a",
        current_supporting_facts="[]", new_facts="x"
    )
    clinical_dedup_prompt.format(
        natural_query="q", reference_question="q", reference_answer="a", qa_pairs="[]"
    )
    FACT_SYS.format(FACTS="[]")
    HP_SYS.format(NOTE="note")
    TIMESTAMP_SYS.format(TIMESTAMP="time")
    TOPIC_USER.format(QUESTION="q")
    ANSWER_SPAN_USER.format(FACTS_LIST="[]", ANSWER="a")
    ANSWER_USER.format(TIMESTAMP="time", QUESTION="q", ANSWER="a", COMMENT="c", FACTS="[]")
