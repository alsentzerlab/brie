from brie.evaluation.elo_annotation.build_elo_examples import (
    PAIRWISE_PROMPT as EXAMPLE_ELO_PROMPT,
)
from brie.evaluation.fact_annotation.build_fact_examples import (
    PRECISION_PROMPT as EXAMPLE_PRECISION_PROMPT,
)
from brie.evaluation.fact_annotation.build_fact_examples import (
    RECALL_PROMPT as EXAMPLE_RECALL_PROMPT,
)
from brie.evaluation.prompts import ELO_PAIRWISE, FACT_ENTAILMENT
from brie.evaluation.score_elo_batch import PAIRWISE_PROMPT, SYSTEM_PROMPT as ELO_SYSTEM
from brie.evaluation.score_facts_batch import (
    PRECISION_PROMPT,
    RECALL_PROMPT,
    SYSTEM_PROMPT as FACT_SYSTEM,
)


def test_fact_scorer_uses_packaged_final_prompt() -> None:
    assert FACT_ENTAILMENT["status"] == "final-evaluation"
    assert FACT_SYSTEM == FACT_ENTAILMENT["system"]
    assert RECALL_PROMPT == EXAMPLE_RECALL_PROMPT == FACT_ENTAILMENT["recall"]
    assert PRECISION_PROMPT == EXAMPLE_PRECISION_PROMPT == FACT_ENTAILMENT["precision"]
    for prompt in (RECALL_PROMPT, PRECISION_PROMPT):
        assert "{REFERENCE_FACTS}" in prompt
        assert "{CANDIDATE_FACTS}" in prompt


def test_elo_scorer_uses_packaged_final_prompt() -> None:
    assert ELO_PAIRWISE["status"] == "final-evaluation"
    assert ELO_SYSTEM == ELO_PAIRWISE["system"]
    assert PAIRWISE_PROMPT == EXAMPLE_ELO_PROMPT == ELO_PAIRWISE["pairwise"]
    for field in ("QUESTION", "REFERENCE", "RESPONSE_A", "RESPONSE_B"):
        assert "{" + field + "}" in PAIRWISE_PROMPT
