from brie.evaluation.hallucination.score import score_claim_support


def test_claim_support_metrics() -> None:
    result = score_claim_support(["a", "b", "c"], [0, 2])
    assert result["unsupported_claim_count"] == 1
    assert result["hallucination_rate"] == 1 / 3


def test_empty_answer_is_not_hallucinated() -> None:
    result = score_claim_support([], [])
    assert result["claim_precision"] == 1.0
    assert result["hallucination_rate"] == 0.0
