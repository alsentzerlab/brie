import pandas as pd

from brie.evaluation.score_elo_batch import compute_elo


def test_reversed_display_order_is_not_double_flipped() -> None:
    rows = pd.DataFrame([
        {
            "model_a": "model_x",
            "model_b": "model_y",
            "position": "ab",
            "completeness_winner": "A",
            "relevancy_winner": "A",
            "concision_winner": "A",
            "overall_winner": "A",
        },
        {
            "model_a": "model_y",
            "model_b": "model_x",
            "position": "ba",
            "completeness_winner": "B",
            "relevancy_winner": "B",
            "concision_winner": "B",
            "overall_winner": "B",
        },
    ])
    summary = compute_elo(rows).set_index("model")
    assert summary.loc["model_x", "wins"] == 2
    assert summary.loc["model_x", "win_rate"] == 1.0
    assert summary.loc["model_y", "losses"] == 2
