import json

import pandas as pd

from brie.evaluation.find_facts import classify_timeline, load_fact_rows


def test_classify_timeline() -> None:
    assert classify_timeline(["2024-01-01"], "2024-02-01") == "before_cutoff"
    assert classify_timeline(["2024-03-01"], "2024-02-01") == "after_cutoff"
    assert classify_timeline(["2024-01-01", "2024-03-01"], "2024-02-01") == "spans_cutoff"
    assert classify_timeline([], "2024-02-01") == "undated"


def test_load_fact_rows_expands_serialized_lists(tmp_path) -> None:
    path = tmp_path / "facts.csv"
    pd.DataFrame(
        [
            {
                "question_id": "subject1_q1",
                "timestamp": "2024-02-01",
                "facts_atomic": json.dumps(["a", "b"]),
            }
        ]
    ).to_csv(path, index=False)

    rows = load_fact_rows(str(path))

    assert [row["fact"] for row in rows] == ["a", "b"]
    assert [row["fact_id"] for row in rows] == [
        "subject1_q1_fact_0",
        "subject1_q1_fact_1",
    ]
    assert {row["patient_id"] for row in rows} == {"subject1"}
