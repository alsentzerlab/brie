from brie.evaluation.temporality.score import audit_note_dates, temporal_fact_score
from brie.inference.utils import filter_records_as_of


def test_future_note_audit() -> None:
    result = audit_note_dates("2020-02-01", ["2020-01-01", "2020-03-01", None])
    assert result["future_note_count"] == 1
    assert result["undated_note_count"] == 1


def test_inference_cutoff_excludes_future_and_undated_notes() -> None:
    records = [
        {"note_date": "2020-01-01", "text": "allowed"},
        {"note_date": "2020-03-01", "text": "future"},
        {"text": "undated"},
    ]
    assert filter_records_as_of(records, "2020-02-01") == [records[0]]


def test_temporal_fact_overlap() -> None:
    result = temporal_fact_score(["2020-01-01", "2020-02-01"], ["2020-02-01"])
    assert result["date_precision"] == 1.0
    assert result["date_recall"] == 0.5
