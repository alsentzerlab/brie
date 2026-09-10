"""Audit future-note leakage and score dated-fact temporal consistency."""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Iterable
from datetime import datetime, timezone


def parse_time(value: object) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)


def audit_note_dates(query_timestamp: object, note_dates: Iterable[object]) -> dict[str, float | int]:
    cutoff = parse_time(query_timestamp)
    if cutoff is None:
        raise ValueError(f"invalid query timestamp: {query_timestamp!r}")
    parsed = [parse_time(value) for value in note_dates]
    future = sum(stamp is not None and stamp > cutoff for stamp in parsed)
    undated = sum(stamp is None for stamp in parsed)
    return {
        "note_count": len(parsed),
        "future_note_count": future,
        "undated_note_count": undated,
        "future_note_rate": future / len(parsed) if parsed else 0.0,
        "has_future_leakage": int(future > 0),
    }


def temporal_fact_score(reference_dates: Iterable[object], candidate_dates: Iterable[object]) -> dict[str, float | int]:
    reference = {stamp.date() for value in reference_dates if (stamp := parse_time(value)) is not None}
    candidate = {stamp.date() for value in candidate_dates if (stamp := parse_time(value)) is not None}
    overlap = reference & candidate
    return {
        "reference_date_count": len(reference),
        "candidate_date_count": len(candidate),
        "date_precision": len(overlap) / len(candidate) if candidate else float(not reference),
        "date_recall": len(overlap) / len(reference) if reference else 1.0,
    }


def _list(value: object) -> list:
    if isinstance(value, list):
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(str(value))
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return []


def _note_dates(value: object) -> list[object]:
    items = _list(value)
    return [item.get("note_date") if isinstance(item, dict) else item for item in items]


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="CSV containing query timestamps and note-date lists")
    parser.add_argument("output", help="Destination CSV")
    parser.add_argument("--questions", help="Questions CSV used to add timestamps by question_id")
    parser.add_argument("--query-column", default="timestamp")
    parser.add_argument("--note-dates-column", default="retrieved_chunks")
    parser.add_argument("--reference-dates-column")
    parser.add_argument("--candidate-dates-column")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if args.query_column not in frame.columns:
        if not args.questions:
            raise ValueError(f"{args.query_column!r} is absent; provide --questions")
        questions = pd.read_csv(args.questions, usecols=["question_id", args.query_column])
        frame = frame.merge(questions.drop_duplicates("question_id"), on="question_id", how="left")
    rows = []
    for _, row in frame.iterrows():
        metrics = audit_note_dates(row[args.query_column], _note_dates(row[args.note_dates_column]))
        if args.reference_dates_column and args.candidate_dates_column:
            metrics.update(temporal_fact_score(
                _list(row[args.reference_dates_column]), _list(row[args.candidate_dates_column])
            ))
        rows.append(metrics)
    pd.concat([frame, pd.DataFrame(rows)], axis=1).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
