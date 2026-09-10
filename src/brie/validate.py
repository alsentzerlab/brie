"""Validate external BRIE inputs without reading them into the repository."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def _columns(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return set(next(csv.reader(handle), []))


def _require_columns(path: Path, required: set[str]) -> None:
    missing = required - _columns(path)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def _question_subjects(path: Path) -> set[str]:
    subjects: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            question_id = str(row["question_id"])
            if "_" not in question_id:
                raise ValueError(f"{path}:{row_number}: question_id must contain an underscore")
            subject, _ = question_id.split("_", 1)
            if not subject:
                raise ValueError(f"{path}:{row_number}: empty subject portion of question_id")
            try:
                datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"{path}:{row_number}: invalid ISO timestamp") from exc
            subjects.add(subject)
    return subjects


def _validate_notes(notes_dir: Path, subjects: set[str]) -> None:
    for subject in sorted(subjects):
        path = notes_dir / f"{subject}_subsetrecords.json"
        if not path.is_file():
            raise ValueError(f"missing notes file: {path}")
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"{path}: top-level value must be a JSON array")
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"{path}: record {index} must be an object")
            missing = {"note_date", "note_title", "text"} - set(record)
            if missing:
                raise ValueError(f"{path}: record {index} missing {sorted(missing)}")
            try:
                datetime.fromisoformat(str(record["note_date"]).replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"{path}: record {index} has an invalid ISO note_date") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--notes", type=Path)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--require-references", action="store_true")
    args = parser.parse_args()

    question_columns = {"question_id", "natural_query", "timestamp"}
    if args.require_references:
        question_columns.add("annotation_sub_answer")
    _require_columns(args.questions, question_columns)
    subjects = _question_subjects(args.questions)

    if args.notes:
        if not args.notes.is_dir():
            raise ValueError(f"not a notes directory: {args.notes}")
        _validate_notes(args.notes, subjects)
    if args.responses:
        _require_columns(args.responses, {"question_id", "model", "response"})

    print(
        f"valid: {len(subjects)} subjects; "
        f"notes={'checked' if args.notes else 'not requested'}; "
        f"responses={'checked' if args.responses else 'not requested'}"
    )


if __name__ == "__main__":
    main()
