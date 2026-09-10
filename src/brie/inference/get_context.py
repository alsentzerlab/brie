'''
Prediction script
This script will take in the following:
1. Question file: question_id contains {patient_id}_{question number}, question
2. Notes Directory: Json files labeled with {patient_id}_subsetrecords.json (ordered most recent to least recent)
3. Output path: file to write all answers

For each question in the question file, send an api call for claude, gemini, and gpt using the question and notes input.

Outfile will have the following format:
-question_id
-response
-model (claude|gemini|gpt)
'''

import argparse
import json
import os

import pandas as pd

from .utils import MODEL_CONTEXT_LIMITS, ALL_MODELS



def format_notes(records: list[dict], char_budget: int) -> tuple[str, int, int]:
    """
    Format notes from most-recent to least-recent, stopping before exceeding char_budget.
    Returns (formatted_notes_string, notes_included, notes_total).
    """
    parts = []
    used = 0
    for i, item in enumerate(records):
        chunk = (
            f"Note Title: {item['note_title']}\n"
            f"Note Date: {item['note_date']}\n"
            f"Text: {item['text']}\n"
        )
        if used + len(chunk) > char_budget:
            return i, len(records), item['note_date']
        parts.append(chunk)
        used += len(chunk)
    return len(records), len(records), records[len(records)-1]['note_date']


def load_patient_records(notes_dir: str, patient_id: str) -> list[dict] | None:
    """Load raw records list for a patient. Returns None on failure."""
    notes_path = os.path.join(notes_dir, f"{patient_id}_subsetrecords.json")
    with open(notes_path, "r") as f:
        return json.load(f)



def main(args):
    df = pd.read_csv(args.questions)

    df["patient_id"] = df["question_id"].str.split("_").str[0]
    ret = []
    for _, row in df.iterrows():
        patient_id=row['patient_id']
        notes = load_patient_records(args.notes, patient_id)
        for model in args.models:
            n_notes, n_total, date = format_notes(notes, MODEL_CONTEXT_LIMITS[model])
            ret.append(
                {
                    'patient_id': patient_id,
                    'model': model,
                    'n_notes': n_notes,
                    'n_total': n_total,
                    'date': date
                }
            )
    pd.DataFrame(ret).to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QA Generation Pipeline")
    parser.add_argument("-q", "--questions", type=str, required=True,
                    help="CSV file containing questions")
    parser.add_argument("-n", "--notes", type=str, required=True,
                    help="Directory containing patient notes")
    parser.add_argument("-o", "--output", type=str, required=True,
                    help="Path to output CSV")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
                    metavar="MODEL",
                    help=f"Models to run (default: all). Choices: {', '.join(ALL_MODELS)}")
    args = parser.parse_args()
    main(args)
