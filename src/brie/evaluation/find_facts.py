"""Locate answer facts in longitudinal notes and flag unsupported facts.

The command accepts either one fact per CSV row or a serialized list of facts
per row. It retrieves candidate notes, asks a configured model to confirm
support, and writes one JSON object per fact. A fact is marked hallucinated
when the configured search completes without finding supporting evidence.
Model-call failures produce an ``inconclusive`` result instead of a false
hallucination label.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import logging
import math
import os
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from brie.inference.rag_utils import (
    build_embeddings_batch,
    chunk_patient_notes,
    encode_queries,
    load_cached_embeddings,
)
from brie.inference.utils import ALL_MODELS, safe_json_parse, send_single_message

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "gemini_flash"

_CONFIRM_SYSTEM = (
    "You are a clinical evidence verifier. Decide whether the note independently "
    "documents the exact fact, including the same event, medication, measurement, and date."
)


def _confirm_prompt(fact: str, note: dict[str, Any]) -> str:
    return (
        f"Fact: {fact}\n\n"
        f"Note date: {note.get('note_date', '')}\n"
        f"Note title: {note.get('note_title', '')}\n"
        f"Note text:\n{note.get('text', '')}\n\n"
        "Return JSON only: "
        '{"supported": true or false, "evidence": "shortest exact supporting substring"}. '
        "Use supported=false and an empty evidence string unless the note clearly supports "
        "every material part of the fact."
    )


def _parse_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(str(value))
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def _parse_time(value: object) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)


def classify_timeline(note_dates: Iterable[object], cutoff: object) -> str:
    """Classify supporting-note dates relative to the question cutoff."""
    dates = [parsed for value in note_dates if (parsed := _parse_time(value)) is not None]
    cutoff_time = _parse_time(cutoff)
    if not dates:
        return "undated"
    if cutoff_time is None:
        return "cutoff_unknown"
    before = any(value < cutoff_time for value in dates)
    equal = any(value == cutoff_time for value in dates)
    after = any(value > cutoff_time for value in dates)
    if after and (before or equal):
        return "spans_cutoff"
    if after:
        return "after_cutoff"
    if equal and not before:
        return "at_cutoff"
    return "before_cutoff"


def _patient_from_question(question_id: object) -> str:
    return str(question_id).split("_", 1)[0]


def load_fact_rows(
    facts_path: str,
    questions_path: str | None = None,
    fact_column: str = "fact",
    facts_column: str | None = None,
    patient_column: str = "patient_id",
    cutoff_column: str = "timestamp",
) -> list[dict[str, str]]:
    """Normalize row-per-fact or list-per-row CSV input into fact records."""
    frame = pd.read_csv(facts_path)
    if questions_path:
        questions = pd.read_csv(questions_path)
        join_columns = ["question_id"]
        for column in (patient_column, cutoff_column):
            if column in questions.columns and column not in frame.columns:
                join_columns.append(column)
        frame = frame.merge(
            questions[join_columns].drop_duplicates("question_id"),
            on="question_id",
            how="left",
            validate="many_to_one",
        )

    list_column = facts_column
    if list_column is None and fact_column not in frame.columns:
        list_column = next(
            (name for name in ("facts_atomic", "facts", "candidate_facts") if name in frame),
            None,
        )
    if fact_column not in frame.columns and list_column is None:
        raise ValueError(
            f"input needs {fact_column!r} or a serialized facts column; "
            "pass --facts-column explicitly"
        )

    rows: list[dict[str, str]] = []
    for row_index, row in frame.iterrows():
        question_id = str(row.get("question_id", row_index))
        patient_value = row.get(patient_column)
        patient_id = (
            str(patient_value)
            if patient_value is not None and not pd.isna(patient_value)
            else _patient_from_question(question_id)
        )
        facts = _parse_list(row[list_column]) if list_column else [str(row[fact_column])]
        for fact_index, fact in enumerate(facts):
            supplied_id = row.get("fact_id") if not list_column else None
            fact_id = (
                str(supplied_id)
                if supplied_id is not None and not pd.isna(supplied_id)
                else f"{question_id}_fact_{fact_index}"
            )
            cutoff = row.get(cutoff_column, "")
            rows.append(
                {
                    "patient_id": patient_id,
                    "question_id": question_id,
                    "fact_id": fact_id,
                    "fact": fact,
                    "cutoff": "" if pd.isna(cutoff) else str(cutoff),
                }
            )
    return rows


def _load_notes(notes_dir: str, patient_id: str) -> list[dict[str, Any]]:
    path = Path(notes_dir) / f"{patient_id}_subsetrecords.json"
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON array")
    return records


def _bm25_candidates(fact: str, chunks: list[dict], top_k: int) -> dict[int, float]:
    from rank_bm25 import BM25Okapi

    def tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.casefold())

    corpus = [tokenize(chunk.get("text", "")) for chunk in chunks]
    scores = BM25Okapi(corpus).get_scores(tokenize(fact))
    ranked = np.argsort(scores)[::-1]
    note_scores: dict[int, float] = {}
    for chunk_index in ranked:
        score = float(scores[chunk_index])
        if score <= 0:
            continue
        note_index = int(chunks[chunk_index]["note_idx"])
        note_scores[note_index] = max(note_scores.get(note_index, 0.0), score)
        if len(note_scores) >= top_k:
            break
    return note_scores


def _semantic_candidates(
    query_embedding: np.ndarray,
    chunks: list[dict],
    document_embeddings: np.ndarray,
    threshold: float,
    top_k: int,
) -> dict[int, float]:
    similarities = document_embeddings @ query_embedding
    note_scores: dict[int, float] = {}
    for chunk_index in np.argsort(similarities)[::-1]:
        score = float(similarities[chunk_index])
        if score < threshold:
            break
        note_index = int(chunks[chunk_index]["note_idx"])
        note_scores[note_index] = max(note_scores.get(note_index, -1.0), score)
        if len(note_scores) >= top_k:
            break
    return note_scores


async def _confirm(
    fact: str,
    note: dict[str, Any],
    model: str,
    semaphore: asyncio.Semaphore,
) -> tuple[bool, str, str | None]:
    async with semaphore:
        try:
            response = await send_single_message(
                _confirm_prompt(fact, note), _CONFIRM_SYSTEM, model_id=model
            )
            parsed = safe_json_parse(response)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("supported"), bool):
                raise ValueError("confirmation response lacks a boolean supported field")
            supported = parsed["supported"]
            evidence = str(parsed.get("evidence", "")).strip()
            if supported and not evidence:
                raise ValueError("supported confirmation lacks evidence")
            return supported, evidence, None
        except Exception as error:  # preserve an inconclusive result instead of false certainty
            return False, "", str(error)


def _result(
    fact_row: dict[str, str],
    entries: list[dict[str, Any]],
    candidate_count: int,
    errors: list[str],
) -> dict[str, Any]:
    dates = [str(entry.get("note_date", "")) for entry in entries]
    parsed_dates = sorted(value for value in (_parse_time(date) for date in dates) if value)
    if entries:
        status = "supported"
        hallucinated: bool | None = False
        timeline = classify_timeline(dates, fact_row["cutoff"])
    elif errors:
        status = "inconclusive"
        hallucinated = None
        timeline = "inconclusive"
    else:
        status = "unsupported"
        hallucinated = True
        timeline = "unsupported"
    return {
        **fact_row,
        "status": status,
        "is_hallucinated": hallucinated,
        "timeline_position": timeline,
        "earliest_note_date": parsed_dates[0].date().isoformat() if parsed_dates else None,
        "latest_note_date": parsed_dates[-1].date().isoformat() if parsed_dates else None,
        "candidate_note_count": candidate_count,
        "supporting_note_count": len(entries),
        "confirmation_error_count": len(errors),
        "confirmation_errors": errors,
        "entries": entries,
    }


async def _process_patient(
    patient_id: str,
    fact_rows: list[dict[str, str]],
    args: argparse.Namespace,
    output_lock: asyncio.Lock,
) -> None:
    records = _load_notes(args.notes, patient_id)
    chunks = chunk_patient_notes(records, patient_id)
    embeddings: np.ndarray | None = None
    query_embeddings: np.ndarray | None = None
    if args.retrieval == "hybrid":
        if not args.embeddings_dir:
            raise ValueError("--embeddings-dir is required for hybrid retrieval")
        cached = load_cached_embeddings(patient_id, args.embeddings_dir)
        if cached is None:
            cached = build_embeddings_batch({patient_id: records}, args.embeddings_dir)[patient_id]
        chunks, embeddings = cached
        query_embeddings = encode_queries([row["fact"] for row in fact_rows])

    semaphore = asyncio.Semaphore(args.concurrent)
    for fact_index, fact_row in enumerate(fact_rows):
        bm25 = _bm25_candidates(fact_row["fact"], chunks, args.top_k)
        semantic: dict[int, float] = {}
        if query_embeddings is not None and embeddings is not None:
            semantic = _semantic_candidates(
                query_embeddings[fact_index],
                chunks,
                embeddings,
                args.sim_threshold,
                args.top_k,
            )
        candidate_ids = list(dict.fromkeys([*bm25, *semantic]))[: args.top_k]
        confirmations = await asyncio.gather(
            *(
                _confirm(fact_row["fact"], records[note_index], args.model, semaphore)
                for note_index in candidate_ids
            )
        )
        entries: list[dict[str, Any]] = []
        errors: list[str] = []
        for note_index, (supported, evidence, error) in zip(candidate_ids, confirmations):
            if error:
                errors.append(error)
                continue
            if not supported:
                continue
            note = records[note_index]
            entries.append(
                {
                    "note_id": note.get("note_id", note_index),
                    "note_date": str(note.get("note_date", "")),
                    "note_title": note.get("note_title", ""),
                    "evidence": evidence,
                    "retrieved_by": [
                        name
                        for name, scores in (("bm25", bm25), ("semantic", semantic))
                        if note_index in scores
                    ],
                    "bm25_score": bm25.get(note_index),
                    "cosine_similarity": semantic.get(note_index),
                }
            )
        result = _result(fact_row, entries, len(candidate_ids), errors)
        async with output_lock:
            with open(args.output, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")


def _completed(output: str) -> set[str]:
    if not os.path.exists(output):
        return set()
    completed: set[str] = set()
    with open(output, encoding="utf-8") as handle:
        for line in handle:
            try:
                completed.add(str(json.loads(line)["fact_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


async def run(args: argparse.Namespace) -> None:
    rows = load_fact_rows(
        args.facts,
        args.questions,
        args.fact_column,
        args.facts_column,
        args.patient_column,
        args.cutoff_column,
    )
    completed = _completed(args.output)
    rows = [row for row in rows if row["fact_id"] not in completed]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output_lock = asyncio.Lock()
    patient_rows: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        patient_rows.setdefault(row["patient_id"], []).append(row)
    for patient_id, group in patient_rows.items():
        LOGGER.info("Processing %s facts for subject %s", len(group), patient_id)
        await _process_patient(patient_id, group, args, output_lock)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", required=True, help="CSV containing facts or fact lists")
    parser.add_argument("--questions", help="Optional question CSV supplying subject and cutoff")
    parser.add_argument("--notes", required=True, help="Directory of subject note JSON files")
    parser.add_argument("--output", required=True, help="Append/resume JSONL output")
    parser.add_argument("--fact-column", default="fact")
    parser.add_argument("--facts-column", help="Serialized list column; inferred when omitted")
    parser.add_argument("--patient-column", default="patient_id")
    parser.add_argument("--cutoff-column", default="timestamp")
    parser.add_argument("--retrieval", choices=["bm25", "hybrid"], default="bm25")
    parser.add_argument("--embeddings-dir", help="Required only for hybrid retrieval")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--sim-threshold", type=float, default=0.4)
    parser.add_argument("--model", choices=ALL_MODELS, default=DEFAULT_MODEL)
    parser.add_argument("--concurrent", type=int, default=8)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
