"""Shared runtime helpers for dataset-generation stages."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import pandas as pd

VERTEX_PROJECT = os.environ.get("BRIE_VERTEX_PROJECT", "")
VERTEX_LOCATION = os.environ.get("BRIE_VERTEX_LOCATION", "global")
GEMINI_MODEL_ID = os.environ.get("BRIE_GEMINI_MODEL", "gemini-2.5-pro")

_TOKENIZER: Any = None
_vertex_client: Any = None


def _get_tokenizer() -> Any:
    """Load the optional tokenizer lazily; never perform network I/O at import time."""
    global _TOKENIZER
    if _TOKENIZER is False:
        return None
    if _TOKENIZER is None:
        try:
            import tiktoken
            _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TOKENIZER = False
    return _TOKENIZER or None


def count_tokens(text: str) -> int:
    tokenizer = _get_tokenizer()
    return len(tokenizer.encode(str(text))) if tokenizer is not None else len(str(text).split())


def _get_vertex_client() -> Any:
    global _vertex_client
    if _vertex_client is None:
        if not VERTEX_PROJECT:
            raise RuntimeError("Set BRIE_VERTEX_PROJECT before using Vertex AI")
        from google import genai
        _vertex_client = genai.Client(
            vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION
        )
    return _vertex_client


def send_single_message(
    user_prompt: str,
    system_instructions: str | None = None,
    max_retries: int = 8,
    backoff_factor: float = 1.4,
) -> str:
    """Send one synchronous Vertex request with bounded exponential retry."""
    from google.genai import types

    config = types.GenerateContentConfig(
        max_output_tokens=65535,
        system_instruction=system_instructions,
    )
    error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = _get_vertex_client().models.generate_content(
                model=GEMINI_MODEL_ID,
                contents=user_prompt,
                config=config,
            )
            return response.text or ""
        except Exception as exc:
            error = exc
            if attempt + 1 < max_retries:
                time.sleep(backoff_factor**attempt)
    raise RuntimeError(f"Vertex request failed after {max_retries} attempts: {error}")


def safe_json_parse(response_text: str) -> Any:
    """Parse plain, fenced, or prose-wrapped JSON model output."""
    candidates = [response_text]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    enclosed = re.search(r"([\[{][\s\S]*[\]}])", response_text)
    if enclosed:
        candidates.append(enclosed.group(1))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("could not parse JSON response", response_text, 0)


def load_notes(path: str) -> pd.DataFrame:
    """Load a CSV or JSON note table and normalize its `note_date` column."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".json":
        frame = pd.read_json(path, orient="records")
    else:
        raise ValueError("notes must be CSV or JSON")
    required = {"note_date", "note_title", "text"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"notes are missing required columns: {sorted(missing)}")
    frame["note_date"] = pd.to_datetime(frame["note_date"], errors="raise")
    return frame
