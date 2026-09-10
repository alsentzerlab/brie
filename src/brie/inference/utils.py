"""Provider-neutral model and incremental-output utilities.

No credential, endpoint, cloud project, bucket, or organization identifier is
embedded here. Provider configuration is read from the environment at runtime.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

_TOKENIZER = None


def _get_tokenizer():
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
_token_stats = {"prompt": 0, "completion": 0, "calls": 0}

# Public aliases are intentionally configurable. Empty cloud values fail only
# when the corresponding backend is selected.
VERTEX_PROJECT = os.getenv("BRIE_VERTEX_PROJECT", "")
VERTEX_GEMINI_PROJECT = os.getenv("BRIE_VERTEX_GEMINI_PROJECT", VERTEX_PROJECT)
VERTEX_LOCATION = os.getenv("BRIE_VERTEX_LOCATION", "global")
VERTEX_GCS_BUCKET = os.getenv("BRIE_VERTEX_GCS_BUCKET", "")
VERTEX_CLAUDE_GCS_BUCKET = os.getenv("BRIE_VERTEX_CLAUDE_GCS_BUCKET", "")

_VERTEX_GEMINI_MODELS = {
    "gemini_pro": os.getenv("BRIE_GEMINI_PRO_MODEL", "gemini-2.5-pro"),
    "gemini_flash": os.getenv("BRIE_GEMINI_FLASH_MODEL", "gemini-2.5-flash"),
    "gemini_flash_juror": os.getenv("BRIE_GEMINI_JUDGE_MODEL", "gemini-2.5-flash"),
}
_VERTEX_CLAUDE_MODELS = {
    "claude_opus": os.getenv("BRIE_CLAUDE_OPUS_MODEL", "claude-opus-4-1"),
    "claude_sonnet": os.getenv("BRIE_CLAUDE_SONNET_MODEL", "claude-sonnet-4"),
    "claude_haiku": os.getenv("BRIE_CLAUDE_HAIKU_MODEL", "claude-haiku-4-5"),
}

# Batch-looking aliases are accepted for compatibility, but this clean release
# dispatches them as concurrent online calls and stores no cloud locations.
_BATCH_BASE = {
    "gemini_pro_batch": "gemini_pro",
    "gemini_flash_batch": "gemini_flash",
    "gemini_flash_juror_batch": "gemini_flash_juror",
    "claude_opus_batch": "claude_opus",
    "claude_sonnet_batch": "claude_sonnet",
    "claude_haiku_batch": "claude_haiku",
}
VERTEX_BATCH_MODELS = frozenset(_BATCH_BASE)

_OPENAI_ALIASES = {
    name.strip(): name.strip()
    for name in os.getenv("BRIE_OPENAI_MODELS", "openai,gpt5,gpt5_nano,gpt5_nano_sandbox").split(",")
    if name.strip()
}
_LOCAL_ALIASES = {
    name.strip(): name.strip()
    for name in os.getenv("BRIE_LOCAL_MODELS", "local").split(",")
    if name.strip()
}
VLLM_MODELS = frozenset(_LOCAL_ALIASES)

MODEL_CONTEXT_LIMITS = {
    **{name: 950_000 for name in _VERTEX_GEMINI_MODELS},
    **{name: 150_000 for name in _VERTEX_CLAUDE_MODELS},
    **{name: 120_000 for name in _OPENAI_ALIASES},
    **{name: 120_000 for name in _LOCAL_ALIASES},
    **{name: 150_000 for name in _BATCH_BASE},
}
ALL_MODELS = sorted(MODEL_CONTEXT_LIMITS)


def count_tokens(text: str) -> int:
    text = str(text)
    tokenizer = _get_tokenizer()
    return len(tokenizer.encode(text)) if tokenizer is not None else len(text.split())


def get_token_stats() -> dict[str, int]:
    return dict(_token_stats)


def reset_token_stats() -> None:
    _token_stats.update(prompt=0, completion=0, calls=0)


def log_token_stats(logger: Any = None, prefix: str = "") -> None:
    total = _token_stats["prompt"] + _token_stats["completion"]
    message = f"{prefix}tokens: {total:,}; calls: {_token_stats['calls']:,}"
    logger.info(message) if logger else print(message)


class _TPMRateLimiter:
    def __init__(self, tokens_per_minute: int):
        self.limit = max(1, tokens_per_minute)
        self.events: list[tuple[float, int]] = []
        self.lock = asyncio.Lock()

    async def wait(self, tokens: int) -> None:
        while True:
            async with self.lock:
                now = time.monotonic()
                self.events = [(stamp, count) for stamp, count in self.events if now - stamp < 60]
                if sum(count for _, count in self.events) + tokens <= self.limit:
                    self.events.append((now, tokens))
                    return
                delay = max(0.05, 60 - (now - self.events[0][0]))
            await asyncio.sleep(delay)


_tpm_limiter: _TPMRateLimiter | None = None


def set_tpm_limit(tpm: int) -> None:
    global _tpm_limiter
    _tpm_limiter = _TPMRateLimiter(tpm)


def _example_gemini_contents(examples: list[dict] | None) -> list[dict]:
    return [
        {"role": "model" if item["role"] in {"assistant", "model"} else "user",
         "parts": [{"text": item["content"]}]}
        for item in examples or []
    ]


def _chat_messages(system: str | None, examples: list[dict] | None, user: str) -> list[dict]:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    for item in examples or []:
        role = "assistant" if item["role"] in {"assistant", "model"} else "user"
        messages.append({"role": role, "content": item["content"]})
    messages.append({"role": "user", "content": user})
    return messages


def _require(value: str, variable: str) -> str:
    if not value:
        raise RuntimeError(f"Set {variable} before using this backend")
    return value


async def _send_vertex_gemini(user: str, system: str | None, alias: str) -> str:
    from google import genai
    from google.genai import types

    project = _require(VERTEX_GEMINI_PROJECT, "BRIE_VERTEX_GEMINI_PROJECT")
    client = genai.Client(vertexai=True, project=project, location=VERTEX_LOCATION)
    config = types.GenerateContentConfig(system_instruction=system) if system else None
    response = await client.aio.models.generate_content(
        model=_VERTEX_GEMINI_MODELS[alias], contents=user, config=config
    )
    return response.text or ""


async def _send_vertex_claude(user: str, system: str | None, alias: str) -> str:
    from anthropic import AsyncAnthropicVertex

    project = _require(VERTEX_PROJECT, "BRIE_VERTEX_PROJECT")
    client = AsyncAnthropicVertex(project_id=project, region=VERTEX_LOCATION)
    kwargs: dict[str, Any] = {
        "model": _VERTEX_CLAUDE_MODELS[alias],
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        kwargs["system"] = system
    response = await client.messages.create(**kwargs)
    return response.content[0].text if response.content else ""


async def _send_openai(
    user: str, system: str | None, alias: str, examples: list[dict] | None, local: bool
) -> str:
    import openai

    prefix = "BRIE_LOCAL" if local else "BRIE_OPENAI"
    base_url = _require(os.getenv(f"{prefix}_BASE_URL", ""), f"{prefix}_BASE_URL")
    api_key = _require(os.getenv(f"{prefix}_API_KEY", ""), f"{prefix}_API_KEY")
    model = os.getenv(f"{prefix}_MODEL", alias)
    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
    response = await client.chat.completions.create(
        model=model, messages=_chat_messages(system, examples, user)
    )
    return response.choices[0].message.content or ""


async def send_single_message(
    user_prompt: str,
    system_instructions: str | None = None,
    model_id: str = "gemini_pro",
    backend: str = "vertex",
    cache_system: bool = False,
    user_prefix: str | None = None,
    example_messages: list[dict] | None = None,
) -> str:
    del cache_system
    user = (user_prefix or "") + user_prompt
    alias = _BATCH_BASE.get(model_id, model_id)
    if _tpm_limiter:
        await _tpm_limiter.wait(count_tokens((system_instructions or "") + user))
    if alias in _VERTEX_GEMINI_MODELS:
        result = await _send_vertex_gemini(user, system_instructions, alias)
    elif alias in _VERTEX_CLAUDE_MODELS:
        result = await _send_vertex_claude(user, system_instructions, alias)
    elif alias in _LOCAL_ALIASES:
        result = await _send_openai(user, system_instructions, alias, example_messages, True)
    elif alias in _OPENAI_ALIASES:
        result = await _send_openai(user, system_instructions, alias, example_messages, False)
    else:
        raise ValueError(f"unknown model alias {model_id!r}; choose from {ALL_MODELS}")
    _token_stats["prompt"] += count_tokens((system_instructions or "") + user)
    _token_stats["completion"] += count_tokens(result)
    _token_stats["calls"] += 1
    return result


async def send_batch_messages(requests: list[dict], poll_interval: int = 60) -> dict[str, str | None]:
    del poll_interval
    responses = await asyncio.gather(
        *(
            send_single_message(
                item["user_prompt"], item.get("system_instructions"), item["model_id"],
                item.get("backend", "vertex"), example_messages=item.get("example_messages")
            )
            for item in requests
        ),
        return_exceptions=True,
    )
    return {
        item["custom_id"]: None if isinstance(response, BaseException) else str(response)
        for item, response in zip(requests, responses)
    }


def safe_json_parse(response_text: str) -> Any:
    candidates = [response_text]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    enclosed = re.search(r"[\[{][\s\S]*[\]}]", response_text)
    if enclosed:
        candidates.append(enclosed.group(0))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("could not parse JSON response", response_text, 0)


class CsvWriter:
    def __init__(self, path: str, fieldnames: list[str], overwrite: bool = False):
        self.path, self.fieldnames = path, fieldnames
        self.lock = asyncio.Lock()
        if overwrite:
            open(path, "w", encoding="utf-8").close()
        self.write_header = not os.path.exists(path) or os.path.getsize(path) == 0

    async def write(self, row: dict) -> None:
        async with self.lock:
            with open(self.path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames, extrasaction="ignore", restval="")
                if self.write_header:
                    writer.writeheader()
                    self.write_header = False
                writer.writerow(row)


def load_completed_pairs(path: str, key_cols: list[str], nonempty_col: str | None = None) -> set[tuple]:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return set()
    import pandas as pd
    frame = pd.read_csv(path, dtype=str).fillna("")
    if nonempty_col:
        frame = frame[frame[nonempty_col].str.strip().ne("")]
    return {tuple(row[column] for column in key_cols) for _, row in frame.iterrows()}


def load_force_ids(ids: list[str] | None, ids_file: str | None) -> set[str]:
    output = {str(value).strip() for value in ids or [] if str(value).strip()}
    if ids_file:
        with open(ids_file, encoding="utf-8") as handle:
            output.update(line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#"))
    return output


def _normalize_model(value: object) -> str:
    return re.sub(r"_(batch|sandbox)$", "", str(value or ""))


def _drop_rows(path: str, mask: Any) -> int:
    import pandas as pd
    frame = pd.read_csv(path, dtype=str)
    selected = mask(frame)
    removed = int(selected.sum())
    if removed:
        temporary = f"{path}.tmp"
        frame.loc[~selected].to_csv(temporary, index=False)
        os.replace(temporary, path)
    return removed


def drop_rows_for_ids(path: str, id_col: str, force_ids: set[str]) -> int:
    if not force_ids or not os.path.isfile(path) or os.path.getsize(path) == 0:
        return 0
    return _drop_rows(path, lambda frame: frame[id_col].astype(str).isin(force_ids))


def drop_rows_for_id_models(path: str, id_col: str, force_ids: set[str], models: Iterable[str], model_col: str = "model") -> int:
    normalized = {_normalize_model(model) for model in models}
    if not force_ids or not normalized or not os.path.isfile(path) or os.path.getsize(path) == 0:
        return 0
    return _drop_rows(path, lambda frame: frame[id_col].astype(str).isin(force_ids) & frame[model_col].map(_normalize_model).isin(normalized))


def drop_rows_for_source_models(path: str, sources: Iterable[str], models: Iterable[str], source_col: str = "source_name", model_col: str = "model") -> int:
    source_set, model_set = set(sources), {_normalize_model(model) for model in models}
    if not source_set or not model_set or not os.path.isfile(path) or os.path.getsize(path) == 0:
        return 0
    return _drop_rows(path, lambda frame: frame[source_col].isin(source_set) & frame[model_col].map(_normalize_model).isin(model_set))


def load_rolling_progress(path: str) -> tuple[set[tuple], dict[tuple, tuple[int, str]]]:
    completed: set[tuple] = set()
    partial: dict[tuple, tuple[int, str]] = {}
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return completed, partial
    import pandas as pd
    frame = pd.read_csv(path, dtype=str).fillna("")
    for key, group in frame.groupby(["question_id", "model"]):
        good = group[group["response"].str.strip().ne("")].copy()
        if good.empty:
            continue
        good["batch_num"] = good["batch_num"].astype(int)
        last = good.sort_values("batch_num").iloc[-1]
        if int(last["batch_num"]) == int(last["total_batches"]) - 1:
            completed.add(tuple(map(str, key)))
        else:
            partial[tuple(map(str, key))] = (int(last["batch_num"]), str(last["response"]))
    return completed, partial


def parse_timestamp(value: object) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)


def filter_records_as_of(records: list[dict], query_timestamp: object, date_key: str = "note_date") -> list[dict]:
    """Exclude notes after the question time; undated notes are excluded by default."""
    cutoff = parse_timestamp(query_timestamp)
    if cutoff is None:
        raise ValueError(f"invalid query timestamp: {query_timestamp!r}")
    return [record for record in records if (stamp := parse_timestamp(record.get(date_key))) is not None and stamp <= cutoff]
