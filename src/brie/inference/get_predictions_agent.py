"""
Agent inference script for BRIE.

Workflow: query + patient_meta → planning → repeat up to N steps:
  [ search_notes / get_note / summarize_notes → reasoning → answer draft → decide to end ]

Only the tool call signature, reasoning, and answer draft are kept in context between
steps; raw tool outputs are discarded after each step.

Output CSV columns:
  question_id, model, approach, response, n_steps, tools_called
"""

import argparse
import asyncio
import json
import logging
import os
import traceback

import pandas as pd

from .utils import (
    ALL_MODELS,
    CsvWriter,
    load_completed_pairs,
    load_force_ids,
    drop_rows_for_id_models,
    log_token_stats,
    safe_json_parse,
    send_single_message,
    set_tpm_limit,
    filter_records_as_of,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS  = 300
DEFAULT_CONCURRENT = 3


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a clinical QA agent. Answer a clinical question about a specific patient by \
strategically retrieving and reading their medical notes.

Available tools:
- get_patient_meta() — note count, note types with counts, date range
- search_notes(keywords?, start?, end?, types?, match?) — returns matching note IDs
    keywords: list of strings matched case-insensitively against note title + text
    start/end: inclusive ISO date strings
    types: list of strings matched case-insensitively against note_title
    match: "any" (default) or "all" — whether items within keywords/types require any-one or all to match
    All filter axes (date, keywords, types) are ANDed together
- get_note(note_id) — full text of one note by integer ID
- summarize_notes(note_ids) — LLM summary of a batch of notes

At every step respond with JSON only:
{
  "reasoning": "...",
  "answer_draft": "...",
  "stop": true | false,
  "tool_call": {"name": "...", "args": {...}} | null
}
Set tool_call to null when stop is true. answer_draft is your best current answer."""

# Cached across all steps for one question (question + patient_meta are identical every call)
USER_PREFIX = """\
<question>{question}</question>
<query_date>{query_date}</query_date>
<patient_meta>
{patient_meta}
</patient_meta>"""

# Suffixes — sent uncached; only the unique content for each step
PLANNING_SUFFIX = """\

Plan your retrieval strategy. Identify which note types, timeframe(s), and keywords are \
likely relevant, and state what evidence you need to answer the question. \
Then choose your first tool call, or set stop to true if the metadata is sufficient.

Respond with JSON only."""

STEP_SUFFIX = """\

<history>
{history}
</history>

<tool_output>
{tool_output}
</tool_output>

Reason about what this result tells you. Update your answer draft. \
Then either call the next tool or set stop to true and emit your final answer.

Respond with JSON only."""

_SUMMARIZE_NOTES_PROMPT = """\
Summarize the following patient notes concisely, preserving key clinical facts, dates, \
medications, diagnoses, and procedures.

{notes}

Summary:"""

COMPACT_KEEP_RECENT = 10

_COMPACT_HISTORY_PROMPT = """\
Summarize the following agent reasoning steps concisely. Preserve key decisions, \
findings, tool calls made, and how the answer evolved. Omit redundant detail.

{prior_summary_section}\
Steps to summarize:
{entries}

Summary:"""

# ── Tool layer ─────────────────────────────────────────────────────────────────
# Records are list[dict] with keys: note_date, note_title, text
# note_id == list index

def get_patient_meta(records: list[dict]) -> str:
    if not records:
        return "No notes available."
    dates = sorted(r["note_date"] for r in records if r.get("note_date"))
    type_counts: dict[str, int] = {}
    for r in records:
        t = str(r.get("note_title", "unknown")).strip()
        type_counts[t] = type_counts.get(t, 0) + 1
    date_range = f"{dates[0]} to {dates[-1]}" if dates else "unknown"
    types_str = ", ".join(
        f"{t} ({n})" for t, n in sorted(type_counts.items(), key=lambda x: -x[1])
    )
    return f"n_notes={len(records)}, date_range={date_range}\nnote_types: {types_str}"


def search_notes(
    records: list[dict],
    keywords: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    types: list[str] | None = None,
    match: str = "any",
) -> list[int]:
    results = []
    require_all = match == "all"
    for idx, note in enumerate(records):
        date = str(note.get("note_date", ""))
        if start and date < start:
            continue
        if end and date > end:
            continue

        haystack = (str(note.get("note_title", "")) + " " + str(note.get("text", ""))).lower()
        title    = str(note.get("note_title", "")).lower()

        if keywords:
            hits = [kw.lower() in haystack for kw in keywords]
            if require_all and not all(hits):
                continue
            if not require_all and not any(hits):
                continue

        if types:
            hits = [t.lower() in title for t in types]
            if require_all and not all(hits):
                continue
            if not require_all and not any(hits):
                continue

        results.append(idx)
    return results


def get_note(records: list[dict], note_id: int) -> str:
    if note_id < 0 or note_id >= len(records):
        return f"Note ID {note_id} not found (corpus has {len(records)} notes)."
    note = records[note_id]
    return (
        f"Note ID: {note_id}\n"
        f"Note Title: {note.get('note_title', '')}\n"
        f"Note Date: {note.get('note_date', '')}\n"
        f"Text:\n{note.get('text', '')}"
    )


async def summarize_notes(
    records: list[dict], note_ids: list[int], model: str, backend: str
) -> str:
    notes_str = "\n\n---\n\n".join(get_note(records, nid) for nid in note_ids)
    return await send_single_message(
        user_prompt=_SUMMARIZE_NOTES_PROMPT.format(notes=notes_str),
        model_id=model,
        backend=backend,
    )


async def _compact_history(
    old_entries: list[dict],
    prior_summary: str,
    model: str,
    backend: str,
) -> str:
    """Summarize old history entries into a compact string."""
    prior_section = f"Prior summary:\n{prior_summary}\n\n" if prior_summary else ""
    entries_text = "\n\n".join(_format_history_entry(e) for e in old_entries)
    return await send_single_message(
        user_prompt=_COMPACT_HISTORY_PROMPT.format(
            prior_summary_section=prior_section,
            entries=entries_text,
        ),
        model_id=model,
        backend=backend,
    )


# ── Agent core ─────────────────────────────────────────────────────────────────

def _make_history_entry(tool_call: dict | None, resp: dict) -> dict:
    return {
        "tool_call":    tool_call,
        "reasoning":    resp.get("reasoning", ""),
        "answer_draft": resp.get("answer_draft", ""),
    }


def _format_history_entry(entry: dict) -> str:
    tc = entry["tool_call"]
    tool_str = "[planning]" if tc is None else f"{tc['name']}({json.dumps(tc.get('args', {}))})"
    return (
        f"Tool call:    {tool_str}\n"
        f"Reasoning:    {entry['reasoning']}\n"
        f"Answer draft: {entry['answer_draft']}"
    )


def _parse_response(text: str) -> dict | None:
    """Return parsed dict, or None if JSON parsing failed."""
    try:
        parsed = safe_json_parse(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return None


MAX_PARSE_RETRIES = 3


async def _send_and_parse(
    user_prompt: str,
    system_prompt: str,
    model: str,
    backend: str,
    user_prefix: str,
    question_id: str,
) -> dict:
    """Send an LLM request and retry up to MAX_PARSE_RETRIES times if JSON parsing fails."""
    last_raw = ""
    for attempt in range(MAX_PARSE_RETRIES):
        last_raw = await send_single_message(
            user_prompt, system_prompt, model, backend, cache_system=True, user_prefix=user_prefix,
        )
        resp = _parse_response(last_raw)
        if resp is not None:
            return resp
        logger.warning(
            f"question_id={question_id}: JSON parse failed (attempt {attempt + 1}/{MAX_PARSE_RETRIES}), retrying..."
        )
    logger.error(f"question_id={question_id}: parse failed after {MAX_PARSE_RETRIES} attempts, treating as stop")
    return {"reasoning": last_raw, "answer_draft": last_raw, "stop": True, "tool_call": None}


async def _execute_tool(
    tool_call: dict, records: list[dict], model: str, backend: str
) -> str:
    name = tool_call.get("name", "")
    args = tool_call.get("args", {})
    try:
        if name == "get_patient_meta":
            return get_patient_meta(records)
        elif name == "search_notes":
            note_ids = search_notes(records, **args)
            if not note_ids:
                return "No notes matched the search criteria."
            lines = [f"Matched {len(note_ids)} note(s):"]
            for nid in note_ids:
                note = records[nid]
                preview = note.get("text", "")[:100].replace("\n", " ")
                lines.append(
                    f"  [{nid}] {note.get('note_title', '').strip()} ({note.get('note_date', '')}) — {preview}"
                )
            return "\n".join(lines)
        elif name == "get_note":
            return get_note(records, int(args["note_id"]))
        elif name == "summarize_notes":
            note_ids = [int(i) for i in args.get("note_ids", [])]
            if not note_ids:
                return "No note IDs provided."
            return await summarize_notes(records, note_ids, model, backend)
        else:
            return f"Unknown tool: {name!r}"
    except Exception as e:
        logger.warning(f"Tool {name} raised: {e}")
        return f"Tool error ({name}): {e}"


def _is_stop(resp: dict) -> bool:
    v = resp.get("stop", False)
    if isinstance(v, str):
        return v.lower() == "true"
    return bool(v)


def _looks_like_json(text: str) -> bool:
    s = text.strip()
    return s.startswith("{") or s.startswith("[") or s.startswith("```json") or s.startswith("```")


async def run_agent(
    question_id: str,
    question: str,
    query_date: str,
    records: list[dict],
    model: str,
    backend: str,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict:
    records = filter_records_as_of(records, query_date)
    patient_meta = get_patient_meta(records)
    prefix       = USER_PREFIX.format(question=question, query_date=query_date, patient_meta=patient_meta)
    full_history:   list[dict] = []
    recent_history: list[dict] = []
    compacted_summary = ""

    # Step 0: planning — prefix is cached here; all subsequent steps get a cache hit
    resp  = await _send_and_parse(PLANNING_SUFFIX, SYSTEM_PROMPT, model, backend, prefix, question_id)
    entry = _make_history_entry(None, resp)
    full_history.append(entry)
    recent_history.append(entry)

    for _ in range(max_steps):
        if _is_stop(resp) or resp.get("tool_call") is None:
            break
        tool_call = resp["tool_call"]

        tool_output = await _execute_tool(tool_call, records, model, backend)

        # Compact older entries when recent_history exceeds the keep window
        if len(recent_history) > COMPACT_KEEP_RECENT:
            to_compact = recent_history[:-COMPACT_KEEP_RECENT]
            compacted_summary = await _compact_history(to_compact, compacted_summary, model, backend)
            recent_history = recent_history[-COMPACT_KEEP_RECENT:]

        history_text = ""
        if compacted_summary:
            history_text = f"[Summary of earlier steps]\n{compacted_summary}\n\n[Recent steps]\n"
        history_text += "\n\n".join(_format_history_entry(e) for e in recent_history)

        resp  = await _send_and_parse(
            STEP_SUFFIX.format(history=history_text, tool_output=tool_output),
            SYSTEM_PROMPT, model, backend, prefix, question_id,
        )
        entry = _make_history_entry(tool_call, resp)
        full_history.append(entry)
        recent_history.append(entry)

    tools_called = [e["tool_call"]["name"] for e in full_history if e["tool_call"] is not None]
    logger.info(
        f"question_id={question_id} model={model}: "
        f"{len(tools_called)} step(s), tools={tools_called}"
    )
    return {
        "question_id":  question_id,
        "model":        model,
        "approach":     "agent",
        "response":     resp.get("answer_draft", ""),
        "n_steps":      len(tools_called),
        "tools_called": json.dumps(tools_called),
        "trace":        json.dumps(full_history),
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_patient_records(notes_dir: str, patient_id: str) -> list[dict] | None:
    path = os.path.join(notes_dir, f"{patient_id}_subsetrecords.json")
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Notes file not found: {path}")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse notes for {patient_id}: {e}")
        return None


_OUTPUT_FIELDS = ["question_id", "model", "approach", "response", "n_steps", "tools_called", "trace"]


DEFAULT_TPM = 50_000


async def main(args):
    if args.backend == "openai":
        set_tpm_limit(DEFAULT_TPM)
        logger.info(f"TPM rate limiter enabled: {DEFAULT_TPM:,} tokens/min")

    df = pd.read_csv(args.questions)
    df = df[["question_id", args.question_column, "timestamp"]].drop_duplicates()
    logger.info(f"Loaded {len(df)} questions from {args.questions}")

    force_ids = load_force_ids(args.force_ids, args.force_ids_file)
    if force_ids:
        df = df[df["question_id"].astype(str).isin(force_ids)]
        removed = drop_rows_for_id_models(args.output, "question_id", force_ids, args.models)
        logger.info(
            f"--force-ids: {len(df)} question(s) kept for {len(force_ids)} forced id(s); "
            f"dropped {removed} stale row(s) from {args.output}"
        )

    completed = load_completed_pairs(args.output, ["question_id", "model"], nonempty_col="response")
    if completed and os.path.exists(args.output) and os.path.getsize(args.output) > 0:
        df_out = pd.read_csv(args.output, dtype=str)
        is_json = df_out["response"].notna() & df_out["response"].apply(lambda r: _looks_like_json(str(r)))
        good_pairs = set(
            zip(df_out.loc[~is_json, "question_id"].astype(str),
                df_out.loc[~is_json, "model"].astype(str))
        )
        to_rerun = set(
            zip(df_out.loc[is_json, "question_id"].astype(str),
                df_out.loc[is_json, "model"].astype(str))
        ) - good_pairs
        if to_rerun:
            for key in to_rerun:
                completed.discard(key)
            logger.info(f"Flagged {len(to_rerun)} JSON-looking response(s) as failed — will rerun")
    if completed:
        logger.info(f"Resuming: {len(completed)} (question_id, model) pairs already done")

    records_cache: dict[str, list[dict] | None] = {}
    for _, row in df.iterrows():
        patient_id = str(row["question_id"]).split("_")[0]
        if patient_id not in records_cache:
            records_cache[patient_id] = _load_patient_records(args.notes, patient_id)

    tasks = []
    for _, row in df.iterrows():
        patient_id = str(row["question_id"]).split("_")[0]
        if records_cache.get(patient_id) is None:
            logger.warning(f"Skipping question_id={row['question_id']} — notes unavailable.")
            continue
        for model in args.models:
            if (str(row["question_id"]), str(model)) in completed:
                logger.debug(f"Already done: question_id={row['question_id']} model={model}")
                continue
            safe_records = filter_records_as_of(records_cache[patient_id], row["timestamp"])
            tasks.append((row["question_id"], row[args.question_column], str(row["timestamp"]), safe_records, model))

    if not tasks:
        logger.info("All tasks already complete.")
        return

    logger.info(f"{len(tasks)} task(s) pending across {len(set(m for *_, m in tasks))} model(s)")
    if args.dry_run:
        for qid, _, _, _, model in tasks:
            logger.info(f"  [dry-run] would run: question_id={qid} model={model}")
        return

    writer    = CsvWriter(args.output, _OUTPUT_FIELDS)
    semaphore = asyncio.Semaphore(args.concurrent)

    async def _run(question_id, question, query_date, records, model):
        async with semaphore:
            try:
                result = await run_agent(
                    question_id, question, query_date, records, model, args.backend, args.max_steps
                )
                await writer.write(result)
                return result
            except Exception as e:
                logger.error(
                    f"question_id={question_id} model={model} failed: {type(e).__name__}: {e}\n"
                    + traceback.format_exc()
                )
                raise

    results = await asyncio.gather(
        *[_run(qid, q, ts, rec, m) for qid, q, ts, rec, m in tasks],
        return_exceptions=True,
    )

    failed = sum(1 for r in results if isinstance(r, BaseException))
    if failed:
        logger.warning(f"{failed} task(s) raised exceptions — see errors above.")

    logger.info("Done.")
    log_token_stats(logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent inference for BRIE")
    parser.add_argument("-q", "--questions", required=True,
                        help="CSV with question_id + question column")
    parser.add_argument("-n", "--notes", required=True,
                        help="Directory of {patient_id}_subsetrecords.json files")
    parser.add_argument("-o", "--output", required=True,
                        help="Output CSV path")
    parser.add_argument("-c", "--question-column", default="natural_query",
                        help="Question column name (default: natural_query)")
    parser.add_argument("--models", nargs="+", default=["claude_haiku_sandbox"], choices=ALL_MODELS,
                        metavar="MODEL",
                        help=f"Models to run (default: claude_haiku_sandbox). Choices: {', '.join(ALL_MODELS)}")
    parser.add_argument("--backend", choices=["vertex", "openai"], default="vertex")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help=f"Max tool-call steps per question (default: {DEFAULT_MAX_STEPS})")
    parser.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENT,
                        help=f"Max simultaneous agent runs (default: {DEFAULT_CONCURRENT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print pending tasks and exit without running the agent")
    parser.add_argument("--force-ids", nargs="+", default=None, metavar="QUESTION_ID",
                        help="Re-run only these question_ids, dropping their stale output rows first.")
    parser.add_argument("--force-ids-file", type=str, default=None,
                        help="File with one question_id per line to re-run (unioned with --force-ids).")
    args = parser.parse_args()
    asyncio.run(main(args))
