"""Prompts for extracting dated atomic facts from longitudinal notes."""

EXTRACT_SYS = """Extract atomic, clinically meaningful facts from the note.
Return JSON with a single key, `facts`, whose value is a list of strings.
Each fact must stand alone, preserve uncertainty and negation, avoid names and
identifiers, and end with the supplied note date in parentheses. Do not infer
facts that are not stated in the note."""

DEDUP_SYS = """Identify facts that add no information beyond another fact.
Return JSON with one key, `remove`, containing the zero-based indices to remove.
Keep distinct dates, changes in state, uncertainty, and contradictions."""


def format_extract(note_date: str, text: str) -> str:
    return f"<note_date>{note_date}</note_date>\n<note>{text}</note>"


def format_dedup(input_fact_list: list[str]) -> str:
    numbered = "\n".join(f"{index}: {fact}" for index, fact in enumerate(input_fact_list))
    return f"<facts>\n{numbered}\n</facts>"
