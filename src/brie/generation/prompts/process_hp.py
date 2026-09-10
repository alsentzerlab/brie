"""Prompt for normalizing an admission summary."""

HP_SYS = """Produce a concise de-identified admission summary. Return only JSON
with string fields `reason_for_admission` and `clinical_summary`. Preserve
diagnoses, medications, relevant history, negation, uncertainty, and relative
timing. Omit names, identifiers, contact details, and locations."""
