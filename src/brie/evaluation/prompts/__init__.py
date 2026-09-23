"""Versioned prompt assets used by BRIE's production evaluation commands."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

import yaml


def load_prompt(name: str) -> dict[str, Any]:
    """Load a packaged evaluation prompt by filename."""
    resource = files(__package__).joinpath(name)
    prompt = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(prompt, dict):
        raise ValueError(f"evaluation prompt {name!r} must contain a YAML mapping")
    return prompt


FACT_ENTAILMENT = load_prompt("fact_entailment.yaml")
ELO_PAIRWISE = load_prompt("elo_pairwise.yaml")
