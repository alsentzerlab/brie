"""Compute unsupported-claim (hallucination) metrics from fact judgments."""

from __future__ import annotations

import argparse
import ast
import json
import math
from collections.abc import Iterable


def _list(value: object) -> list:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(str(value))
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return []


def score_claim_support(candidate_facts: Iterable[str], supported_items: Iterable[object]) -> dict[str, float | int]:
    facts = list(candidate_facts)
    supported: set[int] = set()
    normalized = {str(fact).strip().casefold(): index for index, fact in enumerate(facts)}
    for item in supported_items:
        try:
            index = int(item)
        except (TypeError, ValueError):
            index = normalized.get(str(item).strip().casefold(), -1)
        if 0 <= index < len(facts):
            supported.add(index)
    unsupported = len(facts) - len(supported)
    return {
        "claim_count": len(facts),
        "supported_claim_count": len(supported),
        "unsupported_claim_count": unsupported,
        "claim_precision": len(supported) / len(facts) if facts else 1.0,
        "hallucination_rate": unsupported / len(facts) if facts else 0.0,
        "has_hallucination": int(unsupported > 0),
    }


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="CSV containing candidate facts and supported indices")
    parser.add_argument("output", help="Destination CSV")
    parser.add_argument("--precision-column", default="consensus_precision")
    parser.add_argument("--facts-column")
    parser.add_argument("--supported-column")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if args.facts_column and args.supported_column:
        metrics = [
            score_claim_support(_list(row[args.facts_column]), _list(row[args.supported_column]))
            for _, row in frame.iterrows()
        ]
    else:
        metrics = [
            {
                "claim_precision": float(value),
                "hallucination_rate": 1.0 - float(value),
                "has_hallucination": int(float(value) < 1.0),
            }
            for value in frame[args.precision_column]
        ]
    pd.concat([frame, pd.DataFrame(metrics)], axis=1).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
