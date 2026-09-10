#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 QUESTIONS_CSV RESPONSES_CSV OUTPUT_CSV" >&2
  exit 2
fi

python -m brie.evaluation.score_predictions --questions "$1" --responses "$2" --output "$3"
