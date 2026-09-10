#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 QUESTIONS_CSV NOTES_DIR OUTPUT_CSV MODEL [MODEL ...]" >&2
  exit 2
fi

questions=$1
notes=$2
output=$3
shift 3

python -m brie.inference.get_predictions \
  --questions "$questions" --notes "$notes" --output "$output" --models "$@"
