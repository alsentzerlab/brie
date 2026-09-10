#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 SUBJECT_IDS NOTES_DIR OUTPUT_DIR" >&2
  exit 2
fi

subject_ids=$1
notes_dir=$2
output_dir=$3

python -m brie.generation.extract_facts \
  --id "$subject_ids" --input "$notes_dir" --output "$output_dir/facts"
python -m brie.generation.generate_questions \
  --id "$subject_ids" --input "$output_dir/facts" --note "$notes_dir" \
  --output "$output_dir/questions"
python -m brie.generation.filter_questions \
  --id "$subject_ids" --input "$output_dir/questions" --note "$notes_dir" \
  --output "$output_dir/selected"
