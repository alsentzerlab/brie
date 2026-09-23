#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
export MPLCONFIGDIR
MPLCONFIGDIR=$(mktemp -d)

python -c 'from pathlib import Path; files=list(Path("src").rglob("*.py"))+list(Path("tests").rglob("*.py")); [compile(path.read_text(encoding="utf-8"), str(path), "exec") for path in files]; print(f"parsed {len(files)} Python files")'
bash -n scripts/*.sh
uv lock --check
ruff check src tests
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider

mapfile -t command_files < <(grep -Rl '__name__.*__main__' src/brie --include='*.py' | sort)
for file in "${command_files[@]}"; do
  module=${file#src/}
  module=${module%.py}
  module=${module//\//.}
  PYTHONDONTWRITEBYTECODE=1 python -m "$module" --help >/dev/null
done
echo "checked ${#command_files[@]} command modules"

if git ls-files | grep -Eq '\.(csv|json|jsonl|parquet|pkl|pickle|npy|npz|pt|bin|log)$'; then
  echo "error: repository contains a data or output artifact" >&2
  exit 1
fi

if git grep -nEI \
  'https?://|/home/|/projects/|/data/|/Users/|arn:aws|medical record number|stanford|alsentzer|cahoon|ema2016|som-nero|shahlab|starr' \
  -- ':!uv.lock' ':!scripts/check_repository.sh'; then
  echo "error: repository contains a prohibited path, endpoint, or identifier" >&2
  exit 1
fi

if git grep -nEI \
  'AKIA[[:alnum:]]{16}|AIza[[:alnum:]_-]{20,}|gh[pousr]_[[:alnum:]]{20,}|xox[baprs]-[[:alnum:]-]{10,}|sk-[[:alnum:]_-]{20,}|-----BEGIN .*PRIVATE KEY-----|[[:alnum:]._%+-]+@[[:alnum:].-]+\.[[:alpha:]]{2,}|[0-9]{8,}' \
  -- ':!uv.lock' ':!scripts/check_repository.sh'; then
  echo "error: repository contains a possible credential or long identifier" >&2
  exit 1
fi

echo "repository checks passed"
