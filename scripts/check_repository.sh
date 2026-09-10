#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

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

if find . \( -path ./.git -o -path ./.venv -o -path ./.pytest_cache -o -path ./.ruff_cache \) -prune -o -type f \( \
  -name '*.csv' -o -name '*.json' -o -name '*.jsonl' -o -name '*.parquet' -o \
  -name '*.pkl' -o -name '*.npy' -o -name '*.log' \) -print | grep -q .; then
  echo "error: repository contains a data or output artifact" >&2
  exit 1
fi

if grep -RIE 'https?://|/home/|/projects/|/data/|/Users/|arn:aws|medical record number' \
  --exclude-dir=.git --exclude-dir=.venv --exclude-dir=.pytest_cache --exclude-dir=.ruff_cache \
  --exclude=uv.lock --exclude=check_repository.sh .; then
  echo "error: repository contains a prohibited path, endpoint, or identifier" >&2
  exit 1
fi

if grep -RIE 'AKIA[[:alnum:]]{16}|-----BEGIN .*PRIVATE KEY-----|[0-9]{8,}' \
  --exclude-dir=.git --exclude-dir=.venv --exclude-dir=.pytest_cache --exclude-dir=.ruff_cache \
  --exclude=uv.lock --exclude=check_repository.sh .; then
  echo "error: repository contains a possible credential or long identifier" >&2
  exit 1
fi

echo "repository checks passed"
