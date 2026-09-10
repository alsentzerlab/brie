# Reproducibility guide

This repository reproduces the BRIE processing and evaluation methods when the
external BRIE data, provider access, and model artifacts are available. It does
not include data or previously generated outputs.

## What must be recorded for a run

Exact regeneration requires more than code. Record these values beside every
external run directory:

- Git commit or staged-tree checksum.
- Python version and installed dependency versions.
- Model alias and exact provider model identifier or local model revision.
- Prompt file and optional few-shot example file checksum.
- Input question, note, response, fact, and rubric file checksums.
- Inference approach, command-line arguments, context limit, retrieval depth,
  random seed, and generation parameters.
- Provider region and batch-job identifiers needed to retrieve raw responses.

Model APIs can be nondeterministic or silently revised by a provider. Retaining
raw model responses and the exact provider model revision is therefore required
to reproduce numeric results exactly. Keep those artifacts outside this Git
repository under the applicable data controls.

## Environment

Python 3.12 is the reference runtime. Direct dependencies are exactly pinned in
`pyproject.toml`, and `uv.lock` pins the complete dependency graph. Install the
locked optional groups required by the experiment:

```bash
uv sync --locked --all-extras
. .venv/bin/activate
```

Before using data, validate it:

```bash
python -m brie.validate \
  --questions "$BRIE_DATA/questions.csv" \
  --notes "$BRIE_DATA/notes" \
  --require-references
```

After inference, add `--responses "$BRIE_RUN/responses.csv"` to validate the
output schema too.

## Reproduction order

For an existing BRIE release, dataset generation is not rerun. The minimum
result-reproduction path is:

1. Validate external questions and notes.
2. Run one or more inference approaches with the exact recorded model IDs.
3. Run general metrics and/or atomic-fact entailment.
4. Derive unsupported-claim rates from fact precision.
5. Audit retrieved-note timestamps for temporal leakage.
6. Run two-position pairwise judging, then compute win rates and Elo.
7. If using calibrated prompts, build examples only from the development split
   and evaluate them on held-out record IDs.

Detailed commands and schemas are in `DATA_FORMAT.md`.

## Dataset-construction path

Only use this path when rebuilding a dataset release:

1. `extract_facts.py` converts dated note records to atomic-fact TSV files.
2. `generate_questions.py` creates candidate QA items from facts and admission
   summaries.
3. `filter_questions.py` produces the canonical `questions_filtered.csv`.
4. `get_topics.py` adds topic labels.
5. `generate_answer.py` creates derived QA variants; `revise_answer.py` applies
   reviewer-guided corrections.
6. `get_factspans.py` maps answer facts to source-note spans.

`process_hp.py` is retained because it produces the normalized admission
summary table used during dataset analysis, but question generation reads the
original de-identified admission-summary JSON files.

## Evaluation definitions

- Fact recall: fraction of reference atomic facts entailed by candidate facts.
- Fact precision: fraction of candidate atomic facts entailed by reference facts.
- Hallucination rate: `1 - fact precision`.
- Win rate: wins divided by all pairwise judgments, with ties retained in the
  denominator.
- Non-tie win rate: wins divided by wins plus losses.
- Elo: sequential Elo updates after deterministic row shuffling with the
  configured seed and K-factor.
- Temporal leakage: any retrieved note whose date is after the question cutoff.
- Date precision/recall: overlap of unique candidate and reference event dates.

Every unordered answer pair is judged in both display orders. Stored
`model_a`/`model_b` always match displayed response A/B; the Elo implementation
therefore maps winner labels directly and does not flip the reversed row again.

## Human calibration

The HTML files under `evaluation/*_annotation/` are empty annotation
applications. Sampling scripts inject external records at runtime. Merge and
agreement scripts consume exported annotations outside the repository.

`build_fact_examples.py` and `build_elo_examples.py` create few-shot message
files. Pass those through `--examples`. The current de-identified entailment
prompt is `fact_annotation/prompt_versions/current.yaml`. This is prompt/ICL
calibration; there is no weight-updating training code.

## Repository self-check

Run before publishing or executing experiments:

```bash
bash scripts/check_repository.sh
```

The check validates the lockfile, parses and lints every Python and Bash file,
runs the tests, invokes `--help` for every command module, rejects committed
data/output extensions, and searches for common institutional path, endpoint,
credential, and identifier patterns.
