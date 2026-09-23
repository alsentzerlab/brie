# BRIE

BRIE contains the code used to build longitudinal clinical question-answer
datasets, generate model responses with full-context and retrieval-based
methods, and evaluate those responses. The repository contains code only: it
does not include clinical records, questions, annotations, model outputs,
credentials, or precomputed results.

## Repository contents

- `brie.generation`: fact extraction, question generation and filtering,
  topic assignment, reference-answer generation and revision, and evidence
  span extraction.
- `brie.inference`: full-context, rolling-context, agentic, BM25, dense, and
  late-interaction response generation.
- `brie.evaluation`: traditional metrics, atomic-fact entailment,
  hallucination and rubric evaluation, temporal-leakage checks, pairwise win
  rates and Elo, and human/ICL calibration utilities.
- `scripts/`: generic entry points for the three top-level workflows and the
  repository audit.

Benchmark-repair, one-off rerun, migration, and cohort-specific scripts are
intentionally excluded. See [SCRIPT_INVENTORY.md](SCRIPT_INVENTORY.md) for the
retained and excluded source families.

## Installation

BRIE requires Python 3.10 or newer and uses `uv` for a locked environment.
Python 3.12 is the reference runtime.

```bash
git clone <repository-url>
cd brie
uv sync --locked --all-extras
uv run python -m brie.validate --help
```

Use `--all-extras` to reproduce every workflow. Smaller installations can use
`--extra retrieval` for dense/late-interaction retrieval, `--extra metrics`
for optional text metrics, or `--extra dev` for tests and linting.

Provider credentials and infrastructure settings are supplied at runtime.
Common variables include:

```bash
export BRIE_VERTEX_PROJECT='<project>'
export BRIE_VERTEX_LOCATION='<region>'
export BRIE_VERTEX_GCS_BUCKET='<bucket>'

# Or an OpenAI-compatible endpoint:
export BRIE_OPENAI_BASE_URL='<endpoint>'
export BRIE_OPENAI_API_KEY='<secret>'
export BRIE_OPENAI_MODEL='<model>'
```

The complete provider-variable list is in
[DATA_FORMAT.md](DATA_FORMAT.md#provider-configuration). Do not place secrets
in this checkout.

## External data layout

Keep input data and generated artifacts outside the repository:

```text
$BRIE_DATA/
  questions.csv
  notes/
    <subject_id>_subsetrecords.json

$BRIE_RUN/
  responses.csv
  scores.csv
  atomic_facts.csv
  fact_scores.csv
  pairwise.csv
  elo.csv
```

Set those locations and validate the inputs before running a model:

```bash
export BRIE_DATA='/approved/location/brie-data'
export BRIE_RUN='/approved/location/brie-run'
mkdir -p "$BRIE_RUN"

uv run python -m brie.validate \
  --questions "$BRIE_DATA/questions.csv" \
  --notes "$BRIE_DATA/notes" \
  --require-references
```

The code treats record identifiers as opaque strings and does not perform
de-identification. Use it only in an environment approved for the input data,
and keep inputs, responses, caches, logs, and evaluation outputs under the
applicable data controls. Exact schemas are documented in
[DATA_FORMAT.md](DATA_FORMAT.md).

## 1. Generate a dataset

Skip this section when using an existing BRIE release. Dataset construction
expects the canonical note files plus one admission-summary file per subject,
as described under
[dataset-construction inputs](DATA_FORMAT.md#dataset-construction-inputs-and-outputs).

Run fact extraction, question generation, and question filtering:

```bash
uv run bash scripts/run_generation.sh \
  '<comma-separated-subject-ids>' \
  "$BRIE_DATA/notes" \
  "$BRIE_RUN/generation"
```

The main output is
`$BRIE_RUN/generation/selected/questions_filtered.csv`. Optional later stages
can add topics, generate or revise reference answers, and extract evidence
spans:

```bash
uv run python -m brie.generation.get_topics --help
uv run python -m brie.generation.generate_answer --help
uv run python -m brie.generation.revise_answer --help
uv run python -m brie.generation.get_factspans --help
```

Each module exposes its complete input contract with `--help`; the generated
files and required columns are listed in
[DATA_FORMAT.md](DATA_FORMAT.md#dataset-construction-inputs-and-outputs).

## 2. Generate model responses

The generic launcher runs full-context inference:

```bash
uv run bash scripts/run_inference.sh \
  "$BRIE_DATA/questions.csv" \
  "$BRIE_DATA/notes" \
  "$BRIE_RUN/responses.csv" \
  gemini_flash
```

The same input contract supports the other paper inference approaches:

```bash
# Rolling context
uv run python -m brie.inference.get_predictions_rolling \
  --questions "$BRIE_DATA/questions.csv" \
  --notes "$BRIE_DATA/notes" \
  --output "$BRIE_RUN/responses_rolling.csv" \
  --models gemini_flash

# BM25 retrieval
uv run python -m brie.inference.get_predictions_rag_bm25 \
  --questions "$BRIE_DATA/questions.csv" \
  --notes "$BRIE_DATA/notes" \
  --output "$BRIE_RUN/responses_bm25.csv" \
  --models gemini_flash
```

Agentic, dense-retrieval, and late-interaction commands are
`brie.inference.get_predictions_agent`,
`brie.inference.get_predictions_rag_embedding`, and
`brie.inference.get_predictions_rag_late`. Dense and late-interaction runs
require their corresponding precomputation command and an external embedding
directory. All inference paths enforce the question timestamp cutoff before
constructing model context.

## 3. Evaluate responses

Run general answer metrics:

```bash
uv run bash scripts/run_evaluation.sh \
  "$BRIE_DATA/questions.csv" \
  "$BRIE_RUN/responses.csv" \
  "$BRIE_RUN/scores.csv"
```

For atomic-fact recall, precision, and hallucination evaluation, create an
external atomization configuration and run:

```bash
uv run python -m brie.evaluation.atomize_facts_batch \
  --config "$BRIE_RUN/atomize.yaml" \
  --output "$BRIE_RUN/atomic_facts.csv"

uv run python -m brie.evaluation.score_facts_batch \
  --facts "$BRIE_RUN/atomic_facts.csv" \
  --gcs-location '<cloud-output-prefix>' \
  --output "$BRIE_RUN/fact_scores.csv"

uv run python -m brie.evaluation.hallucination.score \
  "$BRIE_RUN/fact_scores.csv" \
  "$BRIE_RUN/hallucination.csv"
```

Audit retrieval for future-note leakage:

```bash
uv run python -m brie.evaluation.temporality.score \
  "$BRIE_RUN/responses_bm25.csv" \
  "$BRIE_RUN/temporality.csv" \
  --questions "$BRIE_DATA/questions.csv"
```

Compute two-position pairwise judgments, win rates, and Elo for a response file
containing at least two models per question:

```bash
uv run python -m brie.evaluation.score_elo_batch \
  --questions "$BRIE_DATA/questions.csv" \
  --responses "$BRIE_RUN/responses.csv" \
  --output "$BRIE_RUN/pairwise.csv" \
  --elo "$BRIE_RUN/elo.csv" \
  --judge jury \
  --gcs-location '<cloud-output-prefix>'
```

Detailed configuration, output schemas, rubric scoring, and human-calibration
commands are in [DATA_FORMAT.md](DATA_FORMAT.md). Metric definitions and the
exact-reproduction checklist are in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Verification

Before using or publishing a change, run:

```bash
uv run bash scripts/check_repository.sh
```

This validates the lockfile, parses and lints Python and Bash files, runs the
tests, loads every retained CLI, rejects data/output artifacts, and scans for
embedded credentials, internal paths, endpoints, and long identifiers.
