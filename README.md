# BRIE

BRIE contains code for building longitudinal clinical question-answer datasets,
running full-context and retrieval-based inference, and evaluating answers. It
contains no clinical data, credentials, institution-specific endpoints, or
precomputed results.

## Safety boundary

Only use this code in an environment approved for the data being processed.
Keep notes, questions, model outputs, annotations, caches, and logs outside the
repository. The included `.gitignore` blocks common data and artifact formats,
but it is not a substitute for an approved data-handling process.

## Installation

```bash
uv sync --locked --all-extras
. .venv/bin/activate
```

Provider credentials and infrastructure values must be supplied through the
environment. No organization-specific project, bucket, endpoint, model path,
or credential has a repository default. Public provider model aliases have
documented defaults and can be overridden for exact reproduction.

## Workflows

- `brie.generation`: note preparation, fact extraction, question generation,
  filtering, topic tagging, answer generation, and evidence extraction.
- `brie.inference`: full-context, rolling, agentic, BM25, dense, and
  late-interaction inference.
- `brie.evaluation`: traditional metrics, atomic-fact entailment, rubric and
  hallucination evaluation, pairwise win-rate/Elo, human calibration, and
  temporality evaluation.

Generic shell entry points are under `scripts/`. All input and output paths are
arguments; the scripts contain no cohort-specific configuration.

See [DATA_FORMAT.md](DATA_FORMAT.md) for the BRIE file contract, provider
configuration, and runnable commands for inference, evaluation, hallucination,
temporality, win rate/Elo, and human-calibrated in-context tuning.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the minimal reproduction path,
metric definitions, required run metadata, and repository self-check.
