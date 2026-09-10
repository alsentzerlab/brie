# BRIE data contract and run guide

BRIE data is not distributed with this repository. Keep the dataset in an
approved location outside the checkout and pass its location on the command
line. The examples below use `$BRIE_DATA` and `$BRIE_RUN`; neither variable has
a repository default.

All identifiers must be de-identified strings. Never place source-system
identifiers, names, contact details, or dates of birth in these fields.

## Canonical layout

```text
$BRIE_DATA/
  questions.csv
  notes/
    <subject_id>_subsetrecords.json

$BRIE_RUN/
  responses.csv
  scores.csv
  pairwise.csv
  elo.csv
```

The output directory is deliberately external because model responses and
evaluation artifacts may contain sensitive record content.

## Notes

Each notes file is a UTF-8 JSON array. Its filename is
`<subject_id>_subsetrecords.json`. Each object requires:

| Field | Type | Meaning |
| --- | --- | --- |
| `note_date` | ISO-8601 string | Note timestamp. |
| `note_title` | string | De-identified note type or title. |
| `text` | string | Note text. |

Optional metadata fields are preserved by retrieval code. Sort records newest
first for full-context truncation. Every inference path independently removes
records with `note_date` after the question timestamp; undated records are also
excluded.

The current launchers resolve a notes filename from the first underscore-
delimited portion of `question_id`. Therefore use subject IDs without
underscores and encode question IDs as `<subject_id>_<question_index>`.

## Questions and references

`questions.csv` requires:

| Field | Type | Meaning |
| --- | --- | --- |
| `question_id` | string | Unique de-identified question key. |
| `natural_query` | string | Question shown to the model. |
| `timestamp` | ISO-8601 string | Information cutoff for the question. |
| `annotation_sub_answer` | string | Reference answer used for scoring. |

Optional fields used by specific evaluators are `sub_question_id`, `question`
(an alias for `natural_query`), `answer` (an alias for
`annotation_sub_answer`), `facts_edited` (a JSON/Python list of reference
facts), and `question_type` (`recent`, `past`, or `multi`). When
`sub_question_id` is absent, answer scoring creates `<question_id>_r`.

## Model responses

Every inference command writes CSV with at least:

| Field | Type | Meaning |
| --- | --- | --- |
| `question_id` | string | Key matching `questions.csv`. |
| `model` | string | Runtime model alias. |
| `approach` | string | Full context, rolling, agent, or RAG variant. |
| `response` | string | Generated answer. |

RAG outputs also contain `retrieved_chunks`, a JSON list of objects with
`chunk_id`, `note_title`, `note_date`, `text`, and `score`. Rolling inference
adds `batch_num` and `total_batches`; agent inference adds `n_steps`,
`tools_called`, and `trace`.

## Provider configuration

No provider values are committed. Set only those needed by the selected path:

- Vertex: `BRIE_VERTEX_PROJECT`, optionally `BRIE_VERTEX_GEMINI_PROJECT`,
  `BRIE_VERTEX_LOCATION`, and `BRIE_VERTEX_GCS_BUCKET` for batch jobs.
- Provider model overrides: `BRIE_GEMINI_PRO_MODEL`,
  `BRIE_GEMINI_FLASH_MODEL`, `BRIE_GEMINI_JUDGE_MODEL`,
  `BRIE_CLAUDE_OPUS_MODEL`, `BRIE_CLAUDE_SONNET_MODEL`, and
  `BRIE_CLAUDE_HAIKU_MODEL`.
- OpenAI-compatible service: `BRIE_OPENAI_BASE_URL`, `BRIE_OPENAI_API_KEY`,
  `BRIE_OPENAI_MODEL`, and optionally comma-separated `BRIE_OPENAI_MODELS`.
- Local OpenAI-compatible service: `BRIE_LOCAL_BASE_URL`,
  `BRIE_LOCAL_API_KEY`, `BRIE_LOCAL_MODEL`, and optionally
  `BRIE_LOCAL_MODELS`.
- Dense retrieval: `BRIE_EMBEDDING_MODEL`.
- Late-interaction retrieval: `BRIE_LATE_INTERACTION_MODEL`.
- Optional retrieval reranker: `BRIE_RERANKER_MODEL`.
- BERTScore: `BRIE_BERTSCORE_MODEL` and `BRIE_BERTSCORE_LAYERS`.

Use runtime secret injection; do not store environment files in the repository.

## Validate external inputs

```bash
python -m brie.validate \
  --questions "$BRIE_DATA/questions.csv" \
  --notes "$BRIE_DATA/notes" \
  --require-references
```

## Inference

Install the package, create an external run directory, and run full-context
inference:

```bash
uv sync --locked --all-extras
mkdir -p "$BRIE_RUN"
python -m brie.inference.get_predictions \
  --questions "$BRIE_DATA/questions.csv" \
  --notes "$BRIE_DATA/notes" \
  --output "$BRIE_RUN/responses.csv" \
  --models gemini_flash
```

Inference entry points use these additional requirements:

| Module | Additional arguments or setup |
| --- | --- |
| `get_predictions` | Optional `--context-limit`. |
| `get_predictions_rolling` | No additional required arguments. |
| `get_predictions_agent` | Optional `--max-steps` and `--concurrent`. |
| `get_predictions_rag_bm25` | Optional `--top-k`. |
| `get_predictions_rag_embedding` | Required `--embeddings-dir`; set `BRIE_EMBEDDING_MODEL`. |
| `get_predictions_rag_late` | Required `--embeddings-dir`; set `BRIE_LATE_INTERACTION_MODEL`. |

All use `--questions`, `--notes`, `--output`, and `--models`. Run
`python -m brie.inference.<module> --help` before launching an experiment.

## Dataset-construction inputs and outputs

Dataset construction is optional when an existing BRIE release is available.
It requires both the canonical note files described above and one admission
summary file per subject named `<subject_id>_hp.json`. That file is a JSON table
with columns `type` and `text`; exactly one row has `type` equal to `Full Note`,
and its `text` value is an object containing `reference_timestamp`,
`original_timestamp`, `note_title`, and `text`.

The core construction sequence is:

```bash
bash scripts/run_generation.sh \
  '<comma-separated-subject-ids>' \
  "$BRIE_DATA/notes" \
  "$BRIE_RUN/generation"
```

The notes directory supplied here must contain both file families. The stages
produce:

| Stage | Output contract |
| --- | --- |
| `extract_facts` | `<subject_id>_raw.json` checkpoints and `<subject_id>.tsv` with `index`, `fact`. |
| `generate_questions` | `<subject_id>.csv` candidate items containing question, answer, evidence-fact, type, rationale, iteration, and cutoff fields. |
| `filter_questions` | `questions_filtered.csv`; its `question_id`, `natural_query`, `timestamp`, `answer`, and `facts` columns are compatible with downstream tools. |
| `get_topics` | Input rows plus serialized `question_topics`. |

`generate_answer` is a separate augmentation path. Its input requires
`question_id`, `question`, `original_question`, `answer`, and `facts`; it reads
per-subject fact TSVs and writes `answers.csv` under `--output`. It also requires
external `--log` and `--checkpoint` directories. `revise_answer` expects
`question_id`, `timestamp`, `question`, `answer`, and `comment` and writes
per-subject revised CSVs. `get_factspans` expects generated and reference answer
columns plus serialized fact lists and writes one JSON annotation export.

## General answer metrics

```bash
python -m brie.evaluation.score_predictions \
  --questions "$BRIE_DATA/questions.csv" \
  --responses "$BRIE_RUN/responses.csv" \
  --output "$BRIE_RUN/scores.csv" \
  --metrics rouge,bleu,bert,jury
```

The available metrics include lexical overlap, BERTScore, fact-level variants,
multi-model jury completeness/relevance/faithfulness, and grounded rubric
scores. Jury faithfulness is an answer-level hallucination assessment.

## Atomic facts, entailment, and hallucination

Atomic-fact extraction uses a YAML runtime config kept outside the repository:

```yaml
gcs_location: <cloud-output-prefix>
sources:
  - source_name: reference
    path: <questions-csv>
    text_column: annotation_sub_answer
    facts_column: facts_edited
  - source_name: candidate
    path: <responses-csv>
    text_column: response
```

Run atomization and the bidirectional three-juror entailment scorer:

```bash
python -m brie.evaluation.atomize_facts_batch \
  --config "$BRIE_RUN/atomize.yaml" \
  --output "$BRIE_RUN/atomic_facts.csv"

python -m brie.evaluation.score_facts_batch \
  --facts "$BRIE_RUN/atomic_facts.csv" \
  --gcs-location '<cloud-output-prefix>' \
  --output "$BRIE_RUN/fact_scores.csv"
```

`atomic_facts.csv` contains `source_name`, `question_id`, optional `model`, and
`facts_atomic` (a serialized list). The scorer reports precision as candidate
facts supported by the reference, recall as reference facts covered by the
candidate, each juror's entailed facts, macro averages, and majority consensus.

Hallucination rate is `1 - fact precision`:

```bash
python -m brie.evaluation.hallucination.score \
  "$BRIE_RUN/fact_scores.csv" "$BRIE_RUN/hallucination.csv"
```

For a custom file containing all candidate facts and supported facts or their
indices, pass `--facts-column` and `--supported-column`.

## Temporality and leakage

Audit a RAG output by joining its question timestamps and checking every
retrieved chunk:

```bash
python -m brie.evaluation.temporality.score \
  "$BRIE_RUN/responses.csv" "$BRIE_RUN/temporality.csv" \
  --questions "$BRIE_DATA/questions.csv"
```

The output includes future-note and undated-note counts, leakage rate, and a
row-level leakage flag. If candidate/reference date lists are available, add
`--reference-dates-column` and `--candidate-dates-column` to compute date
precision and recall. Aggregate these metrics by `question_type` to compare
recent, past, and multi-event questions.

## Pairwise win rate and Elo

At least two model responses per question are required:

```bash
python -m brie.evaluation.score_elo_batch \
  --questions "$BRIE_DATA/questions.csv" \
  --responses "$BRIE_RUN/responses.csv" \
  --output "$BRIE_RUN/pairwise.csv" \
  --elo "$BRIE_RUN/elo.csv" \
  --judge jury \
  --gcs-location '<cloud-output-prefix>'
```

Each unordered model pair is judged in both A/B positions. The pairwise output
stores dimension winners; the summary contains wins, losses, ties, win rate,
non-tie win rate, and Elo ratings computed from deterministically shuffled
judgments with the configured K-factor and seed.

## Human calibration and in-context tuning

The `fact_annotation` and `elo_annotation` packages sample blinded comparisons,
merge exports, calculate agreement and leniency, compare model judges with
human labels, and build few-shot message files. Pass those files through
`--examples` to the entailment or Elo scorer, then evaluate on held-out record
IDs with `python -m brie.evaluation.fact_annotation.eval_entailment_prompt
--exclude-records ...`.

This is prompt and in-context-example tuning only. No module updates model
weights, and annotation exports or generated few-shot files must remain outside
the repository.
