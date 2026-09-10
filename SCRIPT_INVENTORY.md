# Script inventory

The repository was assembled from the experiment code paths that create a
dataset, produce model answers, or evaluate those answers. Repair and migration
utilities were intentionally excluded.

## Dataset generation

- `generation/process_hp.py`: normalize admission summaries.
- `generation/extract_facts.py`: extract and deduplicate dated atomic facts.
- `generation/generate_questions.py`: create recent, past, and multi-event questions.
- `generation/filter_questions.py`: select useful, answerable, non-leaking questions.
- `generation/get_topics.py`: assign clinical topics.
- `generation/generate_answer.py` and `revise_answer.py`: create and refine references.
- `generation/get_factspans.py`: evidence/span annotation preparation.
- `generation/batch_utils.py`, `utils.py`, and `prompts/`: shared runtime code.

## Inference

- `inference/get_predictions.py`: full-context inference.
- `inference/get_predictions_rolling.py`: rolling-window inference.
- `inference/get_predictions_agent.py`: tool-using record navigation.
- `inference/get_predictions_rag_*.py`: BM25, dense, and late-interaction RAG.
- `inference/precompute_*_embeddings.py`: retrieval caches.
- `inference/rag_utils.py` and `utils.py`: retrieval/model adapters and temporal cutoffs.

## Answer evaluation and calibration

- `evaluation/score_predictions.py`: lexical, semantic, and LLM-judge metrics.
- `evaluation/atomize_facts_batch.py`, `score_facts_batch.py`, and
  `score_facts_subq.py`: answer atomization and bidirectional fact entailment.
- `evaluation/generate_rubrics.py` and `score_rubric_facts.py`: grounded rubric scoring.
- `evaluation/score_elo_batch.py`: pairwise judgments, win rates, and Elo inputs.
- `evaluation/hallucination/score.py`: unsupported-claim rate and answer-level flags.
- `evaluation/temporality/score.py`: future-note leakage and dated-fact consistency.
- `evaluation/fact_annotation/` and `elo_annotation/`: human agreement,
  adjudication, held-out prompt evaluation, and few-shot example construction.

“Fine-tuning” in this codebase means prompt and in-context-example calibration;
there is no weight-updating training pipeline. Prompt variants must be evaluated
on records held out from example construction.

## Excluded

Benchmark repairs, index repairs, one-off reruns, data migrations, cohort SQL,
data files, outputs, logs, cached embeddings, credentials, and infrastructure
launchers are not part of this repository.

Every retained executable is listed above or is a helper used by one of these
families. Empty HTML annotation applications and the current YAML entailment
prompt are retained because the human-calibration scripts load them at runtime.
