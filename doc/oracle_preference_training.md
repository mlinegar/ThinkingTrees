# Oracle Preference Training and GenRM Judging for OPS

This document describes the current preference-learning pipeline in ThinkingTrees for Oracle-Preserving Summarization (OPS). It covers the OPS laws, how preference pairs are produced (GenRM or numeric oracle), how a comparison module is trained, and how DPO-style training data is generated. It is a standalone guide to the implemented code.

## Contents

1. Overview
2. OPS laws and comparison types
3. Preference data schema
4. GenRM judging (default)
5. Preference collection pipelines
6. Training the comparison module (GEPA)
7. DPO data generation (sufficiency only)
8. Configuration and temperatures
9. End-to-end usage examples
10. Extending to new oracles or human judges
11. Code map

---

## Overview

The core idea is to distill oracle-relevant judgments into pairwise preferences and then use those preferences to train smaller models. The pipeline is:

1. Generate diverse candidate summaries.
2. Judge candidate pairs under an OPS law using GenRM (default) or a numeric oracle (optional).
3. Store those judgments as preference pairs.
4. Train a comparison module that predicts preferences.
5. Use the trained comparison module to generate DPO data for the summarizer.

GenRM is the default judge for all comparisons. Numeric-oracle preference collection is available for ground-truth anchoring and sanity checks.
When a task defines a scale, numeric-oracle errors are normalized to 0-1 for optimization; raw task scores are preserved for reporting.

---

## OPS laws and comparison types

OPS uses three law types. Each law is implemented as a distinct comparison path in both GenRM and oracle-based collectors.

### Sufficiency (C1)

Goal: a summary should preserve the oracle-relevant information of the original.

Numeric oracle criterion:

```
error_a = |oracle(summary_a) - oracle(original)| / scale.range
error_b = |oracle(summary_b) - oracle(original)| / scale.range
preferred = A if error_a < error_b - tie_margin
preferred = B if error_b < error_a - tie_margin
preferred = tie otherwise
```

### Idempotence (C2)

Goal: re-summarizing a summary should not change its oracle score.

Numeric oracle criterion:

```
re_a = summarize(summary_a)
re_b = summarize(summary_b)
drift_a = |oracle(re_a) - oracle(summary_a)| / scale.range
drift_b = |oracle(re_b) - oracle(summary_b)| / scale.range
preferred = A if drift_a < drift_b - tie_margin
preferred = B if drift_b < drift_a - tie_margin
preferred = tie otherwise
```

GenRM criterion:
- Compare the original summaries, while providing both re-summaries as extra context.
- Original text is optional and omitted for idempotence in GenRM prompts.

### Merge consistency (C3B)

Goal: merging two summaries should preserve their combined information.

Numeric oracle criterion:

```
left = summarize(first_half)
right = summarize(second_half)
merge_a = summarize(left + right)  # candidate A
merge_b = summarize(left' + right') # candidate B
error_a = |oracle(merge_a) - oracle(original)| / scale.range
error_b = |oracle(merge_b) - oracle(original)| / scale.range
preferred = A if error_a < error_b - tie_margin
preferred = B if error_b < error_a - tie_margin
preferred = tie otherwise
```

GenRM criterion:
- Compare merged summaries.
- Provide the child summaries (left/right) as extra context.
- Original text is optional and omitted for merge in GenRM prompts.

### Tie policy

- Numeric oracle: tie if `abs(error_a - error_b) < tie_margin` (default tie margin is 0.05 normalized; raw margin is `tie_margin * scale.range`).
- GenRM: treat ranking score `3` as a tie on the 1-6 scale.

---

## Preference data schema

Preferences are stored as `PreferencePair` objects in `src/ops_engine/training_framework/preference.py`.

Fields:
- `pair_id`, `source_example_id`
- `law_type` (`sufficiency`, `idempotence`, or `merge`)
- `original_text`, `rubric`, `reference_score`
- `summary_a`, `summary_b`
- `preferred` (`A`, `B`, or `tie`)
- `confidence` (0.0 to 1.0)
- `reasoning` (debug)
- `score_estimate_a`, `score_estimate_b` (judge or oracle scores)
- `oracle_error_a`, `oracle_error_b` (numeric oracle, normalized to 0-1 when scale is known)
- `generation_config_a`, `generation_config_b`
- `judge_model`, `timestamp`

Example:

```json
{
  "pair_id": "genrm_000123",
  "source_example_id": "doc_42",
  "law_type": "sufficiency",
  "original_text": " ... ",
  "rubric": "Preserve the political positioning ...",
  "reference_score": -35.0,
  "summary_a": " ... ",
  "summary_b": " ... ",
  "preferred": "A",
  "confidence": 0.85,
  "reasoning": "GenRM ranking score 2 ...",
  "score_estimate_a": 4.4,
  "score_estimate_b": 3.8,
  "oracle_error_a": null,
  "oracle_error_b": null,
  "generation_config_a": {"temperature": 0.3},
  "generation_config_b": {"temperature": 0.7},
  "judge_model": "qwen3-nemotron-genrm"
}
```

`PreferenceDataset` can save/load these pairs and convert them to DSPy examples or DPO-format records.

---

## GenRM judging (default)

GenRM comparisons are implemented in `src/ops_engine/training_framework/genrm_preference.py`.

### Prompt format

GenRM requires a special chat format with `response_1` and `response_2` roles:

```python
messages = [
    {"role": "user", "content": "Comparison instructions..."},
    {"role": "response_1", "content": summary_a},
    {"role": "response_2", "content": summary_b},
]
```

The prompt is law-aware:
- `law_type` controls the instructions.
- `original_text` is included for sufficiency and omitted for idempotence/merge.
- `extra_context` is used for idempotence (re-summaries) and merge (child summaries).

### Ranking and confidence

The GenRM output includes a ranking score from 1 to 6. We map it to a preference and confidence:
- `1-2` => prefer A, high confidence
- `3` => tie, confidence 0.5
- `4` => prefer B, low confidence
- `5-6` => prefer B, high confidence

This keeps a single "tie" bucket and avoids introducing score binning.

### Minimal usage

```python
from src.training.preference.genrm import GenRMJudge

judge = GenRMJudge(base_url="http://localhost:8001/v1")
result = judge.compare(
    context=rubric,
    original_text=doc_text,
    summary_a=summary_a,
    summary_b=summary_b,
    law_type="sufficiency",
)
print(result.preferred, result.confidence)
```

---

## Preference collection pipelines

### GenRM preference collection (primary)

`GenRMPreferenceCollector` generates candidate summaries and uses GenRM to compare all pairs:

- Candidate generation uses per-candidate temperatures, stored in `generation_config_a/b`.
- Pair order is randomly swapped before judging to reduce position bias.
- Law-specific handling is built in:
  - Sufficiency: compare summaries against the original text.
  - Idempotence: compare summaries with re-summaries provided in extra context.
  - Merge: compare merged summaries with child summaries provided in extra context.

Minimal usage:

```python
from src.training.preference.genrm import (
    GenRMJudge,
    GenRMPreferenceCollector,
)
from src.tasks.manifesto import LeafSummarizer

summarizer = LeafSummarizer(use_cot=True)
judge = GenRMJudge(base_url="http://localhost:8001/v1")

collector = GenRMPreferenceCollector(
    summarizer=summarizer,
    judge=judge,
    k_candidates=4,
    temperatures=[0.3, 0.5, 0.7, 0.9],
)

pairs = collector.collect_pairs_for_example(
    example_id="doc_1",
    original_text=doc_text,
    rubric=rubric,
    reference_score=reference_score,
    law_type="sufficiency",
)
```

The GenRM preference collection is now integrated into the unified training pipeline.

### Oracle-labeled preference collection (optional)

`OraclePreferenceCollector` uses a numeric oracle (e.g., RILE score) to create labels:

- Computes oracle errors for each candidate.
- Uses a tie margin (default 5.0) to determine ties.
- Computes a confidence based on error difference.

Minimal usage:

```python
from src.training.preference import (
    PreferenceCollector,
    PreferenceConfig,
)

collector = PreferenceCollector(
    summarizer=summarizer,
    oracle_predict=oracle_predict,
    config=PreferenceConfig(tie_margin=5.0),
)
```

The oracle preference collection is integrated into `src/training/collect_preferences.py`.

---

## Training the comparison module (GEPA)

The comparison module is `OPSComparisonModule` in `src/ops_engine/training_framework/ops_comparison_module.py`. It is law-aware and predicts `preferred`, `confidence`, and `reasoning`:

```python
result = comparison_module(
    law_type="sufficiency",
    rubric=rubric,
    original_text=doc_text,
    summary_a=summary_a,
    summary_b=summary_b,
    reference_score=reference_score,
)
```

Training is integrated into the unified training pipeline via `src/training/run_pipeline.py`.

Notes:
- Ties are excluded from DSPy training examples.
- You can filter by law type (`sufficiency`, `idempotence`, `merge`, or `all`).
- GEPA optimizes prompts with a preference accuracy metric.

---

## DPO data generation (sufficiency only)

DPO data is currently generated only for sufficiency.

Mechanics:
- Generate candidate summaries.
- Use a trained comparison module to select `chosen` vs `rejected`.
- Drop ties.
- Export in DPO format via `PreferenceDataset.to_dpo_format(law_type="sufficiency")`.

DPO data generation is integrated into the unified training pipeline.

---

## Configuration and temperatures

All pipeline-stage temperatures and sampling settings are centralized in `config/settings.yaml` and read via `src/config/settings.py`. Most scripts accept `--config` to override the path.

Key section:

```yaml
generation:
  summarizer:
    temperature: 0.5
    max_tokens: 2048
    candidate_temperatures: [0.3, 0.5, 0.7, 0.9]
  oracle:
    temperature: 0.3
    max_tokens: 2048
  genrm_judge:
    temperature: 0.6
    top_p: 0.95
    max_tokens: 2048
  comparison_judge:
    temperature: 0.3
    max_tokens: 2048
```

These values feed:
- Summarizer candidate generation (per-candidate temperature)
- Oracle scoring LMs (if applicable)
- GenRM comparisons
- DSPy comparison module training and DPO generation

---

## End-to-end usage examples

Start model servers:

```bash
# Summarizer (defaults from config/settings.yaml)
./scripts/start_vllm.sh qwen-30b-thinking

# GenRM judge (default is qwen3-nemotron-genrm-gguf on port 8001)
./scripts/start_oracle_server.sh --model qwen3-nemotron-genrm-gguf --port 8001
```

Run the unified training pipeline:

```bash
# Full training pipeline with all phases
python -m src.training.run_pipeline \
  --task manifesto_rile \
  --phase all \
  --samples 100 \
  --output-dir outputs/training_run

# Or use the shell script wrapper
./scripts/run_training_pipeline.sh \
  --task manifesto_rile \
  --train-samples 100 \
  --val-samples 30 \
  --test-samples 30
```

Optimize the judge specifically:

```bash
python experiments/manifesto_rile/optimize_judge.py \
  --preference-data data/preferences/preferences.json \
  --output-dir models/optimized_judge
```

---

## Extending to new oracles or human judges

The code is designed to accept arbitrary oracle judgments.

### Numeric oracle (LLM or heuristic)

Implement `oracle_predict(text) -> float` and plug it into `OraclePreferenceCollector`. This can wrap:
- A numeric LLM scorer (like RILE)
- A deterministic metric
- A human-in-the-loop scoring interface

### Preference-only oracle (LLM or human)

If you want a direct pairwise judge, implement a class with a `compare(...)` method that returns:

```
preferred: "A" | "B" | "tie"
confidence: float
```

You can then use `GenRMPreferenceCollector` as a template for a custom collector that calls your judge. The rest of the pipeline (PreferenceDataset, GEPA training, DPO generation) remains unchanged.

---

## Code map

Core preference infrastructure:
- `src/ops_engine/training_framework/preference.py` - `PreferencePair`, `PreferenceDataset`, `PreferenceCollector`
- `src/ops_engine/training_framework/genrm_preference.py` - GenRM judging and preference collection
- `src/ops_engine/training_framework/oracle_preference.py` - numeric-oracle preference collection
- `src/ops_engine/training_framework/ops_comparison_module.py` - law-aware comparison module
- `src/ops_engine/training_framework/preference_engine.py` - unified preference derivation

Training pipeline:
- `src/training/run_pipeline.py` - main training entry point
- `src/training/collect_preferences.py` - preference collection
- `src/training/judge_optimization.py` - judge training/optimization
- `src/training/tournament_loop.py` - tournament-based training

Experiment scripts:
- `experiments/manifesto_rile/optimize_judge.py` - judge optimization script
- `experiments/manifesto_rile/run_training_pipeline.py` - full pipeline script

Configuration:
- `config/settings.yaml` - model profiles and generation settings
- `src/config/settings.py` - YAML loader for scripts and modules
