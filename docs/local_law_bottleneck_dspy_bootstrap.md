# Local-Law Bottleneck DSPy Bootstrap

## Goal
Optimize summarization prompts/modules for local-law preservation by prioritizing the weakest law signal, not average performance.

This uses the existing LawStress DSPy/GEPA pipeline with a bottleneck objective:

- `C1`: source -> summary score preservation
- `C2`: re-summary stability + strict same-side
- `C3`: merge/substitution consistency

## Objective
For each training example, compute local-law component scores in `[0, 1]` and optimize:

`score = min(component_1, component_2, ..., component_k)`

Equivalent CLI values:

- `--objective-aggregate min`
- `--objective-aggregate bottleneck_min` (alias)

This prevents "average-score cheating" where one law regresses while others improve.

## Current Defaults

The following now default to bottleneck mode (`min`):

- `scripts/bootstrap_lawstress_summarizer.py`
- `scripts/run_manifesto_local_law_bootstrap_poc.py`
- `scripts/run_manifesto_local_law_bootstrap_manual.py`
- `LawStressBootstrapObjectiveConfig.aggregate_mode`

## DSPy Run (Smoke)

Assumes:

- Student vLLM is on `:8000`
- Embedding server is on `:8003`
- LawStress records exist

Example:

```bash
python scripts/bootstrap_lawstress_summarizer.py \
  --records outputs/bootstrap_poc_manual_20260304_174817/stage_a/lawstress_data/lawstress_records.jsonl \
  --output-dir outputs/_tmp_lawstress_bottleneck_gepa_smoke \
  --student-port 8000 \
  --student-model /mnt/data/models/AxionML/Qwen3.5-35B-A3B-NVFP4 \
  --embedding-url http://localhost:8003/v1 \
  --embedding-model Qwen/Qwen3-Embedding-8B \
  --gepa-budget light \
  --num-threads 8 \
  --objective-aggregate min
```

## Validation

Check output:

- `bootstrap_stats.json` -> `objective.aggregate_mode`
- `val_metric.baseline` vs `val_metric.optimized`
- `trained_modules/unified_g_final.json`

Then run held-out evaluation (`--splits test`) with:

- `scripts/eval_lawstress_dspy_module.py`
- `scripts/eval_manifesto_teacher_trace_local_laws.py`

