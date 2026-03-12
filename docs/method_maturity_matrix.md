# Method Maturity Matrix (v1)

This document tracks accessibility parity across the four optimization paths:

1. LLM prompt optimization (Phase 2)
2. Embedding proxy heads (Phase 1.25)
3. Neural operators (Phase 1.3)
4. Generator fine-tuning (Phase 3.25 / 3.5)

## Fast Commands

```bash
# Baseline LLM optimization
./scripts/run_training_pipeline.sh --optimizer bootstrap_random_search --optimizer-budget light

# Embedding proxy
./scripts/run_training_pipeline.sh \
  --adaptive-embedding-proxy \
  --adaptive-embedding-head-method ridge \
  --embedding-proxy-fail-on-error

# Neural operators
./scripts/run_training_pipeline.sh \
  --train-neural-operators \
  --neural-operators-which both \
  --hybrid-oracle-seeded-ensemble

# Generator fine-tune (LoRA)
./scripts/run_training_pipeline.sh \
  --enable-genrm \
  --train-generator \
  --generator-method dpo \
  --generator-use-lora
```

## One-Command Compare

```bash
python scripts/run_method_compare.py \
  --output-root outputs/method_compare_$(date +%Y%m%d_%H%M%S)
```

Outputs:

- `<output_root>/<profile_name>/...` (per-profile pipeline run)
- `<output_root>/method_compare_manifest.json`
- `<output_root>/comparison_summary.json`
- `<output_root>/comparison_summary.md`

## Maturity Matrix

| Method | Primary Flags | Resume Control | Error Policy | Artifact Checkpoint | Summary Exposure |
|---|---|---|---|---|---|
| LLM Prompt Optimization | `--optimizer`, `--optimizer-budget` | `--resume`, `--rerun-optimization` | pipeline-level | `checkpoints/phase2_complete.json` | `final_stats.method_status.llm_prompt_optimization` |
| Embedding Proxy | `--adaptive-embedding-*` | `--rerun-embedding-proxy-on-resume` | `--embedding-proxy-fail-on-error` | `checkpoints/phase1_25_embedding_proxy_complete.json` | `final_stats.method_status.embedding_proxy` |
| Neural Operators | `--train-neural-operators`, `--neural-operators-*` | `--rerun-neural-operators-on-resume` | `--neural-operators-fail-on-error` | `checkpoints/phase1_3_neural_operators_complete.json` | `final_stats.method_status.neural_operators` |
| Generator Fine-Tune | `--train-generator`, `--generator-*` | `--rerun-generator-on-resume` | `--generator-fail-on-error` | `checkpoints/phase3_25_complete.json` | `final_stats.method_status.generator_finetune` |

## Settings Defaults

`config/settings.yaml` now includes:

- `chunking.adaptive.embedding_proxy.fail_on_error`
- `chunking.adaptive.embedding_proxy.rerun_on_resume`
- `training.generator.*`:
  - `enabled`, `method`, `model`, `use_lora`
  - `learning_rate`, `epochs`, `batch_size`
  - `min_preferences`, `fail_on_error`, `rerun_on_resume`
