# ThinkingTrees runtime artifacts

This document summarizes the expected inputs, outputs, and configuration surfaces for each CLI mode. The defaults assume the YAML files in `config/` but can be overridden on the command line.

## Training (`python main.py train`)

* **Inputs**
  * Preference or supervised dataset (`data.preference_dataset`).
  * Raw corpus for distillation (`data.raw_corpus`).
  * RNG seed (`seed`) to make checkpointing deterministic across runs.
* **Outputs**
  * Model checkpoints in `artifacts.checkpoint_dir` (default: `experiments/checkpoints`).
  * Distilled summaries in `data.distilled_summaries_dir` for later inference warm-starts.
  * Evaluation reports in `artifacts.eval_reports_dir` when `evaluation.run_eval` is enabled.
  * Training metadata in `training_run.yaml` beside checkpoints (records seeds, model paths, evaluation toggles).
* **Config notes**
  * `model.path` selects the base model; `adapter_dir` controls where adapters/LoRA weights are stored.
  * `evaluation.run_eval` and `evaluation.metrics` control whether checkpoints are scored before saving the best checkpoint.

## Inference (`python main.py infer`)

* **Inputs**
  * Source document passed via `--input`.
  * RNG seed (`seed`) plus generation settings for reproducible sampling.
* **Outputs**
  * Serialized tree summaries in `--output` or `artifacts.output_dir`.
  * Console tree statistics including chunk counts and height.
  * Optional evaluation scores if `evaluation.run_eval` is set.
* **Config notes**
  * `chunking` governs how large each chunk is and whether sentence splitting is used.
  * `model.max_model_len` and generation parameters (`temperature`, `max_tokens`) map to the runtime LLM service.

## Audit (`python main.py audit`)

* **Inputs**
  * Serialized tree path passed via `--input`.
  * RNG seed (`seed`) to make sampled node subsets reproducible.
  * Audit policies from `audit.*` to select sampling behavior.
* **Outputs**
  * Audit report YAML at `artifacts.report_path` (default: `experiments/audit/report.yaml`).
  * Flagged nodes or discrepancy examples in `artifacts.flagged_nodes_dir` when added to the pipeline.
* **Config notes**
  * `audit.discrepancy_threshold` and `audit.sample_budget` control the intensity of the audit.
  * `evaluation.run_eval` enables rubric re-scoring or hallucination sweeps using the metrics list.
