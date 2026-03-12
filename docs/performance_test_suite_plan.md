# ThinkingTrees Performance Test Suite Plan

## Goals

- Cover performance at every layer: micro, component, integration, end-to-end.
- Make runs reproducible with scenario configs committed to the repo.
- Support regression checks via baseline vs candidate comparisons in CI/local.
- Keep smoke runs fast and full runs comprehensive.

## Current Suite Architecture

- Runner core: `src/benchmark/perf_suite.py`
- Microbench core: `src/benchmark/component_microbench.py`
- Run CLI: `scripts/run_performance_suite.py`
- Compare CLI: `scripts/compare_performance_suite_runs.py`
- Microbench CLI: `scripts/run_component_microbench.py`
- Memory reuse probe CLI: `scripts/run_memory_reuse_probe.py`
- Prefix metrics probe CLI: `scripts/probe_vllm_prefix_metrics.py`
- Recovery path probe CLI: `scripts/run_recovery_path_probe.py`
- Budget sweep probe CLI: `scripts/run_budget_sweep_probe.py`
- Full scenario: `benchmarks/scenarios/performance_suite_full.yaml`
- Smoke scenario: `benchmarks/scenarios/performance_suite_smoke.yaml`

## Layer Coverage Matrix

| Layer | Pipeline Surface | Case IDs | Status |
|---|---|---|---|
| Micro | Chunking perf | `micro_chunker` | Implemented |
| Micro | Conditional memory perf | `micro_conditional_memory` | Implemented |
| Micro | Prompt build + numeric parse | `micro_prompting` | Implemented |
| Micro | Tree assembly overhead | `micro_tree_builder` | Implemented |
| Component | Task model throughput | `component_task_throughput` | Implemented |
| Component | GenRM throughput | `component_genrm_throughput` | Implemented |
| Component | GPU transition latency | `component_gpu_transitions` | Implemented |
| Component | Prefix-cache metric visibility | `component_prefix_cache_metrics` | Implemented |
| Integration | Cold/warm baseline | `integration_perf_baseline` | Implemented |
| Integration | Architecture gate (manifesto) | `integration_arch_gate_manifesto` | Implemented |
| Integration | ConditionalMemory pass2 reuse | `integration_memory_reuse_pass2` | Implemented |
| Integration | Crash-recovery overhead | `integration_recovery_path` | Implemented (default-disabled) |
| E2E | Full training pipeline smoke | `e2e_training_pipeline_smoke` | Implemented |
| E2E | Optimizer budget scaling | `e2e_budget_sweep_light_med_heavy` | Implemented |

## Gaps To Implement Next

- Optional: add domain-specific budget sweeps for non-manifesto tasks.
- Optional: add parser-router/adaptive-chunking perf cases once those paths are enabled in default runs.
- Optional: add long-run stability case (N-hour) to capture drift and leak behavior.

## Resume/Recovery Hardening (Phase 2 GEPA)

- Added runtime operation journal for Phase 2 optimization:
  - Path: `checkpoints/phase2_runtime/<signature_id>/state.json`
  - Tracks per-iteration and per-component status (`running/completed/failed/skipped`), artifacts, and timestamps.
- Added per-component artifacts for interruption-safe resume:
  - `checkpoints/phase2_runtime/<signature_id>/artifacts/iteration_<N>/{scorer,leaf_summarizer,merge_summarizer}.json`
- Added GEPA prompt/trajectory exports from `gepa_state.bin`:
  - In GEPA log dir: `gepa_trajectory_snapshot.json`, `gepa_prompt_trajectory.jsonl`
  - Mirrored into runtime journal dir: `gepa_exports/<component>_*.{json,jsonl}`
- Resume behavior:
  - On `--resume`, if an iteration/component artifact exists and was marked `compiled/completed`, load artifact and continue without rerunning that sub-step.
  - If interrupted during GEPA compile, GEPA `log_dir` state plus exported prompt trajectory are preserved; rerun with `--resume` continues.

### Recovery Validation Steps

```bash
# Unit coverage for journal/signature + GEPA export
python3 -m pytest tests/training/test_phase2_recovery_journal.py -q

# Integration check: interrupt a run mid-Phase2, then resume
./scripts/run_training_pipeline.sh <args> --output-dir <run_dir>
# interrupt during scorer/leaf/merge GEPA
./scripts/run_training_pipeline.sh <same args> --output-dir <run_dir> --resume
```

## Resume/Recovery Hardening (Full Pipeline Runtime Journal)

- Added a pipeline-wide runtime journal:
  - Path: `checkpoints/pipeline_runtime_state.json`
  - Tracks phase transitions with status: `running`, `completed`, `failed`, `skipped`, `interrupted`
  - Phases covered: `setup`, `phase1`, `phase1_25`, `phase1_5`, `phase1_55`, `phase1_6`, `phase1_75`, `phase2`, `phase3`, `phase3_1`, `phase3_25`, `phase3_5`, `finalize`
- Resume behavior:
  - On `--resume`, any previously `running` phase is marked `interrupted` before restart.
  - Journal tracks `resume_count` and appends transition events for each new run.
- Final stats now include:
  - `pipeline_runtime_state_path`
  - `pipeline_runtime_status`
  - `pipeline_runtime_current_phase`

### Validation Steps

```bash
# Unit tests for runtime journal state transitions
python3 -m pytest tests/training/test_pipeline_runtime_journal.py -q

# Tiny run + resume check (verify checkpoints/pipeline_runtime_state.json updates)
./scripts/run_training_pipeline.sh <args> --output-dir <run_dir>
./scripts/run_training_pipeline.sh <args> --output-dir <run_dir> --resume
```

## Metric Contract

- Throughput (`higher` is better):
  - `docs_per_second`
  - `recommended_req_per_s`
  - `peak_tokens_per_second`
- Latency (`lower` is better):
  - `latency_ms_p95`
  - transition duration seconds
- Quality (`lower` is better unless correlation):
  - `mae`
  - `frac_neutral`
  - `pearson_r` (`higher`)
- Reliability:
  - command failures/timeouts
  - pipeline `warnings` / `errors` from `run.log`

Each case defines thresholds using `metric_rules`:

- `path`: dotted path into repeat artifact payload
- `direction`: `higher` or `lower`
- `max_regression_pct`: percent tolerance
- `max_regression_abs`: optional absolute tolerance
- `aggregate`: repeat aggregation (`median` default)

## Standard Workflows

### 1) Fast local smoke

```bash
python3 scripts/run_performance_suite.py \
  --scenario benchmarks/scenarios/performance_suite_smoke.yaml
```

### 2) Full benchmark run

```bash
python3 scripts/run_performance_suite.py \
  --scenario benchmarks/scenarios/performance_suite_full.yaml
```

### 3) Baseline vs candidate comparison

```bash
python3 scripts/compare_performance_suite_runs.py \
  --scenario benchmarks/scenarios/performance_suite_full.yaml \
  --baseline <baseline_suite_results.json> \
  --candidate <candidate_suite_results.json>
```

### 4) Targeted layer/case runs

```bash
python3 scripts/run_performance_suite.py \
  --scenario benchmarks/scenarios/performance_suite_full.yaml \
  --include-layers micro,component \
  --include-cases micro_chunker,component_task_throughput
```

## Artifact Layout

Each suite run writes:

- `suite_results.json`
- `suite_results.md`
- `cases/<case_id>/repeat_XX/command.log`
- extractor-specific JSON/CSV outputs per case

Recommended storage:

- Smoke runs under `outputs/performance_suite/run_<timestamp>/`
- Named baselines copied to `outputs/performance_suite/baselines/`

## CI Recommendation

- Run smoke scenario on every PR.
- Run full scenario nightly (or on-demand for performance-sensitive changes).
- Fail compare step if regressions exceed thresholds (`exit code 2`).

## Ownership

- Scenario upkeep: benchmark/system owners.
- Threshold tuning: jointly by throughput + quality owners.
- Regression adjudication: require explicit approval to raise tolerated regressions.
