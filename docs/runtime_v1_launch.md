# Runtime v1 Launch Guide

Runtime v1 has one public vocabulary:

- `scorer`: practical task scorer `f`.
- `summarizer`: compression/summarization map `g`.
- `embedder`: embedding model used for retrieval or proxy features.
- `state_model`: deterministic or learned state-realization machinery.
- `oracle`: trusted target/evaluator `f*`, usually benchmark labels or a teacher.

Internal engine surfaces remain implementation details.

## Quick Local Gate

Run the v1 launch checks:

```bash
python scripts/run_experiment.py check --suite v1 --report
```

This writes a canonical experiment under `outputs/v1_launch_checks_*` and checks:

- umbrella inventory coverage;
- LongBench v2 smoke planning;
- LongBench v2 fixture init/run/aggregate over all v1 methods;
- required runtime artifacts;
- focused pytest gate unless `--skip-tests` is passed.

For a fast schema-only gate:

```bash
python scripts/run_experiment.py check --suite v1 --skip-tests --report
```

For a live endpoint gate, start the relevant servers first and run:

```bash
python scripts/run_experiment.py check --suite v1 --live --check-endpoints --report
```

The live path uses the configured endpoints instead of `--mock-llm`.
If the active scorer is not the endpoint in the config, override it centrally:

```bash
python scripts/run_experiment.py check --suite v1 \
  --live --check-endpoints --report \
  --scorer-endpoint http://localhost:8010/v1 \
  --scorer-model nvidia/Gemma-4-31B-IT-NVFP4
```

## Smoke Config

The checked-in smoke config is:

```bash
config/runtime_eval/longbench_v2_smoke.yaml
```

It points at:

```bash
tests/fixtures/runtime/longbench_v2_tiny.jsonl
```

and runs:

- `full_context`
- `retrieval`
- `summary_tree`
- `state_tree`
- `neural_operator`

## Full Config

The full-stack template is:

```bash
config/runtime_eval/longbench_v2_full_stack.yaml
```

Use it with:

```bash
python scripts/run_runtime_eval.py plan \
  --config config/runtime_eval/longbench_v2_full_stack.yaml \
  --output-dir outputs/runtime_eval \
  --experiment-id longbench_v2_full_stack \
  --check-endpoints

python scripts/run_runtime_eval.py init \
  --config config/runtime_eval/longbench_v2_full_stack.yaml \
  --output-dir outputs/runtime_eval \
  --experiment-id longbench_v2_full_stack

python scripts/run_runtime_eval.py run \
  --experiment-dir outputs/runtime_eval/longbench_v2_full_stack

python scripts/run_runtime_eval.py aggregate \
  --experiment-dir outputs/runtime_eval/longbench_v2_full_stack
```

## Paper Result Table

After aggregation, produce the paper-facing method matrix:

```bash
python scripts/run_experiment.py report \
  --profile runtime_v1 \
  --output-root outputs/runtime_eval/longbench_v2_full_stack
```

This writes:

- `paper_summary/runtime_v1_summary.json`
- `paper_summary/runtime_v1_summary.md`

and registers both in the run's `artifacts.json`.

## State Model Semantics

`state_model` is not an answerer in v1. It is a state-realization role used by
methods such as `neural_operator` to select or render evidence through typed
operator calls. The final task answer still routes through `scorer`.

Supported operation vocabulary:

- `encode_leaf`
- `merge_state`
- `score_root`
- `select_evidence`
- `render_state`

If `neural_operator` has no configured operator-backed `state_model`, it may
fall back to embedder-based evidence selection. Traces and artifacts must record
that fallback as embedding-backed selection, not state-model selection.

## Required Output Contract

Every supported runtime-eval run should produce:

- `experiment_manifest.json`
- `experiment_status.json`
- `artifacts.json`
- `results.jsonl`
- `metrics.json`
- `predictions.jsonl`
- `steps.jsonl`
- `calls.jsonl` when model, embedding, or operator traffic occurs

Large contexts and prompts should not be copied into canonical sidecars.

## Public Config Contract

Runtime-eval configs are v2-only. These are the public keys:

- `methods`
- `scorer`
- `summarizer`
- `embedder`
- `state_model`
- `oracle`

Old public names such as `answerer`, `state_operator`, `model`, `resources`,
`surfaces`, and `modes` should fail fast in runtime-eval configs.
