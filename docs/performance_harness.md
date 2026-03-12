# Performance Harness (Micro/Meso/Macro)

The perf harness runs a manifest-defined benchmark matrix and writes one
standard JSON artifact per run.

## Runner

```bash
python3 scripts/run_perf_harness.py --manifest config/perf/perf_matrix.yaml --profile ci
```

Output defaults to:

```text
outputs/perf_harness/run_<UTCSTAMP>.json
```

Each artifact includes:

- run metadata (`created_utc`, host, manifest path, profile)
- per-scenario execution details (`exit_code`, `wall_seconds`, `log_path`)
- extracted metrics (from JSON metric files via dotted paths)
- regression rule outcomes (`error` and `warn`)
- rollup summary counts

## Profiles

Defined in [`config/perf/perf_matrix.yaml`](/home/mlinegar/ThinkingTrees/config/perf/perf_matrix.yaml):

- `ci`: micro + meso scenarios (no live server dependency)
- `nightly`: micro + meso + macro scenarios

## Useful Flags

```bash
# List selected scenarios without running
python3 scripts/run_perf_harness.py --profile ci --list-scenarios

# Write artifact but skip command execution
python3 scripts/run_perf_harness.py --profile ci --dry-run

# Fail build on regression-rule errors
python3 scripts/run_perf_harness.py --profile ci --fail-on-regression
```

## Expected-Failure Scenarios

Scenarios can declare expected outcomes so intentional negative controls do not
fail the whole run:

```yaml
expected:
  outcome: fail
  failure_modes: ["regression"]  # or ["command"] / ["any"]
```

- `outcome: pass` (default) means command + error-severity regressions must pass.
- `outcome: fail` means a failure is required; if it unexpectedly passes, the scenario fails.
- Run summaries include `expected_failures`, `expected_failures_met`, and `unexpected_passes`.

## Overnight Memory Matrix

Use the expanded manifesto memory matrix:

```bash
python3 scripts/run_perf_harness.py \
  --manifest config/perf/manifesto_memory_overnight_matrix.yaml \
  --profile overnight_memory_full \
  --fail-on-regression
```

Background launcher:

```bash
./scripts/run_overnight_memory_matrix.sh --profile overnight_memory_full --fail-on-regression
```

That launcher now also writes:

- `outputs/perf_harness/overnight_memory/<run_id>/result.json`
- `outputs/perf_harness/overnight_memory/<run_id>/recommended_defaults.json`

You can recompute recommendations manually from any artifact:

```bash
python3 scripts/recommend_manifesto_memory_defaults.py \
  --artifact outputs/perf_harness/overnight_memory/<run_id>/result.json
```
