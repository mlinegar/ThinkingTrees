# Pipeline Throughput Limit Suite

This benchmark suite measures throughput ceilings for key runtime steps:

- `task_single`: task model on one endpoint (typically `8000`)
- `task_merge`: task model on merge-style summary prompts
- `task_score`: task/oracle model on scorer-style numeric prompts
- `task_dp2`: task model load-balanced across two endpoints (`8000` + `8002`)
- `genrm_raw`: direct GenRM `chat/completions` endpoint throughput
- `genrm_batch`: GenRM throughput through `AsyncBatchGenRMClient` (pipeline path)

It sweeps concurrency, then reports:

- request success rate
- request/sec
- token/sec
- p50/p95 latency
- recommended and peak concurrency points

For GenRM, generic steps expand into two mode variants by default:

- `fast`: `disable_thinking=true`, `force_json_response=true`
- `think`: `disable_thinking=false`, `force_json_response=false`

So `--steps genrm_batch` runs both `genrm_batch_fast` and `genrm_batch_think`
unless you override `--genrm-modes`.

## CLI

```bash
./venv/bin/python scripts/run_pipeline_throughput_limits.py --help
```

## Quick Smoke Run

```bash
./venv/bin/python scripts/run_pipeline_throughput_limits.py \
  --steps task_single,task_merge,task_score,genrm_batch \
  --genrm-modes fast,think \
  --concurrency-grid 1,2 \
  --min-requests-per-point 8 \
  --requests-per-concurrency 2 \
  --warmup-requests 2 \
  --output-json outputs/_tmp_throughput_limits_smoke.json \
  --output-csv outputs/_tmp_throughput_limits_smoke.csv
```

## Full Limit Sweep (Recommended)

```bash
./venv/bin/python scripts/run_pipeline_throughput_limits.py \
  --steps task_single,task_merge,task_score,task_dp2,genrm_batch,genrm_raw \
  --genrm-modes fast,think \
  --task-url http://localhost:8000/v1 \
  --task-replica-url http://localhost:8002/v1 \
  --genrm-url http://localhost:8001/v1 \
  --concurrency-grid 1,2,4,8,12,16,24 \
  --min-requests-per-point 48 \
  --requests-per-concurrency 6 \
  --warmup-requests 4 \
  --task-timeout-seconds 120 \
  --genrm-timeout-seconds 360 \
  --task-max-tokens 256 \
  --genrm-max-tokens 256 \
  --min-success-rate 0.98 \
  --output-json outputs/throughput_limits_manifesto.json \
  --output-csv outputs/throughput_limits_manifesto.csv
```

## Notes

- By default, the runner starts fresh local servers, clears required ports, and removes stale vLLM state (`--auto-start-servers`, `--clear-ports`, `--clear-vllm-state`).
- Use `--no-auto-start-servers` if you want to target already-running or remote endpoints.
- For `task_dp2` with `--no-auto-start-servers`, both task ports must already be live.
- Use `--genrm-modes fast,think` to compare both quality/latency regimes.
- For production throughput, `fast` mode is typically more stable.
- If no stable points are found, lower concurrency or increase timeout and rerun.

## Winner Quality A/B (Same Documents)

Use this when you want quality, not just throughput:

- Same documents
- Same candidate pools
- Fast and think tournament winners compared head-to-head
- Winner quality measured against reference RILE via oracle scoring
- Use a non-GenRM scorer endpoint for `--oracle-url` (typically task/oracle model on `:8000`)

```bash
./venv/bin/python scripts/run_genrm_winner_ab_test.py \
  --fast-url http://localhost:8001/v1 \
  --think-url http://localhost:8002/v1 \
  --candidate-url http://localhost:8001/v1 \
  --oracle-url http://localhost:8000/v1 \
  --max-docs 80 \
  --countries 51,41 \
  --min-year 2000 \
  --k-candidates 4 \
  --candidate-temperatures 0.3,0.5,0.7,0.9 \
  --doc-concurrency 4 \
  --genrm-max-concurrent 8 \
  --output-json outputs/genrm_winner_ab.json \
  --output-csv outputs/genrm_winner_ab.csv
```
