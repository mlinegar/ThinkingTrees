#!/usr/bin/env bash
# Parallel qsentence LLM comparison queue.
# Splits GPUs between DiffusionGemma and Gemma4 so neither model blocks the other.
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-/home/mlinegar/ThinkingTrees}
cd "$REPO_ROOT" || exit 2

PY=${PY:-./venv/bin/python}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-outputs/manifesto_parallel_llm_qsentence_${STAMP}}
FULL_GRID=${FULL_GRID:-outputs/manifesto_qsentence_dspy_labeled_grid}
LEAF_QS=${LEAF_QS:-16,8,4,2}
MAX_EVAL_TREES=${MAX_EVAL_TREES:-12}
DGEMMA_MODEL=${DGEMMA_MODEL:-openai/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4}
GEMMA4_MODEL=${GEMMA4_MODEL:-openai/nvidia/Gemma-4-31B-IT-NVFP4}
DGEMMA_DIRECT_WEIGHT=${DGEMMA_DIRECT_WEIGHT:-0.75}
DGEMMA_PROXY_WEIGHT=${DGEMMA_PROXY_WEIGHT:-0.25}
GEMMA4_DIRECT_WEIGHT=${GEMMA4_DIRECT_WEIGHT:-0.75}
GEMMA4_PROXY_WEIGHT=${GEMMA4_PROXY_WEIGHT:-0.25}
DGEMMA_NUM_THREADS=${DGEMMA_NUM_THREADS:-12}
DGEMMA_MAX_CONCURRENT=${DGEMMA_MAX_CONCURRENT:-16}
DGEMMA_BATCH_SIZE=${DGEMMA_BATCH_SIZE:-8}
GEMMA4_NUM_THREADS=${GEMMA4_NUM_THREADS:-192}
GEMMA4_MAX_CONCURRENT=${GEMMA4_MAX_CONCURRENT:-384}
GEMMA4_BATCH_SIZE=${GEMMA4_BATCH_SIZE:-192}
QUALITY_FAILURE_PATTERN=${QUALITY_FAILURE_PATTERN:-"ServerDisconnectedError|Request dspy_batch_.* failed|AdapterParseError|Adapter JSONAdapter failed|LM response cannot be serialized|ERROR dspy|Traceback|Q-sentence g call produced no valid compact state|Q-sentence g call produced no parseable completion|Q-sentence compact target scorer call failed|no parseable completion|row error|Failing because|ClientConnectorError|Cannot connect to host|Internal Server Error|Default vLLM sampling parameters|OutOfMemoryError|CUDA out of memory|HTTP [45][0-9][0-9]"}
mkdir -p "$OUT_ROOT"/logs "$OUT_ROOT"/reports "$OUT_ROOT"/servers

log() {
  printf "[%s] %s\n" "$(date -Is)" "$*" | tee -a "$OUT_ROOT/logs/queue.log"
}

stop_job_root() {
  local root=$1
  if [[ -f "$root/manifest.json" ]]; then
    "$PY" scripts/long_job.py stop --job-root "$root" >/dev/null 2>&1 || true
  fi
}

stop_model_servers() {
  ./scripts/stop_small_servers.sh --all >/dev/null 2>&1 || true
  for i in 0 1 2 3; do
    stop_job_root "outputs/diffusiongemma_qsentence_worker_gpu${i}"
    stop_job_root "outputs/gemma4_qsentence_server_gpu${i}_launcher"
  done
  for root in "$OUT_ROOT"/servers/* outputs/manifesto_conference_overnight_*/servers/gemma4_*; do
    [[ -d "$root" ]] || continue
    stop_job_root "$root"
  done
  sleep 8
}

wait_for_server() {
  local url=$1
  local seconds=${2:-1200}
  local deadline=$((SECONDS + seconds))
  until curl -fsS "$url/models" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      log "WARN server did not become ready: $url"
      return 1
    fi
    sleep 10
  done
  log "READY $url"
}

start_dgemma_workers() {
  log "Starting DiffusionGemma workers on GPUs 0,1 ports 8004,8005"
  ./scripts/start_diffusiongemma_qsentence_worker.sh 0 8004 8 0.75 8192 32768 >/dev/null
  ./scripts/start_diffusiongemma_qsentence_worker.sh 1 8005 8 0.75 8192 32768 >/dev/null
  wait_for_server http://localhost:8004/v1 1200 || return 1
  wait_for_server http://localhost:8005/v1 1200 || return 1
}

start_gemma4_workers() {
  log "Starting Gemma4 workers on GPUs 2,3 ports 8010,8011"
  for spec in 2:8010 3:8011; do
    IFS=: read -r gpu port <<< "$spec"
    local root="$OUT_ROOT/servers/gemma4_gpu${gpu}"
    stop_job_root "$root"
    "$PY" scripts/long_job.py launch       --name "parallel_gemma4_gpu${gpu}"       --job-root "$root"       --cwd "$REPO_ROOT"       --replace-existing       -- ./scripts/start_vllm.sh gemma-4-31b-it-nvfp4 --port "$port" --cuda-devices "$gpu" --generation-config vllm       >/dev/null
  done
  wait_for_server http://localhost:8010/v1 1500 || return 1
  wait_for_server http://localhost:8011/v1 1500 || return 1
}

run_dgemma() {
  log "START dgemma_fixed leafs=$LEAF_QS max_eval_trees=$MAX_EVAL_TREES"
  TT_DSPY_DROP_RESPONSE_FORMAT=1 TT_DSPY_WRAP_BARE_FIELD_OUTPUT=1 TT_SKIP_FULL_TREE_TRACES=1     "$PY" scripts/run_manifesto_qsentence_dspy_ladder.py       --fg-grid-dir "$FULL_GRID"       --output-dir "$OUT_ROOT/dgemma_fixed_leafgrid"       --leaf-qsentences "$LEAF_QS"       --max-iterations 2       --target-dimensions all       --max-eval-trees "$MAX_EVAL_TREES"       --eval-sample-seed 20260621       --dspy-optimizer gepa       --dspy-budget light       --dspy-max-train-records 2048       --dspy-g-direct-parse-reward-weight "$DGEMMA_DIRECT_WEIGHT"       --dspy-g-f-proxy-reward-weight "$DGEMMA_PROXY_WEIGHT"       --fail-on-row-error       --dspy-model "$DGEMMA_MODEL"       --dspy-api-base http://localhost:8004/v1,http://localhost:8005/v1       --dspy-batch-routing-policy affinity_load_aware       --dspy-num-threads "$DGEMMA_NUM_THREADS"       --dspy-batch-max-concurrent "$DGEMMA_MAX_CONCURRENT"       --dspy-batch-size "$DGEMMA_BATCH_SIZE"       --dspy-lm-context-tokens 32768       --verbose       >"$OUT_ROOT/logs/dgemma_fixed.log" 2>&1
  local status=$?
  echo "$status" > "$OUT_ROOT/logs/dgemma_fixed.status"
  log "END dgemma_fixed status=$status log=$OUT_ROOT/logs/dgemma_fixed.log"
  return "$status"
}

run_gemma4() {
  log "START gemma4_fixed leafs=$LEAF_QS max_eval_trees=$MAX_EVAL_TREES"
  TT_DSPY_DROP_RESPONSE_FORMAT=0 TT_DSPY_WRAP_BARE_FIELD_OUTPUT=1 TT_SKIP_FULL_TREE_TRACES=1     "$PY" scripts/run_manifesto_qsentence_dspy_ladder.py       --fg-grid-dir "$FULL_GRID"       --output-dir "$OUT_ROOT/gemma4_fixed_leafgrid"       --leaf-qsentences "$LEAF_QS"       --max-iterations 2       --target-dimensions all       --max-eval-trees "$MAX_EVAL_TREES"       --eval-sample-seed 20260621       --dspy-optimizer gepa       --dspy-budget light       --dspy-max-train-records 2048       --dspy-g-direct-parse-reward-weight "$GEMMA4_DIRECT_WEIGHT"       --dspy-g-f-proxy-reward-weight "$GEMMA4_PROXY_WEIGHT"       --fail-on-row-error       --dspy-model "$GEMMA4_MODEL"       --dspy-api-base http://localhost:8010/v1,http://localhost:8011/v1       --dspy-batch-routing-policy round_robin       --dspy-num-threads "$GEMMA4_NUM_THREADS"       --dspy-batch-max-concurrent "$GEMMA4_MAX_CONCURRENT"       --dspy-batch-size "$GEMMA4_BATCH_SIZE"       --dspy-lm-context-tokens 32768       --verbose       >"$OUT_ROOT/logs/gemma4_fixed.log" 2>&1
  local status=$?
  echo "$status" > "$OUT_ROOT/logs/gemma4_fixed.status"
  log "END gemma4_fixed status=$status log=$OUT_ROOT/logs/gemma4_fixed.log"
  return "$status"
}

write_reports() {
  log "START reports"
  local report_status=0
  "$PY" scripts/compare_manifesto_qsentence_substrates.py     dgemma_old=outputs/manifesto_qsentence_diffusiongemma_FULL218_leafgrid     dgemma_fixed="$OUT_ROOT/dgemma_fixed_leafgrid"     gemma4_fixed="$OUT_ROOT/gemma4_fixed_leafgrid"     --output-dir "$OUT_ROOT/reports/substrate_comparison"     >>"$OUT_ROOT/logs/reports.log" 2>&1 || report_status=$?
  "$PY" scripts/summarize_manifesto_qsentence_per_dimension.py     dgemma_old=outputs/manifesto_qsentence_diffusiongemma_FULL218_leafgrid     dgemma_fixed="$OUT_ROOT/dgemma_fixed_leafgrid"     gemma4_fixed="$OUT_ROOT/gemma4_fixed_leafgrid"     --output-dir "$OUT_ROOT/reports/per_dimension"     >>"$OUT_ROOT/logs/reports.log" 2>&1 || report_status=$?
  log "END reports status=$report_status"
  return "$report_status"
}

log "Output root: $OUT_ROOT"
log "Stopping existing model servers before split-GPU run"
stop_model_servers
start_dgemma_workers || exit 1
start_gemma4_workers || exit 1

run_dgemma &
dgemma_pid=$!
run_gemma4 &
gemma4_pid=$!

quality_failure_file="$OUT_ROOT/logs/quality_failure.txt"

monitor_quality_failures() {
  local dlog="$OUT_ROOT/logs/dgemma_fixed.log"
  local glog="$OUT_ROOT/logs/gemma4_fixed.log"
  local match
  while kill -0 "$dgemma_pid" >/dev/null 2>&1 || kill -0 "$gemma4_pid" >/dev/null 2>&1; do
    if [[ -f "$dlog" ]] && match=$(rg -a -n -m 1 "$QUALITY_FAILURE_PATTERN" "$dlog" 2>/dev/null); then
      printf "dgemma_fixed:%s\n" "$match" > "$quality_failure_file"
      log "FAILFAST quality failure in dgemma_fixed: ${match:0:500}"
      kill "$dgemma_pid" "$gemma4_pid" >/dev/null 2>&1 || true
      return 97
    fi
    if [[ -f "$glog" ]] && match=$(rg -a -n -m 1 "$QUALITY_FAILURE_PATTERN" "$glog" 2>/dev/null); then
      printf "gemma4_fixed:%s\n" "$match" > "$quality_failure_file"
      log "FAILFAST quality failure in gemma4_fixed: ${match:0:500}"
      kill "$dgemma_pid" "$gemma4_pid" >/dev/null 2>&1 || true
      return 97
    fi
    sleep 5
  done
}

monitor_quality_failures &
quality_monitor_pid=$!

read_leg_statuses() {
  if [[ -z "${dgemma_status:-}" && -f "$OUT_ROOT/logs/dgemma_fixed.status" ]]; then
    dgemma_status=$(<"$OUT_ROOT/logs/dgemma_fixed.status")
  fi
  if [[ -z "${gemma4_status:-}" && -f "$OUT_ROOT/logs/gemma4_fixed.status" ]]; then
    gemma4_status=$(<"$OUT_ROOT/logs/gemma4_fixed.status")
  fi
}

abort_peer_leg() {
  local failed_name=$1
  local peer_name=$2
  local peer_pid=$3
  local peer_status_file=$4
  log "FAILFAST $failed_name failed; stopping $peer_name"
  kill "$peer_pid" >/dev/null 2>&1 || true
  wait "$peer_pid" >/dev/null 2>&1 || true
  if [[ ! -f "$peer_status_file" ]]; then
    echo 143 > "$peer_status_file"
    log "END $peer_name status=143 aborted_after_peer_failure"
  fi
}

dgemma_status=
gemma4_status=
while [[ -z "$dgemma_status" || -z "$gemma4_status" ]]; do
  wait -n
  wait_status=$?
  read_leg_statuses

  if [[ -f "$quality_failure_file" ]]; then
    log "FAILFAST quality failure detected; stopping model legs"
    kill "$dgemma_pid" "$gemma4_pid" >/dev/null 2>&1 || true
    wait "$dgemma_pid" >/dev/null 2>&1 || true
    wait "$gemma4_pid" >/dev/null 2>&1 || true
    if [[ ! -f "$OUT_ROOT/logs/dgemma_fixed.status" ]]; then
      echo 97 > "$OUT_ROOT/logs/dgemma_fixed.status"
    fi
    if [[ ! -f "$OUT_ROOT/logs/gemma4_fixed.status" ]]; then
      echo 97 > "$OUT_ROOT/logs/gemma4_fixed.status"
    fi
    read_leg_statuses
    break
  fi

  if [[ -n "$dgemma_status" && "$dgemma_status" -ne 0 && -z "$gemma4_status" ]]; then
    abort_peer_leg dgemma_fixed gemma4_fixed "$gemma4_pid" "$OUT_ROOT/logs/gemma4_fixed.status"
    read_leg_statuses
    break
  fi
  if [[ -n "$gemma4_status" && "$gemma4_status" -ne 0 && -z "$dgemma_status" ]]; then
    abort_peer_leg gemma4_fixed dgemma_fixed "$dgemma_pid" "$OUT_ROOT/logs/dgemma_fixed.status"
    read_leg_statuses
    break
  fi

  if [[ "$wait_status" -ne 0 && -z "$dgemma_status" && -z "$gemma4_status" ]]; then
    log "FAILFAST unknown model leg failed with status=$wait_status; stopping both legs"
    kill "$dgemma_pid" "$gemma4_pid" >/dev/null 2>&1 || true
    wait "$dgemma_pid" >/dev/null 2>&1 || true
    wait "$gemma4_pid" >/dev/null 2>&1 || true
    read_leg_statuses
    dgemma_status=${dgemma_status:-$wait_status}
    gemma4_status=${gemma4_status:-143}
    break
  fi

done

kill "$quality_monitor_pid" >/dev/null 2>&1 || true
wait "$quality_monitor_pid" >/dev/null 2>&1 || true

if [[ "$dgemma_status" -ne 0 || "$gemma4_status" -ne 0 ]]; then
  log "SKIP reports because a model run failed: dgemma_status=$dgemma_status gemma4_status=$gemma4_status"
  stop_model_servers
  exit 1
fi

write_reports
report_status=$?
stop_model_servers
log "complete dgemma_status=$dgemma_status gemma4_status=$gemma4_status report_status=$report_status"
exit "$report_status"
