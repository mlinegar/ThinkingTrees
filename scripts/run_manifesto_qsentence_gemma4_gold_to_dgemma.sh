#!/usr/bin/env bash
# Queue a transfer experiment: train f/g with Gemma4 on gold q-sentence labels,
# then evaluate those compiled artifacts with DGemma as the runtime LM.
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-/home/mlinegar/ThinkingTrees}
cd "$REPO_ROOT" || exit 2

PY=${PY:-./venv/bin/python}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
WAIT_JOB_ROOT=${WAIT_JOB_ROOT:-outputs/manifesto_parallel_llm_qsentence_failfast_launcher}
infer_latest_source_root() {
  if [[ -f "$WAIT_JOB_ROOT/job.log" ]]; then
    rg -a "Output root:" "$WAIT_JOB_ROOT/job.log" 2>/dev/null | tail -n 1 | sed "s/^.*Output root: //"
  fi
}
SOURCE_RUN_ROOT=${SOURCE_RUN_ROOT:-$(infer_latest_source_root)}
if [[ -z "${SOURCE_RUN_ROOT:-}" ]]; then
  printf "SOURCE_RUN_ROOT is unset and no Output root line was found in %s/job.log\n" "$WAIT_JOB_ROOT" >&2
  exit 2
fi
SOURCE_GRID=${SOURCE_GRID:-$SOURCE_RUN_ROOT/gemma4_fixed_leafgrid}
OUT_ROOT=${OUT_ROOT:-outputs/manifesto_qsentence_gemma4_gold_to_dgemma_${STAMP}}
FULL_GRID=${FULL_GRID:-outputs/manifesto_qsentence_dspy_labeled_grid}
LEAF_QS=${LEAF_QS:-1,2,4,8,16}
MAX_EVAL_TREES=${MAX_EVAL_TREES:-0}
DGEMMA_MODEL=${DGEMMA_MODEL:-openai/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4}
DGEMMA_NUM_THREADS=${DGEMMA_NUM_THREADS:-12}
DGEMMA_MAX_CONCURRENT=${DGEMMA_MAX_CONCURRENT:-16}
DGEMMA_BATCH_SIZE=${DGEMMA_BATCH_SIZE:-8}
DGEMMA_GPUS=${DGEMMA_GPUS:-0,1,2,3}
DGEMMA_PORT_BASE=${DGEMMA_PORT_BASE:-8004}
DGEMMA_API_BASES=${DGEMMA_API_BASES:-}
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
  for root in "$OUT_ROOT"/servers/*; do
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

wait_for_source_run() {
  if [[ ! -f "$WAIT_JOB_ROOT/manifest.json" ]]; then
    log "No wait manifest at $WAIT_JOB_ROOT; continuing immediately"
    return 0
  fi
  log "Waiting for source launcher to finish: $WAIT_JOB_ROOT"
  while true; do
    local status_json
    status_json=$("$PY" scripts/long_job.py status --job-root "$WAIT_JOB_ROOT" 2>&1)
    printf '%s\n' "$status_json" > "$OUT_ROOT/logs/source_status_latest.json"
    if ! printf '%s\n' "$status_json" | "$PY" -c 'import json,sys; print(int(bool(json.load(sys.stdin).get("running"))))' >/tmp/gemma4_gold_to_dgemma_running.$$ 2>/dev/null; then
      log "Could not parse source status; retrying"
      sleep 300
      continue
    fi
    local running
    running=$(cat /tmp/gemma4_gold_to_dgemma_running.$$)
    rm -f /tmp/gemma4_gold_to_dgemma_running.$$
    if [[ "$running" == "0" ]]; then
      log "Source launcher inactive"
      return 0
    fi
    sleep 300
  done
}

check_source_artifacts() {
  local missing=0
  local leaf leafq f g
  for leaf in ${LEAF_QS//,/ }; do
    leafq=$(printf 'leafq%03d' "$leaf")
    f="$SOURCE_GRID/dspy/$leafq/iter_01_train_f/f_qsentence_dspy_iter_01"
    g="$SOURCE_GRID/dspy/$leafq/iter_02_train_g/g_qsentence_dspy_iter_02.json"
    if [[ ! -e "$f" ]]; then
      log "MISSING Gemma4-trained f artifact for $leafq: $f"
      missing=1
    fi
    if [[ ! -e "$g" ]]; then
      log "MISSING Gemma4-trained g artifact for $leafq: $g"
      missing=1
    fi
  done
  return "$missing"
}

start_dgemma_workers() {
  local gpus=()
  local api_bases=()
  local idx=0
  local gpu port api_base
  IFS="," read -r -a gpus <<< "$DGEMMA_GPUS"
  log "Starting DiffusionGemma workers on GPUs $DGEMMA_GPUS from port $DGEMMA_PORT_BASE"
  for gpu in "${gpus[@]}"; do
    gpu=${gpu//[[:space:]]/}
    [[ -n "$gpu" ]] || continue
    port=$((DGEMMA_PORT_BASE + idx))
    api_base="http://localhost:${port}/v1"
    ./scripts/start_diffusiongemma_qsentence_worker.sh "$gpu" "$port" 8 0.75 8192 32768 >/dev/null
    api_bases+=("$api_base")
    idx=$((idx + 1))
  done
  if (( ${#api_bases[@]} == 0 )); then
    log "No DiffusionGemma workers requested; DGEMMA_GPUS=$DGEMMA_GPUS"
    return 1
  fi
  for api_base in "${api_bases[@]}"; do
    wait_for_server "$api_base" 1200 || return 1
  done
  if [[ -z "$DGEMMA_API_BASES" ]]; then
    DGEMMA_API_BASES=$(IFS=,; printf "%s" "${api_bases[*]}")
  fi
  export DGEMMA_API_BASES
}

run_transfer_eval() {
  log "START dgemma_runtime_gemma4_gold_fg leafs=$LEAF_QS max_eval_trees=$MAX_EVAL_TREES"
  TT_DSPY_DROP_RESPONSE_FORMAT=1 TT_DSPY_WRAP_BARE_FIELD_OUTPUT=1 TT_SKIP_FULL_TREE_TRACES=1 \
    "$PY" scripts/run_manifesto_qsentence_dspy_ladder.py \
      --fg-grid-dir "$FULL_GRID" \
      --output-dir "$OUT_ROOT/dgemma_runtime_gemma4_gold_fg" \
      --leaf-qsentences "$LEAF_QS" \
      --max-iterations 0 \
      --initial-f-degree 2 \
      --initial-g-degree 2 \
      --stage-naming powers \
      --target-dimensions all \
      --max-eval-trees "$MAX_EVAL_TREES" \
      --eval-sample-seed 20260621 \
      --dspy-optimizer gepa \
      --dspy-budget light \
      --dspy-max-train-records 2048 \
      --dspy-initial-f-artifact-template "$SOURCE_GRID/dspy/{leafq}/iter_01_train_f/f_qsentence_dspy_iter_01" \
      --dspy-initial-g-artifact-template "$SOURCE_GRID/dspy/{leafq}/iter_02_train_g/g_qsentence_dspy_iter_02.json" \
      --fail-on-row-error \
      --dspy-model "$DGEMMA_MODEL" \
      --dspy-api-base "$DGEMMA_API_BASES" \
      --dspy-batch-routing-policy affinity_load_aware \
      --dspy-num-threads "$DGEMMA_NUM_THREADS" \
      --dspy-batch-max-concurrent "$DGEMMA_MAX_CONCURRENT" \
      --dspy-batch-size "$DGEMMA_BATCH_SIZE" \
      --dspy-lm-context-tokens 32768 \
      --verbose \
      >"$OUT_ROOT/logs/dgemma_runtime_gemma4_gold_fg.log" 2>&1
  local status=$?
  echo "$status" > "$OUT_ROOT/logs/dgemma_runtime_gemma4_gold_fg.status"
  log "END dgemma_runtime_gemma4_gold_fg status=$status log=$OUT_ROOT/logs/dgemma_runtime_gemma4_gold_fg.log"
  return "$status"
}

monitor_quality_failures() {
  local log_file="$OUT_ROOT/logs/dgemma_runtime_gemma4_gold_fg.log"
  local qfail="$OUT_ROOT/logs/quality_failure.txt"
  local match
  while kill -0 "$1" >/dev/null 2>&1; do
    if [[ -f "$log_file" ]] && match=$(rg -a -n -m 1 "$QUALITY_FAILURE_PATTERN" "$log_file" 2>/dev/null); then
      printf "dgemma_runtime_gemma4_gold_fg:%s\n" "$match" > "$qfail"
      log "FAILFAST quality failure in transfer eval: ${match:0:500}"
      kill "$1" >/dev/null 2>&1 || true
      return 97
    fi
    sleep 5
  done
}

write_reports() {
  log "START reports"
  local report_status=0
  "$PY" scripts/compare_manifesto_qsentence_substrates.py \
    dgemma_fixed="$SOURCE_RUN_ROOT/dgemma_fixed_leafgrid" \
    gemma4_fixed="$SOURCE_RUN_ROOT/gemma4_fixed_leafgrid" \
    dgemma_gemma4_gold_fg="$OUT_ROOT/dgemma_runtime_gemma4_gold_fg" \
    --output-dir "$OUT_ROOT/reports/substrate_comparison" \
    >>"$OUT_ROOT/logs/reports.log" 2>&1 || report_status=$?
  "$PY" scripts/summarize_manifesto_qsentence_per_dimension.py \
    dgemma_fixed="$SOURCE_RUN_ROOT/dgemma_fixed_leafgrid" \
    gemma4_fixed="$SOURCE_RUN_ROOT/gemma4_fixed_leafgrid" \
    dgemma_gemma4_gold_fg="$OUT_ROOT/dgemma_runtime_gemma4_gold_fg" \
    --output-dir "$OUT_ROOT/reports/per_dimension" \
    >>"$OUT_ROOT/logs/reports.log" 2>&1 || report_status=$?
  log "END reports status=$report_status"
  return "$report_status"
}

log "Output root: $OUT_ROOT"
log "Source grid: $SOURCE_GRID"
wait_for_source_run
if ! check_source_artifacts; then
  log "Source Gemma4 artifacts incomplete; aborting transfer experiment"
  exit 2
fi
log "Source Gemma4 f/g artifacts present"

stop_model_servers
start_dgemma_workers || exit 1
run_transfer_eval &
run_pid=$!
monitor_quality_failures "$run_pid" &
monitor_pid=$!
wait "$run_pid"
run_status=$?
kill "$monitor_pid" >/dev/null 2>&1 || true
wait "$monitor_pid" >/dev/null 2>&1 || true

if [[ -f "$OUT_ROOT/logs/quality_failure.txt" ]]; then
  log "SKIP reports because transfer eval had a quality failure"
  stop_model_servers
  exit 1
fi
if [[ "$run_status" -ne 0 ]]; then
  log "SKIP reports because transfer eval failed: status=$run_status"
  stop_model_servers
  exit "$run_status"
fi

write_reports
report_status=$?
stop_model_servers
log "complete transfer_status=$run_status report_status=$report_status"
exit "$report_status"
