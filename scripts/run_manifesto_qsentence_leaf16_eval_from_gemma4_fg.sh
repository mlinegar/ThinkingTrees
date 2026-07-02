#!/usr/bin/env bash
set -euo pipefail

cd /home/mlinegar/ThinkingTrees

PY="${PY:-./venv/bin/python}"
LEAF_Q="${LEAF_Q:-16}"
LEAFQ="$(printf "leafq%03d" "$LEAF_Q")"
SOURCE_ROOT="${SOURCE_ROOT:-outputs/manifesto_qsentence_single_good_leaf16_e2e_failfast_20260623_1920}"
OUT_ROOT="${1:-outputs/manifesto_qsentence_leaf16_eval_from_gemma4_fg_$(date +%Y%m%d_%H%M%S)}"
FULL_GRID="${FULL_GRID:-outputs/manifesto_qsentence_dspy_labeled_grid}"
GEMMA4_MODEL="${GEMMA4_MODEL:-openai/nvidia/Gemma-4-31B-IT-NVFP4}"
DGEMMA_MODEL="${DGEMMA_MODEL:-openai/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4}"

F_ARTIFACT="${F_ARTIFACT:-${SOURCE_ROOT}/gemma4_full_${LEAFQ}/dspy/${LEAFQ}/iter_01_train_f/f_qsentence_dspy_iter_01}"
G_ARTIFACT="${G_ARTIFACT:-${SOURCE_ROOT}/gemma4_full_${LEAFQ}/dspy/${LEAFQ}/iter_02_train_g/g_qsentence_dspy_iter_02.json}"
QUALITY_REGEX="${QUALITY_REGEX:-ServerDisconnectedError|Request dspy_batch_.* failed|AdapterParseError|Adapter JSONAdapter failed|LM response cannot be serialized|ERROR dspy|Traceback|Q-sentence g call produced no valid compact state|Q-sentence g call produced no parseable completion|Q-sentence compact target scorer call failed|no parseable completion|row error|Failing because|ClientConnectorError|Cannot connect to host|Internal Server Error|Default vLLM sampling parameters|OutOfMemoryError|CUDA out of memory|HTTP [45][0-9][0-9]|context length|maximum context|maximum prompt|fail-fast enabled}"

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/reports" "$OUT_ROOT/servers"

log() {
  printf '[%s] %s
' "$(date -Is)" "$*" | tee -a "$OUT_ROOT/logs/campaign.log"
}

stop_job() {
  local root="$1"
  if [ -f "$root/manifest.json" ]; then
    "$PY" scripts/long_job.py stop --job-root "$root" >/dev/null 2>&1 || true
  fi
}

stop_all_servers() {
  ./scripts/stop_small_servers.sh --all >/dev/null 2>&1 || true
  stop_job "$OUT_ROOT/servers/gemma4_all4"
  stop_job outputs/manifesto_qsentence_followon_all_missing_20260623_182238/servers/gemma4_all4
  for g in 0 1 2 3; do
    stop_job "outputs/diffusiongemma_qsentence_worker_gpu${g}"
    stop_job "outputs/gemma4_qsentence_server_gpu${g}_launcher"
  done
  sleep 8
}

cleanup() {
  stop_all_servers
}
trap cleanup EXIT

wait_server() {
  local base="$1"
  local seconds="${2:-1800}"
  local deadline=$((SECONDS + seconds))
  until curl -fsS "${base}/models" >/dev/null 2>&1; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      log "FAIL server not ready: $base"
      return 1
    fi
    sleep 10
  done
  log "READY $base"
}

check_quality() {
  local name="$1"
  local file="$2"
  local match
  if match="$(rg -a -n -m 1 "$QUALITY_REGEX" "$file" 2>/dev/null)"; then
    printf '%s:%s
' "$name" "$match" > "$OUT_ROOT/logs/quality_failure.txt"
    log "FAIL quality failure in $name: ${match:0:500}"
    return 97
  fi
}

start_gemma4() {
  log "Starting clean Gemma4 all4 server on port 8000"
  "$PY" scripts/long_job.py launch     --name manifesto_leaf16_eval_gemma4_all4     --job-root "$OUT_ROOT/servers/gemma4_all4"     --cwd /home/mlinegar/ThinkingTrees     --replace-existing     -- ./scripts/start_vllm.sh gemma-4-31b-it-nvfp4       --port 8000 --cuda-devices 0,1,2,3 --tensor-parallel 4       --generation-config vllm >/dev/null
  wait_server http://localhost:8000/v1 1800
}

run_eval() {
  local label="$1"
  local model="$2"
  local api_base="$3"
  local output_dir="$4"
  local threads="$5"
  local concurrent="$6"
  local batch_size="$7"
  local drop_response_format="$8"
  local log_file="$OUT_ROOT/logs/${label}_${LEAFQ}.log"
  log "START ${label} eval-only ${LEAFQ} from Gemma4-trained f/g"
  TT_DSPY_SKIP_COMPILE_IF_BASE_SCORE_AT_LEAST=0.99999   TT_DSPY_BASE_SCORE_MAX_EXAMPLES=256   TT_DSPY_OPTIMIZER_VALSET_MAX_EXAMPLES=256   TT_DSPY_DROP_RESPONSE_FORMAT="$drop_response_format"   TT_DSPY_WRAP_BARE_FIELD_OUTPUT=1   TT_SKIP_FULL_TREE_TRACES=1   "$PY" scripts/run_manifesto_qsentence_dspy_ladder.py     --fg-grid-dir "$FULL_GRID"     --output-dir "$output_dir"     --leaf-qsentences "$LEAF_Q"     --max-iterations 0     --initial-f-degree 2     --initial-g-degree 2     --stage-naming powers     --target-dimensions all     --max-eval-trees 0     --eval-sample-seed 20260621     --dspy-optimizer gepa     --dspy-budget light     --dspy-max-train-records 2048     --dspy-initial-f-artifact "$F_ARTIFACT"     --dspy-initial-g-artifact "$G_ARTIFACT"     --dspy-g-fail-fast-on-invalid-state     --fail-on-row-error     --dspy-model "$model"     --dspy-api-base "$api_base"     --dspy-batch-routing-policy affinity_load_aware     --dspy-num-threads "$threads"     --dspy-batch-max-concurrent "$concurrent"     --dspy-batch-size "$batch_size"     --dspy-lm-context-tokens 32768     --verbose >"$log_file" 2>&1
  check_quality "$label" "$log_file"
  log "END ${label} eval-only ${LEAFQ}"
}

start_dgemma_workers() {
  log "Starting clean DiffusionGemma workers on GPUs 0,1,2,3"
  for g in 0 1 2 3; do
    local port=$((8004 + g))
    ./scripts/start_diffusiongemma_qsentence_worker.sh "$g" "$port" 8 0.75 8192 32768 >/dev/null
  done
  for port in 8004 8005 8006 8007; do
    wait_server "http://localhost:${port}/v1" 1800
  done
}

write_reports() {
  log "START reports"
  "$PY" scripts/compare_manifesto_qsentence_substrates.py     "gemma4_eval=${OUT_ROOT}/gemma4_eval_gemma4_fg_${LEAFQ}"     "dgemma_eval_gemma4_fg=${OUT_ROOT}/dgemma_eval_gemma4_fg_${LEAFQ}"     dgemma_sampled=outputs/manifesto_qsentence_sampled_dgemma_leaf_sweep_20260623_082200     fno_full_leafgrid=outputs/manifesto_qsentence_fno_embeddinggemma_full_leafgrid     --output-dir "$OUT_ROOT/reports/substrate_comparison"     >"$OUT_ROOT/logs/reports.log" 2>&1 || log "WARN substrate comparison report failed"
  "$PY" scripts/summarize_manifesto_qsentence_per_dimension.py     "gemma4_eval=${OUT_ROOT}/gemma4_eval_gemma4_fg_${LEAFQ}"     "dgemma_eval_gemma4_fg=${OUT_ROOT}/dgemma_eval_gemma4_fg_${LEAFQ}"     dgemma_sampled=outputs/manifesto_qsentence_sampled_dgemma_leaf_sweep_20260623_082200     fno_full_leafgrid=outputs/manifesto_qsentence_fno_embeddinggemma_full_leafgrid     --output-dir "$OUT_ROOT/reports/per_dimension"     >>"$OUT_ROOT/logs/reports.log" 2>&1 || log "WARN per-dimension report failed"
  log "END reports"
}

log "Output root: $OUT_ROOT"
test -d "$F_ARTIFACT"
test -f "$G_ARTIFACT"
test -f "$FULL_GRID/$LEAFQ/labeled_trees.jsonl"

stop_all_servers
start_gemma4
run_eval "gemma4" "$GEMMA4_MODEL" "http://localhost:8000/v1" "$OUT_ROOT/gemma4_eval_gemma4_fg_${LEAFQ}" 128 128 64 0
stop_all_servers
start_dgemma_workers
run_eval "dgemma" "$DGEMMA_MODEL" "http://localhost:8004/v1,http://localhost:8005/v1,http://localhost:8006/v1,http://localhost:8007/v1" "$OUT_ROOT/dgemma_eval_gemma4_fg_${LEAFQ}" 32 32 16 1
write_reports
log "complete: $OUT_ROOT"
