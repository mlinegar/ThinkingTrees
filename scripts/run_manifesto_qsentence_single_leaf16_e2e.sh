#!/usr/bin/env bash
set -euo pipefail

cd /home/mlinegar/ThinkingTrees

PY="${PY:-./venv/bin/python}"
LEAF_Q="${LEAF_Q:-16}"
LEAFQ="$(printf "leafq%03d" "$LEAF_Q")"
OUT_ROOT="${1:-outputs/manifesto_qsentence_single_good_leaf16_e2e_$(date +%Y%m%d_%H%M%S)}"
FULL_GRID="${FULL_GRID:-outputs/manifesto_qsentence_dspy_labeled_grid}"
GEMMA4_MODEL="${GEMMA4_MODEL:-openai/nvidia/Gemma-4-31B-IT-NVFP4}"
DGEMMA_MODEL="${DGEMMA_MODEL:-openai/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4}"

QUALITY_REGEX="${QUALITY_REGEX:-ServerDisconnectedError|Request dspy_batch_.* failed|AdapterParseError|Adapter JSONAdapter failed|LM response cannot be serialized|ERROR dspy|Traceback|Q-sentence g call produced no valid compact state|Q-sentence g call produced no parseable completion|Q-sentence compact target scorer call failed|no parseable completion|row error|Failing because|ClientConnectorError|Cannot connect to host|Internal Server Error|Default vLLM sampling parameters|OutOfMemoryError|CUDA out of memory|HTTP [45][0-9][0-9]|context length|maximum context|maximum prompt}"

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/reports" "$OUT_ROOT/servers"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$OUT_ROOT/logs/campaign.log"
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
    printf '%s:%s\n' "$name" "$match" > "$OUT_ROOT/logs/quality_failure.txt"
    log "FAIL quality failure in $name: ${match:0:500}"
    return 97
  fi
}

start_gemma4_if_needed() {
  if curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1; then
    log "Gemma4 server already ready on port 8000"
    return 0
  fi
  log "Starting Gemma4 all4 server on port 8000"
  "$PY" scripts/long_job.py launch \
    --name manifesto_single_leaf16_gemma4_all4 \
    --job-root "$OUT_ROOT/servers/gemma4_all4" \
    --cwd /home/mlinegar/ThinkingTrees \
    --replace-existing \
    -- ./scripts/start_vllm.sh gemma-4-31b-it-nvfp4 \
      --port 8000 --cuda-devices 0,1,2,3 --tensor-parallel 4 \
      --generation-config vllm >/dev/null
  wait_server http://localhost:8000/v1 1800
}

run_gemma4_leaf() {
  local log_file="$OUT_ROOT/logs/gemma4_${LEAFQ}.log"
  log "START Gemma4 full qsentence ${LEAFQ}"
  TT_DSPY_SKIP_COMPILE_IF_BASE_SCORE_AT_LEAST=0.99999 \
  TT_DSPY_BASE_SCORE_MAX_EXAMPLES=256 \
  TT_DSPY_OPTIMIZER_VALSET_MAX_EXAMPLES=256 \
  TT_DSPY_DROP_RESPONSE_FORMAT=0 \
  TT_DSPY_WRAP_BARE_FIELD_OUTPUT=1 \
  TT_SKIP_FULL_TREE_TRACES=1 \
  "$PY" scripts/run_manifesto_qsentence_dspy_ladder.py \
    --fg-grid-dir "$FULL_GRID" \
    --output-dir "$OUT_ROOT/gemma4_full_${LEAFQ}" \
    --leaf-qsentences "$LEAF_Q" \
    --max-iterations 2 \
    --target-dimensions all \
    --max-eval-trees 0 \
    --eval-sample-seed 20260621 \
    --dspy-optimizer gepa \
    --dspy-budget light \
    --dspy-max-train-records 2048 \
    --dspy-g-direct-parse-reward-weight 0.75 \
    --dspy-g-f-proxy-reward-weight 0.25 \
    --dspy-g-fail-fast-on-invalid-state \
    --fail-on-row-error \
    --dspy-model "$GEMMA4_MODEL" \
    --dspy-api-base http://localhost:8000/v1 \
    --dspy-batch-routing-policy affinity_load_aware \
    --dspy-num-threads 128 \
    --dspy-batch-max-concurrent 128 \
    --dspy-batch-size 64 \
    --dspy-lm-context-tokens 32768 \
    --verbose >"$log_file" 2>&1
  check_quality "gemma4_${LEAFQ}" "$log_file"
  log "END Gemma4 full qsentence ${LEAFQ}"
}

start_dgemma_workers() {
  log "Starting DiffusionGemma workers on GPUs 0,1,2,3"
  stop_all_servers
  for g in 0 1 2 3; do
    local port=$((8004 + g))
    ./scripts/start_diffusiongemma_qsentence_worker.sh "$g" "$port" 8 0.75 8192 32768 >/dev/null
  done
  for port in 8004 8005 8006 8007; do
    wait_server "http://localhost:${port}/v1" 1800
  done
}

run_dgemma_transfer() {
  local log_file="$OUT_ROOT/logs/dgemma_transfer_${LEAFQ}.log"
  local f_artifact="$OUT_ROOT/gemma4_full_${LEAFQ}/dspy/${LEAFQ}/iter_01_train_f/f_qsentence_dspy_iter_01"
  local g_artifact="$OUT_ROOT/gemma4_full_${LEAFQ}/dspy/${LEAFQ}/iter_02_train_g/g_qsentence_dspy_iter_02.json"
  if [ ! -e "$f_artifact" ] || [ ! -e "$g_artifact" ]; then
    log "FAIL missing Gemma4 f/g artifacts for DGemma transfer"
    return 2
  fi
  log "START DGemma transfer ${LEAFQ} using Gemma4-trained f/g"
  TT_DSPY_DROP_RESPONSE_FORMAT=1 \
  TT_DSPY_WRAP_BARE_FIELD_OUTPUT=1 \
  TT_SKIP_FULL_TREE_TRACES=1 \
  "$PY" scripts/run_manifesto_qsentence_dspy_ladder.py \
    --fg-grid-dir "$FULL_GRID" \
    --output-dir "$OUT_ROOT/dgemma_runtime_gemma4_fg_${LEAFQ}" \
    --leaf-qsentences "$LEAF_Q" \
    --max-iterations 0 \
    --initial-f-degree 2 \
    --initial-g-degree 2 \
    --stage-naming powers \
    --target-dimensions all \
    --max-eval-trees 0 \
    --eval-sample-seed 20260621 \
    --dspy-optimizer gepa \
    --dspy-budget light \
    --dspy-max-train-records 2048 \
    --dspy-initial-f-artifact "$f_artifact" \
    --dspy-initial-g-artifact "$g_artifact" \
    --dspy-g-fail-fast-on-invalid-state \
    --fail-on-row-error \
    --dspy-model "$DGEMMA_MODEL" \
    --dspy-api-base http://localhost:8004/v1,http://localhost:8005/v1,http://localhost:8006/v1,http://localhost:8007/v1 \
    --dspy-batch-routing-policy affinity_load_aware \
    --dspy-num-threads 32 \
    --dspy-batch-max-concurrent 32 \
    --dspy-batch-size 16 \
    --dspy-lm-context-tokens 32768 \
    --verbose >"$log_file" 2>&1
  check_quality "dgemma_transfer_${LEAFQ}" "$log_file"
  log "END DGemma transfer ${LEAFQ}"
}

write_reports() {
  log "START reports"
  "$PY" scripts/compare_manifesto_qsentence_substrates.py \
    "gemma4_single=${OUT_ROOT}/gemma4_full_${LEAFQ}" \
    "dgemma_gemma4_fg_single=${OUT_ROOT}/dgemma_runtime_gemma4_fg_${LEAFQ}" \
    dgemma_sampled=outputs/manifesto_qsentence_sampled_dgemma_leaf_sweep_20260623_082200 \
    fno_full_leafgrid=outputs/manifesto_qsentence_fno_embeddinggemma_full_leafgrid \
    --output-dir "$OUT_ROOT/reports/substrate_comparison" \
    >"$OUT_ROOT/logs/reports.log" 2>&1
  "$PY" scripts/summarize_manifesto_qsentence_per_dimension.py \
    "gemma4_single=${OUT_ROOT}/gemma4_full_${LEAFQ}" \
    "dgemma_gemma4_fg_single=${OUT_ROOT}/dgemma_runtime_gemma4_fg_${LEAFQ}" \
    dgemma_sampled=outputs/manifesto_qsentence_sampled_dgemma_leaf_sweep_20260623_082200 \
    fno_full_leafgrid=outputs/manifesto_qsentence_fno_embeddinggemma_full_leafgrid \
    --output-dir "$OUT_ROOT/reports/per_dimension" \
    >>"$OUT_ROOT/logs/reports.log" 2>&1
  log "END reports"
}

log "Output root: $OUT_ROOT"
test -f "$FULL_GRID/$LEAFQ/labeled_trees.jsonl"
start_gemma4_if_needed
run_gemma4_leaf
start_dgemma_workers
run_dgemma_transfer
write_reports
log "complete: $OUT_ROOT"
