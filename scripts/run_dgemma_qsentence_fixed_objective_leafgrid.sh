#!/usr/bin/env bash
# Fixed-objective DiffusionGemma rerun on the full qsentence leaf grid.
# Uses the direct-parse g reward added after diagnosing proxy-only DGemma drift.
# Queue this after broader GPU jobs finish; it intentionally does not rerun Gemma4/FNO.

set -uo pipefail
cd /home/mlinegar/ThinkingTrees

PY=./venv/bin/python
FULL_GRID=${FULL_GRID:-outputs/manifesto_qsentence_dspy_labeled_grid}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-outputs/manifesto_dgemma_fixed_objective_${STAMP}}
RUN_DIR="$OUT_ROOT/dgemma_full_leafgrid_fixed_direct"
DGEMMA_DIRECT_WEIGHT=${DGEMMA_DIRECT_WEIGHT:-0.75}
DGEMMA_PROXY_WEIGHT=${DGEMMA_PROXY_WEIGHT:-0.25}
DGEMMA_FLEET="http://localhost:8004/v1,http://localhost:8005/v1,http://localhost:8006/v1,http://localhost:8007/v1"
export TT_DSPY_DROP_RESPONSE_FORMAT=1
export TT_SKIP_FULL_TREE_TRACES=1

log() { echo "[dgemma-fixed $(date -u +%H:%M:%S)] $*"; }

wait_for_server() {
  local url=$1 attempts=${2:-90}
  for _ in $(seq 1 "$attempts"); do
    curl -s --max-time 3 "$url/models" >/dev/null && return 0
    sleep 10
  done
  return 1
}

mkdir -p "$OUT_ROOT/reports"
log "fixed-objective output root: $OUT_ROOT"
log "reward weights: direct_parse=$DGEMMA_DIRECT_WEIGHT f_proxy=$DGEMMA_PROXY_WEIGHT"
log "stopping any leftover model workers"
./scripts/stop_small_servers.sh --all >/dev/null 2>&1 || true
for g in 0 1 2 3; do
  $PY scripts/long_job.py stop --job-root "outputs/diffusiongemma_qsentence_worker_gpu${g}" >/dev/null 2>&1 || true
  $PY scripts/long_job.py stop --job-root "outputs/gemma4_qsentence_server_gpu${g}_launcher" >/dev/null 2>&1 || true
done
sleep 8
log "ensuring DiffusionGemma fleet"

for g in 0 1 2 3; do
  p=$((8004 + g))
  if ! curl -s --max-time 3 "http://localhost:${p}/v1/models" >/dev/null; then
    ./scripts/start_diffusiongemma_qsentence_worker.sh "$g" "$p" || true
  fi
done
FLEET_UP=1
for g in 0 1 2 3; do
  wait_for_server "http://localhost:$((8004 + g))/v1" || { log "WARN: worker gpu${g} not up"; FLEET_UP=0; }
done
[ "$FLEET_UP" = 1 ] || log "WARN: proceeding with partial fleet"

log "DiffusionGemma full-grid leaves 16,8,4,2 with fixed direct-parse objective"
run_status=0
$PY scripts/run_manifesto_qsentence_dspy_ladder.py \
  --fg-grid-dir "$FULL_GRID" \
  --leaf-qsentences "16,8,4,2" \
  --max-iterations 2 \
  --target-dimensions all \
  --dspy-optimizer gepa --dspy-budget light \
  --dspy-max-train-records 2048 \
  --dspy-g-direct-parse-reward-weight "$DGEMMA_DIRECT_WEIGHT" \
  --dspy-g-f-proxy-reward-weight "$DGEMMA_PROXY_WEIGHT" \
  --dspy-num-threads 640 \
  --dspy-batch-max-concurrent 1024 \
  --dspy-model "openai/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4" \
  --dspy-api-base "$DGEMMA_FLEET" \
  --dspy-lm-context-tokens 32768 \
  --output-dir "$RUN_DIR" \
  --verbose \
  || { run_status=$?; log "WARN: dgemma fixed-objective leaf-grid failed status=$run_status"; }

log "fixed-objective reports"
$PY scripts/compare_manifesto_qsentence_substrates.py \
  dgemma_old=outputs/manifesto_qsentence_diffusiongemma_FULL218_leafgrid \
  dgemma_fixed="$RUN_DIR" \
  --output-dir "$OUT_ROOT/reports/substrate_comparison" \
  || log "WARN: substrate comparator failed"
$PY scripts/summarize_manifesto_qsentence_per_dimension.py \
  dgemma_old=outputs/manifesto_qsentence_diffusiongemma_FULL218_leafgrid \
  dgemma_fixed="$RUN_DIR" \
  --output-dir "$OUT_ROOT/reports/per_dimension" \
  || log "WARN: per-dimension report failed"

log "fixed-objective DGemma complete: $OUT_ROOT"
exit "$run_status"
