#!/usr/bin/env bash
# Day-2 queue: complete the substrate-comparison matrix after the overnight
# v2 orchestrator finishes.
#
#   1. wait for run_overnight_substrate_comparison_v2.sh to exit
#   2. swap fleet -> 4x Gemma-4-31B and run the FULL-grid coarse cells
#      (leaves 16,8,4,2 — cheap first; leaf=1 stays smoke-only for 31B)
#   3. restore the DiffusionGemma fleet
#   4. final comparator over every run to date
#
# The FNO leaf-axis runs separately on CPU (run_day2_fno_leafgrid.sh).
#
# Launch:
#   ./venv/bin/python scripts/long_job.py launch \
#     --name day2_substrate_comparison \
#     --job-root outputs/day2_substrate_comparison_launcher \
#     --cwd /home/mlinegar/ThinkingTrees --replace-existing \
#     -- bash scripts/run_day2_substrate_comparison.sh
set -uo pipefail
cd /home/mlinegar/ThinkingTrees

PY=./venv/bin/python
FULL_GRID=outputs/manifesto_qsentence_dspy_labeled_grid
GEMMA4_FLEET="http://localhost:8010/v1,http://localhost:8011/v1,http://localhost:8012/v1,http://localhost:8013/v1"

log() { echo "[day2 $(date -u +%H:%M:%S)] $*"; }

wait_for_server() { # url, attempts
  local url=$1 attempts=${2:-90}
  for _ in $(seq 1 "$attempts"); do
    curl -s --max-time 3 "$url/models" >/dev/null && return 0
    sleep 10
  done
  return 1
}

ensure_dgemma_fleet() {
  local g p
  for g in 0 1 2 3; do
    p=$((8004 + g))
    if ! curl -s --max-time 3 "http://localhost:${p}/v1/models" >/dev/null; then
      log "starting dgemma worker gpu${g}:${p}"
      ./scripts/start_diffusiongemma_qsentence_worker.sh "$g" "$p" || true
    fi
  done
  for g in 0 1 2 3; do
    wait_for_server "http://localhost:$((8004 + g))/v1" || log "WARN: dgemma worker gpu${g} not up"
  done
}

log "waiting for overnight v2 orchestrator to finish"
while pgrep -f "run_overnight_substrate_comparison_v2.sh" >/dev/null 2>&1; do sleep 120; done
log "overnight v2 done"

log "swap fleet -> 4x Gemma-4-31B (ports 8010-8013)"
for g in 0 1 2 3; do
  $PY scripts/long_job.py stop --job-root "outputs/diffusiongemma_qsentence_worker_gpu${g}" || true
done
sleep 20
for g in 0 1 2 3; do
  $PY scripts/long_job.py launch \
    --name "gemma4_qsentence_server_gpu${g}" \
    --job-root "outputs/gemma4_qsentence_server_gpu${g}_launcher" \
    --cwd /home/mlinegar/ThinkingTrees \
    --replace-existing \
    -- ./scripts/start_vllm.sh gemma-4-31b-it-nvfp4 --port "$((8010 + g))" --cuda-devices "$g" \
    || log "WARN: gemma-4 server gpu${g} launch failed"
done

GEMMA4_UP=1
for g in 0 1 2 3; do
  wait_for_server "http://localhost:$((8010 + g))/v1" || { log "WARN: gemma-4 gpu${g} never came up"; GEMMA4_UP=0; }
done

if [ "$GEMMA4_UP" = 1 ]; then
  log "gemma-4 fleet up; FULL-grid coarse cells 16,8,4,2"
  TT_DSPY_DROP_RESPONSE_FORMAT=0 $PY scripts/run_manifesto_qsentence_dspy_ladder.py \
    --fg-grid-dir "$FULL_GRID" \
    --leaf-qsentences "16,8,4,2" \
    --max-iterations 2 \
    --target-dimensions all \
    --dspy-optimizer gepa --dspy-budget light \
    --dspy-max-train-records 2048 \
    --dspy-num-threads 640 \
    --dspy-batch-max-concurrent 1024 \
    --dspy-model "openai/nvidia/Gemma-4-31B-IT-NVFP4" \
    --dspy-api-base "$GEMMA4_FLEET" \
    --dspy-lm-context-tokens 32768 \
    --output-dir outputs/manifesto_qsentence_gemma4_full_coarse \
    --verbose \
    || log "WARN: gemma-4 full-coarse leg failed (continuing)"
else
  log "WARN: skipping gemma-4 full-coarse leg (fleet incomplete)"
fi

log "restore DiffusionGemma fleet"
for g in 0 1 2 3; do
  $PY scripts/long_job.py stop --job-root "outputs/gemma4_qsentence_server_gpu${g}_launcher" || true
done
sleep 20
ensure_dgemma_fleet

log "final day-2 comparator"
$PY scripts/compare_manifesto_qsentence_substrates.py \
  dgemma_full_leaf1=outputs/manifesto_qsentence_diffusiongemma_full_leaf1 \
  dgemma_full_leafgrid=outputs/manifesto_qsentence_diffusiongemma_full_leafgrid \
  gemma4_full_coarse=outputs/manifesto_qsentence_gemma4_full_coarse \
  gemma4_small=outputs/manifesto_qsentence_gemma4_small \
  fno_embeddinggemma_full=outputs/manifesto_qsentence_fno_embeddinggemma_full \
  fno_full_leafgrid=outputs/manifesto_qsentence_fno_embeddinggemma_full_leafgrid \
  fno_embeddinggemma_small=outputs/manifesto_qsentence_fno_embeddinggemma_small_fixed \
  --output-dir outputs/manifesto_qsentence_substrate_comparison_day2 \
  || log "WARN: comparator failed (some legs may be missing)"

log "day-2 queue complete"
