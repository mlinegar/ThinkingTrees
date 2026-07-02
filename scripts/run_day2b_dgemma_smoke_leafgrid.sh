#!/usr/bin/env bash
# Day-2b: DiffusionGemma on the SMOKE grid leaves 2,4,8,16 — completes the
# matched pairs against Gemma-4's smoke run (leaves 1-16): every Gemma-4 cell
# then has a DiffusionGemma twin on the same bundle/split. Expected outcome
# per user hypothesis: diffusion slightly worse than the AR model at matched
# cells; this run is the evidence either way.
#
# Waits for the day-2 GPU queue (Gemma-4 full-coarse leg) to exit, since that
# owns the GPUs until it restores the DiffusionGemma fleet.
#
# Launch:
#   ./venv/bin/python scripts/long_job.py launch \
#     --name day2b_dgemma_smoke_leafgrid \
#     --job-root outputs/day2b_dgemma_smoke_leafgrid_launcher \
#     --cwd /home/mlinegar/ThinkingTrees --replace-existing \
#     -- bash scripts/run_day2b_dgemma_smoke_leafgrid.sh
set -uo pipefail
cd /home/mlinegar/ThinkingTrees

PY=./venv/bin/python
SMOKE_GRID=outputs/manifesto_qsentence_dspy_labeled_grid_smoke
DGEMMA_FLEET="http://localhost:8004/v1,http://localhost:8005/v1,http://localhost:8006/v1,http://localhost:8007/v1"
export TT_DSPY_DROP_RESPONSE_FORMAT=1

log() { echo "[day2b $(date -u +%H:%M:%S)] $*"; }

wait_for_server() {
  local url=$1 attempts=${2:-90}
  for _ in $(seq 1 "$attempts"); do
    curl -s --max-time 3 "$url/models" >/dev/null && return 0
    sleep 10
  done
  return 1
}

log "waiting for day-2 GPU queue to finish"
while pgrep -f "run_day2_substrate_comparison.sh" >/dev/null 2>&1; do sleep 120; done
log "day-2 done; ensuring DiffusionGemma fleet"

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

log "DiffusionGemma smoke-grid leaves 2,4,8,16"
$PY scripts/run_manifesto_qsentence_dspy_ladder.py \
  --fg-grid-dir "$SMOKE_GRID" \
  --leaf-qsentences "16,8,4,2" \
  --max-iterations 2 \
  --target-dimensions all \
  --dspy-optimizer gepa --dspy-budget light \
  --dspy-max-train-records 2048 \
  --dspy-num-threads 640 \
  --dspy-batch-max-concurrent 1024 \
  --dspy-model "openai/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4" \
  --dspy-api-base "$DGEMMA_FLEET" \
  --dspy-lm-context-tokens 32768 \
  --output-dir outputs/manifesto_qsentence_diffusiongemma_small_leafgrid \
  --verbose \
  || log "WARN: dgemma smoke leaf-grid failed"

log "matched-pairs comparator (smoke + full, both LLM substrates + FNO)"
$PY scripts/compare_manifesto_qsentence_substrates.py \
  dgemma_full_leaf1=outputs/manifesto_qsentence_diffusiongemma_full_leaf1 \
  dgemma_full_leafgrid=outputs/manifesto_qsentence_diffusiongemma_full_leafgrid \
  dgemma_small_leafgrid=outputs/manifesto_qsentence_diffusiongemma_small_leafgrid \
  gemma4_full_coarse=outputs/manifesto_qsentence_gemma4_full_coarse \
  gemma4_small=outputs/manifesto_qsentence_gemma4_small \
  fno_embeddinggemma_full=outputs/manifesto_qsentence_fno_embeddinggemma_full \
  fno_full_leafgrid=outputs/manifesto_qsentence_fno_embeddinggemma_full_leafgrid \
  fno_embeddinggemma_small=outputs/manifesto_qsentence_fno_embeddinggemma_small_fixed \
  --output-dir outputs/manifesto_qsentence_substrate_comparison_matched \
  || log "WARN: comparator failed (some legs may be missing)"

log "day-2b complete"
