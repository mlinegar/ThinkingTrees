#!/usr/bin/env bash
# Overnight queue for the manifesto q-sentence substrate comparison
# (docs/manifesto_qsentence_substrate_comparison_runbook.md, full-grid phase).
#
# Sequence (fleet legs serialized; each leg tolerated to fail independently):
#   0. wait for any in-flight small runs to release the DiffusionGemma fleet
#   1. DiffusionGemma FULL grid (140/30/48) leaf=1            [fleet]
#   2. DiffusionGemma FULL grid leaves 2,4,8,16               [fleet]
#   3. swap GPU3 worker -> Gemma-4-31B on :8010; Gemma-4 SMOKE grid leaves 1-16
#   4. restore GPU3 DiffusionGemma worker; final comparator over all runs
#
# FNO full-grid leg runs as a SEPARATE CPU job (launched alongside this one).
#
# Launch via:
#   ./venv/bin/python scripts/long_job.py launch \
#     --name overnight_substrate_comparison \
#     --job-root outputs/overnight_substrate_comparison_launcher \
#     --cwd /home/mlinegar/ThinkingTrees --replace-existing \
#     -- bash scripts/run_overnight_substrate_comparison.sh
set -uo pipefail
cd /home/mlinegar/ThinkingTrees

PY=./venv/bin/python
FULL_GRID=outputs/manifesto_qsentence_dspy_labeled_grid
SMOKE_GRID=outputs/manifesto_qsentence_dspy_labeled_grid_smoke
FLEET="http://localhost:8004/v1,http://localhost:8005/v1,http://localhost:8006/v1,http://localhost:8007/v1"
DGEMMA_MODEL="openai/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4"
export TT_DSPY_DROP_RESPONSE_FORMAT=1

log() { echo "[overnight $(date -u +%H:%M:%S)] $*"; }

wait_for_no_process() { # pattern
  while pgrep -f "$1" >/dev/null 2>&1; do sleep 60; done
}

wait_for_server() { # url, attempts
  local url=$1 attempts=${2:-60}
  for _ in $(seq 1 "$attempts"); do
    curl -s --max-time 3 "$url/models" >/dev/null && return 0
    sleep 10
  done
  return 1
}

log "step 0: waiting for in-flight small runs to finish"
wait_for_no_process "output-dir outputs/manifesto_qsentence_diffusiongemma_small\b"
log "fleet free"

log "step 1: DiffusionGemma full grid leaf=1"
$PY scripts/run_manifesto_qsentence_dspy_ladder.py \
  --fg-grid-dir "$FULL_GRID" \
  --leaf-qsentences "1" \
  --max-iterations 2 \
  --target-dimensions all \
  --dspy-optimizer gepa --dspy-budget light \
  --dspy-max-train-records 2048 \
  --dspy-model "$DGEMMA_MODEL" \
  --dspy-api-base "$FLEET" \
  --dspy-lm-context-tokens 32768 \
  --output-dir outputs/manifesto_qsentence_diffusiongemma_full_leaf1 \
  --verbose \
  || log "WARN: step 1 failed (continuing)"

log "step 2: DiffusionGemma full grid leaves 2,4,8,16"
$PY scripts/run_manifesto_qsentence_dspy_ladder.py \
  --fg-grid-dir "$FULL_GRID" \
  --leaf-qsentences "2,4,8,16" \
  --max-iterations 2 \
  --target-dimensions all \
  --dspy-optimizer gepa --dspy-budget light \
  --dspy-max-train-records 2048 \
  --dspy-model "$DGEMMA_MODEL" \
  --dspy-api-base "$FLEET" \
  --dspy-lm-context-tokens 32768 \
  --output-dir outputs/manifesto_qsentence_diffusiongemma_full_leafgrid \
  --verbose \
  || log "WARN: step 2 failed (continuing)"

log "step 3: swap GPU3 -> Gemma-4-31B on :8010"
$PY scripts/long_job.py stop --job-root outputs/diffusiongemma_qsentence_worker_gpu3 || true
sleep 20
$PY scripts/long_job.py launch \
  --name gemma4_qsentence_server \
  --job-root outputs/gemma4_qsentence_server_launcher \
  --cwd /home/mlinegar/ThinkingTrees \
  --replace-existing \
  -- ./scripts/start_vllm.sh gemma-4-31b-it-nvfp4 --port 8010 --cuda-devices 3 \
  || log "WARN: gemma-4 server launch failed"

if wait_for_server "http://localhost:8010/v1" 60; then
  log "gemma-4 up; running smoke-grid leaves 1-16 (no response_format guard needed)"
  TT_DSPY_DROP_RESPONSE_FORMAT=0 $PY scripts/run_manifesto_qsentence_dspy_ladder.py \
    --fg-grid-dir "$SMOKE_GRID" \
    --leaf-qsentences "1,2,4,8,16" \
    --max-iterations 2 \
    --target-dimensions all \
    --dspy-optimizer gepa --dspy-budget light \
    --dspy-max-train-records 2048 \
    --dspy-model "openai/nvidia/Gemma-4-31B-IT-NVFP4" \
    --dspy-api-base "http://localhost:8010/v1" \
    --dspy-lm-context-tokens 32768 \
    --output-dir outputs/manifesto_qsentence_gemma4_small \
    --verbose \
    || log "WARN: gemma-4 leg failed (continuing)"
else
  log "WARN: gemma-4 server never came up; skipping leg A"
fi

log "step 4: restore GPU3 worker and run final comparator"
$PY scripts/long_job.py stop --job-root outputs/gemma4_qsentence_server_launcher || true
sleep 20
./scripts/start_diffusiongemma_qsentence_worker.sh 3 8007 || log "WARN: worker restore failed"

$PY scripts/compare_manifesto_qsentence_substrates.py \
  dgemma_full_leaf1=outputs/manifesto_qsentence_diffusiongemma_full_leaf1 \
  dgemma_full_leafgrid=outputs/manifesto_qsentence_diffusiongemma_full_leafgrid \
  fno_embeddinggemma_full=outputs/manifesto_qsentence_fno_embeddinggemma_full \
  gemma4_small=outputs/manifesto_qsentence_gemma4_small \
  dgemma_small_leaf1=outputs/manifesto_qsentence_diffusiongemma_small \
  fno_embeddinggemma_small=outputs/manifesto_qsentence_fno_embeddinggemma_small_fixed \
  --output-dir outputs/manifesto_qsentence_substrate_comparison_overnight \
  || log "WARN: comparator failed (some legs may be missing)"

log "overnight queue complete"
