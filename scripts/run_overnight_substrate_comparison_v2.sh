#!/usr/bin/env bash
# Overnight queue v2 — every stage uses ALL FOUR GPUs for serving consistency.
#
# Changes vs v1:
#   - assumes the small leaf=1 run has been stopped (kickoff script does it);
#     its iter-0/1 metrics live in step_checkpoints, and the full-grid run
#     supersedes iter-2.
#   - Gemma-4 stage runs FOUR servers (ports 8010-8013, one per GPU), not one.
#   - picks up the level-wave batched g eval (dspy_family fix, 2026-06-12).
#
# Launch via:
#   ./venv/bin/python scripts/long_job.py launch \
#     --name overnight_substrate_comparison \
#     --job-root outputs/overnight_substrate_comparison_launcher \
#     --cwd /home/mlinegar/ThinkingTrees --replace-existing \
#     -- bash scripts/run_overnight_substrate_comparison_v2.sh
set -uo pipefail
cd /home/mlinegar/ThinkingTrees

PY=./venv/bin/python
FULL_GRID=outputs/manifesto_qsentence_dspy_labeled_grid
SMOKE_GRID=outputs/manifesto_qsentence_dspy_labeled_grid_smoke
DGEMMA_FLEET="http://localhost:8004/v1,http://localhost:8005/v1,http://localhost:8006/v1,http://localhost:8007/v1"
GEMMA4_FLEET="http://localhost:8010/v1,http://localhost:8011/v1,http://localhost:8012/v1,http://localhost:8013/v1"
DGEMMA_MODEL="openai/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4"
export TT_DSPY_DROP_RESPONSE_FORMAT=1
export TT_SKIP_FULL_TREE_TRACES=1

log() { echo "[overnight2 $(date -u +%H:%M:%S)] $*"; }

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

ensure_dgemma_fleet

log "stage 2: DiffusionGemma full grid leaves 2,4,8,16 (all 4 GPUs)"
$PY scripts/run_manifesto_qsentence_dspy_ladder.py \
  --fg-grid-dir "$FULL_GRID" \
  --leaf-qsentences "2,4,8,16" \
  --max-iterations 2 \
  --target-dimensions all \
  --dspy-optimizer gepa --dspy-budget light \
  --dspy-max-train-records 2048 \
  --dspy-num-threads 640 \
  --dspy-batch-max-concurrent 1024 \
  --dspy-model "$DGEMMA_MODEL" \
  --dspy-api-base "$DGEMMA_FLEET" \
  --dspy-lm-context-tokens 32768 \
  --output-dir outputs/manifesto_qsentence_diffusiongemma_full_leafgrid \
  --verbose \
  || log "WARN: stage 2 failed (continuing)"

log "stage 3: swap fleet -> 4x Gemma-4-31B (ports 8010-8013, one per GPU)"
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
  log "gemma-4 fleet up; smoke-grid leaves 1-16 over all 4 GPUs"
  TT_DSPY_DROP_RESPONSE_FORMAT=0 $PY scripts/run_manifesto_qsentence_dspy_ladder.py \
    --fg-grid-dir "$SMOKE_GRID" \
    --leaf-qsentences "1,2,4,8,16" \
    --max-iterations 2 \
    --target-dimensions all \
    --dspy-optimizer gepa --dspy-budget light \
    --dspy-max-train-records 2048 \
  --dspy-num-threads 640 \
  --dspy-batch-max-concurrent 1024 \
    --dspy-model "openai/nvidia/Gemma-4-31B-IT-NVFP4" \
    --dspy-api-base "$GEMMA4_FLEET" \
    --dspy-lm-context-tokens 32768 \
    --output-dir outputs/manifesto_qsentence_gemma4_small \
    --verbose \
    || log "WARN: gemma-4 leg failed (continuing)"
else
  log "WARN: skipping gemma-4 leg (fleet incomplete)"
fi

log "stage 4: restore DiffusionGemma fleet; final comparator"
for g in 0 1 2 3; do
  $PY scripts/long_job.py stop --job-root "outputs/gemma4_qsentence_server_gpu${g}_launcher" || true
done
sleep 20
ensure_dgemma_fleet

$PY scripts/compare_manifesto_qsentence_substrates.py \
  dgemma_full_leaf1=outputs/manifesto_qsentence_diffusiongemma_full_leaf1 \
  dgemma_full_leafgrid=outputs/manifesto_qsentence_diffusiongemma_full_leafgrid \
  fno_embeddinggemma_full=outputs/manifesto_qsentence_fno_embeddinggemma_full \
  gemma4_small=outputs/manifesto_qsentence_gemma4_small \
  fno_embeddinggemma_small=outputs/manifesto_qsentence_fno_embeddinggemma_small_fixed \
  --output-dir outputs/manifesto_qsentence_substrate_comparison_overnight \
  || log "WARN: comparator failed (some legs may be missing)"

log "overnight v2 queue complete"
