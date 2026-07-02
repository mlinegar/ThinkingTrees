#!/usr/bin/env bash
# FNO leaf=1 sweep over the 6 Benoit expert dimensions (q-sentence "from sentence
# scores" arm). Broadcast doc-level Benoit supervision; root reconstruction =
# external expert metric. Economic runs FIRST (the "min" number), then the rest.
# Sequential to avoid CPU-embedding contention. One scalar head per dim.
set -u
ROOT=/home/mlinegar/ThinkingTrees
cd "$ROOT"
OUT=outputs/benoit_fno_leaf1
mkdir -p "$OUT"
LOG="$OUT/runner.log"
GPU=${FNO_GPU:-0}
EPOCHS=${FNO_EPOCHS:-8}
GRID=${BENOIT_GRID:-outputs/benoit_qsentence_grid_full}

log() { echo "[benoit-fno-l1 $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

DIMS=(economic social immigration eu environment decentralization)
log "start: leaf=1 dims=${DIMS[*]} epochs=$EPOCHS gpu=$GPU grid=$GRID"

for dim in "${DIMS[@]}"; do
  ddir="$OUT/$dim"
  if [ -f "$ddir/fno/leafq001/iteration_history.json" ]; then
    log "$dim already done, skipping"; continue
  fi
  log "=== $dim: starting ==="
  CUDA_VISIBLE_DEVICES=$GPU ./venv/bin/python scripts/run_manifesto_qsentence_dspy_ladder.py \
    --family fno --embedding-backend local-hf \
    --embedding-model /mnt/data/models/google/embeddinggemma-300m \
    --embedding-device cuda --embedding-batch-size 32 \
    --fg-grid-dir "$GRID" \
    --leaf-qsentences "1" --max-iterations 2 --fno-epochs "$EPOCHS" \
    --fno-batch-size 16 --fno-learning-rate 3e-3 \
    --fno-target-dimension "$dim" \
    --output-dir "$ddir" >> "$ddir.log" 2>&1 \
    && log "=== $dim: done ===" || log "WARN: $dim FAILED (see $ddir.log)"
done
log "all Benoit dimensions complete (leaf=1)"
