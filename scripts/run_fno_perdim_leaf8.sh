#!/usr/bin/env bash
# FNO per-dimension grid at the leaf-8 sweet spot (rile + domain_1..7).
# One scalar-head run per dimension (retargets node.score), respecting the
# out_channels=1 channel invariant. Sequential to avoid CPU-embedding contention
# with the live Gemma-4 ladder. Each dimension writes its own iteration_history.
set -u

ROOT=/home/mlinegar/ThinkingTrees
cd "$ROOT"
OUT=outputs/manifesto_qsentence_fno_perdim_leaf8
mkdir -p "$OUT"
LOG="$OUT/runner.log"
GPU=${FNO_GPU:-3}
EPOCHS=${FNO_EPOCHS:-4}

log() { echo "[fno-perdim $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

DIMS=(rile domain_1 domain_2 domain_3 domain_4 domain_5 domain_6 domain_7)
log "start: leaf=8 dims=${DIMS[*]} epochs=$EPOCHS gpu=$GPU"

for dim in "${DIMS[@]}"; do
  ddir="$OUT/$dim"
  if [ -f "$ddir/fno/leafq008/iteration_history.json" ]; then
    log "$dim already done, skipping"
    continue
  fi
  log "=== $dim: starting ==="
  CUDA_VISIBLE_DEVICES=$GPU ./venv/bin/python scripts/run_manifesto_qsentence_dspy_ladder.py \
    --family fno \
    --embedding-backend local-hf \
    --embedding-model /mnt/data/models/google/embeddinggemma-300m \
    --embedding-device cpu \
    --embedding-batch-size 64 \
    --fg-grid-dir outputs/manifesto_qsentence_dspy_labeled_grid \
    --leaf-qsentences "8" \
    --max-iterations 2 \
    --fno-epochs "$EPOCHS" \
    --fno-batch-size 16 \
    --fno-learning-rate 3e-3 \
    --fno-target-dimension "$dim" \
    --output-dir "$ddir" \
    --verbose >> "$ddir.log" 2>&1 \
    && log "=== $dim: done ===" \
    || log "WARN: $dim FAILED (see $ddir.log)"
done

log "all dimensions complete"
