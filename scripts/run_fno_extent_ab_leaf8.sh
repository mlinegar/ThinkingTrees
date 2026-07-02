#!/usr/bin/env bash
# FNO extent-latent A/B at leaf8, domain_4 (Economy).
#
# Tests whether the learned 'extent' latent (mass-aware general g) lets the FNO
# merge beat naive equal-averaging on the per-merge-node task. Three arms, all
# merge_mode=gated, same dim/leaf/seed:
#   baseline : extent OFF (the mass-blind gated merge — the current ceiling)
#   armA     : extent ON, neutral init, flat g-loss (pure laws; may collapse)
#   armB     : extent ON, additive init, depth x lopsided g-loss reweight (strength 4.0)
# Each arm: train (2 iters f->g) -> dump per-node g-states -> score merge-by-level.
# Win = armB learned_g wmae < equal_avg bar (~0.004), toward mass_wtd=0 ceiling.
# Mechanism finding = armA-vs-armB gap (how much init+reweight identifies the latent).
set -u

ROOT=/home/mlinegar/ThinkingTrees
cd "$ROOT"
OUT=${AB_ROOT:-outputs/fno_extent_ab_leaf8}
mkdir -p "$OUT"
LOG="$OUT/runner.log"
GPU=${FNO_GPU:-0}
EPOCHS=${FNO_EPOCHS:-8}
DIM=${FNO_DIM:-domain_4}
LEAF=8
GRID=outputs/manifesto_qsentence_dspy_labeled_grid
PY=./venv/bin/python

log() { echo "[fno-extent-ab $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# train_arm <name> <extra ladder flags...>
train_arm() {
  local name="$1"; shift
  local ddir="$OUT/$name"
  if [ -f "$ddir/fno/leafq00${LEAF}/iteration_history.json" ]; then
    log "$name already trained, skipping train"
  else
    log "=== $name: training ==="
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_manifesto_qsentence_dspy_ladder.py \
      --family fno \
      --embedding-backend local-hf \
      --embedding-model /mnt/data/models/google/embeddinggemma-300m \
      --embedding-device cuda \
      --embedding-batch-size 128 \
      --fg-grid-dir "$GRID" \
      --leaf-qsentences "$LEAF" \
      --max-iterations 2 \
      --fno-epochs "$EPOCHS" \
      --fno-batch-size 16 \
      --fno-learning-rate 3e-3 \
      --fno-merge-mode gated \
      --fno-target-dimension "$DIM" \
      "$@" \
      --output-dir "$ddir" \
      --verbose >> "$ddir.train.log" 2>&1 \
      && log "=== $name: trained ===" \
      || { log "WARN: $name TRAIN FAILED (see $ddir.train.log)"; return 1; }
  fi
}

# score_arm <name> <extent-flags for the dump...>
score_arm() {
  local name="$1"; shift
  local ddir="$OUT/$name"
  local states="$ddir/g_node_states_${DIM}_leaf${LEAF}.jsonl"
  log "=== $name: dumping per-node g-states ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/dump_fno_g_node_states.py \
    --run-dir "$ddir/fno" \
    --leaf-qsentences "$LEAF" \
    --fg-grid-dir "$GRID" \
    --target-dimension "$DIM" \
    --embedding-model /mnt/data/models/google/embeddinggemma-300m \
    --embedding-device cuda \
    --fno-merge-mode gated \
    "$@" \
    --out-jsonl "$states" >> "$ddir.score.log" 2>&1 \
    || { log "WARN: $name DUMP FAILED (see $ddir.score.log)"; return 1; }
  log "=== $name: scoring merge-by-level ==="
  $PY scripts/eval_qsentence_merge_by_level.py \
    --labeled-trees "$GRID/leafq00${LEAF}/labeled_trees.jsonl" \
    --split test \
    --g-states-jsonl "$states" \
    --lopsidedness-strength 4.0 \
    --out-json "$ddir/merge_by_level_${DIM}.json" >> "$ddir.score.log" 2>&1 \
    && log "=== $name: scored -> $ddir/merge_by_level_${DIM}.json ===" \
    || log "WARN: $name SCORE FAILED (see $ddir.score.log)"
}

log "start: dim=$DIM leaf=$LEAF epochs=$EPOCHS gpu=$GPU root=$OUT"

# baseline: extent OFF
train_arm baseline \
  && score_arm baseline

# arm A: extent ON, neutral init, flat g-loss
train_arm armA --fno-extent --fno-extent-merge-init neutral \
  && score_arm armA --fno-extent --fno-extent-merge-init neutral

# arm B: extent ON, additive init, depth x lopsided reweight
train_arm armB --fno-extent --fno-extent-merge-init additive --fno-g-depth-lopsided-strength 4.0 \
  && score_arm armB --fno-extent --fno-extent-merge-init additive

log "all arms complete"
