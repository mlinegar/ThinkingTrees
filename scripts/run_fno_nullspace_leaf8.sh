#!/usr/bin/env bash
# FNO f-null-space salience law, leaf8, domain_4. NO explicit merge weight.
#
# Tests the reframe (deregulation: salience != mass; don't estimate a weight): a
# free/additive merge (merge_mode=mlp, non-convex) + a law that pushes low-impact
# content into f's null space, so the merge ignores it via GEOMETRY. Two arms:
#   control : mlp merge, no law (--fno-g-null-space-weight 0)
#   law     : mlp merge + null-space law (weight swept by NS_WEIGHT)
# CAVEAT: domain_4 (domain-share ratio) IS additive, so this is a WEAK test — it
# checks the law doesn't BREAK composition and can shape geometry. The decisive
# test is a NON-additive construct (follow-up).
set -u

ROOT=/home/mlinegar/ThinkingTrees
cd "$ROOT"
OUT=${NS_ROOT:-outputs/fno_nullspace_leaf8}
mkdir -p "$OUT"
LOG="$OUT/runner.log"
GPU=${FNO_GPU:-0}
EPOCHS=${FNO_EPOCHS:-8}
DIM=${FNO_DIM:-domain_4}
NS_WEIGHT=${NS_WEIGHT:-1.0}
LEAF=8
GRID=outputs/manifesto_qsentence_dspy_labeled_grid
PY=./venv/bin/python

log() { echo "[fno-nullspace $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

train_arm() {
  local name="$1"; shift
  local ddir="$OUT/$name"
  if [ -f "$ddir/fno/leafq00${LEAF}/iteration_history.json" ]; then
    log "$name already trained, skipping"; return 0
  fi
  log "=== $name: training ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_manifesto_qsentence_dspy_ladder.py \
    --family fno --embedding-backend local-hf \
    --embedding-model /mnt/data/models/google/embeddinggemma-300m \
    --embedding-device cuda --embedding-batch-size 128 \
    --fg-grid-dir "$GRID" --leaf-qsentences "$LEAF" \
    --max-iterations 2 --fno-epochs "$EPOCHS" --fno-batch-size 16 \
    --fno-learning-rate 3e-3 --fno-merge-mode mlp \
    --fno-target-dimension "$DIM" "$@" \
    --output-dir "$ddir" --verbose >> "$ddir.train.log" 2>&1 \
    && log "=== $name: trained ===" \
    || { log "WARN: $name TRAIN FAILED (see $ddir.train.log)"; return 1; }
}

score_arm() {
  local name="$1"
  local ddir="$OUT/$name"
  local states="$ddir/g_node_states_${DIM}_leaf${LEAF}.jsonl"
  log "=== $name: dump ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/dump_fno_g_node_states.py \
    --run-dir "$ddir/fno" --leaf-qsentences "$LEAF" --fg-grid-dir "$GRID" \
    --target-dimension "$DIM" \
    --embedding-model /mnt/data/models/google/embeddinggemma-300m \
    --embedding-device cuda --fno-merge-mode mlp \
    --out-jsonl "$states" >> "$ddir.score.log" 2>&1 \
    || { log "WARN: $name DUMP FAILED"; return 1; }
  log "=== $name: score ==="
  $PY scripts/eval_qsentence_merge_by_level.py \
    --labeled-trees "$GRID/leafq00${LEAF}/labeled_trees.jsonl" --split test \
    --g-states-jsonl "$states" --lopsidedness-strength 4.0 \
    --out-json "$ddir/merge_by_level_${DIM}.json" >> "$ddir.score.log" 2>&1 \
    && log "=== $name: scored ===" || log "WARN: $name SCORE FAILED"
}

log "start: dim=$DIM leaf=$LEAF epochs=$EPOCHS ns_weight=$NS_WEIGHT gpu=$GPU root=$OUT"
train_arm control --fno-g-null-space-weight 0   && score_arm control
train_arm law     --fno-g-null-space-weight "$NS_WEIGHT" && score_arm law
log "all arms complete"
