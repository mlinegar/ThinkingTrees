#!/usr/bin/env bash
# Single arm of the FNO null-space experiment (train -> dump -> score), so the two
# arms can run in PARALLEL on separate GPUs. Args via env:
#   ARM_NAME   control | law
#   NS_WEIGHT  null-space law weight (0 for control)
#   FNO_GPU    GPU index for this arm
#   NS_ROOT    shared output root; this arm writes NS_ROOT/ARM_NAME
set -u
ROOT=/home/mlinegar/ThinkingTrees
cd "$ROOT"
: "${ARM_NAME:?need ARM_NAME}"; : "${NS_ROOT:?need NS_ROOT}"
GPU=${FNO_GPU:-0}; EPOCHS=${FNO_EPOCHS:-8}; DIM=${FNO_DIM:-domain_4}
NS_WEIGHT=${NS_WEIGHT:-0}; LEAF=8
GRID=outputs/manifesto_qsentence_dspy_labeled_grid
PY=./venv/bin/python
ddir="$NS_ROOT/$ARM_NAME"
mkdir -p "$NS_ROOT"
log() { echo "[ns-$ARM_NAME $(date +%H:%M:%S)] $*"; }

log "start gpu=$GPU ns_weight=$NS_WEIGHT dim=$DIM -> $ddir"
if [ ! -f "$ddir/fno/leafq00${LEAF}/iteration_history.json" ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_manifesto_qsentence_dspy_ladder.py \
    --family fno --embedding-backend local-hf \
    --embedding-model /mnt/data/models/google/embeddinggemma-300m \
    --embedding-device cuda --embedding-batch-size 128 \
    --fg-grid-dir "$GRID" --leaf-qsentences "$LEAF" \
    --max-iterations 2 --fno-epochs "$EPOCHS" --fno-batch-size 16 \
    --fno-learning-rate 3e-3 --fno-merge-mode mlp \
    --fno-target-dimension "$DIM" --fno-g-null-space-weight "$NS_WEIGHT" \
    --output-dir "$ddir" --verbose >> "$ddir.train.log" 2>&1 \
    || { log "TRAIN FAILED (see $ddir.train.log)"; exit 1; }
  log "trained"
else
  log "already trained, skipping"
fi

states="$ddir/g_node_states_${DIM}_leaf${LEAF}.jsonl"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/dump_fno_g_node_states.py \
  --run-dir "$ddir/fno" --leaf-qsentences "$LEAF" --fg-grid-dir "$GRID" \
  --target-dimension "$DIM" \
  --embedding-model /mnt/data/models/google/embeddinggemma-300m \
  --embedding-device cuda --fno-merge-mode mlp \
  --out-jsonl "$states" >> "$ddir.score.log" 2>&1 \
  || { log "DUMP FAILED"; exit 1; }
log "dumped"

$PY scripts/eval_qsentence_merge_by_level.py \
  --labeled-trees "$GRID/leafq00${LEAF}/labeled_trees.jsonl" --split test \
  --g-states-jsonl "$states" --lopsidedness-strength 4.0 \
  --out-json "$ddir/merge_by_level_${DIM}.json" >> "$ddir.score.log" 2>&1 \
  && log "scored -> $ddir/merge_by_level_${DIM}.json" || { log "SCORE FAILED"; exit 1; }
log "done"
