#!/usr/bin/env bash
# eu LEAF-STATE sweep: does balanced leaf loss lift the weak eu leaf states (f=0.25)?
#
# The merge-architecture sweep (EU_MERGE_ARCH_FINDINGS.md) proved capacity is NOT the eu
# lever — the bottleneck is the LEAF STATES (eu f-stage only 0.25) and root-only supervision.
# Leaf diagnostic: eu LLM-span leaf scores are 86.5% exactly 0.5 (neutral), only ~10% on-topic
# (>0.5). Plain MSE on a 0.5-neutral-majority target drives the f head to a constant ~0.5 —
# the SAME gradient-dilution trap we cured on MPDS sparse dims with --fno-leaf-pos-weight, but
# the eu leaf neutral is 0.5 (not 0), so it needs --fno-leaf-pos-neutral 0.5.
#
# Single variable: leaf_pos_weight ∈ {1, 5, 20} at leaf_pos_neutral=0.5. Everything else =
# supmix winner (econ arch, merge_supervision=none, merge_mode=mean, per-dim best root_weight).
# PRIMARY readout = f-stage (iter_01): does the leaf head come off 0.25? Secondary = g-stage.
# Dims: eu (target) + economic (control — must not break; its leaves are less neutral 72%).
# 2 dims × 3 pos_weights × 3 seeds = 18 runs (pw=1 reuses the existing mean cells as control).
set -uo pipefail
cd /home/mlinegar/ThinkingTrees
PY=./venv/bin/python
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

SEEDS=(101 202 303)
LEAF=16
ARCH=(--fno-n-modes 384 --fno-hidden-channels 32 --fno-n-layers 2
  --fno-head-hidden-dim 256 --fno-epochs 12 --fno-learning-rate 0.00237
  --fno-weight-decay 0.0000185 --fno-leaf-weight 1.569 --fno-merge-weight 0.0)
declare -A RW=( [eu]=10 [economic]=3 )
POS_WEIGHTS=(1 5 20)
NEUTRAL=0.5

GPU=0
launch_cell() {  # dim, pw, seed, gpu
  local dim=$1 pw=$2 sd=$3 g=$4
  local grid=outputs/benoit_chunkgrid_forced_${dim}_none
  local rw=${RW[$dim]}
  local out=outputs/eu_leaf_balance/${dim}/pw${pw}/seed_${sd}
  if [ -f "$out/fno/leafq016/prediction_records/iter_02_post_eval.jsonl" ]; then
    LOG "  skip (done): $out"; return 0; fi
  mkdir -p "$(dirname "$out")"
  [ -d "$grid" ] || { LOG "MISSING grid $grid"; return 1; }
  CUDA_VISIBLE_DEVICES=$g TT_EXPORT_FULL_TREE_TRACES=0 \
    $PY scripts/run_manifesto_qsentence_dspy_ladder.py \
    --family fno --embedding-backend local-hf \
    --embedding-model /mnt/data/models/google/embeddinggemma-300m \
    --embedding-device cuda --embedding-batch-size 64 --fno-device cuda \
    --fg-grid-dir "$grid" --leaf-qsentences 16 --max-iterations 2 \
    --fno-target-dimension "$dim" --eval-split test \
    "${ARCH[@]}" --fno-root-weight "$rw" \
    --fno-leaf-pos-weight "$pw" --fno-leaf-pos-neutral "$NEUTRAL" \
    --fno-seed "$sd" --output-dir "$out" > "${out}.log" 2>&1 &
}

LOG "eu_leaf_balance sweep: leaf_pos_weight {1,5,20} @ neutral=0.5 x {eu,economic} x 3 seeds"
for dim in eu economic; do
  for pw in "${POS_WEIGHTS[@]}"; do
    for sd in "${SEEDS[@]}"; do
      launch_cell "$dim" "$pw" "$sd" "$GPU"
      GPU=$(( (GPU+1) % 4 ))
      [ "$GPU" -eq 0 ] && wait
    done
  done
done
wait
LOG "EU_LEAF_BALANCE_SWEEP_COMPLETE"
