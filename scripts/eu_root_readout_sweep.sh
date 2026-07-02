#!/usr/bin/env bash
# eu ROOT-READOUT sweep (all-gemma): read the doc prediction from the LEAF score
# distribution instead of the mean-composed root state.
#
# Four converging negatives (merge capacity, merge supervision, leaf-loss balance, encoder)
# localized the eu floor to the READOUT, not the leaves/merge/encoder: the eu doc score is
# carried by the single most-EU q-sentence (top1-leaf r=0.793 ~= 0.78 ceiling; top-k
# DECREASING), but the FNO reads the root from the mean-composed tree state, which averages
# that peak away. These readouts use the leaf SCORES directly:
#   mean_root (control) | topk k=1 (the max leaf) | topk k=3 | softmax temp=0.2 (~soft max)
# f trains end-to-end toward the readout, so the leaf head learns to make the top leaves
# accurate. Encoder stays embeddinggemma-300m (won the encoder probe). Recipe = supmix winner.
# Arms are INDEPENDENT -> launched in parallel across all 4 GPUs.
# 4 readouts x 2 dims x 3 seeds = 24 runs. ~25-35 min.
set -uo pipefail
cd /home/mlinegar/ThinkingTrees
PY=./venv/bin/python
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

SEEDS=(101 202 303)
ARCH=(--fno-n-modes 384 --fno-hidden-channels 32 --fno-n-layers 2
  --fno-head-hidden-dim 256 --fno-epochs 12 --fno-learning-rate 0.00237
  --fno-weight-decay 0.0000185 --fno-leaf-weight 1.569 --fno-merge-weight 0.0)
declare -A RW=( [eu]=10 [economic]=3 )
EMB=/mnt/data/models/google/embeddinggemma-300m

# arm tag -> readout flags
declare -A ARM_FLAGS=(
  [meanroot]="--fno-root-readout mean_root"
  [top1]="--fno-root-readout topk --fno-root-readout-k 1"
  [top3]="--fno-root-readout topk --fno-root-readout-k 3"
  [softmax02]="--fno-root-readout softmax --fno-root-readout-attn-temp 0.2"
)

GPU=0
launch_cell() {  # dim, arm, seed, gpu
  local dim=$1 arm=$2 sd=$3 g=$4
  local grid=outputs/benoit_chunkgrid_forced_${dim}_none
  local rw=${RW[$dim]}
  local out=outputs/eu_root_readout/${dim}/${arm}/seed_${sd}
  if [ -f "$out/fno/leafq016/prediction_records/iter_02_post_eval.jsonl" ]; then
    LOG "  skip (done): $out"; return 0; fi
  mkdir -p "$(dirname "$out")"
  [ -d "$grid" ] || { LOG "MISSING grid $grid"; return 1; }
  CUDA_VISIBLE_DEVICES=$g TT_EXPORT_FULL_TREE_TRACES=0 \
    $PY scripts/run_manifesto_qsentence_dspy_ladder.py \
    --family fno --embedding-backend local-hf \
    --embedding-model "$EMB" \
    --embedding-device cuda --embedding-batch-size 64 --fno-device cuda \
    --fg-grid-dir "$grid" --leaf-qsentences 16 --max-iterations 2 \
    --fno-target-dimension "$dim" --eval-split test \
    "${ARCH[@]}" --fno-root-weight "$rw" ${ARM_FLAGS[$arm]} \
    --fno-seed "$sd" --output-dir "$out" > "${out}.log" 2>&1 &
}

LOG "eu_root_readout sweep: {meanroot,top1,top3,softmax02} x {eu,economic} x 3 seeds"
for dim in eu economic; do
  for arm in meanroot top1 top3 softmax02; do
    for sd in "${SEEDS[@]}"; do
      launch_cell "$dim" "$arm" "$sd" "$GPU"
      GPU=$(( (GPU+1) % 4 ))
      [ "$GPU" -eq 0 ] && wait
    done
  done
done
wait
LOG "EU_ROOT_READOUT_SWEEP_COMPLETE"
