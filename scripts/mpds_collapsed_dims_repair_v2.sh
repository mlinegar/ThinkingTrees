#!/usr/bin/env bash
# Paper-worthy before/after: do the 4 dims that COLLAPSED under plain-MSE gold leaves in the
# global-vs-local sweep (domain_1/2/3/7) recover with balanced leaf loss?
#   pw=1  = collapse baseline (reproduces the constant-0 failure)
#   pw=10 = repair (sweet spot 5-20; balanced positive-class reweighting)
# root-only (GLOBAL) baseline is reused from outputs/mpds_global_vs_local/root/.
# 4 dims x {pw1, pw10} x 3 seeds = 24 runs. Full gold leaves (root_leaf arm).
# Scored offline vs CORRECT per-dim gold by doc_id.
set -uo pipefail
cd /home/mlinegar/ThinkingTrees
PY=./venv/bin/python
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

LEAF=1
DIMS=(domain_1 domain_2 domain_3 domain_7)
SEEDS=(101 202 303)
POSW=(1 10)
GRID=outputs/mpds_supmix_root_leaf_leaf${LEAF}   # full gold leaves, already built
ARCH=(--fno-n-modes 384 --fno-hidden-channels 32 --fno-n-layers 2
  --fno-head-hidden-dim 256 --fno-epochs 12 --fno-learning-rate 0.00237
  --fno-weight-decay 0.0000185 --fno-leaf-weight 1.569 --fno-root-weight 10
  --fno-merge-weight 0.0)
COMMON=(--family fno --embedding-backend local-hf
  --embedding-model /mnt/data/models/google/embeddinggemma-300m
  --embedding-device cuda --embedding-batch-size 64 --fno-device cuda
  --fg-grid-dir "$GRID" --leaf-qsentences $LEAF --max-iterations 2 --eval-split test)

GPU=0
LOG "Collapsed-dims repair: 4 dims x {pw1,pw10} x 3 seeds"
for dim in "${DIMS[@]}"; do
  for pw in "${POSW[@]}"; do
    for sd in "${SEEDS[@]}"; do
      out=outputs/mpds_collapsed_repair_v2/${dim}/pw${pw}/seed_${sd}
      mkdir -p "$(dirname "$out")"
      CUDA_VISIBLE_DEVICES=$GPU TT_EXPORT_FULL_TREE_TRACES=0 \
        $PY scripts/run_manifesto_qsentence_dspy_ladder.py "${COMMON[@]}" "${ARCH[@]}" \
        --fno-target-dimension "$dim" --fno-seed "$sd" --fno-leaf-pos-weight "$pw" \
        --output-dir "$out" > "${out}.log" 2>&1 &
      GPU=$(( (GPU+1) % 4 )); [ "$GPU" -eq 0 ] && wait
    done
  done
done
wait
LOG "MPDS_COLLAPSED_REPAIR_V2_COMPLETE"
