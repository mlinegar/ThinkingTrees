#!/usr/bin/env bash
# Seed-stabilize the HPO-best econ config: 3 seeds, one GPU each, TEST split.
set -u
cd /home/mlinegar/ThinkingTrees

GRID="outputs/benoit_chunkgrid_forced_economic_llmspan"
COMMON=(
  --family fno --embedding-backend local-hf
  --embedding-model /mnt/data/models/google/embeddinggemma-300m
  --embedding-device cuda --embedding-batch-size 64
  --fno-device cuda
  --fg-grid-dir "$GRID" --leaf-qsentences 16
  --max-iterations 2 --fno-target-dimension economic
  --eval-split test
  --fno-n-modes 384 --fno-hidden-channels 32 --fno-n-layers 2
  --fno-head-hidden-dim 256 --fno-epochs 12
  --fno-learning-rate 0.00237 --fno-weight-decay 0.0000185
  --fno-leaf-weight 1.569 --fno-merge-weight 0.886 --fno-root-weight 3.076
)

SEEDS=(101 202 303)
GPUS=(0 1 2)
for i in 0 1 2; do
  s=${SEEDS[$i]}; g=${GPUS[$i]}
  out="outputs/econ_seedcheck/seed_${s}"
  CUDA_VISIBLE_DEVICES=$g TT_EXPORT_FULL_TREE_TRACES=0 \
    ./venv/bin/python scripts/run_manifesto_qsentence_dspy_ladder.py \
    "${COMMON[@]}" --fno-seed "$s" --output-dir "$out" \
    > "${out}.log" 2>&1 &
done
wait
echo "SEEDS_DONE"
