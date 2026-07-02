#!/usr/bin/env bash
# RILE arm of the LLM-recreation A/B, plus orchestrates the Benoit 4-dim FNO.
# RILE differs from the Benoit dims:
#   - leaf=1 (one q-sentence per leaf = the gold observation unit), NOT leaf=16.
#   - gold arm  = the EXISTING MPDS grid (outputs/manifesto_qsentence_dspy_labeled_grid,
#     leafq001) which already carries rile leaves (CMP-derived) + real doc-rile roots.
#   - llm  arm  = outputs/mpds_rile_llmseg_none_grid (gemma re-segmented + gemma
#     left-right leaf scores + doc-rile roots, merge_supervision=none).
# Splits are per-tree metadata.split, identical TEST docs (48/48 verified).
# merge_weight=0 (supmix) so internal-node merge labels are out of the loss; the
# gold MPDS grid's free-g merge labels are harmless.
set -uo pipefail
cd /home/mlinegar/ThinkingTrees
PY=./venv/bin/python
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

EMB=/mnt/data/models/google/embeddinggemma-300m
SEEDS=(101 202 303)
ARCH=(--fno-n-modes 384 --fno-hidden-channels 32 --fno-n-layers 2
  --fno-head-hidden-dim 256 --fno-epochs 12 --fno-learning-rate 0.00237
  --fno-weight-decay 0.0000185 --fno-leaf-weight 1.569 --fno-merge-weight 0.0)
RILE_RW=10  # rile is leaf-sparse / weak like eu -> high root weight (supmix finding)

run_rile() {  # tag grid seed gpu
  local tag=$1 grid=$2 sd=$3 g=$4
  local out=outputs/llm_recreation/rile/${tag}/seed_${sd}
  [ -f "$out/fno/leafq001/prediction_records/iter_02_post_eval.jsonl" ] && { LOG "  skip $out"; return; }
  mkdir -p "$(dirname "$out")"
  CUDA_VISIBLE_DEVICES=$g TT_EXPORT_FULL_TREE_TRACES=0 \
    $PY scripts/run_manifesto_qsentence_dspy_ladder.py \
    --family fno --embedding-backend local-hf --embedding-model "$EMB" \
    --embedding-device cuda --embedding-batch-size 64 --fno-device cuda \
    --fg-grid-dir "$grid" --leaf-qsentences 1 --max-iterations 2 \
    --fno-target-dimension rile --eval-split test \
    "${ARCH[@]}" --fno-root-weight "$RILE_RW" \
    --fno-seed "$sd" --output-dir "$out" > "${out}.log" 2>&1 &
}

stage_rile() {
  LOG "FNO RILE: gold (MPDS grid) + llm (llmseg none), leaf=1, 3 seeds."
  GPU=0
  for sd in "${SEEDS[@]}"; do
    run_rile gold outputs/manifesto_qsentence_dspy_labeled_grid "$sd" "$GPU"
    GPU=$(( (GPU+1) % 4 )); [ "$GPU" -eq 0 ] && wait
    run_rile llm  outputs/mpds_rile_llmseg_none_grid          "$sd" "$GPU"
    GPU=$(( (GPU+1) % 4 )); [ "$GPU" -eq 0 ] && wait
  done
  wait
  LOG "RILE_FNO_COMPLETE"
}

LOG "=== Phase A: Benoit 4-dim FNO ==="
bash scripts/llm_recreation_extend_benoit.sh fno
LOG "=== Phase B: RILE FNO ==="
stage_rile
LOG "ALL_LLM_RECREATION_FNO_COMPLETE"
