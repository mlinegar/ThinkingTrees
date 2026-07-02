#!/usr/bin/env bash
set -u
cd /home/mlinegar/ThinkingTrees
AB=$(cat scratchpad/fno_extent_ab_root.txt)
GRID=outputs/manifesto_qsentence_dspy_labeled_grid
PY=./venv/bin/python
dump_score() {
  local name="$1"; shift
  local ddir="$AB/$name"
  local states="$ddir/g_node_states_domain_4_leaf8.jsonl"
  echo "=== $name: dump $(date +%H:%M:%S) ==="
  CUDA_VISIBLE_DEVICES=0 $PY scripts/dump_fno_g_node_states.py \
    --run-dir "$ddir/fno" --leaf-qsentences 8 --fg-grid-dir "$GRID" \
    --target-dimension domain_4 \
    --embedding-model /mnt/data/models/google/embeddinggemma-300m \
    --embedding-device cuda --fno-merge-mode gated "$@" \
    --out-jsonl "$states" 2>&1 | grep -vE "Loading weights|Materializing|torch_dtype|UserWarning|warnings.warn" | tail -2
  echo "=== $name: score ==="
  $PY scripts/eval_qsentence_merge_by_level.py \
    --labeled-trees "$GRID/leafq008/labeled_trees.jsonl" --split test \
    --g-states-jsonl "$states" --lopsidedness-strength 4.0 \
    --out-json "$ddir/merge_by_level_domain_4.json" 2>&1 | grep -iE "VERDICT|pooled wmae|equal_avg:|learned_g:|mass_wtd:" | tail -6
}
dump_score baseline
dump_score armA --fno-extent --fno-extent-merge-init neutral
dump_score armB --fno-extent --fno-extent-merge-init additive
echo "ALL DONE $(date +%H:%M:%S)"
