#!/usr/bin/env bash
# Supervision-mix 2D sweep: LLM-teacher-vs-gold-root weighting  ×  merge-label content.
#
# Question (user): what mix/weighting of gold-standard q-sentence (LLM) supervision vs
# gold-standard expert labels works best — the Benoit analog of the old global-vs-local
# supervision dial. These are DIFFERENT signals (LLM teacher = abundant local q-sentence
# scores; gold = scarce holistic expert mean), so this is a genuine mix, not one quantity
# at two scales.
#
# Two axes (non-redundant — note merge_weight=0 ~= merge_supervision=none in the g-loss,
# since fno_family.py skips None-score AND weight<=0 nodes):
#   A) root_weight in {3, 10, 30}            (econ default -> root-dominant -> root-extreme)
#   B) merge label content:
#        llm_span   (merge_w=0.89)  = flat LLM holistic span teacher       [current/default]
#        none       (merge_w=0)     = g unsupervised on merges, root-driven only [the freedom
#                                      the eu diagnosis says g is denied]
#        mean_rollup(merge_w=0.89)  = fixed-mean parametric reference (NOT a proposed fix;
#                                      shows the ceiling a hard-coded mean g would hit)
#
# Architecture HELD FIXED at the econ-tuned recipe so the only thing varying is supervision
# (no per-cell HPO -> clean attribution). Leaf labels (LLM q-sentence scores) are ON in
# every cell; leaf_weight fixed.
#
# Dims: eu (the clean failure: ceiling 0.78, recovers 27%) + economic (working control,
# ceiling 0.83/works near 0.66 — the winning mix must NOT hurt this).
# 3 cells(rw) x 3 cells(merge) x 2 dims x 3 seeds = 54 FNO runs. No LLM scoring (all
# chunk/merge scores already exist). ~2-3h on 4 GPUs.
set -uo pipefail
cd /home/mlinegar/ThinkingTrees
PY=./venv/bin/python
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

DIMS=(eu economic)
ROOT_WEIGHTS=(3 10 30)
SEEDS=(101 202 303)
SRC=outputs/benoit_qsentence_grid_full
EXPERT=outputs/benoit_qsentence_targets/expert_means_raw.json
LEAF=16
# Fixed architecture (econ HPO winner) minus root_weight/merge_weight which the sweep sets.
ARCH=(--fno-n-modes 384 --fno-hidden-channels 32 --fno-n-layers 2
  --fno-head-hidden-dim 256 --fno-epochs 12 --fno-learning-rate 0.00237
  --fno-weight-decay 0.0000185 --fno-leaf-weight 1.569)

# ---------- Phase 0: build the missing relabeled grids (no LLM calls) ----------
build_grid() {  # dim, supervision, outdir
  local dim=$1 sup=$2 out=$3
  [ -d "$out" ] && { LOG "  grid exists: $out"; return; }
  LOG "  building grid $out ($sup)"
  local extra=()
  [ "$sup" = "llm_span" ] && extra=(--merge-scores "outputs/benoit_chunk_scores_forced/leafq016_${dim}_merges.json")
  $PY scripts/relabel_benoit_grid_with_chunks.py \
    --src-grid "$SRC" --leaf $LEAF --dim "$dim" \
    --chunk-scores "outputs/benoit_chunk_scores_forced/leafq016_${dim}.json" \
    "${extra[@]}" \
    --expert-targets "$EXPERT" --merge-supervision "$sup" \
    --output-dir "$out" > "outputs/relabel_${dim}_${sup}.log" 2>&1
}
LOG "Phase 0: ensure relabeled grids for {llm_span,none,mean_rollup} x {eu,economic}"
for dim in "${DIMS[@]}"; do
  build_grid "$dim" llm_span   "outputs/benoit_chunkgrid_forced_${dim}_llmspan"
  build_grid "$dim" none       "outputs/benoit_chunkgrid_forced_${dim}_none"
  build_grid "$dim" mean_rollup "outputs/benoit_chunkgrid_forced_${dim}_meanroll"
done

# merge-label cell -> (grid suffix, merge_weight)
declare -A GRIDSUF=( [llm_span]=llmspan [none]=none [mean_rollup]=meanroll )
declare -A MWEIGHT=( [llm_span]=0.886 [none]=0.0 [mean_rollup]=0.886 )

# ---------- Phase 1: the 2D sweep ----------
GPU=0
launch_cell() {  # dim, merge_label, rw, seed, gpu
  local dim=$1 ml=$2 rw=$3 sd=$4 g=$5
  local grid=outputs/benoit_chunkgrid_forced_${dim}_${GRIDSUF[$ml]}
  local mw=${MWEIGHT[$ml]}
  local out=outputs/benoit_supmix/${dim}/ml_${ml}_rw${rw}/seed_${sd}
  mkdir -p "$(dirname "$out")"
  CUDA_VISIBLE_DEVICES=$g TT_EXPORT_FULL_TREE_TRACES=0 \
    $PY scripts/run_manifesto_qsentence_dspy_ladder.py \
    --family fno --embedding-backend local-hf \
    --embedding-model /mnt/data/models/google/embeddinggemma-300m \
    --embedding-device cuda --embedding-batch-size 64 --fno-device cuda \
    --fg-grid-dir "$grid" --leaf-qsentences 16 --max-iterations 2 \
    --fno-target-dimension "$dim" --eval-split test \
    "${ARCH[@]}" --fno-merge-weight "$mw" --fno-root-weight "$rw" \
    --fno-seed "$sd" --output-dir "$out" > "${out}.log" 2>&1 &
}

LOG "Phase 1: 2D sweep (root_weight x merge_label) x {eu,economic} x 3 seeds"
for dim in "${DIMS[@]}"; do
  for ml in llm_span none mean_rollup; do
    for rw in "${ROOT_WEIGHTS[@]}"; do
      for sd in "${SEEDS[@]}"; do
        launch_cell "$dim" "$ml" "$rw" "$sd" "$GPU"
        GPU=$(( (GPU+1) % 4 ))
        [ "$GPU" -eq 0 ] && wait   # batch of 4 across the 4 GPUs
      done
    done
  done
done
wait
LOG "SUPMIX_SWEEP_COMPLETE"
