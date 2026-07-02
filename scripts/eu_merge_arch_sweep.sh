#!/usr/bin/env bash
# eu learned-merge architecture sweep: does a GATED merge head break the eu floor?
#
# Context (outputs/benoit_6dim_fno/WHY_WEAK_DIMS_FAIL.md, SUPERVISION_MIX_FINDINGS.md):
#   eu has a leaf-rollup ceiling of 0.78 but the LEARNED g recovers only ~0.33 (best,
#   merge_supervision=none/rw10). 48 HPO trials over weight/capacity/LR could not move it,
#   and dropping flat merge labels (none) only lifted 0.23->0.33. The diagnosis: the learned
#   g MERGE is the bottleneck — it averages a sparse on-topic-minority child's signal away.
#   The flagged next lever is g ARCHITECTURE, not weighting/supervision.
#
# This sweep tests exactly that single variable. Everything else is held at the supmix
# WINNER (econ arch, merge_supervision=none, per-dim best root_weight) so the only thing
# that changes is the merge baseline:
#   merge_mode=mean   = 0.5*(l+r) + FNO residual          [supmix best, same-code control]
#   merge_mode=gated  = alpha(l,r)*l + (1-alpha)*r + resid [NEW: learned per-dim routing]
# Both are identical at init (gate alpha=0.5); gated is a strict generalization of mean.
#
# Dims: eu (the target: ceiling 0.78, floor ~0.33) + economic (control: gated must NOT
#       hurt the working dim, ~0.73). Per-dim best root_weight from supmix: eu=10, econ=3.
# Grids reuse the existing merge_supervision=none relabeled grids (no LLM scoring).
# 2 dims x 2 merge_modes x 3 seeds = 12 FNO runs. ~30-45 min on 4 GPUs.
set -uo pipefail
cd /home/mlinegar/ThinkingTrees
PY=./venv/bin/python
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

SEEDS=(101 202 303)
LEAF=16
# Fixed architecture (econ HPO winner), merge_supervision=none (merge_weight=0).
ARCH=(--fno-n-modes 384 --fno-hidden-channels 32 --fno-n-layers 2
  --fno-head-hidden-dim 256 --fno-epochs 12 --fno-learning-rate 0.00237
  --fno-weight-decay 0.0000185 --fno-leaf-weight 1.569 --fno-merge-weight 0.0)
# per-dim winning root_weight (supmix): eu wants rw10, econ wants rw3.
declare -A RW=( [eu]=10 [economic]=3 )
# gate hidden width — modest; the gate is a per-dim [0,1] router, not a deep net.
GATE_HID=256

GPU=0
launch_cell() {  # dim, merge_mode, seed, gpu
  local dim=$1 mm=$2 sd=$3 g=$4
  local grid=outputs/benoit_chunkgrid_forced_${dim}_none
  local rw=${RW[$dim]}
  local out=outputs/eu_merge_arch/${dim}/mm_${mm}_rw${rw}/seed_${sd}
  if [ -f "$out/fno/leafq016/prediction_records/iter_02_post_eval.jsonl" ]; then
    LOG "  skip (done): $out"; return 0
  fi
  mkdir -p "$(dirname "$out")"
  if [ ! -d "$grid" ]; then
    LOG "MISSING grid $grid — run supervision_mix_sweep.sh Phase 0 first"; return 1
  fi
  local gate_args=()
  # gate hidden dim doubles as the mlp hidden width.
  { [ "$mm" = "gated" ] || [ "$mm" = "mlp" ]; } && gate_args=(--fno-merge-gate-hidden-dim "$GATE_HID")
  CUDA_VISIBLE_DEVICES=$g TT_EXPORT_FULL_TREE_TRACES=0 \
    $PY scripts/run_manifesto_qsentence_dspy_ladder.py \
    --family fno --embedding-backend local-hf \
    --embedding-model /mnt/data/models/google/embeddinggemma-300m \
    --embedding-device cuda --embedding-batch-size 64 --fno-device cuda \
    --fg-grid-dir "$grid" --leaf-qsentences 16 --max-iterations 2 \
    --fno-target-dimension "$dim" --eval-split test \
    "${ARCH[@]}" --fno-root-weight "$rw" \
    --fno-merge-mode "$mm" "${gate_args[@]}" \
    --fno-seed "$sd" --output-dir "$out" > "${out}.log" 2>&1 &
}

LOG "eu_merge_arch sweep: merge_mode {mean,gated,maxpool,mlp} x {eu,economic} x 3 seeds"
for dim in eu economic; do
  for mm in mean gated maxpool mlp; do
    for sd in "${SEEDS[@]}"; do
      launch_cell "$dim" "$mm" "$sd" "$GPU"
      GPU=$(( (GPU+1) % 4 ))
      [ "$GPU" -eq 0 ] && wait
    done
  done
done
wait
LOG "EU_MERGE_ARCH_SWEEP_COMPLETE"
