#!/usr/bin/env bash
# Two ablations on the MPDS gold-LEAF supervision arm, run on the easy + hard test dims:
#   domain_4 = EASY  (balanced, gold leaves HELP: root +0.50 -> gold-leaf +0.61)
#   domain_3 = HARD  (sparse 91% zeros, gold leaves COLLAPSE g to 0: root +0.58 -> 0.00)
#
# Exp A (balanced leaf loss): full gold-leaf arm, sweep --fno-leaf-pos-weight {1,5,20,50}.
#   pos_weight=1 reproduces the collapse baseline. Tests WHETHER the collapse is the LABELS
#   (signal absent) or the MSE OBJECTIVE (zeros dominate the gradient). If upweighting the
#   ~9% positive leaves rescues domain_3, the signal was there and MSE was the culprit.
#
# Exp B (dose-response): gold-leaf arm with --leaf-keep-frac {0,0.25,0.5,0.75,1.0}.
#   0% = root-only; 100% = full gold-leaf. Finds where each dim TIPS into collapse.
#
# Scored offline vs CORRECT per-dim gold by doc_id (records' expert_score is RILE-only; see
# MPDS_GLOBAL_VS_LOCAL_FINDINGS.md scoring-bug note).
set -uo pipefail
cd /home/mlinegar/ThinkingTrees
PY=./venv/bin/python
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

SRC=outputs/manifesto_qsentence_dspy_labeled_grid
LEAF=1
DIMS=(domain_3 domain_4)
SEEDS=(101 202 303)
POSW=(1 5 20 50)
FRACS=(0.0 0.25 0.5 0.75 1.0)
ARCH=(--fno-n-modes 384 --fno-hidden-channels 32 --fno-n-layers 2
  --fno-head-hidden-dim 256 --fno-epochs 12 --fno-learning-rate 0.00237
  --fno-weight-decay 0.0000185 --fno-leaf-weight 1.569 --fno-root-weight 10
  --fno-merge-weight 0.0)
COMMON=(--family fno --embedding-backend local-hf
  --embedding-model /mnt/data/models/google/embeddinggemma-300m
  --embedding-device cuda --embedding-batch-size 64 --fno-device cuda
  --leaf-qsentences $LEAF --max-iterations 2 --eval-split test)

run() {  # $1=grid $2=dim $3=seed $4=outdir $5=posweight
  CUDA_VISIBLE_DEVICES=$GPU TT_EXPORT_FULL_TREE_TRACES=0 \
    $PY scripts/run_manifesto_qsentence_dspy_ladder.py "${COMMON[@]}" "${ARCH[@]}" \
    --fg-grid-dir "$1" --fno-target-dimension "$2" --fno-seed "$3" \
    --fno-leaf-pos-weight "$5" --output-dir "$4" > "${4}.log" 2>&1 &
  GPU=$(( (GPU+1) % 4 )); [ "$GPU" -eq 0 ] && wait
}

GPU=0

# ---------- Exp A: balanced leaf loss (full gold-leaf arm, pos_weight sweep) ----------
LOG "Exp A: balanced leaf loss (pos_weight sweep) on full gold-leaf arm"
GRID_A=outputs/mpds_supmix_root_leaf_leaf${LEAF}   # already built (full gold leaves)
if [ ! -d "$GRID_A/leafq$(printf %03d $LEAF)" ]; then
  $PY scripts/build_mpds_supervision_arm.py --src-grid "$SRC" --leaf $LEAF \
    --keep root_leaf --output-dir "$GRID_A" 2>&1 | sed 's/^/  /'
fi
for dim in "${DIMS[@]}"; do
  for pw in "${POSW[@]}"; do
    for sd in "${SEEDS[@]}"; do
      out=outputs/mpds_leaf_balance/expA_posweight/${dim}/pw${pw}/seed_${sd}
      mkdir -p "$(dirname "$out")"
      run "$GRID_A" "$dim" "$sd" "$out" "$pw"
    done
  done
done

# ---------- Exp B: dose-response (leaf-keep-frac sweep, pos_weight=1) ----------
LOG "Exp B: dose-response (leaf-keep-frac sweep)"
for f in "${FRACS[@]}"; do
  fl=$(echo "$f" | tr -d '.')
  grid=outputs/mpds_supmix_root_leaf_frac${fl}_leaf${LEAF}
  if [ ! -d "$grid/leafq$(printf %03d $LEAF)" ]; then
    $PY scripts/build_mpds_supervision_arm.py --src-grid "$SRC" --leaf $LEAF \
      --keep root_leaf --leaf-keep-frac "$f" --output-dir "$grid" 2>&1 | sed 's/^/  /'
  fi
  for dim in "${DIMS[@]}"; do
    for sd in "${SEEDS[@]}"; do
      out=outputs/mpds_leaf_balance/expB_frac/${dim}/frac${fl}/seed_${sd}
      mkdir -p "$(dirname "$out")"
      run "$grid" "$dim" "$sd" "$out" 1
    done
  done
done
wait
LOG "MPDS_LEAF_BALANCE_DOSE_COMPLETE"
