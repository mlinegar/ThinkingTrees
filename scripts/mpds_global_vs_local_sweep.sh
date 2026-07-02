#!/usr/bin/env bash
# MPDS global-vs-local GOLD supervision sweep.
#
# The TRUE test Benoit could not run: MPDS (manifesto_qsentence_dspy_labeled_grid) has
# GOLD CMP human codes per quasi-sentence (label_source manifesto_qsentence_cmp_annotations_v1)
# that VARY within a doc -- real per-sentence gold, not LLM-teacher, not doc-mean broadcast.
#
# Arms (which gold nodes are kept; all else -> None = unsupervised/learned):
#   root            = ONLY root gold (pure GLOBAL / doc-level supervision)
#   root_leaf       = root + gold per-q-sentence LEAVES; merges unsupervised (root + gold LOCAL)
#   root_leaf_merge = full gold supervision (reference / upper bound)
# Arms are dim-independent (just node-nulling) -> build once, reuse across all dims.
#
# leaf=1 (one quasi-sentence per leaf = observation granularity = the gold unit).
# Dims: rile + domain_1..7 (8 dims). Per-dim scored vs gold doc rollup.
# 3 arms x 8 dims x 3 seeds = 72 FNO runs. Arch fixed (econ recipe) so only the supervision
# arm varies. merge_weight=0 in all arms (merges either gold-or-none; never flat teacher).
set -uo pipefail
cd /home/mlinegar/ThinkingTrees
PY=./venv/bin/python
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

SRC=outputs/manifesto_qsentence_dspy_labeled_grid
LEAF=1
DIMS=(rile domain_1 domain_2 domain_3 domain_4 domain_5 domain_6 domain_7)
ARMS=(root root_leaf root_leaf_merge)
SEEDS=(101 202 303)
ARCH=(--fno-n-modes 384 --fno-hidden-channels 32 --fno-n-layers 2
  --fno-head-hidden-dim 256 --fno-epochs 12 --fno-learning-rate 0.00237
  --fno-weight-decay 0.0000185 --fno-leaf-weight 1.569 --fno-root-weight 10)

# ---------- Phase 0: build the 3 arms (dim-independent) ----------
LOG "Phase 0: build supervision arms (node-nulling, dim-independent)"
for arm in "${ARMS[@]}"; do
  out=outputs/mpds_supmix_${arm}_leaf${LEAF}
  if [ -d "$out/leafq$(printf %03d $LEAF)" ]; then LOG "  arm exists: $out"; continue; fi
  $PY scripts/build_mpds_supervision_arm.py --src-grid "$SRC" --leaf $LEAF \
    --keep "$arm" --output-dir "$out" 2>&1 | sed 's/^/  /'
done

# for root_leaf_merge keep gold merges -> they ARE supervised, so give merges weight.
mweight_for() { [ "$1" = "root_leaf_merge" ] && echo 0.886 || echo 0.0; }

# ---------- Phase 1: 3 arms x 8 dims x 3 seeds ----------
GPU=0
LOG "Phase 1: sweep (arm x dim x seed) at leaf=$LEAF"
for arm in "${ARMS[@]}"; do
  grid=outputs/mpds_supmix_${arm}_leaf${LEAF}
  mw=$(mweight_for "$arm")
  for dim in "${DIMS[@]}"; do
    for sd in "${SEEDS[@]}"; do
      out=outputs/mpds_global_vs_local/${arm}/${dim}/seed_${sd}
      mkdir -p "$(dirname "$out")"
      CUDA_VISIBLE_DEVICES=$GPU TT_EXPORT_FULL_TREE_TRACES=0 \
        $PY scripts/run_manifesto_qsentence_dspy_ladder.py \
        --family fno --embedding-backend local-hf \
        --embedding-model /mnt/data/models/google/embeddinggemma-300m \
        --embedding-device cuda --embedding-batch-size 64 --fno-device cuda \
        --fg-grid-dir "$grid" --leaf-qsentences $LEAF --max-iterations 2 \
        --fno-target-dimension "$dim" --eval-split test \
        "${ARCH[@]}" --fno-merge-weight "$mw" \
        --fno-seed "$sd" --output-dir "$out" > "${out}.log" 2>&1 &
      GPU=$(( (GPU+1) % 4 ))
      [ "$GPU" -eq 0 ] && wait
    done
  done
done
wait
LOG "MPDS_GLOBAL_VS_LOCAL_COMPLETE"
