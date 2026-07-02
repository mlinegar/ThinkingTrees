#!/usr/bin/env bash
# eu ENCODER sweep: is the eu floor a leaf-REPRESENTATION limit? Swap the embedding model.
#
# Four converging negatives (EU_MERGE_ARCH + eu_leaf_balance) localized the eu floor to the
# LEAF STATES (eu f-stage stuck ~0.25), and balanced leaf loss could NOT lift it (while it
# helped econ) -> the embedding can't represent the eu signal, not a loss problem. KEY clue:
# the eu docs are MULTILINGUAL EUROPEAN (Hungarian / Swedish / Estonian text observed), and
# the baseline encoder is embeddinggemma-300m (768-dim, small). A stronger multilingual
# encoder (Qwen3-Embedding 0.6B=1024d / 4B=2560d, 32k ctx) is the direct test.
#
# Single variable: --embedding-model. Recipe = supmix winner (econ arch, merge_supervision=
# none, merge_mode=mean, per-dim best root_weight). No grid rebuild needed — the FNO re-embeds
# leaf TEXT at runtime; embedding_dim is probed and flows into the FNO spatial axis automatically.
# --embedding-max-length 4096 to clear the no-truncation guard (longest eu leaf ~2448 tok under
# Qwen tokenizer; Qwen3 ctx=32k). PRIMARY readout = f-stage (iter_01): does eu leaf score lift?
# Dims: eu (target) + economic (control). 3 encoders x 2 dims x 3 seeds = 18 runs.
set -uo pipefail
cd /home/mlinegar/ThinkingTrees
PY=./venv/bin/python
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

SEEDS=(101 202 303)
ARCH=(--fno-n-modes 384 --fno-hidden-channels 32 --fno-n-layers 2
  --fno-head-hidden-dim 256 --fno-epochs 12 --fno-learning-rate 0.00237
  --fno-weight-decay 0.0000185 --fno-leaf-weight 1.569 --fno-merge-weight 0.0)
declare -A RW=( [eu]=10 [economic]=3 )

# encoder tag -> model path. n_modes(384) <= every dim below so it's safe as-is.
declare -A ENC=(
  [gemma300m]=/mnt/data/models/google/embeddinggemma-300m
  [qwen0_6b]=/mnt/data/models/Qwen/Qwen3-Embedding-0.6B
  [qwen4b]=/mnt/data/models/Qwen/Qwen3-Embedding-4B
)
# bigger encoders -> smaller embedding batch to stay in VRAM.
declare -A EBATCH=( [gemma300m]=64 [qwen0_6b]=16 [qwen4b]=8 )

GPU=0
launch_cell() {  # dim, enc, seed, gpu
  local dim=$1 enc=$2 sd=$3 g=$4
  local grid=outputs/benoit_chunkgrid_forced_${dim}_none
  local rw=${RW[$dim]}
  local out=outputs/eu_encoder/${dim}/${enc}/seed_${sd}
  if [ -f "$out/fno/leafq016/prediction_records/iter_02_post_eval.jsonl" ]; then
    LOG "  skip (done): $out"; return 0; fi
  mkdir -p "$(dirname "$out")"
  [ -d "$grid" ] || { LOG "MISSING grid $grid"; return 1; }
  CUDA_VISIBLE_DEVICES=$g TT_EXPORT_FULL_TREE_TRACES=0 \
    $PY scripts/run_manifesto_qsentence_dspy_ladder.py \
    --family fno --embedding-backend local-hf \
    --embedding-model "${ENC[$enc]}" \
    --embedding-device cuda --embedding-batch-size "${EBATCH[$enc]}" \
    --embedding-max-length 4096 --fno-device cuda \
    --fg-grid-dir "$grid" --leaf-qsentences 16 --max-iterations 2 \
    --fno-target-dimension "$dim" --eval-split test \
    "${ARCH[@]}" --fno-root-weight "$rw" \
    --fno-seed "$sd" --output-dir "$out" > "${out}.log" 2>&1 &
}

LOG "eu_encoder sweep: {gemma300m,qwen0_6b,qwen4b} x {eu,economic} x 3 seeds"
for dim in eu economic; do
  for enc in gemma300m qwen0_6b qwen4b; do
    for sd in "${SEEDS[@]}"; do
      launch_cell "$dim" "$enc" "$sd" "$GPU"
      GPU=$(( (GPU+1) % 4 ))
      [ "$GPU" -eq 0 ] && wait
    done
  done
done
wait
LOG "EU_ENCODER_SWEEP_COMPLETE"
