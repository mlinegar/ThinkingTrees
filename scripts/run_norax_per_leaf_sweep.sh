#!/usr/bin/env bash
# jax_fno full-leaf-range sweep with per-leaf diagnostics.
#
# Push the self-contained JAX FNO summary across the full leaf-size axis
# (1...128) on both regime_one_hot and raw one_hot_token_ids encodings under the
# learned_local_laws lane. Now that contextual_sbijax.py exposes
# per-leaf scores (per_leaf_theta_mae, per_leaf_count_raw_abs_err,
# per_leaf_law_loss, per_merge_law_loss, per_idempotence_law_loss, plus
# quantile summaries), every cell emits the per-leaf distribution alongside
# the aggregates we've been tracking.
#
# Higher n_iter than path A v4 (4000 vs 2000) to give the FNO summary more
# time to converge — we're "pushing" the JAX FNO here, not just sweeping.
set -euo pipefail
source venv/bin/activate

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_fno_round3_jax_fno_per_leaf_${STAMP}}"
BUNDLE="${BUNDLE:-outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json}"
GPUS="${GPUS:-0,2,3}"
N_ITER="${N_ITER:-4000}"

IFS=',' read -r -a GPU_LIST <<<"$GPUS"
N_GPUS="${#GPU_LIST[@]}"
mkdir -p "$OUT_ROOT/lanes"

# Cells: jax_fno × {regime_one_hot, one_hot_token_ids} × {1,2,4,8,16,32,64,128}
CELLS=()
for encoding in regime_one_hot one_hot_token_ids; do
  for leaf in 1 2 4 8 16 32 64 128; do
    CELLS+=("jax_fno__${encoding}__leaf${leaf}__analytic|jax_fno|$encoding|$leaf|analytic|theta|0")
  done
done
echo "Total cells: ${#CELLS[@]}"

run_cell() {
  local label="$1" fam="$2" enc="$3" leaf="$4" arch="$5" c2="$6" seed="$7" gpu="$8"
  local out="$OUT_ROOT/$label"
  if [ -f "$out/summary.json" ]; then echo "[gpu$gpu] skip $label"; return 0; fi

  # FNO modes scaled by leaf size (clamped automatically to L//2+1).
  local n_modes
  if [ "$leaf" -le 2 ]; then n_modes=1
  elif [ "$leaf" -le 4 ]; then n_modes=2
  elif [ "$leaf" -le 16 ]; then n_modes=8
  elif [ "$leaf" -le 32 ]; then n_modes=16
  else n_modes=32
  fi

  echo "[gpu$gpu] start $label"
  CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
  ./venv/bin/ctreepo sim run contextual-sbijax \
    --data-source markov \
    --load-data-bundle "$BUNDLE" \
    --sbijax-trainer learned_local_laws \
    --sbijax-method nasss \
    --sbijax-package-theta markov_exact_sketch \
    --sbijax-input-encoding "$enc" \
    --local-law-summary-family "$fam" \
    --local-law-summary-fno-n-modes "$n_modes" \
    --local-law-summary-fno-n-layers 3 \
    --local-law-summary-fno-pooling-mode sum \
    --law-architecture "$arch" \
    --c2-merge-target "$c2" \
    --train-docs 1024 --val-docs 256 --test-docs 256 \
    --fragment-len "$leaf" \
    --context-samples-per-doc 1 \
    --response-signature-contexts 16 --response-signature-slices 8 \
    --embedding-dim 64 --state-dim 25 --hidden-dim 128 \
    --learning-rate 0.0003 --lr-schedule cosine \
    --n-iter "$N_ITER" --batch-size 128 \
    --local-law-weight 1.0 --local-law-leaf-weight 1.0 --local-law-merge-weight 1.0 \
    --local-law-idempotence-weight 1.0 --local-law-contextual-weight 1.0 --local-law-package-weight 0.0 \
    --seed "$seed" \
    --output-root "$out" >"$OUT_ROOT/lanes/${label}.log" 2>&1
  echo "[gpu$gpu] done $label"
}

declare -A LANE_PIDS
for ((i=0; i<N_GPUS; i++)); do
  gpu="${GPU_LIST[$i]}"
  (
    for ((j=i; j<${#CELLS[@]}; j+=N_GPUS)); do
      cell_spec="${CELLS[$j]}"
      IFS='|' read -r label fam enc leaf arch c2 seed <<<"$cell_spec"
      run_cell "$label" "$fam" "$enc" "$leaf" "$arch" "$c2" "$seed" "$gpu"
    done
  ) >"$OUT_ROOT/lanes/lane_gpu${gpu}.log" 2>&1 &
  LANE_PIDS[$gpu]=$!
done

for pid in "${LANE_PIDS[@]}"; do wait "$pid" || true; done
echo ">>> jax_fno per-leaf sweep complete: $OUT_ROOT"
