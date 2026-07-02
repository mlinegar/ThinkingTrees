#!/usr/bin/env bash
# Round 3 sbijax-internal sweep: new configurations using sbijax NASS/NASSS
# as actual training signal (not just diagnostic).
#
# Two new families layered on top of the existing unified sweep results:
#
#   * pkg_aux:  jax_fno + sketch + analytic + package_weight=0.5 + nasss
#               (the original "sbijax NASSS aux training") — now lifted on
#               jax_fno via the new package-aux head.
#   * merge_jsd: jax_fno + count_only + fully_learned + nass_jsd merge loss
#               (sbijax-style NASS contrastive loss INSIDE the merge step,
#               not just MSE).
#
# Both run at leaves {16, 32, 64, 128} on both encodings — the leaves where
# the sketch-vs-count-only difference was largest in the unified sweep.
set -euo pipefail
source venv/bin/activate

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_fno_round3_sbijax_internal_${STAMP}}"
BUNDLE="${BUNDLE:-outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json}"
GPUS="${GPUS:-0,1,2,3}"
N_ITER="${N_ITER:-4000}"

IFS=',' read -r -a GPU_LIST <<<"$GPUS"
N_GPUS="${#GPU_LIST[@]}"
mkdir -p "$OUT_ROOT/lanes"

CELLS=()
# Family A: jax_fno + sketch + analytic + sbijax NASSS aux training
# (the package_weight>0 path runs the actual sbijax.NASSS contrastive
# objective alongside the local-law signal during training).
for encoding in regime_one_hot one_hot_token_ids; do
  for leaf in 16 32 64 128; do
    CELLS+=("sbijax_nasss_aux__jax_fno_sketch__${encoding}__leaf${leaf}|jax_fno|$encoding|$leaf|analytic|theta|0|0.5|mse")
  done
done
# Family B: jax_fno + count_only + fully_learned + sbijax NASS-style merge
# (the merge_net is supervised by a sbijax NASS-style JSD MI lower bound
# on I(merge_state; merge_truth) instead of by MSE).
for encoding in regime_one_hot one_hot_token_ids; do
  for leaf in 16 32 64 128; do
    CELLS+=("sbijax_nass_merge__jax_fno_count_only__${encoding}__leaf${leaf}|jax_fno|$encoding|$leaf|fully_learned|theta|1|0.0|nass_jsd")
  done
done

echo "Total cells: ${#CELLS[@]}"

run_cell() {
  local label="$1" fam="$2" enc="$3" leaf="$4" arch="$5" c2="$6" count_only="$7" pkg_w="$8" merge_loss="$9" gpu="${10}"
  local out="$OUT_ROOT/$label"
  if [ -f "$out/summary.json" ]; then echo "[gpu$gpu] skip $label"; return 0; fi

  local n_modes
  if [ "$leaf" -le 16 ]; then n_modes=8
  elif [ "$leaf" -le 32 ]; then n_modes=16
  else n_modes=32
  fi

  local count_args=()
  local idemp_w=1.0
  if [ "$count_only" = "1" ]; then
    count_args+=(--local-law-count-only --local-law-rep-dim 0)
    idemp_w=0.0
  fi

  echo "[gpu$gpu] start $label"
  CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
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
    --learned-merge-hidden-dim 128 \
    --learned-decoder-hidden-dim 128 \
    --local-law-merge-loss "$merge_loss" \
    --train-docs 1024 --val-docs 256 --test-docs 256 \
    --fragment-len "$leaf" \
    --context-samples-per-doc 1 \
    --response-signature-contexts 16 --response-signature-slices 8 \
    --embedding-dim 64 --state-dim 25 --hidden-dim 128 \
    --learning-rate 0.0003 --lr-schedule cosine \
    --n-iter "$N_ITER" --batch-size 128 \
    --local-law-weight 1.0 --local-law-leaf-weight 1.0 --local-law-merge-weight 1.0 \
    --local-law-idempotence-weight "$idemp_w" \
    --local-law-contextual-weight 1.0 \
    --local-law-package-weight "$pkg_w" \
    --seed 0 \
    --output-root "$out" \
    "${count_args[@]}" >"$OUT_ROOT/lanes/${label}.log" 2>&1
  echo "[gpu$gpu] done $label"
}

declare -A LANE_PIDS
for ((i=0; i<N_GPUS; i++)); do
  gpu="${GPU_LIST[$i]}"
  (
    for ((j=i; j<${#CELLS[@]}; j+=N_GPUS)); do
      cell_spec="${CELLS[$j]}"
      IFS='|' read -r label fam enc leaf arch c2 co pkg ml <<<"$cell_spec"
      run_cell "$label" "$fam" "$enc" "$leaf" "$arch" "$c2" "$co" "$pkg" "$ml" "$gpu"
    done
  ) >"$OUT_ROOT/lanes/lane_gpu${gpu}.log" 2>&1 &
  LANE_PIDS[$gpu]=$!
done

for pid in "${LANE_PIDS[@]}"; do wait "$pid" || true; done
echo ">>> sbijax-internal sweep complete: $OUT_ROOT"
