#!/usr/bin/env bash
# jax_fno single-leaf / small-leaf sweep.
#
# Path A v4 covered leaves {16, 64, 128}. This fills in the small-leaf end
# {1, 2, 4, 8} so we see the self-contained JAX FNO behavior across the full
# leaf-size regime. fragment_len=1 disables the merge axis entirely (per
# project_general_fg_resolved.md) so it's the pure leaf-encoder test.
set -euo pipefail
source venv/bin/activate

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_fno_round3_jax_fno_small_leaf_${STAMP}}"
BUNDLE="${BUNDLE:-outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json}"
GPUS="${GPUS:-0,2,3}"
N_ITER="${N_ITER:-2000}"

IFS=',' read -r -a GPU_LIST <<<"$GPUS"
N_GPUS="${#GPU_LIST[@]}"
mkdir -p "$OUT_ROOT/lanes"

# Cells: jax_fno + mlp at small leaves on both encodings, plus a few
# regime_transition_sum control cells where applicable (leaf>=2).
CELLS=()
for encoding in regime_one_hot one_hot_token_ids; do
  for leaf in 1 2 4 8; do
    for fam in mlp jax_fno; do
      CELLS+=("${fam}__${encoding}__leaf${leaf}__analytic|$fam|$encoding|$leaf|analytic|theta|0")
    done
    if [ "$encoding" = "regime_one_hot" ] && [ "$leaf" -ge 2 ]; then
      CELLS+=("regime_transition_sum__${encoding}__leaf${leaf}__analytic|regime_transition_sum|$encoding|$leaf|analytic|theta|0")
    fi
  done
done

echo "Total cells: ${#CELLS[@]}"

run_cell() {
  local label="$1" fam="$2" enc="$3" leaf="$4" arch="$5" c2="$6" seed="$7" gpu="$8"
  local out="$OUT_ROOT/$label"
  if [ -f "$out/summary.json" ]; then echo "[gpu$gpu] skip $label"; return 0; fi

  local fno_args=()
  if [ "$fam" = "jax_fno" ] || [ "$fam" = "norax_fno" ]; then
    local n_modes
    if [ "$leaf" -le 2 ]; then n_modes=1
    elif [ "$leaf" -le 4 ]; then n_modes=2
    else n_modes=4
    fi
    fno_args+=(
      --local-law-summary-fno-n-modes "$n_modes"
      --local-law-summary-fno-n-layers 3
      --local-law-summary-fno-pooling-mode sum
    )
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
    --law-architecture "$arch" \
    --c2-merge-target "$c2" \
    --learned-merge-hidden-dim 128 \
    --learned-decoder-hidden-dim 128 \
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
    --output-root "$out" \
    "${fno_args[@]}" >"$OUT_ROOT/lanes/${label}.log" 2>&1
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
echo ">>> small-leaf sweep complete: $OUT_ROOT"
