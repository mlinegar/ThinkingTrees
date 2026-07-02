#!/usr/bin/env bash
# Round 3 path A: native JAX pipeline — sbijax NASS/NASSS + JAX FNO + laws.
#
# Frame (per docs/markov_sim_status.md "Original JAX Purpose"): generic neural
# sufficient summaries (sbijax.NASS/NASSS) by themselves do NOT pick the
# canonical Markov sketch; local laws are the structural tie-breaker. This
# phase tests whether that framework holds when the summary network is a
# proper self-contained JAX FNO encoder instead of a flat MLP — the FNO summary
# is the JAX analogue of the PyTorch CleanUnifiedNO leaf encoder.
#
# Compare summary families under matched supervision:
#   * mlp                   — flat MLP (Round 1 baseline; degrades at large leaves)
#   * jax_fno               — repo-owned JAX FFT FNO + enriched pool
#                             (sum + first + last concat). `norax_fno` remains
#                             a backward-compatible alias but does not import
#                             norax.
#   * regime_transition_sum — structured boundary-counter (architectural prior;
#                             control / upper bound)
#
# The route is self-contained: norax and pardax are design references, not
# runtime dependencies.
#
# Plus two input encodings:
#   * regime_one_hot       — regime IDs as one-hot per position
#   * one_hot_token_ids    — raw token IDs (regime is latent; harder)
#
# All under learned_local_laws + sbijax NASSS supervision (laws as
# tie-breaker; package_weight kept 0 since regime_transition_sum and FNO
# require it).
set -euo pipefail
source venv/bin/activate

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_fno_round3_path_a_jax_${STAMP}}"
BUNDLE="${BUNDLE:-outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json}"
GPUS="${GPUS:-0,2,3}"
N_ITER="${N_ITER:-2000}"
TRAIN_DOCS="${TRAIN_DOCS:-1024}"
VAL_DOCS="${VAL_DOCS:-256}"
TEST_DOCS="${TEST_DOCS:-256}"

IFS=',' read -r -a GPU_LIST <<<"$GPUS"
N_GPUS="${#GPU_LIST[@]}"

mkdir -p "$OUT_ROOT/logs"

# Build the cell list (label, family, encoding, fragment_len, arch, c2, seed)
CELLS=()
for encoding in regime_one_hot one_hot_token_ids; do
  for leaf in 16 64 128; do
    # mlp baseline + self-contained JAX FNO headline.
    for fam in mlp jax_fno; do
      CELLS+=("${fam}__${encoding}__leaf${leaf}__analytic|$fam|$encoding|$leaf|analytic|theta|0")
    done
    # regime_transition_sum control (regime_one_hot only)
    if [ "$encoding" = "regime_one_hot" ]; then
      CELLS+=("regime_transition_sum__${encoding}__leaf${leaf}__analytic|regime_transition_sum|$encoding|$leaf|analytic|theta|0")
    fi
  done
done

# jax_fno winners with learned_merge — only at the most informative leaves.
for leaf in 32 64; do
  CELLS+=("jax_fno__regime_one_hot__leaf${leaf}__learned_merge|jax_fno|regime_one_hot|$leaf|learned_merge|self_consistency|0")
done

echo "Total cells: ${#CELLS[@]}"

run_jax_cell() {
  local label="$1" fam="$2" enc="$3" leaf="$4" arch="$5" c2="$6" seed="$7" gpu="$8"
  local out="$OUT_ROOT/$label"
  if [ -f "$out/summary.json" ]; then
    echo "[gpu$gpu] skip $label"
    return 0
  fi

  local fno_args=()
  if [ "$fam" = "jax_fno" ] || [ "$fam" = "norax_fno" ]; then
    # FNO modes scaled by leaf size (clamped automatically to L//2+1).
    local n_modes
    if [ "$leaf" -le 16 ]; then n_modes=8
    elif [ "$leaf" -le 32 ]; then n_modes=16
    else n_modes=32
    fi
    fno_args+=(
      --local-law-summary-fno-n-modes "$n_modes"
      --local-law-summary-fno-n-layers 3
      --local-law-summary-fno-pooling-mode sum
    )
  fi

  echo "[gpu$gpu] start $label  fam=$fam enc=$enc leaf=$leaf arch=$arch c2=$c2 seed=$seed"

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
    --train-docs "$TRAIN_DOCS" --val-docs "$VAL_DOCS" --test-docs "$TEST_DOCS" \
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
    "${fno_args[@]}" >"$OUT_ROOT/logs/${label}.log" 2>&1
  echo "[gpu$gpu] done $label"
}

# Round-robin dispatch: each GPU gets a subset of cells, runs them serially.
mkdir -p "$OUT_ROOT/lanes"
declare -A LANE_PIDS
for ((i=0; i<N_GPUS; i++)); do
  gpu="${GPU_LIST[$i]}"
  (
    for ((j=i; j<${#CELLS[@]}; j+=N_GPUS)); do
      cell_spec="${CELLS[$j]}"
      IFS='|' read -r label fam enc leaf arch c2 seed <<<"$cell_spec"
      run_jax_cell "$label" "$fam" "$enc" "$leaf" "$arch" "$c2" "$seed" "$gpu"
    done
  ) >"$OUT_ROOT/lanes/lane_gpu${gpu}.log" 2>&1 &
  LANE_PIDS[$gpu]=$!
  echo "lane gpu=$gpu pid=${LANE_PIDS[$gpu]} log=$OUT_ROOT/lanes/lane_gpu${gpu}.log"
done

for pid in "${LANE_PIDS[@]}"; do
  wait "$pid" || true
done

echo ">>> Aggregating path A results"
./venv/bin/python - <<PY
import json, csv, os
root = "$OUT_ROOT"
rows = []
for cell in sorted(os.listdir(root)):
    sj = os.path.join(root, cell, "summary.json")
    if not os.path.exists(sj):
        continue
    try:
        d = json.load(open(sj))
    except Exception as exc:
        print(f"  skip {sj}: {exc}")
        continue
    test = d.get("diagnostics", {}).get("test", {})
    prov = d.get("provenance", {})
    cfg = d.get("config", {}) or {}
    rows.append({
        "cell": cell,
        "summary_family": cfg.get("local_law_summary_family") or prov.get("local_law_summary_family"),
        "input_encoding": d.get("input_encoding"),
        "fragment_len": prov.get("fragment_len"),
        "law_architecture": prov.get("law_architecture"),
        "c2_merge_target": prov.get("c2_merge_target"),
        "fno_n_modes": cfg.get("local_law_summary_fno_n_modes") or prov.get("local_law_summary_fno_n_modes"),
        "fno_n_layers": cfg.get("local_law_summary_fno_n_layers") or prov.get("local_law_summary_fno_n_layers"),
        "fno_pooling_mode": cfg.get("local_law_summary_fno_pooling_mode") or prov.get("local_law_summary_fno_pooling_mode"),
        "theta_mae": test.get("theta_mae"),
        "raw_count_mae": test.get("theta_count_raw_mae"),
        "contextual_mae": test.get("contextual_mae"),
        "contextual_raw_mae": test.get("contextual_raw_mae"),
        "theta_first_regime_accuracy": test.get("theta_first_regime_accuracy"),
        "theta_last_regime_accuracy": test.get("theta_last_regime_accuracy"),
        "eps_leaf": test.get("eps_leaf"),
        "eps_merge": test.get("eps_merge"),
        "eps_idemp": test.get("eps_idemp"),
    })

if rows:
    fields = list(rows[0].keys())
    csv_path = os.path.join(root, "path_a_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")
else:
    print("no summary.json files found")
PY

echo ">>> Path A complete: $OUT_ROOT"
