#!/usr/bin/env bash
# Round 3 unified sweep: single launcher across all 4 GPUs covering the
# four supervision/encoder families we want to compare side by side, with
# the new sbijax-style JSD MI sufficiency diagnostic on every cell (both
# the strict full-sketch I(rep; (count, first, last)) and the f*-aligned
# I(rep; count) flavors). Per-leaf scores, leaf structure, and per-merge
# sufficiency all land in summary.json.
#
# Families:
#   * jax_fno + count_only + fully_learned  — headline f*-supervision test
#                                              (count_readout, no sketch
#                                              prior baked in; idempotence
#                                              disabled in this mode since
#                                              it encodes a discrete-count
#                                              assumption).
#   * jax_fno + sketch + analytic           — sketch-supervised FNO baseline
#                                              (the "supervised on
#                                              (count, first, last)"
#                                              comparison).
#   * mlp + sketch + analytic               — flat-MLP family baseline.
#   * regime_transition_sum + sketch + analytic — architectural-prior
#                                              control (count = sum of
#                                              adjacency MLP scores;
#                                              regime_one_hot only).
#
# Encodings: regime_one_hot (regime IDs given) + one_hot_token_ids (raw
# tokens; regime is latent — the genuinely flexible-encoder test).
#
# Leaves: 1, 2, 4, 8, 16, 32, 64, 128.
set -euo pipefail
source venv/bin/activate

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_fno_round3_unified_${STAMP}}"
BUNDLE="${BUNDLE:-outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json}"
GPUS="${GPUS:-0,1,2,3}"
N_ITER="${N_ITER:-4000}"

IFS=',' read -r -a GPU_LIST <<<"$GPUS"
N_GPUS="${#GPU_LIST[@]}"
mkdir -p "$OUT_ROOT/lanes"

# Cells: (label, family, encoding, leaf, arch, c2, count_only, seed)
CELLS=()

# Family 1: jax_fno + count_only + fully_learned (headline)
for encoding in regime_one_hot one_hot_token_ids; do
  for leaf in 1 2 4 8 16 32 64 128; do
    CELLS+=("jax_fno_count_only__${encoding}__leaf${leaf}|jax_fno|$encoding|$leaf|fully_learned|theta|1|0")
  done
done

# Family 2: jax_fno + sketch + analytic
for encoding in regime_one_hot one_hot_token_ids; do
  for leaf in 1 2 4 8 16 32 64 128; do
    CELLS+=("jax_fno_sketch__${encoding}__leaf${leaf}|jax_fno|$encoding|$leaf|analytic|theta|0|0")
  done
done

# Family 3: mlp + sketch + analytic (baseline)
for encoding in regime_one_hot one_hot_token_ids; do
  for leaf in 1 2 4 8 16 32 64 128; do
    CELLS+=("mlp_sketch__${encoding}__leaf${leaf}|mlp|$encoding|$leaf|analytic|theta|0|0")
  done
done

# Family 4: regime_transition_sum + sketch + analytic (architectural ceiling, regime_one_hot only)
for leaf in 2 4 8 16 32 64 128; do
  CELLS+=("regime_transition_sum_sketch__regime_one_hot__leaf${leaf}|regime_transition_sum|regime_one_hot|$leaf|analytic|theta|0|0")
done

echo "Total cells: ${#CELLS[@]}"

run_cell() {
  local label="$1" fam="$2" enc="$3" leaf="$4" arch="$5" c2="$6" count_only="$7" seed="$8" gpu="$9"
  local out="$OUT_ROOT/$label"
  if [ -f "$out/summary.json" ]; then echo "[gpu$gpu] skip $label"; return 0; fi

  local fno_args=()
  if [ "$fam" = "jax_fno" ]; then
    local n_modes
    if [ "$leaf" -le 2 ]; then n_modes=1
    elif [ "$leaf" -le 4 ]; then n_modes=2
    elif [ "$leaf" -le 16 ]; then n_modes=8
    elif [ "$leaf" -le 32 ]; then n_modes=16
    else n_modes=32
    fi
    fno_args+=(
      --local-law-summary-fno-n-modes "$n_modes"
      --local-law-summary-fno-n-layers 3
      --local-law-summary-fno-pooling-mode sum
    )
  fi

  # Idempotence: disabled (weight 0) in count-only mode because the C3
  # round-then-distance term encodes a Markov-specific integer-count
  # assumption. Sketch-supervised cells retain idempotence at weight 1.
  local idemp_w=1.0
  local count_args=()
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
    --local-law-idempotence-weight "$idemp_w" \
    --local-law-contextual-weight 1.0 --local-law-package-weight 0.0 \
    --seed "$seed" \
    --output-root "$out" \
    "${fno_args[@]}" "${count_args[@]}" >"$OUT_ROOT/lanes/${label}.log" 2>&1
  echo "[gpu$gpu] done $label"
}

declare -A LANE_PIDS
for ((i=0; i<N_GPUS; i++)); do
  gpu="${GPU_LIST[$i]}"
  (
    for ((j=i; j<${#CELLS[@]}; j+=N_GPUS)); do
      cell_spec="${CELLS[$j]}"
      IFS='|' read -r label fam enc leaf arch c2 count_only seed <<<"$cell_spec"
      run_cell "$label" "$fam" "$enc" "$leaf" "$arch" "$c2" "$count_only" "$seed" "$gpu"
    done
  ) >"$OUT_ROOT/lanes/lane_gpu${gpu}.log" 2>&1 &
  LANE_PIDS[$gpu]=$!
done

for pid in "${LANE_PIDS[@]}"; do wait "$pid" || true; done

# Aggregate
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
    rows.append({
        "cell": cell,
        "summary_family": prov.get("local_law_summary_family"),
        "input_encoding": d.get("input_encoding"),
        "fragment_len": prov.get("fragment_len"),
        "law_architecture": prov.get("law_architecture"),
        "count_only": prov.get("local_law_count_only"),
        "supervision_mode": test.get("supervision_mode"),
        "theta_count_raw_mae": test.get("theta_count_raw_mae"),
        "theta_first_regime_accuracy": test.get("theta_first_regime_accuracy"),
        "theta_last_regime_accuracy": test.get("theta_last_regime_accuracy"),
        "eps_leaf": test.get("eps_leaf"),
        "eps_merge": test.get("eps_merge"),
        # NEW sufficiency diagnostics — these are the headline columns now.
        "leaf_rep_jsd_loss":          test.get("sufficiency_leaf_rep_jsd_loss"),
        "leaf_rep_proxy":             test.get("sufficiency_leaf_rep_proxy"),
        "leaf_rep_count_jsd_loss":    test.get("sufficiency_leaf_rep_count_jsd_loss"),
        "leaf_rep_count_proxy":       test.get("sufficiency_leaf_rep_count_proxy"),
        "merge_rep_jsd_loss":         test.get("sufficiency_merge_rep_jsd_loss"),
        "merge_rep_proxy":            test.get("sufficiency_merge_rep_proxy"),
        "merge_rep_count_jsd_loss":   test.get("sufficiency_merge_rep_count_jsd_loss"),
        "merge_rep_count_proxy":      test.get("sufficiency_merge_rep_count_proxy"),
    })

if rows:
    fields = list(rows[0].keys())
    csv_path = os.path.join(root, "unified_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")
PY

echo ">>> unified Round 3 sweep complete: $OUT_ROOT"
