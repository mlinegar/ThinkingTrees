#!/usr/bin/env bash
# Round 4 sbijax ablations: feature sbijax more prominently across the
# pipeline, with targeted ablations on the axes we've left untested.
#
# Tier 1 — headline:
#   1A pure_sbijax    : NASSS aux alone, laws OFF (local_law_weight=0).
#                       Tests "does sbijax recover sufficiency without
#                       laws as the structural tie-breaker?"
#                       9 cells (incl. 1 count_only failure-mode datapoint).
#   1B sbijax_everywhere : NASSS aux + nass_jsd merge + count_only +
#                       fully_learned. Maximally feature sbijax in one
#                       configuration.
#                       8 cells.
#
# Tier 2 — ablation axes:
#   2C nass_aux       : --sbijax-method nass (NASS InfoNCE) at matched
#                       setup vs the existing NASSS sweep.
#                       8 cells.
#   2D weight_sweep   : NASSS aux at package_weight ∈ {0.1, 1.0, 2.0} on
#                       jax_fno + sketch + analytic.
#                       6 cells (regime_one_hot leaf={32, 128} only).
#
# Tier 3 — bonus:
#   3E standalone_trainer: --sbijax-trainer package + nass_nle.
#                       Run "pure sbijax inference" without our hybrid
#                       lane. 4 cells (regime_one_hot leaf={32, 128}).
#
# Tier 4 — truly pure sbijax (added after discovering the analytic
# decoder + contextual loss provides implicit slot-aligned supervision
# even with laws=0 — making Tier 1A "pure sbijax" not actually pure):
#   4F pure_nasss_learned_decoder : laws=0 + fully_learned arch
#                       (no analytic decoder/merge slot pressure) +
#                       contextual=1 + NASSS aux=1. The rep is shaped
#                       only by NASSS contrastive + response MSE through
#                       a learned decoder that doesn't impose slot
#                       allocation. 8 cells.
#   4G only_nasss_learned : laws=0 + fully_learned + contextual=0 +
#                       NASSS=1. Only NASSS contrastive shapes the rep —
#                       no other supervision at all. The truest pure-
#                       sbijax baseline. 8 cells.
#
# Total: 35 + 16 = 51 cells; ~70 min on 4 GPUs at n_iter=4000.
set -euo pipefail
source venv/bin/activate

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_fno_round4_sbijax_ablations_${STAMP}}"
BUNDLE="${BUNDLE:-outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json}"
GPUS="${GPUS:-0,1,2,3}"
N_ITER="${N_ITER:-4000}"

IFS=',' read -r -a GPU_LIST <<<"$GPUS"
N_GPUS="${#GPU_LIST[@]}"
mkdir -p "$OUT_ROOT/lanes"

# Cell spec format:
# label|family|encoding|leaf|arch|c2|count_only|pkg_w|merge_loss|method|law_weight|trainer
CELLS=()

# Tier 1A: pure_sbijax (laws OFF, NASSS aux ONLY) on jax_fno + sketch
for encoding in regime_one_hot one_hot_token_ids; do
  for leaf in 16 32 64 128; do
    CELLS+=("sbijax_pure_nasss__jax_fno_sketch__${encoding}__leaf${leaf}|jax_fno|$encoding|$leaf|analytic|theta|0|1.0|mse|nasss|0.0|learned_local_laws")
  done
done
# 1A 9th cell: pure_sbijax + count_only (failure-mode datapoint)
CELLS+=("sbijax_pure_nasss_count_only__jax_fno__regime_one_hot__leaf64|jax_fno|regime_one_hot|64|fully_learned|theta|1|1.0|mse|nasss|0.0|learned_local_laws")

# Tier 1B: sbijax_everywhere (NASSS aux + nass_jsd merge + count_only)
for encoding in regime_one_hot one_hot_token_ids; do
  for leaf in 16 32 64 128; do
    CELLS+=("sbijax_everywhere__jax_fno_count_only__${encoding}__leaf${leaf}|jax_fno|$encoding|$leaf|fully_learned|theta|1|0.5|nass_jsd|nasss|1.0|learned_local_laws")
  done
done

# Tier 2C: NASS (InfoNCE, non-sliced) at matched setup vs NASSS sweep
for encoding in regime_one_hot one_hot_token_ids; do
  for leaf in 16 32 64 128; do
    CELLS+=("sbijax_nass_aux__jax_fno_sketch__${encoding}__leaf${leaf}|jax_fno|$encoding|$leaf|analytic|theta|0|0.5|mse|nass|1.0|learned_local_laws")
  done
done

# Tier 2D: NASSS weight sweep (regime_one_hot, leaves 32 and 128)
for w in 0.1 1.0 2.0; do
  for leaf in 32 128; do
    label="sbijax_nasss_aux_w${w}__jax_fno_sketch__regime_one_hot__leaf${leaf}"
    CELLS+=("${label}|jax_fno|regime_one_hot|$leaf|analytic|theta|0|$w|mse|nasss|1.0|learned_local_laws")
  done
done

# Tier 3E: standalone sbijax trainer comparison (regime_one_hot, leaf=32, 128)
for trainer in package nass_nle; do
  for leaf in 32 128; do
    CELLS+=("sbijax_trainer_${trainer}__regime_one_hot__leaf${leaf}|mlp|regime_one_hot|$leaf|analytic|theta|0|1.0|mse|nasss|1.0|$trainer")
  done
done

# Tier 4F: truly pure sbijax — learned decoder removes the analytic-
# decoder implicit slot pressure. laws=0, contextual=1, NASSS=1.
# Cell label adds `_learned_dec` so it's distinguishable from the
# analytic-decoder pure_nasss cells in Tier 1A.
for encoding in regime_one_hot one_hot_token_ids; do
  for leaf in 16 32 64 128; do
    CELLS+=("sbijax_pure_nasss_learned_dec__jax_fno__${encoding}__leaf${leaf}|jax_fno|$encoding|$leaf|fully_learned|theta|0|1.0|mse|nasss|0.0|learned_local_laws")
  done
done

# Tier 4G: only NASSS — laws=0 AND contextual=0 AND learned decoder.
# Only the NASSS contrastive aux shapes the rep. No analytic decoder
# slot pressure, no contextual MSE pressure, no laws. The truest pure
# sbijax baseline.
# Note: contextual_weight=0 is set inline below since the launcher's
# default is 1.0; we only override for these cells.
for encoding in regime_one_hot one_hot_token_ids; do
  for leaf in 16 32 64 128; do
    CELLS+=("sbijax_only_nasss_learned_dec__jax_fno__${encoding}__leaf${leaf}|jax_fno|$encoding|$leaf|fully_learned|theta|0|1.0|mse|nasss|0.0|learned_local_laws")
  done
done

echo "Total cells: ${#CELLS[@]}"

run_cell() {
  local label="$1" fam="$2" enc="$3" leaf="$4" arch="$5" c2="$6"
  local count_only="$7" pkg_w="$8" merge_loss="$9" method="${10}"
  local law_w="${11}" trainer="${12}" gpu="${13}"
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
  # Tier 4G: only_nasss cells turn off contextual loss as well — only
  # NASSS contrastive shapes the rep.
  local contextual_w=1.0
  if [[ "$label" == sbijax_only_nasss* ]]; then
    contextual_w=0.0
  fi

  echo "[gpu$gpu] start $label"
  CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
  ./venv/bin/ctreepo sim run contextual-sbijax \
    --data-source markov \
    --load-data-bundle "$BUNDLE" \
    --sbijax-trainer "$trainer" \
    --sbijax-method "$method" \
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
    --local-law-weight "$law_w" \
    --local-law-leaf-weight 1.0 --local-law-merge-weight 1.0 \
    --local-law-idempotence-weight "$idemp_w" \
    --local-law-contextual-weight "$contextual_w" \
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
      IFS='|' read -r label fam enc leaf arch c2 co pkg ml meth lw tr <<<"$cell_spec"
      run_cell "$label" "$fam" "$enc" "$leaf" "$arch" "$c2" "$co" "$pkg" "$ml" "$meth" "$lw" "$tr" "$gpu"
    done
  ) >"$OUT_ROOT/lanes/lane_gpu${gpu}.log" 2>&1 &
  LANE_PIDS[$gpu]=$!
done

for pid in "${LANE_PIDS[@]}"; do wait "$pid" || true; done

# Aggregate to CSV with cross-sweep schema
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
        "family": prov.get("local_law_summary_family"),
        "input_encoding": d.get("input_encoding"),
        "trainer": prov.get("trainer"),
        "method": prov.get("method"),
        "law_arch": prov.get("law_architecture"),
        "count_only": prov.get("local_law_count_only"),
        "pkg_weight": prov.get("local_law_package_weight"),
        "law_weight": prov.get("local_law_weight"),
        "merge_loss": prov.get("local_law_merge_loss"),
        "supervision_mode": test.get("supervision_mode"),
        "theta_count_raw_mae": test.get("theta_count_raw_mae"),
        "theta_first_regime_accuracy": test.get("theta_first_regime_accuracy"),
        "theta_last_regime_accuracy": test.get("theta_last_regime_accuracy"),
        "leaf_count_jsd": test.get("sufficiency_leaf_rep_count_jsd_loss"),
        "leaf_count_proxy": test.get("sufficiency_leaf_rep_count_proxy"),
        "merge_count_jsd": test.get("sufficiency_merge_rep_count_jsd_loss"),
        "merge_count_proxy": test.get("sufficiency_merge_rep_count_proxy"),
    })

if rows:
    fields = list(rows[0].keys())
    csv_path = os.path.join(root, "round4_sbijax_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")
PY

echo ">>> Round 4 sbijax ablations complete: $OUT_ROOT"
