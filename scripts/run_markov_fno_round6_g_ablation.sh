#!/usr/bin/env bash
# Round 6 g-ablation: NASS / NASSS / FNO-as-g / decoder-head sweep at the
# largest data scale (102400) and the headline leaf (64). All cells share
# the f-side setup that won R5 (jax_fno + count_only + regime_one_hot +
# fully_learned). The merge / decoder axes vary.
#
# Cells (12):
#   merge_family ∈ {mlp, fno_rep}
#   merge_loss   ∈ {mse, nass_jsd, nasss_jsd}
#   decoder_head ∈ {mlp, linear}
# = 2 × 3 × 2 = 12.
#
# Notes:
#   * count_only requires fully_learned arch (analytic decoder is
#     incompatible with arbitrary rep dim). So decoder_head=linear means
#     "linear final readout from the learned merged rep -> response."
#   * NASSS-merge slices merge_target onto 16 random unit projections.
set -euo pipefail
source venv/bin/activate

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_fno_round6_g_ablation_${STAMP}}"
GPUS="${GPUS:-0,2,3}"
N_ITER="${N_ITER:-200}"
TRAIN_DOCS="${TRAIN_DOCS:-102400}"
LEAF="${LEAF:-64}"

BUNDLE_BASE="outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json"
BUNDLE_102400="outputs/_bundles/markov_hazard_panels_train102400/paper_hazard_panel_v1_t128/seed_0/base_bundle.json"

if [ "$TRAIN_DOCS" -ge 102400 ]; then
  BUNDLE="$BUNDLE_102400"
else
  BUNDLE="$BUNDLE_BASE"
fi
if [ ! -f "$BUNDLE" ]; then
  echo "ERROR: bundle missing: $BUNDLE"; exit 1
fi

IFS=',' read -r -a GPU_LIST <<<"$GPUS"
N_GPUS="${#GPU_LIST[@]}"
mkdir -p "$OUT_ROOT/lanes"

# Cell spec format: label|merge_family|merge_loss|decoder_head
CELLS=()
for merge_family in mlp fno_rep; do
  for merge_loss in mse nass_jsd nasss_jsd; do
    for decoder_head in mlp linear; do
      label="g${merge_family}__loss${merge_loss}__head${decoder_head}"
      CELLS+=("${label}|${merge_family}|${merge_loss}|${decoder_head}")
    done
  done
done

echo "Total cells: ${#CELLS[@]}"
echo "GPUs: $GPUS (N=$N_GPUS)"
echo "Train docs: $TRAIN_DOCS  Leaf: $LEAF  N_iter: $N_ITER"
echo "Output: $OUT_ROOT"

# n_modes for the leaf-side FNO encoder (matches Round 5 schedule).
N_MODES=32
if [ "$LEAF" -le 16 ]; then N_MODES=8
elif [ "$LEAF" -le 32 ]; then N_MODES=16
fi

run_cell() {
  local label="$1" mfam="$2" mloss="$3" dhead="$4" gpu="$5"
  local out="$OUT_ROOT/$label"
  if [ -f "$out/summary.json" ]; then echo "[gpu$gpu] skip $label"; return 0; fi

  echo "[gpu$gpu] start $label"
  CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
  ./venv/bin/ctreepo sim run contextual-sbijax \
    --data-source markov --load-data-bundle "$BUNDLE" \
    --sbijax-trainer learned_local_laws --sbijax-method nasss --sbijax-package-theta markov_exact_sketch \
    --sbijax-input-encoding regime_one_hot --local-law-summary-family jax_fno \
    --local-law-summary-fno-n-modes "$N_MODES" --local-law-summary-fno-n-layers 3 --local-law-summary-fno-pooling-mode sum \
    --law-architecture fully_learned --c2-merge-target theta \
    --merge-family "$mfam" --merge-fno-n-modes 1 --merge-fno-n-layers 2 \
    --decoder-head "$dhead" \
    --local-law-merge-loss "$mloss" --merge-nasss-n-slices 16 \
    --learned-merge-hidden-dim 128 --learned-decoder-hidden-dim 128 \
    --train-docs "$TRAIN_DOCS" --val-docs 256 --test-docs 256 \
    --fragment-len "$LEAF" --context-samples-per-doc 1 \
    --response-signature-contexts 16 --response-signature-slices 8 \
    --embedding-dim 64 --state-dim 25 --hidden-dim 128 \
    --learning-rate 0.0003 --lr-schedule cosine \
    --n-iter "$N_ITER" --batch-size 128 \
    --local-law-weight 1.0 --local-law-leaf-weight 1.0 --local-law-merge-weight 1.0 \
    --local-law-idempotence-weight 0.0 --local-law-contextual-weight 1.0 --local-law-package-weight 0.0 \
    --local-law-count-only --local-law-rep-dim 0 \
    --seed 0 --output-root "$out" \
    >"$OUT_ROOT/lanes/${label}.log" 2>&1
  echo "[gpu$gpu] done $label"
}

declare -A LANE_PIDS
for ((i=0; i<N_GPUS; i++)); do
  gpu="${GPU_LIST[$i]}"
  (
    for ((j=i; j<${#CELLS[@]}; j+=N_GPUS)); do
      cell_spec="${CELLS[$j]}"
      IFS='|' read -r label mfam mloss dhead <<<"$cell_spec"
      run_cell "$label" "$mfam" "$mloss" "$dhead" "$gpu"
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
    if not os.path.exists(sj): continue
    try: d = json.load(open(sj))
    except Exception: continue
    test = d.get("diagnostics", {}).get("test", {})
    prov = d.get("provenance", {})
    rows.append({
        "cell": cell,
        "merge_family": prov.get("merge_family"),
        "merge_loss": prov.get("local_law_merge_loss"),
        "decoder_head": prov.get("decoder_head"),
        "fragment_len": prov.get("fragment_len"),
        "n_train_docs": prov.get("n_train_docs") or prov.get("train_docs"),
        "supervision_mode": test.get("supervision_mode"),
        "theta_count_raw_mae": test.get("theta_count_raw_mae"),
        "leaf_count_proxy": test.get("sufficiency_leaf_rep_count_proxy"),
        "merge_count_proxy": test.get("sufficiency_merge_rep_count_proxy"),
    })
if rows:
    csv_path = os.path.join(root, "round6_g_ablation_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")
PY

echo ">>> Round 6 g-ablation complete: $OUT_ROOT"
