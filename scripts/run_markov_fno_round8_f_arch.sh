#!/usr/bin/env bash
# Round 8: f-side architecture sweep at the headline cell.
#
# R6 + R7 ruled out g as the bottleneck (count_mae stays in
# 0.022-0.045 across 20 g-side cells). Headline residual to the
# architectural ceiling (0.0005) lives on the f side. R8 directly
# probes f by varying the leaf-FNO encoder.
#
# Hold:
#   train_docs=102400, leaf=64, regime_one_hot, count_only,
#   fully_learned, mlp + nass_jsd + linear + rep_dim=0 (R7 winner).
#
# Vary (12 cells):
#   n_modes  ∈ {16, 32, 64}  × n_layers ∈ {2, 3, 4}     at pool=sum  (9)
#   n_modes  ∈ {16, 32, 64}  × n_layers = 3             at pool=mean (3)
#
# Pooling-mode validator only accepts {sum, mean}.
set -euo pipefail
source venv/bin/activate

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_fno_round8_f_arch_${STAMP}}"
GPUS="${GPUS:-0,1}"
N_ITER="${N_ITER:-200}"
TRAIN_DOCS=102400
LEAF=64

BUNDLE="outputs/_bundles/markov_hazard_panels_train102400/paper_hazard_panel_v1_t128/seed_0/base_bundle.json"
if [ ! -f "$BUNDLE" ]; then echo "ERROR: bundle missing: $BUNDLE"; exit 1; fi

IFS=',' read -r -a GPU_LIST <<<"$GPUS"
N_GPUS="${#GPU_LIST[@]}"
mkdir -p "$OUT_ROOT/lanes"

# Cell spec: label|n_modes|n_layers|pool
CELLS=()
for nm in 16 32 64; do
  for nl in 2 3 4; do
    label="m${nm}__l${nl}__psum"
    CELLS+=("${label}|${nm}|${nl}|sum")
  done
done
for nm in 16 32 64; do
  label="m${nm}__l3__pmean"
  CELLS+=("${label}|${nm}|3|mean")
done

echo "Total cells: ${#CELLS[@]}  GPUs: $GPUS  N_iter: $N_ITER  Output: $OUT_ROOT"

run_cell() {
  local label="$1" nm="$2" nl="$3" pool="$4" gpu="$5"
  local out="$OUT_ROOT/$label"
  if [ -f "$out/summary.json" ]; then echo "[gpu$gpu] skip $label"; return 0; fi

  echo "[gpu$gpu] start $label (modes=$nm layers=$nl pool=$pool)"
  CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
  ./venv/bin/ctreepo sim run contextual-sbijax \
    --data-source markov --load-data-bundle "$BUNDLE" \
    --sbijax-trainer learned_local_laws --sbijax-method nasss --sbijax-package-theta markov_exact_sketch \
    --sbijax-input-encoding regime_one_hot --local-law-summary-family jax_fno \
    --local-law-summary-fno-n-modes "$nm" --local-law-summary-fno-n-layers "$nl" --local-law-summary-fno-pooling-mode "$pool" \
    --law-architecture fully_learned --c2-merge-target theta \
    --merge-family mlp --merge-fno-n-modes 16 --merge-fno-n-layers 2 --merge-fno-hidden-channels 32 \
    --decoder-head linear \
    --local-law-merge-loss nass_jsd --merge-nasss-n-slices 16 \
    --learned-merge-hidden-dim 128 --learned-decoder-hidden-dim 128 \
    --train-docs $TRAIN_DOCS --val-docs 256 --test-docs 256 \
    --fragment-len $LEAF --context-samples-per-doc 1 \
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
      IFS='|' read -r label nm nl pool <<<"$cell_spec"
      run_cell "$label" "$nm" "$nl" "$pool" "$gpu"
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
        "summary_fno_n_modes": prov.get("local_law_summary_fno_n_modes"),
        "summary_fno_n_layers": prov.get("local_law_summary_fno_n_layers"),
        "summary_fno_pool": prov.get("local_law_summary_fno_pooling_mode"),
        "theta_count_raw_mae": test.get("theta_count_raw_mae"),
        "theta_first_regime_acc": test.get("theta_first_regime_accuracy"),
        "theta_last_regime_acc": test.get("theta_last_regime_accuracy"),
        "leaf_count_proxy": test.get("sufficiency_leaf_rep_count_proxy"),
        "merge_count_proxy": test.get("sufficiency_merge_rep_count_proxy"),
    })
if rows:
    csv_path = os.path.join(root, "round8_f_arch_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")
PY

echo ">>> Round 8 complete: $OUT_ROOT"
