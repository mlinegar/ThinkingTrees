#!/usr/bin/env bash
# Round 7: properly-sized FNO-as-g.
#
# R6 hypothesis was wrong: length-2 FNO is degenerate. R7 reframes:
# spatial axis = rep_dim (state_dim_effective), channels = (left, right)
# lifted to ``merge_fno_hidden_channels``. With state_dim=256 / n_modes=32
# this is a real FNO with 32 spectral modes along the rep dim.
#
# Sweep: rep_dim × merge_family at the headline cell (102400 / leaf=64 /
# regime_oh / count_only / fully_learned / nass_jsd merge / linear
# decoder = R6 winner = 0.0227).
#
# Cells (8):
#   rep_dim       ∈ {50 (default 2*theta_dim), 128, 256}
#   merge_family  ∈ {mlp, fno_rep}
#   + 2 extras: fno_rep with deeper / more modes at rep_dim=256
set -euo pipefail
source venv/bin/activate

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_fno_round7_repdim_fno_g_${STAMP}}"
GPUS="${GPUS:-0,2,3}"
N_ITER="${N_ITER:-200}"
LEAF=64
TRAIN_DOCS=102400

BUNDLE="outputs/_bundles/markov_hazard_panels_train102400/paper_hazard_panel_v1_t128/seed_0/base_bundle.json"
if [ ! -f "$BUNDLE" ]; then echo "ERROR: bundle missing: $BUNDLE"; exit 1; fi

IFS=',' read -r -a GPU_LIST <<<"$GPUS"
N_GPUS="${#GPU_LIST[@]}"
mkdir -p "$OUT_ROOT/lanes"

# Cell spec: label|merge_family|rep_dim|fno_modes|fno_layers|fno_hidden
CELLS=(
  "mlp__rep050|mlp|0|16|2|32"
  "mlp__rep128|mlp|128|16|2|32"
  "mlp__rep256|mlp|256|16|2|32"
  "fno__rep050__m16|fno_rep|0|16|2|32"
  "fno__rep128__m32|fno_rep|128|32|2|32"
  "fno__rep256__m32|fno_rep|256|32|2|32"
  "fno__rep256__m64__l3|fno_rep|256|64|3|64"
  "fno__rep256__m32__hid64|fno_rep|256|32|2|64"
)
echo "Total cells: ${#CELLS[@]}  GPUs: $GPUS  N_iter: $N_ITER"

run_cell() {
  local label="$1" mfam="$2" repd="$3" modes="$4" layers="$5" hid="$6" gpu="$7"
  local out="$OUT_ROOT/$label"
  if [ -f "$out/summary.json" ]; then echo "[gpu$gpu] skip $label"; return 0; fi

  echo "[gpu$gpu] start $label (rep_dim=$repd merge=$mfam modes=$modes hid=$hid)"
  CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
  ./venv/bin/ctreepo sim run contextual-sbijax \
    --data-source markov --load-data-bundle "$BUNDLE" \
    --sbijax-trainer learned_local_laws --sbijax-method nasss --sbijax-package-theta markov_exact_sketch \
    --sbijax-input-encoding regime_one_hot --local-law-summary-family jax_fno \
    --local-law-summary-fno-n-modes 32 --local-law-summary-fno-n-layers 3 --local-law-summary-fno-pooling-mode sum \
    --law-architecture fully_learned --c2-merge-target theta \
    --merge-family "$mfam" \
    --merge-fno-n-modes "$modes" --merge-fno-n-layers "$layers" --merge-fno-hidden-channels "$hid" \
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
    --local-law-count-only --local-law-rep-dim "$repd" \
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
      IFS='|' read -r label mfam repd modes layers hid <<<"$cell_spec"
      run_cell "$label" "$mfam" "$repd" "$modes" "$layers" "$hid" "$gpu"
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
        "merge_fno_n_modes": prov.get("merge_fno_n_modes"),
        "merge_fno_n_layers": prov.get("merge_fno_n_layers"),
        "merge_fno_hidden_channels": prov.get("merge_fno_hidden_channels"),
        "local_law_rep_dim": prov.get("local_law_rep_dim"),
        "theta_count_raw_mae": test.get("theta_count_raw_mae"),
        "leaf_count_proxy": test.get("sufficiency_leaf_rep_count_proxy"),
    })
if rows:
    csv_path = os.path.join(root, "round7_repdim_fno_g_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")
PY

echo ">>> Round 7 complete: $OUT_ROOT"
