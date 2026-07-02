#!/usr/bin/env bash
# Round 10: stack the R8 architecture winner with the R9 compute winner.
#
# R8 (n_iter=200): m64/l4/sum = 0.0117 (best architecture).
# R9 (m32/l3/sum):  n_iter=1500 = 0.0140 (compute saturates ~1500).
# R10 stacks: wider/deeper f-side at n_iter=1500.
#
# Hold: 102400 docs, leaf=64, regime_one_hot, count_only, fully_learned,
#       mlp + nass_jsd + linear + rep_dim=0.
#
# Vary (5 cells):
#   modes=64 / layers=4 / pool=sum   ← R8 winner + R9 compute (headline)
#   modes=64 / layers=4 / pool=mean  ← untested at l=4 (mean won at l=3)
#   modes=64 / layers=5 / pool=sum   ← deeper f
#   modes=64 / layers=5 / pool=mean  ← deeper f + mean
#   modes=32 / layers=4 / pool=sum   ← control: does m=64 still beat m=32 at l=4?
#
# Wall-clock at n_iter=1500 / 102400 / leaf=64: ~2.5h per cell on a clean GPU
# (extrapolating from R9 iter1500). Julia is currently using ~10.5 GiB per GPU
# (out of 97 GiB), so we may be ~1.5-2x slower; budget 4-5h wall.
set -euo pipefail
source venv/bin/activate

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_fno_round10_stacked_winner_${STAMP}}"
GPUS="${GPUS:-0,1,2,3}"
N_ITER="${N_ITER:-1500}"
TRAIN_DOCS=102400
LEAF=64

BUNDLE="outputs/_bundles/markov_hazard_panels_train102400/paper_hazard_panel_v1_t128/seed_0/base_bundle.json"
if [ ! -f "$BUNDLE" ]; then echo "ERROR: bundle missing: $BUNDLE"; exit 1; fi

IFS=',' read -r -a GPU_LIST <<<"$GPUS"
N_GPUS="${#GPU_LIST[@]}"
mkdir -p "$OUT_ROOT/lanes"

# Cell spec: label|n_modes|n_layers|pool
CELLS=(
  "m64__l4__psum|64|4|sum"     # headline: R8 winner + R9 compute
  "m64__l4__pmean|64|4|mean"   # mean pooling at l=4 (mean won at l=3)
  "m64__l5__psum|64|5|sum"     # deeper f
  "m64__l5__pmean|64|5|mean"   # deeper f + mean
  "m32__l4__psum|32|4|sum"     # control: m=64 vs m=32 at l=4
)

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
        "n_iter": prov.get("n_iter"),
        "theta_count_raw_mae": test.get("theta_count_raw_mae"),
        "theta_first_regime_acc": test.get("theta_first_regime_accuracy"),
        "theta_last_regime_acc": test.get("theta_last_regime_accuracy"),
        "leaf_count_proxy": test.get("sufficiency_leaf_rep_count_proxy"),
        "merge_count_proxy": test.get("sufficiency_merge_rep_count_proxy"),
    })
if rows:
    csv_path = os.path.join(root, "round10_stacked_winner_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")
PY

echo ">>> Round 10 complete: $OUT_ROOT"
