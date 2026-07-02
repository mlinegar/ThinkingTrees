#!/usr/bin/env bash
# Round 9: compute scaling at the headline cell.
#
# Every R5/R6/R7 number is at n_iter=200. The 0.022 floor at the R7
# winner could be a compute floor or an architecture floor; we don't
# know which. R9 sweeps n_iter at the R7 winner config to settle it.
#
# Hold (R7 winner):
#   train_docs=102400, leaf=64, regime_one_hot, count_only,
#   fully_learned, mlp + nass_jsd + linear + rep_dim=0, leaf-FNO
#   modes=32 / layers=3 / pool=sum.
#
# Vary:
#   n_iter ∈ {500, 1500, 4000}
#
# Wall-clock estimates at 102400/leaf=64 (extrapolated from R5):
#   200 iter ≈ 30 min  (have this from R7)
#   500 iter ≈ 75 min
#  1500 iter ≈ 3.75h
#  4000 iter ≈ 10h
#
# Default GPU plan: GPU 2 = n_iter=4000 (long-running),
#                   GPU 3 = n_iter=500 then n_iter=1500 sequentially
#                           (~75 min + ~3.75h ≈ 5h).
set -euo pipefail
source venv/bin/activate

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_fno_round9_compute_scaling_${STAMP}}"
GPUS="${GPUS:-2,3}"
TRAIN_DOCS=102400
LEAF=64

BUNDLE="outputs/_bundles/markov_hazard_panels_train102400/paper_hazard_panel_v1_t128/seed_0/base_bundle.json"
if [ ! -f "$BUNDLE" ]; then echo "ERROR: bundle missing: $BUNDLE"; exit 1; fi

IFS=',' read -r -a GPU_LIST <<<"$GPUS"
N_GPUS="${#GPU_LIST[@]}"
mkdir -p "$OUT_ROOT/lanes"

# Cell list — fixed per-cell n_iter; cells assigned to GPUs deterministically
# below so the longest cell sits on a dedicated GPU.
CELLS=(
  "iter4000|4000"   # lane 0 (GPU 2): the long pole
  "iter500|500"     # lane 1 (GPU 3): completes first; lane runs iter1500 next
  "iter1500|1500"   # lane 1 (GPU 3): runs after iter500
)

echo "Total cells: ${#CELLS[@]}  GPUs: $GPUS  Output: $OUT_ROOT"

run_cell() {
  local label="$1" niter="$2" gpu="$3"
  local out="$OUT_ROOT/$label"
  if [ -f "$out/summary.json" ]; then echo "[gpu$gpu] skip $label"; return 0; fi

  echo "[gpu$gpu] start $label (n_iter=$niter)"
  CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 \
  ./venv/bin/ctreepo sim run contextual-sbijax \
    --data-source markov --load-data-bundle "$BUNDLE" \
    --sbijax-trainer learned_local_laws --sbijax-method nasss --sbijax-package-theta markov_exact_sketch \
    --sbijax-input-encoding regime_one_hot --local-law-summary-family jax_fno \
    --local-law-summary-fno-n-modes 32 --local-law-summary-fno-n-layers 3 --local-law-summary-fno-pooling-mode sum \
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
    --n-iter "$niter" --batch-size 128 \
    --local-law-weight 1.0 --local-law-leaf-weight 1.0 --local-law-merge-weight 1.0 \
    --local-law-idempotence-weight 0.0 --local-law-contextual-weight 1.0 --local-law-package-weight 0.0 \
    --local-law-count-only --local-law-rep-dim 0 \
    --seed 0 --output-root "$out" \
    >"$OUT_ROOT/lanes/${label}.log" 2>&1
  echo "[gpu$gpu] done $label"
}

# Hand-assign cells to lanes so the long pole gets a dedicated GPU.
# Lane 0 (GPUs[0]): iter4000.
# Lane 1 (GPUs[1]): iter500 then iter1500.
GPU0="${GPU_LIST[0]}"
GPU1="${GPU_LIST[1]:-${GPU_LIST[0]}}"

(
  IFS='|' read -r label niter <<<"${CELLS[0]}"
  run_cell "$label" "$niter" "$GPU0"
) >"$OUT_ROOT/lanes/lane_gpu${GPU0}.log" 2>&1 &
PID0=$!

(
  IFS='|' read -r label niter <<<"${CELLS[1]}"
  run_cell "$label" "$niter" "$GPU1"
  IFS='|' read -r label niter <<<"${CELLS[2]}"
  run_cell "$label" "$niter" "$GPU1"
) >"$OUT_ROOT/lanes/lane_gpu${GPU1}.log" 2>&1 &
PID1=$!

wait "$PID0" || true
wait "$PID1" || true

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
        "n_iter": prov.get("n_iter"),
        "theta_count_raw_mae": test.get("theta_count_raw_mae"),
        "theta_first_regime_acc": test.get("theta_first_regime_accuracy"),
        "theta_last_regime_acc": test.get("theta_last_regime_accuracy"),
        "leaf_count_proxy": test.get("sufficiency_leaf_rep_count_proxy"),
        "merge_count_proxy": test.get("sufficiency_merge_rep_count_proxy"),
    })
if rows:
    csv_path = os.path.join(root, "round9_compute_scaling_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")
PY

echo ">>> Round 9 complete: $OUT_ROOT"
