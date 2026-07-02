#!/usr/bin/env bash
# Round 13: t=2048 headline sweep at 102400 docs.
#
# R12 smoke (10240 docs / batch=16 / iter=400) showed:
#   - rts ceiling at t=2048/leaf=128 = 0.000688 (≈ t=128's 0.0005)
#   - flexible learners far from ceiling (0.15-0.49) — under-trained
#
# R13 runs at the matched-to-R5 schedule (batch=128 / iter=200) on the
# fresh 102400-doc t=2048 bundle. This is the apples-to-apples comparison
# with the t=128 R5 headline cells.
#
# Cells (5):
#   rts__leaf128        : architectural ceiling at 102400/t=2048
#   r7base__leaf128     : R7 baseline at t=2048
#   r10win__leaf128     : R10 winner at t=2048 (does compute rescue it?)
#   r7base__leaf64      : R7 baseline at deeper tree (32 leaves)
#   r10win__leaf64      : R10 winner at deeper tree
#
# Wall: at batch=128 / t=2048 / leaf=128, per-step cost is ~16× t=128
# baseline → 200 iter ≈ 2.7h per cell. 5 cells / 3 GPUs ≈ 5.4h wall.
# Julia co-tenant uses ~10.6 GiB; we set MEM_FRACTION=0.50.
set -euo pipefail
source venv/bin/activate

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_fno_round13_t2048_headline_${STAMP}}"
GPUS="${GPUS:-0,2,3}"
N_ITER="${N_ITER:-200}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TRAIN_DOCS=102400

BUNDLE="outputs/_bundles/markov_hazard_panels_train102400_t2048/paper_hazard_panel_v1_t2048/seed_0/base_bundle.json"
if [ ! -f "$BUNDLE" ]; then echo "ERROR: bundle missing: $BUNDLE"; exit 1; fi

IFS=',' read -r -a GPU_LIST <<<"$GPUS"
N_GPUS="${#GPU_LIST[@]}"
mkdir -p "$OUT_ROOT/lanes"

# Cell spec: label|family|leaf|n_modes|n_layers|pool
CELLS=(
  "rts__leaf128|regime_transition_sum|128|32|3|sum"
  "r7base__leaf128|jax_fno|128|32|3|sum"
  "r10win__leaf128|jax_fno|128|64|5|mean"
  "r7base__leaf64|jax_fno|64|32|3|sum"
  "r10win__leaf64|jax_fno|64|64|5|mean"
)

echo "Total cells: ${#CELLS[@]}  GPUs: $GPUS  N_iter: $N_ITER  batch=$BATCH_SIZE  train=$TRAIN_DOCS  Output: $OUT_ROOT"

run_cell() {
  local label="$1" fam="$2" leaf="$3" nm="$4" nl="$5" pool="$6" gpu="$7"
  local out="$OUT_ROOT/$label"
  if [ -f "$out/summary.json" ]; then echo "[gpu$gpu] skip $label"; return 0; fi

  local family_args=()
  if [ "$fam" = "regime_transition_sum" ]; then
    family_args=(
      --local-law-summary-family regime_transition_sum
      --law-architecture analytic --c2-merge-target theta
      --merge-family mlp --decoder-head mlp
      --local-law-merge-loss mse
      --local-law-rep-dim 0
    )
  else
    family_args=(
      --local-law-summary-family jax_fno
      --local-law-summary-fno-n-modes "$nm"
      --local-law-summary-fno-n-layers "$nl"
      --local-law-summary-fno-pooling-mode "$pool"
      --law-architecture fully_learned --c2-merge-target theta
      --merge-family mlp --merge-fno-n-modes 16 --merge-fno-n-layers 2 --merge-fno-hidden-channels 32
      --decoder-head linear
      --local-law-merge-loss nass_jsd --merge-nasss-n-slices 16
      --local-law-count-only --local-law-rep-dim 0
    )
  fi

  echo "[gpu$gpu] start $label (family=$fam leaf=$leaf modes=$nm layers=$nl pool=$pool)"
  CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.50 \
  ./venv/bin/ctreepo sim run contextual-sbijax \
    --data-source markov --load-data-bundle "$BUNDLE" \
    --sbijax-trainer learned_local_laws --sbijax-method nasss --sbijax-package-theta markov_exact_sketch \
    --sbijax-input-encoding regime_one_hot \
    "${family_args[@]}" \
    --learned-merge-hidden-dim 128 --learned-decoder-hidden-dim 128 \
    --train-docs $TRAIN_DOCS --val-docs 256 --test-docs 256 \
    --fragment-len "$leaf" --context-samples-per-doc 1 \
    --response-signature-contexts 16 --response-signature-slices 8 \
    --embedding-dim 64 --state-dim 25 --hidden-dim 128 \
    --learning-rate 0.0003 --lr-schedule cosine \
    --n-iter "$N_ITER" --batch-size "$BATCH_SIZE" \
    --local-law-weight 1.0 --local-law-leaf-weight 1.0 --local-law-merge-weight 1.0 \
    --local-law-idempotence-weight 0.0 --local-law-contextual-weight 1.0 --local-law-package-weight 0.0 \
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
      IFS='|' read -r label fam leaf nm nl pool <<<"$cell_spec"
      run_cell "$label" "$fam" "$leaf" "$nm" "$nl" "$pool" "$gpu"
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
        "summary_family": prov.get("local_law_summary_family"),
        "fragment_len": prov.get("fragment_len"),
        "summary_fno_n_modes": prov.get("local_law_summary_fno_n_modes"),
        "summary_fno_n_layers": prov.get("local_law_summary_fno_n_layers"),
        "summary_fno_pool": prov.get("local_law_summary_fno_pooling_mode"),
        "n_iter": prov.get("n_iter"),
        "batch_size": prov.get("batch_size"),
        "n_train_docs": prov.get("n_train_docs") or prov.get("train_docs"),
        "theta_count_raw_mae": test.get("theta_count_raw_mae"),
        "theta_first_regime_acc": test.get("theta_first_regime_accuracy"),
        "theta_last_regime_acc": test.get("theta_last_regime_accuracy"),
        "leaf_count_proxy": test.get("sufficiency_leaf_rep_count_proxy"),
        "merge_count_proxy": test.get("sufficiency_merge_rep_count_proxy"),
    })
if rows:
    csv_path = os.path.join(root, "round13_t2048_headline_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")
PY

echo ">>> Round 13 complete: $OUT_ROOT"
