#!/usr/bin/env bash
# Round 12: t=2048 longer-DGP screening sweep.
#
# Tests whether the recovery story from the t=128 rounds (R5..R11)
# generalizes to longer documents with deeper merge trees. At t=2048 a
# leaf=128 fragmentation produces 16 leaves per doc → 4-level merge
# tree (vs 1 merge at t=128 / leaf=64). leaf=256 is a shallower control.
#
# Cells (4):
#   rts__leaf128           : architectural ceiling at t=2048
#   r7base__leaf128        : R7 baseline (m32/l3/sum + mlp+linear+nass_jsd)
#   r10win__leaf128        : R10 winner (m64/l5/mean + mlp+linear+nass_jsd)
#   r10win__leaf256        : R10 winner with shallower merge tree
#
# Wall: at batch=16 (vs the R10 batch=128), per-step cost is roughly
#       1.3× of the t=128 / leaf=64 / batch=128 baseline (3.6 sec/step
#       baseline → ~4.7 sec/step here). iter=400 lands at ~31 min per
#       cell. 4 cells / 2 GPUs ≈ 1h wall. Total doc-passes per cell =
#       16 * 400 = 6400 — meaningful screen, ~25% of R5 (200*128=25600).
#       If results look good, follow-up sweep at batch=32 / iter=800.
#
# Bundle: existing 10240-doc t=2048 bundle. A 102400-doc t=2048 bundle
# can be generated in the background for a follow-up.
set -euo pipefail
source venv/bin/activate

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_fno_round12_t2048_screen_${STAMP}}"
GPUS="${GPUS:-2,3}"
N_ITER="${N_ITER:-400}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TRAIN_DOCS=10240

BUNDLE="outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t2048/seed_0/base_bundle.json"
if [ ! -f "$BUNDLE" ]; then echo "ERROR: bundle missing: $BUNDLE"; exit 1; fi

IFS=',' read -r -a GPU_LIST <<<"$GPUS"
N_GPUS="${#GPU_LIST[@]}"
mkdir -p "$OUT_ROOT/lanes"

# Cell spec: label|family|leaf|n_modes|n_layers|pool
CELLS=(
  "rts__leaf128|regime_transition_sum|128|32|3|sum"
  "r7base__leaf128|jax_fno|128|32|3|sum"
  "r10win__leaf128|jax_fno|128|64|5|mean"
  "r10win__leaf256|jax_fno|256|64|5|mean"
)

echo "Total cells: ${#CELLS[@]}  GPUs: $GPUS  N_iter: $N_ITER  batch=$BATCH_SIZE  train=$TRAIN_DOCS  Output: $OUT_ROOT"

run_cell() {
  local label="$1" fam="$2" leaf="$3" nm="$4" nl="$5" pool="$6" gpu="$7"
  local out="$OUT_ROOT/$label"
  if [ -f "$out/summary.json" ]; then echo "[gpu$gpu] skip $label"; return 0; fi

  # rts uses analytic decoder + sketch supervision; flexible families use
  # count_only + fully_learned with the merged-rep linear decoder head.
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
        "n_train_docs": prov.get("n_train_docs") or prov.get("train_docs"),
        "theta_count_raw_mae": test.get("theta_count_raw_mae"),
        "theta_first_regime_acc": test.get("theta_first_regime_accuracy"),
        "theta_last_regime_acc": test.get("theta_last_regime_accuracy"),
        "leaf_count_proxy": test.get("sufficiency_leaf_rep_count_proxy"),
        "merge_count_proxy": test.get("sufficiency_merge_rep_count_proxy"),
    })
if rows:
    csv_path = os.path.join(root, "round12_t2048_screen_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")
PY

echo ">>> Round 12 complete: $OUT_ROOT"
