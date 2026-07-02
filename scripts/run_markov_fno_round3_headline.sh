#!/usr/bin/env bash
# Round 3 headline cells: NASS/NASSS + laws hybrid in PyTorch FNO.
#
# Tests whether combining the local-law objective with the contextual
# sufficient-summary objective (sbijax NASS/NASSS analogue) recovers a
# canonical Markov sketch in the multi-leaf FNO setting where laws-only
# (Round 1 best 1.94 at leaf=32) and contextual-only (Round 1 best 1.15 at
# leaf=16) each plateau separately.
#
# Frame: per docs/markov_sim_status.md "Original JAX Purpose" section,
# generic NASS/NASSS objectives are not enough on this Markov task; laws
# add the structural tie-breaker. Tests whether that framework transfers
# to the PyTorch FNO bridge.
set -euo pipefail
source venv/bin/activate

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_fno_round3_nasss_plus_laws_${STAMP}}"
BUNDLE="${BUNDLE:-outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json}"
GPU="${GPU:-1}"
EPOCHS="${EPOCHS:-48}"
TRAIN_DOCS="${TRAIN_DOCS:-10240}"
EVAL_DOCS="${EVAL_DOCS:-1024}"
PY="${PY:-./venv/bin/python}"
SEED="${SEED:-0}"

# Cells: (label, training_objective, enable_contextual, dep_objective, leaf_tokens)
# Headline pair at the Round-1 anchor (leaf=32):
CELLS=(
  "laws_only__leaf32                       markov_local_laws_fno   0   none      32"
  "contextual_only_infonce__leaf32          contextual_sufficiency  0   infonce   32"
  "contextual_only_dcorr__leaf32            contextual_sufficiency  0   dcorr     32"
  "laws_plus_contextual_infonce__leaf32     markov_local_laws_fno   1   infonce   32"
  "laws_plus_contextual_dcorr__leaf32       markov_local_laws_fno   1   dcorr     32"
)

mkdir -p "$OUT_ROOT/logs"

run_cell() {
  local label="$1" obj="$2" enable_ctx="$3" dep="$4" leaf="$5"
  local out="$OUT_ROOT/$label"
  if [ -f "$out/summary.json" ]; then
    echo "skip $label (already done)"
    return 0
  fi
  echo "==> $label  obj=$obj enable_ctx=$enable_ctx dep=$dep leaf=$leaf"

  local extra_args=()
  if [ "$enable_ctx" = "1" ]; then
    extra_args+=(--enable-contextual-sufficiency)
  fi
  if [ "$obj" = "contextual_sufficiency" ] || [ "$enable_ctx" = "1" ]; then
    extra_args+=(
      --context-samples-per-doc 1
      --contextual-loss-weight 1.0
      --contextual-dependence-objective "$dep"
      --response-signature-contexts 16
      --response-signature-slices 8
      --infomax-loss-weight 1.0
    )
  fi
  if [ "$obj" = "markov_local_laws_fno" ]; then
    extra_args+=(
      --markov-law-leaf-weight 1.0
      --markov-law-merge-weight 1.0
      --markov-law-idempotence-weight 0.1
      --markov-law-readout flatten
    )
  fi

  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/probe_clean_unified_no.py \
    --load-data-bundle "$BUNDLE" \
    --leaf-tokens "$leaf" \
    --train-docs "$TRAIN_DOCS" \
    --eval-docs "$EVAL_DOCS" \
    --epochs "$EPOCHS" \
    --batch-size 64 \
    --channels 128 \
    --g-n-modes 16 \
    --g-n-layers 2 \
    --scorer-n-modes 16 \
    --scorer-n-layers 2 \
    --lr 0.0001 \
    --optimizer adamw \
    --weight-decay 0.01 \
    --lr-schedule cosine \
    --grad-clip 1.0 \
    --leaf-pool sum \
    --diagnostic-baselines none \
    --seed "$SEED" \
    --device cuda \
    --training-objective "$obj" \
    --output-root "$out" \
    "${extra_args[@]}" 2>&1 | tee -a "$OUT_ROOT/logs/${label}.log" | tail -3
}

for cell in "${CELLS[@]}"; do
  read -r label obj enable_ctx dep leaf <<<"$cell"
  run_cell "$label" "$obj" "$enable_ctx" "$dep" "$leaf"
done

echo ">>> Aggregating headline cells"
"$PY" - <<PY
import csv, json, os
root = "$OUT_ROOT"
rows = []
for cell in sorted(os.listdir(root)):
    sj = os.path.join(root, cell, "summary.json")
    if not os.path.exists(sj):
        continue
    d = json.load(open(sj))
    diag_law = ((d.get("markov_local_law_fno_diagnostics") or {}).get("splits") or {}).get("test") or {}
    diag_ctx = (d.get("contextual_sufficiency_diagnostics") or {}).get("test") or {}
    leaf = (diag_law or {}).get("leaf") or {}
    cd = leaf.get("count_diagnostics") or {}
    rows.append({
        "cell": cell,
        "training_objective": d.get("args", {}).get("training_objective"),
        "enable_contextual_sufficiency": d.get("args", {}).get("enable_contextual_sufficiency"),
        "contextual_dependence_objective": d.get("args", {}).get("contextual_dependence_objective"),
        "test_root_mae": d.get("test_root_mae"),
        "best_val_root_mae": d.get("best_val_root_mae"),
        "leaf_theta_mae": leaf.get("theta_mae"),
        "leaf_first_acc": leaf.get("theta_first_regime_accuracy"),
        "leaf_last_acc": leaf.get("theta_last_regime_accuracy"),
        "leaf_full_exact": leaf.get("full_witness_exact_rate"),
        "leaf_count_mae": cd.get("root_mae"),
        "ctx_test_mae": diag_ctx.get("contextual_mae"),
    })
if rows:
    csv_path = os.path.join(root, "headline_summary.csv")
    keys = list(rows[0].keys())
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")
PY

echo ">>> Round 3 headline complete: $OUT_ROOT"
