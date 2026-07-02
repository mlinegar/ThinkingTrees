#!/usr/bin/env bash
# CleanUnifiedNO contextual-sufficiency ablation.
#
# This is the cleaner "general f/g only" path: learned leaf adapter, shared
# learned g, and learned f. No exact Markov merge/decoder is installed in the
# model. The exact Markov witness is reported only as a diagnostic/control.
set -euo pipefail
source venv/bin/activate

OUT_ROOT="${OUT_ROOT:-outputs/clean_unified_fg_contextual_ablation_t128}"
GPU="${1:-2}"

DOC_TOKENS="${DOC_TOKENS:-128}"
TRAIN_DOCS="${TRAIN_DOCS:-1024}"
EVAL_DOCS="${EVAL_DOCS:-256}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CHANNELS="${CHANNELS:-64}"
LEAF_TOKENS="${LEAF_TOKENS:-128 32 16}"
DEPENDENCE_OBJECTIVES="${DEPENDENCE_OBJECTIVES:-none regression dcorr infonce}"

mkdir -p "$OUT_ROOT"

IFS=' ' read -r -a leaf_tokens_grid <<< "$LEAF_TOKENS"
IFS=' ' read -r -a dependence_objectives <<< "$DEPENDENCE_OBJECTIVES"

for leaf_tokens in "${leaf_tokens_grid[@]}"; do
  for objective in root contextual_sufficiency; do
    if [ "$objective" = "root" ]; then
      deps=("none")
    else
      deps=("${dependence_objectives[@]}")
    fi
    for dep in "${deps[@]}"; do
      out="$OUT_ROOT/${objective}/dep_${dep}/leaf_tokens_${leaf_tokens}"
      if [ -f "$out/summary.json" ]; then
        echo "skip $out"
        continue
      fi
      echo "==> objective=$objective dependence=$dep leaf_tokens=$leaf_tokens"
      infomax_weight="0.0"
      response_slices="0"
      if [ "$objective" = "contextual_sufficiency" ] && [ "$dep" != "none" ]; then
        infomax_weight="1.0"
        response_slices="8"
      fi
      CUDA_VISIBLE_DEVICES="$GPU" \
      ./venv/bin/python scripts/probe_clean_unified_no.py \
        --doc-tokens "$DOC_TOKENS" \
        --leaf-tokens "$leaf_tokens" \
        --train-docs "$TRAIN_DOCS" \
        --eval-docs "$EVAL_DOCS" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --channels "$CHANNELS" \
        --g-n-modes 32 --g-n-layers 2 \
        --scorer-n-modes 16 --scorer-n-layers 2 \
        --training-objective "$objective" \
        --context-samples-per-doc 1 \
        --contextual-loss-weight 1.0 \
        --infomax-loss-weight "$infomax_weight" \
        --contextual-dependence-objective "$dep" \
        --response-signature-contexts 16 \
        --response-signature-slices "$response_slices" \
        --contextual-response-regressor mean_linear \
        --diagnostic-baselines palette_ridge \
        --seed 0 \
        --gpu 0 \
        --output-root "$out"
    done
  done
done

python3 - <<PY
import json
import os

root = "$OUT_ROOT"
rows = []
for dirpath, _dirnames, filenames in os.walk(root):
    if "summary.json" not in filenames:
        continue
    path = os.path.join(dirpath, "summary.json")
    if path == os.path.join(root, "summary.json"):
        continue
    with open(path) as fh:
        d = json.load(fh)
    args = d.get("args", {})
    learned = d.get("learned_prediction_diagnostics", {}).get("test", {})
    ctx = (
        d.get("contextual_sufficiency_diagnostics", {})
        .get("splits", {})
        .get("test", {})
    )
    ctx_diag = ctx.get("diagnostics", {}) if isinstance(ctx, dict) else {}
    exact = (
        d.get("exact_palette_block_witness", {})
        .get("splits", {})
        .get("test", {})
    )
    exact_diag = exact.get("diagnostics", exact) if isinstance(exact, dict) else {}
    rows.append({
        "path": os.path.relpath(dirpath, root),
        "training_objective": args.get("training_objective"),
        "contextual_dependence_objective": args.get("contextual_dependence_objective"),
        "leaf_tokens": args.get("leaf_tokens"),
        "n_leaves_per_doc": d.get("n_leaves_per_doc"),
        "channels": args.get("channels"),
        "test_root_mae": d.get("test_root_mae"),
        "test_root_corr": learned.get("pred_truth_corr"),
        "test_contextual_mae": ctx_diag.get("root_mae"),
        "test_contextual_corr": ctx_diag.get("pred_truth_corr"),
        "exact_witness_test_root_mae": exact_diag.get("root_mae"),
        "best_val_root_mae": d.get("best_val_root_mae"),
        "best_val_epoch": d.get("best_val_epoch"),
        "n_params_total": d.get("n_params_total"),
        "n_params_g": d.get("n_params_g"),
        "n_params_f": d.get("n_params_f"),
    })
rows.sort(key=lambda row: row["path"])
out = os.path.join(root, "summary.json")
with open(out, "w") as fh:
    json.dump(rows, fh, indent=2)
print(f"wrote {out} with {len(rows)} rows")
PY
