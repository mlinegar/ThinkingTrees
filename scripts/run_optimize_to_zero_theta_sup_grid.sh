#!/usr/bin/env bash
# theta_supervised leaf grid for the optimize_to_zero acceptance criteria.
# Runs all 14 cells: leaves {1,2,4,8,16,32,64} x encodings {markov_exact_sketch, regime_one_hot}.
set -euo pipefail
source venv/bin/activate

BUNDLE=outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json
OUT_ROOT=outputs/optimize_to_zero_theta_sup_grid_t128
mkdir -p "$OUT_ROOT"

GPU="${1:-2}"

for encoding in markov_exact_sketch regime_one_hot; do
  for leaf in 1 2 4 8 16 32 64; do
    out="$OUT_ROOT/${encoding}/leaf_${leaf}"
    if [ -f "$out/summary.json" ]; then
      echo "skip $out (already exists)"
      continue
    fi
    echo "==> $encoding leaf=$leaf"
    CUDA_VISIBLE_DEVICES="$GPU" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
    ctreepo sim run contextual-sbijax \
      --data-source markov \
      --load-data-bundle "$BUNDLE" \
      --sbijax-trainer theta_supervised \
      --sbijax-method nasss \
      --sbijax-package-theta markov_exact_sketch \
      --sbijax-input-encoding "$encoding" \
      --train-docs 1024 --val-docs 256 --test-docs 256 \
      --fragment-len "$leaf" \
      --context-samples-per-doc 1 \
      --response-signature-contexts 16 --response-signature-slices 8 \
      --embedding-dim 32 --state-dim 25 --hidden-dim 128 \
      --learning-rate 0.0003 --n-iter 1000 --batch-size 128 \
      --seed 0 \
      --output-root "$out" 2>&1 | tail -2
  done
done

# Aggregate.
python3 - <<PY
import json, os
rows = []
root = "$OUT_ROOT"
for enc in ("markov_exact_sketch", "regime_one_hot"):
    for leaf in (1, 2, 4, 8, 16, 32, 64):
        p = os.path.join(root, enc, f"leaf_{leaf}", "summary.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        t = d["diagnostics"]["test"]
        rows.append({
            "encoding": enc,
            "leaf": leaf,
            "test_contextual_mae": t.get("contextual_mae"),
            "test_corr": t.get("pred_truth_corr"),
            "test_pred_std": t.get("pred_std"),
            "test_truth_std": t.get("truth_std"),
            "theta_mae": t.get("theta_mae"),
            "theta_first_regime_accuracy": t.get("theta_first_regime_accuracy"),
            "theta_last_regime_accuracy": t.get("theta_last_regime_accuracy"),
            "exact_sketch_oracle_mae": d["diagnostics"]["markov_exact_sketch_oracle"]["test"]["contextual_mae"],
            "root_witness_mae": d["diagnostics"]["exact_root_witness"]["test"]["root_mae"],
        })
out = os.path.join(root, "leaf_grid_summary.json")
json.dump(rows, open(out, "w"), indent=2)
print(f"wrote {out} with {len(rows)} rows")
PY
