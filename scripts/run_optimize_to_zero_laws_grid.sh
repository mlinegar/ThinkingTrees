#!/usr/bin/env bash
# learned_local_laws leaf grid (parallel to theta_supervised grid).
# 7 leaves x 2 encodings = 14 cells. All-laws supervision (c1=c2=c3=1).
set -euo pipefail
source venv/bin/activate

BUNDLE=outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json
OUT_ROOT=outputs/optimize_to_zero_laws_grid_t128
mkdir -p "$OUT_ROOT"

GPU="${1:-2}"

for encoding in markov_exact_sketch regime_one_hot; do
  for leaf in 1 2 4 8 16 32 64; do
    out="$OUT_ROOT/${encoding}/leaf_${leaf}"
    if [ -f "$out/summary.json" ]; then
      echo "skip $out"
      continue
    fi
    echo "==> $encoding leaf=$leaf"
    CUDA_VISIBLE_DEVICES="$GPU" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
    ctreepo sim run contextual-sbijax \
      --data-source markov \
      --load-data-bundle "$BUNDLE" \
      --sbijax-trainer learned_local_laws \
      --sbijax-method nasss \
      --sbijax-package-theta markov_exact_sketch \
      --sbijax-input-encoding "$encoding" \
      --train-docs 1024 --val-docs 256 --test-docs 256 \
      --fragment-len "$leaf" \
      --context-samples-per-doc 1 \
      --response-signature-contexts 16 --response-signature-slices 8 \
      --embedding-dim 32 --state-dim 25 --hidden-dim 128 \
      --learning-rate 0.0003 --n-iter 1000 --batch-size 128 \
      --local-law-weight 1.0 \
      --local-law-leaf-weight 1.0 \
      --local-law-merge-weight 1.0 \
      --local-law-idempotence-weight 1.0 \
      --local-law-contextual-weight 1.0 \
      --seed 0 \
      --output-root "$out" 2>&1 | tail -2
  done
done

python3 - <<PY
import json, os
rows = []
root = "$OUT_ROOT"
for enc in ("markov_exact_sketch", "regime_one_hot"):
    for leaf in (1, 2, 4, 8, 16, 32, 64):
        p = os.path.join(root, enc, f"leaf_{leaf}", "summary.json")
        if not os.path.exists(p): continue
        d = json.load(open(p))
        t = d["diagnostics"]["test"]
        h = d["history"][-1] if d["history"] else {}
        rows.append({
            "encoding": enc,
            "leaf": leaf,
            "test_contextual_mae": t.get("contextual_mae"),
            "test_corr": t.get("pred_truth_corr"),
            "theta_mae": t.get("theta_mae"),
            "theta_first_regime_accuracy": t.get("theta_first_regime_accuracy"),
            "theta_last_regime_accuracy": t.get("theta_last_regime_accuracy"),
            "val_l1_leaf_mse": h.get("val_l1_leaf_mse"),
            "val_l2_merge_mse": h.get("val_l2_merge_mse"),
            "val_l3_idempotence_mse": h.get("val_l3_idempotence_mse"),
            "val_contextual_mse": h.get("val_contextual_mse"),
        })
out = os.path.join(root, "leaf_grid_summary.json")
json.dump(rows, open(out, "w"), indent=2)
print(f"wrote {out} with {len(rows)} rows")
PY
