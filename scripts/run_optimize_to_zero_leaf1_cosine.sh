#!/usr/bin/env bash
# leaf=1 with cosine LR schedule + best-params, all hard encodings.
# Goal: drive every leaf=1 cell to numerical zero.
set -euo pipefail
source venv/bin/activate

BUNDLE=outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json
OUT_ROOT=outputs/optimize_to_zero_leaf1_cosine
mkdir -p "$OUT_ROOT"

GPU="${1:-3}"

ENCODINGS=(regime_one_hot regime_ids one_hot_token_ids normalized_token_ids)

for enc in "${ENCODINGS[@]}"; do
  out="$OUT_ROOT/${enc}"
  if [ -f "$out/summary.json" ]; then
    echo "skip $out"
    continue
  fi
  echo "==> $enc cosine n=5000"
  CUDA_VISIBLE_DEVICES="$GPU" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
  ctreepo sim run contextual-sbijax \
    --data-source markov \
    --load-data-bundle "$BUNDLE" \
    --sbijax-trainer learned_local_laws \
    --sbijax-method nasss \
    --sbijax-package-theta markov_exact_sketch \
    --sbijax-input-encoding "$enc" \
    --train-docs 1024 --val-docs 256 --test-docs 256 \
    --fragment-len 1 \
    --context-samples-per-doc 1 \
    --response-signature-contexts 16 --response-signature-slices 8 \
    --embedding-dim 32 --state-dim 25 --hidden-dim 128 \
    --learning-rate 0.0003 --lr-schedule cosine --n-iter 5000 --batch-size 128 \
    --local-law-weight 1.0 \
    --local-law-leaf-weight 1.0 \
    --local-law-merge-weight 1.0 \
    --local-law-idempotence-weight 1.0 \
    --local-law-contextual-weight 1.0 \
    --seed 0 \
    --output-root "$out" 2>&1 | tail -2
done

python3 - <<PY
import json, os
rows = []
root = "$OUT_ROOT"
for enc in ("regime_one_hot","regime_ids","one_hot_token_ids","normalized_token_ids"):
    p = os.path.join(root, enc, "summary.json")
    if not os.path.exists(p): continue
    d = json.load(open(p))
    t = d["diagnostics"]["test"]
    final_meta = next((r for r in d["history"] if r.get("iteration") == -1), {})
    rows.append({
        "encoding": enc,
        "test_contextual_mae": t.get("contextual_mae"),
        "test_corr": t.get("pred_truth_corr"),
        "theta_mae": t.get("theta_mae"),
        "first_acc": t.get("theta_first_regime_accuracy"),
        "last_acc": t.get("theta_last_regime_accuracy"),
        "best_iteration": final_meta.get("best_iteration"),
        "best_val_law_score": final_meta.get("best_val_law_score"),
    })
out = os.path.join(root, "summary.json")
json.dump(rows, open(out, "w"), indent=2)
print(f"wrote {out} with {len(rows)} rows")
PY
