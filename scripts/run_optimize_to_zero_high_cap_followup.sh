#!/usr/bin/env bash
# High-capacity follow-up: re-run cells where the default h512_s64+n=5000
# didn't hit numerical zero. Use h1024 + n=10000 cosine.
set -euo pipefail
source venv/bin/activate

BUNDLE=outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json
OUT_ROOT=outputs/optimize_to_zero_high_cap_followup
mkdir -p "$OUT_ROOT"

GPU="${1:-2}"

# Skip markov_exact_sketch (already zero), skip leaf=1 (already covered).
# Focus on leaves 2,4,8,16,32,64 with all 4 hard encodings.

ENCODINGS=(regime_one_hot one_hot_token_ids regime_ids normalized_token_ids)
LEAVES=(2 4 8 16 32 64)

for enc in "${ENCODINGS[@]}"; do
  for leaf in "${LEAVES[@]}"; do
    out="$OUT_ROOT/${enc}/leaf_${leaf}"
    if [ -f "$out/summary.json" ]; then
      echo "skip $out"
      continue
    fi
    echo "==> $enc leaf=$leaf h=1024 n=10000"
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
      --fragment-len "$leaf" \
      --context-samples-per-doc 1 \
      --response-signature-contexts 16 --response-signature-slices 8 \
      --embedding-dim 32 --state-dim 64 --hidden-dim 1024 \
      --learning-rate 0.0003 --lr-schedule cosine --n-iter 10000 --batch-size 128 \
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
for enc in ("regime_one_hot","one_hot_token_ids","regime_ids","normalized_token_ids"):
    for leaf in (2, 4, 8, 16, 32, 64):
        p = os.path.join(root, enc, f"leaf_{leaf}", "summary.json")
        if not os.path.exists(p): continue
        d = json.load(open(p))
        t = d["diagnostics"]["test"]
        final_meta = next((r for r in d["history"] if r.get("iteration") == -1), {})
        rows.append({
            "encoding": enc,
            "leaf": leaf,
            "test_contextual_mae": t.get("contextual_mae"),
            "test_corr": t.get("pred_truth_corr"),
            "theta_mae": t.get("theta_mae"),
            "first_acc": t.get("theta_first_regime_accuracy"),
            "last_acc": t.get("theta_last_regime_accuracy"),
            "best_iteration": final_meta.get("best_iteration"),
        })
out = os.path.join(root, "summary.json")
json.dump(rows, open(out, "w"), indent=2)
print(f"wrote {out} with {len(rows)} rows")
PY
