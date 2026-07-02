#!/usr/bin/env bash
# Capacity sweep: does wider encoder fix leaf>=4 degradation in laws-aligned learning?
# Tests hidden_dim={128,512,1024} x state_dim={25,64} on the worst cells.
set -euo pipefail
source venv/bin/activate

BUNDLE=outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json
OUT_ROOT=outputs/optimize_to_zero_capacity_sweep
mkdir -p "$OUT_ROOT"

GPU="${1:-2}"

# Worst cells from prior runs:
#   one_hot_token_ids leaf=4: ctx_mae=7.0e-3, theta_mae=1.3e-2, last_acc=99.2%
#   normalized_token_ids leaf=4: ctx_mae=1.2e-2, last_acc=93.8%
# Also include leaf=2 (already perfect on one_hot) as a control: should stay perfect.

for cfg in \
  "h128_s25:128:25" \
  "h512_s25:512:25" \
  "h512_s64:512:64" \
  "h1024_s64:1024:64" \
; do
  IFS=':' read -r tag hd sd <<< "$cfg"
  for encoding in one_hot_token_ids normalized_token_ids; do
    for leaf in 2 4; do
      out="$OUT_ROOT/${tag}/${encoding}/leaf_${leaf}"
      if [ -f "$out/summary.json" ]; then
        echo "skip $out"
        continue
      fi
      echo "==> $tag $encoding leaf=$leaf (h=$hd s=$sd)"
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
        --embedding-dim 32 --state-dim "$sd" --hidden-dim "$hd" \
        --learning-rate 0.0003 --n-iter 2000 --batch-size 128 \
        --local-law-weight 1.0 \
        --local-law-leaf-weight 1.0 \
        --local-law-merge-weight 1.0 \
        --local-law-idempotence-weight 1.0 \
        --local-law-contextual-weight 1.0 \
        --seed 0 \
        --output-root "$out" 2>&1 | tail -2
    done
  done
done

python3 - <<PY
import json, os
rows = []
root = "$OUT_ROOT"
for tag in ("h128_s25","h512_s25","h512_s64","h1024_s64"):
    for enc in ("one_hot_token_ids", "normalized_token_ids"):
        for leaf in (2, 4):
            p = os.path.join(root, tag, enc, f"leaf_{leaf}", "summary.json")
            if not os.path.exists(p): continue
            d = json.load(open(p))
            t = d["diagnostics"]["test"]
            h = d["history"][-1] if d["history"] else {}
            rows.append({
                "capacity": tag,
                "encoding": enc,
                "leaf": leaf,
                "test_contextual_mae": t.get("contextual_mae"),
                "test_corr": t.get("pred_truth_corr"),
                "theta_mae": t.get("theta_mae"),
                "first_acc": t.get("theta_first_regime_accuracy"),
                "last_acc": t.get("theta_last_regime_accuracy"),
                "val_l1_leaf_mse": h.get("val_l1_leaf_mse"),
                "val_l2_merge_mse": h.get("val_l2_merge_mse"),
            })
out = os.path.join(root, "summary.json")
json.dump(rows, open(out, "w"), indent=2)
print(f"wrote {out} with {len(rows)} rows")
PY
