#!/usr/bin/env bash
# Hybrid NASS/NASSS + local-laws ablation for the Markov optimize-to-zero lane.
# This keeps the theorem-facing learned_local_laws trainer and adds an
# opt-in package-style auxiliary loss through --local-law-package-weight.
set -euo pipefail
source venv/bin/activate

BUNDLE=outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json
OUT_ROOT="${OUT_ROOT:-outputs/optimize_to_zero_laws_hybrid_grid_t128}"
GPU="${1:-2}"

mkdir -p "$OUT_ROOT"

for method in nass nasss; do
  for package_weight in 0.1 0.5 1.0; do
    for encoding in regime_one_hot one_hot_token_ids normalized_token_ids; do
      for leaf in 1 2 4; do
        if [ "$encoding" != "regime_one_hot" ] && [ "$leaf" = "1" ]; then
          continue
        fi
        out="$OUT_ROOT/${method}/w_${package_weight}/${encoding}/leaf_${leaf}"
        if [ -f "$out/summary.json" ]; then
          echo "skip $out"
          continue
        fi
        echo "==> method=$method weight=$package_weight encoding=$encoding leaf=$leaf"
        CUDA_VISIBLE_DEVICES="$GPU" \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
        ctreepo sim run contextual-sbijax \
          --data-source markov \
          --load-data-bundle "$BUNDLE" \
          --sbijax-trainer learned_local_laws \
          --sbijax-method "$method" \
          --sbijax-package-theta markov_exact_sketch \
          --sbijax-input-encoding "$encoding" \
          --train-docs 1024 --val-docs 256 --test-docs 256 \
          --fragment-len "$leaf" \
          --context-samples-per-doc 1 \
          --response-signature-contexts 16 --response-signature-slices 8 \
          --embedding-dim 32 --state-dim 25 --hidden-dim 128 \
          --learning-rate 0.0003 --lr-schedule cosine \
          --n-iter 2000 --batch-size 128 \
          --local-law-weight 1.0 \
          --local-law-leaf-weight 1.0 \
          --local-law-merge-weight 1.0 \
          --local-law-idempotence-weight 1.0 \
          --local-law-contextual-weight 1.0 \
          --local-law-package-weight "$package_weight" \
          --seed 0 \
          --output-root "$out" 2>&1 | tail -2
      done
    done
  done
done

python3 - <<PY
import json, os

rows = []
root = "$OUT_ROOT"
for method in ("nass", "nasss"):
    for weight in ("0.1", "0.5", "1.0"):
        for enc in ("regime_one_hot", "one_hot_token_ids", "normalized_token_ids"):
            for leaf in (1, 2, 4):
                p = os.path.join(root, method, f"w_{weight}", enc, f"leaf_{leaf}", "summary.json")
                if not os.path.exists(p):
                    continue
                d = json.load(open(p))
                t = d["diagnostics"]["test"]
                h = d["history"][-1] if d["history"] else {}
                prov = d.get("provenance", {})
                rows.append({
                    "method": method,
                    "local_law_package_weight": float(weight),
                    "encoding": enc,
                    "leaf": leaf,
                    "test_contextual_mae": t.get("contextual_mae"),
                    "test_corr": t.get("pred_truth_corr"),
                    "theta_mae": t.get("theta_mae"),
                    "theta_first_regime_accuracy": t.get("theta_first_regime_accuracy"),
                    "theta_last_regime_accuracy": t.get("theta_last_regime_accuracy"),
                    "eps_leaf": t.get("eps_leaf"),
                    "eps_merge": t.get("eps_merge"),
                    "eps_idemp": t.get("eps_idemp"),
                    "val_l1_leaf_mse": h.get("val_l1_leaf_mse"),
                    "val_l2_merge_mse": h.get("val_l2_merge_mse"),
                    "val_l3_idempotence_mse": h.get("val_l3_idempotence_mse"),
                    "val_contextual_mse": h.get("val_contextual_mse"),
                    "val_package_loss": h.get("val_package_loss"),
                    "package_aux_active": prov.get("local_law_package_aux_active"),
                })
out = os.path.join(root, "summary.json")
json.dump(rows, open(out, "w"), indent=2)
print(f"wrote {out} with {len(rows)} rows")
PY
