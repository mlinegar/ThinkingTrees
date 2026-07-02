#!/usr/bin/env bash
# Markov optimize-to-zero f/g architecture ablation.
#
# This keeps the contextual-sbijax learned_local_laws trainer but removes one
# hard-coded Markov component at a time:
#   analytic        = exact Markov merge + exact Markov decoder
#   learned_merge   = learned g(s_left, s_right), exact decoder
#   learned_decoder = exact merge, learned f(state)
#   fully_learned   = learned g and learned f
#
# Optional --local-law-package-weight values test whether NASS/NASSS-style
# auxiliary objectives help once the local laws remain in the objective.
set -euo pipefail
source venv/bin/activate

BUNDLE="${BUNDLE:-outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json}"
OUT_ROOT="${OUT_ROOT:-outputs/optimize_to_zero_fg_architecture_ablation_t128}"
GPU="${1:-2}"

METHODS="${METHODS:-nasss}"
PACKAGE_WEIGHTS="${PACKAGE_WEIGHTS:-0 0.5}"
ARCHITECTURES="${ARCHITECTURES:-analytic learned_merge learned_decoder fully_learned}"
C2_TARGETS="${C2_TARGETS:-theta self_consistency}"
LEAVES="${LEAVES:-1 2 4}"
ENCODING="${ENCODING:-regime_one_hot}"

mkdir -p "$OUT_ROOT"

IFS=' ' read -r -a methods <<< "$METHODS"
IFS=' ' read -r -a package_weights <<< "$PACKAGE_WEIGHTS"
IFS=' ' read -r -a architectures <<< "$ARCHITECTURES"
IFS=' ' read -r -a c2_targets <<< "$C2_TARGETS"
IFS=' ' read -r -a leaves <<< "$LEAVES"

for method in "${methods[@]}"; do
  for package_weight in "${package_weights[@]}"; do
    for architecture in "${architectures[@]}"; do
      if [[ "$architecture" == "learned_merge" || "$architecture" == "fully_learned" ]]; then
        architecture_c2_targets=("${c2_targets[@]}")
      else
        architecture_c2_targets=("theta")
      fi
      for c2_target in "${architecture_c2_targets[@]}"; do
        for leaf in "${leaves[@]}"; do
          out="$OUT_ROOT/${method}/w_${package_weight}/${architecture}/c2_${c2_target}/leaf_${leaf}"
          if [ -f "$out/summary.json" ]; then
            echo "skip $out"
            continue
          fi
          echo "==> method=$method weight=$package_weight arch=$architecture c2=$c2_target leaf=$leaf"
          CUDA_VISIBLE_DEVICES="$GPU" \
          XLA_PYTHON_CLIENT_PREALLOCATE=false \
          XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
          ctreepo sim run contextual-sbijax \
            --data-source markov \
            --load-data-bundle "$BUNDLE" \
            --sbijax-trainer learned_local_laws \
            --sbijax-method "$method" \
            --sbijax-package-theta markov_exact_sketch \
            --sbijax-input-encoding "$ENCODING" \
            --law-architecture "$architecture" \
            --c2-merge-target "$c2_target" \
            --learned-merge-hidden-dim 128 \
            --learned-decoder-hidden-dim 128 \
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
    test = d.get("diagnostics", {}).get("test", {})
    prov = d.get("provenance", {})
    metric = d.get("metric_summary", {})
    last_history = d.get("history", [{}])[-1] if d.get("history") else {}
    rows.append({
        "path": os.path.relpath(dirpath, root),
        "method": prov.get("method"),
        "input_encoding": d.get("input_encoding"),
        "law_architecture": prov.get("law_architecture"),
        "c2_merge_target": prov.get("c2_merge_target"),
        "local_law_package_weight": prov.get("local_law_package_weight"),
        "local_law_package_aux_active": prov.get("local_law_package_aux_active"),
        "decoder_kind": prov.get("decoder_kind"),
        "merge_network": prov.get("merge_network"),
        "contextual_mae": test.get("contextual_mae"),
        "contextual_raw_mae": test.get("contextual_raw_mae"),
        "theta_mae": test.get("theta_mae"),
        "raw_count_mae": test.get("theta_count_raw_mae"),
        "theta_first_regime_accuracy": test.get("theta_first_regime_accuracy"),
        "theta_last_regime_accuracy": test.get("theta_last_regime_accuracy"),
        "eps_leaf": test.get("eps_leaf"),
        "eps_merge": test.get("eps_merge"),
        "eps_idemp": test.get("eps_idemp"),
        "pred_truth_corr": test.get("pred_truth_corr"),
        "val_l1_leaf_mse": last_history.get("val_l1_leaf_mse"),
        "val_l2_merge_mse": last_history.get("val_l2_merge_mse"),
        "val_l3_idempotence_mse": last_history.get("val_l3_idempotence_mse"),
        "val_contextual_mse": last_history.get("val_contextual_mse"),
        "val_package_loss": last_history.get("val_package_loss"),
        "metric_summary": metric,
    })
rows.sort(key=lambda row: row["path"])
out = os.path.join(root, "summary.json")
with open(out, "w") as fh:
    json.dump(rows, fh, indent=2)
print(f"wrote {out} with {len(rows)} rows")
PY
