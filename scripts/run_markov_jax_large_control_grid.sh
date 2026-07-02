#!/usr/bin/env bash
# Large-doc JAX learned_local_laws controls for the FNO bridge run.
set -euo pipefail

source venv/bin/activate

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_contextual_sbijax_large_controls_${STAMP}}"
BUNDLE="${BUNDLE:-outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json}"
GPU="${GPU:-0}"

ENCODINGS="${ENCODINGS:-markov_exact_sketch regime_one_hot one_hot_token_ids}"
LEAVES="${LEAVES:-1 2 16 64}"
ARCHITECTURES="${ARCHITECTURES:-analytic learned_merge}"

mkdir -p "$OUT_ROOT"

IFS=' ' read -r -a encodings <<< "$ENCODINGS"
IFS=' ' read -r -a leaves <<< "$LEAVES"
IFS=' ' read -r -a architectures <<< "$ARCHITECTURES"

for enc in "${encodings[@]}"; do
  for architecture in "${architectures[@]}"; do
    if [[ "$architecture" == "learned_merge" ]]; then
      c2_targets=(self_consistency)
    else
      c2_targets=(theta)
    fi
    for c2_target in "${c2_targets[@]}"; do
      for leaf in "${leaves[@]}"; do
        if [[ "$architecture" == "learned_merge" && "$leaf" == "1" ]]; then
          echo "skip learned_merge leaf=1 (merge axis inactive)"
          continue
        fi
        out="$OUT_ROOT/$enc/$architecture/c2_${c2_target}/leaf_${leaf}"
        if [[ -f "$out/summary.json" ]]; then
          echo "skip $out"
          continue
        fi
        echo "==> enc=$enc arch=$architecture c2=$c2_target leaf=$leaf"
        CUDA_VISIBLE_DEVICES="$GPU" \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.45}" \
        ctreepo sim run contextual-sbijax \
          --data-source markov \
          --load-data-bundle "$BUNDLE" \
          --sbijax-trainer learned_local_laws \
          --sbijax-method nasss \
          --sbijax-package-theta markov_exact_sketch \
          --sbijax-input-encoding "$enc" \
          --law-architecture "$architecture" \
          --c2-merge-target "$c2_target" \
          --learned-merge-hidden-dim 128 \
          --learned-decoder-hidden-dim 128 \
          --train-docs 10240 --val-docs 1024 --test-docs 1024 \
          --fragment-len "$leaf" \
          --context-samples-per-doc 1 \
          --response-signature-contexts 16 --response-signature-slices 8 \
          --embedding-dim 32 --state-dim 25 --hidden-dim 128 \
          --learning-rate 0.0003 --lr-schedule cosine \
          --n-iter 300 --batch-size 256 \
          --local-law-weight 1.0 \
          --local-law-leaf-weight 1.0 \
          --local-law-merge-weight 1.0 \
          --local-law-idempotence-weight 1.0 \
          --local-law-contextual-weight 1.0 \
          --local-law-package-weight 0.0 \
          --seed 0 \
          --output-root "$out" 2>&1 | tail -4
      done
    done
  done
done

python3 - <<PY
import csv
import json
import os
from pathlib import Path

root = Path("$OUT_ROOT")
rows = []
for dirpath, _dirnames, filenames in os.walk(root):
    if "summary.json" not in filenames:
        continue
    path = Path(dirpath) / "summary.json"
    if path == root / "summary.json":
        continue
    with path.open() as fh:
        data = json.load(fh)
    test = dict(data.get("diagnostics", {}).get("test", {}) or {})
    prov = dict(data.get("provenance", {}) or {})
    history = list(data.get("history") or [])
    final = dict(history[-1]) if history else {}
    rows.append({
        "path": str(path.parent.relative_to(root)),
        "input_encoding": data.get("input_encoding"),
        "law_architecture": prov.get("law_architecture"),
        "c2_merge_target": prov.get("c2_merge_target"),
        "train_docs": 10240,
        "val_docs": 1024,
        "test_docs": 1024,
        "fragment_len": data.get("fragment_len"),
        "contextual_mae": test.get("contextual_mae"),
        "contextual_raw_mae": test.get("contextual_raw_mae"),
        "theta_mae": test.get("theta_mae"),
        "theta_count_raw_mae": test.get("theta_count_raw_mae"),
        "theta_first_regime_accuracy": test.get("theta_first_regime_accuracy"),
        "theta_last_regime_accuracy": test.get("theta_last_regime_accuracy"),
        "eps_leaf": test.get("eps_leaf"),
        "eps_merge": test.get("eps_merge"),
        "eps_idemp": test.get("eps_idemp"),
        "pred_truth_corr": test.get("pred_truth_corr"),
        "best_iteration": final.get("best_iteration"),
        "best_val_law_score": final.get("best_val_law_score"),
    })
rows.sort(key=lambda row: row["path"])
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
if rows:
    fields = list(rows[0].keys())
    with (root / "grid_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
with (root / "grid_report.md").open("w") as fh:
    fh.write("# JAX Large Control Grid\\n\\n")
    fh.write(f"Rows: {len(rows)}\\n\\n")
    fh.write("| path | theta MAE | first/last | eps leaf | eps merge | contextual MAE |\\n")
    fh.write("| --- | ---: | ---: | ---: | ---: | ---: |\\n")
    for row in rows:
        def fmt(key):
            value = row.get(key)
            return "" if value is None else f"{float(value):.6g}"
        first = row.get("theta_first_regime_accuracy")
        last = row.get("theta_last_regime_accuracy")
        edge = "" if first is None or last is None else f"{float(first):.3f}/{float(last):.3f}"
        fh.write(
            f"| {row['path']} | {fmt('theta_mae')} | {edge} | "
            f"{fmt('eps_leaf')} | {fmt('eps_merge')} | {fmt('contextual_mae')} |\\n"
        )
print(f"wrote {root / 'summary.json'} with {len(rows)} rows")
PY
