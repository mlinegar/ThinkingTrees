#!/usr/bin/env bash
# Round 2 JAX phases C and D for the Markov FNO bridge follow-up.
#
# Phase C: wider-encoder sweep on JAX learned_local_laws to test whether
#          encoder capacity (state_dim, hidden_dim, embedding_dim) closes
#          the count-MAE drift seen in regime_one_hot/leaf=64 (Mode 1).
#          Inputs: regime_one_hot, one_hot_token_ids.
#
# Phase D: JAX one_hot_token_ids baseline at full leaf grid with seeds —
#          establishes a JAX floor for "raw token input" that PyTorch FNO
#          must beat or match.
set -euo pipefail
source venv/bin/activate

BUNDLE="${BUNDLE:-outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_fno_round2_jax_$(date -u +%Y%m%d_%H%M%S)}"
GPU="${GPU:-1}"

mkdir -p "$OUT_ROOT"

run_jax_cell() {
  local out="$1"
  local encoding="$2"
  local leaf="$3"
  local state_dim="$4"
  local hidden_dim="$5"
  local embedding_dim="$6"
  local arch="$7"
  local c2_target="$8"
  local seed="${9:-0}"

  if [ -f "$out/summary.json" ]; then
    echo "skip $out"
    return 0
  fi
  echo "==> $out"
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
    --law-architecture "$arch" \
    --c2-merge-target "$c2_target" \
    --learned-merge-hidden-dim "$hidden_dim" \
    --learned-decoder-hidden-dim "$hidden_dim" \
    --train-docs 1024 --val-docs 256 --test-docs 256 \
    --fragment-len "$leaf" \
    --context-samples-per-doc 1 \
    --response-signature-contexts 16 --response-signature-slices 8 \
    --embedding-dim "$embedding_dim" --state-dim "$state_dim" --hidden-dim "$hidden_dim" \
    --learning-rate 0.0003 --lr-schedule cosine \
    --n-iter 2000 --batch-size 128 \
    --local-law-weight 1.0 \
    --local-law-leaf-weight 1.0 \
    --local-law-merge-weight 1.0 \
    --local-law-idempotence-weight 1.0 \
    --local-law-contextual-weight 1.0 \
    --local-law-package-weight 0.0 \
    --seed "$seed" \
    --output-root "$out" 2>&1 | tail -3
}

echo ">>> Phase C: wider-encoder sweep"
PHASE_C_ROOT="$OUT_ROOT/phase_c_wider_encoder"
for encoding in regime_one_hot one_hot_token_ids; do
  for leaf in 16 64; do
    for state_dim in 32 128 256; do
      # Match hidden_dim and embedding_dim to encoder scale.
      hidden_dim=$((state_dim * 2))
      if [ "$state_dim" -ge 128 ]; then
        embedding_dim=128
      else
        embedding_dim=64
      fi
      out="$PHASE_C_ROOT/${encoding}/leaf_${leaf}/sd${state_dim}_hd${hidden_dim}_ed${embedding_dim}"
      run_jax_cell "$out" "$encoding" "$leaf" "$state_dim" "$hidden_dim" "$embedding_dim" "analytic" "theta" 0
    done
  done
done

echo ">>> Phase D: one_hot_token_ids baseline at full leaf grid"
PHASE_D_ROOT="$OUT_ROOT/phase_d_token_ids_baseline"
# Use widest config from phase C as the encoder.
for leaf in 2 4 16 64; do
  for arch_pair in "analytic theta" "learned_merge self_consistency"; do
    arch=$(echo "$arch_pair" | awk '{print $1}')
    c2t=$(echo "$arch_pair" | awk '{print $2}')
    for seed in 0 1 2; do
      out="$PHASE_D_ROOT/${arch}_${c2t}/leaf_${leaf}/seed_${seed}"
      run_jax_cell "$out" "one_hot_token_ids" "$leaf" "256" "512" "128" "$arch" "$c2t" "$seed"
    done
  done
done

echo ">>> Aggregating results"
python3 - <<PY
import json
import os
import csv

root = "$OUT_ROOT"
rows = []
for dirpath, _dirnames, filenames in os.walk(root):
    if "summary.json" not in filenames:
        continue
    path = os.path.join(dirpath, "summary.json")
    try:
        with open(path) as fh:
            d = json.load(fh)
    except Exception as exc:
        print(f"  skip {path}: {exc}")
        continue
    test = d.get("diagnostics", {}).get("test", {})
    prov = d.get("provenance", {})
    rows.append({
        "path": os.path.relpath(dirpath, root),
        "input_encoding": d.get("input_encoding"),
        "law_architecture": prov.get("law_architecture"),
        "c2_merge_target": prov.get("c2_merge_target"),
        "fragment_len": prov.get("fragment_len"),
        "state_dim": prov.get("state_dim"),
        "hidden_dim": prov.get("hidden_dim"),
        "embedding_dim": prov.get("embedding_dim"),
        "seed": prov.get("seed"),
        "contextual_mae": test.get("contextual_mae"),
        "contextual_raw_mae": test.get("contextual_raw_mae"),
        "theta_mae": test.get("theta_mae"),
        "raw_count_mae": test.get("theta_count_raw_mae"),
        "theta_first_regime_accuracy": test.get("theta_first_regime_accuracy"),
        "theta_last_regime_accuracy": test.get("theta_last_regime_accuracy"),
    })

if rows:
    fieldnames = list(rows[0].keys())
    csv_path = os.path.join(root, "phases_c_d_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")
else:
    print("no summary.json files found")
PY

echo ">>> Done. Output root: $OUT_ROOT"
