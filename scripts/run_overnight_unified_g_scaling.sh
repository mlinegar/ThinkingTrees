#!/usr/bin/env bash
# Overnight unified_g robustness + 4-leaf scaling experiments.
# Phase A: solidify 2-leaf (4 runs, GPUs 0-3, ~2h)
# Phase B: scale to 4-leaf (4 runs, GPUs 0-3, ~3-5h)
#
# Launch with:
#   nohup bash scripts/run_overnight_unified_g_scaling.sh > outputs/overnight_master.log 2>&1 &
set -euo pipefail

# Activate the project venv so `python` resolves correctly.
source "$(dirname "$0")/../venv/bin/activate"

OUT_ROOT="outputs/overnight_$(date +%Y%m%d)"
mkdir -p "$OUT_ROOT"

DATA_ROOT="outputs/parity_corpus_20260403_v4/benchmark_corpora/recoverable_v4"

# Config matches outputs/unified_g_leaf64_test/worker_invocation_snapshot.json exactly.
# Flags where default != winning config are commented.
COMMON=(
  --family tree_neural
  --tree-model-version unified_g
  --tree-score-merge-mode exact_projected_sketch          # default: gated_affine
  --preserve-requested-leaf-tokens
  --train-doc-count 4096
  --benchmark recoverable_v4
  --gpu-runtime-data-mode resident
  --tree-batch-pack-mode fixed_fused
  --tree-batch-autotune
  --prepared-data-root "$DATA_ROOT/prepared_tree_data"
  --prepared-data-allow-create
  --base-bundle-path "$DATA_ROOT/bundles/bundle_train4096.pkl"
  # Architecture
  --state-dim 128 --hidden-dim 512
  --tree-leaf-fno-width 128 --tree-leaf-fno-n-modes 8 --tree-leaf-fno-n-layers 4
  # Training
  --batch-size 64 --lr 5e-4 --weight-decay 0.0
  --n-epochs 40
  --tree-training-schedule two_stage
  --tree-stage1-epochs 10 --tree-stage2-epochs 30
  --tree-local-law-weight 0.8
  --tree-task-objective-weight 1.0
  # Supervision
  --tree-root-supervision-kind mse
  --internal-supervision-kind full_sketch                  # default: none
  --internal-label-rate 1.0                                # default: 0.0
  --leaf-supervision-kind full_sketch
  --tree-join-bit-weight 1.0                               # default: 0.0
  --tree-phi-compose-weight 0.0                            # default: 1.0
  --tree-phi-contrastive-weight 0.0                        # default: 0.25
  # Theorem surface
  --tree-task-head-mode theorem_feature_scalar             # default: full_state_scalar
  --tree-theorem-surface-mode factorized_score_fiber       # default: slotwise
  --tree-summary-spec-root-mode factored_theorem_readout   # default: task_split_ablation
  --tree-theorem-count-head-mode scalar_mse
  --tree-theorem-feature-dim 48 --tree-theorem-feature-hidden-dim 256
  --tree-theorem-fiber-dim 47
  --tree-theorem-count-dim 8 --tree-theorem-first-dim 8 --tree-theorem-last-dim 8
  --tree-theorem-score-dim 1                               # default: 0
  --summary-spec-name markov_count_sketch                  # default: ""
  --slot-count 4                                           # default: 0
  # Checkpointing
  --tree-stage1-checkpoint-metric val_theorem_bootstrap_direct
  --tree-checkpoint-metric val_root_mae
  --tree-exact-eval-max-docs 64                            # default: 0
  # Runtime
  --use-cuda
)

echo "=== Phase A: 2-leaf robustness ($(date)) ==="
echo "Output: $OUT_ROOT"

# A1: 1-leaf baseline (tree = pure FNO, no merges)
CUDA_VISIBLE_DEVICES=0 python scripts/run_tree_neural_full_doc_mig.py worker \
  --job-name A1_leaf128_s0 --output-dir "$OUT_ROOT/A1_leaf128_s0" \
  "${COMMON[@]}" --fixed-leaf-tokens 128 --seeds 0 \
  > "$OUT_ROOT/A1_leaf128_s0.log" 2>&1 &

# A2: 2-leaf seed 1
CUDA_VISIBLE_DEVICES=1 python scripts/run_tree_neural_full_doc_mig.py worker \
  --job-name A2_leaf64_s1 --output-dir "$OUT_ROOT/A2_leaf64_s1" \
  "${COMMON[@]}" --fixed-leaf-tokens 64 --seeds 1 \
  > "$OUT_ROOT/A2_leaf64_s1.log" 2>&1 &

# A3: 2-leaf seed 2
CUDA_VISIBLE_DEVICES=2 python scripts/run_tree_neural_full_doc_mig.py worker \
  --job-name A3_leaf64_s2 --output-dir "$OUT_ROOT/A3_leaf64_s2" \
  "${COMMON[@]}" --fixed-leaf-tokens 64 --seeds 2 \
  > "$OUT_ROOT/A3_leaf64_s2.log" 2>&1 &

# A4: 2-leaf seed 0 reproducibility
CUDA_VISIBLE_DEVICES=3 python scripts/run_tree_neural_full_doc_mig.py worker \
  --job-name A4_leaf64_s0_repro --output-dir "$OUT_ROOT/A4_leaf64_s0_repro" \
  "${COMMON[@]}" --fixed-leaf-tokens 64 --seeds 0 \
  > "$OUT_ROOT/A4_leaf64_s0_repro.log" 2>&1 &

wait
echo "=== Phase A complete ($(date)) ==="

echo "=== Phase B: 4-leaf scaling ($(date)) ==="

# B1: 4-leaf baseline (same hyperparams as winning 2-leaf)
CUDA_VISIBLE_DEVICES=0 python scripts/run_tree_neural_full_doc_mig.py worker \
  --job-name B1_leaf32_s0 --output-dir "$OUT_ROOT/B1_leaf32_s0" \
  "${COMMON[@]}" --fixed-leaf-tokens 32 --seeds 0 \
  > "$OUT_ROOT/B1_leaf32_s0.log" 2>&1 &

# B2: 4-leaf with 80 epochs (deeper trees may need longer training)
CUDA_VISIBLE_DEVICES=1 python scripts/run_tree_neural_full_doc_mig.py worker \
  --job-name B2_leaf32_s0_80ep --output-dir "$OUT_ROOT/B2_leaf32_s0_80ep" \
  "${COMMON[@]}" --fixed-leaf-tokens 32 --seeds 0 \
  --n-epochs 80 --tree-stage1-epochs 20 --tree-stage2-epochs 60 \
  > "$OUT_ROOT/B2_leaf32_s0_80ep.log" 2>&1 &

# B3: 4-leaf with higher local law weight (stronger C3 algebra signal)
CUDA_VISIBLE_DEVICES=2 python scripts/run_tree_neural_full_doc_mig.py worker \
  --job-name B3_leaf32_s0_highlaw --output-dir "$OUT_ROOT/B3_leaf32_s0_highlaw" \
  "${COMMON[@]}" --fixed-leaf-tokens 32 --seeds 0 \
  --tree-local-law-weight 1.2 \
  > "$OUT_ROOT/B3_leaf32_s0_highlaw.log" 2>&1 &

# B4: 4-leaf seed 1 (variance check)
CUDA_VISIBLE_DEVICES=3 python scripts/run_tree_neural_full_doc_mig.py worker \
  --job-name B4_leaf32_s1 --output-dir "$OUT_ROOT/B4_leaf32_s1" \
  "${COMMON[@]}" --fixed-leaf-tokens 32 --seeds 1 \
  > "$OUT_ROOT/B4_leaf32_s1.log" 2>&1 &

wait
echo "=== Phase B complete ($(date)) ==="

# Summary
echo ""
echo "=== Results Summary ($(date)) ==="
for d in "$OUT_ROOT"/*/; do
  name=$(basename "$d")
  echo "--- $name ---"
  python3 -c "
import csv, sys
try:
    with open('${d}runs.csv') as f:
        for r in csv.DictReader(f):
            print(f'  root_mae={r.get(\"test_root_mae\",\"?\")}, leaf_mae={r.get(\"test_leaf_mae\",\"?\")}, merge_mae={r.get(\"test_merge_mae\",\"?\")}, c2_idem={r.get(\"test_c2_idempotence_mae\",\"?\")}, best_ep={r.get(\"best_epoch\",\"?\")}')
except Exception as e:
    print(f'  (no results: {e})')
" 2>/dev/null || echo "  (no results)"
done

echo ""
echo "All experiments done at $(date)"
