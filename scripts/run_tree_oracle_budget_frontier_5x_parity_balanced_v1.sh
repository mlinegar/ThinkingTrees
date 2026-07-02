#!/usr/bin/env bash
set -euo pipefail

cd /home/mlinegar/ThinkingTrees
source venv/bin/activate

OUTPUT_ROOT="outputs/tree_oracle_budget_frontier_5x_parity_balanced_v1"
LIVE_LOG="logs/tree_oracle_budget_frontier_5x_parity_balanced_v1_live.log"
AUDIT_LOG="logs/markov_alignment_followup_5x_parity_balanced_v1.log"

mkdir -p "$(dirname "$LIVE_LOG")" "$OUTPUT_ROOT"

python scripts/run_tree_neural_full_doc_mig.py budget_frontier \
  --output-root "$OUTPUT_ROOT" \
  --benchmark recoverable_v4 \
  --train-doc-count 5120 \
  --tree-families tree_neural tree_neural_c2 tree_neural_c2c3 \
  --reference-families official_fno official_fno_sumlen tree_doc_ridge \
  --budget-calls-per-doc 0.5 1.0 2.0 \
  --full-doc-budget-shares 0.0 0.25 0.5 0.75 1.0 \
  --doc-consumption-modes root_only doc_sequence \
  --local-split-modes balanced \
  --budget-tree-config-mode parity \
  --seeds 0 1 2 3 4 \
  --job-granularity family_train_seed \
  --resume \
  --state-dim 128 \
  --hidden-dim 512 \
  --n-epochs 32 \
  --batch-size 64 \
  --lr 5e-4 \
  --local-law-weight 0.3 \
  2>&1 | tee "$LIVE_LOG"

if [[ -f "$OUTPUT_ROOT/summary.json" ]]; then
  python scripts/validate_markov_alignment.py \
    --diagnostics-root "$OUTPUT_ROOT" \
    --full-tree-ipw-root outputs/markov_full_tree_ipw_grid_endpoints_v1 \
    --no-run-lean-build \
    --output-json "$OUTPUT_ROOT/markov_alignment_audit.json" \
    --output-markdown "$OUTPUT_ROOT/markov_alignment_audit.md" \
    2>&1 | tee "$AUDIT_LOG"
fi
