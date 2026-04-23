#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

stamp="${1:-$(date -u +%Y%m%d_%H%M%S)}"

python3 scripts/run_tree_neural_learned_surface_push.py \
  --output-root "outputs/tree_neural_slotwise_push_tw1_${stamp}" \
  --theorem-surface-mode slotwise \
  --task-weight 1.0

python3 scripts/run_tree_neural_learned_surface_push.py \
  --output-root "outputs/tree_neural_slotwise_push_tw2_${stamp}" \
  --theorem-surface-mode slotwise \
  --task-weight 2.0

python3 scripts/run_tree_neural_learned_surface_push.py \
  --output-root "outputs/tree_neural_learnedproj_push_tw2_${stamp}" \
  --theorem-surface-mode learned_projection \
  --task-weight 2.0
