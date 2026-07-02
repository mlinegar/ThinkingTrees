#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"

OUT_ROOT="${1:-outputs/classical_sketches_paper_all_cpu_$(date +%Y%m%d_%H%M%S)}"

PYTHONPATH="$REPO_ROOT/treepo/src:$REPO_ROOT/parallel/unified_g_v1/src:$REPO_ROOT" \
  "$REPO_ROOT/venv/bin/python" "$REPO_ROOT/scripts/run_classical_sketches_paper_bundle.py" \
    --out-root "$OUT_ROOT" \
    --jobs "${CLASSICAL_SKETCH_JOBS:-32}" \
    --seeds "${CLASSICAL_SKETCH_SEEDS:-0,1,2}" \
    --capacities "${CLASSICAL_SKETCH_CAPACITIES:-small,medium,large}" \
    ${CLASSICAL_SKETCH_LEAF_COUNTS:+--leaf-counts $CLASSICAL_SKETCH_LEAF_COUNTS} \
    ${CLASSICAL_SKETCH_LEAF_SIZES:+--leaf-sizes $CLASSICAL_SKETCH_LEAF_SIZES} \
    ${CLASSICAL_SKETCH_SKIP_EXISTING:+--skip-existing} \
    --learned-targets "${CLASSICAL_SKETCH_LEARNED_TARGETS:-all}" \
    --learned-variants "${CLASSICAL_SKETCH_LEARNED_VARIANTS:-g,fg}" \
    --learned-readout-archs "${CLASSICAL_SKETCH_LEARNED_READOUT_ARCHS:-structured}" \
    --learned-epochs "${CLASSICAL_SKETCH_LEARNED_EPOCHS:-150}" \
    --learned-n-train "${CLASSICAL_SKETCH_LEARNED_N_TRAIN:-128}" \
    --learned-n-val "${CLASSICAL_SKETCH_LEARNED_N_VAL:-48}"
