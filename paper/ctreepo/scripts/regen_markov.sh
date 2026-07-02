#!/usr/bin/env bash
# Regenerate every Markov-changepoint asset (figures, optional tables)
# consumed by sections 4 and 6 of the C-TreePO paper.
#
# Reads from:
#   outputs/markov_v5_*  (root-only and reallocation-policy run dirs)
#   outputs/allocation_figure_variants/   (current canonical render destination)
#
# Writes to:
#   paper/ctreepo/assets/markov/{tables,figures}/

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ASSETS_DIR="$REPO_ROOT/paper/ctreepo/assets/markov"
TABLES="$ASSETS_DIR/tables"
FIGURES="$ASSETS_DIR/figures"
mkdir -p "$TABLES" "$FIGURES"
cd "$REPO_ROOT"

if [[ -x ./venv/bin/python ]]; then PY=./venv/bin/python; else PY=python3; fi

# Render the leaf-mass and budget-split figures into outputs/allocation_figure_variants/.
# This script discovers its own input runs from outputs/markov_v5_* and is
# idempotent on existing renders.
"$PY" scripts/render_allocation_figure_variants.py > /dev/null 2>&1 || \
    echo "[markov] render_allocation_figure_variants.py exited nonzero — assuming already-rendered PNGs are reusable"

NATIVE=outputs/allocation_figure_variants
# Canonical-named copies under assets/. The paper LaTeX references these
# names; the variant_b_plain_panels.png suffix is the production preset.
[[ -f "$NATIVE/simple_variant_b_plain_panels.png" ]] && cp "$NATIVE/simple_variant_b_plain_panels.png" "$FIGURES/markov_simple_leaf_mass.png"
[[ -f "$NATIVE/hard_variant_b_plain_panels.png"   ]] && cp "$NATIVE/hard_variant_b_plain_panels.png"   "$FIGURES/markov_hard_leaf_mass.png"
[[ -f "$NATIVE/simple_budget_split.png"           ]] && cp "$NATIVE/simple_budget_split.png"           "$FIGURES/markov_budget_split.png"

# Section 4's overview figure (hand-built combined slide) is checked in under
# paper/figures/. Stage a copy under assets/markov/figures/ for consistency.
LEGACY=paper/figures/markov_changepoint_combined_slide.pdf
[[ -f "$LEGACY" ]] && cp "$LEGACY" "$FIGURES/markov_changepoint_combined.pdf"

echo "[markov] done. Files:"
ls -la "$TABLES" "$FIGURES" 2>/dev/null | awk '/^-/{printf "  %s %s %s %s\n", $5, $6, $7, $NF}'
