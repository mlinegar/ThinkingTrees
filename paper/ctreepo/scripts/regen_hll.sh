#!/usr/bin/env bash
# Regenerate every HLL-parity asset (table + curve figure) consumed by
# section 7 and appendix F of the C-TreePO paper.
#
# Reads from:
#   outputs/classical_parity/hll/summary.csv
#   outputs/classical_parity/hll/curve.png   (rendered by classical_parity_report.py)
#
# Writes to:
#   paper/ctreepo/assets/hll/{tables,figures}/

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ASSETS_DIR="$REPO_ROOT/paper/ctreepo/assets/hll"
TABLES="$ASSETS_DIR/tables"
FIGURES="$ASSETS_DIR/figures"
mkdir -p "$TABLES" "$FIGURES"
cd "$REPO_ROOT"

if [[ -x ./venv/bin/python ]]; then PY=./venv/bin/python; else PY=python3; fi

CSV=outputs/classical_parity/hll/summary.csv
if [[ ! -f "$CSV" ]]; then
    echo "[hll] missing $CSV — run scripts/run_classical_parity_benchmark.py first to populate."
    exit 0
fi

# classical_parity_report.py emits both the markdown summary table and the curve figure
# from the same CSV. We point it at outputs/classical_parity/hll/ (its native --out-dir),
# then copy the renamed canonical assets into the paper tree.
NATIVE=outputs/classical_parity/hll
"$PY" -m unified_g_v1.sketch.classical_parity_report \
    --in-csv "$CSV" \
    --out-dir "$NATIVE" \
    > /dev/null

# Stage canonical-named copies under assets/. Renaming here keeps the paper
# label scheme uniform (hll_parity_curves, hll_parity_table) regardless of
# what the generator names them in raw outputs/.
[[ -f "$NATIVE/curve.png" ]]            && cp "$NATIVE/curve.png"            "$FIGURES/hll_parity_curves.png"
[[ -f "$NATIVE/curve.pdf" ]]            && cp "$NATIVE/curve.pdf"            "$FIGURES/hll_parity_curves.pdf"
[[ -f "$NATIVE/summary_pivot.tex" ]]    && cp "$NATIVE/summary_pivot.tex"    "$TABLES/classical_parity_hll.tex"
[[ -f "$NATIVE/summary_pivot.md" ]]     && cp "$NATIVE/summary_pivot.md"     "$TABLES/classical_parity_hll.md"

echo "[hll] done. Files:"
ls -la "$TABLES" "$FIGURES" 2>/dev/null | awk '/^-/{printf "  %s %s %s %s\n", $5, $6, $7, $NF}'
