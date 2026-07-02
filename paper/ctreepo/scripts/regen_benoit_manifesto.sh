#!/usr/bin/env bash
# Regenerate the manifesto f/g ladder publication figures and table.
#
# Sources the canonical per-lane CSVs from the live bundle output dirs plus
# the Apr 24 raw-init retry run, then writes:
#   paper/ctreepo/assets/benoit/figures/manifesto_fg_ladder_{benoit_init,raw_init,headline}.{pdf,png}
#   paper/ctreepo/assets/benoit/tables/manifesto_fg_ladder.tex
#
# Referenced from paper/ctreepo/figures/README.md.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

PY="venv/bin/python"
OUT="manifesto_fg_alternating"

BENOIT_CSV="outputs/${OUT}/benoit_grid_plots_benoit_init/manifesto_fg_ladder_grid_rows.csv"
RAW_CSV_BUNDLE="outputs/${OUT}/benoit_grid_plots_raw_init/manifesto_fg_ladder_grid_rows.csv"
RAW_CSV_RETRY="outputs/${OUT}/economic_benoit_g0init_largeleaves_retry_20260424_085154/plots/manifesto_fg_ladder_grid_rows.csv"

for f in "$BENOIT_CSV" "$RAW_CSV_BUNDLE" "$RAW_CSV_RETRY"; do
    if [ ! -f "$f" ]; then
        echo "missing CSV: $f" >&2
        exit 1
    fi
done

FIG_DIR="paper/ctreepo/assets/benoit/figures"
TABLE_PATH="paper/ctreepo/assets/benoit/tables/manifesto_fg_ladder.tex"

echo "=> rendering manifesto f/g ladder publication figures"
"$PY" paper/ctreepo/scripts/build_manifesto_fg_figures.py \
    --benoit-csv "$BENOIT_CSV" \
    --raw-csv    "$RAW_CSV_BUNDLE" \
    --raw-csv    "$RAW_CSV_RETRY" \
    --output-dir "$FIG_DIR"

echo "=> regenerating manifesto_fg_ladder.tex"
"$PY" paper/ctreepo/scripts/render_ladder_table.py \
    --benoit-csv "$BENOIT_CSV" \
    --raw-csv    "$RAW_CSV_BUNDLE" \
    --raw-csv    "$RAW_CSV_RETRY" \
    --output     "$TABLE_PATH"

echo
echo "done. updated:"
ls -la "$FIG_DIR"/manifesto_fg_ladder_*.pdf "$FIG_DIR"/manifesto_fg_ladder_*.png 2>/dev/null || true
ls -la "$TABLE_PATH"

echo
echo "README row suggestion (paste into paper/ctreepo/figures/README.md):"
today=$(date -u +%Y-%m-%d)
echo "| \`fig:min-manifesto-ladder\` (App. H) | \`paper/ctreepo/assets/benoit/figures/manifesto_fg_ladder_benoit_init.png\` | \`bash paper/ctreepo/scripts/regen_benoit_manifesto.sh\` | ${today} |"
echo "| \`fig:manifesto-fg-headline\` (§9.4) | \`paper/ctreepo/assets/benoit/figures/manifesto_fg_ladder_headline.pdf\` | \`bash paper/ctreepo/scripts/regen_benoit_manifesto.sh\` | ${today} |"
