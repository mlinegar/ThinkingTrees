#!/usr/bin/env bash
# Regenerate every Benoit-replication asset (tables + figures) consumed
# by section 9 and appendix H of the C-TreePO paper.
#
# Reads from:
#   - outputs/overnight_benoit/{scorer_only,full_pipeline,optimizer_*}/{dim}/report.json
#   - outputs/phase2/{combined_pipeline,joint_optimize,joint_gepa}/report.json
#   - outputs/phase3/combined_c{chunk}/report.json
#   - outputs/chunk_sweep/{dim}_c{chunk}/report.json
#   - outputs/gemma3/scorer_*/{dim}/report.json
#   - outputs/rescore/T*_N*/<...>/report.json
#
# Writes to:
#   paper/ctreepo/assets/benoit/{tables,figures}/

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ASSETS_DIR="$REPO_ROOT/paper/ctreepo/assets/benoit"
TABLES="$ASSETS_DIR/tables"
FIGURES="$ASSETS_DIR/figures"
mkdir -p "$TABLES" "$FIGURES"
cd "$REPO_ROOT"

if [[ -x ./venv/bin/python ]]; then PY=./venv/bin/python; else PY=python3; fi

echo "[benoit] regenerating tables in $TABLES"

# Headline comparison: Pearson r (the table that section 9 inputs)
"$PY" scripts/comparison_table.py --metric pearson \
    --out-md  "$TABLES/benoit_comparison_pearson.md" \
    --out-tex "$TABLES/benoit_comparison_pearson.tex" \
    > /dev/null

# Same comparison in MAE
"$PY" scripts/comparison_table.py --metric mae \
    --out-md  "$TABLES/benoit_comparison_mae.md" \
    --out-tex "$TABLES/benoit_comparison_mae.tex" \
    > /dev/null

# T,N rescore variants (Benoit-style averaging at T=0.2 N=3, plus T=0.3 / 0.7 single-sample)
"$PY" scripts/comparison_table.py --metric pearson --rescore-key T0.2_N3 \
    --hide-missing-rows \
    --out-md  "$TABLES/benoit_comparison_T0.2_N3.md" \
    --out-tex "$TABLES/benoit_comparison_T0.2_N3.tex" \
    > /dev/null 2>&1 || true
for k in T0.3_N1 T0.7_N1; do
    "$PY" scripts/comparison_table.py --metric pearson --rescore-key "$k" \
        --hide-missing-rows \
        --out-md  "$TABLES/benoit_comparison_${k}.md" \
        --out-tex "$TABLES/benoit_comparison_${k}.tex" \
        > /dev/null 2>&1 || true
done

# Per-dim chunk-size sweep + combined chunk-size sweep (markdown only — appendix uses these as reference)
"$PY" scripts/aggregate_chunk_sweep.py    --out-md "$TABLES/chunk_sweep_per_dim.md"  > /dev/null
"$PY" scripts/aggregate_phase3_combined.py --out-md "$TABLES/chunk_sweep_combined.md" > /dev/null

# Post-hoc breakdowns (referenced from appendix H)
"$PY" scripts/analyze_by_language.py --out-md "$TABLES/language_breakdown.md" > /dev/null 2>&1 || true
"$PY" scripts/analyze_by_length.py   --out-md "$TABLES/length_buckets.md"     > /dev/null 2>&1 || true
"$PY" scripts/analyze_by_era.py      --out-md "$TABLES/era_breakdown.md"      > /dev/null 2>&1 || true
"$PY" scripts/analyze_vs_mp_logit.py --out-md "$TABLES/vs_mp_logit.md"        > /dev/null 2>&1 || true

# Overnight roundup (kept for reference in the appendix)
"$PY" scripts/roundup_overnight.py > /dev/null 2>&1 || true
[[ -f outputs/overnight_benoit/roundup.md ]] && cp outputs/overnight_benoit/roundup.md "$TABLES/overnight_roundup.md"

echo "[benoit] done. Files:"
ls -la "$TABLES" "$FIGURES" 2>/dev/null | awk '/^-/{printf "  %s %s %s %s\n", $5, $6, $7, $NF}'
