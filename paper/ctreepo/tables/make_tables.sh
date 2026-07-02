#!/usr/bin/env bash
# Regenerate all data-driven tables for paper/ctreepo.
# Reads from outputs/{overnight_benoit,phase2,phase3,chunk_sweep,gemma3}/*
# and writes markdown + LaTeX into paper/ctreepo/tables/.
#
# Idempotent and safe to re-run at any time. New data files just show up in
# the next run; missing cells render as em-dashes.
#
# Usage:
#   bash paper/ctreepo/tables/make_tables.sh

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TABLES_DIR="$REPO_ROOT/paper/ctreepo/tables"
cd "$REPO_ROOT"

if [[ -x ./venv/bin/python ]]; then
    PY="./venv/bin/python"
else
    PY="python3"
fi

echo "Regenerating paper/ctreepo/tables/ (repo=$REPO_ROOT)"

# 1. Main Benoit-vs-ours comparison, Pearson r (markdown + LaTeX)
"$PY" scripts/comparison_table.py --metric pearson \
    --out-md  "$TABLES_DIR/benoit_comparison_pearson.md" \
    --out-tex "$TABLES_DIR/benoit_comparison_pearson.tex" \
    > /dev/null

# 2. Same comparison table, MAE metric (sanity + alternative reporting)
"$PY" scripts/comparison_table.py --metric mae \
    --out-md  "$TABLES_DIR/benoit_comparison_mae.md" \
    --out-tex "$TABLES_DIR/benoit_comparison_mae.tex" \
    > /dev/null

# 2b. Rescored variants: N=3 ensemble at T=0.2 (Benoit-style averaging)
"$PY" scripts/comparison_table.py --metric pearson --rescore-key T0.2_N3 \
    --hide-missing-rows \
    --out-md  "$TABLES_DIR/benoit_comparison_T0.2_N3.md" \
    --out-tex "$TABLES_DIR/benoit_comparison_T0.2_N3.tex" \
    > /dev/null 2>&1 || true

# 2c. Temperature sweep: T=0.3 single-sample and T=0.7 single-sample
for k in T0.3_N1 T0.7_N1; do
    "$PY" scripts/comparison_table.py --metric pearson --rescore-key "$k" \
        --hide-missing-rows \
        --out-md  "$TABLES_DIR/benoit_comparison_${k}.md" \
        --out-tex "$TABLES_DIR/benoit_comparison_${k}.tex" \
        > /dev/null 2>&1 || true
done

# 3. Per-dim chunk-size sweep (6 dims × 4 leaf sizes)
"$PY" scripts/aggregate_chunk_sweep.py \
    --out-md "$TABLES_DIR/chunk_sweep_per_dim.md" \
    > /dev/null

# 4. Combined pipeline × chunk-size sweep
"$PY" scripts/aggregate_phase3_combined.py \
    --out-md "$TABLES_DIR/chunk_sweep_combined.md" \
    > /dev/null

# 5. Per-language breakdown (post-hoc re-aggregation of existing full_pipeline data)
"$PY" scripts/analyze_by_language.py \
    --out-md "$TABLES_DIR/language_breakdown.md" > /dev/null 2>&1 || true

# 5b. Per-length-bucket breakdown (shows tree > flat on long documents)
"$PY" scripts/analyze_by_length.py \
    --out-md "$TABLES_DIR/length_buckets.md" > /dev/null 2>&1 || true

# 5c. Cross-era breakdown (1989-99 / 2000-09 / 2010-19)
"$PY" scripts/analyze_by_era.py \
    --out-md "$TABLES_DIR/era_breakdown.md" > /dev/null 2>&1 || true

# 5d. vs MP hand-coded logit scores (Benoit's secondary benchmark)
"$PY" scripts/analyze_vs_mp_logit.py \
    --out-md "$TABLES_DIR/vs_mp_logit.md" > /dev/null 2>&1 || true

# 6. Full overnight roundup (per-dim + combined + optimizer baseline/optimized + GEPA)
"$PY" scripts/roundup_overnight.py > /dev/null 2>&1 || true
if [[ -f outputs/overnight_benoit/roundup.md ]]; then
    cp outputs/overnight_benoit/roundup.md "$TABLES_DIR/overnight_roundup.md"
fi

# 7. Classical-HLL parity (Appendix F) — only if the sweep has been run.
if [[ -f outputs/classical_parity/hll/summary.csv ]]; then
    PYTHONPATH="$REPO_ROOT/parallel/unified_g_v1/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        "$PY" -m unified_g_v1.sketch.classical_parity_report \
        --summary outputs/classical_parity/hll/summary.csv \
        --out-dir outputs/classical_parity/hll \
        --tables-dir "$TABLES_DIR" > /dev/null 2>&1 || true
fi

echo "Updated:"
ls -la "$TABLES_DIR"/*.md "$TABLES_DIR"/*.tex 2>/dev/null | awk '{printf "  %s %s %s %s\n", $5, $6, $7, $NF}'
