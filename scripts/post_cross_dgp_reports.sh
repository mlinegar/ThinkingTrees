#!/usr/bin/env bash
# Post-completion report generation for the Markov cross-DGP refresh.
# Waits for the MIG queue to finish, then generates all reports.
#
# Usage:
#   nohup bash scripts/post_cross_dgp_reports.sh > logs/post_cross_dgp_reports.out 2>&1 &

set -euo pipefail
cd /home/mlinegar/ThinkingTrees
source venv/bin/activate
export PYTHONPATH=.
export MPLBACKEND=Agg

MARKOV_OUTPUT="outputs/markov_law_stress_20260309_cross_dgp_refresh"
MIG_PID_FILE="logs/markov_law_stress_20260309_cross_dgp_refresh/mig_queue.pid"
MANIFEST="outputs/existing_local_law_inventory_20260309/existing_local_law_manifest.jsonl"
REPORT_TS="$(date +%Y%m%d_%H%M%S)"

echo "=== post_cross_dgp_reports.sh started at $(date -u) ==="

# --- Step 0: Wait for MIG queue to finish ---
if [ -f "$MIG_PID_FILE" ]; then
    MIG_PID=$(cat "$MIG_PID_FILE")
    echo "Waiting for MIG queue (PID $MIG_PID) to finish..."
    while kill -0 "$MIG_PID" 2>/dev/null; do
        DONE=$(find "$MARKOV_OUTPUT" -name "*.json" | wc -l)
        echo "  $(date -u): $DONE JSON files so far"
        sleep 120
    done
fi
TOTAL=$(find "$MARKOV_OUTPUT" -name "*.json" | wc -l)
echo "MIG queue done. Total JSON outputs: $TOTAL"

# --- Step 1: Generate Markov law-stress report ---
echo ""
echo "=== Step 1: Markov law-stress report ==="
python scripts/report_markov_law_stress.py \
    --input-root "$MARKOV_OUTPUT" \
    --output-dir "$MARKOV_OUTPUT/report" \
    2>&1

# --- Step 2: Regenerate cross-DGP report (fresh Markov + existing LDA) ---
echo ""
echo "=== Step 2: Cross-DGP report (fresh Markov + existing LDA manifest) ==="
CROSS_DGP_OUT="outputs/existing_local_law_inventory_20260309/cross_dgp_report_${REPORT_TS}"
python scripts/report_cross_dgp_law_stress.py \
    --markov-dir "$MARKOV_OUTPUT/report" \
    --manifest "$MANIFEST" \
    --output-dir "$CROSS_DGP_OUT" \
    2>&1

# Also symlink as "latest"
ln -sfn "cross_dgp_report_${REPORT_TS}" \
    outputs/existing_local_law_inventory_20260309/cross_dgp_report_latest

# --- Step 3: Regenerate meta report ---
echo ""
echo "=== Step 3: Meta report ==="
META_OUT="outputs/existing_local_law_inventory_20260309/meta_report_${REPORT_TS}"
python scripts/report_local_law_meta.py \
    --manifest "$MANIFEST" \
    --output-dir "$META_OUT" \
    2>&1

ln -sfn "meta_report_${REPORT_TS}" \
    outputs/existing_local_law_inventory_20260309/meta_report_latest

# --- Step 4: Rebuild inventory to include fresh Markov ---
echo ""
echo "=== Step 4: Rebuild inventory manifest ==="
python scripts/organize_existing_local_law_runs.py \
    --output-dir "outputs/existing_local_law_inventory_20260309" \
    --include-root "$MARKOV_OUTPUT" \
    2>&1

# --- Done ---
echo ""
echo "=== All reports complete at $(date -u) ==="
echo "Cross-DGP report: $CROSS_DGP_OUT"
echo "Meta report:      $META_OUT"
echo "Markov report:    $MARKOV_OUTPUT/report"
