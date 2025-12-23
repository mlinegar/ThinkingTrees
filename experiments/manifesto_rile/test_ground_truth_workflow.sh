#!/bin/bash
#
# Test Ground Truth Preference Pair Collection Workflow
#
# This script demonstrates the end-to-end workflow for collecting
# preference pairs using oracle ground truth trees.
#
# Prerequisites:
# 1. Oracle server running on port 8001 (large Nemotron model)
# 2. Summarizer server running on port 8000 (small model)
# 3. Manifesto data available
#
# Usage:
#   ./test_ground_truth_workflow.sh

set -e  # Exit on error

echo "=============================================================================="
echo "  GROUND TRUTH PREFERENCE PAIR COLLECTION - TEST WORKFLOW"
echo "=============================================================================="
echo ""

# Configuration
ORACLE_PORT=8001
SUMMARIZER_PORT=8000
GENRM_PORT=8001  # Same as oracle for this test
MAX_DOCS=5       # Small number for testing
CHUNK_SIZE=4000
K_CANDIDATES=4

# Output directories
GT_DIR="data/ground_truth_test"
PREF_DIR="data/preferences_test"

echo "Configuration:"
echo "  Oracle Port:       $ORACLE_PORT"
echo "  Summarizer Port:   $SUMMARIZER_PORT"
echo "  GenRM Port:        $GENRM_PORT"
echo "  Max Documents:     $MAX_DOCS"
echo "  Ground Truth Dir:  $GT_DIR"
echo "  Preferences Dir:   $PREF_DIR"
echo ""

# Step 1: Generate Oracle Ground Truth Trees
echo "=============================================================================="
echo "  STEP 1: Generate Oracle Ground Truth Trees"
echo "=============================================================================="
echo ""

python experiments/manifesto_rile/generate_oracle_ground_truth.py \
    --oracle-port $ORACLE_PORT \
    --max-documents $MAX_DOCS \
    --chunk-size $CHUNK_SIZE \
    --output-dir $GT_DIR

echo ""
echo "Ground truth trees generated!"
echo ""

# Step 2: Collect Sufficiency Preferences
echo "=============================================================================="
echo "  STEP 2: Collect Sufficiency Preference Pairs"
echo "=============================================================================="
echo ""

python experiments/manifesto_rile/collect_preferences_with_ground_truth.py \
    --ground-truth-dir $GT_DIR \
    --law-type sufficiency \
    --genrm-port $GENRM_PORT \
    --summarizer-port $SUMMARIZER_PORT \
    --k-candidates $K_CANDIDATES \
    --max-trees 3 \
    --max-chunks-per-tree 2 \
    --output-dir $PREF_DIR

echo ""
echo "Sufficiency preferences collected!"
echo ""

# Step 3: Collect Idempotence Preferences
echo "=============================================================================="
echo "  STEP 3: Collect Idempotence Preference Pairs"
echo "=============================================================================="
echo ""

python experiments/manifesto_rile/collect_preferences_with_ground_truth.py \
    --ground-truth-dir $GT_DIR \
    --law-type idempotence \
    --genrm-port $GENRM_PORT \
    --summarizer-port $SUMMARIZER_PORT \
    --k-candidates $K_CANDIDATES \
    --max-trees 3 \
    --max-chunks-per-tree 2 \
    --output-dir $PREF_DIR

echo ""
echo "Idempotence preferences collected!"
echo ""

# Step 4: Collect Merge Preferences
echo "=============================================================================="
echo "  STEP 4: Collect Merge Preference Pairs"
echo "=============================================================================="
echo ""

python experiments/manifesto_rile/collect_preferences_with_ground_truth.py \
    --ground-truth-dir $GT_DIR \
    --law-type merge \
    --genrm-port $GENRM_PORT \
    --summarizer-port $SUMMARIZER_PORT \
    --k-candidates $K_CANDIDATES \
    --max-trees 3 \
    --max-chunks-per-tree 2 \
    --output-dir $PREF_DIR

echo ""
echo "Merge preferences collected!"
echo ""

# Summary
echo "=============================================================================="
echo "  TEST WORKFLOW COMPLETE!"
echo "=============================================================================="
echo ""
echo "Output directories:"
echo "  Ground Truth:  $GT_DIR"
echo "  Preferences:   $PREF_DIR"
echo ""
echo "Files generated:"
echo ""
echo "Ground Truth Trees:"
ls -lh $GT_DIR/*.json | head -n 10
echo ""
echo "Preference Pairs:"
ls -lh $PREF_DIR/*.json | head -n 10
echo ""
echo "=============================================================================="
echo ""
echo "Next steps:"
echo "  1. Inspect preference pairs: cat $PREF_DIR/preferences_gt_*.json | jq '.pairs[0]'"
echo "  2. View statistics: cat $PREF_DIR/collection_stats_gt_*.json | jq '.'"
echo "  3. Train OPS comparison module: experiments/manifesto_rile/train_ops_comparison.py"
echo "  4. Generate DPO data: experiments/manifesto_rile/generate_dpo_data.py"
echo ""
echo "=============================================================================="
