#!/usr/bin/env bash
# End-to-end grid run: teacher traces at 1024+2048 tokens (512 cached from
# prior run), then alternating ladder at all three sizes with MIPRO light
# DSPy + FNO + TRL passthrough. Designed to run fully unattended after a
# manual kickoff; logs land in /tmp/economic_grid_*.log.

set -u  # no -e: we want to continue even if one step partially fails.

TEACHER_DIR="outputs/manifesto_teacher_fg_size_grid/economic_gemma4_size0512tok"
STAMP=$(date +%Y%m%d_%H%M%S)
LADDER_DIR="outputs/manifesto_fg_alternating/economic_multisize_mipro_${STAMP}"
TEACHER_LOG=/tmp/economic_grid_teacher.log
LADDER_LOG=/tmp/economic_grid_ladder.log

cd /home/mlinegar/ThinkingTrees
source venv/bin/activate

echo "=== $(date -u) :: STEP 1/2 :: teacher traces for 1024 + 2048 ==="
echo "Output dir: $TEACHER_DIR"
echo "Log: $TEACHER_LOG"
python3 scripts/run_manifesto_teacher_fg_leaf_grid.py \
    --dimension economic \
    --leaf-size-tokens 1024,2048 \
    --teacher-base-url http://localhost:8010/v1 \
    --teacher-model nvidia/Gemma-4-31B-IT-NVFP4 \
    --teacher-api-key EMPTY \
    --num-workers 64 \
    --lm-concurrency 32 \
    --skip-existing \
    --missing-score-policy neutral \
    --resummary-max-chars 7500 \
    --node-summary-max-chars 16000 \
    --score-max-chars 8000 \
    --output-dir "$TEACHER_DIR" \
    > "$TEACHER_LOG" 2>&1
TEACHER_RC=$?
echo "=== $(date -u) :: teacher exit=$TEACHER_RC ==="

if [[ $TEACHER_RC -ne 0 ]]; then
    echo "TEACHER FAILED; still attempting ladder on whatever trees exist."
fi

# Check what leaf-size dirs actually have labeled_trees.jsonl at this point.
SIZES=""
for t in 512 1024 2048; do
    if [[ -f "$TEACHER_DIR/leaf$(printf '%04d' $t)tok/labeled_trees.jsonl" ]]; then
        SIZES+="$t,"
    fi
done
SIZES="${SIZES%,}"  # trim trailing comma
echo "=== leaf sizes with traces ready: $SIZES ==="

if [[ -z "$SIZES" ]]; then
    echo "No traces available — aborting ladder step."
    exit 1
fi

echo "=== $(date -u) :: STEP 2/2 :: MIPRO ladder across sizes $SIZES ==="
echo "Output dir: $LADDER_DIR"
echo "Log: $LADDER_LOG"

# max_completion_tokens = 4096 satisfies 2xleaf for all of {512, 1024, 2048}:
# leaf=2048 needs output >= 4096 (equality passes the config check).
# Total budget: 2*leaf + 4096 output + 1500 overhead <= 12000 for every size.
python3 scripts/run_alternating_ladder.py \
    --families fno,dspy,trl \
    --leaf-size-tokens "$SIZES" \
    --max-iterations 2 \
    --fg-grid-dir "$TEACHER_DIR" \
    --embedding-backend local-hf \
    --embedding-model /mnt/data/models/google/embeddinggemma-300m \
    --embedding-device cuda \
    --embedding-batch-size 16 \
    --embedding-max-length 2048 \
    --dspy-optimizer mipro \
    --dspy-budget light \
    --dspy-num-threads 16 \
    --dspy-max-tokens 4096 \
    --fno-hidden-channels 32 --fno-n-modes 64 --fno-n-layers 2 --fno-head-hidden-dim 64 \
    --fno-epochs 20 \
    --output-dir "$LADDER_DIR" \
    > "$LADDER_LOG" 2>&1
LADDER_RC=$?
echo "=== $(date -u) :: ladder exit=$LADDER_RC ==="

if [[ -f "$LADDER_DIR/grid_summary.md" ]]; then
    echo
    echo "=== GRID SUMMARY (cat of grid_summary.md) ==="
    cat "$LADDER_DIR/grid_summary.md"
fi

echo
echo "=== DONE $(date -u) ==="
echo "Teacher: $TEACHER_DIR"
echo "Ladder:  $LADDER_DIR"
echo "Teacher log: $TEACHER_LOG"
echo "Ladder log:  $LADDER_LOG"
