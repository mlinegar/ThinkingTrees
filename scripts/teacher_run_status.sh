#!/usr/bin/env bash
# Quick status check for the size-based teacher trace generation.
# Usage: bash scripts/teacher_run_status.sh [run_dir]
set -u

RUN_DIR="${1:-outputs/manifesto_teacher_fg_size_grid/economic_gemma4_size0512tok}"
LEAF_DIR=$(find "$RUN_DIR" -maxdepth 1 -mindepth 1 -type d -name 'leaf*tok' | head -1)
LOG="${TEACHER_LOG:-/tmp/teacher_overnight.log}"
PID_FILE="${TEACHER_PID:-/tmp/teacher_overnight.pid}"

echo "=== run dir: $RUN_DIR ==="
if [[ -n "$LEAF_DIR" ]]; then
    echo "=== leaf dir: $LEAF_DIR ==="
    for f in teacher_g_summary_cache.jsonl teacher_f_score_cache.jsonl teacher_g_resummary_cache.jsonl; do
        path="$LEAF_DIR/$f"
        if [[ -f "$path" ]]; then
            n=$(wc -l < "$path")
            mtime=$(stat -c '%y' "$path" 2>/dev/null | cut -d. -f1)
            echo "  $f: $n entries (last write: $mtime)"
        fi
    done
    if [[ -f "$LEAF_DIR/labeled_trees.jsonl" ]]; then
        echo "  labeled_trees.jsonl: $(wc -l < "$LEAF_DIR/labeled_trees.jsonl") trees (FINAL OUTPUT)"
    fi
fi

echo
echo "=== process ==="
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        ps -p "$PID" -o pid,etime,%cpu,rss,cmd 2>&1 | head -3
    else
        echo "PID $PID is NOT running."
        echo "Last log lines:"
        tail -5 "$LOG"
    fi
fi

echo
echo "=== last log lines ==="
tail -5 "$LOG"
