#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source venv/bin/activate

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/lda_law_stress_${STAMP}}"
CMD_DIR="${CMD_DIR:-logs/lda_law_stress_${STAMP}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
JOBS="${JOBS:-4}"
SMOKE_FLAG="${SMOKE_FLAG:-0}"

mkdir -p "$OUTPUT_ROOT" "$CMD_DIR"

COMMON_BUILD_ARGS=(
  --output-root "$OUTPUT_ROOT"
  --cmd-dir "$CMD_DIR"
  --python-bin "$PYTHON_BIN"
)
if [[ "$SMOKE_FLAG" == "1" ]]; then
  COMMON_BUILD_ARGS+=(--smoke)
fi

run_cmd_file() {
  local cmd_file="$1"
  local log_dir="$2"
  "$PYTHON_BIN" - <<'PY' "$cmd_file" "$log_dir" "$JOBS"
from pathlib import Path
import sys

from src.ctreepo.sim.runner import read_cmds_file, run_commands

cmd_file = Path(sys.argv[1])
log_dir = Path(sys.argv[2])
jobs = int(sys.argv[3])
commands = read_cmds_file(cmd_file)
results = run_commands(commands, jobs=jobs, log_dir=log_dir, fail_fast=False)
bad = [r for r in results if int(r.returncode) != 0]
if bad:
    print(f"WARNING: {len(bad)} command(s) failed for {cmd_file}", file=sys.stderr)
PY
}

echo "=== LDA Law-Stress Suites ==="
echo "Output root: $OUTPUT_ROOT"
echo "Smoke: $SMOKE_FLAG"

# --- Sanity Suite ---
echo "--- Building sanity_suite ---"
"$PYTHON_BIN" scripts/build_lda_law_stress_suite_cmds.py --suite sanity_suite "${COMMON_BUILD_ARGS[@]}"
SANITY_FILE="$CMD_DIR/lda_law_stress_sanity_suite_cmds.txt"
echo "--- Running sanity_suite ---"
run_cmd_file "$SANITY_FILE" "$CMD_DIR/sanity_logs"
echo "--- Reporting sanity_suite ---"
"$PYTHON_BIN" scripts/report_tree_relevant_lda_local_law.py \
  --input-root "$OUTPUT_ROOT/sanity_suite" \
  --output-dir "$OUTPUT_ROOT/sanity_suite/law_stress_report" \
  --snapshot-label "sanity_suite"

# --- Transition Map Suite ---
echo "--- Building transition_map_suite ---"
"$PYTHON_BIN" scripts/build_lda_law_stress_suite_cmds.py --suite transition_map_suite "${COMMON_BUILD_ARGS[@]}"
TRANSITION_FILE="$CMD_DIR/lda_law_stress_transition_map_suite_cmds.txt"
echo "--- Running transition_map_suite ---"
run_cmd_file "$TRANSITION_FILE" "$CMD_DIR/transition_map_logs"
echo "--- Reporting transition_map_suite ---"
"$PYTHON_BIN" scripts/report_tree_relevant_lda_local_law.py \
  --input-root "$OUTPUT_ROOT/transition_map_suite" \
  --output-dir "$OUTPUT_ROOT/transition_map_suite/law_stress_report" \
  --snapshot-label "transition_map_suite"

# --- Mechanism Suite ---
echo "--- Building mechanism_suite ---"
"$PYTHON_BIN" scripts/build_lda_law_stress_suite_cmds.py --suite mechanism_suite "${COMMON_BUILD_ARGS[@]}"
MECHANISM_FILE="$CMD_DIR/lda_law_stress_mechanism_suite_cmds.txt"
echo "--- Running mechanism_suite ---"
run_cmd_file "$MECHANISM_FILE" "$CMD_DIR/mechanism_logs"
echo "--- Reporting mechanism_suite ---"
"$PYTHON_BIN" scripts/report_tree_relevant_lda_local_law.py \
  --input-root "$OUTPUT_ROOT/mechanism_suite" \
  --output-dir "$OUTPUT_ROOT/mechanism_suite/law_stress_report" \
  --snapshot-label "mechanism_suite"

echo "=== Done ==="
echo "Reports in:"
echo "  $OUTPUT_ROOT/sanity_suite/law_stress_report/"
echo "  $OUTPUT_ROOT/transition_map_suite/law_stress_report/"
echo "  $OUTPUT_ROOT/mechanism_suite/law_stress_report/"
