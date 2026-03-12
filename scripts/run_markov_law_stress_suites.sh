#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source venv/bin/activate

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/markov_law_stress_${STAMP}}"
CMD_DIR="${CMD_DIR:-logs/markov_law_stress_${STAMP}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
CUDA_DEVICE="${CUDA_DEVICE:-}"
TORCH_THREADS="${TORCH_THREADS:-1}"
JOBS="${JOBS:-4}"
SMOKE_FLAG="${SMOKE_FLAG:-0}"

mkdir -p "$OUTPUT_ROOT" "$CMD_DIR"

COMMON_BUILD_ARGS=(
  --output-root "$OUTPUT_ROOT"
  --cmd-dir "$CMD_DIR"
  --python-bin "$PYTHON_BIN"
  --device "$DEVICE"
  --torch-threads "$TORCH_THREADS"
)
if [[ -n "$CUDA_DEVICE" ]]; then
  COMMON_BUILD_ARGS+=(--cuda-device "$CUDA_DEVICE")
fi
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
    raise SystemExit(f"{len(bad)} command(s) failed for {cmd_file}")
PY
}

"$PYTHON_BIN" scripts/build_markov_law_stress_suite_cmds.py --suite sanity_suite "${COMMON_BUILD_ARGS[@]}"
SANITY_LEARNED="$CMD_DIR/sanity_suite_learned_cmds.txt"
SANITY_EXACT="$CMD_DIR/sanity_suite_exact_cmds.txt"
run_cmd_file "$SANITY_LEARNED" "$CMD_DIR/sanity_learned_logs"
run_cmd_file "$SANITY_EXACT" "$CMD_DIR/sanity_exact_logs"
"$PYTHON_BIN" scripts/report_markov_law_stress.py \
  --input-root "$OUTPUT_ROOT/sanity_suite/markov_changepoint_ops_count" \
  --output-dir "$OUTPUT_ROOT/sanity_suite/markov_changepoint_ops_count/law_stress_report" \
  --suite-type sanity_suite

"$PYTHON_BIN" scripts/build_markov_law_stress_suite_cmds.py --suite transition_map_suite "${COMMON_BUILD_ARGS[@]}"
TRANSITION_CMDS="$CMD_DIR/transition_map_suite_cmds.txt"
run_cmd_file "$TRANSITION_CMDS" "$CMD_DIR/transition_logs"
"$PYTHON_BIN" scripts/report_markov_law_stress.py \
  --input-root "$OUTPUT_ROOT/transition_map_suite/markov_changepoint_ops_count" \
  --output-dir "$OUTPUT_ROOT/transition_map_suite/markov_changepoint_ops_count/law_stress_report" \
  --suite-type transition_map_suite

TRANSITION_SUMMARY="$OUTPUT_ROOT/transition_map_suite/markov_changepoint_ops_count/law_stress_report/markov_law_stress_summary.json"
"$PYTHON_BIN" scripts/build_markov_law_stress_suite_cmds.py \
  --suite mechanism_suite \
  --transition-summary "$TRANSITION_SUMMARY" \
  "${COMMON_BUILD_ARGS[@]}"
MECHANISM_CMDS="$CMD_DIR/mechanism_suite_cmds.txt"
run_cmd_file "$MECHANISM_CMDS" "$CMD_DIR/mechanism_logs"
"$PYTHON_BIN" scripts/report_markov_law_stress.py \
  --input-root "$OUTPUT_ROOT/mechanism_suite/markov_changepoint_ops_count" \
  --output-dir "$OUTPUT_ROOT/mechanism_suite/markov_changepoint_ops_count/law_stress_report" \
  --suite-type mechanism_suite

"$PYTHON_BIN" scripts/build_markov_law_stress_suite_cmds.py --suite capacity_appendix_suite "${COMMON_BUILD_ARGS[@]}"
CAPACITY_CMDS="$CMD_DIR/capacity_appendix_suite_cmds.txt"
run_cmd_file "$CAPACITY_CMDS" "$CMD_DIR/capacity_logs"
"$PYTHON_BIN" scripts/report_markov_law_stress.py \
  --input-root "$OUTPUT_ROOT/capacity_appendix_suite/markov_changepoint_ops_count" \
  --output-dir "$OUTPUT_ROOT/capacity_appendix_suite/markov_changepoint_ops_count/law_stress_report" \
  --suite-type capacity_appendix_suite

echo "{\"output_root\":\"$OUTPUT_ROOT\"}"
