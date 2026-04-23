#!/usr/bin/env bash
set -euo pipefail

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_capability_suites_${STAMP}}"
LOG_DIR="${LOG_DIR:-logs}"
SUITE="${SUITE:-all}"
DEVICE="${DEVICE:-cpu}"
TORCH_THREADS="${TORCH_THREADS:-1}"
PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
CMD_DIR="${CMD_DIR:-${LOG_DIR}/markov_capability_suites_${STAMP}}"
TRANSITION_SUMMARY="${TRANSITION_SUMMARY:-}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}" "${CMD_DIR}"

run_cmd_file() {
  local cmd_file="$1"
  local jobs="${JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 1)}"
  if [[ ! -f "${cmd_file}" ]]; then
    return 0
  fi
  cat "${cmd_file}" | xargs -d $'\n' -P "${jobs}" -I {} bash -lc "{}"
}

build_suite() {
  local suite_name="$1"
  local extra_args=()
  if [[ "${suite_name}" == "mechanism_suite" ]]; then
    extra_args+=(--transition-summary "${TRANSITION_SUMMARY}")
  fi
  "${PYTHON_BIN}" -u scripts/build_markov_capability_suite_cmds.py \
    --suite "${suite_name}" \
    --output-root "${OUT_ROOT}" \
    --cmd-dir "${CMD_DIR}" \
    --python-bin "${PYTHON_BIN}" \
    --device "${DEVICE}" \
    --torch-threads "${TORCH_THREADS}" \
    "${extra_args[@]}"
}

report_suite() {
  local suite_name="$1"
  local input_root="${OUT_ROOT}/${suite_name}/markov_changepoint_ops_count"
  local report_dir="${input_root}/capability_report"
  if [[ ! -d "${input_root}" ]]; then
    return 0
  fi
  echo "SKIP report for ${suite_name}; capability report is archived (see docs/markov_report_archive.md)"
}

if [[ "${SUITE}" == "sanity_suite" || "${SUITE}" == "all" ]]; then
  build_suite sanity_suite
  run_cmd_file "${CMD_DIR}/sanity_suite_cmds.txt"
  report_suite sanity_suite
fi

if [[ "${SUITE}" == "transition_map_suite" || "${SUITE}" == "all" ]]; then
  build_suite transition_map_suite
  run_cmd_file "${CMD_DIR}/transition_map_suite_cmds.txt"
  report_suite transition_map_suite
  TRANSITION_SUMMARY="${OUT_ROOT}/transition_map_suite/markov_changepoint_ops_count/capability_report/markov_capability_summary.json"
fi

if [[ "${SUITE}" == "mechanism_suite" || "${SUITE}" == "all" ]]; then
  if [[ -z "${TRANSITION_SUMMARY}" ]]; then
    echo "TRANSITION_SUMMARY is required for mechanism_suite" >&2
    exit 1
  fi
  build_suite mechanism_suite
  run_cmd_file "${CMD_DIR}/mechanism_suite_cmds.txt"
  report_suite mechanism_suite
fi

echo "DONE | OUT_ROOT=${OUT_ROOT}"
