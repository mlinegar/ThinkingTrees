#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

FORMAL_ROOT=""
LOG_DIR=""
MIG_UUIDS=""
WAIT_PIDS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_formal_publication_mig_backfill.sh --formal-root PATH [options]

Options:
  --formal-root PATH   Formal rerun root to backfill.
  --log-dir PATH       Log directory. Default: <formal-root>/paper_reports/logs.
  --mig-uuids TEXT     Space/comma-separated MIG UUIDs. Default: auto-discover all MIG UUIDs.
  --wait-pid PID       Wait for this PID to exit before starting work. Repeat as needed.
  -h, --help           Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --formal-root)
      FORMAL_ROOT="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --mig-uuids)
      MIG_UUIDS="$2"
      shift 2
      ;;
    --wait-pid)
      WAIT_PIDS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${FORMAL_ROOT}" ]]; then
  echo "Missing --formal-root" >&2
  usage >&2
  exit 2
fi

if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="${FORMAL_ROOT}/paper_reports/logs"
fi

discover_migs() {
  nvidia-smi -L 2>/dev/null | sed -n '/MIG /s/.*(UUID: \([^)]*\)).*/\1/p' | paste -sd' ' -
}

if [[ -z "${MIG_UUIDS}" ]]; then
  MIG_UUIDS="$(discover_migs)"
fi

if [[ -z "${MIG_UUIDS}" ]]; then
  echo "No MIG UUIDs discovered; pass --mig-uuids explicitly." >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${FORMAL_ROOT}"

CONTROLLER_LOG="${LOG_DIR}/formal_publication_mig_backfill.controller.log"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${CONTROLLER_LOG}"
}

trap 'log "controller_error exit=$? line=${LINENO} cmd=${BASH_COMMAND}"' ERR

wait_for_pids() {
  if [[ "${#WAIT_PIDS[@]}" -eq 0 ]]; then
    return 0
  fi
  while true; do
    local active=0
    for pid in "${WAIT_PIDS[@]}"; do
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        active=1
      fi
    done
    if [[ "${active}" -eq 0 ]]; then
      return 0
    fi
    sleep 30
  done
}

run_suite() {
  local name="$1"
  local cmd_file="$2"
  local log_path="$3"
  local queue_log_dir="$4"

  log "suite_start name=${name} cmd_file=${cmd_file} log=${log_path}"
  venv/bin/python scripts/run_mig_command_queue.py \
    --cmd-file "${cmd_file}" \
    --log-dir "${queue_log_dir}" \
    --mig-uuids "${MIG_UUIDS}" \
    >>"${log_path}" 2>&1
  log "suite_done name=${name}"
}

run_report() {
  local tag="$1"
  shift
  log "report_start tag=${tag}"
  (
    set -euo pipefail
    "$@"
  ) >>"${LOG_DIR}/formal_publication_mig_backfill.reports.log" 2>&1
  log "report_done tag=${tag}"
}

log "controller_start formal_root=${FORMAL_ROOT} wait_pids=${WAIT_PIDS[*]:-} mig_count=$(wc -w <<<"${MIG_UUIDS}" | tr -d ' ')"
wait_for_pids
log "wait_complete"

log "suite_build_start name=publication_clean"
venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-publication build \
  --profile publication_clean \
  --output-root "${FORMAL_ROOT}/identifiable_zero_longrun_clean" \
  >>"${LOG_DIR}/formal_publication_mig_backfill.reports.log" 2>&1
log "suite_build_done name=publication_clean"

run_suite \
  "publication_clean_gpu" \
  "${FORMAL_ROOT}/identifiable_zero_longrun_clean/suite_groups/cmds/gpu.txt" \
  "${LOG_DIR}/identifiable_zero_longrun_clean_gpu.backfill.log" \
  "${LOG_DIR}/identifiable_zero_longrun_clean_gpu_queue_logs"

run_report \
  "publication_clean" \
  venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-publication report \
    --profile publication_clean \
    --output-root "${FORMAL_ROOT}/identifiable_zero_longrun_clean" \
    --emit-pdf

log "bundle_refresh_start"
venv/bin/python scripts/generate_paper_simulation_report_bundle.py --formal-root "${FORMAL_ROOT}" \
  >>"${LOG_DIR}/formal_publication_mig_backfill.bundle.log" 2>&1
log "bundle_refresh_done"

log "controller_done"
