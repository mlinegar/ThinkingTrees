#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FORMAL_ROOT=""
JOBS="128"
LOG_DIR=""
MIG_UUIDS=""

usage() {
  cat <<'EOF'
Usage: scripts/launch_formal_publication_backfill_detached.sh --formal-root PATH [options]

Options:
  --formal-root PATH   Formal rerun root to backfill.
  --jobs N             CPU worker count. Default: 128.
  --log-dir PATH       Log directory. Default: <formal-root>/paper_reports/logs.
  --mig-uuids TEXT     Space/comma-separated MIG UUIDs. Default: auto-discover all MIG UUIDs.
  -h, --help           Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --formal-root)
      FORMAL_ROOT="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
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

mkdir -p "${LOG_DIR}" "${FORMAL_ROOT}"

LAUNCH_LOG="${LOG_DIR}/formal_publication_backfill_detached.launch.log"
CPU_PID_FILE="${LOG_DIR}/formal_publication_backfill_cpu.pid"
MIG_PID_FILE="${LOG_DIR}/formal_publication_backfill_mig.pid"
CPU_SESSION_FILE="${LOG_DIR}/formal_publication_backfill_cpu.session"
MIG_SESSION_FILE="${LOG_DIR}/formal_publication_backfill_mig.session"
TMUX_AVAILABLE="0"
if command -v tmux >/dev/null 2>&1; then
  TMUX_AVAILABLE="1"
fi

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LAUNCH_LOG}" >&2
}

pid_is_live() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

tmux_session_live() {
  local session_name="$1"
  [[ -n "${session_name}" ]] && tmux has-session -t "${session_name}" 2>/dev/null
}

launch_tmux_job() {
  local name="$1"
  local session_file="$2"
  local pid_file="$3"
  local log_file="$4"
  shift 4
  local session_name="formal_publication_${name}"
  local job_script="${LOG_DIR}/${name}.job.sh"
  local existing_session=""
  if [[ -f "${session_file}" ]]; then
    existing_session="$(tr -d '[:space:]' < "${session_file}")"
  fi
  if tmux_session_live "${existing_session}"; then
    local live_pid=""
    live_pid="$(tmux display-message -p -t "${existing_session}:0" '#{pane_pid}' 2>/dev/null || true)"
    if [[ -n "${live_pid}" ]]; then
      echo "${live_pid}" > "${pid_file}"
    fi
    log "already_running name=${name} session=${existing_session} pid=${live_pid:-unknown} log=${log_file}"
    printf '%s %s\n' "${existing_session}" "${live_pid:-}"
    return 0
  fi
  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "${REPO_ROOT}"
    printf 'exec '
    printf '%q ' "$@"
    printf '\n'
  } > "${job_script}"
  chmod +x "${job_script}"
  tmux kill-session -t "${session_name}" 2>/dev/null || true
  tmux new-session -d -s "${session_name}" "cd ${REPO_ROOT@Q} && bash ${job_script@Q} 2>&1 | tee ${log_file@Q}"
  local pane_pid=""
  pane_pid="$(tmux display-message -p -t "${session_name}:0" '#{pane_pid}' 2>/dev/null || true)"
  echo "${session_name}" > "${session_file}"
  if [[ -n "${pane_pid}" ]]; then
    echo "${pane_pid}" > "${pid_file}"
  fi
  log "launched name=${name} session=${session_name} pid=${pane_pid:-unknown} log=${log_file} job_script=${job_script}"
  printf '%s %s\n' "${session_name}" "${pane_pid:-}"
}

launch_nohup_job() {
  local name="$1"
  local _session_file="$2"
  local pid_file="$3"
  local log_file="$4"
  shift 4
  local job_script="${LOG_DIR}/${name}.job.sh"
  local existing_pid=""
  if [[ -f "${pid_file}" ]]; then
    existing_pid="$(tr -d '[:space:]' < "${pid_file}")"
  fi
  if pid_is_live "${existing_pid}"; then
    log "already_running name=${name} pid=${existing_pid} log=${log_file}"
    echo "${existing_pid}"
    return 0
  fi
  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "${REPO_ROOT}"
    printf 'exec '
    printf '%q ' "$@"
    printf '\n'
  } > "${job_script}"
  chmod +x "${job_script}"
  nohup bash "${job_script}" >"${log_file}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${pid_file}"
  log "launched name=${name} pid=${pid} log=${log_file} job_script=${job_script}"
  echo "${pid}"
}

launch_job() {
  if [[ "${TMUX_AVAILABLE}" == "1" ]]; then
    launch_tmux_job "$@"
  else
    launch_nohup_job "$@"
  fi
}

log "launcher_start formal_root=${FORMAL_ROOT} jobs=${JOBS}"

CPU_LAUNCH_INFO="$(launch_job \
  cpu_backfill \
  "${CPU_SESSION_FILE}" \
  "${CPU_PID_FILE}" \
  "${LOG_DIR}/formal_publication_backfill_cpu.stdout.log" \
  bash ./scripts/run_formal_publication_recovery_cpu_job.sh \
    --formal-root "${FORMAL_ROOT}" \
    --jobs "${JOBS}")"

if [[ -n "${MIG_UUIDS}" ]]; then
  MIG_LAUNCH_INFO="$(launch_job \
    mig_backfill \
    "${MIG_SESSION_FILE}" \
    "${MIG_PID_FILE}" \
    "${LOG_DIR}/formal_publication_backfill_mig.stdout.log" \
    bash ./scripts/run_formal_publication_recovery_mig_job.sh \
      --formal-root "${FORMAL_ROOT}" \
      --log-dir "${LOG_DIR}" \
      --mig-uuids "${MIG_UUIDS}")"
else
  MIG_LAUNCH_INFO=""
  log "skip_mig_backfill reason=no_mig_uuids"
fi

CPU_SESSION="$(awk '{print $1}' <<<"${CPU_LAUNCH_INFO}")"
CPU_PID="$(awk '{print $2}' <<<"${CPU_LAUNCH_INFO}")"
MIG_SESSION="$(awk '{print $1}' <<<"${MIG_LAUNCH_INFO}")"
MIG_PID="$(awk '{print $2}' <<<"${MIG_LAUNCH_INFO}")"
log "launcher_done cpu_session=${CPU_SESSION} cpu_pid=${CPU_PID} mig_session=${MIG_SESSION} mig_pid=${MIG_PID}"
