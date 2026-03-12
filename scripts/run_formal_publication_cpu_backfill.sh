#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

FORMAL_ROOT=""
JOBS="128"
LOG_DIR=""
WAIT_PIDS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_formal_publication_cpu_backfill.sh --formal-root PATH [options]

Options:
  --formal-root PATH   Formal rerun root to backfill.
  --jobs N             CPU worker count for each suite. Default: 128.
  --log-dir PATH       Log directory. Default: <formal-root>/paper_reports/logs.
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
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
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

mkdir -p "${LOG_DIR}" "${FORMAL_ROOT}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

CONTROLLER_LOG="${LOG_DIR}/formal_publication_cpu_backfill.controller.log"

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
  local log_path="$2"
  shift 2
  log "suite_start name=${name} log=${log_path}"
  (
    set -euo pipefail
    "$@"
  ) >>"${log_path}" 2>&1
  log "suite_done name=${name}"
}

refresh_bundle() {
  local tag="$1"
  log "bundle_refresh_start tag=${tag}"
  venv/bin/python scripts/generate_paper_simulation_report_bundle.py --formal-root "${FORMAL_ROOT}" \
    >>"${LOG_DIR}/formal_publication_cpu_backfill.bundle.log" 2>&1
  log "bundle_refresh_done tag=${tag}"
}

run_report() {
  local tag="$1"
  shift
  log "report_start tag=${tag}"
  (
    set -euo pipefail
    "$@"
  ) >>"${LOG_DIR}/formal_publication_cpu_backfill.reports.log" 2>&1
  log "report_done tag=${tag}"
}

log "controller_start formal_root=${FORMAL_ROOT} jobs=${JOBS} wait_pids=${WAIT_PIDS[*]:-}"
wait_for_pids
log "wait_complete"

run_suite \
  "lda_leafnoise" \
  "${LOG_DIR}/identifiable_zero_lda_leafnoise.backfill.log" \
  env JOBS="${JOBS}" GPU_TOKENS="none" OUT_ROOT="${FORMAL_ROOT}/identifiable_zero_lda_leafnoise" \
    bash ./scripts/run_identifiable_zero_lda_leafnoise_overnight.sh
run_report \
  "lda_leafnoise" \
  venv/bin/python scripts/report_identifiable_zero_lda_leafnoise_progression.py \
    --output-root "${FORMAL_ROOT}/identifiable_zero_lda_leafnoise" \
    --emit-pdf
refresh_bundle "lda_leafnoise"

run_suite \
  "dtm_lda" \
  "${LOG_DIR}/identifiable_zero_dtm_lda.backfill.log" \
  env JOBS="${JOBS}" GPU_TOKENS="none" OUT_ROOT="${FORMAL_ROOT}/identifiable_zero_dtm_lda" \
    bash ./scripts/run_identifiable_zero_dtm_lda_overnight.sh
run_report \
  "dtm_lda" \
  venv/bin/python scripts/report_identifiable_zero_dtm_lda.py \
    --output-root "${FORMAL_ROOT}/identifiable_zero_dtm_lda" \
    --emit-pdf
refresh_bundle "dtm_lda"

run_suite \
  "learnability" \
  "${LOG_DIR}/identifiable_zero_learnability.backfill.log" \
  env JOBS="${JOBS}" GPU_TOKENS="none" MARKOV_DEVICE="cpu" OUT_ROOT="${FORMAL_ROOT}/identifiable_zero_learnability" \
    bash ./scripts/run_identifiable_zero_learnability_overnight.sh
run_report \
  "learnability" \
  venv/bin/python scripts/report_identifiable_zero_learnability.py \
    --output-root "${FORMAL_ROOT}/identifiable_zero_learnability" \
    --emit-pdf
refresh_bundle "learnability"

log "controller_done"
