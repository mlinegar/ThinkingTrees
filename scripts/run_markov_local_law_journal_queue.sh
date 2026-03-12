#!/usr/bin/env bash
set -euo pipefail

# Queue the next journal-oriented Markov local-law sweeps.
#
# The queue waits for any currently running Markov jobs to finish, then launches:
# 1. An audit-budget / sample-efficiency frontier on all visible MIG GPUs.
# 2. A root-vs-theorem Pareto sweep on all visible MIG GPUs.
# 3. A schedule-regularization interaction sweep on all visible MIG GPUs.
# 4. A focused CPU-only capacity companion in parallel with the first GPU phase.
#
# Example:
#   WAIT_PIDS="3608806 3962709" ./scripts/run_markov_local_law_journal_queue.sh

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_local_law_journal_suite_${STAMP}}"
LOG_DIR="${LOG_DIR:-logs}"
MASTER_LOG="${LOG_DIR}/${STAMP}_markov_local_law_journal_queue.log"

WAIT_PIDS="${WAIT_PIDS:-}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-120}"

RUN_AUDIT_FRONTIER="${RUN_AUDIT_FRONTIER:-1}"
RUN_PARETO="${RUN_PARETO:-1}"
RUN_SCHEDULE="${RUN_SCHEDULE:-1}"
RUN_CAPACITY_CPU="${RUN_CAPACITY_CPU:-1}"

JOURNAL_DATA_SEEDS="${JOURNAL_DATA_SEEDS:-0 1 2}"
JOURNAL_MODEL_SEEDS="${JOURNAL_MODEL_SEEDS:-0 1 2}"
JOURNAL_TEST_DOCS="${JOURNAL_TEST_DOCS:-2048}"
JOURNAL_STATE_DIMS="${JOURNAL_STATE_DIMS:-64}"
JOURNAL_HIDDEN_DIM_MULTIPLIER="${JOURNAL_HIDDEN_DIM_MULTIPLIER:-4}"
JOURNAL_HIDDEN_DIM_MIN="${JOURNAL_HIDDEN_DIM_MIN:-128}"
JOURNAL_EPOCHS="${JOURNAL_EPOCHS:-20}"
CAPACITY_CPU_JOBS="${CAPACITY_CPU_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || grep -c '^processor' /proc/cpuinfo || echo 128)}"

AUDIT_FRONTIER_TRAIN_DOCS="${AUDIT_FRONTIER_TRAIN_DOCS:-128 256 512 1024 2048 4096}"
AUDIT_FRONTIER_AUDITS="${AUDIT_FRONTIER_AUDITS:-0.01 0.025 0.05 0.1 0.25 0.5 1.0}"
AUDIT_FRONTIER_LLW="${AUDIT_FRONTIER_LLW:-0 0.25 0.5 0.8 0.9 1.0}"
AUDIT_FRONTIER_SCW="${AUDIT_FRONTIER_SCW:-0.1}"

PARETO_TRAIN_DOCS="${PARETO_TRAIN_DOCS:-2048}"
PARETO_AUDITS="${PARETO_AUDITS:-0.1 1.0}"
PARETO_LLW="${PARETO_LLW:-0.1 0.25 0.5 0.8 0.9 1.0}"
PARETO_SCW="${PARETO_SCW:-0.1 0.2}"
PARETO_ROOT_WEIGHTS="${PARETO_ROOT_WEIGHTS:-0 0.1 0.25 0.5 1 2 4}"

SCHEDULE_TRAIN_DOCS="${SCHEDULE_TRAIN_DOCS:-2048}"
SCHEDULE_AUDITS="${SCHEDULE_AUDITS:-0.1 1.0}"
SCHEDULE_LLW="${SCHEDULE_LLW:-0 0.25 0.5 0.8 0.9 1.0}"
SCHEDULE_SCW="${SCHEDULE_SCW:-0 0.025 0.05 0.1 0.2 0.4}"

CAPACITY_CPU_TRAIN_DOCS="${CAPACITY_CPU_TRAIN_DOCS:-2048}"
CAPACITY_CPU_AUDITS="${CAPACITY_CPU_AUDITS:-0.1 1.0}"
CAPACITY_CPU_LLW="${CAPACITY_CPU_LLW:-0.5 0.8 0.9 1.0}"
CAPACITY_CPU_SCW="${CAPACITY_CPU_SCW:-0.1 0.2}"
CAPACITY_CPU_STATE_DIMS="${CAPACITY_CPU_STATE_DIMS:-32 64 128}"
CAPACITY_CPU_DATA_SEEDS="${CAPACITY_CPU_DATA_SEEDS:-0 1}"
CAPACITY_CPU_MODEL_SEEDS="${CAPACITY_CPU_MODEL_SEEDS:-0 1}"
CAPACITY_CPU_EPOCHS="${CAPACITY_CPU_EPOCHS:-24}"

mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

timestamp_utc() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

log() {
  echo "[$(timestamp_utc)] $*" | tee -a "${MASTER_LOG}"
}

wait_for_pids() {
  if [[ -z "${WAIT_PIDS}" ]]; then
    return 0
  fi
  local -a tracked
  local -a alive
  read -r -a tracked <<<"${WAIT_PIDS}"
  while true; do
    alive=()
    for pid in "${tracked[@]}"; do
      if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
        alive+=("${pid}")
      fi
    done
    if [[ "${#alive[@]}" -eq 0 ]]; then
      log "No tracked wait PIDs remain."
      return 0
    fi
    log "Waiting on prior jobs: ${alive[*]}"
    sleep "${WAIT_POLL_SECONDS}"
  done
}

run_longrun_phase() {
  local phase_name="$1"
  shift
  local phase_log="${LOG_DIR}/${STAMP}_${phase_name}.log"
  log "START ${phase_name} | phase_log=${phase_log}"
  env \
    STAMP="${STAMP}_${phase_name}" \
    OUT_ROOT="${OUT_ROOT}/${phase_name}" \
    LOG_DIR="${LOG_DIR}" \
    "$@" \
    bash ./scripts/run_markov_local_law_learnability_longrun.sh >>"${phase_log}" 2>&1
  log "DONE ${phase_name}"
}

launch_capacity_cpu_bg() {
  local phase_name="capacity_cpu"
  local phase_log="${LOG_DIR}/${STAMP}_${phase_name}.log"
  log "START ${phase_name} in background | phase_log=${phase_log}"
  (
    env \
      STAMP="${STAMP}_${phase_name}" \
      OUT_ROOT="${OUT_ROOT}/${phase_name}" \
      LOG_DIR="${LOG_DIR}" \
      MARKOV_DEVICE="cpu" \
      MARKOV_JOBS="${CAPACITY_CPU_JOBS}" \
      MARKOV_TEST_DOCS="${JOURNAL_TEST_DOCS}" \
      MARKOV_TRAIN_DOCS="${CAPACITY_CPU_TRAIN_DOCS}" \
      MARKOV_AUDIT_FRACTIONS="${CAPACITY_CPU_AUDITS}" \
      MARKOV_LOCAL_LAW_WEIGHTS="${CAPACITY_CPU_LLW}" \
      MARKOV_SCW_GRID="${CAPACITY_CPU_SCW}" \
      MARKOV_ROOT_WEIGHTS="1.0" \
      MARKOV_DATA_SEEDS="${CAPACITY_CPU_DATA_SEEDS}" \
      MARKOV_MODEL_SEEDS="${CAPACITY_CPU_MODEL_SEEDS}" \
      MARKOV_STATE_DIMS="${CAPACITY_CPU_STATE_DIMS}" \
      MARKOV_HIDDEN_DIM_MULTIPLIER="${JOURNAL_HIDDEN_DIM_MULTIPLIER}" \
      MARKOV_HIDDEN_DIM_MIN="${JOURNAL_HIDDEN_DIM_MIN}" \
      MARKOV_EPOCHS="${CAPACITY_CPU_EPOCHS}" \
      bash ./scripts/run_markov_local_law_learnability_longrun.sh >>"${phase_log}" 2>&1
  ) &
  CAPACITY_CPU_PID=$!
  log "capacity_cpu pid=${CAPACITY_CPU_PID}"
}

CAPACITY_CPU_PID=""

log "OUT_ROOT=${OUT_ROOT}"
log "WAIT_PIDS=${WAIT_PIDS:-<none>}"
log "RUN_AUDIT_FRONTIER=${RUN_AUDIT_FRONTIER}"
log "RUN_PARETO=${RUN_PARETO}"
log "RUN_SCHEDULE=${RUN_SCHEDULE}"
log "RUN_CAPACITY_CPU=${RUN_CAPACITY_CPU}"
log "CAPACITY_CPU_JOBS=${CAPACITY_CPU_JOBS}"
log "JOURNAL_DATA_SEEDS=${JOURNAL_DATA_SEEDS}"
log "JOURNAL_MODEL_SEEDS=${JOURNAL_MODEL_SEEDS}"

if [[ "${RUN_CAPACITY_CPU}" == "1" ]]; then
  launch_capacity_cpu_bg
fi

wait_for_pids

if [[ "${RUN_AUDIT_FRONTIER}" == "1" ]]; then
  run_longrun_phase \
    "audit_frontier" \
    MARKOV_DEVICE="auto" \
    MARKOV_TEST_DOCS="${JOURNAL_TEST_DOCS}" \
    MARKOV_TRAIN_DOCS="${AUDIT_FRONTIER_TRAIN_DOCS}" \
    MARKOV_AUDIT_FRACTIONS="${AUDIT_FRONTIER_AUDITS}" \
    MARKOV_LOCAL_LAW_WEIGHTS="${AUDIT_FRONTIER_LLW}" \
    MARKOV_SCW_GRID="${AUDIT_FRONTIER_SCW}" \
    MARKOV_ROOT_WEIGHTS="1.0" \
    MARKOV_DATA_SEEDS="${JOURNAL_DATA_SEEDS}" \
    MARKOV_MODEL_SEEDS="${JOURNAL_MODEL_SEEDS}" \
    MARKOV_STATE_DIMS="${JOURNAL_STATE_DIMS}" \
    MARKOV_HIDDEN_DIM_MULTIPLIER="${JOURNAL_HIDDEN_DIM_MULTIPLIER}" \
    MARKOV_HIDDEN_DIM_MIN="${JOURNAL_HIDDEN_DIM_MIN}" \
    MARKOV_EPOCHS="${JOURNAL_EPOCHS}"
fi

if [[ "${RUN_PARETO}" == "1" ]]; then
  run_longrun_phase \
    "pareto" \
    MARKOV_DEVICE="auto" \
    MARKOV_TEST_DOCS="${JOURNAL_TEST_DOCS}" \
    MARKOV_TRAIN_DOCS="${PARETO_TRAIN_DOCS}" \
    MARKOV_AUDIT_FRACTIONS="${PARETO_AUDITS}" \
    MARKOV_LOCAL_LAW_WEIGHTS="${PARETO_LLW}" \
    MARKOV_SCW_GRID="${PARETO_SCW}" \
    MARKOV_ROOT_WEIGHTS="${PARETO_ROOT_WEIGHTS}" \
    MARKOV_DATA_SEEDS="${JOURNAL_DATA_SEEDS}" \
    MARKOV_MODEL_SEEDS="${JOURNAL_MODEL_SEEDS}" \
    MARKOV_STATE_DIMS="${JOURNAL_STATE_DIMS}" \
    MARKOV_HIDDEN_DIM_MULTIPLIER="${JOURNAL_HIDDEN_DIM_MULTIPLIER}" \
    MARKOV_HIDDEN_DIM_MIN="${JOURNAL_HIDDEN_DIM_MIN}" \
    MARKOV_EPOCHS="${JOURNAL_EPOCHS}"
fi

if [[ "${RUN_SCHEDULE}" == "1" ]]; then
  run_longrun_phase \
    "schedule_interaction" \
    MARKOV_DEVICE="auto" \
    MARKOV_TEST_DOCS="${JOURNAL_TEST_DOCS}" \
    MARKOV_TRAIN_DOCS="${SCHEDULE_TRAIN_DOCS}" \
    MARKOV_AUDIT_FRACTIONS="${SCHEDULE_AUDITS}" \
    MARKOV_LOCAL_LAW_WEIGHTS="${SCHEDULE_LLW}" \
    MARKOV_SCW_GRID="${SCHEDULE_SCW}" \
    MARKOV_ROOT_WEIGHTS="1.0" \
    MARKOV_DATA_SEEDS="${JOURNAL_DATA_SEEDS}" \
    MARKOV_MODEL_SEEDS="${JOURNAL_MODEL_SEEDS}" \
    MARKOV_STATE_DIMS="${JOURNAL_STATE_DIMS}" \
    MARKOV_HIDDEN_DIM_MULTIPLIER="${JOURNAL_HIDDEN_DIM_MULTIPLIER}" \
    MARKOV_HIDDEN_DIM_MIN="${JOURNAL_HIDDEN_DIM_MIN}" \
    MARKOV_EPOCHS="${JOURNAL_EPOCHS}"
fi

if [[ -n "${CAPACITY_CPU_PID}" ]]; then
  log "Waiting for capacity_cpu pid=${CAPACITY_CPU_PID}"
  wait "${CAPACITY_CPU_PID}"
  log "DONE capacity_cpu"
fi

log "DONE journal queue | OUT_ROOT=${OUT_ROOT}"
