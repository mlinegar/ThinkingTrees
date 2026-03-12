#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-outputs/exact_utility_transport_targeted_round2_20260306}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"

FREE4_MIGS="${FREE4_MIGS:-MIG-a0ff6aee-294a-5e13-8ccd-4fe32c421500 MIG-73b2c88b-25a2-5660-9c43-ea301ecfa3bf MIG-ed1343e7-7104-5739-87c8-6c80add3514f MIG-ee618671-a644-5598-ad47-9e9ec0c928f0}"
WAIT12_MIGS="${WAIT12_MIGS:-MIG-c304e34d-66fc-5d15-af50-4db15f270b34 MIG-fa544c03-e7a6-5573-a20f-13a22b2cda58 MIG-a66c8334-06ca-55ee-978a-ff8c01447dcd MIG-8f438dd0-fe62-5f8f-849f-304a5aec1acc MIG-31e81ccd-c7c3-56d4-8375-f2fc4293a042 MIG-82d169c9-8b93-5c11-889c-b32effd10b5e MIG-089e2acd-ac49-5cb5-8467-4718f2d51591 MIG-95802abc-9460-53fc-86ca-382324bea437 MIG-074b2b2a-b8ff-5a41-8f44-8293b7f28f6e MIG-71997662-977d-56ca-b0fc-b327f443632d MIG-84bbcb3d-1ced-5a9b-acc6-9087ba00fdcd MIG-4b3f74c5-11d4-5e09-9fdf-4023f7366988}"

mkdir -p "${LOG_ROOT}"

FREE4_CMD_FILE="${OUT_ROOT}/cmds/gpu_cmds_free4.txt"
WAIT12_CMD_FILE="${OUT_ROOT}/cmds/gpu_cmds_wait12.txt"

FREE4_QUEUE_LOG="${LOG_ROOT}/free4_queue_resume_${STAMP}.log"
WAIT12_QUEUE_LOG="${LOG_ROOT}/wait12_queue_resume_${STAMP}.log"

FREE4_WORKER_LOG_DIR="${OUT_ROOT}/_launcher_logs_free4_resume_${STAMP}"
WAIT12_WORKER_LOG_DIR="${OUT_ROOT}/_launcher_logs_wait12_resume_${STAMP}"

venv/bin/python scripts/run_mig_command_queue.py \
  --cmd-file "${FREE4_CMD_FILE}" \
  --log-dir "${FREE4_WORKER_LOG_DIR}" \
  --mig-uuids "${FREE4_MIGS}" \
  >"${FREE4_QUEUE_LOG}" 2>&1 &
FREE4_PID=$!

venv/bin/python scripts/run_mig_command_queue.py \
  --cmd-file "${WAIT12_CMD_FILE}" \
  --log-dir "${WAIT12_WORKER_LOG_DIR}" \
  --mig-uuids "${WAIT12_MIGS}" \
  >"${WAIT12_QUEUE_LOG}" 2>&1 &
WAIT12_PID=$!

printf '%s free4_pid=%s wait12_pid=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${FREE4_PID}" "${WAIT12_PID}"
printf '%s free4_log=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${FREE4_QUEUE_LOG}"
printf '%s wait12_log=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${WAIT12_QUEUE_LOG}"

wait "${FREE4_PID}" "${WAIT12_PID}"
