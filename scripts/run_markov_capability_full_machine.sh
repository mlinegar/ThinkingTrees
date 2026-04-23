#!/usr/bin/env bash
set -euo pipefail

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_capability_full_machine_${STAMP}}"
LOG_DIR="${LOG_DIR:-logs}"
PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
MASTER_LOG="${LOG_DIR}/${STAMP}_markov_capability_full_machine.log"
CMD_DIR="${CMD_DIR:-${LOG_DIR}/markov_capability_full_machine_${STAMP}}"

RUN_SANITY="${RUN_SANITY:-1}"
RUN_TRANSITION="${RUN_TRANSITION:-1}"
RUN_MECHANISM="${RUN_MECHANISM:-1}"
DRY_RUN="${DRY_RUN:-0}"

SANITY_CPU_JOBS="${SANITY_CPU_JOBS:-32}"
TRANSITION_CPU_JOBS="${TRANSITION_CPU_JOBS:-96}"
MECHANISM_CPU_JOBS="${MECHANISM_CPU_JOBS:-128}"

TRANSITION_CPU_SHARE_NUM="${TRANSITION_CPU_SHARE_NUM:-1}"
TRANSITION_SHARE_DEN="${TRANSITION_SHARE_DEN:-4}"
MECHANISM_CPU_SHARE_NUM="${MECHANISM_CPU_SHARE_NUM:-1}"
MECHANISM_SHARE_DEN="${MECHANISM_SHARE_DEN:-2}"

MECHANISM_CELLS="${MECHANISM_CELLS:-4}"
MIG_UUIDS="${MIG_UUIDS:-}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}" "${CMD_DIR}"

timestamp_utc() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

log() {
  echo "[$(timestamp_utc)] $*" | tee -a "${MASTER_LOG}"
}

detect_mig_uuids() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  nvidia-smi -L 2>/dev/null | grep -o 'MIG-[A-Za-z0-9-]*' | awk '!seen[$0]++' | paste -sd' ' -
}

run_xargs_cmd_file() {
  local cmd_file="$1"
  local jobs="$2"
  local phase_log="$3"
  if [[ ! -s "${cmd_file}" ]]; then
    log "SKIP empty cmd file ${cmd_file}"
    return 0
  fi
  cat "${cmd_file}" | xargs -d $'\n' -P "${jobs}" -I {} bash -lc "{}" >>"${phase_log}" 2>&1
}

run_mig_cmd_file() {
  local cmd_file="$1"
  local phase_log="$2"
  if [[ ! -s "${cmd_file}" ]]; then
    log "SKIP empty cmd file ${cmd_file}"
    return 0
  fi
  "${PYTHON_BIN}" -u scripts/run_mig_command_queue.py \
    --cmd-file "${cmd_file}" \
    --log-dir "${cmd_file%.txt}_mig_logs" \
    --mig-uuids "${MIG_UUIDS}" \
    --append-cuda-device-zero >>"${phase_log}" 2>&1
}

build_suite() {
  local suite_name="$1"
  shift
  local phase_log="$1"
  shift
  log "BUILD ${suite_name}"
  "${PYTHON_BIN}" -u scripts/build_markov_capability_suite_cmds.py \
    --suite "${suite_name}" \
    --output-root "${OUT_ROOT}" \
    --cmd-dir "${CMD_DIR}" \
    --python-bin "${PYTHON_BIN}" \
    "$@" >>"${phase_log}" 2>&1
}

report_suite() {
  local suite_name="$1"
  local phase_log="$2"
  local input_root="${OUT_ROOT}/${suite_name}/markov_changepoint_ops_count"
  local report_dir="${input_root}/capability_report"
  if [[ ! -d "${input_root}" ]]; then
    log "SKIP report for ${suite_name}; missing input root ${input_root}"
    return 0
  fi
  log "SKIP report for ${suite_name}; capability report is archived (see docs/markov_report_archive.md)"
}

split_cmd_file() {
  local src="$1"
  local cpu_dst="$2"
  local gpu_dst="$3"
  local cpu_num="$4"
  local den="$5"
  local phase_log="$6"
  "${PYTHON_BIN}" - "$src" "$cpu_dst" "$gpu_dst" "$cpu_num" "$den" >>"${phase_log}" 2>&1 <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
cpu_dst = Path(sys.argv[2])
gpu_dst = Path(sys.argv[3])
cpu_num = int(sys.argv[4])
den = int(sys.argv[5])

lines = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
cpu_lines = []
gpu_lines = []
for idx, line in enumerate(lines):
    if den > 0 and cpu_num > 0 and (idx % den) < cpu_num:
        cpu_lines.append(line.replace("--device cuda", "--device cpu"))
    else:
        gpu_lines.append(line)

cpu_dst.parent.mkdir(parents=True, exist_ok=True)
gpu_dst.parent.mkdir(parents=True, exist_ok=True)
cpu_dst.write_text("\n".join(cpu_lines) + ("\n" if cpu_lines else ""), encoding="utf-8")
gpu_dst.write_text("\n".join(gpu_lines) + ("\n" if gpu_lines else ""), encoding="utf-8")
print(
    {
        "src": str(src),
        "cpu_lines": len(cpu_lines),
        "gpu_lines": len(gpu_lines),
        "cpu_dst": str(cpu_dst),
        "gpu_dst": str(gpu_dst),
    }
)
PY
}

SANITY_PID=""
TRANSITION_CPU_PID=""
MECHANISM_CPU_PID=""

if [[ -z "${MIG_UUIDS}" ]]; then
  MIG_UUIDS="$(detect_mig_uuids)"
fi

log "OUT_ROOT=${OUT_ROOT}"
log "CMD_DIR=${CMD_DIR}"
log "MIG_UUIDS=${MIG_UUIDS:-<none>}"
log "RUN_SANITY=${RUN_SANITY}"
log "RUN_TRANSITION=${RUN_TRANSITION}"
log "RUN_MECHANISM=${RUN_MECHANISM}"
log "SANITY_CPU_JOBS=${SANITY_CPU_JOBS}"
log "TRANSITION_CPU_JOBS=${TRANSITION_CPU_JOBS}"
log "MECHANISM_CPU_JOBS=${MECHANISM_CPU_JOBS}"
log "TRANSITION_CPU_SHARE=${TRANSITION_CPU_SHARE_NUM}/${TRANSITION_SHARE_DEN}"
log "MECHANISM_CPU_SHARE=${MECHANISM_CPU_SHARE_NUM}/${MECHANISM_SHARE_DEN}"
log "MECHANISM_CELLS=${MECHANISM_CELLS}"
log "DRY_RUN=${DRY_RUN}"

if [[ "${RUN_SANITY}" == "1" ]]; then
  SANITY_LOG="${LOG_DIR}/${STAMP}_sanity_suite.log"
  build_suite sanity_suite "${SANITY_LOG}" --device cpu --torch-threads 1
  if [[ "${DRY_RUN}" != "1" ]]; then
    (
      run_xargs_cmd_file "${CMD_DIR}/sanity_suite_cmds.txt" "${SANITY_CPU_JOBS}" "${SANITY_LOG}"
      report_suite sanity_suite "${SANITY_LOG}"
    ) &
    SANITY_PID=$!
    log "START sanity_suite pid=${SANITY_PID}"
  fi
fi

if [[ "${RUN_TRANSITION}" == "1" ]]; then
  TRANSITION_LOG="${LOG_DIR}/${STAMP}_transition_map_suite.log"
  build_suite transition_map_suite "${TRANSITION_LOG}" --device cuda --torch-threads 1
  split_cmd_file \
    "${CMD_DIR}/transition_map_suite_cmds.txt" \
    "${CMD_DIR}/transition_map_suite_cpu_cmds.txt" \
    "${CMD_DIR}/transition_map_suite_gpu_cmds.txt" \
    "${TRANSITION_CPU_SHARE_NUM}" \
    "${TRANSITION_SHARE_DEN}" \
    "${TRANSITION_LOG}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    (
      run_xargs_cmd_file "${CMD_DIR}/transition_map_suite_cpu_cmds.txt" "${TRANSITION_CPU_JOBS}" "${TRANSITION_LOG}"
    ) &
    TRANSITION_CPU_PID=$!
    log "START transition_map_suite_cpu pid=${TRANSITION_CPU_PID}"
    if [[ -n "${MIG_UUIDS}" ]]; then
      log "RUN transition_map_suite GPU shard"
      run_mig_cmd_file "${CMD_DIR}/transition_map_suite_gpu_cmds.txt" "${TRANSITION_LOG}"
    else
      log "RUN transition_map_suite GPU shard fallback on CPU"
      run_xargs_cmd_file "${CMD_DIR}/transition_map_suite_gpu_cmds.txt" "${TRANSITION_CPU_JOBS}" "${TRANSITION_LOG}"
    fi
    if [[ -n "${TRANSITION_CPU_PID}" ]]; then
      log "WAIT transition_map_suite_cpu pid=${TRANSITION_CPU_PID}"
      wait "${TRANSITION_CPU_PID}"
    fi
    report_suite transition_map_suite "${TRANSITION_LOG}"
  fi
fi

TRANSITION_SUMMARY="${OUT_ROOT}/transition_map_suite/markov_changepoint_ops_count/capability_report/markov_capability_summary.json"

if [[ "${RUN_MECHANISM}" == "1" && "${DRY_RUN}" != "1" ]]; then
  if [[ ! -f "${TRANSITION_SUMMARY}" ]]; then
    log "ERROR missing transition summary at ${TRANSITION_SUMMARY}"
    exit 1
  fi
fi

if [[ "${RUN_MECHANISM}" == "1" ]]; then
  MECHANISM_LOG="${LOG_DIR}/${STAMP}_mechanism_suite.log"
  build_suite mechanism_suite "${MECHANISM_LOG}" \
    --device cuda \
    --torch-threads 1 \
    --transition-summary "${TRANSITION_SUMMARY}" \
    --mechanism-cells "${MECHANISM_CELLS}"
  split_cmd_file \
    "${CMD_DIR}/mechanism_suite_cmds.txt" \
    "${CMD_DIR}/mechanism_suite_cpu_cmds.txt" \
    "${CMD_DIR}/mechanism_suite_gpu_cmds.txt" \
    "${MECHANISM_CPU_SHARE_NUM}" \
    "${MECHANISM_SHARE_DEN}" \
    "${MECHANISM_LOG}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    (
      run_xargs_cmd_file "${CMD_DIR}/mechanism_suite_cpu_cmds.txt" "${MECHANISM_CPU_JOBS}" "${MECHANISM_LOG}"
    ) &
    MECHANISM_CPU_PID=$!
    log "START mechanism_suite_cpu pid=${MECHANISM_CPU_PID}"
    if [[ -n "${MIG_UUIDS}" ]]; then
      log "RUN mechanism_suite GPU shard"
      run_mig_cmd_file "${CMD_DIR}/mechanism_suite_gpu_cmds.txt" "${MECHANISM_LOG}"
    else
      log "RUN mechanism_suite GPU shard fallback on CPU"
      run_xargs_cmd_file "${CMD_DIR}/mechanism_suite_gpu_cmds.txt" "${MECHANISM_CPU_JOBS}" "${MECHANISM_LOG}"
    fi
    if [[ -n "${MECHANISM_CPU_PID}" ]]; then
      log "WAIT mechanism_suite_cpu pid=${MECHANISM_CPU_PID}"
      wait "${MECHANISM_CPU_PID}"
    fi
    report_suite mechanism_suite "${MECHANISM_LOG}"
  fi
fi

if [[ -n "${SANITY_PID}" && "${DRY_RUN}" != "1" ]]; then
  log "WAIT sanity_suite pid=${SANITY_PID}"
  wait "${SANITY_PID}"
fi

log "DONE | OUT_ROOT=${OUT_ROOT}"
