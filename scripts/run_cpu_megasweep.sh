#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_cpu_megasweep.sh --cmds <cmds.txt> [--plot-cmds <plots.txt>] [--jobs <n>] [--gpu-tokens <spec>] [--log <path>]

Runs the unified sims command list through the shared resource queue, then runs plot commands sequentially.

Notes:
- Sets CPU-thread env vars to avoid BLAS/PyTorch oversubscription in parallel runs.
- If `--jobs` is omitted, defaults to 128 workers.
- `--gpu-tokens auto` uses all visible MIG slices (or full GPUs when MIG is absent).
- Safe to resume: builders should have emitted --skip-existing-aware commands.
EOF
}

CMDS=""
PLOT_CMDS=""
JOBS=""
LOG_PATH=""
GPU_TOKENS="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cmds) CMDS="$2"; shift 2;;
    --plot-cmds) PLOT_CMDS="$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    --gpu-tokens) GPU_TOKENS="$2"; shift 2;;
    --log) LOG_PATH="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "${CMDS}" ]]; then
  echo "Missing --cmds" >&2
  usage
  exit 2
fi
if [[ ! -f "${CMDS}" ]]; then
  echo "cmds file not found: ${CMDS}" >&2
  exit 2
fi

if [[ -z "${JOBS}" ]]; then
  JOBS="128"
fi

# Avoid runaway oversubscription for numpy/BLAS + torch.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ -n "${LOG_PATH}" ]]; then
  mkdir -p "$(dirname "${LOG_PATH}")"
  exec > >(tee -a "${LOG_PATH}") 2>&1
fi

echo "megasweep | start | $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "megasweep | cmds=${CMDS} | jobs=${JOBS}"
if [[ -n "${PLOT_CMDS}" ]]; then
  echo "megasweep | plot_cmds=${PLOT_CMDS}"
fi
echo "megasweep | gpu_tokens=${GPU_TOKENS}"
echo "megasweep | threads | OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS}"

CMD_COUNT="$(grep -cve '^[[:space:]]*$' "${CMDS}" || true)"
echo "megasweep | sim_commands=${CMD_COUNT}"

QUEUE_LOG_DIR="$(dirname "${CMDS}")/$(basename "${CMDS}" .txt)_logs"

set +e
venv/bin/python scripts/run_simulation_resource_queue.py \
  --cmd-file "${CMDS}" \
  --cpu-workers "${JOBS}" \
  --gpu-tokens "${GPU_TOKENS}" \
  --log-dir "${QUEUE_LOG_DIR}"
SIM_RC=$?
set -e

echo "megasweep | sims_done | rc=${SIM_RC} | $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

if [[ -n "${PLOT_CMDS}" && -f "${PLOT_CMDS}" ]]; then
  PLOT_COUNT="$(grep -cve '^[[:space:]]*$' "${PLOT_CMDS}" || true)"
  echo "megasweep | plot_commands=${PLOT_COUNT}"
  while IFS= read -r line; do
    [[ -z "${line// }" ]] && continue
    echo "plot | ${line}"
    bash -lc "${line}"
  done < "${PLOT_CMDS}"
  echo "megasweep | plots_done | $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
fi

exit "${SIM_RC}"
