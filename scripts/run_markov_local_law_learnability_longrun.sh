#!/usr/bin/env bash
set -euo pipefail

# Long-run Markov local-law learnability sweep.
#
# Purpose:
# - separate corpus variance (`data_seed`) from optimization variance (`seed` / `model_seed`)
# - sweep theorem-facing local-law weight with a nontrivial optimization budget
# - produce a direct C1/C3 learnability report in addition to the generic OPS report

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_local_law_learnability_${STAMP}}"
LOG_DIR="${LOG_DIR:-logs}"
MARKOV_OUT="${OUT_ROOT}/markov_changepoint_ops_count/local_law_learnability"
CMD_FILE="${LOG_DIR}/${STAMP}_markov_local_law_learnability_cmds.txt"
RUN_LOG="${LOG_DIR}/${STAMP}_markov_local_law_learnability.log"

MARKOV_TRAIN_DOCS="${MARKOV_TRAIN_DOCS:-256 1024 4096}"
MARKOV_TEST_DOCS="${MARKOV_TEST_DOCS:-1024}"
MARKOV_AUDIT_FRACTIONS="${MARKOV_AUDIT_FRACTIONS:-0.1 0.5 1.0}"
MARKOV_LOCAL_LAW_WEIGHTS="${MARKOV_LOCAL_LAW_WEIGHTS:-0 0.025 0.05 0.075 0.1 0.15 0.2 0.25 0.35 0.5 0.65 0.8 0.9 1.0}"
MARKOV_C1_RELATIVE_WEIGHTS="${MARKOV_C1_RELATIVE_WEIGHTS:-1.0}"
MARKOV_C3_RELATIVE_WEIGHTS="${MARKOV_C3_RELATIVE_WEIGHTS:-4.0}"
MARKOV_ROOT_WEIGHTS="${MARKOV_ROOT_WEIGHTS:-1.0}"
MARKOV_SCW_GRID="${MARKOV_SCW_GRID:-0.0 0.05 0.1}"
MARKOV_DATA_SEEDS="${MARKOV_DATA_SEEDS:-0 1 2 3}"
MARKOV_MODEL_SEEDS="${MARKOV_MODEL_SEEDS:-0 1 2 3}"
MARKOV_STATE_DIMS="${MARKOV_STATE_DIMS:-64}"
MARKOV_HIDDEN_DIM_MULTIPLIER="${MARKOV_HIDDEN_DIM_MULTIPLIER:-4}"
MARKOV_HIDDEN_DIM_MIN="${MARKOV_HIDDEN_DIM_MIN:-128}"
MARKOV_EPOCHS="${MARKOV_EPOCHS:-20}"
MARKOV_DEVICE="${MARKOV_DEVICE:-auto}"
MARKOV_TORCH_THREADS="${MARKOV_TORCH_THREADS:-1}"

mkdir -p "${LOG_DIR}" "${OUT_ROOT}" "${MARKOV_OUT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

detect_mig_uuids() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  nvidia-smi -L 2>/dev/null | grep -o 'MIG-[A-Za-z0-9-]*' | awk '!seen[$0]++' | paste -sd' ' -
}

detect_cpu_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return 0
  fi
  getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1
}

MIG_UUIDS="${MIG_UUIDS:-$(detect_mig_uuids)}"
USE_MIG=0
if [[ -n "${MIG_UUIDS}" && "${MARKOV_DEVICE}" != "cpu" ]]; then
  USE_MIG=1
  MARKOV_DEVICE="cuda"
fi
TOTAL_CPUS="$(detect_cpu_count)"
if [[ -z "${MARKOV_JOBS:-}" ]]; then
  if [[ "${MARKOV_DEVICE}" == "cpu" ]]; then
    MARKOV_JOBS="${TOTAL_CPUS}"
  else
    MARKOV_JOBS="64"
  fi
fi

{
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "MARKOV_OUT=${MARKOV_OUT}"
  echo "MARKOV_DEVICE=${MARKOV_DEVICE}"
  echo "USE_MIG=${USE_MIG}"
  echo "TOTAL_CPUS=${TOTAL_CPUS}"
  echo "MARKOV_JOBS=${MARKOV_JOBS}"
  echo "MIG_UUIDS=${MIG_UUIDS}"
  echo "MARKOV_TRAIN_DOCS=${MARKOV_TRAIN_DOCS}"
  echo "MARKOV_AUDIT_FRACTIONS=${MARKOV_AUDIT_FRACTIONS}"
  echo "MARKOV_LOCAL_LAW_WEIGHTS=${MARKOV_LOCAL_LAW_WEIGHTS}"
  echo "MARKOV_C1_RELATIVE_WEIGHTS=${MARKOV_C1_RELATIVE_WEIGHTS}"
  echo "MARKOV_C3_RELATIVE_WEIGHTS=${MARKOV_C3_RELATIVE_WEIGHTS}"
  echo "MARKOV_ROOT_WEIGHTS=${MARKOV_ROOT_WEIGHTS}"
  echo "MARKOV_SCW_GRID=${MARKOV_SCW_GRID}"
  echo "MARKOV_DATA_SEEDS=${MARKOV_DATA_SEEDS}"
  echo "MARKOV_MODEL_SEEDS=${MARKOV_MODEL_SEEDS}"
} | tee "${RUN_LOG}"

venv/bin/python -u scripts/build_markov_changepoint_ops_count_cmds.py \
  --out-cmds "${CMD_FILE}" \
  --output-root "${MARKOV_OUT}" \
  --train-docs "${MARKOV_TRAIN_DOCS}" \
  --test-docs "${MARKOV_TEST_DOCS}" \
  --model-family "neural" \
  --audit-fractions "${MARKOV_AUDIT_FRACTIONS}" \
  --leaf-query-rates "1.0" \
  --include-root-query "true" \
  --local-law-weights "${MARKOV_LOCAL_LAW_WEIGHTS}" \
  --c1-relative-weights "${MARKOV_C1_RELATIVE_WEIGHTS}" \
  --c3-relative-weights "${MARKOV_C3_RELATIVE_WEIGHTS}" \
  --root-weights "${MARKOV_ROOT_WEIGHTS}" \
  --schedule-consistency-weights "${MARKOV_SCW_GRID}" \
  --feature-modes "full" \
  --state-dims "${MARKOV_STATE_DIMS}" \
  --hidden-dim-multiplier "${MARKOV_HIDDEN_DIM_MULTIPLIER}" \
  --hidden-dim-min "${MARKOV_HIDDEN_DIM_MIN}" \
  --n-epochs "${MARKOV_EPOCHS}" \
  --device "${MARKOV_DEVICE}" \
  --torch-threads "${MARKOV_TORCH_THREADS}" \
  --data-seeds "${MARKOV_DATA_SEEDS}" \
  --seeds "${MARKOV_MODEL_SEEDS}" \
  --skip-existing | tee -a "${RUN_LOG}"

if [[ "${USE_MIG}" == "1" ]]; then
  MIG_LOG_DIR="${MARKOV_OUT}/mig_logs"
  mkdir -p "${MIG_LOG_DIR}"
  echo "=== RUN: MIG queue ===" | tee -a "${RUN_LOG}"
  venv/bin/python -u scripts/run_mig_command_queue.py \
    --cmd-file "${CMD_FILE}" \
    --log-dir "${MIG_LOG_DIR}" \
    --mig-uuids "${MIG_UUIDS}" \
    --append-cuda-device-zero | tee -a "${RUN_LOG}"
else
  echo "=== RUN: xargs queue ===" | tee -a "${RUN_LOG}"
  cat "${CMD_FILE}" | xargs -d $'\n' -P "${MARKOV_JOBS}" -I {} bash -lc "{}" | tee -a "${RUN_LOG}"
fi

echo "=== REPORT: local-law learnability ===" | tee -a "${RUN_LOG}"
venv/bin/python -u scripts/report_learnability.py \
  --family markov \
  --input-root "${MARKOV_OUT}" \
  --output-dir "${MARKOV_OUT}/local_law_report" | tee -a "${RUN_LOG}"

echo "=== REPORT: generic OPS report archived; see docs/markov_report_archive.md ===" | tee -a "${RUN_LOG}"

echo "DONE | OUT_ROOT=${OUT_ROOT}" | tee -a "${RUN_LOG}"
