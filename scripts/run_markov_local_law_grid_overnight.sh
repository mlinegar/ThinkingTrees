#!/usr/bin/env bash
set -euo pipefail

# Focused overnight Markov sweep for the theorem-facing local-law tradeoff.
#
# Main idea:
# - Treat `local_law_weight=0` as the no-local-law baseline.
# - Sweep up to `local_law_weight=1.0` as the strongest theorem-facing supervision lane.
# - Keep schedule-consistency as a separate proxy-only axis.
#
# Override via env vars, for example:
#   JOBS="$(nproc)" MARKOV_LOCAL_LAW_WEIGHTS="0 0.025 0.05 0.075 0.1 0.15 0.2 0.25 0.35 0.5 0.65 0.8 0.9 1.0" \
#   OUT_ROOT=outputs/... ./scripts/run_markov_local_law_grid_overnight.sh

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_local_law_grid_${STAMP}}"

MARKOV_LOCAL_LAW_WEIGHTS="${MARKOV_LOCAL_LAW_WEIGHTS:-0 0.025 0.05 0.075 0.1 0.15 0.2 0.25 0.35 0.5 0.65 0.8 0.9 1.0}"
MARKOV_C1_RELATIVE_WEIGHTS="${MARKOV_C1_RELATIVE_WEIGHTS:-1.0}"
MARKOV_C3_RELATIVE_WEIGHTS="${MARKOV_C3_RELATIVE_WEIGHTS:-4.0}"
MARKOV_SCW_GRID="${MARKOV_SCW_GRID:-0.0 0.1}"
MARKOV_TRAIN_DOCS="${MARKOV_TRAIN_DOCS:-200 1000 8000}"
MARKOV_AUDIT_FRACTIONS="${MARKOV_AUDIT_FRACTIONS:-0.1 1.0}"
MARKOV_SEEDS="${MARKOV_SEEDS:-0 1 2 3 4 5}"
MARKOV_GUIDANCE_QS="${MARKOV_GUIDANCE_QS:-0 0.1 0.25 0.5 0.75 1.0}"
MARKOV_GUIDANCE_TRIALS="${MARKOV_GUIDANCE_TRIALS:-8}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

detect_cpu_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return 0
  fi
  getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1
}

TOTAL_CPUS="$(detect_cpu_count)"
JOBS="${JOBS:-${TOTAL_CPUS}}"

MARKOV_OUT="${OUT_ROOT}/markov_changepoint_ops_count/local_law_grid"
MARKOV_CMDS="${LOG_DIR}/${STAMP}_markov_local_law_grid_cmds.txt"
RUN_LOG="${LOG_DIR}/${STAMP}_markov_local_law_grid_run.log"

# Avoid BLAS oversubscription when running many processes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

echo "OUT_ROOT=${OUT_ROOT}" | tee "${RUN_LOG}"
echo "JOBS=${JOBS}" | tee -a "${RUN_LOG}"
echo "TOTAL_CPUS=${TOTAL_CPUS}" | tee -a "${RUN_LOG}"
echo "MARKOV_LOCAL_LAW_WEIGHTS=${MARKOV_LOCAL_LAW_WEIGHTS}" | tee -a "${RUN_LOG}"
echo "MARKOV_SCW_GRID=${MARKOV_SCW_GRID}" | tee -a "${RUN_LOG}"

echo "=== Building Markov local-law grid commands ===" | tee -a "${RUN_LOG}"
venv/bin/python -u scripts/build_markov_changepoint_ops_count_cmds.py \
  --out-cmds "${MARKOV_CMDS}" \
  --output-root "${MARKOV_OUT}" \
  --train-docs "${MARKOV_TRAIN_DOCS}" \
  --test-docs 2000 \
  --model-family "neural" \
  --audit-fractions "${MARKOV_AUDIT_FRACTIONS}" \
  --leaf-query-rates "1.0" \
  --include-root-query "true" \
  --local-law-weights "${MARKOV_LOCAL_LAW_WEIGHTS}" \
  --c1-relative-weights "${MARKOV_C1_RELATIVE_WEIGHTS}" \
  --c3-relative-weights "${MARKOV_C3_RELATIVE_WEIGHTS}" \
  --root-weights "1.0" \
  --schedule-consistency-weights "${MARKOV_SCW_GRID}" \
  --guidance-override-modes "adjust" \
  --feature-modes "full" \
  --state-dims "32" \
  --hidden-dim-multiplier 4 \
  --hidden-dim-min 64 \
  --eval-guidance-qs "${MARKOV_GUIDANCE_QS}" \
  --eval-guidance-trials "${MARKOV_GUIDANCE_TRIALS}" \
  --eval-guidance-seed-offset 100000 \
  --eval-guidance-include-root \
  --n-epochs 12 \
  --device cpu \
  --torch-threads 1 \
  --seeds "${MARKOV_SEEDS}" \
  --skip-existing
echo "Markov commands: ${MARKOV_CMDS}" | tee -a "${RUN_LOG}"

echo "=== RUN: Markov local-law grid sweep ===" | tee -a "${RUN_LOG}"
cat "${MARKOV_CMDS}" | xargs -d $'\n' -P "${JOBS}" -I {} bash -lc "{}" | tee -a "${RUN_LOG}"

echo "=== REPORT: Markov local-law grid report archived; see docs/markov_report_archive.md ===" | tee -a "${RUN_LOG}"

echo "DONE | OUT_ROOT=${OUT_ROOT}" | tee -a "${RUN_LOG}"
