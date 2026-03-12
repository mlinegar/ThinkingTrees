#!/usr/bin/env bash
set -euo pipefail

# Overnight CPU sweep v2: extend the neural-operator analysis in a paper-visualization-driven way.
#
# Focus:
# - Markov learned-sketch: schedule-consistency regularization sweep (fine grid) at baseline capacity.
# - Segmented-LDA C-TreePO: compare neural topic-phi operators (ctreepo vs mergeable vs hybrid) vs baselines
#   across decision-time oracle visibility (q_infer), keeping topic-phi docs high (4096).
#
# Override via env vars:
#   JOBS=96 OUT_ROOT=outputs/... ./scripts/run_identifiable_zero_neural_operator_overnight_v2.sh

JOBS="${JOBS:-80}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/identifiable_zero_suite_20260303_longrun_equiv_v1_neural_operator_overnight2_${STAMP}}"
MARKOV_DEVICE="${MARKOV_DEVICE:-auto}"
GPU_TOKENS="${GPU_TOKENS:-auto}"
BUILD_ONLY="${BUILD_ONLY:-0}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

# Avoid BLAS oversubscription when running many processes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

RUN_LOG="${LOG_DIR}/${STAMP}_neural_operator_overnight2_run.log"

MARKOV_OUT="${OUT_ROOT}/markov_changepoint_ops_count/equivalence"
CTREE_OUT="${OUT_ROOT}/segmented_lda_ctreepo/equivalence"

MARKOV_CMDS="${LOG_DIR}/${STAMP}_markov_scw_fine_cmds.txt"
CTREE_CMDS="${LOG_DIR}/${STAMP}_ctree_operator_family_cmds.txt"
ALL_CMDS="${LOG_DIR}/${STAMP}_neural_operator_all_cmds.txt"

echo "OUT_ROOT=${OUT_ROOT}" | tee "${RUN_LOG}"
echo "JOBS=${JOBS}" | tee -a "${RUN_LOG}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}" | tee -a "${RUN_LOG}"
echo "MKL_NUM_THREADS=${MKL_NUM_THREADS}" | tee -a "${RUN_LOG}"
echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}" | tee -a "${RUN_LOG}"
echo "MARKOV_DEVICE=${MARKOV_DEVICE}" | tee -a "${RUN_LOG}"
echo "GPU_TOKENS=${GPU_TOKENS}" | tee -a "${RUN_LOG}"
echo "BUILD_ONLY=${BUILD_ONLY}" | tee -a "${RUN_LOG}"

echo "=== Building Markov SCW-fine commands ===" | tee -a "${RUN_LOG}"
venv/bin/python -u scripts/build_markov_changepoint_ops_count_cmds.py \
  --out-cmds "${MARKOV_CMDS}" \
  --output-root "${MARKOV_OUT}" \
  --train-docs "8000" \
  --test-docs 2000 \
  --model-family "neural" \
  --audit-fractions "0.1 1.0" \
  --leaf-query-rates "1.0" \
  --include-root-query "true" \
  --root-weights "1.0" \
  --schedule-consistency-weights "0 0.01 0.03 0.1 0.3" \
  --guidance-override-modes "adjust" \
  --feature-modes "full" \
  --state-dims "32" \
  --hidden-dim-multiplier 4 \
  --hidden-dim-min 64 \
  --eval-guidance-qs "0 0.1 0.25 0.5 0.75 1.0" \
  --eval-guidance-trials 8 \
  --eval-guidance-seed-offset 100000 \
  --eval-guidance-include-root \
  --n-epochs 12 \
  --device "${MARKOV_DEVICE}" \
  --torch-threads 1 \
  --seeds "0 1 2 3 4 5 6 7 8 9 10 11" \
  --skip-existing
echo "Markov commands: ${MARKOV_CMDS}" | tee -a "${RUN_LOG}"

echo "=== Building C-TreePO operator-family commands (coupled q_infer) ===" | tee -a "${RUN_LOG}"
rm -f "${CTREE_CMDS}"
touch "${CTREE_CMDS}"
for q in 0 0.5 1.0; do
  TMP="${LOG_DIR}/${STAMP}_ctree_operator_family_q_${q}.txt"
  venv/bin/python -u scripts/build_segmented_lda_ctreepo_cmds.py \
    --out-cmds "${TMP}" \
    --output-root "${CTREE_OUT}" \
    --train-docs "4096" \
    --n-books-test 5000 \
    --calibration-rates "0.1" \
    --eval-leaf-rates "${q}" \
    --eval-internal-rates "${q}" \
    --topic-phi-estimators "spectral_numpy tensor_lda online_tensor_lda neural_ctreepo neural_mergeable_sketch neural_hybrid" \
    --topic-phi-docs 4096 \
    --neural-topic-seed-fractions "0.35 0.5 0.75 1.0" \
    --n-topics 4 \
    --vocab-size 256 \
    --min-segments 6 --max-segments 6 \
    --min-seg-tokens 24 --max-seg-tokens 48 \
    --fixed-leaf-tokens 32 \
    --topic-phi-permute \
    --eval-internal-query-design risk \
    --seeds "0 1 2 3 4 5" \
    --skip-existing
  cat "${TMP}" >> "${CTREE_CMDS}"
done
echo "C-TreePO commands: ${CTREE_CMDS}" | tee -a "${RUN_LOG}"
{
  cat "${MARKOV_CMDS}"
  cat "${CTREE_CMDS}"
} > "${ALL_CMDS}"
echo "combined commands: ${ALL_CMDS}" | tee -a "${RUN_LOG}"
echo "markov_n_cmds=$(wc -l < "${MARKOV_CMDS}" | tr -d ' ')" | tee -a "${RUN_LOG}"
echo "ctree_n_cmds=$(wc -l < "${CTREE_CMDS}" | tr -d ' ')" | tee -a "${RUN_LOG}"
echo "all_n_cmds=$(wc -l < "${ALL_CMDS}" | tr -d ' ')" | tee -a "${RUN_LOG}"

if [[ "${BUILD_ONLY}" != "0" ]]; then
  echo "BUILD_ONLY=1 -> exiting after command generation." | tee -a "${RUN_LOG}"
  exit 0
fi

echo "=== RUN: combined mixed-resource neural-operator sweep ===" | tee -a "${RUN_LOG}"
venv/bin/python scripts/run_simulation_resource_queue.py \
  --cmd-file "${ALL_CMDS}" \
  --cpu-workers "${JOBS}" \
  --gpu-tokens "${GPU_TOKENS}" \
  --log-dir "${LOG_DIR}/${STAMP}_neural_operator_queue_logs" | tee -a "${RUN_LOG}"

echo "DONE | OUT_ROOT=${OUT_ROOT}" | tee -a "${RUN_LOG}"
