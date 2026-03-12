#!/usr/bin/env bash
set -euo pipefail

# Overnight CPU sweep focusing on "information density" and operator comparisons:
# - Markov learned sketch capacity/feature ablations
#   (state_dim, feature_mode, guidance semantics, llw, c1/c3 mix, scw)
# - C-TreePO topic-phi estimator density sweep (topic_phi_docs + estimator; plus neural seed fraction)
#
# Defaults are chosen to be sweep-friendly and paper-visualization oriented.
# Override via env vars:
#   JOBS=64 OUT_ROOT=outputs/... ./scripts/run_identifiable_zero_neural_operator_overnight.sh

JOBS="${JOBS:-48}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/identifiable_zero_suite_20260303_longrun_equiv_v1_neural_operator_overnight_${STAMP}}"
MARKOV_DEVICE="${MARKOV_DEVICE:-auto}"
MARKOV_LOCAL_LAW_WEIGHTS="${MARKOV_LOCAL_LAW_WEIGHTS:-0 0.25 0.5 1.0}"
MARKOV_C1_RELATIVE_WEIGHTS="${MARKOV_C1_RELATIVE_WEIGHTS:-1.0}"
MARKOV_C3_RELATIVE_WEIGHTS="${MARKOV_C3_RELATIVE_WEIGHTS:-4.0}"
MARKOV_ROOT_WEIGHTS="${MARKOV_ROOT_WEIGHTS:-1.0}"
MARKOV_SCHEDULE_CONSISTENCY_WEIGHTS="${MARKOV_SCHEDULE_CONSISTENCY_WEIGHTS:-0.0 0.1}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

MARKOV_OUT="${OUT_ROOT}/markov_changepoint_ops_count/equivalence"
CTREE_OUT="${OUT_ROOT}/segmented_lda_ctreepo/equivalence"

MARKOV_CMDS="${LOG_DIR}/${STAMP}_markov_operator_capacity_cmds.txt"
CTREE_CMDS="${LOG_DIR}/${STAMP}_ctree_phi_density_cmds.txt"
RUN_LOG="${LOG_DIR}/${STAMP}_neural_operator_overnight_run.log"

echo "OUT_ROOT=${OUT_ROOT}"
echo "JOBS=${JOBS}"
echo "MARKOV_LOCAL_LAW_WEIGHTS=${MARKOV_LOCAL_LAW_WEIGHTS}"
echo "MARKOV_C1_RELATIVE_WEIGHTS=${MARKOV_C1_RELATIVE_WEIGHTS}"
echo "MARKOV_C3_RELATIVE_WEIGHTS=${MARKOV_C3_RELATIVE_WEIGHTS}"
echo "MARKOV_ROOT_WEIGHTS=${MARKOV_ROOT_WEIGHTS}"
echo "MARKOV_SCHEDULE_CONSISTENCY_WEIGHTS=${MARKOV_SCHEDULE_CONSISTENCY_WEIGHTS}"
echo "MARKOV_DEVICE=${MARKOV_DEVICE}"
echo "RUN_LOG=${RUN_LOG}"

echo "=== Building Markov capacity commands ===" | tee "${RUN_LOG}"
venv/bin/python -u scripts/build_markov_changepoint_ops_count_cmds.py \
  --out-cmds "${MARKOV_CMDS}" \
  --output-root "${MARKOV_OUT}" \
  --train-docs "8000" \
  --test-docs 2000 \
  --model-family "neural" \
  --audit-fractions "0.1 1.0" \
  --leaf-query-rates "1.0" \
  --include-root-query "true" \
  --local-law-weights "${MARKOV_LOCAL_LAW_WEIGHTS}" \
  --c1-relative-weights "${MARKOV_C1_RELATIVE_WEIGHTS}" \
  --c3-relative-weights "${MARKOV_C3_RELATIVE_WEIGHTS}" \
  --root-weights "${MARKOV_ROOT_WEIGHTS}" \
  --schedule-consistency-weights "${MARKOV_SCHEDULE_CONSISTENCY_WEIGHTS}" \
  --guidance-override-modes "reset adjust" \
  --feature-modes "full no_endpoints" \
  --state-dims "8 16 32 64" \
  --hidden-dim-multiplier 4 \
  --hidden-dim-min 64 \
  --eval-guidance-qs "0 0.05 0.1 0.15 0.2 0.25 0.35 0.5 0.75 1.0" \
  --eval-guidance-trials 8 \
  --eval-guidance-seed-offset 100000 \
  --eval-guidance-include-root \
  --n-epochs 12 \
  --device "${MARKOV_DEVICE}" \
  --torch-threads 1 \
  --seeds "0 1 2 3 4 5" \
  --skip-existing

echo "Markov commands: ${MARKOV_CMDS}" | tee -a "${RUN_LOG}"

echo "=== Building C-TreePO topic-phi density commands ===" | tee -a "${RUN_LOG}"
venv/bin/python -u scripts/build_segmented_lda_ctreepo_cmds.py \
  --out-cmds "${CTREE_CMDS}" \
  --output-root "${CTREE_OUT}" \
  --train-docs "4096" \
  --n-books-test 5000 \
  --calibration-rates "0.1" \
  --eval-leaf-rates "0" \
  --eval-internal-rates "0" \
  --topic-phi-estimators "spectral_numpy tensor_lda online_tensor_lda neural_ctreepo" \
  --topic-phi-docs-grid "64 128 256 512 1024 2048 4096" \
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

echo "C-TreePO commands: ${CTREE_CMDS}" | tee -a "${RUN_LOG}"

echo "=== RUN: Markov capacity sweep ===" | tee -a "${RUN_LOG}"
cat "${MARKOV_CMDS}" | xargs -d $'\n' -P "${JOBS}" -I {} bash -lc "{}" | tee -a "${RUN_LOG}"

echo "=== RUN: C-TreePO phi density sweep ===" | tee -a "${RUN_LOG}"
cat "${CTREE_CMDS}" | xargs -d $'\n' -P "${JOBS}" -I {} bash -lc "{}" | tee -a "${RUN_LOG}"

echo "DONE | OUT_ROOT=${OUT_ROOT}" | tee -a "${RUN_LOG}"
