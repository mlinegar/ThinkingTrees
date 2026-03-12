#!/usr/bin/env bash
set -euo pipefail

# Overnight CPU sweep focused on *direct* DTM->LDA baselines (bag-of-words DGP).
#
# Compares topic-word estimators that consume a document-term matrix:
#   - tensor_lda (in-repo moment/whitening baseline)
#   - sklearn_lda (scikit-learn variational Bayes baseline)
#
# This is intentionally narrower than `run_identifiable_zero_learnability_overnight.sh`.
#
# Override via env vars:
#   JOBS=128 OUT_ROOT=outputs/... STAMP=... \
#     TOPIC_PHI_ESTIMATORS="tensor_lda sklearn_lda" \
#     ./scripts/run_identifiable_zero_dtm_lda_overnight.sh

JOBS="${JOBS:-128}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/identifiable_zero_dtm_lda_${STAMP}}"
TOPIC_PHI_ESTIMATORS="${TOPIC_PHI_ESTIMATORS:-tensor_lda sklearn_lda}"
GPU_TOKENS="${GPU_TOKENS:-auto}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

RUN_LOG="${LOG_DIR}/${STAMP}_identifiable_zero_dtm_lda_run.log"

CTREE_LDA_OUT="${OUT_ROOT}/segmented_lda_ctreepo/equivalence/lda"
CTREE_LDA_CMDS="${LOG_DIR}/${STAMP}_ctree_dtm_lda_cmds.txt"
CTREE_TMP_CMDS="${LOG_DIR}/${STAMP}_ctree_dtm_lda_tmp_cmds.txt"

echo "OUT_ROOT=${OUT_ROOT}"
echo "JOBS=${JOBS}"
echo "TOPIC_PHI_ESTIMATORS=${TOPIC_PHI_ESTIMATORS}"
echo "RUN_LOG=${RUN_LOG}"
echo "GPU_TOKENS=${GPU_TOKENS}"

echo "=== Building C-TreePO commands (bag_of_words DGP; lstsq θ; DTM-based topic estimators) ===" | tee "${RUN_LOG}"
: > "${CTREE_LDA_CMDS}"
for Q in 0 0.5; do
  venv/bin/python -u scripts/build_segmented_lda_ctreepo_cmds.py \
    --out-cmds "${CTREE_TMP_CMDS}" \
    --output-root "${CTREE_LDA_OUT}" \
    --train-docs "256 512 1024 2048 4096" \
    --n-books-test 5000 \
    --calibration-rates "0.02 0.05 0.1 0.2 0.4" \
    --eval-leaf-rates "${Q}" \
    --eval-internal-rates "${Q}" \
    --topic-process bag_of_words \
    --leaf-theta-estimator lstsq \
    --topic-phi-estimators "${TOPIC_PHI_ESTIMATORS}" \
    --topic-phi-docs 0 \
    --n-topics 4 --vocab-size 256 \
    --min-segments 6 --max-segments 6 \
    --min-seg-tokens 24 --max-seg-tokens 48 \
    --fixed-leaf-tokens 32 \
    --alpha-topic 0.20 --beta-word 0.10 \
    --segment-concentration 80.0 --segment-background 2.0 \
    --topic-phi-permute \
    --eval-internal-query-design risk \
    --seeds "0 1 2 3 4 5" \
    --skip-existing
  cat "${CTREE_TMP_CMDS}" >> "${CTREE_LDA_CMDS}"
done

echo "CTree DTM/LDA commands: ${CTREE_LDA_CMDS}" | tee -a "${RUN_LOG}"
echo "n_cmds=$(wc -l < "${CTREE_LDA_CMDS}" | tr -d ' ')" | tee -a "${RUN_LOG}"

echo "=== RUN: C-TreePO (DTM/LDA baselines; bag_of_words DGP) ===" | tee -a "${RUN_LOG}"
venv/bin/python scripts/run_simulation_resource_queue.py \
  --cmd-file "${CTREE_LDA_CMDS}" \
  --cpu-workers "${JOBS}" \
  --gpu-tokens "${GPU_TOKENS}" \
  --log-dir "${LOG_DIR}/${STAMP}_ctree_dtm_lda_queue_logs" | tee -a "${RUN_LOG}"

echo "DONE | OUT_ROOT=${OUT_ROOT}" | tee -a "${RUN_LOG}"
