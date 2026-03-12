#!/usr/bin/env bash
set -euo pipefail

# Overnight CPU sweep for a "learnability progression" where the difficulty axis is leaf noise:
# we hold the LDA DGP fixed and vary `fixed_leaf_tokens` (full-doc leaves -> small noisy leaves).
#
# This sweep is focused on the scikit-learn LDA baseline end-to-end:
#   topic_phi_estimator=sklearn_lda, leaf_theta_estimator=sklearn_lda
#
# Override via env vars:
#   JOBS=128 OUT_ROOT=outputs/... STAMP=... \
#   TRAIN_DOCS="16 32 64 ..." LEAF_TOKENS="2048 512 ..." CAL_RATES="0 0.1" SEEDS="0 1 ..." \
#     ./scripts/run_identifiable_zero_lda_leafnoise_overnight.sh

JOBS="${JOBS:-128}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/identifiable_zero_lda_leafnoise_${STAMP}}"
GPU_TOKENS="${GPU_TOKENS:-auto}"

TRAIN_DOCS="${TRAIN_DOCS:-16 32 64 128 256 512 1024 2048}"
LEAF_TOKENS="${LEAF_TOKENS:-2048 512 128 32 8}"
CAL_RATES="${CAL_RATES:-0 0.1}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"

# LDA DGP: bag_of_words, fixed document length via segment params.
N_TOPICS="${N_TOPICS:-4}"
VOCAB_SIZE="${VOCAB_SIZE:-256}"
DOC_TOKENS="${DOC_TOKENS:-2048}"
ALPHA_TOPIC="${ALPHA_TOPIC:-0.20}"
BETA_WORD="${BETA_WORD:-0.10}"
N_BOOKS_TEST="${N_BOOKS_TEST:-2000}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

RUN_LOG="${LOG_DIR}/${STAMP}_identifiable_zero_lda_leafnoise_run.log"

CTREE_OUT="${OUT_ROOT}/segmented_lda_ctreepo/equivalence/lda_leafnoise"
CTREE_CMDS="${LOG_DIR}/${STAMP}_ctree_lda_leafnoise_cmds.txt"
CTREE_TMP="${LOG_DIR}/${STAMP}_ctree_lda_leafnoise_tmp_cmds.txt"

echo "OUT_ROOT=${OUT_ROOT}" | tee "${RUN_LOG}"
echo "JOBS=${JOBS}" | tee -a "${RUN_LOG}"
echo "TRAIN_DOCS=${TRAIN_DOCS}" | tee -a "${RUN_LOG}"
echo "LEAF_TOKENS=${LEAF_TOKENS}" | tee -a "${RUN_LOG}"
echo "CAL_RATES=${CAL_RATES}" | tee -a "${RUN_LOG}"
echo "SEEDS=${SEEDS}" | tee -a "${RUN_LOG}"
echo "DOC_TOKENS=${DOC_TOKENS} | N_TOPICS=${N_TOPICS} | VOCAB_SIZE=${VOCAB_SIZE}" | tee -a "${RUN_LOG}"
echo "N_BOOKS_TEST=${N_BOOKS_TEST}" | tee -a "${RUN_LOG}"
echo "GPU_TOKENS=${GPU_TOKENS}" | tee -a "${RUN_LOG}"

echo "=== Building commands ===" | tee -a "${RUN_LOG}"
: > "${CTREE_CMDS}"
for LT in ${LEAF_TOKENS}; do
  venv/bin/python -u scripts/build_segmented_lda_ctreepo_cmds.py \
    --out-cmds "${CTREE_TMP}" \
    --output-root "${CTREE_OUT}" \
    --train-docs "${TRAIN_DOCS}" \
    --n-books-test "${N_BOOKS_TEST}" \
    --calibration-rates "${CAL_RATES}" \
    --eval-leaf-rates "0" \
    --eval-internal-rates "0" \
    --topic-process bag_of_words \
    --leaf-theta-estimator sklearn_lda \
    --topic-phi-estimators "sklearn_lda" \
    --topic-phi-docs 0 \
    --n-topics "${N_TOPICS}" --vocab-size "${VOCAB_SIZE}" \
    --min-segments 1 --max-segments 1 \
    --min-seg-tokens "${DOC_TOKENS}" --max-seg-tokens "${DOC_TOKENS}" \
    --fixed-leaf-tokens "${LT}" \
    --alpha-topic "${ALPHA_TOPIC}" --beta-word "${BETA_WORD}" \
    --segment-concentration 80.0 --segment-background 2.0 \
    --topic-phi-permute \
    --eval-internal-query-design risk \
    --seeds "${SEEDS}" \
    --skip-existing
  cat "${CTREE_TMP}" >> "${CTREE_CMDS}"
done

echo "CTree commands: ${CTREE_CMDS}" | tee -a "${RUN_LOG}"
echo "n_cmds=$(wc -l < "${CTREE_CMDS}" | tr -d ' ')" | tee -a "${RUN_LOG}"

echo "=== RUN: C-TreePO leaf-noise progression (sklearn LDA baseline) ===" | tee -a "${RUN_LOG}"
venv/bin/python scripts/run_simulation_resource_queue.py \
  --cmd-file "${CTREE_CMDS}" \
  --cpu-workers "${JOBS}" \
  --gpu-tokens "${GPU_TOKENS}" \
  --log-dir "${LOG_DIR}/${STAMP}_ctree_lda_leafnoise_queue_logs" | tee -a "${RUN_LOG}"

echo "DONE | OUT_ROOT=${OUT_ROOT}" | tee -a "${RUN_LOG}"
