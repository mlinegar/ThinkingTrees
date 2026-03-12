#!/usr/bin/env bash
set -euo pipefail

# Publication-oriented CPU sweep for C-TreePO vs LDA learnability.
#
# Goals:
# - Show learning progress with more data for both direct LDA and C-TreePO lanes.
# - Make tasks harder (more topics/vocab, smaller leaves, tougher segment stats).
# - Audit "oracle signal" strength in neural_ctreepo via weak/default/upper lanes.
# - Keep held-out comparisons stable and reproducible across train-size sweeps.
#
# Override via env vars:
#   JOBS=128 OUT_ROOT=outputs/... STAMP=... \
#   BUILD_ONLY=1 ./scripts/run_identifiable_zero_publication_ctreepo_cpu_pass.sh

JOBS="${JOBS:-128}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/identifiable_zero_publication_ctreepo_${STAMP}}"
BUILD_ONLY="${BUILD_ONLY:-0}"
RUN_COMBINED="${RUN_COMBINED:-1}"
GPU_TOKENS="${GPU_TOKENS:-auto}"

SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
Q_RATES="${Q_RATES:-0 0.25 0.5}"

TRAIN_DOCS_LDA="${TRAIN_DOCS_LDA:-128 256 512 1024 2048 4096}"
TRAIN_DOCS_HARD="${TRAIN_DOCS_HARD:-128 256 512 1024 2048}"
TRAIN_DOCS_HARD_UPPER="${TRAIN_DOCS_HARD_UPPER:-1024 2048 4096}"

LEAF_TOKENS_LDA="${LEAF_TOKENS_LDA:-32 16 8}"
LEAF_TOKENS_HARD="${LEAF_TOKENS_HARD:-16 8}"

CAL_RATES_LDA="${CAL_RATES_LDA:-0 0.05 0.1}"
CAL_RATES_HARD="${CAL_RATES_HARD:-0.05 0.1 0.2}"
CAL_RATES_UPPER="${CAL_RATES_UPPER:-0.1}"
Q_RATES_UPPER="${Q_RATES_UPPER:-0 0.25}"

N_BOOKS_TEST_LDA="${N_BOOKS_TEST_LDA:-4000}"
N_BOOKS_TEST_HARD="${N_BOOKS_TEST_HARD:-5000}"

DOC_TOKENS_LDA="${DOC_TOKENS_LDA:-2048}"

LOG_DIR="logs"
CMDS_DIR="${LOG_DIR}/${STAMP}_publication_ctreepo_cmds"
RUN_LOG="${LOG_DIR}/${STAMP}_identifiable_zero_publication_ctreepo_run.log"

mkdir -p "${LOG_DIR}" "${CMDS_DIR}" "${OUT_ROOT}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

CMD_TMP="${CMDS_DIR}/_tmp_cmds.txt"
CMD_ALL="${CMDS_DIR}/all_cmds.txt"

CMD_LDA_DIRECT="${CMDS_DIR}/lda_direct.txt"
CMD_LDA_BASE="${CMDS_DIR}/lda_phi_base.txt"
CMD_LDA_NEURAL_WEAK="${CMDS_DIR}/lda_neural_weak.txt"
CMD_LDA_NEURAL_DEFAULT="${CMDS_DIR}/lda_neural_default.txt"
CMD_HARD_BASE="${CMDS_DIR}/hard_phi_base.txt"
CMD_HARD_NEURAL_WEAK="${CMDS_DIR}/hard_neural_weak.txt"
CMD_HARD_NEURAL_DEFAULT="${CMDS_DIR}/hard_neural_default.txt"
CMD_HARD_NEURAL_UPPER="${CMDS_DIR}/hard_neural_upper.txt"

count_cmds() {
  local f="$1"
  if [[ -f "${f}" ]]; then
    wc -l < "${f}" | tr -d ' '
  else
    echo "0"
  fi
}

append_built_cmds() {
  local out_file="$1"
  shift
  venv/bin/python -u scripts/build_segmented_lda_ctreepo_cmds.py --out-cmds "${CMD_TMP}" "$@"
  cat "${CMD_TMP}" >> "${out_file}"
}

run_stage() {
  local stage_name="$1"
  local stage_slug="$2"
  local cmd_file="$3"
  local n_cmds
  n_cmds="$(count_cmds "${cmd_file}")"
  echo "=== RUN: ${stage_name} | n_cmds=${n_cmds} ===" | tee -a "${RUN_LOG}"
  if [[ "${n_cmds}" == "0" ]]; then
    return 0
  fi
  venv/bin/python scripts/run_simulation_resource_queue.py \
    --cmd-file "${cmd_file}" \
    --cpu-workers "${JOBS}" \
    --gpu-tokens "${GPU_TOKENS}" \
    --log-dir "${CMDS_DIR}/${stage_slug}_queue_logs" | tee -a "${RUN_LOG}"
}

echo "OUT_ROOT=${OUT_ROOT}" | tee "${RUN_LOG}"
echo "JOBS=${JOBS}" | tee -a "${RUN_LOG}"
echo "RUN_LOG=${RUN_LOG}" | tee -a "${RUN_LOG}"
echo "CMDS_DIR=${CMDS_DIR}" | tee -a "${RUN_LOG}"
echo "SEEDS=${SEEDS}" | tee -a "${RUN_LOG}"
echo "Q_RATES=${Q_RATES}" | tee -a "${RUN_LOG}"
echo "TRAIN_DOCS_LDA=${TRAIN_DOCS_LDA}" | tee -a "${RUN_LOG}"
echo "TRAIN_DOCS_HARD=${TRAIN_DOCS_HARD}" | tee -a "${RUN_LOG}"
echo "LEAF_TOKENS_LDA=${LEAF_TOKENS_LDA}" | tee -a "${RUN_LOG}"
echo "LEAF_TOKENS_HARD=${LEAF_TOKENS_HARD}" | tee -a "${RUN_LOG}"
echo "GPU_TOKENS=${GPU_TOKENS}" | tee -a "${RUN_LOG}"
echo "RUN_COMBINED=${RUN_COMBINED}" | tee -a "${RUN_LOG}"

echo "=== Building commands: LDA regime (k=8,v=512,bag_of_words) ===" | tee -a "${RUN_LOG}"
: > "${CMD_LDA_DIRECT}"
for LT in ${LEAF_TOKENS_LDA}; do
  for Q in ${Q_RATES}; do
    append_built_cmds "${CMD_LDA_DIRECT}" \
      --output-root "${OUT_ROOT}/segmented_lda_ctreepo/equivalence/lda/k8_v512/lane_lda_direct" \
      --train-docs "${TRAIN_DOCS_LDA}" \
      --n-books-test "${N_BOOKS_TEST_LDA}" \
      --calibration-rates "${CAL_RATES_LDA}" \
      --eval-leaf-rates "${Q}" \
      --eval-internal-rates "${Q}" \
      --topic-process bag_of_words \
      --leaf-theta-estimator sklearn_lda \
      --topic-phi-estimators "sklearn_lda" \
      --topic-phi-docs 0 \
      --n-topics 8 --vocab-size 512 \
      --min-segments 1 --max-segments 1 \
      --min-seg-tokens "${DOC_TOKENS_LDA}" --max-seg-tokens "${DOC_TOKENS_LDA}" \
      --fixed-leaf-tokens "${LT}" \
      --alpha-topic 0.20 --beta-word 0.10 \
      --segment-concentration 80.0 --segment-background 2.0 \
      --topic-phi-permute \
      --eval-internal-query-design risk \
      --seeds "${SEEDS}" \
      --skip-existing
  done
done

: > "${CMD_LDA_BASE}"
for LT in ${LEAF_TOKENS_LDA}; do
  for Q in ${Q_RATES}; do
    append_built_cmds "${CMD_LDA_BASE}" \
      --output-root "${OUT_ROOT}/segmented_lda_ctreepo/equivalence/lda/k8_v512/lane_phi_base" \
      --train-docs "${TRAIN_DOCS_LDA}" \
      --n-books-test "${N_BOOKS_TEST_LDA}" \
      --calibration-rates "${CAL_RATES_LDA}" \
      --eval-leaf-rates "${Q}" \
      --eval-internal-rates "${Q}" \
      --topic-process bag_of_words \
      --leaf-theta-estimator lstsq \
      --topic-phi-estimators "tensor_lda" \
      --topic-phi-docs 0 \
      --n-topics 8 --vocab-size 512 \
      --min-segments 1 --max-segments 1 \
      --min-seg-tokens "${DOC_TOKENS_LDA}" --max-seg-tokens "${DOC_TOKENS_LDA}" \
      --fixed-leaf-tokens "${LT}" \
      --alpha-topic 0.20 --beta-word 0.10 \
      --segment-concentration 80.0 --segment-background 2.0 \
      --topic-phi-permute \
      --eval-internal-query-design risk \
      --seeds "${SEEDS}" \
      --skip-existing
  done
done

: > "${CMD_LDA_NEURAL_WEAK}"
for LT in ${LEAF_TOKENS_LDA}; do
  for Q in ${Q_RATES}; do
    append_built_cmds "${CMD_LDA_NEURAL_WEAK}" \
      --output-root "${OUT_ROOT}/segmented_lda_ctreepo/equivalence/lda/k8_v512/lane_neural_weak" \
      --train-docs "${TRAIN_DOCS_LDA}" \
      --n-books-test "${N_BOOKS_TEST_LDA}" \
      --calibration-rates "${CAL_RATES_LDA}" \
      --eval-leaf-rates "${Q}" \
      --eval-internal-rates "${Q}" \
      --topic-process bag_of_words \
      --leaf-theta-estimator lstsq \
      --topic-phi-estimators "neural_ctreepo" \
      --topic-phi-docs 0 \
      --n-topics 8 --vocab-size 512 \
      --min-segments 1 --max-segments 1 \
      --min-seg-tokens "${DOC_TOKENS_LDA}" --max-seg-tokens "${DOC_TOKENS_LDA}" \
      --fixed-leaf-tokens "${LT}" \
      --alpha-topic 0.20 --beta-word 0.10 \
      --segment-concentration 80.0 --segment-background 2.0 \
      --neural-topic-base-estimator tensor_lda \
      --neural-topic-seed-fraction 0.125 \
      --neural-topic-operator-boost 0.6 \
      --neural-topic-seed-llm-min-weight 0.02 \
      --neural-topic-seed-llm-max-weight 0.15 \
      --neural-topic-mix-samples 64 \
      --topic-phi-permute \
      --eval-internal-query-design risk \
      --seeds "${SEEDS}" \
      --skip-existing
  done
done

: > "${CMD_LDA_NEURAL_DEFAULT}"
for LT in ${LEAF_TOKENS_LDA}; do
  for Q in ${Q_RATES}; do
    append_built_cmds "${CMD_LDA_NEURAL_DEFAULT}" \
      --output-root "${OUT_ROOT}/segmented_lda_ctreepo/equivalence/lda/k8_v512/lane_neural_default" \
      --train-docs "${TRAIN_DOCS_LDA}" \
      --n-books-test "${N_BOOKS_TEST_LDA}" \
      --calibration-rates "${CAL_RATES_LDA}" \
      --eval-leaf-rates "${Q}" \
      --eval-internal-rates "${Q}" \
      --topic-process bag_of_words \
      --leaf-theta-estimator lstsq \
      --topic-phi-estimators "neural_ctreepo" \
      --topic-phi-docs 0 \
      --n-topics 8 --vocab-size 512 \
      --min-segments 1 --max-segments 1 \
      --min-seg-tokens "${DOC_TOKENS_LDA}" --max-seg-tokens "${DOC_TOKENS_LDA}" \
      --fixed-leaf-tokens "${LT}" \
      --alpha-topic 0.20 --beta-word 0.10 \
      --segment-concentration 80.0 --segment-background 2.0 \
      --neural-topic-base-estimator tensor_lda \
      --neural-topic-seed-fractions "0.25 0.5" \
      --neural-topic-operator-boost 1.0 \
      --neural-topic-seed-llm-min-weight 0.10 \
      --neural-topic-seed-llm-max-weight 0.35 \
      --neural-topic-mix-samples 128 \
      --topic-phi-permute \
      --eval-internal-query-design risk \
      --seeds "${SEEDS}" \
      --skip-existing
  done
done

echo "=== Building commands: hard regime (k=12,v=1024,segments) ===" | tee -a "${RUN_LOG}"
: > "${CMD_HARD_BASE}"
for LT in ${LEAF_TOKENS_HARD}; do
  for Q in ${Q_RATES}; do
    append_built_cmds "${CMD_HARD_BASE}" \
      --output-root "${OUT_ROOT}/segmented_lda_ctreepo/equivalence/hard/k12_v1024/lane_phi_base" \
      --train-docs "${TRAIN_DOCS_HARD}" \
      --n-books-test "${N_BOOKS_TEST_HARD}" \
      --calibration-rates "${CAL_RATES_HARD}" \
      --eval-leaf-rates "${Q}" \
      --eval-internal-rates "${Q}" \
      --topic-process segments \
      --leaf-theta-estimator lstsq \
      --topic-phi-estimators "tensor_lda" \
      --topic-phi-docs 0 \
      --n-topics 12 --vocab-size 1024 \
      --min-segments 10 --max-segments 12 \
      --min-seg-tokens 16 --max-seg-tokens 32 \
      --fixed-leaf-tokens "${LT}" \
      --alpha-topic 0.35 --beta-word 0.40 \
      --segment-concentration 18.0 --segment-background 6.0 \
      --topic-phi-permute \
      --eval-internal-query-design risk \
      --seeds "${SEEDS}" \
      --skip-existing
  done
done

: > "${CMD_HARD_NEURAL_WEAK}"
for LT in ${LEAF_TOKENS_HARD}; do
  for Q in ${Q_RATES}; do
    append_built_cmds "${CMD_HARD_NEURAL_WEAK}" \
      --output-root "${OUT_ROOT}/segmented_lda_ctreepo/equivalence/hard/k12_v1024/lane_neural_weak" \
      --train-docs "${TRAIN_DOCS_HARD}" \
      --n-books-test "${N_BOOKS_TEST_HARD}" \
      --calibration-rates "${CAL_RATES_HARD}" \
      --eval-leaf-rates "${Q}" \
      --eval-internal-rates "${Q}" \
      --topic-process segments \
      --leaf-theta-estimator lstsq \
      --topic-phi-estimators "neural_ctreepo" \
      --topic-phi-docs 0 \
      --n-topics 12 --vocab-size 1024 \
      --min-segments 10 --max-segments 12 \
      --min-seg-tokens 16 --max-seg-tokens 32 \
      --fixed-leaf-tokens "${LT}" \
      --alpha-topic 0.35 --beta-word 0.40 \
      --segment-concentration 18.0 --segment-background 6.0 \
      --neural-topic-base-estimator tensor_lda \
      --neural-topic-seed-fraction 0.0833333333 \
      --neural-topic-operator-boost 0.6 \
      --neural-topic-seed-llm-min-weight 0.02 \
      --neural-topic-seed-llm-max-weight 0.15 \
      --neural-topic-mix-samples 64 \
      --topic-phi-permute \
      --eval-internal-query-design risk \
      --seeds "${SEEDS}" \
      --skip-existing
  done
done

: > "${CMD_HARD_NEURAL_DEFAULT}"
for LT in ${LEAF_TOKENS_HARD}; do
  for Q in ${Q_RATES}; do
    append_built_cmds "${CMD_HARD_NEURAL_DEFAULT}" \
      --output-root "${OUT_ROOT}/segmented_lda_ctreepo/equivalence/hard/k12_v1024/lane_neural_default" \
      --train-docs "${TRAIN_DOCS_HARD}" \
      --n-books-test "${N_BOOKS_TEST_HARD}" \
      --calibration-rates "${CAL_RATES_HARD}" \
      --eval-leaf-rates "${Q}" \
      --eval-internal-rates "${Q}" \
      --topic-process segments \
      --leaf-theta-estimator lstsq \
      --topic-phi-estimators "neural_ctreepo" \
      --topic-phi-docs 0 \
      --n-topics 12 --vocab-size 1024 \
      --min-segments 10 --max-segments 12 \
      --min-seg-tokens 16 --max-seg-tokens 32 \
      --fixed-leaf-tokens "${LT}" \
      --alpha-topic 0.35 --beta-word 0.40 \
      --segment-concentration 18.0 --segment-background 6.0 \
      --neural-topic-base-estimator tensor_lda \
      --neural-topic-seed-fractions "0.2 0.35" \
      --neural-topic-operator-boost 1.0 \
      --neural-topic-seed-llm-min-weight 0.10 \
      --neural-topic-seed-llm-max-weight 0.35 \
      --neural-topic-mix-samples 128 \
      --topic-phi-permute \
      --eval-internal-query-design risk \
      --seeds "${SEEDS}" \
      --skip-existing
  done
done

: > "${CMD_HARD_NEURAL_UPPER}"
for LT in ${LEAF_TOKENS_HARD}; do
  for Q in ${Q_RATES_UPPER}; do
    append_built_cmds "${CMD_HARD_NEURAL_UPPER}" \
      --output-root "${OUT_ROOT}/segmented_lda_ctreepo/equivalence/hard/k12_v1024/lane_neural_upper" \
      --train-docs "${TRAIN_DOCS_HARD_UPPER}" \
      --n-books-test "${N_BOOKS_TEST_HARD}" \
      --calibration-rates "${CAL_RATES_UPPER}" \
      --eval-leaf-rates "${Q}" \
      --eval-internal-rates "${Q}" \
      --topic-process segments \
      --leaf-theta-estimator lstsq \
      --topic-phi-estimators "neural_ctreepo" \
      --topic-phi-docs 0 \
      --n-topics 12 --vocab-size 1024 \
      --min-segments 10 --max-segments 12 \
      --min-seg-tokens 16 --max-seg-tokens 32 \
      --fixed-leaf-tokens "${LT}" \
      --alpha-topic 0.35 --beta-word 0.40 \
      --segment-concentration 18.0 --segment-background 6.0 \
      --neural-topic-base-estimator tensor_lda \
      --neural-topic-seed-fraction 1.0 \
      --neural-topic-operator-boost 1.4 \
      --neural-topic-seed-llm-min-weight 0.35 \
      --neural-topic-seed-llm-max-weight 0.85 \
      --neural-topic-mix-samples 128 \
      --topic-phi-permute \
      --eval-internal-query-design risk \
      --seeds "${SEEDS}" \
      --skip-existing
  done
done

{
  cat "${CMD_LDA_DIRECT}"
  cat "${CMD_LDA_BASE}"
  cat "${CMD_LDA_NEURAL_WEAK}"
  cat "${CMD_LDA_NEURAL_DEFAULT}"
  cat "${CMD_HARD_BASE}"
  cat "${CMD_HARD_NEURAL_WEAK}"
  cat "${CMD_HARD_NEURAL_DEFAULT}"
  cat "${CMD_HARD_NEURAL_UPPER}"
} > "${CMD_ALL}"

echo "=== Command counts ===" | tee -a "${RUN_LOG}"
echo "lda_direct=$(count_cmds "${CMD_LDA_DIRECT}")" | tee -a "${RUN_LOG}"
echo "lda_phi_base=$(count_cmds "${CMD_LDA_BASE}")" | tee -a "${RUN_LOG}"
echo "lda_neural_weak=$(count_cmds "${CMD_LDA_NEURAL_WEAK}")" | tee -a "${RUN_LOG}"
echo "lda_neural_default=$(count_cmds "${CMD_LDA_NEURAL_DEFAULT}")" | tee -a "${RUN_LOG}"
echo "hard_phi_base=$(count_cmds "${CMD_HARD_BASE}")" | tee -a "${RUN_LOG}"
echo "hard_neural_weak=$(count_cmds "${CMD_HARD_NEURAL_WEAK}")" | tee -a "${RUN_LOG}"
echo "hard_neural_default=$(count_cmds "${CMD_HARD_NEURAL_DEFAULT}")" | tee -a "${RUN_LOG}"
echo "hard_neural_upper=$(count_cmds "${CMD_HARD_NEURAL_UPPER}")" | tee -a "${RUN_LOG}"
echo "all=$(count_cmds "${CMD_ALL}")" | tee -a "${RUN_LOG}"
echo "cmd_manifest=${CMD_ALL}" | tee -a "${RUN_LOG}"

if [[ "${BUILD_ONLY}" != "0" ]]; then
  echo "BUILD_ONLY=1 -> exiting after command generation." | tee -a "${RUN_LOG}"
  exit 0
fi

if [[ "${RUN_COMBINED}" != "0" ]]; then
  echo "=== RUN: combined mixed-resource queue | n_cmds=$(count_cmds "${CMD_ALL}") ===" | tee -a "${RUN_LOG}"
  venv/bin/python scripts/run_simulation_resource_queue.py \
    --cmd-file "${CMD_ALL}" \
    --cpu-workers "${JOBS}" \
    --gpu-tokens "${GPU_TOKENS}" \
    --log-dir "${CMDS_DIR}/all_queue_logs" | tee -a "${RUN_LOG}"
else
  run_stage "LDA direct baseline (sklearn_lda phi + sklearn_lda theta)" "lda_direct" "${CMD_LDA_DIRECT}"
  run_stage "LDA regime base phi estimators (tensor_lda, spectral_numpy)" "lda_base" "${CMD_LDA_BASE}"
  run_stage "LDA regime neural_ctreepo weak oracle lane" "lda_neural_weak" "${CMD_LDA_NEURAL_WEAK}"
  run_stage "LDA regime neural_ctreepo default oracle lane" "lda_neural_default" "${CMD_LDA_NEURAL_DEFAULT}"
  run_stage "Hard regime base phi estimators (tensor_lda, spectral_numpy)" "hard_base" "${CMD_HARD_BASE}"
  run_stage "Hard regime neural_ctreepo weak oracle lane" "hard_neural_weak" "${CMD_HARD_NEURAL_WEAK}"
  run_stage "Hard regime neural_ctreepo default oracle lane" "hard_neural_default" "${CMD_HARD_NEURAL_DEFAULT}"
  run_stage "Hard regime neural_ctreepo upper-oracle stress lane" "hard_neural_upper" "${CMD_HARD_NEURAL_UPPER}"
fi

echo "DONE | OUT_ROOT=${OUT_ROOT}" | tee -a "${RUN_LOG}"
