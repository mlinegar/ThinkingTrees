#!/usr/bin/env bash
set -euo pipefail

# Overnight CPU sweep for "Identifiable-Zero Learnability Benchmarks (v1)".
#
# Design intent:
# - Clean learning surfaces: performance vs (train size × oracle-label rate)
# - Held-out evaluation with fixed test sets across training-size sweeps
# - Simple external baselines only (RandomForest + tiny MLP)
#
# Override via env vars:
#   JOBS=64 OUT_ROOT=outputs/... STAMP=... ./scripts/run_identifiable_zero_learnability_overnight.sh

JOBS="${JOBS:-128}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/identifiable_zero_learnability_v1_${STAMP}}"
HERO="${HERO:-1}"
MARKOV_DEVICE="${MARKOV_DEVICE:-auto}"
GPU_TOKENS="${GPU_TOKENS:-auto}"
TRAIN_DOCS_GRID="${TRAIN_DOCS_GRID:-500 1000 2000 4000 8000}"
LABEL_RATE_GRID="${LABEL_RATE_GRID:-0.02 0.05 0.1 0.2 0.4}"
HELDOUT_DOCS="${HELDOUT_DOCS:-2000}"
CTREE_EVAL_GUIDANCE_RATES="${CTREE_EVAL_GUIDANCE_RATES:-0}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

RUN_LOG="${LOG_DIR}/${STAMP}_identifiable_zero_learnability_run.log"

MARKOV_BASE_OUT="${OUT_ROOT}/markov_changepoint_ops_count/equivalence/baseline"
MARKOV_HARD_OUT="${OUT_ROOT}/markov_changepoint_ops_count/equivalence/hard"
CTREE_BASE_OUT="${OUT_ROOT}/segmented_lda_ctreepo/equivalence/baseline"
CTREE_HARD_OUT="${OUT_ROOT}/segmented_lda_ctreepo/equivalence/hard"
CTREE_LDA_OUT="${OUT_ROOT}/segmented_lda_ctreepo/equivalence/lda"

MARKOV_BASE_CMDS="${LOG_DIR}/${STAMP}_markov_learnability_baseline_cmds.txt"
MARKOV_HARD_CMDS="${LOG_DIR}/${STAMP}_markov_learnability_hard_cmds.txt"
MARKOV_HARD_HERO_CMDS="${LOG_DIR}/${STAMP}_markov_learnability_hard_hero_cmds.txt"
CTREE_BASE_LSTSQ_CMDS="${LOG_DIR}/${STAMP}_ctree_learnability_baseline_lstsq_cmds.txt"
CTREE_BASE_THETA_CMDS="${LOG_DIR}/${STAMP}_ctree_learnability_baseline_theta_cmds.txt"
CTREE_HARD_LSTSQ_CMDS="${LOG_DIR}/${STAMP}_ctree_learnability_hard_lstsq_cmds.txt"
CTREE_HARD_THETA_CMDS="${LOG_DIR}/${STAMP}_ctree_learnability_hard_theta_cmds.txt"
CTREE_HARD_HERO_LSTSQ_CMDS="${LOG_DIR}/${STAMP}_ctree_learnability_hard_hero_lstsq_cmds.txt"
CTREE_HARD_HERO_THETA_CMDS="${LOG_DIR}/${STAMP}_ctree_learnability_hard_hero_theta_cmds.txt"
CTREE_LDA_LSTSQ_CMDS="${LOG_DIR}/${STAMP}_ctree_learnability_lda_lstsq_cmds.txt"
CTREE_LDA_THETA_CMDS="${LOG_DIR}/${STAMP}_ctree_learnability_lda_theta_cmds.txt"
CTREE_TMP_CMDS="${LOG_DIR}/${STAMP}_ctree_learnability_tmp_cmds.txt"

echo "OUT_ROOT=${OUT_ROOT}"
echo "JOBS=${JOBS}"
echo "RUN_LOG=${RUN_LOG}"
echo "MARKOV_DEVICE=${MARKOV_DEVICE}"
echo "GPU_TOKENS=${GPU_TOKENS}"
echo "TRAIN_DOCS_GRID=${TRAIN_DOCS_GRID}"
echo "LABEL_RATE_GRID=${LABEL_RATE_GRID}"
echo "HELDOUT_DOCS=${HELDOUT_DOCS}"
echo "CTREE_EVAL_GUIDANCE_RATES=${CTREE_EVAL_GUIDANCE_RATES}"

run_queue() {
  local slug="$1"
  shift
  local queue_log_dir="${LOG_DIR}/${STAMP}_${slug}_queue_logs"
  local args=(venv/bin/python scripts/run_simulation_resource_queue.py --cpu-workers "${JOBS}" --gpu-tokens "${GPU_TOKENS}" --log-dir "${queue_log_dir}")
  local cmd_file=""
  for cmd_file in "$@"; do
    args+=(--cmd-file "${cmd_file}")
  done
  "${args[@]}" | tee -a "${RUN_LOG}"
}

echo "=== Building Markov learnability commands (baseline regime) ===" | tee "${RUN_LOG}"
venv/bin/python -u scripts/build_markov_changepoint_ops_count_cmds.py \
  --out-cmds "${MARKOV_BASE_CMDS}" \
  --output-root "${MARKOV_BASE_OUT}" \
  --n-regimes 4 \
  --vocab-size 96 \
  --min-tokens 384 --max-tokens 384 \
  --min-segments 12 --max-segments 24 \
  --fixed-leaf-tokens 16 \
  --train-docs "${TRAIN_DOCS_GRID}" \
  --test-docs "${HELDOUT_DOCS}" \
  --model-family "neural additive" \
  --audit-fractions "${LABEL_RATE_GRID}" \
  --leaf-query-rates "1.0" \
  --include-root-query "true" \
  --schedule-consistency-weights "0.0" \
  --guidance-override-modes "reset" \
  --include-rf-root-baseline \
  --device "${MARKOV_DEVICE}" \
  --torch-threads 1 \
  --n-epochs 10 \
  --seeds "0 1 2 3 4 5" \
  --skip-existing

echo "Markov baseline commands: ${MARKOV_BASE_CMDS}" | tee -a "${RUN_LOG}"

echo "=== Building Markov learnability commands (hard regime) ===" | tee -a "${RUN_LOG}"
venv/bin/python -u scripts/build_markov_changepoint_ops_count_cmds.py \
  --out-cmds "${MARKOV_HARD_CMDS}" \
  --output-root "${MARKOV_HARD_OUT}" \
  --n-regimes 6 \
  --vocab-size 128 \
  --min-tokens 768 --max-tokens 768 \
  --min-segments 24 --max-segments 48 \
  --fixed-leaf-tokens 16 \
  --train-docs "${TRAIN_DOCS_GRID}" \
  --test-docs "${HELDOUT_DOCS}" \
  --model-family "neural additive" \
  --audit-fractions "${LABEL_RATE_GRID}" \
  --leaf-query-rates "1.0" \
  --include-root-query "true" \
  --schedule-consistency-weights "0.0" \
  --guidance-override-modes "reset" \
  --include-rf-root-baseline \
  --device "${MARKOV_DEVICE}" \
  --torch-threads 1 \
  --n-epochs 10 \
  --seeds "0 1 2 3 4 5" \
  --skip-existing

echo "Markov hard commands: ${MARKOV_HARD_CMDS}" | tee -a "${RUN_LOG}"

if [[ "${HERO}" != "0" ]]; then
  echo "=== Building Markov learnability HERO slice (hard regime; extra seeds 6..11) ===" | tee -a "${RUN_LOG}"
  venv/bin/python -u scripts/build_markov_changepoint_ops_count_cmds.py \
    --out-cmds "${MARKOV_HARD_HERO_CMDS}" \
    --output-root "${MARKOV_HARD_OUT}" \
    --n-regimes 6 \
    --vocab-size 128 \
    --min-tokens 768 --max-tokens 768 \
    --min-segments 24 --max-segments 48 \
    --fixed-leaf-tokens 16 \
    --train-docs "8000" \
    --test-docs "${HELDOUT_DOCS}" \
    --model-family "neural additive" \
    --audit-fractions "0.05 0.1 0.2 0.4" \
    --leaf-query-rates "1.0" \
    --include-root-query "true" \
    --schedule-consistency-weights "0.0" \
    --guidance-override-modes "reset" \
    --include-rf-root-baseline \
    --device "${MARKOV_DEVICE}" \
    --torch-threads 1 \
    --n-epochs 10 \
    --seeds "6 7 8 9 10 11" \
    --skip-existing

  echo "Markov hard HERO commands: ${MARKOV_HARD_HERO_CMDS}" | tee -a "${RUN_LOG}"
else
  : > "${MARKOV_HARD_HERO_CMDS}"
fi

echo "=== Building C-TreePO learnability commands (baseline regime; lstsq θ) ===" | tee -a "${RUN_LOG}"
: > "${CTREE_BASE_LSTSQ_CMDS}"
for Q in ${CTREE_EVAL_GUIDANCE_RATES}; do
  venv/bin/python -u scripts/build_segmented_lda_ctreepo_cmds.py \
    --out-cmds "${CTREE_TMP_CMDS}" \
    --output-root "${CTREE_BASE_OUT}" \
    --train-docs "${TRAIN_DOCS_GRID}" \
    --n-books-test "${HELDOUT_DOCS}" \
    --calibration-rates "${LABEL_RATE_GRID}" \
    --eval-leaf-rates "${Q}" \
    --eval-internal-rates "${Q}" \
    --leaf-theta-estimator lstsq \
    --topic-phi-estimators "spectral_numpy embedding_spectral" \
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
  cat "${CTREE_TMP_CMDS}" >> "${CTREE_BASE_LSTSQ_CMDS}"
done

echo "C-TreePO baseline lstsq commands: ${CTREE_BASE_LSTSQ_CMDS}" | tee -a "${RUN_LOG}"

echo "=== Building C-TreePO learnability commands (baseline regime; supervised θ) ===" | tee -a "${RUN_LOG}"
: > "${CTREE_BASE_THETA_CMDS}"
for Q in ${CTREE_EVAL_GUIDANCE_RATES}; do
  venv/bin/python -u scripts/build_segmented_lda_ctreepo_cmds.py \
    --out-cmds "${CTREE_TMP_CMDS}" \
    --output-root "${CTREE_BASE_OUT}" \
    --train-docs "${TRAIN_DOCS_GRID}" \
    --n-books-test "${HELDOUT_DOCS}" \
    --calibration-rates "${LABEL_RATE_GRID}" \
    --eval-leaf-rates "${Q}" \
    --eval-internal-rates "${Q}" \
    --leaf-theta-estimators "rf mlp" \
    --topic-phi-estimators "spectral_numpy" \
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
  cat "${CTREE_TMP_CMDS}" >> "${CTREE_BASE_THETA_CMDS}"
done

echo "C-TreePO baseline supervised-θ commands: ${CTREE_BASE_THETA_CMDS}" | tee -a "${RUN_LOG}"

echo "=== Building C-TreePO learnability commands (hard regime; lstsq θ) ===" | tee -a "${RUN_LOG}"
: > "${CTREE_HARD_LSTSQ_CMDS}"
for Q in ${CTREE_EVAL_GUIDANCE_RATES}; do
  venv/bin/python -u scripts/build_segmented_lda_ctreepo_cmds.py \
    --out-cmds "${CTREE_TMP_CMDS}" \
    --output-root "${CTREE_HARD_OUT}" \
    --train-docs "${TRAIN_DOCS_GRID}" \
    --n-books-test "${HELDOUT_DOCS}" \
    --calibration-rates "${LABEL_RATE_GRID}" \
    --eval-leaf-rates "${Q}" \
    --eval-internal-rates "${Q}" \
    --leaf-theta-estimator lstsq \
    --topic-phi-estimators "spectral_numpy embedding_spectral" \
    --topic-phi-docs 0 \
    --n-topics 8 --vocab-size 512 \
    --min-segments 8 --max-segments 8 \
    --min-seg-tokens 16 --max-seg-tokens 32 \
    --fixed-leaf-tokens 16 \
    --alpha-topic 0.30 --beta-word 0.30 \
    --segment-concentration 20.0 --segment-background 5.0 \
    --topic-phi-permute \
    --eval-internal-query-design risk \
    --seeds "0 1 2 3 4 5" \
    --skip-existing
  cat "${CTREE_TMP_CMDS}" >> "${CTREE_HARD_LSTSQ_CMDS}"
done

echo "C-TreePO hard lstsq commands: ${CTREE_HARD_LSTSQ_CMDS}" | tee -a "${RUN_LOG}"

echo "=== Building C-TreePO learnability commands (hard regime; supervised θ) ===" | tee -a "${RUN_LOG}"
: > "${CTREE_HARD_THETA_CMDS}"
for Q in ${CTREE_EVAL_GUIDANCE_RATES}; do
  venv/bin/python -u scripts/build_segmented_lda_ctreepo_cmds.py \
    --out-cmds "${CTREE_TMP_CMDS}" \
    --output-root "${CTREE_HARD_OUT}" \
    --train-docs "${TRAIN_DOCS_GRID}" \
    --n-books-test "${HELDOUT_DOCS}" \
    --calibration-rates "${LABEL_RATE_GRID}" \
    --eval-leaf-rates "${Q}" \
    --eval-internal-rates "${Q}" \
    --leaf-theta-estimators "rf mlp" \
    --topic-phi-estimators "spectral_numpy" \
    --topic-phi-docs 0 \
    --n-topics 8 --vocab-size 512 \
    --min-segments 8 --max-segments 8 \
    --min-seg-tokens 16 --max-seg-tokens 32 \
    --fixed-leaf-tokens 16 \
    --alpha-topic 0.30 --beta-word 0.30 \
    --segment-concentration 20.0 --segment-background 5.0 \
    --topic-phi-permute \
    --eval-internal-query-design risk \
    --seeds "0 1 2 3 4 5" \
    --skip-existing
  cat "${CTREE_TMP_CMDS}" >> "${CTREE_HARD_THETA_CMDS}"
done

echo "C-TreePO hard supervised-θ commands: ${CTREE_HARD_THETA_CMDS}" | tee -a "${RUN_LOG}"

echo "=== Building C-TreePO learnability commands (regular LDA DGP; bag_of_words) ===" | tee -a "${RUN_LOG}"
: > "${CTREE_LDA_LSTSQ_CMDS}"
for Q in ${CTREE_EVAL_GUIDANCE_RATES}; do
  venv/bin/python -u scripts/build_segmented_lda_ctreepo_cmds.py \
    --out-cmds "${CTREE_TMP_CMDS}" \
    --output-root "${CTREE_LDA_OUT}" \
    --train-docs "${TRAIN_DOCS_GRID}" \
    --n-books-test "${HELDOUT_DOCS}" \
    --calibration-rates "${LABEL_RATE_GRID}" \
    --eval-leaf-rates "${Q}" \
    --eval-internal-rates "${Q}" \
    --topic-process bag_of_words \
    --leaf-theta-estimator lstsq \
    --topic-phi-estimators "spectral_numpy embedding_spectral tensor_lda sklearn_lda" \
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
  cat "${CTREE_TMP_CMDS}" >> "${CTREE_LDA_LSTSQ_CMDS}"
done

: > "${CTREE_LDA_THETA_CMDS}"
for Q in ${CTREE_EVAL_GUIDANCE_RATES}; do
  venv/bin/python -u scripts/build_segmented_lda_ctreepo_cmds.py \
    --out-cmds "${CTREE_TMP_CMDS}" \
    --output-root "${CTREE_LDA_OUT}" \
    --train-docs "${TRAIN_DOCS_GRID}" \
    --n-books-test "${HELDOUT_DOCS}" \
    --calibration-rates "${LABEL_RATE_GRID}" \
    --eval-leaf-rates "${Q}" \
    --eval-internal-rates "${Q}" \
    --topic-process bag_of_words \
    --leaf-theta-estimators "rf mlp" \
    --topic-phi-estimators "spectral_numpy" \
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
  cat "${CTREE_TMP_CMDS}" >> "${CTREE_LDA_THETA_CMDS}"
done

echo "C-TreePO LDA-DGP lstsq commands: ${CTREE_LDA_LSTSQ_CMDS}" | tee -a "${RUN_LOG}"
echo "C-TreePO LDA-DGP supervised-θ commands: ${CTREE_LDA_THETA_CMDS}" | tee -a "${RUN_LOG}"

if [[ "${HERO}" != "0" ]]; then
  echo "=== Building C-TreePO learnability HERO slice (hard regime; extra seeds 6..11) ===" | tee -a "${RUN_LOG}"

  : > "${CTREE_HARD_HERO_LSTSQ_CMDS}"
  venv/bin/python -u scripts/build_segmented_lda_ctreepo_cmds.py \
    --out-cmds "${CTREE_TMP_CMDS}" \
    --output-root "${CTREE_HARD_OUT}" \
    --train-docs "8000" \
    --n-books-test "${HELDOUT_DOCS}" \
    --calibration-rates "0.05 0.1 0.2" \
    --eval-leaf-rates "0" \
    --eval-internal-rates "0" \
    --leaf-theta-estimator lstsq \
    --topic-phi-estimators "spectral_numpy" \
    --topic-phi-docs 0 \
    --n-topics 8 --vocab-size 512 \
    --min-segments 8 --max-segments 8 \
    --min-seg-tokens 16 --max-seg-tokens 32 \
    --fixed-leaf-tokens 16 \
    --alpha-topic 0.30 --beta-word 0.30 \
    --segment-concentration 20.0 --segment-background 5.0 \
    --topic-phi-permute \
    --eval-internal-query-design risk \
    --seeds "6 7 8 9 10 11" \
    --skip-existing
  cat "${CTREE_TMP_CMDS}" >> "${CTREE_HARD_HERO_LSTSQ_CMDS}"

  : > "${CTREE_HARD_HERO_THETA_CMDS}"
  venv/bin/python -u scripts/build_segmented_lda_ctreepo_cmds.py \
    --out-cmds "${CTREE_TMP_CMDS}" \
    --output-root "${CTREE_HARD_OUT}" \
    --train-docs "8000" \
    --n-books-test "${HELDOUT_DOCS}" \
    --calibration-rates "0.05 0.1 0.2" \
    --eval-leaf-rates "0" \
    --eval-internal-rates "0" \
    --leaf-theta-estimators "rf mlp" \
    --topic-phi-estimators "spectral_numpy" \
    --topic-phi-docs 0 \
    --n-topics 8 --vocab-size 512 \
    --min-segments 8 --max-segments 8 \
    --min-seg-tokens 16 --max-seg-tokens 32 \
    --fixed-leaf-tokens 16 \
    --alpha-topic 0.30 --beta-word 0.30 \
    --segment-concentration 20.0 --segment-background 5.0 \
    --topic-phi-permute \
    --eval-internal-query-design risk \
    --seeds "6 7 8 9 10 11" \
    --skip-existing
  cat "${CTREE_TMP_CMDS}" >> "${CTREE_HARD_HERO_THETA_CMDS}"

  echo "C-TreePO hard HERO lstsq commands: ${CTREE_HARD_HERO_LSTSQ_CMDS}" | tee -a "${RUN_LOG}"
  echo "C-TreePO hard HERO supervised-θ commands: ${CTREE_HARD_HERO_THETA_CMDS}" | tee -a "${RUN_LOG}"
else
  : > "${CTREE_HARD_HERO_LSTSQ_CMDS}"
  : > "${CTREE_HARD_HERO_THETA_CMDS}"
fi

echo "=== RUN: Markov learnability (baseline) ===" | tee -a "${RUN_LOG}"
run_queue markov_learnability_baseline "${MARKOV_BASE_CMDS}"

echo "=== RUN: Markov learnability (hard) ===" | tee -a "${RUN_LOG}"
run_queue markov_learnability_hard "${MARKOV_HARD_CMDS}" "${MARKOV_HARD_HERO_CMDS}"

echo "=== RUN: C-TreePO learnability (baseline) ===" | tee -a "${RUN_LOG}"
run_queue ctree_learnability_baseline "${CTREE_BASE_LSTSQ_CMDS}" "${CTREE_BASE_THETA_CMDS}"

echo "=== RUN: C-TreePO learnability (hard) ===" | tee -a "${RUN_LOG}"
run_queue ctree_learnability_hard "${CTREE_HARD_LSTSQ_CMDS}" "${CTREE_HARD_THETA_CMDS}" "${CTREE_HARD_HERO_LSTSQ_CMDS}" "${CTREE_HARD_HERO_THETA_CMDS}"

echo "=== RUN: C-TreePO learnability (regular LDA DGP) ===" | tee -a "${RUN_LOG}"
run_queue ctree_learnability_lda "${CTREE_LDA_LSTSQ_CMDS}" "${CTREE_LDA_THETA_CMDS}"

echo "DONE | OUT_ROOT=${OUT_ROOT}" | tee -a "${RUN_LOG}"
