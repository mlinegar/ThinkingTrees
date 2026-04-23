#!/usr/bin/env bash
# Rebuild Benoit-aligned teacher traces from existing economic summaries, then
# run a deeper DSPy f/g alternating ladder.

set -euo pipefail

ROOT="${1:-outputs/manifesto_fg_alternating/economic_benoit_existing_summary_dspy_medium_$(date +%Y%m%d_%H%M%S)}"
TEACHER_DIR="${ROOT}/teacher"
LADDER_DIR="${ROOT}/ladder"
TEACHER_LOG="${ROOT}/teacher.log"
LADDER_LOG="${ROOT}/ladder.log"
DSPY_NUM_THREADS="${DSPY_NUM_THREADS:-8}"
NOFILE_LIMIT="${NOFILE_LIMIT:-65535}"
LEAF_SIZE_TOKENS="${LEAF_SIZE_TOKENS:-256,512}"
MAX_ITERATIONS="${MAX_ITERATIONS:-4}"
FIRST_TRAIN_SIDE="${FIRST_TRAIN_SIDE:-f}"
INITIAL_F_DEGREE="${INITIAL_F_DEGREE:-1}"
INITIAL_G_DEGREE="${INITIAL_G_DEGREE:-1}"
STAGE_NAMING="${STAGE_NAMING:-legacy}"
SUMMARY_MAX_TOKENS="${SUMMARY_MAX_TOKENS:-0}"
RESUMMARY_MAX_TOKENS="${RESUMMARY_MAX_TOKENS:-0}"
DSPY_MAX_TOKENS="${DSPY_MAX_TOKENS:-2048}"
LM_CONTEXT_TOKENS="${LM_CONTEXT_TOKENS:-32768}"
SCORE_MAX_CHARS="${SCORE_MAX_CHARS:-24000}"
NODE_SUMMARY_MAX_CHARS="${NODE_SUMMARY_MAX_CHARS:-32000}"
RESUMMARY_MAX_CHARS="${RESUMMARY_MAX_CHARS:-24000}"
DSPY_PROMPT_OVERHEAD_TOKENS="${DSPY_PROMPT_OVERHEAD_TOKENS:-1500}"
SOURCE_RESULTS="${SOURCE_RESULTS:-outputs/overnight_benoit/full_pipeline/economic/per_manifesto.jsonl}"
SPLIT_SOURCE="${SPLIT_SOURCE:-results-order}"
TREE_TEXT_SOURCE="${TREE_TEXT_SOURCE:-existing_summary}"
TEACHER_SUMMARY_MODE="${TEACHER_SUMMARY_MODE:-teacher}"
TEACHER_SUMMARY_TEMPERATURE="${TEACHER_SUMMARY_TEMPERATURE:-0.0}"
TEACHER_IDEMPOTENCE_MODE="${TEACHER_IDEMPOTENCE_MODE:-off}"
TEACHER_SCORE_INPUT="${TEACHER_SCORE_INPUT:-teacher_summary}"
TEACHER_MISSING_SCORE_POLICY="${TEACHER_MISSING_SCORE_POLICY:-neutral}"
TRAIN_N="${TRAIN_N:-140}"
VAL_N="${VAL_N:-30}"
TEST_N="${TEST_N:-48}"

ulimit -n "${NOFILE_LIMIT}" 2>/dev/null || true

mkdir -p "${ROOT}"

leaf_traces_exist=1
IFS=',' read -r -a leaf_size_token_values <<< "${LEAF_SIZE_TOKENS}"
for raw_leaf_size in "${leaf_size_token_values[@]}"; do
  leaf_size="$(echo "${raw_leaf_size}" | xargs)"
  if [[ -z "${leaf_size}" ]]; then
    continue
  fi
  leaf_dir="$(printf 'leaf%04dtok' "${leaf_size}")"
  if [[ ! -f "${TEACHER_DIR}/${leaf_dir}/labeled_trees.jsonl" ]]; then
    leaf_traces_exist=0
    break
  fi
done

if [[ "${SKIP_TEACHER:-0}" == "1" ]] || [[ "${leaf_traces_exist}" == "1" ]]; then
  echo "=== $(date -u) :: reusing existing teacher traces in ${TEACHER_DIR} ==="
else
  echo "=== $(date -u) :: teacher traces source=${SOURCE_RESULTS} tree_text_source=${TREE_TEXT_SOURCE} summary_mode=${TEACHER_SUMMARY_MODE} leaf_size_tokens=${LEAF_SIZE_TOKENS} ==="
  ./venv/bin/python scripts/run_manifesto_teacher_fg_leaf_grid.py \
    --dimension economic \
    --source-results "${SOURCE_RESULTS}" \
    --split-source "${SPLIT_SOURCE}" \
    --tree-text-source "${TREE_TEXT_SOURCE}" \
    --summary-mode "${TEACHER_SUMMARY_MODE}" \
    --summary-temperature "${TEACHER_SUMMARY_TEMPERATURE}" \
    --summary-max-tokens "${SUMMARY_MAX_TOKENS}" \
    --idempotence-mode "${TEACHER_IDEMPOTENCE_MODE}" \
    --score-input "${TEACHER_SCORE_INPUT}" \
    --missing-score-policy "${TEACHER_MISSING_SCORE_POLICY}" \
    --resummary-max-tokens "${RESUMMARY_MAX_TOKENS}" \
    --leaf-size-tokens "${LEAF_SIZE_TOKENS}" \
    --train-n "${TRAIN_N}" \
    --val-n "${VAL_N}" \
    --test-n "${TEST_N}" \
    --teacher-base-url http://localhost:8010/v1 \
    --teacher-model nvidia/Gemma-4-31B-IT-NVFP4 \
    --teacher-api-key EMPTY \
    --scorer-base-url http://localhost:8010/v1 \
    --scorer-model nvidia/Gemma-4-31B-IT-NVFP4 \
    --scorer-api-key EMPTY \
    --num-workers 32 \
    --lm-concurrency 16 \
    --score-max-chars "${SCORE_MAX_CHARS}" \
    --node-summary-max-chars "${NODE_SUMMARY_MAX_CHARS}" \
    --resummary-max-chars "${RESUMMARY_MAX_CHARS}" \
    --output-dir "${TEACHER_DIR}" \
    2>&1 | tee "${TEACHER_LOG}"
fi

echo "=== $(date -u) :: dspy alternating ladder max_iterations=${MAX_ITERATIONS} first_train_side=${FIRST_TRAIN_SIDE} initial_f_degree=${INITIAL_F_DEGREE} initial_g_degree=${INITIAL_G_DEGREE} stage_naming=${STAGE_NAMING} (leaf_size_tokens=${LEAF_SIZE_TOKENS}, threads=${DSPY_NUM_THREADS}, nofile=$(ulimit -n)) ==="
./venv/bin/python scripts/run_alternating_ladder.py \
  --families dspy \
  --dimension economic \
  --leaf-size-tokens "${LEAF_SIZE_TOKENS}" \
  --max-iterations "${MAX_ITERATIONS}" \
  --first-train-side "${FIRST_TRAIN_SIDE}" \
  --initial-f-degree "${INITIAL_F_DEGREE}" \
  --initial-g-degree "${INITIAL_G_DEGREE}" \
  --stage-naming "${STAGE_NAMING}" \
  --fg-grid-dir "${TEACHER_DIR}" \
  --dspy-optimizer mipro \
  --dspy-budget medium \
  --dspy-num-threads "${DSPY_NUM_THREADS}" \
  --dspy-api-base http://localhost:8010/v1 \
  --dspy-model openai/nvidia/Gemma-4-31B-IT-NVFP4 \
  --dspy-api-key EMPTY \
  --dspy-max-tokens "${DSPY_MAX_TOKENS}" \
  --dspy-lm-context-tokens "${LM_CONTEXT_TOKENS}" \
  --dspy-prompt-overhead-tokens "${DSPY_PROMPT_OVERHEAD_TOKENS}" \
  --output-dir "${LADDER_DIR}" \
  2>&1 | tee "${LADDER_LOG}"

echo "=== $(date -u) :: done ==="
if [[ -f "${LADDER_DIR}/grid_summary.md" ]]; then
  cat "${LADDER_DIR}/grid_summary.md"
fi

if [[ "${PLOT_LADDER_GRID:-1}" == "1" ]]; then
  PLOT_DIR="${PLOT_DIR:-${ROOT}/plots}"
  plot_args=()
  if [[ -n "${PLOT_INPUT_ROOTS:-}" ]]; then
    read -r -a plot_input_roots <<< "${PLOT_INPUT_ROOTS}"
    for plot_root in "${plot_input_roots[@]}"; do
      plot_args+=(--input-root "${plot_root}")
    done
  else
    plot_args+=(--input-root "${ROOT}")
  fi
  if [[ -n "${PLOT_STAGES:-}" ]]; then
    plot_args+=(--stages "${PLOT_STAGES}")
  fi
  if [[ -n "${PLOT_FIGURE_TITLE:-}" ]]; then
    plot_args+=(--figure-title "${PLOT_FIGURE_TITLE}")
  fi
  if [[ -n "${PLOT_FIGURE_SUBTITLE:-}" ]]; then
    plot_args+=(--figure-subtitle "${PLOT_FIGURE_SUBTITLE}")
  fi
  if [[ -n "${PLOT_EXTERNAL_PEARSON_MIN:-}" ]]; then
    plot_args+=(--external-pearson-min "${PLOT_EXTERNAL_PEARSON_MIN}")
  fi
  if [[ -n "${PLOT_EXTERNAL_PEARSON_MAX:-}" ]]; then
    plot_args+=(--external-pearson-max "${PLOT_EXTERNAL_PEARSON_MAX}")
  fi
  if [[ -n "${PLOT_STAGE_LABELS:-}" ]]; then
    read -r -a plot_stage_labels <<< "${PLOT_STAGE_LABELS}"
    for plot_stage_label in "${plot_stage_labels[@]}"; do
      plot_args+=(--stage-label "${plot_stage_label}")
    done
  fi
  echo "=== $(date -u) :: plotting ladder grid -> ${PLOT_DIR} ==="
  ./venv/bin/python scripts/plot_manifesto_fg_ladder_grid.py \
    "${plot_args[@]}" \
    --output-dir "${PLOT_DIR}" \
    || echo "warning: ladder grid plotting failed" >&2
fi

if [[ -n "${PLOT_BUNDLES:-}" ]]; then
  bundle_args=()
  read -r -a plot_bundle_names <<< "${PLOT_BUNDLES}"
  for plot_bundle_name in "${plot_bundle_names[@]}"; do
    bundle_args+=(--bundle "${plot_bundle_name}")
  done
  if [[ -n "${PLOT_BUNDLE_RAW_RUN_ROOT:-}" ]]; then
    bundle_args+=(--raw-run-root "${PLOT_BUNDLE_RAW_RUN_ROOT}")
  else
    bundle_args+=(--raw-run-root "${ROOT}")
  fi
  echo "=== $(date -u) :: rendering plot bundles ${PLOT_BUNDLES} ==="
  ./venv/bin/python scripts/render_manifesto_fg_plot_bundles.py \
    "${bundle_args[@]}" \
    || echo "warning: bundle plotting failed" >&2
fi
