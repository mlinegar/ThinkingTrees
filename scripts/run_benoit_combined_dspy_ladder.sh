#!/usr/bin/env bash
# Build all-six-dimension Manifesto teacher traces from the existing combined
# pipeline summaries, combine them into vector-labeled trees, then run the
# joint DSPy f/g alternating ladder.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/manifesto_ladder_runtime.sh"

ROOT="${1:-outputs/manifesto_fg_alternating/combined_benoit_g0init_all6_dspy_$(date +%Y%m%d_%H%M%S)}"
DIM_TRACE_ROOT="${ROOT}/teacher_by_dim"
TEACHER_DIR="${ROOT}/teacher"
LADDER_DIR="${ROOT}/ladder"
PLOT_DIR="${ROOT}/plots"
NOFILE_LIMIT="${NOFILE_LIMIT:-65535}"
LEAF_SIZE_TOKENS="${LEAF_SIZE_TOKENS:-256,512,1024,2048,4096,8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3}"
FIRST_TRAIN_SIDE="${FIRST_TRAIN_SIDE:-g}"
INITIAL_F_DEGREE="${INITIAL_F_DEGREE:-1}"
INITIAL_G_DEGREE="${INITIAL_G_DEGREE:-0}"
STAGE_NAMING="${STAGE_NAMING:-powers}"
DSPY_NUM_THREADS="${DSPY_NUM_THREADS:-128}"
DSPY_BUDGET="${DSPY_BUDGET:-light}"
DSPY_OPTIMIZER="${DSPY_OPTIMIZER:-mipro}"
DSPY_MAX_TOKENS="${DSPY_MAX_TOKENS:-0}"
DSPY_LM_TRANSPORT="${DSPY_LM_TRANSPORT:-batch}"
DSPY_BATCH_MAX_CONCURRENT="${DSPY_BATCH_MAX_CONCURRENT:-}"
DSPY_BATCH_SIZE="${DSPY_BATCH_SIZE:-64}"
DSPY_BATCH_TIMEOUT="${DSPY_BATCH_TIMEOUT:-0.02}"
DSPY_BATCH_REQUEST_TIMEOUT="${DSPY_BATCH_REQUEST_TIMEOUT:-300}"
DSPY_BATCH_AWAIT_RESPONSE_TIMEOUT="${DSPY_BATCH_AWAIT_RESPONSE_TIMEOUT:-}"
DSPY_BATCH_ROUTING_POLICY="${DSPY_BATCH_ROUTING_POLICY:-affinity_load_aware}"
LM_CONTEXT_TOKENS="${LM_CONTEXT_TOKENS:-}"
DSPY_PROMPT_OVERHEAD_TOKENS="${DSPY_PROMPT_OVERHEAD_TOKENS:-1500}"
SOURCE_RESULTS="${SOURCE_RESULTS:-outputs/phase2/combined_pipeline/per_manifesto.jsonl}"
SOURCE_REPORT="${SOURCE_REPORT:-outputs/phase2/combined_pipeline/report.json}"
SPLIT_SOURCE="${SPLIT_SOURCE:-results-order}"
SOURCE_KIND="${SOURCE_KIND:-raw_input}"
case "${SOURCE_KIND}" in
  raw_input) TREE_TEXT_SOURCE="${TREE_TEXT_SOURCE:-aligned_text}" ;;
  external_state)
    TREE_TEXT_SOURCE="${TREE_TEXT_SOURCE:-existing_summary}"
    EXTERNAL_STATE_PRODUCER="${EXTERNAL_STATE_PRODUCER:-g_benoit}"
    ;;
  *) echo "ERROR: unsupported SOURCE_KIND=${SOURCE_KIND}" >&2; exit 2 ;;
esac
TEACHER_SUMMARY_MODE="${TEACHER_SUMMARY_MODE:-identity}"
TEACHER_IDEMPOTENCE_MODE="${TEACHER_IDEMPOTENCE_MODE:-off}"
TEACHER_SCORE_INPUT="${TEACHER_SCORE_INPUT:-teacher_summary}"
TEACHER_MISSING_SCORE_POLICY="${TEACHER_MISSING_SCORE_POLICY:-neutral}"
TRAIN_N="${TRAIN_N:-140}"
VAL_N="${VAL_N:-30}"
TEST_N="${TEST_N:-48}"
TEACHER_NUM_WORKERS="${TEACHER_NUM_WORKERS:-16}"
TEACHER_LM_CONCURRENCY="${TEACHER_LM_CONCURRENCY:-8}"
SCORE_MAX_CHARS="${SCORE_MAX_CHARS:-24000}"
NODE_SUMMARY_MAX_CHARS="${NODE_SUMMARY_MAX_CHARS:-32000}"
RESUMMARY_MAX_CHARS="${RESUMMARY_MAX_CHARS:-24000}"
DIMENSIONS="${DIMENSIONS:-economic social immigration eu environment decentralization}"

LM_CONTEXT_TOKENS="$(manifesto_resolve_lm_context_tokens "${LEAF_SIZE_TOKENS}" "${LM_CONTEXT_TOKENS}")"
DSPY_BATCH_MAX_CONCURRENT="$(manifesto_resolve_dspy_batch_max_concurrent "${LM_CONTEXT_TOKENS}" "${DSPY_BATCH_MAX_CONCURRENT}")"

ulimit -n "${NOFILE_LIMIT}" 2>/dev/null || true
mkdir -p "${ROOT}" "${DIM_TRACE_ROOT}"

dspy_batch_args=(
  --dspy-lm-transport "${DSPY_LM_TRANSPORT}"
  --dspy-batch-max-concurrent "${DSPY_BATCH_MAX_CONCURRENT}"
  --dspy-batch-size "${DSPY_BATCH_SIZE}"
  --dspy-batch-timeout "${DSPY_BATCH_TIMEOUT}"
  --dspy-batch-request-timeout "${DSPY_BATCH_REQUEST_TIMEOUT}"
  --dspy-batch-routing-policy "${DSPY_BATCH_ROUTING_POLICY}"
)
if [[ -n "${DSPY_BATCH_AWAIT_RESPONSE_TIMEOUT}" ]]; then
  dspy_batch_args+=(--dspy-batch-await-response-timeout "${DSPY_BATCH_AWAIT_RESPONSE_TIMEOUT}")
fi

dspy_mipro_args=()
if [[ -n "${DSPY_MIPRO_NUM_CANDIDATES:-}" ]]; then
  dspy_mipro_args+=(--dspy-mipro-num-candidates "${DSPY_MIPRO_NUM_CANDIDATES}")
fi
if [[ -n "${DSPY_MIPRO_NUM_TRIALS:-}" ]]; then
  dspy_mipro_args+=(--dspy-mipro-num-trials "${DSPY_MIPRO_NUM_TRIALS}")
fi
if [[ -n "${DSPY_MIPRO_MAX_BOOTSTRAPPED_DEMOS:-}" ]]; then
  dspy_mipro_args+=(--dspy-mipro-max-bootstrapped-demos "${DSPY_MIPRO_MAX_BOOTSTRAPPED_DEMOS}")
fi
if [[ -n "${DSPY_MIPRO_MAX_LABELED_DEMOS:-}" ]]; then
  dspy_mipro_args+=(--dspy-mipro-max-labeled-demos "${DSPY_MIPRO_MAX_LABELED_DEMOS}")
fi
if [[ -n "${DSPY_MIPRO_MINIBATCH_SIZE:-}" ]]; then
  dspy_mipro_args+=(--dspy-mipro-minibatch-size "${DSPY_MIPRO_MINIBATCH_SIZE}")
fi
if [[ -n "${DSPY_MIPRO_MINIBATCH_FULL_EVAL_STEPS:-}" ]]; then
  dspy_mipro_args+=(--dspy-mipro-minibatch-full-eval-steps "${DSPY_MIPRO_MINIBATCH_FULL_EVAL_STEPS}")
fi

echo "=== $(date -u) :: combined all-six Manifesto ladder root=${ROOT} ==="
echo "=== $(date -u) :: source=${SOURCE_RESULTS} leaves=${LEAF_SIZE_TOKENS} max_iterations=${MAX_ITERATIONS} ==="
echo "=== $(date -u) :: runtime context=${LM_CONTEXT_TOKENS} dspy_batch_max_concurrent=${DSPY_BATCH_MAX_CONCURRENT} ==="

echo "=== $(date -u) :: preflighting DSPy f/g arity budgets ==="
./venv/bin/python scripts/run_alternating_ladder.py \
  --families dspy \
  --dimension combined \
  --leaf-size-tokens "${LEAF_SIZE_TOKENS}" \
  --max-iterations "${MAX_ITERATIONS}" \
  --first-train-side "${FIRST_TRAIN_SIDE}" \
  --initial-f-degree "${INITIAL_F_DEGREE}" \
  --initial-g-degree "${INITIAL_G_DEGREE}" \
  --stage-naming "${STAGE_NAMING}" \
  --dspy-optimizer "${DSPY_OPTIMIZER}" \
  --dspy-budget "${DSPY_BUDGET}" \
  --dspy-num-threads "${DSPY_NUM_THREADS}" \
  --dspy-api-base http://localhost:8010/v1 \
  --dspy-model openai/nvidia/Gemma-4-31B-IT-NVFP4 \
  --dspy-api-key EMPTY \
  --dspy-max-tokens "${DSPY_MAX_TOKENS}" \
  --dspy-lm-context-tokens "${LM_CONTEXT_TOKENS}" \
  --dspy-prompt-overhead-tokens "${DSPY_PROMPT_OVERHEAD_TOKENS}" \
  "${dspy_batch_args[@]}" \
  "${dspy_mipro_args[@]}" \
  --preflight-only \
  --output-dir "${ROOT}/preflight/ladder_budget" \
  2>&1 | tee "${ROOT}/preflight_ladder_budget.log"

for dim in ${DIMENSIONS}; do
  dim_dir="${DIM_TRACE_ROOT}/${dim}"
  leaf_traces_exist=1
  IFS=',' read -r -a leaf_size_token_values <<< "${LEAF_SIZE_TOKENS}"
  for raw_leaf_size in "${leaf_size_token_values[@]}"; do
    leaf_size="$(echo "${raw_leaf_size}" | xargs)"
    [[ -z "${leaf_size}" ]] && continue
    leaf_dir="$(printf 'leaf%04dtok' "${leaf_size}")"
    if [[ ! -f "${dim_dir}/${leaf_dir}/labeled_trees.jsonl" ]]; then
      leaf_traces_exist=0
      break
    fi
  done

  if [[ "${SKIP_TEACHER:-0}" == "1" ]] || [[ "${leaf_traces_exist}" == "1" ]]; then
    echo "=== $(date -u) :: ${dim}: reusing teacher traces in ${dim_dir} ==="
  else
    echo "=== $(date -u) :: ${dim}: scoring node traces source_kind=${SOURCE_KIND} ==="
    ./venv/bin/python scripts/run_manifesto_teacher_fg_leaf_grid.py \
      --dimension "${dim}" \
      --source-results "${SOURCE_RESULTS}" \
      --source-report "${SOURCE_REPORT}" \
      --split-source "${SPLIT_SOURCE}" \
      --source-kind "${SOURCE_KIND}" \
      --summary-mode "${TEACHER_SUMMARY_MODE}" \
      --idempotence-mode "${TEACHER_IDEMPOTENCE_MODE}" \
      --score-input "${TEACHER_SCORE_INPUT}" \
      --missing-score-policy "${TEACHER_MISSING_SCORE_POLICY}" \
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
      --num-workers "${TEACHER_NUM_WORKERS}" \
      --lm-concurrency "${TEACHER_LM_CONCURRENCY}" \
      --score-max-chars "${SCORE_MAX_CHARS}" \
      --node-summary-max-chars "${NODE_SUMMARY_MAX_CHARS}" \
      --resummary-max-chars "${RESUMMARY_MAX_CHARS}" \
      --output-dir "${dim_dir}" \
      2>&1 | tee "${ROOT}/teacher_${dim}.log"
  fi
done

echo "=== $(date -u) :: combining scalar dimension traces -> ${TEACHER_DIR} ==="
./venv/bin/python scripts/combine_manifesto_dimension_traces.py \
  --dimension-root "${DIM_TRACE_ROOT}" \
  --output-dir "${TEACHER_DIR}" \
  --leaf-size-tokens "${LEAF_SIZE_TOKENS}" \
  --dimensions "$(echo "${DIMENSIONS}" | tr ' ' ',')" \
  2>&1 | tee "${ROOT}/combine_teacher.log"

manifesto_audit_tree_bundle "${TEACHER_DIR}" "${SOURCE_KIND}"
tree_bundle_ladder_args=(--dspy-g-init-mode raw_concat)
if [[ "${SOURCE_KIND}" == "external_state" ]]; then
  tree_bundle_ladder_args=(
    --allow-external-state-tree-bundle
    --dspy-g-init-mode teacher_passthrough
  )
fi
if [[ "${ALLOW_LEGACY_TREE_BUNDLE:-0}" == "1" ]]; then
  tree_bundle_ladder_args+=(--allow-legacy-tree-bundle)
fi

echo "=== $(date -u) :: running joint DSPy alternating ladder ==="
./venv/bin/python scripts/run_alternating_ladder.py \
  --families dspy \
  --dimension combined \
  --leaf-size-tokens "${LEAF_SIZE_TOKENS}" \
  --max-iterations "${MAX_ITERATIONS}" \
  --first-train-side "${FIRST_TRAIN_SIDE}" \
  --initial-f-degree "${INITIAL_F_DEGREE}" \
  --initial-g-degree "${INITIAL_G_DEGREE}" \
  --stage-naming "${STAGE_NAMING}" \
  --fg-grid-dir "${TEACHER_DIR}" \
  --dspy-optimizer "${DSPY_OPTIMIZER}" \
  --dspy-budget "${DSPY_BUDGET}" \
  --dspy-num-threads "${DSPY_NUM_THREADS}" \
  --dspy-api-base http://localhost:8010/v1 \
  --dspy-model openai/nvidia/Gemma-4-31B-IT-NVFP4 \
  --dspy-api-key EMPTY \
  --dspy-max-tokens "${DSPY_MAX_TOKENS}" \
  --dspy-lm-context-tokens "${LM_CONTEXT_TOKENS}" \
  --dspy-prompt-overhead-tokens "${DSPY_PROMPT_OVERHEAD_TOKENS}" \
  "${tree_bundle_ladder_args[@]}" \
  "${dspy_batch_args[@]}" \
  "${dspy_mipro_args[@]}" \
  --fail-on-row-error \
  --output-dir "${LADDER_DIR}" \
  2>&1 | tee "${ROOT}/ladder.log"

echo "=== $(date -u) :: plotting joint ladder -> ${PLOT_DIR} ==="
./venv/bin/python scripts/plot_manifesto_fg_ladder_grid.py \
  --input-root "${ROOT}" \
  --figure-title "Manifesto all-six-dimension joint f/g ladder" \
  --figure-subtitle "Macro average across economic, social, immigration, EU, environment, and decentralization; shared joint g and shared JointDimensionScorer f." \
  --output-dir "${PLOT_DIR}" \
  2>&1 | tee "${ROOT}/plot.log"

echo "=== $(date -u) :: done ==="
if [[ -f "${LADDER_DIR}/grid_summary.md" ]]; then
  cat "${LADDER_DIR}/grid_summary.md"
fi
