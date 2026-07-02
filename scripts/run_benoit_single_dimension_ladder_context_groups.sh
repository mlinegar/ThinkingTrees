#!/usr/bin/env bash
# Run one Manifesto dimension through the DSPy f/g ladder, grouped by leaf-size
# context buckets so vLLM gets an efficient max-model-len / max-num-seqs config.
#
# This is the scalar analogue of run_benoit_joint_ladder_context_groups.sh:
# start a context-sized vLLM server for each leaf group, then call the shared
# single-dimension scalar wrapper. The scalar wrapper owns teacher trace
# generation/reuse and the DSPy ladder, just like the economic runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/manifesto_ladder_runtime.sh"

DIMENSION="${DIMENSION:-decentralization}"
ROOT="${1:-outputs/manifesto_fg_alternating/${DIMENSION}_benoit_single_dspy_$(date +%Y%m%d_%H%M%S)}"
TEACHER_DIR="${TEACHER_DIR:-${ROOT}/teacher}"
PLOT_DIR="${ROOT}/plots"

PROFILE="${PROFILE:-gemma-4-31b-it-nvfp4}"
PORT="${PORT:-8010}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}"
SERVER_START_TIMEOUT_SECONDS="${SERVER_START_TIMEOUT_SECONDS:-900}"
KEEP_LAST_SERVER="${KEEP_LAST_SERVER:-0}"
PRESTOP_SERVER_JOB_ROOT="${PRESTOP_SERVER_JOB_ROOT:-}"
REUSE_FIRST_SERVER_JOB_ROOT="${REUSE_FIRST_SERVER_JOB_ROOT:-}"
LEAF_CONTEXT_GROUPS="${LEAF_CONTEXT_GROUPS:-$(manifesto_leaf_context_group_defaults)}"

NOFILE_LIMIT="${NOFILE_LIMIT:-65535}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3}"
FIRST_TRAIN_SIDE="${FIRST_TRAIN_SIDE:-g}"
INITIAL_F_DEGREE="${INITIAL_F_DEGREE:-1}"
INITIAL_G_DEGREE="${INITIAL_G_DEGREE:-0}"
STAGE_NAMING="${STAGE_NAMING:-powers}"
DSPY_OPTIMIZER="${DSPY_OPTIMIZER:-mipro}"
DSPY_BUDGET="${DSPY_BUDGET:-light}"
DSPY_NUM_THREADS="${DSPY_NUM_THREADS:-128}"
DSPY_MAX_TOKENS="${DSPY_MAX_TOKENS:-0}"
DSPY_LM_TRANSPORT="${DSPY_LM_TRANSPORT:-batch}"
DSPY_BATCH_SIZE="${DSPY_BATCH_SIZE:-64}"
DSPY_BATCH_TIMEOUT="${DSPY_BATCH_TIMEOUT:-0.02}"
DSPY_BATCH_REQUEST_TIMEOUT="${DSPY_BATCH_REQUEST_TIMEOUT:-300}"
DSPY_BATCH_AWAIT_RESPONSE_TIMEOUT="${DSPY_BATCH_AWAIT_RESPONSE_TIMEOUT:-}"
DSPY_BATCH_ROUTING_POLICY="${DSPY_BATCH_ROUTING_POLICY:-affinity_load_aware}"
DSPY_F_INIT_PATH="${DSPY_F_INIT_PATH:-}"
DSPY_F_INIT_MODE="${DSPY_F_INIT_MODE:-pretuned_scorer}"
DSPY_MAX_TRAIN_RECORDS="${DSPY_MAX_TRAIN_RECORDS:-}"
DSPY_MIPRO_MINIBATCH_FULL_EVAL_STEPS="${DSPY_MIPRO_MINIBATCH_FULL_EVAL_STEPS:-3}"
DSPY_PROMPT_OVERHEAD_TOKENS="${DSPY_PROMPT_OVERHEAD_TOKENS:-1500}"
TEACHER_TIMEOUT_SECONDS="${TEACHER_TIMEOUT_SECONDS:-600}"
SCORER_TIMEOUT_SECONDS="${SCORER_TIMEOUT_SECONDS:-600}"
EXPERT_TARGET_SCALE="${EXPERT_TARGET_SCALE:-normalized_1_7}"
SCORING_CONTEXT_SOURCE="${SCORING_CONTEXT_SOURCE:-compact}"
PLOT_LADDER_GRID="${PLOT_LADDER_GRID:-1}"
PLOT_PREDICTION_DISTS="${PLOT_PREDICTION_DISTS:-1}"

ulimit -n "${NOFILE_LIMIT}" 2>/dev/null || true
mkdir -p "${ROOT}"

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

current_server_job_root=""

stop_server_job_root() {
  local job_root="$1"
  if [[ -z "${job_root}" || ! -f "${job_root}/manifest.json" ]]; then
    return
  fi
  ./venv/bin/python scripts/long_job.py stop --job-root "${job_root}" >/dev/null 2>&1 || true
}

wait_for_server() {
  local port="$1"
  local timeout="$2"
  local deadline=$((SECONDS + timeout))
  until curl -sS --max-time 3 "http://localhost:${port}/v1/models" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "ERROR: server on port ${port} did not become ready within ${timeout}s" >&2
      return 1
    fi
    sleep 5
  done
}

start_server_for_group() {
  local idx="$1"
  local context_len="$2"
  local gpu_mem="$3"
  local max_num_seqs="$4"
  local job_root="${ROOT}/server_ctx${context_len}_seq${max_num_seqs}_group${idx}"

  echo "=== $(date -u) :: starting ${PROFILE} on :${PORT} context=${context_len} gpu_mem=${gpu_mem} max_num_seqs=${max_num_seqs} ==="
  ./venv/bin/python scripts/long_job.py launch \
    --name "${DIMENSION}_gemma4_31b_8010_ctx${context_len}_seq${max_num_seqs}_group${idx}" \
    --job-root "${job_root}" \
    --cwd "$(pwd)" \
    -- ./scripts/start_vllm.sh "${PROFILE}" \
      --port "${PORT}" \
      --cuda-devices "${CUDA_DEVICES}" \
      --tensor-parallel "${TENSOR_PARALLEL}" \
      --max-model-len "${context_len}" \
      --gpu-mem "${gpu_mem}" \
      --max-num-seqs "${max_num_seqs}" \
    >/dev/null
  current_server_job_root="${job_root}"
  wait_for_server "${PORT}" "${SERVER_START_TIMEOUT_SECONDS}"
}

if [[ -n "${PRESTOP_SERVER_JOB_ROOT}" ]]; then
  stop_server_job_root "${PRESTOP_SERVER_JOB_ROOT}"
fi

idx=0
for group in ${LEAF_CONTEXT_GROUPS}; do
  idx=$((idx + 1))
  IFS=':' read -r leaves context_len gpu_mem max_num_seqs dspy_batch_max_concurrent <<< "${group}"
  if [[ -z "${leaves:-}" || -z "${context_len:-}" || -z "${gpu_mem:-}" || -z "${max_num_seqs:-}" ]]; then
    echo "ERROR: bad LEAF_CONTEXT_GROUPS entry '${group}'" >&2
    exit 2
  fi
  dspy_batch_max_concurrent="${dspy_batch_max_concurrent:-${max_num_seqs}}"

  if [[ "${idx}" == "1" && -n "${REUSE_FIRST_SERVER_JOB_ROOT}" ]]; then
    current_server_job_root="${REUSE_FIRST_SERVER_JOB_ROOT}"
    echo "=== $(date -u) :: reusing first group server ${current_server_job_root} leaves=${leaves} context=${context_len} ==="
    wait_for_server "${PORT}" "${SERVER_START_TIMEOUT_SECONDS}"
  elif [[ -n "${current_server_job_root}" ]]; then
    echo "=== $(date -u) :: stopping prior group server ${current_server_job_root} ==="
    stop_server_job_root "${current_server_job_root}"
    current_server_job_root=""
    start_server_for_group "${idx}" "${context_len}" "${gpu_mem}" "${max_num_seqs}"
  else
    start_server_for_group "${idx}" "${context_len}" "${gpu_mem}" "${max_num_seqs}"
  fi

  echo "=== $(date -u) :: ${DIMENSION}: running scalar wrapper leaves=${leaves} context=${context_len} dspy_batch_max_concurrent=${dspy_batch_max_concurrent} ==="
  env \
    DIMENSION="${DIMENSION}" \
    TEACHER_DIR="${TEACHER_DIR}" \
    LEAF_SIZE_TOKENS="${leaves}" \
    LM_CONTEXT_TOKENS="${context_len}" \
    DSPY_BATCH_MAX_CONCURRENT="${dspy_batch_max_concurrent}" \
    MAX_ITERATIONS="${MAX_ITERATIONS}" \
    FIRST_TRAIN_SIDE="${FIRST_TRAIN_SIDE}" \
    INITIAL_F_DEGREE="${INITIAL_F_DEGREE}" \
    INITIAL_G_DEGREE="${INITIAL_G_DEGREE}" \
    STAGE_NAMING="${STAGE_NAMING}" \
    DSPY_OPTIMIZER="${DSPY_OPTIMIZER}" \
    DSPY_BUDGET="${DSPY_BUDGET}" \
    DSPY_NUM_THREADS="${DSPY_NUM_THREADS}" \
    DSPY_MAX_TOKENS="${DSPY_MAX_TOKENS}" \
    DSPY_LM_TRANSPORT="${DSPY_LM_TRANSPORT}" \
    DSPY_BATCH_SIZE="${DSPY_BATCH_SIZE}" \
    DSPY_BATCH_TIMEOUT="${DSPY_BATCH_TIMEOUT}" \
    DSPY_BATCH_REQUEST_TIMEOUT="${DSPY_BATCH_REQUEST_TIMEOUT}" \
    DSPY_BATCH_AWAIT_RESPONSE_TIMEOUT="${DSPY_BATCH_AWAIT_RESPONSE_TIMEOUT}" \
    DSPY_BATCH_ROUTING_POLICY="${DSPY_BATCH_ROUTING_POLICY}" \
    DSPY_F_INIT_PATH="${DSPY_F_INIT_PATH}" \
    DSPY_F_INIT_MODE="${DSPY_F_INIT_MODE}" \
    DSPY_MAX_TRAIN_RECORDS="${DSPY_MAX_TRAIN_RECORDS}" \
    DSPY_MIPRO_MINIBATCH_FULL_EVAL_STEPS="${DSPY_MIPRO_MINIBATCH_FULL_EVAL_STEPS}" \
    DSPY_PROMPT_OVERHEAD_TOKENS="${DSPY_PROMPT_OVERHEAD_TOKENS}" \
    SOURCE_RESULTS="${SOURCE_RESULTS:-outputs/overnight_benoit/full_pipeline/${DIMENSION}/per_manifesto.jsonl}" \
    SPLIT_SOURCE="${SPLIT_SOURCE:-results-order}" \
    SOURCE_KIND="${SOURCE_KIND:-raw_input}" \
    TREE_TEXT_SOURCE="${TREE_TEXT_SOURCE:-}" \
    TEACHER_SUMMARY_MODE="${TEACHER_SUMMARY_MODE:-teacher}" \
    TEACHER_SUMMARY_TEMPERATURE="${TEACHER_SUMMARY_TEMPERATURE:-0.0}" \
    TEACHER_IDEMPOTENCE_MODE="${TEACHER_IDEMPOTENCE_MODE:-off}" \
    TEACHER_SCORE_INPUT="${TEACHER_SCORE_INPUT:-teacher_summary}" \
    TEACHER_MISSING_SCORE_POLICY="${TEACHER_MISSING_SCORE_POLICY:-neutral}" \
    TEACHER_TIMEOUT_SECONDS="${TEACHER_TIMEOUT_SECONDS}" \
    SCORER_TIMEOUT_SECONDS="${SCORER_TIMEOUT_SECONDS}" \
    EXPERT_TARGET_SCALE="${EXPERT_TARGET_SCALE}" \
    SCORING_CONTEXT_SOURCE="${SCORING_CONTEXT_SOURCE}" \
    SUMMARY_MAX_TOKENS="${SUMMARY_MAX_TOKENS:-0}" \
    RESUMMARY_MAX_TOKENS="${RESUMMARY_MAX_TOKENS:-0}" \
    SCORE_MAX_CHARS="${SCORE_MAX_CHARS:-24000}" \
    NODE_SUMMARY_MAX_CHARS="${NODE_SUMMARY_MAX_CHARS:-32000}" \
    RESUMMARY_MAX_CHARS="${RESUMMARY_MAX_CHARS:-24000}" \
    TRAIN_N="${TRAIN_N:-140}" \
    VAL_N="${VAL_N:-30}" \
    TEST_N="${TEST_N:-48}" \
    TEACHER_NUM_WORKERS="${TEACHER_NUM_WORKERS:-32}" \
    TEACHER_LM_CONCURRENCY="${TEACHER_LM_CONCURRENCY:-16}" \
    PLOT_LADDER_GRID=0 \
    bash scripts/run_benoit_supervised_dspy_ladder.sh "${ROOT}" \
    2>&1 | tee "${ROOT}/ladder_group${idx}_${DIMENSION}.log"
done

if [[ "${KEEP_LAST_SERVER}" != "1" && -n "${current_server_job_root}" ]]; then
  echo "=== $(date -u) :: stopping final group server ${current_server_job_root} ==="
  stop_server_job_root "${current_server_job_root}"
fi

if [[ "${PLOT_LADDER_GRID}" == "1" ]]; then
  echo "=== $(date -u) :: plotting ${DIMENSION} ladder -> ${PLOT_DIR} ==="
  ./venv/bin/python scripts/plot_manifesto_fg_ladder_grid.py \
    --input-root "${ROOT}" \
    --figure-title "Manifesto ${DIMENSION} f/g ladder" \
    --figure-subtitle "Single-dimension optimization with fresh scalar ${DIMENSION} teacher traces; same grouped server/batching launcher as the all-six run." \
    --output-dir "${PLOT_DIR}" \
    2>&1 | tee "${ROOT}/plot.log" \
    || echo "warning: ladder grid plotting failed" >&2
fi

if [[ "${PLOT_PREDICTION_DISTS}" == "1" ]]; then
  echo "=== $(date -u) :: plotting ${DIMENSION} prediction distributions ==="
  ./venv/bin/python scripts/plot_manifesto_prediction_distributions.py \
    --source-root "$(dirname "$(dirname "${SOURCE_RESULTS:-outputs/overnight_benoit/full_pipeline/${DIMENSION}/per_manifesto.jsonl}")")" \
    --dimension "${DIMENSION}" \
    --ladder-root "${DIMENSION}=${ROOT}" \
    --output-dir "${ROOT}/plots_prediction_distributions" \
    2>&1 | tee "${ROOT}/plot_prediction_distributions.log" \
    || echo "warning: prediction distribution plotting failed" >&2
fi

echo "=== $(date -u) :: done ==="
if [[ -f "${ROOT}/ladder/grid_summary.md" ]]; then
  cat "${ROOT}/ladder/grid_summary.md"
fi
