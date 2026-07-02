#!/usr/bin/env bash
# Run the joint all-six Benoit DSPy ladder in leaf-size groups, restarting the
# vLLM server with a context/KV-cache configuration matched to each group.
#
# Default groups aim for roughly stable throughput:
#   small leaves: low context, high sequence capacity / DSPy concurrency
#   large leaves: higher context, lower sequence capacity to preserve KV RAM
#
# Override with:
#   LEAF_CONTEXT_GROUPS="256,512:8192:0.90:1024:1024 4096:51200:0.92:256:256"
# Format per group:
#   leaves:max_model_len:gpu_mem:max_num_seqs[:dspy_batch_max_concurrent]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/manifesto_ladder_runtime.sh"

ROOT="${1:-outputs/manifesto_fg_alternating/combined_benoit_joint_teacher_dspy_$(date +%Y%m%d_%H%M%S)}"
PROFILE="${PROFILE:-gemma-4-31b-it-nvfp4}"
PORT="${PORT:-8010}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}"
SERVER_START_TIMEOUT_SECONDS="${SERVER_START_TIMEOUT_SECONDS:-900}"
KEEP_LAST_SERVER="${KEEP_LAST_SERVER:-1}"
PRESTOP_SERVER_JOB_ROOT="${PRESTOP_SERVER_JOB_ROOT:-}"
REUSE_FIRST_SERVER_JOB_ROOT="${REUSE_FIRST_SERVER_JOB_ROOT:-}"
LEAF_CONTEXT_GROUPS="${LEAF_CONTEXT_GROUPS:-$(manifesto_leaf_context_group_defaults)}"

mkdir -p "${ROOT}"

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
    --name "gemma4_31b_8010_ctx${context_len}_seq${max_num_seqs}_group${idx}" \
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

  echo "=== $(date -u) :: running ladder leaves=${leaves} context=${context_len} ==="
  env \
    SKIP_TEACHER=1 \
    LEAF_SIZE_TOKENS="${leaves}" \
    LM_CONTEXT_TOKENS="${context_len}" \
    DSPY_MAX_TOKENS="${DSPY_MAX_TOKENS:-0}" \
    MAX_ITERATIONS="${MAX_ITERATIONS:-3}" \
    FIRST_TRAIN_SIDE="${FIRST_TRAIN_SIDE:-g}" \
    INITIAL_F_DEGREE="${INITIAL_F_DEGREE:-1}" \
    INITIAL_G_DEGREE="${INITIAL_G_DEGREE:-0}" \
    STAGE_NAMING="${STAGE_NAMING:-powers}" \
    DSPY_BUDGET="${DSPY_BUDGET:-medium}" \
    DSPY_NUM_THREADS="${DSPY_NUM_THREADS:-128}" \
    DSPY_LM_TRANSPORT="${DSPY_LM_TRANSPORT:-batch}" \
    DSPY_BATCH_MAX_CONCURRENT="${dspy_batch_max_concurrent}" \
    bash scripts/run_benoit_combined_joint_teacher_dspy_ladder.sh "${ROOT}"
done

if [[ "${KEEP_LAST_SERVER}" != "1" && -n "${current_server_job_root}" ]]; then
  echo "=== $(date -u) :: stopping final group server ${current_server_job_root} ==="
  stop_server_job_root "${current_server_job_root}"
fi
