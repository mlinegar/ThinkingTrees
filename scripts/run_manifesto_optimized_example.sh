#!/bin/bash
# Build DSPy-optimized summarizers/scorer, then run batched manifesto ID examples.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_VENV_PYTHON="${PROJECT_ROOT}/venv/bin/python"

PORT=${PORT:-8000}
OPT_MODEL_PORT=${OPT_MODEL_PORT:-}
TASK_REPLICA_PORT=${TASK_REPLICA_PORT:-8002}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-80}
VAL_SAMPLES=${VAL_SAMPLES:-24}
CHUNK_SIZE=${CHUNK_SIZE:-8000}
CHUNK_TOKENS=${CHUNK_TOKENS:-}
CONCURRENT_DOCS=${CONCURRENT_DOCS:-20}
CONCURRENT_REQUESTS=${CONCURRENT_REQUESTS:-200}
OPTIMIZER=${OPTIMIZER:-gepa}
OPTIMIZER_BUDGET=${OPTIMIZER_BUDGET:-medium}
NUM_THREADS=${NUM_THREADS:-16}
GEPA_REFLECTION_MINIBATCH_SIZE=${GEPA_REFLECTION_MINIBATCH_SIZE:-3}
SKIP_ORACLE_OPT=${SKIP_ORACLE_OPT:-false}
SKIP_SUMMARIZER_OPT=${SKIP_SUMMARIZER_OPT:-false}
INIT_MODULES_DIR=${INIT_MODULES_DIR:-}
RERUN_OPTIMIZATION=${RERUN_OPTIMIZATION:-false}
ENABLE_GENRM=${ENABLE_GENRM:-false}
GENRM_PORT=${GENRM_PORT:-8001}
GENRM_CUDA_DEVICES=${GENRM_CUDA_DEVICES:-2,3}
N_ITERATIONS=${N_ITERATIONS:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/rile_optimized_example_$(date +%Y%m%d_%H%M%S)"}
OUTPUT_DIR_EXPLICIT=false
RESUME=${RESUME:-false}
START_SERVER=${START_SERVER:-false}
MODEL=${MODEL:-nemotron-30b-nvfp4}
FALLBACK_MODEL=${FALLBACK_MODEL:-nemotron-30b-fp8}
ENABLE_FALLBACK=${ENABLE_FALLBACK:-true}
CUDA_DEVICES=${CUDA_DEVICES:-0,1}
DYNAMIC_GPU=${DYNAMIC_GPU:-true}
PHASE1_SCORE_REQUESTS=${PHASE1_SCORE_REQUESTS:-false}
PHASE1_RUN_BASELINE=${PHASE1_RUN_BASELINE:-false}
PHASE1_MAX_TOKENS_SUMMARY=${PHASE1_MAX_TOKENS_SUMMARY:-180}
PHASE1_MAX_TOKENS_SCORE=${PHASE1_MAX_TOKENS_SCORE:-80}
SUMMARIZER_LEAF_MAX_RATIO=${SUMMARIZER_LEAF_MAX_RATIO:-0.25}
SUMMARIZER_MERGE_MAX_RATIO=${SUMMARIZER_MERGE_MAX_RATIO:-0.6}
SUMMARIZER_RATIO_MIN_INPUT_CHARS=${SUMMARIZER_RATIO_MIN_INPUT_CHARS:-200}
INITIAL_SCORER_INSTRUCTION=${INITIAL_SCORER_INSTRUCTION:-}
DEFAULT_INITIAL_SCORER_INSTRUCTION_FILE=${DEFAULT_INITIAL_SCORER_INSTRUCTION_FILE:-"prompts/manifesto_rile/initial_scorer_instruction.txt"}
INITIAL_SCORER_INSTRUCTION_FILE=${INITIAL_SCORER_INSTRUCTION_FILE:-"${DEFAULT_INITIAL_SCORER_INSTRUCTION_FILE}"}
IDS=("51320_198306")
PUBLISH_LATEST=${PUBLISH_LATEST:-true}
LATEST_ROOT_DIR=${LATEST_ROOT_DIR:-"outputs/latest/manifesto_rile"}

show_help() {
  cat <<'EOF'
Run a batched RILE optimized example (training + manifesto-ID evaluation).
Uses fixed chunking only: adaptive chunking and honesty splits are disabled.

Usage:
  ./scripts/run_manifesto_optimized_example.sh [options]

Options:
  --start-server              Auto-start task model server if PORT is down
  --no-start-server           Require an already-running server (default)
  --model PROFILE             Model profile for --start-server (default: nemotron-30b-nvfp4)
  --fallback-model PROFILE    Fallback profile if primary model startup fails
                              (default: nemotron-30b-fp8; use --no-fallback-model to disable)
  --no-fallback-model         Disable fallback startup model
  --cuda-devices IDS          GPU device list for auto-start (default: 0,1)
  --dynamic-gpu              Enable dynamic GPU orchestration inside run_pipeline (default)
  --no-dynamic-gpu           Disable dynamic GPU orchestration (static single-server mode)
  --phase1-score-requests     Enable Phase 1 score requests (slower, default: disabled)
  --no-phase1-score-requests  Disable Phase 1 score requests (faster, default)
  --phase1-run-baseline       Enable Phase 1 baseline requests (slowest, default: disabled)
  --no-phase1-run-baseline    Disable Phase 1 baseline requests (default)
  --phase1-max-tokens-summary N  Max output tokens for Phase 1 leaf/merge summaries (default: 180)
  --phase1-max-tokens-score N    Max output tokens for Phase 1 score/baseline requests (default: 80)
  --summarizer-leaf-max-ratio R  Soft max len(summary)/len(chunk) for leaf optimization (default: 0.25)
  --summarizer-merge-max-ratio R Soft max len(merged)/len(left+right) for merge optimization (default: 0.6)
  --summarizer-ratio-min-input-chars N
                              Only apply ratio penalties when input chars >= N (default: 200)
  --port N                    Task model port (default: 8000)
  --opt-model-port N          Optional separate optimization model port (run_pipeline --opt-model-port)
  --train-samples N           Training sample count for optimization (default: 80)
  --val-samples N             Validation sample count for optimization (default: 24)
  --chunk-size N              Max chunk chars for tree building (default: 8000)
  --chunk-tokens N            Max chunk tokens for tree building; takes precedence over char chunking
  --concurrent-docs N         Concurrent docs for batched runs (default: 20)
  --concurrent-requests N     Concurrent LLM requests (default: 200)
  --optimizer TYPE            DSPy optimizer (default: gepa)
  --optimizer-budget BUDGET   Optimizer budget (default: medium)
  --num-threads N             Parallel metric evaluations for optimizer (default: 16)
  --gepa-reflection-minibatch-size N
                              GEPA reflection minibatch size (default: 3)
  --skip-oracle-opt           Skip scorer/oracle optimization (optimize summarizers only)
  --skip-summarizer-opt       Skip summarizer optimization (optimize scorer only)
  --init-modules-dir PATH     Initialize Phase A optimization from prior modules
                              (expects scorer_final.json / leaf_summarizer_final.json / merge_summarizer_final.json)
  --rerun-optimization        When used with --resume, rerun Phase A optimization even if artifacts exist
  --enable-genrm              Deprecated and blocked (large-model-only path)
  --optimize-judge            Deprecated and blocked (large-model-only path)
  --tournament-of-tournaments Deprecated and blocked (large-model-only path)
  --initial-scorer-instruction TEXT
                              Optional initial scorer instruction prompt to seed optimization
  --initial-scorer-instruction-file PATH
                              Path to text file containing initial scorer instruction prompt
                              (default: prompts/manifesto_rile/initial_scorer_instruction.txt)
  --no-initial-scorer-instruction-file
                              Disable default file-based scorer instruction seeding
  --n-iterations N            Optimization rounds (default: 1)
  --resume                    Resume latest interrupted run (or --output-dir target)
  --no-resume                 Start fresh (default)
  --output-dir PATH           Output directory for artifacts/results
  --ids ID [ID ...]           Manifesto IDs to evaluate (default: 51320_198306)

Example:
  ./scripts/run_manifesto_optimized_example.sh \
    --ids 51320_198306 51620_198306 \
    --chunk-size 8000 \
    --train-samples 100 \
    --val-samples 30
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      show_help
      exit 0
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --opt-model-port)
      OPT_MODEL_PORT="$2"
      shift 2
      ;;
    --start-server)
      START_SERVER=true
      shift
      ;;
    --no-start-server)
      START_SERVER=false
      shift
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --fallback-model)
      FALLBACK_MODEL="$2"
      ENABLE_FALLBACK=true
      shift 2
      ;;
    --no-fallback-model)
      ENABLE_FALLBACK=false
      shift
      ;;
    --cuda-devices)
      CUDA_DEVICES="$2"
      shift 2
      ;;
    --dynamic-gpu)
      DYNAMIC_GPU=true
      shift
      ;;
    --no-dynamic-gpu)
      DYNAMIC_GPU=false
      shift
      ;;
    --phase1-score-requests)
      PHASE1_SCORE_REQUESTS=true
      shift
      ;;
    --no-phase1-score-requests)
      PHASE1_SCORE_REQUESTS=false
      shift
      ;;
    --phase1-run-baseline)
      PHASE1_RUN_BASELINE=true
      shift
      ;;
    --no-phase1-run-baseline)
      PHASE1_RUN_BASELINE=false
      shift
      ;;
    --phase1-max-tokens-summary)
      PHASE1_MAX_TOKENS_SUMMARY="$2"
      shift 2
      ;;
    --phase1-max-tokens-score)
      PHASE1_MAX_TOKENS_SCORE="$2"
      shift 2
      ;;
    --summarizer-leaf-max-ratio)
      SUMMARIZER_LEAF_MAX_RATIO="$2"
      shift 2
      ;;
    --summarizer-merge-max-ratio)
      SUMMARIZER_MERGE_MAX_RATIO="$2"
      shift 2
      ;;
    --summarizer-ratio-min-input-chars)
      SUMMARIZER_RATIO_MIN_INPUT_CHARS="$2"
      shift 2
      ;;
    --train-samples)
      TRAIN_SAMPLES="$2"
      shift 2
      ;;
    --val-samples)
      VAL_SAMPLES="$2"
      shift 2
      ;;
    --chunk-size)
      CHUNK_SIZE="$2"
      shift 2
      ;;
    --chunk-tokens)
      CHUNK_TOKENS="$2"
      shift 2
      ;;
    --concurrent-docs)
      CONCURRENT_DOCS="$2"
      shift 2
      ;;
    --concurrent-requests)
      CONCURRENT_REQUESTS="$2"
      shift 2
      ;;
    --optimizer)
      OPTIMIZER="$2"
      shift 2
      ;;
    --optimizer-budget)
      OPTIMIZER_BUDGET="$2"
      shift 2
      ;;
    --num-threads)
      NUM_THREADS="$2"
      shift 2
      ;;
    --gepa-reflection-minibatch-size)
      GEPA_REFLECTION_MINIBATCH_SIZE="$2"
      shift 2
      ;;
    --skip-oracle-opt)
      SKIP_ORACLE_OPT=true
      shift
      ;;
    --skip-summarizer-opt)
      SKIP_SUMMARIZER_OPT=true
      shift
      ;;
    --init-modules-dir)
      INIT_MODULES_DIR="$2"
      shift 2
      ;;
    --rerun-optimization)
      RERUN_OPTIMIZATION=true
      shift
      ;;
    --enable-genrm)
      echo "ERROR: --enable-genrm is deprecated and blocked in this runner." >&2
      echo "Use local-law bootstrap (teacher scorer + proxy/GEPA), no GenRM." >&2
      exit 2
      ;;
    --disable-genrm)
      ENABLE_GENRM=false
      shift
      ;;
    --optimize-judge|--tournament-of-tournaments)
      echo "ERROR: $1 is deprecated and blocked in this runner." >&2
      echo "Use local-law bootstrap (teacher scorer + proxy/GEPA), no GenRM." >&2
      exit 2
      ;;
    --genrm-port)
      GENRM_PORT="$2"
      shift 2
      ;;
    --initial-scorer-instruction)
      INITIAL_SCORER_INSTRUCTION="$2"
      shift 2
      ;;
    --initial-scorer-instruction-file)
      INITIAL_SCORER_INSTRUCTION_FILE="$2"
      shift 2
      ;;
    --no-initial-scorer-instruction-file)
      INITIAL_SCORER_INSTRUCTION_FILE=""
      shift
      ;;
    --n-iterations)
      N_ITERATIONS="$2"
      shift 2
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --no-resume)
      RESUME=false
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      OUTPUT_DIR_EXPLICIT=true
      shift 2
      ;;
    --ids)
      shift
      IDS=()
      while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
        IDS+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Use --help for usage." >&2
      exit 1
      ;;
  esac
done

if [[ "${ENABLE_GENRM}" == "true" ]]; then
  echo "ERROR: GenRM mode is deprecated and blocked in this runner." >&2
  echo "Use local-law bootstrap (teacher scorer + proxy/GEPA), no GenRM." >&2
  exit 2
fi

if [[ "${RESUME}" == "true" ]]; then
  if [[ "${OUTPUT_DIR_EXPLICIT}" != "true" ]]; then
    CHECKPOINT_DIR=$(find outputs -maxdepth 2 -type d -path "outputs/rile_optimized_example_*/checkpoints" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [[ -z "${CHECKPOINT_DIR}" ]]; then
      echo "ERROR: --resume specified but no prior optimized-example checkpoints were found." >&2
      echo "Run once without --resume first, or pass --output-dir PATH to resume a specific run." >&2
      exit 1
    fi
    OUTPUT_DIR="$(dirname "${CHECKPOINT_DIR}")"
    echo "Resuming from latest run: ${OUTPUT_DIR}"
  else
    echo "Resuming from explicit output dir: ${OUTPUT_DIR}"
  fi
fi

mkdir -p "${OUTPUT_DIR}"

if [[ -x "${DEFAULT_VENV_PYTHON}" ]]; then
  PYTHON_BIN="${DEFAULT_VENV_PYTHON}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "ERROR: No python interpreter found in PATH." >&2
  exit 1
fi

if [[ "${RESUME}" == "true" ]]; then
  # If a prior run published artifacts to outputs/latest but the target output-dir
  # is missing one (e.g., scorer_final.json), restore so Phase A can resume
  # without re-running long optimizations.
  MODULE_DIR="${OUTPUT_DIR}/trained_modules"
  LATEST_MODULE_DIR="${LATEST_ROOT_DIR}/trained_modules"
  mkdir -p "${MODULE_DIR}"
  for module_name in leaf_summarizer_final.json merge_summarizer_final.json scorer_final.json; do
    if [[ ! -f "${MODULE_DIR}/${module_name}" && -f "${LATEST_MODULE_DIR}/${module_name}" ]]; then
      echo "Restoring missing ${module_name} from ${LATEST_MODULE_DIR}"
      cp -f "${LATEST_MODULE_DIR}/${module_name}" "${MODULE_DIR}/${module_name}"
    fi
  done
fi

if [[ "${DYNAMIC_GPU}" == "true" ]]; then
  # When dynamic GPU is enabled, ports are controlled by config/settings.yaml.
  ORCH_PORTS="$("${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
import sys
try:
    import yaml
except Exception:
    sys.exit(1)

with open("config/settings.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

orch = cfg.get("orchestration", {}) if isinstance(cfg, dict) else {}
	task_primary = orch.get("task_primary_port", 8000)
	task_replica = orch.get("task_replica_port", 8002)
	print(f"{int(task_primary)} {int(task_replica)}")
PY
)"
  if [[ -n "${ORCH_PORTS}" ]]; then
    read -r ORCH_TASK_PRIMARY_PORT ORCH_TASK_REPLICA_PORT <<< "${ORCH_PORTS}"
    if [[ -n "${ORCH_TASK_PRIMARY_PORT:-}" && "${PORT}" != "${ORCH_TASK_PRIMARY_PORT}" ]]; then
      echo "WARNING: --port ${PORT} ignored under dynamic GPU; using ${ORCH_TASK_PRIMARY_PORT} from config/settings.yaml" >&2
    fi
    if [[ -n "${ORCH_TASK_REPLICA_PORT:-}" && "${TASK_REPLICA_PORT}" != "${ORCH_TASK_REPLICA_PORT}" ]]; then
      echo "WARNING: TASK_REPLICA_PORT ${TASK_REPLICA_PORT} overridden under dynamic GPU; using ${ORCH_TASK_REPLICA_PORT} from config/settings.yaml" >&2
    fi
    PORT="${ORCH_TASK_PRIMARY_PORT}"
    TASK_REPLICA_PORT="${ORCH_TASK_REPLICA_PORT}"
  else
    echo "WARNING: Could not read orchestration ports from config/settings.yaml; using defaults." >&2
  fi
fi

check_server() {
  curl -s "http://localhost:${PORT}/v1/models" >/dev/null 2>&1
}

check_opt_server() {
  if [[ -z "${OPT_MODEL_PORT}" ]]; then
    return 0
  fi
  curl -s "http://localhost:${OPT_MODEL_PORT}/v1/models" >/dev/null 2>&1
}

log_contains() {
  local pattern="$1"
  local file_path="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -q "${pattern}" "${file_path}"
  else
    grep -q "${pattern}" "${file_path}"
  fi
}

start_server_process() {
  local model="$1"
  local log_path="$2"

  mkdir -p "$(dirname "${log_path}")"
  : > "${log_path}"
  ln -sfn "$(basename "${log_path}")" "${OUTPUT_DIR}/task_model.log" >/dev/null 2>&1 || true

  local args=(
    --port "${PORT}"
    --kv-cache-dtype auto
  )

  # Prefer eager startup for Nemotron profiles; improves compatibility.
  if [[ "${model}" == nemotron-* ]]; then
    args+=(--enforce-eager)
  fi
  if [[ -n "${CUDA_DEVICES}" ]]; then
    args+=(--cuda-devices "${CUDA_DEVICES}")
  fi

  ./scripts/start_vllm.sh "${model}" "${args[@]}" > "${log_path}" 2>&1 &
  echo $!
}

wait_for_server() {
  local max_wait="${1:-240}"
  local server_pid="$2"
  local log_path="$3"
  local waited=0

  while ! check_server; do
    if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      echo "ERROR: Server process exited before becoming ready (PID ${server_pid})" >&2
      if [[ -f "${log_path}" ]]; then
        if log_contains "Non-gated activations are only supported by the flashinfer CUTLASS backend" "${log_path}"; then
          echo "Detected NVFP4 startup incompatibility in vLLM for this host." >&2
        fi
        if log_contains "NVMLError_Unknown" "${log_path}" || log_contains "Can't initialize NVML" "${log_path}"; then
          echo "Detected GPU/NVML initialization failure while starting vLLM." >&2
          echo "This is a host GPU state issue (not an optimizer/prompt issue)." >&2
          echo "Run 'nvidia-smi' to confirm, then restart GPU services or reboot before retrying." >&2
        fi
      fi
      return 2
    fi
    sleep 2
    waited=$((waited + 2))
    if [[ "${waited}" -ge "${max_wait}" ]]; then
      echo "ERROR: Server did not become ready within ${max_wait}s (PID ${server_pid})" >&2
      return 1
    fi
  done
  return 0
}

if [[ "${DYNAMIC_GPU}" == "true" ]]; then
  echo "Dynamic GPU enabled: run_pipeline will manage vLLM servers (DP=2 on ${PORT} and ${TASK_REPLICA_PORT})."
else
  if check_server; then
    echo "Using existing task model server on port ${PORT}"
  else
    if [[ "${START_SERVER}" == "true" ]]; then
      ACTIVE_MODEL="${MODEL}"
      PRIMARY_LOG="${OUTPUT_DIR}/task_model_${MODEL}.log"
      echo "Task server not detected on port ${PORT}; starting ${MODEL}..."
      TASK_SERVER_PID="$(start_server_process "${MODEL}" "${PRIMARY_LOG}")"
      if wait_for_server 300 "${TASK_SERVER_PID}" "${PRIMARY_LOG}"; then
        echo "Task model ready on port ${PORT} (PID ${TASK_SERVER_PID}, model ${MODEL})"
      elif [[ "${ENABLE_FALLBACK}" == "true" && "${FALLBACK_MODEL}" != "${MODEL}" ]]; then
        if kill -0 "${TASK_SERVER_PID}" >/dev/null 2>&1; then
          kill "${TASK_SERVER_PID}" >/dev/null 2>&1 || true
          sleep 1
        fi
        FALLBACK_LOG="${OUTPUT_DIR}/task_model_${FALLBACK_MODEL}.log"
        echo "Primary model failed; attempting fallback model ${FALLBACK_MODEL}..."
        TASK_SERVER_PID="$(start_server_process "${FALLBACK_MODEL}" "${FALLBACK_LOG}")"
        if wait_for_server 300 "${TASK_SERVER_PID}" "${FALLBACK_LOG}"; then
          ACTIVE_MODEL="${FALLBACK_MODEL}"
          echo "Task model ready on port ${PORT} (PID ${TASK_SERVER_PID}, model ${ACTIVE_MODEL})"
        else
          echo "ERROR: Task model server failed to start on port ${PORT}" >&2
          echo "Primary log:  ${PRIMARY_LOG}" >&2
          echo "Fallback log: ${FALLBACK_LOG}" >&2
          exit 1
        fi
      else
        echo "ERROR: Task model server failed to start on port ${PORT}" >&2
        echo "See ${PRIMARY_LOG}" >&2
        exit 1
      fi
    else
      echo "ERROR: Task model server is not running on port ${PORT}" >&2
      echo "Start it first (e.g. ./scripts/start_dual_servers.sh --small-only)" >&2
      echo "or rerun this command with --start-server." >&2
      exit 1
    fi
  fi
fi

if [[ -n "${OPT_MODEL_PORT}" ]]; then
  if check_opt_server; then
    echo "Using existing optimization model server on port ${OPT_MODEL_PORT}"
  else
    echo "ERROR: Optimization model server is not running on port ${OPT_MODEL_PORT}" >&2
    echo "Start it first (e.g. ./scripts/start_dual_servers.sh)" >&2
    exit 1
  fi
fi

if [[ "${DYNAMIC_GPU}" == "true" ]]; then
  DYNAMIC_GPU_FLAG="--dynamic-gpu"
else
  DYNAMIC_GPU_FLAG="--no-dynamic-gpu"
fi

if [[ "${PHASE1_SCORE_REQUESTS}" == "true" ]]; then
  PHASE1_SCORE_FLAG="--phase1-score-requests"
else
  PHASE1_SCORE_FLAG="--no-phase1-score-requests"
fi

if [[ "${PHASE1_RUN_BASELINE}" == "true" ]]; then
  PHASE1_BASELINE_FLAG="--phase1-run-baseline"
else
  PHASE1_BASELINE_FLAG="--no-phase1-run-baseline"
fi

RESUME_ARGS=()
if [[ "${RESUME}" == "true" ]]; then
  RESUME_ARGS+=(--resume)
fi

RERUN_OPT_ARGS=()
if [[ "${RERUN_OPTIMIZATION}" == "true" ]]; then
  RERUN_OPT_ARGS+=(--rerun-optimization)
fi

OPT_MODEL_ARGS=()
if [[ -n "${OPT_MODEL_PORT}" ]]; then
  OPT_MODEL_ARGS+=(--opt-model-port "${OPT_MODEL_PORT}")
fi

INIT_MODULES_ARGS=()
if [[ -n "${INIT_MODULES_DIR}" ]]; then
  INIT_MODULES_ARGS+=(--init-modules-dir "${INIT_MODULES_DIR}")
fi

KEEP_SERVERS_ARGS=()
if [[ "${DYNAMIC_GPU}" == "true" ]]; then
  KEEP_SERVERS_ARGS+=(--keep-servers-running)
fi

SKIP_OPT_ARGS=()
if [[ "${SKIP_ORACLE_OPT}" == "true" ]]; then
  SKIP_OPT_ARGS+=(--skip-oracle-opt)
fi
if [[ "${SKIP_SUMMARIZER_OPT}" == "true" ]]; then
  SKIP_OPT_ARGS+=(--skip-summarizer-opt)
fi

INITIAL_SCORER_ARGS=()
if [[ -n "${INITIAL_SCORER_INSTRUCTION}" ]]; then
  INITIAL_SCORER_ARGS+=(--initial-scorer-instruction "${INITIAL_SCORER_INSTRUCTION}")
fi
if [[ -n "${INITIAL_SCORER_INSTRUCTION_FILE}" ]]; then
  if [[ -f "${INITIAL_SCORER_INSTRUCTION_FILE}" ]]; then
    echo "Using initial scorer instruction seed: ${INITIAL_SCORER_INSTRUCTION_FILE}"
    INITIAL_SCORER_ARGS+=(--initial-scorer-instruction-file "${INITIAL_SCORER_INSTRUCTION_FILE}")
  else
    echo "WARNING: initial scorer instruction file not found: ${INITIAL_SCORER_INSTRUCTION_FILE}" >&2
    echo "Continuing without file-based scorer instruction seed." >&2
  fi
fi

echo "============================================================"
echo "PHASE A: Optimize Scorer + Summarizers (batched)"
echo "============================================================"
RUN_PIPELINE_CMD=(
  "${PYTHON_BIN}" -m src.training.run_pipeline
  --task manifesto_rile
  --port "${PORT}"
  --train-samples "${TRAIN_SAMPLES}"
  --val-samples "${VAL_SAMPLES}"
  --test-samples 0
  --concurrent-docs "${CONCURRENT_DOCS}"
  --concurrent-requests "${CONCURRENT_REQUESTS}"
  --max-chunk-chars "${CHUNK_SIZE}"
  --optimizer "${OPTIMIZER}"
  --optimizer-budget "${OPTIMIZER_BUDGET}"
  --num-threads "${NUM_THREADS}"
  --gepa-reflection-minibatch-size "${GEPA_REFLECTION_MINIBATCH_SIZE}"
  --n-iterations "${N_ITERATIONS}"
  "${PHASE1_SCORE_FLAG}"
  "${PHASE1_BASELINE_FLAG}"
  --phase1-max-tokens-summary "${PHASE1_MAX_TOKENS_SUMMARY}"
  --phase1-max-tokens-score "${PHASE1_MAX_TOKENS_SCORE}"
  --summarizer-leaf-max-ratio "${SUMMARIZER_LEAF_MAX_RATIO}"
  --summarizer-merge-max-ratio "${SUMMARIZER_MERGE_MAX_RATIO}"
  --summarizer-ratio-min-input-chars "${SUMMARIZER_RATIO_MIN_INPUT_CHARS}"
  --no-adaptive-chunking
  --no-honest-chunking
  --no-three-layer-honesty
  --no-adaptive-embedding-proxy
  "${INITIAL_SCORER_ARGS[@]}"
  "${OPT_MODEL_ARGS[@]}"
  "${RESUME_ARGS[@]}"
  "${RERUN_OPT_ARGS[@]}"
  "${SKIP_OPT_ARGS[@]}"
  "${INIT_MODULES_ARGS[@]}"
  --output-dir "${OUTPUT_DIR}"
  "${DYNAMIC_GPU_FLAG}"
  "${KEEP_SERVERS_ARGS[@]}"
)
if [[ -n "${CHUNK_TOKENS}" ]]; then
  RUN_PIPELINE_CMD+=(--max-chunk-tokens "${CHUNK_TOKENS}")
fi
"${RUN_PIPELINE_CMD[@]}"

LEAF_PATH="${OUTPUT_DIR}/trained_modules/leaf_summarizer_final.json"
MERGE_PATH="${OUTPUT_DIR}/trained_modules/merge_summarizer_final.json"
SCORER_PATH="${OUTPUT_DIR}/trained_modules/scorer_final.json"
FINAL_STATS_PATH="${OUTPUT_DIR}/final_stats.json"

if [[ ! -f "${LEAF_PATH}" ]] || [[ ! -f "${MERGE_PATH}" ]] || [[ ! -f "${SCORER_PATH}" ]]; then
  echo "Missing optimized module artifacts in ${OUTPUT_DIR}/trained_modules" >&2
  if [[ -f "${FINAL_STATS_PATH}" ]]; then
    PIPELINE_ERROR="$("${PYTHON_BIN}" - "${FINAL_STATS_PATH}" <<'PY'
import json
import sys

error = ""
try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)
    error = str(data.get("error") or "")
except Exception:
    pass
print(error)
PY
)"
    if [[ "${PIPELINE_ERROR}" == "insufficient_training_data" ]]; then
      echo "run_pipeline reported insufficient_training_data." >&2
      echo "Likely cause: the task model was unreachable during Phase A, so summaries were empty." >&2
      echo "Check server health: curl http://localhost:${PORT}/v1/models" >&2
      echo "Inspect logs: ${OUTPUT_DIR}/task_model.log" >&2
    fi
  fi
  exit 1
fi

if [[ "${PUBLISH_LATEST}" == "true" ]]; then
  echo
  echo "============================================================"
  echo "PHASE A.5: Publish Optimized Modules (latest)"
  echo "============================================================"
  mkdir -p "${LATEST_ROOT_DIR}/trained_modules"
  cp -f "${LEAF_PATH}" "${LATEST_ROOT_DIR}/trained_modules/leaf_summarizer_final.json"
  cp -f "${MERGE_PATH}" "${LATEST_ROOT_DIR}/trained_modules/merge_summarizer_final.json"
  cp -f "${SCORER_PATH}" "${LATEST_ROOT_DIR}/trained_modules/scorer_final.json"
  {
    echo "task=manifesto_rile"
    echo "published_at=$(date --iso-8601=seconds)"
    echo "source_output_dir=${OUTPUT_DIR}"
  } > "${LATEST_ROOT_DIR}/published_from.txt"
  echo "Published to ${LATEST_ROOT_DIR}/trained_modules"
fi

if ! check_server; then
  echo "ERROR: Task model server on port ${PORT} is no longer reachable before Phase B." >&2
  echo "Restart server and rerun Phase B with scripts/run_manifesto_batched_example.py." >&2
  exit 1
fi

echo
echo "============================================================"
echo "PHASE B: Run Batched Manifesto-ID Example (optimized modules)"
echo "============================================================"
PHASE_B_PORT_ARGS=()
if [[ "${DYNAMIC_GPU}" == "true" ]]; then
  PHASE_B_PORT_ARGS+=(--ports "${PORT}" "${TASK_REPLICA_PORT}")
else
  PHASE_B_PORT_ARGS+=(--port "${PORT}")
fi

PHASE_B_CMD=(
  "${PYTHON_BIN}" scripts/run_manifesto_batched_example.py
  --ids "${IDS[@]}"
  "${PHASE_B_PORT_ARGS[@]}"
  --chunk-size "${CHUNK_SIZE}"
  --concurrent-docs "${CONCURRENT_DOCS}"
  --concurrent-requests "${CONCURRENT_REQUESTS}"
  --leaf-module-path "${LEAF_PATH}"
  --merge-module-path "${MERGE_PATH}"
  --scorer-module-path "${SCORER_PATH}"
  --output "${OUTPUT_DIR}/rile_optimized_example.json"
)
if [[ -n "${CHUNK_TOKENS}" ]]; then
  PHASE_B_CMD+=(--chunk-tokens "${CHUNK_TOKENS}")
fi
"${PHASE_B_CMD[@]}"

echo
echo "Done."
echo "Training artifacts: ${OUTPUT_DIR}"
echo "Optimized example: ${OUTPUT_DIR}/rile_optimized_example.json"
