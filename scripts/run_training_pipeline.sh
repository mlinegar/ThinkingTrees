#!/bin/bash
# Training Pipeline for OPS (Oracle-Preserving Summarization)
# Runs iterative optimization: tree building + score prediction
#
# Usage:
#   ./scripts/run_training_pipeline.sh                        # Default (requires server running)
#   ./scripts/run_training_pipeline.sh --start-server         # Auto-start vLLM with all 4 GPUs
#   ./scripts/run_training_pipeline.sh --start-server --model qwen-235b  # Use specific model
#   ./scripts/run_training_pipeline.sh --n-iterations 0       # Until convergence
#   ./scripts/run_training_pipeline.sh --max-metric-calls 10000  # Max out GPU usage
#   ./scripts/run_training_pipeline.sh --resume               # Resume from latest checkpoint
#   nohup ./scripts/run_training_pipeline.sh > training.log 2>&1 &
#
# Budget options: light, medium, heavy (default)
# For unlimited compute, use --max-metric-calls directly (e.g., 10000)
#
# FAST 8-HOUR CONFIG (for 4× GPU machine):
#   ./scripts/run_training_pipeline.sh \
#     --train-samples 30 --val-samples 15 --n-iterations 2 \
#     --num-threads 64 --concurrent-requests 80
#
# --start-server: Stops any running servers and starts a fresh vLLM instance
# with tensor_parallel=4 across all GPUs. Default model: qwen-30b-thinking
# Available models: qwen-80b, qwen-30b-thinking, qwen-235b, glm-4.6, olmo-32b-think
#
# --resume: Auto-finds the most recent run in the output directory and resumes
# from where it left off. Checkpoints are saved after:
#   - Phase 1: Document processing (train/val results)
#   - Phase 2: Training data creation (collector state)
#   - Each optimization round (classifier state + stats)

set -e

# ============================================================================
# Configuration (override with command line args or environment)
# ============================================================================
PORT=${PORT:-8000}
OPT_MODEL_PORT=${OPT_MODEL_PORT:-}  # Optional separate port for optimization model
TASK=${TASK:-}                      # Task plugin name (default: settings.yaml tasks.default)
DATASET=${DATASET:-}                # Dataset plugin name (default: settings.yaml datasets.default)
DATASET_PATH=${DATASET_PATH:-}      # File path for file-based datasets (e.g., jsonl)
TRAIN_SAMPLES=${TRAIN_SAMPLES:-50}
VAL_SAMPLES=${VAL_SAMPLES:-17}
TEST_SAMPLES=${TEST_SAMPLES:-17}
ROUNDS=${ROUNDS:-3}
CONCURRENT_DOCS=${CONCURRENT_DOCS:-20}
CONCURRENT_REQUESTS=${CONCURRENT_REQUESTS:-200}

# Optimizer settings
# Options: gepa, bootstrap, bootstrap_random_search, mipro, labeled_fewshot
# Budget options (for gepa/mipro): light, medium, heavy (or use MAX_METRIC_CALLS for direct control)
# Note: Default aligned with run_pipeline.py for consistency
OPTIMIZER=${OPTIMIZER:-bootstrap_random_search}
OPTIMIZER_BUDGET=${OPTIMIZER_BUDGET:-heavy}
MAX_METRIC_CALLS=${MAX_METRIC_CALLS:-}  # Direct control (overrides budget)
NUM_THREADS=${NUM_THREADS:-64}  # Parallel metric evaluations (64 avoids retry storms)
START_SERVER=${START_SERVER:-false}  # Auto-start vLLM server
MODEL=${MODEL:-nemotron-30b-fp8}  # Model to use with --start-server

# Iterative optimization settings
# N_ITERATIONS: 1=single-pass oracle only, 2+=iterative (oracle→summarizer), 0=until convergence
N_ITERATIONS=${N_ITERATIONS:-1}
CONVERGENCE_THRESHOLD=${CONVERGENCE_THRESHOLD:-0.01}
CONVERGENCE_PATIENCE=${CONVERGENCE_PATIENCE:-3}
SKIP_SUMMARIZER_OPT=${SKIP_SUMMARIZER_OPT:-false}
SKIP_ORACLE_OPT=${SKIP_ORACLE_OPT:-false}

# Top-down initialization (oracle-aligned demo seeding from short docs)
USE_TOP_DOWN_INIT=${USE_TOP_DOWN_INIT:-false}
N_INIT_DEMOS=${N_INIT_DEMOS:-8}
MAX_INIT_PROMPT_TOKENS=${MAX_INIT_PROMPT_TOKENS:-${MAX_INIT_DOC_CHARS:-4000}}

# Resume from checkpoint
RESUME=${RESUME:-false}

# GenRM OPS Tree Building settings
# Builds trees with tournament selection, collecting demos and preferences
START_GENRM=${START_GENRM:-false}
GENRM_PORT=${GENRM_PORT:-8001}
GENRM_MODEL=${GENRM_MODEL:-genrm-nvfp4}
GENRM_INIT_SAMPLES=${GENRM_INIT_SAMPLES:-8}      # Number of OPS trees to build
GENRM_INIT_CANDIDATES=${GENRM_INIT_CANDIDATES:-4}  # Candidates per node for tournament
TRAIN_COMPARISON_MODULE=${TRAIN_COMPARISON_MODULE:-false}

# Paths (auto-detect project root from script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_ENV="${VLLM_ENV:-${HOME}/vllm-env}"  # Override with VLLM_ENV env var
# Output directory based on task (defaults to 'default' if no task specified)
TASK_DIR="${TASK:-default}"
OUTPUT_BASE="${PROJECT_ROOT}/data/results/${TASK_DIR}/training_pipeline"

# ============================================================================
# Help
# ============================================================================
show_help() {
    cat << 'EOF'
TRAINING PIPELINE

Usage: ./scripts/run_training_pipeline.sh [OPTIONS]

SERVER OPTIONS:
  --start-server          Auto-start vLLM server (stops any running servers)
  --no-start-server       Don't auto-start (default, requires server running)
  --model MODEL           Model to use with --start-server (default: nemotron-30b-fp8)
                          Available: nemotron-30b-fp8, qwen-30b-thinking, qwen-235b
  --port PORT             vLLM server port (default: 8000)
  --opt-model-port PORT   Separate port for optimization model (optional)

DATA OPTIONS:
  --task NAME            Task plugin (default: settings.yaml tasks.default)
  --dataset NAME         Dataset plugin (default: settings.yaml datasets.default)
  --dataset-path PATH    Dataset path for file-based datasets (e.g., jsonl)
  --train-samples N       Number of training samples (default: 30)
  --val-samples N         Number of validation samples (default: 15)
  --test-samples N        Number of test samples (default: 10)
  --rounds N              Document processing rounds (default: 3)

CONCURRENCY OPTIONS:
  --concurrent-docs N     Docs to process in parallel (default: 20)
  --concurrent-requests N Concurrent LLM requests (default: 100)
  --num-threads N         Parallel metric evaluations (default: 64)

OPTIMIZER OPTIONS:
  --optimizer TYPE        Optimizer type (default: bootstrap_random_search)
                          Options: gepa, bootstrap, bootstrap_random_search, mipro, labeled_fewshot
  --optimizer-budget BUDGET  Budget level (default: heavy)
                          Options: light, medium, heavy
  --max-metric-calls N    Direct control over metric calls (overrides budget)

ITERATIVE OPTIMIZATION:
  --n-iterations N        Number of iterations (default: 2)
                          1=single-pass oracle, 2+=iterative, 0=until convergence
  --convergence-threshold N  Threshold for early stopping (default: 0.01)
  --convergence-patience N   Rounds without improvement before stopping (default: 2)
  --skip-summarizer-opt   Skip summarizer optimization
  --skip-oracle-opt       Skip oracle optimization

TOP-DOWN INITIALIZATION:
  --use-top-down-init     Enable oracle-aligned demo seeding from short docs
  --n-init-demos N        Number of initialization demos (default: 8)
  --max-init-prompt-tokens N  Max tokens for init prompts (doc + rubric + instructions)
  --max-init-doc-chars N      Deprecated alias for --max-init-prompt-tokens

GENRM OPS TREE BUILDING (Unified demo + preference collection):
  --start-genrm           Auto-start GenRM server on GPUs 2,3
  --genrm-port PORT       GenRM server port (default: 8001)
  --genrm-model MODEL     GenRM model to use (default: genrm-nvfp4)
  --genrm-init-samples N  Number of OPS trees to build (default: 8)
  --genrm-init-candidates N  Candidates per node for tournament (default: 4)
  --max-init-prompt-tokens N  Max tokens for init prompts (doc + rubric + instructions)
  --max-init-doc-chars N      Deprecated alias for --max-init-prompt-tokens
  --train-comparison-module  Train OPSComparisonModule from collected preferences

RESUME:
  --resume                Resume from latest checkpoint
  --no-resume             Don't resume (default, start fresh)

EXAMPLES:
  # Basic run (requires server already running)
  ./scripts/run_training_pipeline.sh

  # Auto-start vLLM server
  ./scripts/run_training_pipeline.sh --start-server

  # Use specific model
  ./scripts/run_training_pipeline.sh --start-server --model qwen-235b

  # Fast 8-hour config for 4x GPU
  ./scripts/run_training_pipeline.sh --train-samples 30 --val-samples 15 \
    --n-iterations 2 --num-threads 64 --concurrent-requests 80

  # Run until convergence with top-down init
  ./scripts/run_training_pipeline.sh --n-iterations 0 --use-top-down-init

  # Resume from checkpoint
  ./scripts/run_training_pipeline.sh --resume

  # Run with GenRM OPS tree building (unified demo + preference collection)
  ./scripts/run_training_pipeline.sh --start-server --start-genrm

  # Tighten init prompt budget (filters to shorter docs)
  ./scripts/run_training_pipeline.sh --start-genrm --max-init-prompt-tokens 3000 --genrm-init-samples 4

  # Run in background
  nohup ./scripts/run_training_pipeline.sh > training.log 2>&1 &
EOF
    exit 0
}

# ============================================================================
# Parse command line arguments
# ============================================================================
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --opt-model-port)
            OPT_MODEL_PORT="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
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
        --test-samples)
            TEST_SAMPLES="$2"
            shift 2
            ;;
        --rounds)
            ROUNDS="$2"
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
        --max-metric-calls)
            MAX_METRIC_CALLS="$2"
            shift 2
            ;;
        --num-threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        --n-iterations)
            N_ITERATIONS="$2"
            shift 2
            ;;
        --convergence-threshold)
            CONVERGENCE_THRESHOLD="$2"
            shift 2
            ;;
        --convergence-patience)
            CONVERGENCE_PATIENCE="$2"
            shift 2
            ;;
        --skip-summarizer-opt)
            SKIP_SUMMARIZER_OPT="true"
            shift
            ;;
        --skip-oracle-opt)
            SKIP_ORACLE_OPT="true"
            shift
            ;;
        --use-top-down-init)
            USE_TOP_DOWN_INIT="true"
            shift
            ;;
        --n-init-demos)
            N_INIT_DEMOS="$2"
            shift 2
            ;;
        --max-init-prompt-tokens)
            MAX_INIT_PROMPT_TOKENS="$2"
            shift 2
            ;;
        --max-init-doc-chars)
            MAX_INIT_PROMPT_TOKENS="$2"
            shift 2
            ;;
        --resume)
            RESUME="true"
            shift
            ;;
        --no-resume)
            RESUME="false"
            shift
            ;;
        --start-server)
            START_SERVER="true"
            shift
            ;;
        --no-start-server)
            START_SERVER="false"
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --start-genrm)
            START_GENRM="true"
            shift
            ;;
        --no-start-genrm)
            START_GENRM="false"
            shift
            ;;
        --genrm-port)
            GENRM_PORT="$2"
            shift 2
            ;;
        --genrm-model)
            GENRM_MODEL="$2"
            shift 2
            ;;
        --train-comparison-module)
            TRAIN_COMPARISON_MODULE="true"
            shift
            ;;
        --genrm-init-samples)
            GENRM_INIT_SAMPLES="$2"
            shift 2
            ;;
        --genrm-init-candidates)
            GENRM_INIT_CANDIDATES="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ============================================================================
# Setup
# ============================================================================

# Handle resume: find most recent run directory instead of creating new one
if [[ "${RESUME}" == "true" ]]; then
    # Find the most recent directory with checkpoints (handles nested dirs from old bug)
    CHECKPOINT_DIR=$(find "${OUTPUT_BASE}" -name "checkpoints" -type d -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

    if [[ -z "${CHECKPOINT_DIR}" ]]; then
        echo "ERROR: --resume specified but no checkpoints found in ${OUTPUT_BASE}"
        exit 1
    fi

    # Check that checkpoint has actual files
    if [[ -n "$(ls -A "${CHECKPOINT_DIR}" 2>/dev/null)" ]]; then
        # Get parent directory (the actual run dir)
        OUTPUT_DIR=$(dirname "${CHECKPOINT_DIR}")
        echo "Resuming from: ${OUTPUT_DIR}"
        echo "  Checkpoints: ${CHECKPOINT_DIR}"
        ls "${CHECKPOINT_DIR}"
    else
        echo "ERROR: Checkpoint directory is empty: ${CHECKPOINT_DIR}"
        echo "Cannot resume. Run without --resume to start fresh."
        exit 1
    fi
else
    # Create new timestamped directory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${OUTPUT_BASE}/run_${TIMESTAMP}"
    mkdir -p "${OUTPUT_DIR}"
fi

LOG_FILE="${OUTPUT_DIR}/run.log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

# ============================================================================
# Banner
# ============================================================================
echo ""
echo "========================================================================"
echo "  TRAINING PIPELINE"
echo "========================================================================"
echo "  Started:           $(date)"
echo "  Output:            ${OUTPUT_DIR}"
echo ""
echo "  Settings:"
echo "    vLLM Port:       ${PORT}"
if [[ -n "${OPT_MODEL_PORT}" ]]; then
echo "    Opt Model Port:  ${OPT_MODEL_PORT}"
fi
echo "    Train Samples:   ${TRAIN_SAMPLES}"
echo "    Val Samples:     ${VAL_SAMPLES}"
echo "    Test Samples:    ${TEST_SAMPLES}"
echo "    Rounds:          ${ROUNDS}"
echo "    Concurrent Docs: ${CONCURRENT_DOCS}"
echo "    Concurrent Reqs: ${CONCURRENT_REQUESTS}"
echo ""
echo "  Optimizer:"
echo "    Type:            ${OPTIMIZER}"
echo "    Budget:          ${OPTIMIZER_BUDGET}"
if [[ -n "${MAX_METRIC_CALLS}" ]]; then
echo "    Max Metric Calls: ${MAX_METRIC_CALLS} (overrides budget)"
fi
echo "    Threads:         ${NUM_THREADS}"
echo "    Start Server:    ${START_SERVER}"
if [[ "${START_SERVER}" == "true" ]]; then
echo "    Model:           ${MODEL}"
fi
echo ""
echo "  Iterative Optimization:"
echo "    Iterations:      ${N_ITERATIONS} (0=until convergence)"
echo "    Conv Threshold:  ${CONVERGENCE_THRESHOLD}"
echo "    Conv Patience:   ${CONVERGENCE_PATIENCE}"
echo "    Skip Summarizer: ${SKIP_SUMMARIZER_OPT}"
echo "    Top-Down Init:   ${USE_TOP_DOWN_INIT} (demos: ${N_INIT_DEMOS}, max_tokens: ${MAX_INIT_PROMPT_TOKENS})"
echo ""
echo "  GenRM OPS Tree Building:"
echo "    Start GenRM:     ${START_GENRM}"
if [[ "${START_GENRM}" == "true" ]]; then
echo "    GenRM Model:     ${GENRM_MODEL}"
fi
echo "    GenRM Port:      ${GENRM_PORT}"
echo "    Trees to Build:  ${GENRM_INIT_SAMPLES}"
echo "    Candidates/Node: ${GENRM_INIT_CANDIDATES}"
echo "    Init Prompt Max: ${MAX_INIT_PROMPT_TOKENS} tokens"
echo "    Train Comparison: ${TRAIN_COMPARISON_MODULE}"
echo ""
echo "  Resume:"
echo "    Resume:          ${RESUME}"
echo "========================================================================"
echo ""

# ============================================================================
# Activate environment
# ============================================================================
log "Activating vLLM environment..."
source "${VLLM_ENV}/bin/activate"
cd "${PROJECT_ROOT}"

# Add project root to Python path
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ============================================================================
# Auto-start vLLM server (optional)
# ============================================================================
VLLM_PID=""
ORIGINAL_MODEL=""

if [[ "${START_SERVER}" == "true" ]]; then
    log ""
    log "========================================================================"
    log "Starting vLLM server: ${MODEL}"
    log "========================================================================"

    # Remember what model was running (if any)
    if curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        ORIGINAL_MODEL=$(curl -s "http://localhost:${PORT}/v1/models" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else '')" 2>/dev/null || echo "")
        log "Original model: ${ORIGINAL_MODEL}"
    fi

    # Stop ALL vLLM servers
    log "Stopping all vLLM servers..."
    "${PROJECT_ROOT}/scripts/stop_small_servers.sh" --all || true

    # Wait for GPU memory to be released
    sleep 5

    # Get model config from settings.yaml
    read MODEL_PATH MODEL_TP MODEL_MAX_LEN < <(python3 -c "
import yaml
with open('${PROJECT_ROOT}/config/settings.yaml') as f:
    cfg = yaml.safe_load(f)
model_cfg = cfg.get('vllm', {}).get('models', {}).get('${MODEL}', {})
print(model_cfg.get('path', ''), model_cfg.get('tensor_parallel', 2), model_cfg.get('max_model_len', 32768))
" 2>/dev/null)

    if [[ -z "${MODEL_PATH}" ]]; then
        log "ERROR: Model '${MODEL}' not found in config/settings.yaml"
        log "Available models: nemotron-30b-fp8, qwen-30b-thinking, qwen-235b, qwen-80b"
        exit 1
    fi

    # Use config values, default to TP=2 for GPU splitting with GenRM
    MODEL_TP=${MODEL_TP:-2}
    MODEL_MAX_LEN=${MODEL_MAX_LEN:-32768}

    log "Model path: ${MODEL_PATH}"
    log "Starting vLLM on GPUs 0,1 with tensor_parallel=${MODEL_TP}..."

    # Start in background on GPUs 0,1 (GenRM uses GPUs 2,3)
    CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --host "0.0.0.0" \
        --port ${PORT} \
        --tensor-parallel-size ${MODEL_TP} \
        --max-model-len ${MODEL_MAX_LEN} \
        --gpu-memory-utilization 0.90 \
        --trust-remote-code \
        --disable-log-requests \
        --enforce-eager \
        > "${OUTPUT_DIR}/vllm.log" 2>&1 &

    VLLM_PID=$!
    log "vLLM server starting (PID: ${VLLM_PID})"

    # Wait for server to be ready (up to 120s for larger models)
    log "Waiting for vLLM server to be ready..."
    for i in {1..120}; do
        if curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
            MODEL_INFO=$(curl -s "http://localhost:${PORT}/v1/models" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else 'unknown')" 2>/dev/null || echo "unknown")
            log "vLLM ready with model: ${MODEL_INFO}"
            break
        fi
        if [[ $i -eq 120 ]]; then
            log "ERROR: vLLM server failed to start within 120 seconds"
            log "Check ${OUTPUT_DIR}/vllm.log for details"
            exit 1
        fi
        sleep 2
    done
else
    # Just check that server is running
    log "Checking vLLM server on port ${PORT}..."

    check_server() {
        curl -s "http://localhost:$1/v1/models" > /dev/null 2>&1
        return $?
    }

    if check_server ${PORT}; then
        log "vLLM server is running on port ${PORT}"
        MODEL_INFO=$(curl -s "http://localhost:${PORT}/v1/models" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else 'unknown')" 2>/dev/null || echo "unknown")
        log "Model: ${MODEL_INFO}"
    else
        log "ERROR: vLLM server not running on port ${PORT}"
        log ""
        log "Please start the vLLM server first:"
        log "  ./scripts/start_vllm.sh"
        log ""
        log "Or auto-start with:"
        log "  --start-server"
        exit 1
    fi

    # Check optimization model port if specified
    if [[ -n "${OPT_MODEL_PORT}" ]]; then
        log ""
        log "Checking optimization model on port ${OPT_MODEL_PORT}..."
        if check_server ${OPT_MODEL_PORT}; then
            OPT_MODEL_INFO=$(curl -s "http://localhost:${OPT_MODEL_PORT}/v1/models" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else 'unknown')" 2>/dev/null || echo "unknown")
            log "Optimization model: ${OPT_MODEL_INFO}"
        else
            log "ERROR: Optimization model not running on port ${OPT_MODEL_PORT}"
            log ""
            log "Start a smaller model for fast optimization:"
            log "  ./scripts/start_vllm.sh qwen-30b-thinking --port ${OPT_MODEL_PORT}"
            exit 1
        fi
    fi
fi

# ============================================================================
# Start GenRM Server (if enabled)
# ============================================================================
GENRM_PID=""

if [[ "${START_GENRM}" == "true" ]]; then
    log ""
    log "========================================================================"
    log "Starting GenRM Server: ${GENRM_MODEL} (port ${GENRM_PORT})"
    log "========================================================================"

    # Start GenRM server in background using start_oracle_server.sh
    "${PROJECT_ROOT}/scripts/start_oracle_server.sh" \
        --model "${GENRM_MODEL}" \
        --port "${GENRM_PORT}" \
        > "${OUTPUT_DIR}/genrm.log" 2>&1 &

    GENRM_PID=$!
    log "GenRM server starting (PID: ${GENRM_PID})"

    # Wait for server to be ready (up to 600s for large models like GenRM-NVFP4)
    log "Waiting for GenRM server to be ready..."
    for i in {1..300}; do
        if curl -s "http://localhost:${GENRM_PORT}/v1/models" > /dev/null 2>&1; then
            GENRM_MODEL_INFO=$(curl -s "http://localhost:${GENRM_PORT}/v1/models" | \
                python3 -c "import sys, json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else 'unknown')" 2>/dev/null || echo "unknown")
            log "GenRM server ready: ${GENRM_MODEL_INFO}"
            break
        fi
        if [[ $i -eq 300 ]]; then
            log "ERROR: GenRM server failed to start within 600 seconds"
            log "Check ${OUTPUT_DIR}/genrm.log for details"
            exit 1
        fi
        sleep 2
    done
else
    # Check if GenRM is already running (user might have started it externally)
    if curl -s "http://localhost:${GENRM_PORT}/v1/models" > /dev/null 2>&1; then
        GENRM_MODEL_INFO=$(curl -s "http://localhost:${GENRM_PORT}/v1/models" | \
            python3 -c "import sys, json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else 'unknown')" 2>/dev/null || echo "unknown")
        log "GenRM server already running on port ${GENRM_PORT}: ${GENRM_MODEL_INFO}"
    fi
fi

# ============================================================================
# Run Training Pipeline
# ============================================================================
log ""
log "========================================================================"
log "Starting Training Pipeline"
log "========================================================================"
log ""

# Build command with optional arguments
CMD=(
    python -m src.training.run_pipeline
    --port ${PORT}
    --train-samples ${TRAIN_SAMPLES}
    --val-samples ${VAL_SAMPLES}
    --test-samples ${TEST_SAMPLES}
    --rounds ${ROUNDS}
    --concurrent-docs ${CONCURRENT_DOCS}
    --concurrent-requests ${CONCURRENT_REQUESTS}
    --optimizer ${OPTIMIZER}
    --optimizer-budget ${OPTIMIZER_BUDGET}
    --num-threads ${NUM_THREADS}
    --n-iterations ${N_ITERATIONS}
    --convergence-threshold ${CONVERGENCE_THRESHOLD}
    --convergence-patience ${CONVERGENCE_PATIENCE}
    --output-dir "${OUTPUT_DIR}"
)

# Add optional arguments
if [[ -n "${OPT_MODEL_PORT}" ]]; then
    CMD+=(--opt-model-port ${OPT_MODEL_PORT})
fi

if [[ -n "${TASK}" ]]; then
    CMD+=(--task ${TASK})
fi

if [[ -n "${DATASET}" ]]; then
    CMD+=(--dataset ${DATASET})
fi

if [[ -n "${DATASET_PATH}" ]]; then
    CMD+=(--dataset-path "${DATASET_PATH}")
fi

if [[ -n "${MAX_METRIC_CALLS}" ]]; then
    CMD+=(--max-metric-calls ${MAX_METRIC_CALLS})
fi

if [[ "${SKIP_SUMMARIZER_OPT}" == "true" ]]; then
    CMD+=(--skip-summarizer-opt)
fi

if [[ "${SKIP_ORACLE_OPT}" == "true" ]]; then
    CMD+=(--skip-oracle-opt)
fi

if [[ "${USE_TOP_DOWN_INIT}" == "true" ]]; then
    CMD+=(--use-top-down-init --n-init-demos ${N_INIT_DEMOS} --max-init-prompt-tokens ${MAX_INIT_PROMPT_TOKENS})
fi

if [[ "${RESUME}" == "true" ]]; then
    CMD+=(--resume)
fi

# Add GenRM OPS tree building arguments if server is available
if [[ "${START_GENRM}" == "true" ]] || curl -s "http://localhost:${GENRM_PORT}/v1/models" > /dev/null 2>&1; then
    CMD+=(--enable-genrm --genrm-port ${GENRM_PORT})
    CMD+=(--genrm-init-samples ${GENRM_INIT_SAMPLES})
    CMD+=(--genrm-init-candidates ${GENRM_INIT_CANDIDATES})
    CMD+=(--max-init-prompt-tokens ${MAX_INIT_PROMPT_TOKENS})
fi

if [[ "${TRAIN_COMPARISON_MODULE}" == "true" ]]; then
    CMD+=(--train-comparison-module)
fi

# Add any extra args passed through
CMD+=("${EXTRA_ARGS[@]}")

# Run the command
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

# ============================================================================
# Cleanup: Stop vLLM server if we started it
# ============================================================================
if [[ -n "${VLLM_PID}" ]]; then
    log ""
    log "Stopping vLLM server we started (PID: ${VLLM_PID})..."
    kill -TERM ${VLLM_PID} 2>/dev/null || true
    sleep 2
    log "vLLM server stopped"
fi

# Stop GenRM server if we started it
if [[ -n "${GENRM_PID}" ]]; then
    log ""
    log "Stopping GenRM server we started (PID: ${GENRM_PID})..."
    kill -TERM ${GENRM_PID} 2>/dev/null || true
    sleep 2
    log "GenRM server stopped"
fi

# ============================================================================
# Summary
# ============================================================================
log ""
log "========================================================================"
log "TRAINING PIPELINE COMPLETE"
log "========================================================================"
log "Exit code: ${EXIT_CODE}"
log "Finished:  $(date)"
log "Results:   ${OUTPUT_DIR}"
log ""

# Print final stats if available
if [ -f "${OUTPUT_DIR}/final_stats.json" ]; then
    log "Final Statistics:"
    python3 -c "
import json
with open('${OUTPUT_DIR}/final_stats.json') as f:
    stats = json.load(f)

print()
if 'baseline' in stats:
    print(f\"  Baseline Train MAE: {stats['baseline']['train']['mae']:.2f}\")
    print(f\"  Baseline Val MAE:   {stats['baseline']['val']['mae']:.2f}\")

if 'test' in stats:
    print(f\"  Test Pipeline MAE:  {stats['test']['pipeline']['mae']:.2f}\")
    print(f\"  Test Classifier MAE: {stats['test']['classifier']['mae']:.2f}\")

if 'rounds' in stats:
    print()
    print('  Optimization Rounds:')
    for r in stats['rounds']:
        if 'error' not in r:
            val_mae = r.get('val_eval', {}).get('mae', 'N/A')
            if isinstance(val_mae, float):
                val_mae = f'{val_mae:.2f}'
            print(f\"    Round {r['round']}: {r['metric_before']:.3f} -> {r['metric_after']:.3f} (Val MAE: {val_mae})\")
" 2>/dev/null || true
fi

log ""
log "To view full results:"
log "  cat ${OUTPUT_DIR}/final_stats.json | python -m json.tool"
log ""

exit ${EXIT_CODE}
