#!/bin/bash
# Manifesto RILE Training Pipeline
# Runs two-step iterative optimization: oracle + summarizer
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
# --start-server: Stops any running servers and starts a fresh vLLM instance
# with tensor_parallel=4 across all GPUs. Default model: qwen-30b-thinking
# Available models: qwen-80b, qwen-30b-thinking, qwen-235b, glm-4.6, olmo-32b-think
#
# --resume: Auto-finds the most recent run in the output directory and resumes
# from where it left off. Checkpoints are saved after:
#   - Phase 1: Manifesto processing (train/val results)
#   - Phase 2: Training data creation (collector state)
#   - Each optimization round (classifier state + stats)

set -e

# ============================================================================
# Configuration (override with command line args or environment)
# ============================================================================
PORT=${PORT:-8000}
OPT_MODEL_PORT=${OPT_MODEL_PORT:-}  # Optional separate port for optimization model
TRAIN_SAMPLES=${TRAIN_SAMPLES:-30}
VAL_SAMPLES=${VAL_SAMPLES:-15}
TEST_SAMPLES=${TEST_SAMPLES:-10}
ROUNDS=${ROUNDS:-3}
CONCURRENT_DOCS=${CONCURRENT_DOCS:-20}
CONCURRENT_REQUESTS=${CONCURRENT_REQUESTS:-100}

# Optimizer settings
# Options: gepa, bootstrap, bootstrap_random_search, mipro, labeled_fewshot
# Budget options (for gepa/mipro): light, medium, heavy (or use MAX_METRIC_CALLS for direct control)
OPTIMIZER=${OPTIMIZER:-bootstrap_random_search}
OPTIMIZER_BUDGET=${OPTIMIZER_BUDGET:-heavy}
MAX_METRIC_CALLS=${MAX_METRIC_CALLS:-}  # Direct control (overrides budget)
NUM_THREADS=${NUM_THREADS:-128}  # Parallel metric evaluations (can go very high)
START_SERVER=${START_SERVER:-false}  # Auto-start vLLM server
MODEL=${MODEL:-qwen-30b-thinking}  # Model to use with --start-server

# Iterative optimization settings
# N_ITERATIONS: 1=single-pass oracle only, 2+=iterative (oracle→summarizer), 0=until convergence
N_ITERATIONS=${N_ITERATIONS:-3}
CONVERGENCE_THRESHOLD=${CONVERGENCE_THRESHOLD:-0.01}
CONVERGENCE_PATIENCE=${CONVERGENCE_PATIENCE:-2}
SKIP_SUMMARIZER_OPT=${SKIP_SUMMARIZER_OPT:-false}
SKIP_ORACLE_OPT=${SKIP_ORACLE_OPT:-false}

# Resume from checkpoint
RESUME=${RESUME:-false}

# Paths
PROJECT_ROOT="/home/mlinegar/ThinkingTrees"
VLLM_ENV="/home/mlinegar/vllm-env"
OUTPUT_BASE="${PROJECT_ROOT}/data/results/manifesto_rile/training_pipeline"

# ============================================================================
# Parse command line arguments
# ============================================================================
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --opt-model-port)
            OPT_MODEL_PORT="$2"
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
echo "  MANIFESTO RILE TRAINING PIPELINE"
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
    log "Starting vLLM server: ${MODEL} (tensor_parallel=4)"
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
    MODEL_PATH=$(python3 -c "
import yaml
with open('${PROJECT_ROOT}/config/settings.yaml') as f:
    cfg = yaml.safe_load(f)
model_cfg = cfg.get('vllm', {}).get('models', {}).get('${MODEL}', {})
print(model_cfg.get('path', ''))
" 2>/dev/null)

    if [[ -z "${MODEL_PATH}" ]]; then
        log "ERROR: Model '${MODEL}' not found in config/settings.yaml"
        log "Available models: qwen-80b, qwen-30b-thinking, qwen-235b, qwen-vl-235b, glm-4.6, olmo-32b-think"
        exit 1
    fi

    log "Model path: ${MODEL_PATH}"
    log "Starting vLLM with tensor_parallel=4..."

    # Start in background and capture PID
    python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --host "0.0.0.0" \
        --port ${PORT} \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
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
# Run Training Pipeline
# ============================================================================
log ""
log "========================================================================"
log "Starting Training Pipeline"
log "========================================================================"
log ""

# Build command with optional arguments
CMD=(
    python experiments/manifesto_rile/run_training_pipeline.py
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

if [[ -n "${MAX_METRIC_CALLS}" ]]; then
    CMD+=(--max-metric-calls ${MAX_METRIC_CALLS})
fi

if [[ "${SKIP_SUMMARIZER_OPT}" == "true" ]]; then
    CMD+=(--skip-summarizer-opt)
fi

if [[ "${SKIP_ORACLE_OPT}" == "true" ]]; then
    CMD+=(--skip-oracle-opt)
fi

if [[ "${RESUME}" == "true" ]]; then
    CMD+=(--resume)
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
