#!/bin/bash
# Overnight RILE Scoring Test with DSPy Optimization
# This script runs the full manifesto RILE pipeline with optimization
#
# Usage: ./scripts/run_overnight_test.sh
# Or with nohup: nohup ./scripts/run_overnight_test.sh > overnight.log 2>&1 &

set -e

# Configuration
TASK_PORT=8000
AUDITOR_PORT=8001
CHUNK_SIZE=2000
ITERATIONS=5
MAX_SAMPLES=50  # Adjust based on how many samples to process
SPLIT="train"   # Use train for optimization
OUTPUT_BASE="/home/mlinegar/ThinkingTrees/data/results/manifesto_rile"

# Paths
PROJECT_ROOT="/home/mlinegar/ThinkingTrees"
VLLM_ENV="/home/mlinegar/vllm-env"
TASK_MODEL="/mnt/data/models/NVFP4/Qwen3-30B-A3B-Thinking-2507-FP4"
AUDITOR_MODEL="/mnt/data/models/Qwen3-Next-80B-A3B-Instruct-NVFP4"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/overnight_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/run.log"

echo "=============================================="
echo "OVERNIGHT RILE SCORING TEST"
echo "=============================================="
echo "Started at: $(date)"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

# Function to check if a vLLM server is running on a port
check_server() {
    local port=$1
    curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1
    return $?
}

# Function to start vLLM server
start_vllm() {
    local model=$1
    local port=$2
    local gpus=$3
    local name=$4

    log "Starting vLLM server for ${name} on port ${port} (GPUs: ${gpus})..."

    CUDA_VISIBLE_DEVICES=${gpus} ${VLLM_ENV}/bin/python -m vllm.entrypoints.openai.api_server \
        --model "${model}" \
        --port ${port} \
        --max-model-len 32768 \
        --tensor-parallel-size 1 \
        --trust-remote-code \
        --disable-log-requests \
        > "${OUTPUT_DIR}/${name}_server.log" 2>&1 &

    local pid=$!
    echo $pid > "${OUTPUT_DIR}/${name}_server.pid"

    # Wait for server to be ready (up to 5 minutes)
    log "Waiting for ${name} server to be ready..."
    for i in {1..60}; do
        if check_server ${port}; then
            log "${name} server is ready!"
            return 0
        fi
        sleep 5
    done

    log "ERROR: ${name} server failed to start within 5 minutes"
    return 1
}

# Check if servers are already running
log "Checking if vLLM servers are running..."

if check_server ${TASK_PORT}; then
    log "Task model (30b) server already running on port ${TASK_PORT}"
else
    log "Task model (30b) server not found, starting..."
    start_vllm "${TASK_MODEL}" ${TASK_PORT} "0" "task_model" || exit 1
fi

if check_server ${AUDITOR_PORT}; then
    log "Auditor model (80b) server already running on port ${AUDITOR_PORT}"
else
    log "Auditor model (80b) server not found, starting..."
    start_vllm "${AUDITOR_MODEL}" ${AUDITOR_PORT} "2,3" "auditor_model" || exit 1
fi

# Run the optimization
log ""
log "=============================================="
log "RUNNING OPTIMIZATION"
log "=============================================="
log "Iterations: ${ITERATIONS}"
log "Max samples: ${MAX_SAMPLES}"
log "Chunk size: ${CHUNK_SIZE}"
log "Split: ${SPLIT}"
log ""

cd "${PROJECT_ROOT}"
source "${VLLM_ENV}/bin/activate"

python experiments/manifesto_rile/run_with_optimization.py \
    --task-port ${TASK_PORT} \
    --auditor-port ${AUDITOR_PORT} \
    --iterations ${ITERATIONS} \
    --max-samples ${MAX_SAMPLES} \
    --split ${SPLIT} \
    --chunk-size ${CHUNK_SIZE} \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=$?

log ""
log "=============================================="
log "RUN COMPLETE"
log "=============================================="
log "Exit code: ${EXIT_CODE}"
log "Finished at: $(date)"
log "Results saved to: ${OUTPUT_DIR}"
log ""

# Summary of results
if [ -f "${OUTPUT_DIR}/iteration_${ITERATIONS}_results.json" ]; then
    log "Final iteration results available at: ${OUTPUT_DIR}/iteration_${ITERATIONS}_results.json"
fi

exit ${EXIT_CODE}
