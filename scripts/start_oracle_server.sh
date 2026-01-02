#!/bin/bash
# Start Oracle Model Server for Synthetic Data Generation
#
# This script starts a large oracle model (default: Nemotron-Ultra-253B-FP8)
# for generating synthetic training data for OPS summarization.
#
# Usage:
#   ./scripts/start_oracle_server.sh                    # Default Nemotron-253B on port 8001
#   ./scripts/start_oracle_server.sh --port 8002        # Custom port
#   ./scripts/start_oracle_server.sh --model qwen-235b  # Use Qwen-235B instead
#
# The oracle model is typically larger than the inference model and is used for:
# - Generating synthetic training data
# - Preference learning (comparing outputs)
# - Quality validation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_ROOT/config/settings.yaml"

# Defaults
PORT=${PORT:-8001}
MODEL=${MODEL:-genrm-nvfp4}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
GPU_MEM=${GPU_MEM:-0.95}
CUDA_DEVICES=${CUDA_DEVICES:-2,3}  # Default to GPUs 2,3 (main model uses 0,1)

# Help
show_help() {
    cat << 'EOF'
ORACLE MODEL SERVER

Starts a large model for synthetic data generation and preference learning.

Usage: ./scripts/start_oracle_server.sh [OPTIONS]

OPTIONS:
  --port PORT           Server port (default: 8001)
  --model MODEL         Model to use (default: genrm-nvfp4)
                        Available: genrm-nvfp4, qwen3-nemotron-genrm, qwen-235b
  --tensor-parallel N   Number of GPUs (default: 2, reads from config)
  --max-model-len N     Max context length (default: from config)
  --gpu-mem RATIO       GPU memory utilization (default: 0.95)
  --cuda-devices IDS    CUDA devices to use (default: 2,3)
  -h, --help            Show this help

EXAMPLES:
  # Start Nemotron-253B on port 8001
  ./scripts/start_oracle_server.sh

  # Start Qwen-235B instead
  ./scripts/start_oracle_server.sh --model qwen-235b

  # Custom port
  ./scripts/start_oracle_server.sh --port 8002

NOTES:
  - Nemotron-253B requires ~150GB VRAM (4-8 GPUs recommended)
  - Set GPU_MEM lower if running alongside other models
  - For Nemotron, reasoning mode is controlled via system prompt in requests
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --tensor-parallel)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --gpu-mem)
            GPU_MEM="$2"
            shift 2
            ;;
        --cuda-devices)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Activate vLLM environment
source /home/mlinegar/vllm-env/bin/activate
cd "$PROJECT_ROOT"

# Get model path and tensor_parallel from config
read MODEL_PATH MODEL_TP MODEL_MAX_LEN PREFIX_CACHE < <(python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
vllm = cfg.get('vllm', {})
model_cfg = vllm.get('models', {}).get('$MODEL', {})
prefix_cache = vllm.get('enable_prefix_caching', False)
print(model_cfg.get('path', ''), model_cfg.get('tensor_parallel', 2), model_cfg.get('max_model_len', 16384), str(prefix_cache).lower())
" 2>/dev/null)

# Use config values if not overridden by command line
if [[ "$TENSOR_PARALLEL" == "4" ]]; then
    # Only use config value if user didn't explicitly set TP
    TENSOR_PARALLEL=${MODEL_TP:-2}
fi
if [[ "$MAX_MODEL_LEN" == "32768" ]]; then
    # Only use config value if user didn't explicitly set max_model_len
    MAX_MODEL_LEN=${MODEL_MAX_LEN:-16384}
fi

if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: Model '$MODEL' not found in config/settings.yaml"
    echo ""
    echo "Available models:"
    python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
for name in cfg.get('vllm', {}).get('models', {}):
    print(f'  - {name}')
"
    exit 1
fi

# Normalize prefix caching flag from config.
PREFIX_CACHE=${PREFIX_CACHE:-false}
if [[ "$PREFIX_CACHE" == "true" ]]; then
    PREFIX_CACHE_FLAG="--enable-prefix-caching"
else
    PREFIX_CACHE_FLAG=""
fi

# Check if model exists (can be a directory or a single GGUF file)
if [[ ! -d "$MODEL_PATH" && ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo ""
    echo "For Nemotron-253B, download with:"
    echo "  huggingface-cli download nvidia/Llama-3_1-Nemotron-Ultra-253B-v1-FP8 \\"
    echo "    --local-dir $MODEL_PATH --local-dir-use-symlinks False"
    exit 1
fi

# Banner
echo ""
echo "========================================"
echo "Starting Oracle Model Server"
echo "========================================"
echo "Model:          $MODEL"
echo "Path:           $MODEL_PATH"
echo "Port:           $PORT"
echo "CUDA Devices:   $CUDA_DEVICES"
echo "Tensor Parallel:$TENSOR_PARALLEL"
echo "Max Model Len:  $MAX_MODEL_LEN"
echo "GPU Memory:     $GPU_MEM"
echo "Prefix Cache:   $PREFIX_CACHE"
echo "========================================"
echo ""

# Start vLLM server on specified GPUs
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "0.0.0.0" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM" \
    --trust-remote-code \
    --enforce-eager \
    $PREFIX_CACHE_FLAG \
    --disable-log-requests
