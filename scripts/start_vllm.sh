#!/bin/bash
# vLLM Server Launch Script for ThinkingTrees
# Reads model config from config/settings.yaml
#
# Usage:
#   ./scripts/start_vllm.sh          # Uses default model from config
#   ./scripts/start_vllm.sh small    # Uses "small" model profile
#   ./scripts/start_vllm.sh large    # Uses "large" model profile

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_ROOT/config/settings.yaml"

# Model profile: first arg or read default from config
PROFILE="${1:-}"

# Activate the vllm environment
source /home/mlinegar/vllm-env/bin/activate

# Parse config with Python (more reliable than shell YAML parsing)
read_config() {
    python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)

vllm = cfg.get('vllm', {})
profile = '${PROFILE}' or vllm.get('default', 'small')
models = vllm.get('models', {})

if profile not in models:
    print(f'ERROR: Profile \"{profile}\" not found. Available: {list(models.keys())}', file=__import__('sys').stderr)
    exit(1)

m = models[profile]
print(f\"PROFILE={profile}\")
print(f\"MODEL_PATH={m['path']}\")
print(f\"TENSOR_PARALLEL={m.get('tensor_parallel', 1)}\")
print(f\"MAX_MODEL_LEN={m.get('max_model_len', 8192)}\")
print(f\"HOST={vllm.get('host', '0.0.0.0')}\")
print(f\"PORT={vllm.get('port', 8000)}\")
print(f\"GPU_MEM={vllm.get('gpu_memory_utilization', 0.90)}\")
"
}

# Load config
eval "$(read_config)"

echo "=========================================="
echo "Starting vLLM Server"
echo "=========================================="
echo "Profile: $PROFILE"
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Tensor Parallel: $TENSOR_PARALLEL"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "=========================================="

# Launch vLLM server
# Using --enforce-eager to bypass Triton compilation issues
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM" \
    --trust-remote-code \
    --disable-log-requests \
    --enforce-eager \
    "${@:2}"
