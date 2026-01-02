#!/bin/bash
# vLLM Server Launch Script for ThinkingTrees
# Reads model config from config/settings.yaml
#
# Usage:
#   ./scripts/start_vllm.sh                    # Uses default model from config
#   ./scripts/start_vllm.sh qwen-80b           # Uses "qwen-80b" model profile
#   ./scripts/start_vllm.sh --preset training  # Uses speculative decoding preset
#   ./scripts/start_vllm.sh --preset inference # Uses inference preset (small model only)
#
# Speculative Decoding:
#   When a preset with speculative decoding is used, the server runs with:
#   - Target model: the main model for verification
#   - Draft model: smaller model for fast token proposal
#   - Result: 1.5-3x faster generation with identical outputs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_ROOT/config/settings.yaml"

# Parse arguments
PROFILE=""
PRESET=""
PORT_OVERRIDE=""
KV_CACHE_DTYPE=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --port)
            PORT_OVERRIDE="$2"
            shift 2
            ;;
        --kv-cache-dtype)
            KV_CACHE_DTYPE="$2"
            shift 2
            ;;
        --*)
            # Collect unknown --flags as extra args to pass to vLLM
            EXTRA_ARGS+=("$1")
            if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
                EXTRA_ARGS+=("$2")
                shift
            fi
            shift
            ;;
        *)
            PROFILE="$1"
            shift
            ;;
    esac
done

# Activate the vllm environment
source /home/mlinegar/vllm-env/bin/activate

# Parse config with Python (more reliable than shell YAML parsing)
read_config() {
    python3 -c "
import yaml
import sys

with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)

vllm = cfg.get('vllm', {})
models = vllm.get('models', {})
speculative = cfg.get('speculative', {})
presets = speculative.get('presets', {})

preset_name = '${PRESET}'
profile = '${PROFILE}'

# If preset specified, use it to determine target and draft models
if preset_name:
    if preset_name not in presets:
        print(f'ERROR: Preset \"{preset_name}\" not found. Available: {list(presets.keys())}', file=sys.stderr)
        sys.exit(1)

    preset = presets[preset_name]
    target_profile = preset['target']
    draft_profile = preset.get('draft')
    spec_enabled = preset.get('enabled', False)
    num_spec_tokens = preset.get('num_speculative_tokens', speculative.get('num_speculative_tokens', 5))

    if target_profile not in models:
        print(f'ERROR: Target model \"{target_profile}\" not found. Available: {list(models.keys())}', file=sys.stderr)
        sys.exit(1)

    target = models[target_profile]
    print(f'PROFILE={target_profile}')
    print(f'MODEL_PATH={target[\"path\"]}')
    print(f'TENSOR_PARALLEL={target.get(\"tensor_parallel\", 1)}')
    print(f'MAX_MODEL_LEN={target.get(\"max_model_len\", 8192)}')

    # Speculative decoding settings
    if spec_enabled and draft_profile:
        if draft_profile not in models:
            print(f'ERROR: Draft model \"{draft_profile}\" not found. Available: {list(models.keys())}', file=sys.stderr)
            sys.exit(1)
        draft = models[draft_profile]
        print(f'SPEC_ENABLED=true')
        print(f'DRAFT_MODEL_PATH={draft[\"path\"]}')
        print(f'DRAFT_TENSOR_PARALLEL={draft.get(\"tensor_parallel\", 1)}')
        print(f'NUM_SPEC_TOKENS={num_spec_tokens}')
    else:
        print('SPEC_ENABLED=false')
else:
    # No preset - use profile directly (backwards compatible)
    profile = profile or vllm.get('default', 'small')

    if profile not in models:
        print(f'ERROR: Profile \"{profile}\" not found. Available: {list(models.keys())}', file=sys.stderr)
        sys.exit(1)

    m = models[profile]
    print(f'PROFILE={profile}')
    print(f'MODEL_PATH={m[\"path\"]}')
    print(f'TENSOR_PARALLEL={m.get(\"tensor_parallel\", 1)}')
    print(f'MAX_MODEL_LEN={m.get(\"max_model_len\", 8192)}')
    print('SPEC_ENABLED=false')

print(f'HOST={vllm.get(\"host\", \"0.0.0.0\")}')
print(f'PORT={vllm.get(\"port\", 8000)}')
print(f'GPU_MEM={vllm.get(\"gpu_memory_utilization\", 0.90)}')
print(f'PREFIX_CACHE={str(vllm.get(\"enable_prefix_caching\", False)).lower()}')
"
}

# Load config
eval "$(read_config)"

# Apply command-line overrides
if [[ -n "$PORT_OVERRIDE" ]]; then
    PORT="$PORT_OVERRIDE"
fi

echo "=========================================="
echo "Starting vLLM Server"
echo "=========================================="
echo "Profile: $PROFILE"
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Tensor Parallel: $TENSOR_PARALLEL"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Prefix Cache: $PREFIX_CACHE"
if [[ -n "$KV_CACHE_DTYPE" ]]; then
    echo "KV Cache DType: $KV_CACHE_DTYPE"
fi
if [[ "$SPEC_ENABLED" == "true" ]]; then
    echo "------------------------------------------"
    echo "Speculative Decoding: ENABLED"
    echo "Draft Model: $DRAFT_MODEL_PATH"
    echo "Draft Tensor Parallel: $DRAFT_TENSOR_PARALLEL"
    echo "Speculative Tokens: $NUM_SPEC_TOKENS"
fi
echo "=========================================="

# Build vLLM command
VLLM_CMD=(
    python -m vllm.entrypoints.openai.api_server
    --model "$MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM"
    --trust-remote-code
    --enforce-eager
)

# Add prefix caching unless explicitly overridden
PREFIX_FLAG_PRESENT=false
for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == "--enable-prefix-caching" || "$arg" == "--disable-prefix-caching" ]]; then
        PREFIX_FLAG_PRESENT=true
        break
    fi
done
if [[ "$PREFIX_CACHE" == "true" && "$PREFIX_FLAG_PRESENT" == "false" ]]; then
    VLLM_CMD+=(--enable-prefix-caching)
fi

# Add --kv-cache-dtype if specified
if [[ -n "$KV_CACHE_DTYPE" ]]; then
    VLLM_CMD+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

# Add speculative decoding flags if enabled
if [[ "$SPEC_ENABLED" == "true" ]]; then
    VLLM_CMD+=(
        --speculative-model "$DRAFT_MODEL_PATH"
        --num-speculative-tokens "$NUM_SPEC_TOKENS"
        --speculative-draft-tensor-parallel-size "$DRAFT_TENSOR_PARALLEL"
    )
fi

# Launch vLLM server
"${VLLM_CMD[@]}" "${EXTRA_ARGS[@]}"
