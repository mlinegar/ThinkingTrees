#!/bin/bash
# Quick model throughput comparison script
#
# Usage:
#   ./scripts/compare_models.sh nemotron-30b-fp8 qwen-30b-thinking
#   ./scripts/compare_models.sh nemotron-30b-fp8 qwen-30b-thinking 100  # 100 samples
#   ./scripts/compare_models.sh --parallel nemotron-30b-fp8 qwen-30b-thinking 100  # parallel on 4 GPUs
#
# List available profiles:
#   ./scripts/compare_models.sh --list

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate vLLM environment
source /home/mlinegar/vllm-env/bin/activate

# Handle --list flag
if [[ "$1" == "--list" ]]; then
    python "$PROJECT_ROOT/scripts/benchmark_throughput.py" --list-profiles
    exit 0
fi

# Handle --parallel flag
PARALLEL_FLAG=""
if [[ "$1" == "--parallel" ]]; then
    PARALLEL_FLAG="--parallel"
    shift
fi

# Check arguments
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 [--parallel] <profile-a> <profile-b> [samples]"
    echo ""
    echo "Examples:"
    echo "  $0 nemotron-30b-fp8 qwen-30b-thinking"
    echo "  $0 nemotron-30b-fp8 qwen-30b-thinking 100"
    echo "  $0 --parallel nemotron-30b-fp8 qwen-30b-thinking 100  # Both models on 4 GPUs"
    echo ""
    echo "Parallel mode runs both models simultaneously:"
    echo "  - Model A on GPUs 0,1 (tensor_parallel=2)"
    echo "  - Model B on GPUs 2,3 (tensor_parallel=2)"
    echo ""
    echo "Use --list to see available model profiles"
    exit 1
fi

PROFILE_A="$1"
PROFILE_B="$2"
SAMPLES="${3:-50}"

echo "========================================"
echo "Model Comparison: $PROFILE_A vs $PROFILE_B"
echo "Samples: $SAMPLES"
if [[ -n "$PARALLEL_FLAG" ]]; then
    echo "Mode: PARALLEL (both models on separate GPU sets)"
else
    echo "Mode: SEQUENTIAL (one model at a time)"
fi
echo "========================================"
echo ""

# Use multiple Western European countries for full manifesto texts
# Sweden(11), Norway(12), Denmark(13), Netherlands(22), Belgium(23), France(31), UK(51)
python "$PROJECT_ROOT/scripts/benchmark_throughput.py" \
    --profile-a "$PROFILE_A" \
    --profile-b "$PROFILE_B" \
    --samples "$SAMPLES" \
    --countries 11 12 13 22 23 31 51 \
    --min-year 1990 \
    --chunk-size 2000 \
    $PARALLEL_FLAG
