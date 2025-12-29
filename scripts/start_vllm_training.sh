#!/bin/bash
# vLLM Server for Training with Speculative Decoding
#
# Starts vLLM with:
#   - Target: Qwen3-80B (large model for DSPy optimization quality)
#   - Draft: Qwen3-30B-Thinking (fast draft model for speculative decoding)
#   - Result: 1.5-3x faster generation during training
#
# Usage:
#   ./scripts/start_vllm_training.sh              # Uses default "training" preset
#   ./scripts/start_vllm_training.sh heavy        # Uses "training-heavy" preset (235B target)
#
# After training, use start_vllm_inference.sh to deploy the small model
# with the DSPy-optimized prompts/demos.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine preset
PRESET="${1:-training}"
if [[ "$PRESET" == "heavy" ]]; then
    PRESET="training-heavy"
fi

echo "Starting vLLM for Training (Speculative Decoding Enabled)"
echo "Preset: $PRESET"
echo ""

exec "$SCRIPT_DIR/start_vllm.sh" --preset "$PRESET" "${@:2}"
