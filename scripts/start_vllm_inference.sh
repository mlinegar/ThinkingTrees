#!/bin/bash
# vLLM Server for Inference (Small Model with Optimized Prompts)
#
# Starts vLLM with:
#   - Model: Qwen3-30B-Thinking (the small model)
#   - No speculative decoding (not needed for inference)
#
# This is designed to be used AFTER training with start_vllm_training.sh:
#   1. Training: Run DSPy optimization with large model + speculative decoding
#   2. Inference: Deploy small model with the DSPy-optimized prompts/demos
#
# The optimized prompts contain "distilled" knowledge from the large model,
# helping the small model perform better at inference time.
#
# Usage:
#   ./scripts/start_vllm_inference.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting vLLM for Inference (Small Model)"
echo "Model: Qwen3-30B-Thinking with DSPy-optimized prompts"
echo ""

exec "$SCRIPT_DIR/start_vllm.sh" --preset inference "$@"
