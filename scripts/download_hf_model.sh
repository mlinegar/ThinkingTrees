#!/bin/bash
# Download a Hugging Face model to cache or to a specific local directory.
#
# Usage:
#   ./scripts/download_hf_model.sh
#   ./scripts/download_hf_model.sh Qwen/Qwen3-Embedding-8B
#   ./scripts/download_hf_model.sh Qwen/Qwen3-Embedding-8B /mnt/data/models/Qwen/Qwen3-Embedding-8B

set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen3-Embedding-8B}"
LOCAL_DIR="${2:-}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "ERROR: huggingface-cli not found."
    echo "Install with: pip install -U huggingface_hub"
    exit 1
fi

echo "Downloading model: ${MODEL_ID}"

if [[ -n "${LOCAL_DIR}" ]]; then
    mkdir -p "${LOCAL_DIR}"
    huggingface-cli download "${MODEL_ID}" \
        --local-dir "${LOCAL_DIR}" \
        --local-dir-use-symlinks False
    echo "Download complete: ${LOCAL_DIR}"
else
    huggingface-cli download "${MODEL_ID}"
    echo "Download complete in Hugging Face cache."
fi

echo ""
echo "To start the tiny embedding server:"
echo "  ./scripts/start_vllm.sh qwen3-embedding-8b --port 8003"
