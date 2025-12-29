#!/bin/bash
# Quantize Nemotron-Nano-30B using Docker container

MODEL_DIR="/mnt/data/models/nvidia"
SCRIPT_DIR="/home/mlinegar/ThinkingTrees"
PYTORCH_IMAGE="${PYTORCH_IMAGE:-nvcr.io/nvidia/pytorch:25.02-py3}"

echo "Starting Nemotron-Nano NVFP4 quantization in Docker..."
echo "Using PyTorch image: ${PYTORCH_IMAGE}"

sudo docker run --runtime nvidia --gpus all \
    -v ${MODEL_DIR}:${MODEL_DIR} \
    -v ${SCRIPT_DIR}:${SCRIPT_DIR} \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/pip:/root/.cache/pip \
    -e HF_HUB_TRUST_REMOTE_CODE=1 \
    -e PIP_CACHE_DIR=/root/.cache/pip \
    -e MODEL_OPT_VERSION \
    -e TRANSFORMERS_VERSION \
    -e SAVE_QUANTIZED_MODEL \
    -e QUANTIZED_MODEL_SAVE_PATH \
    -e LOAD_QUANTIZED_MODEL_PATH \
    --ipc=host \
    --rm \
    --entrypoint /bin/bash \
    ${PYTORCH_IMAGE} \
    -c "
        set -e
        echo '=== Checking environment ==='
        python3 -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}\")'
        nvidia-smi --query-gpu=name,memory.total --format=csv

        echo '=== Freezing base package versions ==='
        python3 - <<'PY' > /tmp/pip_constraints.txt
import importlib.metadata as md

pins = [
    \"torch\",
    \"torchvision\",
    \"torchaudio\",
    \"numpy\",
    \"packaging\",
    \"six\",
    \"pyarrow\",
    \"fsspec\",
    \"transformers\",
    \"tokenizers\",
    \"datasets\",
    \"accelerate\",
    \"scipy\",
    \"safetensors\",
    \"huggingface-hub\",
]

for name in pins:
    try:
        version = md.version(name)
    except md.PackageNotFoundError:
        continue
    print(f\"{name}=={version}\")
PY
        export PIP_CONSTRAINT=/tmp/pip_constraints.txt

        echo '=== Installing dependencies ==='
        TRANSFORMERS_VERSION=\${TRANSFORMERS_VERSION:-4.48.2}
        # Check if nvcc is available for building
        if command -v nvcc >/dev/null 2>&1; then
            nvcc --version
            BUILD_FROM_SOURCE=1
        else
            echo 'No nvcc - will try prebuilt'
            BUILD_FROM_SOURCE=0
        fi

        # Force rebuild from source with Blackwell (SM 120) support
        export TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0;10.0;12.0'
        # Don't upgrade - use container's existing compatible versions
        python3 -m pip install -q datasets tqdm accelerate "transformers==\${TRANSFORMERS_VERSION}" tokenizers safetensors huggingface-hub einops ninja dill cloudpickle
        if [ -n \"\$MODEL_OPT_VERSION\" ]; then
            python3 -m pip install -q --no-deps \"nvidia-modelopt==\$MODEL_OPT_VERSION\"
        else
            python3 -m pip install -q --no-deps --upgrade nvidia-modelopt
        fi
        if [ \"\$BUILD_FROM_SOURCE\" -eq 1 ]; then
            python3 -m pip install --no-deps --no-build-isolation --no-binary causal-conv1d --no-cache-dir --force-reinstall causal-conv1d
            python3 -m pip install --no-deps --no-build-isolation --no-binary mamba-ssm --no-cache-dir --force-reinstall mamba-ssm
        else
            python3 -m pip install --no-deps --no-cache-dir --force-reinstall causal-conv1d
            python3 -m pip install --no-deps --no-cache-dir --force-reinstall mamba-ssm
        fi

        echo '=== Running quantization ==='
        cd ${SCRIPT_DIR}
        python3 quantize_nemotron_nano_nvfp4.py 2>&1 | tee ${SCRIPT_DIR}/quantize_nano_docker.log
    "

echo "Done!"
