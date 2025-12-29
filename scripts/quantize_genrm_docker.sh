#!/bin/bash
# NVFP4 Quantization for Qwen3-Nemotron-235B-A22B-GenRM using TensorRT Model Optimizer
# Uses NVIDIA's official Docker image for quantization

set -e

# Configuration
MODEL_PATH="/mnt/data/models/nvidia/Qwen3-Nemotron-235B-A22B-GenRM"
OUTPUT_DIR="/mnt/data/models/nvidia/Qwen3-Nemotron-235B-A22B-GenRM-NVFP4"
TENSOR_PARALLEL=4

# Create output directory
mkdir -p "$OUTPUT_DIR"
chmod 755 "$OUTPUT_DIR"

echo "========================================"
echo "NVFP4 Quantization for GenRM"
echo "========================================"
echo "Input:  $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "TP:     $TENSOR_PARALLEL"
echo "========================================"

# Check if HF_TOKEN is set
if [[ -z "$HF_TOKEN" ]]; then
    echo "Note: HF_TOKEN not set. Using local model path."
fi

# Run quantization with Docker
docker run --rm -it --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$OUTPUT_DIR:/workspace/output_models" \
  -v "$MODEL_PATH:/workspace/input_model:ro" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ${HF_TOKEN:+-e HF_TOKEN=$HF_TOKEN} \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c "
    set -e
    echo 'Installing TensorRT Model Optimizer...'
    git clone -b 0.35.0 --single-branch https://github.com/NVIDIA/TensorRT-Model-Optimizer.git /app/TensorRT-Model-Optimizer
    cd /app/TensorRT-Model-Optimizer
    pip install -e '.[dev]'

    echo 'Starting NVFP4 quantization...'
    export ROOT_SAVE_PATH='/workspace/output_models'

    python -c '
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from datasets import load_dataset
from tqdm import tqdm

MODEL_PATH = \"/workspace/input_model\"
OUTPUT_PATH = \"/workspace/output_models\"
NUM_SAMPLES = 256
MAX_LENGTH = 2048

print(\"Loading tokenizer...\")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(\"Loading calibration data...\")
ds = load_dataset(\"wikitext\", \"wikitext-2-raw-v1\", split=\"train\")
texts = [t for t in ds[\"text\"] if len(t.strip()) > 100][:NUM_SAMPLES]

calib_data = []
for text in tqdm(texts, desc=\"Tokenizing\"):
    tokens = tokenizer(text, return_tensors=\"pt\", max_length=MAX_LENGTH, truncation=True, padding=\"max_length\")
    if tokens[\"input_ids\"].shape[1] > 100:
        calib_data.append(tokens)
    if len(calib_data) >= NUM_SAMPLES:
        break

print(f\"Created {len(calib_data)} calibration samples\")

print(\"Loading model...\")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=\"auto\",
    trust_remote_code=True,
)

device = next(model.parameters()).device

def forward_loop(model):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(calib_data, desc=\"Calibrating\")):
            try:
                model(input_ids=batch[\"input_ids\"].to(device), attention_mask=batch[\"attention_mask\"].to(device))
            except Exception as e:
                print(f\"Warning batch {i}: {e}\")
            if i % 50 == 0:
                torch.cuda.empty_cache()

print(\"Quantizing to NVFP4...\")
model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop=forward_loop)
mtq.print_quant_summary(model)

print(f\"Exporting to {OUTPUT_PATH}...\")
export_hf_checkpoint(model, OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print(\"Done!\")
'
  "

echo ""
echo "========================================"
echo "Quantization complete!"
echo "Output: $OUTPUT_DIR"
echo "========================================"
