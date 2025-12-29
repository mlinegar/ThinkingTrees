#!/usr/bin/env python3
"""
NVFP4 Quantization script for NVIDIA Nemotron-3-Nano-30B-A3B
Using llm-compressor (vllm-project)
Manual model loading to handle trust_remote_code properly
"""

import os
# Set transformers trust remote code globally BEFORE importing
os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
from datasets import load_dataset
import gc

# Configuration
MODEL_PATH = "/mnt/data/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
OUTPUT_PATH = "/mnt/data/models/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQ_LENGTH = 2048


def get_calibration_data(tokenizer, num_samples=512, max_length=2048):
    """Create calibration dataset."""
    print("Loading ultrachat dataset for calibration...")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=42).select(range(num_samples))

    def preprocess(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    ds = ds.map(preprocess)
    return ds


def main():
    print("=" * 60)
    print("NVFP4 Quantization for Nemotron-3-Nano-30B-A3B")
    print("Using llm-compressor with manual model loading")
    print("=" * 60)

    # Check GPU availability
    print(f"\nGPU(s) available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    # Load tokenizer first
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model explicitly with trust_remote_code
    print(f"\nLoading model from {MODEL_PATH}...")
    print("This may take several minutes for a 59GB model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")

    # Get calibration data
    ds = get_calibration_data(tokenizer, NUM_CALIBRATION_SAMPLES, MAX_SEQ_LENGTH)

    # Configure quantization recipe
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=[
            "re:.*lm_head",
            "re:.*mlp.gate$",
            "re:.*router.*",
            "re:.*mixer.conv1d.*",
            "re:.*shared_expert_gate.*",
        ],
    )

    print("\nQuantization recipe:")
    print(f"  Scheme: NVFP4")
    print(f"  Targets: Linear layers")
    print(f"  Ignored patterns: {recipe.ignore}")

    print(f"\nStarting quantization...")
    print("=" * 60)

    # Run oneshot with pre-loaded model
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=ds,
        recipe=recipe,
        output_dir=OUTPUT_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    print("\n" + "=" * 60)
    print("Quantization complete!")
    print(f"Output saved to: {OUTPUT_PATH}")
    print("=" * 60)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
