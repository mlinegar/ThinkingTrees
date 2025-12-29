#!/usr/bin/env python3
"""
NVFP4 Quantization script for Qwen3-Nemotron-235B-A22B-GenRM
Using llm-compressor (vllm-project)

This quantizes the 438GB BF16 model to ~110GB NVFP4 format,
making it usable with vLLM on 4x95GB GPUs.
"""

import os
# Set transformers trust remote code globally BEFORE importing
os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
from datasets import load_dataset
import gc

# Configuration
MODEL_PATH = "/mnt/data/models/nvidia/Qwen3-Nemotron-235B-A22B-GenRM"
OUTPUT_PATH = "/mnt/data/models/nvidia/Qwen3-Nemotron-235B-A22B-GenRM-NVFP4"
NUM_CALIBRATION_SAMPLES = 256  # Reduced for faster calibration
MAX_SEQ_LENGTH = 2048


def get_calibration_data(tokenizer, num_samples=256, max_length=2048):
    """Create calibration dataset using preference-style data."""
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
    print("=" * 70)
    print("NVFP4 Quantization for Qwen3-Nemotron-235B-A22B-GenRM")
    print("Using llm-compressor")
    print("=" * 70)

    # Check GPU availability
    print(f"\nGPU(s) available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    total_vram = sum(
        torch.cuda.get_device_properties(i).total_memory
        for i in range(torch.cuda.device_count())
    ) / 1e9
    print(f"  Total VRAM: {total_vram:.1f} GB")

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
    print("This will take a while for a 438GB model...")
    print("Using device_map='auto' to distribute across all GPUs")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print(f"Model loaded successfully!")
    print(f"Model type: {type(model)}")
    if hasattr(model, 'hf_device_map'):
        # Print device distribution summary
        device_counts = {}
        for name, device in model.hf_device_map.items():
            device_counts[device] = device_counts.get(device, 0) + 1
        print(f"Layer distribution: {device_counts}")

    # Get calibration data
    ds = get_calibration_data(tokenizer, NUM_CALIBRATION_SAMPLES, MAX_SEQ_LENGTH)

    # Configure quantization recipe for MoE model
    # Ignore router/gate layers which are critical for MoE routing
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=[
            "re:.*lm_head",
            "re:.*mlp.gate$",          # MoE gate
            "re:.*router.*",            # MoE router
            "re:.*shared_expert_gate.*",  # Shared expert gate
        ],
    )

    print("\nQuantization recipe:")
    print(f"  Scheme: NVFP4 (4-bit weights)")
    print(f"  Targets: Linear layers")
    print(f"  Ignored: lm_head, MoE gates/routers")
    print(f"  Calibration samples: {NUM_CALIBRATION_SAMPLES}")

    print(f"\nStarting quantization...")
    print("=" * 70)

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

    print("\n" + "=" * 70)
    print("Quantization complete!")
    print(f"Output saved to: {OUTPUT_PATH}")
    print("=" * 70)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
