#!/usr/bin/env python3
"""
NVFP4 Quantization script for NVIDIA Nemotron-3-Nano-30B-A3B
Using llm-compressor (vllm-project)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
import gc

# Configuration
MODEL_PATH = "/mnt/data/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
OUTPUT_PATH = "/mnt/data/models/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQ_LENGTH = 2048


def main():
    print("=" * 60)
    print("NVFP4 Quantization for Nemotron-3-Nano-30B-A3B")
    print("Using llm-compressor")
    print("=" * 60)

    # Check GPU availability
    print(f"\nGPU(s) available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    # Configure quantization recipe
    # Exclude layers that shouldn't be quantized:
    # - lm_head
    # - MoE gate layers
    # - Mamba conv1d layers
    # - Router layers
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=[
            "re:.*lm_head",
            "re:.*mlp.gate$",             # MoE gates
            "re:.*router.*",               # Router layers
            "re:.*mixer.conv1d.*",         # Mamba conv1d
            "re:.*shared_expert_gate.*",   # Shared expert gates
        ],
    )

    print("\nQuantization recipe:")
    print(f"  Scheme: NVFP4")
    print(f"  Targets: Linear layers")
    print(f"  Ignored patterns: {recipe.ignore}")

    print(f"\nLoading model and running quantization...")
    print(f"This will take 1-4 hours depending on calibration...")

    # Run oneshot quantization
    # This handles model loading, calibration, and quantization in one call
    oneshot(
        model=MODEL_PATH,
        dataset="ultrachat_200k",  # Common calibration dataset
        recipe=recipe,
        output_dir=OUTPUT_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        trust_remote_code_model=True,
    )

    print("\n" + "=" * 60)
    print("Quantization complete!")
    print(f"Output saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
