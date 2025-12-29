#!/usr/bin/env python3
"""
NVFP4 Quantization script for NVIDIA Nemotron-3-Nano-30B-A3B
Using NVIDIA ModelOpt (nvidia-modelopt)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from tqdm import tqdm
import gc

# Configuration
MODEL_PATH = "/mnt/data/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
OUTPUT_PATH = "/mnt/data/models/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 1

def get_calibration_dataloader(tokenizer, num_samples=512, max_length=2048):
    """Create calibration dataloader from wikitext dataset."""
    print(f"Loading calibration dataset (wikitext)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Concatenate texts and tokenize
    texts = [t for t in dataset["text"] if len(t.strip()) > 100][:num_samples * 2]

    calibration_data = []
    for text in tqdm(texts[:num_samples], desc="Tokenizing calibration data"):
        tokens = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        if tokens["input_ids"].shape[1] > 100:  # Skip very short sequences
            calibration_data.append(tokens)
        if len(calibration_data) >= num_samples:
            break

    print(f"Created {len(calibration_data)} calibration samples")
    return calibration_data


def create_forward_loop(model, calibration_data, device):
    """Create forward loop for calibration."""
    def forward_loop(model_to_calib):
        model_to_calib.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(calibration_data, desc="Calibrating")):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                try:
                    model_to_calib(input_ids=input_ids, attention_mask=attention_mask)
                except Exception as e:
                    print(f"Warning: Error in calibration batch {i}: {e}")
                    continue

                # Clear cache periodically
                if i % 50 == 0:
                    torch.cuda.empty_cache()

    return forward_loop


def main():
    print("=" * 60)
    print("NVFP4 Quantization for Nemotron-3-Nano-30B-A3B")
    print("=" * 60)

    # Check GPU availability
    print(f"\nGPU(s) available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    # Load tokenizer
    print(f"\nLoading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create calibration data first (before loading model to save memory)
    calibration_data = get_calibration_dataloader(
        tokenizer,
        num_samples=NUM_CALIBRATION_SAMPLES,
        max_length=MAX_SEQ_LENGTH
    )

    # Load model with accelerate for multi-GPU
    print(f"\nLoading model from {MODEL_PATH}...")
    print("This may take several minutes for a 59GB model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Automatically distribute across GPUs
        trust_remote_code=True,
    )

    print(f"Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'single device'}")

    # Get device for calibration
    device = next(model.parameters()).device
    print(f"Primary device: {device}")

    # Print model structure for debugging
    print("\nModel module names (first 20):")
    for i, (name, _) in enumerate(model.named_modules()):
        if i < 20:
            print(f"  {name}")
        elif i == 20:
            print("  ...")
            break

    # Get quantization config
    print("\nUsing NVFP4_DEFAULT_CFG for quantization")
    quant_cfg = mtq.NVFP4_DEFAULT_CFG
    print(f"Quantization config: {quant_cfg}")

    # Create forward loop
    forward_loop = create_forward_loop(model, calibration_data, device)

    # Quantize
    print("\n" + "=" * 60)
    print("Starting NVFP4 quantization...")
    print("=" * 60)

    model = mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

    # Print quantization summary
    print("\nQuantization summary:")
    mtq.print_quant_summary(model)

    # Export checkpoint
    print(f"\nExporting quantized model to {OUTPUT_PATH}...")
    export_hf_checkpoint(model, OUTPUT_PATH)

    # Copy tokenizer files
    print("Copying tokenizer files...")
    tokenizer.save_pretrained(OUTPUT_PATH)

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
