#!/usr/bin/env python3
"""
NVFP4 Quantization with aggressive CPU offloading.
Keeps most of the model on CPU, only loads layers to GPU for calibration.
"""

import os
os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Configuration
MODEL_PATH = "/mnt/data/models/nvidia/Qwen3-Nemotron-235B-A22B-GenRM"
OUTPUT_PATH = "/mnt/data/models/nvidia/Qwen3-Nemotron-235B-A22B-GenRM-NVFP4"
NUM_CALIBRATION_SAMPLES = 128  # Reduced
MAX_SEQ_LENGTH = 1024  # Reduced

def main():
    print("=" * 70)
    print("NVFP4 Quantization with CPU Offloading")
    print("=" * 70)

    # Check output directory permissions FIRST (before wasting hours)
    print(f"\nValidating output directory: {OUTPUT_PATH}")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    test_file = os.path.join(OUTPUT_PATH, ".write_test")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"  ✓ Write permission confirmed")
    except PermissionError:
        print(f"\n❌ ERROR: Cannot write to {OUTPUT_PATH}")
        print("Fix with: sudo chown -R $(whoami) " + OUTPUT_PATH)
        return
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return

    # Check GPU
    print(f"\nGPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    # Use 90GB per GPU - more aggressive, faster calibration
    max_memory = {i: "90GiB" for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = "200GiB"  # Allow plenty of CPU RAM

    print(f"\nMax memory config: {max_memory}")

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with memory limits
    print(f"\nLoading model with CPU offloading...")
    print(f"Model path: {MODEL_PATH}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        offload_folder="/tmp/offload",  # Disk offload if needed
    )

    print(f"Model loaded!")
    if hasattr(model, 'hf_device_map'):
        device_counts = {}
        for name, device in model.hf_device_map.items():
            device_counts[device] = device_counts.get(device, 0) + 1
        print(f"Layer distribution: {device_counts}")

    # Clear cache
    gc.collect()
    torch.cuda.empty_cache()

    # Check memory after loading
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    # Prepare calibration data (minimal)
    print(f"\nPreparing calibration data ({NUM_CALIBRATION_SAMPLES} samples)...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 100][:NUM_CALIBRATION_SAMPLES * 2]

    calib_data = []
    for text in texts:
        tokens = tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding=False,  # No padding to save memory
        )
        if tokens["input_ids"].shape[1] > 50:
            calib_data.append(tokens)
        if len(calib_data) >= NUM_CALIBRATION_SAMPLES:
            break

    print(f"Created {len(calib_data)} calibration samples")

    # Import modelopt
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint

    # Get primary device
    device = next(model.parameters()).device
    print(f"\nPrimary device: {device}")

    # Create forward loop for calibration
    def forward_loop(model_to_calib):
        model_to_calib.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(calib_data, desc="Calibrating")):
                try:
                    input_ids = batch["input_ids"].to(device)
                    model_to_calib(input_ids=input_ids)
                except Exception as e:
                    print(f"Warning batch {i}: {e}")
                    continue

                # Aggressive cleanup every batch
                if i % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

    # Quantize
    print("\n" + "=" * 70)
    print("Starting NVFP4 quantization...")
    print("=" * 70)

    output_dir = OUTPUT_PATH
    try:
        model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop=forward_loop)
        mtq.print_quant_summary(model)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nOOM Error. Trying FP8 instead (less memory overhead)...")
            gc.collect()
            torch.cuda.empty_cache()

            # Reload model
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
            )

            model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop=forward_loop)
            mtq.print_quant_summary(model)
            output_dir = OUTPUT_PATH.replace("NVFP4", "FP8")
        else:
            raise

    # Export (ensure dir exists for FP8 fallback case)
    print(f"\nExporting to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    export_hf_checkpoint(model, export_dir=output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 70)
    print("Quantization complete!")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
