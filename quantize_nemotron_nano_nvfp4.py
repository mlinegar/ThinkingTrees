#!/usr/bin/env python3
"""
NVFP4 Quantization for NVIDIA-Nemotron-3-Nano-30B-A3B
Hybrid Mamba-2 + Transformer MoE architecture
"""

import os
import json
os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Configuration
MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_PATH = "/mnt/data/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
OUTPUT_PATH = "/mnt/data/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQ_LENGTH = 2048


def resolve_nvfp4_cfg(mtq):
    if hasattr(mtq, "NVFP4_DEFAULT_CFG"):
        return mtq.NVFP4_DEFAULT_CFG, "NVFP4_DEFAULT_CFG"

    candidates = [name for name in dir(mtq) if "NVFP4" in name and name.endswith("_CFG")]
    if candidates:
        name = sorted(candidates)[0]
        return getattr(mtq, name), name

    available_cfgs = sorted(name for name in dir(mtq) if name.endswith("_CFG"))
    available = ", ".join(available_cfgs) if available_cfgs else "none"
    raise RuntimeError(
        "NVFP4 config not available in modelopt. "
        f"Available configs: {available}. "
        "Install a newer nvidia-modelopt (or TensorRT-Model-Optimizer)."
    )


def env_flag(name, default="1"):
    value = os.environ.get(name, default)
    if value is None:
        return False
    value = value.strip().lower()
    return value not in ("0", "false", "no", "off")


def load_pickle_module(name):
    if not name:
        return None
    try:
        if name == "dill":
            import dill as pickle
            return pickle
        if name == "cloudpickle":
            import cloudpickle as pickle
            return pickle
    except Exception as e:
        print(f"Warning: failed to import {name}: {e}")
    return None


def _set_quantizer_amax(quantizer, value):
    updated = False
    if hasattr(quantizer, "amax"):
        try:
            quantizer.amax = value
            updated = True
        except Exception:
            pass
    if hasattr(quantizer, "_amax"):
        try:
            quantizer._amax = value
            updated = True
        except Exception:
            pass
    return updated


def patch_zero_amax(model, eps=1e-6):
    total = 0
    patched = 0
    seen = set()
    quantizer_attrs = ("input_quantizer", "weight_quantizer", "output_quantizer", "activation_quantizer")
    for module in model.modules():
        candidates = [module]
        for attr in quantizer_attrs:
            q = getattr(module, attr, None)
            if q is not None:
                candidates.append(q)
        for quantizer in candidates:
            qid = id(quantizer)
            if qid in seen:
                continue
            seen.add(qid)
            if not (hasattr(quantizer, "amax") or hasattr(quantizer, "_amax")):
                continue
            amax = getattr(quantizer, "amax", None)
            if amax is None and hasattr(quantizer, "_amax"):
                amax = getattr(quantizer, "_amax")
            if amax is None:
                continue
            total += 1
            if torch.is_tensor(amax):
                if torch.all(amax > 0):
                    continue
                new_amax = amax.clone()
                new_amax[new_amax <= 0] = eps
                if _set_quantizer_amax(quantizer, new_amax):
                    patched += 1
            elif isinstance(amax, (int, float)):
                if amax > 0:
                    continue
                if _set_quantizer_amax(quantizer, eps):
                    patched += 1
    if patched:
        print(f"Patched {patched} quantizers with non-positive amax (out of {total}).")
    return patched


def patch_nvfp4_activation_scaling(eps=1e-6):
    try:
        from modelopt.torch.quantization.qtensor import nvfp4_tensor
    except Exception:
        return False

    original = nvfp4_tensor.NVFP4QTensor.get_activation_scaling_factor

    def safe_get_activation_scaling_factor(*args, **kwargs):
        scaling = original(*args, **kwargs)
        if torch.is_tensor(scaling):
            if torch.any(scaling <= 0):
                scaling = scaling.clone()
                scaling[scaling <= 0] = eps
        elif isinstance(scaling, (int, float)) and scaling <= 0:
            scaling = eps
        return scaling

    nvfp4_tensor.NVFP4QTensor.get_activation_scaling_factor = staticmethod(
        safe_get_activation_scaling_factor
    )
    print("Patched NVFP4 activation scaling to clamp non-positive values.")
    return True


def save_quantized_checkpoint(model, output_dir, model_source, quant_cfg_name):
    if not env_flag("SAVE_QUANTIZED_MODEL", "1"):
        return None

    save_path = os.environ.get("QUANTIZED_MODEL_SAVE_PATH")
    if not save_path:
        save_path = os.path.join(output_dir, "quantized_model.pt")

    print(f"Saving quantized model checkpoint to {save_path}...")
    pickle_module_used = "pickle"
    try:
        torch.save(model, save_path)
    except Exception as e:
        print(f"Warning: failed to save full model checkpoint: {e}")
        pickle_module_used = None
        for name in ("dill", "cloudpickle"):
            pickle_module = load_pickle_module(name)
            if pickle_module is None:
                continue
            try:
                torch.save(model, save_path, pickle_module=pickle_module)
                pickle_module_used = name
                print(f"Saved full model with {name}.")
                break
            except Exception as e2:
                print(f"Warning: failed to save with {name}: {e2}")
        if pickle_module_used is None:
            try:
                state_path = save_path.replace(".pt", "_state.pt")
                torch.save({"state_dict": model.state_dict()}, state_path)
                print(f"Saved state_dict checkpoint to {state_path}")
                return state_path
            except Exception as e2:
                print(f"Warning: failed to save state_dict checkpoint: {e2}")
                return None

    meta = {
        "model_source": model_source,
        "quant_cfg_name": quant_cfg_name,
        "torch_version": torch.__version__,
        "pickle_module": pickle_module_used,
    }
    meta_path = save_path + ".meta.json"
    try:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        print(f"Wrote metadata to {meta_path}")
    except Exception as e:
        print(f"Warning: failed to write metadata: {e}")

    return save_path


def load_quantized_checkpoint(load_path):
    meta_path = load_path + ".meta.json"
    pickle_module = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            pickle_module = meta.get("pickle_module")
        except Exception as e:
            print(f"Warning: failed to read metadata: {e}")

    pickle_impl = load_pickle_module(pickle_module)
    try:
        if pickle_impl is not None:
            obj = torch.load(load_path, pickle_module=pickle_impl)
        else:
            obj = torch.load(load_path)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load quantized checkpoint: {e}")
        return None

    if isinstance(obj, dict) and "state_dict" in obj:
        print(
            "\n❌ ERROR: Loaded a state_dict-only checkpoint. "
            "Re-export requires a full model checkpoint. "
            "Re-run quantization with SAVE_QUANTIZED_MODEL=1 and dill installed."
        )
        return None

    return obj


def main():
    print("=" * 70)
    print("NVFP4 Quantization for Nemotron-3-Nano-30B-A3B")
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

    # For 30B model, should fit comfortably with less aggressive memory limits
    max_memory = {i: "80GiB" for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = "100GiB"

    print(f"\nMax memory config: {max_memory}")

    # Check if model is downloaded, if not download it
    if not os.path.exists(MODEL_PATH):
        print(f"\nModel not found at {MODEL_PATH}")
        print(f"Downloading from {MODEL_ID}...")
        # Will download on first load
        model_source = MODEL_ID
    else:
        print(f"\nUsing local model: {MODEL_PATH}")
        model_source = MODEL_PATH

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_path = os.environ.get("LOAD_QUANTIZED_MODEL_PATH")
    did_quantize = False
    if load_path:
        if not os.path.exists(load_path):
            print(f"\n❌ ERROR: Quantized checkpoint not found at {load_path}")
            return
        print(f"\nLoading quantized model checkpoint from {load_path}...")
        model = load_quantized_checkpoint(load_path)
        if model is None:
            return
        print("Quantized model loaded!")
    else:
        # Load model
        print(f"\nLoading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print(f"Model loaded!")
        did_quantize = True

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

    calib_data = None
    if did_quantize:
        # Prepare calibration data
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
                padding=False,
            )
            if tokens["input_ids"].shape[1] > 50:
                calib_data.append(tokens)
            if len(calib_data) >= NUM_CALIBRATION_SAMPLES:
                break

        print(f"Created {len(calib_data)} calibration samples")

    # ModelOpt imports FSDP symbols on load; stub missing ones for compatibility.
    try:
        import torch.distributed.fsdp as fsdp
    except Exception:
        fsdp = None
    if fsdp is not None:
        patched = []
        if not hasattr(fsdp, "FSDPModule"):
            class FSDPModule:
                pass
            fsdp.FSDPModule = FSDPModule
            patched.append("FSDPModule")
        if not hasattr(fsdp, "MixedPrecisionPolicy"):
            if hasattr(fsdp, "MixedPrecision"):
                fsdp.MixedPrecisionPolicy = fsdp.MixedPrecision
            else:
                class MixedPrecisionPolicy:
                    pass
                fsdp.MixedPrecisionPolicy = MixedPrecisionPolicy
            patched.append("MixedPrecisionPolicy")
        if not hasattr(fsdp, "fully_shard"):
            def fully_shard(*args, **kwargs):
                raise RuntimeError(
                    "fully_shard is not available in this PyTorch build."
                )
            fsdp.fully_shard = fully_shard
            patched.append("fully_shard")
        if patched:
            print(f"Note: patched torch.distributed.fsdp with stubs: {', '.join(patched)}")

    # Import modelopt
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    patch_nvfp4_activation_scaling()
    if did_quantize:
        quant_cfg, quant_cfg_name = resolve_nvfp4_cfg(mtq)
        print(f"Using quantization config: {quant_cfg_name}")

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

                    if i % 10 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

        # Quantize
        print("\n" + "=" * 70)
        print("Starting NVFP4 quantization...")
        print("=" * 70)

        model = mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
        mtq.print_quant_summary(model)
    else:
        quant_cfg_name = "loaded_checkpoint"
        print(f"Using quantization config: {quant_cfg_name}")
        try:
            mtq.print_quant_summary(model)
        except Exception as e:
            print(f"Warning: failed to print quant summary: {e}")

    output_dir = OUTPUT_PATH

    # Export
    print(f"\nExporting to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    patch_zero_amax(model)
    if did_quantize:
        save_quantized_checkpoint(model, output_dir, model_source, quant_cfg_name)
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
