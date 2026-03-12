#!/usr/bin/env python3
"""
General NVFP4 conversion utility using llmcompressor oneshot.

Supports:
- Presets (including Qwen3.5-397B-A17B defaults)
- JSON config files
- CLI overrides for per-run adjustments
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

# Keep trust-remote-code on by default for large custom architectures.
os.environ.setdefault("HF_HUB_TRUST_REMOTE_CODE", "1")
os.environ.setdefault("TRUST_REMOTE_CODE", "1")

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText
except Exception:  # pragma: no cover - depends on transformers version
    AutoModelForImageTextToText = None


DEFAULT_CONFIG: dict[str, Any] = {
    "model": "",
    "output_dir": "",
    "output_root": "",
    "dtype": "bfloat16",
    "device_map": "auto",
    "trust_remote_code": True,
    "low_cpu_mem_usage": True,
    "dataset": "HuggingFaceH4/ultrachat_200k",
    "dataset_config": None,
    "dataset_split": "train_sft",
    "text_field": "text",
    "messages_field": "messages",
    "calibration_sources": [],
    "shuffle_calibration": True,
    "seed": 42,
    "num_calibration_samples": 256,
    "max_seq_length": 2048,
    "moe_calibrate_all_experts": True,
    "quantization_aware_calibration": True,
    "targets": "Linear",
    "scheme": "NVFP4",
    "ignore": [
        "re:.*lm_head",
        "re:.*mlp.gate$",
        "re:.*mlp.shared_expert_gate$",
        "re:.*router.*",
        "re:.*linear_attn.*",
    ],
    "model_loader": "auto",
}


PRESETS: dict[str, dict[str, Any]] = {
    "qwen3_5_397b": {
        "model": "Qwen/Qwen3.5-397B-A17B",
        "output_dir": "/mnt/raid0/huggingface/nvfp4/Qwen3.5-397B-A17B-NVFP4",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "dataset": "HuggingFaceH4/ultrachat_200k",
        "dataset_config": None,
        "dataset_split": "train_sft",
        "messages_field": "messages",
        "text_field": "text",
        "calibration_sources": [],
        "shuffle_calibration": True,
        "seed": 42,
        "num_calibration_samples": 256,
        "max_seq_length": 2048,
        "moe_calibrate_all_experts": True,
        "quantization_aware_calibration": True,
        "targets": "Linear",
        "scheme": "NVFP4",
        "ignore": [
            "re:.*lm_head",
            "re:.*mlp.gate$",
            "re:.*mlp.shared_expert_gate$",
            "re:.*router.*",
            "re:.*linear_attn.*",
        ],
    }
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="General NVFP4 conversion with llmcompressor oneshot."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="qwen3_5_397b",
        help="Built-in preset to start from.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to JSON config file.",
    )
    parser.add_argument("--model", type=str, default="", help="Model ID or local path.")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Used only if output_dir is not set.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="",
        help="Model load dtype.",
    )
    parser.add_argument("--device-map", type=str, default="", help="Transformers device_map.")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Set trust_remote_code for tokenizer/model loading.",
    )
    parser.add_argument(
        "--low-cpu-mem-usage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Set low_cpu_mem_usage for model loading.",
    )
    parser.add_argument("--dataset", type=str, default="", help="HF dataset name.")
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="",
        help="Optional HF dataset config/subset name.",
    )
    parser.add_argument("--dataset-split", type=str, default="", help="Dataset split.")
    parser.add_argument("--text-field", type=str, default="", help="Text field name.")
    parser.add_argument("--messages-field", type=str, default="", help="Messages field name.")
    parser.add_argument(
        "--calibration-sources",
        type=str,
        default="",
        help=(
            "JSON list describing mixed calibration sources. "
            "Each source can set dataset/dataset_config/dataset_split/"
            "text_field/messages_field/num_samples."
        ),
    )
    parser.add_argument(
        "--shuffle-calibration",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Shuffle dataset before selecting calibration examples.",
    )
    parser.add_argument("--seed", type=int, default=-1, help="Random seed.")
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=-1,
        help="Number of calibration samples.",
    )
    parser.add_argument("--max-seq-length", type=int, default=-1, help="Max seq length.")
    parser.add_argument(
        "--moe-calibrate-all-experts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to calibrate all MoE experts.",
    )
    parser.add_argument(
        "--quantization-aware-calibration",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to enable quantization-aware calibration.",
    )
    parser.add_argument("--targets", type=str, default="", help="Quant targets.")
    parser.add_argument("--scheme", type=str, default="", help="Quant scheme.")
    parser.add_argument(
        "--model-loader",
        choices=["auto", "causal", "image_text"],
        default="",
        help=(
            "Model class to load before quantization. "
            "'auto' selects image_text when architecture is ConditionalGeneration."
        ),
    )
    parser.add_argument(
        "--ignore",
        nargs="+",
        default=[],
        help="Override ignore regex list.",
    )
    parser.add_argument(
        "--append-ignore",
        action="append",
        default=[],
        help="Append one ignore regex (can be passed multiple times).",
    )
    parser.add_argument(
        "--save-effective-config",
        type=str,
        default="",
        help="Optional path to save merged runtime config as JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print effective config and exit.",
    )
    return parser.parse_args()


def load_json_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_model_name(model: str) -> str:
    name = model.strip().strip("/").split("/")[-1]
    return name.replace(" ", "_")


def resolve_output_dir(cfg: dict[str, Any]) -> str:
    if cfg.get("output_dir"):
        return cfg["output_dir"]

    output_root = cfg.get("output_root")
    if not output_root:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            output_root = os.path.join(hf_home, "nvfp4")
        else:
            output_root = os.path.join(str(Path.home()), ".cache", "huggingface", "nvfp4")

    model_name = normalize_model_name(cfg["model"])
    return os.path.join(output_root, f"{model_name}-NVFP4")


def to_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def extract_text(
    example: dict[str, Any],
    text_field: str,
    messages_field: str,
    tokenizer: AutoTokenizer,
) -> dict[str, str]:
    if messages_field and messages_field in example and example[messages_field]:
        text = tokenizer.apply_chat_template(
            example[messages_field],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}
    if text_field and text_field in example and example[text_field]:
        return {"text": str(example[text_field])}
    if "prompt" in example and "completion" in example:
        return {"text": f"{example['prompt']}\n{example['completion']}"}

    keys = ", ".join(sorted(example.keys()))
    raise KeyError(f"Could not find text/messages fields in dataset row. Available keys: {keys}")


def prepare_single_source_dataset(
    source_cfg: dict[str, Any],
    tokenizer: AutoTokenizer,
    global_cfg: dict[str, Any],
    source_idx: int,
) -> Dataset:
    dataset_name = source_cfg.get("dataset") or global_cfg["dataset"]
    dataset_config = source_cfg.get("dataset_config", global_cfg.get("dataset_config"))
    dataset_split = source_cfg.get("dataset_split") or global_cfg["dataset_split"]
    text_field = source_cfg.get("text_field") or global_cfg["text_field"]
    messages_field = source_cfg.get("messages_field") or global_cfg["messages_field"]
    sample_count = int(source_cfg["num_samples"])
    seed = int(global_cfg["seed"])

    print(
        f"Loading calibration source {source_idx + 1}: "
        f"{dataset_name} [{dataset_split}] ({sample_count} samples)"
    )
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
    else:
        ds = load_dataset(dataset_name, split=dataset_split)

    if global_cfg.get("shuffle_calibration", True):
        ds = ds.shuffle(seed=seed + source_idx)

    sample_count = min(sample_count, len(ds))
    ds = ds.select(range(sample_count))

    ds = ds.map(
        lambda row: extract_text(row, text_field, messages_field, tokenizer),
        remove_columns=ds.column_names,
        desc=f"Formatting source {source_idx + 1}",
    )
    return ds


def prepare_calibration_dataset(cfg: dict[str, Any], tokenizer: AutoTokenizer) -> Dataset:
    total = int(cfg["num_calibration_samples"])
    sources = cfg.get("calibration_sources") or []

    if not sources:
        sources = [
            {
                "dataset": cfg["dataset"],
                "dataset_config": cfg.get("dataset_config"),
                "dataset_split": cfg["dataset_split"],
                "text_field": cfg["text_field"],
                "messages_field": cfg["messages_field"],
                "num_samples": total,
            }
        ]
    else:
        explicit = sum(int(s.get("num_samples", 0)) for s in sources)
        missing_idxs = [i for i, s in enumerate(sources) if "num_samples" not in s]
        remaining = max(total - explicit, 0)
        if missing_idxs:
            even = remaining // len(missing_idxs)
            extra = remaining % len(missing_idxs)
            for j, idx in enumerate(missing_idxs):
                sources[idx]["num_samples"] = even + (1 if j < extra else 0)

    source_datasets: list[Dataset] = []
    for i, source in enumerate(sources):
        if int(source.get("num_samples", 0)) <= 0:
            continue
        source_datasets.append(prepare_single_source_dataset(source, tokenizer, cfg, i))

    if not source_datasets:
        raise ValueError("No calibration samples selected from calibration_sources.")

    if len(source_datasets) == 1:
        return source_datasets[0]
    return concatenate_datasets(source_datasets)


def build_effective_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    if args.preset:
        cfg = deep_merge(cfg, PRESETS[args.preset])

    if args.config:
        cfg = deep_merge(cfg, load_json_config(args.config))

    if args.model:
        cfg["model"] = args.model
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.output_root:
        cfg["output_root"] = args.output_root
    if args.dtype:
        cfg["dtype"] = args.dtype
    if args.device_map:
        cfg["device_map"] = args.device_map
    if args.trust_remote_code is not None:
        cfg["trust_remote_code"] = args.trust_remote_code
    if args.low_cpu_mem_usage is not None:
        cfg["low_cpu_mem_usage"] = args.low_cpu_mem_usage
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.dataset_config:
        cfg["dataset_config"] = args.dataset_config
    if args.dataset_split:
        cfg["dataset_split"] = args.dataset_split
    if args.text_field:
        cfg["text_field"] = args.text_field
    if args.messages_field:
        cfg["messages_field"] = args.messages_field
    if args.calibration_sources:
        cfg["calibration_sources"] = json.loads(args.calibration_sources)
    if args.shuffle_calibration is not None:
        cfg["shuffle_calibration"] = args.shuffle_calibration
    if args.seed >= 0:
        cfg["seed"] = args.seed
    if args.num_calibration_samples > 0:
        cfg["num_calibration_samples"] = args.num_calibration_samples
    if args.max_seq_length > 0:
        cfg["max_seq_length"] = args.max_seq_length
    if args.moe_calibrate_all_experts is not None:
        cfg["moe_calibrate_all_experts"] = args.moe_calibrate_all_experts
    if args.quantization_aware_calibration is not None:
        cfg["quantization_aware_calibration"] = args.quantization_aware_calibration
    if args.targets:
        cfg["targets"] = args.targets
    if args.scheme:
        cfg["scheme"] = args.scheme
    if args.model_loader:
        cfg["model_loader"] = args.model_loader
    if args.ignore:
        cfg["ignore"] = args.ignore
    if args.append_ignore:
        cfg["ignore"] = list(cfg.get("ignore", [])) + args.append_ignore

    if not cfg.get("model"):
        raise ValueError("Model is required. Set it via preset, config, or --model.")

    cfg["output_dir"] = resolve_output_dir(cfg)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_effective_config(args)

    os.makedirs(cfg["output_dir"], exist_ok=True)

    if args.save_effective_config:
        with open(args.save_effective_config, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)

    print("=" * 80)
    print("NVFP4 Conversion (General)")
    print("=" * 80)
    print(json.dumps(cfg, indent=2, sort_keys=True))

    if args.dry_run:
        print("Dry run complete.")
        return

    print(f"\nLoading tokenizer from: {cfg['model']}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"],
        trust_remote_code=cfg["trust_remote_code"],
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_loader_choice = str(cfg.get("model_loader", "auto")).strip().lower() or "auto"
    model_loader_name = "causal"
    model_loader_cls = AutoModelForCausalLM
    if model_loader_choice == "image_text":
        if AutoModelForImageTextToText is None:
            raise RuntimeError(
                "Requested --model-loader image_text but "
                "AutoModelForImageTextToText is unavailable in this transformers build."
            )
        model_loader_name = "image_text"
        model_loader_cls = AutoModelForImageTextToText
    elif model_loader_choice == "auto":
        try:
            hf_cfg = AutoConfig.from_pretrained(
                cfg["model"],
                trust_remote_code=cfg["trust_remote_code"],
            )
            archs = hf_cfg.architectures or []
            if (
                AutoModelForImageTextToText is not None
                and any("ConditionalGeneration" in a for a in archs)
            ):
                model_loader_name = "image_text"
                model_loader_cls = AutoModelForImageTextToText
        except Exception as exc:
            print(f"Warning: could not auto-detect model loader, falling back to causal ({exc})")
    elif model_loader_choice != "causal":
        raise ValueError(f"Unsupported model_loader: {model_loader_choice}")

    print(f"\nLoading model from: {cfg['model']}")
    print(f"Using model loader: {model_loader_name}")
    model = model_loader_cls.from_pretrained(
        cfg["model"],
        torch_dtype=to_torch_dtype(cfg["dtype"]),
        device_map=cfg["device_map"],
        trust_remote_code=cfg["trust_remote_code"],
        low_cpu_mem_usage=cfg["low_cpu_mem_usage"],
    )

    if hasattr(model, "hf_device_map"):
        device_counts: dict[str, int] = {}
        for _, device in model.hf_device_map.items():
            key = str(device)
            device_counts[key] = device_counts.get(key, 0) + 1
        print(f"Layer distribution: {device_counts}")

    ds = prepare_calibration_dataset(cfg, tokenizer)
    effective_num_calib = len(ds)
    print(f"Total calibration samples after source merge: {effective_num_calib}")

    recipe = QuantizationModifier(
        targets=cfg["targets"],
        scheme=cfg["scheme"],
        ignore=cfg["ignore"],
    )

    print("\nStarting oneshot quantization...")
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=ds,
        recipe=recipe,
        output_dir=cfg["output_dir"],
        max_seq_length=int(cfg["max_seq_length"]),
        num_calibration_samples=effective_num_calib,
        moe_calibrate_all_experts=bool(cfg["moe_calibrate_all_experts"]),
        quantization_aware_calibration=bool(cfg["quantization_aware_calibration"]),
    )

    print("\n" + "=" * 80)
    print("Quantization complete.")
    print(f"Output: {cfg['output_dir']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
