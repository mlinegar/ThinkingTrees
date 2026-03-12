#!/usr/bin/env python3
"""Train a summary SFT model from teacher-trace summary pairs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml


LOGGER = logging.getLogger(__name__)
DEFAULT_STUDENT_MODEL = "/mnt/data/models/AxionML/Qwen3.5-35B-A3B-NVFP4"


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_summary_prompt(
    *,
    input_text: str,
    source_rile_raw: float,
    hop: int,
    include_score_conditioning: bool = False,
) -> str:
    rendered_input = str(input_text or "").strip()
    lines: List[str] = [
        "You are an oracle-preserving summarizer for manifesto documents.",
        "Preserve directional stance, factual commitments, entities, and caveats.",
        "Do NOT mention any numeric score or the term RILE.",
    ]
    if include_score_conditioning:
        lines.append(f"Target directional score to preserve: {float(source_rile_raw):.2f}")
    lines.append(f"Resummary hop: {int(hop)}")
    lines.append("")
    lines.append("INPUT_TEXT:")
    lines.append(rendered_input)
    lines.append("")
    lines.append("SUMMARY:")
    return "\n".join(lines).strip()


def build_sft_example(row: Dict[str, Any], *, include_score_conditioning: bool = False) -> Dict[str, Any]:
    input_text = str(row.get("input_text", "") or "")
    target_summary = str(row.get("target_summary", "") or "").strip()
    source_rile_raw = float(row.get("source_rile_raw", 0.0) or 0.0)
    hop = int(row.get("hop", 1) or 1)

    prompt = build_summary_prompt(
        input_text=input_text,
        source_rile_raw=source_rile_raw,
        hop=hop,
        include_score_conditioning=bool(include_score_conditioning),
    )
    text = f"{prompt}\n{target_summary}".strip()

    return {
        "id": str(row.get("id", "") or ""),
        "example_id": str(row.get("example_id", "") or ""),
        "split": str(row.get("split", "") or ""),
        "hop": hop,
        "source_rile_raw": source_rile_raw,
        "input_text": input_text,
        "prompt": prompt,
        "target_summary": target_summary,
        "text": text,
    }


def build_sft_dataset(
    rows: Sequence[Dict[str, Any]],
    *,
    include_score_conditioning: bool = False,
) -> List[Dict[str, Any]]:
    built: List[Dict[str, Any]] = []
    for row in rows:
        example = build_sft_example(row, include_score_conditioning=bool(include_score_conditioning))
        if not example["input_text"] or not example["target_summary"]:
            continue
        built.append(example)
    return built


def resolve_model_path(model_or_profile: str, settings_path: Path) -> str:
    rendered = str(model_or_profile or "").strip()
    if not rendered:
        raise ValueError("Model/profile value cannot be empty")

    direct = Path(rendered)
    if direct.exists():
        return str(direct)

    settings = yaml.safe_load(Path(settings_path).read_text(encoding="utf-8")) or {}
    vllm = settings.get("vllm", {}) if isinstance(settings, dict) else {}
    models = vllm.get("models", {}) if isinstance(vllm, dict) else {}
    profile = models.get(rendered)
    if isinstance(profile, dict) and profile.get("path"):
        return str(profile["path"])

    return rendered


def _build_sft_config(sft_config_cls: Any, *, args: argparse.Namespace, output_dir: Path) -> Any:
    params = set(inspect.signature(sft_config_cls.__init__).parameters.keys())
    kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": float(args.learning_rate),
        "num_train_epochs": float(args.epochs),
        "per_device_train_batch_size": int(args.per_device_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "logging_steps": int(args.logging_steps),
        "save_steps": int(args.save_steps),
        "seed": int(args.seed),
    }
    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "steps" if args.eval_pairs.exists() else "no"
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "steps" if args.eval_pairs.exists() else "no"
    if "eval_steps" in params:
        kwargs["eval_steps"] = int(args.eval_steps)
    if "bf16" in params:
        kwargs["bf16"] = bool(args.bf16)
    if "max_seq_length" in params:
        kwargs["max_seq_length"] = int(args.max_seq_length)
    return sft_config_cls(**kwargs)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train manifesto summary SFT model from summary pairs")

    parser.add_argument("--train-pairs", type=Path, default=Path("summary_pairs_train.jsonl"))
    parser.add_argument("--eval-pairs", type=Path, default=Path("summary_pairs_val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--base-model", type=str, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--settings-path", type=Path, default=Path("config/settings.yaml"))

    parser.add_argument(
        "--include-score-conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include the teacher-provided source_rile_raw in the prompt. "
            "This is label-conditioning; keep disabled to avoid leakage unless you will also "
            "provide an estimated score at inference."
        ),
    )

    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=4096)

    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="*",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.train_pairs.exists():
        raise FileNotFoundError(f"Train pairs file not found: {args.train_pairs}")
    if not args.eval_pairs.exists():
        LOGGER.warning("Eval pairs file not found: %s; continuing without eval set", args.eval_pairs)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("outputs") / f"manifesto_summary_sft_{_now_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _load_jsonl(args.train_pairs)
    eval_rows = _load_jsonl(args.eval_pairs) if args.eval_pairs.exists() else []

    train_dataset_rows = build_sft_dataset(train_rows, include_score_conditioning=bool(args.include_score_conditioning))
    eval_dataset_rows = build_sft_dataset(eval_rows, include_score_conditioning=bool(args.include_score_conditioning))

    if not train_dataset_rows:
        raise ValueError("No usable train examples in summary pairs")

    train_snapshot = output_dir / "train_dataset_snapshot.jsonl"
    eval_snapshot = output_dir / "eval_dataset_snapshot.jsonl"
    _write_jsonl(train_snapshot, train_dataset_rows)
    _write_jsonl(eval_snapshot, eval_dataset_rows)

    resolved_model = resolve_model_path(str(args.base_model), Path(args.settings_path))

    model_output_dir = output_dir / "model"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "created_at": _now_iso(),
        "status": "pending",
        "base_model_requested": str(args.base_model),
        "base_model_resolved": str(resolved_model),
        "config": {
            "epochs": float(args.epochs),
            "learning_rate": float(args.learning_rate),
            "per_device_batch_size": int(args.per_device_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "max_seq_length": int(args.max_seq_length),
            "use_lora": bool(args.use_lora),
            "load_in_4bit": bool(args.load_in_4bit),
            "bf16": bool(args.bf16),
            "seed": int(args.seed),
            "include_score_conditioning": bool(args.include_score_conditioning),
        },
        "counts": {
            "train_examples": len(train_dataset_rows),
            "eval_examples": len(eval_dataset_rows),
        },
        "artifacts": {
            "train_dataset_snapshot": str(train_snapshot),
            "eval_dataset_snapshot": str(eval_snapshot),
            "model_output_dir": str(model_output_dir),
            "adapter_or_model_path": None,
        },
    }

    if args.dry_run:
        manifest["status"] = "dry_run"
        (output_dir / "sft_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        LOGGER.info("Dry run complete. Output directory: %s", output_dir)
        return 0

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        manifest["status"] = "missing_dependencies"
        manifest["error"] = str(exc)
        (output_dir / "sft_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        LOGGER.error("Training dependencies unavailable: %s", exc)
        return 1

    try:
        train_dataset = Dataset.from_list([{"text": row["text"]} for row in train_dataset_rows])
        eval_dataset = Dataset.from_list([{"text": row["text"]} for row in eval_dataset_rows]) if eval_dataset_rows else None

        quantization_config = None
        if bool(args.load_in_4bit):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        model = AutoModelForCausalLM.from_pretrained(
            resolved_model,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(resolved_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        peft_config = None
        if bool(args.use_lora):
            peft_config = LoraConfig(
                r=int(args.lora_r),
                lora_alpha=int(args.lora_alpha),
                lora_dropout=float(args.lora_dropout),
                target_modules=[str(value) for value in args.lora_target_modules],
                task_type=TaskType.CAUSAL_LM,
            )

        training_args = _build_sft_config(SFTConfig, args=args, output_dir=model_output_dir)

        trainer_kwargs: Dict[str, Any] = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "peft_config": peft_config,
        }
        signature = inspect.signature(SFTTrainer.__init__).parameters
        if "processing_class" in signature:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in signature:
            trainer_kwargs["tokenizer"] = tokenizer
        if "dataset_text_field" in signature:
            trainer_kwargs["dataset_text_field"] = "text"

        trainer = SFTTrainer(**trainer_kwargs)
        trainer.train()

        final_dir = model_output_dir / "final"
        trainer.save_model(str(final_dir))

        manifest["status"] = "completed"
        manifest["artifacts"]["adapter_or_model_path"] = str(final_dir)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        (output_dir / "sft_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        LOGGER.exception("SFT training failed")
        return 1

    (output_dir / "sft_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("SFT training complete. Output directory: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
