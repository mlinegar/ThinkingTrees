#!/usr/bin/env python3
"""Train manifesto summary model with GRPO using local-law reward."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

# Add project root for direct script execution.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.tasks.manifesto.openai_chat import OpenAIChatClient

from src.training.preference.oracle_reward import create_local_law_summary_reward_func


LOGGER = logging.getLogger(__name__)
DEFAULT_STUDENT_MODEL = "/mnt/data/models/AxionML/Qwen3.5-35B-A3B-NVFP4"
DEFAULT_SCORER_MODEL = "/mnt/data/models/nvidia/Qwen3.5-397B-A17B-NVFP4"
_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


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


def _parse_score(text: str) -> Optional[float]:
    rendered = str(text or "").strip()
    if not rendered:
        return None
    matches = _NUMERIC_RE.findall(rendered)
    if not matches:
        return None
    try:
        value = float(matches[-1])
    except (TypeError, ValueError):
        return None
    return max(-100.0, min(100.0, value))


def _parse_score_last_line(text: str) -> Optional[float]:
    rendered = str(text or "")
    if not rendered.strip():
        return None
    lines = [line.strip() for line in rendered.splitlines() if line.strip()]
    if not lines:
        return None
    last_line = lines[-1]
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", last_line):
        return None
    try:
        value = float(last_line)
    except (TypeError, ValueError):
        return None
    return max(-100.0, min(100.0, value))


def _parse_last_number(text: str) -> Optional[float]:
    rendered = str(text or "").strip()
    if not rendered:
        return None
    matches = _NUMERIC_RE.findall(rendered)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except (TypeError, ValueError):
        return None


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


def build_grpo_example(
    row: Dict[str, Any],
    *,
    include_score_conditioning: bool = False,
) -> Optional[Dict[str, Any]]:
    input_text = str(row.get("input_text", "") or "").strip()
    if not input_text:
        return None
    source_rile_raw = float(row.get("source_rile_raw", 0.0) or 0.0)
    hop = int(row.get("hop", 1) or 1)
    prompt = build_summary_prompt(
        input_text=input_text,
        source_rile_raw=source_rile_raw,
        hop=hop,
        include_score_conditioning=bool(include_score_conditioning),
    )
    return {
        "id": str(row.get("id", "")),
        "example_id": str(row.get("example_id", "")),
        "split": str(row.get("split", "")),
        "hop": hop,
        "prompt": prompt,
        "reference_score": source_rile_raw,
        "input_text": input_text,
        "target_summary": str(row.get("target_summary", "") or ""),
        "sample_weight": 1.0,
    }


def build_grpo_dataset(
    rows: Sequence[Dict[str, Any]],
    *,
    include_score_conditioning: bool = False,
) -> List[Dict[str, Any]]:
    built: List[Dict[str, Any]] = []
    for row in rows:
        example = build_grpo_example(row, include_score_conditioning=bool(include_score_conditioning))
        if example is not None:
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


def resolve_trainable_model_path(
    requested_model: str,
    *,
    settings_path: Path,
    explicit_trainable_model: Optional[str] = None,
) -> str:
    """Resolve a model path suitable for HF/TRL training."""
    if explicit_trainable_model:
        return resolve_model_path(str(explicit_trainable_model), settings_path)

    resolved = resolve_model_path(str(requested_model), settings_path)
    if "NVFP4" not in str(resolved).upper():
        return resolved

    candidate = None
    resolved_path = Path(resolved)
    name = resolved_path.name
    if name.upper().endswith("-NVFP4"):
        candidate = resolved_path.parent / name[:-6]
    if candidate and candidate.exists():
        LOGGER.warning(
            "Using trainable fallback model '%s' for NVFP4 requested model '%s'.",
            candidate,
            resolved,
        )
        return str(candidate)

    # Common local mirror path used in this workspace.
    fallback_map = {
        "Qwen3.5-35B-A3B-NVFP4": Path("/mnt/data/models/Qwen/Qwen3.5-35B-A3B"),
        "Qwen3.5-4B-NVFP4": Path("/mnt/data/models/Qwen/Qwen3.5-4B"),
        "Qwen3.5-2B-NVFP4": Path("/mnt/data/models/Qwen/Qwen3.5-2B"),
        "Qwen3.5-2B-Base-NVFP4": Path("/mnt/data/models/Qwen/Qwen3.5-2B-Base"),
        "Qwen3.5-0.8B-NVFP4": Path("/mnt/data/models/Qwen/Qwen3.5-0.8B"),
        "Qwen3.5-0.8B-Base-NVFP4": Path("/mnt/data/models/Qwen/Qwen3.5-0.8B-Base"),
    }
    for key, path in fallback_map.items():
        if key.upper() in str(resolved).upper() and path.exists():
            LOGGER.warning(
                "Using mapped trainable fallback model '%s' for NVFP4 requested model '%s'.",
                path,
                resolved,
            )
            return str(path)

    return resolved


def _build_grpo_config(grpo_config_cls: Any, *, args: argparse.Namespace, output_dir: Path) -> Any:
    params = set(inspect.signature(grpo_config_cls.__init__).parameters.keys())
    kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": float(args.learning_rate),
        "num_train_epochs": float(args.epochs),
        "per_device_train_batch_size": int(args.per_device_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "warmup_ratio": float(args.warmup_ratio),
        "num_generations": int(args.num_generations),
        "logging_steps": int(args.logging_steps),
        "save_steps": int(args.save_steps),
        "seed": int(args.seed),
    }
    if "bf16" in params:
        kwargs["bf16"] = bool(args.bf16)
    if "gradient_checkpointing" in params:
        kwargs["gradient_checkpointing"] = bool(args.gradient_checkpointing)
    if "max_prompt_length" in params:
        kwargs["max_prompt_length"] = int(args.max_prompt_length)
    if "max_completion_length" in params:
        kwargs["max_completion_length"] = int(args.max_completion_length)
    return grpo_config_cls(**kwargs)


def _build_score_fn(
    client: OpenAIChatClient,
    *,
    temperature: float,
    max_tokens: int,
):
    def _score(text: str) -> float:
        response = client.chat(
            system=(
                "You are a strict numeric coder for directional signal.\n"
                "Return exactly two lines:\n"
                "SCORE\n"
                "<numeric value in [-100,100]>\n"
                "No extra text."
            ),
            user=(
                "Score this text on a RILE-style directional scale.\n"
                "End with a newline followed by only the numeric score.\n\n"
                f"TEXT:\n{text}"
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = _parse_score_last_line(response)
        if parsed is None:
            parsed = _parse_score(response)
        retry = None
        if parsed is None:
            # Retry once with stricter formatting constraints to reduce reward fallbacks.
            retry = client.chat(
                system=(
                    "Return exactly one line containing only a numeric value in [-100,100]. "
                    "No words, no punctuation other than optional minus sign and decimal point."
                ),
                user=(
                    "Return exactly one numeric score for this text.\n"
                    "Output format example: -12.50\n\n"
                    f"TEXT:\n{text}"
                ),
                temperature=0.0,
                max_tokens=12,
            )
            parsed = _parse_score_last_line(retry)
            if parsed is None:
                parsed = _parse_score(retry)
        if parsed is None:
            # Third pass: salvage the last numeric token from either response.
            salvage = _parse_last_number(f"{response}\n{retry or ''}")
            if salvage is not None:
                parsed = max(-100.0, min(100.0, float(salvage)))
        if parsed is None:
            LOGGER.debug(
                "Scorer returned no parseable numeric token; marking parse failure with NaN. "
                "first=%r retry=%r",
                response,
                retry,
            )
            return float("nan")
        return float(parsed)

    return _score


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train manifesto summary GRPO model with local-law reward")

    parser.add_argument("--train-pairs", type=Path, default=Path("summary_pairs_train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--base-model", type=str, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument(
        "--trainable-base-model",
        type=str,
        default=None,
        help=(
            "Optional HF/TRL-trainable model path/profile. "
            "If omitted and --base-model is NVFP4, script attempts local non-NVFP4 fallback."
        ),
    )
    parser.add_argument("--settings-path", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--max-train-examples", type=int, default=0)

    parser.add_argument(
        "--include-score-conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include source_rile_raw in prompt conditioning. "
            "Disable to avoid label leakage unless inference also provides score estimates."
        ),
    )

    parser.add_argument("--scorer-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--scorer-model", type=str, default=DEFAULT_SCORER_MODEL)
    parser.add_argument("--scorer-api-key", type=str, default="EMPTY")
    parser.add_argument("--scorer-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--scorer-temperature", type=float, default=0.0)
    parser.add_argument("--scorer-max-tokens", type=int, default=64)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--c1-threshold-raw", type=float, default=10.0)
    parser.add_argument("--c2-threshold-raw", type=float, default=6.0)
    parser.add_argument("--neutral-raw", type=float, default=0.0)
    parser.add_argument("--c1-weight", type=float, default=0.6)
    parser.add_argument("--c2-weight", type=float, default=0.3)
    parser.add_argument("--same-side-weight", type=float, default=0.1)
    parser.add_argument("--neutral-reward", type=float, default=0.25)
    parser.add_argument(
        "--parse-failure-reward",
        type=float,
        default=0.0,
        help="Reward assigned when scorer returns no parseable number after all retries.",
    )
    parser.add_argument("--min-completion-chars", type=int, default=8)
    parser.add_argument("--short-penalty", type=float, default=0.1)
    parser.add_argument("--reward-cache-size", type=int, default=8192)
    parser.add_argument(
        "--reward-scorer-parallelism",
        type=int,
        default=16,
        help=(
            "Max concurrent scorer calls per GRPO reward evaluation step. "
            "Higher values improve vLLM continuous batching."
        ),
    )

    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=3072)
    parser.add_argument("--max-completion-length", type=int, default=512)

    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="*",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
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

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("outputs") / f"manifesto_summary_grpo_{_now_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(args.train_pairs)
    dataset_rows = build_grpo_dataset(
        rows,
        include_score_conditioning=bool(args.include_score_conditioning),
    )

    if int(args.max_train_examples) > 0:
        dataset_rows = dataset_rows[: int(args.max_train_examples)]

    if not dataset_rows:
        raise ValueError("No usable train examples in summary pairs")

    snapshot_path = output_dir / "train_dataset_snapshot.jsonl"
    _write_jsonl(snapshot_path, dataset_rows)

    resolved_model = resolve_model_path(str(args.base_model), Path(args.settings_path))
    resolved_trainable_model = resolve_trainable_model_path(
        str(args.base_model),
        settings_path=Path(args.settings_path),
        explicit_trainable_model=args.trainable_base_model,
    )
    resolved_scorer_model = resolve_model_path(str(args.scorer_model), Path(args.settings_path))
    effective_load_in_4bit = bool(args.load_in_4bit)
    if effective_load_in_4bit and "NVFP4" in str(resolved_trainable_model).upper():
        LOGGER.warning(
            "Disabling bitsandbytes 4-bit load for NVFP4 model '%s' to avoid incompatible quantized kernels.",
            resolved_trainable_model,
        )
        effective_load_in_4bit = False

    model_output_dir = output_dir / "model"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "created_at": _now_iso(),
        "status": "pending",
        "base_model_requested": str(args.base_model),
        "base_model_resolved": str(resolved_model),
        "base_model_trainable_requested": str(args.trainable_base_model or ""),
        "base_model_trainable_resolved": str(resolved_trainable_model),
        "scorer_model_requested": str(args.scorer_model),
        "scorer_model_resolved": str(resolved_scorer_model),
        "counts": {
            "train_examples": len(dataset_rows),
        },
        "reward": {
            "type": "local_law_summary",
            "c1_threshold_raw": float(args.c1_threshold_raw),
            "c2_threshold_raw": float(args.c2_threshold_raw),
            "neutral_raw": float(args.neutral_raw),
            "c1_weight": float(args.c1_weight),
            "c2_weight": float(args.c2_weight),
            "same_side_weight": float(args.same_side_weight),
            "neutral_reward": float(args.neutral_reward),
            "parse_failure_reward": float(args.parse_failure_reward),
            "min_completion_chars": int(args.min_completion_chars),
            "short_penalty": float(args.short_penalty),
            "cache_size": int(args.reward_cache_size),
            "scorer_parallelism": int(args.reward_scorer_parallelism),
        },
        "config": {
            "epochs": float(args.epochs),
            "learning_rate": float(args.learning_rate),
            "per_device_batch_size": int(args.per_device_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "warmup_ratio": float(args.warmup_ratio),
            "num_generations": int(args.num_generations),
            "max_prompt_length": int(args.max_prompt_length),
            "max_completion_length": int(args.max_completion_length),
            "use_lora": bool(args.use_lora),
            "load_in_4bit": bool(args.load_in_4bit),
            "effective_load_in_4bit": bool(effective_load_in_4bit),
            "bf16": bool(args.bf16),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "seed": int(args.seed),
            "include_score_conditioning": bool(args.include_score_conditioning),
        },
        "artifacts": {
            "train_dataset_snapshot": str(snapshot_path),
            "model_output_dir": str(model_output_dir),
            "adapter_or_model_path": None,
        },
    }

    if args.dry_run:
        manifest["status"] = "dry_run"
        (output_dir / "grpo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        LOGGER.info("Dry run complete. Output directory: %s", output_dir)
        return 0

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:
        manifest["status"] = "missing_dependencies"
        manifest["error"] = str(exc)
        (output_dir / "grpo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        LOGGER.error("Training dependencies unavailable: %s", exc)
        return 1

    try:
        train_dataset = Dataset.from_list(dataset_rows)

        quantization_config = None
        if bool(effective_load_in_4bit):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        model = AutoModelForCausalLM.from_pretrained(
            resolved_trainable_model,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(resolved_trainable_model, trust_remote_code=True)
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

        scorer_client = OpenAIChatClient(
            base_url=str(args.scorer_base_url),
            model=str(resolved_scorer_model),
            api_key=str(args.scorer_api_key),
            timeout_seconds=float(args.scorer_timeout_seconds),
            enable_thinking=bool(args.enable_thinking),
            max_connections=max(64, int(args.reward_scorer_parallelism) * 4),
        )
        score_fn = _build_score_fn(
            scorer_client,
            temperature=float(args.scorer_temperature),
            max_tokens=int(args.scorer_max_tokens),
        )
        reward_func = create_local_law_summary_reward_func(
            score_fn,
            c1_threshold_raw=float(args.c1_threshold_raw),
            c2_threshold_raw=float(args.c2_threshold_raw),
            neutral_raw=float(args.neutral_raw),
            c1_weight=float(args.c1_weight),
            c2_weight=float(args.c2_weight),
            same_side_weight=float(args.same_side_weight),
            neutral_reward=float(args.neutral_reward),
            parse_failure_reward=float(args.parse_failure_reward),
            min_completion_chars=int(args.min_completion_chars),
            short_completion_penalty=float(args.short_penalty),
            cache_size=int(args.reward_cache_size),
            scorer_parallelism=int(args.reward_scorer_parallelism),
        )

        training_args = _build_grpo_config(GRPOConfig, args=args, output_dir=model_output_dir)

        trainer_kwargs: Dict[str, Any] = {
            "model": model,
            "reward_funcs": [reward_func],
            "args": training_args,
            "train_dataset": train_dataset,
            "peft_config": peft_config,
        }
        signature = inspect.signature(GRPOTrainer.__init__).parameters
        if "processing_class" in signature:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in signature:
            trainer_kwargs["tokenizer"] = tokenizer

        trainer = GRPOTrainer(**trainer_kwargs)
        trainer.train()

        final_dir = model_output_dir / "final"
        trainer.save_model(str(final_dir))

        manifest["status"] = "completed"
        manifest["artifacts"]["adapter_or_model_path"] = str(final_dir)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        (output_dir / "grpo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        LOGGER.exception("GRPO training failed")
        return 1

    (output_dir / "grpo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("GRPO training complete. Output directory: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
