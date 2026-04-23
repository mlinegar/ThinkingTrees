#!/usr/bin:env python3
"""
Phase 3: train a LoRA adapter on the scorer using the pooled joint training set.

Swap DSPy's prompt-optimization for real weight updates. Data is the same
Benoit-disjoint pool used by `scripts/phase2_joint_optimize.py`:
~1,263 (summary, dimension, expert_mean) triples. Labels come from Benoit's
open-weight LLM ensemble on non-test manifestos. Zero overlap with the
235-manifesto Benoit expert-benchmark test set (guaranteed via
`load_joint_train_pairs(..., global_holdout_keys=...)`).

Training target: chat-formatted `(system=task_context, user=summary,
assistant=<score>N</score>)` triples, with loss masked to the assistant
turn only. Output format matches `DimensionScoreSignature` XML tags so the
existing inference/evaluation path reads the adapter output unchanged.

Usage:
    python scripts/phase3_scorer_lora.py \\
        --base-model /mnt/data/models/Qwen/Qwen3.5-4B \\
        --output-dir outputs/phase3_scorer_lora/qwen3p5_4b \\
        --epochs 3 --lora-rank 16 --batch-size 1 --grad-accum 8

After training, serve via vllm with LoRA:
    vllm serve <base> --enable-lora \\
        --lora-modules scorer_lora=outputs/phase3_scorer_lora/<dir>/adapter \\
        --port 8011
Then evaluate with the existing phase2/phase0 scripts pointed at port 8011
and `--model scorer_lora` if DSPy exposes a lora-name override.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.tasks.manifesto.dimensions import PolicyDimension
from src.tasks.manifesto.expert_benchmarks import load_joint_train_pairs
from src.tasks.manifesto.scoring_contexts import get_scoring_context

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", type=str,
                   default="/mnt/data/models/Qwen/Qwen3.5-4B",
                   help="bf16 base model dir (NVFP4 models are inference-only; do NOT use here)")
    p.add_argument("--train-pool", choices=["openweight", "expert"], default="openweight")
    p.add_argument("--output-dir", type=Path,
                   default=project_root / "outputs" / "phase3_scorer_lora" /
                   f"run_{datetime.now(timezone.utc):%Y%m%d_%H%M}")
    # LoRA hyperparameters
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", type=str, nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj"],
                   help="LoRA target modules (typical attn projections).")
    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1, help="per-device batch")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-strategy", choices=["epoch", "no"], default="epoch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-chat-template", action="store_true", default=True,
                   help="Format examples via tokenizer.apply_chat_template (default).")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _format_score_target(label: float) -> str:
    """XML-matching format so DSPy's XMLAdapter parses the adapter's output."""
    n = max(1, min(7, int(round(float(label)))))
    return f"<score>{n}</score>"


def _dataset_from_joint_pool(tokenizer, train_pool: str, max_seq_len: int):
    import pandas as pd
    from datasets import Dataset

    # Build per-dim test key exclusion set (union) — we pass only to keep
    # the loader API symmetric with phase2_joint_optimize.
    from scripts.phase2_joint_optimize import _load_test_examples  # type: ignore
    test_keys_per_dim = {
        dim: set(_load_test_examples(dim)["manifesto_stem"])
        for dim in PolicyDimension
    }
    global_holdout_keys = set().union(*test_keys_per_dim.values())
    train_df = load_joint_train_pairs(
        train_pool,
        test_keys_per_dim=test_keys_per_dim,
        global_holdout_keys=global_holdout_keys,
    )
    logger.info("Loaded pooled train pool: %d rows across %d dims",
                len(train_df), train_df["dimension"].nunique())

    def _format(row) -> dict:
        dim = PolicyDimension(row["dimension"])
        ctx = get_scoring_context(dim)
        target = _format_score_target(row["label"])
        messages = [
            {"role": "system", "content": ctx},
            {"role": "user", "content": row["summary"]},
            {"role": "assistant", "content": target},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    records = train_df.to_dict("records")
    return Dataset.from_list([_format(r) for r in records])


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Heavy imports after logging so early errors surface cleanly.
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    logger.info("Loading tokenizer from %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Building dataset from joint pool (%s)", args.train_pool)
    train_ds = _dataset_from_joint_pool(tokenizer, args.train_pool, args.max_seq_len)
    logger.info("Train dataset rows: %d", len(train_ds))

    logger.info("Loading base model in bf16 with gradient checkpointing")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info("LoRA trainable params: %s / %s (%.2f%%)",
                f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=2,
        report_to=[],
        max_seq_length=args.max_seq_len,
        packing=False,
        seed=args.seed,
        assistant_only_loss=True,  # mask loss for system/user; train on assistant only
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()

    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("Saved adapter to %s", adapter_dir)

    manifest = {
        "base_model": args.base_model,
        "train_pool": args.train_pool,
        "trainable_params": trainable,
        "total_params": total,
        "epochs": args.epochs,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_seq_len": args.max_seq_len,
        "learning_rate": args.learning_rate,
        "lora": {
            "r": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.target_modules,
        },
        "n_train_examples": len(train_ds),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    next_step = (
        f"\nNext step — serve adapter via vllm on a new port and evaluate:\n"
        f"  vllm serve {args.base_model} \\\n"
        f"    --enable-lora \\\n"
        f"    --lora-modules scorer_lora={adapter_dir} \\\n"
        f"    --port 8011 --served-model-name scorer_lora\n\n"
        f"Then rerun evaluation against the new port:\n"
        f"  python scripts/phase2_joint_optimize.py --port 8011 --optimizer none \\\n"
        f"      --output-dir outputs/phase3_scorer_lora/eval\n"
    )
    logger.info(next_step)
    print(next_step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
