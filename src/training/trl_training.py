"""
TRL Integration for Preference-Based Training.

This module provides wrappers around TRL (Transformers Reinforcement Learning)
trainers for DPO, GRPO, and reward model training. It bridges our preference
collection system with TRL's training infrastructure.

Dependencies:
    pip install trl>=0.7.0 transformers>=4.40.0 peft>=0.8.0

Architecture:
    PreferenceDataset → Export Format → HuggingFace Dataset → TRL Trainer

Usage:
    from src.training.trl_training import (
        train_dpo,
        train_grpo,
        train_reward_model,
        TRLTrainingConfig,
    )

    # Load preference data
    dataset = PreferenceDataset.load("preferences.json")

    # Train DPO
    train_dpo(
        dataset=dataset,
        model_name="nvidia/Nemotron-Nano-8B",
        output_dir="models/dpo_trained",
        config=TRLTrainingConfig(
            learning_rate=1e-5,
            num_train_epochs=3,
            use_lora=True,
        ),
    )

    # Train reward model
    train_reward_model(
        dataset=dataset,
        model_name="nvidia/Nemotron-Nano-8B",
        output_dir="models/reward_model",
    )
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from src.training.preference.types import render_prompt, PromptBuilder

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TRLTrainingConfig:
    """Configuration for TRL-based training."""

    # Training hyperparameters
    learning_rate: float = 1e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    max_length: int = 2048
    max_prompt_length: int = 1024

    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Quantization (for large models)
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # DPO-specific
    beta: float = 0.1  # KL penalty coefficient

    # GRPO-specific
    num_generations: int = 4  # Number of generations per prompt

    # Reward model-specific
    reward_use_margin: bool = False
    reward_margin_source: Literal["score_estimate", "oracle_error"] = "score_estimate"
    reward_margin_scale: Optional[float] = None

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100

    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True


# =============================================================================
# Dataset Conversion
# =============================================================================

def _preference_to_hf_dpo(
    preference_data: List[Dict[str, Any]],
) -> "Dataset":
    """
    Convert DPO format data to HuggingFace Dataset.

    Args:
        preference_data: Output from PreferenceDataset.to_preference_format("dpo")

    Returns:
        HuggingFace Dataset with prompt, chosen, rejected columns
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")

    # Filter out ties (no chosen/rejected for ties)
    filtered = [
        {
            "prompt": d["prompt"],
            "chosen": d["chosen"],
            "rejected": d["rejected"],
        }
        for d in preference_data
        if d.get("chosen") and d.get("rejected")
    ]

    logger.info(f"Converted {len(filtered)} preference pairs to DPO format")
    return Dataset.from_list(filtered)


def _concat_prompt_response(prompt: str, response: str) -> str:
    """Join prompt and response into a single sequence."""
    if not prompt:
        return response
    if prompt.endswith(("\n", " ")):
        return f"{prompt}{response}"
    return f"{prompt}\n{response}"


def _preference_to_hf_reward(
    reward_pairs: List[Dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> "Dataset":
    """
    Convert chosen/rejected pairs into RewardTrainer tokenized format.

    Args:
        reward_pairs: List with prompt/chosen/rejected and optional margin
        tokenizer: HuggingFace tokenizer
        max_length: Max sequence length for tokenization

    Returns:
        HuggingFace Dataset with input_ids_* and attention_mask_* fields
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")

    converted = []
    for pair in reward_pairs:
        chosen_text = _concat_prompt_response(pair["prompt"], pair["chosen"])
        rejected_text = _concat_prompt_response(pair["prompt"], pair["rejected"])

        chosen_enc = tokenizer(chosen_text, truncation=True, max_length=max_length)
        rejected_enc = tokenizer(rejected_text, truncation=True, max_length=max_length)

        entry = {
            "input_ids_chosen": chosen_enc["input_ids"],
            "attention_mask_chosen": chosen_enc["attention_mask"],
            "input_ids_rejected": rejected_enc["input_ids"],
            "attention_mask_rejected": rejected_enc["attention_mask"],
        }
        if pair.get("margin") is not None:
            entry["margin"] = pair["margin"]
        converted.append(entry)

    logger.info(f"Converted {len(converted)} preference pairs to reward model format")
    return Dataset.from_list(converted)


def _preference_to_hf_grpo(
    grpo_data: List[Dict[str, Any]],
) -> "Dataset":
    """
    Convert GRPO format to HuggingFace Dataset.

    Args:
        grpo_data: Output from PreferenceDataset.to_grouped_grpo_format()

    Returns:
        HuggingFace Dataset with prompt, responses, ranks columns
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")

    converted = [
        {
            "prompt": d["prompt"],
            "responses": d["responses"],
            "ranks": d["ranks"],
        }
        for d in grpo_data
    ]

    logger.info(f"Converted {len(converted)} groups to GRPO format")
    return Dataset.from_list(converted)


def _compute_reward_margin(
    chosen_score: Optional[float],
    rejected_score: Optional[float],
    chosen_error: Optional[float],
    rejected_error: Optional[float],
    config: TRLTrainingConfig,
) -> Optional[float]:
    """Compute optional margin for reward modeling."""
    if not config.reward_use_margin:
        return None

    margin = None
    if config.reward_margin_source == "oracle_error":
        if chosen_error is not None and rejected_error is not None:
            margin = rejected_error - chosen_error
    else:
        if chosen_score is not None and rejected_score is not None:
            margin = chosen_score - rejected_score

    if margin is None:
        return None

    if config.reward_margin_scale:
        margin = margin / config.reward_margin_scale

    if margin <= 0:
        return None

    return margin


# =============================================================================
# Model Loading Utilities
# =============================================================================

def _load_model_for_training(
    model_name: str,
    config: TRLTrainingConfig,
    is_reward_model: bool = False,
):
    """
    Load model with optional quantization and LoRA.

    Args:
        model_name: HuggingFace model name or path
        config: Training configuration
        is_reward_model: Whether loading for reward model training

    Returns:
        Tuple of (model, tokenizer, peft_config or None)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
        import torch
    except ImportError:
        raise ImportError(
            "transformers library required. Install with: pip install transformers"
        )

    # Determine compute dtype
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)

    # Quantization config
    quantization_config = None
    if config.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            )
        except ImportError:
            logger.warning("bitsandbytes not available, skipping quantization")
            quantization_config = None

    # Load model
    model_cls = AutoModelForSequenceClassification if is_reward_model else AutoModelForCausalLM
    model_kwargs = {
        "quantization_config": quantization_config,
        "device_map": "auto",
        "torch_dtype": compute_dtype,
        "trust_remote_code": True,
    }
    if is_reward_model:
        model_kwargs["num_labels"] = 1

    model = model_cls.from_pretrained(model_name, **model_kwargs)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    peft_config = None
    if config.use_lora:
        try:
            from peft import LoraConfig, TaskType
            peft_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                task_type=TaskType.SEQ_CLS if is_reward_model else TaskType.CAUSAL_LM,
            )
        except ImportError:
            logger.warning("peft not available, training without LoRA")

    return model, tokenizer, peft_config


# =============================================================================
# Training Functions
# =============================================================================

def train_dpo(
    dataset: "PreferenceDataset",
    model_name: str,
    output_dir: Union[str, Path],
    config: Optional[TRLTrainingConfig] = None,
    ref_model_name: Optional[str] = None,
    law_type: Optional[str] = None,
    prompt_builder: Optional[PromptBuilder] = None,
) -> str:
    """
    Train model using Direct Preference Optimization (DPO).

    Args:
        dataset: PreferenceDataset with collected preferences
        model_name: HuggingFace model name to fine-tune
        output_dir: Directory to save trained model
        config: Training configuration (uses defaults if None)
        ref_model_name: Reference model (uses model_name if None)
        law_type: Optional filter for specific law type
        prompt_builder: Optional prompt builder for generating prompts

    Returns:
        Path to saved model
    """
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError:
        raise ImportError("TRL library required. Install with: pip install trl>=0.7.0")

    config = config or TRLTrainingConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting DPO training with model: {model_name}")

    # Convert preference data to HF format
    dpo_data = dataset.to_preference_format(
        "dpo",
        law_type=law_type,
        prompt_builder=prompt_builder,
    )

    train_dataset = _preference_to_hf_dpo(dpo_data)

    # Load models
    model, tokenizer, peft_config = _load_model_for_training(model_name, config)

    # Reference model (for KL penalty)
    ref_model = None
    if ref_model_name and ref_model_name != model_name:
        ref_model, _, _ = _load_model_for_training(ref_model_name, config)

    # DPO config
    training_args = DPOConfig(
        output_dir=str(output_dir),
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        beta=config.beta,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
    )

    # Create trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # Train
    logger.info("Starting DPO training...")
    trainer.train()

    # Save
    trainer.save_model(str(output_dir / "final"))
    logger.info(f"DPO training complete. Model saved to {output_dir / 'final'}")

    return str(output_dir / "final")


def train_grpo(
    dataset: "PreferenceDataset",
    model_name: str,
    output_dir: Union[str, Path],
    config: Optional[TRLTrainingConfig] = None,
    law_type: Optional[str] = None,
    reward_funcs: Optional[Union[Callable, List[Callable]]] = None,
    prompt_builder: Optional[PromptBuilder] = None,
) -> str:
    """
    Train model using Group Relative Policy Optimization (GRPO).

    GRPO in TRL is an online method that generates completions and scores
    them using reward functions. It does not consume offline ranked groups.

    Args:
        dataset: PreferenceDataset used to extract prompts for training
        model_name: HuggingFace model name to fine-tune
        output_dir: Directory to save trained model
        config: Training configuration
        law_type: Optional filter for specific law type
        reward_funcs: Reward function(s) compatible with TRL GRPOTrainer
        prompt_builder: Optional prompt builder for generating prompts

    Returns:
        Path to saved model
    """
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        raise ImportError(
            "TRL library with GRPO support required. "
            "Install with: pip install trl>=0.8.0"
        )

    config = config or TRLTrainingConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if reward_funcs is None:
        raise ValueError(
            "GRPO training requires reward_funcs. TRL GRPOTrainer is online and "
            "does not consume offline ranked preference groups."
        )

    logger.info(f"Starting GRPO training with model: {model_name}")

    # Build prompt-only dataset from preferences
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")

    prompts = []
    seen = set()
    for pair in dataset.pairs:
        if law_type is not None and pair.law_type != law_type:
            continue
        prompt = render_prompt(pair.original_text, pair.rubric, prompt_builder)
        if prompt in seen:
            continue
        seen.add(prompt)
        prompts.append({"prompt": prompt})

    if not prompts:
        raise ValueError("No prompts available for GRPO training after filtering")

    train_dataset = Dataset.from_list(prompts)

    # Load model
    model, tokenizer, peft_config = _load_model_for_training(model_name, config)

    # GRPO config
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        num_generations=config.num_generations,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train
    logger.info("Starting GRPO training...")
    trainer.train()

    # Save
    trainer.save_model(str(output_dir / "final"))
    logger.info(f"GRPO training complete. Model saved to {output_dir / 'final'}")

    return str(output_dir / "final")


def train_reward_model(
    dataset: "PreferenceDataset",
    model_name: str,
    output_dir: Union[str, Path],
    config: Optional[TRLTrainingConfig] = None,
    law_type: Optional[str] = None,
    prompt_builder: Optional[PromptBuilder] = None,
) -> str:
    """
    Train a reward model from preference data.

    The reward model learns to assign higher reward to preferred responses.

    Args:
        dataset: PreferenceDataset with collected preferences
        model_name: HuggingFace model name to fine-tune
        output_dir: Directory to save trained model
        config: Training configuration
        law_type: Optional filter for specific law type
        prompt_builder: Optional prompt builder for generating prompts

    Returns:
        Path to saved model
    """
    try:
        from trl import RewardConfig, RewardTrainer
    except ImportError:
        raise ImportError("TRL library required. Install with: pip install trl>=0.7.0")

    config = config or TRLTrainingConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting reward model training with model: {model_name}")

    # Build reward pairs (chosen/rejected) from preference data
    reward_pairs = []
    for pair in dataset.pairs:
        if pair.preferred == "tie":
            continue
        if law_type is not None and pair.law_type != law_type:
            continue

        prompt = render_prompt(pair.original_text, pair.rubric, prompt_builder)

        if pair.preferred == "A":
            chosen = pair.summary_a
            rejected = pair.summary_b
            chosen_score = pair.score_estimate_a
            rejected_score = pair.score_estimate_b
            chosen_error = pair.oracle_error_a
            rejected_error = pair.oracle_error_b
        else:
            chosen = pair.summary_b
            rejected = pair.summary_a
            chosen_score = pair.score_estimate_b
            rejected_score = pair.score_estimate_a
            chosen_error = pair.oracle_error_b
            rejected_error = pair.oracle_error_a

        margin = _compute_reward_margin(
            chosen_score,
            rejected_score,
            chosen_error,
            rejected_error,
            config,
        )

        reward_pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "margin": margin,
        })

    if not reward_pairs:
        raise ValueError("No reward pairs available after filtering")

    # Load model (as sequence classification model)
    model, tokenizer, peft_config = _load_model_for_training(
        model_name, config, is_reward_model=True
    )

    # Tokenize reward pairs for RewardTrainer
    train_dataset = _preference_to_hf_reward(
        reward_pairs,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )

    # Reward training config
    training_args = RewardConfig(
        output_dir=str(output_dir),
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        max_length=config.max_length,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
    )

    # Create trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # Train
    logger.info("Starting reward model training...")
    trainer.train()

    # Save
    trainer.save_model(str(output_dir / "final"))
    logger.info(f"Reward model training complete. Model saved to {output_dir / 'final'}")

    return str(output_dir / "final")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI entry point for TRL training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train models using TRL (DPO, GRPO, Reward Model)"
    )
    parser.add_argument(
        "--method",
        choices=["dpo", "grpo", "reward"],
        required=True,
        help="Training method (grpo requires reward_funcs; see train_grpo)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to PreferenceDataset JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--law-type",
        type=str,
        default=None,
        help="Filter preferences by law type",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning)",
    )

    args = parser.parse_args()

    # Import here to avoid circular imports
    from src.training.preference.types import PreferenceDataset

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    dataset = PreferenceDataset.load(args.dataset)

    # Create config
    config = TRLTrainingConfig(
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        use_lora=not args.no_lora,
    )

    # Train
    if args.method == "dpo":
        train_dpo(dataset, args.model, args.output_dir, config, law_type=args.law_type)
    elif args.method == "grpo":
        train_grpo(dataset, args.model, args.output_dir, config, law_type=args.law_type)
    elif args.method == "reward":
        train_reward_model(dataset, args.model, args.output_dir, config, law_type=args.law_type)


if __name__ == "__main__":
    main()
