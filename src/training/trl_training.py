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
import random
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from src.training.preference.types import render_prompt, PromptBuilder
from src.stats.sampling import (
    largest_remainder_allocation as _largest_remainder_allocation,
    pps_inclusion_probabilities as _pps_inclusion_probabilities,
    systematic_pps_sample_indices as _systematic_pps_sample_indices,
)

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

    # Propensity/IPW weighting
    use_propensity_weighting: bool = True
    propensity_resample: bool = True
    propensity_native_loss_weighting: bool = True
    propensity_weight_clip: Optional[float] = None
    propensity_random_seed: int = 42
    propensity_sampling_strategy: Literal[
        "multinomial",
        "pps_systematic",
        "stratified_multinomial",
    ] = "pps_systematic"
    propensity_stratify_key: Optional[str] = "law_type"
    propensity_stratify_by: Optional[str] = None  # backwards-compatible alias

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

def _extract_sample_weight(
    record: Dict[str, Any],
    default_weight: float = 1.0,
) -> float:
    """Extract sample weight from exported preference record."""
    if "sample_weight" in record:
        try:
            value = float(record["sample_weight"])
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass

    metadata = record.get("metadata") or {}
    try:
        value = float(metadata.get("sample_weight", default_weight))
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass

    treepo = metadata.get("treepo") if isinstance(metadata, dict) else None
    if isinstance(treepo, dict):
        try:
            propensity = float(treepo.get("joint_propensity", 1.0))
            if propensity > 0:
                return 1.0 / propensity
        except (TypeError, ValueError):
            pass

    return default_weight


def _resample_records_by_weight(
    records: List[Dict[str, Any]],
    config: TRLTrainingConfig,
) -> List[Dict[str, Any]]:
    """
    Resample records by sample_weight when weighting is enabled.

    This provides weighting support for trainers that do not consume
    per-example weights natively.
    """
    if not config.use_propensity_weighting or not config.propensity_resample or not records:
        return records

    weights = [
        min(_extract_sample_weight(record), config.propensity_weight_clip)
        if config.propensity_weight_clip is not None
        else _extract_sample_weight(record)
        for record in records
    ]
    total_weight = sum(weights)
    if total_weight <= 0:
        return records

    strategy = config.propensity_sampling_strategy
    rng = random.Random(config.propensity_random_seed)
    size = len(records)

    if strategy == "multinomial":
        return rng.choices(records, weights=weights, k=size)

    if strategy == "pps_systematic":
        sum_w = sum(weights)
        sum_w_sq = sum(weight * weight for weight in weights)
        neff = int(round((sum_w * sum_w / sum_w_sq))) if sum_w_sq > 0 else 0
        base_size = max(1, min(len(records), neff))

        inclusion_probs = _pps_inclusion_probabilities(weights, base_size)
        sampled_indices = _systematic_pps_sample_indices(inclusion_probs, base_size, rng)
        sampled = [records[index] for index in sampled_indices]

        if len(sampled) < size:
            sampled.extend(rng.choices(records, weights=weights, k=size - len(sampled)))
        return sampled

    if strategy == "stratified_multinomial":
        stratify_key = config.propensity_stratify_key
        if stratify_key is None:
            stratify_key = config.propensity_stratify_by
        if not stratify_key:
            return rng.choices(records, weights=weights, k=size)

        groups: Dict[str, List[int]] = {}
        for index, record in enumerate(records):
            value = record.get(stratify_key)
            if value is None and isinstance(record.get("metadata"), dict):
                value = record["metadata"].get(stratify_key)
            key = str(value)
            groups.setdefault(key, []).append(index)

        if not groups:
            return rng.choices(records, weights=weights, k=size)

        keys = list(groups.keys())
        group_mass = [sum(weights[index] for index in groups[key]) for key in keys]
        total_mass = sum(group_mass)
        if total_mass <= 0:
            return rng.choices(records, k=size)

        quotas = [size * (mass / total_mass) for mass in group_mass]
        allocation = _largest_remainder_allocation(size, quotas)

        sampled: List[Dict[str, Any]] = []
        for key, alloc in zip(keys, allocation):
            if alloc <= 0:
                continue
            group_indices = groups[key]
            group_records = [records[index] for index in group_indices]
            group_weights = [weights[index] for index in group_indices]
            sampled.extend(rng.choices(group_records, weights=group_weights, k=alloc))
        return sampled

    logger.warning(
        "Unknown propensity sampling strategy '%s'; falling back to multinomial",
        strategy,
    )
    return rng.choices(records, weights=weights, k=size)


def _build_processing_class_kwargs(
    trainer_cls: Any,
    processing_class: Any,
) -> Dict[str, Any]:
    """
    Return trainer kwargs for tokenizer/processing_class across TRL versions.

    Newer TRL trainers use `processing_class=...`; older releases used
    `tokenizer=...`.
    """
    init_params = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in init_params:
        return {"processing_class": processing_class}
    if "tokenizer" in init_params:
        return {"tokenizer": processing_class}
    return {}

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
            "sample_weight": _extract_sample_weight(d),
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
            "sample_weight": float(pair.get("sample_weight", 1.0)),
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

def _coerce_sample_weight_tensor(
    raw_weights: Any,
    batch_size: int,
    device: Any,
):
    """Convert batch sample weights to a nonnegative tensor or return None."""
    if raw_weights is None:
        return None

    import torch

    if torch.is_tensor(raw_weights):
        weights = raw_weights.to(device=device, dtype=torch.float32)
    else:
        try:
            weights = torch.tensor(raw_weights, dtype=torch.float32, device=device)
        except Exception:
            return None

    weights = weights.reshape(-1)
    if weights.numel() == 1 and batch_size > 1:
        weights = weights.expand(batch_size)
    if weights.numel() != batch_size:
        return None
    weights = torch.clamp(weights, min=0.0)
    if float(weights.sum().item()) <= 0:
        return None
    return weights


def _build_weighted_dpo_trainer(base_cls):
    """Create a DPOTrainer subclass that applies per-example sample weights."""
    import torch

    class WeightedDPOTrainer(base_cls):
        def _weighted_reduce(self, values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
            values = self._per_example_mean(values)
            if weights is None:
                return values.mean()
            denom = weights.sum().clamp(min=1e-12)
            return (values * weights).sum() / denom

        def _per_example_mean(self, values: torch.Tensor) -> torch.Tensor:
            if values.ndim == 0:
                return values.reshape(1)
            if values.ndim <= 1:
                return values
            return values.reshape(values.shape[0], -1).mean(dim=1)

        def get_batch_loss_metrics(self, model, batch, train_eval: str = "train"):
            metrics = {}
            prefix = "eval_" if train_eval == "eval" else ""

            model_output = self.concatenated_forward(model, batch)
            if isinstance(model_output, dict):
                if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
                    reference_chosen_logps = batch["ref_chosen_logps"]
                    reference_rejected_logps = batch["ref_rejected_logps"]
                elif "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
                    reference_chosen_logps = batch["reference_chosen_logps"]
                    reference_rejected_logps = batch["reference_rejected_logps"]
                else:
                    reference_chosen_logps, reference_rejected_logps = self.compute_ref_log_probs(batch)

                losses = 0
                chosen_rewards = 0
                rejected_rewards = 0
                loss_types = self.loss_type if isinstance(self.loss_type, (list, tuple)) else [self.loss_type]
                loss_weights = getattr(self, "loss_weights", None)
                for index, loss_type in enumerate(loss_types):
                    _losses, _chosen_rewards, _rejected_rewards = self.dpo_loss(
                        model_output["chosen_logps"],
                        model_output["rejected_logps"],
                        reference_chosen_logps,
                        reference_rejected_logps,
                        loss_type,
                        model_output,
                    )
                    weight = loss_weights[index] if loss_weights else 1.0
                    losses = losses + _losses * weight
                    chosen_rewards = chosen_rewards + _chosen_rewards * weight
                    rejected_rewards = rejected_rewards + _rejected_rewards * weight

                if getattr(self.args, "rpo_alpha", None) is not None and "nll_loss" in model_output:
                    losses = losses + self.args.rpo_alpha * model_output["nll_loss"]

                if getattr(self, "use_weighting", False) and "policy_weights" in model_output:
                    losses = losses * model_output["policy_weights"]

                if getattr(self, "aux_loss_enabled", False) and "aux_loss" in model_output:
                    losses = losses + self.aux_loss_coef * model_output["aux_loss"]

                batch_size = model_output["chosen_logps"].shape[0]
                weights = _coerce_sample_weight_tensor(
                    batch.get("sample_weight"),
                    batch_size=batch_size,
                    device=model_output["chosen_logps"].device,
                )
                loss = self._weighted_reduce(losses, weights)

                reward_accuracies = (chosen_rewards > rejected_rewards).float()

                metrics[f"{prefix}rewards/chosen"] = float(
                    self._weighted_reduce(chosen_rewards.detach(), weights).cpu().item()
                )
                metrics[f"{prefix}rewards/rejected"] = float(
                    self._weighted_reduce(rejected_rewards.detach(), weights).cpu().item()
                )
                metrics[f"{prefix}rewards/accuracies"] = float(
                    self._weighted_reduce(reward_accuracies.detach(), weights).cpu().item()
                )
                metrics[f"{prefix}rewards/margins"] = float(
                    self._weighted_reduce((chosen_rewards - rejected_rewards).detach(), weights).cpu().item()
                )
                metrics[f"{prefix}logps/chosen"] = float(
                    self._weighted_reduce(model_output["chosen_logps"].detach(), weights).cpu().item()
                )
                metrics[f"{prefix}logps/rejected"] = float(
                    self._weighted_reduce(model_output["rejected_logps"].detach(), weights).cpu().item()
                )
                if "mean_chosen_logits" in model_output:
                    metrics[f"{prefix}logits/chosen"] = float(
                        self._weighted_reduce(model_output["mean_chosen_logits"].detach(), weights).cpu().item()
                    )
                if "mean_rejected_logits" in model_output:
                    metrics[f"{prefix}logits/rejected"] = float(
                        self._weighted_reduce(model_output["mean_rejected_logits"].detach(), weights).cpu().item()
                    )
                if getattr(self.args, "rpo_alpha", None) is not None and "nll_loss" in model_output:
                    metrics[f"{prefix}nll_loss"] = float(
                        self._weighted_reduce(model_output["nll_loss"].detach(), weights).cpu().item()
                    )
                if getattr(self, "aux_loss_enabled", False) and "aux_loss" in model_output:
                    metrics[f"{prefix}aux_loss"] = float(
                        self._weighted_reduce(model_output["aux_loss"].detach(), weights).cpu().item()
                    )

                return loss, metrics

            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
            ) = model_output

            if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
                reference_chosen_logps = batch["reference_chosen_logps"]
                reference_rejected_logps = batch["reference_rejected_logps"]
            else:
                with torch.no_grad():
                    if self.ref_model is None:
                        with self.null_ref_context():
                            (
                                reference_chosen_logps,
                                reference_rejected_logps,
                                _,
                                _,
                            ) = self.concatenated_forward(self.model, batch)
                    else:
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.ref_model, batch)

            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
            weights = _coerce_sample_weight_tensor(
                batch.get("sample_weight"),
                batch_size=losses.shape[0],
                device=losses.device,
            )
            loss = self._weighted_reduce(losses, weights)

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            metrics[f"{prefix}rewards/chosen"] = float(self._weighted_reduce(
                chosen_rewards.detach(),
                weights,
            ).cpu().item())
            metrics[f"{prefix}rewards/rejected"] = float(self._weighted_reduce(
                rejected_rewards.detach(),
                weights,
            ).cpu().item())
            metrics[f"{prefix}rewards/accuracies"] = float(self._weighted_reduce(
                reward_accuracies.detach(),
                weights,
            ).cpu().item())
            metrics[f"{prefix}rewards/margins"] = float(self._weighted_reduce(
                (chosen_rewards - rejected_rewards).detach(),
                weights,
            ).cpu().item())
            metrics[f"{prefix}logps/rejected"] = float(self._weighted_reduce(
                policy_rejected_logps.detach(),
                weights,
            ).cpu().item())
            metrics[f"{prefix}logps/chosen"] = float(self._weighted_reduce(
                policy_chosen_logps.detach(),
                weights,
            ).cpu().item())
            metrics[f"{prefix}logits/rejected"] = float(self._weighted_reduce(
                self._per_example_mean(policy_rejected_logits.detach()),
                weights,
            ).cpu().item())
            metrics[f"{prefix}logits/chosen"] = float(self._weighted_reduce(
                self._per_example_mean(policy_chosen_logits.detach()),
                weights,
            ).cpu().item())

            return loss, metrics

    return WeightedDPOTrainer


def _build_weighted_reward_data_collator(tokenizer: Any, max_length: Optional[int]):
    """Create a RewardTrainer data collator that preserves sample_weight."""
    import torch
    base_collator = None
    try:
        from trl.trainer.reward_trainer import DataCollatorForPreference

        base_collator = DataCollatorForPreference(
            pad_token_id=tokenizer.pad_token_id,
            return_tensors="pt",
        )
    except Exception:
        base_collator = None

    class WeightedRewardDataCollator:
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            sample_weights = torch.tensor(
                [float(feature.get("sample_weight", 1.0)) for feature in features],
                dtype=torch.float32,
            )

            # TRL >=0.26 reward format (chosen_input_ids/rejected_input_ids)
            if "chosen_input_ids" in features[0] and "rejected_input_ids" in features[0]:
                if base_collator is not None:
                    batch = base_collator(features)
                else:
                    chosen_input_ids = [torch.tensor(feature["chosen_input_ids"]) for feature in features]
                    rejected_input_ids = [torch.tensor(feature["rejected_input_ids"]) for feature in features]
                    input_ids = chosen_input_ids + rejected_input_ids
                    attention_mask = [torch.ones_like(ids) for ids in input_ids]
                    input_ids = tokenizer.pad(
                        {"input_ids": input_ids},
                        padding=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )["input_ids"]
                    attention_mask = tokenizer.pad(
                        {"input_ids": attention_mask},
                        padding=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )["input_ids"]
                    batch = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    }
                    if "margin" in features[0]:
                        batch["margin"] = torch.tensor(
                            [float(feature["margin"]) for feature in features],
                            dtype=torch.float32,
                        )
                batch["sample_weight"] = sample_weights
                return batch

            # Legacy format (already tokenized chosen/rejected pairs)
            features_chosen = []
            features_rejected = []
            margins: List[float] = []

            has_margin = "margin" in features[0]
            for feature in features:
                features_chosen.append(
                    {
                        "input_ids": feature["input_ids_chosen"],
                        "attention_mask": feature["attention_mask_chosen"],
                    }
                )
                features_rejected.append(
                    {
                        "input_ids": feature["input_ids_rejected"],
                        "attention_mask": feature["attention_mask_rejected"],
                    }
                )
                if has_margin:
                    margins.append(float(feature["margin"]))

            batch_chosen = tokenizer.pad(
                features_chosen,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch_rejected = tokenizer.pad(
                features_rejected,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )

            batch = {
                "input_ids_chosen": batch_chosen["input_ids"],
                "attention_mask_chosen": batch_chosen["attention_mask"],
                "input_ids_rejected": batch_rejected["input_ids"],
                "attention_mask_rejected": batch_rejected["attention_mask"],
                "return_loss": True,
                "sample_weight": sample_weights,
            }
            if has_margin:
                batch["margin"] = torch.tensor(margins, dtype=torch.float32)
            return batch

    return WeightedRewardDataCollator()


def _build_weighted_reward_trainer(base_cls):
    """Create a RewardTrainer subclass that applies per-example sample weights."""
    import torch.nn as nn

    class WeightedRewardTrainer(base_cls):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if "input_ids" in inputs and "attention_mask" in inputs:
                model_inputs = {key: value for key, value in inputs.items() if key != "sample_weight"}
                model_inputs["use_cache"] = False
                outputs = model(**model_inputs)
                rewards_chosen, rewards_rejected = outputs.logits.squeeze(-1).chunk(2)
                margin = inputs.get("margin")
                if margin is not None:
                    margin = margin.to(device=rewards_chosen.device, dtype=rewards_chosen.dtype)
                    per_example_loss = -nn.functional.logsigmoid(
                        rewards_chosen - rewards_rejected - margin
                    )
                else:
                    per_example_loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected)

                weights = _coerce_sample_weight_tensor(
                    inputs.get("sample_weight"),
                    batch_size=per_example_loss.shape[0],
                    device=per_example_loss.device,
                )
                if weights is None:
                    loss = per_example_loss.mean()
                else:
                    denom = weights.sum().clamp(min=1e-12)
                    loss = (per_example_loss * weights).sum() / denom

                if getattr(self.args, "center_rewards_coefficient", None) is not None:
                    loss = loss + self.args.center_rewards_coefficient * torch.mean(
                        (rewards_chosen + rewards_rejected) ** 2
                    )

                if return_outputs:
                    return loss, outputs
                return loss

            rewards_chosen = model(
                input_ids=inputs["input_ids_chosen"],
                attention_mask=inputs["attention_mask_chosen"],
                return_dict=True,
            )["logits"].squeeze(-1)
            rewards_rejected = model(
                input_ids=inputs["input_ids_rejected"],
                attention_mask=inputs["attention_mask_rejected"],
                return_dict=True,
            )["logits"].squeeze(-1)

            margin = inputs.get("margin")
            if margin is not None:
                margin = margin.to(device=rewards_chosen.device, dtype=rewards_chosen.dtype)
                per_example_loss = -nn.functional.logsigmoid(
                    rewards_chosen - rewards_rejected - margin
                )
            else:
                per_example_loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected)

            weights = _coerce_sample_weight_tensor(
                inputs.get("sample_weight"),
                batch_size=per_example_loss.shape[0],
                device=per_example_loss.device,
            )
            if weights is None:
                loss = per_example_loss.mean()
            else:
                denom = weights.sum().clamp(min=1e-12)
                loss = (per_example_loss * weights).sum() / denom

            if return_outputs:
                return loss, {
                    "rewards_chosen": rewards_chosen,
                    "rewards_rejected": rewards_rejected,
                }
            return loss

    return WeightedRewardTrainer


def _build_weighted_grpo_trainer(base_cls):
    """Create a GRPOTrainer subclass that applies per-example sample weights."""
    import torch

    class WeightedGRPOTrainer(base_cls):
        @staticmethod
        def _coerce_local_sample_weights(
            raw_inputs: List[Dict[str, Any]],
            device: Any,
            dtype: Any,
        ) -> Optional[torch.Tensor]:
            values: List[float] = []
            for example in raw_inputs:
                try:
                    values.append(max(0.0, float(example.get("sample_weight", 1.0))))
                except (TypeError, ValueError, AttributeError):
                    values.append(1.0)

            if not values:
                return None

            weights = torch.tensor(values, device=device, dtype=dtype)
            if float(weights.sum().item()) <= 0:
                return None

            # Keep average scale near one to stabilize optimizer hyperparameters.
            return weights / weights.mean().clamp(min=1e-12)

        def _generate_and_score_completions(self, inputs):
            batch = super()._generate_and_score_completions(inputs)
            advantages = batch.get("advantages")
            if advantages is None:
                return batch

            sample_weights = self._coerce_local_sample_weights(
                raw_inputs=inputs,
                device=advantages.device,
                dtype=advantages.dtype,
            )
            if sample_weights is None:
                return batch

            if sample_weights.shape[0] != advantages.shape[0]:
                logger.warning(
                    "Skipping GRPO native sample weighting due to shape mismatch "
                    "(weights=%s, advantages=%s)",
                    tuple(sample_weights.shape),
                    tuple(advantages.shape),
                )
                return batch

            if advantages.ndim == 1:
                batch["advantages"] = advantages * sample_weights
            else:
                batch["advantages"] = advantages * sample_weights.unsqueeze(-1)
            batch["sample_weight"] = sample_weights
            return batch

    return WeightedGRPOTrainer


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
    if (
        config.use_propensity_weighting
        and config.propensity_resample
        and not config.propensity_native_loss_weighting
    ):
        dpo_data = _resample_records_by_weight(dpo_data, config)

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
    trainer_cls = DPOTrainer
    if config.use_propensity_weighting and config.propensity_native_loss_weighting:
        trainer_cls = _build_weighted_dpo_trainer(DPOTrainer)

    trainer = trainer_cls(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        **_build_processing_class_kwargs(trainer_cls, tokenizer),
        peft_config=peft_config,
    )

    # Train
    logger.info("Starting DPO training...")
    trainer.train()

    # Save
    trainer.save_model(str(output_dir / "final"))
    logger.info(f"DPO training complete. Model saved to {output_dir / 'final'}")

    return str(output_dir / "final")


def _build_grpo_train_records(
    dataset: "PreferenceDataset",
    *,
    config: TRLTrainingConfig,
    law_type: Optional[str],
    prompt_builder: Optional[PromptBuilder],
) -> List[Dict[str, Any]]:
    """
    Build GRPO prompt records while preserving reward-context columns.

    Reward functions may rely on `reference_score`/`original_text`, so these
    fields must survive any de-duplication or resampling path.
    """
    prompt_records: List[Dict[str, Any]] = []
    for pair in dataset.pairs:
        if pair.preferred == "tie":
            continue
        if law_type is not None and pair.law_type != law_type:
            continue
        prompt_records.append(
            {
                "prompt": render_prompt(pair.original_text, pair.rubric, prompt_builder),
                "sample_weight": pair.ipw_weight(max_weight=config.propensity_weight_clip),
                "reference_score": pair.reference_score,
                "original_text": pair.original_text,
                "rubric": pair.rubric,
                "law_type": pair.law_type,
            }
        )

    if not prompt_records:
        return []

    if config.use_propensity_weighting:
        if config.propensity_resample and not config.propensity_native_loss_weighting:
            logger.info(
                "Using weighted prompt resampling fallback for GRPO (native weighting disabled)."
            )
            prompt_records = _resample_records_by_weight(prompt_records, config)
        elif config.propensity_native_loss_weighting:
            logger.info(
                "Using native GRPO sample-weighted advantages for propensity weighting."
            )
        return [
            {
                "prompt": str(record.get("prompt", "")),
                "sample_weight": float(record.get("sample_weight", 1.0)),
                "reference_score": record.get("reference_score"),
                "original_text": record.get("original_text"),
                "rubric": record.get("rubric"),
                "law_type": record.get("law_type"),
            }
            for record in prompt_records
            if str(record.get("prompt", "")).strip()
        ]

    deduped: List[Dict[str, Any]] = []
    seen: set[tuple[str, Any, str]] = set()
    for record in prompt_records:
        prompt = str(record.get("prompt", "")).strip()
        if not prompt:
            continue
        reference_score = record.get("reference_score")
        original_text = str(record.get("original_text", "") or "")
        key = (prompt, reference_score, original_text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            {
                "prompt": prompt,
                "sample_weight": 1.0,
                "reference_score": reference_score,
                "original_text": original_text,
                "rubric": record.get("rubric"),
                "law_type": record.get("law_type"),
            }
        )
    return deduped


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

    prompt_records = _build_grpo_train_records(
        dataset,
        config=config,
        law_type=law_type,
        prompt_builder=prompt_builder,
    )

    if not prompt_records:
        raise ValueError("No prompts available for GRPO training after filtering")
    train_dataset = Dataset.from_list(prompt_records)

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
    trainer_cls = GRPOTrainer
    if config.use_propensity_weighting and config.propensity_native_loss_weighting:
        trainer_cls = _build_weighted_grpo_trainer(GRPOTrainer)

    trainer = trainer_cls(
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

        entry = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "sample_weight": pair.ipw_weight(max_weight=config.propensity_weight_clip),
        }
        if margin is not None:
            entry["margin"] = margin
        reward_pairs.append(entry)

    if not reward_pairs:
        raise ValueError("No reward pairs available after filtering")

    if (
        config.use_propensity_weighting
        and config.propensity_resample
        and not config.propensity_native_loss_weighting
    ):
        reward_pairs = _resample_records_by_weight(reward_pairs, config)

    # Load model (as sequence classification model)
    model, tokenizer, peft_config = _load_model_for_training(
        model_name, config, is_reward_model=True
    )

    # RewardTrainer API compatibility: newer TRL expects raw chosen/rejected text
    # with `processing_class`, while older paths used pre-tokenized pair fields.
    trainer_cls = RewardTrainer
    if config.use_propensity_weighting and config.propensity_native_loss_weighting:
        trainer_cls = _build_weighted_reward_trainer(RewardTrainer)
    processing_kwargs = _build_processing_class_kwargs(trainer_cls, tokenizer)
    uses_processing_class = "processing_class" in processing_kwargs

    if uses_processing_class:
        from datasets import Dataset

        train_dataset = Dataset.from_list(reward_pairs)
    else:
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
    data_collator = None
    if config.use_propensity_weighting and config.propensity_native_loss_weighting:
        data_collator = _build_weighted_reward_data_collator(
            tokenizer=tokenizer,
            max_length=config.max_length,
        )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        **processing_kwargs,
        data_collator=data_collator,
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
