"""Fine-tuning adapter bridge for ThinkingTrees trainer wrappers.

The public treepo package owns framework-neutral fine-tuning rows. This module
keeps ThinkingTrees-specific trainer dispatch thin and lazy: export/dry-run
paths do not import TRL, DSPy, PEFT, Accelerate, or model stacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from treepo.finetune import build_finetune_views, export_for_adapter


_CORE_ADAPTER_BY_NAME = {
    "thinkingtrees_trl_sft": "trl_sft",
    "thinkingtrees_trl_dpo": "trl_dpo",
    "thinkingtrees_trl_reward": "trl_reward",
    "thinkingtrees_trl_scalar_reward": "trl_scalar_reward",
    "thinkingtrees_trl_grpo": "trl_grpo",
    "thinkingtrees_dspy": "dspy_examples",
}

_TRL_TRAINER_BY_NAME = {
    "thinkingtrees_trl_sft": ("src.training.trl_training", "train_sft"),
    "thinkingtrees_trl_dpo": ("src.training.trl_training", "train_dpo"),
    "thinkingtrees_trl_reward": ("src.training.trl_training", "train_reward_model"),
    "thinkingtrees_trl_scalar_reward": (
        "src.training.trl_training",
        "train_scalar_reward_records",
    ),
    "thinkingtrees_trl_grpo": ("src.training.trl_training", "train_grpo"),
}


@dataclass(frozen=True)
class ThinkingTreesFineTuneAdapter:
    """Concrete ThinkingTrees adapter metadata."""

    name: str
    framework: str
    core_adapter: str
    trainer: str | None
    description: str


def list_thinkingtrees_finetune_adapters() -> list[ThinkingTreesFineTuneAdapter]:
    """List concrete trainer adapters exposed by ThinkingTrees."""

    adapters: list[ThinkingTreesFineTuneAdapter] = []
    for name in sorted(_CORE_ADAPTER_BY_NAME):
        module_attr = _TRL_TRAINER_BY_NAME.get(name)
        trainer = None if module_attr is None else f"{module_attr[0]}:{module_attr[1]}"
        framework = "dspy" if name == "thinkingtrees_dspy" else "trl"
        adapters.append(
            ThinkingTreesFineTuneAdapter(
                name=name,
                framework=framework,
                core_adapter=_CORE_ADAPTER_BY_NAME[name],
                trainer=trainer or "src.ctreepo.dspy_family:DSPyFamily",
                description=_description_for(name),
            )
        )
    return adapters


def prepare_finetune_adapter(
    adapter_name: str,
    preference_data: Any,
    output_dir: str | Path,
    *,
    save_hf: bool = True,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare framework-ready rows for a ThinkingTrees adapter."""

    core_adapter = _core_adapter_name(adapter_name)
    result = export_for_adapter(
        core_adapter,
        preference_data,
        output_dir,
        save_hf=save_hf,
        config=config,
    )
    return {
        **result,
        "adapter": adapter_name,
        "core_adapter": core_adapter,
        "thinkingtrees_adapter": adapter_name,
    }


def train_finetune_adapter(
    adapter_name: str,
    preference_data: Any,
    output_dir: str | Path,
    *,
    model_name: str | None = None,
    dry_run: bool = True,
    prepare_output_dir: str | Path | None = None,
    save_hf: bool = False,
    training_config: Any | None = None,
    eval_data: Any | None = None,
    ref_model_name: str | None = None,
    law_type: str | None = None,
    prompt_builder: Any | None = None,
    reward_funcs: Callable[..., Any] | Sequence[Callable[..., Any]] | None = None,
    **train_kwargs: Any,
) -> dict[str, Any]:
    """Prepare rows and optionally dispatch to an existing trainer wrapper.

    ``dry_run=True`` is the default so callers can validate row projection and
    trainer routing without importing optional trainer stacks or touching GPUs.
    """

    adapter_name = _normalize_adapter_name(adapter_name)
    prepared = prepare_finetune_adapter(
        adapter_name,
        preference_data,
        prepare_output_dir or Path(output_dir) / "prepared",
        save_hf=save_hf,
    )
    if dry_run:
        return {
            "adapter": adapter_name,
            "core_adapter": prepared["core_adapter"],
            "dry_run": True,
            "prepared": prepared,
            "trainer": _trainer_name(adapter_name),
            "requires_model_name": adapter_name.startswith("thinkingtrees_trl_"),
            "missing": _dry_run_missing(adapter_name, model_name=model_name, reward_funcs=reward_funcs),
        }

    if adapter_name == "thinkingtrees_dspy":
        artifact = _train_dspy_adapter(
            preference_data,
            output_dir=Path(output_dir),
            prepared=prepared,
            **train_kwargs,
        )
    else:
        if not model_name:
            raise ValueError(f"{adapter_name} training requires model_name")
        artifact = _train_trl_adapter(
            adapter_name,
            preference_data,
            output_dir=Path(output_dir),
            model_name=model_name,
            training_config=training_config,
            eval_data=eval_data,
            ref_model_name=ref_model_name,
            law_type=law_type,
            prompt_builder=prompt_builder,
            reward_funcs=reward_funcs,
        )

    return {
        "adapter": adapter_name,
        "core_adapter": prepared["core_adapter"],
        "dry_run": False,
        "prepared": prepared,
        "trainer": _trainer_name(adapter_name),
        "artifact": artifact,
    }


def _train_trl_adapter(
    adapter_name: str,
    preference_data: Any,
    *,
    output_dir: Path,
    model_name: str,
    training_config: Any | None,
    eval_data: Any | None,
    ref_model_name: str | None,
    law_type: str | None,
    prompt_builder: Any | None,
    reward_funcs: Callable[..., Any] | Sequence[Callable[..., Any]] | None,
) -> str:
    trainer = _load_trainer(adapter_name)
    if adapter_name == "thinkingtrees_trl_sft":
        return trainer(
            records=build_finetune_views(preference_data, views=("sft",))["sft"],
            model_name=model_name,
            output_dir=output_dir,
            config=training_config,
            eval_records=_view_rows(eval_data, "sft") if eval_data is not None else None,
        )
    if adapter_name == "thinkingtrees_trl_scalar_reward":
        records = _scalar_reward_rows(preference_data)
        eval_records = _scalar_reward_rows(eval_data) if eval_data is not None else None
        return trainer(
            records=records,
            model_name=model_name,
            output_dir=output_dir,
            config=training_config,
            eval_records=eval_records,
        )
    if adapter_name == "thinkingtrees_trl_dpo":
        return trainer(
            dataset=preference_data,
            model_name=model_name,
            output_dir=output_dir,
            config=training_config,
            ref_model_name=ref_model_name,
            law_type=law_type,
            prompt_builder=prompt_builder,
        )
    if adapter_name == "thinkingtrees_trl_reward":
        return trainer(
            dataset=preference_data,
            model_name=model_name,
            output_dir=output_dir,
            config=training_config,
            law_type=law_type,
            prompt_builder=prompt_builder,
        )
    if adapter_name == "thinkingtrees_trl_grpo":
        if reward_funcs is None:
            raise ValueError("thinkingtrees_trl_grpo training requires reward_funcs")
        return trainer(
            dataset=preference_data,
            model_name=model_name,
            output_dir=output_dir,
            config=training_config,
            law_type=law_type,
            reward_funcs=reward_funcs,
            prompt_builder=prompt_builder,
        )
    raise ValueError(f"unsupported TRL fine-tune adapter: {adapter_name}")


def _train_dspy_adapter(
    preference_data: Any,
    *,
    output_dir: Path,
    prepared: Mapping[str, Any],
    family_runtime: Any | None = None,
    kind: str = "g",
    traces: Sequence[Any] | None = None,
    iteration: int = 1,
    f_init: Any = None,
    g_init: Any = None,
    f: Any = None,
    g: Any = None,
    trainer_callable: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Any:
    if trainer_callable is not None:
        return trainer_callable(
            preference_data=preference_data,
            output_dir=output_dir,
            prepared=prepared,
            **kwargs,
        )
    if family_runtime is None:
        raise ValueError(
            "thinkingtrees_dspy training requires trainer_callable or family_runtime. "
            "PreferenceDataset exports are available from the dry-run/prepare path; "
            "direct DSPy optimization needs the existing labeled-tree traces."
        )
    if traces is None:
        raise ValueError("thinkingtrees_dspy family_runtime training requires traces")
    if kind == "f":
        return family_runtime.train_f(
            f_init=f_init,
            g=g,
            traces=traces,
            output_dir=output_dir,
            iteration=iteration,
        )
    if kind == "g":
        return family_runtime.train_g(
            g_init=g_init,
            f=f,
            traces=traces,
            output_dir=output_dir,
            iteration=iteration,
        )
    raise ValueError("thinkingtrees_dspy kind must be 'f' or 'g'")


def _core_adapter_name(adapter_name: str) -> str:
    return _CORE_ADAPTER_BY_NAME[_normalize_adapter_name(adapter_name)]


def _normalize_adapter_name(adapter_name: str) -> str:
    key = str(adapter_name).strip()
    if key not in _CORE_ADAPTER_BY_NAME:
        available = ", ".join(sorted(_CORE_ADAPTER_BY_NAME))
        raise KeyError(f"unknown ThinkingTrees fine-tune adapter {key!r}; available: {available}")
    return key


def _trainer_name(adapter_name: str) -> str:
    module_attr = _TRL_TRAINER_BY_NAME.get(adapter_name)
    if module_attr is None:
        return "src.ctreepo.dspy_family:DSPyFamily"
    return f"{module_attr[0]}:{module_attr[1]}"


def _load_trainer(adapter_name: str) -> Callable[..., str]:
    module_name, attr_name = _TRL_TRAINER_BY_NAME[adapter_name]
    return getattr(import_module(module_name), attr_name)


def _view_rows(preference_data: Any, view: str) -> list[dict[str, Any]]:
    return build_finetune_views(preference_data, views=(view,))[view]


def _scalar_reward_rows(preference_data: Any) -> list[dict[str, Any]]:
    rows = []
    for row in _view_rows(preference_data, "sft"):
        score = row.get("metadata", {}).get("score", row.get("score"))
        if score is None:
            continue
        rows.append(
            {
                "prompt": str(row.get("prompt") or ""),
                "response": str(row.get("completion") or ""),
                "score": float(score),
                "sample_weight": float(row.get("sample_weight", 1.0) or 1.0),
                "metadata": dict(row.get("metadata") or {}),
            }
        )
    return rows


def _dry_run_missing(
    adapter_name: str,
    *,
    model_name: str | None,
    reward_funcs: Callable[..., Any] | Sequence[Callable[..., Any]] | None,
) -> list[str]:
    missing = []
    if adapter_name.startswith("thinkingtrees_trl_") and not model_name:
        missing.append("model_name")
    if adapter_name == "thinkingtrees_trl_grpo" and reward_funcs is None:
        missing.append("reward_funcs")
    return missing


def _description_for(name: str) -> str:
    return {
        "thinkingtrees_trl_sft": "Dispatch SFT rows to src.training.trl_training.train_sft.",
        "thinkingtrees_trl_dpo": "Dispatch PreferenceDataset rows to TRL DPO training.",
        "thinkingtrees_trl_reward": "Dispatch PreferenceDataset rows to pairwise reward training.",
        "thinkingtrees_trl_scalar_reward": "Dispatch supervised rows to scalar reward regression.",
        "thinkingtrees_trl_grpo": "Dispatch prompts to TRL GRPO with caller-supplied reward funcs.",
        "thinkingtrees_dspy": "Prepare DSPy examples or call an existing DSPy family runtime.",
    }[name]


__all__ = [
    "ThinkingTreesFineTuneAdapter",
    "list_thinkingtrees_finetune_adapters",
    "prepare_finetune_adapter",
    "train_finetune_adapter",
]
