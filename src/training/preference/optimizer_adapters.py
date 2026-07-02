"""TRL optimizer adapters over the shared treepo preference boundary."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union

from src.training.supervision.adapters import (
    BinaryProjection,
    build_dense_scalar_training_records as _build_dense_scalar_training_records,
    build_dpo_training_records as _build_dpo_training_records,
    build_group_grpo_training_records as _build_group_grpo_training_records,
    build_reward_model_training_records as _build_reward_model_training_records,
    build_scalar_reward_training_records as _build_scalar_reward_training_records,
    coerce_binary_projection as _coerce_binary_projection,
    coerce_comparative_dataset,
    prepare_binary_optimizer_dataset,
)
from src.training.supervision.comparative_types import PromptBuilder
from src.training.supervision.types import (
    BinaryComparison,
    BinaryProjectionDataset,
    ComparativeDataset,
    ComparativeJudgment,
    SupervisionDataset,
)

try:
    from treepo.methods.preference import (
        PreferenceDataset as TreepoPreferenceDataset,
        PreferenceRecord as TreepoPreferenceRecord,
        normalize_preference_data,
    )
except Exception:  # pragma: no cover - treepo is optional outside this repo env.
    TreepoPreferenceDataset = None  # type: ignore[assignment]
    TreepoPreferenceRecord = None  # type: ignore[assignment]
    normalize_preference_data = None  # type: ignore[assignment]


SupervisionInput = Union[
    SupervisionDataset,
    BinaryProjectionDataset,
    ComparativeDataset,
    Sequence[BinaryComparison],
    Sequence[ComparativeJudgment],
]


def coerce_preference_dataset(
    supervision: Any,
    *,
    projection: BinaryProjection = "adjacent",
) -> Any:
    """Return a treepo preference dataset when possible, else a binary projection."""
    treepo_dataset = coerce_treepo_preference_dataset(supervision)
    if treepo_dataset is not None:
        return treepo_dataset
    return _coerce_binary_projection(supervision, projection=projection)


def coerce_treepo_preference_dataset(supervision: Any) -> Any | None:
    """Coerce treepo unit/candidate preference inputs without stealing rich datasets."""
    if normalize_preference_data is None:
        return None
    if TreepoPreferenceDataset is not None and isinstance(supervision, TreepoPreferenceDataset):
        return supervision
    if TreepoPreferenceRecord is not None and isinstance(supervision, TreepoPreferenceRecord):
        return TreepoPreferenceDataset((supervision,))
    if hasattr(supervision, "keys") and {"units", "candidates"} <= set(supervision.keys()):
        return normalize_preference_data(supervision)
    if hasattr(supervision, "to_list") and callable(supervision.to_list):
        rows = supervision.to_list()
        if _sequence_contains_treepo_preferences(rows):
            return normalize_preference_data(rows)
        return None
    if isinstance(supervision, (str, Path)):
        if _path_contains_treepo_preferences(Path(supervision)):
            return normalize_preference_data(supervision)
        return None
    if isinstance(supervision, Mapping):
        if _mapping_contains_treepo_preferences(supervision):
            return normalize_preference_data(supervision)
        return None
    if isinstance(supervision, Sequence) and not isinstance(supervision, (str, bytes)):
        if _sequence_contains_treepo_preferences(supervision):
            return normalize_preference_data(supervision)
    return None


def build_dpo_training_records(
    supervision: Any,
    *,
    law_type: Optional[str] = None,
    prompt_builder: Optional[PromptBuilder] = None,
    projection: BinaryProjection = "adjacent",
    tree_objective_weighting_mode: str = "legacy_channel",
    discount_gamma: float = 1.0,
) -> list[dict[str, Any]]:
    treepo_dataset = coerce_treepo_preference_dataset(supervision)
    if treepo_dataset is not None:
        return _normalize_records(treepo_dataset.to_records("dpo"), law_type=law_type)
    return _build_dpo_training_records(
        supervision,
        law_type=law_type,
        prompt_builder=prompt_builder,
        projection=projection,
        tree_objective_weighting_mode=tree_objective_weighting_mode,
        discount_gamma=discount_gamma,
    )


def build_reward_model_training_records(
    supervision: Any,
    *,
    law_type: Optional[str] = None,
    prompt_builder: Optional[PromptBuilder] = None,
    projection: BinaryProjection = "adjacent",
    include_oracle_scores: bool = True,
    tree_objective_weighting_mode: str = "legacy_channel",
    discount_gamma: float = 1.0,
) -> list[dict[str, Any]]:
    treepo_dataset = coerce_treepo_preference_dataset(supervision)
    if treepo_dataset is not None:
        records = _normalize_records(treepo_dataset.to_records("reward"), law_type=law_type)
        if not include_oracle_scores:
            for record in records:
                record["chosen_score"] = None
                record["rejected_score"] = None
        return records
    return _build_reward_model_training_records(
        supervision,
        law_type=law_type,
        prompt_builder=prompt_builder,
        projection=projection,
        include_oracle_scores=include_oracle_scores,
        tree_objective_weighting_mode=tree_objective_weighting_mode,
        discount_gamma=discount_gamma,
    )


def build_group_grpo_training_records(
    supervision: Any,
    *,
    law_type: Optional[str] = None,
    prompt_builder: Optional[PromptBuilder] = None,
    min_group_size: int = 2,
    tree_objective_weighting_mode: str = "legacy_channel",
    discount_gamma: float = 1.0,
) -> list[dict[str, Any]]:
    treepo_dataset = coerce_treepo_preference_dataset(supervision)
    if treepo_dataset is not None:
        records = _normalize_records(treepo_dataset.to_records("grpo"), law_type=law_type)
        return [
            record
            for record in records
            if len(record.get("responses", []) or []) >= int(min_group_size)
        ]
    return _build_group_grpo_training_records(
        supervision,
        law_type=law_type,
        prompt_builder=prompt_builder,
        min_group_size=min_group_size,
        tree_objective_weighting_mode=tree_objective_weighting_mode,
        discount_gamma=discount_gamma,
    )


def build_scalar_reward_training_records(
    supervision: SupervisionInput,
    *,
    law_type: Optional[str] = None,
    prompt_builder: Optional[PromptBuilder] = None,
    tree_objective_weighting_mode: str = "legacy_channel",
    discount_gamma: float = 1.0,
) -> list[dict[str, Any]]:
    return _build_scalar_reward_training_records(
        supervision,
        law_type=law_type,
        prompt_builder=prompt_builder,
        tree_objective_weighting_mode=tree_objective_weighting_mode,
        discount_gamma=discount_gamma,
    )


def build_dense_scalar_training_records(
    supervision: SupervisionInput,
    *,
    law_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    return _build_dense_scalar_training_records(supervision, law_type=law_type)


def _path_contains_treepo_preferences(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(payload, Mapping) and _mapping_contains_treepo_preferences(payload)


def _mapping_contains_treepo_preferences(value: Mapping[str, Any]) -> bool:
    if {"units", "candidates"} <= set(value.keys()):
        return True
    for key in ("records", "preference_records", "pairs", "preference_pairs"):
        rows = value.get(key)
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            return _sequence_contains_treepo_preferences(rows)
    return _is_treepo_preference_mapping(value)


def _sequence_contains_treepo_preferences(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    return any(_is_treepo_preference_value(row) for row in value)


def _is_treepo_preference_value(value: Any) -> bool:
    if TreepoPreferenceRecord is not None and isinstance(value, TreepoPreferenceRecord):
        return True
    return isinstance(value, Mapping) and _is_treepo_preference_mapping(value)


def _is_treepo_preference_mapping(value: Mapping[str, Any]) -> bool:
    keys = set(value.keys())
    if "candidates" in keys and bool(keys & {"unit_id", "unit_type", "node_id", "doc_id"}):
        return True
    if "unit_id" in keys and bool(keys & {"candidate_id", "response_id", "value", "response"}):
        return True
    return _is_treepo_pair_mapping(value)


def _is_treepo_pair_mapping(value: Mapping[str, Any]) -> bool:
    keys = set(value.keys())
    has_left = bool(keys & {"response_a", "candidate_a"})
    has_right = bool(keys & {"response_b", "candidate_b"})
    has_compact_summary_names = (
        "summary_a" in keys
        and "summary_b" in keys
        and bool(keys & {"prompt", "target"})
        and "reference_score" not in keys
    )
    has_preference = bool(keys & {"preferred", "winner"})
    return ((has_left and has_right) or has_compact_summary_names) and has_preference


def _normalize_records(
    records: list[dict[str, Any]],
    *,
    law_type: Optional[str],
) -> list[dict[str, Any]]:
    out = []
    for row in records:
        record = dict(row)
        metadata = dict(record.get("metadata", {}) or {})
        nested = metadata.get("metadata")
        if isinstance(nested, Mapping):
            for key, value in nested.items():
                metadata.setdefault(str(key), value)
        if law_type is not None and metadata.get("law_type") != law_type:
            continue
        record["metadata"] = metadata
        for key in ("preference_supervision", "comparative_signal", "treepo"):
            if key not in record and isinstance(metadata.get(key), Mapping):
                record[key] = dict(metadata[key])
        if "law_type" not in record and metadata.get("law_type") is not None:
            record["law_type"] = metadata["law_type"]
        out.append(record)
    return out


__all__ = [
    "BinaryProjection",
    "build_dense_scalar_training_records",
    "build_dpo_training_records",
    "build_group_grpo_training_records",
    "build_reward_model_training_records",
    "build_scalar_reward_training_records",
    "coerce_comparative_dataset",
    "coerce_preference_dataset",
    "coerce_treepo_preference_dataset",
    "prepare_binary_optimizer_dataset",
]
