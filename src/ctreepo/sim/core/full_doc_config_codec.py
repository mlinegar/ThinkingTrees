from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping


FULL_DOC_CONFIG_ALIAS_PAIRS: tuple[tuple[str, str], ...] = (
    ("tree_local_law_weight", "local_law_weight"),
    ("tree_task_objective_weight", "task_objective_weight"),
    ("tree_c1_relative_weight", "c1_relative_weight"),
    ("tree_c2_relative_weight", "c2_relative_weight"),
    ("tree_c3_relative_weight", "c3_relative_weight"),
)


def mapping_from_config_like(config_like: Any) -> Dict[str, Any]:
    if config_like is None:
        return {}
    if isinstance(config_like, Mapping):
        return dict(config_like)
    if is_dataclass(config_like):
        return asdict(config_like)
    if hasattr(config_like, "__dict__"):
        return dict(vars(config_like))
    return dict(config_like)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_value(item)
            for key, item in dict(value).items()
        }
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, set):
        return [_json_safe_value(item) for item in sorted(value, key=repr)]
    return value


def canonicalize_full_doc_config_mapping(
    config_like: Any,
    *,
    include_tree_aliases: bool = True,
    include_runtime_aliases: bool = True,
) -> Dict[str, Any]:
    mapping = mapping_from_config_like(config_like)
    normalized = dict(mapping)
    for tree_key, runtime_key in FULL_DOC_CONFIG_ALIAS_PAIRS:
        tree_value = normalized.get(tree_key)
        runtime_value = normalized.get(runtime_key)
        if (
            tree_value not in {"", None}
            and runtime_value not in {"", None}
            and tree_value != runtime_value
        ):
            raise ValueError(
                f"Config alias conflict: {tree_key}={tree_value!r} vs "
                f"{runtime_key}={runtime_value!r}. "
                f"Set only one, or ensure they match."
            )
        if runtime_value in {"", None} and tree_value not in {"", None}:
            normalized[runtime_key] = tree_value
        if tree_value in {"", None} and runtime_value not in {"", None}:
            normalized[tree_key] = runtime_value
    if not include_tree_aliases:
        for tree_key, _runtime_key in FULL_DOC_CONFIG_ALIAS_PAIRS:
            normalized.pop(tree_key, None)
    if not include_runtime_aliases:
        for _tree_key, runtime_key in FULL_DOC_CONFIG_ALIAS_PAIRS:
            normalized.pop(runtime_key, None)
    return normalized


def runtime_config_overrides_from_config_like(config_like: Any) -> Dict[str, Any]:
    return canonicalize_full_doc_config_mapping(
        config_like,
        include_tree_aliases=False,
        include_runtime_aliases=True,
    )


def tree_run_config_mapping_from_config_like(config_like: Any) -> Dict[str, Any]:
    return canonicalize_full_doc_config_mapping(
        config_like,
        include_tree_aliases=True,
        include_runtime_aliases=False,
    )


def serialize_full_doc_runtime_config(
    config_like: Any,
    *,
    metadata: Any | None = None,
) -> Dict[str, Any]:
    payload = runtime_config_overrides_from_config_like(config_like)
    if metadata is not None:
        payload.update(mapping_from_config_like(metadata))
    return _json_safe_value(payload)


def serialize_tree_run_config(
    config_like: Any,
    *,
    metadata: Any | None = None,
) -> Dict[str, Any]:
    payload = tree_run_config_mapping_from_config_like(config_like)
    if metadata is not None:
        payload.update(mapping_from_config_like(metadata))
    return _json_safe_value(payload)


def write_tree_run_config_json(path: Path, config_like: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_tree_run_config(config_like)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
