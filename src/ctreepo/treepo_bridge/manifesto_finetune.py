"""Shared Manifesto fine-tune export helpers.

This module keeps runner scripts out of the fine-tune adapter plumbing. The
runners should only decide which labeled trees to export and where the sidecar
bundle belongs.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from src.ctreepo.treepo_bridge.manifesto_preferences import (
    DEFAULT_MANIFESTO_FINETUNE_ADAPTERS,
    DEFAULT_MANIFESTO_LEARNING_ADAPTERS,
    DEFAULT_QSENTENCE_FINETUNE_ADAPTERS,
    DEFAULT_QSENTENCE_LEARNING_ADAPTERS,
    export_manifesto_labeled_tree_finetune_bundle,
    export_manifesto_qsentence_finetune_bundle,
)
from src.tree.labeled import LabeledTree

ManifestoFineTuneKind = Literal["generic", "qsentence"]


def parse_name_grid(value: Any, *, default: Sequence[str] | None = None) -> tuple[str, ...]:
    """Parse comma/semicolon-separated adapter names."""

    if value is None:
        return tuple(default or ())
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return tuple()
        values = raw.replace(";", ",").split(",")
    elif isinstance(value, Sequence):
        values = value
    else:
        values = (value,)
    return tuple(str(item).strip() for item in values if str(item).strip())


def default_finetune_adapters(kind: ManifestoFineTuneKind) -> tuple[str, ...]:
    if kind == "qsentence":
        return DEFAULT_QSENTENCE_FINETUNE_ADAPTERS
    return DEFAULT_MANIFESTO_FINETUNE_ADAPTERS


def default_learning_adapters(kind: ManifestoFineTuneKind) -> tuple[str, ...]:
    if kind == "qsentence":
        return DEFAULT_QSENTENCE_LEARNING_ADAPTERS
    return DEFAULT_MANIFESTO_LEARNING_ADAPTERS


def add_manifesto_finetune_args(
    parser: argparse.ArgumentParser,
    *,
    kind: ManifestoFineTuneKind,
    help_text: str = "Write treepo PreferenceDataset/fine-tune adapter bundles.",
    default_enabled: bool = True,
) -> None:
    """Add the common Manifesto fine-tune export flags to a runner parser."""

    parser.add_argument(
        "--export-finetune-views",
        action=argparse.BooleanOptionalAction,
        default=default_enabled,
        help=help_text,
    )
    parser.add_argument("--finetune-mode", choices=("scores", "pairwise", "ranked"), default="ranked")
    parser.add_argument("--finetune-max-records", type=int, default=None)
    parser.add_argument("--finetune-adapters", default=",".join(default_finetune_adapters(kind)))
    parser.add_argument("--learning-adapters", default=",".join(default_learning_adapters(kind)))
    parser.add_argument("--save-finetune-hf", action="store_true")


def finetune_export_config(args: Any) -> dict[str, Any]:
    """Return manifest-safe config for common fine-tune export flags."""

    return {
        "enabled": bool(getattr(args, "export_finetune_views", True)),
        "mode": str(getattr(args, "finetune_mode", "ranked")),
        "adapters": list(parse_name_grid(getattr(args, "finetune_adapters", ""))),
        "learning_adapters": list(parse_name_grid(getattr(args, "learning_adapters", ""))),
        "save_hf": bool(getattr(args, "save_finetune_hf", False)),
    }


def export_manifesto_finetune_bundle_from_args(
    *,
    args: Any,
    trees: Sequence[LabeledTree],
    output_dir: Path,
    kind: ManifestoFineTuneKind,
    leaf_unit_type: str = "leaf",
    logger: logging.Logger | None = None,
    log_label: str = "Manifesto",
    respect_enabled: bool = True,
) -> dict[str, Any] | None:
    """Export one Manifesto fine-tune sidecar bundle using standard runner flags."""

    if respect_enabled and not bool(getattr(args, "export_finetune_views", True)):
        return None
    mode = str(getattr(args, "finetune_mode", "ranked"))
    max_records = getattr(args, "finetune_max_records", None)
    adapters = parse_name_grid(
        getattr(args, "finetune_adapters", None),
        default=default_finetune_adapters(kind),
    )
    learning_adapters = parse_name_grid(
        getattr(args, "learning_adapters", None),
        default=default_learning_adapters(kind),
    )
    save_hf = bool(getattr(args, "save_finetune_hf", False))

    if kind == "qsentence":
        bundle = export_manifesto_qsentence_finetune_bundle(
            trees,
            output_dir,
            mode=mode,
            max_records=max_records,
            adapters=adapters,
            learning_adapters=learning_adapters,
            save_hf=save_hf,
        )
    else:
        bundle = export_manifesto_labeled_tree_finetune_bundle(
            trees,
            output_dir,
            mode=mode,
            max_records=max_records,
            leaf_unit_type=str(leaf_unit_type or "leaf"),
            adapters=adapters,
            learning_adapters=learning_adapters,
            save_hf=save_hf,
        )

    if logger is not None:
        logger.info(
            "Wrote %s fine-tune bundle for %s: units=%s adapters=%s",
            log_label,
            output_dir,
            bundle.get("summary", {}).get("n_units"),
            bundle.get("finetune_adapters", {}).get("summary", {}).get("n_adapters"),
        )
    return bundle


def resolve_manifesto_finetune_kind(kind: str, trees: Sequence[LabeledTree]) -> ManifestoFineTuneKind:
    normalized = str(kind or "auto").strip().lower()
    if normalized == "auto":
        return "qsentence" if looks_like_qsentence_bundle(trees) else "generic"
    if normalized in {"generic", "qsentence"}:
        return normalized  # type: ignore[return-value]
    raise ValueError(f"unknown Manifesto fine-tune kind: {kind!r}")


def looks_like_qsentence_bundle(trees: Sequence[LabeledTree]) -> bool:
    for tree in list(trees)[:8]:
        metadata = dict(tree.metadata or {})
        if "qsentence" in str(metadata.get("label_source") or "").lower():
            return True
        for node in tree.nodes.values():
            node_meta = dict(node.metadata or {})
            if "leaf_qsentences" in node_meta or "qsentence_start_index" in node_meta:
                return True
    return False


__all__ = [
    "DEFAULT_MANIFESTO_FINETUNE_ADAPTERS",
    "DEFAULT_MANIFESTO_LEARNING_ADAPTERS",
    "DEFAULT_QSENTENCE_FINETUNE_ADAPTERS",
    "DEFAULT_QSENTENCE_LEARNING_ADAPTERS",
    "ManifestoFineTuneKind",
    "add_manifesto_finetune_args",
    "default_finetune_adapters",
    "default_learning_adapters",
    "export_manifesto_finetune_bundle_from_args",
    "finetune_export_config",
    "looks_like_qsentence_bundle",
    "parse_name_grid",
    "resolve_manifesto_finetune_kind",
]
