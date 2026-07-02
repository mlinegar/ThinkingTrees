"""Shared q-sentence runner helpers for Manifesto script entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.ctreepo.distillation import load_labeled_trees
from src.tree.labeled import LabeledTree


def leafq_label(leaf_qsentences: int) -> str:
    return f"leafq{int(leaf_qsentences):03d}"


def leafq_dir(root: str | Path, leaf_qsentences: int) -> Path:
    return Path(root) / leafq_label(int(leaf_qsentences))


def load_leafq_trees(fg_grid_dir: str | Path, leaf_qsentences: int) -> Optional[list[LabeledTree]]:
    path = leafq_dir(fg_grid_dir, int(leaf_qsentences)) / "labeled_trees.jsonl"
    if not path.exists():
        return None
    return load_labeled_trees(path)


def format_leaf_artifact_template(
    value: Optional[str],
    leaf_qsentences: int,
) -> Optional[str]:
    if value is None or not str(value).strip():
        return None
    leaf = int(leaf_qsentences)
    label = leafq_label(leaf)
    return str(value).format(
        leaf=leaf,
        leaf_qsentences=leaf,
        leafq=label,
        row_label=label,
    )


def resolve_leaf_artifact(
    static_value: Optional[str],
    template_value: Optional[str],
    default: str,
    kind: str,
    leaf_qsentences: int,
) -> str:
    if static_value and template_value:
        raise ValueError(
            f"Use either --dspy-initial-{kind}-artifact or "
            f"--dspy-initial-{kind}-artifact-template, not both."
        )
    resolved = format_leaf_artifact_template(template_value, int(leaf_qsentences))
    if resolved:
        return resolved
    if static_value and str(static_value).strip():
        return str(static_value).strip()
    return default


def retarget_trees_to_dimension(trees: list[LabeledTree] | tuple[LabeledTree, ...], dimension: str) -> int:
    """Rewrite scalar node targets to a compact dimension for one-head FNO runs."""
    n_set = 0
    for tree in trees:
        nodes = tree.nodes
        node_list = list(nodes.values()) if isinstance(nodes, dict) else list(nodes)
        child_ids = set()
        for node in node_list:
            for cid in (
                getattr(node, "left_child_id", None),
                getattr(node, "right_child_id", None),
            ):
                if cid is not None:
                    child_ids.add(cid)
        for node in node_list:
            dim_scores = getattr(node, "dimension_scores", None) or {}
            if dimension in dim_scores:
                node.score = float(dim_scores[dimension])
                n_set += 1
        roots = [
            node
            for node in node_list
            if getattr(node, "node_id", None) not in child_ids
        ]
        if roots:
            root_scores = getattr(roots[0], "dimension_scores", None) or {}
            if dimension in root_scores:
                tree.document_score = float(root_scores[dimension])
    return n_set
