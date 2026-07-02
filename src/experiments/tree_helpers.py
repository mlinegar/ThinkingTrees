from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.tree.labeled import LabeledNode, LabeledTree


def root_node(tree: LabeledTree) -> Optional[LabeledNode]:
    if not tree.levels:
        return None
    for node_id in reversed(tree.levels[-1]):
        node = tree.get_node(str(node_id))
        if node is not None:
            return node
    return None


def root_node_id(tree: LabeledTree) -> str:
    node = root_node(tree)
    return str(getattr(node, "node_id", "") or "") if node is not None else ""


def root_node_ids(trees: Sequence[LabeledTree]) -> dict[str, str]:
    out: dict[str, str] = {}
    for tree in trees:
        rid = root_node_id(tree)
        if rid:
            out[str(tree.doc_id)] = rid
    return out


def tree_node_lookup(trees: Sequence[LabeledTree]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for tree in trees:
        for node in tree.nodes.values():
            out[(str(tree.doc_id), str(node.node_id))] = {
                "tree": tree,
                "node": node,
                "metadata": dict(node.metadata or {}),
            }
    return out


def select_trees_by_splits(trees: Sequence[LabeledTree], splits: Sequence[str]) -> list[LabeledTree]:
    wanted = {str(split).lower() for split in splits}
    return [
        tree
        for tree in trees
        if str((tree.metadata or {}).get("split") or "").lower() in wanted
    ]


def split_trees_for_eval(
    trees: Sequence[LabeledTree],
    *,
    eval_split: str,
    train_split: str,
) -> tuple[list[LabeledTree], list[LabeledTree]]:
    train_trees: list[LabeledTree] = []
    eval_trees: list[LabeledTree] = []
    for tree in trees:
        split = str((tree.metadata or {}).get("split") or "").lower()
        if split == train_split.lower():
            train_trees.append(tree)
        if split == eval_split.lower():
            eval_trees.append(tree)
    return train_trees, eval_trees


def node_summary(node: LabeledNode) -> str:
    metadata = node.metadata if isinstance(node.metadata, Mapping) else {}
    return str(metadata.get("teacher_summary") or metadata.get("target_summary") or "").strip()


def root_summary_for_tree(tree: LabeledTree) -> str:
    node = root_node(tree)
    return node_summary(node) if node is not None else ""


def summary_coverage(trees: Sequence[LabeledTree]) -> dict[str, Any]:
    total_nodes = 0
    with_summary = 0
    root_with_summary = 0
    for tree in trees:
        for node in tree.nodes.values():
            total_nodes += 1
            if node_summary(node):
                with_summary += 1
        if root_summary_for_tree(tree):
            root_with_summary += 1
    return {
        "trees": len(trees),
        "total_nodes": total_nodes,
        "nodes_with_summary": with_summary,
        "node_summary_rate": float(with_summary / total_nodes) if total_nodes else None,
        "roots_with_summary": root_with_summary,
        "root_summary_rate": float(root_with_summary / len(trees)) if trees else None,
        "partial_artifact": bool(with_summary < total_nodes) if total_nodes else True,
    }


def load_leaf_count_trees(root: str | Path, leaf_count: int) -> Optional[list[LabeledTree]]:
    from src.ctreepo.distillation import load_labeled_trees

    path = Path(root) / f"leaf_{int(leaf_count):03d}" / "labeled_trees.jsonl"
    if not path.exists():
        return None
    return load_labeled_trees(path)


def load_leaf_size_trees(root: str | Path, leaf_size_tokens: int) -> Optional[list[LabeledTree]]:
    from src.ctreepo.distillation import load_labeled_trees

    path = Path(root) / f"leaf{int(leaf_size_tokens):04d}tok" / "labeled_trees.jsonl"
    if not path.exists():
        return None
    return load_labeled_trees(path)


__all__ = [
    "load_leaf_count_trees",
    "load_leaf_size_trees",
    "node_summary",
    "root_node",
    "root_node_id",
    "root_node_ids",
    "root_summary_for_tree",
    "select_trees_by_splits",
    "split_trees_for_eval",
    "summary_coverage",
    "tree_node_lookup",
]
