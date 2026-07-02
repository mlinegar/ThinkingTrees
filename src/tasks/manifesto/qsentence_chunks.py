from __future__ import annotations

from pathlib import Path
from typing import Literal

from src.ctreepo.distillation import load_labeled_trees

NodeLevelSelection = Literal["leaves", "merges", "all"]


def collect_chunks(
    grid_dir: str | Path,
    leaf: int,
    node_levels: NodeLevelSelection | str = "leaves",
) -> list[tuple[str, str, str]]:
    """Collect q-sentence tree nodes as ``(doc_id, node_id, span_text)`` rows."""
    selection = str(node_levels)
    if selection not in {"leaves", "merges", "all"}:
        raise ValueError("node_levels must be one of: leaves, merges, all")
    trees = load_labeled_trees(
        Path(grid_dir) / f"leafq{int(leaf):03d}" / "labeled_trees.jsonl"
    )
    out: list[tuple[str, str, str]] = []
    for tree in trees:
        for node in tree.nodes.values():
            level = int(node.level)
            keep = (
                (selection == "leaves" and level == 0)
                or (selection == "merges" and level > 0)
                or selection == "all"
            )
            text = str(node.text or "").strip()
            if keep and text:
                out.append((str(tree.doc_id), str(node.node_id), str(node.text)))
    return out


__all__ = ["NodeLevelSelection", "collect_chunks"]
