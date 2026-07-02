from __future__ import annotations

from pathlib import Path

import pytest

from src.ctreepo.distillation import write_labeled_trees_jsonl
from src.tasks.manifesto.qsentence_chunks import collect_chunks
from src.tree.labeled import LabeledNode, LabeledTree


def _tree() -> LabeledTree:
    tree = LabeledTree(doc_id="doc1", document_text="text", document_score=1.0)
    left = LabeledNode(node_id="leaf0", doc_id="doc1", level=0, text="left text", score=1.0)
    right = LabeledNode(node_id="leaf1", doc_id="doc1", level=0, text="right text", score=1.0)
    root = LabeledNode(
        node_id="root",
        doc_id="doc1",
        level=1,
        text="root text",
        score=1.0,
        left_child_id="leaf0",
        right_child_id="leaf1",
    )
    tree.add_node(left)
    tree.add_node(right)
    tree.add_node(root)
    return tree


def test_collect_qsentence_chunks_selects_requested_node_levels(tmp_path: Path) -> None:
    leaf_dir = tmp_path / "leafq016"
    write_labeled_trees_jsonl(leaf_dir / "labeled_trees.jsonl", [_tree()])

    assert collect_chunks(tmp_path, 16, "leaves") == [
        ("doc1", "leaf0", "left text"),
        ("doc1", "leaf1", "right text"),
    ]
    assert collect_chunks(tmp_path, 16, "merges") == [("doc1", "root", "root text")]
    assert [node_id for _, node_id, _ in collect_chunks(tmp_path, 16, "all")] == [
        "leaf0",
        "leaf1",
        "root",
    ]


def test_collect_qsentence_chunks_rejects_unknown_selection(tmp_path: Path) -> None:
    leaf_dir = tmp_path / "leafq016"
    write_labeled_trees_jsonl(leaf_dir / "labeled_trees.jsonl", [_tree()])

    with pytest.raises(ValueError):
        collect_chunks(tmp_path, 16, "roots")
