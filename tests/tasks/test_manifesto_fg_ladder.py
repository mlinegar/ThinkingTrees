from __future__ import annotations

import importlib
import json

import pytest
from pathlib import Path

from src.ctreepo.distillation import load_labeled_trees, write_labeled_trees_jsonl
from src.tree.labeled import LabeledNode, LabeledTree


def _make_fg_tree(idx: int, split: str) -> LabeledTree:
    doc_id = f"doc_{idx}"
    score = 2.0 + idx
    tree = LabeledTree(
        doc_id=doc_id,
        document_text=f"Manifesto text {idx} with economic policy.",
        document_score=score,
        label_source="fake_teacher_fg",
        metadata={
            "split": split,
            "dimension": "economic",
            "teacher_score_1_7_existing_root": score + 0.1,
            "expert_score_1_7": score + 0.2,
        },
    )
    left = LabeledNode(
        node_id="node_l0_00000",
        doc_id=doc_id,
        level=0,
        text=f"Left span {idx}",
        score=score - 0.2,
        metadata={
            "is_leaf": True,
            "teacher_summary": f"Left summary {idx}",
            "target_summary": f"Left summary {idx}",
        },
    )
    right = LabeledNode(
        node_id="node_l0_00001",
        doc_id=doc_id,
        level=0,
        text=f"Right span {idx}",
        score=score + 0.1,
        metadata={
            "is_leaf": True,
            "teacher_summary": f"Right summary {idx}",
            "target_summary": f"Right summary {idx}",
        },
    )
    parent = LabeledNode(
        node_id="node_l1_00000",
        doc_id=doc_id,
        level=1,
        text=f"Parent span {idx}",
        score=score,
        left_child_id=left.node_id,
        right_child_id=right.node_id,
        metadata={
            "is_leaf": False,
            "teacher_summary": f"Merged summary {idx}",
            "target_summary": f"Merged summary {idx}",
        },
    )
    tree.add_node(left)
    tree.add_node(right)
    tree.add_node(parent)
    tree.metadata["idempotence_pairs"] = [
        {
            "node_id": parent.node_id,
            "input_summary": f"Merged summary {idx}",
            "target_resummary": f"Merged summary again {idx}",
        }
    ]
    return tree


def _make_f_tree(idx: int, split: str) -> LabeledTree:
    doc_id = f"doc_{idx}"
    score = 2.0 + idx
    summary = f"Baseline summary {idx}"
    tree = LabeledTree(
        doc_id=doc_id,
        document_text=summary,
        document_score=score,
        label_source="fake_f_baseline",
        metadata={
            "split": split,
            "dimension": "economic",
            "teacher_score_1_7": score,
            "expert_score_1_7": score + 0.2,
        },
    )
    tree.add_node(
        LabeledNode(
            node_id="node_l0_00000",
            doc_id=doc_id,
            level=0,
            text=summary,
            score=score,
            metadata={
                "is_leaf": True,
                "teacher_summary": summary,
                "target_summary": summary,
            },
        )
    )
    return tree


@pytest.mark.skip(
    reason=(
        "subject archived: the ladder builder lives at "
        "scripts/OLD_build_manifesto_fg_ladder_legacy.py; current fg-ladder runs "
        "go through run_manifesto_fg_real_training_grid.py"
    )
)
def test_fg_ladder_exports_contract_fit_artifacts(tmp_path: Path) -> None:
    cli = importlib.import_module("scripts.build_manifesto_fg_ladder")
    splits = ["train", "train", "val", "test"]

    grid_dir = tmp_path / "fg_grid"
    leaf_dir = grid_dir / "leaf_002"
    leaf_dir.mkdir(parents=True)
    fg_trees = [_make_fg_tree(idx, split) for idx, split in enumerate(splits)]
    write_labeled_trees_jsonl(leaf_dir / "labeled_trees.jsonl", fg_trees)

    f_path = tmp_path / "f_baseline" / "labeled_trees.jsonl"
    f_trees = [_make_f_tree(idx, split) for idx, split in enumerate(splits)]
    write_labeled_trees_jsonl(f_path, f_trees)

    f_doc_path = tmp_path / "f_doc" / "labeled_trees.jsonl"
    f_doc_trees = [_make_f_tree(idx, split) for idx, split in enumerate(splits)]
    for tree in f_doc_trees:
        tree.metadata["summary_representation"] = "raw_whole_document"
        for node in tree.nodes.values():
            node.text = f"Raw whole document text for {tree.doc_id}"
            node.metadata.pop("teacher_summary", None)
            node.metadata.pop("target_summary", None)
    write_labeled_trees_jsonl(f_doc_path, f_doc_trees)

    source_results = tmp_path / "per_manifesto.jsonl"
    with source_results.open("w", encoding="utf-8") as handle:
        for idx in range(4):
            handle.write(
                json.dumps(
                    {
                        "manifesto_id": f"doc_{idx}",
                        "summary": f"Baseline summary {idx}",
                        "llm_score_1_7": 2.0 + idx,
                        "benoit_expert_mean": 2.2 + idx,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    out = tmp_path / "ladder"
    rc = cli.main(
        [
            "--dimension",
            "economic",
            "--source-results",
            str(source_results),
            "--fg-grid-dir",
            str(grid_dir),
            "--f-baseline-labeled-trees",
            str(f_path),
            "--f-doc-labeled-trees",
            str(f_doc_path),
            "--output-dir",
            str(out),
            "--leaf-grid",
            "2",
            "--embedding-backend",
            "hashing",
            "--hashing-embedding-dim",
            "32",
        ]
    )
    assert rc == 0

    manifest = json.loads((out / "fg_ladder_manifest.json").read_text(encoding="utf-8"))
    assert set(["f", "f_doc", "fg", "fgf", "fgfg"]).issubset(manifest)
    assert manifest["composition_ladder"]["notation"]["right_to_left"] == "fg means f after g"
    assert manifest["missing_leaves"] == []
    assert manifest["fg"]["leaf_002"]["node_count"] == 12
    assert manifest["f"]["tree_counts"]["total"] == 4
    assert manifest["f_doc"]["tree_counts"]["total"] == 4

    assert (out / "f" / "f_lm_records" / "fit" / "f_lm_regression_train.jsonl").exists()
    assert (out / "f_doc" / "f_lm_records" / "fit" / "f_lm_regression_train.jsonl").exists()
    assert (out / "f" / "f_embedding_proxy" / "fit" / "f_embedding_proxy.json").exists()
    assert (out / "f_doc" / "f_embedding_proxy" / "fit" / "f_embedding_proxy.json").exists()
    assert (out / "leaf_002_fgf" / "f_lm_records" / "fit" / "f_lm_regression_train.jsonl").exists()
    assert (out / "leaf_002_fgf" / "f_embedding_proxy" / "fit" / "f_embedding_proxy.json").exists()
    assert (out / "leaf_002_fgfg" / "g_sft_records" / "fit" / "g_sft_train.jsonl").exists()

    loaded = load_labeled_trees(leaf_dir / "labeled_trees.jsonl")
    assert len(loaded) == 4
