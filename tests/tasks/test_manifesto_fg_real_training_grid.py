from __future__ import annotations

import importlib
import json
from pathlib import Path

from src.ctreepo.distillation import write_labeled_trees_jsonl
from src.tree.labeled import LabeledNode, LabeledTree


def _tree(idx: int, split: str, *, leaf_count: int) -> LabeledTree:
    doc_id = f"doc_{idx}"
    root_score = 2.0 + idx
    tree = LabeledTree(
        doc_id=doc_id,
        document_text=f"Manifesto {idx}",
        document_score=root_score,
        label_source="fake_teacher_fg",
        metadata={
            "split": split,
            "dimension": "economic",
            "expert_score_1_7": root_score + 0.1,
            "teacher_score_1_7_existing_root": root_score - 0.1,
        },
    )
    leaves = []
    for leaf_idx in range(leaf_count):
        node = LabeledNode(
            node_id=f"node_l0_{leaf_idx:05d}",
            doc_id=doc_id,
            level=0,
            text=f"Leaf span {idx}-{leaf_idx}",
            score=root_score - 0.2 + leaf_idx * 0.1,
            metadata={
                "is_leaf": True,
                "teacher_summary": f"Leaf summary {idx}-{leaf_idx}",
                "target_summary": f"Leaf summary {idx}-{leaf_idx}",
            },
        )
        tree.add_node(node)
        leaves.append(node)
    if leaf_count == 1:
        return tree
    parent = LabeledNode(
        node_id="node_l1_00000",
        doc_id=doc_id,
        level=1,
        text=f"Root span {idx}",
        score=root_score,
        left_child_id=leaves[0].node_id,
        right_child_id=leaves[-1].node_id,
        metadata={
            "is_leaf": False,
            "teacher_summary": f"Root summary {idx}",
            "target_summary": f"Root summary {idx}",
        },
    )
    tree.add_node(parent)
    return tree


def test_real_training_grid_dry_run_smoke_writes_row_manifests(tmp_path: Path) -> None:
    cli = importlib.import_module("scripts.run_manifesto_fg_real_training_grid")
    grid = tmp_path / "fg_grid"
    splits = ["train", "train", "val", "test"]
    for leaf_count in (1, 8):
        leaf_dir = grid / f"leaf_{leaf_count:03d}"
        leaf_dir.mkdir(parents=True)
        write_labeled_trees_jsonl(
            leaf_dir / "labeled_trees.jsonl",
            [_tree(idx, split, leaf_count=leaf_count) for idx, split in enumerate(splits)],
        )

    init_artifact = tmp_path / "init" / "labeled_trees.jsonl"
    write_labeled_trees_jsonl(
        init_artifact,
        [_tree(idx, split, leaf_count=1) for idx, split in enumerate(splits)],
    )
    output_dir = tmp_path / "real_grid"
    rc = cli.main(
        [
            "--fg-grid-dir",
            str(grid),
            "--benoit-init-artifact",
            str(init_artifact),
            "--full-doc-init-artifact",
            str(init_artifact),
            "--output-dir",
            str(output_dir),
            "--leaf-grid",
            "1,8",
            "--smoke",
            "--dry-run",
            "--epochs",
            "1",
        ]
    )

    assert rc == 0
    manifest = json.loads((output_dir / "grid_manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["dry_run"] is True
    assert manifest["config"]["init_modes"] == ["fresh"]
    assert manifest["config"]["leaf_grid"] == [1, 8]
    assert manifest["config"]["backend_matrix"] == "smoke"
    assert manifest["rows_total"] == 16

    fgf = json.loads(
        (
            output_dir
            / "rows"
            / "fresh_leaf_008_fgf_g-teacher_f-trl_lm"
            / "row_manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    assert fgf["command_result"]["status"] == "dry_run"
    assert fgf["g_backend"] == "teacher"
    assert fgf["f_backend"] == "trl_lm"
    assert fgf["artifacts"]["f_artifact_type"] == "hf_trl_lm"
    assert fgf["teacher_metrics"]["tree_counts"]["test"] == 1
    assert fgf["teacher_metrics"]["summary_target_coverage"] == 1.0
    assert fgf["f_lm_evaluation"]["status"] == "skipped"
    cmd = fgf["step_results"][0]["command"]
    assert "--run-f-lm-regression" in cmd
    assert "--skip-f-fit" in cmd
    assert "--target-min" in cmd and "1.0" in cmd
    assert "--target-max" in cmd and "7.0" in cmd

    fgfg = json.loads(
        (
            output_dir
            / "rows"
            / "fresh_leaf_008_fgfg_g-trl_lm_f-trl_lm"
            / "row_manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    assert fgfg["artifacts"]["g_artifact_type"] == "hf_trl_lm"
    assert fgfg["artifacts"]["f_artifact_type"] == "hf_trl_lm"
    assert "--run-g-sft" in fgfg["step_results"][0]["command"]

    dspy = json.loads(
        (
            output_dir
            / "rows"
            / "fresh_leaf_008_fgf_g-teacher_f-dspy_lm"
            / "row_manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    assert dspy["artifacts"]["f_artifact_type"] == "dspy_program"
    assert dspy["distillation_manifest"]["f"]["test_records"] > 0

    embedding = json.loads(
        (
            output_dir
            / "rows"
            / "fresh_leaf_008_fgfg_g-dspy_lm_f-embedding_ridge"
            / "row_manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    assert embedding["artifacts"]["g_artifact_type"] == "dspy_program"
    assert embedding["artifacts"]["f_artifact_type"] == "embedding_proxy_json"
    assert embedding["step_results"][0]["state_artifact_type"] == "labeled_tree_g_states"
