from __future__ import annotations

import importlib
import json
from pathlib import Path

from src.ctreepo.distillation import load_labeled_trees
from src.preprocessing.leaf_size_utils import count_tokens


def _example_text(idx: int) -> str:
    return (
        f"Manifesto {idx} supports public services, jobs, and fiscal responsibility. "
        "It discusses tax policy, welfare, investment, and public administration. "
        "The text contains enough economic policy evidence for a short tree."
    )


def _fake_chat(self, *, system: str, user: str, temperature: float, max_tokens: int) -> str:  # noqa: ARG001
    if "scalar scorer" in system or "Return only one number" in system:
        return json.dumps({"score": 5, "reasoning": "economic evidence"})
    if "idempotence" in system.lower() or "Resummarize" in user:
        return "Resummarized economic teacher state."
    if "LEFT_CHILD_SUMMARY" in user:
        return "Merged economic teacher state."
    return "Leaf economic teacher state."


def test_teacher_fg_leaf_grid_writes_node_aligned_labeled_trees(tmp_path: Path, monkeypatch) -> None:
    cli = importlib.import_module("scripts.run_manifesto_teacher_fg_leaf_grid")
    monkeypatch.setattr(cli.OpenAIChatClient, "chat", _fake_chat)

    source_path = tmp_path / "per_manifesto.jsonl"
    output_dir = tmp_path / "teacher_fg"
    with source_path.open("w", encoding="utf-8") as handle:
        for idx in range(4):
            row = {
                "manifesto_id": f"doc_{idx}",
                "text": _example_text(idx),
                "llm_score_1_7": 4.0 + idx * 0.1,
                "benoit_expert_mean": 4.2 + idx * 0.1,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    rc = cli.main(
        [
            "--dimension",
            "economic",
            "--source-results",
            str(source_path),
            "--output-dir",
            str(output_dir),
            "--split-source",
            "results-order",
            "--train-n",
            "2",
            "--val-n",
            "1",
            "--test-n",
            "1",
            "--leaf-grid",
            "2",
            "--num-workers",
            "1",
            "--idempotence-mode",
            "all",
        ]
    )

    assert rc == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["leaf_grid"] == [2]
    assert manifest["config"]["schema_version"] == "ctreepo.tree_bundle.v1"
    assert manifest["config"]["source_kind"] == "raw_input"
    assert manifest["config"]["leaf_unit"] == "text_token"
    assert manifest["config"]["state_contract"] == "raw_concat"
    assert manifest["config"]["reducer_contract"] == "bottom_up"
    assert manifest["config"]["tree_bundle_kind"] == "raw_manifesto_token_tree"
    assert manifest["config"]["finetune_export"]["enabled"] is True
    assert manifest["config"]["tree_state_source"] == "raw_input"
    assert manifest["config"]["tree_text_source"] == "aligned_text"
    assert manifest["run_manifest"]["schema_version"] == "ctreepo.run_manifest.v1"
    assert manifest["run_manifest"]["domain"] == "manifesto_rile"
    assert manifest["run_manifest"]["role"] == "teacher_tree_bundle"
    assert manifest["run_manifest"]["publication_ready"] is True
    assert manifest["teacher_fg_model"]["g_model"]

    leaf_dir = output_dir / "leaf_002"
    trees = load_labeled_trees(leaf_dir / "labeled_trees.jsonl")
    assert len(trees) == 4
    assert (leaf_dir / "teacher_node_rows.jsonl").exists()
    assert (leaf_dir / "teacher_g_summary_cache.jsonl").exists()
    assert (leaf_dir / "teacher_f_score_cache.jsonl").exists()
    finetune = manifest["runs"]["2"]["finetune"]
    assert finetune["bundle_kind"] == "manifesto_labeled_tree"
    assert finetune["summary"]["n_trees"] == 4
    assert finetune["counts"]["dataset"] == 16
    assert finetune["finetune_adapters"]["summary"]["n_adapters"] == 7
    assert Path(finetune["files"]["tree_records"]).exists()
    assert Path(finetune["finetune_adapters"]["adapters"]["trl_dpo"]["files"]["dpo"]).exists()
    pref_dataset = json.loads(Path(finetune["files"]["dataset"]).read_text(encoding="utf-8"))
    assert {row["target"] for row in pref_dataset["units"]} == {"f", "g"}
    assert any(row["unit_type"] == "root" and row["target"] == "f" for row in pref_dataset["units"])
    assert any(row["unit_type"] == "text_token" and row["target"] == "g" for row in pref_dataset["units"])

    first = trees[0]
    assert first.metadata["schema_version"] == "ctreepo.tree_bundle.v1"
    assert first.metadata["source_kind"] == "raw_input"
    assert first.metadata["state_contract"] == "raw_concat"
    assert first.metadata["reducer_contract"] == "bottom_up"
    assert first.metadata["tree_bundle_kind"] == "raw_manifesto_token_tree"
    assert first.metadata["tree_state_source"] == "raw_input"
    assert first.metadata["tree_text_source"] == "aligned_text"
    assert first.metadata["node_score_source"] == "teacher_f_dimension_score_1_7"
    assert first.metadata["summary_coverage"]["partial_artifact"] is False
    assert len(first.get_leaves()) == 2
    assert len(first.metadata["idempotence_pairs"]) == len(first.nodes)
    assert all(node.metadata.get("teacher_summary") for node in first.nodes.values())
    assert all(float(node.score) == 5.0 for node in first.nodes.values())


def test_teacher_fg_leaf_size_tokens_writes_size_axis_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    cli = importlib.import_module("scripts.run_manifesto_teacher_fg_leaf_grid")
    monkeypatch.setattr(cli.OpenAIChatClient, "chat", _fake_chat)

    source_path = tmp_path / "per_manifesto.jsonl"
    output_dir = tmp_path / "teacher_fg_size"
    with source_path.open("w", encoding="utf-8") as handle:
        for idx in range(4):
            row = {
                "manifesto_id": f"size_doc_{idx}",
                "text": _example_text(idx) + " " + _example_text(idx),
                "llm_score_1_7": 4.0 + idx * 0.1,
                "benoit_expert_mean": 4.2 + idx * 0.1,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    rc = cli.main(
        [
            "--dimension",
            "economic",
            "--source-results",
            str(source_path),
            "--output-dir",
            str(output_dir),
            "--split-source",
            "results-order",
            "--train-n",
            "2",
            "--val-n",
            "1",
            "--test-n",
            "1",
            "--leaf-size-tokens",
            "16",
            "--num-workers",
            "1",
            "--idempotence-mode",
            "off",
        ]
    )

    assert rc == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["topology_axis"] == "size_tokens"
    assert manifest["config"]["leaf_grid"] is None
    assert manifest["config"]["leaf_size_tokens"] == [16]
    assert manifest["config"]["source_kind"] == "raw_input"
    assert manifest["config"]["tree_bundle_kind"] == "raw_manifesto_token_tree"

    leaf_dir = output_dir / "leaf0016tok"
    summary = json.loads((leaf_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "ctreepo.tree_bundle.v1"
    assert summary["source_kind"] == "raw_input"
    assert summary["tree_bundle_kind"] == "raw_manifesto_token_tree"
    trees = load_labeled_trees(leaf_dir / "labeled_trees.jsonl")
    assert len(trees) == 4
    first = trees[0]
    assert first.metadata["topology_axis"] == "size_tokens"
    assert first.metadata["leaf_count"] is None
    assert first.metadata["leaf_size_tokens"] == 16
    assert first.metadata["topology_policy"]["kind"] == "explicit_char_windows"
    assert first.metadata["derived_leaf_count"] == len(first.get_leaves())
    assert "".join(leaf.text for leaf in first.get_leaves()) == first.document_text
    assert all(count_tokens(leaf.text) <= 16 for leaf in first.get_leaves())


def test_teacher_fg_leaf_grid_writes_external_summary_bundle_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    cli = importlib.import_module("scripts.run_manifesto_teacher_fg_leaf_grid")
    monkeypatch.setattr(cli.OpenAIChatClient, "chat", _fake_chat)

    source_path = tmp_path / "per_manifesto.jsonl"
    output_dir = tmp_path / "teacher_fg_summary_bundle"
    with source_path.open("w", encoding="utf-8") as handle:
        for idx in range(4):
            row = {
                "manifesto_id": f"summary_doc_{idx}",
                "text": _example_text(idx),
                "summary": f"External g_benoit summary for manifesto {idx}.",
                "llm_score_1_7": 4.0 + idx * 0.1,
                "benoit_expert_mean": 4.2 + idx * 0.1,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    rc = cli.main(
        [
            "--dimension",
            "economic",
            "--source-results",
            str(source_path),
            "--output-dir",
            str(output_dir),
            "--split-source",
            "results-order",
            "--train-n",
            "2",
            "--val-n",
            "1",
            "--test-n",
            "1",
            "--leaf-size-tokens",
            "16",
            "--source-kind",
            "external_state",
            "--external-state-producer",
            "g_benoit",
            "--num-workers",
            "1",
            "--idempotence-mode",
            "off",
        ]
    )

    assert rc == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["schema_version"] == "ctreepo.tree_bundle.v1"
    assert manifest["config"]["source_kind"] == "external_state"
    assert manifest["config"]["leaf_unit"] == "text_token"
    assert manifest["config"]["state_contract"] == "external_passthrough"
    assert manifest["config"]["reducer_contract"] == "bottom_up"
    assert manifest["config"]["tree_bundle_kind"] == "external_summary_token_tree"
    assert manifest["config"]["tree_state_source"] == "external_state"
    assert manifest["config"]["external_state_producer"] == "g_benoit"
    assert manifest["config"]["tree_text_source"] == "existing_summary"
    assert manifest["run_manifest"]["schema_version"] == "ctreepo.run_manifest.v1"
    assert manifest["run_manifest"]["publication_ready"] is False

    leaf_dir = output_dir / "leaf0016tok"
    summary = json.loads((leaf_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "ctreepo.tree_bundle.v1"
    assert summary["source_kind"] == "external_state"
    assert summary["tree_bundle_kind"] == "external_summary_token_tree"
    assert summary["external_state_producer"] == "g_benoit"
    trees = load_labeled_trees(leaf_dir / "labeled_trees.jsonl")
    assert len(trees) == 4
    assert trees[0].metadata["source_kind"] == "external_state"
    assert trees[0].metadata["state_contract"] == "external_passthrough"
    assert trees[0].metadata["tree_bundle_kind"] == "external_summary_token_tree"
    assert trees[0].metadata["external_state_producer"] == "g_benoit"
    assert "External g_benoit summary" in trees[0].document_text
