from __future__ import annotations

import json
from pathlib import Path

from scripts.run_manifesto_dimension_fit_existing_results import main
from src.ctreepo.distillation import write_labeled_trees_jsonl
from src.experiments.embedding_clients import HashingEmbeddingClient
from src.tasks.manifesto.result_trees import build_partial_labeled_tree


def _example_text(idx: int) -> str:
    return (
        f"Document {idx} discusses economic investment and jobs. "
        "It also mentions tax fairness, public services, welfare policy, "
        "and industry planning for a stable economy."
    )


def test_existing_result_row_projects_to_partial_labeled_tree(tmp_path: Path):
    row = {
        "manifesto_id": "doc_1",
        "summary": "The manifesto emphasizes economic investment.",
        "llm_score_1_7": 5.5,
        "benoit_expert_mean": 4.25,
    }

    tree = build_partial_labeled_tree(
        row=row,
        text=_example_text(1),
        split="train",
        dimension="economic",
        target_source="teacher",
        expert_target_scale="raw_benoit",
        chunk_chars=48,
        source_results_path=tmp_path / "per_manifesto.jsonl",
    )

    assert tree is not None
    assert tree.metadata["topology_replay"] == "exact_artifact_spans"
    assert tree.metadata["partial_artifact"] is True
    assert tree.metadata["teacher_score_1_7"] == 5.5
    assert tree.metadata["expert_score_1_7"] == 4.25
    assert len(tree.get_leaves()) >= 1

    root = tree.get_node(tree.levels[-1][0])
    assert root is not None
    assert root.metadata["teacher_summary"] == row["summary"]
    assert root.metadata["f_input_kind"] == "summary_embedding"


def test_existing_results_fit_smoke_uses_contract_runner_and_hashing_embeddings(tmp_path: Path):
    source_path = tmp_path / "per_manifesto.jsonl"
    report_path = tmp_path / "report.json"
    output_dir = tmp_path / "fit"

    with source_path.open("w", encoding="utf-8") as handle:
        for idx in range(7):
            row = {
                "manifesto_id": f"doc_{idx}",
                "text": _example_text(idx),
                "summary": f"Economic summary {idx}",
                "llm_score_1_7": 1.5 + 0.5 * idx,
                "benoit_expert_mean": 1.25 + 0.4 * idx,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report_path.write_text(
        json.dumps({"run": {"chunk_chars": 48, "model": "fake-gemma-4"}}, sort_keys=True),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--dimension",
                "economic",
                "--source-results",
                str(source_path),
                "--source-report",
                str(report_path),
                "--output-dir",
                str(output_dir),
                "--split-source",
                "results-order",
                "--train-n",
                "4",
                "--val-n",
                "1",
                "--test-n",
                "2",
                "--embedding-backend",
                "hashing",
                "--hashing-embedding-dim",
                "32",
            ]
        )
        == 0
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["config"]["chunk_chars"] == 48
    assert manifest["config"]["target_source"] == "teacher"
    assert manifest["contract"]["supervision_source"] == "labeled_tree_artifact"
    assert manifest["tree_counts"]["total"] == 7
    assert manifest["dataset_counts"]["g_sft_records"] == 7
    assert manifest["dataset_counts"]["f_lm_records"] == 7
    assert (output_dir / "labeled_trees.jsonl").exists()
    assert (output_dir / "g_fit" / "g_sft_train.jsonl").exists()
    assert (output_dir / "f_lm_fit" / "f_lm_regression_train.jsonl").exists()
    assert (output_dir / "f_embedding_fit" / "f_embedding_proxy.json").exists()
    assert "f_embedding_test_report" in manifest["results"]
    assert manifest["config"]["finetune_export"]["enabled"] is True
    finetune = manifest["finetune"]
    assert finetune["bundle_kind"] == "manifesto_labeled_tree"
    assert finetune["summary"]["n_trees"] == 7
    assert Path(finetune["files"]["tree_records"]).exists()
    assert Path(finetune["finetune_adapters"]["adapters"]["trl_dpo"]["files"]["dpo"]).exists()


def test_generic_manifesto_labeled_tree_exporter_handles_existing_artifact(tmp_path: Path):
    from scripts import export_manifesto_labeled_tree_preferences as exporter

    tree = build_partial_labeled_tree(
        row={
            "manifesto_id": "doc_generic",
            "summary": "The manifesto emphasizes public investment.",
            "llm_score_1_7": 4.5,
            "benoit_expert_mean": 4.0,
        },
        text=_example_text(3),
        split="train",
        dimension="economic",
        target_source="teacher",
        expert_target_scale="normalized_1_7",
        chunk_chars=48,
        source_results_path=tmp_path / "per_manifesto.jsonl",
    )
    assert tree is not None
    labeled_path = write_labeled_trees_jsonl(tmp_path / "labeled_trees.jsonl", [tree])
    output_dir = tmp_path / "treepo_finetune"

    assert exporter.main(["--labeled-trees", str(labeled_path), "--output-dir", str(output_dir), "--kind", "generic"]) == 0

    result = json.loads((output_dir / "manifesto_labeled_tree_preferences_result.json").read_text(encoding="utf-8"))
    assert result["bundle_kind"] == "manifesto_labeled_tree"
    assert result["summary"]["n_trees"] == 1
    assert Path(result["files"]["tree_records"]).exists()
    assert Path(result["finetune_adapters"]["adapters"]["embedding"]["files"]["embedding_ranked"]).exists()


def test_hashing_embedding_client_is_deterministic():
    client = HashingEmbeddingClient(dim=32)

    first = client.embed_texts(["economic summary"])[0]
    second = client.embed_texts(["economic summary"])[0]

    assert first == second
    assert len(first) == 32
