from __future__ import annotations

import importlib
import json
from pathlib import Path

from src.ctreepo.distillation import load_labeled_trees


def _example_text(idx: int) -> str:
    return (
        f"Manifesto {idx} discusses jobs, welfare, civil rights, and public services. "
        "It has enough policy evidence to form several token leaves."
    )


def _fake_joint_chat(self, *, system: str, user: str, temperature: float, max_tokens: int) -> str:  # noqa: ARG001
    return json.dumps(
        {
            "scores": {
                "economic": 5.0,
                "social": 4.0,
            },
            "reasoning": "joint score smoke",
        }
    )


def test_joint_teacher_fg_grid_defaults_to_generic_raw_tree_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    cli = importlib.import_module("scripts.run_manifesto_teacher_fg_joint_leaf_grid")
    monkeypatch.setattr(cli.OpenAIChatClient, "chat", _fake_joint_chat)

    from src.preprocessing import leaf_size_utils

    def fake_windows(text: str, leaf_size_tokens: int):  # noqa: ARG001
        midpoint = max(1, len(text) // 2)
        return [(0, midpoint), (midpoint, len(text))]

    monkeypatch.setattr(leaf_size_utils, "char_windows_from_token_budget", fake_windows)

    source_path = tmp_path / "per_manifesto.jsonl"
    output_dir = tmp_path / "joint_teacher"
    with source_path.open("w", encoding="utf-8") as handle:
        for idx in range(4):
            row = {
                "manifesto_id": f"joint_doc_{idx}",
                "text": _example_text(idx),
                "expert_dimension_scores_1_7": {
                    "economic": 4.5,
                    "social": 3.5,
                },
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    rc = cli.main(
        [
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
            "--dimensions",
            "economic,social",
            "--num-workers",
            "1",
            "--summary-mode",
            "identity",
        ]
    )

    assert rc == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["schema_version"] == "ctreepo.tree_bundle.v1"
    assert manifest["config"]["source_kind"] == "raw_input"
    assert manifest["config"]["leaf_unit"] == "text_token"
    assert manifest["config"]["state_contract"] == "raw_concat"
    assert manifest["config"]["reducer_contract"] == "bottom_up"
    assert manifest["config"]["tree_bundle_kind"] == "raw_manifesto_token_tree"
    assert manifest["config"]["tree_text_source"] == "aligned_text"
    assert manifest["config"]["finetune_export"]["enabled"] is True
    assert manifest["run_manifest"]["schema_version"] == "ctreepo.run_manifest.v1"
    assert manifest["run_manifest"]["role"] == "joint_teacher_tree_bundle"
    assert manifest["run_manifest"]["publication_ready"] is True

    summary = json.loads((output_dir / "leaf0016tok" / "summary.json").read_text(encoding="utf-8"))
    assert summary["source_kind"] == "raw_input"
    assert summary["tree_bundle_manifest"]["domain"] == "manifesto_rile"

    trees = load_labeled_trees(output_dir / "leaf0016tok" / "labeled_trees.jsonl")
    assert len(trees) == 4
    finetune = manifest["runs"]["tok_16"]["finetune"]
    assert finetune["bundle_kind"] == "manifesto_labeled_tree"
    assert finetune["summary"]["n_trees"] == 4
    assert finetune["counts"]["dataset"] > len(trees)
    assert Path(finetune["files"]["tree_records"]).exists()
    assert Path(finetune["finetune_adapters"]["adapters"]["embedding"]["files"]["embedding_ranked"]).exists()
    pref_dataset = json.loads(Path(finetune["files"]["dataset"]).read_text(encoding="utf-8"))
    root_f = [row for row in pref_dataset["units"] if row["unit_type"] == "root" and row["target"] == "f"]
    assert root_f
    first_candidate = next(
        row for row in pref_dataset["candidates"] if row["unit_id"] == root_f[0]["unit_id"]
    )
    measures = first_candidate["value"]["measures"]
    assert set(measures) >= {"economic", "social"}
    assert measures["economic"] == 4.0 / 6.0
    assert measures["social"] == 3.0 / 6.0
    assert trees[0].metadata["source_kind"] == "raw_input"
    assert trees[0].metadata["reducer_contract"] == "bottom_up"
    assert trees[0].metadata["tree_text_source"] == "aligned_text"
