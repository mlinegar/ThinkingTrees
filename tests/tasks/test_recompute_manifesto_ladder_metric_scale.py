from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_recompute_ladder_metric_scale_writes_companion_tree(tmp_path: Path) -> None:
    cli = importlib.import_module("scripts.recompute_manifesto_ladder_metric_scale")
    run_root = tmp_path / "run"
    leaf_dir = run_root / "ladder" / "dspy" / "leaf0256tok"
    history_path = leaf_dir / "iteration_history.json"
    source_results = tmp_path / "environment_results.jsonl"

    original_history = {
        "family": "dspy",
        "axis_kind": "leaf_size_tokens",
        "axis_value": 256,
        "leaf_count": None,
        "leaf_size_tokens": 256,
        "iterations": [
            {
                "iteration": 0,
                "stage_name": "f1g0",
                "stage_label": "f^1 g^0",
                "trained": "none",
                "split_metrics": {
                    "test": {
                        "n": 3,
                        "internal_f_pearson": 0.9,
                        "external_expert_pearson": 0.25,
                        "f_star_gap": 0.65,
                        "internal_f_mae_1_7": 0.1,
                        "external_expert_mae_1_7": 9.9,
                        "mean_prediction_1_7": 4.0,
                        "mean_teacher_1_7": 4.0,
                        "mean_expert_1_7": 5.0,
                    }
                },
            }
        ],
    }
    _write_json(history_path, original_history)
    _write_jsonl(
        leaf_dir / "prediction_records" / "iter_00_post_eval.jsonl",
        [
            {"doc_id": "doc_a", "split": "test", "prediction_1_7": 1.0},
            {"doc_id": "doc_b", "split": "test", "prediction_1_7": 4.0},
            {"doc_id": "doc_c", "split": "test", "prediction_1_7": 7.0},
        ],
    )
    _write_jsonl(
        source_results,
        [
            {"manifesto_id": "doc_a", "benoit_expert_mean": 0.0},
            {"manifesto_id": "doc_b", "benoit_expert_mean": 5.0},
            {"manifesto_id": "doc_c", "benoit_expert_mean": 10.0},
        ],
    )

    output_root = run_root / "scale_corrected" / "raw_benoit"
    rc = cli.main(
        [
            "--run-root",
            str(run_root),
            "--source-results",
            str(source_results),
            "--output-root",
            str(output_root),
            "--dimension",
            "environment",
        ]
    )
    assert rc == 0

    assert json.loads(history_path.read_text(encoding="utf-8")) == original_history

    corrected_history_path = output_root / "ladder" / "dspy" / "leaf0256tok" / "iteration_history.json"
    corrected = json.loads(corrected_history_path.read_text(encoding="utf-8"))
    iteration = corrected["iterations"][0]
    metrics = iteration["split_metrics"]["test"]
    assert iteration["stage_name"] == "f1g0"
    assert iteration["stage_label"] == "f^1 g^0"
    assert corrected["metrics_scale"] == "raw_benoit"
    assert metrics["metrics_scale"] == "raw_benoit"
    assert metrics["external_expert_pearson"] == pytest.approx(1.0)
    assert metrics["external_expert_mae"] == pytest.approx(0.0)
    assert metrics["external_expert_mae_1_7"] is None
    assert metrics["mean_prediction"] == pytest.approx(5.0)
    assert metrics["mean_expert"] == pytest.approx(5.0)
    assert metrics["mean_prediction_1_7"] is None
    assert metrics["mean_expert_1_7"] is None
    assert metrics["f_star_gap"] == pytest.approx(-0.1)

    root_summary = json.loads((output_root / "grid_summary.json").read_text(encoding="utf-8"))
    ladder_summary = json.loads((output_root / "ladder" / "grid_summary.json").read_text(encoding="utf-8"))
    assert root_summary["rows"] == ladder_summary["rows"]
    row = root_summary["rows"][0]
    assert row["stage_name"] == "f1g0"
    assert row["external_expert_mae"] == pytest.approx(0.0)
    assert row["external_expert_mae_1_7"] is None
    assert row["metrics_scale"] == "raw_benoit"
    assert root_summary["scale_correction"]["target_expert_scale"] == "raw_benoit"
    assert root_summary["scale_correction"]["prediction_transform"] == "scorer_1_7_to_raw_benoit"
    assert root_summary["scale_correction"]["counts"]["pairs_used"] == 3
