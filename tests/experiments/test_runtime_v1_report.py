from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def test_report_runtime_v1_results_builds_method_matrix(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    experiment_dir = tmp_path / "runtime_experiment"
    experiment_dir.mkdir()
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": "r1",
            "benchmark": {"name": "longbench_v2"},
            "roles": {"scorer": {"model": "mock", "surface": "chat_openai"}},
            "oracle": {"kind": "benchmark_labels"},
        },
    )
    _write_json(experiment_dir / "metrics.json", {"n_predictions": 3, "n_surface_calls": 2})
    _write_json(experiment_dir / "experiment_manifest.json", {"experiment_id": "e1"})
    for method, score, domain in [
        ("full_context", 1.0, "law"),
        ("full_context", 0.0, "finance"),
        ("retrieval", 1.0, "law"),
    ]:
        _append_jsonl(
            experiment_dir / "predictions.jsonl",
            {
                "phase_id": "S0",
                "task_id": "all",
                "max_seq_length": 4096,
                "method": method,
                "primary_metric": "longbench_v2_accuracy",
                "metrics": {"longbench_v2_accuracy": score},
                "cost": {"prompt_tokens": 10, "completion_tokens": 1},
                "metadata": {
                    "problem": {
                        "domain": domain,
                        "difficulty": "easy",
                        "length": "short",
                    }
                },
            },
        )
    _append_jsonl(
        experiment_dir / "calls.jsonl",
        {"method_id": "full_context", "role": "scorer", "surface": "chat_openai"},
    )
    _append_jsonl(
        experiment_dir / "calls.jsonl",
        {"method_id": "retrieval", "role": "embedder", "surface": "embedding"},
    )

    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_runtime_v1_results.py",
            "--experiment-dir",
            str(experiment_dir),
        ],
        cwd=repo_root,
    )

    summary_path = experiment_dir / "paper_summary" / "runtime_v1_summary.json"
    md_path = experiment_dir / "paper_summary" / "runtime_v1_summary.md"
    assert summary_path.exists()
    assert md_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = {row["method_id"]: row for row in summary["method_rows"]}
    assert rows["full_context"]["mean_score"] == 0.5
    assert rows["retrieval"]["mean_score"] == 1.0
    assert summary["calls_by_role"] == {"embedder": 1, "scorer": 1}
    assert summary["experiment_id"] == "r1"
    assert summary["experiment_dir"] == str(experiment_dir.resolve())
    artifacts = json.loads((experiment_dir / "artifacts.json").read_text(encoding="utf-8"))
    assert "runtime_v1_summary_json" in artifacts["artifacts"]
    assert "Full context" in md_path.read_text(encoding="utf-8")
