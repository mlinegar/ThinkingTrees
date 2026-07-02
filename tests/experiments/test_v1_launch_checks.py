from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_v1_launch_checks_skip_tests_runs_fixture_gate(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out = subprocess.check_output(
        [
            sys.executable,
            "scripts/run_v1_launch_checks.py",
            "--output-dir",
            str(tmp_path),
            "--skip-tests",
            "--max-problems",
            "1",
            "--scorer-endpoint",
            "http://localhost:8010/v1",
            "--scorer-model",
            "test-scorer",
            "--json",
        ],
        cwd=repo_root,
        text=True,
    )
    payload = json.loads(out)

    assert payload["ok"] is True
    assert Path(payload["config"]) == tmp_path / "resolved_runtime_config.yaml"
    resolved = yaml.safe_load((tmp_path / "resolved_runtime_config.yaml").read_text(encoding="utf-8"))
    assert resolved["scorer"]["endpoint"] == "http://localhost:8010/v1"
    assert resolved["scorer"]["model"] == "test-scorer"
    assert resolved["summarizer"]["endpoint"] == "http://localhost:8010/v1"
    assert resolved["summarizer"]["model"] == "test-scorer"
    assert (tmp_path / "v1_launch_report.json").exists()
    assert (tmp_path / "experiment_manifest.json").exists()
    assert (tmp_path / "experiment_status.json").exists()
    assert (tmp_path / "results.jsonl").exists()
    runtime_experiment_dir = Path(payload["runtime_experiment_dir"])
    assert (runtime_experiment_dir / "metrics.json").exists()
    assert (runtime_experiment_dir / "calls.jsonl").exists()
    check_names = {item["name"] for item in payload["checks"]}
    assert "runtime_artifacts" in check_names
    assert "focused_pytest" not in check_names
