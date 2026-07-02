from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_experiment_plan_infers_runtime_eval_adapter(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixture_config = tmp_path / "runtime.yaml"
    fixture_config.write_text(
        "\n".join(
            [
                "benchmark:",
                "  name: ruler_synthetic",
                "  family: runtime_benchmark",
                "scorer:",
                "  endpoint: http://localhost:8000/v1",
                "  model: mock-model",
                "runtime_defaults: {}",
                "phases:",
                "  - phase_id: P0",
                "    tasks: [vt]",
                "    lengths: [1024]",
                "    seeds: [0]",
                "    num_samples: 1",
                "    split: validation",
                "    methods: [full_context]",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "runtime_out"

    out = subprocess.check_output(
        [
            sys.executable,
            "scripts/run_experiment.py",
            "plan",
            "--",
            sys.executable,
            "scripts/run_runtime_eval.py",
            "init",
            "--config",
            str(fixture_config),
            "--output-dir",
            str(output_dir),
            "--experiment-id",
            "frontdoor",
        ],
        cwd=repo_root,
        text=True,
    )
    payload = json.loads(out)
    assert payload["adapter_id"] == "runtime_eval"
    assert payload["output_root"].endswith("runtime_out/frontdoor")
    assert (output_dir / "frontdoor" / "experiment_manifest.json").exists()


def test_run_experiment_plan_infers_supported_sidecar_script(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "lb.json"

    out = subprocess.check_output(
        [
            sys.executable,
            "scripts/run_experiment.py",
            "plan",
            "--summary",
            "--",
            sys.executable,
            "scripts/run_longbench_batched_example.py",
            "--dry-run",
            "--output",
            str(output),
        ],
        cwd=repo_root,
        text=True,
    )

    assert "Adapter:    runtime_umbrella_script" in out
    assert "longbench_v2" in out
    assert (output.parent / "lb_experiment" / "experiment_manifest.json").exists()


def test_run_experiment_list_shows_supported_entrypoints() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out = subprocess.check_output(
        [sys.executable, "scripts/run_experiment.py", "list", "--json"],
        cwd=repo_root,
        text=True,
    )
    payload = json.loads(out)
    assert "runtime_eval" in payload["adapters"]
    assert any(
        item["path"] == "scripts/run_longbench_batched_example.py"
        for item in payload["supported"]
    )
    assert "runtime_v1" in payload["report_profiles"]


def test_run_experiment_runtime_v1_report_profile(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    experiment_dir = tmp_path / "runtime_experiment"
    experiment_dir.mkdir()
    (experiment_dir / "metrics.json").write_text(
        json.dumps({"primary_metric": "longbench_v2_accuracy"}) + "\n",
        encoding="utf-8",
    )
    (experiment_dir / "config.json").write_text(
        json.dumps(
            {
                "experiment_id": "fixture",
                "benchmark": {"name": "longbench_v2"},
                "roles": {"scorer": {"model": "mock"}},
                "oracle": {"kind": "benchmark_labels"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (experiment_dir / "experiment_manifest.json").write_text(
        json.dumps({"experiment_id": "fixture-manifest"}) + "\n",
        encoding="utf-8",
    )
    (experiment_dir / "predictions.jsonl").write_text(
        json.dumps(
            {
                "phase_id": "S0",
                "task_id": "all",
                "max_seq_length": 4096,
                "method": "full_context",
                "primary_metric": "longbench_v2_accuracy",
                "metrics": {"longbench_v2_accuracy": 1.0},
                "cost": {"prompt_tokens": 10, "completion_tokens": 1},
                "metadata": {"problem": {"domain": "law", "difficulty": "easy", "length": "short"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (experiment_dir / "calls.jsonl").write_text(
        json.dumps({"method_id": "full_context", "role": "scorer", "surface": "chat_openai"})
        + "\n",
        encoding="utf-8",
    )
    report_dir = tmp_path / "report"

    out = subprocess.check_output(
        [
            sys.executable,
            "scripts/run_experiment.py",
            "report",
            "--profile",
            "runtime_v1",
            "--output-root",
            str(experiment_dir),
            "--report-output-dir",
            str(report_dir),
        ],
        cwd=repo_root,
        text=True,
    )
    payload = json.loads(out)

    assert payload["profile"] == "runtime_v1"
    assert payload["n_predictions"] == 1
    assert payload["calls_by_role"] == {"scorer": 1}
    assert (report_dir / "runtime_v1_summary.json").exists()
    assert (report_dir / "runtime_v1_summary.md").exists()


def test_run_experiment_check_v1_runs_gate_and_report(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out = subprocess.check_output(
        [
            sys.executable,
            "scripts/run_experiment.py",
            "check",
            "--suite",
            "v1",
            "--output-dir",
            str(tmp_path),
            "--skip-tests",
            "--max-problems",
            "1",
            "--scorer-endpoint",
            "http://localhost:8010/v1",
            "--scorer-model",
            "test-scorer",
            "--report",
            "--json",
        ],
        cwd=repo_root,
        text=True,
    )
    payload = json.loads(out)

    assert payload["ok"] is True
    assert payload["check"]["ok"] is True
    assert payload["report"]["profile"] == "runtime_v1"
    assert (tmp_path / "v1_launch_report.json").exists()
    assert (tmp_path / "paper_report" / "runtime_v1_summary.json").exists()


def test_run_experiment_collect_refreshes_status_from_manifest(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "lb.json"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_experiment.py",
            "plan",
            "--",
            sys.executable,
            "scripts/run_longbench_batched_example.py",
            "--dry-run",
            "--output",
            str(output),
        ],
        cwd=repo_root,
    )
    manifest = output.parent / "lb_experiment" / "experiment_manifest.json"

    out = subprocess.check_output(
        [
            sys.executable,
            "scripts/run_experiment.py",
            "collect",
            "--manifest",
            str(manifest),
        ],
        cwd=repo_root,
        text=True,
    )
    payload = json.loads(out)

    assert payload["state"] == "collected"
    assert (output.parent / "lb_experiment" / "experiment_status.json").exists()


def test_run_experiment_run_collects_after_foreground_command(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "markov_demo.json"

    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_experiment.py",
            "launch",
            "--",
            sys.executable,
            "scripts/run_treepo_stack_markov_demo.py",
            "--path",
            "a a b b",
            "--output-json",
            str(output),
        ],
        cwd=repo_root,
    )

    sidecar_root = output.parent / "markov_demo_experiment"
    status = json.loads((sidecar_root / "experiment_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    assert (sidecar_root / "artifacts.json").exists()
