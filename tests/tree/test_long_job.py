from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time


def _read_json(text: str) -> dict:
    payload = json.loads(text)
    assert isinstance(payload, dict)
    return payload


def test_long_job_launch_status_and_stop(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    job_root = tmp_path / "job"
    log_path = job_root / "job.log"

    launch = _read_json(
        subprocess.check_output(
            [
                sys.executable,
                "scripts/long_job.py",
                "launch",
                "--name",
                "smoke_job",
                "--job-root",
                str(job_root),
                "--launch-backend",
                "double_fork",
                "--cwd",
                str(repo_root),
                "--",
                sys.executable,
                "-c",
                "import time; print('launcher_smoke', flush=True); time.sleep(30)",
            ],
            cwd=repo_root,
            text=True,
        )
    )
    manifest_path = Path(launch["manifest_path"])
    assert manifest_path.exists()
    assert log_path.exists()
    assert int(launch["pid"]) > 0
    assert int(launch["pgid"]) > 0
    assert launch["launch_backend"] == "double_fork"
    assert float(launch["progress_refresh_interval_seconds"]) == 30.0
    runner_text = Path(launch["runner_script"]).read_text(encoding="utf-8")
    assert "long_job.py" in runner_text
    assert " refresh" in runner_text
    assert str(manifest_path) in runner_text

    deadline = time.time() + 5.0
    while time.time() < deadline:
        text = log_path.read_text(encoding="utf-8")
        if "launcher_smoke" in text:
            break
        time.sleep(0.1)
    assert "launcher_smoke" in log_path.read_text(encoding="utf-8")

    status = _read_json(
        subprocess.check_output(
            [
                sys.executable,
                "scripts/long_job.py",
                "status",
                "--manifest",
                str(manifest_path),
                "--tail-lines",
                "5",
            ],
            cwd=repo_root,
            text=True,
        )
    )
    assert status["running"] is True
    assert status["pid"] == launch["pid"]
    assert any("launcher_smoke" in line for line in status["tail"])

    stop = _read_json(
        subprocess.check_output(
            [
                sys.executable,
                "scripts/long_job.py",
                "stop",
                "--manifest",
                str(manifest_path),
                "--force-kill",
            ],
            cwd=repo_root,
            text=True,
        )
    )
    assert stop["running_after_stop"] is False

    final_status = _read_json(
        subprocess.check_output(
            [
                sys.executable,
                "scripts/long_job.py",
                "status",
                "--manifest",
                str(manifest_path),
                "--tail-lines",
                "0",
            ],
            cwd=repo_root,
            text=True,
        )
    )
    assert final_status["running"] is False


def test_long_job_refresh_writes_combined_status(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "output"
    recoverable_root = output_root / "package_capacity" / "recoverable" / "full20"
    recoverable_root.mkdir(parents=True, exist_ok=True)
    (recoverable_root / "scheduler_status.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-04-01T06:42:20+00:00",
                "state": "completed",
                "items_total": 82,
                "completed_items": 98,
                "failed_items": 0,
                "active_items": 0,
                "pending_items": 0,
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest_refresh.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "refresh_job",
                "job_root": str(tmp_path),
                "pid": 0,
                "pgid": 0,
                "launched_at": "2026-04-01T00:00:00+00:00",
                "log_path": str(tmp_path / "job.log"),
                "cwd": str(repo_root),
                "command": [
                    "/bin/bash",
                    "-lc",
                    (
                        f"set -euo pipefail\nOUTPUT_ROOT=\"{output_root}\"\n"
                        "packages=(\"full20\" \"full20_leaf_count10_internal_count10\")\n"
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    subprocess.check_call(
        [
            sys.executable,
            "scripts/long_job.py",
            "refresh",
            "--manifest",
            str(manifest_path),
        ],
        cwd=repo_root,
    )

    combined_path = output_root / "combined_scheduler_status.json"
    assert combined_path.exists()
    payload = _read_json(combined_path.read_text(encoding="utf-8"))
    assert payload["status_kind"] == "combined_scheduler_progress"
    assert "structural/full20" in payload["phase_progress"]
    assert payload["phase_progress"]["recoverable/full20"]["items_total"] == 98


def test_long_job_status_formats_progress_for_display(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "scheduler_status.json").write_text(
        json.dumps(
            {
                "by_train_docs": {
                    "10240": {"percent_complete": 12.5, "epochs_total": 15},
                    "2048": {"percent_complete": 25.0, "epochs_total": 15},
                    "1024": {"percent_complete": 50.0, "epochs_total": 15},
                },
                "percent_complete": 33.3333,
                "phase_progress": {
                    "supervision_recovery": {
                        "percent_complete": 40.0,
                        "epoch_percent": 12.5,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "display_job",
                "job_root": str(tmp_path),
                "pid": 0,
                "pgid": 0,
                "launched_at": "2026-03-27T00:00:00+00:00",
                "log_path": str(tmp_path / "job.log"),
                "command": [
                    sys.executable,
                    "scripts/run_markov_optimization_tradeoff_pipeline.py",
                    "--output-root",
                    str(output_root),
                ],
            }
        ),
        encoding="utf-8",
    )

    status = _read_json(
        subprocess.check_output(
            [
                sys.executable,
                "scripts/long_job.py",
                "status",
                "--manifest",
                str(manifest_path),
                "--tail-lines",
                "0",
            ],
            cwd=repo_root,
            text=True,
        )
    )

    assert list(status["progress"]["by_train_docs"]) == ["1024", "2048", "10240"]
    assert status["progress"]["percent_complete"] == "33.3%"
    assert status["progress"]["phase_progress"]["supervision_recovery"]["percent_complete"] == "40.0%"
    assert status["progress"]["phase_progress"]["supervision_recovery"]["epoch_percent"] == "12.5%"


def test_long_job_status_refreshes_live_active_progress(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "output"
    task_root = output_root / "supervision_recovery" / "raw" / "task_a"
    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / "progress.json").write_text(
        json.dumps(
            {
                "state": "running",
                "epoch_completed": 7,
                "epochs_total": 52,
                "stage": "stage1_train",
            }
        ),
        encoding="utf-8",
    )
    (output_root / "scheduler_status.json").write_text(
        json.dumps(
            {
                "phase_progress": {
                    "supervision_recovery": {
                        "total": 1,
                        "completed": 0,
                        "active": 1,
                        "pending": 0,
                        "failed": 0,
                        "epochs_completed": 0,
                        "epochs_total": 52,
                        "epoch_percent": 0.0,
                    }
                },
                "by_scope": {
                    "recoverable_v4": {
                        "total": 1,
                        "completed": 0,
                        "active": 1,
                        "pending": 0,
                        "failed": 0,
                        "epochs_completed": 0,
                        "epochs_total": 52,
                        "epoch_percent": 0.0,
                    }
                },
                "by_train_docs": {
                    "1024": {
                        "total": 1,
                        "completed": 0,
                        "active": 1,
                        "pending": 0,
                        "failed": 0,
                        "epochs_completed": 0,
                        "epochs_total": 52,
                        "epoch_percent": 0.0,
                    }
                },
                "by_model_family": {
                    "tree_neural": {
                        "total": 1,
                        "completed": 0,
                        "active": 1,
                        "pending": 0,
                        "failed": 0,
                        "epochs_completed": 0,
                        "epochs_total": 52,
                        "epoch_percent": 0.0,
                    }
                },
                "by_package": {
                    "full100": {
                        "total": 1,
                        "completed": 0,
                        "active": 1,
                        "pending": 0,
                        "failed": 0,
                        "epochs_completed": 0,
                        "epochs_total": 52,
                        "epoch_percent": 0.0,
                    }
                },
                "by_worker_kind": {
                    "full_doc_diagnostics": {
                        "total": 1,
                        "completed": 0,
                        "active": 1,
                        "pending": 0,
                        "failed": 0,
                        "epochs_completed": 0,
                        "epochs_total": 52,
                        "epoch_percent": 0.0,
                    }
                },
                "active_item_details": [
                    {
                        "item_id": "supervision_recovery::task_a",
                        "phase": "supervision_recovery",
                        "log_path": str(task_root / "run.log"),
                        "scope": "recoverable_v4",
                        "train_docs": 1024,
                        "model_family": "tree_neural",
                        "package": "full100",
                        "worker_kind": "full_doc_diagnostics",
                        "progress": {
                            "state": "running",
                            "epoch_completed": 0,
                            "epochs_total": 52,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "display_job",
                "job_root": str(tmp_path),
                "pid": 0,
                "pgid": 0,
                "launched_at": "2026-03-27T00:00:00+00:00",
                "log_path": str(tmp_path / "job.log"),
                "command": [
                    sys.executable,
                    "scripts/run_markov_optimization_tradeoff_pipeline.py",
                    "--output-root",
                    str(output_root),
                ],
            }
        ),
        encoding="utf-8",
    )

    status = _read_json(
        subprocess.check_output(
            [
                sys.executable,
                "scripts/long_job.py",
                "status",
                "--manifest",
                str(manifest_path),
                "--tail-lines",
                "0",
            ],
            cwd=repo_root,
            text=True,
        )
    )

    assert status["progress"]["phase_progress"]["supervision_recovery"]["epochs_completed"] == 7
    assert status["progress"]["phase_progress"]["supervision_recovery"]["epoch_percent"] == "13.5%"
    assert status["progress"]["active_item_details"][0]["progress"]["epoch_completed"] == 7
    assert status["progress"]["by_model_family"]["tree_neural"]["epochs_completed"] == 7


def test_long_job_status_falls_back_to_nested_scheduler_status(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "output"
    nested_root = output_root / "package_capacity" / "structural" / "full20"
    nested_root.mkdir(parents=True, exist_ok=True)
    (nested_root / "scheduler_status.json").write_text(
        json.dumps(
            {
                "state": "running",
                "percent_complete": 37.5,
                "progress_bar": "########------------",
                "phase_progress": {
                    "capacity": {
                        "percent_complete": 37.5,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "nested_progress_job",
                "job_root": str(tmp_path),
                "pid": 0,
                "pgid": 0,
                "launched_at": "2026-04-01T00:00:00+00:00",
                "log_path": str(tmp_path / "job.log"),
                "command": [
                    sys.executable,
                    "scripts/run_markov_optimization_tradeoff_pipeline.py",
                    "--output-root",
                    str(output_root),
                ],
            }
        ),
        encoding="utf-8",
    )

    status = _read_json(
        subprocess.check_output(
            [
                sys.executable,
                "scripts/long_job.py",
                "status",
                "--manifest",
                str(manifest_path),
                "--tail-lines",
                "0",
            ],
            cwd=repo_root,
            text=True,
        )
    )

    assert status["progress"]["percent_complete"] == "37.5%"
    assert status["progress"]["status_path"] == str(nested_root / "scheduler_status.json")


def test_long_job_status_reads_output_root_from_shell_wrapper(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "wrapped_output"
    nested_root = output_root / "package_capacity" / "structural" / "full20"
    nested_root.mkdir(parents=True, exist_ok=True)
    (nested_root / "scheduler_status.json").write_text(
        json.dumps(
            {
                "state": "running",
                "percent_complete": 12.5,
                "progress_bar": "##------------------",
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest_shell.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "wrapped_job",
                "job_root": str(tmp_path),
                "pid": 0,
                "pgid": 0,
                "launched_at": "2026-04-01T00:00:00+00:00",
                "log_path": str(tmp_path / "job.log"),
                "command": [
                    "/bin/bash",
                    "-lc",
                    f"set -euo pipefail\nOUTPUT_ROOT=\"{output_root}\"\necho ready\n",
                ],
            }
        ),
        encoding="utf-8",
    )

    status = _read_json(
        subprocess.check_output(
            [
                sys.executable,
                "scripts/long_job.py",
                "status",
                "--manifest",
                str(manifest_path),
                "--tail-lines",
                "0",
            ],
            cwd=repo_root,
            text=True,
        )
    )

    assert status["progress"]["percent_complete"] == "12.5%"
    assert status["progress"]["status_path"] == str(nested_root / "scheduler_status.json")


def test_long_job_status_aggregates_multiple_nested_schedulers(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "output"
    recoverable_root = output_root / "package_capacity" / "recoverable" / "full20"
    structural_root = output_root / "package_capacity" / "structural" / "full20"
    recoverable_root.mkdir(parents=True, exist_ok=True)
    structural_root.mkdir(parents=True, exist_ok=True)
    (recoverable_root / "scheduler_status.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-04-01T06:42:20+00:00",
                "state": "completed",
                "items_total": 82,
                "completed_items": 98,
                "failed_items": 0,
                "active_items": 0,
                "pending_items": 0,
            }
        ),
        encoding="utf-8",
    )
    (structural_root / "scheduler_status.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-04-01T19:26:14+00:00",
                "state": "running",
                "items_total": 82,
                "completed_items": 16,
                "failed_items": 0,
                "active_items": 16,
                "pending_items": 50,
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest_shell.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "wrapped_job",
                "job_root": str(tmp_path),
                "pid": 0,
                "pgid": 0,
                "launched_at": "2026-04-01T00:00:00+00:00",
                "log_path": str(tmp_path / "job.log"),
                "command": [
                    "/bin/bash",
                    "-lc",
                    f"set -euo pipefail\nOUTPUT_ROOT=\"{output_root}\"\necho ready\n",
                ],
            }
        ),
        encoding="utf-8",
    )

    status = _read_json(
        subprocess.check_output(
            [
                sys.executable,
                "scripts/long_job.py",
                "status",
                "--manifest",
                str(manifest_path),
                "--tail-lines",
                "0",
            ],
            cwd=repo_root,
            text=True,
        )
    )

    combined_path = output_root / "combined_scheduler_status.json"
    assert status["progress"]["state"] == "running"
    assert status["progress"]["items_total"] == 196
    assert status["progress"]["completed_items"] == 114
    assert status["progress"]["active_items"] == 16
    assert status["progress"]["pending_items"] == 66
    assert status["progress"]["status_path"] == str(combined_path)
    assert combined_path.exists()
    assert "package_capacity/recoverable/full20" not in status["progress"]["phase_progress"]
    assert "package_capacity/structural/full20" not in status["progress"]["phase_progress"]
    assert "package_capacity" not in status["progress"]["phase_progress"]
    assert "recoverable/full20" in status["progress"]["phase_progress"]
    assert "structural/full20" in status["progress"]["phase_progress"]
    assert status["progress"]["phase_progress"]["structural/full20"]["items_total"] == 98
    assert status["progress"]["phase_progress"]["structural/full20"]["observed_items_total"] == 82
    assert status["progress"]["phase_progress"]["structural/full20"]["items_total_source"] == "observed_global_total"


def test_long_job_status_includes_planned_package_capacity_children(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "output"
    recoverable_root = output_root / "package_capacity" / "recoverable" / "full20"
    recoverable_root.mkdir(parents=True, exist_ok=True)
    (recoverable_root / "scheduler_status.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-04-01T06:42:20+00:00",
                "state": "completed",
                "items_total": 82,
                "completed_items": 98,
                "failed_items": 0,
                "active_items": 0,
                "pending_items": 0,
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest_planned.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "planned_job",
                "job_root": str(tmp_path),
                "pid": 0,
                "pgid": 0,
                "launched_at": "2026-04-01T00:00:00+00:00",
                "log_path": str(tmp_path / "job.log"),
                "command": [
                    "/bin/bash",
                    "-lc",
                    (
                        f"set -euo pipefail\nOUTPUT_ROOT=\"{output_root}\"\n"
                        "packages=(\"full20\" \"full20_leaf_count10_internal_count10\")\n"
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    status = _read_json(
        subprocess.check_output(
            [
                sys.executable,
                "scripts/long_job.py",
                "status",
                "--manifest",
                str(manifest_path),
                "--tail-lines",
                "0",
            ],
            cwd=repo_root,
            text=True,
        )
    )

    progress = status["progress"]
    assert progress["state"] == "running"
    assert "recoverable/full20" in progress["phase_progress"]
    assert "structural/full20" in progress["phase_progress"]
    assert "recoverable/full20_leaf_count10_internal_count10" in progress["phase_progress"]
    assert "structural/full20_leaf_count10_internal_count10" in progress["phase_progress"]
    assert progress["phase_progress"]["structural/full20"]["state"] == "planned"
    assert progress["phase_progress"]["structural/full20"]["expected_status_path"].endswith(
        "package_capacity/structural/full20/scheduler_status.json"
    )
    assert progress["phase_progress"]["recoverable/full20_leaf_count10_internal_count10"]["pending_items"] == 98


def test_long_job_status_prefers_experiment_status_json(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "experiment_status.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp123",
                "state": "running",
                "active_phase": "capacity",
                "items_total": 10,
                "completed_items": 6,
                "failed_items": 0,
                "active_items": 2,
                "pending_items": 2,
                "percent_complete": 60.0,
            }
        ),
        encoding="utf-8",
    )
    (output_root / "scheduler_status.json").write_text(
        json.dumps({"state": "failed", "percent_complete": 5.0}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "canonical_status_job",
                "job_root": str(tmp_path),
                "pid": 0,
                "pgid": 0,
                "launched_at": "2026-04-01T00:00:00+00:00",
                "log_path": str(tmp_path / "job.log"),
                "command": [
                    sys.executable,
                    "scripts/run_markov_optimization_tradeoff_pipeline.py",
                    "--output-root",
                    str(output_root),
                ],
            }
        ),
        encoding="utf-8",
    )

    status = _read_json(
        subprocess.check_output(
            [
                sys.executable,
                "scripts/long_job.py",
                "status",
                "--manifest",
                str(manifest_path),
                "--tail-lines",
                "0",
            ],
            cwd=repo_root,
            text=True,
        )
    )

    assert status["progress"]["state"] == "running"
    assert status["progress"]["active_phase"] == "capacity"
    assert status["progress"]["percent_complete"] == "60.0%"
