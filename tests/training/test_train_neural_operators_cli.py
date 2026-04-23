from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[2]
    mod_path = root / "scripts" / "train_neural_operators.py"
    spec = importlib.util.spec_from_file_location("train_neural_operators", str(mod_path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_neural_operators_wires_explicit_ctreepo_local_law_args(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    recorded: list[list[str]] = []

    def _fake_run_command(label: str, cmd: list[str], log_path: Path):
        recorded.append(cmd)
        return {
            "label": label,
            "returncode": 0,
            "log": str(log_path),
            "started_at": "2026-03-07T00:00:00",
            "ended_at": "2026-03-07T00:00:01",
        }

    monkeypatch.setattr(mod, "_run_command", _fake_run_command)
    monkeypatch.setattr(mod, "_detect_artifacts", lambda label, run_dir: {"primary_model_path": None})
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "train_neural_operators.py",
            "--output-dir",
            str(tmp_path),
            "--which",
            "ctreepo",
            "--ctreepo-args",
            "--pilot --root-weight 9.0",
            "--ctreepo-root-weight",
            "1.25",
            "--ctreepo-leaf-audit-weight",
            "0.3",
            "--ctreepo-merge-audit-weight",
            "0.8",
            "--ctreepo-local-law-violation-threshold",
            "7.5",
            "--ctreepo-local-law-oracle",
            "tests.training.fake_oracle:score_span",
            "--ctreepo-require-local-law-supervision",
        ],
    )

    rc = int(mod.main())
    assert rc == 0
    assert len(recorded) == 1

    cmd = recorded[0]
    assert cmd.count("--root-weight") == 2
    assert cmd[cmd.index("--root-weight") + 1] == "9.0"
    assert cmd[cmd.index("--root-weight", cmd.index("--root-weight") + 1) + 1] == "1.25"
    assert cmd[-11:] == [
        "--root-weight",
        "1.25",
        "--leaf-audit-weight",
        "0.3",
        "--merge-audit-weight",
        "0.8",
        "--local-law-violation-threshold",
        "7.5",
        "--local-law-oracle",
        "tests.training.fake_oracle:score_span",
        "--require-local-law-supervision",
    ]
    assert cmd[-1] == "--require-local-law-supervision"

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["ctreepo_local_law"]["root_weight"] == 1.25
    assert summary["ctreepo_local_law"]["leaf_audit_weight"] == 0.3
    assert summary["ctreepo_local_law"]["merge_audit_weight"] == 0.8
    assert summary["ctreepo_local_law"]["oracle_module"] == "tests.training.fake_oracle:score_span"
    assert summary["ctreepo_local_law"]["label_source_kind"] == "oracle_callback"
    assert summary["ctreepo_local_law"]["require_supervision"] is True


def test_train_neural_operators_wires_task_oracle_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    recorded: list[list[str]] = []

    def _fake_run_command(label: str, cmd: list[str], log_path: Path):
        recorded.append(cmd)
        return {
            "label": label,
            "returncode": 0,
            "log": str(log_path),
            "started_at": "2026-03-07T00:00:00",
            "ended_at": "2026-03-07T00:00:01",
        }

    monkeypatch.setattr(mod, "_run_command", _fake_run_command)
    monkeypatch.setattr(mod, "_detect_artifacts", lambda label, run_dir: {"primary_model_path": None})
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "train_neural_operators.py",
            "--output-dir",
            str(tmp_path),
            "--which",
            "ctreepo",
            "--ctreepo-local-law-oracle",
            "task",
        ],
    )

    rc = int(mod.main())
    assert rc == 0
    cmd = recorded[0]
    assert "--task" in cmd
    assert "--local-law-oracle" in cmd
    assert cmd[cmd.index("--local-law-oracle") + 1] == "task"

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["task"] == "manifesto_rile"
    assert summary["ctreepo_local_law"]["oracle_module"] == "task"
    assert summary["ctreepo_local_law"]["label_source_kind"] == "task_oracle"


def test_train_neural_operators_wires_teacher_labeling_flags(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    recorded: list[list[str]] = []

    def _fake_run_command(label: str, cmd: list[str], log_path: Path):
        recorded.append(cmd)
        return {
            "label": label,
            "returncode": 0,
            "log": str(log_path),
            "started_at": "2026-03-07T00:00:00",
            "ended_at": "2026-03-07T00:00:01",
        }

    monkeypatch.setattr(mod, "_run_command", _fake_run_command)
    monkeypatch.setattr(mod, "_detect_artifacts", lambda label, run_dir: {"primary_model_path": None})
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "train_neural_operators.py",
            "--output-dir",
            str(tmp_path),
            "--which",
            "ctreepo",
            "--ctreepo-local-law-teacher-port",
            "8001",
            "--ctreepo-local-law-teacher-model",
            "teacher-model",
            "--ctreepo-local-law-teacher-max-tokens",
            "96",
            "--ctreepo-local-law-teacher-temperature",
            "0.1",
            "--ctreepo-allow-model-based-local-law-labeling",
        ],
    )

    rc = int(mod.main())
    assert rc == 0
    cmd = recorded[0]
    assert "--local-law-teacher-port" in cmd
    assert "--local-law-teacher-model" in cmd
    assert "--local-law-teacher-max-tokens" in cmd
    assert "--local-law-teacher-temperature" in cmd
    assert "--allow-model-based-local-law-labeling" in cmd

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["ctreepo_local_law"]["label_source_kind"] == "model_backed_teacher"
    assert summary["ctreepo_local_law"]["teacher_port"] == 8001
    assert summary["ctreepo_local_law"]["score_port"] == 8001
    assert summary["ctreepo_local_law"]["allow_model_based_labeling"] is True
    assert summary["ctreepo_local_law"]["allow_model_based_scoring"] is True


def test_detect_artifacts_reads_ctreepo_local_law_summary(tmp_path: Path) -> None:
    mod = _load_module()
    run_dir = tmp_path / "ctreepo"
    run_dir.mkdir()
    (run_dir / "best.pt").write_text("stub", encoding="utf-8")
    (run_dir / "reproducibility_manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "training_result.json").write_text(
        json.dumps(
            {
                "best_epoch": 0,
                "local_law_summary": {
                    "node_oracle_predictor_attached": True,
                    "node_label_source_kind": "oracle_callback",
                    "labeled_leaves": 4,
                    "labeled_internal": 2,
                    "compositional_learning_problem": {
                        "name": "ctreepo_local_law_training",
                    },
                },
                "compositional_learning_problem": {
                    "name": "ctreepo_local_law_training",
                    "uses_sampled_substructure_labels": True,
                },
            }
        ),
        encoding="utf-8",
    )

    artifacts = mod._detect_artifacts("ctreepo", run_dir)

    assert artifacts["best_model_path"] == str(run_dir / "best.pt")
    assert artifacts["training_result_path"] == str(run_dir / "training_result.json")
    assert artifacts["reproducibility_manifest_path"] == str(run_dir / "reproducibility_manifest.json")
    assert artifacts["local_law_summary"] == {
        "node_oracle_predictor_attached": True,
        "node_label_source_kind": "oracle_callback",
        "labeled_leaves": 4,
        "labeled_internal": 2,
        "compositional_learning_problem": {
            "name": "ctreepo_local_law_training",
        },
    }
    assert artifacts["compositional_learning_problem"] == {
        "name": "ctreepo_local_law_training",
        "uses_sampled_substructure_labels": True,
    }


def test_train_neural_operators_writes_canonical_control_plane(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()

    def _fake_run_command(label: str, cmd: list[str], log_path: Path):
        run_dir = tmp_path / label
        run_dir.mkdir(parents=True, exist_ok=True)
        if label == "ctreepo":
            (run_dir / "best.pt").write_text("stub", encoding="utf-8")
            (run_dir / "training_result.json").write_text(
                json.dumps(
                    {
                        "best_epoch": 3,
                        "best_root_mae": 0.12,
                        "training_time_seconds": 5.0,
                        "epochs_completed": 4,
                        "eval_metrics": [
                            {
                                "root_mae": 0.12,
                                "leaf_oracle_mae": 0.04,
                                "merge_oracle_mae": 0.06,
                                "leaf_violation_rate": 0.1,
                                "merge_violation_rate": 0.2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
        return {
            "label": label,
            "returncode": 0,
            "log": str(log_path),
            "started_at": "2026-03-07T00:00:00",
            "ended_at": "2026-03-07T00:00:01",
        }

    monkeypatch.setattr(mod, "_run_command", _fake_run_command)
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "train_neural_operators.py",
            "--output-dir",
            str(tmp_path),
            "--which",
            "ctreepo",
            "--ctreepo-local-law-oracle",
            "task",
            "--ctreepo-leaf-audit-weight",
            "0.3",
            "--ctreepo-merge-audit-weight",
            "0.8",
        ],
    )

    rc = int(mod.main())
    assert rc == 0
    assert (tmp_path / "experiment_manifest.json").exists()
    assert (tmp_path / "experiment_status.json").exists()
    assert (tmp_path / "artifacts.json").exists()
    assert (tmp_path / "reproducibility_manifest.json").exists()
    assert (tmp_path / "search_spec.json").exists()
    assert (tmp_path / "search_results.json").exists()
    assert (tmp_path / "results.jsonl").exists()
    status = json.loads((tmp_path / "experiment_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    artifacts = json.loads((tmp_path / "artifacts.json").read_text(encoding="utf-8"))
    assert "reproducibility_manifest_json" in artifacts["artifacts"]
    assert "search_spec_json" in artifacts["artifacts"]
    assert "search_results_json" in artifacts["artifacts"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row["metric_name"] == "best_root_mae" for row in rows)
    assert any(row["metric_name"] == "leaf_violation_rate" for row in rows)


def test_train_neural_operators_runs_search_trials_and_selects_best(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    spec_path = tmp_path / "ctreepo_search.json"
    spec_path.write_text(
        json.dumps(
            {
                "mode": "grid",
                "selection_metric": "validation_mae",
                "tie_breaker_metric": "training_time_seconds",
                "dimensions": [
                    {"flag": "--lr", "values": [0.001, 0.0005]},
                ],
            }
        ),
        encoding="utf-8",
    )

    def _fake_run_command(label: str, cmd: list[str], log_path: Path):
        run_dir = Path(cmd[cmd.index("--output-dir") + 1])
        run_dir.mkdir(parents=True, exist_ok=True)
        lr = float(cmd[cmd.index("--lr") + 1]) if "--lr" in cmd else 0.001
        if label == "ctreepo":
            (run_dir / "best.pt").write_text("stub", encoding="utf-8")
            (run_dir / "training_result.json").write_text(
                json.dumps(
                    {
                        "best_epoch": 1,
                        "best_root_mae": 0.30 if lr >= 0.001 else 0.10,
                        "training_time_seconds": 9.0 if lr >= 0.001 else 5.0,
                        "epochs_completed": 2,
                        "eval_metrics": [
                            {
                                "root_mae": 0.30 if lr >= 0.001 else 0.10,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
        return {
            "label": label,
            "returncode": 0,
            "log": str(log_path),
            "started_at": "2026-03-07T00:00:00",
            "ended_at": "2026-03-07T00:00:01",
        }

    monkeypatch.setattr(mod, "_run_command", _fake_run_command)
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "train_neural_operators.py",
            "--output-dir",
            str(tmp_path),
            "--which",
            "ctreepo",
            "--ctreepo-search-spec",
            str(spec_path),
        ],
    )

    rc = int(mod.main())

    assert rc == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    ctreepo_search = summary["search"]["methods"]["ctreepo"]
    assert ctreepo_search["search_enabled"] is True
    assert ctreepo_search["selected_trial_id"] == "trial_001"
    assert len(ctreepo_search["trials"]) == 2
    assert summary["runs"][0]["trial_id"] == "trial_001"
    assert summary["runs"][0]["run_dir"].endswith("ctreepo/trials/trial_001")
    search_results = json.loads((tmp_path / "search_results.json").read_text(encoding="utf-8"))
    assert search_results["methods"]["ctreepo"]["selected_trial_id"] == "trial_001"
