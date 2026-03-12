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
