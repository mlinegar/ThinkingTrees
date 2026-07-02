from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_build_markov_capability_sanity_suite_writes_val_doc_commands(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    cmd_dir = tmp_path / "cmds"
    repo_root = Path(__file__).resolve().parents[2]

    subprocess.check_call(
        [
            sys.executable,
            "scripts/build_markov_capability_suite_cmds.py",
            "--suite",
            "sanity_suite",
            "--output-root",
            str(out_root),
            "--cmd-dir",
            str(cmd_dir),
            "--device",
            "cpu",
        ],
        cwd=repo_root,
    )

    manifest = json.loads((cmd_dir / "markov_capability_suite_manifest.json").read_text(encoding="utf-8"))
    cmd_file = Path(manifest["sanity_suite"]["cmd_file"])
    text = cmd_file.read_text(encoding="utf-8")
    assert "--val-docs 256" in text
    assert "--train-docs 128" in text
    assert "--train-docs 2048" in text


def test_build_markov_capability_mechanism_suite_reads_transition_summary(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    cmd_dir = tmp_path / "cmds"
    summary_path = tmp_path / "transition_summary.json"
    repo_root = Path(__file__).resolve().parents[2]

    summary_path.write_text(
        json.dumps(
            {
                "aggregated_rows": [
                    {
                        "n_regimes": 4,
                        "fixed_leaf_tokens": 16,
                        "train_docs": 512,
                        "val_docs": 128,
                        "test_docs": 256,
                        "audit_fraction": 0.1,
                        "root_weight": 1.0,
                        "state_dim": 64,
                        "hidden_dim": 256,
                        "n_epochs": 24,
                        "feature_mode": "full",
                        "selected_local_law_weight": 0.9,
                        "selected_lambda_sched": 0.2,
                        "full_success_rate": 0.5,
                        "dominant_capability_status": "failure",
                        "theorem_margin": -0.01,
                        "spread_margin": -0.02,
                        "root_margin": 0.01,
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    subprocess.check_call(
        [
            sys.executable,
            "scripts/build_markov_capability_suite_cmds.py",
            "--suite",
            "mechanism_suite",
            "--output-root",
            str(out_root),
            "--cmd-dir",
            str(cmd_dir),
            "--transition-summary",
            str(summary_path),
            "--device",
            "cpu",
        ],
        cwd=repo_root,
    )

    manifest = json.loads((cmd_dir / "markov_capability_suite_manifest.json").read_text(encoding="utf-8"))
    cmd_file = Path(manifest["mechanism_suite"]["cmd_file"])
    text = cmd_file.read_text(encoding="utf-8")
    assert "--root-weight 0.5" in text
    assert "--root-weight 4.0" in text
    assert "--local-law-weight 0.9" in text
    assert "--schedule-consistency-weight 0.2" in text
    assert "--val-docs 128" in text


def test_build_markov_capability_mechanism_suite_rejects_legacy_lambda_field(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    cmd_dir = tmp_path / "cmds"
    summary_path = tmp_path / "transition_summary.json"
    repo_root = Path(__file__).resolve().parents[2]

    summary_path.write_text(
        json.dumps(
            {
                "aggregated_rows": [
                    {
                        "n_regimes": 4,
                        "fixed_leaf_tokens": 16,
                        "train_docs": 512,
                        "val_docs": 128,
                        "test_docs": 256,
                        "audit_fraction": 0.1,
                        "root_weight": 1.0,
                        "state_dim": 64,
                        "hidden_dim": 256,
                        "n_epochs": 24,
                        "feature_mode": "full",
                        "selected_lambda_local": 0.9,
                        "selected_lambda_sched": 0.2,
                        "full_success_rate": 0.5,
                        "dominant_capability_status": "failure",
                        "theorem_margin": -0.01,
                        "spread_margin": -0.02,
                        "root_margin": 0.01,
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_markov_capability_suite_cmds.py",
            "--suite",
            "mechanism_suite",
            "--output-root",
            str(out_root),
            "--cmd-dir",
            str(cmd_dir),
            "--transition-summary",
            str(summary_path),
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "selected_lambda_local" in result.stderr
