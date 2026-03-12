from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.ctreepo.sim.manifest import read_manifest_jsonl


def test_build_markov_law_stress_sanity_suite_writes_packages_and_exact_families(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "out"
    cmd_dir = tmp_path / "cmds"

    subprocess.check_call(
        [
            sys.executable,
            "scripts/build_markov_law_stress_suite_cmds.py",
            "--suite",
            "sanity_suite",
            "--output-root",
            str(output_root),
            "--cmd-dir",
            str(cmd_dir),
            "--smoke",
        ],
        cwd=repo_root,
    )

    manifest = json.loads((cmd_dir / "markov_law_stress_suite_manifest.json").read_text(encoding="utf-8"))
    runs = read_manifest_jsonl(Path(manifest["runspec_manifest"]))
    assert runs
    learned_cmd = Path(manifest["sanity_suite"]["learned_cmd_file"])
    exact_cmd = Path(manifest["sanity_suite"]["exact_cmd_file"])
    learned_text = learned_cmd.read_text(encoding="utf-8")
    exact_text = exact_cmd.read_text(encoding="utf-8")
    assert "--suite-role positive_controls" in learned_text
    assert "--suite-role failure_modes" in exact_text
    assert "--law-package c2_only" in learned_text
    assert "--law-package all_laws_plus_sched" in learned_text
    assert "--data-seed 0" in learned_text
    assert "--model-seed 0" in learned_text
    assert "--exact-family exact" in exact_text
    assert "--exact-family flip_R2" in exact_text


def test_build_markov_law_stress_mechanism_suite_reads_transition_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "out"
    cmd_dir = tmp_path / "cmds"
    transition_summary = tmp_path / "transition_summary.json"
    transition_summary.write_text(
        json.dumps(
            {
                "aggregated_rows": [
                    {
                        "n_regimes": 4,
                        "fixed_leaf_tokens": 16,
                        "train_docs": 128,
                        "val_docs": 32,
                        "test_docs": 64,
                        "audit_fraction": 0.1,
                        "root_weight": 1.0,
                        "state_dim": 64,
                        "hidden_dim": 256,
                        "n_epochs": 4,
                        "feature_mode": "full",
                        "law_package": "all_laws_plus_sched",
                        "bundle_full_success_rate": 0.5,
                        "bundle_margin_mean": 0.0,
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
            "scripts/build_markov_law_stress_suite_cmds.py",
            "--suite",
            "mechanism_suite",
            "--transition-summary",
            str(transition_summary),
            "--output-root",
            str(output_root),
            "--cmd-dir",
            str(cmd_dir),
            "--smoke",
        ],
        cwd=repo_root,
    )

    manifest = json.loads((cmd_dir / "markov_law_stress_suite_manifest.json").read_text(encoding="utf-8"))
    runs = read_manifest_jsonl(Path(manifest["runspec_manifest"]))
    assert runs
    cmd_file = Path(manifest["mechanism_suite"]["cmd_file"])
    text = cmd_file.read_text(encoding="utf-8")
    assert "--suite-role relevance_mediation" in text
    assert "--law-package c1_only" in text
    assert "--law-package sched_only" in text
    assert "--law-package all_laws_plus_sched" in text
