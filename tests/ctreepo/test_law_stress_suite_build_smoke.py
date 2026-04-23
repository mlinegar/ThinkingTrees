from __future__ import annotations

import json
import sys
from pathlib import Path

from src.ctreepo.cli import main as cli_main
from src.ctreepo.sim.suite.law_stress import main as suite_main


def _nonempty_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_transition_summary(path: Path) -> None:
    path.write_text(
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
        )
        + "\n",
        encoding="utf-8",
    )


def test_law_stress_suite_build_smoke(tmp_path: Path) -> None:
    out_root = tmp_path / "law_stress_suite"
    rc = int(
        suite_main(
            [
                "build",
                "--python-bin",
                sys.executable,
                "--output-root",
                str(out_root),
                "--groups",
                "markov_sanity_suite lda_sanity_suite",
                "--smoke",
                "--skip-existing",
                "--markov-device",
                "cpu",
            ]
        )
    )
    assert rc == 0

    meta = json.loads((out_root / "suite_meta.json").read_text(encoding="utf-8"))
    cmds = _nonempty_lines(out_root / "suite_cmds.txt")

    assert (out_root / "suite_manifest.jsonl").exists()
    assert meta["selected_groups"] == ["markov_sanity_suite", "lda_sanity_suite"]
    assert meta["counts_by_group"]["markov_sanity_suite"] > 0
    assert meta["counts_by_group"]["lda_sanity_suite"] > 0
    assert meta["group_families"]["markov_sanity_suite"] == "markov-law-stress"
    assert meta["group_families"]["lda_sanity_suite"] == "lda-law-stress"
    assert meta["group_policies"]["markov_sanity_suite"]["smoke"] is True
    assert meta["group_policies"]["lda_sanity_suite"]["sanity"]["train_docs"] == 32
    assert Path(str(meta["group_manifest_files"]["markov_sanity_suite"])).exists()
    assert Path(str(meta["group_manifest_files"]["lda_sanity_suite"])).exists()
    assert all("build_markov_law_stress_suite_cmds.py" not in line for line in cmds)
    assert all("build_lda_law_stress_suite_cmds.py" not in line for line in cmds)


def test_law_stress_cli_dispatch_mechanism_smoke(tmp_path: Path) -> None:
    out_root = tmp_path / "law_stress_cli"
    transition_summary = tmp_path / "transition_summary.json"
    _write_transition_summary(transition_summary)

    rc = int(
        cli_main(
            [
                "sim",
                "suite",
                "law-stress",
                "build",
                "--python-bin",
                sys.executable,
                "--output-root",
                str(out_root),
                "--groups",
                "markov_mechanism_suite",
                "--transition-summary",
                str(transition_summary),
                "--smoke",
                "--markov-device",
                "cpu",
            ]
        )
    )
    assert rc == 0

    meta = json.loads((out_root / "suite_meta.json").read_text(encoding="utf-8"))
    cmds = _nonempty_lines(out_root / "suite_cmds.txt")

    assert meta["selected_groups"] == ["markov_mechanism_suite"]
    assert meta["counts_by_group"]["markov_mechanism_suite"] > 0
    assert meta["group_policies"]["markov_mechanism_suite"]["mechanism"]["selection_limit"] == 1
    assert Path(str(meta["group_manifest_files"]["markov_mechanism_suite"])).exists()
    assert any("--suite-role relevance_mediation" in line for line in cmds)
