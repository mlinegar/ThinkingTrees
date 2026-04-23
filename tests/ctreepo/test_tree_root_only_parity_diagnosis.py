from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_tree_root_only_parity_report_is_archived() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/report_tree_root_only_parity_pdf.py",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "archived" in result.stderr.lower()
    assert "report_markov_optimization_tradeoffs.py" in result.stderr


def test_tree_root_only_parity_diagnosis_is_archived() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_tree_root_only_parity_diagnosis.py",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "archived" in result.stderr.lower()
    assert "run_markov_supervision_recovery_parity_grid.py" in result.stderr
