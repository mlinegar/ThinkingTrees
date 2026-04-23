from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_tree_fno_tuning_report_is_archived() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/report_tree_fno_tuning_pdf.py",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "archived" in result.stderr.lower()
    assert "report_markov_optimization_tradeoffs.py" in result.stderr


def test_full_doc_anchor_diagnostics_pdf_is_archived() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/report_full_doc_anchor_diagnostics_pdf.py",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "archived" in result.stderr.lower()
    assert "report_markov_optimization_tradeoffs.py" in result.stderr
