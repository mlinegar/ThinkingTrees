from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_markov_cohort_compare_report_is_archived(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/report_markov_cohort_compare.py",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "archived" in result.stderr.lower()
    assert "report_markov_optimization_tradeoffs.py" in result.stderr
