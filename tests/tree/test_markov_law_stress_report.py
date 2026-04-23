from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_markov_law_stress_report_is_archived() -> None:
    script = Path("/home/mlinegar/ThinkingTrees/scripts/report_markov_law_stress.py")
    result = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
        cwd="/home/mlinegar/ThinkingTrees",
    )
    assert result.returncode == 2
    assert "archived" in result.stderr.lower()
    assert "sim suite law-stress report --family markov" in result.stderr
