from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_markov_capability_report_is_archived() -> None:
    script = Path("/home/mlinegar/ThinkingTrees/scripts/report_markov_capability_map.py")
    result = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
        cwd="/home/mlinegar/ThinkingTrees",
    )
    assert result.returncode == 2
    assert "archived" in result.stderr.lower()
    assert "report_markov_optimization_tradeoffs.py" in result.stderr
