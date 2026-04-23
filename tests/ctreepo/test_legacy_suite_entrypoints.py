from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_python_legacy_entrypoint_fails_fast() -> None:
    script = REPO_ROOT / "scripts" / "build_cpu_megasweep_cmds.py"
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)

    assert proc.returncode == 2
    assert "retired in the v2 simulation API" in proc.stderr
    assert "cpu-megasweep build" in proc.stderr


def test_shell_legacy_entrypoint_fails_fast() -> None:
    script = REPO_ROOT / "scripts" / "launch_lda_tree_recovery_production_sweeps.sh"
    proc = subprocess.run(["bash", str(script)], capture_output=True, text=True)

    assert proc.returncode == 2
    assert "retired in the v2 simulation API" in proc.stderr
    assert "lda-tree-recovery-progress" in proc.stderr
