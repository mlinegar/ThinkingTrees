from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_identifiable_zero_neural_operator_overnight_is_migration_stub() -> None:
    script = REPO_ROOT / "scripts" / "run_identifiable_zero_neural_operator_overnight.sh"
    proc = subprocess.run(["bash", str(script)], capture_output=True, text=True)

    assert proc.returncode == 2
    assert "retired in the v2 simulation API" in proc.stderr
    assert "identifiable-zero-neural-operator" in proc.stderr
