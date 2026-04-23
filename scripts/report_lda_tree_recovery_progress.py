#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ctreepo.sim.suite.legacy_entrypoint import fail_legacy_entrypoint


if __name__ == "__main__":
    raise SystemExit(
        fail_legacy_entrypoint(
            script_name="scripts/report_lda_tree_recovery_progress.py",
            replacement=(
                "venv/bin/python -m src.ctreepo.cli sim suite lda-tree-recovery-progress report "
                "--output-root <root>"
            ),
        )
    )
