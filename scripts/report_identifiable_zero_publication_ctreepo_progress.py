#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ctreepo.sim.suite.legacy_entrypoint import fail_legacy_entrypoint


if __name__ == "__main__":
    raise SystemExit(
        fail_legacy_entrypoint(
            script_name="scripts/report_identifiable_zero_publication_ctreepo_progress.py",
            replacement="venv/bin/python -m src.ctreepo.cli sim suite publication-ctreepo report --output-root <root>",
        )
    )
