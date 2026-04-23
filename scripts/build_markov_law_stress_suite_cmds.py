#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ctreepo.sim.suite.legacy_entrypoint import fail_legacy_entrypoint


if __name__ == "__main__":
    raise SystemExit(
        fail_legacy_entrypoint(
            script_name="scripts/build_markov_law_stress_suite_cmds.py",
            replacement=(
                "venv/bin/python -m src.ctreepo.cli sim suite law-stress build "
                "--groups 'markov_sanity_suite markov_mechanism_suite' --output-root <root>"
            ),
        )
    )
