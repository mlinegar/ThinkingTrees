from __future__ import annotations

import json
import sys
from pathlib import Path

from src.ctreepo.sim.suite.publication_ctreepo import main as suite_main


def _nonempty_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_publication_ctreepo_suite_build_smoke(tmp_path: Path) -> None:
    out_root = tmp_path / "pub_suite"
    rc = int(
        suite_main(
            [
                "build",
                "--profile",
                "smoke",
                "--python-bin",
                sys.executable,
                "--output-root",
                str(out_root),
                "--skip-existing",
            ]
        )
    )
    assert rc == 0

    assert (out_root / "suite_meta.json").exists()
    assert (out_root / "suite_cmds.txt").exists()
    assert (out_root / "suite_manifest.jsonl").exists()
    assert _nonempty_lines(out_root / "suite_cmds.txt")


def test_publication_ctreepo_suite_build_records_device_policy(tmp_path: Path) -> None:
    out_root = tmp_path / "pub_suite_device"
    rc = int(
        suite_main(
            [
                "build",
                "--profile",
                "smoke",
                "--python-bin",
                sys.executable,
                "--output-root",
                str(out_root),
                "--device",
                "cpu",
                "--torch-threads",
                "2",
                "--skip-existing",
            ]
        )
    )
    assert rc == 0

    meta = json.loads((out_root / "suite_meta.json").read_text(encoding="utf-8"))
    cmds = _nonempty_lines(out_root / "suite_cmds.txt")
    assert meta["device"] == "cpu"
    assert meta["torch_threads"] == 2
    assert any("--device cpu" in line for line in cmds)


def test_publication_ctreepo_expectations_smoke(tmp_path: Path) -> None:
    out_root = tmp_path / "pub_suite_expectations"
    rc_build = int(
        suite_main(
            [
                "build",
                "--profile",
                "smoke",
                "--python-bin",
                sys.executable,
                "--output-root",
                str(out_root),
                "--skip-existing",
            ]
        )
    )
    assert rc_build == 0

    rc_expect = int(
        suite_main(
            [
                "expectations",
                "--output-root",
                str(out_root),
            ]
        )
    )
    assert rc_expect == 0
    assert (out_root / "simulation_expectations.json").exists()
    assert (out_root / "simulation_expectations.md").exists()
