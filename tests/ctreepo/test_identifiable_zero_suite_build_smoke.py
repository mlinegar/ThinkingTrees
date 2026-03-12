from __future__ import annotations

import json
import sys
from pathlib import Path

from src.ctreepo.sim.suite.identifiable_zero import main as suite_main


def _nonempty_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_identifiable_zero_suite_build_smoke(tmp_path: Path) -> None:
    out_root = tmp_path / "suite_out"
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
    assert (out_root / "suite_plot_cmds.txt").exists()
    assert (out_root / "suite_manifest.jsonl").exists()

    assert _nonempty_lines(out_root / "suite_cmds.txt")
    assert _nonempty_lines(out_root / "suite_plot_cmds.txt")


def test_identifiable_zero_suite_build_records_lane_device_policy(tmp_path: Path) -> None:
    out_root = tmp_path / "suite_out_device"
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
                "--segment-device",
                "cpu",
                "--ctree-device",
                "cpu",
                "--markov-device",
                "auto",
                "--torch-threads",
                "2",
                "--skip-existing",
            ]
        )
    )
    assert rc == 0

    meta = json.loads((out_root / "suite_meta.json").read_text(encoding="utf-8"))
    cmds = _nonempty_lines(out_root / "suite_cmds.txt")
    assert meta["segment_device"] == "cpu"
    assert meta["ctree_device"] == "cpu"
    assert meta["markov_device"] == "auto"
    assert meta["torch_threads"] == 2
    assert any("--device cpu" in line for line in cmds)
