from __future__ import annotations

import sys
from pathlib import Path

from src.ctreepo.sim.cli.sweep_segment_lda_ops_weight_recovery import main as sweep_ops_main
from src.ctreepo.sim.cli.sweep_segmented_lda_ctreepo import main as sweep_main
from src.ctreepo.sim.manifest import read_manifest_jsonl


def test_family_sweep_build_smoke(tmp_path: Path) -> None:
    out_cmds = tmp_path / "cmds.txt"
    out_manifest = tmp_path / "manifest.jsonl"
    out_root = tmp_path / "outputs"

    rc = int(
        sweep_main(
            [
                "--python-bin",
                sys.executable,
                "--out-cmds",
                str(out_cmds),
                "--out-manifest",
                str(out_manifest),
                "--output-root",
                str(out_root),
                "--train-docs",
                "64",
                "--seeds",
                "0",
                "--calibration-rates",
                "0",
                "--eval-leaf-rates",
                "0",
                "--eval-internal-rates",
                "0",
                "--device",
                "auto",
                "--torch-threads",
                "1",
                "--skip-existing",
            ]
        )
    )
    assert rc == 0

    cmds = [ln for ln in out_cmds.read_text(encoding="utf-8").splitlines() if ln.strip()]
    runs = read_manifest_jsonl(out_manifest)
    assert len(cmds) == len(runs)
    assert len(cmds) == 1
    assert "--device auto" in cmds[0]
    assert runs[0].config["device"] == "auto"
    assert runs[0].config["torch_threads"] == 1


def test_segment_lda_ops_sweep_build_carries_device_policy(tmp_path: Path) -> None:
    out_cmds = tmp_path / "ops_cmds.txt"
    out_manifest = tmp_path / "ops_manifest.jsonl"
    out_root = tmp_path / "ops_outputs"

    rc = int(
        sweep_ops_main(
            [
                "--python-bin",
                sys.executable,
                "--out-cmds",
                str(out_cmds),
                "--out-manifest",
                str(out_manifest),
                "--output-root",
                str(out_root),
                "--train-docs",
                "32",
                "--test-docs",
                "32",
                "--audit-fractions",
                "0.1",
                "--topic-phi-docs",
                "0",
                "--topic-phi-estimators",
                "true",
                "--topic-processes",
                "segments",
                "--lambda-multipliers",
                "0",
                "--seeds",
                "0",
                "--device",
                "cpu",
                "--torch-threads",
                "2",
                "--skip-existing",
            ]
        )
    )
    assert rc == 0

    cmds = [ln for ln in out_cmds.read_text(encoding="utf-8").splitlines() if ln.strip()]
    runs = read_manifest_jsonl(out_manifest)
    assert len(cmds) == len(runs) == 1
    assert "--device cpu" in cmds[0]
    assert "--torch-threads 2" in cmds[0]
    assert runs[0].config["device"] == "cpu"
    assert runs[0].config["torch_threads"] == 2
