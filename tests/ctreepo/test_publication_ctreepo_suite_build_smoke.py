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
    assert meta["policy"]["profile"] == "smoke"
    assert meta["policy"]["seeds"] == [0]
    assert meta["lane_catalog"]["lda_lane_lda_direct"]["regime"] == "lda"
    assert meta["lane_catalog"]["hard_lane_phi_base"]["fixed_leaf_tokens_grid"] == [16]
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


def test_publication_ctreepo_run_uses_manifest_queue_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    out_root = tmp_path / "pub_suite_run"
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

    captured: dict[str, object] = {}

    def _fake_queue(
        *,
        manifest_paths: list[Path],
        cpu_workers: int,
        gpu_tokens: str,
        log_dir: Path,
        set_thread_env: bool,
    ) -> dict[str, object]:
        captured["manifest_paths"] = manifest_paths
        captured["cpu_workers"] = cpu_workers
        captured["gpu_tokens"] = gpu_tokens
        captured["log_dir"] = log_dir
        captured["set_thread_env"] = set_thread_env
        return {
            "manifest_paths": [str(path) for path in manifest_paths],
            "log_dir": str(log_dir),
            "gpu_tokens": [],
            "summary": {"n_fail": 0},
        }

    monkeypatch.setattr("src.ctreepo.sim.suite.publication_ctreepo.run_manifest_queue_suite", _fake_queue)

    rc_run = int(
        suite_main(
            [
                "run",
                "--output-root",
                str(out_root),
                "--jobs",
                "3",
                "--gpu-tokens",
                "cpu",
            ]
        )
    )
    assert rc_run == 0
    assert captured["manifest_paths"] == [out_root / "suite_manifest.jsonl"]
    assert captured["cpu_workers"] == 3
    assert captured["gpu_tokens"] == "cpu"
    assert captured["set_thread_env"] is False
