from __future__ import annotations

import json
import sys
from pathlib import Path

from src.ctreepo.sim.suite.identifiable_zero_publication import main as suite_main


def _nonempty_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_identifiable_zero_publication_clean_suite_build_smoke(tmp_path: Path) -> None:
    out_root = tmp_path / "publication_clean_suite"
    rc = int(
        suite_main(
            [
                "build",
                "--profile",
                "publication_clean",
                "--python-bin",
                sys.executable,
                "--output-root",
                str(out_root),
                "--groups",
                "cpu",
                "--n-seeds",
                "1",
                "--segment-test-docs",
                "16",
                "--ctree-test-books",
                "16",
                "--markov-test-docs",
                "16",
                "--markov-n-epochs",
                "1",
                "--skip-existing",
            ]
        )
    )
    assert rc == 0

    meta = json.loads((out_root / "suite_meta.json").read_text(encoding="utf-8"))
    assert meta["profile"] == "publication_clean"
    assert meta["selected_groups"] == ["cpu"]
    assert meta["policy"]["segment"]["train_docs"] == [12000]
    assert meta["counts_by_group"]["cpu"] > 0
    assert meta["counts_by_group"]["gpu"] == 0
    assert _nonempty_lines(out_root / "suite_cmds.txt")
    assert _nonempty_lines(out_root / "suite_plot_cmds.txt")


def test_identifiable_zero_longrun_suite_build_smoke(tmp_path: Path) -> None:
    out_root = tmp_path / "longrun_suite"
    rc = int(
        suite_main(
            [
                "build",
                "--profile",
                "longrun_equiv_v1",
                "--python-bin",
                sys.executable,
                "--output-root",
                str(out_root),
                "--groups",
                "equiv",
                "--n-seeds",
                "1",
                "--pilot-cmd-count",
                "12",
                "--segment-test-docs",
                "16",
                "--ctree-test-books",
                "16",
                "--markov-test-docs",
                "16",
                "--markov-n-epochs",
                "1",
                "--skip-existing",
            ]
        )
    )
    assert rc == 0

    meta = json.loads((out_root / "suite_meta.json").read_text(encoding="utf-8"))
    assert meta["profile"] == "longrun_equiv_v1"
    assert meta["selected_groups"] == ["equiv"]
    assert meta["counts_by_group"]["equiv"] > 0
    assert meta["counts_by_group"]["scale"] == 0
    assert meta["counts_by_group"]["pilot"] == 12
    assert Path(str(meta["group_manifest_files"]["pilot"])).exists()


def test_identifiable_zero_publication_run_uses_manifest_queue_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    out_root = tmp_path / "publication_clean_run"
    rc_build = int(
        suite_main(
            [
                "build",
                "--profile",
                "publication_clean",
                "--python-bin",
                sys.executable,
                "--output-root",
                str(out_root),
                "--groups",
                "cpu",
                "--n-seeds",
                "1",
                "--segment-test-docs",
                "16",
                "--ctree-test-books",
                "16",
                "--markov-test-docs",
                "16",
                "--markov-n-epochs",
                "1",
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

    monkeypatch.setattr(
        "src.ctreepo.sim.suite.identifiable_zero_publication.run_manifest_queue_suite",
        _fake_queue,
    )

    rc_run = int(
        suite_main(
            [
                "run",
                "--profile",
                "publication_clean",
                "--output-root",
                str(out_root),
                "--jobs",
                "2",
                "--gpu-tokens",
                "cpu",
            ]
        )
    )
    assert rc_run == 0
    assert captured["manifest_paths"] == [out_root / "suite_groups" / "manifests" / "cpu.jsonl"]
    assert captured["cpu_workers"] == 2
    assert captured["gpu_tokens"] == "cpu"
    assert captured["set_thread_env"] is True
