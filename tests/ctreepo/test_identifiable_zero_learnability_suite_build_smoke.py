from __future__ import annotations

import json
import sys
from pathlib import Path

from src.ctreepo.cli import main as cli_main
from src.ctreepo.sim.suite.identifiable_zero_learnability import main as suite_main


def _nonempty_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_identifiable_zero_learnability_suite_build_smoke(tmp_path: Path) -> None:
    out_root = tmp_path / "learnability_suite"
    rc = int(
        suite_main(
            [
                "build",
                "--python-bin",
                sys.executable,
                "--output-root",
                str(out_root),
                "--groups",
                "markov_baseline ctree_baseline_theta",
                "--train-docs-grid",
                "64",
                "--label-rate-grid",
                "0.1",
                "--base-seeds",
                "0",
                "--no-hero",
                "--skip-existing",
            ]
        )
    )
    assert rc == 0

    meta = json.loads((out_root / "suite_meta.json").read_text(encoding="utf-8"))
    cmds = _nonempty_lines(out_root / "suite_cmds.txt")

    assert (out_root / "suite_manifest.jsonl").exists()
    assert meta["selected_groups"] == ["markov_baseline", "ctree_baseline_theta"]
    assert meta["policy"]["train_docs_grid"] == [64]
    assert meta["policy"]["label_rate_grid"] == [0.1]
    assert meta["policy"]["base_seeds"] == [0]
    assert set(meta["counts_by_group"]) == {"markov_baseline", "ctree_baseline_theta"}
    assert any("--include-doc-level-baseline" in line for line in cmds)
    assert any("--include-doc-level-ridge-baseline" in line for line in cmds)
    assert any("--include-leaf-ridge-tree-baseline" in line for line in cmds)
    assert any("--include-leaf-endpoint-table-tree-baseline" in line for line in cmds)
    assert any("--include-leaf-dt-tree-baseline" in line for line in cmds)
    assert any("--include-leaf-knn-tree-baseline" in line for line in cmds)
    assert any("--include-leaf-rf-tree-baseline" in line for line in cmds)
    assert any("--include-full-doc-theta-baseline" in line for line in cmds)


def test_identifiable_zero_learnability_cli_dispatch_smoke(tmp_path: Path) -> None:
    out_root = tmp_path / "learnability_cli"
    rc = int(
        cli_main(
            [
                "sim",
                "suite",
                "identifiable-zero-learnability",
                "build",
                "--python-bin",
                sys.executable,
                "--output-root",
                str(out_root),
                "--groups",
                "markov_baseline",
                "--train-docs-grid",
                "32",
                "--label-rate-grid",
                "0.2",
                "--base-seeds",
                "0",
                "--no-hero",
            ]
        )
    )
    assert rc == 0

    meta = json.loads((out_root / "suite_meta.json").read_text(encoding="utf-8"))
    assert meta["selected_groups"] == ["markov_baseline"]
    assert meta["counts_by_group"]["markov_baseline"] > 0


def test_identifiable_zero_learnability_smoke_profile_defaults_to_markov_only_hero_off(tmp_path: Path) -> None:
    out_root = tmp_path / "learnability_smoke"
    rc = int(
        suite_main(
            [
                "build",
                "--python-bin",
                sys.executable,
                "--output-root",
                str(out_root),
                "--profile",
                "smoke",
                "--groups",
                "markov_baseline",
                "--markov-device",
                "cpu",
            ]
        )
    )
    assert rc == 0

    meta = json.loads((out_root / "suite_meta.json").read_text(encoding="utf-8"))
    assert meta["profile"] == "smoke"
    assert meta["hero"] is False
    assert meta["policy"]["profile"] == "smoke"
    assert meta["policy"]["train_docs_grid"] == [16]
    assert meta["policy"]["label_rate_grid"] == [0.1]
    assert meta["policy"]["heldout_docs"] == 16
    assert meta["policy"]["base_seeds"] == [0]
    assert meta["policy"]["markov_sampled_leaf_pool_leaf_counts"] == [1, 2, 4, 8]
    cmds = _nonempty_lines(out_root / "suite_cmds.txt")
    assert any("--include-sampled-leaf-pool-ridge-baseline" in line for line in cmds)
    assert any("--include-sampled-leaf-pool-rf-baseline" in line for line in cmds)
    assert any("--include-leaf-endpoint-table-tree-baseline" in line for line in cmds)
    assert any("--include-leaf-dt-tree-baseline" in line for line in cmds)
    assert any("--include-leaf-knn-tree-baseline" in line for line in cmds)
    assert any("--include-leaf-rf-tree-baseline" in line for line in cmds)
    assert any("--sampled-leaf-pool-leaf-counts 1,2,4,8" in line for line in cmds)
