from __future__ import annotations

import json
from pathlib import Path

from src.ctreepo.sim.suite.law_stress_builders import build_lda_law_stress_suites


def _nonempty_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_lda_law_stress_builder_skips_existing_and_writes_artifact_dirs(tmp_path: Path) -> None:
    output_root = tmp_path / "suite_out"
    cmd_dir = tmp_path / "cmds"
    cmd_file = cmd_dir / "lda_law_stress_sanity_suite_cmds.txt"

    build_lda_law_stress_suites(
        suite="sanity_suite",
        output_root=output_root,
        cmd_dir=cmd_dir,
        python_bin="python",
        skip_existing=True,
        smoke=True,
    )

    meta = json.loads((cmd_dir / "lda_law_stress_meta.json").read_text(encoding="utf-8"))
    lines = _nonempty_lines(cmd_file)
    assert len(lines) == 32
    assert all("--artifact-dir" in line for line in lines)
    assert Path(str(meta["runspec_manifest"])).exists()
    assert meta["policy"]["smoke"] is True
    assert meta["policy"]["sanity"]["train_docs"] == 32

    existing_json = output_root / "sanity_suite" / "results" / "sanity_learned" / "tau1_lam0_pkg_root_only_mode_aligned_s0.json"
    existing_csv = existing_json.with_suffix(".csv")
    existing_json.parent.mkdir(parents=True, exist_ok=True)
    existing_json.write_text("{}", encoding="utf-8")
    existing_csv.write_text("", encoding="utf-8")

    build_lda_law_stress_suites(
        suite="sanity_suite",
        output_root=output_root,
        cmd_dir=cmd_dir,
        python_bin="python",
        skip_existing=True,
        smoke=True,
    )

    lines = _nonempty_lines(cmd_file)
    assert len(lines) == 31
    assert not any("tau1_lam0_pkg_root_only_mode_aligned_s0" in line for line in lines)
