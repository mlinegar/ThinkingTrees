from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _nonempty_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_lda_law_stress_builder_skips_existing_and_writes_artifact_dirs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "suite_out"
    cmd_dir = tmp_path / "cmds"
    cmd_file = cmd_dir / "lda_law_stress_sanity_suite_cmds.txt"

    subprocess.check_call(
        [
            sys.executable,
            "scripts/build_lda_law_stress_suite_cmds.py",
            "--suite",
            "sanity_suite",
            "--output-root",
            str(output_root),
            "--cmd-dir",
            str(cmd_dir),
            "--python-bin",
            sys.executable,
            "--smoke",
            "--skip-existing",
        ],
        cwd=repo_root,
    )

    lines = _nonempty_lines(cmd_file)
    assert len(lines) == 32
    assert all("--artifact-dir" in line for line in lines)

    existing_json = output_root / "sanity_suite" / "results" / "sanity_learned" / "tau1_lam0_pkg_root_only_mode_aligned_s0.json"
    existing_csv = existing_json.with_suffix(".csv")
    existing_json.parent.mkdir(parents=True, exist_ok=True)
    existing_json.write_text("{}", encoding="utf-8")
    existing_csv.write_text("", encoding="utf-8")

    subprocess.check_call(
        [
            sys.executable,
            "scripts/build_lda_law_stress_suite_cmds.py",
            "--suite",
            "sanity_suite",
            "--output-root",
            str(output_root),
            "--cmd-dir",
            str(cmd_dir),
            "--python-bin",
            sys.executable,
            "--smoke",
            "--skip-existing",
        ],
        cwd=repo_root,
    )

    lines = _nonempty_lines(cmd_file)
    assert len(lines) == 31
    assert not any("tau1_lam0_pkg_root_only_mode_aligned_s0" in line for line in lines)
