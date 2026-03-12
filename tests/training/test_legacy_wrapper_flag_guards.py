from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_run_training_pipeline_blocks_start_genrm_flag() -> None:
    script = REPO_ROOT / "scripts" / "run_training_pipeline.sh"
    proc = subprocess.run(
        ["bash", str(script), "--start-genrm"],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    text = proc.stdout + proc.stderr
    assert "local-law bootstrap" in text
    assert "no GenRM" in text


def test_run_training_pipeline_blocks_forwarded_enable_genrm_flag() -> None:
    script = REPO_ROOT / "scripts" / "run_training_pipeline.sh"
    proc = subprocess.run(
        ["bash", str(script), "--enable-genrm"],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    text = proc.stdout + proc.stderr
    assert "Deprecated flag '--enable-genrm'" in text


def test_run_manifesto_optimized_example_blocks_enable_genrm_flag() -> None:
    script = REPO_ROOT / "scripts" / "run_manifesto_optimized_example.sh"
    proc = subprocess.run(
        ["bash", str(script), "--enable-genrm"],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    text = proc.stdout + proc.stderr
    assert "deprecated and blocked" in text.lower()


def test_run_manifesto_optimized_example_blocks_optimize_judge_flag() -> None:
    script = REPO_ROOT / "scripts" / "run_manifesto_optimized_example.sh"
    proc = subprocess.run(
        ["bash", str(script), "--optimize-judge"],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    text = proc.stdout + proc.stderr
    assert "local-law bootstrap" in text


def test_run_manifesto_optimized_example_blocks_tot_flag() -> None:
    script = REPO_ROOT / "scripts" / "run_manifesto_optimized_example.sh"
    proc = subprocess.run(
        ["bash", str(script), "--tournament-of-tournaments"],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    text = proc.stdout + proc.stderr
    assert "local-law bootstrap" in text


def test_start_dual_servers_defaults_large_profile_to_qwen_teacher() -> None:
    script = REPO_ROOT / "scripts" / "start_dual_servers.sh"
    content = script.read_text(encoding="utf-8")
    assert "qwen3.5-397b-a17b-nvfp4" in content
