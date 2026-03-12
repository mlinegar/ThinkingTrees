from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from src.ctreepo.sim.runner import read_cmds_file, run_commands


def test_markov_law_stress_sanity_smoke_end_to_end(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "out"
    cmd_dir = tmp_path / "cmds"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    subprocess.check_call(
        [
            sys.executable,
            "scripts/build_markov_law_stress_suite_cmds.py",
            "--suite",
            "sanity_suite",
            "--output-root",
            str(output_root),
            "--cmd-dir",
            str(cmd_dir),
            "--device",
            "cpu",
            "--torch-threads",
            "1",
            "--smoke",
        ],
        cwd=repo_root,
        env=env,
    )

    learned_cmds = read_cmds_file(cmd_dir / "sanity_suite_learned_cmds.txt")
    exact_cmds = read_cmds_file(cmd_dir / "sanity_suite_exact_cmds.txt")
    results = run_commands(
        learned_cmds + exact_cmds,
        jobs=2,
        log_dir=tmp_path / "logs",
        fail_fast=True,
        env=env,
    )
    assert results
    assert all(int(r.returncode) == 0 for r in results)

    report_dir = output_root / "sanity_suite" / "markov_changepoint_ops_count" / "law_stress_report"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_law_stress.py",
            "--input-root",
            str(output_root / "sanity_suite" / "markov_changepoint_ops_count"),
            "--output-dir",
            str(report_dir),
            "--suite-type",
            "sanity_suite",
        ],
        cwd=repo_root,
        env={**env, "MPLBACKEND": "Agg"},
    )

    assert (report_dir / "markov_law_stress_summary.json").exists()
    assert (report_dir / "markov_law_stress_assessed_rows.csv").exists()
    assert (report_dir / "markov_law_stress.md").exists()
    assert (report_dir / "markov_law_stress_report.pdf").exists()
    summary = json.loads((report_dir / "markov_law_stress_summary.json").read_text(encoding="utf-8"))
    assert int(summary["raw_row_count"]) > 0

    learned_run = next(
        (output_root / "sanity_suite" / "markov_changepoint_ops_count" / "learned").rglob("seed_0.json")
    )
    learned_payload = json.loads(learned_run.read_text(encoding="utf-8"))
    assert "local_law_learnability" in learned_payload
    assert "g_artifacts" in learned_payload
    assert "oracle_g" in learned_payload["g_artifacts"]
    assert any(
        str(policy["role"]) in {"baseline_g", "learned_g"}
        for policy in learned_payload["local_law_learnability"]["policies"].values()
    )
