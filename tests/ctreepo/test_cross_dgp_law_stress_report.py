"""Smoke tests for scripts/report_cross_dgp_law_stress.py.

Tests the Markov CSV loading, LDA JSON loading, combined report generation,
and text table formatting paths. The unified/backfill mode is not tested here
(it has heavy dependencies); it is covered indirectly via
test_local_law_learnability_protocol.py.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "report_cross_dgp_law_stress.py"
REPO_ROOT = str(Path(__file__).resolve().parents[2])


def _env_with_agg_backend() -> dict[str, str]:
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = REPO_ROOT + (os.pathsep + env.get("PYTHONPATH", ""))
    return env


def _write_markov_csv(markov_dir: Path) -> None:
    markov_dir.mkdir(parents=True, exist_ok=True)
    csv_path = markov_dir / "markov_law_stress_aggregated_rows.csv"
    rows = [
        {
            "law_package": "root_only",
            "n_runs": "4",
            "primary_pass_rate": "0.25",
            "c1_pass_rate": "0.50",
            "c2_pass_rate": "0.50",
            "c3_pass_rate": "0.75",
            "mean_laws_improved": "1.75",
            "mean_primary_gain": "0.05",
        },
        {
            "law_package": "all_laws",
            "n_runs": "6",
            "primary_pass_rate": "0.67",
            "c1_pass_rate": "0.83",
            "c2_pass_rate": "0.67",
            "c3_pass_rate": "1.00",
            "mean_laws_improved": "2.50",
            "mean_primary_gain": "0.18",
        },
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_lda_jsons(lda_dir: Path) -> None:
    for idx, (pkg, c1, c2, c3, root, status) in enumerate(
        [
            ("all_laws", True, True, False, True, "primary_only"),
            ("all_laws", False, True, True, True, "laws_only"),
            ("all_laws", True, True, True, True, "full_success"),
        ]
    ):
        out = lda_dir / f"seed_{idx}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {"law_package": pkg},
            "local_law": {
                "selection": {"selected_candidate": "learned_g"},
                "law_stress": {
                    "learned_g": {
                        "c1_pass": c1,
                        "c2_pass": c2,
                        "c3_pass": c3,
                        "root_pass": root,
                        "bundle_full_success": (c1 and c2 and c3 and root),
                        "bundle_status": status,
                    }
                },
            },
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_report(*extra_args: str, cwd: str = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *extra_args],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=_env_with_agg_backend(),
    )


def test_cross_dgp_markov_csv_loading(tmp_path: Path):
    markov_dir = tmp_path / "markov_report"
    _write_markov_csv(markov_dir)
    out_dir = tmp_path / "output"
    result = _run_report("--markov-dir", str(markov_dir), "--output-dir", str(out_dir))
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    assert (out_dir / "cross_dgp_law_stress_table.txt").exists()
    assert (out_dir / "cross_dgp_law_stress_report.pdf").exists()

    summary = json.loads((out_dir / "cross_dgp_law_stress_summary.json").read_text())
    rows = summary["rows"]
    assert len(rows) == 2
    assert all(r["dgp"] == "markov_ops_count" for r in rows)
    pkgs = {r["law_package"] for r in rows}
    assert pkgs == {"root_only", "all_laws"}


def test_cross_dgp_lda_json_loading(tmp_path: Path):
    lda_dir = tmp_path / "lda_report"
    _write_lda_jsons(lda_dir)
    out_dir = tmp_path / "output"
    result = _run_report("--lda-dir", str(lda_dir), "--output-dir", str(out_dir))
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    summary = json.loads((out_dir / "cross_dgp_law_stress_summary.json").read_text())
    rows = summary["rows"]
    assert len(rows) >= 1
    assert all(r["dgp"] == "lda" for r in rows)
    # 3 LDA files, all with all_laws => 1 aggregated row
    lda_row = rows[0]
    assert lda_row["n_runs"] == 3
    assert lda_row["law_package"] == "all_laws"
    # c2 passes in all 3: rate should be 1.0
    assert lda_row["c2_pass_rate"] == pytest.approx(1.0)


def test_cross_dgp_combined_report(tmp_path: Path):
    markov_dir = tmp_path / "markov_report"
    lda_dir = tmp_path / "lda_report"
    _write_markov_csv(markov_dir)
    _write_lda_jsons(lda_dir)
    out_dir = tmp_path / "output"
    result = _run_report(
        "--markov-dir", str(markov_dir),
        "--lda-dir", str(lda_dir),
        "--output-dir", str(out_dir),
    )
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    summary = json.loads((out_dir / "cross_dgp_law_stress_summary.json").read_text())
    rows = summary["rows"]
    dgps = {r["dgp"] for r in rows}
    assert "markov_ops_count" in dgps
    assert "lda" in dgps
    # 2 markov rows + 1 lda row = 3 total
    assert len(rows) == 3


def test_cross_dgp_text_table_has_correct_structure(tmp_path: Path):
    markov_dir = tmp_path / "markov_report"
    lda_dir = tmp_path / "lda_report"
    _write_markov_csv(markov_dir)
    _write_lda_jsons(lda_dir)
    out_dir = tmp_path / "output"
    _run_report(
        "--markov-dir", str(markov_dir),
        "--lda-dir", str(lda_dir),
        "--output-dir", str(out_dir),
    )
    table = (out_dir / "cross_dgp_law_stress_table.txt").read_text()
    lines = [l for l in table.strip().splitlines() if l.strip()]
    # Find the header row (contains "DGP" and "Package")
    header_idx = next(i for i, l in enumerate(lines) if "DGP" in l and "Package" in l)
    assert header_idx >= 0
    assert "---" in lines[header_idx + 1]
    # At least 2 data rows after separator
    assert len(lines) >= header_idx + 4
