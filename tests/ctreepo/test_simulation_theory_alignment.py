from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from src.ctreepo.sim.expectations import ExpectationFinding, ExpectationReport, merge_expectation_reports
from src.ctreepo.sim.theory_alignment import (
    SimulationTheoryAlignmentReport,
    build_simulation_theory_alignment_report,
    render_simulation_theory_alignment_markdown,
)
from tests.ctreepo.test_simulation_expectations import _build_fixture_tree


def _write_bundle_manifest(path: Path) -> Path:
    payload = {
        "formal_root": str(path.parent),
        "results": [
            {
                "name": "cpu_megasweep",
                "status": "completed",
                "root": "/tmp/cpu_megasweep",
                "bundle_role": "paper",
            },
            {
                "name": "simulation_buildout",
                "status": "completed",
                "root": "/tmp/simulation_buildout",
                "bundle_role": "paper",
            },
            {
                "name": "publication_clean",
                "status": "partial",
                "root": "/tmp/publication_clean",
                "bundle_role": "paper",
            },
            {
                "name": "publication_ctreepo_progress",
                "status": "completed",
                "root": "/tmp/publication_ctreepo_progress",
                "bundle_role": "paper",
            },
            {
                "name": "learnability",
                "status": "partial",
                "root": "/tmp/learnability",
                "bundle_role": "paper",
            },
            {
                "name": "neural_operator_overnight",
                "status": "completed",
                "root": "/tmp/neural_operator_overnight",
                "bundle_role": "paper",
            },
            {
                "name": "lda_leafnoise",
                "status": "completed",
                "root": "/tmp/lda_leafnoise",
                "bundle_role": "paper",
            },
            {
                "name": "dtm_lda",
                "status": "completed",
                "root": "/tmp/dtm_lda",
                "bundle_role": "paper",
            },
            {
                "name": "lda_tree_recovery_progress",
                "status": "partial",
                "root": "/tmp/lda_tree_recovery_progress",
                "bundle_role": "diagnostic",
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_build_simulation_theory_alignment_report_from_fixture_root(tmp_path: Path) -> None:
    root = _build_fixture_tree(tmp_path)
    bundle_manifest = _write_bundle_manifest(tmp_path / "paper_reports" / "paper_report_bundle_manifest.json")

    report = build_simulation_theory_alignment_report(
        formal_root=root,
        bundle_manifest_path=bundle_manifest,
    )

    statuses = {row.family: row.overall_status for row in report.family_statuses}
    assert statuses["markov_ops_count"] == "provisionally_aligned"
    assert statuses["segment_lda_ops_weight_recovery"] == "provisionally_aligned"
    assert statuses["segmented_lda_ctreepo"] == "provisionally_aligned"
    assert statuses["mergeable_ablation"] == "aligned"
    assert statuses["local_law_learnability"] in {"incomplete", "provisionally_aligned"}
    assert int(report.summary["n_families"]) == 5
    assert int(report.summary["provisionally_aligned_families"]) >= 3
    assert int(report.summary["provisionally_aligned_families"]) >= 3

    markdown = render_simulation_theory_alignment_markdown(report)
    assert "Simulation Theory Alignment" in markdown
    assert "markov_path_local_laws_of_encoded_state" in markdown
    assert "sketchReduce_countSketch_eq_bagOfWords" in markdown


def test_merge_expectation_reports_deduplicates_findings() -> None:
    finding = ExpectationFinding(
        kind="demo",
        title="Demo expectation",
        status="pass",
        family="markov_ops_count",
        scenario="demo",
        metric="root_mae",
        method="exact",
        direction="decreasing",
        observed_summary={},
        thresholds={},
        supporting_rows=[],
    )
    left = ExpectationReport(
        input_root="/tmp/a",
        manifest=None,
        families_scanned=["markov_ops_count"],
        rows_scanned=10,
        expectations=[finding],
        summary={"n_pass": 1},
    )
    right = ExpectationReport(
        input_root="/tmp/b",
        manifest=None,
        families_scanned=["markov_ops_count", "mergeable_ablation"],
        rows_scanned=20,
        expectations=[finding],
        summary={"n_pass": 1},
    )

    merged = merge_expectation_reports([left, right], input_root="/tmp/merged")

    assert merged.input_root == "/tmp/merged"
    assert merged.rows_scanned == 30
    assert merged.families_scanned == ["markov_ops_count", "mergeable_ablation"]
    assert len(merged.expectations) == 1
    assert int(merged.summary["n_pass"]) == 1


def test_simulation_theory_alignment_cli_reuses_written_expectations(tmp_path: Path) -> None:
    root = _build_fixture_tree(tmp_path)
    paper_reports = root / "paper_reports"
    bundle_manifest = _write_bundle_manifest(paper_reports / "paper_report_bundle_manifest.json")
    script = Path("/home/mlinegar/ThinkingTrees/scripts/report_simulation_theory_alignment.py")

    proc_first = subprocess.run(
        [
            sys.executable,
            str(script),
            "--formal-root",
            str(root),
            "--bundle-manifest",
            str(bundle_manifest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc_first.returncode == 0, proc_first.stdout + proc_first.stderr

    expectation_json = paper_reports / "simulation_expectations.json"
    theory_json = paper_reports / "simulation_theory_alignment.json"
    theory_md = paper_reports / "simulation_theory_alignment.md"
    assert expectation_json.exists()
    assert theory_json.exists()
    assert theory_md.exists()

    proc_second = subprocess.run(
        [
            sys.executable,
            str(script),
            "--formal-root",
            str(root),
            "--bundle-manifest",
            str(bundle_manifest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc_second.returncode == 0, proc_second.stdout + proc_second.stderr

    report = SimulationTheoryAlignmentReport.from_dict(json.loads(theory_json.read_text(encoding="utf-8")))
    assert report.formal_root == str(root.resolve())
    assert report.expectation_source == str(expectation_json.resolve())
    assert any(row.family == "mergeable_ablation" and row.overall_status == "aligned" for row in report.family_statuses)
