from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def _write_run_artifact(job_dir: Path, *, elapsed_s: float, resident_store_hits: float, gpu_reserved_mem_peak_gb: float) -> None:
    run_dir = job_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "elapsed_s": elapsed_s,
                "runtime_efficiency": {
                    "resident_store_hits": resident_store_hits,
                    "resident_store_misses": 0.0,
                    "steady_state_h2d_bytes": 0.0,
                    "gpu_reserved_mem_peak_gb": gpu_reserved_mem_peak_gb,
                    "gpu_allocated_mem_peak_gb": gpu_reserved_mem_peak_gb - 0.2,
                    "runtime_data_mode": "resident",
                    "runtime_bucket_mode": "leaf_count_auto_queue",
                    "runtime_workers_per_mig": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )


def _make_row(
    simulation_root: Path,
    *,
    job_name: str,
    scope_label: str,
    claim_level: str,
    baseline_family: str,
    recipe_id: str,
    fixed_leaf_tokens: int,
    test_root_mae_mean: float,
    test_leaf_mae_mean: float,
    test_merge_mae_mean: float,
    val_root_mae_mean: float,
    train_doc_count: int = 4096,
    strict_collapse_pass: bool = False,
) -> dict[str, object]:
    job_dir = simulation_root / "jobs" / job_name
    return {
        "job_name": job_name,
        "job_output_dir": str(job_dir),
        "scope_label": scope_label,
        "claim_level": claim_level,
        "train_doc_count": train_doc_count,
        "recipe_id": recipe_id,
        "baseline_family": baseline_family,
        "fixed_leaf_tokens": fixed_leaf_tokens,
        "state": "completed",
        "test_root_mae_mean": test_root_mae_mean,
        "test_leaf_mae_mean": test_leaf_mae_mean,
        "test_merge_mae_mean": test_merge_mae_mean,
        "val_root_mae_mean": val_root_mae_mean,
        "strict_collapse_pass": strict_collapse_pass,
        "reference_bundle_source": "bundle_train4096",
        "train_prefix_counts": [1024, 2048, 4096],
        "config_diff_vs_official_fno": {},
    }


def _write_simulation_root(
    simulation_root: Path,
    *,
    rows: list[dict[str, object]],
    train_doc_counts: list[int] | None = None,
    assumed_doc_tokens: int = 128,
    one_leaf_target_fixed_leaf_tokens: int = 128,
) -> None:
    simulation_root.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(rows):
        job_dir = simulation_root / "jobs" / str(row["job_name"])
        _write_run_artifact(
            job_dir,
            elapsed_s=120.0 + idx,
            resident_store_hits=640.0 + idx,
            gpu_reserved_mem_peak_gb=1.4 + 0.05 * idx,
        )
    (simulation_root / "parity_grid_manifest.json").write_text(
        json.dumps(
            {
                "assumed_doc_tokens": assumed_doc_tokens,
                "one_leaf_target_fixed_leaf_tokens": one_leaf_target_fixed_leaf_tokens,
                "train_doc_counts": train_doc_counts or [4096],
                "jobs": [],
            }
        ),
        encoding="utf-8",
    )
    (simulation_root / "scheduler_status.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "items_total": len(rows),
                "completed_items": len(rows),
                "failed_items": 0,
                "active_items": 0,
                "pending_items": 0,
            }
        ),
        encoding="utf-8",
    )
    (simulation_root / "parity_grid_summary.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-04-03T22:30:00+00:00",
                "state": "completed",
                "evidence_status": "complete_ready",
                "assumed_doc_tokens": assumed_doc_tokens,
                "one_leaf_target_fixed_leaf_tokens": one_leaf_target_fixed_leaf_tokens,
                "items_total": len(rows),
                "completed_items": len(rows),
                "failed_items": 0,
                "active_items": 0,
                "pending_items": 0,
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )


def test_markov_parity_self_contained_report_is_plot_first_and_self_contained(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    simulation_root = tmp_path / "parity_root"

    rows = [
        _make_row(
            simulation_root,
            job_name="recoverable_matched_root_leaf16_a",
            scope_label="recoverable",
            claim_level="empirical_geometry",
            baseline_family="tree_neural",
            recipe_id="matched_root",
            fixed_leaf_tokens=16,
            test_root_mae_mean=0.640,
            test_leaf_mae_mean=1.220,
            test_merge_mae_mean=0.810,
            val_root_mae_mean=0.650,
        ),
        _make_row(
            simulation_root,
            job_name="recoverable_matched_root_leaf16_b",
            scope_label="recoverable",
            claim_level="empirical_geometry",
            baseline_family="tree_neural",
            recipe_id="matched_root",
            fixed_leaf_tokens=16,
            test_root_mae_mean=0.630,
            test_leaf_mae_mean=1.180,
            test_merge_mae_mean=0.790,
            val_root_mae_mean=0.645,
        ),
        _make_row(
            simulation_root,
            job_name="recoverable_matched_root_leaf128",
            scope_label="recoverable",
            claim_level="empirical_geometry",
            baseline_family="tree_neural",
            recipe_id="matched_root",
            fixed_leaf_tokens=128,
            test_root_mae_mean=0.615,
            test_leaf_mae_mean=1.050,
            test_merge_mae_mean=0.720,
            val_root_mae_mean=0.625,
        ),
        _make_row(
            simulation_root,
            job_name="recoverable_fairfno_leaf128",
            scope_label="recoverable",
            claim_level="empirical_geometry",
            baseline_family="tree_neural",
            recipe_id="fairfno_matched_root",
            fixed_leaf_tokens=128,
            test_root_mae_mean=0.605,
            test_leaf_mae_mean=1.000,
            test_merge_mae_mean=0.700,
            val_root_mae_mean=0.615,
        ),
        _make_row(
            simulation_root,
            job_name="recoverable_fno",
            scope_label="recoverable",
            claim_level="empirical_geometry",
            baseline_family="official_fno",
            recipe_id="fno_baseline",
            fixed_leaf_tokens=128,
            test_root_mae_mean=0.620,
            test_leaf_mae_mean=1.020,
            test_merge_mae_mean=0.740,
            val_root_mae_mean=0.630,
        ),
        _make_row(
            simulation_root,
            job_name="recoverable_fno_sumlen",
            scope_label="recoverable",
            claim_level="empirical_geometry",
            baseline_family="official_fno_sumlen",
            recipe_id="fno_baseline",
            fixed_leaf_tokens=128,
            test_root_mae_mean=0.600,
            test_leaf_mae_mean=0.990,
            test_merge_mae_mean=0.690,
            val_root_mae_mean=0.610,
        ),
        _make_row(
            simulation_root,
            job_name="recoverable_exact",
            scope_label="recoverable",
            claim_level="exact_collapse_candidate",
            baseline_family="tree_neural",
            recipe_id="exact_collapse_candidate",
            fixed_leaf_tokens=128,
            test_root_mae_mean=0.598,
            test_leaf_mae_mean=0.985,
            test_merge_mae_mean=0.688,
            val_root_mae_mean=0.608,
            strict_collapse_pass=True,
        ),
        _make_row(
            simulation_root,
            job_name="structural_matched_root",
            scope_label="structural",
            claim_level="empirical_geometry",
            baseline_family="tree_neural",
            recipe_id="matched_root",
            fixed_leaf_tokens=128,
            test_root_mae_mean=0.705,
            test_leaf_mae_mean=2.410,
            test_merge_mae_mean=1.305,
            val_root_mae_mean=0.715,
        ),
        _make_row(
            simulation_root,
            job_name="structural_fno",
            scope_label="structural",
            claim_level="empirical_geometry",
            baseline_family="official_fno",
            recipe_id="fno_baseline",
            fixed_leaf_tokens=128,
            test_root_mae_mean=0.695,
            test_leaf_mae_mean=2.320,
            test_merge_mae_mean=1.240,
            val_root_mae_mean=0.705,
        ),
        _make_row(
            simulation_root,
            job_name="structural_fno_sumlen",
            scope_label="structural",
            claim_level="empirical_geometry",
            baseline_family="official_fno_sumlen",
            recipe_id="fno_baseline",
            fixed_leaf_tokens=128,
            test_root_mae_mean=0.685,
            test_leaf_mae_mean=2.250,
            test_merge_mae_mean=1.210,
            val_root_mae_mean=0.695,
        ),
        _make_row(
            simulation_root,
            job_name="structural_exact",
            scope_label="structural",
            claim_level="exact_collapse_candidate",
            baseline_family="tree_neural",
            recipe_id="exact_collapse_candidate",
            fixed_leaf_tokens=128,
            test_root_mae_mean=0.684,
            test_leaf_mae_mean=2.240,
            test_merge_mae_mean=1.205,
            val_root_mae_mean=0.694,
            strict_collapse_pass=False,
        ),
    ]

    _write_simulation_root(simulation_root, rows=rows)

    output_dir = tmp_path / "self_contained_report"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_parity_self_contained.py",
            "--simulation-root",
            str(simulation_root),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    report_md = (output_dir / "report.md").read_text(encoding="utf-8")

    assert summary["report_kind"] == "markov_parity_self_contained"
    assert summary["data_contract"]["self_contained"] is True
    assert summary["coverage"]["train_doc_counts"] == [4096]
    assert summary["coverage"]["assumed_doc_tokens"] == 128
    assert summary["coverage"]["one_leaf_target_fixed_leaf_tokens"] == 128

    assert summary["palette"]["families"]["official_fno"] == "#0f766e"
    assert summary["palette"]["families"]["official_fno_sumlen"] == "#dc2626"
    assert summary["palette"]["families"]["tree_neural"] == "#2563eb"

    assert summary["empirical_by_scope"]["recoverable"]["4096"]["chart_style"] == "geometry"
    assert summary["empirical_by_scope"]["structural"]["4096"]["chart_style"] == "comparison"

    figure_inventory = summary["figure_inventory"]
    assert figure_inventory["Recoverable Quality Geometry @ train_docs=4096"]["chart_style"] == "geometry_triptych"
    assert figure_inventory["Recoverable Quality Comparison @ train_docs=4096"]["chart_style"] == "comparison_triptych"
    assert figure_inventory["Recoverable Runtime Geometry @ train_docs=4096"]["chart_style"] == "geometry_panel"
    assert figure_inventory["Structural Quality Comparison @ train_docs=4096"]["chart_style"] == "comparison_triptych"
    assert figure_inventory["Structural Runtime Comparison @ train_docs=4096"]["chart_style"] == "comparison_panel"

    for title in [
        "Recoverable Quality Overview",
        "Recoverable Runtime Overview",
        "Recoverable Quality Geometry @ train_docs=4096",
        "Recoverable Quality Comparison @ train_docs=4096",
        "Recoverable Runtime Geometry @ train_docs=4096",
        "Recoverable Exact Collapse vs Best FNO",
        "Structural Quality Comparison @ train_docs=4096",
        "Structural Runtime Comparison @ train_docs=4096",
        "Structural Exact Collapse vs Best FNO",
    ]:
        assert Path(figure_inventory[title]["path"]).exists()

    assert "|---" not in report_md
    assert "Recipe |" not in report_md
    assert "This report only uses artifacts already present inside the specified simulation root." in report_md
    assert "figure:" in report_md

    coverage = summary["row_figure_coverage"]
    assert set(coverage) == {str(row["job_name"]) for row in rows}
    assert coverage["recoverable_matched_root_leaf16_a"]["quality_figures"]
    assert coverage["recoverable_matched_root_leaf16_a"]["runtime_figures"]
    assert coverage["structural_exact"]["quality_figures"]
    assert coverage["structural_exact"]["runtime_figures"]

    cell_key = "recoverable::4096::empirical_geometry::matched_root::16"
    assert summary["n_rows_per_cell"][cell_key] == 2
    matching_cell = next(item for item in summary["cell_aggregates"] if item["cell_key"] == cell_key)
    assert sorted(matching_cell["raw_job_names"]) == [
        "recoverable_matched_root_leaf16_a",
        "recoverable_matched_root_leaf16_b",
    ]

    normalized_rows = {row["job_name"]: row for row in summary["normalized_rows"]}
    assert normalized_rows["recoverable_fno"]["elapsed_s"] == 124.0
    assert normalized_rows["recoverable_fno"]["runtime_data_mode"] == "resident"
    assert normalized_rows["recoverable_fno"]["steady_state_h2d_bytes"] == 0.0

    assert (output_dir / "report.pdf").exists()


def test_markov_parity_self_contained_report_aggregates_multiple_roots(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    root_a = tmp_path / "parity_root_a"
    root_b = tmp_path / "parity_root_b"

    rows_a = [
        _make_row(
            root_a,
            job_name="recoverable_matched_root_seed0",
            scope_label="recoverable",
            claim_level="empirical_geometry",
            baseline_family="tree_neural",
            recipe_id="matched_root",
            fixed_leaf_tokens=16,
            test_root_mae_mean=0.310,
            test_leaf_mae_mean=0.710,
            test_merge_mae_mean=0.510,
            val_root_mae_mean=0.320,
        ),
    ]
    rows_b = [
        _make_row(
            root_b,
            job_name="recoverable_matched_root_seed1",
            scope_label="recoverable",
            claim_level="empirical_geometry",
            baseline_family="tree_neural",
            recipe_id="matched_root",
            fixed_leaf_tokens=16,
            test_root_mae_mean=0.290,
            test_leaf_mae_mean=0.690,
            test_merge_mae_mean=0.490,
            val_root_mae_mean=0.300,
        ),
    ]
    _write_simulation_root(root_a, rows=rows_a, train_doc_counts=[4096])
    _write_simulation_root(root_b, rows=rows_b, train_doc_counts=[4096])

    output_dir = tmp_path / "combined_self_contained_report"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_parity_self_contained.py",
            "--simulation-root",
            str(root_a),
            "--simulation-root",
            str(root_b),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    report_md = (output_dir / "report.md").read_text(encoding="utf-8")

    assert summary["status"]["source_run_count"] == 2
    assert summary["source_files"]["simulation_roots"] == [str(root_a), str(root_b)]
    assert summary["status"]["completed_items"] == 2
    assert summary["coverage"]["train_doc_counts"] == [4096]
    assert summary["n_rows_per_cell"]["recoverable::4096::empirical_geometry::matched_root::16"] == 2
    assert len(summary["normalized_rows"]) == 2
    assert {
        row["source_simulation_root"]
        for row in summary["normalized_rows"]
    } == {str(root_a), str(root_b)}
    assert "specified simulation roots" in report_md
    assert "source runs: `2`" in report_md
    assert (output_dir / "report.pdf").exists()


def test_markov_parity_self_contained_report_distinguishes_local_target_and_weighting_variants(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    simulation_root = tmp_path / "lean_diag_root"

    count_only_row = _make_row(
        simulation_root,
        job_name="recoverable_r10_count_only_subset",
        scope_label="recoverable",
        claim_level="empirical_geometry",
        baseline_family="tree_neural",
        recipe_id="r10_local_20",
        fixed_leaf_tokens=16,
        test_root_mae_mean=0.610,
        test_leaf_mae_mean=0.700,
        test_merge_mae_mean=0.500,
        val_root_mae_mean=0.620,
    )
    count_only_row["leaf_supervision_kind"] = "count_only"
    count_only_row["internal_supervision_kind"] = "count_only"
    count_only_row["tree_local_weighting_mode"] = "subset_mean"
    count_only_row["local_law_c1_weight"] = 1.0 / 12.0
    count_only_row["local_law_c2_weight"] = 1.0 / 12.0
    count_only_row["local_law_c3_weight"] = 1.0 / 12.0
    count_only_row["optimization_root_weight"] = 0.75

    bounded_row = _make_row(
        simulation_root,
        job_name="recoverable_r10_bounded_hajek",
        scope_label="recoverable",
        claim_level="empirical_geometry",
        baseline_family="tree_neural",
        recipe_id="r10_local_20",
        fixed_leaf_tokens=16,
        test_root_mae_mean=0.210,
        test_leaf_mae_mean=0.400,
        test_merge_mae_mean=0.300,
        val_root_mae_mean=0.220,
    )
    bounded_row["leaf_supervision_kind"] = "bounded_full_sketch"
    bounded_row["internal_supervision_kind"] = "bounded_full_sketch"
    bounded_row["tree_local_weighting_mode"] = "fixed_k_hajek"
    bounded_row["local_law_c1_weight"] = 1.0 / 30.0
    bounded_row["local_law_c2_weight"] = 1.0 / 30.0
    bounded_row["local_law_c3_weight"] = 1.0 / 30.0
    bounded_row["optimization_root_weight"] = 0.90

    sweep_row = _make_row(
        simulation_root,
        job_name="recoverable_v4__tree_neural__train_4096__leaf_16__stage_r20_local_50__weight_sweep__lw10__c1_2__c3_1__seed_0",
        scope_label="recoverable",
        claim_level="empirical_geometry",
        baseline_family="tree_neural",
        recipe_id="r20_local_50",
        fixed_leaf_tokens=16,
        test_root_mae_mean=0.180,
        test_leaf_mae_mean=0.360,
        test_merge_mae_mean=0.260,
        val_root_mae_mean=0.190,
    )
    sweep_row["leaf_supervision_kind"] = "bounded_full_sketch"
    sweep_row["internal_supervision_kind"] = "bounded_full_sketch"
    sweep_row["tree_local_weighting_mode"] = "fixed_k_hajek"
    sweep_row["local_law_c1_weight"] = 1.0 / 30.0
    sweep_row["local_law_c2_weight"] = 1.0 / 30.0
    sweep_row["local_law_c3_weight"] = 1.0 / 30.0
    sweep_row["optimization_root_weight"] = 0.90

    _write_simulation_root(simulation_root, rows=[count_only_row, bounded_row, sweep_row], train_doc_counts=[4096])

    output_dir = tmp_path / "variant_report"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_parity_self_contained.py",
            "--simulation-root",
            str(simulation_root),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    normalized_rows = {row["job_name"]: row for row in summary["normalized_rows"]}

    assert normalized_rows["recoverable_r10_count_only_subset"]["comparison_label"] == "R10 local 20% [count-only | subset mean]"
    assert normalized_rows["recoverable_r10_bounded_hajek"]["comparison_label"] == "R10 local 20% [bounded sketch | Hajek | lw=0.10]"
    assert normalized_rows["recoverable_v4__tree_neural__train_4096__leaf_16__stage_r20_local_50__weight_sweep__lw10__c1_2__c3_1__seed_0"]["comparison_label"] == "R20 local 50% [bounded sketch | Hajek | lw=0.10, c1:c3=2:1]"
    cell_counts = summary["n_rows_per_cell"]
    assert cell_counts["recoverable::4096::empirical_geometry::r10_local_20__count_only__subset_mean::16"] == 1
    assert cell_counts["recoverable::4096::empirical_geometry::r10_local_20__bounded_full_sketch__fixed_k_hajek__lw10::16"] == 1
    assert cell_counts["recoverable::4096::empirical_geometry::r20_local_50__bounded_full_sketch__fixed_k_hajek__lw10__c1x2::16"] == 1


def test_markov_parity_self_contained_report_filters_quarantined_rows_from_headlines(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    simulation_root = tmp_path / "parity_root"

    current_row = _make_row(
        simulation_root,
        job_name="recoverable_current_tree",
        scope_label="recoverable",
        claim_level="empirical_geometry",
        baseline_family="tree_neural",
        recipe_id="matched_root",
        fixed_leaf_tokens=16,
        test_root_mae_mean=0.610,
        test_leaf_mae_mean=0.900,
        test_merge_mae_mean=0.700,
        val_root_mae_mean=0.620,
    )
    current_row["contract_status"] = "current"

    quarantined_row = _make_row(
        simulation_root,
        job_name="recoverable_legacy_tree",
        scope_label="recoverable",
        claim_level="empirical_geometry",
        baseline_family="tree_neural",
        recipe_id="matched_root",
        fixed_leaf_tokens=16,
        test_root_mae_mean=0.100,
        test_leaf_mae_mean=0.200,
        test_merge_mae_mean=0.300,
        val_root_mae_mean=0.110,
    )
    quarantined_row["contract_status"] = "legacy_quarantined"

    _write_simulation_root(
        simulation_root,
        rows=[current_row, quarantined_row],
        train_doc_counts=[4096],
    )

    output_dir = tmp_path / "self_contained_report"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_parity_self_contained.py",
            "--simulation-root",
            str(simulation_root),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["data_contract"]["headline_filters_quarantined_rows"] is True
    assert summary["status"]["headline_row_count"] == 1
    assert summary["status"]["quarantined_row_count"] == 1
    assert summary["coverage"]["contract_statuses"] == ["current", "legacy_quarantined"]
    assert len(summary["normalized_rows"]) == 2
    assert summary["n_rows_per_cell"]["recoverable::4096::empirical_geometry::matched_root::16"] == 1


def test_markov_parity_self_contained_report_includes_exact_collapse_repair_arms(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    simulation_root = tmp_path / "exact_collapse_repair_root"

    rows: list[dict[str, object]] = []
    for train_doc_count, fno_root, candidate_root, runtime_root, legacy_root in [
        (1024, 0.2920, 0.6520, 0.2920, 0.6540),
        (10240, 0.1338, 0.2214, 0.1338, 0.6440),
    ]:
        rows.extend(
            [
                _make_row(
                    simulation_root,
                    job_name=f"recoverable_fno_repair_{train_doc_count}__exact_collapse_repair_arm_official_fno",
                    scope_label="recoverable",
                    claim_level="empirical_geometry",
                    baseline_family="official_fno",
                    recipe_id="fno_baseline",
                    fixed_leaf_tokens=128,
                    train_doc_count=train_doc_count,
                    test_root_mae_mean=fno_root,
                    test_leaf_mae_mean=fno_root,
                    test_merge_mae_mean=0.0,
                    val_root_mae_mean=fno_root + 0.01,
                ),
                _make_row(
                    simulation_root,
                    job_name=f"recoverable_candidate_repair_{train_doc_count}__exact_collapse_repair_arm_exact_collapse_candidate",
                    scope_label="recoverable",
                    claim_level="exact_collapse_candidate",
                    baseline_family="tree_neural",
                    recipe_id="exact_collapse_candidate",
                    fixed_leaf_tokens=128,
                    train_doc_count=train_doc_count,
                    test_root_mae_mean=candidate_root,
                    test_leaf_mae_mean=candidate_root + 1.0,
                    test_merge_mae_mean=candidate_root + 0.5,
                    val_root_mae_mean=candidate_root + 0.01,
                ),
                _make_row(
                    simulation_root,
                    job_name=f"recoverable_runtime_repair_{train_doc_count}__exact_collapse_repair_arm_exact_collapse_runtime_match",
                    scope_label="recoverable",
                    claim_level="empirical_geometry",
                    baseline_family="tree_neural",
                    recipe_id="exact_collapse_runtime_match",
                    fixed_leaf_tokens=128,
                    train_doc_count=train_doc_count,
                    test_root_mae_mean=runtime_root,
                    test_leaf_mae_mean=runtime_root,
                    test_merge_mae_mean=0.0,
                    val_root_mae_mean=runtime_root + 0.01,
                ),
                _make_row(
                    simulation_root,
                    job_name=f"recoverable_legacy_repair_{train_doc_count}__exact_collapse_repair_arm_exact_collapse_legacy_control",
                    scope_label="recoverable",
                    claim_level="empirical_geometry",
                    baseline_family="tree_neural",
                    recipe_id="exact_collapse_legacy_control",
                    fixed_leaf_tokens=128,
                    train_doc_count=train_doc_count,
                    test_root_mae_mean=legacy_root,
                    test_leaf_mae_mean=legacy_root + 1.2,
                    test_merge_mae_mean=legacy_root + 0.7,
                    val_root_mae_mean=legacy_root + 0.01,
                ),
            ]
        )

    _write_simulation_root(simulation_root, rows=rows, train_doc_counts=[1024, 10240])

    output_dir = tmp_path / "repair_report"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_parity_self_contained.py",
            "--simulation-root",
            str(simulation_root),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    figure_inventory = summary["figure_inventory"]

    assert figure_inventory["Recoverable Exact Collapse Repair Arms"]["chart_style"] == "repair_panel"
    assert Path(figure_inventory["Recoverable Exact Collapse Repair Arms"]["path"]).exists()
    assert "recoverable_candidate_repair_1024__exact_collapse_repair_arm_exact_collapse_candidate" in figure_inventory["Recoverable Exact Collapse Repair Arms"]["job_names"]
    assert "recoverable_runtime_repair_10240__exact_collapse_repair_arm_exact_collapse_runtime_match" in figure_inventory["Recoverable Exact Collapse Repair Arms"]["job_names"]

    coverage = summary["row_figure_coverage"]
    assert "Recoverable Exact Collapse Repair Arms" in coverage["recoverable_candidate_repair_1024__exact_collapse_repair_arm_exact_collapse_candidate"]["quality_figures"]
    assert "Recoverable Exact Collapse Repair Arms" in coverage["recoverable_runtime_repair_10240__exact_collapse_repair_arm_exact_collapse_runtime_match"]["quality_figures"]


def test_markov_parity_self_contained_report_includes_full_local_laws_topology_4096(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    legacy_root = tmp_path / "legacy_topology_root"
    unified_root = tmp_path / "unified_topology_root"
    deeper_root = tmp_path / "deeper_topology_root"
    anchor_root = tmp_path / "anchor_root"

    legacy_rows: list[dict[str, object]] = []
    unified_rows: list[dict[str, object]] = []
    deeper_rows: list[dict[str, object]] = []
    for seed, fno128, tree128, tree64, tree32, tree16, tree8 in [
        (0, 0.0410, 0.0350, 0.0220, 0.0074, 0.0069, 0.0066),
        (1, 0.0420, 0.0360, 0.0230, 0.0070, 0.0067, 0.0064),
    ]:
        legacy_rows.extend(
            [
                _make_row(
                    legacy_root,
                    job_name=f"recoverable_legacy_tree_leaf128_seed{seed}",
                    scope_label="recoverable",
                    claim_level="empirical_geometry",
                    baseline_family="tree_neural",
                    recipe_id="full_local_laws_tree",
                    fixed_leaf_tokens=16,
                    train_doc_count=4096,
                    test_root_mae_mean=0.0890 + 0.001 * seed,
                    test_leaf_mae_mean=0.3200 + 0.01 * seed,
                    test_merge_mae_mean=0.2100 + 0.01 * seed,
                    val_root_mae_mean=0.0990 + 0.001 * seed,
                )
                | {
                    "study_axis": "full_local_laws_topology_4096",
                    "axis_value": "tree_neural_leaf128",
                    "config_label": "full_local_laws_tree__recoverable__leaf128",
                    "locked_tree_neural_config_label": "common_factorized_sketch_v1",
                },
            ]
        )
        unified_rows.extend(
            [
                _make_row(
                    unified_root,
                    job_name=f"recoverable_topology_fno_leaf128_seed{seed}",
                    scope_label="recoverable",
                    claim_level="empirical_geometry",
                    baseline_family="official_fno",
                    recipe_id="fno_baseline",
                    fixed_leaf_tokens=128,
                    train_doc_count=4096,
                    test_root_mae_mean=fno128,
                    test_leaf_mae_mean=fno128,
                    test_merge_mae_mean=0.0,
                    val_root_mae_mean=fno128 + 0.01,
                )
                | {
                    "study_axis": "unified_g_topology_4096",
                    "axis_value": "official_fno_leaf128",
                    "config_label": "fno_baseline__recoverable__leaf128",
                },
                _make_row(
                    unified_root,
                    job_name=f"recoverable_topology_tree_leaf128_seed{seed}",
                    scope_label="recoverable",
                    claim_level="empirical_geometry",
                    baseline_family="tree_neural",
                    recipe_id="unified_g_full_local_laws_tree",
                    fixed_leaf_tokens=128,
                    train_doc_count=4096,
                    test_root_mae_mean=tree128,
                    test_leaf_mae_mean=0.0310 + 0.001 * seed,
                    test_merge_mae_mean=0.0,
                    val_root_mae_mean=tree128 + 0.01,
                )
                | {
                    "study_axis": "unified_g_topology_4096",
                    "axis_value": "tree_neural_leaf128",
                    "config_label": "unified_g_full_local_laws_tree__recoverable__leaf128",
                    "locked_tree_neural_config_label": "unified_g_full_local_laws_v1",
                },
                _make_row(
                    unified_root,
                    job_name=f"recoverable_topology_tree_leaf64_seed{seed}",
                    scope_label="recoverable",
                    claim_level="empirical_geometry",
                    baseline_family="tree_neural",
                    recipe_id="unified_g_full_local_laws_tree",
                    fixed_leaf_tokens=64,
                    train_doc_count=4096,
                    test_root_mae_mean=tree64,
                    test_leaf_mae_mean=0.0210 + 0.001 * seed,
                    test_merge_mae_mean=0.0120 + 0.001 * seed,
                    val_root_mae_mean=tree64 + 0.01,
                )
                | {
                    "study_axis": "unified_g_topology_4096",
                    "axis_value": "tree_neural_leaf64",
                    "config_label": "unified_g_full_local_laws_tree__recoverable__leaf64",
                    "locked_tree_neural_config_label": "unified_g_full_local_laws_v1",
                },
                _make_row(
                    unified_root,
                    job_name=f"recoverable_topology_tree_leaf32_seed{seed}",
                    scope_label="recoverable",
                    claim_level="empirical_geometry",
                    baseline_family="tree_neural",
                    recipe_id="unified_g_full_local_laws_tree",
                    fixed_leaf_tokens=32,
                    train_doc_count=4096,
                    test_root_mae_mean=tree32,
                    test_leaf_mae_mean=0.0070 + 0.001 * seed,
                    test_merge_mae_mean=0.0060 + 0.001 * seed,
                    val_root_mae_mean=tree32 + 0.01,
                )
                | {
                    "study_axis": "unified_g_topology_4096",
                    "axis_value": "tree_neural_leaf32",
                    "config_label": "unified_g_full_local_laws_tree__recoverable__leaf32",
                    "locked_tree_neural_config_label": "unified_g_full_local_laws_v1",
                },
            ]
        )
        deeper_rows.extend(
            [
                _make_row(
                    deeper_root,
                    job_name=f"recoverable_deeper_tree_leaf16_seed{seed}",
                    scope_label="recoverable",
                    claim_level="empirical_geometry",
                    baseline_family="tree_neural",
                    recipe_id="unified_g_full_local_laws_tree",
                    fixed_leaf_tokens=16,
                    train_doc_count=4096,
                    test_root_mae_mean=tree16,
                    test_leaf_mae_mean=0.0100 + 0.001 * seed,
                    test_merge_mae_mean=0.0130 + 0.001 * seed,
                    val_root_mae_mean=tree16 + 0.01,
                )
                | {
                    "study_axis": "unified_g_topology_4096",
                    "axis_value": "tree_neural_leaf16",
                    "config_label": "unified_g_full_local_laws_tree__recoverable__leaf16",
                    "locked_tree_neural_config_label": "unified_g_full_local_laws_v1",
                },
                _make_row(
                    deeper_root,
                    job_name=f"recoverable_deeper_tree_leaf8_seed{seed}",
                    scope_label="recoverable",
                    claim_level="empirical_geometry",
                    baseline_family="tree_neural",
                    recipe_id="unified_g_full_local_laws_tree",
                    fixed_leaf_tokens=8,
                    train_doc_count=4096,
                    test_root_mae_mean=tree8,
                    test_leaf_mae_mean=0.0015 + 0.0005 * seed,
                    test_merge_mae_mean=0.0060 + 0.0005 * seed,
                    val_root_mae_mean=tree8 + 0.01,
                )
                | {
                    "study_axis": "unified_g_topology_4096",
                    "axis_value": "tree_neural_leaf8",
                    "config_label": "unified_g_full_local_laws_tree__recoverable__leaf8",
                    "locked_tree_neural_config_label": "unified_g_full_local_laws_v1",
                },
            ]
        )

    anchor_rows = [
        _make_row(
            anchor_root,
            job_name=f"recoverable_anchor_candidate_4096_seed{seed}__exact_collapse_repair_arm_exact_collapse_candidate",
            scope_label="recoverable",
            claim_level="exact_collapse_candidate",
            baseline_family="tree_neural",
            recipe_id="exact_collapse_candidate",
            fixed_leaf_tokens=128,
            train_doc_count=4096,
            test_root_mae_mean=value,
            test_leaf_mae_mean=value,
            test_merge_mae_mean=0.0,
            val_root_mae_mean=value + 0.01,
        )
        for seed, value in [(0, 0.0410), (1, 0.1094)]
    ]

    _write_simulation_root(legacy_root, rows=legacy_rows, train_doc_counts=[4096])
    _write_simulation_root(unified_root, rows=unified_rows, train_doc_counts=[4096])
    _write_simulation_root(deeper_root, rows=deeper_rows, train_doc_counts=[4096])
    _write_simulation_root(anchor_root, rows=anchor_rows, train_doc_counts=[4096])

    output_dir = tmp_path / "topology_report"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_parity_self_contained.py",
            "--simulation-root",
            str(legacy_root),
            "--simulation-root",
            str(unified_root),
            "--simulation-root",
            str(deeper_root),
            "--simulation-root",
            str(anchor_root),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    figure_inventory = summary["figure_inventory"]

    assert (
        figure_inventory["Recoverable Full Local Laws Topology @ train_docs=4096"]["chart_style"]
        == "topology_curve"
    )
    assert (
        figure_inventory["Recoverable Topology Ladder Diagnostics @ train_docs=4096"]["chart_style"]
        == "metric_bars"
    )
    assert Path(
        figure_inventory["Recoverable Full Local Laws Topology @ train_docs=4096"]["path"]
    ).exists()
    assert Path(
        figure_inventory["Recoverable Topology Ladder Diagnostics @ train_docs=4096"]["path"]
    ).exists()
    assert (
        "recoverable_deeper_tree_leaf8_seed0"
        in figure_inventory["Recoverable Topology Ladder Diagnostics @ train_docs=4096"]["job_names"]
    )
    assert (
        "recoverable_anchor_candidate_4096_seed0__exact_collapse_repair_arm_exact_collapse_candidate"
        in figure_inventory["Recoverable Full Local Laws Topology @ train_docs=4096"]["job_names"]
    )
    assert (
        "recoverable_legacy_tree_leaf128_seed0"
        not in figure_inventory["Recoverable Full Local Laws Topology @ train_docs=4096"]["job_names"]
    )
    assert (
        "recoverable_topology_tree_leaf128_seed0"
        in figure_inventory["Recoverable Full Local Laws Topology @ train_docs=4096"]["job_names"]
    )
    assert (
        "recoverable_topology_tree_leaf64_seed0"
        in figure_inventory["Recoverable Full Local Laws Topology @ train_docs=4096"]["job_names"]
    )
    assert (
        "recoverable_deeper_tree_leaf16_seed0"
        in figure_inventory["Recoverable Full Local Laws Topology @ train_docs=4096"]["job_names"]
    )
    assert (
        "recoverable_topology_tree_leaf16_seed0"
        not in figure_inventory["Recoverable Full Local Laws Topology @ train_docs=4096"]["job_names"]
    )
