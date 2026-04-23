from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from src.ctreepo.sim.core.markov_alignment_validation import (
    _bundle_consistency_check,
    _package_vs_parity_separation_check,
    _rung_nestedness_check,
    _strict_collapse_readiness_check,
)
from src.ctreepo.sim.core.markov_tree_fno_validation import (
    _duplicate_aggregate_check,
    _legacy_current_mixing_check,
    _score_contract_check,
    build_markov_tree_fno_validation_report,
)
from tests.ctreepo.test_simulation_expectations import (
    _full_doc_diagnostics_payload,
    _full_doc_ladder_payload,
)


def _write_diagnostics_root(path: Path) -> Path:
    payload = _full_doc_diagnostics_payload()
    runs_dir = path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for index, run in enumerate(list(payload.get("runs") or [])):
        run_path = runs_dir / f"run_{index}.json"
        run_path.write_text(json.dumps(run, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_prepared_metadata(
    root: Path,
    *,
    signature: str,
    counts: list[int],
    train_signature: str = "train-fixed",
    val_signature: str = "val-fixed",
    test_signature: str = "test-fixed",
) -> Path:
    prepared_dir = root / f"prepared_{signature}"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    (prepared_dir / "metadata.json").write_text(
        json.dumps(
            {
                "train_prefix_counts": list(counts),
                "train_prefix_signatures": {
                    str(int(count)): f"{signature}-{int(count)}" for count in counts
                },
                "train_corpus_signature": train_signature,
                "val_corpus_signature": val_signature,
                "test_corpus_signature": test_signature,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return prepared_dir


def test_markov_tree_fno_validation_report_passes_on_matched_fixture(
    tmp_path: Path,
) -> None:
    diagnostics_root = _write_diagnostics_root(tmp_path / "diagnostics")
    ladder_json = tmp_path / "ladder.json"
    ladder_json.write_text(
        json.dumps(_full_doc_ladder_payload(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    report = build_markov_tree_fno_validation_report(
        diagnostics_root=diagnostics_root,
        ladder_json=ladder_json,
        run_lean_build=False,
    )

    assert int(report.summary["n_fail"]) == 0
    checks = {check.name: check.status for check in report.checks}
    assert checks["aggregate_grouping"] == "pass"
    assert checks["score_contract"] == "pass"
    assert checks["official_fno_provenance"] == "pass"
    assert checks["current_tree_neural_semantics"] == "warn"
    assert checks["ladder_reference_reproduction_pairing"] == "pass"


def test_markov_tree_fno_validation_report_fails_on_mismatched_ladder_pair(
    tmp_path: Path,
) -> None:
    diagnostics_root = _write_diagnostics_root(tmp_path / "diagnostics")
    ladder_json = tmp_path / "ladder_bad.json"
    ladder_json.write_text(
        json.dumps(
            _full_doc_ladder_payload(mismatch=True),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = build_markov_tree_fno_validation_report(
        diagnostics_root=diagnostics_root,
        ladder_json=ladder_json,
        run_lean_build=False,
    )

    checks = {check.name: check.status for check in report.checks}
    assert checks["ladder_reference_reproduction_pairing"] == "fail"
    assert int(report.summary["n_fail"]) >= 1


def test_markov_tree_fno_validation_ignores_stale_nested_shard_summaries(
    tmp_path: Path,
) -> None:
    diagnostics_root = _write_diagnostics_root(tmp_path / "diagnostics")
    stale_summary = {
        "simulation": "markov_full_doc_anchor_diagnostics",
        "runs": [
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural",
                "seed": 0,
                "train_doc_count": 1024,
                "n_regimes": 4,
                "segment_density_band": "",
                "segment_min": 0,
                "segment_max": 0,
                "bundle_source": "/tmp/legacy_bundle.json",
                "train_corpus_signature": "train-legacy",
                "val_corpus_signature": "val-legacy",
                "test_corpus_signature": "test-legacy",
                "test_root_mae": 0.3,
                "test_exact_match_rate": 0.7,
                "test_c2_idempotence_mae": 0.1,
            }
        ],
    }
    stale_dir = diagnostics_root / "jobs" / "stale_job"
    stale_dir.mkdir(parents=True, exist_ok=True)
    (stale_dir / "summary.json").write_text(
        json.dumps(stale_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    report = build_markov_tree_fno_validation_report(
        diagnostics_root=diagnostics_root,
        run_lean_build=False,
    )

    assert int(report.expectation_report["summary"]["n_fail"]) == 0


def test_legacy_current_mixing_check_flags_shared_labels() -> None:
    check = _legacy_current_mixing_check(
        {
            "runs": [
                {
                    "baseline_family": "tree_neural_c2",
                    "comparison_semantics": "current",
                    "comparison_semantics_label": "shared_label",
                    "legacy_semantics": False,
                },
                {
                    "baseline_family": "tree_neural",
                    "comparison_semantics": "legacy",
                    "comparison_semantics_label": "shared_label",
                    "legacy_semantics": True,
                },
            ]
        }
    )
    assert check.status == "fail"


def test_score_contract_check_fails_when_primary_metric_is_missing() -> None:
    payload = _full_doc_diagnostics_payload()
    payload.pop("primary_report_metric", None)
    check = _score_contract_check(payload)
    assert check.status == "fail"
    assert "primary_report_metric" in check.details["mismatches"]


def test_markov_tree_fno_validation_allows_study_metadata_without_duplicate_failures(
    tmp_path: Path,
) -> None:
    diagnostics_root = tmp_path / "diagnostics"
    runs_dir = diagnostics_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    base_run = {
        "benchmark": "recoverable_v4",
        "cell_id": "recoverable_v4",
        "baseline_family": "tree_neural",
        "train_doc_count": 10240,
        "n_regimes": 4,
        "segment_density_band": "",
        "segment_min": 0,
        "segment_max": 0,
        "parameterization": "formal_local_law_weight",
        "optimization_root_weight": 0.7,
        "local_law_c1_weight": 0.1,
        "local_law_c2_weight": 0.1,
        "local_law_c3_weight": 0.1,
        "task_objective_weight_source": "configured_objective_builder",
        "objective_weights_active": True,
        "c2_metric_kind": "score_drift",
        "semantics_version": "tree_neural_objective_v2",
        "comparison_semantics": "current",
        "comparison_semantics_label": "current(score_drift)",
        "study_name": "leaf_geometry",
        "study_axis": "fixed_leaf_tokens",
        "locked_tree_neural_config_label": "cfg_a",
        "selection_metric": "val_root_mae_mean",
        "bundle_source": "/tmp/recoverable_bundle.json",
        "train_corpus_signature": "train-fixed",
        "val_corpus_signature": "val-fixed",
        "test_corpus_signature": "test-fixed",
        "train_root_mae": 0.2,
        "val_root_mae": 0.1,
        "train_exact_match_rate": 0.8,
        "val_exact_match_rate": 0.85,
        "test_exact_match_rate": 0.9,
        "test_c2_idempotence_mae": 0.02,
    }
    index = 0
    for fixed_leaf_tokens, seed, test_root_mae in (
        (8, 0, 0.06),
        (8, 1, 0.05),
        (16, 0, 0.04),
        (16, 1, 0.03),
    ):
        run = {
            **base_run,
            "seed": int(seed),
            "fixed_leaf_tokens": int(fixed_leaf_tokens),
            "axis_value": str(int(fixed_leaf_tokens)),
            "test_root_mae": float(test_root_mae),
        }
        (runs_dir / f"run_{index}.json").write_text(
            json.dumps(run, indent=2, sort_keys=True), encoding="utf-8"
        )
        index += 1

    report = build_markov_tree_fno_validation_report(
        diagnostics_root=diagnostics_root,
        run_lean_build=False,
    )
    checks = {check.name: check.status for check in report.checks}
    assert checks["aggregate_grouping"] == "pass"
    assert checks["partial_coverage"] == "pass"


def test_duplicate_aggregate_check_distinguishes_parity_tree_metadata() -> None:
    check = _duplicate_aggregate_check(
        {
            "aggregate_rows": [
                {
                    "benchmark": "recoverable_v4",
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 10240,
                    "fixed_leaf_tokens": 16,
                    "comparison_semantics_label": "current(score_drift)",
                    "parameterization": "formal_local_law_weight",
                    "optimization_root_weight": 0.7,
                    "local_law_c1_weight": 0.1,
                    "local_law_c2_weight": 0.1,
                    "local_law_c3_weight": 0.1,
                    "task_objective_weight_source": "configured_objective_builder",
                    "c2_metric_kind": "score_drift",
                    "backend_name": "tree_neural_fno",
                    "backend_package": "neuraloperator",
                    "backend_version": "1.0",
                    "operator_class": "FNOCountSketch",
                    "operator_evidence_status": "APPROX_AUDITED",
                    "theorem_relevance": True,
                    "objective_weights_active": True,
                    "tree_root_supervision_kind": "mse",
                    "tree_leaf_fno_width": 64,
                    "tree_leaf_fno_n_modes": 8,
                    "tree_leaf_fno_n_layers": 2,
                    "tree_aux_doc_sequence_fraction": 0.0,
                    "config_label": "default",
                    "tuning_stage": "",
                    "study_name": "",
                    "study_axis": "",
                    "axis_value": "",
                    "locked_tree_neural_config_label": "",
                    "selection_metric": "",
                },
                {
                    "benchmark": "recoverable_v4",
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 10240,
                    "fixed_leaf_tokens": 16,
                    "comparison_semantics_label": "current(score_drift)",
                    "parameterization": "formal_local_law_weight",
                    "optimization_root_weight": 0.7,
                    "local_law_c1_weight": 0.1,
                    "local_law_c2_weight": 0.1,
                    "local_law_c3_weight": 0.1,
                    "task_objective_weight_source": "configured_objective_builder",
                    "c2_metric_kind": "score_drift",
                    "backend_name": "tree_neural_fno",
                    "backend_package": "neuraloperator",
                    "backend_version": "1.0",
                    "operator_class": "FNOCountSketch",
                    "operator_evidence_status": "APPROX_AUDITED",
                    "theorem_relevance": True,
                    "objective_weights_active": True,
                    "tree_root_supervision_kind": "count_ce",
                    "tree_leaf_fno_width": 128,
                    "tree_leaf_fno_n_modes": 8,
                    "tree_leaf_fno_n_layers": 4,
                    "tree_aux_doc_sequence_fraction": 0.0,
                    "config_label": "fair_fno_v1",
                    "tuning_stage": "",
                    "study_name": "",
                    "study_axis": "",
                    "axis_value": "",
                    "locked_tree_neural_config_label": "",
                    "selection_metric": "",
                },
            ]
        }
    )
    assert check.status == "pass"


def test_duplicate_aggregate_check_distinguishes_aux_upper_bound_metadata() -> None:
    check = _duplicate_aggregate_check(
        {
            "aggregate_rows": [
                {
                    "benchmark": "recoverable_v4",
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 10240,
                    "fixed_leaf_tokens": 16,
                    "comparison_semantics_label": "current(score_drift)",
                    "parameterization": "formal_local_law_weight",
                    "optimization_root_weight": 0.7,
                    "local_law_c1_weight": 0.1,
                    "local_law_c2_weight": 0.1,
                    "local_law_c3_weight": 0.1,
                    "task_objective_weight_source": "configured_objective_builder",
                    "c2_metric_kind": "score_drift",
                    "backend_name": "tree_neural_fno",
                    "backend_package": "neuraloperator",
                    "backend_version": "1.0",
                    "operator_class": "FNOCountSketch",
                    "operator_evidence_status": "APPROX_AUDITED",
                    "theorem_relevance": True,
                    "objective_weights_active": True,
                    "tree_root_supervision_kind": "count_ce",
                    "tree_leaf_fno_width": 128,
                    "tree_leaf_fno_n_modes": 8,
                    "tree_leaf_fno_n_layers": 4,
                    "tree_aux_doc_sequence_fraction": 0.0,
                    "config_label": "fair_fno_v1",
                    "tuning_stage": "",
                    "study_name": "",
                    "study_axis": "",
                    "axis_value": "",
                    "locked_tree_neural_config_label": "",
                    "selection_metric": "",
                },
                {
                    "benchmark": "recoverable_v4",
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 10240,
                    "fixed_leaf_tokens": 16,
                    "comparison_semantics_label": "current(score_drift)",
                    "parameterization": "formal_local_law_weight",
                    "optimization_root_weight": 0.7,
                    "local_law_c1_weight": 0.1,
                    "local_law_c2_weight": 0.1,
                    "local_law_c3_weight": 0.1,
                    "task_objective_weight_source": "configured_objective_builder",
                    "c2_metric_kind": "score_drift",
                    "backend_name": "tree_neural_fno",
                    "backend_package": "neuraloperator",
                    "backend_version": "1.0",
                    "operator_class": "FNOCountSketch",
                    "operator_evidence_status": "APPROX_AUDITED",
                    "theorem_relevance": True,
                    "objective_weights_active": True,
                    "tree_root_supervision_kind": "count_ce",
                    "tree_leaf_fno_width": 128,
                    "tree_leaf_fno_n_modes": 8,
                    "tree_leaf_fno_n_layers": 4,
                    "tree_aux_doc_sequence_fraction": 0.25,
                    "config_label": "fair_fno_v1_aux25",
                    "tuning_stage": "upper_bound",
                    "study_name": "",
                    "study_axis": "",
                    "axis_value": "",
                    "locked_tree_neural_config_label": "",
                    "selection_metric": "",
                },
            ]
        }
    )
    assert check.status == "pass"


def test_validate_markov_tree_fno_alignment_cli(tmp_path: Path) -> None:
    diagnostics_root = _write_diagnostics_root(tmp_path / "diagnostics")
    ladder_json = tmp_path / "ladder.json"
    ladder_json.write_text(
        json.dumps(_full_doc_ladder_payload(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    script = Path("/home/mlinegar/ThinkingTrees/scripts/validate_markov_tree_fno_alignment.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--diagnostics-root",
            str(diagnostics_root),
            "--ladder-json",
            str(ladder_json),
            "--no-run-lean-build",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd="/home/mlinegar/ThinkingTrees",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (diagnostics_root / "markov_tree_fno_validation.json").exists()
    assert (diagnostics_root / "markov_tree_fno_validation.md").exists()


def test_rung_nestedness_and_bundle_consistency_checks_pass_on_canonical_metadata(
    tmp_path: Path,
) -> None:
    prepared_root = tmp_path / "prepared"
    _write_prepared_metadata(
        prepared_root,
        signature="sig",
        counts=[1024, 4096, 10240],
    )
    payload = {
        "runs": [
            {
                "benchmark": "recoverable_v4",
                "baseline_family": "tree_neural",
                "train_doc_count": count,
                "seed": 0,
                "bundle_source": f"/tmp/recoverable_bundle.json::train_prefix_{count}",
                "train_corpus_signature": f"train-{count}",
                "val_corpus_signature": "val-fixed",
                "test_corpus_signature": "test-fixed",
                "config": {
                    "prepared_data_root": str(prepared_root),
                    "prepared_data_signature": "sig",
                },
            }
            for count in (1024, 4096, 10240)
        ]
    }

    rung_check = _rung_nestedness_check(
        payload,
        canonical_train_ladder=(1024, 4096, 10240),
    )
    bundle_check = _bundle_consistency_check(
        payload,
        canonical_train_ladder=(1024, 4096, 10240),
    )

    assert rung_check.status == "pass"
    assert bundle_check.status == "pass"
    assert rung_check.details["train_prefix_counts"] == [1024, 4096, 10240]
    assert bundle_check.details["val_corpus_signature"]["recoverable_v4@10240"] == "val-fixed"


def test_rung_nestedness_check_fails_on_mismatched_prefix_ladder(tmp_path: Path) -> None:
    prepared_root = tmp_path / "prepared"
    _write_prepared_metadata(
        prepared_root,
        signature="sig_bad",
        counts=[1024, 2048, 4096],
    )
    payload = {
        "runs": [
            {
                "benchmark": "recoverable_v4",
                "baseline_family": "tree_neural",
                "train_doc_count": 4096,
                "seed": 0,
                "config": {
                    "prepared_data_root": str(prepared_root),
                    "prepared_data_signature": "sig_bad",
                },
            }
        ]
    }

    check = _rung_nestedness_check(
        payload,
        canonical_train_ladder=(1024, 4096, 10240),
    )

    assert check.status == "fail"
    assert check.details["sample_bad_rows"][0]["reason"] == "canonical_ladder_mismatch"


def test_package_vs_parity_separation_and_strict_collapse_readiness_checks() -> None:
    separation = _package_vs_parity_separation_check(
        {
            "rows": [
                {
                    "source_kind": "supervision_recovery_parity_grid",
                    "claim_level": "empirical_geometry",
                }
            ]
        }
    )
    assert separation.status == "fail"

    readiness_fail = _strict_collapse_readiness_check(
        [
            {
                "rows": [
                    {
                        "claim_level": "exact_collapse_candidate",
                        "scope_label": "recoverable",
                        "state": "completed",
                        "strict_collapse_pass": False,
                        "reference_bundle_source": "",
                        "train_prefix_signatures": {},
                        "config_diff_vs_official_fno": {
                            "local_law_weight": {"expected": 0.0, "actual": 0.8}
                        },
                    },
                    {
                        "claim_level": "exact_collapse_candidate",
                        "scope_label": "recoverable",
                        "state": "completed",
                        "strict_collapse_pass": False,
                        "reference_bundle_source": "",
                        "train_prefix_signatures": {},
                        "config_diff_vs_official_fno": {},
                    }
                ]
            }
        ]
    )
    readiness_pass = _strict_collapse_readiness_check(
        [
            {
                "rows": [
                    {
                        "claim_level": "exact_collapse_candidate",
                        "scope_label": "recoverable",
                        "state": "completed",
                        "strict_collapse_pass": True,
                        "config_diff_vs_official_fno": {},
                    }
                ]
            }
        ]
    )

    assert readiness_fail.status == "fail"
    assert readiness_pass.status == "pass"
