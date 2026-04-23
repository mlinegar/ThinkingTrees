from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (
    ORACLE_BUDGET_STUDY_NAME,
    _budget_manifest_metadata,
    _payload_from_saved_runs,
)
from src.ctreepo.sim.core.markov_alignment_validation import (
    _budget_manifest_accounting_check,
    _comparable_surface_drift_check,
    build_markov_alignment_audit_report,
)
from src.ctreepo.sim.core.markov_changepoint_ops_count import (
    OPSCountConfig,
    _build_objective_summary,
    build_budgeted_train_supervision_manifest,
    build_markov_changepoint_ops_count_data_bundle,
)
from tests.ctreepo.test_simulation_expectations import (
    _full_doc_diagnostics_payload,
    _full_doc_ladder_payload,
    _full_tree_ipw_grid_payload,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _budget_augmented_diagnostics_payload() -> dict:
    base_runs = list(_full_doc_diagnostics_payload().get("runs") or [])
    cfg = OPSCountConfig(
        train_docs=4,
        val_docs=0,
        test_docs=0,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=16,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        seed=0,
        data_seed=0,
        model_seed=0,
        local_law_weight=0.3,
        c1_relative_weight=0.0,
        c2_relative_weight=1.0,
        c3_relative_weight=0.0,
        schedule_consistency_weight=0.0,
        budget_total_calls=4,
        full_doc_budget_share=0.5,
        doc_consumption_mode="root_only",
        local_split_mode="balanced",
        local_allocation_policy="breadth_first",
        tree_root_supervision_kind="count_ce",
        tree_leaf_fno_width=64,
        tree_leaf_fno_n_modes=8,
        tree_leaf_fno_n_layers=4,
        doc_sequence_train_fraction=0.0,
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(cfg)
    manifest = build_budgeted_train_supervision_manifest(
        docs=tuple(bundle.train_docs),
        config=cfg,
        baseline_family="tree_neural_c2",
        seed=0,
    )
    assert manifest is not None
    budget_metadata = _budget_manifest_metadata(manifest)
    objective = dict(_build_objective_summary(cfg))
    base_runs.append(
        {
            "benchmark": "recoverable_v4",
            "cell_id": "recoverable_v4",
            "baseline_family": "tree_neural_c2",
            "seed": 0,
            "train_doc_count": 10240,
            "n_regimes": 4,
            "segment_density_band": "",
            "segment_min": 0,
            "segment_max": 0,
            "bundle_source": "/tmp/recoverable_budget_bundle.json",
            "train_corpus_signature": "train-budget",
            "val_corpus_signature": "val-fixed",
            "test_corpus_signature": "test-fixed",
            "test_root_mae": 0.14,
            "test_exact_match_rate": 0.82,
            "test_c2_idempotence_mae": 0.03,
            "config": {
                "local_law_weight": 0.3,
                "c1_relative_weight": 0.0,
                "c2_relative_weight": 1.0,
                "c3_relative_weight": 0.0,
                "schedule_consistency_weight": 0.0,
                "tree_root_supervision_kind": "count_ce",
                "doc_sequence_train_fraction": 0.0,
                "tree_leaf_fno_width": 64,
                "tree_leaf_fno_n_modes": 8,
                "tree_leaf_fno_n_layers": 4,
                "fixed_leaf_tokens": 16,
                "budget_total_calls": int(budget_metadata["budget_total_calls"]),
                "budget_total_calls_per_doc": float(
                    budget_metadata["budget_total_calls_per_doc"]
                ),
                "full_doc_budget_share": float(
                    budget_metadata["full_doc_budget_share"]
                ),
                "doc_consumption_mode": str(
                    budget_metadata["doc_consumption_mode"]
                ),
                "local_split_mode": str(budget_metadata["local_split_mode"]),
                "local_allocation_policy": str(
                    budget_metadata["local_allocation_policy"]
                ),
                "resolved_objective": objective,
            },
            "resolved_objective": objective,
            "parameterization": str(objective["parameterization"]),
            "weighting_scheme": str(objective["weighting_scheme"]),
            "optimization_root_weight": float(objective["optimization_root_weight"]),
            "local_law_c1_weight": float(objective["local_law_c1_weight"]),
            "local_law_c2_weight": float(objective["local_law_c2_weight"]),
            "local_law_c3_weight": float(objective["local_law_c3_weight"]),
            "task_objective_weight_source": str(
                objective["task_objective_weight_source"]
            ),
            "proxy_schedule_consistency_weight": float(
                objective["proxy_schedule_consistency_weight"]
            ),
            "theorem_terms": list(objective["theorem_terms"]),
            "proxy_terms": list(objective["proxy_terms"]),
            "formal_notes": str(objective["formal_notes"]),
            "objective_weights_active": True,
            "c2_metric_kind": "score_drift",
            "c2_proxy_metric_kind": "state_replay_mse",
            "semantics_version": "tree_neural_objective_v2",
            "comparison_semantics": "current",
            "comparison_semantics_label": "tree_neural_objective_v2",
            "legacy_semantics": False,
            "backend_name": "tree_neural_neuraloperator",
            "backend_package": "neuraloperator",
            "backend_version": "2.0.0",
            "operator_class": "src.ctreepo.sim.core.markov_neural_operator_baselines.FNOCountSketch",
            "operator_evidence_status": "APPROX_AUDITED",
            "theorem_relevance": True,
            "tree_root_supervision_kind": "count_ce",
            "tree_leaf_fno_width": 64,
            "tree_leaf_fno_n_modes": 8,
            "tree_leaf_fno_n_layers": 4,
            "tree_aux_doc_sequence_fraction": 0.0,
            "budget_total_calls": int(budget_metadata["budget_total_calls"]),
            "budget_total_calls_per_doc": float(
                budget_metadata["budget_total_calls_per_doc"]
            ),
            "budget_total_calls_used": int(budget_metadata["budget_total_calls_used"]),
            "budget_utilization": float(budget_metadata["budget_utilization"]),
            "full_doc_budget_share": float(budget_metadata["full_doc_budget_share"]),
            "full_doc_calls_requested": int(
                budget_metadata["full_doc_calls_requested"]
            ),
            "full_doc_calls_total": int(budget_metadata["full_doc_calls_total"]),
            "local_calls_requested": int(budget_metadata["local_calls_requested"]),
            "local_calls_total": int(budget_metadata["local_calls_total"]),
            "doc_consumption_mode": str(budget_metadata["doc_consumption_mode"]),
            "local_split_mode": str(budget_metadata["local_split_mode"]),
            "local_allocation_policy": str(
                budget_metadata["local_allocation_policy"]
            ),
            "effective_full_doc_mass_total": float(
                budget_metadata["effective_full_doc_mass_total"]
            ),
            "effective_full_doc_mass_per_doc": float(
                budget_metadata["effective_full_doc_mass_per_doc"]
            ),
            "document_mass_share": float(budget_metadata["document_mass_share"]),
            "leaf_mass_share": float(budget_metadata["leaf_mass_share"]),
            "internal_mass_share": float(budget_metadata["internal_mass_share"]),
            "document_call_share": float(budget_metadata["document_call_share"]),
            "leaf_call_share": float(budget_metadata["leaf_call_share"]),
            "internal_call_share": float(budget_metadata["internal_call_share"]),
            "doc_touch_rate": float(budget_metadata["doc_touch_rate"]),
            "mean_labels_per_touched_doc": float(
                budget_metadata["mean_labels_per_touched_doc"]
            ),
            "touched_docs_total": int(budget_metadata["touched_docs_total"]),
            "budget_manifest": dict(budget_metadata),
            "study_name": ORACLE_BUDGET_STUDY_NAME,
            "selection_metric": "val_root_mae_mean",
        }
    )
    return _payload_from_saved_runs(runs=base_runs)


def _write_diagnostics_root(path: Path) -> Path:
    payload = _budget_augmented_diagnostics_payload()
    runs_dir = path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for index, run in enumerate(list(payload.get("runs") or [])):
        run_path = runs_dir / f"run_{index}.json"
        run_path.write_text(json.dumps(run, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_markov_alignment_audit_report_passes_on_matched_fixture(
    tmp_path: Path,
) -> None:
    diagnostics_root = _write_diagnostics_root(tmp_path / "diagnostics")
    ladder_json = _write_json(tmp_path / "ladder.json", _full_doc_ladder_payload())
    full_tree_root = tmp_path / "full_tree"
    _write_json(full_tree_root / "summary.json", _full_tree_ipw_grid_payload())

    report = build_markov_alignment_audit_report(
        diagnostics_root=diagnostics_root,
        full_tree_ipw_root=full_tree_root,
        ladder_json=ladder_json,
        run_lean_build=False,
    )

    checks = {check.name: check.status for check in report.checks}
    assert checks["theorem_proxy_labeling"] == "pass"
    assert checks["budget_manifest_accounting"] == "pass"
    assert checks["comparable_surface_drift"] == "pass"
    assert checks["full_tree_ipw_semantics"] == "pass"
    assert checks["full_tree_ipw_endpoints"] == "pass"
    assert checks["objective_surface_distinction"] == "pass"


def test_comparable_surface_drift_check_flags_mixed_family_drift() -> None:
    check = _comparable_surface_drift_check(
        {
            "runs": [
                {
                    "benchmark": "recoverable_v4",
                    "cell_id": "recoverable_v4",
                    "baseline_family": "official_fno",
                    "train_doc_count": 1024,
                    "fixed_leaf_tokens": 128,
                    "comparison_mode": "comparable",
                    "comparison_surface_diff": {},
                },
                {
                    "benchmark": "recoverable_v4",
                    "cell_id": "recoverable_v4",
                    "baseline_family": "tree_neural",
                    "train_doc_count": 1024,
                    "fixed_leaf_tokens": 128,
                    "comparison_mode": "comparable",
                    "comparison_surface_diff": {
                        "state_dim": {"expected": 128, "actual": 256}
                    },
                },
            ]
        }
    )

    assert check.status == "fail"
    assert check.details["n_bad_groups"] == 1


def test_markov_alignment_audit_flags_nonrandom_budget_sampling_scheme(
) -> None:
    payload = _budget_augmented_diagnostics_payload()
    target_run = deepcopy(
        next(
            row
            for row in list(payload.get("runs") or [])
            if int(row.get("budget_total_calls", 0)) > 0
            or float(row.get("budget_total_calls_per_doc", 0.0)) > 0.0
            or str(row.get("study_name", "")) == ORACLE_BUDGET_STUDY_NAME
        )
    )
    target_run["sampling_scheme"] = "deterministic_span_order"
    target_run["budget_manifest"]["sampling_scheme"] = "deterministic_span_order"

    accounting_check = _budget_manifest_accounting_check({"runs": [target_run]})

    assert accounting_check.status == "fail"


def test_validate_markov_alignment_cli(tmp_path: Path) -> None:
    diagnostics_root = _write_diagnostics_root(tmp_path / "diagnostics")
    ladder_json = _write_json(tmp_path / "ladder.json", _full_doc_ladder_payload())
    full_tree_root = tmp_path / "full_tree"
    _write_json(full_tree_root / "summary.json", _full_tree_ipw_grid_payload())
    script = Path("/home/mlinegar/ThinkingTrees/scripts/validate_markov_alignment.py")
    out_json = tmp_path / "audit.json"
    out_md = tmp_path / "audit.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--diagnostics-root",
            str(diagnostics_root),
            "--full-tree-ipw-root",
            str(full_tree_root),
            "--ladder-json",
            str(ladder_json),
            "--output-json",
            str(out_json),
            "--output-markdown",
            str(out_md),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["summary"]["n_fail"] == 0
    assert "surface_coverage" in payload
    assert out_md.exists()
