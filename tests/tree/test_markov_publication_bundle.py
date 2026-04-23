from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.run_markov_publication_bundle import (
    _build_tradeoff_plan_for_bundle,
    _preflight_tradeoff_kwargs,
    _publication_experiment_spec,
    _bundle_markdown,
    _parse_args,
    _strip_detach_args,
    _validate_phase_dependencies,
    build_publication_run_plan,
    estimate_publication_runtime,
)


def _args(**overrides: object) -> argparse.Namespace:
    payload = {
        "phases": "tradeoff,capacity,parity,bundle",
        "with_preflight": True,
        "preflight_only": False,
        "tradeoff_preset": "standard",
        "tradeoff_tree_exact_eval_max_docs": 0,
        "tradeoff_prepared_data_root": None,
        "tradeoff_prepared_data_allow_create": True,
        "capacity_widths": "64,128,256",
        "capacity_modes": "2,4,8",
        "capacity_layers": "2,4,6",
        "capacity_screen_seeds": "0,1,2",
        "capacity_locked_seeds": "0,1,2,3,4",
        "capacity_top_k": 3,
        "parity_seeds": "0,1,2,3,4",
        "parity_scale_train_doc_counts": "1024,2048,3072,4096,5120,8192,10240",
        "parity_upper_bound_aux_fractions": "0.25,1.0",
        "parity_run_aux_upper_bound": True,
        "parity_backfill_on_success": True,
        "render_full_doc_parity_pdf": True,
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_estimate_publication_runtime_has_expected_sections() -> None:
    estimate = estimate_publication_runtime(_args(), mig_count=16)
    assert estimate["mig_count"] == 16
    assert estimate["with_preflight"] is True
    assert "preflight" in estimate["breakdown"]
    assert "tradeoff" in estimate["breakdown"]
    assert "capacity" in estimate["breakdown"]
    assert "parity" in estimate["breakdown"]
    assert "render_bundle" in estimate["breakdown"]
    assert estimate["total"]["eta_high_min"] > estimate["total"]["eta_low_min"] > 0.0
    assert estimate["breakdown"]["capacity"]["jobs"] == 96.0
    assert estimate["breakdown"]["parity"]["jobs"] == 230.0


def test_bundle_markdown_lists_key_artifacts_and_steps() -> None:
    markdown = _bundle_markdown(
        {
            "generated_at": "2026-03-24T00:00:00+00:00",
            "output_root": "/tmp/markov_publication_bundle",
            "eta_estimate": {"total": {"eta_low_min": 60.0, "eta_high_min": 90.0}},
            "reference_contract": {
                "identifiable_zero_reference_kind": "full_doc_fno_upper_bound",
                "full_doc_fno_families": ["official_fno", "official_fno_sumlen"],
                "full_doc_fno_training_backend": "shared_flat_trainer",
                "note": "Current publication path only.",
            },
            "artifacts": {
                "tradeoff_report_pdf": "/tmp/tradeoff.pdf",
                "learnability_report_pdf": "/tmp/learnability.pdf",
                "full_doc_fno_upper_bound_summary_json": "/tmp/fno_upper.json",
                "oracle_budget_frontier_summary_json": "/tmp/oracle_budget.json",
                "efficiency_suite_summary_json": "/tmp/efficiency_suite.json",
                "supervision_recovery_summary_json": "/tmp/supervision_recovery.json",
                "fair_parity_run_summary_json": "/tmp/parity.json",
            },
            "steps": [
                {
                    "name": "tradeoff",
                    "status": "completed",
                    "wall_clock_s": 120.0,
                    "log_path": "/tmp/tradeoff.log",
                },
                {
                    "name": "capacity",
                    "status": "reused",
                    "wall_clock_s": 0.0,
                    "log_path": "/tmp/capacity.log",
                },
            ],
        }
    )
    assert "Markov Publication Bundle" in markdown
    assert "Preflight enabled" in markdown
    assert "Canonical identifiable-zero reference" in markdown
    assert "Full-doc FNO training backend" in markdown
    assert "tradeoff_report_pdf" in markdown
    assert "full_doc_fno_upper_bound_summary_json" in markdown
    assert "oracle_budget_frontier_summary_json" in markdown
    assert "efficiency_suite_summary_json" in markdown
    assert "supervision_recovery_summary_json" in markdown
    assert "`tradeoff`: `completed`" in markdown
    assert "`capacity`: `reused`" in markdown


def test_validate_phase_dependencies_rejects_archived_report_phases() -> None:
    args = _args(phases="tree_fno_pdf")
    with pytest.raises(ValueError, match="archived publication phases"):
        _validate_phase_dependencies(args, {"tree_fno_pdf"})


def test_strip_detach_args_removes_detach_only_flags() -> None:
    argv = [
        "--detach",
        "--detach-name",
        "bundle_job",
        "--detach-job-root=/tmp/launcher",
        "--detach-description",
        "detached run",
        "--output-root",
        "outputs/run",
        "--no-with-preflight",
    ]
    assert _strip_detach_args(argv) == [
        "--output-root",
        "outputs/run",
        "--no-with-preflight",
    ]


def test_publication_bundle_selection_config_flattens_sections(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "bundle_selection.toml"
    config_path.write_text(
        "\n".join(
            [
                "[publication_bundle]",
                'phases = ["tradeoff", "parity", "bundle"]',
                "",
                "[publication_bundle.tradeoff]",
                'preset = "smoke"',
                'phases = ["law_packages", "report"]',
                "train_docs = 4096",
                "tree_exact_eval_max_docs = 64",
                'prepared_data_root = "/tmp/prepared_data"',
                "prepared_data_allow_create = false",
                "",
                "[publication_bundle.tradeoff.tree_reference]",
                'mode = "capacity_locked"',
                'capacity_root = "/tmp/capacity_root"',
                "",
                "[publication_bundle.parity]",
                'tree_families = ["tree_neural"]',
                'fno_families = ["official_fno"]',
                "seeds = [0, 1]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_markov_publication_bundle.py",
            "--selection-config",
            str(config_path),
        ],
    )
    args = _parse_args()
    assert args.phases == "tradeoff parity bundle"
    assert args.tradeoff_preset == "smoke"
    assert args.tradeoff_phases == "law_packages report"
    assert args.tradeoff_train_docs == 4096
    assert args.tradeoff_tree_exact_eval_max_docs == 64
    assert Path(args.tradeoff_prepared_data_root) == Path("/tmp/prepared_data")
    assert args.tradeoff_prepared_data_allow_create is False
    assert args.tradeoff_tree_reference_mode == "capacity_locked"
    assert Path(args.tradeoff_tree_reference_capacity_root) == Path("/tmp/capacity_root")
    assert args.parity_tree_families == "tree_neural"
    assert args.parity_fno_families == "official_fno"
    assert args.parity_seeds == "0 1"


def test_publication_run_plan_includes_nested_tradeoff_and_parity_selection(tmp_path: Path) -> None:
    args = _args(
        output_root=tmp_path / "bundle",
        selection_config=None,
        estimate_only=False,
        detach=False,
        detach_name="bundle",
        detach_job_root=None,
        detach_description="bundle",
        reuse_existing=True,
        with_preflight=True,
        preflight_only=False,
        python_bin="python3",
        migs="MIG-a MIG-b",
        tradeoff_root=None,
        tradeoff_preset="smoke",
        tradeoff_device_mode="cuda",
        tradeoff_phases="law_packages report",
        tradeoff_train_docs=4096,
        tradeoff_tree_exact_eval_max_docs=64,
        tradeoff_prepared_data_root=tmp_path / "prepared_data",
        tradeoff_prepared_data_allow_create=False,
        tradeoff_tree_reference_mode="preset",
        tradeoff_tree_reference_capacity_root=None,
        tradeoff_tree_reference_preset="common_factorized_sketch_v1",
        capacity_root=None,
        capacity_benchmark="recoverable_v4",
        capacity_train_doc_count=10240,
        capacity_screen_seeds="0",
        capacity_locked_seeds="0",
        capacity_top_k=1,
        capacity_widths="64",
        capacity_modes="2",
        capacity_layers="2",
        parity_root=None,
        parity_benchmark="recoverable_v4",
        parity_gate_train_doc_count=10240,
        parity_scale_train_doc_counts="1024 2048",
        parity_seeds="0 1",
        parity_tree_families="tree_neural",
        parity_fno_families="official_fno",
        parity_backfill_on_success=True,
        parity_run_aux_upper_bound=False,
        parity_upper_bound_aux_fractions="1.0",
        bundle_root=None,
        render_full_doc_parity_pdf=True,
        write_selection_template=None,
        write_run_plan=None,
        plan_only=False,
    )
    plan = build_publication_run_plan(args, mig_uuids=["MIG-a", "MIG-b"], output_root=Path(args.output_root))
    assert plan["resolved_selection"]["tradeoff"]["train_docs"] == 4096
    assert plan["resolved_selection"]["tradeoff"]["tree_exact_eval_max_docs"] == 64
    assert plan["resolved_selection"]["tradeoff"]["prepared_data_root"] == str(
        tmp_path / "prepared_data"
    )
    assert plan["resolved_selection"]["tradeoff"]["prepared_data_allow_create"] is False
    assert plan["resolved_selection"]["tradeoff"]["tree_reference"]["mode"] == "preset"
    assert plan["resolved_selection"]["tradeoff"]["tree_reference"]["preset"] == "common_factorized_sketch_v1"
    assert plan["resolved_selection"]["parity"]["tree_families"] == ["tree_neural"]
    assert plan["resolved_selection"]["parity"]["fno_families"] == ["official_fno"]
    assert plan["tradeoff_run_plan"]["phase_task_counts"]["law_packages"]["worker_tasks"] >= 1
    assert "oracle_budget_frontier" not in plan["tradeoff_run_plan"]["phase_task_counts"]
    tradeoff_step = next(step for step in plan["step_commands"] if step["name"] == "tradeoff")
    assert "--tree-reference-mode" in tradeoff_step["command"]
    assert "--tree-reference-preset" in tradeoff_step["command"]
    assert "--tree-exact-eval-max-docs" in tradeoff_step["command"]
    assert "--prepared-data-root" in tradeoff_step["command"]
    assert "--no-prepared-data-allow-create" in tradeoff_step["command"]
    assert any(step["name"] == "parity" for step in plan["step_commands"])


def test_publication_experiment_spec_uses_canonical_surface(tmp_path: Path) -> None:
    args = _args(
        output_root=tmp_path / "bundle",
        selection_config=None,
        estimate_only=False,
        detach=False,
        detach_name="bundle",
        detach_job_root=None,
        detach_description="bundle",
        reuse_existing=True,
        with_preflight=True,
        preflight_only=False,
        python_bin="python3",
        migs="MIG-a MIG-b",
        phases="tradeoff bundle",
    )
    spec = _publication_experiment_spec(
        args=args,
        output_root=Path(args.output_root),
        manifest={"tradeoff": {"supervision_recovery_structural_cell": "r12_p079"}},
    )
    assert spec.adapter_id == "markov_tree"
    assert spec.title == "markov_publication_bundle"
    assert "bundle" in {phase.phase_id for phase in spec.phases}


def test_preflight_tradeoff_supervision_recovery_is_minimal_probe(tmp_path: Path) -> None:
    args = _args(
        output_root=tmp_path / "bundle",
        selection_config=None,
        tradeoff_preset="standard",
        tradeoff_phases="supervision_recovery report",
        tradeoff_device_mode="cuda",
        tradeoff_train_docs=10240,
        tradeoff_runtime_data_mode="resident",
        tradeoff_runtime_bucket_mode="leaf_count_auto_queue",
        tradeoff_tree_reference_mode="preset",
        tradeoff_tree_reference_preset="common_factorized_sketch_v1",
        tradeoff_supervision_recovery_tree_family="tree_neural",
        tradeoff_supervision_recovery_structural_cell="r12_p079",
    )
    kwargs = _preflight_tradeoff_kwargs(args, phases=str(args.tradeoff_phases))
    assert kwargs["preset"] == "standard"
    assert kwargs["selection_config"] is None
    assert kwargs["supervision_recovery_train_docs"] == [1024]
    assert kwargs["supervision_recovery_seeds"] == [0]
    assert kwargs["supervision_recovery_packages"] == ["full100"]

    plan = _build_tradeoff_plan_for_bundle(
        args=args,
        output_root=tmp_path / "preflight_tradeoff",
        mig_uuids=["MIG-a", "MIG-b"],
        tree_reference_mode="preset",
        tree_reference_capacity_root=None,
        tree_reference_preset="common_factorized_sketch_v1",
        runtime_data_mode="resident",
        runtime_bucket_mode="leaf_count_auto_queue",
        supervision_recovery_tree_family=str(kwargs["supervision_recovery_tree_family"]),
        supervision_recovery_structural_cell=str(kwargs["supervision_recovery_structural_cell"]),
        supervision_recovery_train_docs=kwargs["supervision_recovery_train_docs"],
        supervision_recovery_seeds=kwargs["supervision_recovery_seeds"],
        supervision_recovery_packages=kwargs["supervision_recovery_packages"],
    )
    assert plan["phase_task_counts"]["supervision_recovery"]["worker_tasks"] == 4
    assert plan["resolved_selection"]["supervision_recovery_train_docs"] == [1024]
    assert plan["resolved_selection"]["supervision_recovery_seeds"] == [0]
    assert plan["resolved_selection"]["supervision_recovery_packages"] == ["full100"]
    assert plan["resolved_selection"]["runtime"]["bucket_mode"] == "leaf_count_auto_queue"


def test_checked_in_publication_config_builds_plan() -> None:
    config_path = Path("config/markov/publication_bundle.standard.toml")
    assert config_path.exists()
    plan = build_publication_run_plan(
        _parse_args(["--config", str(config_path), "--plan-only"]),
        mig_uuids=["MIG-a", "MIG-b"],
        output_root=Path("/tmp/markov_publication_bundle"),
    )
    assert plan["resolved_selection"]["tradeoff"]["train_docs"] == 10240
    assert plan["resolved_selection"]["tradeoff"]["supervision_recovery_train_docs"] == [1024, 4096, 10240]
    assert plan["resolved_selection"]["parity"]["tree_families"] == [
        "tree_neural_c2",
        "tree_neural_c2c3",
        "tree_neural",
    ]
    assert plan["resolved_selection"]["tradeoff"]["tree_reference"]["mode"] == "preset"
    assert plan["resolved_selection"]["tradeoff"]["tree_reference"]["preset"] == "common_factorized_sketch_v1"
    assert plan["resolved_selection"]["capacity"]["runtime"]["data_mode"] == "resident"
    assert plan["resolved_selection"]["capacity"]["runtime"]["capacity_workers_per_mig"] == 2
    assert plan["resolved_selection"]["parity"]["runtime"]["allow_multi_worker_screen"] is False
    assert set(plan["tradeoff_run_plan"]["phase_task_counts"]) == {"supervision_recovery", "report"}


def test_checked_in_iteration_and_publication_bundle_configs_build_plans() -> None:
    iteration = Path("config/markov/publication_bundle.iteration.toml")
    publication = Path("config/markov/publication_bundle.publication.toml")
    no10240 = Path("config/markov/publication_bundle.no10240.toml")
    assert iteration.exists()
    assert publication.exists()
    assert no10240.exists()

    iteration_plan = build_publication_run_plan(
        _parse_args(["--config", str(iteration), "--plan-only"]),
        mig_uuids=["MIG-a", "MIG-b"],
        output_root=Path("/tmp/markov_publication_bundle_iteration"),
    )
    publication_plan = build_publication_run_plan(
        _parse_args(["--config", str(publication), "--plan-only"]),
        mig_uuids=["MIG-a", "MIG-b"],
        output_root=Path("/tmp/markov_publication_bundle_publication"),
    )
    no10240_plan = build_publication_run_plan(
        _parse_args(["--config", str(no10240), "--plan-only"]),
        mig_uuids=["MIG-a", "MIG-b"],
        output_root=Path("/tmp/markov_publication_bundle_no10240"),
    )

    assert iteration_plan["resolved_selection"]["tradeoff"]["train_docs"] == 4096
    assert publication_plan["resolved_selection"]["tradeoff"]["train_docs"] == 10240
    assert no10240_plan["resolved_selection"]["tradeoff"]["train_docs"] == 4096
    assert iteration_plan["resolved_selection"]["tradeoff"]["supervision_recovery_train_docs"] == [1024, 2048, 4096]
    assert publication_plan["resolved_selection"]["tradeoff"]["supervision_recovery_train_docs"] == [1024, 4096, 10240]
    assert no10240_plan["resolved_selection"]["tradeoff"]["supervision_recovery_train_docs"] == [1024, 2048, 4096]
    assert iteration_plan["resolved_selection"]["tradeoff"]["tree_reference"]["mode"] == "preset"
    assert publication_plan["resolved_selection"]["tradeoff"]["tree_reference"]["mode"] == "preset"
    assert no10240_plan["resolved_selection"]["tradeoff"]["tree_reference"]["mode"] == "preset"
    assert iteration_plan["resolved_selection"]["parity"]["backfill_enabled"] is False
    assert publication_plan["resolved_selection"]["parity"]["backfill_enabled"] is True
    assert no10240_plan["resolved_selection"]["with_preflight"] is False
    assert "capacity" not in no10240_plan["resolved_selection"]["phases"]
    assert set(iteration_plan["tradeoff_run_plan"]["phase_task_counts"]) == {"supervision_recovery", "report"}
    assert set(publication_plan["tradeoff_run_plan"]["phase_task_counts"]) == {"supervision_recovery", "report"}
    assert set(no10240_plan["tradeoff_run_plan"]["phase_task_counts"]) == {"supervision_recovery", "report"}


def test_checked_in_v3_publication_bundle_config_builds_plan() -> None:
    config_path = Path("config/markov/publication_bundle.v3.toml")
    assert config_path.exists()

    plan = build_publication_run_plan(
        _parse_args(["--config", str(config_path), "--plan-only"]),
        mig_uuids=["MIG-a", "MIG-b"],
        output_root=Path("/tmp/markov_publication_bundle_v3"),
    )

    tradeoff_selection = plan["tradeoff_run_plan"]["resolved_selection"]

    assert plan["resolved_selection"]["tradeoff"]["preset"] == "v3"
    assert tradeoff_selection["supervision_recovery_recoverable_benchmark"] == "recoverable_v5_t128"
    assert tradeoff_selection["supervision_recovery_structural_grid"] == "structural_core_v2_t128"
    assert tradeoff_selection["supervision_recovery_leaf_token_ladder"] == [32, 16, 8]
    assert tradeoff_selection["supervision_recovery_packages"] == [
        "full100",
        "r100_superset_local_eq_10p0",
        "r100_superset_local_eq_15p0",
        "r100_superset_local_eq_20p0",
    ]
    assert plan["resolved_selection"]["parity"]["benchmark"] == "recoverable_v5_t128"
    assert plan["resolved_selection"]["parity"]["tree_families"] == ["tree_neural"]
    assert set(plan["tradeoff_run_plan"]["phase_task_counts"]) == {"supervision_recovery", "report"}


def test_checked_in_long_v4_supervision_bundle_configs_build_plans() -> None:
    long_v4 = Path("config/markov/publication_bundle.long_v4.toml")
    long_v4_incremental = Path("config/markov/publication_bundle.long_v4_incremental.toml")
    supervision_followup = Path("config/markov/publication_bundle.long_v4_supervision_followup.toml")
    assert long_v4.exists()
    assert long_v4_incremental.exists()
    assert supervision_followup.exists()

    long_plan = build_publication_run_plan(
        _parse_args(["--config", str(long_v4), "--plan-only"]),
        mig_uuids=["MIG-a", "MIG-b"],
        output_root=Path("/tmp/markov_publication_bundle_long_v4"),
    )
    incremental_plan = build_publication_run_plan(
        _parse_args(["--config", str(long_v4_incremental), "--plan-only"]),
        mig_uuids=["MIG-a", "MIG-b"],
        output_root=Path("/tmp/markov_publication_bundle_long_v4_incremental"),
    )
    followup_plan = build_publication_run_plan(
        _parse_args(["--config", str(supervision_followup), "--plan-only"]),
        mig_uuids=["MIG-a", "MIG-b"],
        output_root=Path("/tmp/markov_publication_bundle_long_v4_supervision_followup"),
    )

    assert "supervision_recovery" in long_plan["tradeoff_run_plan"]["phase_task_counts"]
    assert "supervision_recovery" in incremental_plan["tradeoff_run_plan"]["phase_task_counts"]
    assert "supervision_recovery" in followup_plan["tradeoff_run_plan"]["phase_task_counts"]
    assert "parity" not in followup_plan["resolved_selection"]["phases"]
