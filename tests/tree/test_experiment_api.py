from __future__ import annotations

from pathlib import Path

from src.ctreepo.sim.manifest import RunSpec as LegacyRunSpec
from src.experiments.adapters import CTreePOSimAdapter, RuntimeEvalAdapter, TreePOTrainingAdapter
from src.experiments.contracts import (
    ControlRef,
    ExperimentSpec,
    ResultRow,
    SupervisionRef,
    benchmark_ref_from_parts,
    default_phase_specs,
    method_ref_from_parts,
)
from src.experiments.control_plane import write_experiment_manifest
from src.experiments.legacy import ctreepo_runs_to_experiment, runtime_run_spec_to_experiment
from src.experiments.markov_full_doc import (
    control_ref_from_markov_full_doc_contract,
    method_ref_from_markov_full_doc_run,
)
from src.experiments.normalization import (
    control_ref_from_ctreepo_local_law_config,
    control_ref_from_treepo_local_law_config,
    supervision_ref_from_markov_config,
    supervision_ref_from_treepo_supervision_spec,
)
from src.experiments.reporting import (
    build_canonical_report_views,
    derive_comparison_view,
    load_canonical_result_rows,
)
from src.runtime.contracts import RunPhaseSpec, RunSpec


def test_experiment_spec_round_trip() -> None:
    spec = ExperimentSpec.create(
        adapter_id="runtime_eval",
        output_root="/tmp/exp",
        title="runtime_eval",
        benchmark_refs=(
            benchmark_ref_from_parts(
                family="runtime_benchmark",
                scope="ruler_synthetic",
                name="ruler_synthetic",
            ),
        ),
        method_refs=(
            method_ref_from_parts(
                family="runtime_eval",
                variant="runtime_full",
                adapter="runtime_eval",
                supervision=SupervisionRef(
                    topology_scope="tree",
                    unit_selector="leaves",
                    supervision_kind="scalar",
                    label_source="requests",
                    labeler_kind="oracle_score",
                    coverage_label="tree_local_law_labels",
                ),
                control_ref=ControlRef(
                    control_family="tree_local_law",
                    law_ids=("L1", "L2"),
                    applies_to="tree_nodes",
                    enabled=True,
                    source_kind="verifier",
                    sample_budget=16,
                ),
            ),
        ),
        phases=default_phase_specs(("init", "run", "aggregate")),
        launch_command=("python3", "scripts/run_runtime_eval.py", "init"),
        resume_command=("python3", "scripts/run_runtime_eval.py", "run"),
    )
    restored = ExperimentSpec.from_dict(spec.to_dict())
    assert restored.experiment_id == spec.experiment_id
    assert restored.adapter_id == "runtime_eval"
    assert restored.phases[0].phase_id == "init"
    assert restored.method_refs[0].supervision is not None
    assert restored.method_refs[0].control_ref is not None
    assert restored.method_refs[0].control_ref.law_ids == ("L1", "L2")


def test_runtime_legacy_runspec_imports_into_canonical_experiment() -> None:
    runtime_spec = RunSpec(
        run_id="runtime_001",
        created_utc="2026-04-01T00:00:00+00:00",
        output_dir="outputs/runtime",
        benchmark={"name": "ruler_synthetic", "family": "runtime_benchmark"},
        model={"model": "demo-model", "engine": "vllm"},
        runtime_defaults={},
        phases=[
            RunPhaseSpec(
                phase_id="P0",
                tasks=["vt"],
                lengths=[1024],
                seeds=[0, 1],
                num_samples=4,
                split="validation",
                modes=["runtime_full"],
            )
        ],
    )
    experiment = runtime_run_spec_to_experiment(
        runtime_spec,
        launch_command=("python3", "scripts/run_runtime_eval.py", "init"),
    )
    assert experiment.adapter_id == "runtime_eval"
    assert experiment.output_root.endswith("outputs/runtime/runtime_001")
    assert len(experiment.tasks) == 2
    assert experiment.tasks[0].phase_id == "P0"


def test_ctreepo_legacy_runspec_manifest_imports_into_canonical_experiment(tmp_path: Path) -> None:
    runs = [
        LegacyRunSpec.create(
            family="markov_suite",
            config={"variant": "smoke"},
            outputs={"json_summary": str(tmp_path / "summary.json")},
            command="python3 scripts/run_markov_suite.py --smoke",
        )
    ]
    experiment = ctreepo_runs_to_experiment(
        runs,
        output_root=str(tmp_path / "suite"),
        title="suite_manifest",
    )
    assert experiment.adapter_id == "ctreepo_sim"
    assert len(experiment.tasks) == 1
    assert experiment.tasks[0].task_kind == "legacy_runspec_command"


def test_runtime_eval_adapter_builds_canonical_spec(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                "benchmark:",
                "  name: ruler_synthetic",
                "  family: runtime_benchmark",
                "model:",
                "  model: demo-model",
                "  engine: vllm",
                "runtime_defaults: {}",
                "phases:",
                "  - phase_id: P0",
                "    tasks: [vt]",
                "    lengths: [1024]",
                "    seeds: [0]",
                "    num_samples: 2",
                "    split: validation",
                "    modes: [runtime_full]",
            ]
        ),
        encoding="utf-8",
    )
    adapter = RuntimeEvalAdapter()
    spec = adapter.build_experiment_spec(
        [
            "python3",
            "scripts/run_runtime_eval.py",
            "init",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "outputs"),
        ],
        cwd=Path.cwd(),
    )
    assert spec.adapter_id == "runtime_eval"
    assert spec.report_profiles == ("runtime_eval_summary",)


def test_ctreepo_adapter_reads_manifest_flag(tmp_path: Path) -> None:
    manifest_path = tmp_path / "runspec_manifest.jsonl"
    manifest_path.write_text(
        '{"id":"abc","family":"toy_suite","config":{"variant":"smoke"},"outputs":{"json_summary":"'
        + str(tmp_path / "summary.json")
        + '"},"command":"python3 toy.py","requires":[],"resources":{}}\n',
        encoding="utf-8",
    )
    adapter = CTreePOSimAdapter()
    spec = adapter.build_experiment_spec(
        ["python3", "scripts/suite.py", "--manifest", str(manifest_path)],
        cwd=Path.cwd(),
    )
    assert spec.adapter_id == "ctreepo_sim"
    assert spec.tasks[0].phase_id == "toy_suite"


def test_topology_aware_normalization_builders() -> None:
    markov = supervision_ref_from_markov_config(
        {
            "leaf_supervision_kind": "count_only",
            "leaf_label_rate": 0.5,
            "internal_supervision_kind": "count_only",
            "internal_label_rate": 0.5,
        },
        package_name="full10_leaf_count50_internal_count50",
    )
    assert markov is not None
    assert markov.root_rate == 0.1
    assert markov.coverage_label == "full10_leaf_count50_internal_count50"

    treepo_sup = supervision_ref_from_treepo_supervision_spec(
        {
            "mode": "label_now",
            "unit_selector": "internal",
            "supervision_kind": "comparative",
            "labeler_kind": "oracle_score",
            "doc_sample_probability": 0.25,
            "unit_sampling_probability": 0.5,
            "sampling_strategy": "level_weighted",
            "max_units": 8,
        }
    )
    assert treepo_sup is not None
    assert treepo_sup.unit_selector == "internal"
    assert treepo_sup.sampling_strategy == "level_weighted"

    treepo_control = control_ref_from_treepo_local_law_config(
        {
            "enable_l1": True,
            "enable_l2": False,
            "enable_l3": True,
            "sample_budget": 12,
            "sampling_probability": 0.2,
            "sampling_strategy": "random",
            "discrepancy_threshold": 0.1,
        }
    )
    assert treepo_control is not None
    assert treepo_control.law_ids == ("L1", "L3")

    ctreepo_control = control_ref_from_ctreepo_local_law_config(
        {
            "leaf_audit_weight": 0.4,
            "merge_audit_weight": 0.8,
            "violation_threshold": 7.5,
            "label_source_kind": "oracle_callback",
        }
    )
    assert ctreepo_control is not None
    assert ctreepo_control.law_ids == ("L1", "L2")
    assert ctreepo_control.source_kind == "oracle_callback"


def test_markov_full_doc_method_ref_surfaces_tree_law_contract() -> None:
    method_ref = method_ref_from_markov_full_doc_run(
        family="tree_neural",
        variant="superset_plus10",
        adapter="markov_tree",
        config_like={
            "root_label_rate": 1.0,
            "leaf_supervision_kind": "count_only",
            "leaf_label_rate": 0.1,
            "internal_supervision_kind": "count_only",
            "internal_label_rate": 0.1,
            "objective_weights_active": True,
            "local_law_c1_weight": 0.1,
            "local_law_c2_weight": 0.1,
            "local_law_c3_weight": 0.1,
            "tree_c2_mode": "reconstruction",
            "summary_spec_name": "markov_count_sketch",
            "fixed_leaf_tokens": 16,
            "run_intent_hash": "intent_demo",
        },
        package_name="full100_superset10",
        mean_leaves_per_doc=8.0,
    )
    assert method_ref.supervision is not None
    assert method_ref.supervision.root_rate == 1.0
    assert method_ref.supervision.leaf_rate == 0.1
    assert method_ref.supervision.internal_rate == 0.1
    assert method_ref.control_ref is not None
    assert method_ref.control_ref.law_ids == ("L1", "L3", "L2")
    assert method_ref.metadata["family_api_group"] == "markov_full_doc_neuraloperator"
    assert method_ref.metadata["law_alignment_status"] == "approximate_audited"
    assert "c2_replay_proxy_not_exact_paper_idempotence" in method_ref.metadata["law_contract_limitations"]
    assert method_ref.metadata["run_intent_hash"] == "intent_demo"


def test_markov_full_doc_method_ref_keeps_fno_local_budget_root_only() -> None:
    method_ref = method_ref_from_markov_full_doc_run(
        family="official_fno",
        variant="matched_reference",
        adapter="markov_tree",
        config_like={
            "root_label_rate": 1.0,
            "leaf_supervision_kind": "count_only",
            "leaf_label_rate": 0.2,
            "internal_supervision_kind": "count_only",
            "internal_label_rate": 0.2,
        },
        package_name="full100_superset20",
    )
    assert method_ref.supervision is not None
    assert method_ref.supervision.root_rate == 1.0
    assert method_ref.supervision.leaf_rate == 0.0
    assert method_ref.supervision.internal_rate == 0.0
    assert method_ref.control_ref is None
    assert method_ref.metadata["law_alignment_status"] == "proxy_only_reference"
    assert "root_only_reference_no_tree_local_law_channel" in method_ref.metadata["law_contract_limitations"]


def test_markov_full_doc_control_ref_maps_paper_c2_to_lean_l3() -> None:
    control = control_ref_from_markov_full_doc_contract(
        "tree_neural_c2",
        config_like={
            "objective_weights_active": True,
            "local_law_c2_weight": 0.2,
            "tree_c2_mode": "reconstruction",
            "summary_spec_name": "markov_count_sketch",
        },
        mean_leaves_per_doc=8.0,
    )
    assert control is not None
    assert control.control_family == "markov_full_doc_local_law"
    assert control.enabled is True
    assert control.law_ids == ("L3",)
    assert control.metadata["c2_nontriviality_status"] == "decoded_summary_replay"


def test_treepo_training_adapter_builds_canonical_spec(tmp_path: Path) -> None:
    adapter = TreePOTrainingAdapter()
    spec = adapter.build_experiment_spec(
        [
            "python3",
            "scripts/train_neural_operators.py",
            "--output-dir",
            str(tmp_path / "operators"),
            "--task",
            "manifesto_rile",
            "--which",
            "both",
        ],
        cwd=Path.cwd(),
    )
    assert spec.adapter_id == "treepo_training"
    assert spec.benchmark_refs[0].scope == "manifesto_rile"
    assert {item.family for item in spec.method_refs} == {"ctreepo", "mergeable_sketch"}


def test_ctreepo_legacy_summary_backfills_canonical_result_rows(tmp_path: Path) -> None:
    summary_path = tmp_path / "suite_summary.json"
    summary_path.write_text(
        '{"oracle_gap":0.12,"audit":{"violation_rate":0.03,"n_nodes":128}}',
        encoding="utf-8",
    )
    runs = [
        LegacyRunSpec.create(
            family="segmented_lda_ctreepo",
            config={"variant": "smoke"},
            outputs={"json_summary": str(summary_path)},
            command="python3 scripts/run_segmented_lda_ctreepo.py --smoke",
        )
    ]
    experiment = ctreepo_runs_to_experiment(
        runs,
        output_root=str(tmp_path),
        title="suite_manifest",
    )
    write_experiment_manifest(tmp_path, experiment)
    rows = load_canonical_result_rows(tmp_path)
    metric_names = {row.metric_name for row in rows}
    assert rows
    assert "oracle_gap" in metric_names
    assert "audit.violation_rate" in metric_names
    assert "audit.n_nodes" in metric_names


def test_build_canonical_report_views_separates_supervision_and_control() -> None:
    treepo_benchmark = benchmark_ref_from_parts(
        family="treepo_task",
        scope="manifesto_rile",
        name="manifesto_rile",
    )
    markov_benchmark = benchmark_ref_from_parts(
        family="markov_full_doc",
        scope="recoverable_v4",
        name="recoverable_v4",
    )
    markov_row = ResultRow(
        experiment_id="exp_markov",
        phase="eval",
        benchmark_ref=markov_benchmark,
        method_ref=method_ref_from_parts(
            family="tree_neural",
            variant="markov",
            adapter="markov_tree",
        ),
        split="test",
        seed=0,
        train_docs=10240,
        supervision_ref=SupervisionRef(
            root_rate=0.1,
            leaf_rate=0.5,
            internal_rate=0.5,
            topology_scope="tree",
            unit_selector="root+leaf+internal",
            supervision_kind="scalar",
            label_source="dataset_labels",
            labeler_kind="precomputed",
            coverage_label="R10+LcIa50",
        ),
        control_ref=ControlRef(
            control_family="tree_local_law",
            law_ids=("L1", "L2"),
            applies_to="tree_nodes",
            enabled=True,
            source_kind="verifier",
        ),
        metric_name="root_mae",
        metric_value=0.12,
        artifact_refs=(),
    )
    treepo_row = ResultRow(
        experiment_id="exp_treepo",
        phase="eval",
        benchmark_ref=treepo_benchmark,
        method_ref=method_ref_from_parts(
            family="ctreepo",
            variant="local_law_training",
            adapter="treepo_training",
        ),
        split="test",
        seed=0,
        train_docs=10240,
        supervision_ref=SupervisionRef(
            topology_scope="tree",
            unit_selector="internal",
            supervision_kind="comparative",
            label_source="label_now",
            labeler_kind="oracle_score",
            coverage_label="internal_only",
        ),
        control_ref=ControlRef(
            control_family="ctreepo_local_law",
            law_ids=("L1", "L2"),
            applies_to="leaf+internal",
            enabled=True,
            source_kind="oracle_callback",
        ),
        metric_name="root_mae",
        metric_value=0.09,
        artifact_refs=(),
    )
    runtime_row = ResultRow(
        experiment_id="exp_runtime",
        phase="eval",
        benchmark_ref=benchmark_ref_from_parts(
            family="runtime_benchmark",
            scope="ruler_synthetic",
            name="ruler_synthetic",
        ),
        method_ref=method_ref_from_parts(
            family="runtime_eval",
            variant="runtime_full",
            adapter="runtime_eval",
        ),
        split="validation",
        seed=0,
        train_docs=None,
        metric_name="score",
        metric_value=1.0,
        artifact_refs=(),
    )
    views = build_canonical_report_views([markov_row, treepo_row, runtime_row])
    assert views["method_families"] == ["ctreepo", "runtime_eval", "tree_neural"]
    assert views["supervision_labels"] == ["R10+LcIa50", "internal_only"]
    assert "ctreepo_local_law:L1+L2" in views["control_labels"]
    assert "tree_local_law:L1+L2" in views["control_labels"]
    assert views["comparable_metrics"]["root_mae"]["method_families"] == ["ctreepo", "tree_neural"]
    markov_view = derive_comparison_view(markov_row)
    treepo_view = derive_comparison_view(treepo_row)
    runtime_view = derive_comparison_view(runtime_row)
    assert markov_view["comparison_domain"] == "supervised_root_regression"
    assert markov_view["direct_label_budget"]["label"] == "R10"
    assert treepo_view["local_supervision_budget"]["label"] == "internal_only"
    assert runtime_view["comparison_domain"] == "runtime_context_eval"
    assert any(
        spec["plot_kind"] == "runtime_context_scaling"
        for spec in views["appendix_plot_specs"]
    )


def test_build_canonical_report_views_selects_budget_matched_plot_specs() -> None:
    benchmark = benchmark_ref_from_parts(
        family="treepo_task",
        scope="manifesto_rile",
        name="manifesto_rile",
    )
    rows = [
        ResultRow(
            experiment_id="exp_low_docs",
            phase="eval",
            benchmark_ref=benchmark,
            method_ref=method_ref_from_parts(
                family="llm_prompt_optimization",
                variant="prompt_opt",
                adapter="treepo_training",
            ),
            split="test",
            train_docs=512,
            supervision_ref=SupervisionRef(
                topology_scope="document",
                unit_selector="document",
                supervision_kind="scalar",
                label_source="dataset_labels",
                labeler_kind="gold_score",
                doc_sample_probability=0.1,
                coverage_label="10% labeled docs",
            ),
            metric_name="mae",
            metric_value=0.24,
        ),
        ResultRow(
            experiment_id="exp_high_docs",
            phase="eval",
            benchmark_ref=benchmark,
            method_ref=method_ref_from_parts(
                family="llm_prompt_optimization",
                variant="prompt_opt",
                adapter="treepo_training",
            ),
            split="test",
            train_docs=1024,
            supervision_ref=SupervisionRef(
                topology_scope="document",
                unit_selector="document",
                supervision_kind="scalar",
                label_source="dataset_labels",
                labeler_kind="gold_score",
                doc_sample_probability=0.1,
                coverage_label="10% labeled docs",
            ),
            metric_name="mae",
            metric_value=0.19,
        ),
        ResultRow(
            experiment_id="exp_full",
            phase="eval",
            benchmark_ref=benchmark,
            method_ref=method_ref_from_parts(
                family="llm_prompt_optimization",
                variant="prompt_opt",
                adapter="treepo_training",
            ),
            split="test",
            train_docs=1024,
            supervision_ref=SupervisionRef(
                topology_scope="document",
                unit_selector="document",
                supervision_kind="scalar",
                label_source="dataset_labels",
                labeler_kind="gold_score",
                doc_sample_probability=1.0,
                coverage_label="100% labeled docs",
            ),
            metric_name="mae",
            metric_value=0.12,
        ),
    ]
    views = build_canonical_report_views(rows)
    assert any(
        spec["plot_kind"] == "train_doc_scaling"
        for spec in views["main_body_plot_specs"]
    )
    assert any(
        spec["plot_kind"] == "direct_label_budget_ladder"
        for spec in views["main_body_plot_specs"]
    )
    ladder_title = next(
        spec["title"]
        for spec in views["main_body_plot_specs"]
        if spec["plot_kind"] == "direct_label_budget_ladder"
    )
    assert "Same benchmark, split, train-doc count, and direct document/root label budget." in views["caption_contracts"][ladder_title]["match_note"]
    assert "Family baseline means the maximal direct-label run" in views["caption_contracts"][ladder_title]["reference_note"]
