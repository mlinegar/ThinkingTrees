from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
import subprocess
import sys


def _load_module():
    root = Path(__file__).resolve().parents[2]
    mod_path = root / "scripts" / "run_tree_neural_full_doc_mig.py"
    spec = importlib.util.spec_from_file_location("run_tree_neural_full_doc_mig", str(mod_path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _capture_scheduler_jobs(monkeypatch, mod) -> list[dict[str, object]]:
    captured: list[dict[str, object]] = []
    real = mod._scheduler_item_for_job

    def _wrapped(
        *,
        phase,
        item_id,
        output_root,
        job,
        torch_threads,
        use_cuda,
        gpu_slots=1,
        allowed_devices=(),
    ):
        captured.append(
            {
                "phase": str(phase),
                "item_id": str(item_id),
                "output_root": Path(str(output_root)),
                "job": job,
                "gpu_slots": int(gpu_slots),
                "allowed_devices": tuple(str(token) for token in tuple(allowed_devices)),
            }
        )
        return real(
            phase=phase,
            item_id=item_id,
            output_root=output_root,
            job=job,
            torch_threads=torch_threads,
            use_cuda=use_cuda,
            gpu_slots=gpu_slots,
            allowed_devices=allowed_devices,
        )

    monkeypatch.setattr(mod, "_scheduler_item_for_job", _wrapped)
    return captured


def _find_scheduler_item(items, item_id: str):
    for item in items:
        if str(item.item_id) == str(item_id):
            return item
    raise AssertionError(f"missing scheduler item: {item_id}")


def _synthetic_mig_layout(
    *,
    gpu_count: int = 4,
    migs_per_gpu: int = 4,
) -> list[dict[str, object]]:
    layout: list[dict[str, object]] = []
    for gpu_index in range(int(gpu_count)):
        for mig_index in range(int(migs_per_gpu)):
            layout.append(
                {
                    "gpu_index": int(gpu_index),
                    "gpu_uuid": f"GPU-{gpu_index}",
                    "mig_uuid": f"MIG-g{gpu_index}-{mig_index}",
                }
            )
    return layout


def _synthetic_mig_tokens(
    *,
    gpu_count: int = 4,
    migs_per_gpu: int = 4,
) -> list[str]:
    return [
        str(entry["mig_uuid"])
        for entry in _synthetic_mig_layout(
            gpu_count=int(gpu_count),
            migs_per_gpu=int(migs_per_gpu),
        )
    ]


def _synthetic_nvidia_smi_listing(
    *,
    gpu_count: int = 2,
    migs_per_gpu: int = 2,
) -> str:
    lines: list[str] = []
    for gpu_index in range(int(gpu_count)):
        lines.append(f"GPU {gpu_index}: Fake GPU {gpu_index} (UUID: GPU-{gpu_index})")
        for mig_index in range(int(migs_per_gpu)):
            lines.append(
                "  MIG 1g.24gb Device "
                f"{mig_index}: (UUID: MIG-g{gpu_index}-{mig_index})"
            )
    return "\n".join(lines)


def test_select_top_config_rows_uses_validation_metric_not_test_metric() -> None:
    mod = _load_module()
    rows = mod._select_top_config_rows(
        {
            "aggregate_rows": [
                {
                    "baseline_family": "tree_neural",
                    "tuning_stage": "locked",
                    "train_doc_count": 10240,
                    "config_label": "val_winner",
                    "val_root_mae_mean": 0.05,
                    "test_root_mae_mean": 0.40,
                    "n_runs": 5,
                },
                {
                    "baseline_family": "tree_neural",
                    "tuning_stage": "locked",
                    "train_doc_count": 10240,
                    "config_label": "test_winner",
                    "val_root_mae_mean": 0.09,
                    "test_root_mae_mean": 0.01,
                    "n_runs": 5,
                },
            ]
        },
        baseline_family="tree_neural",
        tuning_stage="locked",
        train_doc_count=10240,
        metric_key="val_root_mae_mean",
        top_k=1,
    )
    assert len(rows) == 1
    assert rows[0]["config_label"] == "val_winner"


def test_job_output_dir_name_preserves_short_names() -> None:
    mod = _load_module()

    assert mod._job_output_dir_name("short_job_name") == "short_job_name"


def test_job_output_dir_name_hashes_long_names_deterministically() -> None:
    mod = _load_module()
    long_name = "recoverable_v4__tree_neural__" + ("verylongsegment__" * 32)

    first = mod._job_output_dir_name(long_name, max_component_length=80)
    second = mod._job_output_dir_name(long_name, max_component_length=80)

    assert first == second
    assert len(first) <= 80
    assert first.endswith(hashlib.sha1(long_name.encode("utf-8")).hexdigest()[:12])


def test_parse_mig_layout_from_nvidia_smi_listing_groups_slices_by_physical_gpu() -> None:
    mod = _load_module()

    entries = mod._parse_mig_layout_from_nvidia_smi_listing(
        _synthetic_nvidia_smi_listing(gpu_count=2, migs_per_gpu=2)
    )

    assert entries == [
        {"gpu_index": 0, "gpu_uuid": "GPU-0", "mig_uuid": "MIG-g0-0"},
        {"gpu_index": 0, "gpu_uuid": "GPU-0", "mig_uuid": "MIG-g0-1"},
        {"gpu_index": 1, "gpu_uuid": "GPU-1", "mig_uuid": "MIG-g1-0"},
        {"gpu_index": 1, "gpu_uuid": "GPU-1", "mig_uuid": "MIG-g1-1"},
    ]


def test_apply_screen_device_order_interleaves_by_physical_gpu() -> None:
    mod = _load_module()
    tokens = _synthetic_mig_tokens(gpu_count=3, migs_per_gpu=2)
    layout_by_uuid = mod._mig_layout_by_uuid(_synthetic_mig_layout(gpu_count=3, migs_per_gpu=2))

    ordered = mod._apply_screen_device_order(
        tokens,
        layout_by_uuid=layout_by_uuid,
        order_mode="interleave_by_physical_gpu",
    )

    assert ordered == [
        "MIG-g0-0",
        "MIG-g1-0",
        "MIG-g2-0",
        "MIG-g0-1",
        "MIG-g1-1",
        "MIG-g2-1",
    ]


def test_worker_command_for_job_includes_batching_flags(tmp_path: Path) -> None:
    mod = _load_module()
    output_dir = tmp_path / "tree_batch_job"
    job = mod._JobSpec(
        family="tree_neural",
        train_doc_count=128,
        benchmark="smoke",
        hardness_grid="",
        grid_cell_ids=(),
        seeds=(0,),
        config=mod._RunConfigSpec(
            label="cfg_batch",
            state_dim=8,
            hidden_dim=16,
            n_epochs=2,
            batch_size=4,
            lr=1e-3,
            weight_decay=0.0,
            tree_posttrain_train_doc_limit=128,
            tree_batch_pack_mode="structure_bucket",
            tree_batch_token_budget=4096,
            tree_batch_node_budget=512,
            tree_batch_autotune=False,
            tree_eval_workers_per_mig=2,
            exact_metric_selection_doc_limit=256,
            exact_metric_selection_interval=5,
            gpu_runtime_data_mode="resident",
            gpu_runtime_bucket_mode="exact_then_bucketed",
            gpu_runtime_preload_splits=("train", "val", "test"),
            gpu_runtime_preload_targets=True,
            gpu_runtime_workers_per_mig=1,
            gpu_runtime_allow_multi_worker_screen=True,
            gpu_runtime_capacity_workers_per_mig=2,
            posttrain_diagnostics_mode="minimal",
            budget_total_calls_per_doc=0.2,
            mass_target_per_doc=0.1,
            full_doc_budget_share=1.0,
            doc_consumption_mode="root_only",
            local_split_mode="balanced",
            local_allocation_policy="breadth_first",
            package_semantics="mass_matched",
            depth_discount_gamma=0.9,
            tree_c2_mode="reconstruction",
        ),
        tuning_stage="stage1_surrogate",
        study_name="teacher_first_tournament",
        study_axis="stage1_surrogate",
        axis_value="cfg_batch",
        selection_metric="teacher_first_total_bound",
    )

    cmd = mod._worker_command_for_job(
        job,
        output_dir=output_dir,
        torch_threads=2,
        use_cuda=False,
    )

    assert "--tree-batch-pack-mode" in cmd
    assert "--tree-batch-token-budget" in cmd
    assert "--tree-batch-node-budget" in cmd
    assert "--no-tree-batch-autotune" in cmd
    assert "--tree-eval-workers-per-mig" in cmd
    assert "--gpu-runtime-data-mode" in cmd
    assert "--gpu-runtime-bucket-mode" in cmd
    assert "--gpu-runtime-preload-splits" in cmd
    assert "--gpu-runtime-workers-per-mig" in cmd
    assert "--memory-probe-jsonl" in cmd
    assert "--exact-metric-selection-doc-limit" in cmd
    assert "--exact-metric-selection-interval" in cmd
    assert "--tree-posttrain-train-doc-limit" in cmd
    assert "--posttrain-diagnostics-mode" in cmd
    assert "--mass-target-per-doc" in cmd
    assert "--package-semantics" in cmd
    assert "--depth-discount-gamma" in cmd
    assert "--config-spec-json-path" in cmd
    assert str(output_dir / "memory_probe.jsonl") in cmd
    config_spec_path = output_dir / "requested_run_config.json"
    assert config_spec_path.exists()
    config_spec = json.loads(config_spec_path.read_text(encoding="utf-8"))
    assert config_spec["baseline_family"] == "tree_neural"
    assert config_spec["tree_c2_mode"] == "reconstruction"
    assert config_spec["depth_discount_gamma"] == pytest.approx(0.9)


def test_job_spec_backfills_config_baseline_family() -> None:
    mod = _load_module()
    job = mod._JobSpec(
        family="official_fno_sumlen",
        train_doc_count=1024,
        benchmark="recoverable_v4",
        hardness_grid="",
        grid_cell_ids=(),
        seeds=(0,),
        config=mod._RunConfigSpec(
            label="cfg",
            state_dim=32,
            hidden_dim=64,
            n_epochs=8,
            batch_size=16,
            lr=1e-3,
            weight_decay=0.0,
        ),
    )

    assert job.config.baseline_family == "official_fno_sumlen"


def test_capacity_grid_preserves_runtime_overrides() -> None:
    mod = _load_module()
    args = argparse.Namespace(
        state_dim=128,
        hidden_dim=512,
        n_epochs=32,
        batch_size=64,
        lr=5e-4,
        weight_decay=0.0,
        tree_local_law_weight=0.3,
        tree_task_objective_weight=None,
        tree_c1_relative_weight=1.0,
        tree_c2_relative_weight=1.0,
        tree_c3_relative_weight=1.0,
        tree_root_supervision_kind="mse",
        tree_checkpoint_metric="val_root_mae",
        tree_stage1_checkpoint_metric="val_root_mae",
        tree_stage1_eval_mode="per_epoch",
        tree_stage1_screen_doc_limit=0,
        tree_stage1_final_exact_doc_limit=0,
        exact_metric_selection_doc_limit=256,
        exact_metric_selection_interval=5,
        tree_batch_pack_mode="fixed_fused",
        tree_batch_token_budget=1234,
        tree_batch_node_budget=567,
        tree_batch_autotune=False,
        tree_eval_workers_per_mig=3,
        gpu_runtime_data_mode="cpu_debug",
        gpu_runtime_bucket_mode="exact_then_bucketed",
        gpu_runtime_preload_splits=("train", "val"),
        gpu_runtime_preload_targets=False,
        gpu_runtime_workers_per_mig=4,
        gpu_runtime_allow_multi_worker_screen=False,
        gpu_runtime_capacity_workers_per_mig=1,
        tree_stage1_artifact_dir="",
        tree_stage1_root_weight=0.0,
        tree_join_bit_weight=0.0,
        tree_training_schedule="two_stage",
        tree_stage1_epochs=12,
        tree_stage2_epochs=20,
        tree_task_head_mode="full_state_scalar",
        tree_theorem_surface_mode="slotwise",
        tree_theorem_count_head_mode="scalar_mse",
        tree_theorem_count_ordinal_weight=1.0,
        tree_theorem_count_scalar_aux_weight=0.25,
        tree_theorem_count_threshold_balance=True,
        tree_theorem_feature_dim=48,
        tree_theorem_feature_hidden_dim=256,
        tree_theorem_score_dim=0,
        tree_theorem_fiber_dim=0,
        tree_theorem_aux_dim=0,
        tree_score_merge_mode="gated_affine",
        tree_phi_compose_weight=1.0,
        tree_phi_contrastive_weight=0.25,
        tree_phi_alignment_loss="cosine_mse",
        tree_c2_mode="reconstruction",
        oracle_metric_name="",
        oracle_same_threshold=0.0,
        oracle_diff_threshold=0.0,
        theorem_feature_adapter="markov_count_sketch",
        theorem_pair_same_threshold=None,
        theorem_pair_diff_threshold=None,
        tree_summary_spec_root_mode="task_split_ablation",
        aligned_sketch_surface="",
        summary_spec_name="",
        slot_count=0,
        tree_theorem_count_dim=0,
        tree_theorem_first_dim=0,
        tree_theorem_last_dim=0,
        leaf_supervision_kind="full_sketch",
        internal_supervision_kind="none",
        internal_label_rate=0.0,
        leaf_exact_supervision=False,
        leaf_label_rate=1.0,
        endpoint_loss_scale=1.0,
        doc_sequence_train_fraction=0.0,
        fixed_leaf_tokens=None,
        budget_total_calls=0,
        budget_total_calls_per_doc=0.0,
        full_doc_budget_share=1.0,
        doc_consumption_mode="",
        local_split_mode="",
        local_allocation_policy="",
        benchmark="recoverable_v4",
        capacity_widths=(64,),
        capacity_modes=(2,),
        capacity_layers=(4,),
    )
    [config] = mod._capacity_grid(args)

    assert config.gpu_runtime_data_mode == "cpu_debug"
    assert config.gpu_runtime_preload_targets is False
    assert config.gpu_runtime_workers_per_mig == 4
    assert config.gpu_runtime_allow_multi_worker_screen is False
    assert config.gpu_runtime_capacity_workers_per_mig == 1
    assert config.tree_batch_pack_mode == "fixed_fused"
    assert config.tree_batch_autotune is False
    assert config.exact_metric_selection_doc_limit == 256
    assert config.exact_metric_selection_interval == 5


def test_run_job_batch_uses_worker_command_helper(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    job = mod._JobSpec(
        family="tree_neural",
        train_doc_count=128,
        benchmark="smoke",
        hardness_grid="",
        grid_cell_ids=(),
        seeds=(0,),
        config=mod._RunConfigSpec(
            label="cfg_runtime",
            state_dim=8,
            hidden_dim=16,
            n_epochs=1,
            batch_size=4,
            lr=1e-3,
            weight_decay=0.0,
            gpu_runtime_data_mode="cpu_debug",
            gpu_runtime_preload_targets=False,
            gpu_runtime_workers_per_mig=3,
        ),
        tuning_stage="capacity_screen",
        test_metrics_hidden_during_selection=True,
        selection_metric="val_root_mae_mean",
    )

    seen: list[tuple[str, bool, int]] = []

    def _fake_worker_command(job_arg, *, output_dir, torch_threads, use_cuda):
        seen.append(
            (
                str(job_arg.config.gpu_runtime_data_mode),
                bool(job_arg.config.gpu_runtime_preload_targets),
                int(job_arg.config.gpu_runtime_workers_per_mig),
            )
        )
        payload = {
            "job_name": str(job_arg.job_name),
            "job_seeds": [0],
            "test_metrics_hidden_during_selection": True,
            "val_root_mae": 0.123,
            "selection_metric_name": "val_root_mae_mean",
            "config_label": str(job_arg.config.label),
            "test_root_mae": 0.456,
            "objective_weights_active": False,
            "parameterization": "",
            "local_law_c1_weight": 0.0,
            "local_law_c2_weight": 0.0,
            "local_law_c3_weight": 0.0,
        }
        return [
            sys.executable,
            "-c",
            "import json; print(json.dumps(" + repr(payload) + "))",
        ]

    monkeypatch.setattr(mod, "_worker_command_for_job", _fake_worker_command)

    result = mod._run_job_batch(
        output_root=tmp_path / "run",
        jobs=[job],
        mig_uuids=["MIG-test-token"],
        resume_enabled=False,
        use_cuda=False,
        torch_threads=1,
        manifest_payload={"jobs": []},
    )

    assert seen == [("cpu_debug", False, 3)]
    assert not result["failed_jobs"]
    assert len(result["completed_jobs"]) == 1


def test_tuning_grid_matches_screen_product() -> None:
    mod = _load_module()
    args = argparse.Namespace(
        state_dim=128,
        hidden_dim=512,
        n_epochs=32,
        batch_size=64,
        lr=5e-4,
        weight_decay=0.0,
        tree_local_law_weight=0.3,
        tree_task_objective_weight=None,
        benchmark="recoverable_v4",
        screen_n_epochs=(32, 64),
        screen_lrs=(5e-4, 2e-4),
        screen_tree_local_law_weights=(0.15, 0.3, 0.45),
    )
    configs = mod._tuning_grid(args)
    assert len(configs) == 12
    assert len({config.label for config in configs}) == 12
    assert {
        (int(config.n_epochs), float(config.lr), float(config.tree_local_law_weight))
        for config in configs
    } == {
        (32, 5e-4, 0.15),
        (32, 5e-4, 0.3),
        (32, 5e-4, 0.45),
        (32, 2e-4, 0.15),
        (32, 2e-4, 0.3),
        (32, 2e-4, 0.45),
        (64, 5e-4, 0.15),
        (64, 5e-4, 0.3),
        (64, 5e-4, 0.45),
        (64, 2e-4, 0.15),
        (64, 2e-4, 0.3),
        (64, 2e-4, 0.45),
    }


def test_render_tuning_summary_markdown_records_selection_contract() -> None:
    mod = _load_module()
    markdown = mod._render_tuning_summary_markdown(
        {
            "benchmark": "recoverable_v4",
            "train_doc_count": 10240,
            "priority_family": "tree_neural",
            "dev_selection_metric": "val_root_mae_mean",
            "test_metrics_hidden_during_selection": True,
            "screen_rankings": [
                {
                    "config_label": "cfg_a",
                    "val_root_mae_mean": 0.08,
                    "train_root_mae_mean": 0.07,
                    "n_runs": 3,
                }
            ],
            "locked_rankings": [
                {
                    "config_label": "cfg_a",
                    "val_root_mae_mean": 0.06,
                    "test_root_mae_mean": 0.05,
                    "n_runs": 5,
                }
            ],
            "winning_config": {
                "config_label": "cfg_a",
                "val_root_mae_mean": 0.06,
                "test_root_mae_mean": 0.05,
            },
            "final_locked_summary_json": "outputs/final_locked/summary.json",
        }
    )
    assert "dev_selection_metric: `val_root_mae_mean`" in markdown
    assert "test metrics hidden during config selection: `True`" in markdown
    assert "## Winning Config" in markdown
    assert "cfg_a" in markdown


def test_main_dispatches_tune_mode(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod.sys, "argv", ["run_tree_neural_full_doc_mig.py", "tune"])
    monkeypatch.setattr(mod, "_launch_tune", lambda args: 17)
    monkeypatch.setattr(mod, "_launch_parity", lambda args: 13)
    monkeypatch.setattr(mod, "_launch_controller", lambda args: 99)

    rc = int(mod.main())

    assert rc == 17


def test_main_dispatches_exact_sanity_mode(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        mod.sys,
        "argv",
        ["run_tree_neural_full_doc_mig.py", "exact_sanity"],
    )
    monkeypatch.setattr(mod, "_launch_exact_sanity", lambda args: 23)
    monkeypatch.setattr(mod, "_launch_budget_frontier", lambda args: 19)
    monkeypatch.setattr(mod, "_launch_controller", lambda args: 99)

    rc = int(mod.main())

    assert rc == 23


def test_fair_fno_parity_tree_config_matches_preset(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "_resolve_benchmark_leaf_tokens", lambda **kwargs: 16)
    args = mod._parser().parse_args(
        [
            "parity",
            "--benchmark",
            "recoverable_v4",
            "--gate-train-doc-count",
            "10240",
            "--state-dim",
            "128",
            "--hidden-dim",
            "512",
            "--n-epochs",
            "32",
            "--batch-size",
            "64",
            "--lr",
            "5e-4",
            "--weight-decay",
            "0.0",
            "--local-law-weight",
            "0.3",
        ]
    )

    config = mod._fair_fno_parity_tree_config(args)

    assert config.label == mod.FAIR_FNO_PARITY_CONFIG_LABEL
    assert config.tree_leaf_fno_width == 128
    assert config.tree_leaf_fno_n_modes == 8
    assert config.tree_leaf_fno_n_layers == 4
    assert config.tree_root_supervision_kind == "count_ce"
    assert config.tree_local_law_weight == 0.3
    assert config.doc_sequence_train_fraction == 0.0


def test_capacity_grid_builds_full_width_modes_layers_product() -> None:
    mod = _load_module()
    args = argparse.Namespace(
        state_dim=128,
        hidden_dim=512,
        n_epochs=32,
        batch_size=64,
        lr=5e-4,
        weight_decay=0.0,
        tree_local_law_weight=0.3,
        tree_task_objective_weight=None,
        capacity_widths=(64, 128, 256),
        capacity_modes=(2, 4, 8),
        capacity_layers=(2, 4, 6),
    )
    configs = mod._capacity_grid(args)
    assert len(configs) == 27
    assert len({config.label for config in configs}) == 27
    assert {config.tree_root_supervision_kind for config in configs} == {"count_ce"}
    assert {config.doc_sequence_train_fraction for config in configs} == {0.0}
    assert {
        (
            int(config.tree_leaf_fno_width),
            int(config.tree_leaf_fno_n_modes),
            int(config.tree_leaf_fno_n_layers),
        )
        for config in configs
    } == {
        (width, n_modes, n_layers)
        for width in (64, 128, 256)
        for n_modes in (2, 4, 8)
        for n_layers in (2, 4, 6)
    }


def test_launch_exact_sanity_builds_tree_only_jobs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "_resolve_benchmark_leaf_tokens", lambda **kwargs: 16)
    args = mod._parser().parse_args(
        [
            "exact_sanity",
            "--output-root",
            str(tmp_path),
            "--benchmark",
            "recoverable_v4",
            "--train-doc-counts",
            "1024",
            "5120",
            "--seeds",
            "0",
            "1",
            "--state-dim",
            "128",
            "--hidden-dim",
            "512",
            "--n-epochs",
            "32",
            "--batch-size",
            "64",
            "--lr",
            "5e-4",
            "--local-law-weight",
            "0.3",
        ]
    )
    bundle = mod.build_exact_sanity_job_bundle(args)
    jobs = list(bundle["jobs"])
    assert jobs
    assert {job.family for job in jobs} == {"tree_neural", "official_fno"}
    assert {job.study_name for job in jobs} == {mod.EXACT_SANITY_STUDY_NAME}
    assert {job.config.label for job in jobs} == {
        mod.FAIR_FNO_PARITY_CONFIG_LABEL,
        "official_fno_root_probe_reference",
        "tree_neural_slot_align_v1_root_only",
        "tree_neural_slot_align_v1_leaf_sampled",
        "tree_neural_slot_align_v1_leaf_dense",
        "tree_neural_slot_align_v1_internal_count_r0p25",
        "tree_neural_slot_align_v1_internal_full_r0p25",
        "tree_neural_slot_align_v1_internal_full_r0p25_task_split_ablation",
        "tree_neural_slot_align_v1_internal_count_dense",
        "tree_neural_slot_align_v1_internal_full_dense",
        "tree_neural_slot_align_v1_balanced_full_r0p25",
        "tree_neural_slot_align_v1_leaf_ep_count_r0p25",
        "tree_neural_slot_align_v1_internal_full_r0p5",
        "tree_neural_slot_align_v1_unified_f_root_only",
        "tree_neural_slot_align_v1_unified_f_count_r0p25",
        "tree_neural_slot_align_v1_unified_f_full_r0p25",
        "tree_neural_slot_align_v1_unified_f_count_dense",
    }
    assert {
        job.config.tree_root_supervision_kind
        for job in jobs
        if job.config.label == mod.FAIR_FNO_PARITY_CONFIG_LABEL
    } == {"count_ce"}
    assert {
        job.config.tree_root_supervision_kind
        for job in jobs
        if job.family == "tree_neural"
        and job.config.label != mod.FAIR_FNO_PARITY_CONFIG_LABEL
    } == {"mse"}
    assert {job.config.doc_sequence_train_fraction for job in jobs} == {0.0}
    assert {
        job.config.summary_spec_name
        for job in jobs
        if job.family == "tree_neural"
        and job.config.label != mod.FAIR_FNO_PARITY_CONFIG_LABEL
    } == {"markov_count_sketch"}
    assert {
        (job.config.label, job.config.leaf_exact_supervision)
        for job in jobs
        if job.family == "tree_neural"
        and job.config.label != mod.FAIR_FNO_PARITY_CONFIG_LABEL
    } == {
        ("tree_neural_slot_align_v1_root_only", False),
        ("tree_neural_slot_align_v1_leaf_sampled", False),
        ("tree_neural_slot_align_v1_leaf_dense", False),
        ("tree_neural_slot_align_v1_internal_count_r0p25", False),
        ("tree_neural_slot_align_v1_internal_full_r0p25", False),
        ("tree_neural_slot_align_v1_internal_full_r0p25_task_split_ablation", False),
        ("tree_neural_slot_align_v1_internal_count_dense", False),
        ("tree_neural_slot_align_v1_internal_full_dense", False),
        ("tree_neural_slot_align_v1_balanced_full_r0p25", False),
        ("tree_neural_slot_align_v1_leaf_ep_count_r0p25", True),
        ("tree_neural_slot_align_v1_internal_full_r0p5", False),
        ("tree_neural_slot_align_v1_unified_f_root_only", False),
        ("tree_neural_slot_align_v1_unified_f_count_r0p25", False),
        ("tree_neural_slot_align_v1_unified_f_full_r0p25", False),
        ("tree_neural_slot_align_v1_unified_f_count_dense", False),
    }
    assert {
        (job.config.label, job.config.leaf_supervision_kind)
        for job in jobs
        if job.family == "tree_neural"
        and job.config.label != mod.FAIR_FNO_PARITY_CONFIG_LABEL
    } == {
        ("tree_neural_slot_align_v1_root_only", "count_only"),
        ("tree_neural_slot_align_v1_leaf_sampled", "full_sketch"),
        ("tree_neural_slot_align_v1_leaf_dense", "full_sketch"),
        ("tree_neural_slot_align_v1_internal_count_r0p25", "count_only"),
        ("tree_neural_slot_align_v1_internal_full_r0p25", "full_sketch"),
        ("tree_neural_slot_align_v1_internal_full_r0p25_task_split_ablation", "full_sketch"),
        ("tree_neural_slot_align_v1_internal_count_dense", "count_only"),
        ("tree_neural_slot_align_v1_internal_full_dense", "full_sketch"),
        ("tree_neural_slot_align_v1_balanced_full_r0p25", "full_sketch"),
        ("tree_neural_slot_align_v1_leaf_ep_count_r0p25", "full_sketch"),
        ("tree_neural_slot_align_v1_internal_full_r0p5", "full_sketch"),
        ("tree_neural_slot_align_v1_unified_f_root_only", "count_only"),
        ("tree_neural_slot_align_v1_unified_f_count_r0p25", "count_only"),
        ("tree_neural_slot_align_v1_unified_f_full_r0p25", "full_sketch"),
        ("tree_neural_slot_align_v1_unified_f_count_dense", "count_only"),
    }
    assert {
        job.config.slot_count
        for job in jobs
        if job.family == "tree_neural"
        and job.config.label != mod.FAIR_FNO_PARITY_CONFIG_LABEL
    } == {4}
    assert {
        job.config.tree_theorem_count_head_mode
        for job in jobs
        if job.family == "tree_neural"
        and job.config.label != mod.FAIR_FNO_PARITY_CONFIG_LABEL
    } == {"scalar_mse"}
    assert {
        job.config.tree_theorem_surface_mode
        for job in jobs
        if job.family == "tree_neural"
        and job.config.label != mod.FAIR_FNO_PARITY_CONFIG_LABEL
    } == {"shared_bottleneck"}
    assert {
        (
            job.config.tree_theorem_feature_dim,
            job.config.tree_theorem_feature_hidden_dim,
            job.config.tree_phi_compose_weight,
            job.config.tree_phi_contrastive_weight,
            job.config.tree_phi_alignment_loss,
        )
        for job in jobs
        if job.family == "tree_neural"
        and job.config.label != mod.FAIR_FNO_PARITY_CONFIG_LABEL
    } == {(48, 256, 1.0, 0.25, "cosine_mse")}
    assert {
        (
            job.config.tree_theorem_count_ordinal_weight,
            job.config.tree_theorem_count_scalar_aux_weight,
            job.config.tree_theorem_count_threshold_balance,
        )
        for job in jobs
        if job.family == "tree_neural"
        and job.config.label != mod.FAIR_FNO_PARITY_CONFIG_LABEL
    } == {(1.0, 0.25, True)}
    assert {
        (
            job.config.tree_theorem_count_dim,
            job.config.tree_theorem_first_dim,
            job.config.tree_theorem_last_dim,
        )
        for job in jobs
        if job.family == "tree_neural"
        and job.config.label != mod.FAIR_FNO_PARITY_CONFIG_LABEL
    } == {(8, 8, 8)}
    assert {
        job.config.tree_stage1_checkpoint_metric
        for job in jobs
        if job.family == "tree_neural"
        and job.config.label != mod.FAIR_FNO_PARITY_CONFIG_LABEL
    } == {"val_theorem_bootstrap_direct"}
    assert {
        job.config.tree_training_schedule
        for job in jobs
        if job.family == "tree_neural"
        and job.config.label != mod.FAIR_FNO_PARITY_CONFIG_LABEL
    } == {"two_stage"}
    assert {
        (job.config.label, job.config.tree_summary_spec_root_mode)
        for job in jobs
        if job.family == "tree_neural"
        and job.config.label != mod.FAIR_FNO_PARITY_CONFIG_LABEL
    } == {
        ("tree_neural_slot_align_v1_root_only", "factored_theorem_readout"),
        ("tree_neural_slot_align_v1_leaf_sampled", "factored_theorem_readout"),
        ("tree_neural_slot_align_v1_leaf_dense", "factored_theorem_readout"),
        ("tree_neural_slot_align_v1_internal_count_r0p25", "factored_theorem_readout"),
        ("tree_neural_slot_align_v1_internal_full_r0p25", "factored_theorem_readout"),
        ("tree_neural_slot_align_v1_internal_full_r0p25_task_split_ablation", "task_split_ablation"),
        ("tree_neural_slot_align_v1_internal_count_dense", "factored_theorem_readout"),
        ("tree_neural_slot_align_v1_internal_full_dense", "factored_theorem_readout"),
        ("tree_neural_slot_align_v1_balanced_full_r0p25", "factored_theorem_readout"),
        ("tree_neural_slot_align_v1_leaf_ep_count_r0p25", "factored_theorem_readout"),
        ("tree_neural_slot_align_v1_internal_full_r0p5", "factored_theorem_readout"),
        ("tree_neural_slot_align_v1_unified_f_root_only", "unified_f"),
        ("tree_neural_slot_align_v1_unified_f_count_r0p25", "unified_f"),
        ("tree_neural_slot_align_v1_unified_f_full_r0p25", "unified_f"),
        ("tree_neural_slot_align_v1_unified_f_count_dense", "unified_f"),
    }
    assert {
        (job.config.tree_stage1_epochs, job.config.tree_stage2_epochs)
        for job in jobs
        if job.family == "tree_neural"
        and job.config.label != mod.FAIR_FNO_PARITY_CONFIG_LABEL
    } == {(12, 20)}
    assert {
        (job.config.label, job.config.leaf_label_rate)
        for job in jobs
        if job.family == "tree_neural"
        and job.config.label != mod.FAIR_FNO_PARITY_CONFIG_LABEL
    } == {
        ("tree_neural_slot_align_v1_root_only", 0.0),
        ("tree_neural_slot_align_v1_leaf_sampled", 0.25),
        ("tree_neural_slot_align_v1_leaf_dense", 1.0),
        ("tree_neural_slot_align_v1_internal_count_r0p25", 0.25),
        ("tree_neural_slot_align_v1_internal_full_r0p25", 0.25),
        ("tree_neural_slot_align_v1_internal_full_r0p25_task_split_ablation", 0.25),
        ("tree_neural_slot_align_v1_internal_count_dense", 1.0),
        ("tree_neural_slot_align_v1_internal_full_dense", 1.0),
        ("tree_neural_slot_align_v1_balanced_full_r0p25", 0.25),
        ("tree_neural_slot_align_v1_leaf_ep_count_r0p25", 0.25),
        ("tree_neural_slot_align_v1_internal_full_r0p5", 0.25),
        ("tree_neural_slot_align_v1_unified_f_root_only", 0.0),
        ("tree_neural_slot_align_v1_unified_f_count_r0p25", 0.25),
        ("tree_neural_slot_align_v1_unified_f_full_r0p25", 0.25),
        ("tree_neural_slot_align_v1_unified_f_count_dense", 1.0),
    }
    assert {
        (job.train_doc_count, job.config.label)
        for job in jobs
        if job.train_doc_count == 1024
    } == {
        (1024, mod.FAIR_FNO_PARITY_CONFIG_LABEL),
        (1024, "official_fno_root_probe_reference"),
        (1024, "tree_neural_slot_align_v1_root_only"),
        (1024, "tree_neural_slot_align_v1_leaf_sampled"),
        (1024, "tree_neural_slot_align_v1_leaf_dense"),
        (1024, "tree_neural_slot_align_v1_internal_count_r0p25"),
        (1024, "tree_neural_slot_align_v1_internal_full_r0p25"),
        (1024, "tree_neural_slot_align_v1_internal_full_r0p25_task_split_ablation"),
        (1024, "tree_neural_slot_align_v1_internal_count_dense"),
        (1024, "tree_neural_slot_align_v1_internal_full_dense"),
        (1024, "tree_neural_slot_align_v1_balanced_full_r0p25"),
        (1024, "tree_neural_slot_align_v1_leaf_ep_count_r0p25"),
        (1024, "tree_neural_slot_align_v1_unified_f_root_only"),
        (1024, "tree_neural_slot_align_v1_unified_f_count_r0p25"),
        (1024, "tree_neural_slot_align_v1_unified_f_full_r0p25"),
        (1024, "tree_neural_slot_align_v1_unified_f_count_dense"),
    }
    assert {
        (job.train_doc_count, job.config.label)
        for job in jobs
        if job.train_doc_count == 5120
    } == {
        (5120, "tree_neural_slot_align_v1_root_only"),
        (5120, "tree_neural_slot_align_v1_leaf_sampled"),
        (5120, "tree_neural_slot_align_v1_leaf_dense"),
        (5120, "tree_neural_slot_align_v1_internal_count_r0p25"),
        (5120, "tree_neural_slot_align_v1_internal_full_r0p25"),
        (5120, "tree_neural_slot_align_v1_internal_full_r0p25_task_split_ablation"),
        (5120, "tree_neural_slot_align_v1_internal_count_dense"),
        (5120, "tree_neural_slot_align_v1_internal_full_dense"),
        (5120, "tree_neural_slot_align_v1_balanced_full_r0p25"),
        (5120, "tree_neural_slot_align_v1_leaf_ep_count_r0p25"),
        (5120, "tree_neural_slot_align_v1_internal_full_r0p5"),
        (5120, "tree_neural_slot_align_v1_unified_f_root_only"),
        (5120, "tree_neural_slot_align_v1_unified_f_count_r0p25"),
        (5120, "tree_neural_slot_align_v1_unified_f_full_r0p25"),
        (5120, "tree_neural_slot_align_v1_unified_f_count_dense"),
    }

    def _fake_write_summary_outputs(output_root):
        payload = {"runs": [], "aggregate_rows": []}
        summary_json = Path(output_root) / "summary.json"
        summary_md = Path(output_root) / "summary.md"
        summary_json.write_text(json.dumps(payload), encoding="utf-8")
        summary_md.write_text("# Summary\n", encoding="utf-8")
        payload["summary_json"] = str(summary_json)
        payload["summary_md"] = str(summary_md)
        return payload

    monkeypatch.setattr(mod, "_write_summary_outputs", _fake_write_summary_outputs)
    result = mod.finalize_exact_sanity_output(tmp_path)
    assert Path(result["tree_neural_exact_sanity_summary_json"]).exists()
    assert Path(result["tree_neural_exact_sanity_summary_md"]).exists()


def test_launch_parity_builds_gate_then_backfill_jobs(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "_resolve_benchmark_leaf_tokens", lambda **kwargs: 16)
    captured = _capture_scheduler_jobs(monkeypatch, mod)
    args = mod._parser().parse_args(
        [
            "parity",
            "--output-root",
            str(tmp_path / "parity"),
            "--mig-uuids",
            "MIG-a",
            "--no-use-cuda",
            "--no-run-aux-upper-bound",
        ]
    )
    graph = mod._build_scheduler_graph(
        args,
        output_root=tmp_path / "parity",
        mig_uuids=["MIG-a"],
    )
    gate_jobs = [
        entry["job"]
        for entry in captured
        if str(entry["item_id"]).startswith("parity::gate::")
    ]
    backfill_jobs = [
        entry["job"]
        for entry in captured
        if str(entry["item_id"]).startswith("parity::backfill::")
    ]
    assert {job.family for job in gate_jobs} == set(mod.PARITY_COMPARISON_FAMILIES)
    assert {job.train_doc_count for job in gate_jobs} == {mod.PARITY_GATE_TRAIN_DOC_COUNT}
    parity_tree_jobs = [job for job in gate_jobs if job.family in mod.PARITY_TREE_FAMILIES]
    assert {job.config.label for job in parity_tree_jobs} == {mod.FAIR_FNO_PARITY_CONFIG_LABEL}
    assert {job.config.tree_root_supervision_kind for job in parity_tree_jobs} == {"count_ce"}
    assert {job.config.tree_leaf_fno_width for job in parity_tree_jobs} == {128}
    assert {job.config.tree_leaf_fno_n_modes for job in parity_tree_jobs} == {8}
    assert {job.config.tree_leaf_fno_n_layers for job in parity_tree_jobs} == {4}
    assert {job.train_doc_count for job in backfill_jobs} == set(mod.PARITY_SCALE_CURVE_TRAIN_DOC_COUNTS)
    assert {job.family for job in backfill_jobs} == set(mod.PARITY_COMPARISON_FAMILIES)
    assert {int(entry["gpu_slots"]) for entry in captured} == {1}
    reduce_item = _find_scheduler_item(graph["items"], "parity::reduce")
    assert set(reduce_item.deps) == {
        str(entry["item_id"]) for entry in captured
    }


def test_launch_parity_backfills_even_when_gate_fails(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "_resolve_benchmark_leaf_tokens", lambda **kwargs: 16)
    captured = _capture_scheduler_jobs(monkeypatch, mod)
    args = mod._parser().parse_args(
        [
            "parity",
            "--output-root",
            str(tmp_path / "parity"),
            "--mig-uuids",
            "MIG-a",
            "--no-use-cuda",
            "--no-run-aux-upper-bound",
        ]
    )
    mod._build_scheduler_graph(
        args,
        output_root=tmp_path / "parity",
        mig_uuids=["MIG-a"],
    )
    backfill_jobs = [
        entry["job"]
        for entry in captured
        if str(entry["item_id"]).startswith("parity::backfill::")
    ]
    assert {job.train_doc_count for job in backfill_jobs} == set(mod.PARITY_SCALE_CURVE_TRAIN_DOC_COUNTS)
    assert {job.family for job in backfill_jobs} == set(mod.PARITY_COMPARISON_FAMILIES)


def test_launch_parity_skips_backfill_when_disabled(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "_resolve_benchmark_leaf_tokens", lambda **kwargs: 16)
    captured = _capture_scheduler_jobs(monkeypatch, mod)
    args = mod._parser().parse_args(
        [
            "parity",
            "--output-root",
            str(tmp_path / "parity"),
            "--mig-uuids",
            "MIG-a",
            "--no-use-cuda",
            "--no-run-aux-upper-bound",
            "--no-backfill-on-success",
        ]
    )
    graph = mod._build_scheduler_graph(
        args,
        output_root=tmp_path / "parity",
        mig_uuids=["MIG-a"],
    )
    assert not [
        entry
        for entry in captured
        if str(entry["item_id"]).startswith("parity::backfill::")
    ]
    assert not [
        item for item in graph["items"] if str(item.item_id).startswith("parity::backfill::")
    ]


def test_launch_parity_respects_custom_family_selection(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "_resolve_benchmark_leaf_tokens", lambda **kwargs: 16)
    captured = _capture_scheduler_jobs(monkeypatch, mod)
    args = mod._parser().parse_args(
        [
            "parity",
            "--output-root",
            str(tmp_path / "parity"),
            "--mig-uuids",
            "MIG-a",
            "--no-use-cuda",
            "--no-run-aux-upper-bound",
            "--tree-families",
            "tree_neural",
            "--fno-families",
            "official_fno",
        ]
    )
    mod._build_scheduler_graph(
        args,
        output_root=tmp_path / "parity",
        mig_uuids=["MIG-a"],
    )
    gate_jobs = [
        entry["job"]
        for entry in captured
        if str(entry["item_id"]).startswith("parity::gate::")
    ]
    backfill_jobs = [
        entry["job"]
        for entry in captured
        if str(entry["item_id"]).startswith("parity::backfill::")
    ]
    assert {job.family for job in gate_jobs} == {"official_fno", "tree_neural"}
    assert {job.family for job in backfill_jobs} == {"official_fno", "tree_neural"}


def test_launch_parity_builds_upper_bound_jobs(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "_resolve_benchmark_leaf_tokens", lambda **kwargs: 16)
    captured = _capture_scheduler_jobs(monkeypatch, mod)
    args = mod._parser().parse_args(
        [
            "parity",
            "--output-root",
            str(tmp_path / "parity"),
            "--mig-uuids",
            "MIG-a",
            "--no-use-cuda",
            "--no-backfill-on-success",
            "--upper-bound-aux-fractions",
            "0.25",
            "1.0",
        ]
    )
    graph = mod._build_scheduler_graph(
        args,
        output_root=tmp_path / "parity",
        mig_uuids=["MIG-a"],
    )
    upper_jobs = [
        entry["job"]
        for entry in captured
        if str(entry["item_id"]).startswith("parity::upper::")
    ]
    assert {job.tuning_stage for job in upper_jobs} == {"upper_bound"}
    assert {job.family for job in upper_jobs} == set(mod.PARITY_TREE_FAMILIES)
    assert {
        float(job.config.doc_sequence_train_fraction) for job in upper_jobs
    } == {0.25, 1.0}
    assert {
        str(job.config.label) for job in upper_jobs
    } == {f"{mod.FAIR_FNO_PARITY_CONFIG_LABEL}_aux25", f"{mod.FAIR_FNO_PARITY_CONFIG_LABEL}_aux100"}
    assert any(str(item.item_id).startswith("parity::upper::") for item in graph["items"])


def test_exact_sanity_summary_preserves_config_level_condition_ids() -> None:
    mod = _load_module()

    def _run(config_label: str) -> dict:
        level_payload = {
            level: {
                "direct": {
                    "count_mae": 0.1,
                    "count_match_rate": 0.9,
                    "first_accuracy": 0.9,
                    "last_accuracy": 0.9,
                    "exact_summary_match_rate": 0.8,
                },
                "probe": {
                    "count_mae": 0.1,
                    "count_match_rate": 0.9,
                    "first_accuracy": 0.9,
                    "last_accuracy": 0.9,
                    "exact_summary_match_rate": 0.8,
                },
            }
            for level in mod.EXACT_SANITY_LEVELS
        }
        level_payload["merge"]["decoded_consistency"] = {
            "merge_join_bit_accuracy": 0.95,
            "merge_decoded_consistency_count_mae": 0.1,
            "merge_decoded_consistency_first_accuracy": 0.9,
            "merge_decoded_consistency_last_accuracy": 0.9,
        }
        exact_witness = {
            split: {
                "law_metrics": {
                    metric: 0.0 for metric in mod.EXACT_SANITY_LAW_METRICS
                },
                **{
                    level: {
                        "direct": {
                            metric: 1.0 if metric != "count_mae" else 0.0
                            for metric in mod.EXACT_SANITY_COMPONENT_METRICS
                        },
                        "probe_control": {
                            metric: 1.0 if metric != "count_mae" else 0.0
                            for metric in mod.EXACT_SANITY_COMPONENT_METRICS
                        },
                    }
                    for level in mod.EXACT_SANITY_LEVELS
                },
            }
            for split in ("train", "val", "test")
        }
        return {
            "study_name": mod.EXACT_SANITY_STUDY_NAME,
            "baseline_family": mod.EXACT_SANITY_FAMILY,
            "train_doc_count": 1024,
            "config_label": config_label,
            "seed": 0,
            "summary_spec_name": "markov_count_sketch",
            "slot_count": 4,
            "leaf_supervision_kind": "full_sketch",
            "internal_supervision_kind": "full_sketch",
            "internal_label_rate": 0.25,
            "leaf_label_rate": 0.25,
            "tree_training_schedule": "two_stage",
            "tree_stage1_epochs": 12,
            "tree_stage2_epochs": 20,
            "tree_checkpoint_metric": "val_exact_sketch_direct",
            "tree_stage1_checkpoint_metric": "val_theorem_bootstrap_direct",
            "tree_summary_spec_root_mode": (
                "task_split_ablation"
                if config_label.endswith("_task_split_ablation")
                else "factored_theorem_readout"
            ),
            "exact_sketch_diagnostics": {
                "exact_witness": exact_witness,
                "tree_neural": {
                    split: dict(level_payload)
                    for split in ("train", "val", "test")
                },
                "direct_selection_metrics": {
                    split: {
                        "task_root_mae": 0.2,
                        "task_root_mae_ablation": 0.2,
                        "c2_on_range_exact_match": 0.9,
                        "val_theorem_bootstrap_direct": 0.3,
                    }
                    for split in ("train", "val", "test")
                },
                "failure_attribution": {
                    "bucket": "leaf_boundary_encoding_gap",
                    "leaf_gap_score": 0.2,
                    "merge_gap_score": 0.1,
                    "subtree_label_value_gap_score": 0.05,
                    "readout_gap_score": 0.02,
                },
                "theorem_contract": {
                    "summary_ref": "MarkovCountSketch",
                    "codec_ref": "SketchCodecExactAssumptions",
                    "bundle_ref": "approx_bundle_of_nodewise",
                },
            },
        }

    summary = mod._tree_neural_exact_sanity_summary(
        {
            "benchmark": "recoverable_v4",
            "runs": [
                _run("tree_neural_slot_align_v1_balanced_full_r0p25"),
                _run("tree_neural_slot_align_v1_leaf_ep_count_r0p25"),
            ],
        }
    )

    conditions = list(summary["groups"][0]["conditions"])
    assert {
        condition["condition_id"] for condition in conditions
    } == {
        "tree_neural_slot_align_v1_balanced_full_r0p25",
        "tree_neural_slot_align_v1_leaf_ep_count_r0p25",
    }


def test_exact_sanity_summary_reports_root_mode_alignment_pairs() -> None:
    mod = _load_module()

    def _run(config_label: str, *, root_mode: str) -> dict:
        return {
            "study_name": mod.EXACT_SANITY_STUDY_NAME,
            "baseline_family": mod.EXACT_SANITY_FAMILY,
            "train_doc_count": 1024,
            "config_label": config_label,
            "seed": 0,
            "summary_spec_name": "markov_count_sketch",
            "slot_count": 4,
            "leaf_supervision_kind": "full_sketch",
            "internal_supervision_kind": "full_sketch",
            "internal_label_rate": 0.25,
            "leaf_label_rate": 0.25,
            "tree_training_schedule": "two_stage",
            "tree_stage1_epochs": 12,
            "tree_stage2_epochs": 20,
            "tree_checkpoint_metric": "val_exact_sketch_direct",
            "tree_stage1_checkpoint_metric": "val_theorem_bootstrap_direct",
            "tree_summary_spec_root_mode": root_mode,
            "exact_sketch_diagnostics": {
                "tree_neural": {
                    "test": {
                        "leaf": {"probe": {"exact_summary_match_rate": 0.8}},
                        "merge": {
                            "probe": {"exact_summary_match_rate": 0.85},
                            "decoded_consistency": {"merge_join_bit_accuracy": 0.95},
                        },
                        "root": {
                            "direct": {
                                "count_mae": (
                                    0.12
                                    if root_mode == "factored_theorem_readout"
                                    else 0.16
                                )
                            },
                            "probe": {"count_mae": 0.2},
                        },
                    }
                },
                "exact_witness": {
                    "test": {"law_metrics": {metric: 0.0 for metric in mod.EXACT_SANITY_LAW_METRICS}}
                },
                "direct_selection_metrics": {
                    "test": {
                        "task_root_mae": 0.22,
                        "task_root_mae_ablation": (
                            0.22
                            if root_mode == "factored_theorem_readout"
                            else 0.25
                        ),
                        "c2_on_range_exact_match": 0.9,
                        "val_theorem_bootstrap_direct": 0.3,
                    }
                },
                "theorem_contract": {},
            },
        }

    summary = mod._tree_neural_exact_sanity_summary(
        {
            "benchmark": "recoverable_v4",
            "runs": [
                _run(
                    "tree_neural_slot_align_v1_internal_full_r0p25",
                    root_mode="factored_theorem_readout",
                ),
                _run(
                    "tree_neural_slot_align_v1_internal_full_r0p25_task_split_ablation",
                    root_mode="task_split_ablation",
                ),
            ],
        }
    )

    alignment = dict(
        (summary["groups"][0]["acceptance_readout"] or {}).get(
            "root_mode_alignment_by_base_config",
            {},
        )
    )
    pair = dict(alignment["tree_neural_slot_align_v1_internal_full_r0p25"])
    assert pair["aligned_primary_condition_id"] == "tree_neural_slot_align_v1_internal_full_r0p25"
    assert pair["theorem_primary_condition_id"] == "tree_neural_slot_align_v1_internal_full_r0p25"
    assert pair["aligned_primary_root_mode"] == "factored_theorem_readout"
    assert pair["task_split_ablation_condition_id"] == (
        "tree_neural_slot_align_v1_internal_full_r0p25_task_split_ablation"
    )
    assert pair["aligned_primary_improves_or_matches_theorem_root"] is True


def test_launch_capacity_builds_screen_then_locked_jobs(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    captured = _capture_scheduler_jobs(monkeypatch, mod)
    args = mod._parser().parse_args(
        [
            "capacity",
            "--output-root",
            str(tmp_path / "capacity"),
            "--mig-uuids",
            "MIG-a",
            "--no-use-cuda",
        ]
    )
    bundle = mod.build_capacity_screen_job_bundle(args)
    graph = mod._build_scheduler_graph(
        args,
        output_root=tmp_path / "capacity",
        mig_uuids=["MIG-a"],
    )
    screen_jobs = [
        entry["job"]
        for entry in captured
        if str(entry["item_id"]).startswith("capacity::screen::")
    ]
    assert {job.tuning_stage for job in screen_jobs} == {"capacity_screen"}
    assert {job.family for job in screen_jobs} == {"tree_neural"}
    assert len({job.config.label for job in screen_jobs}) == 27
    selected_configs = sorted(bundle["config_by_label"].values(), key=lambda cfg: cfg.label)[:3]

    def _fake_finalize_capacity_screen_output(**kwargs):
        output_root = Path(kwargs["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        screen_summary_json = output_root / "tree_fno_capacity_screen_summary.json"
        screen_summary_md = output_root / "tree_fno_capacity_screen_summary.md"
        screen_summary_json.write_text("{}", encoding="utf-8")
        screen_summary_md.write_text("# Screen\n", encoding="utf-8")
        top_rankings = [
            {"config_label": str(config.label)} for config in selected_configs
        ]
        return {
            "screen_rankings": list(top_rankings),
            "top_rankings": list(top_rankings),
            "locked_configs": list(selected_configs),
            "screen_summary_json": str(screen_summary_json),
            "screen_summary_md": str(screen_summary_md),
        }

    def _fake_finalize_capacity_locked_output(**kwargs):
        output_root = Path(kwargs["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        locked_summary_json = output_root / "tree_fno_capacity_locked_summary.json"
        locked_summary_md = output_root / "tree_fno_capacity_locked_summary.md"
        locked_summary_json.write_text("{}", encoding="utf-8")
        locked_summary_md.write_text("# Locked\n", encoding="utf-8")
        return {
            "tree_fno_capacity_locked_summary_json": str(locked_summary_json),
            "tree_fno_capacity_locked_summary_md": str(locked_summary_md),
            "winning_config_label": str(selected_configs[0].label),
        }

    monkeypatch.setattr(mod, "finalize_capacity_screen_output", _fake_finalize_capacity_screen_output)
    monkeypatch.setattr(mod, "finalize_capacity_locked_output", _fake_finalize_capacity_locked_output)
    screen_reduce = _find_scheduler_item(graph["items"], "capacity::screen::reduce")
    callback_result = screen_reduce.callback()
    locked_jobs = [
        entry["job"]
        for entry in captured
        if str(entry["item_id"]).startswith("capacity::locked::")
    ]
    assert {job.tuning_stage for job in locked_jobs} == {"capacity_locked"}
    assert len({job.config.label for job in locked_jobs}) == 3
    assert {int(entry["gpu_slots"]) for entry in captured} == {1}
    locked_reduce = _find_scheduler_item(callback_result["new_items"], "capacity::locked::reduce")
    locked_reduce.callback()
    assert (tmp_path / "capacity" / "tree_fno_capacity_screen_summary.json").exists()
    assert (tmp_path / "capacity" / "tree_fno_capacity_locked_summary.json").exists()


def test_capacity_profile_root_only_historical_replay_uses_profile_defaults(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    args = mod._parser().parse_args(
        [
            "capacity",
            "--output-root",
            str(tmp_path / "capacity"),
            "--mig-uuids",
            "MIG-a",
            "--capacity-profile",
            "root_only_parity_historical_replay",
            "--no-use-cuda",
        ]
    )

    bundle = mod.build_capacity_screen_job_bundle(args)

    assert bundle["screen_manifest_payload"]["capacity_profile"] == (
        "root_only_parity_historical_replay"
    )
    assert bundle["screen_manifest_payload"]["capacity_widths"] == [128]
    assert bundle["screen_manifest_payload"]["capacity_modes"] == [8]
    assert bundle["screen_manifest_payload"]["capacity_layers"] == [4]
    assert bundle["screen_manifest_payload"]["capacity_state_dims"] == [128]
    assert bundle["screen_manifest_payload"]["capacity_hidden_dims"] == [512]
    assert bundle["screen_manifest_payload"]["capacity_n_epochs"] == [52]
    assert bundle["screen_manifest_payload"]["capacity_tree_training_schedules"] == [
        "two_stage"
    ]
    assert bundle["screen_manifest_payload"]["capacity_tree_checkpoint_metrics"] == [
        "val_exact_sketch_direct"
    ]
    assert bundle["screen_manifest_payload"][
        "capacity_tree_stage1_checkpoint_metrics"
    ] == ["val_theorem_bootstrap_direct"]
    assert bundle["screen_manifest_payload"]["capacity_tree_stage1_root_weights"] == [
        0.0
    ]
    assert bundle["screen_manifest_payload"]["capacity_slot_counts"] == [4]
    assert bundle["screen_manifest_payload"]["capacity_fixed_leaf_tokens"] == [16]

    configs = {job.config.label: job.config for job in bundle["screen_jobs"]}
    assert len(configs) == 1
    config = next(iter(configs.values()))
    assert config.tree_leaf_fno_width == 128
    assert config.tree_leaf_fno_n_modes == 8
    assert config.tree_leaf_fno_n_layers == 4
    assert config.leaf_supervision_kind == "count_only"
    assert config.leaf_label_rate == 0.0
    assert config.internal_supervision_kind == "none"
    assert config.internal_label_rate == 0.0
    assert config.state_dim == 128
    assert config.hidden_dim == 512
    assert config.n_epochs == 52
    assert config.tree_training_schedule == "two_stage"
    assert config.tree_checkpoint_metric == "val_exact_sketch_direct"
    assert config.tree_stage1_checkpoint_metric == "val_theorem_bootstrap_direct"
    assert config.tree_stage1_root_weight == 0.0
    assert config.slot_count == 4
    assert config.fixed_leaf_tokens == 16


def test_capacity_profile_explicit_axes_override_profile_defaults(tmp_path: Path) -> None:
    mod = _load_module()
    args = mod._parser().parse_args(
        [
            "capacity",
            "--output-root",
            str(tmp_path / "capacity"),
            "--mig-uuids",
            "MIG-a",
            "--capacity-profile",
            "root_only_parity_historical_replay",
            "--capacity-widths",
            "64",
            "128",
            "--capacity-state-dims",
            "256",
            "--capacity-tree-training-schedules",
            "single_stage",
            "--no-use-cuda",
        ]
    )

    bundle = mod.build_capacity_screen_job_bundle(args)

    assert bundle["screen_manifest_payload"]["capacity_widths"] == [64, 128]
    assert bundle["screen_manifest_payload"]["capacity_state_dims"] == [256]
    assert bundle["screen_manifest_payload"]["capacity_tree_training_schedules"] == [
        "single_stage"
    ]
    configs = {job.config.label: job.config for job in bundle["screen_jobs"]}
    assert {config.tree_leaf_fno_width for config in configs.values()} == {64, 128}
    assert {config.leaf_supervision_kind for config in configs.values()} == {
        "count_only"
    }
    assert {config.internal_supervision_kind for config in configs.values()} == {
        "none"
    }
    assert {config.state_dim for config in configs.values()} == {256}
    assert {config.tree_training_schedule for config in configs.values()} == {
        "single_stage"
    }
    assert {config.tree_stage1_checkpoint_metric for config in configs.values()} == {
        "val_theorem_bootstrap_direct"
    }


def test_capacity_screen_runtime_overrides_apply_only_to_screen_phase(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    captured = _capture_scheduler_jobs(monkeypatch, mod)
    args = mod._parser().parse_args(
        [
            "capacity",
            "--output-root",
            str(tmp_path / "capacity"),
            "--mig-uuids",
            "MIG-a",
            "--base-config-preset",
            "common_factorized_sketch_v1",
            "--screen-gpu-runtime-preload-splits",
            "train",
            "--no-screen-gpu-runtime-preload-targets",
            "--no-use-cuda",
        ]
    )
    bundle = mod.build_capacity_screen_job_bundle(args)

    assert bundle["screen_manifest_payload"]["screen_runtime_overrides"] == {
        "gpu_runtime_preload_splits": ("train",),
        "gpu_runtime_preload_targets": False,
    }

    screen_jobs = [
        entry["job"]
        for entry in captured
        if str(entry["item_id"]).startswith("capacity::screen::")
    ]
    assert screen_jobs == []
    assert {job.config.gpu_runtime_preload_splits for job in bundle["screen_jobs"]} == {
        ("train",)
    }
    assert {job.config.gpu_runtime_preload_targets for job in bundle["screen_jobs"]} == {
        False
    }

    locked_config = bundle["config_by_label"][bundle["screen_jobs"][0].config.label]
    assert locked_config.gpu_runtime_preload_splits == ("train", "val", "test")
    assert locked_config.gpu_runtime_preload_targets is True

    graph = mod._build_scheduler_graph(
        args,
        output_root=tmp_path / "capacity",
        mig_uuids=["MIG-a"],
    )
    screen_jobs = [
        entry["job"]
        for entry in captured
        if str(entry["item_id"]).startswith("capacity::screen::")
    ]
    assert {job.config.gpu_runtime_preload_splits for job in screen_jobs} == {("train",)}
    assert {job.config.gpu_runtime_preload_targets for job in screen_jobs} == {False}

    selected_configs = sorted(bundle["config_by_label"].values(), key=lambda cfg: cfg.label)[:1]

    def _fake_finalize_capacity_screen_output(**kwargs):
        output_root = Path(kwargs["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        screen_summary_json = output_root / "tree_fno_capacity_screen_summary.json"
        screen_summary_md = output_root / "tree_fno_capacity_screen_summary.md"
        screen_summary_json.write_text("{}", encoding="utf-8")
        screen_summary_md.write_text("# Screen\n", encoding="utf-8")
        top_rankings = [
            {"config_label": str(config.label)} for config in selected_configs
        ]
        return {
            "screen_rankings": list(top_rankings),
            "top_rankings": list(top_rankings),
            "locked_configs": list(selected_configs),
            "screen_summary_json": str(screen_summary_json),
            "screen_summary_md": str(screen_summary_md),
        }

    monkeypatch.setattr(mod, "finalize_capacity_screen_output", _fake_finalize_capacity_screen_output)
    screen_reduce = _find_scheduler_item(graph["items"], "capacity::screen::reduce")
    screen_reduce.callback()
    locked_jobs = [
        entry["job"]
        for entry in captured
        if str(entry["item_id"]).startswith("capacity::locked::")
    ]
    assert {job.config.gpu_runtime_preload_splits for job in locked_jobs} == {
        ("train", "val", "test")
    }
    assert {job.config.gpu_runtime_preload_targets for job in locked_jobs} == {True}


def test_capacity_screen_preflight_preserves_full_strong_layout_with_interleaving(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    mig_layout = _synthetic_mig_layout(gpu_count=4, migs_per_gpu=4)
    mig_tokens = [str(entry["mig_uuid"]) for entry in mig_layout]
    monkeypatch.setattr(mod, "_discover_mig_layout", lambda: list(mig_layout))
    monkeypatch.setattr(
        mod,
        "estimate_tree_worker_runtime_preflight",
        lambda **_kwargs: {
            "available": True,
            "resident_store_bytes_total": 1024,
            "split_estimates": [
                {"split_name": "train", "resident_store_bytes": 768},
                {"split_name": "val", "resident_store_bytes": 256},
            ],
        },
    )
    args = mod._parser().parse_args(
        [
            "capacity",
            "--output-root",
            str(tmp_path / "capacity"),
            "--mig-uuids",
            " ".join(mig_tokens),
            "--base-config-preset",
            "common_factorized_sketch_v1",
            "--capacity-widths",
            "128",
            "--capacity-modes",
            "2",
            "4",
            "--capacity-layers",
            "2",
            "4",
            "6",
            "--screen-seeds",
            "0",
            "1",
            "2",
            "--no-gpu-runtime-allow-multi-worker-screen",
            "--gpu-runtime-capacity-workers-per-mig",
            "1",
            "--no-use-cuda",
        ]
    )

    bundle = mod.build_capacity_screen_job_bundle(args)
    preflight = dict(bundle["screen_preflight"])
    first_wave = [
        (str(job.config.label), int(job.seeds[0]) if job.seeds else -1)
        for job in list(bundle["screen_jobs"])[:16]
    ]

    assert preflight["status"] == "ok"
    assert preflight["strong_guard_enabled"] is True
    assert preflight["auto_safe_applied"] is False
    assert preflight["requested_screen_max_concurrent_per_physical_gpu"] == 0
    assert preflight["effective_screen_max_concurrent_per_physical_gpu"] == 0
    assert preflight["requested_screen_device_order"] == "input"
    assert preflight["effective_screen_device_order"] == "interleave_by_physical_gpu"
    assert preflight["recommended_safe_rerun_flags"] == []
    assert bundle["screen_allowed_devices"][:8] == (
        "MIG-g0-0",
        "MIG-g1-0",
        "MIG-g2-0",
        "MIG-g3-0",
        "MIG-g0-1",
        "MIG-g1-1",
        "MIG-g2-1",
        "MIG-g3-1",
    )
    assert len(bundle["screen_allowed_devices"]) == 16
    assert preflight["first_wave_jobs_by_physical_gpu"][0]["gpu_index"] == 0
    assert len(preflight["first_wave_jobs_by_physical_gpu"][0]["jobs"]) == 4
    assert preflight["projected_first_wave_bytes_by_physical_gpu"][0]["active_screen_workers"] == 4
    assert (
        preflight["projected_first_wave_bytes_by_physical_gpu"][0][
            "projected_resident_store_bytes_total"
        ]
        == 4096
    )
    assert ("fair_fno_v1_w128_m4_l6", 2) in first_wave
    assert ("fair_fno_v1_w128_m2_l6", 2) in first_wave
    assert ("fair_fno_v1_w128_m4_l2", 2) not in first_wave
    assert ("fair_fno_v1_w128_m2_l2", 2) not in first_wave


def test_capacity_screen_preflight_leaves_legacy_capacity_layout_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    mig_layout = _synthetic_mig_layout(gpu_count=4, migs_per_gpu=4)
    mig_tokens = [str(entry["mig_uuid"]) for entry in mig_layout]
    monkeypatch.setattr(mod, "_discover_mig_layout", lambda: list(mig_layout))
    monkeypatch.setattr(
        mod,
        "estimate_tree_worker_runtime_preflight",
        lambda **_kwargs: {
            "available": True,
            "resident_store_bytes_total": 512,
        },
    )
    args = mod._parser().parse_args(
        [
            "capacity",
            "--output-root",
            str(tmp_path / "capacity"),
            "--mig-uuids",
            " ".join(mig_tokens),
            "--no-gpu-runtime-allow-multi-worker-screen",
            "--gpu-runtime-capacity-workers-per-mig",
            "1",
            "--no-use-cuda",
        ]
    )

    bundle = mod.build_capacity_screen_job_bundle(args)
    preflight = dict(bundle["screen_preflight"])

    assert preflight["status"] == "ok"
    assert preflight["strong_guard_enabled"] is False
    assert preflight["auto_safe_applied"] is False
    assert preflight["violations"] == []


def test_capacity_screen_preflight_flags_explicit_unsafe_strong_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    mig_layout = _synthetic_mig_layout(gpu_count=4, migs_per_gpu=4)
    mig_tokens = [str(entry["mig_uuid"]) for entry in mig_layout]
    monkeypatch.setattr(mod, "_discover_mig_layout", lambda: list(mig_layout))
    monkeypatch.setattr(
        mod,
        "estimate_tree_worker_runtime_preflight",
        lambda **_kwargs: {
            "available": True,
            "resident_store_bytes_total": 1024,
        },
    )
    args = mod._parser().parse_args(
        [
            "capacity",
            "--output-root",
            str(tmp_path / "capacity"),
            "--mig-uuids",
            " ".join(mig_tokens),
            "--base-config-preset",
            "common_factorized_sketch_v1",
            "--screen-max-concurrent-per-physical-gpu",
            "2",
            "--no-gpu-runtime-allow-multi-worker-screen",
            "--gpu-runtime-capacity-workers-per-mig",
            "1",
            "--no-use-cuda",
        ]
    )

    bundle = mod.build_capacity_screen_job_bundle(args)
    preflight = dict(bundle["screen_preflight"])

    assert preflight["status"] == "unsafe_capacity_screen_layout"
    assert preflight["strong_guard_enabled"] is True
    assert preflight["auto_safe_applied"] is False
    assert preflight["requested_screen_max_concurrent_per_physical_gpu"] == 2
    assert preflight["effective_screen_max_concurrent_per_physical_gpu"] == 2
    assert preflight["effective_screen_device_order"] == "interleave_by_physical_gpu"
    assert preflight["recommended_safe_rerun_flags"] == [
        "--screen-max-concurrent-per-physical-gpu 1",
        "--screen-device-order interleave_by_physical_gpu",
    ]
    assert preflight["active_screen_worker_slots_by_physical_gpu"] == [
        {
            "gpu_uuid": "GPU-0",
            "gpu_index": 0,
            "mig_uuids": ["MIG-g0-0", "MIG-g0-1"],
        },
        {
            "gpu_uuid": "GPU-1",
            "gpu_index": 1,
            "mig_uuids": ["MIG-g1-0", "MIG-g1-1"],
        },
        {
            "gpu_uuid": "GPU-2",
            "gpu_index": 2,
            "mig_uuids": ["MIG-g2-0", "MIG-g2-1"],
        },
        {
            "gpu_uuid": "GPU-3",
            "gpu_index": 3,
            "mig_uuids": ["MIG-g3-0", "MIG-g3-1"],
        },
    ]


def test_worker_payload_writes_snapshot_before_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()

    def _should_not_run(**_kwargs):
        raise AssertionError("worker execution should not start in snapshot-only mode")

    monkeypatch.setattr(
        mod,
        "run_markov_full_doc_anchor_diagnostics",
        _should_not_run,
    )
    snapshot_json = tmp_path / "manual_snapshot.json"
    args = mod._parser().parse_args(
        [
            "worker",
            "--job-name",
            "debug_worker",
            "--output-dir",
            str(tmp_path / "worker"),
            "--family",
            "tree_neural",
            "--train-doc-count",
            "10240",
            "--state-dim",
            "128",
            "--hidden-dim",
            "512",
            "--n-epochs",
            "52",
            "--batch-size",
            "64",
            "--lr",
            "0.0005",
            "--weight-decay",
            "0.0",
            "--summary-spec-name",
            "markov_count_sketch",
            "--slot-count",
            "4",
            "--leaf-supervision-kind",
            "count_only",
            "--leaf-label-rate",
            "0.0",
            "--tree-local-weighting-mode",
            "subset_mean",
            "--comparison-mode",
            "comparable",
            "--preserve-requested-leaf-tokens",
            "--base-bundle-path",
            str(tmp_path / "bundle.pkl"),
            "--gpu-runtime-preload-splits",
            "train",
            "--debug-snapshot-json",
            str(snapshot_json),
            "--debug-stop-after-snapshot",
        ]
    )

    payload = mod._worker_payload(args)

    assert payload["status"] == "snapshot_only"
    worker_snapshot = tmp_path / "worker" / "worker_invocation_snapshot.json"
    assert worker_snapshot.exists()
    assert snapshot_json.exists()
    snapshot = json.loads(worker_snapshot.read_text(encoding="utf-8"))
    assert snapshot["job_name"] == "debug_worker"
    assert snapshot["base_bundle_path"] == str(tmp_path / "bundle.pkl")
    assert snapshot["config_overrides"]["summary_spec_name"] == "markov_count_sketch"
    assert snapshot["config_overrides"]["gpu_runtime_preload_splits"] == ["train"]
    assert snapshot["config_overrides"]["tree_local_weighting_mode"] == "subset_mean"
    assert snapshot["config_overrides"]["comparison_mode"] == "comparable"
    assert snapshot["config_overrides"]["preserve_requested_leaf_tokens"] is True


def test_worker_payload_capacity_screen_snapshot_records_device_and_preflight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()

    def _should_not_run(**_kwargs):
        raise AssertionError("worker execution should not start in snapshot-only mode")

    monkeypatch.setattr(
        mod,
        "run_markov_full_doc_anchor_diagnostics",
        _should_not_run,
    )
    monkeypatch.setattr(
        mod,
        "_estimate_capacity_screen_worker_preflight",
        lambda **_kwargs: {
            "available": True,
            "resident_store_bytes_total": 2048,
            "split_estimates": [
                {"split_name": "train", "resident_store_bytes": 1536},
                {"split_name": "val", "resident_store_bytes": 512},
            ],
        },
    )
    monkeypatch.setattr(
        mod,
        "_discover_mig_layout",
        lambda: [
            {
                "gpu_index": 2,
                "gpu_uuid": "GPU-2",
                "mig_uuid": "MIG-test",
            }
        ],
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-test")
    args = mod._parser().parse_args(
        [
            "worker",
            "--job-name",
            "capacity_screen_worker",
            "--output-dir",
            str(tmp_path / "worker"),
            "--family",
            "tree_neural",
            "--train-doc-count",
            "10240",
            "--state-dim",
            "128",
            "--hidden-dim",
            "512",
            "--n-epochs",
            "52",
            "--batch-size",
            "64",
            "--lr",
            "0.0005",
            "--weight-decay",
            "0.0",
            "--summary-spec-name",
            "markov_count_sketch",
            "--slot-count",
            "4",
            "--tree-task-head-mode",
            "theorem_feature_scalar",
            "--tree-batch-pack-mode",
            "fixed_fused",
            "--fixed-leaf-tokens",
            "16",
            "--tuning-stage",
            "capacity_screen",
            "--use-cuda",
            "--debug-stop-after-snapshot",
        ]
    )

    payload = mod._worker_payload(args)

    assert payload["status"] == "snapshot_only"
    snapshot = json.loads(
        (tmp_path / "worker" / "worker_invocation_snapshot.json").read_text(encoding="utf-8")
    )
    assert snapshot["device_context"]["resolved_device"] == {
        "mig_uuid": "MIG-test",
        "gpu_index": 2,
        "gpu_uuid": "GPU-2",
    }
    assert snapshot["runtime_preflight"]["resident_store_bytes_total"] == 2048
    assert snapshot["runtime_preflight"]["split_estimates"] == [
        {"split_name": "train", "resident_store_bytes": 1536},
        {"split_name": "val", "resident_store_bytes": 512},
    ]


def test_replay_worker_snapshot_uses_saved_invocation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    captured: dict[str, object] = {}

    def _fake_run_markov_full_doc_anchor_diagnostics(**kwargs):
        captured.update(kwargs)
        return {"runs": [], "aggregate_rows": []}

    monkeypatch.setattr(
        mod,
        "run_markov_full_doc_anchor_diagnostics",
        _fake_run_markov_full_doc_anchor_diagnostics,
    )
    snapshot_json = tmp_path / "snapshot.json"
    snapshot_json.write_text(
        json.dumps(
            {
                "job_name": "debug_worker",
                "output_dir": str(tmp_path / "orig"),
                "benchmark_name": "recoverable_v4",
                "hardness_grid": "",
                "grid_cell_ids": [],
                "seeds": [0],
                "train_doc_counts": [10240],
                "baseline_families": ["tree_neural"],
                "emit_confusion": False,
                "use_cuda": True,
                "cuda_device": 0,
                "torch_threads": 1,
                "config_overrides": {
                    "state_dim": 128,
                    "hidden_dim": 512,
                    "n_epochs": 52,
                    "batch_size": 64,
                    "lr": 0.0005,
                    "weight_decay": 0.0,
                    "gpu_runtime_preload_splits": ["train"],
                },
                "run_metadata": {
                    "config_label": "cfg",
                    "base_bundle_path": "/tmp/bundle_from_metadata.pkl",
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    args = mod._parser().parse_args(
        [
            "replay_worker_snapshot",
            "--snapshot-json",
            str(snapshot_json),
            "--output-dir",
            str(tmp_path / "replay"),
            "--no-use-cuda",
            "--torch-threads",
            "7",
        ]
    )

    payload = mod._replay_worker_snapshot_payload(args)

    assert payload["job_name"] == "debug_worker"
    assert payload["replayed_from_snapshot_json"] == str(snapshot_json)
    assert Path(payload["replay_snapshot_json"]).exists()
    assert captured["output_dir"] == tmp_path / "replay"
    assert captured["use_cuda"] is False
    assert captured["cuda_device"] is None
    assert captured["torch_threads"] == 7
    assert captured["base_bundle_path"] == "/tmp/bundle_from_metadata.pkl"
    assert captured["config_overrides"]["gpu_runtime_preload_splits"] == ("train",)


def test_worker_payload_preserves_exact_sketch_summary_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        mod,
        "_execute_worker_invocation",
        lambda _snapshot: {
            "runs": [
                {
                    "tree_leaf_fno_width": 64,
                    "tree_leaf_fno_n_modes": 8,
                    "tree_leaf_fno_n_layers": 2,
                    "tree_root_supervision_kind": "mse",
                    "tree_checkpoint_metric": "val_exact_sketch_direct",
                    "tree_stage1_checkpoint_metric": "val_theorem_bootstrap_direct",
                    "tree_stage1_eval_mode": "per_epoch",
                    "tree_stage1_screen_doc_limit": 8,
                    "tree_stage1_final_exact_doc_limit": 8,
                    "tree_stage1_artifact_dir": str(tmp_path / "artifact"),
                    "tree_stage1_root_weight": 0.0,
                    "tree_training_schedule": "two_stage",
                    "tree_stage1_epochs": 1,
                    "tree_stage2_epochs": 1,
                    "tree_task_head_mode": "theorem_feature_scalar",
                    "tree_theorem_surface_mode": "opaque_carrier_exact_sketch",
                    "tree_summary_spec_root_mode": "factored_theorem_readout",
                    "tree_join_bit_weight": 1.0,
                    "summary_spec_name": "markov_count_sketch",
                    "slot_count": 4,
                    "internal_supervision_kind": "full_sketch",
                    "internal_label_rate": 1.0,
                    "leaf_supervision_kind": "full_sketch",
                    "leaf_label_rate": 1.0,
                    "train_root_mae": 1.0,
                    "val_root_mae": 0.9,
                    "test_root_mae": 0.8,
                    "train_exact_match_rate": 0.0,
                    "val_exact_match_rate": 0.0,
                    "test_exact_match_rate": 0.0,
                    "fit_diagnostics": {
                        "selection_metric_name": "val_exact_sketch_direct",
                        "selection_metric_value": 1.23,
                    },
                    "parameterization": "formal_local_law_weight",
                    "optimization_root_weight": 0.2,
                    "local_law_c1_weight": 0.2,
                    "local_law_c2_weight": 0.2,
                    "local_law_c3_weight": 0.2,
                    "c2_metric_kind": "count_drift",
                    "comparison_semantics": "current",
                    "exact_sketch_diagnostics": {
                        "direct_selection_metrics": {
                            "test": {
                                "exact_projected_root_mae": 0.4,
                                "certified_projected_root_mae": 0.4,
                                "root_mae_predicted_counts_predicted_endpoints": 0.4,
                                "root_mae_oracle_counts_predicted_endpoints": 0.2,
                                "root_mae_predicted_counts_oracle_endpoints": 0.3,
                                "learned_merger_gap": 0.0,
                                "leaf_first_accuracy": 0.9,
                                "leaf_last_accuracy": 0.95,
                                "merge_first_accuracy": 0.85,
                                "merge_last_accuracy": 0.8,
                                "leaf_count_off_by_k_histogram": {"0": 0.5, "1": 0.5},
                                "merge_exact_summary_match_rate_by_depth": {"0": 0.4},
                            }
                        }
                    },
                    "exact_sketch_markov_sufficiency_gap_score": 0.05,
                    "exact_projected_root_mae": 0.4,
                    "test_exact_projected_root_mae": 0.4,
                    "certified_projected_root_mae": 0.4,
                    "test_certified_projected_root_mae": 0.4,
                    "root_mae_predicted_counts_predicted_endpoints": 0.4,
                    "test_root_mae_predicted_counts_predicted_endpoints": 0.4,
                    "root_mae_oracle_counts_predicted_endpoints": 0.2,
                    "test_root_mae_oracle_counts_predicted_endpoints": 0.2,
                    "root_mae_predicted_counts_oracle_endpoints": 0.3,
                    "test_root_mae_predicted_counts_oracle_endpoints": 0.3,
                    "learned_merger_gap": 0.0,
                    "test_learned_merger_gap": 0.0,
                    "leaf_first_accuracy": 0.9,
                    "test_leaf_first_accuracy": 0.9,
                    "leaf_last_accuracy": 0.95,
                    "test_leaf_last_accuracy": 0.95,
                    "merge_first_accuracy": 0.85,
                    "test_merge_first_accuracy": 0.85,
                    "merge_last_accuracy": 0.8,
                    "test_merge_last_accuracy": 0.8,
                    "leaf_count_off_by_k_histogram": {"0": 0.5, "1": 0.5},
                    "merge_exact_summary_match_rate_by_depth": {"0": 0.4},
                }
            ],
            "aggregate_rows": [{"config_label": "opaque"}],
        },
    )
    args = mod._parser().parse_args(
        [
            "worker",
            "--job-name",
            "exact_summary_worker",
            "--output-dir",
            str(tmp_path / "worker"),
            "--family",
            "tree_neural",
            "--benchmark",
            "recoverable_v4",
            "--train-doc-count",
            "32",
            "--state-dim",
            "128",
            "--hidden-dim",
            "512",
            "--n-epochs",
            "1",
            "--batch-size",
            "4",
            "--lr",
            "0.0005",
            "--weight-decay",
            "0.0",
            "--summary-spec-name",
            "markov_count_sketch",
            "--slot-count",
            "4",
            "--tree-task-head-mode",
            "theorem_feature_scalar",
            "--tree-theorem-surface-mode",
            "opaque_carrier_exact_sketch",
            "--tree-score-merge-mode",
            "exact_projected_sketch",
            "--tree-merge-hidden-dim",
            "256",
            "--tree-stage1-screen-doc-limit",
            "8",
            "--tree-stage1-final-exact-doc-limit",
            "8",
            "--exact-metric-selection-doc-limit",
            "8",
            "--tree-exact-eval-max-docs",
            "8",
            "--seeds",
            "0",
        ]
    )

    payload = mod._worker_payload(args)

    assert abs(float(payload["exact_sketch_markov_sufficiency_gap_score"]) - 0.05) < 1e-9
    assert abs(float(payload["test_exact_projected_root_mae"]) - 0.4) < 1e-9
    assert abs(float(payload["test_certified_projected_root_mae"]) - 0.4) < 1e-9
    assert abs(float(payload["test_root_mae_oracle_counts_predicted_endpoints"]) - 0.2) < 1e-9
    assert abs(float(payload["test_root_mae_predicted_counts_oracle_endpoints"]) - 0.3) < 1e-9
    assert abs(float(payload["test_learned_merger_gap"]) - 0.0) < 1e-9
    assert payload["exact_sketch_diagnostics"]["direct_selection_metrics"]["test"][
        "exact_projected_root_mae"
    ] == 0.4


def test_replay_worker_snapshot_writes_memory_probe_jsonl(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()

    def _fake_run_markov_full_doc_anchor_diagnostics(**kwargs):
        probe = kwargs.get("memory_probe")
        assert probe is not None
        probe("stage1_boundary", {"epoch": 12, "phase": "final_exact_metrics"})
        return {"runs": [], "aggregate_rows": []}

    monkeypatch.setattr(
        mod,
        "run_markov_full_doc_anchor_diagnostics",
        _fake_run_markov_full_doc_anchor_diagnostics,
    )
    snapshot_json = tmp_path / "snapshot.json"
    snapshot_json.write_text(
        json.dumps(
            {
                "job_name": "debug_worker",
                "output_dir": str(tmp_path / "orig"),
                "benchmark_name": "recoverable_v4",
                "hardness_grid": "",
                "grid_cell_ids": [],
                "seeds": [0],
                "train_doc_counts": [10240],
                "baseline_families": ["tree_neural"],
                "emit_confusion": False,
                "use_cuda": False,
                "cuda_device": None,
                "torch_threads": 1,
                "config_overrides": {
                    "state_dim": 128,
                    "hidden_dim": 512,
                    "n_epochs": 52,
                    "batch_size": 64,
                    "lr": 0.0005,
                    "weight_decay": 0.0,
                },
                "run_metadata": {"config_label": "cfg"},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    memory_probe_jsonl = tmp_path / "probe.jsonl"
    args = mod._parser().parse_args(
        [
            "replay_worker_snapshot",
            "--snapshot-json",
            str(snapshot_json),
            "--output-dir",
            str(tmp_path / "replay"),
            "--memory-probe-jsonl",
            str(memory_probe_jsonl),
        ]
    )

    payload = mod._replay_worker_snapshot_payload(args)

    assert payload["job_name"] == "debug_worker"
    assert memory_probe_jsonl.exists()
    rows = [
        json.loads(line)
        for line in memory_probe_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["event"] == "stage1_boundary"
    assert rows[0]["payload"] == {"epoch": 12, "phase": "final_exact_metrics"}
    assert "rss_kib" in rows[0]
    assert "pss_kib" in rows[0]
    assert "private_dirty_kib" in rows[0]


def test_write_memory_probe_summary_reports_peak_and_exact_eval_boundary(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    job_a = tmp_path / "jobs" / "job_a"
    job_b = tmp_path / "jobs" / "job_b"
    job_a.mkdir(parents=True)
    job_b.mkdir(parents=True)
    (job_a / "memory_probe.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "post_eval_fno_model",
                        "private_dirty_kib": 1024,
                        "rss_kib": 2048,
                        "swap_kib": 0,
                    }
                ),
                json.dumps(
                    {
                        "event": "pre_exact_eval_batch",
                        "private_dirty_kib": 4096,
                        "rss_kib": 8192,
                        "swap_kib": 32,
                    }
                ),
                json.dumps(
                    {
                        "event": "post_exact_eval_batch_trim",
                        "private_dirty_kib": 3072,
                        "rss_kib": 7168,
                        "swap_kib": 16,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (job_b / "memory_probe.jsonl").write_text(
        json.dumps(
            {
                "event": "post_eval_fno_model",
                "private_dirty_kib": 512,
                "rss_kib": 1024,
                "swap_kib": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = mod._write_memory_probe_summary(tmp_path)

    assert payload["probe_files_found"] == 2
    assert payload["jobs_reaching_pre_exact_eval_batch"] == 1
    assert payload["jobs_reaching_post_exact_eval_batch_trim"] == 1
    assert payload["peak_private_dirty_jobs"][0]["job_dir_name"] == "job_a"
    assert payload["peak_private_dirty_jobs"][0]["max_private_dirty_kib"] == 4096
    assert payload["largest_private_dirty_delta_jobs"][0]["job_dir_name"] == "job_a"
    assert (
        payload["largest_private_dirty_delta_jobs"][0][
            "largest_private_dirty_delta_to_event"
        ]
        == "pre_exact_eval_batch"
    )
    summary_json = Path(payload["summary_json"])
    assert summary_json.exists()
    written = json.loads(summary_json.read_text(encoding="utf-8"))
    assert written["jobs_with_rows"] == 2


def test_prepare_data_mode_prints_prepare_summary(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        mod,
        "prepare_markov_full_doc_anchor_diagnostics_data",
        lambda **kwargs: {
            "simulation": "markov_full_doc_anchor_diagnostics_prepare_data",
            "prepared": [
                {
                    "prepared_data_root": str(tmp_path / "prepared"),
                    "prepared_data_signature": "abc123",
                }
            ],
        },
    )
    args = mod._parser().parse_args(
        [
            "prepare_data",
            "--benchmark",
            "smoke",
            "--prepared-data-root",
            str(tmp_path / "prepared"),
        ]
    )

    assert mod._launch_prepare_data(args) == 0


def test_launch_capacity_reports_unsafe_layout_cleanly(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    mod = _load_module()
    args = mod._parser().parse_args(
        [
            "capacity",
            "--output-root",
            str(tmp_path / "capacity"),
            "--no-use-cuda",
        ]
    )
    monkeypatch.setattr(
        mod,
        "_cached_capacity_screen_job_bundle",
        lambda _args: {
            "screen_preflight": {
                "status": "unsafe_capacity_screen_layout",
                "recommended_safe_rerun_flags": [
                    "--screen-max-concurrent-per-physical-gpu 1",
                    "--screen-device-order interleave_by_physical_gpu",
                ],
            }
        },
    )

    def _should_not_run(_args):
        raise AssertionError("unsafe preflight should abort before scheduler launch")

    monkeypatch.setattr(mod, "_run_scheduler_mode", _should_not_run)

    rc = int(mod._launch_capacity(args))
    payload = json.loads(capsys.readouterr().out)

    assert rc == 1
    assert payload["status"] == "unsafe_capacity_screen_layout"
    assert payload["failed_jobs"] == 0
    assert payload["recommended_safe_rerun_flags"] == [
        "--screen-max-concurrent-per-physical-gpu 1",
        "--screen-device-order interleave_by_physical_gpu",
    ]
    assert payload["tree_fno_capacity_locked_summary_json"].endswith(
        "tree_fno_capacity_locked_summary.json"
    )


def test_launch_capacity_reports_missing_locked_summary_cleanly(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    mod = _load_module()
    args = mod._parser().parse_args(
        [
            "capacity",
            "--output-root",
            str(tmp_path / "capacity"),
            "--no-use-cuda",
        ]
    )

    monkeypatch.setattr(
        mod,
        "_run_scheduler_mode",
        lambda parsed_args: {
            "plan_only": False,
            "failed_jobs": [{"item_id": "capacity::screen::example"}],
        },
    )

    rc = int(mod._launch_capacity(args))
    payload = json.loads(capsys.readouterr().out)

    assert rc == 1
    assert payload["status"] == "missing_locked_summary"
    assert payload["failed_jobs"] == 1
    assert payload["tree_fno_capacity_locked_summary_json"].endswith(
        "tree_fno_capacity_locked_summary.json"
    )


def test_run_scheduler_bundle_records_scheduler_debug_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()

    def _fake_run_scheduler(items, *, config):
        Path(str(config.status_path)).write_text(
            json.dumps({"state": "running"}),
            encoding="utf-8",
        )
        Path(str(config.event_log_path)).write_text(
            json.dumps({"event": "launch"}) + "\n",
            encoding="utf-8",
        )
        Path(str(config.failure_snapshot_path)).write_text(
            json.dumps({"reason": "first_failure_detected"}),
            encoding="utf-8",
        )
        return {
            "completed_items": {},
            "failed_items": {},
            "timeline": [],
            "failure_cleanup_events": [],
            "live_status_path": str(config.status_path),
            "event_log_path": str(config.event_log_path),
            "failure_snapshot_path": str(config.failure_snapshot_path),
        }

    monkeypatch.setattr(mod, "run_scheduler", _fake_run_scheduler)

    output_root = tmp_path / "scheduler_bundle"
    mod._run_scheduler_bundle(
        output_root=output_root,
        items=(),
        devices=("MIG-0",),
        max_gpu_items_per_mig=1,
        launch_stagger_seconds=0.0,
        cleanup_stale_children=False,
        resume_enabled=False,
        manifest_payload={"mode": "test"},
    )

    controller_results = json.loads(
        (output_root / "controller_results.json").read_text(encoding="utf-8")
    )
    assert (output_root / "scheduler_status.json").exists()
    assert (output_root / "scheduler_events.jsonl").exists()
    assert (output_root / "scheduler_failure_snapshot.json").exists()
    assert controller_results["scheduler_status_json"].endswith("scheduler_status.json")
    assert controller_results["scheduler_events_jsonl"].endswith("scheduler_events.jsonl")
    assert controller_results["scheduler_failure_snapshot_json"].endswith(
        "scheduler_failure_snapshot.json"
    )


def test_scheduler_item_for_job_populates_progress_bucket_metadata(tmp_path: Path) -> None:
    mod = _load_module()
    config = mod._RunConfigSpec(
        label="cfg",
        state_dim=128,
        hidden_dim=512,
        n_epochs=10,
        batch_size=512,
        lr=5e-4,
        weight_decay=0.0,
        fixed_leaf_tokens=16,
        leaf_supervision_kind="count_only",
        internal_supervision_kind="none",
        leaf_label_rate=0.0,
        internal_label_rate=0.0,
    )
    job = mod._JobSpec(
        family="tree_neural",
        train_doc_count=1024,
        benchmark="recoverable_v4",
        hardness_grid="",
        grid_cell_ids=(),
        seeds=(0,),
        config=config,
    )

    item = mod._scheduler_item_for_job(
        phase="parity",
        item_id="parity::job",
        output_root=tmp_path,
        job=job,
        torch_threads=1,
        use_cuda=False,
    )

    assert item.metadata["task_name"] == job.job_name
    assert item.metadata["scope"] == "recoverable_v4"
    assert item.metadata["train_docs"] == 1024
    assert item.metadata["model_family"] == "tree_neural"
    assert item.metadata["package"] == "full100"
    assert item.metadata["worker_kind"] == "full_doc_diagnostics"
    assert item.metadata["n_epochs"] == 10


def test_run_config_from_mapping_preserves_requested_leaf_tokens_by_default() -> None:
    mod = _load_module()

    config = mod._run_config_from_mapping(
        {
            "label": "cfg",
            "state_dim": 128,
            "hidden_dim": 512,
            "n_epochs": 10,
            "batch_size": 64,
            "lr": 5e-4,
            "weight_decay": 0.0,
            "fixed_leaf_tokens": 32,
        }
    )

    assert config.fixed_leaf_tokens == 32
    assert config.preserve_requested_leaf_tokens is True
    assert config.official_fno_preserve_requested_leaf_tokens is True


def test_run_config_from_mapping_preserves_topology() -> None:
    mod = _load_module()

    config = mod._run_config_from_mapping(
        {
            "label": "cfg_topology",
            "state_dim": 128,
            "hidden_dim": 512,
            "n_epochs": 10,
            "batch_size": 64,
            "lr": 5e-4,
            "weight_decay": 0.0,
            "topology": "full_doc",
            "fixed_leaf_tokens": 128,
        }
    )

    assert config.topology == "full_doc"


def test_run_config_from_mapping_preserves_run_intent_fields() -> None:
    mod = _load_module()

    config = mod._run_config_from_mapping(
        {
            "label": "cfg_intent",
            "state_dim": 128,
            "hidden_dim": 512,
            "n_epochs": 10,
            "batch_size": 64,
            "lr": 5e-4,
            "weight_decay": 0.0,
            "budget_total_calls": 1024,
            "budget_total_calls_per_doc": 0.2,
            "mass_target_per_doc": 0.1,
            "full_doc_budget_share": 1.0,
            "doc_consumption_mode": "root_only",
            "local_split_mode": "balanced",
            "local_allocation_policy": "breadth_first",
            "package_semantics": "mass_matched",
            "depth_discount_gamma": 0.9,
            "tree_c2_mode": "fiber",
            "tree_document_loss_normalization_mode": "supervised_docs",
            "tree_supervision_source": "manifest",
        }
    )

    assert config.budget_total_calls == 1024
    assert config.budget_total_calls_per_doc == pytest.approx(0.2)
    assert config.mass_target_per_doc == pytest.approx(0.1)
    assert config.full_doc_budget_share == pytest.approx(1.0)
    assert config.doc_consumption_mode == "root_only"
    assert config.local_split_mode == "balanced"
    assert config.local_allocation_policy == "breadth_first"
    assert config.package_semantics == "mass_matched"
    assert config.depth_discount_gamma == pytest.approx(0.9)
    assert config.tree_c2_mode == "fiber"
    assert config.tree_document_loss_normalization_mode == "supervised_docs"
    assert config.tree_supervision_source == "manifest"


def test_run_config_from_mapping_accepts_preset_style_objective_keys() -> None:
    mod = _load_module()

    config = mod._run_config_from_mapping(
        {
            "label": "cfg_preset",
            "state_dim": 128,
            "hidden_dim": 512,
            "n_epochs": 10,
            "batch_size": 64,
            "lr": 5e-4,
            "weight_decay": 0.0,
            "local_law_weight": 0.8,
            "c1_relative_weight": 2.0,
            "c2_relative_weight": 1.0,
            "c3_relative_weight": 0.5,
        }
    )

    assert config.tree_local_law_weight == pytest.approx(0.8)
    assert config.tree_task_objective_weight is None
    assert config.tree_c1_relative_weight == pytest.approx(2.0)
    assert config.tree_c2_relative_weight == pytest.approx(1.0)
    assert config.tree_c3_relative_weight == pytest.approx(0.5)


def test_run_config_from_mapping_rejects_lambda_and_explicit_root_hybrid() -> None:
    mod = _load_module()

    with pytest.raises(ValueError, match="mutually exclusive"):
        mod._run_config_from_mapping(
            {
                "label": "cfg_preset",
                "state_dim": 128,
                "hidden_dim": 512,
                "n_epochs": 10,
                "batch_size": 64,
                "lr": 5e-4,
                "weight_decay": 0.0,
                "local_law_weight": 0.8,
                "task_objective_weight": 1.0,
            }
        )


def test_worker_payload_uses_authoritative_config_spec_when_cli_flags_are_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()

    def _should_not_run(**_kwargs):
        raise AssertionError("worker execution should not start in snapshot-only mode")

    monkeypatch.setattr(
        mod,
        "run_markov_full_doc_anchor_diagnostics",
        _should_not_run,
    )
    output_dir = tmp_path / "worker"
    job = mod._JobSpec(
        family="tree_neural",
        train_doc_count=4096,
        benchmark="recoverable_v4",
        hardness_grid="",
        grid_cell_ids=(),
        seeds=(0,),
        config=mod._RunConfigSpec(
            label="cfg_authoritative",
            state_dim=128,
            hidden_dim=512,
            n_epochs=10,
            batch_size=32,
            lr=5e-4,
            weight_decay=0.0,
            fixed_leaf_tokens=16,
            tree_local_law_weight=0.8,
            tree_c1_relative_weight=2.0,
            tree_c2_relative_weight=1.0,
            tree_c3_relative_weight=0.5,
            tree_document_loss_normalization_mode="supervised_docs",
            tree_supervision_source="manifest",
            summary_spec_name="markov_count_sketch",
            slot_count=4,
            leaf_supervision_kind="count_only",
            leaf_label_rate=0.0,
            internal_supervision_kind="none",
            internal_label_rate=0.0,
        ),
        tuning_stage="matched_root",
    )
    cmd = mod._worker_command_for_job(
        job,
        output_dir=output_dir,
        torch_threads=1,
        use_cuda=False,
    )
    assert "--local-law-weight" in cmd
    assert "--root-share" not in cmd
    assert "--tree-document-loss-normalization-mode" in cmd
    assert "--tree-supervision-source" in cmd

    def _strip_flag(arguments: list[str], flag: str) -> list[str]:
        trimmed = list(arguments)
        if flag in trimmed:
            idx = trimmed.index(flag)
            del trimmed[idx : idx + 2]
        return trimmed

    parsed_cmd = _strip_flag(cmd[2:], "--local-law-weight")
    parsed_cmd = _strip_flag(parsed_cmd, "--root-share")
    parsed_cmd = _strip_flag(parsed_cmd, "--tree-document-loss-normalization-mode")
    parsed_cmd = _strip_flag(parsed_cmd, "--tree-supervision-source")
    args = mod._parser().parse_args([*parsed_cmd, "--debug-stop-after-snapshot"])

    payload = mod._worker_payload(args)

    assert payload["status"] == "snapshot_only"
    snapshot = json.loads(
        (output_dir / "worker_invocation_snapshot.json").read_text(encoding="utf-8")
    )
    assert snapshot["requested_run_config"]["tree_local_law_weight"] == pytest.approx(
        0.8
    )
    assert snapshot["requested_run_config"][
        "tree_document_loss_normalization_mode"
    ] == "supervised_docs"
    assert snapshot["requested_run_config"]["tree_supervision_source"] == "manifest"
    assert snapshot["config_overrides"]["local_law_weight"] == pytest.approx(0.8)
    assert "task_objective_weight" not in snapshot["config_overrides"]
    assert snapshot["config_overrides"]["c1_relative_weight"] == pytest.approx(2.0)
    assert snapshot["config_overrides"]["c3_relative_weight"] == pytest.approx(0.5)
    assert (
        snapshot["config_overrides"]["tree_document_loss_normalization_mode"]
        == "supervised_docs"
    )
    assert snapshot["config_overrides"]["tree_supervision_source"] == "manifest"


def test_launch_study_leaf_geometry_builds_expected_fixed_leaf_token_grid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    tuning_root = tmp_path / "tuning"
    tuning_root.mkdir(parents=True, exist_ok=True)
    (tuning_root / "tuning_summary.json").write_text(
        """
{
  "dev_selection_metric": "val_root_mae_mean",
  "test_metrics_hidden_during_selection": true,
  "winning_config_label": "cfg_locked",
  "winning_config_spec": {
    "label": "cfg_locked",
    "state_dim": 128,
    "hidden_dim": 512,
    "n_epochs": 64,
    "batch_size": 64,
    "lr": 0.0002,
    "weight_decay": 0.0,
    "tree_local_law_weight": 0.3,
    "tree_task_objective_weight": null
  }
}
""".strip(),
        encoding="utf-8",
    )
    captured = _capture_scheduler_jobs(monkeypatch, mod)
    args = mod._parser().parse_args(
        [
            "study",
            "--output-root",
            str(tmp_path / "leaf_geometry"),
            "--tuning-root",
            str(tuning_root),
            "--study-name",
            "leaf_geometry",
            "--mig-uuids",
            "MIG-a",
            "--no-use-cuda",
        ]
    )
    mod._build_scheduler_graph(
        args,
        output_root=tmp_path / "leaf_geometry",
        mig_uuids=["MIG-a"],
    )
    jobs = [
        entry["job"]
        for entry in captured
        if str(entry["item_id"]).startswith("study::leaf_geometry::")
    ]
    assert {job.config.fixed_leaf_tokens for job in jobs} == {8, 16, 32}
    assert {job.study_name for job in jobs} == {"leaf_geometry"}
    assert {job.study_axis for job in jobs} == {"fixed_leaf_tokens"}
    assert {job.locked_tree_neural_config_label for job in jobs} == {"cfg_locked"}
    assert {job.axis_value for job in jobs} == {"8", "16", "32"}


def test_select_representative_structural_cells_returns_easiest_median_hardest() -> None:
    mod = _load_module()
    selected = mod._select_representative_structural_cells(
        {
            "aggregate_rows": [
                {
                    "baseline_family": "tree_neural",
                    "tuning_stage": "study_screen",
                    "train_doc_count": 10240,
                    "cell_id": "cell_a",
                    "test_root_mae_mean": 0.30,
                    "n_regimes": 4,
                    "segment_density_band": "low",
                },
                {
                    "baseline_family": "tree_neural",
                    "tuning_stage": "study_screen",
                    "train_doc_count": 10240,
                    "cell_id": "cell_b",
                    "test_root_mae_mean": 0.10,
                    "n_regimes": 4,
                    "segment_density_band": "mid",
                },
                {
                    "baseline_family": "tree_neural",
                    "tuning_stage": "study_screen",
                    "train_doc_count": 10240,
                    "cell_id": "cell_c",
                    "test_root_mae_mean": 0.20,
                    "n_regimes": 8,
                    "segment_density_band": "high",
                },
            ]
        },
        family="tree_neural",
        tuning_stage="study_screen",
        train_doc_count=10240,
    )
    assert selected["easiest"]["cell_id"] == "cell_b"
    assert selected["median"]["cell_id"] == "cell_c"
    assert selected["hardest"]["cell_id"] == "cell_a"


def test_launch_study_structural_complexity_builds_screen_and_representative_jobs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    tuning_root = tmp_path / "tuning"
    tuning_root.mkdir(parents=True, exist_ok=True)
    (tuning_root / "tuning_summary.json").write_text(
        """
{
  "dev_selection_metric": "val_root_mae_mean",
  "test_metrics_hidden_during_selection": true,
  "winning_config_label": "cfg_locked",
  "winning_config_spec": {
    "label": "cfg_locked",
    "state_dim": 128,
    "hidden_dim": 512,
    "n_epochs": 64,
    "batch_size": 64,
    "lr": 0.0002,
    "weight_decay": 0.0,
    "tree_local_law_weight": 0.3,
    "tree_task_objective_weight": null
  }
}
""".strip(),
        encoding="utf-8",
    )
    cell_ids = [str(cell.cell_id) for cell in mod.resolve_full_doc_diagnostic_grid("structural_core_v1")]
    captured = _capture_scheduler_jobs(monkeypatch, mod)
    args = mod._parser().parse_args(
        [
            "study",
            "--output-root",
            str(tmp_path / "structural"),
            "--tuning-root",
            str(tuning_root),
            "--study-name",
            "structural_complexity",
            "--mig-uuids",
            "MIG-a",
            "--no-use-cuda",
        ]
    )
    graph = mod._build_scheduler_graph(
        args,
        output_root=tmp_path / "structural",
        mig_uuids=["MIG-a"],
    )
    screen_jobs = [
        entry["job"]
        for entry in captured
        if str(entry["item_id"]).startswith("study::screen::")
    ]
    assert {job.study_name for job in screen_jobs} == {"structural_complexity"}
    assert {cell for job in screen_jobs for cell in job.grid_cell_ids} == set(cell_ids)
    screen_root = tmp_path / "structural" / "screen"
    screen_root.mkdir(parents=True, exist_ok=True)
    (screen_root / "summary.json").write_text(
        json.dumps({"aggregate_rows": []}),
        encoding="utf-8",
    )
    representative_selection = {
        "easiest": {"cell_id": cell_ids[0]},
        "median": {"cell_id": cell_ids[1]},
        "hardest": {"cell_id": cell_ids[2]},
    }
    monkeypatch.setattr(
        mod,
        "_select_representative_structural_cells",
        lambda *_args, **_kwargs: representative_selection,
    )
    screen_reduce = _find_scheduler_item(graph["items"], "study::screen::reduce")
    callback_result = screen_reduce.callback()
    representative_jobs = [
        entry["job"]
        for entry in captured
        if str(entry["item_id"]).startswith("study::representative::")
    ]
    representative_cells = {cell for job in representative_jobs for cell in job.grid_cell_ids}
    assert len(representative_cells) == 3
    assert representative_cells == {cell_ids[0], cell_ids[1], cell_ids[2]}
    assert any(
        str(item.item_id) == "study::structural_complexity::reduce"
        for item in callback_result["new_items"]
    )


def test_main_dispatches_study_mode(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "run_tree_neural_full_doc_mig.py",
            "study",
            "--tuning-root",
            "/tmp/tuning",
            "--study-name",
            "leaf_geometry",
        ],
    )
    monkeypatch.setattr(mod, "_launch_study", lambda args: 23)
    monkeypatch.setattr(mod, "_launch_tune", lambda args: 17)
    monkeypatch.setattr(mod, "_launch_parity", lambda args: 13)
    monkeypatch.setattr(mod, "_launch_controller", lambda args: 99)

    rc = int(mod.main())

    assert rc == 23


def test_main_dispatches_parity_mode(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod.sys, "argv", ["run_tree_neural_full_doc_mig.py", "parity"])
    monkeypatch.setattr(mod, "_launch_parity", lambda args: 13)
    monkeypatch.setattr(mod, "_launch_tune", lambda args: 17)
    monkeypatch.setattr(mod, "_launch_study", lambda args: 23)
    monkeypatch.setattr(mod, "_launch_controller", lambda args: 99)

    rc = int(mod.main())

    assert rc == 13


def test_main_dispatches_capacity_mode(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod.sys, "argv", ["run_tree_neural_full_doc_mig.py", "capacity"])
    monkeypatch.setattr(mod, "_launch_capacity", lambda args: 29)
    monkeypatch.setattr(mod, "_launch_parity", lambda args: 13)
    monkeypatch.setattr(mod, "_launch_tune", lambda args: 17)
    monkeypatch.setattr(mod, "_launch_study", lambda args: 23)
    monkeypatch.setattr(mod, "_launch_controller", lambda args: 99)

    rc = int(mod.main())

    assert rc == 29


def test_launch_budget_frontier_builds_expected_job_grid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    captured = _capture_scheduler_jobs(monkeypatch, mod)
    args = mod._parser().parse_args(
        [
            "budget_frontier",
            "--output-root",
            str(tmp_path / "budget_frontier"),
            "--hardness-grid",
            "structural_core_v1",
            "--grid-cell-ids",
            "r4_seg4to6",
            "--tree-families",
            "tree_neural",
            "--reference-families",
            "official_fno",
            "--budget-calls-per-doc",
            "0.5",
            "1.0",
            "--full-doc-budget-shares",
            "0.0",
            "0.5",
            "1.0",
            "--seeds",
            "0",
            "--mig-uuids",
            "MIG-a",
            "--no-use-cuda",
        ]
    )
    graph = mod._build_scheduler_graph(
        args,
        output_root=tmp_path / "budget_frontier",
        mig_uuids=["MIG-a"],
    )
    jobs = [
        entry["job"]
        for entry in captured
        if str(entry["item_id"]).startswith("budget_frontier::")
    ]
    assert len(jobs) == 24
    assert {job.study_name for job in jobs} == {mod.ORACLE_BUDGET_STUDY_NAME}
    assert {job.study_axis for job in jobs} == {
        "budget_total_calls_per_doc__full_doc_budget_share"
    }
    assert {job.local_allocation_policy for job in jobs} == {"breadth_first"}
    ref_jobs = [job for job in jobs if job.family == "official_fno"]
    assert ref_jobs
    assert {job.full_doc_budget_share for job in ref_jobs} == {1.0}
    assert {job.config.label for job in ref_jobs} == {"budget_reference_default"}
    assert {job.hardness_grid for job in jobs} == {"structural_core_v1"}
    assert {cell for job in jobs for cell in job.grid_cell_ids} == {"r4_seg4to6"}
    tree_zero_share = [
        job
        for job in jobs
        if job.family == "tree_neural" and abs(job.full_doc_budget_share - 0.0) <= 1e-12
    ]
    assert {job.config.label for job in tree_zero_share} == {
        mod.FAIR_FNO_PARITY_CONFIG_LABEL
    }
    assert {job.config.tree_root_supervision_kind for job in tree_zero_share} == {
        "count_ce"
    }
    assert {job.doc_consumption_mode for job in tree_zero_share} == {"root_only"}
    assert {job.local_split_mode for job in tree_zero_share} == {
        "balanced",
        "leaf_heavy",
        "internal_heavy",
    }
    tree_full_share = [
        job
        for job in jobs
        if job.family == "tree_neural" and abs(job.full_doc_budget_share - 1.0) <= 1e-12
    ]
    assert {job.config.label for job in tree_full_share} == {
        mod.FAIR_FNO_PARITY_CONFIG_LABEL
    }
    assert {job.config.tree_root_supervision_kind for job in tree_full_share} == {
        "count_ce"
    }
    assert {job.doc_consumption_mode for job in tree_full_share} == {
        "root_only",
        "doc_sequence",
    }
    assert {job.local_split_mode for job in tree_full_share} == {"balanced"}
    assert {int(entry["gpu_slots"]) for entry in captured} == {1}
    reduce_item = _find_scheduler_item(graph["items"], "budget_frontier::reduce")
    assert set(reduce_item.deps) == {
        str(entry["item_id"]) for entry in captured
    }


def test_main_dispatches_budget_frontier_mode(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        mod.sys,
        "argv",
        ["run_tree_neural_full_doc_mig.py", "budget_frontier"],
    )
    monkeypatch.setattr(mod, "_launch_budget_frontier", lambda args: 31)
    monkeypatch.setattr(mod, "_launch_capacity", lambda args: 29)
    monkeypatch.setattr(mod, "_launch_parity", lambda args: 13)
    monkeypatch.setattr(mod, "_launch_tune", lambda args: 17)
    monkeypatch.setattr(mod, "_launch_study", lambda args: 23)
    monkeypatch.setattr(mod, "_launch_controller", lambda args: 99)

    rc = int(mod.main())

    assert rc == 31


def test_main_capacity_accepts_tree_stage1_screen_flags(monkeypatch) -> None:
    mod = _load_module()
    captured: dict[str, object] = {}

    def _launch_capacity(args):
        captured["tree_stage1_eval_mode"] = args.tree_stage1_eval_mode
        captured["tree_stage1_screen_doc_limit"] = int(args.tree_stage1_screen_doc_limit)
        captured["tree_stage1_final_exact_doc_limit"] = int(
            args.tree_stage1_final_exact_doc_limit
        )
        captured["exact_metric_selection_doc_limit"] = int(
            args.exact_metric_selection_doc_limit
        )
        captured["exact_metric_selection_interval"] = int(
            args.exact_metric_selection_interval
        )
        return 41

    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "run_tree_neural_full_doc_mig.py",
            "capacity",
            "--tree-stage1-eval-mode",
            "end_only",
            "--tree-stage1-screen-doc-limit",
            "128",
            "--tree-stage1-final-exact-doc-limit",
            "128",
            "--exact-metric-selection-doc-limit",
            "256",
            "--exact-metric-selection-interval",
            "5",
        ],
    )
    monkeypatch.setattr(mod, "_launch_capacity", _launch_capacity)

    rc = int(mod.main())

    assert rc == 41
    assert captured == {
        "tree_stage1_eval_mode": "end_only",
        "tree_stage1_screen_doc_limit": 128,
        "tree_stage1_final_exact_doc_limit": 128,
        "exact_metric_selection_doc_limit": 256,
        "exact_metric_selection_interval": 5,
    }


def test_load_or_write_summary_outputs_rebuilds_missing_summary(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "phase"
    output_root.mkdir(parents=True, exist_ok=True)
    fake_payload = {"aggregate_rows": [{"config_label": "cfg"}], "runs": []}

    monkeypatch.setattr(
        mod,
        "load_markov_full_doc_anchor_diagnostics_from_output_dir",
        lambda path: dict(fake_payload),
    )
    monkeypatch.setattr(
        mod,
        "render_full_doc_anchor_diagnostic_markdown",
        lambda payload: "# fake summary\n",
    )

    payload = mod._load_or_write_summary_outputs(output_root)

    assert payload["aggregate_rows"][0]["config_label"] == "cfg"
    assert (output_root / "summary.json").exists()
    assert (output_root / "summary.md").exists()


def test_finalize_capacity_screen_output_rebuilds_phase_summary(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "capacity"
    screen_root = output_root / "screen"
    screen_root.mkdir(parents=True, exist_ok=True)
    config = mod._RunConfigSpec(
        label="cfg",
        state_dim=128,
        hidden_dim=512,
        n_epochs=32,
        batch_size=64,
        lr=5e-4,
        weight_decay=0.0,
    )
    args = argparse.Namespace(
        benchmark="recoverable_v4",
        train_doc_count=2048,
        priority_family="tree_neural",
        top_k=1,
        capacity_profile="default",
    )
    monkeypatch.setattr(
        mod,
        "_load_or_write_summary_outputs",
        lambda root: {
            "aggregate_rows": [
                {
                    "baseline_family": "tree_neural",
                    "tuning_stage": "capacity_screen",
                    "train_doc_count": 2048,
                    "config_label": "cfg",
                    "val_root_mae_mean": 0.125,
                    "n_runs": 1,
                }
            ],
            "runs": [],
            "summary_json": str(root / "summary.json"),
            "summary_md": str(root / "summary.md"),
        },
    )

    payload = mod.finalize_capacity_screen_output(
        args=args,
        output_root=output_root,
        screen_root=screen_root,
        config_by_label={"cfg": config},
    )

    assert payload["locked_configs"] == [config]
    assert payload["top_rankings"][0]["config_label"] == "cfg"
    assert (output_root / "tree_fno_capacity_screen_summary.json").exists()
    summary = json.loads(
        (output_root / "tree_fno_capacity_screen_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["capacity_profile"] == "default"
    assert summary["capacity_state_dims"] == [128]
    assert summary["capacity_hidden_dims"] == [512]
    assert summary["capacity_n_epochs"] == [32]
    assert summary["top_config_specs"]["cfg"]["tree_checkpoint_metric"] == "val_root_mae"
    summary_md = (output_root / "tree_fno_capacity_screen_summary.md").read_text(
        encoding="utf-8"
    )
    assert "state_dim axis" in summary_md
    assert "Top Config Specs" in summary_md


def test_finalize_capacity_locked_output_records_winning_config_recipe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    output_root = tmp_path / "capacity"
    screen_root = output_root / "screen"
    locked_root = output_root / "locked"
    screen_root.mkdir(parents=True, exist_ok=True)
    locked_root.mkdir(parents=True, exist_ok=True)
    config = mod._RunConfigSpec(
        label="cfg_root_match",
        state_dim=256,
        hidden_dim=1024,
        n_epochs=128,
        batch_size=64,
        lr=2e-4,
        weight_decay=0.0,
        tree_leaf_fno_width=128,
        tree_leaf_fno_n_modes=4,
        tree_leaf_fno_n_layers=4,
        tree_training_schedule="single_stage",
        tree_checkpoint_metric="val_root_mae",
        tree_stage1_checkpoint_metric="val_root_mae",
        tree_stage1_root_weight=1.0,
        slot_count=4,
        fixed_leaf_tokens=16,
    )
    args = argparse.Namespace(
        benchmark="recoverable_v4",
        train_doc_count=10240,
        priority_family="tree_neural",
        top_k=1,
        capacity_profile="root_only_parity_matched_root",
    )

    def _fake_load(root: Path) -> dict[str, object]:
        if Path(root) == locked_root:
            return {
                "aggregate_rows": [
                    {
                        "baseline_family": "tree_neural",
                        "tuning_stage": "capacity_locked",
                        "train_doc_count": 10240,
                        "config_label": "cfg_root_match",
                        "tree_leaf_fno_width": 128,
                        "tree_leaf_fno_n_modes": 4,
                        "tree_leaf_fno_n_layers": 4,
                        "state_dim": 256,
                        "hidden_dim": 1024,
                        "n_epochs": 128,
                        "tree_training_schedule": "single_stage",
                        "tree_checkpoint_metric": "val_root_mae",
                        "tree_stage1_checkpoint_metric": "val_root_mae",
                        "tree_stage1_root_weight": 1.0,
                        "slot_count": 4,
                        "fixed_leaf_tokens": 16,
                        "val_root_mae_mean": 0.01,
                        "test_root_mae_mean": 0.012,
                        "elapsed_s_mean": 100.0,
                        "n_runs": 5,
                    }
                ],
                "runs": [],
            }
        return {"aggregate_rows": [], "runs": []}

    monkeypatch.setattr(mod, "_load_or_write_summary_outputs", _fake_load)
    monkeypatch.setattr(mod, "_write_summary_outputs", lambda root: {"summary_json": str(root / "summary.json")})

    payload = mod.finalize_capacity_locked_output(
        args=args,
        output_root=output_root,
        screen_root=screen_root,
        locked_root=locked_root,
        screen_rankings=[{"config_label": "cfg_root_match"}],
        config_by_label={"cfg_root_match": config},
    )

    assert payload["winning_config_label"] == "cfg_root_match"
    summary = json.loads(
        (output_root / "tree_fno_capacity_locked_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["capacity_profile"] == "root_only_parity_matched_root"
    assert summary["winning_config_spec"]["state_dim"] == 256
    assert summary["winning_config_spec"]["hidden_dim"] == 1024
    assert summary["winning_config_spec"]["tree_training_schedule"] == "single_stage"
    summary_md = (output_root / "tree_fno_capacity_locked_summary.md").read_text(
        encoding="utf-8"
    )
    assert "Winning Config Spec" in summary_md
    assert "tree_stage1_root_weight" in summary_md


def test_scheduler_result_from_summary_accepts_list_completed_items(tmp_path: Path) -> None:
    mod = _load_module()
    output_root = tmp_path / "capacity"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    scheduler_summary = {
        "completed_items": [
            {
                "item_id": "gpu::done",
                "phase": "screen",
                "kind": "gpu_command",
                "metadata": {"job_name": "screen_job"},
                "log_path": str(output_root / "screen.log"),
                "expected_outputs": [str(output_root / "jobs" / "screen" / "summary.json")],
                "gpu_slots": 1,
                "reused": False,
            },
            {
                "item_id": "gpu::reused",
                "phase": "screen",
                "kind": "gpu_command",
                "metadata": {"job_name": "screen_job_reused"},
                "log_path": str(output_root / "screen_reused.log"),
                "expected_outputs": [str(output_root / "jobs" / "screen_reused" / "summary.json")],
                "gpu_slots": 1,
                "reused": True,
            },
        ],
        "failed_items": [
            {
                "item_id": "gpu::failed",
                "phase": "locked",
                "kind": "gpu_command",
                "metadata": {"job_name": "locked_job"},
                "log_path": str(output_root / "locked.log"),
                "expected_outputs": [str(output_root / "jobs" / "locked" / "summary.json")],
                "gpu_slots": 2,
                "returncode": 9,
            }
        ],
    }

    payload = mod._scheduler_result_from_summary(
        output_root=output_root,
        scheduler_summary=scheduler_summary,
        resume_enabled=True,
    )

    assert payload["completed_jobs"][0]["item_id"] == "gpu::done"
    assert payload["skipped_jobs"][0]["item_id"] == "gpu::reused"
    assert payload["failed_jobs"][0]["item_id"] == "gpu::failed"
    assert payload["payload"]["ok"] is True


def test_load_completed_run_keys_preserves_zero_full_doc_budget_share(tmp_path: Path) -> None:
    mod = _load_module()
    runs_dir = tmp_path / "jobs" / "cell" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "run.json").write_text(
        json.dumps(
            {
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural",
                "train_doc_count": 5120,
                "seed": 0,
                "config_label": "budget_tree_default",
                "tuning_stage": "",
                "fixed_leaf_tokens": 16,
                "study_name": "oracle_budget_share_frontier",
                "study_axis": "budget_total_calls_per_doc__full_doc_budget_share",
                "axis_value": "b0p5_a0_root_only_balanced",
                "budget_total_calls": 2560,
                "budget_total_calls_per_doc": 0.5,
                "full_doc_budget_share": 0.0,
                "doc_consumption_mode": "root_only",
                "local_split_mode": "balanced",
                "local_allocation_policy": "breadth_first",
            }
        ),
        encoding="utf-8",
    )

    completed = mod._load_completed_run_keys(tmp_path)

    assert (
        "recoverable_v4",
        "tree_neural",
        5120,
        0,
        "budget_tree_default",
        "",
        0,
        "oracle_budget_share_frontier",
        "budget_total_calls_per_doc__full_doc_budget_share",
        "b0p5_a0_root_only_balanced",
        2560,
        0.5,
        0.0,
        "root_only",
        "balanced",
        "breadth_first",
    ) in completed


def _representation_learnability_run(
    mod,
    *,
    benchmark: str,
    family: str,
    config_label: str,
    train_doc_count: int,
    seed: int,
    val_exact: float,
    gap: float,
    leaf_match: float,
    merge_match: float,
    c2_exact: float,
    root_mae: float,
) -> dict[str, object]:
    return {
        "study_name": mod.REPRESENTATION_LEARNABILITY_STUDY_NAME,
        "tuning_stage": mod.REPRESENTATION_LEARNABILITY_SWEEP_STAGE,
        "benchmark": str(benchmark),
        "baseline_family": str(family),
        "config_label": str(config_label),
        "train_doc_count": int(train_doc_count),
        "seed": int(seed),
        "test_root_mae": float(root_mae),
        "test_mean_leaves_per_doc": 1.0,
        "exact_sketch_markov_sufficiency_gap_score": float(gap),
        "exact_sketch_diagnostics": {
            "direct_selection_metrics": {
                "val": {"val_exact_sketch_direct": float(val_exact)},
                "test": {
                    "c2_on_range_exact_match": float(c2_exact),
                    "exact_projected_root_mae": float(root_mae),
                    "certified_projected_root_mae": float(root_mae),
                    "root_mae_predicted_counts_predicted_endpoints": float(root_mae),
                    "root_mae_oracle_counts_predicted_endpoints": float(root_mae / 2.0),
                    "root_mae_predicted_counts_oracle_endpoints": float(root_mae / 2.0),
                    "learned_merger_gap": 0.0,
                    "leaf_first_accuracy": 1.0,
                    "leaf_last_accuracy": 1.0,
                    "merge_first_accuracy": 1.0,
                    "merge_last_accuracy": 1.0,
                    "leaf_count_off_by_k_histogram": {"0": 1.0},
                    "merge_exact_summary_match_rate_by_depth": {"0": float(merge_match)},
                },
            },
            "tree_neural": {
                "test": {
                    "leaf": {
                        "probe": {"exact_summary_match_rate": float(leaf_match)}
                    },
                    "merge": {
                        "probe": {
                            "exact_summary_match_rate": float(merge_match),
                            "count_mae": 0.0,
                            "first_accuracy": 1.0,
                            "last_accuracy": 1.0,
                        },
                        "decoded_consistency": {
                            "merge_join_bit_accuracy": 1.0,
                            "merge_decoded_consistency_count_mae": 0.0,
                            "merge_decoded_consistency_first_accuracy": 1.0,
                            "merge_decoded_consistency_last_accuracy": 1.0,
                        },
                    },
                }
            },
        },
    }


def test_representation_learnability_benchmark_selection_defaults_and_full_grid() -> None:
    mod = _load_module()
    parser = mod._parser()

    args = parser.parse_args(["representation_learnability"])
    benchmarks = mod._representation_learnability_benchmark_specs(args)

    assert [str(spec.cell_id or spec.name) for spec in benchmarks] == [
        "recoverable_v4",
        "r4_seg4to6",
        "r8_seg7to9",
        "r12_seg10to12",
    ]

    args_full = parser.parse_args(
        ["representation_learnability", "--full-structural-grid"]
    )
    full_benchmarks = mod._representation_learnability_benchmark_specs(args_full)

    assert len(full_benchmarks) == 10
    assert str(full_benchmarks[0].cell_id or full_benchmarks[0].name) == "recoverable_v4"
    assert {
        str(spec.cell_id or spec.name) for spec in full_benchmarks[1:]
    } == {
        "r4_seg4to6",
        "r4_seg7to9",
        "r4_seg10to12",
        "r8_seg4to6",
        "r8_seg7to9",
        "r8_seg10to12",
        "r12_seg4to6",
        "r12_seg7to9",
        "r12_seg10to12",
    }


def test_representation_sufficiency_screen_specs_include_opaque_carrier_matrix() -> None:
    mod = _load_module()
    parser = mod._parser()

    args = parser.parse_args(["representation_sufficiency"])
    specs = mod._representation_sufficiency_screen_config_specs(args)
    config_by_label = dict(specs["config_by_label"])

    assert "slotwise_control_s128" in config_by_label
    assert "shared_feature_s128_phi128" in config_by_label
    assert (
        "opaque_carrier_exact_sketch_s128_phi128_m256_head_scalar_mse" in config_by_label
    )
    assert (
        "opaque_carrier_exact_sketch_s128_phi128_m256_head_support_classifier"
        in config_by_label
    )
    assert (
        "opaque_carrier_exact_sketch_s128_phi128_m256_head_hybrid_ordinal"
        in config_by_label
    )

    opaque_cfg = config_by_label[
        "opaque_carrier_exact_sketch_s128_phi128_m256_head_hybrid_ordinal"
    ]
    assert opaque_cfg.tree_theorem_surface_mode == "opaque_carrier_exact_sketch"
    assert opaque_cfg.tree_score_merge_mode == "exact_projected_sketch"
    assert opaque_cfg.tree_merge_hidden_dim == 256
    assert opaque_cfg.tree_theorem_count_head_mode == "hybrid_ordinal"
    assert opaque_cfg.tree_phi_compose_weight == 0.0
    assert opaque_cfg.tree_phi_contrastive_weight == 0.0

    opaque_meta = specs["config_metadata_by_label"][
        "opaque_carrier_exact_sketch_s128_phi128_m256_head_hybrid_ordinal"
    ]
    assert opaque_meta["promotion_eligible"] is True
    assert opaque_meta["representation_family"] == "opaque_carrier_exact_sketch"
    assert opaque_meta["carrier_merge_input_dim"] == 256
    assert opaque_meta["tree_theorem_count_head_mode"] == "hybrid_ordinal"
    assert opaque_meta["exact_lane"] is True

    shared_meta = specs["config_metadata_by_label"]["shared_feature_s128_phi128"]
    assert shared_meta["promotion_eligible"] is False


def test_representation_metrics_prefer_checkpoint_curve_for_val_exact() -> None:
    mod = _load_module()

    runs = [
        {
            "training_selection_metric_curve": [5.5, 4.25, 4.75],
            "exact_sketch_diagnostics": {
                "direct_selection_metrics": {
                    "val": {
                        "val_exact_sketch_direct": 1.5,
                    }
                }
            },
        },
        {
            "stage2_selection_metric_curve": [3.5, 3.25],
            "exact_sketch_diagnostics": {
                "direct_selection_metrics": {
                    "val": {
                        "val_exact_sketch_direct": 0.75,
                    }
                }
            },
        },
    ]

    metrics = mod._representation_metrics_for_runs(runs)

    assert metrics["val_exact_sketch_direct"]["mean"] == pytest.approx(
        (4.25 + 3.25) / 2.0
    )
    assert metrics["val_exact_sketch_direct"]["std"] == pytest.approx(0.5)


def test_finalize_representation_learnability_output_computes_thresholds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    output_root = tmp_path / "learnability"
    sweep_root = output_root / "sweep"
    output_root.mkdir(parents=True, exist_ok=True)
    sweep_root.mkdir(parents=True, exist_ok=True)

    winner_label = "shared_feature_s128_phi128"
    control_label = "slotwise_control_s128"
    official_label = "official_fno_reference"

    config_by_label = {
        winner_label: mod._RunConfigSpec(
            label=winner_label,
            state_dim=128,
            hidden_dim=512,
            n_epochs=1,
            batch_size=1,
            lr=1e-3,
            weight_decay=0.0,
            tree_theorem_surface_mode="shared_feature",
            tree_theorem_feature_dim=128,
            tree_theorem_feature_hidden_dim=256,
            slot_count=4,
        ),
        control_label: mod._RunConfigSpec(
            label=control_label,
            state_dim=128,
            hidden_dim=512,
            n_epochs=1,
            batch_size=1,
            lr=1e-3,
            weight_decay=0.0,
            tree_theorem_surface_mode="slotwise",
            tree_theorem_feature_dim=128,
            tree_theorem_feature_hidden_dim=256,
            slot_count=4,
        ),
        official_label: mod._RunConfigSpec(
            label=official_label,
            state_dim=128,
            hidden_dim=512,
            n_epochs=1,
            batch_size=1,
            lr=1e-3,
            weight_decay=0.0,
        ),
    }

    synthetic_payload = {
        "runs": [
            _representation_learnability_run(
                mod,
                benchmark="recoverable_v4",
                family="tree_neural",
                config_label=winner_label,
                train_doc_count=128,
                seed=0,
                val_exact=0.10,
                gap=0.08,
                leaf_match=0.93,
                merge_match=0.93,
                c2_exact=0.93,
                root_mae=0.09,
            ),
            _representation_learnability_run(
                mod,
                benchmark="recoverable_v4",
                family="tree_neural",
                config_label=control_label,
                train_doc_count=128,
                seed=0,
                val_exact=0.02,
                gap=0.03,
                leaf_match=0.96,
                merge_match=0.96,
                c2_exact=0.96,
                root_mae=0.05,
            ),
            _representation_learnability_run(
                mod,
                benchmark="recoverable_v4",
                family="official_fno",
                config_label=official_label,
                train_doc_count=128,
                seed=0,
                val_exact=0.0,
                gap=0.01,
                leaf_match=1.0,
                merge_match=1.0,
                c2_exact=1.0,
                root_mae=0.04,
            ),
            _representation_learnability_run(
                mod,
                benchmark="recoverable_v4",
                family="tree_neural",
                config_label=winner_label,
                train_doc_count=512,
                seed=0,
                val_exact=0.04,
                gap=0.04,
                leaf_match=0.95,
                merge_match=0.95,
                c2_exact=0.95,
                root_mae=0.05,
            ),
            _representation_learnability_run(
                mod,
                benchmark="recoverable_v4",
                family="tree_neural",
                config_label=control_label,
                train_doc_count=512,
                seed=0,
                val_exact=0.02,
                gap=0.02,
                leaf_match=0.97,
                merge_match=0.97,
                c2_exact=0.97,
                root_mae=0.05,
            ),
            _representation_learnability_run(
                mod,
                benchmark="recoverable_v4",
                family="official_fno",
                config_label=official_label,
                train_doc_count=512,
                seed=0,
                val_exact=0.0,
                gap=0.01,
                leaf_match=1.0,
                merge_match=1.0,
                c2_exact=1.0,
                root_mae=0.04,
            ),
        ]
    }

    monkeypatch.setattr(
        mod,
        "_write_summary_outputs",
        lambda root: {
            "summary_json": str(Path(root) / "summary.json"),
            "summary_md": str(Path(root) / "summary.md"),
        },
    )
    monkeypatch.setattr(mod, "_load_or_write_summary_outputs", lambda root: synthetic_payload)

    args = argparse.Namespace(
        sweep_train_doc_counts=(128, 512),
        sweep_seeds=(0,),
    )
    result = mod.finalize_representation_learnability_output(
        args=args,
        output_root=output_root,
        sweep_root=sweep_root,
        config_by_label=config_by_label,
        winner_summary={},
        winner_label=winner_label,
        matched_control_label=control_label,
        official_fno_label=official_label,
    )

    summary = json.loads(
        (output_root / "tree_neural_representation_learnability_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["final_status"] == "threshold_estimated_conservative"
    assert summary["winner_label"] == winner_label
    assert len(summary["cell_summaries"]) == 1
    cell_summary = dict(summary["cell_summaries"][0])
    assert cell_summary["benchmark_cell"] == "recoverable_v4"
    assert cell_summary["lean_recoverable_in_principle"] is True
    assert cell_summary["slotwise_control_healthy"] is True
    assert cell_summary["pass_mean"] is True
    assert cell_summary["pass_conservative"] is True
    assert cell_summary["n_min_mean"] == 512
    assert cell_summary["n_min_conservative"] == 512


def test_main_dispatches_representation_learnability_mode(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        mod.sys,
        "argv",
        ["run_tree_neural_full_doc_mig.py", "representation_learnability"],
    )
    monkeypatch.setattr(mod, "_launch_representation_learnability", lambda args: 31)
    monkeypatch.setattr(mod, "_launch_controller", lambda args: 99)

    rc = int(mod.main())

    assert rc == 31
