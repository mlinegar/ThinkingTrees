from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

from scripts.benchmark_supervision_recovery_runtime_ablation import (
    _plan_from_args,
    _summarize_pairs,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_completed_raw_task(
    root: Path,
    *,
    task_name: str,
    package_name: str,
    scope_key: str,
    baseline_family: str,
    train_docs: int,
    seed: int,
    wall_clock_s: float,
    epochs_total: int,
    train_loop_s: float,
) -> None:
    task_dir = root / "attempts" / "20260327_000000" / "raw" / task_name
    _write_json(
        task_dir / "progress.json",
        {
            "state": "completed",
            "stage": "completed",
            "epochs_total": epochs_total,
            "wall_clock_s": wall_clock_s,
        },
    )
    _write_json(
        task_dir / "summary.json",
        {
            "wall_clock_s": wall_clock_s,
            "config": {
                "pipeline_supervision_recovery_package": package_name,
                "pipeline_supervision_recovery_scope": scope_key,
                "pipeline_supervision_recovery_scope_label": scope_key,
                "train_docs": train_docs,
                "gpu_runtime_data_mode": "resident",
                "gpu_runtime_bucket_mode": "leaf_count_auto_queue",
                "tree_batch_pack_mode": "fixed_fused",
                "tree_training_schedule": "two_stage",
                "tree_stage1_epochs": 5,
                "tree_stage2_epochs": 10,
            },
            "runs": [
                {
                    "baseline_family": baseline_family,
                    "cell_id": scope_key,
                    "train_doc_count": train_docs,
                    "seed": seed,
                    "fit_diagnostics": {"epochs_completed": epochs_total},
                    "timing_breakdown": {
                        "train_loop_s": train_loop_s,
                        "stage1_train_loop_s": train_loop_s * 0.4,
                        "stage2_train_loop_s": train_loop_s * 0.6,
                        "exact_metric_eval_s": 2.0,
                    },
                    "runtime_efficiency": {
                        "runtime_data_mode": "resident",
                        "runtime_bucket_mode": "leaf_count_auto_queue",
                        "steady_state_h2d_bytes": 0.0,
                        "steady_state_h2d_events": 0.0,
                        "resident_store_hits": 32.0,
                        "resident_store_misses": 0.0,
                        "auto_queue_fused_batches": 18.0,
                        "fixed_shape_dense_bucket_store_hits": 9.0,
                    },
                }
            ],
        },
    )


def test_diagnose_supervision_recovery_runtime_cli(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _make_completed_raw_task(
        root / "supervision_recovery",
        task_name="recoverable_v4__train01024__full10__tree_neural__d0",
        package_name="full10",
        scope_key="recoverable_v4",
        baseline_family="tree_neural",
        train_docs=1024,
        seed=42,
        wall_clock_s=45.0,
        epochs_total=15,
        train_loop_s=30.0,
    )
    _make_completed_raw_task(
        root / "supervision_recovery",
        task_name="recoverable_v4__train01024__full10__fno__d0",
        package_name="full10",
        scope_key="recoverable_v4",
        baseline_family="official_fno",
        train_docs=1024,
        seed=42,
        wall_clock_s=40.0,
        epochs_total=128,
        train_loop_s=0.0,
    )
    output_dir = tmp_path / "diagnosis"
    repo_root = Path(__file__).resolve().parents[2]
    subprocess.check_call(
        [
            sys.executable,
            "scripts/diagnose_supervision_recovery_runtime.py",
            "--input-root",
            str(root),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
    )
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "report.md").read_text(encoding="utf-8")

    assert payload["tree_fast_path_confirmed_runs"] == 1
    assert payload["tree_partial_or_fallback_runs"] == 0
    assert payload["tree_zero_h2d_rate"] == 1.0
    assert payload["fno_context"]["completed_fno_rows"] == 1
    assert "## A/B Proof Template" in markdown
    assert "official_fno" in markdown


@dataclass
class _FakeTask:
    name: str
    output_path: Path
    metadata: dict[str, object]


def test_runtime_ablation_plan_keeps_task_identity_and_swaps_runtime_knobs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_dir = tmp_path / "prepared" / "raw" / "recoverable_v4__train01024__full10__tree_neural__d0"
    _write_json(
        task_dir / "task.request",
        {
            "name": "recoverable_v4__train01024__full10__tree_neural__d0",
            "config": {
                "pipeline_supervision_recovery_scope": "recoverable_v4",
                "pipeline_supervision_recovery_package": "full10",
                "train_docs": 1024,
                "seed": 42,
                "tree_batch_pack_mode": "fixed_fused",
                "gpu_runtime_bucket_mode": "leaf_count_auto_queue",
            },
        },
    )
    fake_tasks = [
        _FakeTask(
            name="recoverable_v4__train01024__full10__tree_neural__d0",
            output_path=task_dir / "summary.json",
            metadata={
                "model_family": "tree_neural",
                "scope": "recoverable_v4",
                "package": "full10",
                "train_docs": 1024,
            },
        )
    ]

    monkeypatch.setattr(
        "scripts.benchmark_supervision_recovery_runtime_ablation._build_base_tasks",
        lambda **_: fake_tasks,
    )

    class Args:
        config = tmp_path / "cfg.toml"
        output_dir = tmp_path / "bench"
        scope = ("recoverable_v4",)
        package = ("full10",)
        train_docs = (1024,)
        data_seeds = (0,)
        tree_family = "tree_neural"
        device_label = ""
        optimized_pack_mode = "fixed_fused"
        optimized_bucket_mode = "leaf_count_auto_queue"
        baseline_pack_mode = "structure_bucket"
        baseline_bucket_mode = "exact_then_bucketed"
        runtime_data_mode = "resident"
        tree_batch_autotune = False
        runtime_tree_batch_structural_pad_limit = 0.5
        runtime_tree_batch_auto_queue_min_docs = 8
        runtime_tree_batch_auto_queue_min_fill_ratio = 0.5

    plan = _plan_from_args(Args())
    assert len(plan["pairs"]) == 1
    pair = plan["pairs"][0]
    assert pair["identity"]["scope"] == "recoverable_v4"
    assert pair["identity"]["package"] == "full10"
    assert pair["baseline_task"]["config"]["tree_batch_pack_mode"] == "structure_bucket"
    assert pair["baseline_task"]["config"]["gpu_runtime_bucket_mode"] == "exact_then_bucketed"
    assert pair["optimized_task"]["config"]["tree_batch_pack_mode"] == "fixed_fused"
    assert pair["optimized_task"]["config"]["gpu_runtime_bucket_mode"] == "leaf_count_auto_queue"
    assert pair["optimized_task"]["config"]["train_docs"] == pair["baseline_task"]["config"]["train_docs"]


def test_runtime_ablation_summary_computes_speedups() -> None:
    summary = _summarize_pairs(
        [
            {
                "variant": "baseline",
                "scope_key": "recoverable_v4",
                "train_doc_count": 1024,
                "package_name": "full10",
                "seed": 42,
                "wall_clock_s": 20.0,
                "train_loop_s": 10.0,
            },
            {
                "variant": "optimized",
                "scope_key": "recoverable_v4",
                "train_doc_count": 1024,
                "package_name": "full10",
                "seed": 42,
                "wall_clock_s": 5.0,
                "train_loop_s": 2.0,
            },
        ]
    )

    assert summary["median_wall_clock_speedup"] == 4.0
    assert summary["median_train_loop_speedup"] == 5.0
    assert summary["grouped_rows"][0]["median_wall_clock_speedup"] == 4.0
