from __future__ import annotations

"""Shared process-launch helpers for tree-neural full-doc workers."""

import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Set, Tuple

from src.experiments.scheduler import (
    SchedulerConfig,
    SchedulerItem,
    SchedulerRunError,
    run_scheduler,
    summarize_scheduler_plan,
)
from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (
    load_markov_full_doc_anchor_diagnostics_from_output_dir,
    render_full_doc_anchor_diagnostic_markdown,
)
from src.ctreepo.sim.core.tree_neural_facade import (
    JobSpec,
    job_output_dir_name,
    write_run_config_spec,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WORKER_SCRIPT = _REPO_ROOT / "scripts" / "run_tree_neural_full_doc_mig.py"


def _worker_script_path(worker_script: Path | str | None) -> Path:
    if worker_script is None:
        return DEFAULT_WORKER_SCRIPT
    return Path(worker_script)


def worker_command_for_job(
    job: JobSpec,
    *,
    output_dir: Path,
    torch_threads: int,
    use_cuda: bool,
    worker_script: Path | str | None = None,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_family = str(getattr(job.config, "baseline_family", "") or "").strip()
    if config_family != str(job.family).strip():
        raise ValueError(
            "job config baseline_family must match job.family before worker launch "
            f"(config={config_family!r}, job={str(job.family).strip()!r})"
        )
    config_spec_path = output_dir / "requested_run_config.json"
    write_run_config_spec(config_spec_path, job.config)
    cmd = [
        sys.executable,
        str(_worker_script_path(worker_script)),
        "worker",
        "--job-name",
        str(job.job_name),
        "--output-dir",
        str(output_dir),
        "--memory-probe-jsonl",
        str(output_dir / "memory_probe.jsonl"),
        "--family",
        str(job.family),
        "--train-doc-count",
        str(int(job.train_doc_count)),
        "--benchmark",
        str(job.benchmark),
        "--hardness-grid",
        str(job.hardness_grid),
        "--state-dim",
        str(int(job.config.state_dim)),
        "--hidden-dim",
        str(int(job.config.hidden_dim)),
        "--n-epochs",
        str(int(job.config.n_epochs)),
        "--batch-size",
        str(int(job.config.batch_size)),
        "--lr",
        str(float(job.config.lr)),
        "--weight-decay",
        str(float(job.config.weight_decay)),
        "--torch-threads",
        str(int(torch_threads)),
        "--config-label",
        str(job.config.label),
        "--config-spec-json-path",
        str(config_spec_path),
        "--tuning-stage",
        str(job.tuning_stage),
    ]
    if bool(job.grid_cell_ids):
        cmd.extend(["--grid-cell-ids", *[str(cell) for cell in job.grid_cell_ids]])
    if job.config.tree_local_law_weight is not None:
        cmd.extend(
            ["--local-law-weight", str(float(job.config.tree_local_law_weight))]
        )
    if job.config.fixed_leaf_tokens is not None:
        cmd.extend(
            ["--fixed-leaf-tokens", str(int(job.config.fixed_leaf_tokens))]
        )
    if job.config.tree_task_objective_weight is not None:
        cmd.extend(
            [
                "--root-share",
                str(float(job.config.tree_task_objective_weight)),
            ]
        )
    if str(job.config.tree_local_weighting_mode).strip():
        cmd.extend(
            [
                "--tree-local-weighting-mode",
                str(job.config.tree_local_weighting_mode),
            ]
        )
    if str(job.config.tree_exact_collapse_mode).strip():
        cmd.extend(
            [
                "--tree-exact-collapse-mode",
                str(job.config.tree_exact_collapse_mode),
            ]
        )
    if bool(job.config.official_fno_preserve_requested_leaf_tokens):
        cmd.append("--official-fno-preserve-requested-leaf-tokens")
    if bool(job.config.preserve_requested_leaf_tokens):
        cmd.append("--preserve-requested-leaf-tokens")
    if str(job.config.comparison_mode).strip():
        cmd.extend(["--comparison-mode", str(job.config.comparison_mode)])
    if float(job.config.tree_c1_relative_weight) != 1.0:
        cmd.extend(
            [
                "--tree-c1-relative-weight",
                str(float(job.config.tree_c1_relative_weight)),
            ]
        )
    if float(job.config.tree_c2_relative_weight) != 1.0:
        cmd.extend(
            [
                "--tree-c2-relative-weight",
                str(float(job.config.tree_c2_relative_weight)),
            ]
        )
    if float(job.config.tree_c3_relative_weight) != 1.0:
        cmd.extend(
            [
                "--tree-c3-relative-weight",
                str(float(job.config.tree_c3_relative_weight)),
            ]
        )
    if job.config.tree_leaf_fno_width is not None:
        cmd.extend(
            ["--tree-leaf-fno-width", str(int(job.config.tree_leaf_fno_width))]
        )
    if job.config.tree_leaf_fno_n_modes is not None:
        cmd.extend(
            [
                "--tree-leaf-fno-n-modes",
                str(int(job.config.tree_leaf_fno_n_modes)),
            ]
        )
    if job.config.tree_leaf_fno_n_layers is not None:
        cmd.extend(
            [
                "--tree-leaf-fno-n-layers",
                str(int(job.config.tree_leaf_fno_n_layers)),
            ]
        )
    if str(job.config.tree_model_version).strip():
        cmd.extend(
            [
                "--tree-model-version",
                str(job.config.tree_model_version),
            ]
        )
    if str(job.config.tree_batch_runtime_mode).strip():
        cmd.extend(
            [
                "--tree-batch-runtime-mode",
                str(job.config.tree_batch_runtime_mode),
            ]
        )
    if str(job.config.tree_root_supervision_kind).strip():
        cmd.extend(
            [
                "--tree-root-supervision-kind",
                str(job.config.tree_root_supervision_kind),
            ]
        )
    if str(job.config.tree_document_loss_normalization_mode).strip():
        cmd.extend(
            [
                "--tree-document-loss-normalization-mode",
                str(job.config.tree_document_loss_normalization_mode),
            ]
        )
    if str(job.config.tree_supervision_source).strip():
        cmd.extend(
            [
                "--tree-supervision-source",
                str(job.config.tree_supervision_source),
            ]
        )
    if str(job.config.tree_checkpoint_metric).strip():
        cmd.extend(
            [
                "--tree-checkpoint-metric",
                str(job.config.tree_checkpoint_metric),
            ]
        )
    if str(job.config.tree_stage1_checkpoint_metric).strip():
        cmd.extend(
            [
                "--tree-stage1-checkpoint-metric",
                str(job.config.tree_stage1_checkpoint_metric),
            ]
        )
    if str(job.config.tree_stage1_eval_mode).strip():
        cmd.extend(
            [
                "--tree-stage1-eval-mode",
                str(job.config.tree_stage1_eval_mode),
            ]
        )
    if int(job.config.tree_stage1_screen_doc_limit) != 0:
        cmd.extend(
            [
                "--tree-stage1-screen-doc-limit",
                str(int(job.config.tree_stage1_screen_doc_limit)),
            ]
        )
    if int(job.config.tree_stage1_final_exact_doc_limit) != 0:
        cmd.extend(
            [
                "--tree-stage1-final-exact-doc-limit",
                str(int(job.config.tree_stage1_final_exact_doc_limit)),
            ]
        )
    if int(job.config.exact_metric_selection_doc_limit) != 0:
        cmd.extend(
            [
                "--exact-metric-selection-doc-limit",
                str(int(job.config.exact_metric_selection_doc_limit)),
            ]
        )
    if int(job.config.exact_metric_selection_interval) != 1:
        cmd.extend(
            [
                "--exact-metric-selection-interval",
                str(int(job.config.exact_metric_selection_interval)),
            ]
        )
    if int(job.config.tree_exact_eval_max_docs) != 0:
        cmd.extend(
            [
                "--tree-exact-eval-max-docs",
                str(int(job.config.tree_exact_eval_max_docs)),
            ]
        )
    if int(job.config.tree_posttrain_train_doc_limit) != 0:
        cmd.extend(
            [
                "--tree-posttrain-train-doc-limit",
                str(int(job.config.tree_posttrain_train_doc_limit)),
            ]
        )
    if str(job.config.tree_batch_pack_mode).strip():
        cmd.extend(
            [
                "--tree-batch-pack-mode",
                str(job.config.tree_batch_pack_mode),
            ]
        )
    if int(job.config.tree_batch_token_budget) != 0:
        cmd.extend(
            [
                "--tree-batch-token-budget",
                str(int(job.config.tree_batch_token_budget)),
            ]
        )
    if int(job.config.tree_batch_node_budget) != 0:
        cmd.extend(
            [
                "--tree-batch-node-budget",
                str(int(job.config.tree_batch_node_budget)),
            ]
        )
    cmd.append(
        "--tree-batch-autotune"
        if bool(job.config.tree_batch_autotune)
        else "--no-tree-batch-autotune"
    )
    if float(job.config.tree_batch_structural_pad_limit) != 0.5:
        cmd.extend(
            [
                "--tree-batch-structural-pad-limit",
                str(float(job.config.tree_batch_structural_pad_limit)),
            ]
        )
    if int(job.config.tree_batch_auto_queue_min_docs) != 8:
        cmd.extend(
            [
                "--tree-batch-auto-queue-min-docs",
                str(int(job.config.tree_batch_auto_queue_min_docs)),
            ]
        )
    if float(job.config.tree_batch_auto_queue_min_fill_ratio) != 0.5:
        cmd.extend(
            [
                "--tree-batch-auto-queue-min-fill-ratio",
                str(float(job.config.tree_batch_auto_queue_min_fill_ratio)),
            ]
        )
    if int(job.config.tree_eval_workers_per_mig) != 0:
        cmd.extend(
            [
                "--tree-eval-workers-per-mig",
                str(int(job.config.tree_eval_workers_per_mig)),
            ]
        )
    cmd.extend(
        [
            "--gpu-runtime-data-mode",
            str(job.config.gpu_runtime_data_mode),
            "--gpu-runtime-bucket-mode",
            str(job.config.gpu_runtime_bucket_mode),
            "--gpu-runtime-preload-splits",
            *[str(value) for value in job.config.gpu_runtime_preload_splits],
            (
                "--gpu-runtime-preload-targets"
                if bool(job.config.gpu_runtime_preload_targets)
                else "--no-gpu-runtime-preload-targets"
            ),
            "--gpu-runtime-workers-per-mig",
            str(int(job.config.gpu_runtime_workers_per_mig)),
            (
                "--gpu-runtime-allow-multi-worker-screen"
                if bool(job.config.gpu_runtime_allow_multi_worker_screen)
                else "--no-gpu-runtime-allow-multi-worker-screen"
            ),
            "--gpu-runtime-capacity-workers-per-mig",
            str(int(job.config.gpu_runtime_capacity_workers_per_mig)),
        ]
    )
    if str(job.config.tree_stage1_artifact_dir).strip():
        cmd.extend(
            [
                "--tree-stage1-artifact-dir",
                str(job.config.tree_stage1_artifact_dir),
            ]
        )
    if str(job.config.prepared_data_root).strip():
        cmd.extend(
            [
                "--prepared-data-root",
                str(job.config.prepared_data_root),
            ]
        )
    cmd.append(
        "--prepared-data-allow-create"
        if bool(job.config.prepared_data_allow_create)
        else "--no-prepared-data-allow-create"
    )
    if str(job.config.base_bundle_path).strip():
        cmd.extend(
            [
                "--base-bundle-path",
                str(job.config.base_bundle_path),
            ]
        )
    if str(job.config.diagnostic_detail_mode).strip():
        cmd.extend(
            [
                "--diagnostic-detail-mode",
                str(job.config.diagnostic_detail_mode),
            ]
        )
    if str(job.config.posttrain_diagnostics_mode).strip():
        cmd.extend(
            [
                "--posttrain-diagnostics-mode",
                str(job.config.posttrain_diagnostics_mode),
            ]
        )
    if str(job.config.raw_diagnostic_artifact_dir).strip():
        cmd.extend(
            [
                "--raw-diagnostic-artifact-dir",
                str(job.config.raw_diagnostic_artifact_dir),
            ]
        )
    if float(job.config.tree_stage1_root_weight) > 0.0:
        cmd.extend(
            [
                "--tree-stage1-root-weight",
                str(float(job.config.tree_stage1_root_weight)),
            ]
        )
    if float(job.config.tree_join_bit_weight) > 0.0:
        cmd.extend(
            [
                "--tree-join-bit-weight",
                str(float(job.config.tree_join_bit_weight)),
            ]
        )
    if str(job.config.tree_training_schedule).strip():
        cmd.extend(
            [
                "--tree-training-schedule",
                str(job.config.tree_training_schedule),
                "--tree-stage1-epochs",
                str(int(job.config.tree_stage1_epochs)),
                "--tree-stage2-epochs",
                str(int(job.config.tree_stage2_epochs)),
                "--tree-task-head-mode",
                str(job.config.tree_task_head_mode),
                "--tree-theorem-surface-mode",
                str(job.config.tree_theorem_surface_mode),
                "--tree-theorem-count-head-mode",
                str(job.config.tree_theorem_count_head_mode),
                "--tree-theorem-count-ordinal-weight",
                str(float(job.config.tree_theorem_count_ordinal_weight)),
                "--tree-theorem-count-scalar-aux-weight",
                str(float(job.config.tree_theorem_count_scalar_aux_weight)),
                "--tree-theorem-feature-dim",
                str(int(job.config.tree_theorem_feature_dim)),
                "--tree-theorem-feature-hidden-dim",
                str(int(job.config.tree_theorem_feature_hidden_dim)),
                "--tree-merge-hidden-dim",
                str(int(job.config.tree_merge_hidden_dim)),
                "--tree-theorem-score-dim",
                str(int(job.config.tree_theorem_score_dim)),
                "--tree-theorem-fiber-dim",
                str(int(job.config.tree_theorem_fiber_dim)),
                "--tree-theorem-aux-dim",
                str(int(job.config.tree_theorem_aux_dim)),
                "--tree-score-merge-mode",
                str(job.config.tree_score_merge_mode),
                "--tree-phi-compose-weight",
                str(float(job.config.tree_phi_compose_weight)),
                "--tree-phi-contrastive-weight",
                str(float(job.config.tree_phi_contrastive_weight)),
                "--tree-phi-alignment-loss",
                str(job.config.tree_phi_alignment_loss),
                "--tree-c2-mode",
                str(job.config.tree_c2_mode),
                "--tree-summary-spec-root-mode",
                str(job.config.tree_summary_spec_root_mode),
                "--leaf-supervision-kind",
                str(job.config.leaf_supervision_kind),
            ]
        )
        if not bool(job.config.tree_theorem_count_threshold_balance):
            cmd.append("--no-tree-theorem-count-threshold-balance")
    if str(job.config.aligned_sketch_surface).strip():
        cmd.extend(
            [
                "--aligned-sketch-surface",
                str(job.config.aligned_sketch_surface),
            ]
        )
    if str(job.config.summary_spec_name).strip():
        cmd.extend(
            [
                "--summary-spec-name",
                str(job.config.summary_spec_name),
                "--slot-count",
                str(int(job.config.slot_count)),
                "--tree-theorem-count-dim",
                str(int(job.config.tree_theorem_count_dim)),
                "--tree-theorem-first-dim",
                str(int(job.config.tree_theorem_first_dim)),
                "--tree-theorem-last-dim",
                str(int(job.config.tree_theorem_last_dim)),
                "--leaf-label-rate",
                str(float(job.config.leaf_label_rate)),
            ]
        )
    cmd.extend(
        [
            "--theorem-feature-adapter",
            str(job.config.theorem_feature_adapter),
        ]
    )
    if str(job.config.oracle_metric_name).strip():
        cmd.extend(
            [
                "--oracle-metric-name",
                str(job.config.oracle_metric_name),
                "--oracle-same-threshold",
                str(float(job.config.oracle_same_threshold)),
                "--oracle-diff-threshold",
                str(float(job.config.oracle_diff_threshold)),
            ]
        )
    if job.config.theorem_pair_same_threshold is not None:
        cmd.extend(
            [
                "--theorem-pair-same-threshold",
                str(float(job.config.theorem_pair_same_threshold)),
            ]
        )
    if job.config.theorem_pair_diff_threshold is not None:
        cmd.extend(
            [
                "--theorem-pair-diff-threshold",
                str(float(job.config.theorem_pair_diff_threshold)),
            ]
        )
    cmd.extend(
        [
            "--internal-supervision-kind",
            str(job.config.internal_supervision_kind),
            "--internal-label-rate",
            str(float(job.config.internal_label_rate)),
            "--max-internal-depth",
            str(int(job.config.max_internal_depth)),
        ]
    )
    if bool(job.config.leaf_exact_supervision):
        cmd.append("--leaf-exact-supervision")
    if float(job.config.root_weight) != 1.0:
        cmd.extend(["--root-weight", str(float(job.config.root_weight))])
    if float(job.config.schedule_consistency_weight) != 0.0:
        cmd.extend(
            [
                "--schedule-consistency-weight",
                str(float(job.config.schedule_consistency_weight)),
            ]
        )
    if float(job.config.endpoint_loss_scale) != 1.0:
        cmd.extend(
            ["--endpoint-loss-scale", str(float(job.config.endpoint_loss_scale))]
        )
    if bool(job.test_metrics_hidden_during_selection):
        cmd.append("--test-metrics-hidden-during-selection")
    if str(job.study_name).strip():
        cmd.extend(["--study-name", str(job.study_name)])
    if str(job.study_axis).strip():
        cmd.extend(["--study-axis", str(job.study_axis)])
    if str(job.axis_value).strip():
        cmd.extend(["--axis-value", str(job.axis_value)])
    if str(job.locked_tree_neural_config_label).strip():
        cmd.extend(
            [
                "--locked-tree-neural-config-label",
                str(job.locked_tree_neural_config_label),
            ]
        )
    if str(job.selection_metric).strip():
        cmd.extend(["--selection-metric", str(job.selection_metric)])
    if int(job.budget_total_calls) > 0:
        cmd.extend(["--budget-total-calls", str(int(job.budget_total_calls))])
    if float(job.budget_total_calls_per_doc) > 0.0:
        cmd.extend(
            [
                "--budget-total-calls-per-doc",
                str(float(job.budget_total_calls_per_doc)),
            ]
        )
    if math.isfinite(float(job.mass_target_per_doc)):
        cmd.extend(["--mass-target-per-doc", str(float(job.mass_target_per_doc))])
    cmd.extend(
        [
            "--full-doc-budget-share",
            str(float(job.full_doc_budget_share)),
            "--doc-consumption-mode",
            str(job.doc_consumption_mode),
            "--local-split-mode",
            str(job.local_split_mode),
            "--local-allocation-policy",
            str(job.local_allocation_policy),
        ]
    )
    if str(job.package_semantics).strip():
        cmd.extend(["--package-semantics", str(job.package_semantics)])
    if not math.isclose(
        float(job.config.depth_discount_gamma),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        cmd.extend(
            ["--depth-discount-gamma", str(float(job.config.depth_discount_gamma))]
        )
    cmd.extend(["--seeds", *[str(seed) for seed in job.seeds]])
    if bool(use_cuda):
        cmd.append("--use-cuda")
    return cmd


def worker_env_for_token(
    token: str,
    *,
    use_cuda: bool,
) -> dict[str, str]:
    env = dict(os.environ)
    if bool(use_cuda):
        env["CUDA_VISIBLE_DEVICES"] = str(token)
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    return env




def job_completion_keys(
    job: JobSpec,
) -> Set[
    Tuple[
        str,
        str,
        int,
        int,
        str,
        str,
        int,
        str,
        str,
        str,
        int,
        float,
        float,
        str,
        str,
        str,
    ]
]:
    scope_ids = tuple(str(cell) for cell in job.grid_cell_ids) or (str(job.benchmark),)
    leaf_token_key = (
        0
        if job.config.fixed_leaf_tokens is None
        else int(job.config.fixed_leaf_tokens)
    )
    return {
        (
            str(scope_id),
            str(job.family),
            int(job.train_doc_count),
            int(seed),
            str(job.config.label),
            str(job.tuning_stage),
            int(leaf_token_key),
            str(job.study_name),
            str(job.study_axis),
            str(job.axis_value),
            int(job.budget_total_calls),
            float(job.budget_total_calls_per_doc),
            float(job.full_doc_budget_share),
            str(job.doc_consumption_mode),
            str(job.local_split_mode),
            str(job.local_allocation_policy),
        )
        for scope_id in scope_ids
        for seed in job.seeds
    }


def load_completed_run_keys(
    output_root: Path,
) -> Set[
    Tuple[
        str,
        str,
        int,
        int,
        str,
        str,
        int,
        str,
        str,
        str,
        int,
        float,
        float,
        str,
        str,
        str,
    ]
]:
    completed: Set[
        Tuple[
            str,
            str,
            int,
            int,
            str,
            str,
            int,
            str,
            str,
            str,
            int,
            float,
            float,
            str,
            str,
            str,
        ]
    ] = set()
    for path in sorted(Path(output_root).glob("**/runs/*.json")):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        scope_id = str(run.get("cell_id") or run.get("benchmark") or "").strip()
        family = str(run.get("baseline_family") or "").strip()
        if not scope_id or not family:
            continue
        try:
            train_doc_count = int(run.get("train_doc_count"))
            seed = int(run.get("seed"))
        except (TypeError, ValueError):
            continue
        stored_leaf_tokens = (
            int(run.get("fixed_leaf_tokens"))
            if run.get("fixed_leaf_tokens") not in {"", None}
            else 0
        )
        leaf_token_keys = {int(stored_leaf_tokens)}
        if int(stored_leaf_tokens) > 0:
            leaf_token_keys.add(0)
        for leaf_token_key in leaf_token_keys:
            raw_budget_total_calls = run.get("budget_total_calls", 0)
            raw_budget_calls_per_doc = run.get("budget_total_calls_per_doc", 0.0)
            raw_full_doc_budget_share = run.get("full_doc_budget_share", 1.0)
            completed.add(
                (
                    scope_id,
                    family,
                    train_doc_count,
                    seed,
                    str(run.get("config_label", "")),
                    str(run.get("tuning_stage", "")),
                    int(leaf_token_key),
                    str(run.get("study_name", "")),
                    str(run.get("study_axis", "")),
                    str(run.get("axis_value", "")),
                    (
                        0
                        if raw_budget_total_calls in {"", None}
                        else int(raw_budget_total_calls)
                    ),
                    (
                        0.0
                        if raw_budget_calls_per_doc in {"", None}
                        else float(raw_budget_calls_per_doc)
                    ),
                    (
                        1.0
                        if raw_full_doc_budget_share in {"", None}
                        else float(raw_full_doc_budget_share)
                    ),
                    str(run.get("doc_consumption_mode", "")),
                    str(run.get("local_split_mode", "")),
                    str(run.get("local_allocation_policy", "")),
                )
            )
    return completed


def read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:

    rows: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return rows
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, Mapping):
            rows.append({str(key): value for key, value in dict(payload).items()})
    return rows


def summarize_memory_probe_file(path: Path) -> Dict[str, Any]:

    rows = read_jsonl_rows(path)
    job_dir = path.parent
    first_event = str(rows[0].get("event", "")) if rows else ""
    last_event = str(rows[-1].get("event", "")) if rows else ""
    max_private_dirty_kib = 0
    max_private_dirty_event = ""
    max_rss_kib = 0
    max_rss_event = ""
    max_swap_kib = 0
    max_swap_event = ""
    largest_private_dirty_delta_kib = 0
    largest_private_dirty_delta_from_event = ""
    largest_private_dirty_delta_to_event = ""
    largest_private_dirty_delta_from_kib = 0
    largest_private_dirty_delta_to_kib = 0
    reached_pre_exact_eval_batch = False
    reached_post_exact_eval_batch = False
    reached_post_exact_eval_batch_trim = False
    previous_row: Dict[str, Any] | None = None
    for row in rows:
        event = str(row.get("event", ""))
        private_dirty_kib = int(row.get("private_dirty_kib", 0) or 0)
        rss_kib = int(row.get("rss_kib", 0) or 0)
        swap_kib = int(row.get("swap_kib", 0) or 0)
        if private_dirty_kib >= max_private_dirty_kib:
            max_private_dirty_kib = int(private_dirty_kib)
            max_private_dirty_event = event
        if rss_kib >= max_rss_kib:
            max_rss_kib = int(rss_kib)
            max_rss_event = event
        if swap_kib >= max_swap_kib:
            max_swap_kib = int(swap_kib)
            max_swap_event = event
        if event == "pre_exact_eval_batch":
            reached_pre_exact_eval_batch = True
        elif event == "post_exact_eval_batch":
            reached_post_exact_eval_batch = True
        elif event == "post_exact_eval_batch_trim":
            reached_post_exact_eval_batch_trim = True
        if previous_row is not None:
            previous_private_dirty_kib = int(
                previous_row.get("private_dirty_kib", 0) or 0
            )
            delta_kib = int(private_dirty_kib - previous_private_dirty_kib)
            if delta_kib >= largest_private_dirty_delta_kib:
                largest_private_dirty_delta_kib = int(delta_kib)
                largest_private_dirty_delta_from_event = str(
                    previous_row.get("event", "")
                )
                largest_private_dirty_delta_to_event = event
                largest_private_dirty_delta_from_kib = int(previous_private_dirty_kib)
                largest_private_dirty_delta_to_kib = int(private_dirty_kib)
        previous_row = row
    return {
        "job_dir": str(job_dir),
        "job_dir_name": str(job_dir.name),
        "probe_jsonl": str(path),
        "n_rows": int(len(rows)),
        "first_event": first_event,
        "last_event": last_event,
        "reached_pre_exact_eval_batch": bool(reached_pre_exact_eval_batch),
        "reached_post_exact_eval_batch": bool(reached_post_exact_eval_batch),
        "reached_post_exact_eval_batch_trim": bool(reached_post_exact_eval_batch_trim),
        "max_private_dirty_kib": int(max_private_dirty_kib),
        "max_private_dirty_event": max_private_dirty_event,
        "max_rss_kib": int(max_rss_kib),
        "max_rss_event": max_rss_event,
        "max_swap_kib": int(max_swap_kib),
        "max_swap_event": max_swap_event,
        "largest_private_dirty_delta_kib": int(largest_private_dirty_delta_kib),
        "largest_private_dirty_delta_from_event": largest_private_dirty_delta_from_event,
        "largest_private_dirty_delta_to_event": largest_private_dirty_delta_to_event,
        "largest_private_dirty_delta_from_kib": int(
            largest_private_dirty_delta_from_kib
        ),
        "largest_private_dirty_delta_to_kib": int(
            largest_private_dirty_delta_to_kib
        ),
    }


def write_memory_probe_summary(output_root: Path) -> Dict[str, Any]:

    probe_paths = sorted(output_root.rglob("memory_probe.jsonl"))
    worker_summaries = [
        summarize_memory_probe_file(path)
        for path in probe_paths
    ]
    peak_private_dirty = sorted(
        worker_summaries,
        key=lambda row: int(row.get("max_private_dirty_kib", 0) or 0),
        reverse=True,
    )
    peak_private_dirty_deltas = sorted(
        worker_summaries,
        key=lambda row: int(row.get("largest_private_dirty_delta_kib", 0) or 0),
        reverse=True,
    )
    payload = {
        "output_root": str(output_root),
        "probe_files_found": int(len(probe_paths)),
        "jobs_with_rows": int(
            sum(1 for row in worker_summaries if int(row.get("n_rows", 0) or 0) > 0)
        ),
        "jobs_reaching_pre_exact_eval_batch": int(
            sum(
                1
                for row in worker_summaries
                if bool(row.get("reached_pre_exact_eval_batch", False))
            )
        ),
        "jobs_reaching_post_exact_eval_batch": int(
            sum(
                1
                for row in worker_summaries
                if bool(row.get("reached_post_exact_eval_batch", False))
            )
        ),
        "jobs_reaching_post_exact_eval_batch_trim": int(
            sum(
                1
                for row in worker_summaries
                if bool(row.get("reached_post_exact_eval_batch_trim", False))
            )
        ),
        "peak_private_dirty_jobs": list(peak_private_dirty[:8]),
        "largest_private_dirty_delta_jobs": list(peak_private_dirty_deltas[:8]),
        "workers": list(worker_summaries),
    }
    summary_path = output_root / "memory_probe_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload["summary_json"] = str(summary_path)
    return payload


def run_job_batch(
    *,
    output_root: Path,
    jobs: Sequence[JobSpec],
    mig_uuids: Sequence[str],
    resume_enabled: bool,
    use_cuda: bool,
    torch_threads: int,
    manifest_payload: Mapping[str, Any],
    write_summary_outputs_func: Callable[[Path], Mapping[str, Any]] | None = None,
    worker_command_func: Callable[..., Sequence[str]] | None = None,
    load_completed_run_keys_func: Callable[[Path], Set[Tuple[Any, ...]]] | None = None,
    job_completion_keys_func: Callable[[JobSpec], Set[Tuple[Any, ...]]] | None = None,
    cwd: Path | str | None = None,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_writer = write_summary_outputs if write_summary_outputs_func is None else write_summary_outputs_func
    worker_command_builder = worker_command_for_job if worker_command_func is None else worker_command_func
    completed_key_loader = load_completed_run_keys if load_completed_run_keys_func is None else load_completed_run_keys_func
    completion_key_builder = job_completion_keys if job_completion_keys_func is None else job_completion_keys_func
    run_cwd = _REPO_ROOT if cwd is None else Path(cwd)
    job_root = output_root / "jobs"
    job_root.mkdir(parents=True, exist_ok=True)
    (output_root / "mig_job_manifest.json").write_text(
        json.dumps(dict(manifest_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    completed_run_keys = completed_key_loader(output_root) if bool(resume_enabled) else set()
    skipped_jobs: List[Dict[str, Any]] = []
    pending: List[_JobSpec] = []
    for job in jobs:
        required_keys = completion_key_builder(job)
        if required_keys and required_keys.issubset(completed_run_keys):
            skipped_jobs.append(
                {
                    "job_name": job.job_name,
                    "family": job.family,
                    "train_doc_count": int(job.train_doc_count),
                    "config_label": str(job.config.label),
                    "tuning_stage": str(job.tuning_stage),
                    "seeds": [int(seed) for seed in job.seeds],
                    "reason": "already_completed",
                }
            )
            continue
        pending.append(job)

    active: List[Dict[str, Any]] = []
    completed: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    available_tokens = list(mig_uuids)
    stop_requested = False
    force_terminate_requested = False

    def _request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested, force_terminate_requested
        if not stop_requested:
            stop_requested = True
            print(
                f"received signal {int(signum)}; pausing launch queue and waiting for active jobs to finish",
                flush=True,
            )
            return
        if force_terminate_requested:
            return
        force_terminate_requested = True
        print(
            f"received signal {int(signum)} again; terminating {len(active)} active workers",
            flush=True,
        )
        for entry in active:
            proc = entry.get("proc")
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    continue

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    if skipped_jobs:
        print(
            f"skipping {len(skipped_jobs)} completed jobs already present under {output_root}",
            flush=True,
        )

    while pending or active:
        while pending and available_tokens and not stop_requested:
            token = available_tokens.pop(0)
            job = pending.pop(0)
            job_output_dir = job_root / job_output_dir_name(job.job_name)
            job_output_dir.mkdir(parents=True, exist_ok=True)
            log_path = job_output_dir / "worker.log"
            log_fh = open(log_path, "w", encoding="utf-8")
            cmd = worker_command_builder(
                job,
                output_dir=job_output_dir,
                torch_threads=int(torch_threads),
                use_cuda=bool(use_cuda),
            )
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(token)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=log_fh,
                cwd=str(run_cwd),
                env=env,
                text=True,
            )
            active.append(
                {
                    "job": job,
                    "proc": proc,
                    "log_path": log_path,
                    "log_fh": log_fh,
                    "mig_uuid": token,
                }
            )
            print(
                f"launched {job.job_name} seeds={list(job.seeds)} on {token[:18]} pid={proc.pid}",
                flush=True,
            )

        time.sleep(1.0)
        still_active: List[Dict[str, Any]] = []
        for entry in active:
            proc = entry["proc"]
            if proc.poll() is None:
                still_active.append(entry)
                continue
            stdout_text = proc.stdout.read() if proc.stdout is not None else ""
            entry["log_fh"].close()
            available_tokens.append(str(entry["mig_uuid"]))
            if int(proc.returncode) != 0:
                failed.append(
                    {
                        "job_name": entry["job"].job_name,
                        "family": entry["job"].family,
                        "train_doc_count": int(entry["job"].train_doc_count),
                        "config_label": str(entry["job"].config.label),
                        "tuning_stage": str(entry["job"].tuning_stage),
                        "returncode": int(proc.returncode),
                        "log_path": str(entry["log_path"]),
                        "stdout_tail": stdout_text[-500:],
                    }
                )
                print(
                    f"failed {entry['job'].job_name} rc={proc.returncode} log={entry['log_path']}",
                    flush=True,
                )
                continue
            result = json.loads(stdout_text.strip().splitlines()[-1])
            completed.append(result)
            seed_label = ",".join(str(seed) for seed in list(result.get("job_seeds") or []))
            if bool(result.get("test_metrics_hidden_during_selection", False)):
                print(
                    "completed "
                    f"{result['job_name']} "
                    f"seeds=[{seed_label}] "
                    f"val_root_mae={result['val_root_mae']:.6g} "
                    f"selection={result['selection_metric_name'] or 'val_root_mae'} "
                    f"cfg={result['config_label']} "
                    "(test hidden for selection)",
                    flush=True,
                )
            elif bool(result.get("objective_weights_active", False)):
                print(
                    "completed "
                    f"{result['job_name']} "
                    f"seeds=[{seed_label}] "
                    f"root_mae={result['test_root_mae']:.6g} "
                    f"param={result['parameterization']} "
                    f"weights=({result['local_law_c1_weight']:.4g},"
                    f"{result['local_law_c2_weight']:.4g},"
                    f"{result['local_law_c3_weight']:.4g})",
                    flush=True,
                )
            else:
                print(
                    "completed "
                    f"{result['job_name']} "
                    f"seeds=[{seed_label}] "
                    f"root_mae={result['test_root_mae']:.6g} "
                    "(closed_form_control; local-law weights inactive)",
                    flush=True,
                )
        active = still_active

    controller_summary = {
        "completed_jobs": completed,
        "failed_jobs": failed,
        "skipped_jobs": skipped_jobs,
        "resume_enabled": bool(resume_enabled),
        "stop_requested": bool(stop_requested),
    }
    (output_root / "controller_results.json").write_text(
        json.dumps(controller_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    try:
        payload = dict(summary_writer(output_root))
    except FileNotFoundError:
        payload = {"runs": [], "aggregate_rows": []}
    return {
        "payload": payload,
        "summary_json": str(output_root / "summary.json"),
        "summary_md": str(output_root / "summary.md"),
        "completed_jobs": completed,
        "failed_jobs": failed,
        "skipped_jobs": skipped_jobs,
        "resume_enabled": bool(resume_enabled),
        "stop_requested": bool(stop_requested),
        "output_root": str(output_root),
    }


def scheduler_item_for_job(
    *,
    phase: str,
    item_id: str,
    output_root: Path,
    job: JobSpec,
    torch_threads: int,
    use_cuda: bool,
    gpu_slots: int = 1,
    allowed_devices: Sequence[str] = (),
) -> SchedulerItem:
    def _scheduler_scope() -> str:
        if str(job.hardness_grid).strip():
            return str(job.hardness_grid)
        return str(job.benchmark)

    def _scheduler_package() -> str:
        cfg = job.config
        if float(job.full_doc_budget_share) < 0.999999:
            return ""
        if str(job.doc_consumption_mode or "root_only") not in {"", "root_only"}:
            return ""
        if str(job.local_split_mode or "balanced") not in {"", "balanced"}:
            return ""
        if str(cfg.leaf_supervision_kind or "count_only") != "count_only":
            return ""
        if abs(float(cfg.leaf_label_rate)) > 1e-9:
            return ""
        if str(cfg.internal_supervision_kind or "none") != "none":
            return ""
        if abs(float(cfg.internal_label_rate)) > 1e-9:
            return ""
        return "full100"

    job_output_dir = output_root / "jobs" / job_output_dir_name(str(job.job_name))
    metadata: Dict[str, Any] = {
        "job_name": str(job.job_name),
        "task_name": str(job.job_name),
        "train_docs": int(job.train_doc_count),
        "model_family": str(job.family),
        "worker_kind": "full_doc_diagnostics",
        "n_epochs": int(job.config.n_epochs),
    }
    scope = _scheduler_scope().strip()
    if scope:
        metadata["scope"] = scope
    package = _scheduler_package().strip()
    if package:
        metadata["package"] = package
    return SchedulerItem(
        item_id=str(item_id),
        phase=str(phase),
        kind="gpu_command",
        expected_outputs=(str(job_output_dir / "summary.json"),),
        command=tuple(
            str(arg)
            for arg in worker_command_for_job(
                job,
                output_dir=job_output_dir,
                torch_threads=int(torch_threads),
                use_cuda=bool(use_cuda),
            )
        ),
        log_path=str(job_output_dir / "worker.log"),
        metadata=metadata,
        gpu_slots=max(1, int(gpu_slots)),
        allowed_devices=tuple(str(token) for token in allowed_devices if str(token).strip()),
    )


def scheduler_result_from_summary(
    *,
    output_root: Path,
    scheduler_summary: Mapping[str, Any],
    resume_enabled: bool,
) -> Dict[str, Any]:
    def _iter_item_infos(name: str) -> List[Mapping[str, Any]]:
        raw = dict(scheduler_summary).get(name)
        if isinstance(raw, Mapping):
            return [dict(info) for info in raw.values()]
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            return [dict(info) for info in raw if isinstance(info, Mapping)]
        return []

    completed_jobs: List[Dict[str, Any]] = []
    skipped_jobs: List[Dict[str, Any]] = []
    failed_jobs: List[Dict[str, Any]] = []
    for info in _iter_item_infos("completed_items"):
        if str(info.get("kind", "")) != "gpu_command":
            continue
        payload = {
            "item_id": str(info.get("item_id", "")),
            "phase": str(info.get("phase", "")),
            "job_name": str(dict(info.get("metadata") or {}).get("job_name", "")),
            "log_path": str(info.get("log_path", "")),
            "expected_outputs": [str(path) for path in list(info.get("expected_outputs") or [])],
            "gpu_slots": int(info.get("gpu_slots", 1) or 1),
        }
        if bool(info.get("reused", False)):
            payload["reason"] = "already_completed"
            skipped_jobs.append(payload)
        else:
            completed_jobs.append(payload)
    for info in _iter_item_infos("failed_items"):
        if str(info.get("kind", "")) != "gpu_command":
            continue
        failed_jobs.append(
            {
                "item_id": str(info.get("item_id", "")),
                "phase": str(info.get("phase", "")),
                "job_name": str(dict(info.get("metadata") or {}).get("job_name", "")),
                "returncode": int(info.get("returncode", 1) or 1),
                "log_path": str(info.get("log_path", "")),
                "expected_outputs": [
                    str(path) for path in list(info.get("expected_outputs") or [])
                ],
                "gpu_slots": int(info.get("gpu_slots", 1) or 1),
            }
        )
    return {
        "payload": (
            json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            if (output_root / "summary.json").exists()
            else {}
        ),
        "summary_json": str(output_root / "summary.json"),
        "summary_md": str(output_root / "summary.md"),
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "skipped_jobs": skipped_jobs,
        "resume_enabled": bool(resume_enabled),
        "stop_requested": False,
        "output_root": str(output_root),
        "scheduler_summary": dict(scheduler_summary),
    }


def run_scheduler_bundle(
    *,
    output_root: Path,
    items: Sequence[SchedulerItem],
    devices: Sequence[str],
    max_gpu_items_per_mig: int,
    launch_stagger_seconds: float,
    cleanup_stale_children: bool,
    resume_enabled: bool,
    manifest_payload: Mapping[str, Any],
    min_mem_available_kib: int = 128 * 1024 * 1024,
    min_swap_free_kib: int = 2 * 1024 * 1024,
    cancel_on_failure: bool = True,
    run_scheduler_func: Callable[..., Mapping[str, Any]] | None = None,
    memory_probe_summary_writer: Callable[[Path], Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    scheduler_runner = run_scheduler if run_scheduler_func is None else run_scheduler_func
    memory_probe_writer = (
        write_memory_probe_summary
        if memory_probe_summary_writer is None
        else memory_probe_summary_writer
    )
    manifest_path = output_root / "mig_job_manifest.json"
    scheduler_status_path = output_root / "scheduler_status.json"
    scheduler_event_log_path = output_root / "scheduler_events.jsonl"
    scheduler_failure_snapshot_path = output_root / "scheduler_failure_snapshot.json"
    manifest_path.write_text(
        json.dumps(dict(manifest_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    try:
        scheduler_summary = scheduler_runner(
            items,
            config=SchedulerConfig(
                devices=tuple(str(device) for device in devices),
                max_gpu_items_per_mig=int(max(1, int(max_gpu_items_per_mig))),
                launch_stagger_seconds=float(max(0.0, float(launch_stagger_seconds))),
                cleanup_stale_children=bool(cleanup_stale_children),
                cancel_on_failure=bool(cancel_on_failure),
                min_mem_available_kib=int(max(0, int(min_mem_available_kib))),
                min_swap_free_kib=int(max(0, int(min_swap_free_kib))),
                root_markers=(str(output_root),),
                status_path=str(scheduler_status_path),
                event_log_path=str(scheduler_event_log_path),
                failure_snapshot_path=str(scheduler_failure_snapshot_path),
            ),
        )
    except SchedulerRunError as exc:
        scheduler_summary = dict(exc.summary)
    result = scheduler_result_from_summary(
        output_root=output_root,
        scheduler_summary=scheduler_summary,
        resume_enabled=bool(resume_enabled),
    )
    memory_probe_summary = dict(memory_probe_writer(output_root))
    controller_summary = {
        "completed_jobs": list(result["completed_jobs"]),
        "failed_jobs": list(result["failed_jobs"]),
        "skipped_jobs": list(result["skipped_jobs"]),
        "resume_enabled": bool(result["resume_enabled"]),
        "stop_requested": bool(result["stop_requested"]),
        "scheduler": dict(scheduler_summary),
        "scheduler_status_json": str(scheduler_status_path),
        "scheduler_events_jsonl": str(scheduler_event_log_path),
        "scheduler_failure_snapshot_json": str(scheduler_failure_snapshot_path),
        "memory_probe_summary_json": str(memory_probe_summary["summary_json"]),
    }
    (output_root / "controller_results.json").write_text(
        json.dumps(controller_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result["memory_probe_summary_json"] = str(memory_probe_summary["summary_json"])
    return result


def scheduler_cli_payload(
    *,
    items: Sequence[SchedulerItem],
    devices: Sequence[str],
    max_gpu_items_per_mig: int,
    launch_stagger_seconds: float,
    min_mem_available_kib: int,
    min_swap_free_kib: int,
    manifest_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "scheduler": {
            **summarize_scheduler_plan(
                items,
                devices=tuple(str(device) for device in devices),
                max_gpu_items_per_mig=int(max(1, int(max_gpu_items_per_mig))),
                launch_stagger_seconds=float(max(0.0, float(launch_stagger_seconds))),
            ),
            "min_mem_available_kib": int(max(0, int(min_mem_available_kib))),
            "min_swap_free_kib": int(max(0, int(min_swap_free_kib))),
        },
        "manifest": dict(manifest_payload),
    }


def sanitize_label(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value).strip()).strip("_")
    return cleaned or "default"


def write_summary_outputs(output_root: Path) -> Dict[str, Any]:
    payload = load_markov_full_doc_anchor_diagnostics_from_output_dir(output_root)
    summary_json = output_root / "summary.json"
    summary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_md = output_root / "summary.md"
    summary_md.write_text(
        render_full_doc_anchor_diagnostic_markdown(payload),
        encoding="utf-8",
    )
    payload["summary_json"] = str(summary_json)
    payload["summary_md"] = str(summary_md)
    return payload


def write_combined_runs_output(
    *,
    output_root: Path,
    runs: Sequence[Mapping[str, Any]],
    write_summary_outputs_func: Callable[[Path], Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    runs_dir = output_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for index, run in enumerate(runs):
        family = str(run.get("baseline_family", "run"))
        seed = int(run.get("seed", index))
        config_label = sanitize_label(str(run.get("config_label", "")) or "default")
        stage_label = sanitize_label(str(run.get("tuning_stage", "")) or "final")
        cell_id = sanitize_label(str(run.get("cell_id", "") or run.get("benchmark", "")))
        study_axis = sanitize_label(str(run.get("study_axis", "")))
        axis_value = sanitize_label(str(run.get("axis_value", "")))
        leaf_tokens = (
            ""
            if run.get("fixed_leaf_tokens") in {"", None}
            else f"__leaf_{int(run.get('fixed_leaf_tokens', 0))}"
        )
        study_suffix = ""
        if study_axis and axis_value:
            study_suffix = f"__{study_axis}_{axis_value}"
        stem = (
            f"{family}__{cell_id}__cfg_{config_label}__stage_{stage_label}"
            f"{leaf_tokens}{study_suffix}__seed_{seed}"
        )
        (runs_dir / f"{stem}.json").write_text(
            json.dumps(dict(run), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    summary_writer = write_summary_outputs if write_summary_outputs_func is None else write_summary_outputs_func
    return dict(summary_writer(output_root))

__all__ = [
    "DEFAULT_WORKER_SCRIPT",
    "job_completion_keys",
    "load_completed_run_keys",
    "read_jsonl_rows",
    "run_job_batch",
    "run_scheduler_bundle",
    "sanitize_label",
    "scheduler_cli_payload",
    "scheduler_item_for_job",
    "scheduler_result_from_summary",
    "summarize_memory_probe_file",
    "worker_command_for_job",
    "worker_env_for_token",
    "write_combined_runs_output",
    "write_memory_probe_summary",
    "write_summary_outputs",
]
