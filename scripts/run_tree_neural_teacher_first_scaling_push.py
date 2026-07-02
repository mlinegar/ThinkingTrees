#!/usr/bin/env python3
"""Teacher-first frontier scaling push.

Runs the current teacher-first frontier families across a practical
train-size ladder and aggregates the per-size tournament summaries.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.core.tree_neural_execution import (  # noqa: E402
    job_completion_keys,
    load_completed_run_keys,
    worker_command_for_job,
    worker_env_for_token,
    write_summary_outputs,
)
from src.ctreepo.sim.core.tree_neural_config_recipes import (  # noqa: E402
    resolved_tree_batch_pack_mode,
)
from src.ctreepo.sim.core.tree_neural_facade import (  # noqa: E402
    JobSpec,
    RunConfigSpec,
    discover_mig_uuids,
    job_output_dir_name,
    parse_mig_uuids,
)
from scripts import run_tree_neural_teacher_first_push as tfpush  # noqa: E402



FRONTIER_FAMILY_ROOT_WEIGHTS: Mapping[str, tuple[float, ...]] = {
    "teacherfirst_shared_feature_adapters_phi128": (0.0, 0.5, 1.0),
    "teacherfirst_shared_feature_phi192": (0.0, 0.5),
    "teacherfirst_scorefiber_s1_f15": (0.0, 0.5),
    "teacherfirst_scorefiber_s1_f31": (0.0, 0.5),
}

DEFAULT_STAGE1_RUNG_PREFIX_EPOCHS: tuple[int, ...] = (2, 6)
DEFAULT_STAGE1_RUNG_PROMOTE_K: tuple[int, ...] = (3, 2)
DEFAULT_STAGE1_SCREEN_DOC_LIMITS: Mapping[int, int] = {
    128: 16,
    512: 32,
    1024: 64,
    2048: 128,
}
DEFAULT_STAGE2_EPOCHS_BY_COUNT: Mapping[int, int] = {
    128: 4,
    512: 6,
    1024: 8,
    2048: 12,
}
DEFAULT_STAGE2_SURVIVORS_BY_COUNT: Mapping[int, int] = {
    128: 1,
    512: 1,
    1024: 2,
    2048: 2,
}


@dataclass(frozen=True)
class _Stage1RungSpec:
    index: int
    total_epochs: int
    promote_k: int | None = None


@dataclass(frozen=True)
class _AsyncLaunchTask:
    task_type: str
    train_doc_count: int
    phase_key: str
    phase_root: str
    task_name: str
    job: Mapping[str, Any]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=str,
        default=f"outputs/tree_neural_teacher_first_scaling_push_{tfpush._timestamp()}",
    )
    parser.add_argument("--benchmark", type=str, default="recoverable_v4")
    parser.add_argument(
        "--train-doc-counts",
        nargs="*",
        type=int,
        default=(128, 512, 1024, 2048),
    )
    parser.add_argument(
        "--phase2-train-multiplier",
        type=float,
        default=1.0,
        help="Phase-2 train docs are round(multiplier * phase-1 train docs).",
    )
    parser.add_argument("--phase1-seeds", nargs="*", type=int, default=(0,))
    parser.add_argument("--phase2-seeds", nargs="*", type=int, default=(0,))
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--local-law-weight", dest="tree_local_law_weight", type=float, default=0.8)
    parser.add_argument("--tree-join-bit-weight", type=float, default=1.0)
    parser.add_argument("--stage1-epochs", type=int, default=12)
    parser.add_argument("--stage1-rung-epochs", nargs="*", type=int, default=())
    parser.add_argument("--stage1-rung-promote-k", nargs="*", type=int, default=())
    parser.add_argument("--stage1-screen-metric", type=str, default="val_root_mae")
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--stage2-epochs-by-count", nargs="*", type=str, default=())
    parser.add_argument(
        "--stage2-survivors-by-count",
        nargs="*",
        type=str,
        default=(),
    )
    parser.add_argument(
        "--async-promote-per-count",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--group-stage2-conditions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--tree-batch-pack-mode", type=str, default="")
    parser.add_argument("--batch-token-budget", type=int, default=0)
    parser.add_argument("--batch-node-budget", type=int, default=0)
    parser.add_argument(
        "--batch-autotune",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--eval-workers-per-mig", type=int, default=0)
    parser.add_argument("--tree-theorem-count-dim", type=int, default=8)
    parser.add_argument("--tree-theorem-first-dim", type=int, default=8)
    parser.add_argument("--tree-theorem-last-dim", type=int, default=8)
    parser.add_argument("--promote-top-k", type=int, default=2)
    parser.add_argument(
        "--mig-uuids",
        type=str,
        default="",
        help="Optional comma/space separated MIG UUID subset.",
    )
    parser.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser


def _variant_index() -> Mapping[str, Mapping[str, Any]]:
    return {str(variant["label"]): dict(variant) for variant in tfpush.SURROGATE_VARIANTS}


def _frontier_variants() -> tuple[dict[str, Any], ...]:
    by_label = _variant_index()
    variants: List[dict[str, Any]] = []
    for base_label, root_weights in FRONTIER_FAMILY_ROOT_WEIGHTS.items():
        base = dict(by_label[base_label])
        for root_weight in root_weights:
            variant = dict(base)
            if float(root_weight) > 0.0:
                variant["label"] = f"{base_label}_root{tfpush._weight_label(float(root_weight))}"
                variant["stage1_root_weight"] = float(root_weight)
                variant["stage1_checkpoint_metric"] = "val_root_mae"
            else:
                variant["stage1_root_weight"] = 0.0
                variant["stage1_checkpoint_metric"] = "val_root_mae"
            variants.append(variant)
    return tuple(variants)


def _mean_or_default(values: Sequence[float], *, default: float) -> float:
    if not values:
        return float(default)
    return float(sum(float(value) for value in values) / float(len(values)))


def _metric_or_default(value: Any, *, default: float) -> float:
    if value in {"", None}:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_count_value_pairs(
    entries: Sequence[str],
) -> Dict[int, int]:
    parsed: Dict[int, int] = {}
    for entry in entries:
        token = str(entry).strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"expected COUNT:VALUE entry, got {entry!r}")
        count_text, value_text = token.split(":", 1)
        parsed[int(count_text)] = int(value_text)
    return parsed


def _count_value_with_fallback(
    count: int,
    *,
    overrides: Mapping[int, int],
    defaults: Mapping[int, int],
    global_default: int,
) -> int:
    if int(count) in overrides:
        return int(overrides[int(count)])
    if int(count) in defaults:
        return int(defaults[int(count)])
    return int(global_default)


def _resolved_stage2_epochs_by_count(
    args: argparse.Namespace,
) -> Dict[int, int]:
    overrides = _parse_count_value_pairs(
        getattr(args, "stage2_epochs_by_count", ()) or (),
    )
    return {
        int(count): _count_value_with_fallback(
            int(count),
            overrides=overrides,
            defaults=DEFAULT_STAGE2_EPOCHS_BY_COUNT,
            global_default=int(args.stage2_epochs),
        )
        for count in [int(value) for value in args.train_doc_counts]
    }


def _resolved_stage2_survivors_by_count(
    args: argparse.Namespace,
) -> Dict[int, int]:
    overrides = _parse_count_value_pairs(
        getattr(args, "stage2_survivors_by_count", ()) or (),
    )
    return {
        int(count): max(
            1,
            _count_value_with_fallback(
                int(count),
                overrides=overrides,
                defaults=DEFAULT_STAGE2_SURVIVORS_BY_COUNT,
                global_default=int(args.promote_top_k),
            ),
        )
        for count in [int(value) for value in args.train_doc_counts]
    }


def _resolved_stage1_screen_doc_limit(train_doc_count: int) -> int:
    if int(train_doc_count) in DEFAULT_STAGE1_SCREEN_DOC_LIMITS:
        return int(DEFAULT_STAGE1_SCREEN_DOC_LIMITS[int(train_doc_count)])
    return int(tfpush._screen_doc_limit_for_train_docs(int(train_doc_count)))


def _resolved_stage1_rungs(
    args: argparse.Namespace,
    *,
    variant_count: int,
) -> tuple[_Stage1RungSpec, ...]:
    final_epoch = max(1, int(args.stage1_epochs))
    requested_epochs = tuple(
        int(value) for value in getattr(args, "stage1_rung_epochs", ()) if int(value) > 0
    )
    if requested_epochs:
        rung_epochs = requested_epochs
    else:
        prefix = tuple(
            int(value)
            for value in DEFAULT_STAGE1_RUNG_PREFIX_EPOCHS
            if int(value) < int(final_epoch)
        )
        rung_epochs = prefix + (int(final_epoch),)
    if any(
        int(right) <= int(left)
        for left, right in zip(rung_epochs, rung_epochs[1:])
    ):
        raise ValueError("stage1_rung_epochs must be strictly increasing")

    requested_promote_k = tuple(
        int(value)
        for value in getattr(args, "stage1_rung_promote_k", ())
        if int(value) > 0
    )
    if requested_promote_k:
        promote_k = requested_promote_k
    elif len(rung_epochs) <= 1:
        promote_k = tuple()
    elif len(rung_epochs) == 2:
        promote_k = (min(2, int(variant_count)),)
    elif len(rung_epochs) == 3:
        promote_k = tuple(
            min(int(value), int(variant_count))
            for value in DEFAULT_STAGE1_RUNG_PROMOTE_K
        )
    else:
        raise ValueError(
            "stage1_rung_promote_k must be provided when using more than 3 rung epochs"
        )
    if len(promote_k) != max(0, len(rung_epochs) - 1):
        raise ValueError(
            "stage1_rung_promote_k must have exactly len(stage1_rung_epochs) - 1 values"
        )

    rung_specs: List[_Stage1RungSpec] = []
    for idx, total_epochs in enumerate(rung_epochs, start=1):
        promote_value: int | None = None
        if idx <= len(promote_k):
            promote_value = max(1, min(int(promote_k[idx - 1]), int(variant_count)))
        rung_specs.append(
            _Stage1RungSpec(
                index=int(idx),
                total_epochs=int(total_epochs),
                promote_k=promote_value,
            )
        )
    return tuple(rung_specs)


def _count_args(args: argparse.Namespace, *, train_doc_count: int) -> argparse.Namespace:
    phase2_train_docs = max(
        1,
        int(round(float(args.phase2_train_multiplier) * float(train_doc_count))),
    )
    resolved_stage2_epochs = _resolved_stage2_epochs_by_count(args)
    return argparse.Namespace(
        benchmark=str(args.benchmark),
        output_root=str(args.output_root),
        phase1_train_docs=int(train_doc_count),
        phase2_train_docs=int(phase2_train_docs),
        phase1_seeds=tuple(int(seed) for seed in args.phase1_seeds),
        phase2_seeds=tuple(int(seed) for seed in args.phase2_seeds),
        state_dim=int(args.state_dim),
        hidden_dim=int(args.hidden_dim),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        tree_local_law_weight=float(args.tree_local_law_weight),
        tree_join_bit_weight=float(args.tree_join_bit_weight),
        stage1_epochs=int(args.stage1_epochs),
        stage1_screen_metric=str(args.stage1_screen_metric),
        stage2_epochs=int(
            resolved_stage2_epochs.get(int(train_doc_count), int(args.stage2_epochs))
        ),
        tree_theorem_count_dim=int(args.tree_theorem_count_dim),
        tree_theorem_first_dim=int(args.tree_theorem_first_dim),
        tree_theorem_last_dim=int(args.tree_theorem_last_dim),
        promote_top_k=int(args.promote_top_k),
        tree_stage1_eval_mode="end_only",
        tree_stage1_screen_doc_limit=int(
            _resolved_stage1_screen_doc_limit(int(train_doc_count))
        ),
        tree_stage1_final_exact_doc_limit=0,
        tree_batch_pack_mode=resolved_tree_batch_pack_mode(
            benchmark=str(getattr(args, "benchmark", "")),
            raw_value=getattr(args, "tree_batch_pack_mode", ""),
        ),
        batch_token_budget=int(getattr(args, "batch_token_budget", 0)),
        batch_node_budget=int(getattr(args, "batch_node_budget", 0)),
        batch_autotune=bool(getattr(args, "batch_autotune", True)),
        eval_workers_per_mig=int(getattr(args, "eval_workers_per_mig", 0)),
        group_stage2_conditions=bool(args.group_stage2_conditions),
        async_promote_per_count=bool(args.async_promote_per_count),
        mig_uuids=str(args.mig_uuids),
        use_cuda=bool(args.use_cuda),
        torch_threads=int(args.torch_threads),
        root_search_labels=tuple(),
        stage1_root_weight_grid=tuple(),
        surrogate_labels=tuple(),
    )


def _runs_for_train_doc_count(
    runs: Sequence[Mapping[str, Any]],
    *,
    train_doc_count: int,
) -> List[Dict[str, Any]]:
    target = int(train_doc_count)
    filtered: List[Dict[str, Any]] = []
    for run in runs:
        if int(run.get("train_doc_count", -1)) != target:
            continue
        filtered.append(dict(run))
    return filtered


def _stage1_screen_metric_value(
    run: Mapping[str, Any],
    *,
    metric_name: str,
    default: float = float("inf"),
) -> float:
    metric = str(metric_name or "").strip()
    value = _metric_or_default(run.get(metric), default=float("nan"))
    if value == value:
        return float(value)
    if metric == str(run.get("selection_metric_name", "")).strip():
        value = _metric_or_default(run.get("best_val_mae"), default=float("nan"))
        if value == value:
            return float(value)
    direct_val = (
        (
            (run.get("exact_sketch_diagnostics") or {}).get("direct_selection_metrics")
            or {}
        ).get("val")
        or {}
    )
    value = _metric_or_default(direct_val.get(metric), default=float("nan"))
    if value == value:
        return float(value)
    return float(default)


def _stage1_rung_run_rank_key(
    run: Mapping[str, Any],
    *,
    metric_name: str,
) -> tuple[float, float, float, float, float, str, int]:
    return (
        float(_stage1_screen_metric_value(run, metric_name=metric_name, default=float("inf"))),
        *tfpush._summary_rank_key(
            {
                "mean_teacher_first_total_bound": tfpush._direct_metric(
                    run,
                    "teacher_first_total_bound",
                    default=float("inf"),
                ),
                "mean_stage1_substitution_cost": tfpush._direct_metric(
                    run,
                    "stage1_substitution_cost",
                    default=float("inf"),
                ),
                "mean_test_root_mae": tfpush._direct_metric(
                    run,
                    "test_root_mae",
                    default=float("inf"),
                ),
                "mean_stage2_transport_budget": tfpush._direct_metric(
                    run,
                    "stage2_transport_budget",
                    default=float("inf"),
                ),
            }
        ),
        str(run.get("config_label", "")),
        int(run.get("seed", 0)),
    )


def _stage1_candidate_summary_rank_key(
    row: Mapping[str, Any],
) -> tuple[float, float, float, float, float, str]:
    return (
        float(row.get("mean_screen_metric", float("inf"))),
        *tfpush._summary_rank_key(row),
        str(row.get("candidate_label", "")),
    )


def _aggregate_stage1_rung_candidate_summary(
    runs: Sequence[Mapping[str, Any]],
    *,
    screen_metric_name: str,
) -> List[Dict[str, Any]]:
    by_candidate_seed: Dict[tuple[str, int], Dict[str, Any]] = {}
    for run in runs:
        label = str(run.get("config_label", "")).strip()
        if not label:
            continue
        seed = int(run.get("seed", 0))
        key = (label, seed)
        current = by_candidate_seed.get(key)
        if current is None or _stage1_rung_run_rank_key(
            run,
            metric_name=screen_metric_name,
        ) < _stage1_rung_run_rank_key(
            current,
            metric_name=screen_metric_name,
        ):
            by_candidate_seed[key] = dict(run)

    by_candidate: Dict[str, List[Dict[str, Any]]] = {}
    for (candidate_label, _seed), run in by_candidate_seed.items():
        by_candidate.setdefault(candidate_label, []).append(dict(run))

    summaries: List[Dict[str, Any]] = []
    for candidate_label, candidate_runs in sorted(by_candidate.items()):
        ordered_runs = sorted(candidate_runs, key=lambda run: int(run.get("seed", 0)))
        summary = {
            "candidate_label": str(candidate_label),
            "n_stage1_seed_wins": int(len(ordered_runs)),
            "screen_metric_name": str(screen_metric_name),
            "mean_screen_metric": _mean_or_default(
                [
                    _stage1_screen_metric_value(
                        run,
                        metric_name=screen_metric_name,
                        default=float("inf"),
                    )
                    for run in ordered_runs
                ],
                default=float("inf"),
            ),
            "mean_teacher_first_total_bound": _mean_or_default(
                [
                    tfpush._direct_metric(
                        run,
                        "teacher_first_total_bound",
                        default=float("inf"),
                    )
                    for run in ordered_runs
                ],
                default=float("inf"),
            ),
            "mean_stage1_substitution_cost": _mean_or_default(
                [
                    tfpush._direct_metric(
                        run,
                        "stage1_substitution_cost",
                        default=float("inf"),
                    )
                    for run in ordered_runs
                ],
                default=float("inf"),
            ),
            "mean_test_root_mae": _mean_or_default(
                [
                    tfpush._direct_metric(
                        run,
                        "test_root_mae",
                        default=float("inf"),
                    )
                    for run in ordered_runs
                ],
                default=float("inf"),
            ),
            "mean_stage2_transport_budget": _mean_or_default(
                [
                    tfpush._direct_metric(
                        run,
                        "stage2_transport_budget",
                        default=float("inf"),
                    )
                    for run in ordered_runs
                ],
                default=float("inf"),
            ),
            "mean_stage1_root_weight": _mean_or_default(
                [
                    _metric_or_default(
                        run.get("tree_stage1_root_weight"),
                        default=0.0,
                    )
                    for run in ordered_runs
                ],
                default=0.0,
            ),
            "stage1_artifacts_by_seed": {
                str(int(run.get("seed", 0))): str(run.get("tree_stage1_artifact_dir", ""))
                for run in ordered_runs
                if str(run.get("tree_stage1_artifact_dir", "")).strip()
            },
        }
        summaries.append(summary)
    summaries.sort(key=_stage1_candidate_summary_rank_key)
    return summaries


def _make_rung_stage1_config(
    config: RunConfigSpec,
    *,
    total_epochs: int,
    screen_metric_name: str,
) -> RunConfigSpec:
    return replace(
        config,
        n_epochs=int(total_epochs),
        tree_stage1_epochs=int(total_epochs),
        tree_stage2_epochs=0,
        tree_stage1_checkpoint_metric=str(screen_metric_name),
    )


def _build_stage1_rung_jobs(
    *,
    args: argparse.Namespace,
    train_doc_counts: Sequence[int],
    count_args_by_train: Mapping[int, argparse.Namespace],
    stage1_configs_by_train: Mapping[int, Mapping[str, RunConfigSpec]],
    active_labels_by_count: Mapping[int, Sequence[str]],
    rung: _Stage1RungSpec,
    screen_metric_name: str,
) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    for train_doc_count in [int(value) for value in train_doc_counts]:
        count_args = count_args_by_train[int(train_doc_count)]
        stage1_configs = stage1_configs_by_train[int(train_doc_count)]
        active_labels = [str(label) for label in active_labels_by_count[int(train_doc_count)]]
        configs_by_train = [
            (
                int(count_args.phase1_train_docs),
                _make_rung_stage1_config(
                    stage1_configs[str(label)],
                    total_epochs=int(rung.total_epochs),
                    screen_metric_name=screen_metric_name,
                ),
            )
            for label in active_labels
            if str(label) in stage1_configs
        ]
        jobs.extend(
            tfpush._build_phase_jobs(
                args=count_args,
                configs_by_train=configs_by_train,
                seeds=tuple(int(seed) for seed in count_args.phase1_seeds),
                tuning_stage="stage1_surrogate",
                study_axis="stage1_surrogate",
            )
        )
    return jobs


def _build_stage2_jobs_for_counts(
    *,
    train_doc_counts: Sequence[int],
    count_args_by_train: Mapping[int, argparse.Namespace],
    stage1_configs_by_train: Mapping[int, Mapping[str, RunConfigSpec]],
    active_labels_by_count: Mapping[int, Sequence[str]],
    final_stage1_runs: Sequence[Mapping[str, Any]],
    stage2_survivors_by_count: Mapping[int, int],
) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    for train_doc_count in [int(value) for value in train_doc_counts]:
        count_args = count_args_by_train[int(train_doc_count)]
        survivor_limit = max(
            1,
            int(stage2_survivors_by_count.get(int(train_doc_count), int(count_args.promote_top_k))),
        )
        active_labels = {
            str(label)
            for label in list(active_labels_by_count[int(train_doc_count)])[:survivor_limit]
        }
        base_configs = {
            str(label): config
            for label, config in stage1_configs_by_train[int(train_doc_count)].items()
            if str(label) in active_labels
        }
        stage1_runs_for_count = [
            dict(run)
            for run in _runs_for_train_doc_count(
                final_stage1_runs,
                train_doc_count=int(count_args.phase1_train_docs),
            )
            if str(run.get("config_label", "")).strip() in base_configs
        ]
        jobs.extend(
            tfpush._build_stage2_jobs(
                args=count_args,
                stage1_runs=stage1_runs_for_count,
                base_configs=base_configs,
                stage2_epochs=int(count_args.stage2_epochs),
            )
        )
    return jobs


def _count_root(output_root: Path, train_doc_count: int) -> Path:
    return output_root / f"train_{int(train_doc_count)}"


def _stage1_phase_root(
    output_root: Path,
    *,
    train_doc_count: int,
    rung_index: int,
) -> Path:
    return _count_root(output_root, int(train_doc_count)) / f"phase1_rung{int(rung_index)}"


def _stage2_phase_root(
    output_root: Path,
    *,
    train_doc_count: int,
) -> Path:
    return _count_root(output_root, int(train_doc_count)) / "phase2"


def _task_priority(task: _AsyncLaunchTask) -> tuple[int, int, str]:
    if str(task.task_type) == "stage2_grouped":
        return (0, int(task.train_doc_count), str(task.task_name))
    rung_suffix = str(task.phase_key).rsplit(":", 1)[-1]
    try:
        rung_index = int(rung_suffix)
    except ValueError:
        rung_index = 0
    return (10 - int(rung_index), int(task.train_doc_count), str(task.task_name))


def _ensure_manifest(
    phase_root: Path,
    payload: Mapping[str, Any],
) -> None:
    phase_root.mkdir(parents=True, exist_ok=True)
    manifest_path = phase_root / "mig_job_manifest.json"
    if manifest_path.exists():
        return
    manifest_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_global_scaling_summary(
    output_root: Path,
    *,
    args: argparse.Namespace,
    variants: Sequence[Mapping[str, Any]],
    rung_specs: Sequence[_Stage1RungSpec],
    mig_uuids: Sequence[str],
    scaling_rows: Sequence[Mapping[str, Any]],
    phase1_output_roots_by_count: Mapping[int, Sequence[str]],
    global_phase2_output_roots: Mapping[int, str],
) -> None:
    train_doc_counts = [int(value) for value in args.train_doc_counts]
    payload = {
        "benchmark": str(args.benchmark),
        "train_doc_counts": train_doc_counts,
        "phase2_train_multiplier": float(args.phase2_train_multiplier),
        "phase1_seeds": [int(seed) for seed in args.phase1_seeds],
        "phase2_seeds": [int(seed) for seed in args.phase2_seeds],
        "stage1_rung_epochs": [int(rung.total_epochs) for rung in rung_specs],
        "stage1_rung_promote_k": [
            int(rung.promote_k)
            for rung in rung_specs
            if rung.promote_k is not None
        ],
        "stage1_screen_metric": str(args.stage1_screen_metric),
        "mig_uuids": [str(uuid) for uuid in mig_uuids],
        "frontier_variants": [dict(variant) for variant in variants],
        "phase1_output_roots_by_count": {
            str(count): list(paths)
            for count, paths in phase1_output_roots_by_count.items()
        },
        "phase2_output_roots_by_count": {
            str(count): str(path)
            for count, path in global_phase2_output_roots.items()
        },
        "scaling_rows": [dict(row) for row in scaling_rows],
    }
    (output_root / "teacher_first_scaling_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "teacher_first_scaling_summary.md").write_text(
        _render_scaling_markdown(scaling_rows, mig_uuids=mig_uuids),
        encoding="utf-8",
    )


def _run_async_scaling(
    *,
    args: argparse.Namespace,
    output_root: Path,
    variants: Sequence[Mapping[str, Any]],
    rung_specs: Sequence[_Stage1RungSpec],
    screen_metric_name: str,
    mig_uuids: Sequence[str],
) -> Dict[str, Any]:
    if not bool(args.group_stage2_conditions):
        raise ValueError(
            "async_promote_per_count currently requires --group-stage2-conditions"
        )
    train_doc_counts = [int(value) for value in args.train_doc_counts]
    count_args_by_train = {
        int(train_doc_count): _count_args(args, train_doc_count=int(train_doc_count))
        for train_doc_count in train_doc_counts
    }
    stage1_configs_by_train = {
        int(train_doc_count): {
            str(variant["label"]): tfpush._make_stage1_config(
                count_args_by_train[int(train_doc_count)],
                train_doc_count=int(count_args_by_train[int(train_doc_count)].phase1_train_docs),
                variant=variant,
            )
            for variant in variants
        }
        for train_doc_count in train_doc_counts
    }
    stage2_survivors_by_count = _resolved_stage2_survivors_by_count(args)
    active_labels_by_count: Dict[int, List[str]] = {
        int(train_doc_count): [str(variant["label"]) for variant in variants]
        for train_doc_count in train_doc_counts
    }
    stage1_rung_history_by_count: Dict[int, List[Dict[str, Any]]] = {
        int(train_doc_count): []
        for train_doc_count in train_doc_counts
    }
    phase1_output_roots_by_count: Dict[int, List[str]] = {
        int(train_doc_count): []
        for train_doc_count in train_doc_counts
    }
    phase2_output_roots_by_count: Dict[int, str] = {}
    scaling_rows_by_count: Dict[int, Dict[str, Any]] = {}
    phase_state: Dict[str, Dict[str, Any]] = {}
    pending_tasks: List[_AsyncLaunchTask] = []
    active: List[Dict[str, Any]] = []
    base_tokens = list(mig_uuids if bool(args.use_cuda) else ["cpu0"])
    available_tokens = list(base_tokens)
    failed: List[Dict[str, Any]] = []
    stop_requested = False
    force_terminate_requested = False
    desired_eval_workers_per_mig = max(
        1,
        int(getattr(args, "eval_workers_per_mig", 0) or 0),
    )

    def _physical_token(token: str) -> str:
        return str(token).split("#", 1)[0]

    def _maybe_expand_eval_aliases() -> None:
        if desired_eval_workers_per_mig <= 1:
            return
        if any(str(task.task_type) == "stage1_worker" for task in pending_tasks):
            return
        if any(str(entry["task"].task_type) == "stage1_worker" for entry in active):
            return
        current_alias_counts: Dict[str, int] = {}
        for token in list(available_tokens):
            current_alias_counts[_physical_token(str(token))] = (
                current_alias_counts.get(_physical_token(str(token)), 0) + 1
            )
        for entry in active:
            token = str(entry["token"])
            current_alias_counts[_physical_token(token)] = (
                current_alias_counts.get(_physical_token(token), 0) + 1
            )
        for base_token in base_tokens:
            existing = int(current_alias_counts.get(str(base_token), 0))
            while existing < desired_eval_workers_per_mig:
                alias = (
                    str(base_token)
                    if existing == 0
                    else f"{str(base_token)}#eval{int(existing + 1)}"
                )
                if alias not in available_tokens:
                    available_tokens.append(alias)
                existing += 1

    def _request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested, force_terminate_requested
        if not stop_requested:
            stop_requested = True
            print(
                f"received signal {int(signum)}; pausing async scaling queue",
                flush=True,
            )
            return
        if force_terminate_requested:
            return
        force_terminate_requested = True
        for entry in active:
            proc = entry.get("proc")
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    continue

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    def _record_task_done(phase_key: str, task_name: str) -> bool:
        phase_entry = phase_state[str(phase_key)]
        phase_entry["done_tasks"].add(str(task_name))
        return phase_entry["done_tasks"] >= phase_entry["expected_tasks"]

    def _enqueue_stage1_rung(
        *,
        train_doc_count: int,
        rung: _Stage1RungSpec,
        labels: Sequence[str],
    ) -> None:
        count_args = count_args_by_train[int(train_doc_count)]
        phase_root = _stage1_phase_root(
            output_root,
            train_doc_count=int(train_doc_count),
            rung_index=int(rung.index),
        )
        jobs = _build_stage1_rung_jobs(
            args=args,
            train_doc_counts=(int(train_doc_count),),
            count_args_by_train=count_args_by_train,
            stage1_configs_by_train=stage1_configs_by_train,
            active_labels_by_count={int(train_doc_count): list(labels)},
            rung=rung,
            screen_metric_name=screen_metric_name,
        )
        _ensure_manifest(
            phase_root,
            {
                "mode": "teacher_first_stage1_surrogate_rung_async",
                "benchmark": str(args.benchmark),
                "train_doc_count": int(train_doc_count),
                "rung_index": int(rung.index),
                "stage1_total_epochs": int(rung.total_epochs),
                "screen_metric_name": str(screen_metric_name),
                "active_labels": [str(label) for label in labels],
                "jobs": [asdict(job) for job in jobs],
            },
        )
        completed_keys = load_completed_run_keys(phase_root)
        phase_key = f"stage1:{int(train_doc_count)}:{int(rung.index)}"
        phase_entry = {
            "kind": "stage1",
            "train_doc_count": int(train_doc_count),
            "rung_index": int(rung.index),
            "phase_root": str(phase_root),
            "expected_tasks": {str(job.job_name) for job in jobs},
            "done_tasks": set(),
            "rung": rung,
            "labels": [str(label) for label in labels],
        }
        for job in jobs:
            required_keys = job_completion_keys(job)
            if required_keys and required_keys.issubset(completed_keys):
                phase_entry["done_tasks"].add(str(job.job_name))
                continue
            pending_tasks.append(
                _AsyncLaunchTask(
                    task_type="stage1_worker",
                    train_doc_count=int(train_doc_count),
                    phase_key=str(phase_key),
                    phase_root=str(phase_root),
                    task_name=str(job.job_name),
                    job=asdict(job),
                )
            )
        phase_state[str(phase_key)] = phase_entry
        phase1_output_roots_by_count[int(train_doc_count)].append(str(phase_root))

    def _enqueue_stage2_for_count(
        *,
        train_doc_count: int,
        survivor_labels: Sequence[str],
    ) -> None:
        count_args = count_args_by_train[int(train_doc_count)]
        phase_root = _stage2_phase_root(
            output_root,
            train_doc_count=int(train_doc_count),
        )
        flat_jobs = _build_stage2_jobs_for_counts(
            train_doc_counts=(int(train_doc_count),),
            count_args_by_train=count_args_by_train,
            stage1_configs_by_train=stage1_configs_by_train,
            active_labels_by_count={int(train_doc_count): list(survivor_labels)},
            final_stage1_runs=list(
                (phase_state[f"stage1:{int(train_doc_count)}:{int(rung_specs[-1].index)}"].get("runs") or [])
            ),
            stage2_survivors_by_count={int(train_doc_count): int(len(list(survivor_labels)))},
        )
        grouped_jobs = tfpush._build_grouped_stage2_jobs(flat_jobs)
        _ensure_manifest(
            phase_root,
            {
                "mode": "teacher_first_stage2_judge_async",
                "benchmark": str(args.benchmark),
                "train_doc_count": int(train_doc_count),
                "survivor_labels": [str(label) for label in survivor_labels],
                "grouped_stage2_conditions": True,
                "jobs": grouped_jobs,
            },
        )
        completed_keys = load_completed_run_keys(phase_root)
        phase_key = f"stage2:{int(train_doc_count)}"
        phase_entry = {
            "kind": "stage2",
            "train_doc_count": int(train_doc_count),
            "phase_root": str(phase_root),
            "expected_tasks": {str(item["job_name"]) for item in grouped_jobs},
            "done_tasks": set(),
        }
        for grouped_job in grouped_jobs:
            required_keys = set()
            for job_mapping in list(grouped_job.get("jobs", ()) or ()):
                required_keys.update(
                    job_completion_keys(tfpush._job_from_mapping(job_mapping))
                )
            if required_keys and required_keys.issubset(completed_keys):
                phase_entry["done_tasks"].add(str(grouped_job["job_name"]))
                continue
            pending_tasks.append(
                _AsyncLaunchTask(
                    task_type="stage2_grouped",
                    train_doc_count=int(train_doc_count),
                    phase_key=str(phase_key),
                    phase_root=str(phase_root),
                    task_name=str(grouped_job["job_name"]),
                    job=dict(grouped_job),
                )
            )
        phase_state[str(phase_key)] = phase_entry
        phase2_output_roots_by_count[int(train_doc_count)] = str(phase_root)

    def _finalize_stage1_phase(phase_key: str) -> None:
        phase_entry = phase_state[str(phase_key)]
        phase_root = Path(str(phase_entry["phase_root"]))
        payload = write_summary_outputs(phase_root)
        train_doc_count = int(phase_entry["train_doc_count"])
        count_args = count_args_by_train[int(train_doc_count)]
        runs = _runs_for_train_doc_count(
            payload.get("runs", ()),
            train_doc_count=int(count_args.phase1_train_docs),
        )
        phase_entry["runs"] = [dict(run) for run in runs]
        candidate_summary = _aggregate_stage1_rung_candidate_summary(
            runs,
            screen_metric_name=screen_metric_name,
        )
        if not candidate_summary:
            raise RuntimeError(
                f"no stage-1 runs completed for train_doc_count={int(train_doc_count)} "
                f"rung={int(phase_entry['rung_index'])}"
            )
        promoted_candidates = [
            str(row.get("candidate_label", ""))
            for row in candidate_summary
        ]
        rung: _Stage1RungSpec = phase_entry["rung"]
        if rung.promote_k is not None:
            promoted_candidates = promoted_candidates[
                : max(1, min(int(rung.promote_k), len(promoted_candidates)))
            ]
        stage1_rung_history_by_count[int(train_doc_count)].append(
            {
                "rung_index": int(phase_entry["rung_index"]),
                "total_epochs": int(rung.total_epochs),
                "screen_metric_name": str(screen_metric_name),
                "active_candidates": list(phase_entry["labels"]),
                "promoted_candidates": [str(label) for label in promoted_candidates],
                "phase_output_root": str(phase_root),
                "candidate_summary": [dict(row) for row in candidate_summary],
            }
        )
        active_labels_by_count[int(train_doc_count)] = list(promoted_candidates)
        _write_stage1_rung_summary(
            _count_root(output_root, int(train_doc_count)),
            args=count_args,
            stage1_screen_metric=screen_metric_name,
            rung_specs=rung_specs,
            rung_history=stage1_rung_history_by_count[int(train_doc_count)],
            final_survivors=list(promoted_candidates),
        )
        next_rung_index = int(phase_entry["rung_index"]) + 1
        if next_rung_index <= int(len(rung_specs)):
            _enqueue_stage1_rung(
                train_doc_count=int(train_doc_count),
                rung=rung_specs[int(next_rung_index) - 1],
                labels=promoted_candidates,
            )
        else:
            final_survivors = list(promoted_candidates)[
                : max(1, int(stage2_survivors_by_count[int(train_doc_count)]))
            ]
            _write_stage1_rung_summary(
                _count_root(output_root, int(train_doc_count)),
                args=count_args,
                stage1_screen_metric=screen_metric_name,
                rung_specs=rung_specs,
                rung_history=stage1_rung_history_by_count[int(train_doc_count)],
                final_survivors=final_survivors,
            )
            _enqueue_stage2_for_count(
                train_doc_count=int(train_doc_count),
                survivor_labels=final_survivors,
            )

    def _finalize_stage2_phase(phase_key: str) -> None:
        phase_entry = phase_state[str(phase_key)]
        phase_root = Path(str(phase_entry["phase_root"]))
        payload = write_summary_outputs(phase_root)
        train_doc_count = int(phase_entry["train_doc_count"])
        count_args = count_args_by_train[int(train_doc_count)]
        phase2_runs_for_count = _runs_for_train_doc_count(
            payload.get("runs", ()),
            train_doc_count=int(count_args.phase2_train_docs),
        )
        candidate_summary = tfpush._aggregate_candidate_summary(phase2_runs_for_count)
        pareto_frontier = [
            str(row.get("candidate_label", ""))
            for row in candidate_summary
            if bool(row.get("on_pareto_frontier", False))
        ]
        promoted = [
            str(row.get("candidate_label", ""))
            for row in candidate_summary[: max(0, int(count_args.promote_top_k))]
        ]
        final_survivors = list(active_labels_by_count[int(train_doc_count)])[
            : max(1, int(stage2_survivors_by_count[int(train_doc_count)]))
        ]
        count_root = _count_root(output_root, int(train_doc_count))
        _write_count_summary(
            count_root,
            args=count_args,
            candidate_summary=candidate_summary,
            promoted=promoted,
            pareto_frontier=pareto_frontier,
            variants=variants,
            phase1_runs=int(
                sum(
                    len(
                        phase_state.get(f"stage1:{int(train_doc_count)}:{int(rung.index)}", {}).get(
                            "runs",
                            (),
                        )
                    )
                    for rung in rung_specs
                )
            ),
            phase2_runs=int(len(phase2_runs_for_count)),
            stage1_rungs=stage1_rung_history_by_count[int(train_doc_count)],
            stage1_final_survivors=final_survivors,
        )
        best = dict(candidate_summary[0]) if candidate_summary else {}
        scaling_rows_by_count[int(train_doc_count)] = {
            "train_doc_count": int(train_doc_count),
            "phase2_train_docs": int(count_args.phase2_train_docs),
            "output_root": str(count_root),
            "best_candidate": best,
            "pareto_frontier": pareto_frontier,
            "promoted_candidates": promoted,
            "phase1_runs": int(
                sum(
                    len(
                        phase_state.get(f"stage1:{int(train_doc_count)}:{int(rung.index)}", {}).get(
                            "runs",
                            (),
                        )
                    )
                    for rung in rung_specs
                )
            ),
            "phase2_runs": int(len(phase2_runs_for_count)),
            "stage1_final_survivors": list(final_survivors),
        }
        _write_global_scaling_summary(
            output_root,
            args=args,
            variants=variants,
            rung_specs=rung_specs,
            mig_uuids=mig_uuids,
            scaling_rows=[
                scaling_rows_by_count[count]
                for count in sorted(scaling_rows_by_count)
            ],
            phase1_output_roots_by_count=phase1_output_roots_by_count,
            global_phase2_output_roots=phase2_output_roots_by_count,
        )

    def _maybe_finalize_phase(phase_key: str) -> None:
        while True:
            phase_entry = phase_state.get(str(phase_key))
            if phase_entry is None or phase_entry.get("finalized", False):
                return
            if phase_entry["done_tasks"] < phase_entry["expected_tasks"]:
                return
            phase_entry["finalized"] = True
            if phase_entry["kind"] == "stage1":
                _finalize_stage1_phase(str(phase_key))
            else:
                _finalize_stage2_phase(str(phase_key))
            return

    for train_doc_count in train_doc_counts:
        _enqueue_stage1_rung(
            train_doc_count=int(train_doc_count),
            rung=rung_specs[0],
            labels=active_labels_by_count[int(train_doc_count)],
        )

    for phase_key in list(phase_state):
        _maybe_finalize_phase(str(phase_key))

    while pending_tasks or active:
        pending_tasks.sort(key=_task_priority)
        _maybe_expand_eval_aliases()
        while pending_tasks and available_tokens and not stop_requested:
            token = available_tokens.pop(0)
            task = pending_tasks.pop(0)
            phase_root = Path(str(task.phase_root))
            job_root = phase_root / "jobs"
            job_root.mkdir(parents=True, exist_ok=True)
            log_path = job_root / f"{job_output_dir_name(str(task.task_name))}.log"
            log_fh = open(log_path, "w", encoding="utf-8")
            if str(task.task_type) == "stage1_worker":
                job = tfpush._job_from_mapping(task.job)
                job_output_dir = job_root / job_output_dir_name(job.job_name)
                job_output_dir.mkdir(parents=True, exist_ok=True)
                cmd = worker_command_for_job(
                    job,
                    output_dir=job_output_dir,
                    torch_threads=int(args.torch_threads),
                    use_cuda=bool(args.use_cuda),
                )
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=log_fh,
                    cwd=str(REPO_ROOT),
                    env=worker_env_for_token(
                        _physical_token(str(token)),
                        use_cuda=bool(args.use_cuda),
                    ),
                    text=True,
                )
            else:
                job_output_dir = job_root / job_output_dir_name(str(task.task_name))
                job_output_dir.mkdir(parents=True, exist_ok=True)
                manifest_path = job_output_dir / "group_manifest.json"
                manifest_path.write_text(
                    json.dumps(dict(task.job), indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                cmd = [
                    sys.executable,
                    str((REPO_ROOT / "scripts/run_tree_neural_teacher_first_push.py").resolve()),
                    "--grouped-stage2-worker-manifest",
                    str(manifest_path),
                    "--grouped-stage2-worker-output-dir",
                    str(job_output_dir),
                    "--grouped-stage2-worker-job-name",
                    str(task.task_name),
                    "--torch-threads",
                    str(int(args.torch_threads)),
                ]
                cmd.append("--use-cuda" if bool(args.use_cuda) else "--no-use-cuda")
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    cwd=str(REPO_ROOT),
                    env=worker_env_for_token(
                        _physical_token(str(token)),
                        use_cuda=bool(args.use_cuda),
                    ),
                    text=True,
                )
            active.append(
                {
                    "task": task,
                    "proc": proc,
                    "log_fh": log_fh,
                    "log_path": str(log_path),
                    "token": str(token),
                    "job_output_dir": str(job_output_dir),
                    "summary_ready_at": None,
                }
            )
            print(
                f"launched {task.task_name} on {str(token)[:18]} pid={proc.pid}",
                flush=True,
            )

        if not active:
            break
        time.sleep(1.0)
        still_active: List[Dict[str, Any]] = []
        for entry in active:
            proc = entry["proc"]
            task: _AsyncLaunchTask = entry["task"]
            grouped_summary: Dict[str, Any] | None = None
            if str(task.task_type) == "stage2_grouped":
                grouped_summary = tfpush._load_grouped_stage2_summary(
                    Path(str(entry["job_output_dir"]))
                )
                if grouped_summary is not None and entry.get("summary_ready_at") is None:
                    entry["summary_ready_at"] = time.monotonic()
            if proc.poll() is None:
                if str(task.task_type) == "stage2_grouped":
                    summary_ready_at = entry.get("summary_ready_at")
                    if (
                        grouped_summary is not None
                        and summary_ready_at is not None
                        and (
                            time.monotonic() - float(summary_ready_at)
                            >= float(tfpush.GROUPED_STAGE2_COMPLETION_GRACE_S)
                        )
                    ):
                        try:
                            proc.terminate()
                        except ProcessLookupError:
                            pass
                        try:
                            proc.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            try:
                                proc.kill()
                            except ProcessLookupError:
                                pass
                            try:
                                proc.wait(timeout=2.0)
                            except subprocess.TimeoutExpired:
                                pass
                    else:
                        still_active.append(entry)
                        continue
                else:
                    still_active.append(entry)
                    continue
            entry["log_fh"].close()
            available_tokens.append(str(entry["token"]))
            returncode = int(proc.returncode) if proc.returncode is not None else 0
            if int(returncode) != 0:
                if str(task.task_type) != "stage2_grouped" or grouped_summary is None:
                    failed.append(
                        {
                            "task_name": str(task.task_name),
                            "task_type": str(task.task_type),
                            "returncode": int(returncode),
                            "log_path": str(entry["log_path"]),
                        }
                    )
                    continue
            if str(task.task_type) == "stage1_worker":
                stdout_text = proc.stdout.read() if proc.stdout is not None else ""
                try:
                    result = json.loads(stdout_text.strip().splitlines()[-1])
                except Exception:
                    result = {}
                print(
                    "completed "
                    f"{task.task_name} "
                    f"root_mae={float(result.get('test_root_mae', float('nan'))):.6g}",
                    flush=True,
                )
            else:
                print(
                    "completed "
                    f"{task.task_name} "
                    f"summary={tfpush._grouped_stage2_summary_path(Path(str(entry['job_output_dir'])))}",
                    flush=True,
                )
            phase_complete = _record_task_done(str(task.phase_key), str(task.task_name))
            if phase_complete:
                _maybe_finalize_phase(str(task.phase_key))
        active = still_active
        available_tokens.sort()

    if failed:
        raise RuntimeError(
            "async scaling failures: "
            + ", ".join(str(item["task_name"]) for item in failed)
        )

    scaling_rows = [
        scaling_rows_by_count[count]
        for count in sorted(scaling_rows_by_count)
    ]
    _write_global_scaling_summary(
        output_root,
        args=args,
        variants=variants,
        rung_specs=rung_specs,
        mig_uuids=mig_uuids,
        scaling_rows=scaling_rows,
        phase1_output_roots_by_count=phase1_output_roots_by_count,
        global_phase2_output_roots=phase2_output_roots_by_count,
    )
    return {
        "scaling_rows": scaling_rows,
        "phase1_output_roots_by_count": phase1_output_roots_by_count,
        "phase2_output_roots_by_count": phase2_output_roots_by_count,
    }


def _render_stage1_rung_markdown(
    rung_history: Sequence[Mapping[str, Any]],
    *,
    final_survivors: Sequence[str],
    screen_metric_name: str,
) -> str:
    lines = [
        "# Stage-1 Rung Summary",
        "",
        f"- screen_metric: `{str(screen_metric_name)}`",
        f"- final_survivors: `{', '.join(str(label) for label in final_survivors)}`",
        "",
    ]
    for entry in rung_history:
        lines.extend(
            [
                f"## Rung {int(entry.get('rung_index', 0))} / `{int(entry.get('total_epochs', 0))}` epochs",
                f"- active_candidates: `{', '.join(str(label) for label in entry.get('active_candidates', []))}`",
                f"- promoted_candidates: `{', '.join(str(label) for label in entry.get('promoted_candidates', []))}`",
                "",
            ]
        )
        for row in entry.get("candidate_summary", []):
            lines.extend(
                [
                    f"- `{str(row.get('candidate_label', ''))}` "
                    f"screen=`{float(row.get('mean_screen_metric', float('nan'))):.6g}` "
                    f"bound=`{float(row.get('mean_teacher_first_total_bound', float('nan'))):.6g}`",
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_stage1_rung_summary(
    count_root: Path,
    *,
    args: argparse.Namespace,
    stage1_screen_metric: str,
    rung_specs: Sequence[_Stage1RungSpec],
    rung_history: Sequence[Mapping[str, Any]],
    final_survivors: Sequence[str],
) -> None:
    payload = {
        "benchmark": str(args.benchmark),
        "phase1_train_docs": int(args.phase1_train_docs),
        "phase2_train_docs": int(args.phase2_train_docs),
        "stage1_screen_metric": str(stage1_screen_metric),
        "stage1_rung_epochs": [int(rung.total_epochs) for rung in rung_specs],
        "stage1_rung_promote_k": [
            int(rung.promote_k)
            for rung in rung_specs
            if rung.promote_k is not None
        ],
        "final_survivors": [str(label) for label in final_survivors],
        "rungs": [dict(entry) for entry in rung_history],
    }
    (count_root / "stage1_rung_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (count_root / "stage1_rung_summary.md").write_text(
        _render_stage1_rung_markdown(
            rung_history,
            final_survivors=final_survivors,
            screen_metric_name=stage1_screen_metric,
        ),
        encoding="utf-8",
    )


def _write_count_summary(
    count_root: Path,
    *,
    args: argparse.Namespace,
    candidate_summary: Sequence[Mapping[str, Any]],
    promoted: Sequence[str],
    pareto_frontier: Sequence[str],
    variants: Sequence[Mapping[str, Any]],
    phase1_runs: int,
    phase2_runs: int,
    stage1_rungs: Sequence[Mapping[str, Any]] = (),
    stage1_final_survivors: Sequence[str] = (),
) -> None:
    payload = {
        "benchmark": str(args.benchmark),
        "phase1_train_docs": int(args.phase1_train_docs),
        "phase2_train_docs": int(args.phase2_train_docs),
        "phase1_runs": int(phase1_runs),
        "phase2_runs": int(phase2_runs),
        "candidate_summary": [dict(row) for row in candidate_summary],
        "pareto_frontier": [str(label) for label in pareto_frontier],
        "promoted_candidates": [str(label) for label in promoted],
        "surrogate_variants": [dict(variant) for variant in variants],
        "stage1_rungs": [dict(entry) for entry in stage1_rungs],
        "stage1_final_survivors": [str(label) for label in stage1_final_survivors],
        "stage1_rung_summary_path": str(count_root / "stage1_rung_summary.json"),
        "stage2_conditions": [dict(condition) for condition in tfpush.STAGE2_JUDGE_CONDITIONS],
    }
    (count_root / "teacher_first_tournament_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (count_root / "teacher_first_tournament_summary.md").write_text(
        tfpush._render_summary_markdown(
            candidate_summary,
            phase1_count=int(phase1_runs),
            phase2_count=int(phase2_runs),
            pareto_frontier=pareto_frontier,
        ),
        encoding="utf-8",
    )


def _run_single_count(
    output_root: Path,
    *,
    args: argparse.Namespace,
    train_doc_count: int,
    variants: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    count_args = _count_args(args, train_doc_count=int(train_doc_count))
    count_root = output_root / f"train_{int(train_doc_count)}"
    count_root.mkdir(parents=True, exist_ok=True)

    stage1_configs = {
        str(variant["label"]): tfpush._make_stage1_config(
            count_args,
            train_doc_count=int(count_args.phase1_train_docs),
            variant=variant,
        )
        for variant in variants
    }
    phase1_jobs = tfpush._build_phase_jobs(
        args=count_args,
        configs_by_train=[
            (int(count_args.phase1_train_docs), config)
            for config in stage1_configs.values()
        ],
        seeds=tuple(int(seed) for seed in count_args.phase1_seeds),
        tuning_stage="stage1_surrogate",
        study_axis="stage1_surrogate",
    )
    phase1 = tfpush._run_phase(
        output_root=count_root / "phase1",
        jobs=phase1_jobs,
        args=count_args,
        manifest_payload={
            "mode": "teacher_first_stage1_surrogate",
            "benchmark": str(count_args.benchmark),
            "train_doc_count": int(train_doc_count),
            "jobs": [asdict(job) for job in phase1_jobs],
        },
    )

    phase2_jobs = tfpush._build_stage2_jobs(
        args=count_args,
        stage1_runs=phase1["runs"],
        base_configs=stage1_configs,
    )
    phase2 = tfpush._run_phase(
        output_root=count_root / "phase2",
        jobs=phase2_jobs,
        args=count_args,
        manifest_payload={
            "mode": "teacher_first_stage2_judge",
            "benchmark": str(count_args.benchmark),
            "train_doc_count": int(train_doc_count),
            "jobs": [asdict(job) for job in phase2_jobs],
        },
    )

    candidate_summary = tfpush._aggregate_candidate_summary(phase2["runs"])
    pareto_frontier = [
        str(row.get("candidate_label", ""))
        for row in candidate_summary
        if bool(row.get("on_pareto_frontier", False))
    ]
    promoted = [
        str(row.get("candidate_label", ""))
        for row in candidate_summary[: max(0, int(count_args.promote_top_k))]
    ]
    _write_count_summary(
        count_root,
        args=count_args,
        candidate_summary=candidate_summary,
        promoted=promoted,
        pareto_frontier=pareto_frontier,
        variants=variants,
        phase1_runs=int(len(phase1["runs"])),
        phase2_runs=int(len(phase2["runs"])),
    )

    best = dict(candidate_summary[0]) if candidate_summary else {}
    return {
        "train_doc_count": int(train_doc_count),
        "phase2_train_docs": int(count_args.phase2_train_docs),
        "output_root": str(count_root),
        "best_candidate": best,
        "pareto_frontier": pareto_frontier,
        "promoted_candidates": promoted,
        "phase1_runs": int(len(phase1["runs"])),
        "phase2_runs": int(len(phase2["runs"])),
    }


def _render_scaling_markdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    mig_uuids: Sequence[str],
) -> str:
    lines = [
        "# Teacher-First Scaling Summary",
        "",
        f"- mig_count: `{int(len(mig_uuids))}`",
        f"- mig_uuids: `{', '.join(str(uuid) for uuid in mig_uuids)}`",
        "",
        "## Best Per Train Size",
        "",
    ]
    for row in rows:
        best = dict(row.get("best_candidate") or {})
        lines.extend(
            [
                f"### train_docs=`{int(row.get('train_doc_count', 0))}`",
                f"- phase2_train_docs: `{int(row.get('phase2_train_docs', 0))}`",
                f"- best_candidate: `{str(best.get('candidate_label', ''))}`",
                f"- mean_teacher_first_total_bound: `{float(best.get('mean_teacher_first_total_bound', float('nan'))):.6g}`",
                f"- mean_stage1_substitution_cost: `{float(best.get('mean_stage1_substitution_cost', float('nan'))):.6g}`",
                f"- mean_test_root_mae: `{float(best.get('mean_test_root_mae', float('nan'))):.6g}`",
                f"- mean_stage2_transport_budget: `{float(best.get('mean_stage2_transport_budget', float('nan'))):.6g}`",
                f"- mean_stage1_root_weight: `{float(best.get('mean_stage1_root_weight', float('nan'))):.6g}`",
                f"- stage1_final_survivors: `{', '.join(str(label) for label in row.get('stage1_final_survivors', []))}`",
                f"- pareto_frontier: `{', '.join(str(label) for label in row.get('pareto_frontier', []))}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = Path(str(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    variants = _frontier_variants()
    rung_specs = _resolved_stage1_rungs(args, variant_count=int(len(variants)))
    screen_metric_name = str(args.stage1_screen_metric or "val_root_mae").strip() or "val_root_mae"
    stage2_survivors_by_count = _resolved_stage2_survivors_by_count(args)

    mig_uuids = (
        parse_mig_uuids(str(args.mig_uuids))
        if str(args.mig_uuids).strip()
        else discover_mig_uuids()
    )
    if bool(args.use_cuda) and not mig_uuids:
        raise RuntimeError("No MIG UUIDs discovered")

    if bool(args.async_promote_per_count):
        _run_async_scaling(
            args=args,
            output_root=output_root,
            variants=variants,
            rung_specs=rung_specs,
            screen_metric_name=screen_metric_name,
            mig_uuids=mig_uuids,
        )
        summary_path = output_root / "teacher_first_scaling_summary.json"
        print(summary_path.read_text(encoding="utf-8"))
        return 0

    train_doc_counts = [int(value) for value in args.train_doc_counts]
    count_args_by_train = {
        int(train_doc_count): _count_args(args, train_doc_count=int(train_doc_count))
        for train_doc_count in train_doc_counts
    }
    stage1_configs_by_train = {
        int(train_doc_count): {
            str(variant["label"]): tfpush._make_stage1_config(
                count_args_by_train[int(train_doc_count)],
                train_doc_count=int(count_args_by_train[int(train_doc_count)].phase1_train_docs),
                variant=variant,
            )
            for variant in variants
        }
        for train_doc_count in train_doc_counts
    }
    active_labels_by_count: Dict[int, List[str]] = {
        int(train_doc_count): [str(variant["label"]) for variant in variants]
        for train_doc_count in train_doc_counts
    }
    stage1_rung_history_by_count: Dict[int, List[Dict[str, Any]]] = {
        int(train_doc_count): []
        for train_doc_count in train_doc_counts
    }
    stage1_runs_by_rung: Dict[int, List[Dict[str, Any]]] = {}
    phase1_output_roots: List[str] = []

    for rung in rung_specs:
        rung_jobs = _build_stage1_rung_jobs(
            args=args,
            train_doc_counts=train_doc_counts,
            count_args_by_train=count_args_by_train,
            stage1_configs_by_train=stage1_configs_by_train,
            active_labels_by_count=active_labels_by_count,
            rung=rung,
            screen_metric_name=screen_metric_name,
        )
        rung_output_root = output_root / f"global_phase1_rung{int(rung.index)}"
        phase1 = tfpush._run_phase(
            output_root=rung_output_root,
            jobs=rung_jobs,
            args=args,
            manifest_payload={
                "mode": "teacher_first_stage1_surrogate_rung",
                "benchmark": str(args.benchmark),
                "train_doc_counts": train_doc_counts,
                "rung_index": int(rung.index),
                "stage1_total_epochs": int(rung.total_epochs),
                "screen_metric_name": str(screen_metric_name),
                "active_labels_by_count": {
                    str(count): list(labels)
                    for count, labels in active_labels_by_count.items()
                },
                "jobs": [asdict(job) for job in rung_jobs],
            },
        )
        phase1_output_roots.append(str(rung_output_root))
        stage1_runs_by_rung[int(rung.index)] = [dict(run) for run in phase1["runs"]]

        for train_doc_count in train_doc_counts:
            count_args = count_args_by_train[int(train_doc_count)]
            count_runs = _runs_for_train_doc_count(
                phase1["runs"],
                train_doc_count=int(count_args.phase1_train_docs),
            )
            candidate_summary = _aggregate_stage1_rung_candidate_summary(
                count_runs,
                screen_metric_name=screen_metric_name,
            )
            if not candidate_summary:
                raise RuntimeError(
                    "no stage-1 runs completed for "
                    f"train_doc_count={int(train_doc_count)} rung={int(rung.index)}"
                )
            active_candidates = list(active_labels_by_count[int(train_doc_count)])
            promoted_candidates = [
                str(row.get("candidate_label", ""))
                for row in candidate_summary
            ]
            if rung.promote_k is not None:
                promoted_candidates = promoted_candidates[
                    : max(1, min(int(rung.promote_k), len(promoted_candidates)))
                ]
            active_labels_by_count[int(train_doc_count)] = list(promoted_candidates)
            stage1_rung_history_by_count[int(train_doc_count)].append(
                {
                    "rung_index": int(rung.index),
                    "total_epochs": int(rung.total_epochs),
                    "screen_metric_name": str(screen_metric_name),
                    "active_candidates": [str(label) for label in active_candidates],
                    "promoted_candidates": [
                        str(label) for label in promoted_candidates
                    ],
                    "phase_output_root": str(rung_output_root),
                    "candidate_summary": [dict(row) for row in candidate_summary],
                }
            )

    final_rung = rung_specs[-1]
    final_stage1_runs = stage1_runs_by_rung[int(final_rung.index)]
    phase2_jobs = _build_stage2_jobs_for_counts(
        train_doc_counts=train_doc_counts,
        count_args_by_train=count_args_by_train,
        stage1_configs_by_train=stage1_configs_by_train,
        active_labels_by_count=active_labels_by_count,
        final_stage1_runs=final_stage1_runs,
        stage2_survivors_by_count=stage2_survivors_by_count,
    )
    if not phase2_jobs:
        raise RuntimeError("no stage-2 jobs were created from the final stage-1 survivors")
    phase2 = tfpush._run_stage2_phase(
        output_root=output_root / "global_phase2",
        args=args,
        jobs=phase2_jobs,
        grouped_conditions=bool(args.group_stage2_conditions),
        manifest_payload={
            "mode": "teacher_first_stage2_judge_global",
            "benchmark": str(args.benchmark),
            "train_doc_counts": train_doc_counts,
            "final_stage1_survivors": {
                str(count): list(labels)
                for count, labels in active_labels_by_count.items()
            },
            "stage2_survivors_by_count": {
                str(count): int(value)
                for count, value in stage2_survivors_by_count.items()
            },
            "grouped_stage2_conditions": bool(args.group_stage2_conditions),
            "jobs": [asdict(job) for job in phase2_jobs],
        },
    )

    scaling_rows: List[Dict[str, Any]] = []
    for train_doc_count in train_doc_counts:
        count_args = count_args_by_train[int(train_doc_count)]
        count_root = output_root / f"train_{int(train_doc_count)}"
        count_root.mkdir(parents=True, exist_ok=True)
        stage1_rung_history = stage1_rung_history_by_count[int(train_doc_count)]
        final_survivors = list(
            active_labels_by_count[int(train_doc_count)][
                : max(1, int(stage2_survivors_by_count.get(int(train_doc_count), 1)))
            ]
        )
        _write_stage1_rung_summary(
            count_root,
            args=count_args,
            stage1_screen_metric=screen_metric_name,
            rung_specs=rung_specs,
            rung_history=stage1_rung_history,
            final_survivors=final_survivors,
        )
        phase2_runs_for_count = _runs_for_train_doc_count(
            phase2["runs"],
            train_doc_count=int(count_args.phase2_train_docs),
        )
        candidate_summary = tfpush._aggregate_candidate_summary(phase2_runs_for_count)
        pareto_frontier = [
            str(row.get("candidate_label", ""))
            for row in candidate_summary
            if bool(row.get("on_pareto_frontier", False))
        ]
        promoted = [
            str(row.get("candidate_label", ""))
            for row in candidate_summary[: max(0, int(count_args.promote_top_k))]
        ]
        _write_count_summary(
            count_root,
            args=count_args,
            candidate_summary=candidate_summary,
            promoted=promoted,
            pareto_frontier=pareto_frontier,
            variants=variants,
            phase1_runs=int(
                sum(
                    len(
                        _runs_for_train_doc_count(
                            stage1_runs_by_rung[int(rung.index)],
                            train_doc_count=int(count_args.phase1_train_docs),
                        )
                    )
                    for rung in rung_specs
                )
            ),
            phase2_runs=int(len(phase2_runs_for_count)),
            stage1_rungs=stage1_rung_history,
            stage1_final_survivors=final_survivors,
        )
        best = dict(candidate_summary[0]) if candidate_summary else {}
        scaling_rows.append(
            {
                "train_doc_count": int(train_doc_count),
                "phase2_train_docs": int(count_args.phase2_train_docs),
                "output_root": str(count_root),
                "best_candidate": best,
                "pareto_frontier": pareto_frontier,
                "promoted_candidates": promoted,
                "phase1_runs": int(
                    sum(
                        len(
                            _runs_for_train_doc_count(
                                stage1_runs_by_rung[int(rung.index)],
                                train_doc_count=int(count_args.phase1_train_docs),
                            )
                        )
                        for rung in rung_specs
                    )
                ),
                "phase2_runs": int(len(phase2_runs_for_count)),
                "stage1_final_survivors": list(final_survivors),
            }
        )

    payload = {
        "benchmark": str(args.benchmark),
        "train_doc_counts": train_doc_counts,
        "phase2_train_multiplier": float(args.phase2_train_multiplier),
        "phase1_seeds": [int(seed) for seed in args.phase1_seeds],
        "phase2_seeds": [int(seed) for seed in args.phase2_seeds],
        "stage1_rung_epochs": [int(rung.total_epochs) for rung in rung_specs],
        "stage1_rung_promote_k": [
            int(rung.promote_k)
            for rung in rung_specs
            if rung.promote_k is not None
        ],
        "stage1_screen_metric": str(screen_metric_name),
        "mig_uuids": [str(uuid) for uuid in mig_uuids],
        "frontier_variants": [dict(variant) for variant in variants],
        "global_phase1_output_roots": phase1_output_roots,
        "global_phase2_output_root": str(output_root / "global_phase2"),
        "global_phase1_runs": int(
            sum(len(runs) for runs in stage1_runs_by_rung.values())
        ),
        "global_phase2_runs": int(len(phase2["runs"])),
        "scaling_rows": scaling_rows,
    }
    (output_root / "teacher_first_scaling_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "teacher_first_scaling_summary.md").write_text(
        _render_scaling_markdown(scaling_rows, mig_uuids=mig_uuids),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
