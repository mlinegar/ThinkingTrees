#!/usr/bin/env python3
"""Autonomous multi-phase exact-sanity learning push.

This script focuses on the least-constrained promising lane:
theorem-primary summary-spec with the simple scalar count head.
It spends GPU budget on learning-dynamics variants rather than
more architectural structure, then promotes winners to larger
sample sizes automatically.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_tree_neural_full_doc_mig as mig  # noqa: E402


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _base_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        benchmark=str(args.benchmark),
        train_doc_counts=(int(args.phase1_train_small),),
        seeds=(0,),
        job_granularity="family_train_seed",
        resume=True,
        mig_uuids="",
        state_dim=int(args.state_dim),
        hidden_dim=int(args.hidden_dim),
        n_epochs=int(args.n_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        tree_local_law_weight=float(args.tree_local_law_weight),
        tree_task_objective_weight=None,
        tree_c1_relative_weight=1.0,
        tree_c2_relative_weight=1.0,
        tree_c3_relative_weight=1.0,
        tree_checkpoint_metric="val_exact_sketch_direct",
        tree_stage1_checkpoint_metric="val_theorem_bootstrap_direct",
        tree_join_bit_weight=float(args.tree_join_bit_weight),
        tree_training_schedule="two_stage",
        tree_stage1_epochs=int(args.tree_stage1_epochs),
        tree_stage2_epochs=int(args.tree_stage2_epochs),
        tree_task_head_mode="full_state_scalar",
        tree_theorem_count_head_mode="scalar_mse",
        tree_theorem_count_ordinal_weight=1.0,
        tree_theorem_count_scalar_aux_weight=0.25,
        tree_theorem_count_threshold_balance=True,
        tree_summary_spec_root_mode="theorem_primary",
        tree_theorem_count_dim=int(args.tree_theorem_count_dim),
        tree_theorem_first_dim=int(args.tree_theorem_first_dim),
        tree_theorem_last_dim=int(args.tree_theorem_last_dim),
        leaf_supervision_kind="full_sketch",
        doc_sequence_train_fraction=0.0,
        torch_threads=int(args.torch_threads),
        use_cuda=bool(args.use_cuda),
    )


def _make_slot_config(
    args: argparse.Namespace,
    *,
    train_doc_count: int,
    label: str,
    leaf_label_rate: float,
    leaf_supervision_kind: str,
    internal_supervision_kind: str,
    internal_label_rate: float,
    tree_summary_spec_root_mode: str = "theorem_primary",
    state_dim: int | None = None,
    hidden_dim: int | None = None,
    n_epochs: int | None = None,
    tree_training_schedule: str | None = None,
    tree_stage1_epochs: int | None = None,
    tree_stage2_epochs: int | None = None,
    tree_local_law_weight: float | None = None,
    tree_task_objective_weight: float | None = None,
    tree_c1_relative_weight: float = 1.0,
    tree_c2_relative_weight: float = 1.0,
    tree_c3_relative_weight: float = 1.0,
) -> mig._RunConfigSpec:
    base = mig._slot_exact_sanity_config(
        _base_args(args),
        train_doc_count=int(train_doc_count),
        config_label=str(label),
        leaf_label_rate=float(leaf_label_rate),
        leaf_supervision_kind=str(leaf_supervision_kind),
        internal_supervision_kind=str(internal_supervision_kind),
        internal_label_rate=float(internal_label_rate),
        tree_summary_spec_root_mode=str(tree_summary_spec_root_mode),
    )
    final_stage1 = (
        int(tree_stage1_epochs)
        if tree_stage1_epochs is not None
        else int(base.tree_stage1_epochs)
    )
    final_stage2 = (
        int(tree_stage2_epochs)
        if tree_stage2_epochs is not None
        else int(base.tree_stage2_epochs)
    )
    final_schedule = (
        str(tree_training_schedule)
        if tree_training_schedule is not None
        else str(base.tree_training_schedule)
    )
    final_epochs = (
        int(n_epochs)
        if n_epochs is not None
        else (
            int(final_stage1 + final_stage2)
            if final_schedule == "two_stage"
            else int(base.n_epochs)
        )
    )
    return replace(
        base,
        label=str(label),
        state_dim=int(state_dim if state_dim is not None else base.state_dim),
        hidden_dim=int(hidden_dim if hidden_dim is not None else base.hidden_dim),
        n_epochs=int(final_epochs),
        tree_training_schedule=str(final_schedule),
        tree_stage1_epochs=int(final_stage1),
        tree_stage2_epochs=int(final_stage2),
        tree_local_law_weight=(
            float(tree_local_law_weight)
            if tree_local_law_weight is not None
            else base.tree_local_law_weight
        ),
        tree_task_objective_weight=(
            None
            if tree_task_objective_weight is None
            else float(tree_task_objective_weight)
        ),
        tree_c1_relative_weight=float(tree_c1_relative_weight),
        tree_c2_relative_weight=float(tree_c2_relative_weight),
        tree_c3_relative_weight=float(tree_c3_relative_weight),
        tree_theorem_count_head_mode="scalar_mse",
        tree_summary_spec_root_mode=str(tree_summary_spec_root_mode),
    )


def _phase1_configs(args: argparse.Namespace) -> List[tuple[int, mig._RunConfigSpec]]:
    configs: List[tuple[int, mig._RunConfigSpec]] = []
    small = int(args.phase1_train_small)
    large = int(args.phase1_train_large)

    def add(
        train_doc_count: int,
        *,
        label: str,
        leaf_label_rate: float,
        leaf_supervision_kind: str,
        internal_supervision_kind: str,
        internal_label_rate: float,
        root_mode: str = "theorem_primary",
        state_dim: int | None = None,
        hidden_dim: int | None = None,
        n_epochs: int | None = None,
        schedule: str | None = None,
        stage1_epochs: int | None = None,
        stage2_epochs: int | None = None,
        task_weight: float | None = None,
        c1: float = 1.0,
        c2: float = 1.0,
        c3: float = 1.0,
    ) -> None:
        configs.append(
            (
                int(train_doc_count),
                _make_slot_config(
                    args,
                    train_doc_count=int(train_doc_count),
                    label=str(label),
                    leaf_label_rate=float(leaf_label_rate),
                    leaf_supervision_kind=str(leaf_supervision_kind),
                    internal_supervision_kind=str(internal_supervision_kind),
                    internal_label_rate=float(internal_label_rate),
                    tree_summary_spec_root_mode=str(root_mode),
                    state_dim=state_dim,
                    hidden_dim=hidden_dim,
                    n_epochs=n_epochs,
                    tree_training_schedule=schedule,
                    tree_stage1_epochs=stage1_epochs,
                    tree_stage2_epochs=stage2_epochs,
                    tree_local_law_weight=float(args.tree_local_law_weight),
                    tree_task_objective_weight=task_weight,
                    tree_c1_relative_weight=float(c1),
                    tree_c2_relative_weight=float(c2),
                    tree_c3_relative_weight=float(c3),
                ),
            )
        )

    # 256-doc learning-dynamics sweep.
    add(
        small,
        label="learningpush_scalar_leaf_dense_base",
        leaf_label_rate=1.0,
        leaf_supervision_kind="full_sketch",
        internal_supervision_kind="none",
        internal_label_rate=0.0,
    )
    add(
        small,
        label="learningpush_scalar_icd_base",
        leaf_label_rate=1.0,
        leaf_supervision_kind="count_only",
        internal_supervision_kind="count_only",
        internal_label_rate=1.0,
    )
    add(
        small,
        label="learningpush_scalar_ifd_base",
        leaf_label_rate=1.0,
        leaf_supervision_kind="full_sketch",
        internal_supervision_kind="full_sketch",
        internal_label_rate=1.0,
    )
    add(
        small,
        label="learningpush_scalar_icd_c2half",
        leaf_label_rate=1.0,
        leaf_supervision_kind="count_only",
        internal_supervision_kind="count_only",
        internal_label_rate=1.0,
        c2=0.5,
    )
    add(
        small,
        label="learningpush_scalar_icd_c2quarter",
        leaf_label_rate=1.0,
        leaf_supervision_kind="count_only",
        internal_supervision_kind="count_only",
        internal_label_rate=1.0,
        c2=0.25,
    )
    add(
        small,
        label="learningpush_scalar_icd_noc2",
        leaf_label_rate=1.0,
        leaf_supervision_kind="count_only",
        internal_supervision_kind="count_only",
        internal_label_rate=1.0,
        c2=0.0,
    )
    add(
        small,
        label="learningpush_scalar_icd_root03",
        leaf_label_rate=1.0,
        leaf_supervision_kind="count_only",
        internal_supervision_kind="count_only",
        internal_label_rate=1.0,
        task_weight=0.3,
    )
    add(
        small,
        label="learningpush_scalar_icd_longstage2",
        leaf_label_rate=1.0,
        leaf_supervision_kind="count_only",
        internal_supervision_kind="count_only",
        internal_label_rate=1.0,
        stage1_epochs=12,
        stage2_epochs=40,
        n_epochs=52,
    )
    add(
        small,
        label="learningpush_scalar_icd_bigcap",
        leaf_label_rate=1.0,
        leaf_supervision_kind="count_only",
        internal_supervision_kind="count_only",
        internal_label_rate=1.0,
        state_dim=256,
        hidden_dim=1024,
    )
    add(
        small,
        label="learningpush_scalar_ifd_c2quarter",
        leaf_label_rate=1.0,
        leaf_supervision_kind="full_sketch",
        internal_supervision_kind="full_sketch",
        internal_label_rate=1.0,
        c2=0.25,
    )
    add(
        small,
        label="learningpush_scalar_icd_single_stage48",
        leaf_label_rate=1.0,
        leaf_supervision_kind="count_only",
        internal_supervision_kind="count_only",
        internal_label_rate=1.0,
        schedule="single_stage",
        stage1_epochs=0,
        stage2_epochs=0,
        n_epochs=48,
    )
    add(
        small,
        label="learningpush_scalar_leaf_dense_root03",
        leaf_label_rate=1.0,
        leaf_supervision_kind="full_sketch",
        internal_supervision_kind="none",
        internal_label_rate=0.0,
        task_weight=0.3,
    )

    # 1024-doc confirmations to keep the fleet full and reduce iteration risk.
    add(
        large,
        label="learningpush_scalar_icd_base_1024",
        leaf_label_rate=1.0,
        leaf_supervision_kind="count_only",
        internal_supervision_kind="count_only",
        internal_label_rate=1.0,
    )
    add(
        large,
        label="learningpush_scalar_icd_c2quarter_1024",
        leaf_label_rate=1.0,
        leaf_supervision_kind="count_only",
        internal_supervision_kind="count_only",
        internal_label_rate=1.0,
        c2=0.25,
    )
    add(
        large,
        label="learningpush_scalar_ifd_base_1024",
        leaf_label_rate=1.0,
        leaf_supervision_kind="full_sketch",
        internal_supervision_kind="full_sketch",
        internal_label_rate=1.0,
    )
    add(
        large,
        label="learningpush_scalar_leaf_dense_1024",
        leaf_label_rate=1.0,
        leaf_supervision_kind="full_sketch",
        internal_supervision_kind="none",
        internal_label_rate=0.0,
    )
    return configs


def _build_phase_jobs(
    args: argparse.Namespace,
    configs_by_train: Sequence[tuple[int, mig._RunConfigSpec]],
    *,
    seeds: Sequence[int],
    tuning_stage: str,
) -> List[mig._JobSpec]:
    jobs: List[mig._JobSpec] = []
    for train_doc_count, config in configs_by_train:
        jobs.extend(
            mig._build_jobs_for_configs(
                families=(mig.EXACT_SANITY_FAMILY,),
                train_doc_counts=(int(train_doc_count),),
                benchmark=str(args.benchmark),
                hardness_grid="",
                grid_cell_ids=(),
                seeds=tuple(int(seed) for seed in seeds),
                job_granularity="family_train_seed",
                repeat_closed_form_controls=True,
                configs=(config,),
                tuning_stage=str(tuning_stage),
                study_name="tree_neural_learning_push",
                study_axis="learning_push_config",
                axis_value=str(config.label),
                selection_metric="exact_sketch_diagnostic_only",
            )
        )
    return jobs


def _write_exact_summary(output_root: Path, payload: Mapping[str, Any]) -> None:
    exact_summary = mig._tree_neural_exact_sanity_summary(dict(payload or {}))
    (output_root / "tree_neural_exact_sanity_summary.json").write_text(
        json.dumps(exact_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "tree_neural_exact_sanity_summary.md").write_text(
        mig._render_exact_sanity_summary_markdown(exact_summary)
        if exact_summary
        else "# Tree-Neural Exact-Sketch Sanity Summary\n\nNo exact-sanity runs found.\n",
        encoding="utf-8",
    )


def _load_run_records(output_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted((output_root / "jobs").glob("*/runs/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["_path"] = str(path)
        records.append(payload)
    return records


def _direct_metric(run: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    if key in run and run.get(key) not in {"", None}:
        return float(run.get(key))
    direct = dict((run.get("exact_sketch_diagnostics") or {}).get("direct_selection_metrics", {}).get("test", {}) or {})
    if key in direct and direct.get(key) not in {"", None}:
        return float(direct.get(key))
    return float(default)


def _score_run(run: Mapping[str, Any]) -> float:
    root = _direct_metric(run, "root_direct_count_mae", default=1e9)
    merge_exact = _direct_metric(run, "merge_direct_exact_match", default=0.0)
    leaf_exact = _direct_metric(run, "leaf_direct_exact_match", default=0.0)
    c2_exact = _direct_metric(run, "c2_on_range_exact_match", default=0.0)
    join_acc = _direct_metric(run, "merge_join_bit_accuracy", default=0.0)
    penalty = 0.0
    penalty += max(0.0, 0.90 - merge_exact) * 5.0
    penalty += max(0.0, 0.95 - leaf_exact) * 2.0
    penalty += max(0.0, 0.90 - c2_exact) * 2.0
    penalty += max(0.0, 0.99 - join_acc) * 2.0
    return float(root + penalty)


def _select_promotions(
    runs: Sequence[Mapping[str, Any]],
    *,
    train_doc_count: int,
    max_configs: int,
) -> List[str]:
    candidates = [
        run
        for run in runs
        if int(run.get("train_doc_count", 0)) == int(train_doc_count)
        and str(run.get("baseline_family", "")) == mig.EXACT_SANITY_FAMILY
    ]
    best_by_label: Dict[str, Dict[str, Any]] = {}
    for run in candidates:
        label = str(run.get("config_label", "")).strip()
        if not label:
            continue
        current = best_by_label.get(label)
        if current is None or _score_run(run) < _score_run(current):
            best_by_label[label] = dict(run)
    ranked = sorted(best_by_label.values(), key=_score_run)
    promoted: List[str] = []
    for run in ranked:
        promoted.append(str(run.get("config_label", "")))
        if len(promoted) >= int(max_configs):
            break
    return promoted


def _promotion_configs(
    args: argparse.Namespace,
    phase1_runs: Sequence[Mapping[str, Any]],
) -> List[tuple[int, mig._RunConfigSpec]]:
    label_to_config: Dict[str, mig._RunConfigSpec] = {
        config.label: config for _, config in _phase1_configs(args)
    }
    promoted_small = _select_promotions(
        phase1_runs,
        train_doc_count=int(args.phase1_train_small),
        max_configs=int(args.promote_from_small),
    )
    promoted_large = _select_promotions(
        phase1_runs,
        train_doc_count=int(args.phase1_train_large),
        max_configs=int(args.promote_from_large),
    )
    unique_labels: List[str] = []
    for label in [*promoted_small, *promoted_large]:
        if label not in unique_labels:
            unique_labels.append(label)

    promotions: List[tuple[int, mig._RunConfigSpec]] = []
    for label in unique_labels:
        base = label_to_config.get(label)
        if base is None:
            continue
        # Always include the phase-2 large-doc confirmation lane, even when the
        # doc count matches phase 1. Phase 2 expands across additional seeds, so
        # "same train_doc_count" is not redundant.
        promotions.append(
            (
                int(args.phase2_train_large),
                replace(
                    base,
                    label=f"{base.label}__p2t{int(args.phase2_train_large)}",
                ),
            )
        )
        promotions.append(
            (
                int(args.phase2_train_x5),
                replace(
                    base,
                    label=f"{base.label}__p2t{int(args.phase2_train_x5)}",
                ),
            )
        )
    return promotions


def _run_phase(
    *,
    output_root: Path,
    jobs: Sequence[mig._JobSpec],
    args: argparse.Namespace,
    manifest_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    mig_uuids = mig._discover_mig_uuids()
    if not mig_uuids:
        raise RuntimeError("No MIG UUIDs discovered")
    result = mig._run_job_batch(
        output_root=output_root,
        jobs=jobs,
        mig_uuids=mig_uuids,
        resume_enabled=True,
        use_cuda=bool(args.use_cuda),
        torch_threads=int(args.torch_threads),
        manifest_payload=dict(manifest_payload),
    )
    payload = mig._write_summary_outputs(output_root)
    _write_exact_summary(output_root, payload)
    return {
        "result": result,
        "payload": payload,
        "runs": _load_run_records(output_root),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=str,
        default=f"outputs/tree_neural_learning_push_{_timestamp()}",
    )
    parser.add_argument("--benchmark", type=str, default="recoverable_v4")
    parser.add_argument("--phase1-train-small", type=int, default=256)
    parser.add_argument("--phase1-train-large", type=int, default=1024)
    parser.add_argument("--phase2-train-large", type=int, default=1024)
    parser.add_argument("--phase2-train-x5", type=int, default=5120)
    parser.add_argument("--phase2-seeds", nargs="*", type=int, default=(0, 1, 2))
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--tree-local-law-weight", type=float, default=0.8)
    parser.add_argument("--tree-join-bit-weight", type=float, default=1.0)
    parser.add_argument("--tree-stage1-epochs", type=int, default=12)
    parser.add_argument("--tree-stage2-epochs", type=int, default=20)
    parser.add_argument("--tree-theorem-count-dim", type=int, default=8)
    parser.add_argument("--tree-theorem-first-dim", type=int, default=8)
    parser.add_argument("--tree-theorem-last-dim", type=int, default=8)
    parser.add_argument("--promote-from-small", type=int, default=4)
    parser.add_argument("--promote-from-large", type=int, default=2)
    parser.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_root = Path(str(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    phase1_configs = _phase1_configs(args)
    phase1_jobs = _build_phase_jobs(
        args,
        phase1_configs,
        seeds=(0,),
        tuning_stage="phase1",
    )
    phase1 = _run_phase(
        output_root=output_root / "phase1",
        jobs=phase1_jobs,
        args=args,
        manifest_payload={
            "mode": "learning_push_phase1",
            "benchmark": str(args.benchmark),
            "jobs": [asdict(job) for job in phase1_jobs],
        },
    )

    phase2_configs = _promotion_configs(args, phase1["runs"])
    phase2_jobs = _build_phase_jobs(
        args,
        phase2_configs,
        seeds=tuple(int(seed) for seed in args.phase2_seeds),
        tuning_stage="phase2",
    )
    phase2 = _run_phase(
        output_root=output_root / "phase2",
        jobs=phase2_jobs,
        args=args,
        manifest_payload={
            "mode": "learning_push_phase2",
            "benchmark": str(args.benchmark),
            "phase1_promoted_config_labels": [config.label for _, config in phase2_configs],
            "jobs": [asdict(job) for job in phase2_jobs],
        },
    )

    summary = {
        "output_root": str(output_root),
        "phase1_output_root": str(output_root / "phase1"),
        "phase2_output_root": str(output_root / "phase2"),
        "phase1_promoted_labels": [config.label for _, config in phase2_configs],
        "phase1_completed_jobs": len(list(phase1["result"]["completed_jobs"])),
        "phase1_failed_jobs": len(list(phase1["result"]["failed_jobs"])),
        "phase2_completed_jobs": len(list(phase2["result"]["completed_jobs"])),
        "phase2_failed_jobs": len(list(phase2["result"]["failed_jobs"])),
    }
    (output_root / "learning_push_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not phase1["result"]["failed_jobs"] and not phase2["result"]["failed_jobs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
