#!/usr/bin/env python3
"""Reliable slotwise data-scaling push for the strongest validated branch."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.core.tree_neural_execution import (  # noqa: E402
    run_job_batch,
    write_summary_outputs,
)
from src.ctreepo.sim.core.tree_neural_exact_sanity import (  # noqa: E402
    EXACT_SANITY_FAMILY,
    render_exact_sanity_summary_markdown,
    tree_neural_exact_sanity_summary,
)
from src.ctreepo.sim.core.tree_neural_facade import (  # noqa: E402
    build_jobs_for_configs,
    JobSpec,
    RunConfigSpec,
    discover_mig_uuids,
    parse_mig_uuids,
)
from scripts import run_tree_neural_learning_push as lp  # noqa: E402


SCALING_CONDITIONS = (
    ("internal_full_dense", 1.0, "full_sketch", "full_sketch", 1.0),
    ("internal_full_r0p25", 0.25, "full_sketch", "full_sketch", 0.25),
)


def _parser() -> argparse.ArgumentParser:
    parser = lp._parser()
    parser.description = __doc__
    parser.set_defaults(
        output_root=f"outputs/tree_neural_slotwise_scaling_push_{lp._timestamp()}",
        phase1_train_large=1024,
        phase2_train_x5=5120,
        phase2_seeds=(0, 1),
        n_epochs=52,
        tree_stage1_epochs=12,
        tree_stage2_epochs=40,
    )
    parser.add_argument(
        "--mig-uuids",
        type=str,
        default="",
        help="Optional comma/space separated MIG UUID subset for this runner.",
    )
    parser.add_argument(
        "--phase3-seeds",
        nargs="*",
        type=int,
        default=(2, 3, 4),
        help="Optional extra seeds for the stronger 5120-doc condition.",
    )
    return parser


def _resolve_mig_uuids(args: argparse.Namespace) -> list[str]:
    raw = str(getattr(args, "mig_uuids", "") or "").strip()
    mig_uuids = parse_mig_uuids(raw) if raw else discover_mig_uuids()
    if not mig_uuids:
        raise RuntimeError("No MIG UUIDs discovered")
    return list(mig_uuids)


def _write_outputs(output_root: Path) -> None:
    payload = write_summary_outputs(output_root)
    exact = tree_neural_exact_sanity_summary(dict(payload or {}))
    (output_root / "tree_neural_exact_sanity_summary.json").write_text(
        json.dumps(exact, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "tree_neural_exact_sanity_summary.md").write_text(
        render_exact_sanity_summary_markdown(exact),
        encoding="utf-8",
    )


def _collect_all_runs(output_root: Path) -> list[dict[str, Any]]:
    summary_path = output_root / "summary.json"
    if not summary_path.exists():
        return []
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return [dict(run) for run in payload.get("runs") or []]


def _run_jobs(
    *,
    output_root: Path,
    jobs: Sequence[JobSpec],
    torch_threads: int,
    use_cuda: bool,
    mig_uuids: Sequence[str],
    mode: str,
) -> Dict[str, Any]:
    manifest = {
        "mode": str(mode),
        "jobs": [asdict(job) for job in jobs],
    }
    return run_job_batch(
        output_root=output_root,
        jobs=jobs,
        mig_uuids=tuple(str(uuid) for uuid in mig_uuids),
        resume_enabled=True,
        use_cuda=bool(use_cuda),
        torch_threads=int(torch_threads),
        manifest_payload=manifest,
    )


def _make_slotwise_config(
    args: argparse.Namespace,
    *,
    train_doc_count: int,
    label: str,
    leaf_label_rate: float,
    leaf_supervision_kind: str,
    internal_supervision_kind: str,
    internal_label_rate: float,
) -> RunConfigSpec:
    base = lp._make_slot_config(
        args,
        train_doc_count=int(train_doc_count),
        label=str(label),
        leaf_label_rate=float(leaf_label_rate),
        leaf_supervision_kind=str(leaf_supervision_kind),
        internal_supervision_kind=str(internal_supervision_kind),
        internal_label_rate=float(internal_label_rate),
        tree_summary_spec_root_mode="factored_theorem_readout",
        n_epochs=int(args.n_epochs),
        tree_training_schedule="two_stage",
        tree_stage1_epochs=int(args.tree_stage1_epochs),
        tree_stage2_epochs=int(args.tree_stage2_epochs),
        tree_task_objective_weight=1.0,
    )
    return replace(
        base,
        tree_task_head_mode="theorem_feature_scalar",
        tree_theorem_surface_mode="slotwise",
        tree_theorem_count_head_mode="scalar_mse",
        tree_checkpoint_metric="val_exact_sketch_direct",
        tree_stage1_checkpoint_metric="val_theorem_bootstrap_direct",
        tree_summary_spec_root_mode="factored_theorem_readout",
        tree_phi_compose_weight=0.0,
        tree_phi_contrastive_weight=0.0,
    )


def _build_jobs_for_configs(
    args: argparse.Namespace,
    configs_by_train: Sequence[tuple[int, RunConfigSpec]],
    *,
    seeds: Sequence[int],
    tuning_stage: str,
    axis_value: str,
) -> list[JobSpec]:
    jobs: list[JobSpec] = []
    for train_doc_count, config in configs_by_train:
        jobs.extend(
            build_jobs_for_configs(
                families=(EXACT_SANITY_FAMILY,),
                train_doc_counts=(int(train_doc_count),),
                benchmark=str(args.benchmark),
                hardness_grid="",
                grid_cell_ids=(),
                seeds=tuple(int(seed) for seed in seeds),
                job_granularity="family_train_seed",
                repeat_closed_form_controls=False,
                configs=(config,),
                tuning_stage=str(tuning_stage),
                study_name="slotwise_scaling_push",
                study_axis="promotion_stage",
                axis_value=str(axis_value),
                selection_metric="exact_sketch_diagnostic_only",
            )
        )
    return jobs


def _configs_for_train_doc_count(
    args: argparse.Namespace,
    *,
    train_doc_count: int,
    suffix: str,
) -> list[tuple[int, RunConfigSpec]]:
    configs: list[tuple[int, RunConfigSpec]] = []
    for condition_label, leaf_rate, leaf_kind, internal_kind, internal_rate in SCALING_CONDITIONS:
        configs.append(
            (
                int(train_doc_count),
                _make_slotwise_config(
                    args,
                    train_doc_count=int(train_doc_count),
                    label=f"slotwise_scaling_{condition_label}_{suffix}",
                    leaf_label_rate=float(leaf_rate),
                    leaf_supervision_kind=str(leaf_kind),
                    internal_supervision_kind=str(internal_kind),
                    internal_label_rate=float(internal_rate),
                ),
            )
        )
    return configs


def _scaling_score(run: Mapping[str, Any]) -> float:
    root = lp._direct_metric(run, "root_direct_count_mae", default=1e9)
    merge = lp._direct_metric(run, "merge_direct_exact_match", default=0.0)
    leaf = lp._direct_metric(run, "leaf_direct_exact_match", default=0.0)
    return float(root + max(0.0, 0.80 - merge) * 3.0 + max(0.0, 0.88 - leaf) * 2.0)


def _best_5120_label(runs: Sequence[Mapping[str, Any]]) -> str | None:
    candidates = [
        dict(run)
        for run in runs
        if str(run.get("tuning_stage", "")) == "phase2"
        and int(run.get("train_doc_count", 0)) > 0
        and int(run.get("train_doc_count", 0)) == 5120
    ]
    if not candidates:
        return None
    return str(min(candidates, key=_scaling_score).get("config_label", ""))


def _write_status(output_root: Path, payload: Mapping[str, Any]) -> None:
    (output_root / "slotwise_scaling_status.json").write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    mig_uuids = _resolve_mig_uuids(args)

    phase1_jobs = _build_jobs_for_configs(
        args,
        _configs_for_train_doc_count(
            args,
            train_doc_count=int(args.phase1_train_large),
            suffix=str(int(args.phase1_train_large)),
        ),
        seeds=tuple(int(seed) for seed in args.phase2_seeds),
        tuning_stage="phase1",
        axis_value="phase1",
    )
    phase1_result = _run_jobs(
        output_root=output_root,
        jobs=phase1_jobs,
        torch_threads=int(args.torch_threads),
        use_cuda=bool(args.use_cuda),
        mig_uuids=mig_uuids,
        mode="slotwise_scaling_phase1",
    )
    _write_outputs(output_root)

    phase2_jobs = _build_jobs_for_configs(
        args,
        _configs_for_train_doc_count(
            args,
            train_doc_count=int(args.phase2_train_x5),
            suffix=str(int(args.phase2_train_x5)),
        ),
        seeds=tuple(int(seed) for seed in args.phase2_seeds),
        tuning_stage="phase2",
        axis_value="phase2",
    )
    phase2_result = _run_jobs(
        output_root=output_root,
        jobs=phase2_jobs,
        torch_threads=int(args.torch_threads),
        use_cuda=bool(args.use_cuda),
        mig_uuids=mig_uuids,
        mode="slotwise_scaling_phase2",
    )
    _write_outputs(output_root)

    all_runs = _collect_all_runs(output_root)
    best_label = _best_5120_label(all_runs)
    _write_status(
        output_root,
        {
            "output_root": str(output_root),
            "mig_uuids": list(mig_uuids),
            "phase1_completed_jobs": len(list(phase1_result.get("completed_jobs", ()))),
            "phase1_failed_jobs": len(list(phase1_result.get("failed_jobs", ()))),
            "phase2_completed_jobs": len(list(phase2_result.get("completed_jobs", ()))),
            "phase2_failed_jobs": len(list(phase2_result.get("failed_jobs", ()))),
            "best_5120_label": best_label or "",
        },
    )

    if best_label:
        followup_configs: list[tuple[int, RunConfigSpec]] = []
        for train_doc_count, config in _configs_for_train_doc_count(
            args,
            train_doc_count=int(args.phase2_train_x5),
            suffix=str(int(args.phase2_train_x5)),
        ):
            if str(config.label) == str(best_label):
                followup_configs.append(
                    (
                        train_doc_count,
                        replace(
                            config,
                            label=f"{config.label}__extra_seeds",
                        ),
                    )
                )
                break
        if followup_configs:
            phase3_jobs = _build_jobs_for_configs(
                args,
                followup_configs,
                seeds=tuple(int(seed) for seed in args.phase3_seeds),
                tuning_stage="phase3",
                axis_value="phase3",
            )
            phase3_result = _run_jobs(
                output_root=output_root,
                jobs=phase3_jobs,
                torch_threads=int(args.torch_threads),
                use_cuda=bool(args.use_cuda),
                mig_uuids=mig_uuids,
                mode="slotwise_scaling_phase3",
            )
            _write_outputs(output_root)
            _write_status(
                output_root,
                {
                    "output_root": str(output_root),
                    "mig_uuids": list(mig_uuids),
                    "best_5120_label": best_label,
                    "phase3_completed_jobs": len(
                        list(phase3_result.get("completed_jobs", ()))
                    ),
                    "phase3_failed_jobs": len(
                        list(phase3_result.get("failed_jobs", ()))
                    ),
                },
            )

    final_status = {
        "output_root": str(output_root),
        "mig_uuids": list(mig_uuids),
        "best_5120_label": best_label or "",
    }
    print(json.dumps(final_status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
