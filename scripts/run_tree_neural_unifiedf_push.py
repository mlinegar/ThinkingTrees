#!/usr/bin/env python3
"""Focused unified_f exact-sanity push.

This sidecar targets the looser `unified_f` formulation with the scalar
theorem-count head. It is intended to run alongside or after the main
theorem-primary learning push so we do not overcommit to a single routing
choice while exploring the most general viable formulation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_tree_neural_learning_push as lp  # noqa: E402
from src.ctreepo.sim.core.tree_neural_facade import RunConfigSpec  # noqa: E402


def _parser():
    parser = lp._parser()
    parser.description = __doc__
    parser.set_defaults(
        output_root=f"outputs/tree_neural_unifiedf_push_{lp._timestamp()}",
        promote_from_small=3,
        promote_from_large=2,
    )
    return parser


def _make_unifiedf_config(
    args,
    *,
    train_doc_count: int,
    label: str,
    leaf_label_rate: float,
    leaf_supervision_kind: str,
    internal_supervision_kind: str,
    internal_label_rate: float,
) -> RunConfigSpec:
    return lp._make_slot_config(
        args,
        train_doc_count=int(train_doc_count),
        label=str(label),
        leaf_label_rate=float(leaf_label_rate),
        leaf_supervision_kind=str(leaf_supervision_kind),
        internal_supervision_kind=str(internal_supervision_kind),
        internal_label_rate=float(internal_label_rate),
        tree_summary_spec_root_mode="unified_f",
    )


def _phase1_configs(args) -> List[tuple[int, RunConfigSpec]]:
    small = int(args.phase1_train_small)
    large = int(args.phase1_train_large)
    specs: List[tuple[int, RunConfigSpec]] = []

    def add(
        train_doc_count: int,
        *,
        label: str,
        leaf_label_rate: float,
        leaf_supervision_kind: str,
        internal_supervision_kind: str,
        internal_label_rate: float,
    ) -> None:
        specs.append(
            (
                int(train_doc_count),
                _make_unifiedf_config(
                    args,
                    train_doc_count=int(train_doc_count),
                    label=str(label),
                    leaf_label_rate=float(leaf_label_rate),
                    leaf_supervision_kind=str(leaf_supervision_kind),
                    internal_supervision_kind=str(internal_supervision_kind),
                    internal_label_rate=float(internal_label_rate),
                ),
            )
        )

    for train_doc_count in (small, large):
        suffix = "" if int(train_doc_count) == small else "_1024"
        add(
            train_doc_count,
            label=f"learningpush_unifiedf_root_only{suffix}",
            leaf_label_rate=0.0,
            leaf_supervision_kind="count_only",
            internal_supervision_kind="none",
            internal_label_rate=0.0,
        )
        add(
            train_doc_count,
            label=f"learningpush_unifiedf_count_r0p25{suffix}",
            leaf_label_rate=0.25,
            leaf_supervision_kind="count_only",
            internal_supervision_kind="count_only",
            internal_label_rate=0.25,
        )
        add(
            train_doc_count,
            label=f"learningpush_unifiedf_full_r0p25{suffix}",
            leaf_label_rate=0.25,
            leaf_supervision_kind="full_sketch",
            internal_supervision_kind="full_sketch",
            internal_label_rate=0.25,
        )
        add(
            train_doc_count,
            label=f"learningpush_unifiedf_count_dense{suffix}",
            leaf_label_rate=1.0,
            leaf_supervision_kind="count_only",
            internal_supervision_kind="count_only",
            internal_label_rate=1.0,
        )
        add(
            train_doc_count,
            label=f"learningpush_unifiedf_full_dense{suffix}",
            leaf_label_rate=1.0,
            leaf_supervision_kind="full_sketch",
            internal_supervision_kind="full_sketch",
            internal_label_rate=1.0,
        )
    return specs


def _promotion_configs(
    args,
    phase1_runs: Sequence[Mapping[str, Any]],
) -> List[tuple[int, RunConfigSpec]]:
    label_to_config: Dict[str, RunConfigSpec] = {
        config.label: config for _, config in _phase1_configs(args)
    }
    promoted_small = lp._select_promotions(
        phase1_runs,
        train_doc_count=int(args.phase1_train_small),
        max_configs=int(args.promote_from_small),
    )
    promoted_large = lp._select_promotions(
        phase1_runs,
        train_doc_count=int(args.phase1_train_large),
        max_configs=int(args.promote_from_large),
    )
    unique_labels: List[str] = []
    for label in [*promoted_small, *promoted_large]:
        if label not in unique_labels:
            unique_labels.append(label)

    promotions: List[tuple[int, RunConfigSpec]] = []
    for label in unique_labels:
        base = label_to_config.get(label)
        if base is None:
            continue
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


def main() -> int:
    args = _parser().parse_args()
    output_root = Path(str(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    phase1_configs = _phase1_configs(args)
    phase1_jobs = lp._build_phase_jobs(
        args,
        phase1_configs,
        seeds=(0,),
        tuning_stage="phase1",
    )
    phase1 = lp._run_phase(
        output_root=output_root / "phase1",
        jobs=phase1_jobs,
        args=args,
        manifest_payload={
            "mode": "unifiedf_push_phase1",
            "benchmark": str(args.benchmark),
            "jobs": [asdict(job) for job in phase1_jobs],
        },
    )

    phase2_configs = _promotion_configs(args, phase1["runs"])
    phase2_jobs = lp._build_phase_jobs(
        args,
        phase2_configs,
        seeds=tuple(int(seed) for seed in args.phase2_seeds),
        tuning_stage="phase2",
    )
    phase2 = lp._run_phase(
        output_root=output_root / "phase2",
        jobs=phase2_jobs,
        args=args,
        manifest_payload={
            "mode": "unifiedf_push_phase2",
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
    return (
        0
        if not phase1["result"]["failed_jobs"] and not phase2["result"]["failed_jobs"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
