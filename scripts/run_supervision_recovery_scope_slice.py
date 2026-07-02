from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_markov_optimization_tradeoff_pipeline as pipeline
from src.experiments.scheduler import (
    SchedulerConfig,
    SchedulerItem,
    SchedulerRunError,
    run_scheduler,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a filtered supervision-recovery scope slice with the standard "
            "Markov GPU scheduler."
        )
    )
    parser.add_argument("--config", required=True, help="Tradeoff pipeline TOML config.")
    parser.add_argument("--output-root", required=True, help="Fresh output root.")
    parser.add_argument(
        "--scope",
        required=True,
        help="Exact supervision_recovery scope key to keep, e.g. r12_seg10to12.",
    )
    parser.add_argument(
        "--model-families",
        default="tree_neural,fno",
        help="Comma-separated model families to keep.",
    )
    parser.add_argument(
        "--packages",
        default="",
        help=(
            "Optional comma-separated supervision-recovery packages to keep. "
            "When omitted, the default package ladder is used."
        ),
    )
    parser.add_argument("--device-mode", default="cuda")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument(
        "--tree-stage1-artifact-root",
        default="",
        help="Optional stage-1 artifact root override.",
    )
    parser.add_argument(
        "--no-tree-stage1-resume-if-available",
        action="store_true",
        help="Disable stage-1 resume reuse for a clean rerun.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    pipeline_args = [
        "--config",
        str(Path(args.config).resolve()),
        "--output-root",
        str(output_root),
        "--phases",
        "supervision_recovery",
        "--device-mode",
        str(args.device_mode),
        "--max-workers",
        str(int(args.max_workers)),
    ]
    stage1_root = str(args.tree_stage1_artifact_root or "").strip()
    if stage1_root:
        pipeline_args.extend(
            ["--tree-stage1-artifact-root", str(Path(stage1_root).resolve())]
        )
    if bool(args.no_tree_stage1_resume_if_available):
        pipeline_args.append("--no-tree-stage1-resume-if-available")

    tradeoff_args = pipeline._parse_args(pipeline_args)
    attempt_id = pipeline._new_attempt_id()
    attempt_root = pipeline._phase_attempt_root(output_root, "supervision_recovery", attempt_id)
    explicit_package_filter = [
        part.strip() for part in str(args.packages or "").split(",") if part.strip()
    ]
    selected_package_order = (
        pipeline._resolve_supervision_recovery_package_order(explicit_package_filter)
        if explicit_package_filter
        else pipeline._supervision_recovery_package_order_from_args(tradeoff_args)
    )
    tasks, phase_root = pipeline._build_supervision_recovery_phase(
        tradeoff_args,
        attempt_root,
        package_order=selected_package_order,
    )

    allowed_families = {
        part.strip() for part in str(args.model_families or "").split(",") if part.strip()
    }
    selected = []
    for task in tasks:
        metadata = dict(task.metadata)
        if str(metadata.get("scope", "") or "").strip() != str(args.scope).strip():
            continue
        if allowed_families and str(metadata.get("model_family", "") or "").strip() not in allowed_families:
            continue
        if (
            selected_package_order
            and str(metadata.get("package", "") or "").strip() not in set(selected_package_order)
        ):
            continue
        selected.append(task)

    selected.sort(
        key=lambda task: (
            int(task.metadata.get("train_docs", 0)),
            str(task.metadata.get("model_family", "")),
            str(task.metadata.get("package", "")),
            str(task.name),
        )
    )

    devices = tuple(pipeline._resolve_devices(tradeoff_args))
    phase_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "scheduler_status.json"
    summary_path = phase_root / "summary.json"
    manifest_path = output_root / "scope_slice_manifest.json"
    manifest = {
        "output_root": str(output_root),
        "config": str(Path(args.config).resolve()),
        "attempt_id": attempt_id,
        "attempt_root": str(attempt_root),
        "phase_root": str(phase_root),
        "status_path": str(status_path),
        "devices": list(devices),
        "scope": str(args.scope),
        "model_families": sorted(allowed_families),
        "package_order": list(selected_package_order),
        "selected_task_count": len(selected),
        "selected_counts_by_family": dict(
            Counter(str(task.metadata.get("model_family", "")) for task in selected)
        ),
        "selected_counts_by_package": dict(
            Counter(str(task.metadata.get("package", "")) for task in selected)
        ),
        "selected_task_names": [str(task.name) for task in selected],
    }
    pipeline._write_json(manifest_path, manifest)

    gpu_items = [
        pipeline._scheduler_item_from_subprocess_task("supervision_recovery", task)
        for task in selected
    ]
    dep_ids = [str(item.item_id) for item in gpu_items]

    def _reduce() -> dict[str, object]:
        payloads = pipeline._load_ops_payloads(phase_root / "raw")
        summary = pipeline._aggregate_supervision_recovery_from_payloads(
            payloads,
            tree_family=str(
                getattr(
                    tradeoff_args,
                    "supervision_recovery_tree_family",
                    pipeline.PRESET_DEFAULTS[str(tradeoff_args.preset)][
                        "supervision_recovery_tree_family"
                    ],
                )
            ),
            structural_cell=str(
                getattr(
                    tradeoff_args,
                    "supervision_recovery_structural_cell",
                    pipeline.PRESET_DEFAULTS[str(tradeoff_args.preset)][
                        "supervision_recovery_structural_cell"
                    ],
                )
            ),
            package_order=selected_package_order,
        )
        pipeline._write_json(summary_path, summary)
        return {"status": "completed", "summary_path": str(summary_path)}

    items = list(gpu_items)
    items.append(
        SchedulerItem(
            item_id="supervision_recovery::reduce",
            phase="supervision_recovery",
            kind="cpu_callback",
            deps=tuple(dep_ids),
            expected_outputs=(str(summary_path),),
            callback=_reduce,
            reuse_existing=False,
        )
    )

    config = SchedulerConfig(
        devices=devices,
        max_gpu_items_per_mig=int(tradeoff_args.max_gpu_items_per_mig),
        cleanup_stale_children=bool(tradeoff_args.cleanup_stale_children),
        root_markers=(str(output_root),),
        status_path=str(status_path),
    )
    scheduler_summary_path = output_root / "scheduler_summary.json"
    try:
        summary = run_scheduler(items, config=config)
    except SchedulerRunError as exc:
        pipeline._write_json(scheduler_summary_path, dict(exc.summary))
        raise
    except Exception as exc:  # pragma: no cover - launcher failure path
        pipeline._write_json(scheduler_summary_path, {"error": str(exc)})
        raise
    pipeline._write_json(scheduler_summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
