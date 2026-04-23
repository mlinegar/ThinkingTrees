#!/usr/bin/env python3
"""Run the staged root-only tree/FNO parity diagnosis workflow."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.report_tree_root_only_parity_pdf import (
    ROOT_STAGE_NAMES,
    classify_root_only_diagnosis,
    generate_root_only_parity_report,
    load_historical_reference,
    load_stage_result,
)


CAPACITY_SCRIPT = REPO_ROOT / "scripts" / "run_tree_neural_full_doc_mig.py"


@dataclass(frozen=True)
class StageSpec:
    stage_name: str
    stage_title: str
    output_root: Path
    benchmark: str
    capacity_profile: str
    command: Sequence[str]
    deferred: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the staged root-only tree/FNO parity diagnosis workflow."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--historical-summary", type=Path, required=True)
    parser.add_argument(
        "--search-mode",
        choices=("exploratory", "full"),
        default="exploratory",
    )
    parser.add_argument("--benchmark", type=str, default="recoverable_v4")
    parser.add_argument(
        "--structural-benchmark",
        type=str,
        default="structural_core_v1::r12_seg10to12",
    )
    parser.add_argument("--train-doc-count", type=int, default=10240)
    parser.add_argument("--screen-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--locked-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--capacity-sweep-gap-ceiling", type=float, default=0.003)
    parser.add_argument("--structural-confirmation-gap-ceiling", type=float, default=0.002)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--mig-uuids", type=str, default="")
    parser.add_argument("--prepared-data-root", type=str, default="")
    parser.add_argument(
        "--prepared-data-allow-create",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--tree-exact-eval-max-docs", type=int, default=64)
    parser.add_argument("--screen-device-order", type=str, default="interleave_by_physical_gpu")
    parser.add_argument("--screen-max-concurrent-per-physical-gpu", type=int, default=0)
    parser.add_argument("--gpu-runtime-data-mode", type=str, default="resident")
    parser.add_argument("--gpu-runtime-bucket-mode", type=str, default="leaf_count_auto_queue")
    parser.add_argument(
        "--gpu-runtime-allow-multi-worker-screen",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--gpu-runtime-capacity-workers-per-mig", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-structural-confirmation",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--plan-only", action="store_true", default=False)
    parser.add_argument("--report-only", action="store_true", default=False)
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _summary_path(stage_root: Path) -> Path:
    return stage_root / "tree_fno_capacity_locked_summary.json"


def _common_capacity_command_args(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(CAPACITY_SCRIPT),
        "capacity",
        "--train-doc-count",
        str(int(args.train_doc_count)),
        "--top-k",
        str(int(args.top_k)),
        "--screen-device-order",
        str(args.screen_device_order),
        "--screen-max-concurrent-per-physical-gpu",
        str(int(args.screen_max_concurrent_per_physical_gpu)),
        "--gpu-runtime-data-mode",
        str(args.gpu_runtime_data_mode),
        "--gpu-runtime-bucket-mode",
        str(args.gpu_runtime_bucket_mode),
        "--gpu-runtime-capacity-workers-per-mig",
        str(int(args.gpu_runtime_capacity_workers_per_mig)),
        "--torch-threads",
        str(int(args.torch_threads)),
        "--tree-exact-eval-max-docs",
        str(int(args.tree_exact_eval_max_docs)),
    ]
    screen_seeds = _resolved_screen_seeds(args)
    locked_seeds = _resolved_locked_seeds(args)
    if list(screen_seeds):
        cmd.extend(["--screen-seeds", *[str(int(seed)) for seed in screen_seeds]])
    if list(locked_seeds):
        cmd.extend(["--locked-seeds", *[str(int(seed)) for seed in locked_seeds]])
    if str(args.mig_uuids).strip():
        cmd.extend(["--mig-uuids", str(args.mig_uuids).strip()])
    if str(args.prepared_data_root).strip():
        cmd.extend(["--prepared-data-root", str(args.prepared_data_root).strip()])
    if not bool(args.prepared_data_allow_create):
        cmd.append("--no-prepared-data-allow-create")
    if not bool(args.gpu_runtime_allow_multi_worker_screen):
        cmd.append("--no-gpu-runtime-allow-multi-worker-screen")
    if not bool(args.use_cuda):
        cmd.append("--no-use-cuda")
    return cmd


def _resolved_screen_seeds(args: argparse.Namespace) -> Sequence[int]:
    explicit = getattr(args, "screen_seeds", None)
    if explicit:
        return tuple(int(seed) for seed in explicit)
    if str(getattr(args, "search_mode", "exploratory")) == "full":
        return (0, 1, 2)
    return (0,)


def _resolved_locked_seeds(args: argparse.Namespace) -> Sequence[int]:
    explicit = getattr(args, "locked_seeds", None)
    if explicit:
        return tuple(int(seed) for seed in explicit)
    if str(getattr(args, "search_mode", "exploratory")) == "full":
        return (0, 1, 2, 3, 4)
    return (0,)


def _capacity_stage_command(
    args: argparse.Namespace,
    *,
    output_root: Path,
    benchmark: str,
    capacity_profile: str,
    width_values: Sequence[int] | None = None,
    mode_values: Sequence[int] | None = None,
    layer_values: Sequence[int] | None = None,
    state_dims: Sequence[int] | None = None,
    hidden_dims: Sequence[int] | None = None,
    n_epochs: Sequence[int] | None = None,
    tree_training_schedules: Sequence[str] | None = None,
    tree_checkpoint_metrics: Sequence[str] | None = None,
    tree_stage1_checkpoint_metrics: Sequence[str] | None = None,
    tree_stage1_root_weights: Sequence[float] | None = None,
    slot_counts: Sequence[int] | None = None,
    fixed_leaf_tokens: Sequence[int | None] | None = None,
) -> List[str]:
    cmd = _common_capacity_command_args(args)
    cmd.extend(
        [
            "--output-root",
            str(output_root),
            "--benchmark",
            str(benchmark),
            "--capacity-profile",
            str(capacity_profile),
        ]
    )

    def _extend(flag: str, values: Sequence[Any] | None) -> None:
        if values:
            cmd.extend([flag, *[str(value) for value in values]])

    _extend("--capacity-widths", width_values)
    _extend("--capacity-modes", mode_values)
    _extend("--capacity-layers", layer_values)
    _extend("--capacity-state-dims", state_dims)
    _extend("--capacity-hidden-dims", hidden_dims)
    _extend("--capacity-n-epochs", n_epochs)
    _extend("--capacity-tree-training-schedules", tree_training_schedules)
    _extend("--capacity-tree-checkpoint-metrics", tree_checkpoint_metrics)
    _extend(
        "--capacity-tree-stage1-checkpoint-metrics",
        tree_stage1_checkpoint_metrics,
    )
    _extend("--capacity-tree-stage1-root-weights", tree_stage1_root_weights)
    _extend("--capacity-slot-counts", slot_counts)
    if fixed_leaf_tokens:
        _extend(
            "--capacity-fixed-leaf-tokens",
            [value for value in fixed_leaf_tokens if value is not None],
        )
    return cmd


def build_stage_plan(args: argparse.Namespace) -> List[StageSpec]:
    output_root = Path(args.output_root)
    plan = [
        StageSpec(
            stage_name="historical_replay",
            stage_title="Historical Replay",
            output_root=output_root / "01_historical_replay",
            benchmark=str(args.benchmark),
            capacity_profile="root_only_parity_historical_replay",
            command=_capacity_stage_command(
                args,
                output_root=output_root / "01_historical_replay",
                benchmark=str(args.benchmark),
                capacity_profile="root_only_parity_historical_replay",
            ),
        ),
        StageSpec(
            stage_name="optimization_fairness",
            stage_title="Optimization-Fairness Fix",
            output_root=output_root / "02_optimization_fairness",
            benchmark=str(args.benchmark),
            capacity_profile="root_only_parity_optimization_fairness",
            command=_capacity_stage_command(
                args,
                output_root=output_root / "02_optimization_fairness",
                benchmark=str(args.benchmark),
                capacity_profile="root_only_parity_optimization_fairness",
            ),
        ),
        StageSpec(
            stage_name="capacity_fairness",
            stage_title="Capacity-Fairness Fix",
            output_root=output_root / "03_capacity_fairness",
            benchmark=str(args.benchmark),
            capacity_profile="root_only_parity_capacity_fairness",
            command=_capacity_stage_command(
                args,
                output_root=output_root / "03_capacity_fairness",
                benchmark=str(args.benchmark),
                capacity_profile="root_only_parity_capacity_fairness",
            ),
        ),
        StageSpec(
            stage_name="matched_root",
            stage_title="Combined Matched-Root Recipe",
            output_root=output_root / "04_matched_root",
            benchmark=str(args.benchmark),
            capacity_profile="root_only_parity_matched_root",
            command=_capacity_stage_command(
                args,
                output_root=output_root / "04_matched_root",
                benchmark=str(args.benchmark),
                capacity_profile="root_only_parity_matched_root",
            ),
        ),
        StageSpec(
            stage_name="capacity_sweep",
            stage_title="Matched-Root Capacity Sweep",
            output_root=output_root / "05_capacity_sweep",
            benchmark=str(args.benchmark),
            capacity_profile="root_only_parity_matched_root",
            command=_capacity_stage_command(
                args,
                output_root=output_root / "05_capacity_sweep",
                benchmark=str(args.benchmark),
                capacity_profile="root_only_parity_matched_root",
                width_values=(64, 128, 256),
                mode_values=(2, 4, 8),
                layer_values=(2, 4, 6),
            ),
        ),
        StageSpec(
            stage_name="representation_sweep",
            stage_title="Representation Bottleneck Sweep",
            output_root=output_root / "06_representation_sweep",
            benchmark=str(args.benchmark),
            capacity_profile="root_only_parity_matched_root",
            command=(),
            deferred=True,
        ),
        StageSpec(
            stage_name="structural_confirmation",
            stage_title="Structural Confirmation",
            output_root=output_root / "07_structural_confirmation",
            benchmark=str(args.structural_benchmark),
            capacity_profile="root_only_parity_structural_matched_root",
            command=(),
            deferred=True,
        ),
    ]
    return plan


def _stage_by_name(stages: Sequence[StageSpec]) -> Dict[str, StageSpec]:
    return {stage.stage_name: stage for stage in stages}


def _best_recoverable_result(stage_results: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    best: Mapping[str, Any] = {}
    for stage_name in ROOT_STAGE_NAMES:
        result = dict(stage_results.get(stage_name) or {})
        if not result:
            continue
        value = result.get("test_root_mae")
        if not best or float(value) < float(best.get("test_root_mae", float("inf"))):
            best = result
    return best


def _dynamic_representation_stage(
    args: argparse.Namespace,
    *,
    best_result: Mapping[str, Any],
) -> StageSpec:
    output_root = Path(args.output_root) / "06_representation_sweep"
    return StageSpec(
        stage_name="representation_sweep",
        stage_title="Representation Bottleneck Sweep",
        output_root=output_root,
        benchmark=str(args.benchmark),
        capacity_profile="root_only_parity_matched_root",
        command=_capacity_stage_command(
            args,
            output_root=output_root,
            benchmark=str(args.benchmark),
            capacity_profile="root_only_parity_matched_root",
            width_values=(int(best_result.get("tree_leaf_fno_width", 128) or 128),),
            mode_values=(int(best_result.get("tree_leaf_fno_n_modes", 4) or 4),),
            layer_values=(int(best_result.get("tree_leaf_fno_n_layers", 4) or 4),),
            state_dims=(int(best_result.get("state_dim", 256) or 256),),
            hidden_dims=(int(best_result.get("hidden_dim", 1024) or 1024),),
            n_epochs=(int(best_result.get("n_epochs", 128) or 128),),
            tree_training_schedules=(str(best_result.get("tree_training_schedule", "single_stage") or "single_stage"),),
            tree_checkpoint_metrics=(str(best_result.get("tree_checkpoint_metric", "val_root_mae") or "val_root_mae"),),
            tree_stage1_checkpoint_metrics=(str(best_result.get("tree_stage1_checkpoint_metric", "val_root_mae") or "val_root_mae"),),
            tree_stage1_root_weights=(float(best_result.get("tree_stage1_root_weight", 1.0) or 1.0),),
            slot_counts=(4, 8),
            fixed_leaf_tokens=(16, 32),
        ),
    )


def _dynamic_structural_stage(
    args: argparse.Namespace,
    *,
    best_result: Mapping[str, Any],
) -> StageSpec:
    output_root = Path(args.output_root) / "07_structural_confirmation"
    fixed_leaf_tokens = best_result.get("fixed_leaf_tokens")
    fixed_leaf_values: Sequence[int | None] | None = (
        None if fixed_leaf_tokens is None else (int(fixed_leaf_tokens),)
    )
    return StageSpec(
        stage_name="structural_confirmation",
        stage_title="Structural Confirmation",
        output_root=output_root,
        benchmark=str(args.structural_benchmark),
        capacity_profile="root_only_parity_structural_matched_root",
        command=_capacity_stage_command(
            args,
            output_root=output_root,
            benchmark=str(args.structural_benchmark),
            capacity_profile="root_only_parity_structural_matched_root",
            width_values=(int(best_result.get("tree_leaf_fno_width", 128) or 128),),
            mode_values=(int(best_result.get("tree_leaf_fno_n_modes", 4) or 4),),
            layer_values=(int(best_result.get("tree_leaf_fno_n_layers", 4) or 4),),
            state_dims=(int(best_result.get("state_dim", 256) or 256),),
            hidden_dims=(int(best_result.get("hidden_dim", 1024) or 1024),),
            n_epochs=(int(best_result.get("n_epochs", 128) or 128),),
            tree_training_schedules=(str(best_result.get("tree_training_schedule", "single_stage") or "single_stage"),),
            tree_checkpoint_metrics=(str(best_result.get("tree_checkpoint_metric", "val_root_mae") or "val_root_mae"),),
            tree_stage1_checkpoint_metrics=(str(best_result.get("tree_stage1_checkpoint_metric", "val_root_mae") or "val_root_mae"),),
            tree_stage1_root_weights=(float(best_result.get("tree_stage1_root_weight", 1.0) or 1.0),),
            slot_counts=(int(best_result.get("slot_count", 4) or 4),),
            fixed_leaf_tokens=fixed_leaf_values,
        ),
    )


def _write_status(
    args: argparse.Namespace,
    *,
    plan: Sequence[StageSpec],
    stage_results: Mapping[str, Mapping[str, Any]],
    report_result: Mapping[str, Any] | None = None,
    execution_notes: Sequence[str] | None = None,
) -> None:
    payload = {
        "output_root": str(Path(args.output_root)),
        "historical_summary": str(Path(args.historical_summary)),
        "search_mode": str(args.search_mode),
        "resolved_screen_seeds": [int(seed) for seed in _resolved_screen_seeds(args)],
        "resolved_locked_seeds": [int(seed) for seed in _resolved_locked_seeds(args)],
        "threshold": float(args.threshold),
        "capacity_sweep_gap_ceiling": float(args.capacity_sweep_gap_ceiling),
        "structural_confirmation_gap_ceiling": float(
            args.structural_confirmation_gap_ceiling
        ),
        "stage_plan": [
            {
                **asdict(stage),
                "output_root": str(stage.output_root),
                "command": list(stage.command),
            }
            for stage in plan
        ],
        "stage_results": dict(stage_results),
        "best_recoverable_stage": dict(_best_recoverable_result(stage_results) or {}),
        "report_result": dict(report_result or {}),
        "execution_notes": list(execution_notes or []),
    }
    _write_json(Path(args.output_root) / "diagnosis_status.json", payload)


def _run_stage(stage: StageSpec, *, resume: bool) -> None:
    stage.output_root.mkdir(parents=True, exist_ok=True)
    if bool(resume) and _summary_path(stage.output_root).exists():
        return
    completed = subprocess.run(
        list(stage.command),
        cwd=str(REPO_ROOT),
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"stage {stage.stage_name!r} failed with return code {completed.returncode}"
        )
    if not _summary_path(stage.output_root).exists():
        raise SystemExit(
            f"stage {stage.stage_name!r} completed without writing {_summary_path(stage.output_root)}"
        )


def main() -> int:
    from scripts._markov_report_archive import archived_report_exit

    return archived_report_exit(
        legacy_script="scripts/run_tree_root_only_parity_diagnosis.py",
        replacements=(
            "python3 scripts/run_markov_supervision_recovery_parity_grid.py --help",
            "python3 scripts/report_markov_optimization_tradeoffs.py --summary-json <tradeoff_pipeline/tradeoff_report/summary.json>",
            "python3 scripts/report_markov_parity_self_contained.py --simulation-root <parity_root>",
        ),
        note=(
            "The root-only parity diagnosis workflow is archived. Use the v3 parity-grid "
            "and tradeoff report path instead."
        ),
    )

    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    recoverable_reference = load_historical_reference(
        Path(args.historical_summary),
        scope="recoverable",
        train_doc_count=int(args.train_doc_count),
    )
    plan = build_stage_plan(args)
    _write_json(
        output_root / "diagnosis_plan.json",
        {
            "output_root": str(output_root),
            "search_mode": str(args.search_mode),
            "resolved_screen_seeds": [int(seed) for seed in _resolved_screen_seeds(args)],
            "resolved_locked_seeds": [int(seed) for seed in _resolved_locked_seeds(args)],
            "threshold": float(args.threshold),
            "capacity_sweep_gap_ceiling": float(args.capacity_sweep_gap_ceiling),
            "structural_confirmation_gap_ceiling": float(
                args.structural_confirmation_gap_ceiling
            ),
            "recoverable_fno_reference_mae": float(
                recoverable_reference.best_fno_test_root_mae
            ),
            "stage_plan": [
                {
                    **asdict(stage),
                    "output_root": str(stage.output_root),
                    "command": list(stage.command),
                }
                for stage in plan
            ],
        },
    )
    if bool(args.plan_only):
        print(str(output_root / "diagnosis_plan.json"))
        return 0

    stage_map = _stage_by_name(plan)
    stage_results: Dict[str, Mapping[str, Any]] = {}
    execution_notes: List[str] = []
    _write_status(
        args,
        plan=plan,
        stage_results=stage_results,
        execution_notes=execution_notes,
    )

    if not bool(args.report_only):
        for stage_name in (
            "historical_replay",
            "optimization_fairness",
            "capacity_fairness",
            "matched_root",
        ):
            stage = stage_map[stage_name]
            _run_stage(stage, resume=bool(args.resume))
            result = load_stage_result(
                stage_name,
                stage.output_root,
                best_fno_mae=float(recoverable_reference.best_fno_test_root_mae),
            )
            stage_results[stage_name] = asdict(result) if result is not None else {}
            _write_status(
                args,
                plan=plan,
                stage_results=stage_results,
                execution_notes=execution_notes,
            )

        matched = dict(stage_results.get("matched_root") or {})
        matched_gap = float(matched.get("gap_vs_best_fno", float("inf")))
        if matched and matched_gap > float(args.threshold):
            if (
                str(args.search_mode) == "exploratory"
                and matched_gap > float(args.capacity_sweep_gap_ceiling)
            ):
                execution_notes.append(
                    "Exploratory mode early stop: matched-root gap exceeded the "
                    "capacity-sweep ceiling, so no recoverable follow-up sweeps were run."
                )
            else:
                stage = stage_map["capacity_sweep"]
                _run_stage(stage, resume=bool(args.resume))
                result = load_stage_result(
                    "capacity_sweep",
                    stage.output_root,
                    best_fno_mae=float(recoverable_reference.best_fno_test_root_mae),
                )
                stage_results["capacity_sweep"] = (
                    asdict(result) if result is not None else {}
                )
                _write_status(
                    args,
                    plan=plan,
                    stage_results=stage_results,
                    execution_notes=execution_notes,
                )

            classification = classify_root_only_diagnosis(
                {
                    key: (
                        None
                        if not stage_results.get(key)
                        else load_stage_result(
                            key,
                            Path(stage_results[key]["root"]),
                            best_fno_mae=float(recoverable_reference.best_fno_test_root_mae),
                        )
                    )
                    for key in ROOT_STAGE_NAMES
                },
                threshold=float(args.threshold),
            )
            if (
                str(args.search_mode) == "full"
                and classification == "root_only_architecture_gap_persists"
            ):
                best_result = dict(_best_recoverable_result(stage_results) or {})
                if best_result:
                    stage = _dynamic_representation_stage(args, best_result=best_result)
                    _run_stage(stage, resume=bool(args.resume))
                    result = load_stage_result(
                        "representation_sweep",
                        stage.output_root,
                        best_fno_mae=float(recoverable_reference.best_fno_test_root_mae),
                    )
                    stage_results["representation_sweep"] = (
                        asdict(result) if result is not None else {}
                    )
                    _write_status(
                        args,
                        plan=plan,
                        stage_results=stage_results,
                        execution_notes=execution_notes,
                    )
            elif (
                str(args.search_mode) == "exploratory"
                and classification == "root_only_architecture_gap_persists"
            ):
                execution_notes.append(
                    "Exploratory mode skipped the representation bottleneck sweep; "
                    "promote to --search-mode full only if the recoverable root-only "
                    "recipe is still worth chasing after the cheaper stages."
                )

        if not bool(args.skip_structural_confirmation):
            best_result = dict(_best_recoverable_result(stage_results) or {})
            best_gap = float(best_result.get("gap_vs_best_fno", float("inf")))
            should_run_structural = bool(best_result)
            if (
                should_run_structural
                and str(args.search_mode) == "exploratory"
                and best_gap > float(args.structural_confirmation_gap_ceiling)
            ):
                should_run_structural = False
                execution_notes.append(
                    "Exploratory mode skipped structural confirmation because the best "
                    "recoverable root-only gap is still too large."
                )
            if should_run_structural:
                stage = _dynamic_structural_stage(args, best_result=best_result)
                _run_stage(stage, resume=bool(args.resume))
                structural_reference = load_historical_reference(
                    Path(args.historical_summary),
                    scope="structural",
                    train_doc_count=int(args.train_doc_count),
                )
                result = load_stage_result(
                    "structural_confirmation",
                    stage.output_root,
                    best_fno_mae=float(structural_reference.best_fno_test_root_mae),
                )
                stage_results["structural_confirmation"] = (
                    asdict(result) if result is not None else {}
                )
                _write_status(
                    args,
                    plan=plan,
                    stage_results=stage_results,
                    execution_notes=execution_notes,
                )

    report_result = generate_root_only_parity_report(
        historical_summary=Path(args.historical_summary),
        output_dir=output_root / "report",
        train_doc_count=int(args.train_doc_count),
        threshold=float(args.threshold),
        stage_roots={
            "historical_replay": stage_map["historical_replay"].output_root,
            "optimization_fairness": stage_map["optimization_fairness"].output_root,
            "capacity_fairness": stage_map["capacity_fairness"].output_root,
            "matched_root": stage_map["matched_root"].output_root,
            "capacity_sweep": stage_map["capacity_sweep"].output_root,
            "representation_sweep": output_root / "06_representation_sweep",
            "structural_confirmation": output_root / "07_structural_confirmation",
        },
    )
    _write_status(
        args,
        plan=plan,
        stage_results=stage_results,
        report_result=report_result,
        execution_notes=execution_notes,
    )
    print(str(report_result["report_pdf"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
