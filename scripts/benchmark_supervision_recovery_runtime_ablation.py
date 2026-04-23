#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_markov_optimization_tradeoff_pipeline import (
    SUPERVISION_RECOVERY_TREE_FAMILY,
    _build_supervision_recovery_phase,
    _common_worker_env,
    _parse_args as _parse_tradeoff_args,
    _safe_float,
    _safe_int,
    _supervision_recovery_runtime_row_from_payload,
)


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / f"supervision_recovery_runtime_ablation_{stamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched supervision-recovery tree slices under baseline and optimized runtime settings."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scope", nargs="*", default=("recoverable_v4", "r12_seg10to12"))
    parser.add_argument(
        "--package",
        nargs="*",
        default=("full10", "full10_leaf_full100_internal_count100"),
    )
    parser.add_argument("--train-docs", nargs="*", type=int, default=(1024, 4096, 10240))
    parser.add_argument("--data-seeds", nargs="*", type=int, default=(0, 1))
    parser.add_argument("--tree-family", type=str, default=SUPERVISION_RECOVERY_TREE_FAMILY)
    parser.add_argument("--device-label", type=str, default="")
    parser.add_argument("--plan-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tree-batch-autotune", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--runtime-data-mode", type=str, default="resident")
    parser.add_argument("--optimized-pack-mode", type=str, default="fixed_fused")
    parser.add_argument("--optimized-bucket-mode", type=str, default="leaf_count_auto_queue")
    parser.add_argument("--baseline-pack-mode", type=str, default="structure_bucket")
    parser.add_argument("--baseline-bucket-mode", type=str, default="exact_then_bucketed")
    parser.add_argument("--runtime-tree-batch-structural-pad-limit", type=float, default=0.5)
    parser.add_argument("--runtime-tree-batch-auto-queue-min-docs", type=int, default=8)
    parser.add_argument("--runtime-tree-batch-auto-queue-min-fill-ratio", type=float, default=0.5)
    return parser.parse_args()


def _task_data_seed(task_name: str) -> int:
    token = str(task_name).rsplit("__d", 1)
    if len(token) == 2:
        return int(_safe_int(token[1], 0))
    return 0


def _task_request_path(output_path: Path) -> Path:
    return output_path.parent / "task.request"


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_base_tasks(
    *,
    config_path: Path,
    scratch_root: Path,
    device_label: str,
) -> Sequence[Any]:
    argv = [
        "--config",
        str(config_path),
        "--output-root",
        str(scratch_root),
        "--phases",
        "supervision_recovery",
        "--max-workers",
        "1",
    ]
    if str(device_label).strip():
        argv.extend(["--device-mode", "cuda", "--migs", str(device_label).strip()])
    else:
        argv.extend(["--device-mode", "auto"])
    args = _parse_tradeoff_args(argv)
    tasks, _ = _build_supervision_recovery_phase(args, scratch_root / "prepared")
    return tasks


def _selected_tree_task_payloads(
    tasks: Sequence[Any],
    *,
    scope_filter: set[str],
    package_filter: set[str],
    train_doc_filter: set[int],
    data_seed_filter: set[int],
    tree_family: str,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for task in tasks:
        metadata = dict(getattr(task, "metadata", {}) or {})
        if str(metadata.get("model_family", "")) != str(tree_family):
            continue
        if scope_filter and str(metadata.get("scope", "")) not in scope_filter:
            continue
        if package_filter and str(metadata.get("package", "")) not in package_filter:
            continue
        train_docs = int(_safe_int(metadata.get("train_docs"), 0))
        if train_doc_filter and train_docs not in train_doc_filter:
            continue
        data_seed = _task_data_seed(str(getattr(task, "name", "")))
        if data_seed_filter and data_seed not in data_seed_filter:
            continue
        request_path = _task_request_path(Path(str(getattr(task, "output_path"))))
        if not request_path.exists():
            continue
        payload = dict(_read_json(request_path))
        payload["_task_name"] = str(getattr(task, "name", ""))
        payload["_data_seed"] = int(data_seed)
        selected.append(payload)
    selected.sort(
        key=lambda item: (
            str((item.get("config") or {}).get("pipeline_supervision_recovery_scope", "")),
            int(_safe_int((item.get("config") or {}).get("train_docs"), 0)),
            str((item.get("config") or {}).get("pipeline_supervision_recovery_package", "")),
            int(_safe_int(item.get("_data_seed"), 0)),
        )
    )
    return selected


def _variant_task_payload(
    *,
    base_task: Mapping[str, Any],
    output_dir: Path,
    variant_name: str,
    pack_mode: str,
    bucket_mode: str,
    runtime_data_mode: str,
    tree_batch_autotune: bool,
    structural_pad_limit: float,
    auto_queue_min_docs: int,
    auto_queue_min_fill_ratio: float,
) -> Dict[str, Any]:
    payload = dict(base_task)
    config = dict(payload.get("config") or {})
    task_name = str(base_task.get("_task_name", payload.get("name", "task")) or "task")
    task_dir = output_dir / variant_name / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    payload["progress_path"] = str(task_dir / "progress.json")
    payload["output_json"] = str(task_dir / "summary.json")
    config["artifact_dir"] = str(task_dir / "summary_artifacts")
    config["tree_batch_pack_mode"] = str(pack_mode)
    config["gpu_runtime_bucket_mode"] = str(bucket_mode)
    config["gpu_runtime_data_mode"] = str(runtime_data_mode)
    config["tree_batch_autotune"] = bool(tree_batch_autotune)
    config["tree_batch_structural_pad_limit"] = float(structural_pad_limit)
    config["tree_batch_auto_queue_min_docs"] = int(auto_queue_min_docs)
    config["tree_batch_auto_queue_min_fill_ratio"] = float(auto_queue_min_fill_ratio)
    payload["config"] = config
    return payload


def _plan_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = args.output_dir or _default_output_dir()
    prepared_root = output_dir / "prepared"
    tasks = _build_base_tasks(
        config_path=args.config,
        scratch_root=prepared_root,
        device_label=str(args.device_label or ""),
    )
    selected = _selected_tree_task_payloads(
        tasks,
        scope_filter={str(item) for item in list(args.scope or ()) if str(item).strip()},
        package_filter={str(item) for item in list(args.package or ()) if str(item).strip()},
        train_doc_filter={int(value) for value in list(args.train_docs or ()) if int(value) > 0},
        data_seed_filter={int(value) for value in list(args.data_seeds or ()) if int(value) >= 0},
        tree_family=str(args.tree_family),
    )
    pairs = []
    for base_task in selected:
        config = dict(base_task.get("config") or {})
        identity = {
            "task_name": str(base_task.get("_task_name", base_task.get("name", ""))),
            "scope": str(config.get("pipeline_supervision_recovery_scope", "")),
            "package": str(config.get("pipeline_supervision_recovery_package", "")),
            "train_docs": int(_safe_int(config.get("train_docs"), 0)),
            "data_seed": int(_safe_int(base_task.get("_data_seed"), 0)),
            "seed": int(_safe_int(config.get("seed"), 0)),
        }
        optimized = _variant_task_payload(
            base_task=base_task,
            output_dir=output_dir,
            variant_name="optimized",
            pack_mode=str(args.optimized_pack_mode),
            bucket_mode=str(args.optimized_bucket_mode),
            runtime_data_mode=str(args.runtime_data_mode),
            tree_batch_autotune=bool(args.tree_batch_autotune),
            structural_pad_limit=float(args.runtime_tree_batch_structural_pad_limit),
            auto_queue_min_docs=int(args.runtime_tree_batch_auto_queue_min_docs),
            auto_queue_min_fill_ratio=float(args.runtime_tree_batch_auto_queue_min_fill_ratio),
        )
        baseline = _variant_task_payload(
            base_task=base_task,
            output_dir=output_dir,
            variant_name="baseline",
            pack_mode=str(args.baseline_pack_mode),
            bucket_mode=str(args.baseline_bucket_mode),
            runtime_data_mode=str(args.runtime_data_mode),
            tree_batch_autotune=bool(args.tree_batch_autotune),
            structural_pad_limit=float(args.runtime_tree_batch_structural_pad_limit),
            auto_queue_min_docs=int(args.runtime_tree_batch_auto_queue_min_docs),
            auto_queue_min_fill_ratio=float(args.runtime_tree_batch_auto_queue_min_fill_ratio),
        )
        pairs.append(
            {
                "identity": identity,
                "optimized_task": optimized,
                "baseline_task": baseline,
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "output_dir": str(output_dir),
        "device_label": str(args.device_label or ""),
        "tree_family": str(args.tree_family),
        "pairs": pairs,
    }


def _run_variant_task(task_payload: Mapping[str, Any], *, device_label: str) -> Dict[str, Any]:
    task_dir = Path(str(task_payload["output_json"])).parent
    task_request = task_dir / "task.request"
    _write_json(task_request, task_payload)
    log_path = task_dir / "run.log"
    env = _common_worker_env(str(device_label or ""))
    argv = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_markov_optimization_tradeoff_pipeline.py"),
        "--worker-task",
        str(task_request),
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(argv) + "\n\n")
        handle.flush()
        subprocess.run(
            argv,
            cwd=REPO_ROOT,
            check=True,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    summary_path = Path(str(task_payload["output_json"]))
    progress_path = Path(str(task_payload["progress_path"]))
    payload = dict(_read_json(summary_path))
    progress = dict(_read_json(progress_path)) if progress_path.exists() else {}
    runs = [run for run in list(payload.get("runs") or []) if isinstance(run, Mapping)]
    if not runs:
        raise RuntimeError(f"no run rows found in {summary_path}")
    row = _supervision_recovery_runtime_row_from_payload(
        payload,
        runs[0],
        progress=progress,
    )
    row["summary_json"] = str(summary_path)
    row["progress_json"] = str(progress_path)
    row["log_path"] = str(log_path)
    return row


def _summarize_pairs(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_identity: Dict[tuple[str, int, str, int], Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("scope_key", "")),
            int(_safe_int(row.get("train_doc_count"), 0)),
            str(row.get("package_name", "")),
            int(_safe_int(row.get("seed"), 0)),
        )
        by_identity.setdefault(key, {})[str(row.get("variant", ""))] = dict(row)
    pair_rows: List[Dict[str, Any]] = []
    for key, variants in sorted(by_identity.items()):
        optimized = dict(variants.get("optimized") or {})
        baseline = dict(variants.get("baseline") or {})
        if not optimized or not baseline:
            continue
        baseline_wall = _safe_float(baseline.get("wall_clock_s"), float("nan"))
        optimized_wall = _safe_float(optimized.get("wall_clock_s"), float("nan"))
        baseline_train = _safe_float(baseline.get("train_loop_s"), float("nan"))
        optimized_train = _safe_float(optimized.get("train_loop_s"), float("nan"))
        pair_rows.append(
            {
                "scope_key": str(key[0]),
                "train_doc_count": int(key[1]),
                "package_name": str(key[2]),
                "seed": int(key[3]),
                "baseline_wall_clock_s": float(baseline_wall),
                "optimized_wall_clock_s": float(optimized_wall),
                "baseline_train_loop_s": float(baseline_train),
                "optimized_train_loop_s": float(optimized_train),
                "wall_clock_speedup": (
                    float(baseline_wall / optimized_wall)
                    if baseline_wall > 0.0 and optimized_wall > 0.0
                    else float("nan")
                ),
                "train_loop_speedup": (
                    float(baseline_train / optimized_train)
                    if baseline_train > 0.0 and optimized_train > 0.0
                    else float("nan")
                ),
            }
        )
    grouped: Dict[tuple[str, int, str], List[Dict[str, Any]]] = {}
    for row in pair_rows:
        grouped.setdefault(
            (
                str(row.get("scope_key", "")),
                int(_safe_int(row.get("train_doc_count"), 0)),
                str(row.get("package_name", "")),
            ),
            [],
        ).append(dict(row))
    grouped_rows: List[Dict[str, Any]] = []
    for (scope_key, train_doc_count, package_name), items in sorted(grouped.items()):
        wall_values = [
            _safe_float(item.get("wall_clock_speedup"), float("nan"))
            for item in items
            if _safe_float(item.get("wall_clock_speedup"), float("nan")) > 0.0
        ]
        train_values = [
            _safe_float(item.get("train_loop_speedup"), float("nan"))
            for item in items
            if _safe_float(item.get("train_loop_speedup"), float("nan")) > 0.0
        ]
        grouped_rows.append(
            {
                "scope_key": str(scope_key),
                "train_doc_count": int(train_doc_count),
                "package_name": str(package_name),
                "n_seeds": int(len(items)),
                "median_wall_clock_speedup": float(median(wall_values)) if wall_values else float("nan"),
                "median_train_loop_speedup": float(median(train_values)) if train_values else float("nan"),
            }
        )
    overall_wall = [
        _safe_float(row.get("wall_clock_speedup"), float("nan"))
        for row in pair_rows
        if _safe_float(row.get("wall_clock_speedup"), float("nan")) > 0.0
    ]
    overall_train = [
        _safe_float(row.get("train_loop_speedup"), float("nan"))
        for row in pair_rows
        if _safe_float(row.get("train_loop_speedup"), float("nan")) > 0.0
    ]
    return {
        "pair_rows": pair_rows,
        "grouped_rows": grouped_rows,
        "median_wall_clock_speedup": float(median(overall_wall)) if overall_wall else float("nan"),
        "median_train_loop_speedup": float(median(overall_train)) if overall_train else float("nan"),
    }


def _write_markdown(summary: Mapping[str, Any], output_path: Path) -> None:
    lines = [
        "# Supervision-Recovery Runtime A/B Benchmark",
        "",
        f"Generated: `{summary.get('generated_at', '')}`",
        f"Config: `{summary.get('config', '')}`",
        "",
        "## Summary",
        (
            f"- Median wall-clock speedup: "
            f"`{_safe_float(summary.get('median_wall_clock_speedup'), float('nan')):.3f}x`."
        ),
        (
            f"- Median train-loop speedup: "
            f"`{_safe_float(summary.get('median_train_loop_speedup'), float('nan')):.3f}x`."
        ),
        "",
        "| scope | train_docs | package | seeds | median_wall_speedup | median_train_speedup |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(summary.get("grouped_rows") or []):
        lines.append(
            f"| {row.get('scope_key')} | {_safe_int(row.get('train_doc_count'))} | `{row.get('package_name')}` | "
            f"{_safe_int(row.get('n_seeds'))} | "
            f"{_safe_float(row.get('median_wall_clock_speedup'), float('nan')):.3f}x | "
            f"{_safe_float(row.get('median_train_loop_speedup'), float('nan')):.3f}x |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir or _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = _plan_from_args(args)
    _write_json(output_dir / "plan.json", plan)
    if args.plan_only:
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "plan_json": str(output_dir / "plan.json"),
                    "selected_pairs": len(list(plan.get("pairs") or [])),
                },
                indent=2,
            )
        )
        return 0

    executed_rows: List[Dict[str, Any]] = []
    for pair in list(plan.get("pairs") or []):
        baseline_row = _run_variant_task(
            pair["baseline_task"],
            device_label=str(args.device_label or ""),
        )
        baseline_row["variant"] = "baseline"
        optimized_row = _run_variant_task(
            pair["optimized_task"],
            device_label=str(args.device_label or ""),
        )
        optimized_row["variant"] = "optimized"
        executed_rows.extend([baseline_row, optimized_row])

    summary = _summarize_pairs(executed_rows)
    summary.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": str(args.config),
            "output_dir": str(output_dir),
            "rows": executed_rows,
        }
    )
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "report.md"
    _write_json(summary_path, summary)
    _write_markdown(summary, markdown_path)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "plan_json": str(output_dir / "plan.json"),
                "summary_json": str(summary_path),
                "markdown": str(markdown_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
