#!/usr/bin/env python3
"""
Build or execute a bounded optimizer-audit grid.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.config import OptimizationConfig
from src.training.optimization.performance import dataset_regime_label
from src.training.optimization.performance import summarize_optimizer_runs


def _representative_train_sizes(cfg: OptimizationConfig) -> List[int]:
    bootstrap = int(cfg.bootstrap_threshold)
    random_search = int(cfg.random_search_threshold)
    mipro = int(cfg.mipro_threshold)
    sizes = [
        max(1, bootstrap),
        max(1, min(random_search, bootstrap + 1)),
        max(1, min(mipro, random_search + 1)),
        max(1, mipro + 1),
    ]
    deduped: List[int] = []
    for size in sizes:
        if size not in deduped:
            deduped.append(size)
    return deduped


def _build_entry(
    *,
    optimizer: str,
    train_samples: int,
    seed: int,
    output_root: Path,
    task: str,
    port: int,
    budget: str,
    val_samples: int,
    test_samples: int,
) -> Dict[str, Any]:
    run_dir_name = f"{optimizer}__train_{train_samples}__seed_{seed}"
    if str(budget).strip().lower() != "medium":
        run_dir_name = f"{run_dir_name}__budget_{budget}"
    run_dir = output_root / run_dir_name
    cmd = [
        sys.executable,
        str(REPO_ROOT / "src" / "training" / "run_pipeline.py"),
        "--task",
        task,
        "--output-dir",
        str(run_dir),
        "--optimizer",
        optimizer,
        "--optimizer-budget",
        budget,
        "--train-samples",
        str(train_samples),
        "--val-samples",
        str(val_samples),
        "--test-samples",
        str(test_samples),
        "--data-seed",
        str(seed),
        "--port",
        str(port),
    ]
    return {
        "optimizer": optimizer,
        "train_samples": int(train_samples),
        "dataset_regime": dataset_regime_label(train_samples, OptimizationConfig()),
        "seed": int(seed),
        "budget": budget,
        "command": [str(part) for part in cmd],
        "command_shell": " ".join(shlex.quote(str(part)) for part in cmd),
        "run_dir": str(run_dir),
        "status": "pending",
    }


def _execute_entry(entry: Dict[str, Any]) -> None:
    cmd = [str(part) for part in entry["command"]]
    run_dir = Path(str(entry["run_dir"]))
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    entry["duration_seconds"] = float(time.time() - started)
    entry["returncode"] = int(completed.returncode)
    entry["status"] = "success" if completed.returncode == 0 else "failed"
    log_path = run_dir / "optimizer_audit_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        completed.stdout + ("\n" if completed.stdout and completed.stderr else "") + completed.stderr,
        encoding="utf-8",
    )
    entry["log_path"] = str(log_path)


def _load_cell_summaries(run_dir: Path) -> List[Dict[str, Any]]:
    final_stats_path = run_dir / "final_stats.json"
    if not final_stats_path.exists():
        return []
    payload = json.loads(final_stats_path.read_text(encoding="utf-8"))
    diag = dict(payload.get("optimizer_diagnostics") or {})
    summaries = list(diag.get("cell_summaries") or [])
    if summaries:
        return [dict(row) for row in summaries]
    runs = list(diag.get("runs") or [])
    return summarize_optimizer_runs(runs)


def _should_escalate_budgets(entry: Dict[str, Any]) -> bool:
    summaries = _load_cell_summaries(Path(str(entry["run_dir"])))
    if not summaries:
        return False
    target_optimizer = str(entry.get("optimizer", "")).strip()
    for row in summaries:
        if str(row.get("optimizer_requested", "")).strip() != target_optimizer:
            continue
        if str(row.get("classification", "")) in {"unstable_search", "objective_mismatch"}:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or stage a DSPy optimizer performance audit grid.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--task", default="manifesto_rile")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--optimizer-budget", default="medium")
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--test-samples", type=int, default=8)
    parser.add_argument("--optimizers", nargs="*", default=[
        "bootstrap",
        "bootstrap_random_search",
        "mipro",
        "gepa",
        "labeled_fewshot",
    ])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--train-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--escalate-budgets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After medium-budget runs, rerun unstable/objective-mismatch cases at "
            "light and heavy budgets."
        ),
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = OptimizationConfig()
    train_sizes = list(args.train_sizes or _representative_train_sizes(cfg))
    entries = [
        _build_entry(
            optimizer=str(optimizer),
            train_samples=int(train_size),
            seed=int(seed),
            output_root=output_root,
            task=str(args.task),
            port=int(args.port),
            budget=str(args.optimizer_budget),
            val_samples=int(args.val_samples),
            test_samples=int(args.test_samples),
        )
        for optimizer in list(args.optimizers)
        for train_size in train_sizes
        for seed in list(args.seeds)
    ]

    manifest = {
        "mode": "dry-run" if args.dry_run else "execute",
        "task": str(args.task),
        "budget": str(args.optimizer_budget),
        "entries": entries,
    }
    manifest_path = output_root / "optimizer_audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        print(str(manifest_path))
        return 0

    for entry in entries:
        _execute_entry(entry)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if str(args.optimizer_budget).strip().lower() == "medium" and bool(args.escalate_budgets):
        existing_run_dirs = {str(entry.get("run_dir", "")) for entry in entries}
        escalated_entries: List[Dict[str, Any]] = []
        for entry in list(entries):
            if str(entry.get("status", "")) != "success":
                continue
            if not _should_escalate_budgets(entry):
                continue
            for budget in ("light", "heavy"):
                rerun_entry = _build_entry(
                    optimizer=str(entry["optimizer"]),
                    train_samples=int(entry["train_samples"]),
                    seed=int(entry["seed"]),
                    output_root=output_root,
                    task=str(args.task),
                    port=int(args.port),
                    budget=budget,
                    val_samples=int(args.val_samples),
                    test_samples=int(args.test_samples),
                )
                rerun_dir = str(rerun_entry.get("run_dir", ""))
                if rerun_dir in existing_run_dirs:
                    continue
                existing_run_dirs.add(rerun_dir)
                escalated_entries.append(rerun_entry)

        if escalated_entries:
            manifest["escalated_from_medium"] = True
            manifest["entries"].extend(escalated_entries)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            for entry in escalated_entries:
                _execute_entry(entry)
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
