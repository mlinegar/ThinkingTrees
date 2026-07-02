#!/usr/bin/env python3
"""
Run standardized method-comparison profiles for ThinkingTrees.

This script executes multiple pipeline profiles sequentially, writes
`method_compare_manifest.json`, and (unless --dry-run) invokes
`scripts/report_method_compare.py` to produce summary outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.contracts import (
    MethodRef,
    ResultRow,
    benchmark_ref_from_parts,
    method_ref_from_parts,
)
from src.experiments.roles import (
    ROLE_SCORER,
    ROLE_STATE_MODEL,
    chat_role_ref,
    embedder_role_ref,
    metadata_with_roles,
    oracle_ref,
    state_model_role_ref,
)
from src.experiments.sidecars import write_canonical_sidecars


DEFAULT_MODE = "fast-smoke"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _mode_defaults(mode: str) -> Dict[str, Any]:
    if mode != "fast-smoke":
        raise ValueError(f"Unsupported mode: {mode}")
    return {
        "train_samples": 30,
        "val_samples": 15,
        "test_samples": 15,
        "n_iterations": 1,
        "optimizer": "bootstrap_random_search",
        "optimizer_budget": "light",
        "max_chunk_chars": 8000,
        "data_seed": 42,
        "dynamic_gpu": True,
    }


def _profiles() -> Dict[str, List[str]]:
    return {
        "baseline_llm": [
            "--no-adaptive-embedding-proxy",
            "--no-train-neural-operators",
            "--no-train-generator",
            "--no-enable-unified-training",
        ],
        "embedding_proxy_ridge": [
            "--adaptive-chunking",
            "--adaptive-embedding-proxy",
            "--adaptive-embedding-head-method",
            "ridge",
            "--no-train-neural-operators",
            "--no-train-generator",
            "--no-enable-unified-training",
        ],
        "neural_operator_hybrid": [
            "--no-adaptive-embedding-proxy",
            "--train-neural-operators",
            "--neural-operators-which",
            "both",
            "--neural-operators-ctreepo-args",
            "--pilot --device cuda",
            "--neural-operators-mergeable-args",
            "--device cuda",
            "--neural-operators-auto-wire-representation",
            "--hybrid-oracle-seeded-ensemble",
            "--hybrid-seed-llm-min-weight",
            "0.20",
            "--hybrid-seed-llm-max-weight",
            "0.55",
            "--hybrid-operator-boost",
            "1.40",
            "--no-train-generator",
            "--no-enable-unified-training",
        ],
        "generator_lora_dpo": [
            "--train-generator",
            "--generator-method",
            "dpo",
            "--generator-use-lora",
            "--generator-min-preferences",
            "20",
            "--no-adaptive-embedding-proxy",
            "--no-train-neural-operators",
            "--no-enable-unified-training",
        ],
    }


def _profile_method_ref(profile_name: str, *, task: str) -> MethodRef:
    roles: Dict[str, Any] = {
        ROLE_SCORER: chat_role_ref(role=ROLE_SCORER, metadata={"task": task})
    }
    if profile_name == "embedding_proxy_ridge":
        roles["embedder"] = embedder_role_ref(engine="local", model="embedding_proxy")
    if profile_name == "neural_operator_hybrid":
        roles[ROLE_STATE_MODEL] = state_model_role_ref(
            engine="pytorch",
            model="neural_operator_hybrid",
            execution_mode="training_or_inference",
        )
    if profile_name == "generator_lora_dpo":
        roles[ROLE_SCORER] = chat_role_ref(
            role=ROLE_SCORER,
            model="generator_lora_dpo",
            metadata={"task": task, "adapter": "lora"},
        )
    return method_ref_from_parts(
        family=str(profile_name),
        variant="method_compare_profile",
        adapter="method_compare",
        metadata=metadata_with_roles(
            {"profile": profile_name, "task": task},
            roles=roles,
            oracle=oracle_ref(kind="task_labels", source=task),
        ),
    )


def _build_base_cmd(
    *,
    python_exe: str,
    run_dir: Path,
    task: str,
    dataset: str,
    mode_defaults: Dict[str, Any],
    resume: bool,
    extra_args: List[str],
) -> List[str]:
    cmd = [
        python_exe,
        "-m",
        "src.training.run_pipeline",
        "--task",
        task,
        "--dataset",
        dataset,
        "--train-samples",
        str(mode_defaults["train_samples"]),
        "--val-samples",
        str(mode_defaults["val_samples"]),
        "--test-samples",
        str(mode_defaults["test_samples"]),
        "--n-iterations",
        str(mode_defaults["n_iterations"]),
        "--optimizer",
        str(mode_defaults["optimizer"]),
        "--optimizer-budget",
        str(mode_defaults["optimizer_budget"]),
        "--max-chunk-chars",
        str(mode_defaults["max_chunk_chars"]),
        "--data-seed",
        str(mode_defaults["data_seed"]),
        "--output-dir",
        str(run_dir),
    ]
    if bool(mode_defaults.get("dynamic_gpu", True)):
        cmd.append("--dynamic-gpu")
    else:
        cmd.append("--no-dynamic-gpu")
    if resume:
        cmd.append("--resume")
    cmd.extend(extra_args)
    return cmd


def parse_args() -> argparse.Namespace:
    profile_names = list(_profiles().keys())
    parser = argparse.ArgumentParser(description="Run method-comparison profiles.")
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=[DEFAULT_MODE])
    parser.add_argument("--task", default="manifesto_rile")
    parser.add_argument("--dataset", default="manifesto")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "outputs" / f"method_compare_{_timestamp()}"),
        help="Directory containing per-profile run dirs and comparison artifacts.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=profile_names,
        choices=profile_names,
        help="Subset of profiles to execute.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for pipeline runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--resume", action="store_true", help="Pass --resume to each profile run.")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after first failed profile run.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument appended to every profile run (repeatable).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    defaults = _mode_defaults(args.mode)
    profile_specs = _profiles()
    manifest_path = output_root / "method_compare_manifest.json"

    manifest: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "mode": args.mode,
        "task": args.task,
        "dataset": args.dataset,
        "base_defaults": defaults,
        "profiles_requested": list(args.profiles),
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
        "fail_fast": bool(args.fail_fast),
        "entries": [],
    }

    overall_ok = True
    for profile_name in args.profiles:
        profile_args = profile_specs[profile_name]
        run_dir = output_root / profile_name
        run_dir.mkdir(parents=True, exist_ok=True)
        run_log = run_dir / "compare_run.log"
        cmd = _build_base_cmd(
            python_exe=args.python,
            run_dir=run_dir,
            task=args.task,
            dataset=args.dataset,
            mode_defaults=defaults,
            resume=bool(args.resume),
            extra_args=list(args.extra_arg or []),
        )
        cmd.extend(profile_args)

        entry: Dict[str, Any] = {
            "profile": profile_name,
            "run_dir": str(run_dir),
            "log_path": str(run_log),
            "command": cmd,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        if args.dry_run:
            print(f"[dry-run] {profile_name}: {' '.join(cmd)}")
            entry["status"] = "dry_run"
            entry["exit_code"] = None
            entry["finished_at"] = datetime.now(timezone.utc).isoformat()
            entry["duration_seconds"] = 0.0
            manifest["entries"].append(entry)
            continue

        print(f"[run] {profile_name}")
        start = time.time()
        with open(run_log, "w", encoding="utf-8") as handle:
            handle.write("Command:\n" + " ".join(cmd) + "\n\n")
            handle.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=dict(os.environ),
                check=False,
            )
        duration = time.time() - start
        final_stats_path = run_dir / "final_stats.json"
        success = proc.returncode == 0 and final_stats_path.exists()
        entry["status"] = "success" if success else "failed"
        entry["exit_code"] = int(proc.returncode)
        entry["finished_at"] = datetime.now(timezone.utc).isoformat()
        entry["duration_seconds"] = float(duration)
        entry["final_stats_path"] = str(final_stats_path)
        manifest["entries"].append(entry)

        if not success:
            overall_ok = False
            print(f"[fail] {profile_name} (exit={proc.returncode})")
            if args.fail_fast:
                break
        else:
            print(f"[ok] {profile_name} ({duration:.1f}s)")

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Manifest: {manifest_path}")

    benchmark_ref = benchmark_ref_from_parts(
        family=str(args.task),
        scope=str(args.mode),
        dataset_id=str(args.dataset),
        name=str(args.task),
        metadata={"dataset": args.dataset, "mode": args.mode},
    )
    method_refs = tuple(
        _profile_method_ref(profile_name, task=str(args.task))
        for profile_name in list(args.profiles)
    )
    result_rows = []
    for entry in manifest["entries"]:
        if entry.get("status") not in {"success", "dry_run"}:
            continue
        profile = str(entry.get("profile") or "")
        method_ref = next((item for item in method_refs if item.family == profile), None)
        if method_ref is None:
            continue
        result_rows.append(
            ResultRow(
                experiment_id="",
                phase=str(args.mode),
                benchmark_ref=benchmark_ref,
                method_ref=method_ref,
                split="compare",
                metric_name="profile_completed",
                metric_value=entry.get("status") == "success",
                artifact_refs=("method_compare_manifest_json",),
                metadata=dict(entry),
            )
        )
    write_canonical_sidecars(
        output_root,
        title="method_compare",
        adapter_id="method_compare",
        benchmark_refs=(benchmark_ref,),
        method_refs=method_refs,
        phases=(str(args.mode),),
        artifacts={"method_compare_manifest_json": str(manifest_path)},
        result_rows=result_rows,
        state="dry_run" if args.dry_run else ("completed" if overall_ok else "failed"),
        metadata={"mode": args.mode, "task": args.task, "dataset": args.dataset},
        launch_command=sys.argv,
        report_profiles=("runtime_eval_summary",),
    )

    if args.dry_run:
        return 0

    report_cmd = [
        args.python,
        str(REPO_ROOT / "scripts" / "report_method_compare.py"),
        "--manifest",
        str(manifest_path),
    ]
    report_proc = subprocess.run(report_cmd, cwd=str(REPO_ROOT), check=False)
    if report_proc.returncode != 0:
        overall_ok = False

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
