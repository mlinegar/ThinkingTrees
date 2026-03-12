#!/usr/bin/env python3
"""Build named Markov capability-suite command files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.cli.sweep_markov_changepoint_ops_count import _iter_runs


def _write_cmd_file(path: Path, runs: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmds = [str(getattr(run, "command")) for run in runs]
    path.write_text("\n".join(cmds) + ("\n" if cmds else ""), encoding="utf-8")


def _parse_ints(text: str) -> List[int]:
    return [int(x) for x in str(text).replace(",", " ").split() if x.strip()]


def _base_runs(
    *,
    python_bin: str,
    output_root: Path,
    n_regimes: int,
    fixed_leaf_tokens: int,
    train_docs: Iterable[int],
    val_docs: int,
    test_docs: int,
    audit_fractions: Iterable[float],
    local_law_weights: Iterable[float],
    schedule_weights: Iterable[float],
    state_dims: Iterable[int],
    hidden_dims: Iterable[int],
    root_weights: Iterable[float],
    data_seeds: Iterable[int],
    model_seeds: Iterable[int],
    n_epochs: int,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
) -> List[object]:
    return _iter_runs(
        python_bin=str(python_bin),
        n_regimes=int(n_regimes),
        vocab_size=96,
        min_tokens=384,
        max_tokens=384,
        min_segments=12,
        max_segments=24,
        fixed_leaf_tokens=int(fixed_leaf_tokens),
        train_docs=list(train_docs),
        val_docs=int(val_docs),
        test_docs=int(test_docs),
        audit_fractions=list(audit_fractions),
        c3_audit_strategies=["uniform"],
        c3_include_root=True,
        leaf_query_rates=[1.0],
        include_root_queries=[True],
        local_law_weights=list(local_law_weights),
        c1_relative_weights=[1.0],
        c3_relative_weights=[4.0],
        root_weights=list(root_weights),
        schedule_consistency_weights=list(schedule_weights),
        guidance_override_modes=["reset"],
        eval_guidance_qs=[],
        eval_guidance_trials=0,
        eval_guidance_seed_offset=100_000,
        eval_guidance_include_root=True,
        include_rf_root_baseline=False,
        rf_n_estimators=200,
        rf_max_depth=16,
        rf_min_samples_leaf=5,
        data_seeds=list(data_seeds),
        seeds=list(model_seeds),
        output_root=output_root,
        model_families=["neural"],
        feature_modes=["full"],
        state_dims=list(state_dims),
        hidden_dims=list(hidden_dims),
        hidden_dim_multiplier=None,
        hidden_dim_min=64,
        n_epochs=int(n_epochs),
        device=str(device),
        cuda_device=int(cuda_device) if cuda_device is not None else None,
        violation_tau=0.0,
        torch_threads=int(torch_threads),
        skip_existing=True,
    )


def _build_sanity_suite(
    *,
    python_bin: str,
    suite_root: Path,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
) -> List[object]:
    runs: List[object] = []
    for n_regimes in (2, 4):
        for fixed_leaf_tokens in (8, 16, 32):
            runs.extend(
                _base_runs(
                    python_bin=python_bin,
                    output_root=suite_root / f"nreg_{n_regimes}" / f"leaf_{fixed_leaf_tokens}",
                    n_regimes=n_regimes,
                    fixed_leaf_tokens=fixed_leaf_tokens,
                    train_docs=[128, 512, 2048],
                    val_docs=256,
                    test_docs=512,
                    audit_fractions=[1.0],
                    local_law_weights=[0.0, 0.25, 0.5, 0.75, 1.0],
                    schedule_weights=[0.0, 0.1, 0.2],
                    state_dims=[64],
                    hidden_dims=[256],
                    root_weights=[1.0],
                    data_seeds=range(5),
                    model_seeds=range(5),
                    n_epochs=24,
                    device=device,
                    cuda_device=cuda_device,
                    torch_threads=torch_threads,
                )
            )
    return runs


def _build_transition_map_suite(
    *,
    python_bin: str,
    suite_root: Path,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
) -> List[object]:
    return _base_runs(
        python_bin=python_bin,
        output_root=suite_root,
        n_regimes=4,
        fixed_leaf_tokens=16,
        train_docs=[128, 256, 512, 1024, 2048, 4096],
        val_docs=512,
        test_docs=1024,
        audit_fractions=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        local_law_weights=[0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0],
        schedule_weights=[0.0, 0.05, 0.1, 0.2],
        state_dims=[32, 64, 128],
        hidden_dims=[128, 256, 512],
        root_weights=[1.0],
        data_seeds=range(4),
        model_seeds=range(4),
        n_epochs=24,
        device=device,
        cuda_device=cuda_device,
        torch_threads=torch_threads,
    )


def _choose_mechanism_rows(summary_rows: Sequence[dict], *, limit: int) -> List[dict]:
    partial = [
        row
        for row in summary_rows
        if 0.0 < float(row.get("full_success_rate", 0.0)) < 1.0
    ]
    if partial:
        ranked = sorted(
            partial,
            key=lambda row: (
                abs(float(row.get("full_success_rate", 0.0)) - 0.5),
                abs(float(row.get("theorem_margin", 0.0))),
                abs(float(row.get("root_margin", 0.0))),
            ),
        )
        return ranked[: int(limit)]
    failures = [
        row
        for row in summary_rows
        if str(row.get("dominant_capability_status")) != "full_success"
    ]
    ranked = sorted(
        failures or list(summary_rows),
        key=lambda row: (
            abs(float(row.get("theorem_margin", 0.0))),
            abs(float(row.get("spread_margin", 0.0))),
            abs(float(row.get("root_margin", 0.0))),
        ),
    )
    return ranked[: int(limit)]


def _positive_fallback(
    rows: Sequence[dict],
    *,
    key: str,
    n_regimes: int,
    fixed_leaf_tokens: int,
    state_dim: int,
    hidden_dim: int,
) -> float:
    candidates = [
        float(row.get(key, 0.0))
        for row in rows
        if int(row.get("n_regimes", -1)) == int(n_regimes)
        and int(row.get("fixed_leaf_tokens", -1)) == int(fixed_leaf_tokens)
        and int(row.get("state_dim", -1)) == int(state_dim)
        and int(row.get("hidden_dim", -1)) == int(hidden_dim)
        and float(row.get(key, 0.0)) > 0.0
    ]
    return max(candidates) if candidates else (0.2 if key.endswith("sched") else 0.5)


def _build_mechanism_suite(
    *,
    python_bin: str,
    suite_root: Path,
    transition_summary: Path,
    mechanism_cells: int,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
) -> tuple[List[object], List[dict]]:
    payload = json.loads(transition_summary.read_text(encoding="utf-8"))
    rows = list(payload.get("aggregated_rows") or [])
    chosen = _choose_mechanism_rows(rows, limit=int(mechanism_cells))

    runs: List[object] = []
    selected_cells: List[dict] = []
    for idx, row in enumerate(chosen):
        boundary_llw = float(row.get("selected_lambda_local", 0.0))
        boundary_scw = float(row.get("selected_lambda_sched", 0.0))
        if boundary_llw <= 0.0:
            boundary_llw = _positive_fallback(
                rows,
                key="selected_lambda_local",
                n_regimes=int(row["n_regimes"]),
                fixed_leaf_tokens=int(row["fixed_leaf_tokens"]),
                state_dim=int(row["state_dim"]),
                hidden_dim=int(row["hidden_dim"]),
            )
        if boundary_scw <= 0.0:
            boundary_scw = _positive_fallback(
                rows,
                key="selected_lambda_sched",
                n_regimes=int(row["n_regimes"]),
                fixed_leaf_tokens=int(row["fixed_leaf_tokens"]),
                state_dim=int(row["state_dim"]),
                hidden_dim=int(row["hidden_dim"]),
            )
        selected_cells.append(
            {
                "index": int(idx),
                "n_regimes": int(row["n_regimes"]),
                "fixed_leaf_tokens": int(row["fixed_leaf_tokens"]),
                "train_docs": int(row["train_docs"]),
                "audit_fraction": float(row["audit_fraction"]),
                "state_dim": int(row["state_dim"]),
                "hidden_dim": int(row["hidden_dim"]),
                "n_epochs": int(row["n_epochs"]),
                "boundary_lambda_local": float(boundary_llw),
                "boundary_lambda_sched": float(boundary_scw),
            }
        )
        runs.extend(
            _base_runs(
                python_bin=python_bin,
                output_root=suite_root / f"cell_{idx}",
                n_regimes=int(row["n_regimes"]),
                fixed_leaf_tokens=int(row["fixed_leaf_tokens"]),
                train_docs=[int(row["train_docs"])],
                val_docs=int(row["val_docs"]),
                test_docs=int(row["test_docs"]),
                audit_fractions=[float(row["audit_fraction"])],
                local_law_weights=[0.0, float(boundary_llw)],
                schedule_weights=[0.0, float(boundary_scw)],
                state_dims=[int(row["state_dim"])],
                hidden_dims=[int(row["hidden_dim"])],
                root_weights=[0.5, 1.0, 2.0, 4.0],
                data_seeds=range(4),
                model_seeds=range(4),
                n_epochs=int(row["n_epochs"]),
                device=device,
                cuda_device=cuda_device,
                torch_threads=torch_threads,
            )
        )
    return runs, selected_cells


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build named Markov capability-suite command files.")
    parser.add_argument(
        "--suite",
        choices=["sanity_suite", "transition_map_suite", "mechanism_suite", "all"],
        default="all",
    )
    parser.add_argument("--output-root", type=str, default="outputs/markov_capability_suites")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--cmd-dir", type=str, default="logs/markov_capability_suites")
    parser.add_argument("--transition-summary", type=str, default="")
    parser.add_argument("--mechanism-cells", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_root = Path(args.output_root)
    cmd_dir = Path(args.cmd_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cmd_dir.mkdir(parents=True, exist_ok=True)

    suite = str(args.suite)
    manifest: Dict[str, object] = {
        "output_root": str(output_root),
        "suite": suite,
        "device": str(args.device),
    }

    if suite in {"sanity_suite", "all"}:
        runs = _build_sanity_suite(
            python_bin=str(args.python_bin),
            suite_root=output_root / "sanity_suite" / "markov_changepoint_ops_count",
            device=str(args.device),
            cuda_device=int(args.cuda_device) if args.cuda_device is not None else None,
            torch_threads=int(args.torch_threads),
        )
        cmd_path = cmd_dir / "sanity_suite_cmds.txt"
        _write_cmd_file(cmd_path, runs)
        manifest["sanity_suite"] = {"n_commands": len(runs), "cmd_file": str(cmd_path)}

    if suite in {"transition_map_suite", "all"}:
        runs = _build_transition_map_suite(
            python_bin=str(args.python_bin),
            suite_root=output_root / "transition_map_suite" / "markov_changepoint_ops_count",
            device=str(args.device),
            cuda_device=int(args.cuda_device) if args.cuda_device is not None else None,
            torch_threads=int(args.torch_threads),
        )
        cmd_path = cmd_dir / "transition_map_suite_cmds.txt"
        _write_cmd_file(cmd_path, runs)
        manifest["transition_map_suite"] = {"n_commands": len(runs), "cmd_file": str(cmd_path)}

    if suite in {"mechanism_suite", "all"}:
        if not str(args.transition_summary).strip():
            raise SystemExit("--transition-summary is required for mechanism_suite")
        runs, selected_cells = _build_mechanism_suite(
            python_bin=str(args.python_bin),
            suite_root=output_root / "mechanism_suite" / "markov_changepoint_ops_count",
            transition_summary=Path(args.transition_summary),
            mechanism_cells=int(args.mechanism_cells),
            device=str(args.device),
            cuda_device=int(args.cuda_device) if args.cuda_device is not None else None,
            torch_threads=int(args.torch_threads),
        )
        cmd_path = cmd_dir / "mechanism_suite_cmds.txt"
        _write_cmd_file(cmd_path, runs)
        manifest["mechanism_suite"] = {
            "n_commands": len(runs),
            "cmd_file": str(cmd_path),
            "selected_cells": selected_cells,
        }

    manifest_path = cmd_dir / "markov_capability_suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
