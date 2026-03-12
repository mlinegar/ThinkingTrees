#!/usr/bin/env python3
"""Build command files for the Markov local-law stress suites."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.cli.sweep_markov_changepoint_ops_count import _iter_runs
from src.ctreepo.sim.manifest import write_manifest_jsonl


def _write_cmd_file(path: Path, runs: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmds = [str(getattr(run, "command")) for run in runs]
    path.write_text("\n".join(cmds) + ("\n" if cmds else ""), encoding="utf-8")


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
    law_packages: Iterable[str],
    exact_families: Iterable[str],
    state_dims: Iterable[int],
    hidden_dims: Iterable[int],
    root_weights: Iterable[float],
    data_seeds: Iterable[int],
    model_seeds: Iterable[int],
    n_epochs: int,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
    suite_role: str,
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
        local_law_weights=[1.0],
        c1_relative_weights=[1.0],
        c2_relative_weights=[0.0],
        c3_relative_weights=[4.0],
        c2_weights=[0.0],
        root_weights=list(root_weights),
        schedule_consistency_weights=[0.0],
        law_packages=list(law_packages),
        exact_families=list(exact_families),
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
        suite_role=str(suite_role),
    )


def _build_sanity_suite(
    *,
    python_bin: str,
    suite_root: Path,
    smoke: bool,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
) -> Tuple[List[object], List[object]]:
    train_docs = [128, 512, 2048] if not smoke else [32]
    regimes = [2, 4] if not smoke else [2]
    leaf_tokens = [8, 16] if not smoke else [8]
    data_seeds = [0, 1] if not smoke else [0]
    model_seeds = [0, 1] if not smoke else [0]
    learned: List[object] = []
    exact: List[object] = []
    for n_regimes in regimes:
        for leaf in leaf_tokens:
            learned.extend(
                _base_runs(
                    python_bin=python_bin,
                    output_root=suite_root / "learned" / f"nreg_{n_regimes}" / f"leaf_{leaf}",
                    n_regimes=n_regimes,
                    fixed_leaf_tokens=leaf,
                    train_docs=train_docs,
                    val_docs=64 if smoke else 256,
                    test_docs=64 if smoke else 512,
                    audit_fractions=[1.0],
                    law_packages=["root_only", "c1_only", "c2_only", "c3_only", "c1c3", "all_laws", "all_laws_plus_sched"],
                    exact_families=[],
                    state_dims=[64],
                    hidden_dims=[256],
                    root_weights=[1.0],
                    data_seeds=data_seeds,
                    model_seeds=model_seeds,
                    n_epochs=2 if smoke else 12,
                    device=device,
                    cuda_device=cuda_device,
                    torch_threads=torch_threads,
                    suite_role="positive_controls",
                )
            )
            exact.extend(
                _base_runs(
                    python_bin=python_bin,
                    output_root=suite_root / "exact" / f"nreg_{n_regimes}" / f"leaf_{leaf}",
                    n_regimes=n_regimes,
                    fixed_leaf_tokens=leaf,
                    train_docs=train_docs,
                    val_docs=64 if smoke else 256,
                    test_docs=64 if smoke else 512,
                    audit_fractions=[1.0],
                    law_packages=[],
                    exact_families=["exact", "leaf_bucket", "count_only", "flip_R2"],
                    state_dims=[64],
                    hidden_dims=[256],
                    root_weights=[1.0],
                    data_seeds=data_seeds,
                    model_seeds=[0],
                    n_epochs=1,
                    device=device,
                    cuda_device=cuda_device,
                    torch_threads=torch_threads,
                    suite_role="failure_modes",
                )
            )
    return learned, exact


def _build_transition_map_suite(
    *,
    python_bin: str,
    suite_root: Path,
    smoke: bool,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
) -> List[object]:
    return _base_runs(
        python_bin=python_bin,
        output_root=suite_root,
        n_regimes=4,
        fixed_leaf_tokens=16,
        train_docs=[128, 512, 2048, 4096] if not smoke else [64, 128],
        val_docs=128 if smoke else 256,
        test_docs=128 if smoke else 512,
        audit_fractions=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0] if not smoke else [0.1, 1.0],
        law_packages=["root_only", "c1c3", "c2_only", "all_laws", "all_laws_plus_sched"],
        exact_families=[],
        state_dims=[64],
        hidden_dims=[256],
        root_weights=[1.0],
        data_seeds=[0, 1] if not smoke else [0],
        model_seeds=[0, 1] if not smoke else [0],
        n_epochs=2 if smoke else 12,
        device=device,
        cuda_device=cuda_device,
        torch_threads=torch_threads,
        suite_role="support_scaling",
    )


def _choose_boundary_cells(rows: Sequence[dict], *, limit: int) -> List[dict]:
    main_rows = [
        row
        for row in rows
        if str(row.get("law_package", "")) in {"all_laws_plus_sched", "all_laws"}
    ]
    ranked = sorted(
        main_rows,
        key=lambda row: (
            abs(float(row.get("val_bundle_full_success_rate", row.get("bundle_full_success_rate", 0.0))) - 0.5),
            abs(float(row.get("val_bundle_margin_mean", row.get("bundle_margin_mean", 0.0)))),
            int(row.get("train_docs", 0)),
            float(row.get("audit_fraction", 0.0)),
        ),
    )
    return ranked[: int(limit)]


def _build_mechanism_suite(
    *,
    python_bin: str,
    suite_root: Path,
    transition_summary: Path,
    smoke: bool,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
) -> Tuple[List[object], List[dict]]:
    payload = json.loads(transition_summary.read_text(encoding="utf-8"))
    rows = list(payload.get("aggregated_rows") or [])
    chosen = _choose_boundary_cells(rows, limit=1 if smoke else 2)
    runs: List[object] = []
    selected: List[dict] = []
    for idx, row in enumerate(chosen):
        selected.append(
            {
                "index": int(idx),
                "n_regimes": int(row["n_regimes"]),
                "fixed_leaf_tokens": int(row["fixed_leaf_tokens"]),
                "train_docs": int(row["train_docs"]),
                "val_docs": int(row["val_docs"]),
                "test_docs": int(row["test_docs"]),
                "audit_fraction": float(row["audit_fraction"]),
                "state_dim": int(row["state_dim"]),
                "hidden_dim": int(row["hidden_dim"]),
                "n_epochs": int(row["n_epochs"]),
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
                law_packages=["root_only", "c1_only", "c2_only", "c3_only", "c1c3", "all_laws", "sched_only", "all_laws_plus_sched"],
                exact_families=[],
                state_dims=[64],
                hidden_dims=[256],
                root_weights=[0.5, 1.0, 2.0] if not smoke else [1.0],
                data_seeds=[0, 1] if not smoke else [0],
                model_seeds=[0, 1] if not smoke else [0],
                n_epochs=int(row["n_epochs"]),
                device=device,
                cuda_device=cuda_device,
                torch_threads=torch_threads,
                suite_role="relevance_mediation",
            )
        )
    return runs, selected


def _build_capacity_appendix_suite(
    *,
    python_bin: str,
    suite_root: Path,
    smoke: bool,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
) -> List[object]:
    caps = [(32, 128), (64, 256), (128, 512)] if not smoke else [(32, 128), (64, 256)]
    runs: List[object] = []
    for state_dim, hidden_dim in caps:
        runs.extend(
            _base_runs(
                python_bin=python_bin,
                output_root=suite_root / f"sd_{state_dim}_hd_{hidden_dim}",
                n_regimes=4,
                fixed_leaf_tokens=16,
                train_docs=[512, 2048] if not smoke else [128],
                val_docs=128 if smoke else 256,
                test_docs=128 if smoke else 512,
                audit_fractions=[0.1, 1.0] if not smoke else [0.1],
                law_packages=["root_only", "all_laws_plus_sched"],
                exact_families=[],
                state_dims=[state_dim],
                hidden_dims=[hidden_dim],
                root_weights=[1.0],
                data_seeds=[0, 1] if not smoke else [0],
                model_seeds=[0, 1] if not smoke else [0],
        n_epochs=2 if smoke else 12,
        device=device,
        cuda_device=cuda_device,
        torch_threads=torch_threads,
        suite_role="hardness",
    )
        )
    return runs


def _build_cross_dgp_suite(
    *,
    python_bin: str,
    suite_root: Path,
    smoke: bool,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
) -> List[object]:
    """Fresh matched suite for cross-DGP comparison.

    Every config is run with root_only (baseline) AND each treatment package so
    that the cross-DGP reporter can pair them by scenario key.
    """
    caps = [(64, 256), (128, 512)] if not smoke else [(64, 256)]
    runs: List[object] = []
    for state_dim, hidden_dim in caps:
        runs.extend(
            _base_runs(
                python_bin=python_bin,
                output_root=suite_root / f"sd_{state_dim}_hd_{hidden_dim}",
                n_regimes=4,
                fixed_leaf_tokens=16,
                train_docs=[256, 1024] if not smoke else [64],
                val_docs=128 if smoke else 256,
                test_docs=128 if smoke else 512,
                audit_fractions=[0.1, 0.5, 1.0] if not smoke else [0.1, 1.0],
                law_packages=[
                    "root_only", "c1_only", "c2_only", "c3_only",
                    "c1c3", "all_laws", "all_laws_plus_sched",
                ],
                exact_families=[],
                state_dims=[state_dim],
                hidden_dims=[hidden_dim],
                root_weights=[1.0],
                data_seeds=[0, 1] if not smoke else [0],
                model_seeds=[0],
                n_epochs=2 if smoke else 12,
                device=device,
                cuda_device=cuda_device,
                torch_threads=torch_threads,
                suite_role="cross_dgp_matched",
            )
        )
    return runs


WEIGHT_ABLATION_PROFILES: List[Tuple[str, float, float, float]] = [
    # (label, c1_rel, c2_rel, c3_rel)
    # --- Pure extremes ---
    ("pure_c2", 0.0, 1.0, 0.0),
    ("no_c2", 1.0, 0.0, 4.0),
    # --- Fine gradient: transition from pure_c2 to equal ---
    ("c2_trace_c1c3", 0.05, 1.0, 0.05),
    ("c2_light_c1c3", 0.1, 1.0, 0.1),
    ("c2_mild_c1c3", 0.25, 1.0, 0.25),
    ("c2_moderate_c1c3", 0.5, 1.0, 0.5),
    # --- Coarse C2-heavy combinations ---
    ("c2_very_dominant", 1.0, 8.0, 1.0),
    ("c2_dominant", 1.0, 4.0, 1.0),
    ("c2_heavy", 1.0, 2.0, 1.0),
    # --- Equal and C1C3-biased ---
    ("equal", 1.0, 1.0, 1.0),
    ("c1c3_heavy", 2.0, 1.0, 2.0),
    ("c3_dominant", 1.0, 1.0, 4.0),
]


def _build_weight_ablation_suite(
    *,
    python_bin: str,
    suite_root: Path,
    smoke: bool,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
) -> List[object]:
    """Sweep C1/C2/C3 relative weights to find the optimal mix.

    Uses formal_local_law_weight parameterization (no law_package) so that
    relative weights are respected directly. Includes root_only baseline
    via law_package for matched comparison.
    """
    caps = [(64, 256)] if smoke else [(64, 256), (128, 512)]
    train_docs_list = [64] if smoke else [256, 1024]
    audit_list = [0.1, 1.0] if smoke else [0.1, 0.5, 1.0]
    data_seeds = [0] if smoke else [0, 1]
    model_seeds = [0]
    n_epochs = 2 if smoke else 12
    val_docs = 128 if smoke else 256
    test_docs = 128 if smoke else 512
    runs: List[object] = []

    for state_dim, hidden_dim in caps:
        # Root-only baseline via law_package
        runs.extend(
            _base_runs(
                python_bin=python_bin,
                output_root=suite_root / f"sd_{state_dim}_hd_{hidden_dim}",
                n_regimes=4,
                fixed_leaf_tokens=16,
                train_docs=train_docs_list,
                val_docs=val_docs,
                test_docs=test_docs,
                audit_fractions=audit_list,
                law_packages=["root_only"],
                exact_families=[],
                state_dims=[state_dim],
                hidden_dims=[hidden_dim],
                root_weights=[1.0],
                data_seeds=data_seeds,
                model_seeds=model_seeds,
                n_epochs=n_epochs,
                device=device,
                cuda_device=cuda_device,
                torch_threads=torch_threads,
                suite_role="weight_ablation_baseline",
            )
        )
        # Weight profiles via formal_local_law_weight (no law_package)
        for _label, c1_rel, c2_rel, c3_rel in WEIGHT_ABLATION_PROFILES:
            runs.extend(
                _iter_runs(
                    python_bin=str(python_bin),
                    n_regimes=4,
                    vocab_size=96,
                    min_tokens=384,
                    max_tokens=384,
                    min_segments=12,
                    max_segments=24,
                    fixed_leaf_tokens=16,
                    train_docs=train_docs_list,
                    val_docs=val_docs,
                    test_docs=test_docs,
                    audit_fractions=audit_list,
                    c3_audit_strategies=["uniform"],
                    c3_include_root=True,
                    leaf_query_rates=[1.0],
                    include_root_queries=[True],
                    local_law_weights=[1.0],
                    c1_relative_weights=[c1_rel],
                    c2_relative_weights=[c2_rel],
                    c3_relative_weights=[c3_rel],
                    c2_weights=[0.0],
                    root_weights=[1.0],
                    schedule_consistency_weights=[0.0],
                    law_packages=[],
                    exact_families=[],
                    guidance_override_modes=["reset"],
                    eval_guidance_qs=[],
                    eval_guidance_trials=0,
                    eval_guidance_seed_offset=100_000,
                    eval_guidance_include_root=True,
                    include_rf_root_baseline=False,
                    rf_n_estimators=200,
                    rf_max_depth=16,
                    rf_min_samples_leaf=5,
                    data_seeds=data_seeds,
                    seeds=model_seeds,
                    output_root=suite_root / f"sd_{state_dim}_hd_{hidden_dim}",
                    model_families=["neural"],
                    feature_modes=["full"],
                    state_dims=[state_dim],
                    hidden_dims=[hidden_dim],
                    hidden_dim_multiplier=None,
                    hidden_dim_min=64,
                    n_epochs=n_epochs,
                    device=str(device),
                    cuda_device=int(cuda_device) if cuda_device is not None else None,
                    violation_tau=0.0,
                    torch_threads=int(torch_threads),
                    skip_existing=True,
                    suite_role="weight_ablation",
                )
            )
    return runs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Markov local-law stress suite commands.")
    parser.add_argument(
        "--suite",
        choices=["sanity_suite", "transition_map_suite", "mechanism_suite", "capacity_appendix_suite", "cross_dgp_suite", "weight_ablation_suite", "all"],
        default="all",
    )
    parser.add_argument("--output-root", type=str, default="outputs/markov_law_stress_suites")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--cmd-dir", type=str, default="logs/markov_law_stress_suites")
    parser.add_argument("--transition-summary", type=str, default="")
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_root = Path(args.output_root)
    cmd_dir = Path(args.cmd_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cmd_dir.mkdir(parents=True, exist_ok=True)
    all_runs: List[object] = []

    manifest: dict[str, object] = {
        "output_root": str(output_root),
        "suite": str(args.suite),
        "smoke": bool(args.smoke),
    }

    if args.suite in {"sanity_suite", "all"}:
        learned, exact = _build_sanity_suite(
            python_bin=str(args.python_bin),
            suite_root=output_root / "sanity_suite" / "markov_changepoint_ops_count",
            smoke=bool(args.smoke),
            device=str(args.device),
            cuda_device=int(args.cuda_device) if args.cuda_device is not None else None,
            torch_threads=int(args.torch_threads),
        )
        learned_cmd = cmd_dir / "sanity_suite_learned_cmds.txt"
        exact_cmd = cmd_dir / "sanity_suite_exact_cmds.txt"
        _write_cmd_file(learned_cmd, learned)
        _write_cmd_file(exact_cmd, exact)
        all_runs.extend(learned)
        all_runs.extend(exact)
        manifest["sanity_suite"] = {
            "learned_n_commands": len(learned),
            "learned_cmd_file": str(learned_cmd),
            "exact_n_commands": len(exact),
            "exact_cmd_file": str(exact_cmd),
        }

    if args.suite in {"transition_map_suite", "all"}:
        runs = _build_transition_map_suite(
            python_bin=str(args.python_bin),
            suite_root=output_root / "transition_map_suite" / "markov_changepoint_ops_count",
            smoke=bool(args.smoke),
            device=str(args.device),
            cuda_device=int(args.cuda_device) if args.cuda_device is not None else None,
            torch_threads=int(args.torch_threads),
        )
        cmd_file = cmd_dir / "transition_map_suite_cmds.txt"
        _write_cmd_file(cmd_file, runs)
        all_runs.extend(runs)
        manifest["transition_map_suite"] = {"n_commands": len(runs), "cmd_file": str(cmd_file)}

    if args.suite in {"mechanism_suite", "all"}:
        if not str(args.transition_summary).strip():
            raise SystemExit("--transition-summary is required for mechanism_suite")
        runs, selected = _build_mechanism_suite(
            python_bin=str(args.python_bin),
            suite_root=output_root / "mechanism_suite" / "markov_changepoint_ops_count",
            transition_summary=Path(args.transition_summary),
            smoke=bool(args.smoke),
            device=str(args.device),
            cuda_device=int(args.cuda_device) if args.cuda_device is not None else None,
            torch_threads=int(args.torch_threads),
        )
        cmd_file = cmd_dir / "mechanism_suite_cmds.txt"
        _write_cmd_file(cmd_file, runs)
        all_runs.extend(runs)
        manifest["mechanism_suite"] = {
            "n_commands": len(runs),
            "cmd_file": str(cmd_file),
            "selected_cells": selected,
        }

    if args.suite in {"capacity_appendix_suite", "all"}:
        runs = _build_capacity_appendix_suite(
            python_bin=str(args.python_bin),
            suite_root=output_root / "capacity_appendix_suite" / "markov_changepoint_ops_count",
            smoke=bool(args.smoke),
            device=str(args.device),
            cuda_device=int(args.cuda_device) if args.cuda_device is not None else None,
            torch_threads=int(args.torch_threads),
        )
        cmd_file = cmd_dir / "capacity_appendix_suite_cmds.txt"
        _write_cmd_file(cmd_file, runs)
        all_runs.extend(runs)
        manifest["capacity_appendix_suite"] = {"n_commands": len(runs), "cmd_file": str(cmd_file)}

    if args.suite in {"cross_dgp_suite"}:
        runs = _build_cross_dgp_suite(
            python_bin=str(args.python_bin),
            suite_root=output_root / "cross_dgp_suite" / "markov_changepoint_ops_count",
            smoke=bool(args.smoke),
            device=str(args.device),
            cuda_device=int(args.cuda_device) if args.cuda_device is not None else None,
            torch_threads=int(args.torch_threads),
        )
        cmd_file = cmd_dir / "cross_dgp_suite_cmds.txt"
        _write_cmd_file(cmd_file, runs)
        all_runs.extend(runs)
        manifest["cross_dgp_suite"] = {"n_commands": len(runs), "cmd_file": str(cmd_file)}

    if args.suite in {"weight_ablation_suite"}:
        runs = _build_weight_ablation_suite(
            python_bin=str(args.python_bin),
            suite_root=output_root / "weight_ablation_suite" / "markov_changepoint_ops_count",
            smoke=bool(args.smoke),
            device=str(args.device),
            cuda_device=int(args.cuda_device) if args.cuda_device is not None else None,
            torch_threads=int(args.torch_threads),
        )
        cmd_file = cmd_dir / "weight_ablation_suite_cmds.txt"
        _write_cmd_file(cmd_file, runs)
        all_runs.extend(runs)
        manifest["weight_ablation_suite"] = {"n_commands": len(runs), "cmd_file": str(cmd_file)}

    manifest_jsonl_path = cmd_dir / "markov_law_stress_suite_manifest.jsonl"
    write_manifest_jsonl(manifest_jsonl_path, all_runs)
    manifest_path = cmd_dir / "markov_law_stress_suite_manifest.json"
    manifest["runspec_manifest"] = str(manifest_jsonl_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
