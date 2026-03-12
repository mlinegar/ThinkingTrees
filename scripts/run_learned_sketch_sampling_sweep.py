#!/usr/bin/env python3
"""Run aggressive multi-seed learned-sketch sampling sweeps."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import json
import multiprocessing as mp
from pathlib import Path
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.learned_sketch_simulation import (
    DEFAULT_LAW_COMPONENT_SHARE,
    DEFAULT_LAW_STRENGTH,
    DEFAULT_REGULARIZER_WEIGHT,
    DEFAULT_SUMMARY_SHARE,
    LearningRunSummary,
    SimulationConfig,
    VALID_AUDIT_POLICIES,
    VALID_SIMULATION_MODES,
    run_learning_vs_hll_experiment,
)


MetricKey = str
_THREADS_CONFIGURED = False

AGG_METRICS: Tuple[MetricKey, ...] = (
    "train_loss_final",
    "val_loss_final",
    "learned_mae",
    "learned_rmse",
    "learned_relative_rmse",
    "learned_mean_abs_rel_error",
    "learned_schedule_spread_mean",
    "latent_merge_state_mse",
    "eps_leaf",
    "eps_merge",
    "eps_idemp",
    "hll_mae",
    "hll_rmse",
    "hll_relative_rmse",
    "hll_mean_abs_rel_error",
    "hll_schedule_spread_mean",
    "distance_to_hll_floor_rel_rmse",
    "distance_to_hll_empirical_rel_rmse",
    "train_mean_tokens",
    "train_mean_leaves",
    "train_mean_internal_nodes",
    "train_audit_nodes_mean",
    "train_audit_coverage_mean",
    "train_root_queries_total",
    "train_audit_nodes_total",
    "train_total_queries_estimate",
    "rmse_gap_vs_hll",
    "abs_rel_error_gap_vs_hll",
    "excess_rmse",
    "ratio_to_floor_rmse",
    "ratio_to_floor_rel_rmse",
    "hll_empirical_excess_rmse",
    "hll_empirical_excess_rel_rmse",
    "test_cardinality_rms",
    "test_cardinality_mean",
    "regularized_objective_total",
    "regularized_objective_global_error",
    "regularized_objective_summary_budget_penalty",
    "regularized_objective_law_penalty",
    "regularized_objective_combined_regularizer",
)


def _parse_int_csv(s: str) -> tuple[int, ...]:
    out = tuple(int(x.strip()) for x in s.split(",") if x.strip())
    if len(out) == 0:
        raise ValueError("expected a non-empty int CSV")
    return out


def _parse_float_csv(s: str) -> tuple[float, ...]:
    out = tuple(float(x.strip()) for x in s.split(",") if x.strip())
    if len(out) == 0:
        raise ValueError("expected a non-empty float CSV")
    return out


def _resolve_summary_share(
    *,
    summary_share: float | None,
    law_strength: float | None,
) -> float:
    if summary_share is not None and law_strength is not None:
        raise ValueError(
            "use either --summary-regularizer-share or --law-strength, not both"
        )
    if law_strength is not None:
        if not (0.0 <= float(law_strength) <= 1.0):
            raise ValueError("law_strength must be in [0, 1]")
        return float(1.0 - float(law_strength))
    if summary_share is not None:
        if not (0.0 <= float(summary_share) <= 1.0):
            raise ValueError("summary_regularizer_share must be in [0, 1]")
        return float(summary_share)
    return float(DEFAULT_SUMMARY_SHARE)


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if len(rows) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _emit_progress(
    path: Path,
    *,
    status: str,
    started_unix: float,
    total_seeds: int,
    completed_seeds: Sequence[int],
    pending_seeds: Sequence[int],
    running_seeds: Sequence[int],
    use_cuda: bool,
    gpu_ids: Sequence[int],
    workers: int,
    state_dims: Sequence[int],
    train_sizes: Sequence[int],
    n_rows_collected: int,
    seed_runtime_rows: Sequence[dict],
    summary_path: Path,
    raw_csv_path: Path,
    agg_csv_path: Path,
    heartbeat_seconds: float,
    active_seed: Optional[int] = None,
    last_event: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    now = time.time()
    payload = {
        "status": str(status),
        "timestamp_unix": float(now),
        "started_unix": float(started_unix),
        "elapsed_seconds": float(max(0.0, now - started_unix)),
        "heartbeat_seconds": float(max(1.0, heartbeat_seconds)),
        "total_seeds": int(total_seeds),
        "completed_count": int(len(completed_seeds)),
        "pending_count": int(len(pending_seeds)),
        "running_count": int(len(running_seeds)),
        "completed_seeds": [int(x) for x in sorted(completed_seeds)],
        "pending_seeds": [int(x) for x in sorted(pending_seeds)],
        "running_seeds": [int(x) for x in sorted(running_seeds)],
        "active_seed": None if active_seed is None else int(active_seed),
        "n_rows_collected": int(n_rows_collected),
        "run_seconds_by_seed": list(seed_runtime_rows),
        "runtime_seconds_sum_completed": float(
            sum(float(x.get("runtime_seconds", 0.0)) for x in seed_runtime_rows)
        ),
        "execution": {
            "use_cuda": bool(use_cuda),
            "cuda_devices": [int(x) for x in gpu_ids],
            "parallel_workers": int(workers),
        },
        "grid": {
            "state_dims": [int(x) for x in state_dims],
            "train_sizes": [int(x) for x in train_sizes],
        },
        "outputs": {
            "json_summary": str(summary_path),
            "raw_csv": str(raw_csv_path),
            "agg_csv": str(agg_csv_path),
        },
        "last_event": None if last_event is None else str(last_event),
        "error": None if error is None else str(error),
    }
    _write_json_atomic(path, payload)


def _rows_from_results(
    results: Sequence[LearningRunSummary],
    *,
    seed: int,
    seed_runtime_seconds: float,
) -> List[dict]:
    rows: List[dict] = []
    for r in results:
        lm = r.learned_metrics
        hm = r.hll_metrics
        ro = r.regularized_objective
        rows.append(
            {
                "seed": int(seed),
                "seed_runtime_seconds": float(seed_runtime_seconds),
                "state_dim": int(r.state_dim),
                "learned_memory_bits": int(r.learned_memory_bits),
                "train_size": int(r.train_size),
                "train_loss_final": float(r.train_loss_final),
                "val_loss_final": float(r.val_loss_final),
                "learned_mae": float(lm.mae),
                "learned_rmse": float(lm.rmse),
                "learned_relative_rmse": float(lm.relative_rmse),
                "learned_mean_abs_rel_error": float(lm.mean_abs_rel_error),
                "learned_schedule_spread_mean": float(lm.schedule_spread_mean),
                "latent_merge_state_mse": float(lm.latent_merge_state_mse),
                "eps_leaf": float(lm.eps_leaf),
                "eps_merge": float(lm.eps_merge),
                "eps_idemp": float(lm.eps_idemp),
                "evidence_status": str(lm.evidence_status),
                "simulation_mode": str(lm.simulation_mode),
                "hll_precision": int(hm.precision),
                "hll_memory_bits": int(hm.memory_bits),
                "hll_rse_theory": float(r.hll_rse_theory),
                "hll_mae": float(hm.mae),
                "hll_rmse": float(hm.rmse),
                "hll_relative_rmse": float(hm.relative_rmse),
                "hll_mean_abs_rel_error": float(hm.mean_abs_rel_error),
                "hll_schedule_spread_mean": float(hm.schedule_spread_mean),
                "distance_to_hll_floor_rel_rmse": float(r.distance_to_hll_floor_rel_rmse),
                "distance_to_hll_empirical_rel_rmse": float(r.distance_to_hll_empirical_rel_rmse),
                "train_mean_tokens": float(r.train_mean_tokens),
                "train_mean_leaves": float(r.train_mean_leaves),
                "train_mean_internal_nodes": float(r.train_mean_internal_nodes),
                "train_audit_nodes_mean": float(r.train_audit_nodes_mean),
                "train_audit_coverage_mean": float(r.train_audit_coverage_mean),
                "train_root_queries_total": int(r.train_root_queries_total),
                "train_audit_nodes_total": int(r.train_audit_nodes_total),
                "train_total_queries_estimate": int(r.train_total_queries_estimate),
                "rmse_gap_vs_hll": float(r.rmse_gap_vs_hll),
                "abs_rel_error_gap_vs_hll": float(r.abs_rel_error_gap_vs_hll),
                "theoretical_floor_rmse": float(r.theoretical_floor_rmse),
                "excess_rmse": float(r.excess_rmse),
                "ratio_to_floor_rmse": float(r.ratio_to_floor_rmse),
                "ratio_to_floor_rel_rmse": float(r.ratio_to_floor_rel_rmse),
                "hll_empirical_excess_rmse": float(r.hll_empirical_excess_rmse),
                "hll_empirical_excess_rel_rmse": float(r.hll_empirical_excess_rel_rmse),
                "test_cardinality_rms": float(r.test_cardinality_rms),
                "test_cardinality_mean": float(r.test_cardinality_mean),
                "regularized_objective_total": float(ro.total),
                "regularized_objective_global_error": float(ro.global_error),
                "regularized_objective_summary_budget_penalty": float(ro.summary_budget_penalty),
                "regularized_objective_law_penalty": float(ro.law_penalty),
                "regularized_objective_combined_regularizer": float(ro.combined_regularizer),
                "regularized_objective_lambda": float(ro.regularizer_weight),
                "regularized_objective_summary_share": float(ro.summary_share),
                "regularized_objective_law_strength": float(ro.law_strength),
                "regularized_objective_leaf_share": float(ro.leaf_share),
                "regularized_objective_merge_share": float(ro.merge_share),
                "regularized_objective_idemp_share": float(ro.idemp_share),
                "regularized_objective_law_scale": float(ro.law_scale),
                "regularized_objective_uses_proxy_law_penalty": bool(
                    ro.uses_proxy_law_penalty
                ),
            }
        )
    return rows


def _group_key(row: dict) -> Tuple[int, int]:
    return int(row["state_dim"]), int(row["train_size"])


def _aggregate_rows(rows: Sequence[dict]) -> List[dict]:
    grouped: Dict[Tuple[int, int], List[dict]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)

    out: List[dict] = []
    for (state_dim, train_size), grows in sorted(grouped.items()):
        base = grows[0]
        agg = {
            "state_dim": int(state_dim),
            "train_size": int(train_size),
            "n_seeds": int(len(grows)),
            "learned_memory_bits": int(base["learned_memory_bits"]),
            "hll_precision": int(base["hll_precision"]),
            "hll_memory_bits": int(base["hll_memory_bits"]),
            "hll_rse_theory_mean": float(np.mean([float(x["hll_rse_theory"]) for x in grows])),
            "regularized_objective_lambda": float(base["regularized_objective_lambda"]),
            "regularized_objective_summary_share": float(
                base["regularized_objective_summary_share"]
            ),
            "regularized_objective_law_strength": float(
                base["regularized_objective_law_strength"]
            ),
            "regularized_objective_leaf_share": float(
                base["regularized_objective_leaf_share"]
            ),
            "regularized_objective_merge_share": float(
                base["regularized_objective_merge_share"]
            ),
            "regularized_objective_idemp_share": float(
                base["regularized_objective_idemp_share"]
            ),
            "regularized_objective_law_scale_mean": float(
                np.mean([float(x["regularized_objective_law_scale"]) for x in grows])
            ),
            "regularized_objective_proxy_fraction": float(
                np.mean(
                    [
                        1.0 if bool(x["regularized_objective_uses_proxy_law_penalty"]) else 0.0
                        for x in grows
                    ]
                )
            ),
        }
        for metric in AGG_METRICS:
            vals = np.array([float(x[metric]) for x in grows], dtype=np.float64)
            agg[f"{metric}_mean"] = float(np.mean(vals))
            agg[f"{metric}_std"] = float(np.std(vals, ddof=0))
            agg[f"{metric}_p10"] = float(np.percentile(vals, 10.0))
            agg[f"{metric}_p90"] = float(np.percentile(vals, 90.0))
        out.append(agg)
    return out


def _best_by_train_size(agg_rows: Sequence[dict]) -> List[dict]:
    train_sizes = sorted({int(r["train_size"]) for r in agg_rows})
    out: List[dict] = []
    for train_size in train_sizes:
        cands = [r for r in agg_rows if int(r["train_size"]) == train_size]
        if len(cands) == 0:
            continue
        best = min(
            cands,
            key=lambda x: (
                max(0.0, float(x["distance_to_hll_floor_rel_rmse_mean"])),
                abs(float(x["distance_to_hll_floor_rel_rmse_mean"])),
                float(x["learned_mean_abs_rel_error_mean"]),
            ),
        )
        out.append(best)
    return out


def _best_by_train_size_regularized_objective(agg_rows: Sequence[dict]) -> List[dict]:
    train_sizes = sorted({int(r["train_size"]) for r in agg_rows})
    out: List[dict] = []
    for train_size in train_sizes:
        cands = [r for r in agg_rows if int(r["train_size"]) == train_size]
        if len(cands) == 0:
            continue
        best = min(
            cands,
            key=lambda x: (
                float(x["regularized_objective_total_mean"]),
                float(x["regularized_objective_global_error_mean"]),
                float(x["regularized_objective_combined_regularizer_mean"]),
                int(x["state_dim"]),
            ),
        )
        out.append(best)
    return out


def _summarize_sampling_gain(agg_rows: Sequence[dict]) -> List[dict]:
    out: List[dict] = []
    for state_dim in sorted({int(r["state_dim"]) for r in agg_rows}):
        srows = sorted(
            [r for r in agg_rows if int(r["state_dim"]) == state_dim],
            key=lambda r: int(r["train_size"]),
        )
        if len(srows) < 2:
            continue
        start = srows[0]
        end = srows[-1]
        start_floor_dist = float(start["distance_to_hll_floor_rel_rmse_mean"])
        end_floor_dist = float(end["distance_to_hll_floor_rel_rmse_mean"])
        start_abs_rel = float(start["learned_mean_abs_rel_error_mean"])
        end_abs_rel = float(end["learned_mean_abs_rel_error_mean"])
        out.append(
            {
                "state_dim": int(state_dim),
                "train_size_min": int(start["train_size"]),
                "train_size_max": int(end["train_size"]),
                "distance_to_floor_at_min_train": start_floor_dist,
                "distance_to_floor_at_max_train": end_floor_dist,
                "distance_to_floor_drop": float(start_floor_dist - end_floor_dist),
                "distance_to_floor_drop_pct": float(
                    100.0
                    * (start_floor_dist - end_floor_dist)
                    / max(1e-12, abs(start_floor_dist))
                ),
                "learned_abs_rel_at_min_train": start_abs_rel,
                "learned_abs_rel_at_max_train": end_abs_rel,
                "absolute_drop": float(start_abs_rel - end_abs_rel),
                "relative_drop_pct": float(
                    100.0 * (start_abs_rel - end_abs_rel) / max(1e-12, start_abs_rel)
                ),
            }
        )
    return out


def _resolve_execution_device(
    *,
    device_mode: str,
    cpu_flag: bool,
    cuda_devices_csv: str | None,
    cuda_device: int | None,
) -> Tuple[bool, Tuple[int, ...]]:
    mode = str(device_mode)
    if cpu_flag:
        mode = "cpu"
    if mode not in ("auto", "cpu", "cuda"):
        raise ValueError(f"unsupported device mode: {mode}")

    if mode == "auto":
        use_cuda = bool(torch.cuda.is_available())
    elif mode == "cpu":
        use_cuda = False
    else:
        use_cuda = True

    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")
    if not use_cuda:
        if cuda_devices_csv is not None:
            raise ValueError("--cuda-devices cannot be used with CPU mode.")
        if cuda_device is not None:
            raise ValueError("--cuda-device cannot be used with CPU mode.")
        return False, tuple()

    n_cuda = int(torch.cuda.device_count())
    if cuda_devices_csv is not None:
        gpu_ids = _parse_int_csv(cuda_devices_csv)
    elif cuda_device is not None:
        gpu_ids = (int(cuda_device),)
    else:
        gpu_ids = (0,)

    for gpu_id in gpu_ids:
        if gpu_id < 0 or gpu_id >= n_cuda:
            raise ValueError(
                f"requested GPU {gpu_id} out of range; available devices: 0..{n_cuda - 1}"
            )
    return True, tuple(int(x) for x in gpu_ids)


def _run_seed_task(task: dict) -> dict:
    global _THREADS_CONFIGURED
    torch_threads = int(task["torch_threads"])
    if torch_threads > 0 and not _THREADS_CONFIGURED:
        torch.set_num_threads(torch_threads)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(torch_threads)
            except RuntimeError:
                # torch disallows resetting interop threads once work started.
                pass
        _THREADS_CONFIGURED = True

    use_cuda = bool(task["use_cuda"])
    cuda_device = task.get("cuda_device")
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("worker requested CUDA but torch.cuda.is_available() is false")
        if cuda_device is not None:
            torch.cuda.set_device(int(cuda_device))

    config = SimulationConfig(
        universe_size=int(task["universe_size"]),
        min_tokens=int(task["min_tokens"]),
        max_tokens=int(task["max_tokens"]),
        leaf_size=int(task["leaf_size"]),
        zipf_alphas=tuple(float(x) for x in task["zipf_alphas"]),
        state_dims=tuple(int(x) for x in task["state_dims"]),
        train_sizes=tuple(int(x) for x in task["train_sizes"]),
        n_val=int(task["n_val"]),
        n_test=int(task["n_test"]),
        hidden_dim=int(task["hidden_dim"]),
        n_epochs=int(task["n_epochs"]),
        batch_size=int(task["batch_size"]),
        lr=float(task["lr"]),
        weight_decay=float(task["weight_decay"]),
        c3_weight=float(task["c3_weight"]),
        leaf_weight=float(task["leaf_weight"]),
        idemp_weight=float(task["idemp_weight"]),
        regularizer_weight=float(task["regularizer_weight"]),
        summary_regularizer_share=float(task["summary_regularizer_share"]),
        law_leaf_share=float(task["law_leaf_share"]),
        law_merge_share=float(task["law_merge_share"]),
        law_idemp_share=float(task["law_idemp_share"]),
        grad_clip_norm=float(task["grad_clip_norm"]),
        audit_policy=str(task["audit_policy"]),
        audit_fixed_nodes=int(task["audit_fixed_nodes"]),
        audit_fraction=float(task["audit_fraction"]),
        audit_scale=float(task["audit_scale"]),
        audit_include_root_query=bool(task["audit_include_root_query"]),
        simulation_mode=str(task["simulation_mode"]),
        use_cuda=use_cuda,
        cuda_device=int(cuda_device) if cuda_device is not None else None,
        seed=int(task["seed"]),
    )

    t0 = time.perf_counter()
    summary = run_learning_vs_hll_experiment(config)
    dt = time.perf_counter() - t0
    rows = _rows_from_results(summary.results, seed=int(task["seed"]), seed_runtime_seconds=dt)
    return {
        "seed": int(task["seed"]),
        "runtime_seconds": float(dt),
        "rows": rows,
        "cuda_device": int(cuda_device) if cuda_device is not None else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-seed learned-sketch vs HLL sweeps and aggregate sampling-budget "
            "effects (how train-size/oracle-query budget changes error)."
        )
    )
    parser.add_argument("--universe-size", type=int, default=2048)
    parser.add_argument("--min-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--leaf-size", type=int, default=32)
    parser.add_argument("--zipf-alphas", type=str, default="0.6,0.8,1.0,1.2,1.4")
    parser.add_argument("--state-dims", type=str, default="32,64,96,128")
    parser.add_argument("--train-sizes", type=str, default="128,256,512,1024")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--n-val", type=int, default=128)
    parser.add_argument("--n-test", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--n-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--c3-weight", type=float, default=0.20)
    parser.add_argument("--leaf-weight", type=float, default=0.05)
    parser.add_argument("--idemp-weight", type=float, default=0.05)
    parser.add_argument(
        "--regularizer-weight",
        type=float,
        default=DEFAULT_REGULARIZER_WEIGHT,
        help=(
            "Lambda in (1-lambda) * global_error + lambda * regularizer. "
            f"Default: {DEFAULT_REGULARIZER_WEIGHT:.2f}."
        ),
    )
    parser.add_argument(
        "--summary-regularizer-share",
        type=float,
        default=None,
        help=(
            "Share of the regularizer placed on summary-budget pressure. "
            f"Default: {DEFAULT_SUMMARY_SHARE:.2f}."
        ),
    )
    parser.add_argument(
        "--law-strength",
        type=float,
        default=None,
        help=(
            "Alias for 1 - summary_regularizer_share. "
            f"Default endpoint pairing implies {DEFAULT_LAW_STRENGTH:.2f}."
        ),
    )
    parser.add_argument(
        "--law-leaf-share",
        type=float,
        default=DEFAULT_LAW_COMPONENT_SHARE,
        help="Leaf-law share inside the law penalty before normalization.",
    )
    parser.add_argument(
        "--law-merge-share",
        type=float,
        default=DEFAULT_LAW_COMPONENT_SHARE,
        help="Merge-law share inside the law penalty before normalization.",
    )
    parser.add_argument(
        "--law-idemp-share",
        type=float,
        default=DEFAULT_LAW_COMPONENT_SHARE,
        help="Idempotence-law share inside the law penalty before normalization.",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--audit-policy",
        type=str,
        choices=list(VALID_AUDIT_POLICIES),
        default="all",
        help="Internal-node audit sampling policy used for latent merge supervision.",
    )
    parser.add_argument(
        "--simulation-mode",
        type=str,
        choices=list(VALID_SIMULATION_MODES),
        default="latent_proxy_baseline",
        help="Whether to emit proxy-only latent metrics or decoded approximate local-law metrics.",
    )
    parser.add_argument(
        "--audit-fixed-nodes",
        type=int,
        default=0,
        help="Fixed sampled internal nodes/doc when --audit-policy fixed.",
    )
    parser.add_argument(
        "--audit-fraction",
        type=float,
        default=1.0,
        help="Sample fraction of internal nodes/doc when --audit-policy fraction.",
    )
    parser.add_argument(
        "--audit-scale",
        type=float,
        default=1.0,
        help="Scale multiplier for sqrt/log2 audit policies.",
    )
    parser.add_argument(
        "--no-root-query",
        action="store_true",
        help="Disable root oracle-loss supervision during training.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device mode. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Alias for --device cpu (kept for backward compatibility).",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="Single CUDA device index when running on GPU.",
    )
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default=None,
        help="CSV CUDA device indices to round-robin across seeds in parallel (e.g., 2,3).",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of seeds to run concurrently.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Set torch intra-op/inter-op thread count per worker (<=0 keeps torch default).",
    )
    parser.add_argument(
        "--json-summary",
        type=str,
        default="outputs/learned_sketch_sampling_sweep_summary.json",
    )
    parser.add_argument(
        "--progress-json",
        type=str,
        default=None,
        help=(
            "Live progress status JSON path. "
            "Default: <json-summary stem>_progress.json"
        ),
    )
    parser.add_argument(
        "--progress-heartbeat-sec",
        type=float,
        default=15.0,
        help="Heartbeat interval for progress JSON updates during parallel waits.",
    )
    parser.add_argument(
        "--raw-csv",
        type=str,
        default="outputs/learned_sketch_sampling_sweep_raw.csv",
    )
    parser.add_argument(
        "--agg-csv",
        type=str,
        default="outputs/learned_sketch_sampling_sweep_agg.csv",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also emit JSON summary to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = _parse_int_csv(args.seeds)
    state_dims = _parse_int_csv(args.state_dims)
    train_sizes = _parse_int_csv(args.train_sizes)
    zipf_alphas = _parse_float_csv(args.zipf_alphas)
    if len(seeds) == 0:
        raise ValueError("seeds must be non-empty")
    heartbeat_seconds = float(max(1.0, float(args.progress_heartbeat_sec)))
    summary_share = _resolve_summary_share(
        summary_share=args.summary_regularizer_share,
        law_strength=args.law_strength,
    )

    use_cuda, gpu_ids = _resolve_execution_device(
        device_mode=str(args.device),
        cpu_flag=bool(args.cpu),
        cuda_devices_csv=args.cuda_devices,
        cuda_device=args.cuda_device,
    )
    requested_workers = max(1, int(args.parallel_workers))
    workers = min(requested_workers, len(seeds))
    if use_cuda and len(gpu_ids) > 0 and workers > len(gpu_ids):
        print(
            f"warning | parallel_workers={workers} exceeds gpu_count={len(gpu_ids)}; "
            "multiple workers may share a GPU."
        )

    summary_path = Path(args.json_summary)
    progress_path = (
        Path(args.progress_json)
        if args.progress_json is not None
        else summary_path.with_name(f"{summary_path.stem}_progress.json")
    )
    raw_csv_path = Path(args.raw_csv)
    agg_csv_path = Path(args.agg_csv)

    raw_rows: List[dict] = []
    run_seconds_by_seed: List[dict] = []
    t_all_start = time.perf_counter()
    started_unix = time.time()
    completed_seed_set: set[int] = set()
    task_specs: List[dict] = []
    for idx, seed in enumerate(seeds):
        cuda_device = int(gpu_ids[idx % len(gpu_ids)]) if use_cuda else None
        task_specs.append(
            {
                "seed": int(seed),
                "universe_size": int(args.universe_size),
                "min_tokens": int(args.min_tokens),
                "max_tokens": int(args.max_tokens),
                "leaf_size": int(args.leaf_size),
                "zipf_alphas": tuple(float(x) for x in zipf_alphas),
                "state_dims": tuple(int(x) for x in state_dims),
                "train_sizes": tuple(int(x) for x in train_sizes),
                "n_val": int(args.n_val),
                "n_test": int(args.n_test),
                "hidden_dim": int(args.hidden_dim),
                "n_epochs": int(args.n_epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "c3_weight": float(args.c3_weight),
                "leaf_weight": float(args.leaf_weight),
                "idemp_weight": float(args.idemp_weight),
                "regularizer_weight": float(args.regularizer_weight),
                "summary_regularizer_share": float(summary_share),
                "law_leaf_share": float(args.law_leaf_share),
                "law_merge_share": float(args.law_merge_share),
                "law_idemp_share": float(args.law_idemp_share),
                "grad_clip_norm": float(args.grad_clip_norm),
                "audit_policy": str(args.audit_policy),
                "audit_fixed_nodes": int(args.audit_fixed_nodes),
                "audit_fraction": float(args.audit_fraction),
                "audit_scale": float(args.audit_scale),
                "audit_include_root_query": not bool(args.no_root_query),
                "simulation_mode": str(args.simulation_mode),
                "use_cuda": bool(use_cuda),
                "cuda_device": cuda_device,
                "torch_threads": int(args.torch_threads),
            }
        )

    print(
        f"launch | mode={'cuda' if use_cuda else 'cpu'} | workers={workers} | "
        f"seeds={list(seeds)} | state_dims={list(state_dims)} | train_sizes={list(train_sizes)}"
    )
    if use_cuda:
        print(f"launch_gpus | {list(gpu_ids)}")
    print(f"progress_json | {progress_path}")

    _emit_progress(
        progress_path,
        status="running",
        started_unix=started_unix,
        total_seeds=len(seeds),
        completed_seeds=[],
        pending_seeds=list(seeds),
        running_seeds=list(seeds[:workers]),
        use_cuda=use_cuda,
        gpu_ids=gpu_ids,
        workers=workers,
        state_dims=state_dims,
        train_sizes=train_sizes,
        n_rows_collected=0,
        seed_runtime_rows=[],
        summary_path=summary_path,
        raw_csv_path=raw_csv_path,
        agg_csv_path=agg_csv_path,
        heartbeat_seconds=heartbeat_seconds,
        last_event="launch",
    )

    completed = 0
    if workers == 1:
        try:
            for task in task_specs:
                active_seed = int(task["seed"])
                pending = [int(s) for s in seeds if int(s) not in completed_seed_set and int(s) != active_seed]
                _emit_progress(
                    progress_path,
                    status="running",
                    started_unix=started_unix,
                    total_seeds=len(seeds),
                    completed_seeds=list(completed_seed_set),
                    pending_seeds=pending,
                    running_seeds=[active_seed],
                    use_cuda=use_cuda,
                    gpu_ids=gpu_ids,
                    workers=workers,
                    state_dims=state_dims,
                    train_sizes=train_sizes,
                    n_rows_collected=len(raw_rows),
                    seed_runtime_rows=run_seconds_by_seed,
                    summary_path=summary_path,
                    raw_csv_path=raw_csv_path,
                    agg_csv_path=agg_csv_path,
                    heartbeat_seconds=heartbeat_seconds,
                    active_seed=active_seed,
                    last_event=f"seed_start:{active_seed}",
                )
                result = _run_seed_task(task)
                completed += 1
                completed_seed_set.add(int(result["seed"]))
                run_seconds_by_seed.append(
                    {
                        "seed": int(result["seed"]),
                        "runtime_seconds": float(result["runtime_seconds"]),
                        "cuda_device": result["cuda_device"],
                    }
                )
                raw_rows.extend(result["rows"])
                device_label = (
                    "cpu"
                    if result["cuda_device"] is None
                    else f"cuda:{int(result['cuda_device'])}"
                )
                print(
                    f"seed_complete | {completed}/{len(task_specs)} | seed={result['seed']} | "
                    f"rows={len(result['rows'])} | runtime_sec={float(result['runtime_seconds']):.2f} | "
                    f"device={device_label}"
                )
                pending = [int(s) for s in seeds if int(s) not in completed_seed_set]
                _emit_progress(
                    progress_path,
                    status="running",
                    started_unix=started_unix,
                    total_seeds=len(seeds),
                    completed_seeds=list(completed_seed_set),
                    pending_seeds=pending,
                    running_seeds=[pending[0]] if len(pending) > 0 else [],
                    use_cuda=use_cuda,
                    gpu_ids=gpu_ids,
                    workers=workers,
                    state_dims=state_dims,
                    train_sizes=train_sizes,
                    n_rows_collected=len(raw_rows),
                    seed_runtime_rows=run_seconds_by_seed,
                    summary_path=summary_path,
                    raw_csv_path=raw_csv_path,
                    agg_csv_path=agg_csv_path,
                    heartbeat_seconds=heartbeat_seconds,
                    last_event=f"seed_complete:{int(result['seed'])}",
                )
        except Exception as e:
            pending = [int(s) for s in seeds if int(s) not in completed_seed_set]
            _emit_progress(
                progress_path,
                status="failed",
                started_unix=started_unix,
                total_seeds=len(seeds),
                completed_seeds=list(completed_seed_set),
                pending_seeds=pending,
                running_seeds=[],
                use_cuda=use_cuda,
                gpu_ids=gpu_ids,
                workers=workers,
                state_dims=state_dims,
                train_sizes=train_sizes,
                n_rows_collected=len(raw_rows),
                seed_runtime_rows=run_seconds_by_seed,
                summary_path=summary_path,
                raw_csv_path=raw_csv_path,
                agg_csv_path=agg_csv_path,
                heartbeat_seconds=heartbeat_seconds,
                last_event="exception",
                error=repr(e),
            )
            raise
    else:
        try:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                pending_futures = {ex.submit(_run_seed_task, task): task for task in task_specs}
                while pending_futures:
                    done, _ = wait(
                        pending_futures.keys(),
                        timeout=heartbeat_seconds,
                        return_when=FIRST_COMPLETED,
                    )
                    if len(done) == 0:
                        pending_seeds = [
                            int(task["seed"])
                            for task in pending_futures.values()
                            if int(task["seed"]) not in completed_seed_set
                        ]
                        _emit_progress(
                            progress_path,
                            status="running",
                            started_unix=started_unix,
                            total_seeds=len(seeds),
                            completed_seeds=list(completed_seed_set),
                            pending_seeds=pending_seeds,
                            running_seeds=pending_seeds[:workers],
                            use_cuda=use_cuda,
                            gpu_ids=gpu_ids,
                            workers=workers,
                            state_dims=state_dims,
                            train_sizes=train_sizes,
                            n_rows_collected=len(raw_rows),
                            seed_runtime_rows=run_seconds_by_seed,
                            summary_path=summary_path,
                            raw_csv_path=raw_csv_path,
                            agg_csv_path=agg_csv_path,
                            heartbeat_seconds=heartbeat_seconds,
                            last_event="heartbeat",
                        )
                        continue

                    for fut in done:
                        task = pending_futures.pop(fut)
                        result = fut.result()
                        completed += 1
                        completed_seed_set.add(int(result["seed"]))
                        run_seconds_by_seed.append(
                            {
                                "seed": int(result["seed"]),
                                "runtime_seconds": float(result["runtime_seconds"]),
                                "cuda_device": result["cuda_device"],
                            }
                        )
                        raw_rows.extend(result["rows"])
                        device_label = (
                            "cpu"
                            if result["cuda_device"] is None
                            else f"cuda:{int(result['cuda_device'])}"
                        )
                        print(
                            f"seed_complete | {completed}/{len(task_specs)} | seed={result['seed']} | "
                            f"rows={len(result['rows'])} | runtime_sec={float(result['runtime_seconds']):.2f} | "
                            f"device={device_label}"
                        )
                        pending_seeds = [
                            int(t["seed"])
                            for t in pending_futures.values()
                            if int(t["seed"]) not in completed_seed_set
                        ]
                        _emit_progress(
                            progress_path,
                            status="running",
                            started_unix=started_unix,
                            total_seeds=len(seeds),
                            completed_seeds=list(completed_seed_set),
                            pending_seeds=pending_seeds,
                            running_seeds=pending_seeds[:workers],
                            use_cuda=use_cuda,
                            gpu_ids=gpu_ids,
                            workers=workers,
                            state_dims=state_dims,
                            train_sizes=train_sizes,
                            n_rows_collected=len(raw_rows),
                            seed_runtime_rows=run_seconds_by_seed,
                            summary_path=summary_path,
                            raw_csv_path=raw_csv_path,
                            agg_csv_path=agg_csv_path,
                            heartbeat_seconds=heartbeat_seconds,
                            last_event=f"seed_complete:{int(result['seed'])}",
                            active_seed=int(task["seed"]),
                        )
        except Exception as e:
            pending_seeds = [
                int(task["seed"])
                for task in task_specs
                if int(task["seed"]) not in completed_seed_set
            ]
            _emit_progress(
                progress_path,
                status="failed",
                started_unix=started_unix,
                total_seeds=len(seeds),
                completed_seeds=list(completed_seed_set),
                pending_seeds=pending_seeds,
                running_seeds=[],
                use_cuda=use_cuda,
                gpu_ids=gpu_ids,
                workers=workers,
                state_dims=state_dims,
                train_sizes=train_sizes,
                n_rows_collected=len(raw_rows),
                seed_runtime_rows=run_seconds_by_seed,
                summary_path=summary_path,
                raw_csv_path=raw_csv_path,
                agg_csv_path=agg_csv_path,
                heartbeat_seconds=heartbeat_seconds,
                last_event="exception",
                error=repr(e),
            )
            raise

    total_runtime = time.perf_counter() - t_all_start
    run_seconds_by_seed = sorted(run_seconds_by_seed, key=lambda x: int(x["seed"]))
    raw_rows = sorted(
        raw_rows,
        key=lambda x: (int(x["seed"]), int(x["state_dim"]), int(x["train_size"])),
    )
    agg_rows = _aggregate_rows(raw_rows)
    best_rows = _best_by_train_size(agg_rows)
    best_rows_regularized = _best_by_train_size_regularized_objective(agg_rows)
    sampling_gain = _summarize_sampling_gain(agg_rows)

    _write_csv(raw_csv_path, raw_rows)
    _write_csv(agg_csv_path, agg_rows)

    payload = {
        "config": {
            "universe_size": int(args.universe_size),
            "min_tokens": int(args.min_tokens),
            "max_tokens": int(args.max_tokens),
            "leaf_size": int(args.leaf_size),
            "zipf_alphas": list(zipf_alphas),
            "state_dims": list(state_dims),
            "train_sizes": list(train_sizes),
            "n_val": int(args.n_val),
            "n_test": int(args.n_test),
            "hidden_dim": int(args.hidden_dim),
            "n_epochs": int(args.n_epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "c3_weight": float(args.c3_weight),
            "leaf_weight": float(args.leaf_weight),
            "idemp_weight": float(args.idemp_weight),
            "regularizer_weight": float(args.regularizer_weight),
            "summary_regularizer_share": float(summary_share),
            "law_strength": float(1.0 - summary_share),
            "law_leaf_share": float(args.law_leaf_share),
            "law_merge_share": float(args.law_merge_share),
            "law_idemp_share": float(args.law_idemp_share),
            "grad_clip_norm": float(args.grad_clip_norm),
            "audit_policy": str(args.audit_policy),
            "audit_fixed_nodes": int(args.audit_fixed_nodes),
            "audit_fraction": float(args.audit_fraction),
            "audit_scale": float(args.audit_scale),
            "audit_include_root_query": not bool(args.no_root_query),
            "simulation_mode": str(args.simulation_mode),
            "device_mode": str(args.device),
            "cpu_flag": bool(args.cpu),
            "use_cuda": bool(use_cuda),
            "cuda_devices": list(gpu_ids),
            "parallel_workers": int(workers),
            "torch_threads": int(args.torch_threads),
        },
        "seeds": list(seeds),
        "n_raw_rows": int(len(raw_rows)),
        "n_agg_rows": int(len(agg_rows)),
        "runtime_seconds_total": float(total_runtime),
        "runtime_seconds_mean_per_seed": float(
            statistics.mean([float(x["runtime_seconds"]) for x in run_seconds_by_seed])
        ),
        "run_seconds_by_seed": run_seconds_by_seed,
        "sampling_gain": sampling_gain,
        "best_by_train_size": best_rows,
        "best_by_train_size_regularized_objective": best_rows_regularized,
        "raw_rows": raw_rows,
        "aggregated_rows": agg_rows,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    _emit_progress(
        progress_path,
        status="completed",
        started_unix=started_unix,
        total_seeds=len(seeds),
        completed_seeds=list(completed_seed_set),
        pending_seeds=[],
        running_seeds=[],
        use_cuda=use_cuda,
        gpu_ids=gpu_ids,
        workers=workers,
        state_dims=state_dims,
        train_sizes=train_sizes,
        n_rows_collected=len(raw_rows),
        seed_runtime_rows=run_seconds_by_seed,
        summary_path=summary_path,
        raw_csv_path=raw_csv_path,
        agg_csv_path=agg_csv_path,
        heartbeat_seconds=heartbeat_seconds,
        last_event="completed",
    )

    if args.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")

    print(
        "state_dim | train_size | n_seeds | reg_obj_mean±std | global_err_mean | "
        "regularizer_mean | audit_cov_mean | queries_mean"
    )
    for row in sorted(agg_rows, key=lambda r: (int(r["state_dim"]), int(r["train_size"]))):
        print(
            f"{int(row['state_dim'])} | {int(row['train_size'])} | {int(row['n_seeds'])} | "
            f"{float(row['regularized_objective_total_mean']):.5f}±{float(row['regularized_objective_total_std']):.5f} | "
            f"{float(row['regularized_objective_global_error_mean']):.5f} | "
            f"{float(row['regularized_objective_combined_regularizer_mean']):.5f} | "
            f"{float(row['train_audit_coverage_mean_mean']):.3f} | "
            f"{float(row['train_total_queries_estimate_mean']):.1f}"
        )

    print(
        "best_by_train | train_size | state_dim | dist_to_floor_mean±std | "
        "learned_rel_rmse_mean | hll_rse_theory_mean"
    )
    for row in best_rows:
        print(
            f"{int(row['train_size'])} | {int(row['state_dim'])} | "
            f"{float(row['distance_to_hll_floor_rel_rmse_mean']):+.5f}±{float(row['distance_to_hll_floor_rel_rmse_std']):.5f} | "
            f"{float(row['learned_relative_rmse_mean']):.5f} | "
            f"{float(row['hll_rse_theory_mean']):.5f}"
        )

    print(
        "best_by_train_regobj | train_size | state_dim | reg_obj_mean±std | "
        "global_err_mean | regularizer_mean"
    )
    for row in best_rows_regularized:
        print(
            f"{int(row['train_size'])} | {int(row['state_dim'])} | "
            f"{float(row['regularized_objective_total_mean']):.5f}±"
            f"{float(row['regularized_objective_total_std']):.5f} | "
            f"{float(row['regularized_objective_global_error_mean']):.5f} | "
            f"{float(row['regularized_objective_combined_regularizer_mean']):.5f}"
        )

    print(f"runtime_total_sec | {total_runtime:.2f}")
    print(f"wrote_progress_json | {progress_path}")
    print(f"wrote_json | {summary_path}")
    print(f"wrote_raw_csv | {raw_csv_path}")
    print(f"wrote_agg_csv | {agg_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
