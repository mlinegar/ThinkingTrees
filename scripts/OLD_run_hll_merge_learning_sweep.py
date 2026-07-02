# OLD_: archived 2026-07-02; driver for OLD_hll_merge_learning_simulation.py. Kept for reference; do not import or run.
#!/usr/bin/env python3
"""Run embarrassingly-parallel HLL merge-learning sweeps (multi-seed).

This is the theory-linked mergeable-sketch simulation suite:
learn an elementwise merge law over HLL registers via local (C3-style) checks,
and compare the learned merger's error to the HLL theoretical RSE floor.

Outputs:
- JSON summary (raw per-seed rows + aggregated rows)
- Raw CSV (per-seed rows)
- Aggregated CSV (mean/std/p10/p90 across seeds)
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.hll_merge_learning_simulation import (  # noqa: E402
    HLLMergeLearningConfig,
    experiment_rows,
    run_hll_merge_learning_experiment,
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


def _parse_str_csv(s: str) -> tuple[str, ...]:
    out = tuple(x.strip() for x in s.split(",") if x.strip())
    if len(out) == 0:
        raise ValueError("expected a non-empty string CSV")
    return out


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if len(rows) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _finite_floats(vals: Iterable[object]) -> List[float]:
    out: List[float] = []
    for v in vals:
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(x):  # type: ignore[name-defined]
            continue
        out.append(x)
    return out


def _summary_stats(xs: List[float]) -> Dict[str, float | None]:
    if len(xs) == 0:
        return {"mean": None, "std": None, "p10": None, "p90": None}
    arr = np.array(xs, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    p10 = float(np.percentile(arr, 10.0))
    p90 = float(np.percentile(arr, 90.0))
    return {"mean": mean, "std": std, "p10": p10, "p90": p90}


def aggregate_rows(
    raw_rows: Sequence[dict],
    *,
    group_keys: Sequence[str],
    metric_keys: Sequence[str],
) -> List[dict]:
    groups: Dict[Tuple[object, ...], List[dict]] = {}
    for row in raw_rows:
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)

    out: List[dict] = []
    for key, rows in sorted(groups.items(), key=lambda kv: kv[0]):
        base = {k: v for k, v in zip(group_keys, key)}
        base["n_seeds"] = int(len({r.get("seed") for r in rows}))
        # Copy through common non-metrics (e.g., memory bits) if present.
        for passthru in ("registers", "memory_bits", "memory_bytes", "hll_rse_theory"):
            if passthru in rows[0]:
                base[passthru] = rows[0][passthru]
        for mk in metric_keys:
            xs = _finite_floats(r.get(mk) for r in rows)
            stats = _summary_stats(xs)
            base[f"{mk}_mean"] = stats["mean"]
            base[f"{mk}_std"] = stats["std"]
            base[f"{mk}_p10"] = stats["p10"]
            base[f"{mk}_p90"] = stats["p90"]
        out.append(base)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-seed HLL merge-learning sweeps (embarrassingly parallel)."
    )
    parser.add_argument("--universe-size", type=int, default=65536)
    parser.add_argument("--min-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--leaf-size", type=int, default=512)
    parser.add_argument("--zipf-alphas", type=str, default="0.8,1.0,1.2")
    parser.add_argument("--precisions", type=str, default="6,7,8,9,10,11,12")
    parser.add_argument("--train-docs", type=str, default="25,50,100,200,500,1000")
    parser.add_argument("--audit-policies", type=str, default="all,sqrt,log2,fraction")
    parser.add_argument("--audit-fixed-nodes", type=int, default=0)
    parser.add_argument("--audit-fraction", type=float, default=0.25)
    parser.add_argument("--audit-scale", type=float, default=1.0)
    parser.add_argument("--n-test", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--n-epochs", type=int, default=6)
    parser.add_argument("--batch-docs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--idem-weight", type=float, default=0.10)
    parser.add_argument("--comm-weight", type=float, default=0.10)
    parser.add_argument(
        "--weighting-modes",
        type=str,
        default="doc,leaf,token",
        help="Comma-separated weighting modes for side-by-side reporting.",
    )
    parser.add_argument(
        "--legacy-weighting-mode",
        type=str,
        default="doc",
        choices=("doc", "leaf", "token"),
        help="Explicit label for legacy scalar fields.",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument(
        "--data-seed",
        type=int,
        default=0,
        help="Fixed synthetic corpus seed shared across optimization seeds.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device mode. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="Single CUDA device index to use (when --device cuda/auto).",
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
        help="Torch thread count per worker (<=0 keeps torch default).",
    )
    parser.add_argument(
        "--auto-cpu",
        action="store_true",
        help="Auto-tune CPU workers/threads for CPU runs (uses physical cores).",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="Seconds between heartbeat progress updates (0 disables).",
    )
    parser.add_argument(
        "--json-summary",
        type=str,
        default="outputs/hll_merge_learning_summary.json",
    )
    parser.add_argument(
        "--raw-csv",
        type=str,
        default="outputs/hll_merge_learning_raw.csv",
    )
    parser.add_argument(
        "--agg-csv",
        type=str,
        default="outputs/hll_merge_learning_agg.csv",
    )
    return parser.parse_args()


def _physical_core_count() -> Optional[int]:
    try:
        out = subprocess.check_output(["lscpu"], text=True)
    except Exception:
        return None

    cores = None
    sockets = None
    for line in out.splitlines():
        if line.startswith("Core(s) per socket:"):
            try:
                cores = int(line.split(":")[1].strip())
            except Exception:
                cores = None
        elif line.startswith("Socket(s):"):
            try:
                sockets = int(line.split(":")[1].strip())
            except Exception:
                sockets = None
    if cores is not None and sockets is not None:
        return int(cores * sockets)
    return None


def _run_seed(
    seed: int,
    *,
    cfg_kwargs: dict,
    use_cuda: bool,
    cuda_device: Optional[int],
    torch_threads: int,
) -> Tuple[int, float, List[dict]]:
    started = time.time()
    if int(torch_threads) > 0:
        torch.set_num_threads(int(torch_threads))
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(int(torch_threads))
            except RuntimeError as exc:
                # ProcessPool workers may execute multiple seeds sequentially. In that case,
                # PyTorch forbids resetting the interop pool after parallel work has started.
                # Keeping the existing interop setting is safe for subsequent seeds.
                if "cannot set number of interop threads after parallel work has started" not in str(exc):
                    raise

    cfg = HLLMergeLearningConfig(
        **cfg_kwargs,
        use_cuda=bool(use_cuda),
        cuda_device=int(cuda_device) if cuda_device is not None else None,
        torch_threads=int(torch_threads),
        seed=int(seed),
    )
    runs = run_hll_merge_learning_experiment(cfg)
    rows = experiment_rows(runs)
    return int(seed), float(time.time() - started), rows


def main() -> int:
    args = parse_args()

    device_mode = str(args.device)
    if device_mode == "auto":
        use_cuda = bool(torch.cuda.is_available())
    elif device_mode == "cuda":
        use_cuda = True
    else:
        use_cuda = False

    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")

    seeds = _parse_int_csv(args.seeds)
    cuda_devices = None
    if args.cuda_devices is not None:
        cuda_devices = _parse_int_csv(args.cuda_devices)
        if len(cuda_devices) == 0:
            cuda_devices = None

    if bool(args.auto_cpu) and not use_cuda:
        phys = _physical_core_count()
        if phys is None:
            phys = max(1, int(os.cpu_count() or 1))
        args.parallel_workers = int(max(1, min(int(phys), len(seeds))))
        args.torch_threads = 1
        print(
            f"auto_cpu | physical_cores={phys} | "
            f"parallel_workers={args.parallel_workers} | torch_threads={args.torch_threads}"
        )
        sys.stdout.flush()

    cfg_kwargs = dict(
        universe_size=int(args.universe_size),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        leaf_size=int(args.leaf_size),
        zipf_alphas=_parse_float_csv(args.zipf_alphas),
        precisions=_parse_int_csv(args.precisions),
        train_docs_grid=_parse_int_csv(args.train_docs),
        audit_policies=_parse_str_csv(args.audit_policies),
        audit_fixed_nodes=int(args.audit_fixed_nodes),
        audit_fraction=float(args.audit_fraction),
        audit_scale=float(args.audit_scale),
        n_test=int(args.n_test),
        hidden_dim=int(args.hidden_dim),
        n_epochs=int(args.n_epochs),
        batch_docs=int(args.batch_docs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip_norm),
        idem_weight=float(args.idem_weight),
        comm_weight=float(args.comm_weight),
        weighting_modes=_parse_str_csv(args.weighting_modes),
        legacy_weighting_mode=str(args.legacy_weighting_mode),
        data_seed=int(args.data_seed),
    )

    started = time.time()
    raw_rows: List[dict] = []
    runtime_by_seed: List[dict] = []

    max_workers = int(max(1, args.parallel_workers))
    ctx = mp.get_context("spawn")
    total_seeds = len(seeds)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futs = []
        for idx, seed in enumerate(seeds):
            if use_cuda:
                if cuda_devices is not None:
                    assigned = int(cuda_devices[idx % len(cuda_devices)])
                else:
                    assigned = int(args.cuda_device) if args.cuda_device is not None else None
            else:
                assigned = None
            futs.append(
                ex.submit(
                    _run_seed,
                    int(seed),
                    cfg_kwargs=cfg_kwargs,
                    use_cuda=bool(use_cuda),
                    cuda_device=assigned,
                    torch_threads=int(args.torch_threads),
                )
            )

        completed = 0
        last_heartbeat = float(time.time())
        print(
            f"submitted_seeds | total={total_seeds} | "
            f"parallel_workers={max_workers} | torch_threads={int(args.torch_threads)}"
        )
        sys.stdout.flush()

        pending = set(futs)
        while pending:
            timeout = None
            if float(args.progress_interval) > 0.0:
                timeout = float(args.progress_interval)
            done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)

            for fut in done:
                seed, runtime_s, rows = fut.result()
                completed += 1
                runtime_by_seed.append({"seed": int(seed), "runtime_seconds": float(runtime_s)})
                raw_rows.extend(rows)
                elapsed = float(time.time() - started)
                mean_runtime = float(
                    statistics.mean(r["runtime_seconds"] for r in runtime_by_seed)
                )
                remaining = max(0, total_seeds - completed)
                eta = mean_runtime * float(remaining)
                print(
                    f"seed_done | {completed}/{total_seeds} | seed={seed} | "
                    f"rows={len(rows)} | runtime={runtime_s:.1f}s | "
                    f"elapsed={elapsed:.1f}s | eta~{eta:.1f}s"
                )
                sys.stdout.flush()

            if float(args.progress_interval) > 0.0:
                now = float(time.time())
                if now - last_heartbeat >= float(args.progress_interval):
                    pending_count = total_seeds - completed
                    print(
                        f"heartbeat | completed={completed}/{total_seeds} | "
                        f"pending={pending_count} | elapsed={now - started:.1f}s"
                    )
                    sys.stdout.flush()
                    last_heartbeat = now

    # Aggregate across seeds by (precision, train_docs, audit_policy).
    group_keys = ("precision", "train_docs", "audit_policy")
    metric_keys = (
        "learned_relative_rmse",
        "learned_mean_abs_rel_error",
        "learned_schedule_spread_mean",
        "learned_schedule_spread_p95",
        "merge_mse_mean",
        "distance_to_hll_floor_rel_rmse",
        "ratio_to_hll_floor_rel_rmse",
        "collapse_indicator",
        "train_audit_nodes_mean",
        "train_audit_coverage_mean",
        "train_total_queries_estimate",
        "hll_relative_rmse",
        "hll_schedule_spread_mean",
    )
    for mode in _parse_str_csv(args.weighting_modes):
        for prefix in ("hll", "learned"):
            for metric in (
                "relative_rmse",
                "mean_abs_rel_error",
                "schedule_spread_mean",
                "schedule_spread_p95",
            ):
                metric_keys = metric_keys + (
                    f"{prefix}_{metric}_{mode}",
                    f"{prefix}_{metric}_{mode}_se",
                    f"{prefix}_{metric}_{mode}_ci95_low",
                    f"{prefix}_{metric}_{mode}_ci95_high",
                )
    aggregated_rows = aggregate_rows(raw_rows, group_keys=group_keys, metric_keys=metric_keys)

    runtime_total = float(time.time() - started)
    payload = {
        "config": cfg_kwargs,
        "seeds": list(seeds),
        "execution": {
            "use_cuda": bool(use_cuda),
            "cuda_device": None if args.cuda_device is None else int(args.cuda_device),
            "cuda_devices": None if cuda_devices is None else list(cuda_devices),
            "parallel_workers": int(max_workers),
            "torch_threads_per_worker": int(args.torch_threads),
        },
        "runtime_seconds_total": runtime_total,
        "runtime_seconds_by_seed": sorted(runtime_by_seed, key=lambda r: int(r["seed"])),
        "n_raw_rows": int(len(raw_rows)),
        "raw_rows": raw_rows,
        "n_agg_rows": int(len(aggregated_rows)),
        "aggregated_rows": aggregated_rows,
    }

    json_path = Path(args.json_summary)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote_json | {json_path}")

    raw_csv = Path(args.raw_csv)
    _write_csv(raw_csv, raw_rows)
    print(f"wrote_raw_csv | {raw_csv}")

    agg_csv = Path(args.agg_csv)
    _write_csv(agg_csv, aggregated_rows)
    print(f"wrote_agg_csv | {agg_csv}")

    if len(aggregated_rows) > 0:
        gaps = [
            float(r["distance_to_hll_floor_rel_rmse_mean"])
            for r in aggregated_rows
            if r.get("distance_to_hll_floor_rel_rmse_mean") is not None
        ]
        if gaps:
            print(
                "agg_floor_gap_summary | mean | median | min | max"
            )
            print(
                f"{statistics.mean(gaps):+.6f} | {statistics.median(gaps):+.6f} | "
                f"{min(gaps):+.6f} | {max(gaps):+.6f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
