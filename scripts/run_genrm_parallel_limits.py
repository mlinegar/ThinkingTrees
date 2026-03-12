#!/usr/bin/env python3
"""
Run fast-vs-think GenRM limit sweeps in parallel on two endpoints.

Intended setup with 4 GPUs:
- fast mode GenRM server on GPUs 0,1 (e.g. port 8001)
- think mode GenRM server on GPUs 2,3 (e.g. port 8002)

This can optionally auto-start/stop both servers for reproducible A/B runs.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

# Allow running from repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.pipeline_limits import (  # noqa: E402
    StepSweepResult,
    expand_genrm_steps,
    format_human_summary,
    parse_concurrency_grid,
    run_pipeline_throughput_suite,
)
from src.benchmark.throughput import VLLMServerManager  # noqa: E402


def _dedupe_keep_order(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def _parse_steps(step_csv: str) -> List[str]:
    raw = [p.strip() for p in str(step_csv).split(",") if p.strip()]
    if not raw:
        raise ValueError("At least one step must be provided")
    valid = {"genrm_batch", "genrm_raw"}
    bad = [x for x in raw if x not in valid]
    if bad:
        raise ValueError(f"Invalid step(s): {bad}. Valid: {sorted(valid)}")
    return raw


@dataclass
class ComparisonRow:
    step_base: str
    concurrency: int
    fast_req_per_s: float
    think_req_per_s: float
    req_per_s_ratio_fast_over_think: float
    fast_p95_ms: float
    think_p95_ms: float
    fast_success_rate: float
    think_success_rate: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def build_comparison_rows(
    *,
    fast_result: Dict[str, StepSweepResult],
    think_result: Dict[str, StepSweepResult],
    base_steps: Sequence[str],
) -> List[ComparisonRow]:
    rows: List[ComparisonRow] = []
    for base in base_steps:
        fast_step = f"{base}_fast"
        think_step = f"{base}_think"
        fast_points = {p.concurrency: p for p in fast_result[fast_step].points}
        think_points = {p.concurrency: p for p in think_result[think_step].points}
        for concurrency in sorted(set(fast_points.keys()) & set(think_points.keys())):
            fp = fast_points[concurrency]
            tp = think_points[concurrency]
            ratio = fp.requests_per_second / tp.requests_per_second if tp.requests_per_second > 0 else float("inf")
            rows.append(
                ComparisonRow(
                    step_base=base,
                    concurrency=concurrency,
                    fast_req_per_s=fp.requests_per_second,
                    think_req_per_s=tp.requests_per_second,
                    req_per_s_ratio_fast_over_think=ratio,
                    fast_p95_ms=fp.latency_p95_ms,
                    think_p95_ms=tp.latency_p95_ms,
                    fast_success_rate=fp.success_rate,
                    think_success_rate=tp.success_rate,
                )
            )
    return rows


def print_comparison_table(rows: Sequence[ComparisonRow]) -> None:
    print("=" * 92)
    print("Parallel GenRM Mode Comparison (Fast vs Think)")
    print("=" * 92)
    print("step         c  fast_req/s  think_req/s  ratio(f/t)  fast_p95  think_p95  fast_ok  think_ok")
    for r in rows:
        print(
            f"{r.step_base:<12} {r.concurrency:>2} "
            f"{r.fast_req_per_s:>10.3f} {r.think_req_per_s:>11.3f} "
            f"{r.req_per_s_ratio_fast_over_think:>10.2f} "
            f"{r.fast_p95_ms:>9.1f} {r.think_p95_ms:>10.1f} "
            f"{100.0 * r.fast_success_rate:>6.1f}% {100.0 * r.think_success_rate:>8.1f}%"
        )
    print()


def write_comparison_csv(path: Path, rows: Sequence[ComparisonRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "step_base",
        "concurrency",
        "fast_req_per_s",
        "think_req_per_s",
        "req_per_s_ratio_fast_over_think",
        "fast_p95_ms",
        "think_p95_ms",
        "fast_success_rate",
        "think_success_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


async def _run_mode_sweep(
    *,
    mode: str,
    genrm_url: str,
    base_steps: Sequence[str],
    concurrency_grid: Sequence[int],
    min_requests_per_point: int,
    requests_per_concurrency: int,
    warmup_requests: int,
    timeout_seconds: float,
    max_tokens: int,
    min_success_rate: float,
    max_p95_latency_ms: float,
) -> Dict[str, StepSweepResult]:
    steps = expand_genrm_steps(list(base_steps), [mode])
    return await run_pipeline_throughput_suite(
        steps=steps,
        concurrency_grid=concurrency_grid,
        min_requests_per_point=min_requests_per_point,
        requests_per_concurrency=requests_per_concurrency,
        warmup_requests=warmup_requests,
        task_url="http://localhost:8000/v1",  # Unused for GenRM-only steps.
        task_replica_url=None,
        genrm_url=genrm_url,
        task_timeout_seconds=120.0,
        genrm_timeout_seconds=timeout_seconds,
        task_max_tokens=256,
        genrm_max_tokens=max_tokens,
        task_batch_timeout=0.05,
        task_chars=1200,
        api_key="EMPTY",
        genrm_disable_thinking=(mode == "fast"),
        genrm_force_json_response=(mode == "fast"),
        genrm_temperature=0.6,
        genrm_top_p=0.95,
        min_success_rate=min_success_rate,
        max_p95_latency_ms=max_p95_latency_ms,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel GenRM fast-vs-think throughput sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        default="genrm_batch,genrm_raw",
        help="Comma-separated subset of: genrm_batch,genrm_raw",
    )
    parser.add_argument("--concurrency-grid", default="1,2,4,8,12,16")
    parser.add_argument("--min-requests-per-point", type=int, default=48)
    parser.add_argument("--requests-per-concurrency", type=int, default=6)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--genrm-timeout-seconds", type=float, default=360.0)
    parser.add_argument("--genrm-max-tokens", type=int, default=256)
    parser.add_argument("--min-success-rate", type=float, default=0.98)
    parser.add_argument("--max-p95-latency-ms", type=float, default=0.0)

    parser.add_argument("--fast-url", default="http://localhost:8001/v1")
    parser.add_argument("--think-url", default="http://localhost:8002/v1")
    parser.add_argument(
        "--auto-start-servers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Auto-start/stop two GenRM servers for the sweep",
    )
    parser.add_argument("--profile", default="genrm-nvfp4")
    parser.add_argument("--fast-port", type=int, default=8001)
    parser.add_argument("--think-port", type=int, default=8002)
    parser.add_argument("--fast-cuda-devices", default="0,1")
    parser.add_argument("--think-cuda-devices", default="2,3")
    parser.add_argument("--tensor-parallel", type=int, default=2)
    parser.add_argument("--startup-timeout-seconds", type=float, default=480.0)

    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/genrm_parallel_fast_vs_think.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/genrm_parallel_fast_vs_think.csv"),
        help="Comparison table CSV",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    base_steps = _dedupe_keep_order(_parse_steps(args.steps))
    concurrency_grid = parse_concurrency_grid(args.concurrency_grid)

    fast_manager = None
    think_manager = None
    if args.auto_start_servers:
        fast_manager = VLLMServerManager(
            profile=args.profile,
            port=int(args.fast_port),
            cuda_devices=str(args.fast_cuda_devices),
            tensor_parallel=int(args.tensor_parallel),
            startup_timeout=float(args.startup_timeout_seconds),
        )
        think_manager = VLLMServerManager(
            profile=args.profile,
            port=int(args.think_port),
            cuda_devices=str(args.think_cuda_devices),
            tensor_parallel=int(args.tensor_parallel),
            startup_timeout=float(args.startup_timeout_seconds),
        )
        logging.info(
            "Starting GenRM servers in parallel: fast=%s (GPU %s), think=%s (GPU %s)",
            args.fast_url,
            args.fast_cuda_devices,
            args.think_url,
            args.think_cuda_devices,
        )
        await asyncio.gather(fast_manager.start(), think_manager.start())

    try:
        fast_task = _run_mode_sweep(
            mode="fast",
            genrm_url=str(args.fast_url),
            base_steps=base_steps,
            concurrency_grid=concurrency_grid,
            min_requests_per_point=int(args.min_requests_per_point),
            requests_per_concurrency=int(args.requests_per_concurrency),
            warmup_requests=int(args.warmup_requests),
            timeout_seconds=float(args.genrm_timeout_seconds),
            max_tokens=int(args.genrm_max_tokens),
            min_success_rate=float(args.min_success_rate),
            max_p95_latency_ms=float(args.max_p95_latency_ms),
        )
        think_task = _run_mode_sweep(
            mode="think",
            genrm_url=str(args.think_url),
            base_steps=base_steps,
            concurrency_grid=concurrency_grid,
            min_requests_per_point=int(args.min_requests_per_point),
            requests_per_concurrency=int(args.requests_per_concurrency),
            warmup_requests=int(args.warmup_requests),
            timeout_seconds=float(args.genrm_timeout_seconds),
            max_tokens=int(args.genrm_max_tokens),
            min_success_rate=float(args.min_success_rate),
            max_p95_latency_ms=float(args.max_p95_latency_ms),
        )
        fast_result, think_result = await asyncio.gather(fast_task, think_task)
    finally:
        if args.auto_start_servers:
            if fast_manager is not None:
                fast_manager.stop()
            if think_manager is not None:
                think_manager.stop()

    print("\n=== FAST MODE RESULT ===")
    print(format_human_summary(fast_result))
    print("\n=== THINK MODE RESULT ===")
    print(format_human_summary(think_result))

    comparison_rows = build_comparison_rows(
        fast_result=fast_result,
        think_result=think_result,
        base_steps=base_steps,
    )
    print_comparison_table(comparison_rows)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "base_steps": base_steps,
            "concurrency_grid": concurrency_grid,
            "min_requests_per_point": int(args.min_requests_per_point),
            "requests_per_concurrency": int(args.requests_per_concurrency),
            "warmup_requests": int(args.warmup_requests),
            "genrm_timeout_seconds": float(args.genrm_timeout_seconds),
            "genrm_max_tokens": int(args.genrm_max_tokens),
            "min_success_rate": float(args.min_success_rate),
            "max_p95_latency_ms": float(args.max_p95_latency_ms),
            "fast_url": str(args.fast_url),
            "think_url": str(args.think_url),
            "auto_start_servers": bool(args.auto_start_servers),
            "profile": str(args.profile),
        },
        "fast": {k: v.to_dict() for k, v in fast_result.items()},
        "think": {k: v.to_dict() for k, v in think_result.items()},
        "comparison": [row.to_dict() for row in comparison_rows],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_comparison_csv(args.output_csv, comparison_rows)

    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV:  {args.output_csv}")
    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("Interrupted")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
