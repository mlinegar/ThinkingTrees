#!/usr/bin/env python3
"""Run tree-summary batching sweeps for local vLLM-compatible endpoints.

Examples:
  # No-server artifact/smoke run
  ./venv/bin/python scripts/run_tree_batching_benchmark.py --fake \
    --total-input-tokens 16000 --documents 2 \
    --leaf-tokens-grid 1000,2000 --summary-max-tokens-grid 256,512

  # Live vLLM smoke run
  ./venv/bin/python scripts/run_tree_batching_benchmark.py \
    --base-url http://localhost:8000/v1 \
    --total-input-tokens 16000 --documents 2 \
    --leaf-tokens-grid 1000,2000 \
    --summary-max-tokens-grid 256,512 \
    --concurrency-grid 16,32 --batch-size-grid 16,32
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.tree_batching import (  # noqa: E402
    DEFAULT_RUBRIC,
    default_output_paths,
    expand_tree_batch_grid,
    parse_positive_float_grid,
    parse_positive_int_grid,
    render_tree_batch_markdown,
    run_tree_batching_suite,
    write_tree_batch_jsonl,
    write_tree_batch_markdown,
)
from src.experiments import (  # noqa: E402
    JsonlCallTraceSink,
    ResultRow,
    benchmark_ref_from_parts,
    chat_role_ref,
    metadata_with_roles,
    method_ref_from_parts,
    oracle_ref,
    write_canonical_sidecars,
)


def _parse_dspy_workers_grid(raw: str) -> List[Optional[int]]:
    values = parse_positive_int_grid(raw, name="dspy-workers-grid", allow_zero=True)
    return [None if value == 0 else int(value) for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep C-TreePO tree-summary batching settings over a fixed token workload.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--fake", action="store_true", help="Use deterministic no-server fake mode")
    parser.add_argument("--total-input-tokens", type=int, default=32768)
    parser.add_argument("--documents", type=int, default=4)
    parser.add_argument("--leaf-tokens-grid", default="1000,2000,4000,8000")
    parser.add_argument("--summary-max-tokens-grid", default="256,512,1024")
    parser.add_argument("--concurrency-grid", default="16,32,64,128")
    parser.add_argument("--batch-size-grid", default="16,32,64")
    parser.add_argument("--batch-timeout-grid", default="0.01,0.02,0.05")
    parser.add_argument(
        "--dspy-workers-grid",
        default="0",
        help="Comma-separated offload worker caps; 0 means unset/default",
    )
    parser.add_argument("--limit-points", type=int, default=0)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--await-response-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--metrics-poll-seconds", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--rubric", default=DEFAULT_RUBRIC)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--prompt-overhead-tokens", type=int, default=512)
    parser.add_argument("--budget-safety-fraction", type=float, default=0.90)
    parser.add_argument("--min-success-rate", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output-prefix", default="tree_batching")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


async def _main_async(args: argparse.Namespace) -> int:
    leaf_grid = parse_positive_int_grid(args.leaf_tokens_grid, name="leaf-tokens-grid")
    out_grid = parse_positive_int_grid(args.summary_max_tokens_grid, name="summary-max-tokens-grid")
    conc_grid = parse_positive_int_grid(args.concurrency_grid, name="concurrency-grid")
    batch_grid = parse_positive_int_grid(args.batch_size_grid, name="batch-size-grid")
    timeout_grid = parse_positive_float_grid(args.batch_timeout_grid, name="batch-timeout-grid")
    worker_grid = _parse_dspy_workers_grid(args.dspy_workers_grid)

    points = expand_tree_batch_grid(
        leaf_tokens=leaf_grid,
        summary_max_tokens=out_grid,
        max_concurrent_requests=conc_grid,
        batch_sizes=batch_grid,
        batch_timeouts=timeout_grid,
        dspy_workers=worker_grid,
    )
    if args.limit_points and int(args.limit_points) > 0:
        points = points[: int(args.limit_points)]

    await_timeout = (
        None
        if float(args.await_response_timeout_seconds or 0.0) <= 0.0
        else float(args.await_response_timeout_seconds)
    )
    default_jsonl, default_md = default_output_paths(
        output_dir=Path(args.output_dir),
        prefix=str(args.output_prefix),
    )
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else default_jsonl
    output_md = Path(args.output_md) if args.output_md else default_md
    call_trace_path = output_jsonl.parent / f"{output_jsonl.stem}_calls.jsonl"
    suite = await run_tree_batching_suite(
        points=points,
        base_url=str(args.base_url),
        total_input_tokens=int(args.total_input_tokens),
        document_count=int(args.documents),
        request_timeout_seconds=float(args.request_timeout_seconds),
        await_response_timeout_seconds=await_timeout,
        metrics_poll_seconds=float(args.metrics_poll_seconds),
        api_key=str(args.api_key),
        temperature=float(args.temperature),
        rubric=str(args.rubric),
        fake=bool(args.fake),
        max_model_len=int(args.max_model_len),
        max_num_seqs=int(args.max_num_seqs),
        max_num_batched_tokens=int(args.max_num_batched_tokens),
        prompt_overhead_tokens=int(args.prompt_overhead_tokens),
        budget_safety_fraction=float(args.budget_safety_fraction),
        min_success_rate=float(args.min_success_rate),
        call_sink=JsonlCallTraceSink(
            call_trace_path,
            defaults={
                "method_id": "tree_batching_benchmark",
                "runner_id": "run_tree_batching_suite",
                "surface": "chat_openai",
            },
        )
        if not bool(args.fake)
        else None,
    )

    write_tree_batch_jsonl(output_jsonl=output_jsonl, suite=suite)
    write_tree_batch_markdown(output_markdown=output_md, suite=suite)
    benchmark_ref = benchmark_ref_from_parts(
        family="tree_batching",
        scope="throughput",
        name="tree_batching",
        dataset_id="synthetic",
        metadata=dict(suite.config),
    )
    method_ref = method_ref_from_parts(
        family="tree_batching_benchmark",
        variant="fake" if bool(args.fake) else "live",
        adapter="tree_batching",
        metadata=metadata_with_roles(
            {"fake": bool(args.fake)},
            roles={
                "summarizer": chat_role_ref(
                    role="summarizer",
                    model="default",
                    base_url=str(args.base_url),
                )
            },
            oracle=oracle_ref(kind="synthetic_workload", source="tree_batching"),
        ),
    )
    artifacts = {
        "tree_batching_jsonl": str(output_jsonl),
        "tree_batching_md": str(output_md),
    }
    if call_trace_path.exists():
        artifacts["calls_jsonl"] = str(call_trace_path)
    rows = []
    if suite.summary.best_tokens_point is not None:
        rows.append(
            ResultRow(
                experiment_id="",
                phase="benchmark",
                benchmark_ref=benchmark_ref,
                method_ref=method_ref,
                metric_name="best_tokens_per_second",
                metric_value=float(suite.summary.best_tokens_point.tokens_per_second),
                artifact_refs=tuple(artifacts.keys()),
                metadata={"best_point": suite.summary.best_tokens_point.to_dict()},
            )
        )
    write_canonical_sidecars(
        output_jsonl.parent / f"{output_jsonl.stem}_run",
        title="tree_batching_benchmark",
        adapter_id="tree_batching",
        benchmark_refs=(benchmark_ref,),
        method_refs=(method_ref,),
        phases=("benchmark",),
        artifacts=artifacts,
        result_rows=tuple(rows),
        state="completed",
        metadata={"fake": bool(args.fake)},
        launch_command=sys.argv,
        report_profiles=("runtime_eval_summary",),
    )

    print(render_tree_batch_markdown(suite))
    print(f"Wrote JSONL: {output_jsonl}")
    print(f"Wrote Markdown: {output_md}")
    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
