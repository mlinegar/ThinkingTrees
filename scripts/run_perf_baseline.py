#!/usr/bin/env python3
"""
Standard perf/quality baseline runner for ThinkingTrees.

Runs a fixed workload twice (cold + warm) and writes a JSON artifact under:
  outputs/perf_baselines/<utc_timestamp>.json

Workload definition (defaults):
- 20 Manifesto docs
- 3 iterations (repeat the same docs 3 times)
- BatchedDocPipeline with scoring enabled (MAE computed vs reference RILE)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.documents import DocumentSample, DocumentResult
from src.core.engram_memory import EngramMemoryConfig, extract_engram_memory_items
from src.pipelines.batched import BatchedDocPipeline, BatchedPipelineConfig
from src.tasks import get_task
from src.tasks.manifesto.data_loader import ManifestoDataset


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _git_short_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return out.decode("utf-8", errors="replace").strip() or None


def _mean_stderr(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    n = len(values)
    mean = sum(values) / n
    if n <= 1:
        return mean, None
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    stderr = (var ** 0.5) / (n ** 0.5)
    return mean, stderr


def _batch_stats_to_dict(stats: Any) -> Dict[str, Any]:
    if stats is None:
        return {}
    return {
        "total_requests": int(getattr(stats, "total_requests", 0)),
        "completed_requests": int(getattr(stats, "completed_requests", 0)),
        "failed_requests": int(getattr(stats, "failed_requests", 0)),
        "cache_hits": int(getattr(stats, "cache_hits", 0)),
        "cache_misses": int(getattr(stats, "cache_misses", 0)),
        "cache_writes": int(getattr(stats, "cache_writes", 0)),
        "total_tokens": int(getattr(stats, "total_tokens", 0)),
        "prompt_tokens": int(getattr(stats, "prompt_tokens", 0)),
        "completion_tokens": int(getattr(stats, "completion_tokens", 0)),
        "avg_latency_ms": float(getattr(stats, "avg_latency_ms", 0.0)),
        "wall_clock_seconds": float(getattr(stats, "wall_clock_elapsed", 0.0)),
    }


@dataclass(frozen=True)
class IterationResult:
    iteration: int
    wall_clock_seconds: float
    batch_stats: Dict[str, Any]
    docs_processed: int
    docs_succeeded: int
    mae: Optional[float]
    mae_stderr: Optional[float]
    brittle_retention: Optional[float]


def _compute_iteration_metrics(
    *,
    results: List[DocumentResult],
    batch_stats: Any,
    wall_clock_seconds: float,
    brittle_items_by_doc: Dict[str, List[str]],
) -> Tuple[int, int, Optional[float], Optional[float], Optional[float]]:
    succeeded = [r for r in results if r is not None and not getattr(r, "error", None)]
    errors: List[float] = []
    for r in succeeded:
        if r.estimated_score is None or r.reference_score is None:
            continue
        errors.append(abs(float(r.estimated_score) - float(r.reference_score)))
    mae, mae_stderr = _mean_stderr(errors)

    kept = 0
    total = 0
    for r in succeeded:
        items = brittle_items_by_doc.get(str(r.doc_id), [])
        if not items:
            continue
        summary = str(r.final_summary or "")
        for item in items:
            total += 1
            if item in summary:
                kept += 1
    brittle_retention = (kept / total) if total > 0 else None

    return (
        len(results),
        len(succeeded),
        mae,
        mae_stderr,
        brittle_retention,
    )


def _run_once(
    *,
    label: str,
    docs: List[DocumentSample],
    task_name: str,
    task_model_urls: List[str],
    iterations: int,
    max_chunk_chars: int,
    concurrent_docs: int,
    concurrent_requests: int,
    routing_policy: str,
    enable_engram_memory: bool,
    engram_max_items: int,
    engram_max_chars: int,
    response_cache_dir: Optional[Path],
    response_cache_mode: str,
    response_cache_types: str,
) -> Dict[str, Any]:
    task = get_task(task_name)
    rubric = task.create_rubric()
    task_context = task.get_task_context()
    prompt_builders = task.create_prompt_builders()

    engram_cfg = EngramMemoryConfig(
        enabled=bool(enable_engram_memory),
        max_items=int(engram_max_items),
        max_chars=int(engram_max_chars),
    )

    if response_cache_dir is not None:
        os.environ["TT_RESPONSE_CACHE_DIR"] = str(response_cache_dir)
        os.environ["TT_RESPONSE_CACHE_MODE"] = str(response_cache_mode)
        os.environ["TT_RESPONSE_CACHE_REQUEST_TYPES"] = str(response_cache_types)

    brittle_items_by_doc: Dict[str, List[str]] = {}
    if engram_cfg.enabled:
        for doc in docs:
            brittle_items_by_doc[str(doc.doc_id)] = extract_engram_memory_items(doc.text, engram_cfg)

    iteration_rows: List[IterationResult] = []
    for idx in range(int(iterations)):
        cfg = BatchedPipelineConfig(
            task_model_url=task_model_urls[0],
            task_model_urls=task_model_urls if len(task_model_urls) > 1 else None,
            max_concurrent_documents=int(concurrent_docs),
            max_concurrent_requests=int(concurrent_requests),
            max_chunk_chars=int(max_chunk_chars),
            routing_policy=str(routing_policy),
            show_progress=False,
            rubric=rubric,
            task_context=task_context,
            prompt_builders=prompt_builders,
            score_parser=task.parse_score,
            run_baseline=False,
            engram_memory=engram_cfg,
        )
        pipeline = BatchedDocPipeline(config=cfg)

        started = time.time()
        results = pipeline.process_batch(docs, show_progress=False)
        elapsed = time.time() - started

        batch_stats = pipeline.last_stats
        docs_total, docs_succeeded, mae, mae_stderr, brittle_retention = _compute_iteration_metrics(
            results=results,
            batch_stats=batch_stats,
            wall_clock_seconds=elapsed,
            brittle_items_by_doc=brittle_items_by_doc,
        )
        iteration_rows.append(
            IterationResult(
                iteration=idx + 1,
                wall_clock_seconds=float(elapsed),
                batch_stats=_batch_stats_to_dict(batch_stats),
                docs_processed=int(docs_total),
                docs_succeeded=int(docs_succeeded),
                mae=mae,
                mae_stderr=mae_stderr,
                brittle_retention=brittle_retention,
            )
        )

    total_wall = sum(row.wall_clock_seconds for row in iteration_rows)
    total_docs = sum(row.docs_processed for row in iteration_rows)
    total_tokens = sum(int(row.batch_stats.get("total_tokens", 0) or 0) for row in iteration_rows)
    agg_mae_vals = [row.mae for row in iteration_rows if row.mae is not None]
    agg_mae, agg_mae_stderr = _mean_stderr([float(v) for v in agg_mae_vals])
    brittle_vals = [row.brittle_retention for row in iteration_rows if row.brittle_retention is not None]
    brittle_mean, _ = _mean_stderr([float(v) for v in brittle_vals])

    return {
        "label": label,
        "task": task_name,
        "task_model_urls": task_model_urls,
        "iterations": [asdict(r) for r in iteration_rows],
        "aggregate": {
            "wall_clock_seconds": float(total_wall),
            "docs_processed": int(total_docs),
            "docs_per_second": float(total_docs / max(total_wall, 1e-9)),
            "tokens_total": int(total_tokens),
            "tokens_per_second": float(total_tokens / max(total_wall, 1e-9)),
            "mae_mean": agg_mae,
            "mae_stderr": agg_mae_stderr,
            "brittle_retention_mean": brittle_mean,
        },
        "response_cache": {
            "dir": str(response_cache_dir) if response_cache_dir is not None else None,
            "mode": str(response_cache_mode),
            "request_types": str(response_cache_types),
        },
        "engram_memory": asdict(engram_cfg),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ThinkingTrees perf baseline (cold + warm).")
    parser.add_argument("--task", default="manifesto_rile", help="Task name (default: manifesto_rile)")
    parser.add_argument("--samples", type=int, default=20, help="Number of docs (default: 20)")
    parser.add_argument("--iterations", type=int, default=3, help="Iterations per run (default: 3)")
    parser.add_argument("--seed", type=int, default=0, help="Sample seed (default: 0)")

    parser.add_argument("--port", type=int, default=8000, help="Task model port (default: 8000)")
    parser.add_argument(
        "--ports",
        type=int,
        nargs="*",
        default=None,
        help="Optional multiple task ports for multi-server routing (e.g., 8000 8002).",
    )
    parser.add_argument("--routing-policy", default="affinity_load_aware", help="Routing policy for multi-server.")

    parser.add_argument("--max-chunk-chars", type=int, default=8000, help="Chunk size in chars (default: 8000)")
    parser.add_argument("--concurrent-docs", type=int, default=20, help="Concurrent docs (default: 20)")
    parser.add_argument("--concurrent-requests", type=int, default=200, help="Concurrent requests (default: 200)")

    parser.add_argument("--countries", nargs="+", type=int, default=[51, 41], help="Country codes filter.")
    parser.add_argument("--min-year", type=int, default=2000, help="Min year filter (default: 2000)")

    parser.add_argument("--engram-memory", action="store_true", help="Enable Engram STATIC MEMORY injection.")
    parser.add_argument("--engram-memory-max-items", type=int, default=32)
    parser.add_argument("--engram-memory-max-chars", type=int, default=1200)

    parser.add_argument(
        "--response-cache-dir",
        type=Path,
        default=None,
        help="Disk cache dir for chat responses (default: outputs/perf_baselines/cache/<stamp>).",
    )
    parser.add_argument(
        "--response-cache-mode",
        default="readwrite",
        choices=["off", "read", "write", "readwrite"],
        help="Response cache mode (default: readwrite).",
    )
    parser.add_argument(
        "--response-cache-request-types",
        default="summarize,merge,score,baseline",
        help="Comma-separated request types to cache (default: summarize,merge,score,baseline).",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit JSON output path.",
    )
    args = parser.parse_args()

    ports = args.ports if args.ports else [int(args.port)]
    task_model_urls = [f"http://localhost:{int(p)}/v1" for p in ports]

    dataset = ManifestoDataset(countries=args.countries, min_year=args.min_year)
    all_ids = dataset.get_all_ids()
    rng = random.Random(int(args.seed))
    rng.shuffle(all_ids)
    selected = all_ids[: int(args.samples)]

    docs: List[DocumentSample] = []
    for mid in selected:
        sample = dataset.get_sample(mid)
        if sample is None:
            continue
        docs.append(
            DocumentSample(
                doc_id=str(sample.manifesto_id),
                text=str(sample.text or ""),
                reference_score=float(sample.rile),
                metadata={
                    "country_code": int(sample.country_code),
                    "year": int(sample.year),
                    "party_id": int(sample.party_id),
                },
            )
        )

    if not docs:
        raise RuntimeError("No documents loaded; check dataset path/filters.")

    stamp = _utc_stamp()
    out_dir = PROJECT_ROOT / "outputs" / "perf_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.response_cache_dir
    if cache_dir is None and args.response_cache_mode != "off":
        cache_dir = out_dir / "cache" / stamp
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_short_sha(),
        "workload": {
            "task": str(args.task),
            "samples": int(len(docs)),
            "iterations": int(args.iterations),
            "seed": int(args.seed),
            "max_chunk_chars": int(args.max_chunk_chars),
            "concurrent_docs": int(args.concurrent_docs),
            "concurrent_requests": int(args.concurrent_requests),
            "routing_policy": str(args.routing_policy),
            "engram_memory": bool(args.engram_memory),
        },
        "runs": [],
    }

    payload["runs"].append(
        _run_once(
            label="cold",
            docs=docs,
            task_name=str(args.task),
            task_model_urls=task_model_urls,
            iterations=int(args.iterations),
            max_chunk_chars=int(args.max_chunk_chars),
            concurrent_docs=int(args.concurrent_docs),
            concurrent_requests=int(args.concurrent_requests),
            routing_policy=str(args.routing_policy),
            enable_engram_memory=bool(args.engram_memory),
            engram_max_items=int(args.engram_memory_max_items),
            engram_max_chars=int(args.engram_memory_max_chars),
            response_cache_dir=cache_dir,
            response_cache_mode=str(args.response_cache_mode),
            response_cache_types=str(args.response_cache_request_types),
        )
    )
    payload["runs"].append(
        _run_once(
            label="warm",
            docs=docs,
            task_name=str(args.task),
            task_model_urls=task_model_urls,
            iterations=int(args.iterations),
            max_chunk_chars=int(args.max_chunk_chars),
            concurrent_docs=int(args.concurrent_docs),
            concurrent_requests=int(args.concurrent_requests),
            routing_policy=str(args.routing_policy),
            enable_engram_memory=bool(args.engram_memory),
            engram_max_items=int(args.engram_memory_max_items),
            engram_max_chars=int(args.engram_memory_max_chars),
            response_cache_dir=cache_dir,
            response_cache_mode=str(args.response_cache_mode),
            response_cache_types=str(args.response_cache_request_types),
        )
    )

    out_path = args.output or (out_dir / f"{stamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

