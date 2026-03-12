#!/usr/bin/env python3
"""
Run manifesto examples through the batched OPS pipeline.

This script is focused on paper-ready RILE examples:
- fixed chunking by character budget (default: 8000 chars)
- batched tree construction/scoring path
- explicit per-manifesto predicted-vs-expert RILE reporting
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add project root for direct script execution.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.documents import DocumentSample
from src.core.strategy import DSPyStrategy
from src.pipelines.batched import BatchedDocPipeline, BatchedPipelineConfig
from src.preprocessing.chunker import chunk_for_ops
from src.tasks.manifesto import ManifestoDataset, RILE_SCALE
from src.tasks.manifesto.rubrics import RILE_PRESERVATION_RUBRIC, RILE_TASK_CONTEXT
from src.tasks import get_task
from src.tasks.prompting import (
    PromptBuilders,
    default_merge_prompt,
    default_summarize_prompt,
    parse_numeric_score,
)
from src.config.dspy_config import configure_dspy, create_vllm_lm, create_vllm_lm_multi
from src.config.settings import load_settings


logger = logging.getLogger(__name__)

DEFAULT_IDS = [
    "51320_198306",  # Labour 1983
    "51620_198306",  # Conservative 1983
    "51320_199705",  # Labour 1997
]


def normalize_rile(raw_value: Optional[float]) -> Optional[float]:
    if raw_value is None:
        return None
    normalized = RILE_SCALE.normalize(float(raw_value))
    return max(0.0, min(1.0, normalized))


def denormalize_rile(normalized_value: Optional[float]) -> Optional[float]:
    if normalized_value is None:
        return None
    return float(RILE_SCALE.denormalize(float(normalized_value)))


def parse_score(response: str) -> Optional[float]:
    raw = parse_numeric_score(response, min_value=-100.0, max_value=100.0)
    if raw is None:
        return None
    return normalize_rile(raw)


def _extract_prediction_score(prediction: Any) -> Optional[float]:
    if prediction is None:
        return None
    if isinstance(prediction, dict):
        for key in ("score", "value", "prediction"):
            if key in prediction:
                try:
                    return float(prediction[key])
                except Exception:
                    continue
        return None
    for attr in ("score", "value"):
        value = getattr(prediction, attr, None)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    try:
        return float(prediction)
    except Exception:
        return None


def _predict_summary_score(scorer: Any, summary: str, task_context: str) -> Optional[float]:
    """Call a scorer across common signatures and return normalized score."""
    for kwargs in (
        {"text": summary, "task_context": task_context},
        {"summary": summary, "rubric": task_context, "original_content": summary},
        {"summary": summary, "task_context": task_context},
    ):
        try:
            prediction = scorer(**kwargs)
            score = _extract_prediction_score(prediction)
            if score is not None:
                return max(0.0, min(1.0, score))
        except TypeError:
            continue
        except Exception:
            break
    return None


def build_score_prompt(summary: str, task_context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert CMP manifesto coder. "
                "Return exactly one numeric RILE score between -100 and +100."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{task_context}\n\n"
                f"SUMMARY:\n{summary}\n\n"
                "Output only the numeric RILE score in [-100, +100]."
            ),
        },
    ]


def to_document_sample(sample) -> DocumentSample:
    return DocumentSample(
        doc_id=sample.manifesto_id,
        text=sample.text,
        reference_score=normalize_rile(sample.rile),
        metadata={
            "party_name": sample.party_name,
            "party_abbrev": sample.party_abbrev,
            "country_name": sample.country_name,
            "country_code": sample.country_code,
            "year": sample.year,
            "rile_raw": sample.rile,
        },
    )


def create_parser() -> argparse.ArgumentParser:
    settings = load_settings()
    summarizer_cfg = (
        (settings.get("generation", {}) or {}).get("summarizer", {}) if isinstance(settings, dict) else {}
    )
    default_dspy_temperature = float(summarizer_cfg.get("temperature", 0.5))
    default_dspy_max_tokens = int(summarizer_cfg.get("max_tokens", 4096))

    parser = argparse.ArgumentParser(
        description="Run manifesto IDs through the batched OPS RILE pipeline.",
    )
    parser.add_argument("--ids", nargs="+", default=DEFAULT_IDS, help="Manifesto IDs to run")
    parser.add_argument("--chunk-size", type=int, default=8000, help="Chunk size in characters")
    parser.add_argument("--port", type=int, default=8000, help="Task model port")
    parser.add_argument(
        "--ports",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of task model ports for load balancing (overrides --port)",
    )
    parser.add_argument("--concurrent-docs", type=int, default=8, help="Concurrent documents")
    parser.add_argument("--concurrent-requests", type=int, default=200, help="Concurrent LLM requests")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--batch-timeout", type=float, default=0.02, help="Batch timeout (seconds)")
    parser.add_argument("--no-baseline", action="store_true", help="Disable baseline scoring")
    parser.add_argument("--countries", type=int, nargs="+", default=[51], help="Country filter for dataset load")
    parser.add_argument("--min-year", type=int, default=1900, help="Min year filter for dataset load")
    parser.add_argument("--dry-run", action="store_true", help="Only report chunk stats; skip model calls")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--leaf-module-path", type=Path, default=None, help="Path to optimized leaf summarizer module JSON")
    parser.add_argument("--merge-module-path", type=Path, default=None, help="Path to optimized merge summarizer module JSON")
    parser.add_argument("--scorer-module-path", type=Path, default=None, help="Optional path to optimized scorer module JSON")
    parser.add_argument(
        "--dspy-temperature",
        type=float,
        default=default_dspy_temperature,
        help="DSPy temperature for leaf/merge summaries (default: generation.summarizer.temperature)",
    )
    parser.add_argument(
        "--dspy-max-tokens",
        type=int,
        default=default_dspy_max_tokens,
        help="DSPy max_tokens for leaf/merge summaries (default: generation.summarizer.max_tokens)",
    )
    parser.add_argument(
        "--use-published-modules",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-load published optimized modules when explicit paths are omitted",
    )
    parser.add_argument(
        "--published-root-dir",
        type=Path,
        default=Path("outputs/latest/manifesto_rile"),
        help="Root directory containing published modules (default: outputs/latest/manifesto_rile)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    dataset = ManifestoDataset(
        countries=args.countries,
        min_year=args.min_year,
        require_text=True,
    )

    loaded_samples = []
    missing_ids = []
    for manifesto_id in args.ids:
        sample = dataset.get_sample(manifesto_id)
        if sample is None:
            missing_ids.append(manifesto_id)
            continue
        loaded_samples.append(sample)

    if missing_ids:
        logger.warning("Missing IDs: %s", ", ".join(missing_ids))
    if not loaded_samples:
        logger.error("No manifestos loaded. Nothing to run.")
        return 1

    analysis_rows = []
    for sample in loaded_samples:
        chunks = chunk_for_ops(sample.text, max_chars=args.chunk_size, strategy="axis")
        analysis_rows.append(
            {
                "manifesto_id": sample.manifesto_id,
                "party_name": sample.party_name,
                "year": sample.year,
                "chars": len(sample.text),
                "chunk_size": args.chunk_size,
                "chunk_count": len(chunks),
                "expert_rile": sample.rile,
            }
        )

    logger.info("Chunk analysis (strategy=axis, max_chars=%d):", args.chunk_size)
    for row in analysis_rows:
        logger.info(
            "  %s | %s %s | chars=%d | chunks=%d | expert=%+.1f",
            row["manifesto_id"],
            row["party_name"],
            row["year"],
            row["chars"],
            row["chunk_count"],
            row["expert_rile"],
        )

    published_module_dir = args.published_root_dir / "trained_modules"
    if args.use_published_modules and args.leaf_module_path is None and args.merge_module_path is None:
        leaf_candidate = published_module_dir / "leaf_summarizer_final.json"
        merge_candidate = published_module_dir / "merge_summarizer_final.json"
        scorer_candidate = published_module_dir / "scorer_final.json"
        if leaf_candidate.exists() and merge_candidate.exists():
            args.leaf_module_path = leaf_candidate
            args.merge_module_path = merge_candidate
            if args.scorer_module_path is None and scorer_candidate.exists():
                args.scorer_module_path = scorer_candidate
            logger.info("Using published optimized modules from %s", published_module_dir)

    use_dspy_modules = args.leaf_module_path is not None or args.merge_module_path is not None
    if args.leaf_module_path is None and args.merge_module_path is not None:
        logger.error("--merge-module-path requires --leaf-module-path")
        return 1
    if args.merge_module_path is None and args.leaf_module_path is not None:
        logger.error("--leaf-module-path requires --merge-module-path")
        return 1

    payload = {
        "timestamp": datetime.now().isoformat(),
        "mode": (
            "dry_run"
            if args.dry_run
            else ("batched_dspy_modules" if use_dspy_modules else "batched")
        ),
        "chunk_analysis": analysis_rows,
        "results": [],
        "missing_ids": missing_ids,
        "config": {
            "port": args.port,
            "ports": args.ports,
            "chunk_size": args.chunk_size,
            "concurrent_docs": args.concurrent_docs,
            "concurrent_requests": args.concurrent_requests,
            "batch_size": args.batch_size,
            "batch_timeout": args.batch_timeout,
            "baseline_enabled": not args.no_baseline,
            "leaf_module_path": str(args.leaf_module_path) if args.leaf_module_path else None,
            "merge_module_path": str(args.merge_module_path) if args.merge_module_path else None,
            "scorer_module_path": str(args.scorer_module_path) if args.scorer_module_path else None,
        },
    }

    if not args.dry_run:
        task_ports = args.ports or [args.port]
        task_ports_deduped: list[int] = []
        seen_ports = set()
        for port in task_ports:
            try:
                port_int = int(port)
            except (TypeError, ValueError):
                continue
            if port_int in seen_ports:
                continue
            seen_ports.add(port_int)
            task_ports_deduped.append(port_int)
        if not task_ports_deduped:
            logger.error("No valid ports provided.")
            return 1
        task_ports = task_ports_deduped

        # Drop unreachable ports early to avoid "connection error" spam when load balancing.
        def _port_ready(port: int) -> bool:
            try:
                import urllib.request

                with urllib.request.urlopen(
                    f"http://localhost:{port}/v1/models",
                    timeout=2,
                ) as resp:
                    return int(getattr(resp, "status", 0) or 0) == 200
            except Exception:
                return False

        if len(task_ports) > 1:
            ready_ports = [p for p in task_ports if _port_ready(int(p))]
            if ready_ports:
                if len(ready_ports) != len(task_ports):
                    logger.warning(
                        "Some task ports are unreachable; using reachable subset: %s",
                        ", ".join(str(p) for p in ready_ports),
                    )
                task_ports = ready_ports
            else:
                logger.error(
                    "None of the provided task ports are reachable: %s",
                    ", ".join(str(p) for p in task_ports),
                )
                return 1

        if args.ports:
            args.port = int(task_ports[0])

        task_model_urls = [f"http://localhost:{port}/v1" for port in task_ports]
        config = BatchedPipelineConfig(
            task_model_url=task_model_urls[0],
            task_model_urls=task_model_urls if len(task_model_urls) > 1 else None,
            max_concurrent_requests=args.concurrent_requests,
            max_concurrent_documents=args.concurrent_docs,
            batch_size=args.batch_size,
            batch_timeout=args.batch_timeout,
            max_chunk_chars=args.chunk_size,
            run_baseline=not args.no_baseline,
            show_progress=True,
            rubric=RILE_PRESERVATION_RUBRIC,
            task_context=RILE_TASK_CONTEXT,
            prompt_builders=PromptBuilders(
                summarize=default_summarize_prompt,
                merge=default_merge_prompt,
                score=build_score_prompt,
                audit=None,
            ),
            score_parser=parse_score,
        )
        doc_samples = [to_document_sample(sample) for sample in loaded_samples]
        sample_by_id = {sample.manifesto_id: sample for sample in loaded_samples}
        start = time.time()

        if use_dspy_modules:
            task = get_task("manifesto_rile")
            if len(task_ports) > 1:
                lm = create_vllm_lm_multi(
                    ports=task_ports,
                    temperature=float(args.dspy_temperature),
                    max_tokens=int(args.dspy_max_tokens),
                )
            else:
                lm = create_vllm_lm(
                    port=args.port,
                    temperature=float(args.dspy_temperature),
                    max_tokens=int(args.dspy_max_tokens),
                )
            configure_dspy(lm=lm)

            leaf_module = task.create_summarizer()
            merge_module = task.create_merge_summarizer()
            leaf_module.load(str(args.leaf_module_path))
            merge_module.load(str(args.merge_module_path))

            scorer = task.create_predictor()
            if args.scorer_module_path:
                scorer.load(str(args.scorer_module_path))

            pipeline = BatchedDocPipeline(config)
            strategy = DSPyStrategy(
                leaf_module=leaf_module,
                merge_module=merge_module,
                default_temperature=float(args.dspy_temperature),
                max_tokens=int(args.dspy_max_tokens),
            )
            results = asyncio.run(
                pipeline.process_batch_with_strategy(
                    doc_samples,
                    strategy=strategy,
                    show_progress=True,
                )
            )

            for result in results:
                sample = sample_by_id.get(result.doc_id)
                expert_rile = float(sample.rile) if sample else None
                predicted_norm = _predict_summary_score(scorer, result.final_summary, RILE_TASK_CONTEXT)
                predicted_rile = denormalize_rile(predicted_norm)
                gap = (
                    abs(predicted_rile - expert_rile)
                    if predicted_rile is not None and expert_rile is not None
                    else None
                )
                payload["results"].append(
                    {
                        "manifesto_id": result.doc_id,
                        "expert_rile": expert_rile,
                        "predicted_rile": predicted_rile,
                        "predicted_normalized": predicted_norm,
                        "absolute_gap_rile": gap,
                        "representation_selected_backend": result.metadata.get("representation_selected_backend"),
                        "representation_selected_score": result.metadata.get("representation_selected_score"),
                        "representation_selected_score_raw": result.metadata.get("representation_selected_score_raw"),
                        "representation_backend_scores": result.metadata.get("representation_backend_scores"),
                        "llm_score_failure_reason": result.metadata.get("llm_score_failure_reason"),
                        "llm_score_failure_preview": result.metadata.get("llm_score_failure_preview"),
                        "missing_score_default_applied": result.metadata.get("missing_score_default_applied"),
                        "missing_score_default_reason": result.metadata.get("missing_score_default_reason"),
                        "score_reasoning_preview": str(result.reasoning or "")[:240],
                        "final_summary_preview": str(result.final_summary or "")[:240],
                        "tree_leaves": result.tree_leaves,
                        "tree_height": result.tree_height,
                        "error": result.error,
                    }
                )
        else:
            pipeline = BatchedDocPipeline(config)
            results = pipeline.process_batch(doc_samples)

            for result in results:
                sample = sample_by_id.get(result.doc_id)
                expert_rile = float(sample.rile) if sample else None
                predicted_norm = result.estimated_score
                predicted_rile = denormalize_rile(predicted_norm)
                gap = (
                    abs(predicted_rile - expert_rile)
                    if predicted_rile is not None and expert_rile is not None
                    else None
                )
                payload["results"].append(
                    {
                        "manifesto_id": result.doc_id,
                        "expert_rile": expert_rile,
                        "predicted_rile": predicted_rile,
                        "predicted_normalized": predicted_norm,
                        "absolute_gap_rile": gap,
                        "representation_selected_backend": result.metadata.get("representation_selected_backend"),
                        "representation_selected_score": result.metadata.get("representation_selected_score"),
                        "representation_selected_score_raw": result.metadata.get("representation_selected_score_raw"),
                        "representation_backend_scores": result.metadata.get("representation_backend_scores"),
                        "llm_score_failure_reason": result.metadata.get("llm_score_failure_reason"),
                        "llm_score_failure_preview": result.metadata.get("llm_score_failure_preview"),
                        "missing_score_default_applied": result.metadata.get("missing_score_default_applied"),
                        "missing_score_default_reason": result.metadata.get("missing_score_default_reason"),
                        "score_reasoning_preview": str(result.reasoning or "")[:240],
                        "final_summary_preview": str(result.final_summary or "")[:240],
                        "tree_leaves": result.tree_leaves,
                        "tree_height": result.tree_height,
                        "error": result.error,
                    }
                )

        elapsed = time.time() - start
        logger.info(
            "Batched run complete: %d docs in %.1fs (%.2f docs/sec)",
            len(doc_samples),
            elapsed,
            len(doc_samples) / max(elapsed, 1e-9),
        )

        for row in payload["results"]:
            gap = row.get("absolute_gap_rile")
            if gap is None:
                logger.info(
                    "  %s | prediction unavailable | error=%s",
                    row.get("manifesto_id"),
                    row.get("error"),
                )
            else:
                logger.info(
                    "  %s | pred=%+.1f | expert=%+.1f | gap=%.1f | leaves=%s",
                    row.get("manifesto_id"),
                    row.get("predicted_rile"),
                    row.get("expert_rile"),
                    gap,
                    row.get("tree_leaves"),
                )

    if args.dry_run and args.output is None:
        return 0

    output_path = args.output
    if output_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("data/results/manifesto_rile") / f"batched_example_{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved output: %s", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
