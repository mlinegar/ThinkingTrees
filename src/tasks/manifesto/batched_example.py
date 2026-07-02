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
from typing import Any, Optional, Sequence

from src.config.local_inference import add_local_inference_args, resolve_local_inference_config
from src.config.settings import load_settings
from src.core.documents import DocumentSample
from src.experiments.contracts import MethodRef, ResultRow
from src.experiments.sidecars import (
    sidecar_root_for_output_file,
    simple_benchmark_ref,
    write_canonical_sidecars,
)
from src.preprocessing.chunker import chunk_for_ops
from src.runtime.results import runtime_method_ref
from src.tasks.manifesto import RILE_SCALE, ManifestoDataset
from src.tasks.manifesto.rubrics import RILE_PRESERVATION_RUBRIC, RILE_TASK_CONTEXT
from src.tasks.prompting import (
    PromptBuilders,
    default_merge_prompt,
    default_summarize_prompt,
    parse_numeric_score,
)

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


def _result_payload_row(
    *,
    result: Any,
    expert_rile: Optional[float],
    predicted_norm: Optional[float],
) -> dict[str, Any]:
    predicted_rile = denormalize_rile(predicted_norm)
    gap = (
        abs(predicted_rile - expert_rile)
        if predicted_rile is not None and expert_rile is not None
        else None
    )
    return {
        "manifesto_id": result.doc_id,
        "expert_rile": expert_rile,
        "predicted_rile": predicted_rile,
        "predicted_normalized": predicted_norm,
        "absolute_gap_rile": gap,
        "selected_program_family": result.metadata.get("selected_program_family"),
        "selected_program_score": result.metadata.get("selected_program_score"),
        "selected_program_score_raw": result.metadata.get("selected_program_score_raw"),
        "program_family_scores": result.metadata.get("program_family_scores"),
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


def resolve_manifesto_module_paths(
    *,
    g_module_path: Optional[Path],
    scorer_module_path: Optional[Path],
    use_published_modules: bool,
    published_root_dir: Path,
) -> tuple[Optional[Path], Optional[Path]]:
    """Resolve canonical optimized module artifacts for the active unified-g path."""
    if scorer_module_path is not None and not scorer_module_path.exists():
        raise ValueError(f"Scorer module does not exist: {scorer_module_path}")

    if g_module_path is not None:
        if not g_module_path.exists():
            raise ValueError(f"Unified-g module does not exist: {g_module_path}")
        return g_module_path, scorer_module_path

    if not use_published_modules:
        return None, scorer_module_path

    module_dir = published_root_dir / "trained_modules"
    unified_candidate = module_dir / "unified_g_final.json"
    scorer_candidate = module_dir / "scorer_final.json"
    if unified_candidate.exists():
        resolved_scorer = scorer_module_path
        if resolved_scorer is None and scorer_candidate.exists():
            resolved_scorer = scorer_candidate
        logger.info("Using published optimized unified-g module from %s", module_dir)
        return unified_candidate, resolved_scorer

    legacy_leaf = module_dir / "leaf_summarizer_final.json"
    legacy_merge = module_dir / "merge_summarizer_final.json"
    if legacy_leaf.exists() or legacy_merge.exists():
        raise ValueError(
            "Published legacy split leaf/merge modules were found, but the active "
            f"batched path requires {unified_candidate}. Re-run optimization to "
            "produce unified_g_final.json or pass --g-module-path explicitly."
        )

    return None, scorer_module_path


def create_parser() -> argparse.ArgumentParser:
    settings = load_settings()
    summarizer_cfg = (
        (settings.get("generation", {}) or {}).get("summarizer", {})
        if isinstance(settings, dict)
        else {}
    )
    default_temperature = float(summarizer_cfg.get("temperature", 0.5))
    default_max_tokens = int(summarizer_cfg.get("max_tokens", 4096))

    parser = argparse.ArgumentParser(
        description="Run manifesto IDs through the batched OPS RILE pipeline.",
    )
    parser.add_argument("--ids", nargs="+", default=DEFAULT_IDS, help="Manifesto IDs to run")
    parser.add_argument("--chunk-size", type=int, default=8000, help="Chunk size in characters")
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=None,
        help="Optional token budget per leaf; takes precedence over --chunk-size.",
    )
    add_local_inference_args(
        parser,
        include_generation=True,
        default_concurrent_requests=200,
        default_batch_size=50,
        default_batch_timeout=0.02,
        default_temperature=default_temperature,
        default_max_tokens=default_max_tokens,
    )
    parser.add_argument("--concurrent-docs", type=int, default=8, help="Concurrent documents")
    parser.add_argument("--no-baseline", action="store_true", help="Disable baseline scoring")
    parser.add_argument(
        "--countries", type=int, nargs="+", default=[51], help="Country filter for dataset load"
    )
    parser.add_argument(
        "--min-year", type=int, default=1900, help="Min year filter for dataset load"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only report chunk stats; skip model calls"
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--g-module-path", type=Path, default=None, help="Path to optimized unified_g_final.json"
    )
    parser.add_argument(
        "--scorer-module-path",
        type=Path,
        default=None,
        help="Optional path to optimized scorer module JSON",
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

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
        chunks = chunk_for_ops(
            sample.text,
            max_chars=args.chunk_size,
            max_tokens=args.chunk_tokens,
            strategy="axis",
        )
        analysis_rows.append(
            {
                "manifesto_id": sample.manifesto_id,
                "party_name": sample.party_name,
                "year": sample.year,
                "chars": len(sample.text),
                "chunk_size": args.chunk_size,
                "chunk_tokens": args.chunk_tokens,
                "chunk_count": len(chunks),
                "expert_rile": sample.rile,
            }
        )

    if args.chunk_tokens:
        logger.info(
            "Chunk analysis (strategy=token_budget, max_tokens=%d, fallback_max_chars=%d):",
            args.chunk_tokens,
            args.chunk_size,
        )
    else:
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

    if args.dry_run and args.g_module_path is None:
        g_module_path = None
        scorer_module_path = args.scorer_module_path
    else:
        try:
            g_module_path, scorer_module_path = resolve_manifesto_module_paths(
                g_module_path=args.g_module_path,
                scorer_module_path=args.scorer_module_path,
                use_published_modules=bool(args.use_published_modules),
                published_root_dir=args.published_root_dir,
            )
        except ValueError as exc:
            logger.error("%s", exc)
            return 1

    use_dspy_modules = g_module_path is not None

    payload = {
        "timestamp": datetime.now().isoformat(),
        "mode": (
            "dry_run"
            if args.dry_run
            else ("batched_dspy_modules" if use_dspy_modules else "batched")
        ),
        "method_ref": runtime_method_ref(
            method_id="manifesto_batched_dspy" if use_dspy_modules else "manifesto_batched_tree",
            runner_id="batched_doc_pipeline_dspy" if use_dspy_modules else "batched_doc_pipeline",
            roles={
                "scorer": {
                    "surface": "chat_openai",
                    "engine": args.engine,
                    "model": args.model,
                    "base_url": f"http://{args.host}:{args.port}/v1",
                }
            },
            surfaces={
                "chat_openai": {
                    "engine": args.engine,
                    "model": args.model,
                    "base_url": f"http://{args.host}:{args.port}/v1",
                }
            },
            adapter="batched_doc_pipeline",
        ),
        "chunk_analysis": analysis_rows,
        "results": [],
        "missing_ids": missing_ids,
        "config": {
            "engine": args.engine,
            "host": args.host,
            "model": args.model,
            "port": args.port,
            "ports": args.ports,
            "chunk_size": args.chunk_size,
            "chunk_tokens": args.chunk_tokens,
            "concurrent_docs": args.concurrent_docs,
            "concurrent_requests": args.concurrent_requests,
            "batch_size": args.batch_size,
            "batch_timeout": args.batch_timeout,
            "request_timeout_seconds": args.request_timeout_seconds,
            "await_response_timeout_seconds": args.await_response_timeout_seconds,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "baseline_enabled": not args.no_baseline,
            "g_module_path": str(g_module_path) if g_module_path else None,
            "scorer_module_path": str(scorer_module_path) if scorer_module_path else None,
        },
    }
    output_path = args.output
    if output_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("data/results/manifesto_rile") / f"batched_example_{stamp}.json"
    sidecar_root = sidecar_root_for_output_file(output_path)
    call_trace_path = sidecar_root / "calls.jsonl"

    if not args.dry_run:
        settings = load_settings()
        try:
            local_inference = resolve_local_inference_config(
                args,
                settings=settings,
                usage="manifesto batched task model",
                filter_unreachable=True,
            )
        except (RuntimeError, ValueError) as exc:
            logger.error("%s", exc)
            return 1

        payload["config"].update(local_inference.to_dict())

        from src.experiments.call_tracing import JsonlCallTraceSink
        from src.pipelines.batched import BatchedDocPipeline, BatchedPipelineConfig

        config = BatchedPipelineConfig(
            **local_inference.pipeline_kwargs(max_concurrent_documents=args.concurrent_docs),
            max_chunk_chars=args.chunk_size,
            max_chunk_tokens=args.chunk_tokens,
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
            call_trace_sink=JsonlCallTraceSink(
                call_trace_path,
                defaults={
                    "method_id": "manifesto_batched_dspy" if use_dspy_modules else "manifesto_batched_tree",
                    "runner_id": "batched_doc_pipeline_dspy" if use_dspy_modules else "batched_doc_pipeline",
                    "surface": "chat_openai",
                },
            ),
        )
        doc_samples = [to_document_sample(sample) for sample in loaded_samples]
        sample_by_id = {sample.manifesto_id: sample for sample in loaded_samples}
        start = time.time()

        if use_dspy_modules:
            from src.config.dspy_config import configure_dspy, create_local_engine_lm
            from src.core.strategy import DSPyStrategy
            from src.tasks import get_task

            task = get_task("manifesto_rile")
            lm = create_local_engine_lm(
                **local_inference.dspy_kwargs(settings=settings),
            )
            logger.info(
                "DSPy optimized modules are using batched %s transport "
                "(max_concurrent=%d batch_size=%d batch_timeout=%.3fs routing=%s)",
                local_inference.engine,
                int(local_inference.max_concurrent_requests),
                int(local_inference.batch_size),
                float(local_inference.batch_timeout),
                local_inference.routing_policy,
            )
            configure_dspy(lm=lm)

            g_module = task.create_summarizer()
            g_module.load(str(g_module_path))

            scorer = task.create_predictor()
            if scorer_module_path:
                try:
                    scorer.load(str(scorer_module_path))
                except Exception as exc:
                    logger.warning(
                        "Could not load scorer module %s; falling back to runtime predictor. error=%s",
                        scorer_module_path,
                        exc,
                    )

            pipeline = BatchedDocPipeline(config)
            strategy_temperature = float(
                local_inference.temperature if local_inference.temperature is not None else 0.5
            )
            strategy_max_tokens = int(
                local_inference.max_tokens if local_inference.max_tokens is not None else 4096
            )
            strategy = DSPyStrategy(
                leaf_module=g_module,
                merge_module=None,
                unified_mode=True,
                default_temperature=strategy_temperature,
                max_tokens=strategy_max_tokens,
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
                predicted_norm = _predict_summary_score(
                    scorer, result.final_summary, RILE_TASK_CONTEXT
                )
                payload["results"].append(
                    _result_payload_row(
                        result=result,
                        expert_rile=expert_rile,
                        predicted_norm=predicted_norm,
                    )
                )
        else:
            pipeline = BatchedDocPipeline(config)
            results = pipeline.process_batch(doc_samples)

            for result in results:
                sample = sample_by_id.get(result.doc_id)
                expert_rile = float(sample.rile) if sample else None
                predicted_norm = result.estimated_score
                payload["results"].append(
                    _result_payload_row(
                        result=result,
                        expert_rile=expert_rile,
                        predicted_norm=predicted_norm,
                    )
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    method_ref = MethodRef.from_dict(payload["method_ref"])
    benchmark_ref = simple_benchmark_ref(
        family="manifesto_rile",
        name="manifesto_batched_example",
        dataset_id="manifesto",
        metadata={"ids": list(args.ids), "missing_ids": list(missing_ids)},
    )
    result_rows = []
    scored_rows = [
        row for row in list(payload.get("results") or [])
        if row.get("absolute_gap_rile") is not None
    ]
    if scored_rows:
        mean_abs_gap = sum(float(row["absolute_gap_rile"]) for row in scored_rows) / len(scored_rows)
        result_rows.append(
            ResultRow(
                experiment_id="",
                phase=str(payload.get("mode") or ""),
                benchmark_ref=benchmark_ref,
                method_ref=method_ref,
                split="example",
                metric_name="mean_absolute_gap_rile",
                metric_value=mean_abs_gap,
                artifact_refs=("output_json", "calls_jsonl") if call_trace_path.exists() else ("output_json",),
                metadata={"n_scored": len(scored_rows), "n_requested": len(args.ids)},
            )
        )
    artifacts = {"output_json": str(output_path)}
    if call_trace_path.exists():
        artifacts["calls_jsonl"] = str(call_trace_path)
    write_canonical_sidecars(
        sidecar_root,
        title="manifesto_batched_example",
        adapter_id="batched_doc_pipeline",
        benchmark_refs=(benchmark_ref,),
        method_refs=(method_ref,),
        phases=("dry_run" if args.dry_run else "run",),
        artifacts=artifacts,
        result_rows=result_rows,
        state="dry_run" if args.dry_run else "completed",
        metadata={"source_output": str(output_path), "mode": str(payload.get("mode") or "")},
        launch_command=sys.argv,
        report_profiles=("runtime_eval_summary",),
    )
    logger.info("Saved output: %s", output_path)

    return 0

