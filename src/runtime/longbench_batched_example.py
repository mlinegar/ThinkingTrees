"""
Run LongBench v2 through the batched document/tree pipeline.

This is a throughput-oriented companion to scripts/run_runtime_eval.py:
LongBench rows are treated as document samples, tree summarization builds
question-aware evidence memories, and the final A-D answer calls are batched
through BatchedDocPipeline / BatchTreeOrchestrator.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.config.local_inference import add_local_inference_args, resolve_local_inference_config
from src.core.documents import DocumentResult, DocumentSample
from src.core.engram_prompting import engram_document_metadata
from src.core.strategy import tournament_doc_id
from src.experiments.contracts import MethodRef, ResultRow
from src.experiments.sidecars import (
    sidecar_root_for_output_file,
    simple_benchmark_ref,
    write_canonical_sidecars,
)
from src.preprocessing.chunker import chunk_for_ops
from src.runtime.adapters.longbench import LongBenchV2Adapter, LongBenchV2Spec
from src.runtime.answering import parse_multi_choice_text, render_choices
from src.runtime.results import runtime_method_ref
from src.tasks.prompting import PromptBuilders

logger = logging.getLogger(__name__)

CHOICE_LETTERS = ("A", "B", "C", "D")
CHOICE_TO_SCORE = {letter: float(idx) for idx, letter in enumerate(CHOICE_LETTERS)}
SCORE_TO_CHOICE = {idx: letter for idx, letter in enumerate(CHOICE_LETTERS)}
LONG_BENCH_TASK_CONTEXT = (
    "Answer LongBench v2 multiple-choice questions from long context evidence. "
    "The only valid final answers are A, B, C, or D."
)

# The batched tree builder exposes doc identity via tournament_doc_id while
# building leaves/merges. Keep LongBench question metadata reachable there.
_PROMPT_METADATA_BY_DOC_ID: Dict[str, Dict[str, Any]] = {}


def _parse_csv(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)


def _current_prompt_metadata() -> Dict[str, Any]:
    meta = engram_document_metadata.get(None)
    if isinstance(meta, dict) and (meta.get("question") or meta.get("choices")):
        return dict(meta)

    doc_id = str(tournament_doc_id.get() or "").strip()
    if doc_id:
        cached = _PROMPT_METADATA_BY_DOC_ID.get(doc_id)
        if isinstance(cached, dict):
            return dict(cached)
    return {}


def _choice_block(choices: Mapping[str, Any]) -> str:
    normalized = {
        str(letter).strip().upper()[:1]: str(value)
        for letter, value in dict(choices or {}).items()
        if str(letter).strip()
    }
    return render_choices(
        {letter: normalized[letter] for letter in CHOICE_LETTERS if letter in normalized}
    )


def _question_block(meta: Mapping[str, Any]) -> str:
    question = str(meta.get("question") or "").strip()
    choices = _choice_block(meta.get("choices") or {})
    parts = []
    if question:
        parts.append(f"Question:\n{question}")
    if choices:
        parts.append(f"Choices:\n{choices}")
    return "\n\n".join(parts).strip()


def build_longbench_summarize_prompt(text: str, rubric: str) -> List[Dict[str, str]]:
    meta = _current_prompt_metadata()
    question = _question_block(meta)
    user = (
        f"{question}\n\n"
        f"Context chunk:\n{text}\n\n"
        "Write a compact evidence memory for answering the question. Preserve named entities, "
        "numbers, conditions, contradictions, and option-specific evidence. Do not answer yet."
    ).strip()
    return [
        {
            "role": "system",
            "content": (
                "You compress long-context evidence for a multiple-choice scorer. "
                "Keep only information that can help distinguish A, B, C, and D."
            ),
        },
        {"role": "user", "content": user},
    ]


def build_longbench_merge_prompt(left: str, right: str, rubric: str) -> List[Dict[str, str]]:
    meta = _current_prompt_metadata()
    question = _question_block(meta)
    user = (
        f"{question}\n\n"
        f"Evidence memory 1:\n{left}\n\n"
        f"Evidence memory 2:\n{right}\n\n"
        "Merge these into one compact evidence memory. Remove duplication, keep disagreements, "
        "and preserve evidence tied to the answer choices. Do not answer yet."
    ).strip()
    return [
        {
            "role": "system",
            "content": "You merge evidence memories for a long-context multiple-choice task.",
        },
        {"role": "user", "content": user},
    ]


def build_longbench_score_prompt(evidence: str, task_context: str) -> List[Dict[str, str]]:
    meta = _current_prompt_metadata()
    question = _question_block(meta)
    user = (
        f"{question}\n\n"
        f"Evidence/context:\n{evidence}\n\n"
        "Choose the single best option. Return only one letter: A, B, C, or D."
    ).strip()
    return [
        {
            "role": "system",
            "content": (
                "You answer LongBench v2 multiple-choice questions. "
                "Use the supplied evidence only. Output exactly one of A, B, C, or D."
            ),
        },
        {"role": "user", "content": user},
    ]


def parse_choice_score(response: str) -> Optional[float]:
    letter = parse_multi_choice_text(response, valid_choices=CHOICE_LETTERS)
    return CHOICE_TO_SCORE.get(letter)


def score_to_choice(score: Any) -> str:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return ""
    idx = int(round(value))
    if abs(value - float(idx)) > 0.35:
        return ""
    return SCORE_TO_CHOICE.get(idx, "")


def _build_adapter(args: argparse.Namespace) -> LongBenchV2Adapter:
    return LongBenchV2Adapter(
        spec=LongBenchV2Spec(
            task_id=str(args.task_id),
            split=str(args.split),
            max_seq_length=int(args.max_seq_length),
            num_samples=int(args.limit or 0),
            seed=int(args.seed),
        ),
        dataset_path=args.dataset_path,
        hf_dataset=str(args.hf_dataset),
        hf_config=args.hf_config,
        streaming=bool(args.streaming),
        domains=_parse_csv(args.domains),
        sub_domains=_parse_csv(args.sub_domains),
        difficulties=_parse_csv(args.difficulties),
        length_buckets=_parse_csv(args.lengths),
    )


def _problem_to_sample(adapter: LongBenchV2Adapter, problem: Any) -> DocumentSample:
    view = adapter.task_view(problem)
    gold = str((problem.references or [""])[0] or "").strip().upper()[:1]
    doc_id = str(problem.metadata.get("_id") or problem.problem_id).strip()
    if not doc_id:
        doc_id = str(problem.problem_id)
    metadata = dict(problem.metadata or {})
    metadata.update(
        {
            "problem_id": problem.problem_id,
            "_id": metadata.get("_id") or doc_id,
            "question": view.question,
            "choices": dict(view.choices),
            "answer": gold,
            "benchmark": "longbench_v2",
        }
    )
    _PROMPT_METADATA_BY_DOC_ID[doc_id] = dict(metadata)
    return DocumentSample(
        doc_id=doc_id,
        text=view.context,
        reference_score=CHOICE_TO_SCORE.get(gold),
        metadata=metadata,
    )


def load_longbench_samples(
    adapter: LongBenchV2Adapter,
    *,
    split: str,
    limit: Optional[int],
) -> List[DocumentSample]:
    _PROMPT_METADATA_BY_DOC_ID.clear()
    problems = list(adapter.load_split(split, limit=limit))
    return [_problem_to_sample(adapter, problem) for problem in problems]


def analyze_chunks(
    samples: Sequence[DocumentSample],
    *,
    chunk_size: int,
    chunk_tokens: Optional[int],
    chunk_token_encoding: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        chunks = chunk_for_ops(
            sample.text,
            max_chars=int(chunk_size),
            max_tokens=chunk_tokens,
            token_encoding=chunk_token_encoding,
        )
        meta = sample.metadata
        rows.append(
            {
                "_id": meta.get("_id") or sample.doc_id,
                "domain": meta.get("domain"),
                "sub_domain": meta.get("sub_domain"),
                "difficulty": meta.get("difficulty"),
                "length": meta.get("length"),
                "chars": len(sample.text or ""),
                "chunk_size": int(chunk_size),
                "chunk_tokens": chunk_tokens,
                "chunk_count": len(chunks),
                "answer": meta.get("answer"),
            }
        )
    return rows


def _result_payload_row(
    result: DocumentResult,
    *,
    summary_preview_chars: int,
) -> Dict[str, Any]:
    meta = dict(result.metadata or {})
    gold = str(meta.get("answer") or "").strip().upper()[:1]
    prediction = score_to_choice(result.estimated_score)
    baseline_prediction = score_to_choice(result.baseline_score)
    row = {
        "_id": meta.get("_id") or result.doc_id,
        "problem_id": meta.get("problem_id"),
        "domain": meta.get("domain"),
        "sub_domain": meta.get("sub_domain"),
        "difficulty": meta.get("difficulty"),
        "length": meta.get("length"),
        "answer": gold,
        "prediction": prediction,
        "correct": bool(prediction and prediction == gold),
        "baseline_prediction": baseline_prediction,
        "baseline_correct": (
            None if not baseline_prediction or not gold else bool(baseline_prediction == gold)
        ),
        "estimated_score": result.estimated_score,
        "baseline_score": result.baseline_score,
        "selected_program_family": meta.get("selected_program_family"),
        "selected_program_score": meta.get("selected_program_score"),
        "program_family_scores": meta.get("program_family_scores"),
        "program_fallback_used": meta.get("program_fallback_used"),
        "llm_score_failure_reason": meta.get("llm_score_failure_reason"),
        "llm_score_skipped_reason": meta.get("llm_score_skipped_reason"),
        "tree_height": result.tree_height,
        "tree_nodes": result.tree_nodes,
        "tree_leaves": result.tree_leaves,
        "summary_length": result.summary_length,
        "compression_ratio": result.compression_ratio,
        "processing_time": result.processing_time,
        "error": result.error,
    }
    if summary_preview_chars > 0:
        row["final_summary_preview"] = (result.final_summary or "")[:summary_preview_chars]
    return row


def _group_accuracy(rows: Iterable[Mapping[str, Any]], key: str) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    correct: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
        correct[value] = correct.get(value, 0) + (1 if row.get("correct") else 0)
    return {
        value: (float(correct.get(value, 0)) / float(count) if count else 0.0)
        for value, count in sorted(counts.items())
    }


def aggregate_results(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    baseline_rows = [row for row in rows if row.get("baseline_correct") is not None]
    baseline_correct = sum(1 for row in baseline_rows if row.get("baseline_correct"))
    return {
        "n_predictions": total,
        "accuracy": (float(correct) / float(total) if total else 0.0),
        "baseline_n_predictions": len(baseline_rows),
        "baseline_accuracy": (
            float(baseline_correct) / float(len(baseline_rows)) if baseline_rows else None
        ),
        "by_domain": _group_accuracy(rows, "domain"),
        "by_difficulty": _group_accuracy(rows, "difficulty"),
        "by_length": _group_accuracy(rows, "length"),
    }


def _stats_payload(pipeline: Any) -> Dict[str, Any]:
    stats = pipeline.last_stats
    payload: Dict[str, Any] = {}
    if stats is not None:
        payload["batch_stats"] = {
            **asdict(stats),
            "wall_clock_seconds": stats.wall_clock_seconds,
            "tokens_per_second": stats.tokens_per_second,
            "read_tokens_per_second": stats.read_tokens_per_second,
            "write_tokens_per_second": stats.write_tokens_per_second,
        }
    if pipeline.last_diagnostics:
        payload["diagnostics"] = pipeline.last_diagnostics
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LongBench v2 as a batched document/tree workload.",
    )
    parser.add_argument(
        "--dataset-path", type=Path, default=None, help="Local JSON/JSONL LongBench rows."
    )
    parser.add_argument("--hf-dataset", default="THUDM/LongBench-v2")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--task-id", default="all")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--domains", default=None, help="Comma-separated domain filter.")
    parser.add_argument("--sub-domains", default=None, help="Comma-separated sub-domain filter.")
    parser.add_argument("--difficulties", default=None, help="Comma-separated difficulty filter.")
    parser.add_argument("--lengths", default=None, help="Comma-separated length bucket filter.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--dry-run", action="store_true", help="Only load/analyze rows; do not call an LLM."
    )

    parser.add_argument("--concurrent-docs", type=int, default=20)
    add_local_inference_args(
        parser,
        include_generation=False,
        default_concurrent_requests=128,
        default_batch_size=32,
        default_batch_timeout=0.02,
        default_request_timeout_seconds=300.0,
        default_await_response_timeout_seconds=600.0,
    )
    parser.add_argument("--chunk-size", type=int, default=8000)
    parser.add_argument("--chunk-tokens", type=int, default=None)
    parser.add_argument("--chunk-token-encoding", default="cl100k_base")
    parser.add_argument("--max-tokens-summary", type=int, default=384)
    parser.add_argument("--max-tokens-score", type=int, default=8)
    parser.add_argument("--runtime-mode", default="unified_v2")
    parser.add_argument(
        "--program-families", default="llm", help="Comma-separated program families."
    )
    parser.add_argument("--primary-program-family", default="llm")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--summary-preview-chars", type=int, default=500)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    adapter = _build_adapter(args)
    samples = load_longbench_samples(adapter, split=args.split, limit=args.limit)
    if not samples:
        logger.error("No LongBench rows matched the requested filters.")
        return 1

    analysis_rows = analyze_chunks(
        samples,
        chunk_size=args.chunk_size,
        chunk_tokens=args.chunk_tokens,
        chunk_token_encoding=args.chunk_token_encoding,
    )
    logger.info("Loaded %d LongBench rows for batched processing.", len(samples))
    for row in analysis_rows[:10]:
        logger.info(
            "  %s | domain=%s difficulty=%s length=%s chars=%d chunks=%d answer=%s",
            row["_id"],
            row.get("domain"),
            row.get("difficulty"),
            row.get("length"),
            row["chars"],
            row["chunk_count"],
            row.get("answer"),
        )

    output_path = args.output
    if output_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("outputs") / f"longbench_batched_{stamp}.json"

    payload: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "mode": "dry_run" if args.dry_run else "batched_longbench_tree",
        "method_ref": runtime_method_ref(
            method_id="longbench_batched_tree",
            runner_id="batched_doc_pipeline",
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
        "dataset": {
            "dataset_path": str(args.dataset_path) if args.dataset_path else None,
            "hf_dataset": args.hf_dataset,
            "hf_config": args.hf_config,
            "split": args.split,
            "task_id": args.task_id,
            "limit": args.limit,
        },
        "config": {
            "engine": args.engine,
            "host": args.host,
            "model": args.model,
            "port": args.port,
            "ports": args.ports,
            "concurrent_docs": args.concurrent_docs,
            "concurrent_requests": args.concurrent_requests,
            "batch_size": args.batch_size,
            "batch_timeout": args.batch_timeout,
            "request_timeout_seconds": args.request_timeout_seconds,
            "await_response_timeout_seconds": args.await_response_timeout_seconds,
            "chunk_size": args.chunk_size,
            "chunk_tokens": args.chunk_tokens,
            "runtime_mode": args.runtime_mode,
            "program_families": _parse_csv(args.program_families),
            "primary_program_family": args.primary_program_family,
            "baseline_enabled": not args.no_baseline,
        },
        "chunk_analysis": analysis_rows,
        "results": [],
        "aggregate": None,
        "runtime": {},
    }
    sidecar_root = sidecar_root_for_output_file(output_path)
    call_trace_path = sidecar_root / "calls.jsonl"

    if not args.dry_run:
        try:
            local_inference = resolve_local_inference_config(
                args,
                usage="LongBench batched task model",
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
            max_chunk_chars=int(args.chunk_size),
            max_chunk_tokens=args.chunk_tokens,
            chunk_token_encoding=str(args.chunk_token_encoding),
            max_tokens_summary=int(args.max_tokens_summary),
            max_tokens_score=int(args.max_tokens_score),
            run_baseline=not args.no_baseline,
            runtime_mode=str(args.runtime_mode),
            show_progress=True,
            rubric=LONG_BENCH_TASK_CONTEXT,
            task_context=LONG_BENCH_TASK_CONTEXT,
            prompt_builders=PromptBuilders(
                summarize=build_longbench_summarize_prompt,
                merge=build_longbench_merge_prompt,
                score=build_longbench_score_prompt,
                audit=None,
            ),
            score_parser=parse_choice_score,
            program_families=_parse_csv(args.program_families),
            primary_program_family=str(args.primary_program_family),
            missing_score_default=None,
            call_trace_sink=JsonlCallTraceSink(
                call_trace_path,
                defaults={
                    "method_id": "longbench_batched_tree",
                    "runner_id": "batched_doc_pipeline",
                    "surface": "chat_openai",
                },
            ),
        )

        start = time.time()
        pipeline = BatchedDocPipeline(config)
        results = pipeline.process_batch(samples)
        elapsed = time.time() - start
        rows = [
            _result_payload_row(
                result,
                summary_preview_chars=int(args.summary_preview_chars),
            )
            for result in results
        ]
        payload["results"] = rows
        payload["aggregate"] = aggregate_results(rows)
        payload["runtime"] = {
            "elapsed_seconds": elapsed,
            "examples_per_second": len(samples) / max(elapsed, 1e-9),
            **_stats_payload(pipeline),
        }
        logger.info(
            "Batched LongBench run complete: %d rows in %.1fs (%.2f rows/sec), accuracy=%.3f",
            len(samples),
            elapsed,
            len(samples) / max(elapsed, 1e-9),
            payload["aggregate"]["accuracy"],
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    method_ref = MethodRef.from_dict(payload["method_ref"])
    benchmark_ref = simple_benchmark_ref(
        family="longbench_v2",
        name=str(args.task_id),
        dataset_id=str(args.dataset_path or args.hf_dataset),
        metadata=dict(payload.get("dataset") or {}),
    )
    aggregate = dict(payload.get("aggregate") or {})
    result_rows = []
    if aggregate:
        result_rows.append(
            ResultRow(
                experiment_id="",
                phase=str(payload.get("mode") or ""),
                benchmark_ref=benchmark_ref,
                method_ref=method_ref,
                split=str(args.split),
                metric_name="accuracy",
                metric_value=aggregate.get("accuracy"),
                artifact_refs=("output_json", "calls_jsonl") if call_trace_path.exists() else ("output_json",),
                metadata={"aggregate": aggregate},
            )
        )
    artifacts = {"output_json": str(output_path)}
    if call_trace_path.exists():
        artifacts["calls_jsonl"] = str(call_trace_path)
    write_canonical_sidecars(
        sidecar_root,
        title="longbench_batched_example",
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
    logger.info("Wrote %s", output_path)
    return 0

