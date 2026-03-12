#!/usr/bin/env python3
"""Prompt-optimize a unified summarizer g on LawStress using an embedding proxy."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Add project root for direct script execution.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import dspy

from src.config.dspy_config import configure_dspy, create_vllm_lm
from src.tasks.manifesto.lawstress_eval import LawStressEvalConfig, RILE_RUBRIC
from src.tasks.manifesto.lawstress_generator import load_lawstress_records_jsonl
from src.tasks.manifesto.lawstress_proxy import (
    build_proxy_training_examples,
    evaluate_embedding_proxy,
    load_embedding_proxy,
)
from src.tasks.manifesto.lawstress_bootstrap_metric import (
    LawStressBootstrapObjectiveConfig,
    create_lawstress_bootstrap_metric,
)
from src.tasks.manifesto.lawstress_bootstrap_program import LawStressLocalLawProgram, UnifiedG
from src.training.embedding_proxy import VLLMEmbeddingClient, fit_embedding_ridge_proxy


LOGGER = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap unified summarizer g on LawStress.")
    parser.add_argument("--records", type=Path, required=True, help="Path to lawstress_records.jsonl")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default timestamped path)")

    # Student (summarizer) server.
    parser.add_argument("--student-port", type=int, default=8000)
    parser.add_argument("--student-model", type=str, default=None, help="Served model id (default: auto-detect)")
    parser.add_argument("--student-temperature", type=float, default=0.2)
    parser.add_argument(
        "--student-max-tokens",
        type=int,
        default=0,
        help="Max tokens for student generations (<=0 uses model/context-window default).",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable model thinking/reasoning traces for DSPy calls (default: disabled).",
    )
    parser.add_argument(
        "--gepa-reflection-model",
        type=str,
        default=None,
        help="Optional model id for GEPA reflection (default: same as student model).",
    )
    parser.add_argument(
        "--gepa-reflection-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for GEPA reflection generations.",
    )
    parser.add_argument(
        "--gepa-reflection-max-tokens",
        type=int,
        default=0,
        help="Max tokens for GEPA reflection generations (<=0 uses model/context-window default).",
    )

    # Embedding proxy server.
    parser.add_argument("--embedding-url", type=str, default="http://localhost:8003/v1")
    parser.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--embedding-api-key", type=str, default="EMPTY")
    parser.add_argument("--embedding-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--embedding-batch-size", type=int, default=32)

    # Proxy training.
    parser.add_argument("--proxy-path", type=Path, default=None, help="Optional existing embedding_proxy.json")
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--proxy-model-id", type=str, default="lawstress_embedding_ridge_proxy_v1")

    # GEPA.
    parser.add_argument("--gepa-budget", type=str, default="light", choices=["light", "medium", "heavy"])
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument(
        "--gepa-max-metric-calls",
        type=int,
        default=0,
        help="Optional hard cap on GEPA metric calls (<=0 disables cap).",
    )
    parser.add_argument(
        "--gepa-max-full-evals",
        type=int,
        default=0,
        help="Optional hard cap on GEPA full evaluations (<=0 disables cap).",
    )
    parser.add_argument("--seed", type=int, default=0)

    # Metric thresholds.
    parser.add_argument("--c1-threshold-norm", type=float, default=0.10)
    parser.add_argument("--c2-threshold-norm", type=float, default=0.06)
    parser.add_argument("--c3-threshold-norm", type=float, default=0.08)
    parser.add_argument(
        "--objective-aggregate",
        type=str,
        default="min",
        choices=["weighted_mean", "min", "bottleneck_min", "softmin", "floor_then_weighted"],
        help=(
            "How to aggregate multi-signal local-law components into one GEPA scalar. "
            "`min`/`bottleneck_min` optimize the weakest component."
        ),
    )
    parser.add_argument(
        "--objective-softmin-temperature",
        type=float,
        default=0.08,
        help="Temperature for softmin aggregation (used when --objective-aggregate=softmin).",
    )
    parser.add_argument(
        "--objective-component-floor",
        type=float,
        default=0.55,
        help="Component floor for floor_then_weighted mode.",
    )

    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _save_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _to_dspy_examples(records) -> List[dspy.Example]:
    examples: List[dspy.Example] = []
    for record in records:
        examples.append(
            dspy.Example(
                text=record.text,
                segment_a=record.segment_a,
                segment_b=record.segment_b,
                law_target=record.law_target,
                rubric=RILE_RUBRIC,
                # Labels / metadata used by the bootstrap metric.
                y_doc_norm=float(record.teacher_score_doc),
                teacher_score_segment_a_raw=float(record.teacher_score_segment_a),
                teacher_score_segment_b_raw=float(record.teacher_score_segment_b),
                split=record.split,
                difficulty=record.difficulty,
                family=record.family,
                bin_name=record.bin_name,
            ).with_inputs("text", "segment_a", "segment_b", "law_target", "rubric")
        )
    return examples


def _attach_normalized_targets(dspy_examples: Sequence[dspy.Example]) -> None:
    """Mutate examples to store y_doc_norm in [0,1] rather than raw."""
    from src.tasks.manifesto.lawstress_generator import normalize_rile

    for ex in dspy_examples:
        raw = getattr(ex, "y_doc_norm", 0.0)
        setattr(ex, "y_doc_norm", float(normalize_rile(float(raw))))


def _mean_metric(metric_fn: Any, program: Any, examples: Sequence[dspy.Example]) -> Dict[str, Any]:
    if not examples:
        return {"n": 0, "mean_score": None}
    scores: List[float] = []
    for ex in examples:
        pred = program(
            text=getattr(ex, "text"),
            segment_a=getattr(ex, "segment_a"),
            segment_b=getattr(ex, "segment_b"),
            law_target=getattr(ex, "law_target"),
            rubric=getattr(ex, "rubric"),
        )
        out = metric_fn(ex, pred, None, None, None)
        value = out.get("score") if isinstance(out, dict) else out
        try:
            scores.append(float(value))
        except Exception:
            continue
    return {"n": len(scores), "mean_score": (sum(scores) / len(scores)) if scores else None}


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / "lawstress_bootstrap" / f"run_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure DSPy LM for the student summarizer.
    llm_extra_kwargs: Dict[str, Any] = {}
    if not bool(args.enable_thinking):
        llm_extra_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    student_max_tokens = int(args.student_max_tokens)
    student_max_tokens_resolved = student_max_tokens if student_max_tokens > 0 else None
    student_lm = create_vllm_lm(
        port=int(args.student_port),
        model=args.student_model,
        temperature=float(args.student_temperature),
        max_tokens=student_max_tokens_resolved,
        cache=True,
        **llm_extra_kwargs,
    )
    configure_dspy(lm=student_lm)

    # Reflection often needs a much larger token budget than summarization.
    # Keep it separate to avoid GEPA truncation at student max_tokens.
    reflection_max_tokens = int(args.gepa_reflection_max_tokens)
    reflection_max_tokens_resolved = reflection_max_tokens if reflection_max_tokens > 0 else None
    reflection_lm = create_vllm_lm(
        port=int(args.student_port),
        model=(str(args.gepa_reflection_model) if args.gepa_reflection_model else args.student_model),
        temperature=float(args.gepa_reflection_temperature),
        max_tokens=reflection_max_tokens_resolved,
        cache=True,
        **llm_extra_kwargs,
    )

    # Load data.
    records = load_lawstress_records_jsonl(args.records)
    train_records = [r for r in records if r.split == "train"]
    val_records = [r for r in records if r.split == "val"]
    if not train_records:
        raise SystemExit("No train records found in provided LawStress file.")

    trainset = _to_dspy_examples(train_records)
    valset = _to_dspy_examples(val_records)
    _attach_normalized_targets(trainset)
    _attach_normalized_targets(valset)

    # Embedding proxy: train or load.
    proxy_dir = output_dir / "proxy"
    proxy_dir.mkdir(parents=True, exist_ok=True)

    if args.proxy_path is not None:
        proxy_model = load_embedding_proxy(args.proxy_path)
        proxy_path = Path(args.proxy_path)
        proxy_metrics_path = proxy_dir / "proxy_metrics_loaded.json"
        proxy_metrics = {"loaded_from": str(proxy_path), "model_id": proxy_model.model_id}
        _save_json(proxy_metrics_path, proxy_metrics)
    else:
        embedding_client = VLLMEmbeddingClient(
            api_base=str(args.embedding_url),
            model=str(args.embedding_model),
            api_key=str(args.embedding_api_key),
            timeout_seconds=float(args.embedding_timeout_seconds),
            batch_size=int(args.embedding_batch_size),
        )
        proxy_train_examples = build_proxy_training_examples(train_records)
        proxy_val_examples = build_proxy_training_examples(val_records)

        LOGGER.info("Training embedding ridge proxy: train=%d val=%d", len(proxy_train_examples), len(proxy_val_examples))
        proxy_model = fit_embedding_ridge_proxy(
            proxy_train_examples,
            embedding_client=embedding_client,
            ridge_lambda=float(args.ridge_lambda),
            model_id=str(args.proxy_model_id),
        )
        proxy_path = proxy_dir / "embedding_proxy.json"
        proxy_model.save_json(proxy_path)
        proxy_metrics = evaluate_embedding_proxy(
            proxy_model,
            embedding_client=embedding_client,
            eval_examples=proxy_val_examples,
        )
        proxy_metrics_path = proxy_dir / "proxy_metrics.json"
        _save_json(proxy_metrics_path, proxy_metrics)

    # Metric config.
    metric_config = LawStressEvalConfig(
        c1_threshold_norm=float(args.c1_threshold_norm),
        c2_threshold_norm=float(args.c2_threshold_norm),
        c3_threshold_norm=float(args.c3_threshold_norm),
    )
    objective_config = LawStressBootstrapObjectiveConfig(
        aggregate_mode=str(args.objective_aggregate),
        softmin_temperature=float(args.objective_softmin_temperature),
        component_floor=float(args.objective_component_floor),
    )

    embedding_client_for_metric = VLLMEmbeddingClient(
        api_base=str(args.embedding_url),
        model=str(args.embedding_model),
        api_key=str(args.embedding_api_key),
        timeout_seconds=float(args.embedding_timeout_seconds),
        batch_size=int(args.embedding_batch_size),
    )

    metric_fn = create_lawstress_bootstrap_metric(
        proxy_model=proxy_model,
        embedding_client=embedding_client_for_metric,
        config=metric_config,
        objective=objective_config,
    )
    # GEPA's parallel evaluator expects scalar metric values. Keep the richer
    # dict-form metric for diagnostics, but pass a scalar adapter to GEPA.
    def metric_for_gepa(gold, pred, trace=None, pred_name=None, pred_trace=None):
        out = metric_fn(gold, pred, trace, pred_name, pred_trace)
        if isinstance(out, dict):
            return float(out.get("score", 0.0))
        return float(out)

    # Program and baseline evaluation.
    baseline_program = LawStressLocalLawProgram(g=UnifiedG())
    baseline_stats = _mean_metric(metric_fn, baseline_program, valset)
    LOGGER.info("Baseline mean metric on val: %s", baseline_stats)

    # Optimize.
    gepa_log_dir = output_dir / "checkpoints" / "gepa" / "lawstress_unified_g"
    gepa_log_dir.mkdir(parents=True, exist_ok=True)
    capped_metric_calls = int(args.gepa_max_metric_calls) if int(args.gepa_max_metric_calls) > 0 else None
    capped_full_evals = int(args.gepa_max_full_evals) if int(args.gepa_max_full_evals) > 0 else None
    if capped_metric_calls is not None and capped_full_evals is not None:
        raise SystemExit("Set only one of --gepa-max-metric-calls or --gepa-max-full-evals.")

    gepa_auto = str(args.gepa_budget)
    if capped_metric_calls is not None or capped_full_evals is not None:
        gepa_auto = None  # type: ignore[assignment]

    optimizer = dspy.GEPA(
        metric=metric_for_gepa,
        auto=gepa_auto,
        num_threads=int(args.num_threads),
        max_metric_calls=capped_metric_calls,
        max_full_evals=capped_full_evals,
        reflection_lm=reflection_lm,
        log_dir=str(gepa_log_dir),
        track_stats=True,
        seed=int(args.seed),
    )
    compile_kwargs: Dict[str, Any] = {"student": baseline_program, "trainset": trainset}
    if valset:
        compile_kwargs["valset"] = valset

    LOGGER.info(
        "Starting GEPA compile (budget=%s threads=%d reflection_max_tokens=%s reflection_temperature=%.3f enable_thinking=%s)",
        args.gepa_budget,
        int(args.num_threads),
        "auto" if reflection_max_tokens_resolved is None else str(reflection_max_tokens_resolved),
        float(args.gepa_reflection_temperature),
        str(bool(args.enable_thinking)).lower(),
    )
    optimized_program = optimizer.compile(**compile_kwargs)

    optimized_stats = _mean_metric(metric_fn, optimized_program, valset)
    LOGGER.info("Optimized mean metric on val: %s", optimized_stats)

    # Save unified g artifact.
    trained_dir = output_dir / "trained_modules"
    trained_dir.mkdir(parents=True, exist_ok=True)
    g_module = getattr(optimized_program, "g", None)
    if g_module is None:
        raise SystemExit("Optimized program has no attribute 'g'; cannot save unified module.")
    g_path = trained_dir / "unified_g_final.json"
    g_module.save(str(g_path))

    # Write bootstrap stats.
    stats = {
        "created_at": datetime.utcnow().isoformat(),
        "records": str(Path(args.records)),
        "output_dir": str(output_dir),
        "student": {
            "port": int(args.student_port),
            "model": args.student_model,
            "temperature": float(args.student_temperature),
            "max_tokens": student_max_tokens_resolved,
            "enable_thinking": bool(args.enable_thinking),
        },
        "embedding": {
            "url": str(args.embedding_url),
            "model": str(args.embedding_model),
        },
        "proxy": {
            "path": str(proxy_path),
            "metrics_path": str(proxy_metrics_path),
            "model_id": getattr(proxy_model, "model_id", None),
        },
        "gepa": {
            "budget": str(args.gepa_budget),
            "effective_auto": gepa_auto,
            "num_threads": int(args.num_threads),
            "max_metric_calls": int(args.gepa_max_metric_calls),
            "max_full_evals": int(args.gepa_max_full_evals),
            "reflection_model": (str(args.gepa_reflection_model) if args.gepa_reflection_model else args.student_model),
            "reflection_temperature": float(args.gepa_reflection_temperature),
            "reflection_max_tokens": reflection_max_tokens_resolved,
            "seed": int(args.seed),
            "log_dir": str(gepa_log_dir),
        },
        "objective": {
            "aggregate_mode": str(args.objective_aggregate),
            "softmin_temperature": float(args.objective_softmin_temperature),
            "component_floor": float(args.objective_component_floor),
        },
        "val_metric": {
            "baseline": baseline_stats,
            "optimized": optimized_stats,
        },
        "paths": {
            "unified_g": str(g_path),
        },
    }
    _save_json(output_dir / "bootstrap_stats.json", stats)

    LOGGER.info("Saved unified g: %s", g_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
