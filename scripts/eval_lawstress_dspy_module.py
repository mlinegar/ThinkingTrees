#!/usr/bin/env python3
"""Evaluate LawStress with a DSPy unified-g module + teacher scorer.

Supports two-stage execution so the student summarizer and teacher scorer do
not need to be running at the same time:
1) summarize_only: run student -> write predictions
2) score_only: run teacher scorer -> compute local-law metrics
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional


# Add project root for direct script execution.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.tasks.manifesto.openai_chat import OpenAIChatClient
from src.config.dspy_config import configure_dspy, create_local_engine_lm
from src.config.local_inference import resolve_local_inference_config
from src.core.protocols import format_merge_input
from src.tasks.manifesto.lawstress_eval import (
    LawStressEvalConfig,
    build_eval_metrics,
    build_predictions,
    compute_metric_row,
    load_predictions_jsonl,
    render_eval_report_markdown,
    score_and_judge_predictions,
    write_eval_results_jsonl,
    write_predictions_jsonl,
    RILE_RUBRIC,
)
from src.tasks.manifesto.lawstress_generator import load_lawstress_records_jsonl
from src.tasks.manifesto.lawstress_bootstrap_program import UnifiedG


LOGGER = logging.getLogger(__name__)

DEFAULT_TEACHER_MODEL = "/mnt/data/models/nvidia/Qwen3.5-397B-A17B-NVFP4"



_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _parse_score(text: str) -> Optional[float]:
    rendered = str(text or "").strip()
    if not rendered:
        return None
    matches = _NUMERIC_RE.findall(rendered)
    if not matches:
        return None
    try:
        value = float(matches[-1])
    except (TypeError, ValueError):
        return None
    return max(-100.0, min(100.0, value))


def _parse_last_number(text: str) -> Optional[float]:
    rendered = str(text or "").strip()
    if not rendered:
        return None
    matches = _NUMERIC_RE.findall(rendered)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except (TypeError, ValueError):
        return None


def _normalize_split(value: str) -> str:
    rendered = str(value or "").strip().lower()
    if rendered == "validation":
        return "val"
    return rendered


def _build_teacher_score_fn(
    client: OpenAIChatClient,
    *,
    temperature: float,
    max_tokens: int,
):
    def _score(text: str) -> float:
        response = client.chat(
            system="Return exactly one numeric RILE score in [-100,100] for directional information signal.",
            user=(
                "Score this text on a RILE-style directional scale. Return only one number.\n\n"
                f"TEXT:\n{text}"
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = _parse_score(response)
        retry = None
        if parsed is None:
            retry = client.chat(
                system=(
                    "Output exactly one numeric RILE score in [-100,100]. "
                    "No words, no explanation, no JSON."
                ),
                user=(
                    "Extract and return only the numeric RILE score.\n"
                    "Output format example: -12.50\n\n"
                    f"TEXT:\n{text}"
                ),
                temperature=0.0,
                max_tokens=max(8, int(max_tokens)),
            )
            parsed = _parse_score(retry)
        if parsed is None:
            salvage = _parse_last_number(f"{response}\n{retry or ''}")
            if salvage is not None:
                parsed = max(-100.0, min(100.0, float(salvage)))
        if parsed is None:
            raise ValueError(f"Could not parse score responses: first={response!r} retry={retry!r}")
        return float(parsed)

    return _score


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LawStress using DSPy unified-g module.")
    parser.add_argument("--records", type=Path, required=True, help="Path to lawstress_records.jsonl")
    parser.add_argument("--module", type=Path, required=True, help="Path to trained unified_g_final.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["summarize_only", "score_only", "full"],
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Optional split labels to evaluate (e.g. test). Default: all splits.",
    )

    # Student summarizer LM (DSPy).
    parser.add_argument("--student-port", type=int, default=8000)
    parser.add_argument("--student-model", type=str, default=None)
    parser.add_argument("--student-temperature", type=float, default=0.2)
    parser.add_argument("--student-max-tokens", type=int, default=800)

    # Teacher scorer LM (OpenAI-compatible chat).
    parser.add_argument("--teacher-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--teacher-model", type=str, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--teacher-api-key", type=str, default="EMPTY")
    parser.add_argument("--teacher-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--teacher-temperature", type=float, default=0.0)
    parser.add_argument("--teacher-max-tokens", type=int, default=32)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable model thinking/reasoning traces for teacher scoring calls (default: disabled).",
    )

    # Local-law config.
    parser.add_argument("--resummary-hops", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--c1-threshold-norm", type=float, default=0.10)
    parser.add_argument("--c2-threshold-norm", type=float, default=0.06)
    parser.add_argument("--c3-threshold-norm", type=float, default=0.08)

    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _save_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _summarize_with_g(g: UnifiedG, text: str, rubric: str) -> str:
    return str(g(content=str(text or ""), rubric=str(rubric or "")) or "").strip()


def _validate_num_workers(num_workers: int, workload_size: int, *, label: str) -> int:
    requested = int(num_workers)
    if requested < 1:
        raise ValueError(f"{label}: --num-workers must be >= 1 (got {requested})")
    if int(workload_size) > 1 and requested < 2:
        raise ValueError(
            f"{label}: single-worker mode is disabled for multi-item workloads. "
            f"Set --num-workers >= 2 (got {requested}, workload_size={workload_size})."
        )
    return requested


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_lawstress_records_jsonl(args.records)
    if not records:
        raise ValueError(f"No records loaded from {args.records}")

    if args.splits:
        requested_splits = sorted({_normalize_split(value) for value in args.splits if str(value).strip()})
        if requested_splits:
            allowed = set(requested_splits)
            records = [row for row in records if _normalize_split(row.split) in allowed]
            if not records:
                raise ValueError(f"No LawStress records remain after split filter: {requested_splits}")
            LOGGER.info("Evaluating only splits=%s (n=%d)", ",".join(requested_splits), len(records))

    _validate_num_workers(int(args.num_workers), len(records), label="eval_lawstress_dspy_module")

    baseline_predictions_path = output_dir / "baseline_predictions.jsonl"
    optimized_predictions_path = output_dir / "optimized_predictions.jsonl"

    if args.mode in ("summarize_only", "full"):
        local_inference = resolve_local_inference_config(
            {
                "port": int(args.student_port),
                "model": args.student_model,
                "temperature": float(args.student_temperature),
                "max_tokens": int(args.student_max_tokens),
            }
        )
        lm = create_local_engine_lm(**local_inference.dspy_kwargs(cache=True))
        configure_dspy(lm=lm)

        # Baseline module (unoptimized).
        baseline_g = UnifiedG()

        # Optimized module.
        optimized_g = UnifiedG()
        optimized_g.load(str(args.module))

        def baseline_summarize(text: str, rubric: str) -> str:
            return _summarize_with_g(baseline_g, text, rubric)

        def baseline_merge(left: str, right: str, rubric: str) -> str:
            return _summarize_with_g(baseline_g, format_merge_input(left, right), rubric)

        def optimized_summarize(text: str, rubric: str) -> str:
            return _summarize_with_g(optimized_g, text, rubric)

        def optimized_merge(left: str, right: str, rubric: str) -> str:
            return _summarize_with_g(optimized_g, format_merge_input(left, right), rubric)

        LOGGER.info("Building baseline predictions...")
        baseline_predictions = build_predictions(
            records,
            summarize_fn=baseline_summarize,
            merge_fn=baseline_merge,
            rubric=RILE_RUBRIC,
            resummary_hops=int(args.resummary_hops),
            num_workers=int(args.num_workers),
        )
        write_predictions_jsonl(baseline_predictions_path, baseline_predictions)

        LOGGER.info("Building optimized predictions...")
        optimized_predictions = build_predictions(
            records,
            summarize_fn=optimized_summarize,
            merge_fn=optimized_merge,
            rubric=RILE_RUBRIC,
            resummary_hops=int(args.resummary_hops),
            num_workers=int(args.num_workers),
        )
        write_predictions_jsonl(optimized_predictions_path, optimized_predictions)

        LOGGER.info("Wrote: %s", baseline_predictions_path)
        LOGGER.info("Wrote: %s", optimized_predictions_path)

    if args.mode in ("score_only", "full"):
        baseline_predictions = load_predictions_jsonl(baseline_predictions_path)
        optimized_predictions = load_predictions_jsonl(optimized_predictions_path)

        teacher_client = OpenAIChatClient(
            base_url=args.teacher_base_url,
            model=args.teacher_model,
            api_key=args.teacher_api_key,
            timeout_seconds=float(args.teacher_timeout_seconds),
            enable_thinking=bool(args.enable_thinking),
        )
        score_fn = _build_teacher_score_fn(
            teacher_client,
            temperature=float(args.teacher_temperature),
            max_tokens=int(args.teacher_max_tokens),
        )

        config = LawStressEvalConfig(
            chunk_size=2000,
            resummary_hops=int(args.resummary_hops),
            c1_threshold_norm=float(args.c1_threshold_norm),
            c2_threshold_norm=float(args.c2_threshold_norm),
            c3_threshold_norm=float(args.c3_threshold_norm),
        )

        LOGGER.info("Scoring baseline predictions with teacher...")
        baseline_results = score_and_judge_predictions(
            records,
            baseline_predictions,
            score_fn=score_fn,
            judge_fn=None,
            config=config,
            num_workers=int(args.num_workers),
        )
        baseline_overall = compute_metric_row(baseline_results)

        LOGGER.info("Scoring optimized predictions with teacher...")
        optimized_results = score_and_judge_predictions(
            records,
            optimized_predictions,
            score_fn=score_fn,
            judge_fn=None,
            config=config,
            num_workers=int(args.num_workers),
        )

        baseline_metrics, baseline_groups = build_eval_metrics(
            baseline_results,
            config=config,
            baseline_overall=None,
        )
        optimized_metrics, optimized_groups = build_eval_metrics(
            optimized_results,
            config=config,
            baseline_overall=baseline_overall,
        )

        # Persist.
        write_eval_results_jsonl(output_dir / "baseline_eval_results.jsonl", baseline_results)
        write_eval_results_jsonl(output_dir / "optimized_eval_results.jsonl", optimized_results)

        _save_json(output_dir / "baseline_eval_metrics.json", {"metrics": baseline_metrics, "groups": baseline_groups})
        _save_json(output_dir / "optimized_eval_metrics.json", {"metrics": optimized_metrics, "groups": optimized_groups})

        report_lines: List[str] = []
        report_lines.append("# LawStress DSPy Module Evaluation")
        report_lines.append("")
        report_lines.append("## Baseline")
        report_lines.append("")
        report_lines.append(render_eval_report_markdown(baseline_metrics, baseline_groups))
        report_lines.append("")
        report_lines.append("## Optimized (relative to baseline)")
        report_lines.append("")
        report_lines.append(render_eval_report_markdown(optimized_metrics, optimized_groups))
        report_path = output_dir / "report.md"
        report_path.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

        manifest = {
            "created_at": datetime.utcnow().isoformat(),
            "records": str(Path(args.records)),
            "module": str(Path(args.module)),
            "student": {
                "port": int(args.student_port),
                "model": args.student_model,
                "temperature": float(args.student_temperature),
                "max_tokens": int(args.student_max_tokens),
            },
            "teacher": {
                "base_url": str(args.teacher_base_url),
                "model": str(args.teacher_model),
            },
            "paths": {
                "baseline_predictions": str(baseline_predictions_path),
                "optimized_predictions": str(optimized_predictions_path),
                "baseline_metrics": str(output_dir / "baseline_eval_metrics.json"),
                "optimized_metrics": str(output_dir / "optimized_eval_metrics.json"),
                "report": str(report_path),
            },
        }
        _save_json(output_dir / "manifest.json", manifest)

        LOGGER.info("Wrote report: %s", report_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
