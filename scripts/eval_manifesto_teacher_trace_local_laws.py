#!/usr/bin/env python3
"""Evaluate local-law preservation on real-anchor teacher-trace records."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import sys
from typing import List, Optional, Sequence

import requests

# Add project root for direct script execution.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config.dspy_config import configure_dspy, create_vllm_lm
from src.core.protocols import format_merge_input
from src.tasks.manifesto.teacher_trace_eval import (
    TeacherTraceEvalConfig,
    build_eval_metrics,
    build_predictions,
    load_predictions_jsonl,
    render_eval_report_markdown,
    score_predictions,
    write_eval_results_jsonl,
    write_predictions_jsonl,
)
from src.tasks.manifesto.lawstress_bootstrap_program import UnifiedG
from src.tasks.manifesto.teacher_trace_generator import (
    TeacherTraceRecord,
    load_teacher_trace_records_jsonl,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_MAIN_MODEL = "/mnt/data/models/nvidia/Qwen3.5-397B-A17B-NVFP4"
DEFAULT_STUDENT_MODEL = "/mnt/data/models/AxionML/Qwen3.5-35B-A3B-NVFP4"
_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


class OpenAIChatClient:
    """Minimal OpenAI-compatible chat client."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: float = 120.0,
        enable_thinking: bool = False,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.api_key = str(api_key)
        self.timeout_seconds = float(timeout_seconds)
        self.enable_thinking = bool(enable_thinking)

    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "chat_template_kwargs": {
                "enable_thinking": bool(self.enable_thinking),
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "").strip()


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


def _build_summarize_fn(
    client: OpenAIChatClient,
    *,
    temperature: float,
    max_tokens: int,
    include_score_conditioning: bool,
):
    def _summarize(text: str, rubric: str, source_rile_raw: float, hop: int) -> str:
        conditioning = ""
        if include_score_conditioning:
            conditioning = f"Target directional score to preserve: {float(source_rile_raw):.2f}\n"
        return client.chat(
            system=(
                "Summarize faithfully for information extraction while preserving directional signal, "
                "factual commitments, and caveats."
            ),
            user=(
                f"Rubric:\n{rubric}\n\n"
                f"{conditioning}"
                f"Resummary hop: {int(hop)}\n\n"
                "Do NOT mention any numeric score or the term RILE.\n\n"
                f"Text:\n{text}\n\n"
                "Summary:"
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return _summarize


def _build_merge_fn(
    client: OpenAIChatClient,
    *,
    temperature: float,
    max_tokens: int,
    include_score_conditioning: bool,
):
    def _merge(left: str, right: str, rubric: str, source_rile_raw: float) -> str:
        conditioning = ""
        if include_score_conditioning:
            conditioning = f"Target directional score to preserve: {float(source_rile_raw):.2f}\n\n"
        return client.chat(
            system=(
                "Merge two summaries into one faithful information-extraction summary preserving "
                "directional constraints, entities, and qualifiers."
            ),
            user=(
                f"Rubric:\n{rubric}\n\n"
                f"{conditioning}"
                "Do NOT mention any numeric score or the term RILE.\n\n"
                f"Summary A:\n{left}\n\n"
                f"Summary B:\n{right}\n\n"
                "Merged Summary:"
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return _merge


def _build_score_fn(
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


def _filter_records_by_split(
    records: Sequence[TeacherTraceRecord],
    requested_splits: Sequence[str],
) -> List[TeacherTraceRecord]:
    if not requested_splits:
        return list(records)
    normalized = {str(value).strip().lower() for value in requested_splits if str(value).strip()}
    if not normalized or "all" in normalized:
        return list(records)
    return [row for row in records if str(row.split).strip().lower() in normalized]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate real-anchor teacher-trace local laws")
    parser.add_argument("--records", type=Path, required=True, help="Path to teacher_trace_records.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["summarize_only", "score_only", "full"],
    )
    parser.add_argument("--predictions-path", type=Path, default=None)
    parser.add_argument("--splits", nargs="*", default=["test"]) 

    parser.add_argument("--dspy-module", type=Path, default=None, help="Optional unified_g_final.json path")
    parser.add_argument("--student-port", type=int, default=8000)
    parser.add_argument("--student-model", type=str, default=None)
    parser.add_argument("--student-temperature", type=float, default=0.2)
    parser.add_argument("--student-max-tokens", type=int, default=900)

    parser.add_argument("--summarizer-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--summarizer-model", type=str, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--summarizer-api-key", type=str, default="EMPTY")
    parser.add_argument("--summarizer-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--summarizer-temperature", type=float, default=0.2)
    parser.add_argument("--summarizer-max-tokens", type=int, default=900)
    parser.add_argument(
        "--include-score-conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include the teacher-provided source_rile_raw in the summarizer/merge prompt. "
            "Disable to avoid label leakage unless you will provide an estimated score at inference."
        ),
    )

    parser.add_argument("--scorer-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--scorer-model", type=str, default=DEFAULT_MAIN_MODEL)
    parser.add_argument("--scorer-api-key", type=str, default="EMPTY")
    parser.add_argument("--scorer-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--scorer-temperature", type=float, default=0.0)
    parser.add_argument("--scorer-max-tokens", type=int, default=64)

    parser.add_argument("--resummary-hops", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--c1-threshold-raw", type=float, default=10.0)
    parser.add_argument("--c2-threshold-raw", type=float, default=6.0)
    parser.add_argument("--c3-threshold-raw", type=float, default=8.0)
    parser.add_argument("--neutral-raw", type=float, default=0.0)

    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


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

    predictions_path = args.predictions_path or (output_dir / "predictions.jsonl")

    all_records = load_teacher_trace_records_jsonl(args.records)
    records = _filter_records_by_split(all_records, args.splits)
    if not records:
        raise ValueError(
            f"No records after split filtering from {args.records}; requested splits={list(args.splits)}"
        )
    _validate_num_workers(int(args.num_workers), len(records), label="eval_manifesto_teacher_trace_local_laws")

    config = TeacherTraceEvalConfig(
        mode=str(args.mode),
        resummary_hops=int(args.resummary_hops),
        c1_threshold_raw=float(args.c1_threshold_raw),
        c2_threshold_raw=float(args.c2_threshold_raw),
        c3_threshold_raw=float(args.c3_threshold_raw),
        neutral_raw=float(args.neutral_raw),
    )

    predictions = None
    if args.mode in {"summarize_only", "full"}:
        if args.dspy_module is not None:
            lm = create_vllm_lm(
                port=int(args.student_port),
                model=args.student_model,
                temperature=float(args.student_temperature),
                max_tokens=int(args.student_max_tokens),
                cache=True,
            )
            configure_dspy(lm=lm)
            g_module = UnifiedG()
            g_module.load(str(args.dspy_module))

            def summarize_fn(text: str, rubric: str, source_rile_raw: float, hop: int) -> str:  # noqa: ARG001
                return str(g_module(content=str(text or ""), rubric=str(rubric or "")) or "").strip()

            def merge_fn(left: str, right: str, rubric: str, source_rile_raw: float) -> str:  # noqa: ARG001
                payload = format_merge_input(str(left or ""), str(right or ""))
                return str(g_module(content=payload, rubric=str(rubric or "")) or "").strip()
        else:
            summarizer_client = OpenAIChatClient(
                base_url=args.summarizer_base_url,
                model=args.summarizer_model,
                api_key=args.summarizer_api_key,
                timeout_seconds=float(args.summarizer_timeout_seconds),
                enable_thinking=bool(args.enable_thinking),
            )
            summarize_fn = _build_summarize_fn(
                summarizer_client,
                temperature=float(args.summarizer_temperature),
                max_tokens=int(args.summarizer_max_tokens),
                include_score_conditioning=bool(args.include_score_conditioning),
            )
            merge_fn = _build_merge_fn(
                summarizer_client,
                temperature=float(args.summarizer_temperature),
                max_tokens=int(args.summarizer_max_tokens),
                include_score_conditioning=bool(args.include_score_conditioning),
            )

        predictions = build_predictions(
            records,
            summarize_fn=summarize_fn,
            merge_fn=merge_fn,
            resummary_hops=int(config.resummary_hops),
            num_workers=int(args.num_workers),
        )
        write_predictions_jsonl(predictions_path, predictions)
        LOGGER.info("Wrote predictions: %s", predictions_path)
        if args.mode == "summarize_only":
            return 0

    if args.mode == "score_only":
        predictions = load_predictions_jsonl(predictions_path)

    if predictions is None:
        raise RuntimeError("Predictions missing; run summarize_only/full first")

    scorer_client = OpenAIChatClient(
        base_url=args.scorer_base_url,
        model=args.scorer_model,
        api_key=args.scorer_api_key,
        timeout_seconds=float(args.scorer_timeout_seconds),
        enable_thinking=bool(args.enable_thinking),
    )
    score_fn = _build_score_fn(
        scorer_client,
        temperature=float(args.scorer_temperature),
        max_tokens=int(args.scorer_max_tokens),
    )

    results = score_predictions(
        records,
        predictions,
        score_fn=score_fn,
        config=config,
        num_workers=int(args.num_workers),
    )

    write_eval_results_jsonl(output_dir / "eval_results.jsonl", results)

    metrics, groups = build_eval_metrics(results)
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "eval_by_group.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    (output_dir / "eval_report.md").write_text(
        render_eval_report_markdown(metrics, groups),
        encoding="utf-8",
    )

    LOGGER.info("Evaluation complete. Output directory: %s", output_dir)
    LOGGER.info("Overall metrics: %s", json.dumps(metrics.get("overall", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
