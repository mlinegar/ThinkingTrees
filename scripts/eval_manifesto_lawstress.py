#!/usr/bin/env python3
"""Evaluate summaries on the information-extraction law-stress benchmark."""

from __future__ import annotations

import argparse
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
from src.tasks.manifesto.lawstress_eval import (
    LawStressEvalConfig,
    build_eval_metrics,
    build_predictions,
    load_predictions_jsonl,
    render_eval_report_markdown,
    score_and_judge_predictions,
    write_eval_results_jsonl,
    write_predictions_jsonl,
)
from src.tasks.manifesto.lawstress_generator import load_lawstress_records_jsonl
from src.experiments import (
    ResultRow,
    benchmark_ref_from_parts,
    chat_role_ref,
    metadata_with_roles,
    method_ref_from_parts,
    oracle_ref,
    write_canonical_sidecars,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_MAIN_MODEL = "/mnt/data/models/nvidia/Qwen3.5-397B-A17B-NVFP4"
DEFAULT_STUDENT_MODEL = "/mnt/data/models/AxionML/Qwen3.5-35B-A3B-NVFP4"



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


def _build_summarize_fn(
    client: OpenAIChatClient,
    *,
    temperature: float,
    max_tokens: int,
):
    def _summarize(text: str, rubric: str) -> str:
        return client.chat(
            system=(
                "Summarize faithfully for information extraction while preserving directional signal, "
                "factual commitments, and caveats."
            ),
            user=f"Rubric:\n{rubric}\n\nText:\n{text}\n\nSummary:",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return _summarize


def _build_merge_fn(
    client: OpenAIChatClient,
    *,
    temperature: float,
    max_tokens: int,
):
    def _merge(left: str, right: str, rubric: str) -> str:
        return client.chat(
            system=(
                "Merge two summaries into one faithful information-extraction summary preserving "
                "directional constraints, entities, and qualifiers."
            ),
            user=(
                f"Rubric:\n{rubric}\n\n"
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
            user=f"Score this text on a RILE-style directional scale. Return only one number.\n\nTEXT:\n{text}",
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


def _build_genrm_judge(*_args, **_kwargs):
    """Legacy compatibility hook: GenRM judging is intentionally unavailable."""
    raise ValueError(
        "GenRM judging is no longer supported in LawStress evaluation. "
        "Use local-law bootstrap (teacher scorer + proxy/GEPA), no GenRM."
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate information-extraction law-stress benchmark")
    parser.add_argument("--records", type=Path, required=True, help="Path to lawstress_records.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["summarize_only", "score_and_judge_only", "full"],
    )
    parser.add_argument("--predictions-path", type=Path, default=None)
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Optional split labels to evaluate (e.g. train val test). Default: all splits.",
    )

    parser.add_argument("--summarizer-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--summarizer-model", type=str, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--summarizer-api-key", type=str, default="EMPTY")
    parser.add_argument("--summarizer-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--summarizer-temperature", type=float, default=0.2)
    parser.add_argument("--summarizer-max-tokens", type=int, default=800)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable model thinking/reasoning traces for summarizer/scorer calls (default: disabled).",
    )

    parser.add_argument("--scorer-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--scorer-model", type=str, default=DEFAULT_MAIN_MODEL)
    parser.add_argument("--scorer-api-key", type=str, default="EMPTY")
    parser.add_argument("--scorer-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--scorer-temperature", type=float, default=0.0)
    parser.add_argument("--scorer-max-tokens", type=int, default=32)

    parser.add_argument("--genrm-base-url", type=str, default="http://localhost:8001/v1")
    parser.add_argument("--genrm-model", type=str, default=None)
    parser.add_argument("--genrm-temperature", type=float, default=0.6)
    parser.add_argument("--genrm-top-p", type=float, default=0.95)
    parser.add_argument("--genrm-max-tokens", type=int, default=1024)
    parser.add_argument(
        "--disable-genrm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "GenRM judging is deprecated in this evaluator. Keep enabled (default) for scorer-only local-law "
            "evaluation."
        ),
    )

    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--resummary-hops", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--c1-threshold-norm", type=float, default=0.10)
    parser.add_argument("--c2-threshold-norm", type=float, default=0.06)
    parser.add_argument("--c3-threshold-norm", type=float, default=0.08)

    parser.add_argument("--baseline-metrics", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _load_baseline_overall(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        overall = payload.get("overall")
        if isinstance(overall, dict):
            return dict(overall)
    return None


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

    if not bool(args.disable_genrm):
        raise ValueError(
            "GenRM judging is no longer supported in LawStress evaluation. "
            "Use local-law bootstrap (teacher scorer + proxy/GEPA), no GenRM."
        )

    predictions_path = args.predictions_path or (output_dir / "predictions.jsonl")

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

    _validate_num_workers(int(args.num_workers), len(records), label="eval_manifesto_lawstress")

    config = LawStressEvalConfig(
        mode=str(args.mode),
        chunk_size=int(args.chunk_size),
        resummary_hops=int(args.resummary_hops),
        c1_threshold_norm=float(args.c1_threshold_norm),
        c2_threshold_norm=float(args.c2_threshold_norm),
        c3_threshold_norm=float(args.c3_threshold_norm),
    )

    predictions = None
    if args.mode in {"summarize_only", "full"}:
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
        )
        merge_fn = _build_merge_fn(
            summarizer_client,
            temperature=float(args.summarizer_temperature),
            max_tokens=int(args.summarizer_max_tokens),
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
            benchmark_ref = benchmark_ref_from_parts(
                family="manifesto_lawstress",
                scope="eval",
                name="manifesto_lawstress",
                dataset_id=str(args.records),
            )
            method_ref = method_ref_from_parts(
                family="lawstress_summarization",
                variant=str(args.mode),
                adapter="lawstress_eval",
                metadata=metadata_with_roles(
                    {"mode": str(args.mode)},
                    roles={
                        "summarizer": chat_role_ref(
                            role="summarizer",
                            model=str(args.summarizer_model),
                            base_url=str(args.summarizer_base_url),
                        )
                    },
                    oracle=oracle_ref(kind="benchmark_labels", source=str(args.records)),
                ),
            )
            write_canonical_sidecars(
                output_dir,
                title="eval_manifesto_lawstress",
                adapter_id="lawstress_eval",
                benchmark_refs=(benchmark_ref,),
                method_refs=(method_ref,),
                phases=("summarize",),
                artifacts={"predictions_jsonl": str(predictions_path)},
                result_rows=(
                    ResultRow(
                        experiment_id="",
                        phase="summarize",
                        benchmark_ref=benchmark_ref,
                        method_ref=method_ref,
                        metric_name="predictions",
                        metric_value=len(predictions),
                        artifact_refs=("predictions_jsonl",),
                    ),
                ),
                state="completed",
                metadata={"mode": str(args.mode)},
                launch_command=sys.argv,
                report_profiles=("runtime_eval_summary",),
            )
            return 0

    if args.mode == "score_and_judge_only":
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

    # GenRM judging is intentionally disabled in large-model-only mode.
    judge_fn = None

    results = score_and_judge_predictions(
        records,
        predictions,
        score_fn=score_fn,
        judge_fn=judge_fn,
        config=config,
        num_workers=int(args.num_workers),
    )

    write_eval_results_jsonl(output_dir / "eval_results.jsonl", results)

    baseline_overall = _load_baseline_overall(args.baseline_metrics)
    metrics, groups = build_eval_metrics(
        results,
        config=config,
        baseline_overall=baseline_overall,
    )

    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "eval_by_group.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    (output_dir / "eval_report.md").write_text(render_eval_report_markdown(metrics, groups), encoding="utf-8")
    benchmark_ref = benchmark_ref_from_parts(
        family="manifesto_lawstress",
        scope="eval",
        name="manifesto_lawstress",
        dataset_id=str(args.records),
    )
    method_ref = method_ref_from_parts(
        family="lawstress_eval",
        variant=str(args.mode),
        adapter="lawstress_eval",
        metadata=metadata_with_roles(
            {"mode": str(args.mode)},
            roles={
                "summarizer": chat_role_ref(
                    role="summarizer",
                    model=str(args.summarizer_model),
                    base_url=str(args.summarizer_base_url),
                ),
                "scorer": chat_role_ref(
                    role="scorer",
                    model=str(args.scorer_model),
                    base_url=str(args.scorer_base_url),
                ),
            },
            oracle=oracle_ref(kind="benchmark_labels", source=str(args.records)),
        ),
    )
    overall = dict(metrics.get("overall") or {})
    result_rows = [
        ResultRow(
            experiment_id="",
            phase="eval",
            benchmark_ref=benchmark_ref,
            method_ref=method_ref,
            metric_name=str(key),
            metric_value=value,
            artifact_refs=("eval_metrics_json", "eval_results_jsonl"),
            metadata={"mode": str(args.mode)},
        )
        for key, value in overall.items()
        if isinstance(value, (int, float, bool))
    ]
    write_canonical_sidecars(
        output_dir,
        title="eval_manifesto_lawstress",
        adapter_id="lawstress_eval",
        benchmark_refs=(benchmark_ref,),
        method_refs=(method_ref,),
        phases=("eval",),
        artifacts={
            "predictions_jsonl": str(predictions_path),
            "eval_results_jsonl": str(output_dir / "eval_results.jsonl"),
            "eval_metrics_json": str(output_dir / "eval_metrics.json"),
            "eval_by_group_json": str(output_dir / "eval_by_group.json"),
            "eval_report_md": str(output_dir / "eval_report.md"),
        },
        result_rows=tuple(result_rows),
        state="completed",
        metadata={"mode": str(args.mode)},
        launch_command=sys.argv,
        report_profiles=("runtime_eval_summary",),
    )

    LOGGER.info("Evaluation complete. Output directory: %s", output_dir)
    LOGGER.info("Overall metrics: %s", json.dumps(metrics.get("overall", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
