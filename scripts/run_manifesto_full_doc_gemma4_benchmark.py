#!/usr/bin/env python3
"""Evaluate Gemma-4 by scoring raw full Manifesto documents directly."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.script_io import (  # noqa: E402
    append_jsonl as _append_jsonl,
    now_iso as _now_iso,
    now_stamp as _now_stamp,
    write_json as _write_json,
)
from src.experiments.script_parse import parse_csv as _parse_csv  # noqa: E402
from src.tasks.manifesto.coverage_split import (  # noqa: E402
    DEFAULT_DIMENSIONS,
    SPLIT_SCHEMA_VERSION,
    load_split_ids as _load_split_ids,
    load_split_summary as _load_split_summary,
    resolve_full_doc_data_dir as _resolve_full_doc_data_dir,
)
from src.tasks.manifesto.full_doc_helpers import (  # noqa: E402
    DIMENSION_BY_NAME as _DIM_FROM_NAME,
    dimension_metrics as _dimension_metrics,
    limit_text as _limit_text,
    load_tokenizer as _load_tokenizer,
    parse_response as _parse_response,
    usage_dict as _usage_dict,
)
from src.tasks.manifesto.result_rows import load_rows_by_dimension as _load_rows_by_dimension  # noqa: E402
from src.ctreepo.contracts import objective_metadata, run_manifest_metadata  # noqa: E402
from src.experiments import (  # noqa: E402
    ARTIFACT_CALLS_JSONL,
    ARTIFACT_PREDICTIONS_JSONL,
    ARTIFACT_SUMMARY_JSON,
    ExperimentContext,
    JsonlCallTraceSink,
    SamplingPlan,
    benchmark_ref_from_parts,
    chat_role_ref,
    experiment_method_ref,
    oracle_ref,
)
from src.tasks.manifesto import ManifestoDataset  # noqa: E402
from src.tasks.manifesto.benoit_scoring_contexts import get_benoit_scoring_context  # noqa: E402
from src.tasks.manifesto.expert_scale import (  # noqa: E402
    EXPERT_SCALE_NORMALIZED_1_7,
    resolve_benoit_expert_target,
)


_HUMAN_TEMPLATE = "Analyze the following political text:\n\n{document}"



def _build_context(
    *,
    output_dir: Path,
    split_dir: Path,
    split_summary: Mapping[str, Any],
    split_names: Sequence[str],
    dimensions: Sequence[str],
    model: str,
    base_url: str,
    seed: int,
    command: Sequence[str],
) -> ExperimentContext:
    split_digest = str(split_summary.get("split_manifest_digest") or "")
    benchmark_ref = benchmark_ref_from_parts(
        family="manifesto_rile",
        scope="full_doc_direct",
        dataset_id=str(split_dir),
        name="Manifesto all-six full-document direct scoring",
        metadata={
            "split_manifest_digest": split_digest,
            "dimensions": list(dimensions),
        },
    )
    method_ref = experiment_method_ref(
        family="full_doc_direct_scorer",
        variant="gemma4_raw_document",
        engine="vllm_openai",
        model=str(model),
        adapter="manifesto_full_doc_direct",
        roles={
            "scorer": chat_role_ref(
                role="scorer",
                engine="vllm",
                model=str(model),
                base_url=str(base_url),
            )
        },
        oracle=oracle_ref(
            kind="benchmark_labels",
            source="benoit_expert_means",
            metadata={"target_scale": EXPERT_SCALE_NORMALIZED_1_7},
        ),
        metadata={"split_manifest_digest": split_digest},
    )
    sampling_plan = SamplingPlan(
        seed=int(seed),
        split=",".join(split_names),
        strategy=str(
            ((split_summary.get("sampling_plan") or {}).get("strategy"))
            or "manifesto_coverage_split"
        ),
        unit="document",
        frame="manifesto_all6_labeled",
        metadata={
            "split_manifest_digest": split_digest,
            "split_dir": str(split_dir),
            "dimensions": list(dimensions),
        },
    )
    return ExperimentContext(
        output_root=output_dir,
        benchmark_ref=benchmark_ref,
        method_ref=method_ref,
        title="manifesto_full_doc_gemma4_benchmark",
        adapter_id="manifesto_full_doc_direct",
        phases=("evaluate",),
        sampling=sampling_plan,
        report_profiles=("runtime_eval_summary",),
        launch_command=tuple(str(item) for item in command),
        metadata={
            "split_summary_schema_version": str(split_summary.get("schema_version") or ""),
            "split_manifest_digest": split_digest,
        },
    )


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_ids = _load_split_ids(split_dir)
    split_summary = _load_split_summary(split_dir)
    dimensions = _parse_csv(args.dimensions or ",".join(split_summary.get("dimensions") or DEFAULT_DIMENSIONS))
    splits = _parse_csv(args.splits)
    source_root = Path(args.source_root or split_summary.get("source_root") or "outputs/overnight_benoit/full_pipeline")
    rows_by_dimension = _load_rows_by_dimension(source_root, dimensions)
    resolved_mp_data_dir = _resolve_full_doc_data_dir(args.mp_data_dir, project_root=PROJECT_ROOT)
    dataset = ManifestoDataset(data_dir=resolved_mp_data_dir, require_text=True)
    base_url = str(args.base_url).rstrip("/")
    model = str(args.model)
    context = _build_context(
        output_dir=output_dir,
        split_dir=split_dir,
        split_summary=split_summary,
        split_names=splits,
        dimensions=dimensions,
        model=model,
        base_url=base_url,
        seed=int(args.seed),
        command=sys.argv,
    )
    calls_path = output_dir / "calls.jsonl"
    call_sink = JsonlCallTraceSink(
        calls_path,
        defaults={"surface": "chat_openai", "engine": "vllm", "model": model},
    )
    tokenizer = _load_tokenizer(str(args.tokenizer_model)) if args.tokenizer_model else None
    client = None
    if not bool(args.mock_predictions):
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=str(args.api_key))
        if not model:
            models = client.models.list().data
            if not models:
                raise RuntimeError(f"no models served by {base_url}")
            model = str(models[0].id)

    predictions_path = output_dir / "predictions.jsonl"
    if bool(args.reuse_predictions) and predictions_path.exists():
        prediction_rows = [
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        if predictions_path.exists():
            predictions_path.unlink()
        max_n = None if args.max_n is None else int(args.max_n)
        selected_items: List[Tuple[str, str]] = [
            (str(split), str(doc_id))
            for split in splits
            for doc_id in list(split_ids.get(split, []))
        ]
        if max_n is not None:
            selected_items = selected_items[:max_n]

        prediction_rows = []
        for split, doc_id in selected_items:
            sample = dataset.get_sample(str(doc_id))
            if sample is None or not str(getattr(sample, "text", "") or "").strip():
                continue
            raw_text = str(sample.text)
            limited_text, limit_meta = _limit_text(
                raw_text,
                max_input_chars=int(args.max_input_chars),
                max_input_tokens=int(args.max_input_tokens),
                tokenizer=tokenizer,
            )
            for dim in dimensions:
                row = rows_by_dimension.get(dim, {}).get(str(doc_id))
                if row is None:
                    continue
                expert = resolve_benoit_expert_target(
                    row,
                    dimension=str(dim),
                    scale=EXPERT_SCALE_NORMALIZED_1_7,
                )
                if expert is None:
                    continue
                started = time.time()
                raw_response = ""
                usage: Dict[str, Any] = {}
                error = ""
                if bool(args.mock_predictions):
                    prediction = max(1.0, min(7.0, round(float(expert))))
                    raw_response = str(int(prediction))
                else:
                    messages = [
                        {
                            "role": "system",
                            "content": get_benoit_scoring_context(_DIM_FROM_NAME[str(dim)]),
                        },
                        {
                            "role": "user",
                            "content": _HUMAN_TEMPLATE.format(document=limited_text),
                        },
                    ]
                    try:
                        response = client.chat.completions.create(  # type: ignore[union-attr]
                            model=model,
                            messages=messages,
                            temperature=float(args.temperature),
                            max_tokens=int(args.max_tokens),
                            extra_body={"seed": int(args.seed), "top_p": 1.0},
                        )
                        raw_response = response.choices[0].message.content or ""
                        usage = _usage_dict(getattr(response, "usage", None))
                    except Exception as exc:  # noqa: BLE001
                        error = str(exc)
                        raw_response = ""
                    prediction = _parse_response(raw_response)
                latency_ms = 1000.0 * (time.time() - started)
                call_meta = context.call_metadata(
                    role="scorer",
                    request_kind="full_doc_direct_score",
                    problem_id=f"{doc_id}:{dim}",
                    runner_id="manifesto_full_doc_direct",
                    artifacts={"predictions_jsonl": str(predictions_path)},
                )
                call_sink(
                    {
                        **call_meta,
                        "document_id": str(doc_id),
                        "unit_id": f"{doc_id}:{dim}",
                        "latency_ms": latency_ms,
                        "usage": usage,
                        "error": error,
                        "metadata": {
                            "dimension": str(dim),
                            "split": str(split),
                            "truncated": bool(limit_meta.get("truncated")),
                            "limit_kind": str(limit_meta.get("limit_kind") or ""),
                        },
                    }
                )
                prediction_rows.append(
                    {
                        "manifesto_id": str(doc_id),
                        "split": str(split),
                        "dimension": str(dim),
                        "expert_score_1_7": float(expert),
                        "prediction": prediction,
                        "is_na": prediction is None,
                        "raw_response": raw_response[:200],
                        "error": error,
                        "char_len_full": len(raw_text),
                        "char_len_input": len(limited_text),
                        **limit_meta,
                    }
                )
        _append_jsonl(predictions_path, prediction_rows)
    per_dimension = {
        dim: _dimension_metrics(prediction_rows, dimension=dim)
        for dim in dimensions
    }
    pearsons = [
        float(metrics["pearson"].get("pearson_r"))
        for metrics in per_dimension.values()
        if metrics["pearson"].get("pearson_r") is not None
    ]
    macro_pearson = sum(pearsons) / len(pearsons) if pearsons else None
    metrics: Dict[str, Any] = {
        "macro_external_expert_pearson": macro_pearson,
        "prediction_rows": len(prediction_rows),
        "na_count": sum(1 for row in prediction_rows if row.get("prediction") is None),
    }
    for dim, dim_metrics in per_dimension.items():
        metrics[f"{dim}_external_expert_pearson"] = dim_metrics["pearson"].get("pearson_r")
        metrics[f"{dim}_mae"] = dim_metrics.get("mae")
        metrics[f"{dim}_n_scored"] = dim_metrics.get("n_scored")
    summary = {
        "created_at": _now_iso(),
        "schema_version": "ctreepo.manifesto_full_doc_direct_benchmark.v1",
        "split_manifest_digest": str(split_summary.get("split_manifest_digest") or ""),
        "split_summary_schema_version": str(split_summary.get("schema_version") or ""),
        "split_dir": str(split_dir),
        "source_root": str(source_root),
        "mp_data_dir": str(resolved_mp_data_dir or ""),
        "text_source_kind": "raw_manifesto_full_document",
        "dimensions": list(dimensions),
        "splits": list(splits),
        "model": model,
        "base_url": base_url,
        "max_input_chars": int(args.max_input_chars),
        "max_input_tokens": int(args.max_input_tokens),
        "tokenizer_model": str(args.tokenizer_model or ""),
        "mock_predictions": bool(args.mock_predictions),
        "per_dimension": per_dimension,
        "metrics": metrics,
        "artifacts": {
            ARTIFACT_PREDICTIONS_JSONL: str(predictions_path),
            ARTIFACT_CALLS_JSONL: str(calls_path),
        },
    }
    summary_path = _write_json(output_dir / "summary.json", summary)
    artifacts = {
        ARTIFACT_PREDICTIONS_JSONL: str(predictions_path),
        ARTIFACT_CALLS_JSONL: str(calls_path),
        ARTIFACT_SUMMARY_JSON: str(summary_path),
    }
    context.record(
        {"metrics": metrics, "artifacts": artifacts, "metadata": {"summary": "full_doc_direct"}},
        phase="evaluate",
        artifacts=artifacts,
        metadata={
            "split_manifest_digest": str(split_summary.get("split_manifest_digest") or ""),
            "splits": list(splits),
            "dimensions": list(dimensions),
        },
        state="completed",
    )
    run_manifest = run_manifest_metadata(
        run_id=f"manifesto.full_doc_gemma4_benchmark.{_now_stamp()}",
        domain="manifesto_rile",
        role="full_doc_direct_scorer",
        backend="vllm",
        status="completed",
        input_contracts=[
            {
                "kind": "manifesto_coverage_split",
                "schema_version": SPLIT_SCHEMA_VERSION,
                "digest": str(split_summary.get("split_manifest_digest") or ""),
                "uri": str(split_dir),
            }
        ],
        f_init="full_doc_direct_scorer",
        g_init="none",
        f_lineage={"model": model, "base_url": base_url, "prompt": "benoit_exact_rubric"},
        g_lineage={"init": "none"},
        objective=objective_metadata(
            objective_family="manifesto_full_doc_direct",
            local_law_estimator="none",
            root_share=1.0,
            local_law_component_weights={},
            metadata={
                "dimensions": list(dimensions),
                "splits": list(splits),
                "split_manifest_digest": str(split_summary.get("split_manifest_digest") or ""),
            },
        ),
        optimizer_config={
            "temperature": float(args.temperature),
            "max_tokens": int(args.max_tokens),
            "max_input_chars": int(args.max_input_chars),
            "max_input_tokens": int(args.max_input_tokens),
        },
        output_artifacts=[
            {"kind": ARTIFACT_SUMMARY_JSON, "uri": str(summary_path)},
            {"kind": ARTIFACT_PREDICTIONS_JSONL, "uri": str(predictions_path)},
            {"kind": ARTIFACT_CALLS_JSONL, "uri": str(calls_path)},
            {"kind": "experiment_directory", "uri": str(output_dir)},
        ],
        audit_results={"ok": True, "prediction_rows": len(prediction_rows)},
        quarantine={"classification": "valid_run_manifest_v1"},
        command=sys.argv,
        publication_ready=True,
        metadata={
            "runner": "scripts/run_manifesto_full_doc_gemma4_benchmark.py",
            "experiment_api": "ExperimentContext",
            "split_manifest_digest": str(split_summary.get("split_manifest_digest") or ""),
        },
    )
    _write_json(output_dir / "run_manifest.json", run_manifest)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--dimensions", default="")
    parser.add_argument("--splits", default="test")
    parser.add_argument("--mp-data-dir", type=Path, default=None)
    parser.add_argument("--base-url", default="http://localhost:8010/v1")
    parser.add_argument("--model", default="nvidia/Gemma-4-31B-IT-NVFP4")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-input-chars", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument("--tokenizer-model", default="")
    parser.add_argument("--mock-predictions", action="store_true")
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--reuse-predictions", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = Path("outputs") / "manifesto_full_doc_gemma4_benchmark" / _now_stamp()
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_benchmark(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "prediction_rows": summary["metrics"]["prediction_rows"],
                "macro_external_expert_pearson": summary["metrics"]["macro_external_expert_pearson"],
                "summary": str(Path(args.output_dir) / "summary.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
