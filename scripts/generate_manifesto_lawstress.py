#!/usr/bin/env python3
"""Generate the MVP synthetic local-law stress benchmark for information extraction."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from src.tasks.manifesto.lawstress_generator import (
    LawStressSpec,
    build_reference_summary_rows,
    generate_lawstress_records,
    generate_lawstress_specs,
    summarize_spec_balance,
    write_benchmark_docs_jsonl,
    write_jsonl,
    write_lawstress_records_jsonl,
)
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



_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _parse_single_score(text: str) -> Optional[float]:
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


def _build_teacher_scorer(
    client: OpenAIChatClient,
    *,
    temperature: float,
    max_tokens: int,
):
    def _score(text: str) -> float:
        response = client.chat(
            system=(
                "You are a strict directional coder for information extraction. "
                "Return exactly one numeric RILE score in [-100, 100]."
            ),
            user=(
                "Score this text on a RILE-style directional scale. Return only one number.\n\n"
                f"TEXT:\n{text}"
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = _parse_single_score(response)
        if parsed is None:
            raise ValueError(f"Unable to parse teacher score from: {response!r}")
        return float(parsed)

    return _score


def _build_teacher_rewriter(
    client: OpenAIChatClient,
    *,
    temperature: float,
    max_tokens: int,
):
    def _rewrite(text: str, spec: LawStressSpec, truth_raw: float) -> str:
        return client.chat(
            system=(
                "Rewrite this source text to be natural and difficult to summarize while preserving "
                "all key factual commitments, caveats, and directional stance. "
                "Do not introduce math exercises, coding tasks, or question-answer formats. "
                "Do not output explanations."
            ),
            user=(
                f"Family={spec.family}; law_target={spec.law_target}; difficulty={spec.difficulty}; "
                f"target_rile={truth_raw:.2f}.\n"
                "Preserve the intended stance, entities, and qualifiers.\n\n"
                f"TEXT:\n{text}"
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return _rewrite


def _build_reference_summarizer(
    client: OpenAIChatClient,
    *,
    temperature: float,
    max_tokens: int,
):
    def _summarize(text: str, spec: LawStressSpec, truth_raw: float) -> str:
        return client.chat(
            system=(
                "Create a concise summary for information extraction that preserves directional stance, "
                "factual commitments, and key qualifiers. Return only the summary text."
            ),
            user=(
                f"Law target={spec.law_target}; family={spec.family}; target_rile={truth_raw:.2f}.\n"
                "Do not convert this into math/coding tasks or QA format.\n\n"
                f"TEXT:\n{text}"
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return _summarize


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate information-extraction law-stress synthetic benchmark"
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default timestamped path)")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-size", type=int, default=600)
    parser.add_argument("--val-size", type=int, default=150)
    parser.add_argument("--test-size", type=int, default=150)
    parser.add_argument("--hard-ratio", type=float, default=0.8)
    parser.add_argument("--real-anchor-ratio", type=float, default=0.3)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Concurrent worker shards for generation (must be >=2 for multi-record runs).",
    )

    parser.add_argument("--teacher-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--teacher-model", type=str, default=DEFAULT_MAIN_MODEL)
    parser.add_argument("--teacher-api-key", type=str, default="EMPTY")
    parser.add_argument("--teacher-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable model thinking/reasoning traces for teacher calls (default: disabled).",
    )

    parser.add_argument("--teacher-score-temperature", type=float, default=0.0)
    parser.add_argument("--teacher-score-max-tokens", type=int, default=32)

    parser.add_argument("--teacher-rewrite-temperature", type=float, default=0.4)
    parser.add_argument("--teacher-rewrite-max-tokens", type=int, default=2048)
    parser.add_argument("--reference-temperature", type=float, default=0.3)
    parser.add_argument("--reference-max-tokens", type=int, default=800)

    parser.add_argument("--disable-teacher-gates", action="store_true")
    parser.add_argument("--disable-teacher-rewrite", action="store_true")
    parser.add_argument("--disable-reference-summary", action="store_true")
    parser.add_argument("--skip-counterexample-pairs", action="store_true")

    parser.add_argument("--hard-drift-threshold-norm", type=float, default=0.20)
    parser.add_argument("--control-drift-threshold-norm", type=float, default=0.08)
    parser.add_argument("--doc-score-tolerance-raw", type=float, default=10.0)
    parser.add_argument("--segment-score-tolerance-raw", type=float, default=12.0)

    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _build_teacher_components(args: argparse.Namespace):
    teacher_client = OpenAIChatClient(
        base_url=args.teacher_base_url,
        model=args.teacher_model,
        api_key=args.teacher_api_key,
        timeout_seconds=float(args.teacher_timeout_seconds),
        enable_thinking=bool(args.enable_thinking),
    )

    teacher_score_fn = None
    if not args.disable_teacher_gates:
        teacher_score_fn = _build_teacher_scorer(
            teacher_client,
            temperature=float(args.teacher_score_temperature),
            max_tokens=int(args.teacher_score_max_tokens),
        )

    teacher_rewrite_fn = None
    if not args.disable_teacher_rewrite:
        teacher_rewrite_fn = _build_teacher_rewriter(
            teacher_client,
            temperature=float(args.teacher_rewrite_temperature),
            max_tokens=int(args.teacher_rewrite_max_tokens),
        )

    reference_summary_fn = None
    if not args.disable_reference_summary:
        reference_summary_fn = _build_reference_summarizer(
            teacher_client,
            temperature=float(args.reference_temperature),
            max_tokens=int(args.reference_max_tokens),
        )

    return teacher_score_fn, teacher_rewrite_fn, reference_summary_fn


def _generate_records_single(
    args: argparse.Namespace,
    specs: List[LawStressSpec],
    *,
    seed: int,
):
    teacher_score_fn, teacher_rewrite_fn, reference_summary_fn = _build_teacher_components(args)
    return generate_lawstress_records(
        specs,
        seed=int(seed),
        max_attempts=int(args.max_attempts),
        teacher_score_fn=teacher_score_fn,
        teacher_rewrite_fn=teacher_rewrite_fn,
        reference_summary_fn=reference_summary_fn,
        hard_drift_threshold_norm=float(args.hard_drift_threshold_norm),
        control_drift_threshold_norm=float(args.control_drift_threshold_norm),
        doc_score_tolerance_raw=float(args.doc_score_tolerance_raw),
        segment_score_tolerance_raw=float(args.segment_score_tolerance_raw),
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data") / "lawstress" / f"run_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_sizes = {
        "train": int(args.train_size),
        "val": int(args.val_size),
        "test": int(args.test_size),
    }

    specs = generate_lawstress_specs(
        split_sizes=split_sizes,
        hard_ratio=float(args.hard_ratio),
        real_anchor_ratio=float(args.real_anchor_ratio),
        seed=int(args.seed),
    )
    balance = summarize_spec_balance(specs)

    requested_workers = int(args.num_workers)
    if requested_workers < 1:
        raise ValueError(f"--num-workers must be >= 1 (got {requested_workers})")
    if len(specs) > 1 and requested_workers < 2:
        raise ValueError(
            "Single-worker generation is disabled for multi-record LawStress runs. "
            f"Set --num-workers >= 2 (got {requested_workers}, requested_records={len(specs)})."
        )

    max_workers = requested_workers
    if max_workers > 1 and len(specs) > 1:
        shards: List[List[LawStressSpec]] = [[] for _ in range(max_workers)]
        for idx, spec in enumerate(specs):
            shards[idx % max_workers].append(spec)
        shards = [bucket for bucket in shards if bucket]
        LOGGER.info("Generating with %d concurrent worker shards", len(shards))

        records = []
        worker_stats: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(shards)) as pool:
            futures = {}
            for worker_idx, shard_specs in enumerate(shards):
                worker_seed = int(args.seed) + 1009 * int(worker_idx)
                future = pool.submit(
                    _generate_records_single,
                    args,
                    shard_specs,
                    seed=worker_seed,
                )
                futures[future] = {
                    "worker_idx": int(worker_idx),
                    "seed": int(worker_seed),
                    "requested": int(len(shard_specs)),
                }
            for future in as_completed(futures):
                meta = futures[future]
                produced = future.result()
                records.extend(list(produced))
                worker_stats.append(
                    {
                        **meta,
                        "generated": int(len(produced)),
                        "dropped": int(meta["requested"] - len(produced)),
                    }
                )
        records.sort(key=lambda row: str(row.example_id))
    else:
        worker_stats = []
        records = _generate_records_single(
            args,
            list(specs),
            seed=int(args.seed),
        )

    records_path = output_dir / "lawstress_records.jsonl"
    benchmark_path = output_dir / "benchmark_docs.jsonl"
    refs_path = output_dir / "reference_summaries.jsonl"

    write_lawstress_records_jsonl(records_path, records)
    write_benchmark_docs_jsonl(benchmark_path, records)
    write_jsonl(refs_path, build_reference_summary_rows(records))

    if not args.skip_counterexample_pairs:
        pair_rows: List[Dict[str, Any]] = []
        for record in records:
            pair_rows.append(
                {
                    "example_id": record.example_id,
                    "split": record.split,
                    "law_target": record.law_target,
                    "summary_bad": record.naive_summary,
                    "summary_good": record.reference_summary,
                    "y_raw": record.y_raw,
                }
            )
        write_jsonl(output_dir / "counterexample_pairs.jsonl", pair_rows)

    summary = {
        "generated": len(records),
        "requested": len(specs),
        "dropped": len(specs) - len(records),
        "num_workers": int(max_workers),
        "worker_stats": worker_stats,
        "teacher_model": args.teacher_model,
        "teacher_base_url": args.teacher_base_url,
        "split_sizes": split_sizes,
        "balance": balance,
        "paths": {
            "records": str(records_path),
            "benchmark_docs": str(benchmark_path),
            "reference_summaries": str(refs_path),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    benchmark_ref = benchmark_ref_from_parts(
        family="manifesto_lawstress",
        scope="generation",
        name="manifesto_lawstress",
        dataset_id=str(output_dir),
        metadata={"split_sizes": split_sizes, "balance": balance},
    )
    method_ref = method_ref_from_parts(
        family="benchmark_generation",
        variant="lawstress_teacher",
        adapter="lawstress_generation",
        metadata=metadata_with_roles(
            {"teacher_model": str(args.teacher_model)},
            roles={
                "summarizer": chat_role_ref(
                    role="summarizer",
                    model=str(args.teacher_model),
                    base_url=str(args.teacher_base_url),
                )
            },
            oracle=oracle_ref(kind="teacher_generated_labels", source=str(args.teacher_model)),
        ),
    )
    write_canonical_sidecars(
        output_dir,
        title="generate_manifesto_lawstress",
        adapter_id="lawstress_generation",
        benchmark_refs=(benchmark_ref,),
        method_refs=(method_ref,),
        phases=("generate",),
        artifacts={
            "manifest_json": str(output_dir / "manifest.json"),
            "records_jsonl": str(records_path),
            "benchmark_docs_jsonl": str(benchmark_path),
            "reference_summaries_jsonl": str(refs_path),
        },
        result_rows=(
            ResultRow(
                experiment_id="",
                phase="generate",
                benchmark_ref=benchmark_ref,
                method_ref=method_ref,
                metric_name="generated_records",
                metric_value=len(records),
                artifact_refs=("manifest_json", "records_jsonl"),
                metadata={"requested": len(specs), "dropped": len(specs) - len(records)},
            ),
        ),
        state="completed",
        metadata={"teacher_model": str(args.teacher_model)},
        launch_command=sys.argv,
        report_profiles=("runtime_eval_summary",),
    )

    LOGGER.info("Generated %d/%d records", len(records), len(specs))
    LOGGER.info("Output directory: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
