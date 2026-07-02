#!/usr/bin/env python3
"""Expert-label k-fold GEPA experiment for Benoit scorer optimization."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dspy
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.phase1_optimize_scorer import (  # noqa: E402
    _DIM_FROM_NAME,
    _load_test_pairs,
    _make_examples,
    _make_gepa_metric,
    _make_metric,
    _predict_and_correlate,
)
from src.config.dspy_config import configure_dspy, create_local_engine_lm  # noqa: E402
from src.config.local_inference import resolve_local_inference_config  # noqa: E402
from src.tasks.manifesto.dimensions import BENOIT_DIMENSIONS, PolicyDimension  # noqa: E402
from src.tasks.manifesto.dimension_scorer import DimensionScorer  # noqa: E402


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dimensions", nargs="+", default=sorted(_DIM_FROM_NAME))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--dev-frac", type=float, default=0.2)
    parser.add_argument("--optimizer", choices=["gepa", "bootstrap", "none"], default="gepa")
    parser.add_argument("--metric-mode", choices=["mae", "rank"], default="rank")
    parser.add_argument("--feedback-mode", choices=["scalar", "rich"], default="rich")
    parser.add_argument("--gepa-auto", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--gepa-threads", type=int, default=8)
    parser.add_argument("--gepa-valset-cap", type=int, default=48)
    parser.add_argument("--gepa-max-metric-calls", type=int, default=0)
    parser.add_argument("--max-demos", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reflection-max-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs") / f"phase1_gepa_expert_kfold_{datetime.now(timezone.utc):%Y%m%d_%H%M}")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _fold_indices(n: int, k: int) -> list[list[int]]:
    if k < 2:
        raise ValueError("--k must be at least 2")
    folds = [[] for _ in range(k)]
    for idx in range(n):
        folds[idx % k].append(idx)
    return folds


def _compile_program(
    *,
    args: argparse.Namespace,
    spec,
    baseline: DimensionScorer,
    trainset: list[dspy.Example],
    devset: list[dspy.Example],
    label_center: float | None,
    reflection_lm,
):
    metric = _make_metric(spec.scale.range, mode=args.metric_mode, label_center=label_center)
    if args.optimizer == "none":
        return baseline
    if args.optimizer == "bootstrap":
        compiler = dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=args.max_demos,
            max_labeled_demos=args.max_demos,
        )
        return compiler.compile(baseline, trainset=trainset)

    gepa_kwargs: dict[str, Any] = {
        "metric": _make_gepa_metric(
            spec,
            mode=args.metric_mode,
            label_center=label_center,
            feedback_mode=args.feedback_mode,
        ),
        "reflection_lm": reflection_lm,
        "num_threads": args.gepa_threads,
        "track_stats": True,
    }
    if args.gepa_max_metric_calls > 0:
        gepa_kwargs["max_metric_calls"] = int(args.gepa_max_metric_calls)
    else:
        gepa_kwargs["auto"] = args.gepa_auto
    compiler = dspy.GEPA(**gepa_kwargs)
    gepa_valset = list(devset[:args.gepa_valset_cap]) if args.gepa_valset_cap > 0 else devset
    return compiler.compile(student=baseline, trainset=trainset, valset=gepa_valset)


def _run_fold(args: argparse.Namespace, dim: PolicyDimension, fold_index: int, train_dev_df: pd.DataFrame,
              test_df: pd.DataFrame, reflection_lm) -> dict[str, Any]:
    spec = BENOIT_DIMENSIONS[dim]
    shuffled = train_dev_df.sample(frac=1.0, random_state=args.seed + fold_index).reset_index(drop=True)
    n_dev = max(5, int(round(len(shuffled) * args.dev_frac)))
    dev_df = shuffled.iloc[:n_dev]
    train_df = shuffled.iloc[n_dev:]
    trainset = _make_examples(train_df)
    devset = _make_examples(dev_df)
    testset = _make_examples(test_df)

    label_center = None
    if args.metric_mode == "rank":
        labels = [float(ex.expert_mean) for ex in trainset]
        label_center = sum(labels) / len(labels)

    baseline = DimensionScorer(spec, use_cot=False)
    baseline_dev = _predict_and_correlate(baseline, devset)["pearson"]
    baseline_test = _predict_and_correlate(baseline, testset)["pearson"]

    t0 = time.time()
    optimized = _compile_program(
        args=args,
        spec=spec,
        baseline=baseline,
        trainset=trainset,
        devset=devset,
        label_center=label_center,
        reflection_lm=reflection_lm,
    )
    compile_seconds = time.time() - t0

    if args.optimizer == "none":
        optimized_dev = baseline_dev
        final_program = baseline
        final_source = "baseline"
    else:
        optimized_dev = _predict_and_correlate(optimized, devset)["pearson"]
        opt_dev_r = optimized_dev.get("pearson_r")
        base_dev_r = baseline_dev.get("pearson_r")
        if opt_dev_r is not None and base_dev_r is not None and opt_dev_r < base_dev_r:
            final_program = baseline
            final_source = "baseline"
        else:
            final_program = optimized
            final_source = "optimized"

    final_test = baseline_test if final_source == "baseline" else _predict_and_correlate(final_program, testset)["pearson"]
    return {
        "dimension": dim.value,
        "fold": fold_index,
        "n_train": len(trainset),
        "n_dev": len(devset),
        "n_test": len(testset),
        "label_center": label_center,
        "compile_time_seconds": round(compile_seconds, 1),
        "baseline_dev": baseline_dev,
        "optimized_dev": optimized_dev,
        "baseline_test": baseline_test,
        "final_source": final_source,
        "final_test": final_test,
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    local_inference = resolve_local_inference_config(args)
    lm = create_local_engine_lm(**local_inference.dspy_kwargs(cache=True))
    configure_dspy(lm=lm)
    reflection_lm = create_local_engine_lm(
        engine=local_inference.engine,
        endpoints=local_inference.endpoints,
        model=args.model,
        temperature=0.7,
        max_tokens=args.reflection_max_tokens,
        cache=True,
    )

    all_rows: list[dict[str, Any]] = []
    for dim_name in args.dimensions:
        dim = _DIM_FROM_NAME[dim_name]
        df = _load_test_pairs(dim).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        folds = _fold_indices(len(df), args.k)
        for fold_index, test_indices in enumerate(folds):
            test_set = set(test_indices)
            test_df = df.iloc[test_indices].reset_index(drop=True)
            train_dev_df = df.iloc[[i for i in range(len(df)) if i not in test_set]].reset_index(drop=True)
            fold_dir = args.output_dir / dim.value / f"fold_{fold_index}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Running %s fold %d/%d", dim.value, fold_index + 1, args.k)
            row = _run_fold(args, dim, fold_index, train_dev_df, test_df, reflection_lm)
            all_rows.append(row)
            (fold_dir / "report.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    dim_macro = {}
    for dim_name in sorted({r["dimension"] for r in all_rows}):
        vals = [
            float(r["final_test"]["pearson_r"])
            for r in all_rows
            if r["dimension"] == dim_name and r["final_test"].get("pearson_r") is not None
        ]
        dim_macro[dim_name] = sum(vals) / len(vals) if vals else None
    macro_vals = [v for v in dim_macro.values() if v is not None]
    summary = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": getattr(lm, "model", "unknown"),
            "optimizer": args.optimizer,
            "metric_mode": args.metric_mode,
            "feedback_mode": args.feedback_mode,
            "reflection_max_tokens": args.reflection_max_tokens,
            "k": args.k,
            "seed": args.seed,
        },
        "folds": all_rows,
        "dimension_macro_final_test_pearson": dim_macro,
        "macro_final_test_pearson": sum(macro_vals) / len(macro_vals) if macro_vals else None,
    }
    (args.output_dir / "kfold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", args.output_dir / "kfold_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
