#!/usr/bin/env python3
"""
Phase 2a: joint-across-dimensions scorer optimization.

One shared `dspy.Predict` (inside `JointDimensionScorer`), pooled training
examples from all 6 dims (~1,350 total before holdout exclusion; labels
from Benoit's open-weight LLM ensemble on non-test manifestos), evaluated
per-dimension on the held-out Benoit expert-benchmark rows.

Compares against:
  * baseline = the same `JointDimensionScorer` before optimization (should
    reproduce the per-dim scorer-only numbers in §4b of the plan doc).
  * benoit = Figure 1 proprietary 18-score ensemble published values.

Writes:
  * report.json   macro + per-dim r for baseline and optimized.
  * per_dim.jsonl one row per (manifesto, dim) prediction pair.

Usage:
    python scripts/phase2_joint_optimize.py \\
        --port 8010 \\
        --optimizer bootstrap \\
        --output-dir outputs/phase2/joint_optimize
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import dspy
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config.dspy_config import configure_dspy, create_vllm_lm, create_vllm_lm_multi
from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r
from src.tasks.manifesto.dimensions import BENOIT_DIMENSIONS, PolicyDimension
from src.tasks.manifesto.expert_benchmarks import (
    load_benoit_expert_means,
    load_benoit_masked_summaries,
    load_joint_train_pairs,
)
from src.tasks.manifesto.joint_scorer import JointDimensionScorer

logger = logging.getLogger(__name__)

_ORDER = [
    PolicyDimension.ECONOMIC,
    PolicyDimension.SOCIAL,
    PolicyDimension.IMMIGRATION,
    PolicyDimension.EU,
    PolicyDimension.ENVIRONMENT,
    PolicyDimension.DECENTRALIZATION,
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--ports", type=int, nargs="+", default=None,
                   help="Multiple vLLM ports for round-robin load-balancing (overrides --port).")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--optimizer", choices=["bootstrap", "miprov2", "gepa", "none"], default="bootstrap")
    p.add_argument("--gepa-auto", choices=["light", "medium", "heavy"], default="light",
                   help="GEPA budget preset. light ~300 metric calls, medium ~1000, heavy ~3000.")
    p.add_argument("--gepa-threads", type=int, default=4,
                   help="GEPA num_threads for parallel evaluation.")
    p.add_argument("--train-pool", default="openweight", choices=["openweight", "expert"])
    p.add_argument("--dev-frac", type=float, default=0.15)
    p.add_argument("--max-demos", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional explicit generation cap. Default uses create_vllm_lm() auto-sizing.",
    )
    p.add_argument("--output-dir", type=Path,
                   default=project_root / "outputs" / "phase2" /
                   f"joint_optimize_{datetime.now(timezone.utc):%Y%m%d_%H%M}")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _load_test_examples(dim: PolicyDimension) -> pd.DataFrame:
    summaries = load_benoit_masked_summaries(dimension=dim)
    experts = load_benoit_expert_means(dim)
    lookup = {
        str(r.manifesto).removesuffix(".txt"): float(r.expert_mean)
        for r in experts.itertuples()
    }
    summaries["label"] = summaries["manifesto_stem"].map(lookup)
    return summaries.dropna(subset=["label"]).reset_index(drop=True)


def _example(row) -> dspy.Example:
    spec = BENOIT_DIMENSIONS[PolicyDimension(row.dimension)]
    return dspy.Example(
        summary=row.summary,
        dimension_spec=spec,
        expert_mean=float(row.label),
        dimension=row.dimension,
    ).with_inputs("summary", "dimension_spec")


def _metric(example, prediction, trace=None):
    if isinstance(prediction, dict):
        raw = prediction.get("score")
    else:
        raw = getattr(prediction, "score", None)
    if raw is None or raw == "" or str(raw).strip().lower() in {"na", "n/a", "none"}:
        return 0.0
    try:
        pred_val = float(raw)
    except (TypeError, ValueError):
        return 0.0
    err = abs(pred_val - float(example.expert_mean)) / 6.0  # 1-7 range → 6
    return max(0.0, 1.0 - err)


def _predict(program: dspy.Module, ex: dspy.Example) -> float | None:
    try:
        pred = program(summary=ex.summary, dimension_spec=ex.dimension_spec)
    except Exception:  # noqa: BLE001
        return None
    raw = pred.get("score") if isinstance(pred, dict) else getattr(pred, "score", None)
    if raw in (None, "", "NA", "N/A"):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _per_dim_report(program: dspy.Module, label: str, output_dir: Path) -> dict:
    per_dim = {}
    raw_rows = []
    for dim in _ORDER:
        test_df = _load_test_examples(dim)
        preds, truths = [], []
        for row in test_df.itertuples():
            ex = _example(row)
            p = _predict(program, ex)
            preds.append(p)
            truths.append(float(row.label))
            raw_rows.append({
                "phase": label, "dimension": dim.value,
                "manifesto_stem": row.manifesto_stem,
                "pred": p, "expert_mean": float(row.label),
            })
        rep = compute_corpus_pearson_r(preds, truths)
        per_dim[dim.value] = rep.as_dict()
        logger.info(
            "[%s] %-18s r=%+.3f n=%d CI[%+.3f,%+.3f]",
            label, dim.value, rep.pearson_r, rep.n, rep.pearson_ci_low, rep.pearson_ci_high,
        )
    # Append raw per-pair rows
    out = output_dir / f"per_dim_{label}.jsonl"
    with out.open("w") as fp:
        for r in raw_rows:
            fp.write(json.dumps(r) + "\n")
    macro = sum(v["pearson_r"] for v in per_dim.values()) / len(per_dim)
    return {"per_dim": per_dim, "macro_pearson_r": macro}


def _save_joint_scorer_artifacts(
    output_dir: Path,
    *,
    final_program: dspy.Module,
    optimized_program: dspy.Module | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for label, program, filename in (
        ("final_program", final_program, "final_program.json"),
        ("scorer_final", final_program, "scorer_final.json"),
        ("optimized_program", optimized_program, "optimized_program.json"),
        ("optimized_scorer", optimized_program, "optimized_scorer.json"),
    ):
        if program is None:
            continue
        path = output_dir / filename
        try:
            program.save(str(path))
            saved[label] = str(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save %s artifact to %s: %s", label, path, exc)
    return saved


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build per-dim test key exclusion map and exclude the union globally so
    # the pooled scorer never trains on a manifesto that appears in any
    # dimension's held-out evaluation set.
    test_keys_per_dim = {dim: set(_load_test_examples(dim)["manifesto_stem"]) for dim in _ORDER}
    global_holdout_keys = set().union(*test_keys_per_dim.values())

    logger.info("Loading joint training pool (%s)", args.train_pool)
    train_full = load_joint_train_pairs(
        args.train_pool,
        test_keys_per_dim=test_keys_per_dim,
        global_holdout_keys=global_holdout_keys,
    )
    logger.info("Pooled train rows: %d (across %d dims)", len(train_full), train_full["dimension"].nunique())

    shuffled = train_full.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_dev = max(5, int(round(len(shuffled) * args.dev_frac)))
    dev_df = shuffled.iloc[:n_dev]
    train_df = shuffled.iloc[n_dev:]
    logger.info("Split: train=%d dev=%d", len(train_df), len(dev_df))

    trainset = [_example(r) for r in train_df.itertuples()]
    devset = [_example(r) for r in dev_df.itertuples()]

    lm_kwargs = {
        "model": args.model,
        "temperature": args.temperature,
        "cache": True,
    }
    if args.max_tokens is not None:
        lm_kwargs["max_tokens"] = args.max_tokens
    if args.ports:
        logger.info("Load-balancing LM across ports %s (T=%g)", args.ports, args.temperature)
        lm = create_vllm_lm_multi(ports=args.ports, **lm_kwargs)
    else:
        logger.info("Configuring LM on port %d (T=%g)", args.port, args.temperature)
        lm = create_vllm_lm(port=args.port, **lm_kwargs)
    configure_dspy(lm=lm)

    baseline = JointDimensionScorer(use_cot=False)

    logger.info("Evaluating baseline per-dim (n=6 dims)")
    t0 = time.time()
    baseline_report = _per_dim_report(baseline, "baseline", args.output_dir)
    logger.info("Baseline macro r=%+.3f (%.1fs)", baseline_report["macro_pearson_r"], time.time() - t0)

    report = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": getattr(lm, "model", "unknown"),
            "optimizer": args.optimizer,
            "train_pool": args.train_pool,
            "n_train": len(trainset), "n_dev": len(devset),
            "dev_frac": args.dev_frac,
            "max_demos": args.max_demos,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "global_holdout_manifestos": len(global_holdout_keys),
        },
        "baseline": baseline_report,
    }
    final_program = baseline
    optimized_program = None

    if args.optimizer != "none":
        if args.optimizer == "bootstrap":
            compiler = dspy.BootstrapFewShot(
                metric=_metric,
                max_bootstrapped_demos=args.max_demos,
                max_labeled_demos=args.max_demos,
            )
            logger.info("Compiling with BootstrapFewShot(max_demos=%d)", args.max_demos)
            t0 = time.time()
            optimized = compiler.compile(baseline, trainset=trainset)
        elif args.optimizer == "miprov2":
            compiler = dspy.MIPROv2(metric=_metric, auto="light", num_threads=4)
            logger.info("Compiling with MIPROv2(auto=light)")
            t0 = time.time()
            optimized = compiler.compile(baseline, trainset=trainset, valset=devset)
        elif args.optimizer == "gepa":
            # GEPA metric signature: (gold, pred, trace, pred_name, pred_trace) -> float | ScoreWithFeedback
            def _gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
                return _metric(gold, pred, trace=trace)
            compiler = dspy.GEPA(
                metric=_gepa_metric,
                auto=args.gepa_auto,
                reflection_lm=lm,
                num_threads=args.gepa_threads,
                track_stats=True,
            )
            logger.info("Compiling with GEPA(auto=%s, threads=%d)", args.gepa_auto, args.gepa_threads)
            t0 = time.time()
            optimized = compiler.compile(student=baseline, trainset=trainset, valset=devset)
        else:
            raise ValueError(args.optimizer)
        logger.info("Compile done in %.1fs", time.time() - t0)

        logger.info("Evaluating optimized per-dim")
        t0 = time.time()
        opt_report = _per_dim_report(optimized, "optimized", args.output_dir)
        logger.info("Optimized macro r=%+.3f (%.1fs)", opt_report["macro_pearson_r"], time.time() - t0)
        report["optimized"] = opt_report
        final_program = optimized
        optimized_program = optimized

    report["artifacts"] = _save_joint_scorer_artifacts(
        args.output_dir,
        final_program=final_program,
        optimized_program=optimized_program,
    )

    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    logger.info("Outputs in %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
