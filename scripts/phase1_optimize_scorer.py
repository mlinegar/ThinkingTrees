#!/usr/bin/env python3
"""
Phase 1b ablation: DSPy-optimize f_score on Benoit summaries × supervision
that DOES NOT include any of Benoit's 235 expert-benchmarked manifestos.

Frozen: Benoit's anonymized summaries from data_masked.csv (f_summarize held
fixed to Benoit's GPT-4o output, on a Benoit-disjoint training pool).
Optimized: the `score` predictor inside `DimensionScorer`.
Training labels: Benoit's open-weight LLM ensemble mean from
data_llms_all_openweight.rds, which covers ~245 manifestos NOT in the
expert-benchmark test set. (We could also use expert means on those 245 if
they happen to have CHES coverage, but openweight-mean is uniformly available.)
Held-out test: the 235 Benoit expert manifestos with expert_mean labels.

Metric is per-example for the optimizer; the final report also computes
corpus-level Pearson r on the test set for direct comparability with Benoit
Figure 1 published values.

Usage:
    python scripts/phase1_optimize_scorer.py \\
        --port 8010 --dimension economic \\
        --optimizer bootstrap \\
        --output-dir outputs/phase1_opt_scorer_economic
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional

import dspy
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config.dspy_config import configure_dspy, create_vllm_lm
from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r
from src.tasks.manifesto.dimensions import BENOIT_DIMENSIONS, PolicyDimension
from src.tasks.manifesto.dimension_scorer import DimensionScorer
from src.tasks.manifesto.expert_benchmarks import (
    benoit_ensemble_mean,
    load_benoit_expert_means,
    load_benoit_llm_scores,
    load_benoit_masked_summaries,
)

logger = logging.getLogger(__name__)

_DIM_FROM_NAME = {d.value: d for d in PolicyDimension}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--dimension", type=str, default="economic", choices=sorted(_DIM_FROM_NAME))
    p.add_argument("--optimizer", type=str, default="bootstrap",
                   choices=["bootstrap", "miprov2", "gepa", "none"],
                   help="DSPy optimizer. `none` = just evaluate baseline. "
                        "GEPA here is *scorer-only*: each rollout is one scoring "
                        "call on a Benoit cached summary, not a full tree pipeline. "
                        "An order of magnitude cheaper than phase3 GEPA.")
    p.add_argument("--gepa-auto", type=str, default="light",
                   choices=["light", "medium", "heavy"])
    p.add_argument("--gepa-threads", type=int, default=8)
    p.add_argument("--gepa-valset-cap", type=int, default=48,
                   help="Cap GEPA's internal Pareto/evaluation valset. "
                        "Use 0 to pass the full dev set.")
    p.add_argument("--gepa-max-metric-calls", type=int, default=0,
                   help="Explicit GEPA metric-call budget. If 0, use --gepa-auto.")
    p.add_argument("--metric-mode", type=str, default="mae",
                   choices=["mae", "rank"],
                   help="Per-example training metric. mae = level accuracy "
                        "(default, classical). rank = sign-of-deviation-from-center, "
                        "which is closer in spirit to the corpus-level Pearson r "
                        "we report on test. Rank mode helps when train and test "
                        "metrics would otherwise reward different things.")
    p.add_argument("--feedback-mode", type=str, default="rich",
                   choices=["scalar", "rich"],
                   help="GEPA metric return shape. scalar returns only the numeric "
                        "score. rich returns ScoreWithFeedback with directional "
                        "error, rank-side, scale-anchor, parse/NA, and reasoning "
                        "feedback for reflection.")
    p.add_argument("--selection-guard", type=str, default="none",
                   choices=["none", "dev"],
                   help="Select final scorer by dev Pearson r. `dev` keeps the "
                        "baseline when optimized dev r regresses. `none` always "
                        "uses the optimized scorer. Test data is not used for "
                        "selection.")
    p.add_argument("--keep-baseline-on-regression", action="store_true",
                   default=False,
                   help="Deprecated diagnostic only. Also evaluate the optimized "
                        "program on the test set and record whether the old "
                        "test-set guard would have triggered. This does not "
                        "select the paper-facing final scorer.")
    p.add_argument("--train-pool", type=str, default="openweight",
                   choices=["openweight", "expert"],
                   help="openweight: train labels from Benoit's LLaMA/DeepSeek/Gemma ensemble on "
                        "non-test manifestos. expert: train labels from data_experts.rda *also* on "
                        "non-test manifestos (much smaller pool).")
    p.add_argument("--dev-frac", type=float, default=0.2)
    p.add_argument("--max-demos", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--reflection-max-tokens", type=int, default=2048,
                   help="Max tokens for GEPA's reflection/proposal LM. Kept "
                        "separate from scorer --max-tokens so scoring can stay "
                        "short while prompt rewrites have enough room.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs") / f"phase1_opt_scorer_{datetime.now(timezone.utc):%Y%m%d_%H%M}")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def _load_test_pairs(dim: PolicyDimension) -> pd.DataFrame:
    """Held-out test set: 235 Benoit expert manifestos with expert_mean labels."""
    summaries = load_benoit_masked_summaries(dimension=dim)
    experts = load_benoit_expert_means(dim)
    experts_lookup = {
        str(r.manifesto).removesuffix(".txt"): float(r.expert_mean)
        for r in experts.itertuples()
    }
    summaries["label"] = summaries["manifesto_stem"].map(experts_lookup)
    return summaries.dropna(subset=["label"]).reset_index(drop=True)


def _load_train_pairs(dim: PolicyDimension, pool: str, test_keys: set[str]) -> pd.DataFrame:
    """Training pool: summaries with labels from a Benoit-disjoint source."""
    summaries = load_benoit_masked_summaries(dimension=dim)
    summaries = summaries[~summaries["manifesto_stem"].isin(test_keys)].copy()
    if pool == "openweight":
        # Benoit's LLaMA/DeepSeek/Gemma ensemble mean per (manifesto, dim)
        scores = load_benoit_llm_scores(kind="openweight", dimension=dim)
        ensemble = benoit_ensemble_mean(scores)
        ensemble["manifesto_stem"] = ensemble["manifesto"].astype(str).str.removesuffix(".txt")
        lookup = dict(zip(ensemble["manifesto_stem"], ensemble["score_llm_mean"]))
    elif pool == "expert":
        experts = load_benoit_expert_means(dim)
        lookup = {
            str(r.manifesto).removesuffix(".txt"): float(r.expert_mean)
            for r in experts.itertuples()
        }
    else:
        raise ValueError(pool)
    summaries["label"] = summaries["manifesto_stem"].map(lookup)
    return summaries.dropna(subset=["label"]).reset_index(drop=True)


def _make_examples(rows: pd.DataFrame) -> List[dspy.Example]:
    out = []
    for r in rows.itertuples():
        out.append(
            dspy.Example(summary=r.summary, expert_mean=float(r.label)).with_inputs("summary")
        )
    return out


_NA_STRINGS = {"na", "n/a", "none", "null", "unknown", ""}


def _prediction_raw_score(prediction: Any) -> Any:
    if isinstance(prediction, dict):
        return prediction.get("score")
    return getattr(prediction, "score", None)


def _prediction_reasoning(prediction: Any) -> str:
    if isinstance(prediction, dict):
        return str(prediction.get("reasoning", "") or "")
    return str(getattr(prediction, "reasoning", "") or "")


def _parse_prediction_score(prediction: Any) -> Optional[float]:
    raw = _prediction_raw_score(prediction)
    if raw is None or str(raw).strip().lower() in _NA_STRINGS:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _truncate(text: str, n: int = 800) -> str:
    text = str(text or "")
    return text if len(text) <= n else text[: n - 3] + "..."


def _make_metric(
    scale_range: float,
    mode: str = "mae",
    *,
    label_center: Optional[float] = None,
) -> Callable:
    """Per-example score in [0, 1]; higher is better.

    Modes:
    - "mae" (default): 1 - |pred - gold|/scale_range. Rewards level accuracy.
    - "rank": 1 if (pred - center) and (gold - center) share sign, else 0,
      with a small MAE-style tiebreak. Rewards rank-preservation, which is
      what the test-set Pearson r metric actually measures. ``label_center``
      should be the dataset mean/median of the training labels.
    """
    if mode not in {"mae", "rank"}:
        raise ValueError(f"unknown metric mode: {mode}")
    def metric(example, prediction, trace=None):
        pred_val = _parse_prediction_score(prediction)
        if pred_val is None:
            return 0.0
        err = abs(pred_val - example.expert_mean) / scale_range
        mae_score = max(0.0, 1.0 - err)
        if mode == "mae":
            return mae_score
        # mode == "rank"
        if label_center is None:
            return mae_score  # safety: behave like mae if center wasn't passed
        pred_side = 1 if pred_val >= label_center else -1
        gold_side = 1 if example.expert_mean >= label_center else -1
        # Concordant pair → 1.0 (with tiny mae tiebreak); discordant → 0.0
        # (with the same tiebreak so GEPA's Pareto sees something monotonic).
        same_side = (pred_side == gold_side)
        return (0.85 if same_side else 0.0) + 0.15 * mae_score
    return metric


def _rank_side(value: float, center: Optional[float]) -> str:
    if center is None:
        return "n/a"
    return "high" if float(value) >= float(center) else "low"


def _rich_feedback(
    *,
    spec,
    example,
    prediction,
    score: float,
    mode: str,
    label_center: Optional[float],
) -> str:
    raw = _prediction_raw_score(prediction)
    reasoning = _truncate(_prediction_reasoning(prediction), 800)
    target = float(example.expert_mean)
    anchor_low = f"{spec.scale.min_value:g} = {spec.anchor_low}"
    anchor_high = f"{spec.scale.max_value:g} = {spec.anchor_high}"
    pred_val = _parse_prediction_score(prediction)
    if pred_val is None:
        return (
            "Output failed to parse as a Benoit 1-7 score or returned NA.\n"
            f"Raw score: {raw!r}\n"
            f"Target score: {target:.3f}\n"
            f"Dimension: {spec.dimension.value}\n"
            f"Scale anchors: {anchor_low}; {anchor_high}; neutral = 4.\n"
            "Return a numeric score on the 1-7 scale unless the summary truly "
            "contains no relevant evidence.\n"
            f"Reasoning was:\n{reasoning or '(empty)'}"
        )

    err = abs(pred_val - target)
    if pred_val < target:
        direction = (
            f"score higher, toward `{spec.anchor_high}`"
        )
    elif pred_val > target:
        direction = (
            f"score lower, toward `{spec.anchor_low}`"
        )
    else:
        direction = "score is exactly on target"

    rank_line = "Rank-side feedback: rank mode is inactive."
    if label_center is not None:
        pred_side = _rank_side(pred_val, label_center)
        gold_side = _rank_side(target, label_center)
        if pred_side == gold_side:
            rank_line = (
                "Rank-side feedback: prediction and target are on the same "
                f"side of center {label_center:.3f} ({pred_side})."
            )
        else:
            rank_line = (
                "Rank-side mismatch: prediction is on the "
                f"{pred_side} side of center {label_center:.3f}, but target "
                f"is on the {gold_side} side. Preserve ordering relative to "
                "the corpus center."
            )

    return (
        f"Predicted {pred_val:.3f} vs target {target:.3f} "
        f"(abs error {err:.3f}, metric score {float(score):.3f}, mode={mode}).\n"
        f"Correction: {direction}.\n"
        f"{rank_line}\n"
        f"Dimension: {spec.dimension.value}.\n"
        f"Scale anchors: {anchor_low}; {anchor_high}; neutral = 4.\n"
        "Use the summary evidence, not generic party stereotypes. If the "
        "summary contains mixed cues, weight the dimension-specific evidence "
        "that best matches the anchors.\n"
        f"Reasoning was:\n{reasoning or '(empty)'}"
    )


def _make_gepa_metric(
    spec,
    *,
    mode: str = "mae",
    label_center: Optional[float] = None,
    feedback_mode: str = "rich",
) -> Callable:
    scalar_metric = _make_metric(
        spec.scale.range,
        mode=mode,
        label_center=label_center,
    )

    if feedback_mode == "scalar":
        def scalar_gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            del pred_name, pred_trace
            return scalar_metric(gold, pred, trace=trace)
        return scalar_gepa_metric

    if feedback_mode != "rich":
        raise ValueError(f"unknown feedback mode: {feedback_mode}")

    from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

    def rich_gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        del trace, pred_name, pred_trace
        score = float(scalar_metric(gold, pred))
        feedback = _rich_feedback(
            spec=spec,
            example=gold,
            prediction=pred,
            score=score,
            mode=mode,
            label_center=label_center,
        )
        return ScoreWithFeedback(score=score, feedback=feedback)

    return rich_gepa_metric


def _predict_and_correlate(
    program: dspy.Module,
    examples: List[dspy.Example],
) -> dict:
    preds, truths = [], []
    for ex in examples:
        try:
            pred = program(summary=ex.summary)
            val = _parse_prediction_score(pred)
        except Exception:  # noqa: BLE001
            val = None
        preds.append(val)
        truths.append(ex.expert_mean)
    report = compute_corpus_pearson_r(preds, truths)
    return {
        "n_total": len(examples),
        "pearson": report.as_dict(),
        "preds": preds,
        "truths": truths,
    }


def _save_scorer_artifacts(
    output_dir: Path,
    *,
    final_scorer: dspy.Module,
    optimized_scorer: Optional[dspy.Module] = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for label, scorer, filename in (
        ("final", final_scorer, "scorer_final.json"),
        ("optimized", optimized_scorer, "optimized_scorer.json"),
    ):
        if scorer is None:
            continue
        path = output_dir / filename
        try:
            scorer.save(str(path))
            saved[label] = str(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save %s scorer to %s: %s", label, path, exc)
    return saved


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dim = _DIM_FROM_NAME[args.dimension]
    spec = BENOIT_DIMENSIONS[dim]

    logger.info("Loading Benoit test set (expert-benchmarked) for %s", dim.value)
    test_df = _load_test_pairs(dim)
    test_keys = set(test_df["manifesto_stem"])
    logger.info("Test set size: %d (expert-benchmark held-out)", len(test_df))

    logger.info("Loading training pool (%s, Benoit-disjoint) for %s", args.train_pool, dim.value)
    train_full = _load_train_pairs(dim, args.train_pool, test_keys)
    logger.info("Train+dev pool size: %d", len(train_full))
    if len(train_full) < 20:
        raise SystemExit(f"Train pool too small ({len(train_full)}); pick a different --train-pool.")

    shuffled = train_full.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_dev = max(5, int(round(len(shuffled) * args.dev_frac)))
    dev_df = shuffled.iloc[:n_dev]
    train_df = shuffled.iloc[n_dev:]
    logger.info("Split: train=%d dev=%d test=%d (test = held-out Benoit experts)",
                len(train_df), len(dev_df), len(test_df))

    trainset = _make_examples(train_df)
    devset = _make_examples(dev_df)
    testset = _make_examples(test_df)

    logger.info("Configuring LM on port %d (T=%g)", args.port, args.temperature)
    lm = create_vllm_lm(port=args.port, model=args.model, temperature=args.temperature,
                       max_tokens=args.max_tokens, cache=True)
    configure_dspy(lm=lm)
    reflection_lm = create_vllm_lm(
        port=args.port,
        model=args.model,
        temperature=0.7,
        max_tokens=args.reflection_max_tokens,
        cache=True,
    )

    baseline = DimensionScorer(spec, use_cot=False)

    logger.info("Evaluating baseline on dev set (n=%d)", len(devset))
    t0 = time.time()
    baseline_dev_result = _predict_and_correlate(baseline, devset)
    baseline_dev_elapsed = time.time() - t0
    logger.info(
        "Baseline dev r=%+.3f (n=%d, CI [%+.3f, %+.3f], %.1fs)",
        baseline_dev_result["pearson"]["pearson_r"],
        baseline_dev_result["pearson"]["n"],
        baseline_dev_result["pearson"]["pearson_ci_low"],
        baseline_dev_result["pearson"]["pearson_ci_high"],
        baseline_dev_elapsed,
    )

    logger.info("Evaluating baseline on test set (n=%d)", len(testset))
    t0 = time.time()
    baseline_result = _predict_and_correlate(baseline, testset)
    baseline_elapsed = time.time() - t0
    logger.info(
        "Baseline test r=%+.3f (n=%d, CI [%+.3f, %+.3f], %.1fs)",
        baseline_result["pearson"]["pearson_r"],
        baseline_result["pearson"]["n"],
        baseline_result["pearson"]["pearson_ci_low"],
        baseline_result["pearson"]["pearson_ci_high"],
        baseline_elapsed,
    )

    report = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": getattr(lm, "model", "unknown"),
            "dimension": dim.value,
            "optimizer": args.optimizer,
            "train_pool": args.train_pool,
            "seed": args.seed,
            "n_train": len(trainset), "n_dev": len(devset), "n_test": len(testset),
            "max_demos": args.max_demos,
            "metric_mode": args.metric_mode,
            "feedback_mode": args.feedback_mode,
            "selection_guard": args.selection_guard,
            "gepa_valset_cap": args.gepa_valset_cap,
            "gepa_max_metric_calls": args.gepa_max_metric_calls,
            "reflection_max_tokens": args.reflection_max_tokens,
        },
        "baseline_dev": baseline_dev_result["pearson"],
        "baseline_dev_time_seconds": round(baseline_dev_elapsed, 1),
        "baseline_test": baseline_result["pearson"],
        "baseline_test_time_seconds": round(baseline_elapsed, 1),
    }

    if args.optimizer == "none":
        logger.info("--optimizer=none; skipping compile step.")
        report["final_source"] = "baseline"
        report["dev_selection_guard_triggered"] = False
        report["baseline_guard_triggered"] = False
        report["baseline_guard_note"] = (
            "Deprecated test-set guard field. Final selection is controlled "
            "by dev_selection_guard_triggered; test-set guard diagnostics "
            "are reported under legacy_test_guard_diagnostic."
        )
        report["final_test"] = baseline_result["pearson"]
        report["legacy_test_guard_diagnostic"] = {"enabled": False}
        final_scorer = baseline
        optimized_scorer = None
    else:
        # Compute label center if rank mode requested. Use the train-pool
        # mean so it's consistent across the metric calls regardless of
        # what subsample GEPA happens to pick.
        label_center = None
        if args.metric_mode == "rank":
            train_labels = [float(ex.expert_mean) for ex in trainset]
            label_center = sum(train_labels) / len(train_labels)
            logger.info(
                "metric=rank; train-pool label center=%.3f (n=%d)",
                label_center, len(train_labels),
            )
        metric = _make_metric(spec.scale.range, mode=args.metric_mode, label_center=label_center)
        if args.optimizer == "bootstrap":
            compiler = dspy.BootstrapFewShot(
                metric=metric, max_bootstrapped_demos=args.max_demos,
                max_labeled_demos=args.max_demos,
            )
        elif args.optimizer == "miprov2":
            compiler = dspy.MIPROv2(
                metric=metric, auto="light",
                num_threads=4,
            )
        elif args.optimizer == "gepa":
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
        else:
            raise ValueError(args.optimizer)

        logger.info("Compiling with optimizer=%s", args.optimizer)
        t0 = time.time()
        if args.optimizer == "gepa":
            gepa_valset = list(devset[:args.gepa_valset_cap]) if args.gepa_valset_cap > 0 else devset
            optimized = compiler.compile(student=baseline, trainset=trainset, valset=gepa_valset)
        elif args.optimizer == "miprov2":
            optimized = compiler.compile(baseline, trainset=trainset, valset=devset)
        else:
            optimized = compiler.compile(baseline, trainset=trainset)
        compile_elapsed = time.time() - t0
        logger.info("Compile done in %.1fs. Evaluating optimized on dev set.", compile_elapsed)

        t0 = time.time()
        opt_dev_result = _predict_and_correlate(optimized, devset)
        opt_dev_elapsed = time.time() - t0
        logger.info(
            "Optimized dev r=%+.3f (n=%d, CI [%+.3f, %+.3f], %.1fs)",
            opt_dev_result["pearson"]["pearson_r"],
            opt_dev_result["pearson"]["n"],
            opt_dev_result["pearson"]["pearson_ci_low"],
            opt_dev_result["pearson"]["pearson_ci_high"],
            opt_dev_elapsed,
        )
        report["optimized_dev"] = opt_dev_result["pearson"]
        report["compile_time_seconds"] = round(compile_elapsed, 1)
        report["optimized_dev_time_seconds"] = round(opt_dev_elapsed, 1)

        opt_dev_r = opt_dev_result["pearson"].get("pearson_r")
        base_dev_r = baseline_dev_result["pearson"].get("pearson_r")
        if (
            args.selection_guard == "dev"
            and opt_dev_r is not None and base_dev_r is not None
            and opt_dev_r < base_dev_r
        ):
            logger.warning(
                "Dev selection guard: optimized dev r=%+.3f < baseline dev r=%+.3f. "
                "Saving baseline as final program.",
                opt_dev_r, base_dev_r,
            )
            saved = baseline
            final_source = "baseline"
            report["dev_selection_guard_triggered"] = True
        else:
            saved = optimized
            final_source = "optimized"
            report["dev_selection_guard_triggered"] = False

        report["baseline_guard_triggered"] = False
        report["baseline_guard_note"] = (
            "Deprecated test-set guard field. Final selection is controlled "
            "by dev_selection_guard_triggered; test-set guard diagnostics "
            "are reported under legacy_test_guard_diagnostic."
        )
        report["final_source"] = final_source

        if final_source == "baseline":
            report["final_test"] = baseline_result["pearson"]
            report["final_test_time_seconds"] = 0.0
        else:
            logger.info("Evaluating dev-selected optimized scorer on test set.")
            t0 = time.time()
            opt_test_result = _predict_and_correlate(optimized, testset)
            opt_test_elapsed = time.time() - t0
            report["optimized_test"] = opt_test_result["pearson"]
            report["optimized_test_time_seconds"] = round(opt_test_elapsed, 1)
            report["final_test"] = opt_test_result["pearson"]
            report["final_test_time_seconds"] = round(opt_test_elapsed, 1)

        if args.keep_baseline_on_regression:
            if final_source == "optimized":
                legacy_opt_test = report["final_test"]
            else:
                logger.info("Running deprecated diagnostic optimized-test evaluation.")
                t0 = time.time()
                legacy_eval = _predict_and_correlate(optimized, testset)
                report["optimized_test"] = legacy_eval["pearson"]
                report["optimized_test_time_seconds"] = round(time.time() - t0, 1)
                legacy_opt_test = legacy_eval["pearson"]
            legacy_opt_r = legacy_opt_test.get("pearson_r")
            legacy_base_r = baseline_result["pearson"].get("pearson_r")
            report["legacy_test_guard_diagnostic"] = {
                "enabled": True,
                "would_trigger": bool(
                    legacy_opt_r is not None
                    and legacy_base_r is not None
                    and legacy_opt_r < legacy_base_r
                ),
                "optimized_test_r": legacy_opt_r,
                "baseline_test_r": legacy_base_r,
                "note": (
                    "Diagnostic only. Test-set performance is not used for "
                    "paper-facing final scorer selection."
                ),
            }
        else:
            report["legacy_test_guard_diagnostic"] = {"enabled": False}

        final_scorer = saved
        optimized_scorer = optimized

    report["artifacts"] = _save_scorer_artifacts(
        args.output_dir,
        final_scorer=final_scorer,
        optimized_scorer=optimized_scorer,
    )

    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    logger.info("Outputs in %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
