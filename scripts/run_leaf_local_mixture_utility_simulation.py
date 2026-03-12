#!/usr/bin/env python3
"""Run the Stage-2 leaf-local-mixture utility simulation."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
from pathlib import Path
import sys
from typing import List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.core.leaf_local_mixture_utility import (  # noqa: E402
    LeafLocalMixtureUtilityConfig,
    VALID_ANALYSIS_PARTITION_MODES,
    VALID_BUDGET_REGIMES,
    VALID_LATENT_LENGTH_PROFILES,
    VALID_LATENT_PARTITION_MODES,
    VALID_LOCAL_LAW_MODES,
    VALID_LAW_INTERNAL_QUERY_DESIGNS,
    VALID_PROPENSITY_PROXIES,
    VALID_QUERY_DESIGNS,
    run_leaf_local_mixture_utility_experiment,
)


def _parse_fraction(text: str) -> float:
    raw = str(text).strip()
    if not raw:
        raise ValueError("fraction must be non-empty")
    if "/" in raw:
        return float(Fraction(raw))
    return float(raw)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the Stage-2 leaf-local-mixture utility simulation."
    )
    p.add_argument("--n-topics", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--doc-tokens", type=int, default=384)
    p.add_argument("--doc-topic-concentration", type=float, default=0.6)

    p.add_argument("--topic-concentration", type=float, default=0.2)
    p.add_argument("--emission-mode", type=str, default="anchored")
    p.add_argument("--anchor-words-per-topic", type=int, default=20)
    p.add_argument("--anchor-multiplier", type=float, default=25.0)

    p.add_argument("--utility-dim", type=int, default=16)
    p.add_argument("--utility-design", type=str, default="topic_anchored_sparse")
    p.add_argument("--atomic-block-tokens", type=int, default=16)
    p.add_argument("--latent-leaf-tokens", type=int, default=16)
    p.add_argument(
        "--latent-partition-mode",
        type=str,
        choices=list(VALID_LATENT_PARTITION_MODES),
        default="equal",
    )
    p.add_argument(
        "--latent-length-profile",
        type=str,
        choices=list(VALID_LATENT_LENGTH_PROFILES),
        default="equal",
    )
    p.add_argument("--leaf-fraction", type=str, default="1/24")
    p.add_argument(
        "--analysis-partition-mode",
        type=str,
        choices=list(VALID_ANALYSIS_PARTITION_MODES),
        default="aligned",
    )
    p.add_argument("--analysis-leaf-tokens", type=int, default=0)
    p.add_argument("--local-mixture-concentration", type=float, default=1.0)

    p.add_argument("--relevant-topics", type=int, default=2)
    p.add_argument("--theta-scale", type=float, default=1.0)
    p.add_argument("--zero-diagonal", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--lambda-multiplier", type=float, default=1.0)

    p.add_argument("--train-docs", type=int, default=512)
    p.add_argument("--val-docs", type=int, default=0)
    p.add_argument("--test-docs", type=int, default=256)
    p.add_argument(
        "--budget-regime",
        type=str,
        choices=list(VALID_BUDGET_REGIMES),
        default="all_leaves_labeled",
    )
    p.add_argument("--leaf-label-budget", type=float, default=8.0)
    p.add_argument("--ridge-alpha", type=float, default=1e-3)
    p.add_argument("--query-design", type=str, choices=list(VALID_QUERY_DESIGNS), default="uniform")
    p.add_argument("--doc-sample-rate", type=float, default=1.0)
    p.add_argument("--heldout-doc-sample-rate", type=float, default=0.5)
    p.add_argument("--target-query-budget-per-doc", type=float, default=0.0)
    p.add_argument("--propensity-floor", type=float, default=0.10)
    p.add_argument("--propensity-ceiling", type=float, default=0.90)
    p.add_argument(
        "--propensity-proxy",
        type=str,
        choices=list(VALID_PROPENSITY_PROXIES),
        default="l1_deviation",
    )
    p.add_argument("--ipw-stabilized-clip", type=float, default=20.0)
    p.add_argument("--ipw-delta", type=float, default=0.05)
    p.add_argument("--local-law-mode", type=str, choices=list(VALID_LOCAL_LAW_MODES), default="off")
    p.add_argument(
        "--law-package",
        type=str,
        default="all_laws",
        choices=["root_only", "c1_only", "c3_only", "c1c3", "c2_only", "all_laws"],
    )
    p.add_argument(
        "--exact-family",
        type=str,
        default="",
        choices=["", "oracle", "scrambled_topics", "uniform_prior", "adversarial_merge"],
    )
    p.add_argument("--law-leaf-query-rate", type=float, default=0.10)
    p.add_argument("--law-internal-query-rate", type=float, default=0.10)
    p.add_argument(
        "--law-leaf-query-design", type=str, choices=list(VALID_QUERY_DESIGNS), default="uniform"
    )
    p.add_argument(
        "--law-internal-query-design",
        type=str,
        choices=list(VALID_LAW_INTERNAL_QUERY_DESIGNS),
        default="uniform",
    )
    p.add_argument("--law-task-objective-weight", type=float, default=1.0)
    p.add_argument("--law-c1-weight", type=float, default=1.0 / 3.0)
    p.add_argument("--law-c3-weight", type=float, default=1.0 / 3.0)
    p.add_argument("--law-c2-proxy-weight", type=float, default=1.0 / 3.0)
    p.add_argument("--law-calibration-ridge", type=float, default=1e-3)
    p.add_argument("--law-eval-leaf-sample-rate", type=float, default=0.25)
    p.add_argument("--law-eval-internal-sample-rate", type=float, default=0.25)
    p.add_argument("--law-c1-threshold", type=float, default=0.20)
    p.add_argument("--law-c3-threshold", type=float, default=0.20)
    p.add_argument("--law-c2-threshold", type=float, default=0.20)

    p.add_argument("--inference-prior-mass", type=float, default=0.25)
    p.add_argument("--inference-max-iter", type=int, default=200)
    p.add_argument("--inference-tol", type=float, default=1e-9)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-seed-offset", type=int, default=5_000)
    p.add_argument("--test-seed-offset", type=int, default=10_000)
    p.add_argument("--json-summary", type=str, required=True)
    p.add_argument("--csv-summary", type=str, required=True)
    p.add_argument("--artifact-dir", type=str, default="")
    p.add_argument("--suite-role", type=str, default="")
    p.add_argument("--json", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def _rows_from_summary(summary) -> List[dict]:
    cfg = dict(summary.config)
    world = dict(summary.world_stats)
    heterogeneity = dict(summary.heterogeneity)
    stage3 = dict(getattr(summary, "stage3", {}) or {})
    local_law = dict(getattr(summary, "local_law", {}) or {})
    local_law_objective = (
        dict(local_law.get("objective", {}))
        if isinstance(local_law.get("objective", {}), dict)
        else {}
    )
    ipw_eval = (
        dict(stage3.get("ipw_evaluation", {}))
        if isinstance(stage3.get("ipw_evaluation", {}), dict)
        else {}
    )
    target_eval = (
        dict(ipw_eval.get("target", {})) if isinstance(ipw_eval.get("target", {}), dict) else {}
    )
    delta_eval = (
        dict(ipw_eval.get("delta", {})) if isinstance(ipw_eval.get("delta", {}), dict) else {}
    )
    local_law_policy_metrics = (
        dict(local_law.get("policy_metrics", {}))
        if isinstance(local_law.get("policy_metrics", {}), dict)
        else {}
    )
    rows: List[dict] = []
    methods = summary.methods if isinstance(summary.methods, dict) else {}
    for method, metrics in methods.items():
        if not isinstance(metrics, dict):
            continue
        row = {
            "family": str(summary.family),
            "target_kind": str(summary.target_kind),
            "method": str(method),
            "is_stale_generation": bool(summary.is_stale_generation),
            **{f"cfg_{k}": v for k, v in cfg.items()},
            **{
                f"local_law_objective_{k}": v
                for k, v in local_law_objective.items()
                if not isinstance(v, (dict, list))
            },
            **{f"world_{k}": v for k, v in world.items()},
            **{f"hetero_{k}": v for k, v in heterogeneity.items()},
            **{
                f"stage3_target_{k}": v
                for k, v in target_eval.items()
                if not isinstance(v, (dict, list))
            },
        }
        for group_name in ("local_law_weights", "proxy_weights"):
            group = (
                dict(local_law_objective.get(group_name, {}))
                if isinstance(local_law_objective.get(group_name, {}), dict)
                else {}
            )
            row.update(
                {
                    f"local_law_objective_{group_name}_{k}": v
                    for k, v in group.items()
                    if not isinstance(v, (dict, list))
                }
            )
        row.update(metrics)
        method_delta = (
            dict(delta_eval.get(method, {})) if isinstance(delta_eval.get(method, {}), dict) else {}
        )
        row.update(
            {
                f"stage3_delta_{k}": v
                for k, v in method_delta.items()
                if not isinstance(v, (dict, list))
            }
        )
        if "infer_identity" in local_law_policy_metrics:
            row.update(
                {
                    f"local_law_identity_{k}": v
                    for k, v in dict(local_law_policy_metrics["infer_identity"]).items()
                    if not isinstance(v, (dict, list))
                }
            )
        if "law_calibrated_ipw" in local_law_policy_metrics:
            row.update(
                {
                    f"local_law_ipw_{k}": v
                    for k, v in dict(local_law_policy_metrics["law_calibrated_ipw"]).items()
                    if not isinstance(v, (dict, list))
                }
            )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    json_path = Path(args.json_summary)
    artifact_dir = (
        str(Path(args.artifact_dir))
        if str(args.artifact_dir).strip()
        else str(json_path.parent / f"{json_path.stem}_artifacts")
    )
    cfg = LeafLocalMixtureUtilityConfig(
        n_topics=int(args.n_topics),
        vocab_size=int(args.vocab_size),
        doc_tokens=int(args.doc_tokens),
        doc_topic_concentration=float(args.doc_topic_concentration),
        topic_concentration=float(args.topic_concentration),
        emission_mode=str(args.emission_mode),
        anchor_words_per_topic=int(args.anchor_words_per_topic),
        anchor_multiplier=float(args.anchor_multiplier),
        utility_dim=int(args.utility_dim),
        utility_design=str(args.utility_design),
        atomic_block_tokens=int(args.atomic_block_tokens),
        latent_leaf_tokens=int(args.latent_leaf_tokens),
        latent_partition_mode=str(args.latent_partition_mode),
        latent_length_profile=str(args.latent_length_profile),
        leaf_fraction=float(_parse_fraction(args.leaf_fraction)),
        analysis_partition_mode=str(args.analysis_partition_mode),
        analysis_leaf_tokens=int(args.analysis_leaf_tokens),
        local_mixture_concentration=float(args.local_mixture_concentration),
        relevant_topics=int(args.relevant_topics),
        theta_scale=float(args.theta_scale),
        zero_diagonal=bool(args.zero_diagonal),
        lambda_multiplier=float(args.lambda_multiplier),
        train_docs=int(args.train_docs),
        val_docs=int(args.val_docs),
        test_docs=int(args.test_docs),
        budget_regime=str(args.budget_regime),
        leaf_label_budget=float(args.leaf_label_budget),
        ridge_alpha=float(args.ridge_alpha),
        query_design=str(args.query_design),
        doc_sample_rate=float(args.doc_sample_rate),
        heldout_doc_sample_rate=float(args.heldout_doc_sample_rate),
        target_query_budget_per_doc=float(args.target_query_budget_per_doc),
        propensity_floor=float(args.propensity_floor),
        propensity_ceiling=float(args.propensity_ceiling),
        propensity_proxy=str(args.propensity_proxy),
        ipw_stabilized_clip=float(args.ipw_stabilized_clip),
        ipw_delta=float(args.ipw_delta),
        local_law_mode=str(args.local_law_mode),
        law_package=str(args.law_package),
        exact_family=str(args.exact_family),
        law_leaf_query_rate=float(args.law_leaf_query_rate),
        law_internal_query_rate=float(args.law_internal_query_rate),
        law_leaf_query_design=str(args.law_leaf_query_design),
        law_internal_query_design=str(args.law_internal_query_design),
        law_task_objective_weight=float(args.law_task_objective_weight),
        law_c1_weight=float(args.law_c1_weight),
        law_c3_weight=float(args.law_c3_weight),
        law_c2_proxy_weight=float(args.law_c2_proxy_weight),
        law_calibration_ridge=float(args.law_calibration_ridge),
        law_eval_leaf_sample_rate=float(args.law_eval_leaf_sample_rate),
        law_eval_internal_sample_rate=float(args.law_eval_internal_sample_rate),
        law_c1_threshold=float(args.law_c1_threshold),
        law_c3_threshold=float(args.law_c3_threshold),
        law_c2_threshold=float(args.law_c2_threshold),
        inference_prior_mass=float(args.inference_prior_mass),
        inference_max_iter=int(args.inference_max_iter),
        inference_tol=float(args.inference_tol),
        seed=int(args.seed),
        val_seed_offset=int(args.val_seed_offset),
        test_seed_offset=int(args.test_seed_offset),
        artifact_dir=str(artifact_dir),
        suite_role=str(args.suite_role),
    )
    summary = run_leaf_local_mixture_utility_experiment(cfg)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(summary.to_json(), encoding="utf-8")
    csv_path = Path(args.csv_summary)
    _write_csv(csv_path, _rows_from_summary(summary))

    methods = dict(summary.methods)
    pooled = dict(methods.get("pooled_doc_wrong_model", {}))
    infer = dict(methods.get("leaf_infer_sum", {}))
    analysis = dict(methods.get("analysis_infer_weighted_sum", {}))
    base = dict(methods.get("leaf_ridge_from_u", {}))
    coarse = dict(methods.get("coarse_leaf_ridge_from_u", {}))
    ipw = dict(methods.get("budgeted_leaf_ridge_ipw", {}))
    print(f"wrote_json | {json_path}")
    print(f"wrote_csv | {csv_path}")
    print(
        "pooled_doc_wrong_model | utility_abs_to_true={:.4f}".format(
            float(pooled.get("utility_abs_to_true_mean", float("nan"))),
        )
    )
    print(
        "leaf_infer_sum | utility_abs_to_true={:.4f}".format(
            float(infer.get("utility_abs_to_true_mean", float("nan"))),
        )
    )
    print(
        "analysis_infer_weighted_sum | utility_abs_to_true={:.4f} | delta={:+.4f}".format(
            float(analysis.get("utility_abs_to_true_mean", float("nan"))),
            float(analysis.get("delta_mean", float("nan"))),
        )
    )
    print(
        "leaf_ridge_from_u | utility_abs_to_true={:.4f} | queried_cost/doc={:.2f}".format(
            float(base.get("utility_abs_to_true_mean", float("nan"))),
            float(base.get("mean_queried_cost_train_per_doc", float("nan"))),
        )
    )
    print(
        "coarse_leaf_ridge_from_u | utility_abs_to_true={:.4f} | queried_cost/doc={:.2f}".format(
            float(coarse.get("utility_abs_to_true_mean", float("nan"))),
            float(coarse.get("mean_queried_cost_train_per_doc", float("nan"))),
        )
    )
    print(
        "budgeted_leaf_ridge_ipw | utility_abs_to_true={:.4f} | delta={:+.4f}".format(
            float(ipw.get("utility_abs_to_true_mean", float("nan"))),
            float(ipw.get("delta_mean", float("nan"))),
        )
    )
    if bool(args.json):
        print(summary.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
