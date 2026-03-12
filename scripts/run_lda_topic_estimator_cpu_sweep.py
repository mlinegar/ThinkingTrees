#!/usr/bin/env python3
"""CPU sweep comparing LDA topic estimators across simulation pipelines.

This script runs a compact comparison over train-doc budgets for:
- tensor_lda
- online_tensor_lda
- neural_hybrid
- neural_embedding_hybrid

and writes raw rows (CSV) plus grouped summaries (JSON).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.segment_lda_ops_weight_recovery_simulation import (  # noqa: E402
    SegmentLDAOpsWeightRecoveryConfig,
    run_segment_lda_ops_weight_recovery_experiment,
)
from src.tree.segmented_lda_ctreepo_simulation import (  # noqa: E402
    SegmentedLDACtreePOConfig,
    run_segmented_lda_ctreepo_simulation,
)


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).replace(" ", ",").split(",") if x.strip()]


def _parse_str_list(text: str) -> List[str]:
    return [str(x.strip()) for x in str(text).replace(" ", ",").split(",") if x.strip()]


def _effective_neural_base(estimator: str, default_base: str) -> str:
    est = str(estimator).strip().lower()
    if est == "neural_embedding_hybrid":
        return "embedding_spectral"
    if est.startswith("neural_"):
        return str(default_base).strip().lower()
    return ""


def _fmt_float(x: float) -> str:
    if x != x:
        return "nan"
    return f"{float(x):.6g}"


def _run_segmented(
    *,
    estimator: str,
    train_docs: int,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    neural_base = _effective_neural_base(estimator, args.neural_topic_base_estimator)
    cfg = SegmentedLDACtreePOConfig(
        n_topics=int(args.n_topics),
        vocab_size=int(args.vocab_size),
        alpha_topic=float(args.alpha_topic),
        beta_word=float(args.beta_word),
        n_books_train=int(train_docs),
        n_books_test=int(args.segmented_test_docs),
        min_segments=int(args.min_segments),
        max_segments=int(args.max_segments),
        min_seg_tokens=int(args.min_seg_tokens),
        max_seg_tokens=int(args.max_seg_tokens),
        segment_concentration=float(args.segment_concentration),
        segment_background=float(args.segment_background),
        fixed_leaf_tokens=int(args.fixed_leaf_tokens),
        topic_phi_estimator=str(estimator),
        topic_phi_docs=int(args.topic_phi_docs),
        tlda_delta=float(args.tlda_delta),
        tlda_rate_constant=float(args.tlda_rate_constant),
        tlda_sigmaK_floor=float(args.tlda_sigmaK_floor),
        topic_phi_permute=bool(args.topic_phi_permute),
        online_tensor_lda_burn_in_docs=int(args.online_tensor_lda_burn_in_docs),
        online_tensor_lda_batch_docs=int(args.online_tensor_lda_batch_docs),
        online_tensor_lda_passes=int(args.online_tensor_lda_passes),
        online_tensor_lda_lr=float(args.online_tensor_lda_lr),
        online_tensor_lda_grad_clip_norm=float(args.online_tensor_lda_grad_clip_norm),
        embedding_topic_svd_dim_extra=int(args.embedding_topic_svd_dim_extra),
        embedding_topic_kmeans_inits=int(args.embedding_topic_kmeans_inits),
        embedding_topic_kmeans_max_iter=int(args.embedding_topic_kmeans_max_iter),
        embedding_topic_assignment_temperature=float(args.embedding_topic_assignment_temperature),
        embedding_topic_ppmi_shift=float(args.embedding_topic_ppmi_shift),
        neural_topic_base_estimator=(str(neural_base) if neural_base else str(args.neural_topic_base_estimator)),
        neural_topic_seed_fraction=float(args.neural_topic_seed_fraction),
        neural_topic_hidden_dim=int(args.neural_topic_hidden_dim),
        neural_topic_steps=int(args.neural_topic_steps),
        neural_topic_lr=float(args.neural_topic_lr),
        neural_topic_weight_decay=float(args.neural_topic_weight_decay),
        neural_topic_mix_samples=int(args.neural_topic_mix_samples),
        neural_topic_mix_temperature=float(args.neural_topic_mix_temperature),
        neural_topic_operator_boost=float(args.neural_topic_operator_boost),
        neural_topic_seed_llm_min_weight=float(args.neural_topic_seed_llm_min_weight),
        neural_topic_seed_llm_max_weight=float(args.neural_topic_seed_llm_max_weight),
        neural_topic_similarity_temperature=float(args.neural_topic_similarity_temperature),
        neural_topic_ridge=float(args.neural_topic_ridge),
        calibration_leaf_query_rate=float(args.calibration_leaf_query_rate),
        eval_leaf_query_rate=float(args.eval_leaf_query_rate),
        eval_internal_query_rate=float(args.eval_internal_query_rate),
        eval_internal_query_design=str(args.eval_internal_query_design),
        seed=int(seed),
    )
    t0 = time.perf_counter()
    out = run_segmented_lda_ctreepo_simulation(cfg)
    dt = float(time.perf_counter() - t0)

    budgeted = out.metrics.get("estimated_calibrated_budgeted")
    row: Dict[str, object] = {
        "pipeline": "segmented",
        "estimator": str(estimator),
        "neural_base_effective": str(_effective_neural_base(estimator, args.neural_topic_base_estimator)),
        "train_docs": int(train_docs),
        "seed": int(seed),
        "runtime_sec": dt,
        "topic_phi_l2_error_mean": float(out.topic_meta.get("topic_phi_l2_error_mean", float("nan"))),
        "root_l1_mean": float(budgeted.root_l1_mean) if budgeted is not None else float("nan"),
        "c1_violation_rate": float(budgeted.c1_violation_rate) if budgeted is not None else float("nan"),
        "c3_violation_rate": float(budgeted.c3_violation_rate) if budgeted is not None else float("nan"),
        "mean_total_queries": float(budgeted.mean_total_queries) if budgeted is not None else float("nan"),
        "decomposition_total_root_l1_mean": float(out.decomposition.total_root_l1_mean),
        "decomposition_upper_bound_mean": float(out.decomposition.upper_bound_mean),
    }
    return row


def _run_ops(
    *,
    estimator: str,
    train_docs: int,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    neural_base = _effective_neural_base(estimator, args.neural_topic_base_estimator)
    cfg = SegmentLDAOpsWeightRecoveryConfig(
        n_topics=int(args.n_topics),
        vocab_size=int(args.vocab_size),
        min_tokens=int(args.ops_min_tokens),
        max_tokens=int(args.ops_max_tokens),
        min_segments=int(args.min_segments),
        max_segments=int(args.max_segments),
        min_seg_len=int(args.ops_min_seg_len),
        max_seg_len=int(args.ops_max_seg_len),
        leaf_tokens=int(args.ops_leaf_tokens),
        align_segments_to_leaves=bool(args.ops_align_segments_to_leaves),
        doc_topic_concentration=float(args.doc_topic_concentration),
        topic_process=str(args.ops_topic_process),
        boundary_profile=str(args.ops_boundary_profile),
        boundary_profile_strength=float(args.ops_boundary_profile_strength),
        boundary_profile_seed=int(args.ops_boundary_profile_seed),
        segment_length_power=float(args.ops_segment_length_power),
        topic_concentration=float(args.topic_concentration),
        emission_mode=str(args.ops_emission_mode),
        anchor_words_per_topic=int(args.ops_anchor_words_per_topic),
        anchor_multiplier=float(args.ops_anchor_multiplier),
        relevant_topics=int(args.ops_relevant_topics),
        theta_scale=float(args.ops_theta_scale),
        zero_diagonal=bool(args.ops_zero_diagonal),
        lambda_multiplier=float(args.ops_lambda_multiplier),
        oracle_noise_std=float(args.ops_oracle_noise_std),
        audit_policy=str(args.ops_audit_policy),
        audit_fraction=float(args.ops_audit_fraction),
        audit_strategy=str(args.ops_audit_strategy),
        ridge_lambda=float(args.ops_ridge_lambda),
        topic_source=str(args.ops_topic_source),
        feature_inference=str(args.ops_feature_inference),
        topic_phi_estimator=str(estimator),
        topic_phi_docs=int(args.topic_phi_docs),
        tlda_delta=float(args.tlda_delta),
        tlda_rate_constant=float(args.tlda_rate_constant),
        tlda_sigmaK_floor=float(args.tlda_sigmaK_floor),
        topic_phi_permute=bool(args.topic_phi_permute),
        online_tensor_lda_burn_in_docs=int(args.online_tensor_lda_burn_in_docs),
        online_tensor_lda_batch_docs=int(args.online_tensor_lda_batch_docs),
        online_tensor_lda_passes=int(args.online_tensor_lda_passes),
        online_tensor_lda_lr=float(args.online_tensor_lda_lr),
        online_tensor_lda_grad_clip_norm=float(args.online_tensor_lda_grad_clip_norm),
        embedding_topic_svd_dim_extra=int(args.embedding_topic_svd_dim_extra),
        embedding_topic_kmeans_inits=int(args.embedding_topic_kmeans_inits),
        embedding_topic_kmeans_max_iter=int(args.embedding_topic_kmeans_max_iter),
        embedding_topic_assignment_temperature=float(args.embedding_topic_assignment_temperature),
        embedding_topic_ppmi_shift=float(args.embedding_topic_ppmi_shift),
        neural_topic_base_estimator=(str(neural_base) if neural_base else str(args.neural_topic_base_estimator)),
        neural_topic_seed_fraction=float(args.neural_topic_seed_fraction),
        neural_topic_hidden_dim=int(args.neural_topic_hidden_dim),
        neural_topic_steps=int(args.neural_topic_steps),
        neural_topic_lr=float(args.neural_topic_lr),
        neural_topic_weight_decay=float(args.neural_topic_weight_decay),
        neural_topic_mix_samples=int(args.neural_topic_mix_samples),
        neural_topic_mix_temperature=float(args.neural_topic_mix_temperature),
        neural_topic_operator_boost=float(args.neural_topic_operator_boost),
        neural_topic_seed_llm_min_weight=float(args.neural_topic_seed_llm_min_weight),
        neural_topic_seed_llm_max_weight=float(args.neural_topic_seed_llm_max_weight),
        neural_topic_similarity_temperature=float(args.neural_topic_similarity_temperature),
        neural_topic_ridge=float(args.neural_topic_ridge),
        run_all_feature_modes=bool(args.ops_run_all_feature_modes),
        violation_tau=float(args.ops_violation_tau),
        train_docs=int(train_docs),
        test_docs=int(args.ops_test_docs),
        seed=int(seed),
    )
    t0 = time.perf_counter()
    out = run_segment_lda_ops_weight_recovery_experiment(cfg)
    dt = float(time.perf_counter() - t0)

    ridge = out.metrics.get("ridge", {}) if isinstance(out.metrics, dict) else {}
    row = {
        "pipeline": "ops",
        "estimator": str(estimator),
        "neural_base_effective": str(_effective_neural_base(estimator, args.neural_topic_base_estimator)),
        "train_docs": int(train_docs),
        "seed": int(seed),
        "runtime_sec": dt,
        "topic_phi_l2_error_mean": float(out.topic_meta.get("topic_phi_l2_error_mean", float("nan"))),
        "root_l1_mean": float(ridge.get("root_mae", float("nan"))),
        "c1_violation_rate": float(ridge.get("leaf_violation_rate", float("nan"))),
        "c3_violation_rate": float(ridge.get("merge_violation_rate", float("nan"))),
        "mean_total_queries": float(ridge.get("oracle_queries_total", float("nan"))),
        "decomposition_total_root_l1_mean": float("nan"),
        "decomposition_upper_bound_mean": float("nan"),
    }
    return row


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k in seen:
                continue
            seen.add(k)
            keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})


def _aggregate(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("pipeline", "")), str(row.get("estimator", "")), int(row.get("train_docs", 0)))
        grouped[key].append(dict(row))

    out: List[Dict[str, object]] = []
    metric_keys = [
        "runtime_sec",
        "topic_phi_l2_error_mean",
        "root_l1_mean",
        "c1_violation_rate",
        "c3_violation_rate",
        "mean_total_queries",
    ]
    for (pipeline, estimator, train_docs), rows_g in sorted(grouped.items()):
        row: Dict[str, object] = {
            "pipeline": pipeline,
            "estimator": estimator,
            "train_docs": int(train_docs),
            "n_runs": int(len(rows_g)),
        }
        for mk in metric_keys:
            vals: List[float] = []
            for r in rows_g:
                x = r.get(mk)
                try:
                    xf = float(x)
                except Exception:
                    continue
                if xf == xf and abs(xf) < float("inf"):
                    vals.append(xf)
            row[f"{mk}_mean"] = float(sum(vals) / len(vals)) if vals else float("nan")
        out.append(row)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPU sweep for LDA topic estimators across segmented/ops simulators.")
    p.add_argument("--pipelines", type=str, default="segmented,ops", help="Comma list: segmented,ops")
    p.add_argument(
        "--estimators",
        type=str,
        default="tensor_lda,online_tensor_lda,neural_hybrid,neural_embedding_hybrid",
    )
    p.add_argument("--train-docs-grid", type=str, default="64,128,256")
    p.add_argument("--seed-grid", type=str, default="0,1")

    p.add_argument("--n-topics", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=120)
    p.add_argument("--topic-phi-docs", type=int, default=0)
    p.add_argument("--topic-phi-permute", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--alpha-topic", type=float, default=0.2)
    p.add_argument("--beta-word", type=float, default=0.1)
    p.add_argument("--doc-topic-concentration", type=float, default=0.6)
    p.add_argument("--topic-concentration", type=float, default=0.2)

    p.add_argument("--tlda-delta", type=float, default=0.10)
    p.add_argument("--tlda-rate-constant", type=float, default=1.0)
    p.add_argument("--tlda-sigmaK-floor", type=float, default=1e-6)
    p.add_argument("--online-tensor-lda-burn-in-docs", type=int, default=0)
    p.add_argument("--online-tensor-lda-batch-docs", type=int, default=32)
    p.add_argument("--online-tensor-lda-passes", type=int, default=1)
    p.add_argument("--online-tensor-lda-lr", type=float, default=0.1)
    p.add_argument("--online-tensor-lda-grad-clip-norm", type=float, default=1.0)

    p.add_argument("--embedding-topic-svd-dim-extra", type=int, default=4)
    p.add_argument("--embedding-topic-kmeans-inits", type=int, default=8)
    p.add_argument("--embedding-topic-kmeans-max-iter", type=int, default=80)
    p.add_argument("--embedding-topic-assignment-temperature", type=float, default=0.35)
    p.add_argument("--embedding-topic-ppmi-shift", type=float, default=1.0)

    p.add_argument("--neural-topic-base-estimator", type=str, default="tensor_lda")
    p.add_argument("--neural-topic-seed-fraction", type=float, default=0.35)
    p.add_argument("--neural-topic-hidden-dim", type=int, default=48)
    p.add_argument("--neural-topic-steps", type=int, default=60)
    p.add_argument("--neural-topic-lr", type=float, default=3e-3)
    p.add_argument("--neural-topic-weight-decay", type=float, default=1e-4)
    p.add_argument("--neural-topic-mix-samples", type=int, default=128)
    p.add_argument("--neural-topic-mix-temperature", type=float, default=1.0)
    p.add_argument("--neural-topic-operator-boost", type=float, default=1.4)
    p.add_argument("--neural-topic-seed-llm-min-weight", type=float, default=0.2)
    p.add_argument("--neural-topic-seed-llm-max-weight", type=float, default=0.55)
    p.add_argument("--neural-topic-similarity-temperature", type=float, default=0.15)
    p.add_argument("--neural-topic-ridge", type=float, default=1e-3)

    p.add_argument("--min-segments", type=int, default=5)
    p.add_argument("--max-segments", type=int, default=8)
    p.add_argument("--segment-concentration", type=float, default=80.0)
    p.add_argument("--segment-background", type=float, default=2.0)

    p.add_argument("--segmented-test-docs", type=int, default=64)
    p.add_argument("--min-seg-tokens", type=int, default=12)
    p.add_argument("--max-seg-tokens", type=int, default=24)
    p.add_argument("--fixed-leaf-tokens", type=int, default=16)
    p.add_argument("--calibration-leaf-query-rate", type=float, default=0.10)
    p.add_argument("--eval-leaf-query-rate", type=float, default=0.0)
    p.add_argument("--eval-internal-query-rate", type=float, default=0.15)
    p.add_argument("--eval-internal-query-design", choices=["none", "uniform", "risk"], default="risk")

    p.add_argument("--ops-test-docs", type=int, default=64)
    p.add_argument("--ops-min-tokens", type=int, default=192)
    p.add_argument("--ops-max-tokens", type=int, default=192)
    p.add_argument("--ops-min-seg-len", type=int, default=24)
    p.add_argument("--ops-max-seg-len", type=int, default=80)
    p.add_argument("--ops-leaf-tokens", type=int, default=16)
    p.add_argument("--ops-align-segments-to-leaves", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ops-topic-process", choices=["segments", "bag_of_words"], default="segments")
    p.add_argument("--ops-boundary-profile", choices=["uniform", "start", "middle", "end", "bimodal", "random"], default="uniform")
    p.add_argument("--ops-boundary-profile-strength", type=float, default=0.0)
    p.add_argument("--ops-boundary-profile-seed", type=int, default=-1)
    p.add_argument("--ops-segment-length-power", type=float, default=1.0)
    p.add_argument("--ops-emission-mode", choices=["anchored", "disjoint"], default="anchored")
    p.add_argument("--ops-anchor-words-per-topic", type=int, default=4)
    p.add_argument("--ops-anchor-multiplier", type=float, default=10.0)
    p.add_argument("--ops-relevant-topics", type=int, default=2)
    p.add_argument("--ops-theta-scale", type=float, default=1.0)
    p.add_argument("--ops-zero-diagonal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ops-lambda-multiplier", type=float, default=1.0)
    p.add_argument("--ops-oracle-noise-std", type=float, default=0.0)
    p.add_argument("--ops-audit-policy", choices=["all", "fixed", "fraction", "sqrt", "log2"], default="fraction")
    p.add_argument("--ops-audit-fraction", type=float, default=0.2)
    p.add_argument("--ops-audit-strategy", choices=["random", "active_small", "profile"], default="random")
    p.add_argument("--ops-ridge-lambda", type=float, default=1e-3)
    p.add_argument("--ops-topic-source", choices=["true", "infer"], default="infer")
    p.add_argument("--ops-feature-inference", choices=["hard", "soft"], default="hard")
    p.add_argument("--ops-run-all-feature-modes", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ops-violation-tau", type=float, default=0.0)

    p.add_argument("--out-csv", type=str, default="outputs/lda_topic_estimator_cpu_sweep/raw.csv")
    p.add_argument("--out-json", type=str, default="outputs/lda_topic_estimator_cpu_sweep/summary.json")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pipelines = [x.lower() for x in _parse_str_list(args.pipelines)]
    estimators = [x.lower() for x in _parse_str_list(args.estimators)]
    train_docs_grid = _parse_int_list(args.train_docs_grid)
    seed_grid = _parse_int_list(args.seed_grid)

    valid_pipelines = {"segmented", "ops"}
    if not pipelines or any(p not in valid_pipelines for p in pipelines):
        raise ValueError("pipelines must be a comma list of: segmented,ops")
    if not estimators:
        raise ValueError("estimators must be non-empty")
    if not train_docs_grid:
        raise ValueError("train-docs-grid must be non-empty")
    if not seed_grid:
        raise ValueError("seed-grid must be non-empty")

    rows: List[Dict[str, object]] = []
    total = len(pipelines) * len(estimators) * len(train_docs_grid) * len(seed_grid)
    done = 0

    for pipeline in pipelines:
        for estimator in estimators:
            for train_docs in train_docs_grid:
                for seed in seed_grid:
                    if pipeline == "segmented":
                        row = _run_segmented(estimator=estimator, train_docs=int(train_docs), seed=int(seed), args=args)
                    else:
                        row = _run_ops(estimator=estimator, train_docs=int(train_docs), seed=int(seed), args=args)
                    rows.append(row)
                    done += 1
                    if not bool(args.quiet):
                        print(
                            "[{}/{}] pipeline={} est={} train={} seed={} | l2={} | root={} | t={}s".format(
                                done,
                                total,
                                pipeline,
                                estimator,
                                train_docs,
                                seed,
                                _fmt_float(float(row.get("topic_phi_l2_error_mean", float("nan")))),
                                _fmt_float(float(row.get("root_l1_mean", float("nan")))),
                                _fmt_float(float(row.get("runtime_sec", float("nan")))),
                            )
                        )

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    _write_csv(out_csv, rows)

    summary = {
        "config": {
            "pipelines": pipelines,
            "estimators": estimators,
            "train_docs_grid": [int(x) for x in train_docs_grid],
            "seed_grid": [int(x) for x in seed_grid],
        },
        "raw_rows": rows,
        "grouped": _aggregate(rows),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote_csv | {out_csv} | rows={len(rows)}")
    print(f"wrote_json | {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
