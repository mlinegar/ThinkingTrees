#!/usr/bin/env python3
"""Grid runner for segmented-LDA end-to-end C-TreePO simulation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from itertools import product
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.segmented_lda_ctreepo_simulation import (  # noqa: E402
    SegmentedLDACtreePOConfig,
    VALID_TOPIC_PHI_ESTIMATORS,
    run_segmented_lda_ctreepo_simulation,
)


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_str_list(text: str) -> list[str]:
    return [str(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid sweep for segmented-LDA end-to-end C-TreePO simulation.")
    p.add_argument("--n-topics", type=int, default=5)
    p.add_argument("--vocab-size", type=int, default=600)
    p.add_argument("--alpha-topic", type=float, default=0.20)
    p.add_argument("--beta-word", type=float, default=0.10)
    p.add_argument("--min-segments", type=int, default=8)
    p.add_argument("--max-segments", type=int, default=20)
    p.add_argument("--min-seg-tokens", type=int, default=24)
    p.add_argument("--max-seg-tokens", type=int, default=64)
    p.add_argument("--segment-concentration", type=float, default=80.0)
    p.add_argument("--segment-background", type=float, default=2.0)
    p.add_argument("--fixed-leaf-tokens", type=int, default=32)
    p.add_argument(
        "--topic-phi-estimator",
        choices=list(VALID_TOPIC_PHI_ESTIMATORS),
        default="noisy_theory",
    )
    p.add_argument("--topic-phi-docs", type=int, default=0)
    p.add_argument("--tlda-delta", type=float, default=0.10)
    p.add_argument("--tlda-rate-constant", type=float, default=1.0)
    p.add_argument("--tlda-sigmaK-floor", type=float, default=1e-6)
    p.add_argument(
        "--topic-phi-permute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, randomly permute estimated topics (identifiability is up to permutation).",
    )
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
    p.add_argument("--spectral-svd-dim-extra", type=int, default=2)
    p.add_argument("--spectral-max-leaves", type=int, default=4000)
    p.add_argument("--spectral-kmeans-inits", type=int, default=6)
    p.add_argument("--spectral-kmeans-max-iter", type=int, default=60)
    p.add_argument("--calibration-policy", choices=["uniform", "entropy"], default="uniform")
    p.add_argument("--calibration-ridge", type=float, default=1e-4)
    p.add_argument("--calibration-pi-min", type=float, default=0.01)
    p.add_argument("--eval-internal-query-design", choices=["none", "uniform", "risk"], default="risk")
    p.add_argument("--c1-threshold", type=float, default=0.20)
    p.add_argument("--c3-threshold", type=float, default=0.20)
    p.add_argument("--selection-audit-trials", type=int, default=0)
    p.add_argument("--selection-audit-sample-rate", type=float, default=0.10)
    p.add_argument("--selection-audit-pi-min", type=float, default=0.01)

    p.add_argument("--train-docs-grid", type=str, default="64,128,256")
    p.add_argument("--test-docs-grid", type=str, default="128")
    p.add_argument("--tokens-grid", type=str, default="16,32,64")
    p.add_argument("--calibration-rate-grid", type=str, default="0.05,0.10,0.25")
    p.add_argument("--eval-leaf-rate-grid", type=str, default="0.00")
    p.add_argument("--eval-internal-rate-grid", type=str, default="0.00,0.10,0.25,0.50")
    p.add_argument("--seed-grid", type=str, default="0,1,2")

    p.add_argument(
        "--out-csv",
        type=str,
        default="outputs/segmented_lda_ctreepo/grid.csv",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    train_docs_grid = _parse_int_list(args.train_docs_grid)
    test_docs_grid = _parse_int_list(args.test_docs_grid)
    tokens_grid = _parse_int_list(args.tokens_grid)
    calib_grid = _parse_float_list(args.calibration_rate_grid)
    eval_leaf_grid = _parse_float_list(args.eval_leaf_rate_grid)
    eval_internal_grid = _parse_float_list(args.eval_internal_rate_grid)
    seed_grid = _parse_int_list(args.seed_grid)

    out_rows: list[dict] = []
    total = (
        len(train_docs_grid)
        * len(test_docs_grid)
        * len(tokens_grid)
        * len(calib_grid)
        * len(eval_leaf_grid)
        * len(eval_internal_grid)
        * len(seed_grid)
    )
    done = 0
    for (n_train, n_test, leaf_tok, cal_r, eval_leaf_r, eval_int_r, seed) in product(
        train_docs_grid,
        test_docs_grid,
        tokens_grid,
        calib_grid,
        eval_leaf_grid,
        eval_internal_grid,
        seed_grid,
    ):
        cfg = SegmentedLDACtreePOConfig(
            n_topics=int(args.n_topics),
            vocab_size=int(args.vocab_size),
            alpha_topic=float(args.alpha_topic),
            beta_word=float(args.beta_word),
            n_books_train=int(n_train),
            n_books_test=int(n_test),
            min_segments=int(args.min_segments),
            max_segments=int(args.max_segments),
            min_seg_tokens=int(args.min_seg_tokens),
            max_seg_tokens=int(args.max_seg_tokens),
            segment_concentration=float(args.segment_concentration),
            segment_background=float(args.segment_background),
            fixed_leaf_tokens=int(leaf_tok),
            topic_phi_estimator=str(args.topic_phi_estimator),
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
            neural_topic_base_estimator=str(args.neural_topic_base_estimator),
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
            spectral_svd_dim_extra=int(args.spectral_svd_dim_extra),
            spectral_max_leaves=int(args.spectral_max_leaves),
            spectral_kmeans_inits=int(args.spectral_kmeans_inits),
            spectral_kmeans_max_iter=int(args.spectral_kmeans_max_iter),
            calibration_leaf_query_rate=float(cal_r),
            calibration_policy=str(args.calibration_policy),
            calibration_ridge=float(args.calibration_ridge),
            calibration_pi_min=float(args.calibration_pi_min),
            eval_leaf_query_rate=float(eval_leaf_r),
            eval_internal_query_rate=float(eval_int_r),
            eval_internal_query_design=str(args.eval_internal_query_design),
            c1_threshold=float(args.c1_threshold),
            c3_threshold=float(args.c3_threshold),
            selection_audit_trials=int(args.selection_audit_trials),
            selection_audit_sample_rate=float(args.selection_audit_sample_rate),
            selection_audit_pi_min=float(args.selection_audit_pi_min),
            seed=int(seed),
        )
        out = run_segmented_lda_ctreepo_simulation(cfg)
        row = {f"config_{k}": v for k, v in out.config.items()}
        row["topic_phi_l2_error_mean"] = float(out.topic_meta.get("topic_phi_l2_error_mean", float("nan")))
        row["calibration_samples"] = out.calibration_samples
        for k, v in asdict(out.decomposition).items():
            row[f"decomposition_{k}"] = v
        budgeted = out.metrics["estimated_calibrated_budgeted"]
        row["budgeted_root_l1_mean"] = budgeted.root_l1_mean
        row["budgeted_c1_violation_rate"] = budgeted.c1_violation_rate
        row["budgeted_c3_violation_rate"] = budgeted.c3_violation_rate
        row["budgeted_mean_total_queries"] = budgeted.mean_total_queries
        out_rows.append(row)

        done += 1
        print(
            "[{}/{}] train={} test={} leaf={} cal={} eval_leaf={} eval_int={} seed={} | total={:.4f} upper={:.4f}".format(
                done,
                total,
                n_train,
                n_test,
                leaf_tok,
                cal_r,
                eval_leaf_r,
                eval_int_r,
                seed,
                out.decomposition.total_root_l1_mean,
                out.decomposition.upper_bound_mean,
            )
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_rows:
        out_path.write_text("", encoding="utf-8")
        print(f"wrote_csv | {out_path} | rows=0")
        return 0

    keys = list(out_rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    print(f"wrote_csv | {out_path} | rows={len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
