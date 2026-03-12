#!/usr/bin/env python3
"""Run mergeable bigram-score oracle-guidance simulations."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.bigram_score_guidance_simulation import (  # noqa: E402
    BigramScoreGuidanceConfig,
    BigramScoreGuidanceSummary,
    run_bigram_score_guidance_experiment,
)


def _parse_float_csv(s: str) -> tuple[float, ...]:
    out = tuple(float(x.strip()) for x in s.split(",") if x.strip())
    if len(out) == 0:
        raise ValueError("expected a non-empty float CSV")
    return out


def _parse_str_csv(s: str) -> tuple[str, ...]:
    out = tuple(x.strip() for x in s.split(",") if x.strip())
    if len(out) == 0:
        raise ValueError("expected a non-empty string CSV")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate learning a mergeable bigram-score oracle from span-level oracle queries. "
            "Queries can be on leaves/internal nodes of a fixed balanced tree."
        )
    )
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--n-topics", type=int, default=4)
    parser.add_argument("--topic-concentration", type=float, default=0.4)
    parser.add_argument(
        "--oracle-feature-mode",
        type=str,
        default="topic_bigrams",
        choices=("token_bigrams", "topic_bigrams"),
        help="Which bigram alphabet the oracle score uses.",
    )
    parser.add_argument(
        "--align-segments-to-leaves",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, topic changes occur only at leaf boundaries (makes boundary info matter).",
    )
    parser.add_argument(
        "--disjoint-topic-vocab",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, each topic uses a disjoint vocab block (sharper boundary bigrams).",
    )
    parser.add_argument(
        "--cross-topic-weight-multiplier",
        "--lambda",
        type=float,
        default=5.0,
        help="If disjoint-topic-vocab, multiplies oracle weights for cross-topic bigrams.",
    )
    parser.add_argument("--min-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--min-segments", type=int, default=3)
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument("--min-seg-len", type=int, default=16)
    parser.add_argument("--max-seg-len", type=int, default=96)
    parser.add_argument("--leaf-tokens", type=int, default=32)
    parser.add_argument(
        "--boundary-profile",
        type=str,
        default="uniform",
        choices=("uniform", "start", "middle", "end", "bimodal", "random"),
        help="Global segment-boundary location profile (learnable across documents).",
    )
    parser.add_argument(
        "--boundary-profile-strength",
        type=float,
        default=1.0,
        help="How strongly to bias boundaries toward the chosen profile (0=uniform).",
    )
    parser.add_argument(
        "--boundary-profile-seed",
        type=int,
        default=-1,
        help="Seed for a random boundary profile (negative derives from --seed).",
    )
    parser.add_argument("--w-scale", type=float, default=1.0)
    parser.add_argument("--w-sparsity", type=float, default=0.25)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--oracle-cost-power", type=float, default=1.25)
    parser.add_argument("--oracle-cost-per-query", type=float, default=0.0)
    parser.add_argument(
        "--guidance-per-leaf",
        type=str,
        default="0,0.25,0.5,1,2",
        help=(
            "Additional internal-node oracle queries per leaf (base leaf labels are always included). "
            "CSV of floats."
        ),
    )
    parser.add_argument("--guidance-strategies", type=str, default="random,active")
    parser.add_argument("--train-docs", type=int, default=200)
    parser.add_argument("--test-docs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-summary", type=str, required=True)
    parser.add_argument("--csv-summary", type=str, required=True)
    return parser.parse_args()


def _write_csv(summary: BigramScoreGuidanceSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "policy",
        "guidance_strategy",
        "guidance_per_leaf",
        "oracle_queries_leaf_total",
        "oracle_queries_extra_total",
        "oracle_queries_total",
        "oracle_cost_leaf_total",
        "oracle_cost_extra_total",
        "oracle_cost_total",
        "mean_abs_error",
        "rmse",
        "weight_rmse",
        "weight_cosine",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, m in summary.metrics.items():
            row = {"policy": key}
            row.update(asdict(m))
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args()

    config = BigramScoreGuidanceConfig(
        vocab_size=int(args.vocab_size),
        n_topics=int(args.n_topics),
        topic_concentration=float(args.topic_concentration),
        oracle_feature_mode=str(args.oracle_feature_mode),
        align_segments_to_leaves=bool(args.align_segments_to_leaves),
        disjoint_topic_vocab=bool(args.disjoint_topic_vocab),
        cross_topic_weight_multiplier=float(args.cross_topic_weight_multiplier),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        min_segments=int(args.min_segments),
        max_segments=int(args.max_segments),
        min_seg_len=int(args.min_seg_len),
        max_seg_len=int(args.max_seg_len),
        leaf_tokens=int(args.leaf_tokens),
        boundary_profile=str(args.boundary_profile),
        boundary_profile_strength=float(args.boundary_profile_strength),
        boundary_profile_seed=int(args.boundary_profile_seed),
        w_scale=float(args.w_scale),
        w_sparsity=float(args.w_sparsity),
        ridge_lambda=float(args.ridge_lambda),
        oracle_cost_power=float(args.oracle_cost_power),
        oracle_cost_per_query=float(args.oracle_cost_per_query),
        guidance_per_leaf=_parse_float_csv(args.guidance_per_leaf),
        guidance_strategies=_parse_str_csv(args.guidance_strategies),
        train_docs=int(args.train_docs),
        test_docs=int(args.test_docs),
        seed=int(args.seed),
    )

    summary = run_bigram_score_guidance_experiment(config)

    json_path = Path(args.json_summary)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(summary.to_json(), encoding="utf-8")
    print(f"wrote_json | {json_path}")

    csv_path = Path(args.csv_summary)
    _write_csv(summary, csv_path)
    print(f"wrote_csv | {csv_path}")

    print(
        f"train_full_doc_cost={summary.train_full_doc_cost_total:.3f} | "
        f"mean_leaves_train={summary.mean_leaf_count_train:.2f}"
    )
    for key, m in summary.metrics.items():
        print(
            f"policy={key} | q_leaf={m.oracle_queries_leaf_total} | q_extra={m.oracle_queries_extra_total} | "
            f"q_total={m.oracle_queries_total} | cost={m.oracle_cost_total:.1f} | "
            f"mae={m.mean_abs_error:.4f} | rmse={m.rmse:.4f} | "
            f"w_rmse={m.weight_rmse:.4f} | w_cos={m.weight_cosine:.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
