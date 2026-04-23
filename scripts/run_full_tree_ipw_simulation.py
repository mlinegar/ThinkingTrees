#!/usr/bin/env python3
"""Estimator-only Monte Carlo for full-tree IPW with separate document labels."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
from typing import Callable, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.core.markov_changepoint_ops_count import (  # noqa: E402
    OPSCountConfig,
    build_markov_changepoint_ops_count_data_bundle,
)
from src.ctreepo.sim.core.markov_neural_operator_baselines import (  # noqa: E402
    FNOCountSketch,
    _prepare_fno_count_docs,
)
from src.tree.full_tree_ipw import (  # noqa: E402
    DEFAULT_LAYERED_RATE_GRID,
    DocumentLevelPredictionRecord,
    FullTreeNodeRecord,
    classify_layered_sampling_regime,
    layered_propensity_policy,
    run_full_tree_estimator_monte_carlo,
)
from src.tree.ipw import NodeType  # noqa: E402


def _parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for raw in str(text).replace(",", " ").split():
        token = raw.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run estimator-only full-tree IPW Monte Carlo on realized Markov trees."
    )
    parser.add_argument("--n-docs", type=int, default=128)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-regimes", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--min-segments", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument("--min-seg-len", type=int, default=4)
    parser.add_argument("--max-seg-len", type=int, default=16)
    parser.add_argument("--fixed-leaf-tokens", type=int, default=16)
    parser.add_argument("--base-node-rate", type=float, default=0.35)
    parser.add_argument("--noise-scale", type=float, default=0.08)
    parser.add_argument(
        "--grid-rates",
        type=str,
        default=",".join(f"{x:g}" for x in DEFAULT_LAYERED_RATE_GRID),
        help="Comma/space separated primary 2D grid rates for leaf and internal sampling.",
    )
    parser.add_argument(
        "--secondary-target-rates",
        type=str,
        default="",
        help=(
            "Optional comma/space separated target mean sample rates for the secondary "
            "uniform/depth-biased/hard-node-biased sweeps. Defaults to --grid-rates."
        ),
    )
    parser.add_argument(
        "--secondary-policy-families",
        type=str,
        default="uniform,depth_biased,hard_node_biased",
        help="Comma/space separated list drawn from {uniform,depth_biased,hard_node_biased}.",
    )
    parser.add_argument(
        "--json-summary",
        type=str,
        default="outputs/full_tree_ipw/full_tree_ipw_summary.json",
        help="Where to write the JSON summary.",
    )
    parser.add_argument("--json", action="store_true", help="Also print JSON to stdout.")
    return parser.parse_args()


def _synthetic_full_tree_population(
    *,
    docs: Sequence[object],
    target_scale: float,
    noise_scale: float,
    seed: int,
) -> tuple[list[FullTreeNodeRecord], list[DocumentLevelPredictionRecord]]:
    rng = random.Random(int(seed))
    node_records: list[FullTreeNodeRecord] = []
    document_records: list[DocumentLevelPredictionRecord] = []

    for doc_idx, doc in enumerate(docs):
        doc_id = f"doc_{doc_idx}"
        n_leaves = int(len(doc.leaf_token_ids))
        layout = FNOCountSketch._balanced_tree_layout(n_leaves)
        all_targets = [float(x) for x in doc.leaf_counts] + [
            float(x) for x in doc.merge_counts_balanced
        ]
        max_depth = max(int(v) for v in layout["depth_by_global_idx"].values()) or 1
        root_global_idx = int(layout["root_global_idx"])
        root_prediction = float("nan")

        for global_idx, raw_target in enumerate(all_targets):
            depth = int(layout["depth_by_global_idx"].get(global_idx, 0))
            is_leaf = bool(global_idx < n_leaves)
            is_root = bool(global_idx == root_global_idx)
            normalized_target = float(raw_target) / float(target_scale)
            depth_scale = float(depth) / float(max(1, max_depth))
            hardness = (
                1.0
                + 0.5 * depth_scale
                + (0.35 if not is_leaf else 0.0)
                + (0.45 if is_root else 0.0)
            )
            systematic_bias = 0.03 * depth_scale + (0.02 if not is_leaf else 0.0)
            prediction = min(
                1.0,
                max(
                    0.0,
                    normalized_target
                    + systematic_bias
                    + rng.gauss(0.0, float(noise_scale) * float(hardness)),
                ),
            )
            if is_root:
                root_prediction = float(prediction)
            node_records.append(
                FullTreeNodeRecord(
                    doc_id=doc_id,
                    node_id=str(layout["node_id_by_global_idx"][global_idx]),
                    depth=int(depth),
                    node_type=(NodeType.LEAF if is_leaf else NodeType.MERGE),
                    is_root=bool(is_root),
                    prediction=float(prediction),
                    target=float(normalized_target),
                    sampled=False,
                    propensity=1.0,
                    metadata={
                        "hardness_score": float(hardness),
                        "depth_fraction": float(depth_scale),
                    },
                )
            )

        document_records.append(
            DocumentLevelPredictionRecord(
                doc_id=doc_id,
                prediction=float(root_prediction),
                target=float(doc.root_count) / float(target_scale),
                metadata={
                    "raw_prediction": float(root_prediction) * float(target_scale),
                    "raw_target": float(doc.root_count),
                },
            )
        )

    return node_records, document_records


def _uniform_policy(base_rate: float) -> Callable[[FullTreeNodeRecord], float]:
    return lambda record: float(base_rate)


def _depth_biased_policy(base_rate: float) -> Callable[[FullTreeNodeRecord], float]:
    def _policy(record: FullTreeNodeRecord) -> float:
        depth_fraction = float(record.metadata.get("depth_fraction", 0.0))
        return float(base_rate) * (0.55 + 0.9 * depth_fraction)

    return _policy


def _hard_node_biased_policy(base_rate: float) -> Callable[[FullTreeNodeRecord], float]:
    def _policy(record: FullTreeNodeRecord) -> float:
        hardness = float(record.metadata.get("hardness_score", 1.0))
        return float(base_rate) * (0.45 + 0.45 * hardness)

    return _policy


def _mean_propensity(
    records: Sequence[FullTreeNodeRecord],
    policy_fn: Callable[[FullTreeNodeRecord], float],
) -> float:
    if not records:
        return float("nan")
    total = 0.0
    for record in records:
        total += min(1.0, max(0.0, float(policy_fn(record))))
    return float(total) / float(len(records))


def _calibrated_weighted_policy(
    records: Sequence[FullTreeNodeRecord],
    *,
    target_rate: float,
    weight_fn: Callable[[FullTreeNodeRecord], float],
) -> tuple[Callable[[FullTreeNodeRecord], float], float]:
    rate = min(1.0, max(0.0, float(target_rate)))
    if rate <= 0.0:
        return (lambda record: 0.0), 0.0
    if rate >= 1.0:
        return (lambda record: 1.0), 1.0

    weights = [max(1e-8, float(weight_fn(record))) for record in records]
    if not weights:
        return (lambda record: 0.0), 0.0

    def _mean_with_scale(scale: float) -> float:
        total = 0.0
        for weight in weights:
            total += min(1.0, float(scale) * float(weight))
        return float(total) / float(len(weights))

    lo = 0.0
    hi = 1.0
    while _mean_with_scale(hi) < rate - 1e-10:
        hi *= 2.0
        if hi >= 1e9:
            break

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _mean_with_scale(mid) < rate:
            lo = mid
        else:
            hi = mid

    scale = float(hi)

    def _policy(record: FullTreeNodeRecord) -> float:
        return min(1.0, max(0.0, scale * max(1e-8, float(weight_fn(record)))))

    return _policy, _mean_propensity(records, _policy)


def _matrix_from_cells(
    *,
    rate_values: Sequence[float],
    cells: Sequence[dict],
    value_getter: Callable[[dict], float],
) -> list[list[float]]:
    index = {
        (float(cell["p_internal"]), float(cell["p_leaf"])): float(value_getter(cell))
        for cell in cells
    }
    matrix: list[list[float]] = []
    for internal_rate in rate_values:
        row: list[float] = []
        for leaf_rate in rate_values:
            row.append(float(index[(float(internal_rate), float(leaf_rate))]))
        matrix.append(row)
    return matrix


def _find_grid_cell(
    cells: Sequence[dict],
    *,
    p_internal: float,
    p_leaf: float,
) -> dict | None:
    for cell in cells:
        if abs(float(cell["p_internal"]) - float(p_internal)) <= 1e-12 and abs(
            float(cell["p_leaf"]) - float(p_leaf)
        ) <= 1e-12:
            return cell
    return None


def main() -> int:
    args = parse_args()
    config = OPSCountConfig(
        n_regimes=int(args.n_regimes),
        vocab_size=int(args.vocab_size),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        min_segments=int(args.min_segments),
        max_segments=int(args.max_segments),
        min_seg_len=int(args.min_seg_len),
        max_seg_len=int(args.max_seg_len),
        fixed_leaf_tokens=int(args.fixed_leaf_tokens),
        train_docs=0,
        val_docs=0,
        test_docs=int(args.n_docs),
        seed=int(args.seed),
        use_cuda=False,
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(config)
    docs = _prepare_fno_count_docs(
        bundle.test_docs[: int(args.n_docs)],
        leaf_tokens=int(args.fixed_leaf_tokens),
    )
    target_scale = float(max(1, int(args.max_segments) - 1))
    node_records, document_records = _synthetic_full_tree_population(
        docs=docs,
        target_scale=float(target_scale),
        noise_scale=float(args.noise_scale),
        seed=int(args.seed) + 17,
    )

    grid_rates = sorted(
        {float(x) for x in (_parse_float_list(str(args.grid_rates)) or list(DEFAULT_LAYERED_RATE_GRID))}
    )
    if any(float(rate) < 0.0 or float(rate) > 1.0 for rate in grid_rates):
        raise ValueError("grid rates must lie in [0, 1]")
    secondary_target_rates = _parse_float_list(str(args.secondary_target_rates))
    if not secondary_target_rates:
        secondary_target_rates = list(grid_rates)
    if any(float(rate) < 0.0 or float(rate) > 1.0 for rate in secondary_target_rates):
        raise ValueError("secondary target rates must lie in [0, 1]")
    secondary_policy_families = [
        str(name).strip()
        for name in str(args.secondary_policy_families).replace(",", " ").split()
        if str(name).strip()
    ]

    payload = {
        "simulation": "full_tree_ipw_estimator_only",
        "n_docs": int(args.n_docs),
        "trials": int(args.trials),
        "target_scale": float(target_scale),
        "base_node_rate": float(args.base_node_rate),
        "noise_scale": float(args.noise_scale),
        "population_reference": {},
        "primary_grid": {
            "rate_axis": [float(x) for x in grid_rates],
            "cells": [],
            "anchors": {},
            "diagonal": [],
            "matrices": {},
        },
        "secondary_policy_sweeps": {},
    }

    primary_cells: list[dict] = []
    for internal_idx, internal_rate in enumerate(grid_rates):
        for leaf_idx, leaf_rate in enumerate(grid_rates):
            summary = run_full_tree_estimator_monte_carlo(
                node_records,
                document_records,
                propensity_fn=layered_propensity_policy(
                    leaf_rate=float(leaf_rate),
                    internal_rate=float(internal_rate),
                ),
                n_trials=int(args.trials),
                seed=int(args.seed) + 10_000 + 101 * internal_idx + 13 * leaf_idx,
                policy_name=f"layered_internal_{internal_rate:g}_leaf_{leaf_rate:g}",
            )
            cell = {
                "p_internal": float(internal_rate),
                "p_leaf": float(leaf_rate),
                "regime": classify_layered_sampling_regime(
                    leaf_rate=float(leaf_rate),
                    internal_rate=float(internal_rate),
                ),
                "summary": asdict(summary),
            }
            primary_cells.append(cell)

    payload["primary_grid"]["cells"] = primary_cells
    payload["population_reference"] = {
        "true_full_node_mean": float(primary_cells[0]["summary"]["true_full_node_mean"])
        if primary_cells
        else float("nan"),
        "document_top_loss": float(primary_cells[0]["summary"]["document_top_loss"])
        if primary_cells
        else float("nan"),
        "document_top_mae": float(primary_cells[0]["summary"]["document_top_mae"])
        if primary_cells
        else float("nan"),
    }
    anchors: dict[str, dict] = {}
    doc_only_cell = _find_grid_cell(primary_cells, p_internal=0.0, p_leaf=0.0)
    if doc_only_cell is not None:
        anchors["doc_only"] = doc_only_cell
    full_tree_cell = _find_grid_cell(primary_cells, p_internal=1.0, p_leaf=1.0)
    if full_tree_cell is not None:
        anchors["full_tree"] = full_tree_cell
    payload["primary_grid"]["anchors"] = anchors
    payload["primary_grid"]["diagonal"] = [
        cell
        for cell in primary_cells
        if abs(float(cell["p_internal"]) - float(cell["p_leaf"])) <= 1e-12
    ]
    payload["primary_grid"]["matrices"] = {
        "naive_bias": _matrix_from_cells(
            rate_values=grid_rates,
            cells=primary_cells,
            value_getter=lambda cell: float(cell["summary"]["naive"]["bias"]),
        ),
        "naive_rmse": _matrix_from_cells(
            rate_values=grid_rates,
            cells=primary_cells,
            value_getter=lambda cell: float(cell["summary"]["naive"]["rmse"]),
        ),
        "ht_bias": _matrix_from_cells(
            rate_values=grid_rates,
            cells=primary_cells,
            value_getter=lambda cell: float(cell["summary"]["ht"]["bias"]),
        ),
        "ht_rmse": _matrix_from_cells(
            rate_values=grid_rates,
            cells=primary_cells,
            value_getter=lambda cell: float(cell["summary"]["ht"]["rmse"]),
        ),
        "hajek_bias": _matrix_from_cells(
            rate_values=grid_rates,
            cells=primary_cells,
            value_getter=lambda cell: float(cell["summary"]["hajek"]["bias"]),
        ),
        "hajek_rmse": _matrix_from_cells(
            rate_values=grid_rates,
            cells=primary_cells,
            value_getter=lambda cell: float(cell["summary"]["hajek"]["rmse"]),
        ),
        "mean_sample_count": _matrix_from_cells(
            rate_values=grid_rates,
            cells=primary_cells,
            value_getter=lambda cell: float(cell["summary"]["mean_sample_count"]),
        ),
        "mean_effective_sample_size": _matrix_from_cells(
            rate_values=grid_rates,
            cells=primary_cells,
            value_getter=lambda cell: float(cell["summary"]["mean_effective_sample_size"]),
        ),
        "mean_max_weight": _matrix_from_cells(
            rate_values=grid_rates,
            cells=primary_cells,
            value_getter=lambda cell: float(cell["summary"]["mean_max_weight"]),
        ),
    }

    base_rate = float(args.base_node_rate)
    secondary_policy_builders: Dict[str, Callable[[float], Callable[[FullTreeNodeRecord], float]]] = {
        "uniform": _uniform_policy,
        "depth_biased": _depth_biased_policy,
        "hard_node_biased": _hard_node_biased_policy,
    }
    calibrated_weight_builders: Dict[str, Callable[[FullTreeNodeRecord], float]] = {
        "uniform": lambda record: 1.0,
        "depth_biased": lambda record: 0.4 + 0.9 * float(record.metadata.get("depth_fraction", 0.0)),
        "hard_node_biased": lambda record: max(
            0.1, float(record.metadata.get("hardness_score", 1.0))
        ),
    }
    for policy_idx, policy_name in enumerate(secondary_policy_families):
        if policy_name not in secondary_policy_builders:
            raise ValueError(f"unknown secondary policy family: {policy_name!r}")
        points: list[dict] = []
        for rate_idx, target_rate in enumerate(secondary_target_rates):
            calibrated_policy_fn, realized_mean_propensity = _calibrated_weighted_policy(
                node_records,
                target_rate=float(target_rate),
                weight_fn=calibrated_weight_builders[policy_name],
            )
            summary = run_full_tree_estimator_monte_carlo(
                node_records,
                document_records,
                propensity_fn=calibrated_policy_fn,
                n_trials=int(args.trials),
                seed=int(args.seed) + 20_000 + 1_003 * (policy_idx + 1) + 97 * rate_idx,
                policy_name=f"{policy_name}_target_{target_rate:g}",
            )
            points.append(
                {
                    "target_rate": float(target_rate),
                    "realized_mean_propensity": float(realized_mean_propensity),
                    "summary": asdict(summary),
                }
            )
        summary = run_full_tree_estimator_monte_carlo(
            node_records,
            document_records,
            propensity_fn=secondary_policy_builders[policy_name](base_rate),
            n_trials=int(args.trials),
            seed=int(args.seed),
            policy_name=str(policy_name),
        )
        payload["secondary_policy_sweeps"][str(policy_name)] = {
            "rate_axis": [float(x) for x in secondary_target_rates],
            "points": points,
            "legacy_base_rate_reference": {
                "base_node_rate": float(base_rate),
                "mean_population_propensity": float(
                    _mean_propensity(
                        node_records,
                        secondary_policy_builders[policy_name](base_rate),
                    )
                ),
                "summary": asdict(summary),
            },
        }

    out_path = Path(str(args.json_summary))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps({"json_summary": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
