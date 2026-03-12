#!/usr/bin/env python3
"""Run learned mergeable-sketch vs HLL baseline simulations."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.learned_sketch_simulation import (
    DEFAULT_LAW_COMPONENT_SHARE,
    DEFAULT_LAW_STRENGTH,
    DEFAULT_REGULARIZER_WEIGHT,
    DEFAULT_SUMMARY_SHARE,
    LearningRunSummary,
    SimulationConfig,
    VALID_AUDIT_POLICIES,
    VALID_SIMULATION_MODES,
    run_learning_vs_hll_experiment,
)


def _parse_int_csv(s: str) -> tuple[int, ...]:
    out = tuple(int(x.strip()) for x in s.split(",") if x.strip())
    if len(out) == 0:
        raise ValueError("expected a non-empty int CSV")
    return out


def _parse_float_csv(s: str) -> tuple[float, ...]:
    out = tuple(float(x.strip()) for x in s.split(",") if x.strip())
    if len(out) == 0:
        raise ValueError("expected a non-empty float CSV")
    return out


def _resolve_summary_share(
    *,
    summary_share: float | None,
    law_strength: float | None,
) -> float:
    if summary_share is not None and law_strength is not None:
        raise ValueError(
            "use either --summary-regularizer-share or --law-strength, not both"
        )
    if law_strength is not None:
        if not (0.0 <= float(law_strength) <= 1.0):
            raise ValueError("law_strength must be in [0, 1]")
        return float(1.0 - float(law_strength))
    if summary_share is not None:
        if not (0.0 <= float(summary_share) <= 1.0):
            raise ValueError("summary_regularizer_share must be in [0, 1]")
        return float(summary_share)
    return float(DEFAULT_SUMMARY_SHARE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a learned tree sketch from oracle cardinality queries and compare "
            "against matched-memory HyperLogLog."
        )
    )
    parser.add_argument("--universe-size", type=int, default=2048)
    parser.add_argument("--min-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--leaf-size", type=int, default=32)
    parser.add_argument("--zipf-alphas", type=str, default="0.6,0.8,1.0,1.2,1.4")
    parser.add_argument("--state-dims", type=str, default="16,32,64")
    parser.add_argument("--train-sizes", type=str, default="128,256,512")
    parser.add_argument("--n-val", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--c3-weight", type=float, default=0.20)
    parser.add_argument("--leaf-weight", type=float, default=0.05)
    parser.add_argument("--idemp-weight", type=float, default=0.05)
    parser.add_argument(
        "--regularizer-weight",
        type=float,
        default=DEFAULT_REGULARIZER_WEIGHT,
        help=(
            "Lambda in (1-lambda) * global_error + lambda * regularizer. "
            f"Default: {DEFAULT_REGULARIZER_WEIGHT:.2f}."
        ),
    )
    parser.add_argument(
        "--summary-regularizer-share",
        type=float,
        default=None,
        help=(
            "Share of the regularizer placed on summary-budget pressure. "
            f"Default: {DEFAULT_SUMMARY_SHARE:.2f}."
        ),
    )
    parser.add_argument(
        "--law-strength",
        type=float,
        default=None,
        help=(
            "Alias for 1 - summary_regularizer_share. "
            f"Default endpoint pairing implies {DEFAULT_LAW_STRENGTH:.2f}."
        ),
    )
    parser.add_argument(
        "--law-leaf-share",
        type=float,
        default=DEFAULT_LAW_COMPONENT_SHARE,
        help="Leaf-law share inside the law penalty before normalization.",
    )
    parser.add_argument(
        "--law-merge-share",
        type=float,
        default=DEFAULT_LAW_COMPONENT_SHARE,
        help="Merge-law share inside the law penalty before normalization.",
    )
    parser.add_argument(
        "--law-idemp-share",
        type=float,
        default=DEFAULT_LAW_COMPONENT_SHARE,
        help="Idempotence-law share inside the law penalty before normalization.",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--audit-policy",
        type=str,
        choices=list(VALID_AUDIT_POLICIES),
        default="all",
        help="Internal-node audit sampling policy used for latent merge supervision.",
    )
    parser.add_argument(
        "--simulation-mode",
        type=str,
        choices=list(VALID_SIMULATION_MODES),
        default="latent_proxy_baseline",
        help="Whether to report proxy-only latent metrics or decoded approximate local-law metrics.",
    )
    parser.add_argument(
        "--audit-fixed-nodes",
        type=int,
        default=0,
        help="Fixed sampled internal nodes/doc when --audit-policy fixed.",
    )
    parser.add_argument(
        "--audit-fraction",
        type=float,
        default=1.0,
        help="Sample fraction of internal nodes/doc when --audit-policy fraction.",
    )
    parser.add_argument(
        "--audit-scale",
        type=float,
        default=1.0,
        help="Scale multiplier for sqrt/log2 audit policies.",
    )
    parser.add_argument(
        "--no-root-query",
        action="store_true",
        help="Disable root oracle-loss supervision during training.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device mode. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Alias for --device cpu (kept for backward compatibility).",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="CUDA device index to target when using GPU.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="Set torch intra-op/inter-op threads (<=0 keeps torch defaults).",
    )
    parser.add_argument(
        "--json-summary",
        type=str,
        default="outputs/learned_sketch_simulation_summary.json",
        help="JSON summary output path.",
    )
    parser.add_argument(
        "--csv-summary",
        type=str,
        default="outputs/learned_sketch_simulation_summary.csv",
        help="CSV summary output path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON to stdout (in addition to saving files).",
    )
    return parser.parse_args()


def _rows_from_results(results: Sequence[LearningRunSummary]) -> List[dict]:
    rows: List[dict] = []
    for r in results:
        lm = r.learned_metrics
        hm = r.hll_metrics
        ro = r.regularized_objective
        rows.append(
            {
                "state_dim": r.state_dim,
                "learned_memory_bits": r.learned_memory_bits,
                "train_size": r.train_size,
                "train_loss_final": r.train_loss_final,
                "val_loss_final": r.val_loss_final,
                "learned_mae": lm.mae,
                "learned_rmse": lm.rmse,
                "learned_relative_rmse": lm.relative_rmse,
                "learned_mean_abs_rel_error": lm.mean_abs_rel_error,
                "learned_schedule_spread_mean": lm.schedule_spread_mean,
                "latent_merge_state_mse": lm.latent_merge_state_mse,
                "eps_leaf": lm.eps_leaf,
                "eps_merge": lm.eps_merge,
                "eps_idemp": lm.eps_idemp,
                "evidence_status": lm.evidence_status,
                "simulation_mode": lm.simulation_mode,
                "hll_precision": hm.precision,
                "hll_registers": hm.registers,
                "hll_memory_bits": hm.memory_bits,
                "hll_rse_theory": r.hll_rse_theory,
                "hll_mae": hm.mae,
                "hll_rmse": hm.rmse,
                "hll_relative_rmse": hm.relative_rmse,
                "hll_mean_abs_rel_error": hm.mean_abs_rel_error,
                "hll_schedule_spread_mean": hm.schedule_spread_mean,
                "distance_to_hll_floor_rel_rmse": r.distance_to_hll_floor_rel_rmse,
                "distance_to_hll_empirical_rel_rmse": r.distance_to_hll_empirical_rel_rmse,
                "train_mean_tokens": r.train_mean_tokens,
                "train_mean_leaves": r.train_mean_leaves,
                "train_mean_internal_nodes": r.train_mean_internal_nodes,
                "train_audit_nodes_mean": r.train_audit_nodes_mean,
                "train_audit_coverage_mean": r.train_audit_coverage_mean,
                "train_root_queries_total": r.train_root_queries_total,
                "train_audit_nodes_total": r.train_audit_nodes_total,
                "train_total_queries_estimate": r.train_total_queries_estimate,
                "rmse_gap_vs_hll": r.rmse_gap_vs_hll,
                "abs_rel_error_gap_vs_hll": r.abs_rel_error_gap_vs_hll,
                "theoretical_floor_rmse": r.theoretical_floor_rmse,
                "excess_rmse": r.excess_rmse,
                "ratio_to_floor_rmse": r.ratio_to_floor_rmse,
                "ratio_to_floor_rel_rmse": r.ratio_to_floor_rel_rmse,
                "hll_empirical_excess_rmse": r.hll_empirical_excess_rmse,
                "hll_empirical_excess_rel_rmse": r.hll_empirical_excess_rel_rmse,
                "test_cardinality_rms": r.test_cardinality_rms,
                "test_cardinality_mean": r.test_cardinality_mean,
                "regularized_objective_total": ro.total,
                "regularized_objective_global_error": ro.global_error,
                "regularized_objective_summary_budget_penalty": ro.summary_budget_penalty,
                "regularized_objective_law_penalty": ro.law_penalty,
                "regularized_objective_combined_regularizer": ro.combined_regularizer,
                "regularized_objective_lambda": ro.regularizer_weight,
                "regularized_objective_summary_share": ro.summary_share,
                "regularized_objective_law_strength": ro.law_strength,
                "regularized_objective_leaf_share": ro.leaf_share,
                "regularized_objective_merge_share": ro.merge_share,
                "regularized_objective_idemp_share": ro.idemp_share,
                "regularized_objective_law_scale": ro.law_scale,
                "regularized_objective_uses_proxy_law_penalty": ro.uses_proxy_law_penalty,
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if len(rows) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    device_mode = "cpu" if bool(args.cpu) else str(args.device)
    summary_share = _resolve_summary_share(
        summary_share=args.summary_regularizer_share,
        law_strength=args.law_strength,
    )
    if device_mode == "auto":
        use_cuda = bool(torch.cuda.is_available())
    elif device_mode == "cpu":
        use_cuda = False
    else:
        use_cuda = True

    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")
    if not use_cuda and args.cuda_device is not None:
        raise ValueError("--cuda-device is only valid when using CUDA.")

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(int(args.torch_threads))

    config = SimulationConfig(
        universe_size=int(args.universe_size),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        leaf_size=int(args.leaf_size),
        zipf_alphas=_parse_float_csv(args.zipf_alphas),
        state_dims=_parse_int_csv(args.state_dims),
        train_sizes=_parse_int_csv(args.train_sizes),
        n_val=int(args.n_val),
        n_test=int(args.n_test),
        hidden_dim=int(args.hidden_dim),
        n_epochs=int(args.n_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        c3_weight=float(args.c3_weight),
        leaf_weight=float(args.leaf_weight),
        idemp_weight=float(args.idemp_weight),
        regularizer_weight=float(args.regularizer_weight),
        summary_regularizer_share=float(summary_share),
        law_leaf_share=float(args.law_leaf_share),
        law_merge_share=float(args.law_merge_share),
        law_idemp_share=float(args.law_idemp_share),
        grad_clip_norm=float(args.grad_clip_norm),
        audit_policy=str(args.audit_policy),
        audit_fixed_nodes=int(args.audit_fixed_nodes),
        audit_fraction=float(args.audit_fraction),
        audit_scale=float(args.audit_scale),
        audit_include_root_query=not bool(args.no_root_query),
        simulation_mode=str(args.simulation_mode),
        use_cuda=use_cuda,
        cuda_device=int(args.cuda_device) if args.cuda_device is not None else None,
        seed=int(args.seed),
    )

    summary = run_learning_vs_hll_experiment(config)
    rows = _rows_from_results(summary.results)

    json_path = Path(args.json_summary)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            **asdict(config),
            "law_strength": float(1.0 - summary_share),
        },
        "runtime_config": summary.config,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = Path(args.csv_summary)
    _write_csv(csv_path, rows)

    if args.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")

    print(
        "state_dim | train_size | reg_obj | global_err | regularizer | hll_rel_rmse | "
        "dist_to_floor | audit_cov | queries"
    )
    for row in sorted(rows, key=lambda r: (r["state_dim"], r["train_size"])):
        print(
            f"{int(row['state_dim'])} | {int(row['train_size'])} | "
            f"{float(row['regularized_objective_total']):.5f} | "
            f"{float(row['regularized_objective_global_error']):.5f} | "
            f"{float(row['regularized_objective_combined_regularizer']):.5f} | "
            f"{float(row['hll_relative_rmse']):.5f} | "
            f"{float(row['distance_to_hll_floor_rel_rmse']):+.5f} | "
            f"{float(row['train_audit_coverage_mean']):.3f} | {int(row['train_total_queries_estimate'])}"
        )

    print(f"wrote_json | {json_path}")
    print(f"wrote_csv | {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
