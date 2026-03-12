#!/usr/bin/env python3
"""Run hashed-counts classification honesty simulation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.hashed_classification_honesty import (
    HashedClassificationConfig,
    run_hashed_classification_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a hashed-counts classifier/merger and report C1/C2/C3 honesty."
    )
    parser.add_argument("--n-classes", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--hash-size", type=int, default=2048)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--min-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--min-leaf-tokens", type=int, default=16)
    parser.add_argument("--max-leaf-tokens", type=int, default=64)
    parser.add_argument("--train-docs", type=int, default=200)
    parser.add_argument("--test-docs", type=int, default=80)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--merger-hidden-dim", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--leaf-weight", type=float, default=1.0)
    parser.add_argument("--c2-weight", type=float, default=0.1)
    parser.add_argument("--c3-weight", type=float, default=1.0)
    parser.add_argument("--c3-state-weight", type=float, default=0.5)
    parser.add_argument("--audit-policy", type=str, default="all")
    parser.add_argument("--audit-fixed-nodes", type=int, default=0)
    parser.add_argument("--audit-fraction", type=float, default=1.0)
    parser.add_argument("--audit-scale", type=float, default=1.0)
    parser.add_argument("--use-log1p", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--discrepancy-threshold", type=float, default=0.1)
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
        default="outputs/hashed_classification_honesty_summary.json",
        help="JSON summary output path.",
    )
    parser.add_argument(
        "--csv-summary",
        type=str,
        default="outputs/hashed_classification_honesty_summary.csv",
        help="CSV summary output path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON to stdout (in addition to saving files).",
    )
    return parser.parse_args()


def _write_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()

    device_mode = "cpu" if bool(args.cpu) else str(args.device)
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

    config = HashedClassificationConfig(
        n_classes=int(args.n_classes),
        vocab_size=int(args.vocab_size),
        hash_size=int(args.hash_size),
        dirichlet_alpha=float(args.dirichlet_alpha),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        min_leaf_tokens=int(args.min_leaf_tokens),
        max_leaf_tokens=int(args.max_leaf_tokens),
        train_docs=int(args.train_docs),
        test_docs=int(args.test_docs),
        state_dim=int(args.state_dim),
        hidden_dim=int(args.hidden_dim),
        merger_hidden_dim=int(args.merger_hidden_dim),
        n_epochs=int(args.n_epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip_norm),
        leaf_weight=float(args.leaf_weight),
        c2_weight=float(args.c2_weight),
        c3_weight=float(args.c3_weight),
        c3_state_weight=float(args.c3_state_weight),
        audit_policy=str(args.audit_policy),
        audit_fixed_nodes=int(args.audit_fixed_nodes),
        audit_fraction=float(args.audit_fraction),
        audit_scale=float(args.audit_scale),
        use_log1p=bool(args.use_log1p),
        normalize_counts=not bool(args.no_normalize),
        discrepancy_threshold=float(args.discrepancy_threshold),
        seed=int(args.seed),
        use_cuda=bool(use_cuda),
        cuda_device=args.cuda_device,
        torch_threads=int(args.torch_threads),
    )

    summary = run_hashed_classification_experiment(config)

    json_path = Path(args.json_summary)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(summary.to_json(), encoding="utf-8")

    row = {
        **{f"config_{k}": v for k, v in summary.config.items()},
        "train_loss_final": summary.train_loss_final,
        "leaf_accuracy": summary.leaf_accuracy,
        "root_accuracy": summary.root_accuracy,
        "c1_mean_discrepancy": summary.c1.mean_discrepancy,
        "c1_violation_rate": summary.c1.violation_rate,
        "c1_n": summary.c1.n,
        "c2_mean_discrepancy": summary.c2.mean_discrepancy,
        "c2_violation_rate": summary.c2.violation_rate,
        "c2_n": summary.c2.n,
        "c3_mean_discrepancy": summary.c3.mean_discrepancy,
        "c3_violation_rate": summary.c3.violation_rate,
        "c3_n": summary.c3.n,
    }
    _write_csv(Path(args.csv_summary), row)

    if bool(args.json):
        print(summary.to_json())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
