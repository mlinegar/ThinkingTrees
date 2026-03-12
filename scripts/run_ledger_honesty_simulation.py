#!/usr/bin/env python3
"""Run key-value ledger honesty simulations."""

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

from src.tree.ledger_honesty_simulation import (
    LedgerHonestyConfig,
    run_ledger_honesty_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a learned ledger summarizer/merger and report C1/C2/C3 honesty."
    )
    parser.add_argument("--num-keys", type=int, default=16)
    parser.add_argument("--num-values", type=int, default=8)
    parser.add_argument("--key-zipf-alpha", type=float, default=1.1)
    parser.add_argument("--min-updates", type=int, default=64)
    parser.add_argument("--max-updates", type=int, default=256)
    parser.add_argument("--min-leaf-updates", type=int, default=8)
    parser.add_argument("--max-leaf-updates", type=int, default=32)
    parser.add_argument("--train-docs", type=int, default=120)
    parser.add_argument("--test-docs", type=int, default=40)
    parser.add_argument("--emb-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--merger-hidden-dim", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--c1-weight", type=float, default=1.0)
    parser.add_argument("--c2-weight", type=float, default=0.5)
    parser.add_argument("--c3-weight", type=float, default=1.0)
    parser.add_argument("--audit-policy", type=str, default="all")
    parser.add_argument("--audit-fixed-nodes", type=int, default=0)
    parser.add_argument("--audit-fraction", type=float, default=1.0)
    parser.add_argument("--audit-scale", type=float, default=1.0)
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
        default="outputs/ledger_honesty_summary.json",
        help="JSON summary output path.",
    )
    parser.add_argument(
        "--csv-summary",
        type=str,
        default="outputs/ledger_honesty_summary.csv",
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

    config = LedgerHonestyConfig(
        num_keys=int(args.num_keys),
        num_values=int(args.num_values),
        key_zipf_alpha=float(args.key_zipf_alpha),
        min_updates=int(args.min_updates),
        max_updates=int(args.max_updates),
        min_leaf_updates=int(args.min_leaf_updates),
        max_leaf_updates=int(args.max_leaf_updates),
        train_docs=int(args.train_docs),
        test_docs=int(args.test_docs),
        emb_dim=int(args.emb_dim),
        hidden_dim=int(args.hidden_dim),
        merger_hidden_dim=int(args.merger_hidden_dim),
        n_epochs=int(args.n_epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip_norm),
        c1_weight=float(args.c1_weight),
        c2_weight=float(args.c2_weight),
        c3_weight=float(args.c3_weight),
        audit_policy=str(args.audit_policy),
        audit_fixed_nodes=int(args.audit_fixed_nodes),
        audit_fraction=float(args.audit_fraction),
        audit_scale=float(args.audit_scale),
        seed=int(args.seed),
        use_cuda=bool(use_cuda),
        cuda_device=args.cuda_device,
        torch_threads=int(args.torch_threads),
    )

    summary = run_ledger_honesty_experiment(config)

    json_path = Path(args.json_summary)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(summary.to_json(), encoding="utf-8")

    row = {
        **{f"config_{k}": v for k, v in summary.config.items()},
        "train_loss_final": summary.train_loss_final,
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
