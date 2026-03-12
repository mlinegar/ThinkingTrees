#!/usr/bin/env python3
"""Run Markov boundary honesty simulations (adaptive chunking toy model)."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.markov_boundary_honesty_simulation import (  # noqa: E402
    MarkovBoundaryConfig,
    run_markov_boundary_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Markov boundary honesty simulation.")
    parser.add_argument("--n-classes", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--min-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--min-leaf-tokens", type=int, default=16)
    parser.add_argument("--max-leaf-tokens", type=int, default=64)
    parser.add_argument("--fixed-leaf-tokens", type=int, default=64)
    parser.add_argument("--train-docs", type=int, default=200)
    parser.add_argument("--test-docs", type=int, default=80)
    parser.add_argument("--sinkhorn-iters", type=int, default=40)
    parser.add_argument("--transition-log-std", type=float, default=1.25)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--boundary-emb-dim", type=int, default=32)
    parser.add_argument("--boundary-hidden-dim", type=int, default=64)
    parser.add_argument("--boundary-batch-size", type=int, default=256)
    parser.add_argument("--boundary-max-train-samples", type=int, default=120000)
    parser.add_argument("--n-epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
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
        default="outputs/markov_boundary_honesty_summary.json",
        help="JSON summary output path.",
    )
    parser.add_argument(
        "--csv-summary",
        type=str,
        default="outputs/markov_boundary_honesty_summary.csv",
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

    config = MarkovBoundaryConfig(
        n_classes=int(args.n_classes),
        vocab_size=int(args.vocab_size),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        min_leaf_tokens=int(args.min_leaf_tokens),
        max_leaf_tokens=int(args.max_leaf_tokens),
        fixed_leaf_tokens=int(args.fixed_leaf_tokens),
        train_docs=int(args.train_docs),
        test_docs=int(args.test_docs),
        sinkhorn_iters=int(args.sinkhorn_iters),
        transition_log_std=float(args.transition_log_std),
        window_size=int(args.window_size),
        boundary_emb_dim=int(args.boundary_emb_dim),
        boundary_hidden_dim=int(args.boundary_hidden_dim),
        boundary_batch_size=int(args.boundary_batch_size),
        boundary_max_train_samples=int(args.boundary_max_train_samples),
        n_epochs=int(args.n_epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip_norm),
        seed=int(args.seed),
        use_cuda=bool(use_cuda),
        cuda_device=args.cuda_device,
        torch_threads=int(args.torch_threads),
    )

    summary = run_markov_boundary_experiment(config)

    json_path = Path(args.json_summary)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(summary.to_json(), encoding="utf-8")

    row = {f"config_{k}": v for k, v in summary.config.items()}
    row["boundary_model_train_loss_final"] = summary.boundary_model_train_loss_final
    for policy, metrics in summary.metrics.items():
        d = asdict(metrics)
        for k, v in d.items():
            row[f"{policy}_{k}"] = v
    _write_csv(Path(args.csv_summary), row)

    if bool(args.json):
        print(summary.to_json())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
