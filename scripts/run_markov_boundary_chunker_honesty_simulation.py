#!/usr/bin/env python3
"""Run Markov boundary-cost simulation bridged to the adaptive chunker honesty split."""

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

from src.preprocessing.chunker import AdaptiveChunkingConfig, HonestChunkingPolicy  # noqa: E402
from src.tree.markov_boundary_chunker_honesty_simulation import (  # noqa: E402
    run_markov_chunker_honesty_experiment,
)
from src.tree.markov_boundary_honesty_simulation import MarkovBoundaryConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Markov chunker honesty bridge simulation."
    )
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=96)
    parser.add_argument("--min-tokens", type=int, default=96)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--min-leaf-tokens", type=int, default=8)
    parser.add_argument("--max-leaf-tokens", type=int, default=32)
    parser.add_argument("--fixed-leaf-tokens", type=int, default=16)
    parser.add_argument("--train-docs", type=int, default=120)
    parser.add_argument("--test-docs", type=int, default=60)
    parser.add_argument("--sinkhorn-iters", type=int, default=30)
    parser.add_argument("--transition-log-std", type=float, default=1.35)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--boundary-emb-dim", type=int, default=24)
    parser.add_argument("--boundary-hidden-dim", type=int, default=48)
    parser.add_argument("--boundary-batch-size", type=int, default=256)
    parser.add_argument("--boundary-max-train-samples", type=int, default=60000)
    parser.add_argument("--n-epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device mode. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--token-char-width",
        type=int,
        default=300,
        help="Fixed character width per token in the synthetic text encoding.",
    )
    parser.add_argument("--honest", action="store_true", help="Enable honest signal roles.")
    parser.add_argument(
        "--json-summary",
        type=str,
        default="outputs/markov_boundary_chunker_honesty_summary.json",
        help="JSON summary output path.",
    )
    parser.add_argument(
        "--csv-summary",
        type=str,
        default="outputs/markov_boundary_chunker_honesty_summary.csv",
        help="CSV summary output path.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout too.")
    return parser.parse_args()


def _write_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()

    if str(args.device) == "auto":
        use_cuda = bool(torch.cuda.is_available())
    elif str(args.device) == "cpu":
        use_cuda = False
    else:
        use_cuda = True

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
        seed=int(args.seed),
        use_cuda=bool(use_cuda),
        torch_threads=0,
    )

    honest_policy = HonestChunkingPolicy(enabled=bool(args.honest))
    adaptive_config = AdaptiveChunkingConfig(
        enabled=True,
        min_chars=int(config.min_leaf_tokens) * int(args.token_char_width),
        max_chars=int(config.max_leaf_tokens) * int(args.token_char_width),
        low_info_expansion_weight=1.0,
        noise_expansion_weight=0.0,
        high_info_compression_weight=0.0,
        proxy_blend=0.0,
    )

    summary = run_markov_chunker_honesty_experiment(
        config,
        token_char_width=int(args.token_char_width),
        honest_policy=honest_policy,
        adaptive_config=adaptive_config,
    )

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

