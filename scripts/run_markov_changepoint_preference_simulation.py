#!/usr/bin/env python3
"""Run Markov changepoint preference-learning gap simulation."""

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
from src.tree.markov_changepoint_preference_simulation import (  # noqa: E402
    MarkovChangepointPreferenceConfig,
    run_markov_changepoint_preference_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Markov changepoint preference gap simulation."
    )

    parser.add_argument("--n-regimes", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=96)
    parser.add_argument("--min-tokens", type=int, default=384)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--min-segments", type=int, default=12)
    parser.add_argument("--max-segments", type=int, default=24)
    parser.add_argument("--min-seg-len", type=int, default=8)
    parser.add_argument("--max-seg-len", type=int, default=32)

    parser.add_argument("--min-leaf-tokens", type=int, default=8)
    parser.add_argument("--max-leaf-tokens", type=int, default=32)
    parser.add_argument("--fixed-leaf-tokens", type=int, default=16)
    parser.add_argument("--token-char-width", type=int, default=300)
    parser.add_argument("--boundary-tolerance-tokens", type=int, default=2)

    parser.add_argument("--train-docs", type=int, default=1000)
    parser.add_argument("--test-docs", type=int, default=1000)
    parser.add_argument("--sinkhorn-iters", type=int, default=30)
    parser.add_argument("--transition-log-std", type=float, default=1.25)

    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--boundary-emb-dim", type=int, default=96)
    parser.add_argument("--boundary-hidden-dim", type=int, default=256)
    parser.add_argument("--boundary-batch-size", type=int, default=256)
    parser.add_argument("--boundary-max-train-samples", type=int, default=60000)
    parser.add_argument(
        "--balance-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance positive/negative boundary examples during training.",
    )
    parser.add_argument(
        "--positive-class-weight",
        type=float,
        default=None,
        help="Override BCE positive-class weight (default auto from class ratio).",
    )
    parser.add_argument(
        "--calibrate-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply a Bayes prior logit shift so predicted boundary probs reflect the true boundary rate.",
    )
    parser.add_argument(
        "--calibrate-pos-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Undo the training-time BCE pos_weight logit shift so probabilities are comparable across runs.",
    )

    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    parser.add_argument("--dpo-beta", type=float, default=1.0)
    parser.add_argument("--dpo-policy-sharpness", type=float, default=2.5)
    parser.add_argument("--dpo-negatives-per-doc", type=int, default=8)

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
        "--honest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable boundary/evaluation role split in chunk feedback.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="axis",
        choices=["axis", "sentence", "paragraph"],
        help="Chunking strategy passed to chunk_for_ops.",
    )
    parser.add_argument("--low-info-expansion-weight", type=float, default=1.0)
    parser.add_argument("--noise-expansion-weight", type=float, default=0.0)
    parser.add_argument("--high-info-compression-weight", type=float, default=1.0)
    parser.add_argument("--proxy-blend", type=float, default=0.0)

    parser.add_argument(
        "--json-summary",
        type=str,
        default="outputs/markov_changepoint_preference/train_1000_seed_0.json",
        help="JSON summary output path.",
    )
    parser.add_argument(
        "--csv-summary",
        type=str,
        default="outputs/markov_changepoint_preference/train_1000_seed_0.csv",
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

    config = MarkovChangepointPreferenceConfig(
        n_regimes=int(args.n_regimes),
        vocab_size=int(args.vocab_size),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        min_segments=int(args.min_segments),
        max_segments=int(args.max_segments),
        min_seg_len=int(args.min_seg_len),
        max_seg_len=int(args.max_seg_len),
        min_leaf_tokens=int(args.min_leaf_tokens),
        max_leaf_tokens=int(args.max_leaf_tokens),
        fixed_leaf_tokens=int(args.fixed_leaf_tokens),
        token_char_width=int(args.token_char_width),
        boundary_tolerance_tokens=int(args.boundary_tolerance_tokens),
        train_docs=int(args.train_docs),
        test_docs=int(args.test_docs),
        sinkhorn_iters=int(args.sinkhorn_iters),
        transition_log_std=float(args.transition_log_std),
        window_size=int(args.window_size),
        boundary_emb_dim=int(args.boundary_emb_dim),
        boundary_hidden_dim=int(args.boundary_hidden_dim),
        boundary_batch_size=int(args.boundary_batch_size),
        boundary_max_train_samples=int(args.boundary_max_train_samples),
        balance_training=bool(args.balance_training),
        positive_class_weight=args.positive_class_weight,
        calibrate_prior=bool(args.calibrate_prior),
        calibrate_pos_weight=bool(args.calibrate_pos_weight),
        n_epochs=int(args.n_epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip_norm),
        dpo_beta=float(args.dpo_beta),
        dpo_policy_sharpness=float(args.dpo_policy_sharpness),
        dpo_negatives_per_doc=int(args.dpo_negatives_per_doc),
        seed=int(args.seed),
        use_cuda=bool(use_cuda),
        cuda_device=args.cuda_device,
        torch_threads=int(args.torch_threads),
    )

    honest_policy = HonestChunkingPolicy(enabled=bool(args.honest))
    adaptive_config = AdaptiveChunkingConfig(
        enabled=True,
        min_chars=int(config.min_leaf_tokens) * int(config.token_char_width),
        max_chars=int(config.max_leaf_tokens) * int(config.token_char_width),
        low_info_expansion_weight=float(args.low_info_expansion_weight),
        noise_expansion_weight=float(args.noise_expansion_weight),
        high_info_compression_weight=float(args.high_info_compression_weight),
        proxy_blend=float(args.proxy_blend),
    )

    summary = run_markov_changepoint_preference_experiment(
        config,
        honest_policy=honest_policy,
        adaptive_config=adaptive_config,
        strategy=str(args.strategy),
    )

    json_path = Path(args.json_summary)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(summary.to_json(), encoding="utf-8")

    row = {f"config_{k}": v for k, v in summary.config.items()}
    row["action_max_count"] = summary.action_max_count
    row["boundary_model_train_loss_final"] = summary.boundary_model_train_loss_final
    row["boundary_train_positive_rate"] = summary.boundary_train_positive_rate
    for policy, metrics in summary.metrics.items():
        d = asdict(metrics)
        for k, v in d.items():
            row[f"{policy}_{k}"] = v
    _write_csv(Path(args.csv_summary), row)

    print(f"wrote_json | {json_path}")
    print(f"wrote_csv | {Path(args.csv_summary)}")
    for policy in ("fixed", "chunker_honest", "chunker_leaky", "oracle_cut"):
        if policy not in summary.metrics:
            continue
        m = summary.metrics[policy]
        print(
            "policy={} | f1={:.6f} | ratio={:.3f} | abs_err={:.3f} | dpo_gap={:.6f}".format(
                policy,
                m.boundary_f1,
                m.predicted_to_true_ratio,
                m.mean_abs_count_error,
                m.mean_dpo_loss_gap_to_opt,
            )
        )

    if bool(args.json):
        print(summary.to_json())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
