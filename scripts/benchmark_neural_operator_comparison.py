#!/usr/bin/env python3
"""
Standalone benchmark: compare neural operator baselines on the Markov
changepoint-count recoverable task.

Runs all available baselines on the same fixed data bundle and prints
a comparison table.

Usage:
    python scripts/benchmark_neural_operator_comparison.py \
        --data-bundle outputs/markov_observed_token_recoverable_v4/markov_data/observed_token_bundle.json \
        --device cpu \
        --n-epochs 50 \
        --output outputs/neural_operator_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.ctreepo.sim.core.markov_changepoint_ops_count import (
    MarkovOPSDataBundle,
    OPSCountConfig,
    _eval_root_predictions,
    _exact_match_rate,
    _token_sequence_arrays,
)
from src.ctreepo.sim.core.markov_neural_operator_baselines import (
    HAS_NEURAL_OPERATOR,
    _fit_cnn1d_baseline,
    _fit_deeponet_baseline,
    _fit_mlp_bigram_baseline,
)


def _ridge_bigram_baseline(
    train_docs, val_docs, test_docs, *, vocab_size: int
) -> dict:
    """Ridge regression on bigram features — the exact-recovery reference."""
    from sklearn.linear_model import RidgeCV

    from src.ctreepo.sim.core.markov_neural_operator_baselines import (
        _bigram_features_from_tokens,
    )

    pad_id = int(vocab_size)
    train_tokens, train_mask, train_y = _token_sequence_arrays(train_docs, pad_id=pad_id)
    val_tokens, val_mask, val_y = _token_sequence_arrays(val_docs, pad_id=pad_id)
    test_tokens, test_mask, test_y = _token_sequence_arrays(test_docs, pad_id=pad_id)

    train_feat = _bigram_features_from_tokens(train_tokens, train_mask, vocab_size=vocab_size)
    test_feat = _bigram_features_from_tokens(test_tokens, test_mask, vocab_size=vocab_size)

    model = RidgeCV(alphas=[1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0])
    model.fit(train_feat, train_y)
    test_pred = model.predict(test_feat)
    test_mae = float(np.mean(np.abs(test_pred - test_y.astype(np.float64))))
    test_exact = float(np.mean((np.rint(test_pred) == np.rint(test_y)).astype(np.float64)))
    return {
        "test_root_mae": test_mae,
        "test_exact_match_rate": test_exact,
        "alpha": float(model.alpha_),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural operator comparison benchmark")
    parser.add_argument("--data-bundle", required=True, help="Path to saved data bundle JSON")
    parser.add_argument("--device", default="cpu", help="torch device (cpu or cuda:N)")
    parser.add_argument("--n-epochs", type=int, default=50, help="Training epochs for all baselines")
    parser.add_argument("--state-dim", type=int, default=64, help="State/embedding dim")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dim")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--output", default="", help="Path to write results JSON")
    parser.add_argument(
        "--skip-fno", action="store_true",
        help="Skip FNO baseline (e.g. if neuraloperator not installed)",
    )
    args = parser.parse_args()

    bundle_path = Path(args.data_bundle)
    if not bundle_path.exists():
        print(f"ERROR: bundle not found: {bundle_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data bundle: {bundle_path}")
    bundle = MarkovOPSDataBundle.load(bundle_path)
    print(f"  train: {len(bundle.train_docs)} docs")
    print(f"  val:   {len(bundle.val_docs)} docs")
    print(f"  test:  {len(bundle.test_docs)} docs")

    device = torch.device(args.device)
    config = OPSCountConfig(
        n_epochs=args.n_epochs,
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    seeds = {"effective_model_seed": 42}

    results: dict = {}

    # 1. Ridge bigram (reference)
    print("\n--- Ridge bigram (reference) ---")
    t0 = time.time()
    ridge_result = _ridge_bigram_baseline(
        bundle.train_docs, bundle.val_docs, bundle.test_docs,
        vocab_size=int(config.vocab_size),
    )
    dt = time.time() - t0
    results["ridge_bigram"] = {**ridge_result, "wall_seconds": round(dt, 2)}
    print(f"  test_root_mae:        {ridge_result['test_root_mae']:.8f}")
    print(f"  test_exact_match:     {ridge_result['test_exact_match_rate']:.6f}")
    print(f"  time:                 {dt:.1f}s")

    # 2. MLP bigram (critical diagnostic)
    print("\n--- MLP on bigram features (critical diagnostic) ---")
    t0 = time.time()
    mlp_train, mlp_val, mlp_test, mlp_fit = _fit_mlp_bigram_baseline(
        config=config, seeds=seeds, device=device,
        train_docs=bundle.train_docs, val_docs=bundle.val_docs, test_docs=bundle.test_docs,
    )
    dt = time.time() - t0
    results["mlp_bigram"] = {
        "test_root_mae": mlp_test.root_mae,
        "test_exact_match_rate": mlp_fit.test_exact_match_rate,
        "best_epoch": mlp_fit.best_epoch,
        "wall_seconds": round(dt, 2),
    }
    print(f"  test_root_mae:        {mlp_test.root_mae:.8f}")
    print(f"  test_exact_match:     {mlp_fit.test_exact_match_rate:.6f}")
    print(f"  best_epoch:           {mlp_fit.best_epoch}")
    print(f"  time:                 {dt:.1f}s")

    # 3. CNN1D
    print("\n--- 1D CNN (kernel_size=2 transition detector) ---")
    t0 = time.time()
    cnn_train, cnn_val, cnn_test, cnn_fit = _fit_cnn1d_baseline(
        config=config, seeds=seeds, device=device,
        train_docs=bundle.train_docs, val_docs=bundle.val_docs, test_docs=bundle.test_docs,
    )
    dt = time.time() - t0
    results["cnn1d"] = {
        "test_root_mae": cnn_test.root_mae,
        "test_exact_match_rate": cnn_fit.test_exact_match_rate,
        "best_epoch": cnn_fit.best_epoch,
        "wall_seconds": round(dt, 2),
    }
    print(f"  test_root_mae:        {cnn_test.root_mae:.8f}")
    print(f"  test_exact_match:     {cnn_fit.test_exact_match_rate:.6f}")
    print(f"  best_epoch:           {cnn_fit.best_epoch}")
    print(f"  time:                 {dt:.1f}s")

    # 4. DeepONet
    print("\n--- DeepONet (branch-net only) ---")
    t0 = time.time()
    don_train, don_val, don_test, don_fit = _fit_deeponet_baseline(
        config=config, seeds=seeds, device=device,
        train_docs=bundle.train_docs, val_docs=bundle.val_docs, test_docs=bundle.test_docs,
    )
    dt = time.time() - t0
    results["deeponet"] = {
        "test_root_mae": don_test.root_mae,
        "test_exact_match_rate": don_fit.test_exact_match_rate,
        "best_epoch": don_fit.best_epoch,
        "wall_seconds": round(dt, 2),
    }
    print(f"  test_root_mae:        {don_test.root_mae:.8f}")
    print(f"  test_exact_match:     {don_fit.test_exact_match_rate:.6f}")
    print(f"  best_epoch:           {don_fit.best_epoch}")
    print(f"  time:                 {dt:.1f}s")

    # 5. FNO (requires neuraloperator)
    if not args.skip_fno and HAS_NEURAL_OPERATOR:
        from src.ctreepo.sim.core.markov_neural_operator_baselines import _fit_fno_baseline

        print("\n--- FNO (official neuraloperator package) ---")
        t0 = time.time()
        fno_train, fno_val, fno_test, fno_fit = _fit_fno_baseline(
            config=config, seeds=seeds, device=device,
            train_docs=bundle.train_docs, val_docs=bundle.val_docs, test_docs=bundle.test_docs,
        )
        dt = time.time() - t0
        results["fno"] = {
            "test_root_mae": fno_test.root_mae,
            "test_exact_match_rate": fno_fit.test_exact_match_rate,
            "best_epoch": fno_fit.best_epoch,
            "wall_seconds": round(dt, 2),
        }
        print(f"  test_root_mae:        {fno_test.root_mae:.8f}")
        print(f"  test_exact_match:     {fno_fit.test_exact_match_rate:.6f}")
        print(f"  best_epoch:           {fno_fit.best_epoch}")
        print(f"  time:                 {dt:.1f}s")
    elif args.skip_fno:
        print("\n--- FNO: SKIPPED (--skip-fno) ---")
    else:
        print("\n--- FNO: SKIPPED (neuraloperator not installed) ---")

    # Summary table
    print("\n" + "=" * 72)
    print(f"{'Baseline':<30} {'Test MAE':>12} {'Exact Match':>12}")
    print("-" * 72)
    ref_mae = results.get("ridge_bigram", {}).get("test_root_mae", float("nan"))
    for label in ["ridge_bigram", "mlp_bigram", "cnn1d", "deeponet", "fno"]:
        if label not in results:
            continue
        r = results[label]
        mae = r["test_root_mae"]
        em = r["test_exact_match_rate"]
        print(f"  {label:<28} {mae:>12.6f} {em:>12.6f}")
    print("=" * 72)
    print(f"\nFor reference, CTreePO operator best: root_mae = 0.277")

    # Write results
    if args.output.strip():
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results["config"] = {
            "n_epochs": args.n_epochs,
            "state_dim": args.state_dim,
            "hidden_dim": args.hidden_dim,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": args.device,
            "data_bundle": str(bundle_path),
        }
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
