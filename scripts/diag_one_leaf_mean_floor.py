#!/usr/bin/env python3
"""Cheap analytic diagnostic: does the one-leaf 3.7 plateau equal the
mean-predictor MAE on recoverable_v5_t2048?

No training. Just loads the same data the probe uses and computes:
  - target distribution (mean, std, min, max)
  - constant mean / median predictor MAE on val and test
  - exact palette-witness MAE (sanity)
  - palette-block-count "smart constant" predictor (predict via doc-level
    statistic that is reachable by a sum-pool readout)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.probe_clean_unified_no import (  # noqa: E402
    _constant_baseline_diagnostics,
    _palette_block_exact_root_mae,
    _load_split_docs,
    _root_truths,
    _palette_block_bigram_features,
)
def main() -> None:
    benchmark = "recoverable_v5_t2048"
    train_docs_n = 1024
    leaf_tokens = 2048

    print(f"loading {benchmark}: train_docs={train_docs_n}, leaf_tokens={leaf_tokens}")
    train, val, test = _load_split_docs(
        benchmark=benchmark,
        train_docs=train_docs_n,
        fixed_leaf_tokens=leaf_tokens,
        seed=0,
    )

    # recoverable_v5_t2048 has vocab=16, n_regimes=4 (matches saved summaries)
    vocab_size = 16
    n_regimes = 4
    print(f"  vocab_size={vocab_size}  n_regimes={n_regimes}")
    print(f"  splits: train={len(train)}  val={len(val)}  test={len(test)}")

    # Target distribution
    for name, docs in [("train", train), ("val", val), ("test", test)]:
        t = _root_truths(docs)
        print(f"  {name}: count mean={t.mean():.3f} std={t.std():.3f} "
              f"min={t.min():.0f} max={t.max():.0f}  "
              f"E[|N-mean|]={np.mean(np.abs(t - t.mean())):.4f}")

    # Mean / median predictor (already in probe)
    base = _constant_baseline_diagnostics(
        train_docs=train, val_docs=val, test_docs=test
    )
    print("\nconstant baselines:")
    print(f"  train_mean_count={base['train_mean_count']:.4f}  "
          f"train_median_count={base['train_median_count']:.4f}")
    for split in ("val", "test"):
        s = base["splits"][split]
        print(f"  {split} train-mean-predictor MAE = "
              f"{s['train_mean_predictor']['root_mae']:.4f}")
        print(f"  {split} split-mean-predictor MAE = "
              f"{s['split_mean_predictor']['root_mae']:.4f}")
        print(f"  {split} train-median-predictor MAE = "
              f"{s['train_median_predictor']['root_mae']:.4f}")

    # Exact witness
    print("\nexact palette-witness:")
    for name, docs in [("val", val), ("test", test)]:
        w = _palette_block_exact_root_mae(
            docs, vocab_size=vocab_size, n_regimes=n_regimes
        )
        print(f"  {name}: n={w['n']}  mae={w['mae']:.6f}  "
              f"max_abs_error={w['max_abs_error']:.4f}")

    # Palette-block bigram least-squares
    # This is the BEST a sum-pool readout can do if the embedding perfectly
    # one-hots palette blocks: linear over per-position bigram indicators.
    # Closed-form OLS on train, evaluated on test.
    print("\npalette-block bigram OLS (best sum-pool-of-block-bigrams predictor):")
    train_X = _palette_block_bigram_features(
        train, vocab_size=vocab_size, n_regimes=n_regimes
    )
    test_X = _palette_block_bigram_features(
        test, vocab_size=vocab_size, n_regimes=n_regimes
    )
    train_y = _root_truths(train)
    test_y = _root_truths(test)
    # Add bias column
    train_Xb = np.concatenate([train_X, np.ones((train_X.shape[0], 1))], axis=1)
    test_Xb = np.concatenate([test_X, np.ones((test_X.shape[0], 1))], axis=1)
    w, *_ = np.linalg.lstsq(train_Xb, train_y, rcond=None)
    train_pred = train_Xb @ w
    test_pred = test_Xb @ w
    print(f"  train MAE={np.mean(np.abs(train_pred - train_y)):.6f}")
    print(f"  test  MAE={np.mean(np.abs(test_pred - test_y)):.6f}")
    print(f"  test  max_abs_error={np.max(np.abs(test_pred - test_y)):.6f}")
    print(f"  fitted weight stats: mean={w.mean():.3f} std={w.std():.3f} "
          f"max_abs={np.max(np.abs(w)):.3f}")

    # Save JSON
    out = REPO / "outputs" / "diag_one_leaf_mean_floor.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": benchmark,
        "leaf_tokens": leaf_tokens,
        "train_docs": train_docs_n,
        "vocab_size": vocab_size,
        "n_regimes": n_regimes,
        "constant_baselines": base,
        "exact_witness": {
            "val": _palette_block_exact_root_mae(
                val, vocab_size=vocab_size, n_regimes=n_regimes
            ),
            "test": _palette_block_exact_root_mae(
                test, vocab_size=vocab_size, n_regimes=n_regimes
            ),
        },
        "palette_bigram_ols": {
            "train_mae": float(np.mean(np.abs(train_pred - train_y))),
            "test_mae": float(np.mean(np.abs(test_pred - test_y))),
            "test_max_abs_error": float(np.max(np.abs(test_pred - test_y))),
        },
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: float(o))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
