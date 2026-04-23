#!/usr/bin/env python3
"""Quick comparison: FNO tree with/without local laws vs doc-level FNO baseline.

Runs three configs on the same data:
  1. tree_root_only  – FNO tree, root supervision only (no laws)
  2. tree_all_laws   – FNO tree, C1+C2+C3 local law supervision
  3. tree_c2_only    – FNO tree, C2-only (best single law from prior experiments)

Compares root MAE (the fair comparison metric) across all three,
plus the doc-level FNO baseline that runs automatically.
"""
from __future__ import annotations

import json
import sys
import time

from src.ctreepo.sim.core.markov_changepoint_ops_count import (
    OPSCountConfig,
    run_markov_changepoint_ops_count_experiment,
)

# Shared data config — moderate size, fast on CPU.
_SHARED = dict(
    n_regimes=4,
    vocab_size=32,
    min_tokens=64,
    max_tokens=64,
    min_segments=4,
    max_segments=8,
    min_seg_len=4,
    max_seg_len=16,
    fixed_leaf_tokens=16,
    train_docs=128,
    val_docs=32,
    test_docs=64,
    feature_mode="full",
    model_family="neural",
    state_dim=16,
    hidden_dim=32,
    n_epochs=30,
    batch_size=16,
    lr=1e-3,
    weight_decay=1e-5,
    fno_width=32,
    fno_n_modes=8,
    fno_n_layers=2,
    audit_policy="fraction",
    audit_fraction=0.5,
    c3_audit_strategy="uniform",
    leaf_query_rate=1.0,
    include_root_query=True,
    include_fno_baseline=True,
    use_cuda=False,
    torch_threads=2,
    seed=42,
)

CONFIGS = {
    "tree_root_only": OPSCountConfig(
        **_SHARED,
        law_package="root_only",
    ),
    "tree_c2_only": OPSCountConfig(
        **_SHARED,
        law_package="c2_only",
        local_law_weight=0.5,
    ),
    "tree_all_laws": OPSCountConfig(
        **_SHARED,
        law_package="all_laws",
        local_law_weight=0.5,
    ),
}


def main():
    results = {}
    for name, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        t0 = time.time()
        out = run_markov_changepoint_ops_count_experiment(cfg)
        elapsed = time.time() - t0

        learned = out.metrics["learned"]
        obj = out.objective

        r = {
            "test_root_mae": float(learned["root_mae"]),
            "test_leaf_mae": float(learned["leaf_mae"]),
            "test_merge_mae": float(learned["merge_mae"]),
            "test_c2_mae": float(learned["c2_idempotence_mae"]),
            "train_root_mae": float(learned["train_root_mae"]),
            "train_leaf_mae": float(learned["train_leaf_mae"]),
            "train_merge_mae": float(learned["train_merge_mae"]),
            "epochs": int(learned["epochs_completed"]),
            "best_epoch": int(learned["training_selection_best_epoch"]),
            "c1_weight": float(obj["local_law_c1_weight"]),
            "c2_weight": float(obj["local_law_c2_weight"]),
            "c3_weight": float(obj["local_law_c3_weight"]),
            "wall_seconds": round(elapsed, 1),
        }

        # Doc-level FNO baseline (full-document supervision).
        doc_fno = out.metrics.get("fno", {})
        doc_fno_train = out.metrics.get("fno_train", {})
        if doc_fno:
            r["doc_fno_test_root_mae"] = float(doc_fno.get("root_mae", float("nan")))
        if doc_fno_train:
            r["doc_fno_train_root_mae"] = float(doc_fno_train.get("root_mae", float("nan")))

        # Training diagnostics.
        fno_training = out.metrics.get("fno_training", {})
        if fno_training:
            r["doc_fno_best_epoch"] = int(fno_training.get("best_epoch", -1))

        # Loss curve (last 5 values).
        learned_meta = out.metrics.get("learned", {})
        if "training_selection_metric_curve" in learned_meta:
            curve = learned_meta["training_selection_metric_curve"]
            if curve:
                r["loss_curve_last5"] = [round(float(v), 4) for v in list(curve)[-5:]]

        results[name] = r
        print(f"  test_root_mae={r['test_root_mae']:.4f}  "
              f"train_root_mae={r['train_root_mae']:.4f}  "
              f"wall={r['wall_seconds']}s")

    # Summary table.
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'Test Root MAE':>14} {'Train Root MAE':>15} "
          f"{'Leaf MAE':>10} {'Merge MAE':>10} {'C2 MAE':>10}")
    print("-" * 85)
    for name, r in results.items():
        print(f"{name:<20} {r['test_root_mae']:>14.4f} {r['train_root_mae']:>15.4f} "
              f"{r['test_leaf_mae']:>10.4f} {r['test_merge_mae']:>10.4f} {r['test_c2_mae']:>10.4f}")

    # Doc-level baseline for reference.
    for name, r in results.items():
        if "doc_fno_test_root_mae" in r:
            print(f"\nDoc-level FNO baseline (from {name}):")
            print(f"  test_root_mae={r['doc_fno_test_root_mae']:.4f}")
            if "doc_fno_train_root_mae" in r:
                print(f"  train_root_mae={r['doc_fno_train_root_mae']:.4f}")
            if "doc_fno_best_epoch" in r:
                print(f"  best_epoch={r['doc_fno_best_epoch']}")
            break

    # Training diagnostics.
    for name, r in results.items():
        if "loss_curve_last5" in r:
            print(f"\n{name} loss curve (last 5): {r['loss_curve_last5']}")

    # Save results.
    with open("outputs/fno_tree_law_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to outputs/fno_tree_law_comparison.json")


if __name__ == "__main__":
    main()
