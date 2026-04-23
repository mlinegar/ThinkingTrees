#!/usr/bin/env python3
"""FNO tree labeling study: unified IPW node-level supervision.

Runs the FNO tree model under different node sampling regimes to understand
what it takes to learn g both globally (root) and locally (nodes).

Phase 1: Full sampling baseline (all nodes supervised, equal weight)
Phase 2: Root-only baseline (no local supervision)
Phase 3: Partial sampling curves (vary leaf/internal sample rates)
Phase 4: Old law-based approach for comparison
"""
from __future__ import annotations

import json
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
    c3_audit_strategy="uniform",
    include_root_query=True,
    include_fno_baseline=False,
    use_cuda=False,
    torch_threads=2,
    seed=42,
)


def _ipw_config(
    leaf_sample_rate: float = 1.0,
    internal_sample_rate: float = 1.0,
    include_fno_baseline: bool = False,
    use_residual_decomposition: bool = True,
) -> OPSCountConfig:
    return OPSCountConfig(
        **_SHARED,
        use_unified_ipw=True,
        ipw_leaf_sample_rate=leaf_sample_rate,
        ipw_internal_sample_rate=internal_sample_rate,
        include_fno_baseline=include_fno_baseline,
        use_residual_decomposition=use_residual_decomposition,
    )


def _law_config(
    law_package: str,
    local_law_weight: float = 0.5,
) -> OPSCountConfig:
    return OPSCountConfig(
        **_SHARED,
        use_unified_ipw=False,
        law_package=law_package,
        local_law_weight=local_law_weight,
        leaf_query_rate=1.0,
        audit_policy="fraction",
        audit_fraction=1.0,
    )


CONFIGS = {
    # Residual decomposition — full sampling (main experiment).
    "ipw_residual_full": _ipw_config(
        include_fno_baseline=True, use_residual_decomposition=True,
    ),
    # Flat IPW — full sampling (baseline comparison).
    "ipw_flat_full": _ipw_config(use_residual_decomposition=False),
    # Residual with partial leaf sampling.
    "ipw_residual_leaf50": _ipw_config(
        leaf_sample_rate=0.5, use_residual_decomposition=True,
    ),
    # Old law-based approach for reference.
    "laws_all": _law_config("all_laws"),
}


def _run_one(name: str, cfg: OPSCountConfig) -> dict:
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    out = run_markov_changepoint_ops_count_experiment(cfg)
    elapsed = time.time() - t0

    learned = out.metrics["learned"]
    r = {
        "test_root_mae": float(learned["root_mae"]),
        "test_leaf_mae": float(learned["leaf_mae"]),
        "test_merge_mae": float(learned["merge_mae"]),
        "train_root_mae": float(learned["train_root_mae"]),
        "epochs": int(learned["epochs_completed"]),
        "best_epoch": int(learned["training_selection_best_epoch"]),
        "wall_seconds": round(elapsed, 1),
    }

    doc_fno = out.metrics.get("fno", {})
    if doc_fno:
        r["doc_fno_test_root_mae"] = float(doc_fno.get("root_mae", float("nan")))

    print(f"  test_root_mae={r['test_root_mae']:.4f}  "
          f"train_root_mae={r['train_root_mae']:.4f}  "
          f"wall={r['wall_seconds']}s")
    return r


def main():
    results = {}
    for name, cfg in CONFIGS.items():
        results[name] = _run_one(name, cfg)

    # Summary table.
    print(f"\n{'='*70}")
    print("IPW LABELING STUDY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'Test Root MAE':>14} {'Leaf MAE':>10} "
          f"{'Merge MAE':>10} {'Best Epoch':>10}")
    print("-" * 73)
    for name, r in results.items():
        print(f"{name:<25} {r['test_root_mae']:>14.4f} {r['test_leaf_mae']:>10.4f} "
              f"{r['test_merge_mae']:>10.4f} {r['best_epoch']:>10}")

    for name, r in results.items():
        if "doc_fno_test_root_mae" in r:
            print(f"\nDoc-level FNO baseline: test_root_mae={r['doc_fno_test_root_mae']:.4f}")
            break

    with open("outputs/fno_tree_ipw_labeling_study.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to outputs/fno_tree_ipw_labeling_study.json")


if __name__ == "__main__":
    main()
