#!/usr/bin/env python3
"""Quick convergence study: tree FNO vs flat FNO on the smoke benchmark.

Goal: verify that tree FNO (FNOCountSketch with tree merge) at least
converges to flat FNO (official_fno) performance with enough data.

Reports BOTH:
  1. Unweighted root MAE (the evaluation metric we care about)
  2. Law-level metrics: leaf MAE (C1), C2 count-drift MAE, merge MAE (C3),
     schedule spread — available for tree baselines via _eval_fno_model

Usage:
    python scripts/quick_fno_tree_convergence_study.py [--seeds 0 1 2] [--epochs 16]
    python scripts/quick_fno_tree_convergence_study.py --families official_fno tree_neural_c2 tree_neural
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# Metrics extracted from each run dict for the report.
_METRIC_KEYS = (
    "test_root_mae",
    "test_leaf_mae",
    "test_c2_count_drift_r1_mae",
    "test_merge_mae",
    "test_schedule_spread_mean",
)
_METRIC_PRETTY = {
    "test_root_mae": "Root MAE",
    "test_leaf_mae": "Leaf MAE (C1)",
    "test_c2_count_drift_r1_mae": "C2 Count Drift",
    "test_merge_mae": "Merge MAE (C3)",
    "test_schedule_spread_mean": "Sched. Spread",
}


def _collect(
    runs: list[dict], metric_key: str,
) -> dict[str, dict[int, list[float]]]:
    """Organize: {family: {train_count: [value_per_seed]}}."""
    legacy_metric_key = (
        "test_c2_idempotence_mae"
        if metric_key == "test_c2_count_drift_r1_mae"
        else metric_key
    )
    out: dict[str, dict[int, list[float]]] = {}
    for run in runs:
        family = str(run["baseline_family"])
        tc = int(run["train_doc_count"])
        val = float(run.get(metric_key, run.get(legacy_metric_key, float("nan"))))
        out.setdefault(family, {}).setdefault(tc, []).append(val)
    return out


def _print_table(
    results: dict[str, dict[int, list[float]]],
    *,
    metric_name: str,
) -> None:
    families = sorted(results.keys())
    if not families:
        return
    col_w = max(22, max(len(f) for f in families) + 4)
    header = f"{'train_docs':>12s}"
    for fam in families:
        header += f"  {fam:>{col_w}s}"
    print(f"\n  {metric_name}")
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    train_counts = sorted({tc for fam in results.values() for tc in fam})
    for tc in train_counts:
        row = f"  {tc:>12d}"
        for fam in families:
            vals = results.get(fam, {}).get(tc, [])
            if vals and any(np.isfinite(v) for v in vals):
                m = float(np.nanmean(vals))
                s = float(np.nanstd(vals))
                row += f"  {m:>{col_w - 8}.4f} ± {s:<5.4f}"
            else:
                row += f"  {'N/A':>{col_w}s}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-doc-counts", type=int, nargs="+", default=[8, 16, 32, 64, 128],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--families", nargs="+", default=["official_fno", "tree_neural_c2", "tree_neural"],
    )
    parser.add_argument("--benchmark", type=str, default="smoke")
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (
    run_markov_full_doc_anchor_diagnostics,
)
from src.ctreepo.sim.core.full_doc_config_codec import (
    runtime_config_overrides_from_config_like,
)

    output_dir = Path(args.output_dir) if args.output_dir else None

    config_overrides = runtime_config_overrides_from_config_like(
        {
            "n_epochs": args.epochs,
            "state_dim": args.state_dim,
            "hidden_dim": args.hidden_dim,
            "batch_size": args.batch_size,
            "lr": args.lr,
        }
    )

    print("=" * 72)
    print("FNO Tree Convergence Study")
    print("=" * 72)
    print(f"  benchmark:        {args.benchmark}")
    print(f"  families:         {args.families}")
    print(f"  train_doc_counts: {args.train_doc_counts}")
    print(f"  seeds:            {args.seeds}")
    print(f"  epochs={args.epochs}  state_dim={args.state_dim}  "
          f"hidden_dim={args.hidden_dim}  batch_size={args.batch_size}  lr={args.lr}")
    print("=" * 72)

    t0 = time.monotonic()

    payload = run_markov_full_doc_anchor_diagnostics(
        benchmark_name=args.benchmark,
        seeds=tuple(args.seeds),
        train_doc_counts=tuple(args.train_doc_counts),
        baseline_families=tuple(args.families),
        emit_confusion=False,
        output_dir=output_dir,
        use_cuda=args.use_cuda,
        torch_threads=1,
        config_overrides=config_overrides,
    )

    elapsed = time.monotonic() - t0

    # Print tables for each metric.
    print()
    print("=" * 72)
    print("RESULTS (mean ± std across seeds)")
    print("=" * 72)

    for metric_key in _METRIC_KEYS:
        results = _collect(payload["runs"], metric_key)
        # Skip metrics that are all NaN/zero for every family.
        all_vals = [
            v for fam in results.values() for tc_vals in fam.values() for v in tc_vals
        ]
        if all_vals and any(np.isfinite(v) and abs(v) > 1e-12 for v in all_vals):
            _print_table(results, metric_name=_METRIC_PRETTY.get(metric_key, metric_key))

    # Convergence summary: compare root MAE at largest train size.
    root_results = _collect(payload["runs"], "test_root_mae")
    families = sorted(root_results.keys())
    train_counts = sorted({tc for fam in root_results.values() for tc in fam})
    if train_counts:
        last_tc = train_counts[-1]
        print()
        print(f"--- Convergence at largest train size ({last_tc} docs) ---")
        for fam in families:
            vals = root_results.get(fam, {}).get(last_tc, [])
            if vals:
                print(f"  {fam:30s}  root MAE = {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        # Pairwise gaps vs first family (flat FNO).
        if len(families) >= 2:
            ref_fam = families[0]
            ref_vals = root_results.get(ref_fam, {}).get(last_tc, [])
            ref_mean = float(np.mean(ref_vals)) if ref_vals else float("nan")
            for fam in families[1:]:
                fam_vals = root_results.get(fam, {}).get(last_tc, [])
                fam_mean = float(np.mean(fam_vals)) if fam_vals else float("nan")
                if np.isfinite(ref_mean) and np.isfinite(fam_mean):
                    gap_pct = 100.0 * (fam_mean - ref_mean) / max(ref_mean, 1e-9)
                    print(f"  {fam} vs {ref_fam}: {gap_pct:+.1f}%")

    print(f"\nTotal elapsed: {elapsed:.1f}s")

    # Save JSON summary.
    if output_dir:
        summary = {
            "config": {
                "benchmark": args.benchmark,
                "families": args.families,
                "train_doc_counts": args.train_doc_counts,
                "seeds": args.seeds,
                "epochs": args.epochs,
                "state_dim": args.state_dim,
                "hidden_dim": args.hidden_dim,
            },
            "metrics": {},
        }
        for metric_key in _METRIC_KEYS:
            results = _collect(payload["runs"], metric_key)
            summary["metrics"][metric_key] = {
                fam: {
                    str(tc): {
                        "mean": float(np.nanmean(vals)),
                        "std": float(np.nanstd(vals)),
                        "values": [float(v) for v in vals],
                    }
                    for tc, vals in sorted(counts.items())
                }
                for fam, counts in results.items()
            }
        summary["elapsed_seconds"] = elapsed
        summary_path = output_dir / "convergence_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Summary written to: {summary_path}")

    print()


if __name__ == "__main__":
    main()
