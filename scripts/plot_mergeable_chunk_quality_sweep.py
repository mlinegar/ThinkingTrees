#!/usr/bin/env python3
"""Plot chunk-quality and leaf-granularity sweeps for generic-k recovery."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.mergeable_ablation import (
    ChunkerPolicy,
    SelectorPolicy,
    SpikeCountMixtureDistributionSpec,
    run_chunk_quality_sweep,
)
from src.ctreepo.sim.objective_semantics import mergeable_probability_target_objective_semantics


def _parse_int_csv(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def _parse_float_csv(s: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def _parse_str_csv(s: str) -> tuple[str, ...]:
    out = tuple(x.strip() for x in s.split(",") if x.strip())
    if len(out) == 0:
        raise ValueError("expected a non-empty string CSV")
    return out


def _parse_chunker_policy(s: str) -> ChunkerPolicy:
    val = s.strip().lower()
    mapping = {
        "fixed": ChunkerPolicy.FIXED,
        "adaptive-aligned": ChunkerPolicy.ADAPTIVE_ALIGNED,
        "adaptive_misspecified": ChunkerPolicy.ADAPTIVE_MISSPECIFIED,
        "adaptive-misspecified": ChunkerPolicy.ADAPTIVE_MISSPECIFIED,
    }
    if val not in mapping:
        raise ValueError(f"Unsupported chunker={s!r}")
    return mapping[val]


def _parse_selector_policy(s: str) -> SelectorPolicy:
    val = s.strip().lower()
    mapping = {
        "all": SelectorPolicy.ALL,
        "top-proxy": SelectorPolicy.TOP_PROXY,
        "bottom-proxy": SelectorPolicy.BOTTOM_PROXY,
        "random": SelectorPolicy.RANDOM,
    }
    if val not in mapping:
        raise ValueError(f"Unsupported selector={s!r}")
    return mapping[val]


def _bias_ci(row: Dict[str, object], z: float = 1.96) -> tuple[float, float, float]:
    """Approximate CI for mean signed bias from rmse^2 = var + bias^2."""
    n_rep = max(1, int(row["n_replicates"]))
    bias = float(row["bias"])
    rmse = float(row["rmse"])
    var = max(0.0, (rmse * rmse) - (bias * bias))
    se = math.sqrt(var / float(n_rep))
    return se, bias - z * se, bias + z * se


def _xsize(row: Dict[str, object]) -> int:
    chunker = str(row["chunker"])
    if chunker == ChunkerPolicy.FIXED.value:
        return int(row["fixed_chunk_size"])
    return int(row["max_chunk_size"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep chunk leaf granularity and budget; report bias plus chunk-quality diagnostics."
        )
    )
    parser.add_argument("--p-spike-doc", type=float, default=0.62)
    parser.add_argument("--p-boundary-given-spike", type=float, default=0.35)
    parser.add_argument("--spike-count-support", type=str, default="1,2,3,4,5")
    parser.add_argument("--spike-count-probs", type=str, default="0.10,0.20,0.25,0.25,0.20")
    parser.add_argument("--target-k", type=int, default=5)
    parser.add_argument("--sketch-order", type=int, default=None)
    parser.add_argument("--chunk-sizes", type=str, default="1,2,4,8,16")
    parser.add_argument("--chunk-budgets", type=str, default="1,2,3,4,6,8")
    parser.add_argument(
        "--chunker",
        type=str,
        default="fixed",
        help="fixed | adaptive-aligned | adaptive-misspecified",
    )
    parser.add_argument(
        "--selector",
        type=str,
        default="top-proxy",
        help="all | top-proxy | bottom-proxy | random",
    )
    parser.add_argument("--n-tokens", type=int, default=32)
    parser.add_argument("--proxy-noise", type=float, default=0.12)
    parser.add_argument("--boundary-span-tokens", type=int, default=4)
    parser.add_argument("--n-replicates", type=int, default=120)
    parser.add_argument("--docs-per-replicate", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--weighting-modes",
        type=str,
        default="doc,leaf,token",
        help="Comma-separated weighting modes for side-by-side reporting.",
    )
    parser.add_argument(
        "--legacy-weighting-mode",
        type=str,
        default="doc",
        choices=("doc", "leaf", "token"),
        help="Explicit label for legacy scalar fields.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/mergeable_chunk_quality_sweep.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--json-summary",
        type=str,
        default="outputs/mergeable_chunk_quality_sweep_summary.json",
        help="JSON summary output path.",
    )
    return parser.parse_args()


def _matrix_from_rows(
    rows: List[Dict[str, object]],
    *,
    budgets: List[int],
    sizes: List[int],
    metric: str,
) -> np.ndarray:
    out = np.full((len(budgets), len(sizes)), np.nan, dtype=float)
    b_idx = {b: i for i, b in enumerate(budgets)}
    s_idx = {s: i for i, s in enumerate(sizes)}
    for row in rows:
        b = int(row["chunk_budget"])
        s = _xsize(row)
        out[b_idx[b], s_idx[s]] = float(row[metric])
    return out


def main() -> int:
    args = parse_args()
    support = _parse_int_csv(args.spike_count_support)
    probs = _parse_float_csv(args.spike_count_probs)
    chunk_sizes = _parse_int_csv(args.chunk_sizes)
    chunk_budgets = _parse_int_csv(args.chunk_budgets)
    weighting_modes = _parse_str_csv(args.weighting_modes)
    chunker = _parse_chunker_policy(args.chunker)
    selector = _parse_selector_policy(args.selector)

    spec = SpikeCountMixtureDistributionSpec(
        p_spike_doc=args.p_spike_doc,
        p_boundary_given_spike=args.p_boundary_given_spike,
        spike_count_support=support,
        spike_count_probs_given_spike=probs,
        n_tokens=args.n_tokens,
        proxy_noise=args.proxy_noise,
        boundary_span_tokens=args.boundary_span_tokens,
    )

    summaries = run_chunk_quality_sweep(
        distribution=spec,
        target_k=args.target_k,
        sketch_order=args.sketch_order,
        chunk_sizes=chunk_sizes,
        chunk_budgets=chunk_budgets,
        chunker=chunker,
        selector=selector,
        n_replicates=args.n_replicates,
        docs_per_replicate=args.docs_per_replicate,
        seed=args.seed,
        include_references=True,
        weighting_modes=weighting_modes,
        legacy_weighting_mode=args.legacy_weighting_mode,
    )
    rows = [asdict(s) for s in summaries]
    for row in rows:
        se, lo, hi = _bias_ci(row)
        row["se_bias"] = se
        row["ci95_bias_low"] = lo
        row["ci95_bias_high"] = hi

    refs = {r["method_name"]: r for r in rows if not str(r["method_name"]).startswith("grid_")}
    grid_rows = [r for r in rows if str(r["method_name"]).startswith("grid_")]

    budgets = sorted({int(r["chunk_budget"]) for r in grid_rows})
    sizes = sorted({_xsize(r) for r in grid_rows})

    bias_mat = _matrix_from_rows(grid_rows, budgets=budgets, sizes=sizes, metric="mean_abs_bias")
    cap_mat = _matrix_from_rows(grid_rows, budgets=budgets, sizes=sizes, metric="mean_target_capture_rate")
    iso_mat = _matrix_from_rows(grid_rows, budgets=budgets, sizes=sizes, metric="mean_spike_token_isolation")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    ax1 = axes[0]
    im1 = ax1.imshow(bias_mat, aspect="auto", origin="lower", cmap="viridis")
    ax1.set_xticks(range(len(sizes)))
    ax1.set_xticklabels([str(s) for s in sizes])
    ax1.set_yticks(range(len(budgets)))
    ax1.set_yticklabels([str(b) for b in budgets])
    ax1.set_xlabel("Leaf size (or max leaf size)")
    ax1.set_ylabel("Chunk budget")
    ax1.set_title("Mean Abs Bias")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = axes[1]
    im2 = ax2.imshow(cap_mat, aspect="auto", origin="lower", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels([str(s) for s in sizes])
    ax2.set_yticks(range(len(budgets)))
    ax2.set_yticklabels([str(b) for b in budgets])
    ax2.set_xlabel("Leaf size (or max leaf size)")
    ax2.set_ylabel("Chunk budget")
    ax2.set_title(f"Target Capture Rate (>=k, k={args.target_k})")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = axes[2]
    low_b = budgets[0]
    high_b = budgets[-1]
    line_configs = [
        (low_b, "#d62728", f"budget={low_b}"),
        (high_b, "#2ca02c", f"budget={high_b}"),
    ]
    xmin = min(sizes)
    xmax = max(sizes)
    for b, color, label in line_configs:
        sub = sorted((r for r in grid_rows if int(r["chunk_budget"]) == b), key=_xsize)
        xs = [_xsize(r) for r in sub]
        ys = [float(r["bias"]) for r in sub]
        err = [1.96 * float(r["se_bias"]) for r in sub]
        ax3.errorbar(xs, ys, yerr=err, marker="o", color=color, capsize=3, label=f"{label} bias (95% CI)")

    one_pass = refs.get("one_pass_reference")
    perfect = refs.get("perfect_token_leaves_all")
    if one_pass is not None:
        b = float(one_pass["bias"])
        ci = 1.96 * float(one_pass["se_bias"])
        ax3.hlines(b, xmin=xmin, xmax=xmax, colors="#1f77b4", linestyles="-", label="one-pass bias")
        ax3.fill_between(
            sizes,
            [b - ci for _ in sizes],
            [b + ci for _ in sizes],
            color="#1f77b4",
            alpha=0.12,
            linewidth=0,
            label="one-pass 95% CI",
        )
    if perfect is not None:
        ax3.hlines(
            float(perfect["bias"]),
            xmin=xmin,
            xmax=xmax,
            colors="#9467bd",
            linestyles="--",
            label="perfect token-leaf bias",
        )

    ax3.axhline(0.0, color="#444444", linewidth=1)
    ax3.set_xlabel("Leaf size (or max leaf size)")
    ax3.set_ylabel("Signed bias")
    ax3.set_title("Bias CI Cross-Section")
    ax3.legend(frameon=False, fontsize=8)
    ax3.grid(alpha=0.2)

    title_bits = [
        f"chunker={chunker.value}",
        f"selector={selector.value}",
        f"target_k={args.target_k}",
        f"sketch_order={args.sketch_order if args.sketch_order is not None else args.target_k}",
    ]
    if perfect is not None:
        title_bits.append(
            "perfect_iso="
            f"{float(perfect['mean_spike_token_isolation']):.3f}"
        )
    fig.suptitle("Chunk Quality Sweep | " + " | ".join(title_bits), fontsize=11)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)

    summary = {
        "distribution": asdict(spec),
        "target_k": int(args.target_k),
        "sketch_order": int(args.sketch_order) if args.sketch_order is not None else int(args.target_k),
        "objective": mergeable_probability_target_objective_semantics(
            name="generic_k_recovery_target",
            target_k=int(args.target_k),
            metadata={"family": "mergeable_chunk_quality_sweep"},
        ),
        "chunker": chunker.value,
        "selector": selector.value,
        "chunk_sizes": list(chunk_sizes),
        "chunk_budgets": list(chunk_budgets),
        "weighting_modes": list(weighting_modes),
        "legacy_weighting_mode": str(args.legacy_weighting_mode),
        "rows": rows,
        "grid_shape": {"budgets": budgets, "sizes": sizes},
        "reference_rows": refs,
        "output_figure": str(out_path),
    }
    summary_path = Path(args.json_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote_figure | {out_path}")
    print(f"wrote_summary | {summary_path}")
    if perfect is not None:
        print(
            "perfect_reference | "
            f"abs_bias={float(perfect['mean_abs_bias']):.4f} "
            f"target_capture={float(perfect['mean_target_capture_rate']):.4f} "
            f"spike_recall={float(perfect['mean_spike_token_recall']):.4f} "
            f"spike_isolation={float(perfect['mean_spike_token_isolation']):.4f}"
        )
    if one_pass is not None:
        print(
            "one_pass_reference | "
            f"abs_bias={float(one_pass['mean_abs_bias']):.4f} "
            f"bias={float(one_pass['bias']):+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
