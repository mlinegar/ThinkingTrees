#!/usr/bin/env python3
"""Non-language scenario suite for mergeable chunk-quality experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.mergeable_ablation import (
    ChunkerPolicy,
    SelectorPolicy,
    default_nonlanguage_chunk_quality_scenarios,
    run_chunk_quality_sweep,
)
from src.ctreepo.sim.objective_semantics import mergeable_probability_target_objective_semantics


def _parse_int_csv(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def _parse_str_csv(s: str) -> tuple[str, ...]:
    out = tuple(x.strip() for x in s.split(",") if x.strip())
    if len(out) == 0:
        raise ValueError("expected a non-empty string CSV")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run non-language scenario suite and plot budget-vs-bias/capture curves "
            "for fixed/aligned/misspecified chunking."
        )
    )
    parser.add_argument("--target-k", type=int, default=5)
    parser.add_argument("--sketch-order", type=int, default=None)
    parser.add_argument("--chunk-sizes", type=str, default="1,2,4,8,16")
    parser.add_argument("--chunk-budgets", type=str, default="1,2,3,4,6,8")
    parser.add_argument("--n-replicates", type=int, default=60)
    parser.add_argument("--docs-per-replicate", type=int, default=120)
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
        default="outputs/mergeable_nonlanguage_suite.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--json-summary",
        type=str,
        default="outputs/mergeable_nonlanguage_suite_summary.json",
        help="JSON summary output path.",
    )
    return parser.parse_args()


def _best_by_budget(rows: List[dict], *, budgets: List[int], metric: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for b in budgets:
        sub = [r for r in rows if int(r["chunk_budget"]) == b]
        if len(sub) == 0:
            out[b] = float("nan")
            continue
        best = min(sub, key=lambda r: float(r["mean_abs_bias"]))
        out[b] = float(best[metric])
    return out


def main() -> int:
    args = parse_args()
    chunk_sizes = _parse_int_csv(args.chunk_sizes)
    chunk_budgets = sorted(_parse_int_csv(args.chunk_budgets))
    weighting_modes = _parse_str_csv(args.weighting_modes)
    scenarios = list(default_nonlanguage_chunk_quality_scenarios())
    if len(scenarios) == 0:
        raise ValueError("No non-language scenarios configured")

    family_configs = [
        ("fixed_top_proxy", ChunkerPolicy.FIXED, SelectorPolicy.TOP_PROXY),
        ("adaptive_aligned_top_proxy", ChunkerPolicy.ADAPTIVE_ALIGNED, SelectorPolicy.TOP_PROXY),
        (
            "adaptive_misspecified_bottom_proxy",
            ChunkerPolicy.ADAPTIVE_MISSPECIFIED,
            SelectorPolicy.BOTTOM_PROXY,
        ),
    ]
    family_color = {
        "fixed_top_proxy": "#1f77b4",
        "adaptive_aligned_top_proxy": "#2ca02c",
        "adaptive_misspecified_bottom_proxy": "#d62728",
    }

    suite_rows: List[dict] = []
    panel_data: Dict[str, dict] = {}
    for s_idx, scenario in enumerate(scenarios):
        scenario_key = scenario.name
        panel_data[scenario_key] = {
            "intuition": scenario.intuition,
            "families": {},
            "references": {},
        }

        for f_idx, (family, chunker, selector) in enumerate(family_configs):
            include_refs = f_idx == 0
            run = run_chunk_quality_sweep(
                distribution=scenario.distribution,
                target_k=args.target_k,
                sketch_order=args.sketch_order,
                chunk_sizes=chunk_sizes,
                chunk_budgets=tuple(chunk_budgets),
                chunker=chunker,
                selector=selector,
                n_replicates=args.n_replicates,
                docs_per_replicate=args.docs_per_replicate,
                seed=args.seed + (100_000 * s_idx) + (1_000 * f_idx),
                include_references=include_refs,
                weighting_modes=weighting_modes,
                legacy_weighting_mode=args.legacy_weighting_mode,
            )
            rows = [asdict(r) for r in run]
            for r in rows:
                r["scenario_name"] = scenario.name
                r["scenario_intuition"] = scenario.intuition
                r["family"] = family
                suite_rows.append(r)

            grid_rows = [r for r in rows if str(r["method_name"]).startswith("grid_")]
            panel_data[scenario_key]["families"][family] = {
                "best_abs_bias_by_budget": _best_by_budget(
                    grid_rows, budgets=chunk_budgets, metric="mean_abs_bias"
                ),
                "best_target_capture_by_budget": _best_by_budget(
                    grid_rows, budgets=chunk_budgets, metric="mean_target_capture_rate"
                ),
            }
            if include_refs:
                refs = {r["method_name"]: r for r in rows if not str(r["method_name"]).startswith("grid_")}
                panel_data[scenario_key]["references"] = {
                    "one_pass_abs_bias": float(refs["one_pass_reference"]["mean_abs_bias"]),
                    "perfect_abs_bias": float(refs["perfect_token_leaves_all"]["mean_abs_bias"]),
                    "perfect_target_capture": float(
                        refs["perfect_token_leaves_all"]["mean_target_capture_rate"]
                    ),
                }

    n = len(scenarios)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.6 * nrows), constrained_layout=True)
    if nrows == 1 and ncols == 1:
        axes_list = [axes]
    elif nrows == 1:
        axes_list = list(axes)
    else:
        axes_list = [ax for row in axes for ax in row]

    for idx, scenario in enumerate(scenarios):
        ax = axes_list[idx]
        pdata = panel_data[scenario.name]
        for family, _, _ in family_configs:
            d = pdata["families"][family]
            ys = [d["best_abs_bias_by_budget"][b] for b in chunk_budgets]
            cap = [d["best_target_capture_by_budget"][b] for b in chunk_budgets]
            ax.plot(
                chunk_budgets,
                ys,
                marker="o",
                color=family_color[family],
                label=f"{family}: abs-bias",
            )
            ax.plot(
                chunk_budgets,
                cap,
                marker="x",
                linestyle="--",
                color=family_color[family],
                alpha=0.55,
                label=f"{family}: capture",
            )

        refs = pdata["references"]
        ax.hlines(
            refs["one_pass_abs_bias"],
            xmin=min(chunk_budgets),
            xmax=max(chunk_budgets),
            colors="#444444",
            linestyles="-.",
            label="one-pass abs-bias",
        )
        ax.hlines(
            refs["perfect_abs_bias"],
            xmin=min(chunk_budgets),
            xmax=max(chunk_budgets),
            colors="#9467bd",
            linestyles=":",
            label="perfect-token abs-bias",
        )
        ax.set_title(f"{scenario.name}\n{scenario.intuition}")
        ax.set_xlabel("Chunk budget")
        ax.set_ylabel("Metric value")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.2)

    for j in range(n, len(axes_list)):
        axes_list[j].axis("off")

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(
        f"Non-Language Suite | target_k={args.target_k} "
        f"sketch_order={args.sketch_order if args.sketch_order is not None else args.target_k}",
        fontsize=12,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)

    payload = {
        "target_k": int(args.target_k),
        "sketch_order": int(args.sketch_order) if args.sketch_order is not None else int(args.target_k),
        "objective": mergeable_probability_target_objective_semantics(
            name="generic_k_recovery_target",
            target_k=int(args.target_k),
            metadata={"family": "mergeable_nonlanguage_suite"},
        ),
        "chunk_sizes": list(chunk_sizes),
        "chunk_budgets": list(chunk_budgets),
        "n_replicates": int(args.n_replicates),
        "docs_per_replicate": int(args.docs_per_replicate),
        "seed": int(args.seed),
        "weighting_modes": list(weighting_modes),
        "legacy_weighting_mode": str(args.legacy_weighting_mode),
        "scenario_panel_data": panel_data,
        "rows": suite_rows,
        "output_figure": str(out_path),
    }
    summary_path = Path(args.json_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote_figure | {out_path}")
    print(f"wrote_summary | {summary_path}")
    for scenario in scenarios:
        pdata = panel_data[scenario.name]
        aligned = pdata["families"]["adaptive_aligned_top_proxy"]["best_abs_bias_by_budget"]
        wrong = pdata["families"]["adaptive_misspecified_bottom_proxy"]["best_abs_bias_by_budget"]
        b0 = chunk_budgets[0]
        b1 = chunk_budgets[-1]
        print(
            f"scenario={scenario.name} | aligned_abs_bias@b{b0}={aligned[b0]:.4f} "
            f"aligned_abs_bias@b{b1}={aligned[b1]:.4f} "
            f"wrong_abs_bias@b{b1}={wrong[b1]:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
