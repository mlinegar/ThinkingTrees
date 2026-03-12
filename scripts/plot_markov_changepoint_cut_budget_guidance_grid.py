#!/usr/bin/env python3
"""Plot a guidance grid for cut-budgeted changepoint DP policies.

This expects per-run JSON outputs from `run_markov_changepoint_cut_budget_simulation.py`
with guided policies enabled via `--guidance-multipliers` / `--guidance-per-leaf` and
`--guidance-strategies`.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot guidance grid for cut-budgeted changepoint DP.")
    parser.add_argument(
        "--input-glob",
        type=str,
        default="outputs/markov_changepoint_cut_budget_guidance/train_*_seed_*.json",
        help="Glob for per-run JSON outputs.",
    )
    parser.add_argument(
        "--output-figure",
        type=str,
        default="outputs/markov_changepoint_cut_budget_guidance_grid.png",
        help="Output PNG figure path.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/markov_changepoint_cut_budget_guidance_grid_report.json",
        help="Output JSON report path.",
    )
    return parser.parse_args()


def _stats(xs: List[float]) -> Dict[str, float]:
    if len(xs) == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {"n": int(len(xs)), "mean": float(fmean(xs)), "std": float(pstdev(xs))}


def _parse_guided_policy(policy: str) -> Optional[str]:
    if not policy.startswith("dp_guided_"):
        return None
    parts = policy.split("_")
    if len(parts) < 4:
        return None
    strat = str(parts[2])
    if strat not in {"random", "uncertainty", "active"}:
        return None
    return str(strat)


def _format_per_leaf(x: float) -> str:
    x = float(x)
    if not np.isfinite(x):
        return "nan"
    if abs(x) < 1.0:
        return f"{x:.3f}".rstrip("0").rstrip(".")
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _guidance_tick_labels(levels: List[Tuple[int, int]]) -> List[str]:
    labels: List[str] = []
    for q, leaves in levels:
        q = int(q)
        leaves = int(max(1, leaves))
        per_leaf = float(q) / float(leaves)
        labels.append(f"{_format_per_leaf(per_leaf)}/leaf (q≈{q})")
    return labels


def main() -> int:
    args = parse_args()
    files = [Path(p) for p in sorted(glob.glob(str(args.input_glob)))]
    if len(files) == 0:
        raise ValueError(f"no files matched: {args.input_glob}")

    rows: List[dict] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        train_docs = int(cfg.get("train_docs", -1))
        seed = int(cfg.get("seed", -1))
        if train_docs < 0:
            raise ValueError(f"missing config.train_docs in {path}")

        metrics = payload.get("metrics", {})
        if "fixed" not in metrics or "oracle_opt" not in metrics:
            continue

        fixed_ham = float(metrics["fixed"]["mean_hamming_loss"])
        mean_fixed_cut_budget = float(
            payload.get("mean_fixed_cut_budget", metrics["fixed"].get("mean_predicted_boundary_count", float("nan")))
        )
        fixed_leaves = int(round(mean_fixed_cut_budget)) + 1
        dp_honest = metrics.get("dp_honest")

        # Base row: dp_honest as q=0 (duplicated across strategies at plot time).
        if dp_honest is not None:
            rows.append(
                {
                    "train_docs": int(train_docs),
                    "seed": int(seed),
                    "strategy": "base",
                    "guidance_q": 0,
                    "guidance_leaves": int(fixed_leaves),
                    "policy": "dp_honest",
                    "mean_hamming_loss": float(dp_honest["mean_hamming_loss"]),
                    "mean_hamming_gap_to_oracle": float(dp_honest["mean_hamming_gap_to_oracle"]),
                    "mean_theory_gap_upper_bound": float(dp_honest.get("mean_theory_gap_upper_bound", float("nan"))),
                    "mean_predicted_boundary_count": float(dp_honest["mean_predicted_boundary_count"]),
                    "mean_oracle_queries_used": float(dp_honest.get("mean_oracle_queries_used", 0.0)),
                    "hamming_improvement_vs_fixed": float(fixed_ham - float(dp_honest["mean_hamming_loss"])),
                    "cuts_saved_vs_fixed": float(
                        float(metrics["fixed"]["mean_predicted_boundary_count"])
                        - float(dp_honest["mean_predicted_boundary_count"])
                    ),
                }
            )

        for policy, m in metrics.items():
            strat = _parse_guided_policy(str(policy))
            if strat is None:
                continue
            ham = float(m["mean_hamming_loss"])
            q = int(round(float(m.get("mean_oracle_queries_used", float("nan")))))
            rows.append(
                {
                    "train_docs": int(train_docs),
                    "seed": int(seed),
                    "strategy": str(strat),
                    "guidance_q": int(q),
                    "guidance_leaves": int(fixed_leaves),
                    "policy": str(policy),
                    "mean_hamming_loss": float(ham),
                    "mean_hamming_gap_to_oracle": float(m["mean_hamming_gap_to_oracle"]),
                    "mean_theory_gap_upper_bound": float(m.get("mean_theory_gap_upper_bound", float("nan"))),
                    "mean_predicted_boundary_count": float(m["mean_predicted_boundary_count"]),
                    "mean_oracle_queries_used": float(m.get("mean_oracle_queries_used", float("nan"))),
                    "hamming_improvement_vs_fixed": float(fixed_ham - ham),
                    "cuts_saved_vs_fixed": float(
                        float(metrics["fixed"]["mean_predicted_boundary_count"]) - float(m["mean_predicted_boundary_count"])
                    ),
                }
            )

    if len(rows) == 0:
        raise ValueError(
            "no guidance rows found (did you run with --guidance-multipliers/--guidance-per-leaf and --guidance-strategies?)"
        )

    train_docs_values = sorted({int(r["train_docs"]) for r in rows})
    strategies = sorted({str(r["strategy"]) for r in rows if str(r["strategy"]) != "base"})
    if not strategies:
        raise ValueError("no guided strategies found in inputs")

    guidance_levels = sorted(
        {(int(r["guidance_q"]), int(r["guidance_leaves"])) for r in rows},
        key=lambda x: (float(x[0]) / float(max(1, x[1])), int(x[0])),
    )
    if not guidance_levels or guidance_levels[0][0] != 0:
        raise ValueError("missing baseline guidance level q=0 (dp_honest)")
    guidance_tick_labels = _guidance_tick_labels(guidance_levels)

    metrics_to_plot = (
        ("mean_hamming_gap_to_oracle", "Mean Hamming gap to oracle (↓)", "viridis_r", None),
        ("mean_theory_gap_upper_bound", "Lean upper bound Σ|δ| on gap (↓)", "viridis_r", None),
        ("hamming_improvement_vs_fixed", "Hamming improvement vs fixed (↑)", "coolwarm", 0.0),
        ("cuts_saved_vs_fixed", "Cuts saved vs fixed (↑)", "plasma", 0.0),
    )

    aggregated: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = {}
    for metric, _ylabel, _cmap, _center in metrics_to_plot:
        aggregated[metric] = {}
        for strat in ["base"] + strategies:
            aggregated[metric][strat] = {}
            for td in train_docs_values:
                aggregated[metric][strat][str(td)] = {}
                for q, leaves in guidance_levels:
                    key = f"{int(q)}/{int(leaves)}"
                    vals = [
                        float(r[metric])
                        for r in rows
                        if int(r["train_docs"]) == int(td)
                        and str(r["strategy"]) == str(strat)
                        and int(r["guidance_q"]) == int(q)
                        and int(r["guidance_leaves"]) == int(leaves)
                        and np.isfinite(float(r[metric]))
                    ]
                    aggregated[metric][strat][str(td)][key] = _stats(vals)

    # Prepare plotting arrays.
    fig, axs = plt.subplots(
        len(metrics_to_plot),
        len(strategies),
        figsize=(5.2 * len(strategies) + 2.0, 3.8 * len(metrics_to_plot)),
        constrained_layout=True,
    )
    if len(metrics_to_plot) == 1:
        axs = np.asarray([axs])
    if len(strategies) == 1:
        axs = axs.reshape(len(metrics_to_plot), 1)

    for row_i, (metric, ylabel, cmap_name, center) in enumerate(metrics_to_plot):
        for col_i, strat in enumerate(strategies):
            # Fill q=0 with dp_honest baseline (strategy=base).
            arr = np.full((len(guidance_levels), len(train_docs_values)), np.nan, dtype=np.float64)
            for j, td in enumerate(train_docs_values):
                # baseline row
                base_key = f"{int(guidance_levels[0][0])}/{int(guidance_levels[0][1])}"
                base_stat = aggregated[metric]["base"][str(td)].get(base_key, {"mean": float("nan")})
                arr[0, j] = float(base_stat["mean"])
                for i, (q, leaves) in enumerate(guidance_levels[1:], start=1):
                    key = f"{int(q)}/{int(leaves)}"
                    stat = aggregated[metric][strat][str(td)][key]
                    arr[i, j] = float(stat["mean"])

            ax = axs[row_i, col_i]
            masked = np.ma.masked_invalid(arr)
            if center is None:
                im = ax.imshow(masked, aspect="auto", origin="lower", cmap=cmap_name)
            else:
                vmax = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else 1.0
                vmax = max(vmax, 1e-6)
                im = ax.imshow(
                    masked,
                    aspect="auto",
                    origin="lower",
                    cmap=cmap_name,
                    vmin=-vmax,
                    vmax=vmax,
                )

            ax.set_title(f"{strat} | {ylabel}")
            ax.set_xticks(list(range(len(train_docs_values))))
            ax.set_xticklabels([str(x) for x in train_docs_values], rotation=45, ha="right")
            ax.set_yticks(list(range(len(guidance_levels))))
            ax.set_yticklabels(guidance_tick_labels)
            ax.set_xlabel("train_docs")
            ax.set_ylabel("oracle queries per leaf (q shown)")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Markov Changepoint: Oracle Guidance Grid (Cut-Budgeted DP)", fontsize=12)

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=170)
    plt.close(fig)

    report = {
        "input_glob": str(args.input_glob),
        "n_files": int(len(files)),
        "n_rows": int(len(rows)),
        "train_docs_values": train_docs_values,
        "strategies": strategies,
        "guidance_levels": [
            {"q": int(q), "leaves": int(leaves), "q_per_leaf": float(q) / float(max(1, int(leaves)))}
            for q, leaves in guidance_levels
        ],
        "metrics": {m: {"ylabel": y} for m, y, _c, _ctr in metrics_to_plot},
        "aggregated": aggregated,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote_figure | {out_fig}")
    print(f"wrote_json | {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
