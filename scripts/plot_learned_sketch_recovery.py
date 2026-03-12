#!/usr/bin/env python3
"""Plot learned sketch recovery experiment results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Colors consistent with rest of project
COLORS = {
    1: "#d62728",   # red
    2: "#d62728",   # red — below sufficiency
    3: "#ff7f0e",   # orange — below sufficiency
    4: "#1f77b4",   # blue — at boundary (k=4)
    5: "#2ca02c",   # green — above
    6: "#9467bd",   # purple — above
    7: "#8c564b",   # brown — above
}
DEFAULT_COLOR = "#444444"


def plot_learning_curves(data: dict, out_path: Path) -> None:
    """Two-panel figure: MSE learning curves + accuracy comparison bar chart."""
    results = data["results"]
    target_k = data["target_k"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # ---- Left panel: learning curves ----
    for run in results:
        m = run["state_dim"]
        metrics = run["metrics"]
        steps = [d["step"] for d in metrics]
        root_mse = [d["root_oracle_mse"] for d in metrics]
        color = COLORS.get(m, DEFAULT_COLOR)
        below = m < target_k
        ls = "--" if below else "-"
        label_suffix = " (m<k)" if below else (" (m=k)" if m == target_k else " (m>k)")

        ax1.plot(steps, root_mse, color=color, linestyle=ls,
                 marker="o", markersize=2.5, linewidth=1.5,
                 label=f"m={m}{label_suffix}")

    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Root MSE (per-type count prediction)")
    ax1.set_yscale("log")
    ax1.grid(alpha=0.2)
    ax1.legend(frameon=False, fontsize=9)

    # ---- Right panel: accuracy comparison ----
    ms = [r["state_dim"] for r in results]
    learned_accs = [r["final_threshold_accuracy"] for r in results]
    hd_accs = [r["hand_designed_accuracy"] for r in results]

    x = np.arange(len(ms))
    width = 0.35

    bars_learned = ax2.bar(x - width / 2, learned_accs, width,
                           color="#1f77b4", alpha=0.85, label="Learned sketch")
    bars_hd = ax2.bar(x + width / 2, hd_accs, width,
                      color="#ff7f0e", alpha=0.85, label="Hand-designed top-m")

    ax2.set_xlabel("State dimension m")
    ax2.set_ylabel("Threshold accuracy")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(m) for m in ms])
    ax2.set_ylim(0.0, 1.12)
    ax2.axhline(y=1.0, color="#444444", linestyle=":", linewidth=0.8, alpha=0.5)

    # Annotate bars with values
    for bar, val in zip(bars_hd, hd_accs):
        if val < 0.95:
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8,
                     color="#ff7f0e")

    # Draw the m=k boundary
    for i, m in enumerate(ms):
        if m == target_k:
            ax2.axvline(x=i - 0.5, color="#444444", linestyle="--",
                        linewidth=0.8, alpha=0.4)
            ax2.text(i - 0.5, 0.05, f"m=k={target_k}",
                     ha="center", fontsize=7, color="#444444", alpha=0.7)
            break

    ax2.grid(alpha=0.2, axis="y")
    ax2.legend(frameon=False, fontsize=9, loc="lower right")

    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote_figure | {out_path}")


def plot_phase_diagram(data: dict, out_path: Path) -> None:
    """Heatmap of threshold accuracy across (k, m) grid."""
    results = data["results"]
    target_ks = sorted(set(r["target_k"] for r in results))
    state_dims = sorted(set(r["state_dim"] for r in results))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    learned_grid = np.full((len(target_ks), len(state_dims)), float("nan"))
    hd_grid = np.full_like(learned_grid, float("nan"))

    for r in results:
        ki = target_ks.index(r["target_k"])
        mi = state_dims.index(r["state_dim"])
        learned_grid[ki, mi] = r["final_threshold_accuracy"]
        hd_grid[ki, mi] = r["hand_designed_accuracy"]

    for ax, grid, title in [
        (ax1, learned_grid, "Learned sketch"),
        (ax2, hd_grid, "Hand-designed sketch"),
    ]:
        im = ax.imshow(grid, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(state_dims)))
        ax.set_xticklabels([str(m) for m in state_dims])
        ax.set_yticks(range(len(target_ks)))
        ax.set_yticklabels([str(k) for k in target_ks])
        ax.set_xlabel("State dimension m")
        ax.set_ylabel("Target k")
        ax.set_title(title)

        for i in range(len(target_ks)):
            for j in range(len(state_dims)):
                val = grid[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=9, color="black" if val > 0.7 else "white")

        # Mark m=k diagonal
        for i, k in enumerate(target_ks):
            for j, m in enumerate(state_dims):
                if m == k:
                    ax.plot(j, i, "k*", markersize=12)

    fig.colorbar(im, ax=[ax1, ax2], label="Threshold accuracy", shrink=0.8)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote_figure | {out_path}")


def plot_budget(data: dict, out_path: Path) -> None:
    """Audit budget sensitivity: convergence speed vs n_audit."""
    results = data["results"]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for run in results:
        n_audit = run["n_audit"]
        metrics = run["metrics"]
        steps = [d["step"] for d in metrics]
        root_mse = [d["root_oracle_mse"] for d in metrics]
        ax.plot(steps, root_mse, marker="o", markersize=3, linewidth=1.5,
                label=f"n_audit={n_audit}")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Root MSE")
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote_figure | {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot learned sketch recovery results.")
    parser.add_argument("input", type=str, help="Input JSON path.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (default: derived from input).")
    parser.add_argument("--experiment", type=str, default=None,
                        choices=["curves", "phase", "budget"],
                        help="Which plot to produce (auto-detected from JSON if omitted).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    data = json.loads(input_path.read_text())

    exp_type = args.experiment
    if exp_type is None:
        exp_type = data.get("experiment", "curves")
        if exp_type == "learning_curves":
            exp_type = "curves"
        elif exp_type == "audit_budget":
            exp_type = "budget"
        elif exp_type == "phase_diagram":
            exp_type = "phase"

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = input_path.with_suffix(".png")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if exp_type == "curves":
        plot_learning_curves(data, out_path)
    elif exp_type == "phase":
        plot_phase_diagram(data, out_path)
    elif exp_type == "budget":
        plot_budget(data, out_path)
    else:
        print(f"Unknown experiment type: {exp_type}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
