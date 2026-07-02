#!/usr/bin/env python3
"""Plot and report the HLL explicit JAX f/g Round 4 grid."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


DEFAULT_GRID = Path(
    "outputs/hll_jax_local_law_round4_overnight_grid_20260508_065221/grid_summary.csv"
)

NUMERIC_COLUMNS = (
    "train_docs",
    "val_docs",
    "test_docs",
    "n_iter",
    "fragment_len",
    "summary_dim",
    "estimate_weight",
    "elapsed_seconds",
    "test_theta_mae",
    "test_hll_register_mae",
    "test_hll_estimate_raw_mae",
    "test_hll_estimate_norm_mae",
    "test_contextual_mae",
    "test_contextual_raw_mae",
    "test_eps_leaf",
    "test_eps_merge",
    "test_eps_idemp",
    "val_theta_mae",
    "val_hll_estimate_raw_mae",
    "val_contextual_raw_mae",
)

METRIC_LABELS = {
    "test_hll_estimate_raw_mae": "HLL Estimate MAE",
    "test_theta_mae": "Register MAE",
    "test_contextual_raw_mae": "Contextual Raw MAE",
    "test_eps_merge": "Merge Law Eps",
}

TRAIN_COLORS = {
    10240: "#2b6cb0",
    40960: "#2f855a",
    102400: "#c05621",
}


def _load_grid(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "status" in df.columns:
        df = df[df["status"].eq("ok")].copy()
    if "exit_code" in df.columns:
        df = df[pd.to_numeric(df["exit_code"], errors="coerce").eq(0)].copy()
    if df.empty:
        raise ValueError(f"no successful rows in {path}")
    return df


def _ensure_dirs(output_dir: Path) -> Path:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    return figures


def _save(fig: plt.Figure, figures: Path, stem: str) -> tuple[Path, Path]:
    png = figures / f"{stem}.png"
    pdf = figures / f"{stem}.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def _main_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["group"].eq("main")
        & df["law_architecture"].eq("learned_merge")
        & df["merge_loss"].eq("mse")
    ].copy()


def _plot_data_scaling(df: pd.DataFrame, figures: Path) -> list[Path]:
    rows = _main_rows(df)
    summary_dims = sorted(int(x) for x in rows["summary_dim"].dropna().unique())
    metrics = ["test_hll_estimate_raw_mae", "test_theta_mae"]
    fig, axes = plt.subplots(
        len(metrics),
        len(summary_dims),
        figsize=(5.2 * len(summary_dims), 7.5),
        squeeze=False,
        constrained_layout=True,
    )
    for row_idx, metric in enumerate(metrics):
        for col_idx, summary_dim in enumerate(summary_dims):
            ax = axes[row_idx][col_idx]
            subset = rows[rows["summary_dim"].eq(summary_dim)]
            for train_docs, group in subset.groupby("train_docs"):
                group = group.sort_values("fragment_len")
                ax.plot(
                    group["fragment_len"],
                    group[metric],
                    marker="o",
                    linewidth=1.8,
                    color=TRAIN_COLORS.get(int(train_docs), None),
                    label=f"{int(train_docs):,} docs",
                )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(sorted(rows["fragment_len"].dropna().unique()))
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.grid(True, which="both", alpha=0.24)
            ax.set_title(f"summary dim {summary_dim}")
            ax.set_xlabel("leaf / fragment tokens")
            ax.set_ylabel(METRIC_LABELS[metric])
            if row_idx == 0 and col_idx == len(summary_dims) - 1:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("HLL Round 4 Main Grid: Data Size x Leaf Size")
    png, _ = _save(fig, figures, "hll_round4_main_data_scaling")
    return [png]


def _heatmap_matrix(
    rows: pd.DataFrame,
    *,
    summary_dim: int,
    metric: str,
) -> tuple[np.ndarray, list[int], list[int]]:
    subset = rows[rows["summary_dim"].eq(summary_dim)]
    train_docs = sorted(int(x) for x in subset["train_docs"].dropna().unique())
    leaves = sorted(int(x) for x in subset["fragment_len"].dropna().unique())
    matrix = np.full((len(train_docs), len(leaves)), np.nan, dtype=float)
    for i, train in enumerate(train_docs):
        for j, leaf in enumerate(leaves):
            cell = subset[subset["train_docs"].eq(train) & subset["fragment_len"].eq(leaf)]
            if not cell.empty:
                matrix[i, j] = float(cell.iloc[0][metric])
    return matrix, train_docs, leaves


def _plot_main_heatmaps(df: pd.DataFrame, figures: Path) -> list[Path]:
    rows = _main_rows(df)
    summary_dims = sorted(int(x) for x in rows["summary_dim"].dropna().unique())
    metrics = ["test_hll_estimate_raw_mae", "test_theta_mae"]
    fig, axes = plt.subplots(
        len(metrics),
        len(summary_dims),
        figsize=(5.8 * len(summary_dims), 8.0),
        squeeze=False,
        constrained_layout=True,
    )
    for row_idx, metric in enumerate(metrics):
        positive_values = rows[metric].replace([np.inf, -np.inf], np.nan).dropna()
        norm = LogNorm(
            vmin=max(float(positive_values.min()), 1e-12),
            vmax=float(positive_values.max()),
        )
        for col_idx, summary_dim in enumerate(summary_dims):
            ax = axes[row_idx][col_idx]
            matrix, train_docs, leaves = _heatmap_matrix(
                rows,
                summary_dim=summary_dim,
                metric=metric,
            )
            image = ax.imshow(matrix, aspect="auto", cmap="viridis_r", norm=norm)
            ax.set_title(f"{METRIC_LABELS[metric]}, dz={summary_dim}")
            ax.set_xticks(range(len(leaves)), [str(x) for x in leaves])
            ax.set_yticks(range(len(train_docs)), [f"{x:,}" for x in train_docs])
            ax.set_xlabel("leaf / fragment tokens")
            ax.set_ylabel("train docs")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix[i, j]
                    if np.isfinite(value):
                        text = f"{value:.2g}" if value < 1 else f"{value:.2f}"
                        ax.text(j, i, text, ha="center", va="center", fontsize=7)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("HLL Round 4 Main Grid Heatmaps")
    png, _ = _save(fig, figures, "hll_round4_main_heatmaps")
    return [png]


def _pareto_mask(xs: Iterable[float], ys: Iterable[float]) -> list[bool]:
    points = list(zip(xs, ys))
    mask: list[bool] = []
    for i, (x_i, y_i) in enumerate(points):
        dominated = False
        for j, (x_j, y_j) in enumerate(points):
            if i == j:
                continue
            if x_j <= x_i and y_j <= y_i and (x_j < x_i or y_j < y_i):
                dominated = True
                break
        mask.append(not dominated)
    return mask


def _plot_tradeoff(df: pd.DataFrame, figures: Path) -> list[Path]:
    rows = df.dropna(subset=["test_theta_mae", "test_hll_estimate_raw_mae"]).copy()
    fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    markers = {"main": "o", "est": "s", "full": "^", "nasss": "D", "wide": "P"}
    for group, group_rows in rows.groupby("group"):
        marker = markers.get(str(group), "o")
        scatter = ax.scatter(
            group_rows["test_theta_mae"],
            group_rows["test_hll_estimate_raw_mae"],
            c=group_rows["fragment_len"],
            cmap="plasma",
            norm=LogNorm(
                vmin=float(rows["fragment_len"].min()),
                vmax=float(rows["fragment_len"].max()),
            ),
            marker=marker,
            s=54,
            alpha=0.86,
            edgecolors="white",
            linewidths=0.5,
            label=str(group),
        )
    pareto = rows[_pareto_mask(rows["test_theta_mae"], rows["test_hll_estimate_raw_mae"])]
    pareto = pareto.sort_values("test_theta_mae")
    ax.plot(
        pareto["test_theta_mae"],
        pareto["test_hll_estimate_raw_mae"],
        color="#222222",
        linewidth=1.3,
        linestyle="--",
        label="Pareto frontier",
    )
    for _, row in pareto.iterrows():
        if len(str(row["name"])) > 0:
            ax.annotate(
                f"L{int(row['fragment_len'])} n{int(row['train_docs']) // 1000}k",
                (float(row["test_theta_mae"]), float(row["test_hll_estimate_raw_mae"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("test register MAE")
    ax.set_ylabel("test HLL estimate MAE")
    ax.grid(True, which="both", alpha=0.24)
    ax.legend(frameon=False, fontsize=8, loc="best")
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("leaf / fragment tokens")
    fig.suptitle("HLL Round 4 Accuracy Tradeoff")
    png, _ = _save(fig, figures, "hll_round4_tradeoff")
    return [png]


def _plot_estimate_weight_controls(df: pd.DataFrame, figures: Path) -> list[Path]:
    rows = df[
        df["law_architecture"].eq("learned_merge")
        & df["merge_loss"].eq("mse")
        & df["summary_dim"].eq(128)
        & df["train_docs"].isin([10240, 102400])
        & df["fragment_len"].isin([32, 64, 128, 512])
        & df["estimate_weight"].isin([0.0, 0.1, 1.0])
    ].copy()
    metrics = ["test_hll_estimate_raw_mae", "test_theta_mae"]
    train_docs_values = sorted(int(x) for x in rows["train_docs"].dropna().unique())
    fig, axes = plt.subplots(
        len(metrics),
        len(train_docs_values),
        figsize=(5.6 * len(train_docs_values), 7.4),
        squeeze=False,
        constrained_layout=True,
    )
    for row_idx, metric in enumerate(metrics):
        for col_idx, train_docs in enumerate(train_docs_values):
            ax = axes[row_idx][col_idx]
            subset = rows[rows["train_docs"].eq(train_docs)]
            for leaf, group in subset.groupby("fragment_len"):
                group = group.sort_values("estimate_weight")
                ax.plot(
                    group["estimate_weight"],
                    group[metric],
                    marker="o",
                    linewidth=1.7,
                    label=f"L{int(leaf)}",
                )
            ax.set_yscale("log")
            ax.set_xticks([0.0, 0.1, 1.0])
            ax.set_xlabel("estimate auxiliary weight")
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_title(f"{int(train_docs):,} train docs")
            ax.grid(True, which="both", alpha=0.24)
            if row_idx == 0 and col_idx == len(train_docs_values) - 1:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("HLL Round 4 Estimate-Aware Loss Controls")
    png, _ = _save(fig, figures, "hll_round4_estimate_weight_controls")
    return [png]


def _fmt(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return ""
    if abs(numeric) >= 100:
        return f"{numeric:.1f}"
    if abs(numeric) >= 1:
        return f"{numeric:.4f}"
    return f"{numeric:.6g}"


def _top_table(df: pd.DataFrame, metric: str, *, n: int = 8) -> str:
    columns = [
        "name",
        "group",
        "train_docs",
        "fragment_len",
        "summary_dim",
        "estimate_weight",
        metric,
        "test_theta_mae",
        "test_contextual_raw_mae",
    ]
    rows = df.sort_values(metric, ascending=True).head(n)
    lines = [
        "| name | group | train | leaf | dz | est w | metric | register | contextual raw |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in rows.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["name"]),
                    str(row["group"]),
                    str(int(row["train_docs"])),
                    str(int(row["fragment_len"])),
                    str(int(row["summary_dim"])),
                    _fmt(row["estimate_weight"]),
                    _fmt(row[metric]),
                    _fmt(row["test_theta_mae"]),
                    _fmt(row["test_contextual_raw_mae"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _write_report(
    df: pd.DataFrame,
    *,
    grid_summary: Path,
    output_dir: Path,
    figures: list[Path],
) -> Path:
    report = output_dir / "report.md"
    best_est = df.loc[df["test_hll_estimate_raw_mae"].idxmin()]
    best_theta = df.loc[df["test_theta_mae"].idxmin()]
    best_context = df.loc[df["test_contextual_raw_mae"].idxmin()]
    lines = [
        "# HLL JAX Local-Law Round 4 Report",
        "",
        f"Source CSV: `{grid_summary}`",
        "",
        "## Headline Best Cells",
        "",
        (
            "- Best raw HLL estimate MAE: "
            f"`{_fmt(best_est['test_hll_estimate_raw_mae'])}` "
            f"from `{best_est['name']}`."
        ),
        (
            "- Best register MAE: "
            f"`{_fmt(best_theta['test_theta_mae'])}` "
            f"from `{best_theta['name']}`."
        ),
        (
            "- Best contextual raw MAE: "
            f"`{_fmt(best_context['test_contextual_raw_mae'])}` "
            f"from `{best_context['name']}`."
        ),
        "",
        "## Figures",
        "",
    ]
    for fig in figures:
        rel = fig.relative_to(output_dir)
        lines.extend([f"![{fig.stem}]({rel.as_posix()})", ""])
    lines.extend(
        [
            "## Top Raw HLL Estimate MAE",
            "",
            _top_table(df, "test_hll_estimate_raw_mae"),
            "",
            "## Top Register MAE",
            "",
            _top_table(df, "test_theta_mae"),
            "",
            "## Top Contextual Raw MAE",
            "",
            _top_table(df, "test_contextual_raw_mae"),
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-summary", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    grid_summary = Path(args.grid_summary).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else grid_summary.parent / "visual_report"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = _ensure_dirs(output_dir)
    df = _load_grid(grid_summary)

    figure_paths: list[Path] = []
    figure_paths.extend(_plot_data_scaling(df, figures))
    figure_paths.extend(_plot_main_heatmaps(df, figures))
    figure_paths.extend(_plot_tradeoff(df, figures))
    figure_paths.extend(_plot_estimate_weight_controls(df, figures))
    report = _write_report(
        df,
        grid_summary=grid_summary,
        output_dir=output_dir,
        figures=figure_paths,
    )
    print(f"wrote {report}")
    for path in figure_paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
