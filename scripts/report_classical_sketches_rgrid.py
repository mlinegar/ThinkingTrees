#!/usr/bin/env python3
"""Build a focused R-grid report for the classical-sketch paper bundle."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


EPS = 1.0e-12
R_ORDER = ("R10", "R30", "R50", "R100")
DEFAULT_LEAF_ORDER = (32, 64, 128, 256, 512)
NULL_CACHE_NAME = "measured_constant_null_baselines.csv"

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra_path in (REPO_ROOT / "treepo" / "src", REPO_ROOT / "parallel" / "unified_g_v1" / "src"):
    if _extra_path.exists() and str(_extra_path) not in sys.path:
        sys.path.insert(0, str(_extra_path))


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower())
    return re.sub(r"_+", "_", value).strip("_") or "unknown"


def _r_label(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return ""
    except TypeError:
        pass
    try:
        return f"R{int(round(float(value) * 100))}"
    except (TypeError, ValueError, OverflowError):
        return ""


def _r_rate(label: str) -> float:
    try:
        return float(str(label).lstrip("R")) / 100.0
    except ValueError:
        return float("nan")


def _ordered_rates(values: Iterable[object]) -> tuple[str, ...]:
    present = {str(value) for value in values if str(value)}
    return tuple(rate for rate in R_ORDER if rate in present)


def _fmt(x: object) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return "" if x is None else str(x)
    if not math.isfinite(value):
        return ""
    if abs(value - round(value)) < 1.0e-9 and abs(value) < 1.0e6:
        return str(int(round(value)))
    if abs(value) < 0.5e-12:
        return "0"
    if abs(value) < 1.0e-3:
        return f"{value:.1e}"
    if abs(value) < 10:
        return f"{value:.3g}"
    return f"{value:.2g}"


def _safe_rel_error(pred: float, truth: float) -> float:
    return (float(pred) - float(truth)) / max(1.0, abs(float(truth)))


def _md_table(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    data = frame.head(max_rows) if max_rows is not None else frame
    if data.empty:
        return "_No rows._\n"
    cols = [str(c) for c in data.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in data.iterrows():
        vals = []
        for col in data.columns:
            val = row[col]
            vals.append(_fmt(val) if isinstance(val, (float, int, np.floating, np.integer)) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _method_label(row: pd.Series) -> str:
    status = str(row.get("implementation_status", ""))
    if status == "learned_empirical":
        raw = str(row.get("learned_run_slug") or row.get("sketch") or "")
        variant = str(row.get("learned_variant") or "")
        label = raw
        for old, new in (
            ("learned_joint_fg_", "f+g:"),
            ("learned_joint_", "f+g:"),
            ("learned_g_", "g-only:"),
            ("_reference", ":ref"),
            ("_state_space", ":state"),
            ("_summary_sum", ":sum"),
            ("exact_", "exact:"),
            ("count_min", "cms"),
            ("frequent_strings", "freq_strings"),
            ("total_weight", "total_wt"),
            ("a_not_b", "A\\B"),
            ("intersection", "intersect"),
        ):
            label = label.replace(old, new)
        if variant and not label.startswith(("f+g:", "g-only:")):
            label = f"{variant.replace('fg', 'f+g')}:{label}"
        return label
    sketch = str(row.get("sketch") or "")
    status_prefix = {
        "official_empirical": "official",
        "lean_backed": "lean",
        "control": "control",
        "negative_control": "neg",
    }.get(status, status or "row")
    return f"{status_prefix}:{sketch}"


def _sketch_type(row: pd.Series) -> str:
    raw = " ".join(
        str(row.get(col) or "")
        for col in ("learned_target_kind", "learned_run_slug", "sketch")
    ).lower()
    checks = (
        ("Count-Min", ("count_min", "cms")),
        ("Frequent Strings", ("frequent_strings",)),
        ("HLL", ("hll",)),
        ("CPC", ("cpc",)),
        ("Theta", ("theta",)),
        ("KLL", ("kll",)),
        ("Quantiles", ("quantiles",)),
        ("REQ", ("req",)),
        ("T-Digest", ("tdigest",)),
        ("Tuple", ("tuple", "accumulator")),
        ("VarOpt", ("varopt",)),
        ("Exact Set", ("exact_set", "exact_distinct", "exact_frequency", "exact_total_weight")),
        ("Exact Quantile", ("exact_quantile",)),
        ("Negative Control", ("sum_leaf_uniques",)),
    )
    for label, needles in checks:
        if any(needle in raw for needle in needles):
            return label
    return "Other"


def _prepare(raw: pd.DataFrame, *, leaf_order: tuple[int, ...]) -> pd.DataFrame:
    df = raw.copy()
    for col in (
        "relative_rmse_mean",
        "official_floor_rel_rmse_mean",
        "distance_to_official_floor_mean",
        "learned_root_query_rate",
        "learned_leaf_query_rate",
        "learned_internal_query_rate",
        "leaf_size",
        "memory_bytes_mean",
    ):
        if col in df.columns:
            df[col] = _num(df[col])
    df = df[df["leaf_size"].notna()].copy()
    df["leaf_size"] = df["leaf_size"].astype(int)
    if leaf_order:
        df = df[df["leaf_size"].isin(leaf_order)].copy()
    df["R"] = df["learned_root_query_rate"].map(_r_label)
    df = df[df["R"].isin(R_ORDER)].copy()
    df["task"] = df["family"].astype(str) + "/" + df["query"].astype(str)
    df["method_label"] = df.apply(_method_label, axis=1)
    df["sketch_type"] = df.apply(_sketch_type, axis=1)
    df["task_method"] = df["task"] + " | " + df["method_label"]
    df["leaf_R"] = df["leaf_size"].astype(str) + "/" + df["R"]
    df["metric"] = df["relative_rmse_mean"].clip(lower=0)
    df["metric_plot"] = df["metric"].clip(lower=EPS)
    df["official_floor"] = df["official_floor_rel_rmse_mean"]
    df["floor_ratio"] = np.where(df["official_floor"] > 0, df["metric"] / df["official_floor"], np.nan)
    return df


def _read_aggregate_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    try:
        source = str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        source = str(path.resolve())
    frame["_source_csv"] = source
    return frame


def _load_aggregate_sources(primary_csv: Path, extra_csvs: list[Path]) -> pd.DataFrame:
    primary = _read_aggregate_csv(primary_csv)
    if not extra_csvs:
        return primary

    primary_rates = set(_ordered_rates(primary.get("learned_root_query_rate", pd.Series(dtype=float)).map(_r_label)))
    frames = [primary]
    filled_rates = set(primary_rates)
    for path in extra_csvs:
        extra = _read_aggregate_csv(path)
        extra = extra.copy()
        extra["_extra_R"] = extra.get("learned_root_query_rate", pd.Series(dtype=float)).map(_r_label)
        missing_rates = set(R_ORDER) - filled_rates
        keep = extra["_extra_R"].isin(missing_rates)
        kept = extra[keep].drop(columns=["_extra_R"])
        frames.append(kept)
        filled_rates.update(set(kept.get("learned_root_query_rate", pd.Series(dtype=float)).map(_r_label)))
    return pd.concat(frames, ignore_index=True, sort=False)


def _parse_leaf_sizes(text: str) -> tuple[int, ...]:
    leaves: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        leaves.append(int(part))
    return tuple(leaves)


def _source_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    if "_source_csv" not in df.columns:
        return pd.DataFrame()
    rows = []
    for source, group in df.groupby("_source_csv", dropna=False):
        rates = "/".join(_ordered_rates(group["R"].dropna().unique()))
        leaves = ",".join(str(x) for x in sorted(group["leaf_size"].dropna().astype(int).unique()))
        rows.append(
            {
                "source": str(source),
                "rows_used": int(len(group)),
                "R_values_used": rates,
                "leaf_sizes_used": leaves,
            }
        )
    return pd.DataFrame(rows).sort_values("source")


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _heatmap(
    matrix: pd.DataFrame,
    path: Path,
    *,
    title: str,
    cbar_label: str,
    log_scale: bool,
    cmap: str = "viridis_r",
    row_font: int = 7,
) -> Path:
    values = matrix.to_numpy(dtype=float)
    plot_values = np.log10(values + EPS) if log_scale else values
    plot_values = np.ma.masked_invalid(plot_values)
    height = max(4.0, 0.28 * max(1, len(matrix.index)) + 1.5)
    width = max(8.0, 0.52 * max(1, len(matrix.columns)) + 3.0)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(plot_values, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(list(matrix.columns), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([str(x).replace(" | ", "\n") for x in matrix.index], fontsize=row_font)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label(cbar_label)
    return _save(fig, path)


def _line_breakout(task: str, rows: pd.DataFrame, out_dir: Path, leaf_order: tuple[int, ...]) -> Path:
    methods = (
        rows.groupby("method_label")["metric_plot"]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    if len(methods) > 12:
        best = set(methods[:10])
        official = set(rows.loc[rows["implementation_status"].ne("learned_empirical"), "method_label"])
        methods = [m for m in methods if m in best or m in official]
        plot_rows = rows[rows["method_label"].isin(methods)].copy()
    else:
        plot_rows = rows.copy()
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(methods))))
    color_map = {method: colors[i] for i, method in enumerate(methods)}
    fig, axes = plt.subplots(1, len(R_ORDER), figsize=(max(17, 4.2 * len(R_ORDER)), 4.6), sharey=True)
    for ax, rate in zip(axes, R_ORDER):
        panel = plot_rows[plot_rows["R"].eq(rate)]
        for method in methods:
            data = (
                panel[panel["method_label"].eq(method)]
                .groupby("leaf_size")["metric_plot"]
                .median()
                .reindex(leaf_order)
            )
            if data.notna().sum() == 0:
                continue
            status = panel.loc[panel["method_label"].eq(method), "implementation_status"].iloc[0]
            linestyle = "-" if status == "learned_empirical" else "--"
            alpha = 0.92 if status == "learned_empirical" else 0.65
            ax.plot(
                data.index,
                data.to_numpy(dtype=float),
                marker="o",
                linewidth=1.55,
                markersize=3.5,
                linestyle=linestyle,
                alpha=alpha,
                color=color_map[method],
                label=method,
            )
        ax.set_title(rate)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(leaf_order)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("Leaf tokens")
    axes[0].set_ylabel("Relative RMSE")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.suptitle(f"{task}: sketch methods by leaf size and R", y=1.02)
    return _save(fig, out_dir / f"task_{_slug(task)}_by_sketch_leaf_r.png")


def _winner_table_plot(winners: pd.DataFrame, out_path: Path) -> Path:
    sketch_types = sorted(
        {
            str(cell.get("sketch_type"))
            for cell in winners.to_numpy(dtype=object).ravel()
            if isinstance(cell, dict) and cell.get("sketch_type") is not None
        }
    )
    cmap = ListedColormap(plt.cm.Set3(np.linspace(0, 1, max(3, len(sketch_types)))))
    type_to_idx = {label: i for i, label in enumerate(sketch_types)}
    tasks = list(winners.index)
    cols = list(winners.columns)
    color_values = np.full((len(tasks), len(cols)), np.nan)
    labels = [["" for _ in cols] for _ in tasks]
    for i, task in enumerate(tasks):
        for j, col in enumerate(cols):
            cell = winners.loc[task, col]
            if isinstance(cell, dict):
                color_values[i, j] = type_to_idx.get(str(cell.get("sketch_type")), np.nan)
                labels[i][j] = str(cell.get("short", ""))
    fig, ax = plt.subplots(figsize=(max(10, 0.72 * len(cols) + 3), max(4.5, 0.45 * len(tasks) + 1.8)))
    ax.imshow(np.ma.masked_invalid(color_values), aspect="auto", cmap=cmap, vmin=-0.5, vmax=max(0.5, len(sketch_types) - 0.5))
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=8)
    ax.set_title("Best sketch type per task and leaf/R cell")
    for i in range(len(tasks)):
        for j in range(len(cols)):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=6.5, color="black")
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tasks), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=8, color=cmap(type_to_idx[label]))
        for label in sketch_types
    ]
    ax.legend(handles, sketch_types, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, title="Sketch type")
    return _save(fig, out_path)


def _rate_gain_plot(learned: pd.DataFrame, out_path: Path) -> tuple[Path, pd.DataFrame]:
    pivot = (
        learned.pivot_table(
            index=["task", "method_label", "sketch_type", "leaf_size"],
            columns="R",
            values="metric_plot",
            aggfunc="median",
        )
        .reset_index()
    )
    ratio_cols: list[tuple[str, str]] = []
    if "R10" in pivot.columns:
        for rate in R_ORDER[1:]:
            if rate not in pivot.columns:
                continue
            col = f"{rate}_over_R10"
            pivot[col] = pivot[rate] / pivot["R10"]
            ratio_cols.append((rate, col))
    if ratio_cols:
        long = pivot.replace([np.inf, -np.inf], np.nan).melt(
            id_vars=["sketch_type"],
            value_vars=[col for _rate, col in ratio_cols],
            var_name="comparison",
            value_name="ratio",
        )
        long["comparison"] = long["comparison"].str.replace("_over_", "/", regex=False)
        gain = (
            long.dropna(subset=["ratio"])
            .groupby(["sketch_type", "comparison"])["ratio"]
            .agg(["count", "median"])
            .reset_index()
        )
    else:
        gain = pd.DataFrame(columns=["sketch_type", "comparison", "count", "median"])

    order_comparison = f"{R_ORDER[-1]}/R10" if f"{R_ORDER[-1]}/R10" in set(gain["comparison"]) else "R50/R10"
    sketch_order = (
        gain[gain["comparison"].eq(order_comparison)]
        .sort_values("median")["sketch_type"]
        .tolist()
    )
    for label in gain.sort_values("median")["sketch_type"].tolist():
        if label not in sketch_order:
            sketch_order.append(label)
    comparisons = [f"{rate}/R10" for rate, _col in ratio_cols if f"{rate}/R10" in set(gain["comparison"])]

    fig, ax = plt.subplots(figsize=(9.8, max(3.5, 0.42 * max(1, len(sketch_order)) + 1.5)))
    y = np.arange(len(sketch_order))
    bar_height = min(0.24, 0.72 / max(1, len(comparisons)))
    colors = plt.cm.Set2(np.linspace(0, 1, max(1, len(comparisons))))
    for i, comparison in enumerate(comparisons):
        vals = (
            gain[gain["comparison"].eq(comparison)]
            .set_index("sketch_type")
            .reindex(sketch_order)
        )
        offset = (i - (len(comparisons) - 1) / 2.0) * bar_height
        ax.barh(
            y + offset,
            vals["median"].to_numpy(dtype=float),
            height=bar_height,
            color=colors[i],
            alpha=0.88,
            label=comparison,
        )
    ax.axvline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(sketch_order)
    ax.invert_yaxis()
    ax.set_xlabel("Median relative RMSE ratio versus R10, learned rows")
    ax.set_title("R-grid gain by sketch type")
    ax.grid(axis="x", alpha=0.25)
    if comparisons:
        ax.legend(loc="lower right", fontsize=8)
    comparison_order = {f"{rate}/R10": i for i, rate in enumerate(R_ORDER[1:])}
    gain = gain.copy()
    gain["_comparison_order"] = gain["comparison"].map(comparison_order).fillna(999).astype(int)
    gain = gain.sort_values(["_comparison_order", "median"]).drop(columns=["_comparison_order"])
    return _save(fig, out_path), gain


def _official_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["implementation_status"].eq("official_empirical")].copy()


def _standard_official_rows(official: pd.DataFrame) -> pd.DataFrame:
    """Official empirical rows suitable for headline comparison.

    Exact-zero rows are often oracle/control-like for these synthetic tasks:
    useful for audits, but misleading as a plotted performance comparator.
    The main report keeps only positive-error same-task references.
    """

    if official.empty:
        return official.copy()
    return official[official["metric"].gt(EPS)].copy()


def _learned_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["implementation_status"].eq("learned_empirical")].copy()


def _full_doc_metric_lookup(learned: pd.DataFrame, *, full_doc_leaf: int) -> dict[tuple[str, str, str], float]:
    full = learned[learned["leaf_size"].eq(full_doc_leaf)]
    out: dict[tuple[str, str, str], float] = {}
    for _, row in full.iterrows():
        out[(str(row["task"]), str(row["method_label"]), str(row["R"]))] = float(row["metric"])
    return out


def _with_full_doc_reference(learned: pd.DataFrame, *, full_doc_leaf: int) -> pd.DataFrame:
    out = learned.copy()
    lookup = _full_doc_metric_lookup(out, full_doc_leaf=full_doc_leaf)
    out["full_doc_metric"] = [
        lookup.get((str(row.task), str(row.method_label), str(row.R)), np.nan)
        for row in out.itertuples(index=False)
    ]
    out["full_doc_ratio"] = np.where(
        out["full_doc_metric"] > EPS,
        out["metric"] / out["full_doc_metric"],
        np.nan,
    )
    out["full_doc_delta"] = out["metric"] - out["full_doc_metric"]
    return out


def _best_official_by_task_rate(official: pd.DataFrame, *, positive_only: bool = True) -> pd.DataFrame:
    if official.empty:
        return pd.DataFrame(columns=["task", "R", "official_metric", "official_method"])
    pool = official.copy()
    if positive_only:
        pool = pool[pool["metric"].gt(EPS)].copy()
    else:
        pool = pool[pool["metric"].ge(0)].copy()
    if pool.empty:
        return pd.DataFrame(columns=["task", "R", "official_metric", "official_method"])
    idx = pool.groupby(["task", "R"])["metric"].idxmin()
    cols = ["task", "R", "metric", "method_label", "sketch_type"]
    out = pool.loc[idx, cols].copy()
    out = out.rename(
        columns={
            "metric": "official_metric",
            "method_label": "official_method",
            "sketch_type": "official_sketch_type",
        }
    )
    return out


def _parse_quantile_query(row: pd.Series) -> float:
    for value in (row.get("query"), row.get("learned_run_slug"), row.get("sketch")):
        match = re.search(r"q([0-9]+(?:\.[0-9]+)?)", str(value))
        if match:
            return float(match.group(1))
    return 0.5


def _first_summary_config(run_root: Path | None) -> dict[str, object]:
    if run_root is None:
        return {}
    root = Path(run_root)
    candidates = sorted((root / "classical_sketches" / "paper").glob("**/summary.json"))
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        config = data.get("config")
        if isinstance(config, dict):
            return dict(config)
    return {}


def _base_run_config(run_root: Path | None) -> dict[str, object]:
    config: dict[str, object] = {
        "seed": 0,
        "learned_n_train": 128,
        "learned_n_val": 48,
        "min_tokens": 128,
        "max_tokens": 512,
        "universe_size": 4096,
        "learned_leaf_feature_mode": "count_vector",
    }
    config.update(_first_summary_config(run_root))
    if run_root is not None:
        manifest = Path(run_root) / "paper_bundle_manifest.json"
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            args = data.get("args") if isinstance(data, dict) else None
            if isinstance(args, dict):
                for key in ("learned_n_train", "learned_n_val"):
                    if key in args:
                        config[key] = args[key]
                seeds = str(args.get("seeds", "")).split(",")
                if seeds and seeds[0].strip():
                    config["seed"] = seeds[0].strip()
    return config


def _int_value(value: object, default: int) -> int:
    try:
        if value is None or pd.isna(value):
            return int(default)
    except TypeError:
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _float_value(value: object, default: float) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _base_int(base: dict[str, object], key: str, default: int) -> int:
    return _int_value(base.get(key, default), default)


def _row_int(row: pd.Series, key: str, default: int) -> int:
    return _int_value(row.get(key, default), default)


def _null_key(row: pd.Series, base: dict[str, object]) -> str:
    fields = [
        str(row.get("family", "")),
        str(row.get("query", "")),
        str(row.get("learned_target_kind", "")),
        str(_row_int(row, "distinct_lg_k", 8)),
        str(_row_int(row, "theta_lg_k", _row_int(row, "distinct_lg_k", 8))),
        str(_row_int(row, "cms_num_hashes", 5)),
        str(_row_int(row, "cms_num_buckets", 256)),
        str(_row_int(row, "frequent_lg_max_map_size", 8)),
        str(_row_int(row, "kll_k", 128)),
        str(_row_int(row, "quantiles_k", 128)),
        str(_row_int(row, "req_k", 12)),
        str(_row_int(row, "tdigest_k", 100)),
        str(_row_int(row, "tuple_lg_k", 12)),
        str(_row_int(row, "varopt_k", 64)),
        str(_parse_quantile_query(row)),
        str(_base_int(base, "seed", 0)),
        str(_base_int(base, "learned_n_train", 128)),
        str(_base_int(base, "learned_n_val", 48)),
        str(_base_int(base, "min_tokens", 128)),
        str(_base_int(base, "max_tokens", 512)),
        str(_base_int(base, "universe_size", 4096)),
    ]
    return "|".join(fields)


def _weighted_constant(truth: np.ndarray) -> float:
    if truth.size == 0:
        return 0.0
    weights = 1.0 / np.maximum(1.0, np.abs(truth)) ** 2
    denom = float(np.sum(weights))
    if denom <= 0:
        return float(np.mean(truth))
    return float(np.sum(weights * truth) / denom)


def _rank_at_sorted(sorted_tokens: np.ndarray, value: float) -> float:
    if sorted_tokens.size == 0:
        return 0.0
    return float(np.searchsorted(sorted_tokens, float(value), side="right")) / float(sorted_tokens.size)


def _quantile_constant_null(
    train_tokens: list[list[int]],
    train_targets: np.ndarray,
    val_tokens: list[list[int]],
    q: float,
) -> tuple[float, float, float]:
    if train_targets.size == 0:
        constant = 0.0
    else:
        candidates = np.unique(np.quantile(train_targets, np.linspace(0.0, 1.0, 201)))
        train_sorted = [np.sort(np.asarray(tokens, dtype=np.float64)) for tokens in train_tokens]
        best_loss = float("inf")
        constant = float(candidates[0]) if candidates.size else 0.0
        for candidate in candidates:
            errs = [
                _safe_rel_error(_rank_at_sorted(tokens, float(candidate)), q)
                for tokens in train_sorted
            ]
            loss = float(np.mean(np.square(errs))) if errs else 0.0
            if loss < best_loss:
                best_loss = loss
                constant = float(candidate)
    val_sorted = [np.sort(np.asarray(tokens, dtype=np.float64)) for tokens in val_tokens]
    rel = [_safe_rel_error(_rank_at_sorted(tokens, constant), q) for tokens in val_sorted]
    zero_rel = [_safe_rel_error(_rank_at_sorted(tokens, 0.0), q) for tokens in val_sorted]
    null_rmse = float(math.sqrt(float(np.mean(np.square(rel))))) if rel else 0.0
    zero_rmse = float(math.sqrt(float(np.mean(np.square(zero_rel))))) if zero_rel else 0.0
    return constant, null_rmse, zero_rmse


def _is_set_target_kind(target_kind: str) -> bool:
    return target_kind in {
        "exact_set_union",
        "exact_set_intersection",
        "exact_set_a_not_b",
        "theta_union_reference",
        "theta_intersection_reference",
        "theta_a_not_b_reference",
    }


def _null_documents(cfg: object) -> tuple[list[list[int]], np.ndarray]:
    from unified_g_v1.sketch.classical_parity import (  # type: ignore[import-not-found]
        ClassicalHLLParityConfig,
        generate_documents,
    )
    from unified_g_v1.sketch.learned_scalar_sketch import _build_target_fn  # type: ignore[import-not-found]

    target_kind = str(cfg.target_kind)
    target_fn = _build_target_fn(cfg)
    total = int(cfg.n_train) + int(cfg.n_val)
    if _is_set_target_kind(target_kind):
        base = dict(
            precision=int(cfg.precision),
            n_leaves=1,
            schedule=str(cfg.schedule),
            backend=str(cfg.backend),
            n_val=total,
            universe_size=int(cfg.universe_size),
            min_tokens=max(1, int(cfg.min_tokens) // 2),
            max_tokens=max(1, int(cfg.max_tokens) // 2),
            zipf_alphas=tuple(float(a) for a in cfg.zipf_alphas),
            oracle_kind="analytic",
        )
        a_cfg = ClassicalHLLParityConfig(seed=int(cfg.seed), **base)
        b_cfg = ClassicalHLLParityConfig(seed=int(cfg.seed) + 7919, **base)
        docs = [
            [int(t) for t in a_flat] + [int(t) + int(cfg.universe_size) for t in b_flat]
            for (_a_leaves, _a_truth, a_flat), (_b_leaves, _b_truth, b_flat) in zip(
                generate_documents(a_cfg),
                generate_documents(b_cfg),
            )
        ]
    else:
        data_cfg = ClassicalHLLParityConfig(
            precision=int(cfg.precision),
            n_leaves=int(cfg.n_leaves or 1),
            schedule=str(cfg.schedule),
            backend=str(cfg.backend),
            n_val=total,
            seed=int(cfg.seed),
            universe_size=int(cfg.universe_size),
            min_tokens=int(cfg.min_tokens),
            max_tokens=int(cfg.max_tokens),
            zipf_alphas=tuple(float(a) for a in cfg.zipf_alphas),
            oracle_kind="analytic",
        )
        docs = [[int(t) for t in flat_tokens] for _leaves, _truth, flat_tokens in generate_documents(data_cfg)]
    targets = np.asarray([float(target_fn(tokens)) for tokens in docs], dtype=np.float64)
    return docs, targets


def _constant_null_for_row(row: pd.Series, base: dict[str, object]) -> dict[str, object]:
    from unified_g_v1.sketch.learned_scalar_sketch import (  # type: ignore[import-not-found]
        LearnedScalarSketchConfig,
    )

    target_kind = str(row.get("learned_target_kind") or "")
    q = _parse_quantile_query(row)
    n_leaves_raw = _row_int(row, "n_leaves", -1)
    n_leaves = None if n_leaves_raw < 0 else n_leaves_raw
    cfg = LearnedScalarSketchConfig(
        target_kind=target_kind,  # type: ignore[arg-type]
        precision=_row_int(row, "distinct_lg_k", 8),
        n_leaves=n_leaves,
        leaf_size=_row_int(row, "leaf_size", 64),
        schedule="balanced",
        backend="datasketches" if target_kind == "hll_reference" else "native",
        n_train=_base_int(base, "learned_n_train", 128),
        n_val=_base_int(base, "learned_n_val", 48),
        seed=_base_int(base, "seed", 0),
        universe_size=_base_int(base, "universe_size", 4096),
        min_tokens=_base_int(base, "min_tokens", 128),
        max_tokens=_base_int(base, "max_tokens", 512),
        focus_token=0,
        cms_num_hashes=_row_int(row, "cms_num_hashes", 5),
        cms_num_buckets=_row_int(row, "cms_num_buckets", 256),
        frequent_lg_max_map_size=_row_int(row, "frequent_lg_max_map_size", 8),
        theta_lg_k=_row_int(row, "theta_lg_k", _row_int(row, "distinct_lg_k", 8)),
        quantile_query=q,
        kll_k=_row_int(row, "kll_k", 128),
        quantiles_k=_row_int(row, "quantiles_k", 128),
        req_k=_row_int(row, "req_k", 12),
        tdigest_k=_row_int(row, "tdigest_k", 100),
        tuple_lg_k=_row_int(row, "tuple_lg_k", 12),
        varopt_k=_row_int(row, "varopt_k", 64),
        leaf_feature_mode=str(base.get("learned_leaf_feature_mode", "count_vector")),
    )
    docs, targets = _null_documents(cfg)
    split = int(cfg.n_train)
    train_tokens = docs[:split]
    val_tokens = docs[split : split + int(cfg.n_val)]
    train_truth = targets[:split]
    val_truth = targets[split : split + int(cfg.n_val)]
    if str(row.get("family")) == "quantile":
        constant, null_rmse, zero_rmse = _quantile_constant_null(train_tokens, train_truth, val_tokens, q)
        truth_mean = q
        truth_sd = 0.0
    else:
        constant = _weighted_constant(train_truth)
        rel = [_safe_rel_error(constant, truth) for truth in val_truth]
        zero_rel = [_safe_rel_error(0.0, truth) for truth in val_truth]
        null_rmse = float(math.sqrt(float(np.mean(np.square(rel))))) if rel else 0.0
        zero_rmse = float(math.sqrt(float(np.mean(np.square(zero_rel))))) if zero_rel else 0.0
        truth_mean = float(np.mean(val_truth)) if val_truth.size else 0.0
        truth_sd = float(np.std(val_truth)) if val_truth.size else 0.0
    return {
        "null_key": _null_key(row, base),
        "family": str(row.get("family", "")),
        "query": str(row.get("query", "")),
        "learned_target_kind": target_kind,
        "leaf_size": _row_int(row, "leaf_size", 0),
        "n_leaves": _row_int(row, "n_leaves", -1),
        "null_kind": "train_metric_optimal_constant",
        "constant_null_pred": constant,
        "constant_null_rel_rmse": null_rmse,
        "zero_null_rel_rmse": zero_rmse,
        "val_target_mean": truth_mean,
        "val_target_sd": truth_sd,
    }


def _attach_measured_nulls(
    learned: pd.DataFrame,
    *,
    out_dir: Path,
    run_root: Path | None,
) -> pd.DataFrame:
    out = learned.copy()
    if out.empty:
        return out
    base = _base_run_config(run_root)
    out["null_key"] = out.apply(lambda row: _null_key(row, base), axis=1)
    cache_path = out_dir / NULL_CACHE_NAME
    cached = pd.DataFrame()
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path)
        except Exception:
            cached = pd.DataFrame()
    cached_by_key = {
        str(row["null_key"]): dict(row)
        for _, row in cached.iterrows()
        if "null_key" in cached.columns and pd.notna(row.get("null_key"))
    }
    null_rows: list[dict[str, object]] = []
    unique = out.drop_duplicates("null_key")
    for _, row in unique.iterrows():
        key = str(row["null_key"])
        if key in cached_by_key:
            null_rows.append(cached_by_key[key])
            continue
        try:
            null_rows.append(_constant_null_for_row(row, base))
        except Exception as exc:
            null_rows.append(
                {
                    "null_key": key,
                    "family": str(row.get("family", "")),
                    "query": str(row.get("query", "")),
                    "learned_target_kind": str(row.get("learned_target_kind") or ""),
                    "leaf_size": _row_int(row, "leaf_size", 0),
                    "n_leaves": _row_int(row, "n_leaves", -1),
                    "null_kind": f"unavailable: {type(exc).__name__}",
                    "constant_null_pred": np.nan,
                    "constant_null_rel_rmse": np.nan,
                    "zero_null_rel_rmse": np.nan,
                    "val_target_mean": np.nan,
                    "val_target_sd": np.nan,
                }
            )
    nulls = pd.DataFrame(null_rows).drop_duplicates("null_key")
    if not nulls.empty:
        merged_cache = pd.concat([cached, nulls], ignore_index=True)
        merged_cache = merged_cache.drop_duplicates("null_key", keep="last")
        merged_cache.to_csv(cache_path, index=False)
    out = out.merge(
        nulls[
            [
                "null_key",
                "null_kind",
                "constant_null_pred",
                "constant_null_rel_rmse",
                "zero_null_rel_rmse",
                "val_target_mean",
                "val_target_sd",
            ]
        ],
        on="null_key",
        how="left",
    )
    out["null_rmse"] = _num(out["constant_null_rel_rmse"])
    out["metric_over_null"] = np.where(out["null_rmse"] > EPS, out["metric"] / out["null_rmse"], np.nan)
    out["skill_vs_null"] = 1.0 - out["metric_over_null"]
    return out


def _negative_control_lookup(df: pd.DataFrame) -> dict[tuple[str, str, int], float]:
    controls = df[df["implementation_status"].eq("negative_control")].copy()
    out: dict[tuple[str, str, int], float] = {}
    for _, row in controls.iterrows():
        out[(str(row["task"]), str(row["R"]), int(row["leaf_size"]))] = float(row["metric"])
    return out


def _neural_vs_full_doc_plot(
    *,
    task: str,
    learned_task: pd.DataFrame,
    official_task: pd.DataFrame,
    out_dir: Path,
    leaf_order: tuple[int, ...],
    full_doc_leaf: int,
    filename_suffix: str = "",
    title_note: str = "",
) -> Path:
    methods = (
        learned_task.groupby("method_label")["metric_plot"]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(methods))))
    color_map = {method: colors[i] for i, method in enumerate(methods)}
    fig, axes = plt.subplots(2, len(R_ORDER), figsize=(max(18, 4.8 * len(R_ORDER)), 8.6), sharex=True)
    any_ratio: dict[str, bool] = {rate: False for rate in R_ORDER}
    for col, rate in enumerate(R_ORDER):
        top = axes[0, col]
        bottom = axes[1, col]
        panel = learned_task[learned_task["R"].eq(rate)]
        for method in methods:
            m = panel[panel["method_label"].eq(method)].sort_values("leaf_size")
            if m.empty:
                continue
            x = m["leaf_size"].to_numpy(dtype=float)
            y = m["metric_plot"].to_numpy(dtype=float)
            label = method
            top.plot(
                x,
                y,
                marker="o",
                linewidth=1.7,
                markersize=3.8,
                color=color_map[method],
                label=label,
            )
            full = m[m["leaf_size"].eq(full_doc_leaf)]
            if not full.empty:
                top.scatter(
                    [full_doc_leaf],
                    [float(full["metric_plot"].iloc[0])],
                    marker="*",
                    s=110,
                    color=color_map[method],
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=5,
                )
            ratio = m[m["full_doc_ratio"].notna()]
            if not ratio.empty:
                any_ratio[rate] = True
                bottom.plot(
                    ratio["leaf_size"].to_numpy(dtype=float),
                    ratio["full_doc_ratio"].to_numpy(dtype=float),
                    marker="o",
                    linewidth=1.55,
                    markersize=3.5,
                    color=color_map[method],
                    label=label,
                )

        official_panel = official_task[official_task["R"].eq(rate)]
        official_methods = (
            official_panel.groupby("method_label")["metric_plot"]
            .median()
            .sort_values()
            .index
            .tolist()
        )
        gray = np.linspace(0.15, 0.55, max(1, len(official_methods)))
        for i, method in enumerate(official_methods):
            ref = (
                official_panel[official_panel["method_label"].eq(method)]
                .groupby("leaf_size")["metric_plot"]
                .median()
                .reindex(leaf_order)
            )
            if ref.notna().sum() == 0:
                continue
            top.plot(
                ref.index,
                ref.to_numpy(dtype=float),
                linestyle="--",
                linewidth=1.45,
                color=str(float(gray[i])),
                alpha=0.9,
                label=method,
            )

        for ax in (top, bottom):
            ax.set_xscale("log", base=2)
            ax.set_xticks(leaf_order)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.grid(True, which="both", alpha=0.25)
            ax.axvline(full_doc_leaf, color="black", linewidth=0.8, alpha=0.3)
        top.set_yscale("log")
        top.set_title(rate)
        bottom.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.75)
        bottom.set_yscale("log")
        bottom.set_xlabel("Leaf tokens")
        if not any_ratio[rate]:
            bottom.text(
                0.5,
                0.5,
                "full-doc RMSE is zero\nratio not defined",
                ha="center",
                va="center",
                transform=bottom.transAxes,
                fontsize=9,
                color="0.35",
            )
    axes[0, 0].set_ylabel("Relative RMSE\nlearned solid, standard sketch dashed")
    axes[1, 0].set_ylabel(f"Learned RMSE / leaf-{full_doc_leaf} RMSE")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7.5)
    fig.suptitle(
        f"{task}: learned tree operators vs full-doc neural operator and standard sketch{title_note}",
        y=1.01,
        fontsize=13,
    )
    return _save(fig, out_dir / f"family_{_slug(task)}{filename_suffix}_neural_vs_full_doc_official.png")


def _primary_f_plus_g_rows(learned: pd.DataFrame) -> pd.DataFrame:
    """Rows for the main neural-operator view.

    Excludes exact-state parity/control lanes such as HLL register-space and
    exact union-state recovery. Those are useful diagnostics, but they answer a
    different question from "how good is a learned f+g mergeable projection?"
    """

    return learned[
        learned["learned_variant"].astype(str).eq("fg")
        & learned["projection_kind"].astype(str).eq("mergeable_projection")
    ].copy()


def _component_audit_table(learned: pd.DataFrame) -> pd.DataFrame:
    rows = learned.copy()
    rows["report_role"] = np.where(
        rows["learned_variant"].astype(str).eq("fg")
        & rows["projection_kind"].astype(str).eq("mergeable_projection"),
        "main learned f+g projection",
        "diagnostic/excluded",
    )
    grouped = (
        rows.groupby(
            [
                "report_role",
                "learned_variant",
                "projection_kind",
                "learned_stage_components",
                "learned_trained_stage_components",
                "learned_reused_prefix",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="rows")
        .sort_values(["report_role", "learned_variant", "projection_kind"])
    )
    grouped["learned_variant"] = grouped["learned_variant"].map(
        lambda value: {"fg": "learned f+g", "g": "g-only"}.get(str(value), str(value))
    )
    grouped["learned_reused_prefix"] = grouped["learned_reused_prefix"].map(
        lambda value: "True" if bool(value) else "False"
    )
    return grouped.rename(
        columns={
            "learned_variant": "lane",
            "projection_kind": "projection",
            "learned_stage_components": "requested_components",
            "learned_trained_stage_components": "trained_components",
            "learned_reused_prefix": "reused_prefix",
        }
    )


def _rate_definition_table(run_root: Path | None, observed_rates: Iterable[object]) -> pd.DataFrame:
    base = _base_run_config(run_root)
    n_train = _base_int(base, "learned_n_train", 128)
    observed = set(_ordered_rates(observed_rates))
    rows = []
    for label in R_ORDER:
        rate = _r_rate(label)
        included = label in observed
        rows.append(
            {
                "R": label,
                "uniform_node_query_rate": rate,
                "expected_root_labels_train": rate * float(n_train),
                "in_this_bundle": "yes" if included else "no",
                "meaning": (
                    "all root/leaf/internal node labels observed"
                    if rate >= 1.0 - EPS
                    else "uniform sample over root/leaf/internal training labels"
                ),
            }
        )
    return pd.DataFrame(rows)


def _node_budget_table(primary: pd.DataFrame, run_root: Path | None) -> pd.DataFrame:
    if primary.empty:
        return pd.DataFrame()
    base = _base_run_config(run_root)
    n_train = _base_int(base, "learned_n_train", 128)
    rows = []
    leaf_stats = (
        primary.groupby("leaf_size")["leaf_count_mean"]
        .median()
        .reset_index()
        .sort_values("leaf_size")
    )
    for _, row in leaf_stats.iterrows():
        leaf_count = float(row["leaf_count_mean"])
        # The uniform_all_nodes objective samples root + leaves + non-root
        # internal merge targets. With one leaf, root and leaf supervision are
        # both present, so the node pool has width two.
        node_pool = max(2.0, 2.0 * leaf_count - 1.0)
        out = {
            "leaf_tokens": int(row["leaf_size"]),
            "median_leaf_count": leaf_count,
            "node_pool_per_doc": node_pool,
        }
        for label in R_ORDER:
            rate = _r_rate(label)
            out[f"{label}_expected_train_node_labels"] = rate * float(n_train) * node_pool
        rows.append(out)
    return pd.DataFrame(rows)


def _full_doc_ratio_heatmap(learned_ref: pd.DataFrame, out_path: Path, leaf_order: tuple[int, ...]) -> Path:
    rows = learned_ref.copy()
    rows = rows[rows["full_doc_ratio"].notna()].copy()
    rows["leaf_R"] = rows["leaf_size"].astype(str) + "/" + rows["R"]
    col_order = [f"{leaf}/{rate}" for leaf in leaf_order for rate in R_ORDER]
    row_order = (
        rows.groupby("task_method")["full_doc_ratio"]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    matrix = (
        rows.pivot_table(index="task_method", columns="leaf_R", values="full_doc_ratio", aggfunc="median")
        .reindex(index=row_order, columns=col_order)
    )
    return _heatmap(
        matrix,
        out_path,
        title="Learned tree operator gap to full-doc neural operator",
        cbar_label="log10(RMSE / leaf-512 RMSE)",
        log_scale=True,
        cmap="magma",
        row_font=6,
    )


def _null_ratio_heatmap(learned_ref: pd.DataFrame, out_path: Path, leaf_order: tuple[int, ...]) -> Path | None:
    rows = learned_ref[learned_ref["metric_over_null"].notna()].copy()
    rows = rows[rows["null_rmse"].gt(EPS)].copy()
    if rows.empty:
        return None
    rows["leaf_R"] = rows["leaf_size"].astype(str) + "/" + rows["R"]
    col_order = [f"{leaf}/{rate}" for leaf in leaf_order for rate in R_ORDER]
    row_order = (
        rows.groupby("task_method")["metric_over_null"]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    matrix = (
        rows.pivot_table(index="task_method", columns="leaf_R", values="metric_over_null", aggfunc="median")
        .reindex(index=row_order, columns=col_order)
    )
    return _heatmap(
        matrix,
        out_path,
        title="Learned error as fraction of measured constant-null error",
        cbar_label="RMSE / train-fitted constant-null RMSE",
        log_scale=False,
        cmap="viridis_r",
        row_font=6,
    )


def _full_doc_to_official_plot(
    learned_ref: pd.DataFrame,
    official: pd.DataFrame,
    out_path: Path,
    *,
    full_doc_leaf: int,
) -> tuple[Path | None, pd.DataFrame]:
    official_best = _best_official_by_task_rate(official)
    full = learned_ref[learned_ref["leaf_size"].eq(full_doc_leaf)].copy()
    merged = full.merge(official_best, on=["task", "R"], how="left")
    merged["full_doc_over_official"] = np.where(
        merged["official_metric"] > EPS,
        merged["metric"] / merged["official_metric"],
        np.nan,
    )
    plot_rows = merged[merged["full_doc_over_official"].notna()].copy()
    if plot_rows.empty:
        return None, merged
    row_order = (
        plot_rows.groupby("task_method")["full_doc_over_official"]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    matrix = (
        plot_rows.pivot_table(
            index="task_method",
            columns="R",
            values="full_doc_over_official",
            aggfunc="median",
        )
        .reindex(index=row_order, columns=R_ORDER)
    )
    path = _heatmap(
        matrix,
        out_path,
        title="Full-doc learned neural operator vs standard official sketch",
        cbar_label="log10(leaf-512 learned RMSE / standard official RMSE)",
        log_scale=True,
        cmap="viridis",
        row_font=6,
    )
    return path, merged


def _family_summary_table(
    learned_ref: pd.DataFrame,
    official: pd.DataFrame,
    *,
    full_doc_leaf: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    standard_official_best = _best_official_by_task_rate(official, positive_only=True)
    rows: list[dict[str, object]] = []
    for task, task_rows in learned_ref.groupby("task"):
        non_full = task_rows[task_rows["leaf_size"].ne(full_doc_leaf)].copy()
        if non_full.empty:
            continue
        best_idx = non_full["metric"].idxmin()
        best = non_full.loc[best_idx]
        fd = task_rows[
            task_rows["method_label"].eq(best["method_label"])
            & task_rows["R"].eq(best["R"])
            & task_rows["leaf_size"].eq(full_doc_leaf)
        ]
        official_match = standard_official_best[
            standard_official_best["task"].eq(task)
            & standard_official_best["R"].eq(best["R"])
        ]
        full_doc_metric = float(fd["metric"].iloc[0]) if not fd.empty else np.nan
        full_doc_null = float(fd["null_rmse"].iloc[0]) if not fd.empty and "null_rmse" in fd else np.nan
        null_metric = float(best.get("null_rmse", np.nan))
        if not math.isfinite(null_metric) and math.isfinite(full_doc_null):
            null_metric = full_doc_null
        official_metric = (
            float(official_match["official_metric"].iloc[0])
            if not official_match.empty
            else np.nan
        )
        official_over_null = (
            official_metric / null_metric
            if math.isfinite(official_metric)
            and math.isfinite(null_metric)
            and null_metric > EPS
            else np.nan
        )
        composition_loss_share = (
            (float(best["metric"]) - full_doc_metric) / (null_metric - full_doc_metric)
            if math.isfinite(full_doc_metric)
            and math.isfinite(null_metric)
            and null_metric > full_doc_metric + EPS
            else np.nan
        )
        rows.append(
            {
                "task": task,
                "best_tree_leaf": int(best["leaf_size"]),
                "R": best["R"],
                "method": best["method_label"],
                "tree_rmse": float(best["metric"]),
                "full_doc_rmse": full_doc_metric,
                "tree_over_full_doc": (
                    float(best["metric"]) / full_doc_metric
                    if math.isfinite(full_doc_metric) and full_doc_metric > EPS
                    else np.nan
                ),
                "constant_null_rmse": null_metric,
                "tree_over_null": (
                    float(best["metric"]) / null_metric
                    if math.isfinite(null_metric) and null_metric > EPS
                    else np.nan
                ),
                "full_doc_over_null": (
                    full_doc_metric / null_metric
                    if math.isfinite(full_doc_metric)
                    and math.isfinite(null_metric)
                    and null_metric > EPS
                    else np.nan
                ),
                "composition_loss_share": composition_loss_share,
                "standard_official_rmse": official_metric,
                "standard_official_over_null": official_over_null,
                "tree_over_standard_official": (
                    float(best["metric"]) / official_metric
                    if math.isfinite(official_metric) and official_metric > EPS
                    else np.nan
                ),
                "full_doc_over_standard_official": (
                    full_doc_metric / official_metric
                    if math.isfinite(full_doc_metric)
                    and math.isfinite(official_metric)
                    and official_metric > EPS
                    else np.nan
                ),
            }
        )
    numeric = pd.DataFrame(rows)
    if numeric.empty:
        return numeric, numeric
    formatted = numeric.copy()
    for col in (
        "tree_rmse",
        "full_doc_rmse",
        "tree_over_full_doc",
        "constant_null_rmse",
        "tree_over_null",
        "full_doc_over_null",
        "composition_loss_share",
        "standard_official_rmse",
        "standard_official_over_null",
        "tree_over_standard_official",
        "full_doc_over_standard_official",
    ):
        formatted[col] = formatted[col].map(_fmt)
    return formatted, numeric


def _context_bar_plot(context: pd.DataFrame, out_path: Path) -> Path | None:
    if context.empty:
        return None
    tasks = list(context["task"])
    y = np.arange(len(tasks))
    height = 0.18
    fig, ax = plt.subplots(figsize=(11, max(4.8, 0.48 * len(tasks) + 1.8)))
    series = [
        ("best tree f+g", "tree_over_null", "#4C78A8", -1.5 * height),
        ("full-doc f+g", "full_doc_over_null", "#F58518", -0.5 * height),
        ("standard official", "standard_official_over_null", "#54A24B", 0.5 * height),
        ("constant null", "_null_line", "#B279A2", 1.5 * height),
    ]
    for label, col, color, offset in series:
        if col == "_null_line":
            vals = np.ones(len(context), dtype=float)
        else:
            vals = pd.to_numeric(context[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(vals) & (vals >= 0)
        ax.barh(y[mask] + offset, vals[mask], height=height, label=label, color=color, alpha=0.88)
    vals = pd.to_numeric(
        context[["tree_over_null", "full_doc_over_null", "standard_official_over_null"]].stack(),
        errors="coerce",
    ).to_numpy(dtype=float)
    finite = vals[np.isfinite(vals) & (vals >= 0)]
    xmax = max(1.05, float(np.nanmax(finite)) * 1.08) if finite.size else 1.05
    ax.set_xlim(0.0, min(max(xmax, 1.05), 2.5))
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(tasks)
    ax.invert_yaxis()
    ax.set_xlabel("Relative RMSE / train-fitted constant-null RMSE")
    ax.set_title("Null-normalized loss context: 0 is perfect, 1 is the measured null")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, out_path)


def _family_summary_series(rows: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if rows.empty or value_col not in rows.columns:
        return pd.DataFrame(columns=["family", "leaf_size", "R", value_col])
    values = rows.copy()
    values[value_col] = pd.to_numeric(values[value_col], errors="coerce")
    values = values[values[value_col].notna()].copy()
    if values.empty:
        return pd.DataFrame(columns=["family", "leaf_size", "R", value_col])
    per_task = (
        values.groupby(["family", "task", "leaf_size", "R"], dropna=False)[value_col]
        .min()
        .reset_index()
    )
    return (
        per_task.groupby(["family", "leaf_size", "R"], dropna=False)[value_col]
        .mean()
        .reset_index()
    )


def _style_family_axis(
    ax: plt.Axes,
    *,
    leaf_order: tuple[int, ...],
    family: str,
    leaf_counts: pd.Series,
    show_top: bool,
) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(leaf_order)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("leaf size (tokens)")
    if show_top:
        top = ax.twiny()
        top.set_xscale("log", base=2)
        top.set_xlim(ax.get_xlim())
        top.set_xticks(leaf_order)
        labels = []
        for leaf in leaf_order:
            value = leaf_counts.get((family, leaf), np.nan)
            labels.append(_fmt(value) if pd.notna(value) else "")
        top.set_xticklabels(labels)
        top.tick_params(axis="x", labelsize=8, pad=1)
        top.set_xlabel("leaves/doc", labelpad=3)


def _summary_style_overview_plot(
    primary: pd.DataFrame,
    official: pd.DataFrame,
    out_path: Path,
    *,
    leaf_order: tuple[int, ...],
) -> Path | None:
    if primary.empty:
        return None
    families = [family for family in ("distinct", "frequency", "quantile", "sampling", "set") if family in set(primary["family"])]
    if not families:
        families = sorted(primary["family"].dropna().astype(str).unique())
    if not families:
        return None

    raw = _family_summary_series(primary, "metric")
    norm = _family_summary_series(primary, "metric_over_null")

    null_ref = (
        primary.groupby(["task", "leaf_size", "R"], dropna=False)["null_rmse"]
        .median()
        .reset_index()
    )
    official_norm = official.merge(null_ref, on=["task", "leaf_size", "R"], how="left")
    official_norm["metric_over_null"] = np.where(
        official_norm["null_rmse"] > EPS,
        official_norm["metric"] / official_norm["null_rmse"],
        np.nan,
    )
    official_raw = _family_summary_series(official_norm, "metric")
    official_ratio = _family_summary_series(official_norm, "metric_over_null")

    leaf_counts = primary.groupby(["family", "leaf_size"])["leaf_count_mean"].median()
    colors = {
        "R10": "#4C78A8",
        "R30": "#F58518",
        "R50": "#54A24B",
        "R100": "#B279A2",
    }
    fig, axes = plt.subplots(
        2,
        len(families),
        figsize=(max(12.0, 2.75 * len(families)), 6.2),
        squeeze=False,
        sharex=False,
        constrained_layout=False,
    )
    row_specs = [
        (raw, official_raw, "relative RMSE"),
        (norm, official_ratio, "RMSE / constant null"),
    ]
    for row_idx, (learned_rows, official_rows, ylabel) in enumerate(row_specs):
        for col_idx, family in enumerate(families):
            ax = axes[row_idx, col_idx]
            panel = learned_rows[learned_rows["family"].astype(str).eq(family)]
            for rate in R_ORDER:
                series = (
                    panel[panel["R"].eq(rate)]
                    .groupby("leaf_size")[row_specs[row_idx][0].columns[-1]]
                    .median()
                    .reindex(leaf_order)
                )
                if series.notna().sum() == 0:
                    continue
                ax.plot(
                    series.index,
                    series.to_numpy(dtype=float),
                    marker="o",
                    linewidth=1.65,
                    markersize=3.8,
                    color=colors.get(rate, "#333333"),
                    label=rate,
                )
            off_panel = official_rows[official_rows["family"].astype(str).eq(family)]
            if not off_panel.empty:
                value_col = row_specs[row_idx][0].columns[-1]
                off = (
                    off_panel.groupby("leaf_size")[value_col]
                    .median()
                    .reindex(leaf_order)
                )
                if off.notna().sum() > 0:
                    ax.plot(
                        off.index,
                        off.to_numpy(dtype=float),
                        linestyle="--",
                        marker="s",
                        linewidth=1.35,
                        markersize=3.2,
                        color="#333333",
                        alpha=0.75,
                        label="official positive",
                    )
            if row_idx == 0:
                ax.set_title(family, pad=4)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            _style_family_axis(
                ax,
                leaf_order=leaf_order,
                family=family,
                leaf_counts=leaf_counts,
                show_top=row_idx == 0,
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(handles)), fontsize=8)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.13, top=0.86, wspace=0.35, hspace=0.42)
    fig.suptitle("Summary-style view: best learned f+g by family, leaf size, and supervision rate", y=0.99)
    return _save(fig, out_path)


def _task_verdict_table(primary: pd.DataFrame) -> pd.DataFrame:
    if primary.empty or "metric_over_null" not in primary.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for task, group in primary[primary["metric_over_null"].notna()].groupby("task"):
        if group.empty:
            continue
        best = group.loc[group["metric_over_null"].idxmin()]
        score = float(best["metric_over_null"])
        if score <= 0.10:
            verdict = "strong"
        elif score <= 0.25:
            verdict = "working"
        elif score <= 0.50:
            verdict = "partial"
        else:
            verdict = "weak"
        rows.append(
            {
                "task": task,
                "verdict": verdict,
                "best_R": best["R"],
                "best_leaf": int(best["leaf_size"]),
                "best_method": best["method_label"],
                "best_rmse": float(best["metric"]),
                "best_over_null": score,
                "tree_over_full_doc": float(best.get("full_doc_ratio", np.nan)),
            }
        )
    out = pd.DataFrame(rows).sort_values(["best_over_null", "task"])
    for col in ("best_rmse", "best_over_null", "tree_over_full_doc"):
        if col in out.columns:
            out[col] = out[col].map(_fmt)
    return out


def _hll_focus_plot(
    primary: pd.DataFrame,
    official: pd.DataFrame,
    out_path: Path,
    *,
    leaf_order: tuple[int, ...],
    full_doc_leaf: int,
) -> tuple[Path | None, pd.DataFrame]:
    task = "distinct/cardinality"
    learned_task = primary[primary["task"].eq(task)].copy()
    official_task = official[official["task"].eq(task)].copy()
    learned_hll = learned_task[learned_task["learned_target_kind"].astype(str).eq("hll_reference")].copy()
    official_hll = official_task[official_task["sketch"].astype(str).eq("hll_datasketches")].copy()
    if learned_hll.empty or official_hll.empty:
        return None, pd.DataFrame()

    best_idx = learned_task.groupby(["leaf_size", "R"])["metric"].idxmin()
    best = learned_task.loc[best_idx].copy()
    table = learned_hll[
        ["leaf_size", "R", "metric", "full_doc_metric", "metric_over_null"]
    ].rename(
        columns={
            "metric": "learned_hll_f_plus_g_rmse",
            "full_doc_metric": "full_doc_learned_hll_rmse",
            "metric_over_null": "learned_hll_over_null",
        }
    )
    table = table.merge(
        official_hll[["leaf_size", "R", "metric"]].rename(columns={"metric": "official_hll_rmse"}),
        on=["leaf_size", "R"],
        how="left",
    )
    table = table.merge(
        best[["leaf_size", "R", "method_label", "learned_target_kind", "metric"]].rename(
            columns={
                "method_label": "best_learned_f_plus_g_method",
                "learned_target_kind": "best_learned_target_kind",
                "metric": "best_learned_f_plus_g_rmse",
            }
        ),
        on=["leaf_size", "R"],
        how="left",
    )
    table["learned_hll_over_official_hll"] = np.where(
        table["official_hll_rmse"] > EPS,
        table["learned_hll_f_plus_g_rmse"] / table["official_hll_rmse"],
        np.nan,
    )
    table["learned_hll_over_full_doc_hll"] = np.where(
        table["full_doc_learned_hll_rmse"] > EPS,
        table["learned_hll_f_plus_g_rmse"] / table["full_doc_learned_hll_rmse"],
        np.nan,
    )
    table["best_learned_over_official_hll"] = np.where(
        table["official_hll_rmse"] > EPS,
        table["best_learned_f_plus_g_rmse"] / table["official_hll_rmse"],
        np.nan,
    )

    fig, axes = plt.subplots(1, len(R_ORDER), figsize=(max(15.5, 4.2 * len(R_ORDER)), 4.5), sharey=True)
    for ax, rate in zip(axes, R_ORDER):
        learned_line = (
            learned_hll[learned_hll["R"].eq(rate)]
            .groupby("leaf_size")["metric_plot"]
            .median()
            .reindex(leaf_order)
        )
        official_line = (
            official_hll[official_hll["R"].eq(rate)]
            .groupby("leaf_size")["metric_plot"]
            .median()
            .reindex(leaf_order)
        )
        best_line = (
            best[best["R"].eq(rate)]
            .groupby("leaf_size")["metric_plot"]
            .median()
            .reindex(leaf_order)
        )
        ax.plot(
            learned_line.index,
            learned_line.to_numpy(dtype=float),
            marker="o",
            linewidth=1.9,
            color="#4C78A8",
            label="learned f+g HLL target",
        )
        ax.plot(
            official_line.index,
            official_line.to_numpy(dtype=float),
            marker="s",
            linewidth=1.6,
            linestyle="--",
            color="#54A24B",
            label="official empirical HLL",
        )
        ax.plot(
            best_line.index,
            best_line.to_numpy(dtype=float),
            marker="^",
            linewidth=1.5,
            linestyle=":",
            color="#F58518",
            label="best learned f+g distinct",
        )
        full = learned_line.loc[full_doc_leaf] if full_doc_leaf in learned_line.index else np.nan
        if pd.notna(full):
            ax.scatter(
                [full_doc_leaf],
                [float(full)],
                marker="*",
                s=120,
                color="#4C78A8",
                edgecolor="black",
                linewidth=0.5,
                zorder=5,
            )
        ax.set_title(rate)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(leaf_order)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("Leaf tokens")
    axes[0].set_ylabel("Relative RMSE")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    table = table.copy()
    table["_R_order"] = table["R"].map({rate: i for i, rate in enumerate(R_ORDER)}).fillna(999).astype(int)
    table = table.sort_values(["leaf_size", "_R_order"]).drop(columns=["_R_order"]).reset_index(drop=True)
    fig.suptitle("HLL focus: official empirical HLL vs learned f+g HLL target", y=1.02)
    return _save(fig, out_path), table


def _task_rate_frontier(
    primary: pd.DataFrame,
    out_path: Path,
) -> tuple[Path | None, pd.DataFrame]:
    if primary.empty:
        return None, pd.DataFrame()
    rows = []
    for (task, rate), group in primary.groupby(["task", "R"]):
        best = group.loc[group["metric"].idxmin()]
        rows.append(
            {
                "task": task,
                "R": rate,
                "best_leaf": int(best["leaf_size"]),
                "best_method": str(best["method_label"]),
                "best_rmse": float(best["metric"]),
                "best_over_null": float(best.get("metric_over_null", np.nan)),
            }
        )
    best = pd.DataFrame(rows)
    if best.empty:
        return None, best
    pivot = best.pivot_table(index="task", columns="R", values="best_rmse", aggfunc="min").reindex(columns=R_ORDER)
    norm = pivot.divide(pivot["R10"], axis=0)
    table = best.pivot_table(
        index="task",
        columns="R",
        values=["best_leaf", "best_method", "best_rmse", "best_over_null"],
        aggfunc="first",
    )
    flat = pd.DataFrame(index=table.index)
    for rate in R_ORDER:
        for field in ("best_leaf", "best_method", "best_rmse", "best_over_null"):
            if (field, rate) in table.columns:
                flat[f"{rate}_{field}"] = table[(field, rate)]
    if "R10" in pivot.columns:
        for rate in R_ORDER[1:]:
            if rate in pivot.columns:
                flat[f"{rate}_over_R10_best_rmse"] = np.where(
                    pivot["R10"] > EPS,
                    pivot[rate] / pivot["R10"],
                    np.nan,
                )
    flat = flat.reset_index()

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = np.arange(len(R_ORDER))
    for task in norm.index:
        vals = norm.loc[task, list(R_ORDER)].to_numpy(dtype=float)
        if np.isfinite(vals).sum() == 0:
            continue
        ax.plot(x, vals, marker="o", linewidth=1.5, label=str(task))
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(R_ORDER)
    ax.set_ylabel("Best learned f+g RMSE / same-task R10 best RMSE")
    ax.set_title("R frontier by task: lower means denser supervision helped")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7.5)
    return _save(fig, out_path), flat


def _make_report(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    aggregate_csv: Path,
    run_root: Path | None,
    leaf_order: tuple[int, ...] = DEFAULT_LEAF_ORDER,
) -> Path:
    leaf_order = tuple(x for x in leaf_order if x in set(df["leaf_size"]))
    full_doc_leaf = max(leaf_order)
    learned_all = _with_full_doc_reference(_learned_rows(df), full_doc_leaf=full_doc_leaf)
    official_all = _official_rows(df)
    lean_omitted = int(df["implementation_status"].eq("lean_backed").sum())
    official = _standard_official_rows(official_all)
    component_audit = _component_audit_table(learned_all)
    primary = _with_full_doc_reference(_primary_f_plus_g_rows(learned_all), full_doc_leaf=full_doc_leaf)
    primary = _attach_measured_nulls(primary, out_dir=out_dir, run_root=run_root)
    present_rates = _ordered_rates(primary["R"].dropna().unique())
    present_rate_text = "/".join(present_rates) if present_rates else "/".join(R_ORDER)
    has_r100 = "R100" in present_rates
    source_summary = _source_summary_table(df)

    null_ratio_path = _null_ratio_heatmap(
        primary,
        out_dir / "learned_error_over_constant_null_heatmap.png",
        leaf_order,
    )
    full_doc_gap_path = _full_doc_ratio_heatmap(
        primary,
        out_dir / "learned_gap_to_full_doc_heatmap.png",
        leaf_order,
    )
    full_doc_official_path, full_doc_official = _full_doc_to_official_plot(
        primary,
        official,
        out_dir / "full_doc_learned_vs_official_heatmap.png",
        full_doc_leaf=full_doc_leaf,
    )

    task_breakouts: list[tuple[str, Path]] = []
    learned_tasks = sorted(primary["task"].unique())
    for task in learned_tasks:
        primary_task = primary[primary["task"].eq(task)].copy()
        if primary_task.empty:
            continue
        task_breakouts.append(
            (
                task,
                _neural_vs_full_doc_plot(
                    task=task,
                    learned_task=primary_task,
                    official_task=official[official["task"].eq(task)].copy(),
                    out_dir=out_dir,
                    leaf_order=leaf_order,
                    full_doc_leaf=full_doc_leaf,
                    filename_suffix="_primary_f_plus_g",
                    title_note=" (primary learned f+g projection rows)",
                ),
            )
        )

    gain_path, gain = _rate_gain_plot(primary, out_dir / "learned_rate_gain_by_sketch_type.png")

    coverage = (
        primary.groupby(["leaf_size", "R"])
        .size()
        .unstack("R")
        .reindex(index=leaf_order, columns=R_ORDER)
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"leaf_size": "leaf_tokens"})
    )
    filled_leaf_r_cells = int((coverage.set_index("leaf_tokens")[list(R_ORDER)] > 0).sum().sum())
    expected_leaf_r_cells = len(leaf_order) * len(R_ORDER)
    median_by_leaf = (
        primary.groupby(["leaf_size", "R"])["metric"]
        .median()
        .unstack("R")
        .reindex(index=leaf_order, columns=R_ORDER)
        .reset_index()
        .rename(columns={"leaf_size": "leaf_tokens"})
    )
    family_summary, family_summary_numeric = _family_summary_table(
        primary,
        official,
        full_doc_leaf=full_doc_leaf,
    )
    rate_defs = _rate_definition_table(run_root, present_rates)
    node_budget = _node_budget_table(primary, run_root)
    task_rate_path, task_rate_table_numeric = _task_rate_frontier(
        primary,
        out_dir / "best_learned_f_plus_g_r_frontier.png",
    )
    task_rate_table = task_rate_table_numeric.copy()
    if not task_rate_table.empty:
        for col in task_rate_table.columns:
            if (
                col.endswith("_best_rmse")
                or col.endswith("_best_over_null")
                or col.endswith("_over_R10_best_rmse")
            ):
                task_rate_table[col] = task_rate_table[col].map(_fmt)
    context_plot_path = _context_bar_plot(
        family_summary_numeric,
        out_dir / "context_tree_full_doc_official_constant_null.png",
    )
    summary_style_path = _summary_style_overview_plot(
        primary,
        official,
        out_dir / "summary_style_primary_f_plus_g_by_family.png",
        leaf_order=leaf_order,
    )
    task_verdict = _task_verdict_table(primary)
    hll_focus_path, hll_focus_table_numeric = _hll_focus_plot(
        primary,
        official,
        out_dir / "hll_official_vs_learned_f_plus_g.png",
        leaf_order=leaf_order,
        full_doc_leaf=full_doc_leaf,
    )
    hll_focus_table = hll_focus_table_numeric.copy()
    if not hll_focus_table.empty:
        keep_cols = [
            "leaf_size",
            "R",
            "learned_hll_f_plus_g_rmse",
            "official_hll_rmse",
            "learned_hll_over_official_hll",
            "full_doc_learned_hll_rmse",
            "learned_hll_over_full_doc_hll",
            "learned_hll_over_null",
            "best_learned_f_plus_g_method",
            "best_learned_f_plus_g_rmse",
            "best_learned_over_official_hll",
        ]
        hll_focus_table = hll_focus_table[keep_cols]
        for col in (
            "learned_hll_f_plus_g_rmse",
            "official_hll_rmse",
            "learned_hll_over_official_hll",
            "full_doc_learned_hll_rmse",
            "learned_hll_over_full_doc_hll",
            "learned_hll_over_null",
            "best_learned_f_plus_g_rmse",
            "best_learned_over_official_hll",
        ):
            hll_focus_table[col] = hll_focus_table[col].map(_fmt)

    report = []
    report.append("# Classical Sketch R-Grid Report\n")
    report.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
    report.append(f"Primary source CSV: [{aggregate_csv.name}](../{aggregate_csv.name})\n")
    if run_root is not None:
        manifest = run_root / "paper_bundle_manifest.json"
        report.append(f"Bundle manifest: [{manifest.name}](../../../{manifest.name})\n")
    report.append(
        f"This report treats `leaf_size={full_doc_leaf}` as the full-doc neural-operator reference "
        "for this bundle. The main question is not whether the official implementation wins; it should. "
        "The question is how much performance is lost as the neural operator is forced through smaller "
        "tree leaves, and how far the full-doc neural operator remains from vetted standard sketch references.\n"
    )
    if not source_summary.empty:
        report.append("## Source Inputs\n")
        report.append(
            "The primary aggregate supplies the current R-grid. Extra aggregate CSVs only fill R values missing from the primary aggregate, so the R10/R30/R50 cells here are not averaged with older runs.\n"
        )
        report.append(_md_table(source_summary))

    report.append("## Comparison Scheme\n")
    report.append(
        f"- **Full-doc neural reference:** `leaf_size={full_doc_leaf}` has exactly one leaf per document in this run, so it is the full-doc neural-operator point.\n"
    )
    report.append(
        "- **Within-family plots:** each learned sketch task gets its own figure. The top row is absolute relative RMSE against vetted same-task standard sketch references; the bottom row is learned RMSE divided by the same method's full-doc RMSE.\n"
    )
    report.append(
        "- **Primary learned lines:** the main family plots show only the learned `f+g` mergeable-projection lane: `learned_variant=fg`, `projection_kind=mergeable_projection`, `requested_components=['f','g']`, `trained_components=['f','g']`, and `reused_prefix=False`. Here `f` is the learned scalar readout/projection and `g` is the learned merge/composition operator. The separate `g-only` rows in the aggregate are exact-state diagnostic lanes and are excluded from the headline plots.\n"
    )
    report.append(
        "- **Combined alignment:** the cross-family heatmap uses `RMSE(leaf, R) / RMSE(full-doc, R)` for learned rows only. That aligns unlike sketches by asking how much tree decomposition costs relative to each method's own full-doc neural version.\n"
    )
    report.append(
        "- **Measured null normalization:** the report also fits a constant predictor on each learned target's train split and evaluates it on the same validation split with the same relative-RMSE metric. `RMSE / constant-null RMSE` is the cross-task scale: `1` is the measured null, `0` is perfect, and values above `1` are worse than the null.\n"
    )
    report.append(
        "- **Standard sketch comparators:** headline comparisons keep only positive-error same-task official empirical sketch rows. Lean-backed, exact-zero official, control, and negative-control rows are omitted from the main plots and tables because they are audit/oracle diagnostics, not stable performance baselines.\n"
    )
    report.append("## How To Read Whether It Is Working\n")
    report.append(
        f"Use four separate checks. First, `RMSE / constant-null RMSE` says whether the learned operator is meaningfully above a measured no-sketch baseline. Second, the `leaf={full_doc_leaf}` star says whether the full-doc neural operator can learn the target at all; if that is already far from the standard sketch reference, the bottleneck is not tree composition. Third, the bottom row in each family plot says how much error tree composition adds relative to that full-doc neural point; values near `1` are good, values around `2` mean the tree path roughly doubles the neural error. Fourth, compare the R columns: if denser R settings improve over `R10`, local-law supervision is helping.\n"
    )
    report.append(
        "The official sketch is a reference, not a competitor in the winner sense. When an official/control row is exact-zero on this synthetic task, the main report leaves it out rather than treating it as a SOTA baseline.\n"
    )

    report.append("## Coverage\n")
    report.append(f"- Total aggregate rows: `{len(df)}`\n")
    report.append(f"- Learned rows: `{len(learned_all)}`\n")
    report.append(f"- Primary learned f+g projection rows: `{len(primary)}`\n")
    report.append(f"- Standard same-task official comparator rows used: `{len(official)}` (`{len(official_all) - len(official)}` exact-zero official rows and `{lean_omitted}` Lean-backed rows omitted)\n")
    report.append(f"- Leaf/R cells: `{filled_leaf_r_cells}/{expected_leaf_r_cells}` over `{present_rate_text}` and leaf sizes `{','.join(str(x) for x in leaf_order)}`\n")
    report.append(_md_table(coverage))

    report.append("## Learned Component Audit\n")
    report.append(
        "This table separates the included learned f+g projection lane from diagnostic rows. The included rows train both `f` and `g`; rows that learn only `g` use exact/oracle state spaces and are excluded from the main comparison.\n"
    )
    report.append(_md_table(component_audit, max_rows=40))

    if summary_style_path is not None:
        report.append("## Summary-Style Overview\n")
        report.append(
            "This is the up-to-date analogue of `classical_sketches_summary.png`, computed from the merged R10/R30/R50/R100 primary learned f+g rows. Within each family/leaf/R cell, it first picks the best primary f+g row per task and then averages across tasks in that family. The top row shows raw relative RMSE; the bottom row uses the measured constant-null denominator, where lower is better and `1` is the no-sketch constant baseline.\n"
        )
        report.append(f"![Summary-style primary f+g overview]({summary_style_path.name})\n")
    if not task_verdict.empty:
        report.append("## Task-Level Verdict\n")
        report.append(
            "The table below uses the best primary f+g point over leaf size and R for each task. The verdict bands are based on `best_over_null`: strong `<=0.10`, working `<=0.25`, partial `<=0.50`, weak `>0.50`.\n"
        )
        report.append(_md_table(task_verdict, max_rows=80))

    report.append("## Supervision Rate Semantics\n")
    if has_r100:
        report.append(
            f"`{present_rate_text}` are training-supervision densities, not evaluation subsets. With `learned_supervision_sampling_policy=uniform_all_nodes`, the same node-query rate is applied to root, leaf, and non-root internal labels during training. Validation RMSE is still computed on the full validation set. `R100` is included here from the matched full-supervision aggregate and means all root/leaf/internal training labels are observed.\n"
        )
    else:
        report.append(
            f"`{present_rate_text}` are training-supervision densities, not evaluation subsets. With `learned_supervision_sampling_policy=uniform_all_nodes`, the same node-query rate is applied to root, leaf, and non-root internal labels during training. Validation RMSE is still computed on the full validation set. `R100` is the natural all-label setting but is not present in the loaded aggregates.\n"
        )
    report.append(_md_table(rate_defs))
    report.append(
        "The node-label budget depends on leaf size because smaller leaves create more supervised leaf/internal nodes. The values below are expected training labels under the uniform node sampler.\n"
    )
    report.append(_md_table(node_budget))

    report.append("## Combined Learned Alignment\n")
    report.append(
        "This first heatmap is the cross-task normalization: values are learned `RMSE / RMSE(constant null)` for a train-fitted constant predictor evaluated on the same validation examples. Smaller is better; `1` is the measured null baseline.\n"
    )
    if null_ratio_path is not None:
        report.append(f"![Learned error over constant null]({null_ratio_path.name})\n")
    report.append(
        f"This second heatmap isolates tree-composition cost: values are learned `RMSE(leaf, R) / RMSE(leaf={full_doc_leaf}, R)` for the same method. `1` means the tree-composed operator matches its full-doc neural version. Exact-zero full-doc methods are excluded because the ratio is undefined.\n"
    )
    report.append(f"![Learned gap to full doc]({full_doc_gap_path.name})\n")
    if full_doc_official_path is not None:
        report.append(
            "This heatmap asks the second question: for the full-doc neural operator only, how far is it from the best positive standard sketch reference for the same task and R?\n"
        )
        report.append(f"![Full-doc learned vs official]({full_doc_official_path.name})\n")
    report.append(
        "The rate-gain plot summarizes how much denser learned supervision helps each sketch type. Values below `1` mean the denser setting improves over R10.\n"
    )
    report.append(f"![Learned rate gain by sketch type]({gain_path.name})\n")

    if task_rate_path is not None and not task_rate_table.empty:
        report.append("## R Frontier Across Tasks\n")
        report.append(
            f"For each task and R value, this table picks the best learned f+g point over leaf sizes and learned sketch targets. The plot normalizes each task to its own R10 best point, so values below `1` mean denser training supervision helped. Loaded R values: `{present_rate_text}`.\n"
        )
        report.append(f"![Best learned f+g R frontier]({task_rate_path.name})\n")
        report.append(_md_table(task_rate_table, max_rows=80))

    if hll_focus_path is not None and not hll_focus_table.empty:
        report.append("## HLL Focus\n")
        report.append(
            f"This is the clean HLL comparison for `distinct/cardinality`. `learned_hll_f_plus_g_rmse` is our learned f+g operator trained on the HLL reference target. `official_hll_rmse` is the official empirical HLL implementation. The `best_learned_f_plus_g_*` columns show whether another learned distinct target in the same f+g family does better than the HLL-targeted row. Loaded R values: `{present_rate_text}`.\n"
        )
        report.append(f"![HLL focus]({hll_focus_path.name})\n")
        report.append(_md_table(hll_focus_table, max_rows=80))

    report.append("## Loss Context\n")
    report.append(
        "This bar plot uses the same measured-null denominator. It compares the best non-full-doc primary learned f+g point, the matching full-doc learned f+g point, and the best positive standard sketch reference when present. Lower is better; `1` is the train-fitted constant null and `0` is exact.\n"
    )
    if context_plot_path is not None:
        report.append(f"![Loss context]({context_plot_path.name})\n")
    report.append(
        "The constant null replaces the earlier unit-error yardstick: it is measured from the same synthetic distribution, uses the same metric, and is available for every primary learned target. For external SOTA context, the relevant rows in this bundle are the positive-error Apache DataSketches-style implementations (`HLL/CPC/Theta/KLL/REQ/T-Digest/Count-Min/FrequentStrings/Tuple/VarOpt`). Exact-zero and negative-control rows remain diagnostic artifacts, not headline comparators; no separate neural SOTA baseline was run.\n"
    )

    report.append("## Primary Learned F+G Median Relative RMSE by Leaf and R\n")
    report.append(_md_table(median_by_leaf))

    report.append("## Primary Learned F+G Rate Gain by Sketch Type\n")
    gain_table = gain.rename(columns={"count": "n", "median": "median_ratio_over_R10"})
    report.append(_md_table(gain_table))

    report.append("## Best Non-Full-Doc Learned Point by Task\n")
    report.append(
        "This table ignores official winners and asks which tree-composed learned point gets closest, then compares that point to the same method's full-doc neural result, the measured constant null, and the best positive standard sketch reference when defined.\n"
    )
    report.append(
        "`tree_over_null` and `full_doc_over_null` are the easiest cross-task scores. `composition_loss_share` is `(tree - full_doc) / (null - full_doc)`: `0` means no tree-composition penalty, `1` means the tree point has fallen back to the null, and negative means the tree point beat its full-doc counterpart. Blank standard-official cells mean this bundle has no positive same-task official empirical reference after Lean-backed and exact-zero rows are excluded.\n"
    )
    report.append(_md_table(family_summary, max_rows=80))

    report.append("## One Sketch Family at a Time\n")
    report.append(
        f"Each figure has `{present_rate_text}` columns. Top row: relative RMSE over leaf sizes, with primary learned f+g projection operators as solid lines and positive standard sketch references as dashed lines. Star markers denote the `leaf_size={full_doc_leaf}` full-doc neural point. Bottom row: the same learned operators normalized to their own full-doc neural RMSE.\n"
    )
    for task, path in task_breakouts:
        report.append(f"### {task}\n")
        report.append(f"![{task}]({path.name})\n")

    report.append("## Existing Standard Report Artifacts\n")
    report.append("- [classical_sketches_report.md](../classical_sketches_report.md)\n")
    report.append("- [classical_sketches_summary.png](../classical_sketches_summary.png)\n")
    report.append("- [learned_sketch_leaf_size_diagnostic.png](../learned_sketch_leaf_size_diagnostic.png)\n")

    report_path = out_dir / "classical_sketches_rgrid_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    return report_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-csv", type=Path, required=True)
    parser.add_argument(
        "--extra-aggregate-csv",
        type=Path,
        action="append",
        default=[],
        help="Additional aggregate CSVs used only to fill R values missing from the primary CSV.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--leaf-sizes", default=",".join(str(x) for x in DEFAULT_LEAF_ORDER))
    args = parser.parse_args(list(argv) if argv is not None else None)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    leaf_order = _parse_leaf_sizes(args.leaf_sizes)
    raw = _load_aggregate_sources(args.aggregate_csv, args.extra_aggregate_csv)
    df = _prepare(raw, leaf_order=leaf_order)
    report_path = _make_report(
        df,
        out_dir=args.out_dir,
        aggregate_csv=args.aggregate_csv,
        run_root=args.run_root,
        leaf_order=leaf_order,
    )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
