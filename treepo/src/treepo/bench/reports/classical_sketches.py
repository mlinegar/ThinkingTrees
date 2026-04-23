from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from treepo.bench.io import atomic_write_text, dump_json, write_csv_rows

CAPACITY_ORDER = {"small": 0, "medium": 1, "large": 2}


def _scan_rows(output_root: Path) -> List[dict]:
    rows: List[dict] = []
    for path in Path(output_root).rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        candidate = payload.get("rows")
        if not isinstance(candidate, list) or not candidate or not isinstance(candidate[0], dict):
            continue
        if "family" not in candidate[0] or "sketch" not in candidate[0]:
            continue
        rows.extend(dict(r) for r in candidate)
    return rows


def _aggregate(rows: Sequence[dict]) -> List[dict]:
    groups: Dict[Tuple[str, str, str, str, str, str], List[dict]] = {}
    for row in rows:
        learned_variant = (
            str(row.get("learned_variant", ""))
            if str(row.get("implementation_status", "")) == "learned_empirical"
            else ""
        )
        key = (
            str(row.get("family")),
            str(row.get("sketch")),
            str(row.get("query")),
            str(row.get("capacity_label", "single")),
            str(row.get("n_leaves", "-1")),
            learned_variant,
        )
        groups.setdefault(key, []).append(row)
    out: List[dict] = []
    for (family, sketch, query, capacity_label, n_leaves, _learned_variant), grows in sorted(groups.items()):
        def arr(name: str) -> np.ndarray:
            vals = []
            for r in grows:
                try:
                    vals.append(float(r.get(name, np.nan)))
                except Exception:
                    vals.append(float("nan"))
            return np.asarray(vals, dtype=np.float64)

        def nanmean(values: np.ndarray) -> float:
            finite = values[np.isfinite(values)]
            return float(np.mean(finite)) if len(finite) else float("nan")

        def nanstd(values: np.ndarray) -> float:
            finite = values[np.isfinite(values)]
            return float(np.std(finite)) if len(finite) else float("nan")

        def ci95(values: np.ndarray) -> float:
            finite = values[np.isfinite(values)]
            if len(finite) < 2:
                return 0.0 if len(finite) == 1 else float("nan")
            return float(1.96 * np.std(finite, ddof=1) / math.sqrt(float(len(finite))))

        rel = arr("relative_rmse")
        spread = arr("schedule_spread_mean")
        dist = arr("distance_to_official_floor")
        floor = arr("official_floor_rel_rmse")
        coverage = arr("bound_coverage_2sigma")
        theory = arr("theoretical_error")
        row = {
            "family": family,
            "sketch": sketch,
            "query": query,
            "capacity_label": capacity_label,
            "n_leaves": int(float(n_leaves)),
            "n_runs": int(len(grows)),
            "implementation_status": str(grows[0].get("implementation_status", "")),
            "formal_status": str(grows[0].get("formal_status", "")),
            "relative_rmse_mean": nanmean(rel),
            "relative_rmse_std": nanstd(rel),
            "relative_rmse_ci95": ci95(rel),
            "schedule_spread_mean": nanmean(spread),
            "schedule_spread_ci95": ci95(spread),
            "bound_coverage_2sigma_mean": nanmean(coverage),
            "theoretical_error_mean": nanmean(theory),
            "official_floor_rel_rmse_mean": nanmean(floor),
            "distance_to_official_floor_mean": nanmean(dist),
            "distance_to_official_floor_ci95": ci95(dist),
            "memory_bytes_mean": nanmean(arr("memory_bytes_mean")),
            "memory_bytes_ci95": ci95(arr("memory_bytes_mean")),
        }
        for key in ("learned_variant", "learned_codename", "learned_run_slug"):
            if key in grows[0]:
                row[key] = grows[0].get(key)
        out.append(row)
    out.sort(
        key=lambda r: (
            str(r.get("family")),
            str(r.get("sketch")),
            str(r.get("query")),
            CAPACITY_ORDER.get(str(r.get("capacity_label")), 999),
            int(r.get("n_leaves", -1)),
            str(r.get("learned_variant", "")),
        )
    )
    return out


def _capacity_x(label: object) -> int:
    return CAPACITY_ORDER.get(str(label), 999)


def _series_label(sketch: str, query: str) -> str:
    if query in {"cardinality", "top5_point_frequency", "total_weight", "accumulator_summary_sum"}:
        return sketch
    return f"{sketch}:{query}"


METHOD_GROUPS = ("official", "learned_f", "learned_g", "learned_joint", "learned_other")
METHOD_LABELS = {
    "official": "official sketch floor",
    "learned_f": r"learned $f$",
    "learned_g": r"learned $g$",
    "learned_joint": r"learned joint",
    "learned_other": r"learned (other variant)",
}
METHOD_COLORS = {
    "official": "#1f4e79",
    "learned_f": "#7b3294",
    "learned_g": "#b44b2a",
    "learned_joint": "#2a8c6f",
    "learned_other": "#666666",
}
CAPACITY_COLORS = {
    "small": "#8b8b8b",
    "medium": "#4c78a8",
    "large": "#f58518",
}


def _method_group(row: dict) -> str:
    sketch = str(row.get("sketch", ""))
    status = str(row.get("implementation_status", ""))
    variant = str(row.get("learned_variant", ""))
    if variant:
        if variant == "f":
            return "learned_f"
        if variant == "g":
            return "learned_g"
        if all(c in ("f", "g") for c in variant):
            return "learned_joint"
    if sketch.startswith("learned_joint_"):
        return "learned_joint"
    # Legacy split names now report as the single joint codename.
    if sketch.startswith("learned_fg_") or sketch.startswith("learned_gf_"):
        return "learned_joint"
    if sketch.startswith("learned_f_"):
        return "learned_f"
    if sketch.startswith("learned_g_"):
        return "learned_g"
    if sketch.startswith("learned_") and status == "learned_empirical":
        parts = sketch.split("_", 2)
        if len(parts) >= 2 and parts[1] and all(c in ("f", "g") for c in parts[1]):
            return "learned_joint"
        return "learned_other"
    if status in {"official_empirical", "lean_backed"}:
        return "official"
    return status


def _finite_float(row: dict, key: str) -> float:
    try:
        value = float(row.get(key, np.nan))
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _best_row(
    rows: Sequence[dict],
    *,
    method: str,
    family: str,
    query: str,
    capacity: str | None,
    n_leaves: int,
    metric: str,
) -> dict | None:
    candidates: List[tuple[float, dict]] = []
    for row in rows:
        if _method_group(row) != method:
            continue
        if str(row.get("family")) != family or str(row.get("query")) != query:
            continue
        if capacity is not None and str(row.get("capacity_label")) != capacity:
            continue
        if int(row.get("n_leaves", -1)) != int(n_leaves):
            continue
        value = _finite_float(row, metric)
        if np.isfinite(value):
            candidates.append((value, row))
    return min(candidates, key=lambda x: x[0])[1] if candidates else None


def _panel_axes(rows: Sequence[dict]) -> tuple[list[tuple[str, str]], list[int], list[str]]:
    panels = sorted({(str(r.get("family")), str(r.get("query"))) for r in rows})
    leaves = sorted({int(r.get("n_leaves", -1)) for r in rows if int(r.get("n_leaves", -1)) > 0})
    capacities = sorted({str(r.get("capacity_label", "")) for r in rows}, key=_capacity_x)
    return panels, leaves, capacities


def _plot_summary(rows: Sequence[dict], output: Path) -> None:
    if not rows:
        return
    families = sorted({str(r.get("family")) for r in rows})
    leaves = sorted({int(r.get("n_leaves", -1)) for r in rows if int(r.get("n_leaves", -1)) > 0})
    capacities = sorted({str(r.get("capacity_label", "")) for r in rows}, key=_capacity_x)
    preferred_capacity = "large" if "large" in capacities else capacities[-1]
    ncols = 3
    nrows = int(np.ceil(len(families) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15, max(3.2, 3.2 * nrows)),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, family in zip(axes.ravel(), families):
        queries = sorted({str(r.get("query")) for r in rows if str(r.get("family")) == family})
        for method in METHOD_GROUPS:
            xs: list[int] = []
            ys: list[float] = []
            for n_leaves in leaves:
                vals: list[float] = []
                for query in queries:
                    best = _best_row(
                        rows,
                        method=method,
                        family=family,
                        query=query,
                        capacity=preferred_capacity,
                        n_leaves=n_leaves,
                        metric="relative_rmse_mean",
                    )
                    if best is not None:
                        vals.append(_finite_float(best, "relative_rmse_mean"))
                finite = [v for v in vals if np.isfinite(v)]
                if finite:
                    xs.append(n_leaves)
                    ys.append(float(np.mean(finite)))
            if xs:
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=1.8,
                    markersize=4.0,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                )
        ax.set_title(family)
        ax.set_xlabel("leaf count L")
        ax.set_ylabel("mean best RMSE")
        ax.set_xticks(leaves)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best", frameon=False)
    for ax in axes.ravel()[len(families) :]:
        ax.axis("off")
    fig.suptitle(
        "Broad Sketch Summary: Best Large-Capacity Row Per Method",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _plot_method_group(rows: Sequence[dict], method: str, output: Path) -> None:
    panels, leaves, capacities = _panel_axes(rows)
    if not panels:
        return
    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(13, max(3.2, 2.85 * nrows)),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, (family, query) in zip(axes.ravel(), panels):
        for capacity in capacities:
            xs: list[int] = []
            ys: list[float] = []
            yerr: list[float] = []
            for n_leaves in leaves:
                best = _best_row(
                    rows,
                    method=method,
                    family=family,
                    query=query,
                    capacity=capacity,
                    n_leaves=n_leaves,
                    metric="relative_rmse_mean",
                )
                if best is None:
                    continue
                xs.append(n_leaves)
                ys.append(_finite_float(best, "relative_rmse_mean"))
                yerr.append(_finite_float(best, "relative_rmse_ci95"))
            if xs:
                clean_err = [0.0 if not np.isfinite(v) else v for v in yerr]
                ax.errorbar(
                    xs,
                    ys,
                    yerr=clean_err,
                    label=capacity,
                    color=CAPACITY_COLORS.get(capacity, "#555555"),
                    marker="o",
                    linewidth=1.6,
                    markersize=3.5,
                    capsize=2.0,
                )
        ax.set_title(f"{family}: {query}", fontsize=9)
        ax.set_xlabel("leaf count L")
        ax.set_ylabel("relative/rank RMSE")
        ax.set_xticks(leaves)
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(fontsize=7, frameon=False, loc="best")
    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")
    fig.suptitle(f"{METHOD_LABELS[method]}: Raw Error by Capacity", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=185)
    plt.close(fig)


def _plot_gold_gap(rows: Sequence[dict], output: Path) -> None:
    panels, leaves, capacities = _panel_axes(rows)
    if not panels:
        return
    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(13, max(3.2, 2.85 * nrows)),
        squeeze=False,
        constrained_layout=True,
    )
    linestyle = {"learned_f": ":", "learned_g": "--", "learned_joint": "-"}
    marker = {"learned_f": "^", "learned_g": "s", "learned_joint": "o"}
    for ax, (family, query) in zip(axes.ravel(), panels):
        for method in ("learned_f", "learned_g", "learned_joint"):
            for capacity in capacities:
                xs: list[int] = []
                ys: list[float] = []
                for n_leaves in leaves:
                    best = _best_row(
                        rows,
                        method=method,
                        family=family,
                        query=query,
                        capacity=capacity,
                        n_leaves=n_leaves,
                        metric="distance_to_official_floor_mean",
                    )
                    if best is None:
                        continue
                    value = _finite_float(best, "distance_to_official_floor_mean")
                    if not np.isfinite(value):
                        continue
                    xs.append(n_leaves)
                    ys.append(max(0.0, value))
                if xs:
                    ax.plot(
                        xs,
                        ys,
                        label=f"{METHOD_LABELS[method]}, {capacity}",
                        color=CAPACITY_COLORS.get(capacity, "#555555"),
                        linestyle=linestyle[method],
                        marker=marker[method],
                        linewidth=1.4,
                        markersize=3.2,
                    )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        ax.set_title(f"{family}: {query}", fontsize=9)
        ax.set_xlabel("leaf count L")
        ax.set_ylabel("excess RMSE over official floor")
        ax.set_xticks(leaves)
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(fontsize=6, frameon=False, loc="best")
    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")
    fig.suptitle("Learned Excess Error Over the Gold-Standard Floor", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=185)
    plt.close(fig)


def _plot_paper_summary_figures(rows: Sequence[dict], out_dir: Path) -> None:
    _plot_summary(rows, out_dir / "classical_sketches_summary.png")
    for method in METHOD_GROUPS:
        _plot_method_group(rows, method, out_dir / f"classical_sketches_method_{method}.png")
    _plot_gold_gap(rows, out_dir / "classical_sketches_gold_gap.png")


def _markdown(rows: Sequence[dict]) -> str:
    def fmt(value: object) -> str:
        try:
            v = float(value)
        except Exception:
            return "—"
        return f"{v:.4g}" if np.isfinite(v) else "—"

    lines = [
        "# Classical Mergeable Sketch Comparison",
        "",
        "| family | sketch | query | capacity | L | implementation | formal | rel/rank RMSE | official floor | distance | 2σ coverage | schedule spread | memory bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {family} | {sketch} | {query} | {capacity} | {n_leaves} | {status} | {formal} | {rel} | {floor} | {dist} | {coverage} | {spread} | {mem} |".format(
                family=r["family"],
                sketch=r["sketch"],
                query=r["query"],
                capacity=r.get("capacity_label", "single"),
                n_leaves=r.get("n_leaves", -1),
                status=r["implementation_status"],
                formal=r.get("formal_status", ""),
                rel=fmt(r.get("relative_rmse_mean", float("nan"))),
                floor=fmt(r.get("official_floor_rel_rmse_mean", float("nan"))),
                dist=fmt(r.get("distance_to_official_floor_mean", float("nan"))),
                coverage=fmt(r.get("bound_coverage_2sigma_mean", float("nan"))),
                spread=fmt(r.get("schedule_spread_mean", float("nan"))),
                mem=fmt(r.get("memory_bytes_mean", float("nan"))),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _latex_escape(value: object) -> str:
    return str(value).replace("_", "\\_")


def _latex_table(rows: Sequence[dict]) -> str:
    def fmt(value: object) -> str:
        try:
            v = float(value)
        except Exception:
            return "--"
        return f"{v:.4g}" if np.isfinite(v) else "--"

    lines = [
        "% Auto-generated by treepo.bench.reports.classical_sketches; do not edit.",
        "\\begin{tabular}{lllrlrrrrrrr}",
        "\\toprule",
        "family & sketch & query & cap. & $L$ & rel RMSE & 95\\% CI & floor & dist. & spread & bytes \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            "{family} & {sketch} & {query} & {capacity} & {n_leaves} & {rel} & {ci} & {floor} & {dist} & {spread} & {mem} \\\\".format(
                family=_latex_escape(r["family"]),
                sketch=_latex_escape(r["sketch"]),
                query=_latex_escape(r["query"]),
                capacity=_latex_escape(r.get("capacity_label", "")),
                n_leaves=r.get("n_leaves", -1),
                rel=fmt(r.get("relative_rmse_mean")),
                ci=fmt(r.get("relative_rmse_ci95")),
                floor=fmt(r.get("official_floor_rel_rmse_mean")),
                dist=fmt(r.get("distance_to_official_floor_mean")),
                spread=fmt(r.get("schedule_spread_mean")),
                mem=fmt(r.get("memory_bytes_mean")),
            )
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def _best_compact_rows(rows: Sequence[dict]) -> List[dict]:
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for row in rows:
        groups.setdefault((str(row.get("family")), str(row.get("query"))), []).append(row)
    out: List[dict] = []
    for (family, query), grows in sorted(groups.items()):
        preferred = [r for r in grows if str(r.get("capacity_label")) == "large"] or list(grows)

        def best(
            status: str,
            sketch_prefix: str | None = None,
            *,
            row_filter=None,
        ) -> dict | None:
            candidates = [r for r in preferred if str(r.get("implementation_status")) == status]
            if sketch_prefix is not None:
                candidates = [r for r in candidates if str(r.get("sketch", "")).startswith(sketch_prefix)]
            if row_filter is not None:
                candidates = [r for r in candidates if row_filter(r)]
            finite = []
            for r in candidates:
                try:
                    value = float(r.get("relative_rmse_mean", np.nan))
                except Exception:
                    continue
                if np.isfinite(value):
                    finite.append((value, r))
            return min(finite, key=lambda x: x[0])[1] if finite else None

        official = best("official_empirical")
        # Use _method_group so legacy split names and current joint names land
        # in the same report bucket while exact variants remain in metadata.
        def best_in_group(group: str):
            return best(
                "learned_empirical",
                None,
                row_filter=lambda r: _method_group(r) == group,
            )

        learned_f = best_in_group("learned_f")
        learned_g = best_in_group("learned_g")
        # Joint = any multi-letter variant. The aggregate already keys by
        # learned_variant, so fg, gf, fgf, etc. live as distinct rows; this
        # picks the best one for the joint column. The chosen row's
        # `learned_variant` field is reported as `joint_variant` so the table
        # records *which* schedule won.
        learned_joint = best_in_group("learned_joint")
        learned_any = best("learned_empirical")
        if all(x is None for x in (official, learned_f, learned_g, learned_joint)):
            continue
        out.append(
            {
                "family": family,
                "query": query,
                "official_sketch": official.get("sketch", "--") if official else "--",
                "official_rel_rmse": official.get("relative_rmse_mean", np.nan) if official else np.nan,
                "official_L": official.get("n_leaves", "--") if official else "--",
                "learned_sketch": learned_any.get("sketch", "--") if learned_any else "--",
                "learned_rel_rmse": learned_any.get("relative_rmse_mean", np.nan) if learned_any else np.nan,
                "learned_L": learned_any.get("n_leaves", "--") if learned_any else "--",
                "learned_distance": learned_any.get("distance_to_official_floor_mean", np.nan) if learned_any else np.nan,
                "learned_f_sketch": learned_f.get("sketch", "--") if learned_f else "--",
                "learned_f_rel_rmse": learned_f.get("relative_rmse_mean", np.nan) if learned_f else np.nan,
                "learned_f_L": learned_f.get("n_leaves", "--") if learned_f else "--",
                "learned_f_distance": learned_f.get("distance_to_official_floor_mean", np.nan) if learned_f else np.nan,
                "learned_g_sketch": learned_g.get("sketch", "--") if learned_g else "--",
                "learned_g_rel_rmse": learned_g.get("relative_rmse_mean", np.nan) if learned_g else np.nan,
                "learned_g_L": learned_g.get("n_leaves", "--") if learned_g else "--",
                "learned_g_distance": learned_g.get("distance_to_official_floor_mean", np.nan) if learned_g else np.nan,
                "learned_joint_sketch": learned_joint.get("sketch", "--") if learned_joint else "--",
                "learned_joint_variant": str(learned_joint.get("learned_variant", "--")) if learned_joint else "--",
                "learned_joint_rel_rmse": learned_joint.get("relative_rmse_mean", np.nan) if learned_joint else np.nan,
                "learned_joint_L": learned_joint.get("n_leaves", "--") if learned_joint else "--",
                "learned_joint_distance": learned_joint.get("distance_to_official_floor_mean", np.nan) if learned_joint else np.nan,
            }
        )
    return out


def _compact_markdown(rows: Sequence[dict]) -> str:
    def fmt(value: object) -> str:
        try:
            v = float(value)
        except Exception:
            return "—"
        return f"{v:.4g}" if np.isfinite(v) else "—"

    lines = [
        "# Classical Sketch Compact Learned Overlay",
        "",
        "| family | query | best official | official RMSE | learned f | f RMSE | learned g | g RMSE | learned joint (best) | joint variant | joint RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _best_compact_rows(rows):
        lines.append(
            "| {family} | {query} | {official} (L={official_l}) | {official_rmse} | "
            "{learned_f} (L={learned_f_l}) | {learned_f_rmse} | "
            "{learned_g} (L={learned_g_l}) | {learned_g_rmse} | "
            "{learned_joint} (L={learned_joint_l}) | {joint_variant} | {learned_joint_rmse} |".format(
                family=row["family"],
                query=row["query"],
                official=row["official_sketch"],
                official_l=row["official_L"],
                official_rmse=fmt(row["official_rel_rmse"]),
                learned_f=row["learned_f_sketch"],
                learned_f_l=row["learned_f_L"],
                learned_f_rmse=fmt(row["learned_f_rel_rmse"]),
                learned_g=row["learned_g_sketch"],
                learned_g_l=row["learned_g_L"],
                learned_g_rmse=fmt(row["learned_g_rel_rmse"]),
                learned_joint=row["learned_joint_sketch"],
                learned_joint_l=row["learned_joint_L"],
                joint_variant=row["learned_joint_variant"],
                learned_joint_rmse=fmt(row["learned_joint_rel_rmse"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _compact_latex(rows: Sequence[dict]) -> str:
    def fmt(value: object) -> str:
        try:
            v = float(value)
        except Exception:
            return "--"
        return f"{v:.4g}" if np.isfinite(v) else "--"

    lines = [
        "% Auto-generated by treepo.bench.reports.classical_sketches; do not edit.",
        "\\begin{tabular}{lllrrrrrrrr}",
        "\\toprule",
        "family & query & best official & official & "
        "learned $f$ & $f$ RMSE & learned $g$ & $g$ RMSE & "
        "learned joint (best) & joint variant & joint RMSE \\\\",
        "\\midrule",
    ]
    for row in _best_compact_rows(rows):
        lines.append(
            "{family} & {query} & {official} & {official_rmse} & "
            "{learned_f} & {learned_f_rmse} & {learned_g} & {learned_g_rmse} & "
            "{learned_joint} & {joint_variant} & {learned_joint_rmse} \\\\".format(
                family=_latex_escape(row["family"]),
                query=_latex_escape(row["query"]),
                official=_latex_escape(row["official_sketch"]),
                official_rmse=fmt(row["official_rel_rmse"]),
                learned_f=_latex_escape(row["learned_f_sketch"]),
                learned_f_rmse=fmt(row["learned_f_rel_rmse"]),
                learned_g=_latex_escape(row["learned_g_sketch"]),
                learned_g_rmse=fmt(row["learned_g_rel_rmse"]),
                learned_joint=_latex_escape(row["learned_joint_sketch"]),
                joint_variant=_latex_escape(row["learned_joint_variant"]),
                learned_joint_rmse=fmt(row["learned_joint_rel_rmse"]),
            )
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def _plot_family(rows: Sequence[dict], family: str, output: Path) -> None:
    subset = [r for r in rows if str(r.get("family")) == family]
    if not subset:
        return
    leaf_counts = sorted({int(r.get("n_leaves", -1)) for r in subset})
    capacities = sorted({str(r.get("capacity_label", "single")) for r in subset}, key=_capacity_x)
    cap_x = np.arange(len(capacities), dtype=np.float64)
    cap_pos = {cap: idx for idx, cap in enumerate(capacities)}
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for row in subset:
        groups.setdefault((str(row.get("sketch")), str(row.get("query"))), []).append(row)

    fig, axes = plt.subplots(
        1,
        max(1, len(leaf_counts)),
        figsize=(4.2 * max(1, len(leaf_counts)), 3.4),
        sharey=True,
        squeeze=False,
    )
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf", "#bcbd22"]
    for panel_idx, n_leaves in enumerate(leaf_counts):
        ax = axes[0, panel_idx]
        for idx, ((sketch, query), grows) in enumerate(sorted(groups.items())):
            by_cap = {str(r.get("capacity_label", "single")): r for r in grows if int(r.get("n_leaves", -1)) == n_leaves}
            xs: List[float] = []
            ys: List[float] = []
            yerr: List[float] = []
            for cap in capacities:
                row = by_cap.get(cap)
                if row is None:
                    continue
                xs.append(float(cap_pos[cap]))
                ys.append(float(row.get("relative_rmse_mean", np.nan)))
                yerr.append(float(row.get("relative_rmse_ci95", 0.0)))
            if xs:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=yerr,
                    label=_series_label(sketch, query),
                    color=palette[idx % len(palette)],
                    linestyle="-" if "official" in str(grows[0].get("implementation_status")) else "--",
                    marker="o",
                    linewidth=1.2,
                    markersize=3.0,
                    capsize=2.0,
                )
        floors = []
        for cap in capacities:
            vals = [
                float(r.get("official_floor_rel_rmse_mean", np.nan))
                for r in subset
                if int(r.get("n_leaves", -1)) == n_leaves and str(r.get("capacity_label", "single")) == cap
            ]
            vals = [v for v in vals if np.isfinite(v)]
            floors.append(min(vals) if vals else np.nan)
        if any(np.isfinite(floors)):
            ax.plot(cap_x, floors, linestyle=":", color="black", linewidth=1.0, label="official floor")
        ax.set_xticks(cap_x)
        ax.set_xticklabels(capacities, rotation=20, ha="right")
        ax.set_xlabel("capacity preset")
        if panel_idx == 0:
            ax.set_ylabel("relative RMSE / rank RMSE")
        ax.set_title(f"L = {n_leaves}")
        ax.grid(True, alpha=0.3)
        if panel_idx == len(leaf_counts) - 1:
            ax.legend(fontsize=6, loc="best", frameon=False)
    fig.suptitle(f"Classical sketch grid: {family}", fontsize=11)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_family_figures(rows: Sequence[dict], out_dir: Path) -> None:
    for family in sorted({str(r.get("family")) for r in rows}):
        _plot_family(rows, family, out_dir / f"classical_sketches_{family}.png")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate the classical-sketch comparison report.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--tables-dir", type=Path, default=Path("paper/ctreepo/tables"))
    p.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.output_root) / "reports" / "classical_sketches"
    rows = _scan_rows(Path(args.output_root))
    agg = _aggregate(rows)
    write_csv_rows(out_dir / "classical_sketches_aggregate.csv", agg)
    atomic_write_text(out_dir / "classical_sketches_aggregate.json", dump_json({"rows": agg}))
    atomic_write_text(out_dir / "classical_sketches_report.md", _markdown(agg))
    atomic_write_text(out_dir / "classical_sketches_grid.md", _markdown(agg))
    atomic_write_text(out_dir / "classical_sketches_grid.tex", _latex_table(agg))
    atomic_write_text(out_dir / "classical_sketches_compact.md", _compact_markdown(agg))
    atomic_write_text(out_dir / "classical_sketches_compact.tex", _compact_latex(agg))
    if args.tables_dir is not None:
        tables_dir = Path(args.tables_dir)
        atomic_write_text(tables_dir / "classical_sketches_grid.md", _markdown(agg))
        atomic_write_text(tables_dir / "classical_sketches_grid.tex", _latex_table(agg))
        atomic_write_text(tables_dir / "classical_sketches_compact.md", _compact_markdown(agg))
        atomic_write_text(tables_dir / "classical_sketches_compact.tex", _compact_latex(agg))
    _plot_paper_summary_figures(agg, out_dir)
    _plot_family_figures(agg, out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
