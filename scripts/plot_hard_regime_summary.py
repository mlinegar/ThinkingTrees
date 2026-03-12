#!/usr/bin/env python3
"""Summarize hard-regime simulation sweeps across Markov / Segment-LDA / C-TreePO."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import re
import statistics
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


REGIME_RE = re.compile(r"/regime_([^/]+)/")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot hard-regime summary curves.")
    p.add_argument("--markov-glob", type=str, required=True)
    p.add_argument("--segment-glob", type=str, required=True)
    p.add_argument("--ctree-glob", type=str, required=True)
    p.add_argument("--aggregate", choices=["median", "mean"], default="median")
    p.add_argument("--output-figure", type=str, default="outputs/hard_regime_summary.png")
    p.add_argument("--output-json", type=str, default="outputs/hard_regime_summary_report.json")
    return p.parse_args()


def _reduce(vals: List[float], agg: str) -> float:
    clean = [float(x) for x in vals if np.isfinite(float(x))]
    if not clean:
        return float("nan")
    if agg == "mean":
        return float(np.mean(np.asarray(clean, dtype=np.float64)))
    return float(statistics.median(clean))


def _regime_from_path(path: str) -> str:
    m = REGIME_RE.search(path.replace("\\", "/"))
    if m is None:
        return "unknown"
    return str(m.group(1))


def _collect_markov(glob_pat: str) -> Dict[str, Dict[int, List[float]]]:
    out: Dict[str, Dict[int, List[float]]] = {}
    for fp in glob.glob(glob_pat, recursive=True):
        payload = json.loads(Path(fp).read_text(encoding="utf-8"))
        cfg = payload.get("config", {}) or {}
        if abs(float(cfg.get("audit_fraction", float("nan"))) - 1.0) > 1e-12:
            continue
        td = int(cfg.get("train_docs", -1))
        y = float(((payload.get("metrics", {}) or {}).get("learned", {}) or {}).get("root_mae", float("nan")))
        reg = _regime_from_path(fp)
        out.setdefault(reg, {}).setdefault(td, []).append(y)
    return out


def _collect_segment(glob_pat: str) -> Dict[str, Dict[int, List[float]]]:
    out: Dict[str, Dict[int, List[float]]] = {}
    for fp in glob.glob(glob_pat, recursive=True):
        payload = json.loads(Path(fp).read_text(encoding="utf-8"))
        cfg = payload.get("config", {}) or {}
        if abs(float(cfg.get("audit_fraction", float("nan"))) - 1.0) > 1e-12:
            continue
        td = int(cfg.get("train_docs", -1))
        y = float(((payload.get("metrics", {}) or {}).get("ridge", {}) or {}).get("root_mae", float("nan")))
        reg = _regime_from_path(fp)
        out.setdefault(reg, {}).setdefault(td, []).append(y)
    return out


def _collect_ctree(glob_pat: str) -> Dict[str, Dict[int, List[float]]]:
    out: Dict[str, Dict[int, List[float]]] = {}
    for fp in glob.glob(glob_pat, recursive=True):
        payload = json.loads(Path(fp).read_text(encoding="utf-8"))
        cfg = payload.get("config", {}) or {}
        leaf = float(cfg.get("eval_leaf_query_rate", float("nan")))
        internal = float(cfg.get("eval_internal_query_rate", float("nan")))
        if abs(leaf - 1.0) > 1e-12 or abs(internal - 1.0) > 1e-12:
            continue
        td = int(cfg.get("n_books_train", -1))
        y = float(
            ((payload.get("metrics", {}) or {}).get("estimated_calibrated_budgeted", {}) or {}).get(
                "root_l1_mean", float("nan")
            )
        )
        reg = _regime_from_path(fp)
        out.setdefault(reg, {}).setdefault(td, []).append(y)
    return out


def _plot_panel(ax: plt.Axes, data: Dict[str, Dict[int, List[float]]], agg: str, y_label: str, title: str) -> Dict[str, object]:
    payload: Dict[str, object] = {}
    if not data:
        ax.text(0.5, 0.5, "No rows", ha="center", va="center")
        ax.set_axis_off()
        return payload
    regimes = sorted(data.keys())
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(2, len(regimes))))
    for idx, reg in enumerate(regimes):
        xs = sorted(data[reg].keys())
        ys = [_reduce(data[reg][x], agg) for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.9, color=colors[idx], label=reg)
        payload[reg] = {
            "x": [int(x) for x in xs],
            "y": [float(y) for y in ys],
            "n_rows": int(sum(len(data[reg][x]) for x in xs)),
        }
    ax.set_xlabel("train_docs")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    return payload


def main() -> int:
    args = _parse_args()
    agg = str(args.aggregate)
    markov = _collect_markov(str(args.markov_glob))
    segment = _collect_segment(str(args.segment_glob))
    ctree = _collect_ctree(str(args.ctree_glob))

    fig, axes = plt.subplots(1, 3, figsize=(16.3, 5.2), constrained_layout=True)
    p0 = _plot_panel(axes[0], markov, agg, "learned root MAE", "Hard Regimes: Markov @ audit=1")
    p1 = _plot_panel(axes[1], segment, agg, "ridge root MAE", "Hard Regimes: Segment-LDA @ audit=1")
    p2 = _plot_panel(axes[2], ctree, agg, "budgeted root L1", "Hard Regimes: C-TreePO @ leaf=1,int=1")
    fig.suptitle("Hard-Regime Stress Summary", fontsize=13)

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=190)
    plt.close(fig)

    report = {
        "aggregate": agg,
        "input_globs": {
            "markov": str(args.markov_glob),
            "segment": str(args.segment_glob),
            "ctree": str(args.ctree_glob),
        },
        "markov": p0,
        "segment": p1,
        "ctree": p2,
        "output_figure": str(out_fig),
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_figure": str(out_fig), "output_json": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

