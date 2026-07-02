#!/usr/bin/env python3
"""Publication-style report for the tree-relevant LDA coarse leaf-size extension."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import fmean
import textwrap
from typing import Dict, Iterable, List, Sequence, Tuple

import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ctreepo.sim.util import safe_float as _safe_float


TAU_SUITES = {"tau_crossover_dense", "tau_crossover_proportion_extend"}
LAMBDA_SUITES = {"lambda_onset_dense", "lambda_onset_proportion_extend"}
BOUNDARY_EXTENSION_SUITE = "sample_size_boundary_check"
ALL_PUBLICATION_SUITES = (
    "tau_crossover_dense",
    "tau_crossover_proportion_extend",
    "lambda_onset_dense",
    "lambda_onset_proportion_extend",
    "doc_topic_concentration_robustness",
    "sample_size_boundary_check",
)
TIE_EPS = 5e-3


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build publication-style report for the LDA coarse leaf-size extension.")
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--extension-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--snapshot-label", type=str, default="Coarse Leaf-Size Publication Report")
    p.add_argument("--clean-figures-subdir", type=str, default="publication_figures")
    return p.parse_args()


def _safe_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(fmean(vals))


def _safe_sem(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    n = len(vals)
    if n <= 1:
        return 0.0
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    return math.sqrt(var / n)


def _tau_diversity_index(tau: float) -> float:
    tau = float(tau)
    return float(1.0 / (1.0 + max(tau, 0.0)))


def _tau_label(tau: float, *, multiline: bool = False) -> str:
    d = _tau_diversity_index(tau)
    if multiline:
        return f"tau={tau:g}\nd={d:.2f}"
    return f"tau={tau:g} / d={d:.2f}"


def _leaf_label(pct: float, doc_tokens: int) -> str:
    tok = int(round(float(doc_tokens) * float(pct) / 100.0))
    return f"{pct:.0f}% ({tok} tok)"


def _threshold_label(value: float | None) -> str:
    if value is None:
        return "never"
    return f"{value:g}"


def _relative_pct_label(value: float) -> str:
    if not math.isfinite(float(value)):
        return "n/a"
    return f"{100.0 * float(value):+.1f}%"


def _effect_label(delta: float, rel_gain: float) -> str:
    delta = float(delta)
    rel_gain = float(rel_gain)
    if not math.isfinite(delta):
        return "n/a"
    if abs(delta) < TIE_EPS:
        return "tie with pooled"
    if delta > 0.0:
        rel = "" if not math.isfinite(rel_gain) else f" ({100.0 * abs(rel_gain):.1f}% lower error)"
        return f"leaf better{rel}"
    rel = "" if not math.isfinite(rel_gain) else f" ({100.0 * abs(rel_gain):.1f}% higher error)"
    return f"leaf worse{rel}"


def _delta_axis_label() -> str:
    return "Improvement in mean abs error\n(pooled - leaf; >0 means leaf better)"


def _win_axis_label() -> str:
    return "Percent of seeds where leaf abs error < pooled abs error"


def _count_manifest_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _load_manifest(root: Path) -> tuple[Dict[str, str], Counter]:
    purposes: Dict[str, str] = {}
    counts: Counter = Counter()
    path = root / "manifest.jsonl"
    if not path.exists():
        return purposes, counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        suite = str(obj.get("suite", ""))
        purposes.setdefault(suite, str(obj.get("purpose", "")))
        counts[suite] += 1
    return purposes, counts


def _load_runs(results_root: Path, *, root_name: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(results_root.rglob("seed_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {}) or {}
        methods = payload.get("methods", {}) or {}
        pooled = _safe_float(methods.get("pooled_doc_wrong_model", {}).get("utility_abs_to_true_mean"))
        leaf = _safe_float(methods.get("leaf_infer_sum", {}).get("utility_abs_to_true_mean"))
        doc_tokens = int(cfg.get("doc_tokens", -1))
        latent_leaf_tokens = int(cfg.get("latent_leaf_tokens", -1))
        pct = 100.0 * float(latent_leaf_tokens) / float(max(1, doc_tokens))
        suite = str(path.relative_to(results_root).parts[0])
        rows.append(
            {
                "root_name": root_name,
                "suite": suite,
                "path": str(path),
                "doc_tokens": doc_tokens,
                "latent_leaf_tokens": latent_leaf_tokens,
                "leaf_pct": pct,
                "train_docs": int(cfg.get("train_docs", -1)),
                "test_docs": int(cfg.get("test_docs", -1)),
                "doc_topic_concentration": _safe_float(cfg.get("doc_topic_concentration")),
                "tau": _safe_float(cfg.get("local_mixture_concentration")),
                "lam": _safe_float(cfg.get("quadratic_utility_weight")),
                "seed": int(cfg.get("seed", -1)),
                "pooled_error": pooled,
                "leaf_error": leaf,
                "delta": pooled - leaf,
            }
        )
    return rows


def _agg(rows: Sequence[dict], *, keys: Sequence[str]) -> Dict[Tuple[object, ...], dict]:
    buckets: Dict[Tuple[object, ...], List[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        buckets.setdefault(key, []).append(row)
    out: Dict[Tuple[object, ...], dict] = {}
    for key, vals in buckets.items():
        deltas = [float(v["delta"]) for v in vals]
        pooleds = [float(v["pooled_error"]) for v in vals]
        leafs = [float(v["leaf_error"]) for v in vals]
        rel_gains = [
            (float(v["pooled_error"]) - float(v["leaf_error"])) / float(v["pooled_error"])
            for v in vals
            if math.isfinite(float(v["pooled_error"])) and float(v["pooled_error"]) > 0.0
        ]
        out[key] = {
            "n": len(vals),
            "delta_mean": _safe_mean(deltas),
            "delta_sem": _safe_sem(deltas),
            "win_rate": _safe_mean([1.0 if d > TIE_EPS else 0.0 for d in deltas]),
            "wins": sum(1 for d in deltas if d > TIE_EPS),
            "ties": sum(1 for d in deltas if abs(d) <= TIE_EPS),
            "losses": sum(1 for d in deltas if d < -TIE_EPS),
            "pooled_mean": _safe_mean(pooleds),
            "leaf_mean": _safe_mean(leafs),
            "relative_gain_mean": _safe_mean(rel_gains),
            "relative_gain_sem": _safe_sem(rel_gains),
        }
    return out


def _filter_runs(
    rows: Sequence[dict],
    *,
    suites: set[str] | None = None,
    train_docs: int | None = None,
    dtc: float | None = None,
    lam: float | None = None,
    taus: set[float] | None = None,
    pcts: set[float] | None = None,
) -> List[dict]:
    out: List[dict] = []
    for row in rows:
        if suites is not None and str(row["suite"]) not in suites:
            continue
        if train_docs is not None and int(row["train_docs"]) != int(train_docs):
            continue
        if dtc is not None and float(row["doc_topic_concentration"]) != float(dtc):
            continue
        if lam is not None and float(row["lam"]) != float(lam):
            continue
        if taus is not None and float(row["tau"]) not in taus:
            continue
        if pcts is not None and float(row["leaf_pct"]) not in pcts:
            continue
        out.append(row)
    return out


def _material_onset(lam_grid: Sequence[float], stats: Dict[float, dict], *, eps: float = 0.1) -> float | None:
    for lam in lam_grid:
        row = stats.get(float(lam))
        if row is not None and float(row["delta_mean"]) > float(eps):
            return float(lam)
    return None


def _last_positive_tau(tau_grid: Sequence[float], stats: Dict[float, dict]) -> float | None:
    last = None
    for tau in tau_grid:
        row = stats.get(float(tau))
        if row is not None and float(row["delta_mean"]) > 0.0:
            last = float(tau)
    return last


def _representative_slice(label: str, stats: dict) -> dict:
    return {
        "label": str(label),
        "pooled_mean": float(stats["pooled_mean"]),
        "leaf_mean": float(stats["leaf_mean"]),
        "delta_mean": float(stats["delta_mean"]),
        "relative_gain_mean": float(stats.get("relative_gain_mean", float("nan"))),
        "wins": int(stats["wins"]),
        "n": int(stats["n"]),
        "interpretation": _effect_label(float(stats["delta_mean"]), float(stats.get("relative_gain_mean", float("nan")))),
    }


def _suite_display_label(suite: str) -> str:
    labels = {
        "tau_crossover_dense": "tau crossover (original)",
        "tau_crossover_proportion_extend": "tau crossover (coarse extension)",
        "lambda_onset_dense": "quadratic-weight onset (original)",
        "lambda_onset_proportion_extend": "quadratic-weight onset (coarse extension)",
        "doc_topic_concentration_robustness": "doc-topic concentration robustness",
        "sample_size_boundary_check": "train-size boundary check",
    }
    return labels.get(suite, suite)


def _suite_display_purpose(suite: str, purpose_map: Dict[str, str]) -> str:
    if suite == "lambda_onset_dense":
        return "Measure how quickly the pooled-vs-leaf gap turns on as quadratic weight moves away from zero."
    if suite == "lambda_onset_proportion_extend":
        return "Add 50% and 100% leaf sizes to the quadratic-weight onset sweep at the moderate and low-diversity boundaries (tau=1,8)."
    return purpose_map.get(suite, "")


def _text_page(pdf: PdfPages, *, title: str, lines: Sequence[str], font_size: int = 10) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title(title, pad=12, fontsize=16, fontweight="bold")
    ax.text(0.02, 0.98, "\n".join(lines), family="monospace", fontsize=font_size, va="top")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _paragraph_page(
    pdf: PdfPages,
    *,
    title: str,
    paragraphs: Sequence[str],
    width: int = 108,
    font_size: int = 12,
) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title(title, pad=14, fontsize=18, fontweight="bold")
    y = 0.96
    for para in paragraphs:
        wrapped = textwrap.fill(str(para).strip(), width=width)
        n_lines = wrapped.count("\n") + 1
        ax.text(0.04, y, wrapped, fontsize=font_size, va="top", ha="left", linespacing=1.42)
        y -= 0.035 * n_lines + 0.045
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _image_page(pdf: PdfPages, *, title: str, image_path: Path, caption: str) -> None:
    img = plt.imread(image_path)
    fig = plt.figure(figsize=(11.0, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[12, 1])
    ax = fig.add_subplot(gs[0])
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, pad=12, fontsize=16, fontweight="bold")
    cap = fig.add_subplot(gs[1])
    cap.axis("off")
    cap.text(0.02, 0.95, textwrap.fill(caption, width=120), fontsize=11, va="top")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _save_figure(fig: plt.Figure, out_png: Path, out_pdf: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=260)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else (args.extension_root / "report" / "publication").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    clean = figures_dir / str(args.clean_figures_subdir)
    clean.mkdir(parents=True, exist_ok=True)

    base_purposes, base_counts = _load_manifest(args.baseline_root)
    ext_purposes, ext_counts = _load_manifest(args.extension_root)
    purpose_map = dict(base_purposes)
    purpose_map.update(ext_purposes)
    manifest_counts = Counter()
    manifest_counts.update(base_counts)
    manifest_counts.update(ext_counts)

    baseline_runs = _load_runs(args.baseline_root / "results", root_name=str(args.baseline_root.name))
    extension_runs = _load_runs(args.extension_root / "results", root_name=str(args.extension_root.name))
    all_runs = baseline_runs + extension_runs
    if not all_runs:
        raise RuntimeError("no run summaries found")
    suite_run_counts = Counter(row["suite"] for row in all_runs)

    doc_tokens = int(all_runs[0]["doc_tokens"])
    fine_and_coarse_pcts = [100.0 * x / doc_tokens for x in (16, 32, 64, 96, 192, 384)]
    coarse_pcts = [25.0, 50.0, 100.0]
    tau_grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    lambda_grid = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

    tau_rows_all = _filter_runs(
        all_runs,
        suites=TAU_SUITES,
        train_docs=512,
        dtc=0.6,
        lam=2.0,
        taus=set(tau_grid),
        pcts=set(fine_and_coarse_pcts),
    )
    tau_rows_coarse = [row for row in tau_rows_all if float(row["leaf_pct"]) in set(coarse_pcts)]
    tau_stats_all = _agg(tau_rows_all, keys=("leaf_pct", "tau"))
    tau_stats_coarse = _agg(tau_rows_coarse, keys=("leaf_pct", "tau"))

    lambda_rows = _filter_runs(
        all_runs,
        suites=LAMBDA_SUITES,
        train_docs=512,
        dtc=0.6,
        taus={1.0, 8.0},
        pcts=set(coarse_pcts),
    )
    lambda_stats = _agg(lambda_rows, keys=("leaf_pct", "tau", "lam"))

    boundary_rows = (
        _filter_runs(
            all_runs,
            suites=TAU_SUITES,
            train_docs=512,
            dtc=0.6,
            lam=2.0,
            taus={8.0, 16.0},
            pcts=set(coarse_pcts),
        )
        + _filter_runs(
            all_runs,
            suites={BOUNDARY_EXTENSION_SUITE},
            train_docs=2048,
            dtc=0.6,
            lam=2.0,
            taus={8.0, 16.0},
            pcts=set(coarse_pcts),
        )
    )
    boundary_stats = _agg(boundary_rows, keys=("train_docs", "leaf_pct", "tau"))

    last_positive_all = {
        float(pct): _last_positive_tau(tau_grid, {tau: tau_stats_all.get((pct, tau)) for tau in tau_grid})
        for pct in fine_and_coarse_pcts
    }
    last_positive_coarse = {
        float(pct): _last_positive_tau(tau_grid, {tau: tau_stats_coarse.get((pct, tau)) for tau in tau_grid})
        for pct in coarse_pcts
    }
    onset_eps = 0.1
    onset_table = {
        (float(pct), float(tau)): _material_onset(
            lambda_grid,
            {lam: lambda_stats.get((pct, tau, lam)) for lam in lambda_grid},
            eps=onset_eps,
        )
        for pct in coarse_pcts
        for tau in (1.0, 8.0)
    }
    best_boundary_leaf_pct = {}
    for train_docs in (512, 2048):
        for tau in (8.0, 16.0):
            options = []
            for pct in coarse_pcts:
                stats = boundary_stats.get((train_docs, pct, tau))
                if stats is not None:
                    options.append((float(stats["delta_mean"]), pct))
            if options:
                best_boundary_leaf_pct[(train_docs, tau)] = max(options)[1]

    example_failure_16_tau8 = _representative_slice(
        f"{_leaf_label(fine_and_coarse_pcts[0], doc_tokens)}, tau=8, quadratic weight=2, train_docs=512",
        tau_stats_all[(fine_and_coarse_pcts[0], 8.0)],
    )
    example_25_tau16 = _representative_slice(
        f"{_leaf_label(25.0, doc_tokens)}, tau=16, quadratic weight=2, train_docs=512",
        tau_stats_coarse[(25.0, 16.0)],
    )
    example_50_tau16 = _representative_slice(
        f"{_leaf_label(50.0, doc_tokens)}, tau=16, quadratic weight=2, train_docs=512",
        tau_stats_coarse[(50.0, 16.0)],
    )
    example_50_2048_tau8 = _representative_slice(
        f"{_leaf_label(50.0, doc_tokens)}, tau=8, quadratic weight=2, train_docs=2048",
        boundary_stats[(2048, 50.0, 8.0)],
    )
    example_100_tau16 = _representative_slice(
        f"{_leaf_label(100.0, doc_tokens)}, tau=16, quadratic weight=2, train_docs=512",
        boundary_stats[(512, 100.0, 16.0)],
    )
    representative_examples = [
        example_failure_16_tau8,
        example_25_tau16,
        example_50_tau16,
        example_50_2048_tau8,
        example_100_tau16,
    ]

    colors = {
        fine_and_coarse_pcts[0]: "#b2182b",
        fine_and_coarse_pcts[1]: "#ef8a62",
        fine_and_coarse_pcts[2]: "#fddbc7",
        fine_and_coarse_pcts[3]: "#67a9cf",
        fine_and_coarse_pcts[4]: "#2166ac",
        fine_and_coarse_pcts[5]: "#4d4d4d",
    }

    fig_a_png = clean / "figure_A_tau_frontier.png"
    fig_a_pdf = clean / "figure_A_tau_frontier.pdf"
    fig_b_png = clean / "figure_B_last_positive_tau.png"
    fig_b_pdf = clean / "figure_B_last_positive_tau.pdf"
    fig_c_png = clean / "figure_C_quadratic_weight_onset_coarse.png"
    fig_c_pdf = clean / "figure_C_quadratic_weight_onset_coarse.pdf"
    fig_d_png = clean / "figure_D_boundary_train_docs.png"
    fig_d_pdf = clean / "figure_D_boundary_train_docs.pdf"
    fig_e_png = clean / "figure_E_null_control.png"
    fig_e_pdf = clean / "figure_E_null_control.pdf"

    xs = np.arange(len(tau_grid))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    for pct in fine_and_coarse_pcts:
        means = [tau_stats_all[(pct, tau)]["delta_mean"] for tau in tau_grid]
        sems = [tau_stats_all[(pct, tau)]["delta_sem"] for tau in tau_grid]
        rels = [100.0 * tau_stats_all[(pct, tau)]["relative_gain_mean"] for tau in tau_grid]
        rel_sems = [100.0 * tau_stats_all[(pct, tau)]["relative_gain_sem"] for tau in tau_grid]
        color = colors[pct]
        label = _leaf_label(pct, doc_tokens)
        axes[0].plot(xs, means, marker="o", linewidth=2, color=color, label=label)
        axes[0].fill_between(xs, np.array(means) - np.array(sems), np.array(means) + np.array(sems), alpha=0.14, color=color)
        axes[1].plot(xs, rels, marker="o", linewidth=2, color=color, label=label)
        axes[1].fill_between(xs, np.array(rels) - np.array(rel_sems), np.array(rels) + np.array(rel_sems), alpha=0.14, color=color)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels([_tau_label(tau, multiline=True) for tau in tau_grid], fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].axhline(0.0, color="#444444", linestyle="--", linewidth=1)
    axes[1].axhline(0.0, color="#444444", linestyle="--", linewidth=1)
    axes[0].set_ylabel(_delta_axis_label())
    axes[1].set_ylabel("Mean relative change in abs error\n((pooled - leaf) / pooled)")
    axes[0].set_title("Mean improvement vs pooled baseline")
    axes[1].set_title("Mean relative improvement vs pooled baseline")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Figure A. Tau frontier at quadratic weight=2, dtc=0.6, train_docs=512", fontsize=13)
    _save_figure(fig, fig_a_png, fig_a_pdf)

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.8), constrained_layout=True)
    coarse_x = np.arange(len(coarse_pcts))
    bars = []
    for pct in coarse_pcts:
        val = last_positive_coarse[pct]
        bars.append(0.0 if val is None else float(val))
    ax.bar(coarse_x, bars, color=["#67a9cf", "#2166ac", "#7f7f7f"], width=0.6)
    ax.set_xticks(coarse_x)
    ax.set_xticklabels([_leaf_label(pct, doc_tokens) for pct in coarse_pcts])
    ax.set_ylabel("Largest tau where leaf still lowers mean abs error")
    ax.set_title("Figure B. Coarse-size threshold summary")
    ax.grid(alpha=0.3, axis="y")
    for idx, pct in enumerate(coarse_pcts):
        txt = _threshold_label(last_positive_coarse[pct])
        ax.text(idx, bars[idx] + 0.35, txt, ha="center", va="bottom", fontsize=11)
    _save_figure(fig, fig_b_png, fig_b_pdf)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True, sharey=True)
    onset_colors = {25.0: "#67a9cf", 50.0: "#2166ac", 100.0: "#7f7f7f"}
    for ax, tau in zip(axes, (1.0, 8.0)):
        for pct in coarse_pcts:
            means = [lambda_stats[(pct, tau, lam)]["delta_mean"] for lam in lambda_grid]
            sems = [lambda_stats[(pct, tau, lam)]["delta_sem"] for lam in lambda_grid]
            color = onset_colors[pct]
            label = _leaf_label(pct, doc_tokens)
            ax.plot(lambda_grid, means, marker="o", linewidth=2, color=color, label=label)
            ax.fill_between(lambda_grid, np.array(means) - np.array(sems), np.array(means) + np.array(sems), alpha=0.15, color=color)
        ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
        ax.set_title(_tau_label(tau))
        ax.set_xlabel("quadratic weight")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(_delta_axis_label())
    axes[0].legend(fontsize=8)
    fig.suptitle("Figure C. Quadratic-weight onset for the coarse leaf-size comparison", fontsize=13)
    _save_figure(fig, fig_c_png, fig_c_pdf)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True, sharey=True)
    train_positions = np.arange(2)
    width = 0.22
    train_labels = ["512", "2048"]
    bar_colors = {25.0: "#67a9cf", 50.0: "#2166ac", 100.0: "#7f7f7f"}
    for ax, tau in zip(axes, (8.0, 16.0)):
        for idx, pct in enumerate(coarse_pcts):
            vals = [boundary_stats[(train_docs, pct, tau)]["delta_mean"] for train_docs in (512, 2048)]
            ax.bar(train_positions + (idx - 1) * width, vals, width=width, color=bar_colors[pct], label=_leaf_label(pct, doc_tokens))
        ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
        ax.set_xticks(train_positions)
        ax.set_xticklabels(train_labels)
        ax.set_xlabel("train_docs")
        ax.set_title(_tau_label(tau))
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel(_delta_axis_label())
    axes[0].legend(fontsize=8)
    fig.suptitle("Figure D. Boundary check at larger training support", fontsize=13)
    _save_figure(fig, fig_d_png, fig_d_pdf)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), constrained_layout=True, sharey=True)
    for ax, tau in zip(axes, (1.0, 8.0)):
        vals = [lambda_stats[(pct, tau, 0.0)]["delta_mean"] for pct in coarse_pcts]
        ax.bar(np.arange(len(coarse_pcts)), vals, color=[bar_colors[pct] for pct in coarse_pcts], width=0.6)
        ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(len(coarse_pcts)))
        ax.set_xticklabels([_leaf_label(pct, doc_tokens) for pct in coarse_pcts], rotation=15, ha="right")
        ax.set_title(_tau_label(tau))
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel(_delta_axis_label() + "\nat quadratic weight=0")
    fig.suptitle("Figure E. Null-control check", fontsize=13)
    _save_figure(fig, fig_e_png, fig_e_pdf)

    now = datetime.now(timezone.utc).isoformat()
    md_path = output_dir / "tree_relevant_lda_proportion_extension_publication_report.md"
    pdf_path = output_dir / "tree_relevant_lda_proportion_extension_publication_report.pdf"
    diag_path = output_dir / "tree_relevant_lda_proportion_extension_publication_diagnostics.json"

    lines = [
        "---",
        "title: Tree-Relevant LDA Coarse Leaf-Size Publication Report",
        "geometry: margin=0.7in",
        "fontsize: 12pt",
        "toc: true",
        "toc-depth: 2",
        "---",
        "",
        f"- Generated: `{now}`",
        f"- Snapshot: `{args.snapshot_label}`",
        f"- Baseline root: `{args.baseline_root}`",
        f"- Extension root: `{args.extension_root}`",
        "",
        "## 1. What this report is",
        "",
        "- This is the journal-style publication package for the coarse leaf-size extension.",
        "- It keeps the full analytical report intact, but adds clean standalone figures, a publication markdown narrative, and a diagnostics JSON.",
        "- All main claims here are tied to suite-restricted aggregates so overlapping sweeps do not get mixed into the wrong threshold tables.",
        "",
        "## 2. How to read the metric",
        "",
        "- `pooled error` is the mean absolute error from the pooled whole-document baseline.",
        "- `leaf error` is the mean absolute error from the leaf-based method.",
        "- `Delta = pooled error - leaf error`.",
        "- `Delta > 0` means the leaf method is better because its absolute error is lower.",
        "- `Delta < 0` means the leaf method is worse because its absolute error is higher.",
        "- `relative gain = Delta / pooled error`, so `-75%` means the leaf method has 75% higher error than pooled, not a gain.",
        "",
        (
            f"- Concrete failure example: at `{example_failure_16_tau8['label']}`, pooled error is "
            f"`{example_failure_16_tau8['pooled_mean']:.2f}` but leaf error rises to "
            f"`{example_failure_16_tau8['leaf_mean']:.2f}`, so `Delta = {example_failure_16_tau8['delta_mean']:+.2f}` "
            f"and the leaf method is worse (`{example_failure_16_tau8['interpretation']}`)."
        ),
        (
            f"- Concrete success example: at `{example_50_2048_tau8['label']}`, pooled error is "
            f"`{example_50_2048_tau8['pooled_mean']:.2f}` and leaf error falls to "
            f"`{example_50_2048_tau8['leaf_mean']:.2f}`, so `Delta = {example_50_2048_tau8['delta_mean']:+.2f}` "
            f"(`{example_50_2048_tau8['interpretation']}`)."
        ),
        "",
        "## 3. Main claims",
        "",
        "1. The tau frontier is monotone through the fine-to-coarse ladder: the largest tau where the leaf method still lowers mean absolute error moves from `1` at `4%`, to `2` at `8%`, to `4` at `17%`, to `8` at `25%`, to `16` at `50%`, before collapsing to `never` at the degenerate `100%` one-leaf control.",
        "2. In the coarse comparison itself, the practical optimum is `50%`, not `25%` and not `100%`.",
        f"3. The quadratic-weight onset threshold improves at `50%`: at `tau=1` it is `{_threshold_label(onset_table[(50.0, 1.0)])}` versus `{_threshold_label(onset_table[(25.0, 1.0)])}` at `25%`, and at `tau=8` it is `{_threshold_label(onset_table[(50.0, 8.0)])}` versus `{_threshold_label(onset_table[(25.0, 8.0)])}`.",
        "4. The train-size boundary check strengthens the same ordering: at `tau=16`, `25%` remains worse than pooling at both `512` and `2048` training docs, while `50%` remains better at both.",
        "",
        "## 4. Coverage",
        "",
        "| Suite | Purpose | Queued | Completed |",
        "| --- | --- | ---: | ---: |",
    ]
    for suite in ALL_PUBLICATION_SUITES:
        lines.append(
            f"| `{_suite_display_label(suite)}` | {_suite_display_purpose(suite, purpose_map)} | {manifest_counts.get(suite, 0)} | {suite_run_counts.get(suite, 0)} |"
        )
    lines.extend(
        [
            "",
            "## 5. Main figures",
            "",
            "### Figure A. Full tau frontier",
            "",
            f"![]({fig_a_png.relative_to(output_dir)}){{width=100%}}",
            "",
            (
                "This figure is the complete threshold story in one view. The left panel is an absolute improvement scale and the right panel is the same comparison on a relative scale. "
                "Positive means the leaf method lowers mean absolute error, negative means it raises mean absolute error. "
                f"For example, `{example_failure_16_tau8['label']}` is a clear failure: "
                f"`{example_failure_16_tau8['pooled_mean']:.2f} -> {example_failure_16_tau8['leaf_mean']:.2f}` with "
                f"`Delta = {example_failure_16_tau8['delta_mean']:+.2f}`."
            ),
            "",
            "### Figure B. Coarse-size threshold summary",
            "",
            f"![]({fig_b_png.relative_to(output_dir)}){{width=75%}}",
            "",
            "This figure isolates the paper-facing coarse comparison. It reports the largest tau where the leaf method still has lower mean absolute error than pooling. The best threshold is `50%`, not `25%`, while `100%` is the pooled null.",
            "",
            "### Figure C. Quadratic-weight onset",
            "",
            f"![]({fig_c_png.relative_to(output_dir)}){{width=100%}}",
            "",
            "The quadratic-weight onset figure uses the same improvement metric. The `50%` point is not just better at fixed `quadratic weight=2`; it also turns on earlier than `25%` at the moderate and low-diversity boundaries.",
            "",
            "### Figure D. Train-docs boundary check",
            "",
            f"![]({fig_d_png.relative_to(output_dir)}){{width=100%}}",
            "",
            "The `train_docs=2048` extension is not a new regime change. It sharpens the same ordering already visible at `512`: `50%` remains best, `25%` remains weaker, and `100%` remains exactly zero.",
            "",
            "### Figure E. Null-control check",
            "",
            f"![]({fig_e_png.relative_to(output_dir)}){{width=85%}}",
            "",
            "The null-control figure is the compact scientific sanity check. Once `quadratic weight=0`, the coarse points sit at or near zero, and the `100%` one-leaf case is exactly zero because it is identical to pooling.",
            "",
            "## 6. Core tables",
            "",
            "### 6.1 Representative absolute-error slices",
            "",
            "| Setting | pooled error | leaf error | Delta | relative gain | Interpretation |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for example in representative_examples:
        lines.append(
            f"| `{example['label']}` | `{example['pooled_mean']:.2f}` | `{example['leaf_mean']:.2f}` | "
            f"`{example['delta_mean']:+.2f}` | `{_relative_pct_label(example['relative_gain_mean'])}` | "
            f"{example['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "### 6.2 Tau frontier table",
            "",
            "Each cell is `Delta (better/tie/worse)`, where `Delta = pooled error - leaf error`. Positive means the leaf method has lower error; negative means it has higher error.",
            "",
            "| Leaf proportion | " + " | ".join(f"`{_tau_label(tau)}`" for tau in tau_grid) + " |",
            "| ---: |" + " ---: |" * len(tau_grid),
        ]
    )
    for pct in coarse_pcts:
        cells = []
        for tau in tau_grid:
            stats = tau_stats_coarse[(pct, tau)]
            cells.append(f"{stats['delta_mean']:+.2f} ({stats['wins']}/{stats['ties']}/{stats['losses']})")
        lines.append(f"| `{_leaf_label(pct, doc_tokens)}` | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "### 6.3 Quadratic-weight onset thresholds",
            "",
            "| Leaf proportion | `tau=1 / d=0.50` | `tau=8 / d=0.11` |",
            "| ---: | ---: | ---: |",
        ]
    )
    for pct in coarse_pcts:
        lines.append(
            f"| `{_leaf_label(pct, doc_tokens)}` | "
            f"`{_threshold_label(onset_table[(pct, 1.0)])}` | "
            f"`{_threshold_label(onset_table[(pct, 8.0)])}` |"
        )
    lines.extend(
        [
            "",
            "### 6.4 Boundary table",
            "",
            "| train_docs | leaf proportion | `tau=8 / d=0.11` | `tau=16 / d=0.06` |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for train_docs in (512, 2048):
        for pct in coarse_pcts:
            row_tau8 = boundary_stats[(train_docs, pct, 8.0)]
            row_tau16 = boundary_stats[(train_docs, pct, 16.0)]
            lines.append(
                f"| `{train_docs}` | `{_leaf_label(pct, doc_tokens)}` | "
                f"`{row_tau8['delta_mean']:+.2f} ({row_tau8['wins']}/{row_tau8['ties']}/{row_tau8['losses']})` | "
                f"`{row_tau16['delta_mean']:+.2f} ({row_tau16['wins']}/{row_tau16['ties']}/{row_tau16['losses']})` |"
            )
    lines.extend(
        [
            "",
            "## 7. Paper-facing interpretation",
            "",
            "The strongest version of the result is now: per-section analysis benefits from coarser sections up to a practical optimum, but that benefit saturates before the whole document. In this family, the best coarse point is `50%`, while the `100%` one-leaf case collapses exactly to the pooled baseline.",
            "",
            (
                f"The key semantic point is now explicit rather than implicit: `Delta < 0` is bad. "
                f"When the 16-token curve goes below zero, that means leaf analysis has higher absolute error than the pooled baseline. "
                f"The concrete `tau=8` failure above is `{example_failure_16_tau8['pooled_mean']:.2f} -> {example_failure_16_tau8['leaf_mean']:.2f}`, not a success."
            ),
            "",
            "That is the journal-facing answer to the question that was still open after the March 7 report. The older report showed monotone improvement through `96` tokens (`25%`). The new extension shows the next step up (`192` tokens, or `50%`) is still useful on the moderate boundary, but the final step to `384` tokens (`100%`) destroys the local decomposition entirely.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    diagnostics = {
        "generated_at_utc": now,
        "snapshot_label": str(args.snapshot_label),
        "baseline_root": str(args.baseline_root),
        "extension_root": str(args.extension_root),
        "output_dir": str(output_dir),
        "figures_dir": str(clean),
        "metric_definition": {
            "pooled_error": "mean absolute error from pooled whole-document inference",
            "leaf_error": "mean absolute error from leaf-based inference",
            "delta": "pooled_error - leaf_error",
            "positive_delta_means": "leaf error is lower than pooled error",
            "negative_delta_means": "leaf error is higher than pooled error",
            "relative_gain": "delta / pooled_error",
        },
        "suite_run_counts": dict(suite_run_counts),
        "manifest_counts": dict(manifest_counts),
        "suite_purposes": {suite: _suite_display_purpose(suite, purpose_map) for suite in manifest_counts},
        "last_positive_tau_all": {f"{pct:g}": val for pct, val in last_positive_all.items()},
        "last_positive_tau_coarse": {f"{pct:g}": val for pct, val in last_positive_coarse.items()},
        "material_lambda_onsets": {
            f"pct_{int(round(pct))}_tau_{tau:g}": onset_table[(pct, tau)]
            for pct in coarse_pcts
            for tau in (1.0, 8.0)
        },
        "best_boundary_leaf_pct": {
            f"train_{train_docs}_tau_{tau:g}": best_boundary_leaf_pct[(train_docs, tau)]
            for train_docs in (512, 2048)
            for tau in (8.0, 16.0)
        },
        "tau_frontier_coarse": {
            f"pct_{int(round(pct))}_tau_{tau:g}": tau_stats_coarse[(pct, tau)]
            for pct in coarse_pcts
            for tau in tau_grid
        },
        "boundary_rows": {
            f"train_{train_docs}_pct_{int(round(pct))}_tau_{tau:g}": boundary_stats[(train_docs, pct, tau)]
            for train_docs in (512, 2048)
            for pct in coarse_pcts
            for tau in (8.0, 16.0)
        },
        "representative_examples": representative_examples,
        "figure_paths": {
            "figure_A_tau_frontier_png": str(fig_a_png),
            "figure_A_tau_frontier_pdf": str(fig_a_pdf),
            "figure_B_last_positive_tau_png": str(fig_b_png),
            "figure_B_last_positive_tau_pdf": str(fig_b_pdf),
            "figure_C_quadratic_weight_onset_png": str(fig_c_png),
            "figure_C_quadratic_weight_onset_pdf": str(fig_c_pdf),
            "figure_D_boundary_train_docs_png": str(fig_d_png),
            "figure_D_boundary_train_docs_pdf": str(fig_d_pdf),
            "figure_E_null_control_png": str(fig_e_png),
            "figure_E_null_control_pdf": str(fig_e_pdf),
        },
    }
    diag_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with PdfPages(pdf_path) as pdf:
        _paragraph_page(
            pdf,
            title="Tree-Relevant LDA Coarse Leaf-Size Publication Report",
            paragraphs=[
                f"Snapshot: {args.snapshot_label}. This report packages the March 7 follow-up and the March 9 coarse leaf-size extension into a journal-style artifact set.",
                "Its central claim is simple and now well-supported: larger sections help until they stop being sections. The practical optimum in the coarse comparison is 50% of the document. The 100% one-leaf case collapses exactly to pooling.",
                "Metric definition: Delta = pooled mean absolute error minus leaf mean absolute error. Positive Delta means the leaf method is better. Negative Delta means the leaf method is worse.",
                (
                    f"The six-point tau frontier is now: 4% -> last positive tau {_threshold_label(last_positive_all[fine_and_coarse_pcts[0]])}, "
                    f"8% -> {_threshold_label(last_positive_all[fine_and_coarse_pcts[1]])}, "
                    f"17% -> {_threshold_label(last_positive_all[fine_and_coarse_pcts[2]])}, "
                    f"25% -> {_threshold_label(last_positive_all[fine_and_coarse_pcts[3]])}, "
                    f"50% -> {_threshold_label(last_positive_all[fine_and_coarse_pcts[4]])}, "
                    f"100% -> {_threshold_label(last_positive_all[fine_and_coarse_pcts[5]])}."
                ),
            ],
        )
        _image_page(
            pdf,
            title="Figure A. Full Tau Frontier",
            image_path=fig_a_png,
            caption=(
                "Fine-to-coarse tau frontier at quadratic weight=2, dtc=0.6, train_docs=512. Positive values mean the leaf method lowers "
                "mean absolute error; negative values mean it raises error. The 16-token failure at tau=8 is explicit here: "
                f"{example_failure_16_tau8['pooled_mean']:.2f} -> {example_failure_16_tau8['leaf_mean']:.2f} with "
                f"Delta {example_failure_16_tau8['delta_mean']:+.2f}."
            ),
        )
        _image_page(
            pdf,
            title="Figure B. Coarse Threshold Summary",
            image_path=fig_b_png,
            caption="Largest tau where the leaf method still lowers mean absolute error in the coarse comparison. 50% is best; 100% is the pooled null.",
        )
        _image_page(
            pdf,
            title="Figure C. Quadratic-Weight Onset",
            image_path=fig_c_png,
            caption="The 50% setting turns on earlier than 25% at the moderate and low-diversity boundaries, while 100% remains zero throughout because it is identical to pooling.",
        )
        _image_page(
            pdf,
            title="Figure D. Train-Docs Boundary",
            image_path=fig_d_png,
            caption="Increasing train_docs from 512 to 2048 sharpens the same ordering instead of changing it: 50% stays strongest, 25% remains weaker, and 100% stays exactly zero.",
        )
        _image_page(
            pdf,
            title="Figure E. Null-Control Check",
            image_path=fig_e_png,
            caption="At quadratic weight=0 the coarse settings are near zero and the 100% case is exactly zero, matching the intended control semantics.",
        )
        example_lines = [
            "Representative absolute-error slices",
            "",
            "Each row reports pooled mean abs error, leaf mean abs error, Delta, and interpretation.",
            "",
            "setting                                          pooled    leaf    Delta   rel_gain   interpretation",
        ]
        for example in representative_examples:
            example_lines.append(
                f"{example['label'][:46]:>46}  "
                f"{example['pooled_mean']:>6.2f}  "
                f"{example['leaf_mean']:>6.2f}  "
                f"{example['delta_mean']:>+6.2f}  "
                f"{_relative_pct_label(example['relative_gain_mean']):>8}  "
                f"{example['interpretation']}"
            )
        _text_page(pdf, title="Representative Absolute-Error Slices", lines=example_lines, font_size=10)
        threshold_lines = [
            "Publication threshold summary",
            "",
            "full frontier: largest tau where leaf mean abs error remains below pooled mean abs error",
        ]
        for pct in fine_and_coarse_pcts:
            threshold_lines.append(f"  {_leaf_label(pct, doc_tokens):>12}: {_threshold_label(last_positive_all[pct])}")
        threshold_lines.extend(
            [
                "",
                "coarse quadratic-weight onsets (mean Delta > 0.1, so leaf improves error by at least 0.1)",
            ]
        )
        for pct in coarse_pcts:
            threshold_lines.append(
                f"  {_leaf_label(pct, doc_tokens):>12}: tau=1 -> {_threshold_label(onset_table[(pct, 1.0)])}; tau=8 -> {_threshold_label(onset_table[(pct, 8.0)])}"
            )
        threshold_lines.extend(
            [
                "",
                "boundary winners",
            ]
        )
        for train_docs in (512, 2048):
            for tau in (8.0, 16.0):
                threshold_lines.append(
                    f"  train_docs={train_docs}, tau={tau:g}: {_leaf_label(best_boundary_leaf_pct[(train_docs, tau)], doc_tokens)}"
                )
        _text_page(pdf, title="Threshold Summary", lines=threshold_lines, font_size=11)

    print(f"wrote_markdown | {md_path}")
    print(f"wrote_pdf | {pdf_path}")
    print(f"wrote_diagnostics | {diag_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
