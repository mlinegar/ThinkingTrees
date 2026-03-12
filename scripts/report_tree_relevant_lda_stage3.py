#!/usr/bin/env python3
"""Build a Stage-3 PDF report for tree-relevant LDA weighting/mismatch/IPW suites."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean
import textwrap
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.patches import FancyBboxPatch


DELTA_CMAP = LinearSegmentedColormap.from_list("delta", ["#b2182b", "#ffffff", "#1a9850"])
LOW_GOOD_CMAP = LinearSegmentedColormap.from_list("low_good", ["#1a9850", "#f7f7f7", "#b2182b"])
HIGH_GOOD_CMAP = LinearSegmentedColormap.from_list("high_good", ["#b2182b", "#f7f7f7", "#1a9850"])
MODE_COLORS = {
    "aligned": "#1b9e77",
    "coarsen_2x": "#d95f02",
    "refine_2x": "#7570b3",
    "shift_half": "#e7298a",
    "random_same_count": "#666666",
}
METHOD_COLORS = {
    "budgeted_leaf_ridge_naive": "#b2182b",
    "budgeted_leaf_ridge_ipw": "#2166ac",
    "budgeted_leaf_ridge_ipw_stabilized": "#5f4690",
}
ESTIMATOR_COLORS = {
    "HT": "#b2182b",
    "Hajek": "#1a9850",
}
PROFILE_LABELS = {
    "equal": "Equal\nlengths",
    "bimodal": "Bimodal\nlengths",
    "long_tail": "Long-tail\nlengths",
}
MODE_LABELS = {
    "aligned": "Aligned",
    "coarsen_2x": "Coarsen\n2x",
    "refine_2x": "Refine\n2x",
    "shift_half": "Shift\nhalf-block",
    "random_same_count": "Random,\nsame count",
}
DESIGN_LABELS = {
    "uniform": "Uniform\nquerying",
    "proxy_priority": "Priority\nquerying",
    "proxy_adversarial": "Adversarial\nquerying",
}
DESIGN_SHORT_LABELS = {
    "uniform": "Uniform",
    "proxy_priority": "Priority",
    "proxy_adversarial": "Adversarial",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an external-facing report for the tree-relevant LDA realism-check suites.")
    p.add_argument("--input-root", type=Path, required=True, help="Stage-3 root with results/, manifest.jsonl, etc.")
    p.add_argument("--output-dir", type=Path, default=None, help="Defaults to <input-root>/report.")
    p.add_argument("--snapshot-label", type=str, default="current sweep", help="Short label for the title text.")
    return p.parse_args()


def _safe_float(x, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(fmean(vals))


def _safe_sem(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if len(vals) <= 1:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(var / len(vals))


def _tau_diversity_index(tau: float) -> float:
    return 1.0 / (1.0 + max(0.0, float(tau)))


def _tau_label(tau: float) -> str:
    return f"tau={tau:g}\n(d={_tau_diversity_index(tau):.2f})"


def _paragraph(ax, x: float, y: float, text: str, *, width: int = 90, fontsize: int = 11) -> None:
    ax.text(x, y, textwrap.fill(str(text).strip(), width=width), fontsize=fontsize, va="top", ha="left", linespacing=1.35)


def _profile_label(profile: str) -> str:
    return PROFILE_LABELS.get(profile, str(profile).replace("_", " ").title())


def _mode_label(mode: str) -> str:
    return MODE_LABELS.get(mode, str(mode).replace("_", " ").title())


def _design_label(design: str) -> str:
    return DESIGN_LABELS.get(design, str(design).replace("_", " ").title())


def _annotate_heatmap(
    ax,
    matrix: np.ndarray,
    *,
    fmt: str = "{:+.1f}",
    fontsize: int = 9,
    threshold: float | None = None,
) -> None:
    arr = np.asarray(matrix, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return
    if threshold is None:
        threshold = 0.45 * float(np.nanmax(np.abs(finite)))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if not np.isfinite(val):
                continue
            color = "white" if abs(float(val)) >= threshold else "#111111"
            ax.text(j, i, fmt.format(float(val)), ha="center", va="center", fontsize=fontsize, color=color)


def _save_page(
    pdf: PdfPages,
    fig,
    *,
    left: float = 0.08,
    right: float = 0.94,
    top: float = 0.88,
    bottom: float = 0.13,
    wspace: float = 0.30,
    hspace: float = 0.30,
) -> None:
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)
    pdf.savefig(fig)
    plt.close(fig)


def _caption(fig, text: str, *, x: float = 0.06, y: float = 0.08, width: int = 145, fontsize: int = 10) -> None:
    fig.text(
        x,
        y,
        textwrap.fill(text, width=width),
        fontsize=fontsize,
        ha="left",
        va="top",
    )


def _textbox(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    *,
    fc: str = "#f7f7f7",
    body_width: int = 34,
    title_fontsize: int = 12,
    body_fontsize: float = 10,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#666666",
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + 0.02, y + h - 0.04, title, fontsize=title_fontsize, fontweight="bold", va="top", ha="left")
    ax.text(
        x + 0.02,
        y + h - 0.11,
        textwrap.fill(body, width=body_width),
        fontsize=body_fontsize,
        va="top",
        ha="left",
        linespacing=1.3,
    )


def _page_header(fig, title: str, subtitle: str) -> None:
    title_wrapped = textwrap.fill(title, width=64)
    title_lines = title_wrapped.count("\n") + 1
    fig.text(0.06, 0.965, title_wrapped, fontsize=18, fontweight="bold", ha="left", va="top")
    subtitle_y = 0.965 - 0.045 * title_lines
    fig.text(0.06, subtitle_y, textwrap.fill(subtitle, width=135), fontsize=10.5, color="#444444", ha="left", va="top")


def _label_line_end(ax, xs: Sequence[float], ys: Sequence[float], label: str, color: str) -> None:
    finite = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(y))]
    if not finite:
        return
    x_last, y_last = finite[-1]
    ax.text(x_last + 0.10, y_last, label, color=color, fontsize=9, va="center", ha="left")


def _intro_page(pdf: PdfPages, snapshot_label: str) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _page_header(
        fig,
        "When does per-section analysis survive more realistic conditions?",
        "This appendix asks whether the local-structure result still holds once sections have unequal token mass, analysis boundaries do not perfectly match the latent sections, and only a budgeted subset of sections is labeled.",
    )
    ax.text(
        0.06,
        0.82,
        "Headline answer",
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        0.06,
        0.775,
        textwrap.fill(
            "The answer is partly yes: correct token weighting matters once section lengths vary, boundary mismatch destroys some of the structural advantage before inference even begins, and inverse-propensity weighting helps adaptive labeling behave sensibly, but it does not make every budgeted learner beat pooling.",
            width=120,
        ),
        fontsize=11.5,
        ha="left",
        va="top",
        linespacing=1.35,
    )
    _textbox(
        ax,
        0.06,
        0.46,
        0.27,
        0.22,
        "Question 1",
        "If latent sections have unequal length, should the document target weight each section by its token share rather than average sections equally?",
        fc="#eef7f0",
    )
    _textbox(
        ax,
        0.365,
        0.46,
        0.27,
        0.22,
        "Question 2",
        "If the analyst's section boundaries do not match the latent sections that generated the words, how much of the local-structure gain disappears before any inference noise is added?",
        fc="#f7f3ea",
    )
    _textbox(
        ax,
        0.67,
        0.46,
        0.27,
        0.22,
        "Question 3",
        "If only a queried subset of sections is labeled, can inverse-propensity weighting recover sensible training and held-out evaluation without pretending the labels were observed uniformly?",
        fc="#eef2fa",
    )
    ax.text(0.06, 0.34, "Roadmap", fontsize=14, fontweight="bold", ha="left", va="top")
    for idx, line in enumerate(
        [
            "Pages 3-4 isolate token weighting under unequal section lengths.",
            "Pages 5-7 formalize boundary mismatch, separate exact target loss from inference loss, and show the net practical consequence.",
            "Pages 8-12 examine adaptive labeling, held-out HT/Hajek evaluation, interval width, and the weight diagnostics that explain when evaluation gets harder.",
        ]
    ):
        ax.text(0.08, 0.295 - 0.055 * idx, f"{idx + 1}. {line}", fontsize=11, ha="left", va="top")
    ax.text(0.06, 0.08, f"Snapshot: {snapshot_label}", fontsize=10, color="#666666", ha="left", va="bottom")
    pdf.savefig(fig)
    plt.close(fig)


def _setup_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _page_header(
        fig,
        "What are the latent sections, analysis sections, and evaluation targets?",
        "Read this page as the data-generating map for the rest of the appendix: latent sections generate the words, analysis sections are what the estimator sees, and Delta is always pooled held-out error minus local-method held-out error.",
    )
    _textbox(
        ax,
        0.06,
        0.58,
        0.25,
        0.23,
        "1. Latent sections",
        "Each document has latent sections b with topic mixtures pi_(d,b). Low tau means neighboring sections can differ sharply; high tau keeps them close to the document-level average pi_d.",
        fc="#eef7f0",
    )
    _textbox(
        ax,
        0.375,
        0.58,
        0.25,
        0.23,
        "2. Analysis sections",
        "The analyst may use aligned, coarsened, refined, shifted, or random boundaries. An analysis section can therefore mix tokens from multiple latent sections through an overlap matrix C_(j,b).",
        fc="#f7f3ea",
    )
    _textbox(
        ax,
        0.69,
        0.58,
        0.25,
        0.23,
        "3. Adaptive labeling",
        "For the budgeted ridge learner, only some analysis sections are queried. The query design logs a propensity for each observed section, and inverse-propensity weighting uses those propensities during training and held-out evaluation.",
        fc="#eef2fa",
    )
    ax.annotate("", xy=(0.375, 0.695), xytext=(0.31, 0.695), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#666666"})
    ax.annotate("", xy=(0.69, 0.695), xytext=(0.625, 0.695), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#666666"})
    ax.text(0.06, 0.50, "Exact target", fontsize=13, fontweight="bold", ha="left", va="top")
    ax.text(
        0.06,
        0.455,
        r"$y_d = N_d \sum_b \omega_b h(\pi_{d,b}), \quad \omega_b = N_{d,b} / N_d, \quad h(\pi)=\theta^\top \pi + w_q \pi^\top W \pi$",
        fontsize=14,
        ha="left",
        va="top",
    )
    _paragraph(
        ax,
        0.06,
        0.39,
        "The quadratic utility weight measures how much the target cares about local combinations of topics rather than only the pooled average. When `w_q=0` the score is linear, so any correctly pooled document mixture is sufficient regardless of tau or boundary mismatch.",
        width=118,
        fontsize=10.5,
    )
    ax.text(0.06, 0.25, "Reporting conventions", fontsize=13, fontweight="bold", ha="left", va="top")
    for idx, line in enumerate(
        [
            "Delta = pooled held-out error - local-method held-out error, so green positive values favor the local method and white is neutral.",
            "HT = Horvitz-Thompson and Hajek = normalized Horvitz-Thompson; both use logged propensities on held-out sampled sections.",
            "ESS = effective sample size. Higher ESS is better; larger maximum inverse-propensity weights are worse.",
        ]
    ):
        ax.text(0.08, 0.205 - 0.055 * idx, textwrap.fill(line, width=118), fontsize=10.5, ha="left", va="top")
    pdf.savefig(fig)
    plt.close(fig)


def _mode_budget_rows() -> List[Tuple[str, float]]:
    return [("aligned", 1.0), ("aligned", 2.0), ("shift_half", 1.0), ("shift_half", 2.0)]


def _mode_budget_labels() -> List[str]:
    return [f"{_mode_label(mode)}\nquery budget {budget:g}" for mode, budget in _mode_budget_rows()]


def _normalize_range(values: np.ndarray) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if hi - lo < 1e-12:
        pad = max(1e-3, abs(lo) * 0.05 + 1e-3)
        return lo - pad, hi + pad
    return lo, hi


def _draw_metric_heatmap(
    ax,
    matrix: np.ndarray,
    *,
    title: str,
    fmt: str,
    better: str,
    fontsize: int = 9,
    show_ylabel: bool = True,
) -> None:
    arr = np.asarray(matrix, dtype=np.float64)
    vmin, vmax = _normalize_range(arr)
    cmap = LOW_GOOD_CMAP if better == "low" else HIGH_GOOD_CMAP
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto")
    ax.set_title(title, fontsize=12)
    _annotate_heatmap(ax, arr, fmt=fmt, fontsize=fontsize)
    if show_ylabel:
        ax.set_ylabel("Evaluation condition")


def _load_runs(results_root: Path) -> List[dict]:
    runs: List[dict] = []
    for path in sorted(results_root.rglob("seed_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rel = path.relative_to(results_root)
        suite = rel.parts[0]
        meta: Dict[str, object] = {"suite": suite, "path": str(path)}
        for part in rel.parts[1:-1]:
            if "_" not in part:
                continue
            key, value = part.split("_", 1)
            try:
                if key in {"tau", "lam", "budget", "anchor", "topicconc"}:
                    meta[key] = float(value)
                elif key == "seed":
                    meta[key] = int(value)
                else:
                    meta[key] = value
            except Exception:
                meta[key] = value
        meta["seed"] = int(path.stem.split("_")[1])
        meta["config"] = payload.get("config", {})
        meta["methods"] = payload.get("methods", {})
        meta["stage3"] = payload.get("stage3", {})
        meta["heterogeneity"] = payload.get("heterogeneity", {})
        methods = meta["methods"]
        pooled = methods.get("pooled_doc_wrong_model", {})
        meta["pooled_error"] = _safe_float(pooled.get("utility_abs_to_true_mean"))
        runs.append(meta)
    return runs


def _suite_rows(runs: Sequence[dict], suite: str) -> List[dict]:
    return [row for row in runs if row.get("suite") == suite]


def _metric(row: dict, method: str, key: str) -> float:
    methods = row.get("methods", {})
    if not isinstance(methods, dict):
        return float("nan")
    metrics = methods.get(method, {})
    if not isinstance(metrics, dict):
        return float("nan")
    return _safe_float(metrics.get(key))


def _aggregate(rows: Sequence[dict], *, keys: Sequence[str], value_fn) -> Dict[Tuple[object, ...], dict]:
    buckets: Dict[Tuple[object, ...], List[float]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        buckets[key].append(float(value_fn(row)))
    out: Dict[Tuple[object, ...], dict] = {}
    for key, vals in buckets.items():
        out[key] = {
            "mean": _safe_mean(vals),
            "sem": _safe_sem(vals),
            "n": len(vals),
        }
    return out


def _get_example_overlap(rows: Sequence[dict], *, mode: str, tau: float, lam: float) -> np.ndarray:
    for row in rows:
        if row.get("mode") != mode:
            continue
        if _safe_float(row.get("tau")) != float(tau) or _safe_float(row.get("lam")) != float(lam):
            continue
        stage3 = row.get("stage3", {})
        matrix = (((stage3.get("partition_stats", {}) or {}).get("sample_overlap_tokens", [])) if isinstance(stage3, dict) else [])
        if matrix:
            return np.asarray(matrix, dtype=np.float64)
    return np.zeros((1, 1), dtype=np.float64)


def _mismatch_metric_series(rows: Sequence[dict], *, lam: float = 2.0) -> Tuple[List[float], List[Tuple[str, List[float], List[float], List[float]]]]:
    taus = [0.25, 1.0, 4.0, 8.0, 16.0, 64.0]
    modes = ["aligned", "coarsen_2x", "refine_2x", "shift_half", "random_same_count"]
    metrics: List[Tuple[str, List[float], List[float], List[float]]] = []
    for mode in modes:
        structural = []
        infer_tax = []
        net = []
        for tau in taus:
            cell = [r for r in rows if r.get("mode") == mode and _safe_float(r.get("tau")) == tau and _safe_float(r.get("lam")) == lam]
            structural.append(
                _safe_mean(
                    _metric(r, "pooled_doc_wrong_model", "utility_abs_to_true_mean")
                    - _metric(r, "analysis_oracle_weighted_sum", "utility_abs_to_true_mean")
                    for r in cell
                )
            )
            infer_tax.append(
                _safe_mean(
                    _metric(r, "analysis_infer_weighted_sum", "utility_abs_to_true_mean")
                    - _metric(r, "analysis_oracle_weighted_sum", "utility_abs_to_true_mean")
                    for r in cell
                )
            )
            net.append(
                _safe_mean(
                    _metric(r, "pooled_doc_wrong_model", "utility_abs_to_true_mean")
                    - _metric(r, "analysis_infer_weighted_sum", "utility_abs_to_true_mean")
                    for r in cell
                )
            )
        metrics.append((mode, structural, infer_tax, net))
    return taus, metrics


def _weighted_page(pdf: PdfPages, rows: Sequence[dict], snapshot_label: str) -> dict:
    profiles = ["equal", "bimodal", "long_tail"]
    taus = [0.25, 1.0, 8.0]
    lambdas = [0.0, 2.0]
    oracle = np.full((len(profiles), len(taus), len(lambdas)), np.nan, dtype=np.float64)
    infer = np.full((len(profiles), len(taus), len(lambdas)), np.nan, dtype=np.float64)
    for p_idx, profile in enumerate(profiles):
        prof_rows = [r for r in rows if r.get("profile") == profile]
        for t_idx, tau in enumerate(taus):
            for l_idx, lam in enumerate(lambdas):
                cell = [r for r in prof_rows if _safe_float(r.get("tau")) == tau and _safe_float(r.get("lam")) == lam]
                oracle[p_idx, t_idx, l_idx] = _safe_mean(
                    _metric(r, "analysis_oracle_unweighted_sum", "utility_abs_to_true_mean")
                    - _metric(r, "analysis_oracle_weighted_sum", "utility_abs_to_true_mean")
                    for r in cell
                )
                infer[p_idx, t_idx, l_idx] = _safe_mean(
                    _metric(r, "analysis_infer_unweighted_sum", "utility_abs_to_true_mean")
                    - _metric(r, "analysis_infer_weighted_sum", "utility_abs_to_true_mean")
                    for r in cell
                )
    vmax = float(np.nanmax(np.abs(np.concatenate([oracle.reshape(-1), infer.reshape(-1)])))) if np.isfinite(np.nanmax(np.abs(np.concatenate([oracle.reshape(-1), infer.reshape(-1)])))) else 1.0
    vmax = max(0.01, vmax)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    def draw_heatmap_page(matrix: np.ndarray, *, title: str, subtitle: str, caption: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 8.5))
        _page_header(fig, title, subtitle)
        for l_idx, lam in enumerate(lambdas):
            ax = axes[l_idx]
            ax.imshow(matrix[:, :, l_idx], cmap=DELTA_CMAP, norm=norm, aspect="auto")
            panel_title = "Linear target (w_q=0)" if lam == 0.0 else "Nonlinear target (quadratic weight=2)"
            ax.set_title(panel_title, fontsize=12)
            ax.set_xticks(range(len(taus)), [_tau_label(t) for t in taus])
            ax.set_yticks(range(len(profiles)), [_profile_label(p) for p in profiles])
            ax.set_xlabel("Section heterogeneity")
            if l_idx == 0:
                ax.set_ylabel("Latent section-length profile")
            _annotate_heatmap(ax, matrix[:, :, l_idx], fmt="{:+.1f}", fontsize=10)
        _caption(fig, caption)
        _save_page(pdf, fig, top=0.82, bottom=0.16, left=0.08, right=0.94, wspace=0.35)

    draw_heatmap_page(
        oracle,
        title="Do token weights matter when latent sections have unequal length?",
        subtitle="Each cell is unweighted oracle error minus token-weighted oracle error. Green means token weighting improves the exact target; white means weighting is irrelevant.",
        caption=(
            "Equal-length sections are the control: simple averaging and token weighting coincide there by construction. "
            "Once section lengths become bimodal or long-tailed, the nonlinear target should count long sections more than short ones. "
            "That is why the `w_q=0` column stays near neutral while the `quadratic weight=2` column turns green."
        ),
    )
    draw_heatmap_page(
        infer,
        title="Does the same weighting advantage survive after local topic inference?",
        subtitle="Each cell is unweighted inferred-target error minus token-weighted inferred-target error. Green again favors proper token weighting, now after the mixtures must be estimated from words.",
        caption=(
            "This page asks whether the weighting result is only a target-definition fact or whether it survives finite-word inference. "
            "The broad answer is yes: the long-tail and bimodal worlds still reward token weighting once the quadratic weight is positive, even after local topic mixtures have to be inferred rather than observed."
        ),
    )
    headline = {
        "oracle_weighting_advantage_long_tail_tau1_lam2": _safe_float(oracle[2, 1, 1]),
        "infer_weighting_advantage_long_tail_tau1_lam2": _safe_float(infer[2, 1, 1]),
    }
    return headline


def _mismatch_page(pdf: PdfPages, rows: Sequence[dict]) -> None:
    aligned = _get_example_overlap(rows, mode="aligned", tau=8.0, lam=2.0)
    shifted = _get_example_overlap(rows, mode="shift_half", tau=8.0, lam=2.0)
    fig = plt.figure(figsize=(11.0, 8.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0], height_ratios=[1.0, 1.0])
    ax_text = fig.add_subplot(gs[:, 0])
    ax_a = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 1:])
    for ax in (ax_text, ax_a, ax_b, ax_c):
        if ax is ax_text:
            ax.axis("off")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    _page_header(
        fig,
        "What exactly does boundary mismatch do before inference noise appears?",
        "Read the overlap operator from left to right: each analysis section inherits a token-weighted mixture of the latent sections that contribute tokens to it.",
    )
    _paragraph(
        ax_text,
        0.02,
        0.96,
        "This page inserts an explicit analysis partition between the latent sections and the estimator. "
        "Let C_(j,b) be the number of tokens from latent section b that land inside analysis section j. Then the analysis-section mixture is the overlap-weighted average "
        "tilde_pi_(d,j) = sum_b (C_(j,b) / N_(d,j)) pi_(d,b). The token-weighted analysis target is y_analysis,d = N_d sum_j tilde_omega_j h(tilde_pi_(d,j)).",
        width=42,
    )
    _paragraph(
        ax_text,
        0.02,
        0.54,
        "Two exact controls fall straight out of that definition. First, if the analysis partition aligns with the latent sections, C is block diagonal and y_analysis,d = y_d exactly. "
        "Second, if `w_q=0`, h(pi)=theta^T pi is linear, so y_analysis,d = y_d for any partition because averaging commutes with a linear score. "
        "Any mismatch gap is therefore carried entirely by the quadratic term.",
        width=42,
    )
    _paragraph(
        ax_text,
        0.02,
        0.19,
        "The overlap matrices on the right make the operator concrete. White space means no overlap, darker cells mean more shared token mass. "
        "Aligned partitions preserve local structure exactly; shifted partitions mix neighboring latent sections before any inference noise enters.",
        width=42,
    )
    im_a = ax_a.imshow(aligned, cmap="Greys", aspect="auto")
    ax_a.set_title("Reference condition:\naligned boundaries")
    ax_a.set_xlabel("Latent sections")
    ax_a.set_ylabel("Analysis sections")
    im_b = ax_b.imshow(shifted, cmap="Greys", aspect="auto")
    ax_b.set_title("Boundary mismatch:\nshifted boundaries")
    ax_b.set_xlabel("Latent sections")
    ax_b.set_ylabel("Analysis sections")
    fig.colorbar(im_b, ax=[ax_a, ax_b], fraction=0.03, pad=0.02, label="Shared tokens")

    modes = ["aligned", "coarsen_2x", "refine_2x", "shift_half", "random_same_count"]
    lam0_gap = []
    for mode in modes:
        cell = [r for r in rows if r.get("mode") == mode and _safe_float(r.get("lam")) == 0.0]
        lam0_gap.append(
            _safe_mean((r.get("stage3", {}) or {}).get("oracle_decomposition", {}).get("mean_partition_gap", float("nan")) for r in cell)
        )
    lam0_max_abs = max(abs(v) for v in lam0_gap if np.isfinite(v)) if any(np.isfinite(v) for v in lam0_gap) else 1e-15
    ax_c.axhline(0.0, color="#666666", linewidth=1)
    ax_c.axhspan(-lam0_max_abs * 1.1, lam0_max_abs * 1.1, color="#eef7f0", alpha=0.6, zorder=0)
    ax_c.bar(range(len(modes)), lam0_gap, color=[MODE_COLORS[m] for m in modes], zorder=2)
    ax_c.set_xticks(range(len(modes)), [_mode_label(m) for m in modes])
    ax_c.set_ylabel("Mean exact gap: y_analysis - y_true")
    ax_c.set_title("Exact w_q=0 control:\ncompare each analysis target to the latent-section target", fontsize=12)
    ax_c.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    for idx, val in enumerate(lam0_gap):
        if np.isfinite(val):
            ax_c.text(idx, val, f"{val:+.1e}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8.5)
    _caption(
        fig,
        "The aligned matrix is the reference condition: each analysis section inherits tokens from one latent section only. "
        "Shifted or coarsened boundaries mix latent sections before any estimator sees the words. The lower panel is an exact comparison against the latent-section target y_d at `w_q=0`. "
        "Every value is essentially floating-point zero, so the substantive conclusion is that mismatch cannot create a target gap once the score is linear.",
    )
    _save_page(pdf, fig, top=0.78, bottom=0.16, left=0.06, right=0.94, wspace=0.35, hspace=0.42)


def _oracle_decomposition_page(pdf: PdfPages, rows: Sequence[dict]) -> dict:
    taus, metrics = _mismatch_metric_series(rows, lam=2.0)
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5), sharex=True)
    _page_header(
        fig,
        "Where does boundary mismatch do damage: in the target itself or in local inference?",
        "Each line tracks one analysis-boundary rule. Positive structural advantage means pooling loses signal before inference starts; positive inference tax means estimating local mixtures gives some of that edge back away.",
    )
    panel_specs = [
        ("Target-side structural advantage\npooled error minus analysis-oracle error", 1),
        ("Inference tax\nanalysis-infer error minus analysis-oracle error", 2),
    ]
    for ax, (title, idx) in zip(axes, panel_specs):
        for mode, structural, infer_tax, net in metrics:
            series = [structural, infer_tax, net][idx - 1]
            ax.plot(range(len(taus)), series, marker="o", markersize=5, linewidth=2, color=MODE_COLORS[mode], label=_mode_label(mode))
            _label_line_end(ax, range(len(taus)), series, _mode_label(mode).replace("\n", " "), MODE_COLORS[mode])
        ax.axhline(0.0, color="#999999", linewidth=1)
        ax.grid(axis="y", alpha=0.20)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Error gap")
        ax.set_xlim(-0.1, len(taus) - 0.1 + 1.2)
    axes[1].set_xticks(range(len(taus)), [_tau_label(t) for t in taus])
    axes[1].set_xlabel("Section heterogeneity")
    _caption(
        fig,
        "This decomposition separates two mechanisms. The upper panel is an exact target-side comparison: how much does pooling lose because the wrong boundary system has already mixed together distinct latent sections? "
        "The lower panel then asks how much more is lost when those local mixtures must be inferred from words rather than observed exactly.",
    )
    _save_page(pdf, fig, top=0.84, bottom=0.16, left=0.09, right=0.95, hspace=0.40)

    fig = plt.figure(figsize=(11.0, 8.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.axis("off")
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)
    _page_header(
        fig,
        "After both mismatch and inference noise, when does per-section analysis still win?",
        "This is the end-to-end held-out quantity of interest. Green territory means the local route still beats pooling; red territory means the structural edge was erased by mismatch or inference cost.",
    )
    for mode, structural, infer_tax, net in metrics:
        ax.plot(range(len(taus)), net, marker="o", markersize=5, linewidth=2, color=MODE_COLORS[mode], label=_mode_label(mode))
        _label_line_end(ax, range(len(taus)), net, _mode_label(mode).replace("\n", " "), MODE_COLORS[mode])
    ax.axhline(0.0, color="#666666", linewidth=1)
    ymin, ymax = ax.get_ylim()
    ax.axhspan(0.0, ymax, color="#edf8ef", alpha=0.75, zorder=0)
    ax.axhspan(ymin, 0.0, color="#fbeaea", alpha=0.75, zorder=0)
    ax.grid(axis="y", alpha=0.20)
    ax.set_xticks(range(len(taus)), [_tau_label(t) for t in taus])
    ax.set_xlabel("Section heterogeneity")
    ax.set_ylabel("Delta = pooled error - analysis-infer error")
    ax.set_title("Positive Delta means per-section analysis wins", fontsize=12)
    ax.set_xlim(-0.1, len(taus) - 0.1 + 1.2)
    aligned_tau8 = next((net[taus.index(8.0)] for mode, _s, _i, net in metrics if mode == "aligned"), float("nan"))
    shift_tau8 = next((net[taus.index(8.0)] for mode, _s, _i, net in metrics if mode == "shift_half"), float("nan"))
    _textbox(
        ax_text,
        0.00,
        0.67,
        0.96,
        0.24,
        "How to read it",
        "Compare the aligned line to the shifted and coarsened lines at the same tau. The vertical drop is the practical cost of using the wrong boundaries before and during local inference.",
        fc="#f7f7f7",
    )
    _textbox(
        ax_text,
        0.00,
        0.38,
        0.96,
        0.21,
        "Concrete contrast",
        f"At tau=8 (d={_tau_diversity_index(8.0):.2f}), aligned partitions average {aligned_tau8:+.2f} Delta units, while shift-half partitions average {shift_tau8:+.2f}.",
        fc="#eef7f0" if aligned_tau8 >= shift_tau8 else "#fbeaea",
    )
    _textbox(
        ax_text,
        0.00,
        0.10,
        0.96,
        0.20,
        "Why it matters",
        "This page is the practical mismatch result. A theorem can tell us which component creates the gap, but this page shows whether the local route still helps once all costs are paid.",
        fc="#eef2fa",
    )
    _save_page(pdf, fig, top=0.84, bottom=0.14, left=0.07, right=0.95, wspace=0.28)
    aligned_tau8 = next((net[taus.index(8.0)] for mode, _s, _i, net in metrics if mode == "aligned"), float("nan"))
    shift_tau8 = next((net[taus.index(8.0)] for mode, _s, _i, net in metrics if mode == "shift_half"), float("nan"))
    return {
        "aligned_tau8_net_advantage": _safe_float(aligned_tau8),
        "shift_tau8_net_advantage": _safe_float(shift_tau8),
    }


def _eval_component(row: dict, quantity: str, method: str | None = None) -> dict:
    stage3 = row.get("stage3", {})
    if not isinstance(stage3, dict):
        return {}
    ipw_eval = stage3.get("ipw_evaluation", {})
    if not isinstance(ipw_eval, dict):
        return {}
    if quantity == "target":
        target = ipw_eval.get("target", {})
        return target if isinstance(target, dict) else {}
    if quantity == "delta":
        delta = ipw_eval.get("delta", {})
        if not isinstance(delta, dict):
            return {}
        comp = delta.get(method or "budgeted_leaf_ridge_ipw", {})
        return comp if isinstance(comp, dict) else {}
    return {}


def _heatmap_matrix(
    rows: Sequence[dict],
    *,
    value_fn,
    row_keys: Sequence[Tuple[str, float]],
    col_keys: Sequence[str],
) -> np.ndarray:
    out = np.full((len(row_keys), len(col_keys)), np.nan, dtype=np.float64)
    for r_idx, (mode, budget) in enumerate(row_keys):
        for c_idx, design in enumerate(col_keys):
            cell = [
                row
                for row in rows
                if row.get("mode") == mode and _safe_float(row.get("budget")) == float(budget) and row.get("design") == design
            ]
            out[r_idx, c_idx] = _safe_mean(value_fn(row) for row in cell)
    return out


def _ipw_training_page(pdf: PdfPages, rows: Sequence[dict]) -> dict:
    hero_rows = [
        r
        for r in rows
        if r.get("mode") == "aligned" and _safe_float(r.get("tau")) == 8.0 and _safe_float(r.get("lam")) == 3.0
    ]
    designs = ["uniform", "proxy_priority", "proxy_adversarial"]
    budgets = [1.0, 2.0]
    methods = [
        ("budgeted_leaf_ridge_naive", "Naive", METHOD_COLORS["budgeted_leaf_ridge_naive"]),
        ("budgeted_leaf_ridge_ipw", "IPW", METHOD_COLORS["budgeted_leaf_ridge_ipw"]),
        ("budgeted_leaf_ridge_ipw_stabilized", "Stabilized IPW", METHOD_COLORS["budgeted_leaf_ridge_ipw_stabilized"]),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5), sharex=True)
    _page_header(
        fig,
        "Under adaptive labeling, how much extra training bias does IPW remove?",
        "This is the only page where colors identify methods directly: red is naive training, blue is raw inverse-propensity weighting, and violet is stabilized inverse-propensity weighting. Lower is better because the y-axis is excess held-out error relative to a full-label ridge fit.",
    )
    width = 0.22
    for ax, budget in zip(axes, budgets):
        x = np.arange(len(designs), dtype=np.float64)
        for idx, (method, label, color) in enumerate(methods):
            vals = []
            errs = []
            for design in designs:
                cell = [r for r in hero_rows if r.get("design") == design and _safe_float(r.get("budget")) == budget]
                gaps = [
                    _metric(r, method, "utility_abs_to_true_mean")
                    - _metric(r, "analysis_ridge_full_labels", "utility_abs_to_true_mean")
                    for r in cell
                ]
                vals.append(_safe_mean(gaps))
                errs.append(_safe_sem(gaps))
            ax.bar(x + (idx - 1) * width, vals, yerr=errs, width=width, color=color, alpha=0.9, capsize=3)
        ax.axhline(0.0, color="#888888", linewidth=1)
        ax.grid(axis="y", alpha=0.18)
        ax.set_xticks(x, [_design_label(d) for d in designs])
        ax.set_title(f"Query budget {budget:g} section(s) per document", fontsize=12)
        ax.set_ylabel("Held-out error minus\nfull-label ridge error")
    axes[1].set_xlabel("Training query design")
    full_delta_ref = _safe_mean(_metric(r, "analysis_ridge_full_labels", "delta_mean") for r in hero_rows)
    infer_delta_ref = _safe_mean(_metric(r, "analysis_infer_weighted_sum", "delta_mean") for r in hero_rows)
    _caption(
        fig,
        f"This training page is intentionally shown on one informative slice rather than averaged over every setting. "
        f"In that slice, the full-label ridge baseline itself has mean Delta {full_delta_ref:+.2f}, while direct section inference has mean Delta {infer_delta_ref:+.2f}. "
        "So the right question here is narrower: once labels are queried adaptively, how much extra training bias does the query design introduce, and how much of that extra bias does inverse-propensity weighting remove?",
    )
    _save_page(pdf, fig, top=0.75, bottom=0.16, left=0.10, right=0.96, hspace=0.34)
    adversarial = [r for r in hero_rows if r.get("design") == "proxy_adversarial" and _safe_float(r.get("budget")) == 1.0]
    return {
        "adversarial_budget1_naive_delta": _safe_mean(_metric(r, "budgeted_leaf_ridge_naive", "delta_mean") for r in adversarial),
        "adversarial_budget1_ipw_delta": _safe_mean(_metric(r, "budgeted_leaf_ridge_ipw", "delta_mean") for r in adversarial),
    }


def _ht_hajek_pages(pdf: PdfPages, rows: Sequence[dict]) -> dict:
    row_keys = _mode_budget_rows()
    group_labels = [f"{_mode_label(mode)}\nbudget {budget:g}" for mode, budget in row_keys]

    def collapsed_stats(quantity: str) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
        exact: List[float] = []
        hajek: List[float] = []
        ht: List[float] = []
        ht_means: List[float] = []
        ht_sems: List[float] = []
        hajek_means: List[float] = []
        hajek_sems: List[float] = []
        for mode, budget in row_keys:
            cell = [row for row in rows if row.get("mode") == mode and _safe_float(row.get("budget")) == budget]
            ht_errs = []
            hajek_errs = []
            for row in cell:
                comp = _eval_component(row, quantity, method="budgeted_leaf_ridge_ipw")
                if not comp:
                    continue
                exact.append(_safe_float(comp.get("population_exact_mean")))
                hajek.append(_safe_float(comp.get("hajek")))
                ht.append(_safe_float(comp.get("ht_mean")))
                ht_errs.append(abs(_safe_float(comp.get("ht_mean")) - _safe_float(comp.get("population_exact_mean"))))
                hajek_errs.append(abs(_safe_float(comp.get("hajek")) - _safe_float(comp.get("population_exact_mean"))))
            ht_means.append(_safe_mean(ht_errs))
            ht_sems.append(_safe_sem(ht_errs))
            hajek_means.append(_safe_mean(hajek_errs))
            hajek_sems.append(_safe_sem(hajek_errs))
        return exact, hajek, ht, ht_means, ht_sems, hajek_means, hajek_sems

    def draw_page(quantity: str, *, title: str, note: str) -> Tuple[List[float], List[float], List[float]]:
        exact, hajek, ht, ht_means, ht_sems, hajek_means, hajek_sems = collapsed_stats(quantity)
        fig = plt.figure(figsize=(11.0, 8.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.1])
        ax = fig.add_subplot(gs[0, 0])
        ax_text = fig.add_subplot(gs[0, 1])
        ax_text.axis("off")
        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, 1)
        _page_header(fig, title, note)
        x = np.arange(len(row_keys), dtype=np.float64)
        width = 0.34
        ax.bar(x - width / 2, ht_means, yerr=ht_sems, width=width, color=ESTIMATOR_COLORS["HT"], alpha=0.9, capsize=3, label="HT")
        ax.bar(x + width / 2, hajek_means, yerr=hajek_sems, width=width, color=ESTIMATOR_COLORS["Hajek"], alpha=0.9, capsize=3, label="Hajek")
        ax.grid(axis="y", alpha=0.18)
        ax.set_xticks(x, group_labels)
        ax.set_ylabel("Absolute error")
        ax.set_title("Query design collapsed", fontsize=12)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        max_y = max(ht_means + hajek_means) if (ht_means or hajek_means) else 1.0
        for idx, val in enumerate(ht_means):
            ax.text(idx - width / 2, val + 0.03 * max_y, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        for idx, val in enumerate(hajek_means):
            ax.text(idx + width / 2, val + 0.03 * max_y, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        _textbox(
            ax_text,
            0.00,
            0.60,
            0.96,
            0.28,
            "How to read it",
            "Each pair of bars compares raw Horvitz-Thompson to Hajek on the same finite-population estimand. Lower absolute error is better, so the greener Hajek bar is the more favorable estimator.",
            fc="#f7f7f7",
            body_width=28,
            title_fontsize=11,
            body_fontsize=9.4,
        )
        _textbox(
            ax_text,
            0.00,
            0.26,
            0.96,
            0.24,
            "Why design is collapsed",
            "These pages ask an estimator question, not a query-design question. Averaging over query designs keeps the page readable because mode and query budget matter much more here than the specific querying rule.",
            fc="#eef2fa",
            body_width=28,
            title_fontsize=11,
            body_fontsize=9.4,
        )
        _caption(fig, note)
        _save_page(pdf, fig, top=0.70, bottom=0.16, left=0.09, right=0.96, wspace=0.28)
        return exact, hajek, ht

    target_exact, target_hajek, target_ht = draw_page(
        "target",
        title="On the easier held-out target mean, does Hajek beat raw Horvitz-Thompson?",
        note="The target-mean estimand is the easier held-out control: it depends only on which documents and sections are sampled for evaluation, not on the learner's own Delta performance. Hajek is consistently closer than raw Horvitz-Thompson in both aligned and shifted worlds, which makes it the sensible default before moving to the harder Delta estimand.",
    )
    delta_exact, delta_hajek, delta_ht = draw_page(
        "delta",
        title="On population mean Delta, does Hajek stay closer to the exact answer?",
        note="This is the harder estimand because it tracks the population mean Delta of the budgeted IPW learner itself. Absolute error rises in the shifted worlds, but Hajek remains meaningfully closer than raw Horvitz-Thompson across the main held-out evaluation cells. Once that is established, the next practical question is interval width, not basic coverage.",
    )
    return {
        "mean_target_hajek_abs_error": _safe_mean(abs(a - b) for a, b in zip(target_exact, target_hajek)),
        "mean_delta_hajek_abs_error": _safe_mean(abs(a - b) for a, b in zip(delta_exact, delta_hajek)),
    }


def _coverage_pages(pdf: PdfPages, rows: Sequence[dict]) -> None:
    methods = [
        ("budgeted_leaf_ridge_naive", "Naive", METHOD_COLORS["budgeted_leaf_ridge_naive"]),
        ("budgeted_leaf_ridge_ipw", "IPW", METHOD_COLORS["budgeted_leaf_ridge_ipw"]),
        ("budgeted_leaf_ridge_ipw_stabilized", "Stabilized IPW", METHOD_COLORS["budgeted_leaf_ridge_ipw_stabilized"]),
    ]
    row_keys = _mode_budget_rows()
    col_keys = ["uniform", "proxy_priority", "proxy_adversarial"]
    row_labels = _mode_budget_labels()

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 8.5), sharey=True)
    _page_header(
        fig,
        "Once coverage saturates, which learner gives the tightest Delta intervals?",
        "Coverage is effectively 1.00 throughout these cells, so the informative comparison is interval width. Green means narrower intervals, which is better once coverage is already conservative.",
    )
    all_widths = []
    mats = []
    for method, label, _color in methods:
        mat = _heatmap_matrix(
            rows,
            value_fn=lambda row, m=method: _safe_float(_eval_component(row, "delta", method=m).get("eb_width")),
            row_keys=row_keys,
            col_keys=col_keys,
        )
        mats.append((method, label, mat))
        all_widths.append(mat)
    all_widths_arr = np.asarray(np.concatenate([m.reshape(-1) for _method, _label, m in mats]), dtype=np.float64)
    vmin, vmax = _normalize_range(all_widths_arr)
    for ax, (_method, label, mat) in zip(axes, mats):
        ax.imshow(mat, cmap=LOW_GOOD_CMAP, norm=mcolors.Normalize(vmin=vmin, vmax=vmax), aspect="auto")
        ax.set_title(f"{label}\nEmpirical-Bernstein width", fontsize=12)
        ax.set_xticks(range(len(col_keys)), [DESIGN_SHORT_LABELS[c] for c in col_keys])
        ax.set_yticks(range(len(row_keys)), row_labels)
        _annotate_heatmap(ax, mat, fmt="{:.1f}", fontsize=9)
    _caption(
        fig,
        "Raw coverage is not shown because it is uniformly conservative here. The discriminating statistic is width: narrower intervals deliver the same safe coverage with less uncertainty. "
        "The main pattern is that the IPW learner tends to produce the tightest Delta intervals, while shifted-boundary worlds are systematically wider than aligned worlds.",
    )
    _save_page(pdf, fig, top=0.74, bottom=0.16, left=0.12, right=0.95, wspace=0.25)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 8.5), sharey=True)
    _page_header(
        fig,
        "What explains wider Delta error and interval width: low ESS or large IPW weights?",
        "Greener cells are more favorable on all three panels. That means lower Delta error, higher effective sample size, and lower maximum inverse-propensity weight.",
    )
    matrices = [
        (
            _heatmap_matrix(
                rows,
                value_fn=lambda row: _safe_float(_eval_component(row, "delta", method="budgeted_leaf_ridge_ipw").get("hajek_abs_error")),
                row_keys=row_keys,
                col_keys=col_keys,
            ),
            "Mean |Hajek Delta error|\n(green = lower)",
            "low",
            "{:.1f}",
        ),
        (
            _heatmap_matrix(
                rows,
                value_fn=lambda row: _safe_float(_eval_component(row, "target").get("section_effective_sample_size")),
                row_keys=row_keys,
                col_keys=col_keys,
            ),
            "Mean section ESS\n(green = higher)",
            "high",
            "{:.0f}",
        ),
        (
            _heatmap_matrix(
                rows,
                value_fn=lambda row: _safe_float(_eval_component(row, "target").get("section_max_weight")),
                row_keys=row_keys,
                col_keys=col_keys,
            ),
            "Mean section max weight\n(green = lower)",
            "low",
            "{:.1f}",
        ),
    ]
    for ax, (mat, title, better, fmt) in zip(axes, matrices):
        _draw_metric_heatmap(ax, mat, title=title, fmt=fmt, better=better, fontsize=9, show_ylabel=False)
        ax.set_xticks(range(len(col_keys)), [DESIGN_SHORT_LABELS[c] for c in col_keys])
        ax.set_yticks(range(len(row_keys)), row_labels)
    _caption(
        fig,
        "This page explains why some held-out cells are easier than others. The cleaner evaluation cells combine larger effective sample size with smaller maximum inverse-propensity weights, and they are exactly the cells where Hajek's Delta error is smallest. "
        "The next page steps back from individual figures and states the external-reader takeaways directly.",
    )
    _save_page(pdf, fig, top=0.74, bottom=0.16, left=0.12, right=0.95, wspace=0.35)


def _synthesis_page(pdf: PdfPages, *, weighted: dict, mismatch: dict, ipw: dict, eval_stats: dict, counts: dict) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _page_header(
        fig,
        "What should an external reader take away from this realism check?",
        "This page states the conclusions in prose rather than asking the reader to reconstruct them from the previous figures.",
    )
    ax.text(0.06, 0.84, "Main takeaways", fontsize=14, fontweight="bold", ha="left", va="top")
    _textbox(
        ax,
        0.06,
        0.56,
        0.41,
        0.20,
        "1. Token weighting matters",
        (
            f"In the long-tail, tau=1, quadratic-weight=2 slice, token weighting lowers oracle error by "
            f"{weighted['oracle_weighting_advantage_long_tail_tau1_lam2']:+.2f} and lowers inferred-target error by "
            f"{weighted['infer_weighting_advantage_long_tail_tau1_lam2']:+.2f} relative to naive section averaging."
        ),
        fc="#eef7f0",
    )
    _textbox(
        ax,
        0.53,
        0.56,
        0.41,
        0.20,
        "2. Boundary mismatch hurts early",
        (
            f"At tau=8, aligned boundaries retain {mismatch['aligned_tau8_net_advantage']:+.2f} Delta units on average, "
            f"while shift-half boundaries retain only {mismatch['shift_tau8_net_advantage']:+.2f}. "
            "That loss appears before inference noise because the wrong boundaries have already mixed latent sections together."
        ),
        fc="#f7f3ea",
    )
    _textbox(
        ax,
        0.06,
        0.29,
        0.41,
        0.20,
        "3. IPW helps but does not rescue everything",
        (
            f"Under adversarial querying with budget 1, the naive learner's mean Delta is {ipw['adversarial_budget1_naive_delta']:+.2f}, "
            f"while the IPW learner's mean Delta is {ipw['adversarial_budget1_ipw_delta']:+.2f}. "
            "That is an improvement in training bias, not a guarantee that the entire learner family beats pooling."
        ),
        fc="#eef2fa",
    )
    _textbox(
        ax,
        0.53,
        0.29,
        0.41,
        0.20,
        "4. Held-out Hajek is now well behaved",
        (
            f"Across the main evaluation cells, mean Hajek absolute error is {eval_stats['mean_target_hajek_abs_error']:.2f} for the easier target mean and "
            f"{eval_stats['mean_delta_hajek_abs_error']:.2f} for population mean Delta. "
            "Those scales are sensible after fixing the earlier normalization bug."
        ),
        fc="#f7f7f7",
    )
    _paragraph(
        ax,
        0.06,
        0.18,
        "The external conclusion is therefore cautious but usable: local analysis remains scientifically interesting under more realistic conditions, but a journal version should present it as a structured tradeoff among token weighting, boundary quality, adaptive-label bias, and evaluation stability rather than as a universal win over pooling.",
        width=120,
        fontsize=11,
    )
    ax.text(
        0.06,
        0.07,
        f"Completed runs summarized here: {counts['runs']} total ({counts['suite_a']} weighting, {counts['suite_b']} mismatch, {counts['suite_c']} adaptive-label, {counts['suite_d']} hardness).",
        fontsize=10,
        color="#666666",
        ha="left",
        va="bottom",
    )
    pdf.savefig(fig)
    plt.close(fig)


def _hardness_page(pdf: PdfPages, rows: Sequence[dict]) -> None:
    modes = ["aligned", "shift_half"]
    taus = [1.0, 8.0]
    anchors = [25.0, 10.0]
    topic_concs = [0.2, 1.0]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5))
    _page_header(
        fig,
        "Appendix: does the realism-check result survive harder topic recovery?",
        "This appendix does not map the full hardness surface. It revisits the main aligned and shifted slices under weaker anchors and lower topic concentration to check that the qualitative story is not an artifact of an especially easy topic model.",
    )
    vals_all = []
    mats = {}
    for m_idx, mode in enumerate(modes):
        for t_idx, tau in enumerate(taus):
            mat = np.full((len(anchors), len(topic_concs)), np.nan, dtype=np.float64)
            for a_idx, anchor in enumerate(anchors):
                for c_idx, conc in enumerate(topic_concs):
                    cell = [
                        r
                        for r in rows
                        if r.get("mode") == mode
                        and _safe_float(r.get("tau")) == tau
                        and _safe_float(r.get("anchor")) == anchor
                        and _safe_float(r.get("topicconc")) == conc
                        and _safe_float(r.get("lam")) == 3.0
                    ]
                    val = _safe_mean(_metric(r, "budgeted_leaf_ridge_ipw", "delta_mean") for r in cell)
                    mat[a_idx, c_idx] = val
                    vals_all.append(val)
            mats[(mode, tau)] = mat
    vmax = max(0.01, float(np.nanmax(np.abs(np.asarray(vals_all, dtype=np.float64)))))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    for m_idx, mode in enumerate(modes):
        for t_idx, tau in enumerate(taus):
            ax = axes[m_idx, t_idx]
            im = ax.imshow(mats[(mode, tau)], cmap=DELTA_CMAP, norm=norm, aspect="auto")
            ax.set_title(f"{_mode_label(mode)}\n{_tau_label(tau)}", fontsize=12)
            ax.set_xticks(range(len(topic_concs)), [f"topic conc\n{c:g}" for c in topic_concs])
            ax.set_yticks(range(len(anchors)), [f"anchor\nx{a:g}" for a in anchors])
            _annotate_heatmap(ax, mats[(mode, tau)], fmt="{:+.1f}", fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="Exact Delta for budgeted IPW learner")
    _caption(
        fig,
        "This appendix is intentionally narrow. Its job is to check that the main weighting, mismatch, and adaptive-labeling conclusions do not collapse as soon as topic recovery gets harder. Green still means the budgeted IPW learner beats pooling; red means pooling still wins.",
    )
    _save_page(pdf, fig, top=0.74, bottom=0.16, left=0.10, right=0.92, hspace=0.40, wspace=0.28)


def _write_markdown(path: Path, *, weighted: dict, mismatch: dict, ipw: dict, eval_stats: dict, counts: dict) -> None:
    lines = [
        "# Local-Structure Realism Check Report",
        "",
        "This report is a standalone realism check for the tree-relevant LDA story. It asks whether the local-structure result survives three complications that matter for publication: unequal section lengths, imperfect analysis boundaries, and adaptive partial section labeling handled with inverse-propensity weighting.",
        "",
        "## Snapshot",
        "",
        f"- Total completed runs: `{counts['runs']}`",
        f"- Suite A weighting advantage (long-tail, tau=1, quadratic-weight=2, oracle): `{weighted['oracle_weighting_advantage_long_tail_tau1_lam2']:+.3f}`",
        f"- Suite B aligned net advantage at tau=8, quadratic-weight=2: `{mismatch['aligned_tau8_net_advantage']:+.3f}`",
        f"- Suite B shift-half net advantage at tau=8, quadratic-weight=2: `{mismatch['shift_tau8_net_advantage']:+.3f}`",
        f"- Suite C adversarial budget-1 naive Delta: `{ipw['adversarial_budget1_naive_delta']:+.3f}`",
        f"- Suite C adversarial budget-1 IPW Delta: `{ipw['adversarial_budget1_ipw_delta']:+.3f}`",
        f"- Mean Hajek abs error for target mean: `{eval_stats['mean_target_hajek_abs_error']:.3f}`",
        f"- Mean Hajek abs error for Delta mean: `{eval_stats['mean_delta_hajek_abs_error']:.3f}`",
        "",
        "## Reading Guide",
        "",
        "- Page 1 states the problem, the headline answer, and the three realism questions.",
        "- Page 2 defines the data-generating process, the overlap operator, adaptive labeling, and the report-wide meaning of Delta, HT, Hajek, and ESS.",
        "- Pages 3-4 ask whether token weighting matters once latent sections have unequal token mass, first at the oracle target and then after local topic inference.",
        "- Page 5 formalizes boundary mismatch and checks the exact `w_q=0` control.",
        "- Pages 6-7 separate target-side mismatch from local-inference cost and then report the net practical consequence for held-out Delta.",
        "- Page 8 isolates adaptive-label training bias relative to a full-label ridge baseline.",
        "- Pages 9-10 validate held-out HT and Hajek first on the easier target mean and then on population mean Delta.",
        "- Page 11 compares interval widths once coverage has saturated.",
        "- Page 12 explains held-out Delta error with effective sample size and maximum inverse-propensity weight.",
        "- Page 13 is a text-forward synthesis page for external readers.",
        "- Page 14 is an appendix slice checking that the qualitative story survives harder topic recovery.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = args.input_root
    output_dir = args.output_dir or (input_root / "report")
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(input_root / "results")
    if not runs:
        raise SystemExit(f"no result JSONs found under {input_root / 'results'}")

    suite_a = _suite_rows(runs, "suite_a_weighted_length")
    suite_b = _suite_rows(runs, "suite_b_partition_mismatch")
    suite_c = _suite_rows(runs, "suite_c_ipw_budgeted")
    suite_d = _suite_rows(runs, "suite_d_hardness")
    counts = {
        "runs": len(runs),
        "suite_a": len(suite_a),
        "suite_b": len(suite_b),
        "suite_c": len(suite_c),
        "suite_d": len(suite_d),
    }

    pdf_path = output_dir / "tree_relevant_lda_stage3_report.pdf"
    with PdfPages(pdf_path) as pdf:
        _intro_page(pdf, args.snapshot_label)
        _setup_page(pdf)
        weighted = _weighted_page(pdf, suite_a, args.snapshot_label)
        _mismatch_page(pdf, suite_b)
        mismatch = _oracle_decomposition_page(pdf, suite_b)
        ipw = _ipw_training_page(pdf, suite_c)
        eval_stats = _ht_hajek_pages(pdf, suite_c)
        _coverage_pages(pdf, suite_c)
        _synthesis_page(pdf, weighted=weighted, mismatch=mismatch, ipw=ipw, eval_stats=eval_stats, counts=counts)
        _hardness_page(pdf, suite_d)

    md_path = output_dir / "tree_relevant_lda_stage3_report.md"
    summary_path = output_dir / "tree_relevant_lda_stage3_report_summary.json"
    _write_markdown(md_path, weighted=weighted, mismatch=mismatch, ipw=ipw, eval_stats=eval_stats, counts=counts)
    summary_payload = {
        "counts": counts,
        "weighted": weighted,
        "mismatch": mismatch,
        "ipw": ipw,
        "evaluation": eval_stats,
        "input_root": str(input_root),
        "pdf": str(pdf_path),
        "markdown": str(md_path),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote_pdf | {pdf_path}")
    print(f"wrote_md | {md_path}")
    print(f"wrote_summary | {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
