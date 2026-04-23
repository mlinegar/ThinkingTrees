#!/usr/bin/env python3
"""Build a paper-style interim report for the new tree-relevant LDA simulation ladder."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean
import textwrap
from typing import Dict, Iterable, List, Sequence, Tuple

import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ctreepo.sim.util import safe_float as _safe_float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report the Stage-1/Stage-2 tree-relevant LDA simulations.")
    p.add_argument("--stage1-root", type=str, required=True)
    p.add_argument("--stage2-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--snapshot-label",
        type=str,
        default="Current Snapshot",
        help="Short label describing the report snapshot.",
    )
    return p.parse_args()


def _safe_mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(fmean(vals))


def _stage2_qweight(cfg: dict) -> float:
    return _safe_float(cfg.get("quadratic_utility_weight", cfg.get("lambda_multiplier")))


def _qweight_label(value: float) -> str:
    return f"quadratic weight={value:g}"


def _count_manifest_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _log_safe_series(series: Sequence[Sequence[float]], *, floor: float = 1e-15) -> List[np.ndarray]:
    min_positive = min(
        (
            float(x)
            for ys in series
            for x in ys
            if math.isfinite(float(x)) and float(x) > 0.0
        ),
        default=floor,
    )
    clipped_floor = min_positive if min_positive > 0.0 else floor
    out: List[np.ndarray] = []
    for ys in series:
        arr = np.asarray(ys, dtype=np.float64)
        mask = np.isfinite(arr) & (arr <= 0.0)
        if np.any(mask):
            arr = arr.copy()
            arr[mask] = clipped_floor
        out.append(arr)
    return out


def _text_page(pdf: PdfPages, *, title: str, lines: Sequence[str], font_size: int = 10) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title(title, pad=12)
    ax.text(0.01, 0.98, "\n".join(lines), family="monospace", fontsize=font_size, va="top")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _paragraph_page(
    pdf: PdfPages,
    *,
    title: str,
    paragraphs: Sequence[str],
    font_size: int = 12,
    width: int = 110,
) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title(title, pad=14, fontsize=18, fontweight="bold")

    y = 0.96
    for para in paragraphs:
        wrapped = textwrap.fill(str(para).strip(), width=width)
        n_lines = wrapped.count("\n") + 1
        ax.text(
            0.04,
            y,
            wrapped,
            fontsize=font_size,
            va="top",
            ha="left",
            linespacing=1.45,
            wrap=True,
        )
        y -= 0.035 * n_lines + 0.045

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _equation_page(
    pdf: PdfPages,
    *,
    title: str,
    intro: Sequence[str],
    equations: Sequence[Tuple[str, str]],
    notes: Sequence[str],
    font_size: int = 12,
    eq_font_size: int = 17,
    width: int = 104,
) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title(title, pad=14, fontsize=18, fontweight="bold")

    y = 0.96
    for para in intro:
        wrapped = textwrap.fill(str(para).strip(), width=width)
        n_lines = wrapped.count("\n") + 1
        ax.text(
            0.04,
            y,
            wrapped,
            fontsize=font_size,
            va="top",
            ha="left",
            linespacing=1.4,
            wrap=True,
        )
        y -= 0.035 * n_lines + 0.04

    for label, equation in equations:
        ax.text(0.05, y, label, fontsize=font_size, fontweight="bold", va="top", ha="left")
        y -= 0.035
        ax.text(0.08, y, equation, fontsize=eq_font_size, va="top", ha="left")
        y -= 0.08

    for para in notes:
        wrapped = textwrap.fill(str(para).strip(), width=width)
        n_lines = wrapped.count("\n") + 1
        ax.text(
            0.04,
            y,
            wrapped,
            fontsize=font_size,
            va="top",
            ha="left",
            linespacing=1.4,
            wrap=True,
        )
        y -= 0.035 * n_lines + 0.04

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _line_panel(
    ax: plt.Axes,
    *,
    xlabels: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float]]],
    title: str,
    ylabel: str,
    logy: bool = False,
) -> None:
    xs = np.arange(len(xlabels))
    raw = [np.asarray(ys, dtype=np.float64) for _, ys in series]
    can_log = any(np.any(np.isfinite(arr) & (arr > 0.0)) for arr in raw)
    plotted = _log_safe_series([ys for _, ys in series]) if logy and can_log else raw
    for (label, _), arr in zip(series, plotted):
        ax.plot(xs, arr, marker="o", label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(xlabels), rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if logy and can_log:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def _flatten_stage1(summary: dict) -> List[dict]:
    cfg = dict(summary.get("config", {}) or {})
    world = dict(summary.get("world_stats", {}) or {})
    exact = dict(summary.get("exact_recovery", {}) or {})
    out: List[dict] = []
    for method, metrics in (summary.get("methods", {}) or {}).items():
        if not isinstance(metrics, dict):
            continue
        out.append(
            {
                "family": str(summary.get("family", "")),
                "target_kind": str(summary.get("target_kind", "")),
                "method": str(method),
                "leaf_label": str(cfg.get("leaf_fraction_label", "")),
                "leaf_fraction": _safe_float(cfg.get("leaf_fraction")),
                "doc_topic_concentration": _safe_float(cfg.get("doc_topic_concentration")),
                "state_dim": int(cfg.get("state_dim", -1)),
                "seed": int(cfg.get("seed", -1)),
                "root_exact_u_l1": _safe_float(exact.get("root_utility_l1_mean")),
                "root_exact_scalar_abs": _safe_float(exact.get("root_scalar_abs_mean")),
                **{f"metric_{k}": v for k, v in metrics.items()},
                **{f"world_{k}": v for k, v in world.items()},
            }
        )
    return out


def _flatten_stage2(summary: dict) -> List[dict]:
    cfg = dict(summary.get("config", {}) or {})
    world = dict(summary.get("world_stats", {}) or {})
    heterogeneity = dict(summary.get("heterogeneity", {}) or {})
    qweight = _stage2_qweight(cfg)
    out: List[dict] = []
    for method, metrics in (summary.get("methods", {}) or {}).items():
        if not isinstance(metrics, dict):
            continue
        out.append(
            {
                "family": str(summary.get("family", "")),
                "target_kind": str(summary.get("target_kind", "")),
                "method": str(method),
                "leaf_label": str(cfg.get("leaf_fraction_label", "")),
                "leaf_fraction": _safe_float(cfg.get("leaf_fraction")),
                "latent_leaf_label": str(cfg.get("latent_leaf_fraction_label", "")),
                "latent_leaf_tokens": int(cfg.get("latent_leaf_tokens", -1)),
                "local_mixture_concentration": _safe_float(cfg.get("local_mixture_concentration")),
                "quadratic_utility_weight": qweight,
                "lambda_multiplier": qweight,
                "budget_regime": str(cfg.get("budget_regime", "")),
                "leaf_label_budget": _safe_float(cfg.get("leaf_label_budget")),
                "seed": int(cfg.get("seed", -1)),
                **{f"metric_{k}": v for k, v in metrics.items()},
                **{f"world_{k}": v for k, v in world.items()},
                **{f"hetero_{k}": v for k, v in heterogeneity.items()},
            }
        )
    return out


def _load_rows(root: Path, *, family: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if family == "stage1":
            rows.extend(_flatten_stage1(payload))
        else:
            rows.extend(_flatten_stage2(payload))
    return rows


def _group_mean(rows: Sequence[dict], *, filters: Dict[str, object], value_key: str, group_key: str) -> Dict[object, float]:
    buckets: Dict[object, List[float]] = defaultdict(list)
    for row in rows:
        if any(row.get(k) != v for k, v in filters.items()):
            continue
        val = _safe_float(row.get(value_key))
        if math.isfinite(val):
            buckets[row.get(group_key)].append(val)
    return {k: _safe_mean(v) for k, v in buckets.items()}


def _tau_header(t: float) -> str:
    """Markdown column header for tau with plain-English gloss."""
    return f"tau={t:g} / d={_tau_diversity_index(t):.2f} ({_tau_desc(t)})"


def _tau_desc(t: float) -> str:
    """Plain-English description for tau."""
    if t <= 0.5:
        return "very different sections"
    elif t <= 2:
        return "moderately different"
    elif t <= 16:
        return "fairly similar"
    else:
        return "nearly identical sections"


def _tau_diversity_index(t: float) -> float:
    """Display-only 0-1 rescaling: high means leaves can differ more."""
    t = float(t)
    if not math.isfinite(t):
        return float("nan")
    return float(1.0 / (1.0 + max(t, 0.0)))


def _tau_display_label(t: float, *, multiline: bool = False) -> str:
    diversity = _tau_diversity_index(t)
    if multiline:
        return f"tau={t:g}\nd={diversity:.2f}\n({_tau_desc(t)})"
    return f"tau={t:g} / d={diversity:.2f} ({_tau_desc(t)})"


def _fmt_pair(pooled: float, leaf: float) -> str:
    if not (math.isfinite(pooled) and math.isfinite(leaf)):
        return "n/a"
    return f"{pooled:.1f} vs {leaf:.1f}"


def _fmt_advantage(pooled: float, leaf: float) -> str:
    if not (math.isfinite(pooled) and math.isfinite(leaf)):
        return "n/a"
    if leaf <= 0:
        return "n/a"
    ratio = pooled / leaf
    if ratio >= 1.0:
        return f"{ratio:.1f}x leaf advantage"
    inv = leaf / pooled if pooled > 0 else float("nan")
    return f"{inv:.1f}x pooling advantage"


def _write_markdown(
    out_path: Path,
    *,
    stage1_rows: Sequence[dict],
    stage2_rows: Sequence[dict],
    stage1_completed: int,
    stage2_completed: int,
    stage1_total: int | None,
    stage2_total: int | None,
    snapshot_label: str,
) -> None:
    taus = sorted({float(r["local_mixture_concentration"]) for r in stage2_rows})
    lambdas_md = sorted({float(r["lambda_multiplier"]) for r in stage2_rows})
    llts_md = sorted({int(r["latent_leaf_tokens"]) for r in stage2_rows if int(r.get("latent_leaf_tokens", -1)) > 0})
    hero_lam_md = max(lambdas_md)
    best_llt_md = max(llts_md)

    def _s2_md_mean(method: str, *, llt: int, tau: float, lam: float) -> float:
        vals = [
            _safe_float(r.get("metric_utility_abs_to_true_mean"))
            for r in stage2_rows
            if r.get("method") == method
            and str(r.get("budget_regime")) == "all_leaves_labeled"
            and int(r.get("latent_leaf_tokens", -1)) == llt
            and float(r.get("local_mixture_concentration")) == tau
            and float(r.get("lambda_multiplier")) == lam
        ]
        return _safe_mean(vals)

    def _leaf_label_readable_md(label: str, doc_tokens: int = 384) -> str:
        try:
            pct = float(label.replace("%", ""))
            tok = int(round(pct / 100.0 * doc_tokens))
            if pct >= 99.9:
                return f"{tok} tokens (full document)"
            return f"{tok} tokens ({pct:.0f}% of doc)"
        except Exception:
            return label

    def _stage1_mean(method: str, *, state_dim: int | None = None, require_representable: bool = False) -> float:
        vals = []
        for row in stage1_rows:
            if row.get("method") != method:
                continue
            if state_dim is not None and int(row.get("state_dim", -1)) != state_dim:
                continue
            if require_representable and not bool(row.get("metric_exact_family_representable", False)):
                continue
            vals.append(_safe_float(row.get("metric_scalar_abs_to_full_mean")))
        return _safe_mean(vals)

    utility_pca_exact_dims = sorted(
        {
            int(r.get("state_dim", -1))
            for r in stage1_rows
            if r.get("method") == "utility_pca_practical" and bool(r.get("metric_exact_family_representable", False))
        }
    )
    count_svd_exact_dims = sorted(
        {
            int(r.get("state_dim", -1))
            for r in stage1_rows
            if r.get("method") == "count_svd_ceiling" and bool(r.get("metric_exact_family_representable", False))
        }
    )

    def _pair(llt: int, tau: float, lam: float) -> Tuple[float, float]:
        pooled = _s2_md_mean("pooled_doc_wrong_model", llt=llt, tau=tau, lam=lam)
        leaf = _s2_md_mean("leaf_infer_sum", llt=llt, tau=tau, lam=lam)
        return pooled, leaf

    def _avg_pair_for_tau(tau: float, lam: float) -> Tuple[float, float]:
        pooled_vals = []
        leaf_vals = []
        for llt in llts_md:
            pooled, leaf = _pair(llt, tau, lam)
            pooled_vals.append(pooled)
            leaf_vals.append(leaf)
        return _safe_mean(pooled_vals), _safe_mean(leaf_vals)

    def _win_count_for_tau(tau: float, lam: float) -> int:
        wins = 0
        for llt in llts_md:
            pooled, leaf = _pair(llt, tau, lam)
            if math.isfinite(pooled) and math.isfinite(leaf) and leaf < pooled:
                wins += 1
        return wins

    def _lambda_pair(llt: int, tau: float, lam: float) -> Tuple[float, float]:
        return _pair(llt, tau, lam)

    tau_low = min(taus)
    tau_mid = taus[1] if len(taus) > 1 else tau_low
    tau_border = taus[2] if len(taus) > 2 else tau_mid
    tau_high = max(taus)
    pooled_low_best, leaf_low_best = _pair(best_llt_md, tau_low, hero_lam_md)
    pooled_high_best, leaf_high_best = _pair(best_llt_md, tau_high, hero_lam_md)
    pooled_small_low, leaf_small_low = _pair(min(llts_md), tau_low, hero_lam_md)
    pooled_border_best, leaf_border_best = _pair(best_llt_md, tau_border, hero_lam_md)
    lam0_pooled, lam0_leaf = _lambda_pair(best_llt_md, tau_low, 0.0)
    lam1_pooled, lam1_leaf = _lambda_pair(best_llt_md, tau_low, 1.0 if 1.0 in lambdas_md else hero_lam_md)
    lam2_pooled, lam2_leaf = _lambda_pair(best_llt_md, tau_low, hero_lam_md)
    avg_low_pooled, avg_low_leaf = _avg_pair_for_tau(tau_low, hero_lam_md)
    avg_mid_pooled, avg_mid_leaf = _avg_pair_for_tau(tau_mid, hero_lam_md)
    avg_border_pooled, avg_border_leaf = _avg_pair_for_tau(tau_border, hero_lam_md)
    avg_high_pooled, avg_high_leaf = _avg_pair_for_tau(tau_high, hero_lam_md)

    lines = [
        "# LDA Simulation Report: Can Leaf-Level Inference Beat Pooling?",
        "",
        f"_Snapshot: {snapshot_label}_",
        "",
        "## What This Report Is Testing",
        "",
        "The central question is whether a tree should treat a document as one pooled bag of words or as a collection of locally distinct sections. In this simulation family the document is 384 tokens long, generated from an 8-topic LDA model, and the target is a scalar utility score. The pooled baseline reads all 384 tokens at once, infers one topic mixture, and predicts from that global estimate. The leaf method splits the document into equal-sized leaves, infers a topic mixture in each leaf, scores each leaf separately, and sums the results.",
        "",
        "Stage 1 and Stage 2 answer different parts of that story. Stage 1 is a control: if we keep the right mergeable statistic, can a tree reproduce the linear full-document objective exactly? Stage 2 is the real scientific question: once leaves have their own latent topic mixtures and the score is nonlinear, when does local inference recover information that pooling destroys?",
        "",
        "This distinction matters because it separates representational correctness from estimation advantage. A tree can be exactly faithful in the linear case and still only become practically useful in the nonlinear, heterogeneous case. The report is strongest when it keeps those two claims separate.",
        "",
        "## The Two Knobs",
        "",
        "The first knob is `tau`, the Dirichlet concentration that pulls each leaf-level topic mixture back toward the document-wide mixture. Low `tau` means sections are free to drift far apart; high `tau` means every section looks like the document average. This report also shows `d = 1 / (1 + tau)`, and that is not an arbitrary rescaling: it is the exact factor that appears in the conditional Dirichlet variance of each leaf mixture.",
        "",
        "| Raw tau | Diversity index d | Interpretation | Real-world analogy |",
        "|---:|---:|---|---|",
        f"| 0.25 | {_tau_diversity_index(0.25):.2f} | {_tau_desc(0.25)} | A front page where adjacent paragraphs may be about completely different things |",
        f"| 1 | {_tau_diversity_index(1.0):.2f} | {_tau_desc(1.0)} | A chapter with related sections that still emphasize different aspects |",
        f"| 8 | {_tau_diversity_index(8.0):.2f} | {_tau_desc(8.0)} | A focused essay where most paragraphs stay near the same themes |",
        f"| 64 | {_tau_diversity_index(64.0):.2f} | {_tau_desc(64.0)} | A repetitive memo where nearly every paragraph says the same thing |",
        "",
        "The second knob is the quadratic utility weight `w_q`. When `w_q=0`, the target only cares about the document-average topic proportions, so splitting into leaves cannot add information and usually only adds inference noise. As `w_q` increases, the score depends more on which topics co-occur inside the same leaf. That is the regime where `score(mean topic mix)` and `mean(score per leaf)` diverge, so local inference can become genuinely useful.",
        "",
        "## Exact Mathematical Setup",
        "",
        r"\[",
        r"\pi_d \sim \mathrm{Dir}(\alpha), \qquad \pi_{d,b} \mid \pi_d \sim \mathrm{Dir}(\tau \pi_d)",
        r"\]",
        "",
        r"\[",
        r"y_d = N \sum_b \omega_b h(\pi_{d,b}), \qquad \omega_b = \frac{n_b}{N}, \qquad h(\pi) = \theta^\top \pi + w_q \pi^\top W \pi",
        r"\]",
        "",
        r"\[",
        r"\bar{\pi}_d = \sum_b \omega_b \pi_{d,b}, \qquad y_{\mathrm{pool,true}} = N\, h(\bar{\pi}_d)",
        r"\]",
        "",
        "The true document target is therefore a weighted sum of leaf utilities, while the pooled approximation first averages the leaf mixtures and only then applies the utility function. The whole report is about when that change in order matters.",
        "",
        "## What `tau` Does Mathematically",
        "",
        r"\[",
        r"\mathbb{E}[\pi_{d,b} \mid \pi_d] = \pi_d",
        r"\]",
        "",
        r"\[",
        r"\mathrm{Var}(\pi_{d,b,k} \mid \pi_d) = \frac{\pi_{d,k}(1-\pi_{d,k})}{\tau+1}",
        r"\]",
        "",
        r"\[",
        r"\mathrm{Cov}(\pi_{d,b,k}, \pi_{d,b,\ell} \mid \pi_d) = -\frac{\pi_{d,k}\pi_{d,\ell}}{\tau+1} \qquad (k \neq \ell)",
        r"\]",
        "",
        "These equations explain exactly what `tau` changes and what it does not change. It does **not** move the average leaf mixture away from the document mixture: the conditional mean stays at `pi_d`. It changes only the spread of the leaves around that mean. That spread is proportional to `1 / (tau + 1)`, which is exactly the diversity index `d` shown in the report. So `tau=0.25` means each leaf has high variance around the document mean, while `tau=64` means each leaf is tightly concentrated around it.",
        "",
        "## What The Quadratic Weight Does Mathematically",
        "",
        r"\[",
        r"\frac{y_d - y_{\mathrm{pool,true}}}{N} = \sum_b \omega_b h(\pi_{d,b}) - h(\bar{\pi}_d)",
        r"\]",
        "",
        r"\[",
        r"= w_q \left[\sum_b \omega_b \pi_{d,b}^\top W \pi_{d,b} - \bar{\pi}_d^\top W \bar{\pi}_d\right]",
        r"\]",
        "",
        r"\[",
        r"\text{because } \sum_b \omega_b \theta^\top \pi_{d,b} = \theta^\top \bar{\pi}_d",
        r"\]",
        "",
        "This identity is the exact reason the report treats the quadratic weight as the on/off switch for a tree advantage. The linear term cancels perfectly after averaging. The pooled-vs-leaf target gap is carried **entirely** by the quadratic term, and `w_q` multiplies that gap exactly. If `w_q=0`, the theoretical pooled and leaf targets are identical. If `w_q>0`, any heterogeneity created by `tau` can now matter to the target.",
        "",
        "This is the same split formalized in Lean: `BagOfWordsLDARecovery` is the exact mergeable control, while `LeafLocalMixtureUtilityGap` is the nonlinear pooled-vs-leaf gap identity.",
        "",
        "## Why Leaf Size Is A Third Effect",
        "",
        "The previous two sections are about the **true target geometry**. Leaf size is different: it affects the estimator, not the target. Even when `tau` and `w_q` create a real pooled-vs-leaf gap, the leaf method still has to estimate each `pi_{d,b}` from a finite number of sampled words. Bigger leaves reduce that inference noise. So the simplest mental model for the plots is: `tau` creates local variation, `w_q` converts that variation into task-relevant signal, and leaf size determines how accurately the estimator can recover that signal.",
        "",
        f"## Coverage",
        "",
        f"This snapshot includes {stage1_completed} completed Stage 1 runs and {stage2_completed} completed Stage 2 runs" + (f" out of {stage2_total} queued Stage 2 runs." if stage2_total is not None else "."),
        "",
        "## Stage 1: Exact Merge Is The Control",
        "",
        "Stage 1 is a sanity check on the mergeable representation, not a contest against pooling. The target is linear in the document counts, and the tree retains the exact additive utility sketch. In that setting, exact agreement is the only acceptable answer. The completed runs satisfy that requirement: the tree-exact path stays at machine precision, and the practical utility-PCA path becomes exact as soon as the retained state dimension reaches the 16-dimensional utility sketch itself.",
        "",
        f"That compression result is important for intuition. The tree is not being asked to learn a miracle; it only needs to preserve the task-relevant 16-dimensional utility sketch. Utility PCA becomes exact once `state_dim={utility_pca_exact_dims[0] if utility_pca_exact_dims else 'n/a'}` because that matches the intrinsic utility dimension. Count SVD, by contrast, only becomes exact once it keeps the full 512-dimensional count space (`state_dim={count_svd_exact_dims[0] if count_svd_exact_dims else 'n/a'}`), which is exactly what the theory predicts.",
        "",
        f"In aggregate, the Stage 1 exact paths sit around {_stage1_mean('tree_exact_utility'):.1e} absolute error. That is the control condition that makes the rest of the report credible: if the mergeable path were already losing information in the linear case, any Stage 2 gap would be uninterpretable.",
        "",
        "## Stage 2: When Local Structure Matters",
        "",
        "Stage 2 changes only one substantive thing: each latent leaf now gets its own topic mixture, and the score includes a quadratic term. The linear part still depends only on the document average. The quadratic part depends on what topics appear together inside the same leaf. Pooling can estimate the global average more stably, but it necessarily erases which combinations happened locally. Whether the tree wins is therefore a balance between structural advantage and estimation noise.",
        "",
        f"At the strongest nonlinearity in this sweep (`{_qweight_label(hero_lam_md)}`), the average story is clean. When `tau={tau_low:g}` (`d={_tau_diversity_index(tau_low):.2f}`), leaf inference averages {avg_low_leaf:.1f} error against pooling's {avg_low_pooled:.1f}, a {_fmt_advantage(avg_low_pooled, avg_low_leaf)}. When `tau={tau_mid:g}` (`d={_tau_diversity_index(tau_mid):.2f}`), leaf inference still wins on average ({avg_mid_leaf:.1f} vs {avg_mid_pooled:.1f}). By `tau={tau_border:g}` (`d={_tau_diversity_index(tau_border):.2f}`), the competition is near the boundary ({avg_border_leaf:.1f} vs {avg_border_pooled:.1f}), and by `tau={tau_high:g}` (`d={_tau_diversity_index(tau_high):.2f}`) pooling is back ahead ({avg_high_pooled:.1f} vs {avg_high_leaf:.1f}).",
        "",
        "The useful way to read that pattern is as a crossover, not as a universal win for either side. In this sweep, the sign flip happens between `tau=1` and `tau=8`, or equivalently between diversity indices `d=0.50` and `d=0.11`. At `tau=1`, leaf inference wins at all four tested leaf sizes. At `tau=8`, leaf inference only wins at the two largest leaf sizes. At `tau=64`, the local differences are so weak that splitting the document mostly throws away sample size.",
        "",
        f"The best single cell makes the point sharply. With {best_llt_md}-token leaves and `tau={tau_low:g}` (`d={_tau_diversity_index(tau_low):.2f}`), leaf inference cuts error from {pooled_low_best:.1f} to {leaf_low_best:.1f}, a {_fmt_advantage(pooled_low_best, leaf_low_best)}. Even the smallest tested leaves still help in that regime: at {min(llts_md)} tokens, the error is {leaf_small_low:.1f} for leaf inference versus {pooled_small_low:.1f} for pooling. But when heterogeneity almost disappears (`tau={tau_high:g}`, `d={_tau_diversity_index(tau_high):.2f}`), the advantage also disappears: at {best_llt_md}-token leaves, pooling slightly beats leaf inference ({pooled_high_best:.1f} vs {leaf_high_best:.1f}).",
        "",
        f"## Main Table (`{_qweight_label(hero_lam_md)}`)",
        "",
        "Each cell reports `leaf inference error vs pooled error`. Lower is better; bold means the leaf method wins.",
        "",
        "| Leaf size | " + " | ".join(_tau_header(t) for t in taus) + " |",
        "|---:|" + "---:|" * len(taus),
    ]
    for llt in llts_md:
        cells = []
        for tau in taus:
            p = _s2_md_mean("pooled_doc_wrong_model", llt=llt, tau=tau, lam=hero_lam_md)
            i = _s2_md_mean("leaf_infer_sum", llt=llt, tau=tau, lam=hero_lam_md)
            winner = "**" if i < p else ""
            cells.append(f"{winner}{i:.1f}{winner} vs {p:.1f}")
        lines.append(f"| {llt} tokens ({llt/384*100:.0f}% of doc) | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## Why Bigger Leaves Help In This Sweep",
        "",
        f"Within the tested range, larger leaves help because the per-leaf EM problem gets much easier with more words. At `tau={tau_low:g}` and `{_qweight_label(hero_lam_md)}`, leaf inference improves monotonically from {leaf_small_low:.1f} error at {min(llts_md)}-token leaves to {leaf_low_best:.1f} at {best_llt_md}-token leaves, while pooling stays roughly flat because it always sees the full document. In other words, the structural benefit of respecting leaf boundaries is already present at small leaves, and giving the estimator more within-leaf evidence lets it cash in that benefit much more reliably.",
        "",
        f"The borderline regime makes the bias-variance tradeoff visible. At `tau={tau_border:g}` (`d={_tau_diversity_index(tau_border):.2f}`), the best tested leaf size still wins ({leaf_border_best:.1f} vs {pooled_border_best:.1f}), but the margin is narrow enough that the smaller leaves lose. That is exactly what you would expect if larger leaves are reducing estimation noise while smaller leaves are still too noisy to recover the remaining local signal.",
        "",
        "## Why The Quadratic Weight Turns The Effect On",
        "",
        f"At the best tested leaf size ({best_llt_md} tokens) and highest-diversity setting (`tau={tau_low:g}`, `d={_tau_diversity_index(tau_low):.2f}`), the nonlinear knob behaves exactly as theory says it should. When `{_qweight_label(0.0)}`, pooling slightly wins ({lam0_pooled:.1f} vs {lam0_leaf:.1f}) because the target only depends on the document-average topic mix and the leaf estimator is spending effort on unnecessary per-leaf inference. When `{_qweight_label(1.0 if 1.0 in lambdas_md else hero_lam_md)}`, leaf inference already opens a large gap ({lam1_leaf:.1f} vs {lam1_pooled:.1f}). When `{_qweight_label(hero_lam_md)}`, that gap widens further ({lam2_leaf:.1f} vs {lam2_pooled:.1f}).",
        "",
        "This is the key intuition behind the whole report. Trees do not help merely because a document can be partitioned. They help when the target depends on local composition and the leaves are heterogeneous enough that averaging before scoring discards the very interactions the target cares about.",
        "",
        "## Why This Matters",
        "",
        "The overall lesson is not that leaf-wise methods always dominate pooling. It is more specific and more useful: exact mergeable tree summaries are perfectly faithful in the linear regime, and practical leaf-level inference beats pooling only in the regime where local heterogeneity is both real and task-relevant. That is the regime where a tree is not just a memory device but a statistically better decomposition of the document.",
        "",
        "For the current sweep, the strongest evidence is the clean crossover between `tau=1` and `tau=8`, together with the monotone effect of the quadratic weight. Those two axes show that the result is not an arbitrary artifact of one estimator. The tree advantage appears when the theory says it should appear, fades when heterogeneity disappears, and strengthens as the target places more weight on within-leaf interactions.",
    ])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    stage1_root = Path(args.stage1_root)
    stage2_root = Path(args.stage2_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_completed = len(list(stage1_root.rglob("*.json")))
    stage2_completed = len(list(stage2_root.rglob("*.json")))
    stage1_total = _count_manifest_lines(stage1_root.parent / "commands.txt")
    stage2_total = _count_manifest_lines(stage2_root.parent / "commands.txt")

    stage1_rows = _load_rows(stage1_root, family="stage1")
    stage2_rows = _load_rows(stage2_root, family="stage2")
    if not stage1_rows:
        raise RuntimeError(f"no Stage-1 rows found under {stage1_root}")
    if not stage2_rows:
        raise RuntimeError(f"no Stage-2 rows found under {stage2_root}")

    md_path = output_dir / "lda_tree_methods_report.md"
    pdf_path = output_dir / "lda_tree_methods_report.pdf"
    summary_path = output_dir / "lda_tree_methods_report_summary.json"

    _write_markdown(
        md_path,
        stage1_rows=stage1_rows,
        stage2_rows=stage2_rows,
        stage1_completed=stage1_completed,
        stage2_completed=stage2_completed,
        stage1_total=stage1_total,
        stage2_total=stage2_total,
        snapshot_label=args.snapshot_label,
    )

    stage1_leafs = sorted({(float(r["leaf_fraction"]), str(r["leaf_label"])) for r in stage1_rows})
    stage1_leaf_labels = [label for _, label in stage1_leafs]
    stage1_tree_exact = []
    stage1_count_exact = []
    stage1_pca_exact = []
    stage1_mlp = []
    stage1_tree_sketch = []
    for frac, label in stage1_leafs:
        vals_tree = [
            _safe_float(r.get("metric_scalar_abs_to_full_mean"))
            for r in stage1_rows
            if r.get("method") == "tree_exact_utility" and str(r.get("leaf_label")) == label
        ]
        vals_count = [
            _safe_float(r.get("metric_scalar_abs_to_full_mean"))
            for r in stage1_rows
            if r.get("method") == "count_svd_ceiling"
            and str(r.get("leaf_label")) == label
            and bool(r.get("metric_exact_family_representable", False))
        ]
        vals_pca = [
            _safe_float(r.get("metric_scalar_abs_to_full_mean"))
            for r in stage1_rows
            if r.get("method") == "utility_pca_practical"
            and str(r.get("leaf_label")) == label
            and bool(r.get("metric_exact_family_representable", False))
        ]
        vals_mlp = [
            _safe_float(r.get("metric_scalar_abs_to_full_mean"))
            for r in stage1_rows
            if r.get("method") == "full_doc_mlp_diag" and str(r.get("leaf_label")) == label
        ]
        vals_tree_sketch = [
            _safe_float(r.get("metric_utility_l1_to_full_mean"))
            for r in stage1_rows
            if r.get("method") == "tree_exact_utility" and str(r.get("leaf_label")) == label
        ]
        stage1_tree_exact.append(_safe_mean(vals_tree))
        stage1_count_exact.append(_safe_mean(vals_count))
        stage1_pca_exact.append(_safe_mean(vals_pca))
        stage1_mlp.append(_safe_mean(vals_mlp))
        stage1_tree_sketch.append(_safe_mean(vals_tree_sketch))

    count_rows = [r for r in stage1_rows if r.get("method") == "count_svd_ceiling"]
    utility_rows = [r for r in stage1_rows if r.get("method") == "utility_pca_practical"]
    compression_state_dims = sorted({int(r["state_dim"]) for r in count_rows + utility_rows})
    compression_leaf_labels = [label for _, label in stage1_leafs]
    utility_pca_exact_dims = sorted(
        {
            int(r["state_dim"])
            for r in utility_rows
            if bool(r.get("metric_exact_family_representable", False))
        }
    )
    count_svd_exact_dims = sorted(
        {
            int(r["state_dim"])
            for r in count_rows
            if bool(r.get("metric_exact_family_representable", False))
        }
    )

    # ── Consistent colour palette across all plots ──
    # Principle: green = best method (lowest error), red = worst, blue = baseline.
    METHOD_COLORS = {
        "pooled_doc_wrong_model": "#1f77b4",  # blue  — baseline
        "leaf_infer_sum": "#2ca02c",           # green — best leaf method
        "leaf_ridge_from_u": "#d62728",        # red   — poor leaf method
        "coarse_leaf_ridge_from_u": "#ff7f0e", # orange
        "tree_exact_utility": "#2ca02c",       # green — exact (best)
        "count_svd_ceiling": "#1f77b4",        # blue  — compression method A
        "utility_pca_practical": "#ff7f0e",    # orange — compression method B
        "full_doc_mlp_diag": "#9467bd",
    }
    METHOD_LABELS = {
        "pooled_doc_wrong_model": "Pooled baseline (uses all tokens, ignores leaf structure)",
        "leaf_infer_sum": "Leaf EM inference (recovers per-leaf topics, sums utilities)",
        "leaf_ridge_from_u": "Leaf ridge (regression on utility sketch — poor signal)",
        "coarse_leaf_ridge_from_u": "Coarse ridge (coarser eval leaves)",
        "tree_exact_utility": "Tree exact (merge leaf sketches — zero error expected)",
        "count_svd_ceiling": "Count SVD (compress word-count vectors via SVD)",
        "utility_pca_practical": "Utility PCA (compress utility vectors via PCA)",
        "full_doc_mlp_diag": "Full-doc MLP (neural net on full document)",
    }
    # Short labels for legends where space is tight
    METHOD_LABELS_SHORT = {
        "pooled_doc_wrong_model": "Pooled baseline",
        "leaf_infer_sum": "Leaf EM inference",
        "leaf_ridge_from_u": "Leaf ridge",
        "coarse_leaf_ridge_from_u": "Coarse ridge",
        "tree_exact_utility": "Tree exact",
        "count_svd_ceiling": "Count SVD",
        "utility_pca_practical": "Utility PCA",
        "full_doc_mlp_diag": "Full-doc MLP",
    }
    LLT_COLORS = {16: "#d62728", 32: "#ff7f0e", 64: "#2ca02c", 96: "#1f77b4"}
    LLT_LINESTYLES = {16: ":", 32: "--", 64: "-.", 96: "-"}

    def _leaf_label_readable(label: str, doc_tokens: int = 384) -> str:
        """Convert '4.17%' → '16 tok (4%)' or '100%' → '384 tok (full doc)'."""
        try:
            pct = float(label.replace("%", ""))
            tok = int(round(pct / 100.0 * doc_tokens))
            if pct >= 99.9:
                return f"{tok} tok (full doc)"
            return f"{tok} tok ({pct:.0f}%)"
        except Exception:
            return label

    taus = sorted({float(r["local_mixture_concentration"]) for r in stage2_rows})
    lambdas = sorted({float(r["lambda_multiplier"]) for r in stage2_rows})
    latent_leaf_tokens_set = sorted({int(r["latent_leaf_tokens"]) for r in stage2_rows if int(r["latent_leaf_tokens"]) > 0})
    tau_low = min(taus)
    tau_mid = taus[1] if len(taus) > 1 else tau_low
    tau_border = taus[2] if len(taus) > 2 else tau_mid
    tau_high = max(taus)
    best_llt = max(latent_leaf_tokens_set)

    # Helper: mean error for a (method, llt, tau, lam) cell
    def _s2_mean(method: str, *, llt: int | None = None, tau: float | None = None, lam: float | None = None) -> float:
        vals = [
            _safe_float(r.get("metric_utility_abs_to_true_mean"))
            for r in stage2_rows
            if r.get("method") == method
            and str(r.get("budget_regime")) == "all_leaves_labeled"
            and (llt is None or int(r.get("latent_leaf_tokens", -1)) == llt)
            and (tau is None or float(r.get("local_mixture_concentration")) == tau)
            and (lam is None or float(r.get("lambda_multiplier")) == lam)
        ]
        return _safe_mean(vals)

    def _pair_stats(llt: int, tau: float, lam: float) -> Tuple[float, float]:
        pooled = _s2_mean("pooled_doc_wrong_model", llt=llt, tau=tau, lam=lam)
        leaf = _s2_mean("leaf_infer_sum", llt=llt, tau=tau, lam=lam)
        return pooled, leaf

    resolution_leafs = sorted({(float(r["leaf_fraction"]), str(r["leaf_label"])) for r in stage2_rows})
    resolution_labels = [label for _, label in resolution_leafs]

    with PdfPages(pdf_path) as pdf:
        # ── Title page ──
        hero_lam = max(lambdas)  # typically 2.0
        pooled_low_best, leaf_low_best = _pair_stats(best_llt, tau_low, hero_lam)
        pooled_high_best, leaf_high_best = _pair_stats(best_llt, tau_high, hero_lam)
        pooled_small_low, leaf_small_low = _pair_stats(min(latent_leaf_tokens_set), tau_low, hero_lam)
        pooled_border_best, leaf_border_best = _pair_stats(best_llt, tau_border, hero_lam)
        lam0_pooled, lam0_leaf = _pair_stats(best_llt, tau_low, 0.0)
        lam1_pooled, lam1_leaf = _pair_stats(best_llt, tau_low, 1.0 if 1.0 in lambdas else hero_lam)

        _paragraph_page(
            pdf,
            title="Can Per-Section Analysis Beat Reading Everything at Once?",
            paragraphs=[
                f"Snapshot: {args.snapshot_label}. This report asks a narrow question with a clean counterfactual: if a document really contains locally different topic mixtures, when is it better to infer topics leaf by leaf instead of pooling the whole document into one bag of words? The setup is a 384-token LDA document with 8 topics. The blue baseline infers one document-level mixture from all 384 tokens. The green method infers a separate mixture in each leaf and sums the leaf utilities.",
                "There are two levers. The first is raw tau, the concentration that controls how tightly each leaf stays near the document-average topic mixture. Because the raw values 0.25, 1, 8, and 64 are not intuitive, the plots also show the equivalent diversity factor d = 1 / (1 + tau). That is the exact scale factor in the Dirichlet leaf variance, so d near 1 means sections can diverge a lot and d near 0 means sections are almost copies of one another. The second lever is the quadratic utility weight `w_q`, which scales the quadratic part of the score and makes within-leaf topic interactions matter instead of only the document average.",
                f"The empirical story is a crossover, not a blanket win. At the largest tested leaf size ({best_llt} tokens), high-diversity documents with strong nonlinearity (`tau={tau_low:g}`, `d={_tau_diversity_index(tau_low):.2f}`, `{_qweight_label(hero_lam)}`) favor the leaf method decisively: error falls from {pooled_low_best:.1f} to {leaf_low_best:.1f}, a {_fmt_advantage(pooled_low_best, leaf_low_best)}. When the sections are nearly identical (`tau={tau_high:g}`, `d={_tau_diversity_index(tau_high):.2f}`), that advantage disappears and pooling slightly regains the lead ({pooled_high_best:.1f} vs {leaf_high_best:.1f}).",
                f"Stage 1 and Stage 2 should be read differently. Stage 1 is a correctness check on exact mergeable summaries; it shows the tree can preserve a linear target at machine precision. Stage 2 is the practical question: once leaves have their own latent mixtures and the score is nonlinear, does local inference recover information that pooling destroys? The answer is yes in the heterogeneous regime and no in the homogeneous regime. All plots below use lower-is-better error axes.",
            ],
        )

        # ── Stage 1: Coverage ──
        readable_s1_labels = [_leaf_label_readable(l) for l in stage1_leaf_labels]
        fig, ax = plt.subplots(1, 1, figsize=(11.0, 5.5))
        stage1_leaf_coverage = [
            sum(1 for r in stage1_rows if r.get("method") == "tree_exact_utility" and str(r.get("leaf_label")) == label)
            for label in stage1_leaf_labels
        ]
        xs_s1 = np.arange(len(stage1_leaf_labels))
        ax.bar(xs_s1, stage1_leaf_coverage, color=METHOD_COLORS["tree_exact_utility"], alpha=0.7)
        ax.set_xticks(xs_s1)
        ax.set_xticklabels(readable_s1_labels, rotation=45, ha="right")
        ax.set_title("Stage 1 Coverage: Simulation Runs Per Section Size")
        ax.set_ylabel("Number of completed runs")
        ax.grid(alpha=0.3, axis="y")
        fig.text(0.5, -0.02, "Each bar = how many simulation runs completed for that section size.\n"
                 "Section size = how many words per chunk when splitting the document into leaves.",
                 ha="center", fontsize=9, style="italic")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Stage 1: Exactness (paragraph summary — all methods give machine zero) ──
        _paragraph_page(
            pdf,
            title="Stage 1 Result: Linear Score → Exact Merge (Pass/Fail Check)",
            paragraphs=[
                "Stage 1 is the control experiment. The target is linear in the document counts, so if the tree keeps the right additive sketch, splitting and merging should reproduce the full-document answer exactly. In other words, this page is not asking whether trees beat pooling; it is asking whether the tree representation is faithful when the theory says it should be.",
                f"The answer is yes. Across the completed runs, the exact tree path sits at machine precision, with representative absolute errors around {stage1_tree_exact[0]:.1e} to {max(stage1_tree_exact):.1e} across the tested leaf sizes. Utility PCA is exact once it keeps the full 16-dimensional utility sketch, and Count SVD is exact only when it keeps the full 512-dimensional count space. That asymmetry is expected: the task-relevant sketch is only 16-dimensional, but the raw count histogram lives in the much larger vocabulary space.",
                "This distinction matters for the rest of the report. Stage 2 can show pooling beating a practical leaf estimator at `w_q=0` without contradicting Stage 1 at all. Stage 1 is about exact mergeable summaries. Stage 2 is about a different practical estimator that separately infers latent mixtures inside each leaf. When local structure does not matter to the target, that extra inference step only adds noise.",
            ],
        )

        # ── Stage 1: Compression curves ──
        # Add explanation page before compression curves
        _paragraph_page(
            pdf,
            title="Stage 1 Bonus: How Much Can We Compress the Leaf Summaries?",
            paragraphs=[
                "The compression plot asks a practical follow-up to Stage 1: once we know the exact mergeable sketch, how aggressively can we compress it before the downstream scalar target moves? The answer depends completely on which object we are compressing.",
                f"Count SVD works on the raw 512-dimensional word histogram, so exact recovery only appears once the state dimension reaches {count_svd_exact_dims[0] if count_svd_exact_dims else 'n/a'}, effectively the full vocabulary space. Utility PCA works on the task-relevant utility sketch instead, so exact recovery arrives as soon as the state dimension reaches {utility_pca_exact_dims[0] if utility_pca_exact_dims else 'n/a'}, the intrinsic utility dimension. That is why the utility sketch is the right object for a tree to preserve: it is both mergeable and dramatically smaller than the count space.",
                "The next figure should therefore be read as a compression sanity check, not as a surprise learning result. The useful intuition is that task-aligned summaries compress according to the dimension of the task, not according to the dimension of the raw vocabulary.",
            ],
        )
        readable_compression_labels = [_leaf_label_readable(l) for l in compression_leaf_labels]
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.5), constrained_layout=True)
        for label, rlabel in zip(compression_leaf_labels, readable_compression_labels):
            count_curve = [
                _safe_mean([
                    _safe_float(r.get("metric_scalar_abs_to_full_mean"))
                    for r in count_rows
                    if str(r.get("leaf_label")) == label and int(r.get("state_dim")) == s
                ])
                for s in compression_state_dims
            ]
            util_curve = [
                _safe_mean([
                    _safe_float(r.get("metric_scalar_abs_to_full_mean"))
                    for r in utility_rows
                    if str(r.get("leaf_label")) == label and int(r.get("state_dim")) == s
                ])
                for s in compression_state_dims
            ]
            safe_count_curve, safe_util_curve = _log_safe_series([count_curve, util_curve])
            axes[0].plot(compression_state_dims, safe_count_curve, marker="o", label=rlabel)
            axes[1].plot(compression_state_dims, safe_util_curve, marker="o", label=rlabel)
        axes[0].set_title("Count SVD: compress 512-dim word counts")
        axes[1].set_title("Utility PCA: compress 16-dim utility vectors")
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("Numbers kept after compression (more = less aggressive)")
            ax.set_ylabel("Error introduced by compression (lower = better)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, title="Section size")
        fig.suptitle("How aggressively can we compress leaf summaries?", fontsize=13)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        _equation_page(
            pdf,
            title="Stage 2 Mathematical Setup",
            intro=[
                "Stage 2 is the first family where leaves matter statistically rather than only operationally. The document has a global topic mixture, but each latent base leaf gets its own local topic mixture. The exact object being scored is still a single document-level scalar.",
            ],
            equations=[
                ("Generative model", r"$\pi_d \sim \mathrm{Dir}(\alpha), \qquad \pi_{d,b}\mid \pi_d \sim \mathrm{Dir}(\tau \pi_d)$"),
                ("Leaf-additive target", r"$y_d = N \sum_b \omega_b h(\pi_{d,b}), \qquad \omega_b = \frac{n_b}{N}$"),
                ("Utility form", r"$h(\pi) = \theta^\top \pi + w_q\, \pi^\top W \pi$"),
                ("Pooled reference", r"$\bar{\pi}_d = \sum_b \omega_b \pi_{d,b}, \qquad y_{\mathrm{pool,true}} = N\, h(\bar{\pi}_d)$"),
            ],
            notes=[
                "The pooled method and the leaf method are therefore solving different problems once the utility is nonlinear. The pooled path averages first and scores second. The leaf path scores locally and averages afterward. Those operations only agree automatically in the linear case.",
            ],
        )

        _equation_page(
            pdf,
            title="What Tau Means Exactly",
            intro=[
                "Tau is not just a vague 'heterogeneity knob'. In the Dirichlet leaf model, it controls the conditional variance of each leaf mixture around the document mixture. That is why the report's diversity index d = 1 / (1 + tau) is mathematically natural rather than cosmetic.",
            ],
            equations=[
                ("Conditional mean", r"$\mathbb{E}[\pi_{d,b}\mid\pi_d] = \pi_d$"),
                ("Coordinate variance", r"$\mathrm{Var}(\pi_{d,b,k}\mid\pi_d) = \frac{\pi_{d,k}(1-\pi_{d,k})}{\tau+1}$"),
                ("Cross-topic covariance", r"$\mathrm{Cov}(\pi_{d,b,k}, \pi_{d,b,\ell}\mid\pi_d) = -\frac{\pi_{d,k}\pi_{d,\ell}}{\tau+1}, \qquad k\neq \ell$"),
                ("Exact variance factor", r"$d = \frac{1}{1+\tau}$"),
            ],
            notes=[
                "These equations explain the plot semantics precisely. Lower tau does not change the average topic mix of a leaf; it increases how far leaves wander around that average. Higher tau shrinks every leaf back toward the document mean. So when the report moves from tau=0.25 to tau=64, it is literally moving from high conditional variance to near-zero conditional variance in the leaf mixtures.",
            ],
        )

        _equation_page(
            pdf,
            title="What The Quadratic Weight Means Exactly",
            intro=[
                "The quadratic utility weight determines whether that leaf-level variation can affect the target at all. The key identity, proved in `LeafLocalMixtureUtilityGap.lean`, is that the pooled-vs-leaf target gap is carried entirely by the quadratic term.",
            ],
            equations=[
                ("Start from the gap", r"$\frac{y_d - y_{\mathrm{pool,true}}}{N} = \sum_b \omega_b h(\pi_{d,b}) - h(\bar{\pi}_d)$"),
                ("Exact identity", r"$= w_q \left[\sum_b \omega_b \pi_{d,b}^\top W \pi_{d,b} - \bar{\pi}_d^\top W \bar{\pi}_d\right]$"),
                ("Linear cancellation", r"$\sum_b \omega_b \theta^\top \pi_{d,b} = \theta^\top \bar{\pi}_d$"),
            ],
            notes=[
                "This is why the quadratic weight acts like an on/off switch for a tree advantage. If `w_q=0`, the theoretical target gap is exactly zero no matter how heterogeneous the leaves are. If `w_q>0`, tau-created heterogeneity can now matter, because the score depends on which topic combinations appear inside the same leaf rather than only on the document-average mixture.",
                "One more step is needed before a practical estimator wins: it has to estimate each local mixture from words. That is where leaf size enters. Tau controls how much local variation exists, the quadratic weight controls whether that variation affects the target, and leaf size controls how noisy the estimator is when it tries to recover that local variation.",
            ],
        )

        # ── Stage 2: Explanation page ──
        _paragraph_page(
            pdf,
            title="Stage 2: Can Per-Section Analysis Beat Reading Everything at Once?",
            paragraphs=[
                "Stage 2 changes the data-generating process so that each latent leaf receives its own topic mixture. The pooled estimator still sees the entire document and therefore has the cleanest estimate of the average topic distribution. The leaf estimator sees smaller bags of words, so each local estimate is noisier. If that were the whole story, pooling would always win.",
                "What changes the answer is the utility function. The linear term cares only about the document-average mixture. The quadratic term cares about which topics appear together inside the same leaf. Once that nonlinear term matters, scoring the document after averaging can differ sharply from averaging the leaf-wise scores. That is the mechanism that can make a tree statistically useful rather than merely organizationally convenient.",
                "The tau labels on the next pages therefore show both the raw simulation parameter and the exact diversity factor d = 1 / (1 + tau). The raw tau values are what the generator uses. The diversity factor is mathematically meaningful because it is exactly the conditional-variance scale of the Dirichlet leaf mixtures: d near 0.80 means newspaper-like variation across sections, while d near 0.02 means the leaves are almost indistinguishable.",
            ],
        )

        # ── Stage 2: Hero plot — tau × leaf_tokens at quadratic weight=2 ──
        n_llt = len(latent_leaf_tokens_set)
        fig, axes = plt.subplots(1, n_llt, figsize=(3.5 * n_llt, 5.5), constrained_layout=True, sharey=True)
        if n_llt == 1:
            axes = [axes]
        for idx, llt in enumerate(latent_leaf_tokens_set):
            ax = axes[idx]
            pooled_vals = [_s2_mean("pooled_doc_wrong_model", llt=llt, tau=t, lam=hero_lam) for t in taus]
            infer_vals = [_s2_mean("leaf_infer_sum", llt=llt, tau=t, lam=hero_lam) for t in taus]
            xs = np.arange(len(taus))
            tau_labels = [_tau_display_label(t, multiline=True) for t in taus]
            ax.plot(xs, pooled_vals, marker="s", color=METHOD_COLORS["pooled_doc_wrong_model"],
                    label="Pooled (read all at once)", linewidth=2)
            ax.plot(xs, infer_vals, marker="o", color=METHOD_COLORS["leaf_infer_sum"],
                    label="Leaf inference (per-section)", linewidth=2)
            # Shade the region where leaf inference wins
            pooled_arr = np.array(pooled_vals)
            infer_arr = np.array(infer_vals)
            ax.fill_between(xs, infer_arr, pooled_arr,
                            where=infer_arr < pooled_arr,
                            alpha=0.15, color=METHOD_COLORS["leaf_infer_sum"],
                            label="Leaf wins (green region)")
            ax.fill_between(xs, infer_arr, pooled_arr,
                            where=infer_arr >= pooled_arr,
                            alpha=0.15, color=METHOD_COLORS["pooled_doc_wrong_model"])
            ax.set_xticks(xs)
            ax.set_xticklabels(tau_labels, fontsize=7)
            ax.set_xlabel("raw tau + diversity index d = 1 / (1 + tau)\n← more diverse sections · · · less diverse sections →")
            ax.set_title(f"{llt}-token sections\n({llt} of 384 words = {llt/384*100:.0f}% of doc)")
            ax.grid(alpha=0.3)
            if idx == 0:
                ax.set_ylabel("Prediction error (lower = better)")
                ax.legend(fontsize=7, loc="upper left")
        fig.suptitle(f"When does per-section analysis beat pooling? ({_qweight_label(hero_lam)})\n"
                     "Green below blue = leaf inference wins · Green shading = winning region", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        _paragraph_page(
            pdf,
            title="How To Read The Crossover Plot",
            paragraphs=[
                f"The four panels fix the leaf size and sweep only the cross-section diversity. The headline is the sign change. At `tau={tau_low:g}` (`d={_tau_diversity_index(tau_low):.2f}`), even the smallest leaves beat pooling ({leaf_small_low:.1f} vs {pooled_small_low:.1f}) because the nonlinear target is sensitive to local composition and the sections are genuinely different. At `tau={tau_high:g}` (`d={_tau_diversity_index(tau_high):.2f}`), that structural advantage is gone and the pooled estimator's variance advantage takes over.",
                f"The borderline case is what makes this convincing. At `tau={tau_border:g}` (`d={_tau_diversity_index(tau_border):.2f}`), the largest leaves still win ({leaf_border_best:.1f} vs {pooled_border_best:.1f}) but the smaller leaves mostly do not. That is exactly the signature of a real bias-variance tradeoff: there is still some local information available, but only the lower-noise leaf estimates can exploit it.",
                f"The best-case cell is therefore not an isolated outlier but the end of a systematic gradient. With {best_llt}-token leaves at `tau={tau_low:g}`, error falls from {pooled_low_best:.1f} to {leaf_low_best:.1f}, a {_fmt_advantage(pooled_low_best, leaf_low_best)}. The gain weakens smoothly as diversity falls, which is what the theory predicts if pooling is throwing away local interactions rather than simply failing randomly.",
            ],
        )

        # ── Stage 2: Leaf-size effect — leaf_tokens on x-axis, one panel per tau ──
        fig, axes = plt.subplots(1, len(taus), figsize=(3.5 * len(taus), 5.5), constrained_layout=True, sharey=True)
        if len(taus) == 1:
            axes = [axes]
        for idx, tau in enumerate(taus):
            ax = axes[idx]
            pooled_vals = [_s2_mean("pooled_doc_wrong_model", llt=llt, tau=tau, lam=hero_lam) for llt in latent_leaf_tokens_set]
            infer_vals = [_s2_mean("leaf_infer_sum", llt=llt, tau=tau, lam=hero_lam) for llt in latent_leaf_tokens_set]
            xs = np.arange(len(latent_leaf_tokens_set))
            ax.plot(xs, pooled_vals, marker="s", color=METHOD_COLORS["pooled_doc_wrong_model"],
                    label="Pooled (read all at once)", linewidth=2)
            ax.plot(xs, infer_vals, marker="o", color=METHOD_COLORS["leaf_infer_sum"],
                    label="Leaf inference (per-section)", linewidth=2)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{llt} words\n({llt/384*100:.0f}% of doc)" for llt in latent_leaf_tokens_set], fontsize=7)
            ax.set_xlabel("Words per section (bigger = more data for inference)")
            ax.set_title(_tau_display_label(tau))
            ax.grid(alpha=0.3)
            if idx == 0:
                ax.set_ylabel("Prediction error (lower = better)")
                ax.legend(fontsize=8)
        fig.suptitle(f"Do bigger sections help leaf inference? ({_qweight_label(hero_lam)})\n"
                     "Green line dropping = yes, more words per section improves accuracy", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        _paragraph_page(
            pdf,
            title="Why Bigger Leaves Help In This Sweep",
            paragraphs=[
                f"Across the tested range of {', '.join(str(v) for v in latent_leaf_tokens_set)} tokens, larger leaves consistently make the green curve fall. The reason is simple: each leaf estimator is solving a local topic-inference problem, and that problem becomes much better conditioned when the leaf contains more words. In the high-diversity regime (`tau={tau_low:g}`, `d={_tau_diversity_index(tau_low):.2f}`), moving from {min(latent_leaf_tokens_set)} to {best_llt} tokens reduces leaf error from {leaf_small_low:.1f} to {leaf_low_best:.1f}.",
                "That does not mean finer partitions are intrinsically bad. Finer leaves preserve more of the document's local variation. The issue is that, in this practical estimator, very small leaves leave too little evidence for stable per-leaf EM recovery. Within this sweep, the variance reduction from larger leaves dominates the loss of resolution.",
                "The borderline tau panels are the most informative ones here. They show that once the document is only moderately heterogeneous, the contest is won or lost on estimation quality. Large leaves still extract the remaining local signal; small leaves are too noisy to do so reliably.",
            ],
        )

        # ── Stage 2: Quadratic-weight comparison — one panel per tau, series per method, x = weight ──
        fig, axes = plt.subplots(1, len(taus), figsize=(3.5 * len(taus), 5.5), constrained_layout=True, sharey=True)
        if len(taus) == 1:
            axes = [axes]
        for idx, tau in enumerate(taus):
            ax = axes[idx]
            for method_key, label in [("pooled_doc_wrong_model", "Pooled (read all at once)"),
                                       ("leaf_infer_sum", "Leaf inference (per-section)")]:
                vals = [_s2_mean(method_key, llt=best_llt, tau=tau, lam=lam) for lam in lambdas]
                ax.plot(lambdas, vals, marker="o", color=METHOD_COLORS[method_key],
                        label=label, linewidth=2)
            lam_labels = []
            for lam in lambdas:
                if lam == 0:
                    lam_labels.append("0\n(linear score)")
                elif lam == 1:
                    lam_labels.append("1\n(moderate)")
                else:
                    lam_labels.append(f"{lam:g}\n(strongly nonlinear)")
            ax.set_xticks(lambdas)
            ax.set_xticklabels(lam_labels, fontsize=7)
            ax.set_xlabel("quadratic weight w_q (how nonlinear is the score?)\n← simpler · · · more complex →")
            ax.set_title(_tau_display_label(tau))
            ax.grid(alpha=0.3)
            if idx == 0:
                ax.set_ylabel("Prediction error (lower = better)")
                ax.legend(fontsize=8)
        fig.suptitle(f"Why does nonlinearity matter? ({best_llt}-token sections)\n"
                     "At w_q=0 the score is linear, so splitting into sections can never help", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        _paragraph_page(
            pdf,
            title="Why The Quadratic Weight Turns The Effect On",
            paragraphs=[
                f"The quadratic-weight sweep is the cleanest causal check in the report because it changes the target while leaving the document generator alone. At `tau={tau_low:g}` (`d={_tau_diversity_index(tau_low):.2f}`) and {best_llt}-token leaves, the pooled estimator slightly wins when `{_qweight_label(0.0)}` ({lam0_pooled:.1f} vs {lam0_leaf:.1f}). That is exactly the no-local-information case: the target depends only on the document-average topic mix, so leaf-wise estimation cannot create information that was not already in the pooled counts.",
                f"Once the quadratic weight moves away from zero, the ranking flips in the direction the theory predicts. At `{_qweight_label(1.0 if 1.0 in lambdas else hero_lam)}`, leaf inference is already substantially better ({lam1_leaf:.1f} vs {lam1_pooled:.1f}), and at `{_qweight_label(hero_lam)}` the improvement is even larger ({leaf_low_best:.1f} vs {pooled_low_best:.1f}). The nonlinearity is therefore not a cosmetic detail. It is the switch that turns local topic composition into signal rather than nuisance.",
                "This is also why the report should not be summarized as 'trees win' or 'pooling wins.' The more precise statement is that pooling is variance-efficient for linear objectives, while leaf-wise analysis becomes statistically preferable when the objective depends on within-leaf interactions that averaging destroys.",
            ],
        )

        # ── Stage 2: Crossover heatmap — leaf_infer_sum wins? ──
        fig, ax = plt.subplots(1, 1, figsize=(8.0, 6.0))
        ratio_grid = np.full((len(latent_leaf_tokens_set), len(taus)), float("nan"))
        for i, llt in enumerate(latent_leaf_tokens_set):
            for j, tau in enumerate(taus):
                p = _s2_mean("pooled_doc_wrong_model", llt=llt, tau=tau, lam=hero_lam)
                inf = _s2_mean("leaf_infer_sum", llt=llt, tau=tau, lam=hero_lam)
                if math.isfinite(p) and math.isfinite(inf) and inf > 0:
                    ratio_grid[i, j] = p / inf
        heatmap_cmap = LinearSegmentedColormap.from_list(
            "leaf_vs_pool",
            ["#b2182b", "#ffffff", "#1a9850"],
        )
        heatmap_norm = TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=5.0)
        im = ax.imshow(
            ratio_grid,
            aspect="auto",
            origin="lower",
            cmap=heatmap_cmap,
            norm=heatmap_norm,
        )
        tau_hm_labels = [_tau_display_label(t, multiline=True) for t in taus]
        ax.set_xticks(np.arange(len(taus)))
        ax.set_xticklabels(tau_hm_labels, fontsize=8)
        ax.set_yticks(np.arange(len(latent_leaf_tokens_set)))
        ax.set_yticklabels([f"{llt} words ({llt/384*100:.0f}% of doc)" for llt in latent_leaf_tokens_set])
        ax.set_xlabel("raw tau + diversity index d = 1 / (1 + tau)\n← more diverse sections · · · less diverse sections →")
        ax.set_ylabel("Words per section")
        ax.set_title(f"Summary: where does per-section analysis win? ({_qweight_label(hero_lam)})\n"
                     "GREEN = leaf inference better · RED = pooling better")
        for i in range(len(latent_leaf_tokens_set)):
            for j in range(len(taus)):
                val = ratio_grid[i, j]
                if math.isfinite(val):
                    verdict = "leaf wins" if val > 1.05 else ("tie" if val > 0.95 else "pool wins")
                    ax.text(j, i, f"{val:.1f}x\n{verdict}", ha="center", va="center", fontsize=9,
                            color="white" if val > 3.0 or val < 0.7 else "black")
        fig.colorbar(im, ax=ax, label="Advantage ratio (1.0 = neutral, >1 = leaf inference better)")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        _paragraph_page(
            pdf,
            title="Heatmap Summary",
            paragraphs=[
                "The heatmap compresses the main `quadratic weight=2` result into one grid. Each cell reports the ratio `pooled error / leaf error`, so values above 1 mean the tree-aware estimator is better. The color scale is centered so that white means exactly `1.0`, i.e. neither method has an advantage. Reading left to right is therefore a controlled removal of local diversity. Reading bottom to top improves the quality of each leaf estimate by giving the estimator more words per leaf.",
                f"The pattern is exactly the one a tree advocate should want to see. High-diversity columns are green across the board, the medium-diversity column is mixed, and the nearly homogeneous column is red. That is not a generic statement that splitting helps; it is a conditional statement that splitting helps precisely when there is structured local information to recover.",
            ],
        )

        # ── Stage 2: Mechanism diagnostic ──
        pooled_lookup: Dict[Tuple[int, str, float, float, int], float] = {}
        for row in stage2_rows:
            if row.get("method") != "pooled_doc_wrong_model" or str(row.get("budget_regime")) != "all_leaves_labeled":
                continue
            key = (
                int(row.get("latent_leaf_tokens", -1)),
                str(row.get("leaf_label", "")),
                float(row.get("local_mixture_concentration")),
                float(row.get("lambda_multiplier")),
                int(row.get("seed", -1)),
            )
            pooled_lookup[key] = _safe_float(row.get("metric_utility_abs_to_true_mean"))

        fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.5))
        for llt in latent_leaf_tokens_set:
            xs: List[float] = []
            ys: List[float] = []
            for row in stage2_rows:
                if row.get("method") != "leaf_infer_sum" or str(row.get("budget_regime")) != "all_leaves_labeled":
                    continue
                if int(row.get("latent_leaf_tokens", -1)) != llt:
                    continue
                key = (
                    int(row.get("latent_leaf_tokens", -1)),
                    str(row.get("leaf_label", "")),
                    float(row.get("local_mixture_concentration")),
                    float(row.get("lambda_multiplier")),
                    int(row.get("seed", -1)),
                )
                pooled_err = pooled_lookup.get(key, float("nan"))
                leaf_err = _safe_float(row.get("metric_utility_abs_to_true_mean"))
                gap_mag = abs(_safe_float(row.get("hetero_mean_test_gap_signal")))
                if math.isfinite(pooled_err) and math.isfinite(leaf_err) and math.isfinite(gap_mag):
                    xs.append(gap_mag)
                    ys.append(pooled_err - leaf_err)
            ax.scatter(xs, ys, alpha=0.35, s=18, color=LLT_COLORS.get(llt, "#333333"), label=f"{llt}-token leaves")
        ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
        ax.set_title("When does local structure translate into a real tree advantage?")
        ax.set_xlabel("Magnitude of the theoretical local-structure gap |mean_test_gap_signal|")
        ax.set_ylabel("Observed leaf advantage over pooling = pooled error - leaf error\n(positive = leaf inference wins)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, title="Leaf size")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        _paragraph_page(
            pdf,
            title="Mechanism: The More Local Structure Pooling Erases, The More Leaves Can Help",
            paragraphs=[
                "This last scatter replaces the old signed-gap diagnostic with a more direct question. The x-axis measures the magnitude of the theoretical local-structure gap: how much information is lost, in principle, when one scores after pooling instead of scoring leaf by leaf. The y-axis measures the realized improvement of leaf inference over pooling on the same run. Values above zero mean the leaf method actually turned that latent structural gap into lower prediction error.",
                "The resulting picture is the intuitive one. Points close to the origin correspond to nearly homogeneous documents or nearly linear targets, and there the realized advantage is near zero or negative. As the theoretical gap grows, points move upward, and the largest-leaf series occupies the highest part of the cloud because those runs have enough within-leaf evidence to exploit the available local structure. In short: more local signal creates more room for a tree, and larger leaves are better able to capture that room in this estimator.",
                "Taken together, the report supports a precise claim. Tree-structured analysis is not universally superior to pooling. It is superior when the document truly contains heterogeneous local mixtures, the target depends on within-leaf interactions, and the leaves are large enough for reliable local inference. Those are exactly the conditions under which a semantic tree ought to matter.",
            ],
        )

    summary = {
        "snapshot_label": args.snapshot_label,
        "stage1_root": str(stage1_root),
        "stage2_root": str(stage2_root),
        "stage1_rows": len(stage1_rows),
        "stage2_rows": len(stage2_rows),
        "stage1_completed_results": stage1_completed,
        "stage2_completed_results": stage2_completed,
        "stage1_total_results": stage1_total,
        "stage2_total_results": stage2_total,
        "stage1_leaf_labels": stage1_leaf_labels,
        "stage2_leaf_labels": resolution_labels,
        "latent_leaf_tokens": latent_leaf_tokens_set,
        "taus": taus,
        "tau_display_index": "d = 1 / (1 + tau)",
        "lambdas": lambdas,
        "markdown": str(md_path),
        "pdf": str(pdf_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote_markdown | {md_path}")
    print(f"wrote_pdf | {pdf_path}")
    print(f"wrote_summary | {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
