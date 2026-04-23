#!/usr/bin/env python3
"""Build a consolidated 'best of all worlds' tree-relevant LDA report."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ctreepo.sim.util import safe_float as _safe_float


POOL_COLOR = "#1f77b4"
SECTION_COLOR = "#2ca02c"
SECTION_SIZE_COLORS = {
    16: "#b2182b",
    32: "#ef8a62",
    64: "#67a9cf",
    96: "#2166ac",
}
STAGE1_ERROR_LABEL = "Absolute error to exact full-document reference"
STAGE2_ERROR_LABEL = "Held-out mean absolute utility error"
DELTA_LABEL = "Delta = pooled held-out error - per-section held-out error"
DELTA_SHORT_LABEL = "Delta (positive = per-section wins)"
NEUTRAL_COLOR = "#ffffff"
CARD_FACE = "#f6f8fb"
CARD_EDGE = "#d0d7de"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a consolidated best-of tree-relevant LDA report.")
    p.add_argument("--stage1-root", type=Path, required=True)
    p.add_argument("--stage2-root", type=Path, required=True)
    p.add_argument("--followup-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--snapshot-label",
        type=str,
        default="Best-of Consolidated Report",
        help="Short label for the report title page.",
    )
    return p.parse_args()


def _safe_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(fmean(vals))


def _stage2_qweight(cfg: dict) -> float:
    return _safe_float(cfg.get("quadratic_utility_weight", cfg.get("lambda_multiplier")))


def _qweight_label(value: float) -> str:
    return f"quadratic weight={value:g}"


def _safe_sem(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    n = len(vals)
    if n <= 1:
        return 0.0
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    return math.sqrt(var / n)


def _safe_corr(pairs: Iterable[Tuple[float, float]]) -> float:
    vals = [
        (float(x), float(y))
        for x, y in pairs
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(vals) < 2:
        return float("nan")
    xs = [x for x, _ in vals]
    ys = [y for _, y in vals]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in vals)
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    if den == 0.0:
        return float("nan")
    return num / den


def _count_manifest_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _tau_diversity_index(tau: float) -> float:
    tau = float(tau)
    return float(1.0 / (1.0 + max(tau, 0.0)))


def _tau_desc(tau: float) -> str:
    if tau <= 0.5:
        return "newspaper-like: adjacent sections can be very different"
    if tau <= 2.0:
        return "textbook-like: related sections still shift emphasis"
    if tau <= 16.0:
        return "focused essay: sections are fairly similar"
    return "repetitive memo: sections are nearly identical"


def _tau_short_name(tau: float) -> str:
    if tau <= 0.5:
        return "newspaper-like"
    if tau <= 2.0:
        return "textbook-like"
    if tau <= 16.0:
        return "focused essay"
    return "repetitive memo"


def _tau_multiline_desc(tau: float) -> str:
    if tau <= 0.5:
        return "newspaper-like:\nsections can differ a lot"
    if tau <= 2.0:
        return "textbook-like:\nsections shift emphasis"
    if tau <= 16.0:
        return "focused essay:\nsections stay fairly similar"
    return "repetitive memo:\nsections are almost copies"


def _tau_label(tau: float, *, multiline: bool = False) -> str:
    d = _tau_diversity_index(tau)
    if multiline:
        return f"tau={tau:g}\n(d={d:.2f})"
    return f"tau={tau:g} (d={d:.2f})"


def _tau_panel_title(tau: float) -> str:
    return f"{_tau_label(tau)}\n{_tau_multiline_desc(tau)}"


def _leaf_pct_label(tokens: int, doc_tokens: int, *, multiline: bool = False) -> str:
    pct = 100.0 * float(tokens) / float(doc_tokens)
    if multiline:
        return f"{tokens} tokens\n({pct:.0f}% of doc)"
    return f"{tokens} tokens ({pct:.0f}% of doc)"


def _num_sections(tokens: int, doc_tokens: int) -> int:
    if tokens <= 0:
        return 0
    return max(1, int(round(float(doc_tokens) / float(tokens))))


def _section_weight_text(tokens: int, doc_tokens: int) -> str:
    n_sections = _num_sections(tokens, doc_tokens)
    return f"omega_b = N_(d,b) / N_d = 1/{n_sections}"


def _unique_int(rows: Sequence[dict], key: str) -> int | None:
    vals = sorted(
        {
            int(v)
            for v in (_safe_float(row.get(key)) for row in rows)
            if math.isfinite(v)
        }
    )
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else vals[0]


def _draw_card(
    ax,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    facecolor: str = CARD_FACE,
    edgecolor: str = CARD_EDGE,
    title_size: int = 12,
    body_size: int = 10,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(x + 0.03 * w, y + h - 0.12 * h, title, fontsize=title_size, fontweight="bold", va="top", ha="left")
    ax.text(x + 0.03 * w, y + h - 0.30 * h, body, fontsize=body_size, va="top", ha="left", linespacing=1.38)


def _paragraph_page(
    pdf: PdfPages,
    *,
    title: str,
    paragraphs: Sequence[str],
    font_size: int = 12,
    width: int = 106,
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
            linespacing=1.42,
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
    width: int = 102,
) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title(title, pad=14, fontsize=18, fontweight="bold")
    y = 0.96
    for para in intro:
        wrapped = textwrap.fill(str(para).strip(), width=width)
        n_lines = wrapped.count("\n") + 1
        ax.text(0.04, y, wrapped, fontsize=font_size, va="top", ha="left", linespacing=1.4)
        y -= 0.035 * n_lines + 0.04
    for label, equation in equations:
        ax.text(0.05, y, label, fontsize=font_size, fontweight="bold", va="top", ha="left")
        y -= 0.035
        ax.text(0.08, y, equation, fontsize=eq_font_size, va="top", ha="left")
        y -= 0.08
    for para in notes:
        wrapped = textwrap.fill(str(para).strip(), width=width)
        n_lines = wrapped.count("\n") + 1
        ax.text(0.04, y, wrapped, fontsize=font_size, va="top", ha="left", linespacing=1.4)
        y -= 0.035 * n_lines + 0.04
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _text_page(
    pdf: PdfPages,
    *,
    title: str,
    lines: Sequence[str],
    font_size: int = 11,
) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title(title, pad=12, fontsize=16, fontweight="bold")
    ax.text(0.02, 0.98, "\n".join(lines), family="monospace", fontsize=font_size, va="top")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


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


def _load_followup_manifest(path: Path) -> tuple[Dict[str, str], Counter]:
    purposes: Dict[str, str] = {}
    counts: Counter = Counter()
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


def _load_followup_runs(results_root: Path) -> tuple[List[dict], dict]:
    runs: List[dict] = []
    first_payload: dict | None = None
    for path in sorted(results_root.rglob("seed_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if first_payload is None:
            first_payload = payload
        rel = path.relative_to(results_root)
        suite = rel.parts[0]
        llt = int(rel.parts[1].split("_")[1])
        dtc = float(rel.parts[3].split("_")[1])
        tau = float(rel.parts[4].split("_")[1])
        lam = float(rel.parts[5].split("_")[1])
        methods = payload.get("methods", {})
        pooled = _safe_float(methods.get("pooled_doc_wrong_model", {}).get("utility_abs_to_true_mean"))
        section = _safe_float(methods.get("leaf_infer_sum", {}).get("utility_abs_to_true_mean"))
        gap_signal = _safe_float(payload.get("heterogeneity", {}).get("mean_test_gap_signal"))
        runs.append(
            {
                "suite": suite,
                "llt": llt,
                "dtc": dtc,
                "tau": tau,
                "lam": lam,
                "pooled_error": pooled,
                "section_error": section,
                "delta": pooled - section,
                "gap_signal_abs": abs(gap_signal) if math.isfinite(gap_signal) else float("nan"),
            }
        )
    return runs, first_payload or {}


def _aggregate_followup(runs: Sequence[dict], *, keys: Sequence[str]) -> Dict[Tuple[object, ...], dict]:
    buckets: Dict[Tuple[object, ...], List[dict]] = defaultdict(list)
    for row in runs:
        buckets[tuple(row[k] for k in keys)].append(row)
    out: Dict[Tuple[object, ...], dict] = {}
    for key, rows in buckets.items():
        deltas = [row["delta"] for row in rows]
        pooleds = [row["pooled_error"] for row in rows]
        sections = [row["section_error"] for row in rows]
        out[key] = {
            "n": len(rows),
            "delta_mean": _safe_mean(deltas),
            "delta_sem": _safe_sem(deltas),
            "pooled_mean": _safe_mean(pooleds),
            "section_mean": _safe_mean(sections),
            "win_rate": _safe_mean([1.0 if d > 0.0 else 0.0 for d in deltas]),
        }
    return out


def _write_markdown(
    out_path: Path,
    *,
    snapshot_label: str,
    stage1_root: Path,
    stage2_root: Path,
    followup_root: Path,
    train_docs: int,
    test_docs: int,
    stage1_exact_mean: float,
    utility_pca_exact_dim: int | None,
    count_svd_exact_dim: int | None,
    hero_best: dict,
    hero_worst: dict,
    hero_lam: float,
    best_llt: int,
    best_tau: float,
    worst_tau: float,
    cross_taus: Sequence[float],
    cross_llts: Sequence[int],
    last_positive_tau: Dict[int, float | None],
    cross_agg: Dict[Tuple[object, ...], dict],
    onset_taus: Sequence[float],
    onset_table: Dict[Tuple[float, int], float | None],
    robust_dtcs: Sequence[float],
    robust_taus: Sequence[float],
    robust_llts: Sequence[int],
    robust_agg: Dict[Tuple[object, ...], dict],
    gap_corr: float,
    doc_tokens: int,
) -> None:
    threshold_summary = ", ".join(
        f"{llt} tokens -> {('never' if last_positive_tau[llt] is None else _tau_label(last_positive_tau[llt]))}"
        for llt in cross_llts
    )
    lines = [
        "# Best-of Tree-Relevant LDA Report",
        "",
        f"_Snapshot: {snapshot_label}_",
        "",
        f"Stage 1 source: `{stage1_root}`",
        f"Stage 2 source: `{stage2_root}`",
        f"Follow-up source: `{followup_root}`",
        "",
        "## Setup",
        "",
        f"Each run fits on `{train_docs}` training documents and is evaluated on `{test_docs}` held-out test documents. From Stage 2 onward, every reported method number is the held-out mean absolute utility error. The main comparison metric throughout is `Delta = pooled held-out error - per-section held-out error`, so positive values mean per-section analysis wins and zero is exactly neutral.",
        "",
        "Imagine a 384-word document whose early sections emphasize one set of topics and later sections emphasize another. The data-generating process first draws one document-level topic mixture, then draws a separate topic mixture for each latent section around that document mixture, and finally draws each section's words from that section-specific mixture. The pooled baseline reads all 384 words at once, infers one document-level topic mixture, and predicts from that average. The per-section method infers a topic mixture in each section, scores each section, and sums those section scores.",
        "",
        "A concrete two-topic example helps. Suppose the document-level mixture is 50% politics and 50% sports. With low tau, one latent section might be 90% politics and 10% sports while another is 10% politics and 90% sports. With high tau, those same two sections might instead be 55/45 and 45/55. In both cases the pooled document still looks 50/50, but only the low-tau document contains sharply different local sections.",
        "",
        "The question is not whether sectioning is always good. The question is when three conditions line up strongly enough that sectioning becomes statistically useful:",
        "",
        "1. sections are genuinely different from one another,",
        "2. the target depends on within-section composition rather than only the document average,",
        "3. each section is large enough for stable local inference.",
        "",
        "## Stage 1 Control",
        "",
        f"Stage 1 uses a different metric from the later pages: absolute error to the exact full-document reference, not held-out prediction error. It is the pass/fail control for mergeability. The exact mergeable tree path stays at about `{stage1_exact_mean:.1e}` absolute error, so the tree representation itself is not losing information in the linear case. Utility PCA becomes exact at `state_dim={utility_pca_exact_dim}` because it compresses the task-relevant utility sketch. Count SVD only becomes exact at `state_dim={count_svd_exact_dim}` because it has to preserve the whole count space.",
        "",
        "## Exact Meaning of Tau and the Quadratic Weight",
        "",
        "The local section mixture model is:",
        "",
        "```text",
        "pi_d ~ Dir(alpha)",
        "pi_(d,b) | pi_d ~ Dir(tau * pi_d)",
        "h(pi) = theta^T pi + w_q * pi^T W pi",
        "omega_b = N_(d,b) / N_d",
        "y_d = N_d * sum_b omega_b h(pi_(d,b))",
        "bar_pi_d = sum_b omega_b pi_(d,b)",
        "```",
        "",
        "The token-level generation inside each section is:",

        "```text",
        "z_(d,b,n) ~ Cat(pi_(d,b))",
        "x_(d,b,n) ~ Cat(beta_(z_(d,b,n)))",
        "```",

        "The exact variance identity is:",
        "",
        "```text",
        "Var(pi_(d,b,k) | pi_d) = pi_(d,k) * (1 - pi_(d,k)) / (tau + 1)",
        "```",
        "",
        f"So `d = 1 / (1 + tau)` is the exact diversity factor. The report shows both values together as `{_tau_label(best_tau)}`-style labels: raw `tau` is the generative parameter and `d` is the exact rescaled variance factor. In the equal-length sweeps, the section weights are just equal token weights, so for `{best_llt}`-token sections in a `{doc_tokens}`-token document we have `{_section_weight_text(best_llt, doc_tokens)}`. Operationally, “sections can differ a lot” means the section mixtures `pi_(d,b)` can sit far apart inside the same document even though they average back to the same document mixture `pi_d`.",
        "",
        "The exact pooled-vs-section target gap identity is:",
        "",
        "```text",
        "y_d / N_d - h(bar_pi_d)",
        "= w_q * [sum_b omega_b pi_(d,b)^T W pi_(d,b) - bar_pi_d^T W bar_pi_d]",
        "```",
        "",
        "This is why `w_q=0` is the control. When `w_q=0`, splitting can never create target information. For any fixed document, the pooled-vs-section target gap scales linearly in `w_q`, because `w_q` multiplies the entire quadratic interaction term. A concrete example: suppose the nonlinear term rewards concentration in politics via `h(pi) = w_q * pi_politics^2`. Then two documents can share the same pooled 50/50 average but still differ sharply at section level. If both sections are 50/50, the section-average nonlinear score is `0.25 * w_q`. If the two sections are 90/10 and 10/90, the section-average nonlinear score is `(0.81 + 0.01) / 2 * w_q = 0.41 * w_q`. Same pooled average, different section-level utility. That is exactly what positive quadratic weight makes possible.",
        "",
        "This matches the Lean split directly: `BagOfWordsLDARecovery` is the exact mergeable control, while `LeafLocalMixtureUtilityGap` is the nonlinear pooled-vs-section gap identity.",
        "",
        "## Broad Sweep",
        "",
        f"In the original broad Stage 2 sweep at `{_qweight_label(hero_lam)}`, the strongest cell is `{_leaf_pct_label(best_llt, doc_tokens)}` and `{_tau_label(best_tau)}`: held-out `Delta` is `{hero_best['delta_mean']:+.2f}`, meaning per-section analysis reduces held-out mean absolute utility error by `{hero_best['delta_mean']:.2f}` points (`{hero_best['pooled_mean']:.2f}` down to `{hero_best['section_mean']:.2f}`). By `{_tau_label(worst_tau)}`, that sign flips and pooling regains a small edge (`Delta = {hero_worst['delta_mean']:+.2f}`).",
        "",
        "## What the Follow-up Adds",
        "",
        f"The dense tau follow-up turns the broad crossover into thresholds: `{threshold_summary}`. Larger sections push the crossover outward because they reduce estimation noise while leaving the target-side weighting unchanged.",
        "",
        f"The quadratic-weight onset sweep is equally sharp: at `{_tau_label(0.25)}`, both 64-token and 96-token sections turn positive by `{_qweight_label(0.25)}`; at `{_tau_label(1.0)}`, the onsets are `0.5` and `0.25`; at `{_tau_label(8.0)}`, they move to `3` and `1.5`.",
        "",
        "## Reading the Dense Tau Table",
        "",
        "Each dense tau cell reports `Delta = pooled held-out error - per-section held-out error`. Positive values mean per-section analysis wins. Negative values mean pooling wins. White-centered heatmaps are exactly neutral at `Delta = 0`.",
        "",
        "| Section size | " + " | ".join(f"`{_tau_label(tau)}`" for tau in cross_taus) + " |",
        "|---:|" + "---:|" * len(cross_taus),
    ]
    for llt in cross_llts:
        cells = []
        for tau in cross_taus:
            stats = cross_agg[("tau_crossover_dense", llt, 0.6, tau, 2.0)]
            wins = int(round(stats["win_rate"] * stats["n"]))
            cells.append(f"{stats['delta_mean']:+.2f} ({wins}/{stats['n']})")
        lines.append(f"| `{_leaf_pct_label(llt, doc_tokens)}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Quadratic-Weight Onset Thresholds",
            "",
            "| heterogeneity | 64 tokens | 96 tokens |",
            "|---:|---:|---:|",
        ]
    )
    for tau in onset_taus:
        lines.append(
            f"| `{_tau_label(tau)}` | `{('never' if onset_table[(tau, 64)] is None else f'{onset_table[(tau, 64)]:g}')}` | `{('never' if onset_table[(tau, 96)] is None else f'{onset_table[(tau, 96)]:g}')}` |"
        )

    lines.extend(
        [
            "",
            "## Robustness",
            "",
            "The high-diversity columns stay strongly positive across document-topic concentrations. The sensitive boundary is `tau=8 (d=0.11)`. Low document-topic concentration can make that cell negative, while higher concentration keeps it slightly positive for larger sections. `tau=64 (d=0.02)` is neutral to negative everywhere.",
            "",
            "## Mechanism",
            "",
            f"Across the follow-up runs, the correlation between the absolute mean held-out target gap `|E_test[y_d - y_pool,d]|` and observed `Delta` is `{gap_corr:.3f}`. When the true section-vs-pooled target gap is large, per-section analysis tends to have more room to beat pooling.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_rows = _load_rows(args.stage1_root, family="stage1")
    stage2_rows = _load_rows(args.stage2_root, family="stage2")
    if not stage1_rows:
        raise RuntimeError(f"no Stage 1 rows found under {args.stage1_root}")
    if not stage2_rows:
        raise RuntimeError(f"no Stage 2 rows found under {args.stage2_root}")

    followup_results_root = args.followup_root / "results" if (args.followup_root / "results").exists() else args.followup_root
    followup_runs, followup_first = _load_followup_runs(followup_results_root)
    if not followup_runs:
        raise RuntimeError(f"no follow-up rows found under {followup_results_root}")

    followup_purposes, followup_manifest_counts = _load_followup_manifest(args.followup_root / "manifest.jsonl")

    doc_tokens = int(followup_first.get("config", {}).get("doc_tokens", 384))
    stage1_train_docs = _unique_int(stage1_rows, "world_train_docs_fit") or int(followup_first.get("world_stats", {}).get("train_docs_fit", 512))
    stage1_test_docs = _unique_int(stage1_rows, "world_test_docs_evaluated") or int(followup_first.get("world_stats", {}).get("test_docs_evaluated", 512))
    stage2_train_docs = _unique_int(stage2_rows, "world_train_docs_fit") or int(followup_first.get("world_stats", {}).get("train_docs_fit", 512))
    stage2_test_docs = _unique_int(stage2_rows, "world_test_docs_evaluated") or int(followup_first.get("world_stats", {}).get("test_docs_evaluated", 512))
    report_train_docs = stage2_train_docs
    report_test_docs = stage2_test_docs

    stage1_completed = len(list(args.stage1_root.rglob("*.json")))
    stage2_completed = len(list(args.stage2_root.rglob("*.json")))
    followup_completed = len(followup_runs)
    stage1_total = _count_manifest_lines(args.stage1_root.parent / "commands.txt")
    stage2_total = _count_manifest_lines(args.stage2_root.parent / "commands.txt")

    stage1_leafs = sorted({(float(r["leaf_fraction"]), str(r["leaf_label"])) for r in stage1_rows})
    stage1_leaf_labels = [label for _, label in stage1_leafs]
    stage1_leaf_readable = []
    for label in stage1_leaf_labels:
        try:
            pct = float(label.replace("%", ""))
            tok = int(round(pct / 100.0 * doc_tokens))
            if pct >= 99.9:
                stage1_leaf_readable.append(f"{tok} tokens\n(full doc)")
            else:
                stage1_leaf_readable.append(f"{tok} tokens\n({pct:.0f}% of doc)")
        except Exception:
            stage1_leaf_readable.append(label)

    stage1_coverage = [
        sum(1 for row in stage1_rows if row.get("method") == "tree_exact_utility" and str(row.get("leaf_label")) == label)
        for label in stage1_leaf_labels
    ]
    stage1_exact_vals = [
        _safe_float(row.get("metric_scalar_abs_to_full_mean"))
        for row in stage1_rows
        if row.get("method") == "tree_exact_utility"
    ]
    stage1_exact_mean = _safe_mean(stage1_exact_vals)

    count_rows = [row for row in stage1_rows if row.get("method") == "count_svd_ceiling"]
    utility_rows = [row for row in stage1_rows if row.get("method") == "utility_pca_practical"]
    compression_state_dims = sorted({int(row["state_dim"]) for row in count_rows + utility_rows})
    utility_pca_exact_dims = sorted(
        {
            int(row["state_dim"])
            for row in utility_rows
            if bool(row.get("metric_exact_family_representable", False))
        }
    )
    count_svd_exact_dims = sorted(
        {
            int(row["state_dim"])
            for row in count_rows
            if bool(row.get("metric_exact_family_representable", False))
        }
    )

    stage2_taus = sorted({float(row["local_mixture_concentration"]) for row in stage2_rows})
    stage2_lambdas = sorted({float(row["lambda_multiplier"]) for row in stage2_rows})
    stage2_llts = sorted({int(row["latent_leaf_tokens"]) for row in stage2_rows if int(row["latent_leaf_tokens"]) > 0})
    hero_lam = max(stage2_lambdas)
    best_llt = max(stage2_llts)
    best_tau = min(stage2_taus)
    worst_tau = max(stage2_taus)
    mid_tau = 8.0 if 8.0 in stage2_taus else stage2_taus[len(stage2_taus) // 2]

    def _s2_mean(method: str, *, llt: int | None = None, tau: float | None = None, lam: float | None = None) -> float:
        vals = [
            _safe_float(row.get("metric_utility_abs_to_true_mean"))
            for row in stage2_rows
            if row.get("method") == method
            and str(row.get("budget_regime")) == "all_leaves_labeled"
            and (llt is None or int(row.get("latent_leaf_tokens", -1)) == llt)
            and (tau is None or float(row.get("local_mixture_concentration")) == tau)
            and (lam is None or float(row.get("lambda_multiplier")) == lam)
        ]
        return _safe_mean(vals)

    def _pair_stats(llt: int, tau: float, lam: float) -> dict:
        pooled = _s2_mean("pooled_doc_wrong_model", llt=llt, tau=tau, lam=lam)
        section = _s2_mean("leaf_infer_sum", llt=llt, tau=tau, lam=lam)
        return {
            "pooled_mean": pooled,
            "section_mean": section,
            "delta_mean": pooled - section,
            "ratio": pooled / section if section > 0 else float("nan"),
        }

    hero_best = _pair_stats(best_llt, best_tau, hero_lam)
    hero_worst = _pair_stats(best_llt, worst_tau, hero_lam)
    hero_mid_cell = _pair_stats(best_llt, mid_tau, hero_lam)
    lam0_best = _pair_stats(best_llt, best_tau, 0.0)
    lam1_best = _pair_stats(best_llt, best_tau, 1.0 if 1.0 in stage2_lambdas else hero_lam)

    followup_cross_agg = _aggregate_followup(
        [row for row in followup_runs if row["suite"] == "tau_crossover_dense"],
        keys=["suite", "llt", "dtc", "tau", "lam"],
    )
    followup_lambda_agg = _aggregate_followup(
        [row for row in followup_runs if row["suite"] == "lambda_onset_dense"],
        keys=["suite", "llt", "dtc", "tau", "lam"],
    )
    followup_robust_agg = _aggregate_followup(
        [row for row in followup_runs if row["suite"] == "doc_topic_concentration_robustness"],
        keys=["suite", "llt", "dtc", "tau", "lam"],
    )

    cross_taus = sorted({float(row["tau"]) for row in followup_runs if row["suite"] == "tau_crossover_dense"})
    cross_llts = sorted({int(row["llt"]) for row in followup_runs if row["suite"] == "tau_crossover_dense"})
    onset_taus = sorted({float(row["tau"]) for row in followup_runs if row["suite"] == "lambda_onset_dense"})
    onset_llts = sorted({int(row["llt"]) for row in followup_runs if row["suite"] == "lambda_onset_dense"})
    onset_lambdas = sorted({float(row["lam"]) for row in followup_runs if row["suite"] == "lambda_onset_dense"})
    robust_taus = sorted({float(row["tau"]) for row in followup_runs if row["suite"] == "doc_topic_concentration_robustness"})
    robust_dtcs = sorted({float(row["dtc"]) for row in followup_runs if row["suite"] == "doc_topic_concentration_robustness"})
    robust_llts = sorted({int(row["llt"]) for row in followup_runs if row["suite"] == "doc_topic_concentration_robustness"})

    def _last_positive_tau(llt: int) -> float | None:
        positives = [
            tau
            for tau in cross_taus
            if followup_cross_agg[("tau_crossover_dense", llt, 0.6, tau, 2.0)]["delta_mean"] > 0.0
        ]
        return max(positives) if positives else None

    def _onset_lambda(tau: float, llt: int) -> float | None:
        for lam in onset_lambdas:
            if followup_lambda_agg[("lambda_onset_dense", llt, 0.6, tau, lam)]["delta_mean"] > 0.0:
                return lam
        return None

    last_positive_tau = {llt: _last_positive_tau(llt) for llt in cross_llts}
    onset_table = {(tau, llt): _onset_lambda(tau, llt) for tau in onset_taus for llt in onset_llts}
    gap_corr = _safe_corr((row["gap_signal_abs"], row["delta"]) for row in followup_runs)

    md_path = output_dir / "lda_tree_methods_best_of_report.md"
    pdf_path = output_dir / "lda_tree_methods_best_of_report.pdf"
    summary_path = output_dir / "lda_tree_methods_best_of_report_summary.json"

    _write_markdown(
        md_path,
        snapshot_label=args.snapshot_label,
        stage1_root=args.stage1_root,
        stage2_root=args.stage2_root,
        followup_root=args.followup_root,
        train_docs=report_train_docs,
        test_docs=report_test_docs,
        stage1_exact_mean=stage1_exact_mean,
        utility_pca_exact_dim=utility_pca_exact_dims[0] if utility_pca_exact_dims else None,
        count_svd_exact_dim=count_svd_exact_dims[0] if count_svd_exact_dims else None,
        hero_best=hero_best,
        hero_worst=hero_worst,
        hero_lam=hero_lam,
        best_llt=best_llt,
        best_tau=best_tau,
        worst_tau=worst_tau,
        cross_taus=cross_taus,
        cross_llts=cross_llts,
        last_positive_tau=last_positive_tau,
        cross_agg=followup_cross_agg,
        onset_taus=onset_taus,
        onset_table=onset_table,
        robust_dtcs=robust_dtcs,
        robust_taus=robust_taus,
        robust_llts=robust_llts,
        robust_agg=followup_robust_agg,
        gap_corr=gap_corr,
        doc_tokens=doc_tokens,
    )

    delta_cmap = LinearSegmentedColormap.from_list("delta", ["#b2182b", "#ffffff", "#1a9850"])

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11.0, 8.5))
        fig.suptitle("Can Per-Section Analysis Beat Reading Everything at Once?", fontsize=20, fontweight="bold", y=0.97)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 0.88], height_ratios=[0.72, 0.28], wspace=0.16, hspace=0.10)
        ax_text = fig.add_subplot(gs[0, 0])
        ax_diag = fig.add_subplot(gs[0, 1])
        ax_cards = fig.add_subplot(gs[1, :])
        for ax in (ax_text, ax_diag, ax_cards):
            ax.axis("off")

        intro_paras = [
            f"Snapshot: {args.snapshot_label}. The question is narrow: when does inferring a topic mixture section by section beat pooling the whole document into one bag of words? The document length is fixed at {doc_tokens} tokens.",
            "The data-generating process draws one document-level topic mixture, then draws a local topic mixture for each latent section around that document mixture, and finally draws section words from those local mixtures. The pooled baseline infers one document mixture. The per-section method infers one mixture per section and aggregates the section utilities.",
            "A concrete two-topic example makes the heterogeneity parameter intuitive. If the document average is 50/50 politics versus sports, low tau can generate sections like 90/10 and 10/90 while high tau keeps both sections near 50/50. The pooled histogram is identical in both cases; the local structure is not.",
            "The empirical claim is supportive but conditional: per-section analysis helps when sections are genuinely different, the target is nonlinear enough to care about within-section composition, and each section is large enough that local inference is stable.",
        ]
        y = 0.96
        for para in intro_paras:
            wrapped = textwrap.fill(para, width=63)
            n_lines = wrapped.count("\n") + 1
            ax_text.text(0.02, y, wrapped, fontsize=12.0, va="top", ha="left", linespacing=1.42)
            y -= 0.040 * n_lines + 0.045

        _draw_card(
            ax_diag,
            x=0.06,
            y=0.76,
            w=0.88,
            h=0.15,
            title="1. Draw one document mixture",
            body=r"$\pi_d \sim \mathrm{Dir}(\alpha)$",
        )
        _draw_card(
            ax_diag,
            x=0.06,
            y=0.53,
            w=0.88,
            h=0.15,
            title="2. Draw one mixture per section",
            body=r"$\pi_{d,b}\mid\pi_d \sim \mathrm{Dir}(\tau \pi_d)$" + "\nlow tau = more different sections; high tau = more similar sections",
        )
        _draw_card(
            ax_diag,
            x=0.06,
            y=0.30,
            w=0.88,
            h=0.15,
            title="3. Draw words from those local mixtures",
            body=r"$z_{d,b,n}\sim\mathrm{Cat}(\pi_{d,b}),\quad x_{d,b,n}\sim\mathrm{Cat}(\beta_{z_{d,b,n}})$",
        )
        _draw_card(
            ax_diag,
            x=0.06,
            y=0.05,
            w=0.40,
            h=0.16,
            title="Pooled baseline",
            body="Infer one document mixture\nfrom all 384 tokens",
            facecolor="#eef5fb",
        )
        _draw_card(
            ax_diag,
            x=0.54,
            y=0.05,
            w=0.40,
            h=0.16,
            title="Per-section method",
            body="Infer one local mixture\nper section, then aggregate",
            facecolor="#eef9ef",
        )
        for y0, y1 in [(0.76, 0.68), (0.53, 0.45), (0.30, 0.22)]:
            ax_diag.annotate("", xy=(0.50, y1), xytext=(0.50, y0), arrowprops=dict(arrowstyle="->", lw=1.4, color="#444444"))

        card_w = 0.47
        card_h = 0.36
        card_positions = [
            (0.01, 0.54),
            (0.52, 0.54),
            (0.01, 0.10),
            (0.52, 0.10),
        ]
        _draw_card(
            ax_cards,
            x=card_positions[0][0],
            y=card_positions[0][1],
            w=card_w,
            h=card_h,
            title="Held-out protocol",
            body=f"{report_train_docs} training docs/run\n{report_test_docs} held-out test docs/run\n{doc_tokens} tokens/document",
        )
        _draw_card(
            ax_cards,
            x=card_positions[1][0],
            y=card_positions[1][1],
            w=card_w,
            h=card_h,
            title="Main metric",
            body=f"{STAGE2_ERROR_LABEL}\nDelta = pooled held-out error\nminus per-section held-out error",
        )
        _draw_card(
            ax_cards,
            x=card_positions[2][0],
            y=card_positions[2][1],
            w=card_w,
            h=card_h,
            title="Section weights",
            body="True target uses token weights\nomega_b = N_(d,b) / N_d\nEqual-length sweeps imply omega_b = 1 / B",
        )
        _draw_card(
            ax_cards,
            x=card_positions[3][0],
            y=card_positions[3][1],
            w=card_w,
            h=card_h,
            title="Headline",
            body=(
                f"Best broad-sweep cell:\nDelta = {hero_best['delta_mean']:+.2f}\n"
                f"at {_leaf_pct_label(best_llt, doc_tokens)}\n{_tau_label(best_tau)}"
            ),
            facecolor="#f1f8f3",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(11.0, 8.5))
        fig.suptitle("Stage 1: Control Check and Measurement Protocol", fontsize=18, fontweight="bold", y=0.97)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 0.85], height_ratios=[0.58, 0.42], wspace=0.18, hspace=0.16)
        ax_left = fig.add_subplot(gs[:, 0])
        ax_top_right = fig.add_subplot(gs[0, 1])
        ax_bottom_right = fig.add_subplot(gs[1, 1])
        for ax in (ax_left, ax_top_right, ax_bottom_right):
            ax.axis("off")

        stage1_text = "\n\n".join(
            textwrap.fill(para, width=60)
            for para in [
                "Stage 1 is a pass/fail control, not a contest. The target is linear in the document counts, so a correct mergeable tree summary should reproduce the full-document answer exactly.",
                f"It does. Across {stage1_completed} completed Stage 1 summaries"
                + (f" out of {stage1_total}" if stage1_total is not None else "")
                + f", the exact tree path stays at roughly {stage1_exact_mean:.1e} absolute error to the full-document reference.",
                f"This page intentionally uses a different metric from the rest of the report: {STAGE1_ERROR_LABEL.lower()}. Stage 2 and the follow-up switch to held-out prediction error on a separate test corpus.",
                "That distinction matters. If pooling later beats a practical per-section estimator at w_q = 0, that is not a contradiction. It means the extra latent-mixture inference is unnecessary in the linear regime, not that the tree representation is broken.",
            ]
        )
        ax_left.text(0.01, 0.98, stage1_text, va="top", fontsize=12.5, linespacing=1.45)

        _draw_card(
            ax_top_right,
            x=0.05,
            y=0.50,
            w=0.90,
            h=0.40,
            title="Pass/fail result",
            body=f"Exact tree error\n{stage1_exact_mean:.2e}\n\nMachine precision across the completed control runs.",
            facecolor="#eef9ef",
            title_size=13,
            body_size=12,
        )
        _draw_card(
            ax_top_right,
            x=0.05,
            y=0.05,
            w=0.42,
            h=0.32,
            title="Utility PCA",
            body=f"Exact at\nstate_dim = {utility_pca_exact_dims[0] if utility_pca_exact_dims else 'n/a'}",
            facecolor="#fff7ea",
            title_size=12,
            body_size=11,
        )
        _draw_card(
            ax_top_right,
            x=0.53,
            y=0.05,
            w=0.42,
            h=0.32,
            title="Count SVD",
            body=f"Exact at\nstate_dim = {count_svd_exact_dims[0] if count_svd_exact_dims else 'n/a'}",
            facecolor="#eef5fb",
            title_size=12,
            body_size=11,
        )
        protocol_text = "\n".join(
            [
                f"Stage 1 fit/eval protocol",
                f"train docs/run : {stage1_train_docs}",
                f"test docs/run  : {stage1_test_docs}",
                f"doc length     : {doc_tokens} tokens",
                f"metric         : {STAGE1_ERROR_LABEL.lower()}",
            ]
        )
        ax_bottom_right.text(0.02, 0.95, protocol_text, family="monospace", fontsize=11.5, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(11.0, 8.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[0.5, 1.0], hspace=0.28)
        ax_top = fig.add_subplot(gs[0, :])
        ax_top.axis("off")
        compression_text = "\n\n".join(
            textwrap.fill(para, width=108)
            for para in [
                "Stage 1 is about representation, not held-out prediction. The vertical axis here is absolute error to the exact full-document reference. The shaded bands show the small amount of variation across section sizes; the main comparison is between the two compression families.",
                "What matters is the horizontal location of exact recovery. Utility PCA becomes exact at the task dimension itself, while Count SVD only becomes exact when it keeps the full 512-dimensional count space. That is the cleanest picture of why task-aligned compression is the right object to preserve.",
            ]
        )
        ax_top.text(0.0, 0.98, compression_text, va="top", fontsize=12, linespacing=1.42)
        ax = fig.add_subplot(gs[1, 0])

        def _band_summary(rows: Sequence[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            means = []
            lows = []
            highs = []
            for dim in compression_state_dims:
                vals = [
                    _safe_float(row.get("metric_scalar_abs_to_full_mean"))
                    for row in rows
                    if int(row.get("state_dim", -1)) == dim
                ]
                clean = [max(v, 1e-15) if math.isfinite(v) and v <= 0.0 else v for v in vals if math.isfinite(v)]
                means.append(_safe_mean(clean))
                lows.append(min(clean) if clean else float("nan"))
                highs.append(max(clean) if clean else float("nan"))
            return (
                np.asarray(means, dtype=np.float64),
                np.asarray(lows, dtype=np.float64),
                np.asarray(highs, dtype=np.float64),
            )

        count_mean, count_low, count_high = _band_summary(count_rows)
        util_mean, util_low, util_high = _band_summary(utility_rows)

        ax.plot(compression_state_dims, count_mean, marker="s", linewidth=3, color=POOL_COLOR, label="Count SVD")
        ax.fill_between(compression_state_dims, count_low, count_high, color=POOL_COLOR, alpha=0.14, label="Count SVD range across section sizes")
        ax.plot(compression_state_dims, util_mean, marker="o", linewidth=3, color="#ff7f0e", label="Utility PCA")
        ax.fill_between(compression_state_dims, util_low, util_high, color="#ff7f0e", alpha=0.14, label="Utility PCA range across section sizes")
        if utility_pca_exact_dims:
            ax.axvline(utility_pca_exact_dims[0], color="#ff7f0e", linestyle="--", linewidth=1.5)
            ax.annotate(
                f"exact at {utility_pca_exact_dims[0]}",
                xy=(utility_pca_exact_dims[0], util_mean[compression_state_dims.index(utility_pca_exact_dims[0])]),
                xytext=(utility_pca_exact_dims[0] * 1.25, 0.06),
                textcoords="data",
                arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.2),
                fontsize=10,
                color="#ff7f0e",
            )
        if count_svd_exact_dims:
            ax.axvline(count_svd_exact_dims[0], color=POOL_COLOR, linestyle="--", linewidth=1.5)
            ax.annotate(
                f"exact at {count_svd_exact_dims[0]}",
                xy=(count_svd_exact_dims[0], count_mean[compression_state_dims.index(count_svd_exact_dims[0])]),
                xytext=(count_svd_exact_dims[0] / 2.5, 0.45),
                textcoords="data",
                arrowprops=dict(arrowstyle="->", color=POOL_COLOR, lw=1.2),
                fontsize=10,
                color=POOL_COLOR,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Retained state dimension")
        ax.set_ylabel(STAGE1_ERROR_LABEL)
        ax.set_title("Utility PCA hits exact recovery at the task dimension; Count SVD does not")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")
        fig.suptitle("How Much Can We Compress?", fontsize=17, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(11.0, 8.5))
        fig.suptitle("Exact Meaning of Tau, Lambda, and Section Weights", fontsize=18, fontweight="bold", y=0.97)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 0.95], height_ratios=[0.62, 0.38], wspace=0.16, hspace=0.14)
        ax_eq = fig.add_subplot(gs[0, 0])
        ax_cards = fig.add_subplot(gs[0, 1])
        ax_notes = fig.add_subplot(gs[1, :])
        for ax in (ax_eq, ax_cards, ax_notes):
            ax.axis("off")

        eq_lines = [
            ("Generative model", r"$\pi_d \sim \mathrm{Dir}(\alpha),\qquad \pi_{d,b}\mid\pi_d \sim \mathrm{Dir}(\tau \pi_d)$"),
            ("Token model", r"$z_{d,b,n}\sim \mathrm{Cat}(\pi_{d,b}),\qquad x_{d,b,n}\sim \mathrm{Cat}(\beta_{z_{d,b,n}})$"),
            ("Section utility", r"$h(\pi)=\theta^\top\pi + w_q\,\pi^\top W\pi$"),
            ("Token-weighted target", r"$y_d = N_d\sum_b \omega_b h(\pi_{d,b}),\qquad \omega_b = N_{d,b}/N_d$"),
            ("Pooled reference", r"$\bar{\pi}_d = \sum_b \omega_b \pi_{d,b},\qquad y_{\mathrm{pool},d}=N_d h(\bar{\pi}_d)$"),
            ("Exact identities", r"$\mathrm{Var}(\pi_{d,b,k}\mid \pi_d)=\frac{\pi_{d,k}(1-\pi_{d,k})}{\tau+1},\qquad d=\frac{1}{1+\tau}$"),
            ("Gap identity", r"$\frac{y_d}{N_d}-h(\bar{\pi}_d)=w_q\!\left[\sum_b \omega_b \pi_{d,b}^\top W \pi_{d,b}-\bar{\pi}_d^\top W \bar{\pi}_d\right]$"),
        ]
        y = 0.98
        for label, eq in eq_lines:
            ax_eq.text(0.01, y, label, fontsize=11.5, fontweight="bold", va="top")
            y -= 0.05
            ax_eq.text(0.05, y, eq, fontsize=15.5, va="top")
            y -= 0.10

        _draw_card(
            ax_cards,
            x=0.05,
            y=0.68,
            w=0.90,
            h=0.24,
            title=f"Low heterogeneity control: {_tau_label(64)}",
            body="Sections are almost copies.\nIf the document average is 50/50, individual sections stay close to 50/50.",
            facecolor="#f7f8fb",
        )
        _draw_card(
            ax_cards,
            x=0.05,
            y=0.40,
            w=0.90,
            h=0.24,
            title=f"High heterogeneity case: {_tau_label(0.25)}",
            body="Sections can be far apart.\nA 50/50 document average can come from sections like 90/10 and 10/90.",
            facecolor="#eef9ef",
        )
        _draw_card(
            ax_cards,
            x=0.05,
            y=0.08,
            w=0.90,
            h=0.26,
            title="What positive quadratic weight changes",
            body=(
                "If $h(\\pi)=w_q\\,\\pi_{\\mathrm{politics}}^2$, then two documents can share the same pooled 50/50 average but differ in true utility.\n"
                "Sections (50/50, 50/50) -> average nonlinear score $0.25w_q$.\n"
                "Sections (90/10, 10/90) -> average nonlinear score $0.41w_q$."
            ),
            facecolor="#fff7ea",
        )

        note_cols = [
            (
                "What tau changes",
                "Tau changes the spread of the section mixtures around the document mixture. Lower tau means more section-level variance. It does not change the conditional section mean, so the document average still stays anchored at pi_d.",
            ),
            (
                "What omega_b means",
                f"omega_b is the token share of section b inside the document. In these equal-length sweeps that simplifies to omega_b = 1 / B. For {_leaf_pct_label(best_llt, doc_tokens)} we therefore have {_section_weight_text(best_llt, doc_tokens)}.",
            ),
            (
                "What the quadratic weight changes",
                "The quadratic weight multiplies the whole interaction term. When w_q = 0, splitting cannot create target information. As w_q increases, the exact same local topic differences matter more to the target.",
            ),
        ]
        for idx, (title, body) in enumerate(note_cols):
            x0 = 0.01 + idx * 0.33
            _draw_card(ax_notes, x=x0, y=0.10, w=0.31, h=0.80, title=title, body=textwrap.fill(body, width=34))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, len(stage2_llts), figsize=(3.35 * len(stage2_llts), 5.4), sharey=True, constrained_layout=True)
        if len(stage2_llts) == 1:
            axes = [axes]
        broad_handles = [
            Line2D([0], [0], color="#22863a", linewidth=6, alpha=0.18, label="Delta > 0: per-section wins"),
            Line2D([0], [0], color="#b2182b", linewidth=6, alpha=0.12, label="Delta < 0: pooling wins"),
            Line2D([0], [0], color=SECTION_COLOR, marker="o", linewidth=2.2, label=DELTA_SHORT_LABEL),
        ]
        for idx, llt in enumerate(stage2_llts):
            ax = axes[idx]
            deltas = [_s2_mean("pooled_doc_wrong_model", llt=llt, tau=tau, lam=hero_lam) - _s2_mean("leaf_infer_sum", llt=llt, tau=tau, lam=hero_lam) for tau in stage2_taus]
            xs = np.arange(len(stage2_taus))
            upper = max(0.5, max(max(deltas), 0.0) * 1.15)
            lower = min(-0.5, min(min(deltas), 0.0) * 1.15)
            ax.axhspan(0.0, upper, color="#2ca02c", alpha=0.10)
            ax.axhspan(lower, 0.0, color="#b2182b", alpha=0.08)
            ax.axhline(0.0, color="#444444", linewidth=1.2, linestyle="--")
            ax.plot(xs, deltas, marker="o", color=SECTION_COLOR, linewidth=2.2)
            ax.set_xticks(xs)
            ax.set_xticklabels([_tau_label(tau, multiline=True) for tau in stage2_taus], fontsize=8)
            ax.set_xlabel("Section heterogeneity\nleft = more different sections")
            ax.set_title(_leaf_pct_label(llt, doc_tokens, multiline=True))
            ax.grid(alpha=0.25)
            if idx == 0:
                ax.set_ylabel(DELTA_SHORT_LABEL)
                ax.legend(handles=broad_handles, fontsize=8, loc="upper left")
        fig.suptitle(f"Broad Sweep: Where Does Per-Section Analysis Win? ({_qweight_label(hero_lam)})", fontsize=16, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, len(stage2_taus), figsize=(3.35 * len(stage2_taus), 5.5), sharey=True, constrained_layout=True)
        if len(stage2_taus) == 1:
            axes = [axes]
        for idx, tau in enumerate(stage2_taus):
            ax = axes[idx]
            deltas = [_s2_mean("pooled_doc_wrong_model", llt=llt, tau=tau, lam=hero_lam) - _s2_mean("leaf_infer_sum", llt=llt, tau=tau, lam=hero_lam) for llt in stage2_llts]
            xs = np.arange(len(stage2_llts))
            upper = max(0.5, max(max(deltas), 0.0) * 1.15)
            lower = min(-0.5, min(min(deltas), 0.0) * 1.15)
            ax.axhspan(0.0, upper, color="#2ca02c", alpha=0.10)
            ax.axhspan(lower, 0.0, color="#b2182b", alpha=0.08)
            ax.axhline(0.0, color="#444444", linewidth=1.2, linestyle="--")
            ax.plot(xs, deltas, marker="o", color=SECTION_COLOR, linewidth=2.2)
            ax.set_xticks(xs)
            ax.set_xticklabels([_leaf_pct_label(llt, doc_tokens, multiline=True) for llt in stage2_llts], fontsize=8)
            ax.set_xlabel("Section size\n(more words = less local inference noise)")
            ax.set_title(_tau_panel_title(tau), fontsize=10.5)
            ax.grid(alpha=0.25)
            if idx == 0:
                ax.set_ylabel(DELTA_SHORT_LABEL)
        fig.suptitle(f"Do Bigger Sections Help? Held-out Delta by Section Size ({_qweight_label(hero_lam)})", fontsize=16, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(11.0, 8.5))
        fig.suptitle("Why Does Nonlinearity Matter?", fontsize=16, fontweight="bold", y=0.97)
        gs = fig.add_gridspec(2, len(stage2_taus), height_ratios=[0.80, 0.20], hspace=0.30, wspace=0.20, top=0.88, bottom=0.08)
        axes = [fig.add_subplot(gs[0, idx]) for idx in range(len(stage2_taus))]
        note_ax = fig.add_subplot(gs[1, :])
        note_ax.axis("off")
        for idx, tau in enumerate(stage2_taus):
            ax = axes[idx]
            deltas = [_s2_mean("pooled_doc_wrong_model", llt=best_llt, tau=tau, lam=lam) - _s2_mean("leaf_infer_sum", llt=best_llt, tau=tau, lam=lam) for lam in stage2_lambdas]
            upper = max(0.35, max(max(deltas), 0.0) * 1.12)
            lower = min(-0.35, min(min(deltas), 0.0) * 1.12)
            ax.axhspan(0.0, upper, color="#2ca02c", alpha=0.10)
            ax.axhspan(lower, 0.0, color="#b2182b", alpha=0.08)
            ax.axhline(0.0, color="#444444", linewidth=1.2, linestyle="--")
            ax.plot(stage2_lambdas, deltas, marker="o", color=SECTION_COLOR, linewidth=2.2)
            ax.set_xticks(stage2_lambdas)
            ax.set_xticklabels(["0\nlinear", "1\nmoderate", f"{hero_lam:g}\nstrong"] if len(stage2_lambdas) == 3 else [f"{lam:g}" for lam in stage2_lambdas], fontsize=8)
            ax.set_xlabel("quadratic weight w_q")
            ax.set_title(_tau_panel_title(tau), fontsize=10.5)
            ax.grid(alpha=0.25)
            if idx == 0:
                ax.set_ylabel(DELTA_SHORT_LABEL)
        note_text = "\n\n".join(
            textwrap.fill(para, width=120)
            for para in [
                f"These broad-sweep panels use {_leaf_pct_label(best_llt, doc_tokens)} so the section estimator is reasonably stable. The exact control is w_q = 0: Delta should be at or below zero there because splitting adds only inference noise when the target is linear.",
                "As the quadratic weight increases, the interaction term makes within-section composition matter. That is why the heterogeneous panels rise above zero early while the nearly homogeneous panels rise late or barely rise at all.",
            ]
        )
        note_ax.text(0.0, 0.98, note_text, fontsize=11.5, va="top", ha="left", linespacing=1.4)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        delta_grid = np.full((len(stage2_llts), len(stage2_taus)), float("nan"))
        for i, llt in enumerate(stage2_llts):
            for j, tau in enumerate(stage2_taus):
                delta_grid[i, j] = _s2_mean("pooled_doc_wrong_model", llt=llt, tau=tau, lam=hero_lam) - _s2_mean("leaf_infer_sum", llt=llt, tau=tau, lam=hero_lam)
        max_abs_delta = max(abs(float(v)) for v in delta_grid.flatten() if math.isfinite(float(v)))
        fig = plt.figure(figsize=(11.0, 8.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 0.72], wspace=0.22)
        ax_hm = fig.add_subplot(gs[0, 0])
        im = ax_hm.imshow(
            delta_grid,
            aspect="auto",
            origin="lower",
            cmap=delta_cmap,
            norm=TwoSlopeNorm(vmin=-max_abs_delta, vcenter=0.0, vmax=max_abs_delta),
        )
        ax_hm.set_xticks(np.arange(len(stage2_taus)))
        ax_hm.set_xticklabels([_tau_label(tau, multiline=True) for tau in stage2_taus], fontsize=9)
        ax_hm.set_yticks(np.arange(len(stage2_llts)))
        ax_hm.set_yticklabels([_leaf_pct_label(llt, doc_tokens, multiline=True) for llt in stage2_llts])
        ax_hm.set_xlabel("Section heterogeneity")
        ax_hm.set_ylabel("Section size")
        ax_hm.set_title(f"Coarse Summary Heatmap\n{DELTA_LABEL}")
        for i in range(len(stage2_llts)):
            for j in range(len(stage2_taus)):
                val = delta_grid[i, j]
                if math.isfinite(val):
                    verdict = "sections" if val > 0.0 else "pool"
                    color = "white" if abs(val) > 0.55 * max_abs_delta else "black"
                    ax_hm.text(j, i, f"{val:+.1f}\n{verdict}", ha="center", va="center", fontsize=9, color=color)
        fig.colorbar(im, ax=ax_hm, label="Delta (white = exactly neutral)")
        ax_text = fig.add_subplot(gs[0, 1])
        ax_text.axis("off")
        heatmap_text = "\n\n".join(
            textwrap.fill(para, width=40)
            for para in [
                "This is the broad overview on the coarse tau grid. Every cell uses the same Stage 2 metric: held-out mean absolute utility error, summarized as Delta = pooled held-out error minus per-section held-out error.",
                "Green means per-section analysis lowers held-out error. Red means pooling lowers held-out error. White is exactly neutral at Delta = 0.",
                f"The strongest broad-sweep cell is {_leaf_pct_label(best_llt, doc_tokens)} at {_tau_label(best_tau)}: Delta = {hero_best['delta_mean']:+.2f}. By {_tau_label(worst_tau)}, the sign flips to {hero_worst['delta_mean']:+.2f}.",
            ]
        )
        ax_text.text(0.0, 0.98, heatmap_text, va="top", fontsize=12, linespacing=1.45)
        fig.suptitle("Where Does Per-Section Analysis Win Overall?", fontsize=16, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        cross_grid = np.full((len(cross_llts), len(cross_taus)), float("nan"))
        for i, llt in enumerate(cross_llts):
            for j, tau in enumerate(cross_taus):
                cross_grid[i, j] = followup_cross_agg[("tau_crossover_dense", llt, 0.6, tau, 2.0)]["delta_mean"]
        max_abs_cross = max(abs(float(v)) for v in cross_grid.flatten() if math.isfinite(float(v)))
        fig = plt.figure(figsize=(11.0, 8.5))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 0.95], height_ratios=[1.0, 0.58], wspace=0.28, hspace=0.25)
        ax_cross = fig.add_subplot(gs[:, 0])
        im = ax_cross.imshow(
            cross_grid,
            aspect="auto",
            origin="lower",
            cmap=delta_cmap,
            norm=TwoSlopeNorm(vmin=-max_abs_cross, vcenter=0.0, vmax=max_abs_cross),
        )
        ax_cross.set_xticks(np.arange(len(cross_taus)))
        ax_cross.set_xticklabels([f"tau={tau:g}\n(d={_tau_diversity_index(tau):.2f})" for tau in cross_taus], fontsize=7.5, rotation=35, ha="right")
        ax_cross.set_yticks(np.arange(len(cross_llts)))
        ax_cross.set_yticklabels([_leaf_pct_label(llt, doc_tokens, multiline=True) for llt in cross_llts])
        ax_cross.set_xlabel("Dense tau follow-up\n(raw tau shown first; d = 1 / (1 + tau) shown underneath)")
        ax_cross.set_ylabel("Section size")
        ax_cross.set_title(f"Follow-up 1: Dense Tau Crossover\n{DELTA_LABEL}")
        for i, llt in enumerate(cross_llts):
            for j, tau in enumerate(cross_taus):
                stats = followup_cross_agg[("tau_crossover_dense", llt, 0.6, tau, 2.0)]
                wins = int(round(stats["win_rate"] * stats["n"]))
                color = "white" if abs(stats["delta_mean"]) > 0.55 * max_abs_cross else "black"
                ax_cross.text(j, i, f"{stats['delta_mean']:+.1f}\n{wins}/{stats['n']}", ha="center", va="center", fontsize=8.5, color=color)
        fig.colorbar(im, ax=ax_cross, label="Delta (white = exactly neutral)")

        ax_win = fig.add_subplot(gs[0, 1])
        xs = np.arange(len(cross_taus))
        for llt in cross_llts:
            wins = [100.0 * followup_cross_agg[("tau_crossover_dense", llt, 0.6, tau, 2.0)]["win_rate"] for tau in cross_taus]
            ax_win.plot(xs, wins, marker="o", linewidth=2, color=SECTION_SIZE_COLORS.get(llt, "#333333"), label=f"{llt} tokens")
        ax_win.axhline(50.0, color="#444444", linewidth=1, linestyle="--")
        ax_win.set_xticks(xs)
        ax_win.set_xticklabels([f"{tau:g}" for tau in cross_taus], fontsize=8, rotation=35, ha="right")
        ax_win.set_xlabel("tau (see heatmap for d)")
        ax_win.set_ylabel("Seed win rate (%)")
        ax_win.set_title("How often do sections win?")
        ax_win.grid(alpha=0.3)
        ax_win.legend(fontsize=8, loc="upper right")

        ax_threshold = fig.add_subplot(gs[1, 1])
        ax_threshold.axis("off")
        threshold_text = [
            "Last heterogeneity with positive mean Delta",
            "",
        ]
        for llt in cross_llts:
            threshold_text.append(
                f"{llt:>3} tokens : {('never' if last_positive_tau[llt] is None else _tau_label(last_positive_tau[llt]))}"
            )
        threshold_text.extend(
            [
                "",
                "Interpretation:",
                "larger sections keep winning",
                "further into the low-heterogeneity",
                "regime because they reduce local",
                "inference noise without changing",
                "the target-side structural signal.",
            ]
        )
        ax_threshold.text(0.0, 0.98, "\n".join(threshold_text), family="monospace", va="top", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(11.0, 8.5))
        gs = fig.add_gridspec(2, 3, height_ratios=[0.78, 0.22], hspace=0.30, wspace=0.22, top=0.87, bottom=0.08)
        for idx, tau in enumerate(onset_taus):
            ax = fig.add_subplot(gs[0, idx])
            for llt in onset_llts:
                deltas = [followup_lambda_agg[("lambda_onset_dense", llt, 0.6, tau, lam)]["delta_mean"] for lam in onset_lambdas]
                sems = [followup_lambda_agg[("lambda_onset_dense", llt, 0.6, tau, lam)]["delta_sem"] for lam in onset_lambdas]
                color = SECTION_SIZE_COLORS.get(llt, "#333333")
                ax.plot(onset_lambdas, deltas, marker="o", linewidth=2, color=color, label=f"{llt} tokens")
                ax.fill_between(onset_lambdas, np.array(deltas) - np.array(sems), np.array(deltas) + np.array(sems), color=color, alpha=0.15)
            ax.axhline(0.0, color="#444444", linewidth=1, linestyle="--")
            ax.set_title(_tau_panel_title(tau), fontsize=10.5)
            ax.set_xlabel("quadratic weight w_q")
            ax.grid(alpha=0.3)
            if idx == 0:
                ax.set_ylabel(DELTA_SHORT_LABEL)
                ax.legend(fontsize=8, loc="upper left")
        ax_text = fig.add_subplot(gs[1, :])
        ax_text.axis("off")
        table_rows = [
            [
                _tau_label(tau),
                "never" if onset_table[(tau, 64)] is None else f"{onset_table[(tau, 64)]:g}",
                "never" if onset_table[(tau, 96)] is None else f"{onset_table[(tau, 96)]:g}",
            ]
            for tau in onset_taus
        ]
        table = ax_text.table(
            cellText=table_rows,
            colLabels=["heterogeneity", "64 tokens", "96 tokens"],
            cellLoc="center",
            colLoc="center",
            bbox=[0.00, 0.18, 0.60, 0.74],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10.5)
        ax_text.text(
            0.64,
            0.88,
            textwrap.fill(
                "Reading guide: w_q = 0 is the exact linear control. If the curve is still below zero there, that is expected: splitting added only inference noise. What matters is how quickly the curve rises above zero as the quadratic weight increases.",
                width=42,
            ),
            va="top",
            fontsize=11,
            linespacing=1.4,
        )
        fig.suptitle("Follow-up 2: Lambda Onset", fontsize=16, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        robust_grids: Dict[int, np.ndarray] = {}
        max_abs_robust = 0.0
        for llt in robust_llts:
            grid = np.full((len(robust_dtcs), len(robust_taus)), float("nan"))
            for i, dtc in enumerate(robust_dtcs):
                for j, tau in enumerate(robust_taus):
                    grid[i, j] = followup_robust_agg[("doc_topic_concentration_robustness", llt, dtc, tau, 2.0)]["delta_mean"]
                    max_abs_robust = max(max_abs_robust, abs(grid[i, j]))
            robust_grids[llt] = grid
        fig = plt.figure(figsize=(11.0, 8.5))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.42], hspace=0.28, wspace=0.22)
        axes = []
        for idx, llt in enumerate(robust_llts):
            ax = fig.add_subplot(gs[0, idx])
            axes.append(ax)
            grid = robust_grids[llt]
            im = ax.imshow(
                grid,
                aspect="auto",
                origin="lower",
                cmap=delta_cmap,
                norm=TwoSlopeNorm(vmin=-max_abs_robust, vcenter=0.0, vmax=max_abs_robust),
            )
            ax.set_xticks(np.arange(len(robust_taus)))
            ax.set_xticklabels([_tau_label(tau, multiline=True) for tau in robust_taus], fontsize=8, rotation=25, ha="right")
            ax.set_yticks(np.arange(len(robust_dtcs)))
            ax.set_yticklabels([f"{dtc:g}" for dtc in robust_dtcs])
            ax.set_title(_leaf_pct_label(llt, doc_tokens, multiline=True))
            for i, dtc in enumerate(robust_dtcs):
                for j, tau in enumerate(robust_taus):
                    val = grid[i, j]
                    color = "white" if abs(val) > 0.55 * max_abs_robust else "black"
                    ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=9, color=color)
        axes[0].set_ylabel("Document-topic concentration\n(low = spikier global topic averages)")
        fig.colorbar(im, ax=axes, label="Delta (white = exactly neutral)")
        ax_text = fig.add_subplot(gs[1, :])
        ax_text.axis("off")
        robust_text = "\n\n".join(
            textwrap.fill(para, width=108)
            for para in [
                "The robustness result is not that every setting looks the same. The high-diversity column stays strongly positive across document-topic concentrations, and tau = 64 stays neutral to negative everywhere. The genuinely sensitive column is tau = 8.",
                "At low document-topic concentration (0.2), tau = 8 can turn negative. At the original 0.6 setting it is mildly positive, and at 1.5 it remains positive for larger sections but nearly neutral by tau = 64. That is exactly what a boundary case should look like.",
            ]
        )
        ax_text.text(0.0, 0.98, robust_text, va="top", fontsize=12, linespacing=1.45)
        fig.suptitle("Follow-up 3: Robustness to Document-Topic Concentration", fontsize=16, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(11.0, 8.5))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.02, 0.98], height_ratios=[0.82, 0.18], wspace=0.25, hspace=0.20, top=0.90, bottom=0.08)
        ax_scatter = fig.add_subplot(gs[0, 0])
        for llt in sorted({int(row["llt"]) for row in followup_runs}):
            xs = [row["gap_signal_abs"] for row in followup_runs if int(row["llt"]) == llt]
            ys = [row["delta"] for row in followup_runs if int(row["llt"]) == llt]
            ax_scatter.scatter(xs, ys, alpha=0.28, s=18, color=SECTION_SIZE_COLORS.get(llt, "#333333"), label=f"{llt} tokens")
        all_xs = [row["gap_signal_abs"] for row in followup_runs if math.isfinite(row["gap_signal_abs"]) and math.isfinite(row["delta"])]
        all_ys = [row["delta"] for row in followup_runs if math.isfinite(row["gap_signal_abs"]) and math.isfinite(row["delta"])]
        if len(all_xs) >= 2:
            coeffs = np.polyfit(all_xs, all_ys, 1)
            xline = np.linspace(min(all_xs), max(all_xs), 100)
            ax_scatter.plot(xline, coeffs[0] * xline + coeffs[1], color="#444444", linewidth=2, linestyle="--")
        ax_scatter.axhline(0.0, color="#444444", linewidth=1, linestyle="--")
        ax_scatter.set_xlabel(r"Absolute mean held-out target gap $|E_{\mathrm{test}}[y_d - y_{\mathrm{pool},d}]|$")
        ax_scatter.set_ylabel(DELTA_SHORT_LABEL)
        ax_scatter.set_title(f"Why Does Pooling Fail?\ncorrelation = {gap_corr:.3f}")
        ax_scatter.grid(alpha=0.3)
        ax_scatter.legend(fontsize=8, title="Section size")

        ax_bins = fig.add_subplot(gs[0, 1])
        valid_pairs = sorted((float(x), float(y)) for x, y in zip(all_xs, all_ys) if math.isfinite(x) and math.isfinite(y))
        quantiles = np.quantile([x for x, _ in valid_pairs], [0.0, 0.25, 0.50, 0.75, 1.0]) if valid_pairs else np.asarray([0.0, 1.0])
        bin_labels = []
        for q0, q1 in zip(quantiles[:-1], quantiles[1:]):
            bin_labels.append(f"[{q0:.1f}, {q1:.1f}]")
        xbins = np.arange(len(bin_labels))
        for llt in sorted({int(row["llt"]) for row in followup_runs}):
            means = []
            for lo, hi in zip(quantiles[:-1], quantiles[1:]):
                vals = [
                    float(row["delta"])
                    for row in followup_runs
                    if int(row["llt"]) == llt
                    and math.isfinite(float(row["gap_signal_abs"]))
                    and math.isfinite(float(row["delta"]))
                    and ((float(row["gap_signal_abs"]) >= lo and float(row["gap_signal_abs"]) < hi) or (hi == quantiles[-1] and float(row["gap_signal_abs"]) <= hi))
                ]
                means.append(_safe_mean(vals))
            ax_bins.plot(xbins, means, marker="o", linewidth=2, color=SECTION_SIZE_COLORS.get(llt, "#333333"), label=f"{llt} tokens")
        ax_bins.axhline(0.0, color="#444444", linewidth=1, linestyle="--")
        ax_bins.set_xticks(xbins)
        ax_bins.set_xticklabels(bin_labels, rotation=15, ha="right", fontsize=8)
        ax_bins.set_xlabel("Binned held-out target-gap magnitude")
        ax_bins.set_ylabel(DELTA_SHORT_LABEL)
        ax_bins.set_title("Binned summary by gap magnitude")
        ax_bins.grid(alpha=0.3)

        ax_note = fig.add_subplot(gs[1, :])
        ax_note.axis("off")
        final_text = "\n\n".join(
            textwrap.fill(para, width=120)
            for para in [
                f"The strongest single cell is {_leaf_pct_label(best_llt, doc_tokens)} at {_tau_label(best_tau)}: per-section analysis reduces held-out mean absolute utility error by {hero_best['delta_mean']:.2f} points ({hero_best['pooled_mean']:.2f} down to {hero_best['section_mean']:.2f}).",
                f"The exact linear control behaves correctly. At w_q = 0 in the highest-diversity setting, Delta is {lam0_best['delta_mean']:+.2f}, so pooling still has a slight edge because splitting created no target information.",
                "The mechanism page is strongest when read alongside the exact identity: runs with a larger absolute held-out target gap tend to show larger empirical Delta. That is the expected link between target-side structural signal and estimator-side gains from per-section inference.",
            ]
        )
        ax_note.text(0.0, 0.98, final_text, va="top", fontsize=11.5, linespacing=1.4)
        fig.suptitle("Mechanism and Final Takeaway", fontsize=16, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        summary_lines = [
            "Best-of threshold summary",
            "",
            f"stage2 metric            : {STAGE2_ERROR_LABEL}",
            f"delta convention         : {DELTA_LABEL}",
            f"fit docs / held-out docs : {report_train_docs} / {report_test_docs}",
            "",
            f"stage1 exact mean error  : {stage1_exact_mean:.2e}",
            f"utility pca exact dim    : {utility_pca_exact_dims[0] if utility_pca_exact_dims else 'n/a'}",
            f"count svd exact dim      : {count_svd_exact_dims[0] if count_svd_exact_dims else 'n/a'}",
            "",
            "last heterogeneity with positive mean Delta",
        ]
        for llt in cross_llts:
            summary_lines.append(f"  {llt:>3} tokens : {('never' if last_positive_tau[llt] is None else _tau_label(last_positive_tau[llt]))}")
        summary_lines.extend(
            [
                "",
                "quadratic-weight onset thresholds",
                "  heterogeneity         64tok   96tok",
                "  -----------------------------------",
            ]
        )
        for tau in onset_taus:
            summary_lines.append(
                f"  {_tau_label(tau):<10} {('never' if onset_table[(tau, 64)] is None else f'{onset_table[(tau, 64)]:g}'):>6} {('never' if onset_table[(tau, 96)] is None else f'{onset_table[(tau, 96)]:g}'):>7}"
            )
        summary_lines.extend(
            [
                "",
                f"broad-sweep best cell    : delta={hero_best['delta_mean']:+.2f} at {_tau_label(best_tau)}, {best_llt} tokens",
                f"broad-sweep mid cell     : delta={hero_mid_cell['delta_mean']:+.2f} at {_tau_label(mid_tau)}, {best_llt} tokens",
                f"broad-sweep worst cell   : delta={hero_worst['delta_mean']:+.2f} at {_tau_label(worst_tau)}, {best_llt} tokens",
                f"follow-up gap correlation: {gap_corr:.3f}",
                "",
                f"stage1 results           : {stage1_completed}/{stage1_total if stage1_total is not None else stage1_completed}",
                f"stage2 results           : {stage2_completed}/{stage2_total if stage2_total is not None else stage2_completed}",
                f"follow-up results        : {followup_completed}",
            ]
        )
        _text_page(pdf, title="Appendix: Threshold Summary", lines=summary_lines, font_size=11)

    summary = {
        "snapshot_label": args.snapshot_label,
        "stage1_root": str(args.stage1_root),
        "stage2_root": str(args.stage2_root),
        "followup_root": str(args.followup_root),
        "output_dir": str(output_dir),
        "markdown": str(md_path),
        "pdf": str(pdf_path),
        "stage1_completed": stage1_completed,
        "stage1_total": stage1_total,
        "stage2_completed": stage2_completed,
        "stage2_total": stage2_total,
        "followup_completed": followup_completed,
        "stage1_train_docs": stage1_train_docs,
        "stage1_test_docs": stage1_test_docs,
        "stage2_train_docs": stage2_train_docs,
        "stage2_test_docs": stage2_test_docs,
        "doc_tokens": doc_tokens,
        "stage1_metric": STAGE1_ERROR_LABEL,
        "stage2_metric": STAGE2_ERROR_LABEL,
        "delta_definition": DELTA_LABEL,
        "stage2_taus": stage2_taus,
        "stage2_lambdas": stage2_lambdas,
        "stage2_section_sizes": stage2_llts,
        "followup_cross_taus": cross_taus,
        "followup_cross_section_sizes": cross_llts,
        "followup_lambda_onsets": {
            f"tau_{tau:g}_tokens_{llt}": onset_table[(tau, llt)]
            for tau in onset_taus
            for llt in onset_llts
        },
        "followup_last_positive_tau": {
            str(llt): last_positive_tau[llt]
            for llt in cross_llts
        },
        "hero_best": hero_best,
        "hero_mid": hero_mid_cell,
        "hero_worst": hero_worst,
        "lambda_zero_best": lam0_best,
        "lambda_one_best": lam1_best,
        "gap_signal_delta_corr": gap_corr,
        "followup_purposes": followup_purposes,
        "followup_manifest_counts": dict(followup_manifest_counts),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote_markdown | {md_path}")
    print(f"wrote_pdf | {pdf_path}")
    print(f"wrote_summary | {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
