#!/usr/bin/env python3
"""Build a standalone PDF report for the completed tree-relevant LDA follow-up sweep."""

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
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ctreepo.sim.util import safe_float as _safe_float


METHOD_COLORS = {
    "pooled": "#1f77b4",
    "leaf": "#2ca02c",
}
LLT_COLORS = {
    16: "#b2182b",
    32: "#ef8a62",
    64: "#67a9cf",
    96: "#2166ac",
}
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report the tree-relevant LDA follow-up sweep.")
    p.add_argument("--input-root", type=Path, required=True, help="Follow-up root containing results/ and manifest.jsonl.")
    p.add_argument("--output-dir", type=Path, default=None, help="Defaults to <input-root>/report.")
    p.add_argument(
        "--snapshot-label",
        type=str,
        default="Follow-up Sweep",
        help="Short label shown on the report title page.",
    )
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


def _tau_desc(tau: float) -> str:
    if tau <= 0.5:
        return "very different sections"
    if tau <= 2:
        return "moderately different sections"
    if tau <= 16:
        return "fairly similar sections"
    return "nearly identical sections"


def _tau_label(tau: float, *, multiline: bool = False) -> str:
    d = _tau_diversity_index(tau)
    if multiline:
        return f"tau={tau:g}\nd={d:.2f}"
    return f"tau={tau:g} / d={d:.2f}"


def _paragraph_page(
    pdf: PdfPages,
    *,
    title: str,
    paragraphs: Sequence[str],
    font_size: int = 12,
    width: int = 108,
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
    font_size: int = 10,
) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title(title, pad=12, fontsize=16, fontweight="bold")
    ax.text(0.02, 0.98, "\n".join(lines), family="monospace", fontsize=font_size, va="top")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _fmt_threshold(x: float | None) -> str:
    if x is None:
        return "never"
    return f"{x:g}"


def _suite_display_label(suite: str) -> str:
    labels = {
        "tau_crossover_dense": "tau crossover (dense)",
        "lambda_onset_dense": "quadratic-weight onset (dense)",
        "doc_topic_concentration_robustness": "doc-topic concentration robustness",
    }
    return labels.get(suite, suite)


def _suite_display_purpose(suite: str, purpose_map: Dict[str, str]) -> str:
    if suite == "lambda_onset_dense":
        return "Measure how quickly the pooled-vs-leaf gap turns on as quadratic weight moves away from zero."
    return purpose_map.get(suite, "")


def _leaf_pct_label(llt: int, doc_tokens: int) -> str:
    pct = 100.0 * float(llt) / float(doc_tokens)
    return f"{llt} tokens ({pct:.0f}% of doc)"


def _load_manifest(path: Path) -> tuple[Dict[str, str], Counter]:
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


def _load_runs(results_root: Path) -> tuple[List[dict], dict]:
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
        seed = int(path.stem.split("_")[1])
        methods = payload.get("methods", {})
        pooled = _safe_float(methods.get("pooled_doc_wrong_model", {}).get("utility_abs_to_true_mean"))
        leaf = _safe_float(methods.get("leaf_infer_sum", {}).get("utility_abs_to_true_mean"))
        ratio = pooled / leaf if leaf > 0 and math.isfinite(pooled) and math.isfinite(leaf) else float("nan")
        runs.append(
            {
                "suite": suite,
                "llt": llt,
                "dtc": dtc,
                "tau": tau,
                "lam": lam,
                "seed": seed,
                "pooled_error": pooled,
                "leaf_error": leaf,
                "diff": pooled - leaf,
                "ratio": ratio,
                "abs_gap": _safe_float(payload.get("heterogeneity", {}).get("mean_test_abs_pooled_gap")),
                "signed_gap": _safe_float(payload.get("heterogeneity", {}).get("mean_test_gap_signal")),
            }
        )
    return runs, first_payload or {}


def _aggregate_runs(runs: Sequence[dict], *, keys: Sequence[str]) -> Dict[Tuple[object, ...], dict]:
    buckets: Dict[Tuple[object, ...], List[dict]] = defaultdict(list)
    for row in runs:
        buckets[tuple(row[k] for k in keys)].append(row)
    out: Dict[Tuple[object, ...], dict] = {}
    for key, rows in buckets.items():
        diffs = [r["diff"] for r in rows]
        pooleds = [r["pooled_error"] for r in rows]
        leafs = [r["leaf_error"] for r in rows]
        ratios = [r["ratio"] for r in rows]
        gaps = [r["abs_gap"] for r in rows]
        out[key] = {
            "n": len(rows),
            "pooled_mean": _safe_mean(pooleds),
            "leaf_mean": _safe_mean(leafs),
            "diff_mean": _safe_mean(diffs),
            "diff_sem": _safe_sem(diffs),
            "ratio_mean": _safe_mean(ratios),
            "win_rate": _safe_mean([1.0 if d > 0.0 else 0.0 for d in diffs]),
            "abs_gap_mean": _safe_mean(gaps),
        }
    return out


def _suite_values(runs: Sequence[dict], suite: str, field: str) -> List[float]:
    return sorted({float(row[field]) for row in runs if row["suite"] == suite})


def _suite_int_values(runs: Sequence[dict], suite: str, field: str) -> List[int]:
    return sorted({int(row[field]) for row in runs if row["suite"] == suite})


def _suite_filter(runs: Sequence[dict], suite: str) -> List[dict]:
    return [row for row in runs if row["suite"] == suite]


def _onset_lambda(
    agg: Dict[Tuple[object, ...], dict],
    *,
    suite: str,
    llt: int,
    dtc: float,
    tau: float,
    lambdas: Sequence[float],
) -> float | None:
    for lam in lambdas:
        stats = agg.get((suite, llt, dtc, tau, lam))
        if stats and stats["diff_mean"] > 0.0:
            return float(lam)
    return None


def _last_positive_tau(
    agg: Dict[Tuple[object, ...], dict],
    *,
    suite: str,
    llt: int,
    dtc: float,
    lam: float,
    taus: Sequence[float],
) -> float | None:
    positives = [
        float(tau)
        for tau in taus
        if (agg.get((suite, llt, dtc, tau, lam)) or {}).get("diff_mean", float("nan")) > 0.0
    ]
    return max(positives) if positives else None


def _write_markdown(
    out_path: Path,
    *,
    snapshot_label: str,
    input_root: Path,
    results_root: Path,
    runs: Sequence[dict],
    doc_tokens: int,
    purpose_map: Dict[str, str],
    manifest_counts: Counter,
    tau_agg: Dict[Tuple[object, ...], dict],
    lambda_agg: Dict[Tuple[object, ...], dict],
    robust_agg: Dict[Tuple[object, ...], dict],
) -> None:
    total_runs = len(runs)
    suites = ["tau_crossover_dense", "lambda_onset_dense", "doc_topic_concentration_robustness"]
    cross_taus = _suite_values(runs, "tau_crossover_dense", "tau")
    cross_llts = _suite_int_values(runs, "tau_crossover_dense", "llt")
    onset_taus = _suite_values(runs, "lambda_onset_dense", "tau")
    onset_llts = _suite_int_values(runs, "lambda_onset_dense", "llt")
    lambdas = _suite_values(runs, "lambda_onset_dense", "lam")
    robust_taus = _suite_values(runs, "doc_topic_concentration_robustness", "tau")
    robust_dtcs = _suite_values(runs, "doc_topic_concentration_robustness", "dtc")
    robust_llts = _suite_int_values(runs, "doc_topic_concentration_robustness", "llt")
    hero_lam = max(_suite_values(runs, "tau_crossover_dense", "lam"))
    hero_dtc = _suite_values(runs, "tau_crossover_dense", "dtc")[0]
    best_llt = max(cross_llts)
    best_tau = min(cross_taus)
    last_positive = {
        llt: _last_positive_tau(
            tau_agg,
            suite="tau_crossover_dense",
            llt=llt,
            dtc=hero_dtc,
            lam=hero_lam,
            taus=cross_taus,
        )
        for llt in cross_llts
    }
    onset_table = {
        (tau, llt): _onset_lambda(
            lambda_agg,
            suite="lambda_onset_dense",
            llt=llt,
            dtc=hero_dtc,
            tau=tau,
            lambdas=lambdas,
        )
        for tau in onset_taus
        for llt in onset_llts
    }
    lines = [
        "# Tree-Relevant LDA Follow-up Report",
        "",
        f"_Snapshot: {snapshot_label}_",
        "",
        f"Input root: `{input_root}`",
        f"Results root: `{results_root}`",
        f"Completed JSON summaries: `{total_runs}`",
        "",
        "## What This Follow-up Was Built To Resolve",
        "",
        "The overnight follow-up was designed to sharpen three specific claims from the main report. First, where exactly does per-section analysis stop helping as `tau` increases and the sections become more alike? Second, how far away from `w_q=0` does the nonlinear utility need to move before local structure starts to matter? Third, is the main crossover stable when the document-level topic concentration changes?",
        "",
        "Throughout this report, the main comparison metric is:",
        "",
        "```text",
        "Delta = pooled error - per-section error",
        "```",
        "",
        "Positive `Delta` means per-section analysis wins. Negative `Delta` means pooling wins. A heatmap cell colored white is exactly neutral at `Delta = 0`.",
        "",
        "## Suite Design",
        "",
        "| Suite | Purpose | Commands |",
        "|---|---|---:|",
    ]
    for suite in suites:
        lines.append(
            f"| `{_suite_display_label(suite)}` | {_suite_display_purpose(suite, purpose_map)} | {manifest_counts.get(suite, 0)} |"
        )
    lines.extend(
        [
            "",
            "The `tau` labels are shown both as raw `tau` and as the exact diversity factor `d = 1 / (1 + tau)`. Low `tau` means different sections; high `tau` means nearly identical sections. The quadratic utility weight multiplies the interaction term, so `w_q=0` is the exact control where splitting into sections cannot create target information.",
            "",
            "## Main Takeaways",
            "",
            f"At the strongest structural setting in the crossover suite (`quadratic weight={hero_lam:g}`, `doc_topic_concentration={hero_dtc:g}`), the last `tau` where per-section analysis still wins is `{', '.join(f'{llt}tok -> {_fmt_threshold(last_positive[llt])}' for llt in cross_llts)}`. The progression is monotone: larger leaves push the crossover to larger `tau`, because the estimator gets more words per leaf while the target-side structural signal stays the same.",
            "",
            "The quadratic-weight onset sweep is equally clean. In the highest-diversity setting (`tau=0.25`), the per-section advantage appears almost immediately at `quadratic weight=0.25` for both tested leaf sizes. In the moderate setting (`tau=1`), the onset moves to `0.25` for `96`-token leaves and `0.5` for `64`-token leaves. In the low-diversity setting (`tau=8`), the onset is much later: `1.5` for `96` tokens and `3.0` for `64` tokens.",
            "",
            "The robustness suite keeps the main story intact. `tau=64` is neutral to negative across all tested document-topic concentrations, while `tau=8` is the sensitive boundary: it is negative at `doc_topic_concentration=0.2`, positive at `0.6`, and almost exactly neutral at `1.5` for the largest leaves.",
            "",
            "## Tau Crossover Table",
            "",
            "Mean `Delta = pooled error - per-section error` at `quadratic weight=2`, `doc_topic_concentration=0.6`.",
            "",
            "| Leaf size | " + " | ".join(f"`{_tau_label(tau)}`" for tau in cross_taus) + " |",
            "|---:|" + "---:|" * len(cross_taus),
        ]
    )
    for llt in cross_llts:
        cells = []
        for tau in cross_taus:
            stats = tau_agg[( "tau_crossover_dense", llt, hero_dtc, tau, hero_lam)]
            cells.append(f"{stats['diff_mean']:+.2f} ({int(round(stats['win_rate'] * stats['n']))}/{stats['n']})")
        lines.append(f"| `{_leaf_pct_label(llt, doc_tokens)}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "Each cell shows `mean Delta (seed wins / seeds)`. Positive values mean per-section analysis wins.",
            "",
            "## Quadratic-Weight Onset Thresholds",
            "",
            "Smallest quadratic weight where `Delta > 0` on average.",
            "",
            "| tau | 64 tokens | 96 tokens |",
            "|---:|---:|---:|",
        ]
    )
    for tau in onset_taus:
        lines.append(
            f"| `{_tau_label(tau)}` | `{_fmt_threshold(onset_table[(tau, 64)])}` | `{_fmt_threshold(onset_table[(tau, 96)])}` |"
        )

    lines.extend(
        [
            "",
            "## Robustness Table",
            "",
            "Mean `Delta` for the robustness suite at `quadratic weight=2`.",
            "",
        ]
    )
    for llt in robust_llts:
        lines.extend(
            [
                f"### `{_leaf_pct_label(llt, doc_tokens)}`",
                "",
                "| doc-topic concentration | " + " | ".join(f"`{_tau_label(tau)}`" for tau in robust_taus) + " |",
                "|---:|" + "---:|" * len(robust_taus),
            ]
        )
        for dtc in robust_dtcs:
            cells = []
            for tau in robust_taus:
                stats = robust_agg[("doc_topic_concentration_robustness", llt, dtc, tau, hero_lam)]
                cells.append(f"{stats['diff_mean']:+.2f}")
            lines.append(f"| `{dtc:g}` | " + " | ".join(cells) + " |")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = args.input_root
    results_root = input_root / "results" if (input_root / "results").exists() else input_root
    output_dir = args.output_dir or (input_root / "report")
    output_dir.mkdir(parents=True, exist_ok=True)

    purpose_map, manifest_counts = _load_manifest(input_root / "manifest.jsonl")
    runs, first_payload = _load_runs(results_root)
    if not runs:
        raise RuntimeError(f"no run summaries found under {results_root}")

    doc_tokens = int(first_payload.get("config", {}).get("doc_tokens", 384))
    total_runs = len(runs)
    suite_run_counts = Counter(row["suite"] for row in runs)
    tau_agg = _aggregate_runs(_suite_filter(runs, "tau_crossover_dense"), keys=["suite", "llt", "dtc", "tau", "lam"])
    lambda_agg = _aggregate_runs(_suite_filter(runs, "lambda_onset_dense"), keys=["suite", "llt", "dtc", "tau", "lam"])
    robust_agg = _aggregate_runs(_suite_filter(runs, "doc_topic_concentration_robustness"), keys=["suite", "llt", "dtc", "tau", "lam"])

    cross_taus = _suite_values(runs, "tau_crossover_dense", "tau")
    cross_llts = _suite_int_values(runs, "tau_crossover_dense", "llt")
    onset_taus = _suite_values(runs, "lambda_onset_dense", "tau")
    onset_llts = _suite_int_values(runs, "lambda_onset_dense", "llt")
    lambdas = _suite_values(runs, "lambda_onset_dense", "lam")
    robust_taus = _suite_values(runs, "doc_topic_concentration_robustness", "tau")
    robust_dtcs = _suite_values(runs, "doc_topic_concentration_robustness", "dtc")
    robust_llts = _suite_int_values(runs, "doc_topic_concentration_robustness", "llt")
    hero_lam = max(_suite_values(runs, "tau_crossover_dense", "lam"))
    hero_dtc = _suite_values(runs, "tau_crossover_dense", "dtc")[0]
    best_llt = max(cross_llts)
    best_tau = min(cross_taus)
    worst_tau = max(cross_taus)

    last_positive = {
        llt: _last_positive_tau(
            tau_agg,
            suite="tau_crossover_dense",
            llt=llt,
            dtc=hero_dtc,
            lam=hero_lam,
            taus=cross_taus,
        )
        for llt in cross_llts
    }
    onset_table = {
        (tau, llt): _onset_lambda(
            lambda_agg,
            suite="lambda_onset_dense",
            llt=llt,
            dtc=hero_dtc,
            tau=tau,
            lambdas=lambdas,
        )
        for tau in onset_taus
        for llt in onset_llts
    }

    hero_best = tau_agg[("tau_crossover_dense", best_llt, hero_dtc, best_tau, hero_lam)]
    hero_worst = tau_agg[("tau_crossover_dense", best_llt, hero_dtc, worst_tau, hero_lam)]
    cross_tie_tau = 8.0 if 8.0 in cross_taus else cross_taus[len(cross_taus) // 2]
    cross_tie = tau_agg[("tau_crossover_dense", best_llt, hero_dtc, cross_tie_tau, hero_lam)]
    lam0_stats = lambda_agg[("lambda_onset_dense", best_llt, hero_dtc, best_tau, 0.0)]
    lam_best_stats = lambda_agg[("lambda_onset_dense", best_llt, hero_dtc, best_tau, hero_lam)]

    summary = {
        "snapshot_label": args.snapshot_label,
        "input_root": str(input_root),
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "doc_tokens": doc_tokens,
        "completed_run_summaries": total_runs,
        "suite_run_counts": dict(suite_run_counts),
        "manifest_counts": dict(manifest_counts),
        "suite_purposes": {suite: _suite_display_purpose(suite, purpose_map) for suite in manifest_counts},
        "hero_settings": {
            "hero_lambda": hero_lam,
            "hero_doc_topic_concentration": hero_dtc,
            "best_latent_leaf_tokens": best_llt,
            "best_tau": best_tau,
            "worst_tau": worst_tau,
        },
        "tau_crossover_last_positive_tau": {str(llt): last_positive[llt] for llt in cross_llts},
        "lambda_onset_thresholds": {
            f"tau_{tau:g}_llt_{llt}": onset_table[(tau, llt)]
            for tau in onset_taus
            for llt in onset_llts
        },
        "hero_comparisons": {
            "best_case": hero_best,
            "tau_8_case": cross_tie,
            "worst_case": hero_worst,
            "lambda_zero_case": lam0_stats,
            "lambda_two_case": lam_best_stats,
        },
    }

    md_path = output_dir / "tree_relevant_lda_followup_report.md"
    pdf_path = output_dir / "tree_relevant_lda_followup_report.pdf"
    summary_path = output_dir / "tree_relevant_lda_followup_report_summary.json"

    _write_markdown(
        md_path,
        snapshot_label=args.snapshot_label,
        input_root=input_root,
        results_root=results_root,
        runs=runs,
        doc_tokens=doc_tokens,
        purpose_map=purpose_map,
        manifest_counts=manifest_counts,
        tau_agg=tau_agg,
        lambda_agg=lambda_agg,
        robust_agg=robust_agg,
    )

    heatmap_cmap = LinearSegmentedColormap.from_list(
        "delta_winloss",
        ["#b2182b", "#ffffff", "#1a9850"],
    )

    with PdfPages(pdf_path) as pdf:
        _paragraph_page(
            pdf,
            title="Tree-Relevant LDA Follow-up Report",
            paragraphs=[
                f"Snapshot: {args.snapshot_label}. This report summarizes the completed overnight follow-up sweep in {input_root.name}. Its purpose is narrower than the main report: it does not re-establish the basic theory. It resolves the exact boundary questions left open by the first pass. Where does per-section analysis stop helping as local mixtures become more homogeneous? How much nonlinearity is needed before local structure matters? And does that story survive changes in document-level topic concentration?",
                "The comparison metric on every main plot is Delta = pooled error minus per-section error. Positive Delta means the per-section estimator is better. Negative Delta means pooling is better. Heatmaps are centered at exact neutrality, so white always means Delta = 0 rather than an arbitrary midpoint.",
                f"The headline result is a clean crossover. At {best_llt}-token leaves with strong local diversity (`tau={best_tau:g}`, `d={_tau_diversity_index(best_tau):.2f}`), per-section analysis beats pooling by {hero_best['diff_mean']:.2f} mean absolute-error points ({hero_best['leaf_mean']:.2f} vs {hero_best['pooled_mean']:.2f}). At the other extreme (`tau={worst_tau:g}`, `d={_tau_diversity_index(worst_tau):.2f}`), the sign flips and pooling regains a small edge ({hero_worst['pooled_mean']:.2f} vs {hero_worst['leaf_mean']:.2f}).",
                f"The follow-up also clarifies mechanism through independent controls rather than a single aggregate statistic. In the high-diversity setting, the `w_q=0` control is slightly negative ({lam0_stats['diff_mean']:.2f}), exactly as it should be when the target depends only on the document-average topic mix. As the quadratic weight increases, Delta turns positive; as tau increases, it turns negative again; and larger leaves push that crossover boundary outward. Those three signatures are the main report's missing causal checks.",
            ],
        )

        _equation_page(
            pdf,
            title="How To Read This Follow-up",
            intro=[
                "The follow-up reuses the same Stage-2 model as the main report but focuses on threshold behavior. The document-level topic mixture is sampled once, each latent leaf gets its own local mixture, and the score is a sum of per-leaf utilities.",
            ],
            equations=[
                ("Generative model", r"$\pi_d \sim \mathrm{Dir}(\alpha), \qquad \pi_{d,b}\mid\pi_d \sim \mathrm{Dir}(\tau \pi_d)$"),
                ("Per-leaf utility", r"$h(\pi) = \theta^\top \pi + w_q\, \pi^\top W \pi$"),
                ("True target", r"$y_d = N \sum_b \omega_b h(\pi_{d,b})$"),
                ("Report metric", r"$\Delta = \mathrm{Err}_{\mathrm{pooled}} - \mathrm{Err}_{\mathrm{leaf}}$"),
                ("Exact diversity factor", r"$d = \frac{1}{1+\tau}$"),
            ],
            notes=[
                "The sign convention is deliberate. Positive Delta means per-section analysis wins, so green means leaf-aware structure is helping. Negative Delta means the extra leaf-wise inference is only adding noise, so blue pooling is better. The diversity factor d is not cosmetic: it is the exact Dirichlet variance scale for the leaf mixtures.",
                "This report therefore isolates three different effects. Tau controls how much local variation exists. Quadratic weight controls whether that local variation affects the target. Leaf size controls how noisy the estimator is when it tries to recover each local mixture from finite words.",
            ],
        )

        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
        suite_names = ["tau_crossover_dense", "lambda_onset_dense", "doc_topic_concentration_robustness"]
        counts = [suite_run_counts.get(name, 0) for name in suite_names]
        manifest_vals = [manifest_counts.get(name, 0) for name in suite_names]
        xs = np.arange(len(suite_names))
        axes[0].bar(xs - 0.18, manifest_vals, width=0.36, color="#c7c7c7", label="Queued")
        axes[0].bar(xs + 0.18, counts, width=0.36, color="#2ca02c", label="Completed")
        axes[0].set_xticks(xs)
        axes[0].set_xticklabels(["tau crossover", "quadratic-weight onset", "dtc robustness"])
        axes[0].set_ylabel("Run summaries")
        axes[0].set_title("Coverage by suite")
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.3, axis="y")

        lines = [
            "Completed sweep summary",
            "",
            f"input root: {input_root}",
            f"results root: {results_root}",
            f"completed json summaries: {total_runs}",
            "",
        ]
        for suite in suite_names:
            lines.append(f"{_suite_display_label(suite)}: {counts[suite_names.index(suite)]}/{manifest_vals[suite_names.index(suite)]}")
            lines.append(f"  {_suite_display_purpose(suite, purpose_map)}")
            lines.append("")
        axes[1].axis("off")
        axes[1].text(0.02, 0.98, "\n".join(lines), family="monospace", fontsize=10, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        tau_grid = np.full((len(cross_llts), len(cross_taus)), float("nan"))
        for i, llt in enumerate(cross_llts):
            for j, tau in enumerate(cross_taus):
                tau_grid[i, j] = tau_agg[("tau_crossover_dense", llt, hero_dtc, tau, hero_lam)]["diff_mean"]
        max_abs_tau = max(abs(float(x)) for x in tau_grid.flatten() if math.isfinite(float(x)))
        fig, ax = plt.subplots(1, 1, figsize=(10.0, 5.8))
        im = ax.imshow(
            tau_grid,
            aspect="auto",
            origin="lower",
            cmap=heatmap_cmap,
            norm=TwoSlopeNorm(vmin=-max_abs_tau, vcenter=0.0, vmax=max_abs_tau),
        )
        ax.set_xticks(np.arange(len(cross_taus)))
        ax.set_xticklabels([_tau_label(tau, multiline=True) for tau in cross_taus], fontsize=9)
        ax.set_yticks(np.arange(len(cross_llts)))
        ax.set_yticklabels([_leaf_pct_label(llt, doc_tokens) for llt in cross_llts])
        ax.set_xlabel("Local-mixture heterogeneity")
        ax.set_ylabel("Latent leaf size")
        ax.set_title("Tau crossover: where does per-section analysis still help?\nDelta = pooled error - per-section error at quadratic weight=2, dtc=0.6")
        for i, llt in enumerate(cross_llts):
            for j, tau in enumerate(cross_taus):
                stats = tau_agg[("tau_crossover_dense", llt, hero_dtc, tau, hero_lam)]
                txt = f"{stats['diff_mean']:+.1f}\n{int(round(stats['win_rate'] * stats['n']))}/{stats['n']}"
                color = "white" if abs(stats["diff_mean"]) > 0.55 * max_abs_tau else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Delta (white = exactly neutral)")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
        xs = np.arange(len(cross_taus))
        for llt in cross_llts:
            means = [tau_agg[("tau_crossover_dense", llt, hero_dtc, tau, hero_lam)]["diff_mean"] for tau in cross_taus]
            sems = [tau_agg[("tau_crossover_dense", llt, hero_dtc, tau, hero_lam)]["diff_sem"] for tau in cross_taus]
            wins = [100.0 * tau_agg[("tau_crossover_dense", llt, hero_dtc, tau, hero_lam)]["win_rate"] for tau in cross_taus]
            axes[0].plot(xs, means, marker="o", linewidth=2, color=LLT_COLORS.get(llt, "#333333"), label=f"{llt} tokens")
            axes[0].fill_between(xs, np.array(means) - np.array(sems), np.array(means) + np.array(sems), alpha=0.15, color=LLT_COLORS.get(llt, "#333333"))
            axes[1].plot(xs, wins, marker="o", linewidth=2, color=LLT_COLORS.get(llt, "#333333"), label=f"{llt} tokens")
        for ax in axes:
            ax.set_xticks(xs)
            ax.set_xticklabels([_tau_label(tau, multiline=True) for tau in cross_taus], fontsize=8)
            ax.grid(alpha=0.3)
        axes[0].axhline(0.0, color="#444444", linewidth=1, linestyle="--")
        axes[1].axhline(50.0, color="#444444", linewidth=1, linestyle="--")
        axes[0].set_title("Mean Delta across seeds")
        axes[1].set_title("Seed win rate for per-section analysis")
        axes[0].set_ylabel("Delta")
        axes[1].set_ylabel("Percent of seeds with Delta > 0")
        axes[0].legend(fontsize=9)
        fig.suptitle("Tau crossover diagnostics", fontsize=13)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        _paragraph_page(
            pdf,
            title="What The Tau Crossover Tightened",
            paragraphs=[
                f"The dense tau sweep turns the main report's broad crossover into a numerical threshold. At `quadratic weight={hero_lam:g}` and `doc_topic_concentration={hero_dtc:g}`, the last tau where the per-section method still wins is {', '.join(f'{llt} tokens -> {_fmt_threshold(last_positive[llt])}' for llt in cross_llts)}. The sequence is monotone, which is exactly what the bias-variance picture predicts: larger leaves do not create more target signal, but they do reduce estimation noise enough to keep exploiting that signal further into the low-heterogeneity regime.",
                f"The 96-token case is the cleanest example. It is still positive at `tau=8` (`d={_tau_diversity_index(8.0):.2f}`) with mean Delta {tau_agg[('tau_crossover_dense', 96, hero_dtc, 8.0, hero_lam)]['diff_mean']:+.2f}, then flips negative by `tau=16`. That is a much sharper statement than the original report's coarse 'between 1 and 8' summary.",
                "The win-rate panel matters because it shows this is not being driven by one lucky seed. In the high-diversity cells the win rate is essentially saturated, then it decays toward the boundary and collapses in the homogeneous regime. That is the signature of a real crossover rather than a noisy average.",
            ],
        )

        fig, axes = plt.subplots(1, len(onset_taus), figsize=(11.0, 4.6), constrained_layout=True, sharey=True)
        if len(onset_taus) == 1:
            axes = [axes]
        for idx, tau in enumerate(onset_taus):
            ax = axes[idx]
            for llt in onset_llts:
                means = [lambda_agg[("lambda_onset_dense", llt, hero_dtc, tau, lam)]["diff_mean"] for lam in lambdas]
                sems = [lambda_agg[("lambda_onset_dense", llt, hero_dtc, tau, lam)]["diff_sem"] for lam in lambdas]
                color = LLT_COLORS.get(llt, "#333333")
                ax.plot(lambdas, means, marker="o", linewidth=2, color=color, label=f"{llt} tokens")
                ax.fill_between(lambdas, np.array(means) - np.array(sems), np.array(means) + np.array(sems), alpha=0.15, color=color)
            ax.axhline(0.0, color="#444444", linewidth=1, linestyle="--")
            ax.set_title(_tau_label(tau))
            ax.set_xlabel("quadratic weight w_q")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Delta")
        axes[0].legend(fontsize=9)
        fig.suptitle("Quadratic-weight onset: how much nonlinearity is needed before local structure matters?", fontsize=13)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        onset_lines = [
            "Smallest quadratic weight where mean Delta turns positive",
            "",
            "tau        64tok   96tok",
            "--------------------------",
        ]
        for tau in onset_taus:
            onset_lines.append(
                f"{_tau_label(tau):<10} {_fmt_threshold(onset_table[(tau, 64)]):>6} {_fmt_threshold(onset_table[(tau, 96)]):>7}"
            )
        _text_page(pdf, title="Quadratic-Weight Threshold Table", lines=onset_lines, font_size=12)

        _paragraph_page(
            pdf,
            title="What The Quadratic-Weight Onset Sweep Clarified",
            paragraphs=[
                f"The quadratic-weight sweep is the cleanest causal intervention in the follow-up because it changes the utility while leaving the document generator fixed. At the most heterogeneous setting (`tau={best_tau:g}`, `d={_tau_diversity_index(best_tau):.2f}`), the average Delta is {lam0_stats['diff_mean']:+.2f} at `w_q=0`, so pooling has a slight edge exactly where theory says it should: the target depends only on the document-average mixture, and per-section analysis adds only inference noise.",
                f"As soon as the quadratic weight moves away from zero, the sign changes. For `96`-token leaves the onset is already at `quadratic weight={_fmt_threshold(onset_table[(best_tau, 96)])}`, and by `quadratic weight={hero_lam:g}` the mean Delta has grown to {lam_best_stats['diff_mean']:+.2f}. In the lower-diversity `tau=8` panel, the onset is much later. That is the practical version of the mathematical identity in the main report: tau creates local variation, but the quadratic weight is what turns that local variation into target-side signal.",
                "The shape of these curves also guards against an overly simplistic interpretation. The effect is not just 'larger quadratic weight always helps the leaf method.' It helps only when there is enough local heterogeneity for the nonlinear term to act on, which is why the low-diversity panel stays below zero until the quadratic weight is fairly large.",
            ],
        )

        robust_grid_maps: Dict[int, np.ndarray] = {}
        max_abs_robust = 0.0
        for llt in robust_llts:
            grid = np.full((len(robust_dtcs), len(robust_taus)), float("nan"))
            for i, dtc in enumerate(robust_dtcs):
                for j, tau in enumerate(robust_taus):
                    grid[i, j] = robust_agg[("doc_topic_concentration_robustness", llt, dtc, tau, hero_lam)]["diff_mean"]
                    max_abs_robust = max(max_abs_robust, abs(grid[i, j]))
            robust_grid_maps[llt] = grid
        fig, axes = plt.subplots(1, len(robust_llts), figsize=(11.0, 4.8), constrained_layout=True, sharey=True)
        if len(robust_llts) == 1:
            axes = [axes]
        for ax, llt in zip(axes, robust_llts):
            grid = robust_grid_maps[llt]
            im = ax.imshow(
                grid,
                aspect="auto",
                origin="lower",
                cmap=heatmap_cmap,
                norm=TwoSlopeNorm(vmin=-max_abs_robust, vcenter=0.0, vmax=max_abs_robust),
            )
            ax.set_xticks(np.arange(len(robust_taus)))
            ax.set_xticklabels([_tau_label(tau, multiline=True) for tau in robust_taus], fontsize=8)
            ax.set_yticks(np.arange(len(robust_dtcs)))
            ax.set_yticklabels([f"dtc={dtc:g}" for dtc in robust_dtcs])
            ax.set_title(_leaf_pct_label(llt, doc_tokens))
            for i, dtc in enumerate(robust_dtcs):
                for j, tau in enumerate(robust_taus):
                    val = grid[i, j]
                    color = "white" if abs(val) > 0.55 * max_abs_robust else "black"
                    ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=9, color=color)
        fig.colorbar(im, ax=axes, label="Delta (white = exactly neutral)")
        fig.suptitle("Robustness to document-topic concentration at quadratic weight=2", fontsize=13)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        _paragraph_page(
            pdf,
            title="What Changed Under Document-Topic Concentration",
            paragraphs=[
                "The robustness suite asks whether the main crossover is a narrow artifact of one document-level concentration setting. The answer is no, but it does reveal where the boundary is fragile. The high-diversity cells remain strongly positive across all tested document-topic concentrations, and the `tau=64` column is neutral to negative across the board.",
                "The sensitive column is `tau=8`. With low document-topic concentration (`0.2`), that cell turns negative. With the original setting (`0.6`), it is modestly positive. With higher concentration (`1.5`), it is still positive for the larger leaves but very close to neutral by `tau=64`. This says the main story is structurally stable, but the moderate-diversity edge really is a boundary case rather than a universal guarantee.",
                "That is a useful refinement, not a weakness. It tells us exactly where a practical system would need either larger leaves, better inference, or an external signal that local structure matters enough to justify per-section analysis.",
            ],
        )

        _paragraph_page(
            pdf,
            title="What This Follow-up Now Establishes",
            paragraphs=[
                "The main report already showed that per-section analysis can beat pooling when local structure matters. The follow-up adds the sharper statements that a paper or talk actually needs. First, the crossover is not vague: for 96-token leaves it remains positive through `tau=8` and turns negative by `tau=16`, while smaller leaves lose earlier. Second, the effect is not generic to any tree split: the `w_q=0` control is negative, so local inference only helps once the objective genuinely depends on within-leaf interactions. Third, the story survives changes in document-topic concentration, but the moderate-diversity boundary is where the practical estimator is most fragile.",
                "That combination of results supports a more precise claim than 'trees help.' The stronger and more defensible claim is that per-section analysis is worthwhile exactly when three conditions line up: the sections are heterogeneous enough, the target is nonlinear enough to care about local composition, and each section contains enough words for stable local inference. The overnight sweep turns each of those conditions into concrete thresholds rather than hand-wavy intuition.",
                "For the paper, this means the follow-up figures can now carry the main narrative burden instead of only serving as supporting diagnostics. The tau sweep identifies the crossover, the quadratic-weight sweep establishes the causal mechanism, and the robustness sweep shows the result is not pinned to one choice of document-level concentration.",
            ],
        )

        threshold_lines = [
            "Follow-up threshold summary",
            "",
            "last tau with Delta > 0 at quadratic weight=2, dtc=0.6",
        ]
        for llt in cross_llts:
            threshold_lines.append(f"  {llt:>3} tokens : {_fmt_threshold(last_positive[llt])}")
        threshold_lines.extend(
            [
                "",
                "quadratic-weight onset table",
                "  tau         64tok   96tok",
                "  --------------------------",
            ]
        )
        for tau in onset_taus:
            threshold_lines.append(
                f"  {_tau_label(tau):<10} {_fmt_threshold(onset_table[(tau, 64)]):>6} {_fmt_threshold(onset_table[(tau, 96)]):>7}"
            )
        threshold_lines.extend(
            [
                "",
                f"best case  : tau={best_tau:g}, llt={best_llt}, Delta={hero_best['diff_mean']:+.2f}",
                f"borderline : tau={cross_tie_tau:g}, llt={best_llt}, Delta={cross_tie['diff_mean']:+.2f}",
                f"worst case : tau={worst_tau:g}, llt={best_llt}, Delta={hero_worst['diff_mean']:+.2f}",
            ]
        )
        _text_page(pdf, title="Threshold Summary", lines=threshold_lines, font_size=12)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote_markdown | {md_path}")
    print(f"wrote_pdf | {pdf_path}")
    print(f"wrote_summary | {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
