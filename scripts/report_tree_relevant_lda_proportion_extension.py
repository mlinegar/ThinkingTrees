#!/usr/bin/env python3
"""Full report for the tree-relevant LDA coarse leaf-size extension.

.. deprecated::
    Use ``scripts/report_learnability.py --family lda`` instead.
"""

from __future__ import annotations

import warnings
warnings.warn(
    "Deprecated. Use scripts/report_learnability.py --family lda",
    DeprecationWarning,
    stacklevel=1,
)

import argparse
from collections import Counter
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
TIE_EPS = 5e-3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report the coarse leaf-size follow-up for tree-relevant LDA.")
    p.add_argument("--baseline-root", type=Path, required=True, help="Original follow-up root.")
    p.add_argument("--extension-root", type=Path, required=True, help="Coarse-size extension root.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <extension-root>/report.",
    )
    p.add_argument(
        "--snapshot-label",
        type=str,
        default="Coarse Leaf-Size Extension",
        help="Short label used on the report title page.",
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


def _tau_label(tau: float, *, multiline: bool = False) -> str:
    d = _tau_diversity_index(tau)
    if multiline:
        return f"tau={tau:g}\nd={d:.2f}"
    return f"tau={tau:g} / d={d:.2f}"


def _pct_token_label(pct: float, doc_tokens: int) -> str:
    tok = int(round(float(doc_tokens) * float(pct) / 100.0))
    return f"{pct:.0f}% ({tok} tokens)"


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


def _paragraph_page(
    pdf: PdfPages,
    *,
    title: str,
    paragraphs: Sequence[str],
    width: int = 104,
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
                "path": str(path),
                "suite": suite,
                "train_docs": int(cfg.get("train_docs", -1)),
                "test_docs": int(cfg.get("test_docs", -1)),
                "doc_tokens": doc_tokens,
                "latent_leaf_tokens": latent_leaf_tokens,
                "leaf_pct": pct,
                "doc_topic_concentration": _safe_float(cfg.get("doc_topic_concentration")),
                "tau": _safe_float(cfg.get("local_mixture_concentration")),
                "lam": _safe_float(cfg.get("quadratic_utility_weight", cfg.get("lambda_multiplier"))),
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
    candidate: float | None = None
    for tau in tau_grid:
        row = stats.get(float(tau))
        if row is not None and float(row["delta_mean"]) > 0.0:
            candidate = float(tau)
    return candidate


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


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (args.extension_root / "report")
    output_dir.mkdir(parents=True, exist_ok=True)

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

    doc_tokens = int(all_runs[0]["doc_tokens"])
    focus_pcts = [25.0, 50.0, 100.0]
    tau_grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    lambda_grid = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    suite_names = [
        "tau_crossover_dense",
        "tau_crossover_proportion_extend",
        "lambda_onset_dense",
        "lambda_onset_proportion_extend",
        "doc_topic_concentration_robustness",
        "sample_size_boundary_check",
    ]
    suite_run_counts = Counter(row["suite"] for row in all_runs)

    tau_rows = _filter_runs(
        all_runs,
        suites=TAU_SUITES,
        train_docs=512,
        dtc=0.6,
        lam=2.0,
        taus=set(tau_grid),
        pcts=set(focus_pcts),
    )
    tau_stats = _agg(tau_rows, keys=("leaf_pct", "tau"))

    lambda_rows = _filter_runs(
        all_runs,
        suites=LAMBDA_SUITES,
        train_docs=512,
        dtc=0.6,
        taus={1.0, 8.0},
        pcts=set(focus_pcts),
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
            pcts=set(focus_pcts),
        )
        + _filter_runs(
            all_runs,
            suites={BOUNDARY_EXTENSION_SUITE},
            train_docs=2048,
            dtc=0.6,
            lam=2.0,
            taus={8.0, 16.0},
            pcts=set(focus_pcts),
        )
    )
    boundary_stats = _agg(boundary_rows, keys=("train_docs", "leaf_pct", "tau"))

    last_positive_tau = {
        str(int(round(pct))): _last_positive_tau(
            tau_grid,
            {tau: tau_stats.get((pct, tau)) for tau in tau_grid},
        )
        for pct in focus_pcts
    }
    onset_eps = 0.1
    onset_table = {
        f"pct_{int(round(pct))}_tau_{tau:g}": _material_onset(
            lambda_grid,
            {lam: lambda_stats.get((pct, tau, lam)) for lam in lambda_grid},
            eps=onset_eps,
        )
        for pct in focus_pcts
        for tau in (1.0, 8.0)
    }
    best_boundary_leaf_pct = {}
    for train_docs in (512, 2048):
        for tau in (8.0, 16.0):
            options = []
            for pct in focus_pcts:
                stats = boundary_stats.get((train_docs, pct, tau))
                if stats is not None:
                    options.append((float(stats["delta_mean"]), pct))
            if options:
                best_boundary_leaf_pct[f"train_{train_docs}_tau_{tau:g}"] = max(options)[1]

    example_25_tau16 = _representative_slice(
        f"{_pct_token_label(25.0, doc_tokens)}, tau=16, quadratic weight=2, train_docs=512",
        tau_stats[(25.0, 16.0)],
    )
    example_50_tau16 = _representative_slice(
        f"{_pct_token_label(50.0, doc_tokens)}, tau=16, quadratic weight=2, train_docs=512",
        tau_stats[(50.0, 16.0)],
    )
    example_50_2048_tau8 = _representative_slice(
        f"{_pct_token_label(50.0, doc_tokens)}, tau=8, quadratic weight=2, train_docs=2048",
        boundary_stats[(2048, 50.0, 8.0)],
    )
    example_100_tau16 = _representative_slice(
        f"{_pct_token_label(100.0, doc_tokens)}, tau=16, quadratic weight=2, train_docs=512",
        boundary_stats[(512, 100.0, 16.0)],
    )
    representative_examples = [
        example_25_tau16,
        example_50_tau16,
        example_50_2048_tau8,
        example_100_tau16,
    ]

    summary = {
        "snapshot_label": str(args.snapshot_label),
        "baseline_root": str(args.baseline_root),
        "extension_root": str(args.extension_root),
        "output_dir": str(output_dir),
        "completed_run_summaries": len(all_runs),
        "baseline_run_summaries": len(baseline_runs),
        "extension_run_summaries": len(extension_runs),
        "manifest_counts": dict(manifest_counts),
        "suite_run_counts": dict(suite_run_counts),
        "suite_purposes": {suite: _suite_display_purpose(suite, purpose_map) for suite in manifest_counts},
        "focus_leaf_pcts": focus_pcts,
        "tau_grid": tau_grid,
        "lambda_grid": lambda_grid,
        "tau_crossover_last_positive_tau": last_positive_tau,
        "material_lambda_onset_eps": onset_eps,
        "material_lambda_onsets": onset_table,
        "best_boundary_leaf_pct": best_boundary_leaf_pct,
        "metric_definition": {
            "pooled_error": "mean absolute error from pooled whole-document inference",
            "leaf_error": "mean absolute error from leaf-based inference",
            "delta": "pooled_error - leaf_error",
            "positive_delta_means": "leaf error is lower than pooled error",
            "negative_delta_means": "leaf error is higher than pooled error",
            "relative_gain": "delta / pooled_error",
        },
        "representative_examples": representative_examples,
        "tau_crossover_rows": {
            f"pct_{int(round(pct))}_tau_{tau:g}": tau_stats.get((pct, tau))
            for pct in focus_pcts
            for tau in tau_grid
            if tau_stats.get((pct, tau)) is not None
        },
        "lambda_rows": {
            f"pct_{int(round(pct))}_tau_{tau:g}_lam_{lam:g}": lambda_stats.get((pct, tau, lam))
            for pct in focus_pcts
            for tau in (1.0, 8.0)
            for lam in lambda_grid
            if lambda_stats.get((pct, tau, lam)) is not None
        },
        "boundary_rows": {
            f"train_{train_docs}_pct_{int(round(pct))}_tau_{tau:g}": boundary_stats.get((train_docs, pct, tau))
            for train_docs in (512, 2048)
            for pct in focus_pcts
            for tau in (8.0, 16.0)
            if boundary_stats.get((train_docs, pct, tau)) is not None
        },
    }

    md_lines = [
        "# Tree-Relevant LDA Coarse Leaf-Size Extension Report",
        "",
        f"_Snapshot: {args.snapshot_label}_",
        "",
        f"Baseline root: `{args.baseline_root}`",
        f"Extension root: `{args.extension_root}`",
        f"Completed summaries merged here: `{len(all_runs)}` (`{len(baseline_runs)}` baseline + `{len(extension_runs)}` extension)",
        "",
        "## What This Full Report Adds",
        "",
        "The March 7 follow-up already nailed the `16/32/64/96`-token story. What it did not answer was the paper-facing coarse question: does the gain keep improving when we move from `25%` sections to `50%` sections, or does it disappear as soon as we approach the whole document?",
        "",
        "This merged report keeps the original threshold logic and adds the new coarse sweep. Every main table here is suite-restricted on purpose: tau crossover uses only the tau suites, quadratic-weight onset uses only the quadratic-weight suites, and the train-size boundary check uses the new `train_docs=2048` boundary suite rather than mixing in unrelated overlap.",
        "",
        "## Metric Semantics",
        "",
        "- `pooled error` is the mean absolute error from the pooled whole-document baseline.",
        "- `leaf error` is the mean absolute error from the leaf-based method.",
        "- `Delta = pooled error - leaf error`.",
        "- `Delta > 0` means the leaf method is better because it has lower error.",
        "- `Delta < 0` means the leaf method is worse because it has higher error.",
        "- `relative gain = Delta / pooled error`.",
        "",
        "## Suite Coverage",
        "",
        "| Suite | Purpose | Queued | Completed |",
        "|---|---|---:|---:|",
    ]
    for suite in suite_names:
        md_lines.append(
            f"| `{_suite_display_label(suite)}` | {_suite_display_purpose(suite, purpose_map)} | {manifest_counts.get(suite, 0)} | {suite_run_counts.get(suite, 0)} |"
        )
    md_lines.extend(
        [
            "",
            "## Main Takeaways",
            "",
            (
                "The main practical result is now cleaner than the earlier `96`-token read. Among the paper-facing "
                "`25% / 50% / 100%` section sizes, `50%` is the best point on the moderate-boundary slices. "
                "The `100%` case is exactly the one-leaf pooled control, so its Delta is identically zero."
            ),
            "",
            (
                f"Concrete negative example: `{example_25_tau16['label']}` has pooled error `{example_25_tau16['pooled_mean']:.2f}` "
                f"and leaf error `{example_25_tau16['leaf_mean']:.2f}`, so `Delta = {example_25_tau16['delta_mean']:+.2f}`. "
                f"That is bad: the leaf method is worse (`{example_25_tau16['interpretation']}`)."
            ),
            "",
            (
                f"Concrete positive example: `{example_50_2048_tau8['label']}` has pooled error `{example_50_2048_tau8['pooled_mean']:.2f}` "
                f"and leaf error `{example_50_2048_tau8['leaf_mean']:.2f}`, so `Delta = {example_50_2048_tau8['delta_mean']:+.2f}` "
                f"(`{example_50_2048_tau8['interpretation']}`)."
            ),
            "",
            (
                f"In the tau-crossover sweep at `quadratic weight=2`, `doc_topic_concentration=0.6`, the last positive tau is "
                f"`{_threshold_label(last_positive_tau['25'])}` at `25%`, `{_threshold_label(last_positive_tau['50'])}` at `50%`, "
                f"and `{_threshold_label(last_positive_tau['100'])}` at `100%`."
            ),
            "",
            (
                "The higher-support boundary check does not overturn that picture. At both `tau=8` and `tau=16`, "
                "`50%` remains the strongest point, `25%` improves only slightly, and `100%` stays exactly at zero."
            ),
            "",
            "## Representative Absolute-Error Slices",
            "",
            "| Setting | pooled error | leaf error | Delta | relative gain | Interpretation |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for example in representative_examples:
        md_lines.append(
            f"| `{example['label']}` | `{example['pooled_mean']:.2f}` | `{example['leaf_mean']:.2f}` | "
            f"`{example['delta_mean']:+.2f}` | `{_relative_pct_label(example['relative_gain_mean'])}` | "
            f"{example['interpretation']} |"
        )

    md_lines.extend(
        [
            "",
            "## Tau Crossover Table",
            "",
            "Mean `Delta = pooled error - per-section error` at `quadratic weight=2`, `doc_topic_concentration=0.6`, `train_docs=512`, using only the tau suites.",
            "",
            "| Leaf proportion | " + " | ".join(f"`{_tau_label(tau)}`" for tau in tau_grid) + " |",
            "|---:|" + "---:|" * len(tau_grid),
        ]
    )
    for pct in focus_pcts:
        cells = []
        for tau in tau_grid:
            stats = tau_stats[(pct, tau)]
            cells.append(f"{stats['delta_mean']:+.2f} ({stats['wins']}/{stats['ties']}/{stats['losses']})")
        md_lines.append(f"| `{_pct_token_label(pct, doc_tokens)}` | " + " | ".join(cells) + " |")

    md_lines.extend(
        [
            "",
            "Each cell shows `mean Delta (better/tie/worse)`. Positive values mean the leaf method lowers error; negative values mean it raises error.",
            "",
            "## Material Quadratic-Weight Onset",
            "",
            f"Smallest quadratic weight where mean `Delta > {onset_eps:.1f}` at `train_docs=512`, `doc_topic_concentration=0.6`, using only the quadratic-weight suites. This means the smallest quadratic weight where the leaf method improves mean absolute error by at least `{onset_eps:.1f}`.",
            "",
            "| Leaf proportion | `tau=1 / d=0.50` | `tau=8 / d=0.11` |",
            "|---:|---:|---:|",
        ]
    )
    for pct in focus_pcts:
        md_lines.append(
            f"| `{_pct_token_label(pct, doc_tokens)}` | "
            f"`{_threshold_label(onset_table[f'pct_{int(round(pct))}_tau_1'])}` | "
            f"`{_threshold_label(onset_table[f'pct_{int(round(pct))}_tau_8'])}` |"
        )
    md_lines.extend(
        [
            "",
            "## Quadratic-Weight Detail Table",
            "",
            "| Leaf proportion | tau | " + " | ".join(f"`quadratic weight={lam:g}`" for lam in lambda_grid) + " |",
            "|---:|---:|" + "---:|" * len(lambda_grid),
        ]
    )
    for pct in focus_pcts:
        for tau in (1.0, 8.0):
            cells = []
            for lam in lambda_grid:
                stats = lambda_stats[(pct, tau, lam)]
                cells.append(f"{stats['delta_mean']:+.2f}")
            md_lines.append(
                f"| `{_pct_token_label(pct, doc_tokens)}` | `{_tau_label(tau)}` | " + " | ".join(cells) + " |"
            )

    md_lines.extend(
        [
            "",
            "## Null-Control Check",
            "",
            "Mean `Delta` at `quadratic weight=0` stays near zero for the coarse settings, as it should. Near-zero means the leaf and pooled methods are effectively tied under the null.",
            "",
            "| Leaf proportion | `tau=1 / d=0.50` | `tau=8 / d=0.11` |",
            "|---:|---:|---:|",
        ]
    )
    for pct in focus_pcts:
        row_tau1 = lambda_stats[(pct, 1.0, 0.0)]
        row_tau8 = lambda_stats[(pct, 8.0, 0.0)]
        md_lines.append(
            f"| `{_pct_token_label(pct, doc_tokens)}` | "
            f"`{row_tau1['delta_mean']:+.2f} ({row_tau1['wins']}/{row_tau1['ties']}/{row_tau1['losses']})` | "
            f"`{row_tau8['delta_mean']:+.2f} ({row_tau8['wins']}/{row_tau8['ties']}/{row_tau8['losses']})` |"
        )

    md_lines.extend(
        [
            "",
            "## Train-Docs Boundary Check",
            "",
            "Mean `Delta` at the boundary slices (`quadratic weight=2`, `doc_topic_concentration=0.6`). The `512` rows come from the tau suites; the `2048` rows come from the dedicated boundary suite.",
            "",
            "| Train docs | Leaf proportion | `tau=8 / d=0.11` | `tau=16 / d=0.06` |",
            "|---:|---:|---:|---:|",
        ]
    )
    for train_docs in (512, 2048):
        for pct in focus_pcts:
            row_tau8 = boundary_stats[(train_docs, pct, 8.0)]
            row_tau16 = boundary_stats[(train_docs, pct, 16.0)]
            md_lines.append(
                f"| `{train_docs}` | `{_pct_token_label(pct, doc_tokens)}` | "
                f"`{row_tau8['delta_mean']:+.2f} ({row_tau8['wins']}/{row_tau8['ties']}/{row_tau8['losses']})` | "
                f"`{row_tau16['delta_mean']:+.2f} ({row_tau16['wins']}/{row_tau16['ties']}/{row_tau16['losses']})` |"
            )
    md_lines.extend(
        [
            "",
            "## Readout",
            "",
            (
                "The paper-facing update is now specific: coarse leaves help up to a point. The best point in this "
                "coarse comparison is `50%`, not `25%`, and definitely not the degenerate `100%` pooled control."
            ),
            "",
            (
                f"The key semantic fix in this report is explicit interpretation of sign. "
                f"`Delta < 0` means worse than pooled, not better. "
                f"For example, the `25%` boundary failure at `tau=16` is `{example_25_tau16['pooled_mean']:.2f} -> {example_25_tau16['leaf_mean']:.2f}`, "
                f"which is clearly a loss."
            ),
            "",
            (
                "That makes the new story stronger, not weaker. The earlier report showed that increasing leaf size "
                "helps. The extension now shows the benefit saturates before the whole document: once there is only one "
                "leaf, the local method has no extra structure left to exploit."
            ),
            "",
        ]
    )

    md_path = output_dir / "tree_relevant_lda_proportion_extension_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    json_path = output_dir / "tree_relevant_lda_proportion_extension_report_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    colors = {25.0: "#2166ac", 50.0: "#1b9e77", 100.0: "#7f7f7f"}

    pdf_path = output_dir / "tree_relevant_lda_proportion_extension_report.pdf"
    with PdfPages(pdf_path) as pdf:
        _paragraph_page(
            pdf,
            title="Tree-Relevant LDA Coarse Leaf-Size Extension",
            paragraphs=[
                f"Snapshot: {args.snapshot_label}. This full report merges the original March 7 follow-up root with the March 9 coarse leaf-size extension.",
                (
                    "The key question is whether the earlier 96-token winner keeps improving when we make sections coarser. "
                    "The answer is no: 50% beats 25% on the moderate boundary, while 100% is exactly the pooled null."
                ),
                "Metric definition: Delta = pooled mean absolute error minus leaf mean absolute error. Positive Delta means the leaf method is better. Negative Delta means it is worse.",
                (
                    f"At quadratic weight=2 and dtc=0.6, the last positive tau is {_threshold_label(last_positive_tau['25'])} for 25%, "
                    f"{_threshold_label(last_positive_tau['50'])} for 50%, and {_threshold_label(last_positive_tau['100'])} "
                    "for 100%."
                ),
            ],
        )

        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
        xs = np.arange(len(suite_names))
        queued = [manifest_counts.get(name, 0) for name in suite_names]
        done = [suite_run_counts.get(name, 0) for name in suite_names]
        axes[0].bar(xs - 0.18, queued, width=0.36, color="#d9d9d9", label="Queued")
        axes[0].bar(xs + 0.18, done, width=0.36, color="#2ca25f", label="Completed")
        axes[0].set_xticks(xs)
        axes[0].set_xticklabels(
            ["tau\norig", "tau\ncoarse", "w_q\norig", "w_q\ncoarse", "dtc\nrobust", "2048\nboundary"],
            fontsize=9,
        )
        axes[0].set_ylabel("Run summaries")
        axes[0].set_title("Coverage by suite")
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.3, axis="y")
        lines = [
            f"baseline root : {args.baseline_root}",
            f"extension root: {args.extension_root}",
            "",
            f"baseline summaries : {len(baseline_runs)}",
            f"extension summaries: {len(extension_runs)}",
            f"merged total       : {len(all_runs)}",
            "",
        ]
        for suite in suite_names:
            lines.append(f"{_suite_display_label(suite)}: {suite_run_counts.get(suite, 0)}/{manifest_counts.get(suite, 0)}")
            lines.append(f"  {_suite_display_purpose(suite, purpose_map)}")
        axes[1].axis("off")
        axes[1].text(0.02, 0.98, "\n".join(lines), family="monospace", fontsize=10, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        xs = np.arange(len(tau_grid))
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
        for pct in focus_pcts:
            means = [tau_stats[(pct, tau)]["delta_mean"] for tau in tau_grid]
            sems = [tau_stats[(pct, tau)]["delta_sem"] for tau in tau_grid]
            rels = [100.0 * tau_stats[(pct, tau)]["relative_gain_mean"] for tau in tau_grid]
            rel_sems = [100.0 * tau_stats[(pct, tau)]["relative_gain_sem"] for tau in tau_grid]
            color = colors[pct]
            label = _pct_token_label(pct, doc_tokens)
            axes[0].plot(xs, means, marker="o", linewidth=2, color=color, label=label)
            axes[0].fill_between(xs, np.array(means) - np.array(sems), np.array(means) + np.array(sems), alpha=0.15, color=color)
            axes[1].plot(xs, rels, marker="o", linewidth=2, color=color, label=label)
            axes[1].fill_between(xs, np.array(rels) - np.array(rel_sems), np.array(rels) + np.array(rel_sems), alpha=0.15, color=color)
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
        axes[0].legend(fontsize=9)
        fig.suptitle("Tau crossover at quadratic weight=2, dtc=0.6, train_docs=512", fontsize=13)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        tau_lines = [
            "Mean Delta = pooled error - per-section error",
            "suite family: tau crossover only",
            "",
            "leaf_pct     tau=.25   tau=.5    tau=1    tau=2    tau=4    tau=8   tau=16   tau=32   tau=64",
        ]
        for pct in focus_pcts:
            row = [_pct_token_label(pct, doc_tokens)]
            for tau in tau_grid:
                stats = tau_stats[(pct, tau)]
                row.append(f"{stats['delta_mean']:+.2f}")
            tau_lines.append("  ".join(f"{item:>10}" for item in row))
        _text_page(pdf, title="Tau Crossover Table", lines=tau_lines, font_size=10)

        example_lines = [
            "Representative absolute-error slices",
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

        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True, sharey=True)
        for ax, tau in zip(axes, (1.0, 8.0)):
            for pct in focus_pcts:
                means = [lambda_stats[(pct, tau, lam)]["delta_mean"] for lam in lambda_grid]
                sems = [lambda_stats[(pct, tau, lam)]["delta_sem"] for lam in lambda_grid]
                color = colors[pct]
                label = _pct_token_label(pct, doc_tokens)
                ax.plot(lambda_grid, means, marker="o", linewidth=2, color=color, label=label)
                ax.fill_between(lambda_grid, np.array(means) - np.array(sems), np.array(means) + np.array(sems), alpha=0.15, color=color)
            ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
            ax.set_title(_tau_label(tau))
            ax.set_xlabel("quadratic weight")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(_delta_axis_label())
        axes[0].legend(fontsize=9)
        fig.suptitle("Quadratic-weight onset at train_docs=512, dtc=0.6", fontsize=13)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        lambda_lines = [
            f"Material onset threshold: mean Delta > {onset_eps:.1f} (leaf improves mean abs error by at least {onset_eps:.1f})",
            "",
            "leaf_pct       tau=1 onset   tau=8 onset   tau=1 w_q=0   tau=8 w_q=0",
        ]
        for pct in focus_pcts:
            row_tau1 = lambda_stats[(pct, 1.0, 0.0)]
            row_tau8 = lambda_stats[(pct, 8.0, 0.0)]
            lambda_lines.append(
                f"{_pct_token_label(pct, doc_tokens):>12}"
                f"{_threshold_label(onset_table[f'pct_{int(round(pct))}_tau_1']):>14}"
                f"{_threshold_label(onset_table[f'pct_{int(round(pct))}_tau_8']):>14}"
                f"{row_tau1['delta_mean']:>15.2f}"
                f"{row_tau8['delta_mean']:>15.2f}"
            )
        _text_page(pdf, title="Quadratic-Weight Threshold Summary", lines=lambda_lines, font_size=10)

        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True, sharey=True)
        train_positions = np.arange(2)
        width = 0.22
        train_labels = ["512", "2048"]
        for ax, tau in zip(axes, (8.0, 16.0)):
            for idx, pct in enumerate(focus_pcts):
                vals = [boundary_stats[(train_docs, pct, tau)]["delta_mean"] for train_docs in (512, 2048)]
                ax.bar(train_positions + (idx - 1) * width, vals, width=width, color=colors[pct], label=_pct_token_label(pct, doc_tokens))
            ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
            ax.set_xticks(train_positions)
            ax.set_xticklabels(train_labels)
            ax.set_xlabel("train_docs")
            ax.set_title(_tau_label(tau))
            ax.grid(alpha=0.3, axis="y")
        axes[0].set_ylabel(_delta_axis_label())
        axes[0].legend(fontsize=8)
        fig.suptitle("Boundary check: 50% stays strongest as train_docs increases", fontsize=13)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        boundary_lines = [
            "Boundary slices at quadratic weight=2, dtc=0.6",
            "",
            "train_docs   leaf_pct        tau=8        tau=16",
        ]
        for train_docs in (512, 2048):
            for pct in focus_pcts:
                row_tau8 = boundary_stats[(train_docs, pct, 8.0)]
                row_tau16 = boundary_stats[(train_docs, pct, 16.0)]
                boundary_lines.append(
                    f"{train_docs:>9}  {_pct_token_label(pct, doc_tokens):>12}"
                    f"{row_tau8['delta_mean']:>12.2f}"
                    f"{row_tau16['delta_mean']:>13.2f}"
                )
        _text_page(pdf, title="Train-Docs Boundary Table", lines=boundary_lines, font_size=10)

        _paragraph_page(
            pdf,
            title="Interpretation",
            paragraphs=[
                "The extension changes the paper claim from a vague monotone story to a threshold story. Making leaves coarser helps until we reach a practical optimum around 50% of the document. Past that point, there is no additional local structure to exploit, and the 100% one-leaf case collapses exactly to pooling.",
                f"The key semantic point is that negative Delta is bad. At train_docs=512 and tau=16, the 25% point is a loss: pooled {example_25_tau16['pooled_mean']:.2f}, leaf {example_25_tau16['leaf_mean']:.2f}, Delta {example_25_tau16['delta_mean']:+.2f}.",
                f"The most important boundary is tau=16. At train_docs=512, the 25% point is already negative ({boundary_stats[(512, 25.0, 16.0)]['delta_mean']:+.2f}), but the 50% point stays positive ({boundary_stats[(512, 50.0, 16.0)]['delta_mean']:+.2f}). At train_docs=2048, the same ordering remains: 25% is still negative, 50% is still positive, and 100% is exactly zero.",
                "That is the full-report answer to the question you raised: yes, the extra coarse datapoints were worth running, because they show the practical optimum is not at the finest useful scale and not at the degenerate whole-document scale. It sits in between.",
            ],
        )

    print(f"wrote_markdown | {md_path}")
    print(f"wrote_summary | {json_path}")
    print(f"wrote_pdf | {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
