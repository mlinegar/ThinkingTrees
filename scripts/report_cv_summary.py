#!/usr/bin/env python3
"""
Generate a consolidated k-fold CV PDF report with tables and figures.

The report includes:
- fold-level metric table
- pooled + cross-fold aggregate metrics table
- pooled predicted vs actual distribution overlay
- pooled predicted vs actual scatter
- per-fold distribution overlays
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except Exception:
        return []
    return rows


def _parse_fold_idx(path: Path) -> Optional[int]:
    name = path.name
    if not name.startswith("fold_"):
        return None
    tail = name.split("_", 1)[-1]
    if not tail.isdigit():
        return None
    return int(tail)


def _pearson_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    if den_x <= 0.0 or den_y <= 0.0:
        return None
    return num / math.sqrt(den_x * den_y)


def _mean_std(values: Iterable[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None, None
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    return float(mean), float(math.sqrt(var))


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _fmt_pct(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}%"


def _resolve_task_scale(task_name: Optional[str]) -> Tuple[Callable[[float], float], str]:
    if not task_name:
        return (lambda value: float(value), "score (normalized)")
    try:
        from src.tasks import get_task

        task = get_task(str(task_name))
        scale = getattr(task, "scale", None)
        if scale is not None:
            label = f"{scale.name} ({float(scale.min_value):.0f}..{float(scale.max_value):.0f})"
        else:
            label = "score (normalized)"

        def _to_scale(value: float) -> float:
            denorm = task.denormalize_score(float(value))
            return float(denorm if denorm is not None else value)

        return _to_scale, label
    except Exception:
        return (lambda value: float(value), "score (normalized)")


def _collect_rows(cv_dir: Path, split: str) -> Dict[int, List[Tuple[float, float]]]:
    rows_by_fold: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    folds_dir = cv_dir / "folds"
    for fold_dir in sorted(folds_dir.glob("fold_*")):
        fold_idx = _parse_fold_idx(fold_dir)
        if fold_idx is None:
            continue
        report_path = fold_dir / f"{split}_score_report.jsonl"
        if not report_path.exists():
            continue
        for row in _load_jsonl(report_path):
            p = _as_float(row.get("predicted"))
            a = _as_float(row.get("actual"))
            if p is None or a is None:
                continue
            rows_by_fold[int(fold_idx)].append((float(p), float(a)))
    return rows_by_fold


def _metrics_from_pairs(pairs: List[Tuple[float, float]]) -> Dict[str, Optional[float]]:
    if not pairs:
        return {
            "n": 0,
            "mae": None,
            "pearson_r": None,
            "within_5pct": None,
            "within_10pct": None,
            "pred_mean": None,
            "actual_mean": None,
        }
    preds = [p for p, _ in pairs]
    acts = [a for _, a in pairs]
    errs = [abs(p - a) for p, a in pairs]
    within_5 = 100.0 * sum(1 for e in errs if e <= 0.05) / len(errs)
    within_10 = 100.0 * sum(1 for e in errs if e <= 0.10) / len(errs)
    return {
        "n": float(len(pairs)),
        "mae": sum(errs) / len(errs),
        "pearson_r": _pearson_corr(preds, acts),
        "within_5pct": within_5,
        "within_10pct": within_10,
        "pred_mean": sum(preds) / len(preds),
        "actual_mean": sum(acts) / len(acts),
    }


def _text_page(pdf: PdfPages, title: str, lines: List[str], font_size: int = 10) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(title, fontsize=14, y=0.98)
    text = "\n".join(lines)
    fig.text(0.03, 0.95, text, va="top", ha="left", family="monospace", fontsize=font_size)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _table_page(
    pdf: PdfPages,
    title: str,
    col_labels: List[str],
    rows: List[List[str]],
    font_size: int = 9,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=14, loc="left", pad=12)
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.4)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _distribution_overlay_page(
    pdf: PdfPages,
    *,
    pairs: List[Tuple[float, float]],
    to_scale: Callable[[float], float],
    scale_label: str,
    title: str,
    bins: int,
) -> None:
    preds = [to_scale(p) for p, _ in pairs]
    acts = [to_scale(a) for _, a in pairs]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(acts, bins=bins, alpha=0.55, color="#4c78a8", label="Actual", density=True)
    ax.hist(preds, bins=bins, alpha=0.55, color="#f58518", label="Predicted", density=True)
    ax.set_title(title)
    ax.set_xlabel(scale_label)
    ax.set_ylabel("Density")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _scatter_page(
    pdf: PdfPages,
    *,
    pairs: List[Tuple[float, float]],
    to_scale: Callable[[float], float],
    scale_label: str,
    title: str,
) -> None:
    preds = [to_scale(p) for p, _ in pairs]
    acts = [to_scale(a) for _, a in pairs]
    lo = min(min(preds), min(acts))
    hi = max(max(preds), max(acts))
    pad = max(1e-6, 0.03 * (hi - lo if hi > lo else 1.0))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(acts, preds, alpha=0.55, s=14, color="#4c78a8")
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="#e45756", linewidth=1.5)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(title)
    ax.set_xlabel(f"Actual {scale_label}")
    ax.set_ylabel(f"Predicted {scale_label}")
    ax.grid(True, alpha=0.2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _per_fold_overlay_page(
    pdf: PdfPages,
    *,
    rows_by_fold: Dict[int, List[Tuple[float, float]]],
    to_scale: Callable[[float], float],
    scale_label: str,
    title: str,
    bins: int,
) -> None:
    fold_ids = sorted(rows_by_fold.keys())
    n = len(fold_ids)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, 3.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes_list = [axes]
    elif nrows == 1:
        axes_list = list(axes)
    elif ncols == 1:
        axes_list = list(axes)
    else:
        axes_list = [ax for row in axes for ax in row]
    for ax, fold_idx in zip(axes_list, fold_ids):
        pairs = rows_by_fold[fold_idx]
        preds = [to_scale(p) for p, _ in pairs]
        acts = [to_scale(a) for _, a in pairs]
        ax.hist(acts, bins=bins, alpha=0.5, color="#4c78a8", density=True, label="Actual")
        ax.hist(preds, bins=bins, alpha=0.5, color="#f58518", density=True, label="Predicted")
        ax.set_title(f"Fold {fold_idx} (n={len(pairs)})")
        ax.set_xlabel(scale_label)
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.2)
    for ax in axes_list[n:]:
        ax.axis("off")
    if axes_list:
        axes_list[0].legend(loc="best")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate consolidated CV PDF report from fold score reports.")
    parser.add_argument("--cv-dir", type=Path, required=True, help="CV directory containing folds/ and cv_summary.json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--pdf-path", type=Path, default=None, help="Output PDF path (default: <cv-dir>/cv_report.pdf)")
    parser.add_argument("--bins", type=int, default=28, help="Histogram bin count")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cv_dir = args.cv_dir
    if not cv_dir.exists():
        raise SystemExit(f"cv-dir does not exist: {cv_dir}")

    summary_path = cv_dir / "cv_summary.json"
    summary = _load_json(summary_path) or {}
    task_name = summary.get("task")
    to_scale, scale_label = _resolve_task_scale(task_name)

    rows_by_fold = _collect_rows(cv_dir, split=str(args.split))
    if not rows_by_fold:
        raise SystemExit(f"No fold reports found for split='{args.split}' under {cv_dir / 'folds'}")

    all_pairs: List[Tuple[float, float]] = []
    fold_metric_rows: List[Dict[str, Any]] = []
    for fold_idx in sorted(rows_by_fold.keys()):
        pairs = rows_by_fold[fold_idx]
        all_pairs.extend(pairs)
        metrics = _metrics_from_pairs(pairs)
        fold_metric_rows.append({"fold": fold_idx, **metrics})

    pooled_metrics = _metrics_from_pairs(all_pairs)
    fold_mae_mean, fold_mae_std = _mean_std(row.get("mae") for row in fold_metric_rows)
    fold_pearson_mean, fold_pearson_std = _mean_std(row.get("pearson_r") for row in fold_metric_rows)
    fold_w10_mean, fold_w10_std = _mean_std(row.get("within_10pct") for row in fold_metric_rows)

    pdf_path = args.pdf_path if args.pdf_path is not None else (cv_dir / "cv_report.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        overview_lines = [
            f"generated_utc: {datetime.now(timezone.utc).isoformat()}",
            f"cv_dir: {cv_dir.resolve()}",
            f"split: {args.split}",
            f"task: {summary.get('task', 'n/a')}",
            f"dataset: {summary.get('dataset', 'n/a')}",
            f"k: {summary.get('k', len(rows_by_fold))}",
            f"n_samples: {summary.get('n_samples', 'n/a')}",
            f"folds_with_reports: {len(rows_by_fold)}",
            f"rows_total: {len(all_pairs)}",
            f"scale: {scale_label}",
        ]
        _text_page(pdf, title="CV Report — Overview", lines=overview_lines, font_size=10)

        fold_table = []
        for row in fold_metric_rows:
            fold_table.append(
                [
                    str(row["fold"]),
                    str(int(row.get("n") or 0)),
                    _fmt(row.get("mae"), digits=4),
                    _fmt(row.get("pearson_r"), digits=4),
                    _fmt_pct(row.get("within_10pct"), digits=1),
                ]
            )
        _table_page(
            pdf,
            title=f"Fold Metrics ({args.split})",
            col_labels=["Fold", "n", "MAE", "Pearson r", "Within 10%"],
            rows=fold_table,
            font_size=10,
        )

        aggregate_table = [
            ["Pooled", str(int(pooled_metrics.get("n") or 0)), _fmt(pooled_metrics.get("mae"), 4), _fmt(pooled_metrics.get("pearson_r"), 4), _fmt_pct(pooled_metrics.get("within_10pct"), 1)],
            ["Fold mean ± std", "-", f"{_fmt(fold_mae_mean, 4)} ± {_fmt(fold_mae_std, 4)}", f"{_fmt(fold_pearson_mean, 4)} ± {_fmt(fold_pearson_std, 4)}", f"{_fmt(fold_w10_mean, 1)}% ± {_fmt(fold_w10_std, 1)}%"],
        ]
        _table_page(
            pdf,
            title=f"Aggregate Metrics ({args.split})",
            col_labels=["Scope", "n", "MAE", "Pearson r", "Within 10%"],
            rows=aggregate_table,
            font_size=10,
        )

        _distribution_overlay_page(
            pdf,
            pairs=all_pairs,
            to_scale=to_scale,
            scale_label=scale_label,
            title=f"Pooled {args.split}: Predicted vs Actual Distribution",
            bins=max(8, int(args.bins)),
        )
        _scatter_page(
            pdf,
            pairs=all_pairs,
            to_scale=to_scale,
            scale_label=scale_label,
            title=f"Pooled {args.split}: Predicted vs Actual",
        )
        _per_fold_overlay_page(
            pdf,
            rows_by_fold=rows_by_fold,
            to_scale=to_scale,
            scale_label=scale_label,
            title=f"Per-Fold Distribution Overlay ({args.split})",
            bins=max(8, int(args.bins)),
        )

    logger.info("Wrote %s", pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
