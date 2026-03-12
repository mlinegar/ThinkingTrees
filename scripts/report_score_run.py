#!/usr/bin/env python3
"""
Generate a PDF report from *_score_report.jsonl outputs.

This is intended for both sanity-checking and producing publication-style
diagnostic artifacts:
- metric tables (overall + honest / three-layer splits when present)
- predicted vs actual scatter (+ honest split coloring when available)
- error histogram + calibration curve
- prediction collapse diagnostics (value frequency)
- example document leaf breakdown (from Phase 1 checkpoints, if available)

Example:
  ./venv/bin/python scripts/report_score_run.py \
    --output-dir outputs/manifesto_overnight_20260222_084204 \
    --splits train test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import re
import sys
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)

# Ensure repo root is on sys.path so we can unpickle Phase 1 artifacts that
# reference modules under `src.*`, even when this file is executed as
# `python scripts/report_score_run.py` (sys.path[0] = scripts/).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _format_float(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if num != num:
        return "n/a"
    return f"{num:.{digits}f}"


def _format_pct(value: Any, *, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if num != num:
        return "n/a"
    return f"{num:.{digits}f}%"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _safe_get(obj: Any, keys: List[str], default: Any = None) -> Any:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def _as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _to_raw(score_norm: float) -> float:
    """Map normalized [0,1] to RILE [-100,100]."""
    return float(score_norm) * 200.0 - 100.0


def _wrap(text: str, width: int) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return "\n".join(textwrap.fill(line, width=width) for line in text.splitlines())


def _one_line_preview(text: str, *, max_len: int = 80) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= max_len:
        return collapsed
    if max_len <= 1:
        return "…"
    return collapsed[: max_len - 1] + "…"


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """
    Best-effort cleanup for models that emit `<think>...</think>` blocks.

    We prefer keeping only the content after the final `</think>` marker, when present.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    if "</think>" in raw.lower():
        parts = re.split(r"</think>", raw, flags=re.IGNORECASE)
        tail = parts[-1].strip()
        return tail or raw
    cleaned = _THINK_BLOCK_RE.sub("", raw)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or raw


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


def _average_ranks(values: List[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks: List[float] = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = ((i + 1) + j) / 2.0
        for k in range(i, j):
            orig_idx = indexed[k][0]
            ranks[orig_idx] = avg_rank
        i = j
    return ranks


def _spearman_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson_corr(_average_ranks(xs), _average_ranks(ys))


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    preds = [_as_float(r.get("predicted")) for r in rows]
    acts = [_as_float(r.get("actual")) for r in rows]
    vals = [(p, a) for p, a in zip(preds, acts) if p is not None and a is not None]
    if not vals:
        return {"n": 0}

    xs = [p for p, _ in vals]
    ys = [a for _, a in vals]
    errors = [abs(p - a) for p, a in vals]

    def _mean(values: List[float]) -> float:
        return sum(values) / max(1, len(values))

    def _std(values: List[float]) -> float:
        if not values:
            return 0.0
        mu = _mean(values)
        return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))

    unique_preds = len(set(round(p, 12) for p in xs))
    top_preds = Counter(xs).most_common(8)

    within_5 = sum(1 for e in errors if e <= 0.05) / len(errors) * 100
    within_10 = sum(1 for e in errors if e <= 0.10) / len(errors) * 100
    neutral = 0.5
    same_side = 0
    for pred, actual in vals:
        pred_delta = float(pred) - float(neutral)
        actual_delta = float(actual) - float(neutral)
        # Strict metric: exact-neutral predictions always count as wrong.
        if abs(pred_delta) <= 1e-9:
            continue
        if pred_delta * actual_delta > 0.0:
            same_side += 1
    same_side_pct = (same_side / len(vals)) * 100.0

    return {
        "n": len(vals),
        "mae": _mean(errors),
        "pearson_r": _pearson_corr(xs, ys),
        "spearman_rho": _spearman_corr(xs, ys),
        "pred_mean": _mean(xs),
        "pred_std": _std(xs),
        "actual_mean": _mean(ys),
        "actual_std": _std(ys),
        "within_5pct": within_5,
        "within_10pct": within_10,
        "same_side_of_neutral_pct": same_side_pct,
        "unique_preds": unique_preds,
        "top_preds": top_preds,
    }


def _render_text_page(pdf: PdfPages, *, title: str, lines: List[str], font_size: int = 10) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title(title, pad=12)
    ax.text(0.01, 0.98, "\n".join(lines), family="monospace", fontsize=font_size, va="top")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_table_page(
    pdf: PdfPages,
    *,
    title: str,
    col_labels: List[str],
    cell_text: List[List[str]],
    font_size: int = 9,
    scale_y: float = 1.4,
) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title(title, pad=12)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, scale_y)
    try:
        table.auto_set_column_width(col=list(range(len(col_labels))))
    except Exception:
        pass

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _overview_page(
    pdf: PdfPages,
    *,
    output_dir: Path,
    config: Optional[Dict[str, Any]],
    final_stats: Optional[Dict[str, Any]],
    summaries: Dict[str, Dict[str, Any]],
) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")

    header = f"Score Report — {output_dir.name}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = [header, f"Generated: {timestamp}", ""]

    if config:
        task = config.get("task")
        dataset = config.get("dataset")
        train_n = config.get("train_samples")
        val_n = config.get("val_samples")
        test_n = config.get("test_samples")
        max_chunk_chars = config.get("max_chunk_chars")
        max_metric_calls = config.get("max_metric_calls")
        lines.append(f"Task: {task} | Dataset: {dataset}")
        lines.append(
            f"Samples: train={train_n} val={val_n} test={test_n} | max_chunk_chars={max_chunk_chars} | max_metric_calls={max_metric_calls}"
        )
        lines.append("")

    for split, summary in summaries.items():
        if summary.get("n", 0) <= 0:
            continue
        lines.append(
            f"[{split}] n={summary['n']} | MAE(norm)={summary['mae']:.3f} (~{summary['mae']*200.0:.1f} RILE pts) "
            f"| within10%={summary['within_10pct']:.1f}% | same-side={summary.get('same_side_of_neutral_pct', 0.0):.1f}% "
            f"| Pearson r={summary['pearson_r'] if summary.get('pearson_r') is not None else 'n/a'}"
        )
        lines.append(
            f"  pred μ/σ={summary['pred_mean']:.3f}/{summary['pred_std']:.3f} | actual μ/σ={summary['actual_mean']:.3f}/{summary['actual_std']:.3f}"
        )
        top_preds = summary.get("top_preds") or []
        if top_preds:
            rendered = ", ".join(f"{_to_raw(v):.0f}:{c}" for v, c in top_preds[:6])
            lines.append(f"  unique preds: {summary['unique_preds']} | top preds (RILE): {rendered}")
        lines.append("")

    if final_stats and isinstance(final_stats, dict):
        completed = final_stats.get("completed_at")
        success = final_stats.get("success")
        if completed or success is not None:
            lines.append(f"Run completed_at={completed} success={success}")

    ax.text(0.01, 0.98, "\n".join(lines), va="top", fontsize=11, family="monospace")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _metrics_page(
    pdf: PdfPages,
    *,
    final_stats: Optional[Dict[str, Any]],
    rows_by_split: Dict[str, List[Dict[str, Any]]],
    splits: List[str],
) -> None:
    # Overall split table (derived from rows to avoid relying on final_stats schema).
    col_labels = [
        "Split",
        "N",
        "MAE (norm)",
        "MAE (RILE)",
        "Pearson r",
        "Spearman ρ",
        "Within 5%",
        "Within 10%",
        "Same-side",
        "Unique preds",
    ]
    cell_text: List[List[str]] = []
    for split in splits:
        rows = rows_by_split.get(split, [])
        summary = _summarize_rows(rows)
        if summary.get("n", 0) <= 0:
            continue
        cell_text.append(
            [
                split,
                str(summary.get("n")),
                _format_float(summary.get("mae"), digits=4),
                _format_float((summary.get("mae") or 0.0) * 200.0, digits=1),
                _format_float(summary.get("pearson_r"), digits=3),
                _format_float(summary.get("spearman_rho"), digits=3),
                _format_pct(summary.get("within_5pct"), digits=1),
                _format_pct(summary.get("within_10pct"), digits=1),
                _format_pct(summary.get("same_side_of_neutral_pct"), digits=1),
                str(summary.get("unique_preds")),
            ]
        )
    if cell_text:
        _render_table_page(pdf, title="Metrics — Overall", col_labels=col_labels, cell_text=cell_text, font_size=9)

    # Honest split table (if present in row data).
    honest_rows: List[List[str]] = []
    for split in splits:
        rows = rows_by_split.get(split, [])
        for role in ("boundary", "evaluation"):
            sub = [r for r in rows if str(r.get("honest_chunk_split") or "").strip().lower() == role]
            summary = _summarize_rows(sub)
            if summary.get("n", 0) <= 0:
                continue
            honest_rows.append(
                [
                    split,
                    role,
                    str(summary.get("n")),
                    _format_float(summary.get("mae"), digits=4),
                    _format_float((summary.get("mae") or 0.0) * 200.0, digits=1),
                    _format_float(summary.get("pearson_r"), digits=3),
                    _format_pct(summary.get("within_10pct"), digits=1),
                    _format_pct(summary.get("same_side_of_neutral_pct"), digits=1),
                ]
            )

    if honest_rows:
        _render_table_page(
            pdf,
            title="Metrics — Honest Split",
            col_labels=["Split", "Honest role", "N", "MAE (norm)", "MAE (RILE)", "Pearson r", "Within 10%", "Same-side"],
            cell_text=honest_rows,
            font_size=9,
        )

    # Three-layer joint_eval (when present in final_stats).
    if isinstance(final_stats, dict):
        three_rows: List[List[str]] = []
        for split in splits:
            block = final_stats.get(split, {})
            if not isinstance(block, dict):
                continue
            three = block.get("three_layer_honesty_metrics")
            if not isinstance(three, dict) or not three.get("enabled"):
                continue
            joint = three.get("joint_eval", {})
            if not isinstance(joint, dict):
                continue
            three_rows.append(
                [
                    split,
                    str(joint.get("n_evaluated") or "n/a"),
                    _format_float(joint.get("mae"), digits=4),
                    _format_float((joint.get("mae") or 0.0) * 200.0, digits=1),
                ]
            )
        if three_rows:
            _render_table_page(
                pdf,
                title="Metrics — Three-Layer (joint_eval)",
                col_labels=["Split", "N", "MAE (norm)", "MAE (RILE)"],
                cell_text=three_rows,
                font_size=9,
            )


def _scatter_page(pdf: PdfPages, *, rows: List[Dict[str, Any]], title: str) -> None:
    preds = []
    acts = []
    colors = []
    labels = []
    split_color = {
        "boundary": "#d55e00",
        "evaluation": "#0072b2",
    }

    for r in rows:
        p = _as_float(r.get("predicted"))
        a = _as_float(r.get("actual"))
        if p is None or a is None:
            continue
        preds.append(_to_raw(p))
        acts.append(_to_raw(a))
        split = str(r.get("honest_chunk_split") or "").strip().lower()
        colors.append(split_color.get(split, "#444444"))
        labels.append(split or "unknown")

    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(preds, acts, s=18, alpha=0.8, c=colors, linewidths=0.0)
    ax.set_title(title)
    ax.set_xlabel("Predicted (RILE)")
    ax.set_ylabel("Actual (RILE)")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.plot([-100, 100], [-100, 100], linestyle="--", color="#666666", linewidth=1.0)

    present = {lbl for lbl in labels if lbl in split_color}
    if present:
        handles = []
        for lbl in sorted(present):
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=lbl,
                    markerfacecolor=split_color[lbl],
                    markersize=7,
                )
            )
        ax.legend(handles=handles, loc="lower right", frameon=True)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _hist_page(pdf: PdfPages, *, rows: List[Dict[str, Any]], title: str) -> None:
    errors = []
    for r in rows:
        p = _as_float(r.get("predicted"))
        a = _as_float(r.get("actual"))
        if p is None or a is None:
            continue
        errors.append(abs(_to_raw(p) - _to_raw(a)))

    fig = plt.figure(figsize=(8.5, 6.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(errors, bins=24, color="#4c78a8", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Absolute error (RILE points)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _dist_overlay_page(pdf: PdfPages, *, rows: List[Dict[str, Any]], title: str) -> None:
    preds: List[float] = []
    acts: List[float] = []
    for r in rows:
        p = _as_float(r.get("predicted"))
        a = _as_float(r.get("actual"))
        if p is None or a is None:
            continue
        preds.append(_to_raw(float(p)))
        acts.append(_to_raw(float(a)))

    if not preds or not acts:
        return

    fig = plt.figure(figsize=(8.5, 6.0))
    ax = fig.add_subplot(1, 1, 1)
    bins = list(range(-100, 105, 10))
    ax.hist(acts, bins=bins, alpha=0.55, color="#4c78a8", label="Actual", density=True)
    ax.hist(preds, bins=bins, alpha=0.55, color="#f58518", label="Predicted", density=True)
    ax.set_title(title)
    ax.set_xlabel("RILE")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=True)
    ax.set_xlim(-100, 100)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _residuals_page(pdf: PdfPages, *, rows: List[Dict[str, Any]], title: str) -> None:
    xs: List[float] = []
    residuals: List[float] = []
    for r in rows:
        p = _as_float(r.get("predicted"))
        a = _as_float(r.get("actual"))
        if p is None or a is None:
            continue
        pr = _to_raw(float(p))
        ar = _to_raw(float(a))
        xs.append(ar)
        residuals.append(pr - ar)

    if not xs:
        return

    fig = plt.figure(figsize=(8.5, 6.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(xs, residuals, s=18, alpha=0.75, c="#54a24b", linewidths=0.0)
    ax.axhline(0.0, linestyle="--", color="#666666", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Actual (RILE)")
    ax.set_ylabel("Residual (Pred − Actual, RILE)")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-100, 100)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _binned_mae_page(pdf: PdfPages, *, rows: List[Dict[str, Any]], title: str, bins: int = 10) -> None:
    pairs: List[Tuple[float, float]] = []
    for r in rows:
        p = _as_float(r.get("predicted"))
        a = _as_float(r.get("actual"))
        if p is None or a is None:
            continue
        pairs.append((_to_raw(float(p)), _to_raw(float(a))))

    if not pairs:
        return

    acts = [a for _, a in pairs]
    lo = min(acts)
    hi = max(acts)
    if lo == hi:
        return
    edges = [lo + (hi - lo) * (i / bins) for i in range(bins + 1)]

    centers: List[float] = []
    maes: List[float] = []
    ns: List[int] = []
    for i in range(bins):
        a0, a1 = edges[i], edges[i + 1]
        bucket = [(p, a) for p, a in pairs if (a >= a0 and (a < a1 or (i == bins - 1 and a <= a1)))]
        if not bucket:
            continue
        centers.append((a0 + a1) / 2.0)
        maes.append(sum(abs(p - a) for p, a in bucket) / len(bucket))
        ns.append(len(bucket))

    fig = plt.figure(figsize=(8.5, 6.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar([f"{c:.0f}" for c in centers], maes, color="#b279a2", alpha=0.9)
    for x, mae, n in zip(range(len(centers)), maes, ns):
        ax.text(x, mae, f"n={n}", ha="center", va="bottom", fontsize=8, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Actual bin center (RILE)")
    ax.set_ylabel("Mean absolute error (RILE points)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=45)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _calibration_page(pdf: PdfPages, *, rows: List[Dict[str, Any]], title: str, bins: int = 10) -> None:
    xs: List[float] = []
    ys: List[float] = []
    for r in rows:
        p = _as_float(r.get("predicted"))
        a = _as_float(r.get("actual"))
        if p is None or a is None:
            continue
        xs.append(float(p))
        ys.append(float(a))

    if not xs:
        return

    edges = [i / bins for i in range(bins + 1)]
    bucket_x: List[float] = []
    bucket_y: List[float] = []
    bucket_n: List[int] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        idx = [j for j, x in enumerate(xs) if (x >= lo and (x < hi or (i == bins - 1 and x <= hi)))]
        if not idx:
            continue
        px = sum(xs[j] for j in idx) / len(idx)
        ay = sum(ys[j] for j in idx) / len(idx)
        bucket_x.append(_to_raw(px))
        bucket_y.append(_to_raw(ay))
        bucket_n.append(len(idx))

    fig = plt.figure(figsize=(8.5, 6.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(bucket_x, bucket_y, marker="o", color="#54a24b", linewidth=2)
    for x, y, n in zip(bucket_x, bucket_y, bucket_n):
        ax.text(x, y, f" n={n}", fontsize=8, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Predicted (bin mean, RILE)")
    ax.set_ylabel("Actual (bin mean, RILE)")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.plot([-100, 100], [-100, 100], linestyle="--", color="#666666", linewidth=1.0)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _top_errors_page(pdf: PdfPages, *, rows: List[Dict[str, Any]], title: str, k: int = 15) -> None:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for r in rows:
        p = _as_float(r.get("predicted"))
        a = _as_float(r.get("actual"))
        if p is None or a is None:
            continue
        err = abs(_to_raw(p) - _to_raw(a))
        scored.append((err, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    scored = scored[: max(1, int(k))]

    lines = []
    lines.append("doc_id | pred | actual | abs_err | honest_split")
    lines.append("-" * 72)
    for err, r in scored:
        doc_id = str(r.get("doc_id") or "")
        pred = _to_raw(float(r.get("predicted")))
        act = _to_raw(float(r.get("actual")))
        split = str(r.get("honest_chunk_split") or "")
        lines.append(f"{doc_id} | {pred:6.1f} | {act:6.1f} | {err:7.1f} | {split}")

    _render_text_page(pdf, title=title, lines=lines, font_size=9)


def _prediction_collapse_page(pdf: PdfPages, *, rows_by_split: Dict[str, List[Dict[str, Any]]], splits: List[str]) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    nrows = max(1, len(splits))
    for i, split in enumerate(splits, start=1):
        rows = rows_by_split.get(split, [])
        preds: List[float] = []
        for r in rows:
            p = _as_float(r.get("predicted"))
            if p is None:
                continue
            preds.append(float(round(_to_raw(p))))
        counter = Counter(preds)
        xs = sorted(counter.keys())
        ys = [counter[x] for x in xs]

        ax = fig.add_subplot(nrows, 1, i)
        ax.bar([str(int(x)) for x in xs], ys, color="#9c755f", alpha=0.9)
        ax.set_title(f"{split}: Predicted Value Frequency (RILE, rounded)")
        ax.set_xlabel("Predicted value")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=45)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _choose_example_id(
    rows: List[Dict[str, Any]],
    *,
    mode: str,
    doc_stats: Optional[Dict[str, Tuple[int, int]]] = None,
    min_leaves: int = 0,
) -> Optional[str]:
    if not rows:
        return None
    mode = str(mode or "central").strip().lower()

    scored: List[Tuple[Tuple[float, int, int], str]] = []
    for r in rows:
        doc_id = str(r.get("doc_id") or "").strip()
        p = _as_float(r.get("predicted"))
        a = _as_float(r.get("actual"))
        if not doc_id or p is None or a is None:
            continue
        err = abs(p - a)
        leaves = 0
        length = 0
        if doc_stats is not None:
            stat = doc_stats.get(doc_id)
            if stat is None:
                if min_leaves > 0:
                    continue
            else:
                leaves, length = int(stat[0]), int(stat[1])
                if leaves < int(min_leaves):
                    continue
        if mode == "central":
            key_main = abs(a - 0.5)
        elif mode == "worst":
            key_main = -err
        elif mode == "best":
            key_main = err
        else:
            key_main = abs(a - 0.5)
        scored.append(((float(key_main), -leaves, -length), doc_id))

    if not scored:
        return None
    scored.sort(key=lambda t: t[0])
    return scored[0][1]


def _load_phase1_results(output_dir: Path) -> Tuple[List[Any], List[Any]]:
    phase1_path = output_dir / "checkpoints" / "phase1_data.pkl"
    if not phase1_path.exists():
        return [], []
    with open(phase1_path, "rb") as f:
        data = pickle.load(f)
    train = data.get("train_results", []) or []
    val = data.get("val_results", []) or []
    return list(train), list(val)


def _phase1_doc_stats(results: List[Any]) -> Dict[str, Tuple[int, int]]:
    stats: Dict[str, Tuple[int, int]] = {}
    for r in results:
        if r is None or getattr(r, "error", None):
            continue
        doc_id = str(getattr(r, "doc_id", "") or "").strip()
        if not doc_id:
            continue
        leaves = int(getattr(r, "tree_leaves", 0) or 0)
        length = int(getattr(r, "original_length", 0) or 0)
        stats[doc_id] = (leaves, length)
    return stats


def _rows_from_phase1(results: List[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in results:
        if r is None or getattr(r, "error", None):
            continue
        doc_id = str(getattr(r, "doc_id", "") or "").strip()
        if not doc_id:
            continue
        p = _as_float(getattr(r, "estimated_score", None))
        a = _as_float(getattr(r, "reference_score", None))
        if p is None or a is None:
            continue
        rows.append({"doc_id": doc_id, "predicted": p, "actual": a})
    return rows


def _find_result_by_doc_id(results: List[Any], doc_id: str) -> Optional[Any]:
    doc_id = str(doc_id).strip()
    if not doc_id:
        return None
    for r in results:
        if r is None or getattr(r, "error", None):
            continue
        if str(getattr(r, "doc_id", "")).strip() == doc_id:
            return r
    return None


def _resolve_leaf_bounds(result: Any, *, max_chunk_chars: Optional[int]) -> List[Dict[str, Any]]:
    meta = getattr(result, "metadata", {}) or {}
    if isinstance(meta, dict):
        plan = meta.get("tree_plan")
        if isinstance(plan, dict):
            leaf_nodes = plan.get("leaf_nodes")
            if isinstance(leaf_nodes, list) and leaf_nodes:
                nodes = [n for n in leaf_nodes if isinstance(n, dict)]
                nodes.sort(key=lambda n: int(n.get("chunk_index", 0)))
                return nodes

    original = str(getattr(result, "original_content", "") or "")
    leaf_summaries = list(getattr(result, "leaf_summaries", []) or [])
    if not original or not leaf_summaries:
        return []

    if max_chunk_chars is None:
        return []

    try:
        from src.preprocessing.chunker import chunk_for_ops
    except Exception:
        return []

    chunks = chunk_for_ops(original, max_chars=int(max_chunk_chars), strategy="axis")
    nodes = [
        {
            "chunk_index": int(getattr(chunk, "chunk_index", i)),
            "start_char": int(getattr(chunk, "start_char", 0)),
            "end_char": int(getattr(chunk, "end_char", 0)),
        }
        for i, chunk in enumerate(chunks)
    ]
    nodes.sort(key=lambda n: int(n.get("chunk_index", 0)))
    return nodes


def _example_leaf_breakdown_pages(
    pdf: PdfPages,
    *,
    example_result: Any,
    example_row: Optional[Dict[str, Any]],
    max_leaves: int,
    wrap_width: int,
    max_chunk_chars: Optional[int],
) -> None:
    doc_id = str(getattr(example_result, "doc_id", "")).strip()
    meta = getattr(example_result, "metadata", {}) or {}
    if not isinstance(meta, dict):
        meta = {}

    reference_norm = _as_float(getattr(example_result, "reference_score", None))
    predicted_norm = _as_float(getattr(example_result, "estimated_score", None))
    if example_row:
        predicted_norm = _as_float(example_row.get("predicted")) if predicted_norm is None else predicted_norm
        reference_norm = _as_float(example_row.get("actual")) if reference_norm is None else reference_norm

    lines: List[str] = []
    lines.append(f"doc_id: {doc_id}")
    party = meta.get("party_abbrev") or meta.get("party_name")
    country = meta.get("country_name")
    year = meta.get("year")
    if party or country or year:
        lines.append(f"meta: {country or ''} {year or ''} {party or ''}".strip())
    if reference_norm is not None:
        lines.append(f"actual:    norm={reference_norm:.3f}  RILE={_to_raw(reference_norm):.1f}")
    if predicted_norm is not None:
        lines.append(f"predicted: norm={predicted_norm:.3f}  RILE={_to_raw(predicted_norm):.1f}")
    if reference_norm is not None and predicted_norm is not None:
        lines.append(
            f"abs error: norm={abs(predicted_norm-reference_norm):.3f}  RILE={abs(_to_raw(predicted_norm)-_to_raw(reference_norm)):.1f}"
        )
    lines.append("")
    lines.append(f"original_length chars: {getattr(example_result, 'original_length', None)}")
    lines.append(f"final_summary chars:   {len(getattr(example_result, 'final_summary', '') or '')}")
    lines.append(f"tree_leaves:           {getattr(example_result, 'tree_leaves', None)}")
    lines.append("")

    final_summary = _strip_think(str(getattr(example_result, "final_summary", "") or ""))
    lines.append("FINAL SUMMARY (truncated for PDF):")
    lines.append("-" * 72)
    lines.append(_wrap(final_summary[:8000] + ("…" if len(final_summary) > 8000 else ""), width=wrap_width))
    _render_text_page(pdf, title="Example Document — Overview", lines=lines, font_size=9)

    tree_plan = meta.get("tree_plan") if isinstance(meta, dict) else None
    if isinstance(tree_plan, dict):
        tp_lines: List[str] = []
        tp_lines.append(f"doc_id: {doc_id}")
        tp_lines.append(f"root_id: {tree_plan.get('root_id')}")
        tp_lines.append(
            f"leaf_count: {tree_plan.get('leaf_count')} | merge_count: {tree_plan.get('merge_count')} | levels: {len(tree_plan.get('levels') or [])}"
        )
        tp_lines.append("")
        levels = tree_plan.get("levels")
        if isinstance(levels, list) and levels:
            tp_lines.append("LEVELS (node ids):")
            for i, level in enumerate(levels[:10]):
                if not isinstance(level, list):
                    continue
                rendered = ", ".join(str(x) for x in level[:12])
                if len(level) > 12:
                    rendered += ", …"
                tp_lines.append(f"  L{i}: {rendered}")
            if len(levels) > 10:
                tp_lines.append("  …")
            tp_lines.append("")
        edges = tree_plan.get("edges")
        if isinstance(edges, list) and edges:
            tp_lines.append("EDGES (parent <- left + right):")
            for e in edges[:40]:
                if not isinstance(e, dict):
                    continue
                tp_lines.append(f"  {e.get('parent')} <- {e.get('left')} + {e.get('right')}")
            if len(edges) > 40:
                tp_lines.append("  …")
        _render_text_page(pdf, title="Example Document — Tree Plan", lines=tp_lines, font_size=9)

    original = str(getattr(example_result, "original_content", "") or "")
    leaf_summaries = list(getattr(example_result, "leaf_summaries", []) or [])
    leaf_nodes = _resolve_leaf_bounds(example_result, max_chunk_chars=max_chunk_chars)

    if not leaf_nodes:
        logger.warning("Example %s has no leaf boundary metadata; leaf breakdown skipped.", doc_id)
        return

    # Leaf overview table (one or more pages if needed).
    leaf_rows: List[List[str]] = []
    for node in leaf_nodes:
        idx = int(node.get("chunk_index", 0))
        if idx < 0 or idx >= len(leaf_summaries):
            continue
        start = int(node.get("start_char", 0) or 0)
        end = int(node.get("end_char", start) or start)
        start = max(0, min(start, len(original)))
        end = max(start, min(end, len(original)))
        chunk_text = original[start:end]
        leaf_summary = _strip_think(str(leaf_summaries[idx] or ""))
        leaf_rows.append(
            [
                str(idx),
                f"{start}:{end}",
                str(len(chunk_text)),
                str(len(leaf_summary)),
                _one_line_preview(chunk_text, max_len=70),
            ]
        )

    per_page = 28
    for page_i in range(0, len(leaf_rows), per_page):
        chunk = leaf_rows[page_i : page_i + per_page]
        suffix = ""
        if len(leaf_rows) > per_page:
            suffix = f" (rows {page_i+1}-{min(page_i+per_page,len(leaf_rows))} of {len(leaf_rows)})"
        _render_table_page(
            pdf,
            title=f"Example Document — Leaf Table{suffix}",
            col_labels=["Leaf", "Chars", "Chunk chars", "Summary chars", "Chunk preview"],
            cell_text=chunk,
            font_size=8,
            scale_y=1.25,
        )

    shown = 0
    for node in leaf_nodes:
        idx = int(node.get("chunk_index", shown))
        if idx < 0 or idx >= len(leaf_summaries):
            continue
        start = int(node.get("start_char", 0) or 0)
        end = int(node.get("end_char", start) or start)
        start = max(0, min(start, len(original)))
        end = max(start, min(end, len(original)))

        chunk_text = original[start:end]
        leaf_summary = _strip_think(str(leaf_summaries[idx] or ""))

        page_lines: List[str] = []
        page_lines.append(f"doc_id: {doc_id}")
        page_lines.append(
            f"leaf: {idx}  chars[{start}:{end}]  chunk_len={len(chunk_text)}  summary_len={len(leaf_summary)}"
        )
        page_lines.append("")
        page_lines.append("CHUNK (truncated):")
        page_lines.append("-" * 72)
        page_lines.append(_wrap(chunk_text[:4000] + ("…" if len(chunk_text) > 4000 else ""), width=wrap_width))
        page_lines.append("")
        page_lines.append("LEAF SUMMARY (truncated):")
        page_lines.append("-" * 72)
        page_lines.append(_wrap(leaf_summary[:4000] + ("…" if len(leaf_summary) > 4000 else ""), width=wrap_width))

        _render_text_page(pdf, title=f"Example Leaf {idx}", lines=page_lines, font_size=8)
        shown += 1
        if shown >= max(1, int(max_leaves)):
            break


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a PDF report from score_report.jsonl files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Run output directory (contains config.json)")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Which split reports to include (default: test). Expected files: <split>_score_report.jsonl",
    )
    parser.add_argument(
        "--pdf-path",
        type=Path,
        default=None,
        help="Output PDF path (default: <output-dir>/score_report.pdf)",
    )
    parser.add_argument("--bins", type=int, default=10, help="Calibration bins (default: 10)")
    parser.add_argument("--top-k", type=int, default=15, help="Top errors to display (default: 15)")

    parser.add_argument(
        "--example-id",
        type=str,
        default=None,
        help="Optional example doc_id to include leaf breakdown pages (from checkpoints).",
    )
    parser.add_argument(
        "--example-mode",
        type=str,
        default="central",
        choices=["central", "worst", "best"],
        help="How to choose the example doc if --example-id is not provided (default: central).",
    )
    parser.add_argument(
        "--example-split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split to pick the example from (default: train). Leaf breakdown requires phase1_data.pkl.",
    )
    parser.add_argument(
        "--example-min-leaves",
        type=int,
        default=4,
        help="Minimum tree_leaves for auto-chosen example (default: 4). Use 0 to disable.",
    )
    parser.add_argument(
        "--example-max-leaves",
        type=int,
        default=16,
        help="Max leaf pages to include for the example document (default: 16).",
    )
    parser.add_argument(
        "--wrap-width",
        type=int,
        default=120,
        help="Text wrap width for PDF text pages (default: 120).",
    )

    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not args.verbose:
        for noisy in ("LiteLLM", "litellm", "httpcore", "httpx", "matplotlib", "PIL"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    output_dir = Path(args.output_dir)
    pdf_path = Path(args.pdf_path) if args.pdf_path else output_dir / "score_report.pdf"

    config = _load_json(output_dir / "config.json") or {}
    final_stats = _load_json(output_dir / "final_stats.json")
    max_chunk_chars = _as_float(config.get("max_chunk_chars"))
    max_chunk_chars_int = int(max_chunk_chars) if max_chunk_chars is not None else None

    requested_splits = [str(s).strip() for s in (args.splits or []) if str(s).strip()]
    splits = list(dict.fromkeys(requested_splits)) or ["test"]

    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    summaries: Dict[str, Dict[str, Any]] = {}
    for split in splits:
        report_path = output_dir / f"{split}_score_report.jsonl"
        if not report_path.exists():
            logger.warning("Missing score report: %s", report_path)
            continue
        rows = _load_jsonl(report_path)
        rows_by_split[split] = rows
        summaries[split] = _summarize_rows(rows)

    if not rows_by_split:
        logger.error("No split reports found in %s", output_dir)
        return 2

    # Example doc (leaf breakdown relies on Phase 1 cached results)
    phase1_train, phase1_val = _load_phase1_results(output_dir)
    train_stats = _phase1_doc_stats(phase1_train)
    val_stats = _phase1_doc_stats(phase1_val)

    example_split = str(args.example_split or "train").strip().lower()
    example_id = (args.example_id or "").strip() or None
    example_rows = rows_by_split.get(example_split)
    if example_rows is None:
        report_path = output_dir / f"{example_split}_score_report.jsonl"
        if report_path.exists():
            example_rows = _load_jsonl(report_path)
        else:
            example_rows = []
    if not example_rows and example_split in ("train", "val"):
        example_rows = _rows_from_phase1(phase1_train if example_split == "train" else phase1_val)

    if example_id is None:
        stats = train_stats if example_split == "train" else val_stats if example_split == "val" else train_stats
        example_id = _choose_example_id(
            example_rows,
            mode=args.example_mode,
            doc_stats=stats if stats else None,
            min_leaves=int(args.example_min_leaves),
        )
        if example_id is None and int(args.example_min_leaves) > 0:
            example_id = _choose_example_id(example_rows, mode=args.example_mode)

    example_row: Optional[Dict[str, Any]] = None
    example_result: Optional[Any] = None
    if example_id:
        for row in example_rows:
            if str(row.get("doc_id") or "").strip() == example_id:
                example_row = row
                break
        example_result = _find_result_by_doc_id(phase1_train, example_id) or _find_result_by_doc_id(phase1_val, example_id)
        if example_result is None:
            logger.warning(
                "Example doc_id %s not found in checkpoints; leaf breakdown pages will be skipped.", example_id
            )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        _overview_page(pdf, output_dir=output_dir, config=config, final_stats=final_stats, summaries=summaries)
        _metrics_page(pdf, final_stats=final_stats, rows_by_split=rows_by_split, splits=[s for s in splits if s in rows_by_split])
        _prediction_collapse_page(pdf, rows_by_split=rows_by_split, splits=[s for s in splits if s in rows_by_split])
        for split, rows in rows_by_split.items():
            _scatter_page(pdf, rows=rows, title=f"{split}: Predicted vs Actual")
            _dist_overlay_page(pdf, rows=rows, title=f"{split}: Predicted vs Actual (Distribution)")
            _residuals_page(pdf, rows=rows, title=f"{split}: Residuals vs Actual")
            _hist_page(pdf, rows=rows, title=f"{split}: Absolute Error Histogram")
            _binned_mae_page(pdf, rows=rows, title=f"{split}: MAE vs Actual (Binned)", bins=int(args.bins))
            _calibration_page(pdf, rows=rows, title=f"{split}: Calibration (binned)", bins=int(args.bins))
            _top_errors_page(pdf, rows=rows, title=f"{split}: Top Errors", k=int(args.top_k))
        if example_result is not None:
            _example_leaf_breakdown_pages(
                pdf,
                example_result=example_result,
                example_row=example_row,
                max_leaves=int(args.example_max_leaves),
                wrap_width=int(args.wrap_width),
                max_chunk_chars=max_chunk_chars_int,
            )

    logger.info("Wrote %s", pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
