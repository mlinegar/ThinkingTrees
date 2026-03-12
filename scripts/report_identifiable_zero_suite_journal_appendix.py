#!/usr/bin/env python3
"""Generate an appendix-style oracle-equivalence report for the identifiable-zero suite.

Design goals:
- Journal-appendix readability: one figure per page, large typography, no tiny rotated labels.
- Pedantically clear setup: explicit knob definitions, comparability rules, and normalization math.
- Explicit Figure-B -> Figure-C crosswalk: which (q_train,q_infer) points define each frontier segment.
- Neural-operator deep dives for both:
  (1) Markov "neural merger" (LearnedCountSketch) guidance override semantics, and
  (2) C-TreePO neural topic refiner (neural φ estimators).

This script is intentionally additive: it does not replace the existing publication_clean report.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
import numpy as np

PLOT_FLOOR = 1e-12
CEILING_THRESHOLD = 1e-8
ERROR_AXIS_TOP = 10.0
NORM_EPS_DEN_DEFAULT = 1e-12

# Fixed "identifiable-zero longrun equiv v1" slice (same as publication_clean).
FIXED_SEG_TRAIN_DOCS = 12000
FIXED_SEG_LAMBDA = 1.0
FIXED_CTREE_TRAIN_DOCS = 4096
FIXED_CTREE_MIN_CAL_SAMPLES = 50
FIXED_MARKOV_TRAIN_DOCS = 8000
FIXED_MARKOV_LEAF_QUERY_RATE = 1.0
FIXED_MARKOV_INCLUDE_ROOT_QUERY = True

LEARN_LABEL = "learn-time oracle visibility"
DECISION_LABEL = "decision-time oracle visibility"
LEARN_SYMBOL = "q_train"
DECISION_SYMBOL = "q_infer"

SEG_TRUE_COLOR = "#1f77b4"
SEG_EMBED_COLOR = "#6c757d"
CTREE_COLOR = "#2ca02c"
MARKOV_ADD_COLOR = "#17becf"
MARKOV_NEURAL_COLOR = "#d62728"
NA_COLOR = "#888888"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build appendix-quality report for identifiable oracle-equivalence suite.")
    p.add_argument("--baseline-output-root", type=Path, required=True)
    p.add_argument("--tuning-output-root", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--norm-eps-den", type=float, default=NORM_EPS_DEN_DEFAULT)
    return p.parse_args()


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(x: object) -> Optional[float]:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _float_or_nan(x: object) -> float:
    v = _as_float(x)
    return float(v) if v is not None else float("nan")


def _fmt(x: object) -> str:
    v = _as_float(x)
    if v is None:
        return "nan"
    if abs(v) >= 1000.0 or (0.0 < abs(v) < 1e-3):
        return f"{v:.3e}"
    return f"{v:.6g}"


def _median(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if not xs:
        return float("nan")
    return float(statistics.median(xs))


def _median_q25_q75(vals: Iterable[float]) -> Tuple[float, float, float]:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if not xs:
        return float("nan"), float("nan"), float("nan")
    xs.sort()
    return float(statistics.median(xs)), float(np.percentile(xs, 25.0)), float(np.percentile(xs, 75.0))


def _norm_den(baseline: float, ceiling: float) -> float:
    if not (math.isfinite(float(baseline)) and math.isfinite(float(ceiling))):
        return float("nan")
    return float(float(baseline) - float(ceiling))


def _norm_valid(baseline: float, ceiling: float, *, eps_den: float) -> bool:
    den = _norm_den(float(baseline), float(ceiling))
    return bool(math.isfinite(den) and den > float(eps_den))


def _normalized_gap(err: float, baseline: float, ceiling: float) -> float:
    den = max(float(baseline) - float(ceiling), 1e-12)
    return float((float(err) - float(ceiling)) / den)


def _normalize_series(values: Sequence[float], baseline: float, ceiling: float, *, eps_den: float) -> Tuple[List[float], bool, float]:
    den = _norm_den(float(baseline), float(ceiling))
    valid = _norm_valid(float(baseline), float(ceiling), eps_den=float(eps_den))
    if not valid:
        return ([float("nan")] * len(values), False, float(den))
    out = [_normalized_gap(float(v), float(baseline), float(ceiling)) for v in values]
    return ([float(v) for v in out], True, float(den))


def _clip_norm(v: object, *, clip_min: float = -0.02, clip_max: float = 1.2) -> float:
    x = _as_float(v)
    if x is None:
        return float("nan")
    return float(min(float(clip_max), max(float(clip_min), float(x))))


def _plot_floor(v: object) -> float:
    x = _as_float(v)
    if x is None:
        return float("nan")
    return float(max(PLOT_FLOOR, float(x)))


def _setup_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    plt.rcParams.update(
        {
            "font.size": 13.5,
            "axes.titlesize": 15.5,
            "axes.labelsize": 13.5,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 11.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Prevent cropping of large labels/titles in saved PNG/PDF.
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.18,
        }
    )


def _run_pandoc(md_path: Path, pdf_path: Path) -> bool:
    if shutil.which("pandoc") is None or shutil.which("pdflatex") is None:
        return False
    subprocess.run(
        ["pandoc", str(md_path.name), "-o", str(pdf_path.name), "--pdf-engine=pdflatex"],
        cwd=str(md_path.parent),
        check=True,
    )
    return True


def _family_color(family: str) -> str:
    fam = str(family)
    if fam == "segment":
        return SEG_TRUE_COLOR
    if fam == "ctree":
        return CTREE_COLOR
    if fam == "markov_add":
        return MARKOV_ADD_COLOR
    if fam == "markov_neural":
        return MARKOV_NEURAL_COLOR
    return "#444444"


def _collect_segment_fixed(output_root: Path, *, eps_den: float) -> Dict[str, object]:
    files = sorted(glob.glob(str(output_root / "segment_lda_ops_weight_recovery" / "**" / "*seed_*.json"), recursive=True))
    rows: List[dict] = []
    exact_vals: List[float] = []
    for fp in files:
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        m = payload.get("metrics", {}) or {}
        ex = _as_float(((m.get("exact") or {}).get("root_mae")))
        if ex is not None:
            exact_vals.append(float(ex))
        ridge = _as_float(((m.get("ridge") or {}).get("root_mae")))
        ridge_true = _as_float(((m.get("ridge_true_topics") or {}).get("root_mae")))
        rows.append(
            {
                "train_docs": int(cfg.get("train_docs", -1)),
                "q_train": float(cfg.get("audit_fraction", float("nan"))),
                "lambda_multiplier": float(cfg.get("lambda_multiplier", float("nan"))),
                "topic_phi_estimator": str(cfg.get("topic_phi_estimator", "")),
                "ridge": float(ridge) if ridge is not None else float("nan"),
                "ridge_true": float(ridge_true) if ridge_true is not None else float("nan"),
            }
        )

    fixed = [
        r
        for r in rows
        if int(r["train_docs"]) == FIXED_SEG_TRAIN_DOCS
        and math.isfinite(float(r["lambda_multiplier"]))
        and abs(float(r["lambda_multiplier"]) - FIXED_SEG_LAMBDA) <= 1e-12
    ]

    def _lane(phi: str, metric_key: str) -> Dict[str, object]:
        lane_rows = [r for r in fixed if str(r["topic_phi_estimator"]) == phi and math.isfinite(float(r[metric_key]))]
        qvals = sorted({float(r["q_train"]) for r in lane_rows if math.isfinite(float(r["q_train"]))})
        med: List[float] = []
        q25: List[float] = []
        q75: List[float] = []
        counts: List[int] = []
        for q in qvals:
            vals = [float(r[metric_key]) for r in lane_rows if abs(float(r["q_train"]) - q) <= 1e-12]
            m, a, b = _median_q25_q75(vals)
            med.append(float(m))
            q25.append(float(a))
            q75.append(float(b))
            counts.append(int(len(vals)))
        baseline = float(med[0]) if med else float("nan")
        ceiling = float(min(med)) if med else float("nan")
        norm, valid, den = _normalize_series([float(v) for v in med], baseline, ceiling, eps_den=float(eps_den))
        return {
            "q_train": [float(q) for q in qvals],
            "raw_median": [float(v) for v in med],
            "raw_q25": [float(v) for v in q25],
            "raw_q75": [float(v) for v in q75],
            "n_per_q": counts,
            "baseline": baseline,
            "ceiling": ceiling,
            "norm_den": float(den),
            "norm_valid": bool(valid),
            "norm_gap": [float(v) for v in norm],
            "q1": float(med[qvals.index(1.0)]) if 1.0 in qvals else float("nan"),
        }

    lane_true = _lane("true", "ridge_true")
    lane_embed = _lane("embedding_spectral", "ridge")

    return {
        "present": bool(files),
        "n_files": int(len(files)),
        "exact_root_mae_max": float(max(exact_vals) if exact_vals else float("nan")),
        "fixed": {
            "train_docs": FIXED_SEG_TRAIN_DOCS,
            "lambda_multiplier": FIXED_SEG_LAMBDA,
            "lanes": {
                "phi_true": lane_true,
                "phi_embedding_spectral": lane_embed,
            },
        },
    }


def _collect_ctree_fixed(output_root: Path, *, eps_den: float) -> Dict[str, object]:
    files = sorted(glob.glob(str(output_root / "segmented_lda_ctreepo" / "**" / "*.json"), recursive=True))
    rows: List[dict] = []
    oracle_vals: List[float] = []
    for fp in files:
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        m = payload.get("metrics", {}) or {}
        budgeted = _as_float(((m.get("estimated_calibrated_budgeted") or {}).get("root_l1_mean")))
        oracle = _as_float(((m.get("oracle_tree") or {}).get("root_l1_mean")))
        q_leaf = _as_float(cfg.get("eval_leaf_query_rate"))
        q_int = _as_float(cfg.get("eval_internal_query_rate"))
        q_train = _as_float(cfg.get("calibration_leaf_query_rate"))
        if oracle is not None:
            oracle_vals.append(float(oracle))
        if budgeted is None or q_leaf is None or q_int is None or q_train is None:
            continue
        rows.append(
            {
                "train_docs": int(cfg.get("n_books_train", -1)),
                "q_train": float(q_train),
                "q_leaf": float(q_leaf),
                "q_internal": float(q_int),
                "raw": float(budgeted),
                "calibration_samples": int(payload.get("calibration_samples", 0) or 0),
                "topic_phi_estimator": str(cfg.get("topic_phi_estimator", "")),
                "topic_phi_l2_error_mean": _float_or_nan((payload.get("topic_meta") or {}).get("topic_phi_l2_error_mean")),
                "topic_phi_l2_error_p95": _float_or_nan((payload.get("topic_meta") or {}).get("topic_phi_l2_error_p95")),
                "topic_phi_l2_error_max": _float_or_nan((payload.get("topic_meta") or {}).get("topic_phi_l2_error_max")),
            }
        )

    fixed = [
        r
        for r in rows
        if int(r["train_docs"]) == FIXED_CTREE_TRAIN_DOCS
        and int(r["calibration_samples"]) >= FIXED_CTREE_MIN_CAL_SAMPLES
        and abs(float(r["q_leaf"]) - float(r["q_internal"])) <= 1e-12
    ]

    q_train_vals = sorted({float(r["q_train"]) for r in fixed if math.isfinite(float(r["q_train"]))})
    q_infer_vals = sorted({float(r["q_leaf"]) for r in fixed if math.isfinite(float(r["q_leaf"]))})

    matrix_raw: List[List[float]] = []
    matrix_counts: List[List[int]] = []
    for qtr in q_train_vals:
        row_raw: List[float] = []
        row_n: List[int] = []
        for qinf in q_infer_vals:
            vals = [
                float(r["raw"])
                for r in fixed
                if abs(float(r["q_train"]) - float(qtr)) <= 1e-12 and abs(float(r["q_leaf"]) - float(qinf)) <= 1e-12
            ]
            m, _a, _b = _median_q25_q75(vals)
            row_raw.append(float(m))
            row_n.append(int(len(vals)))
        matrix_raw.append(row_raw)
        matrix_counts.append(row_n)

    flat_raw = [float(v) for rr in matrix_raw for v in rr if math.isfinite(float(v))]
    baseline = float(matrix_raw[0][0]) if matrix_raw and matrix_raw[0] else float("nan")
    ceiling = float(min(flat_raw)) if flat_raw else float("nan")
    norm_flat, norm_valid, norm_den = _normalize_series(
        [float(v) for rr in matrix_raw for v in rr],
        baseline,
        ceiling,
        eps_den=float(eps_den),
    )
    matrix_norm: List[List[float]] = []
    k = 0
    for rr in matrix_raw:
        row = []
        for _ in rr:
            row.append(float(norm_flat[k]))
            k += 1
        matrix_norm.append(row)

    qtrain_max = float(max(q_train_vals)) if q_train_vals else float("nan")

    return {
        "present": bool(files),
        "n_files": int(len(files)),
        "oracle_root_l1_max": float(max(oracle_vals) if oracle_vals else float("nan")),
        "fixed": {
            "train_docs": FIXED_CTREE_TRAIN_DOCS,
            "min_calibration_samples": FIXED_CTREE_MIN_CAL_SAMPLES,
            "q_train": [float(x) for x in q_train_vals],
            "q_infer": [float(x) for x in q_infer_vals],
            "matrix_raw": [[float(v) for v in rr] for rr in matrix_raw],
            "matrix_counts": [[int(v) for v in rr] for rr in matrix_counts],
            "matrix_norm": [[float(v) for v in rr] for rr in matrix_norm],
            "baseline": baseline,
            "ceiling": ceiling,
            "norm_den": float(norm_den),
            "norm_valid": bool(norm_valid),
            "infer_full_context": {"q_train": qtrain_max, "q_infer": 1.0},
        },
        "rows_fixed": fixed,
    }


@dataclass(frozen=True)
class _MarkovRunKey:
    family: str
    q_train: float
    schedule_consistency_weight: float
    guidance_override_mode: str


def _scan_markov_runs(output_root: Path) -> Tuple[List[dict], List[dict]]:
    files = sorted(glob.glob(str(output_root / "markov_changepoint_ops_count" / "**" / "*seed_*.json"), recursive=True))
    learned_rows: List[dict] = []
    guided_rows: List[dict] = []
    for fp in files:
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        metrics = payload.get("metrics", {}) or {}

        train_docs = int(cfg.get("train_docs", -1))
        if train_docs != FIXED_MARKOV_TRAIN_DOCS:
            continue
        leaf = _as_float(cfg.get("leaf_query_rate"))
        if leaf is None or abs(float(leaf) - FIXED_MARKOV_LEAF_QUERY_RATE) > 1e-12:
            continue
        if bool(cfg.get("include_root_query", True)) is not bool(FIXED_MARKOV_INCLUDE_ROOT_QUERY):
            continue

        fam = str(cfg.get("model_family", ""))
        q_train = _as_float(cfg.get("audit_fraction"))
        if q_train is None:
            continue
        scw = float(_as_float(cfg.get("schedule_consistency_weight")) or 0.0)
        gov = str(cfg.get("guidance_override_mode", "reset")).strip().lower() or "reset"

        learned = metrics.get("learned") or {}
        learned_rows.append(
            {
                "family": fam,
                "q_train": float(q_train),
                "schedule_consistency_weight": float(scw),
                "guidance_override_mode": gov,
                "root_mae": _float_or_nan(learned.get("root_mae")),
                "merge_mae": _float_or_nan(learned.get("merge_mae")),
                "merge_violation_rate": _float_or_nan(learned.get("merge_violation_rate")),
                "schedule_spread_mean": _float_or_nan(learned.get("schedule_spread_mean")),
                "schedule_spread_p95": _float_or_nan(learned.get("schedule_spread_p95")),
                "train_loss_final": _float_or_nan(learned.get("train_loss_final")),
            }
        )

        pts = ((metrics.get("guided_eval_curve") or {}).get("points") or [])
        for pt in pts:
            if not isinstance(pt, dict):
                continue
            q = _as_float(pt.get("q"))
            if q is None:
                continue
            guided_rows.append(
                {
                    "family": fam,
                    "q_train": float(q_train),
                    "schedule_consistency_weight": float(scw),
                    "guidance_override_mode": gov,
                    "q_infer": float(q),
                    "root_mae": _float_or_nan(pt.get("root_mae")),
                    "merge_mae": _float_or_nan(pt.get("merge_mae")),
                    "merge_violation_rate": _float_or_nan(pt.get("merge_violation_rate")),
                    "effective_q_mean": _float_or_nan(pt.get("effective_q_mean")),
                    "guided_internal_nodes_mean": _float_or_nan(pt.get("guided_internal_nodes_mean")),
                }
            )
    return learned_rows, guided_rows


def _matrix_from_guided_rows(
    guided_rows: Sequence[dict],
    *,
    family: str,
    scw: float,
    gov: str,
    eps_den: float,
) -> Dict[str, object]:
    rr = [
        r
        for r in guided_rows
        if str(r.get("family")) == family
        and abs(float(r.get("schedule_consistency_weight", 0.0)) - float(scw)) <= 1e-12
        and str(r.get("guidance_override_mode", "reset")) == str(gov)
    ]
    q_train_vals = sorted({float(r["q_train"]) for r in rr})
    q_infer_vals = sorted({float(r["q_infer"]) for r in rr})
    matrix_raw: List[List[float]] = []
    matrix_counts: List[List[int]] = []
    for qtr in q_train_vals:
        row_raw: List[float] = []
        row_n: List[int] = []
        for qinf in q_infer_vals:
            vals = [
                float(r["root_mae"])
                for r in rr
                if abs(float(r["q_train"]) - float(qtr)) <= 1e-12 and abs(float(r["q_infer"]) - float(qinf)) <= 1e-12
            ]
            m, _a, _b = _median_q25_q75(vals)
            row_raw.append(float(m))
            row_n.append(int(len(vals)))
        matrix_raw.append(row_raw)
        matrix_counts.append(row_n)

    flat = [float(v) for row in matrix_raw for v in row if math.isfinite(float(v))]
    baseline = float(matrix_raw[0][0]) if matrix_raw and matrix_raw[0] else float("nan")
    ceiling = float(min(flat)) if flat else float("nan")
    norm_flat, norm_valid, norm_den = _normalize_series([float(v) for row in matrix_raw for v in row], baseline, ceiling, eps_den=float(eps_den))
    matrix_norm: List[List[float]] = []
    k = 0
    for row in matrix_raw:
        out_row = []
        for _ in row:
            out_row.append(float(norm_flat[k]))
            k += 1
        matrix_norm.append(out_row)

    return {
        "q_train": [float(x) for x in q_train_vals],
        "q_infer": [float(x) for x in q_infer_vals],
        "matrix_raw": [[float(v) for v in row] for row in matrix_raw],
        "matrix_norm": [[float(v) for v in row] for row in matrix_norm],
        "matrix_counts": [[int(v) for v in row] for row in matrix_counts],
        "baseline": baseline,
        "ceiling": ceiling,
        "norm_den": float(norm_den),
        "norm_valid": bool(norm_valid),
    }


def _plot_heatmap_page(
    *,
    out_png: Path,
    out_pdf: Path,
    arr: np.ndarray,
    xvals: Sequence[float],
    yvals: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str,
    norm: mcolors.Normalize,
    yticklabels: Optional[Sequence[str]] = None,
    note: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.3, 6.5), constrained_layout=True)
    arr_masked = np.ma.masked_invalid(arr)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color="#f2f2f2")
    im = ax.imshow(arr_masked, aspect="auto", origin="lower", cmap=cm, norm=norm)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(xvals)))
    ax.set_xticklabels([_fmt(v) for v in xvals], rotation=0)
    ax.set_yticks(range(len(yvals)))
    if yticklabels is not None:
        ax.set_yticklabels(list(yticklabels))
    else:
        ax.set_yticklabels([_fmt(v) for v in yvals])
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.92)
    if note:
        ax.text(
            0.02,
            0.02,
            note,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.95),
        )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240)
    fig.savefig(out_pdf)
    plt.close(fig)


def _budget_rate(q_train: float, q_infer: Optional[float]) -> float:
    if q_infer is None:
        return float(q_train)
    return 0.5 * float(q_train) + 0.5 * float(q_infer)


@dataclass(frozen=True)
class FrontierPoint:
    budget: float
    value: float
    q_train: float
    q_infer: Optional[float]


def _frontier_with_crosswalk(points: Sequence[FrontierPoint]) -> List[Dict[str, object]]:
    # Iso-budget min per budget, then cumulative envelope over increasing budgets.
    by_budget: Dict[float, List[FrontierPoint]] = {}
    for p in points:
        if not (math.isfinite(float(p.budget)) and math.isfinite(float(p.value))):
            continue
        by_budget.setdefault(float(p.budget), []).append(p)
    budgets = sorted(by_budget.keys())
    envelope: List[Dict[str, object]] = []
    cur_best = float("inf")
    cur_src: Optional[FrontierPoint] = None
    for b in budgets:
        cand = by_budget[b]
        cand_best = min(cand, key=lambda x: (float(x.value), float(x.q_train), float(x.q_infer or -1.0)))
        if float(cand_best.value) < float(cur_best) - 1e-18:
            cur_best = float(cand_best.value)
            cur_src = cand_best
        envelope.append(
            {
                "budget": float(b),
                "best_value": float(cur_best) if math.isfinite(cur_best) else float("nan"),
                "argmin_q_train": float(cur_src.q_train) if cur_src is not None else float("nan"),
                "argmin_q_infer": (float(cur_src.q_infer) if (cur_src is not None and cur_src.q_infer is not None) else None),
                "improved_at_budget": bool(cand_best.value < float(cur_best) + 1e-18 and cur_src is cand_best),
            }
        )
    return envelope


def _plot_endpoints_dotplot(
    *,
    out_png: Path,
    out_pdf: Path,
    endpoints: Sequence[Dict[str, object]],
) -> None:
    # Horizontal dotplots: raw (log) and normalized side-by-side, y=endpoint.
    names = [str(ep.get("plot_label") or ep.get("name") or "") for ep in endpoints]
    y = np.arange(len(names), dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 7.9), constrained_layout=True, sharey=True)
    ax_raw, ax_norm = axes

    for i, ep in enumerate(endpoints):
        fam = str(ep.get("family", ""))
        col = _family_color(fam)
        stage = str(ep.get("stage", "train"))
        marker = "o" if stage == "infer" else "^"
        raw = _plot_floor(ep.get("raw"))
        ax_raw.scatter(raw, y[i], s=90, marker=marker, color=col, edgecolors="#333333", linewidths=0.8, zorder=3)

        valid = bool(ep.get("norm_valid", False))
        if valid:
            xv = _clip_norm(ep.get("norm"))
            ax_norm.scatter(xv, y[i], s=90, marker=marker, color=col, edgecolors="#333333", linewidths=0.8, zorder=3)
        else:
            ax_norm.scatter(1.17, y[i], s=90, marker="x", color=NA_COLOR, linewidths=2.0, zorder=3)
            ax_norm.text(1.165, y[i], "N/A", ha="right", va="center", fontsize=11.5, color=NA_COLOR)

    ax_raw.set_xscale("log")
    ax_raw.set_xlim(PLOT_FLOOR, ERROR_AXIS_TOP)
    ax_raw.axvline(CEILING_THRESHOLD, color="#666666", linestyle="--", linewidth=1.2)
    ax_raw.set_title("A2: Raw endpoint error\n(within-family only)")
    ax_raw.set_xlabel("raw error (log scale)")
    ax_raw.set_yticks(y)
    ax_raw.set_yticklabels(names)
    ax_raw.invert_yaxis()

    ax_norm.set_xlim(-0.03, 1.25)
    ax_norm.axvline(0.0, color="#1a9850", linestyle="--", linewidth=1.2)
    ax_norm.axvline(1.0, color="#666666", linestyle=":", linewidth=1.2)
    ax_norm.set_title("A3: Normalized gap-to-ceiling\n(cross-family comparable)")
    ax_norm.set_xlabel("normalized gap-to-ceiling\n(lower is better)")

    legend_handles = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#bbbbbb", markeredgecolor="#333333", label=f"{LEARN_LABEL} endpoint", markersize=8),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#bbbbbb", markeredgecolor="#333333", label=f"{DECISION_LABEL} endpoint", markersize=8),
        Line2D([0], [0], marker="x", color=NA_COLOR, linestyle="none", label="N/A normalization lane", markersize=8),
    ]
    ax_norm.legend(handles=legend_handles, frameon=False, loc="lower right")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240)
    fig.savefig(out_pdf)
    plt.close(fig)


def _build_baseline_endpoints(
    segment: Dict[str, object],
    ctree: Dict[str, object],
    markov_fixed: Dict[str, object],
) -> List[Dict[str, object]]:
    seg_lanes = ((segment.get("fixed") or {}).get("lanes") or {})
    seg_true = seg_lanes.get("phi_true") or {}
    seg_embed = seg_lanes.get("phi_embedding_spectral") or {}

    ct_fixed = ctree.get("fixed") or {}
    mk_fams = (markov_fixed.get("families") or {})
    mk_add = mk_fams.get("additive") or {}
    mk_neu = mk_fams.get("neural") or {}

    def _status(raw: object) -> str:
        v = _as_float(raw)
        if v is None:
            return "UNKNOWN"
        return "PASS" if float(v) <= CEILING_THRESHOLD else "FAIL"

    def _norm_display(x: object, valid: bool) -> str:
        if not bool(valid):
            return "N/A"
        v = _as_float(x)
        if v is None:
            return "nan"
        if float(v) > 1.2:
            return ">1.2 (clipped)"
        if float(v) < -0.02:
            return "<-0.02 (clipped)"
        return _fmt(v)

    endpoints = [
        {
            "endpoint_id": "segment_phi_true_learn_full",
            "name": "Segment (phi=true, learn-time full)",
            "plot_label": "Segment phi=true\n(train full)",
            "family": "segment",
            "stage": "train",
            "raw": float(seg_true.get("q1", float("nan"))),
            "norm": float((seg_true.get("norm_gap") or [float("nan")])[-1]),
            "norm_valid": bool(seg_true.get("norm_valid", False)),
        },
        {
            "endpoint_id": "segment_phi_embedding_learn_full",
            "name": "Segment (phi=embedding_spectral, learn-time full)",
            "plot_label": "Segment phi=embedding\n(train full)",
            "family": "segment",
            "stage": "train",
            "raw": float(seg_embed.get("q1", float("nan"))),
            "norm": float((seg_embed.get("norm_gap") or [float("nan")])[-1]),
            "norm_valid": bool(seg_embed.get("norm_valid", False)),
        },
        {
            "endpoint_id": "ctree_decision_full",
            "name": "C-TreePO (decision-time full)",
            "plot_label": "C-TreePO\n(decision full)",
            "family": "ctree",
            "stage": "infer",
            "raw": float(ct_fixed.get("matrix_raw", [[float("nan")]])[-1][-1]) if (ct_fixed.get("matrix_raw") or []) else float("nan"),
            "norm": float(ct_fixed.get("matrix_norm", [[float("nan")]])[-1][-1]) if (ct_fixed.get("matrix_norm") or []) else float("nan"),
            "norm_valid": bool(ct_fixed.get("norm_valid", False)),
        },
        {
            "endpoint_id": "markov_additive_learn_full",
            "name": "Markov additive (learn-time full)",
            "plot_label": "Markov additive\n(train full)",
            "family": "markov_add",
            "stage": "train",
            "raw": float(mk_add.get("train_full_raw", float("nan"))),
            "norm": float(mk_add.get("train_full_norm", float("nan"))),
            "norm_valid": bool(mk_add.get("train_norm_valid", False)),
        },
        {
            "endpoint_id": "markov_additive_decision_full",
            "name": "Markov additive (decision-time full)",
            "plot_label": "Markov additive\n(decision full)",
            "family": "markov_add",
            "stage": "infer",
            "raw": float(mk_add.get("infer_full_raw", float("nan"))),
            "norm": float(mk_add.get("infer_full_norm", float("nan"))),
            "norm_valid": bool(mk_add.get("norm_valid", False)),
        },
        {
            "endpoint_id": "markov_neural_learn_full",
            "name": "Markov neural (learn-time full)",
            "plot_label": "Markov neural\n(train full)",
            "family": "markov_neural",
            "stage": "train",
            "raw": float(mk_neu.get("train_full_raw", float("nan"))),
            "norm": float(mk_neu.get("train_full_norm", float("nan"))),
            "norm_valid": bool(mk_neu.get("train_norm_valid", False)),
        },
        {
            "endpoint_id": "markov_neural_decision_full",
            "name": "Markov neural (decision-time full)",
            "plot_label": "Markov neural\n(decision full)",
            "family": "markov_neural",
            "stage": "infer",
            "raw": float(mk_neu.get("infer_full_raw", float("nan"))),
            "norm": float(mk_neu.get("infer_full_norm", float("nan"))),
            "norm_valid": bool(mk_neu.get("norm_valid", False)),
        },
    ]
    for ep in endpoints:
        ep["status"] = _status(ep.get("raw"))
        ep["norm_display"] = _norm_display(ep.get("norm"), bool(ep.get("norm_valid", False)))
    return endpoints


def _collect_markov_fixed_baseline(output_root: Path, *, eps_den: float) -> Dict[str, object]:
    learned_rows, guided_rows = _scan_markov_runs(output_root)
    families = sorted({str(r.get("family")) for r in learned_rows if str(r.get("family"))})

    fixed_by_family: Dict[str, Dict[str, object]] = {}
    for fam in families:
        # Baseline: scw=0, guidance_override_mode defaults to "reset" if absent.
        matrix = _matrix_from_guided_rows(
            guided_rows,
            family=fam,
            scw=0.0,
            gov="reset",
            eps_den=float(eps_den),
        )

        # Also compute learn-time curve at q_infer=0 via guided points.
        q_train_vals = list(matrix.get("q_train") or [])
        q_infer_vals = list(matrix.get("q_infer") or [])
        q0_idx = q_infer_vals.index(0.0) if 0.0 in q_infer_vals else 0
        train_curve = [float(row[q0_idx]) for row in (matrix.get("matrix_raw") or [])] if q_infer_vals else []
        baseline = float(train_curve[0]) if train_curve else float("nan")
        ceiling = float(min(train_curve)) if train_curve else float("nan")
        train_norm, train_valid, train_den = _normalize_series(train_curve, baseline, ceiling, eps_den=float(eps_den))

        qtr_one_idx = q_train_vals.index(1.0) if 1.0 in q_train_vals else -1
        qinf_one_idx = q_infer_vals.index(1.0) if 1.0 in q_infer_vals else -1
        train_full_raw = float(train_curve[qtr_one_idx]) if qtr_one_idx >= 0 and train_curve else float("nan")
        train_full_norm = float(train_norm[qtr_one_idx]) if qtr_one_idx >= 0 and train_norm else float("nan")
        infer_full_raw = (
            float((matrix.get("matrix_raw") or [[]])[qtr_one_idx][qinf_one_idx])
            if qtr_one_idx >= 0 and qinf_one_idx >= 0
            else float("nan")
        )
        infer_full_norm = (
            float((matrix.get("matrix_norm") or [[]])[qtr_one_idx][qinf_one_idx])
            if qtr_one_idx >= 0 and qinf_one_idx >= 0
            else float("nan")
        )

        fixed_by_family[fam] = {
            **matrix,
            "train_curve_raw": [float(v) for v in train_curve],
            "train_curve_norm": [float(v) for v in train_norm],
            "train_norm_valid": bool(train_valid),
            "train_norm_den": float(train_den),
            "train_full_raw": float(train_full_raw),
            "train_full_norm": float(train_full_norm),
            "infer_full_raw": float(infer_full_raw),
            "infer_full_norm": float(infer_full_norm),
        }

    return {
        "families": fixed_by_family,
        "learned_rows": learned_rows,
        "guided_rows": guided_rows,
    }


def _plot_markov_neural_deep_dive(
    *,
    out_png: Path,
    out_pdf: Path,
    baseline_guided: Sequence[dict],
    tuning_guided: Sequence[dict],
) -> Dict[str, object]:
    # Compare baseline (reset, scw=0) vs tuning (adjust, scw in {0,0.1}) at q_train in {0.1,1.0}.
    def _series(rows: Sequence[dict], *, q_train: float, scw: float, gov: str) -> Dict[float, Dict[str, float]]:
        sub = [
            r
            for r in rows
            if str(r.get("family")) == "neural"
            and abs(float(r.get("q_train", float("nan"))) - float(q_train)) <= 1e-12
            and abs(float(r.get("schedule_consistency_weight", 0.0)) - float(scw)) <= 1e-12
            and str(r.get("guidance_override_mode", "reset")) == str(gov)
        ]
        out: Dict[float, Dict[str, float]] = {}
        for q in sorted({float(r.get("q_infer")) for r in sub}):
            bucket = [r for r in sub if abs(float(r.get("q_infer")) - float(q)) <= 1e-12]
            out[float(q)] = {
                "root_mae": _median(float(b.get("root_mae")) for b in bucket),
                "merge_mae": _median(float(b.get("merge_mae")) for b in bucket),
                "merge_violation_rate": _median(float(b.get("merge_violation_rate")) for b in bucket),
            }
        return out

    variants = [
        ("baseline reset scw=0.0", baseline_guided, 0.0, "reset", "#333333", "-"),
        ("tuned adjust scw=0.0", tuning_guided, 0.0, "adjust", MARKOV_NEURAL_COLOR, "-"),
        ("tuned adjust scw=0.1", tuning_guided, 0.1, "adjust", "#7a1fa2", "-"),
    ]
    qtr_vals = [0.1, 1.0]

    fig, axes = plt.subplots(2, 2, figsize=(7.3, 9.4), constrained_layout=True)
    ax_root_01, ax_root_1 = axes[0, 0], axes[0, 1]
    ax_merge_01, ax_merge_1 = axes[1, 0], axes[1, 1]

    for title_ax, qtr, ax_root, ax_merge in [
        ("q_train=0.1 (limited learn-time labels)", 0.1, ax_root_01, ax_merge_01),
        ("q_train=1.0 (full learn-time labels)", 1.0, ax_root_1, ax_merge_1),
    ]:
        for label, rows, scw, gov, color, ls in variants:
            s = _series(rows, q_train=qtr, scw=scw, gov=gov)
            if not s:
                continue
            xs = sorted(s.keys())
            root = [_plot_floor(s[x]["root_mae"]) for x in xs]
            merge = [_plot_floor(s[x]["merge_mae"]) for x in xs]
            ax_root.plot(xs, root, marker="o", color=color, linestyle=ls, linewidth=2.0, label=label)
            ax_merge.plot(xs, merge, marker="o", color=color, linestyle=ls, linewidth=2.0, label=label)

        ax_root.set_yscale("log")
        ax_root.set_ylim(PLOT_FLOOR, ERROR_AXIS_TOP)
        ax_root.set_xlabel(f"{DECISION_LABEL} ({DECISION_SYMBOL})")
        ax_root.set_ylabel("root MAE (log)")
        ax_root.set_title(f"Root error vs decision-time visibility\n{title_ax}")
        ax_root.axhline(CEILING_THRESHOLD, color="#666666", linestyle="--", linewidth=1.0)

        ax_merge.set_yscale("log")
        ax_merge.set_ylim(PLOT_FLOOR, ERROR_AXIS_TOP)
        ax_merge.set_xlabel(f"{DECISION_LABEL} ({DECISION_SYMBOL})")
        ax_merge.set_ylabel("merge MAE (log)")
        ax_merge.set_title(f"Merge error vs decision-time visibility\n{title_ax}")
        ax_merge.axhline(CEILING_THRESHOLD, color="#666666", linestyle="--", linewidth=1.0)

    ax_root_1.legend(frameon=False, loc="upper right", fontsize=10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240)
    fig.savefig(out_pdf)
    plt.close(fig)

    # Diagnostics: monotonicity violations for tuned series at q_train=1.0.
    def _monotone_violations(s: Dict[float, Dict[str, float]]) -> int:
        xs = sorted(s.keys())
        ys = [float(s[x]["root_mae"]) for x in xs]
        n = 0
        for a, b in zip(range(len(xs) - 1), range(1, len(xs))):
            if ys[b] > ys[a] + 1e-10:
                n += 1
        return n

    tuned_scw0 = _series(tuning_guided, q_train=1.0, scw=0.0, gov="adjust")
    tuned_scw01 = _series(tuning_guided, q_train=1.0, scw=0.1, gov="adjust")
    return {
        "monotonicity_violations_qtrain1_scw0": int(_monotone_violations(tuned_scw0)),
        "monotonicity_violations_qtrain1_scw0p1": int(_monotone_violations(tuned_scw01)),
    }


def _plot_ctree_phi_ablation(
    *,
    out_png: Path,
    out_pdf: Path,
    baseline_rows_fixed: Sequence[dict],
    tuning_rows: Sequence[dict],
) -> Dict[str, object]:
    # Compare spectral_numpy (baseline) vs tensor_lda vs neural_ctreepo (tuning), at q_train=0.1.
    qtr = 0.1
    q_infer_vals = [0.0, 0.5, 1.0]
    methods = [
        ("spectral_numpy (baseline)", "spectral_numpy", baseline_rows_fixed, CTREE_COLOR),
        ("tensor_lda (tuning)", "tensor_lda", tuning_rows, "#ff7f0e"),
        ("neural_ctreepo (tuning)", "neural_ctreepo", tuning_rows, "#7a1fa2"),
    ]

    topic_err: Dict[str, float] = {}
    topic_p95: Dict[str, float] = {}
    topic_max: Dict[str, float] = {}
    root_by_method: Dict[str, List[float]] = {}

    for label, est, rows, _c in methods:
        sub = [
            r
            for r in rows
            if str(r.get("topic_phi_estimator", "")).strip().lower() == est
            and abs(float(r.get("q_train", float("nan"))) - float(qtr)) <= 1e-12
            and abs(float(r.get("q_leaf", float("nan"))) - float(r.get("q_internal", float("nan")))) <= 1e-12
            and float(r.get("q_leaf")) in q_infer_vals
        ]
        topic_err[label] = _median(float(r.get("topic_phi_l2_error_mean", float("nan"))) for r in sub)
        topic_p95[label] = _median(float(r.get("topic_phi_l2_error_p95", float("nan"))) for r in sub)
        topic_max[label] = _median(float(r.get("topic_phi_l2_error_max", float("nan"))) for r in sub)
        root_by_method[label] = [
            _median(float(r.get("raw", float("nan"))) for r in sub if abs(float(r.get("q_leaf")) - q) <= 1e-12)
            for q in q_infer_vals
        ]

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 5.3), constrained_layout=True)
    ax_phi, ax_root = axes

    labels = [m[0] for m in methods]
    xs = np.arange(len(labels), dtype=np.float64)
    phi_vals = [float(topic_err.get(l, float("nan"))) for l in labels]
    phi_p95 = [float(topic_p95.get(l, float("nan"))) for l in labels]

    ax_phi.bar(xs, phi_vals, color=[m[3] for m in methods], alpha=0.85)
    ax_phi.set_xticks(xs)
    ax_phi.set_xticklabels(["spectral\n_numpy", "tensor\n_lda", "neural\n_ctreepo"])
    ax_phi.set_ylabel("topic φ L2 error (mean, aligned)")
    ax_phi.set_title("Upstream topic-word estimation quality")
    for i, (v, p95) in enumerate(zip(phi_vals, phi_p95)):
        mx = float(topic_max.get(labels[i], float("nan")))
        if math.isfinite(v) and math.isfinite(p95) and math.isfinite(mx):
            ax_phi.text(
                float(i),
                float(v),
                f"mean={v:.3f}\np95={p95:.3f}\nmax={mx:.3f}",
                ha="center",
                va="bottom",
                fontsize=10.0,
            )

    for label, _est, _rows, color in methods:
        ys = [_plot_floor(v) for v in root_by_method.get(label, [float("nan")] * len(q_infer_vals))]
        ax_root.plot(q_infer_vals, ys, marker="o", linewidth=2.0, color=color, label=label)
    ax_root.set_yscale("log")
    ax_root.set_ylim(PLOT_FLOOR, ERROR_AXIS_TOP)
    ax_root.axhline(CEILING_THRESHOLD, color="#666666", linestyle="--", linewidth=1.0)
    ax_root.set_xlabel(f"{DECISION_LABEL} ({DECISION_SYMBOL})")
    ax_root.set_ylabel("root L1 (log)")
    ax_root.set_title("End-to-end error vs decision-time visibility\n(fixed q_train=0.1)")
    ax_root.legend(frameon=False, fontsize=9.5, loc="upper right")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240)
    fig.savefig(out_pdf)
    plt.close(fig)

    return {
        "topic_phi_l2_error_mean": topic_err,
        "topic_phi_l2_error_p95": topic_p95,
        "topic_phi_l2_error_max": topic_max,
        "root_l1_median_by_method": root_by_method,
    }


def _scan_ctree_rows(output_root: Path) -> List[dict]:
    files = sorted(glob.glob(str(output_root / "segmented_lda_ctreepo" / "**" / "*.json"), recursive=True))
    rows: List[dict] = []
    for fp in files:
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        m = payload.get("metrics", {}) or {}
        budgeted = _as_float(((m.get("estimated_calibrated_budgeted") or {}).get("root_l1_mean")))
        oracle = _as_float(((m.get("oracle_tree") or {}).get("root_l1_mean")))
        if budgeted is None:
            continue
        rows.append(
            {
                "train_docs": int(cfg.get("n_books_train", -1)),
                "q_train": float(cfg.get("calibration_leaf_query_rate", float("nan"))),
                "q_leaf": float(cfg.get("eval_leaf_query_rate", float("nan"))),
                "q_internal": float(cfg.get("eval_internal_query_rate", float("nan"))),
                "raw": float(budgeted),
                "oracle": float(oracle) if oracle is not None else float("nan"),
                "calibration_samples": int(payload.get("calibration_samples", 0) or 0),
                "topic_phi_estimator": str(cfg.get("topic_phi_estimator", "")),
                "topic_phi_l2_error_mean": _float_or_nan((payload.get("topic_meta") or {}).get("topic_phi_l2_error_mean")),
                "topic_phi_l2_error_p95": _float_or_nan((payload.get("topic_meta") or {}).get("topic_phi_l2_error_p95")),
                "topic_phi_l2_error_max": _float_or_nan((payload.get("topic_meta") or {}).get("topic_phi_l2_error_max")),
            }
        )
    return rows


def _write_md_table(rows: Sequence[Sequence[str]], *, headers: Sequence[str]) -> List[str]:
    lines: List[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return lines


def main() -> int:
    args = _parse_args()
    _setup_style()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_root = Path(args.baseline_output_root)
    tuning_root = Path(args.tuning_output_root) if args.tuning_output_root is not None else None
    eps_den = float(args.norm_eps_den)

    # Collect baseline fixed slices.
    segment = _collect_segment_fixed(baseline_root, eps_den=eps_den)
    ctree = _collect_ctree_fixed(baseline_root, eps_den=eps_den)
    markov = _collect_markov_fixed_baseline(baseline_root, eps_den=eps_den)

    endpoints = _build_baseline_endpoints(segment, ctree, markov_fixed={"families": markov.get("families")})
    fig_a_png = out_dir / "A_endpoints_table_and_dotplot.png"
    fig_a_pdf = out_dir / "A_endpoints_table_and_dotplot.pdf"
    _plot_endpoints_dotplot(out_png=fig_a_png, out_pdf=fig_a_pdf, endpoints=endpoints)

    # Figure B pages (baseline surfaces).
    raw_norm = mcolors.LogNorm(vmin=PLOT_FLOOR, vmax=ERROR_AXIS_TOP)
    norm_norm = mcolors.Normalize(vmin=0.0, vmax=1.2)
    cmap = "RdYlGn_r"

    # Segment (strips).
    seg_fixed = (segment.get("fixed") or {})
    seg_lanes = (seg_fixed.get("lanes") or {})
    seg_true = (seg_lanes.get("phi_true") or {})
    seg_embed = (seg_lanes.get("phi_embedding_spectral") or {})
    seg_q = [float(x) for x in (seg_true.get("q_train") or [])]
    seg_strip_raw = np.asarray(
        [
            [float(v) for v in (seg_true.get("raw_median") or [])],
            [float(v) for v in (seg_embed.get("raw_median") or [])],
        ],
        dtype=np.float64,
    )
    seg_strip_norm = np.asarray(
        [
            [float(v) for v in (seg_true.get("norm_gap") or [])],
            [float(v) for v in (seg_embed.get("norm_gap") or [])],
        ],
        dtype=np.float64,
    )
    _plot_heatmap_page(
        out_png=out_dir / "B_segment_raw_strip.png",
        out_pdf=out_dir / "B_segment_raw_strip.pdf",
        arr=np.clip(seg_strip_raw, PLOT_FLOOR, ERROR_AXIS_TOP),
        xvals=seg_q,
        yvals=[0.0, 1.0],
        yticklabels=["phi=true", "phi=embedding"],
        title="Figure B (Segment): Raw error vs learn-time oracle visibility",
        xlabel=f"{LEARN_LABEL} ({LEARN_SYMBOL})",
        ylabel="lane",
        cmap=cmap,
        norm=raw_norm,
        note="Segment has no native decision-time stage.",
    )
    _plot_heatmap_page(
        out_png=out_dir / "B_segment_norm_strip.png",
        out_pdf=out_dir / "B_segment_norm_strip.pdf",
        arr=np.clip(seg_strip_norm, -0.02, 1.2),
        xvals=seg_q,
        yvals=[0.0, 1.0],
        yticklabels=["phi=true", "phi=embedding"],
        title="Figure B (Segment): Normalized progress vs learn-time oracle visibility",
        xlabel=f"{LEARN_LABEL} ({LEARN_SYMBOL})",
        ylabel="lane",
        cmap=cmap,
        norm=norm_norm,
        note="Gray/N/A means baseline≈ceiling (no measurable improvable gap).",
    )

    # C-TreePO (heatmaps).
    ct_fixed = ctree.get("fixed") or {}
    ct_qtr = [float(x) for x in (ct_fixed.get("q_train") or [])]
    ct_qinf = [float(x) for x in (ct_fixed.get("q_infer") or [])]
    ct_raw = np.asarray(ct_fixed.get("matrix_raw") or [], dtype=np.float64)
    ct_norm = np.asarray(ct_fixed.get("matrix_norm") or [], dtype=np.float64)
    _plot_heatmap_page(
        out_png=out_dir / "B_ctree_raw_surface.png",
        out_pdf=out_dir / "B_ctree_raw_surface.pdf",
        arr=np.clip(ct_raw, PLOT_FLOOR, ERROR_AXIS_TOP),
        xvals=ct_qinf,
        yvals=ct_qtr,
        title="Figure B (C-TreePO): Raw error surface (within-family)",
        xlabel=f"{DECISION_LABEL} ({DECISION_SYMBOL})",
        ylabel=f"{LEARN_LABEL} ({LEARN_SYMBOL})",
        cmap=cmap,
        norm=raw_norm,
        note="Raw error is C-TreePO root L1 (not comparable to Markov MAE).",
    )
    _plot_heatmap_page(
        out_png=out_dir / "B_ctree_norm_surface.png",
        out_pdf=out_dir / "B_ctree_norm_surface.pdf",
        arr=np.clip(ct_norm, -0.02, 1.2),
        xvals=ct_qinf,
        yvals=ct_qtr,
        title="Figure B (C-TreePO): Normalized progress surface (cross-family comparable)",
        xlabel=f"{DECISION_LABEL} ({DECISION_SYMBOL})",
        ylabel=f"{LEARN_LABEL} ({LEARN_SYMBOL})",
        cmap=cmap,
        norm=norm_norm,
        note="Normalized gap-to-ceiling: 0=ceiling reached, 1=baseline.",
    )

    # Markov surfaces (additive + neural baseline).
    mk_fams = markov.get("families") or {}
    for fam_key, fam_title in [("additive", "Markov additive"), ("neural", "Markov neural")]:
        fd = mk_fams.get(fam_key) or {}
        qtr = [float(x) for x in (fd.get("q_train") or [])]
        qinf = [float(x) for x in (fd.get("q_infer") or [])]
        raw = np.asarray(fd.get("matrix_raw") or [], dtype=np.float64)
        norm = np.asarray(fd.get("matrix_norm") or [], dtype=np.float64)
        _plot_heatmap_page(
            out_png=out_dir / f"B_{fam_key}_raw_surface.png",
            out_pdf=out_dir / f"B_{fam_key}_raw_surface.pdf",
            arr=np.clip(raw, PLOT_FLOOR, ERROR_AXIS_TOP),
            xvals=qinf,
            yvals=qtr,
            title=f"Figure B ({fam_title}): Raw error surface (within-family)",
            xlabel=f"{DECISION_LABEL} ({DECISION_SYMBOL})",
            ylabel=f"{LEARN_LABEL} ({LEARN_SYMBOL})",
            cmap=cmap,
            norm=raw_norm,
            note="Raw error is Markov root MAE (not comparable to C-TreePO root L1).",
        )
        _plot_heatmap_page(
            out_png=out_dir / f"B_{fam_key}_norm_surface.png",
            out_pdf=out_dir / f"B_{fam_key}_norm_surface.pdf",
            arr=np.clip(norm, -0.02, 1.2),
            xvals=qinf,
            yvals=qtr,
            title=f"Figure B ({fam_title}): Normalized progress surface (cross-family comparable)",
            xlabel=f"{DECISION_LABEL} ({DECISION_SYMBOL})",
            ylabel=f"{LEARN_LABEL} ({LEARN_SYMBOL})",
            cmap=cmap,
            norm=norm_norm,
            note="Normalized gap-to-ceiling: 0=ceiling reached, 1=baseline.",
        )

    # Figure C: budget frontiers + crosswalk tables.
    crosswalk: Dict[str, object] = {}

    # Build points for each lane (raw + norm) from baseline surfaces.
    def _points_from_matrix(qtr: Sequence[float], qinf: Optional[Sequence[float]], mat: Sequence[Sequence[float]]) -> List[FrontierPoint]:
        out: List[FrontierPoint] = []
        for i, qt in enumerate(qtr):
            if qinf is None:
                v = float(mat[i][0]) if (i < len(mat) and mat[i]) else float("nan")
                out.append(FrontierPoint(budget=_budget_rate(float(qt), None), value=float(v), q_train=float(qt), q_infer=None))
                continue
            for j, qi in enumerate(qinf):
                v = float(mat[i][j]) if (i < len(mat) and j < len(mat[i])) else float("nan")
                out.append(FrontierPoint(budget=_budget_rate(float(qt), float(qi)), value=float(v), q_train=float(qt), q_infer=float(qi)))
        return out

    # Segment phi=true/norm and phi=embedding/norm.
    seg_true_pts_raw = [FrontierPoint(_budget_rate(float(q), None), float(v), float(q), None) for q, v in zip(seg_q, seg_true.get("raw_median") or [])]
    seg_true_pts_norm = [FrontierPoint(_budget_rate(float(q), None), float(v), float(q), None) for q, v in zip(seg_q, seg_true.get("norm_gap") or [])]
    seg_emb_pts_raw = [FrontierPoint(_budget_rate(float(q), None), float(v), float(q), None) for q, v in zip(seg_q, seg_embed.get("raw_median") or [])]
    seg_emb_pts_norm = [FrontierPoint(_budget_rate(float(q), None), float(v), float(q), None) for q, v in zip(seg_q, seg_embed.get("norm_gap") or [])]

    # C-TreePO.
    ct_pts_raw = _points_from_matrix(ct_qtr, ct_qinf, ct_fixed.get("matrix_raw") or [])
    ct_pts_norm = _points_from_matrix(ct_qtr, ct_qinf, ct_fixed.get("matrix_norm") or [])

    # Markov.
    mk_add = mk_fams.get("additive") or {}
    mk_neu = mk_fams.get("neural") or {}
    add_pts_raw = _points_from_matrix(mk_add.get("q_train") or [], mk_add.get("q_infer") or [], mk_add.get("matrix_raw") or [])
    add_pts_norm = _points_from_matrix(mk_add.get("q_train") or [], mk_add.get("q_infer") or [], mk_add.get("matrix_norm") or [])
    neu_pts_raw = _points_from_matrix(mk_neu.get("q_train") or [], mk_neu.get("q_infer") or [], mk_neu.get("matrix_raw") or [])
    neu_pts_norm = _points_from_matrix(mk_neu.get("q_train") or [], mk_neu.get("q_infer") or [], mk_neu.get("matrix_norm") or [])

    frontiers = {
        "Segment phi=true": {
            "color": SEG_TRUE_COLOR,
            "raw": _frontier_with_crosswalk(seg_true_pts_raw),
            "norm_valid": bool(seg_true.get("norm_valid", False)),
            "norm": _frontier_with_crosswalk(seg_true_pts_norm) if bool(seg_true.get("norm_valid", False)) else [],
        },
        "Segment phi=embedding": {
            "color": SEG_EMBED_COLOR,
            "raw": _frontier_with_crosswalk(seg_emb_pts_raw),
            "norm_valid": bool(seg_embed.get("norm_valid", False)),
            "norm": _frontier_with_crosswalk(seg_emb_pts_norm) if bool(seg_embed.get("norm_valid", False)) else [],
        },
        "C-TreePO": {
            "color": CTREE_COLOR,
            "raw": _frontier_with_crosswalk(ct_pts_raw),
            "norm_valid": bool(ct_fixed.get("norm_valid", False)),
            "norm": _frontier_with_crosswalk(ct_pts_norm) if bool(ct_fixed.get("norm_valid", False)) else [],
        },
        "Markov additive": {
            "color": MARKOV_ADD_COLOR,
            "raw": _frontier_with_crosswalk(add_pts_raw),
            "norm_valid": bool(mk_add.get("norm_valid", True)),
            "norm": _frontier_with_crosswalk(add_pts_norm),
        },
        "Markov neural": {
            "color": MARKOV_NEURAL_COLOR,
            "raw": _frontier_with_crosswalk(neu_pts_raw),
            "norm_valid": bool(mk_neu.get("norm_valid", True)),
            "norm": _frontier_with_crosswalk(neu_pts_norm),
        },
    }
    crosswalk["baseline_frontiers"] = frontiers

    # Plot C normalized (overlay) and C raw (small multiples by family).
    # C normalized overlay.
    fig, ax = plt.subplots(1, 1, figsize=(7.3, 5.4), constrained_layout=True)
    for name, lane in frontiers.items():
        if not bool(lane.get("norm_valid", False)):
            continue
        pts = lane.get("norm") or []
        xs = [float(p["budget"]) for p in pts]
        ys = [_clip_norm(p["best_value"]) for p in pts]
        ax.plot(xs, ys, marker="o", linewidth=2.0, color=str(lane.get("color")), label=name)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.03, 1.25)
    ax.axhline(0.0, color="#1a9850", linestyle="--", linewidth=1.1)
    ax.axhline(1.0, color="#666666", linestyle=":", linewidth=1.1)
    ax.set_xlabel("B_rate = 0.5*q_train + 0.5*q_infer  (Segment uses B_rate=q_train)")
    ax.set_ylabel("normalized gap-to-ceiling")
    ax.set_title("Figure C2: Budget frontier (normalized; cross-family comparable)")
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    ax.text(
        0.02,
        0.02,
        "This is an envelope: best error achievable with budget ≤ B_rate.\nCross-family comparison is valid in normalized space only.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.0,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.95),
    )
    fig.savefig(out_dir / "C_budget_frontier_norm.png", dpi=240)
    fig.savefig(out_dir / "C_budget_frontier_norm.pdf")
    plt.close(fig)

    # C raw: separate panels by metric family (Segment, Markov, C-TreePO).
    fig, axes = plt.subplots(3, 1, figsize=(7.3, 9.6), constrained_layout=True)
    ax_seg, ax_mk, ax_ct = axes

    for name, lane in frontiers.items():
        pts = lane.get("raw") or []
        xs = [float(p["budget"]) for p in pts]
        ys = [_plot_floor(p["best_value"]) for p in pts]
        if name.startswith("Segment"):
            ax_seg.plot(xs, ys, marker="o", linewidth=2.0, color=str(lane.get("color")), label=name)
        elif name.startswith("Markov"):
            ax_mk.plot(xs, ys, marker="o", linewidth=2.0, color=str(lane.get("color")), label=name)
        elif name == "C-TreePO":
            ax_ct.plot(xs, ys, marker="o", linewidth=2.0, color=str(lane.get("color")), label=name)

    for axx, title, ylabel in [
        (ax_seg, "Segment raw error frontier (root MAE)", "root MAE (log)"),
        (ax_mk, "Markov raw error frontier (root MAE)", "root MAE (log)"),
        (ax_ct, "C-TreePO raw error frontier (root L1)", "root L1 (log)"),
    ]:
        axx.set_yscale("log")
        axx.set_ylim(PLOT_FLOOR, ERROR_AXIS_TOP)
        axx.axhline(CEILING_THRESHOLD, color="#666666", linestyle="--", linewidth=1.0)
        axx.set_xlim(-0.02, 1.02)
        axx.set_title(title)
        axx.set_ylabel(ylabel)
        axx.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax_ct.set_xlabel("B_rate (as defined above; not cross-family comparable in raw units)")
    fig.savefig(out_dir / "C_budget_frontier_raw.png", dpi=240)
    fig.savefig(out_dir / "C_budget_frontier_raw.pdf")
    plt.close(fig)

    # Tuning section: Markov deep dive + C-TreePO phi estimator ablation.
    tuning_diag: Dict[str, object] = {"present": False}
    fig_d_paths: Dict[str, str] = {}
    fig_e_paths: Dict[str, str] = {}
    if tuning_root is not None and tuning_root.exists():
        tuning_diag["present"] = True

        # Markov tuning deep dive.
        _t_learned, t_guided = _scan_markov_runs(tuning_root)
        d_diag = _plot_markov_neural_deep_dive(
            out_png=out_dir / "D_markov_neural_deep_dive_baseline_vs_adjust.png",
            out_pdf=out_dir / "D_markov_neural_deep_dive_baseline_vs_adjust.pdf",
            baseline_guided=markov.get("guided_rows") or [],
            tuning_guided=t_guided,
        )
        fig_d_paths = {
            "png": str((out_dir / "D_markov_neural_deep_dive_baseline_vs_adjust.png").resolve()),
            "pdf": str((out_dir / "D_markov_neural_deep_dive_baseline_vs_adjust.pdf").resolve()),
        }
        tuning_diag["markov_neural_deep_dive"] = d_diag
        # Markov tuning sanity checks (operational, decision-complete).
        def _guided_root_mae_stats(rows: Sequence[dict], *, q_train: float, scw: float, gov: str, q_infer: float) -> Dict[str, object]:
            sub = [
                r
                for r in rows
                if str(r.get("family")) == "neural"
                and abs(float(r.get("q_train", float("nan"))) - float(q_train)) <= 1e-12
                and abs(float(r.get("schedule_consistency_weight", 0.0)) - float(scw)) <= 1e-12
                and str(r.get("guidance_override_mode", "reset")) == str(gov)
                and abs(float(r.get("q_infer", float("nan"))) - float(q_infer)) <= 1e-12
            ]
            vals = [float(r.get("root_mae", float("nan"))) for r in sub if math.isfinite(float(r.get("root_mae", float("nan"))))]
            return {
                "n": int(len(vals)),
                "median": float(_median(vals)),
                "max": float(max(vals)) if vals else float("nan"),
            }

        def _learned_schedule_spread(rows: Sequence[dict], *, q_train: float, scw: float, gov: str) -> Dict[str, object]:
            sub = [
                r
                for r in rows
                if str(r.get("family")) == "neural"
                and abs(float(r.get("q_train", float("nan"))) - float(q_train)) <= 1e-12
                and abs(float(r.get("schedule_consistency_weight", 0.0)) - float(scw)) <= 1e-12
                and str(r.get("guidance_override_mode", "reset")) == str(gov)
            ]
            mean_vals = [float(r.get("schedule_spread_mean", float("nan"))) for r in sub if math.isfinite(float(r.get("schedule_spread_mean", float("nan"))))]
            p95_vals = [float(r.get("schedule_spread_p95", float("nan"))) for r in sub if math.isfinite(float(r.get("schedule_spread_p95", float("nan"))))]
            return {
                "n": int(len(mean_vals)),
                "schedule_spread_mean_median": float(_median(mean_vals)),
                "schedule_spread_p95_median": float(_median(p95_vals)),
            }

        baseline_learned_rows = markov.get("learned_rows") or []
        tuning_diag["markov_sanity"] = {
            "qi1_root_mae_adjust_scw0_qtr0p1": _guided_root_mae_stats(t_guided, q_train=0.1, scw=0.0, gov="adjust", q_infer=1.0),
            "qi1_root_mae_adjust_scw0_qtr1": _guided_root_mae_stats(t_guided, q_train=1.0, scw=0.0, gov="adjust", q_infer=1.0),
            "qi1_root_mae_adjust_scw0p1_qtr0p1": _guided_root_mae_stats(t_guided, q_train=0.1, scw=0.1, gov="adjust", q_infer=1.0),
            "qi1_root_mae_adjust_scw0p1_qtr1": _guided_root_mae_stats(t_guided, q_train=1.0, scw=0.1, gov="adjust", q_infer=1.0),
            "max_root_mae_at_qi1_adjust": float(
                max(
                    [
                        float(r.get("root_mae", float("nan")))
                        for r in t_guided
                        if str(r.get("family")) == "neural"
                        and str(r.get("guidance_override_mode", "reset")) == "adjust"
                        and abs(float(r.get("q_infer", float("nan"))) - 1.0) <= 1e-12
                        and math.isfinite(float(r.get("root_mae", float("nan"))))
                    ]
                    or [float("nan")]
                )
            ),
            "schedule_spread_qtr0p1_baseline_reset_scw0": _learned_schedule_spread(baseline_learned_rows, q_train=0.1, scw=0.0, gov="reset"),
            "schedule_spread_qtr0p1_tuned_adjust_scw0": _learned_schedule_spread(_t_learned, q_train=0.1, scw=0.0, gov="adjust"),
            "schedule_spread_qtr0p1_tuned_adjust_scw0p1": _learned_schedule_spread(_t_learned, q_train=0.1, scw=0.1, gov="adjust"),
            "schedule_spread_qtr1_baseline_reset_scw0": _learned_schedule_spread(baseline_learned_rows, q_train=1.0, scw=0.0, gov="reset"),
            "schedule_spread_qtr1_tuned_adjust_scw0": _learned_schedule_spread(_t_learned, q_train=1.0, scw=0.0, gov="adjust"),
            "schedule_spread_qtr1_tuned_adjust_scw0p1": _learned_schedule_spread(_t_learned, q_train=1.0, scw=0.1, gov="adjust"),
        }

        # C-TreePO tuning phi estimator ablation.
        tuning_ctree_rows = _scan_ctree_rows(tuning_root)
        e_diag = _plot_ctree_phi_ablation(
            out_png=out_dir / "E_ctree_phi_estimator_ablation.png",
            out_pdf=out_dir / "E_ctree_phi_estimator_ablation.pdf",
            baseline_rows_fixed=ctree.get("rows_fixed") or [],
            tuning_rows=tuning_ctree_rows,
        )
        fig_e_paths = {
            "png": str((out_dir / "E_ctree_phi_estimator_ablation.png").resolve()),
            "pdf": str((out_dir / "E_ctree_phi_estimator_ablation.pdf").resolve()),
        }
        tuning_diag["ctree_phi_ablation"] = e_diag
        # C-TreePO tuning sanity checks.
        def _ctree_stats(rows: Sequence[dict], *, est: str, q_infer: float) -> Dict[str, object]:
            sub = [
                r
                for r in rows
                if str(r.get("topic_phi_estimator", "")).strip().lower() == est
                and abs(float(r.get("q_train", float("nan"))) - 0.1) <= 1e-12
                and abs(float(r.get("q_leaf", float("nan"))) - float(q_infer)) <= 1e-12
                and abs(float(r.get("q_internal", float("nan"))) - float(q_infer)) <= 1e-12
            ]
            raw_vals = [float(r.get("raw", float("nan"))) for r in sub if math.isfinite(float(r.get("raw", float("nan"))))]
            oracle_vals = [float(r.get("oracle", float("nan"))) for r in sub if math.isfinite(float(r.get("oracle", float("nan"))))]
            return {
                "n": int(len(raw_vals)),
                "raw_median": float(_median(raw_vals)),
                "oracle_median": float(_median(oracle_vals)),
                "raw_max": float(max(raw_vals)) if raw_vals else float("nan"),
                "oracle_max": float(max(oracle_vals)) if oracle_vals else float("nan"),
            }

        baseline_ctree_rows = ctree.get("rows_fixed") or []
        tuning_diag["ctree_sanity"] = {
            "spectral_numpy_qi1": _ctree_stats(baseline_ctree_rows, est="spectral_numpy", q_infer=1.0),
            "tensor_lda_qi1": _ctree_stats(tuning_ctree_rows, est="tensor_lda", q_infer=1.0),
            "neural_ctreepo_qi1": _ctree_stats(tuning_ctree_rows, est="neural_ctreepo", q_infer=1.0),
        }

    # Render markdown.
    now = datetime.now(timezone.utc).isoformat()
    md_path = out_dir / "identifiable_zero_journal_appendix_latest.md"
    pdf_path = out_dir / "identifiable_zero_journal_appendix_latest.pdf"
    diag_path = out_dir / "identifiable_zero_journal_appendix_latest_diagnostics.json"

    lines: List[str] = []
    lines.extend(
        [
            "---",
            "title: Identifiable-Zero Oracle-Equivalence Appendix (v4)",
            "geometry: margin=0.7in",
            "fontsize: 12pt",
            "toc: true",
            "toc-depth: 2",
            "header-includes:",
            "  - \\usepackage{amsmath}",
            "  - \\usepackage{amssymb}",
            "  - \\usepackage{booktabs}",
            "---",
            "",
            f"- Generated: `{now}`",
            f"- Baseline output root: `{baseline_root}`",
            f"- Tuning output root: `{tuning_root}`" if tuning_root is not None else "- Tuning output root: `None`",
            f"- Normalization eps-den: `{eps_den:g}`",
            "",
            "## 1. Setup (Pedantically Clear)",
            "",
            "### 1.1 What this report is",
            "",
            "- This is an **appendix-style** report for the identifiable-zero oracle-equivalence suite.",
            "- It is designed for **human reading**: large typography, one figure per page, and explicit comparability rules.",
            "",
            "### 1.2 Knobs (what varies)",
            "",
            f"- `{LEARN_SYMBOL}` (**{LEARN_LABEL}**) = fraction of oracle information available during fitting/calibration.",
            f"- `{DECISION_SYMBOL}` (**{DECISION_LABEL}**) = fraction of oracle information revealed at evaluation/decision time.",
            "",
            "### 1.3 Report knob -> config crosswalk",
            "",
            "The report uses a single notation (`q_train`, `q_infer`), but each lane wires those knobs to different config fields:",
            "",
            "| lane | `q_train` (learn-time) | `q_infer` (decision-time) |",
            "| --- | --- | --- |",
            "| Segment (OPS recovery; 1-stage) | `audit_fraction` | N/A |",
            "| C-TreePO (segmented LDA) | `calibration_leaf_query_rate` | `eval_leaf_query_rate` + `eval_internal_query_rate` |",
            "| Markov OPS-count | `audit_fraction` | `eval_guidance_qs` (guided-eval curve points) |",
            "",
            "### 1.4 Fixed slice used in the main figures",
            "",
            f"- Segment: `train_docs`={FIXED_SEG_TRAIN_DOCS}, `lambda_multiplier`={FIXED_SEG_LAMBDA}.",
            f"- C-TreePO: `n_books_train`={FIXED_CTREE_TRAIN_DOCS} (and we filter to runs with >= {FIXED_CTREE_MIN_CAL_SAMPLES} calibration samples).",
            f"- Markov: `train_docs`={FIXED_MARKOV_TRAIN_DOCS}, `leaf_query_rate`={FIXED_MARKOV_LEAF_QUERY_RATE}, `include_root_query`={FIXED_MARKOV_INCLUDE_ROOT_QUERY}.",
            "",
            "### 1.5 Comparability rules (do not skip)",
            "",
            "- **Raw errors are only comparable *within* a family**, because units differ (C-TreePO uses root L1; Markov uses root MAE).",
            "- **Cross-family comparisons belong in normalized space only.**",
            "",
            "### 1.6 Normalization definition (cross-family comparable)",
            "",
            "For a lane with baseline error `E_base` and best observed ceiling `E_ceil`, define:",
            "",
            "$$g(E) = \\frac{E - E_{\\mathrm{ceil}}}{E_{\\mathrm{base}} - E_{\\mathrm{ceil}}}$$",
            "",
            "- `g(E)=0` means the lane reached its observed ceiling (best).",
            "- `g(E)=1` means the lane is at baseline difficulty.",
            "- If `E_base - E_ceil \\le \\varepsilon` (here $\\varepsilon$ is `norm-eps-den`), we report **N/A**.",
            "",
            "### 1.7 Budget definition (Figure C)",
            "",
            "- For 2-stage methods (C-TreePO, Markov): `B_rate = 0.5*q_train + 0.5*q_infer`.",
            "- For 1-stage Segment: `B_rate = q_train`.",
            "- Frontier is an **envelope**: best error achievable with budget **<=** `B_rate`.",
            "",
            "\\newpage",
            "",
            "## 2. Figure A: Endpoints (readable)",
            "",
            "### 2.1 Endpoint table (each endpoint is one line)",
            "",
        ]
    )

    rows = []
    for ep in endpoints:
        rows.append(
            [
                str(ep.get("endpoint_id", "")),
                str(ep.get("name", "")),
                str(ep.get("stage", "")),
                _fmt(ep.get("raw")),
                str(ep.get("norm_display", "")),
                str(ep.get("status", "")),
            ]
        )
    lines.extend(_write_md_table(rows, headers=["endpoint_id", "name", "stage", "raw", "normalized", "status"]))
    lines.extend(
        [
            "",
            "",
            f"![]({fig_a_png.name}){{width=100%}}",
            "",
            "\\newpage",
            "",
            "## 3. Figure B: Tradeoff surfaces (one per page)",
            "",
            "Raw pages are within-family only; normalized pages are cross-family comparable.",
            "",
            f"![](B_segment_raw_strip.png){{width=100%}}",
            "",
            "\\newpage",
            "",
            f"![](B_segment_norm_strip.png){{width=100%}}",
            "",
            "\\newpage",
            "",
            f"![](B_ctree_raw_surface.png){{width=100%}}",
            "",
            "\\newpage",
            "",
            f"![](B_ctree_norm_surface.png){{width=100%}}",
            "",
            "\\newpage",
            "",
            f"![](B_additive_raw_surface.png){{width=100%}}",
            "",
            "\\newpage",
            "",
            f"![](B_additive_norm_surface.png){{width=100%}}",
            "",
            "\\newpage",
            "",
            f"![](B_neural_raw_surface.png){{width=100%}}",
            "",
            "\\newpage",
            "",
            f"![](B_neural_norm_surface.png){{width=100%}}",
            "",
            "\\newpage",
            "",
            "## 4. Figure C: Budget frontiers + B->C crosswalk",
            "",
            "### 4.1 Normalized frontier (cross-family comparable)",
            "",
            f"![](C_budget_frontier_norm.png){{width=100%}}",
            "",
            "\\newpage",
            "",
            "### 4.2 Raw frontiers (NOT cross-family comparable; shown as small multiples)",
            "",
            f"![](C_budget_frontier_raw.png){{width=100%}}",
            "",
            "\\newpage",
            "",
            "### 4.3 Crosswalk: which (q_train,q_infer) points define the frontier?",
            "",
            "Figure C is derived from Figure B by taking a **budget envelope**:",
            "",
            "- For each `(q_train, q_infer)` grid point, compute its `B_rate`.",
            "- For each distinct `B_rate` value, define the frontier value as the minimum error among all grid points with budget `<= B_rate`.",
            "",
            "The tables below show the exact argmin `(q_train*, q_infer*)` point used at each budget gridpoint. `step='*'` marks budgets where the envelope strictly improves.",
            "",
        ]
    )

    # Crosswalk tables: show all budget gridpoints (pedantic, but prevents Figure-B/Figure-C misreads).
    for lane_name, lane in frontiers.items():
        raw_fw = lane.get("raw") or []
        if not raw_fw:
            continue
        norm_fw = lane.get("norm") or []
        norm_valid = bool(lane.get("norm_valid", False))
        norm_by_budget = {float(r.get("budget")): r for r in norm_fw} if (norm_valid and norm_fw) else {}

        lines.append(f"**{lane_name} (full crosswalk)**")
        rows = []
        for r in raw_fw:
            b = float(r.get("budget", float("nan")))
            step = "*" if bool(r.get("improved_at_budget", False)) else ""
            best_norm = "N/A"
            if norm_by_budget:
                nr = norm_by_budget.get(float(b))
                if nr is not None:
                    best_norm = _fmt(_clip_norm(nr.get("best_value")))
            rows.append(
                [
                    _fmt(b),
                    step,
                    _fmt(r.get("best_value")),
                    best_norm,
                    _fmt(r.get("argmin_q_train")),
                    _fmt(r.get("argmin_q_infer")) if r.get("argmin_q_infer") is not None else "-",
                ]
            )
        lines.extend(_write_md_table(rows, headers=["B_rate", "step", "best_raw", "best_norm", "q_train*", "q_infer*"]))
        lines.append("")

    lines.extend(
        [
            "\\newpage",
            "",
            "## 5. Neural operator deep dive (Markov neural merger)",
            "",
            "### 5.1 What the Markov neural merger is",
            "",
            "In `src/tree/markov_changepoint_ops_count_simulation.py`, the **LearnedCountSketch** state is:",
            "",
            "- a latent vector $h \\in \\mathbb{R}^{\\texttt{state\\_dim}}$ (learned), plus",
            "- explicit endpoint one-hots: `(first_regime, last_regime)`.",
            "",
            "A merge takes `(h_L, last_L, h_R, first_R)` and outputs a new latent `h_parent` via an MLP. This is **not guaranteed associative**, so different merge schedules can disagree (reported as schedule spread).",
            "",
            "### 5.2 What decision-time oracle visibility does in this sim",
            "",
            "During guided evaluation, each internal node is independently selected with probability `q_infer`. If selected, we override that node with the oracle count before continuing merges upward.",
            "",
            "For the neural merger, there are two semantics for the override (controlled by `guidance_override_mode`):",
            "",
            "- **reset** (baseline): replace `h` with a pure readout-aligned vector to hit the desired count.",
            "- **adjust** (tuning): minimally shift `h` only along the readout direction to hit the desired count, preserving orthogonal components.",
            "",
            "Reset can look 'broken' under partial guidance because it deletes any orthogonal information in `h` that the merger may rely on downstream. Adjust tests whether that is the main failure mode.",
            "",
            "### 5.3 Override math (exact; what the code does)",
            "",
            "The learned sketch has a linear readout `logit = w^T h + b` and a sigmoid that maps to a normalized count in `(0,1)`.",
            "",
            "Given an oracle target count, we form `t = clip(target_count / target_scale)` and `z = log(t/(1-t))` (with saturation at `t in {0,1}`). Let the current logit be $\\ell = w^T h + b$ and $\\|w\\|_2^2 = \\sum_i w_i^2$.",
            "",
            "- **reset:** $h \\leftarrow \\frac{z - b}{\\|w\\|_2^2} w$ (forces $h$ parallel to $w$).",
            "- **adjust:** $h \\leftarrow h + \\frac{z - \\ell}{\\|w\\|_2^2} w$ (preserves the orthogonal component of $h$).",
            "",
            "Additive sketches ignore `guidance_override_mode`: they store the count explicitly, so an override just sets the count coordinate.",
            "",
            "### 5.4 What changes in the tuning sweep (vs baseline suite)",
            "",
            "- Run only `model_family=neural` on CPU.",
            "- Fix `train_docs=8000`, `test_docs=2000`, `leaf_query_rate=1.0`, `include_root_query=true`.",
            "- Sweep `q_train in {0.1, 1.0}` (learn-time visibility via `audit_fraction`).",
            "- Sweep `q_infer in {0, 0.1, 0.25, 0.5, 0.75, 1.0}` with 8 stochastic trials each.",
            "- Use `guidance_override_mode=adjust` and `schedule_consistency_weight in {0.0, 0.1}`.",
            "",
            "### 5.5 What we check",
            "",
            "- **Sanity:** at `q_infer=1`, root error should be ~0 (full oracle visibility at decision time).",
            "- **Monotonicity:** root MAE should not get worse as `q_infer` increases (or at least be much less non-monotone).",
            "- **Regularization effect:** `schedule_consistency_weight=0.1` should reduce schedule spread and/or merge MAE.",
            "",
        ]
    )
    if tuning_diag.get("present"):
        lines.append(f"![]({Path(fig_d_paths['png']).name}){{width=100%}}")
        lines.append("")
        d_diag = tuning_diag.get("markov_neural_deep_dive") or {}
        m_sanity = tuning_diag.get("markov_sanity") or {}
        lines.extend(
            [
                "**Tuning diagnostics (Markov)**",
                "",
                f"- Monotonicity violations at `q_train=1.0`: scw=0.0 -> `{int(d_diag.get('monotonicity_violations_qtrain1_scw0', -1))}`, scw=0.1 -> `{int(d_diag.get('monotonicity_violations_qtrain1_scw0p1', -1))}`.",
                f"- Sanity at `q_infer=1`: max tuned root MAE = `{_fmt(m_sanity.get('max_root_mae_at_qi1_adjust'))}` (target <= `1e-12`).",
                "- Schedule spread medians at `q_train=1.0` (mean/p95): "
                f"baseline reset scw=0 -> `{_fmt((m_sanity.get('schedule_spread_qtr1_baseline_reset_scw0') or {}).get('schedule_spread_mean_median'))}`/"
                f"`{_fmt((m_sanity.get('schedule_spread_qtr1_baseline_reset_scw0') or {}).get('schedule_spread_p95_median'))}`, "
                f"tuned adjust scw=0 -> `{_fmt((m_sanity.get('schedule_spread_qtr1_tuned_adjust_scw0') or {}).get('schedule_spread_mean_median'))}`/"
                f"`{_fmt((m_sanity.get('schedule_spread_qtr1_tuned_adjust_scw0') or {}).get('schedule_spread_p95_median'))}`, "
                f"tuned adjust scw=0.1 -> `{_fmt((m_sanity.get('schedule_spread_qtr1_tuned_adjust_scw0p1') or {}).get('schedule_spread_mean_median'))}`/"
                f"`{_fmt((m_sanity.get('schedule_spread_qtr1_tuned_adjust_scw0p1') or {}).get('schedule_spread_p95_median'))}`.",
                "",
            ]
        )
    else:
        lines.extend(["- Tuning data not found; skipping Markov deep dive figure.", ""])

    lines.extend(
        [
            "\\newpage",
            "",
            "## 6. Neural operator deep dive (C-TreePO neural topic refiner)",
            "",
            "This suite uses a segmented-LDA generative model with $K$ topics and vocabulary size $V$.",
            "",
            "- $\\phi \\in \\mathbb{R}^{K\\times V}$ is the topic-word matrix (each row is a probability vector).",
            "- Each leaf segment has a latent topic-mixture $\\theta$; tokens are drawn from the induced word distribution.",
            "",
            "Topics are only identifiable up to permutation. This report measures topic error after best alignment, and the suite can optionally randomize topic order (`topic_phi_permute`) to ensure metrics are permutation-invariant.",
            "",
            "C-TreePO needs an estimate $\\hat{\\phi}$ to produce proxy leaf topic-mixtures, then it uses learn-time calibration (`q_train`) and decision-time guidance (`q_infer`) to control error.",
            "",
            "### 6.1 Estimators compared here",
            "",
            "- `spectral_numpy` (baseline): a lightweight spectral proxy on training leaves (SVD + k-means).",
            "- `tensor_lda` (tuning): a moment-based Tensor-LDA estimator on unlabeled training books.",
            "- `neural_ctreepo` (tuning): run a base estimator, then apply a lightweight CPU topic refiner using oracle-seeded topics.",
            "",
            "### 6.2 What `neural_ctreepo` does (update rule)",
            "",
            "Implementation lives in `src/tree/segment_lda_ops_weight_recovery_simulation.py` (function `_neural_refine_topics`, mode `ctreepo`).",
            "",
            "Given a base estimate $\\hat{\\phi}_{\\mathrm{base}}$ and a small set of seed topics $S$ (oracle anchors), it:",
            "",
            "Note: in this controlled ablation, the seed topics are an oracle input used to test whether a small amount of true topic information can repair the base estimator; they are not the same thing as `q_train`/`q_infer` budgets.",
            "",
            "- computes cosine similarities from each estimated topic to the seed topics,",
            "- turns those into weights with a softmax temperature, and",
            "- applies a residual correction that propagates the seed-topic error signal to all topics.",
            "",
            "Concretely, after building a weighted anchor estimate and weighted anchor truth, it updates:",
            "",
            "$$\\hat{\\phi} \\leftarrow \\mathrm{NormalizeRows}\\big(\\hat{\\phi}_{\\mathrm{base}} + \\beta (\\phi_{\\mathrm{anchor}} - \\hat{\\phi}_{\\mathrm{anchor}})\\big),$$",
            "",
            "then clamps the seed rows to the oracle and renormalizes. Here $\\beta$ is `neural_topic_operator_boost`.",
            "",
            "### 6.3 What decision-time oracle visibility means in this sim",
            "",
            "At evaluation time, C-TreePO independently reveals oracle information at leaves and internal nodes:",
            "",
            "- `eval_leaf_query_rate = q_infer` replaces that fraction of leaf topic-mixtures with oracle leaf $\\theta$.",
            "- `eval_internal_query_rate = q_infer` (coupled in this tuning) replaces that fraction of internal merges with oracle internal aggregates.",
            "",
            "### 6.4 What changes in the tuning ablation",
            "",
            "- Fix downstream settings and compare `tensor_lda` vs `neural_ctreepo` head-to-head.",
            "- Compare both against the baseline `spectral_numpy` lane from v1.",
            "",
        ]
    )
    if tuning_diag.get("present"):
        lines.append(f"![]({Path(fig_e_paths['png']).name}){{width=100%}}")
        lines.append("")
        e_diag = tuning_diag.get("ctree_phi_ablation") or {}
        phi_mean = (e_diag.get("topic_phi_l2_error_mean") or {}) if isinstance(e_diag, dict) else {}
        phi_p95 = (e_diag.get("topic_phi_l2_error_p95") or {}) if isinstance(e_diag, dict) else {}
        phi_max = (e_diag.get("topic_phi_l2_error_max") or {}) if isinstance(e_diag, dict) else {}
        root_rows = (e_diag.get("root_l1_median_by_method") or {}) if isinstance(e_diag, dict) else {}
        labels = ["spectral_numpy (baseline)", "tensor_lda (tuning)", "neural_ctreepo (tuning)"]
        rows = []
        for lab in labels:
            rvals = root_rows.get(lab) or [float("nan"), float("nan"), float("nan")]
            rows.append(
                [
                    lab,
                    _fmt(phi_mean.get(lab)),
                    _fmt(phi_p95.get(lab)),
                    _fmt(phi_max.get(lab)),
                    _fmt(rvals[0] if len(rvals) > 0 else float("nan")),
                    _fmt(rvals[1] if len(rvals) > 1 else float("nan")),
                    _fmt(rvals[2] if len(rvals) > 2 else float("nan")),
                ]
            )
        lines.extend(
            [
                "**Tuning summary table (C-TreePO)**",
                "",
            ]
        )
        lines.extend(
            _write_md_table(
                rows,
                headers=[
                    "estimator",
                    "phi_L2_mean",
                    "phi_L2_p95",
                    "phi_L2_max",
                    "root_L1(q=0)",
                    "root_L1(q=0.5)",
                    "root_L1(q=1.0)",
                ],
            )
        )
        lines.append("")
        c_sanity = tuning_diag.get("ctree_sanity") or {}
        lines.extend(
            [
                "**Sanity at q_infer=1 (raw_median / oracle_median)**",
                "",
                f"- spectral_numpy: `{_fmt((c_sanity.get('spectral_numpy_qi1') or {}).get('raw_median'))}` / `{_fmt((c_sanity.get('spectral_numpy_qi1') or {}).get('oracle_median'))}`",
                f"- tensor_lda: `{_fmt((c_sanity.get('tensor_lda_qi1') or {}).get('raw_median'))}` / `{_fmt((c_sanity.get('tensor_lda_qi1') or {}).get('oracle_median'))}`",
                f"- neural_ctreepo: `{_fmt((c_sanity.get('neural_ctreepo_qi1') or {}).get('raw_median'))}` / `{_fmt((c_sanity.get('neural_ctreepo_qi1') or {}).get('oracle_median'))}`",
                "",
            ]
        )
    else:
        lines.extend(["- Tuning data not found; skipping C-TreePO phi ablation figure.", ""])

    lines.extend(
        [
            "\\newpage",
            "",
            "## 7. Limits & next actions",
            "",
            "- These tuning sweeps test *plausible fixes/ablations*; they do not prove global optimality.",
            "- If the Markov neural lane remains weak even under `adjust` + schedule-consistency regularization, the conclusion is that the unstructured merger is a poor fit for OPS-style guidance without stronger architectural constraints.",
            "- If `neural_ctreepo` improves phi estimation but not end-to-end root error, that suggests downstream calibration/guidance dominates the error budget at the tested point.",
            "",
        ]
    )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pdf_emitted = False
    if bool(args.emit_pdf):
        try:
            pdf_emitted = _run_pandoc(md_path, pdf_path)
        except Exception:
            pdf_emitted = False

    diagnostics = {
        "generated_at": now,
        "baseline_output_root": str(baseline_root),
        "tuning_output_root": str(tuning_root) if tuning_root is not None else None,
        "pdf_emitted": bool(pdf_emitted),
        "figures": {
            "A_endpoints": {"png": str(fig_a_png.resolve()), "pdf": str(fig_a_pdf.resolve())},
            "C_frontier_norm": str((out_dir / "C_budget_frontier_norm.pdf").resolve()),
            "C_frontier_raw": str((out_dir / "C_budget_frontier_raw.pdf").resolve()),
            "D_markov_neural_deep_dive": fig_d_paths,
            "E_ctree_phi_ablation": fig_e_paths,
        },
        "baseline": {
            "segment": segment,
            "ctree_fixed_summary": {k: v for k, v in (ctree.get("fixed") or {}).items() if k != "matrix_raw" and k != "matrix_norm"},
            "markov_families_present": sorted(list((markov.get("families") or {}).keys())),
        },
        "crosswalk": crosswalk,
        "tuning": tuning_diag,
    }
    diag_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote_markdown | {md_path}")
    if pdf_emitted:
        print(f"wrote_pdf | {pdf_path}")
    else:
        print("pdf_not_emitted | pandoc/pdflatex not available or failed")
    print(f"wrote_diagnostics | {diag_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
