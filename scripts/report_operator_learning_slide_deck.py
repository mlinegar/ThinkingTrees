#!/usr/bin/env python3
"""Build a slide-style operator-learning deck from existing publication roots."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


SLIDE_W = 13.333
SLIDE_H = 7.5
PLOT_FLOOR = 1e-12
ERROR_AXIS_TOP = 1e2
CEILING_THRESHOLD = 1e-12

MARKOV_ADD_COLOR = "#4C78A8"
MARKOV_NEURAL_COLOR = "#E45756"
CTREE_COLOR = "#54A24B"
GUIDE_COLOR = "#F58518"
LIGHT_GRAY = "#9EA3A8"

CANONICAL_MARKOV_SLICE = {
    "train_docs": 8000,
    "feature_mode": "full",
    "state_dim": 32,
    "hidden_dim": 128,
    "local_law_weight": 0.2,
    "schedule_consistency_weight": 0.0,
    "guidance_override_mode": "reset",
}
CANONICAL_MARKOV_Q_TRAIN = 1.0
CANONICAL_CTREE_DOCS = 4096
CANONICAL_CTREE_Q_TRAIN = 0.1


@dataclass(frozen=True)
class LearnabilityRow:
    family: str
    q_train: float
    root_mae: float
    merge_mae: float
    schedule_spread_mean: float
    seed: int


@dataclass(frozen=True)
class GuidedRow:
    family: str
    q_train: float
    q_infer: float
    root_mae: float
    merge_mae: float
    effective_q_mean: float
    guided_internal_nodes_mean: float
    seed: int


@dataclass(frozen=True)
class CTreeBudgetRow:
    estimator: str
    seed_topics: int
    q_infer: float
    root_l1: float
    seed: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the operator-learning slide deck.")
    p.add_argument("--learnability-root", type=Path, required=True)
    p.add_argument("--publication-clean-root", type=Path, required=True)
    p.add_argument("--neural-overnight-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--figure-asset-dir", type=Path, default=None)
    p.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _as_float(x: object) -> Optional[float]:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _median(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if not xs:
        return float("nan")
    return float(statistics.median(xs))


def _plot_floor(x: object) -> float:
    v = _as_float(x)
    if v is None:
        return float("nan")
    return float(max(PLOT_FLOOR, float(v)))


def _percent_str(x: float) -> str:
    if x >= 1.0:
        return "100%"
    if x >= 0.1:
        return f"{100.0 * x:.0f}%"
    return f"{100.0 * x:.1f}%"


def _fmt_num(x: object) -> str:
    v = _as_float(x)
    if v is None:
        return "nan"
    if v == 0.0:
        return "0"
    if abs(v) >= 1000.0 or abs(v) < 1e-3:
        return f"{v:.3e}"
    return f"{v:.4f}"


def _setup_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    plt.rcParams.update(
        {
            "font.size": 12.0,
            "axes.titlesize": 13.5,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )


def _run_latexmk(tex_path: Path) -> bool:
    tex_path = tex_path.resolve()
    cwd = tex_path.parent
    if shutil.which("latexmk") is not None:
        subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return tex_path.with_suffix(".pdf").exists()
    if shutil.which("pdflatex") is None:
        return False
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=str(cwd),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return tex_path.with_suffix(".pdf").exists()


def _relative_tex_path(from_dir: Path, target: Path) -> str:
    return Path(os.path.relpath(str(target.resolve()), str(from_dir.resolve()))).as_posix()


def _objective_local_law_weight(payload: Dict) -> float:
    obj = payload.get("objective") or {}
    cfg = payload.get("config") or {}
    v = _as_float(obj.get("local_law_weight"))
    if v is not None:
        return float(v)
    v = _as_float(cfg.get("local_law_weight"))
    if v is not None:
        return float(v)
    return 0.0


def _cfg_matches_markov_slice(cfg: Dict, payload: Dict) -> bool:
    return (
        int(cfg.get("train_docs", -1) or -1) == int(CANONICAL_MARKOV_SLICE["train_docs"])
        and str(cfg.get("feature_mode", "")).strip() == str(CANONICAL_MARKOV_SLICE["feature_mode"])
        and int(cfg.get("state_dim", -1) or -1) == int(CANONICAL_MARKOV_SLICE["state_dim"])
        and int(cfg.get("hidden_dim", -1) or -1) == int(CANONICAL_MARKOV_SLICE["hidden_dim"])
        and abs(_objective_local_law_weight(payload) - float(CANONICAL_MARKOV_SLICE["local_law_weight"])) <= 1e-12
        and abs(float(_as_float(cfg.get("schedule_consistency_weight")) or 0.0) - float(CANONICAL_MARKOV_SLICE["schedule_consistency_weight"])) <= 1e-12
        and str(cfg.get("guidance_override_mode", "reset")).strip().lower()
        == str(CANONICAL_MARKOV_SLICE["guidance_override_mode"])
    )


def _scan_learnability(root: Path) -> List[LearnabilityRow]:
    files = sorted(glob.glob(str(root / "markov_changepoint_ops_count" / "**" / "*.json"), recursive=True))
    rows: List[LearnabilityRow] = []
    for fp in files:
        payload = _load_json(Path(fp))
        cfg = payload.get("config") or {}
        family = str(cfg.get("model_family", "")).strip().lower()
        if family not in {"neural", "additive"}:
            continue
        if not _cfg_matches_markov_slice(cfg, payload):
            continue
        q_train = _as_float(cfg.get("audit_fraction"))
        if q_train is None:
            continue
        learned = (payload.get("metrics") or {}).get("learned") or {}
        root_mae = _as_float(learned.get("root_mae"))
        merge_mae = _as_float(learned.get("merge_mae"))
        spread = _as_float(learned.get("schedule_spread_mean"))
        if root_mae is None or merge_mae is None or spread is None:
            continue
        rows.append(
            LearnabilityRow(
                family=family,
                q_train=float(q_train),
                root_mae=float(root_mae),
                merge_mae=float(merge_mae),
                schedule_spread_mean=float(spread),
                seed=int(cfg.get("seed", -1) or -1),
            )
        )
    return rows


def _scan_guided_publication_clean(root: Path) -> List[GuidedRow]:
    files = sorted(glob.glob(str(root / "markov_changepoint_ops_count" / "**" / "*.json"), recursive=True))
    rows: List[GuidedRow] = []
    for fp in files:
        payload = _load_json(Path(fp))
        cfg = payload.get("config") or {}
        family = str(cfg.get("model_family", "")).strip().lower()
        if family not in {"neural", "additive"}:
            continue
        if not _cfg_matches_markov_slice(cfg, payload):
            continue
        q_train = _as_float(cfg.get("audit_fraction"))
        if q_train is None or abs(float(q_train) - CANONICAL_MARKOV_Q_TRAIN) > 1e-12:
            continue
        pts = (((payload.get("metrics") or {}).get("guided_eval_curve") or {}).get("points") or [])
        for pt in pts:
            if not isinstance(pt, dict):
                continue
            q_infer = _as_float(pt.get("q"))
            root_mae = _as_float(pt.get("root_mae"))
            merge_mae = _as_float(pt.get("merge_mae"))
            eff_q = _as_float(pt.get("effective_q_mean"))
            guided_nodes = _as_float(pt.get("guided_internal_nodes_mean"))
            if None in {q_infer, root_mae, merge_mae, eff_q, guided_nodes}:
                continue
            rows.append(
                GuidedRow(
                    family=family,
                    q_train=float(q_train),
                    q_infer=float(q_infer),
                    root_mae=float(root_mae),
                    merge_mae=float(merge_mae),
                    effective_q_mean=float(eff_q),
                    guided_internal_nodes_mean=float(guided_nodes),
                    seed=int(cfg.get("seed", -1) or -1),
                )
            )
    return rows


def _scan_ctree_budget(root: Path) -> List[CTreeBudgetRow]:
    files = sorted(glob.glob(str(root / "segmented_lda_ctreepo" / "**" / "*.json"), recursive=True))
    rows: List[CTreeBudgetRow] = []
    for fp in files:
        payload = _load_json(Path(fp))
        cfg = payload.get("config") or {}
        if int(cfg.get("topic_phi_docs", 0) or 0) != CANONICAL_CTREE_DOCS:
            continue
        q_leaf = _as_float(cfg.get("eval_leaf_query_rate"))
        q_internal = _as_float(cfg.get("eval_internal_query_rate"))
        if q_leaf is None or q_internal is None or abs(float(q_leaf) - float(q_internal)) > 1e-12:
            continue
        est = str(cfg.get("topic_phi_estimator", "")).strip()
        if not est:
            continue
        root_l1 = _as_float((((payload.get("metrics") or {}).get("estimated_calibrated_budgeted") or {}).get("root_l1_mean")))
        if root_l1 is None:
            continue
        topic_meta = payload.get("topic_meta") or {}
        seed_topics = int(topic_meta.get("topic_phi_neural_seed_count", -1) or -1)
        rows.append(
            CTreeBudgetRow(
                estimator=est,
                seed_topics=seed_topics,
                q_infer=float(q_leaf),
                root_l1=float(root_l1),
                seed=int(cfg.get("seed", -1) or -1),
            )
        )
    return rows


def _aggregate_learnability(rows: Sequence[LearnabilityRow], family: str) -> Dict[str, object]:
    sub = [r for r in rows if r.family == family]
    qs = sorted({float(r.q_train) for r in sub})
    if not qs:
        return {"q_train": [], "root_mae": [], "merge_mae": [], "schedule_spread_mean": [], "counts": []}
    return {
        "q_train": qs,
        "root_mae": [_median(r.root_mae for r in sub if abs(float(r.q_train) - q) <= 1e-12) for q in qs],
        "merge_mae": [_median(r.merge_mae for r in sub if abs(float(r.q_train) - q) <= 1e-12) for q in qs],
        "schedule_spread_mean": [
            _median(r.schedule_spread_mean for r in sub if abs(float(r.q_train) - q) <= 1e-12)
            for q in qs
        ],
        "counts": [len([r for r in sub if abs(float(r.q_train) - q) <= 1e-12]) for q in qs],
    }


def _aggregate_guided(rows: Sequence[GuidedRow], family: str) -> Dict[str, object]:
    sub = [r for r in rows if r.family == family]
    qs = sorted({float(r.q_infer) for r in sub})
    if not qs:
        return {
            "q_infer": [],
            "root_mae": [],
            "merge_mae": [],
            "effective_q_mean": [],
            "guided_internal_nodes_mean": [],
            "counts": [],
        }
    return {
        "q_infer": qs,
        "root_mae": [_median(r.root_mae for r in sub if abs(float(r.q_infer) - q) <= 1e-12) for q in qs],
        "merge_mae": [_median(r.merge_mae for r in sub if abs(float(r.q_infer) - q) <= 1e-12) for q in qs],
        "effective_q_mean": [
            _median(r.effective_q_mean for r in sub if abs(float(r.q_infer) - q) <= 1e-12) for q in qs
        ],
        "guided_internal_nodes_mean": [
            _median(r.guided_internal_nodes_mean for r in sub if abs(float(r.q_infer) - q) <= 1e-12) for q in qs
        ],
        "counts": [len([r for r in sub if abs(float(r.q_infer) - q) <= 1e-12]) for q in qs],
    }


def _aggregate_ctree_budget(rows: Sequence[CTreeBudgetRow]) -> Dict[str, Dict[str, List[float]]]:
    max_seed_topics: Dict[str, int] = {}
    for row in rows:
        max_seed_topics[row.estimator] = max(max_seed_topics.get(row.estimator, -1), int(row.seed_topics))

    grouped: Dict[str, Dict[float, List[float]]] = {}
    for row in rows:
        if row.seed_topics >= 0 and int(row.seed_topics) != int(max_seed_topics.get(row.estimator, row.seed_topics)):
            continue
        est = row.estimator
        if est.startswith("neural_") and row.seed_topics >= 0:
            est = f"{est} (seed_topics={row.seed_topics})"
        grouped.setdefault(est, {}).setdefault(float(row.q_infer), []).append(float(row.root_l1))
    out: Dict[str, Dict[str, List[float]]] = {}
    for est, by_q in grouped.items():
        xs = sorted(by_q)
        out[est] = {
            "q_infer": xs,
            "root_l1": [_median(by_q[q]) for q in xs],
        }
    return out


def _load_bridge_series(diag_path: Path) -> Dict[str, object]:
    payload = _load_json(diag_path)
    diag = (payload.get("diagnostics") or {})
    evidence = diag.get("neural_lag_evidence") or {}
    ctree = evidence.get("ctree_reference") or {}
    add = evidence.get("markov_additive") or {}
    if not ctree or not add:
        raise ValueError("publication-clean diagnostics missing fixed-slice bridge series")
    ctree_series_raw = ctree.get("series") or {}
    ctree_series = {float(k): v for k, v in ctree_series_raw.items()}
    ctree_q = sorted(ctree_series)
    ctree_raw = [float((ctree_series.get(q) or {}).get("root_l1_mean", float("nan"))) for q in ctree_q]
    ctree_baseline = ctree_raw[0]
    ctree_ceiling = ctree_raw[-1]
    ctree_den = ctree_baseline - ctree_ceiling
    ctree_norm = [
        float((v - ctree_ceiling) / ctree_den) if math.isfinite(v) and abs(ctree_den) > 1e-12 else float("nan")
        for v in ctree_raw
    ]

    add_series = {float(k): v for k, v in add.items()}
    add_q = sorted(add_series)
    add_raw = [float((add_series.get(q) or {}).get("root_mae", float("nan"))) for q in add_q]
    add_baseline = add_raw[0]
    add_ceiling = add_raw[-1]
    add_den = add_baseline - add_ceiling
    add_norm = [
        float((v - add_ceiling) / add_den) if math.isfinite(v) and abs(add_den) > 1e-12 else float("nan")
        for v in add_raw
    ]

    fixed = evidence.get("fixed_slice") or {}
    return {
        "ctree_q_infer": ctree_q,
        "ctree_norm": ctree_norm,
        "markov_add_q_infer": add_q,
        "markov_add_norm": add_norm,
        "ctree_context": {
            "q_train": float(((fixed.get("ctree") or {}).get("learn_time_oracle_visibility")) or CANONICAL_CTREE_Q_TRAIN),
            "topic_phi_docs": (fixed.get("ctree") or {}).get("train_docs"),
        },
        "markov_context": {
            "q_train": float(((fixed.get("markov") or {}).get("learn_time_oracle_visibility")) or CANONICAL_MARKOV_Q_TRAIN),
            "train_docs": (fixed.get("markov") or {}).get("train_docs"),
        },
    }


def _assert_nonempty(name: str, rows: Sequence[object]) -> None:
    if not rows:
        raise ValueError(f"no rows found for {name}")


def _validate_markov_sources(learn_rows: Sequence[LearnabilityRow], guided_rows: Sequence[GuidedRow]) -> None:
    neural_learn = [r for r in learn_rows if r.family == "neural"]
    additive_learn = [r for r in learn_rows if r.family == "additive"]
    neural_guided = [r for r in guided_rows if r.family == "neural"]
    additive_guided = [r for r in guided_rows if r.family == "additive"]
    _assert_nonempty("learnability neural canonical slice", neural_learn)
    _assert_nonempty("learnability additive canonical slice", additive_learn)
    _assert_nonempty("publication-clean neural canonical slice", neural_guided)
    _assert_nonempty("publication-clean additive canonical slice", additive_guided)

    neural_learn_q = sorted({float(r.q_train) for r in neural_learn})
    guided_q = sorted({float(r.q_infer) for r in neural_guided})
    if len(neural_learn_q) < 2:
        raise ValueError("need >=2 q_train points in learnability neural slice")
    if len(guided_q) < 2:
        raise ValueError("need >=2 q_infer points in publication-clean neural slice")


def _plot_markov_train(agg_neural: Dict[str, object], agg_add: Dict[str, object], out_pdf: Path) -> None:
    qn = np.array(agg_neural["q_train"], dtype=np.float64)
    qa = np.array(agg_add["q_train"], dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 6.2), sharex=True, constrained_layout=True)
    metrics = [
        ("root_mae", "held-out root MAE (log)"),
        ("merge_mae", "merge MAE (log)"),
        ("schedule_spread_mean", "schedule spread (log)"),
    ]
    for ax, (key, ylabel) in zip(axes, metrics):
        yn = np.array([_plot_floor(v) for v in agg_neural[key]], dtype=np.float64)
        ya = np.array([_plot_floor(v) for v in agg_add[key]], dtype=np.float64)
        ax.plot(qn, yn, color=MARKOV_NEURAL_COLOR, marker="o", linewidth=2.4, label="Markov neural")
        if qa.size:
            ax.plot(
                qa,
                ya,
                color=MARKOV_ADD_COLOR,
                marker="o",
                linewidth=1.8,
                linestyle="--",
                alpha=0.55,
                label="Markov additive reference",
            )
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", alpha=0.4)
        ax.axhline(CEILING_THRESHOLD, color="#666666", linestyle=":", linewidth=0.9)
    axes[0].legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("learn-time oracle visibility ($q_{train}$)")
    axes[-1].set_xticks(qn)
    axes[-1].set_xticklabels([_percent_str(float(x)) for x in qn], rotation=0)
    fig.suptitle("Markov neural operator under more learn-time labels", fontsize=14)
    fig.savefig(out_pdf)
    plt.close(fig)


def _plot_markov_infer(agg_neural: Dict[str, object], agg_add: Dict[str, object], out_pdf: Path) -> None:
    qn = np.array(agg_neural["q_infer"], dtype=np.float64)
    qa = np.array(agg_add["q_infer"], dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.9), sharex=True, constrained_layout=True)
    for ax, key, ylabel in (
        (axes[0], "root_mae", "root MAE (log)"),
        (axes[1], "merge_mae", "merge MAE (log)"),
    ):
        yn = np.array([_plot_floor(v) for v in agg_neural[key]], dtype=np.float64)
        ya = np.array([_plot_floor(v) for v in agg_add[key]], dtype=np.float64)
        ax.plot(qn, yn, color=MARKOV_NEURAL_COLOR, marker="o", linewidth=2.4, label="Markov neural")
        ax.plot(qa, ya, color=MARKOV_ADD_COLOR, marker="o", linewidth=2.0, label="Markov additive")
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", alpha=0.4)
        ax.axhline(CEILING_THRESHOLD, color="#666666", linestyle=":", linewidth=0.9)
    axes[0].legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("decision-time oracle visibility ($q_{infer}$)")
    axes[-1].set_xticks(qn)
    axes[-1].set_xticklabels([_percent_str(float(x)) for x in qn])
    fig.suptitle("Same Markov slice, now varying test-time help", fontsize=14)
    fig.savefig(out_pdf)
    plt.close(fig)


def _plot_markov_override(agg_neural: Dict[str, object], out_pdf: Path) -> None:
    qs = np.array(agg_neural["q_infer"], dtype=np.float64)
    eff_q = np.array(agg_neural["effective_q_mean"], dtype=np.float64)
    nodes = np.array(agg_neural["guided_internal_nodes_mean"], dtype=np.float64)
    total_nodes = float(np.nanmax(nodes)) if nodes.size else float("nan")
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.8), sharex=True, constrained_layout=True)
    axes[0].plot(qs, eff_q, color=GUIDE_COLOR, marker="o", linewidth=2.4, label="effective $q$")
    axes[0].plot(qs, qs, color=LIGHT_GRAY, linestyle="--", linewidth=1.2, label="ideal $q_{infer}$")
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].set_ylabel("effective guidance share")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(qs, nodes, color=MARKOV_NEURAL_COLOR, marker="o", linewidth=2.4)
    axes[1].set_ylabel(f"guided internal nodes\n(out of {int(total_nodes) if math.isfinite(total_nodes) else '?'})")
    axes[1].grid(True, alpha=0.4)
    axes[1].set_xlabel("decision-time oracle visibility ($q_{infer}$)")
    axes[1].set_xticks(qs)
    axes[1].set_xticklabels([_percent_str(float(x)) for x in qs])

    fig.suptitle("Late neural gains line up with actual intervention", fontsize=14)
    fig.savefig(out_pdf)
    plt.close(fig)


def _plot_ctree_budget(series: Dict[str, Dict[str, List[float]]], out_pdf: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.2), constrained_layout=True)
    palette = {
        "neural_ctreepo (seed_topics=1)": "#7A1FA2",
        "neural_hybrid (seed_topics=1)": "#8C564B",
        "neural_mergeable_sketch (seed_topics=1)": CTREE_COLOR,
        "neural_ctreepo": "#7A1FA2",
        "neural_hybrid": "#8C564B",
        "neural_mergeable_sketch": CTREE_COLOR,
    }
    for est, data in sorted(series.items()):
        xs = np.array(data["q_infer"], dtype=np.float64)
        ys = np.array([_plot_floor(v) for v in data["root_l1"]], dtype=np.float64)
        ax.plot(xs, ys, marker="o", linewidth=2.2, label=est, color=palette.get(est, None))
    ax.set_yscale("log")
    ax.set_ylim(PLOT_FLOOR, ERROR_AXIS_TOP)
    ax.axhline(CEILING_THRESHOLD, color="#666666", linestyle=":", linewidth=0.9)
    ax.set_xlabel("decision-time oracle visibility ($q_{infer}$)")
    ax.set_ylabel("root L1 (log)")
    ax.set_title("C-TreePO operator family, topic-phi docs = 4096")
    ax.legend(frameon=False, loc="upper right", fontsize=9.4)
    ax.grid(True, which="major", alpha=0.4)
    fig.savefig(out_pdf)
    plt.close(fig)


def _plot_ctree_bridge(bridge: Dict[str, object], out_pdf: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.1), constrained_layout=True)
    c_q = np.array(bridge["ctree_q_infer"], dtype=np.float64)
    c_y = np.array(bridge["ctree_norm"], dtype=np.float64)
    m_q = np.array(bridge["markov_add_q_infer"], dtype=np.float64)
    m_y = np.array(bridge["markov_add_norm"], dtype=np.float64)
    ax.plot(c_q, c_y, color=CTREE_COLOR, marker="o", linewidth=2.4, label="C-TreePO normalized progress")
    ax.plot(
        m_q,
        m_y,
        color=MARKOV_ADD_COLOR,
        marker="o",
        linewidth=2.0,
        linestyle="--",
        label="Markov additive normalized progress",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("decision-time oracle visibility ($q_{infer}$)")
    ax.set_ylabel("remaining gap / within-family baseline")
    ax.set_title("Publication Figure B read rule: compare normalized progress")
    ax.grid(True, alpha=0.4)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(out_pdf)
    plt.close(fig)


def _slide_tex_preamble() -> str:
    return r"""\documentclass[12pt]{article}
\usepackage[paperwidth=13.333in,paperheight=7.5in,margin=0in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{tikz}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\usetikzlibrary{positioning,calc,fit,backgrounds}
\definecolor{markovadd}{HTML}{4C78A8}
\definecolor{markovneural}{HTML}{E45756}
\definecolor{ctreegreen}{HTML}{54A24B}
\definecolor{guideorange}{HTML}{F58518}
\definecolor{softgray}{HTML}{F4F6F8}
\definecolor{midgray}{HTML}{D9DEE3}
\newcommand{\slidebox}[4]{
  \node[anchor=north west, rounded corners=7pt, draw=#1!70!black, fill=#1!8, line width=0.9pt, inner sep=10pt,
        align=left] at #3 {\parbox{#2}{\raggedright #4}};
}
\begin{document}
\begin{tikzpicture}[x=1in,y=1in]
\useasboundingbox (0,0) rectangle (13.333,7.5);
"""


def _slide_tex_end() -> str:
    return r"""\end{tikzpicture}
\end{document}
"""


def _build_operator_info_axes_slide(asset_dir: Path) -> str:
    dgp_rel = _relative_tex_path(asset_dir, asset_dir / "markov_changepoint_dgp_slide.pdf")
    lines = [
        _slide_tex_preamble(),
        r"\node[anchor=north west, font=\bfseries\fontsize{26}{28}\selectfont] at (0.45,7.08) {Two Ways Information Enters the Operator Story};",
        r"\node[anchor=north west, text=black!65, font=\fontsize{13}{15}\selectfont, text width=12.2in] at (0.47,6.66) {We separate \emph{learn-time} labels from \emph{decision-time} overrides. The first tell the model what summaries to emit; the second patch the tree while it is already running.};",
        rf"\node[anchor=north west] at (0.45,5.98) {{\includegraphics[width=7.45in]{{{dgp_rel}}}}};",
        r"\slidebox{markovadd}{4.65in}{(8.25,5.90)}{\bfseries Learn-time oracle visibility ($q_{train}$)\\[0.5em]\mdseries How often leaf-level labels are available during training. This is what teaches the operator what summary each span should emit.\\[0.6em]\textcolor{black!65}{In the deck: use this first to show what the neural operator internalizes before any test-time help.}}",
        r"\slidebox{guideorange}{4.65in}{(8.25,3.95)}{\bfseries Decision-time oracle visibility ($q_{infer}$)\\[0.5em]\mdseries How often internal decisions are overridden or confirmed during evaluation. This is external help at test time, not the same thing as better learned weights.\\[0.6em]\textcolor{black!65}{In the deck: use this second to show when gains come from intervention rather than from a stable learned merge rule.}}",
        r"\slidebox{ctreegreen}{4.65in}{(8.25,1.82)}{\bfseries Reading rule for the whole deck\\[0.5em]\mdseries 1. Start with the exact Markov control.\\ 2. Swap in the learned Markov operator.\\ 3. Vary $q_{train}$, then vary $q_{infer}$.\\ 4. Only then bridge to C-TreePO.}",
        _slide_tex_end(),
    ]
    return "\n".join(lines)


def _build_markov_neural_operator_slide(asset_dir: Path) -> str:
    lines = [
        _slide_tex_preamble(),
        r"\node[anchor=north west, font=\bfseries\fontsize{26}{28}\selectfont] at (0.45,7.08) {Markov Neural Operator: What Is Learned?};",
        r"\node[anchor=north west, text=black!65, font=\fontsize{13}{15}\selectfont, text width=12.15in] at (0.47,6.66) {The tree topology stays fixed. What changes is the state carried at each node: the additive control stores an exact count channel, while the neural path learns a latent count state and a learned merge map.};",
        r"\slidebox{markovadd}{4.05in}{(0.55,5.85)}{\bfseries Fixed by construction\\[0.45em]\mdseries Leaf partition and balanced tree schedule\\ Endpoint channels $a,b$ when \texttt{feature\_mode=full}\\ Final target remains the changepoint count at the root}",
        r"\slidebox{markovneural}{4.05in}{(8.75,5.85)}{\bfseries Learned in code (\texttt{LearnedCountSketch})\\[0.45em]\mdseries \texttt{encoder}: core leaf features $\rightarrow h$\\ \texttt{merger}: $(h_L,h_R,b_L,a_R)\rightarrow h'$\\ \texttt{readout}: $h \rightarrow \widehat{c}$\\ Endpoints are preserved explicitly; the latent count state is learned.}",
        r"\node[draw=midgray, fill=softgray, rounded corners=8pt, minimum width=2.2in, minimum height=0.9in, align=center] (leaf1) at (2.0,3.7) {{\bfseries Leaf 1}\\ features\\ $\downarrow$\\ $[h_1,a_1,b_1]$};",
        r"\node[draw=midgray, fill=softgray, rounded corners=8pt, minimum width=2.2in, minimum height=0.9in, align=center] (leaf2) at (5.0,3.7) {{\bfseries Leaf 2}\\ features\\ $\downarrow$\\ $[h_2,a_2,b_2]$};",
        r"\node[draw=markovneural!70!black, fill=markovneural!8, rounded corners=8pt, minimum width=2.35in, minimum height=1.0in, align=center] (merge12) at (3.5,2.15) {{\bfseries learned merger}\\ $[h_1,h_2,b_1,a_2]$\\ $\downarrow$\\ $[h_{12},a_1,b_2]$};",
        r"\node[draw=midgray, fill=softgray, rounded corners=8pt, minimum width=2.2in, minimum height=0.9in, align=center] (leaf3) at (8.35,3.7) {{\bfseries Leaf 3}\\ features\\ $\downarrow$\\ $[h_3,a_3,b_3]$};",
        r"\node[draw=midgray, fill=softgray, rounded corners=8pt, minimum width=2.2in, minimum height=0.9in, align=center] (leaf4) at (11.35,3.7) {{\bfseries Leaf 4}\\ features\\ $\downarrow$\\ $[h_4,a_4,b_4]$};",
        r"\node[draw=markovneural!70!black, fill=markovneural!8, rounded corners=8pt, minimum width=2.35in, minimum height=1.0in, align=center] (merge34) at (9.85,2.15) {{\bfseries learned merger}\\ $[h_3,h_4,b_3,a_4]$\\ $\downarrow$\\ $[h_{34},a_3,b_4]$};",
        r"\node[draw=markovneural!80!black, fill=markovneural!14, rounded corners=9pt, minimum width=2.65in, minimum height=1.15in, align=center] (root) at (6.68,0.65) {{\bfseries root readout}\\ learned merge of the two halves\\ then \texttt{readout}$(h_{root})\rightarrow \widehat{c}_{root}$};",
        r"\draw[->, line width=1.1pt] (leaf1.south) -- (merge12.north west);",
        r"\draw[->, line width=1.1pt] (leaf2.south) -- (merge12.north east);",
        r"\draw[->, line width=1.1pt] (leaf3.south) -- (merge34.north west);",
        r"\draw[->, line width=1.1pt] (leaf4.south) -- (merge34.north east);",
        r"\draw[->, line width=1.1pt] (merge12.south) -- (root.north west);",
        r"\draw[->, line width=1.1pt] (merge34.south) -- (root.north east);",
        _slide_tex_end(),
    ]
    return "\n".join(lines)


def _build_plot_slide_tex(
    *,
    title: str,
    subtitle: str,
    figure_rel: str,
    note_color: str,
    note_body: str,
    footer: Optional[str] = None,
    figure_width: str = "7.55in",
) -> str:
    lines = [
        _slide_tex_preamble(),
        rf"\node[anchor=north west, font=\bfseries\fontsize{{26}}{{28}}\selectfont] at (0.45,7.08) {{{title}}};",
        rf"\node[anchor=north west, text=black!65, font=\fontsize{{13}}{{15}}\selectfont, text width=12.2in] at (0.47,6.66) {{{subtitle}}};",
        rf"\node[anchor=north west] at (0.48,5.92) {{\includegraphics[width={figure_width}]{{{figure_rel}}}}};",
        rf"\slidebox{{{note_color}}}{{4.75in}}{{(8.18,5.92)}}{{{note_body}}}",
    ]
    if footer:
        lines.append(
            rf"\node[anchor=north west, text width=12.1in, font=\fontsize{{12.4}}{{14}}\selectfont, text=black!78] at (0.58,0.28) {{{footer}}};"
        )
    lines.append(_slide_tex_end())
    return "\n".join(lines)


def _build_takeaway_slide() -> str:
    lines = [
        _slide_tex_preamble(),
        r"\node[anchor=north west, font=\bfseries\fontsize{26}{28}\selectfont] at (0.45,7.08) {Takeaway: What Changes as Information Increases?};",
        r"\node[anchor=north west, text=black!65, font=\fontsize{13}{15}\selectfont, text width=12.15in] at (0.47,6.66) {The point of the deck is to keep three objects separate: the exact control, the learned Markov operator, and the approximate C-TreePO analogue.};",
        r"\slidebox{markovadd}{3.8in}{(0.60,5.75)}{\bfseries Exact / additive control\\[0.45em]\mdseries This is as close to the DGP as we can get. The tree stores the right object and merges it exactly. It is the theorem-backed benchmark.}",
        r"\slidebox{markovneural}{3.8in}{(4.80,5.75)}{\bfseries Markov neural operator\\[0.45em]\mdseries Same tree, but the count channel becomes a learned latent state. More \emph{train-time} labels teach better states; more \emph{test-time} overrides patch what was not internalized.}",
        r"\slidebox{ctreegreen}{3.8in}{(9.00,5.75)}{\bfseries C-TreePO bridge\\[0.45em]\mdseries Different family and different raw units. Read it through within-family normalized progress: a few labels can quickly approximate the useful operator behavior.}",
        r"\slidebox{guideorange}{12.15in}{(0.60,3.35)}{\bfseries Reading rule for the appendix / lecture\\[0.45em]\mdseries First ask: what exact state would make merging lossless? Then ask: what part of that state is learned? Then split information into learn-time $q_{train}$ and decision-time $q_{infer}$. Only after that should cross-family comparisons appear, and then only in normalized units.}",
        r"\node[anchor=north west, text width=12.0in, font=\fontsize{14}{16}\selectfont, text=black!82] at (0.60,1.45) {Short version: learn-time labels teach the operator; decision-time labels intervene on the tree. Exact/additive shows the target object, Markov neural shows the learned approximation, and C-TreePO shows the broader approximate analogue.};",
        _slide_tex_end(),
    ]
    return "\n".join(lines)


def _master_deck_tex(slide_pdfs: Sequence[Path], out_dir: Path) -> str:
    rels = [_relative_tex_path(out_dir, p) for p in slide_pdfs]
    lines = [
        r"\documentclass{article}",
        r"\usepackage[paperwidth=13.333in,paperheight=7.5in,margin=0in]{geometry}",
        r"\usepackage{pdfpages}",
        r"\pagestyle{empty}",
        r"\begin{document}",
    ]
    lines.extend([rf"\includepdf[pages=-,fitpaper=true]{{{rel}}}" for rel in rels])
    lines.extend([r"\end{document}", ""])
    return "\n".join(lines)


def _compile_existing_shared_slide(source_tex: Path) -> Path:
    if not source_tex.exists():
        raise FileNotFoundError(f"missing shared slide source: {source_tex}")
    if not _run_latexmk(source_tex):
        raise RuntimeError(f"failed to compile {source_tex}")
    return source_tex.with_suffix(".pdf")


def _render_slide(tex_path: Path, source: str, emit_pdf: bool) -> Path:
    _write_text(tex_path, source)
    if emit_pdf:
        if not _run_latexmk(tex_path):
            raise RuntimeError(f"failed to compile {tex_path}")
    return tex_path.with_suffix(".pdf")


def _seed_shared_conceptual_assets(repo_root: Path, asset_dir: Path) -> None:
    shared_names = [
        "markov_changepoint_dgp_slide.tex",
        "markov_changepoint_dgp_slide.tikz",
        "markov_changepoint_exact_merge_slide.tex",
        "markov_changepoint_exact_merge_slide.tikz",
    ]
    src_dir = repo_root / "paper" / "figures"
    for name in shared_names:
        src = src_dir / name
        dst = asset_dir / name
        if dst.exists():
            continue
        shutil.copyfile(src, dst)


def main() -> int:
    args = _parse_args()
    _setup_style()

    repo_root = _repo_root()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = (args.figure_asset_dir or (repo_root / "paper" / "figures")).resolve()
    asset_dir.mkdir(parents=True, exist_ok=True)
    _seed_shared_conceptual_assets(repo_root, asset_dir)

    learn_rows = _scan_learnability(args.learnability_root.resolve())
    guided_rows = _scan_guided_publication_clean(args.publication_clean_root.resolve())
    ctree_rows = _scan_ctree_budget(args.neural_overnight_root.resolve())
    _validate_markov_sources(learn_rows, guided_rows)
    if not ctree_rows:
        raise ValueError("no C-TreePO budget rows found for topic_phi_docs=4096")

    diag_path = args.publication_clean_root.resolve() / "figures" / "identifiable_zero_publication_report_latest_diagnostics.json"
    if not diag_path.exists():
        raise FileNotFoundError(f"missing publication-clean diagnostics: {diag_path}")
    bridge = _load_bridge_series(diag_path)

    learn_neural = _aggregate_learnability(learn_rows, "neural")
    learn_add = _aggregate_learnability(learn_rows, "additive")
    guided_neural = _aggregate_guided(guided_rows, "neural")
    guided_add = _aggregate_guided(guided_rows, "additive")
    ctree_budget = _aggregate_ctree_budget(ctree_rows)

    # Generate plot assets in the shared figure path.
    train_plot = asset_dir / "markov_train_info_plot.pdf"
    infer_plot = asset_dir / "markov_infer_info_plot.pdf"
    override_plot = asset_dir / "markov_override_sanity_plot.pdf"
    ctree_budget_plot = asset_dir / "ctree_budget_4096_plot.pdf"
    ctree_bridge_plot = asset_dir / "ctree_bridge_norm_plot.pdf"
    _plot_markov_train(learn_neural, learn_add, train_plot)
    _plot_markov_infer(guided_neural, guided_add, infer_plot)
    _plot_markov_override(guided_neural, override_plot)
    _plot_ctree_budget(ctree_budget, ctree_budget_plot)
    _plot_ctree_bridge(bridge, ctree_bridge_plot)

    # Compile the existing shared conceptual slides before the wrapper slides reference them.
    dgp_pdf = asset_dir / "markov_changepoint_dgp_slide.pdf"
    exact_pdf = asset_dir / "markov_changepoint_exact_merge_slide.pdf"
    if args.emit_pdf:
        dgp_pdf = _compile_existing_shared_slide(asset_dir / "markov_changepoint_dgp_slide.tex")
        exact_pdf = _compile_existing_shared_slide(asset_dir / "markov_changepoint_exact_merge_slide.tex")

    # Emit new slide sources in the shared asset directory.
    operator_info_tex = asset_dir / "operator_info_axes_slide.tex"
    markov_neural_tex = asset_dir / "markov_neural_operator_slide.tex"
    train_slide_tex = asset_dir / "markov_train_info_slide.tex"
    infer_slide_tex = asset_dir / "markov_infer_info_slide.tex"
    override_slide_tex = asset_dir / "markov_override_sanity_slide.tex"
    ctree_bridge_tex = asset_dir / "ctree_operator_bridge_slide.tex"
    takeaway_tex = asset_dir / "operator_takeaway_slide.tex"

    operator_info_pdf = _render_slide(operator_info_tex, _build_operator_info_axes_slide(asset_dir), args.emit_pdf)
    markov_neural_pdf = _render_slide(markov_neural_tex, _build_markov_neural_operator_slide(asset_dir), args.emit_pdf)

    train_notes = (
        r"\bfseries Canonical Markov neural slice\\[0.45em]"
        + rf"\texttt{{train\_docs={CANONICAL_MARKOV_SLICE['train_docs']}}}, "
        + rf"\texttt{{state\_dim={CANONICAL_MARKOV_SLICE['state_dim']}}}, "
        + rf"\texttt{{hidden\_dim={CANONICAL_MARKOV_SLICE['hidden_dim']}}}\\"
        + rf"\texttt{{feature\_mode={CANONICAL_MARKOV_SLICE['feature_mode']}}}, "
        + rf"\texttt{{llw={CANONICAL_MARKOV_SLICE['local_law_weight']}}}, "
        + rf"\texttt{{scw={CANONICAL_MARKOV_SLICE['schedule_consistency_weight']}}}\\[0.6em]"
        + r"Neural root and merge error improve as more labels arrive, but schedule spread remains large much longer than the exact/additive reference."
    )
    train_slide_pdf = _render_slide(
        train_slide_tex,
        _build_plot_slide_tex(
            title="Learn-Time Information: What Better Labels Teach",
            subtitle="Hold the operator architecture fixed and vary only how much training-time oracle visibility the model receives.",
            figure_rel=_relative_tex_path(asset_dir, train_plot),
            note_color="guideorange",
            note_body=train_notes,
            footer=None,
            figure_width="6.75in",
        ),
        args.emit_pdf,
    )

    infer_notes = (
        r"\bfseries Fixed learn-time slice: $q_{train}=1$\\[0.45em]"
        + r"Both families see the same decision-time override schedule. The additive control improves smoothly because its merge rule is exact. The neural operator stays much worse until overrides are nearly complete."
    )
    infer_slide_pdf = _render_slide(
        infer_slide_tex,
        _build_plot_slide_tex(
            title="Decision-Time Information: What Overrides Repair",
            subtitle="Now freeze the learned weights and vary only how much help the tree gets while it is running.",
            figure_rel=_relative_tex_path(asset_dir, infer_plot),
            note_color="markovneural",
            note_body=infer_notes,
            footer=None,
            figure_width="7.05in",
        ),
        args.emit_pdf,
    )

    override_notes = (
        r"\bfseries What the override panels are for\\[0.45em]"
        + r"Late neural improvements should not be described as purely internalized learning if the tree is also receiving a large fraction of direct oracle corrections. These panels make that explicit."
    )
    override_slide_pdf = _render_slide(
        override_slide_tex,
        _build_plot_slide_tex(
            title="What Test-Time Help Is Actually Doing",
            subtitle="The neural curve improves late for a reason: more and more internal merge decisions are being directly guided by the oracle.",
            figure_rel=_relative_tex_path(asset_dir, override_plot),
            note_color="ctreegreen",
            note_body=override_notes,
            footer=None,
            figure_width="7.05in",
        ),
        args.emit_pdf,
    )

    ctree_notes = (
        r"\bfseries Bridge to C-TreePO\\[0.45em]"
        + r"Left panel: the existing topic-phi-docs $=4096$ operator budget curve.\\[0.25em]"
        + r"Right panel: the paper-safe normalized read from publication Figure B. Compare \emph{B5/B6}-style normalized progress, not raw \emph{B1/B2} units."
        + r"\\[0.55em]\textcolor{black!70}{Message: with a few labels, C-TreePO can close its own gap quickly, but the comparison has to stay within-family or use normalized progress.}"
    )
    ctree_slide_source = "\n".join(
        [
            _slide_tex_preamble(),
            r"\node[anchor=north west, font=\bfseries\fontsize{26}{28}\selectfont] at (0.45,7.08) {Bridge Slide: C-TreePO as the Approximate Analogue};",
            r"\node[anchor=north west, text=black!65, font=\fontsize{13}{15}\selectfont, text width=12.2in] at (0.47,6.66) {The Markov slides separate exact control, learned operator quality, and test-time help. This slide shows the same logic in the broader C-TreePO family.};",
            rf"\node[anchor=north west] at (0.52,5.95) {{\includegraphics[width=5.9in]{{{_relative_tex_path(asset_dir, ctree_budget_plot)}}}}};",
            rf"\node[anchor=north west] at (6.55,5.95) {{\includegraphics[width=5.95in]{{{_relative_tex_path(asset_dir, ctree_bridge_plot)}}}}};",
            rf"\slidebox{{ctreegreen}}{{12.0in}}{{(0.63,1.75)}}{{{ctree_notes}}}",
            _slide_tex_end(),
        ]
    )
    ctree_bridge_pdf = _render_slide(ctree_bridge_tex, ctree_slide_source, args.emit_pdf)

    takeaway_pdf = _render_slide(takeaway_tex, _build_takeaway_slide(), args.emit_pdf)

    slide_pdfs = [
        operator_info_pdf,
        exact_pdf,
        markov_neural_pdf,
        train_slide_pdf,
        infer_slide_pdf,
        override_slide_pdf,
        ctree_bridge_pdf,
        takeaway_pdf,
    ]

    deck_tex = out_dir / "operator_learning_slide_deck.tex"
    deck_pdf = out_dir / "operator_learning_slide_deck.pdf"
    _write_text(deck_tex, _master_deck_tex(slide_pdfs, out_dir))
    if args.emit_pdf:
        if not _run_latexmk(deck_tex):
            raise RuntimeError(f"failed to compile deck: {deck_tex}")

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "canonical_markov_slice": {
            **CANONICAL_MARKOV_SLICE,
            "q_train_for_infer_slide": CANONICAL_MARKOV_Q_TRAIN,
        },
        "ctree_bridge_slice": {
            "topic_phi_docs": CANONICAL_CTREE_DOCS,
            "q_train_for_normalized_bridge": CANONICAL_CTREE_Q_TRAIN,
        },
        "learnability_root": str(args.learnability_root.resolve()),
        "publication_clean_root": str(args.publication_clean_root.resolve()),
        "neural_overnight_root": str(args.neural_overnight_root.resolve()),
        "learnability_aggregates": {
            "neural": learn_neural,
            "additive": learn_add,
        },
        "guided_aggregates": {
            "neural": guided_neural,
            "additive": guided_add,
        },
        "ctree_budget_series": ctree_budget,
        "bridge_series": bridge,
        "slide_pdfs": [str(p) for p in slide_pdfs],
        "deck_tex": str(deck_tex),
        "deck_pdf": str(deck_pdf),
    }
    _write_json(out_dir / "operator_learning_slide_deck_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
