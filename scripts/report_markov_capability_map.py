#!/usr/bin/env python3
"""Capability-first report for Markov OPS local-law simulations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.core.markov_capability import (
    DEFAULT_ROOT_RATIO_LIMIT,
    DEFAULT_SPREAD_GAIN_THRESHOLD,
    DEFAULT_THEOREM_GAIN_THRESHOLD,
    classify_capability,
    markov_theorem_score,
)
from src.ctreepo.sim.util import safe_float as _safe_float_scalar


@dataclass(frozen=True)
class RawRunRow:
    path: str
    n_regimes: int
    fixed_leaf_tokens: int
    train_docs: int
    val_docs: int
    test_docs: int
    audit_fraction: float
    local_law_weight: float
    schedule_consistency_weight: float
    root_weight: float
    state_dim: int
    hidden_dim: int
    n_epochs: int
    feature_mode: str
    effective_data_seed: int
    effective_model_seed: int
    val_root_mae_n: float
    val_leaf_mae_n: float
    val_merge_mae_n: float
    val_spread_n: float
    val_theorem_score_n: float
    val_selection_metric_value: float
    val_selection_metric_name: str
    test_root_mae_n: float
    test_leaf_mae_n: float
    test_merge_mae_n: float
    test_spread_n: float
    test_theorem_score_n: float
    test_objective_for_report: float
    test_objective_metric_name: str
    exploratory_compat: bool


STATUS_TO_CODE = {
    "failure": 0,
    "theorem_only": 1,
    "root_only": 2,
    "full_success": 3,
}
CODE_TO_STATUS = {value: key for key, value in STATUS_TO_CODE.items()}
STATUS_LABELS = {
    "failure": "FAIL",
    "theorem_only": "THEOREM",
    "root_only": "ROOT",
    "full_success": "FULL",
}
STATUS_COLORS = ["#d9d9d9", "#e07a5f", "#3d5a80", "#2a9d8f"]
FAILURE_REASON_ORDER = [
    "objective_conflict",
    "schedule_instability",
    "insufficient_audit",
    "insufficient_data",
    "insufficient_capacity",
]
CONDITION_COLORS = {
    "baseline": "#6c757d",
    "local_only": "#2a9d8f",
    "sched_only": "#e07a5f",
    "both": "#1d3557",
}
CONDITION_MARKERS = {
    "baseline": "o",
    "local_only": "s",
    "sched_only": "^",
    "both": "D",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capability-first Markov local-law report.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--suite-type",
        choices=["auto", "sanity_suite", "transition_map_suite", "mechanism_suite"],
        default="auto",
    )
    parser.add_argument("--aggregate", choices=["mean", "median"], default="mean")
    parser.add_argument("--expected-run-count", type=int, default=None)
    parser.add_argument("--title", type=str, default="Markov Capability Map")
    parser.add_argument("--status-note", type=str, default="")
    parser.add_argument("--pdf-path", type=str, default=None)
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize count errors by max_segments - 1.",
    )
    parser.add_argument(
        "--compat-exploratory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow older train/test-only artifacts by reusing test metrics as validation metrics.",
    )
    return parser.parse_args()


def _safe_float(mapping: dict, key: str, default: float = float("nan")) -> float:
    return _safe_float_scalar(mapping.get(key), default=default)


def _reduce(xs: Sequence[float], *, agg: str) -> float:
    vals = [float(x) for x in xs if np.isfinite(float(x))]
    if not vals:
        return float("nan")
    if agg == "mean":
        return float(fmean(vals))
    if agg == "median":
        return float(median(vals))
    raise ValueError(f"unsupported aggregate: {agg!r}")


def _normalize_metric(value: float, *, scale: float, normalize: bool) -> float:
    return float(value) / float(scale) if normalize else float(value)


def _extract_split_metrics(
    learned: dict,
    learned_val: dict,
    *,
    normalize: bool,
    scale: float,
    compat_exploratory: bool,
) -> tuple[float, float, float, float, float, bool]:
    val_root = _safe_float(learned, "val_root_mae")
    if not np.isfinite(val_root):
        val_root = _safe_float(learned_val, "root_mae")
    val_leaf = _safe_float(learned, "val_leaf_mae")
    if not np.isfinite(val_leaf):
        val_leaf = _safe_float(learned_val, "leaf_mae")
    val_merge = _safe_float(learned, "val_merge_mae")
    if not np.isfinite(val_merge):
        val_merge = _safe_float(learned_val, "merge_mae")
    val_spread = _safe_float(learned, "val_schedule_spread_mean")
    if not np.isfinite(val_spread):
        val_spread = _safe_float(learned_val, "schedule_spread_mean")

    exploratory_compat = False
    if not all(np.isfinite(x) for x in (val_root, val_leaf, val_merge, val_spread)):
        if not compat_exploratory:
            raise ValueError("missing validation metrics; rerun with val_docs>0 or pass --compat-exploratory")
        exploratory_compat = True
        val_root = _safe_float(learned, "test_root_mae", _safe_float(learned, "root_mae"))
        val_leaf = _safe_float(learned, "test_leaf_mae", _safe_float(learned, "leaf_mae"))
        val_merge = _safe_float(learned, "test_merge_mae", _safe_float(learned, "merge_mae"))
        val_spread = _safe_float(
            learned,
            "test_schedule_spread_mean",
            _safe_float(learned, "schedule_spread_mean"),
        )

    val_root_n = _normalize_metric(val_root, scale=scale, normalize=normalize)
    val_leaf_n = _normalize_metric(val_leaf, scale=scale, normalize=normalize)
    val_merge_n = _normalize_metric(val_merge, scale=scale, normalize=normalize)
    val_spread_n = _normalize_metric(val_spread, scale=scale, normalize=normalize)
    val_theorem_n = float(
        markov_theorem_score(
            leaf=val_leaf_n,
            merge=val_merge_n,
            spread=val_spread_n,
        )
    )
    return val_root_n, val_leaf_n, val_merge_n, val_spread_n, val_theorem_n, exploratory_compat


def _selection_metric_with_fallback(
    learned: dict,
    *,
    split: str,
    primary_keys: Sequence[str],
    secondary_keys: Sequence[str],
    theorem_fallback: float,
) -> tuple[float, str]:
    selection_metric_name = str(learned.get(f"{split}_objective_selection_metric_name", "") or "")
    if selection_metric_name:
        direct_key = f"{split}_{selection_metric_name}"
        direct_value = _safe_float(learned, direct_key)
        if np.isfinite(direct_value):
            return float(direct_value), str(selection_metric_name)
        selected_value = _safe_float(learned, f"{split}_objective_selection_metric_value")
        if np.isfinite(selected_value):
            return float(selected_value), str(selection_metric_name)
    for key in list(primary_keys) + list(secondary_keys):
        value = _safe_float(learned, key)
        if np.isfinite(value):
            return float(value), str(key)
    return float(theorem_fallback), "theorem_score_fallback"


def _load_runs(
    files: Sequence[Path],
    *,
    normalize: bool,
    compat_exploratory: bool,
) -> List[RawRunRow]:
    rows: List[RawRunRow] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {}) or {}
        metrics = payload.get("metrics", {}) or {}
        learned = metrics.get("learned", {}) or {}
        learned_val = metrics.get("learned_val", {}) or {}

        train_docs = int(cfg.get("train_docs", -1))
        val_docs = int(cfg.get("val_docs", 0))
        test_docs = int(cfg.get("test_docs", -1))
        audit_fraction = float(cfg.get("audit_fraction", float("nan")))
        max_segments = int(cfg.get("max_segments", -1))
        scale = float(max(1, max_segments - 1)) if max_segments > 0 else float("nan")
        if train_docs <= 0 or test_docs < 0 or not np.isfinite(scale):
            continue

        val_root_n, val_leaf_n, val_merge_n, val_spread_n, val_theorem_n, exploratory = _extract_split_metrics(
            learned,
            learned_val,
            normalize=normalize,
            scale=scale,
            compat_exploratory=compat_exploratory,
        )

        test_root = _safe_float(learned, "test_root_mae", _safe_float(learned, "root_mae"))
        test_leaf = _safe_float(learned, "test_leaf_mae", _safe_float(learned, "leaf_mae"))
        test_merge = _safe_float(learned, "test_merge_mae", _safe_float(learned, "merge_mae"))
        test_spread = _safe_float(
            learned,
            "test_schedule_spread_mean",
            _safe_float(learned, "schedule_spread_mean"),
        )
        test_root_n = _normalize_metric(test_root, scale=scale, normalize=normalize)
        test_leaf_n = _normalize_metric(test_leaf, scale=scale, normalize=normalize)
        test_merge_n = _normalize_metric(test_merge, scale=scale, normalize=normalize)
        test_spread_n = _normalize_metric(test_spread, scale=scale, normalize=normalize)
        test_theorem_n = float(
            markov_theorem_score(
                leaf=test_leaf_n,
                merge=test_merge_n,
                spread=test_spread_n,
            )
        )
        val_selection_metric_value, val_selection_metric_name = _selection_metric_with_fallback(
            learned,
            split="val",
            primary_keys=(
                "val_optimization_objective_full_labels",
                "val_objective_full_labels",
            ),
            secondary_keys=(
                "test_optimization_objective_full_labels",
                "test_objective_full_labels",
            ),
            theorem_fallback=val_theorem_n,
        )
        test_objective_for_report, test_objective_metric_name = _selection_metric_with_fallback(
            learned,
            split="test",
            primary_keys=(
                "test_optimization_objective_full_labels",
                "test_objective_full_labels",
            ),
            secondary_keys=(),
            theorem_fallback=test_theorem_n,
        )

        rows.append(
            RawRunRow(
                path=str(path),
                n_regimes=int(cfg.get("n_regimes", 0)),
                fixed_leaf_tokens=int(cfg.get("fixed_leaf_tokens", 0)),
                train_docs=int(train_docs),
                val_docs=int(val_docs),
                test_docs=int(test_docs),
                audit_fraction=float(audit_fraction),
                local_law_weight=float((payload.get("objective", {}) or {}).get("local_law_weight", cfg.get("local_law_weight", 0.0))),
                schedule_consistency_weight=float(cfg.get("schedule_consistency_weight", 0.0)),
                root_weight=float(cfg.get("root_weight", 1.0)),
                state_dim=int(cfg.get("state_dim", 0)),
                hidden_dim=int(cfg.get("hidden_dim", 0)),
                n_epochs=int(cfg.get("n_epochs", 0)),
                feature_mode=str(cfg.get("feature_mode", "")),
                effective_data_seed=int(cfg.get("effective_data_seed", cfg.get("seed", 0))),
                effective_model_seed=int(cfg.get("effective_model_seed", cfg.get("seed", 0))),
                val_root_mae_n=float(val_root_n),
                val_leaf_mae_n=float(val_leaf_n),
                val_merge_mae_n=float(val_merge_n),
                val_spread_n=float(val_spread_n),
                val_theorem_score_n=float(val_theorem_n),
                val_selection_metric_value=float(val_selection_metric_value),
                val_selection_metric_name=str(val_selection_metric_name),
                test_root_mae_n=float(test_root_n),
                test_leaf_mae_n=float(test_leaf_n),
                test_merge_mae_n=float(test_merge_n),
                test_spread_n=float(test_spread_n),
                test_theorem_score_n=float(test_theorem_n),
                test_objective_for_report=float(test_objective_for_report),
                test_objective_metric_name=str(test_objective_metric_name),
                exploratory_compat=bool(exploratory),
            )
        )
    return rows


def _selection_key(row: RawRunRow) -> Tuple[object, ...]:
    return (
        int(row.n_regimes),
        int(row.fixed_leaf_tokens),
        int(row.train_docs),
        int(row.val_docs),
        int(row.test_docs),
        float(row.audit_fraction),
        float(row.root_weight),
        int(row.state_dim),
        int(row.hidden_dim),
        int(row.n_epochs),
        str(row.feature_mode),
        int(row.effective_data_seed),
        int(row.effective_model_seed),
    )


def _aggregate_key(selected_row: dict, *, include_condition: bool) -> Tuple[object, ...]:
    key = (
        int(selected_row["n_regimes"]),
        int(selected_row["fixed_leaf_tokens"]),
        int(selected_row["train_docs"]),
        int(selected_row["val_docs"]),
        int(selected_row["test_docs"]),
        float(selected_row["audit_fraction"]),
        float(selected_row["root_weight"]),
        int(selected_row["state_dim"]),
        int(selected_row["hidden_dim"]),
        int(selected_row["n_epochs"]),
        str(selected_row["feature_mode"]),
    )
    if include_condition:
        key = (*key, str(selected_row["condition"]))
    return key


def _problem_key(aggregated_row: dict) -> Tuple[object, ...]:
    return (
        int(aggregated_row["n_regimes"]),
        int(aggregated_row["fixed_leaf_tokens"]),
        int(aggregated_row["val_docs"]),
        int(aggregated_row["test_docs"]),
        float(aggregated_row["root_weight"]),
    )


def _capacity_key(row: dict) -> Tuple[int, int, int, str]:
    return (
        int(row["state_dim"]),
        int(row["hidden_dim"]),
        int(row["n_epochs"]),
        str(row["feature_mode"]),
    )


def _condition_from_weights(local_law_weight: float, schedule_weight: float) -> str:
    llw_active = float(local_law_weight) > 1e-12
    scw_active = float(schedule_weight) > 1e-12
    if not llw_active and not scw_active:
        return "baseline"
    if llw_active and not scw_active:
        return "local_only"
    if not llw_active and scw_active:
        return "sched_only"
    return "both"


def _selection_sort_key(row: RawRunRow) -> Tuple[float, float, float, float, float, float]:
    return (
        float(row.val_selection_metric_value),
        float(row.val_theorem_score_n),
        float(row.val_root_mae_n),
        float(row.val_spread_n),
        float(row.schedule_consistency_weight),
        float(row.local_law_weight),
    )


def _matched_baseline(group: Sequence[RawRunRow], *, schedule_weight: float) -> Optional[RawRunRow]:
    candidates = [
        row
        for row in group
        if np.isclose(float(row.schedule_consistency_weight), float(schedule_weight))
    ]
    if not candidates:
        return None
    min_llw = min(float(row.local_law_weight) for row in candidates)
    baseline = [row for row in candidates if np.isclose(float(row.local_law_weight), min_llw)]
    return min(baseline, key=_selection_sort_key) if baseline else None


def _selected_row_payload(
    *,
    row: RawRunRow,
    baseline: RawRunRow,
    selection_metric: str,
) -> dict:
    assessment = classify_capability(
        baseline_theorem_score=float(baseline.test_theorem_score_n),
        baseline_spread=float(baseline.test_spread_n),
        baseline_root_mae=float(baseline.test_root_mae_n),
        selected_theorem_score=float(row.test_theorem_score_n),
        selected_spread=float(row.test_spread_n),
        selected_root_mae=float(row.test_root_mae_n),
    )
    condition = _condition_from_weights(
        local_law_weight=float(row.local_law_weight),
        schedule_weight=float(row.schedule_consistency_weight),
    )
    return {
        "path": str(row.path),
        "selection_metric": str(selection_metric),
        "n_regimes": int(row.n_regimes),
        "fixed_leaf_tokens": int(row.fixed_leaf_tokens),
        "train_docs": int(row.train_docs),
        "val_docs": int(row.val_docs),
        "test_docs": int(row.test_docs),
        "audit_fraction": float(row.audit_fraction),
        "root_weight": float(row.root_weight),
        "state_dim": int(row.state_dim),
        "hidden_dim": int(row.hidden_dim),
        "n_epochs": int(row.n_epochs),
        "feature_mode": str(row.feature_mode),
        "effective_data_seed": int(row.effective_data_seed),
        "effective_model_seed": int(row.effective_model_seed),
        "selected_local_law_weight": float(row.local_law_weight),
        "selected_lambda_sched": float(row.schedule_consistency_weight),
        "condition": str(condition),
        "val_selection_metric_name": str(row.val_selection_metric_name),
        "val_selection_metric_value": float(row.val_selection_metric_value),
        "val_theorem_score_n": float(row.val_theorem_score_n),
        "val_leaf_mae_n": float(row.val_leaf_mae_n),
        "val_merge_mae_n": float(row.val_merge_mae_n),
        "val_spread_n": float(row.val_spread_n),
        "val_root_mae_n": float(row.val_root_mae_n),
        "test_objective_for_report": float(row.test_objective_for_report),
        "test_objective_metric_name": str(row.test_objective_metric_name),
        "test_theorem_score_n": float(row.test_theorem_score_n),
        "test_leaf_mae_n": float(row.test_leaf_mae_n),
        "test_merge_mae_n": float(row.test_merge_mae_n),
        "test_spread_n": float(row.test_spread_n),
        "test_root_mae_n": float(row.test_root_mae_n),
        "baseline_local_law_weight": float(baseline.local_law_weight),
        "baseline_lambda_sched": float(baseline.schedule_consistency_weight),
        "baseline_test_objective_for_report": float(baseline.test_objective_for_report),
        "baseline_test_theorem_score_n": float(baseline.test_theorem_score_n),
        "baseline_test_leaf_mae_n": float(baseline.test_leaf_mae_n),
        "baseline_test_merge_mae_n": float(baseline.test_merge_mae_n),
        "baseline_test_spread_n": float(baseline.test_spread_n),
        "baseline_test_root_mae_n": float(baseline.test_root_mae_n),
        "exploratory_compat": bool(row.exploratory_compat or baseline.exploratory_compat),
        **assessment.to_dict(),
    }


def _build_selected_rows(rows: Sequence[RawRunRow], *, selection_mode: str) -> List[dict]:
    groups: Dict[Tuple[object, ...], List[RawRunRow]] = defaultdict(list)
    for row in rows:
        groups[_selection_key(row)].append(row)

    selected_rows: List[dict] = []
    for group in groups.values():
        if selection_mode == "tuned":
            selected = min(group, key=_selection_sort_key)
            baseline = _matched_baseline(group, schedule_weight=float(selected.schedule_consistency_weight))
            if baseline is None:
                continue
            selected_rows.append(
                _selected_row_payload(
                    row=selected,
                    baseline=baseline,
                    selection_metric=str(selected.val_selection_metric_name),
                )
            )
            continue

        baseline = None
        for row in group:
            if (
                abs(float(row.local_law_weight)) <= 1e-12
                and abs(float(row.schedule_consistency_weight)) <= 1e-12
            ):
                baseline = row
                break
        if baseline is None:
            continue
        for row in group:
            selected_rows.append(
                _selected_row_payload(
                    row=row,
                    baseline=baseline,
                    selection_metric="direct_condition",
                )
            )
    return selected_rows


def _dominant_status(status_counts: Counter[str]) -> str:
    if not status_counts:
        return "failure"
    severity = {"failure": 0, "theorem_only": 1, "root_only": 2, "full_success": 3}
    return sorted(status_counts.items(), key=lambda kv: (-kv[1], severity.get(kv[0], 0)))[0][0]


def _aggregate_selected_rows(
    selected_rows: Sequence[dict],
    *,
    agg: str,
    include_condition: bool,
) -> List[dict]:
    groups: Dict[Tuple[object, ...], List[dict]] = defaultdict(list)
    for row in selected_rows:
        groups[_aggregate_key(row, include_condition=include_condition)].append(row)

    aggregated: List[dict] = []
    for key, group in sorted(groups.items()):
        status_counts = Counter(str(row["capability_status"]) for row in group if row["condition"] != "baseline")
        dominant = _dominant_status(status_counts)
        selection_metric_names = sorted({str(row["selection_metric"]) for row in group})
        aggregated.append(
            {
                "n_regimes": int(key[0]),
                "fixed_leaf_tokens": int(key[1]),
                "train_docs": int(key[2]),
                "val_docs": int(key[3]),
                "test_docs": int(key[4]),
                "audit_fraction": float(key[5]),
                "root_weight": float(key[6]),
                "state_dim": int(key[7]),
                "hidden_dim": int(key[8]),
                "n_epochs": int(key[9]),
                "feature_mode": str(key[10]),
                "condition": str(key[11]) if include_condition else "selected",
                "n_runs": int(len(group)),
                "selection_metric": selection_metric_names[0]
                if len(selection_metric_names) == 1
                else "mixed",
                "selected_local_law_weight": float(
                    _reduce([row["selected_local_law_weight"] for row in group], agg=agg)
                ),
                "selected_lambda_sched": float(_reduce([row["selected_lambda_sched"] for row in group], agg=agg)),
                "val_selection_metric_value": float(
                    _reduce([row["val_selection_metric_value"] for row in group], agg=agg)
                ),
                "val_theorem_score_n": float(_reduce([row["val_theorem_score_n"] for row in group], agg=agg)),
                "test_objective_for_report": float(
                    _reduce([row["test_objective_for_report"] for row in group], agg=agg)
                ),
                "test_theorem_score_n": float(_reduce([row["test_theorem_score_n"] for row in group], agg=agg)),
                "test_leaf_mae_n": float(_reduce([row["test_leaf_mae_n"] for row in group], agg=agg)),
                "test_merge_mae_n": float(_reduce([row["test_merge_mae_n"] for row in group], agg=agg)),
                "test_spread_n": float(_reduce([row["test_spread_n"] for row in group], agg=agg)),
                "test_root_mae_n": float(_reduce([row["test_root_mae_n"] for row in group], agg=agg)),
                "baseline_test_objective_for_report": float(
                    _reduce([row["baseline_test_objective_for_report"] for row in group], agg=agg)
                ),
                "baseline_test_theorem_score_n": float(_reduce([row["baseline_test_theorem_score_n"] for row in group], agg=agg)),
                "baseline_test_spread_n": float(_reduce([row["baseline_test_spread_n"] for row in group], agg=agg)),
                "baseline_test_root_mae_n": float(_reduce([row["baseline_test_root_mae_n"] for row in group], agg=agg)),
                "theorem_gain_frac": float(_reduce([row["theorem_gain_frac"] for row in group], agg=agg)),
                "spread_gain_frac": float(_reduce([row["spread_gain_frac"] for row in group], agg=agg)),
                "root_ratio": float(_reduce([row["root_ratio"] for row in group], agg=agg)),
                "theorem_margin": float(_reduce([row["theorem_margin"] for row in group], agg=agg)),
                "spread_margin": float(_reduce([row["spread_margin"] for row in group], agg=agg)),
                "root_margin": float(_reduce([row["root_margin"] for row in group], agg=agg)),
                "full_success_count": int(status_counts.get("full_success", 0)),
                "theorem_only_count": int(status_counts.get("theorem_only", 0)),
                "root_only_count": int(status_counts.get("root_only", 0)),
                "failure_count": int(status_counts.get("failure", 0)),
                "full_success_rate": float(status_counts.get("full_success", 0) / max(1, len(group))),
                "dominant_capability_status": str(dominant),
                "exploratory_compat": bool(any(bool(row["exploratory_compat"]) for row in group)),
            }
        )
    return aggregated


def _capacity_rank(capacity: Tuple[int, int, int, str]) -> Tuple[int, int, int]:
    return (int(capacity[0]), int(capacity[1]), int(capacity[2]))


def _infer_failure_reason(row: dict, all_rows: Sequence[dict]) -> str:
    status = str(row["dominant_capability_status"])
    if status == "full_success":
        return ""
    if status == "theorem_only":
        return "objective_conflict"
    if float(row["spread_gain_frac"]) < DEFAULT_SPREAD_GAIN_THRESHOLD and float(row["theorem_gain_frac"]) >= DEFAULT_THEOREM_GAIN_THRESHOLD:
        return "schedule_instability"

    same_problem = [
        other
        for other in all_rows
        if _problem_key(other) == _problem_key(row)
    ]
    same_capacity = [other for other in same_problem if _capacity_key(other) == _capacity_key(row)]
    same_capacity_success = [
        other
        for other in same_capacity
        if str(other["dominant_capability_status"]) == "full_success"
    ]
    same_audit_capacity_success = [
        other
        for other in same_problem
        if np.isclose(float(other["audit_fraction"]), float(row["audit_fraction"]))
        and _capacity_key(other) == _capacity_key(row)
        and str(other["dominant_capability_status"]) == "full_success"
    ]
    same_train_capacity_success = [
        other
        for other in same_problem
        if int(other["train_docs"]) == int(row["train_docs"])
        and _capacity_key(other) == _capacity_key(row)
        and str(other["dominant_capability_status"]) == "full_success"
    ]
    same_train_audit_success = [
        other
        for other in same_problem
        if int(other["train_docs"]) == int(row["train_docs"])
        and np.isclose(float(other["audit_fraction"]), float(row["audit_fraction"]))
        and str(other["dominant_capability_status"]) == "full_success"
    ]

    if any(float(other["audit_fraction"]) > float(row["audit_fraction"]) for other in same_train_capacity_success):
        return "insufficient_audit"
    if any(int(other["train_docs"]) > int(row["train_docs"]) for other in same_audit_capacity_success):
        return "insufficient_data"
    if any(_capacity_rank(_capacity_key(other)) > _capacity_rank(_capacity_key(row)) for other in same_train_audit_success):
        return "insufficient_capacity"
    if float(row["root_ratio"]) > DEFAULT_ROOT_RATIO_LIMIT:
        return "objective_conflict"
    if float(row["spread_gain_frac"]) < DEFAULT_SPREAD_GAIN_THRESHOLD:
        return "schedule_instability"
    return "insufficient_data"


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _problem_slug(row: dict) -> str:
    return f"reg_{int(row['n_regimes'])}__leaf_{int(row['fixed_leaf_tokens'])}__rw_{float(row['root_weight']):g}".replace(".", "p")


def _capacity_slug(row: dict) -> str:
    return f"sd_{int(row['state_dim'])}__hd_{int(row['hidden_dim'])}__ep_{int(row['n_epochs'])}"


def _problem_title(row: dict) -> str:
    return (
        f"n_regimes={int(row['n_regimes'])}, leaf_tokens={int(row['fixed_leaf_tokens'])}, "
        f"root_weight={float(row['root_weight']):g}"
    )


def _sorted_axes(rows: Sequence[dict]) -> Tuple[List[int], List[float]]:
    train_docs = sorted({int(row["train_docs"]) for row in rows})
    audits = sorted({float(row["audit_fraction"]) for row in rows})
    return train_docs, audits


def _plot_capability_heatmap(rows: Sequence[dict], *, output_path: Path, title: str) -> None:
    train_docs_vals, audit_vals = _sorted_axes(rows)
    matrix = np.full((len(train_docs_vals), len(audit_vals)), np.nan, dtype=np.float64)
    labels: Dict[Tuple[int, int], str] = {}
    for i, td in enumerate(train_docs_vals):
        for j, audit in enumerate(audit_vals):
            match = [
                row
                for row in rows
                if int(row["train_docs"]) == int(td)
                and np.isclose(float(row["audit_fraction"]), float(audit))
            ]
            if not match:
                continue
            row = match[0]
            status = str(row["dominant_capability_status"])
            matrix[i, j] = float(STATUS_TO_CODE.get(status, 0))
            labels[(i, j)] = f"{STATUS_LABELS.get(status, status)}\n{float(row['full_success_rate']):.2f}"

    fig, ax = plt.subplots(figsize=(1.8 + 1.55 * len(audit_vals), 1.8 + 1.1 * len(train_docs_vals)))
    cmap = ListedColormap(STATUS_COLORS)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=float(len(STATUS_COLORS) - 1))
    _ = im
    ax.set_xticks(range(len(audit_vals)))
    ax.set_yticks(range(len(train_docs_vals)))
    ax.set_xticklabels([f"{100.0 * float(v):.1f}%" if float(v) < 0.1 else f"{100.0 * float(v):.0f}%" for v in audit_vals])
    ax.set_yticklabels([str(int(v)) for v in train_docs_vals])
    ax.set_xlabel("q_audit")
    ax.set_ylabel("train_docs")
    ax.set_title(title)
    for (i, j), label in labels.items():
        ax.text(j, i, label, ha="center", va="center", fontsize=9, color="#111111")
    handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=10, markerfacecolor=color, markeredgecolor=color, label=STATUS_LABELS[status])
        for status, color in zip(["failure", "theorem_only", "root_only", "full_success"], STATUS_COLORS)
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_numeric_heatmap(
    rows: Sequence[dict],
    *,
    output_path: Path,
    title: str,
    value_key: str,
    cmap: str,
    fmt: str,
) -> None:
    train_docs_vals, audit_vals = _sorted_axes(rows)
    matrix = np.full((len(train_docs_vals), len(audit_vals)), np.nan, dtype=np.float64)
    for i, td in enumerate(train_docs_vals):
        for j, audit in enumerate(audit_vals):
            match = [
                row
                for row in rows
                if int(row["train_docs"]) == int(td)
                and np.isclose(float(row["audit_fraction"]), float(audit))
            ]
            if not match:
                continue
            matrix[i, j] = float(match[0][value_key])

    fig, ax = plt.subplots(figsize=(1.8 + 1.55 * len(audit_vals), 1.8 + 1.1 * len(train_docs_vals)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(audit_vals)))
    ax.set_yticks(range(len(train_docs_vals)))
    ax.set_xticklabels([f"{100.0 * float(v):.1f}%" if float(v) < 0.1 else f"{100.0 * float(v):.0f}%" for v in audit_vals])
    ax.set_yticklabels([str(int(v)) for v in train_docs_vals])
    ax.set_xlabel("q_audit")
    ax.set_ylabel("train_docs")
    ax.set_title(title)
    for i in range(len(train_docs_vals)):
        for j in range(len(audit_vals)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, format(float(matrix[i, j]), fmt), ha="center", va="center", fontsize=9, color="#111111")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_mechanism_pareto(rows: Sequence[dict], *, output_path: Path, title: str) -> None:
    root_weights = sorted({float(row["root_weight"]) for row in rows})
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    for condition in ("baseline", "local_only", "sched_only", "both"):
        subset = [row for row in rows if str(row["condition"]) == condition]
        if not subset:
            continue
        xs = [float(row["test_root_mae_n"]) for row in subset]
        ys = [float(row["test_theorem_score_n"]) for row in subset]
        sizes = []
        for row in subset:
            rw = float(row["root_weight"])
            idx = root_weights.index(rw) if rw in root_weights else 0
            sizes.append(60 + 25 * idx)
        ax.scatter(
            xs,
            ys,
            s=sizes,
            alpha=0.85,
            color=CONDITION_COLORS[condition],
            marker=CONDITION_MARKERS[condition],
            label=condition,
        )
        for row in subset:
            ax.text(
                float(row["test_root_mae_n"]),
                float(row["test_theorem_score_n"]),
                f"rw={float(row['root_weight']):g}",
                fontsize=8,
                color="#333333",
            )
    ax.set_xlabel("test root MAE (normalized)")
    ax.set_ylabel("test theorem score (normalized)")
    ax.set_title(title)
    ax.grid(True, linewidth=0.8, alpha=0.3)
    ax.legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_text_page(pdf: PdfPages, *, title: str, lines: Sequence[str]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.90])
    ax.axis("off")
    ax.text(0.0, 1.0, title, fontsize=16, fontweight="bold", va="top")
    y = 0.95
    for raw in lines:
        chunks = textwrap.wrap(str(raw), width=105, break_long_words=False, break_on_hyphens=False) or [""]
        for chunk in chunks:
            ax.text(0.0, y, chunk, fontsize=10.0, va="top")
            y -= 0.024
            if y < 0.05:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(8.5, 11))
                ax = fig.add_axes([0.06, 0.05, 0.88, 0.90])
                ax.axis("off")
                y = 0.97
    pdf.savefig(fig)
    plt.close(fig)


def _write_image_page(pdf: PdfPages, *, image_path: Path, title: str) -> None:
    if not image_path.exists():
        return
    image = plt.imread(str(image_path))
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.90])
    ax.axis("off")
    ax.imshow(image)
    fig.suptitle(title, fontsize=14, y=0.98)
    pdf.savefig(fig)
    plt.close(fig)


def main() -> int:
    try:
        from scripts._markov_report_archive import archived_report_exit
    except ModuleNotFoundError:
        from _markov_report_archive import archived_report_exit

    return archived_report_exit(
        legacy_script="scripts/report_markov_capability_map.py",
        replacements=(
            "scripts/report_markov_optimization_tradeoffs.py",
            "scripts/run_markov_optimization_tradeoff_pipeline.py",
        ),
        note=(
            "Capability-map reporting came from the pre-v3 Markov OPS-count stack and is now "
            "archived rather than carried forward into the supported report surface."
        ),
    )

    args = _parse_args()
    input_root = Path(args.input_root)
    if not input_root.exists():
        raise SystemExit(f"input_root not found: {input_root}")
    output_dir = Path(args.output_dir) if args.output_dir else (input_root / "capability_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_root.rglob("seed_*.json"))
    if not files:
        raise SystemExit(f"no seed_*.json files found under {input_root}")

    raw_rows = _load_runs(
        files,
        normalize=bool(args.normalize),
        compat_exploratory=bool(args.compat_exploratory),
    )
    if not raw_rows:
        raise SystemExit("no valid Markov capability runs loaded")

    suite_type = str(args.suite_type)
    if suite_type == "auto":
        suite_type = "mechanism_suite" if any(float(row.root_weight) != 1.0 for row in raw_rows) else "transition_map_suite"
    selection_mode = "direct" if suite_type == "mechanism_suite" else "tuned"
    selected_rows = _build_selected_rows(raw_rows, selection_mode=selection_mode)
    aggregated_rows = _aggregate_selected_rows(
        selected_rows,
        agg=str(args.aggregate),
        include_condition=(selection_mode == "direct"),
    )
    for row in aggregated_rows:
        row["failure_reason"] = _infer_failure_reason(row, aggregated_rows)

    exploratory_only = bool(any(row.exploratory_compat for row in raw_rows))
    expected_run_count = int(args.expected_run_count) if args.expected_run_count is not None else None
    completion_fraction = (
        float(len(raw_rows)) / float(expected_run_count)
        if expected_run_count and expected_run_count > 0
        else None
    )

    figure_paths: List[str] = []
    figure_titles: Dict[str, str] = {}
    failure_rows: List[dict] = [row for row in aggregated_rows if str(row["failure_reason"])]

    if selection_mode == "tuned":
        contexts: Dict[Tuple[Tuple[object, ...], Tuple[int, int, int, str]], List[dict]] = defaultdict(list)
        for row in aggregated_rows:
            contexts[(_problem_key(row), _capacity_key(row))].append(row)
        for rows in contexts.values():
            exemplar = rows[0]
            slug = f"{_problem_slug(exemplar)}__{_capacity_slug(exemplar)}"
            cap_title = f"{_problem_title(exemplar)} | state_dim={int(exemplar['state_dim'])}, hidden_dim={int(exemplar['hidden_dim'])}"

            cap_path = output_dir / f"capability_heatmap_{slug}.png"
            _plot_capability_heatmap(rows, output_path=cap_path, title=f"Capability status | {cap_title}")
            figure_paths.append(str(cap_path))
            figure_titles[str(cap_path)] = f"Capability status | {cap_title}"

            gain_path = output_dir / f"theorem_gain_heatmap_{slug}.png"
            _plot_numeric_heatmap(
                rows,
                output_path=gain_path,
                title=f"Theorem gain over matched baseline | {cap_title}",
                value_key="theorem_gain_frac",
                cmap="YlGnBu",
                fmt=".2f",
            )
            figure_paths.append(str(gain_path))
            figure_titles[str(gain_path)] = f"Theorem gain over matched baseline | {cap_title}"

            root_path = output_dir / f"root_ratio_heatmap_{slug}.png"
            _plot_numeric_heatmap(
                rows,
                output_path=root_path,
                title=f"Root ratio vs matched baseline | {cap_title}",
                value_key="root_ratio",
                cmap="magma_r",
                fmt=".2f",
            )
            figure_paths.append(str(root_path))
            figure_titles[str(root_path)] = f"Root ratio vs matched baseline | {cap_title}"
    else:
        contexts: Dict[Tuple[Tuple[object, ...], Tuple[int, int, int, str]], List[dict]] = defaultdict(list)
        for row in selected_rows:
            key = (
                (
                    int(row["n_regimes"]),
                    int(row["fixed_leaf_tokens"]),
                    int(row["train_docs"]),
                    int(row["audit_fraction"] * 10_000),
                ),
                (
                    int(row["state_dim"]),
                    int(row["hidden_dim"]),
                    int(row["n_epochs"]),
                    str(row["feature_mode"]),
                ),
            )
            contexts[key].append(row)
        for rows in contexts.values():
            exemplar = rows[0]
            slug = f"mechanism_reg_{int(exemplar['n_regimes'])}__leaf_{int(exemplar['fixed_leaf_tokens'])}__train_{int(exemplar['train_docs'])}__audit_{int(round(float(exemplar['audit_fraction']) * 1000.0))}__{_capacity_slug(exemplar)}"
            title = (
                f"Mechanism Pareto | n_regimes={int(exemplar['n_regimes'])}, "
                f"leaf_tokens={int(exemplar['fixed_leaf_tokens'])}, "
                f"train_docs={int(exemplar['train_docs'])}, "
                f"q_audit={100.0 * float(exemplar['audit_fraction']):.1f}%"
            )
            pareto_path = output_dir / f"{slug}.png"
            _plot_mechanism_pareto(rows, output_path=pareto_path, title=title)
            figure_paths.append(str(pareto_path))
            figure_titles[str(pareto_path)] = title

    _write_csv(output_dir / "markov_capability_selected_rows.csv", selected_rows)
    _write_csv(output_dir / "markov_capability_aggregated_rows.csv", aggregated_rows)
    _write_csv(output_dir / "markov_capability_failure_modes.csv", failure_rows)
    (output_dir / "markov_capability_selected_rows.json").write_text(
        json.dumps(selected_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "input_root": str(input_root),
        "suite_type": str(suite_type),
        "selection_mode": str(selection_mode),
        "aggregate": str(args.aggregate),
        "normalize": bool(args.normalize),
        "expected_run_count": expected_run_count,
        "completion_fraction": completion_fraction,
        "run_count": int(len(raw_rows)),
        "selected_run_count": int(len(selected_rows)),
        "aggregated_row_count": int(len(aggregated_rows)),
        "superseded_for_paper_claims": True,
        "exploratory_only": bool(exploratory_only),
        "compat_exploratory_enabled": bool(args.compat_exploratory),
        "thresholds": {
            "theorem_gain_threshold": float(DEFAULT_THEOREM_GAIN_THRESHOLD),
            "spread_gain_threshold": float(DEFAULT_SPREAD_GAIN_THRESHOLD),
            "root_ratio_limit": float(DEFAULT_ROOT_RATIO_LIMIT),
        },
        "status_note": str(args.status_note),
        "failure_mode_counts": dict(Counter(str(row["failure_reason"]) for row in failure_rows if str(row["failure_reason"]))),
        "raw_rows": [row.__dict__ for row in raw_rows],
        "selected_rows": selected_rows,
        "aggregated_rows": aggregated_rows,
        "figures": figure_paths,
        "figure_titles": figure_titles,
    }
    summary_path = output_dir / "markov_capability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    md_lines = [
        f"# {args.title}",
        "",
        "- This report is exploratory / superseded for paper claims. Use `report_markov_law_stress.py` for the journal-facing law-stress story.",
        f"- Input root: `{input_root}`",
        f"- Suite type: `{suite_type}`",
        f"- Selection mode: `{selection_mode}`",
        f"- Raw runs loaded: `{len(raw_rows)}`",
        f"- Selected rows: `{len(selected_rows)}`",
        f"- Aggregated rows: `{len(aggregated_rows)}`",
        f"- Exploratory compatibility mode used: `{exploratory_only}`",
        f"- Theorem score: `leaf_mae + merge_mae + 0.25 * merge_order_sensitivity`",
        (
            "- Journal claim rule: theorem gain >= 10%, sensitivity gain >= 10%, root ratio <= 1.05 "
            "relative to the matched `local_law_weight=0` baseline."
        ),
        "",
        "## Notes",
        "",
    ]
    if exploratory_only:
        md_lines.append(
            "- This report is exploratory because some runs did not contain validation metrics and were loaded with `--compat-exploratory`."
        )
    else:
        md_lines.append("- This report is validation-selected and test-reported.")
    md_lines.append(
        "- Tuned selection uses the configured validation objective when present, with theorem score only as a fallback."
    )
    if str(args.status_note).strip():
        md_lines.append(f"- Status note: `{args.status_note}`")
    md_lines.extend(
        [
            "",
            "## Failure Modes",
            "",
        ]
    )
    if failure_rows:
        for row in failure_rows:
            md_lines.append(
                f"- `{_problem_title(row)} | train_docs={row['train_docs']} | q_audit={100.0 * float(row['audit_fraction']):.1f}% | "
                f"state_dim={row['state_dim']} | hidden_dim={row['hidden_dim']} | "
                f"status={row['dominant_capability_status']} | reason={row['failure_reason']}`"
            )
    else:
        md_lines.append("- No failure rows detected in the aggregated output.")
    md_lines.extend(["", "## Figures", ""])
    for fig_path in figure_paths:
        md_lines.append(f"- {figure_titles.get(fig_path, Path(fig_path).name)}: `{fig_path}`")
    pdf_path = Path(args.pdf_path) if args.pdf_path else (output_dir / "markov_capability_report.pdf")
    md_lines.append(f"- PDF: `{pdf_path}`")
    (output_dir / "markov_capability.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    with PdfPages(pdf_path) as pdf:
        status_lines = [
            f"Generated at UTC: {summary['generated_at']}",
            f"Input root: {input_root}",
            f"Suite type: {suite_type}",
            f"Selection mode: {selection_mode}",
            f"Raw runs loaded: {len(raw_rows)}",
            f"Selected rows: {len(selected_rows)}",
            f"Aggregated rows: {len(aggregated_rows)}",
            f"Exploratory compatibility mode: {exploratory_only}",
            "Tuned selection metric: configured validation objective when present; theorem score is fallback-only.",
            "Journal rule: theorem gain >= 10%, sensitivity gain >= 10%, root ratio <= 1.05 vs matched baseline.",
            "Main theorem score = leaf MAE + merge MAE + 0.25 * merge-order sensitivity.",
        ]
        if str(args.status_note).strip():
            status_lines.append(f"status_note: {args.status_note}")
        _write_text_page(pdf, title=str(args.title), lines=status_lines)
        failure_lines = [
            f"{row['failure_reason']} | train_docs={row['train_docs']} | q_audit={100.0 * float(row['audit_fraction']):.1f}% | "
            f"state_dim={row['state_dim']} | hidden_dim={row['hidden_dim']} | status={row['dominant_capability_status']}"
            for row in failure_rows
        ] or ["No failure rows detected."]
        _write_text_page(pdf, title=f"{args.title} | Failure Modes", lines=failure_lines)
        for fig_path in figure_paths:
            _write_image_page(pdf, image_path=Path(fig_path), title=figure_titles.get(fig_path, Path(fig_path).name))

    summary["pdf"] = str(pdf_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "pdf": str(pdf_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
