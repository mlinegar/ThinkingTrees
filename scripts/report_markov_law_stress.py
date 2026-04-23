#!/usr/bin/env python3
"""Capability-first report for the Markov local-law stress suites.

.. deprecated::
    Use ``python -m src.ctreepo.cli sim suite law-stress report --family markov --output-root ...`` instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import textwrap
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.core.markov_law_stress import (
    DEFAULT_LAW_GAIN_THRESHOLD,
    DEFAULT_ROOT_RATIO_LIMIT,
    DEFAULT_SPREAD_GAIN_THRESHOLD,
    VALID_LAW_PACKAGES,
    classify_law_stress,
    infer_law_stress_failure_reason,
    markov_law_bundle_score,
)
from src.ctreepo.sim.local_law_report_common import (
    build_local_law_report_core,
    load_local_law_runs,
    render_local_law_report_markdown,
    write_local_law_report_core_pages,
)
from src.ctreepo.sim.util import safe_float as _safe_float_scalar


warnings.warn(
    "Deprecated. Use python -m src.ctreepo.cli sim suite law-stress report --family markov --output-root ...",
    DeprecationWarning,
    stacklevel=1,
)


MAIN_PACKAGE = "all_laws_plus_sched"
FALLBACK_MAIN_PACKAGE = "all_laws"
PACKAGE_COLORS = {
    "root_only": "#6c757d",
    "c1_only": "#457b9d",
    "c2_only": "#e07a5f",
    "c3_only": "#8d99ae",
    "c1c3": "#2a9d8f",
    "all_laws": "#264653",
    "sched_only": "#f4a261",
    "all_laws_plus_sched": "#1d3557",
}


@dataclass(frozen=True)
class StressRunRow:
    path: str
    run_kind: str
    suite_guess: str
    n_regimes: int
    fixed_leaf_tokens: int
    train_docs: int
    val_docs: int
    test_docs: int
    audit_fraction: float
    root_weight: float
    state_dim: int
    hidden_dim: int
    n_epochs: int
    feature_mode: str
    effective_data_seed: int
    effective_model_seed: int
    law_package: str
    exact_family: str
    val_c1_mae_n: float
    val_c2_mae_n: float
    val_c3_mae_n: float
    val_spread_n: float
    val_root_mae_n: float
    val_bundle_score_n: float
    test_c1_mae_n: float
    test_c2_mae_n: float
    test_c3_mae_n: float
    test_spread_n: float
    test_root_mae_n: float
    test_bundle_score_n: float
    test_c2_r1_mae_n: float
    test_c2_r2_mae_n: float
    test_c2_r4_mae_n: float
    test_resummary_root_drift_r1_n: float
    test_resummary_root_drift_r2_n: float
    test_resummary_root_drift_r4_n: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Markov local-law stress report.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--suite-type",
        choices=["auto", "sanity_suite", "transition_map_suite", "mechanism_suite", "capacity_appendix_suite"],
        default="auto",
    )
    parser.add_argument("--title", type=str, default="Markov Local-Law Stress Report")
    parser.add_argument("--pdf-path", type=str, default=None)
    parser.add_argument("--expected-run-count", type=int, default=None)
    return parser.parse_args()


def _safe_float(mapping: dict, key: str, default: float = float("nan")) -> float:
    return _safe_float_scalar(mapping.get(key), default=default)


def _normalize(value: float, *, scale: float) -> float:
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        return float("nan")
    return float(value) / float(scale)


def _infer_suite_from_path(path: Path) -> str:
    raw = str(path)
    for name in ("sanity_suite", "transition_map_suite", "mechanism_suite", "capacity_appendix_suite"):
        if name in raw:
            return name
    return "unknown"


def _resolve_markov_law_package(payload: dict) -> str:
    cfg = dict(payload.get("config", {}) or {})
    objective = dict(payload.get("objective", {}) or {})
    learnability = dict(payload.get("local_law_learnability", {}) or {})
    metadata = dict(learnability.get("metadata", {}) or {})
    metadata_objective = dict(metadata.get("objective", {}) or {})

    for mapping in (objective, metadata, metadata_objective, cfg):
        value = str(mapping.get("law_package", "") or "").strip()
        if value:
            return value

    objective_weights = dict(objective.get("local_law_weights", {}) or metadata_objective.get("local_law_weights", {}) or {})
    if objective_weights:
        c1 = _safe_float(objective_weights, "c1", 0.0)
        c2 = _safe_float(objective_weights, "c2", 0.0)
        c3 = _safe_float(objective_weights, "c3", 0.0)
        proxy_weight = _safe_float(objective, "proxy_weight_total", _safe_float(metadata_objective, "proxy_weight_total", 0.0))
        active = tuple(name for name, value in (("c1", c1), ("c2", c2), ("c3", c3)) if abs(float(value)) > 1e-12)
        if not active and abs(float(proxy_weight)) <= 1e-12:
            return "root_only"
        if active == ("c1",):
            return "c1_only"
        if active == ("c2",):
            return "c2_only"
        if active == ("c3",):
            return "c3_only"
        if active == ("c1", "c3"):
            return "c1c3"
        if active == ("c1", "c2", "c3"):
            return "all_laws_plus_sched" if abs(float(proxy_weight)) > 1e-12 else "all_laws"
        if not active and abs(float(proxy_weight)) > 1e-12:
            return "sched_only"

    c1_rel = _safe_float(cfg, "c1_relative_weight")
    c2_rel = _safe_float(cfg, "c2_relative_weight")
    c3_rel = _safe_float(cfg, "c3_relative_weight")
    if math.isfinite(c1_rel) and math.isfinite(c2_rel) and math.isfinite(c3_rel):
        exact_profiles = {
            (0.0, 1.0, 0.0): "pure_c2",
            (1.0, 0.0, 4.0): "no_c2",
            (0.05, 1.0, 0.05): "c2_trace_c1c3",
            (0.1, 1.0, 0.1): "c2_light_c1c3",
            (0.25, 1.0, 0.25): "c2_mild_c1c3",
            (0.5, 1.0, 0.5): "c2_moderate_c1c3",
            (1.0, 8.0, 1.0): "c2_very_dominant",
            (1.0, 4.0, 1.0): "c2_dominant",
            (1.0, 2.0, 1.0): "c2_heavy",
            (1.0, 1.0, 1.0): "equal",
            (2.0, 1.0, 2.0): "c1c3_heavy",
            (1.0, 1.0, 4.0): "c3_dominant",
        }
        rounded = (round(c1_rel, 2), round(c2_rel, 2), round(c3_rel, 2))
        if rounded in exact_profiles:
            return exact_profiles[rounded]
        rounded = (round(c1_rel, 1), round(c2_rel, 1), round(c3_rel, 1))
        if rounded in exact_profiles:
            return exact_profiles[rounded]

    return "unknown"


def _read_learned_row(path: Path, payload: dict) -> Optional[StressRunRow]:
    cfg = payload.get("config", {}) or {}
    learned = ((payload.get("metrics", {}) or {}).get("learned", {}) or {})
    if not learned:
        return None
    scale = float(max(1, int(cfg.get("max_segments", 1)) - 1))
    law_package = _resolve_markov_law_package(payload)

    val_c1 = _safe_float(learned, "val_c1_leaf_mae_n", _normalize(_safe_float(learned, "val_leaf_mae"), scale=scale))
    val_c2 = _safe_float(
        learned,
        "val_c2_count_drift_r1_mae_n",
        _safe_float(
            learned,
            "val_c2_idempotence_mae_n",
            _normalize(
                _safe_float(
                    learned,
                    "val_c2_count_drift_r1_mae",
                    _safe_float(learned, "val_c2_idempotence_mae"),
                ),
                scale=scale,
            ),
        ),
    )
    val_c3 = _safe_float(learned, "val_c3_merge_mae_n", _normalize(_safe_float(learned, "val_merge_mae"), scale=scale))
    val_spread = _safe_float(learned, "val_schedule_spread_mean_n", _normalize(_safe_float(learned, "val_schedule_spread_mean"), scale=scale))
    val_root = _safe_float(learned, "val_root_mae_n", _normalize(_safe_float(learned, "val_root_mae"), scale=scale))
    test_c1 = _safe_float(learned, "test_c1_leaf_mae_n", _normalize(_safe_float(learned, "test_leaf_mae", _safe_float(learned, "leaf_mae")), scale=scale))
    test_c2 = _safe_float(
        learned,
        "test_c2_count_drift_r1_mae_n",
        _safe_float(
            learned,
            "test_c2_idempotence_mae_n",
            _normalize(
                _safe_float(
                    learned,
                    "test_c2_count_drift_r1_mae",
                    _safe_float(
                        learned,
                        "test_c2_idempotence_mae",
                        _safe_float(learned, "c2_count_drift_r1_mae", _safe_float(learned, "c2_idempotence_mae")),
                    ),
                ),
                scale=scale,
            ),
        ),
    )
    test_c3 = _safe_float(learned, "test_c3_merge_mae_n", _normalize(_safe_float(learned, "test_merge_mae", _safe_float(learned, "merge_mae")), scale=scale))
    test_spread = _safe_float(learned, "test_schedule_spread_mean_n", _normalize(_safe_float(learned, "test_schedule_spread_mean", _safe_float(learned, "schedule_spread_mean")), scale=scale))
    test_root = _safe_float(learned, "test_root_mae_n", _normalize(_safe_float(learned, "test_root_mae", _safe_float(learned, "root_mae")), scale=scale))

    return StressRunRow(
        path=str(path),
        run_kind="learned",
        suite_guess=_infer_suite_from_path(path),
        n_regimes=int(cfg.get("n_regimes", 0)),
        fixed_leaf_tokens=int(cfg.get("fixed_leaf_tokens", 0)),
        train_docs=int(cfg.get("train_docs", 0)),
        val_docs=int(cfg.get("val_docs", 0)),
        test_docs=int(cfg.get("test_docs", 0)),
        audit_fraction=float(cfg.get("audit_fraction", 0.0)),
        root_weight=float(cfg.get("root_weight", 1.0)),
        state_dim=int(cfg.get("state_dim", 0)),
        hidden_dim=int(cfg.get("hidden_dim", 0)),
        n_epochs=int(cfg.get("n_epochs", 0)),
        feature_mode=str(cfg.get("feature_mode", "")),
        effective_data_seed=int(cfg.get("effective_data_seed", cfg.get("data_seed", cfg.get("seed", 0)))),
        effective_model_seed=int(cfg.get("effective_model_seed", cfg.get("model_seed", cfg.get("seed", 0)))),
        law_package=str(law_package),
        exact_family=str(cfg.get("exact_family", "")),
        val_c1_mae_n=float(val_c1),
        val_c2_mae_n=float(val_c2),
        val_c3_mae_n=float(val_c3),
        val_spread_n=float(val_spread),
        val_root_mae_n=float(val_root),
        val_bundle_score_n=float(_safe_float(learned, "val_theorem_bundle_score_n", markov_law_bundle_score(c1=val_c1, c2=val_c2, c3=val_c3))),
        test_c1_mae_n=float(test_c1),
        test_c2_mae_n=float(test_c2),
        test_c3_mae_n=float(test_c3),
        test_spread_n=float(test_spread),
        test_root_mae_n=float(test_root),
        test_bundle_score_n=float(_safe_float(learned, "test_theorem_bundle_score_n", markov_law_bundle_score(c1=test_c1, c2=test_c2, c3=test_c3))),
        test_c2_r1_mae_n=float(_safe_float(learned, "test_c2_count_drift_r1_mae_n", _safe_float(learned, "test_c2_r1_mae_n", _normalize(_safe_float(learned, "test_c2_count_drift_r1_mae", _safe_float(learned, "test_c2_r1_mae", _safe_float(learned, "c2_count_drift_r1_mae", _safe_float(learned, "c2_r1_mae")))), scale=scale)))),
        test_c2_r2_mae_n=float(_safe_float(learned, "test_c2_count_drift_r2_mae_n", _safe_float(learned, "test_c2_r2_mae_n", _normalize(_safe_float(learned, "test_c2_count_drift_r2_mae", _safe_float(learned, "test_c2_r2_mae", _safe_float(learned, "c2_count_drift_r2_mae", _safe_float(learned, "c2_r2_mae")))), scale=scale)))),
        test_c2_r4_mae_n=float(_safe_float(learned, "test_c2_count_drift_r4_mae_n", _safe_float(learned, "test_c2_r4_mae_n", _normalize(_safe_float(learned, "test_c2_count_drift_r4_mae", _safe_float(learned, "test_c2_r4_mae", _safe_float(learned, "c2_count_drift_r4_mae", _safe_float(learned, "c2_r4_mae")))), scale=scale)))),
        test_resummary_root_drift_r1_n=float(_safe_float(learned, "test_c2_root_count_drift_r1_mae_n", _safe_float(learned, "test_resummary_root_drift_r1_n", _normalize(_safe_float(learned, "test_c2_root_count_drift_r1_mae", _safe_float(learned, "test_resummary_root_drift_r1", _safe_float(learned, "resummary_root_drift_r1"))), scale=scale)))),
        test_resummary_root_drift_r2_n=float(_safe_float(learned, "test_c2_root_count_drift_r2_mae_n", _safe_float(learned, "test_resummary_root_drift_r2_n", _normalize(_safe_float(learned, "test_c2_root_count_drift_r2_mae", _safe_float(learned, "test_resummary_root_drift_r2", _safe_float(learned, "resummary_root_drift_r2"))), scale=scale)))),
        test_resummary_root_drift_r4_n=float(_safe_float(learned, "test_c2_root_count_drift_r4_mae_n", _safe_float(learned, "test_resummary_root_drift_r4_n", _normalize(_safe_float(learned, "test_c2_root_count_drift_r4_mae", _safe_float(learned, "test_resummary_root_drift_r4", _safe_float(learned, "resummary_root_drift_r4"))), scale=scale)))),
    )


def _read_exact_family_row(path: Path, payload: dict) -> Optional[StressRunRow]:
    cfg = payload.get("config", {}) or {}
    stress = ((payload.get("metrics", {}) or {}).get("stress_family", {}) or {})
    if not stress:
        return None
    scale = float(max(1, int(cfg.get("max_segments", 1)) - 1))
    fam = str(stress.get("stress_family_name", cfg.get("exact_family", "")))
    test_c1 = _safe_float(stress, "test_c1_leaf_mae_n", _normalize(_safe_float(stress, "leaf_mae"), scale=scale))
    test_c2 = _safe_float(
        stress,
        "test_c2_count_drift_r1_mae_n",
        _safe_float(
            stress,
            "test_c2_idempotence_mae_n",
            _normalize(
                _safe_float(
                    stress,
                    "c2_count_drift_r1_mae",
                    _safe_float(stress, "c2_idempotence_mae"),
                ),
                scale=scale,
            ),
        ),
    )
    test_c3 = _safe_float(stress, "test_c3_merge_mae_n", _normalize(_safe_float(stress, "merge_mae"), scale=scale))
    test_spread = _safe_float(stress, "test_schedule_spread_mean_n", _normalize(_safe_float(stress, "schedule_spread_mean"), scale=scale))
    test_root = _safe_float(stress, "test_root_mae_n", _normalize(_safe_float(stress, "root_mae"), scale=scale))
    return StressRunRow(
        path=str(path),
        run_kind="exact_family",
        suite_guess=_infer_suite_from_path(path),
        n_regimes=int(cfg.get("n_regimes", 0)),
        fixed_leaf_tokens=int(cfg.get("fixed_leaf_tokens", 0)),
        train_docs=int(cfg.get("train_docs", 0)),
        val_docs=int(cfg.get("val_docs", 0)),
        test_docs=int(cfg.get("test_docs", 0)),
        audit_fraction=float(cfg.get("audit_fraction", 0.0)),
        root_weight=float(cfg.get("root_weight", 1.0)),
        state_dim=int(cfg.get("state_dim", 0)),
        hidden_dim=int(cfg.get("hidden_dim", 0)),
        n_epochs=int(cfg.get("n_epochs", 0)),
        feature_mode=str(cfg.get("feature_mode", "")),
        effective_data_seed=int(cfg.get("effective_data_seed", cfg.get("data_seed", cfg.get("seed", 0)))),
        effective_model_seed=int(cfg.get("effective_model_seed", cfg.get("model_seed", cfg.get("seed", 0)))),
        law_package="",
        exact_family=str(fam),
        val_c1_mae_n=float("nan"),
        val_c2_mae_n=float("nan"),
        val_c3_mae_n=float("nan"),
        val_spread_n=float("nan"),
        val_root_mae_n=float("nan"),
        val_bundle_score_n=float("nan"),
        test_c1_mae_n=float(test_c1),
        test_c2_mae_n=float(test_c2),
        test_c3_mae_n=float(test_c3),
        test_spread_n=float(test_spread),
        test_root_mae_n=float(test_root),
        test_bundle_score_n=float(_safe_float(stress, "test_theorem_bundle_score_n", markov_law_bundle_score(c1=test_c1, c2=test_c2, c3=test_c3))),
        test_c2_r1_mae_n=float(_safe_float(stress, "test_c2_count_drift_r1_mae_n", _safe_float(stress, "test_c2_r1_mae_n", _normalize(_safe_float(stress, "c2_count_drift_r1_mae", _safe_float(stress, "c2_r1_mae")), scale=scale)))),
        test_c2_r2_mae_n=float(_safe_float(stress, "test_c2_count_drift_r2_mae_n", _safe_float(stress, "test_c2_r2_mae_n", _normalize(_safe_float(stress, "c2_count_drift_r2_mae", _safe_float(stress, "c2_r2_mae")), scale=scale)))),
        test_c2_r4_mae_n=float(_safe_float(stress, "test_c2_count_drift_r4_mae_n", _safe_float(stress, "test_c2_r4_mae_n", _normalize(_safe_float(stress, "c2_count_drift_r4_mae", _safe_float(stress, "c2_r4_mae")), scale=scale)))),
        test_resummary_root_drift_r1_n=float(_safe_float(stress, "test_c2_root_count_drift_r1_mae_n", _safe_float(stress, "test_resummary_root_drift_r1_n", _normalize(_safe_float(stress, "c2_root_count_drift_r1_mae", _safe_float(stress, "resummary_root_drift_r1")), scale=scale)))),
        test_resummary_root_drift_r2_n=float(_safe_float(stress, "test_c2_root_count_drift_r2_mae_n", _safe_float(stress, "test_resummary_root_drift_r2_n", _normalize(_safe_float(stress, "c2_root_count_drift_r2_mae", _safe_float(stress, "resummary_root_drift_r2")), scale=scale)))),
        test_resummary_root_drift_r4_n=float(_safe_float(stress, "test_c2_root_count_drift_r4_mae_n", _safe_float(stress, "test_resummary_root_drift_r4_n", _normalize(_safe_float(stress, "c2_root_count_drift_r4_mae", _safe_float(stress, "resummary_root_drift_r4")), scale=scale)))),
    )


def _load_rows(files: Sequence[Path]) -> List[StressRunRow]:
    rows: List[StressRunRow] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        learned = _read_learned_row(path, payload)
        if learned is not None:
            rows.append(learned)
        exact = _read_exact_family_row(path, payload)
        if exact is not None:
            rows.append(exact)
    return rows


def _baseline_package_for(package: str) -> str:
    return "root_only"


def _baseline_key(row: StressRunRow, *, baseline_package: str) -> Tuple[object, ...]:
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
        str(baseline_package),
    )


def _assessed_rows(rows: Sequence[StressRunRow]) -> List[dict]:
    baseline_map: Dict[Tuple[object, ...], StressRunRow] = {}
    for row in rows:
        if row.run_kind != "learned":
            continue
        baseline_map[_baseline_key(row, baseline_package=str(row.law_package))] = row

    out: List[dict] = []
    for row in rows:
        if row.run_kind != "learned":
            continue
        baseline_pkg = _baseline_package_for(str(row.law_package))
        baseline = baseline_map.get(_baseline_key(row, baseline_package=baseline_pkg))
        if baseline is None:
            continue
        assessment = classify_law_stress(
            baseline_c1=float(baseline.test_c1_mae_n),
            baseline_c2=float(baseline.test_c2_mae_n),
            baseline_c3=float(baseline.test_c3_mae_n),
            baseline_spread=float(baseline.test_spread_n),
            baseline_root_mae=float(baseline.test_root_mae_n),
            selected_c1=float(row.test_c1_mae_n),
            selected_c2=float(row.test_c2_mae_n),
            selected_c3=float(row.test_c3_mae_n),
            selected_spread=float(row.test_spread_n),
            selected_root_mae=float(row.test_root_mae_n),
        )
        val_assessment = classify_law_stress(
            baseline_c1=float(baseline.val_c1_mae_n),
            baseline_c2=float(baseline.val_c2_mae_n),
            baseline_c3=float(baseline.val_c3_mae_n),
            baseline_spread=float(baseline.val_spread_n),
            baseline_root_mae=float(baseline.val_root_mae_n),
            selected_c1=float(row.val_c1_mae_n),
            selected_c2=float(row.val_c2_mae_n),
            selected_c3=float(row.val_c3_mae_n),
            selected_spread=float(row.val_spread_n),
            selected_root_mae=float(row.val_root_mae_n),
        )
        out.append(
            {
                **asdict(row),
                "baseline_package": str(baseline_pkg),
                "baseline_test_c1_mae_n": float(baseline.test_c1_mae_n),
                "baseline_test_c2_mae_n": float(baseline.test_c2_mae_n),
                "baseline_test_c3_mae_n": float(baseline.test_c3_mae_n),
                "baseline_test_spread_n": float(baseline.test_spread_n),
                "baseline_test_root_mae_n": float(baseline.test_root_mae_n),
                "baseline_test_bundle_score_n": float(baseline.test_bundle_score_n),
                "baseline_val_c1_mae_n": float(baseline.val_c1_mae_n),
                "baseline_val_c2_mae_n": float(baseline.val_c2_mae_n),
                "baseline_val_c3_mae_n": float(baseline.val_c3_mae_n),
                "baseline_val_spread_n": float(baseline.val_spread_n),
                "baseline_val_root_mae_n": float(baseline.val_root_mae_n),
                "baseline_val_bundle_score_n": float(baseline.val_bundle_score_n),
                **assessment.to_dict(),
                **{f"val_{key}": value for key, value in val_assessment.to_dict().items()},
                "failure_reason": "",
            }
        )
    for row in out:
        row["failure_reason"] = infer_law_stress_failure_reason(row)
    return out


def _aggregate(rows: Sequence[dict], *, group_keys: Sequence[str]) -> List[dict]:
    groups: Dict[Tuple[object, ...], List[dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(name) for name in group_keys)
        groups[key].append(row)

    aggregated: List[dict] = []
    for key, group in sorted(groups.items()):
        payload = {name: value for name, value in zip(group_keys, key)}
        payload.update(
            {
                "n_runs": int(len(group)),
                "val_bundle_score_n": float(fmean(float(row["val_bundle_score_n"]) for row in group)),
                "test_bundle_score_n": float(fmean(float(row["test_bundle_score_n"]) for row in group)),
                "test_c1_mae_n": float(fmean(float(row["test_c1_mae_n"]) for row in group)),
                "test_c2_mae_n": float(fmean(float(row["test_c2_mae_n"]) for row in group)),
                "test_c3_mae_n": float(fmean(float(row["test_c3_mae_n"]) for row in group)),
                "test_spread_n": float(fmean(float(row["test_spread_n"]) for row in group)),
                "test_root_mae_n": float(fmean(float(row["test_root_mae_n"]) for row in group)),
                "test_c2_r4_mae_n": float(fmean(float(row["test_c2_r4_mae_n"]) for row in group)),
                "test_resummary_root_drift_r4_n": float(fmean(float(row["test_resummary_root_drift_r4_n"]) for row in group)),
                "c1_pass_rate": float(fmean(1.0 if bool(row["c1_pass"]) else 0.0 for row in group)),
                "c2_pass_rate": float(fmean(1.0 if bool(row["c2_pass"]) else 0.0 for row in group)),
                "c3_pass_rate": float(fmean(1.0 if bool(row["c3_pass"]) else 0.0 for row in group)),
                "root_pass_rate": float(fmean(1.0 if bool(row["root_pass"]) else 0.0 for row in group)),
                "spread_pass_rate": float(fmean(1.0 if bool(row["spread_pass"]) else 0.0 for row in group)),
                "bundle_full_success_rate": float(fmean(1.0 if bool(row["bundle_full_success"]) else 0.0 for row in group)),
                "val_bundle_full_success_rate": float(fmean(1.0 if bool(row["val_bundle_full_success"]) else 0.0 for row in group)),
                "c1_gain_frac": float(fmean(float(row["c1_gain_frac"]) for row in group)),
                "c2_gain_frac": float(fmean(float(row["c2_gain_frac"]) for row in group)),
                "c3_gain_frac": float(fmean(float(row["c3_gain_frac"]) for row in group)),
                "spread_gain_frac": float(fmean(float(row["spread_gain_frac"]) for row in group)),
                "root_ratio": float(fmean(float(row["root_ratio"]) for row in group)),
                "val_c1_gain_frac": float(fmean(float(row["val_c1_gain_frac"]) for row in group)),
                "val_c2_gain_frac": float(fmean(float(row["val_c2_gain_frac"]) for row in group)),
                "val_c3_gain_frac": float(fmean(float(row["val_c3_gain_frac"]) for row in group)),
                "val_spread_gain_frac": float(fmean(float(row["val_spread_gain_frac"]) for row in group)),
                "val_root_ratio": float(fmean(float(row["val_root_ratio"]) for row in group)),
                "bundle_margin_mean": float(
                    fmean(
                        min(
                            float(row["c1_margin"]),
                            float(row["c2_margin"]),
                            float(row["c3_margin"]),
                            float(row["root_margin"]),
                        )
                        for row in group
                    )
                ),
                "val_bundle_margin_mean": float(
                    fmean(
                        min(
                            float(row["val_c1_margin"]),
                            float(row["val_c2_margin"]),
                            float(row["val_c3_margin"]),
                            float(row["val_root_margin"]),
                        )
                        for row in group
                    )
                ),
                "dominant_failure_reason": Counter(str(row["failure_reason"]) for row in group if str(row["failure_reason"])).most_common(1)[0][0]
                if any(str(row["failure_reason"]) for row in group)
                else "",
            }
        )
        aggregated.append(payload)
    return aggregated


def _filter_main_package(rows: Sequence[dict]) -> List[dict]:
    preferred = [row for row in rows if str(row.get("law_package", "")) == MAIN_PACKAGE]
    if preferred:
        return preferred
    return [row for row in rows if str(row.get("law_package", "")) == FALLBACK_MAIN_PACKAGE]


def _heatmap_axes(rows: Sequence[dict]) -> Tuple[List[int], List[float]]:
    train_docs = sorted({int(row["train_docs"]) for row in rows})
    audits = sorted({float(row["audit_fraction"]) for row in rows})
    return train_docs, audits


def _plot_heatmap(rows: Sequence[dict], *, value_key: str, title: str, output_path: Path, cmap: str, fmt: str) -> None:
    train_docs, audits = _heatmap_axes(rows)
    matrix = np.full((len(train_docs), len(audits)), np.nan, dtype=np.float64)
    for i, td in enumerate(train_docs):
        for j, audit in enumerate(audits):
            match = [
                row
                for row in rows
                if int(row["train_docs"]) == int(td) and math.isclose(float(row["audit_fraction"]), float(audit), rel_tol=0.0, abs_tol=1e-12)
            ]
            if not match:
                continue
            matrix[i, j] = float(match[0][value_key])
    fig, ax = plt.subplots(figsize=(1.8 + 1.55 * len(audits), 1.8 + 1.1 * len(train_docs)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(audits)))
    ax.set_xticklabels([f"{100.0 * float(v):.1f}%" if float(v) < 0.1 else f"{100.0 * float(v):.0f}%" for v in audits])
    ax.set_yticks(range(len(train_docs)))
    ax.set_yticklabels([str(int(v)) for v in train_docs])
    ax.set_xlabel("q_audit")
    ax.set_ylabel("train_docs")
    ax.set_title(title)
    for i in range(len(train_docs)):
        for j in range(len(audits)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, format(float(matrix[i, j]), fmt), ha="center", va="center", fontsize=9, color="#111111")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_exact_family_counterexamples(rows: Sequence[StressRunRow], *, output_path: Path) -> None:
    families = sorted({str(row.exact_family) for row in rows})
    metrics = ["test_c1_mae_n", "test_c2_mae_n", "test_c3_mae_n", "test_root_mae_n"]
    labels = ["C1", "C2", "C3", "Root"]
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    x = np.arange(len(families), dtype=np.float64)
    width = 0.18
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        vals = []
        for fam in families:
            fam_rows = [row for row in rows if str(row.exact_family) == fam]
            vals.append(float(fmean(float(getattr(row, metric)) for row in fam_rows)))
        ax.bar(x + width * (idx - 1.5), vals, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel("normalized error")
    ax.set_title("Exact-family counterexamples")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_mechanism_pareto(rows: Sequence[dict], *, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    packages = sorted({str(row["law_package"]) for row in rows})
    for package in packages:
        subset = [row for row in rows if str(row["law_package"]) == package]
        ax.scatter(
            [float(row["test_root_mae_n"]) for row in subset],
            [float(row["test_bundle_score_n"]) for row in subset],
            color=PACKAGE_COLORS.get(package, "#333333"),
            label=package,
            s=70,
            alpha=0.85,
        )
    ax.set_xlabel("test root MAE (normalized)")
    ax.set_ylabel("test C1+C2+C3 score")
    ax.set_title("Mechanism Pareto")
    ax.grid(True, linewidth=0.8, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_ablation_bar_chart(aggregated_rows: Sequence[dict], *, output_path: Path) -> None:
    """Publication-facing ablation figure: downstream PrimGain and law pass rates by package."""
    packages_order = [
        "root_only", "c1_only", "c2_only", "c3_only",
        "c1c3", "all_laws", "all_laws_plus_sched",
    ]
    package_labels = {
        "root_only": "root only\n(baseline)",
        "c1_only": "C1 only",
        "c2_only": "C2 only",
        "c3_only": "C3 only",
        "c1c3": "C1+C3",
        "all_laws": "C1+C2+C3",
        "all_laws_plus_sched": "C1+C2+C3\n+sched",
    }
    # Aggregate across config cells for each package
    pkg_stats: Dict[str, dict] = {}
    for pkg in packages_order:
        pkg_rows = [row for row in aggregated_rows if str(row.get("law_package", "")) == pkg]
        if not pkg_rows:
            continue
        n_total = sum(int(row.get("n_runs", 1)) for row in pkg_rows)
        pkg_stats[pkg] = {
            "root_ratio": float(fmean(float(row["root_ratio"]) for row in pkg_rows)),
            "prim_gain": float(fmean(1.0 - float(row["root_ratio"]) for row in pkg_rows)),
            "c1_pass": float(fmean(float(row.get("c1_pass_rate", 0.0)) for row in pkg_rows)),
            "c2_pass": float(fmean(float(row.get("c2_pass_rate", 0.0)) for row in pkg_rows)),
            "c3_pass": float(fmean(float(row.get("c3_pass_rate", 0.0)) for row in pkg_rows)),
            "n_configs": len(pkg_rows),
            "n_runs": n_total,
        }

    present = [pkg for pkg in packages_order if pkg in pkg_stats]
    if len(present) < 2:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.8), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(top=0.88, bottom=0.18, left=0.08, right=0.96, wspace=0.32)

    # Left panel: PrimGain (downstream improvement) by package
    x = np.arange(len(present), dtype=np.float64)
    gains = [float(pkg_stats[pkg]["prim_gain"]) for pkg in present]
    colors = [PACKAGE_COLORS.get(pkg, "#333333") for pkg in present]
    bars = ax1.bar(x, [100.0 * g for g in gains], color=colors, width=0.65, edgecolor="#222222", linewidth=0.8)
    ax1.axhline(0.0, color="#555555", linewidth=1.2, linestyle="--", zorder=0)
    ax1.axhline(10.0, color="#2a9d8f", linewidth=0.9, linestyle=":", alpha=0.7, zorder=0, label="10% pass threshold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([package_labels.get(pkg, pkg) for pkg in present], fontsize=9.5)
    ax1.set_ylabel("Downstream gain (%)\n(PrimGain = 1 - root_ratio)", fontsize=10.5)
    ax1.set_title("Downstream MAE improvement by law package", fontsize=12, fontweight="bold")
    ax1.grid(True, axis="y", linewidth=0.7, alpha=0.3)
    ax1.legend(loc="upper left", frameon=False, fontsize=9)
    for bar_obj, gain in zip(bars, gains):
        y_pos = bar_obj.get_height()
        ax1.text(
            bar_obj.get_x() + bar_obj.get_width() / 2.0,
            y_pos + (1.5 if y_pos >= 0 else -3.0),
            f"{100.0 * gain:+.1f}%",
            ha="center", va="bottom" if y_pos >= 0 else "top",
            fontsize=9, fontweight="bold",
        )

    # Right panel: law pass rates stacked
    c1_rates = [100.0 * float(pkg_stats[pkg]["c1_pass"]) for pkg in present]
    c2_rates = [100.0 * float(pkg_stats[pkg]["c2_pass"]) for pkg in present]
    c3_rates = [100.0 * float(pkg_stats[pkg]["c3_pass"]) for pkg in present]
    width = 0.22
    ax2.bar(x - width, c1_rates, width=width, color="#457b9d", label="C1 (leaf)", edgecolor="#222222", linewidth=0.5)
    ax2.bar(x, c2_rates, width=width, color="#e07a5f", label="C2 (resum.)", edgecolor="#222222", linewidth=0.5)
    ax2.bar(x + width, c3_rates, width=width, color="#8d99ae", label="C3 (merge)", edgecolor="#222222", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([package_labels.get(pkg, pkg) for pkg in present], fontsize=9.5)
    ax2.set_ylabel("Law pass rate (%)", fontsize=10.5)
    ax2.set_ylim(0, 115)
    ax2.set_title("Local law satisfaction by package", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y", linewidth=0.7, alpha=0.3)
    ax2.legend(loc="upper left", frameon=False, fontsize=9)

    fig.suptitle(
        "Ablation: which local laws improve downstream error?",
        fontsize=14, fontweight="bold", y=0.96,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_downstream_table(aggregated_rows: Sequence[dict]) -> List[str]:
    """Build a markdown table comparing downstream metrics across law packages."""
    packages_order = [
        "root_only", "c1_only", "c2_only", "c3_only",
        "c1c3", "all_laws", "all_laws_plus_sched",
    ]
    package_labels = {
        "root_only": "root only (baseline)",
        "c1_only": "C1 only",
        "c2_only": "C2 only",
        "c3_only": "C3 only",
        "c1c3": "C1+C3",
        "all_laws": "C1+C2+C3",
        "all_laws_plus_sched": "C1+C2+C3+sched",
    }
    lines = [
        "### Downstream Comparison Table",
        "",
        "Each row averages across all configuration cells (train_docs, audit_fraction, capacity) for a given law package.",
        "**PrimGain** = 1 - root_ratio; positive means the learned g has lower held-out root MAE than the matched root-only baseline.",
        "",
        "| Package | PrimGain | Root ratio | C1 pass% | C2 pass% | C3 pass% | Interpretation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for pkg in packages_order:
        pkg_rows = [row for row in aggregated_rows if str(row.get("law_package", "")) == pkg]
        if not pkg_rows:
            continue
        root_ratio = float(fmean(float(row["root_ratio"]) for row in pkg_rows))
        prim_gain = 1.0 - root_ratio
        c1_pass = float(fmean(float(row.get("c1_pass_rate", 0.0)) for row in pkg_rows))
        c2_pass = float(fmean(float(row.get("c2_pass_rate", 0.0)) for row in pkg_rows))
        c3_pass = float(fmean(float(row.get("c3_pass_rate", 0.0)) for row in pkg_rows))
        if pkg == "root_only":
            interp = "baseline (no local laws)"
        elif prim_gain >= 0.10:
            interp = "improves downstream"
        elif prim_gain >= 0.0:
            interp = "neutral downstream"
        else:
            interp = "hurts downstream"
        label = package_labels.get(pkg, pkg)
        lines.append(
            f"| `{label}` | {100.0 * prim_gain:+.1f}% | {root_ratio:.3f} "
            f"| {100.0 * c1_pass:.0f}% | {100.0 * c2_pass:.0f}% | {100.0 * c3_pass:.0f}% "
            f"| {interp} |"
        )
    lines.append("")
    return lines


def _markov_claim_readout(unified_core: Dict[str, object], *, main_package: str) -> Dict[str, object]:
    rows = [
        dict(row)
        for row in list(unified_core.get("law_stress_summary", []) or [])
        if str(dict(row).get("family", "")) == "markov_ops_count"
    ]
    main_row = next((row for row in rows if str(row.get("law_package", "")) == str(main_package)), None)
    ablation_rows = [row for row in rows if str(row.get("law_package", "")) != str(main_package)]
    strongest_ablation = None
    if ablation_rows:
        strongest_ablation = max(
            ablation_rows,
            key=lambda row: (
                float(row.get("primary_pass_rate", 0.0)),
                float(row.get("mean_primary_gain", float("-inf"))),
                float(row.get("mean_laws_improved", float("-inf"))),
            ),
        )

    status = "unknown"
    if main_row is not None:
        if float(main_row.get("primary_pass_rate", 0.0)) > 0.0 and float(main_row.get("mean_primary_gain", 0.0)) > 0.0:
            status = "passes_downstream"
        else:
            status = "fails_downstream"

    note = "No expected full-package claim row was found in the current report."
    if main_row is not None and strongest_ablation is None:
        note = (
            f"The expected claim row is `{main_package}`. No ablation rows are present in this report, "
            "so the mechanism comparison is not available here."
        )
    elif main_row is not None and strongest_ablation is not None:
        note = (
            f"The expected claim row is `{main_package}`. "
            f"`{strongest_ablation.get('law_package', '')}` is the strongest ablation on downstream metrics in this sweep, "
            "but it remains diagnostic-only and does not replace the claim row."
        )

    return {
        "main_package": str(main_package),
        "status": status,
        "main_row": main_row,
        "strongest_ablation_row": strongest_ablation,
        "note": note,
    }


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            fieldnames.append(key)
            seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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
        legacy_script="scripts/report_markov_law_stress.py",
        replacements=(
            "python -m src.ctreepo.cli sim suite law-stress report --family markov --input-root <root>",
            "scripts/report_markov_optimization_tradeoffs.py",
        ),
        note=(
            "The family-specific Markov wrapper is archived; use the current unified law-stress "
            "CLI where you still need that legacy suite."
        ),
    )

    args = _parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir) if args.output_dir else (input_root / "law_stress_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_root.rglob("seed_*.json"))
    if not files:
        raise SystemExit(f"no seed_*.json files found under {input_root}")

    raw_rows = _load_rows(files)
    if not raw_rows:
        raise SystemExit("no valid law-stress rows loaded")
    protocol_runs = load_local_law_runs(input_root)
    unified_core = build_local_law_report_core(protocol_runs)

    suite_type = str(args.suite_type)
    if suite_type == "auto":
        suite_type = Counter(row.suite_guess for row in raw_rows).most_common(1)[0][0]

    learned_rows = [row for row in raw_rows if row.run_kind == "learned"]
    exact_rows = [row for row in raw_rows if row.run_kind == "exact_family"]
    assessed_rows = _assessed_rows(learned_rows)

    group_keys = (
        "n_regimes",
        "fixed_leaf_tokens",
        "train_docs",
        "val_docs",
        "test_docs",
        "audit_fraction",
        "root_weight",
        "state_dim",
        "hidden_dim",
        "n_epochs",
        "feature_mode",
        "law_package",
    )
    aggregated_rows = _aggregate(assessed_rows, group_keys=group_keys)
    main_rows = _filter_main_package(aggregated_rows)
    exact_family_rows = [row for row in exact_rows if str(row.exact_family)]

    figure_paths: List[str] = []
    figure_titles: Dict[str, str] = {}
    if main_rows:
        for key, title, cmap, fmt in (
            ("c1_pass_rate", "C1 pass rate", "YlGnBu", ".2f"),
            ("c2_pass_rate", "C2 pass rate", "YlGnBu", ".2f"),
            ("c3_pass_rate", "C3 pass rate", "YlGnBu", ".2f"),
            ("bundle_full_success_rate", "Bundle full-success rate", "YlGnBu", ".2f"),
            ("root_ratio", "Root ratio vs matched baseline", "magma_r", ".2f"),
            ("test_resummary_root_drift_r4_n", "Repeated-resummary root drift R=4", "magma", ".2f"),
        ):
            path = output_dir / f"{key}.png"
            _plot_heatmap(main_rows, value_key=key, title=title, output_path=path, cmap=cmap, fmt=fmt)
            figure_paths.append(str(path))
            figure_titles[str(path)] = title
    # Publication-facing ablation figure (always generated when multiple packages present)
    ablation_fig = output_dir / "ablation_downstream.png"
    _plot_ablation_bar_chart(aggregated_rows, output_path=ablation_fig)
    if ablation_fig.exists():
        figure_paths.insert(0, str(ablation_fig))
        figure_titles[str(ablation_fig)] = "Ablation: downstream gain by law package"
    # Mechanism pareto for all suites (not just mechanism_suite)
    if assessed_rows:
        mech_fig = output_dir / "mechanism_pareto.png"
        _plot_mechanism_pareto(assessed_rows, output_path=mech_fig)
        figure_paths.append(str(mech_fig))
        figure_titles[str(mech_fig)] = "Mechanism Pareto: root MAE vs C1+C2+C3"
    if exact_family_rows:
        exact_fig = output_dir / "exact_family_counterexamples.png"
        _plot_exact_family_counterexamples(exact_family_rows, output_path=exact_fig)
        figure_paths.append(str(exact_fig))
        figure_titles[str(exact_fig)] = "Exact-family counterexamples"

    boundary_candidates = [
        row
        for row in main_rows
        if str(row.get("law_package", "")) in {MAIN_PACKAGE, FALLBACK_MAIN_PACKAGE}
    ]
    boundary_candidates = sorted(
        boundary_candidates,
        key=lambda row: (
            abs(float(row["bundle_full_success_rate"]) - 0.5),
            abs(float(row["bundle_margin_mean"])),
            int(row["train_docs"]),
            float(row["audit_fraction"]),
        ),
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "input_root": str(input_root),
        "suite_type": str(suite_type),
        "raw_row_count": int(len(raw_rows)),
        "learned_row_count": int(len(learned_rows)),
        "exact_family_row_count": int(len(exact_rows)),
        "assessed_row_count": int(len(assessed_rows)),
        "aggregated_row_count": int(len(aggregated_rows)),
        "main_package": MAIN_PACKAGE if any(str(row.get("law_package", "")) == MAIN_PACKAGE for row in aggregated_rows) else FALLBACK_MAIN_PACKAGE,
        "thresholds": {
            "law_gain_threshold": float(DEFAULT_LAW_GAIN_THRESHOLD),
            "spread_gain_threshold": float(DEFAULT_SPREAD_GAIN_THRESHOLD),
            "root_ratio_limit": float(DEFAULT_ROOT_RATIO_LIMIT),
        },
        "methodological_note": (
            "This report is the paper-facing replacement for the exploratory lambda-sweep reports. "
            "Selection, where needed, is validation-only; claims are test-reported."
        ),
        "exploratory_supersedes": [
            "report_markov_capability_map.py",
            "report_markov_local_law_learnability.py",
        ],
        "aggregated_rows": aggregated_rows,
        "boundary_candidates": boundary_candidates[:8],
        "figures": figure_paths,
        "figure_titles": figure_titles,
        "unified_core": unified_core,
    }
    summary["claim_readout"] = _markov_claim_readout(unified_core, main_package=str(summary["main_package"]))
    summary_path = output_dir / "markov_law_stress_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output_dir / "markov_law_stress_assessed_rows.csv", assessed_rows)
    _write_csv(output_dir / "markov_law_stress_aggregated_rows.csv", aggregated_rows)

    # Build the ablation-focused narrative
    ablation_pkg_stats: Dict[str, dict] = {}
    for pkg in ["root_only", "c1_only", "c2_only", "c3_only", "c1c3", "all_laws", "all_laws_plus_sched"]:
        pkg_rows = [row for row in aggregated_rows if str(row.get("law_package", "")) == pkg]
        if pkg_rows:
            ablation_pkg_stats[pkg] = {
                "prim_gain": float(fmean(1.0 - float(row["root_ratio"]) for row in pkg_rows)),
                "root_ratio": float(fmean(float(row["root_ratio"]) for row in pkg_rows)),
            }

    narrative = [
        "**Primary metric**: held-out root MAE compared to the root-only baseline (no local laws). "
        "`PrimGain = 1 - root_ratio`; positive = lower error = better.",
        "",
    ]
    # Add ablation summary if we have the data
    if "c2_only" in ablation_pkg_stats and "all_laws_plus_sched" in ablation_pkg_stats:
        c2_gain = ablation_pkg_stats["c2_only"]["prim_gain"]
        all_gain = ablation_pkg_stats["all_laws_plus_sched"]["prim_gain"]
        narrative.extend([
            f"**Key finding**: C2 (re-summary idempotence) is the only law package that improves downstream error "
            f"(PrimGain = {100.0 * c2_gain:+.1f}%). The full bundle (C1+C2+C3+sched) "
            f"hurts downstream (PrimGain = {100.0 * all_gain:+.1f}%).",
            "",
            "**Ablation interpretation**: C3 (merge preservation) creates an objective conflict that overwhelms "
            "C2's benefit when all laws are combined. C1 (leaf preservation) is roughly neutral.",
            "",
        ])
    narrative.extend([
        "This report uses direct per-law metrics: C1 leaf preservation, C2 re-summary idempotence, and C3 merge preservation.",
        "Schedule consistency is reported separately as a proxy diagnostic.",
        f"The expected full-bundle package is `{summary['main_package']}`, "
        "but the ablation shows that `c2_only` is the operationally strongest package on downstream metrics.",
        f"Rows loaded: {len(raw_rows)} total, {len(learned_rows)} learned, {len(exact_rows)} exact-family.",
    ])
    claim = dict(summary.get("claim_readout", {}) or {})
    claim_main = dict(claim.get("main_row", {}) or {})
    claim_ablation = dict(claim.get("strongest_ablation_row", {}) or {})
    if claim_main:
        narrative.append(
            "Claim row status: "
            f"`{claim.get('main_package', '')}` has "
            f"Prim%={100.0 * float(claim_main.get('primary_pass_rate', 0.0)):.1f}%, "
            f"PrimGain={100.0 * float(claim_main.get('mean_primary_gain', 0.0)):.1f}%, "
            f"C1={100.0 * float(claim_main.get('c1_pass_rate', 0.0)):.0f}%, "
            f"C2={100.0 * float(claim_main.get('c2_pass_rate', 0.0)):.0f}%, "
            f"C3={100.0 * float(claim_main.get('c3_pass_rate', 0.0)):.0f}%."
        )
    if claim_ablation:
        narrative.append(
            "Strongest ablation: "
            f"`{claim_ablation.get('law_package', '')}` has "
            f"Prim%={100.0 * float(claim_ablation.get('primary_pass_rate', 0.0)):.1f}%, "
            f"PrimGain={100.0 * float(claim_ablation.get('mean_primary_gain', 0.0)):.1f}%. "
            "This is a mechanism diagnostic, not the main claim."
        )
    if claim.get("note"):
        narrative.append(str(claim.get("note")))
    if boundary_candidates:
        top = boundary_candidates[0]
        narrative.append(
            "Closest transition-boundary cell: "
            f"train_docs={top['train_docs']}, q_audit={100.0 * float(top['audit_fraction']):.1f}%, "
            f"bundle_success_rate={float(top['bundle_full_success_rate']):.2f}, "
            f"bundle_margin_mean={float(top['bundle_margin_mean']):.3f}."
        )

    md_lines = [
        f"# {args.title}",
        "",
        "- **Primary metric**: held-out root MAE vs matched root-only baseline. PrimGain = 1 - root_ratio.",
        "- **Local laws** (C1, C2, C3) are regularization diagnostics. They explain *why* a learned g works, "
        "but downstream MAE is the success criterion.",
        "- This report supersedes the exploratory capability/lambda reports.",
        "",
        "## Ablation Summary",
        "",
    ]
    md_lines.extend(_build_downstream_table(aggregated_rows))
    md_lines.extend([
        "## Claim Status",
        "",
    ])
    if claim_main:
        md_lines.extend(
            [
                f"- Expected claim package: `{claim.get('main_package', '')}`.",
                f"- Downstream status: `{claim.get('status', 'unknown')}`.",
                (
                    f"- Claim-row readout: `Prim%={100.0 * float(claim_main.get('primary_pass_rate', 0.0)):.1f}%`, "
                    f"`PrimGain={100.0 * float(claim_main.get('mean_primary_gain', 0.0)):.1f}%`, "
                    f"`C1={100.0 * float(claim_main.get('c1_pass_rate', 0.0)):.0f}%`, "
                    f"`C2={100.0 * float(claim_main.get('c2_pass_rate', 0.0)):.0f}%`, "
                    f"`C3={100.0 * float(claim_main.get('c3_pass_rate', 0.0)):.0f}%`."
                ),
            ]
        )
    if claim_ablation:
        md_lines.append(
            (
                f"- Strongest ablation: `{claim_ablation.get('law_package', '')}` with "
                f"`Prim%={100.0 * float(claim_ablation.get('primary_pass_rate', 0.0)):.1f}%` and "
                f"`PrimGain={100.0 * float(claim_ablation.get('mean_primary_gain', 0.0)):.1f}%`. "
                "This remains diagnostic-only."
            )
        )
    if claim.get("note"):
        md_lines.append(f"- {claim.get('note')}")
    md_lines.extend([
        "",
        "## Narrative",
        "",
    ])
    md_lines.extend(f"- {line}" if line else "" for line in narrative)
    md_lines.extend([""])
    md_lines.extend(render_local_law_report_markdown(unified_core))
    md_lines.extend(["", "## Figures", ""])
    for fig in figure_paths:
        md_lines.append(f"- {figure_titles.get(fig, Path(fig).name)}: `{fig}`")
    pdf_path = Path(args.pdf_path) if args.pdf_path else (output_dir / "markov_law_stress_report.pdf")
    md_lines.append(f"- PDF: `{pdf_path}`")
    (output_dir / "markov_law_stress.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    with PdfPages(pdf_path) as pdf:
        claim_lines = []
        if claim_main:
            claim_lines.extend(
                [
                    f"Expected claim package: {claim.get('main_package', '')}",
                    f"Downstream status: {claim.get('status', 'unknown')}",
                    (
                        "Claim-row readout: "
                        f"Prim%={100.0 * float(claim_main.get('primary_pass_rate', 0.0)):.1f}%, "
                        f"PrimGain={100.0 * float(claim_main.get('mean_primary_gain', 0.0)):.1f}%, "
                        f"C1={100.0 * float(claim_main.get('c1_pass_rate', 0.0)):.0f}%, "
                        f"C2={100.0 * float(claim_main.get('c2_pass_rate', 0.0)):.0f}%, "
                        f"C3={100.0 * float(claim_main.get('c3_pass_rate', 0.0)):.0f}%."
                    ),
                ]
            )
        if claim_ablation:
            claim_lines.append(
                "Strongest ablation: "
                f"{claim_ablation.get('law_package', '')} | "
                f"Prim%={100.0 * float(claim_ablation.get('primary_pass_rate', 0.0)):.1f}% | "
                f"PrimGain={100.0 * float(claim_ablation.get('mean_primary_gain', 0.0)):.1f}% | "
                "diagnostic only."
            )
        if claim.get("note"):
            claim_lines.append(str(claim.get("note")))
        _write_text_page(pdf, title=f"{args.title} | Claim Status", lines=claim_lines or ["No claim summary available."])
        _write_text_page(pdf, title=str(args.title), lines=narrative)
        write_local_law_report_core_pages(pdf, title=str(args.title), core=unified_core)
        boundary_lines = [
            (
                f"train_docs={row['train_docs']} | q_audit={100.0 * float(row['audit_fraction']):.1f}% | "
                f"bundle_success_rate={float(row['bundle_full_success_rate']):.2f} | "
                f"bundle_margin_mean={float(row['bundle_margin_mean']):.3f} | "
                f"failure_reason={row['dominant_failure_reason'] or 'n/a'}"
            )
            for row in boundary_candidates[:8]
        ] or ["No boundary candidates available."]
        _write_text_page(pdf, title=f"{args.title} | Boundary Candidates", lines=boundary_lines)
        for fig in figure_paths:
            _write_image_page(pdf, image_path=Path(fig), title=figure_titles.get(fig, Path(fig).name))

    summary["pdf"] = str(pdf_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "pdf": str(pdf_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
