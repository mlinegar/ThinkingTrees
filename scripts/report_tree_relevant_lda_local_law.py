#!/usr/bin/env python3
"""Build a local-law companion PDF report for tree-relevant LDA.

.. deprecated::
    Use ``python -m src.ctreepo.cli sim suite law-stress report --family lda --output-root ...`` instead.
"""

from __future__ import annotations

import warnings
warnings.warn(
    "Deprecated. Use python -m src.ctreepo.cli sim suite law-stress report --family lda --output-root ...",
    DeprecationWarning,
    stacklevel=1,
)

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean
import sys
import textwrap
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.contracts import (  # noqa: E402
    LAW_ID_LEAF_PRESERVATION,
    LAW_ID_MERGE_PRESERVATION,
    LAW_ID_ON_RANGE_IDEMPOTENCE,
    assert_public_contract_clean,
    canonical_law_set_id,
)

from src.ctreepo.sim.local_law_report_common import (
    build_local_law_report_core,
    load_local_law_runs,
    render_local_law_report_markdown,
    write_local_law_report_core_pages,
)
from src.ctreepo.sim.util import safe_float


DELTA_CMAP = LinearSegmentedColormap.from_list("delta", ["#b2182b", "#ffffff", "#1a9850"])
LOW_GOOD_CMAP = LinearSegmentedColormap.from_list("low_good", ["#1a9850", "#f7f7f7", "#b2182b"])
HIGH_GOOD_CMAP = LinearSegmentedColormap.from_list("high_good", ["#b2182b", "#f7f7f7", "#1a9850"])
MODE_COLORS = {
    "aligned": "#1b9e77",
    "coarsen_2x": "#d95f02",
    "shift_half": "#7570b3",
    "random_same_count": "#666666",
}

_PUBLIC_KEY_RENAMES = {
    "families": "problem_ids",
    "family": "problem_id",
    "law_package": "law_set_id",
}


def _canonical_public_payload(value):
    if isinstance(value, dict):
        out = {}
        for raw_key, child in value.items():
            key = _PUBLIC_KEY_RENAMES.get(str(raw_key), str(raw_key))
            if key == "law_set_id":
                try:
                    child = canonical_law_set_id(str(child), allow_aliases=True)
                except Exception:
                    pass
            out[key] = _canonical_public_payload(child)
        return out
    if isinstance(value, list):
        return [_canonical_public_payload(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the tree-relevant LDA local-law companion report.")
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--snapshot-label", type=str, default="current sweep")
    return p.parse_args()


def _safe_float(x, default: float = float("nan")) -> float:
    return safe_float(x, default=default)


def _safe_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(fmean(vals))


def _safe_sem(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if len(vals) <= 1:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(var / len(vals))


def _page_header(fig, title: str, subtitle: str) -> None:
    title_wrapped = textwrap.fill(title, width=60)
    title_lines = title_wrapped.count("\n") + 1
    fig.text(0.06, 0.965, title_wrapped, fontsize=18, fontweight="bold", ha="left", va="top")
    subtitle_y = 0.965 - 0.046 * title_lines
    fig.text(0.06, subtitle_y, textwrap.fill(subtitle, width=130), fontsize=10.5, color="#444444", ha="left", va="top")


def _caption(fig, text: str, *, x: float = 0.06, y: float = 0.07, width: int = 142, fontsize: int = 10) -> None:
    fig.text(x, y, textwrap.fill(text, width=width), fontsize=fontsize, ha="left", va="top")


def _paragraph(ax, x: float, y: float, text: str, *, width: int = 84, fontsize: int = 11) -> None:
    ax.text(x, y, textwrap.fill(str(text).strip(), width=width), fontsize=fontsize, va="top", ha="left", linespacing=1.35)


def _annotate_heatmap(ax, matrix: np.ndarray, *, fmt: str = "{:.2f}", fontsize: int = 9) -> None:
    arr = np.asarray(matrix, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return
    threshold = 0.45 * float(np.nanmax(np.abs(finite))) if float(np.nanmax(np.abs(finite))) > 0 else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if not np.isfinite(val):
                continue
            color = "white" if abs(float(val)) >= threshold and threshold > 0 else "#111111"
            ax.text(j, i, fmt.format(float(val)), ha="center", va="center", fontsize=fontsize, color=color)


def _save_page(
    pdf: PdfPages,
    fig,
    *,
    left: float = 0.08,
    right: float = 0.95,
    top: float = 0.86,
    bottom: float = 0.14,
    wspace: float = 0.28,
    hspace: float = 0.34,
) -> None:
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)
    pdf.savefig(fig)
    plt.close(fig)


def _mode_label(mode: str) -> str:
    return {
        "aligned": "Aligned",
        "coarsen_2x": "Coarsen 2x",
        "shift_half": "Shift half-block",
        "random_same_count": "Random same-count",
    }.get(str(mode), str(mode).replace("_", " ").title())


def _design_label(leaf_design: str, internal_design: str) -> str:
    leaf = {
        "uniform": "Uniform",
        "proxy_priority": "Priority",
        "proxy_adversarial": "Adversarial",
    }.get(str(leaf_design), str(leaf_design))
    internal = {"uniform": "Uniform", "risk": "Risk"}.get(str(internal_design), str(internal_design))
    return f"{leaf}\n/ {internal}"


def _nested_get(payload: dict, path: Sequence[str], default=float("nan")):
    cur = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _fmt_weight(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    rounded = round(float(value))
    if abs(float(value) - float(rounded)) <= 1e-9:
        return str(int(rounded))
    return f"{float(value):.3g}"


def _law_score_label(records: Sequence[Dict[str, object]], *, multiline: bool = False) -> str:
    tuples = {
        (
            round(_safe_float(dict(rec.get("local_law_component_weights", {}) or {}).get(LAW_ID_LEAF_PRESERVATION)), 12),
            round(_safe_float(dict(rec.get("local_law_component_weights", {}) or {}).get(LAW_ID_ON_RANGE_IDEMPOTENCE)), 12),
            round(_safe_float(dict(rec.get("local_law_component_weights", {}) or {}).get(LAW_ID_MERGE_PRESERVATION)), 12),
        )
        for rec in records
        if dict(rec.get("local_law_component_weights", {}) or {})
    }
    label = "Configured local-law score"
    if len(tuples) != 1:
        return label
    c1_w, c2_w, c3_w = next(iter(tuples))
    terms = [
        f"{_fmt_weight(c1_w)}*C1" if abs(c1_w) > 1e-12 else "",
        f"{_fmt_weight(c2_w)}*C2-proxy" if abs(c2_w) > 1e-12 else "",
        f"{_fmt_weight(c3_w)}*C3" if abs(c3_w) > 1e-12 else "",
    ]
    formula = " + ".join(term for term in terms if term) or "no local-law term"
    sep = "\n" if multiline else " "
    return f"{label}{sep}({formula})"


def _first_finite(*values: object) -> float:
    for value in values:
        candidate = _safe_float(value)
        if math.isfinite(candidate):
            return float(candidate)
    return float("nan")


def _explicit_config_float(config: Dict[str, object], key: str) -> float:
    if key not in config:
        return float("nan")
    return _safe_float(config.get(key))


def _resolve_lda_law_set_id(
    *,
    objective: Dict[str, object],
    summary_metadata: Dict[str, object],
    local_law_cfg: Dict[str, object],
    top_cfg: Dict[str, object],
) -> str:
    for mapping, key in (
        (objective, "law_set_id"),
        (summary_metadata, "law_set_id"),
        (local_law_cfg, "law_set_id"),
        (top_cfg, "law_set_id"),
    ):
        value = str(mapping.get(key, "") or "").strip()
        if value:
            return value
    return "unknown"


def _resolve_lda_law_weights(
    *,
    objective: Dict[str, object],
    summary_metadata: Dict[str, object],
    local_law_cfg: Dict[str, object],
    top_cfg: Dict[str, object],
) -> Dict[str, float]:
    objective_weights = dict(objective.get("local_law_component_weights", {}) or {})
    if objective_weights:
        return {
            "c1": _safe_float(objective_weights.get(LAW_ID_LEAF_PRESERVATION), 0.0),
            "c2_proxy": _safe_float(objective_weights.get(LAW_ID_ON_RANGE_IDEMPOTENCE), 0.0),
            "c3": _safe_float(objective_weights.get(LAW_ID_MERGE_PRESERVATION), 0.0),
        }

    metadata_weights = dict(summary_metadata.get("resolved_local_law_weights", {}) or {})
    if metadata_weights:
        return {
            "c1": _safe_float(metadata_weights.get(LAW_ID_LEAF_PRESERVATION), 0.0),
            "c2_proxy": _safe_float(metadata_weights.get(LAW_ID_ON_RANGE_IDEMPOTENCE), 0.0),
            "c3": _safe_float(metadata_weights.get(LAW_ID_MERGE_PRESERVATION), 0.0),
        }

    base_weights = {
        "c1": _first_finite(
            _explicit_config_float(local_law_cfg, "law_c1_weight"),
            _explicit_config_float(top_cfg, "law_c1_weight"),
        ),
        "c2_proxy": _first_finite(
            _explicit_config_float(local_law_cfg, "law_c2_proxy_weight"),
            _explicit_config_float(top_cfg, "law_c2_proxy_weight"),
        ),
        "c3": _first_finite(
            _explicit_config_float(local_law_cfg, "law_c3_weight"),
            _explicit_config_float(top_cfg, "law_c3_weight"),
        ),
    }
    if not any(math.isfinite(value) for value in base_weights.values()):
        return {name: float("nan") for name in ("c1", "c2_proxy", "c3")}

    pkg = _resolve_lda_law_set_id(
        objective=objective,
        summary_metadata=summary_metadata,
        local_law_cfg=local_law_cfg,
        top_cfg=top_cfg,
    ).strip().lower()
    if pkg == "root_only":
        return {"c1": 0.0, "c2_proxy": 0.0, "c3": 0.0}
    if pkg == "c1_only":
        return {"c1": _safe_float(base_weights["c1"], 0.0), "c2_proxy": 0.0, "c3": 0.0}
    if pkg == "c2_only":
        return {"c1": 0.0, "c2_proxy": _safe_float(base_weights["c2_proxy"], 0.0), "c3": 0.0}
    if pkg == "c3_only":
        return {"c1": 0.0, "c2_proxy": 0.0, "c3": _safe_float(base_weights["c3"], 0.0)}
    if pkg == "c1c3":
        return {
            "c1": _safe_float(base_weights["c1"], 0.0),
            "c2_proxy": 0.0,
            "c3": _safe_float(base_weights["c3"], 0.0),
        }
    return {
        name: (_safe_float(value, 0.0) if math.isfinite(_safe_float(value)) else float("nan"))
        for name, value in base_weights.items()
    }


def _policy_law_score(policy_metrics: Dict[str, object], *, law_weights: Dict[str, float]) -> float:
    if not policy_metrics:
        return float("nan")
    fallback = _safe_float(policy_metrics.get("combined_law_score"))
    active = (
        ("c1", "mean_c1"),
        ("c2_proxy", "mean_c2_proxy"),
        ("c3", "mean_c3"),
    )
    total = 0.0
    saw_active = False
    for weight_key, metric_key in active:
        weight = _safe_float(law_weights.get(weight_key))
        if not math.isfinite(weight):
            return fallback
        if abs(weight) <= 1e-12:
            continue
        raw_value = _safe_float(policy_metrics.get(metric_key))
        if not math.isfinite(raw_value):
            return fallback
        saw_active = True
        total += float(weight) * float(raw_value)
    if saw_active:
        return float(total)
    if all(math.isfinite(_safe_float(law_weights.get(key))) for key, _ in active):
        return 0.0
    return fallback


def _objective_weight_profiles(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    counts: Dict[Tuple[str, Tuple[Tuple[str, float], ...]], int] = defaultdict(int)
    for rec in records:
        component_weights = {
            str(k): round(_safe_float(v), 12)
            for k, v in dict(rec.get("local_law_component_weights", {}) or {}).items()
        }
        if not component_weights:
            continue
        key = (
            str(rec.get("law_set_id", "unknown") or "unknown"),
            tuple(sorted(component_weights.items())),
        )
        counts[key] = int(counts.get(key, 0)) + 1
    rows: List[Dict[str, object]] = []
    for (law_set_id, component_items), n_runs in sorted(
        counts.items(),
        key=lambda item: (-int(item[1]), str(item[0][0]), item[0][1]),
    ):
        rows.append(
            {
                "law_set_id": str(law_set_id),
                "local_law_component_weights": {
                    str(k): float(v) for k, v in component_items
                },
                "n_runs": int(n_runs),
            }
        )
    return rows


def _load_records(input_root: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    results_root = input_root / "results"
    for path in sorted(results_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = dict(payload.get("config", {}))
        local_law = dict(payload.get("local_law", {}))
        local_law_cfg = dict(local_law.get("config", {}) or {})
        learnability = dict(payload.get("local_law_learnability", {}) or {})
        summary_metadata = dict(learnability.get("metadata", {}) or {})
        objective = dict(local_law.get("objective", {}) or summary_metadata.get("objective", {}) or {})
        local_law_weights = _resolve_lda_law_weights(
            objective=objective,
            summary_metadata=summary_metadata,
            local_law_cfg=local_law_cfg,
            top_cfg=cfg,
        )
        policy_metrics = dict(local_law.get("policy_metrics", {}))
        ipw_eval = dict(local_law.get("ipw_evaluation", {}))
        methods = dict(payload.get("methods", {}))
        rel = path.relative_to(results_root)
        suite = rel.parts[0] if rel.parts else "unknown"
        objective_name = str(
            objective.get(
                "selection_metric_name",
                summary_metadata.get("configured_objective_name", "configured_objective"),
            )
        )
        finite_law_weights = [
            float(value)
            for value in local_law_weights.values()
            if math.isfinite(_safe_float(value))
        ]
        local_law_weight_total = (
            float(sum(finite_law_weights)) if finite_law_weights else float("nan")
        )
        identity_metrics = dict(policy_metrics.get("infer_identity", {}) or {})
        oracle_metrics = dict(policy_metrics.get("oracle_true_summary", {}) or {})
        naive_metrics = dict(policy_metrics.get("law_calibrated_naive", {}) or {})
        ipw_metrics = dict(policy_metrics.get("law_calibrated_ipw", {}) or {})
        stab_metrics = dict(policy_metrics.get("law_calibrated_ipw_stabilized", {}) or {})
        rec = {
            "path": str(path),
            "suite": suite,
            "mode": str(cfg.get("analysis_partition_mode", "")),
            "tau": _safe_float(cfg.get("local_mixture_concentration")),
            "lam": _safe_float(cfg.get("quadratic_utility_weight")),
            "quadratic_utility_weight": _safe_float(cfg.get("quadratic_utility_weight")),
            "seed": int(cfg.get("seed", 0)),
            "train_docs": int(cfg.get("train_docs", 0)),
            "objective_name": objective_name,
            "objective_weighting_scheme": str(objective.get("weighting_scheme", "")),
            "root_share": _first_finite(
                objective.get("root_share"),
                local_law_cfg.get("root_share"),
                cfg.get("root_share"),
            ),
            "local_law_weight": _first_finite(
                objective.get("local_law_weight"),
                local_law_cfg.get("local_law_weight"),
                cfg.get("local_law_weight"),
                local_law_weight_total,
            ),
            "local_law_component_weights": {
                LAW_ID_LEAF_PRESERVATION: _safe_float(local_law_weights.get("c1"), 0.0),
                LAW_ID_ON_RANGE_IDEMPOTENCE: _safe_float(local_law_weights.get("c2_proxy"), 0.0),
                LAW_ID_MERGE_PRESERVATION: _safe_float(local_law_weights.get("c3"), 0.0),
            },
            "objective_total_weight_without_proxy": _first_finite(
                objective.get("total_weight_without_proxy"),
                objective.get("weight_total_without_proxy"),
            ),
            "law_leaf_query_rate": _safe_float(cfg.get("law_leaf_query_rate")),
            "law_internal_query_rate": _safe_float(cfg.get("law_internal_query_rate")),
            "law_leaf_query_design": str(cfg.get("law_leaf_query_design", "")),
            "law_internal_query_design": str(cfg.get("law_internal_query_design", "")),
            "anchor_multiplier": _safe_float(cfg.get("anchor_multiplier")),
            "topic_concentration": _safe_float(cfg.get("topic_concentration")),
            "identity_objective": _safe_float(identity_metrics.get(objective_name)),
            "identity_c1": _safe_float(identity_metrics.get("mean_c1")),
            "identity_c3": _safe_float(identity_metrics.get("mean_c3")),
            "identity_c2": _safe_float(identity_metrics.get("mean_c2_proxy")),
            "identity_law_score": _policy_law_score(identity_metrics, law_weights=local_law_weights),
            "identity_combined": _policy_law_score(identity_metrics, law_weights=local_law_weights),
            "identity_task_objective": _safe_float(
                identity_metrics.get("mean_aux_oracle_target_abs_error")
            ),
            "identity_aux_abs": _safe_float(identity_metrics.get("mean_aux_oracle_target_abs_error")),
            "identity_aux_delta": _safe_float(identity_metrics.get("mean_aux_oracle_target_delta")),
            "oracle_objective": _safe_float(oracle_metrics.get(objective_name)),
            "oracle_c1": _safe_float(oracle_metrics.get("mean_c1")),
            "oracle_c3": _safe_float(oracle_metrics.get("mean_c3")),
            "oracle_c2": _safe_float(oracle_metrics.get("mean_c2_proxy")),
            "oracle_law_score": _policy_law_score(oracle_metrics, law_weights=local_law_weights),
            "oracle_combined": _policy_law_score(oracle_metrics, law_weights=local_law_weights),
            "naive_objective": _safe_float(naive_metrics.get(objective_name)),
            "naive_law_score": _policy_law_score(naive_metrics, law_weights=local_law_weights),
            "naive_combined": _policy_law_score(naive_metrics, law_weights=local_law_weights),
            "ipw_objective": _safe_float(ipw_metrics.get(objective_name)),
            "ipw_law_score": _policy_law_score(ipw_metrics, law_weights=local_law_weights),
            "ipw_combined": _policy_law_score(ipw_metrics, law_weights=local_law_weights),
            "stab_objective": _safe_float(stab_metrics.get(objective_name)),
            "stab_law_score": _policy_law_score(stab_metrics, law_weights=local_law_weights),
            "stab_combined": _policy_law_score(stab_metrics, law_weights=local_law_weights),
            "naive_aux_abs": _safe_float(naive_metrics.get("mean_aux_oracle_target_abs_error")),
            "ipw_aux_abs": _safe_float(ipw_metrics.get("mean_aux_oracle_target_abs_error")),
            "stab_aux_abs": _safe_float(stab_metrics.get("mean_aux_oracle_target_abs_error")),
            "naive_aux_delta": _safe_float(naive_metrics.get("mean_aux_oracle_target_delta")),
            "ipw_aux_delta": _safe_float(ipw_metrics.get("mean_aux_oracle_target_delta")),
            "stab_aux_delta": _safe_float(stab_metrics.get("mean_aux_oracle_target_delta")),
            "analysis_delta": _safe_float(_nested_get(methods, ("analysis_infer_weighted_sum", "delta_mean"))),
            "law_method_delta": _safe_float(_nested_get(methods, ("analysis_infer_law_calibrated_oracle_target", "delta_mean"))),
            "ipw_eval_identity_combined_ht_abs_error": _safe_float(_nested_get(ipw_eval, ("infer_identity", "combined", "ht_abs_error"))),
            "ipw_eval_identity_combined_hajek_abs_error": _safe_float(_nested_get(ipw_eval, ("infer_identity", "combined", "hajek_abs_error"))),
            "ipw_eval_identity_combined_width": _safe_float(_nested_get(ipw_eval, ("infer_identity", "combined", "eb_width"))),
            "ipw_eval_identity_leaf_ess": _safe_float(_nested_get(ipw_eval, ("infer_identity", "diagnostics", "leaf_effective_sample_size"))),
            "ipw_eval_identity_internal_ess": _safe_float(_nested_get(ipw_eval, ("infer_identity", "diagnostics", "internal_effective_sample_size"))),
            "ipw_eval_identity_leaf_max_weight": _safe_float(_nested_get(ipw_eval, ("infer_identity", "diagnostics", "leaf_max_weight"))),
            "ipw_eval_identity_internal_max_weight": _safe_float(_nested_get(ipw_eval, ("infer_identity", "diagnostics", "internal_max_weight"))),
            "ipw_eval_law_ipw_combined_ht_abs_error": _safe_float(_nested_get(ipw_eval, ("law_calibrated_ipw", "combined", "ht_abs_error"))),
            "ipw_eval_law_ipw_combined_hajek_abs_error": _safe_float(_nested_get(ipw_eval, ("law_calibrated_ipw", "combined", "hajek_abs_error"))),
            "ipw_eval_law_ipw_combined_width": _safe_float(_nested_get(ipw_eval, ("law_calibrated_ipw", "combined", "eb_width"))),
            "ipw_eval_law_ipw_leaf_ess": _safe_float(_nested_get(ipw_eval, ("law_calibrated_ipw", "diagnostics", "leaf_effective_sample_size"))),
            "ipw_eval_law_ipw_internal_ess": _safe_float(_nested_get(ipw_eval, ("law_calibrated_ipw", "diagnostics", "internal_effective_sample_size"))),
            "ipw_eval_law_ipw_leaf_max_weight": _safe_float(_nested_get(ipw_eval, ("law_calibrated_ipw", "diagnostics", "leaf_max_weight"))),
            "ipw_eval_law_ipw_internal_max_weight": _safe_float(_nested_get(ipw_eval, ("law_calibrated_ipw", "diagnostics", "internal_max_weight"))),
            # Law-stress classification fields
            "law_set_id": _resolve_lda_law_set_id(
                objective=objective,
                summary_metadata=summary_metadata,
                local_law_cfg=local_law_cfg,
                top_cfg=cfg,
            ),
            "exact_family": str(local_law.get("config", {}).get("exact_family", "")),
        }
        # Extract law_stress assessment for the best calibrator (stabilized IPW)
        law_stress = dict(local_law.get("law_stress", {}))
        best_stress = dict(law_stress.get("law_calibrated_ipw_stabilized", law_stress.get("law_calibrated_ipw", {})))
        rec["stress_bundle_status"] = str(best_stress.get("bundle_status", ""))
        rec["stress_bundle_full_success"] = bool(best_stress.get("bundle_full_success", False))
        rec["stress_c1_pass"] = bool(best_stress.get("c1_pass", False))
        rec["stress_c2_pass"] = bool(best_stress.get("c2_pass", False))
        rec["stress_c3_pass"] = bool(best_stress.get("c3_pass", False))
        rec["stress_root_pass"] = bool(best_stress.get("root_pass", False))
        rec["stress_c1_gain"] = _safe_float(best_stress.get("c1_gain_frac"))
        rec["stress_c2_gain"] = _safe_float(best_stress.get("c2_gain_frac"))
        rec["stress_c3_gain"] = _safe_float(best_stress.get("c3_gain_frac"))
        rec["stress_root_ratio"] = _safe_float(best_stress.get("root_ratio"))
        # Extract exact-family metrics if present
        for fam_key in ("exact_oracle", "exact_scrambled_topics", "exact_uniform_prior", "exact_adversarial_merge"):
            fam_pm = dict(policy_metrics.get(fam_key, {}))
            if fam_pm:
                rec[f"{fam_key}_c1"] = _safe_float(fam_pm.get("mean_c1"))
                rec[f"{fam_key}_c3"] = _safe_float(fam_pm.get("mean_c3"))
                rec[f"{fam_key}_c2"] = _safe_float(fam_pm.get("mean_c2_proxy"))
        records.append(rec)
    return records


def _filter(records: Sequence[Dict[str, object]], **conds) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for rec in records:
        ok = True
        for key, val in conds.items():
            if rec.get(key) != val:
                ok = False
                break
        if ok:
            out.append(rec)
    return out


def _matrix(
    records: Sequence[Dict[str, object]],
    row_keys: Sequence[object],
    col_keys: Sequence[object],
    *,
    row_field: str,
    col_field: str,
    value_field: str,
    fixed: Dict[str, object] | None = None,
) -> np.ndarray:
    arr = np.full((len(row_keys), len(col_keys)), np.nan, dtype=np.float64)
    fixed = dict(fixed or {})
    for i, row_key in enumerate(row_keys):
        for j, col_key in enumerate(col_keys):
            vals = []
            for rec in records:
                if rec.get(row_field) != row_key or rec.get(col_field) != col_key:
                    continue
                if any(rec.get(k) != v for k, v in fixed.items()):
                    continue
                vals.append(_safe_float(rec.get(value_field)))
            arr[i, j] = _safe_mean(vals)
    return arr


def _line_stats(
    records: Sequence[Dict[str, object]],
    x_values: Sequence[object],
    *,
    x_field: str,
    value_field: str,
    fixed: Dict[str, object] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    fixed = dict(fixed or {})
    means: List[float] = []
    sems: List[float] = []
    for x in x_values:
        vals = []
        for rec in records:
            if rec.get(x_field) != x:
                continue
            if any(rec.get(k) != v for k, v in fixed.items()):
                continue
            vals.append(_safe_float(rec.get(value_field)))
        means.append(_safe_mean(vals))
        sems.append(_safe_sem(vals))
    return np.asarray(means, dtype=np.float64), np.asarray(sems, dtype=np.float64)


def _scatter(ax, xs: Sequence[float], ys: Sequence[float], colors: Sequence[str], *, xlabel: str, ylabel: str) -> None:
    if not xs or not ys:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        return
    ax.scatter(xs, ys, c=colors, alpha=0.65, s=32, edgecolors="none")
    ax.axhline(0.0, color="#999999", lw=1.0, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, lw=0.6)


def _summarize_records(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    weight_profiles = _objective_weight_profiles(records)
    summary: Dict[str, object] = {
        "n_results": int(len(records)),
        "suites": sorted({str(rec.get("suite")) for rec in records}),
        "law_score_label": _law_score_label(records),
        "law_score_is_uniform": bool(len(weight_profiles) == 1),
        "objective_weight_profiles": weight_profiles,
    }
    suite_a = _filter(records, suite="suite_a_exact_controls")
    if suite_a:
        summary["suite_a_oracle_objective_mean"] = _safe_mean(rec["oracle_objective"] for rec in suite_a)
        summary["suite_a_oracle_law_score_mean"] = _safe_mean(rec["oracle_law_score"] for rec in suite_a)
        groups: Dict[Tuple[object, object, object], List[float]] = defaultdict(list)
        for rec in suite_a:
            key = (rec.get("mode"), rec.get("tau"), rec.get("seed"))
            groups[key].append(_safe_float(rec.get("identity_law_score")))
        summary["suite_a_identity_law_score_quadratic_weight_range_mean"] = _safe_mean(
            (max(vals) - min(vals)) for vals in groups.values() if vals
        )
    suite_b = _filter(records, suite="suite_b_local_law_learnability")
    if suite_b:
        summary["suite_b_identity_objective_mean"] = _safe_mean(rec["identity_objective"] for rec in suite_b)
        summary["suite_b_ipw_objective_mean"] = _safe_mean(rec["ipw_objective"] for rec in suite_b)
        summary["suite_b_identity_law_score_mean"] = _safe_mean(rec["identity_law_score"] for rec in suite_b)
        summary["suite_b_ipw_law_score_mean"] = _safe_mean(rec["ipw_law_score"] for rec in suite_b)
    suite_c = _filter(records, suite="suite_c_mismatch_mediation")
    if suite_c:
        summary["suite_c_shift_half_identity_minus_aligned"] = (
            _safe_mean(rec["identity_law_score"] for rec in suite_c if rec.get("mode") == "shift_half")
            - _safe_mean(rec["identity_law_score"] for rec in suite_c if rec.get("mode") == "aligned")
        )
    suite_d = _filter(records, suite="suite_d_ipw_sparse_labels")
    if suite_d:
        summary["suite_d_ipw_minus_naive_law_score_gain"] = _safe_mean(
            _safe_float(rec.get("naive_law_score")) - _safe_float(rec.get("ipw_law_score"))
            for rec in suite_d
        )
        summary["suite_d_hajek_minus_ht_abs_error_gain"] = _safe_mean(
            _safe_float(rec.get("ipw_eval_law_ipw_combined_ht_abs_error")) - _safe_float(rec.get("ipw_eval_law_ipw_combined_hajek_abs_error"))
            for rec in suite_d
        )
    return summary


def _write_markdown(path: Path, *, snapshot_label: str, summary: Dict[str, object], unified_core: Dict[str, object]) -> None:
    law_score_label = str(summary.get("law_score_label", "Configured local-law score"))
    if bool(summary.get("law_score_is_uniform", False)):
        law_score_line = f"- Local-law score used in this inventory: `{law_score_label}`."
    else:
        law_score_line = (
            "- Local-law score is recomputed from each run's realized configured weights; "
            "see `objective_weight_profiles` in the summary JSON."
        )
    lines = [
        "# Tree-Relevant LDA Local-Law Companion",
        "",
        f"Snapshot: **{snapshot_label}**",
        "",
        "This report adds local-law diagnostics to the existing tree-relevant LDA realism ladder. The local laws act on analysis-section topic-mixture summaries, not directly on utility prediction.",
        "",
        "Reading guide:",
        "",
        "- `C1` is leaf-summary error: how far a calibrated section summary is from the true analysis-section mixture.",
        "- `C3` is merge error: how far the deterministic weighted merge of child summaries is from the true parent summary.",
        "- `C2-proxy` is self-consistency under expected counts and reinference. It is a simulation proxy, not a theorem-facing exact law.",
        law_score_line,
        "- `Delta = pooled held-out error - method held-out error`, so larger positive values favor the local method over pooling.",
        "",
        "Summary statistics:",
        "",
    ]
    lines.extend([""])
    lines.extend(render_local_law_report_markdown(unified_core))
    lines.extend(["", "## Family Summary", ""])
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir) if args.output_dir is not None else input_root / "local_law_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "tree_relevant_lda_local_law_report.pdf"
    md_path = output_dir / "tree_relevant_lda_local_law_report.md"
    summary_path = output_dir / "tree_relevant_lda_local_law_report_summary.json"

    records = _load_records(input_root)
    summary = _summarize_records(records)
    law_score_label = str(summary.get("law_score_label", "Configured local-law score"))
    law_score_label_multiline = _law_score_label(records, multiline=True)
    protocol_runs = load_local_law_runs(input_root)
    unified_core = build_local_law_report_core(protocol_runs)
    summary["unified_core"] = _canonical_public_payload(unified_core)

    with PdfPages(pdf_path) as pdf:
        write_local_law_report_core_pages(
            pdf,
            title="Tree-Relevant LDA Local-Law Companion",
            core=unified_core,
        )
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_axes([0.06, 0.18, 0.88, 0.70])
        ax.axis("off")
        _page_header(
            fig,
            "Why add local laws to the tree-relevant LDA realism ladder?",
            "How to read this page: this companion keeps the same tau / quadratic-weight document generator but asks whether summary quality can be measured, learned, and linked to downstream utility errors.",
        )
        _paragraph(
            ax,
            0.0,
            0.95,
            "The current realism report tells us when local section analysis beats pooling. This companion asks a sharper mechanism question: are the wins and losses driven by summary quality at the section and merge level, and can those local-law errors be learned under sparse adaptive labels?",
            width=92,
            fontsize=12,
        )
        _paragraph(
            ax,
            0.0,
            0.62,
            f"We measure three local quantities on analysis-section topic mixtures. C1 is leaf-summary error. C3 is the weighted-merge error obtained by recursively combining adjacent analysis sections. C2-proxy asks whether a summary is stable under expected counts and reinference. The displayed local-law score is always recomputed from the realized optimization weights, not a reporting default: {law_score_label}. These local-law metrics are quadratic-weight-free: the quadratic utility weight only matters when those summary errors are pushed through the nonlinear utility target.",
            width=92,
            fontsize=12,
        )
        _paragraph(
            ax,
            0.0,
            0.27,
            "The report then walks through four questions. Do the law definitions behave exactly in controls? Do the laws become learnable with more data and more sparse labels? Does boundary mismatch worsen law errors before downstream inference enters? And under adaptive labeling, do IPW-style weights help recover cleaner local summaries and more stable held-out law estimates?",
            width=92,
            fontsize=12,
        )
        _caption(fig, "Headline: local laws provide a mechanism layer between section-summary quality and the downstream pooled-vs-local utility gap.", y=0.10)
        _save_page(pdf, fig, top=0.89, bottom=0.14)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        _page_header(
            fig,
            "What exactly are C1, C3, and C2-proxy, and do the exact controls behave properly?",
            "How to read this page: smaller is better everywhere. The oracle row should be near zero, and the inferred row should stay effectively unchanged as the quadratic weight moves because the law metrics depend on summaries, not on the utility mapping.",
        )
        suite_a = _filter(records, suite="suite_a_exact_controls")
        row_modes = ["aligned", "shift_half"]
        col_taus = [1.0, 8.0, 16.0]
        oracle_c1 = _matrix(suite_a, row_modes, col_taus, row_field="mode", col_field="tau", value_field="oracle_c1")
        oracle_c3 = _matrix(suite_a, row_modes, col_taus, row_field="mode", col_field="tau", value_field="oracle_c3")
        oracle_c2 = _matrix(suite_a, row_modes, col_taus, row_field="mode", col_field="tau", value_field="oracle_c2")
        lambda_range = np.full((len(row_modes), len(col_taus)), np.nan, dtype=np.float64)
        for i, mode in enumerate(row_modes):
            for j, tau in enumerate(col_taus):
                vals_by_seed: Dict[int, List[float]] = defaultdict(list)
                for rec in suite_a:
                    if rec.get("mode") == mode and rec.get("tau") == tau:
                        vals_by_seed[int(rec.get("seed", 0))].append(_safe_float(rec.get("identity_law_score")))
                lambda_range[i, j] = _safe_mean(max(vals) - min(vals) for vals in vals_by_seed.values() if vals)
        panels = [
            (axes[0, 0], oracle_c1, "Oracle C1"),
            (axes[0, 1], oracle_c3, "Oracle C3"),
            (axes[1, 0], oracle_c2, "Oracle C2-proxy"),
            (axes[1, 1], lambda_range, "Identity local-law score\nrange across quadratic weight"),
        ]
        for ax, matrix, title in panels:
            im = ax.imshow(matrix, aspect="auto", cmap=LOW_GOOD_CMAP)
            ax.set_title(title, fontsize=11)
            ax.set_xticks(range(len(col_taus)), [f"tau={t:g}" for t in col_taus])
            ax.set_yticks(range(len(row_modes)), [_mode_label(m) for m in row_modes])
            _annotate_heatmap(ax, matrix, fmt="{:.2e}" if "Oracle" in title or "range" in title else "{:.3f}", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.80)
        _caption(fig, f"Oracle summaries drive C1 and C3 to machine-zero. C2-proxy is also near zero for oracle summaries because reinference from expected counts approximately reproduces the same topic mixture. The right panel reports the seedwise range of the inferred {law_score_label.lower()} across quadratic-weight settings and should stay near zero if the laws are genuinely quadratic-weight-free.", y=0.08)
        _save_page(pdf, fig, top=0.86, bottom=0.15)

        fig, axes = plt.subplots(1, 3, figsize=(11, 8.5))
        _page_header(
            fig,
            "Do local laws become learnable with more training data and more sparse law labels?",
            f"How to read this page: the y-axis is the configured local-law score, using the realized optimization weights for each run. Lower is better. Shaded bands are standard errors across seeds.",
        )
        suite_b = _filter(records, suite="suite_b_local_law_learnability")
        train_grid = [64, 128, 256, 512, 1024]
        rate_styles = {
            0.05: ("#b2182b", "5% leaf + 5% internal"),
            0.10: ("#2166ac", "10% leaf + 10% internal"),
            0.20: ("#1a9850", "20% leaf + 20% internal"),
        }
        for ax, tau in zip(axes, [1.0, 8.0, 16.0]):
            for rate, (color, label) in rate_styles.items():
                subset = [
                    rec for rec in suite_b
                    if rec.get("tau") == tau
                    and rec.get("law_leaf_query_rate") == rate
                    and rec.get("law_internal_query_rate") == rate
                ]
                mean_vals, sem_vals = _line_stats(subset, train_grid, x_field="train_docs", value_field="ipw_law_score")
                xs = np.asarray(train_grid, dtype=np.float64)
                ax.plot(xs, mean_vals, color=color, lw=2.0, label=label)
                ax.fill_between(xs, mean_vals - sem_vals, mean_vals + sem_vals, color=color, alpha=0.18)
            ax.set_title(f"tau={tau:g}")
            ax.set_xlabel("Training documents")
            ax.set_ylabel(law_score_label)
            ax.grid(alpha=0.25, lw=0.6)
        axes[0].legend(frameon=False, fontsize=9, loc="upper right")
        _caption(fig, "More labeled law units and more training documents both improve the learned summary calibrator. This is the learnability page for the mechanism layer, not yet the downstream utility outcome.", y=0.09)
        _save_page(pdf, fig, top=0.84, bottom=0.18)

        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        _page_header(
            fig,
            "Does boundary mismatch increase local-law error before any downstream utility mapping is applied?",
            "How to read this page: greener cells are lower error. The left panel is the raw inferred summary; the right panel applies the IPW-trained local-law calibrator as an auxiliary upper bound.",
        )
        suite_c = _filter(records, suite="suite_c_mismatch_mediation")
        modes = ["aligned", "coarsen_2x", "shift_half", "random_same_count"]
        taus = [1.0, 8.0, 16.0]
        identity_mat = _matrix(suite_c, modes, taus, row_field="mode", col_field="tau", value_field="identity_law_score", fixed={"lam": 1.5})
        ipw_mat = _matrix(suite_c, modes, taus, row_field="mode", col_field="tau", value_field="ipw_law_score", fixed={"lam": 1.5})
        for ax, matrix, title in zip(axes, [identity_mat, ipw_mat], ["Uncalibrated summary", "IPW-calibrated summary"]):
            im = ax.imshow(matrix, aspect="auto", cmap=LOW_GOOD_CMAP)
            ax.set_title(title, fontsize=12)
            ax.set_xticks(range(len(taus)), [f"tau={t:g}" for t in taus])
            ax.set_yticks(range(len(modes)), [_mode_label(m) for m in modes])
            _annotate_heatmap(ax, matrix, fmt="{:.2f}")
            fig.colorbar(im, ax=ax, shrink=0.80)
        _caption(fig, "Mismatch changes the local-law errors before any quadratic-weight-dependent utility is evaluated. The aligned case is the clean control; shifted and random boundaries should worsen C1/C3 because the analysis sections no longer line up with the latent local mixtures.", y=0.09)
        _save_page(pdf, fig, top=0.84, bottom=0.18)

        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        _page_header(
            fig,
            "When quadratic weight = 0, do larger local-law errors actually matter for downstream Delta?",
            f"How to read this page: each point is one run from the mismatch suite. The x-axis is the inferred {law_score_label.lower()}; the y-axis is Delta for the uncalibrated analysis-inference method. At quadratic weight = 0 the cloud should stay near Delta = 0 even when local-law scores differ.",
        )
        zero_records = [rec for rec in suite_c if rec.get("lam") == 0.0]
        colors = [MODE_COLORS.get(str(rec.get("mode")), "#666666") for rec in zero_records]
        xs = [_safe_float(rec.get("identity_law_score")) for rec in zero_records]
        ys = [_safe_float(rec.get("analysis_delta")) for rec in zero_records]
        _scatter(axes[0], xs, ys, colors, xlabel=law_score_label, ylabel="Delta")
        axes[0].set_title("Uncalibrated analysis inference")
        ys2 = [_safe_float(rec.get("law_method_delta")) for rec in zero_records]
        _scatter(axes[1], xs, ys2, colors, xlabel=law_score_label, ylabel="Delta")
        axes[1].set_title("Law-calibrated oracle-target upper bound")
        _caption(fig, "This is the key quadratic-weight-free control. Local-law errors still exist when summaries are poor, but with a linear target those summary differences should not create a persistent pooled-vs-local advantage. Any residual spread is finite-sample inference noise, not structural target-side gain.", y=0.09)
        _save_page(pdf, fig, top=0.84, bottom=0.18)

        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        _page_header(
            fig,
            "When the quadratic weight is positive, do the same local-law errors start to matter for downstream Delta?",
            f"How to read this page: the x-axis is still the configured local-law score, but now the quadratic utility weight makes within-section summary errors relevant to the target. Positive Delta means the local method beats pooling.",
        )
        positive_records = [rec for rec in suite_c if rec.get("lam") in (1.5, 3.0)]
        for ax, lam in zip(axes, [1.5, 3.0]):
            subset = [rec for rec in positive_records if rec.get("lam") == lam]
            colors = [MODE_COLORS.get(str(rec.get("mode")), "#666666") for rec in subset]
            xs = [_safe_float(rec.get("identity_law_score")) for rec in subset]
            ys = [_safe_float(rec.get("analysis_delta")) for rec in subset]
            _scatter(ax, xs, ys, colors, xlabel=law_score_label, ylabel="Delta")
            ax.set_title(f"quadratic weight={lam:g}")
        _caption(fig, "Once the quadratic weight turns on the nonlinear interaction term, summary quality begins to matter to the downstream target. The point of this page is not that local-law error is the only driver of Delta, but that the same summary-quality axis becomes much more consequential after the target makes local structure matter.", y=0.09)
        _save_page(pdf, fig, top=0.84, bottom=0.18)

        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        _page_header(
            fig,
            "Under adaptive labeling, how much extra local-law error does IPW remove?",
            "How to read this page: smaller is better. Each panel averages over tau, quadratic weight, query rates, and seeds within one analysis mode, then compares naive, IPW, and stabilized-IPW learned calibrators across adaptive labeling designs.",
        )
        suite_d = _filter(records, suite="suite_d_ipw_sparse_labels")
        design_rows = [("uniform", "uniform"), ("proxy_priority", "uniform"), ("proxy_adversarial", "risk")]
        method_fields = ["naive_law_score", "ipw_law_score", "stab_law_score"]
        method_labels = ["Naive", "IPW", "Stabilized"]
        for ax, mode in zip(axes, ["aligned", "shift_half"]):
            matrix = np.full((len(design_rows), len(method_fields)), np.nan, dtype=np.float64)
            for i, pair in enumerate(design_rows):
                for j, field in enumerate(method_fields):
                    vals = [
                        _safe_float(rec.get(field))
                        for rec in suite_d
                        if rec.get("mode") == mode
                        and rec.get("law_leaf_query_design") == pair[0]
                        and rec.get("law_internal_query_design") == pair[1]
                    ]
                    matrix[i, j] = _safe_mean(vals)
            im = ax.imshow(matrix, aspect="auto", cmap=LOW_GOOD_CMAP)
            ax.set_title(_mode_label(mode), fontsize=12)
            ax.set_xticks(range(len(method_labels)), method_labels)
            ax.set_yticks(range(len(design_rows)), [_design_label(*pair) for pair in design_rows])
            _annotate_heatmap(ax, matrix, fmt="{:.2f}")
            fig.colorbar(im, ax=ax, shrink=0.80)
        _caption(fig, "This is the sparse-law-label learning page. If adaptive querying creates bias in the law-label sample, the IPW and stabilized-IPW calibrators should outperform naive weighting. The comparison is on exact held-out local-law quality, not yet on a held-out HT/Hajek estimator.", y=0.09)
        _save_page(pdf, fig, top=0.84, bottom=0.18)

        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        _page_header(
            fig,
            "How accurate are HT and Hajek on held-out law estimands?",
            f"How to read this page: bars show absolute error of the held-out population-mean estimators for the configured local-law score. Smaller is better. The left panel uses the uncalibrated summary law scores; the right panel uses the IPW-calibrated summary law scores.",
        )
        metrics = [
            ("C1", "ipw_eval_identity_combined_ht_abs_error", "ipw_eval_identity_combined_hajek_abs_error"),
        ]
        del metrics
        panel_specs = [
            ("Uncalibrated summaries", "identity", [
                ("C1", "c1"),
                ("C3", "c3"),
                ("C2-proxy", "c2_proxy"),
                ("Combined", "combined"),
            ]),
            ("IPW-calibrated summaries", "law_calibrated_ipw", [
                ("C1", "c1"),
                ("C3", "c3"),
                ("C2-proxy", "c2_proxy"),
                ("Combined", "combined"),
            ]),
        ]
        suite_d_payloads = []
        for path in sorted((input_root / "results" / "suite_d_ipw_sparse_labels").rglob("*.json")):
            try:
                suite_d_payloads.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        for ax, (title, policy_name, metric_specs) in zip(axes, panel_specs):
            ht_vals: List[float] = []
            hajek_vals: List[float] = []
            labels: List[str] = []
            for label, metric_name in metric_specs:
                ht_errs = []
                hajek_errs = []
                for payload in suite_d_payloads:
                    item = _nested_get(dict(payload.get("local_law", {})), ("ipw_evaluation", policy_name, metric_name), {})
                    if isinstance(item, dict):
                        ht_errs.append(_safe_float(item.get("ht_abs_error")))
                        hajek_errs.append(_safe_float(item.get("hajek_abs_error")))
                labels.append(label)
                ht_vals.append(_safe_mean(ht_errs))
                hajek_vals.append(_safe_mean(hajek_errs))
            x = np.arange(len(labels), dtype=np.float64)
            ax.bar(x - 0.16, ht_vals, width=0.32, color="#b2182b", label="HT")
            ax.bar(x + 0.16, hajek_vals, width=0.32, color="#1a9850", label="Hajek")
            ax.set_xticks(x, labels)
            ax.set_title(title, fontsize=12)
            ax.set_ylabel("Absolute error")
            ax.grid(alpha=0.25, lw=0.6, axis="y")
        axes[0].legend(frameon=False, fontsize=9)
        _caption(fig, "The held-out evaluation page is separate from the training page. Here the question is whether the design-based estimators recover the population mean of the law metrics themselves. Hajek is the default practical estimator, while HT remains the unbiased reference.", y=0.09)
        _save_page(pdf, fig, top=0.84, bottom=0.18)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        _page_header(
            fig,
            "Once coverage saturates, what actually explains wide intervals and Delta-estimation error?",
            "How to read this page: greener values are better for errors and widths, but higher is better for ESS. The two top panels summarize estimator quality; the two bottom panels summarize the underlying sampling diagnostics.",
        )
        diag_rows = [("uniform", "uniform"), ("proxy_priority", "uniform"), ("proxy_adversarial", "risk")]
        modes2 = ["aligned", "shift_half"]
        matrices = []
        for field in (
            "ipw_eval_law_ipw_combined_hajek_abs_error",
            "ipw_eval_law_ipw_combined_width",
            "ipw_eval_law_ipw_leaf_ess",
            "ipw_eval_law_ipw_leaf_max_weight",
        ):
            mat = np.full((len(diag_rows), len(modes2)), np.nan, dtype=np.float64)
            for i, pair in enumerate(diag_rows):
                for j, mode in enumerate(modes2):
                    vals = [
                        _safe_float(rec.get(field))
                        for rec in suite_d
                        if rec.get("mode") == mode
                        and rec.get("law_leaf_query_design") == pair[0]
                        and rec.get("law_internal_query_design") == pair[1]
                    ]
                    mat[i, j] = _safe_mean(vals)
            matrices.append(mat)
        titles = [
            f"Hajek abs. error\n({law_score_label_multiline})",
            f"Interval width\n({law_score_label_multiline})",
            "Leaf ESS\n(higher is better)",
            "Leaf max weight\n(lower is better)",
        ]
        cmaps = [LOW_GOOD_CMAP, LOW_GOOD_CMAP, HIGH_GOOD_CMAP, LOW_GOOD_CMAP]
        for ax, matrix, title, cmap in zip(axes.flat, matrices, titles, cmaps):
            im = ax.imshow(matrix, aspect="auto", cmap=cmap)
            ax.set_title(title, fontsize=11)
            ax.set_xticks(range(len(modes2)), [_mode_label(m) for m in modes2])
            ax.set_yticks(range(len(diag_rows)), [_design_label(*pair) for pair in diag_rows])
            _annotate_heatmap(ax, matrix, fmt="{:.2f}")
            fig.colorbar(im, ax=ax, shrink=0.78)
        _caption(fig, "Raw coverage was generally conservative, so this page focuses on the statistics that still discriminate settings: point-estimation error, interval width, effective sample size, and large inverse-propensity weights. The diagnostic story should track the estimator story if the held-out design is behaving sensibly.", y=0.08)
        _save_page(pdf, fig, top=0.84, bottom=0.17)

        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        _page_header(
            fig,
            "How much could a learned local-law calibrator help if downstream prediction stayed oracle-targeted?",
            "How to read this page: these are exact held-out auxiliary errors, not learned utility predictions. Smaller values mean the calibrated summary is closer to the analysis-partition oracle target.",
        )
        labels = ["Identity", "Naive", "IPW", "Stabilized"]
        fields = ["identity_aux_abs", "naive_aux_abs", "ipw_aux_abs", "stab_aux_abs"]
        for ax, mode in zip(axes, ["aligned", "shift_half"]):
            vals = [
                _safe_mean(rec.get(field) for rec in suite_c if rec.get("mode") == mode)
                for field in fields
            ]
            ax.bar(np.arange(len(labels)), vals, color=["#666666", "#b2182b", "#2166ac", "#5f4690"])
            ax.set_title(_mode_label(mode), fontsize=12)
            ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
            ax.set_ylabel("Auxiliary oracle-target abs. error")
            ax.grid(alpha=0.25, lw=0.6, axis="y")
        _caption(fig, "This is an auxiliary upper-bound page, not the main predictive claim. It asks whether local-law calibration can improve the topic-mixture summaries themselves before any separate outcome learner is introduced.", y=0.09)
        _save_page(pdf, fig, top=0.84, bottom=0.20)

        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        _page_header(
            fig,
            "Appendix: does the local-law story survive harder topic recovery?",
            "How to read this page: greener cells are better. The x-axis combines the anchor strength and topic-concentration setting; the two panels separate aligned from shifted analysis boundaries.",
        )
        suite_e = _filter(records, suite="suite_e_hardness")
        combos = [(25.0, 0.2), (25.0, 1.0), (10.0, 0.2), (10.0, 1.0)]
        tau_rows = [8.0, 16.0]
        for ax, mode in zip(axes, ["aligned", "shift_half"]):
            matrix = np.full((len(tau_rows), len(combos)), np.nan, dtype=np.float64)
            for i, tau in enumerate(tau_rows):
                for j, combo in enumerate(combos):
                    vals = [
                        _safe_float(rec.get("ipw_law_score"))
                        for rec in suite_e
                        if rec.get("mode") == mode
                        and rec.get("tau") == tau
                        and rec.get("anchor_multiplier") == combo[0]
                        and rec.get("topic_concentration") == combo[1]
                    ]
                    matrix[i, j] = _safe_mean(vals)
            im = ax.imshow(matrix, aspect="auto", cmap=LOW_GOOD_CMAP)
            ax.set_title(_mode_label(mode), fontsize=12)
            ax.set_xticks(range(len(combos)), [f"a={a:g}\ntc={tc:g}" for a, tc in combos])
            ax.set_yticks(range(len(tau_rows)), [f"tau={tau:g}" for tau in tau_rows])
            _annotate_heatmap(ax, matrix, fmt="{:.2f}")
            fig.colorbar(im, ax=ax, shrink=0.80)
        _caption(fig, "This appendix does not try to fully remap the space. It asks a narrower question: do the main local-law patterns survive when topic recovery becomes less favorable because anchors are weaker or topic distributions are flatter?", y=0.09)
        _save_page(pdf, fig, top=0.84, bottom=0.18)

        # === NEW: Law-Stress Classification Page ===
        stress_records = [r for r in records if r.get("stress_bundle_status")]
        if stress_records:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            _page_header(
                fig,
                "Law-Stress Classification: Pass Rates Across (tau, quadratic weight)",
                "How to read: each cell shows the fraction of runs achieving >=10% gain over identity baseline for that law. Green=high pass rate.",
            )
            tau_rows = sorted({_safe_float(r.get("tau")) for r in stress_records if math.isfinite(_safe_float(r.get("tau")))})
            lam_cols = sorted({_safe_float(r.get("lam")) for r in stress_records if math.isfinite(_safe_float(r.get("lam")))})
            for ax_idx, (field, title) in enumerate([
                ("stress_c1_pass", "C1 Pass Rate"),
                ("stress_c2_pass", "C2 Pass Rate"),
                ("stress_c3_pass", "C3 Pass Rate"),
                ("stress_bundle_full_success", "Bundle Full Success Rate"),
            ]):
                ax = axes[ax_idx // 2, ax_idx % 2]
                if tau_rows and lam_cols:
                    matrix = np.full((len(tau_rows), len(lam_cols)), np.nan)
                    for i, tau in enumerate(tau_rows):
                        for j, lam in enumerate(lam_cols):
                            cell = [float(bool(r.get(field))) for r in stress_records
                                    if _safe_float(r.get("tau")) == tau and _safe_float(r.get("lam")) == lam]
                            matrix[i, j] = _safe_mean(cell) if cell else float("nan")
                    im = ax.imshow(matrix, aspect="auto", cmap=HIGH_GOOD_CMAP, vmin=0, vmax=1)
                    ax.set_xticks(range(len(lam_cols)), [f"{l:g}" for l in lam_cols], fontsize=8)
                    ax.set_yticks(range(len(tau_rows)), [f"{t:g}" for t in tau_rows], fontsize=8)
                    ax.set_xlabel("quadratic weight")
                    ax.set_ylabel("tau")
                    _annotate_heatmap(ax, matrix, fmt="{:.0%}")
                    fig.colorbar(im, ax=ax, shrink=0.80)
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(title, fontsize=11)
            _caption(fig, "Law-stress classification uses the shared protocol from law_stress_common: compare learned calibrator (stabilized IPW) against identity baseline. Pass = >=10% relative gain. Bundle = C1+C2+C3 all pass + root ratio <=1.05.", y=0.04)
            _save_page(pdf, fig, top=0.84, bottom=0.12)

        # === NEW: Downstream Mediation Page ===
        mediation_records = [r for r in records if math.isfinite(_safe_float(r.get("stab_law_score")))
                            and math.isfinite(_safe_float(r.get("stab_aux_delta")))]
        if mediation_records:
            fig = plt.figure(figsize=(11, 8.5))
            _page_header(
                fig,
                "Downstream Mediation: Law Improvement vs Utility Improvement",
                f"How to read: each point is one run. X = improvement in {law_score_label.lower()} (identity - stabilized), Y = improvement in downstream utility (Delta). Positive means calibration helped.",
            )
            ax = fig.add_axes([0.12, 0.18, 0.76, 0.62])
            xs = [_safe_float(r.get("identity_law_score", 0)) - _safe_float(r.get("stab_law_score", 0))
                  for r in mediation_records]
            ys = [_safe_float(r.get("stab_aux_delta", 0)) for r in mediation_records]
            colors_list = []
            for r in mediation_records:
                lam = _safe_float(r.get("lam"))
                if lam <= 0.1:
                    colors_list.append("#6c757d")
                elif lam <= 1.0:
                    colors_list.append("#2196F3")
                else:
                    colors_list.append("#d32f2f")
            ax.scatter(xs, ys, c=colors_list, alpha=0.5, s=24, edgecolors="none")
            ax.axhline(0.0, color="#999999", lw=0.8, ls="--")
            ax.axvline(0.0, color="#999999", lw=0.8, ls="--")
            ax.set_xlabel(f"{law_score_label} improvement (identity - calibrated)")
            ax.set_ylabel("Downstream Utility Delta")
            ax.grid(alpha=0.2, lw=0.5)
            # Legend for quadratic-weight ranges
            from matplotlib.lines import Line2D
            legend_handles = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#6c757d", markersize=8, label="lam<=0.1"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3", markersize=8, label="0.1<lam<=1"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#d32f2f", markersize=8, label="lam>1"),
            ]
            ax.legend(handles=legend_handles, fontsize=9, loc="upper left")
            _caption(fig, "When the quadratic weight is large, local-law improvements translate to downstream utility gains. When it is small (grey), the target gap is near zero regardless of summary quality, consistent with the main LDA report.", y=0.06)
            _save_page(pdf, fig, top=0.84, bottom=0.14)

        # === NEW: Exact-Family Counterexample Page ===
        exact_records = [r for r in records if r.get("exact_family")]
        if exact_records:
            fig = plt.figure(figsize=(11, 8.5))
            _page_header(
                fig,
                "Exact Counterexample Families: Designed Law Failures",
                "How to read: each bar group shows C1/C2/C3 for a calibrator family designed to break specific laws.",
            )
            ax = fig.add_axes([0.12, 0.22, 0.80, 0.58])
            families = ["oracle", "scrambled_topics", "uniform_prior", "adversarial_merge"]
            x_pos = np.arange(len(families))
            width = 0.25
            for metric_idx, (metric, label, color) in enumerate([
                ("c1", "C1", "#1b9e77"),
                ("c3", "C3", "#d95f02"),
                ("c2", "C2-proxy", "#7570b3"),
            ]):
                vals = []
                for fam in families:
                    fam_key = f"exact_{fam}"
                    fam_records = [r for r in exact_records if r.get("exact_family") == fam]
                    field = f"{fam_key}_{metric}"
                    fam_vals = [_safe_float(r.get(field)) for r in fam_records if math.isfinite(_safe_float(r.get(field)))]
                    vals.append(_safe_mean(fam_vals) if fam_vals else 0.0)
                ax.bar(x_pos + (metric_idx - 1) * width, vals, width, label=label, color=color, alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f.replace("_", "\n") for f in families], fontsize=10)
            ax.set_ylabel("Mean Error (L1)")
            ax.legend(fontsize=10)
            ax.grid(axis="y", alpha=0.25, lw=0.5)
            _caption(fig, "oracle = identity (baseline). scrambled_topics permutes topic indices (breaks C1). uniform_prior always returns 1/K (breaks C1, C2 is zero). adversarial_merge projects onto half the topics (breaks C3 on cross-boundary merges).", y=0.06)
            _save_page(pdf, fig, top=0.84, bottom=0.14)

        # === NEW: Summary Table Page ===
        if stress_records:
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_axes([0.06, 0.18, 0.88, 0.70])
            ax.axis("off")
            _page_header(
                fig,
                "Law-Stress Summary Table",
                "How to read: aggregated pass rates across all runs, grouped by law set.",
            )
            # Compute summary stats
            packages = sorted({str(r.get("law_set_id", "unknown")) for r in stress_records})
            header = f"{'Law set':<20s} {'N':>5s} {'C1%':>6s} {'C2%':>6s} {'C3%':>6s} {'Bundle%':>8s}"
            lines = [header, "-" * len(header)]
            for pkg in packages:
                pkg_recs = [r for r in stress_records if str(r.get("law_set_id", "unknown")) == pkg]
                n = len(pkg_recs)
                c1 = _safe_mean([float(bool(r.get("stress_c1_pass"))) for r in pkg_recs])
                c2 = _safe_mean([float(bool(r.get("stress_c2_pass"))) for r in pkg_recs])
                c3 = _safe_mean([float(bool(r.get("stress_c3_pass"))) for r in pkg_recs])
                bun = _safe_mean([float(bool(r.get("stress_bundle_full_success"))) for r in pkg_recs])
                lines.append(f"{pkg:<20s} {n:>5d} {c1:>5.0%} {c2:>5.0%} {c3:>5.0%} {bun:>7.0%}")
            y = 0.90
            for line in lines:
                ax.text(0.05, y, line, fontsize=11, family="monospace", va="top")
                y -= 0.04
            _caption(fig, "Pass rates use the shared law_stress_common protocol (>=10% relative gain over identity baseline). This table matches the Markov report's summary structure for cross-DGP comparison.", y=0.06)
            _save_page(pdf, fig, top=0.84, bottom=0.14)

        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_axes([0.06, 0.18, 0.88, 0.70])
        ax.axis("off")
        _page_header(
            fig,
            "What should an external reader take away from this local-law companion?",
            "How to read this page: these are the synthesis claims that survive after the exact-control, learnability, mismatch, and held-out-design pages are taken together.",
        )
        bullets = [
            "Token-weighted local laws are measurable on the same Stage 3 world. Oracle summaries drive C1 and C3 to zero, and the inferred-law scores are effectively unchanged across quadratic-weight settings on a fixed world.",
            "Boundary mismatch worsens local-law error before the utility target enters. That is the mechanism layer: summary quality degrades first, and only then can the quadratic weight make those degradations matter downstream.",
            "Sparse adaptive law labels are learnable enough to improve calibration. IPW typically helps relative to naive weighting, but it does not magically solve every hard labeling design or every mismatch condition.",
            "Held-out Hajek behaves sensibly once the normalization bug is fixed. The design-validity pages now line up with ESS and max-weight diagnostics instead of producing uninterpretable clouds.",
        ]
        y = 0.92
        for idx, bullet in enumerate(bullets, start=1):
            ax.text(0.0, y, f"{idx}.", fontsize=13, fontweight="bold", va="top", ha="left")
            ax.text(0.05, y, textwrap.fill(bullet, width=92), fontsize=12, va="top", ha="left", linespacing=1.35)
            y -= 0.20
        _caption(fig, "This companion is intentionally mechanism-first. It does not replace the main realism report; it explains more precisely why the realism results move when boundaries, token weights, and sparse labels become imperfect.", y=0.10)
        _save_page(pdf, fig, top=0.89, bottom=0.14)

    _write_markdown(
        md_path,
        snapshot_label=str(args.snapshot_label),
        summary=summary,
        unified_core=unified_core,
    )
    assert_public_contract_clean(summary, surface=str(summary_path))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote_pdf | {pdf_path}")
    print(f"wrote_md | {md_path}")
    print(f"wrote_summary | {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
