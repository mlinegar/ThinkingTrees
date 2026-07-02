#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.report.pdf_utils import write_image_page, write_text_page
from src.ctreepo.sim.core.markov_v3_row_contract import is_headline_contract_status
from src.ctreepo.sim.util import safe_float
from src.ctreepo.sim.core.markov_parity_grid_io import (
    FULL_LOCAL_LAWS_TOPOLOGY_STUDY_AXIS,
    PARITY_MANIFEST_NAME,
    PARITY_SUMMARY_NAME,
    RECIPE_DISPLAY_NAMES,
    TOPOLOGY_STUDY_AXES,
    UNIFIED_G_TOPOLOGY_STUDY_AXIS,
)

QUALITY_METRICS: Sequence[tuple[str, str, str]] = (
    ("test_root_mae_mean", "Root MAE", "root"),
    ("test_leaf_mae_mean", "Leaf MAE", "leaf"),
    ("test_merge_mae_mean", "Merge MAE", "merge"),
)

RUNTIME_METRICS: Sequence[tuple[str, str, str, str]] = (
    ("elapsed_s", "Elapsed Time (s)", "elapsed_s", "min"),
    ("gpu_reserved_mem_peak_gb", "Peak Reserved GPU Memory (GB)", "gpu_reserved_mem_peak_gb", "min"),
    ("resident_store_hits", "Resident Store Hits", "resident_store_hits", "max"),
    ("steady_state_h2d_bytes", "Steady-State H2D Bytes", "steady_state_h2d_bytes", "min"),
)

FAMILY_PALETTE: Mapping[str, str] = {
    "official_fno": "#0f766e",
    "official_fno_sumlen": "#dc2626",
    "tree_neural": "#2563eb",
}

FAMILY_LABELS: Mapping[str, str] = {
    "official_fno": "FNO",
    "official_fno_sumlen": "FNO sumlen",
    "tree_neural": "Tree",
}

RECIPE_STYLES: Mapping[str, Mapping[str, str]] = {
    "matched_root": {"marker": "o", "linestyle": "-", "hatch": ""},
    "fairfno_matched_root": {"marker": "s", "linestyle": "-", "hatch": "//"},
    "historical_replay": {"marker": "^", "linestyle": "--", "hatch": "xx"},
    "optimization_fairness": {"marker": "D", "linestyle": "-.", "hatch": ".."},
    "capacity_fairness": {"marker": "v", "linestyle": ":", "hatch": "\\\\"},
    "exact_collapse_candidate": {"marker": "*", "linestyle": "None", "hatch": "**"},
    "exact_collapse_runtime_match": {"marker": "D", "linestyle": "--", "hatch": "++"},
    "exact_collapse_legacy_control": {"marker": "X", "linestyle": "-.", "hatch": "xx"},
    "fno_baseline": {"marker": "o", "linestyle": "--", "hatch": ""},
}

SERIES_ORDER: Mapping[str, int] = {
    "official_fno": 0,
    "official_fno_sumlen": 1,
    "matched_root": 2,
    "fairfno_matched_root": 3,
    "historical_replay": 4,
    "optimization_fairness": 5,
    "capacity_fairness": 6,
    "exact_collapse_candidate": 7,
    "exact_collapse_runtime_match": 8,
    "exact_collapse_legacy_control": 9,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained Markov parity report using only artifacts inside the specified simulation root(s)."
    )
    parser.add_argument(
        "--simulation-root",
        type=Path,
        action="append",
        required=True,
        help="Repeat to aggregate multiple finished parity roots into one report.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


_safe_float = safe_float


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _mean(values: Iterable[float]) -> float:
    seq = [float(value) for value in values if _is_finite(value)]
    if not seq:
        return float("nan")
    return float(sum(seq) / len(seq))


def _slugify(text: str) -> str:
    chars: List[str] = []
    previous_sep = False
    for ch in str(text):
        if ch.isalnum():
            chars.append(ch.lower())
            previous_sep = False
        else:
            if not previous_sep:
                chars.append("_")
            previous_sep = True
    return "".join(chars).strip("_") or "figure"


def _path_is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _recipe_display(recipe_id: str) -> str:
    key = str(recipe_id or "").strip()
    if key == "exact_collapse_candidate":
        return "Exact collapse"
    if key == "exact_collapse_runtime_match":
        return "Exact collapse runtime-match"
    if key == "exact_collapse_legacy_control":
        return "Exact collapse legacy"
    return str(RECIPE_DISPLAY_NAMES.get(key, key))


def _leaf_supervision_kind(row: Mapping[str, Any]) -> str:
    return str(row.get("leaf_supervision_kind", "") or "").strip()


def _tree_local_weighting_mode(row: Mapping[str, Any]) -> str:
    return str(row.get("tree_local_weighting_mode", "") or "").strip()


def _job_name(row: Mapping[str, Any]) -> str:
    return str(row.get("job_name", "") or "")


def _is_local_recipe(row: Mapping[str, Any]) -> bool:
    recipe_id = str(row.get("recipe_id", "") or "").strip()
    return recipe_id.startswith("r") and "_local_" in recipe_id


def _derived_total_local_weight(row: Mapping[str, Any]) -> float:
    job_name = _job_name(row)
    for weight_text, weight_value in (("lw10", 0.10), ("lw25", 0.25), ("lw50", 0.50)):
        if f"__{weight_text}__" in job_name:
            return float(weight_value)
    values = [
        _safe_float(row.get("local_law_c1_weight")),
        _safe_float(row.get("local_law_c2_weight")),
        _safe_float(row.get("local_law_c3_weight")),
    ]
    if not all(_is_finite(value) for value in values):
        return float("nan")
    return float(sum(values))


def _derived_c1_c3_ratio(row: Mapping[str, Any]) -> float:
    job_name = _job_name(row)
    for ratio_text, ratio_value in (("c1_1", 1.0), ("c1_2", 2.0)):
        if f"__{ratio_text}__" in job_name:
            return float(ratio_value)
    c1 = _safe_float(row.get("local_law_c1_weight"))
    c3 = _safe_float(row.get("local_law_c3_weight"))
    if not _is_finite(c1) or not _is_finite(c3) or c3 <= 0.0:
        return float("nan")
    return float(c1 / c3)


def _is_tree_local_variant(row: Mapping[str, Any]) -> bool:
    if str(row.get("baseline_family", "") or "") != "tree_neural":
        return False
    if str(row.get("claim_level", "") or "") == "exact_collapse_candidate":
        return False
    if not _is_local_recipe(row):
        return False
    return any(
        bool(str(value or "").strip())
        for value in (
            row.get("leaf_supervision_kind"),
            row.get("internal_supervision_kind"),
            row.get("tree_local_weighting_mode"),
        )
    )


def _short_supervision_label(value: str) -> str:
    mapping = {
        "count_only": "count-only",
        "bounded_full_sketch": "bounded sketch",
        "full_sketch": "legacy full sketch",
    }
    return str(mapping.get(str(value or "").strip(), str(value or "").strip()))


def _short_weighting_label(value: str) -> str:
    mapping = {
        "subset_mean": "subset mean",
        "fixed_k_hajek": "Hajek",
    }
    return str(mapping.get(str(value or "").strip(), str(value or "").strip()))


def _tree_variant_machine_suffix(row: Mapping[str, Any]) -> str:
    if not _is_tree_local_variant(row):
        return ""
    parts: List[str] = []
    leaf_kind = _leaf_supervision_kind(row)
    weighting_mode = _tree_local_weighting_mode(row)
    if leaf_kind:
        parts.append(leaf_kind)
    if weighting_mode:
        parts.append(weighting_mode)
    total_local_weight = _derived_total_local_weight(row)
    c1_c3_ratio = _derived_c1_c3_ratio(row)
    if _is_finite(total_local_weight) and abs(total_local_weight - 0.25) > 1e-9:
        parts.append(f"lw{int(round(total_local_weight * 100.0))}")
    if _is_finite(c1_c3_ratio) and abs(c1_c3_ratio - 1.0) > 1e-9:
        ratio_text = f"{c1_c3_ratio:.2f}".rstrip("0").rstrip(".")
        parts.append(f"c1x{ratio_text}")
    return "__".join(parts)


def _tree_variant_human_suffix(row: Mapping[str, Any]) -> str:
    if not _is_tree_local_variant(row):
        return ""
    parts: List[str] = []
    leaf_kind = _leaf_supervision_kind(row)
    weighting_mode = _tree_local_weighting_mode(row)
    if leaf_kind:
        parts.append(_short_supervision_label(leaf_kind))
    if weighting_mode:
        parts.append(_short_weighting_label(weighting_mode))
    total_local_weight = _derived_total_local_weight(row)
    c1_c3_ratio = _derived_c1_c3_ratio(row)
    extras: List[str] = []
    if _is_finite(total_local_weight) and abs(total_local_weight - 0.25) > 1e-9:
        extras.append(f"lw={total_local_weight:.2f}")
    if _is_finite(c1_c3_ratio) and abs(c1_c3_ratio - 1.0) > 1e-9:
        ratio_text = f"{c1_c3_ratio:.2f}".rstrip("0").rstrip(".")
        extras.append(f"c1:c3={ratio_text}:1")
    if extras:
        parts.append(", ".join(extras))
    return " | ".join(parts)


def _series_key(row: Mapping[str, Any]) -> str:
    if str(row.get("claim_level", "")) == "exact_collapse_candidate":
        return "exact_collapse_candidate"
    family = str(row.get("baseline_family", "") or "")
    if family in {"official_fno", "official_fno_sumlen"}:
        return family
    recipe_id = str(row.get("recipe_id", "") or "")
    suffix = _tree_variant_machine_suffix(row)
    if suffix:
        return f"{recipe_id}__{suffix}"
    return recipe_id


def _comparison_label(row: Mapping[str, Any]) -> str:
    if str(row.get("claim_level", "")) == "exact_collapse_candidate":
        return "Exact collapse"
    family = str(row.get("baseline_family", "") or "")
    if family in FAMILY_LABELS and family != "tree_neural":
        return str(FAMILY_LABELS[family])
    recipe_label = _recipe_display(str(row.get("recipe_id", "") or ""))
    suffix = _tree_variant_human_suffix(row)
    if suffix:
        return f"{recipe_label} [{suffix}]"
    return recipe_label


def _is_exact_collapse_repair_row(row: Mapping[str, Any]) -> bool:
    job_name = str(row.get("job_name", "") or "")
    if "exact_collapse_repair_arm_" not in job_name:
        return False
    recipe_id = str(row.get("recipe_id", "") or "")
    if recipe_id in {"exact_collapse_candidate", "exact_collapse_runtime_match", "exact_collapse_legacy_control"}:
        return True
    if str(row.get("baseline_family", "") or "") != "official_fno":
        return False
    return "exact_collapse_repair_arm_official_fno" in job_name


def _exact_collapse_repair_series_key(row: Mapping[str, Any]) -> str:
    recipe_id = str(row.get("recipe_id", "") or "")
    if recipe_id in {"exact_collapse_candidate", "exact_collapse_runtime_match", "exact_collapse_legacy_control"}:
        return recipe_id
    return "official_fno"


def _exact_collapse_repair_label(row: Mapping[str, Any]) -> str:
    key = _exact_collapse_repair_series_key(row)
    labels = {
        "official_fno": "Official FNO",
        "exact_collapse_candidate": "One-tree candidate",
        "exact_collapse_runtime_match": "Runtime-match control",
        "exact_collapse_legacy_control": "Legacy control",
    }
    return str(labels.get(key, key))


def _is_full_local_laws_topology_row(row: Mapping[str, Any]) -> bool:
    return str(row.get("study_axis", "") or "") in set(TOPOLOGY_STUDY_AXES)


_TOPOLOGY_LEAF_TOKEN_HINT_RE = re.compile(r"leaf[_]?(\d+)")


def _topology_fixed_leaf_tokens(row: Mapping[str, Any]) -> int:
    raw_value = int(row.get("fixed_leaf_tokens", 0) or 0)
    hint_fields = (
        "axis_value",
        "config_label",
        "job_name",
        "recipe_label",
        "comparison_label",
    )
    hinted_values: set[int] = set()
    for field_name in hint_fields:
        field_value = str(row.get(field_name, "") or "")
        for match in _TOPOLOGY_LEAF_TOKEN_HINT_RE.finditer(field_value):
            try:
                hinted_values.add(int(match.group(1)))
            except Exception:
                continue
    if len(hinted_values) == 1:
        hinted_value = next(iter(hinted_values))
        if hinted_value > 0 and hinted_value != raw_value:
            return hinted_value
    return raw_value


def _topology_leaf_count(row: Mapping[str, Any], *, assumed_doc_tokens: int) -> int:
    fixed_leaf_tokens = int(_topology_fixed_leaf_tokens(row) or 0)
    if assumed_doc_tokens <= 0 or fixed_leaf_tokens <= 0:
        return 0
    if assumed_doc_tokens % fixed_leaf_tokens != 0:
        return 0
    return int(assumed_doc_tokens // fixed_leaf_tokens)


def _topology_curve_series_key(row: Mapping[str, Any]) -> str:
    if str(row.get("claim_level", "")) == "exact_collapse_candidate":
        return "one_tree_anchor"
    family = str(row.get("baseline_family", "") or "")
    if family == "official_fno":
        return "official_fno"
    study_axis = str(row.get("study_axis", "") or "")
    locked_config = str(row.get("locked_tree_neural_config_label", "") or "")
    if (
        study_axis == UNIFIED_G_TOPOLOGY_STUDY_AXIS
        or locked_config == "unified_g_full_local_laws_v1"
    ):
        return "tree_unified_g"
    if (
        study_axis == FULL_LOCAL_LAWS_TOPOLOGY_STUDY_AXIS
        or locked_config == "common_factorized_sketch_v1"
    ):
        return "tree_legacy"
    if family == "tree_neural":
        return f"tree_other::{locked_config or str(row.get('recipe_id', '') or 'tree_neural')}"
    return family or "other"


def _topology_curve_label(series_key: str) -> str:
    labels = {
        "official_fno": "Official FNO anchor",
        "one_tree_anchor": "One-tree repaired anchor",
        "tree_unified_g": "Tree (unified_g)",
        "tree_legacy": "Tree (legacy)",
    }
    if series_key in labels:
        return str(labels[series_key])
    if series_key.startswith("tree_other::"):
        return f"Tree ({series_key.split('::', 1)[1]})"
    return str(series_key)


def _topology_curve_style(series_key: str) -> Dict[str, str]:
    styles: Dict[str, Dict[str, str]] = {
        "official_fno": {"color": FAMILY_PALETTE["official_fno"], "marker": "D", "linestyle": "--"},
        "one_tree_anchor": {"color": "#111827", "marker": "*", "linestyle": "None"},
        "tree_unified_g": {"color": FAMILY_PALETTE["tree_neural"], "marker": "o", "linestyle": "-"},
        "tree_legacy": {"color": "#b45309", "marker": "s", "linestyle": "-."},
    }
    if series_key in styles:
        return dict(styles[series_key])
    return {"color": "#475569", "marker": "^", "linestyle": ":"}


def _timestamp_sort_key(value: Any) -> tuple[float, str]:
    text = str(value or "").strip()
    if not text:
        return (float("-inf"), "")
    normalized = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (float(dt.timestamp()), text)
    except Exception:
        return (float("-inf"), text)


def _style_for_row(row: Mapping[str, Any]) -> Dict[str, str]:
    series_key = _series_key(row)
    family = str(row.get("baseline_family", "") or "")
    style = dict(RECIPE_STYLES.get(series_key, {}))
    style.setdefault("marker", "o")
    style.setdefault("linestyle", "-")
    style.setdefault("hatch", "")
    if family in {"official_fno", "official_fno_sumlen"}:
        style["color"] = str(FAMILY_PALETTE[family])
    else:
        style["color"] = str(FAMILY_PALETTE["tree_neural"])
    return style


def _series_rank(row: Mapping[str, Any]) -> int:
    series_key = _series_key(row)
    base_series_key = str(series_key).split("__", 1)[0]
    return int(SERIES_ORDER.get(base_series_key, 99))


def _cell_key(row: Mapping[str, Any]) -> str:
    return "::".join(
        [
            str(row.get("scope_label", "") or ""),
            str(int(row.get("train_doc_count", 0) or 0)),
            str(row.get("claim_level", "") or ""),
            str(_series_key(row)),
            str(int(row.get("fixed_leaf_tokens", 0) or 0)),
        ]
    )


def _row_brief(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "job_name": str(row.get("job_name", "") or ""),
        "state": str(row.get("state", "") or ""),
        "contract_status": str(row.get("contract_status", "") or ""),
        "tuning_stage": str(row.get("tuning_stage", "") or ""),
        "study_axis": str(row.get("study_axis", "") or ""),
        "axis_value": str(row.get("axis_value", "") or ""),
        "scope_label": str(row.get("scope_label", "") or ""),
        "train_doc_count": int(row.get("train_doc_count", 0) or 0),
        "claim_level": str(row.get("claim_level", "") or ""),
        "baseline_family": str(row.get("baseline_family", "") or ""),
        "recipe_id": str(row.get("recipe_id", "") or ""),
        "recipe_label": str(row.get("recipe_label", "") or ""),
        "comparison_label": str(row.get("comparison_label", "") or ""),
        "series_key": str(row.get("series_key", "") or ""),
        "fixed_leaf_tokens": int(row.get("fixed_leaf_tokens", 0) or 0),
        "strict_collapse_pass": bool(row.get("strict_collapse_pass", False)),
        "config_diff_field_count": len(dict(row.get("config_diff_vs_official_fno") or {})),
        "reference_bundle_source": str(row.get("reference_bundle_source", "") or ""),
        "train_prefix_counts": [int(v) for v in list(row.get("train_prefix_counts") or [])],
        "test_root_mae_mean": _safe_float(row.get("test_root_mae_mean")),
        "test_leaf_mae_mean": _safe_float(row.get("test_leaf_mae_mean")),
        "test_merge_mae_mean": _safe_float(row.get("test_merge_mae_mean")),
        "val_root_mae_mean": _safe_float(row.get("val_root_mae_mean")),
        "elapsed_s": _safe_float(row.get("elapsed_s")),
        "gpu_reserved_mem_peak_gb": _safe_float(row.get("gpu_reserved_mem_peak_gb")),
        "resident_store_hits": _safe_float(row.get("resident_store_hits")),
        "steady_state_h2d_bytes": _safe_float(row.get("steady_state_h2d_bytes")),
        "runtime_data_mode": str(row.get("runtime_data_mode", "") or ""),
        "runtime_bucket_mode": str(row.get("runtime_bucket_mode", "") or ""),
        "leaf_supervision_kind": str(row.get("leaf_supervision_kind", "") or ""),
        "internal_supervision_kind": str(row.get("internal_supervision_kind", "") or ""),
        "tree_local_weighting_mode": str(row.get("tree_local_weighting_mode", "") or ""),
        "local_law_c1_weight": _safe_float(row.get("local_law_c1_weight")),
        "local_law_c2_weight": _safe_float(row.get("local_law_c2_weight")),
        "local_law_c3_weight": _safe_float(row.get("local_law_c3_weight")),
        "optimization_root_weight": _safe_float(row.get("optimization_root_weight")),
        "source_simulation_root": str(row.get("source_simulation_root", "") or ""),
        "source_simulation_generated_at": str(row.get("source_simulation_generated_at", "") or ""),
        "source_simulation_order": int(row.get("source_simulation_order", 0) or 0),
    }


def _resolve_job_dir(row: Mapping[str, Any], simulation_root: Path) -> Path | None:
    candidates: List[Path] = []
    job_output_dir = str(row.get("job_output_dir", "") or "")
    if job_output_dir:
        candidates.append(Path(job_output_dir))
    source_summary_json = str(row.get("source_summary_json", "") or "")
    if source_summary_json:
        candidates.append(Path(source_summary_json).parent)
    job_name = str(row.get("job_name", "") or "")
    if job_name:
        candidates.append(simulation_root / "jobs" / job_name)
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if _path_is_within_root(resolved, simulation_root):
            if resolved.exists():
                return resolved
            if resolved.parent.exists():
                return resolved
    return None


def _load_run_payload(job_dir: Path | None, simulation_root: Path) -> tuple[Mapping[str, Any], str]:
    if job_dir is None or not job_dir.exists() or not _path_is_within_root(job_dir, simulation_root):
        return {}, ""
    run_paths = sorted((job_dir / "runs").glob("*.json"))
    for run_path in run_paths:
        resolved = run_path.resolve()
        if _path_is_within_root(resolved, simulation_root):
            return dict(_load_json(resolved)), str(resolved)
    summary_path = job_dir / "summary.json"
    if summary_path.exists() and _path_is_within_root(summary_path.resolve(), simulation_root):
        summary = dict(_load_json(summary_path))
        runs = [dict(item or {}) for item in list(summary.get("runs") or [])]
        if runs:
            return runs[0], str(summary_path.resolve())
    return {}, ""


def _load_job_summary_payload(job_dir: Path | None, simulation_root: Path) -> tuple[Mapping[str, Any], str]:
    if job_dir is None or not job_dir.exists() or not _path_is_within_root(job_dir, simulation_root):
        return {}, ""
    summary_path = job_dir / "summary.json"
    if not summary_path.exists() or not _path_is_within_root(summary_path.resolve(), simulation_root):
        return {}, ""
    return dict(_load_json(summary_path)), str(summary_path.resolve())


def _best_summary_metric_row(summary_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = [
        dict(item or {})
        for item in list(summary_payload.get("aggregate_rows") or [])
    ] + [
        dict(item or {})
        for item in list(summary_payload.get("heatmap_rows") or [])
    ]
    for candidate in candidates:
        if any(_is_finite(candidate.get(metric_key)) for metric_key, _, _ in QUALITY_METRICS):
            return candidate
    return {}


def _normalize_row(row: Mapping[str, Any], payload: Mapping[str, Any]) -> Dict[str, Any]:
    simulation_root = Path(
        str(row.get("source_simulation_root", "") or payload.get("simulation_root", ""))
    ).resolve()
    job_dir = _resolve_job_dir(row, simulation_root)
    job_summary_payload, job_summary_path = _load_job_summary_payload(job_dir, simulation_root)
    summary_metric_row = _best_summary_metric_row(job_summary_payload)
    run_payload, run_payload_path = _load_run_payload(job_dir, simulation_root)
    runtime_efficiency = dict(run_payload.get("runtime_efficiency") or {})
    bundle_source = str(run_payload.get("bundle_source", "") or "")
    reference_bundle_source = str(row.get("reference_bundle_source", "") or "")
    if not reference_bundle_source and bundle_source:
        reference_bundle_source = bundle_source.split("::", 1)[0]
    state = str(row.get("state", "") or "")
    if state != "completed" and summary_metric_row:
        state = "completed"
    elif not state:
        state = "completed" if summary_metric_row else ""
    normalized = dict(row or {})
    normalized.update(
        {
            "state": state,
            "recipe_label": _recipe_display(str(row.get("recipe_id", "") or "")),
            "comparison_label": _comparison_label(row),
            "job_output_dir": str(job_dir) if job_dir is not None else "",
            "run_json_path": run_payload_path,
            "job_summary_path": job_summary_path,
            "elapsed_s": _safe_float(run_payload.get("elapsed_s")),
            "resident_store_hits": _safe_float(runtime_efficiency.get("resident_store_hits")),
            "resident_store_misses": _safe_float(runtime_efficiency.get("resident_store_misses")),
            "steady_state_h2d_bytes": _safe_float(runtime_efficiency.get("steady_state_h2d_bytes")),
            "gpu_reserved_mem_peak_gb": _safe_float(runtime_efficiency.get("gpu_reserved_mem_peak_gb")),
            "gpu_allocated_mem_peak_gb": _safe_float(runtime_efficiency.get("gpu_allocated_mem_peak_gb")),
            "runtime_data_mode": str(runtime_efficiency.get("runtime_data_mode", "") or ""),
            "runtime_bucket_mode": str(runtime_efficiency.get("runtime_bucket_mode", "") or ""),
            "runtime_workers_per_mig": _safe_float(runtime_efficiency.get("runtime_workers_per_mig")),
            "test_root_mae_mean": _safe_float(
                row.get("test_root_mae_mean")
                if _is_finite(row.get("test_root_mae_mean"))
                else summary_metric_row.get("test_root_mae_mean")
            ),
            "test_leaf_mae_mean": _safe_float(
                row.get("test_leaf_mae_mean")
                if _is_finite(row.get("test_leaf_mae_mean"))
                else summary_metric_row.get("test_leaf_mae_mean")
            ),
            "test_merge_mae_mean": _safe_float(
                row.get("test_merge_mae_mean")
                if _is_finite(row.get("test_merge_mae_mean"))
                else summary_metric_row.get("test_merge_mae_mean")
            ),
            "val_root_mae_mean": _safe_float(
                row.get("val_root_mae_mean")
                if _is_finite(row.get("val_root_mae_mean"))
                else summary_metric_row.get("val_root_mae_mean")
            ),
            "reference_bundle_source": reference_bundle_source,
            "one_leaf_target": bool(
                int(row.get("fixed_leaf_tokens", 0) or 0)
                == int(payload.get("one_leaf_target_fixed_leaf_tokens", 0) or 0)
            ),
            "source_simulation_root": str(
                row.get("source_simulation_root", "") or simulation_root
            ),
            "source_simulation_generated_at": str(
                row.get("source_simulation_generated_at", "") or payload.get("generated_at", "") or ""
            ),
            "source_simulation_order": int(row.get("source_simulation_order", 0) or 0),
            "series_key": _series_key(row),
            "cell_key": "",
        }
    )
    normalized["cell_key"] = _cell_key(normalized)
    return normalized


def _load_parity_payload(root: Path) -> Dict[str, Any]:
    simulation_root = Path(root).expanduser().resolve()
    manifest_path = simulation_root / PARITY_MANIFEST_NAME
    summary_path = simulation_root / PARITY_SUMMARY_NAME
    scheduler_path = simulation_root / "scheduler_status.json"
    manifest = dict(_load_json(manifest_path))
    summary = dict(_load_json(summary_path))
    scheduler = dict(_load_json(scheduler_path)) if scheduler_path.exists() else {}
    rows = [dict(row or {}) for row in list(summary.get("rows") or [])]
    return {
        "simulation_root": str(simulation_root),
        "source_simulation_root": str(simulation_root),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "scheduler_path": str(scheduler_path),
        "generated_at": str(summary.get("generated_at") or scheduler.get("generated_at") or ""),
        "state": str(summary.get("state") or scheduler.get("state") or ""),
        "evidence_status": str(summary.get("evidence_status") or ""),
        "assumed_doc_tokens": int(summary.get("assumed_doc_tokens") or manifest.get("assumed_doc_tokens") or 0),
        "one_leaf_target_fixed_leaf_tokens": int(
            summary.get("one_leaf_target_fixed_leaf_tokens")
            or manifest.get("one_leaf_target_fixed_leaf_tokens")
            or 0
        ),
        "items_total": int(summary.get("items_total") or scheduler.get("items_total") or len(rows)),
        "completed_items": int(summary.get("completed_items") or scheduler.get("completed_items") or 0),
        "failed_items": int(summary.get("failed_items") or scheduler.get("failed_items") or 0),
        "active_items": int(summary.get("active_items") or scheduler.get("active_items") or 0),
        "pending_items": int(summary.get("pending_items") or scheduler.get("pending_items") or 0),
        "rows": rows,
    }


def _combine_states(states: Sequence[str]) -> str:
    values = [str(state or "").strip() for state in states if str(state or "").strip()]
    if not values:
        return ""
    if any(value == "running" for value in values):
        return "running"
    if any(value == "failed" for value in values):
        return "failed"
    if all(value == "completed" for value in values):
        return "completed"
    return values[0]


def _load_parity_payloads(roots: Sequence[Path]) -> Dict[str, Any]:
    payloads = [_load_parity_payload(root) for root in roots]
    if len(payloads) == 1:
        return payloads[0]
    assumed_doc_tokens = {
        int(payload.get("assumed_doc_tokens", 0) or 0)
        for payload in payloads
    }
    one_leaf_targets = {
        int(payload.get("one_leaf_target_fixed_leaf_tokens", 0) or 0)
        for payload in payloads
    }
    if len(assumed_doc_tokens) > 1:
        raise ValueError(
            f"incompatible assumed_doc_tokens across roots: {sorted(assumed_doc_tokens)}"
        )
    if len(one_leaf_targets) > 1:
        raise ValueError(
            "incompatible one_leaf_target_fixed_leaf_tokens across roots: "
            f"{sorted(one_leaf_targets)}"
        )
    combined_rows: List[Dict[str, Any]] = []
    for payload_index, payload in enumerate(payloads):
        source_root = str(payload.get("simulation_root", "") or "")
        source_generated_at = str(payload.get("generated_at", "") or "")
        for row in list(payload.get("rows") or []):
            row_payload = dict(row or {})
            row_payload["source_simulation_root"] = source_root
            row_payload["source_simulation_generated_at"] = source_generated_at
            row_payload["source_simulation_order"] = int(payload_index)
            combined_rows.append(row_payload)
    return {
        "simulation_root": "",
        "simulation_roots": [str(payload.get("simulation_root", "") or "") for payload in payloads],
        "manifest_paths": [str(payload.get("manifest_path", "") or "") for payload in payloads],
        "summary_paths": [str(payload.get("summary_path", "") or "") for payload in payloads],
        "scheduler_paths": [str(payload.get("scheduler_path", "") or "") for payload in payloads],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state": _combine_states([str(payload.get("state", "") or "") for payload in payloads]),
        "evidence_status": (
            str(payloads[0].get("evidence_status", "") or "")
            if len({str(payload.get("evidence_status", "") or "") for payload in payloads}) == 1
            else "aggregated"
        ),
        "assumed_doc_tokens": next(iter(assumed_doc_tokens), 0),
        "one_leaf_target_fixed_leaf_tokens": next(iter(one_leaf_targets), 0),
        "items_total": sum(int(payload.get("items_total", 0) or 0) for payload in payloads),
        "completed_items": sum(int(payload.get("completed_items", 0) or 0) for payload in payloads),
        "failed_items": sum(int(payload.get("failed_items", 0) or 0) for payload in payloads),
        "active_items": sum(int(payload.get("active_items", 0) or 0) for payload in payloads),
        "pending_items": sum(int(payload.get("pending_items", 0) or 0) for payload in payloads),
        "rows": combined_rows,
    }


def _completed_rows(rows: Sequence[Mapping[str, Any]], *, claim_level: str | None = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row or {})
        if claim_level is not None and str(row.get("claim_level", "")) != str(claim_level):
            continue
        if str(row.get("state", "")) != "completed":
            continue
        contract_status = str(row.get("contract_status", "") or "").strip()
        if contract_status and not is_headline_contract_status(contract_status):
            continue
        out.append(row)
    return out


def _aggregate_cells(rows: Sequence[Mapping[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("cell_key", ""))].append(dict(row or {}))
    aggregates: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for cell_key, cell_rows in sorted(grouped.items()):
        if not cell_rows:
            continue
        first = cell_rows[0]
        counts[cell_key] = len(cell_rows)
        aggregate: Dict[str, Any] = {
            "cell_key": cell_key,
            "scope_label": str(first.get("scope_label", "") or ""),
            "train_doc_count": int(first.get("train_doc_count", 0) or 0),
            "claim_level": str(first.get("claim_level", "") or ""),
            "baseline_family": str(first.get("baseline_family", "") or ""),
            "recipe_id": str(first.get("recipe_id", "") or ""),
            "recipe_label": str(first.get("recipe_label", "") or ""),
            "comparison_label": str(first.get("comparison_label", "") or ""),
            "fixed_leaf_tokens": int(first.get("fixed_leaf_tokens", 0) or 0),
            "n_rows": len(cell_rows),
            "raw_job_names": [str(row.get("job_name", "") or "") for row in cell_rows],
            "strict_collapse_pass_any": any(bool(row.get("strict_collapse_pass", False)) for row in cell_rows),
        }
        for metric_key, _, _ in QUALITY_METRICS:
            aggregate[metric_key] = _mean(_safe_float(row.get(metric_key)) for row in cell_rows)
        aggregate["val_root_mae_mean"] = _mean(_safe_float(row.get("val_root_mae_mean")) for row in cell_rows)
        for metric_key, _, _, _ in RUNTIME_METRICS:
            aggregate[metric_key] = _mean(_safe_float(row.get(metric_key)) for row in cell_rows)
        aggregates.append(aggregate)
    return aggregates, counts


def _best_row(rows: Sequence[Mapping[str, Any]], metric_key: str, mode: str) -> Mapping[str, Any] | None:
    finite = [row for row in rows if _is_finite(row.get(metric_key))]
    if not finite:
        return None
    if mode == "max":
        return max(finite, key=lambda row: float(row.get(metric_key, float("-inf"))))
    return min(finite, key=lambda row: float(row.get(metric_key, float("inf"))))


def _slice_chart_style(rows: Sequence[Mapping[str, Any]]) -> str:
    tokens = sorted(
        {
            int(row.get("fixed_leaf_tokens", 0) or 0)
            for row in rows
            if str(row.get("claim_level", "")) == "empirical_geometry"
            and str(row.get("baseline_family", "")) == "tree_neural"
        }
    )
    return "geometry" if len(tokens) > 1 else "comparison"


def _build_empirical_by_scope(
    normalized_rows: Sequence[Mapping[str, Any]],
    *,
    one_leaf_target_fixed_leaf_tokens: int,
) -> Dict[str, Dict[str, Any]]:
    empirical_rows = _completed_rows(normalized_rows, claim_level="empirical_geometry")
    scopes = sorted({str(row.get("scope_label", "") or "") for row in empirical_rows})
    train_doc_counts = sorted({int(row.get("train_doc_count", 0) or 0) for row in empirical_rows})
    empirical_by_scope: Dict[str, Dict[str, Any]] = {}
    for scope_label in scopes:
        empirical_by_scope[scope_label] = {}
        for train_doc_count in train_doc_counts:
            rows = [
                row
                for row in empirical_rows
                if str(row.get("scope_label", "")) == scope_label
                and int(row.get("train_doc_count", 0) or 0) == train_doc_count
            ]
            if not rows:
                continue
            exact_rows = [
                row
                for row in _completed_rows(normalized_rows, claim_level="exact_collapse_candidate")
                if str(row.get("scope_label", "")) == scope_label
                and int(row.get("train_doc_count", 0) or 0) == train_doc_count
            ]
            best_by_metric: Dict[str, Dict[str, Any]] = {}
            fno_rows = [row for row in rows if str(row.get("baseline_family", "")) in {"official_fno", "official_fno_sumlen"}]
            tree_rows = [row for row in rows if str(row.get("baseline_family", "")) == "tree_neural"]
            one_leaf_rows = [
                row for row in tree_rows if int(row.get("fixed_leaf_tokens", 0) or 0) == int(one_leaf_target_fixed_leaf_tokens)
            ]
            for metric_key, _, _ in QUALITY_METRICS:
                metric_summary: Dict[str, Any] = {}
                best_fno = _best_row(fno_rows, metric_key, "min")
                best_tree = _best_row(tree_rows, metric_key, "min")
                best_one_leaf = _best_row(one_leaf_rows, metric_key, "min")
                exact_row = _best_row(exact_rows, metric_key, "min")
                if best_fno is not None:
                    metric_summary["best_fno"] = _row_brief(best_fno)
                if best_tree is not None:
                    metric_summary["best_tree"] = _row_brief(best_tree)
                if best_one_leaf is not None:
                    metric_summary["best_one_leaf_tree"] = _row_brief(best_one_leaf)
                if exact_row is not None:
                    metric_summary["exact_collapse"] = _row_brief(exact_row)
                if best_fno is not None and exact_row is not None:
                    metric_summary["exact_gap_vs_best_fno"] = float(exact_row.get(metric_key)) - float(best_fno.get(metric_key))
                best_by_metric[metric_key] = metric_summary
            empirical_by_scope[scope_label][str(train_doc_count)] = {
                "scope_label": scope_label,
                "train_doc_count": train_doc_count,
                "chart_style": _slice_chart_style(rows),
                "n_rows": len(rows),
                "fixed_leaf_tokens": sorted({int(row.get("fixed_leaf_tokens", 0) or 0) for row in rows}),
                "rows": [_row_brief(row) for row in sorted(rows, key=lambda item: (_series_rank(item), int(item.get("fixed_leaf_tokens", 0) or 0), str(item.get("job_name", ""))))],
                "best_by_metric": best_by_metric,
            }
    return empirical_by_scope


def _exact_collapse_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    relevant = _completed_rows(rows, claim_level="exact_collapse_candidate")
    relevant.sort(
        key=lambda row: (
            str(row.get("scope_label", "")),
            int(row.get("train_doc_count", 0) or 0),
            int(row.get("fixed_leaf_tokens", 0) or 0),
        )
    )
    out: List[Dict[str, Any]] = []
    for row in relevant:
        out.append(
            {
                "job_name": str(row.get("job_name", "") or ""),
                "scope_label": str(row.get("scope_label", "") or ""),
                "train_doc_count": int(row.get("train_doc_count", 0) or 0),
                "fixed_leaf_tokens": int(row.get("fixed_leaf_tokens", 0) or 0),
                "state": str(row.get("state", "") or ""),
                "strict_collapse_pass": bool(row.get("strict_collapse_pass", False)),
                "test_root_mae_mean": _safe_float(row.get("test_root_mae_mean")),
                "test_leaf_mae_mean": _safe_float(row.get("test_leaf_mae_mean")),
                "test_merge_mae_mean": _safe_float(row.get("test_merge_mae_mean")),
                "config_diff_field_count": len(dict(row.get("config_diff_vs_official_fno") or {})),
                "reference_bundle_source": str(row.get("reference_bundle_source", "") or ""),
                "train_prefix_counts": [int(v) for v in list(row.get("train_prefix_counts") or [])],
            }
        )
    return out


def _build_report_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    raw_rows = [dict(row or {}) for row in list(payload.get("rows") or [])]
    normalized_rows = [
        _normalize_row(row, payload)
        for row in raw_rows
    ]
    completed_rows = _completed_rows(normalized_rows)
    contract_statuses = sorted(
        {
            str(row.get("contract_status", "") or "")
            for row in normalized_rows
            if str(row.get("contract_status", "") or "").strip()
        }
    )
    quarantined_rows = [
        row
        for row in normalized_rows
        if str(row.get("contract_status", "") or "").strip()
        and not is_headline_contract_status(row.get("contract_status"))
    ]
    scopes = sorted({str(row.get("scope_label", "") or "") for row in completed_rows if str(row.get("scope_label", "") or "")})
    train_doc_counts = sorted(
        {
            int(row.get("train_doc_count", 0) or 0)
            for row in completed_rows
            if int(row.get("train_doc_count", 0) or 0) > 0
        }
    )
    claim_levels = sorted({str(row.get("claim_level", "") or "") for row in completed_rows if str(row.get("claim_level", "") or "")})
    cell_aggregates, n_rows_per_cell = _aggregate_cells(completed_rows)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_kind": "markov_parity_self_contained",
        "source_files": {
            "simulation_root": str(payload.get("simulation_root", "") or ""),
            "simulation_roots": [
                str(item)
                for item in list(payload.get("simulation_roots") or [])
                if str(item).strip()
            ],
            "manifest": str(payload.get("manifest_path", "") or ""),
            "manifests": [
                str(item)
                for item in list(payload.get("manifest_paths") or [])
                if str(item).strip()
            ],
            "summary": str(payload.get("summary_path", "") or ""),
            "summaries": [
                str(item)
                for item in list(payload.get("summary_paths") or [])
                if str(item).strip()
            ],
            "scheduler_status": str(payload.get("scheduler_path", "") or ""),
            "scheduler_statuses": [
                str(item)
                for item in list(payload.get("scheduler_paths") or [])
                if str(item).strip()
            ],
        },
        "data_contract": {
            "self_contained": True,
            "description": (
                "This report only uses artifacts already present inside the specified simulation root."
                if len(list(payload.get("simulation_roots") or [])) <= 1
                else "This report only uses artifacts already present inside the specified simulation roots."
            ),
            "headline_filters_quarantined_rows": True,
        },
        "status": {
            "state": str(payload.get("state", "") or ""),
            "evidence_status": str(payload.get("evidence_status", "") or ""),
            "items_total": int(payload.get("items_total", 0) or 0),
            "completed_items": int(payload.get("completed_items", 0) or 0),
            "failed_items": int(payload.get("failed_items", 0) or 0),
            "active_items": int(payload.get("active_items", 0) or 0),
            "pending_items": int(payload.get("pending_items", 0) or 0),
            "headline_row_count": int(len(completed_rows)),
            "quarantined_row_count": int(len(quarantined_rows)),
            "source_run_count": max(
                1,
                len(list(payload.get("simulation_roots") or [])),
            ),
        },
        "coverage": {
            "scopes": scopes,
            "train_doc_counts": train_doc_counts,
            "claim_levels": claim_levels,
            "contract_statuses": contract_statuses,
            "assumed_doc_tokens": int(payload.get("assumed_doc_tokens", 0) or 0),
            "one_leaf_target_fixed_leaf_tokens": int(payload.get("one_leaf_target_fixed_leaf_tokens", 0) or 0),
        },
        "palette": {
            "families": dict(FAMILY_PALETTE),
            "labels": dict(FAMILY_LABELS),
            "tree_recipe_styles": {
                recipe_id: dict(style)
                for recipe_id, style in RECIPE_STYLES.items()
                if recipe_id not in {"fno_baseline"}
            },
            "exact_collapse_marker": dict(RECIPE_STYLES["exact_collapse_candidate"]),
        },
        "normalized_rows": [_row_brief(row) | {
            "cell_key": str(row.get("cell_key", "") or ""),
            "run_json_path": str(row.get("run_json_path", "") or ""),
            "source_simulation_root": str(row.get("source_simulation_root", "") or ""),
            "resident_store_misses": _safe_float(row.get("resident_store_misses")),
            "gpu_allocated_mem_peak_gb": _safe_float(row.get("gpu_allocated_mem_peak_gb")),
            "runtime_workers_per_mig": _safe_float(row.get("runtime_workers_per_mig")),
            "one_leaf_target": bool(row.get("one_leaf_target", False)),
        } for row in sorted(normalized_rows, key=lambda row: (str(row.get("scope_label", "")), int(row.get("train_doc_count", 0) or 0), _series_rank(row), int(row.get("fixed_leaf_tokens", 0) or 0), str(row.get("job_name", ""))))],
        "cell_aggregates": cell_aggregates,
        "n_rows_per_cell": n_rows_per_cell,
        "empirical_by_scope": _build_empirical_by_scope(
            normalized_rows,
            one_leaf_target_fixed_leaf_tokens=int(payload.get("one_leaf_target_fixed_leaf_tokens", 0) or 0),
        ),
        "exact_collapse_rows": _exact_collapse_summary(normalized_rows),
        "figures": {},
        "figure_inventory": {},
        "figure_order": [],
        "row_figure_coverage": {},
    }
    return summary


def _register_figure(
    summary: MutableMapping[str, Any],
    *,
    title: str,
    path: Path,
    figure_kind: str,
    chart_style: str,
    job_names: Sequence[str],
    callouts: Sequence[str],
    scope_label: str = "",
    train_doc_count: int | None = None,
) -> None:
    summary["figures"][title] = str(path)
    summary["figure_order"].append(title)
    summary["figure_inventory"][title] = {
        "path": str(path),
        "figure_kind": figure_kind,
        "chart_style": chart_style,
        "scope_label": scope_label,
        "train_doc_count": None if train_doc_count is None else int(train_doc_count),
        "job_names": sorted({str(name) for name in job_names if str(name)}),
        "callouts": [str(item) for item in callouts if str(item).strip()],
    }
    is_runtime = figure_kind.startswith("runtime")
    coverage_key = "runtime_figures" if is_runtime else "quality_figures"
    for job_name in summary["figure_inventory"][title]["job_names"]:
        entry = summary["row_figure_coverage"].setdefault(
            job_name,
            {"quality_figures": [], "runtime_figures": [], "all_figures": []},
        )
        entry[coverage_key].append(title)
        entry["all_figures"].append(title)


def _subplot_axes(nrows: int, ncols: int, figsize: tuple[float, float]) -> tuple[Any, List[Any]]:
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if isinstance(axes, list):
        flat_axes = axes
    else:
        try:
            flat_axes = list(axes.flat)
        except Exception:
            flat_axes = [axes]
    return fig, flat_axes


def _scatter_raw_points(ax: Any, xs: Sequence[float], ys: Sequence[float], *, color: str, marker: str, size: float = 32.0) -> None:
    finite = [(x, y) for x, y in zip(xs, ys) if _is_finite(y)]
    if not finite:
        return
    ax.scatter(
        [item[0] for item in finite],
        [item[1] for item in finite],
        color=color,
        marker=marker,
        s=size,
        alpha=0.75,
        linewidths=0.0,
        zorder=4,
    )


def _metric_text(value: Any) -> str:
    return "n/a" if not _is_finite(value) else f"{float(value):.6f}"


def _gap_text(value: Any) -> str:
    return "n/a" if not _is_finite(value) else f"{float(value):+.6f}"


def _leaf_count_text(value: int) -> str:
    count = int(value)
    return f"{count} leaf" if count == 1 else f"{count} leaves"


def _sort_slice_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        dict(row or {})
        for row in sorted(
            rows,
            key=lambda row: (_series_rank(row), int(row.get("fixed_leaf_tokens", 0) or 0), str(row.get("job_name", ""))),
        )
    ]


def _plot_overview(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope_label: str,
    metric_defs: Sequence[tuple[str, str, str]] | Sequence[tuple[str, str, str, str]],
    output_path: Path,
    is_runtime: bool,
) -> List[str]:
    fig, axes = _subplot_axes(1 if not is_runtime else 2, 3 if not is_runtime else 2, (14.0, 4.8 if not is_runtime else 8.5))
    train_doc_counts = sorted({int(row.get("train_doc_count", 0) or 0) for row in rows})
    callouts: List[str] = []
    for axis_idx, metric_def in enumerate(metric_defs):
        ax = axes[axis_idx]
        metric_key = str(metric_def[0])
        label = str(metric_def[1])
        mode = str(metric_def[3]) if is_runtime else "min"
        xs: List[int] = []
        best_fno: List[float] = []
        best_tree: List[float] = []
        best_one_leaf: List[float] = []
        exact_vals: List[float] = []
        for train_doc_count in train_doc_counts:
            scoped = [
                row
                for row in rows
                if int(row.get("train_doc_count", 0) or 0) == train_doc_count
            ]
            empirical = [row for row in scoped if str(row.get("claim_level", "")) == "empirical_geometry"]
            exact = [row for row in scoped if str(row.get("claim_level", "")) == "exact_collapse_candidate"]
            fno_rows = [row for row in empirical if str(row.get("baseline_family", "")) in {"official_fno", "official_fno_sumlen"}]
            tree_rows = [row for row in empirical if str(row.get("baseline_family", "")) == "tree_neural"]
            one_leaf_rows = [row for row in tree_rows if bool(row.get("one_leaf_target", False))]
            best_fno_row = _best_row(fno_rows, metric_key, mode)
            best_tree_row = _best_row(tree_rows, metric_key, mode)
            best_one_leaf_row = _best_row(one_leaf_rows, metric_key, mode)
            exact_row = _best_row(exact, metric_key, mode)
            xs.append(train_doc_count)
            best_fno.append(_safe_float(None if best_fno_row is None else best_fno_row.get(metric_key)))
            best_tree.append(_safe_float(None if best_tree_row is None else best_tree_row.get(metric_key)))
            best_one_leaf.append(_safe_float(None if best_one_leaf_row is None else best_one_leaf_row.get(metric_key)))
            exact_vals.append(_safe_float(None if exact_row is None else exact_row.get(metric_key)))
        if any(_is_finite(value) for value in best_fno):
            ax.plot(xs, best_fno, color=FAMILY_PALETTE["official_fno"], marker="o", linewidth=2.0, label="best FNO")
        if any(_is_finite(value) for value in best_tree):
            ax.plot(xs, best_tree, color=FAMILY_PALETTE["tree_neural"], marker="o", linewidth=2.0, label="best tree")
        if any(_is_finite(value) for value in best_one_leaf):
            ax.plot(xs, best_one_leaf, color=FAMILY_PALETTE["tree_neural"], marker="s", linewidth=2.0, linestyle="--", label="best one-leaf tree")
        if any(_is_finite(value) for value in exact_vals):
            ax.plot(xs, exact_vals, color=FAMILY_PALETTE["tree_neural"], marker="*", linewidth=0.0, markersize=12, label="exact collapse")
        ax.set_title(label)
        ax.set_xlabel("train_docs")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        if axis_idx == 0:
            ax.legend(loc="best", fontsize=8)
        if xs:
            ax.set_xticks(xs)
        best_fno_overall = _best_row(
            [row for row in rows if str(row.get("claim_level", "")) == "empirical_geometry" and str(row.get("baseline_family", "")) in {"official_fno", "official_fno_sumlen"}],
            metric_key,
            mode,
        )
        if best_fno_overall is not None:
            callouts.append(
                f"{label}: best FNO = {_comparison_label(best_fno_overall)} @ train_docs={int(best_fno_overall.get('train_doc_count', 0) or 0)} ({_metric_text(best_fno_overall.get(metric_key))})."
            )
    for ax in axes[len(metric_defs):]:
        ax.axis("off")
    fig.suptitle(f"{scope_label.title()} {'Runtime' if is_runtime else 'Quality'} Overview")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return callouts[:4]


def _plot_geometry_triptych(
    empirical_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    *,
    scope_label: str,
    train_doc_count: int,
    one_leaf_target_fixed_leaf_tokens: int,
    metric_defs: Sequence[tuple[str, str, str]],
    output_path: Path,
) -> List[str]:
    fig, axes = _subplot_axes(1, 3, (15.0, 4.9))
    sorted_rows = _sort_slice_rows(empirical_rows)
    callouts: List[str] = []
    x_values = sorted({int(row.get("fixed_leaf_tokens", 0) or 0) for row in sorted_rows})
    for axis_idx, (metric_key, label, _) in enumerate(metric_defs):
        ax = axes[axis_idx]
        plotted_series: set[str] = set()
        for series_key in sorted({str(row.get("series_key", "") or "") for row in sorted_rows}, key=lambda key: SERIES_ORDER.get(key, 99)):
            series_rows = [row for row in sorted_rows if str(row.get("series_key", "") or "") == series_key]
            if not series_rows:
                continue
            exemplar = series_rows[0]
            style = _style_for_row(exemplar)
            if str(exemplar.get("baseline_family", "")) in {"official_fno", "official_fno_sumlen"}:
                y_value = _mean(_safe_float(row.get(metric_key)) for row in series_rows)
                if _is_finite(y_value):
                    ax.axhline(
                        y_value,
                        color=style["color"],
                        linestyle="--",
                        linewidth=2.0,
                        label=_comparison_label(exemplar) if series_key not in plotted_series else "_nolegend_",
                    )
                    raw_xs = [float(one_leaf_target_fixed_leaf_tokens)] * len(series_rows)
                    raw_ys = [_safe_float(row.get(metric_key)) for row in series_rows]
                    _scatter_raw_points(ax, raw_xs, raw_ys, color=style["color"], marker=style["marker"], size=42.0)
                    plotted_series.add(series_key)
                continue
            grouped: DefaultDict[int, List[float]] = defaultdict(list)
            for row in series_rows:
                grouped[int(row.get("fixed_leaf_tokens", 0) or 0)].append(_safe_float(row.get(metric_key)))
            xs = sorted(grouped)
            ys = [_mean(grouped[token]) for token in xs]
            ax.plot(
                xs,
                ys,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.0,
                label=_comparison_label(exemplar) if series_key not in plotted_series else "_nolegend_",
            )
            for token in xs:
                raw_values = grouped[token]
                raw_xs = [float(token) + (idx - (len(raw_values) - 1) / 2.0) * 0.8 for idx in range(len(raw_values))]
                _scatter_raw_points(ax, raw_xs, raw_values, color=style["color"], marker=style["marker"])
            plotted_series.add(series_key)
        for exact_row in exact_rows:
            if not _is_finite(exact_row.get(metric_key)):
                continue
            style = _style_for_row(exact_row)
            ax.scatter(
                [int(exact_row.get("fixed_leaf_tokens", 0) or 0)],
                [_safe_float(exact_row.get(metric_key))],
                color=style["color"],
                marker=style["marker"],
                s=130,
                edgecolors="#111827",
                linewidths=0.6,
                label="Exact collapse" if axis_idx == 0 else "_nolegend_",
                zorder=5,
            )
        if int(one_leaf_target_fixed_leaf_tokens) > 0:
            ax.axvline(int(one_leaf_target_fixed_leaf_tokens), color="#334155", linestyle=":", linewidth=1.4, alpha=0.7)
        ax.set_title(label)
        ax.set_xlabel("fixed_leaf_tokens")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.set_xticks(x_values)
        if axis_idx == 0:
            ax.legend(loc="best", fontsize=8)
            best_fno = _best_row([row for row in sorted_rows if str(row.get("baseline_family", "")) in {"official_fno", "official_fno_sumlen"}], metric_key, "min")
            best_tree = _best_row([row for row in sorted_rows if str(row.get("baseline_family", "")) == "tree_neural"], metric_key, "min")
            if best_fno is not None and best_tree is not None:
                gap = float(best_tree.get(metric_key)) - float(best_fno.get(metric_key))
                callouts.append(
                    f"{label}: best tree = {_comparison_label(best_tree)} leaf {int(best_tree.get('fixed_leaf_tokens', 0) or 0)} ({_metric_text(best_tree.get(metric_key))}); gap vs best FNO = {gap:+.6f}."
                )
    fig.suptitle(f"{scope_label.title()} Geometry @ train_docs={train_doc_count}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return callouts[:3]


def _plot_comparison_triptych(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope_label: str,
    train_doc_count: int,
    metric_defs: Sequence[tuple[str, str, str]],
    output_path: Path,
    title_suffix: str,
) -> List[str]:
    fig, axes = _subplot_axes(1, 3, (15.0, 5.4))
    grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    exemplar_by_label: Dict[str, Dict[str, Any]] = {}
    for row in _sort_slice_rows(rows):
        label = str(row.get("comparison_label", "") or "")
        grouped[label].append(dict(row or {}))
        exemplar_by_label.setdefault(label, dict(row or {}))
    labels = sorted(grouped, key=lambda label: _series_rank(exemplar_by_label[label]))
    xs = list(range(len(labels)))
    callouts: List[str] = []
    for axis_idx, (metric_key, label, _) in enumerate(metric_defs):
        ax = axes[axis_idx]
        bar_colors = [str(_style_for_row(exemplar_by_label[item])["color"]) for item in labels]
        bar_hatches = [str(_style_for_row(exemplar_by_label[item])["hatch"]) for item in labels]
        heights = [_mean(_safe_float(row.get(metric_key)) for row in grouped[item]) for item in labels]
        bars = ax.bar(xs, heights, color=bar_colors, alpha=0.85, edgecolor="#0f172a")
        for bar, hatch in zip(bars, bar_hatches):
            bar.set_hatch(hatch)
        for idx, item in enumerate(labels):
            exemplar = exemplar_by_label[item]
            style = _style_for_row(exemplar)
            raw_values = [_safe_float(row.get(metric_key)) for row in grouped[item]]
            raw_xs = [float(idx) + (offset - (len(raw_values) - 1) / 2.0) * 0.08 for offset in range(len(raw_values))]
            _scatter_raw_points(ax, raw_xs, raw_values, color=style["color"], marker=style["marker"], size=34.0)
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.grid(alpha=0.25, axis="y")
        if axis_idx == 0:
            best_row = _best_row(rows, metric_key, "min")
            best_fno = _best_row([row for row in rows if str(row.get("baseline_family", "")) in {"official_fno", "official_fno_sumlen"}], metric_key, "min")
            if best_row is not None and best_fno is not None:
                gap = float(best_row.get(metric_key)) - float(best_fno.get(metric_key))
                callouts.append(
                    f"{label}: best setting = {_comparison_label(best_row)} ({_metric_text(best_row.get(metric_key))}); gap vs best FNO = {gap:+.6f}."
                )
    fig.suptitle(f"{scope_label.title()} {title_suffix} @ train_docs={train_doc_count}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return callouts[:3]


def _plot_runtime_geometry(
    empirical_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    *,
    scope_label: str,
    train_doc_count: int,
    one_leaf_target_fixed_leaf_tokens: int,
    output_path: Path,
) -> List[str]:
    fig, axes = _subplot_axes(2, 2, (14.0, 9.0))
    sorted_rows = _sort_slice_rows(empirical_rows)
    callouts: List[str] = []
    x_values = sorted({int(row.get("fixed_leaf_tokens", 0) or 0) for row in sorted_rows})
    for axis_idx, (metric_key, label, _, _) in enumerate(RUNTIME_METRICS):
        ax = axes[axis_idx]
        plotted_series: set[str] = set()
        for series_key in sorted({str(row.get("series_key", "") or "") for row in sorted_rows}, key=lambda key: SERIES_ORDER.get(key, 99)):
            series_rows = [row for row in sorted_rows if str(row.get("series_key", "") or "") == series_key]
            if not series_rows:
                continue
            exemplar = series_rows[0]
            style = _style_for_row(exemplar)
            if str(exemplar.get("baseline_family", "")) in {"official_fno", "official_fno_sumlen"}:
                y_value = _mean(_safe_float(row.get(metric_key)) for row in series_rows)
                if _is_finite(y_value):
                    ax.axhline(
                        y_value,
                        color=style["color"],
                        linestyle="--",
                        linewidth=2.0,
                        label=_comparison_label(exemplar) if series_key not in plotted_series else "_nolegend_",
                    )
                    raw_xs = [float(one_leaf_target_fixed_leaf_tokens)] * len(series_rows)
                    raw_ys = [_safe_float(row.get(metric_key)) for row in series_rows]
                    _scatter_raw_points(ax, raw_xs, raw_ys, color=style["color"], marker=style["marker"], size=42.0)
                    plotted_series.add(series_key)
                continue
            grouped: DefaultDict[int, List[float]] = defaultdict(list)
            for row in series_rows:
                grouped[int(row.get("fixed_leaf_tokens", 0) or 0)].append(_safe_float(row.get(metric_key)))
            xs = sorted(grouped)
            ys = [_mean(grouped[token]) for token in xs]
            ax.plot(
                xs,
                ys,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.0,
                label=_comparison_label(exemplar) if series_key not in plotted_series else "_nolegend_",
            )
            for token in xs:
                raw_values = grouped[token]
                raw_xs = [float(token) + (idx - (len(raw_values) - 1) / 2.0) * 0.8 for idx in range(len(raw_values))]
                _scatter_raw_points(ax, raw_xs, raw_values, color=style["color"], marker=style["marker"])
            plotted_series.add(series_key)
        for exact_row in exact_rows:
            if not _is_finite(exact_row.get(metric_key)):
                continue
            style = _style_for_row(exact_row)
            ax.scatter(
                [int(exact_row.get("fixed_leaf_tokens", 0) or 0)],
                [_safe_float(exact_row.get(metric_key))],
                color=style["color"],
                marker=style["marker"],
                s=130,
                edgecolors="#111827",
                linewidths=0.6,
                label="Exact collapse" if axis_idx == 0 else "_nolegend_",
                zorder=5,
            )
        if int(one_leaf_target_fixed_leaf_tokens) > 0:
            ax.axvline(int(one_leaf_target_fixed_leaf_tokens), color="#334155", linestyle=":", linewidth=1.4, alpha=0.7)
        ax.set_title(label)
        ax.set_xlabel("fixed_leaf_tokens")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.set_xticks(x_values)
        if axis_idx == 0:
            ax.legend(loc="best", fontsize=8)
    runtime_modes = sorted({str(row.get("runtime_data_mode", "") or "") for row in sorted_rows if str(row.get("runtime_data_mode", "") or "")})
    bucket_modes = sorted({str(row.get("runtime_bucket_mode", "") or "") for row in sorted_rows if str(row.get("runtime_bucket_mode", "") or "")})
    if runtime_modes:
        callouts.append(f"Runtime mode: {', '.join(runtime_modes)}.")
    if bucket_modes:
        callouts.append(f"Bucket mode: {', '.join(bucket_modes)}.")
    if all(_safe_float(row.get("steady_state_h2d_bytes")) == 0.0 for row in sorted_rows + list(exact_rows) if _is_finite(row.get("steady_state_h2d_bytes"))):
        callouts.append("Steady-state H2D bytes are zero across all plotted rows.")
    fig.suptitle(f"{scope_label.title()} Runtime Geometry @ train_docs={train_doc_count}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return callouts[:4]


def _plot_runtime_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope_label: str,
    train_doc_count: int,
    output_path: Path,
) -> List[str]:
    fig, axes = _subplot_axes(2, 2, (14.0, 9.0))
    grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    exemplar_by_label: Dict[str, Dict[str, Any]] = {}
    for row in _sort_slice_rows(rows):
        label = str(row.get("comparison_label", "") or "")
        grouped[label].append(dict(row or {}))
        exemplar_by_label.setdefault(label, dict(row or {}))
    labels = sorted(grouped, key=lambda label: _series_rank(exemplar_by_label[label]))
    xs = list(range(len(labels)))
    callouts: List[str] = []
    for axis_idx, (metric_key, label, _, mode) in enumerate(RUNTIME_METRICS):
        ax = axes[axis_idx]
        heights = [_mean(_safe_float(row.get(metric_key)) for row in grouped[item]) for item in labels]
        bar_colors = [str(_style_for_row(exemplar_by_label[item])["color"]) for item in labels]
        bar_hatches = [str(_style_for_row(exemplar_by_label[item])["hatch"]) for item in labels]
        bars = ax.bar(xs, heights, color=bar_colors, alpha=0.85, edgecolor="#0f172a")
        for bar, hatch in zip(bars, bar_hatches):
            bar.set_hatch(hatch)
        for idx, item in enumerate(labels):
            exemplar = exemplar_by_label[item]
            style = _style_for_row(exemplar)
            raw_values = [_safe_float(row.get(metric_key)) for row in grouped[item]]
            raw_xs = [float(idx) + (offset - (len(raw_values) - 1) / 2.0) * 0.08 for offset in range(len(raw_values))]
            _scatter_raw_points(ax, raw_xs, raw_values, color=style["color"], marker=style["marker"], size=34.0)
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.grid(alpha=0.25, axis="y")
        best_row = _best_row(rows, metric_key, mode)
        if best_row is not None:
            callouts.append(
                f"{label}: best setting = {_comparison_label(best_row)} ({_metric_text(best_row.get(metric_key))})."
            )
    fig.suptitle(f"{scope_label.title()} Runtime Comparison @ train_docs={train_doc_count}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return callouts[:4]


def _plot_exact_collapse_vs_fno(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope_label: str,
    output_path: Path,
) -> List[str]:
    fig, axes = _subplot_axes(2, 2, (14.0, 9.0))
    empirical = [
        row
        for row in _completed_rows(rows, claim_level="empirical_geometry")
        if str(row.get("scope_label", "")) == scope_label
    ]
    exact_rows = [
        row
        for row in _completed_rows(rows, claim_level="exact_collapse_candidate")
        if str(row.get("scope_label", "")) == scope_label
    ]
    train_doc_counts = sorted({int(row.get("train_doc_count", 0) or 0) for row in exact_rows})
    callouts: List[str] = []
    for axis_idx, (metric_key, label, _) in enumerate(QUALITY_METRICS):
        ax = axes[axis_idx]
        xs: List[int] = []
        gaps: List[float] = []
        for train_doc_count in train_doc_counts:
            exact_row = _best_row(
                [row for row in exact_rows if int(row.get("train_doc_count", 0) or 0) == train_doc_count],
                metric_key,
                "min",
            )
            best_fno = _best_row(
                [
                    row
                    for row in empirical
                    if int(row.get("train_doc_count", 0) or 0) == train_doc_count
                    and str(row.get("baseline_family", "")) in {"official_fno", "official_fno_sumlen"}
                ],
                metric_key,
                "min",
            )
            if exact_row is None or best_fno is None:
                continue
            xs.append(train_doc_count)
            gaps.append(float(exact_row.get(metric_key)) - float(best_fno.get(metric_key)))
        if xs:
            ax.plot(xs, gaps, color=FAMILY_PALETTE["tree_neural"], marker="*", linewidth=2.0)
            ax.axhline(0.0, color="#334155", linestyle="--", linewidth=1.2)
            ax.set_xticks(xs)
        ax.set_title(f"{label} gap vs best FNO")
        ax.set_xlabel("train_docs")
        ax.set_ylabel("gap")
        ax.grid(alpha=0.25)
        if gaps:
            callouts.append(f"{label}: latest exact gap vs best FNO = {gaps[-1]:+.6f} at train_docs={xs[-1]}.")
    status_ax = axes[3]
    xs = list(range(len(train_doc_counts)))
    values = [1.0 if bool(_best_row([row for row in exact_rows if int(row.get("train_doc_count", 0) or 0) == train_doc_count], "test_root_mae_mean", "min").get("strict_collapse_pass", False)) else 0.0 for train_doc_count in train_doc_counts] if train_doc_counts else []
    colors = ["#16a34a" if value > 0.5 else "#dc2626" for value in values]
    if xs:
        bars = status_ax.bar(xs, values, color=colors, alpha=0.85)
        status_ax.set_xticks(xs)
        status_ax.set_xticklabels([str(value) for value in train_doc_counts])
        status_ax.set_ylim(0.0, 1.1)
        for idx, bar in enumerate(bars):
            train_doc_count = train_doc_counts[idx]
            exact_row = _best_row([row for row in exact_rows if int(row.get("train_doc_count", 0) or 0) == train_doc_count], "test_root_mae_mean", "min")
            diff_field_count = len(dict(exact_row.get("config_diff_vs_official_fno") or {})) if exact_row is not None else 0
            status_ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.03, f"diff={diff_field_count}", ha="center", va="bottom", fontsize=8)
    status_ax.set_title("Strict collapse status")
    status_ax.set_xlabel("train_docs")
    status_ax.set_ylabel("pass")
    status_ax.grid(alpha=0.25, axis="y")
    fig.suptitle(f"{scope_label.title()} Exact Collapse vs Best FNO")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    passes = sum(1 for value in values if value > 0.5)
    callouts.append(f"Strict collapse passes: {passes}/{len(values)}.")
    return callouts[:4]


def _plot_exact_collapse_repair_arms(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope_label: str,
    output_path: Path,
) -> List[str]:
    repair_rows = [
        dict(row or {})
        for row in _completed_rows(rows)
        if str(row.get("scope_label", "") or "") == scope_label
        and _is_exact_collapse_repair_row(row)
        and _exact_collapse_repair_series_key(row) != "exact_collapse_legacy_control"
    ]
    fig, axes = _subplot_axes(2, 2, (14.0, 9.0))
    metric_defs: Sequence[tuple[str, str, str]] = (
        ("test_root_mae_mean", "Root MAE", "min"),
        ("test_leaf_mae_mean", "Leaf MAE", "min"),
        ("test_merge_mae_mean", "Merge MAE", "min"),
        ("elapsed_s", "Elapsed Time (s)", "min"),
    )
    series_offset_units = {
        "official_fno": -1.5,
        "exact_collapse_runtime_match": -0.5,
        "exact_collapse_candidate": 0.5,
        "exact_collapse_legacy_control": 1.5,
    }
    train_doc_counts = sorted({int(row.get("train_doc_count", 0) or 0) for row in repair_rows})
    if len(train_doc_counts) >= 2:
        min_gap = min(
            max(1, train_doc_counts[idx + 1] - train_doc_counts[idx])
            for idx in range(len(train_doc_counts) - 1)
        )
    else:
        min_gap = max(1, train_doc_counts[0] if train_doc_counts else 1)
    series_offset_scale = float(min_gap) * 0.06
    plotted_labels: set[str] = set()
    callouts: List[str] = []

    for axis_idx, (metric_key, metric_label, _) in enumerate(metric_defs):
        ax = axes[axis_idx]
        for series_key in ("official_fno", "exact_collapse_runtime_match", "exact_collapse_candidate", "exact_collapse_legacy_control"):
            series_rows = [row for row in repair_rows if _exact_collapse_repair_series_key(row) == series_key]
            if not series_rows:
                continue
            exemplar = series_rows[0]
            style = _style_for_row(exemplar)
            xs: List[int] = []
            ys: List[float] = []
            for train_doc_count in train_doc_counts:
                point_rows = [
                    row
                    for row in series_rows
                    if int(row.get("train_doc_count", 0) or 0) == train_doc_count
                ]
                value = _mean(_safe_float(row.get(metric_key)) for row in point_rows)
                if not _is_finite(value):
                    continue
                xs.append(train_doc_count)
                ys.append(value)
            if not xs:
                continue
            ax.plot(
                [float(x) + float(series_offset_units.get(series_key, 0.0)) * series_offset_scale for x in xs],
                ys,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=9.0 if series_key == "exact_collapse_candidate" else 7.0,
                markeredgecolor="#0f172a",
                markeredgewidth=0.5,
                label=_exact_collapse_repair_label(exemplar) if series_key not in plotted_labels else "_nolegend_",
            )
            for train_doc_count in xs:
                raw_rows = [
                    row
                    for row in series_rows
                    if int(row.get("train_doc_count", 0) or 0) == train_doc_count
                ]
                raw_values = [_safe_float(row.get(metric_key)) for row in raw_rows]
                raw_xs = [
                    float(train_doc_count)
                    + float(series_offset_units.get(series_key, 0.0)) * series_offset_scale
                    + (idx - (len(raw_values) - 1) / 2.0) * max(4.0, series_offset_scale * 0.25)
                    for idx in range(len(raw_values))
                ]
                _scatter_raw_points(ax, raw_xs, raw_values, color=style["color"], marker=style["marker"], size=40.0)
            plotted_labels.add(series_key)
        ax.set_title(metric_label)
        ax.set_xlabel("train_docs")
        ax.set_ylabel(metric_label)
        ax.grid(alpha=0.25)
        if train_doc_counts:
            ax.set_xticks(train_doc_counts)
        if axis_idx == 0:
            ax.legend(loc="best", fontsize=8)

    fno_rows = [row for row in repair_rows if _exact_collapse_repair_series_key(row) == "official_fno"]
    candidate_rows = [row for row in repair_rows if _exact_collapse_repair_series_key(row) == "exact_collapse_candidate"]
    runtime_rows = [row for row in repair_rows if _exact_collapse_repair_series_key(row) == "exact_collapse_runtime_match"]
    for train_doc_count in train_doc_counts:
        best_fno = _best_row(
            [row for row in fno_rows if int(row.get("train_doc_count", 0) or 0) == train_doc_count],
            "test_root_mae_mean",
            "min",
        )
        candidate = _best_row(
            [row for row in candidate_rows if int(row.get("train_doc_count", 0) or 0) == train_doc_count],
            "test_root_mae_mean",
            "min",
        )
        runtime_match = _best_row(
            [row for row in runtime_rows if int(row.get("train_doc_count", 0) or 0) == train_doc_count],
            "test_root_mae_mean",
            "min",
        )
        if best_fno is not None and runtime_match is not None:
            runtime_gap = float(runtime_match.get('test_root_mae_mean')) - float(best_fno.get('test_root_mae_mean'))
        else:
            runtime_gap = float("nan")
        if best_fno is not None and candidate is not None:
            candidate_gap = float(candidate.get('test_root_mae_mean')) - float(best_fno.get('test_root_mae_mean'))
        else:
            candidate_gap = float("nan")
        if best_fno is not None:
            callouts.append(
                f"train_docs={train_doc_count}: one-tree gap {_gap_text(candidate_gap)}, runtime-match gap {_gap_text(runtime_gap)}."
            )

    fig.suptitle(f"{scope_label.title()} Exact Collapse Repair Arms")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return callouts


def _scoped_topology_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope_label: str,
    train_doc_count: int,
    prefer_unified_g: bool = True,
) -> List[Dict[str, Any]]:
    scoped_rows = [
        dict(row or {})
        for row in _completed_rows(rows)
        if str(row.get("scope_label", "") or "") == str(scope_label)
        and int(row.get("train_doc_count", 0) or 0) == int(train_doc_count)
        and _is_full_local_laws_topology_row(row)
    ]
    if prefer_unified_g and any(
        str(row.get("study_axis", "") or "") == UNIFIED_G_TOPOLOGY_STUDY_AXIS
        for row in scoped_rows
    ):
        scoped_rows = [
            row
            for row in scoped_rows
            if str(row.get("study_axis", "") or "") == UNIFIED_G_TOPOLOGY_STUDY_AXIS
        ]
    return scoped_rows


def _latest_topology_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope_label: str,
    train_doc_count: int,
    prefer_unified_g: bool = True,
    include_legacy_tree: bool = False,
) -> List[Dict[str, Any]]:
    scoped_rows = _scoped_topology_rows(
        rows,
        scope_label=scope_label,
        train_doc_count=train_doc_count,
        prefer_unified_g=prefer_unified_g,
    )
    if not include_legacy_tree:
        scoped_rows = [
            row
            for row in scoped_rows
            if _topology_curve_series_key(row) != "tree_legacy"
        ]
    if not scoped_rows:
        return []

    best_root_by_group: Dict[tuple[str, int], tuple[tuple[float, str], int, str]] = {}
    for row in scoped_rows:
        series_key = _topology_curve_series_key(row)
        if series_key == "one_tree_anchor":
            group_key = (series_key, 128)
        elif series_key == "official_fno":
            group_key = (series_key, 128)
        else:
            group_key = (series_key, int(_topology_fixed_leaf_tokens(row) or 0))
        source_root = str(row.get("source_simulation_root", "") or "")
        candidate = (
            _timestamp_sort_key(row.get("source_simulation_generated_at")),
            int(row.get("source_simulation_order", 0) or 0),
            source_root,
        )
        current = best_root_by_group.get(group_key)
        if current is None or candidate > current:
            best_root_by_group[group_key] = candidate

    filtered_rows: List[Dict[str, Any]] = []
    for row in scoped_rows:
        series_key = _topology_curve_series_key(row)
        if series_key == "one_tree_anchor":
            group_key = (series_key, 128)
        elif series_key == "official_fno":
            group_key = (series_key, 128)
        else:
            group_key = (series_key, int(_topology_fixed_leaf_tokens(row) or 0))
        row_candidate = (
            _timestamp_sort_key(row.get("source_simulation_generated_at")),
            int(row.get("source_simulation_order", 0) or 0),
            str(row.get("source_simulation_root", "") or ""),
        )
        if row_candidate == best_root_by_group.get(group_key):
            filtered_rows.append(dict(row or {}))
    return filtered_rows


def _plot_full_local_laws_topology_root(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope_label: str,
    train_doc_count: int,
    assumed_doc_tokens: int,
    output_path: Path,
) -> List[str]:
    scoped_rows = [
        dict(row or {})
        for row in _completed_rows(rows)
        if str(row.get("scope_label", "") or "") == str(scope_label)
        and int(row.get("train_doc_count", 0) or 0) == int(train_doc_count)
    ]
    topology_rows = _latest_topology_rows(
        rows,
        scope_label=scope_label,
        train_doc_count=train_doc_count,
        prefer_unified_g=True,
        include_legacy_tree=False,
    )
    anchor_rows = [
        row
        for row in scoped_rows
        if _is_exact_collapse_repair_row(row)
        and _exact_collapse_repair_series_key(row) == "exact_collapse_candidate"
        and int(_topology_fixed_leaf_tokens(row) or 0) == 128
    ]
    leaf_counts = sorted(
        {
            int(_topology_leaf_count(row, assumed_doc_tokens=assumed_doc_tokens) or 0)
            for row in topology_rows
            if str(row.get("baseline_family", "") or "") == "tree_neural"
        }
        - {0}
    )
    series_groups: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in topology_rows:
        series_groups[_topology_curve_series_key(row)].append(row)
    if anchor_rows:
        series_groups["one_tree_anchor"].extend(anchor_rows)
    fig, ax = plt.subplots(figsize=(12.4, 6.0))
    callouts: List[str] = []
    plotted_labels: set[str] = set()
    for series_key, series_rows in sorted(series_groups.items()):
        if not series_rows:
            continue
        label = _topology_curve_label(series_key)
        style = _topology_curve_style(series_key)
        if series_key == "official_fno":
            fno_rows = [
                row
                for row in series_rows
                if int(_topology_fixed_leaf_tokens(row) or 0) == 128
            ]
            if not fno_rows or not leaf_counts:
                continue
            mean_value = _mean(_safe_float(row.get("test_root_mae_mean")) for row in fno_rows)
            ax.plot(
                leaf_counts,
                [mean_value] * len(leaf_counts),
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                marker=style["marker"],
                label=label,
            )
            raw_values = [_safe_float(row.get("test_root_mae_mean")) for row in fno_rows]
            raw_xs = [
                float(min(leaf_counts)) + (raw_idx - (len(raw_values) - 1) / 2.0) * 0.05
                for raw_idx in range(len(raw_values))
            ]
            _scatter_raw_points(
                ax,
                raw_xs,
                raw_values,
                color=style["color"],
                marker=style["marker"],
                size=38.0,
            )
            continue
        point_map: DefaultDict[int, List[float]] = defaultdict(list)
        for row in series_rows:
            leaf_count = int(_topology_leaf_count(row, assumed_doc_tokens=assumed_doc_tokens) or 0)
            if leaf_count <= 0:
                continue
            point_map[leaf_count].append(_safe_float(row.get("test_root_mae_mean")))
        if not point_map:
            continue
        xs = sorted(point_map)
        ys = [_mean(point_map[x]) for x in xs]
        line_label = label if label not in plotted_labels else "_nolegend_"
        ax.plot(
            xs,
            ys,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.3,
            markersize=7.0,
            label=line_label,
        )
        plotted_labels.add(label)
        for x in xs:
            raw_values = point_map[x]
            raw_xs = [
                float(x) + (raw_idx - (len(raw_values) - 1) / 2.0) * 0.09
                for raw_idx in range(len(raw_values))
            ]
            _scatter_raw_points(
                ax,
                raw_xs,
                raw_values,
                color=style["color"],
                marker=style["marker"],
                size=34.0,
            )
    if leaf_counts:
        ax.set_xticks(leaf_counts)
        ax.set_xlim(min(leaf_counts) - 0.4, max(leaf_counts) + 0.6)
    ax.set_xlabel("Leaves per document")
    ax.set_ylabel("Root MAE")
    ax.set_title("Topology root-MAE vs leaves")
    ax.grid(alpha=0.25, axis="both")
    ax.legend(loc="best", fontsize=9)
    fig.suptitle(f"{scope_label.title()} Topology @ train_docs={int(train_doc_count)}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    series_map: Dict[str, Dict[int, float]] = {}
    for series_key, series_rows in series_groups.items():
        point_map: DefaultDict[int, List[float]] = defaultdict(list)
        if series_key == "official_fno":
            fno_rows = [
                row
                for row in series_rows
                if int(_topology_fixed_leaf_tokens(row) or 0) == 128
            ]
            mean_value = _mean(_safe_float(row.get("test_root_mae_mean")) for row in fno_rows)
            if leaf_counts and _is_finite(mean_value):
                point_map[min(leaf_counts)] = [mean_value]
                point_map[max(leaf_counts)] = [mean_value]
        else:
            for row in series_rows:
                leaf_count = int(_topology_leaf_count(row, assumed_doc_tokens=assumed_doc_tokens) or 0)
                if leaf_count > 0:
                    point_map[leaf_count].append(_safe_float(row.get("test_root_mae_mean")))
        series_map[_topology_curve_label(series_key)] = {
            leaf_count: _mean(values)
            for leaf_count, values in sorted(point_map.items())
            if values
        }
    anchor_points = series_map.get("One-tree repaired anchor", {})
    anchor_value = anchor_points.get(1, float("nan"))
    if _is_finite(anchor_value):
        callouts.append(f"One-tree repaired anchor @ {_leaf_count_text(1)}: {_metric_text(anchor_value)}.")
    fno_by_leaf = series_map.get("Official FNO anchor", {})
    fno_value = next(iter(fno_by_leaf.values()), float("nan")) if fno_by_leaf else float("nan")
    unified_by_leaf = series_map.get("Tree (unified_g)", {})
    if _is_finite(fno_value) and _is_finite(unified_by_leaf.get(1, float("nan"))):
        callouts.append(
            f"Tree (unified_g) @ {_leaf_count_text(1)} gap vs FNO: {_gap_text(unified_by_leaf[1] - fno_value)}."
        )
    if _is_finite(fno_value):
        best_deep_leaf = max(unified_by_leaf) if unified_by_leaf else 0
        deep_value = unified_by_leaf.get(best_deep_leaf, float("nan"))
        if best_deep_leaf > 0 and _is_finite(deep_value):
            callouts.append(
                f"Tree (unified_g) @ {_leaf_count_text(best_deep_leaf)} gap vs FNO: {_gap_text(deep_value - fno_value)}."
            )
    for prev_leaf, next_leaf in zip(sorted(unified_by_leaf), sorted(unified_by_leaf)[1:]):
        prev_value = unified_by_leaf.get(prev_leaf, float("nan"))
        next_value = unified_by_leaf.get(next_leaf, float("nan"))
        if _is_finite(prev_value) and _is_finite(next_value):
            callouts.append(
                f"Tree (unified_g) @ {_leaf_count_text(next_leaf)} gap vs {_leaf_count_text(prev_leaf)}: {_gap_text(next_value - prev_value)}."
            )
            break
    return callouts[:4]


def _plot_topology_ladder_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope_label: str,
    train_doc_count: int,
    assumed_doc_tokens: int,
    output_path: Path,
) -> List[str]:
    topology_rows = [
        row
        for row in _latest_topology_rows(
            rows,
            scope_label=scope_label,
            train_doc_count=train_doc_count,
            prefer_unified_g=True,
            include_legacy_tree=False,
        )
        if str(row.get("baseline_family", "") or "") == "tree_neural"
    ]
    leaf_tokens = sorted(
        {int(_topology_fixed_leaf_tokens(row) or 0) for row in topology_rows},
        reverse=True,
    )
    metrics = [
        ("test_root_mae_mean", "Root MAE"),
        ("test_leaf_mae_mean", "Leaf MAE"),
        ("test_merge_mae_mean", "Merge MAE"),
        ("test_exact_match_rate_mean", "Exact Match"),
    ]
    width = 0.8 / max(1, len(leaf_tokens))
    xs = list(range(len(metrics)))
    tree_hatches = ["", "\\\\", "xx", "//", ".."]
    fig, ax = plt.subplots(figsize=(11.4, 5.8))
    callouts: List[str] = []
    for series_idx, leaf_token in enumerate(leaf_tokens):
        series_rows = [
            row
            for row in topology_rows
            if int(_topology_fixed_leaf_tokens(row) or 0) == int(leaf_token)
        ]
        offsets = [
            float(x) + (series_idx - (len(leaf_tokens) - 1) / 2.0) * width
            for x in xs
        ]
        means = [
            _mean(_safe_float(row.get(metric_key)) for row in series_rows)
            for metric_key, _ in metrics
        ]
        ax.bar(
            offsets,
            means,
            width=width,
            color=FAMILY_PALETTE["tree_neural"],
            hatch=tree_hatches[series_idx % len(tree_hatches)],
            edgecolor="#0f172a",
            linewidth=0.8,
            alpha=0.9,
            label=f"Tree ({int(leaf_token)})",
        )
        for metric_idx, (metric_key, _) in enumerate(metrics):
            raw_values = [_safe_float(row.get(metric_key)) for row in series_rows]
            raw_xs = [
                offsets[metric_idx] + (raw_idx - (len(raw_values) - 1) / 2.0) * 0.04
                for raw_idx in range(len(raw_values))
            ]
            _scatter_raw_points(
                ax,
                raw_xs,
                raw_values,
                color=FAMILY_PALETTE["tree_neural"],
                marker="s",
                size=30.0,
            )
    ax.set_xticks(xs)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Metric value")
    ax.set_title("Topology ladder diagnostics")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"{scope_label.title()} Topology Ladder @ train_docs={int(train_doc_count)}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    metric_by_leaf: Dict[int, Dict[str, float]] = {}
    for leaf_token in leaf_tokens:
        series_rows = [
            row
            for row in topology_rows
            if int(_topology_fixed_leaf_tokens(row) or 0) == int(leaf_token)
        ]
        metric_by_leaf[int(leaf_token)] = {
            metric_key: _mean(_safe_float(row.get(metric_key)) for row in series_rows)
            for metric_key, _ in metrics
        }
    for prev_leaf, next_leaf in zip(leaf_tokens, leaf_tokens[1:]):
        prev_root = metric_by_leaf.get(int(prev_leaf), {}).get("test_root_mae_mean", float("nan"))
        next_root = metric_by_leaf.get(int(next_leaf), {}).get("test_root_mae_mean", float("nan"))
        if _is_finite(prev_root) and _is_finite(next_root):
            callouts.append(
                f"Tree ({int(next_leaf)}) root-MAE gap vs Tree ({int(prev_leaf)}): {_gap_text(next_root - prev_root)}."
            )
    exact_32 = metric_by_leaf.get(32, {}).get("test_exact_match_rate_mean", float("nan"))
    if _is_finite(exact_32):
        leaf_count_32 = int(assumed_doc_tokens // 32) if assumed_doc_tokens > 0 and assumed_doc_tokens % 32 == 0 else 4
        callouts.append(f"Tree ({_leaf_count_text(leaf_count_32)} / 32 tokens) exact match: {_metric_text(exact_32)}.")
    return callouts[:4]


def _render_markdown(summary: Mapping[str, Any]) -> List[str]:
    lines: List[str] = [
        "# Self-Contained Markov Parity Report",
        "",
        f"Generated: `{summary.get('generated_at', '')}`",
        "",
        "## Data Contract",
        f"- {summary.get('data_contract', {}).get('description', '')}",
        "",
        "## Status",
    ]
    source_files = dict(summary.get("source_files") or {})
    simulation_roots = [
        str(item)
        for item in list(source_files.get("simulation_roots") or [])
        if str(item).strip()
    ]
    if simulation_roots:
        lines.append(f"- simulation roots: `{', '.join(simulation_roots)}`")
    else:
        lines.append(f"- simulation root: `{source_files.get('simulation_root', '')}`")
    lines.append("")
    status = dict(summary.get("status") or {})
    lines.extend(
        [
            f"- state: `{status.get('state', '')}`",
            f"- evidence_status: `{status.get('evidence_status', '')}`",
            f"- completed: `{status.get('completed_items', 0)}` / `{status.get('items_total', 0)}`",
            f"- failures: `{status.get('failed_items', 0)}`",
            f"- source runs: `{status.get('source_run_count', 1)}`",
            "",
            "## Coverage",
        ]
    )
    coverage = dict(summary.get("coverage") or {})
    lines.extend(
        [
            f"- scopes: `{', '.join(str(v) for v in coverage.get('scopes', []))}`",
            f"- train_docs: `{', '.join(str(int(v)) for v in coverage.get('train_doc_counts', []))}`",
            f"- claim_levels: `{', '.join(str(v) for v in coverage.get('claim_levels', []))}`",
            f"- assumed_doc_tokens: `{coverage.get('assumed_doc_tokens', 0)}`",
            f"- one_leaf_target_fixed_leaf_tokens: `{coverage.get('one_leaf_target_fixed_leaf_tokens', 0)}`",
            "",
            "## Palette",
        ]
    )
    for family, color in dict(summary.get("palette", {}).get("families", {})).items():
        lines.append(f"- `{family}`: `{color}`")
    lines.extend(["", "## Figures"])
    figure_order = list(summary.get("figure_order") or [])
    figure_inventory = dict(summary.get("figure_inventory") or {})
    for title in figure_order:
        entry = dict(figure_inventory.get(title) or {})
        lines.extend(
            [
                "",
                f"### {title}",
                f"- figure: `{entry.get('path', '')}`",
                f"- kind: `{entry.get('figure_kind', '')}`",
                f"- chart_style: `{entry.get('chart_style', '')}`",
            ]
        )
        if str(entry.get("scope_label", "")):
            lines.append(f"- scope: `{entry.get('scope_label', '')}`")
        if entry.get("train_doc_count") is not None:
            lines.append(f"- train_docs: `{int(entry.get('train_doc_count', 0) or 0)}`")
        for callout in list(entry.get("callouts") or []):
            lines.append(f"- {callout}")
    lines.extend(
        [
            "",
            "## Source Files",
        ]
    )
    manifests = [str(item) for item in list(source_files.get("manifests") or []) if str(item).strip()]
    summaries = [str(item) for item in list(source_files.get("summaries") or []) if str(item).strip()]
    scheduler_statuses = [
        str(item)
        for item in list(source_files.get("scheduler_statuses") or [])
        if str(item).strip()
    ]
    if manifests:
        for item in manifests:
            lines.append(f"- manifest: `{item}`")
    else:
        lines.append(f"- manifest: `{source_files.get('manifest', '')}`")
    if summaries:
        for item in summaries:
            lines.append(f"- summary: `{item}`")
    else:
        lines.append(f"- summary: `{source_files.get('summary', '')}`")
    if scheduler_statuses:
        for item in scheduler_statuses:
            lines.append(f"- scheduler_status: `{item}`")
    else:
        lines.append(f"- scheduler_status: `{source_files.get('scheduler_status', '')}`")
    return lines


def _write_pdf(output_pdf: Path, summary: Mapping[str, Any], md_lines: Sequence[str]) -> None:
    with PdfPages(output_pdf) as pdf:
        write_text_page(
            pdf,
            title="Self-Contained Markov Parity Report",
            lines=md_lines[: min(len(md_lines), 60)],
        )
        figure_inventory = dict(summary.get("figure_inventory") or {})
        for title in list(summary.get("figure_order") or []):
            entry = dict(figure_inventory.get(title) or {})
            figure_path = Path(str(entry.get("path", "") or ""))
            if figure_path.exists():
                write_image_page(pdf, image_path=figure_path, title=title)


def _validate_row_coverage(summary: Mapping[str, Any]) -> None:
    row_figure_coverage = dict(summary.get("row_figure_coverage") or {})
    normalized_rows = [dict(row or {}) for row in list(summary.get("normalized_rows") or [])]
    missing_quality: List[str] = []
    missing_runtime: List[str] = []
    for row in normalized_rows:
        job_name = str(row.get("job_name", "") or "")
        if not job_name or str(row.get("state", "completed") or "completed") != "completed":
            continue
        coverage = dict(row_figure_coverage.get(job_name) or {})
        if not list(coverage.get("quality_figures") or []):
            missing_quality.append(job_name)
        has_runtime = any(
            _is_finite(row.get(metric_key))
            for metric_key, _, _, _ in RUNTIME_METRICS
        )
        if has_runtime and not list(coverage.get("runtime_figures") or []):
            missing_runtime.append(job_name)
    if missing_quality or missing_runtime:
        parts: List[str] = []
        if missing_quality:
            parts.append(f"missing quality coverage for {sorted(missing_quality)}")
        if missing_runtime:
            parts.append(f"missing runtime coverage for {sorted(missing_runtime)}")
        raise RuntimeError("; ".join(parts))


def _emit_figures(summary: MutableMapping[str, Any], output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for stale in figures_dir.glob("*.png"):
        stale.unlink()

    normalized_rows = [dict(row or {}) for row in list(summary.get("normalized_rows") or [])]
    completed_rows = [row for row in normalized_rows if str(row.get("state", "completed") or "completed") == "completed"]
    scopes = list(summary.get("coverage", {}).get("scopes") or [])
    train_doc_counts = list(summary.get("coverage", {}).get("train_doc_counts") or [])
    assumed_doc_tokens = int(summary.get("coverage", {}).get("assumed_doc_tokens", 0) or 0)
    one_leaf_target_fixed_leaf_tokens = int(summary.get("coverage", {}).get("one_leaf_target_fixed_leaf_tokens", 0) or 0)

    for scope_label in scopes:
        scoped_rows = [row for row in completed_rows if str(row.get("scope_label", "")) == str(scope_label)]
        if not scoped_rows:
            continue
        quality_overview_path = figures_dir / f"{_slugify(scope_label)}_quality_overview.png"
        quality_title = f"{str(scope_label).title()} Quality Overview"
        quality_callouts = _plot_overview(
            scoped_rows,
            scope_label=str(scope_label),
            metric_defs=QUALITY_METRICS,
            output_path=quality_overview_path,
            is_runtime=False,
        )
        _register_figure(
            summary,
            title=quality_title,
            path=quality_overview_path,
            figure_kind="quality_overview",
            chart_style="overview_triptych",
            job_names=[str(row.get("job_name", "") or "") for row in scoped_rows],
            callouts=quality_callouts,
            scope_label=str(scope_label),
        )

        runtime_overview_path = figures_dir / f"{_slugify(scope_label)}_runtime_overview.png"
        runtime_title = f"{str(scope_label).title()} Runtime Overview"
        runtime_callouts = _plot_overview(
            scoped_rows,
            scope_label=str(scope_label),
            metric_defs=RUNTIME_METRICS,
            output_path=runtime_overview_path,
            is_runtime=True,
        )
        _register_figure(
            summary,
            title=runtime_title,
            path=runtime_overview_path,
            figure_kind="runtime_overview",
            chart_style="overview_panel",
            job_names=[str(row.get("job_name", "") or "") for row in scoped_rows],
            callouts=runtime_callouts,
            scope_label=str(scope_label),
        )

        for train_doc_count in train_doc_counts:
            empirical_rows = [
                row
                for row in completed_rows
                if str(row.get("scope_label", "")) == str(scope_label)
                and int(row.get("train_doc_count", 0) or 0) == int(train_doc_count)
                and str(row.get("claim_level", "")) == "empirical_geometry"
            ]
            exact_rows = [
                row
                for row in completed_rows
                if str(row.get("scope_label", "")) == str(scope_label)
                and int(row.get("train_doc_count", 0) or 0) == int(train_doc_count)
                and str(row.get("claim_level", "")) == "exact_collapse_candidate"
            ]
            if not empirical_rows and not exact_rows:
                continue
            chart_style = _slice_chart_style(empirical_rows)
            slice_job_names = [str(row.get("job_name", "") or "") for row in empirical_rows + exact_rows]
            if chart_style == "geometry":
                geometry_path = figures_dir / f"{_slugify(scope_label)}_quality_geometry_train_docs_{int(train_doc_count)}.png"
                geometry_title = f"{str(scope_label).title()} Quality Geometry @ train_docs={int(train_doc_count)}"
                geometry_callouts = _plot_geometry_triptych(
                    empirical_rows,
                    exact_rows,
                    scope_label=str(scope_label),
                    train_doc_count=int(train_doc_count),
                    one_leaf_target_fixed_leaf_tokens=one_leaf_target_fixed_leaf_tokens,
                    metric_defs=QUALITY_METRICS,
                    output_path=geometry_path,
                )
                _register_figure(
                    summary,
                    title=geometry_title,
                    path=geometry_path,
                    figure_kind="quality_slice",
                    chart_style="geometry_triptych",
                    job_names=slice_job_names,
                    callouts=geometry_callouts,
                    scope_label=str(scope_label),
                    train_doc_count=int(train_doc_count),
                )

                runtime_path = figures_dir / f"{_slugify(scope_label)}_runtime_geometry_train_docs_{int(train_doc_count)}.png"
                runtime_title = f"{str(scope_label).title()} Runtime Geometry @ train_docs={int(train_doc_count)}"
                runtime_callouts = _plot_runtime_geometry(
                    empirical_rows,
                    exact_rows,
                    scope_label=str(scope_label),
                    train_doc_count=int(train_doc_count),
                    one_leaf_target_fixed_leaf_tokens=one_leaf_target_fixed_leaf_tokens,
                    output_path=runtime_path,
                )
                _register_figure(
                    summary,
                    title=runtime_title,
                    path=runtime_path,
                    figure_kind="runtime_slice",
                    chart_style="geometry_panel",
                    job_names=slice_job_names,
                    callouts=runtime_callouts,
                    scope_label=str(scope_label),
                    train_doc_count=int(train_doc_count),
                )

            one_leaf_rows = [
                row
                for row in empirical_rows
                if bool(row.get("one_leaf_target", False))
                or str(row.get("baseline_family", "")) in {"official_fno", "official_fno_sumlen"}
            ] + list(exact_rows)
            if one_leaf_rows:
                one_leaf_path = figures_dir / f"{_slugify(scope_label)}_quality_comparison_train_docs_{int(train_doc_count)}.png"
                one_leaf_title = f"{str(scope_label).title()} Quality Comparison @ train_docs={int(train_doc_count)}"
                one_leaf_callouts = _plot_comparison_triptych(
                    one_leaf_rows,
                    scope_label=str(scope_label),
                    train_doc_count=int(train_doc_count),
                    metric_defs=QUALITY_METRICS,
                    output_path=one_leaf_path,
                    title_suffix="Quality Comparison",
                )
                _register_figure(
                    summary,
                    title=one_leaf_title,
                    path=one_leaf_path,
                    figure_kind="quality_slice",
                    chart_style="comparison_triptych",
                    job_names=[str(row.get("job_name", "") or "") for row in one_leaf_rows],
                    callouts=one_leaf_callouts,
                    scope_label=str(scope_label),
                    train_doc_count=int(train_doc_count),
                )
                if chart_style != "geometry":
                    runtime_path = figures_dir / f"{_slugify(scope_label)}_runtime_comparison_train_docs_{int(train_doc_count)}.png"
                    runtime_title = f"{str(scope_label).title()} Runtime Comparison @ train_docs={int(train_doc_count)}"
                    runtime_callouts = _plot_runtime_comparison(
                        one_leaf_rows,
                        scope_label=str(scope_label),
                        train_doc_count=int(train_doc_count),
                        output_path=runtime_path,
                    )
                    _register_figure(
                        summary,
                        title=runtime_title,
                        path=runtime_path,
                        figure_kind="runtime_slice",
                        chart_style="comparison_panel",
                        job_names=[str(row.get("job_name", "") or "") for row in one_leaf_rows],
                        callouts=runtime_callouts,
                        scope_label=str(scope_label),
                        train_doc_count=int(train_doc_count),
                    )

        if any(str(row.get("claim_level", "")) == "exact_collapse_candidate" for row in scoped_rows):
            exact_path = figures_dir / f"{_slugify(scope_label)}_exact_collapse_vs_fno.png"
            exact_title = f"{str(scope_label).title()} Exact Collapse vs Best FNO"
            exact_callouts = _plot_exact_collapse_vs_fno(completed_rows, scope_label=str(scope_label), output_path=exact_path)
            exact_job_names = [
                str(row.get("job_name", "") or "")
                for row in scoped_rows
                if str(row.get("claim_level", "")) == "exact_collapse_candidate"
            ]
            _register_figure(
                summary,
                title=exact_title,
                path=exact_path,
                figure_kind="quality_exact_collapse",
                chart_style="exact_collapse_gap_panel",
                job_names=exact_job_names,
                callouts=exact_callouts,
                scope_label=str(scope_label),
            )

        repair_rows = [
            row
            for row in scoped_rows
            if _is_exact_collapse_repair_row(row)
            and _exact_collapse_repair_series_key(row) != "exact_collapse_legacy_control"
        ]
        if repair_rows:
            repair_path = figures_dir / f"{_slugify(scope_label)}_exact_collapse_repair_arms.png"
            repair_title = f"{str(scope_label).title()} Exact Collapse Repair Arms"
            repair_callouts = _plot_exact_collapse_repair_arms(
                completed_rows,
                scope_label=str(scope_label),
                output_path=repair_path,
            )
            _register_figure(
                summary,
                title=repair_title,
                path=repair_path,
                figure_kind="quality_exact_collapse_repair",
                chart_style="repair_panel",
                job_names=[str(row.get("job_name", "") or "") for row in repair_rows],
                callouts=repair_callouts,
                scope_label=str(scope_label),
            )

        topology_rows = _latest_topology_rows(
            completed_rows,
            scope_label=str(scope_label),
            train_doc_count=4096,
            prefer_unified_g=True,
            include_legacy_tree=False,
        )
        topology_anchor_rows = [
            row
            for row in scoped_rows
            if _is_exact_collapse_repair_row(row)
            and _exact_collapse_repair_series_key(row) == "exact_collapse_candidate"
            and int(row.get("train_doc_count", 0) or 0) == 4096
            and int(row.get("fixed_leaf_tokens", 0) or 0) == 128
        ]
        if str(scope_label) == "recoverable" and (topology_rows or topology_anchor_rows):
            topology_root_path = figures_dir / f"{_slugify(scope_label)}_topology_root_4096.png"
            topology_root_title = (
                f"{str(scope_label).title()} Full Local Laws Topology @ train_docs=4096"
            )
            topology_root_callouts = _plot_full_local_laws_topology_root(
                completed_rows,
                scope_label=str(scope_label),
                train_doc_count=4096,
                assumed_doc_tokens=assumed_doc_tokens,
                output_path=topology_root_path,
            )
            _register_figure(
                summary,
                title=topology_root_title,
                path=topology_root_path,
                figure_kind="quality_topology_root",
                chart_style="topology_curve",
                job_names=[
                    str(row.get("job_name", "") or "")
                    for row in topology_rows + topology_anchor_rows
                ],
                callouts=topology_root_callouts,
                scope_label=str(scope_label),
                train_doc_count=4096,
            )

            if topology_rows:
                topology_ladder_path = figures_dir / f"{_slugify(scope_label)}_topology_ladder_4096.png"
                topology_ladder_title = (
                    f"{str(scope_label).title()} Topology Ladder Diagnostics @ train_docs=4096"
                )
                topology_ladder_callouts = _plot_topology_ladder_diagnostics(
                    completed_rows,
                    scope_label=str(scope_label),
                    train_doc_count=4096,
                    assumed_doc_tokens=assumed_doc_tokens,
                    output_path=topology_ladder_path,
                )
                _register_figure(
                    summary,
                    title=topology_ladder_title,
                    path=topology_ladder_path,
                    figure_kind="quality_topology_ladder",
                    chart_style="metric_bars",
                    job_names=[
                        str(row.get("job_name", "") or "")
                        for row in topology_rows
                        if str(row.get("baseline_family", "") or "") == "tree_neural"
                    ],
                    callouts=topology_ladder_callouts,
                    scope_label=str(scope_label),
                    train_doc_count=4096,
                )

    _validate_row_coverage(summary)


def main() -> None:
    args = _parse_args()
    simulation_roots = [Path(root).expanduser().resolve() for root in list(args.simulation_root or [])]
    payload = _load_parity_payloads(simulation_roots)
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif len(simulation_roots) == 1:
        output_dir = simulation_roots[0] / "self_contained_report"
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "outputs" / f"markov_parity_multi_root_report_{stamp}"
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _build_report_summary(payload)
    _emit_figures(summary, output_dir)
    md_lines = _render_markdown(summary)

    summary_path = output_dir / "summary.json"
    report_md_path = output_dir / "report.md"
    report_pdf_path = output_dir / "report.pdf"

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    _write_pdf(report_pdf_path, summary, md_lines)


if __name__ == "__main__":
    main()
