#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from collections.abc import Mapping as MappingABC
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.report.pdf_utils import write_image_page, write_text_page
from src.ctreepo.sim.core.markov_alignment_validation import (
    build_markov_alignment_audit_report,
    write_markov_alignment_audit_report,
)
from src.ctreepo.sim.core.markov_v3_row_contract import (
    annotate_downstream_v3_row,
    filtered_headline_rows,
    filtered_quarantined_rows,
    is_headline_contract_status,
    quarantine_sources_from_rows,
)
from src.ctreepo.sim.util import safe_float, safe_int


def _mass_matched_rate_suffix(rate_percent: float) -> str:
    return f"{float(rate_percent):.1f}".replace(".", "p")


def _mass_matched_package_name(root_share: int, rate_percent: float) -> str:
    return f"r{int(root_share)}_mass_local_eq_{_mass_matched_rate_suffix(rate_percent)}"


def _mass_matched_package_order(
    root_share: int,
    local_rate_percents: Sequence[float],
) -> tuple[str, ...]:
    return (
        f"full{int(root_share)}",
        *[
            _mass_matched_package_name(int(root_share), float(rate_percent))
            for rate_percent in local_rate_percents
        ],
    )


def _mass_matched_tick_labels(
    ladders: Mapping[int, Sequence[float]],
) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for root_share, rates in ladders.items():
        for rate_percent in rates:
            labels[_mass_matched_package_name(int(root_share), float(rate_percent))] = (
                f"R{int(root_share)}mm+{float(rate_percent):.1f}"
            )
    return labels


REPORT_VERSION_MANIFEST_NAME = "report_version_manifest.json"
REPORT_SOURCE_SPECS: Dict[str, Dict[str, str]] = {
    "learnability_summary": {"phase": "learnability", "arg": "learnability_summary"},
    "weight_ablation_summary": {"phase": "weight_ablation", "arg": "weight_ablation_summary"},
    "law_comparison_json": {"phase": "law_packages", "arg": "law_comparison_json"},
    "support_summary": {"phase": "support_grid", "arg": "support_summary"},
    "batch_timing_summary": {"phase": "batch_timing", "arg": "batch_timing_summary"},
    "medium_grid_summary": {"phase": "medium_grid", "arg": "medium_grid_summary"},
    "docs_epochs_summary": {"phase": "docs_epochs", "arg": "docs_epochs_summary"},
    "fno_upper_bound_summary": {"phase": "full_doc_anchor", "arg": "fno_upper_bound_summary"},
    "oracle_budget_frontier_summary": {"phase": "oracle_budget_frontier", "arg": "oracle_budget_frontier_summary"},
    "efficiency_suite_summary": {"phase": "efficiency_suite", "arg": "efficiency_suite_summary"},
    "large_batch_diagnosis_summary": {"phase": "large_batch_diagnosis", "arg": "large_batch_diagnosis_summary"},
    "supervision_sweep_summary": {"phase": "supervision_sweep", "arg": "supervision_sweep_summary"},
    "supervision_recovery_summary": {"phase": "supervision_recovery", "arg": "supervision_recovery_summary"},
}
CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES = ("official_fno", "official_fno_sumlen")
EFFICIENCY_TREE_BASELINE_FAMILIES = ("tree_neural_c2", "tree_neural")
SUPERVISION_RECOVERY_TREE_FAMILY = "tree_neural"
SUPERVISION_RECOVERY_RECOVERABLE_BENCHMARK = "recoverable_v5"
SUPERVISION_RECOVERY_STRUCTURAL_GRID = "structural_core_v2"
SUPERVISION_RECOVERY_STRUCTURAL_CELL = "r12_p079"
SUPERVISION_RECOVERY_CANONICAL_TREE_SELECTION_METRIC = "val_root_mae"
SUPERVISION_RECOVERY_CANONICAL_TREE_STAGE1_SELECTION_METRIC = "val_theorem_bootstrap_direct"
SUPERVISION_RECOVERY_CANONICAL_COMPARISON_RULE = (
    "all tree ladder points selected on val_root_mae; local metrics are diagnostics"
)
SUPERVISION_RECOVERY_PACKAGE_ORDER = (
    "full100",
    "full50",
    "full30",
    "full20",
    "full10",
    "full10_leaf_count100",
    "full10_leaf_full100",
    "full10_leaf_full100_internal_depth1_count100",
    "full10_leaf_full100_internal_depth2_count100",
    "full10_leaf_full100_internal_count100",
    "full20_leaf_full100_internal_count100",
    "full30_leaf_full100_internal_count100",
    "full50_leaf_full100_internal_count100",
)
SUPERVISION_RECOVERY_ROOT_ONLY_PACKAGE_ORDER = (
    "full10",
    "full20",
    "full30",
    "full50",
    "full60",
    "full70",
    "full80",
    "full90",
    "full100",
)
SUPERVISION_RECOVERY_RECOVERABLE_SCOPE_FAMILY = (
    "recoverable_v5_t128",
    "recoverable_v4_t128",
    "recoverable_v5",
    "recoverable_v4",
)
SUPERVISION_RECOVERY_ROOT_LOCAL_PACKAGE_ORDER = (
    "full0_leaf_full100_internal_count100",
    "full10_leaf_full100_internal_count100",
    "full20_leaf_full100_internal_count100",
    "full30_leaf_full100_internal_count100",
    "full50_leaf_full100_internal_count100",
    "full60_leaf_full100_internal_count100",
    "full70_leaf_full100_internal_count100",
    "full80_leaf_full100_internal_count100",
    "full90_leaf_full100_internal_count100",
    "full100_leaf_full100_internal_count100",
)
SUPERVISION_RECOVERY_R10_LOCAL_LAW_RATE_PACKAGE_ORDER = (
    "full10",
    "full10_leaf_count10_internal_count10",
    "full10_leaf_count20_internal_count20",
    "full10_leaf_count50_internal_count50",
    "full10_leaf_count100_internal_count100",
)
SUPERVISION_RECOVERY_R20_LOCAL_LAW_RATE_PACKAGE_ORDER = (
    "full20",
    "full20_leaf_count10_internal_count10",
    "full20_leaf_count20_internal_count20",
    "full20_leaf_count50_internal_count50",
    "full20_leaf_count100_internal_count100",
)
SUPERVISION_RECOVERY_MASS_MATCHED_RATE_LADDERS: Dict[int, Sequence[float]] = {
    10: (0.5, 1.0, 1.5, 2.0),
    20: (1.0, 2.0, 3.0, 4.0),
    80: (5.0, 10.0, 15.0, 16.0),
    90: (5.0, 10.0, 15.0, 18.0),
    100: (5.0, 10.0, 15.0, 20.0),
}
SUPERVISION_RECOVERY_R10_MASS_MATCHED_PACKAGE_ORDER = _mass_matched_package_order(
    10,
    SUPERVISION_RECOVERY_MASS_MATCHED_RATE_LADDERS[10],
)
SUPERVISION_RECOVERY_R20_MASS_MATCHED_PACKAGE_ORDER = _mass_matched_package_order(
    20,
    SUPERVISION_RECOVERY_MASS_MATCHED_RATE_LADDERS[20],
)
SUPERVISION_RECOVERY_R80_MASS_MATCHED_PACKAGE_ORDER = _mass_matched_package_order(
    80,
    SUPERVISION_RECOVERY_MASS_MATCHED_RATE_LADDERS[80],
)
SUPERVISION_RECOVERY_R90_MASS_MATCHED_PACKAGE_ORDER = _mass_matched_package_order(
    90,
    SUPERVISION_RECOVERY_MASS_MATCHED_RATE_LADDERS[90],
)
SUPERVISION_RECOVERY_R100_MASS_MATCHED_PACKAGE_ORDER = _mass_matched_package_order(
    100,
    SUPERVISION_RECOVERY_MASS_MATCHED_RATE_LADDERS[100],
)
SUPERVISION_RECOVERY_LOCAL_LAW_RATE_PACKAGE_ORDERS: Dict[int, Sequence[str]] = {
    10: SUPERVISION_RECOVERY_R10_LOCAL_LAW_RATE_PACKAGE_ORDER,
    20: SUPERVISION_RECOVERY_R20_LOCAL_LAW_RATE_PACKAGE_ORDER,
}
SUPERVISION_RECOVERY_MASS_MATCHED_RATE_PACKAGE_ORDERS: Dict[int, Sequence[str]] = {
    10: SUPERVISION_RECOVERY_R10_MASS_MATCHED_PACKAGE_ORDER,
    20: SUPERVISION_RECOVERY_R20_MASS_MATCHED_PACKAGE_ORDER,
    80: SUPERVISION_RECOVERY_R80_MASS_MATCHED_PACKAGE_ORDER,
    90: SUPERVISION_RECOVERY_R90_MASS_MATCHED_PACKAGE_ORDER,
    100: SUPERVISION_RECOVERY_R100_MASS_MATCHED_PACKAGE_ORDER,
}
TREE_PRIMARY_COLOR = "#16a34a"
TREE_LOCAL_COLOR = "#16a34a"
FNO_OFFICIAL_COLOR = "#dc2626"
FNO_SUMLEN_COLOR = "#f59e0b"
NEUTRAL_COLOR = "#64748b"
BEST_FULL_ROOT_CEILING_COLOR = NEUTRAL_COLOR
MASS_MATCHED_OVERLAY_TREE_COLOR = TREE_PRIMARY_COLOR
MASS_MATCHED_OVERLAY_OFFICIAL_FNO_COLOR = FNO_OFFICIAL_COLOR
MASS_MATCHED_OVERLAY_FNO_SUMLEN_COLOR = FNO_SUMLEN_COLOR
MASS_MATCHED_OVERLAY_LINESTYLES = {
    10: "-",
    20: ":",
    80: (0, (6, 2)),
    90: (0, (3, 1, 1, 1)),
    100: "--",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unify existing Markov optimization tradeoff artifacts into one report."
    )
    parser.add_argument("--version-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--learnability-summary", type=Path, default=None)
    parser.add_argument("--weight-ablation-summary", type=Path, default=None)
    parser.add_argument("--law-comparison-json", type=Path, default=None)
    parser.add_argument("--support-summary", type=Path, default=None)
    parser.add_argument("--batch-timing-summary", type=Path, default=None)
    parser.add_argument("--medium-grid-summary", type=Path, default=None)
    parser.add_argument("--docs-epochs-summary", type=Path, default=None)
    parser.add_argument("--fno-upper-bound-summary", type=Path, default=None)
    parser.add_argument("--oracle-budget-frontier-summary", type=Path, default=None)
    parser.add_argument("--efficiency-suite-summary", type=Path, default=None)
    parser.add_argument("--large-batch-diagnosis-summary", type=Path, default=None)
    parser.add_argument("--supervision-sweep-summary", type=Path, default=None)
    parser.add_argument(
        "--supervision-recovery-summary",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--supervision-recovery-ceiling-summary", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--pdf-path", type=Path, default=None)
    parser.add_argument(
        "--report-profile",
        choices=("supervision_recovery_v1", "r10_coverage_focused", "exact_parity_canary"),
        default="supervision_recovery_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / f"markov_optimization_tradeoff_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    return parser.parse_args()


_safe_float = safe_float
_safe_int = safe_int


def _effective_leaves_per_doc(row: Mapping[str, Any]) -> int:
    executed_leaves = int(_safe_int(row.get("executed_leaves_per_doc"), 0))
    if executed_leaves > 0:
        return executed_leaves
    leaves = int(_safe_int(row.get("leaves_per_doc"), 0))
    if leaves > 0:
        return leaves
    doc_tokens = int(_safe_int(row.get("computed_assumed_doc_tokens"), 0))
    leaf_tokens = int(
        _safe_int(
            row.get("executed_fixed_leaf_tokens", row.get("fixed_leaf_tokens")),
            0,
        )
    )
    if doc_tokens > 0 and leaf_tokens > 0:
        return max(1, int(math.ceil(float(doc_tokens) / float(leaf_tokens))))
    return 0


def _effective_fixed_leaf_tokens(row: Mapping[str, Any]) -> int:
    return int(
        _safe_int(
            row.get("executed_fixed_leaf_tokens", row.get("fixed_leaf_tokens")),
            0,
        )
    )


def _tree_root_mae_from_family_row(row: Mapping[str, Any]) -> float:
    return _safe_float(
        row.get("tree_test_root_mae"),
        _safe_float(row.get("test_root_mae_mean"), float("nan")),
    )


def _requested_fixed_leaf_tokens(row: Mapping[str, Any]) -> int:
    return int(
        _safe_int(
            row.get("requested_fixed_leaf_tokens", row.get("fixed_leaf_tokens")),
            0,
        )
    )


def _row_intent_discriminator(row: Mapping[str, Any]) -> str:
    """Discriminator for rows with same family/geometry but different intent."""
    explicit = str(row.get("run_intent_hash", "") or "").strip()
    if explicit:
        return explicit
    parts: List[str] = []
    for field in (
        "c1_relative_weight",
        "c2_relative_weight",
        "c3_relative_weight",
        "depth_discount_gamma",
        "local_law_weight",
        "task_objective_weight",
    ):
        val = row.get(field)
        if val not in {"", None}:
            parts.append(f"{field}={val}")
    return "|".join(parts)


def _row_geometry_identity(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("supervision_recovery_geometry_key", "") or "").strip()
    if explicit:
        return explicit
    requested = _requested_fixed_leaf_tokens(row)
    executed = _effective_fixed_leaf_tokens(row)
    leaves = _effective_leaves_per_doc(row)
    doc_tokens = int(_safe_int(row.get("computed_assumed_doc_tokens"), 0))
    parts: List[str] = []
    if requested > 0:
        parts.append(f"req{requested}")
    if executed > 0:
        parts.append(f"exec{executed}")
    if leaves > 0:
        parts.append(f"n{leaves}")
    if doc_tokens > 0:
        parts.append(f"doc{doc_tokens}")
    return "__".join(parts)


def _row_geometry_label(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("supervision_recovery_geometry_label", "") or "").strip()
    if explicit:
        return explicit
    leaf_tokens = max(_requested_fixed_leaf_tokens(row), _effective_fixed_leaf_tokens(row))
    if leaf_tokens > 0:
        return f"leaf{leaf_tokens:03d}"
    return ""


def _is_exact_full_doc_parity_row(row: Mapping[str, Any]) -> bool:
    if bool(row.get("is_exact_full_doc_parity_row")):
        return True
    if str(row.get("parity_mode", "") or "").strip() != "exact_full_doc":
        return False
    return _effective_leaves_per_doc(row) == 1


def _annotated_recovery_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return annotate_downstream_v3_row(
        dict(row or {}),
        canonical_fno_families=CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES,
        canonical_fno_fixed_leaf_tokens=128,
    )


def _supervision_recovery_source_descriptor(
    path: object,
) -> Dict[str, Any]:
    normalized = str(path or "").strip().replace("\\", "/").lower()
    if not normalized:
        return {"tier_label": "unknown", "tier_rank": 0, "path": ""}
    if "publication_fullval_v3" in normalized:
        return {
            "tier_label": "publication_fullval_v3",
            "tier_rank": 500,
            "path": str(path or ""),
        }
    if "publication_xlarge" in normalized or "overnight_xlarge" in normalized:
        return {
            "tier_label": "publication_xlarge",
            "tier_rank": 500,
            "path": str(path or ""),
        }
    if "publication_fullval" in normalized:
        return {
            "tier_label": "publication_fullval",
            "tier_rank": 300,
            "path": str(path or ""),
        }
    if any(
        token in normalized
        for token in (
            "ablation",
            "check_basics",
            "small_train_local_law",
            "local_law_scaleup",
            "local_law_quickcheck",
            "oneleaf_root_budget_matched",
        )
    ):
        return {
            "tier_label": "protocol_ablation",
            "tier_rank": 200,
            "path": str(path or ""),
        }
    return {
        "tier_label": "exploratory",
        "tier_rank": 100,
        "path": str(path or ""),
    }


def _root_prefix_from_root_name(root_name: str) -> str:
    normalized = str(root_name or "").strip()
    if not normalized:
        return ""
    match = re.match(r"^(.*_)\d{8}_\d{4,6}$", normalized)
    if match:
        return str(match.group(1))
    return normalized


def _source_bundle_metadata(path: object) -> Dict[str, str]:
    raw_path = str(path or "").strip().replace("\\", "/")
    if not raw_path:
        return {
            "source_root_name": "",
            "source_bundle_name": "",
            "source_root_prefix": "",
            "source_attempt_lineage": "",
            "source_lineage_label": "",
        }
    parts = [part for part in Path(raw_path).parts if str(part).strip()]
    root_name = ""
    bundle_name = ""
    try:
        outputs_idx = parts.index("outputs")
    except ValueError:
        outputs_idx = -1
    if outputs_idx >= 0 and len(parts) >= outputs_idx + 3:
        root_name = str(parts[outputs_idx + 1])
        bundle_name = str(parts[outputs_idx + 2])
    root_prefix = _root_prefix_from_root_name(root_name)
    attempt_lineage = "/".join(
        part for part in (root_name, bundle_name) if str(part).strip()
    )
    compact_root_label = str(root_prefix).removeprefix("markov_v3_").strip("_")
    lineage_bits = [bit for bit in (bundle_name, compact_root_label) if str(bit).strip()]
    return {
        "source_root_name": root_name,
        "source_bundle_name": bundle_name,
        "source_root_prefix": root_prefix,
        "source_attempt_lineage": attempt_lineage,
        "source_lineage_label": " @ ".join(lineage_bits),
    }


def _recovery_row_source_descriptor(
    row: Mapping[str, Any],
) -> Dict[str, Any]:
    return _supervision_recovery_source_descriptor(
        row.get("source_summary_json")
        or row.get("source_path")
        or row.get("job_output_dir")
    )


def _with_recovery_source_metadata(
    row: Mapping[str, Any],
    *,
    source_summary_json: str = "",
) -> Dict[str, Any]:
    updated = _annotated_recovery_row(row)
    source_path = str(
        updated.get("source_summary_json", "") or source_summary_json or ""
    ).strip()
    descriptor = _supervision_recovery_source_descriptor(source_path)
    bundle_metadata = _source_bundle_metadata(source_path)
    updated["source_summary_json"] = source_path
    updated["source_tier_label"] = str(descriptor["tier_label"])
    updated["source_tier_rank"] = int(descriptor["tier_rank"])
    updated.update(bundle_metadata)
    return updated


def _with_recovery_scope_source_metadata(
    row: Mapping[str, Any],
    *,
    source_summary_json: str = "",
) -> Dict[str, Any]:
    updated = dict(row or {})
    baseline_family = str(updated.get("baseline_family", "") or "").strip()
    if not baseline_family:
        tree_family = str(updated.get("tree_family", "") or "").strip()
        if tree_family:
            baseline_family = tree_family
        elif any(
            key in updated
            for key in (
                "tree_test_root_mae",
                "tree_val_root_mae",
                "tree_train_root_mae",
            )
        ):
            baseline_family = "tree_neural"
        if baseline_family:
            updated["baseline_family"] = baseline_family
    source_path = str(
        updated.get("source_summary_json", "") or source_summary_json or ""
    ).strip()
    descriptor = _supervision_recovery_source_descriptor(source_path)
    bundle_metadata = _source_bundle_metadata(source_path)
    updated["source_summary_json"] = source_path
    updated["source_tier_label"] = str(descriptor["tier_label"])
    updated["source_tier_rank"] = int(descriptor["tier_rank"])
    updated.update(bundle_metadata)
    contract_status = str(updated.get("contract_status", "") or "").strip()
    if contract_status:
        updated["contract_headline_eligible"] = bool(
            is_headline_contract_status(contract_status)
        )
        return updated
    if baseline_family:
        annotated = _annotated_recovery_row(updated)
        annotated["source_summary_json"] = source_path
        annotated["source_tier_label"] = str(descriptor["tier_label"])
        annotated["source_tier_rank"] = int(descriptor["tier_rank"])
        annotated.update(bundle_metadata)
        return annotated
    return updated


def _recovery_row_semantic_key(
    row: Mapping[str, Any],
) -> Dict[str, Any]:
    row_map = _annotated_recovery_row(row)
    tree_reference_label = ""
    if str(row_map.get("baseline_family", "") or "").startswith("tree_"):
        tree_reference_label = str(row_map.get("tree_reference_label", "") or "")
    return {
        "scope_key": str(row_map.get("scope_key", "") or ""),
        "train_doc_count": int(_safe_int(row_map.get("train_doc_count"), 0)),
        "package_name": str(row_map.get("package_name", "") or ""),
        "baseline_family": str(row_map.get("baseline_family", "") or ""),
        "comparison_semantics_label": str(
            row_map.get("comparison_semantics_label", "") or ""
        ),
        "requested_fixed_leaf_tokens": int(
            _safe_int(row_map.get("requested_fixed_leaf_tokens"), 0)
        ),
        "executed_fixed_leaf_tokens": int(
            _safe_int(row_map.get("executed_fixed_leaf_tokens"), 0)
        ),
        "depth_discount_gamma": round(
            _safe_float(row_map.get("depth_discount_gamma"), float("nan")),
            6,
        ),
        "run_intent_hash": str(row_map.get("run_intent_hash", "") or ""),
        "tree_reference_label": tree_reference_label,
    }


def _recovery_row_grouping_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    semantic_key = _recovery_row_semantic_key(row)
    return (
        str(semantic_key["scope_key"]),
        int(semantic_key["train_doc_count"]),
        str(semantic_key["package_name"]),
        str(semantic_key["baseline_family"]),
        str(semantic_key["comparison_semantics_label"]),
        str(semantic_key["run_intent_hash"]),
        _row_geometry_identity(_annotated_recovery_row(row)),
        float(semantic_key["depth_discount_gamma"]),
        str(semantic_key["tree_reference_label"]),
        str(row.get("source_summary_json", "") or ""),
    )


def _preferred_recovery_row(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    candidates = [
        _with_recovery_source_metadata(row)
        for row in rows
        if isinstance(row, Mapping)
    ]
    if not candidates:
        return {}

    def _sort_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
        return (
            int(_safe_int(row.get("source_tier_rank"), 0)),
            1 if is_headline_contract_status(row.get("contract_status")) else 0,
            str(row.get("source_summary_json", "") or ""),
        )

    return max(candidates, key=_sort_key)


def _preferred_scope_rows_by_package(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for raw_row in rows:
        row = dict(raw_row or {})
        package_name = str(row.get("package_name", "") or "").strip()
        if not package_name:
            continue
        grouped[package_name].append(row)
    return {
        package_name: _preferred_recovery_row(package_rows)
        for package_name, package_rows in grouped.items()
    }


def _preferred_scope_rows_by_package_and_lineage(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for raw_row in rows:
        row = dict(raw_row or {})
        package_name = str(row.get("package_name", "") or "").strip()
        if not package_name:
            continue
        lineage_key = str(
            row.get("source_attempt_lineage")
            or row.get("source_summary_json")
            or row.get("source_lineage_label")
            or "unknown"
        ).strip()
        grouped[package_name][lineage_key].append(row)
    return {
        package_name: {
            lineage_key: _preferred_recovery_row(lineage_rows)
            for lineage_key, lineage_rows in lineages.items()
        }
        for package_name, lineages in grouped.items()
    }


def _hidden_invalid_reasons(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    reasons: set[str] = set()
    for row in rows:
        row_map = dict(row or {})
        for key in ("contract_failures", "contract_diagnostic_reasons"):
            for value in list(row_map.get(key) or []):
                text = str(value or "").strip()
                if text:
                    reasons.add(text)
        status = str(row_map.get("contract_status", "") or "").strip()
        if status and not is_headline_contract_status(status):
            reasons.add(status)
    return sorted(reasons)


def _lineage_key_for_row(row: Mapping[str, Any]) -> str:
    return str(
        row.get("source_attempt_lineage")
        or row.get("source_summary_json")
        or row.get("source_lineage_label")
        or "unknown"
    ).strip()


def _lineage_label_for_row(row: Mapping[str, Any]) -> str:
    label = str(row.get("source_lineage_label", "") or "").strip()
    if label:
        return label
    return _lineage_key_for_row(row)


def _lineage_sort_key_from_row(row: Mapping[str, Any]) -> tuple[int, str, str]:
    return (
        -int(_safe_int(row.get("source_tier_rank"), 0)),
        _lineage_label_for_row(row),
        str(row.get("source_summary_json", "") or ""),
    )


def _lineage_metric_series(
    rows_by_package_and_lineage: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    root_shares: Sequence[int],
    package_for_share: Any,
    metric_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    lineage_rows: Dict[str, Dict[str, Any]] = {}
    for share in root_shares:
        package_name = str(package_for_share(int(share)))
        for lineage_key, row in dict(rows_by_package_and_lineage.get(package_name) or {}).items():
            lineage_rows.setdefault(str(lineage_key), dict(row or {}))
    ordered_lineages = sorted(
        lineage_rows.items(),
        key=lambda item: _lineage_sort_key_from_row(dict(item[1] or {})),
    )
    out: List[Dict[str, Any]] = []
    for lineage_key, exemplar_row in ordered_lineages:
        series = {
            "lineage_key": str(lineage_key),
            "lineage_label": _lineage_label_for_row(exemplar_row),
            "source_tier_label": str(exemplar_row.get("source_tier_label", "") or ""),
            "source_tier_rank": int(_safe_int(exemplar_row.get("source_tier_rank"), 0)),
            "source_summary_json": str(exemplar_row.get("source_summary_json", "") or ""),
        }
        for metric_key in metric_keys:
            series[str(metric_key)] = [
                _safe_float(
                    dict(
                        dict(rows_by_package_and_lineage.get(str(package_for_share(int(share))), {}) or {}).get(
                            str(lineage_key),
                            {},
                        )
                    ).get(metric_key),
                    float("nan"),
                )
                for share in root_shares
            ]
        out.append(series)
    return out


def _series_value_signature(values: Sequence[object]) -> tuple[object, ...]:
    signature: List[object] = []
    for value in values:
        number = _safe_float(value, float("nan"))
        if math.isfinite(number):
            signature.append(round(number, 12))
        else:
            signature.append("nan")
    return tuple(signature)


def _collapse_identical_lineage_series(
    series_list: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[object, ...], List[Dict[str, Any]]] = defaultdict(list)
    for raw_series in series_list:
        series = dict(raw_series or {})
        grouped[_series_value_signature(list(series.get(metric_key) or []))].append(series)
    collapsed: List[Dict[str, Any]] = []
    for items in grouped.values():
        ordered = sorted(
            items,
            key=lambda item: (
                -int(_safe_int(item.get("source_tier_rank"), 0)),
                str(item.get("lineage_label", "") or ""),
                str(item.get("source_summary_json", "") or ""),
            ),
        )
        representative = dict(ordered[0] or {})
        matching_labels = [
            str(item.get("lineage_label", "") or "").strip()
            for item in ordered
            if str(item.get("lineage_label", "") or "").strip()
        ]
        representative["matching_lineage_labels"] = list(matching_labels)
        representative["matching_lineage_count"] = int(len(ordered))
        if len(ordered) > 1:
            representative["lineage_label"] = (
                f"{representative.get('lineage_label', 'lineage')} "
                f"(+{len(ordered) - 1} matching bundles)"
            )
        collapsed.append(representative)
    collapsed.sort(
        key=lambda item: (
            -int(_safe_int(item.get("source_tier_rank"), 0)),
            str(item.get("lineage_label", "") or ""),
            str(item.get("source_summary_json", "") or ""),
        )
    )
    return collapsed


def _resolve_recovery_row_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_summary_json: str = "",
    category: str,
) -> tuple[Dict[tuple[Any, ...], Dict[str, Any]], List[Dict[str, Any]]]:
    candidates_by_key: Dict[tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            continue
        row = _with_recovery_source_metadata(
            raw_row,
            source_summary_json=source_summary_json,
        )
        candidates_by_key[_recovery_row_grouping_key(row)].append(row)

    resolved: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    duplicate_resolution: List[Dict[str, Any]] = []
    for key, candidates in candidates_by_key.items():
        winner = _preferred_recovery_row(candidates)
        if winner:
            resolved[key] = winner
        unique_sources = sorted(
            {
                str(candidate.get("source_summary_json", "") or "")
                for candidate in candidates
                if str(candidate.get("source_summary_json", "") or "").strip()
            }
        )
        if len(candidates) <= 1 and len(unique_sources) <= 1:
            continue
        duplicate_resolution.append(
            {
                "category": str(category),
                "semantic_key": _recovery_row_semantic_key(winner or candidates[0]),
                "chosen_source": str(winner.get("source_summary_json", "") or ""),
                "chosen_source_tier": str(winner.get("source_tier_label", "") or ""),
                "chosen_source_rank": int(
                    _safe_int(winner.get("source_tier_rank"), 0)
                ),
                "superseded_sources": [
                    source
                    for source in unique_sources
                    if source != str(winner.get("source_summary_json", "") or "")
                ],
                "candidate_count": int(len(candidates)),
            }
        )
    duplicate_resolution.sort(
        key=lambda item: (
            str(item.get("category", "")),
            json.dumps(item.get("semantic_key", {}), sort_keys=True),
        )
    )
    return resolved, duplicate_resolution


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_record_paths(record: Mapping[str, Any]) -> List[Path]:
    raw_paths = list(record.get("paths") or [])
    if raw_paths:
        return [Path(str(path)).expanduser() for path in raw_paths if str(path).strip()]
    raw_path = str(record.get("path", "") or "").strip()
    return [Path(raw_path).expanduser()] if raw_path else []


def _merge_supervision_recovery_payloads(
    payloads: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    package_order: List[str] = []
    seen_packages: set[str] = set()
    train_doc_counts: List[int] = []
    seen_docs: set[int] = set()
    family_row_candidates: List[Dict[str, Any]] = []
    scope_row_candidates: Dict[tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    dense_anchor_candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    scope_map: Dict[str, Dict[str, Any]] = {}

    for payload in payloads:
        if not payload:
            continue
        merged.update(dict(payload))
        payload_source_summary_json = str(
            payload.get("source_summary_json", "") or ""
        ).strip()
        for package_name in list(payload.get("package_order") or []):
            normalized = str(package_name)
            if not normalized or normalized in seen_packages:
                continue
            seen_packages.add(normalized)
            package_order.append(normalized)
        for train_doc_count in list(payload.get("train_doc_counts") or []):
            value = int(_safe_int(train_doc_count, 0))
            if value <= 0 or value in seen_docs:
                continue
            seen_docs.add(value)
            train_doc_counts.append(value)
        for row in list(payload.get("family_rows") or []):
            family_row_candidates.append(
                _with_recovery_source_metadata(
                    dict(row or {}),
                    source_summary_json=payload_source_summary_json,
                )
            )
        for scope_key, raw_scope in dict(payload.get("scopes") or {}).items():
            scope_label = str(
                dict(raw_scope or {}).get("scope_label", scope_key) or scope_key
            )
            existing_scope = scope_map.setdefault(
                str(scope_key),
                {
                    "scope_key": str(scope_key),
                    "scope_label": scope_label,
                    "rows_by_train_docs": {},
                    "dense_anchor_rows": [],
                    "best_tree_by_train_docs": {},
                },
            )
            for item in list(dict(raw_scope or {}).get("rows_by_train_docs") or []):
                item_map = dict(item or {})
                train_doc_count = int(_safe_int(item_map.get("train_doc_count"), 0))
                if train_doc_count <= 0:
                    continue
                for row in list(item_map.get("rows") or []):
                    scope_row_candidates[(str(scope_key), int(train_doc_count))].append(
                        _with_recovery_scope_source_metadata(
                            dict(row or {}),
                            source_summary_json=payload_source_summary_json,
                        )
                    )
            for row in list(dict(raw_scope or {}).get("dense_anchor_rows") or []):
                dense_anchor_candidates[str(scope_key)].append(
                    _with_recovery_scope_source_metadata(
                        dict(row or {}),
                        source_summary_json=payload_source_summary_json,
                    )
                )

    family_row_map, family_duplicate_resolution = _resolve_recovery_row_candidates(
        family_row_candidates,
        category="family_rows",
    )
    duplicate_resolution: List[Dict[str, Any]] = list(family_duplicate_resolution)

    for (scope_key, train_doc_count), rows in sorted(
        scope_row_candidates.items(),
        key=lambda item: (str(item[0][0]), int(item[0][1])),
    ):
        resolved_rows, scope_duplicates = _resolve_recovery_row_candidates(
            rows,
            category="scope_rows",
        )
        duplicate_resolution.extend(scope_duplicates)
        scope_payload = scope_map.setdefault(
            str(scope_key),
            {
                "scope_key": str(scope_key),
                "scope_label": str(scope_key),
                "rows_by_train_docs": {},
                "dense_anchor_rows": [],
                "best_tree_by_train_docs": {},
            },
        )
        rows_by_train_docs = scope_payload.setdefault("rows_by_train_docs", {})
        rows_by_train_docs[str(train_doc_count)] = {
            "train_doc_count": int(train_doc_count),
            "rows": list(resolved_rows.values()),
        }

    for scope_key, rows in sorted(dense_anchor_candidates.items()):
        resolved_rows, dense_duplicates = _resolve_recovery_row_candidates(
            rows,
            category="dense_anchor_rows",
        )
        duplicate_resolution.extend(dense_duplicates)
        scope_payload = scope_map.setdefault(
            str(scope_key),
            {
                "scope_key": str(scope_key),
                "scope_label": str(scope_key),
                "rows_by_train_docs": {},
                "dense_anchor_rows": [],
                "best_tree_by_train_docs": {},
            },
        )
        scope_payload["dense_anchor_rows"] = list(resolved_rows.values())

    for scope_payload in scope_map.values():
        rows_by_train_docs = dict(scope_payload.get("rows_by_train_docs") or {})
        best_tree_by_train_docs: Dict[str, Dict[str, Any]] = {}
        for train_doc_count, item in rows_by_train_docs.items():
            rows = [
                dict(row)
                for row in list(dict(item or {}).get("rows") or [])
                if str(dict(row).get("baseline_family", "") or "").startswith("tree_")
                and math.isfinite(
                    _safe_float(
                        dict(row).get(
                            "tree_test_root_mae",
                            dict(row).get("test_root_mae_mean"),
                        ),
                        float("nan"),
                    )
                )
            ]
            best_tree_by_train_docs[str(train_doc_count)] = (
                min(
                    rows,
                    key=lambda row: _safe_float(
                        row.get("tree_test_root_mae", row.get("test_root_mae_mean")),
                        float("inf"),
                    ),
                )
                if rows
                else {}
            )
        scope_payload["best_tree_by_train_docs"] = best_tree_by_train_docs

    merged["package_order"] = package_order or list(merged.get("package_order") or [])
    merged["train_doc_counts"] = sorted(train_doc_counts)
    all_family_rows = list(family_row_map.values())
    merged["all_family_rows"] = all_family_rows
    merged["family_rows"] = filtered_headline_rows(all_family_rows)
    merged["hidden_invalid_family_rows"] = [
        dict(row)
        for row in all_family_rows
        if not is_headline_contract_status(str(dict(row or {}).get("contract_status", "") or ""))
    ]
    merged["quarantined_family_rows"] = []
    merged["quarantined_row_count"] = 0
    merged["quarantined_sources"] = []
    merged["hidden_invalid_row_count"] = int(len(merged["hidden_invalid_family_rows"]))
    merged["hidden_invalid_sources"] = quarantine_sources_from_rows(
        merged["hidden_invalid_family_rows"]
    )
    merged["hidden_invalid_reasons"] = _hidden_invalid_reasons(
        merged["hidden_invalid_family_rows"]
    )
    merged["contract_gate_status"] = "pass"
    merged["duplicate_resolution"] = duplicate_resolution
    merged["lineage_labels"] = sorted(
        {
            str(dict(row or {}).get("source_lineage_label", "") or "").strip()
            for row in merged["family_rows"]
            if str(dict(row or {}).get("source_lineage_label", "") or "").strip()
        }
    )
    best_tree_summary: List[Dict[str, Any]] = []
    for scope_key, scope_payload in scope_map.items():
        scope_label = str(
            dict(scope_payload).get("scope_label", scope_key) or scope_key
        )
        for train_doc_count, item in sorted(
            dict(scope_payload.get("rows_by_train_docs") or {}).items(),
            key=lambda pair: int(_safe_int(pair[0], 0)),
        ):
            best_row = dict(
                dict(scope_payload.get("best_tree_by_train_docs") or {}).get(
                    str(train_doc_count),
                    {},
                )
            )
            if not best_row:
                continue
            best_tree_summary.append(
                {
                    **best_row,
                    "scope_key": str(scope_key),
                    "scope_label": scope_label,
                    "train_doc_count": int(_safe_int(train_doc_count, 0)),
                    "tree_test_root_mae": _safe_float(
                        best_row.get(
                            "tree_test_root_mae",
                            best_row.get("test_root_mae_mean"),
                        ),
                        float("nan"),
                    ),
                }
            )
    merged["best_tree_summary"] = best_tree_summary
    merged["scopes"] = {
        scope_key: {
            **dict(scope_payload),
            "rows_by_train_docs": [
                dict(item)
                for _, item in sorted(
                    dict(scope_payload.get("rows_by_train_docs") or {}).items(),
                    key=lambda pair: int(_safe_int(pair[0], 0)),
                )
            ],
        }
        for scope_key, scope_payload in scope_map.items()
    }
    return merged


def _existing_sources(paths: Mapping[str, Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name, path in paths.items():
        if path.exists():
            out[name] = str(path)
    return out


def _report_manifest_path(version_root: Path) -> Path:
    return version_root / REPORT_VERSION_MANIFEST_NAME


def _empty_source_record(key: str) -> Dict[str, Any]:
    spec = REPORT_SOURCE_SPECS.get(str(key), {})
    return {
        "relpath": "",
        "path": "",
        "origin": "",
        "phase": str(spec.get("phase", "")),
        "sha256": "",
        "config_fingerprint": "",
        "status": "missing",
        "reason": "no local artifact selected",
        "selected_attempt_id": "",
    }


def _coerce_source_record(
    *,
    key: str,
    raw: Mapping[str, Any],
    version_root: Path | None,
) -> Dict[str, Any]:
    record = _empty_source_record(key)
    record.update(dict(raw or {}))
    relpath = str(record.get("relpath", "") or "").strip()
    explicit_path = str(record.get("path", "") or "").strip()
    resolved_path = ""
    if relpath and version_root is not None:
        resolved_path = str((version_root / relpath).expanduser())
    elif explicit_path:
        resolved_path = explicit_path
    record["path"] = resolved_path
    if resolved_path and not Path(resolved_path).exists():
        record["status"] = "missing"
        record["reason"] = "selected artifact is missing from this version root"
    return record


def _load_source_records(
    args: argparse.Namespace,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any] | None, Path | None]:
    manifest_payload: Dict[str, Any] | None = None
    version_root = Path(args.version_root).expanduser() if args.version_root is not None else None
    output_root = Path(args.output_root).expanduser() if args.output_root is not None else None
    manifest_path = Path(args.manifest).expanduser() if args.manifest is not None else None
    if manifest_path is None and version_root is not None:
        candidate = _report_manifest_path(version_root)
        if candidate.exists():
            manifest_path = candidate
    source_records: Dict[str, Dict[str, Any]] = {}
    if manifest_path is not None and manifest_path.exists():
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if version_root is None:
            version_root = Path(str(manifest_payload.get("version_root") or manifest_path.parent)).expanduser()
        raw_sources = dict(manifest_payload.get("selected_sources") or {})
        for key in REPORT_SOURCE_SPECS:
            source_records[key] = _coerce_source_record(
                key=key,
                raw=dict(raw_sources.get(key) or {}),
                version_root=version_root,
            )
        return source_records, manifest_payload, version_root

    for key, spec in REPORT_SOURCE_SPECS.items():
        arg_name = str(spec["arg"])
        raw_path = getattr(args, arg_name)
        record = _empty_source_record(key)
        raw_paths: List[Path] = []
        if isinstance(raw_path, (list, tuple)):
            raw_paths = [Path(path).expanduser() for path in raw_path if path is not None]
        elif raw_path is not None:
            raw_paths = [Path(raw_path).expanduser()]
        existing_raw_paths = [path for path in raw_paths if path.exists()]
        if existing_raw_paths:
            record.update(
                {
                    "path": str(existing_raw_paths[0]),
                    "origin": "cli",
                    "status": "ready",
                    "reason": "",
                    "paths": [str(path) for path in existing_raw_paths],
                }
            )
        elif output_root is not None:
            candidate_paths = []
            if key == "supervision_recovery_summary":
                candidate_paths.extend(
                    [
                        output_root / "supervision_recovery" / "summary.json",
                        output_root / "tradeoff_report" / "summary.json",
                    ]
                )
            elif key == "fno_upper_bound_summary":
                candidate_paths.append(output_root / "tree_fno_upper_bound_summary.json")
            elif key == "oracle_budget_frontier_summary":
                candidate_paths.append(output_root / "oracle_budget_frontier" / "summary.json")
            elif key == "learnability_summary":
                candidate_paths.append(output_root / "learnability_report" / "learnability_summary.json")
            for candidate in candidate_paths:
                if candidate.exists():
                    record.update(
                        {
                            "path": str(candidate),
                            "origin": "output_root",
                            "status": "ready",
                            "reason": "",
                        }
                    )
                    break
        source_records[key] = record
    return source_records, manifest_payload, version_root


def _update_manifest_selected_sources(
    manifest_payload: Dict[str, Any] | None,
    *,
    source_records: Mapping[str, Mapping[str, Any]],
) -> None:
    if manifest_payload is None:
        return
    selected_sources = dict(manifest_payload.get("selected_sources") or {})
    for key, record in source_records.items():
        current = dict(selected_sources.get(key) or {})
        current.update(
            {
                "relpath": str(record.get("relpath", "") or ""),
                "origin": str(record.get("origin", "") or ""),
                "phase": str(record.get("phase", "") or ""),
                "sha256": str(record.get("sha256", "") or ""),
                "config_fingerprint": str(record.get("config_fingerprint", "") or ""),
                "status": str(record.get("status", "") or ""),
                "reason": str(record.get("reason", "") or ""),
                "selected_attempt_id": str(record.get("selected_attempt_id", "") or ""),
            }
        )
        for extra_key in ("expected_train_doc_counts", "staged_from"):
            if extra_key in record:
                current[extra_key] = record[extra_key]
        selected_sources[str(key)] = current
    manifest_payload["selected_sources"] = selected_sources


def _current_path_contract() -> Dict[str, Any]:
    return {
        "canonical_identifiable_zero_reference_kind": "full_doc_fno_upper_bound",
        "canonical_full_doc_fno_families": list(CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES),
        "full_doc_fno_training_entrypoint": (
            "src.ctreepo.sim.core.markov_neural_operator_baselines._fit_fno_baseline_with_predictions"
        ),
        "full_doc_fno_training_backend": (
            "src.ctreepo.sim.core.markov_neural_operator_baselines._train_loop_with_predictions"
        ),
        "tree_training_entrypoint": (
            "src.ctreepo.sim.core.markov_neural_operator_baselines.train_fno_tree"
        ),
        "legacy_identifiable_zero_mixed_appendix_included": False,
    }


def _source_is_ready(summary: Mapping[str, Any], key: str) -> bool:
    record = dict((summary.get("source_records") or {}).get(key) or {})
    return str(record.get("status", "")) == "ready"


def _source_placeholder_lines(summary: Mapping[str, Any], key: str) -> List[str]:
    record = dict((summary.get("source_records") or {}).get(key) or {})
    status = str(record.get("status", "missing") or "missing")
    phase = str(record.get("phase", "") or "")
    lines = [
        f"Source key: {key}",
        f"Phase: {phase or 'n/a'}",
        f"Status: {status}",
    ]
    reason = str(record.get("reason", "") or "").strip()
    if reason:
        lines.append(f"Reason: {reason}")
    path_text = str(record.get("path", "") or "").strip()
    if path_text:
        lines.append(f"Selected artifact: {path_text}")
    return lines


def _focused_train_doc_counts(recovery: Mapping[str, Any]) -> List[int]:
    expected = [
        int(_safe_int(value))
        for value in list(recovery.get("expected_train_doc_counts") or [])
        if int(_safe_int(value)) > 0
    ]
    if expected:
        return expected
    return [
        int(_safe_int(value))
        for value in list(recovery.get("train_doc_counts") or [])
        if int(_safe_int(value)) > 0
    ]


def _report_train_doc_count_set(recovery: Mapping[str, Any]) -> set[int]:
    return {
        int(_safe_int(value))
        for value in _focused_train_doc_counts(recovery)
        if int(_safe_int(value)) > 0
    }


def _root_label_example_line(train_doc_counts: Sequence[int], *, root_share: int) -> str:
    parts: List[str] = []
    for train_docs in train_doc_counts:
        root_labels = int(round(float(train_docs) * float(root_share) / 100.0))
        approx_prefix = "~" if (float(train_docs) * float(root_share) / 100.0) != float(root_labels) else ""
        parts.append(f"`train_docs={int(train_docs)}` -> {approx_prefix}`{int(root_labels)}` root-labeled docs")
    return (
        f"- Concrete example for `R{int(root_share)}`: "
        + ", ".join(parts)
        + "."
    )


def _is_ordered_subsequence(expected: Sequence[str], observed: Sequence[str]) -> bool:
    if not expected:
        return True
    observed_iter = iter(str(item) for item in observed)
    for needle in (str(item) for item in expected):
        for candidate in observed_iter:
            if candidate == needle:
                break
        else:
            return False
    return True


def _summarize_supervision_recovery(
    payload: Mapping[str, Any],
    *,
    expected_train_doc_counts: Sequence[int] | None = None,
    expected_package_order: Sequence[str] | None = None,
    expected_tree_family: str = SUPERVISION_RECOVERY_TREE_FAMILY,
    expected_structural_cell: str = SUPERVISION_RECOVERY_STRUCTURAL_CELL,
) -> Dict[str, Any]:
    family_rows = [dict(row) for row in list(payload.get("family_rows") or [])]
    quarantined_family_rows = [
        dict(row) for row in list(payload.get("quarantined_family_rows") or [])
    ]
    fno_family_lookup: Dict[tuple[str, int, str, str], Dict[str, Any]] = {}
    for row in family_rows:
        baseline_family = str(row.get("baseline_family", "") or "")
        if baseline_family not in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES:
            continue
        scope_key = str(row.get("scope_key", "") or "")
        train_doc_count = int(_safe_int(row.get("train_doc_count")))
        package_name = str(row.get("package_name", "") or "")
        if not scope_key or train_doc_count <= 0 or not package_name:
            continue
        fno_family_lookup[(scope_key, train_doc_count, package_name, baseline_family)] = {
            "baseline_family": baseline_family,
            "test_root_mae": _safe_float(row.get("test_root_mae_mean"), float("nan")),
            "n_runs": int(_safe_int(row.get("n_runs"))),
            "package_name": package_name,
        }

    package_order = [str(item) for item in list(payload.get("package_order") or [])]
    expected_packages_list = (
        [str(item) for item in expected_package_order]
        if expected_package_order
        else (package_order or list(SUPERVISION_RECOVERY_PACKAGE_ORDER))
    )
    normalized_expected_packages = list(expected_packages_list)
    package_contract_notice = ""
    if package_order:
        normalized_expected_packages = list(package_order)
        if expected_packages_list and package_order != expected_packages_list:
            missing_expected = [
                package_name
                for package_name in expected_packages_list
                if package_name not in package_order
            ]
            unexpected_payload = [
                package_name
                for package_name in package_order
                if package_name not in expected_packages_list
            ]
            if missing_expected or unexpected_payload or package_order != expected_packages_list:
                mismatch_parts: List[str] = []
                if not missing_expected and not unexpected_payload:
                    mismatch_parts.append("using payload-defined package order")
                if missing_expected:
                    mismatch_parts.append(f"missing expected packages {missing_expected}")
                if unexpected_payload:
                    mismatch_parts.append(f"including extra packages {unexpected_payload}")
                package_contract_notice = (
                    "supervision-recovery payload package order differs from the focused report contract: "
                    + "; ".join(mismatch_parts)
                )
    tree_family = str(payload.get("tree_family", "") or "").strip()
    if expected_tree_family and tree_family and tree_family != str(expected_tree_family):
        return {
            "status": "incompatible",
            "reason": (
                "supervision-recovery tree family does not match the focused report contract "
                f"({tree_family} vs {expected_tree_family})"
            ),
            "scopes": {},
            "package_order": package_order,
        }
    raw_scopes = dict(payload.get("scopes") or {})
    if not raw_scopes:
        return {
            "status": "missing",
            "reason": "supervision-recovery summary has no scopes",
            "scopes": {},
            "package_order": package_order or list(SUPERVISION_RECOVERY_PACKAGE_ORDER),
        }
    expected_docs = sorted(
        {
            int(_safe_int(value))
            for value in list(expected_train_doc_counts or [])
            if int(_safe_int(value)) > 0
        }
    )
    normalized_scopes: Dict[str, Dict[str, Any]] = {}
    notices: List[str] = []
    if package_contract_notice:
        notices.append(package_contract_notice)
    alignment_warning = str(payload.get("comparator_alignment_warning", "") or "").strip()
    if alignment_warning:
        notices.append(alignment_warning)
    selection_warning = str(payload.get("comparator_selection_warning", "") or "").strip()
    if selection_warning:
        notices.append(selection_warning)
    quarantined_scope_rows: List[Dict[str, Any]] = []
    for scope_key, raw_scope in raw_scopes.items():
        scope = dict(raw_scope or {})
        rows_by_train_docs = {}
        raw_rows_by_train_docs = scope.get("rows_by_train_docs") or {}
        if isinstance(raw_rows_by_train_docs, Mapping):
            row_groups = [
                dict(item or {})
                for item in dict(raw_rows_by_train_docs).values()
            ]
        else:
            row_groups = [dict(item or {}) for item in list(raw_rows_by_train_docs)]
        for item in row_groups:
            train_doc_count = int(_safe_int((item or {}).get("train_doc_count")))
            if train_doc_count <= 0:
                continue
            rows: List[Dict[str, Any]] = []
            for raw_row in list((item or {}).get("rows") or []):
                row = _with_recovery_scope_source_metadata(dict(raw_row))
                if not is_headline_contract_status(row.get("contract_status")):
                    quarantined_scope_rows.append(dict(row))
                    continue
                row["is_authoritative_gamma_row"] = bool(
                    str(row.get("tree_supervision_source", "") or "") == "manifest"
                    and str(row.get("local_estimand_mode", "") or "")
                    == "span_mass_ipw_sum"
                    and str(row.get("c2_pair_weighting_mode", "") or "")
                    == "pair_ipw_geomean"
                )
                package_name = str(row.get("package_name", "") or "")
                row["fno_family_rows"] = {
                    family: dict(
                        fno_family_lookup.get(
                            (str(scope_key), int(train_doc_count), package_name, family),
                            {},
                        )
                    )
                    for family in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES
                }
                rows.append(row)
            rows_by_train_docs[str(train_doc_count)] = {
                "train_doc_count": int(train_doc_count),
                "rows": rows,
            }
        available_docs = sorted(int(_safe_int(key)) for key in rows_by_train_docs)
        missing_docs = [
            int(value)
            for value in expected_docs
            if int(value) not in {int(_safe_int(doc)) for doc in available_docs}
        ]
        missing_packages_by_train_docs: Dict[str, List[str]] = {}
        for train_doc_count, item in rows_by_train_docs.items():
            available_packages = {
                str(row.get("package_name", "") or "")
                for row in list(item.get("rows") or [])
                if str(row.get("package_name", "")).strip()
            }
            missing = [
                package_name
                for package_name in normalized_expected_packages
                if package_name not in available_packages
            ]
            if missing:
                missing_packages_by_train_docs[str(train_doc_count)] = missing
        if missing_docs:
            notices.append(
                f"{scope.get('scope_label', scope_key)} is missing train-doc counts {missing_docs}"
            )
        for train_doc_count, missing in sorted(
            missing_packages_by_train_docs.items(),
            key=lambda item: int(_safe_int(item[0], 0)),
        ):
            notices.append(
                f"{scope.get('scope_label', scope_key)} train_docs={train_doc_count} is missing packages {missing}"
            )
        dense_anchor_rows: List[Dict[str, Any]] = []
        for raw_row in list(scope.get("dense_anchor_rows") or []):
            row = _with_recovery_scope_source_metadata(dict(raw_row))
            if not is_headline_contract_status(row.get("contract_status")):
                quarantined_scope_rows.append(dict(row))
                continue
            package_name = str(row.get("package_name", "full100") or "full100")
            train_doc_count = int(_safe_int(row.get("train_doc_count")))
            row["fno_family_rows"] = {
                family: dict(
                    fno_family_lookup.get(
                        (str(scope_key), int(train_doc_count), package_name, family),
                        {},
                    )
                )
                for family in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES
            }
            dense_anchor_rows.append(row)
        normalized_scopes[str(scope_key)] = {
            **scope,
            "dense_anchor_rows": dense_anchor_rows,
            "rows_by_train_docs": rows_by_train_docs,
            "available_train_docs": available_docs,
            "missing_train_docs": missing_docs,
            "missing_packages_by_train_docs": missing_packages_by_train_docs,
        }
    if expected_structural_cell and str(expected_structural_cell) not in normalized_scopes:
        notices.append(
            f"{SUPERVISION_RECOVERY_STRUCTURAL_GRID}::{expected_structural_cell} is missing from the supervision-recovery summary"
        )
    summary = {
        **dict(payload),
        "status": str(payload.get("status", "ready") or "ready"),
        "reason": str(payload.get("reason", "") or ""),
        "tree_family": tree_family or str(expected_tree_family),
        "depth_discount_gammas": sorted(
            {
                round(_safe_float(row.get("depth_discount_gamma"), 1.0), 6)
                for row in family_rows
                if math.isfinite(_safe_float(row.get("depth_discount_gamma"), float("nan")))
            }
        ),
        "package_order": package_order or list(normalized_expected_packages),
        "contract_gate_status": str(
            payload.get("contract_gate_status", "pass") or "pass"
        ),
        "quarantined_row_count": int(
            _safe_int(
                payload.get("quarantined_row_count", len(quarantined_family_rows)),
                len(quarantined_family_rows),
            )
        ),
        "quarantined_sources": list(payload.get("quarantined_sources") or []),
        "quarantined_family_rows": quarantined_family_rows,
        "quarantined_scope_rows": quarantined_scope_rows,
        "hidden_invalid_family_rows": [
            dict(row) for row in list(payload.get("hidden_invalid_family_rows") or [])
        ],
        "hidden_invalid_scope_rows": quarantined_scope_rows,
        "hidden_invalid_row_count": int(
            _safe_int(
                payload.get("hidden_invalid_row_count", 0),
                0,
            )
        )
        + int(len(quarantined_scope_rows)),
        "hidden_invalid_sources": sorted(
            {
                str(source)
                for source in list(payload.get("hidden_invalid_sources") or [])
                + quarantine_sources_from_rows(quarantined_scope_rows)
                if str(source).strip()
            }
        ),
        "hidden_invalid_reasons": sorted(
            {
                str(reason)
                for reason in list(payload.get("hidden_invalid_reasons") or [])
                + _hidden_invalid_reasons(quarantined_scope_rows)
                if str(reason).strip()
            }
        ),
        "duplicate_resolution": list(payload.get("duplicate_resolution") or []),
        "lineage_labels": list(payload.get("lineage_labels") or []),
        "scopes": normalized_scopes,
        "expected_train_doc_counts": expected_docs,
        "expected_package_order": normalized_expected_packages,
        "expected_structural_cell": str(expected_structural_cell),
        "notices": notices,
    }
    summary["geometry_groups"] = _recovery_geometry_groups(summary)
    return summary


def _recovery_fno_family_lookup(
    recovery: Mapping[str, Any],
) -> Dict[tuple[str, int, str, str], Dict[str, Any]]:
    lookup: Dict[tuple[str, int, str, str], Dict[str, Any]] = {}
    for raw_row in list(recovery.get("family_rows") or []):
        row = dict(raw_row or {})
        scope_key = str(row.get("scope_key", "") or "").strip()
        train_doc_count = int(_safe_int(row.get("train_doc_count")))
        package_name = str(row.get("package_name", "") or "").strip()
        baseline_family = str(row.get("baseline_family", "") or "").strip()
        if (
            not scope_key
            or train_doc_count <= 0
            or not package_name
            or baseline_family not in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES
        ):
            continue
        lookup[(scope_key, train_doc_count, package_name, baseline_family)] = row
    return lookup


def _attach_best_tree_fno_family_breakdown(
    recovery: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    fno_lookup: Mapping[tuple[str, int, str, str], Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    row_map = dict(row or {})
    scope_key = str(row_map.get("scope_key", "") or "").strip()
    train_doc_count = int(_safe_int(row_map.get("train_doc_count")))
    matched_package = str(
        row_map.get("fno_reference_package", row_map.get("package_name", "")) or ""
    ).strip()
    tree_mae = _safe_float(row_map.get("tree_test_root_mae"), float("nan"))
    lookup = dict(fno_lookup or _recovery_fno_family_lookup(recovery))

    def _family_entry(package_name: str, family: str) -> Dict[str, Any]:
        family_row = dict(
            lookup.get((scope_key, train_doc_count, str(package_name), str(family))) or {}
        )
        test_root_mae = _safe_float(family_row.get("test_root_mae_mean"), float("nan"))
        return {
            "baseline_family": str(family),
            "package_name": str(package_name),
            "n_runs": int(_safe_int(family_row.get("n_runs"))),
            "test_root_mae": test_root_mae,
            "comparison_mode": str(family_row.get("comparison_mode", "") or ""),
            "comparison_semantics": str(
                family_row.get("comparison_semantics", "") or ""
            ),
            "comparison_semantics_label": str(
                family_row.get("comparison_semantics_label", "") or ""
            ),
            "run_intent_hash": str(family_row.get("run_intent_hash", "") or ""),
            "run_intent_validation_status": str(
                family_row.get("run_intent_validation_status", "") or ""
            ),
            "requested_fixed_leaf_tokens": int(
                _requested_fixed_leaf_tokens(family_row)
            ),
            "executed_fixed_leaf_tokens": int(
                _effective_fixed_leaf_tokens(family_row)
            ),
            "contract_status": str(family_row.get("contract_status", "") or ""),
            "delta_vs_tree": (
                float(tree_mae - test_root_mae)
                if math.isfinite(tree_mae) and math.isfinite(test_root_mae)
                else float("nan")
            ),
        }

    matched_fno_family_rows = {
        family: _family_entry(matched_package, family)
        for family in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES
    }
    full100_fno_family_rows = {
        family: _family_entry("full100", family)
        for family in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES
    }
    best_full100_family = ""
    best_full100_value = float("nan")
    for family, family_row in full100_fno_family_rows.items():
        value = _safe_float(family_row.get("test_root_mae"), float("nan"))
        if not math.isfinite(value):
            continue
        if not math.isfinite(best_full100_value) or value < best_full100_value:
            best_full100_family = str(family)
            best_full100_value = float(value)
    row_map["matched_fno_family_rows"] = matched_fno_family_rows
    row_map["full100_fno_family_rows"] = full100_fno_family_rows
    row_map["best_full100_fno_family"] = best_full100_family
    row_map["best_full100_fno_test_root_mae"] = best_full100_value
    row_map["delta_vs_best_full100_fno"] = (
        float(tree_mae - best_full100_value)
        if math.isfinite(tree_mae) and math.isfinite(best_full100_value)
        else float("nan")
    )
    return row_map


def _best_tree_summary_rows(recovery: Mapping[str, Any]) -> List[Dict[str, Any]]:
    fno_lookup = _recovery_fno_family_lookup(recovery)
    rows = [
        _attach_best_tree_fno_family_breakdown(recovery, row, fno_lookup=fno_lookup)
        for row in list(recovery.get("best_tree_summary") or [])
    ]
    rows.sort(
        key=lambda row: (
            str(row.get("scope_key", "")),
            int(_safe_int(row.get("train_doc_count"))),
        )
    )
    return rows


def _fno_family_label(family: str) -> str:
    normalized = str(family or "").strip()
    if normalized == "official_fno":
        return "official_fno"
    if normalized == "official_fno_sumlen":
        return "official_fno_sumlen"
    return normalized or "fno"


SUPERVISION_PACKAGE_TICK_LABELS: Dict[str, str] = {
    "full100": "R100",
    "full90": "R90",
    "full80": "R80",
    "full70": "R70",
    "full60": "R60",
    "full50": "R50",
    "full40": "R40",
    "full30": "R30",
    "full20": "R20",
    "full10": "R10",
    "full10_leaf_count10_internal_count10": "R10+LcIa10",
    "full10_leaf_count20_internal_count20": "R10+LcIa20",
    "full10_leaf_count50_internal_count50": "R10+LcIa50",
    "full10_leaf_count100_internal_count100": "R10+LcIa100",
    "full20_leaf_count10_internal_count10": "R20+LcIa10",
    "full20_leaf_count20_internal_count20": "R20+LcIa20",
    "full20_leaf_count50_internal_count50": "R20+LcIa50",
    "full20_leaf_count100_internal_count100": "R20+LcIa100",
    "r10_mass_local_eq_0p5": "R10mm+0.5",
    "r10_mass_local_eq_1p0": "R10mm+1.0",
    "r10_mass_local_eq_1p5": "R10mm+1.5",
    "r10_mass_local_eq_2p0": "R10mm+2.0",
    "r20_mass_local_eq_1p0": "R20mm+1.0",
    "r20_mass_local_eq_2p0": "R20mm+2.0",
    "r20_mass_local_eq_3p0": "R20mm+3.0",
    "r20_mass_local_eq_4p0": "R20mm+4.0",
    "r100_superset_local_eq_10p0": "R100sup+10.0",
    "r100_superset_local_eq_15p0": "R100sup+15.0",
    "r100_superset_local_eq_20p0": "R100sup+20.0",
    "full0_leaf_full100_internal_count100": "R0+Lf+Ia",
    "full10_leaf_count100": "R10+Lc",
    "full10_leaf_full100": "R10+Lf",
    "full10_leaf_full100_internal_depth1_count100": "R10+Lf+I1",
    "full10_leaf_full100_internal_depth2_count100": "R10+Lf+I12",
    "full10_leaf_full100_internal_count100": "R10+Lf+Ia",
    "full20_leaf_full100_internal_count100": "R20+Lf+Ia",
    "full30_leaf_full100_internal_count100": "R30+Lf+Ia",
    "full50_leaf_full100_internal_count100": "R50+Lf+Ia",
    "full60_leaf_full100_internal_count100": "R60+Lf+Ia",
    "full70_leaf_full100_internal_count100": "R70+Lf+Ia",
    "full80_leaf_full100_internal_count100": "R80+Lf+Ia",
    "full90_leaf_full100_internal_count100": "R90+Lf+Ia",
    "full100_leaf_full100_internal_count100": "R100+Lf+Ia",
}
SUPERVISION_PACKAGE_TICK_LABELS.update(
    _mass_matched_tick_labels(SUPERVISION_RECOVERY_MASS_MATCHED_RATE_LADDERS)
)

REPORT_PROFILE_DEFAULT = "supervision_recovery_v1"
REPORT_PROFILE_R10_COVERAGE_FOCUSED = "r10_coverage_focused"
REPORT_PROFILE_EXACT_PARITY_CANARY = "exact_parity_canary"
SUPERVISION_RECOVERY_PRIMARY_COMPARISON_ARM = "primary"


def _report_profile(summary: Mapping[str, Any]) -> str:
    profile = str(summary.get("report_focus", REPORT_PROFILE_DEFAULT) or REPORT_PROFILE_DEFAULT)
    return profile


def _is_r10_coverage_focused(summary: Mapping[str, Any]) -> bool:
    return _report_profile(summary) == REPORT_PROFILE_R10_COVERAGE_FOCUSED


def _is_exact_full_doc_canary_recovery(recovery: Mapping[str, Any]) -> bool:
    if str(recovery.get("status", "")) != "ready":
        return False
    package_order = [str(item) for item in list(recovery.get("package_order") or []) if str(item).strip()]
    if package_order != ["full100"]:
        return False
    tree_family = str(
        recovery.get("tree_family", SUPERVISION_RECOVERY_TREE_FAMILY)
        or SUPERVISION_RECOVERY_TREE_FAMILY
    )
    tree_rows = [
        dict(row or {})
        for row in list(recovery.get("family_rows") or [])
        if str(dict(row or {}).get("baseline_family", "") or "") == tree_family
    ]
    if not tree_rows:
        return False
    return all(
        str(row.get("package_name", "") or "") == "full100"
        and _is_exact_full_doc_parity_row(row)
        for row in tree_rows
    )


def _is_exact_full_doc_canary(summary: Mapping[str, Any]) -> bool:
    return _report_profile(summary) == REPORT_PROFILE_EXACT_PARITY_CANARY


def _package_tick_label(package_name: str) -> str:
    normalized = str(package_name or "").strip()
    label = str(SUPERVISION_PACKAGE_TICK_LABELS.get(normalized, "")).strip()
    if label:
        return label
    if normalized.startswith("r") and "_leaf_mass_eq_" in normalized:
        root_text, local_text = normalized[1:].split("_leaf_mass_eq_", 1)
        return f"R{root_text}+Lf{local_text.replace('p0', '')}"
    if normalized.startswith("r") and "_depth_equal_mass_eq_" in normalized:
        root_text, local_text = normalized[1:].split("_depth_equal_mass_eq_", 1)
        return f"R{root_text}+Eq{local_text.replace('p0', '')}"
    return normalized.replace("_", " ")


def _package_semantics(
    recovery: Mapping[str, Any],
    package_name: str,
) -> str:
    package_definitions = dict(recovery.get("package_definitions") or {})
    package_spec = dict(package_definitions.get(str(package_name)) or {})
    explicit = str(package_spec.get("package_semantics", "") or "").strip()
    if explicit:
        return explicit
    name = str(package_name or "").strip()
    if "_mass_local_eq_" in name:
        return "mass_matched"
    if "_superset_" in name:
        return "superset"
    root_share = _package_root_share(name)
    if root_share is not None and "leaf" not in name and "internal" not in name:
        return "full_doc_only"
    return ""


def _package_root_share(package_name: Any) -> int | None:
    name = str(package_name or "").strip()
    if not name.startswith("full"):
        return None
    digits: List[str] = []
    for char in name[len("full") :]:
        if char.isdigit():
            digits.append(char)
            continue
        break
    if not digits:
        return None
    try:
        return int("".join(digits))
    except Exception:
        return None


def _row_root_supervision_fraction(
    recovery: Mapping[str, Any],
    row: Mapping[str, Any],
) -> float:
    row_map = dict(row or {})
    package_name = str(row_map.get("package_name", "") or "")
    package_definitions = dict(recovery.get("package_definitions") or {})
    package_spec = dict(package_definitions.get(package_name) or {})
    for source in (row_map, package_spec):
        total_calls = _safe_float(source.get("budget_total_calls_per_doc"), float("nan"))
        full_doc_share = _safe_float(source.get("full_doc_budget_share"), float("nan"))
        if math.isfinite(total_calls) and math.isfinite(full_doc_share):
            return float(total_calls * full_doc_share)
    parsed_root_share = _package_root_share(package_name)
    if parsed_root_share is None:
        return float("nan")
    return float(parsed_root_share) / 100.0


def _row_has_full_root_supervision(
    recovery: Mapping[str, Any],
    row: Mapping[str, Any],
) -> bool:
    fraction = _row_root_supervision_fraction(recovery, row)
    return math.isfinite(fraction) and fraction >= 0.999


def _iter_scope_train_doc_payloads(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
    canonical_only: bool = True,
) -> List[tuple[int, Mapping[str, Any]]]:
    scope = dict((recovery.get("scopes") or {}).get(str(scope_key)) or {})
    rows_source = scope.get("rows_by_train_docs") or {}
    if isinstance(rows_source, dict):
        ordered_payloads = [
            (int(_safe_int(train_doc_count)), dict(payload or {}))
            for train_doc_count, payload in rows_source.items()
            if int(_safe_int(train_doc_count)) > 0
        ]
    else:
        ordered_payloads = [
            (int(_safe_int((item or {}).get("train_doc_count"))), dict(item or {}))
            for item in list(rows_source or [])
            if int(_safe_int((item or {}).get("train_doc_count"))) > 0
        ]
    ordered_payloads.sort(key=lambda item: int(item[0]))
    if canonical_only:
        allowed = _report_train_doc_count_set(recovery)
        if allowed:
            ordered_payloads = [
                (train_doc_count, payload)
                for train_doc_count, payload in ordered_payloads
                if int(train_doc_count) in allowed
            ]
    return ordered_payloads


def _best_full_root_ceiling_details_by_train_docs(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for train_doc_count, payload in _iter_scope_train_doc_payloads(recovery, scope_key=scope_key):
        candidates: List[Dict[str, Any]] = []
        for row in list((payload or {}).get("rows") or []):
            row_map = dict(row or {})
            if not _row_has_full_root_supervision(recovery, row_map):
                continue
            package_name = str(row_map.get("package_name", "") or "")
            tree_value = _safe_float(row_map.get("tree_test_root_mae"), float("nan"))
            if math.isfinite(tree_value):
                candidates.append(
                    {
                        "value": float(tree_value),
                        "source_kind": "tree",
                        "package_name": package_name,
                        "series_label": "tree",
                    }
                )
            family_rows = dict(row_map.get("fno_family_rows") or {})
            for family_name, family_row in family_rows.items():
                fno_value = _safe_float(dict(family_row or {}).get("test_root_mae"), float("nan"))
                if math.isfinite(fno_value):
                    candidates.append(
                        {
                            "value": float(fno_value),
                            "source_kind": "fno",
                            "package_name": package_name,
                            "family_name": str(family_name),
                            "series_label": _fno_family_label(str(family_name)),
                        }
                    )
            matched_fno_value = _safe_float(row_map.get("fno_reference_test_root_mae"), float("nan"))
            matched_fno_family = str(row_map.get("fno_reference_family", "") or "").strip()
            if math.isfinite(matched_fno_value):
                candidates.append(
                    {
                        "value": float(matched_fno_value),
                        "source_kind": "fno",
                        "package_name": package_name,
                        "family_name": matched_fno_family,
                        "series_label": _fno_family_label(matched_fno_family) if matched_fno_family else "fno",
                    }
                )
        if candidates:
            out[int(train_doc_count)] = min(
                candidates,
                key=lambda item: float(_safe_float(item.get("value"), float("inf"))),
            )
    if out:
        return out
    ceiling_recovery = dict(recovery.get("ceiling_recovery") or {})
    if ceiling_recovery:
        return _best_full_root_ceiling_details_by_train_docs(ceiling_recovery, scope_key=scope_key)
    return out


def _best_full_root_root_mae_by_train_docs(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
) -> Dict[int, float]:
    return {
        int(train_doc_count): float(_safe_float(detail.get("value"), float("nan")))
        for train_doc_count, detail in _best_full_root_ceiling_details_by_train_docs(
            recovery,
            scope_key=scope_key,
        ).items()
        if math.isfinite(_safe_float(detail.get("value"), float("nan")))
    }


def _full_root_ceiling_source_label(detail: Mapping[str, Any]) -> str:
    package_name = str(detail.get("package_name", "") or "")
    package_label = _package_tick_label(package_name) if package_name else "full-root"
    series_label = str(detail.get("series_label", "") or "").strip() or str(
        detail.get("source_kind", "")
    ).strip()
    if not series_label:
        return package_label
    return f"{series_label}@{package_label}"


def _add_full_root_ceiling_source_note(
    ax: Any,
    *,
    detail: Mapping[str, Any] | None,
) -> None:
    detail_map = dict(detail or {})
    if not detail_map:
        return
    label = _full_root_ceiling_source_label(detail_map)
    if not label:
        return
    ax.text(
        0.985,
        0.03,
        f"ceiling: {label}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=BEST_FULL_ROOT_CEILING_COLOR,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.8},
        zorder=11,
    )


def _caption_root_example(train_doc_counts: Sequence[int], *, root_share: int) -> str:
    docs = [int(_safe_int(value)) for value in train_doc_counts if int(_safe_int(value)) > 0]
    if not docs:
        return ""
    train_docs = max(docs)
    root_labels = int(round(float(train_docs) * float(root_share) / 100.0))
    approx_prefix = "~" if (float(train_docs) * float(root_share) / 100.0) != float(root_labels) else ""
    return (
        f"`R{int(root_share)}` at `train_docs={train_docs}` means {train_docs} total training docs "
        f"and {approx_prefix}{root_labels} root-labeled docs."
    )


def _mass_matched_budget_legend_text(root_shares: Sequence[int]) -> str:
    formatted = [
        f"`R{int(root_share)}` = {float(root_share) / 100.0:.2f} doc-equiv / train doc"
        for root_share in root_shares
    ]
    return ", ".join(formatted)


def _max_train_docs(train_doc_counts: Sequence[int]) -> int:
    docs = [int(_safe_int(value)) for value in train_doc_counts if int(_safe_int(value)) > 0]
    return max(docs) if docs else 0


def _root_budget_example_text(train_doc_counts: Sequence[int], *, root_share: int) -> str:
    train_docs = _max_train_docs(train_doc_counts)
    if train_docs <= 0:
        return ""
    budget_docs = int(round(float(train_docs) * float(root_share) / 100.0))
    if int(root_share) >= 100:
        return (
            f"At `train_docs={train_docs}`, `R{int(root_share)}` means "
            f"`{budget_docs}` training-doc equivalents."
        )
    return (
        f"At `train_docs={train_docs}`, `R{int(root_share)}` means about "
        f"`{budget_docs}` root-labeled docs and `{budget_docs}` training-doc equivalents."
    )


def _add_supervision_plot_caption(
    fig: Any,
    *,
    lines: Sequence[str],
    top: float = 0.94,
    bottom: float = 0.085,
) -> None:
    text = " ".join(str(line or "").strip() for line in lines if str(line or "").strip())
    if not text:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
        return
    fig.text(
        0.015,
        0.012,
        text,
        ha="left",
        va="bottom",
        fontsize=8,
        wrap=True,
    )
    fig.tight_layout(rect=(0.0, bottom, 1.0, top))


def _select_primary_lineage_series(
    series_list: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    max_keep: int = 1,
) -> List[Dict[str, Any]]:
    candidates = [
        dict(series or {})
        for series in series_list
        if any(
            math.isfinite(_safe_float(value, float("nan")))
            for value in list(dict(series or {}).get(metric_key) or [])
        )
    ]
    if not candidates:
        return []
    ordered = sorted(
        candidates,
        key=lambda item: (
            -int(_safe_int(item.get("source_tier_rank"), 0)),
            -int(_safe_int(item.get("matching_lineage_count"), 1)),
            str(item.get("lineage_label", "") or ""),
            str(item.get("source_summary_json", "") or ""),
        ),
    )
    return ordered[: max(1, int(max_keep))]


def _add_bottom_legend(
    fig: Any,
    axes: Sequence[Any] | Any,
    *,
    fontsize: int = 8,
    y_anchor: float = 0.02,
    max_columns: int = 4,
) -> None:
    if hasattr(axes, "ravel"):
        axis_list = list(axes.ravel())
    elif isinstance(axes, (list, tuple)):
        axis_list = list(axes)
    else:
        axis_list = [axes]
    legend_items: Dict[str, Any] = {}
    for ax in axis_list:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            text = str(label or "").strip()
            if not text or text.startswith("_") or text in legend_items:
                continue
            legend_items[text] = handle
        existing = ax.get_legend()
        if existing is not None:
            existing.remove()
    if not legend_items:
        return
    fig.legend(
        list(legend_items.values()),
        list(legend_items.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, y_anchor),
        ncol=min(max_columns, len(legend_items)),
        frameon=False,
        fontsize=fontsize,
    )


def _ordered_family_payloads(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
) -> List[Dict[str, Any]]:
    ordered_payloads = _iter_scope_train_doc_payloads(
        recovery,
        scope_key=scope_key,
        canonical_only=True,
    )
    if not ordered_payloads:
        return []
    root_shares = [0, 10, 20, 30, 50, 60, 70, 80, 90, 100]

    def _root_only_package(share: int) -> str:
        return f"full{int(share)}"

    def _root_local_package(share: int) -> str:
        if int(share) == 0:
            return "full0_leaf_full100_internal_count100"
        return f"full{int(share)}_leaf_full100_internal_count100"

    series_payloads: List[Dict[str, Any]] = []
    for train_doc_count, payload in ordered_payloads:
        payload_rows = list((payload or {}).get("rows") or [])
        row_map = _preferred_scope_rows_by_package(payload_rows)
        row_map_by_lineage = _preferred_scope_rows_by_package_and_lineage(payload_rows)

        def _tree_metric(package_name: str, metric_key: str) -> float:
            return _safe_float(
                dict(row_map.get(package_name) or {}).get(metric_key),
                float("nan"),
            )

        def _fno_metric(package_name: str, family: str) -> float:
            row = dict(row_map.get(package_name) or {})
            family_rows = dict(row.get("fno_family_rows") or {})
            return _safe_float(
                dict(family_rows.get(family) or {}).get("test_root_mae"),
                float("nan"),
            )

        series_payloads.append(
            {
                "train_doc_count": int(_safe_int(train_doc_count)),
                "root_shares": list(root_shares),
                "tree_root_only_series": _collapse_identical_lineage_series(
                    _lineage_metric_series(
                        row_map_by_lineage,
                        root_shares=root_shares,
                        package_for_share=_root_only_package,
                        metric_keys=(
                            "tree_test_root_mae",
                            "tree_test_leaf_mae",
                            "tree_test_merge_mae",
                            "tree_test_full_law_objective",
                        ),
                    ),
                    metric_key="tree_test_root_mae",
                ),
                "tree_root_local_series": _collapse_identical_lineage_series(
                    _lineage_metric_series(
                        row_map_by_lineage,
                        root_shares=root_shares,
                        package_for_share=_root_local_package,
                        metric_keys=(
                            "tree_test_root_mae",
                            "tree_test_leaf_mae",
                            "tree_test_merge_mae",
                            "tree_test_full_law_objective",
                        ),
                    ),
                    metric_key="tree_test_root_mae",
                ),
                "tree_root_only_root_mae": [
                    _tree_metric(_root_only_package(share), "tree_test_root_mae")
                    for share in root_shares
                ],
                "tree_root_only_leaf_mae": [
                    _tree_metric(_root_only_package(share), "tree_test_leaf_mae")
                    for share in root_shares
                ],
                "tree_root_only_merge_mae": [
                    _tree_metric(_root_only_package(share), "tree_test_merge_mae")
                    for share in root_shares
                ],
                "tree_root_only_full_law_objective": [
                    _tree_metric(_root_only_package(share), "tree_test_full_law_objective")
                    for share in root_shares
                ],
                "tree_root_local_root_mae": [
                    _tree_metric(_root_local_package(share), "tree_test_root_mae")
                    for share in root_shares
                ],
                "tree_root_local_leaf_mae": [
                    _tree_metric(_root_local_package(share), "tree_test_leaf_mae")
                    for share in root_shares
                ],
                "tree_root_local_merge_mae": [
                    _tree_metric(_root_local_package(share), "tree_test_merge_mae")
                    for share in root_shares
                ],
                "tree_root_local_full_law_objective": [
                    _tree_metric(_root_local_package(share), "tree_test_full_law_objective")
                    for share in root_shares
                ],
                "official_fno_root_mae": [
                    _fno_metric(_root_only_package(share), "official_fno")
                    for share in root_shares
                ],
                "official_fno_sumlen_root_mae": [
                    _fno_metric(_root_only_package(share), "official_fno_sumlen")
                    for share in root_shares
                ],
            }
        )
    return series_payloads


def _local_ablation_payloads(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
) -> List[Dict[str, Any]]:
    ordered_payloads = _iter_scope_train_doc_payloads(
        recovery,
        scope_key=scope_key,
        canonical_only=True,
    )
    if not ordered_payloads:
        return []
    package_order = [
        "full10",
        "full10_leaf_count100",
        "full10_leaf_full100",
        "full10_leaf_full100_internal_depth1_count100",
        "full10_leaf_full100_internal_depth2_count100",
        "full10_leaf_full100_internal_count100",
    ]
    out: List[Dict[str, Any]] = []
    for train_doc_count, payload in ordered_payloads:
        row_map = _preferred_scope_rows_by_package(
            list((payload or {}).get("rows") or [])
        )
        full10_row = dict(row_map.get("full10") or {})
        full10_fno_rows = dict(full10_row.get("fno_family_rows") or {})
        out.append(
            {
                "train_doc_count": int(_safe_int(train_doc_count)),
                "package_order": list(package_order),
                "tree_root_mae": [
                    _safe_float(
                        dict(row_map.get(package_name) or {}).get("tree_test_root_mae"),
                        float("nan"),
                    )
                    for package_name in package_order
                ],
                "official_fno_root_mae": [
                    _safe_float(
                        dict(full10_fno_rows.get("official_fno") or {}).get("test_root_mae"),
                        float("nan"),
                    )
                ]
                * len(package_order),
                "official_fno_sumlen_root_mae": [
                    _safe_float(
                        dict(full10_fno_rows.get("official_fno_sumlen") or {}).get("test_root_mae"),
                        float("nan"),
                    )
                ]
                * len(package_order),
            }
        )
    return out


def _dense_local_root_ladder_payloads(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
) -> List[Dict[str, Any]]:
    ordered_payloads = _ordered_family_payloads(recovery, scope_key=scope_key)
    if not ordered_payloads:
        return []
    out: List[Dict[str, Any]] = []
    for payload in ordered_payloads:
        out.append(
            {
                "train_doc_count": int(_safe_int(payload.get("train_doc_count"))),
                "root_shares": list(payload.get("root_shares") or []),
                "tree_root_local_series": list(payload.get("tree_root_local_series") or []),
                "tree_root_local_root_mae": list(payload.get("tree_root_local_root_mae") or []),
                "official_fno_root_mae": list(payload.get("official_fno_root_mae") or []),
                "official_fno_sumlen_root_mae": list(
                    payload.get("official_fno_sumlen_root_mae") or []
                ),
            }
        )
    return out


def _tree_constant_density_root_ladder_payloads(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
) -> List[Dict[str, Any]]:
    ordered_payloads = _ordered_family_payloads(recovery, scope_key=scope_key)
    if not ordered_payloads:
        return []
    out: List[Dict[str, Any]] = []
    for payload in ordered_payloads:
        out.append(
            {
                "train_doc_count": int(_safe_int(payload.get("train_doc_count"))),
                "root_shares": list(payload.get("root_shares") or []),
                "tree_root_only_series": list(payload.get("tree_root_only_series") or []),
                "tree_root_local_series": list(payload.get("tree_root_local_series") or []),
                "tree_root_only_root_mae": list(
                    payload.get("tree_root_only_root_mae") or []
                ),
                "tree_root_local_root_mae": list(
                    payload.get("tree_root_local_root_mae") or []
                ),
            }
        )
    return out


def _local_law_rate_payloads(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
    root_share: int,
) -> List[Dict[str, Any]]:
    ordered_payloads = _iter_scope_train_doc_payloads(
        recovery,
        scope_key=scope_key,
        canonical_only=True,
    )
    if not ordered_payloads:
        return []
    package_order = list(
        SUPERVISION_RECOVERY_LOCAL_LAW_RATE_PACKAGE_ORDERS.get(int(root_share), ())
    )
    if not package_order:
        return []
    local_law_shares = [0, 10, 20, 50, 100]
    root_package = f"full{int(root_share)}"
    out: List[Dict[str, Any]] = []
    for train_doc_count, payload in ordered_payloads:
        row_map = _preferred_scope_rows_by_package(
            list((payload or {}).get("rows") or [])
        )
        root_row = dict(row_map.get(root_package) or {})
        root_fno_rows = dict(root_row.get("fno_family_rows") or {})
        tree_root_mae = [
            _safe_float(
                dict(row_map.get(package_name) or {}).get("tree_test_root_mae"),
                float("nan"),
            )
            for package_name in package_order
        ]
        if sum(1 for value in tree_root_mae if math.isfinite(value)) <= 1:
            continue
        out.append(
            {
                "train_doc_count": int(_safe_int(train_doc_count)),
                "root_share": int(root_share),
                "root_package": root_package,
                "package_order": list(package_order),
                "local_law_shares": list(local_law_shares),
                "tree_root_mae": tree_root_mae,
                "official_fno_root_mae": [
                    _safe_float(
                        dict(root_fno_rows.get("official_fno") or {}).get("test_root_mae"),
                        float("nan"),
                    )
                ]
                * len(package_order),
                "official_fno_sumlen_root_mae": [
                    _safe_float(
                        dict(root_fno_rows.get("official_fno_sumlen") or {}).get("test_root_mae"),
                        float("nan"),
                    )
                ]
                * len(package_order),
            }
        )
    return out


def _mass_matched_law_rate_payloads(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
    root_share: int,
) -> List[Dict[str, Any]]:
    ordered_payloads = _iter_scope_train_doc_payloads(
        recovery,
        scope_key=scope_key,
        canonical_only=True,
    )
    if not ordered_payloads:
        return []
    package_order = list(
        SUPERVISION_RECOVERY_MASS_MATCHED_RATE_PACKAGE_ORDERS.get(int(root_share), ())
    )
    if not package_order:
        return []
    root_package = f"full{int(root_share)}"
    package_definitions = dict(recovery.get("package_definitions") or {})
    local_rate_percents = [
        100.0
        * _safe_float(
            dict(package_definitions.get(package_name) or {}).get("leaf_label_rate"),
            0.0,
        )
        for package_name in package_order
    ]
    out: List[Dict[str, Any]] = []
    for train_doc_count, payload in ordered_payloads:
        row_map = _preferred_scope_rows_by_package(
            list((payload or {}).get("rows") or [])
        )
        root_row = dict(row_map.get(root_package) or {})
        root_fno_rows = dict(root_row.get("fno_family_rows") or {})
        tree_root_mae = [
            _safe_float(
                dict(row_map.get(package_name) or {}).get("tree_test_root_mae"),
                float("nan"),
            )
            for package_name in package_order
        ]
        if sum(1 for value in tree_root_mae if math.isfinite(value)) <= 1:
            continue
        out.append(
            {
                "train_doc_count": int(_safe_int(train_doc_count)),
                "root_share": int(root_share),
                "root_package": root_package,
                "package_order": list(package_order),
                "local_rate_percents": list(local_rate_percents),
                "tree_root_mae": tree_root_mae,
                "tree_target_mass_per_doc": [
                    _safe_float(
                        dict(row_map.get(package_name) or {}).get(
                            "tree_mass_target_per_doc"
                        ),
                        float("nan"),
                    )
                    for package_name in package_order
                ],
                "tree_realized_effective_mass_per_doc": [
                    _safe_float(
                        dict(row_map.get(package_name) or {}).get(
                            "tree_effective_full_doc_mass_per_doc"
                        ),
                        float("nan"),
                    )
                    for package_name in package_order
                ],
                "tree_computed_doc_review_mass_per_doc": [
                    _safe_float(
                        dict(row_map.get(package_name) or {}).get(
                            "tree_computed_doc_review_mass_per_doc"
                        ),
                        float("nan"),
                    )
                    for package_name in package_order
                ],
                "official_fno_root_mae": [
                    _safe_float(
                        dict(root_fno_rows.get("official_fno") or {}).get("test_root_mae"),
                        float("nan"),
                    )
                ]
                * len(package_order),
                "official_fno_sumlen_root_mae": [
                    _safe_float(
                        dict(root_fno_rows.get("official_fno_sumlen") or {}).get("test_root_mae"),
                        float("nan"),
                    )
                ]
                * len(package_order),
            }
        )
    return out


def _leaf_geometry_payloads(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
    root_share: int,
) -> List[Dict[str, Any]]:
    if str(recovery.get("status", "")) not in {"", "ready"}:
        return []
    tree_family = str(
        recovery.get("tree_family", SUPERVISION_RECOVERY_TREE_FAMILY)
        or SUPERVISION_RECOVERY_TREE_FAMILY
    )
    package_order = list(
        SUPERVISION_RECOVERY_MASS_MATCHED_RATE_PACKAGE_ORDERS.get(int(root_share), ())
    )
    if not package_order:
        return []
    family_rows = [
        dict(row)
        for row in list(recovery.get("family_rows") or [])
        if str(row.get("scope_key", "") or "") == str(scope_key)
    ]
    if not family_rows:
        return []
    out: List[Dict[str, Any]] = []
    train_doc_counts = sorted(
        {
            int(_safe_int(row.get("train_doc_count"), 0))
            for row in family_rows
            if int(_safe_int(row.get("train_doc_count"), 0)) > 0
        }
    )
    root_package = f"full{int(root_share)}"
    for train_doc_count in train_doc_counts:
        scope_train_rows = [
            row
            for row in family_rows
            if int(_safe_int(row.get("train_doc_count"), 0)) == int(train_doc_count)
        ]
        fno_root_rows = {
            str(row.get("baseline_family", "") or ""): dict(row)
            for row in scope_train_rows
            if str(row.get("package_name", "") or "") == root_package
            and str(row.get("baseline_family", "") or "")
            in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES
        }
        grouped_tree_rows: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for row in scope_train_rows:
            if str(row.get("baseline_family", "") or "") != tree_family:
                continue
            if str(row.get("package_name", "") or "") not in package_order:
                continue
            grouped_tree_rows[_effective_fixed_leaf_tokens(row)].append(row)
        if not grouped_tree_rows:
            continue
        geometry_rows: List[Dict[str, Any]] = []
        for fixed_leaf_tokens, candidates in grouped_tree_rows.items():
            if fixed_leaf_tokens <= 0:
                continue
            candidates = [dict(candidate) for candidate in candidates]
            exact_parity_candidates = [
                candidate
                for candidate in candidates
                if _is_exact_full_doc_parity_row(candidate)
                and str(candidate.get("package_name", "") or "") == root_package
            ]
            if exact_parity_candidates:
                chosen = dict(exact_parity_candidates[0])
            else:
                finite_candidates = [
                    candidate
                    for candidate in candidates
                    if math.isfinite(_tree_root_mae_from_family_row(candidate))
                ]
                if finite_candidates:
                    chosen = min(
                        finite_candidates,
                        key=lambda candidate: _tree_root_mae_from_family_row(candidate),
                    )
                else:
                    chosen = dict(candidates[0])
            geometry_rows.append(chosen)
        geometry_rows.sort(
            key=lambda row: (
                _effective_leaves_per_doc(row),
                _effective_fixed_leaf_tokens(row),
            )
        )
        if not geometry_rows:
            continue
        out.append(
            {
                "train_doc_count": int(train_doc_count),
                "root_share": int(root_share),
                "rows": geometry_rows,
                "official_fno_root_mae": _safe_float(
                    dict(fno_root_rows.get("official_fno") or {}).get(
                        "test_root_mae_mean",
                        dict(fno_root_rows.get("official_fno") or {}).get(
                            "test_root_mae"
                        ),
                    ),
                    float("nan"),
                ),
                "official_fno_sumlen_root_mae": _safe_float(
                    dict(fno_root_rows.get("official_fno_sumlen") or {}).get(
                        "test_root_mae_mean",
                        dict(fno_root_rows.get("official_fno_sumlen") or {}).get(
                            "test_root_mae"
                        ),
                    ),
                    float("nan"),
                ),
            }
        )
    return out


def _recovery_geometry_groups(
    recovery: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    if str(recovery.get("status", "")) not in {"", "ready"}:
        return []
    tree_family = str(
        recovery.get("tree_family", SUPERVISION_RECOVERY_TREE_FAMILY)
        or SUPERVISION_RECOVERY_TREE_FAMILY
    )
    groups: Dict[str, Dict[str, Any]] = {}
    for raw_row in list(recovery.get("family_rows") or []):
        row = dict(raw_row or {})
        if str(row.get("baseline_family", "") or "") != tree_family:
            continue
        if str(row.get("comparison_arm", "") or "") not in {
            "",
            SUPERVISION_RECOVERY_PRIMARY_COMPARISON_ARM,
        }:
            continue
        geometry_key = _row_geometry_identity(row)
        if not geometry_key:
            continue
        group = groups.setdefault(
            geometry_key,
            {
                "geometry_key": geometry_key,
                "geometry_label": _row_geometry_label(row),
                "pipeline_supervision_recovery_leaf_tokens": int(
                    _safe_int(
                        row.get("pipeline_supervision_recovery_leaf_tokens"),
                        0,
                    )
                ),
                "requested_fixed_leaf_tokens": int(
                    _requested_fixed_leaf_tokens(row)
                ),
                "executed_fixed_leaf_tokens": int(
                    _effective_fixed_leaf_tokens(row)
                ),
                "executed_leaves_per_doc": int(_effective_leaves_per_doc(row)),
                "row_count": 0,
            },
        )
        group["row_count"] = int(group.get("row_count", 0)) + 1
        if not str(group.get("geometry_label", "") or "").strip():
            group["geometry_label"] = _row_geometry_label(row)
        group["pipeline_supervision_recovery_leaf_tokens"] = max(
            int(_safe_int(group.get("pipeline_supervision_recovery_leaf_tokens"), 0)),
            int(_safe_int(row.get("pipeline_supervision_recovery_leaf_tokens"), 0)),
        )
        group["requested_fixed_leaf_tokens"] = max(
            int(_safe_int(group.get("requested_fixed_leaf_tokens"), 0)),
            int(_requested_fixed_leaf_tokens(row)),
        )
        group["executed_fixed_leaf_tokens"] = max(
            int(_safe_int(group.get("executed_fixed_leaf_tokens"), 0)),
            int(_effective_fixed_leaf_tokens(row)),
        )
        group["executed_leaves_per_doc"] = max(
            int(_safe_int(group.get("executed_leaves_per_doc"), 0)),
            int(_effective_leaves_per_doc(row)),
        )
    return sorted(
        [dict(group) for group in groups.values()],
        key=lambda group: (
            -(
                int(_safe_int(group.get("pipeline_supervision_recovery_leaf_tokens"), 0))
                or int(_safe_int(group.get("requested_fixed_leaf_tokens"), 0))
                or int(_safe_int(group.get("executed_fixed_leaf_tokens"), 0))
                or 0
            ),
            int(_safe_int(group.get("executed_leaves_per_doc"), 0)) or 10**9,
            str(group.get("geometry_label", "")),
        ),
    )


def _filter_recovery_to_geometry(
    recovery: Mapping[str, Any],
    *,
    geometry_key: str,
) -> Dict[str, Any]:
    geometry_key = str(geometry_key or "").strip()
    if not geometry_key:
        return dict(recovery)
    tree_family = str(
        recovery.get("tree_family", SUPERVISION_RECOVERY_TREE_FAMILY)
        or SUPERVISION_RECOVERY_TREE_FAMILY
    )

    def _keep_family_row(row: Mapping[str, Any]) -> bool:
        baseline_family = str(row.get("baseline_family", "") or "")
        if baseline_family in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES:
            return True
        return _row_geometry_identity(row) == geometry_key

    filtered_family_rows = [
        dict(row)
        for row in list(recovery.get("family_rows") or [])
        if _keep_family_row(dict(row or {}))
    ]
    scope_tree_references = dict(recovery.get("scope_tree_references") or {})
    filtered_scope_tree_references: Dict[str, Dict[str, Any]] = {}
    for row in filtered_family_rows:
        row_map = dict(row or {})
        if str(row_map.get("baseline_family", "") or "") != tree_family:
            continue
        scope_key = str(row_map.get("scope_key", "") or "").strip()
        if not scope_key or scope_key in filtered_scope_tree_references:
            continue
        existing = dict(scope_tree_references.get(scope_key) or {})
        filtered_scope_tree_references[scope_key] = {
            **existing,
            "scope_key": scope_key,
            "scope_label": str(
                row_map.get("scope_label", existing.get("scope_label", scope_key))
                or existing.get("scope_label", scope_key)
            ),
            "requested_fixed_leaf_tokens": int(_requested_fixed_leaf_tokens(row_map)),
            "executed_fixed_leaf_tokens": int(_effective_fixed_leaf_tokens(row_map)),
            "fixed_leaf_tokens": int(_effective_fixed_leaf_tokens(row_map)),
            "executed_leaves_per_doc": int(_effective_leaves_per_doc(row_map)),
            "supervision_recovery_geometry_key": geometry_key,
            "supervision_recovery_geometry_label": str(
                row_map.get("supervision_recovery_geometry_label", "") or ""
            ),
        }
    filtered_scopes: Dict[str, Dict[str, Any]] = {}
    for scope_key, raw_scope in dict(recovery.get("scopes") or {}).items():
        scope = dict(raw_scope or {})
        raw_rows_by_train_docs = scope.get("rows_by_train_docs") or {}
        if isinstance(raw_rows_by_train_docs, Mapping):
            row_groups = [
                dict(item or {})
                for item in dict(raw_rows_by_train_docs).values()
            ]
        else:
            row_groups = [dict(item or {}) for item in list(raw_rows_by_train_docs)]
        rows_by_train_docs: Dict[str, Dict[str, Any]] = {}
        best_tree_by_train_docs: Dict[str, Dict[str, Any]] = {}
        for item_map in row_groups:
            train_doc_count = int(_safe_int(item_map.get("train_doc_count"), 0))
            rows = [
                dict(row)
                for row in list(item_map.get("rows") or [])
                if _row_geometry_identity(dict(row or {})) == geometry_key
            ]
            rows_by_train_docs[str(train_doc_count)] = {
                "train_doc_count": int(train_doc_count),
                "rows": rows,
            }
            finite_rows = [
                row
                for row in rows
                if math.isfinite(_safe_float(row.get("tree_test_root_mae"), float("nan")))
            ]
            best_tree_by_train_docs[str(int(train_doc_count))] = (
                min(
                    finite_rows,
                    key=lambda row: _safe_float(row.get("tree_test_root_mae"), float("inf")),
                )
                if finite_rows
                else {}
            )
        filtered_scopes[str(scope_key)] = {
            **scope,
            "rows_by_train_docs": rows_by_train_docs,
            "dense_anchor_rows": [
                dict(row)
                for row in list(scope.get("dense_anchor_rows") or [])
                if _row_geometry_identity(dict(row or {})) == geometry_key
            ],
            "best_tree_by_train_docs": best_tree_by_train_docs,
        }
    filtered = {
        **dict(recovery),
        "family_rows": filtered_family_rows,
        "scopes": filtered_scopes,
        "scope_tree_references": filtered_scope_tree_references
        or scope_tree_references,
    }
    filtered["best_tree_summary"] = _best_tree_summary_rows(filtered)
    filtered["geometry_groups"] = [
        group
        for group in _recovery_geometry_groups(filtered)
        if str(group.get("geometry_key", "")) == geometry_key
    ]
    return filtered


def _summary_filtered_to_geometry(
    summary: Mapping[str, Any],
    *,
    geometry_key: str,
) -> Dict[str, Any]:
    filtered_recovery = _filter_recovery_to_geometry(
        dict(summary.get("supervision_recovery") or {}),
        geometry_key=geometry_key,
    )
    filtered_summary = dict(summary)
    filtered_summary["supervision_recovery"] = filtered_recovery
    filtered_summary["best_tree_summary"] = _best_tree_summary_rows(filtered_recovery)
    filtered_summary["stability_warnings"] = _supervision_recovery_non_monotone_warnings(
        filtered_recovery
    )
    return filtered_summary


def _geometry_section_suffix(group: Mapping[str, Any]) -> str:
    label = str(group.get("geometry_label", "") or "").strip()
    leaves_per_doc = int(_safe_int(group.get("executed_leaves_per_doc"), 0))
    bits = [label] if label else []
    if leaves_per_doc > 0:
        bits.append(f"{leaves_per_doc} leaves/doc")
    if not bits:
        return ""
    return f" ({', '.join(bits)})"


def _geometry_context_lines(group: Mapping[str, Any]) -> List[str]:
    bits: List[str] = []
    if int(_safe_int(group.get("requested_fixed_leaf_tokens"), 0)) > 0:
        bits.append(
            f"requested `fixed_leaf_tokens={int(_safe_int(group.get('requested_fixed_leaf_tokens'), 0))}`"
        )
    if int(_safe_int(group.get("executed_fixed_leaf_tokens"), 0)) > 0:
        bits.append(
            f"executed `fixed_leaf_tokens={int(_safe_int(group.get('executed_fixed_leaf_tokens'), 0))}`"
        )
    if int(_safe_int(group.get("executed_leaves_per_doc"), 0)) > 0:
        bits.append(
            f"`{int(_safe_int(group.get('executed_leaves_per_doc'), 0))} leaves/doc`"
        )
    if not bits:
        return []
    return ["- Geometry group: " + ", ".join(bits) + "."]


def _has_local_law_rate_payloads(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
    root_share: int,
) -> bool:
    return bool(
        _local_law_rate_payloads(
            recovery,
            scope_key=scope_key,
            root_share=root_share,
        )
    )


def _has_mass_matched_law_rate_payloads(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
    root_share: int,
) -> bool:
    return bool(
        _mass_matched_law_rate_payloads(
            recovery,
            scope_key=scope_key,
            root_share=root_share,
        )
    )


def _r10_local_law_rate_payloads(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
) -> List[Dict[str, Any]]:
    return _local_law_rate_payloads(recovery, scope_key=scope_key, root_share=10)


def _has_r10_local_law_rate_payloads(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
) -> bool:
    return _has_local_law_rate_payloads(recovery, scope_key=scope_key, root_share=10)


def _local_law_endpoint_rows(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
    root_share: int,
) -> List[Dict[str, Any]]:
    package_order = list(
        SUPERVISION_RECOVERY_LOCAL_LAW_RATE_PACKAGE_ORDERS.get(int(root_share), ())
    )
    ordered_payloads = _iter_scope_train_doc_payloads(
        recovery,
        scope_key=scope_key,
        canonical_only=True,
    )
    if not ordered_payloads or len(package_order) < 2:
        return []
    baseline_package = str(package_order[0])
    endpoint_package = str(package_order[-1])
    rows: List[Dict[str, Any]] = []
    for train_doc_count, payload in ordered_payloads:
        row_map = _preferred_scope_rows_by_package(
            list((payload or {}).get("rows") or [])
        )
        baseline_row = dict(row_map.get(baseline_package) or {})
        endpoint_row = dict(row_map.get(endpoint_package) or {})
        if not baseline_row or not endpoint_row:
            continue
        baseline_tree = _safe_float(baseline_row.get("tree_test_root_mae"), float("nan"))
        endpoint_tree = _safe_float(endpoint_row.get("tree_test_root_mae"), float("nan"))
        if not math.isfinite(baseline_tree) or not math.isfinite(endpoint_tree):
            continue
        baseline_fno_rows = dict(baseline_row.get("fno_family_rows") or {})
        official_fno = _safe_float(
            dict(baseline_fno_rows.get("official_fno") or {}).get("test_root_mae"),
            float("nan"),
        )
        official_fno_sumlen = _safe_float(
            dict(baseline_fno_rows.get("official_fno_sumlen") or {}).get("test_root_mae"),
            float("nan"),
        )
        rows.append(
            {
                "train_doc_count": int(_safe_int(train_doc_count)),
                "baseline_package": baseline_package,
                "endpoint_package": endpoint_package,
                "baseline_tree_root_mae": baseline_tree,
                "endpoint_tree_root_mae": endpoint_tree,
                "delta_vs_baseline_tree": float(endpoint_tree - baseline_tree),
                "official_fno_root_mae": official_fno,
                "official_fno_sumlen_root_mae": official_fno_sumlen,
                "delta_vs_official_fno": float(endpoint_tree - official_fno)
                if math.isfinite(official_fno)
                else float("nan"),
                "delta_vs_official_fno_sumlen": float(endpoint_tree - official_fno_sumlen)
                if math.isfinite(official_fno_sumlen)
                else float("nan"),
            }
        )
    return rows


def _add_best_full_root_ceiling_line(
    ax: Any,
    *,
    y_value: Any,
    label: str | None = None,
) -> None:
    ceiling = _safe_float(y_value, float("nan"))
    if not math.isfinite(ceiling):
        return
    # Add a thin white underlay first so the dotted ceiling remains visible
    # even when it lands exactly on top of an existing solid series.
    ax.axhline(
        ceiling,
        color="white",
        linestyle="-",
        linewidth=4.2,
        alpha=0.95,
        zorder=9,
    )
    ax.axhline(
        ceiling,
        color=BEST_FULL_ROOT_CEILING_COLOR,
        linestyle=":",
        linewidth=2.4,
        label=label,
        alpha=1.0,
        zorder=10,
    )


def _plot_supervision_recovery_dense_anchor(
    recovery: Mapping[str, Any],
    output_path: Path,
) -> bool:
    if str(recovery.get("status", "")) not in {"", "ready"}:
        return False
    scope_summaries = dict(recovery.get("scopes") or {})
    scope = dict(
        scope_summaries.get(_preferred_recoverable_scope_key(scope_summaries)) or {}
    )
    rows = [dict(row) for row in list(scope.get("dense_anchor_rows") or [])]
    if not rows:
        return False
    allowed_docs = _report_train_doc_count_set(recovery)
    if allowed_docs:
        rows = [
            row
            for row in rows
            if int(_safe_int(row.get("train_doc_count"), 0)) in allowed_docs
        ]
    if not rows:
        return False
    rows.sort(key=lambda row: int(_safe_int(row.get("train_doc_count"))))
    ordered_docs = sorted(
        {
            int(_safe_int(row.get("train_doc_count"), 0))
            for row in rows
            if int(_safe_int(row.get("train_doc_count"), 0)) > 0
        }
    )
    if not ordered_docs:
        return False
    x = [float(train_doc_count) for train_doc_count in ordered_docs]
    rows_by_doc: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    rows_by_lineage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        train_doc_count = int(_safe_int(row.get("train_doc_count"), 0))
        if train_doc_count <= 0:
            continue
        rows_by_doc[train_doc_count].append(dict(row))
        rows_by_lineage[_lineage_key_for_row(row)].append(dict(row))
    tree_series = _collapse_identical_lineage_series(
        [
            {
                "lineage_key": lineage_key,
                "lineage_label": _lineage_label_for_row(_preferred_recovery_row(lineage_rows)),
                "source_tier_rank": int(
                    _safe_int(
                        _preferred_recovery_row(lineage_rows).get("source_tier_rank"),
                        0,
                    )
                ),
                "source_summary_json": str(
                    _preferred_recovery_row(lineage_rows).get("source_summary_json", "") or ""
                ),
                "tree_test_root_mae": [
                    _safe_float(
                        dict(
                            {
                                int(_safe_int(row.get("train_doc_count"), 0)): dict(row)
                                for row in lineage_rows
                                if int(_safe_int(row.get("train_doc_count"), 0)) > 0
                            }.get(train_doc_count, {})
                        ).get("tree_test_root_mae"),
                        float("nan"),
                    )
                    for train_doc_count in ordered_docs
                ],
            }
            for lineage_key, lineage_rows in rows_by_lineage.items()
        ],
        metric_key="tree_test_root_mae",
    )
    fno_y_by_family = {
        family: [
            _safe_float(
                dict(
                    (
                        _preferred_recovery_row(rows_by_doc.get(train_doc_count, [])).get(
                            "fno_family_rows",
                            {},
                        )
                        or {}
                    ).get(family)
                    or {}
                ).get("test_root_mae"),
                float("nan"),
            )
            for train_doc_count in ordered_docs
        ]
        for family in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    family_styles = {
        "official_fno": {"color": FNO_OFFICIAL_COLOR, "marker": "^"},
        "official_fno_sumlen": {"color": FNO_SUMLEN_COLOR, "marker": "D"},
    }
    for family in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES:
        family_y = fno_y_by_family.get(family, [])
        if not any(math.isfinite(value) for value in family_y):
            continue
        style = family_styles.get(family, {"color": NEUTRAL_COLOR, "marker": "o"})
        ax.plot(
            x,
            family_y,
            marker=str(style["marker"]),
            linewidth=2.0,
            color=str(style["color"]),
            label=_fno_family_label(family),
        )
    lineage_dash_cycle = ["--", "-.", ":", "-"]
    display_tree_series = _select_primary_lineage_series(
        tree_series,
        metric_key="tree_test_root_mae",
        max_keep=1,
    )
    for lineage_idx, series in enumerate(display_tree_series):
        tree_y = list(series.get("tree_test_root_mae") or [])
        if not any(math.isfinite(value) for value in tree_y):
            continue
        label_suffix = (
            f" · {series.get('lineage_label')}"
            if len(tree_series) > 1
            else ""
        )
        ax.plot(
            x,
            tree_y,
            marker="o",
            linewidth=2.0,
            linestyle=lineage_dash_cycle[lineage_idx % len(lineage_dash_cycle)],
            alpha=max(0.45, 1.0 - 0.12 * lineage_idx),
            color=TREE_PRIMARY_COLOR,
            label=f"tree_neural{label_suffix}",
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Train docs")
    ax.set_ylabel("Test root MAE")
    ax.set_title("Recoverable Full-Supervision Baseline")
    ax.grid(True, alpha=0.25)
    _add_bottom_legend(fig, [ax], fontsize=8, y_anchor=0.05, max_columns=3)
    fig.subplots_adjust(bottom=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _exact_full_doc_canary_scope_rows(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    scope = dict((recovery.get("scopes") or {}).get(str(scope_key)) or {})
    scope_label = str(scope.get("scope_label", scope_key) or scope_key)
    for train_doc_count, payload in _iter_scope_train_doc_payloads(recovery, scope_key=scope_key):
        row_map = _preferred_scope_rows_by_package(
            list((payload or {}).get("rows") or [])
        )
        full100_row = dict(row_map.get("full100") or {})
        if not full100_row or not _is_exact_full_doc_parity_row(full100_row):
            continue
        rows.append(
            {
                "scope_key": str(scope_key),
                "scope_label": scope_label,
                "train_doc_count": int(train_doc_count),
                "tree_root_mae": _safe_float(full100_row.get("tree_test_root_mae"), float("nan")),
                "official_fno_root_mae": _row_fno_family_value(full100_row, "official_fno"),
                "official_fno_sumlen_root_mae": _row_fno_family_value(
                    full100_row,
                    "official_fno_sumlen",
                ),
                "effective_fixed_leaf_tokens": int(_effective_fixed_leaf_tokens(full100_row)),
                "effective_leaves_per_doc": int(_effective_leaves_per_doc(full100_row)),
                "computed_assumed_doc_tokens": int(
                    _safe_int(full100_row.get("computed_assumed_doc_tokens"), 0)
                ),
            }
        )
    rows.sort(key=lambda row: int(_safe_int(row.get("train_doc_count"), 0)))
    return rows


def _plot_supervision_recovery_exact_full_doc_canary(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
    output_path: Path,
) -> bool:
    if str(recovery.get("status", "")) not in {"", "ready"}:
        return False
    rows = _exact_full_doc_canary_scope_rows(recovery, scope_key=scope_key)
    if not rows:
        return False
    x = [float(_safe_int(row.get("train_doc_count"), 0)) for row in rows]
    tree_y = [_safe_float(row.get("tree_root_mae"), float("nan")) for row in rows]
    official_fno_y = [
        _safe_float(row.get("official_fno_root_mae"), float("nan")) for row in rows
    ]
    official_fno_sumlen_y = [
        _safe_float(row.get("official_fno_sumlen_root_mae"), float("nan")) for row in rows
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(
        x,
        tree_y,
        marker="o",
        linewidth=2.0,
        color=TREE_PRIMARY_COLOR,
        label="tree_neural",
    )
    if any(math.isfinite(value) for value in official_fno_y):
        ax.plot(
            x,
            official_fno_y,
            marker="^",
            linewidth=2.0,
            color=FNO_OFFICIAL_COLOR,
            label="official_fno",
        )
    if any(math.isfinite(value) for value in official_fno_sumlen_y):
        ax.plot(
            x,
            official_fno_sumlen_y,
            marker="D",
            linewidth=2.0,
            color=FNO_SUMLEN_COLOR,
            label="official_fno_sumlen",
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Train docs")
    ax.set_ylabel("Test root MAE")
    scope_label = str(rows[0].get("scope_label", scope_key) or scope_key)
    ax.set_title(f"{scope_label}: Exact Full-Doc Canary")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    _add_supervision_plot_caption(
        fig,
        lines=[
            "This canary keeps only `full100` rows that execute at `1 leaf/doc` and are marked `parity_mode=exact_full_doc`.",
            "Compare parity primarily against `official_fno`; `official_fno_sumlen` is shown separately as an alternative FNO family.",
        ],
        top=0.93,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_supervision_recovery_ladder(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
    output_path: Path,
) -> bool:
    if str(recovery.get("status", "")) not in {"", "ready"}:
        return False
    scope = dict((recovery.get("scopes") or {}).get(str(scope_key)) or {})
    ordered_payloads = _iter_scope_train_doc_payloads(
        recovery,
        scope_key=scope_key,
        canonical_only=True,
    )
    if not ordered_payloads:
        return False
    package_order = [str(item) for item in list(recovery.get("package_order") or SUPERVISION_RECOVERY_PACKAGE_ORDER)]
    n_panels = len(ordered_payloads)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(max(9.0, 6.6 * n_panels), 6.3),
        sharey=True,
    )
    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    elif not isinstance(axes, list):
        axes = [axes]
    tree_color = TREE_PRIMARY_COLOR
    fno_colors = {
        "official_fno": FNO_OFFICIAL_COLOR,
        "official_fno_sumlen": FNO_SUMLEN_COLOR,
    }
    best_full_root_by_docs = _best_full_root_root_mae_by_train_docs(
        recovery,
        scope_key=scope_key,
    )
    best_full_root_details = _best_full_root_ceiling_details_by_train_docs(
        recovery,
        scope_key=scope_key,
    )
    x = list(range(len(package_order)))
    width = 0.24
    caption_root_share = next(
        (
            int(root_share)
            for root_share in (
                _package_root_share(package_name) for package_name in package_order
            )
            if root_share is not None
        ),
        0,
    )
    for idx, ((train_doc_count, payload), ax) in enumerate(zip(ordered_payloads, axes)):
        row_map = _preferred_scope_rows_by_package(
            list((payload or {}).get("rows") or [])
        )
        tree_y = [
            _safe_float(
                dict(row_map.get(package_name) or {}).get("tree_test_root_mae"),
                float("nan"),
            )
            for package_name in package_order
        ]
        fno_y_by_family = {
            family: [
                _safe_float(
                    dict(
                        dict(row_map.get(package_name) or {}).get("fno_family_rows") or {}
                    ).get(family, {}).get("test_root_mae"),
                    float("nan"),
                )
                for package_name in package_order
            ]
            for family in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES
        }
        tree_x = [value - width for value in x]
        family_x = {
            "official_fno": [value for value in x],
            "official_fno_sumlen": [value + width for value in x],
        }
        ax.bar(
            tree_x,
            tree_y,
            width=width,
            color=tree_color,
            alpha=0.9,
            label="tree" if idx == 0 else None,
        )
        for family in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES:
            ax.bar(
                family_x[family],
                fno_y_by_family[family],
                width=width,
                color=fno_colors.get(family, NEUTRAL_COLOR),
                alpha=0.78,
                label=_fno_family_label(family) if idx == 0 else None,
            )
        _add_best_full_root_ceiling_line(
            ax,
            y_value=best_full_root_by_docs.get(int(_safe_int(train_doc_count))),
            label="best full-root ceiling" if idx == 0 else None,
        )
        _add_full_root_ceiling_source_note(
            ax,
            detail=best_full_root_details.get(int(_safe_int(train_doc_count))),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_package_tick_label(package_name) for package_name in package_order],
            rotation=50,
            ha="right",
            rotation_mode="anchor",
            fontsize=7.5,
        )
        ax.set_title(f"train_docs={int(_safe_int(train_doc_count))}")
        ax.grid(True, axis="y", alpha=0.25)
        if idx == 0:
            ax.set_ylabel("Test root MAE")
        ax.set_xlabel("Supervision package")
    fig.suptitle(f"{scope.get('scope_label', scope_key)}: All Supervision Settings")
    if axes:
        axes[0].legend(frameon=False, fontsize=8)
    _add_supervision_plot_caption(
        fig,
        lines=[
            (
                _caption_root_example(
                    [int(_safe_int(train_doc_count)) for train_doc_count, _ in ordered_payloads],
                    root_share=int(caption_root_share),
                )
                if int(caption_root_share) > 0
                else ""
            ),
            "The dotted benchmark line shows the best result with 100% root supervision at the same train-doc count.",
        ],
        top=0.95,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_supervision_recovery_ordered_families(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
    output_path: Path,
    min_root_share: int = 0,
) -> bool:
    if str(recovery.get("status", "")) not in {"", "ready"}:
        return False
    scope = dict((recovery.get("scopes") or {}).get(str(scope_key)) or {})
    ordered_payloads = _ordered_family_payloads(recovery, scope_key=scope_key)
    if not ordered_payloads:
        return False
    fig, axes = plt.subplots(
        1,
        len(ordered_payloads),
        figsize=(max(11.0, 5.2 * len(ordered_payloads)), 4.6),
        sharey=True,
        squeeze=False,
    )
    family_styles = {
        "tree_root_only": {"color": TREE_PRIMARY_COLOR, "marker": "o", "linestyle": "--", "label": "tree root-only"},
        "tree_root_local": {"color": TREE_LOCAL_COLOR, "marker": "s", "linestyle": "-", "label": "tree + leaf/internal"},
        "official_fno": {"color": FNO_OFFICIAL_COLOR, "marker": "^", "label": "official_fno"},
        "official_fno_sumlen": {"color": FNO_SUMLEN_COLOR, "marker": "D", "label": "official_fno_sumlen"},
    }
    available_root_shares = [int(_safe_int(value)) for value in ordered_payloads[0]["root_shares"]]
    best_full_root_by_docs = _best_full_root_root_mae_by_train_docs(
        recovery,
        scope_key=scope_key,
    )
    best_full_root_details = _best_full_root_ceiling_details_by_train_docs(
        recovery,
        scope_key=scope_key,
    )
    keep_indices = [
        idx for idx, share in enumerate(available_root_shares) if share >= int(min_root_share)
    ]
    if not keep_indices:
        return False
    x = list(range(len(keep_indices)))
    x_labels = [f"{available_root_shares[idx]}%" for idx in keep_indices]
    axes = list(axes.ravel())
    for idx, (payload, ax) in enumerate(zip(ordered_payloads, axes)):
        tree_root_only_series = list(payload.get("tree_root_only_series") or [])
        tree_root_local_series = list(payload.get("tree_root_local_series") or [])
        lineage_dash_cycle = ["--", "-.", ":"]
        if tree_root_only_series or tree_root_local_series:
            for series_name, lineage_series in (
                (
                    "tree_root_only",
                    _select_primary_lineage_series(
                        tree_root_only_series,
                        metric_key="tree_test_root_mae",
                        max_keep=1,
                    ),
                ),
                (
                    "tree_root_local",
                    _select_primary_lineage_series(
                        tree_root_local_series,
                        metric_key="tree_test_root_mae",
                        max_keep=1,
                    ),
                ),
            ):
                style = family_styles[series_name]
                use_suffix = len(lineage_series) > 1
                for lineage_idx, series in enumerate(lineage_series):
                    y = [
                        list(series.get("tree_test_root_mae") or [])[keep_idx]
                        for keep_idx in keep_indices
                    ]
                    if not any(math.isfinite(value) for value in y):
                        continue
                    label_suffix = (
                        f" · {series.get('lineage_label')}"
                        if use_suffix
                        else ""
                    )
                    ax.plot(
                        x,
                        y,
                        marker=str(style["marker"]),
                        linewidth=2.0,
                        linestyle=(
                            lineage_dash_cycle[lineage_idx % len(lineage_dash_cycle)]
                            if series_name == "tree_root_only"
                            else "-"
                        ),
                        alpha=max(0.45, 1.0 - 0.12 * lineage_idx),
                        color=str(style["color"]),
                        label=(
                            f"{style['label']}{label_suffix}"
                            if idx == 0
                            else None
                        ),
                    )
        else:
            series_map = {
                "tree_root_only": [list(payload["tree_root_only_root_mae"])[idx] for idx in keep_indices],
                "tree_root_local": [list(payload["tree_root_local_root_mae"])[idx] for idx in keep_indices],
            }
            for series_name, y in series_map.items():
                if not any(math.isfinite(value) for value in y):
                    continue
                style = family_styles[series_name]
                ax.plot(
                    x,
                    y,
                    marker=str(style["marker"]),
                    linewidth=2.0,
                    linestyle=str(style.get("linestyle", "-")),
                    color=str(style["color"]),
                    label=str(style["label"]) if idx == 0 else None,
                )
        for series_name, y in {
            "official_fno": [list(payload["official_fno_root_mae"])[idx] for idx in keep_indices],
            "official_fno_sumlen": [
                list(payload["official_fno_sumlen_root_mae"])[idx] for idx in keep_indices
            ],
        }.items():
            if not any(math.isfinite(value) for value in y):
                continue
            style = family_styles[series_name]
            ax.plot(
                x,
                y,
                marker=str(style["marker"]),
                linewidth=2.0,
                linestyle=str(style.get("linestyle", "-")),
                color=str(style["color"]),
                label=str(style["label"]) if idx == 0 else None,
            )
        _add_best_full_root_ceiling_line(
            ax,
            y_value=best_full_root_by_docs.get(int(_safe_int(payload["train_doc_count"]))),
            label="best full-root ceiling" if idx == 0 else None,
        )
        _add_full_root_ceiling_source_note(
            ax,
            detail=best_full_root_details.get(int(_safe_int(payload["train_doc_count"]))),
        )
        ax.set_title(f"train_docs={int(_safe_int(payload['train_doc_count']))}")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=0)
        ax.set_xlabel("Root supervision share")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.set_ylabel("Test root MAE")
    scope_label = str(scope.get("scope_label", scope_key) or scope_key)
    fig.suptitle(f"{scope_label}: Root-Supervision Sweep", y=1.02)
    _add_bottom_legend(fig, axes, fontsize=8, y_anchor=0.06, max_columns=4)
    _add_supervision_plot_caption(
        fig,
        lines=[
            "X-axis: root supervision share at fixed train-doc counts.",
            _caption_root_example(
                [int(_safe_int(payload.get("train_doc_count"))) for payload in ordered_payloads],
                root_share=10,
            ),
            "The FNO lines are matched root-only baselines. The dotted benchmark line shows the best result with 100% root supervision at the same train-doc count.",
        ],
        top=0.92,
        bottom=0.18,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_supervision_recovery_tree_diagnostics(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
    output_path: Path,
) -> bool:
    if str(recovery.get("status", "")) not in {"", "ready"}:
        return False
    scope = dict((recovery.get("scopes") or {}).get(str(scope_key)) or {})
    ordered_payloads = _ordered_family_payloads(recovery, scope_key=scope_key)
    if not ordered_payloads:
        return False
    metric_specs = [
        ("tree_root_only_leaf_mae", "tree_root_local_leaf_mae", "Leaf MAE"),
        ("tree_root_only_merge_mae", "tree_root_local_merge_mae", "Merge MAE"),
        ("tree_root_only_full_law_objective", "tree_root_local_full_law_objective", "Full-Law Objective"),
    ]
    fig, axes = plt.subplots(
        len(metric_specs),
        len(ordered_payloads),
        figsize=(max(11.0, 4.8 * len(ordered_payloads)), 9.2),
        sharex=True,
        squeeze=False,
    )
    x = list(range(len(ordered_payloads[0]["root_shares"])))
    x_labels = [f"{share}%" for share in ordered_payloads[0]["root_shares"]]
    for col_idx, payload in enumerate(ordered_payloads):
        for row_idx, (root_only_key, root_local_key, title) in enumerate(metric_specs):
            ax = axes[row_idx][col_idx]
            lineage_dash_cycle = ["--", "-.", ":"]
            lineage_payloads = [
                (
                    "tree root-only",
                    _select_primary_lineage_series(
                        list(payload.get("tree_root_only_series") or []),
                        metric_key="tree_test_leaf_mae" if root_only_key.endswith("leaf_mae") else (
                            "tree_test_merge_mae" if root_only_key.endswith("merge_mae") else (
                                "tree_test_full_law_objective" if root_only_key.endswith("full_law_objective") else "tree_test_root_mae"
                            )
                        ),
                        max_keep=1,
                    ),
                    TREE_PRIMARY_COLOR,
                    "o",
                    "tree_test_" + root_only_key.removeprefix("tree_root_only_"),
                ),
                (
                    "tree + leaf/internal",
                    _select_primary_lineage_series(
                        list(payload.get("tree_root_local_series") or []),
                        metric_key="tree_test_leaf_mae" if root_local_key.endswith("leaf_mae") else (
                            "tree_test_merge_mae" if root_local_key.endswith("merge_mae") else (
                                "tree_test_full_law_objective" if root_local_key.endswith("full_law_objective") else "tree_test_root_mae"
                            )
                        ),
                        max_keep=1,
                    ),
                    TREE_LOCAL_COLOR,
                    "s",
                    "tree_test_" + root_local_key.removeprefix("tree_root_local_"),
                ),
            ]
            rendered_series = False
            for base_label, lineage_series, color, marker, metric_key in lineage_payloads:
                use_suffix = len(lineage_series) > 1
                for lineage_idx, series in enumerate(lineage_series):
                    y = list(series.get(metric_key) or [])
                    if not any(math.isfinite(value) for value in y):
                        continue
                    rendered_series = True
                    label_suffix = (
                        f" · {series.get('lineage_label')}"
                        if use_suffix
                        else ""
                    )
                    ax.plot(
                        x,
                        y,
                        marker=str(marker),
                        linewidth=2.0,
                        linestyle=(
                            lineage_dash_cycle[lineage_idx % len(lineage_dash_cycle)]
                            if base_label == "tree root-only"
                            else "-"
                        ),
                        alpha=max(0.45, 1.0 - 0.12 * lineage_idx),
                        color=str(color),
                        label=(
                            f"{base_label}{label_suffix}"
                            if col_idx == 0 and row_idx == 0
                            else None
                        ),
                    )
            if not rendered_series:
                series_map = {
                    "tree root-only": {
                        "y": list(payload[root_only_key]),
                        "color": TREE_PRIMARY_COLOR,
                        "marker": "o",
                        "linestyle": "--",
                    },
                    "tree + leaf/internal": {
                        "y": list(payload[root_local_key]),
                        "color": TREE_LOCAL_COLOR,
                        "marker": "s",
                        "linestyle": "-",
                    },
                }
                for label, spec in series_map.items():
                    y = list(spec["y"])
                    if not any(math.isfinite(value) for value in y):
                        continue
                    ax.plot(
                        x,
                        y,
                        marker=str(spec["marker"]),
                        linewidth=2.0,
                        linestyle=str(spec.get("linestyle", "-")),
                        color=str(spec["color"]),
                        label=label if col_idx == 0 and row_idx == 0 else None,
                    )
            if row_idx == 0:
                ax.set_title(f"train_docs={int(_safe_int(payload['train_doc_count']))}")
            if col_idx == 0:
                ax.set_ylabel(title)
            ax.grid(True, alpha=0.25)
            if row_idx == len(metric_specs) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels, rotation=0)
                ax.set_xlabel("Root supervision share")
    scope_label = str(scope.get("scope_label", scope_key) or scope_key)
    _add_bottom_legend(fig, axes, fontsize=8, y_anchor=0.03, max_columns=4)
    fig.suptitle(f"{scope_label}: Tree-Only Diagnostics", y=0.995)
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_supervision_recovery_local_ablation_grid(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
    output_path: Path,
) -> bool:
    if str(recovery.get("status", "")) not in {"", "ready"}:
        return False
    scope = dict((recovery.get("scopes") or {}).get(str(scope_key)) or {})
    payloads = _local_ablation_payloads(recovery, scope_key=scope_key)
    if not payloads:
        return False
    fig, axes = plt.subplots(
        1,
        len(payloads),
        figsize=(max(11.0, 5.4 * len(payloads)), 4.8),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes.ravel())
    family_styles = {
        "tree": {"color": TREE_PRIMARY_COLOR, "marker": "o", "label": "tree"},
        "official_fno": {"color": FNO_OFFICIAL_COLOR, "marker": "^", "label": "official_fno"},
        "official_fno_sumlen": {
            "color": FNO_SUMLEN_COLOR,
            "marker": "D",
            "label": "official_fno_sumlen",
        },
    }
    best_full_root_by_docs = _best_full_root_root_mae_by_train_docs(
        recovery,
        scope_key=scope_key,
    )
    best_full_root_details = _best_full_root_ceiling_details_by_train_docs(
        recovery,
        scope_key=scope_key,
    )
    for idx, (payload, ax) in enumerate(zip(payloads, axes)):
        package_order = list(payload.get("package_order") or [])
        x = list(range(len(package_order)))
        series_map = {
            "tree": list(payload.get("tree_root_mae") or []),
            "official_fno": list(payload.get("official_fno_root_mae") or []),
            "official_fno_sumlen": list(payload.get("official_fno_sumlen_root_mae") or []),
        }
        for series_name, y in series_map.items():
            if not any(math.isfinite(value) for value in y):
                continue
            style = family_styles[series_name]
            ax.plot(
                x,
                y,
                marker=str(style["marker"]),
                linewidth=2.0,
                color=str(style["color"]),
                label=str(style["label"]) if idx == 0 else None,
            )
        _add_best_full_root_ceiling_line(
            ax,
            y_value=best_full_root_by_docs.get(int(_safe_int(payload["train_doc_count"]))),
            label="best full-root ceiling" if idx == 0 else None,
        )
        _add_full_root_ceiling_source_note(
            ax,
            detail=best_full_root_details.get(int(_safe_int(payload["train_doc_count"]))),
        )
        ax.set_title(f"train_docs={int(_safe_int(payload['train_doc_count']))}")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_package_tick_label(package_name) for package_name in package_order],
            rotation=35,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_xlabel("Local supervision progression at 10% root")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.set_ylabel("Test root MAE")
    scope_label = str(scope.get("scope_label", scope_key) or scope_key)
    fig.suptitle(f"{scope_label}: What Extra Tree Labels Help at R10?", y=1.02)
    _add_bottom_legend(fig, axes, fontsize=8, y_anchor=0.06, max_columns=4)
    _add_supervision_plot_caption(
        fig,
        lines=[
            _caption_root_example(
                [int(_safe_int(payload.get("train_doc_count"))) for payload in payloads],
                root_share=10,
            ),
            "The root budget stays fixed at `R10` and the x-axis changes only which extra tree labels are added.",
            "The FNO lines are flat `R10` baselines. The dotted benchmark line shows the best result with 100% root supervision at the same train-doc count.",
        ],
        top=0.92,
        bottom=0.18,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_supervision_recovery_dense_local_root_ladder(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
    output_path: Path,
) -> bool:
    if str(recovery.get("status", "")) not in {"", "ready"}:
        return False
    scope = dict((recovery.get("scopes") or {}).get(str(scope_key)) or {})
    payloads = _dense_local_root_ladder_payloads(recovery, scope_key=scope_key)
    if not payloads:
        return False
    fig, axes = plt.subplots(
        1,
        len(payloads),
        figsize=(max(11.0, 5.2 * len(payloads)), 4.6),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes.ravel())
    family_styles = {
        "tree_root_local": {
            "color": TREE_LOCAL_COLOR,
            "marker": "s",
            "label": "tree + leaf/internal",
        },
        "official_fno": {"color": FNO_OFFICIAL_COLOR, "marker": "^", "label": "official_fno"},
        "official_fno_sumlen": {
            "color": FNO_SUMLEN_COLOR,
            "marker": "D",
            "label": "official_fno_sumlen",
        },
    }
    best_full_root_by_docs = _best_full_root_root_mae_by_train_docs(
        recovery,
        scope_key=scope_key,
    )
    best_full_root_details = _best_full_root_ceiling_details_by_train_docs(
        recovery,
        scope_key=scope_key,
    )
    for idx, (payload, ax) in enumerate(zip(payloads, axes)):
        root_shares = list(payload.get("root_shares") or [])
        x = list(range(len(root_shares)))
        x_labels = [f"{share}%" for share in root_shares]
        local_series = list(payload.get("tree_root_local_series") or [])
        display_local_series = _select_primary_lineage_series(
            local_series,
            metric_key="tree_test_root_mae",
            max_keep=1,
        )
        if display_local_series:
            for lineage_idx, series in enumerate(display_local_series):
                y = list(series.get("tree_test_root_mae") or [])
                if not any(math.isfinite(value) for value in y):
                    continue
                label_suffix = (
                    f" · {series.get('lineage_label')}"
                    if len(display_local_series) > 1
                    else ""
                )
                ax.plot(
                    x,
                    y,
                    marker="s",
                    linewidth=2.0,
                    color=TREE_LOCAL_COLOR,
                    alpha=max(0.45, 1.0 - 0.12 * lineage_idx),
                    label=(
                        f"tree + leaf/internal{label_suffix}"
                        if idx == 0
                        else None
                    ),
                )
        else:
            y = list(payload.get("tree_root_local_root_mae") or [])
            if any(math.isfinite(value) for value in y):
                ax.plot(
                    x,
                    y,
                    marker="s",
                    linewidth=2.0,
                    color=TREE_LOCAL_COLOR,
                    label="tree + leaf/internal" if idx == 0 else None,
                )
        for series_name, y in {
            "official_fno": list(payload.get("official_fno_root_mae") or []),
            "official_fno_sumlen": list(payload.get("official_fno_sumlen_root_mae") or []),
        }.items():
            if not any(math.isfinite(value) for value in y):
                continue
            style = family_styles[series_name]
            ax.plot(
                x,
                y,
                marker=str(style["marker"]),
                linewidth=2.0,
                color=str(style["color"]),
                label=str(style["label"]) if idx == 0 else None,
            )
        _add_best_full_root_ceiling_line(
            ax,
            y_value=best_full_root_by_docs.get(int(_safe_int(payload["train_doc_count"]))),
            label="best full-root ceiling" if idx == 0 else None,
        )
        _add_full_root_ceiling_source_note(
            ax,
            detail=best_full_root_details.get(int(_safe_int(payload["train_doc_count"]))),
        )
        ax.set_title(f"train_docs={int(_safe_int(payload['train_doc_count']))}")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=0)
        ax.set_xlabel("Root supervision share")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.set_ylabel("Test root MAE")
    scope_label = str(scope.get("scope_label", scope_key) or scope_key)
    fig.suptitle(f"{scope_label}: Full Local Supervision + Root Sweep", y=1.02)
    _add_bottom_legend(fig, axes, fontsize=8, y_anchor=0.06, max_columns=4)
    _add_supervision_plot_caption(
        fig,
        lines=[
            "The tree keeps full local leaf/internal supervision and only the root supervision share changes.",
            _caption_root_example(
                [int(_safe_int(payload.get("train_doc_count"))) for payload in payloads],
                root_share=10,
            ),
            "The FNO lines are matched root-only baselines. The dotted benchmark line shows the best result with 100% root supervision at the same train-doc count.",
        ],
        top=0.92,
        bottom=0.18,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_supervision_recovery_tree_constant_density_root_ladder(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
    output_path: Path,
) -> bool:
    if str(recovery.get("status", "")) not in {"", "ready"}:
        return False
    scope = dict((recovery.get("scopes") or {}).get(str(scope_key)) or {})
    payloads = _tree_constant_density_root_ladder_payloads(
        recovery,
        scope_key=scope_key,
    )
    if not payloads:
        return False
    fig, axes = plt.subplots(
        1,
        len(payloads),
        figsize=(max(11.0, 5.2 * len(payloads)), 4.6),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes.ravel())
    best_full_root_by_docs = _best_full_root_root_mae_by_train_docs(
        recovery,
        scope_key=scope_key,
    )
    best_full_root_details = _best_full_root_ceiling_details_by_train_docs(
        recovery,
        scope_key=scope_key,
    )
    for idx, (payload, ax) in enumerate(zip(payloads, axes)):
        root_shares = list(payload.get("root_shares") or [])
        x = list(range(len(root_shares)))
        x_labels = [f"{share}%" for share in root_shares]
        lineage_dash_cycle = ["--", "-.", ":"]
        rendered_series = False
        for base_label, lineage_series, color, marker in (
            (
                "tree root-only",
                _select_primary_lineage_series(
                    list(payload.get("tree_root_only_series") or []),
                    metric_key="tree_test_root_mae",
                    max_keep=1,
                ),
                TREE_PRIMARY_COLOR,
                "o",
            ),
            (
                "tree + leaf/internal",
                _select_primary_lineage_series(
                    list(payload.get("tree_root_local_series") or []),
                    metric_key="tree_test_root_mae",
                    max_keep=1,
                ),
                TREE_LOCAL_COLOR,
                "s",
            ),
        ):
            use_suffix = len(lineage_series) > 1
            for lineage_idx, series in enumerate(lineage_series):
                y = list(series.get("tree_test_root_mae") or [])
                if not any(math.isfinite(value) for value in y):
                    continue
                rendered_series = True
                label_suffix = (
                    f" · {series.get('lineage_label')}"
                    if use_suffix
                    else ""
                )
                ax.plot(
                    x,
                    y,
                    marker=str(marker),
                    linewidth=2.0,
                    linestyle=(
                        lineage_dash_cycle[lineage_idx % len(lineage_dash_cycle)]
                        if base_label == "tree root-only"
                        else "-"
                    ),
                    alpha=max(0.45, 1.0 - 0.12 * lineage_idx),
                    color=str(color),
                    label=(
                        f"{base_label}{label_suffix}"
                        if idx == 0
                        else None
                    ),
                )
        if not rendered_series:
            series_map = {
                "tree root-only": {
                    "y": list(payload.get("tree_root_only_root_mae") or []),
                    "color": TREE_PRIMARY_COLOR,
                    "marker": "o",
                    "linestyle": "--",
                },
                "tree + leaf/internal": {
                    "y": list(payload.get("tree_root_local_root_mae") or []),
                    "color": TREE_LOCAL_COLOR,
                    "marker": "s",
                    "linestyle": "-",
                },
            }
            for label, spec in series_map.items():
                y = list(spec["y"])
                if not any(math.isfinite(value) for value in y):
                    continue
                ax.plot(
                    x,
                    y,
                    marker=str(spec["marker"]),
                    linewidth=2.0,
                    linestyle=str(spec["linestyle"]),
                    color=str(spec["color"]),
                    label=label if idx == 0 else None,
                )
        _add_best_full_root_ceiling_line(
            ax,
            y_value=best_full_root_by_docs.get(int(_safe_int(payload["train_doc_count"]))),
            label="best full-root ceiling" if idx == 0 else None,
        )
        _add_full_root_ceiling_source_note(
            ax,
            detail=best_full_root_details.get(int(_safe_int(payload["train_doc_count"]))),
        )
        ax.set_title(f"train_docs={int(_safe_int(payload['train_doc_count']))}")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=0)
        ax.set_xlabel("Root supervision share")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.set_ylabel("Test root MAE")
    scope_label = str(scope.get("scope_label", scope_key) or scope_key)
    fig.suptitle(f"{scope_label}: Tree Constant-Density Root Ladders", y=1.02)
    _add_bottom_legend(fig, axes, fontsize=8, y_anchor=0.06, max_columns=4)
    _add_supervision_plot_caption(
        fig,
        lines=[
            "Each panel fixes the number of training docs and sweeps the root-labeled share `R`.",
            "The dashed line is the root-only tree. The solid line keeps dense local leaf/internal labels on while the root share changes.",
            "This is the tree-only view of the constant-density ladders, without the repeated FNO comparator overlay.",
        ],
        top=0.9,
        bottom=0.18,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_supervision_recovery_local_law_rate_grid(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
    output_path: Path,
    root_share: int,
) -> bool:
    if str(recovery.get("status", "")) not in {"", "ready"}:
        return False
    scope = dict((recovery.get("scopes") or {}).get(str(scope_key)) or {})
    payloads = _local_law_rate_payloads(
        recovery,
        scope_key=scope_key,
        root_share=root_share,
    )
    if not payloads:
        return False
    fig, axes = plt.subplots(
        1,
        len(payloads),
        figsize=(max(11.0, 5.2 * len(payloads)), 4.6),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes.ravel())
    family_styles = {
        "tree": {"color": TREE_PRIMARY_COLOR, "marker": "o", "label": "tree"},
        "official_fno": {"color": FNO_OFFICIAL_COLOR, "marker": "^", "label": "official_fno"},
        "official_fno_sumlen": {
            "color": FNO_SUMLEN_COLOR,
            "marker": "D",
            "label": "official_fno_sumlen",
        },
    }
    best_full_root_by_docs = _best_full_root_root_mae_by_train_docs(
        recovery,
        scope_key=scope_key,
    )
    best_full_root_details = _best_full_root_ceiling_details_by_train_docs(
        recovery,
        scope_key=scope_key,
    )
    for idx, (payload, ax) in enumerate(zip(payloads, axes)):
        local_law_shares = list(payload.get("local_law_shares") or [])
        series_map = {
            "tree": list(payload.get("tree_root_mae") or []),
            "official_fno": list(payload.get("official_fno_root_mae") or []),
            "official_fno_sumlen": list(payload.get("official_fno_sumlen_root_mae") or []),
        }
        for series_name, y in series_map.items():
            if not any(math.isfinite(value) for value in y):
                continue
            style = family_styles[series_name]
            ax.plot(
                local_law_shares,
                y,
                marker=str(style["marker"]),
                linewidth=2.0,
                color=str(style["color"]),
                label=str(style["label"]) if idx == 0 else None,
            )
        _add_best_full_root_ceiling_line(
            ax,
            y_value=best_full_root_by_docs.get(int(_safe_int(payload["train_doc_count"]))),
            label="best full-root ceiling" if idx == 0 else None,
        )
        _add_full_root_ceiling_source_note(
            ax,
            detail=best_full_root_details.get(int(_safe_int(payload["train_doc_count"]))),
        )
        ax.set_title(f"train_docs={int(_safe_int(payload['train_doc_count']))}")
        ax.set_xticks(local_law_shares)
        ax.set_xticklabels([f"{share}%" for share in local_law_shares], rotation=0)
        ax.set_xlabel("Matched leaf/internal count supervision")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.set_ylabel("Test root MAE")
    scope_label = str(scope.get("scope_label", scope_key) or scope_key)
    fig.suptitle(f"{scope_label}: Extra Count Labels at R{int(root_share)}", y=1.02)
    _add_bottom_legend(fig, axes, fontsize=8, y_anchor=0.06, max_columns=4)
    _add_supervision_plot_caption(
        fig,
        lines=[
            _caption_root_example(
                [int(_safe_int(payload.get("train_doc_count"))) for payload in payloads],
                root_share=root_share,
            ),
            f"The root budget stays fixed at `R{int(root_share)}` and the x-axis changes only equal leaf/internal count supervision.",
            "The FNO lines are flat comparison baselines. The dotted benchmark line shows the best result with 100% root supervision at the same train-doc count.",
        ],
        top=0.92,
        bottom=0.18,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_supervision_recovery_r10_local_law_rate_grid(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
    output_path: Path,
) -> bool:
    return _plot_supervision_recovery_local_law_rate_grid(
        recovery,
        scope_key=scope_key,
        output_path=output_path,
        root_share=10,
    )


def _plot_supervision_recovery_mass_matched_rate_grid(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
    output_path: Path,
    root_share: int,
) -> bool:
    if str(recovery.get("status", "")) not in {"", "ready"}:
        return False
    scope = dict((recovery.get("scopes") or {}).get(str(scope_key)) or {})
    payloads = _mass_matched_law_rate_payloads(
        recovery,
        scope_key=scope_key,
        root_share=root_share,
    )
    if not payloads:
        return False
    fig, axes = plt.subplots(
        1,
        len(payloads),
        figsize=(max(11.0, 5.2 * len(payloads)), 4.6),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes.ravel())
    family_styles = {
        "tree": {"color": TREE_PRIMARY_COLOR, "marker": "o", "label": "tree"},
        "official_fno": {"color": FNO_OFFICIAL_COLOR, "marker": "^", "label": "official_fno"},
        "official_fno_sumlen": {
            "color": FNO_SUMLEN_COLOR,
            "marker": "D",
            "label": "official_fno_sumlen",
        },
    }
    best_full_root_by_docs = _best_full_root_root_mae_by_train_docs(
        recovery,
        scope_key=scope_key,
    )
    best_full_root_details = _best_full_root_ceiling_details_by_train_docs(
        recovery,
        scope_key=scope_key,
    )
    for idx, (payload, ax) in enumerate(zip(payloads, axes)):
        local_rate_percents = list(payload.get("local_rate_percents") or [])
        series_map = {
            "tree": list(payload.get("tree_root_mae") or []),
            "official_fno": list(payload.get("official_fno_root_mae") or []),
            "official_fno_sumlen": list(payload.get("official_fno_sumlen_root_mae") or []),
        }
        for series_name, y in series_map.items():
            if not any(math.isfinite(value) for value in y):
                continue
            style = family_styles[series_name]
            ax.plot(
                local_rate_percents,
                y,
                marker=str(style["marker"]),
                linewidth=2.0,
                color=str(style["color"]),
                label=str(style["label"]) if idx == 0 else None,
            )
        _add_best_full_root_ceiling_line(
            ax,
            y_value=best_full_root_by_docs.get(int(_safe_int(payload["train_doc_count"]))),
            label="best full-root ceiling" if idx == 0 else None,
        )
        _add_full_root_ceiling_source_note(
            ax,
            detail=best_full_root_details.get(int(_safe_int(payload["train_doc_count"]))),
        )
        target_masses = [
            _safe_float(value, float("nan"))
            for value in list(payload.get("tree_target_mass_per_doc") or [])
        ]
        realized_masses = [
            _safe_float(value, float("nan"))
            for value in list(payload.get("tree_realized_effective_mass_per_doc") or [])
        ]
        finite_target = [value for value in target_masses if math.isfinite(value)]
        target_text = (
            f"{finite_target[0]:.3f}" if finite_target and len(set(round(v, 6) for v in finite_target)) == 1 else "varies"
        )
        realized_max = max(
            [value for value in realized_masses if math.isfinite(value)],
            default=float("nan"),
        )
        title = f"train_docs={int(_safe_int(payload['train_doc_count']))}"
        if math.isfinite(realized_max):
            title += f"\n target={target_text}, max realized={realized_max:.3f}"
        ax.set_title(title)
        ax.set_xticks(local_rate_percents)
        ax.set_xticklabels([f"{share:.1f}%" for share in local_rate_percents], rotation=0)
        ax.set_xlabel("Equal leaf/internal count rate")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.set_ylabel("Test root MAE")
            ax.legend(frameon=False, fontsize=8)
    scope_label = str(scope.get("scope_label", scope_key) or scope_key)
    fig.suptitle(f"{scope_label}: Fixed-Budget Tree vs FNO at R{int(root_share)}", y=1.02)
    _add_supervision_plot_caption(
        fig,
        lines=[
            f"`R{int(root_share)}` keeps total training-doc equivalents fixed and shifts budget from root labels into equal leaf/internal count labels for the tree.",
            "Panel subtitles report the requested target mass and the maximum realized effective mass observed across the tree ladder points.",
            "The FNO lines are flat comparison baselines with no literal local-rate meaning away from `0%`. The dotted benchmark line shows the best result with 100% root supervision at the same train-doc count.",
        ],
        top=0.9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_supervision_recovery_mass_matched_overlay(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
    output_path: Path,
) -> bool:
    if str(recovery.get("status", "")) not in {"", "ready"}:
        return False
    scope = dict((recovery.get("scopes") or {}).get(str(scope_key)) or {})
    if not scope:
        return False
    payloads_by_root_share: Dict[int, Dict[int, Dict[str, Any]]] = {}
    available_root_shares: List[int] = []
    train_doc_counts: set[int] = set()
    for root_share in sorted(SUPERVISION_RECOVERY_MASS_MATCHED_RATE_PACKAGE_ORDERS):
        payloads = _mass_matched_law_rate_payloads(
            recovery,
            scope_key=scope_key,
            root_share=root_share,
        )
        if not payloads:
            continue
        available_root_shares.append(int(root_share))
        payloads_by_root_share[int(root_share)] = {
            int(_safe_int(payload.get("train_doc_count"))): dict(payload)
            for payload in payloads
        }
        train_doc_counts.update(
            int(_safe_int(payload.get("train_doc_count"))) for payload in payloads
        )
    ordered_train_doc_counts = sorted(count for count in train_doc_counts if count > 0)
    if not ordered_train_doc_counts:
        return False
    fig, axes = plt.subplots(
        1,
        len(ordered_train_doc_counts),
        figsize=(max(11.0, 5.4 * len(ordered_train_doc_counts)), 4.9),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes.ravel())
    family_styles = {
        "tree": {
            "color": MASS_MATCHED_OVERLAY_TREE_COLOR,
            "marker": "o",
            "label": "tree",
        },
        "official_fno": {
            "color": MASS_MATCHED_OVERLAY_OFFICIAL_FNO_COLOR,
            "marker": "^",
            "label": "official_fno",
        },
        "official_fno_sumlen": {
            "color": MASS_MATCHED_OVERLAY_FNO_SUMLEN_COLOR,
            "marker": "D",
            "label": "official_fno_sumlen",
        },
    }
    for idx, (train_doc_count, ax) in enumerate(zip(ordered_train_doc_counts, axes)):
        tick_values: set[float] = set()
        for root_share in available_root_shares:
            payload = dict(payloads_by_root_share.get(root_share, {}).get(train_doc_count) or {})
            if not payload:
                continue
            x = [
                _safe_float(value, float("nan"))
                for value in list(payload.get("local_rate_percents") or [])
            ]
            tick_values.update(value for value in x if math.isfinite(value))
            series_map = {
                "tree": list(payload.get("tree_root_mae") or []),
                "official_fno": list(payload.get("official_fno_root_mae") or []),
                "official_fno_sumlen": list(payload.get("official_fno_sumlen_root_mae") or []),
            }
            for series_name, y in series_map.items():
                if not any(math.isfinite(_safe_float(value, float("nan"))) for value in y):
                    continue
                ax.plot(
                    x,
                    y,
                    color=str(family_styles[series_name]["color"]),
                    marker=str(family_styles[series_name]["marker"]),
                    linestyle=MASS_MATCHED_OVERLAY_LINESTYLES.get(int(root_share), "-."),
                    linewidth=2.0,
                    markersize=5.5,
                )
        if tick_values:
            ordered_ticks = sorted(tick_values)
            ax.set_xticks(ordered_ticks)
            ax.set_xticklabels([f"{value:g}%" for value in ordered_ticks])
        ax.set_xlabel("Equal leaf/internal count rate (tree axis)")
        ax.set_title(f"train_docs={int(train_doc_count)}")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.set_ylabel("Test root MAE")
    family_handles = [
        Line2D(
            [0],
            [0],
            color=str(style["color"]),
            marker=str(style["marker"]),
            linewidth=2.2,
            linestyle="-",
            label=str(style["label"]),
        )
        for style in family_styles.values()
    ]
    root_handles = [
        Line2D(
            [0],
            [0],
            color=NEUTRAL_COLOR,
            linewidth=2.2,
            linestyle=MASS_MATCHED_OVERLAY_LINESTYLES.get(int(root_share), "-."),
            label=f"R{int(root_share)} total budget",
        )
        for root_share in available_root_shares
    ]
    family_legend = fig.legend(
        handles=family_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.11),
        ncol=max(1, len(family_handles)),
        frameon=False,
        fontsize=8,
    )
    fig.add_artist(family_legend)
    fig.legend(
        handles=root_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=max(1, len(root_handles)),
        frameon=False,
        fontsize=8,
    )
    scope_label = str(scope.get("scope_label", scope_key) or scope_key)
    fig.suptitle(f"{scope_label}: Fixed-Budget Comparison Across Budgets", y=1.02)
    available_text = ", ".join(f"`R{int(root_share)}`" for root_share in available_root_shares)
    budget_text = _mass_matched_budget_legend_text(available_root_shares)
    _add_supervision_plot_caption(
        fig,
        lines=[
            f"Each ladder keeps total supervision fixed: {budget_text}.",
            "On the x-axis, `0%` is the root-only anchor for that ladder. Moving right shifts some of that fixed budget into equal leaf/internal count labels for the tree.",
            "The `official_fno` and `official_fno_sumlen` lines are flat comparison baselines repeated across the x-axis. They do not have a literal local-rate interpretation.",
            f"Currently available ladders for this scope: {available_text}.",
        ],
        top=0.9,
        bottom=0.19,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_supervision_recovery_leaf_geometry_grid(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
    output_path: Path,
    root_share: int,
) -> bool:
    if str(recovery.get("status", "")) not in {"", "ready"}:
        return False
    scope = dict((recovery.get("scopes") or {}).get(str(scope_key)) or {})
    if not scope:
        return False
    payloads = _leaf_geometry_payloads(
        recovery,
        scope_key=scope_key,
        root_share=root_share,
    )
    if not payloads:
        return False
    fig, axes = plt.subplots(
        1,
        len(payloads),
        figsize=(max(11.0, 5.4 * len(payloads)), 4.8),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes.ravel())
    for idx, (payload, ax) in enumerate(zip(payloads, axes)):
        rows = [dict(row) for row in list(payload.get("rows") or [])]
        x = [_effective_leaves_per_doc(row) for row in rows]
        tree_y = [_tree_root_mae_from_family_row(row) for row in rows]
        if not any(math.isfinite(value) for value in tree_y):
            continue
        ax.plot(
            x,
            tree_y,
            color=TREE_PRIMARY_COLOR,
            marker="o",
            linewidth=2.2,
            label="tree" if idx == 0 else None,
        )
        official_fno = _safe_float(payload.get("official_fno_root_mae"), float("nan"))
        if math.isfinite(official_fno):
            ax.axhline(
                official_fno,
                color=FNO_OFFICIAL_COLOR,
                linestyle="--",
                linewidth=2.0,
                label="official_fno" if idx == 0 else None,
            )
        official_fno_sumlen = _safe_float(
            payload.get("official_fno_sumlen_root_mae"),
            float("nan"),
        )
        if math.isfinite(official_fno_sumlen):
            ax.axhline(
                official_fno_sumlen,
                color=FNO_SUMLEN_COLOR,
                linestyle=":",
                linewidth=2.0,
                label="official_fno_sumlen" if idx == 0 else None,
            )
        tick_labels = [
            f"{_effective_leaves_per_doc(row)}\n({_effective_fixed_leaf_tokens(row)})"
            for row in rows
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Leaves/doc\n(fixed_leaf_tokens)")
        ax.set_title(f"train_docs={int(_safe_int(payload.get('train_doc_count'), 0))}")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.set_ylabel("Test root MAE")
            ax.legend(frameon=False, fontsize=8)
        first_row = dict(rows[0]) if rows else {}
        if _is_exact_full_doc_parity_row(first_row):
            ax.annotate(
                "FNO-equivalent",
                xy=(x[0], tree_y[0]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=NEUTRAL_COLOR,
            )
    scope_label = str(scope.get("scope_label", scope_key) or scope_key)
    fig.suptitle(
        f"{scope_label}: Leaves/Doc at R{int(root_share)}",
        y=1.02,
    )
    _add_supervision_plot_caption(
        fig,
        lines=[
            f"X-axis: leaves per document. The tick label shows `leaves/doc` and then `fixed_leaf_tokens` in parentheses.",
            f"The leftmost point is the `full{int(root_share)}` root-only anchor at `1 leaf/doc`. It is annotated as FNO-equivalent only when the executed row is marked `parity_mode=exact_full_doc`.",
            f"Moving right keeps the same `R{int(root_share)}` budget family but makes the tree solve a deeper composition problem.",
            "The red and orange lines are flat FNO baselines at the same train-doc count.",
        ],
        top=0.9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _format_doc_count(value: Any) -> str:
    count = int(_safe_int(value))
    if count >= 10000:
        if count % 1000 == 0:
            return f"{count // 1000}k"
        return f"{count / 1000.0:.2f}k"
    return str(count)


def _is_solved_floor(value: Any, *, tol: float = 1e-9) -> bool:
    number = _safe_float(value, float("nan"))
    return math.isfinite(number) and abs(number) <= float(tol)


def _format_metric_or_floor(value: Any, *, tol: float = 1e-9) -> str:
    if _is_solved_floor(value, tol=tol):
        return "solved floor"
    metric = _safe_float(value, float("nan"))
    if math.isfinite(metric):
        return f"{metric:.6g}"
    return "n/a"


def _format_pct_or_floor(value: Any, *, note: str = "solved floor") -> str:
    metric = _safe_float(value, float("nan"))
    if math.isfinite(metric):
        return f"{metric:.1f}%"
    return note


def _format_unavailable(value: Any, *, fmt: str = ".6f") -> str:
    metric = _safe_float(value, float("nan"))
    if math.isfinite(metric):
        return format(metric, fmt)
    return "unavailable"


def _monotone_lower_envelope(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(
        (
            dict(row)
            for row in rows
            if int(_safe_int(row.get("train_doc_count", row.get("train_docs")))) > 0
            and math.isfinite(
                _safe_float(
                    row.get("test_root_mae_mean", row.get("best_full_doc_fno_test_root_mae")),
                    float("nan"),
                )
            )
        ),
        key=lambda row: int(_safe_int(row.get("train_doc_count", row.get("train_docs")))),
    )
    envelope: List[Dict[str, Any]] = []
    best_so_far = float("inf")
    for row in ordered:
        docs = int(_safe_int(row.get("train_doc_count", row.get("train_docs"))))
        mae = _safe_float(
            row.get("test_root_mae_mean", row.get("best_full_doc_fno_test_root_mae")),
            float("nan"),
        )
        best_so_far = min(best_so_far, mae)
        envelope.append({"train_doc_count": docs, "test_root_mae_mean": float(best_so_far), **dict(row)})
    return envelope


def _interpolate_equivalent_docs(
    curve_rows: Sequence[Mapping[str, Any]],
    *,
    target_mae: float,
) -> Dict[str, Any]:
    rows = _monotone_lower_envelope(curve_rows)
    if not rows:
        return {
            "equivalent_train_docs": float("nan"),
            "relation": "unavailable",
            "anchor_saturated": False,
        }
    maes = [_safe_float(row.get("test_root_mae_mean"), float("nan")) for row in rows]
    docs = [int(_safe_int(row.get("train_doc_count"))) for row in rows]
    anchor_saturated = all(_is_solved_floor(mae) for mae in maes)
    if anchor_saturated:
        return {
            "equivalent_train_docs": float("nan"),
            "relation": "anchor_saturated",
            "anchor_saturated": True,
            "min_train_docs": int(min(docs)),
            "max_train_docs": int(max(docs)),
        }
    min_docs = int(min(docs))
    max_docs = int(max(docs))
    worst_mae = float(maes[0])
    best_mae = float(maes[-1])
    if not math.isfinite(target_mae):
        relation = "unavailable"
        value = float("nan")
    elif target_mae < best_mae:
        relation = "gt_max"
        value = float(max_docs)
    elif target_mae > worst_mae:
        relation = "lt_min"
        value = float(min_docs)
    else:
        relation = "interpolated"
        value = float(min_docs)
        for idx in range(len(rows) - 1):
            left_docs = float(int(_safe_int(rows[idx].get("train_doc_count"))))
            right_docs = float(int(_safe_int(rows[idx + 1].get("train_doc_count"))))
            left_mae = float(_safe_float(rows[idx].get("test_root_mae_mean")))
            right_mae = float(_safe_float(rows[idx + 1].get("test_root_mae_mean")))
            if target_mae > left_mae + 1e-12 or target_mae < right_mae - 1e-12:
                continue
            if abs(left_mae - right_mae) <= 1e-12:
                value = float(left_docs)
            else:
                left_x = math.log2(max(left_docs, 1.0))
                right_x = math.log2(max(right_docs, 1.0))
                ratio = float((target_mae - left_mae) / (right_mae - left_mae))
                value = float(2.0 ** (left_x + ratio * (right_x - left_x)))
            break
    return {
        "equivalent_train_docs": float(value),
        "relation": relation,
        "anchor_saturated": False,
        "min_train_docs": min_docs,
        "max_train_docs": max_docs,
    }


def _format_equivalent_docs(entry: Mapping[str, Any], *, key: str) -> str:
    relation = str(entry.get(f"{key}_relation", entry.get("relation", "")) or "")
    value = _safe_float(entry.get(key), float("nan"))
    if relation == "anchor_saturated":
        return "anchor already solved"
    if relation == "gt_max":
        return f"> {_format_doc_count(int(_safe_int(entry.get(f'{key}_max_train_docs', entry.get('max_train_docs')))))}"
    if relation == "lt_min":
        return f"< {_format_doc_count(int(_safe_int(entry.get(f'{key}_min_train_docs', entry.get('min_train_docs')))))}"
    if math.isfinite(value):
        return _format_doc_count(int(round(value)))
    return "n/a"


def _preferred_recoverable_scope_key(
    scope_summaries: Mapping[str, Mapping[str, Any]],
) -> str:
    for key in SUPERVISION_RECOVERY_RECOVERABLE_SCOPE_FAMILY:
        if key in scope_summaries:
            return key
    recoverable_candidates = [
        str(key)
        for key in scope_summaries
        if str(key).startswith("recoverable_")
    ]
    if recoverable_candidates:
        return sorted(recoverable_candidates)[0]
    return SUPERVISION_RECOVERY_RECOVERABLE_BENCHMARK


def _recoverable_scope_label(scope_key: str) -> str:
    return str(scope_key or SUPERVISION_RECOVERY_RECOVERABLE_BENCHMARK)


def _structural_scope_label(scope_key: str, *, grid_name: str = "") -> str:
    grid = str(grid_name or SUPERVISION_RECOVERY_STRUCTURAL_GRID).strip()
    return f"{grid}::{str(scope_key).strip()}"


def _select_primary_structural_scope(scope_summaries: Mapping[str, Mapping[str, Any]]) -> str:
    recoverable_scope_key = _preferred_recoverable_scope_key(scope_summaries)
    candidates = []
    for key, value in scope_summaries.items():
        if key == recoverable_scope_key:
            continue
        best_fno = _safe_float(value.get("best_fno_anchor_mae"), float("nan"))
        if math.isfinite(best_fno):
            candidates.append((best_fno, key))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return recoverable_scope_key


def _rank_rows(rows: Sequence[Mapping[str, Any]], *, higher_is_better: str, lower_is_better: str) -> List[Dict[str, Any]]:
    if not rows:
        return []
    by_high = sorted(rows, key=lambda row: _safe_float(row.get(higher_is_better, float("-inf"))), reverse=True)
    by_low = sorted(rows, key=lambda row: _safe_float(row.get(lower_is_better, float("inf"))))
    high_rank = {int(_safe_int(row.get("batch_size"))): idx for idx, row in enumerate(by_high)}
    low_rank = {int(_safe_int(row.get("batch_size"))): idx for idx, row in enumerate(by_low)}
    ranked: List[Dict[str, Any]] = []
    for row in rows:
        batch_size = int(_safe_int(row.get("batch_size")))
        merged = dict(row)
        merged["speed_rank"] = int(high_rank.get(batch_size, len(rows)))
        merged["quality_rank"] = int(low_rank.get(batch_size, len(rows)))
        merged["combined_rank"] = int(merged["speed_rank"] + merged["quality_rank"])
        ranked.append(merged)
    ranked.sort(key=lambda row: (int(row["combined_rank"]), int(row["batch_size"])))
    return ranked


def _summarize_batch_timing(payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [dict(row) for row in list(payload.get("summary", []) or [])]
    rows.sort(key=lambda row: int(_safe_int(row.get("batch_size"))))
    if not rows:
        return {
            "rows": [],
            "runtime_efficiency": dict(payload.get("runtime_efficiency") or {}),
        }
    best_wall = max(rows, key=lambda row: _safe_float(row.get("docs_per_s_wall")))
    best_train = max(rows, key=lambda row: _safe_float(row.get("docs_per_s_train_loop")))
    fastest_exact = min(rows, key=lambda row: _safe_float(row.get("exact_metric_eval_s", float("inf"))))
    return {
        "rows": rows,
        "best_wall_batch": int(_safe_int(best_wall.get("batch_size"))),
        "best_wall_docs_per_s": _safe_float(best_wall.get("docs_per_s_wall")),
        "best_train_batch": int(_safe_int(best_train.get("batch_size"))),
        "best_train_docs_per_s": _safe_float(best_train.get("docs_per_s_train_loop")),
        "lowest_exact_eval_batch": int(_safe_int(fastest_exact.get("batch_size"))),
        "lowest_exact_eval_s": _safe_float(fastest_exact.get("exact_metric_eval_s")),
        "runtime_efficiency": dict(payload.get("runtime_efficiency") or {}),
    }


def _summarize_runtime_efficiency(
    *,
    batch_timing: Mapping[str, Any],
    batch_quality: Mapping[str, Any],
    docs_epochs: Mapping[str, Any],
) -> Dict[str, Any]:
    runtime = dict(batch_timing.get("runtime_efficiency") or {})
    by_batch_rows = [
        dict(row)
        for row in list((batch_quality.get("rows") or []))
        if str(row.get("runtime_data_mode", "")).strip()
    ]
    docs_rows = [
        dict(row)
        for row in list((docs_epochs.get("rows") or []))
        if str(row.get("runtime_data_mode", "")).strip()
    ]
    best_batch_row = None
    best_batch_size = int(_safe_int(batch_quality.get("best_balanced_batch")))
    for row in by_batch_rows:
        if int(_safe_int(row.get("batch_size"))) == best_batch_size:
            best_batch_row = row
            break
    fastest_docs_row = (
        max(
            docs_rows,
            key=lambda row: _safe_float(row.get("docs_per_s_wall_effective"), float("-inf")),
        )
        if docs_rows
        else None
    )
    runtime_present = bool(
        str(runtime.get("runtime_data_mode", "")).strip()
        or str(runtime.get("runtime_bucket_mode", "")).strip()
        or any(str(row.get("runtime_data_mode", "")).strip() for row in by_batch_rows + docs_rows)
    )
    if not runtime_present:
        return {
            "status": "unavailable",
            "reason": "runtime fields are missing from the selected throughput summaries",
        }
    return {
        "status": "ready",
        "runtime_data_mode": str(runtime.get("runtime_data_mode", "")),
        "runtime_bucket_mode": str(runtime.get("runtime_bucket_mode", "")),
        "runtime_workers_per_mig_mean": _safe_float(
            runtime.get("runtime_workers_per_mig_mean"),
            0.0,
        ),
        "resident_store_build_time_s_mean": _safe_float(
            runtime.get("resident_store_build_time_s_mean"),
            0.0,
        ),
        "steady_state_h2d_bytes_mean": _safe_float(
            runtime.get("steady_state_h2d_bytes_mean"),
            0.0,
        ),
        "steady_state_h2d_time_s_mean": _safe_float(
            runtime.get("steady_state_h2d_time_s_mean"),
            0.0,
        ),
        "resident_store_hits_total": _safe_int(
            runtime.get("resident_store_hits_total"),
            0,
        ),
        "resident_store_misses_total": _safe_int(
            runtime.get("resident_store_misses_total"),
            0,
        ),
        "cpu_fallback_reason_counts": dict(
            runtime.get("cpu_fallback_reason_counts") or {}
        ),
        "best_balanced_batch_resident_store_build_time_s": _safe_float(
            (best_batch_row or {}).get("resident_store_build_time_s"),
            float("nan"),
        ),
        "best_balanced_batch_steady_state_h2d_bytes": _safe_float(
            (best_batch_row or {}).get("steady_state_h2d_bytes"),
            float("nan"),
        ),
        "fastest_docs_epochs_runtime_data_mode": str(
            (fastest_docs_row or {}).get("runtime_data_mode", "")
        ),
        "fastest_docs_epochs_runtime_bucket_mode": str(
            (fastest_docs_row or {}).get("runtime_bucket_mode", "")
        ),
        "fastest_docs_epochs_steady_state_h2d_bytes": _safe_float(
            (fastest_docs_row or {}).get("steady_state_h2d_bytes"),
            float("nan"),
        ),
    }


def _summarize_medium_grid(payload: Mapping[str, Any]) -> Dict[str, Any]:
    by_batch = payload.get("by_batch_size", {}) or {}
    rows: List[Dict[str, Any]] = []
    for key, value in by_batch.items():
        row = dict(value)
        row["batch_size"] = int(_safe_int(key))
        rows.append(row)
    rows.sort(key=lambda row: int(row["batch_size"]))
    if not rows:
        return {"rows": [], "status": "missing", "reason": "no batch rows were available"}
    train_docs = int(_safe_int(payload.get("train_docs")))
    epochs = int(_safe_int(payload.get("epochs")))
    if train_docs <= 0 or epochs <= 0:
        train_candidates: set[int] = set()
        epoch_candidates: set[int] = set()
        runs_payload = payload.get("runs") or {}
        if isinstance(runs_payload, Mapping):
            run_iterable = list(runs_payload.values())
        else:
            run_iterable = list(runs_payload) if isinstance(runs_payload, list) else []
        for run in run_iterable:
            run_map = dict(run or {}) if isinstance(run, Mapping) else {}
            config = dict(run_map.get("config") or {})
            train_value = int(_safe_int(run_map.get("train_docs", config.get("train_docs"))))
            epoch_value = int(
                _safe_int(
                    run_map.get(
                        "epochs",
                        run_map.get(
                            "epochs_completed",
                            config.get("epochs", config.get("n_epochs")),
                        ),
                    )
                )
            )
            if train_value > 0:
                train_candidates.add(train_value)
            if epoch_value > 0:
                epoch_candidates.add(epoch_value)
        if train_docs <= 0 and len(train_candidates) == 1:
            train_docs = next(iter(train_candidates))
        if epochs <= 0 and len(epoch_candidates) == 1:
            epochs = next(iter(epoch_candidates))
    if train_docs <= 0 or epochs <= 0:
        return {
            "rows": rows,
            "status": "incompatible",
            "reason": "medium-grid summary does not expose a single train_docs/epochs setting",
        }
    best_quality = min(rows, key=lambda row: _safe_float(row.get("mean_best_val_mae", float("inf"))))
    best_speed = max(rows, key=lambda row: _safe_float(row.get("mean_docs_per_s_wall_effective")))
    ranked = _rank_rows(
        rows,
        higher_is_better="mean_docs_per_s_wall_effective",
        lower_is_better="mean_best_val_mae",
    )
    best_balanced = ranked[0]
    return {
        "rows": rows,
        "status": "ready",
        "train_docs": int(train_docs),
        "epochs": int(epochs),
        "best_quality_batch": int(best_quality["batch_size"]),
        "best_quality_val_mae": _safe_float(best_quality.get("mean_best_val_mae")),
        "best_speed_batch": int(best_speed["batch_size"]),
        "best_speed_docs_per_s": _safe_float(best_speed.get("mean_docs_per_s_wall_effective")),
        "best_balanced_batch": int(best_balanced["batch_size"]),
        "best_balanced_val_mae": _safe_float(best_balanced.get("mean_best_val_mae")),
        "best_balanced_docs_per_s": _safe_float(best_balanced.get("mean_docs_per_s_wall_effective")),
    }


def _summarize_docs_epochs(payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [dict(row) for row in list(payload.get("rows", []) or [])]
    rows.sort(key=lambda row: (int(_safe_int(row.get("train_docs"))), int(_safe_int(row.get("epochs")))))
    by_train_docs: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(_safe_int(row.get("train_docs")))].append(row)
    for train_docs, sub in sorted(grouped.items()):
        best = min(sub, key=lambda row: _safe_float(row.get("best_val_mae", float("inf"))))
        fastest = max(sub, key=lambda row: _safe_float(row.get("docs_per_s_wall_effective")))
        by_train_docs[str(train_docs)] = {
            "rows": [dict(item) for item in sorted(sub, key=lambda row: int(_safe_int(row.get("epochs"))))],
            "best_val_run": str(best.get("run", "")),
            "best_val_mae": _safe_float(best.get("best_val_mae")),
            "best_val_epochs": int(_safe_int(best.get("epochs"))),
            "fastest_run": str(fastest.get("run", "")),
            "fastest_docs_per_s": _safe_float(fastest.get("docs_per_s_wall_effective")),
            "fastest_epochs": int(_safe_int(fastest.get("epochs"))),
        }
    return {"rows": rows, "by_train_docs": by_train_docs}


def _summarize_learnability(payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [dict(row) for row in list(payload.get("aggregated_rows", []) or [])]
    rows.sort(
        key=lambda row: (
            int(_safe_int(row.get("train_docs"))),
            float(_safe_float(row.get("local_law_weight"))),
            float(_safe_float(row.get("audit_fraction"))),
        )
    )
    grouped: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(_safe_int(row.get("train_docs")))].append(row)
    by_train_docs: Dict[str, Dict[str, Any]] = {}
    for train_docs, sub in sorted(grouped.items()):
        baseline_rows = [row for row in sub if math.isclose(_safe_float(row.get("local_law_weight")), 0.0)]
        best_root = min(sub, key=lambda row: _safe_float(row.get("learned_root_mae_n", float("inf"))))
        best_root_baseline = (
            min(
                baseline_rows,
                key=lambda row: _safe_float(row.get("learned_root_mae_n", float("inf"))),
            )
            if baseline_rows
            else None
        )
        best_objective = min(sub, key=lambda row: _safe_float(row.get("heldout_objective_for_report", float("inf"))))
        best_baseline = (
            min(baseline_rows, key=lambda row: _safe_float(row.get("heldout_objective_for_report", float("inf"))))
            if baseline_rows
            else None
        )
        objective_gain_pct = float("nan")
        root_gain_pct = float("nan")
        if best_root_baseline is not None:
            base_root = _safe_float(best_root_baseline.get("learned_root_mae_n"))
            best_root_val = _safe_float(best_root.get("learned_root_mae_n"))
            if math.isfinite(base_root) and abs(base_root) > 1e-12:
                root_gain_pct = 100.0 * (base_root - best_root_val) / base_root
        if best_baseline is not None:
            base_obj = _safe_float(best_baseline.get("heldout_objective_for_report"))
            best_obj = _safe_float(best_objective.get("heldout_objective_for_report"))
            if math.isfinite(base_obj) and abs(base_obj) > 1e-12:
                objective_gain_pct = 100.0 * (base_obj - best_obj) / base_obj
        by_train_docs[str(train_docs)] = {
            "primary_metric_name": "learned_root_mae_n",
            "diagnostic_metric_name": "heldout_objective_for_report",
            "best_root_baseline": dict(best_root_baseline) if best_root_baseline is not None else None,
            "best_baseline": dict(best_baseline) if best_baseline is not None else None,
            "best_by_root": dict(best_root),
            "best_by_objective": dict(best_objective),
            "root_gain_pct_vs_baseline": float(root_gain_pct),
            "objective_gain_pct_vs_baseline": float(objective_gain_pct),
        }
    return {"rows": rows, "by_train_docs": by_train_docs}


def _summarize_weight_ablation(payload: Mapping[str, Any]) -> Dict[str, Any]:
    matched = [dict(row) for row in list(payload.get("matched_summaries", []) or [])]
    profiles = [dict(row) for row in list(payload.get("profile_summaries", []) or [])]
    matched.sort(key=lambda row: _safe_float(row.get("mean_gain_pct")), reverse=True)
    profiles.sort(key=lambda row: _safe_float(row.get("mean_root_error", float("inf"))))
    return {
        "matched_summaries": matched,
        "profile_summaries": profiles,
        "best_profile_by_gain": dict(matched[0]) if matched else None,
        "best_profile_by_root_error": dict(profiles[0]) if profiles else None,
        "root_surface_saturated": bool(profiles) and all(
            _is_solved_floor(row.get("mean_root_error")) for row in profiles
        ),
    }


def _summarize_law_packages(
    payload: Mapping[str, Any],
    *,
    same_run_doc_fno: bool = True,
) -> Dict[str, Any]:
    rows = []
    for name, row in payload.items():
        if not isinstance(row, Mapping):
            continue
        out = dict(row)
        out["name"] = str(name)
        tree_root = _safe_float(out.get("test_root_mae"), float("nan"))
        doc_fno_root = _safe_float(out.get("doc_fno_test_root_mae"), float("nan"))
        if not math.isfinite(_safe_float(out.get("tree_vs_doc_fno_root_mae_gap"), float("nan"))):
            out["tree_vs_doc_fno_root_mae_gap"] = (
                float(tree_root - doc_fno_root)
                if math.isfinite(tree_root) and math.isfinite(doc_fno_root)
                else float("nan")
            )
        rows.append(out)
    rows.sort(key=lambda row: _safe_float(row.get("test_root_mae", float("inf"))))
    doc_fno_rows = [
        dict(row)
        for row in rows
        if math.isfinite(_safe_float(row.get("doc_fno_test_root_mae"), float("nan")))
    ]
    best_doc_fno = (
        min(doc_fno_rows, key=lambda row: _safe_float(row.get("doc_fno_test_root_mae"), float("inf")))
        if doc_fno_rows
        else None
    )
    return {
        "rows": rows,
        "status": "ready",
        "best_package": dict(rows[0]) if rows else None,
        "best_doc_fno_reference": dict(best_doc_fno) if best_doc_fno is not None else None,
        "same_run_doc_fno": bool(same_run_doc_fno),
        "doc_fno_label": "same-run doc FNO" if same_run_doc_fno else "staged doc FNO reference",
        "root_surface_saturated": bool(rows) and all(
            _is_solved_floor(row.get("test_root_mae")) and _is_solved_floor(row.get("doc_fno_test_root_mae"))
            for row in rows
        ),
    }


def _best_tree_reference_by_train_docs(
    docs_epochs: Mapping[str, Any],
    batch_quality: Mapping[str, Any],
) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for key, row in sorted((docs_epochs.get("by_train_docs") or {}).items(), key=lambda item: int(item[0])):
        train_docs = int(_safe_int(key))
        out[train_docs] = {
            "best_tree_test_root_mae": float(row.get("best_val_mae", float("nan"))),
            "best_tree_source": str(row.get("best_val_run", "")),
            "best_tree_epochs": int(_safe_int(row.get("best_val_epochs"))),
        }
    train_docs = int(_safe_int(batch_quality.get("train_docs")))
    if train_docs > 0:
        candidate = {
            "best_tree_test_root_mae": float(batch_quality.get("best_quality_val_mae", float("nan"))),
            "best_tree_source": f"medium_grid_bs{int(_safe_int(batch_quality.get('best_quality_batch')))}",
            "best_tree_epochs": int(_safe_int(batch_quality.get("epochs"))),
        }
        incumbent = out.get(train_docs)
        incumbent_mae = _safe_float((incumbent or {}).get("best_tree_test_root_mae"), float("inf"))
        candidate_mae = _safe_float(candidate.get("best_tree_test_root_mae"), float("inf"))
        if candidate_mae < incumbent_mae:
            out[train_docs] = candidate
    return out


def _build_budget_review_curve(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    points = []
    for row in rows:
        reviewed_docs = _safe_float(row.get("reviewed_full_doc_equivalent_docs"), float("nan"))
        mae = _safe_float(row.get("test_root_mae_mean"), float("nan"))
        if reviewed_docs > 0.0 and math.isfinite(mae):
            points.append({"reviewed_docs": reviewed_docs, "test_root_mae_mean": mae})
    points.sort(key=lambda row: float(row["reviewed_docs"]))
    out: List[Dict[str, Any]] = []
    best_so_far = float("inf")
    for point in points:
        best_so_far = min(best_so_far, float(point["test_root_mae_mean"]))
        out.append(
            {
                "reviewed_docs": float(point["reviewed_docs"]),
                "test_root_mae_mean": float(best_so_far),
            }
        )
    return out


def _interpolate_reviewed_docs(
    curve_rows: Sequence[Mapping[str, Any]],
    *,
    target_mae: float,
) -> Dict[str, Any]:
    rows = _build_budget_review_curve(curve_rows)
    if not rows:
        return {"reviewed_docs": float("nan"), "relation": "unavailable"}
    maes = [float(_safe_float(row.get("test_root_mae_mean"))) for row in rows]
    reviewed = [float(_safe_float(row.get("reviewed_docs"))) for row in rows]
    if target_mae < maes[-1]:
        return {"reviewed_docs": float(reviewed[-1]), "relation": "gt_max"}
    if target_mae > maes[0]:
        return {"reviewed_docs": float(reviewed[0]), "relation": "lt_min"}
    value = float(reviewed[0])
    relation = "interpolated"
    for idx in range(len(rows) - 1):
        left_x = float(reviewed[idx])
        right_x = float(reviewed[idx + 1])
        left_y = float(maes[idx])
        right_y = float(maes[idx + 1])
        if target_mae > left_y + 1e-12 or target_mae < right_y - 1e-12:
            continue
        if abs(left_y - right_y) <= 1e-12:
            value = left_x
        else:
            left_log = math.log2(max(left_x, 1e-9))
            right_log = math.log2(max(right_x, 1e-9))
            ratio = float((target_mae - left_y) / (right_y - left_y))
            value = float(2.0 ** (left_log + ratio * (right_log - left_log)))
        break
    return {"reviewed_docs": float(value), "relation": relation}


def _summarize_efficiency_suite(payload: Mapping[str, Any]) -> Dict[str, Any]:
    recoverable_anchor = dict(payload.get("recoverable_dense_anchor") or {})
    recoverable_budget = dict(payload.get("recoverable_budget") or {})
    structural_anchor = dict(payload.get("structural_dense_anchor") or {})
    structural_budget = dict(payload.get("structural_budget") or {})

    scopes: Dict[str, Dict[str, Any]] = {}
    dense_scope_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    budget_scope_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    scope_labels_by_key: Dict[str, str] = {}

    recoverable_scope_key = str(
        recoverable_anchor.get("scope_key")
        or recoverable_anchor.get("benchmark")
        or SUPERVISION_RECOVERY_RECOVERABLE_BENCHMARK
    )
    scope_labels_by_key[recoverable_scope_key] = str(
        recoverable_anchor.get("scope_label")
        or recoverable_scope_key
    )
    for row in list(recoverable_anchor.get("rows") or []):
        dense_scope_rows[recoverable_scope_key].append(dict(row))
    for row in list(structural_anchor.get("rows") or []):
        cell_id = str(row.get("cell_id", "") or "")
        if cell_id:
            dense_scope_rows[cell_id].append(dict(row))
            if cell_id not in scope_labels_by_key:
                scope_labels_by_key[cell_id] = str(
                    row.get("scope_label")
                    or _structural_scope_label(
                        cell_id,
                        grid_name=str(row.get("hardness_grid", "") or ""),
                    )
                )
    for row in list(recoverable_budget.get("best_tree_by_budget") or []):
        dense_cell = str(row.get("cell_id", "") or recoverable_scope_key)
        budget_scope_rows[dense_cell].append(dict(row))
    for row in list(structural_budget.get("best_tree_by_budget") or []):
        cell_id = str(row.get("cell_id", "") or "")
        if cell_id:
            budget_scope_rows[cell_id].append(dict(row))

    for scope_key, anchor_rows in dense_scope_rows.items():
        budget_rows = list(budget_scope_rows.get(scope_key, []))
        fno_anchor_raw = [
            row
            for row in anchor_rows
            if str(row.get("baseline_family", "")) in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES
        ]
        best_fno_rows = []
        for train_doc_count in sorted(
            {int(_safe_int(row.get("train_doc_count"))) for row in fno_anchor_raw if int(_safe_int(row.get("train_doc_count"))) > 0}
        ):
            candidates = [
                row
                for row in fno_anchor_raw
                if int(_safe_int(row.get("train_doc_count"))) == train_doc_count
            ]
            if not candidates:
                continue
            best_fno_rows.append(
                min(candidates, key=lambda row: _safe_float(row.get("test_root_mae_mean"), float("inf")))
            )
        tree_anchor_by_family: Dict[str, List[Dict[str, Any]]] = {}
        for family in EFFICIENCY_TREE_BASELINE_FAMILIES:
            family_rows = [
                dict(row)
                for row in anchor_rows
                if str(row.get("baseline_family", "")) == family
            ]
            if family_rows:
                tree_anchor_by_family[family] = family_rows
        fno_saturated = bool(best_fno_rows) and all(
            _is_solved_floor(row.get("test_root_mae_mean")) for row in best_fno_rows
        )
        scope_rows: List[Dict[str, Any]] = []
        for row in sorted(
            budget_rows,
            key=lambda item: (
                int(_safe_int(item.get("train_doc_count"))),
                float(_safe_float(item.get("budget_total_calls_per_doc"))),
                float(_safe_float(item.get("full_doc_budget_share"))),
                str(item.get("doc_consumption_mode", "")),
            ),
        ):
            out = dict(row)
            train_doc_count = int(_safe_int(out.get("train_doc_count")))
            effective_mass = _safe_float(out.get("effective_full_doc_mass_per_doc_mean"), float("nan"))
            reviewed_equivalent_docs = (
                float(train_doc_count) * effective_mass
                if train_doc_count > 0 and math.isfinite(effective_mass)
                else float("nan")
            )
            target_mae = _safe_float(out.get("test_root_mae_mean"), float("nan"))
            fno_interp = _interpolate_equivalent_docs(best_fno_rows, target_mae=target_mae)
            tree_family = str(out.get("baseline_family", "") or "")
            tree_interp = _interpolate_equivalent_docs(
                tree_anchor_by_family.get(tree_family, []),
                target_mae=target_mae,
            )
            out.update(
                {
                    "reviewed_full_doc_equivalent_docs": reviewed_equivalent_docs,
                    "review_fraction_of_training_corpus": float(effective_mass) if math.isfinite(effective_mass) else float("nan"),
                    "fno_equivalent_train_docs": _safe_float(fno_interp.get("equivalent_train_docs"), float("nan")),
                    "fno_equivalent_train_docs_relation": str(fno_interp.get("relation", "")),
                    "fno_equivalent_train_docs_min_train_docs": _safe_int(fno_interp.get("min_train_docs"), 0),
                    "fno_equivalent_train_docs_max_train_docs": _safe_int(fno_interp.get("max_train_docs"), 0),
                    "tree_equivalent_train_docs": _safe_float(tree_interp.get("equivalent_train_docs"), float("nan")),
                    "tree_equivalent_train_docs_relation": str(tree_interp.get("relation", "")),
                    "tree_equivalent_train_docs_min_train_docs": _safe_int(tree_interp.get("min_train_docs"), 0),
                    "tree_equivalent_train_docs_max_train_docs": _safe_int(tree_interp.get("max_train_docs"), 0),
                }
            )
            reviewed_docs = _safe_float(out.get("reviewed_full_doc_equivalent_docs"), float("nan"))
            fno_equiv_docs = _safe_float(out.get("fno_equivalent_train_docs"), float("nan"))
            tree_equiv_docs = _safe_float(out.get("tree_equivalent_train_docs"), float("nan"))
            out["fno_label_efficiency_multiplier"] = (
                float(fno_equiv_docs / reviewed_docs)
                if reviewed_docs > 0.0
                and math.isfinite(fno_equiv_docs)
                and str(out.get("fno_equivalent_train_docs_relation", "")) == "interpolated"
                else float("nan")
            )
            out["tree_label_efficiency_multiplier"] = (
                float(tree_equiv_docs / reviewed_docs)
                if reviewed_docs > 0.0
                and math.isfinite(tree_equiv_docs)
                and str(out.get("tree_equivalent_train_docs_relation", "")) == "interpolated"
                else float("nan")
            )
            scope_rows.append(out)

        palette_rows = [
            row
            for row in anchor_rows
            if str(row.get("baseline_family", "")) == "palette_block_exact"
        ]
        exact_witness_exact = bool(palette_rows) and all(
            _is_solved_floor(row.get("test_root_mae_mean")) for row in palette_rows
        )
        best_fno_anchor_mae = min(
            (
                _safe_float(row.get("test_root_mae_mean"), float("inf"))
                for row in best_fno_rows
            ),
            default=float("nan"),
        )
        scopes[scope_key] = {
            "scope_key": scope_key,
            "scope_label": str(
                scope_labels_by_key.get(
                    scope_key,
                    _recoverable_scope_label(scope_key)
                    if scope_key == recoverable_scope_key
                    else _structural_scope_label(scope_key),
                )
            ),
            "dense_anchor_rows": [dict(row) for row in anchor_rows],
            "dense_fno_anchor_rows": [dict(row) for row in best_fno_rows],
            "dense_tree_anchor_rows_by_family": {
                family: [dict(row) for row in rows]
                for family, rows in tree_anchor_by_family.items()
            },
            "rows": scope_rows,
            "fno_anchor_saturated": fno_saturated,
            "exact_witness_exact": exact_witness_exact,
            "best_fno_anchor_mae": best_fno_anchor_mae,
            "best_fno_efficiency_point": (
                max(
                    (
                        row
                        for row in scope_rows
                        if math.isfinite(_safe_float(row.get("fno_label_efficiency_multiplier"), float("nan")))
                    ),
                    key=lambda row: _safe_float(row.get("fno_label_efficiency_multiplier"), float("-inf")),
                )
                if any(
                    math.isfinite(_safe_float(row.get("fno_label_efficiency_multiplier"), float("nan")))
                    for row in scope_rows
                )
                else None
            ),
            "best_tree_efficiency_point": (
                max(
                    (
                        row
                        for row in scope_rows
                        if math.isfinite(_safe_float(row.get("tree_label_efficiency_multiplier"), float("nan")))
                    ),
                    key=lambda row: _safe_float(row.get("tree_label_efficiency_multiplier"), float("-inf")),
                )
                if any(
                    math.isfinite(_safe_float(row.get("tree_label_efficiency_multiplier"), float("nan")))
                    for row in scope_rows
                )
                else None
            ),
        }

    primary_scope = _select_primary_structural_scope(scopes)
    target_tables: Dict[str, Dict[str, Any]] = {}
    for scope_key, scope in scopes.items():
        rows = list(scope.get("rows") or [])
        frontier_curve = _build_budget_review_curve(rows)
        if not frontier_curve:
            target_tables[scope_key] = {"rows": []}
            continue
        mae_values = sorted(
            {
                round(_safe_float(row.get("test_root_mae_mean")), 6)
                for row in frontier_curve
                if math.isfinite(_safe_float(row.get("test_root_mae_mean"), float("nan")))
            }
        )
        fixed_targets = [0.10, 0.05, 0.025, 0.01, 0.005]
        min_mae = min(mae_values)
        max_mae = max(mae_values)
        targets = [value for value in fixed_targets if min_mae - 1e-12 <= value <= max_mae + 1e-12]
        if mae_values:
            targets.extend([mae_values[0], mae_values[len(mae_values) // 2], mae_values[-1]])
        ordered_targets = []
        seen = set()
        for value in sorted({round(float(target), 6) for target in targets}, reverse=True):
            if value in seen:
                continue
            seen.add(value)
            ordered_targets.append(value)
        table_rows = []
        for target in ordered_targets:
            reviewed_interp = _interpolate_reviewed_docs(frontier_curve, target_mae=float(target))
            fno_interp = _interpolate_equivalent_docs(scope.get("dense_fno_anchor_rows", []), target_mae=float(target))
            tree_interp = _interpolate_equivalent_docs(
                scope.get("dense_tree_anchor_rows_by_family", {}).get("tree_neural", []),
                target_mae=float(target),
            )
            reviewed_docs = _safe_float(reviewed_interp.get("reviewed_docs"), float("nan"))
            table_rows.append(
                {
                    "target_mae": float(target),
                    "reviewed_docs_needed": reviewed_docs,
                    "reviewed_docs_relation": str(reviewed_interp.get("relation", "")),
                    "fno_equivalent_train_docs": _safe_float(fno_interp.get("equivalent_train_docs"), float("nan")),
                    "fno_equivalent_train_docs_relation": str(fno_interp.get("relation", "")),
                    "tree_equivalent_train_docs": _safe_float(tree_interp.get("equivalent_train_docs"), float("nan")),
                    "tree_equivalent_train_docs_relation": str(tree_interp.get("relation", "")),
                    "review_fraction_of_training_corpus": (
                        reviewed_docs / float(max(1, max(int(_safe_int(row.get("train_doc_count"))) for row in rows)))
                        if math.isfinite(reviewed_docs)
                        else float("nan")
                    ),
                }
            )
        target_tables[scope_key] = {"rows": table_rows}

    solved_flags = {
        scope_key: {
            "fno_anchor_saturated": bool(scope.get("fno_anchor_saturated")),
            "exact_witness_exact": bool(scope.get("exact_witness_exact")),
        }
        for scope_key, scope in scopes.items()
    }
    return {
        "primary_scope": primary_scope,
        "scopes": scopes,
        "solved_benchmark_flags": solved_flags,
        "efficiency_target_table": {
            "primary_scope": primary_scope,
            "scopes": target_tables,
        },
    }


def _summarize_fno_upper_bound(
    payload: Mapping[str, Any],
    *,
    docs_epochs: Mapping[str, Any],
    batch_quality: Mapping[str, Any],
) -> Dict[str, Any]:
    tree_by_train_docs = _best_tree_reference_by_train_docs(docs_epochs, batch_quality)
    rows = []
    for row in list(payload.get("rows", []) or []):
        out = dict(row)
        train_docs = int(_safe_int(out.get("train_docs")))
        tree_ref = dict(tree_by_train_docs.get(train_docs) or {})
        best_fno = _safe_float(out.get("best_full_doc_fno_test_root_mae"), float("nan"))
        best_tree = _safe_float(tree_ref.get("best_tree_test_root_mae"), float("nan"))
        gap_ratio = float("nan")
        if math.isfinite(best_fno) and best_fno > 0.0 and math.isfinite(best_tree):
            gap_ratio = float(best_tree / best_fno - 1.0)
        out.update(tree_ref)
        out["gap_ratio_vs_best_fno"] = gap_ratio
        rows.append(out)
    rows.sort(key=lambda item: int(_safe_int(item.get("train_docs"))))
    finite_gap_rows = [
        row
        for row in rows
        if math.isfinite(_safe_float(row.get("gap_ratio_vs_best_fno"), float("nan")))
    ]
    closest_gap = (
        min(
            finite_gap_rows,
            key=lambda row: abs(_safe_float(row.get("gap_ratio_vs_best_fno"), float("inf"))),
        )
        if finite_gap_rows
        else None
    )
    anchor_saturated = bool(rows) and all(
        _is_solved_floor(row.get("best_full_doc_fno_test_root_mae")) for row in rows
    )
    return {
        "benchmark": str(payload.get("benchmark", "")),
        "template_benchmark": str(payload.get("template_benchmark", "")),
        "rows": rows,
        "closest_gap_train_docs": int(_safe_int((closest_gap or {}).get("train_docs"))),
        "anchor_saturated": anchor_saturated,
    }


def _summarize_identifiable_zero_reference(
    fno_upper_bound: Mapping[str, Any],
) -> Dict[str, Any]:
    rows = [dict(row) for row in list(fno_upper_bound.get("rows", []) or [])]
    best_overall = (
        min(
            rows,
            key=lambda row: _safe_float(
                row.get("best_full_doc_fno_test_root_mae"),
                float("inf"),
            ),
        )
        if rows
        else None
    )
    return {
        "reference_kind": "full_doc_fno_upper_bound",
        "reference_label": "best available full-doc FNO upper bound",
        "families": list(CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES),
        "source_section": "fno_upper_bound",
        "best_family_overall": str(
            (best_overall or {}).get("best_full_doc_fno_family", "")
        ),
        "best_train_docs": int(_safe_int((best_overall or {}).get("train_docs"))),
        "best_test_root_mae": _safe_float(
            (best_overall or {}).get("best_full_doc_fno_test_root_mae")
        ),
        "note": (
            "The canonical identifiable-zero reference is the best full-doc FNO "
            "family, not the older mixed tree/FNO appendix view."
        ),
    }


def _summarize_fno_upper_bound_from_efficiency(
    efficiency_suite: Mapping[str, Any],
) -> Dict[str, Any]:
    scopes = dict(efficiency_suite.get("scopes") or {})
    recoverable_scope_key = _preferred_recoverable_scope_key(scopes)
    recoverable = dict(scopes.get(recoverable_scope_key) or {})
    rows = []
    best_tree_rows = [
        dict(row)
        for row in list(recoverable.get("dense_anchor_rows") or [])
        if str(row.get("baseline_family", "")) in EFFICIENCY_TREE_BASELINE_FAMILIES
    ]
    for row in list(recoverable.get("dense_fno_anchor_rows") or []):
        train_docs = int(_safe_int(row.get("train_doc_count", row.get("train_docs"))))
        tree_candidates = [
            candidate
            for candidate in best_tree_rows
            if int(_safe_int(candidate.get("train_doc_count"))) == train_docs
        ]
        best_tree = (
            min(tree_candidates, key=lambda item: _safe_float(item.get("test_root_mae_mean"), float("inf")))
            if tree_candidates
            else None
        )
        best_fno = _safe_float(row.get("test_root_mae_mean"), float("nan"))
        best_tree_mae = _safe_float((best_tree or {}).get("test_root_mae_mean"), float("nan"))
        gap_ratio = float("nan")
        if math.isfinite(best_fno) and best_fno > 0.0 and math.isfinite(best_tree_mae):
            gap_ratio = float(best_tree_mae / best_fno - 1.0)
        rows.append(
            {
                "train_docs": train_docs,
                "best_full_doc_fno_family": str(row.get("baseline_family", "")),
                "best_full_doc_fno_test_root_mae": best_fno,
                "best_tree_test_root_mae": best_tree_mae,
                "best_tree_source": str((best_tree or {}).get("baseline_family", "")),
                "gap_ratio_vs_best_fno": gap_ratio,
            }
        )
    rows.sort(key=lambda item: int(_safe_int(item.get("train_docs"))))
    return {
        "benchmark": recoverable_scope_key,
        "template_benchmark": "efficiency_suite",
        "rows": rows,
        "closest_gap_train_docs": int(_safe_int((rows[-1] if rows else {}).get("train_docs"))),
        "anchor_saturated": bool(recoverable.get("fno_anchor_saturated")),
    }


def _summarize_oracle_budget_from_efficiency(
    efficiency_suite: Mapping[str, Any],
) -> Dict[str, Any]:
    scopes = dict(efficiency_suite.get("scopes") or {})
    primary_scope = str(efficiency_suite.get("primary_scope", "") or "")
    primary = dict(scopes.get(primary_scope) or {})
    return {
        "scope_name": primary_scope,
        "scope_label": str(primary.get("scope_label", primary_scope)),
        "train_doc_count": max(
            (
                int(_safe_int(row.get("train_doc_count")))
                for row in list(primary.get("rows") or [])
            ),
            default=0,
        ),
        "rows": [dict(row) for row in list(primary.get("rows") or [])],
        "best_efficiency_point": dict(primary.get("best_tree_efficiency_point") or {}),
        "best_fno_efficiency_point": dict(primary.get("best_fno_efficiency_point") or {}),
        "fno_anchor_saturated": bool(primary.get("fno_anchor_saturated")),
        "exact_witness_exact": bool(primary.get("exact_witness_exact")),
    }


def _summarize_oracle_budget_frontier(
    payload: Mapping[str, Any],
    *,
    fno_upper_bound: Mapping[str, Any],
) -> Dict[str, Any]:
    tree_rows = [dict(row) for row in list(payload.get("best_tree_by_budget") or [])]
    if not tree_rows:
        return {
            "train_doc_count": 0,
            "budget_levels_per_doc": [float(value) for value in list(payload.get("budget_levels_per_doc") or [])],
            "rows": [],
            "best_efficiency_point": None,
        }

    source_rows = list(payload.get("tree_rows") or []) + list(payload.get("reference_rows") or [])
    train_doc_count = 0
    for row in source_rows:
        value = int(_safe_int((row or {}).get("train_doc_count")))
        if value > 0:
            train_doc_count = value
            break

    fno_rows = sorted(
        [dict(row) for row in list(fno_upper_bound.get("rows") or [])],
        key=lambda row: int(_safe_int(row.get("train_docs"))),
    )
    same_train_docs_reference = next(
        (
            row
            for row in fno_rows
            if int(_safe_int(row.get("train_docs"))) == int(train_doc_count)
        ),
        None,
    )

    rows: List[Dict[str, Any]] = []
    for row in sorted(tree_rows, key=lambda item: float(_safe_float(item.get("budget_total_calls_per_doc")))):
        out = dict(row)
        tree_mae = _safe_float(out.get("test_root_mae_mean"))
        effective_mass = _safe_float(out.get("effective_full_doc_mass_per_doc_mean"))
        reviewed_equivalent_docs = (
            float(train_doc_count) * effective_mass
            if train_doc_count > 0 and math.isfinite(effective_mass)
            else float("nan")
        )
        matched_full_label = next(
            (
                candidate
                for candidate in fno_rows
                if _safe_float(candidate.get("best_full_doc_fno_test_root_mae"), float("inf")) <= tree_mae
            ),
            None,
        )
        matched_train_docs = int(_safe_int((matched_full_label or {}).get("train_docs"))) if matched_full_label else 0
        label_efficiency_multiplier = float("nan")
        if reviewed_equivalent_docs > 0.0 and matched_train_docs > 0:
            label_efficiency_multiplier = float(matched_train_docs / reviewed_equivalent_docs)
        same_train_docs_fno_mae = _safe_float(
            (same_train_docs_reference or {}).get("best_full_doc_fno_test_root_mae"),
            float("nan"),
        )
        gap_vs_same_train_docs_fno = float("nan")
        if math.isfinite(tree_mae) and math.isfinite(same_train_docs_fno_mae):
            gap_vs_same_train_docs_fno = float(tree_mae - same_train_docs_fno_mae)
        out.update(
            {
                "train_doc_count": int(train_doc_count),
                "reviewed_full_doc_equivalent_docs": reviewed_equivalent_docs,
                "matched_full_label_train_docs": int(matched_train_docs),
                "matched_full_label_fno_family": str((matched_full_label or {}).get("best_full_doc_fno_family", "")),
                "matched_full_label_test_root_mae": _safe_float(
                    (matched_full_label or {}).get("best_full_doc_fno_test_root_mae")
                ),
                "label_efficiency_multiplier": label_efficiency_multiplier,
                "same_train_docs_fno_test_root_mae": same_train_docs_fno_mae,
                "gap_vs_same_train_docs_fno": gap_vs_same_train_docs_fno,
            }
        )
        rows.append(out)

    finite_efficiency_rows = [
        row
        for row in rows
        if math.isfinite(_safe_float(row.get("label_efficiency_multiplier"), float("nan")))
    ]
    best_efficiency_point = (
        max(
            finite_efficiency_rows,
            key=lambda row: _safe_float(row.get("label_efficiency_multiplier"), float("-inf")),
        )
        if finite_efficiency_rows
        else None
    )
    return {
        "benchmark": str(payload.get("benchmark", "")),
        "study_name": str(payload.get("study_name", "")),
        "train_doc_count": int(train_doc_count),
        "budget_levels_per_doc": [float(value) for value in list(payload.get("budget_levels_per_doc") or [])],
        "full_doc_budget_shares": [float(value) for value in list(payload.get("full_doc_budget_shares") or [])],
        "rows": rows,
        "best_efficiency_point": dict(best_efficiency_point) if best_efficiency_point is not None else None,
    }


def _summarize_large_batch_diagnosis(payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [dict(row) for row in list(payload.get("rows", []) or [])]
    rows.sort(key=lambda row: (str(row.get("study_block", "")), int(_safe_int(row.get("batch_size"))), float(_safe_float(row.get("lr")))))
    return {
        "train_docs": int(_safe_int(payload.get("train_docs"))),
        "target_total_steps": int(_safe_int(payload.get("target_total_steps"))),
        "rows": rows,
        "classification": str(payload.get("classification", "")),
        "recommendation": dict(payload.get("recommendation") or {}),
        "best_retuned_1024": dict(payload.get("best_retuned_1024") or {}),
        "constant_steps_reference": dict(payload.get("constant_steps_reference") or {}),
    }


def _summarize_supervision_sweep(
    payload: Mapping[str, Any],
    *,
    expected_train_doc_counts: Sequence[int] | None = None,
) -> Dict[str, Any]:
    leaf_profiles = [str(item) for item in list(payload.get("leaf_profiles") or [])]
    internal_profiles = [str(item) for item in list(payload.get("internal_profiles") or [])]

    def _profile_rank(row: Mapping[str, Any]) -> tuple[int, int]:
        leaf_profile = str(row.get("leaf_profile", ""))
        internal_profile = str(row.get("internal_profile", ""))
        return (
            leaf_profiles.index(leaf_profile)
            if leaf_profile in leaf_profiles
            else len(leaf_profiles),
            internal_profiles.index(internal_profile)
            if internal_profile in internal_profiles
            else len(internal_profiles),
        )

    rows = [dict(row) for row in list(payload.get("rows") or [])]
    rows.sort(
        key=lambda row: (
            int(_safe_int(row.get("train_doc_count"))),
            _safe_float(row.get("mean_test_root_mae"), float("inf")),
            *_profile_rank(row),
        )
    )
    expected_docs = {
        int(value)
        for value in list(expected_train_doc_counts or [])
        if int(_safe_int(value)) > 0
    }
    observed_docs = {
        int(_safe_int(row.get("train_doc_count")))
        for row in rows
        if int(_safe_int(row.get("train_doc_count"))) > 0
    }
    if expected_docs and observed_docs and not observed_docs.issubset(expected_docs):
        return {
            "rows": rows,
            "by_train_docs": {},
            "best_by_train_docs": {},
            "leaf_profiles": leaf_profiles,
            "internal_profiles": internal_profiles,
            "best_overall": {},
            "status": "suspicious",
            "reason": (
                "supervision sweep includes train-doc counts outside the selected version root "
                f"({sorted(observed_docs)} vs expected {sorted(expected_docs)})"
            ),
        }
    finite_rows = [
        row
        for row in rows
        if math.isfinite(_safe_float(row.get("mean_test_root_mae"), float("nan")))
    ]
    profile_signatures = {
        (
            str(row.get("leaf_profile", "")),
            str(row.get("internal_profile", "")),
            str(row.get("leaf_supervision_kind", "")),
            float(_safe_float(row.get("leaf_label_rate"), 0.0)),
            str(row.get("internal_supervision_kind", "")),
            float(_safe_float(row.get("internal_label_rate"), 0.0)),
        )
        for row in finite_rows
    }
    metric_values = {
        round(_safe_float(row.get("mean_test_root_mae"), float("nan")), 12)
        for row in finite_rows
    }
    if len(profile_signatures) >= 3 and len(metric_values) <= 1:
        return {
            "rows": rows,
            "by_train_docs": {},
            "best_by_train_docs": {},
            "leaf_profiles": leaf_profiles,
            "internal_profiles": internal_profiles,
            "best_overall": {},
            "status": "suspicious",
            "reason": "supervision metrics are invariant across materially different supervision profiles",
        }
    by_train_docs: Dict[str, Dict[str, Any]] = {}
    for train_docs, value in dict(payload.get("by_train_docs") or {}).items():
        subrows = [dict(row) for row in list(value.get("rows") or [])]
        subrows.sort(
            key=lambda row: (
                _safe_float(row.get("mean_test_root_mae"), float("inf")),
                *_profile_rank(row),
            )
        )
        best_root_row = dict(subrows[0]) if subrows else {}
        baseline = next(
            (
                dict(row)
                for row in subrows
                if str(row.get("leaf_profile", "")) == "none"
                and str(row.get("internal_profile", "")) == "none"
            ),
            {},
        )
        gain_pct = float("nan")
        if baseline and best_root_row:
            baseline_mae = _safe_float(baseline.get("mean_test_root_mae"), float("nan"))
            best_mae = _safe_float(best_root_row.get("mean_test_root_mae"), float("nan"))
            if math.isfinite(baseline_mae) and baseline_mae > 0.0 and math.isfinite(best_mae):
                gain_pct = 100.0 * (baseline_mae - best_mae) / baseline_mae
        by_train_docs[str(train_docs)] = {
            "rows": subrows,
            "best_root_row": best_root_row,
            "baseline_none_none": baseline,
            "gain_pct_vs_none_none": gain_pct,
        }
    best_overall = min(
        rows,
        key=lambda row: (
            _safe_float(row.get("mean_test_root_mae"), float("inf")),
            int(_safe_int(row.get("train_doc_count"), 0)),
            *_profile_rank(row),
        ),
        default={},
    )
    best_by_train_docs = {
        train_docs: dict(value.get("best_root_row") or {})
        for train_docs, value in sorted(
            by_train_docs.items(),
            key=lambda item: int(_safe_int(item[0], 0)),
        )
    }
    return {
        "rows": rows,
        "by_train_docs": by_train_docs,
        "best_by_train_docs": best_by_train_docs,
        "leaf_profiles": leaf_profiles,
        "internal_profiles": internal_profiles,
        "best_overall": dict(best_overall),
        "status": "ready",
    }


def _finite_values(values: Iterable[Any]) -> List[float]:
    out = []
    for value in values:
        fv = _safe_float(value)
        if math.isfinite(fv):
            out.append(fv)
    return out


def _summarize_support(payload: Mapping[str, Any]) -> Dict[str, Any]:
    model_families = {
        str(row.get("model_family", "")).strip()
        for row in list(payload.get("rows", []) or [])
        if str(row.get("model_family", "")).strip()
    }
    train_doc_counts = {
        int(_safe_int(row.get("train_docs")))
        for row in list(payload.get("rows", []) or [])
        if int(_safe_int(row.get("train_docs"))) > 0
    }
    if len(model_families) > 1:
        return {
            "rows": [],
            "status": "incompatible",
            "reason": f"support summary mixes model families: {', '.join(sorted(model_families))}",
        }
    if len(train_doc_counts) > 1:
        return {
            "rows": [],
            "status": "incompatible",
            "reason": (
                "support summary mixes train-doc populations without an explicit selection: "
                f"{sorted(train_doc_counts)}"
            ),
        }
    rows = []
    for row in list(payload.get("rows", []) or []):
        fixed_leaf_tokens = _safe_int((row or {}).get("fixed_leaf_tokens"), default=-1)
        if fixed_leaf_tokens <= 0:
            continue
        rows.append(dict(row))
    grouped: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(_safe_int(row.get("fixed_leaf_tokens")))].append(row)
    by_leaf_tokens: List[Dict[str, Any]] = []
    for fixed_leaf_tokens, sub in sorted(grouped.items()):
        supported = [row for row in sub if _safe_float(row.get("local_oracle_coverage")) >= 0.99]
        unsupported = [row for row in sub if abs(_safe_float(row.get("local_oracle_coverage"))) <= 1e-12]
        mean_leaves = _finite_values(row.get("mean_leaves") for row in sub)
        mean_internal = _finite_values(row.get("mean_internal_nodes") for row in sub)
        recovery_values = _finite_values(row.get("root_utility_recovery") for row in sub)
        by_leaf_tokens.append(
            {
                "fixed_leaf_tokens": int(fixed_leaf_tokens),
                "median_mean_leaves": float(median(mean_leaves)) if mean_leaves else float("nan"),
                "median_mean_internal_nodes": float(median(mean_internal)) if mean_internal else float("nan"),
                "best_supported_root_mae": min(
                    (_safe_float(row.get("learned_root_mae", float("inf"))) for row in supported),
                    default=float("nan"),
                ),
                "median_supported_root_mae": float(
                    median(
                        _finite_values(row.get("learned_root_mae") for row in supported)
                    )
                )
                if supported
                else float("nan"),
                "median_unsupported_root_mae": float(
                    median(
                        _finite_values(row.get("learned_root_mae") for row in unsupported)
                    )
                )
                if unsupported
                else float("nan"),
                "best_root_utility_recovery": max(recovery_values) if recovery_values else float("nan"),
            }
        )
    by_leaf_tokens.sort(key=lambda row: int(row["fixed_leaf_tokens"]))
    return {"rows": by_leaf_tokens, "status": "ready"}


def _plot_batch_throughput(batch_timing: Mapping[str, Any], output_path: Path) -> bool:
    rows = list(batch_timing.get("rows", []) or [])
    if not rows:
        return False
    batch_sizes = [int(_safe_int(row.get("batch_size"))) for row in rows]
    docs_wall = [_safe_float(row.get("docs_per_s_wall")) for row in rows]
    docs_train = [_safe_float(row.get("docs_per_s_train_loop")) for row in rows]
    eval_exact = [_safe_float(row.get("exact_metric_eval_s")) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 10.0), constrained_layout=True)
    axes[0].plot(batch_sizes, docs_wall, marker="o", label="wall docs/s", color="#1d3557")
    axes[0].plot(batch_sizes, docs_train, marker="s", label="train-loop docs/s", color="#2a9d8f")
    axes[0].set_title("Leaf-Laws Full-Pipeline Batch Throughput")
    axes[0].set_xlabel("Batch Size")
    axes[0].set_ylabel("Docs / s")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(batch_sizes, eval_exact, marker="o", color="#c44e52")
    axes[1].set_title("Exact Eval Time vs Batch Size")
    axes[1].set_xlabel("Batch Size")
    axes[1].set_ylabel("Seconds")
    axes[1].grid(alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_medium_grid(batch_quality: Mapping[str, Any], output_path: Path) -> bool:
    if str(batch_quality.get("status", "")) not in {"", "ready"}:
        return False
    rows = list(batch_quality.get("rows", []) or [])
    if not rows:
        return False
    xs = [_safe_float(row.get("mean_docs_per_s_wall_effective")) for row in rows]
    ys = [_safe_float(row.get("mean_best_val_mae")) for row in rows]
    labels = [str(_safe_int(row.get("batch_size"))) for row in rows]

    fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
    ax.scatter(xs, ys, s=80, color="#1d3557")
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(f"bs={label}", (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9)
    doc_label = _format_doc_count(batch_quality.get("train_docs"))
    epochs_label = int(_safe_int(batch_quality.get("epochs"), 0))
    ax.set_title(f"{doc_label} / {epochs_label}ep Speed-Quality Frontier")
    ax.set_xlabel("Effective Docs / s")
    ax.set_ylabel("Mean Best Val MAE")
    ax.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_docs_epochs(docs_epochs: Mapping[str, Any], output_path: Path) -> bool:
    rows = list(docs_epochs.get("rows", []) or [])
    if not rows:
        return False
    grouped: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(_safe_int(row.get("train_docs")))].append(row)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 10.0), constrained_layout=True)
    palette = ["#1d3557", "#d17c00", "#2a9d8f", "#c44e52", "#6c5ce7"]
    for idx, train_docs in enumerate(sorted(grouped)):
        sub = sorted(grouped[train_docs], key=lambda row: int(_safe_int(row.get("epochs"))))
        epochs = [int(_safe_int(row.get("epochs"))) for row in sub]
        maes = [_safe_float(row.get("best_val_mae")) for row in sub]
        docs_per_s = [_safe_float(row.get("docs_per_s_wall_effective")) for row in sub]
        color = palette[idx % len(palette)]
        label = f"train_docs={train_docs}"
        axes[0].plot(epochs, maes, marker="o", label=label, color=color)
        axes[1].plot(epochs, docs_per_s, marker="o", label=label, color=color)

    axes[0].set_title("Docs × Epochs: Validation MAE")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Best Val MAE")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Docs × Epochs: Effective Throughput")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Effective Docs / s")
    axes[1].grid(alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_geometry_support(support: Mapping[str, Any], output_path: Path) -> bool:
    rows = list(support.get("rows", []) or [])
    if not rows:
        return False
    leaf_tokens = [int(_safe_int(row.get("fixed_leaf_tokens"))) for row in rows]
    supported = [_safe_float(row.get("best_supported_root_mae")) for row in rows]
    unsupported = [_safe_float(row.get("median_unsupported_root_mae")) for row in rows]

    fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
    ax.plot(leaf_tokens, supported, marker="o", label="best supported root MAE", color="#2a9d8f")
    ax.plot(leaf_tokens, unsupported, marker="s", label="median unsupported root MAE", color="#c44e52")
    ax.set_title("Tree Geometry vs Support")
    ax.set_xlabel("Fixed Leaf Tokens")
    ax.set_ylabel("Root MAE")
    ax.grid(alpha=0.25)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_law_packages(law_packages: Mapping[str, Any], output_path: Path) -> bool:
    rows = list(law_packages.get("rows", []) or [])
    if not rows:
        return False
    names = [str(row.get("name")) for row in rows]
    tree_root = [_safe_float(row.get("test_root_mae")) for row in rows]
    doc_fno_root = [_safe_float(row.get("doc_fno_test_root_mae")) for row in rows]
    doc_fno_label = str(law_packages.get("doc_fno_label", "doc FNO reference") or "doc FNO reference")

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 8.5), constrained_layout=True)
    x = list(range(len(names)))
    axes[0].bar(x, tree_root, color="#1d3557", label="tree root MAE")
    axes[0].plot(x, doc_fno_root, marker="o", linewidth=2.0, color="#d17c00", label=f"{doc_fno_label} root MAE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, ha="right")
    axes[0].set_ylabel("Root MAE")
    axes[0].set_title(f"Direct Law-Package Comparison With {doc_fno_label.title()}")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    leaf_mae = [_safe_float(row.get("test_leaf_mae")) for row in rows]
    merge_mae = [_safe_float(row.get("test_merge_mae")) for row in rows]
    width = 0.35
    axes[1].bar([i - width / 2.0 for i in x], leaf_mae, width=width, color="#2a9d8f", label="leaf MAE")
    axes[1].bar([i + width / 2.0 for i in x], merge_mae, width=width, color="#c44e52", label="merge MAE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15, ha="right")
    axes[1].set_ylabel("MAE")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_fno_upper_bound(fno_upper_bound: Mapping[str, Any], output_path: Path) -> bool:
    rows = list(fno_upper_bound.get("rows", []) or [])
    if not rows:
        return False
    train_docs = [int(_safe_int(row.get("train_docs"))) for row in rows]
    best_fno = [_safe_float(row.get("best_full_doc_fno_test_root_mae")) for row in rows]
    best_tree = [_safe_float(row.get("best_tree_test_root_mae")) for row in rows]
    gap_pct = [100.0 * _safe_float(row.get("gap_ratio_vs_best_fno")) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 10.0), constrained_layout=True)
    axes[0].plot(train_docs, best_fno, marker="o", linewidth=2.0, color="#1d4ed8", label="best full-doc FNO")
    axes[0].plot(train_docs, best_tree, marker="s", linewidth=2.0, color="#c44e52", label="best tree point")
    axes[0].set_title("Identifiable-Zero Reference: Best Tree vs Full-Doc FNO")
    axes[0].set_xlabel("Train Docs")
    axes[0].set_ylabel("Root MAE")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    finite_gaps = [value for value in gap_pct if math.isfinite(value)]
    if bool(fno_upper_bound.get("anchor_saturated")) or not finite_gaps:
        axes[1].axis("off")
        axes[1].text(
            0.5,
            0.5,
            "FNO anchor already solved on this grid.\nPercent-gap reporting is suppressed.",
            ha="center",
            va="center",
            fontsize=12,
        )
    else:
        axes[1].plot(train_docs, gap_pct, marker="o", linewidth=2.0, color="#6c5ce7")
        axes[1].axhline(0.0, color="#111827", linestyle="--", linewidth=1.5)
        axes[1].set_title("Tree Gap vs Identifiable-Zero FNO Reference")
        axes[1].set_xlabel("Train Docs")
        axes[1].set_ylabel("Gap vs FNO (%)")
        axes[1].grid(alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_oracle_budget_frontier(
    oracle_budget: Mapping[str, Any],
    output_path: Path,
) -> bool:
    rows = list(oracle_budget.get("rows", []) or [])
    if not rows:
        return False
    fno_curve_rows = [
        row
        for row in rows
        if math.isfinite(_safe_float(row.get("reviewed_full_doc_equivalent_docs"), float("nan")))
    ]
    if not fno_curve_rows:
        return False

    tree_x = [_safe_float(row.get("reviewed_full_doc_equivalent_docs")) for row in fno_curve_rows]
    tree_y = [_safe_float(row.get("test_root_mae_mean")) for row in fno_curve_rows]
    tree_labels = [
        f"calls={_safe_float(row.get('budget_total_calls_per_doc')):.2g}, share={_safe_float(row.get('full_doc_budget_share')):.2g}"
        for row in fno_curve_rows
    ]
    matched_rows = [
        row
        for row in fno_curve_rows
        if int(_safe_int(row.get("matched_full_label_train_docs"))) > 0
    ]
    train_doc_count = int(_safe_int(oracle_budget.get("train_doc_count")))
    same_train_docs_mae = _safe_float(
        (matched_rows[0] if matched_rows else {}).get("same_train_docs_fno_test_root_mae"),
        float("nan"),
    )
    fno_reference_x = []
    fno_reference_y = []
    for row in matched_rows:
        matched_docs = int(_safe_int(row.get("matched_full_label_train_docs")))
        matched_mae = _safe_float(row.get("matched_full_label_test_root_mae"))
        if matched_docs > 0 and math.isfinite(matched_mae):
            fno_reference_x.append(float(matched_docs))
            fno_reference_y.append(matched_mae)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 10.0), constrained_layout=True)
    axes[0].scatter(tree_x, tree_y, s=90, color="#166534", label="budgeted tree point")
    if fno_reference_x:
        order = sorted(range(len(fno_reference_x)), key=lambda idx: fno_reference_x[idx])
        axes[0].plot(
            [fno_reference_x[idx] for idx in order],
            [fno_reference_y[idx] for idx in order],
            marker="o",
            linewidth=2.0,
            color="#1d4ed8",
            label="matched full-doc FNO point",
        )
    if train_doc_count > 0 and math.isfinite(same_train_docs_mae):
        axes[0].axvline(float(train_doc_count), color="#6b7280", linestyle="--", linewidth=1.2)
        axes[0].axhline(same_train_docs_mae, color="#6b7280", linestyle=":", linewidth=1.2)
    for x, y, label in zip(tree_x, tree_y, tree_labels):
        axes[0].annotate(label, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8)
    axes[0].set_title("Oracle Budget Frontier: MAE vs Reviewed Full-Doc-Equivalent Labels")
    axes[0].set_xlabel("Reviewed Full-Doc-Equivalent Docs")
    axes[0].set_ylabel("Root MAE")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    if matched_rows:
        xs = [_safe_float(row.get("reviewed_full_doc_equivalent_docs")) for row in matched_rows]
        ys = [float(int(_safe_int(row.get("matched_full_label_train_docs")))) for row in matched_rows]
        axes[1].scatter(xs, ys, s=90, color="#c44e52")
        max_axis = max(max(xs, default=0.0), max(ys, default=0.0), float(train_doc_count))
        if max_axis > 0.0:
            axes[1].plot([0.0, max_axis], [0.0, max_axis], linestyle="--", color="#111827", linewidth=1.2)
        for row, x, y in zip(matched_rows, xs, ys):
            multiplier = _safe_float(row.get("label_efficiency_multiplier"))
            label = (
                f"x{multiplier:.2f}"
                if math.isfinite(multiplier)
                else f"calls={_safe_float(row.get('budget_total_calls_per_doc')):.2g}"
            )
            axes[1].annotate(label, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8)
    else:
        axes[1].text(
            0.5,
            0.5,
            "No full-label-equivalent match found\nagainst the current full-doc FNO curve.",
            ha="center",
            va="center",
            fontsize=11,
        )
    axes[1].set_title("Effective Full-Label Equivalence")
    axes[1].set_xlabel("Reviewed Full-Doc-Equivalent Docs")
    axes[1].set_ylabel("Matched Full-Label Train Docs")
    axes[1].grid(alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_doc_equivalent_frontiers(
    payload: Mapping[str, Any],
    output_path: Path,
) -> bool:
    frontiers = dict(payload.get("frontiers") or {})
    target_table = dict(payload.get("target_table") or {})
    scopes = dict(frontiers.get("scopes") or {})
    primary_scope = str(frontiers.get("primary_scope", "") or "")
    scope = dict(scopes.get(primary_scope) or {})
    rows = list(scope.get("rows") or [])
    if not rows:
        return False
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 13.0), constrained_layout=True)

    budget_rows = [
        row
        for row in rows
        if _safe_float(row.get("reviewed_full_doc_equivalent_docs"), float("nan")) > 0.0
    ]
    x_budget = [_safe_float(row.get("reviewed_full_doc_equivalent_docs")) for row in budget_rows]
    y_budget = [_safe_float(row.get("test_root_mae_mean")) for row in budget_rows]
    axes[0].scatter(x_budget, y_budget, s=80, color="#166534", label="budgeted tree point")
    fno_anchor = list(scope.get("dense_fno_anchor_rows") or [])
    if fno_anchor:
        axes[0].plot(
            [float(_safe_int(row.get("train_doc_count"))) for row in fno_anchor],
            [_safe_float(row.get("test_root_mae_mean")) for row in fno_anchor],
            marker="o",
            linewidth=2.0,
            color="#1d4ed8",
            label="dense full-doc FNO anchor",
        )
    tree_anchors = dict(scope.get("dense_tree_anchor_rows_by_family") or {})
    palette = {"tree_neural_c2": "#c44e52", "tree_neural": "#d17c00"}
    for family, family_rows in tree_anchors.items():
        axes[0].plot(
            [float(_safe_int(row.get("train_doc_count"))) for row in family_rows],
            [_safe_float(row.get("test_root_mae_mean")) for row in family_rows],
            marker="s",
            linewidth=1.8,
            color=palette.get(str(family), "#6c5ce7"),
            label=f"dense {family} anchor",
        )
    axes[0].set_title(f"MAE vs Reviewed Full-Doc-Equivalent Docs: {scope.get('scope_label', primary_scope)}")
    axes[0].set_xlabel("Reviewed Full-Doc-Equivalent Docs")
    axes[0].set_ylabel("Root MAE")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    finite_rows = [
        row
        for row in budget_rows
        if math.isfinite(_safe_float(row.get("fno_equivalent_train_docs"), float("nan")))
        or math.isfinite(_safe_float(row.get("tree_equivalent_train_docs"), float("nan")))
    ]
    if finite_rows:
        reviewed = [_safe_float(row.get("reviewed_full_doc_equivalent_docs")) for row in finite_rows]
        fno_equiv = [
            _safe_float(row.get("fno_equivalent_train_docs"), float("nan"))
            if str(row.get("fno_equivalent_train_docs_relation", "")) == "interpolated"
            else float("nan")
            for row in finite_rows
        ]
        tree_equiv = [
            _safe_float(row.get("tree_equivalent_train_docs"), float("nan"))
            if str(row.get("tree_equivalent_train_docs_relation", "")) == "interpolated"
            else float("nan")
            for row in finite_rows
        ]
        axes[1].plot(reviewed, fno_equiv, marker="o", linewidth=2.0, color="#1d4ed8", label="FNO-equivalent docs")
        axes[1].plot(reviewed, tree_equiv, marker="s", linewidth=2.0, color="#c44e52", label="tree-equivalent docs")
        max_axis = max(
            [value for value in reviewed + fno_equiv + tree_equiv if math.isfinite(value)],
            default=0.0,
        )
        if max_axis > 0.0:
            axes[1].plot([0.0, max_axis], [0.0, max_axis], linestyle="--", color="#111827", linewidth=1.2)
        axes[1].legend()
    else:
        axes[1].axis("off")
        axes[1].text(
            0.5,
            0.5,
            "No non-degenerate equivalent-doc surface\nfor this scope.",
            ha="center",
            va="center",
            fontsize=11,
        )
    axes[1].set_title("Equivalent Fully Labeled Docs vs Reviewed Docs")
    axes[1].set_xlabel("Reviewed Full-Doc-Equivalent Docs")
    axes[1].set_ylabel("Equivalent Fully Labeled Docs")
    axes[1].grid(alpha=0.25)

    axes[2].axis("off")
    primary_table = dict((target_table.get("scopes") or {}).get(primary_scope) or {})
    table_rows = list(primary_table.get("rows") or [])[:8]
    if table_rows:
        cell_text = [
            [
                f"{_safe_float(row.get('target_mae')):.4g}",
                _format_doc_count(int(round(_safe_float(row.get("reviewed_docs_needed"), 0.0))))
                if math.isfinite(_safe_float(row.get("reviewed_docs_needed"), float("nan")))
                else "n/a",
                _format_equivalent_docs(row, key="fno_equivalent_train_docs"),
                _format_equivalent_docs(row, key="tree_equivalent_train_docs"),
                f"{100.0 * _safe_float(row.get('review_fraction_of_training_corpus'), 0.0):.0f}%",
            ]
            for row in table_rows
        ]
        table = axes[2].table(
            cellText=cell_text,
            colLabels=["target MAE", "reviewed docs", "FNO-equiv docs", "tree-equiv docs", "review fraction"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)
    else:
        axes[2].text(0.5, 0.5, "No target-MAE table available.", ha="center", va="center")
    axes[2].set_title("Target-MAE Table")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_large_batch_diagnosis(large_batch: Mapping[str, Any], output_path: Path) -> bool:
    rows = list(large_batch.get("rows", []) or [])
    if not rows:
        return False
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("study_block", ""))].append(row)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 10.0), constrained_layout=True)
    palette = {
        "fixed_epoch": "#c44e52",
        "constant_steps": "#1d3557",
        "retune_1024": "#2a9d8f",
    }
    for block, subrows in sorted(grouped.items()):
        ordered = sorted(subrows, key=lambda row: (int(_safe_int(row.get("batch_size"))), float(_safe_float(row.get("lr")))))
        xs = [int(_safe_int(row.get("batch_size"))) for row in ordered]
        maes = [_safe_float(row.get("best_val_mae")) for row in ordered]
        speeds = [_safe_float(row.get("docs_per_s_wall")) for row in ordered]
        color = palette.get(block, "#6c5ce7")
        axes[0].plot(xs, maes, marker="o", linewidth=2.0, color=color, label=block)
        axes[1].plot(xs, speeds, marker="o", linewidth=2.0, color=color, label=block)

    axes[0].set_title("Large-Batch Diagnosis: Val MAE")
    axes[0].set_xlabel("Batch Size")
    axes[0].set_ylabel("Best Val MAE")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Large-Batch Diagnosis: Wall Throughput")
    axes[1].set_xlabel("Batch Size")
    axes[1].set_ylabel("Docs / s")
    axes[1].grid(alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_supervision_sweep(supervision: Mapping[str, Any], output_path: Path) -> bool:
    if str(supervision.get("status", "")) not in {"", "ready"}:
        return False
    by_train_docs = dict(supervision.get("by_train_docs") or {})
    if not by_train_docs:
        return False
    max_train_docs = max(int(key) for key in by_train_docs)
    payload = dict(by_train_docs.get(str(max_train_docs)) or {})
    rows = [dict(row) for row in list(payload.get("rows") or [])]
    leaf_profiles = list(supervision.get("leaf_profiles") or [])
    internal_profiles = list(supervision.get("internal_profiles") or [])
    if not rows or not leaf_profiles or not internal_profiles:
        return False

    grid = [[float("nan") for _ in leaf_profiles] for _ in internal_profiles]
    for row in rows:
        leaf_profile = str(row.get("leaf_profile", ""))
        internal_profile = str(row.get("internal_profile", ""))
        if leaf_profile not in leaf_profiles or internal_profile not in internal_profiles:
            continue
        x_idx = leaf_profiles.index(leaf_profile)
        y_idx = internal_profiles.index(internal_profile)
        grid[y_idx][x_idx] = _safe_float(row.get("mean_test_root_mae"), float("nan"))
    if not any(math.isfinite(value) for row in grid for value in row):
        return False

    fig, ax = plt.subplots(
        figsize=(max(8.0, 1.15 * len(leaf_profiles)), max(4.5, 0.75 * len(internal_profiles))),
        constrained_layout=True,
    )
    image = ax.imshow(grid, aspect="auto", interpolation="nearest")
    ax.set_title(f"Supervision Sweep Root MAE ({_format_doc_count(max_train_docs)} docs)")
    ax.set_xlabel("Leaf Supervision Profile")
    ax.set_ylabel("Internal Supervision Profile")
    ax.set_xticks(range(len(leaf_profiles)))
    ax.set_xticklabels(leaf_profiles, rotation=45, ha="right")
    ax.set_yticks(range(len(internal_profiles)))
    ax.set_yticklabels(internal_profiles)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Mean Test Root MAE")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _hazard_panel_diag_from_mapping(payload: Mapping[str, Any]) -> Dict[str, Any]:
    for key in (
        "test_target_diagnostics",
        "target_diagnostics",
        "root_target_diagnostics",
    ):
        candidate = dict(payload.get(key) or {})
        if candidate.get("condition_diagnostics"):
            return candidate
    config = dict(payload.get("config") or {})
    for key in ("test_target_diagnostics", "target_diagnostics"):
        candidate = dict(config.get(key) or {})
        if candidate.get("condition_diagnostics"):
            return candidate
    return {}


def _hazard_panel_id_from_mapping(payload: Mapping[str, Any]) -> str:
    for key in ("hazard_panel_id", "panel_id"):
        text = str(payload.get(key, "") or "").strip()
        if text:
            return text
    config = dict(payload.get("config") or {})
    text = str(config.get("hazard_panel_id", "") or "").strip()
    if text:
        return text
    metadata = dict(config.get("data_bundle_metadata") or payload.get("data_bundle_metadata") or {})
    return str(metadata.get("hazard_panel_id", "") or "").strip()


def _build_hazard_panel_mean_guess_check(summary: Mapping[str, Any]) -> Dict[str, Any]:
    direct = dict(summary.get("hazard_panel_mean_guess_check") or {})
    if direct.get("rows"):
        return direct
    recovery = dict(summary.get("supervision_recovery") or summary)
    rows: List[Dict[str, Any]] = []

    def add_candidate(payload: Mapping[str, Any], *, source: str) -> None:
        diag = _hazard_panel_diag_from_mapping(payload)
        if not diag:
            return
        condition_diag = dict(diag.get("condition_diagnostics") or {})
        rows.append(
            {
                "source": str(source),
                "hazard_panel_id": _hazard_panel_id_from_mapping(payload),
                "n_docs": int(_safe_int(diag.get("n_docs"), 0)),
                "n_conditions": int(len(condition_diag)),
                "global_mean_baseline_mae": _safe_float(
                    diag.get("global_mean_baseline_mae"),
                    float("nan"),
                ),
                "condition_mean_baseline_mae": _safe_float(
                    diag.get("condition_mean_baseline_mae"),
                    float("nan"),
                ),
                "mean_guess_gap": _safe_float(diag.get("mean_guess_gap"), float("nan")),
            }
        )

    for idx, row in enumerate(list(recovery.get("family_rows") or [])):
        if isinstance(row, MappingABC):
            add_candidate(dict(row), source=f"family_rows[{idx}]")
    for scope_key, scope in sorted((recovery.get("scopes") or {}).items()):
        if not isinstance(scope, MappingABC):
            continue
        scope_mapping = dict(scope)
        for idx, row in enumerate(list(scope_mapping.get("family_rows") or [])):
            if isinstance(row, MappingABC):
                add_candidate(dict(row), source=f"{scope_key}.family_rows[{idx}]")
        for group in list(scope_mapping.get("rows_by_train_docs") or []):
            if not isinstance(group, MappingABC):
                continue
            group_mapping = dict(group)
            for row in list(group_mapping.get("rows") or []):
                if not isinstance(row, MappingABC):
                    continue
                add_candidate(
                    dict(row),
                    source=(
                        f"{scope_key}.train_docs="
                        f"{_safe_int(group_mapping.get('train_doc_count'), 0)}"
                    ),
                )
    dedup: Dict[tuple[str, str, int], Dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("hazard_panel_id", "")),
            str(row.get("source", "")),
            int(_safe_int(row.get("n_docs"), 0)),
        )
        dedup.setdefault(key, row)
    rows = list(dedup.values())
    rows.sort(key=lambda row: (str(row.get("hazard_panel_id", "")), str(row.get("source", ""))))
    return {
        "status": "ready" if rows else "missing",
        "rows": rows,
    }


def _hazard_panel_mean_guess_lines(summary: Mapping[str, Any]) -> List[str]:
    check = _build_hazard_panel_mean_guess_check(summary)
    rows = [dict(row) for row in list(check.get("rows") or [])]
    if not rows:
        return []
    lines = [
        "| source | panel | docs | conditions | global mean MAE | condition mean MAE | gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:12]:
        lines.append(
            "| "
            f"`{row.get('source', '')}` | "
            f"`{row.get('hazard_panel_id', '')}` | "
            f"{_safe_int(row.get('n_docs'), 0)} | "
            f"{_safe_int(row.get('n_conditions'), 0)} | "
            f"{_format_unavailable(row.get('global_mean_baseline_mae'))} | "
            f"{_format_unavailable(row.get('condition_mean_baseline_mae'))} | "
            f"{_format_unavailable(row.get('mean_guess_gap'))} |"
        )
    return lines


def _render_pdf(summary: Mapping[str, Any], output_path: Path, figure_paths: Mapping[str, str]) -> None:
    batch_timing = summary.get("batch_timing", {}) or {}
    batch_quality = summary.get("batch_quality_tradeoff", {}) or {}
    docs_epochs = summary.get("docs_epochs_grid", {}) or {}
    learnability = summary.get("learnability", {}) or {}
    weight_ablation = summary.get("weight_ablation", {}) or {}
    law_packages = summary.get("law_packages", {}) or {}
    fno_upper_bound = summary.get("fno_upper_bound", {}) or {}
    identifiable_zero = summary.get("identifiable_zero_reference", {}) or {}
    oracle_budget = summary.get("oracle_budget_frontier", {}) or {}
    hazard_lines = _hazard_panel_mean_guess_lines(summary)
    doc_equivalent_frontiers = summary.get("doc_equivalent_frontiers", {}) or {}
    efficiency_target_table = summary.get("efficiency_target_table", {}) or {}
    solved_flags = summary.get("solved_benchmark_flags", {}) or {}
    large_batch = summary.get("large_batch_diagnosis", {}) or {}
    supervision = summary.get("supervision_sweep", {}) or {}
    support = summary.get("support_geometry", {}) or {}
    runtime_efficiency = summary.get("runtime_efficiency", {}) or {}
    medium_doc_label = _format_doc_count(batch_quality.get("train_docs"))
    medium_epochs = int(_safe_int(batch_quality.get("epochs")))

    overview_lines = [f"Generated: {summary.get('generated_at')}", "", "Source status:"]
    for key in sorted(REPORT_SOURCE_SPECS):
        record = dict((summary.get("source_records") or {}).get(key) or {})
        phase = str(record.get("phase", "") or "")
        status = str(record.get("status", "missing") or "missing")
        reason = str(record.get("reason", "") or "").strip()
        line = f"- {key} ({phase}): {status}"
        if reason:
            line += f" — {reason}"
        overview_lines.append(line)

    batch_lines = []
    if _source_is_ready(summary, "batch_timing_summary"):
        batch_lines.extend(
            [
                f"- Best wall-throughput batch: bs={batch_timing.get('best_wall_batch')} at {_safe_float(batch_timing.get('best_wall_docs_per_s')):.1f} docs/s",
                f"- Best train-loop batch: bs={batch_timing.get('best_train_batch')} at {_safe_float(batch_timing.get('best_train_docs_per_s')):.1f} docs/s",
            ]
        )
    else:
        batch_lines.extend(_source_placeholder_lines(summary, "batch_timing_summary"))
    if _source_is_ready(summary, "medium_grid_summary") and str(batch_quality.get("status", "")) == "ready":
        batch_lines.extend(
            [
                f"- Medium-grid quality winner ({medium_doc_label}/{medium_epochs}ep): bs={batch_quality.get('best_quality_batch')} at val MAE {_safe_float(batch_quality.get('best_quality_val_mae')):.6f}",
                f"- Medium-grid throughput winner ({medium_doc_label}/{medium_epochs}ep): bs={batch_quality.get('best_speed_batch')} at {_safe_float(batch_quality.get('best_speed_docs_per_s')):.1f} docs/s",
                f"- Medium-grid balanced point: bs={batch_quality.get('best_balanced_batch')}",
            ]
        )
    else:
        batch_lines.extend([""] + _source_placeholder_lines(summary, "medium_grid_summary"))

    docs_epoch_lines = []
    if _source_is_ready(summary, "docs_epochs_summary"):
        for train_docs, row in sorted((docs_epochs.get("by_train_docs") or {}).items(), key=lambda item: int(item[0])):
            docs_epoch_lines.append(
                f"- train_docs={train_docs}: best val at ep={_safe_int(row.get('best_val_epochs'))} with MAE {_safe_float(row.get('best_val_mae')):.6f}; fastest at ep={_safe_int(row.get('fastest_epochs'))} with {_safe_float(row.get('fastest_docs_per_s')):.1f} docs/s"
            )
    if not docs_epoch_lines:
        docs_epoch_lines = _source_placeholder_lines(summary, "docs_epochs_summary")

    learnability_lines = []
    if _source_is_ready(summary, "learnability_summary"):
        for train_docs, row in sorted((learnability.get("by_train_docs") or {}).items(), key=lambda item: int(item[0])):
            best_root = row.get("best_by_root") or {}
            best_obj = row.get("best_by_objective") or {}
            learnability_lines.append(
                f"- train_docs={train_docs}: primary best-by-root uses llw={_safe_float(best_root.get('local_law_weight')):.3g}, C=({_safe_float(best_root.get('objective_local_law_c1_weight')):.3g}, {_safe_float(best_root.get('objective_local_law_c2_weight')):.3g}, {_safe_float(best_root.get('objective_local_law_c3_weight')):.3g}), root gain={_format_pct_or_floor(row.get('root_gain_pct_vs_baseline'))}"
            )
            learnability_lines.append(
                f"- train_docs={train_docs}: diagnostic best-by-objective uses llw={_safe_float(best_obj.get('local_law_weight')):.3g}, C=({_safe_float(best_obj.get('objective_local_law_c1_weight')):.3g}, {_safe_float(best_obj.get('objective_local_law_c2_weight')):.3g}, {_safe_float(best_obj.get('objective_local_law_c3_weight')):.3g}), objective gain={_format_pct_or_floor(row.get('objective_gain_pct_vs_baseline'))}"
            )
    else:
        learnability_lines = _source_placeholder_lines(summary, "learnability_summary")
    if _source_is_ready(summary, "weight_ablation_summary") and weight_ablation.get("best_profile_by_gain"):
        best = weight_ablation.get("best_profile_by_root_error") or weight_ablation["best_profile_by_gain"]
        learnability_lines.append(
            f"- Weight ablation root-error winner: {best.get('profile')} with mean root error {_safe_float(best.get('mean_root_error')):.6f}"
        )
        diagnostic = weight_ablation.get("best_profile_by_gain") or {}
        if diagnostic:
            learnability_lines.append(
                f"- Diagnostic best-by-gain profile: {diagnostic.get('profile')} with mean gain {_format_pct_or_floor(diagnostic.get('mean_gain_pct'))}"
            )
    elif not _source_is_ready(summary, "weight_ablation_summary"):
        learnability_lines.extend([""] + _source_placeholder_lines(summary, "weight_ablation_summary"))

    supervision_lines = []
    if _source_is_ready(summary, "supervision_sweep_summary") and str(supervision.get("status", "")) == "ready":
        best_overall_supervision = dict(supervision.get("best_overall") or {})
        if best_overall_supervision:
            supervision_lines.append(
                f"- Best overall row: train_docs={_safe_int(best_overall_supervision.get('train_doc_count'))}, "
                f"leaf={best_overall_supervision.get('leaf_profile')} ({best_overall_supervision.get('leaf_supervision_kind')} @ {_safe_float(best_overall_supervision.get('leaf_label_rate')):.2g}), "
                f"internal={best_overall_supervision.get('internal_profile')} ({best_overall_supervision.get('internal_supervision_kind')} @ {_safe_float(best_overall_supervision.get('internal_label_rate')):.2g}), "
                f"test root MAE={_safe_float(best_overall_supervision.get('mean_test_root_mae')):.6f}"
            )
        for train_docs, payload in sorted((supervision.get("by_train_docs") or {}).items(), key=lambda item: int(item[0])):
            best_row = dict(payload.get("best_root_row") or {})
            if not best_row:
                continue
            gain_pct = _safe_float(payload.get("gain_pct_vs_none_none"), float("nan"))
            gain_text = f"{gain_pct:.1f}%" if math.isfinite(gain_pct) else "unavailable"
            supervision_lines.append(
                f"- train_docs={train_docs}: best leaf={best_row.get('leaf_profile')} ({best_row.get('leaf_supervision_kind')} @ {_safe_float(best_row.get('leaf_label_rate')):.2g}), "
                f"best internal={best_row.get('internal_profile')} ({best_row.get('internal_supervision_kind')} @ {_safe_float(best_row.get('internal_label_rate')):.2g}), "
                f"root MAE={_safe_float(best_row.get('mean_test_root_mae')):.6f}, gain vs none/none={gain_text}"
            )
    if not supervision_lines:
        supervision_lines = _source_placeholder_lines(summary, "supervision_sweep_summary")

    geometry_lines = []
    if _source_is_ready(summary, "support_summary") and str(support.get("status", "")) == "ready":
        for row in list(support.get("rows", []) or []):
            geometry_lines.append(
                f"- fixed_leaf_tokens={_safe_int(row.get('fixed_leaf_tokens'))}: median leaves={_safe_float(row.get('median_mean_leaves')):.1f}, supported best root MAE={_safe_float(row.get('best_supported_root_mae')):.6g}, unsupported median root MAE={_safe_float(row.get('median_unsupported_root_mae')):.6g}"
            )
    if not geometry_lines:
        geometry_lines = _source_placeholder_lines(summary, "support_summary")

    if str(runtime_efficiency.get("status", "")) == "ready":
        runtime_lines = [
            f"- data_mode={runtime_efficiency.get('runtime_data_mode', '')}, bucket_mode={runtime_efficiency.get('runtime_bucket_mode', '')}, workers/MIG≈{_safe_float(runtime_efficiency.get('runtime_workers_per_mig_mean'), 0.0):.2f}",
            (
                f"- mean resident-store build time={_safe_float(runtime_efficiency.get('resident_store_build_time_s_mean'), 0.0):.4f}s; "
                f"mean steady-state H2D={_safe_float(runtime_efficiency.get('steady_state_h2d_bytes_mean'), 0.0):.1f} bytes "
                f"in {_safe_float(runtime_efficiency.get('steady_state_h2d_time_s_mean'), 0.0):.6f}s"
            ),
            f"- resident-store hits={_safe_int(runtime_efficiency.get('resident_store_hits_total'))}, misses={_safe_int(runtime_efficiency.get('resident_store_misses_total'))}",
        ]
    else:
        runtime_lines = _source_placeholder_lines(summary, "batch_timing_summary")
    cpu_fallback_reasons = dict(runtime_efficiency.get("cpu_fallback_reason_counts") or {})
    if cpu_fallback_reasons:
        runtime_lines.append(
            "- CPU fallback reasons: "
            + ", ".join(
                f"{key}={int(value)}"
                for key, value in sorted(cpu_fallback_reasons.items())
            )
        )

    law_lines = []
    if _source_is_ready(summary, "law_comparison_json"):
        if law_packages.get("root_surface_saturated"):
            law_lines.append("- Root surfaces are at the solved floor for both tree and doc reference on this benchmark; leaf/merge diagnostics remain the informative comparison.")
        for row in list(law_packages.get("rows", []) or []):
            law_lines.append(
                f"- {row.get('name')}: tree root {_format_unavailable(row.get('test_root_mae'))}, {law_packages.get('doc_fno_label', 'doc FNO reference')} root {_format_unavailable(row.get('doc_fno_test_root_mae'))}, gap {_format_unavailable(row.get('tree_vs_doc_fno_root_mae_gap'))}, leaf {_format_unavailable(row.get('test_leaf_mae'))}, merge {_format_unavailable(row.get('test_merge_mae'))}, wall {_format_unavailable(row.get('wall_seconds'), fmt='.1f')}s"
            )
    else:
        law_lines = _source_placeholder_lines(summary, "law_comparison_json")

    reference_lines = []
    if summary.get("fno_upper_bound"):
        reference_lines.append(
            f"- canonical identifiable-zero reference: {str(identifiable_zero.get('reference_label', ''))} using families {', '.join(str(item) for item in identifiable_zero.get('families', []))}"
        )
        for row in list(fno_upper_bound.get("rows", []) or []):
            gap_ratio = _safe_float(row.get("gap_ratio_vs_best_fno"), float("nan"))
            gap_text = f"{100.0 * gap_ratio:.1f}%" if math.isfinite(gap_ratio) else "unavailable"
            reference_lines.append(
                f"- train_docs={_safe_int(row.get('train_docs'))}: best full-doc FNO={str(row.get('best_full_doc_fno_family', ''))} at {_format_metric_or_floor(row.get('best_full_doc_fno_test_root_mae'))}; best tree point={_format_metric_or_floor(row.get('best_tree_test_root_mae'))}; gap={gap_text}"
            )
    else:
        reference_lines = _source_placeholder_lines(summary, "fno_upper_bound_summary")
    if large_batch.get("recommendation"):
        recommendation = dict(large_batch.get("recommendation") or {})
        reference_lines.extend(
            [
                "",
                f"- classification: {str(large_batch.get('classification', ''))}",
                f"- recommendation: cap at bs={_safe_int(recommendation.get('recommended_max_batch_size'))} ({str(recommendation.get('reason', ''))})",
            ]
        )
    elif not _source_is_ready(summary, "large_batch_diagnosis_summary"):
        reference_lines.extend([""] + _source_placeholder_lines(summary, "large_batch_diagnosis_summary"))

    oracle_budget_lines = []
    if summary.get("oracle_budget_frontier"):
        best_efficiency = dict(oracle_budget.get("best_efficiency_point") or {})
        if str(oracle_budget.get("scope_label", "")).strip():
            oracle_budget_lines.append(f"- Primary efficiency scope: {oracle_budget.get('scope_label')}")
        if best_efficiency:
            oracle_budget_lines.append(
                f"- Best efficiency point: calls/doc={_safe_float(best_efficiency.get('budget_total_calls_per_doc')):.2g}, full-doc share={_safe_float(best_efficiency.get('full_doc_budget_share')):.2g}, reviewed-equiv docs={_safe_float(best_efficiency.get('reviewed_full_doc_equivalent_docs')):.1f}, FNO-equiv docs={_format_equivalent_docs(best_efficiency, key='fno_equivalent_train_docs')}, tree-equiv docs={_format_equivalent_docs(best_efficiency, key='tree_equivalent_train_docs')}, review fraction={100.0 * _safe_float(best_efficiency.get('review_fraction_of_training_corpus'), 0.0):.0f}%"
            )
    else:
        oracle_budget_lines = _source_placeholder_lines(summary, "oracle_budget_frontier_summary")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        write_text_page(pdf, title="Markov Optimization Tradeoff Report", lines=overview_lines)
        write_text_page(pdf, title="Batch Throughput", lines=batch_lines)
        write_text_page(pdf, title="Training Data And Epochs", lines=docs_epoch_lines)
        write_text_page(pdf, title="Local Laws", lines=learnability_lines)
        write_text_page(pdf, title="Law Packages", lines=law_lines)
        write_text_page(pdf, title="Tree Geometry And Support", lines=geometry_lines)
        write_text_page(pdf, title="Supervision Level Sweep", lines=supervision_lines)
        write_text_page(pdf, title="Runtime Efficiency", lines=runtime_lines)
        if hazard_lines:
            write_text_page(pdf, title="Hazard Panel Mean-Guess Check", lines=hazard_lines)
        write_text_page(pdf, title="FNO Upper Bound And Large Batches", lines=reference_lines)
        write_text_page(pdf, title="Oracle Budget Frontier", lines=oracle_budget_lines)
        for title, path in figure_paths.items():
            write_image_page(pdf, image_path=Path(path), title=title)


def _write_markdown(summary: Mapping[str, Any], output_path: Path) -> None:
    lines: List[str] = ["# Markov Optimization Tradeoff Report", "", f"Generated: `{summary['generated_at']}`", ""]
    if summary.get("path_contract"):
        contract = dict(summary.get("path_contract") or {})
        lines.extend(
            [
                "## Current Path Contract",
                f"- Canonical identifiable-zero reference kind: `{contract.get('canonical_identifiable_zero_reference_kind')}`.",
                f"- Full-doc FNO families: `{', '.join(str(item) for item in contract.get('canonical_full_doc_fno_families', []))}`.",
                f"- Full-doc FNO training entrypoint: `{contract.get('full_doc_fno_training_entrypoint')}`.",
                f"- Full-doc FNO training backend: `{contract.get('full_doc_fno_training_backend')}`.",
                f"- Tree training entrypoint: `{contract.get('tree_training_entrypoint')}`.",
                "",
            ]
        )
    lines.append("## Sources")
    for key in sorted(REPORT_SOURCE_SPECS):
        record = dict((summary.get("source_records") or {}).get(key) or {})
        path = str(record.get("path", "") or "")
        status = str(record.get("status", "missing") or "missing")
        reason = str(record.get("reason", "") or "").strip()
        suffix = f" ({status})"
        if reason:
            suffix += f" — {reason}"
        lines.append(f"- `{key}`: `{path}`{suffix}.")
    if summary.get("pdf"):
        lines.append(f"- `pdf`: `{summary['pdf']}`")
    lines.append("")

    hazard_lines = _hazard_panel_mean_guess_lines(summary)
    if hazard_lines:
        lines.append("## Hazard Panel / Mean-Guess Check")
        lines.extend(hazard_lines)
        lines.append("")

    batch_timing = summary.get("batch_timing", {}) or {}
    batch_quality = summary.get("batch_quality_tradeoff", {}) or {}
    docs_epochs = summary.get("docs_epochs_grid", {}) or {}
    learnability = summary.get("learnability", {}) or {}
    weight_ablation = summary.get("weight_ablation", {}) or {}
    law_packages = summary.get("law_packages", {}) or {}
    fno_upper_bound = summary.get("fno_upper_bound", {}) or {}
    identifiable_zero = summary.get("identifiable_zero_reference", {}) or {}
    oracle_budget = summary.get("oracle_budget_frontier", {}) or {}
    doc_equivalent_frontiers = summary.get("doc_equivalent_frontiers", {}) or {}
    efficiency_target_table = summary.get("efficiency_target_table", {}) or {}
    solved_flags = summary.get("solved_benchmark_flags", {}) or {}
    large_batch = summary.get("large_batch_diagnosis", {}) or {}
    supervision = summary.get("supervision_sweep", {}) or {}
    support = summary.get("support_geometry", {}) or {}
    runtime_efficiency = summary.get("runtime_efficiency", {}) or {}
    medium_doc_label = _format_doc_count(batch_quality.get("train_docs"))
    medium_epochs = int(_safe_int(batch_quality.get("epochs")))

    lines.append("## Runtime Efficiency")
    if str(runtime_efficiency.get("status", "")) == "ready":
        lines.append(
            f"- Current runtime path: data mode `{runtime_efficiency.get('runtime_data_mode')}`, bucket mode `{runtime_efficiency.get('runtime_bucket_mode')}`, mean workers/MIG `{_safe_float(runtime_efficiency.get('runtime_workers_per_mig_mean'), 0.0):.2f}`."
        )
        lines.append(
            f"- Mean resident-store build time is `{_safe_float(runtime_efficiency.get('resident_store_build_time_s_mean'), 0.0):.6f}s`; mean steady-state H2D is `{_safe_float(runtime_efficiency.get('steady_state_h2d_bytes_mean'), 0.0):.1f}` bytes in `{_safe_float(runtime_efficiency.get('steady_state_h2d_time_s_mean'), 0.0):.6f}s`."
        )
        lines.append(
            f"- Resident-store hits/misses: `{_safe_int(runtime_efficiency.get('resident_store_hits_total'))}` / `{_safe_int(runtime_efficiency.get('resident_store_misses_total'))}`."
        )
    else:
        for item in _source_placeholder_lines(summary, "batch_timing_summary"):
            lines.append(f"- {item}.")
    lines.append("")

    lines.append("## Batch Throughput")
    if _source_is_ready(summary, "batch_timing_summary"):
        lines.append(
            f"- Best wall-throughput batch from the refreshed leaf-laws sweep: `bs={batch_timing.get('best_wall_batch')}` at `{_safe_float(batch_timing.get('best_wall_docs_per_s')):.1f} docs/s`."
        )
        lines.append(
            f"- Best train-loop batch: `bs={batch_timing.get('best_train_batch')}` at `{_safe_float(batch_timing.get('best_train_docs_per_s')):.1f} docs/s`."
        )
    else:
        for item in _source_placeholder_lines(summary, "batch_timing_summary"):
            lines.append(f"- {item}.")
    if _source_is_ready(summary, "medium_grid_summary") and str(batch_quality.get("status", "")) == "ready":
        lines.append(
            f"- Best quality in the medium `{medium_doc_label}/{medium_epochs}ep` grid: `bs={batch_quality.get('best_quality_batch')}` with mean best val MAE `{_safe_float(batch_quality.get('best_quality_val_mae')):.6f}`."
        )
        lines.append(
            f"- Best throughput in the medium `{medium_doc_label}/{medium_epochs}ep` grid: `bs={batch_quality.get('best_speed_batch')}` at `{_safe_float(batch_quality.get('best_speed_docs_per_s')):.1f} docs/s`."
        )
        lines.append(
            f"- Best balanced batch by joint speed/quality rank: `bs={batch_quality.get('best_balanced_batch')}`."
        )
    else:
        for item in _source_placeholder_lines(summary, "medium_grid_summary"):
            lines.append(f"- {item}.")
    lines.append("")

    lines.append("## Training Data And Epochs")
    if _source_is_ready(summary, "docs_epochs_summary"):
        for train_docs, row in sorted((docs_epochs.get("by_train_docs") or {}).items(), key=lambda item: int(item[0])):
            lines.append(
                f"- `train_docs={train_docs}`: fastest run is `{row.get('fastest_run')}` at `{_safe_float(row.get('fastest_docs_per_s')):.1f} docs/s`; best val run is `{row.get('best_val_run')}` at epoch `{_safe_int(row.get('best_val_epochs'))}` with val MAE `{_safe_float(row.get('best_val_mae')):.6f}`."
            )
    else:
        for item in _source_placeholder_lines(summary, "docs_epochs_summary"):
            lines.append(f"- {item}.")
    lines.append("")

    lines.append("## Local Law Weights")
    if _source_is_ready(summary, "learnability_summary"):
        for train_docs, row in sorted((learnability.get("by_train_docs") or {}).items(), key=lambda item: int(item[0])):
            best_root = row.get("best_by_root") or {}
            best_obj = row.get("best_by_objective") or {}
            lines.append(
                f"- `train_docs={train_docs}`: primary best-by-root uses local-law weight `{_safe_float(best_root.get('local_law_weight')):.3g}` with C-weights `({_safe_float(best_root.get('objective_local_law_c1_weight')):.3g}, {_safe_float(best_root.get('objective_local_law_c2_weight')):.3g}, {_safe_float(best_root.get('objective_local_law_c3_weight')):.3g})`; root-MAE gain vs no-local-law baseline is `{_format_pct_or_floor(row.get('root_gain_pct_vs_baseline'))}`."
            )
            lines.append(
                f"- `train_docs={train_docs}`: diagnostic best-by-objective uses local-law weight `{_safe_float(best_obj.get('local_law_weight')):.3g}` with C-weights `({_safe_float(best_obj.get('objective_local_law_c1_weight')):.3g}, {_safe_float(best_obj.get('objective_local_law_c2_weight')):.3g}, {_safe_float(best_obj.get('objective_local_law_c3_weight')):.3g})`; objective gain vs no-local-law baseline is `{_format_pct_or_floor(row.get('objective_gain_pct_vs_baseline'))}`."
            )
    else:
        for item in _source_placeholder_lines(summary, "learnability_summary"):
            lines.append(f"- {item}.")
    best_profile = weight_ablation.get("best_profile_by_root_error") or {}
    if best_profile:
        lines.append(
            f"- Best weight profile in the ablation summary by root error is `{best_profile.get('profile')}` with mean root error `{_safe_float(best_profile.get('mean_root_error')):.6f}`."
        )
    elif not _source_is_ready(summary, "weight_ablation_summary"):
        for item in _source_placeholder_lines(summary, "weight_ablation_summary"):
            lines.append(f"- {item}.")
    best_gain_profile = weight_ablation.get("best_profile_by_gain") or {}
    if best_gain_profile:
        lines.append(
            f"- Diagnostic best-by-gain profile is `{best_gain_profile.get('profile')}` with mean gain `{_format_pct_or_floor(best_gain_profile.get('mean_gain_pct'))}` and pass rate `{100.0 * _safe_float(best_gain_profile.get('primary_pass_rate')):.0f}%`."
        )
    lines.append("")

    lines.append("## Law Packages")
    if _source_is_ready(summary, "law_comparison_json"):
        if law_packages.get("root_surface_saturated"):
            lines.append("- Root surfaces are at the solved floor for both tree and doc reference on this benchmark; leaf/merge diagnostics remain the informative comparison.")
        for row in list(law_packages.get("rows", []) or []):
            lines.append(
                f"- `{row.get('name')}`: tree root `{_format_unavailable(row.get('test_root_mae'))}`, {law_packages.get('doc_fno_label', 'doc FNO reference')} root `{_format_unavailable(row.get('doc_fno_test_root_mae'))}`, gap `{_format_unavailable(row.get('tree_vs_doc_fno_root_mae_gap'))}`, leaf `{_format_unavailable(row.get('test_leaf_mae'))}`, merge `{_format_unavailable(row.get('test_merge_mae'))}`, wall `{_format_unavailable(row.get('wall_seconds'), fmt='.1f')}s`."
            )
    else:
        for item in _source_placeholder_lines(summary, "law_comparison_json"):
            lines.append(f"- {item}.")
    lines.append("")

    lines.append("## Identifiable-Zero Reference")
    if identifiable_zero:
        lines.append(f"- Canonical reference: `{identifiable_zero.get('reference_label')}`.")
        lines.append(
            f"- Best overall family in this report: `{identifiable_zero.get('best_family_overall')}` at `train_docs={_safe_int(identifiable_zero.get('best_train_docs'))}` with root MAE `{_safe_float(identifiable_zero.get('best_test_root_mae')):.6f}`."
        )
    else:
        for item in _source_placeholder_lines(summary, "fno_upper_bound_summary"):
            lines.append(f"- {item}.")
    lines.append("")

    lines.append("## Full-Doc FNO Upper Bound")
    if summary.get("fno_upper_bound"):
        for row in list(fno_upper_bound.get("rows", []) or []):
            gap_ratio = _safe_float(row.get("gap_ratio_vs_best_fno"), float("nan"))
            gap_text = f"{100.0 * gap_ratio:.1f}%" if math.isfinite(gap_ratio) else "unavailable"
            lines.append(
                f"- `train_docs={_safe_int(row.get('train_docs'))}`: best full-doc FNO is `{row.get('best_full_doc_fno_family')}` at `{_format_metric_or_floor(row.get('best_full_doc_fno_test_root_mae'))}`; best tree point is `{_format_metric_or_floor(row.get('best_tree_test_root_mae'))}` from `{row.get('best_tree_source')}`; gap vs FNO is `{gap_text}`."
            )
    else:
        for item in _source_placeholder_lines(summary, "fno_upper_bound_summary"):
            lines.append(f"- {item}.")
    lines.append("")

    lines.append("## Oracle Budget Frontier")
    if oracle_budget:
        if oracle_budget.get("scope_label"):
            lines.append(f"- Primary efficiency scope: `{oracle_budget.get('scope_label')}`.")
        lines.append(
            f"- Train-doc count for this efficiency study: `{_format_doc_count(oracle_budget.get('train_doc_count'))}`."
        )
    else:
        for item in _source_placeholder_lines(summary, "oracle_budget_frontier_summary"):
            lines.append(f"- {item}.")
    lines.append("")

    if doc_equivalent_frontiers:
        lines.append("## Label-Efficiency Surfaces")
        lines.append(
            f"- Primary plotted efficiency surface: `{doc_equivalent_frontiers.get('primary_scope')}`."
        )
        for scope_name, flags in sorted((solved_flags or {}).items()):
            if flags.get("fno_anchor_saturated"):
                if scope_name in SUPERVISION_RECOVERY_RECOVERABLE_SCOPE_FAMILY:
                    lines.append(
                        f"- `{scope_name}`: FNO anchor already solved on this recoverable grid."
                    )
                else:
                    lines.append(f"- `{scope_name}`: FNO anchor already solved on this grid.")
            if flags.get("exact_witness_exact"):
                lines.append(f"- `{scope_name}`: `palette_block_exact` remains exact.")
        primary_scope = str(doc_equivalent_frontiers.get("primary_scope", "") or "")
        primary_table = dict((efficiency_target_table.get("scopes") or {}).get(primary_scope) or {})
        for row in list(primary_table.get("rows") or [])[:5]:
            lines.append(
                f"- target MAE `{_safe_float(row.get('target_mae')):.4g}`: reviewed docs `{_format_doc_count(int(round(_safe_float(row.get('reviewed_docs_needed'), 0.0)))) if math.isfinite(_safe_float(row.get('reviewed_docs_needed'), float('nan'))) else 'n/a'}`, FNO-equiv `{_format_equivalent_docs(row, key='fno_equivalent_train_docs')}`, tree-equiv `{_format_equivalent_docs(row, key='tree_equivalent_train_docs')}`."
            )
        lines.append("")

    lines.append("## Large-Batch Diagnosis")
    if large_batch:
        lines.append(f"- Classification: `{large_batch.get('classification')}`.")
        recommendation = dict(large_batch.get("recommendation") or {})
        if recommendation:
            lines.append(
                f"- Recommended max training batch is `bs={_safe_int(recommendation.get('recommended_max_batch_size'))}` because {recommendation.get('reason')}."
            )
    else:
        for item in _source_placeholder_lines(summary, "large_batch_diagnosis_summary"):
            lines.append(f"- {item}.")
    lines.append("")

    lines.append("## Supervision Level Sweep")
    if _source_is_ready(summary, "supervision_sweep_summary") and str(supervision.get("status", "")) == "ready":
        best_overall_supervision = dict(supervision.get("best_overall") or {})
        if best_overall_supervision:
            lines.append(
                f"- Best overall supervision row: `train_docs={_safe_int(best_overall_supervision.get('train_doc_count'))}` with leaf `{best_overall_supervision.get('leaf_profile')}` (`{best_overall_supervision.get('leaf_supervision_kind')}` @ `{_safe_float(best_overall_supervision.get('leaf_label_rate')):.3g}`) and internal `{best_overall_supervision.get('internal_profile')}` (`{best_overall_supervision.get('internal_supervision_kind')}` @ `{_safe_float(best_overall_supervision.get('internal_label_rate')):.3g}`), giving mean test root MAE `{_safe_float(best_overall_supervision.get('mean_test_root_mae')):.6f}`."
            )
        for train_docs, payload in sorted((supervision.get("by_train_docs") or {}).items(), key=lambda item: int(item[0])):
            best_row = dict(payload.get("best_root_row") or {})
            if not best_row:
                continue
            lines.append(
                f"- `train_docs={train_docs}`: best leaf profile is `{best_row.get('leaf_profile')}` (`{best_row.get('leaf_supervision_kind')}` @ `{_safe_float(best_row.get('leaf_label_rate')):.3g}`), best internal profile is `{best_row.get('internal_profile')}` (`{best_row.get('internal_supervision_kind')}` @ `{_safe_float(best_row.get('internal_label_rate')):.3g}`), mean test root MAE `{_safe_float(best_row.get('mean_test_root_mae')):.6f}`, gain vs `none/none` `{_format_pct_or_floor(payload.get('gain_pct_vs_none_none'), note='unavailable')}`."
            )
    else:
        for item in _source_placeholder_lines(summary, "supervision_sweep_summary"):
            lines.append(f"- {item}.")
    lines.append("")

    lines.append("## Tree Geometry And Support")
    if _source_is_ready(summary, "support_summary") and str(support.get("status", "")) == "ready":
        for row in list(support.get("rows", []) or []):
            lines.append(
                f"- `fixed_leaf_tokens={_safe_int(row.get('fixed_leaf_tokens'))}`: median tree `{_safe_float(row.get('median_mean_leaves')):.1f}` leaves / `{_safe_float(row.get('median_mean_internal_nodes')):.1f}` internal nodes; best supported root MAE `{_safe_float(row.get('best_supported_root_mae')):.6g}`, median unsupported root MAE `{_safe_float(row.get('median_unsupported_root_mae')):.6g}`."
            )
    else:
        for item in _source_placeholder_lines(summary, "support_summary"):
            lines.append(f"- {item}.")
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _focused_overview_lines(summary: Mapping[str, Any]) -> List[str]:
    if _is_r10_coverage_focused(summary):
        lines = [
            f"Generated: {summary.get('generated_at')}",
            "",
            "This report focuses on one question: what extra tree labels help when the root budget stays at `R10`?",
            "- Read the plots in order: full supervision first, then root-budget sweeps, then fixed-budget tree-vs-FNO comparisons.",
            "- `Recoverable` is the easier benchmark where local structure should be enough. `Structural` is the harder benchmark.",
            "- Appendix sections are secondary checks, not the main story.",
            "",
            "Focused source status:",
        ]
    else:
        lines = [
            f"Generated: {summary.get('generated_at')}",
            "",
            "This report is a short supervision-recovery walkthrough.",
            "- Start with the full-supervision baseline. That is the easiest check and tells you whether the tree can match the dense FNO baselines at all.",
            "- Then read the recoverable plots, then the structural plots. The structural benchmark is the harder one.",
            "- Use the appendix-style sections only after the main plots: they explain details, but they are not the core comparison.",
            "",
            "Focused source status:",
        ]
    for key in sorted(REPORT_SOURCE_SPECS):
        record = dict((summary.get("source_records") or {}).get(key) or {})
        status = str(record.get("status", "missing") or "missing")
        if key != "supervision_recovery_summary" and status == "missing":
            continue
        phase = str(record.get("phase", "") or "")
        reason = str(record.get("reason", "") or "").strip()
        line = f"- {key} ({phase}): {status}"
        if reason:
            line += f" — {reason}"
        lines.append(line)
    return lines


def _focused_key_concept_lines() -> List[str]:
    return [
        "- `training docs` means the total number of documents used for training.",
        "- `root-labeled docs` means training documents that receive document/root supervision.",
        "- `training-doc equivalents` means the total supervision budget measured in full-document-equivalent units.",
        "- `fixed_leaf_tokens` is the token budget per tree leaf. `leaves/doc` is the resulting number of leaves in each 128-token document.",
        "- `1 leaf/doc` is the only geometry that is directly equivalent to the full-document FNO problem.",
        "- `recoverable` is the easier benchmark where local structure should be enough. `structural` is the harder benchmark where local structure alone may not be enough.",
        "- Example: at `train_docs=10240`, `R10` means about `1024` root-labeled docs. At `R100`, the budget is `10240` training-doc equivalents.",
    ]


def _exact_full_doc_canary_overview_lines(summary: Mapping[str, Any]) -> List[str]:
    return [
        f"Generated: {summary.get('generated_at')}",
        "",
        "This report covers the exact full-doc parity canary only.",
        "- Every plotted row is `full100`, `1 leaf/doc`, and `parity_mode=exact_full_doc`.",
        "- `official_fno` is the canonical parity comparator. `official_fno_sumlen` is shown separately as a different FNO family, not the parity contract.",
        "- The goal is simple: verify the tree can reproduce the one-leaf full-document problem before interpreting deeper-tree results.",
    ]


def _exact_full_doc_canary_protocol_lines(summary: Mapping[str, Any]) -> List[str]:
    recovery = dict(summary.get("supervision_recovery") or {})
    train_doc_counts = _focused_train_doc_counts(recovery)
    recoverable_scope = _recoverable_scope_key(recovery)
    recoverable_label = _scope_label_from_recovery(recovery, recoverable_scope)
    structural_scope = _structural_scope_key(recovery)
    structural_label = _scope_label_from_recovery(recovery, structural_scope)
    lines = [
        f"- Tree model: `{recovery.get('tree_family', SUPERVISION_RECOVERY_TREE_FAMILY)}`.",
        "- FNO families shown: `official_fno`, `official_fno_sumlen`.",
        f"- Benchmarks: `{recoverable_label}` and `{structural_label}`.",
        "- Package set: `full100` only.",
        "- Train docs: `"
        + ", ".join(str(int(_safe_int(value))) for value in train_doc_counts)
        + "`.",
        "- Headline plots use those canonical train-doc counts. Extra exploratory counts from intermediate sweeps stay in the bundle histories instead of being overplotted in the main figures.",
        "- Seeds: `"
        + ", ".join(str(int(_safe_int(value))) for value in recovery.get("seeds", []))
        + f"` (n=`{_safe_int(recovery.get('seed_count'))}`).",
        "- Executed geometry contract: exact full-doc parity at `1 leaf/doc`.",
    ]
    observed_leaf_tokens = sorted(
        {
            int(_safe_int(row.get("fixed_leaf_tokens"), 0))
            for row in list(recovery.get("family_rows") or [])
            if str(dict(row or {}).get("baseline_family", "") or "")
            == str(recovery.get("tree_family", SUPERVISION_RECOVERY_TREE_FAMILY))
            and int(_safe_int(dict(row or {}).get("fixed_leaf_tokens"), 0)) > 0
        }
    )
    if observed_leaf_tokens:
        lines.append(
            "- Observed tree geometry: `"
            + ", ".join(str(value) for value in observed_leaf_tokens)
            + "` fixed leaf tokens."
        )
    return lines


def _exact_full_doc_canary_scope_lines(
    summary: Mapping[str, Any],
    *,
    scope_key: str,
) -> List[str]:
    recovery = dict(summary.get("supervision_recovery") or {})
    rows = _exact_full_doc_canary_scope_rows(recovery, scope_key=scope_key)
    if not rows:
        return [f"- No exact full-doc canary rows were available for `{scope_key}`."]
    scope_label = str(rows[0].get("scope_label", scope_key) or scope_key)
    leaf_tokens = sorted(
        {
            int(_safe_int(row.get("effective_fixed_leaf_tokens"), 0))
            for row in rows
            if int(_safe_int(row.get("effective_fixed_leaf_tokens"), 0)) > 0
        }
    )
    leaves_per_doc = sorted(
        {
            int(_safe_int(row.get("effective_leaves_per_doc"), 0))
            for row in rows
            if int(_safe_int(row.get("effective_leaves_per_doc"), 0)) > 0
        }
    )
    lines = [
        f"- Scope: `{scope_label}`.",
        "- X-axis: train docs under full root/doc supervision.",
        "- The tree is parity-valid here because every row executes at `1 leaf/doc` and is marked exact full-doc parity.",
        "- Read `official_fno` as the canonical parity target. `official_fno_sumlen` is a separate family ceiling.",
    ]
    if leaf_tokens:
        lines.append(
            "- Executed fixed leaf tokens: `"
            + ", ".join(str(value) for value in leaf_tokens)
            + "`."
        )
    if leaves_per_doc:
        lines.append(
            "- Executed leaves/doc: `"
            + ", ".join(str(value) for value in leaves_per_doc)
            + "`."
        )
    return lines


def _visible_supervision_title(title: str) -> str:
    suffix = ""
    base_title = str(title)
    if base_title.endswith(")") and " (" in base_title:
        prefix, candidate_suffix = base_title.rsplit(" (", 1)
        candidate_suffix = candidate_suffix.rstrip(")")
        if candidate_suffix:
            base_title = prefix
            suffix = f" ({candidate_suffix})"
    direct_map = {
        "Dense Full-Doc Anchor": "Full-Supervision Baseline",
        "Recoverable Ordered Families": "Recoverable Root-Supervision Sweep",
        "Structural Ordered Families": "Structural Root-Supervision Sweep",
        "Recoverable Dense-Local Root Ladder": "Recoverable Full Local Supervision + Root Sweep",
        "Structural Dense-Local Root Ladder": "Structural Full Local Supervision + Root Sweep",
        "Recoverable R10 Local Ablations": "Recoverable What Extra Tree Labels Help at R10?",
        "Structural R10 Local Ablations": "Structural What Extra Tree Labels Help at R10?",
        "Recoverable Mass-Matched Overlay": "Recoverable Fixed-Budget Comparison Across Budgets",
        "Structural Mass-Matched Overlay": "Structural Fixed-Budget Comparison Across Budgets",
        "Recoverable R10 Leaf Geometry": "Recoverable Leaves/Doc at R10",
        "Recoverable R20 Leaf Geometry": "Recoverable Leaves/Doc at R20",
        "Recoverable R80 Leaf Geometry": "Recoverable Leaves/Doc at R80",
        "Recoverable R90 Leaf Geometry": "Recoverable Leaves/Doc at R90",
        "Recoverable R100 Leaf Geometry": "Recoverable Leaves/Doc at R100",
        "Structural R10 Leaf Geometry": "Structural Leaves/Doc at R10",
        "Structural R20 Leaf Geometry": "Structural Leaves/Doc at R20",
        "Structural R80 Leaf Geometry": "Structural Leaves/Doc at R80",
        "Structural R90 Leaf Geometry": "Structural Leaves/Doc at R90",
        "Structural R100 Leaf Geometry": "Structural Leaves/Doc at R100",
        "Recoverable Package Ladder": "Recoverable All Supervision Settings",
        "Structural Package Ladder": "Structural All Supervision Settings",
        "Recoverable Tree Diagnostics": "Recoverable Tree-Only Diagnostics",
        "Structural Tree Diagnostics": "Structural Tree-Only Diagnostics",
        "How To Read Colors": "Key Concepts",
        "Guided Walkthrough": "How To Read The Plots",
        "Protocol And Setup": "Setup",
        "Recoverable R100 Full-Root Reference": "Recoverable Full-Supervision Reference",
        "Structural R100 Full-Root Reference": "Structural Full-Supervision Reference",
        "Recoverable R10 Endpoint Callout": "Recoverable R10 Endpoints",
        "Structural R10 Endpoint Callout": "Structural R10 Endpoints",
    }
    if base_title in direct_map:
        return direct_map[base_title] + suffix
    for scope_label in ("Recoverable", "Structural"):
        if base_title.startswith(f"{scope_label} R") and base_title.endswith(" Local-Law Coverage"):
            parts = base_title.split()
            budget_label = parts[1] if len(parts) > 1 else "R?"
            return f"{scope_label} Extra Count Labels at {budget_label}{suffix}"
        if base_title.startswith(f"{scope_label} R") and base_title.endswith(" Mass-Matched Coverage"):
            parts = base_title.split()
            budget_label = parts[1] if len(parts) > 1 else "R?"
            return f"{scope_label} Fixed-Budget Tree vs FNO at {budget_label}{suffix}"
    return title


def _recoverable_scope_key(recovery: Mapping[str, Any]) -> str:
    explicit = str(recovery.get("recoverable_scope_key", "") or "").strip()
    if explicit:
        return explicit
    scopes = dict(recovery.get("scopes") or {})
    if SUPERVISION_RECOVERY_RECOVERABLE_BENCHMARK in scopes:
        return SUPERVISION_RECOVERY_RECOVERABLE_BENCHMARK
    for scope_key in scopes:
        if str(scope_key).strip().lower().startswith("recoverable"):
            return str(scope_key)
    return SUPERVISION_RECOVERY_RECOVERABLE_BENCHMARK


def _structural_scope_key(recovery: Mapping[str, Any]) -> str:
    return str(
        recovery.get("structural_scope_key", SUPERVISION_RECOVERY_STRUCTURAL_CELL)
        or SUPERVISION_RECOVERY_STRUCTURAL_CELL
    )


def _structural_hardness_grid(recovery: Mapping[str, Any]) -> str:
    return str(
        recovery.get(
            "structural_hardness_grid",
            SUPERVISION_RECOVERY_STRUCTURAL_GRID,
        )
        or SUPERVISION_RECOVERY_STRUCTURAL_GRID
    )


def _scope_label_from_recovery(recovery: Mapping[str, Any], scope_key: str) -> str:
    scope = dict((recovery.get("scopes") or {}).get(str(scope_key)) or {})
    return str(scope.get("scope_label", scope_key) or scope_key)


def _focused_walkthrough_lines(summary: Mapping[str, Any]) -> List[str]:
    recovery = dict(summary.get("supervision_recovery") or {})
    recoverable_scope = _recoverable_scope_key(recovery)
    structural_scope = _structural_scope_key(recovery)
    structural_label = _scope_label_from_recovery(recovery, structural_scope)
    if _is_r10_coverage_focused(summary):
        return [
            "- Start with the root-supervision sweeps. They show what changes when only the share of root-labeled docs changes.",
            "- Then read the `R10` count-label sweeps. Those hold the root budget fixed and add more tree count labels.",
            "- The endpoint pages compare the same `R10` root budget at the two extremes: root-only versus root plus full tree count labels.",
            f"- Read `Recoverable` first, then `Structural` on `{structural_label}`. The appendix keeps the full `R10` ablation chain as supporting context.",
        ]
    lines = [
        "- Start with the full-supervision baseline. It is the simplest comparison because every method gets fully supervised training docs.",
        "- Then read the recoverable plots in order: root-supervision sweep, full-local-plus-root sweep, and the `R10` tree-label ablation.",
        f"- After that, repeat the same pattern on `Structural`, the harder benchmark on `{structural_label}`.",
        "- Read the fixed-budget plots after the main sweeps. They ask whether tree-local labels can replace some root-labeled docs at the same total training-doc-equivalent budget.",
    ]
    step_index = 8
    for root_share in sorted(SUPERVISION_RECOVERY_LOCAL_LAW_RATE_PACKAGE_ORDERS):
        if _has_local_law_rate_payloads(
            recovery,
            scope_key=recoverable_scope,
            root_share=root_share,
        ):
            lines.append(
                f"- Step {step_index}. Recoverable extra count labels at `R{int(root_share)}`: hold the root budget fixed and add equal leaf/internal count supervision."
            )
            step_index += 1
        if _has_local_law_rate_payloads(
            recovery,
            scope_key=structural_scope,
            root_share=root_share,
        ):
            lines.append(
                f"- Step {step_index}. Structural extra count labels at `R{int(root_share)}`: the same count-label sweep on the harder structural cell."
            )
            step_index += 1
    for root_share in sorted(SUPERVISION_RECOVERY_MASS_MATCHED_RATE_PACKAGE_ORDERS):
        if _has_mass_matched_law_rate_payloads(
            recovery,
            scope_key=recoverable_scope,
            root_share=root_share,
        ):
            lines.append(
                f"- Step {step_index}. Recoverable fixed-budget `R{int(root_share)}` comparison: keep total training-doc equivalents fixed and shift some budget into tree-local labels."
            )
            step_index += 1
        if _has_mass_matched_law_rate_payloads(
            recovery,
            scope_key=structural_scope,
            root_share=root_share,
        ):
            lines.append(
                f"- Step {step_index}. Structural fixed-budget `R{int(root_share)}` comparison: the same budget tradeoff on the harder structural cell."
            )
            step_index += 1
    lines.append(
        f"- Step {step_index}. Finish with the all-settings ladders and the tree-only diagnostics. They are supporting views, not the main result."
    )
    return lines


def _focused_protocol_lines(summary: Mapping[str, Any]) -> List[str]:
    recovery = dict(summary.get("supervision_recovery") or {})
    scope_tree_references = dict(recovery.get("scope_tree_references") or {})
    train_doc_counts = _focused_train_doc_counts(recovery)
    recoverable_scope = _recoverable_scope_key(recovery)
    recoverable_label = _scope_label_from_recovery(recovery, recoverable_scope)
    structural_scope = _structural_scope_key(recovery)
    structural_label = _scope_label_from_recovery(recovery, structural_scope)
    lines = [
        f"- Tree model: `{recovery.get('tree_family', SUPERVISION_RECOVERY_TREE_FAMILY)}`.",
        "- FNO baselines: `"
        + ", ".join(str(item) for item in recovery.get("canonical_fno_families", []))
        + "`.",
        "- `official_fno` is the canonical parity comparator. `official_fno_sumlen` is a separate FNO family and may define a stronger ceiling on some cells.",
        f"- Benchmarks: `{recoverable_label}` and `{structural_label}`.",
        "- Train docs: `"
        + ", ".join(str(int(_safe_int(value))) for value in train_doc_counts)
        + "`.",
        "- Seeds: `"
        + ", ".join(str(int(_safe_int(value))) for value in recovery.get("seeds", []))
        + f"` (n=`{_safe_int(recovery.get('seed_count'))}`).",
        "- Training-doc equivalents are the full-doc-equivalent supervision budget, so `R10` at `train_docs=10240` is about `1024` equivalents and `R100` is `10240` equivalents.",
        "- In root-share plots, the tree and FNO models use the same root supervision budget. In fixed-budget plots, the tree can trade some of that budget for leaf/internal labels while FNO stays at the root-only anchor.",
    ]
    for scope_key in (
        recoverable_scope,
        structural_scope,
    ):
        reference = dict(scope_tree_references.get(scope_key) or {})
        if not reference:
            continue
        scope_label = str(reference.get("scope_label", scope_key) or scope_key)
        detail_parts: List[str] = []
        label = str(reference.get("tree_reference_label", "") or "").strip()
        schedule = str(reference.get("tree_training_schedule", "") or "").strip()
        state_dim = int(_safe_int(reference.get("state_dim"), 0))
        hidden_dim = int(_safe_int(reference.get("hidden_dim"), 0))
        leaf_tokens = int(_safe_int(reference.get("fixed_leaf_tokens"), 0))
        if label:
            detail_parts.append(f"reference={label}")
        if schedule:
            detail_parts.append(f"schedule={schedule}")
        if state_dim > 0 and hidden_dim > 0:
            detail_parts.append(f"state/hidden={state_dim}/{hidden_dim}")
        if leaf_tokens > 0:
            detail_parts.append(f"leaf_tokens={leaf_tokens}")
        if detail_parts:
            lines.append(f"- Tree recipe (`{scope_label}`): " + ", ".join(detail_parts) + ".")
    family_rows = [dict(row) for row in list(recovery.get("family_rows") or [])]
    observed_leaf_tokens = sorted(
        {
            int(_safe_int(row.get("fixed_leaf_tokens"), 0))
            for row in family_rows
            if str(row.get("baseline_family", "") or "") == str(recovery.get("tree_family", SUPERVISION_RECOVERY_TREE_FAMILY))
            and int(_safe_int(row.get("fixed_leaf_tokens"), 0)) > 0
        }
    )
    if observed_leaf_tokens:
        lines.append(
            "- Observed tree geometries in this report: `"
            + ", ".join(str(value) for value in observed_leaf_tokens)
            + "` fixed leaf tokens."
        )
    return lines


def _focused_scope_lines(
    summary: Mapping[str, Any],
    *,
    scope_key: str,
    title_kind: str,
) -> List[str]:
    recovery = dict(summary.get("supervision_recovery") or {})
    if not _source_is_ready(summary, "supervision_recovery_summary") or str(recovery.get("status", "")) != "ready":
        return _source_placeholder_lines(summary, "supervision_recovery_summary")
    scope = dict((recovery.get("scopes") or {}).get(str(scope_key)) or {})
    if not scope:
        return [
            f"Scope: {title_kind}",
            "Status: missing",
            f"Reason: scope {scope_key} is absent from the supervision-recovery summary",
        ]
    lines = [f"- Scope: {scope.get('scope_label', scope_key)}"]
    if title_kind == "dense_anchor":
        if list(scope.get("dense_anchor_rows") or []):
            return [
                "- X-axis: training docs. All models get full root supervision, so the supervision budget rises with the number of training docs.",
                "- What stays fixed: this is always the `full100` setting, so training-doc equivalents equal training docs.",
                "- The FNO lines are the dense baselines. Tree reruns are shown as separate lineage-labeled overlays when more than one valid curve exists.",
                "- Main plots use the canonical train-doc counts in this report. Intermediate exploratory counts remain in the underlying bundle summaries rather than being overlaid here.",
                f"- Example: {_root_budget_example_text(_focused_train_doc_counts(recovery), root_share=100)}",
            ]
        return [
            f"- Scope: {scope.get('scope_label', scope_key)}.",
            "- Status: unavailable.",
            "- Reason: no full-supervision baseline rows are present.",
        ]
    if title_kind == "ordered":
        return [
            "- X-axis: root supervision share. Each panel keeps the number of training docs fixed and changes only how many of those docs get root labels.",
            "- The FNO lines are the matched root-only baselines. Tree reruns are shown as separate lineage-labeled overlays when multiple valid curves exist.",
            "- Root-only and dense-local tree regimes share colors across figures, but exact duplicate copied curves are collapsed so the same point is not drawn repeatedly.",
            "- The dotted benchmark line is the best result with 100% root supervision at the same train-doc count.",
            f"- Example: {_root_budget_example_text(_focused_train_doc_counts(recovery), root_share=10)}",
        ]
    if title_kind == "dense_local_ordered":
        return [
            "- X-axis: root supervision share. The tree always keeps full local leaf/internal labels and only the root budget changes.",
            "- The FNO lines are matched root-only baselines. This isolates the question: once the tree already has strong local labels, how much extra root supervision still helps?",
            "- Main plots use the canonical train-doc counts in this report. Intermediate exploratory counts stay in the bundle summaries instead of being overlaid here.",
            "- The dotted benchmark line is the best result with 100% root supervision at the same train-doc count.",
            f"- Example: {_root_budget_example_text(_focused_train_doc_counts(recovery), root_share=10)}",
        ]
    if title_kind == "local_ablation":
        return [
            "- X-axis: tree label setting at fixed `R10`. Moving right adds richer tree labels while keeping the root budget fixed.",
            "- The FNO lines are flat `R10` baselines, because FNO does not use the extra tree-local labels on the x-axis.",
            "- The dotted benchmark line is the best result with 100% root supervision at the same train-doc count.",
            f"- Example: {_root_budget_example_text(_focused_train_doc_counts(recovery), root_share=10)}",
        ]
    if title_kind == "local_law_rate":
        title_kind = "local_law_rate_r10"
    if title_kind == "mass_matched_overlay":
        available_root_shares = [
            int(root_share)
            for root_share in sorted(SUPERVISION_RECOVERY_MASS_MATCHED_RATE_PACKAGE_ORDERS)
            if _has_mass_matched_law_rate_payloads(
                recovery,
                scope_key=scope_key,
                root_share=int(root_share),
            )
        ]
        ladder_text = ", ".join(f"`R{root_share}`" for root_share in available_root_shares) or "none"
        budget_text = _mass_matched_budget_legend_text(available_root_shares)
        return [
            "- X-axis: equal leaf/internal count rate for the tree. `0%` is the root-only anchor for each budget ladder.",
            f"- What stays fixed: {budget_text}.",
            "- The FNO lines are flat comparison baselines repeated across the x-axis. They do not have a literal local-rate meaning away from `0%`.",
            f"- Available ladders in this scope: {ladder_text}.",
        ]
    if title_kind.startswith("local_law_rate_r"):
        root_share = int(_safe_int(title_kind.removeprefix("local_law_rate_r"), 0))
        return [
            f"- X-axis: equal leaf/internal count coverage at fixed `R{root_share}` root supervision.",
            f"- What stays fixed: the root budget. Moving right adds more tree count labels without changing the number of root-labeled docs.",
            f"- The FNO lines are flat `R{root_share}` baselines, because FNO does not use the extra tree count labels on the x-axis.",
            f"- Example: {_root_budget_example_text(_focused_train_doc_counts(recovery), root_share=root_share)}",
        ]
    if title_kind.startswith("mass_matched_rate_r"):
        root_share = int(_safe_int(title_kind.removeprefix("mass_matched_rate_r"), 0))
        return [
            f"- X-axis: equal leaf/internal count rate for the tree. `0%` is the root-only `full{root_share}` anchor.",
            f"- What stays fixed: the total `R{root_share}` training-doc-equivalent budget. Moving right shifts some of that fixed budget from root labels into tree-local labels.",
            f"- The FNO lines are flat `full{root_share}` baselines. They do not have a literal local-rate meaning away from `0%`.",
            f"- Example: {_root_budget_example_text(_focused_train_doc_counts(recovery), root_share=root_share)}",
        ]
    if title_kind.startswith("leaf_geometry_r"):
        root_share = int(_safe_int(title_kind.removeprefix("leaf_geometry_r"), 0))
        return [
            "- X-axis: leaves per document. Each tick also shows the corresponding `fixed_leaf_tokens` value.",
            f"- The leftmost point is the root-only `full{root_share}` anchor at `1 leaf/doc`.",
            "- It is treated as FNO-equivalent only when the executed row is marked as exact full-doc parity; requested `leaf128` alone is not enough.",
            f"- Moving right keeps the same `R{root_share}` budget family but asks the tree to solve a deeper composition problem.",
            "- The FNO lines are flat same-train-doc baselines and do not change with leaves/doc.",
        ]
    package_order = [
        str(item)
        for item in list(recovery.get("package_order") or [])
        if str(item).strip()
    ]
    has_superset = any(
        _package_semantics(recovery, package_name) == "superset"
        for package_name in package_order
    )
    has_mass_matched = any(
        _package_semantics(recovery, package_name) == "mass_matched"
        for package_name in package_order
    )
    lines.extend(
        [
            "- X-axis: discrete supervision settings, not a continuous curve.",
            "- Each panel keeps training docs fixed and compares the tree with the two FNO families under the same named supervision setting.",
            "- Some tree-only settings have no FNO counterpart. Those bars show what happens when we change tree labels while keeping the model family fixed.",
            "- `R` is root supervision share; `Lc`, `Lf`, `I1`, `I12`, and `Ia` are increasingly rich tree-local labels.",
        ]
    )
    if has_superset:
        lines.append(
            "- `superset` packages keep the named root supervision budget unchanged and add local tree labels on top, so realized full-doc-equivalent mass can exceed the `R` prefix."
        )
    if has_mass_matched:
        lines.append(
            "- `mass-matched` packages keep the total training-doc-equivalent budget fixed and trade some root labels for local tree labels."
        )
    if title_kind == "recoverable":
        lines.append("- `Recoverable` is the easier benchmark, so strong tree-local supervision should often be enough.")
    elif title_kind == "structural":
        lines.append("- `Structural` is the harder benchmark, so the dense full-root baselines are the practical target to beat.")
    return lines


def _leaf_geometry_warning_lines(summary: Mapping[str, Any]) -> List[str]:
    recovery = dict(summary.get("supervision_recovery") or {})
    if str(recovery.get("status", "")) != "ready":
        return []
    tree_family = str(
        recovery.get("tree_family", SUPERVISION_RECOVERY_TREE_FAMILY)
        or SUPERVISION_RECOVERY_TREE_FAMILY
    )
    warnings: List[str] = []
    for row in list(recovery.get("family_rows") or []):
        row_map = dict(row or {})
        if str(row_map.get("baseline_family", "") or "") != tree_family:
            continue
        requested_leaf_tokens = _requested_fixed_leaf_tokens(row_map)
        if requested_leaf_tokens <= 0:
            continue
        executed_leaf_tokens = _effective_fixed_leaf_tokens(row_map)
        executed_leaves = _effective_leaves_per_doc(row_map)
        doc_tokens = int(_safe_int(row_map.get("computed_assumed_doc_tokens"), 0))
        requested_one_leaf = bool(
            requested_leaf_tokens > 0
            and doc_tokens > 0
            and requested_leaf_tokens >= doc_tokens
        )
        mismatch = requested_leaf_tokens != executed_leaf_tokens or (
            requested_one_leaf and executed_leaves != 1
        )
        if not mismatch:
            continue
        warnings.append(
            f"- scope={row_map.get('scope_key', '')}, train_docs={int(_safe_int(row_map.get('train_doc_count'), 0))}, "
            f"package={row_map.get('package_name', '')}: requested `fixed_leaf_tokens={requested_leaf_tokens}` "
            f"but executed as `fixed_leaf_tokens={executed_leaf_tokens}` with `{executed_leaves} leaves/doc`."
        )
    return warnings


def _focused_summary_lines(summary: Mapping[str, Any]) -> List[str]:
    recovery = dict(summary.get("supervision_recovery") or {})
    if not _source_is_ready(summary, "supervision_recovery_summary") or str(recovery.get("status", "")) != "ready":
        return _source_placeholder_lines(summary, "supervision_recovery_summary")

    def _family_bits(
        family_rows: Mapping[str, Any],
    ) -> str:
        parts: List[str] = []
        for family in CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES:
            family_row = dict(family_rows.get(family) or {})
            value = _safe_float(family_row.get("test_root_mae"), float("nan"))
            delta = _safe_float(family_row.get("delta_vs_tree"), float("nan"))
            if not math.isfinite(value) and not math.isfinite(delta):
                continue
            parts.append(
                f"`{family}`={_format_unavailable(value)} (delta={_format_unavailable(delta)})"
            )
        return ", ".join(parts) if parts else "n/a"

    lines = []
    for row in _best_tree_summary_rows(recovery):
        doc_equiv = _safe_float(row.get("doc_equiv_train_docs"), float("nan"))
        doc_equiv_text = (
            _format_doc_count(int(round(doc_equiv))) if math.isfinite(doc_equiv) else "n/a"
        )
        matched_package = str(
            row.get("fno_reference_package", row.get("package_name", "")) or ""
        ).strip() or "matched root-share"
        matched_bits = _family_bits(dict(row.get("matched_fno_family_rows") or {}))
        full100_bits = _family_bits(dict(row.get("full100_fno_family_rows") or {}))
        best_full100_family = str(row.get("best_full100_fno_family", "") or "").strip()
        best_full100_delta = _format_unavailable(row.get("delta_vs_best_full100_fno"))
        best_full100_suffix = (
            f"; best full100 FNO=`{best_full100_family}` (delta={best_full100_delta})"
            if best_full100_family
            else ""
        )
        lines.append(
            f"- {row.get('scope_label')} / train_docs={_safe_int(row.get('train_doc_count'))}: "
            f"{row.get('package_name')} gives tree root MAE {_format_unavailable(row.get('tree_test_root_mae'))}, "
            f"matched `{matched_package}` baselines: {matched_bits}; "
            f"full100 baselines: {full100_bits}"
            f"{best_full100_suffix}, "
            f"doc-equiv≈{doc_equiv_text}"
        )
    return lines or ["- No best-tree summary rows available."]


def _row_fno_family_value(row: Mapping[str, Any], family: str) -> float:
    row_map = dict(row or {})
    family_rows = dict(row_map.get("fno_family_rows") or {})
    direct_value = _safe_float(
        dict(family_rows.get(str(family)) or {}).get("test_root_mae"),
        float("nan"),
    )
    if math.isfinite(direct_value):
        return float(direct_value)
    reference_family = str(row_map.get("fno_reference_family", "") or "").strip()
    if reference_family == str(family):
        return _safe_float(row_map.get("fno_reference_test_root_mae"), float("nan"))
    return float("nan")


def _focused_r100_reference_lines(summary: Mapping[str, Any], *, scope_key: str) -> List[str]:
    recovery = dict(summary.get("supervision_recovery") or {})
    source_recovery = dict(recovery.get("ceiling_recovery") or recovery)
    scope = dict((source_recovery.get("scopes") or {}).get(str(scope_key)) or {})
    scope_label = str(scope.get("scope_label", scope_key) or scope_key)
    if not scope:
        return [
            "- Full-root reference unavailable because no `R100` comparison summary was provided for this scope.",
        ]
    lines = [
        "- `R100` means full root/doc supervision, so training-doc equivalents equal training docs.",
        "- `R100+Lf+Ia` keeps that same full root budget and also adds dense tree-local labels.",
        "- The dotted benchmark line in the plots comes from these full-supervision rows at the same train-doc count.",
    ]
    best_details = _best_full_root_ceiling_details_by_train_docs(recovery, scope_key=scope_key)
    for train_doc_count, payload in _iter_scope_train_doc_payloads(source_recovery, scope_key=scope_key):
        row_map = _preferred_scope_rows_by_package(
            list((payload or {}).get("rows") or [])
        )
        full100_row = dict(row_map.get("full100") or {})
        dense_local_row = dict(row_map.get("full100_leaf_full100_internal_count100") or {})
        detail = dict(best_details.get(int(train_doc_count)) or {})
        lines.append(
            f"- train_docs={int(train_doc_count)}: tree `full100`={_format_unavailable(full100_row.get('tree_test_root_mae'))}, "
            f"`official_fno`={_format_unavailable(_row_fno_family_value(full100_row, 'official_fno'))}, "
            f"`official_fno_sumlen`={_format_unavailable(_row_fno_family_value(full100_row, 'official_fno_sumlen'))}, "
            f"tree `full100+local`={_format_unavailable(dense_local_row.get('tree_test_root_mae'))}, "
            f"benchmark source={_full_root_ceiling_source_label(detail) if detail else 'n/a'} at "
            f"{_format_unavailable(detail.get('value')) if detail else 'n/a'}."
        )
    if recovery.get("ceiling_recovery"):
        lines.extend(
            [
                "",
                "- These `R100` rows come from the matched full-root comparison summary, because the focused `R10` local-law summary does not itself contain `R100` packages.",
            ]
        )
    return lines


def _focused_r10_endpoint_lines(summary: Mapping[str, Any], *, scope_key: str) -> List[str]:
    recovery = dict(summary.get("supervision_recovery") or {})
    scope = dict((recovery.get("scopes") or {}).get(str(scope_key)) or {})
    scope_label = str(scope.get("scope_label", scope_key) or scope_key)
    rows = _local_law_endpoint_rows(recovery, scope_key=scope_key, root_share=10)
    if not rows:
        return [
            "- Endpoint callout unavailable because the `R10 -> R10+LcIa100` rows are incomplete.",
        ]
    lines = [
        "- This is the same `10%` root supervision budget in both packages; only local count supervision changes.",
        "- `full10` is root-only. `full10_leaf_count100_internal_count100` keeps that same root budget and adds full tree count labels.",
        "- Example: at `train_docs=10240`, both settings still use only about `1024` root-labeled docs; the difference is the extra tree count labels.",
    ]
    for row in rows:
        lines.append(
            f"- train_docs={_safe_int(row.get('train_doc_count'))}: tree `full10`={_format_unavailable(row.get('baseline_tree_root_mae'))}, "
            f"`official_fno`={_format_unavailable(row.get('official_fno_root_mae'))}, "
            f"`official_fno_sumlen`={_format_unavailable(row.get('official_fno_sumlen_root_mae'))}, "
            f"tree `R10+count`={_format_unavailable(row.get('endpoint_tree_root_mae'))}, "
            f"delta vs tree={_format_unavailable(row.get('delta_vs_baseline_tree'))}, "
            f"delta vs `official_fno`={_format_unavailable(row.get('delta_vs_official_fno'))}, "
            f"delta vs `official_fno_sumlen`={_format_unavailable(row.get('delta_vs_official_fno_sumlen'))}."
        )
    return lines


def _supervision_recovery_non_monotone_warnings(
    recovery: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []
    scopes = dict(recovery.get("scopes") or {})
    for scope_key, scope in scopes.items():
        ordered_payloads = _ordered_family_payloads(recovery, scope_key=str(scope_key))
        if not ordered_payloads:
            continue
        for family_label, series_key in (
            ("tree root-only", "tree_root_only_root_mae"),
            ("tree + leaf/internal", "tree_root_local_root_mae"),
        ):
            root_shares = list(ordered_payloads[0]["root_shares"])
            for share_idx, root_share in enumerate(root_shares):
                observed: List[tuple[int, float]] = []
                for payload in ordered_payloads:
                    value = _safe_float(payload[series_key][share_idx], float("nan"))
                    if math.isfinite(value):
                        observed.append((int(_safe_int(payload["train_doc_count"])), float(value)))
                for (from_docs, from_value), (to_docs, to_value) in zip(observed, observed[1:]):
                    if to_value <= from_value:
                        continue
                    delta_abs = float(to_value - from_value)
                    delta_pct = (
                        float(100.0 * delta_abs / max(abs(from_value), 1e-9))
                        if math.isfinite(from_value)
                        else float("nan")
                    )
                    if delta_abs < 0.01 and (not math.isfinite(delta_pct) or delta_pct < 15.0):
                        continue
                    warnings.append(
                        {
                            "scope_key": str(scope_key),
                            "scope_label": str(scope.get("scope_label", scope_key)),
                            "family_label": family_label,
                            "root_share": int(root_share),
                            "from_train_docs": int(from_docs),
                            "to_train_docs": int(to_docs),
                            "from_root_mae": float(from_value),
                            "to_root_mae": float(to_value),
                            "delta_abs": delta_abs,
                            "delta_pct": delta_pct,
                        }
                    )
    warnings.sort(
        key=lambda row: (
            str(row.get("scope_label", "")),
            str(row.get("family_label", "")),
            int(_safe_int(row.get("root_share"), 0)),
            int(_safe_int(row.get("from_train_docs"), 0)),
        )
    )
    return warnings


def _focused_runtime_appendix_lines(summary: Mapping[str, Any]) -> List[str]:
    recovery = dict(summary.get("supervision_recovery") or {})
    runtime = dict(recovery.get("runtime_diagnosis") or {})
    if str(runtime.get("status", "")) != "ready":
        return ["- Runtime diagnosis unavailable for this supervision-recovery summary."]
    lines = [
        (
            f"- Tree fast-path confirmed rows: "
            f"`{_safe_int(runtime.get('tree_fast_path_confirmed_runs'))}` / "
            f"`{_safe_int(runtime.get('tree_fast_path_confirmed_runs')) + _safe_int(runtime.get('tree_partial_or_fallback_runs'))}` "
            f"({_safe_float(runtime.get('tree_fast_path_completion_rate'), 0.0) * 100.0:.1f}%)."
        ),
        (
            f"- Zero steady-state H2D rate across completed tree rows: "
            f"`{_safe_float(runtime.get('tree_zero_h2d_rate'), 0.0) * 100.0:.1f}%`."
        ),
        (
            f"- Median tree train-loop time per epoch: "
            f"`{_safe_float(runtime.get('tree_median_train_loop_s_per_epoch'), float('nan')):.4f}s`; "
            f"per epoch per 1k docs: "
            f"`{_safe_float(runtime.get('tree_median_train_loop_s_per_epoch_per_1k_docs'), float('nan')):.4f}s`."
        ),
        (
            f"- Median resident hits / dense-bucket hits / fused batches: "
            f"`{_safe_float(runtime.get('tree_median_resident_store_hits'), float('nan')):.2f}` / "
            f"`{_safe_float(runtime.get('tree_median_dense_bucket_hits'), float('nan')):.2f}` / "
            f"`{_safe_float(runtime.get('tree_median_auto_queue_fused_batches'), float('nan')):.2f}`."
        ),
        (
            f"- Median document-loss batch scale under coverage normalization: "
            f"`{_safe_float(runtime.get('tree_median_document_loss_batch_scale'), float('nan')):.3f}`."
        ),
        (
            f"- Current evidence: "
            f"`{str(runtime.get('current_evidence_status', 'strict_causal_ab_proof_pending'))}`."
        ),
    ]
    grouped_rows = list(runtime.get("grouped_rows") or [])
    if grouped_rows:
        lines.append("- Detailed per-package runtime rows remain in `summary.json`; the report keeps only the top-level runtime takeaways.")
    return lines


def _focused_stability_warning_lines(summary: Mapping[str, Any]) -> List[str]:
    recovery = dict(summary.get("supervision_recovery") or {})
    warnings = list(summary.get("stability_warnings") or [])
    if not warnings:
        return [
            "- No large non-monotone tree root-MAE jumps were detected across train-doc counts in the ordered families.",
        ]
    lines = [
        "- These are diagnostic warnings only. They flag places where a later train-doc count is materially worse than the previous one for the same tree ladder family and root-share point.",
    ]
    for row in warnings:
        lines.append(
            f"- {row.get('scope_label')} / {row.get('family_label')} / root_share={_safe_int(row.get('root_share'))}%: "
            f"{_safe_int(row.get('from_train_docs'))} -> {_safe_int(row.get('to_train_docs'))} worsened "
            f"{_format_unavailable(row.get('from_root_mae'))} -> {_format_unavailable(row.get('to_root_mae'))} "
            f"(delta={_format_unavailable(row.get('delta_abs'))}, { _safe_float(row.get('delta_pct'), float('nan')):.1f}%)."
        )
    return lines


def _focused_appendix_lines(summary: Mapping[str, Any]) -> List[str]:
    recovery = dict(summary.get("supervision_recovery") or {})
    lines: List[str] = ["Selected sources:"]
    for key in sorted(REPORT_SOURCE_SPECS):
        record = dict((summary.get("source_records") or {}).get(key) or {})
        status = str(record.get("status", "missing") or "missing")
        if key != "supervision_recovery_summary" and status == "missing":
            continue
        path_text = str(record.get("path", "") or "").strip()
        line = f"- {key}: {status}"
        if path_text:
            line += f" @ {path_text}"
        reason = str(record.get("reason", "") or "").strip()
        if reason:
            line += f" — {reason}"
        lines.append(line)
    duplicate_resolution = list(recovery.get("duplicate_resolution") or [])
    if duplicate_resolution:
        lines.append(
            f"- Exact duplicate summary rows collapsed during ingestion: {len(duplicate_resolution)}."
        )
    hidden_invalid_row_count = int(_safe_int(recovery.get("hidden_invalid_row_count"), 0))
    if hidden_invalid_row_count > 0:
        lines.append(
            f"- Hidden invalid or diagnostic-only rows excluded from plots: {hidden_invalid_row_count}."
        )
    for notice in list(recovery.get("notices") or []):
        lines.append(f"- Notice: {notice}")
    return lines


def _focused_figure_path_for_title(
    figure_paths: Mapping[str, str],
    title: str,
) -> Path | None:
    for raw_title, raw_path in figure_paths.items():
        if _visible_supervision_title(str(raw_title)) == str(title):
            return Path(str(raw_path))
    return None


def _write_focused_pdf_section(
    pdf: PdfPages,
    *,
    title: str,
    lines: Sequence[str],
    figure_paths: Mapping[str, str],
) -> None:
    write_text_page(pdf, title=title, lines=list(lines))
    image_path = _focused_figure_path_for_title(figure_paths, title)
    if image_path is not None and image_path.exists():
        write_image_page(pdf, image_path=image_path, title=title)


def _append_focused_markdown_section(
    out_lines: List[str],
    *,
    title: str,
    body_lines: Sequence[str],
    output_path: Path,
    figure_paths: Mapping[str, str],
    heading_level: int = 2,
) -> None:
    out_lines.extend(["", f"{'#' * int(heading_level)} {title}"])
    out_lines.extend(list(body_lines))
    image_path = _focused_figure_path_for_title(figure_paths, title)
    if image_path is not None and image_path.exists():
        rel_path = os.path.relpath(str(image_path), start=str(output_path.parent))
        out_lines.extend(["", f"![{title}]({rel_path})"])


def _render_focused_pdf(summary: Mapping[str, Any], output_path: Path, figure_paths: Mapping[str, str]) -> None:
    recovery = dict(summary.get("supervision_recovery") or {})
    recoverable_scope = _recoverable_scope_key(recovery)
    structural_scope = _structural_scope_key(recovery)
    geometry_groups = list(recovery.get("geometry_groups") or [])
    with PdfPages(output_path) as pdf:
        if _is_exact_full_doc_canary(summary):
            write_text_page(
                pdf,
                title="Markov Exact Full-Doc Canary Report",
                lines=_exact_full_doc_canary_overview_lines(summary),
            )
            write_text_page(
                pdf,
                title="Setup",
                lines=_exact_full_doc_canary_protocol_lines(summary),
            )
            _write_focused_pdf_section(
                pdf,
                title="Recoverable Exact Full-Doc Canary",
                lines=_exact_full_doc_canary_scope_lines(summary, scope_key=recoverable_scope),
                figure_paths=figure_paths,
            )
            _write_focused_pdf_section(
                pdf,
                title="Structural Exact Full-Doc Canary",
                lines=_exact_full_doc_canary_scope_lines(summary, scope_key=structural_scope),
                figure_paths=figure_paths,
            )
            write_text_page(pdf, title="Parity Summary", lines=_focused_summary_lines(summary))
            write_text_page(pdf, title="Runtime Notes", lines=_focused_runtime_appendix_lines(summary))
            write_text_page(pdf, title="Appendix", lines=_focused_appendix_lines(summary))
            return
        if _is_r10_coverage_focused(summary):
            write_text_page(pdf, title="Markov Supervision-Recovery Report", lines=_focused_overview_lines(summary))
            write_text_page(pdf, title="Key Concepts", lines=_focused_key_concept_lines())
            write_text_page(pdf, title="How To Read The Plots", lines=_focused_walkthrough_lines(summary))
            write_text_page(pdf, title="Setup", lines=_focused_protocol_lines(summary))
            geometry_warning_lines = _leaf_geometry_warning_lines(summary)
            if geometry_warning_lines:
                write_text_page(pdf, title="Geometry Warnings", lines=geometry_warning_lines)
            write_text_page(
                pdf,
                title="Recoverable Full-Supervision Reference",
                lines=_focused_r100_reference_lines(summary, scope_key=recoverable_scope),
            )
            write_text_page(
                pdf,
                title="Structural Full-Supervision Reference",
                lines=_focused_r100_reference_lines(summary, scope_key=structural_scope),
            )
            _write_focused_pdf_section(
                pdf,
                title="Recoverable Root-Supervision Sweep",
                lines=_focused_scope_lines(summary, scope_key=recoverable_scope, title_kind="ordered"),
                figure_paths=figure_paths,
            )
            _write_focused_pdf_section(
                pdf,
                title="Recoverable Extra Count Labels at R10",
                lines=_focused_scope_lines(summary, scope_key=recoverable_scope, title_kind="local_law_rate_r10"),
                figure_paths=figure_paths,
            )
            write_text_page(
                pdf,
                title="Recoverable R10 Endpoints",
                lines=_focused_r10_endpoint_lines(summary, scope_key=recoverable_scope),
            )
            _write_focused_pdf_section(
                pdf,
                title="Structural Root-Supervision Sweep",
                lines=_focused_scope_lines(summary, scope_key=structural_scope, title_kind="ordered"),
                figure_paths=figure_paths,
            )
            _write_focused_pdf_section(
                pdf,
                title="Structural Extra Count Labels at R10",
                lines=_focused_scope_lines(summary, scope_key=structural_scope, title_kind="local_law_rate_r10"),
                figure_paths=figure_paths,
            )
            write_text_page(
                pdf,
                title="Structural R10 Endpoints",
                lines=_focused_r10_endpoint_lines(summary, scope_key=structural_scope),
            )
            write_text_page(
                pdf,
                title="Appendix",
                lines=[
                    "- The appendix keeps the `R10` local-ablation chains as supporting context.",
                    "- These figures answer a different question from the main report: which local label family helps most once root supervision is already fixed at `10%`.",
                    "",
                    *_focused_appendix_lines(summary),
                ],
            )
            _write_focused_pdf_section(
                pdf,
                title="Recoverable What Extra Tree Labels Help at R10?",
                lines=_focused_scope_lines(summary, scope_key=recoverable_scope, title_kind="local_ablation"),
                figure_paths=figure_paths,
            )
            _write_focused_pdf_section(
                pdf,
                title="Structural What Extra Tree Labels Help at R10?",
                lines=_focused_scope_lines(summary, scope_key=structural_scope, title_kind="local_ablation"),
                figure_paths=figure_paths,
            )
            return
        write_text_page(pdf, title="Markov Supervision-Recovery Report", lines=_focused_overview_lines(summary))
        write_text_page(pdf, title="Key Concepts", lines=_focused_key_concept_lines())
        write_text_page(pdf, title="How To Read The Plots", lines=_focused_walkthrough_lines(summary))
        write_text_page(pdf, title="Setup", lines=_focused_protocol_lines(summary))
        geometry_warning_lines = _leaf_geometry_warning_lines(summary)
        if geometry_warning_lines:
            write_text_page(pdf, title="Geometry Warnings", lines=geometry_warning_lines)
        if len(geometry_groups) > 1:
            for group in geometry_groups:
                group_key = str(group.get("geometry_key", "") or "")
                group_summary = _summary_filtered_to_geometry(summary, geometry_key=group_key)
                group_recovery = dict(group_summary.get("supervision_recovery") or {})
                suffix = _geometry_section_suffix(group)
                context_lines = _geometry_context_lines(group)
                if context_lines:
                    write_text_page(
                        pdf,
                        title=f"Geometry Group{suffix}",
                        lines=context_lines,
                    )
                _write_focused_pdf_section(
                    pdf,
                    title=f"Full-Supervision Baseline{suffix}",
                    lines=context_lines
                    + _focused_scope_lines(
                        group_summary,
                        scope_key=recoverable_scope,
                        title_kind="dense_anchor",
                    ),
                    figure_paths=figure_paths,
                )
                _write_focused_pdf_section(
                    pdf,
                    title=f"Recoverable Root-Supervision Sweep{suffix}",
                    lines=context_lines
                    + _focused_scope_lines(
                        group_summary,
                        scope_key=recoverable_scope,
                        title_kind="ordered",
                    ),
                    figure_paths=figure_paths,
                )
                _write_focused_pdf_section(
                    pdf,
                    title=f"Structural Root-Supervision Sweep{suffix}",
                    lines=context_lines
                    + _focused_scope_lines(
                        group_summary,
                        scope_key=structural_scope,
                        title_kind="ordered",
                    ),
                    figure_paths=figure_paths,
                )
                _write_focused_pdf_section(
                    pdf,
                    title=f"Recoverable Full Local Supervision + Root Sweep{suffix}",
                    lines=context_lines
                    + _focused_scope_lines(
                        group_summary,
                        scope_key=recoverable_scope,
                        title_kind="dense_local_ordered",
                    ),
                    figure_paths=figure_paths,
                )
                _write_focused_pdf_section(
                    pdf,
                    title=f"Structural Full Local Supervision + Root Sweep{suffix}",
                    lines=context_lines
                    + _focused_scope_lines(
                        group_summary,
                        scope_key=structural_scope,
                        title_kind="dense_local_ordered",
                    ),
                    figure_paths=figure_paths,
                )
                _write_focused_pdf_section(
                    pdf,
                    title=f"Recoverable What Extra Tree Labels Help at R10?{suffix}",
                    lines=context_lines
                    + _focused_scope_lines(
                        group_summary,
                        scope_key=recoverable_scope,
                        title_kind="local_ablation",
                    ),
                    figure_paths=figure_paths,
                )
                _write_focused_pdf_section(
                    pdf,
                    title=f"Structural What Extra Tree Labels Help at R10?{suffix}",
                    lines=context_lines
                    + _focused_scope_lines(
                        group_summary,
                        scope_key=structural_scope,
                        title_kind="local_ablation",
                    ),
                    figure_paths=figure_paths,
                )
                for root_share in sorted(SUPERVISION_RECOVERY_LOCAL_LAW_RATE_PACKAGE_ORDERS):
                    if _has_local_law_rate_payloads(
                        group_recovery,
                        scope_key=recoverable_scope,
                        root_share=root_share,
                    ):
                        _write_focused_pdf_section(
                            pdf,
                            title=f"Recoverable Extra Count Labels at R{int(root_share)}{suffix}",
                            lines=context_lines
                            + _focused_scope_lines(
                                group_summary,
                                scope_key=recoverable_scope,
                                title_kind=f"local_law_rate_r{int(root_share)}",
                            ),
                            figure_paths=figure_paths,
                        )
                    if _has_local_law_rate_payloads(
                        group_recovery,
                        scope_key=structural_scope,
                        root_share=root_share,
                    ):
                        _write_focused_pdf_section(
                            pdf,
                            title=f"Structural Extra Count Labels at R{int(root_share)}{suffix}",
                            lines=context_lines
                            + _focused_scope_lines(
                                group_summary,
                                scope_key=structural_scope,
                                title_kind=f"local_law_rate_r{int(root_share)}",
                            ),
                            figure_paths=figure_paths,
                        )
                for root_share in sorted(SUPERVISION_RECOVERY_MASS_MATCHED_RATE_PACKAGE_ORDERS):
                    if _has_mass_matched_law_rate_payloads(
                        group_recovery,
                        scope_key=recoverable_scope,
                        root_share=root_share,
                    ):
                        _write_focused_pdf_section(
                            pdf,
                            title=f"Recoverable Fixed-Budget Tree vs FNO at R{int(root_share)}{suffix}",
                            lines=context_lines
                            + _focused_scope_lines(
                                group_summary,
                                scope_key=recoverable_scope,
                                title_kind=f"mass_matched_rate_r{int(root_share)}",
                            ),
                            figure_paths=figure_paths,
                        )
                    if _has_mass_matched_law_rate_payloads(
                        group_recovery,
                        scope_key=structural_scope,
                        root_share=root_share,
                    ):
                        _write_focused_pdf_section(
                            pdf,
                            title=f"Structural Fixed-Budget Tree vs FNO at R{int(root_share)}{suffix}",
                            lines=context_lines
                            + _focused_scope_lines(
                                group_summary,
                                scope_key=structural_scope,
                                title_kind=f"mass_matched_rate_r{int(root_share)}",
                            ),
                            figure_paths=figure_paths,
                        )
                if any(
                    _has_mass_matched_law_rate_payloads(
                        group_recovery,
                        scope_key=recoverable_scope,
                        root_share=root_share,
                    )
                    for root_share in sorted(SUPERVISION_RECOVERY_MASS_MATCHED_RATE_PACKAGE_ORDERS)
                ):
                    _write_focused_pdf_section(
                        pdf,
                        title=f"Recoverable Fixed-Budget Comparison Across Budgets{suffix}",
                        lines=context_lines
                        + _focused_scope_lines(
                            group_summary,
                            scope_key=recoverable_scope,
                            title_kind="mass_matched_overlay",
                        ),
                        figure_paths=figure_paths,
                    )
                if any(
                    _has_mass_matched_law_rate_payloads(
                        group_recovery,
                        scope_key=structural_scope,
                        root_share=root_share,
                    )
                    for root_share in sorted(SUPERVISION_RECOVERY_MASS_MATCHED_RATE_PACKAGE_ORDERS)
                ):
                    _write_focused_pdf_section(
                        pdf,
                        title=f"Structural Fixed-Budget Comparison Across Budgets{suffix}",
                        lines=context_lines
                        + _focused_scope_lines(
                            group_summary,
                            scope_key=structural_scope,
                            title_kind="mass_matched_overlay",
                        ),
                        figure_paths=figure_paths,
                    )
                _write_focused_pdf_section(
                    pdf,
                    title=f"Recoverable All Supervision Settings{suffix}",
                    lines=context_lines
                    + _focused_scope_lines(
                        group_summary,
                        scope_key=recoverable_scope,
                        title_kind="recoverable",
                    ),
                    figure_paths=figure_paths,
                )
                _write_focused_pdf_section(
                    pdf,
                    title=f"Structural All Supervision Settings{suffix}",
                    lines=context_lines
                    + _focused_scope_lines(
                        group_summary,
                        scope_key=structural_scope,
                        title_kind="structural",
                    ),
                    figure_paths=figure_paths,
                )
                _write_focused_pdf_section(
                    pdf,
                    title=f"Recoverable Tree-Only Diagnostics{suffix}",
                    lines=context_lines
                    + [
                        "- X-axis: root supervision share at fixed train-doc counts.",
                        "- These rows show tree-only leaf, merge, and local-law diagnostics rather than the main root-MAE comparison.",
                        "- Use this as supporting context after the main accuracy plots.",
                    ],
                    figure_paths=figure_paths,
                )
                _write_focused_pdf_section(
                    pdf,
                    title=f"Structural Tree-Only Diagnostics{suffix}",
                    lines=context_lines
                    + [
                        "- X-axis: root supervision share at fixed train-doc counts.",
                        "- Use these plots to separate root-selection problems from leaf, merge, and broader local-law failures.",
                    ],
                    figure_paths=figure_paths,
                )
            for root_share in sorted(SUPERVISION_RECOVERY_MASS_MATCHED_RATE_PACKAGE_ORDERS):
                if _leaf_geometry_payloads(
                    recovery,
                    scope_key=recoverable_scope,
                    root_share=root_share,
                ):
                    _write_focused_pdf_section(
                        pdf,
                        title=f"Recoverable Leaves/Doc at R{int(root_share)}",
                        lines=_focused_scope_lines(
                            summary,
                            scope_key=recoverable_scope,
                            title_kind=f"leaf_geometry_r{int(root_share)}",
                        ),
                        figure_paths=figure_paths,
                    )
                if _leaf_geometry_payloads(
                    recovery,
                    scope_key=structural_scope,
                    root_share=root_share,
                ):
                    _write_focused_pdf_section(
                        pdf,
                        title=f"Structural Leaves/Doc at R{int(root_share)}",
                        lines=_focused_scope_lines(
                            summary,
                            scope_key=structural_scope,
                            title_kind=f"leaf_geometry_r{int(root_share)}",
                        ),
                        figure_paths=figure_paths,
                    )
            write_text_page(pdf, title="Best Tree Summary", lines=_focused_summary_lines(summary))
            write_text_page(pdf, title="Stability Warnings", lines=_focused_stability_warning_lines(summary))
            write_text_page(pdf, title="Runtime Notes", lines=_focused_runtime_appendix_lines(summary))
            write_text_page(pdf, title="Appendix", lines=_focused_appendix_lines(summary))
            return
        _write_focused_pdf_section(pdf, title="Full-Supervision Baseline", lines=_focused_scope_lines(summary, scope_key=recoverable_scope, title_kind="dense_anchor"), figure_paths=figure_paths)
        _write_focused_pdf_section(pdf, title="Recoverable Root-Supervision Sweep", lines=_focused_scope_lines(summary, scope_key=recoverable_scope, title_kind="ordered"), figure_paths=figure_paths)
        _write_focused_pdf_section(pdf, title="Recoverable Full Local Supervision + Root Sweep", lines=_focused_scope_lines(summary, scope_key=recoverable_scope, title_kind="dense_local_ordered"), figure_paths=figure_paths)
        _write_focused_pdf_section(pdf, title="Recoverable What Extra Tree Labels Help at R10?", lines=_focused_scope_lines(summary, scope_key=recoverable_scope, title_kind="local_ablation"), figure_paths=figure_paths)
        for root_share in sorted(SUPERVISION_RECOVERY_LOCAL_LAW_RATE_PACKAGE_ORDERS):
            if _has_local_law_rate_payloads(
                recovery,
                scope_key=recoverable_scope,
                root_share=root_share,
            ):
                _write_focused_pdf_section(
                    pdf,
                    title=f"Recoverable Extra Count Labels at R{int(root_share)}",
                    lines=_focused_scope_lines(
                        summary,
                        scope_key=recoverable_scope,
                        title_kind=f"local_law_rate_r{int(root_share)}",
                    ),
                    figure_paths=figure_paths,
                )
        _write_focused_pdf_section(pdf, title="Structural Root-Supervision Sweep", lines=_focused_scope_lines(summary, scope_key=structural_scope, title_kind="ordered"), figure_paths=figure_paths)
        _write_focused_pdf_section(pdf, title="Structural Full Local Supervision + Root Sweep", lines=_focused_scope_lines(summary, scope_key=structural_scope, title_kind="dense_local_ordered"), figure_paths=figure_paths)
        _write_focused_pdf_section(pdf, title="Structural What Extra Tree Labels Help at R10?", lines=_focused_scope_lines(summary, scope_key=structural_scope, title_kind="local_ablation"), figure_paths=figure_paths)
        for root_share in sorted(SUPERVISION_RECOVERY_LOCAL_LAW_RATE_PACKAGE_ORDERS):
            if _has_local_law_rate_payloads(
                recovery,
                scope_key=structural_scope,
                root_share=root_share,
            ):
                _write_focused_pdf_section(
                    pdf,
                    title=f"Structural Extra Count Labels at R{int(root_share)}",
                    lines=_focused_scope_lines(
                        summary,
                        scope_key=structural_scope,
                        title_kind=f"local_law_rate_r{int(root_share)}",
                    ),
                    figure_paths=figure_paths,
                )
        for root_share in sorted(SUPERVISION_RECOVERY_MASS_MATCHED_RATE_PACKAGE_ORDERS):
            if _has_mass_matched_law_rate_payloads(
                recovery,
                scope_key=recoverable_scope,
                root_share=root_share,
            ):
                _write_focused_pdf_section(
                    pdf,
                    title=f"Recoverable Fixed-Budget Tree vs FNO at R{int(root_share)}",
                    lines=_focused_scope_lines(
                        summary,
                        scope_key=recoverable_scope,
                        title_kind=f"mass_matched_rate_r{int(root_share)}",
                    ),
                    figure_paths=figure_paths,
                )
            if _has_mass_matched_law_rate_payloads(
                recovery,
                scope_key=structural_scope,
                root_share=root_share,
            ):
                _write_focused_pdf_section(
                    pdf,
                    title=f"Structural Fixed-Budget Tree vs FNO at R{int(root_share)}",
                    lines=_focused_scope_lines(
                        summary,
                        scope_key=structural_scope,
                        title_kind=f"mass_matched_rate_r{int(root_share)}",
                    ),
                    figure_paths=figure_paths,
                )
            if _leaf_geometry_payloads(
                recovery,
                scope_key=recoverable_scope,
                root_share=root_share,
            ):
                _write_focused_pdf_section(
                    pdf,
                    title=f"Recoverable Leaves/Doc at R{int(root_share)}",
                    lines=_focused_scope_lines(
                        summary,
                        scope_key=recoverable_scope,
                        title_kind=f"leaf_geometry_r{int(root_share)}",
                    ),
                    figure_paths=figure_paths,
                )
            if _leaf_geometry_payloads(
                recovery,
                scope_key=structural_scope,
                root_share=root_share,
            ):
                _write_focused_pdf_section(
                    pdf,
                    title=f"Structural Leaves/Doc at R{int(root_share)}",
                    lines=_focused_scope_lines(
                        summary,
                        scope_key=structural_scope,
                        title_kind=f"leaf_geometry_r{int(root_share)}",
                    ),
                    figure_paths=figure_paths,
                )
        _write_focused_pdf_section(pdf, title="Recoverable All Supervision Settings", lines=_focused_scope_lines(summary, scope_key=recoverable_scope, title_kind="recoverable"), figure_paths=figure_paths)
        _write_focused_pdf_section(pdf, title="Structural All Supervision Settings", lines=_focused_scope_lines(summary, scope_key=structural_scope, title_kind="structural"), figure_paths=figure_paths)
        _write_focused_pdf_section(pdf, title="Recoverable Tree-Only Diagnostics", lines=[
            "- X-axis: root supervision share at fixed train-doc counts.",
            "- These rows show tree-only leaf, merge, and local-law diagnostics rather than the main root-MAE comparison.",
            "- Tree reruns may appear as separate lineage-labeled overlays. Exact duplicate copied curves are collapsed before plotting.",
            "- Use this as supporting context after the main accuracy plots.",
        ], figure_paths=figure_paths)
        _write_focused_pdf_section(pdf, title="Structural Tree-Only Diagnostics", lines=[
            "- X-axis: root supervision share at fixed train-doc counts.",
            "- These rows help separate leaf, merge, and broader local-law failures on the harder structural benchmark.",
            "- They are explanatory only; the main comparison still happens on root MAE.",
        ], figure_paths=figure_paths)
        write_text_page(pdf, title="Best Tree Summary", lines=_focused_summary_lines(summary))
        write_text_page(pdf, title="Stability Warnings", lines=_focused_stability_warning_lines(summary))
        write_text_page(pdf, title="Runtime Notes", lines=_focused_runtime_appendix_lines(summary))
        write_text_page(pdf, title="Appendix", lines=_focused_appendix_lines(summary))


def _write_focused_markdown(
    summary: Mapping[str, Any],
    output_path: Path,
    figure_paths: Mapping[str, str],
) -> None:
    recovery = dict(summary.get("supervision_recovery") or {})
    recoverable_scope = _recoverable_scope_key(recovery)
    structural_scope = _structural_scope_key(recovery)
    scope_tree_references = dict(recovery.get("scope_tree_references") or {})
    package_definitions = dict(recovery.get("package_definitions") or {})
    train_doc_counts = _focused_train_doc_counts(recovery)
    def add_section(
        title: str,
        body_lines: Sequence[str],
        *,
        heading_level: int = 2,
    ) -> None:
        _append_focused_markdown_section(
            lines,
            title=title,
            body_lines=body_lines,
            output_path=output_path,
            figure_paths=figure_paths,
            heading_level=heading_level,
        )
    if _is_exact_full_doc_canary(summary):
        lines: List[str] = [
            "# Markov Exact Full-Doc Canary Report",
            "",
            f"Generated: `{summary['generated_at']}`",
            "",
            "## What This Covers",
        ]
        for item in _exact_full_doc_canary_overview_lines(summary)[2:]:
            lines.append(item if item.startswith("-") else f"- {item}")
        lines.extend(["", "## Setup"])
        for item in _exact_full_doc_canary_protocol_lines(summary):
            lines.append(item if item.startswith("-") else item)
        add_section(
            "Recoverable Exact Full-Doc Canary",
            [
                item if item.startswith("-") else f"- {item}."
                for item in _exact_full_doc_canary_scope_lines(summary, scope_key=recoverable_scope)
            ],
        )
        add_section(
            "Structural Exact Full-Doc Canary",
            [
                item if item.startswith("-") else f"- {item}."
                for item in _exact_full_doc_canary_scope_lines(summary, scope_key=structural_scope)
            ],
        )
        lines.extend(["", "## Parity Summary"])
        for item in _focused_summary_lines(summary):
            lines.append(item if item.startswith("-") else f"- {item}.")
        lines.extend(["", "## Runtime Notes"])
        for item in _focused_runtime_appendix_lines(summary):
            lines.append(item if item.startswith("-") else f"- {item}.")
        lines.extend(["", "## Appendix"])
        for item in _focused_appendix_lines(summary):
            lines.append(item if item.startswith("-") else f"- {item}.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    if _is_r10_coverage_focused(summary):
        lines: List[str] = [
            "# Markov Supervision-Recovery Report",
            "",
            f"Generated: `{summary['generated_at']}`",
            "",
            "## How To Read This Report",
        ]
        for item in _focused_overview_lines(summary)[2:]:
            if item:
                lines.append(item if item.startswith("-") else f"- {item}")
        lines.extend(["", "## Key Concepts"])
        for item in _focused_key_concept_lines():
            lines.append(item)
        lines.extend(["", "## How To Read The Plots"])
        for item in _focused_walkthrough_lines(summary):
            lines.append(item)
        lines.extend(["", "## Setup"])
        for item in _focused_protocol_lines(summary):
            lines.append(item if item.startswith("-") else item)
        lines.extend(["", "## Recoverable Full-Supervision Reference"])
        for item in _focused_r100_reference_lines(summary, scope_key=recoverable_scope):
            lines.append(item if item.startswith("-") or item == "" else f"- {item}.")
        lines.extend(["", "## Structural Full-Supervision Reference"])
        for item in _focused_r100_reference_lines(summary, scope_key=structural_scope):
            lines.append(item if item.startswith("-") or item == "" else f"- {item}.")
        add_section(
            "Recoverable Root-Supervision Sweep",
            [
                item if item.startswith("-") else f"- {item}."
                for item in _focused_scope_lines(summary, scope_key=recoverable_scope, title_kind="ordered")
            ],
        )
        add_section(
            "Recoverable Extra Count Labels at R10",
            [
                item if item.startswith("-") else f"- {item}."
                for item in _focused_scope_lines(summary, scope_key=recoverable_scope, title_kind="local_law_rate_r10")
            ],
        )
        lines.extend(["", "## Recoverable R10 Endpoints"])
        for item in _focused_r10_endpoint_lines(summary, scope_key=recoverable_scope):
            lines.append(item if item.startswith("-") or item == "" else f"- {item}.")
        add_section(
            "Structural Root-Supervision Sweep",
            [
                item if item.startswith("-") else f"- {item}."
                for item in _focused_scope_lines(summary, scope_key=structural_scope, title_kind="ordered")
            ],
        )
        add_section(
            "Structural Extra Count Labels at R10",
            [
                item if item.startswith("-") else f"- {item}."
                for item in _focused_scope_lines(summary, scope_key=structural_scope, title_kind="local_law_rate_r10")
            ],
        )
        lines.extend(["", "## Structural R10 Endpoints"])
        for item in _focused_r10_endpoint_lines(summary, scope_key=structural_scope):
            lines.append(item if item.startswith("-") or item == "" else f"- {item}.")
        lines.extend(
            [
                "",
                "## Appendix",
                "- The appendix keeps the `R10` local-ablation chains as supporting context.",
                "- These figures answer a different question from the main report: which local label family helps most once root supervision is already fixed at `10%`.",
                "",
            ]
        )
        add_section(
            "Recoverable What Extra Tree Labels Help at R10?",
            [
                item if item.startswith("-") else f"- {item}."
                for item in _focused_scope_lines(summary, scope_key=recoverable_scope, title_kind="local_ablation")
            ],
            heading_level=3,
        )
        add_section(
            "Structural What Extra Tree Labels Help at R10?",
            [
                item if item.startswith("-") else f"- {item}."
                for item in _focused_scope_lines(summary, scope_key=structural_scope, title_kind="local_ablation")
            ],
            heading_level=3,
        )
        for item in _focused_appendix_lines(summary):
            lines.append(item if item.startswith("-") else f"- {item}.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines: List[str] = [
        "# Markov Supervision-Recovery Report",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## How To Read This Report",
        "- Read from simple to complex: full-supervision baseline, recoverable grids, then structural grids.",
        "- The package ladders and tree-only diagnostics are appendix-style cross-checks, not the primary narrative.",
        "",
        "## Key Concepts",
    ]
    for item in _focused_key_concept_lines():
        lines.append(item)
    lines.extend(
        [
            "",
            "## How To Read The Plots",
        ]
    )
    for item in _focused_walkthrough_lines(summary):
        lines.append(item)
    lines.extend(
        [
            "",
            "## Setup",
        ]
    )
    for item in _focused_protocol_lines(summary):
        lines.append(item if item.startswith("-") else item)
    canonical_tree_selection_metric = str(
        recovery.get("canonical_tree_selection_metric", "") or ""
    ).strip()
    canonical_tree_stage1_checkpoint_metric = str(
        recovery.get("canonical_tree_stage1_checkpoint_metric", "") or ""
    ).strip()
    canonical_comparison_rule = str(
        recovery.get("canonical_comparison_rule", "") or ""
    ).strip()
    comparator_selection_warning = str(
        recovery.get("comparator_selection_warning", "") or ""
    ).strip()
    geometry_groups = list(recovery.get("geometry_groups") or [])
    if canonical_tree_selection_metric:
        lines.append(f"- Canonical tree checkpoint metric: `{canonical_tree_selection_metric}`.")
    if canonical_tree_stage1_checkpoint_metric:
        lines.append(
            f"- Canonical tree stage-1 checkpoint metric: `{canonical_tree_stage1_checkpoint_metric}`."
        )
    if canonical_comparison_rule:
        lines.append(f"- Canonical comparison rule: {canonical_comparison_rule}.")
    if comparator_selection_warning:
        lines.append(f"- Selection warning: {comparator_selection_warning}.")
    geometry_warning_lines = _leaf_geometry_warning_lines(summary)
    if geometry_warning_lines:
        lines.extend(["", "## Geometry Warnings", *geometry_warning_lines])
    if len(geometry_groups) > 1:
        for group in geometry_groups:
            group_key = str(group.get("geometry_key", "") or "")
            group_summary = _summary_filtered_to_geometry(summary, geometry_key=group_key)
            group_recovery = dict(group_summary.get("supervision_recovery") or {})
            suffix = _geometry_section_suffix(group)
            context_lines = _geometry_context_lines(group)
            lines.extend(["", f"## Geometry Group{suffix}"])
            for item in context_lines:
                lines.append(item if item.startswith("-") else f"- {item}.")
            add_section(
                f"Full-Supervision Baseline{suffix}",
                [
                    item if item.startswith("-") else f"- {item}."
                    for item in (
                        context_lines
                        + _focused_scope_lines(
                            group_summary,
                            scope_key=recoverable_scope,
                            title_kind="dense_anchor",
                        )
                    )
                ],
                heading_level=3,
            )
            add_section(
                f"Recoverable Root-Supervision Sweep{suffix}",
                [
                    item if item.startswith("-") else f"- {item}."
                    for item in (
                        context_lines
                        + _focused_scope_lines(
                            group_summary,
                            scope_key=recoverable_scope,
                            title_kind="ordered",
                        )
                    )
                ],
                heading_level=3,
            )
            add_section(
                f"Structural Root-Supervision Sweep{suffix}",
                [
                    item if item.startswith("-") else f"- {item}."
                    for item in (
                        context_lines
                        + _focused_scope_lines(
                            group_summary,
                            scope_key=structural_scope,
                            title_kind="ordered",
                        )
                    )
                ],
                heading_level=3,
            )
            add_section(
                f"Recoverable Full Local Supervision + Root Sweep{suffix}",
                [
                    item if item.startswith("-") else f"- {item}."
                    for item in (
                        context_lines
                        + _focused_scope_lines(
                            group_summary,
                            scope_key=recoverable_scope,
                            title_kind="dense_local_ordered",
                        )
                    )
                ],
                heading_level=3,
            )
            add_section(
                f"Structural Full Local Supervision + Root Sweep{suffix}",
                [
                    item if item.startswith("-") else f"- {item}."
                    for item in (
                        context_lines
                        + _focused_scope_lines(
                            group_summary,
                            scope_key=structural_scope,
                            title_kind="dense_local_ordered",
                        )
                    )
                ],
                heading_level=3,
            )
            add_section(
                f"Recoverable What Extra Tree Labels Help at R10?{suffix}",
                [
                    item if item.startswith("-") else f"- {item}."
                    for item in (
                        context_lines
                        + _focused_scope_lines(
                            group_summary,
                            scope_key=recoverable_scope,
                            title_kind="local_ablation",
                        )
                    )
                ],
                heading_level=3,
            )
            add_section(
                f"Structural What Extra Tree Labels Help at R10?{suffix}",
                [
                    item if item.startswith("-") else f"- {item}."
                    for item in (
                        context_lines
                        + _focused_scope_lines(
                            group_summary,
                            scope_key=structural_scope,
                            title_kind="local_ablation",
                        )
                    )
                ],
                heading_level=3,
            )
            for root_share in sorted(SUPERVISION_RECOVERY_LOCAL_LAW_RATE_PACKAGE_ORDERS):
                if _has_local_law_rate_payloads(
                    group_recovery,
                    scope_key=recoverable_scope,
                    root_share=root_share,
                ):
                    add_section(
                        f"Recoverable Extra Count Labels at R{int(root_share)}{suffix}",
                        [
                            item if item.startswith("-") else f"- {item}."
                            for item in (
                                context_lines
                                + _focused_scope_lines(
                                    group_summary,
                                    scope_key=recoverable_scope,
                                    title_kind=f"local_law_rate_r{int(root_share)}",
                                )
                            )
                        ],
                        heading_level=3,
                    )
                if _has_local_law_rate_payloads(
                    group_recovery,
                    scope_key=structural_scope,
                    root_share=root_share,
                ):
                    add_section(
                        f"Structural Extra Count Labels at R{int(root_share)}{suffix}",
                        [
                            item if item.startswith("-") else f"- {item}."
                            for item in (
                                context_lines
                                + _focused_scope_lines(
                                    group_summary,
                                    scope_key=structural_scope,
                                    title_kind=f"local_law_rate_r{int(root_share)}",
                                )
                            )
                        ],
                        heading_level=3,
                    )
            for root_share in sorted(SUPERVISION_RECOVERY_MASS_MATCHED_RATE_PACKAGE_ORDERS):
                if _has_mass_matched_law_rate_payloads(
                    group_recovery,
                    scope_key=recoverable_scope,
                    root_share=root_share,
                ):
                    add_section(
                        f"Recoverable Fixed-Budget Tree vs FNO at R{int(root_share)}{suffix}",
                        [
                            item if item.startswith("-") else f"- {item}."
                            for item in (
                                context_lines
                                + _focused_scope_lines(
                                    group_summary,
                                    scope_key=recoverable_scope,
                                    title_kind=f"mass_matched_rate_r{int(root_share)}",
                                )
                            )
                        ],
                        heading_level=3,
                    )
                if _has_mass_matched_law_rate_payloads(
                    group_recovery,
                    scope_key=structural_scope,
                    root_share=root_share,
                ):
                    add_section(
                        f"Structural Fixed-Budget Tree vs FNO at R{int(root_share)}{suffix}",
                        [
                            item if item.startswith("-") else f"- {item}."
                            for item in (
                                context_lines
                                + _focused_scope_lines(
                                    group_summary,
                                    scope_key=structural_scope,
                                    title_kind=f"mass_matched_rate_r{int(root_share)}",
                                )
                            )
                        ],
                        heading_level=3,
                    )
            if any(
                _has_mass_matched_law_rate_payloads(
                    group_recovery,
                    scope_key=recoverable_scope,
                    root_share=root_share,
                )
                for root_share in sorted(SUPERVISION_RECOVERY_MASS_MATCHED_RATE_PACKAGE_ORDERS)
            ):
                add_section(
                    f"Recoverable Fixed-Budget Comparison Across Budgets{suffix}",
                    [
                        item if item.startswith("-") else f"- {item}."
                        for item in (
                            context_lines
                            + _focused_scope_lines(
                                group_summary,
                                scope_key=recoverable_scope,
                                title_kind="mass_matched_overlay",
                            )
                        )
                    ],
                    heading_level=3,
                )
            if any(
                _has_mass_matched_law_rate_payloads(
                    group_recovery,
                    scope_key=structural_scope,
                    root_share=root_share,
                )
                for root_share in sorted(SUPERVISION_RECOVERY_MASS_MATCHED_RATE_PACKAGE_ORDERS)
            ):
                add_section(
                    f"Structural Fixed-Budget Comparison Across Budgets{suffix}",
                    [
                        item if item.startswith("-") else f"- {item}."
                        for item in (
                            context_lines
                            + _focused_scope_lines(
                                group_summary,
                                scope_key=structural_scope,
                                title_kind="mass_matched_overlay",
                            )
                        )
                    ],
                    heading_level=3,
                )
            add_section(
                f"Recoverable All Supervision Settings{suffix}",
                [
                    item if item.startswith("-") else f"- {item}."
                    for item in (
                        context_lines
                        + _focused_scope_lines(
                            group_summary,
                            scope_key=recoverable_scope,
                            title_kind="recoverable",
                        )
                    )
                ],
                heading_level=3,
            )
            add_section(
                f"Structural All Supervision Settings{suffix}",
                [
                    item if item.startswith("-") else f"- {item}."
                    for item in (
                        context_lines
                        + _focused_scope_lines(
                            group_summary,
                            scope_key=structural_scope,
                            title_kind="structural",
                        )
                    )
                ],
                heading_level=3,
            )
            add_section(
                f"Recoverable Tree-Only Diagnostics{suffix}",
                [
                    item if item.startswith("-") else f"- {item}."
                    for item in (
                        context_lines
                        + [
                            "- X-axis: root supervision share at fixed train-doc counts.",
                            "- These rows show tree-only leaf, merge, and local-law diagnostics rather than the main root-MAE comparison.",
                            "- Use these plots as supporting context after the main accuracy plots.",
                        ]
                    )
                ],
                heading_level=3,
            )
            add_section(
                f"Structural Tree-Only Diagnostics{suffix}",
                [
                    item if item.startswith("-") else f"- {item}."
                    for item in (
                        context_lines
                        + [
                            "- X-axis: root supervision share at fixed train-doc counts.",
                            "- Use these plots to separate root-selection problems from leaf, merge, and broader local-law failures.",
                        ]
                    )
                ],
                heading_level=3,
            )
        for root_share in sorted(SUPERVISION_RECOVERY_MASS_MATCHED_RATE_PACKAGE_ORDERS):
            if _leaf_geometry_payloads(
                recovery,
                scope_key=recoverable_scope,
                root_share=root_share,
            ):
                add_section(
                    f"Recoverable Leaves/Doc at R{int(root_share)}",
                    [
                        item if item.startswith("-") else f"- {item}."
                        for item in _focused_scope_lines(
                            summary,
                            scope_key=recoverable_scope,
                            title_kind=f"leaf_geometry_r{int(root_share)}",
                        )
                    ],
                )
            if _leaf_geometry_payloads(
                recovery,
                scope_key=structural_scope,
                root_share=root_share,
            ):
                add_section(
                    f"Structural Leaves/Doc at R{int(root_share)}",
                    [
                        item if item.startswith("-") else f"- {item}."
                        for item in _focused_scope_lines(
                            summary,
                            scope_key=structural_scope,
                            title_kind=f"leaf_geometry_r{int(root_share)}",
                        )
                    ],
                )
        lines.extend(["", "## Best Tree Summary"])
        for item in _focused_summary_lines(summary):
            lines.append(item if item.startswith("-") else f"- {item}.")
        lines.extend(["", "## Stability Warnings"])
        for item in _focused_stability_warning_lines(summary):
            lines.append(item if item.startswith("-") else f"- {item}.")
        lines.extend(["", "## Runtime Notes"])
        for item in _focused_runtime_appendix_lines(summary):
            lines.append(item if item.startswith("-") else f"- {item}.")
        lines.extend(["", "## Appendix"])
        for item in _focused_appendix_lines(summary):
            lines.append(item if item.startswith("-") else f"- {item}.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    add_section(
        "Full-Supervision Baseline",
        [
            item if item.startswith("-") else f"- {item}."
            for item in _focused_scope_lines(summary, scope_key=recoverable_scope, title_kind="dense_anchor")
        ],
    )
    add_section(
        "Recoverable Root-Supervision Sweep",
        [
            item if item.startswith("-") else f"- {item}."
            for item in _focused_scope_lines(summary, scope_key=recoverable_scope, title_kind="ordered")
        ],
    )
    add_section(
        "Structural Root-Supervision Sweep",
        [
            item if item.startswith("-") else f"- {item}."
            for item in _focused_scope_lines(summary, scope_key=structural_scope, title_kind="ordered")
        ],
    )
    add_section(
        "Recoverable Full Local Supervision + Root Sweep",
        [
            item if item.startswith("-") else f"- {item}."
            for item in _focused_scope_lines(summary, scope_key=recoverable_scope, title_kind="dense_local_ordered")
        ],
    )
    add_section(
        "Structural Full Local Supervision + Root Sweep",
        [
            item if item.startswith("-") else f"- {item}."
            for item in _focused_scope_lines(summary, scope_key=structural_scope, title_kind="dense_local_ordered")
        ],
    )
    add_section(
        "Recoverable What Extra Tree Labels Help at R10?",
        [
            item if item.startswith("-") else f"- {item}."
            for item in _focused_scope_lines(summary, scope_key=recoverable_scope, title_kind="local_ablation")
        ],
    )
    for root_share in sorted(SUPERVISION_RECOVERY_LOCAL_LAW_RATE_PACKAGE_ORDERS):
        if _has_local_law_rate_payloads(
            recovery,
            scope_key=recoverable_scope,
            root_share=root_share,
        ):
            add_section(
                f"Recoverable Extra Count Labels at R{int(root_share)}",
                [
                    item if item.startswith("-") else f"- {item}."
                    for item in _focused_scope_lines(
                        summary,
                        scope_key=recoverable_scope,
                        title_kind=f"local_law_rate_r{int(root_share)}",
                    )
                ],
            )
    add_section(
        "Structural What Extra Tree Labels Help at R10?",
        [
            item if item.startswith("-") else f"- {item}."
            for item in _focused_scope_lines(summary, scope_key=structural_scope, title_kind="local_ablation")
        ],
    )
    for root_share in sorted(SUPERVISION_RECOVERY_LOCAL_LAW_RATE_PACKAGE_ORDERS):
        if _has_local_law_rate_payloads(
            recovery,
            scope_key=structural_scope,
            root_share=root_share,
        ):
            add_section(
                f"Structural Extra Count Labels at R{int(root_share)}",
                [
                    item if item.startswith("-") else f"- {item}."
                    for item in _focused_scope_lines(
                        summary,
                        scope_key=structural_scope,
                        title_kind=f"local_law_rate_r{int(root_share)}",
                    )
                ],
            )
    for root_share in sorted(SUPERVISION_RECOVERY_MASS_MATCHED_RATE_PACKAGE_ORDERS):
        if _has_mass_matched_law_rate_payloads(
            recovery,
            scope_key=recoverable_scope,
            root_share=root_share,
        ):
            add_section(
                f"Recoverable Fixed-Budget Tree vs FNO at R{int(root_share)}",
                [
                    item if item.startswith("-") else f"- {item}."
                    for item in _focused_scope_lines(
                        summary,
                        scope_key=recoverable_scope,
                        title_kind=f"mass_matched_rate_r{int(root_share)}",
                    )
                ],
            )
        if _has_mass_matched_law_rate_payloads(
            recovery,
            scope_key=structural_scope,
            root_share=root_share,
        ):
            add_section(
                f"Structural Fixed-Budget Tree vs FNO at R{int(root_share)}",
                [
                    item if item.startswith("-") else f"- {item}."
                    for item in _focused_scope_lines(
                        summary,
                        scope_key=structural_scope,
                        title_kind=f"mass_matched_rate_r{int(root_share)}",
                    )
                ],
            )
        if _leaf_geometry_payloads(
            recovery,
            scope_key=recoverable_scope,
            root_share=root_share,
        ):
            add_section(
                f"Recoverable Leaves/Doc at R{int(root_share)}",
                [
                    item if item.startswith("-") else f"- {item}."
                    for item in _focused_scope_lines(
                        summary,
                        scope_key=recoverable_scope,
                        title_kind=f"leaf_geometry_r{int(root_share)}",
                    )
                ],
            )
        if _leaf_geometry_payloads(
            recovery,
            scope_key=structural_scope,
            root_share=root_share,
        ):
            add_section(
                f"Structural Leaves/Doc at R{int(root_share)}",
                [
                    item if item.startswith("-") else f"- {item}."
                    for item in _focused_scope_lines(
                        summary,
                        scope_key=structural_scope,
                        title_kind=f"leaf_geometry_r{int(root_share)}",
                    )
                ],
            )
    if any(
        _has_mass_matched_law_rate_payloads(
            recovery,
            scope_key=recoverable_scope,
            root_share=root_share,
        )
        for root_share in sorted(SUPERVISION_RECOVERY_MASS_MATCHED_RATE_PACKAGE_ORDERS)
    ):
        add_section(
            "Recoverable Fixed-Budget Comparison Across Budgets",
            [
                item if item.startswith("-") else f"- {item}."
                for item in _focused_scope_lines(
                    summary,
                    scope_key=recoverable_scope,
                    title_kind="mass_matched_overlay",
                )
            ],
        )
    if any(
        _has_mass_matched_law_rate_payloads(
            recovery,
            scope_key=structural_scope,
            root_share=root_share,
        )
        for root_share in sorted(SUPERVISION_RECOVERY_MASS_MATCHED_RATE_PACKAGE_ORDERS)
    ):
        add_section(
            "Structural Fixed-Budget Comparison Across Budgets",
            [
                item if item.startswith("-") else f"- {item}."
                for item in _focused_scope_lines(
                    summary,
                    scope_key=structural_scope,
                    title_kind="mass_matched_overlay",
                )
            ],
        )
    add_section(
        "Recoverable All Supervision Settings",
        [
            item if item.startswith("-") else f"- {item}."
            for item in _focused_scope_lines(summary, scope_key=recoverable_scope, title_kind="recoverable")
        ],
    )
    add_section(
        "Structural All Supervision Settings",
        [
            item if item.startswith("-") else f"- {item}."
            for item in _focused_scope_lines(summary, scope_key=structural_scope, title_kind="structural")
        ],
    )
    add_section(
        "Recoverable Tree-Only Diagnostics",
        [
            "- X-axis: root supervision share at fixed train-doc counts.",
            "- These rows show tree-only leaf, merge, and local-law diagnostics rather than the main root-MAE comparison.",
            "- Use these plots as supporting context after the main accuracy plots.",
        ],
    )
    add_section(
        "Structural Tree-Only Diagnostics",
        [
            "- X-axis: root supervision share at fixed train-doc counts.",
            "- Use these plots to separate root-selection problems from leaf, merge, and broader local-law failures.",
        ],
    )
    lines.extend(["", "## Best Tree Summary"])
    for item in _focused_summary_lines(summary):
        lines.append(item if item.startswith("-") else f"- {item}.")
    lines.extend(["", "## Stability Warnings"])
    for item in _focused_stability_warning_lines(summary):
        lines.append(item if item.startswith("-") else f"- {item}.")
    lines.extend(["", "## Runtime Notes"])
    for item in _focused_runtime_appendix_lines(summary):
        lines.append(item if item.startswith("-") else f"- {item}.")
    lines.extend(["", "## Appendix"])
    for item in _focused_appendix_lines(summary):
        lines.append(item if item.startswith("-") else f"- {item}.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    source_records, manifest_payload, version_root = _load_source_records(args)

    summary: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version_root": str(version_root) if version_root is not None else "",
        "sources": {
            key: str(record.get("path"))
            for key, record in sorted(source_records.items())
            if str(record.get("status", "")) == "ready" and str(record.get("path", "")).strip()
        },
        "source_records": source_records,
        "report_focus": str(args.report_profile or REPORT_PROFILE_DEFAULT),
    }
    if args.output_root is not None:
        try:
            from src.experiments.reporting import (
                build_canonical_report_views,
                load_canonical_artifacts,
                load_canonical_result_rows,
            )

            canonical_rows = load_canonical_result_rows(args.output_root)
            canonical_artifacts = load_canonical_artifacts(args.output_root)
            summary["canonical_reporting"] = {
                "output_root": str(Path(args.output_root).expanduser()),
                "result_row_count": len(canonical_rows),
                "artifact_count": len(canonical_artifacts),
                "report_views": build_canonical_report_views(canonical_rows),
            }
        except Exception:
            pass

    recovery_paths = (
        _source_record_paths(source_records["supervision_recovery_summary"])
        if _source_is_ready(summary, "supervision_recovery_summary")
        else []
    )
    recovery_summary: Dict[str, Any]
    if recovery_paths:
        merged_recovery_payload = _merge_supervision_recovery_payloads(
            [_load_json(path) for path in recovery_paths]
        )
        recovery_summary = _summarize_supervision_recovery(
            merged_recovery_payload,
            expected_train_doc_counts=list(
                source_records["supervision_recovery_summary"].get("expected_train_doc_counts") or []
            ),
            expected_package_order=list(
                source_records["supervision_recovery_summary"].get("expected_package_order")
                or SUPERVISION_RECOVERY_PACKAGE_ORDER
            ),
            expected_tree_family=str(
                source_records["supervision_recovery_summary"].get("expected_tree_family")
                or SUPERVISION_RECOVERY_TREE_FAMILY
            ),
            expected_structural_cell=str(
                source_records["supervision_recovery_summary"].get("expected_structural_cell")
                or SUPERVISION_RECOVERY_STRUCTURAL_CELL
            ),
        )
        if str(recovery_summary.get("status", "")) != "ready":
            source_records["supervision_recovery_summary"]["status"] = str(
                recovery_summary.get("status", "incompatible")
            )
            source_records["supervision_recovery_summary"]["reason"] = str(
                recovery_summary.get("reason", "")
            )
    else:
        recovery_summary = {
            "status": "missing",
            "reason": "no local artifact selected",
            "scopes": {},
            "package_order": list(SUPERVISION_RECOVERY_PACKAGE_ORDER),
        }
    ceiling_recovery_summary: Dict[str, Any] = {}
    if args.supervision_recovery_ceiling_summary is not None and args.supervision_recovery_ceiling_summary.exists():
        ceiling_recovery_summary = _summarize_supervision_recovery(
            _load_json(args.supervision_recovery_ceiling_summary),
            expected_train_doc_counts=list(
                source_records["supervision_recovery_summary"].get("expected_train_doc_counts") or []
            ),
            expected_package_order=list(
                source_records["supervision_recovery_summary"].get("expected_package_order")
                or SUPERVISION_RECOVERY_PACKAGE_ORDER
            ),
            expected_tree_family=str(
                source_records["supervision_recovery_summary"].get("expected_tree_family")
                or SUPERVISION_RECOVERY_TREE_FAMILY
            ),
            expected_structural_cell=str(
                source_records["supervision_recovery_summary"].get("expected_structural_cell")
                or SUPERVISION_RECOVERY_STRUCTURAL_CELL
            ),
        )
        summary["sources"]["supervision_recovery_ceiling_summary"] = str(
            args.supervision_recovery_ceiling_summary
        )
    if ceiling_recovery_summary:
        recovery_summary = {
            **recovery_summary,
            "ceiling_recovery": ceiling_recovery_summary,
        }
    if (
        str(summary.get("report_focus", REPORT_PROFILE_DEFAULT) or REPORT_PROFILE_DEFAULT)
        == REPORT_PROFILE_DEFAULT
        and _is_exact_full_doc_canary_recovery(recovery_summary)
    ):
        summary["report_focus"] = REPORT_PROFILE_EXACT_PARITY_CANARY
    recoverable_scope = _recoverable_scope_key(recovery_summary)
    structural_scope = _structural_scope_key(recovery_summary)
    structural_label = _scope_label_from_recovery(recovery_summary, structural_scope)
    summary["supervision_recovery"] = recovery_summary
    summary["hazard_panel_mean_guess_check"] = _build_hazard_panel_mean_guess_check(
        {"supervision_recovery": recovery_summary}
    )
    summary["contract_gate_status"] = str(
        recovery_summary.get("contract_gate_status", "pass") or "pass"
    )
    summary["quarantined_row_count"] = int(
        _safe_int(recovery_summary.get("quarantined_row_count"), 0)
    )
    summary["quarantined_sources"] = list(
        recovery_summary.get("quarantined_sources") or []
    )
    summary["hidden_invalid_row_count"] = int(
        _safe_int(recovery_summary.get("hidden_invalid_row_count"), 0)
    )
    summary["hidden_invalid_sources"] = list(
        recovery_summary.get("hidden_invalid_sources") or []
    )
    summary["hidden_invalid_reasons"] = list(
        recovery_summary.get("hidden_invalid_reasons") or []
    )
    summary["lineage_labels"] = list(recovery_summary.get("lineage_labels") or [])
    summary["protocol"] = {
        "tree_family": str(recovery_summary.get("tree_family", SUPERVISION_RECOVERY_TREE_FAMILY)),
        "canonical_fno_families": list(
            recovery_summary.get("canonical_fno_families", CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES)
        ),
        "benchmarks": [
            str(_scope_label_from_recovery(recovery_summary, recoverable_scope)),
            str(structural_label),
        ],
        "train_doc_counts": _focused_train_doc_counts(recovery_summary),
        "seeds": list(recovery_summary.get("seeds", [])),
        "seed_count": int(_safe_int(recovery_summary.get("seed_count"), 0)),
        "package_order": list(recovery_summary.get("package_order", SUPERVISION_RECOVERY_PACKAGE_ORDER)),
        "package_definitions": dict(recovery_summary.get("package_definitions") or {}),
    }
    summary["dense_anchor"] = list(
        dict((recovery_summary.get("scopes") or {}).get(str(recoverable_scope)) or {}).get("dense_anchor_rows")
        or []
    )
    summary["recoverable_ladder"] = dict((recovery_summary.get("scopes") or {}).get(str(recoverable_scope)) or {})
    summary["structural_ladder"] = dict(
        (recovery_summary.get("scopes") or {}).get(str(structural_scope))
        or {}
    )
    summary["best_tree_summary"] = _best_tree_summary_rows(recovery_summary)
    summary["stability_warnings"] = _supervision_recovery_non_monotone_warnings(
        recovery_summary
    )
    summary["runtime_diagnosis"] = dict(recovery_summary.get("runtime_diagnosis") or {})
    summary["source_records"] = source_records

    figure_paths: Dict[str, str] = {}
    report_focus = _report_profile(summary)
    if report_focus == REPORT_PROFILE_EXACT_PARITY_CANARY:
        figure_builders = [
            (
                "Recoverable Exact Full-Doc Canary",
                lambda payload, out: _plot_supervision_recovery_exact_full_doc_canary(
                    payload,
                    scope_key=str(recoverable_scope),
                    output_path=out,
                ),
                summary.get("supervision_recovery", {}),
                figures_dir / "recoverable_exact_full_doc_canary.png",
            ),
            (
                "Structural Exact Full-Doc Canary",
                lambda payload, out: _plot_supervision_recovery_exact_full_doc_canary(
                    payload,
                    scope_key=str(
                        dict(payload or {}).get(
                            "structural_scope_key",
                            SUPERVISION_RECOVERY_STRUCTURAL_CELL,
                        )
                    ),
                    output_path=out,
                ),
                summary.get("supervision_recovery", {}),
                figures_dir / "structural_exact_full_doc_canary.png",
            ),
        ]
    elif report_focus == REPORT_PROFILE_R10_COVERAGE_FOCUSED:
        figure_builders = [
            (
                "Recoverable Ordered Families",
                lambda payload, out: _plot_supervision_recovery_ordered_families(
                    payload,
                    scope_key=str(recoverable_scope),
                    output_path=out,
                    min_root_share=10,
                ),
                summary.get("supervision_recovery", {}),
                figures_dir / "recoverable_ordered_families.png",
            ),
            (
                "Recoverable R10 Local-Law Coverage",
                lambda payload, out: _plot_supervision_recovery_r10_local_law_rate_grid(
                    payload,
                    scope_key=str(recoverable_scope),
                    output_path=out,
                ),
                summary.get("supervision_recovery", {}),
                figures_dir / "recoverable_r10_local_law_coverage.png",
            ),
            (
                "Structural Ordered Families",
                lambda payload, out: _plot_supervision_recovery_ordered_families(
                    payload,
                    scope_key=str(
                        dict(payload or {}).get("structural_scope_key", SUPERVISION_RECOVERY_STRUCTURAL_CELL)
                    ),
                    output_path=out,
                    min_root_share=10,
                ),
                summary.get("supervision_recovery", {}),
                figures_dir / "structural_ordered_families.png",
            ),
            (
                "Structural R10 Local-Law Coverage",
                lambda payload, out: _plot_supervision_recovery_r10_local_law_rate_grid(
                    payload,
                    scope_key=str(
                        dict(payload or {}).get("structural_scope_key", SUPERVISION_RECOVERY_STRUCTURAL_CELL)
                    ),
                    output_path=out,
                ),
                summary.get("supervision_recovery", {}),
                figures_dir / "structural_r10_local_law_coverage.png",
            ),
            (
                "Recoverable R10 Local Ablations",
                lambda payload, out: _plot_supervision_recovery_local_ablation_grid(
                    payload,
                    scope_key=str(recoverable_scope),
                    output_path=out,
                ),
                summary.get("supervision_recovery", {}),
                figures_dir / "recoverable_r10_local_ablations.png",
            ),
            (
                "Structural R10 Local Ablations",
                lambda payload, out: _plot_supervision_recovery_local_ablation_grid(
                    payload,
                    scope_key=str(
                        dict(payload or {}).get("structural_scope_key", SUPERVISION_RECOVERY_STRUCTURAL_CELL)
                    ),
                    output_path=out,
                ),
                summary.get("supervision_recovery", {}),
                figures_dir / "structural_r10_local_ablations.png",
            ),
        ]
    else:
        def _geometry_slug(label: str) -> str:
            slug = "".join(
                char.lower() if char.isalnum() else "_"
                for char in str(label or "").strip()
            )
            while "__" in slug:
                slug = slug.replace("__", "_")
            return slug.strip("_") or "geometry"

        def _default_supervision_figure_builders(
            payload: Mapping[str, Any],
            *,
            title_suffix: str = "",
            file_suffix: str = "",
            include_leaf_geometry: bool,
        ) -> List[tuple[str, Any, Mapping[str, Any], Path]]:
            structural_scope_for = lambda data: str(
                dict(data or {}).get(
                    "structural_scope_key",
                    SUPERVISION_RECOVERY_STRUCTURAL_CELL,
                )
            )
            builders: List[tuple[str, Any, Mapping[str, Any], Path]] = [
                (
                    f"Dense Full-Doc Anchor{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_dense_anchor(data, out),
                    payload,
                    figures_dir / f"dense_full_doc_anchor{file_suffix}.png",
                ),
                (
                    f"Recoverable Ordered Families{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_ordered_families(
                        data,
                        scope_key=str(recoverable_scope),
                        output_path=out,
                    ),
                    payload,
                    figures_dir / f"recoverable_ordered_families{file_suffix}.png",
                ),
                (
                    f"Structural Ordered Families{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_ordered_families(
                        data,
                        scope_key=structural_scope_for(data),
                        output_path=out,
                    ),
                    payload,
                    figures_dir / f"structural_ordered_families{file_suffix}.png",
                ),
                (
                    f"Recoverable Dense-Local Root Ladder{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_dense_local_root_ladder(
                        data,
                        scope_key=str(recoverable_scope),
                        output_path=out,
                    ),
                    payload,
                    figures_dir / f"recoverable_dense_local_root_ladder{file_suffix}.png",
                ),
                (
                    f"Structural Dense-Local Root Ladder{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_dense_local_root_ladder(
                        data,
                        scope_key=structural_scope_for(data),
                        output_path=out,
                    ),
                    payload,
                    figures_dir / f"structural_dense_local_root_ladder{file_suffix}.png",
                ),
                (
                    f"Recoverable Tree Constant-Density Root Ladders{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_tree_constant_density_root_ladder(
                        data,
                        scope_key=str(recoverable_scope),
                        output_path=out,
                    ),
                    payload,
                    figures_dir
                    / f"recoverable_tree_constant_density_root_ladders{file_suffix}.png",
                ),
                (
                    f"Structural Tree Constant-Density Root Ladders{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_tree_constant_density_root_ladder(
                        data,
                        scope_key=structural_scope_for(data),
                        output_path=out,
                    ),
                    payload,
                    figures_dir
                    / f"structural_tree_constant_density_root_ladders{file_suffix}.png",
                ),
                (
                    f"Recoverable R10 Local Ablations{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_local_ablation_grid(
                        data,
                        scope_key=str(recoverable_scope),
                        output_path=out,
                    ),
                    payload,
                    figures_dir / f"recoverable_r10_local_ablations{file_suffix}.png",
                ),
                (
                    f"Recoverable R10 Local-Law Coverage{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_r10_local_law_rate_grid(
                        data,
                        scope_key=str(recoverable_scope),
                        output_path=out,
                    ),
                    payload,
                    figures_dir / f"recoverable_r10_local_law_coverage{file_suffix}.png",
                ),
                (
                    f"Recoverable R20 Local-Law Coverage{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_local_law_rate_grid(
                        data,
                        scope_key=str(recoverable_scope),
                        output_path=out,
                        root_share=20,
                    ),
                    payload,
                    figures_dir / f"recoverable_r20_local_law_coverage{file_suffix}.png",
                ),
                (
                    f"Recoverable R10 Mass-Matched Coverage{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_mass_matched_rate_grid(
                        data,
                        scope_key=str(recoverable_scope),
                        output_path=out,
                        root_share=10,
                    ),
                    payload,
                    figures_dir / f"recoverable_r10_mass_matched_coverage{file_suffix}.png",
                ),
                (
                    f"Recoverable R20 Mass-Matched Coverage{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_mass_matched_rate_grid(
                        data,
                        scope_key=str(recoverable_scope),
                        output_path=out,
                        root_share=20,
                    ),
                    payload,
                    figures_dir / f"recoverable_r20_mass_matched_coverage{file_suffix}.png",
                ),
                (
                    f"Recoverable R80 Mass-Matched Coverage{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_mass_matched_rate_grid(
                        data,
                        scope_key=str(recoverable_scope),
                        output_path=out,
                        root_share=80,
                    ),
                    payload,
                    figures_dir / f"recoverable_r80_mass_matched_coverage{file_suffix}.png",
                ),
                (
                    f"Recoverable R90 Mass-Matched Coverage{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_mass_matched_rate_grid(
                        data,
                        scope_key=str(recoverable_scope),
                        output_path=out,
                        root_share=90,
                    ),
                    payload,
                    figures_dir / f"recoverable_r90_mass_matched_coverage{file_suffix}.png",
                ),
                (
                    f"Recoverable R100 Mass-Matched Coverage{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_mass_matched_rate_grid(
                        data,
                        scope_key=str(recoverable_scope),
                        output_path=out,
                        root_share=100,
                    ),
                    payload,
                    figures_dir / f"recoverable_r100_mass_matched_coverage{file_suffix}.png",
                ),
                (
                    f"Recoverable Mass-Matched Overlay{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_mass_matched_overlay(
                        data,
                        scope_key=str(recoverable_scope),
                        output_path=out,
                    ),
                    payload,
                    figures_dir / f"recoverable_mass_matched_overlay{file_suffix}.png",
                ),
                (
                    f"Structural R10 Local Ablations{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_local_ablation_grid(
                        data,
                        scope_key=structural_scope_for(data),
                        output_path=out,
                    ),
                    payload,
                    figures_dir / f"structural_r10_local_ablations{file_suffix}.png",
                ),
                (
                    f"Structural R10 Local-Law Coverage{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_r10_local_law_rate_grid(
                        data,
                        scope_key=structural_scope_for(data),
                        output_path=out,
                    ),
                    payload,
                    figures_dir / f"structural_r10_local_law_coverage{file_suffix}.png",
                ),
                (
                    f"Structural R20 Local-Law Coverage{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_local_law_rate_grid(
                        data,
                        scope_key=structural_scope_for(data),
                        output_path=out,
                        root_share=20,
                    ),
                    payload,
                    figures_dir / f"structural_r20_local_law_coverage{file_suffix}.png",
                ),
                (
                    f"Structural R10 Mass-Matched Coverage{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_mass_matched_rate_grid(
                        data,
                        scope_key=structural_scope_for(data),
                        output_path=out,
                        root_share=10,
                    ),
                    payload,
                    figures_dir / f"structural_r10_mass_matched_coverage{file_suffix}.png",
                ),
                (
                    f"Structural R20 Mass-Matched Coverage{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_mass_matched_rate_grid(
                        data,
                        scope_key=structural_scope_for(data),
                        output_path=out,
                        root_share=20,
                    ),
                    payload,
                    figures_dir / f"structural_r20_mass_matched_coverage{file_suffix}.png",
                ),
                (
                    f"Structural R80 Mass-Matched Coverage{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_mass_matched_rate_grid(
                        data,
                        scope_key=structural_scope_for(data),
                        output_path=out,
                        root_share=80,
                    ),
                    payload,
                    figures_dir / f"structural_r80_mass_matched_coverage{file_suffix}.png",
                ),
                (
                    f"Structural R90 Mass-Matched Coverage{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_mass_matched_rate_grid(
                        data,
                        scope_key=structural_scope_for(data),
                        output_path=out,
                        root_share=90,
                    ),
                    payload,
                    figures_dir / f"structural_r90_mass_matched_coverage{file_suffix}.png",
                ),
                (
                    f"Structural R100 Mass-Matched Coverage{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_mass_matched_rate_grid(
                        data,
                        scope_key=structural_scope_for(data),
                        output_path=out,
                        root_share=100,
                    ),
                    payload,
                    figures_dir / f"structural_r100_mass_matched_coverage{file_suffix}.png",
                ),
                (
                    f"Structural Mass-Matched Overlay{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_mass_matched_overlay(
                        data,
                        scope_key=structural_scope_for(data),
                        output_path=out,
                    ),
                    payload,
                    figures_dir / f"structural_mass_matched_overlay{file_suffix}.png",
                ),
                (
                    f"Recoverable Package Ladder{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_ladder(
                        data,
                        scope_key=str(recoverable_scope),
                        output_path=out,
                    ),
                    payload,
                    figures_dir / f"recoverable_package_ladder{file_suffix}.png",
                ),
                (
                    f"Structural Package Ladder{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_ladder(
                        data,
                        scope_key=structural_scope_for(data),
                        output_path=out,
                    ),
                    payload,
                    figures_dir / f"structural_package_ladder{file_suffix}.png",
                ),
                (
                    f"Recoverable Tree Diagnostics{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_tree_diagnostics(
                        data,
                        scope_key=str(recoverable_scope),
                        output_path=out,
                    ),
                    payload,
                    figures_dir / f"recoverable_tree_diagnostics{file_suffix}.png",
                ),
                (
                    f"Structural Tree Diagnostics{title_suffix}",
                    lambda data, out: _plot_supervision_recovery_tree_diagnostics(
                        data,
                        scope_key=structural_scope_for(data),
                        output_path=out,
                    ),
                    payload,
                    figures_dir / f"structural_tree_diagnostics{file_suffix}.png",
                ),
            ]
            if include_leaf_geometry:
                builders.extend(
                    [
                        (
                            f"Recoverable R10 Leaf Geometry{title_suffix}",
                            lambda data, out: _plot_supervision_recovery_leaf_geometry_grid(
                                data,
                                scope_key=str(recoverable_scope),
                                output_path=out,
                                root_share=10,
                            ),
                            payload,
                            figures_dir / f"recoverable_r10_leaf_geometry{file_suffix}.png",
                        ),
                        (
                            f"Recoverable R20 Leaf Geometry{title_suffix}",
                            lambda data, out: _plot_supervision_recovery_leaf_geometry_grid(
                                data,
                                scope_key=str(recoverable_scope),
                                output_path=out,
                                root_share=20,
                            ),
                            payload,
                            figures_dir / f"recoverable_r20_leaf_geometry{file_suffix}.png",
                        ),
                        (
                            f"Recoverable R80 Leaf Geometry{title_suffix}",
                            lambda data, out: _plot_supervision_recovery_leaf_geometry_grid(
                                data,
                                scope_key=str(recoverable_scope),
                                output_path=out,
                                root_share=80,
                            ),
                            payload,
                            figures_dir / f"recoverable_r80_leaf_geometry{file_suffix}.png",
                        ),
                        (
                            f"Recoverable R90 Leaf Geometry{title_suffix}",
                            lambda data, out: _plot_supervision_recovery_leaf_geometry_grid(
                                data,
                                scope_key=str(recoverable_scope),
                                output_path=out,
                                root_share=90,
                            ),
                            payload,
                            figures_dir / f"recoverable_r90_leaf_geometry{file_suffix}.png",
                        ),
                        (
                            f"Recoverable R100 Leaf Geometry{title_suffix}",
                            lambda data, out: _plot_supervision_recovery_leaf_geometry_grid(
                                data,
                                scope_key=str(recoverable_scope),
                                output_path=out,
                                root_share=100,
                            ),
                            payload,
                            figures_dir / f"recoverable_r100_leaf_geometry{file_suffix}.png",
                        ),
                        (
                            f"Structural R10 Leaf Geometry{title_suffix}",
                            lambda data, out: _plot_supervision_recovery_leaf_geometry_grid(
                                data,
                                scope_key=structural_scope_for(data),
                                output_path=out,
                                root_share=10,
                            ),
                            payload,
                            figures_dir / f"structural_r10_leaf_geometry{file_suffix}.png",
                        ),
                        (
                            f"Structural R20 Leaf Geometry{title_suffix}",
                            lambda data, out: _plot_supervision_recovery_leaf_geometry_grid(
                                data,
                                scope_key=structural_scope_for(data),
                                output_path=out,
                                root_share=20,
                            ),
                            payload,
                            figures_dir / f"structural_r20_leaf_geometry{file_suffix}.png",
                        ),
                        (
                            f"Structural R80 Leaf Geometry{title_suffix}",
                            lambda data, out: _plot_supervision_recovery_leaf_geometry_grid(
                                data,
                                scope_key=structural_scope_for(data),
                                output_path=out,
                                root_share=80,
                            ),
                            payload,
                            figures_dir / f"structural_r80_leaf_geometry{file_suffix}.png",
                        ),
                        (
                            f"Structural R90 Leaf Geometry{title_suffix}",
                            lambda data, out: _plot_supervision_recovery_leaf_geometry_grid(
                                data,
                                scope_key=structural_scope_for(data),
                                output_path=out,
                                root_share=90,
                            ),
                            payload,
                            figures_dir / f"structural_r90_leaf_geometry{file_suffix}.png",
                        ),
                        (
                            f"Structural R100 Leaf Geometry{title_suffix}",
                            lambda data, out: _plot_supervision_recovery_leaf_geometry_grid(
                                data,
                                scope_key=structural_scope_for(data),
                                output_path=out,
                                root_share=100,
                            ),
                            payload,
                            figures_dir / f"structural_r100_leaf_geometry{file_suffix}.png",
                        ),
                    ]
                )
            return builders

        figure_builders = []
        geometry_groups = list(recovery_summary.get("geometry_groups") or [])
        if len(geometry_groups) > 1:
            for group in geometry_groups:
                group_key = str(group.get("geometry_key", "") or "")
                group_summary = _summary_filtered_to_geometry(summary, geometry_key=group_key)
                geometry_label = str(
                    group.get("geometry_label", group_key) or group_key
                ).strip()
                title_suffix = _geometry_section_suffix(group)
                file_suffix = f"_{_geometry_slug(group_key or geometry_label)}"
                figure_builders.extend(
                    _default_supervision_figure_builders(
                        group_summary.get("supervision_recovery", {}),
                        title_suffix=title_suffix,
                        file_suffix=file_suffix,
                        include_leaf_geometry=False,
                    )
                )
            leaf_geometry_builders = _default_supervision_figure_builders(
                summary.get("supervision_recovery", {}),
                include_leaf_geometry=True,
            )
            figure_builders.extend(
                [
                    builder
                    for builder in leaf_geometry_builders
                    if "Leaf Geometry" in str(builder[0])
                ]
            )
        else:
            figure_builders = _default_supervision_figure_builders(
                summary.get("supervision_recovery", {}),
                include_leaf_geometry=True,
            )
    for title, builder, payload, output_path in figure_builders:
        if builder(payload, output_path):
            figure_paths[title] = str(output_path)

    pdf_path = args.pdf_path if args.pdf_path is not None else (args.output_dir / "report.pdf")
    _render_focused_pdf(summary, pdf_path, figure_paths)
    summary["pdf"] = str(pdf_path)
    summary["figures"] = figure_paths
    _update_manifest_selected_sources(manifest_payload, source_records=source_records)
    if manifest_payload is not None:
        report_outputs = dict(manifest_payload.get("report_outputs") or {})
        report_outputs.update(
            {
                "summary": str(args.output_dir / "summary.json"),
                "markdown": str(args.output_dir / "report.md"),
                "pdf": str(pdf_path),
            }
        )
        manifest_payload["report_outputs"] = report_outputs

    summary_path = args.output_dir / "summary.json"
    markdown_path = args.output_dir / "report.md"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    alignment_outputs = write_markov_alignment_audit_report(
        build_markov_alignment_audit_report(
            family_grids_summary_json=summary_path,
            run_lean_build=False,
        ),
        output_json=args.output_dir / "markov_alignment_audit.json",
        output_markdown=args.output_dir / "markov_alignment_audit.md",
    )
    alignment_audit_payload = _load_json(Path(alignment_outputs["output_json"]))
    summary["alignment_audit_json"] = str(alignment_outputs["output_json"])
    summary["alignment_audit_markdown"] = str(alignment_outputs["output_markdown"])
    summary["alignment_contract_gate_status"] = (
        "fail"
        if int(
            dict(alignment_audit_payload.get("summary") or {}).get("n_fail", 0)
        )
        > 0
        else "pass"
    )
    summary["contract_gate_status"] = str(
        dict(summary.get("supervision_recovery") or {}).get(
            "contract_gate_status",
            summary.get("contract_gate_status", "pass"),
        )
        or "pass"
    )
    if manifest_payload is not None:
        report_outputs = dict(manifest_payload.get("report_outputs") or {})
        report_outputs.update(
            {
                "alignment_audit_json": str(alignment_outputs["output_json"]),
                "alignment_audit_markdown": str(
                    alignment_outputs["output_markdown"]
                ),
            }
        )
        manifest_payload["report_outputs"] = report_outputs
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_focused_markdown(summary, markdown_path, figure_paths)
    if manifest_payload is not None and args.manifest is not None:
        Path(args.manifest).expanduser().write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "summary_json": str(summary_path),
                "markdown": str(markdown_path),
                "pdf": str(pdf_path),
                "sources": summary["sources"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
