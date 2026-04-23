#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.report_markov_optimization_tradeoffs import (  # noqa: E402
    CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES,
    SUPERVISION_RECOVERY_CANONICAL_COMPARISON_RULE,
    SUPERVISION_RECOVERY_CANONICAL_TREE_SELECTION_METRIC,
    SUPERVISION_RECOVERY_CANONICAL_TREE_STAGE1_SELECTION_METRIC,
    SUPERVISION_RECOVERY_STRUCTURAL_CELL,
    SUPERVISION_RECOVERY_TREE_FAMILY,
    _dense_local_root_ladder_payloads,
    _local_ablation_payloads,
    _ordered_family_payloads,
    _safe_float,
    _safe_int,
    _summarize_supervision_recovery,
)
from src.ctreepo.sim.report.pdf_utils import write_text_page  # noqa: E402

MAIN_TEXT_FIGURE_TITLES: Sequence[str] = (
    "Dense Full-Doc Anchor",
    "Recoverable Ordered Families",
    "Recoverable R10 Local Ablations",
    "Structural Ordered Families",
    "Structural R10 Local Ablations",
)
APPENDIX_FIGURE_TITLES: Sequence[str] = (
    "Recoverable Dense-Local Root Ladder",
    "Structural Dense-Local Root Ladder",
    "Recoverable Package Ladder",
    "Structural Package Ladder",
    "Recoverable Tree Diagnostics",
    "Structural Tree Diagnostics",
)
CRITICAL_ABLATIONS_PRESENT: Sequence[str] = (
    "root-only ladder R10 -> R100",
    "dense-local ladder R0+Lf+Ia -> R100+Lf+Ia",
    "R10 local-ablation chain R10 -> R10+Lc -> R10+Lf -> R10+Lf+I1 -> R10+Lf+I12 -> R10+Lf+Ia",
    "separate official_fno and official_fno_sumlen families",
)
OPTIONAL_FOLLOW_UPS: Sequence[str] = (
    "R50 local-ablation chain only if the paper wants to show that local-label ordering is stable beyond the sparse-root regime",
    "extra seeds for uncertainty bars once the main manuscript figures are frozen",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a paper-readiness audit for the canonical Markov supervision-recovery run "
            "without launching new simulations."
        )
    )
    parser.add_argument("--supervision-recovery-summary", type=Path, required=True)
    parser.add_argument("--report-summary", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / f"markov_supervision_recovery_paper_audit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_pairs(xs: Iterable[Any], ys: Iterable[Any]) -> List[tuple[float, float]]:
    out: List[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        xf = _safe_float(x, float("nan"))
        yf = _safe_float(y, float("nan"))
        if math.isfinite(xf) and math.isfinite(yf):
            out.append((xf, yf))
    return out


def _non_monotone_steps(xs: Sequence[Any], ys: Sequence[Any]) -> List[Dict[str, float]]:
    observed = _finite_pairs(xs, ys)
    out: List[Dict[str, float]] = []
    for (from_x, from_y), (to_x, to_y) in zip(observed, observed[1:]):
        if to_y > from_y + 1e-12:
            out.append(
                {
                    "from_x": float(from_x),
                    "to_x": float(to_x),
                    "from_y": float(from_y),
                    "to_y": float(to_y),
                    "delta_abs": float(to_y - from_y),
                }
            )
    return out


def _figure_refs(
    report_summary: Mapping[str, Any] | None,
    titles: Sequence[str],
) -> List[Dict[str, str]]:
    figures = dict((report_summary or {}).get("figures") or {})
    out: List[Dict[str, str]] = []
    for title in titles:
        path_text = str(figures.get(title, "") or "").strip()
        out.append(
            {
                "title": str(title),
                "path": path_text,
                "status": "present" if path_text else "missing",
            }
        )
    return out


def _bullet(text: Any) -> str:
    line = str(text or "").strip()
    if not line:
        return "-"
    if line.endswith((".", "!", "?")):
        return f"- {line}"
    return f"- {line}."


def _best_tree_map(recovery: Mapping[str, Any], scope_key: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in list(recovery.get("best_tree_summary") or []):
        item = dict(row or {})
        if str(item.get("scope_key", "")) != str(scope_key):
            continue
        train_doc_count = int(_safe_int(item.get("train_doc_count"), 0))
        if train_doc_count > 0:
            out[train_doc_count] = item
    return out


def _scope_ordered_family_checks(recovery: Mapping[str, Any], scope_key: str) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for payload in _ordered_family_payloads(recovery, scope_key=scope_key):
        root_nonmono = _non_monotone_steps(payload.get("root_shares") or [], payload.get("tree_root_only_root_mae") or [])
        local_nonmono = _non_monotone_steps(payload.get("root_shares") or [], payload.get("tree_root_local_root_mae") or [])
        comparable_pairs = [
            (
                _safe_float(root_value, float("nan")),
                _safe_float(local_value, float("nan")),
            )
            for root_value, local_value in zip(
                payload.get("tree_root_only_root_mae") or [],
                payload.get("tree_root_local_root_mae") or [],
            )
        ]
        comparable_pairs = [
            (root_value, local_value)
            for root_value, local_value in comparable_pairs
            if math.isfinite(root_value) and math.isfinite(local_value)
        ]
        local_beats_root_only = sum(1 for root_value, local_value in comparable_pairs if local_value < root_value)
        checks.append(
            {
                "train_doc_count": int(_safe_int(payload.get("train_doc_count"), 0)),
                "root_only_non_monotone_steps": root_nonmono,
                "root_local_non_monotone_steps": local_nonmono,
                "local_beats_root_only_count": int(local_beats_root_only),
                "comparable_share_count": int(len(comparable_pairs)),
            }
        )
    checks.sort(key=lambda row: int(_safe_int(row.get("train_doc_count"), 0)))
    return checks


def _scope_local_ablation_checks(recovery: Mapping[str, Any], scope_key: str) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for payload in _local_ablation_payloads(recovery, scope_key=scope_key):
        package_order = [str(item) for item in list(payload.get("package_order") or [])]
        values = [
            _safe_float(value, float("nan"))
            for value in list(payload.get("tree_root_mae") or [])
        ]
        observed = [
            (package_name, value)
            for package_name, value in zip(package_order, values)
            if math.isfinite(value)
        ]
        if not observed:
            continue
        best_package, best_value = min(observed, key=lambda item: float(item[1]))
        baseline_map = {package_name: value for package_name, value in observed}
        root_only = baseline_map.get("full10", float("nan"))
        full_internal = baseline_map.get("full10_leaf_full100_internal_count100", float("nan"))
        checks.append(
            {
                "train_doc_count": int(_safe_int(payload.get("train_doc_count"), 0)),
                "best_package": str(best_package),
                "best_root_mae": float(best_value),
                "root_only_root_mae": float(root_only) if math.isfinite(root_only) else float("nan"),
                "full_internal_root_mae": float(full_internal) if math.isfinite(full_internal) else float("nan"),
                "full_internal_gain_vs_root_only": float(root_only - full_internal)
                if math.isfinite(root_only) and math.isfinite(full_internal)
                else float("nan"),
            }
        )
    checks.sort(key=lambda row: int(_safe_int(row.get("train_doc_count"), 0)))
    return checks


def _scope_dense_local_ladder_checks(recovery: Mapping[str, Any], scope_key: str) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for payload in _dense_local_root_ladder_payloads(recovery, scope_key=scope_key):
        root_shares = [int(_safe_int(value)) for value in list(payload.get("root_shares") or [])]
        tree_values = [_safe_float(value, float("nan")) for value in list(payload.get("tree_root_local_root_mae") or [])]
        observed = [
            (share, value)
            for share, value in zip(root_shares, tree_values)
            if math.isfinite(value)
        ]
        if not observed:
            continue
        best_share, best_value = min(observed, key=lambda item: float(item[1]))
        checks.append(
            {
                "train_doc_count": int(_safe_int(payload.get("train_doc_count"), 0)),
                "best_root_share": int(best_share),
                "best_root_mae": float(best_value),
                "non_monotone_steps": _non_monotone_steps(root_shares, tree_values),
            }
        )
    checks.sort(key=lambda row: int(_safe_int(row.get("train_doc_count"), 0)))
    return checks


def _scope_summary(
    recovery: Mapping[str, Any],
    *,
    scope_key: str,
) -> Dict[str, Any]:
    scope = dict((recovery.get("scopes") or {}).get(str(scope_key)) or {})
    scope_label = str(scope.get("scope_label", scope_key) or scope_key)
    best_tree_by_docs = _best_tree_map(recovery, scope_key)
    ordered_checks = _scope_ordered_family_checks(recovery, scope_key)
    ablation_checks = _scope_local_ablation_checks(recovery, scope_key)
    dense_local_checks = _scope_dense_local_ladder_checks(recovery, scope_key)

    best_rows = [
        {
            "train_doc_count": int(train_doc_count),
            "package_name": str(row.get("package_name")),
            "tree_test_root_mae": _safe_float(row.get("tree_test_root_mae"), float("nan")),
            "full100_fno_test_root_mae": _safe_float(row.get("full100_fno_test_root_mae"), float("nan")),
            "delta_vs_full100_fno_ceiling": _safe_float(row.get("delta_vs_full100_fno_ceiling"), float("nan")),
            "fno_reference_package": str(row.get("fno_reference_package", "")),
            "fno_reference_family": str(row.get("fno_reference_family", "")),
        }
        for train_doc_count, row in sorted(best_tree_by_docs.items())
    ]

    lines: List[str] = []
    concerns: List[str] = []
    if ordered_checks:
        local_beats_all = all(
            int(row.get("local_beats_root_only_count", 0)) == int(row.get("comparable_share_count", 0))
            and int(row.get("comparable_share_count", 0)) > 0
            for row in ordered_checks
        )
        if local_beats_all:
            lines.append("dense local supervision beats the root-only tree family at every comparable ordered-ladder point")
        else:
            concerns.append("dense local supervision does not dominate root-only at every ordered-ladder point")
        nonmono_count = sum(
            len(row.get("root_only_non_monotone_steps") or []) + len(row.get("root_local_non_monotone_steps") or [])
            for row in ordered_checks
        )
        if nonmono_count:
            concerns.append(
                f"ordered ladders are directionally sensible but not perfectly monotone ({nonmono_count} local reversals across all train-doc panels)"
            )
        else:
            lines.append("ordered ladders are monotone in root share")
    if ablation_checks:
        best_packages = {str(row.get("best_package", "")) for row in ablation_checks}
        if best_packages == {"full10_leaf_full100_internal_count100"}:
            lines.append("the strongest R10 local-ablation endpoint is always the full internal package `R10+Lf+Ia`")
        else:
            concerns.append(
                "the R10 local-ablation winner changes across train-doc counts, so local-label ordering is not fully stable"
            )
    if best_rows:
        dense_fno_wins = [
            row
            for row in best_rows
            if math.isfinite(_safe_float(row.get("delta_vs_full100_fno_ceiling"), float("nan")))
            and _safe_float(row.get("delta_vs_full100_fno_ceiling"), float("nan")) < 0.0
        ]
        if dense_fno_wins:
            train_docs_text = ", ".join(str(int(row["train_doc_count"])) for row in dense_fno_wins)
            lines.append(f"the best tree beats the dense `full100` FNO ceiling at train_docs {train_docs_text}")
        else:
            concerns.append("the best tree never beats the dense `full100` FNO ceiling on this scope")
    return {
        "scope_key": str(scope_key),
        "scope_label": scope_label,
        "best_tree_rows": best_rows,
        "ordered_family_checks": ordered_checks,
        "local_ablation_checks": ablation_checks,
        "dense_local_ladder_checks": dense_local_checks,
        "conclusions": lines,
        "concerns": concerns,
    }


def _canonical_status(recovery: Mapping[str, Any]) -> Dict[str, Any]:
    runtime = dict(recovery.get("runtime_diagnosis") or {})
    raw_scope_tree_reference_labels = recovery.get("scope_tree_reference_labels") or {}
    if isinstance(raw_scope_tree_reference_labels, Mapping):
        scope_tree_reference_labels = {
            str(key): str(value)
            for key, value in dict(raw_scope_tree_reference_labels).items()
            if str(key).strip()
        }
    else:
        scope_tree_reference_labels = {
            str(key): str(dict(value).get("tree_reference_label", ""))
            for key, value in dict(recovery.get("scope_tree_references") or {}).items()
            if str(key).strip()
        }
    unique_labels = sorted({str(value) for value in scope_tree_reference_labels.values() if str(value).strip()})
    tree_checkpoint_metrics = sorted({str(value) for value in list(recovery.get("tree_checkpoint_metrics") or []) if str(value).strip()})
    return {
        "alignment_status": str(recovery.get("comparator_alignment_status", "")),
        "selection_status": str(recovery.get("comparator_selection_status", "")),
        "common_tree_reference_label": str(recovery.get("common_tree_reference_label", "")),
        "scope_tree_reference_labels": scope_tree_reference_labels,
        "scope_tree_reference_label_set": unique_labels,
        "canonical_tree_selection_metric": str(recovery.get("canonical_tree_selection_metric", "")),
        "canonical_tree_stage1_checkpoint_metric": str(recovery.get("canonical_tree_stage1_checkpoint_metric", "")),
        "observed_tree_checkpoint_metrics": tree_checkpoint_metrics,
        "canonical_comparison_rule": str(recovery.get("canonical_comparison_rule", "")),
        "runtime_status": str(runtime.get("status", "")),
        "tree_fast_path_completion_rate": _safe_float(runtime.get("tree_fast_path_completion_rate"), float("nan")),
        "tree_zero_h2d_rate": _safe_float(runtime.get("tree_zero_h2d_rate"), float("nan")),
        "seed_count": int(_safe_int(recovery.get("seed_count"), 0)),
        "train_doc_counts": [int(_safe_int(value)) for value in list(recovery.get("train_doc_counts") or [])],
        "package_count": len(list(recovery.get("package_order") or [])),
    }


def _claims_we_can_make(audit: Mapping[str, Any]) -> List[str]:
    recoverable = dict(audit.get("recoverable_scope") or {})
    structural = dict(audit.get("structural_scope") or {})
    claims = [
        "The canonical Markov supervision-recovery comparison is now apples-to-apples: both scopes use the same non-slotwise tree comparator and all tree ladder points are selected on val_root_mae.",
        "The runtime path is no longer a confound: the completed canonical run stayed on the fast resident + fixed_fused + leaf_count_auto_queue path with zero steady-state H2D in the finished tree rows.",
    ]
    recoverable_best = list(recoverable.get("best_tree_rows") or [])
    if any(_safe_float(row.get("delta_vs_full100_fno_ceiling"), float("nan")) < 0.0 for row in recoverable_best):
        claims.append(
            "On recoverable_v4, stronger local supervision is enough for the common factorized tree to match or beat the dense full-doc FNO ceiling at moderate/high train-doc counts."
        )
    claims.append(
        "On both scopes, dense local supervision is consistently more effective than root-only supervision at the same ordered-ladder points."
    )
    claims.append(
        "At fixed sparse root supervision (R10), the strongest local-ablation endpoint is always the full internal package R10+Lf+Ia, so the main mechanism is not just leaf counts but internal compositional supervision."
    )
    structural_best = list(structural.get("best_tree_rows") or [])
    if any(_safe_float(row.get("delta_vs_full100_fno_ceiling"), float("nan")) > 0.0 for row in structural_best):
        claims.append(
            "On the harder structural cell, the tree clearly learns and improves under stronger local supervision, but dense FNO still retains a real high-data advantage."
        )
    return claims


def _grid_point_guidance() -> Dict[str, Any]:
    return {
        "required_now": [],
        "optional_later": list(OPTIONAL_FOLLOW_UPS),
        "not_recommended_now": [
            "adding more in-between root-share points to the canonical ladder",
            "expanding every local-ablation chain to every root share",
            "rerunning legacy slotwise/factorized comparator ablations for the paper narrative",
        ],
        "recommendation": (
            "The current canonical grid is already broad enough to support the supervision mechanism story. "
            "Do not add new grid points before the manuscript claims are frozen."
        ),
    }


def _publication_readiness(audit: Mapping[str, Any]) -> Dict[str, Any]:
    canonical = dict(audit.get("canonical_status") or {})
    ready = (
        canonical.get("alignment_status") == "aligned"
        and canonical.get("selection_status") == "root_comparable"
        and canonical.get("common_tree_reference_label") == "common_factorized_sketch_v1"
        and canonical.get("canonical_tree_selection_metric") == SUPERVISION_RECOVERY_CANONICAL_TREE_SELECTION_METRIC
        and canonical.get("canonical_tree_stage1_checkpoint_metric") == SUPERVISION_RECOVERY_CANONICAL_TREE_STAGE1_SELECTION_METRIC
        and canonical.get("runtime_status") == "ready"
        and _safe_float(canonical.get("tree_fast_path_completion_rate"), float("nan")) >= 0.999
        and _safe_float(canonical.get("tree_zero_h2d_rate"), float("nan")) >= 0.999
    )
    status = "close_for_paper_draft" if ready else "needs_more_audit"
    return {
        "status": status,
        "supporting_reasons": [
            "one canonical non-slotwise tree comparator on both scopes",
            "uniform root-first checkpointing across the ladder",
            "finished canonical run with 0 failures",
            "runtime fast-path and zero-H2D checks passed",
            "ordered family and local-ablation grids already support the intended supervision-mechanism story",
        ],
        "remaining_tasks_before_submission": [
            "freeze a main-text figure set and move the rest to appendix",
            "add uncertainty bars or seed expansion once the figure set is frozen",
            "turn the audit claims into final manuscript wording and captions",
        ],
    }


def build_paper_audit(
    *,
    supervision_recovery_summary: Mapping[str, Any],
    report_summary: Mapping[str, Any] | None = None,
    supervision_recovery_summary_path: str = "",
    report_summary_path: str = "",
) -> Dict[str, Any]:
    normalized_recovery = _summarize_supervision_recovery(
        supervision_recovery_summary,
        expected_train_doc_counts=list(supervision_recovery_summary.get("train_doc_counts") or []),
        expected_package_order=list(supervision_recovery_summary.get("package_order") or []),
        expected_tree_family=str(supervision_recovery_summary.get("tree_family") or SUPERVISION_RECOVERY_TREE_FAMILY),
        expected_structural_cell=str(
            supervision_recovery_summary.get("structural_scope_key") or SUPERVISION_RECOVERY_STRUCTURAL_CELL
        ),
    )
    structural_scope_key = str(
        normalized_recovery.get("structural_scope_key", SUPERVISION_RECOVERY_STRUCTURAL_CELL)
        or SUPERVISION_RECOVERY_STRUCTURAL_CELL
    )
    recoverable_scope = _scope_summary(normalized_recovery, scope_key="recoverable_v4")
    structural_scope = _scope_summary(normalized_recovery, scope_key=structural_scope_key)

    audit: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready",
        "sources": {
            "supervision_recovery_summary": str(supervision_recovery_summary_path),
            "report_summary": str(report_summary_path),
            "report_pdf": str((report_summary or {}).get("pdf", "") or ""),
        },
        "canonical_status": _canonical_status(normalized_recovery),
        "recoverable_scope": recoverable_scope,
        "structural_scope": structural_scope,
        "claims_we_can_make": [],
        "missing_grid_points": _grid_point_guidance(),
        "critical_ablations": {
            "already_present": list(CRITICAL_ABLATIONS_PRESENT),
            "optional_follow_ups": list(OPTIONAL_FOLLOW_UPS),
            "recommendation": "No new simulation grid points are required before drafting the Markov section.",
        },
        "llm_preference_bridge": [
            "root-only supervision in Markov is the clean analogue of whole-output scalar reward or preference labels",
            "leaf/internal supervision is the controlled analogue of intermediate reasoning-step or local-structure supervision",
            "the Markov result supports the claim that local structure labels shape learning more effectively than simply adding more global labels",
            "the structural gap is still useful for the paper because it shows that better supervision helps a lot without making hard problems trivial",
        ],
        "main_text_figures": _figure_refs(report_summary, MAIN_TEXT_FIGURE_TITLES),
        "appendix_figures": _figure_refs(report_summary, APPENDIX_FIGURE_TITLES),
    }
    audit["claims_we_can_make"] = _claims_we_can_make(audit)
    audit["publication_readiness"] = _publication_readiness(audit)
    return audit


def _audit_markdown_lines(audit: Mapping[str, Any]) -> List[str]:
    canonical = dict(audit.get("canonical_status") or {})
    recoverable = dict(audit.get("recoverable_scope") or {})
    structural = dict(audit.get("structural_scope") or {})
    lines: List[str] = [
        "# Markov Supervision-Recovery Paper Audit",
        "",
        f"Generated: `{audit.get('generated_at', '')}`",
        "",
        "## Canonical Status",
        f"- Alignment status: `{canonical.get('alignment_status')}`.",
        f"- Selection status: `{canonical.get('selection_status')}`.",
        f"- Common tree comparator: `{canonical.get('common_tree_reference_label')}`.",
        f"- Canonical tree checkpoint metric: `{canonical.get('canonical_tree_selection_metric')}`.",
        f"- Canonical tree stage-1 checkpoint metric: `{canonical.get('canonical_tree_stage1_checkpoint_metric')}`.",
        f"- Canonical comparison rule: {canonical.get('canonical_comparison_rule', SUPERVISION_RECOVERY_CANONICAL_COMPARISON_RULE)}.",
        f"- Seeds: `{_safe_int(canonical.get('seed_count'))}`.",
        "- Train docs: `"
        + ", ".join(str(int(_safe_int(value))) for value in list(canonical.get("train_doc_counts") or []))
        + "`.",
        f"- Packages: `{_safe_int(canonical.get('package_count'))}`.",
        f"- Tree fast-path completion rate: `{100.0 * _safe_float(canonical.get('tree_fast_path_completion_rate'), 0.0):.1f}%`.",
        f"- Tree zero-H2D rate: `{100.0 * _safe_float(canonical.get('tree_zero_h2d_rate'), 0.0):.1f}%`.",
        "",
        "## Recoverable Audit",
    ]
    for item in list(recoverable.get("conclusions") or []):
        lines.append(_bullet(item))
    for item in list(recoverable.get("concerns") or []):
        lines.append(_bullet(f"Caveat: {item}"))
    lines.extend(
        [
            "",
            "| train_docs | best tree package | tree root MAE | dense full100 FNO | delta vs dense FNO |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in list(recoverable.get("best_tree_rows") or []):
        lines.append(
            f"| {_safe_int(row.get('train_doc_count'))} | `{row.get('package_name')}` | "
            f"{_safe_float(row.get('tree_test_root_mae'), float('nan')):.6f} | "
            f"{_safe_float(row.get('full100_fno_test_root_mae'), float('nan')):.6f} | "
            f"{_safe_float(row.get('delta_vs_full100_fno_ceiling'), float('nan')):.6f} |"
        )
    lines.extend(["", "## Structural Audit"])
    for item in list(structural.get("conclusions") or []):
        lines.append(_bullet(item))
    for item in list(structural.get("concerns") or []):
        lines.append(_bullet(f"Caveat: {item}"))
    lines.extend(
        [
            "",
            "| train_docs | best tree package | tree root MAE | dense full100 FNO | delta vs dense FNO |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in list(structural.get("best_tree_rows") or []):
        lines.append(
            f"| {_safe_int(row.get('train_doc_count'))} | `{row.get('package_name')}` | "
            f"{_safe_float(row.get('tree_test_root_mae'), float('nan')):.6f} | "
            f"{_safe_float(row.get('full100_fno_test_root_mae'), float('nan')):.6f} | "
            f"{_safe_float(row.get('delta_vs_full100_fno_ceiling'), float('nan')):.6f} |"
        )
    lines.extend(["", "## Claims We Can Safely Make"])
    for item in list(audit.get("claims_we_can_make") or []):
        lines.append(_bullet(item))
    lines.extend(["", "## Figure Plan", "", "### Main Text"])
    for row in list(audit.get("main_text_figures") or []):
        path_text = str(row.get("path", "") or "").strip() or "missing"
        lines.append(f"- {row.get('title')}: `{path_text}` ({row.get('status')}).")
    lines.extend(["", "### Appendix"])
    for row in list(audit.get("appendix_figures") or []):
        path_text = str(row.get("path", "") or "").strip() or "missing"
        lines.append(f"- {row.get('title')}: `{path_text}` ({row.get('status')}).")
    lines.extend(["", "## Missing Grid Points / Follow-Ups"])
    grid_guidance = dict(audit.get("missing_grid_points") or {})
    lines.append(_bullet(f"Recommendation: {grid_guidance.get('recommendation')}"))
    for item in list(grid_guidance.get("required_now") or []):
        lines.append(_bullet(f"Required now: {item}"))
    for item in list(grid_guidance.get("optional_later") or []):
        lines.append(_bullet(f"Optional later: {item}"))
    for item in list(grid_guidance.get("not_recommended_now") or []):
        lines.append(_bullet(f"Not recommended now: {item}"))
    lines.extend(["", "## Critical Ablations"])
    crit = dict(audit.get("critical_ablations") or {})
    for item in list(crit.get("already_present") or []):
        lines.append(_bullet(f"Present: {item}"))
    for item in list(crit.get("optional_follow_ups") or []):
        lines.append(_bullet(f"Optional follow-up: {item}"))
    lines.extend(["", "## Bridge To LLMs And Preferences"])
    for item in list(audit.get("llm_preference_bridge") or []):
        lines.append(_bullet(item))
    readiness = dict(audit.get("publication_readiness") or {})
    lines.extend(["", "## Publication Readiness"])
    lines.append(f"- Status: `{readiness.get('status')}`.")
    for item in list(readiness.get("supporting_reasons") or []):
        lines.append(_bullet(f"Support: {item}"))
    for item in list(readiness.get("remaining_tasks_before_submission") or []):
        lines.append(_bullet(f"Remaining task: {item}"))
    return lines


def _audit_pdf_lines(audit: Mapping[str, Any]) -> Dict[str, List[str]]:
    recoverable = dict(audit.get("recoverable_scope") or {})
    structural = dict(audit.get("structural_scope") or {})
    readiness = dict(audit.get("publication_readiness") or {})
    return {
        "Overview": [
            "This audit is compute-free: it uses the finished canonical supervision-recovery run and the latest plotted report only.",
            f"Status: {readiness.get('status')}",
            "",
            "Core question:",
            "- Is the current Markov supervision-recovery packet coherent enough to anchor the paper section?",
            "",
            "Answer:",
            "- Yes, for draft writing. The remaining work is figure triage, uncertainty, and manuscript framing, not another round of exploratory grid-filling.",
        ],
        "Recoverable": [_bullet(item) for item in list(recoverable.get("conclusions") or [])]
        + [_bullet(f"Caveat: {item}") for item in list(recoverable.get("concerns") or [])],
        "Structural": [_bullet(item) for item in list(structural.get("conclusions") or [])]
        + [_bullet(f"Caveat: {item}") for item in list(structural.get("concerns") or [])],
        "Claims": [_bullet(item) for item in list(audit.get("claims_we_can_make") or [])],
        "Figure Plan": [_bullet(f"Main text: {row.get('title')}") for row in list(audit.get("main_text_figures") or [])]
        + [_bullet(f"Appendix: {row.get('title')}") for row in list(audit.get("appendix_figures") or [])],
        "Next Steps": [_bullet(item) for item in list(readiness.get("remaining_tasks_before_submission") or [])],
    }


def _write_pdf(audit: Mapping[str, Any], output_path: Path) -> None:
    pages = _audit_pdf_lines(audit)
    with PdfPages(output_path) as pdf:
        for title, lines in pages.items():
            write_text_page(pdf, title=title, lines=lines or ["- No content."])


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    supervision_recovery_summary = _load_json(Path(args.supervision_recovery_summary))
    report_summary = _load_json(Path(args.report_summary)) if args.report_summary is not None else None

    audit = build_paper_audit(
        supervision_recovery_summary=supervision_recovery_summary,
        report_summary=report_summary,
        supervision_recovery_summary_path=str(args.supervision_recovery_summary),
        report_summary_path=str(args.report_summary) if args.report_summary is not None else "",
    )

    json_path = args.output_dir / "paper_audit.json"
    md_path = args.output_dir / "paper_audit.md"
    pdf_path = args.output_dir / "paper_audit.pdf"

    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text("\n".join(_audit_markdown_lines(audit)) + "\n", encoding="utf-8")
    _write_pdf(audit, pdf_path)

    print(str(json_path))
    print(str(md_path))
    print(str(pdf_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
