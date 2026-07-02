#!/usr/bin/env python3
"""Unified local-law learnability report for any simulation family.

Usage:
    python scripts/report_learnability.py --family markov --input-root outputs/...
    python scripts/report_learnability.py --family lda    --input-root outputs/...

Produces a markdown report, JSON summary, publication-ready figures, and a
PDF — all in the output directory.

The sweep variable differs by family:
  - Markov: local_law_weight (local-law objective share on C1+C2+C3)
  - LDA:    tau (mixture concentration controlling local structure)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.report.aggregation import aggregate_learnability
from src.ctreepo.sim.report.data_loading import (
    THEOREM_SCORE_SPREAD_WEIGHT,
    load_learnability_records,
)
from src.ctreepo.sim.report.family_config import FamilyReportConfig, resolve_family
from src.ctreepo.sim.report.pdf_utils import (
    write_csv,
    write_image_page,
    write_text_page,
)
from src.ctreepo.sim.report.plots import (
    CapacityKey,
    _capacity_key,
    _capacity_label,
    _format_pct,
    _format_weight,
    plot_audit_summary,
    plot_capacity_summary,
    plot_gain_grid,
    plot_optimization_appendix,
    plot_sweep_grid,
)


# ── CLI ──────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified learnability report.")
    p.add_argument("--family", type=str, required=True, help="markov | lda")
    p.add_argument("--input-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--aggregate", choices=["median", "mean"], default="median")
    p.add_argument("--expected-run-count", type=int, default=None)
    p.add_argument("--status-note", type=str, default="")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--pdf-path", type=str, default=None)
    p.add_argument(
        "--base-field",
        type=str,
        default=None,
        help="Optional baseline axis to compare against (defaults to the family baseline field).",
    )
    p.add_argument(
        "--base-value",
        type=str,
        default=None,
        help="Optional baseline value to compare against (defaults to the family baseline value, else min observed).",
    )
    p.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize count errors by scale factor when applicable.",
    )
    p.add_argument(
        "--paper-safe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exclude rows that are not paper-safe, e.g. missing explicit local-law weights or non-normalized lambda semantics.",
    )
    return p.parse_args()


# ── theorem / objective helpers ──────────────────────────────────────────


def _dedup_axis_values(values: Sequence[object]) -> List[object]:
    if not values:
        return []
    numeric = True
    for value in values:
        if isinstance(value, (int, float, np.integer, np.floating)):
            continue
        numeric = False
        break
    if not numeric:
        return sorted({value for value in values if value is not None})
    rounded: Dict[float, object] = {}
    for value in values:
        if value is None:
            continue
        fv = float(value)
        if not np.isfinite(fv):
            continue
        key = float(round(fv, 12))
        rounded.setdefault(key, key)
    return [rounded[key] for key in sorted(rounded)]


def _row_theorem_score(row: dict) -> float:
    for key in ("theorem_score", "learned_law_score_n", "law_score"):
        if key in row:
            v = float(row[key])
            if np.isfinite(v):
                return v
    return float("nan")


def _row_selection_objective(row: dict) -> float:
    for key in (
        "heldout_objective_for_report",
        "test_objective_full_labels",
        "test_unweighted_objective_full_labels",
    ):
        if key not in row:
            continue
        v = float(row[key])
        if np.isfinite(v):
            return v
    return _row_theorem_score(row)


def _parse_scalar(value: Optional[str]) -> object | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return text


def _value_matches(lhs: object, rhs: object) -> bool:
    try:
        lhs_f = float(lhs)
        rhs_f = float(rhs)
        if np.isfinite(lhs_f) and np.isfinite(rhs_f):
            return bool(np.isclose(lhs_f, rhs_f))
    except (TypeError, ValueError):
        pass
    return lhs == rhs


def _format_scalar(value: object) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _format_weight(float(value))
    return str(value)


def _objective_lambda_interpretations(rows: Sequence[dict]) -> List[str]:
    return sorted({
        str(row.get("objective_lambda_interpretation", "")).strip()
        for row in rows
        if str(row.get("objective_lambda_interpretation", "")).strip()
    })


def _baseline_noun(*, baseline_field: str, lambda_interpretations: Sequence[str]) -> str:
    interpretation_set = {str(x) for x in lambda_interpretations}
    if "dgp_term_multiplier" in interpretation_set or "quadratic_utility_weight" in interpretation_set:
        return "zero-multiplier baseline"
    if baseline_field == "local_law_weight":
        return "no-local-law baseline"
    return "configured baseline"


def _paper_safety_reason(row: dict, family: FamilyReportConfig) -> Optional[str]:
    interpretation = str(row.get("objective_lambda_interpretation", "")).strip()
    if interpretation and interpretation in set(str(x) for x in family.disallowed_lambda_interpretations):
        return f"disallowed_lambda_interpretation:{interpretation}"
    expected_package = str(family.expected_main_package or "").strip()
    baseline_field = str(family.baseline_field or family.sweep_field)
    baseline_value = family.baseline_value
    is_baseline_row = (
        baseline_value is not None
        and _value_matches(row.get(baseline_field), baseline_value)
    )
    if expected_package:
        observed_package = str(row.get("law_set_id", "") or "unknown").strip()
        if (not is_baseline_row) and observed_package != expected_package:
            return f"unexpected_law_set:{observed_package or 'unknown'}"
    component_weights = dict(row.get("objective_local_law_component_weights", {}) or {})
    for field in family.required_local_law_weight_fields:
        try:
            value = float(component_weights.get(field, row.get(field, float("nan"))))
        except Exception:
            value = float("nan")
        if not np.isfinite(value):
            return f"missing_explicit_local_law_weight:{field}"
    return None


def _apply_paper_safe_filter(
    rows: Sequence[dict],
    family: FamilyReportConfig,
) -> Tuple[List[dict], Dict[str, int]]:
    kept: List[dict] = []
    excluded: Dict[str, int] = {}
    for row in rows:
        reason = _paper_safety_reason(row, family)
        if reason is None:
            kept.append(dict(row))
            continue
        excluded[reason] = int(excluded.get(reason, 0)) + 1
    return kept, excluded


def _write_excluded_report(
    *,
    title: str,
    family: FamilyReportConfig,
    input_root: Path,
    output_dir: Path,
    pdf_path: Path,
    rows_loaded: int,
    paper_safe_exclusions: Dict[str, int],
    args: argparse.Namespace,
) -> None:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    summary = {
        "generated_at": generated_at,
        "family": family.family,
        "input_root": str(input_root),
        "status": "excluded",
        "normalize": bool(args.normalize),
        "paper_safe": bool(args.paper_safe),
        "rows_loaded_before_filter": int(rows_loaded),
        "run_count": 0,
        "paper_safe_exclusion_reasons": dict(sorted(paper_safe_exclusions.items())),
        "pdf": str(pdf_path),
    }
    summary_path = output_dir / "learnability_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    md_lines = [
        f"# {title}",
        "",
        "## Status",
        "",
        "- This report is excluded from the paper-facing learnability bundle.",
        f"- Family: `{family.display_name}`",
        f"- Input root: `{input_root}`",
        f"- Rows loaded before filter: `{rows_loaded}`",
        f"- Paper-safe mode: `{bool(args.paper_safe)}`",
        "",
        "## Exclusion Reasons",
        "",
    ]
    for reason, count in sorted(paper_safe_exclusions.items()):
        md_lines.append(f"- `{reason}`: `{count}`")
    if str(args.status_note).strip():
        md_lines.extend(["", f"- Status note: `{args.status_note}`"])
    md_lines.extend(["", f"- PDF: `{pdf_path}`", ""])
    (output_dir / "learnability.md").write_text("\n".join(md_lines), encoding="utf-8")

    with PdfPages(pdf_path) as pdf:
        write_text_page(
            pdf,
            title=title,
            lines=[
                "Excluded from paper-facing learnability report.",
                f"Family: {family.display_name} ({family.family})",
                f"Input root: {input_root}",
                f"Rows loaded before filter: {rows_loaded}",
                "",
                "Exclusion reasons:",
                *[f"{reason}: {count}" for reason, count in sorted(paper_safe_exclusions.items())],
            ],
        )


def _baseline_field_name(family: FamilyReportConfig, override: Optional[str] = None) -> str:
    return str(override or family.baseline_field or family.sweep_field)


def _baseline_label(family: FamilyReportConfig, baseline_field: Optional[str] = None) -> str:
    field = str(baseline_field or family.baseline_field or family.sweep_field)
    if field == str(family.baseline_field or "") and family.baseline_label:
        return str(family.baseline_label)
    return field


def _available_field_values(rows: Sequence[dict], field: str) -> List[object]:
    return _dedup_axis_values([row.get(field) for row in rows if row.get(field) is not None])


def _resolve_baseline_value(
    rows: Sequence[dict],
    *,
    baseline_field: str,
    family: FamilyReportConfig,
    explicit_value: object | None,
) -> Tuple[object | None, str]:
    available = _available_field_values(rows, baseline_field)
    if not available:
        return None, "missing"
    if explicit_value is not None:
        if not any(_value_matches(value, explicit_value) for value in available):
            raise SystemExit(
                f"Requested base {baseline_field}={explicit_value!r} not present in loaded runs; "
                f"available values: {available}"
            )
        return explicit_value, "cli"
    family_default = family.baseline_value
    if family_default is not None and any(_value_matches(value, family_default) for value in available):
        return family_default, "family_default"
    return available[0], "observed_min"


def _best_row(
    rows: Sequence[dict],
    *,
    metric: str = "heldout_objective_for_report",
    **filters: object,
) -> Optional[dict]:
    subset = list(rows)
    for key, value in filters.items():
        if value is None:
            continue
        subset = [r for r in subset if _value_matches(r.get(key), value)]
    if not subset:
        return None
    if metric == "theorem_score":
        return min(subset, key=_row_theorem_score)
    if metric == "heldout_objective_for_report":
        return min(subset, key=_row_selection_objective)
    return min(subset, key=lambda r: float(r.get(metric, float("nan"))))


def _matched_baseline_row(
    rows: Sequence[dict],
    target: Optional[dict],
    family: FamilyReportConfig,
    *,
    baseline_field: str,
    baseline_value: object | None,
) -> Optional[dict]:
    """Find the sweep_value=0 (or minimum) baseline that matches target's config."""
    if target is None or baseline_value is None:
        return None
    sweep_field = family.sweep_field

    # Match on all group fields plus capacity
    def _matches(row: dict) -> bool:
        for gf in family.sweep_group_fields:
            if gf == baseline_field:
                continue
            if gf not in target or gf not in row:
                continue
            if not _value_matches(row.get(gf), target.get(gf)):
                return False
        if baseline_field != sweep_field and sweep_field in target and sweep_field in row:
            if not _value_matches(row.get(sweep_field), target.get(sweep_field)):
                return False
        # Match capacity if present
        for cap_field in ("state_dim", "hidden_dim", "n_epochs", "feature_mode"):
            if cap_field in target and cap_field in row:
                if str(row[cap_field]) != str(target[cap_field]):
                    return False
        if not _value_matches(row.get(baseline_field), baseline_value):
            return False
        return True

    matched = [r for r in rows if _matches(r)]
    if not matched:
        return None
    return min(matched, key=_row_selection_objective)


def _best_task_only_row(
    rows: Sequence[dict],
    family: FamilyReportConfig,
    *,
    baseline_field: str,
    baseline_value: object | None,
) -> Optional[dict]:
    if baseline_value is None:
        return None
    candidates = [
        row
        for row in rows
        if _value_matches(row.get(baseline_field), baseline_value)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: float(r.get("learned_root_mae_n", float("nan"))))


# ── formatting helpers ───────────────────────────────────────────────────


def _format_axis_values(values: Sequence, *, audit: bool = False) -> str:
    formatted = []
    for v in values:
        if audit:
            formatted.append(_format_pct(float(v)))
        elif isinstance(v, (int, np.integer)):
            formatted.append(str(int(v)))
        elif isinstance(v, (float, np.floating)):
            formatted.append(_format_weight(float(v)))
        else:
            formatted.append(str(v))
    return "[" + ", ".join(formatted) + "]"


def _format_row_brief(row: dict, family: FamilyReportConfig) -> str:
    parts = []
    for gf in family.sweep_group_fields:
        if gf in row:
            val = row[gf]
            if gf == "audit_fraction":
                parts.append(f"q_audit={_format_pct(float(val))}")
            else:
                parts.append(f"{gf}={val}")
    baseline_field = str(family.baseline_field or "")
    if baseline_field and baseline_field not in family.sweep_group_fields and baseline_field in row:
        parts.append(f"{baseline_field}={row[baseline_field]}")
    sweep_field = family.sweep_field
    parts.append(f"{sweep_field}={_format_weight(float(row.get(sweep_field, 0.0)))}")
    parts.append(f"n={row.get('n_runs', 1)}")
    parts.append(f"objective={_row_selection_objective(row):.4f}")
    parts.append(f"theorem={_row_theorem_score(row):.4f}")
    parts.append(f"leaf={float(row.get('learned_leaf_mae_n', float('nan'))):.4f}")
    parts.append(f"merge={float(row.get('learned_merge_mae_n', float('nan'))):.4f}")
    parts.append(f"sensitivity={float(row.get('learned_spread_n', float('nan'))):.4f}")
    parts.append(f"root={float(row.get('learned_root_mae_n', float('nan'))):.4f}")
    return " | ".join(parts)


def _format_operating_point(label: str, row: Optional[dict], family: FamilyReportConfig) -> str:
    if row is None:
        return f"{label}: unavailable"
    return f"{label}: {_format_row_brief(row, family)}"


def _capacity_slug(cap: CapacityKey) -> str:
    return f"sd_{cap[0]}__hd_{cap[1]}__ep_{cap[2]}__fm_{str(cap[3]).replace('-', '_')}"


# ── takeaway lines ──────────────────────────────────────────────────────


def _takeaway_lines(
    *,
    family: FamilyReportConfig,
    recommended_sparse: Optional[dict],
    recommended_full: Optional[dict],
    sparse_baseline: Optional[dict],
    sparse_no_secondary: Optional[dict],
    best_task_only: Optional[dict],
    best_by_objective: Optional[dict],
    best_by_root: Optional[dict],
    baseline_field: str,
    baseline_label: str,
    baseline_value: object | None,
    lambda_interpretations: Sequence[str],
    axes: dict,
    n_rows: int,
    exact_obj_count: int,
) -> List[str]:
    sweep_label = family.sweep_label
    primary_label = family.primary_metric_label
    lines: List[str] = []
    baseline_noun = _baseline_noun(
        baseline_field=baseline_field,
        lambda_interpretations=lambda_interpretations,
    )

    # Group fields for budget description
    gf = family.sweep_group_fields
    s1_field = gf[0] if gf else "train_docs"
    s2_field = gf[1] if len(gf) > 1 else None

    lines.append(
        f"**Baseline reference**: {baseline_label}={_format_scalar(baseline_value)} is the {baseline_noun} when present."
    )

    if recommended_sparse is not None and sparse_baseline is not None:
        root_base = float(sparse_baseline.get("learned_root_mae_n", float("nan")))
        root_opt = float(recommended_sparse.get("learned_root_mae_n", float("nan")))
        if np.isfinite(root_base) and np.isfinite(root_opt) and root_base > 0:
            root_change = 100.0 * (root_opt - root_base) / root_base
            lines.append(
                f"**Downstream safety**: at the strongest sparse-audit budget, "
                f"{primary_label} moves from {root_base:.4f} to {root_opt:.4f} "
                f"({root_change:+.1f}%) between the matched baseline and the objective-optimal setting."
            )

    if recommended_sparse is not None and sparse_baseline is not None:
        if _baseline_field_name(family) == family.sweep_field:
            baseline_text = f"{baseline_label}={_format_scalar(baseline_value)}"
            sv_field = family.sweep_field
            lines.append(
                f"**Learnability**: moving from {baseline_text} to the objective-optimal setting "
                f"({sweep_label}={_format_weight(float(recommended_sparse.get(sv_field, 0)))}) "
                f"cuts held-out C1 from {float(sparse_baseline.get('learned_leaf_mae_n', float('nan'))):.4f} "
                f"to {float(recommended_sparse.get('learned_leaf_mae_n', float('nan'))):.4f} "
                f"and C3 from {float(sparse_baseline.get('learned_merge_mae_n', float('nan'))):.4f} "
                f"to {float(recommended_sparse.get('learned_merge_mae_n', float('nan'))):.4f}."
            )
        else:
            lines.append(
                f"**Matched baseline comparison**: at fixed {sweep_label}, the report compares each row to the "
                f"matched {baseline_label}={_format_scalar(baseline_value)} baseline rather than treating the minimum {sweep_label} as the control."
            )

    if recommended_sparse is not None and recommended_full is not None:
        audit_vals = axes.get("audit_fraction", [])
        if len(audit_vals) >= 2:
            lines.append(
                f"**Audit efficiency**: sparse and full audit are nearly matched: "
                f"{_format_pct(min(audit_vals))} audit gives {_row_selection_objective(recommended_sparse):.4f} "
                f"and {_format_pct(max(audit_vals))} audit gives {_row_selection_objective(recommended_full):.4f}."
            )

    if best_task_only is not None and best_by_root is not None:
        baseline_root = float(best_task_only.get("learned_root_mae_n", float("nan")))
        best_root = float(best_by_root.get("learned_root_mae_n", float("nan")))
        if np.isfinite(baseline_root) and np.isfinite(best_root):
            best_task_base = best_task_only.get(baseline_field)
            best_root_base = best_by_root.get(baseline_field)
            if _value_matches(best_task_base, best_root_base):
                lines.append(
                    f"**Downstream optimum**: the lowest held-out {primary_label} is achieved at the {baseline_noun} "
                    f"({baseline_label}={_format_scalar(best_root_base)})."
                )
            else:
                lines.append(
                    f"**Downstream vs objective**: the lowest held-out {primary_label} occurs at "
                    f"{baseline_label}={_format_scalar(best_root_base)}, while the configured objective optimum is "
                    f"{sweep_label}={_format_weight(float(best_by_objective.get(family.sweep_field, best_by_objective.get('sweep_value', 0.0))))}."
                )

    if exact_obj_count > 0:
        lines.append(
            f"Exact held-out weighted objectives present for {exact_obj_count}/{n_rows} raw runs."
        )
    return lines


# ── main ─────────────────────────────────────────────────────────────────


def main() -> int:
    args = _parse_args()
    family = resolve_family(args.family)
    input_root = Path(args.input_root)
    title = args.title or f"{family.display_name} Local-Law Learnability"
    output_dir = Path(args.output_dir) if args.output_dir else (input_root / "local_law_report")
    output_dir.mkdir(parents=True, exist_ok=True)
    agg = str(args.aggregate)
    pdf_path = Path(args.pdf_path) if args.pdf_path else (output_dir / "learnability_report.pdf")

    # ── load data ────────────────────────────────────────────────────
    rows = load_learnability_records(input_root, family, do_normalize=bool(args.normalize))
    if not rows:
        raise SystemExit(f"No valid learnability runs loaded from {input_root}")
    rows_loaded_before_filter = len(rows)
    paper_safe_exclusions: Dict[str, int] = {}
    if bool(args.paper_safe):
        rows, paper_safe_exclusions = _apply_paper_safe_filter(rows, family)
        if not rows:
            _write_excluded_report(
                title=title,
                family=family,
                input_root=input_root,
                output_dir=output_dir,
                pdf_path=pdf_path,
                rows_loaded=rows_loaded_before_filter,
                paper_safe_exclusions=paper_safe_exclusions,
                args=args,
            )
            return 0

    baseline_field = _baseline_field_name(family, args.base_field)
    baseline_label = _baseline_label(family, baseline_field)
    baseline_value, baseline_value_source = _resolve_baseline_value(
        rows,
        baseline_field=baseline_field,
        family=family,
        explicit_value=_parse_scalar(args.base_value),
    )

    # ── discover axes ────────────────────────────────────────────────
    sweep_field = family.sweep_field
    axes: Dict[str, list] = {}
    for field in [sweep_field, baseline_field, "train_docs", "audit_fraction"] + list(family.sweep_group_fields):
        vals = _dedup_axis_values([row.get(field) for row in rows if row.get(field) is not None])
        if vals:
            axes[field] = vals
    for field in ["schedule_consistency_weight", "state_dim", "hidden_dim", "n_epochs",
                  "feature_mode", "effective_data_seed", "effective_model_seed"]:
        vals = _dedup_axis_values([row.get(field) for row in rows if row.get(field) is not None])
        if vals:
            axes[field] = vals

    # ── aggregate ────────────────────────────────────────────────────
    # Group keys: sweep variable + group fields + capacity
    group_keys = [sweep_field]
    for gf in family.sweep_group_fields:
        if gf not in group_keys:
            group_keys.append(gf)
    for cap_field in ["state_dim", "hidden_dim", "n_epochs", "feature_mode"]:
        if cap_field in rows[0] and cap_field not in group_keys:
            group_keys.append(cap_field)
    # Add Markov-specific fields if present and multi-valued
    for extra in ["schedule_consistency_weight", "root_share"]:
        if extra in rows[0] and extra not in group_keys:
            group_keys.append(extra)

    aggregated_rows = aggregate_learnability(rows, group_keys=group_keys, agg=agg)

    # ── operating points ─────────────────────────────────────────────
    best_by_objective = min(aggregated_rows, key=_row_selection_objective)
    best_by_theorem = min(aggregated_rows, key=_row_theorem_score)
    best_by_root = min(aggregated_rows, key=lambda r: float(r.get("learned_root_mae_n", float("nan"))))
    best_task_only = _best_task_only_row(
        aggregated_rows, family, baseline_field=baseline_field, baseline_value=baseline_value
    )
    top_by_objective = sorted(aggregated_rows, key=_row_selection_objective)[:10]

    # Recommended operating points
    gf = tuple(field for field in family.sweep_group_fields if field != baseline_field)
    if not gf:
        gf = family.sweep_group_fields
    s1_field = gf[0] if gf else "train_docs"
    s2_field = gf[1] if len(gf) > 1 else None
    s1_vals = axes.get(s1_field, [])
    s2_vals = axes.get(s2_field, []) if s2_field else []
    audit_vals = axes.get("audit_fraction", [1.0])

    max_s1 = max(s1_vals) if s1_vals else None
    max_s2 = max(s2_vals) if s2_vals else None
    min_audit = min(audit_vals) if audit_vals else None
    max_audit = max(audit_vals) if audit_vals else None

    sparse_filters = {s1_field: max_s1}
    if s2_field and max_s2 is not None:
        sparse_filters[s2_field] = float(max_s2)
    if min_audit is not None:
        sparse_filters["audit_fraction"] = float(min_audit)

    full_filters = dict(sparse_filters)
    if max_audit is not None:
        full_filters["audit_fraction"] = float(max_audit)

    recommended_sparse = _best_row(aggregated_rows, **sparse_filters)
    recommended_full = _best_row(aggregated_rows, **full_filters)
    if recommended_sparse is None or recommended_full is None:
        size_filters = {k: v for k, v in sparse_filters.items() if k != "audit_fraction"}
        size_candidates = [
            row for row in aggregated_rows
            if all(
                _value_matches(row.get(key), value)
                for key, value in size_filters.items()
            )
        ]
        available_audits = sorted({
            float(row.get("audit_fraction", float("nan")))
            for row in size_candidates
            if np.isfinite(float(row.get("audit_fraction", float("nan"))))
        })
        if available_audits:
            if recommended_sparse is None:
                recommended_sparse = _best_row(
                    aggregated_rows,
                    audit_fraction=float(min(available_audits)),
                    **size_filters,
                )
            if recommended_full is None:
                recommended_full = _best_row(
                    aggregated_rows,
                    audit_fraction=float(max(available_audits)),
                    **size_filters,
                )
    sparse_baseline = _matched_baseline_row(
        aggregated_rows,
        recommended_sparse,
        family,
        baseline_field=baseline_field,
        baseline_value=baseline_value,
    )
    full_baseline = _matched_baseline_row(
        aggregated_rows,
        recommended_full,
        family,
        baseline_field=baseline_field,
        baseline_value=baseline_value,
    )

    # No-secondary-group variant (for schedule regularization comparison)
    min_s2 = min(s2_vals) if s2_vals else None
    sparse_no_secondary = None
    if s2_field and min_s2 is not None and max_s2 is not None and not np.isclose(float(min_s2), float(max_s2)):
        no_sec_filters = dict(sparse_filters)
        no_sec_filters[s2_field] = float(min_s2)
        sparse_no_secondary = _best_row(aggregated_rows, **no_sec_filters)

    exact_obj_count = sum(1 for r in rows if np.isfinite(float(r.get("test_objective_full_labels", float("nan")))))
    max_group_runs = max((int(r.get("n_runs", 1)) for r in aggregated_rows), default=0)
    partial_group_count = sum(int(r.get("n_runs", 1)) < max_group_runs for r in aggregated_rows)

    # ── figures ──────────────────────────────────────────────────────
    capacity_keys = sorted({_capacity_key(r) for r in rows})
    show_fm = len({c[3] for c in capacity_keys}) > 1
    supports_sweep_gain_baseline = baseline_value is not None

    # Panel and series field config
    panel_field = "audit_fraction"
    series_fields = tuple(family.sweep_group_fields)

    primary_label = family.primary_metric_label
    heldout_core_defs = [
        ("learned_root_mae_n", f"Held-out {primary_label} (primary)", "normalized error"),
        ("learned_leaf_mae_n", "Held-out C1 / leaf MAE", "normalized error"),
        ("learned_merge_mae_n", "Held-out C3 / merge MAE", "normalized error"),
        ("learned_law_score_n", "Held-out theorem score", "normalized theorem error"),
    ]
    heldout_stability_defs = [
        ("learned_spread_n", "Held-out merge-order sensitivity", "normalized error"),
    ]
    gain_core_defs = [
        ("learned_root_mae_n", f"{primary_label} gain (primary)", f"gain vs {baseline_label}={_format_scalar(baseline_value)}"),
        ("learned_leaf_mae_n", "C1 gain", f"gain vs {baseline_label}={_format_scalar(baseline_value)}"),
        ("learned_merge_mae_n", "C3 gain", f"gain vs {baseline_label}={_format_scalar(baseline_value)}"),
        ("learned_law_score_n", "Theorem-score gain", f"gain vs {baseline_label}={_format_scalar(baseline_value)}"),
    ] if supports_sweep_gain_baseline else []
    gain_stability_defs = [
        ("learned_spread_n", "Sensitivity gain", f"gain vs {baseline_label}={_format_scalar(baseline_value)}"),
    ] if supports_sweep_gain_baseline else []

    figure_paths: List[str] = []
    figure_titles: Dict[str, str] = {}
    figure_specs = []

    for cap in capacity_keys:
        slug = _capacity_slug(cap)
        cap_label = _capacity_label(cap, show_fm=show_fm)
        figure_specs.extend([
            (
                output_dir / f"heldout_core_grid_{slug}.png",
                f"Held-out {primary_label}, C1, C3, theorem vs {family.sweep_label} | {cap_label}",
                lambda p, c=cap: plot_sweep_grid(
                    rows, family=family, output_path=p,
                    metric_defs=heldout_core_defs,
                    title_prefix=f"Held-out {primary_label} (primary), C1, C3, theorem score vs {family.sweep_label}",
                    panel_field=panel_field, series_fields=series_fields, capacity=c,
                ),
            ),
            (
                output_dir / f"heldout_stability_grid_{slug}.png",
                f"Held-out sensitivity vs {family.sweep_label} | {cap_label}",
                lambda p, c=cap: plot_sweep_grid(
                    rows, family=family, output_path=p,
                    metric_defs=heldout_stability_defs,
                    title_prefix=f"Held-out merge-order sensitivity vs {family.sweep_label}",
                    panel_field=panel_field, series_fields=series_fields, capacity=c,
                ),
            ),
            (
                output_dir / f"theorem_opt_audit_summary_{slug}.png",
                f"Sparse vs full audit at objective-optimal {family.sweep_label} | {cap_label}",
                lambda p, c=cap: plot_audit_summary(
                    aggregated_rows, family=family, output_path=p,
                    best_row_fn=lambda rows, **kw: _best_row(rows, **kw),
                    theorem_score_fn=_row_theorem_score,
                    capacity=c,
                ),
            ),
            (
                output_dir / f"optimization_appendix_{slug}.png",
                f"Optimization appendix | {cap_label}",
                lambda p, c=cap: plot_optimization_appendix(
                    rows, family=family, output_path=p,
                    panel_field=panel_field, series_fields=series_fields, capacity=c,
                ),
            ),
        ])
        if supports_sweep_gain_baseline:
            figure_specs.extend([
                (
                    output_dir / f"heldout_gain_core_{slug}.png",
                    f"{primary_label}, C1, C3, theorem gains vs {baseline_label}={_format_scalar(baseline_value)} | {cap_label}",
                    lambda p, c=cap: plot_gain_grid(
                        rows, family=family, output_path=p,
                        metric_defs=gain_core_defs,
                        title_prefix=f"{primary_label} (primary), C1, C3, theorem gains vs {baseline_label}={_format_scalar(baseline_value)}",
                        panel_field=panel_field, series_fields=series_fields, capacity=c,
                        baseline_field=baseline_field,
                        baseline_value=baseline_value,
                    ),
                ),
                (
                    output_dir / f"heldout_gain_stability_{slug}.png",
                    f"Sensitivity gain vs {baseline_label}={_format_scalar(baseline_value)} | {cap_label}",
                    lambda p, c=cap: plot_gain_grid(
                        rows, family=family, output_path=p,
                        metric_defs=gain_stability_defs,
                        title_prefix=f"Sensitivity gain vs {baseline_label}={_format_scalar(baseline_value)}",
                        panel_field=panel_field, series_fields=series_fields, capacity=c,
                        baseline_field=baseline_field,
                        baseline_value=baseline_value,
                    ),
                ),
            ])

    if len(capacity_keys) > 1:
        figure_specs.append((
            output_dir / "capacity_summary.png",
            f"Capacity summary at objective-optimal {family.sweep_label}",
            lambda p: plot_capacity_summary(
                aggregated_rows, family=family, output_path=p,
                selection_objective_fn=_row_selection_objective,
                theorem_score_fn=_row_theorem_score,
            ),
        ))

    for path, fig_title, render_fn in figure_specs:
        render_fn(path)
        if path.exists():
            figure_paths.append(str(path))
            figure_titles[str(path)] = fig_title

    # ── takeaways ────────────────────────────────────────────────────
    takeaways = _takeaway_lines(
        family=family,
        recommended_sparse=recommended_sparse,
        recommended_full=recommended_full,
        sparse_baseline=sparse_baseline,
        sparse_no_secondary=sparse_no_secondary,
        best_task_only=best_task_only,
        best_by_objective=best_by_objective,
        best_by_root=best_by_root,
        baseline_field=baseline_field,
        baseline_label=baseline_label,
        baseline_value=baseline_value,
        lambda_interpretations=_objective_lambda_interpretations(rows),
        axes=axes,
        n_rows=len(rows),
        exact_obj_count=exact_obj_count,
    )

    weighting_schemes = sorted({
        str(row.get("objective_weighting_scheme", "")).strip()
        for row in rows
        if str(row.get("objective_weighting_scheme", "")).strip()
    })
    root_share_sources = sorted({
        str(row.get("root_share_source", "")).strip()
        for row in rows
        if str(row.get("root_share_source", "")).strip()
    })
    lambda_interpretations = _objective_lambda_interpretations(rows)
    baseline_noun = _baseline_noun(
        baseline_field=baseline_field,
        lambda_interpretations=lambda_interpretations,
    )

    # ── JSON summary ─────────────────────────────────────────────────
    expected_run_count = int(args.expected_run_count) if args.expected_run_count else None
    completion_fraction = float(len(rows)) / float(expected_run_count) if expected_run_count else None

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "family": family.family,
        "input_root": str(input_root),
        "aggregate": agg,
        "normalize": bool(args.normalize),
        "expected_run_count": expected_run_count,
        "completion_fraction": completion_fraction,
        "status_note": str(args.status_note),
        "paper_safe": bool(args.paper_safe),
        "rows_loaded_before_filter": int(rows_loaded_before_filter),
        "run_count": len(rows),
        "axes": axes,
        "capacity_keys": [
            {"state_dim": int(c[0]), "hidden_dim": int(c[1]), "n_epochs": int(c[2]), "feature_mode": str(c[3])}
            for c in capacity_keys
        ],
        "max_group_runs": max_group_runs,
        "partial_group_count": partial_group_count,
        "exact_test_objective_row_count": exact_obj_count,
        "selection_metric_name": "heldout_objective_for_report",
        "baseline": {
            "field": baseline_field,
            "label": baseline_label,
            "value": baseline_value,
            "source": baseline_value_source,
        },
        "baseline_axis_name": baseline_field,
        "baseline_axis_label": baseline_label,
        "baseline_value_source": baseline_value_source,
        "baseline_sweep_value": baseline_value,
        "best_baseline_point": best_task_only,
        "best_no_local_law_point": best_task_only,
        "best_by_objective": best_by_objective,
        "best_by_theorem_score": best_by_theorem,
        "best_by_root": best_by_root,
        "recommended_sparse_objective_point": recommended_sparse,
        "recommended_full_objective_point": recommended_full,
        "matched_sparse_baseline": sparse_baseline,
        "matched_full_baseline": full_baseline,
        "key_takeaways": takeaways,
        "top_rows_by_objective": top_by_objective,
        "aggregated_rows": aggregated_rows,
        "figures": figure_paths,
        "figure_titles": figure_titles,
        "objective_weighting_schemes": weighting_schemes,
        "objective_lambda_interpretations": lambda_interpretations,
        "root_share_sources": root_share_sources,
        "paper_safe_exclusion_reasons": dict(sorted(paper_safe_exclusions.items())),
    }
    (output_dir / "learnability_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8",
    )
    write_csv(output_dir / "learnability_rows.csv", [dict(r) for r in rows])

    # ── Markdown ─────────────────────────────────────────────────────
    sweep_label = family.sweep_label
    md_lines = [
        f"# {title}",
        "",
        "## Scope",
        "",
        f"- **Purpose**: This report summarizes downstream {primary_label} and local-law metrics across the family sweep, "
        f"using the configured baseline for matched comparisons.",
        f"- **Primary metric**: held-out {primary_label} (shown first in all grids). "
        "The theorem score (C1+C3+sensitivity) is supporting evidence for law learnability.",
        f"- **Baseline comparison**: `{baseline_label}={_format_scalar(baseline_value)}` is the {baseline_noun} when present.",
        "- For the cross-DGP ablation story (which laws drive downstream gains), see the law stress report.",
        f"- Input root: `{input_root}`",
        f"- Runs loaded: `{len(rows)}`",
        f"- Rows loaded before filter: `{rows_loaded_before_filter}`",
        f"- Aggregation across seeds: `{agg}`",
        f"- Normalized metrics: `{bool(args.normalize)}`",
        f"- Paper-safe mode: `{bool(args.paper_safe)}`",
        f"- Theorem score: `leaf_mae + merge_mae + {THEOREM_SCORE_SPREAD_WEIGHT} * merge_order_sensitivity` (not including {primary_label}).",
        "",
    ]
    if expected_run_count is not None:
        md_lines.extend([
            f"- Expected run count: `{expected_run_count}`",
            f"- Completion: `{completion_fraction:.3%}`" if completion_fraction is not None else "- Completion: `n/a`",
            "",
        ])
    if str(args.status_note).strip():
        md_lines.extend([f"- Status note: `{args.status_note}`", ""])
    md_lines.extend([f"- Baseline source: `{baseline_value_source}`", ""])
    if weighting_schemes:
        md_lines.extend([f"- Objective weighting scheme(s): `{', '.join(weighting_schemes)}`", ""])
    if lambda_interpretations:
        md_lines.extend([f"- Objective lambda interpretation(s): `{', '.join(lambda_interpretations)}`", ""])
    if "dgp_term_multiplier" in lambda_interpretations or "quadratic_utility_weight" in lambda_interpretations:
        md_lines.extend([
            f"- **Interpretation note**: `{baseline_label}` is a quadratic-utility multiplier, not the paper local-law lambda, so values above `1` are valid in this family.",
            "",
        ])
    if root_share_sources:
        md_lines.extend([f"- Root-share source(s): `{', '.join(root_share_sources)}`", ""])
    if paper_safe_exclusions:
        md_lines.extend([
            "- Paper-safe exclusions:",
            *[f"  - `{reason}`: `{count}`" for reason, count in sorted(paper_safe_exclusions.items())],
            "",
        ])

    md_lines.extend(["## Coverage", ""])
    for field, vals in axes.items():
        is_audit = (field == "audit_fraction")
        md_lines.append(f"- `{field}`: `{_format_axis_values(vals, audit=is_audit)}`")
    md_lines.extend([
        f"- Partial groups: `{partial_group_count}`",
        "",
        "## Key Takeaways",
        "",
        *[f"- {line}" for line in takeaways],
        "",
        "## Recommended Operating Points",
        "",
        f"- `{_format_operating_point('Best baseline point', best_task_only, family)}`",
        f"- `{_format_operating_point('Best root point (lowest downstream error)', best_by_root, family)}`",
        f"- `{_format_operating_point('Sparse objective point', recommended_sparse, family)}`",
        f"- `{_format_operating_point('Matched sparse baseline', sparse_baseline, family)}`",
        f"- `{_format_operating_point('Full objective point', recommended_full, family)}`",
        f"- `{_format_operating_point('Matched full baseline', full_baseline, family)}`",
        f"- `{_format_operating_point('Overall best objective', best_by_objective, family)}`",
        f"- `{_format_operating_point('Overall best theorem', best_by_theorem, family)}`",
        "",
        "## Top Rows By Selection Objective",
        "",
        *[f"- `{_format_row_brief(row, family)}`" for row in top_by_objective[:5]],
        "",
        "## Figures",
        "",
    ])
    for fig in figure_paths:
        md_lines.append(f"- {figure_titles.get(fig, Path(fig).name)}: `{fig}`")
    md_lines.append(f"- PDF: `{pdf_path}`")
    (output_dir / "learnability.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # ── PDF ──────────────────────────────────────────────────────────
    completion_line = (
        f"completion: {len(rows)} / {expected_run_count} ({completion_fraction:.2%})"
        if expected_run_count is not None and completion_fraction is not None
        else f"runs loaded: {len(rows)}"
    )
    status_lines = [
        f"Generated at UTC: {summary['generated_at']}",
        f"Family: {family.display_name} ({family.family})",
        f"Input root: {input_root}",
        completion_line,
        f"Rows loaded before filter: {rows_loaded_before_filter}",
        "",
        "PURPOSE",
        f"This report summarizes downstream {primary_label} and local-law metrics across the family sweep,",
        "using the configured baseline for matched comparisons.",
        f"{primary_label} is the PRIMARY metric and appears first in all grids.",
        "",
        f"Aggregation: {agg} | normalized: {bool(args.normalize)}",
        f"Paper-safe mode: {bool(args.paper_safe)}",
    ]
    for field, vals in axes.items():
        is_audit = (field == "audit_fraction")
        status_lines.append(f"{field}: {_format_axis_values(vals, audit=is_audit)}")
    status_lines.extend([
        "",
        "Definitions",
        f"Primary metric = held-out {primary_label} (downstream task error, shown first).",
        "Selection objective = held-out configured objective when present, else theorem proxy.",
        f"Theorem score = held-out leaf MAE + held-out merge MAE + {THEOREM_SCORE_SPREAD_WEIGHT} * sensitivity (NOT including {primary_label}).",
        f"{baseline_label}={_format_scalar(baseline_value)} is the {baseline_noun} when present.",
        f"Sweep variable = {sweep_label}.",
    ])
    status_lines.append(f"Baseline source: {baseline_value_source}.")
    if weighting_schemes:
        status_lines.append(f"Objective weighting scheme(s): {', '.join(weighting_schemes)}.")
    if lambda_interpretations:
        status_lines.append(f"Objective lambda interpretation(s): {', '.join(lambda_interpretations)}.")
    if "dgp_term_multiplier" in lambda_interpretations or "quadratic_utility_weight" in lambda_interpretations:
        status_lines.append(
            f"{baseline_label} is a quadratic-utility multiplier, not the paper local-law lambda, so values above 1 are valid in this family."
        )
    if root_share_sources:
        status_lines.append(f"Root-share source(s): {', '.join(root_share_sources)}.")
    if paper_safe_exclusions:
        status_lines.append("Paper-safe exclusions:")
        status_lines.extend(
            [f"  {reason}: {count}" for reason, count in sorted(paper_safe_exclusions.items())]
        )
    if str(args.status_note).strip():
        status_lines.extend(["", "status_note:", str(args.status_note)])

    operating_lines = [
        "Key takeaways",
        *takeaways,
        "",
        "Recommended operating points",
        _format_operating_point("Best baseline point", best_task_only, family),
        _format_operating_point("Sparse objective point", recommended_sparse, family),
        _format_operating_point("Matched sparse baseline", sparse_baseline, family),
        _format_operating_point("Full objective point", recommended_full, family),
        _format_operating_point("Matched full baseline", full_baseline, family),
        _format_operating_point("Overall best objective", best_by_objective, family),
        _format_operating_point("Overall best theorem", best_by_theorem, family),
        _format_operating_point("Best root point", best_by_root, family),
    ]

    with PdfPages(pdf_path) as pdf:
        write_text_page(pdf, title=title, lines=status_lines)
        write_text_page(pdf, title=f"{title} | Operating Points", lines=operating_lines)
        for fig in figure_paths:
            write_image_page(pdf, image_path=Path(fig), title=figure_titles.get(fig, Path(fig).name))

    summary["pdf"] = str(pdf_path)
    (output_dir / "learnability_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "pdf": str(pdf_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
