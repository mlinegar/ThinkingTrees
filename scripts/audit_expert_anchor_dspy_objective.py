#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


class SummaryAuditError(ValueError):
    """Raised when a DSPy training-record summary violates the anchored objective."""


def _as_float(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SummaryAuditError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise SummaryAuditError(f"{field} is not finite: {value!r}")
    return parsed


def _assert_close(actual: float, expected: float, *, field: str, tol: float = 1e-9) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol):
        raise SummaryAuditError(f"{field} expected {expected}, got {actual}")


def _require_objective(
    objective: Mapping[str, Any],
    *,
    root_label_sources: Sequence[str],
    root_label_target: str,
    root_share: float,
    local_law_weight: float,
    node_weight_normalization: str,
    target_min: Optional[float],
    target_max: Optional[float],
    scorer_output_min: Optional[float],
    scorer_output_max: Optional[float],
) -> None:
    expected_strings = {"node_weight_normalization": node_weight_normalization}
    for key, expected in expected_strings.items():
        actual = str(objective.get(key) or "")
        if actual != expected:
            raise SummaryAuditError(f"objective.{key} expected {expected!r}, got {actual!r}")
    actual_sources = tuple(str(source) for source in list(objective.get("root_label_sources") or []))
    expected_sources = tuple(str(source) for source in tuple(root_label_sources or ()))
    if actual_sources != expected_sources:
        raise SummaryAuditError(
            f"objective.root_label_sources expected {expected_sources!r}, got {actual_sources!r}"
        )
    actual_target = str(objective.get("root_label_target") or "")
    if actual_target != str(root_label_target):
        raise SummaryAuditError(
            f"objective.root_label_target expected {root_label_target!r}, got {actual_target!r}"
        )

    expected_numbers = {
        "root_share": root_share,
        "local_law_weight": local_law_weight,
        "target_min": target_min,
        "target_max": target_max,
        "scorer_output_min": scorer_output_min,
        "scorer_output_max": scorer_output_max,
    }
    for key, expected in expected_numbers.items():
        if expected is None:
            continue
        actual = _as_float(objective.get(key), field=f"objective.{key}")
        _assert_close(actual, expected, field=f"objective.{key}")
    component_weights = objective.get("local_law_component_weights") or {}
    if not isinstance(component_weights, Mapping):
        raise SummaryAuditError("objective.local_law_component_weights must be an object")
    _assert_close(
        _as_float(
            component_weights.get("teacher_node"),
            field="objective.local_law_component_weights.teacher_node",
        ),
        local_law_weight,
        field="objective.local_law_component_weights.teacher_node",
    )


def _validate_local_law_weight(value: float) -> float:
    parsed = _as_float(value, field="local_law_weight")
    if parsed < 0.0 or parsed > 1.0:
        raise SummaryAuditError(f"local_law_weight must be in [0, 1], got {value!r}")
    return float(parsed)


def _role_weight(bucket: Mapping[str, Any], *, role: str) -> float:
    return _as_float(bucket.get("weight"), field=f"by_law_role.{role}.weight")


def _role_count(bucket: Mapping[str, Any], *, role: str) -> int:
    raw = _as_float(bucket.get("count"), field=f"by_law_role.{role}.count")
    if not raw.is_integer():
        raise SummaryAuditError(f"by_law_role.{role}.count is not integral: {raw}")
    return int(raw)


def _optional_integral_count(summary: Mapping[str, Any], *, field: str) -> Optional[int]:
    if field not in summary or summary.get(field) is None:
        return None
    raw = _as_float(summary.get(field), field=field)
    if not raw.is_integer():
        raise SummaryAuditError(f"{field} is not integral: {raw}")
    parsed = int(raw)
    if parsed < 0:
        raise SummaryAuditError(f"{field} must be non-negative, got {parsed}")
    return parsed


def _target_sources(summary: Mapping[str, Any]) -> Iterable[str]:
    raw = summary.get("by_target_source") or {}
    if not isinstance(raw, Mapping):
        raise SummaryAuditError("by_target_source must be an object")
    return (str(key) for key in raw.keys())


def audit_summary(
    summary: Mapping[str, Any],
    *,
    expected_role: Optional[str] = None,
    root_label_sources: Sequence[str] = ("stored_summary",),
    root_label_target: str = "expert",
    local_law_weight: float = 0.25,
    node_weight_normalization: str = "per_tree",
    target_min: Optional[float] = 1.0,
    target_max: Optional[float] = 7.0,
    scorer_output_min: Optional[float] = 1.0,
    scorer_output_max: Optional[float] = 7.0,
) -> Dict[str, Any]:
    """Validate a written DSPy training-record summary for the anchored objective.

    The checker is intentionally summary-level: it audits the artifact that DSPy
    writes before optimization, without loading local teacher traces or making
    model calls.
    """

    role = str(expected_role or summary.get("role") or "").strip()
    if role not in {"f", "g"}:
        raise SummaryAuditError(f"role must be 'f' or 'g', got {role!r}")

    objective = summary.get("objective") or {}
    if not isinstance(objective, Mapping):
        raise SummaryAuditError("objective must be an object")
    teacher_local_law_weight = _validate_local_law_weight(float(local_law_weight))
    gold_weight = float(1.0 - teacher_local_law_weight)
    _require_objective(
        objective,
        root_label_sources=tuple(root_label_sources or ()),
        root_label_target=root_label_target,
        root_share=gold_weight,
        local_law_weight=teacher_local_law_weight,
        node_weight_normalization=node_weight_normalization,
        target_min=target_min,
        target_max=target_max,
        scorer_output_min=scorer_output_min,
        scorer_output_max=scorer_output_max,
    )

    by_role = summary.get("by_law_role") or {}
    if not isinstance(by_role, Mapping):
        raise SummaryAuditError("by_law_role must be an object")

    anchor_role = f"full_doc_{role}_anchor"
    if anchor_role not in by_role and gold_weight > 0.0:
        raise SummaryAuditError(f"missing required anchor role {anchor_role!r}")
    anchor_count = 0
    anchor_weight = 0.0
    if anchor_role in by_role:
        anchor_bucket = by_role[anchor_role]
        if not isinstance(anchor_bucket, Mapping):
            raise SummaryAuditError(f"by_law_role.{anchor_role} must be an object")
        anchor_count = _role_count(anchor_bucket, role=anchor_role)
        anchor_weight = _role_weight(anchor_bucket, role=anchor_role)
    if gold_weight > 0.0 and anchor_count <= 0:
        raise SummaryAuditError(f"{anchor_role} count must be positive")
    tree_count = _optional_integral_count(summary, field="tree_count")
    expected_tree_count = anchor_count if tree_count is None else tree_count
    _assert_close(
        anchor_weight,
        anchor_count * float(gold_weight),
        field=f"{anchor_role} total weight",
    )

    teacher_count = 0
    teacher_weight = 0.0
    teacher_roles = []
    for law_role, bucket in by_role.items():
        law_role = str(law_role)
        if law_role == anchor_role:
            continue
        if not isinstance(bucket, Mapping):
            raise SummaryAuditError(f"by_law_role.{law_role} must be an object")
        teacher_roles.append(law_role)
        teacher_count += _role_count(bucket, role=law_role)
        teacher_weight += _role_weight(bucket, role=law_role)

    if teacher_count <= 0 and teacher_local_law_weight > 0.0:
        raise SummaryAuditError("expected at least one teacher-node/local-law record")
    if expected_tree_count <= 0 and teacher_local_law_weight > 0.0:
        raise SummaryAuditError(
            "tree_count is required when local_law_weight=1 removes anchor records"
        )
    _assert_close(
        teacher_weight,
        expected_tree_count * float(teacher_local_law_weight),
        field="teacher-node/local-law total weight",
    )

    total_weight = _as_float(summary.get("total_weight"), field="total_weight")
    _assert_close(
        total_weight,
        anchor_weight + teacher_weight,
        field="total_weight",
    )

    observed_target_count = _as_float(summary.get("observed_target_count"), field="observed_target_count")
    _assert_close(observed_target_count, anchor_count, field="observed_target_count")

    targets = tuple(_target_sources(summary))
    if gold_weight > 0.0 and not any(target.startswith("expert:") for target in targets):
        raise SummaryAuditError("expected an expert:* target source for full-doc anchors")
    if teacher_local_law_weight > 0.0 and not any(target.startswith("teacher") for target in targets):
        raise SummaryAuditError("expected a teacher target source for local-law records")

    return {
        "status": "ok",
        "role": role,
        "anchor_role": anchor_role,
        "anchor_count": anchor_count,
        "anchor_weight": anchor_weight,
        "teacher_roles": sorted(teacher_roles),
        "teacher_count": teacher_count,
        "teacher_weight": teacher_weight,
        "tree_count": expected_tree_count,
        "root_share": gold_weight,
        "local_law_weight": teacher_local_law_weight,
        "expected_teacher_weight": expected_tree_count * float(teacher_local_law_weight),
        "total_weight": total_weight,
    }


def _load_summary(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SummaryAuditError(f"{path} did not contain a JSON object")
    return dict(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a DSPy training-record summary for the expert-anchor objective."
    )
    parser.add_argument("summary_json", type=Path)
    parser.add_argument("--role", choices=("f", "g"), default=None)
    parser.add_argument("--root-label-sources", default="stored_summary")
    parser.add_argument("--root-label-target", default="expert")
    parser.add_argument(
        "--local-law-weight",
        type=float,
        default=0.25,
        help="Expected canonical local-law mass λ.",
    )
    parser.add_argument("--node-weight-normalization", default="per_tree")
    parser.add_argument("--target-min", type=float, default=1.0)
    parser.add_argument("--target-max", type=float, default=7.0)
    parser.add_argument("--scorer-output-min", type=float, default=1.0)
    parser.add_argument("--scorer-output-max", type=float, default=7.0)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = audit_summary(
            _load_summary(args.summary_json),
            expected_role=args.role,
            root_label_sources=tuple(
                part.strip()
                for part in str(args.root_label_sources or "").split(",")
                if part.strip()
            ),
            root_label_target=str(args.root_label_target),
            local_law_weight=float(args.local_law_weight),
            node_weight_normalization=str(args.node_weight_normalization),
            target_min=float(args.target_min),
            target_max=float(args.target_max),
            scorer_output_min=float(args.scorer_output_min),
            scorer_output_max=float(args.scorer_output_max),
        )
    except SummaryAuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
