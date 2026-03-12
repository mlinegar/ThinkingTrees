from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class CompositeObjectiveSpec:
    name: str
    task_name: str
    task_weight: float
    local_law_weights: Dict[str, float] = field(default_factory=dict)
    proxy_weights: Dict[str, float] = field(default_factory=dict)
    weighting_scheme: str = "explicit_weighted_sum"
    task_weight_source: str = ""
    selection_metric_name: str = "configured_objective"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def total_weight_without_proxy(self) -> float:
        return float(self.task_weight + sum(float(v) for v in self.local_law_weights.values()))

    def to_dict(self) -> Dict[str, Any]:
        local_law_weight_total = float(sum(float(v) for v in dict(self.local_law_weights).values()))
        proxy_weight_total = float(sum(float(v) for v in dict(self.proxy_weights).values()))
        total_weight_without_proxy = float(self.total_weight_without_proxy())
        payload = asdict(self)
        payload["task_weight"] = float(self.task_weight)
        payload["local_law_weights"] = {
            str(k): float(v) for k, v in dict(self.local_law_weights).items()
        }
        payload["proxy_weights"] = {str(k): float(v) for k, v in dict(self.proxy_weights).items()}
        payload["local_law_weight_total"] = local_law_weight_total
        payload["proxy_weight_total"] = proxy_weight_total
        payload["total_weight_without_proxy"] = total_weight_without_proxy
        payload["normalized_task_share"] = (
            float(self.task_weight / total_weight_without_proxy)
            if total_weight_without_proxy > 0.0
            else float("nan")
        )
        payload["normalized_local_law_share"] = (
            float(local_law_weight_total / total_weight_without_proxy)
            if total_weight_without_proxy > 0.0
            else float("nan")
        )
        return payload


@dataclass(frozen=True)
class CompositeObjectiveEvaluation:
    total: float
    task_raw: float
    task_term: float
    local_law_raw: Dict[str, float] = field(default_factory=dict)
    local_law_terms: Dict[str, float] = field(default_factory=dict)
    proxy_raw: Dict[str, float] = field(default_factory=dict)
    proxy_terms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": float(self.total),
            "task_raw": float(self.task_raw),
            "task_term": float(self.task_term),
            "local_law_raw": {str(k): float(v) for k, v in dict(self.local_law_raw).items()},
            "local_law_terms": {str(k): float(v) for k, v in dict(self.local_law_terms).items()},
            "proxy_raw": {str(k): float(v) for k, v in dict(self.proxy_raw).items()},
            "proxy_terms": {str(k): float(v) for k, v in dict(self.proxy_terms).items()},
            "local_law_raw_total": float(sum(float(v) for v in self.local_law_raw.values())),
            "local_law_term_total": float(sum(float(v) for v in self.local_law_terms.values())),
            "proxy_raw_total": float(sum(float(v) for v in self.proxy_raw.values())),
            "proxy_term_total": float(sum(float(v) for v in self.proxy_terms.values())),
        }

    def to_flat_dict(self, *, prefix: str) -> Dict[str, float]:
        payload = {
            str(prefix): float(self.total),
            f"{prefix}_task_raw": float(self.task_raw),
            f"{prefix}_task_term": float(self.task_term),
            f"{prefix}_local_law_raw_total": float(
                sum(float(v) for v in self.local_law_raw.values())
            ),
            f"{prefix}_local_law_term_total": float(
                sum(float(v) for v in self.local_law_terms.values())
            ),
            f"{prefix}_proxy_raw_total": float(sum(float(v) for v in self.proxy_raw.values())),
            f"{prefix}_proxy_term_total": float(sum(float(v) for v in self.proxy_terms.values())),
        }
        for name, value in self.local_law_raw.items():
            payload[f"{prefix}_{name}_raw"] = float(value)
        for name, value in self.local_law_terms.items():
            payload[f"{prefix}_{name}_term"] = float(value)
        for name, value in self.proxy_raw.items():
            payload[f"{prefix}_{name}_raw"] = float(value)
        for name, value in self.proxy_terms.items():
            payload[f"{prefix}_{name}_term"] = float(value)
        return payload


OBJECTIVE_ESTIMATOR_KEYS = ("exact", "ht", "hajek", "eb_lo", "eb_hi")


def objective_estimator_alias(base_name: str, estimator: str) -> str:
    name = str(base_name)
    est = str(estimator)
    return name if est == "exact" else f"{name}_{est}"


def _safe_estimator_value(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def scalarize_objective_estimates(
    spec: CompositeObjectiveSpec,
    *,
    task_estimates: Mapping[str, float],
    local_law_estimates: Mapping[str, Mapping[str, float]],
    proxy_estimates: Optional[Mapping[str, Mapping[str, float]]] = None,
    selection_preference: str = "hajek",
) -> Dict[str, Any]:
    base_name = str(spec.name or spec.selection_metric_name or "configured_objective")
    proxy_source = dict(proxy_estimates or {})

    term_breakdown: Dict[str, Dict[str, Dict[str, float]]] = {
        "task": {},
        "local_law": {},
        "proxy": {},
    }
    totals: Dict[str, Dict[str, float]] = {}

    for estimator in OBJECTIVE_ESTIMATOR_KEYS:
        task_raw = _safe_estimator_value(task_estimates.get(estimator))
        task_term = (
            float(spec.task_weight) * float(task_raw) if math.isfinite(task_raw) else float("nan")
        )
        estimator_local_raw: Dict[str, float] = {}
        estimator_local_terms: Dict[str, float] = {}
        for name, weight in dict(spec.local_law_weights).items():
            raw_map = dict(local_law_estimates.get(str(name), {}) or {})
            raw_value = _safe_estimator_value(raw_map.get(estimator))
            estimator_local_raw[str(name)] = float(raw_value)
            estimator_local_terms[str(name)] = (
                float(weight) * float(raw_value) if math.isfinite(raw_value) else float("nan")
            )
        estimator_proxy_raw: Dict[str, float] = {}
        estimator_proxy_terms: Dict[str, float] = {}
        for name, weight in dict(spec.proxy_weights).items():
            raw_map = dict(proxy_source.get(str(name), {}) or {})
            raw_value = _safe_estimator_value(raw_map.get(estimator))
            estimator_proxy_raw[str(name)] = float(raw_value)
            estimator_proxy_terms[str(name)] = (
                float(weight) * float(raw_value) if math.isfinite(raw_value) else float("nan")
            )
        total = (
            float(task_term)
            + sum(float(v) for v in estimator_local_terms.values() if math.isfinite(float(v)))
            + sum(float(v) for v in estimator_proxy_terms.values() if math.isfinite(float(v)))
            if math.isfinite(task_term)
            and all(math.isfinite(float(v)) for v in estimator_local_terms.values())
            and all(math.isfinite(float(v)) for v in estimator_proxy_terms.values())
            else float("nan")
        )
        totals[str(estimator)] = {
            "full_objective_value": float(total),
            "task_objective_value": float(task_raw),
            "task_objective_term": float(task_term),
            "local_law_objective_value": float(
                sum(float(v) for v in estimator_local_raw.values())
            )
            if all(math.isfinite(float(v)) for v in estimator_local_raw.values())
            else float("nan"),
            "local_law_objective_term": float(
                sum(float(v) for v in estimator_local_terms.values())
            )
            if all(math.isfinite(float(v)) for v in estimator_local_terms.values())
            else float("nan"),
            "proxy_objective_value": float(sum(float(v) for v in estimator_proxy_raw.values()))
            if all(math.isfinite(float(v)) for v in estimator_proxy_raw.values())
            else float("nan"),
            "proxy_objective_term": float(sum(float(v) for v in estimator_proxy_terms.values()))
            if all(math.isfinite(float(v)) for v in estimator_proxy_terms.values())
            else float("nan"),
        }
        term_breakdown["task"][str(estimator)] = {
            "raw": float(task_raw),
            "term": float(task_term),
        }
        for name in dict(spec.local_law_weights).keys():
            term_breakdown["local_law"].setdefault(str(name), {})[str(estimator)] = {
                "raw": float(estimator_local_raw[str(name)]),
                "term": float(estimator_local_terms[str(name)]),
            }
        for name in dict(spec.proxy_weights).keys():
            term_breakdown["proxy"].setdefault(str(name), {})[str(estimator)] = {
                "raw": float(estimator_proxy_raw[str(name)]),
                "term": float(estimator_proxy_terms[str(name)]),
            }

    available_estimators = [
        str(estimator)
        for estimator in OBJECTIVE_ESTIMATOR_KEYS
        if math.isfinite(_safe_estimator_value(totals[str(estimator)]["full_objective_value"]))
    ]
    preferred = str(selection_preference)
    if preferred not in available_estimators:
        preferred = "exact" if "exact" in available_estimators else (
            available_estimators[0] if available_estimators else "exact"
        )
    selection_metric_name = objective_estimator_alias(base_name, preferred)
    selection_metric_value = _safe_estimator_value(
        totals.get(preferred, {}).get("full_objective_value")
    )

    payload: Dict[str, Any] = {
        "objective_name": base_name,
        "selection_metric_name": str(selection_metric_name),
        "selection_estimator": str(preferred),
        "selection_metric_value": float(selection_metric_value),
        "available_estimators": [str(x) for x in available_estimators],
        "estimator_components": term_breakdown,
    }
    exact = totals.get("exact", {})
    payload["full_objective_value"] = float(
        _safe_estimator_value(exact.get("full_objective_value"))
    )
    payload["task_objective_value"] = float(
        _safe_estimator_value(exact.get("task_objective_value"))
    )
    payload["task_objective_term"] = float(
        _safe_estimator_value(exact.get("task_objective_term"))
    )
    payload["regular_objective_value"] = float(
        _safe_estimator_value(exact.get("task_objective_value"))
    )
    payload["regular_objective_term"] = float(
        _safe_estimator_value(exact.get("task_objective_term"))
    )
    payload["local_law_objective_value"] = float(
        _safe_estimator_value(exact.get("local_law_objective_value"))
    )
    payload["local_law_objective_term"] = float(
        _safe_estimator_value(exact.get("local_law_objective_term"))
    )
    payload["proxy_objective_value"] = float(
        _safe_estimator_value(exact.get("proxy_objective_value"))
    )
    payload["proxy_objective_term"] = float(
        _safe_estimator_value(exact.get("proxy_objective_term"))
    )
    for estimator, metrics in totals.items():
        alias = objective_estimator_alias(base_name, estimator)
        payload[str(alias)] = float(_safe_estimator_value(metrics.get("full_objective_value")))
        payload[f"{alias}_task_objective_value"] = float(
            _safe_estimator_value(metrics.get("task_objective_value"))
        )
        payload[f"{alias}_task_objective_term"] = float(
            _safe_estimator_value(metrics.get("task_objective_term"))
        )
        payload[f"{alias}_local_law_objective_value"] = float(
            _safe_estimator_value(metrics.get("local_law_objective_value"))
        )
        payload[f"{alias}_local_law_objective_term"] = float(
            _safe_estimator_value(metrics.get("local_law_objective_term"))
        )
        payload[f"{alias}_proxy_objective_value"] = float(
            _safe_estimator_value(metrics.get("proxy_objective_value"))
        )
        payload[f"{alias}_proxy_objective_term"] = float(
            _safe_estimator_value(metrics.get("proxy_objective_term"))
        )
    eb_lo = _safe_estimator_value(totals.get("eb_lo", {}).get("full_objective_value"))
    eb_hi = _safe_estimator_value(totals.get("eb_hi", {}).get("full_objective_value"))
    payload[f"{base_name}_eb_width"] = (
        float(max(0.0, eb_hi - eb_lo))
        if math.isfinite(eb_lo) and math.isfinite(eb_hi)
        else float("nan")
    )
    payload[f"{base_name}_selection_value"] = float(selection_metric_value)
    return payload


def evaluate_composite_objective(
    spec: CompositeObjectiveSpec,
    *,
    task_value: float,
    local_law_values: Mapping[str, float],
    proxy_values: Optional[Mapping[str, float]] = None,
) -> CompositeObjectiveEvaluation:
    task_raw = float(task_value)
    task_term = float(spec.task_weight) * task_raw

    local_law_raw = {
        str(name): float(local_law_values.get(name, 0.0))
        for name in dict(spec.local_law_weights).keys()
    }
    local_law_terms = {
        str(name): float(spec.local_law_weights.get(name, 0.0)) * float(local_law_raw[str(name)])
        for name in dict(spec.local_law_weights).keys()
    }

    proxy_source = dict(proxy_values or {})
    proxy_raw = {
        str(name): float(proxy_source.get(name, 0.0)) for name in dict(spec.proxy_weights).keys()
    }
    proxy_terms = {
        str(name): float(spec.proxy_weights.get(name, 0.0)) * float(proxy_raw[str(name)])
        for name in dict(spec.proxy_weights).keys()
    }

    total = float(
        task_term
        + sum(float(v) for v in local_law_terms.values())
        + sum(float(v) for v in proxy_terms.values())
    )
    return CompositeObjectiveEvaluation(
        total=total,
        task_raw=task_raw,
        task_term=task_term,
        local_law_raw=local_law_raw,
        local_law_terms=local_law_terms,
        proxy_raw=proxy_raw,
        proxy_terms=proxy_terms,
    )


def evaluate_composite_objective_from_metrics(
    spec: CompositeObjectiveSpec,
    *,
    metrics: Mapping[str, object],
    task_metric_name: Optional[str] = None,
    local_law_metric_names: Optional[Mapping[str, str]] = None,
    proxy_metric_names: Optional[Mapping[str, str]] = None,
) -> CompositeObjectiveEvaluation:
    metadata = dict(spec.metadata)
    resolved_task_metric_name = str(
        task_metric_name or metadata.get("task_metric_name") or spec.task_name
    )
    resolved_local_law_metric_names = dict(
        metadata.get("local_law_metric_names", {})
        if isinstance(metadata.get("local_law_metric_names"), Mapping)
        else {}
    )
    if local_law_metric_names is not None:
        resolved_local_law_metric_names.update(
            {str(name): str(metric_name) for name, metric_name in local_law_metric_names.items()}
        )
    resolved_proxy_metric_names = dict(
        metadata.get("proxy_metric_names", {})
        if isinstance(metadata.get("proxy_metric_names"), Mapping)
        else {}
    )
    if proxy_metric_names is not None:
        resolved_proxy_metric_names.update(
            {str(name): str(metric_name) for name, metric_name in proxy_metric_names.items()}
        )

    return evaluate_composite_objective(
        spec,
        task_value=float(metrics.get(resolved_task_metric_name, 0.0)),
        local_law_values={
            str(name): float(
                metrics.get(resolved_local_law_metric_names.get(str(name), str(name)), 0.0)
            )
            for name in dict(spec.local_law_weights).keys()
        },
        proxy_values={
            str(name): float(
                metrics.get(resolved_proxy_metric_names.get(str(name), str(name)), 0.0)
            )
            for name in dict(spec.proxy_weights).keys()
        },
    )
