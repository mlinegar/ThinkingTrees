from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from treepo_cdx.audit import LocalLawAuditRow


MIN_PROPENSITY = 1e-12
LOCAL_LAW_OBJECTIVE_CORRECTED = "corrected_local_law"
LOCAL_LAW_OBJECTIVE_SAMPLED_IPW = "sampled_ipw"
LOCAL_LAW_OBJECTIVE_MODES = (
    LOCAL_LAW_OBJECTIVE_CORRECTED,
    LOCAL_LAW_OBJECTIVE_SAMPLED_IPW,
)


def _finite_float(value: float, *, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def normalize_local_law_objective_mode(mode: str) -> str:
    normalized = str(mode or LOCAL_LAW_OBJECTIVE_CORRECTED).strip().lower()
    aliases = {
        "corrected": LOCAL_LAW_OBJECTIVE_CORRECTED,
        "aipw": LOCAL_LAW_OBJECTIVE_CORRECTED,
        "adjusted": LOCAL_LAW_OBJECTIVE_CORRECTED,
        "adjusted_local_law": LOCAL_LAW_OBJECTIVE_CORRECTED,
        "dr": LOCAL_LAW_OBJECTIVE_CORRECTED,
        "doubly_robust": LOCAL_LAW_OBJECTIVE_CORRECTED,
        "ipw": LOCAL_LAW_OBJECTIVE_SAMPLED_IPW,
        "sampled": LOCAL_LAW_OBJECTIVE_SAMPLED_IPW,
        "hajek": LOCAL_LAW_OBJECTIVE_SAMPLED_IPW,
        "sampled_hajek": LOCAL_LAW_OBJECTIVE_SAMPLED_IPW,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in LOCAL_LAW_OBJECTIVE_MODES:
        raise ValueError(
            f"unknown local-law objective mode {mode!r}; expected one of "
            f"{LOCAL_LAW_OBJECTIVE_MODES}"
        )
    return normalized


def corrected_local_law_loss(
    *,
    proxy_loss: float,
    oracle_loss: float | None,
    observed: bool,
    propensity: float,
    min_propensity: float = MIN_PROPENSITY,
) -> float:
    proxy = _finite_float(proxy_loss, name="proxy_loss")
    if not bool(observed):
        return proxy
    if oracle_loss is None:
        raise ValueError("observed corrected local-law rows require oracle_loss")
    oracle = _finite_float(oracle_loss, name="oracle_loss")
    pi = _finite_float(propensity, name="propensity")
    if pi <= 0.0 or pi > 1.0:
        raise ValueError(f"observed local-law propensity must be in (0, 1], got {propensity!r}")
    return float(proxy + (oracle - proxy) / max(float(min_propensity), pi, MIN_PROPENSITY))


def _depth_weight(depth: int, *, gamma_depth: float) -> float:
    gamma = _finite_float(gamma_depth, name="gamma_depth")
    if gamma < 0.0:
        raise ValueError("gamma_depth must be non-negative")
    return float(gamma ** int(depth))


@dataclass(frozen=True)
class LocalLawObjectiveSummary:
    objective: float
    objective_mode: str
    row_count: int
    observed_count: int
    weight_sum: float
    effective_observed_weight_sum: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "objective": float(self.objective),
            "objective_mode": self.objective_mode,
            "row_count": int(self.row_count),
            "observed_count": int(self.observed_count),
            "weight_sum": float(self.weight_sum),
            "effective_observed_weight_sum": float(self.effective_observed_weight_sum),
        }


def local_law_objective_mean(
    rows: Sequence[LocalLawAuditRow],
    *,
    gamma_depth: float = 1.0,
    objective_mode: str = LOCAL_LAW_OBJECTIVE_CORRECTED,
    min_propensity: float = MIN_PROPENSITY,
) -> float:
    return local_law_objective_summary(
        rows,
        gamma_depth=gamma_depth,
        objective_mode=objective_mode,
        min_propensity=min_propensity,
    ).objective


def local_law_objective_summary(
    rows: Sequence[LocalLawAuditRow],
    *,
    gamma_depth: float = 1.0,
    objective_mode: str = LOCAL_LAW_OBJECTIVE_CORRECTED,
    min_propensity: float = MIN_PROPENSITY,
) -> LocalLawObjectiveSummary:
    row_list = list(rows)
    if not row_list:
        return LocalLawObjectiveSummary(
            objective=0.0,
            objective_mode=normalize_local_law_objective_mode(objective_mode),
            row_count=0,
            observed_count=0,
            weight_sum=0.0,
            effective_observed_weight_sum=0.0,
        )
    min_pi = max(float(min_propensity), MIN_PROPENSITY)
    mode = normalize_local_law_objective_mode(objective_mode)
    weighted_total = 0.0
    weight_sum = 0.0
    effective_observed_weight_sum = 0.0
    observed_count = 0
    if mode == LOCAL_LAW_OBJECTIVE_SAMPLED_IPW:
        for row in row_list:
            if not bool(row.observed):
                continue
            if row.oracle_loss is None:
                raise ValueError("sampled_ipw observed rows require oracle_loss")
            pi = max(min_pi, float(row.propensity))
            weight = float(row.node_weight) * _depth_weight(row.depth, gamma_depth=gamma_depth)
            ipw_weight = weight / pi
            weighted_total += ipw_weight * float(row.oracle_loss)
            effective_observed_weight_sum += ipw_weight
            observed_count += 1
        objective = (
            weighted_total / effective_observed_weight_sum
            if effective_observed_weight_sum > 0.0
            else 0.0
        )
        return LocalLawObjectiveSummary(
            objective=float(objective),
            objective_mode=mode,
            row_count=len(row_list),
            observed_count=observed_count,
            weight_sum=float(sum(float(row.node_weight) for row in row_list)),
            effective_observed_weight_sum=float(effective_observed_weight_sum),
        )

    for row in row_list:
        weight = float(row.node_weight) * _depth_weight(row.depth, gamma_depth=gamma_depth)
        weighted_total += weight * corrected_local_law_loss(
            proxy_loss=float(row.proxy_loss),
            oracle_loss=row.oracle_loss,
            observed=bool(row.observed),
            propensity=float(row.propensity),
            min_propensity=min_pi,
        )
        weight_sum += weight
        if bool(row.observed):
            observed_count += 1
            effective_observed_weight_sum += weight / max(min_pi, float(row.propensity))
    objective = weighted_total / weight_sum if weight_sum > 0.0 else 0.0
    return LocalLawObjectiveSummary(
        objective=float(objective),
        objective_mode=mode,
        row_count=len(row_list),
        observed_count=observed_count,
        weight_sum=float(weight_sum),
        effective_observed_weight_sum=float(effective_observed_weight_sum),
    )


__all__ = [
    "LOCAL_LAW_OBJECTIVE_CORRECTED",
    "LOCAL_LAW_OBJECTIVE_MODES",
    "LOCAL_LAW_OBJECTIVE_SAMPLED_IPW",
    "LocalLawObjectiveSummary",
    "corrected_local_law_loss",
    "local_law_objective_mean",
    "local_law_objective_summary",
    "normalize_local_law_objective_mode",
]
