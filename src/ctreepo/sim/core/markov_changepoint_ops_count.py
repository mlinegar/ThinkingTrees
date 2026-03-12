"""
OPS-style oracle simulations for a Markov changepoint-count target.

This module is designed to match the Lean OPS semantics (C1/C2/C3 ≈ L1/L3/L2):
we audit leaf- and merge-level oracle preservation for a *tree reduction*,
and we separate three error sources that matter for the paper:

1) Approximation bias: insufficient sketch state (chunking loss).
2) Estimation error: finite training docs + finite oracle labels.
3) Selection bias: adaptive node sampling (corrected by IPW / DSL-style AIPW).

Oracle:
    f⋆(x) = number of changepoints (# adjacent regime flips) in a span.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Dict, Generic, List, Mapping, Optional, Sequence, Tuple, TypeVar

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for markov_changepoint_ops_count_simulation. "
        "Install with: pip install torch>=2.0.0"
    ) from e

from src.tree.ipw import (
    NodeType,
    TreePropensity,
    TreeSample,
    effective_sample_size,
    empirical_bernstein_ci,
    hajek_ht_comparison,
    horvitz_thompson_mean,
    max_weight,
)
from src.tree.markov_boundary_honesty_simulation import _make_transition_matrices
from src.tree.markov_changepoint_honesty_simulation import (
    ChangepointMarkovDoc,
    MarkovChangepointConfig as _GeneratorConfig,
    generate_changepoint_docs,
)
from src.core.ops_checks import EvidenceStatus, LawKind
from src.ctreepo.sim.composite_objective import (
    CompositeObjectiveSpec,
    OBJECTIVE_ESTIMATOR_KEYS,
    evaluate_composite_objective_from_metrics,
    objective_estimator_alias,
    scalarize_objective_estimates,
)
from src.ctreepo.sim.core.markov_capability import markov_theorem_score
from src.ctreepo.sim.core.markov_law_stress import (
    VALID_EXACT_FAMILIES,
    VALID_LAW_PACKAGES,
    markov_law_bundle_score,
)
from src.ctreepo.sim.core.training_selection import (
    TrainingSelectionMetadata,
    clone_module_state,
    improved_metric,
    restore_module_state,
)
from src.ctreepo.sim.learning_problem import attach_local_law_learning_problem
from src.ctreepo.sim.local_law_learnability import (
    DownstreamMetrics,
    GArtifact,
    LocalLawCounterexampleEvaluation,
    LocalLawMetrics,
    LocalLawPolicyEvaluation,
    LocalLawRunSummary,
    PolicyRole,
    SupportBudgetSummary,
    artifact_index,
    write_json_g_artifact,
    write_npz_g_artifact,
)


ScheduleName = str
VALID_SCHEDULES: Tuple[ScheduleName, ...] = ("balanced", "left_to_right", "right_to_left")
ModelFamilyName = str
VALID_MODEL_FAMILIES: Tuple[ModelFamilyName, ...] = ("neural", "additive")
GuidanceOverrideModeName = str
VALID_GUIDANCE_OVERRIDE_MODES: Tuple[GuidanceOverrideModeName, ...] = ("reset", "adjust")
AuditPolicyName = str
VALID_AUDIT_POLICIES: Tuple[AuditPolicyName, ...] = (
    "all",
    "fixed",
    "fraction",
    "sqrt",
    "log2",
)
C3AuditStrategyName = str
VALID_C3_AUDIT_STRATEGIES: Tuple[C3AuditStrategyName, ...] = (
    "uniform",
    "top_span",
    "span_weighted",
    "hybrid_top_span",
)
DEFAULT_NORMALIZED_LOCAL_LAW_WEIGHT = 0.25
StateT = TypeVar("StateT")


def audit_sample_count(
    internal_nodes: int,
    *,
    policy: AuditPolicyName,
    fixed_nodes: int = 0,
    fraction: float = 1.0,
    scale: float = 1.0,
) -> int:
    """
    How many realized internal nodes to label (per doc), matching learned_sketch_simulation semantics.
    """

    n = int(max(0, internal_nodes))
    if n <= 0:
        return 0

    pol = str(policy)
    if pol == "all":
        q = n
    elif pol == "fixed":
        q = int(max(0, fixed_nodes))
    elif pol == "fraction":
        q = int(math.ceil(float(fraction) * float(n)))
    elif pol == "sqrt":
        q = int(math.ceil(float(scale) * math.sqrt(float(n))))
    elif pol == "log2":
        q = int(math.ceil(float(scale) * math.log2(float(n) + 1.0)))
    else:
        raise ValueError(
            f"unsupported audit policy: {policy!r}; expected one of {VALID_AUDIT_POLICIES}"
        )
    return int(max(0, min(n, q)))


def leaf_sample_count(leaves: int, *, rate: float) -> int:
    """How many realized leaf nodes to label (per doc)."""

    n = int(max(0, leaves))
    if n <= 0:
        return 0
    r = float(rate)
    if r <= 0.0:
        return 0
    if r >= 1.0:
        return n
    q = int(math.ceil(r * float(n)))
    return int(max(1, min(n, q)))


def _set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _changepoint_count(regimes: Sequence[int]) -> int:
    if len(regimes) < 2:
        return 0
    return int(sum(1 for a, b in zip(regimes[:-1], regimes[1:]) if int(a) != int(b)))


def _oracle_count(doc: ChangepointMarkovDoc, *, start: int, end: int) -> int:
    regs = doc.token_regimes[int(start) : int(end)]
    return _changepoint_count(regs)


def _leaf_spans(n_tokens: int, *, leaf_tokens: int) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    n = int(n_tokens)
    step = int(max(1, leaf_tokens))
    i = 0
    while i < n:
        j = min(n, i + step)
        spans.append((i, j))
        i = j
    return spans


@dataclass(frozen=True)
class OPSCountConfig:
    # Document generator.
    n_regimes: int = 4
    vocab_size: int = 96
    min_tokens: int = 384
    max_tokens: int = 384
    min_segments: int = 12
    max_segments: int = 24
    min_seg_len: int = 8
    max_seg_len: int = 32
    sinkhorn_iters: int = 30
    transition_log_std: float = 1.25

    # Realized partition (fixed leaves).
    fixed_leaf_tokens: int = 16

    # Training / eval sizes.
    train_docs: int = 1000
    val_docs: int = 0
    test_docs: int = 1000
    data_seed: Optional[int] = None
    model_seed: Optional[int] = None
    val_seed_offset: int = 5_000
    test_seed_offset: int = 10_000

    # Learned sketch settings.
    model_family: ModelFamilyName = (
        "neural"  # "neural" learns an unstructured merger; "additive" uses an additive merger.
    )
    feature_mode: str = "full"  # "full" includes endpoints; "no_endpoints" drops them.
    state_dim: int = 32
    hidden_dim: int = 128
    n_epochs: int = 10
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-5
    # Legacy direct term weights. The generic baseline is intentionally root-only:
    # pre-local-law runs should correspond to zero theorem-term weight unless the
    # caller opts into local supervision explicitly.
    #
    # If `local_law_weight` is set, these are only used to preserve old configs
    # in summaries and are not the active objective.
    c3_weight: float = 0.0
    c2_weight: float = 0.0
    leaf_weight: float = 0.0
    law_package: str = ""
    # Formal theorem-facing parameterization of the local-law bundle:
    # (1 - λ) * root_objective + λ * [ρ_C1 * C1/L1 + ρ_C2 * C2/L3 + ρ_C3 * C3/L2].
    # When unset, we fall back to the legacy additive term weights above.
    #
    # `task_objective_weight` can be used to break the normalized simplex and
    # optimize an explicit composite objective
    #   task_weight * task + c1 * C1 + c2 * C2 + c3 * C3 (+ proxy)
    # while keeping the same local-law share parameterization.
    local_law_weight: Optional[float] = None
    task_objective_weight: Optional[float] = None
    c1_relative_weight: float = 1.0
    c2_relative_weight: float = 1.0
    c3_relative_weight: float = 1.0
    # Legacy direct root-term weight. In the normalized theorem-facing parameterization,
    # the active root weight is `1 - local_law_weight`; this field is retained for
    # backward-compatible legacy runs and reporting.
    root_weight: float = 1.0
    # Proxy-only associativity regularizer; not a Lean local law.
    schedule_consistency_weight: float = 0.0
    grad_clip_norm: float = 1.0
    exact_family: str = ""

    # Node-label budgets (oracle queries).
    audit_policy: AuditPolicyName = "fraction"
    audit_fixed_nodes: int = 0
    audit_fraction: float = 0.2
    audit_scale: float = 1.0
    c3_audit_strategy: C3AuditStrategyName = "uniform"
    c3_include_root: bool = True
    leaf_query_rate: float = 1.0
    include_root_query: bool = True
    # Inference-time oracle visibility sweep on realized internal nodes.
    eval_guidance_qs: Tuple[float, ...] = tuple()
    eval_guidance_trials: int = 0
    eval_guidance_seed_offset: int = 100_000
    eval_guidance_include_root: bool = True
    guidance_override_mode: GuidanceOverrideModeName = (
        "reset"  # for neural sketches: reset vs adjust along readout
    )

    # Simple classical baseline (doc-level root regression).
    include_rf_root_baseline: bool = False
    rf_n_estimators: int = 200
    rf_max_depth: int = 16
    rf_min_samples_leaf: int = 5

    # Evaluation / audit thresholds.
    violation_tau: float = 0.0
    suite_role: str = ""
    artifact_dir: str = ""

    # Runtime.
    seed: int = 0
    use_cuda: bool = True
    cuda_device: Optional[int] = None
    torch_threads: int = 0


@dataclass(frozen=True)
class SketchMetrics:
    root_mae: float
    root_median_abs_error: float
    root_p95_abs_error: float
    schedule_spread_mean: float
    schedule_spread_p95: float
    leaf_mae: float
    leaf_violation_rate: float
    c2_idempotence_mae: float
    c2_r1_mae: float
    c2_r2_mae: float
    c2_r4_mae: float
    resummary_root_drift_r1: float
    resummary_root_drift_r2: float
    resummary_root_drift_r4: float
    merge_mae: float
    merge_violation_rate: float
    n_docs: int


@dataclass(frozen=True)
class TrainingGeometry:
    mean_tokens: float
    mean_leaves: float
    mean_internal_nodes: float
    mean_leaf_labels: float
    mean_internal_labels: float
    mean_queries_per_doc: float
    root_queries_total: int
    leaf_labels_total: int
    internal_labels_total: int
    total_queries_estimate: int


@dataclass(frozen=True)
class EstimatorDiagnostics:
    true_mean: float
    naive_bias: float
    ipw_bias: float
    dsl_bias: float
    ipw_var: float
    dsl_var: float


@dataclass(frozen=True)
class TrainFitDiagnostics:
    train_loss_final: float
    train_loss_curve: Tuple[float, ...]
    epochs_completed: int
    selection_metric_curve: Tuple[float, ...] = tuple()
    selection_mode: str = "final_epoch_no_validation"
    selection_split: str = "config"
    selection_metric_name: str = "train_loss_final"
    selection_metric_value: float = float("nan")
    best_epoch: int = 0


@dataclass(frozen=True)
class ObjectiveMetrics:
    optimization_total_loss: float
    optimization_root_loss: float
    optimization_leaf_loss: float
    optimization_c2_loss: float
    optimization_merge_loss: float
    optimization_schedule_consistency_loss: float
    raw_total_loss: float
    raw_root_loss: float
    raw_leaf_loss: float
    raw_c2_loss: float
    raw_merge_loss: float
    raw_schedule_consistency_loss: float
    n_docs: int


@dataclass(frozen=True)
class OPSCountSummary:
    config: Dict[str, object]
    training_geometry: Dict[str, float | int]
    objective: Dict[str, object]
    metrics: Dict[str, object]
    estimator_diagnostics: Dict[str, float]
    local_law_learnability: Dict[str, object] = field(default_factory=dict)
    g_artifacts: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {
            "config": self.config,
            "training_geometry": self.training_geometry,
            "objective": self.objective,
            "metrics": self.metrics,
            "estimator_diagnostics": self.estimator_diagnostics,
        }
        if self.local_law_learnability:
            payload["local_law_learnability"] = self.local_law_learnability
        if self.g_artifacts:
            payload["g_artifacts"] = self.g_artifacts
        return json.dumps(payload, indent=2, sort_keys=True)


def _metrics_with_split_prefix(
    metrics: SketchMetrics,
    *,
    prefix: str,
    target_scale: Optional[float] = None,
) -> Dict[str, float | int]:
    payload: Dict[str, float | int] = {
        f"{prefix}_root_mae": float(metrics.root_mae),
        f"{prefix}_root_median_abs_error": float(metrics.root_median_abs_error),
        f"{prefix}_root_p95_abs_error": float(metrics.root_p95_abs_error),
        f"{prefix}_schedule_spread_mean": float(metrics.schedule_spread_mean),
        f"{prefix}_schedule_spread_p95": float(metrics.schedule_spread_p95),
        f"{prefix}_leaf_mae": float(metrics.leaf_mae),
        f"{prefix}_leaf_violation_rate": float(metrics.leaf_violation_rate),
        f"{prefix}_c2_idempotence_mae": float(metrics.c2_idempotence_mae),
        f"{prefix}_c2_r1_mae": float(metrics.c2_r1_mae),
        f"{prefix}_c2_r2_mae": float(metrics.c2_r2_mae),
        f"{prefix}_c2_r4_mae": float(metrics.c2_r4_mae),
        f"{prefix}_resummary_root_drift_r1": float(metrics.resummary_root_drift_r1),
        f"{prefix}_resummary_root_drift_r2": float(metrics.resummary_root_drift_r2),
        f"{prefix}_resummary_root_drift_r4": float(metrics.resummary_root_drift_r4),
        f"{prefix}_merge_mae": float(metrics.merge_mae),
        f"{prefix}_merge_violation_rate": float(metrics.merge_violation_rate),
        f"{prefix}_n_docs": int(metrics.n_docs),
    }
    scale = float(target_scale) if target_scale is not None else float("nan")
    if math.isfinite(scale) and scale > 0.0:
        payload.update(
            {
                f"{prefix}_root_mae_n": float(metrics.root_mae) / scale,
                f"{prefix}_schedule_spread_mean_n": float(metrics.schedule_spread_mean) / scale,
                f"{prefix}_c1_leaf_mae_n": float(metrics.leaf_mae) / scale,
                f"{prefix}_c2_idempotence_mae_n": float(metrics.c2_idempotence_mae) / scale,
                f"{prefix}_c2_r1_mae_n": float(metrics.c2_r1_mae) / scale,
                f"{prefix}_c2_r2_mae_n": float(metrics.c2_r2_mae) / scale,
                f"{prefix}_c2_r4_mae_n": float(metrics.c2_r4_mae) / scale,
                f"{prefix}_resummary_root_drift_r1_n": float(metrics.resummary_root_drift_r1)
                / scale,
                f"{prefix}_resummary_root_drift_r2_n": float(metrics.resummary_root_drift_r2)
                / scale,
                f"{prefix}_resummary_root_drift_r4_n": float(metrics.resummary_root_drift_r4)
                / scale,
                f"{prefix}_c3_merge_mae_n": float(metrics.merge_mae) / scale,
            }
        )
    return payload


def _objective_with_split_prefix(
    metrics: ObjectiveMetrics,
    *,
    prefix: str,
) -> Dict[str, float | int]:
    return {
        # Backward-compatible aliases: these remain the weighted objective used in optimization.
        f"{prefix}_objective_full_labels": float(metrics.optimization_total_loss),
        f"{prefix}_objective_root_term": float(metrics.optimization_root_loss),
        f"{prefix}_objective_task_objective_term": float(metrics.optimization_root_loss),
        f"{prefix}_objective_leaf_term": float(metrics.optimization_leaf_loss),
        f"{prefix}_objective_c2_term": float(metrics.optimization_c2_loss),
        f"{prefix}_objective_merge_term": float(metrics.optimization_merge_loss),
        f"{prefix}_objective_schedule_consistency_term": float(
            metrics.optimization_schedule_consistency_loss
        ),
        f"{prefix}_optimization_objective_full_labels": float(metrics.optimization_total_loss),
        f"{prefix}_optimization_objective_root_term": float(metrics.optimization_root_loss),
        f"{prefix}_optimization_objective_task_objective_term": float(
            metrics.optimization_root_loss
        ),
        f"{prefix}_optimization_objective_leaf_term": float(metrics.optimization_leaf_loss),
        f"{prefix}_optimization_objective_c2_term": float(metrics.optimization_c2_loss),
        f"{prefix}_optimization_objective_merge_term": float(metrics.optimization_merge_loss),
        f"{prefix}_optimization_objective_schedule_consistency_term": float(
            metrics.optimization_schedule_consistency_loss
        ),
        f"{prefix}_unweighted_objective_full_labels": float(metrics.raw_total_loss),
        f"{prefix}_unweighted_objective_root_term": float(metrics.raw_root_loss),
        f"{prefix}_unweighted_objective_task_objective_term": float(metrics.raw_root_loss),
        f"{prefix}_unweighted_objective_leaf_term": float(metrics.raw_leaf_loss),
        f"{prefix}_unweighted_objective_c2_term": float(metrics.raw_c2_loss),
        f"{prefix}_unweighted_objective_merge_term": float(metrics.raw_merge_loss),
        f"{prefix}_unweighted_objective_schedule_consistency_term": float(
            metrics.raw_schedule_consistency_loss
        ),
        f"{prefix}_objective_n_docs": int(metrics.n_docs),
    }


def _objective_estimator_with_split_prefix(
    estimator_payload: Mapping[str, Any],
    *,
    prefix: str,
) -> Dict[str, object]:
    payload = dict(estimator_payload or {})
    if not payload:
        return {}
    base_name = str(payload.get("objective_name", "configured_objective"))
    out: Dict[str, object] = {
        f"{prefix}_objective_estimator_payload": payload,
        f"{prefix}_objective_name": base_name,
        f"{prefix}_objective_selection_metric_name": str(
            payload.get("selection_metric_name", "")
        ),
        f"{prefix}_objective_selection_estimator": str(
            payload.get("selection_estimator", "exact")
        ),
        f"{prefix}_objective_selection_metric_value": float(
            payload.get("selection_metric_value", float("nan"))
        ),
        f"{prefix}_objective_available_estimators": list(
            payload.get("available_estimators", [])
        ),
    }
    for estimator in OBJECTIVE_ESTIMATOR_KEYS:
        alias = objective_estimator_alias(base_name, estimator)
        if alias in payload:
            out[f"{prefix}_{alias}"] = float(payload[alias])
    width_key = f"{base_name}_eb_width"
    if width_key in payload:
        out[f"{prefix}_{width_key}"] = float(payload[width_key])
    selection_value_key = f"{base_name}_selection_value"
    if selection_value_key in payload:
        out[f"{prefix}_{selection_value_key}"] = float(payload[selection_value_key])
    if "estimator_diagnostics" in payload:
        out[f"{prefix}_objective_estimator_diagnostics"] = dict(
            payload.get("estimator_diagnostics", {}) or {}
        )
    return out


def _markov_local_metrics(metrics: SketchMetrics, *, target_scale: float) -> LocalLawMetrics:
    scale = float(max(1.0, target_scale))
    return LocalLawMetrics(
        c1=float(metrics.leaf_mae) / scale,
        c2=float(metrics.c2_idempotence_mae) / scale,
        c3=float(metrics.merge_mae) / scale,
        combined=float(
            markov_law_bundle_score(
                c1=float(metrics.leaf_mae) / scale,
                c2=float(metrics.c2_idempotence_mae) / scale,
                c3=float(metrics.merge_mae) / scale,
            )
        ),
        root_error=float(metrics.root_mae) / scale,
        schedule_spread=float(metrics.schedule_spread_mean) / scale,
        c1_violation_rate=float(metrics.leaf_violation_rate),
        c3_violation_rate=float(metrics.merge_violation_rate),
    )


def _markov_downstream_metrics(
    metrics: SketchMetrics,
    *,
    target_scale: float,
) -> DownstreamMetrics:
    scale = float(max(1.0, target_scale))
    return DownstreamMetrics(
        root_error=float(metrics.root_mae) / scale,
        schedule_spread=float(metrics.schedule_spread_mean) / scale,
    )


def _maybe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return float(out) if math.isfinite(float(out)) else float(default)


def _constant_estimator_map(value: float) -> Dict[str, float]:
    return {str(name): float(value) for name in OBJECTIVE_ESTIMATOR_KEYS}


def _markov_normalized_ci_and_coverage(
    samples: Sequence[TreeSample],
    *,
    exact_value: float,
    raw_values_population: Sequence[float],
    population_size: float,
    delta: float,
) -> Dict[str, float]:
    raw_arr = np.asarray(raw_values_population, dtype=np.float64)
    lo = float(np.min(raw_arr)) if raw_arr.size > 0 else 0.0
    hi = float(np.max(raw_arr)) if raw_arr.size > 0 else 1.0
    scale = max(1e-12, hi - lo)
    if raw_arr.size == 0 or scale <= 1e-12:
        return {
            "value_min": float(lo),
            "value_max": float(hi),
            "scale": float(scale),
            "ht_mean": float(exact_value),
            "hajek": float(exact_value),
            "ht_abs_error": 0.0,
            "hajek_abs_error": 0.0,
            "eb_lo": float(exact_value),
            "eb_hi": float(exact_value),
            "eb_width": 0.0,
            "eb_contains_exact": 1.0,
            "effective_sample_size": float(effective_sample_size(samples)),
            "max_weight": float(max_weight(samples)),
            "sample_count": float(len(samples)),
            "weight_sum": float(sum(s.weight for s in samples)),
        }

    normalized_samples = [
        TreeSample(
            doc_id=str(sample.doc_id),
            node_id=str(sample.node_id),
            node_type=sample.node_type,
            violation=int(sample.violation),
            preference_loss=float(np.clip((float(sample.preference_loss) - lo) / scale, 0.0, 1.0)),
            propensity=sample.propensity,
            metadata=dict(sample.metadata),
        )
        for sample in samples
    ]
    comp = hajek_ht_comparison(
        normalized_samples,
        lambda s: float(s.preference_loss),
        population_size=float(population_size),
    )
    ci = empirical_bernstein_ci(
        normalized_samples,
        lambda s: float(s.preference_loss),
        float(delta),
        value_min=0.0,
        value_max=1.0,
    )
    exact_norm = (float(exact_value) - lo) / scale
    return {
        "value_min": float(lo),
        "value_max": float(hi),
        "scale": float(scale),
        "ht_mean": float(lo + scale * float(comp["ht_mean"])),
        "hajek": float(lo + scale * float(comp["hajek"])),
        "ht_abs_error": abs(float(lo + scale * float(comp["ht_mean"])) - float(exact_value)),
        "hajek_abs_error": abs(float(lo + scale * float(comp["hajek"])) - float(exact_value)),
        "eb_lo": float(lo + scale * float(ci[0])),
        "eb_hi": float(lo + scale * float(ci[1])),
        "eb_width": float(scale * max(0.0, float(ci[1]) - float(ci[0]))),
        "eb_contains_exact": (
            1.0 if float(ci[0]) - 1e-12 <= exact_norm <= float(ci[1]) + 1e-12 else 0.0
        ),
        "effective_sample_size": float(effective_sample_size(samples)),
        "max_weight": float(max_weight(samples)),
        "sample_count": float(len(samples)),
        "weight_sum": float(comp["weight_sum"]),
    }


def _markov_internal_inclusion_probabilities(
    n_internal: int,
    *,
    q: int,
    strategy: C3AuditStrategyName,
    merge_sizes: Sequence[int],
    include_root: bool,
) -> Tuple[Optional[np.ndarray], str]:
    n = int(max(0, n_internal))
    qq = int(max(0, q))
    if n <= 0:
        return np.zeros((0,), dtype=np.float64), "empty"
    if qq <= 0:
        return np.zeros((n,), dtype=np.float64), "zero_queries"
    if qq >= n:
        return np.ones((n,), dtype=np.float64), "all"

    strat = str(strategy)
    if strat not in VALID_C3_AUDIT_STRATEGIES:
        return None, "unsupported_strategy"

    probs = np.zeros((n,), dtype=np.float64)
    selected: set[int] = set()
    if include_root:
        selected.add(int(n - 1))
        probs[int(n - 1)] = 1.0
    if len(selected) >= qq:
        return probs, "deterministic_support"

    available = [i for i in range(n) if i not in selected]
    need = int(qq - len(selected))
    if need <= 0:
        return probs, "deterministic_support"

    if strat == "uniform":
        if not available:
            return probs, "uniform"
        p = float(min(1.0, float(need) / float(len(available))))
        for idx in available:
            probs[int(idx)] = float(p)
        return probs, "uniform"

    ranked = sorted(
        available,
        key=lambda i: (int(merge_sizes[i]) if i < len(merge_sizes) else 0, int(i)),
        reverse=True,
    )
    if strat == "top_span":
        for idx in ranked[:need]:
            probs[int(idx)] = 1.0
        return None, "top_span_partial_support"

    if strat == "hybrid_top_span":
        top_need = min(len(ranked), max(1, need // 2))
        deterministic = ranked[:top_need]
        for idx in deterministic:
            probs[int(idx)] = 1.0
        remaining_need = int(need - top_need)
        rem = [i for i in available if i not in set(deterministic)]
        if remaining_need >= len(rem):
            for idx in rem:
                probs[int(idx)] = 1.0
        elif remaining_need > 0 and rem:
            p = float(min(1.0, float(remaining_need) / float(len(rem))))
            for idx in rem:
                probs[int(idx)] = float(p)
        return probs, "hybrid_top_span"

    return None, "span_weighted_first_order_unavailable"


def _markov_objective_estimator_payload(
    model: LearnedCountSketch | AdditiveCountSketch,
    docs: Sequence[_CountDoc],
    *,
    device: torch.device,
    objective_summary: Mapping[str, Any],
    exact_objective: ObjectiveMetrics,
    leaf_query_rate: float,
    audit_policy: AuditPolicyName,
    audit_fixed_nodes: int,
    audit_fraction: float,
    audit_scale: float,
    c3_audit_strategy: C3AuditStrategyName,
    c3_include_root: bool,
    objective_ci_delta: float = 0.05,
    seed: int = 0,
) -> Dict[str, Any]:
    if len(docs) == 0:
        return {}

    objective_spec = _markov_composite_objective_spec(objective_summary)
    population_size = float(len(docs))
    rng = random.Random(int(seed))
    leaf_population_values: List[float] = []
    leaf_samples: List[TreeSample] = []
    merge_population_values: List[float] = []
    merge_samples: List[TreeSample] = []
    leaf_support_ok = True
    merge_support_ok = True
    merge_support_mode = "all"

    model.eval()
    with torch.no_grad():
        for doc_idx, doc in enumerate(docs):
            leaf_feats = _to_device(doc.leaf_features, device=device)
            states = [model.encode_leaf(x) for x in leaf_feats]
            _root_state, merge_states = model._merge_states(
                states,
                schedule="balanced",
                collect_merge_states=True,
            )

            n_leaf = int(len(states))
            if n_leaf > 0:
                leaf_losses: List[float] = []
                for st, truth in zip(states, doc.leaf_counts):
                    pred_leaf = model.predict_norm_from_state(st)
                    true_leaf = torch.tensor(
                        float(truth) / float(model.target_scale),
                        device=pred_leaf.device,
                        dtype=pred_leaf.dtype,
                    )
                    leaf_losses.append(
                        float(F.mse_loss(pred_leaf, true_leaf, reduction="mean").detach().cpu())
                    )
                scaled_leaf_values = [float(loss) / float(n_leaf) for loss in leaf_losses]
                leaf_population_values.extend(float(v) for v in scaled_leaf_values)
                q_leaf = leaf_sample_count(n_leaf, rate=float(leaf_query_rate))
                if q_leaf <= 0:
                    leaf_support_ok = False
                else:
                    if q_leaf >= n_leaf:
                        leaf_indices = list(range(n_leaf))
                    else:
                        leaf_indices = rng.sample(range(n_leaf), k=int(q_leaf))
                    pi_leaf = float(min(1.0, float(q_leaf) / float(n_leaf)))
                    if pi_leaf <= 0.0:
                        leaf_support_ok = False
                    for idx in leaf_indices:
                        leaf_samples.append(
                            TreeSample(
                                doc_id=f"markov_eval_{doc_idx}",
                                node_id=f"leaf_{idx}",
                                node_type=NodeType.LEAF,
                                violation=0,
                                preference_loss=float(scaled_leaf_values[int(idx)]),
                                propensity=TreePropensity(node=float(pi_leaf)),
                            )
                        )

            n_internal = int(len(merge_states))
            if n_internal > 0:
                merge_losses: List[float] = []
                for idx, st in enumerate(merge_states):
                    if idx >= len(doc.merge_counts_balanced):
                        break
                    pred = model.predict_norm_from_state(st)
                    truth = torch.tensor(
                        float(doc.merge_counts_balanced[idx]) / float(model.target_scale),
                        device=pred.device,
                        dtype=pred.dtype,
                    )
                    merge_losses.append(
                        float(F.mse_loss(pred, truth, reduction="mean").detach().cpu())
                    )
                if merge_losses:
                    scaled_merge_values = [
                        float(loss) / float(len(merge_losses)) for loss in merge_losses
                    ]
                    merge_population_values.extend(float(v) for v in scaled_merge_values)
                    q_internal = audit_sample_count(
                        n_internal,
                        policy=str(audit_policy),
                        fixed_nodes=int(audit_fixed_nodes),
                        fraction=float(audit_fraction),
                        scale=float(audit_scale),
                    )
                    inclusion_probs, support_mode = _markov_internal_inclusion_probabilities(
                        n_internal,
                        q=int(q_internal),
                        strategy=str(c3_audit_strategy),
                        merge_sizes=doc.merge_sizes_balanced,
                        include_root=bool(c3_include_root),
                    )
                    merge_support_mode = str(support_mode)
                    if inclusion_probs is None:
                        merge_support_ok = False
                    else:
                        sampled_internal = _sample_internal_audit_indices(
                            n_internal,
                            k=int(q_internal),
                            strategy=str(c3_audit_strategy),
                            merge_sizes=doc.merge_sizes_balanced,
                            include_root=bool(c3_include_root),
                            rng=rng,
                        )
                        if sampled_internal is None:
                            internal_indices = list(range(n_internal))
                        else:
                            internal_indices = list(sampled_internal)
                        if np.any(inclusion_probs <= 0.0):
                            merge_support_ok = False
                        for idx in internal_indices:
                            if idx >= len(scaled_merge_values):
                                continue
                            pi_merge = float(inclusion_probs[int(idx)])
                            if pi_merge <= 0.0:
                                continue
                            merge_samples.append(
                                TreeSample(
                                    doc_id=f"markov_eval_{doc_idx}",
                                    node_id=f"merge_{idx}",
                                    node_type=NodeType.MERGE,
                                    violation=0,
                                    preference_loss=float(scaled_merge_values[int(idx)]),
                                    propensity=TreePropensity(node=float(pi_merge)),
                                )
                            )

    task_estimates = _constant_estimator_map(float(exact_objective.raw_root_loss))
    local_law_estimates: Dict[str, Dict[str, float]] = {
        "c1": {"exact": float(exact_objective.raw_leaf_loss)},
        "c2": _constant_estimator_map(float(exact_objective.raw_c2_loss)),
        "c3": {"exact": float(exact_objective.raw_merge_loss)},
    }
    proxy_estimates = {
        "schedule_consistency": _constant_estimator_map(
            float(exact_objective.raw_schedule_consistency_loss)
        )
    }
    estimator_diagnostics: Dict[str, Any] = {
        "population_size_docs": float(population_size),
        "leaf_support_ok": bool(leaf_support_ok),
        "merge_support_ok": bool(merge_support_ok),
        "merge_support_mode": str(merge_support_mode),
    }

    if leaf_support_ok and leaf_samples:
        leaf_eval = _markov_normalized_ci_and_coverage(
            leaf_samples,
            exact_value=float(exact_objective.raw_leaf_loss),
            raw_values_population=leaf_population_values,
            population_size=float(population_size),
            delta=float(objective_ci_delta),
        )
        local_law_estimates["c1"].update(
            {
                "ht": float(leaf_eval.get("ht_mean", float("nan"))),
                "hajek": float(leaf_eval.get("hajek", float("nan"))),
                "eb_lo": float(leaf_eval.get("eb_lo", float("nan"))),
                "eb_hi": float(leaf_eval.get("eb_hi", float("nan"))),
            }
        )
        estimator_diagnostics["c1"] = dict(leaf_eval)
    else:
        estimator_diagnostics["c1"] = {
            "sample_count": float(len(leaf_samples)),
            "effective_sample_size": float(effective_sample_size(leaf_samples)),
            "max_weight": float(max_weight(leaf_samples)),
        }

    if merge_support_ok and merge_samples:
        merge_eval = _markov_normalized_ci_and_coverage(
            merge_samples,
            exact_value=float(exact_objective.raw_merge_loss),
            raw_values_population=merge_population_values,
            population_size=float(population_size),
            delta=float(objective_ci_delta),
        )
        local_law_estimates["c3"].update(
            {
                "ht": float(merge_eval.get("ht_mean", float("nan"))),
                "hajek": float(merge_eval.get("hajek", float("nan"))),
                "eb_lo": float(merge_eval.get("eb_lo", float("nan"))),
                "eb_hi": float(merge_eval.get("eb_hi", float("nan"))),
            }
        )
        estimator_diagnostics["c3"] = dict(merge_eval)
    else:
        estimator_diagnostics["c3"] = {
            "sample_count": float(len(merge_samples)),
            "effective_sample_size": float(effective_sample_size(merge_samples)),
            "max_weight": float(max_weight(merge_samples)),
            "support_mode": str(merge_support_mode),
        }

    prefer_hajek = False
    if float(objective_spec.local_law_weights.get("c1", 0.0)) > 0.0 or float(
        objective_spec.local_law_weights.get("c3", 0.0)
    ) > 0.0:
        c1_ready = (
            float(objective_spec.local_law_weights.get("c1", 0.0)) <= 0.0
            or math.isfinite(_maybe_float(local_law_estimates["c1"].get("hajek")))
        )
        c3_ready = (
            float(objective_spec.local_law_weights.get("c3", 0.0)) <= 0.0
            or math.isfinite(_maybe_float(local_law_estimates["c3"].get("hajek")))
        )
        prefer_hajek = bool(c1_ready and c3_ready and (c1_ready or c3_ready))

    payload = scalarize_objective_estimates(
        objective_spec,
        task_estimates=task_estimates,
        local_law_estimates=local_law_estimates,
        proxy_estimates=proxy_estimates,
        selection_preference=("hajek" if prefer_hajek else "exact"),
    )
    payload["estimator_diagnostics"] = estimator_diagnostics
    return payload


def _markov_composite_objective_spec(
    objective_summary: Mapping[str, Any],
) -> CompositeObjectiveSpec:
    composite = dict(objective_summary.get("composite_objective", {}) or {})
    return CompositeObjectiveSpec(
        name=str(composite.get("name", "configured_objective")),
        selection_metric_name=str(composite.get("selection_metric_name", "configured_objective")),
        task_name=str(composite.get("task_name", "task_objective")),
        task_weight=float(composite.get("task_weight", 0.0)),
        local_law_weights={
            str(name): float(value)
            for name, value in dict(composite.get("local_law_weights", {}) or {}).items()
        },
        proxy_weights={
            str(name): float(value)
            for name, value in dict(composite.get("proxy_weights", {}) or {}).items()
        },
        weighting_scheme=str(composite.get("weighting_scheme", "explicit_weighted_sum")),
        task_weight_source=str(composite.get("task_weight_source", "")),
        metadata={
            "task_metric_name": "root_error",
            "local_law_metric_names": {
                "c1": "c1",
                "c2": "c2",
                "c3": "c3",
            },
            "proxy_metric_names": {
                "schedule_consistency": "schedule_spread",
            },
        },
    )


def _markov_objective_metrics(
    *,
    local_metrics: Mapping[str, object],
    downstream_metrics: Mapping[str, object],
    objective_summary: Mapping[str, Any],
    split_name: str,
    split_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    composite = dict(objective_summary.get("composite_objective", {}) or {})
    objective_name = str(composite.get("name", composite.get("selection_metric_name", "configured_objective")))
    selection_metric_name = str(composite.get("selection_metric_name", objective_name))
    task_metric_name = str(objective_summary.get("task_objective_name", "root_error"))
    full_objective_value = float("nan")
    task_objective_value = float("nan")
    task_objective_term = float("nan")
    local_law_objective_value = float("nan")
    local_law_objective_term = float("nan")
    proxy_objective_value = float("nan")
    proxy_objective_term = float("nan")
    value_source = "recomputed_from_normalized_metrics"
    estimator_payload: Dict[str, Any] = {}
    if split_payload is not None:
        payload = dict(split_payload)
        estimator_payload = dict(payload.get(f"{split_name}_objective_estimator_payload", {}) or {})
        full_objective_value = _maybe_float(payload.get(f"{split_name}_objective_full_labels"))
        task_objective_value = _maybe_float(
            payload.get(f"{split_name}_unweighted_objective_task_objective_term")
        )
        task_objective_term = _maybe_float(
            payload.get(f"{split_name}_objective_task_objective_term")
        )
        local_law_objective_value = float(
            sum(
                _maybe_float(
                    payload.get(f"{split_name}_unweighted_objective_{suffix}"),
                    0.0,
                )
                for suffix in ("leaf_term", "c2_term", "merge_term")
            )
        )
        local_law_objective_term = float(
            sum(
                _maybe_float(payload.get(f"{split_name}_objective_{suffix}"), 0.0)
                for suffix in ("leaf_term", "c2_term", "merge_term")
            )
        )
        proxy_objective_value = _maybe_float(
            payload.get(f"{split_name}_unweighted_objective_schedule_consistency_term"),
            0.0,
        )
        proxy_objective_term = _maybe_float(
            payload.get(f"{split_name}_objective_schedule_consistency_term"),
            0.0,
        )
        if estimator_payload:
            selection_metric_name = str(
                estimator_payload.get("selection_metric_name", selection_metric_name)
            )
        if math.isfinite(full_objective_value):
            value_source = "reported_split_payload"
    if not math.isfinite(full_objective_value):
        objective_eval = evaluate_composite_objective_from_metrics(
            _markov_composite_objective_spec(objective_summary),
            metrics={
                "root_error": float(downstream_metrics.get("root_error", float("nan"))),
                "c1": float(local_metrics.get("c1", float("nan"))),
                "c2": float(local_metrics.get("c2", float("nan"))),
                "c3": float(local_metrics.get("c3", float("nan"))),
                "schedule_spread": float(downstream_metrics.get("schedule_spread", float("nan"))),
            },
        )
        full_objective_value = float(objective_eval.total)
        if not math.isfinite(task_objective_value):
            task_objective_value = float(objective_eval.task_raw)
        if not math.isfinite(task_objective_term):
            task_objective_term = float(objective_eval.task_term)
        if not math.isfinite(local_law_objective_value):
            local_law_objective_value = float(
                sum(float(v) for v in objective_eval.local_law_raw.values())
            )
        if not math.isfinite(local_law_objective_term):
            local_law_objective_term = float(
                sum(float(v) for v in objective_eval.local_law_terms.values())
            )
        if not math.isfinite(proxy_objective_value):
            proxy_objective_value = float(sum(float(v) for v in objective_eval.proxy_raw.values()))
        if not math.isfinite(proxy_objective_term):
            proxy_objective_term = float(sum(float(v) for v in objective_eval.proxy_terms.values()))
    task_weight = float(objective_summary.get("task_objective_weight", 0.0))
    local_law_weight_total = float(
        sum(float(v) for v in dict(composite.get("local_law_weights", {}) or {}).values())
    )
    proxy_weight_total = float(
        sum(float(v) for v in dict(composite.get("proxy_weights", {}) or {}).values())
    )
    total_weight_without_proxy = float(
        composite.get(
            "total_weight_without_proxy",
            objective_summary.get("optimization_weight_mass_no_proxy", float("nan")),
        )
    )
    normalized_task_share = (
        float(task_weight / total_weight_without_proxy)
        if math.isfinite(total_weight_without_proxy) and total_weight_without_proxy > 0.0
        else float("nan")
    )
    normalized_local_law_share = (
        float(local_law_weight_total / total_weight_without_proxy)
        if math.isfinite(total_weight_without_proxy) and total_weight_without_proxy > 0.0
        else float("nan")
    )
    out = {
        "objective_name": objective_name,
        "selection_metric_name": selection_metric_name,
        "weighting_scheme": str(objective_summary.get("weighting_scheme", "")),
        "task_metric_name": task_metric_name,
        "task_weight": task_weight,
        "local_law_weight_total": local_law_weight_total,
        "proxy_weight_total": proxy_weight_total,
        "total_weight_without_proxy": total_weight_without_proxy,
        "lambda_local_law": float(objective_summary.get("local_law_weight", float("nan"))),
        "normalized_task_share": normalized_task_share,
        "normalized_local_law_share": normalized_local_law_share,
        "full_objective_value": float(full_objective_value),
        "task_objective_value": float(task_objective_value),
        "task_objective_term": float(task_objective_term),
        "regular_objective_value": float(task_objective_value),
        "regular_objective_term": float(task_objective_term),
        "local_law_objective_value": float(local_law_objective_value),
        "local_law_objective_term": float(local_law_objective_term),
        "proxy_objective_value": float(proxy_objective_value),
        "proxy_objective_term": float(proxy_objective_term),
        "value_source": str(value_source),
        "local_law_weights": {
            str(name): float(value)
            for name, value in dict(composite.get("local_law_weights", {}) or {}).items()
        },
        "proxy_weights": {
            str(name): float(value)
            for name, value in dict(composite.get("proxy_weights", {}) or {}).items()
        },
    }
    if estimator_payload:
        out["selection_estimator"] = str(estimator_payload.get("selection_estimator", "exact"))
        out["selection_metric_value"] = float(
            estimator_payload.get("selection_metric_value", float("nan"))
        )
        out["available_estimators"] = list(estimator_payload.get("available_estimators", []))
        out["estimator_components"] = dict(
            estimator_payload.get("estimator_components", {}) or {}
        )
        for estimator in OBJECTIVE_ESTIMATOR_KEYS:
            alias = objective_estimator_alias(objective_name, estimator)
            if alias in estimator_payload:
                out[str(alias)] = float(estimator_payload[alias])
        width_key = f"{objective_name}_eb_width"
        if width_key in estimator_payload:
            out[width_key] = float(estimator_payload[width_key])
        selection_value_key = f"{objective_name}_selection_value"
        if selection_value_key in estimator_payload:
            out[selection_value_key] = float(estimator_payload[selection_value_key])
        if "estimator_diagnostics" in estimator_payload:
            out["estimator_diagnostics"] = dict(
                estimator_payload.get("estimator_diagnostics", {}) or {}
            )
    return out


def _markov_split_id(*, split: str, seed: int, n_docs: int) -> str:
    return f"markov:{str(split)}:seed={int(seed)}:docs={int(n_docs)}"


def _markov_artifact_dir(config: OPSCountConfig) -> Optional[Path]:
    text = str(config.artifact_dir).strip()
    if not text:
        return None
    path = Path(text)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _markov_analytic_artifact(
    *,
    output_dir: Optional[Path],
    artifact_id: str,
    name: str,
    role: PolicyRole,
    config: OPSCountConfig,
    target_scale: float,
    notes: str,
    targeted_laws: Optional[Sequence[str]] = None,
) -> Optional[GArtifact]:
    if output_dir is None:
        return None
    return write_json_g_artifact(
        output_dir=output_dir,
        artifact_id=str(artifact_id),
        name=str(name),
        role=role,
        family="markov_ops_count",
        dgp="markov_changepoint_ops_count",
        payload={
            "model_family": "analytic",
            "feature_mode": str(config.feature_mode),
            "n_regimes": int(config.n_regimes),
            "target_scale": float(target_scale),
            "state_layout": str(name),
            "merge_semantics": str(notes),
            "resummary_semantics": (
                "identity" if str(name) != "flip_R2" else "toggle-flip bit on each resummary"
            ),
            "targeted_laws": [str(x) for x in (targeted_laws or [])],
        },
        metadata={
            "suite_role": str(config.suite_role),
            "law_package": str(config.law_package),
            "exact_family": str(config.exact_family),
        },
    )


def _markov_model_artifact(
    *,
    output_dir: Optional[Path],
    artifact_id: str,
    role: PolicyRole,
    name: str,
    model: object,
    config: OPSCountConfig,
    target_scale: float,
) -> Optional[GArtifact]:
    if output_dir is None:
        return None

    arrays: Dict[str, np.ndarray] = {}
    manifest_payload: Dict[str, Any] = {
        "model_family": str(config.model_family),
        "feature_mode": str(config.feature_mode),
        "n_regimes": int(config.n_regimes),
        "target_scale": float(target_scale),
        "state_layout": "",
        "merge_semantics": "",
        "readout_semantics": "",
        "resummary_semantics": "decode_summary/encode_summary over full sketch state summary",
    }

    if hasattr(model, "encoder") and isinstance(getattr(model, "encoder"), nn.Linear):
        enc = getattr(model, "encoder")
        arrays["encoder_weight"] = enc.weight.detach().cpu().numpy()
        arrays["encoder_bias"] = enc.bias.detach().cpu().numpy()
        manifest_payload["state_layout"] = "normalized_count + first_endpoint + last_endpoint"
        manifest_payload["merge_semantics"] = (
            "additive count merge with explicit boundary correction when endpoints differ"
        )
        manifest_payload["readout_semantics"] = "identity normalized count readout"
    elif hasattr(model, "encoder") and hasattr(model, "merger"):
        enc = getattr(model, "encoder")
        mer = getattr(model, "merger")
        sum_enc = getattr(model, "summary_encoder")
        readout = getattr(model, "readout")
        arrays["encoder_linear0_weight"] = enc[0].weight.detach().cpu().numpy()
        arrays["encoder_linear0_bias"] = enc[0].bias.detach().cpu().numpy()
        arrays["encoder_linear1_weight"] = enc[2].weight.detach().cpu().numpy()
        arrays["encoder_linear1_bias"] = enc[2].bias.detach().cpu().numpy()
        arrays["merger_linear0_weight"] = mer[0].weight.detach().cpu().numpy()
        arrays["merger_linear0_bias"] = mer[0].bias.detach().cpu().numpy()
        arrays["merger_linear1_weight"] = mer[2].weight.detach().cpu().numpy()
        arrays["merger_linear1_bias"] = mer[2].bias.detach().cpu().numpy()
        arrays["readout_weight"] = readout.weight.detach().cpu().numpy()
        arrays["readout_bias"] = readout.bias.detach().cpu().numpy()
        arrays["summary_linear0_weight"] = sum_enc[0].weight.detach().cpu().numpy()
        arrays["summary_linear0_bias"] = sum_enc[0].bias.detach().cpu().numpy()
        arrays["summary_linear1_weight"] = sum_enc[2].weight.detach().cpu().numpy()
        arrays["summary_linear1_bias"] = sum_enc[2].bias.detach().cpu().numpy()
        manifest_payload["state_layout"] = "latent_state + first_endpoint + last_endpoint"
        manifest_payload["merge_semantics"] = (
            "learned merger over left/right latent states plus boundary endpoints"
        )
        manifest_payload["readout_semantics"] = "sigmoid(readout(latent_state)) * target_scale"
    else:
        return None

    return write_npz_g_artifact(
        output_dir=output_dir,
        artifact_id=str(artifact_id),
        name=str(name),
        role=role,
        family="markov_ops_count",
        dgp="markov_changepoint_ops_count",
        manifest_payload=manifest_payload,
        arrays=arrays,
        metadata={
            "suite_role": str(config.suite_role),
            "law_package": str(config.law_package),
            "exact_family": str(config.exact_family),
        },
    )


def _policy_split_payload(
    metrics: SketchMetrics,
    *,
    target_scale: float,
    split_name: str,
    objective_summary: Mapping[str, Any],
    split_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    local_metrics = _markov_local_metrics(
        metrics,
        target_scale=float(target_scale),
    ).to_dict()
    downstream_metrics = _markov_downstream_metrics(
        metrics,
        target_scale=float(target_scale),
    ).to_dict()
    return {
        "local_law_metrics": local_metrics,
        "downstream_metrics": downstream_metrics,
        "objective_metrics": _markov_objective_metrics(
            local_metrics=local_metrics,
            downstream_metrics=downstream_metrics,
            objective_summary=objective_summary,
            split_name=str(split_name),
            split_payload=split_payload,
        ),
    }


def _build_markov_local_law_learnability(
    *,
    config: OPSCountConfig,
    seeds: Mapping[str, int],
    target_scale: float,
    objective_summary: Mapping[str, Any],
    geom: TrainingGeometry,
    exact: SketchMetrics,
    leaf_bucket: SketchMetrics,
    undersupported: SketchMetrics,
    flip_r2: SketchMetrics,
    current_name: str,
    current_role: str,
    current_train: Optional[SketchMetrics],
    current_val: Optional[SketchMetrics],
    current_test: SketchMetrics,
    current_selection_metric_name: str,
    current_selection_metric: float,
    current_train_payload: Optional[Mapping[str, Any]] = None,
    current_val_payload: Optional[Mapping[str, Any]] = None,
    current_test_payload: Optional[Mapping[str, Any]] = None,
    model: Optional[object] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    artifact_dir = _markov_artifact_dir(config)
    artifacts: List[GArtifact] = []
    oracle_artifact = _markov_analytic_artifact(
        output_dir=artifact_dir,
        artifact_id="oracle_g",
        name="oracle_g",
        role=PolicyRole.ORACLE_G,
        config=config,
        target_scale=float(target_scale),
        notes="exact changepoint count sketch with endpoints and count",
        targeted_laws=[],
    )
    if oracle_artifact is not None:
        artifacts.append(oracle_artifact)
    current_artifact = (
        _markov_model_artifact(
            output_dir=artifact_dir,
            artifact_id=str(current_role),
            role=PolicyRole(str(current_role)),
            name=str(current_name),
            model=model,
            config=config,
            target_scale=float(target_scale),
        )
        if model is not None
        else None
    )
    if current_artifact is not None:
        artifacts.append(current_artifact)
    counterexample_specs = [
        (
            "leaf_bucket",
            leaf_bucket,
            ["C1"],
            "leaf summaries collapse to a bucket identity and break leaf preservation",
        ),
        (
            "count_only",
            undersupported,
            ["C3"],
            "count-only merge omits boundary state and breaks merge preservation",
        ),
        (
            "flip_R2",
            flip_r2,
            ["C2"],
            "resummary toggles a hidden flip bit and breaks idempotence",
        ),
    ]
    counterexamples: List[LocalLawCounterexampleEvaluation] = []
    for name, metrics, targeted_laws, notes in counterexample_specs:
        artifact = _markov_analytic_artifact(
            output_dir=artifact_dir,
            artifact_id=str(name),
            name=str(name),
            role=PolicyRole.COUNTEREXAMPLE_G,
            config=config,
            target_scale=float(target_scale),
            notes=str(notes),
            targeted_laws=targeted_laws,
        )
        if artifact is not None:
            artifacts.append(artifact)
        counterexamples.append(
            LocalLawCounterexampleEvaluation(
                name=str(name),
                role=PolicyRole.COUNTEREXAMPLE_G,
                targeted_laws=[str(x) for x in targeted_laws],
                artifact_id=(artifact.artifact_id if artifact is not None else None),
                metrics={
                    "test": _policy_split_payload(
                        metrics,
                        target_scale=float(target_scale),
                        split_name="test",
                        objective_summary=objective_summary,
                    )
                },
                metadata={"note": str(notes)},
            )
        )

    policy_role = PolicyRole(str(current_role))
    split_metrics: Dict[str, Dict[str, Any]] = {
        "test": _policy_split_payload(
            current_test,
            target_scale=float(target_scale),
            split_name="test",
            objective_summary=objective_summary,
            split_payload=current_test_payload,
        )
    }
    if current_train is not None:
        split_metrics["train"] = _policy_split_payload(
            current_train,
            target_scale=float(target_scale),
            split_name="train",
            objective_summary=objective_summary,
            split_payload=current_train_payload,
        )
    if current_val is not None:
        split_metrics["val"] = _policy_split_payload(
            current_val,
            target_scale=float(target_scale),
            split_name="val",
            objective_summary=objective_summary,
            split_payload=current_val_payload,
        )

    policies = {
        "oracle_g": LocalLawPolicyEvaluation(
            name="oracle_g",
            role=PolicyRole.ORACLE_G,
            artifact_id=(oracle_artifact.artifact_id if oracle_artifact is not None else None),
            split_metrics={
                "test": _policy_split_payload(
                    exact,
                    target_scale=float(target_scale),
                    split_name="test",
                    objective_summary=objective_summary,
                )
            },
            metadata={"law_package": "exact"},
        ),
        str(current_name): LocalLawPolicyEvaluation(
            name=str(current_name),
            role=policy_role,
            artifact_id=(current_artifact.artifact_id if current_artifact is not None else None),
            selection_metric_value=float(current_selection_metric),
            split_metrics=split_metrics,
            metadata={
                "law_package": str(config.law_package),
                "model_family": str(config.model_family),
            },
        ),
    }

    summary = LocalLawRunSummary(
        family="markov_ops_count",
        dgp="markov_changepoint_ops_count",
        oracle_name="changepoint_count_exact_summary",
        study_role=str(current_role),
        split_ids={
            "train": _markov_split_id(
                split="train",
                seed=int(seeds["effective_data_seed"]),
                n_docs=int(config.train_docs),
            ),
            "val": _markov_split_id(
                split="val",
                seed=int(seeds["effective_val_seed"]),
                n_docs=int(config.val_docs),
            ),
            "test": _markov_split_id(
                split="test",
                seed=int(seeds["effective_test_seed"]),
                n_docs=int(config.test_docs),
            ),
        },
        support_budget=SupportBudgetSummary(
            train_docs=int(config.train_docs),
            val_docs=int(config.val_docs),
            test_docs=int(config.test_docs),
            leaf_query_rate=float(config.leaf_query_rate),
            internal_query_rate=(
                float(geom.mean_internal_labels) / float(max(1.0, geom.mean_internal_nodes))
            ),
            root_query_rate=1.0 if bool(config.include_root_query) else 0.0,
            mean_leaf_labels_per_doc=float(geom.mean_leaf_labels),
            mean_internal_labels_per_doc=float(geom.mean_internal_labels),
            mean_queries_per_doc=float(geom.mean_queries_per_doc),
            total_queries_estimate=float(geom.total_queries_estimate),
            metadata={
                "audit_fraction": float(config.audit_fraction),
                "audit_policy": str(config.audit_policy),
            },
        ),
        selection={
            "selection_split": "val" if int(config.val_docs) > 0 else "config",
            "selection_metric": (
                "configured_exact_family"
                if str(current_role) == PolicyRole.COUNTEREXAMPLE_G.value
                else str(current_selection_metric_name or "configured_objective")
            ),
            "selected_candidate": str(current_name),
            "uses_test_metrics": False,
            "selection_reason": (
                "fixed by configured law_package"
                if str(current_role) != PolicyRole.COUNTEREXAMPLE_G.value
                else "fixed by configured exact_family"
            ),
        },
        policies=policies,
        counterexamples=counterexamples,
        thresholds={
            "c1_tau": float(config.violation_tau) / float(max(1.0, target_scale)),
            "c2_tau": float(config.violation_tau) / float(max(1.0, target_scale)),
            "c3_tau": float(config.violation_tau) / float(max(1.0, target_scale)),
        },
        suite_role=str(config.suite_role),
        metadata={
            "law_package": str(config.law_package),
            "exact_family": str(config.exact_family),
            "root_only_reference_package": "root_only",
            "all_laws_reference_package": "all_laws_plus_sched",
            "configured_objective_name": str(
                dict(objective_summary.get("composite_objective", {}) or {}).get(
                    "selection_metric_name",
                    "configured_objective",
                )
            ),
            "objective": dict(objective_summary.get("composite_objective", {}) or {}),
        },
    )
    summary = attach_local_law_learning_problem(summary)
    return summary.to_dict(), artifact_index(artifacts)


def _resolve_runtime_seeds(config: OPSCountConfig) -> Dict[str, int]:
    shared_seed = int(config.seed)
    data_seed = int(shared_seed if config.data_seed is None else config.data_seed)
    model_seed = int(shared_seed if config.model_seed is None else config.model_seed)
    val_seed = int(data_seed + int(config.val_seed_offset))
    test_seed = int(data_seed + int(config.test_seed_offset))
    return {
        "seed": int(shared_seed),
        "effective_data_seed": int(data_seed),
        "effective_model_seed": int(model_seed),
        "effective_val_seed": int(val_seed),
        "effective_test_seed": int(test_seed),
    }


def _law_package_weights(config: OPSCountConfig) -> Optional[Dict[str, float]]:
    package = str(config.law_package or "").strip().lower()
    if not package:
        return None
    if package not in VALID_LAW_PACKAGES:
        raise ValueError(
            f"law_package={package!r} unsupported; expected one of {VALID_LAW_PACKAGES}"
        )

    if config.local_law_weight is not None:
        lambda_local = float(max(0.0, config.local_law_weight))
        lambda_defaulted = False
    elif package in {"root_only", "sched_only"}:
        lambda_local = 0.0
        lambda_defaulted = False
    else:
        lambda_local = float(DEFAULT_NORMALIZED_LOCAL_LAW_WEIGHT)
        lambda_defaulted = True
    sched_scale = (
        float(config.schedule_consistency_weight)
        if float(config.schedule_consistency_weight) > 0.0
        else 0.1
    )

    law_sets = {
        "root_only": (),
        "c1_only": ("c1",),
        "c2_only": ("c2",),
        "c3_only": ("c3",),
        "c1c3": ("c1", "c3"),
        "all_laws": ("c1", "c2", "c3"),
        "sched_only": (),
        "all_laws_plus_sched": ("c1", "c2", "c3"),
    }
    active = tuple(law_sets[package])
    if not active:
        lambda_local = 0.0
    share = float(lambda_local / float(len(active))) if active else 0.0
    weights = {"c1": 0.0, "c2": 0.0, "c3": 0.0, "schedule": 0.0}
    for key in active:
        weights[key] = float(share)
    if package in {"sched_only", "all_laws_plus_sched"}:
        weights["schedule"] = float(max(0.0, sched_scale))
    weights["lambda_local"] = float(lambda_local)
    weights["lambda_defaulted"] = 1.0 if lambda_defaulted else 0.0
    return weights


def _resolve_local_law_weights(config: OPSCountConfig) -> Dict[str, float | str]:
    package_weights = _law_package_weights(config)
    legacy_c1 = float(config.leaf_weight)
    legacy_c2 = float(config.c2_weight)
    legacy_c3 = float(config.c3_weight)
    configured_task_weight = (
        float(max(0.0, config.task_objective_weight))
        if config.task_objective_weight is not None
        else None
    )
    if package_weights is not None:
        effective_c1 = float(package_weights["c1"])
        effective_c2 = float(package_weights["c2"])
        effective_c3 = float(package_weights["c3"])
        lambda_local = float(package_weights["lambda_local"])
        if lambda_local > 0.0:
            c1_share = float(effective_c1 / lambda_local)
            c2_share = float(effective_c2 / lambda_local)
            c3_share = float(effective_c3 / lambda_local)
        else:
            c1_share = 0.0
            c2_share = 0.0
            c3_share = 0.0
        proxy_weight = float(package_weights["schedule"])
        parameterization = "law_package"
        if configured_task_weight is None:
            weighting_scheme = "normalized_lambda_tradeoff"
            optimization_root_weight = float(max(0.0, 1.0 - lambda_local))
            task_objective_weight_source = "derived_from_local_law_weight"
        else:
            weighting_scheme = "explicit_task_plus_local_law"
            optimization_root_weight = float(configured_task_weight)
            task_objective_weight_source = "explicit_task_objective_weight"
        lambda_defaulted = bool(float(package_weights["lambda_defaulted"]) > 0.0)
    elif config.local_law_weight is None:
        lambda_local = float(max(0.0, legacy_c1 + legacy_c2 + legacy_c3))
        if lambda_local > 0.0:
            c1_share = float(max(0.0, legacy_c1) / lambda_local)
            c2_share = float(max(0.0, legacy_c2) / lambda_local)
            c3_share = float(max(0.0, legacy_c3) / lambda_local)
        else:
            c1_share = 0.0
            c2_share = 0.0
            c3_share = 0.0
        effective_c1 = float(max(0.0, legacy_c1))
        effective_c2 = float(max(0.0, legacy_c2))
        effective_c3 = float(max(0.0, legacy_c3))
        proxy_weight = float(max(0.0, config.schedule_consistency_weight))
        parameterization = "legacy_term_weights"
        weighting_scheme = "legacy_additive_weights"
        if configured_task_weight is None:
            optimization_root_weight = float(max(0.0, config.root_weight))
            task_objective_weight_source = "legacy_root_weight"
        else:
            optimization_root_weight = float(configured_task_weight)
            task_objective_weight_source = "explicit_task_objective_weight"
        lambda_defaulted = False
    else:
        lambda_local = float(max(0.0, config.local_law_weight))
        c1_rel = float(max(0.0, config.c1_relative_weight))
        c2_rel = float(max(0.0, config.c2_relative_weight))
        c3_rel = float(max(0.0, config.c3_relative_weight))
        rel_total = float(c1_rel + c2_rel + c3_rel)
        if lambda_local > 0.0 and rel_total > 0.0:
            c1_share = float(c1_rel / rel_total)
            c2_share = float(c2_rel / rel_total)
            c3_share = float(c3_rel / rel_total)
            effective_c1 = float(lambda_local * c1_share)
            effective_c2 = float(lambda_local * c2_share)
            effective_c3 = float(lambda_local * c3_share)
        else:
            c1_share = 0.0
            c2_share = 0.0
            c3_share = 0.0
            effective_c1 = 0.0
            effective_c2 = 0.0
            effective_c3 = 0.0
        proxy_weight = float(max(0.0, config.schedule_consistency_weight))
        parameterization = "formal_local_law_weight"
        if configured_task_weight is None:
            weighting_scheme = "normalized_lambda_tradeoff"
            optimization_root_weight = float(max(0.0, 1.0 - lambda_local))
            task_objective_weight_source = "derived_from_local_law_weight"
        else:
            weighting_scheme = "explicit_task_plus_local_law"
            optimization_root_weight = float(configured_task_weight)
            task_objective_weight_source = "explicit_task_objective_weight"
        lambda_defaulted = False

    return {
        "parameterization": str(parameterization),
        "weighting_scheme": str(weighting_scheme),
        "law_package": str(config.law_package or ""),
        "local_law_weight": float(lambda_local),
        "local_law_lambda": float(lambda_local),
        "task_objective_weight": float(optimization_root_weight),
        "configured_task_objective_weight": (
            float(configured_task_weight) if configured_task_weight is not None else None
        ),
        "task_objective_weight_source": str(task_objective_weight_source),
        "local_law_c1_weight": float(effective_c1),
        "local_law_c2_weight": float(effective_c2),
        "local_law_c3_weight": float(effective_c3),
        "local_law_c1_share": float(c1_share),
        "local_law_c2_share": float(c2_share),
        "local_law_c3_share": float(c3_share),
        "optimization_root_weight": float(optimization_root_weight),
        "optimization_weight_mass_no_proxy": float(
            optimization_root_weight + effective_c1 + effective_c2 + effective_c3
        ),
        "lambda_defaulted_from_lean_default": bool(lambda_defaulted),
        "legacy_leaf_weight": float(legacy_c1),
        "legacy_c2_weight": float(legacy_c2),
        "legacy_c3_weight": float(legacy_c3),
        "proxy_schedule_consistency_weight": float(proxy_weight),
    }


def _build_objective_summary(config: OPSCountConfig) -> Dict[str, Any]:
    resolved = _resolve_local_law_weights(config)
    optimization_root_weight = float(resolved["optimization_root_weight"])
    root_active = bool(config.include_root_query) and optimization_root_weight > 0.0
    proxy_weight = float(resolved["proxy_schedule_consistency_weight"])
    uses_weighted_neural_objective = str(config.model_family) == "neural"
    weighting_scheme = str(resolved["weighting_scheme"])
    if weighting_scheme == "legacy_additive_weights":
        objective_formula = (
            "`root_weight * task + leaf_weight * C1 + c2_weight * C2 + c3_weight * C3`"
        )
    elif str(resolved["task_objective_weight_source"]) == "explicit_task_objective_weight":
        objective_formula = (
            "`task_objective_weight * task + local_law_penalties`, with local-law shares "
            "still controlled by λ and the relative C1/C2/C3 weights"
        )
    else:
        objective_formula = "`(1 - lambda) * task + lambda * local_laws`"
    composite_spec = CompositeObjectiveSpec(
        name="configured_objective",
        selection_metric_name="configured_objective",
        task_name="task_objective",
        task_weight=float(resolved["task_objective_weight"]),
        local_law_weights={
            "c1": float(resolved["local_law_c1_weight"]),
            "c2": float(resolved["local_law_c2_weight"]),
            "c3": float(resolved["local_law_c3_weight"]),
        },
        proxy_weights={"schedule_consistency": float(proxy_weight)},
        weighting_scheme=str(weighting_scheme),
        task_weight_source=str(resolved["task_objective_weight_source"]),
        metadata={
            "task_metric_name": "root_count_mse",
            "parameterization": str(resolved["parameterization"]),
            "local_law_weight": float(resolved["local_law_weight"]),
        },
    )
    theorem_terms = [
        {
            "law_kind": LawKind.L1_LEAF.value,
            "paper_condition": LawKind.L1_LEAF.paper_condition,
            "lean_name": LawKind.L1_LEAF.lean_name,
            "name": "leaf_preservation",
            "weight": float(resolved["local_law_c1_weight"]),
            "share_within_local_law": float(resolved["local_law_c1_share"]),
            "active": uses_weighted_neural_objective
            and float(resolved["local_law_c1_weight"]) > 0.0,
            "evidence_status": EvidenceStatus.THEOREM_BACKED.value,
        },
        {
            "law_kind": LawKind.L2_MERGE.value,
            "paper_condition": LawKind.L2_MERGE.paper_condition,
            "lean_name": LawKind.L2_MERGE.lean_name,
            "name": "merge_preservation",
            "weight": float(resolved["local_law_c3_weight"]),
            "share_within_local_law": float(resolved["local_law_c3_share"]),
            "active": uses_weighted_neural_objective
            and float(resolved["local_law_c3_weight"]) > 0.0,
            "evidence_status": EvidenceStatus.THEOREM_BACKED.value,
        },
        {
            "law_kind": LawKind.L3_IDEMPOTENCE.value,
            "paper_condition": LawKind.L3_IDEMPOTENCE.paper_condition,
            "lean_name": LawKind.L3_IDEMPOTENCE.lean_name,
            "name": "idempotence",
            "weight": float(resolved["local_law_c2_weight"]),
            "share_within_local_law": float(resolved["local_law_c2_share"]),
            "active": uses_weighted_neural_objective
            and float(resolved["local_law_c2_weight"]) > 0.0,
            "evidence_status": EvidenceStatus.THEOREM_BACKED.value,
        },
    ]
    proxy_terms = [
        {
            "name": "schedule_consistency",
            "weight": float(proxy_weight),
            "active": uses_weighted_neural_objective and float(proxy_weight) > 0.0,
            "evidence_status": EvidenceStatus.PROXY_ONLY.value,
            "notes": "Associativity proxy over schedule spread; not a Lean local law.",
        }
    ]
    return {
        **resolved,
        "model_family": str(config.model_family),
        "training_scheme": (
            "weighted_neural_objective"
            if uses_weighted_neural_objective
            else "closed_form_label_fit"
        ),
        "root_weight": float(config.root_weight),
        "task_objective_name": "root_count_mse",
        "task_objective_weight": float(resolved["task_objective_weight"]),
        "task_objective_weight_source": str(resolved["task_objective_weight_source"]),
        "optimization_root_weight": float(optimization_root_weight),
        "root_supervision_active": bool(uses_weighted_neural_objective and root_active),
        "task_supervision_active": bool(uses_weighted_neural_objective and root_active),
        "proxy_schedule_consistency_weight": float(proxy_weight),
        "local_law_active": bool(
            uses_weighted_neural_objective and float(resolved["local_law_weight"]) > 0.0
        ),
        "parameterization_overrides_legacy": bool(
            config.local_law_weight is not None or str(config.law_package or "").strip()
        ),
        "composite_objective": composite_spec.to_dict(),
        "theorem_terms": theorem_terms,
        "proxy_terms": proxy_terms,
        "formal_notes": (
            "The theorem-facing local-law bundle covers C1/L1 leaf preservation, "
            "C2/L3 on-range idempotence, and C3/L2 merge preservation. "
            f"The active task/local-law objective is {objective_formula}, with "
            "schedule_consistency reported separately as a proxy-only regularizer."
        ),
        "model_family_notes": (
            "These weights are active in the neural lane. The additive lane instead uses "
            "closed-form regression on the available C1/C3 labels and exact scalar re-summary for C2."
        ),
    }


@dataclass(frozen=True)
class _ExactState:
    count: int
    first: int
    last: int


def _exact_from_span(doc: ChangepointMarkovDoc, span: Tuple[int, int]) -> _ExactState:
    start, end = span
    regs = doc.token_regimes[int(start) : int(end)]
    if len(regs) == 0:
        raise ValueError("empty span")
    return _ExactState(
        count=_changepoint_count(regs),
        first=int(regs[0]),
        last=int(regs[-1]),
    )


def _exact_merge(a: _ExactState, b: _ExactState) -> _ExactState:
    join = 0 if int(a.last) == int(b.first) else 1
    return _ExactState(
        count=int(a.count) + int(b.count) + int(join),
        first=int(a.first),
        last=int(b.last),
    )


@dataclass(frozen=True)
class _CountOnlyState:
    count: int


def _count_only_from_span(doc: ChangepointMarkovDoc, span: Tuple[int, int]) -> _CountOnlyState:
    start, end = span
    regs = doc.token_regimes[int(start) : int(end)]
    if len(regs) == 0:
        raise ValueError("empty span")
    return _CountOnlyState(count=_changepoint_count(regs))


def _count_only_merge(a: _CountOnlyState, b: _CountOnlyState) -> _CountOnlyState:
    return _CountOnlyState(count=int(a.count) + int(b.count))


@dataclass(frozen=True)
class _FlipState:
    count: int
    first: int
    last: int
    flipped: bool


def _flip_from_span(doc: ChangepointMarkovDoc, span: Tuple[int, int]) -> _FlipState:
    base = _exact_from_span(doc, span)
    return _FlipState(count=base.count, first=base.first, last=base.last, flipped=False)


def _flip_merge(a: _FlipState, b: _FlipState) -> _FlipState:
    base = _exact_merge(
        _ExactState(a.count, a.first, a.last), _ExactState(b.count, b.first, b.last)
    )
    return _FlipState(count=base.count, first=base.first, last=base.last, flipped=False)


def _flip_resummary(z: _FlipState) -> _FlipState:
    return _FlipState(
        count=int(z.count), first=int(z.first), last=int(z.last), flipped=not bool(z.flipped)
    )


def _flip_value(z: _FlipState) -> int:
    return int(z.count) + (1 if bool(z.flipped) else 0)


@dataclass(frozen=True)
class _LeafBucketState:
    exact_count: int
    readout_count: int
    first: int
    last: int


def _leaf_bucket_from_span(doc: ChangepointMarkovDoc, span: Tuple[int, int]) -> _LeafBucketState:
    base = _exact_from_span(doc, span)
    return _LeafBucketState(
        exact_count=int(base.count),
        readout_count=1,
        first=int(base.first),
        last=int(base.last),
    )


def _leaf_bucket_merge(a: _LeafBucketState, b: _LeafBucketState) -> _LeafBucketState:
    base = _exact_merge(
        _ExactState(a.exact_count, a.first, a.last),
        _ExactState(b.exact_count, b.first, b.last),
    )
    return _LeafBucketState(
        exact_count=int(base.count),
        readout_count=int(base.count),
        first=int(base.first),
        last=int(base.last),
    )


def _leaf_bucket_value(z: _LeafBucketState) -> int:
    return int(z.readout_count)


def _identity_resummary(z: StateT) -> StateT:
    return z


def _apply_resummary(z: StateT, *, rounds: int, resummary: Callable[[StateT], StateT]) -> StateT:
    cur = z
    for _ in range(int(max(0, rounds))):
        cur = resummary(cur)
    return cur


def _zero_sketch_metrics(*, n_docs: int) -> SketchMetrics:
    return SketchMetrics(
        root_mae=0.0,
        root_median_abs_error=0.0,
        root_p95_abs_error=0.0,
        schedule_spread_mean=0.0,
        schedule_spread_p95=0.0,
        leaf_mae=0.0,
        leaf_violation_rate=0.0,
        c2_idempotence_mae=0.0,
        c2_r1_mae=0.0,
        c2_r2_mae=0.0,
        c2_r4_mae=0.0,
        resummary_root_drift_r1=0.0,
        resummary_root_drift_r2=0.0,
        resummary_root_drift_r4=0.0,
        merge_mae=0.0,
        merge_violation_rate=0.0,
        n_docs=int(n_docs),
    )


def _eval_structured_family(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    leaf_tokens: int,
    tau: float,
    from_span: Callable[[ChangepointMarkovDoc, Tuple[int, int]], StateT],
    merge: Callable[[StateT, StateT], StateT],
    value: Callable[[StateT], float],
    resummary: Callable[[StateT], StateT],
) -> SketchMetrics:
    if len(docs) == 0:
        return _zero_sketch_metrics(n_docs=0)

    root_abs: List[float] = []
    spreads: List[float] = []
    leaf_abs: List[float] = []
    merge_abs: List[float] = []
    c2_r1_abs: List[float] = []
    c2_r2_abs: List[float] = []
    c2_r4_abs: List[float] = []
    root_drift_r1: List[float] = []
    root_drift_r2: List[float] = []
    root_drift_r4: List[float] = []

    for doc in docs:
        n_tok = int(len(doc.token_regimes))
        spans = _leaf_spans(n_tok, leaf_tokens=int(leaf_tokens))
        leaf_states = [from_span(doc, sp) for sp in spans]
        leaf_truth = [_oracle_count(doc, start=sp[0], end=sp[1]) for sp in spans]
        for st, truth in zip(leaf_states, leaf_truth):
            leaf_abs.append(abs(float(value(st)) - float(truth)))

        # Root predictions for schedule spread.
        roots: Dict[str, float] = {}
        balanced_root_state: Optional[StateT] = None
        balanced_states: List[StateT] = list(leaf_states)
        for sched in VALID_SCHEDULES:
            if str(sched) == "balanced":
                cur_s = list(leaf_states)
                cur_p = list(spans)
                while len(cur_s) > 1:
                    nxt_s: List[StateT] = []
                    nxt_p: List[Tuple[int, int]] = []
                    i = 0
                    while i < len(cur_s):
                        if i + 1 >= len(cur_s):
                            nxt_s.append(cur_s[i])
                            nxt_p.append(cur_p[i])
                            i += 1
                            continue
                        merged = merge(cur_s[i], cur_s[i + 1])
                        parent = (int(cur_p[i][0]), int(cur_p[i + 1][1]))
                        nxt_s.append(merged)
                        balanced_states.append(merged)
                        nxt_p.append(parent)
                        i += 2
                    cur_s, cur_p = nxt_s, nxt_p
                balanced_root_state = cur_s[0]
                roots[str(sched)] = float(value(cur_s[0]))
            elif str(sched) == "left_to_right":
                acc = leaf_states[0]
                for st in leaf_states[1:]:
                    acc = merge(acc, st)
                roots[str(sched)] = float(value(acc))
            elif str(sched) == "right_to_left":
                acc = leaf_states[-1]
                for st in reversed(leaf_states[:-1]):
                    acc = merge(st, acc)
                roots[str(sched)] = float(value(acc))
            else:
                raise ValueError(f"unsupported schedule: {sched!r}")

        truth_root = float(_oracle_count(doc, start=0, end=n_tok))
        pred = roots["balanced"]
        root_abs.append(abs(pred - truth_root))
        spreads.append(max(roots.values()) - min(roots.values()))
        if balanced_root_state is None:
            raise ValueError("balanced schedule must produce a root state")

        for state in balanced_states:
            base = float(value(state))
            c2_r1_abs.append(
                abs(float(value(_apply_resummary(state, rounds=1, resummary=resummary))) - base)
            )
            c2_r2_abs.append(
                abs(float(value(_apply_resummary(state, rounds=2, resummary=resummary))) - base)
            )
            c2_r4_abs.append(
                abs(float(value(_apply_resummary(state, rounds=4, resummary=resummary))) - base)
            )
        root_base = float(value(balanced_root_state))
        root_drift_r1.append(
            abs(
                float(value(_apply_resummary(balanced_root_state, rounds=1, resummary=resummary)))
                - root_base
            )
        )
        root_drift_r2.append(
            abs(
                float(value(_apply_resummary(balanced_root_state, rounds=2, resummary=resummary)))
                - root_base
            )
        )
        root_drift_r4.append(
            abs(
                float(value(_apply_resummary(balanced_root_state, rounds=4, resummary=resummary)))
                - root_base
            )
        )

        # C3 discrepancies (balanced schedule only).
        cur_s = list(leaf_states)
        cur_p = list(spans)
        while len(cur_s) > 1:
            nxt_s = []
            nxt_p = []
            i = 0
            while i < len(cur_s):
                if i + 1 >= len(cur_s):
                    nxt_s.append(cur_s[i])
                    nxt_p.append(cur_p[i])
                    i += 1
                    continue
                merged = merge(cur_s[i], cur_s[i + 1])
                parent = (int(cur_p[i][0]), int(cur_p[i + 1][1]))
                truth_parent = float(_oracle_count(doc, start=parent[0], end=parent[1]))
                merge_abs.append(abs(float(value(merged)) - truth_parent))
                nxt_s.append(merged)
                nxt_p.append(parent)
                i += 2
            cur_s, cur_p = nxt_s, nxt_p

    leaf_abs_arr = np.asarray(leaf_abs, dtype=np.float64)
    merge_abs_arr = np.asarray(merge_abs, dtype=np.float64)
    root_abs_arr = np.asarray(root_abs, dtype=np.float64)
    spreads_arr = np.asarray(spreads, dtype=np.float64)
    c2_r1_arr = np.asarray(c2_r1_abs, dtype=np.float64)
    c2_r2_arr = np.asarray(c2_r2_abs, dtype=np.float64)
    c2_r4_arr = np.asarray(c2_r4_abs, dtype=np.float64)
    root_drift_r1_arr = np.asarray(root_drift_r1, dtype=np.float64)
    root_drift_r2_arr = np.asarray(root_drift_r2, dtype=np.float64)
    root_drift_r4_arr = np.asarray(root_drift_r4, dtype=np.float64)

    tau = float(tau)
    return SketchMetrics(
        root_mae=float(np.mean(root_abs_arr)),
        root_median_abs_error=float(np.median(root_abs_arr)),
        root_p95_abs_error=float(np.percentile(root_abs_arr, 95.0)),
        schedule_spread_mean=float(np.mean(spreads_arr)),
        schedule_spread_p95=float(np.percentile(spreads_arr, 95.0)),
        leaf_mae=float(np.mean(leaf_abs_arr)) if leaf_abs_arr.size else 0.0,
        leaf_violation_rate=(
            float(np.mean((leaf_abs_arr > tau).astype(np.float64))) if leaf_abs_arr.size else 0.0
        ),
        c2_idempotence_mae=float(np.mean(c2_r1_arr)) if c2_r1_arr.size else 0.0,
        c2_r1_mae=float(np.mean(c2_r1_arr)) if c2_r1_arr.size else 0.0,
        c2_r2_mae=float(np.mean(c2_r2_arr)) if c2_r2_arr.size else 0.0,
        c2_r4_mae=float(np.mean(c2_r4_arr)) if c2_r4_arr.size else 0.0,
        resummary_root_drift_r1=(
            float(np.mean(root_drift_r1_arr)) if root_drift_r1_arr.size else 0.0
        ),
        resummary_root_drift_r2=(
            float(np.mean(root_drift_r2_arr)) if root_drift_r2_arr.size else 0.0
        ),
        resummary_root_drift_r4=(
            float(np.mean(root_drift_r4_arr)) if root_drift_r4_arr.size else 0.0
        ),
        merge_mae=float(np.mean(merge_abs_arr)) if merge_abs_arr.size else 0.0,
        merge_violation_rate=(
            float(np.mean((merge_abs_arr > tau).astype(np.float64))) if merge_abs_arr.size else 0.0
        ),
        n_docs=int(len(docs)),
    )


def _eval_exact_family(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    leaf_tokens: int,
    tau: float,
) -> SketchMetrics:
    return _eval_structured_family(
        docs,
        leaf_tokens=int(leaf_tokens),
        tau=float(tau),
        from_span=_exact_from_span,
        merge=_exact_merge,
        value=lambda z: float(z.count),
        resummary=_identity_resummary,
    )


def _eval_count_only_family(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    leaf_tokens: int,
    tau: float,
) -> SketchMetrics:
    return _eval_structured_family(
        docs,
        leaf_tokens=int(leaf_tokens),
        tau=float(tau),
        from_span=_count_only_from_span,
        merge=_count_only_merge,
        value=lambda z: float(z.count),
        resummary=_identity_resummary,
    )


def _eval_leaf_bucket_family(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    leaf_tokens: int,
    tau: float,
) -> SketchMetrics:
    return _eval_structured_family(
        docs,
        leaf_tokens=int(leaf_tokens),
        tau=float(tau),
        from_span=_leaf_bucket_from_span,
        merge=_leaf_bucket_merge,
        value=lambda z: float(_leaf_bucket_value(z)),
        resummary=_identity_resummary,
    )


def _eval_flip_family(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    leaf_tokens: int,
    tau: float,
    rounds: int,
) -> SketchMetrics:
    base = _eval_structured_family(
        docs,
        leaf_tokens=int(leaf_tokens),
        tau=float(tau),
        from_span=_flip_from_span,
        merge=_flip_merge,
        value=lambda z: float(_flip_value(z)),
        resummary=_flip_resummary,
    )
    if int(rounds) <= 1:
        return base
    drift_by_round = {
        2: float(base.resummary_root_drift_r1),
        3: float(base.resummary_root_drift_r2),
        5: float(base.resummary_root_drift_r4),
    }
    root_round_drift = float(drift_by_round.get(int(rounds), base.resummary_root_drift_r1))
    return SketchMetrics(
        root_mae=float(root_round_drift),
        root_median_abs_error=float(root_round_drift),
        root_p95_abs_error=float(root_round_drift),
        schedule_spread_mean=float(base.schedule_spread_mean),
        schedule_spread_p95=float(base.schedule_spread_p95),
        leaf_mae=float(base.leaf_mae),
        leaf_violation_rate=float(base.leaf_violation_rate),
        c2_idempotence_mae=float(base.c2_idempotence_mae),
        c2_r1_mae=float(base.c2_r1_mae),
        c2_r2_mae=float(base.c2_r2_mae),
        c2_r4_mae=float(base.c2_r4_mae),
        resummary_root_drift_r1=float(base.resummary_root_drift_r1),
        resummary_root_drift_r2=float(base.resummary_root_drift_r2),
        resummary_root_drift_r4=float(base.resummary_root_drift_r4),
        merge_mae=float(base.merge_mae),
        merge_violation_rate=float(base.merge_violation_rate),
        n_docs=int(base.n_docs),
    )


def _span_features(
    doc: ChangepointMarkovDoc,
    span: Tuple[int, int],
    *,
    n_regimes: int,
    mode: str,
) -> torch.Tensor:
    start, end = span
    regs = np.asarray(doc.token_regimes[int(start) : int(end)], dtype=np.int64)
    if regs.size == 0:
        raise ValueError("empty span")
    n = int(n_regimes)

    if mode not in {"full", "no_endpoints"}:
        raise ValueError(f"unsupported feature_mode: {mode!r} (expected 'full' or 'no_endpoints')")

    parts: List[np.ndarray] = []
    if mode == "full":
        first = np.zeros((n,), dtype=np.float32)
        last = np.zeros((n,), dtype=np.float32)
        first[int(regs[0])] = 1.0
        last[int(regs[-1])] = 1.0
        parts.extend([first, last])

    trans = np.zeros((n, n), dtype=np.float32)
    if regs.size >= 2:
        for a, b in zip(regs[:-1], regs[1:]):
            trans[int(a), int(b)] += 1.0
        trans /= float(max(1, regs.size - 1))
    parts.append(trans.reshape(-1))

    # Length feature (helps disambiguate sparse leaves).
    parts.append(np.asarray([float(regs.size)], dtype=np.float32))

    feat = np.concatenate(parts, axis=0)
    return torch.tensor(feat, dtype=torch.float32)


@dataclass(frozen=True)
class _CountDoc:
    n_tokens: int
    leaf_features: Tuple[torch.Tensor, ...]  # CPU float32
    leaf_counts: Tuple[float, ...]
    merge_counts_balanced: Tuple[
        float, ...
    ]  # oracle counts for each realized merge (balanced order)
    merge_sizes_balanced: Tuple[int, ...]  # number of leaves under each realized merge
    root_count: float


def _to_device(xs: Sequence[torch.Tensor], *, device: torch.device) -> List[torch.Tensor]:
    return [x.to(device=device) for x in xs]


def _prepare_count_docs(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    leaf_tokens: int,
    n_regimes: int,
    feature_mode: str,
) -> Tuple[_CountDoc, ...]:
    out: List[_CountDoc] = []
    for doc in docs:
        n_tok = int(len(doc.token_regimes))
        spans = _leaf_spans(n_tok, leaf_tokens=int(leaf_tokens))
        leaf_feats = tuple(
            _span_features(doc, sp, n_regimes=int(n_regimes), mode=str(feature_mode))
            for sp in spans
        )
        leaf_counts = tuple(float(_oracle_count(doc, start=sp[0], end=sp[1])) for sp in spans)
        # Balanced merge labels (oracle on the realized internal nodes).
        cur_spans = list(spans)
        cur_sizes = [1 for _ in spans]
        merge_counts: List[float] = []
        merge_sizes: List[int] = []
        while len(cur_spans) > 1:
            nxt_spans: List[Tuple[int, int]] = []
            nxt_sizes: List[int] = []
            i = 0
            while i < len(cur_spans):
                if i + 1 >= len(cur_spans):
                    nxt_spans.append(cur_spans[i])
                    nxt_sizes.append(int(cur_sizes[i]))
                    i += 1
                    continue
                parent = (int(cur_spans[i][0]), int(cur_spans[i + 1][1]))
                parent_size = int(cur_sizes[i]) + int(cur_sizes[i + 1])
                merge_counts.append(float(_oracle_count(doc, start=parent[0], end=parent[1])))
                merge_sizes.append(int(parent_size))
                nxt_spans.append(parent)
                nxt_sizes.append(int(parent_size))
                i += 2
            cur_spans = nxt_spans
            cur_sizes = nxt_sizes
        root_count = float(_oracle_count(doc, start=0, end=n_tok))
        out.append(
            _CountDoc(
                n_tokens=int(n_tok),
                leaf_features=leaf_feats,
                leaf_counts=leaf_counts,
                merge_counts_balanced=tuple(merge_counts),
                merge_sizes_balanced=tuple(merge_sizes),
                root_count=float(root_count),
            )
        )
    return tuple(out)


def _sample_internal_audit_indices(
    n_internal: int,
    *,
    k: int,
    strategy: C3AuditStrategyName,
    merge_sizes: Sequence[int],
    include_root: bool,
    rng: random.Random,
) -> Optional[set[int]]:
    """
    Sample realized internal nodes for C3 labels.

    Returns:
      - `None`: use all internal nodes
      - empty set: use none
      - non-empty set: selected indices
    """

    n = int(max(0, n_internal))
    q = int(max(0, k))
    if n <= 0 or q <= 0:
        return set()
    if q >= n:
        return None

    strat = str(strategy)
    if strat not in VALID_C3_AUDIT_STRATEGIES:
        raise ValueError(
            f"unsupported c3_audit_strategy: {strategy!r}; expected one of {VALID_C3_AUDIT_STRATEGIES}"
        )

    selected: set[int] = set()
    if include_root and n > 0:
        # In `_prepare_count_docs`, the root merge is appended last.
        selected.add(int(n - 1))
    if len(selected) >= q:
        return set(list(selected)[:q])

    available = [i for i in range(n) if i not in selected]
    need = int(q - len(selected))
    if need <= 0:
        return selected

    if strat == "uniform":
        selected.update(rng.sample(available, k=need))
        return selected

    if strat == "top_span":
        ranked = sorted(
            available,
            key=lambda i: (int(merge_sizes[i]) if i < len(merge_sizes) else 0, int(i)),
            reverse=True,
        )
        selected.update(ranked[:need])
        return selected

    if strat == "hybrid_top_span":
        ranked = sorted(
            available,
            key=lambda i: (int(merge_sizes[i]) if i < len(merge_sizes) else 0, int(i)),
            reverse=True,
        )
        top_need = min(len(ranked), max(1, need // 2))
        selected.update(ranked[:top_need])
        remaining_need = int(need - top_need)
        if remaining_need > 0:
            rem = [i for i in available if i not in selected]
            if remaining_need >= len(rem):
                selected.update(rem)
            else:
                selected.update(rng.sample(rem, k=remaining_need))
        return selected

    # Weighted without replacement (Efraimidis-Spirakis): larger spans are more likely.
    keys: List[Tuple[float, int]] = []
    for i in available:
        w = float(merge_sizes[i]) if i < len(merge_sizes) else 1.0
        w = max(1e-8, w)
        u = max(float(rng.random()), 1e-12)
        keys.append((u ** (1.0 / w), int(i)))
    keys.sort(reverse=True)
    selected.update(i for _k, i in keys[:need])
    return selected


class LearnedCountSketch(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        state_dim: int,
        hidden_dim: int,
        target_scale: float,
        n_regimes: int,
        use_endpoints: bool,
    ) -> None:
        super().__init__()
        self.target_scale = float(target_scale)
        self.n_regimes = int(n_regimes)
        self.use_endpoints = bool(use_endpoints)
        self.state_dim = int(state_dim)
        if self.n_regimes <= 0:
            raise ValueError("n_regimes must be positive")
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")

        endpoint_dim = 2 * int(self.n_regimes) if self.use_endpoints else 0
        encoder_in = int(feature_dim) - int(endpoint_dim)
        if encoder_in <= 0:
            raise ValueError("feature_dim too small for endpoint stripping")
        self.encoder = nn.Sequential(
            nn.Linear(int(encoder_in), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(state_dim)),
        )
        self.merger = nn.Sequential(
            nn.Linear(2 * int(state_dim) + 2 * int(self.n_regimes), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(state_dim)),
        )
        self.readout = nn.Linear(int(state_dim), 1)
        self.summary_encoder = nn.Sequential(
            nn.Linear(1, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(state_dim)),
        )

    @property
    def summary_dim(self) -> int:
        return int(self.state_dim) + 2 * int(self.n_regimes)

    def _split_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split a state into:
          - latent count vector h (R^{state_dim})
          - first regime one-hot (R^{n_regimes})
          - last regime one-hot (R^{n_regimes})

        Endpoints are included to match the exact sketch semantics: the oracle-preserving state
        for a span needs its boundary identities to compute cross-span corrections.
        """

        d = int(self.state_dim)
        n = int(self.n_regimes)
        if state.shape[-1] != d + 2 * n:
            raise ValueError("unexpected state dimension")
        h = state[..., :d]
        first = state[..., d : d + n]
        last = state[..., d + n : d + 2 * n]
        return h, first, last

    def encode_leaf(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode leaf features into a sketch state.

        If `use_endpoints=True`, we *preserve* (first,last) regime one-hots as explicit state
        components instead of asking the network to relearn/carry them implicitly.
        """

        n = int(self.n_regimes)
        if self.use_endpoints:
            if features.shape[-1] < 2 * n:
                raise ValueError("leaf features missing endpoint slots")
            first = features[..., :n]
            last = features[..., n : 2 * n]
            core = features[..., 2 * n :]
        else:
            first = torch.zeros(
                (*features.shape[:-1], n), device=features.device, dtype=features.dtype
            )
            last = torch.zeros(
                (*features.shape[:-1], n), device=features.device, dtype=features.dtype
            )
            core = features

        h = self.encoder(core)
        return torch.cat([h, first, last], dim=-1)

    def predict_norm_from_state(self, state: torch.Tensor) -> torch.Tensor:
        h, _first, _last = self._split_state(state)
        logit = self.readout(h)
        return torch.sigmoid(logit).squeeze(-1)

    def predict_count_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.predict_norm_from_state(state) * float(self.target_scale)

    def decode_summary(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def encode_summary(self, summary: torch.Tensor) -> torch.Tensor:
        x = summary
        if x.ndim == 0:
            x = x.unsqueeze(0)
        if x.shape[-1] == int(self.summary_dim):
            return x
        if x.shape[-1] != 1:
            x = x.unsqueeze(-1)
        h = self.summary_encoder(x)
        zeros = torch.zeros(
            (*h.shape[:-1], 2 * int(self.n_regimes)),
            device=h.device,
            dtype=h.dtype,
        )
        return torch.cat([h, zeros], dim=-1)

    def _merge_states(
        self,
        states: Sequence[torch.Tensor],
        *,
        schedule: ScheduleName,
        collect_merge_states: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if len(states) == 0:
            raise ValueError("need at least one state")
        if len(states) == 1:
            return states[0], []

        merged_states: List[torch.Tensor] = []
        if str(schedule) == "balanced":
            cur = list(states)
            while len(cur) > 1:
                nxt: List[torch.Tensor] = []
                i = 0
                while i < len(cur):
                    if i + 1 >= len(cur):
                        nxt.append(cur[i])
                        i += 1
                        continue
                    left_h, left_first, left_last = self._split_state(cur[i])
                    right_h, right_first, right_last = self._split_state(cur[i + 1])
                    merged_h = self.merger(
                        torch.cat([left_h, right_h, left_last, right_first], dim=-1)
                    )
                    merged = torch.cat([merged_h, left_first, right_last], dim=-1)
                    if collect_merge_states:
                        merged_states.append(merged)
                    nxt.append(merged)
                    i += 2
                cur = nxt
            return cur[0], merged_states

        if str(schedule) in ("left_to_right", "right_to_left"):
            if str(schedule) == "left_to_right":
                acc = states[0]
                for st in states[1:]:
                    left_h, left_first, left_last = self._split_state(acc)
                    right_h, right_first, right_last = self._split_state(st)
                    merged_h = self.merger(
                        torch.cat([left_h, right_h, left_last, right_first], dim=-1)
                    )
                    acc = torch.cat([merged_h, left_first, right_last], dim=-1)
                    if collect_merge_states:
                        merged_states.append(acc)
                return acc, merged_states

            # Right-associated chain that preserves leaf order:
            # s0 ⊕ (s1 ⊕ (s2 ⊕ ...)).
            acc = states[-1]
            for st in reversed(states[:-1]):
                left_h, left_first, left_last = self._split_state(st)
                right_h, right_first, right_last = self._split_state(acc)
                merged_h = self.merger(torch.cat([left_h, right_h, left_last, right_first], dim=-1))
                acc = torch.cat([merged_h, left_first, right_last], dim=-1)
                if collect_merge_states:
                    merged_states.append(acc)
            return acc, merged_states

        raise ValueError(f"unsupported schedule: {schedule!r}")

    def forward_doc(
        self,
        leaf_features: Sequence[torch.Tensor],
        leaf_counts: Sequence[float],
        merge_counts_balanced: Sequence[float],
        *,
        schedule: ScheduleName,
        collect_leaf: bool,
        collect_c3: bool,
        collect_c2: bool,
        leaf_audit_indices: Optional[set[int]] = None,
        c3_audit_indices: Optional[set[int]] = None,
    ) -> Dict[str, torch.Tensor | float]:
        if len(leaf_features) == 0:
            raise ValueError("leaf_features must be non-empty")
        if len(leaf_features) != len(leaf_counts):
            raise ValueError("leaf_features and leaf_counts must align")

        states = [self.encode_leaf(x) for x in leaf_features]
        root_state, merge_states = self._merge_states(
            states,
            schedule=schedule,
            collect_merge_states=(collect_c3 or collect_c2) and str(schedule) == "balanced",
        )
        pred_norm = self.predict_norm_from_state(root_state)
        out: Dict[str, torch.Tensor | float] = {
            "pred_norm": pred_norm,
            "pred_count": self.predict_count_from_state(root_state),
        }

        if collect_leaf:
            leaf_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            leaf_count = 0
            for idx, (st, truth) in enumerate(zip(states, leaf_counts)):
                if leaf_audit_indices is not None and idx not in leaf_audit_indices:
                    continue
                pred_leaf = self.predict_norm_from_state(st)
                true_leaf = torch.tensor(
                    float(truth) / float(self.target_scale),
                    device=pred_norm.device,
                    dtype=pred_leaf.dtype,
                )
                leaf_loss = leaf_loss + F.mse_loss(pred_leaf, true_leaf, reduction="mean")
                leaf_count += 1
            out["leaf_loss"] = leaf_loss / float(max(1, leaf_count))
            out["leaf_count"] = float(leaf_count)
        else:
            out["leaf_loss"] = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            out["leaf_count"] = 0.0

        if collect_c3:
            if str(schedule) != "balanced":
                raise ValueError("collect_c3 is only supported for balanced schedule")
            c3_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            c3_count = 0
            for idx, st in enumerate(merge_states):
                if c3_audit_indices is not None and idx not in c3_audit_indices:
                    continue
                if idx >= len(merge_counts_balanced):
                    continue
                pred = self.predict_norm_from_state(st)
                truth = torch.tensor(
                    float(merge_counts_balanced[idx]) / float(self.target_scale),
                    device=pred_norm.device,
                    dtype=pred.dtype,
                )
                c3_loss = c3_loss + F.mse_loss(pred, truth, reduction="mean")
                c3_count += 1
            out["c3_loss"] = c3_loss / float(max(1, c3_count))
            out["c3_count"] = float(c3_count)
        else:
            out["c3_loss"] = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            out["c3_count"] = 0.0
        if collect_c2:
            c2_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            c2_count = 0
            candidate_states = list(states)
            candidate_states.extend(list(merge_states))
            if len(candidate_states) == 0:
                candidate_states = [root_state]
            for st in candidate_states:
                base_summary = self.decode_summary(st)
                base_state = self.encode_summary(base_summary)
                replay_state = self.encode_summary(self.decode_summary(base_state))
                c2_loss = c2_loss + F.mse_loss(replay_state, base_state, reduction="mean")
                c2_count += 1
            out["c2_loss"] = c2_loss / float(max(1, c2_count))
            out["c2_count"] = float(c2_count)
        else:
            out["c2_loss"] = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            out["c2_count"] = 0.0
        return out


class AdditiveCountSketch(nn.Module):
    """
    Structured sketch family for the Markov changepoint-count target.

    State layout:
      - normalized count scalar (R^1)
      - first regime one-hot (R^{n_regimes})
      - last regime one-hot (R^{n_regimes})

    Merge law (associative by construction):
      c(parent) = c(left) + c(right) + 1[last(left) != first(right)] / target_scale

    This family is intentionally "OPS-shaped": we *separate* endpoint transport (exact) from the
    (learned) scalar count, so that under full labels it can approach the exact ceiling.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int,
        target_scale: float,
        n_regimes: int,
        use_endpoints: bool,
    ) -> None:
        super().__init__()
        self.target_scale = float(target_scale)
        self.n_regimes = int(n_regimes)
        self.use_endpoints = bool(use_endpoints)
        if self.n_regimes <= 0:
            raise ValueError("n_regimes must be positive")
        if self.target_scale <= 0:
            raise ValueError("target_scale must be positive")

        endpoint_dim = 2 * int(self.n_regimes) if self.use_endpoints else 0
        encoder_in = int(feature_dim) - int(endpoint_dim)
        if encoder_in <= 0:
            raise ValueError("feature_dim too small for endpoint stripping")

        # Linear leaf encoder -> scalar normalized count.
        # This matches the default DGP where the changepoint count is a linear functional of transition counts.
        self.encoder = nn.Linear(int(encoder_in), 1, bias=True)

    @property
    def summary_dim(self) -> int:
        return 1 + 2 * int(self.n_regimes)

    def _split_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = int(self.n_regimes)
        if state.shape[-1] != 1 + 2 * n:
            raise ValueError("unexpected state dimension for AdditiveCountSketch")
        count = state[..., 0]
        first = state[..., 1 : 1 + n]
        last = state[..., 1 + n : 1 + 2 * n]
        return count, first, last

    def encode_leaf(self, features: torch.Tensor) -> torch.Tensor:
        n = int(self.n_regimes)
        if self.use_endpoints:
            if features.shape[-1] < 2 * n:
                raise ValueError("leaf features missing endpoint slots")
            first = features[..., :n]
            last = features[..., n : 2 * n]
            core = features[..., 2 * n :]
        else:
            first = torch.zeros(
                (*features.shape[:-1], n), device=features.device, dtype=features.dtype
            )
            last = torch.zeros(
                (*features.shape[:-1], n), device=features.device, dtype=features.dtype
            )
            core = features

        count_norm = self.encoder(core).squeeze(-1)
        return torch.cat([count_norm.unsqueeze(-1), first, last], dim=-1)

    def predict_norm_from_state(self, state: torch.Tensor) -> torch.Tensor:
        count, _first, _last = self._split_state(state)
        return count

    def predict_count_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.predict_norm_from_state(state) * float(self.target_scale)

    def decode_summary(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def encode_summary(self, summary: torch.Tensor) -> torch.Tensor:
        x = summary
        if x.ndim == 0:
            x = x.unsqueeze(0)
        if x.shape[-1] == int(self.summary_dim):
            return x
        if x.shape[-1] != 1:
            x = x.unsqueeze(-1)
        zeros = torch.zeros(
            (*x.shape[:-1], 2 * int(self.n_regimes)),
            device=x.device,
            dtype=x.dtype,
        )
        return torch.cat([x, zeros], dim=-1)

    def _merge_states(
        self,
        states: Sequence[torch.Tensor],
        *,
        schedule: ScheduleName,
        collect_merge_states: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if len(states) == 0:
            raise ValueError("need at least one state")
        if len(states) == 1:
            return states[0], []

        def _merge(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            left_c, left_first, left_last = self._split_state(left)
            right_c, right_first, right_last = self._split_state(right)
            if self.use_endpoints:
                same = torch.sum(left_last * right_first, dim=-1)  # one-hot dot
                join = 1.0 - same
                join_term = join / float(self.target_scale)
            else:
                join_term = torch.zeros_like(left_c)
            merged_c = left_c + right_c + join_term
            return torch.cat([merged_c.unsqueeze(-1), left_first, right_last], dim=-1)

        merged_states: List[torch.Tensor] = []
        if str(schedule) == "balanced":
            cur = list(states)
            while len(cur) > 1:
                nxt: List[torch.Tensor] = []
                i = 0
                while i < len(cur):
                    if i + 1 >= len(cur):
                        nxt.append(cur[i])
                        i += 1
                        continue
                    merged = _merge(cur[i], cur[i + 1])
                    if collect_merge_states:
                        merged_states.append(merged)
                    nxt.append(merged)
                    i += 2
                cur = nxt
            return cur[0], merged_states

        if str(schedule) in ("left_to_right", "right_to_left"):
            if str(schedule) == "left_to_right":
                acc = states[0]
                for st in states[1:]:
                    acc = _merge(acc, st)
                    if collect_merge_states:
                        merged_states.append(acc)
                return acc, merged_states

            acc = states[-1]
            for st in reversed(states[:-1]):
                acc = _merge(st, acc)
                if collect_merge_states:
                    merged_states.append(acc)
            return acc, merged_states

        raise ValueError(f"unsupported schedule: {schedule!r}")

    def forward_doc(
        self,
        leaf_features: Sequence[torch.Tensor],
        leaf_counts: Sequence[float],
        merge_counts_balanced: Sequence[float],
        *,
        schedule: ScheduleName,
        collect_leaf: bool,
        collect_c3: bool,
        collect_c2: bool,
        leaf_audit_indices: Optional[set[int]] = None,
        c3_audit_indices: Optional[set[int]] = None,
    ) -> Dict[str, torch.Tensor | float]:
        if len(leaf_features) == 0:
            raise ValueError("leaf_features must be non-empty")
        if len(leaf_features) != len(leaf_counts):
            raise ValueError("leaf_features and leaf_counts must align")

        states = [self.encode_leaf(x) for x in leaf_features]
        root_state, merge_states = self._merge_states(
            states,
            schedule=schedule,
            collect_merge_states=(collect_c3 or collect_c2) and str(schedule) == "balanced",
        )
        pred_norm = self.predict_norm_from_state(root_state)
        out: Dict[str, torch.Tensor | float] = {
            "pred_norm": pred_norm,
            "pred_count": self.predict_count_from_state(root_state),
        }

        if collect_leaf:
            leaf_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            leaf_count = 0
            for idx, (st, truth) in enumerate(zip(states, leaf_counts)):
                if leaf_audit_indices is not None and idx not in leaf_audit_indices:
                    continue
                pred_leaf = self.predict_norm_from_state(st)
                true_leaf = torch.tensor(
                    float(truth) / float(self.target_scale),
                    device=pred_norm.device,
                    dtype=pred_leaf.dtype,
                )
                leaf_loss = leaf_loss + F.mse_loss(pred_leaf, true_leaf, reduction="mean")
                leaf_count += 1
            out["leaf_loss"] = leaf_loss / float(max(1, leaf_count))
            out["leaf_count"] = float(leaf_count)
        else:
            out["leaf_loss"] = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            out["leaf_count"] = 0.0

        if collect_c3:
            if str(schedule) != "balanced":
                raise ValueError("collect_c3 is only supported for balanced schedule")
            c3_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            c3_count = 0
            for idx, st in enumerate(merge_states):
                if c3_audit_indices is not None and idx not in c3_audit_indices:
                    continue
                if idx >= len(merge_counts_balanced):
                    continue
                pred = self.predict_norm_from_state(st)
                truth = torch.tensor(
                    float(merge_counts_balanced[idx]) / float(self.target_scale),
                    device=pred_norm.device,
                    dtype=pred.dtype,
                )
                c3_loss = c3_loss + F.mse_loss(pred, truth, reduction="mean")
                c3_count += 1
            out["c3_loss"] = c3_loss / float(max(1, c3_count))
            out["c3_count"] = float(c3_count)
        else:
            out["c3_loss"] = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            out["c3_count"] = 0.0
        if collect_c2:
            c2_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            c2_count = 0
            candidate_states = list(states)
            candidate_states.extend(list(merge_states))
            if len(candidate_states) == 0:
                candidate_states = [root_state]
            for st in candidate_states:
                base_summary = self.decode_summary(st)
                base_state = self.encode_summary(base_summary)
                replay_state = self.encode_summary(self.decode_summary(base_state))
                c2_loss = c2_loss + F.mse_loss(replay_state, base_state, reduction="mean")
                c2_count += 1
            out["c2_loss"] = c2_loss / float(max(1, c2_count))
            out["c2_count"] = float(c2_count)
        else:
            out["c2_loss"] = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            out["c2_count"] = 0.0
        return out


def _training_geometry(
    docs: Sequence[_CountDoc],
    *,
    policy: AuditPolicyName,
    fixed_nodes: int,
    fraction: float,
    scale: float,
    leaf_query_rate: float,
    include_root_query: bool,
) -> TrainingGeometry:
    if len(docs) == 0:
        return TrainingGeometry(
            mean_tokens=0.0,
            mean_leaves=0.0,
            mean_internal_nodes=0.0,
            mean_leaf_labels=0.0,
            mean_internal_labels=0.0,
            mean_queries_per_doc=0.0,
            root_queries_total=0,
            leaf_labels_total=0,
            internal_labels_total=0,
            total_queries_estimate=0,
        )

    toks: List[float] = []
    leaves: List[float] = []
    internals: List[float] = []
    leaf_labels: List[float] = []
    internal_labels: List[float] = []
    leaf_labels_total = 0
    internal_labels_total = 0

    for doc in docs:
        n_tok = int(doc.n_tokens)
        n_leaves = int(len(doc.leaf_features))
        n_internal = int(max(0, n_leaves - 1))
        q_leaf = leaf_sample_count(n_leaves, rate=float(leaf_query_rate))
        q_internal = audit_sample_count(
            n_internal,
            policy=str(policy),
            fixed_nodes=int(fixed_nodes),
            fraction=float(fraction),
            scale=float(scale),
        )
        toks.append(float(n_tok))
        leaves.append(float(n_leaves))
        internals.append(float(n_internal))
        leaf_labels.append(float(q_leaf))
        internal_labels.append(float(q_internal))
        leaf_labels_total += int(q_leaf)
        internal_labels_total += int(q_internal)

    n_docs = int(len(docs))
    root_queries_total = int(n_docs if include_root_query else 0)
    total = int(root_queries_total + leaf_labels_total + internal_labels_total)
    mean_leaf = float(np.mean(np.asarray(leaves, dtype=np.float64)))
    mean_leaf_labels = float(np.mean(np.asarray(leaf_labels, dtype=np.float64)))
    mean_internal = float(np.mean(np.asarray(internals, dtype=np.float64)))
    mean_internal_labels = float(np.mean(np.asarray(internal_labels, dtype=np.float64)))
    return TrainingGeometry(
        mean_tokens=float(np.mean(np.asarray(toks, dtype=np.float64))),
        mean_leaves=float(mean_leaf),
        mean_internal_nodes=float(mean_internal),
        mean_leaf_labels=float(mean_leaf_labels),
        mean_internal_labels=float(mean_internal_labels),
        mean_queries_per_doc=float(
            mean_leaf_labels + mean_internal_labels + (1.0 if include_root_query else 0.0)
        ),
        root_queries_total=int(root_queries_total),
        leaf_labels_total=int(leaf_labels_total),
        internal_labels_total=int(internal_labels_total),
        total_queries_estimate=int(total),
    )


def _resummary_summary_sequence(
    model: LearnedCountSketch | AdditiveCountSketch,
    state: torch.Tensor,
    *,
    depths: Sequence[int],
) -> Dict[int, torch.Tensor]:
    wanted = sorted({int(d) for d in depths if int(d) >= 1})
    if not wanted:
        return {}
    cur = model.decode_summary(state)
    out: Dict[int, torch.Tensor] = {}
    max_depth = int(max(wanted))
    for step in range(1, max_depth + 1):
        cur = model.decode_summary(model.encode_summary(cur))
        if step in wanted:
            out[int(step)] = cur
    return out


def _predict_count_from_summary(
    model: LearnedCountSketch | AdditiveCountSketch,
    summary: torch.Tensor,
) -> torch.Tensor:
    return model.predict_count_from_state(model.encode_summary(summary))


def _train_learned_model(
    model: LearnedCountSketch,
    train_docs: Sequence[_CountDoc],
    val_docs: Sequence[_CountDoc],
    *,
    device: torch.device,
    audit_policy: AuditPolicyName,
    audit_fixed_nodes: int,
    audit_fraction: float,
    audit_scale: float,
    c3_audit_strategy: C3AuditStrategyName,
    c3_include_root: bool,
    leaf_query_rate: float,
    include_root_query: bool,
    n_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    c3_weight: float,
    c2_weight: float,
    leaf_weight: float,
    root_weight: float,
    schedule_consistency_weight: float,
    grad_clip_norm: float,
    seed: int,
) -> TrainFitDiagnostics:
    if len(train_docs) == 0:
        return TrainFitDiagnostics(
            train_loss_final=0.0,
            train_loss_curve=(0.0,),
            epochs_completed=0,
            selection_metric_curve=(0.0,) if len(val_docs) <= 0 else tuple(),
            selection_mode="best_val_optimization_objective"
            if len(val_docs) > 0
            else "final_epoch_no_validation",
            selection_split="val" if len(val_docs) > 0 else "config",
            selection_metric_name=(
                "val_optimization_objective_full_labels"
                if len(val_docs) > 0
                else "train_loss_final"
            ),
            selection_metric_value=0.0,
            best_epoch=0,
        )
    rng = random.Random(int(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    target_scale = float(model.target_scale)

    idxs = list(range(len(train_docs)))
    epoch_losses: List[float] = []
    selection_curve: List[float] = []
    best_selection = TrainingSelectionMetadata(
        mode="best_val_optimization_objective"
        if len(val_docs) > 0
        else "final_epoch_no_validation",
        split="val" if len(val_docs) > 0 else "config",
        metric_name=(
            "val_optimization_objective_full_labels"
            if len(val_docs) > 0
            else "train_loss_final"
        ),
        metric_value=float("nan"),
        best_epoch=0,
    )
    best_state = None
    for _ in range(int(max(1, n_epochs))):
        epoch_idx = int(len(epoch_losses))
        rng.shuffle(idxs)
        model.train()
        batch_losses: List[float] = []
        for b0 in range(0, len(idxs), int(max(1, batch_size))):
            batch_idx = idxs[b0 : b0 + int(max(1, batch_size))]
            opt.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=device, dtype=torch.float32)
            for i in batch_idx:
                doc = train_docs[i]
                leaf_feats = _to_device(doc.leaf_features, device=device)
                n_leaf = int(len(leaf_feats))
                n_internal = int(max(0, len(leaf_feats) - 1))
                n_leaf_audit = leaf_sample_count(n_leaf, rate=float(leaf_query_rate))
                if n_leaf_audit <= 0:
                    leaf_audit_indices: Optional[set[int]] = set()
                    collect_leaf = False
                elif n_leaf_audit >= n_leaf:
                    leaf_audit_indices = None
                    collect_leaf = True
                else:
                    leaf_audit_indices = set(rng.sample(range(n_leaf), k=n_leaf_audit))
                    collect_leaf = True
                n_audit = audit_sample_count(
                    n_internal,
                    policy=str(audit_policy),
                    fixed_nodes=int(audit_fixed_nodes),
                    fraction=float(audit_fraction),
                    scale=float(audit_scale),
                )
                c3_audit_indices = _sample_internal_audit_indices(
                    n_internal,
                    k=n_audit,
                    strategy=str(c3_audit_strategy),
                    merge_sizes=doc.merge_sizes_balanced,
                    include_root=bool(c3_include_root),
                    rng=rng,
                )
                out = model.forward_doc(
                    leaf_feats,
                    doc.leaf_counts,
                    doc.merge_counts_balanced,
                    schedule="balanced",
                    collect_leaf=collect_leaf,
                    collect_c3=True,
                    collect_c2=bool(float(c2_weight) > 0.0),
                    leaf_audit_indices=leaf_audit_indices,
                    c3_audit_indices=c3_audit_indices,
                )
                pred_norm = out["pred_norm"]
                true_norm = torch.tensor(
                    float(doc.root_count) / target_scale,
                    device=device,
                    dtype=pred_norm.dtype,
                )
                if include_root_query and float(root_weight) > 0.0:
                    root_loss = float(root_weight) * F.mse_loss(
                        pred_norm, true_norm, reduction="mean"
                    )
                else:
                    root_loss = torch.zeros((), device=device, dtype=pred_norm.dtype)
                if float(schedule_consistency_weight) > 0.0 and n_leaf > 1:
                    states_sched = [model.encode_leaf(x) for x in leaf_feats]
                    sched_preds = []
                    for sched in VALID_SCHEDULES:
                        root_state_sched, _ = model._merge_states(
                            states_sched,
                            schedule=sched,
                            collect_merge_states=False,
                        )
                        sched_preds.append(model.predict_norm_from_state(root_state_sched))
                    pred_stack = torch.stack(sched_preds, dim=0)
                    consistency_loss = torch.mean((pred_stack - torch.mean(pred_stack)) ** 2)
                else:
                    consistency_loss = torch.zeros((), device=device, dtype=pred_norm.dtype)
                doc_loss = (
                    root_loss
                    + float(c3_weight) * out["c3_loss"]
                    + float(c2_weight) * out["c2_loss"]
                    + float(leaf_weight) * out["leaf_loss"]
                    + float(schedule_consistency_weight) * consistency_loss
                )
                batch_loss = batch_loss + doc_loss
            batch_loss = batch_loss / float(len(batch_idx))
            # If there is literally no supervised term in the objective for this configuration
            # (e.g. root query disabled AND leaf_query_rate=0 AND audit_fraction=0), the loss can
            # be a constant tensor without a grad_fn. In that case, skip the optimizer step and
            # let the downstream metrics reflect an untrained sketch.
            if bool(getattr(batch_loss, "requires_grad", False)):
                batch_loss.backward()
                if float(grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                opt.step()
            batch_losses.append(float(batch_loss.detach().cpu()))
        epoch_train_loss = float(np.mean(np.asarray(batch_losses, dtype=np.float64)))
        epoch_losses.append(float(epoch_train_loss))
        if len(val_docs) > 0:
            val_objective = _eval_objective_terms(
                model,
                val_docs,
                device=device,
                leaf_weight=float(leaf_weight),
                c2_weight=float(c2_weight),
                c3_weight=float(c3_weight),
                root_weight=float(root_weight),
                schedule_consistency_weight=float(schedule_consistency_weight),
                include_root_query=bool(include_root_query),
            )
            selection_value = float(val_objective.optimization_total_loss)
            selection_curve.append(float(selection_value))
            if improved_metric(selection_value, best_selection.metric_value):
                best_selection = TrainingSelectionMetadata(
                    mode="best_val_optimization_objective",
                    split="val",
                    metric_name="val_optimization_objective_full_labels",
                    metric_value=float(selection_value),
                    best_epoch=int(epoch_idx),
                )
                best_state = clone_module_state(model)
        else:
            selection_curve.append(float(epoch_train_loss))
    train_loss_final = float(epoch_losses[-1]) if epoch_losses else float("nan")
    if len(val_docs) > 0:
        restore_module_state(model, best_state)
    else:
        best_selection = TrainingSelectionMetadata(
            mode="final_epoch_no_validation",
            split="config",
            metric_name="train_loss_final",
            metric_value=float(train_loss_final),
            best_epoch=max(0, int(len(epoch_losses) - 1)),
        )
    return TrainFitDiagnostics(
        train_loss_final=float(train_loss_final),
        train_loss_curve=tuple(float(x) for x in epoch_losses),
        epochs_completed=int(len(epoch_losses)),
        selection_metric_curve=tuple(float(x) for x in selection_curve),
        selection_mode=str(best_selection.mode),
        selection_split=str(best_selection.split),
        selection_metric_name=str(best_selection.metric_name),
        selection_metric_value=float(best_selection.metric_value),
        best_epoch=int(best_selection.best_epoch),
    )


def _fit_additive_leaf_encoder_closed_form(
    model: AdditiveCountSketch,
    train_docs: Sequence[_CountDoc],
    *,
    device: torch.device,
    audit_policy: AuditPolicyName,
    audit_fixed_nodes: int,
    audit_fraction: float,
    audit_scale: float,
    c3_audit_strategy: C3AuditStrategyName,
    c3_include_root: bool,
    leaf_query_rate: float,
    seed: int,
) -> TrainFitDiagnostics:
    """
    Closed-form fit for the additive sketch's linear leaf encoder.

    Uses whatever oracle labels the run has "paid for":
      - leaf labels (C1) under `leaf_query_rate`
      - internal-node labels (C3) under `audit_policy`/`audit_fraction`/`c3_audit_strategy`

    Each internal-node label yields a linear equation in the leaf encoder weights because:
      1) internal counts are sums of leaf counts + join indicators, and
      2) join indicators are computable from endpoints when `feature_mode='full'`.
    """

    if len(train_docs) == 0:
        return TrainFitDiagnostics(
            train_loss_final=0.0,
            train_loss_curve=(0.0,),
            epochs_completed=0,
            selection_metric_curve=(0.0,),
            selection_mode="closed_form_solution",
            selection_split="config",
            selection_metric_name="closed_form_train_mse",
            selection_metric_value=0.0,
            best_epoch=0,
        )

    rng = random.Random(int(seed))
    n = int(model.n_regimes)
    X_rows: List[np.ndarray] = []  # each row is [core_features..., bias_coeff]
    y_rows: List[float] = []

    def _leaf_core(feat: torch.Tensor) -> np.ndarray:
        if model.use_endpoints:
            return feat[2 * n :].detach().cpu().numpy().astype(np.float64)
        return feat.detach().cpu().numpy().astype(np.float64)

    def _balanced_merge_leaf_ranges(n_leaves: int) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = [(i, i + 1) for i in range(int(n_leaves))]
        merges: List[Tuple[int, int]] = []
        while len(spans) > 1:
            nxt: List[Tuple[int, int]] = []
            i = 0
            while i < len(spans):
                if i + 1 >= len(spans):
                    nxt.append(spans[i])
                    i += 1
                    continue
                merged = (int(spans[i][0]), int(spans[i + 1][1]))
                merges.append(merged)
                nxt.append(merged)
                i += 2
            spans = nxt
        return merges

    for doc in train_docs:
        n_leaf = int(len(doc.leaf_features))
        if n_leaf <= 0:
            continue

        # ----------------------------
        # Leaf labels (C1)
        # ----------------------------
        q_leaf = leaf_sample_count(n_leaf, rate=float(leaf_query_rate))
        if q_leaf > 0:
            if q_leaf >= n_leaf:
                leaf_idxs = list(range(n_leaf))
            else:
                leaf_idxs = rng.sample(range(n_leaf), k=int(q_leaf))
            for idx in leaf_idxs:
                core = _leaf_core(doc.leaf_features[int(idx)])
                X_rows.append(np.concatenate([core, np.asarray([1.0], dtype=np.float64)], axis=0))
                y_rows.append(float(doc.leaf_counts[int(idx)]) / float(model.target_scale))

        # ----------------------------
        # Internal labels (C3), balanced-merge order
        # ----------------------------
        n_internal = int(max(0, n_leaf - 1))
        if n_internal <= 0 or len(doc.merge_counts_balanced) == 0:
            continue
        ranges = _balanced_merge_leaf_ranges(n_leaf)
        if len(ranges) != n_internal:
            raise ValueError("internal merge range reconstruction failed")

        q_internal = audit_sample_count(
            n_internal,
            policy=str(audit_policy),
            fixed_nodes=int(audit_fixed_nodes),
            fraction=float(audit_fraction),
            scale=float(audit_scale),
        )
        internal_idxs = _sample_internal_audit_indices(
            n_internal,
            k=int(q_internal),
            strategy=str(c3_audit_strategy),
            merge_sizes=doc.merge_sizes_balanced,
            include_root=bool(c3_include_root),
            rng=rng,
        )
        if internal_idxs is None:
            internal_iter = range(n_internal)
        else:
            internal_iter = list(internal_idxs)

        # Precompute per-boundary join indicators (between adjacent leaves).
        if model.use_endpoints:
            first_ids: List[int] = []
            last_ids: List[int] = []
            for feat in doc.leaf_features:
                first_ids.append(int(torch.argmax(feat[:n]).item()))
                last_ids.append(int(torch.argmax(feat[n : 2 * n]).item()))
            join_flags = [0 if last_ids[i] == first_ids[i + 1] else 1 for i in range(n_leaf - 1)]
        else:
            join_flags = [0 for _ in range(max(0, n_leaf - 1))]

        core_cache = [_leaf_core(f) for f in doc.leaf_features]
        for idx in internal_iter:
            k = int(idx)
            if k < 0 or k >= n_internal:
                continue
            a, b = ranges[k]
            if not (0 <= a < b <= n_leaf):
                continue
            span_leaves = int(b - a)
            sum_core = np.sum(np.stack(core_cache[a:b], axis=0), axis=0)
            join_sum = int(sum(join_flags[a : b - 1])) if span_leaves >= 2 else 0
            y_internal_norm = float(doc.merge_counts_balanced[k]) / float(model.target_scale)
            y_target = float(y_internal_norm) - float(join_sum) / float(model.target_scale)

            X_rows.append(
                np.concatenate(
                    [sum_core, np.asarray([float(span_leaves)], dtype=np.float64)], axis=0
                )
            )
            y_rows.append(float(y_target))

    if not X_rows:
        return TrainFitDiagnostics(
            train_loss_final=0.0,
            train_loss_curve=(0.0,),
            epochs_completed=1,
            selection_metric_curve=(0.0,),
            selection_mode="closed_form_solution",
            selection_split="config",
            selection_metric_name="closed_form_train_mse",
            selection_metric_value=0.0,
            best_epoch=0,
        )

    X = np.stack(X_rows, axis=0)
    y = np.asarray(y_rows, dtype=np.float64)
    beta, *_rest = np.linalg.lstsq(X, y, rcond=None)
    w = beta[:-1]
    b = beta[-1]

    with torch.no_grad():
        model.encoder.weight.copy_(
            torch.tensor(w.reshape(1, -1), device=device, dtype=torch.float32)
        )
        model.encoder.bias.copy_(torch.tensor([float(b)], device=device, dtype=torch.float32))

    preds = X @ beta
    mse = float(np.mean((preds - y) ** 2))
    return TrainFitDiagnostics(
        train_loss_final=float(mse),
        train_loss_curve=(float(mse),),
        epochs_completed=1,
        selection_metric_curve=(float(mse),),
        selection_mode="closed_form_solution",
        selection_split="config",
        selection_metric_name="closed_form_train_mse",
        selection_metric_value=float(mse),
        best_epoch=0,
    )


@torch.no_grad()
def _eval_learned_model(
    model: LearnedCountSketch | AdditiveCountSketch,
    docs: Sequence[_CountDoc],
    *,
    device: torch.device,
    tau: float,
) -> SketchMetrics:
    if len(docs) == 0:
        return _zero_sketch_metrics(n_docs=0)

    model.eval()
    root_abs: List[float] = []
    spreads: List[float] = []
    leaf_abs: List[float] = []
    merge_abs: List[float] = []
    c2_r1_abs: List[float] = []
    c2_r2_abs: List[float] = []
    c2_r4_abs: List[float] = []
    root_drift_r1: List[float] = []
    root_drift_r2: List[float] = []
    root_drift_r4: List[float] = []

    for doc in docs:
        leaf_feats = _to_device(doc.leaf_features, device=device)
        states = [model.encode_leaf(x) for x in leaf_feats]

        # Leaf C1.
        for st, truth in zip(states, doc.leaf_counts):
            pred = float(model.predict_count_from_state(st).detach().cpu())
            leaf_abs.append(abs(pred - float(truth)))

        # Merge C3 (balanced schedule, all internal nodes).
        _root_state, merge_states = model._merge_states(
            states,
            schedule="balanced",
            collect_merge_states=True,
        )
        for pred_st, truth in zip(merge_states, doc.merge_counts_balanced):
            pred = float(model.predict_count_from_state(pred_st).detach().cpu())
            merge_abs.append(abs(pred - float(truth)))

        c2_states = list(states)
        c2_states.extend(list(merge_states))
        if not c2_states:
            c2_states = [_root_state]
        for st in c2_states:
            base_summary = model.decode_summary(st)
            base_value = float(_predict_count_from_summary(model, base_summary).detach().cpu())
            replay = _resummary_summary_sequence(model, st, depths=(1, 2, 4))
            c2_r1_abs.append(
                abs(
                    float(_predict_count_from_summary(model, replay[1]).detach().cpu()) - base_value
                )
            )
            c2_r2_abs.append(
                abs(
                    float(_predict_count_from_summary(model, replay[2]).detach().cpu()) - base_value
                )
            )
            c2_r4_abs.append(
                abs(
                    float(_predict_count_from_summary(model, replay[4]).detach().cpu()) - base_value
                )
            )

        # Root distortion + schedule spread.
        roots: Dict[str, float] = {}
        for sched in VALID_SCHEDULES:
            root_state, _ = model._merge_states(states, schedule=sched, collect_merge_states=False)
            roots[str(sched)] = float(model.predict_count_from_state(root_state).detach().cpu())
        pred_root = roots["balanced"]
        root_abs.append(abs(pred_root - float(doc.root_count)))
        spreads.append(max(roots.values()) - min(roots.values()))
        root_replay = _resummary_summary_sequence(model, _root_state, depths=(1, 2, 4))
        root_base = float(
            _predict_count_from_summary(model, model.decode_summary(_root_state)).detach().cpu()
        )
        root_drift_r1.append(
            abs(
                float(_predict_count_from_summary(model, root_replay[1]).detach().cpu()) - root_base
            )
        )
        root_drift_r2.append(
            abs(
                float(_predict_count_from_summary(model, root_replay[2]).detach().cpu()) - root_base
            )
        )
        root_drift_r4.append(
            abs(
                float(_predict_count_from_summary(model, root_replay[4]).detach().cpu()) - root_base
            )
        )

    tau = float(tau)
    leaf_abs_arr = np.asarray(leaf_abs, dtype=np.float64)
    merge_abs_arr = np.asarray(merge_abs, dtype=np.float64)
    root_abs_arr = np.asarray(root_abs, dtype=np.float64)
    spreads_arr = np.asarray(spreads, dtype=np.float64)
    c2_r1_arr = np.asarray(c2_r1_abs, dtype=np.float64)
    c2_r2_arr = np.asarray(c2_r2_abs, dtype=np.float64)
    c2_r4_arr = np.asarray(c2_r4_abs, dtype=np.float64)
    root_drift_r1_arr = np.asarray(root_drift_r1, dtype=np.float64)
    root_drift_r2_arr = np.asarray(root_drift_r2, dtype=np.float64)
    root_drift_r4_arr = np.asarray(root_drift_r4, dtype=np.float64)

    return SketchMetrics(
        root_mae=float(np.mean(root_abs_arr)),
        root_median_abs_error=float(np.median(root_abs_arr)),
        root_p95_abs_error=float(np.percentile(root_abs_arr, 95.0)),
        schedule_spread_mean=float(np.mean(spreads_arr)),
        schedule_spread_p95=float(np.percentile(spreads_arr, 95.0)),
        leaf_mae=float(np.mean(leaf_abs_arr)) if leaf_abs_arr.size else 0.0,
        leaf_violation_rate=(
            float(np.mean((leaf_abs_arr > tau).astype(np.float64))) if leaf_abs_arr.size else 0.0
        ),
        c2_idempotence_mae=float(np.mean(c2_r1_arr)) if c2_r1_arr.size else 0.0,
        c2_r1_mae=float(np.mean(c2_r1_arr)) if c2_r1_arr.size else 0.0,
        c2_r2_mae=float(np.mean(c2_r2_arr)) if c2_r2_arr.size else 0.0,
        c2_r4_mae=float(np.mean(c2_r4_arr)) if c2_r4_arr.size else 0.0,
        resummary_root_drift_r1=(
            float(np.mean(root_drift_r1_arr)) if root_drift_r1_arr.size else 0.0
        ),
        resummary_root_drift_r2=(
            float(np.mean(root_drift_r2_arr)) if root_drift_r2_arr.size else 0.0
        ),
        resummary_root_drift_r4=(
            float(np.mean(root_drift_r4_arr)) if root_drift_r4_arr.size else 0.0
        ),
        merge_mae=float(np.mean(merge_abs_arr)) if merge_abs_arr.size else 0.0,
        merge_violation_rate=(
            float(np.mean((merge_abs_arr > tau).astype(np.float64))) if merge_abs_arr.size else 0.0
        ),
        n_docs=int(len(docs)),
    )


@torch.no_grad()
def _eval_objective_terms(
    model: LearnedCountSketch | AdditiveCountSketch,
    docs: Sequence[_CountDoc],
    *,
    device: torch.device,
    leaf_weight: float,
    c2_weight: float,
    c3_weight: float,
    root_weight: float,
    schedule_consistency_weight: float,
    include_root_query: bool,
) -> ObjectiveMetrics:
    if len(docs) == 0:
        return ObjectiveMetrics(
            optimization_total_loss=0.0,
            optimization_root_loss=0.0,
            optimization_leaf_loss=0.0,
            optimization_c2_loss=0.0,
            optimization_merge_loss=0.0,
            optimization_schedule_consistency_loss=0.0,
            raw_total_loss=0.0,
            raw_root_loss=0.0,
            raw_leaf_loss=0.0,
            raw_c2_loss=0.0,
            raw_merge_loss=0.0,
            raw_schedule_consistency_loss=0.0,
            n_docs=0,
        )

    model.eval()
    optimization_total_terms: List[float] = []
    optimization_root_terms: List[float] = []
    optimization_leaf_terms: List[float] = []
    optimization_c2_terms: List[float] = []
    optimization_merge_terms: List[float] = []
    optimization_consistency_terms: List[float] = []
    raw_total_terms: List[float] = []
    raw_root_terms: List[float] = []
    raw_leaf_terms: List[float] = []
    raw_c2_terms: List[float] = []
    raw_merge_terms: List[float] = []
    raw_consistency_terms: List[float] = []

    for doc in docs:
        leaf_feats = _to_device(doc.leaf_features, device=device)
        out = model.forward_doc(
            leaf_feats,
            doc.leaf_counts,
            doc.merge_counts_balanced,
            schedule="balanced",
            collect_leaf=True,
            collect_c3=True,
            collect_c2=True,
            leaf_audit_indices=None,
            c3_audit_indices=None,
        )
        pred_norm = out["pred_norm"]
        leaf_loss_tensor = out["leaf_loss"]
        c2_loss_tensor = out["c2_loss"]
        c3_loss_tensor = out["c3_loss"]
        if not isinstance(pred_norm, torch.Tensor):
            raise TypeError("expected tensor pred_norm from forward_doc")
        if (
            not isinstance(leaf_loss_tensor, torch.Tensor)
            or not isinstance(c2_loss_tensor, torch.Tensor)
            or not isinstance(c3_loss_tensor, torch.Tensor)
        ):
            raise TypeError("expected tensor leaf/c2/c3 losses from forward_doc")

        true_norm = torch.tensor(
            float(doc.root_count) / float(getattr(model, "target_scale", 1.0)),
            device=device,
            dtype=pred_norm.dtype,
        )
        raw_root_term = (
            float(F.mse_loss(pred_norm, true_norm, reduction="mean").detach().cpu())
            if bool(include_root_query)
            else 0.0
        )
        raw_leaf_term = float(leaf_loss_tensor.detach().cpu())
        raw_c2_term = float(c2_loss_tensor.detach().cpu())
        raw_merge_term = float(c3_loss_tensor.detach().cpu())

        if float(schedule_consistency_weight) > 0.0 and len(leaf_feats) > 1:
            states_sched = [model.encode_leaf(x) for x in leaf_feats]
            sched_preds = []
            for sched in VALID_SCHEDULES:
                root_state_sched, _ = model._merge_states(
                    states_sched,
                    schedule=sched,
                    collect_merge_states=False,
                )
                sched_preds.append(model.predict_norm_from_state(root_state_sched))
            pred_stack = torch.stack(sched_preds, dim=0)
            consistency_raw = torch.mean((pred_stack - torch.mean(pred_stack)) ** 2)
            raw_consistency_term = float(consistency_raw.detach().cpu())
        else:
            raw_consistency_term = 0.0

        optimization_root_term = float(root_weight) * raw_root_term
        optimization_leaf_term = float(leaf_weight) * raw_leaf_term
        optimization_c2_term = float(c2_weight) * raw_c2_term
        optimization_merge_term = float(c3_weight) * raw_merge_term
        optimization_consistency_term = float(schedule_consistency_weight) * raw_consistency_term

        optimization_total_terms.append(
            float(
                optimization_root_term
                + optimization_leaf_term
                + optimization_c2_term
                + optimization_merge_term
                + optimization_consistency_term
            )
        )
        optimization_root_terms.append(float(optimization_root_term))
        optimization_leaf_terms.append(float(optimization_leaf_term))
        optimization_c2_terms.append(float(optimization_c2_term))
        optimization_merge_terms.append(float(optimization_merge_term))
        optimization_consistency_terms.append(float(optimization_consistency_term))
        raw_total_terms.append(
            float(
                raw_root_term + raw_leaf_term + raw_c2_term + raw_merge_term + raw_consistency_term
            )
        )
        raw_root_terms.append(float(raw_root_term))
        raw_leaf_terms.append(float(raw_leaf_term))
        raw_c2_terms.append(float(raw_c2_term))
        raw_merge_terms.append(float(raw_merge_term))
        raw_consistency_terms.append(float(raw_consistency_term))

    optimization_total_arr = np.asarray(optimization_total_terms, dtype=np.float64)
    optimization_root_arr = np.asarray(optimization_root_terms, dtype=np.float64)
    optimization_leaf_arr = np.asarray(optimization_leaf_terms, dtype=np.float64)
    optimization_c2_arr = np.asarray(optimization_c2_terms, dtype=np.float64)
    optimization_merge_arr = np.asarray(optimization_merge_terms, dtype=np.float64)
    optimization_consistency_arr = np.asarray(optimization_consistency_terms, dtype=np.float64)
    raw_total_arr = np.asarray(raw_total_terms, dtype=np.float64)
    raw_root_arr = np.asarray(raw_root_terms, dtype=np.float64)
    raw_leaf_arr = np.asarray(raw_leaf_terms, dtype=np.float64)
    raw_c2_arr = np.asarray(raw_c2_terms, dtype=np.float64)
    raw_merge_arr = np.asarray(raw_merge_terms, dtype=np.float64)
    raw_consistency_arr = np.asarray(raw_consistency_terms, dtype=np.float64)
    return ObjectiveMetrics(
        optimization_total_loss=float(np.mean(optimization_total_arr)),
        optimization_root_loss=float(np.mean(optimization_root_arr)),
        optimization_leaf_loss=float(np.mean(optimization_leaf_arr)),
        optimization_c2_loss=float(np.mean(optimization_c2_arr)),
        optimization_merge_loss=float(np.mean(optimization_merge_arr)),
        optimization_schedule_consistency_loss=float(np.mean(optimization_consistency_arr)),
        raw_total_loss=float(np.mean(raw_total_arr)),
        raw_root_loss=float(np.mean(raw_root_arr)),
        raw_leaf_loss=float(np.mean(raw_leaf_arr)),
        raw_c2_loss=float(np.mean(raw_c2_arr)),
        raw_merge_loss=float(np.mean(raw_merge_arr)),
        raw_schedule_consistency_loss=float(np.mean(raw_consistency_arr)),
        n_docs=int(len(docs)),
    )


def _rf_doc_features(doc: _CountDoc) -> np.ndarray:
    leaf = list(doc.leaf_features)
    if not leaf:
        raise ValueError("rf baseline requires at least one leaf per doc")
    feats = torch.stack(leaf, dim=0).to(dtype=torch.float32, device="cpu")
    mean = feats.mean(dim=0)
    std = feats.std(dim=0, unbiased=False)
    n_leaves = torch.tensor([float(feats.shape[0])], dtype=torch.float32)
    out = torch.cat([mean, std, n_leaves], dim=0).detach().cpu().numpy()
    return np.asarray(out, dtype=np.float64)


def _eval_rf_root_baseline(
    train_docs: Sequence[_CountDoc],
    test_docs: Sequence[_CountDoc],
    *,
    seed: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
) -> SketchMetrics:
    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required for include_rf_root_baseline. "
            "Install with: pip install scikit-learn>=1.4.2"
        ) from e

    if not train_docs or not test_docs:
        return SketchMetrics(
            root_mae=float("nan"),
            root_median_abs_error=float("nan"),
            root_p95_abs_error=float("nan"),
            schedule_spread_mean=0.0,
            schedule_spread_p95=0.0,
            leaf_mae=float("nan"),
            leaf_violation_rate=float("nan"),
            c2_idempotence_mae=float("nan"),
            c2_r1_mae=float("nan"),
            c2_r2_mae=float("nan"),
            c2_r4_mae=float("nan"),
            resummary_root_drift_r1=float("nan"),
            resummary_root_drift_r2=float("nan"),
            resummary_root_drift_r4=float("nan"),
            merge_mae=float("nan"),
            merge_violation_rate=float("nan"),
            n_docs=int(len(test_docs)),
        )

    ne = int(n_estimators)
    if ne <= 0:
        raise ValueError("rf_n_estimators must be positive")
    md = int(max_depth)
    if md <= 0:
        raise ValueError("rf_max_depth must be positive")
    msl = int(min_samples_leaf)
    if msl <= 0:
        raise ValueError("rf_min_samples_leaf must be positive")

    X_train = np.stack([_rf_doc_features(d) for d in train_docs], axis=0).astype(
        np.float32, copy=False
    )
    y_train = np.asarray([float(d.root_count) for d in train_docs], dtype=np.float64)
    X_test = np.stack([_rf_doc_features(d) for d in test_docs], axis=0).astype(
        np.float32, copy=False
    )
    y_test = np.asarray([float(d.root_count) for d in test_docs], dtype=np.float64)

    model = RandomForestRegressor(
        n_estimators=int(ne),
        max_depth=int(md),
        min_samples_leaf=int(msl),
        random_state=int(seed),
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    pred = np.asarray(model.predict(X_test), dtype=np.float64)
    abs_err = np.abs(pred - y_test)

    return SketchMetrics(
        root_mae=float(np.mean(abs_err)),
        root_median_abs_error=float(np.median(abs_err)),
        root_p95_abs_error=float(np.percentile(abs_err, 95.0)),
        schedule_spread_mean=0.0,
        schedule_spread_p95=0.0,
        leaf_mae=float("nan"),
        leaf_violation_rate=float("nan"),
        c2_idempotence_mae=float("nan"),
        c2_r1_mae=float("nan"),
        c2_r2_mae=float("nan"),
        c2_r4_mae=float("nan"),
        resummary_root_drift_r1=float("nan"),
        resummary_root_drift_r2=float("nan"),
        resummary_root_drift_r4=float("nan"),
        merge_mae=float("nan"),
        merge_violation_rate=float("nan"),
        n_docs=int(len(test_docs)),
    )


def _clip_norm_target(t: float) -> float:
    v = float(t)
    if v <= 0.0:
        return 0.0
    if v >= 1.0:
        return 1.0
    return v


def _override_state_with_oracle_count(
    model: LearnedCountSketch | AdditiveCountSketch,
    state: torch.Tensor,
    *,
    target_count: float,
    override_mode: GuidanceOverrideModeName,
) -> torch.Tensor:
    target_scale = float(getattr(model, "target_scale", 1.0))
    target_norm = _clip_norm_target(float(target_count) / float(max(1e-12, target_scale)))

    if isinstance(model, AdditiveCountSketch):
        _count, first, last = model._split_state(state)
        guided_count = torch.full_like(_count, float(target_norm))
        return torch.cat([guided_count.unsqueeze(-1), first, last], dim=-1)

    if not isinstance(model, LearnedCountSketch):
        raise TypeError(f"unsupported model type for guidance override: {type(model)!r}")

    mode = str(override_mode).strip().lower()
    if mode not in VALID_GUIDANCE_OVERRIDE_MODES:
        raise ValueError(
            f"unsupported guidance override mode: {override_mode!r}; expected one of {VALID_GUIDANCE_OVERRIDE_MODES}"
        )

    _h, first, last = model._split_state(state)
    w = model.readout.weight.squeeze(0)
    b = model.readout.bias.squeeze()

    if target_norm <= 0.0:
        z = torch.tensor(-80.0, device=state.device, dtype=state.dtype)
    elif target_norm >= 1.0:
        z = torch.tensor(80.0, device=state.device, dtype=state.dtype)
    else:
        tt = torch.tensor(float(target_norm), device=state.device, dtype=state.dtype)
        z = torch.log(tt / (1.0 - tt))

    denom = torch.sum(w * w)
    if float(denom.detach().cpu()) <= 1e-20:
        guided_h = torch.zeros_like(_h)
    else:
        if mode == "reset":
            alpha = (z - b) / denom
            guided_h = alpha * w
        else:
            cur_logit = torch.sum(w * _h) + b
            delta = (z - cur_logit) / denom
            guided_h = _h + delta * w
    return torch.cat([guided_h, first, last], dim=-1)


def _merge_balanced_with_guidance(
    model: LearnedCountSketch | AdditiveCountSketch,
    states: Sequence[torch.Tensor],
    *,
    merge_truth_counts_balanced: Sequence[float],
    guided_internal_indices: set[int],
    guidance_override_mode: GuidanceOverrideModeName,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[float]]:
    if len(states) == 0:
        raise ValueError("need at least one state")
    if len(states) == 1:
        return states[0], [], []

    merge_states: List[torch.Tensor] = []
    merge_pred_counts: List[float] = []
    cur = list(states)
    merge_idx = 0
    while len(cur) > 1:
        nxt: List[torch.Tensor] = []
        i = 0
        while i < len(cur):
            if i + 1 >= len(cur):
                nxt.append(cur[i])
                i += 1
                continue

            left = cur[i]
            right = cur[i + 1]
            if isinstance(model, LearnedCountSketch):
                left_h, left_first, left_last = model._split_state(left)
                right_h, right_first, right_last = model._split_state(right)
                merged_h = model.merger(
                    torch.cat([left_h, right_h, left_last, right_first], dim=-1)
                )
                merged = torch.cat([merged_h, left_first, right_last], dim=-1)
            elif isinstance(model, AdditiveCountSketch):
                left_c, left_first, left_last = model._split_state(left)
                right_c, right_first, right_last = model._split_state(right)
                if bool(model.use_endpoints):
                    same = torch.sum(left_last * right_first, dim=-1)
                    join = 1.0 - same
                    join_term = join / float(model.target_scale)
                else:
                    join_term = torch.zeros_like(left_c)
                merged_c = left_c + right_c + join_term
                merged = torch.cat([merged_c.unsqueeze(-1), left_first, right_last], dim=-1)
            else:
                raise TypeError(f"unsupported model type: {type(model)!r}")

            if merge_idx in guided_internal_indices and merge_idx < len(
                merge_truth_counts_balanced
            ):
                merged = _override_state_with_oracle_count(
                    model,
                    merged,
                    target_count=float(merge_truth_counts_balanced[merge_idx]),
                    override_mode=str(guidance_override_mode),
                )
                merge_pred_count = float(merge_truth_counts_balanced[merge_idx])
            else:
                merge_pred_count = float(model.predict_count_from_state(merged).detach().cpu())
            merge_states.append(merged)
            merge_pred_counts.append(float(merge_pred_count))
            nxt.append(merged)
            merge_idx += 1
            i += 2
        cur = nxt
    return cur[0], merge_states, merge_pred_counts


def _sample_guided_internal_indices(
    n_internal: int,
    *,
    q: float,
    include_root: bool,
    rng: random.Random,
) -> set[int]:
    n = int(max(0, n_internal))
    if n <= 0:
        return set()
    qq = float(max(0.0, min(1.0, float(q))))
    if qq <= 0.0:
        return set()

    if include_root:
        pool = list(range(n))
    else:
        pool = list(range(max(0, n - 1)))
    if not pool:
        return set()
    k = int(math.ceil(qq * float(n)))
    k = int(max(0, min(int(len(pool)), k)))
    if k <= 0:
        return set()
    if k >= len(pool):
        return set(pool)
    return set(int(i) for i in rng.sample(pool, k=k))


@torch.no_grad()
def _eval_guided_model_curve(
    model: LearnedCountSketch | AdditiveCountSketch,
    docs: Sequence[_CountDoc],
    *,
    device: torch.device,
    tau: float,
    guidance_qs: Sequence[float],
    guidance_trials: int,
    guidance_include_root: bool,
    guidance_override_mode: GuidanceOverrideModeName,
    guidance_seed: int,
) -> Dict[str, object]:
    qs = [float(max(0.0, min(1.0, q))) for q in guidance_qs]
    qs = sorted({float(q) for q in qs})
    trials = int(max(0, guidance_trials))
    if len(docs) == 0 or trials <= 0 or not qs:
        return {
            "present": False,
            "include_root": bool(guidance_include_root),
            "trials": int(trials),
            "points": [],
        }

    model.eval()
    points: List[Dict[str, float | int]] = []
    for q_idx, q in enumerate(qs):
        root_abs: List[float] = []
        leaf_abs: List[float] = []
        merge_abs: List[float] = []
        guided_nodes: List[float] = []
        effective_qs: List[float] = []
        for trial in range(trials):
            rng = random.Random(int(guidance_seed) + 7919 * int(q_idx + 1) + 97 * int(trial + 1))
            for doc in docs:
                leaf_feats = _to_device(doc.leaf_features, device=device)
                states = [model.encode_leaf(x) for x in leaf_feats]
                n_internal = int(max(0, len(states) - 1))
                guided_idx = _sample_guided_internal_indices(
                    n_internal,
                    q=float(q),
                    include_root=bool(guidance_include_root),
                    rng=rng,
                )
                root_state, merge_states, merge_pred_counts = _merge_balanced_with_guidance(
                    model,
                    states,
                    merge_truth_counts_balanced=doc.merge_counts_balanced,
                    guided_internal_indices=guided_idx,
                    guidance_override_mode=str(guidance_override_mode),
                )
                root_idx = int(n_internal - 1)
                if (
                    root_idx >= 0
                    and root_idx in guided_idx
                    and root_idx < len(doc.merge_counts_balanced)
                ):
                    pred_root = float(doc.merge_counts_balanced[root_idx])
                else:
                    pred_root = float(model.predict_count_from_state(root_state).detach().cpu())
                root_abs.append(abs(pred_root - float(doc.root_count)))

                for st, truth in zip(states, doc.leaf_counts):
                    pred_leaf = float(model.predict_count_from_state(st).detach().cpu())
                    leaf_abs.append(abs(pred_leaf - float(truth)))

                for idx, pred_st in enumerate(merge_states):
                    if idx >= len(doc.merge_counts_balanced):
                        break
                    if idx < len(merge_pred_counts):
                        pred_merge = float(merge_pred_counts[idx])
                    else:
                        pred_merge = float(model.predict_count_from_state(pred_st).detach().cpu())
                    merge_abs.append(abs(pred_merge - float(doc.merge_counts_balanced[idx])))

                guided_nodes.append(float(len(guided_idx)))
                effective_qs.append(float(len(guided_idx)) / float(max(1, n_internal)))

        root_arr = np.asarray(root_abs, dtype=np.float64)
        leaf_arr = np.asarray(leaf_abs, dtype=np.float64)
        merge_arr = np.asarray(merge_abs, dtype=np.float64)
        guided_arr = np.asarray(guided_nodes, dtype=np.float64)
        effq_arr = np.asarray(effective_qs, dtype=np.float64)
        tau_v = float(tau)
        points.append(
            {
                "q": float(q),
                "root_mae": float(np.mean(root_arr)) if root_arr.size else 0.0,
                "root_median_abs_error": float(np.median(root_arr)) if root_arr.size else 0.0,
                "root_p95_abs_error": (
                    float(np.percentile(root_arr, 95.0)) if root_arr.size else 0.0
                ),
                "leaf_mae": float(np.mean(leaf_arr)) if leaf_arr.size else 0.0,
                "leaf_violation_rate": (
                    float(np.mean((leaf_arr > tau_v).astype(np.float64))) if leaf_arr.size else 0.0
                ),
                "merge_mae": float(np.mean(merge_arr)) if merge_arr.size else 0.0,
                "merge_violation_rate": (
                    float(np.mean((merge_arr > tau_v).astype(np.float64)))
                    if merge_arr.size
                    else 0.0
                ),
                "guided_internal_nodes_mean": (
                    float(np.mean(guided_arr)) if guided_arr.size else 0.0
                ),
                "guided_internal_nodes_p95": (
                    float(np.percentile(guided_arr, 95.0)) if guided_arr.size else 0.0
                ),
                "effective_q_mean": float(np.mean(effq_arr)) if effq_arr.size else 0.0,
                "n_eval_docs": int(len(docs) * trials),
            }
        )

    return {
        "present": True,
        "include_root": bool(guidance_include_root),
        "trials": int(trials),
        "points": points,
    }


def _audit_estimator_diagnostics(
    values: Sequence[float],
    preds: Sequence[float],
    inclusion_probs: Sequence[float],
    *,
    trials: int,
    seed: int,
) -> EstimatorDiagnostics:
    y = np.asarray(values, dtype=np.float64)
    m = np.asarray(preds, dtype=np.float64)
    pi = np.asarray(inclusion_probs, dtype=np.float64)
    if y.size == 0:
        return EstimatorDiagnostics(
            true_mean=0.0,
            naive_bias=0.0,
            ipw_bias=0.0,
            dsl_bias=0.0,
            ipw_var=0.0,
            dsl_var=0.0,
        )
    if y.shape != m.shape or y.shape != pi.shape:
        raise ValueError("values, preds, inclusion_probs must have the same shape")
    if np.any(pi <= 0.0) or np.any(pi > 1.0):
        raise ValueError("inclusion_probs must lie in (0,1]")

    rng = np.random.default_rng(int(seed))
    N = int(y.size)
    true_mean = float(np.mean(y))
    naive: List[float] = []
    ipw: List[float] = []
    dsl: List[float] = []

    for _ in range(int(max(1, trials))):
        inc = rng.random(N) < pi
        if not np.any(inc):
            continue
        y_s = y[inc]
        pi_s = pi[inc]
        m_s = m[inc]

        naive.append(float(np.mean(y_s)))
        ipw.append(float(np.sum(y_s / pi_s) / float(N)))
        dsl.append(float(np.mean(m) + np.sum((y_s - m_s) / pi_s) / float(N)))

    def _bias(xs: Sequence[float]) -> float:
        return float(np.mean(np.asarray(xs, dtype=np.float64)) - true_mean) if xs else 0.0

    def _var(xs: Sequence[float]) -> float:
        arr = np.asarray(xs, dtype=np.float64)
        return float(np.var(arr)) if arr.size else 0.0

    return EstimatorDiagnostics(
        true_mean=float(true_mean),
        naive_bias=float(_bias(naive)),
        ipw_bias=float(_bias(ipw)),
        dsl_bias=float(_bias(dsl)),
        ipw_var=float(_var(ipw)),
        dsl_var=float(_var(dsl)),
    )


def run_markov_changepoint_ops_count_experiment(config: OPSCountConfig) -> OPSCountSummary:
    if int(config.n_regimes) < 1:
        raise ValueError("n_regimes must be >= 1")
    if int(config.fixed_leaf_tokens) <= 0:
        raise ValueError("fixed_leaf_tokens must be positive")
    if int(config.train_docs) < 0 or int(config.val_docs) < 0 or int(config.test_docs) < 0:
        raise ValueError("train_docs/val_docs/test_docs must be non-negative")
    if str(config.audit_policy) not in VALID_AUDIT_POLICIES:
        raise ValueError(
            f"audit_policy={config.audit_policy!r} unsupported; expected one of {VALID_AUDIT_POLICIES}"
        )
    if float(config.audit_fraction) < 0.0:
        raise ValueError("audit_fraction must be non-negative")
    if float(config.audit_scale) <= 0.0:
        raise ValueError("audit_scale must be positive")
    if int(config.audit_fixed_nodes) < 0:
        raise ValueError("audit_fixed_nodes must be non-negative")
    if str(config.c3_audit_strategy) not in VALID_C3_AUDIT_STRATEGIES:
        raise ValueError(
            "c3_audit_strategy="
            f"{config.c3_audit_strategy!r} unsupported; expected one of {VALID_C3_AUDIT_STRATEGIES}"
        )
    if float(config.leaf_query_rate) < 0.0 or float(config.leaf_query_rate) > 1.0:
        raise ValueError("leaf_query_rate must lie in [0,1]")
    if float(config.leaf_weight) < 0.0:
        raise ValueError("leaf_weight must be non-negative")
    if float(config.c2_weight) < 0.0:
        raise ValueError("c2_weight must be non-negative")
    if float(config.c3_weight) < 0.0:
        raise ValueError("c3_weight must be non-negative")
    if config.local_law_weight is not None and float(config.local_law_weight) < 0.0:
        raise ValueError("local_law_weight must be non-negative")
    if config.local_law_weight is not None and float(config.local_law_weight) > 1.0:
        raise ValueError(
            "local_law_weight must lie in [0,1] under the normalized lambda parameterization"
        )
    if config.task_objective_weight is not None and float(config.task_objective_weight) < 0.0:
        raise ValueError("task_objective_weight must be non-negative")
    if float(config.c1_relative_weight) < 0.0:
        raise ValueError("c1_relative_weight must be non-negative")
    if float(config.c2_relative_weight) < 0.0:
        raise ValueError("c2_relative_weight must be non-negative")
    if float(config.c3_relative_weight) < 0.0:
        raise ValueError("c3_relative_weight must be non-negative")
    if (
        config.local_law_weight is not None
        and float(config.local_law_weight) > 0.0
        and float(config.c1_relative_weight)
        + float(config.c2_relative_weight)
        + float(config.c3_relative_weight)
        <= 0.0
    ):
        raise ValueError(
            "local_law_weight > 0 requires c1_relative_weight + c2_relative_weight + c3_relative_weight > 0"
        )
    if float(config.root_weight) < 0.0:
        raise ValueError("root_weight must be non-negative")
    if float(config.schedule_consistency_weight) < 0.0:
        raise ValueError("schedule_consistency_weight must be non-negative")
    if str(config.law_package or "").strip():
        package = str(config.law_package).strip().lower()
        if package not in VALID_LAW_PACKAGES:
            raise ValueError(
                f"law_package={package!r} unsupported; expected one of {VALID_LAW_PACKAGES}"
            )
    if str(config.exact_family or "").strip():
        exact_family = str(config.exact_family).strip()
        if exact_family not in VALID_EXACT_FAMILIES:
            raise ValueError(
                f"exact_family={exact_family!r} unsupported; expected one of {VALID_EXACT_FAMILIES}"
            )
    if str(config.model_family) not in VALID_MODEL_FAMILIES:
        raise ValueError(
            f"model_family={config.model_family!r} unsupported; expected one of {VALID_MODEL_FAMILIES}"
        )
    if int(config.eval_guidance_trials) < 0:
        raise ValueError("eval_guidance_trials must be non-negative")
    for q in tuple(config.eval_guidance_qs):
        qf = float(q)
        if qf < 0.0 or qf > 1.0:
            raise ValueError("eval_guidance_qs must contain values in [0,1]")
    if str(config.guidance_override_mode) not in VALID_GUIDANCE_OVERRIDE_MODES:
        raise ValueError(
            "guidance_override_mode="
            f"{config.guidance_override_mode!r} unsupported; expected one of {VALID_GUIDANCE_OVERRIDE_MODES}"
        )
    if bool(config.include_rf_root_baseline):
        if not bool(config.include_root_query):
            raise ValueError("include_rf_root_baseline requires include_root_query=true")
        if int(config.rf_n_estimators) <= 0:
            raise ValueError("rf_n_estimators must be positive")
        if int(config.rf_max_depth) <= 0:
            raise ValueError("rf_max_depth must be positive")
        if int(config.rf_min_samples_leaf) <= 0:
            raise ValueError("rf_min_samples_leaf must be positive")

    seeds = _resolve_runtime_seeds(config)
    _set_global_seed(int(seeds["effective_model_seed"]))
    if int(config.torch_threads) > 0:
        torch.set_num_threads(int(config.torch_threads))

    if config.use_cuda and torch.cuda.is_available():
        if config.cuda_device is not None:
            idx = int(config.cuda_device)
            if idx < 0 or idx >= int(torch.cuda.device_count()):
                raise ValueError(f"cuda_device={idx} out of range")
            torch.cuda.set_device(idx)
            device = torch.device(f"cuda:{idx}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    objective = _build_objective_summary(config)
    config_payload = {
        **asdict(config),
        **seeds,
    }

    # Generate docs (token_regimes are the observed sequence; tokens are unused by this simulation).
    gen_cfg = _GeneratorConfig(
        n_regimes=int(config.n_regimes),
        vocab_size=int(config.vocab_size),
        min_tokens=int(config.min_tokens),
        max_tokens=int(config.max_tokens),
        min_segments=int(config.min_segments),
        max_segments=int(config.max_segments),
        min_seg_len=int(config.min_seg_len),
        max_seg_len=int(config.max_seg_len),
        train_docs=int(config.train_docs),
        test_docs=int(config.test_docs),
        seed=int(seeds["effective_data_seed"]),
        sinkhorn_iters=int(config.sinkhorn_iters),
        transition_log_std=float(config.transition_log_std),
        use_cuda=False,
    )
    rng = np.random.default_rng(int(seeds["effective_data_seed"]))
    transitions = _make_transition_matrices(
        n_classes=int(config.n_regimes),
        vocab_size=int(config.vocab_size),
        log_std=float(config.transition_log_std),
        sinkhorn_iters=int(config.sinkhorn_iters),
        rng=rng,
    )
    # Generate train/val/test docs separately so that evaluation sets are stable across `train_docs`
    # and hyperparameter settings for a fixed `data_seed`.
    gen_train = _GeneratorConfig(
        **{**asdict(gen_cfg), "train_docs": int(config.train_docs), "test_docs": 0}
    )
    docs_train = generate_changepoint_docs(gen_train, transitions=transitions)
    gen_val = _GeneratorConfig(
        **{
            **asdict(gen_cfg),
            "train_docs": 0,
            "test_docs": int(config.val_docs),
            "seed": int(seeds["effective_val_seed"]),
        }
    )
    docs_val = generate_changepoint_docs(gen_val, transitions=transitions)
    gen_test = _GeneratorConfig(
        **{
            **asdict(gen_cfg),
            "train_docs": 0,
            "test_docs": int(config.test_docs),
            # Ensure the test set differs from the train set while remaining deterministic per run.
            "seed": int(seeds["effective_test_seed"]),
        }
    )
    docs_test = generate_changepoint_docs(gen_test, transitions=transitions)

    # Deterministic baselines (no training).
    exact = _eval_exact_family(
        docs_test,
        leaf_tokens=int(config.fixed_leaf_tokens),
        tau=float(config.violation_tau),
    )
    leaf_bucket = _eval_leaf_bucket_family(
        docs_test,
        leaf_tokens=int(config.fixed_leaf_tokens),
        tau=float(config.violation_tau),
    )
    undersupported = _eval_count_only_family(
        docs_test,
        leaf_tokens=int(config.fixed_leaf_tokens),
        tau=float(config.violation_tau),
    )
    flip_r1 = _eval_flip_family(
        docs_test,
        leaf_tokens=int(config.fixed_leaf_tokens),
        tau=float(config.violation_tau),
        rounds=1,
    )
    flip_r2 = _eval_flip_family(
        docs_test,
        leaf_tokens=int(config.fixed_leaf_tokens),
        tau=float(config.violation_tau),
        rounds=2,
    )
    exact_families: Dict[str, SketchMetrics] = {
        "exact": exact,
        "leaf_bucket": leaf_bucket,
        "count_only": undersupported,
        "flip_R2": flip_r2,
    }

    # Learned sketch.
    train_prepped = _prepare_count_docs(
        docs_train,
        leaf_tokens=int(config.fixed_leaf_tokens),
        n_regimes=int(config.n_regimes),
        feature_mode=str(config.feature_mode),
    )
    val_prepped = _prepare_count_docs(
        docs_val,
        leaf_tokens=int(config.fixed_leaf_tokens),
        n_regimes=int(config.n_regimes),
        feature_mode=str(config.feature_mode),
    )
    test_prepped = _prepare_count_docs(
        docs_test,
        leaf_tokens=int(config.fixed_leaf_tokens),
        n_regimes=int(config.n_regimes),
        feature_mode=str(config.feature_mode),
    )
    target_scale = float(max(1, int(config.max_segments) - 1))
    if str(config.exact_family or "").strip():
        selected_stress_family = exact_families[str(config.exact_family).strip()]
        geom = _training_geometry(
            train_prepped,
            policy=str(config.audit_policy),
            fixed_nodes=int(config.audit_fixed_nodes),
            fraction=float(config.audit_fraction),
            scale=float(config.audit_scale),
            leaf_query_rate=float(config.leaf_query_rate),
            include_root_query=bool(config.include_root_query),
        )
        metrics: Dict[str, object] = {
            "stress_family": {
                **asdict(selected_stress_family),
                "stress_family_name": str(config.exact_family).strip(),
                **_metrics_with_split_prefix(
                    selected_stress_family, prefix="test", target_scale=target_scale
                ),
                "test_theorem_bundle_score_n": float(
                    markov_law_bundle_score(
                        c1=float(selected_stress_family.leaf_mae) / float(target_scale),
                        c2=float(selected_stress_family.c2_idempotence_mae) / float(target_scale),
                        c3=float(selected_stress_family.merge_mae) / float(target_scale),
                    )
                ),
            },
            "exact": asdict(exact),
            "leaf_bucket": asdict(leaf_bucket),
            "undersupported": asdict(undersupported),
            "flip_R1": asdict(flip_r1),
            "flip_R2": asdict(flip_r2),
        }
        current_role = (
            PolicyRole.ORACLE_G.value
            if str(config.exact_family).strip() == "exact"
            else PolicyRole.COUNTEREXAMPLE_G.value
        )
        local_law_learnability, g_artifacts = _build_markov_local_law_learnability(
            config=config,
            seeds=seeds,
            target_scale=float(target_scale),
            objective_summary=objective,
            geom=geom,
            exact=exact,
            leaf_bucket=leaf_bucket,
            undersupported=undersupported,
            flip_r2=flip_r2,
            current_name=str(config.exact_family).strip(),
            current_role=str(current_role),
            current_train=None,
            current_val=None,
            current_test=selected_stress_family,
            current_selection_metric_name="configured_exact_family",
            current_selection_metric=float(
                markov_law_bundle_score(
                    c1=float(selected_stress_family.leaf_mae) / float(target_scale),
                    c2=float(selected_stress_family.c2_idempotence_mae) / float(target_scale),
                    c3=float(selected_stress_family.merge_mae) / float(target_scale),
                )
            ),
            model=None,
        )
        return OPSCountSummary(
            config=config_payload,
            training_geometry=asdict(geom),
            objective=objective,
            metrics=metrics,
            estimator_diagnostics={
                **asdict(
                    EstimatorDiagnostics(
                        true_mean=0.0,
                        naive_bias=0.0,
                        ipw_bias=0.0,
                        dsl_bias=0.0,
                        ipw_var=0.0,
                        dsl_var=0.0,
                    )
                ),
                "selection_demo_base_rate": 0.0,
                "selection_demo_pi_min": 0.0,
                "selection_demo_n_units": 0.0,
            },
            local_law_learnability=local_law_learnability,
            g_artifacts=g_artifacts,
        )
    if train_prepped:
        feature_dim = int(train_prepped[0].leaf_features[0].numel())
        if str(config.model_family) == "additive":
            model = AdditiveCountSketch(
                feature_dim=int(feature_dim),
                hidden_dim=int(config.hidden_dim),
                target_scale=float(target_scale),
                n_regimes=int(config.n_regimes),
                use_endpoints=str(config.feature_mode) == "full",
            ).to(device=device)
            train_loss_final = _fit_additive_leaf_encoder_closed_form(
                model,
                train_prepped,
                device=device,
                audit_policy=str(config.audit_policy),
                audit_fixed_nodes=int(config.audit_fixed_nodes),
                audit_fraction=float(config.audit_fraction),
                audit_scale=float(config.audit_scale),
                c3_audit_strategy=str(config.c3_audit_strategy),
                c3_include_root=bool(config.c3_include_root),
                leaf_query_rate=float(config.leaf_query_rate),
                seed=int(seeds["effective_model_seed"]),
            )
        else:
            model = LearnedCountSketch(
                feature_dim=int(feature_dim),
                state_dim=int(config.state_dim),
                hidden_dim=int(config.hidden_dim),
                target_scale=float(target_scale),
                n_regimes=int(config.n_regimes),
                use_endpoints=str(config.feature_mode) == "full",
            ).to(device=device)
            train_loss_final = _train_learned_model(
                model,
                train_prepped,
                val_prepped,
                device=device,
                audit_policy=str(config.audit_policy),
                audit_fixed_nodes=int(config.audit_fixed_nodes),
                audit_fraction=float(config.audit_fraction),
                audit_scale=float(config.audit_scale),
                c3_audit_strategy=str(config.c3_audit_strategy),
                c3_include_root=bool(config.c3_include_root),
                leaf_query_rate=float(config.leaf_query_rate),
                include_root_query=bool(config.include_root_query),
                n_epochs=int(config.n_epochs),
                batch_size=int(config.batch_size),
                lr=float(config.lr),
                weight_decay=float(config.weight_decay),
                c3_weight=float(objective["local_law_c3_weight"]),
                c2_weight=float(objective["local_law_c2_weight"]),
                leaf_weight=float(objective["local_law_c1_weight"]),
                root_weight=float(objective["optimization_root_weight"]),
                schedule_consistency_weight=float(objective["proxy_schedule_consistency_weight"]),
                grad_clip_norm=float(config.grad_clip_norm),
                seed=int(seeds["effective_model_seed"]),
            )
        learned_train = _eval_learned_model(
            model,
            train_prepped,
            device=device,
            tau=float(config.violation_tau),
        )
        learned_val = _eval_learned_model(
            model,
            val_prepped,
            device=device,
            tau=float(config.violation_tau),
        )
        learned = _eval_learned_model(
            model,
            test_prepped,
            device=device,
            tau=float(config.violation_tau),
        )
        train_weighted_objective = _eval_objective_terms(
            model,
            train_prepped,
            device=device,
            leaf_weight=float(objective["local_law_c1_weight"]),
            c2_weight=float(objective["local_law_c2_weight"]),
            c3_weight=float(objective["local_law_c3_weight"]),
            root_weight=float(objective["optimization_root_weight"]),
            schedule_consistency_weight=float(objective["proxy_schedule_consistency_weight"]),
            include_root_query=bool(config.include_root_query),
        )
        test_weighted_objective = _eval_objective_terms(
            model,
            test_prepped,
            device=device,
            leaf_weight=float(objective["local_law_c1_weight"]),
            c2_weight=float(objective["local_law_c2_weight"]),
            c3_weight=float(objective["local_law_c3_weight"]),
            root_weight=float(objective["optimization_root_weight"]),
            schedule_consistency_weight=float(objective["proxy_schedule_consistency_weight"]),
            include_root_query=bool(config.include_root_query),
        )
        val_weighted_objective = _eval_objective_terms(
            model,
            val_prepped,
            device=device,
            leaf_weight=float(objective["local_law_c1_weight"]),
            c2_weight=float(objective["local_law_c2_weight"]),
            c3_weight=float(objective["local_law_c3_weight"]),
            root_weight=float(objective["optimization_root_weight"]),
            schedule_consistency_weight=float(objective["proxy_schedule_consistency_weight"]),
            include_root_query=bool(config.include_root_query),
        )
        train_objective_estimators = _markov_objective_estimator_payload(
            model,
            train_prepped,
            device=device,
            objective_summary=objective,
            exact_objective=train_weighted_objective,
            leaf_query_rate=float(config.leaf_query_rate),
            audit_policy=str(config.audit_policy),
            audit_fixed_nodes=int(config.audit_fixed_nodes),
            audit_fraction=float(config.audit_fraction),
            audit_scale=float(config.audit_scale),
            c3_audit_strategy=str(config.c3_audit_strategy),
            c3_include_root=bool(config.c3_include_root),
            seed=int(seeds["effective_model_seed"]) + 1_003,
        )
        val_objective_estimators = _markov_objective_estimator_payload(
            model,
            val_prepped,
            device=device,
            objective_summary=objective,
            exact_objective=val_weighted_objective,
            leaf_query_rate=float(config.leaf_query_rate),
            audit_policy=str(config.audit_policy),
            audit_fixed_nodes=int(config.audit_fixed_nodes),
            audit_fraction=float(config.audit_fraction),
            audit_scale=float(config.audit_scale),
            c3_audit_strategy=str(config.c3_audit_strategy),
            c3_include_root=bool(config.c3_include_root),
            seed=int(seeds["effective_model_seed"]) + 2_003,
        )
        test_objective_estimators = _markov_objective_estimator_payload(
            model,
            test_prepped,
            device=device,
            objective_summary=objective,
            exact_objective=test_weighted_objective,
            leaf_query_rate=float(config.leaf_query_rate),
            audit_policy=str(config.audit_policy),
            audit_fixed_nodes=int(config.audit_fixed_nodes),
            audit_fraction=float(config.audit_fraction),
            audit_scale=float(config.audit_scale),
            c3_audit_strategy=str(config.c3_audit_strategy),
            c3_include_root=bool(config.c3_include_root),
            seed=int(seeds["effective_model_seed"]) + 3_003,
        )
    else:
        train_loss_final = TrainFitDiagnostics(
            train_loss_final=float("nan"),
            train_loss_curve=tuple(),
            epochs_completed=0,
            selection_metric_curve=tuple(),
            selection_mode="not_trained",
            selection_split="config",
            selection_metric_name="not_trained",
            selection_metric_value=float("nan"),
            best_epoch=0,
        )
        learned_train = _zero_sketch_metrics(n_docs=int(len(train_prepped)))
        learned_val = _zero_sketch_metrics(n_docs=int(len(val_prepped)))
        learned = _zero_sketch_metrics(n_docs=int(len(test_prepped)))
        train_weighted_objective = ObjectiveMetrics(
            optimization_total_loss=float("nan"),
            optimization_root_loss=float("nan"),
            optimization_leaf_loss=float("nan"),
            optimization_c2_loss=float("nan"),
            optimization_merge_loss=float("nan"),
            optimization_schedule_consistency_loss=float("nan"),
            raw_total_loss=float("nan"),
            raw_root_loss=float("nan"),
            raw_leaf_loss=float("nan"),
            raw_c2_loss=float("nan"),
            raw_merge_loss=float("nan"),
            raw_schedule_consistency_loss=float("nan"),
            n_docs=int(len(train_prepped)),
        )
        val_weighted_objective = ObjectiveMetrics(
            optimization_total_loss=float("nan"),
            optimization_root_loss=float("nan"),
            optimization_leaf_loss=float("nan"),
            optimization_c2_loss=float("nan"),
            optimization_merge_loss=float("nan"),
            optimization_schedule_consistency_loss=float("nan"),
            raw_total_loss=float("nan"),
            raw_root_loss=float("nan"),
            raw_leaf_loss=float("nan"),
            raw_c2_loss=float("nan"),
            raw_merge_loss=float("nan"),
            raw_schedule_consistency_loss=float("nan"),
            n_docs=int(len(val_prepped)),
        )
        train_objective_estimators = {}
        val_objective_estimators = {}
        test_objective_estimators = {}
        test_weighted_objective = ObjectiveMetrics(
            optimization_total_loss=float("nan"),
            optimization_root_loss=float("nan"),
            optimization_leaf_loss=float("nan"),
            optimization_c2_loss=float("nan"),
            optimization_merge_loss=float("nan"),
            optimization_schedule_consistency_loss=float("nan"),
            raw_total_loss=float("nan"),
            raw_root_loss=float("nan"),
            raw_leaf_loss=float("nan"),
            raw_c2_loss=float("nan"),
            raw_merge_loss=float("nan"),
            raw_schedule_consistency_loss=float("nan"),
            n_docs=int(len(test_prepped)),
        )

    rf_root: Optional[SketchMetrics] = None
    rf_root_val: Optional[SketchMetrics] = None
    if bool(config.include_rf_root_baseline):
        rf_root_val = _eval_rf_root_baseline(
            train_prepped,
            val_prepped,
            seed=int(seeds["effective_model_seed"]),
            n_estimators=int(config.rf_n_estimators),
            max_depth=int(config.rf_max_depth),
            min_samples_leaf=int(config.rf_min_samples_leaf),
        )
        rf_root = _eval_rf_root_baseline(
            train_prepped,
            test_prepped,
            seed=int(seeds["effective_model_seed"]),
            n_estimators=int(config.rf_n_estimators),
            max_depth=int(config.rf_max_depth),
            min_samples_leaf=int(config.rf_min_samples_leaf),
        )

    geom = _training_geometry(
        train_prepped,
        policy=str(config.audit_policy),
        fixed_nodes=int(config.audit_fixed_nodes),
        fraction=float(config.audit_fraction),
        scale=float(config.audit_scale),
        leaf_query_rate=float(config.leaf_query_rate),
        include_root_query=bool(config.include_root_query),
    )

    # Selection-bias demo: estimate mean merge discrepancy under risk-biased sampling.
    #
    # We treat the learned model's per-merge absolute error on a fixed test set as a population.
    # We then sample merge nodes with non-uniform inclusion probabilities proportional to a
    # simple "risk score" (span size), and compare:
    #   - naive mean of sampled errors,
    #   - IPW mean,
    #   - DSL/AIPW mean using a crude learned proxy.
    #
    # This is intentionally simple: the point is to show bias under adaptive sampling and
    # how IPW/DSL correct it, while the magnitude of the population itself improves with
    # more training docs / more oracle labels.
    internal_label_rate = (
        float(geom.mean_internal_labels) / float(geom.mean_internal_nodes)
        if float(geom.mean_internal_nodes) > 0
        else 0.0
    )
    base = float(min(1.0, max(0.02, internal_label_rate)))
    pi_min = float(min(0.02, 0.10 * base))

    diag_errs: List[float] = []
    diag_scores: List[float] = []
    diag_preds: List[float] = []
    if train_prepped:
        assert "model" in locals()
        model.eval()
        with torch.no_grad():
            for doc in test_prepped[: min(100, len(test_prepped))]:
                leaf_feats = _to_device(doc.leaf_features, device=device)
                states = [model.encode_leaf(x) for x in leaf_feats]
                _root, merge_states = model._merge_states(
                    states,
                    schedule="balanced",
                    collect_merge_states=True,
                )

                # Track merge span sizes (in leaves) in the same order as `merge_states`.
                cur_sizes = [1 for _ in range(len(states))]
                merge_sizes: List[int] = []
                while len(cur_sizes) > 1:
                    nxt_sizes: List[int] = []
                    i = 0
                    while i < len(cur_sizes):
                        if i + 1 >= len(cur_sizes):
                            nxt_sizes.append(int(cur_sizes[i]))
                            i += 1
                            continue
                        parent = int(cur_sizes[i]) + int(cur_sizes[i + 1])
                        merge_sizes.append(int(parent))
                        nxt_sizes.append(int(parent))
                        i += 2
                    cur_sizes = nxt_sizes

                for idx, st in enumerate(merge_states):
                    if idx >= len(doc.merge_counts_balanced) or idx >= len(merge_sizes):
                        break
                    pred = float(model.predict_count_from_state(st).detach().cpu().numpy())
                    truth = float(doc.merge_counts_balanced[idx])
                    diag_errs.append(abs(pred - truth))
                    diag_scores.append(float(merge_sizes[idx]))
                    diag_preds.append(float(pred))

    errs = np.asarray(diag_errs, dtype=np.float64)
    scores = np.asarray(diag_scores, dtype=np.float64)
    preds = np.asarray(diag_preds, dtype=np.float64)
    if errs.size > 0:
        scaled = base * (scores / float(np.mean(scores)))
        pi = np.clip(scaled, pi_min, 1.0)
        pred_mu = float(np.mean(preds))
        pred_err_proxy = np.abs(preds - pred_mu)
        diagnostics = _audit_estimator_diagnostics(
            errs.tolist(),
            pred_err_proxy.tolist(),
            pi.tolist(),
            trials=400,
            seed=int(seeds["effective_model_seed"]) + 991,
        )
    else:
        pi = np.full((0,), 1.0, dtype=np.float64)
        diagnostics = EstimatorDiagnostics(
            true_mean=0.0,
            naive_bias=0.0,
            ipw_bias=0.0,
            dsl_bias=0.0,
            ipw_var=0.0,
            dsl_var=0.0,
        )
    guided_eval_curve: Dict[str, object] = {
        "present": False,
        "include_root": bool(config.eval_guidance_include_root),
        "trials": int(config.eval_guidance_trials),
        "points": [],
    }
    if train_prepped and int(config.eval_guidance_trials) > 0 and tuple(config.eval_guidance_qs):
        assert "model" in locals()
        guided_eval_curve = _eval_guided_model_curve(
            model,
            test_prepped,
            device=device,
            tau=float(config.violation_tau),
            guidance_qs=tuple(config.eval_guidance_qs),
            guidance_trials=int(config.eval_guidance_trials),
            guidance_include_root=bool(config.eval_guidance_include_root),
            guidance_override_mode=str(config.guidance_override_mode),
            guidance_seed=int(seeds["effective_model_seed"])
            + int(config.eval_guidance_seed_offset),
        )
        points = list(guided_eval_curve.get("points") or [])
        q0_pts = [p for p in points if abs(float(p.get("q", float("nan"))) - 0.0) <= 1e-12]
        if q0_pts:
            q0_root = float(q0_pts[0].get("root_mae", float("nan")))
            if math.isfinite(q0_root) and abs(float(q0_root) - float(learned.root_mae)) > 1e-12:
                raise ValueError(
                    "guided_eval_curve invariant failed: q=0 root_mae must match learned root_mae "
                    f"(got {q0_root} vs {learned.root_mae})"
                )
        if bool(config.eval_guidance_include_root):
            q1_pts = [p for p in points if abs(float(p.get("q", float("nan"))) - 1.0) <= 1e-12]
            if q1_pts:
                q1_root = float(q1_pts[0].get("root_mae", float("nan")))
                if (not math.isfinite(q1_root)) or q1_root > 1e-12:
                    raise ValueError(
                        "guided_eval_curve invariant failed: q=1 root_mae must be <=1e-12 "
                        f"when eval_guidance_include_root is true (got {q1_root})"
                    )

    learned_payload = {
        **asdict(learned),
        "train_loss_final": float(train_loss_final.train_loss_final),
        "train_loss_curve": [float(x) for x in train_loss_final.train_loss_curve],
        "epochs_completed": int(train_loss_final.epochs_completed),
        "training_selection_metric_curve": [
            float(x) for x in train_loss_final.selection_metric_curve
        ],
        "training_selection_mode": str(train_loss_final.selection_mode),
        "training_selection_split": str(train_loss_final.selection_split),
        "training_selection_metric_name": str(train_loss_final.selection_metric_name),
        "training_selection_metric_value": float(train_loss_final.selection_metric_value),
        "training_selection_best_epoch": int(train_loss_final.best_epoch),
        "train_root_mae": float(learned_train.root_mae),
        "train_leaf_mae": float(learned_train.leaf_mae),
        "train_c2_idempotence_mae": float(learned_train.c2_idempotence_mae),
        "train_c2_r1_mae": float(learned_train.c2_r1_mae),
        "train_c2_r2_mae": float(learned_train.c2_r2_mae),
        "train_c2_r4_mae": float(learned_train.c2_r4_mae),
        "train_merge_mae": float(learned_train.merge_mae),
        "train_schedule_spread_mean": float(learned_train.schedule_spread_mean),
        "train_resummary_root_drift_r1": float(learned_train.resummary_root_drift_r1),
        "train_resummary_root_drift_r2": float(learned_train.resummary_root_drift_r2),
        "train_resummary_root_drift_r4": float(learned_train.resummary_root_drift_r4),
        **_objective_with_split_prefix(train_weighted_objective, prefix="train"),
        **_objective_estimator_with_split_prefix(train_objective_estimators, prefix="train"),
        "train_theorem_score": float(
            markov_theorem_score(
                leaf=float(learned_train.leaf_mae),
                merge=float(learned_train.merge_mae),
                spread=float(learned_train.schedule_spread_mean),
            )
        ),
        "train_theorem_bundle_score_n": float(
            markov_law_bundle_score(
                c1=float(learned_train.leaf_mae) / float(target_scale),
                c2=float(learned_train.c2_idempotence_mae) / float(target_scale),
                c3=float(learned_train.merge_mae) / float(target_scale),
            )
        ),
        **_metrics_with_split_prefix(learned_val, prefix="val", target_scale=target_scale),
        **_metrics_with_split_prefix(learned, prefix="test", target_scale=target_scale),
        **_objective_with_split_prefix(val_weighted_objective, prefix="val"),
        **_objective_with_split_prefix(test_weighted_objective, prefix="test"),
        **_objective_estimator_with_split_prefix(val_objective_estimators, prefix="val"),
        **_objective_estimator_with_split_prefix(test_objective_estimators, prefix="test"),
        "val_theorem_score": float(
            markov_theorem_score(
                leaf=float(learned_val.leaf_mae),
                merge=float(learned_val.merge_mae),
                spread=float(learned_val.schedule_spread_mean),
            )
        ),
        "test_theorem_score": float(
            markov_theorem_score(
                leaf=float(learned.leaf_mae),
                merge=float(learned.merge_mae),
                spread=float(learned.schedule_spread_mean),
            )
        ),
        "val_theorem_bundle_score_n": float(
            markov_law_bundle_score(
                c1=float(learned_val.leaf_mae) / float(target_scale),
                c2=float(learned_val.c2_idempotence_mae) / float(target_scale),
                c3=float(learned_val.merge_mae) / float(target_scale),
            )
        ),
        "test_theorem_bundle_score_n": float(
            markov_law_bundle_score(
                c1=float(learned.leaf_mae) / float(target_scale),
                c2=float(learned.c2_idempotence_mae) / float(target_scale),
                c3=float(learned.merge_mae) / float(target_scale),
            )
        ),
        "generalization_gap_optimization_objective_full_labels": float(
            test_weighted_objective.optimization_total_loss
            - train_weighted_objective.optimization_total_loss
        ),
        "generalization_gap_unweighted_objective_full_labels": float(
            test_weighted_objective.raw_total_loss - train_weighted_objective.raw_total_loss
        ),
        "generalization_gap_objective_full_labels": float(
            test_weighted_objective.optimization_total_loss
            - train_weighted_objective.optimization_total_loss
        ),
        "generalization_gap_root_mae": float(learned.root_mae - learned_train.root_mae),
        "generalization_gap_leaf_mae": float(learned.leaf_mae - learned_train.leaf_mae),
        "generalization_gap_c2_idempotence_mae": float(
            learned.c2_idempotence_mae - learned_train.c2_idempotence_mae
        ),
        "generalization_gap_merge_mae": float(learned.merge_mae - learned_train.merge_mae),
        "generalization_gap_schedule_spread_mean": float(
            learned.schedule_spread_mean - learned_train.schedule_spread_mean
        ),
        "generalization_gap_resummary_root_drift_r2": float(
            learned.resummary_root_drift_r2 - learned_train.resummary_root_drift_r2
        ),
        "gap_to_exact_root_mae": float(learned.root_mae - exact.root_mae),
        "gap_to_exact_leaf_mae": float(learned.leaf_mae - exact.leaf_mae),
        "gap_to_exact_c2_idempotence_mae": float(
            learned.c2_idempotence_mae - exact.c2_idempotence_mae
        ),
        "gap_to_exact_merge_mae": float(learned.merge_mae - exact.merge_mae),
        "gap_to_exact_schedule_spread_mean": float(
            learned.schedule_spread_mean - exact.schedule_spread_mean
        ),
        "gap_to_undersupported_root_mae": float(learned.root_mae - undersupported.root_mae),
        "gap_to_undersupported_leaf_mae": float(learned.leaf_mae - undersupported.leaf_mae),
        "gap_to_undersupported_c2_idempotence_mae": float(
            learned.c2_idempotence_mae - undersupported.c2_idempotence_mae
        ),
        "gap_to_undersupported_merge_mae": float(learned.merge_mae - undersupported.merge_mae),
        "gap_to_undersupported_schedule_spread_mean": float(
            learned.schedule_spread_mean - undersupported.schedule_spread_mean
        ),
    }
    metrics: Dict[str, object] = {
        "exact": asdict(exact),
        "leaf_bucket": asdict(leaf_bucket),
        "undersupported": asdict(undersupported),
        "flip_R1": asdict(flip_r1),
        "flip_R2": asdict(flip_r2),
        "learned_train": asdict(learned_train),
        "learned_val": asdict(learned_val),
        "learned_test": learned_payload,
        "learned": learned_payload,
        "guided_eval_curve": guided_eval_curve,
    }
    if rf_root is not None:
        metrics["rf_root"] = asdict(rf_root)
    if rf_root_val is not None:
        metrics["rf_root_val"] = asdict(rf_root_val)

    # Keep the old undersupported comparison as a diagnostic only.
    # Canonical cross-DGP law-stress now pairs learned packages with the matched
    # `root_only` baseline across runs in the unified reporting layer.
    from src.ctreepo.sim.core.law_stress_common import classify_law_stress as _classify_law_stress

    metrics["diagnostic_law_stress_vs_undersupported"] = _classify_law_stress(
        baseline_c1=float(undersupported.leaf_mae),
        baseline_c2=float(undersupported.c2_idempotence_mae),
        baseline_c3=float(undersupported.merge_mae),
        baseline_spread=float(undersupported.schedule_spread_mean),
        baseline_root_mae=float(undersupported.root_mae),
        selected_c1=float(learned.leaf_mae),
        selected_c2=float(learned.c2_idempotence_mae),
        selected_c3=float(learned.merge_mae),
        selected_spread=float(learned.schedule_spread_mean),
        selected_root_mae=float(learned.root_mae),
    ).to_dict()

    # Also expose a compact IPW mean check using TreeSample to tie to the repo's IPW tooling.
    # (We estimate the mean merge-violation rate at threshold tau for the learned-sketch merge error population.)
    tau = float(config.violation_tau)
    if errs.size > 0:
        violations = (errs > tau).astype(np.int64)
        samples: List[TreeSample] = []
        for idx, (v, p) in enumerate(zip(violations.tolist(), pi.tolist())):
            if random.random() < float(p):
                samples.append(
                    TreeSample(
                        doc_id="pop",
                        node_id=str(idx),
                        node_type=NodeType.MERGE,
                        violation=int(v),
                        propensity=TreePropensity(node=float(p)),
                    )
                )
        ipw_violation_rate = float(
            horvitz_thompson_mean(samples, lambda s: float(s.violation), float(len(violations)))
            if violations.size
            else 0.0
        )
        metrics["ipw_violation_rate_demo"] = {
            "population_source": "learned_merge_error",
            "tau": float(tau),
            "population": int(len(violations)),
            "sampled": int(len(samples)),
            "ipw_mean_violation": float(ipw_violation_rate),
            "true_mean_violation": float(np.mean(violations.astype(np.float64))),
        }

    current_name = str(config.law_package).strip() or (
        "root_only" if float(objective.get("local_law_weight", 0.0)) <= 1e-12 else "learned_g"
    )
    current_role = (
        PolicyRole.BASELINE_G.value
        if str(current_name) == "root_only"
        else PolicyRole.LEARNED_G.value
    )
    if int(config.val_docs) > 0:
        current_selection_metric_name = str(
            learned_payload.get(
                "val_objective_selection_metric_name",
                "configured_objective",
            )
        )
        current_selection_metric_value = float(
            learned_payload.get(
                "val_objective_selection_metric_value",
                learned_payload.get("val_objective_full_labels", float("nan")),
            )
        )
    else:
        current_selection_metric_name = str(train_loss_final.selection_metric_name)
        current_selection_metric_value = float(train_loss_final.selection_metric_value)
    local_law_learnability, g_artifacts = _build_markov_local_law_learnability(
        config=config,
        seeds=seeds,
        target_scale=float(target_scale),
        objective_summary=objective,
        geom=geom,
        exact=exact,
        leaf_bucket=leaf_bucket,
        undersupported=undersupported,
        flip_r2=flip_r2,
        current_name=str(current_name),
        current_role=str(current_role),
        current_train=learned_train,
        current_val=learned_val,
        current_test=learned,
        current_selection_metric_name=str(current_selection_metric_name),
        current_selection_metric=float(current_selection_metric_value),
        current_train_payload=learned_payload,
        current_val_payload=learned_payload,
        current_test_payload=learned_payload,
        model=(model if train_prepped else None),
    )

    return OPSCountSummary(
        config=config_payload,
        training_geometry=asdict(geom),
        objective=objective,
        metrics=metrics,
        estimator_diagnostics={
            **asdict(diagnostics),
            "selection_demo_base_rate": float(base),
            "selection_demo_pi_min": float(pi_min),
            "selection_demo_n_units": float(errs.size),
        },
        local_law_learnability=local_law_learnability,
        g_artifacts=g_artifacts,
    )


__all__ = [
    "OPSCountConfig",
    "OPSCountSummary",
    "VALID_AUDIT_POLICIES",
    "VALID_C3_AUDIT_STRATEGIES",
    "VALID_EXACT_FAMILIES",
    "VALID_LAW_PACKAGES",
    "VALID_SCHEDULES",
    "audit_sample_count",
    "leaf_sample_count",
    "_eval_leaf_bucket_family",
    "run_markov_changepoint_ops_count_experiment",
]
