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

from dataclasses import asdict, dataclass
import json
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

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

from src.tree.ipw import NodeType, TreePropensity, TreeSample, horvitz_thompson_mean
from src.tree.markov_boundary_honesty_simulation import _make_transition_matrices
from src.tree.markov_changepoint_honesty_simulation import (
    ChangepointMarkovDoc,
    MarkovChangepointConfig as _GeneratorConfig,
    generate_changepoint_docs,
)


ScheduleName = str
VALID_SCHEDULES: Tuple[ScheduleName, ...] = ("balanced", "left_to_right", "right_to_left")
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
    test_docs: int = 1000

    # Learned sketch settings.
    feature_mode: str = "full"  # "full" includes endpoints; "no_endpoints" drops them.
    state_dim: int = 32
    hidden_dim: int = 128
    n_epochs: int = 10
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-5
    c3_weight: float = 0.20
    leaf_weight: float = 0.05
    root_weight: float = 1.0
    schedule_consistency_weight: float = 0.0
    grad_clip_norm: float = 1.0

    # Node-label budgets (oracle queries).
    audit_policy: AuditPolicyName = "fraction"
    audit_fixed_nodes: int = 0
    audit_fraction: float = 0.2
    audit_scale: float = 1.0
    c3_audit_strategy: C3AuditStrategyName = "uniform"
    c3_include_root: bool = True
    leaf_query_rate: float = 1.0
    include_root_query: bool = True

    # Evaluation / audit thresholds.
    violation_tau: float = 0.0

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
class OPSCountSummary:
    config: Dict[str, object]
    training_geometry: Dict[str, float | int]
    metrics: Dict[str, Dict[str, float | int]]
    estimator_diagnostics: Dict[str, float]

    def to_json(self) -> str:
        payload = {
            "config": self.config,
            "training_geometry": self.training_geometry,
            "metrics": self.metrics,
            "estimator_diagnostics": self.estimator_diagnostics,
        }
        return json.dumps(payload, indent=2, sort_keys=True)


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
    base = _exact_merge(_ExactState(a.count, a.first, a.last), _ExactState(b.count, b.first, b.last))
    return _FlipState(count=base.count, first=base.first, last=base.last, flipped=False)


def _flip_resummary(z: _FlipState) -> _FlipState:
    return _FlipState(count=int(z.count), first=int(z.first), last=int(z.last), flipped=not bool(z.flipped))


def _flip_value(z: _FlipState) -> int:
    return int(z.count) + (1 if bool(z.flipped) else 0)


def _eval_exact_family(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    leaf_tokens: int,
    tau: float,
) -> SketchMetrics:
    if len(docs) == 0:
        return SketchMetrics(
            root_mae=0.0,
            root_median_abs_error=0.0,
            root_p95_abs_error=0.0,
            schedule_spread_mean=0.0,
            schedule_spread_p95=0.0,
            leaf_mae=0.0,
            leaf_violation_rate=0.0,
            merge_mae=0.0,
            merge_violation_rate=0.0,
            n_docs=0,
        )

    root_abs: List[float] = []
    spreads: List[float] = []
    leaf_abs: List[float] = []
    merge_abs: List[float] = []

    for doc in docs:
        n_tok = int(len(doc.token_regimes))
        spans = _leaf_spans(n_tok, leaf_tokens=int(leaf_tokens))
        leaf_states = [_exact_from_span(doc, sp) for sp in spans]
        leaf_truth = [_oracle_count(doc, start=sp[0], end=sp[1]) for sp in spans]
        for st, truth in zip(leaf_states, leaf_truth):
            leaf_abs.append(abs(float(st.count) - float(truth)))

        # Root predictions for schedule spread.
        roots: Dict[str, float] = {}
        for sched in VALID_SCHEDULES:
            if str(sched) == "balanced":
                cur_s = list(leaf_states)
                cur_p = list(spans)
                while len(cur_s) > 1:
                    nxt_s: List[_ExactState] = []
                    nxt_p: List[Tuple[int, int]] = []
                    i = 0
                    while i < len(cur_s):
                        if i + 1 >= len(cur_s):
                            nxt_s.append(cur_s[i])
                            nxt_p.append(cur_p[i])
                            i += 1
                            continue
                        merged = _exact_merge(cur_s[i], cur_s[i + 1])
                        parent = (int(cur_p[i][0]), int(cur_p[i + 1][1]))
                        nxt_s.append(merged)
                        nxt_p.append(parent)
                        i += 2
                    cur_s, cur_p = nxt_s, nxt_p
                roots[str(sched)] = float(cur_s[0].count)
            elif str(sched) == "left_to_right":
                acc = leaf_states[0]
                for st in leaf_states[1:]:
                    acc = _exact_merge(acc, st)
                roots[str(sched)] = float(acc.count)
            elif str(sched) == "right_to_left":
                acc = leaf_states[-1]
                for st in reversed(leaf_states[:-1]):
                    acc = _exact_merge(st, acc)
                roots[str(sched)] = float(acc.count)
            else:
                raise ValueError(f"unsupported schedule: {sched!r}")

        truth_root = float(_oracle_count(doc, start=0, end=n_tok))
        pred = roots["balanced"]
        root_abs.append(abs(pred - truth_root))
        spreads.append(max(roots.values()) - min(roots.values()))

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
                merged = _exact_merge(cur_s[i], cur_s[i + 1])
                parent = (int(cur_p[i][0]), int(cur_p[i + 1][1]))
                truth_parent = float(_oracle_count(doc, start=parent[0], end=parent[1]))
                merge_abs.append(abs(float(merged.count) - truth_parent))
                nxt_s.append(merged)
                nxt_p.append(parent)
                i += 2
            cur_s, cur_p = nxt_s, nxt_p

    leaf_abs_arr = np.asarray(leaf_abs, dtype=np.float64)
    merge_abs_arr = np.asarray(merge_abs, dtype=np.float64)
    root_abs_arr = np.asarray(root_abs, dtype=np.float64)
    spreads_arr = np.asarray(spreads, dtype=np.float64)

    tau = float(tau)
    return SketchMetrics(
        root_mae=float(np.mean(root_abs_arr)),
        root_median_abs_error=float(np.median(root_abs_arr)),
        root_p95_abs_error=float(np.percentile(root_abs_arr, 95.0)),
        schedule_spread_mean=float(np.mean(spreads_arr)),
        schedule_spread_p95=float(np.percentile(spreads_arr, 95.0)),
        leaf_mae=float(np.mean(leaf_abs_arr)) if leaf_abs_arr.size else 0.0,
        leaf_violation_rate=float(np.mean((leaf_abs_arr > tau).astype(np.float64)))
        if leaf_abs_arr.size
        else 0.0,
        merge_mae=float(np.mean(merge_abs_arr)) if merge_abs_arr.size else 0.0,
        merge_violation_rate=float(np.mean((merge_abs_arr > tau).astype(np.float64)))
        if merge_abs_arr.size
        else 0.0,
        n_docs=int(len(docs)),
    )


def _eval_count_only_family(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    leaf_tokens: int,
    tau: float,
) -> SketchMetrics:
    if len(docs) == 0:
        return SketchMetrics(
            root_mae=0.0,
            root_median_abs_error=0.0,
            root_p95_abs_error=0.0,
            schedule_spread_mean=0.0,
            schedule_spread_p95=0.0,
            leaf_mae=0.0,
            leaf_violation_rate=0.0,
            merge_mae=0.0,
            merge_violation_rate=0.0,
            n_docs=0,
        )

    root_abs: List[float] = []
    spreads: List[float] = []
    leaf_abs: List[float] = []
    merge_abs: List[float] = []

    for doc in docs:
        n_tok = int(len(doc.token_regimes))
        spans = _leaf_spans(n_tok, leaf_tokens=int(leaf_tokens))
        leaf_states = [_count_only_from_span(doc, sp) for sp in spans]
        leaf_truth = [_oracle_count(doc, start=sp[0], end=sp[1]) for sp in spans]
        for st, truth in zip(leaf_states, leaf_truth):
            leaf_abs.append(abs(float(st.count) - float(truth)))

        roots: Dict[str, float] = {}
        for sched in VALID_SCHEDULES:
            if str(sched) == "balanced":
                cur_s = list(leaf_states)
                cur_p = list(spans)
                while len(cur_s) > 1:
                    nxt_s: List[_CountOnlyState] = []
                    nxt_p: List[Tuple[int, int]] = []
                    i = 0
                    while i < len(cur_s):
                        if i + 1 >= len(cur_s):
                            nxt_s.append(cur_s[i])
                            nxt_p.append(cur_p[i])
                            i += 1
                            continue
                        merged = _count_only_merge(cur_s[i], cur_s[i + 1])
                        parent = (int(cur_p[i][0]), int(cur_p[i + 1][1]))
                        nxt_s.append(merged)
                        nxt_p.append(parent)
                        i += 2
                    cur_s, cur_p = nxt_s, nxt_p
                roots[str(sched)] = float(cur_s[0].count)
            elif str(sched) == "left_to_right":
                acc = leaf_states[0]
                for st in leaf_states[1:]:
                    acc = _count_only_merge(acc, st)
                roots[str(sched)] = float(acc.count)
            elif str(sched) == "right_to_left":
                acc = leaf_states[-1]
                for st in reversed(leaf_states[:-1]):
                    acc = _count_only_merge(st, acc)
                roots[str(sched)] = float(acc.count)
            else:
                raise ValueError(f"unsupported schedule: {sched!r}")

        truth_root = float(_oracle_count(doc, start=0, end=n_tok))
        pred = roots["balanced"]
        root_abs.append(abs(pred - truth_root))
        spreads.append(max(roots.values()) - min(roots.values()))

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
                merged = _count_only_merge(cur_s[i], cur_s[i + 1])
                parent = (int(cur_p[i][0]), int(cur_p[i + 1][1]))
                truth_parent = float(_oracle_count(doc, start=parent[0], end=parent[1]))
                merge_abs.append(abs(float(merged.count) - truth_parent))
                nxt_s.append(merged)
                nxt_p.append(parent)
                i += 2
            cur_s, cur_p = nxt_s, nxt_p

    leaf_abs_arr = np.asarray(leaf_abs, dtype=np.float64)
    merge_abs_arr = np.asarray(merge_abs, dtype=np.float64)
    root_abs_arr = np.asarray(root_abs, dtype=np.float64)
    spreads_arr = np.asarray(spreads, dtype=np.float64)

    tau = float(tau)
    return SketchMetrics(
        root_mae=float(np.mean(root_abs_arr)),
        root_median_abs_error=float(np.median(root_abs_arr)),
        root_p95_abs_error=float(np.percentile(root_abs_arr, 95.0)),
        schedule_spread_mean=float(np.mean(spreads_arr)),
        schedule_spread_p95=float(np.percentile(spreads_arr, 95.0)),
        leaf_mae=float(np.mean(leaf_abs_arr)) if leaf_abs_arr.size else 0.0,
        leaf_violation_rate=float(np.mean((leaf_abs_arr > tau).astype(np.float64)))
        if leaf_abs_arr.size
        else 0.0,
        merge_mae=float(np.mean(merge_abs_arr)) if merge_abs_arr.size else 0.0,
        merge_violation_rate=float(np.mean((merge_abs_arr > tau).astype(np.float64)))
        if merge_abs_arr.size
        else 0.0,
        n_docs=int(len(docs)),
    )


def _eval_flip_family(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    leaf_tokens: int,
    tau: float,
    rounds: int,
) -> SketchMetrics:
    if len(docs) == 0:
        return SketchMetrics(
            root_mae=0.0,
            root_median_abs_error=0.0,
            root_p95_abs_error=0.0,
            schedule_spread_mean=0.0,
            schedule_spread_p95=0.0,
            leaf_mae=0.0,
            leaf_violation_rate=0.0,
            merge_mae=0.0,
            merge_violation_rate=0.0,
            n_docs=0,
        )

    R = int(max(1, rounds))
    root_abs: List[float] = []
    spreads: List[float] = []
    leaf_abs: List[float] = []
    merge_abs: List[float] = []

    for doc in docs:
        n_tok = int(len(doc.token_regimes))
        spans = _leaf_spans(n_tok, leaf_tokens=int(leaf_tokens))
        leaf_states = [_flip_from_span(doc, sp) for sp in spans]
        leaf_truth = [_oracle_count(doc, start=sp[0], end=sp[1]) for sp in spans]
        for st, truth in zip(leaf_states, leaf_truth):
            leaf_abs.append(abs(float(_flip_value(st)) - float(truth)))

        roots: Dict[str, float] = {}
        for sched in VALID_SCHEDULES:
            if str(sched) == "balanced":
                cur_s = list(leaf_states)
                cur_p = list(spans)
                while len(cur_s) > 1:
                    nxt_s: List[_FlipState] = []
                    nxt_p: List[Tuple[int, int]] = []
                    i = 0
                    while i < len(cur_s):
                        if i + 1 >= len(cur_s):
                            nxt_s.append(cur_s[i])
                            nxt_p.append(cur_p[i])
                            i += 1
                            continue
                        merged = _flip_merge(cur_s[i], cur_s[i + 1])
                        parent = (int(cur_p[i][0]), int(cur_p[i + 1][1]))
                        nxt_s.append(merged)
                        nxt_p.append(parent)
                        i += 2
                    cur_s, cur_p = nxt_s, nxt_p
                z = cur_s[0]
            elif str(sched) == "left_to_right":
                z = leaf_states[0]
                for st in leaf_states[1:]:
                    z = _flip_merge(z, st)
            elif str(sched) == "right_to_left":
                z = leaf_states[-1]
                for st in reversed(leaf_states[:-1]):
                    z = _flip_merge(st, z)
            else:
                raise ValueError(f"unsupported schedule: {sched!r}")

            for _ in range(R - 1):
                z = _flip_resummary(z)
            roots[str(sched)] = float(_flip_value(z))

        truth_root = float(_oracle_count(doc, start=0, end=n_tok))
        pred = roots["balanced"]
        root_abs.append(abs(pred - truth_root))
        spreads.append(max(roots.values()) - min(roots.values()))

        # C3 discrepancies (balanced schedule only, first-pass only).
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
                merged = _flip_merge(cur_s[i], cur_s[i + 1])
                parent = (int(cur_p[i][0]), int(cur_p[i + 1][1]))
                truth_parent = float(_oracle_count(doc, start=parent[0], end=parent[1]))
                merge_abs.append(abs(float(_flip_value(merged)) - truth_parent))
                nxt_s.append(merged)
                nxt_p.append(parent)
                i += 2
            cur_s, cur_p = nxt_s, nxt_p

    leaf_abs_arr = np.asarray(leaf_abs, dtype=np.float64)
    merge_abs_arr = np.asarray(merge_abs, dtype=np.float64)
    root_abs_arr = np.asarray(root_abs, dtype=np.float64)
    spreads_arr = np.asarray(spreads, dtype=np.float64)

    tau = float(tau)
    return SketchMetrics(
        root_mae=float(np.mean(root_abs_arr)),
        root_median_abs_error=float(np.median(root_abs_arr)),
        root_p95_abs_error=float(np.percentile(root_abs_arr, 95.0)),
        schedule_spread_mean=float(np.mean(spreads_arr)),
        schedule_spread_p95=float(np.percentile(spreads_arr, 95.0)),
        leaf_mae=float(np.mean(leaf_abs_arr)) if leaf_abs_arr.size else 0.0,
        leaf_violation_rate=float(np.mean((leaf_abs_arr > tau).astype(np.float64)))
        if leaf_abs_arr.size
        else 0.0,
        merge_mae=float(np.mean(merge_abs_arr)) if merge_abs_arr.size else 0.0,
        merge_violation_rate=float(np.mean((merge_abs_arr > tau).astype(np.float64)))
        if merge_abs_arr.size
        else 0.0,
        n_docs=int(len(docs)),
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
    merge_counts_balanced: Tuple[float, ...]  # oracle counts for each realized merge (balanced order)
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
            _span_features(doc, sp, n_regimes=int(n_regimes), mode=str(feature_mode)) for sp in spans
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
            first = torch.zeros((*features.shape[:-1], n), device=features.device, dtype=features.dtype)
            last = torch.zeros((*features.shape[:-1], n), device=features.device, dtype=features.dtype)
            core = features

        h = self.encoder(core)
        return torch.cat([h, first, last], dim=-1)

    def predict_norm_from_state(self, state: torch.Tensor) -> torch.Tensor:
        h, _first, _last = self._split_state(state)
        logit = self.readout(h)
        return torch.sigmoid(logit).squeeze(-1)

    def predict_count_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.predict_norm_from_state(state) * float(self.target_scale)

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
                merged_h = self.merger(
                    torch.cat([left_h, right_h, left_last, right_first], dim=-1)
                )
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
            collect_merge_states=collect_c3 and str(schedule) == "balanced",
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


def _train_learned_model(
    model: LearnedCountSketch,
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
    include_root_query: bool,
    n_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    c3_weight: float,
    leaf_weight: float,
    root_weight: float,
    schedule_consistency_weight: float,
    grad_clip_norm: float,
    seed: int,
) -> float:
    if len(train_docs) == 0:
        return 0.0
    rng = random.Random(int(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    target_scale = float(model.target_scale)

    idxs = list(range(len(train_docs)))
    train_loss_final = float("nan")
    for _ in range(int(max(1, n_epochs))):
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
                    root_loss = float(root_weight) * F.mse_loss(pred_norm, true_norm, reduction="mean")
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
                    + float(leaf_weight) * out["leaf_loss"]
                    + float(schedule_consistency_weight) * consistency_loss
                )
                batch_loss = batch_loss + doc_loss
            batch_loss = batch_loss / float(len(batch_idx))
            batch_loss.backward()
            if float(grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            opt.step()
            batch_losses.append(float(batch_loss.detach().cpu()))
        train_loss_final = float(np.mean(np.asarray(batch_losses, dtype=np.float64)))
    return float(train_loss_final)


@torch.no_grad()
def _eval_learned_model(
    model: LearnedCountSketch,
    docs: Sequence[_CountDoc],
    *,
    device: torch.device,
    tau: float,
) -> SketchMetrics:
    if len(docs) == 0:
        return SketchMetrics(
            root_mae=0.0,
            root_median_abs_error=0.0,
            root_p95_abs_error=0.0,
            schedule_spread_mean=0.0,
            schedule_spread_p95=0.0,
            leaf_mae=0.0,
            leaf_violation_rate=0.0,
            merge_mae=0.0,
            merge_violation_rate=0.0,
            n_docs=0,
        )

    model.eval()
    root_abs: List[float] = []
    spreads: List[float] = []
    leaf_abs: List[float] = []
    merge_abs: List[float] = []

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

        # Root distortion + schedule spread.
        roots: Dict[str, float] = {}
        for sched in VALID_SCHEDULES:
            root_state, _ = model._merge_states(states, schedule=sched, collect_merge_states=False)
            roots[str(sched)] = float(model.predict_count_from_state(root_state).detach().cpu())
        pred_root = roots["balanced"]
        root_abs.append(abs(pred_root - float(doc.root_count)))
        spreads.append(max(roots.values()) - min(roots.values()))

    tau = float(tau)
    leaf_abs_arr = np.asarray(leaf_abs, dtype=np.float64)
    merge_abs_arr = np.asarray(merge_abs, dtype=np.float64)
    root_abs_arr = np.asarray(root_abs, dtype=np.float64)
    spreads_arr = np.asarray(spreads, dtype=np.float64)

    return SketchMetrics(
        root_mae=float(np.mean(root_abs_arr)),
        root_median_abs_error=float(np.median(root_abs_arr)),
        root_p95_abs_error=float(np.percentile(root_abs_arr, 95.0)),
        schedule_spread_mean=float(np.mean(spreads_arr)),
        schedule_spread_p95=float(np.percentile(spreads_arr, 95.0)),
        leaf_mae=float(np.mean(leaf_abs_arr)) if leaf_abs_arr.size else 0.0,
        leaf_violation_rate=float(np.mean((leaf_abs_arr > tau).astype(np.float64)))
        if leaf_abs_arr.size
        else 0.0,
        merge_mae=float(np.mean(merge_abs_arr)) if merge_abs_arr.size else 0.0,
        merge_violation_rate=float(np.mean((merge_abs_arr > tau).astype(np.float64)))
        if merge_abs_arr.size
        else 0.0,
        n_docs=int(len(docs)),
    )


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
    if int(config.train_docs) < 0 or int(config.test_docs) < 0:
        raise ValueError("train_docs/test_docs must be non-negative")
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
    if float(config.root_weight) < 0.0:
        raise ValueError("root_weight must be non-negative")
    if float(config.schedule_consistency_weight) < 0.0:
        raise ValueError("schedule_consistency_weight must be non-negative")

    _set_global_seed(int(config.seed))
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
        seed=int(config.seed),
        sinkhorn_iters=int(config.sinkhorn_iters),
        transition_log_std=float(config.transition_log_std),
        use_cuda=False,
    )
    rng = np.random.default_rng(int(config.seed))
    transitions = _make_transition_matrices(
        n_classes=int(config.n_regimes),
        vocab_size=int(config.vocab_size),
        log_std=float(config.transition_log_std),
        sinkhorn_iters=int(config.sinkhorn_iters),
        rng=rng,
    )
    # Generate train/test docs separately so that the evaluation set is stable across `train_docs`.
    # This makes learning curves much easier to interpret (and makes baselines constant in `train_docs`).
    gen_train = _GeneratorConfig(**{**asdict(gen_cfg), "train_docs": int(config.train_docs), "test_docs": 0})
    docs_train = generate_changepoint_docs(gen_train, transitions=transitions)
    gen_test = _GeneratorConfig(
        **{
            **asdict(gen_cfg),
            "train_docs": 0,
            "test_docs": int(config.test_docs),
            # Ensure the test set differs from the train set while remaining deterministic per run.
            "seed": int(config.seed) + 10_000,
        }
    )
    docs_test = generate_changepoint_docs(gen_test, transitions=transitions)

    # Deterministic baselines (no training).
    exact = _eval_exact_family(
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

    # Learned sketch.
    train_prepped = _prepare_count_docs(
        docs_train,
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
    if train_prepped:
        feature_dim = int(train_prepped[0].leaf_features[0].numel())
        # Oracle is a changepoint count in a piecewise-constant regime process with <= max_segments
        # segments, so the maximum possible root count is (max_segments - 1). Using max_tokens here
        # makes targets extremely small and can cause the learned model to collapse to predicting 0.
        target_scale = float(max(1, int(config.max_segments) - 1))
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
            c3_weight=float(config.c3_weight),
            leaf_weight=float(config.leaf_weight),
            root_weight=float(config.root_weight),
            schedule_consistency_weight=float(config.schedule_consistency_weight),
            grad_clip_norm=float(config.grad_clip_norm),
            seed=int(config.seed),
        )
        learned = _eval_learned_model(
            model,
            test_prepped,
            device=device,
            tau=float(config.violation_tau),
        )
    else:
        train_loss_final = float("nan")
        learned = SketchMetrics(
            root_mae=0.0,
            root_median_abs_error=0.0,
            root_p95_abs_error=0.0,
            schedule_spread_mean=0.0,
            schedule_spread_p95=0.0,
            leaf_mae=0.0,
            leaf_violation_rate=0.0,
            merge_mae=0.0,
            merge_violation_rate=0.0,
            n_docs=int(len(test_prepped)),
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
            seed=int(config.seed) + 991,
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
    metrics = {
        "exact": asdict(exact),
        "undersupported": asdict(undersupported),
        "flip_R1": asdict(flip_r1),
        "flip_R2": asdict(flip_r2),
        "learned": {**asdict(learned), "train_loss_final": float(train_loss_final)},
    }

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

    return OPSCountSummary(
        config=asdict(config),
        training_geometry=asdict(geom),
        metrics=metrics,
        estimator_diagnostics={
            **asdict(diagnostics),
            "selection_demo_base_rate": float(base),
            "selection_demo_pi_min": float(pi_min),
            "selection_demo_n_units": float(errs.size),
        },
    )


__all__ = [
    "OPSCountConfig",
    "OPSCountSummary",
    "VALID_AUDIT_POLICIES",
    "VALID_C3_AUDIT_STRATEGIES",
    "VALID_SCHEDULES",
    "audit_sample_count",
    "leaf_sample_count",
    "run_markov_changepoint_ops_count_experiment",
]
