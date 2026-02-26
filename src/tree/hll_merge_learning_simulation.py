"""
Learn a merge operator over HyperLogLog (HLL) register state.

Goal: make the "mergeable sketches" story explicit and theory-linked.

We treat HLL as the classical mergeable sketch with a known asymptotic relative
standard error (RSE) floor:

  RSE_theory(p) ~= 1.04 / sqrt(m),  with m = 2^p registers.

This module simulates learning a merge law M_theta over HLL registers using
only local, tree-node supervision (C3-style "joint encoding" checks):

  M_theta(S_left, S_right)  ≈  max(S_left, S_right)   (elementwise)

and evaluates how close learned merges get to the HLL performance curve.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for HLL merge-learning simulations. "
        "Install with: pip install torch>=2.0.0"
    ) from e

from src.tree.learned_sketch_simulation import (
    HLLConfig,
    HyperLogLogSketch,
    audit_sample_count,
    hll_relative_standard_error,
)


ScheduleName = str
VALID_SCHEDULES: Tuple[ScheduleName, ...] = ("balanced", "left_to_right", "right_to_left")

AuditPolicyName = str


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _safe_rel_error(pred: float, truth: float) -> float:
    denom = max(1.0, float(truth))
    return (float(pred) - float(truth)) / denom


def _parse_csv_floats(xs: Sequence[float]) -> Tuple[float, ...]:
    return tuple(float(x) for x in xs)


def _max_rho(hash_bits: int, precision: int) -> int:
    # remaining bits = hash_bits - precision; rho in [1, remaining_bits+1]
    return int(hash_bits) - int(precision) + 1


def _hll_memory_bits(precision: int, hash_bits: int = 64) -> int:
    sk = HyperLogLogSketch(HLLConfig(precision=int(precision), hash_bits=int(hash_bits)))
    return int(sk.memory_bits)


def _merge_schedule_registers_max(
    leaf_registers: Sequence[np.ndarray],
    *,
    schedule: ScheduleName,
) -> np.ndarray:
    """Merge leaf HLL registers with the correct (max) merge under a schedule."""
    if len(leaf_registers) == 0:
        raise ValueError("leaf_registers must be non-empty")
    cur = [np.array(x, copy=True) for x in leaf_registers]
    if schedule == "balanced":
        while len(cur) > 1:
            nxt: List[np.ndarray] = []
            i = 0
            while i < len(cur):
                if i + 1 >= len(cur):
                    nxt.append(cur[i])
                    i += 1
                    continue
                nxt.append(np.maximum(cur[i], cur[i + 1]))
                i += 2
            cur = nxt
        return cur[0]
    if schedule in ("left_to_right", "right_to_left"):
        if schedule == "right_to_left":
            cur = list(reversed(cur))
        acc = np.array(cur[0], copy=True)
        for reg in cur[1:]:
            np.maximum(acc, reg, out=acc)
        return acc
    raise ValueError(f"unsupported schedule: {schedule!r}")


def _hll_estimate_from_registers(
    registers: np.ndarray,
    *,
    precision: int,
    hash_bits: int = 64,
) -> float:
    """Compute HLL estimate given registers (uint8). Mirrors HyperLogLogSketch.estimate()."""
    cfg = HLLConfig(precision=int(precision), hash_bits=int(hash_bits))
    sk = HyperLogLogSketch(cfg)
    sk.registers[:] = registers.astype(np.uint8, copy=False)
    return float(sk.estimate())


@dataclass(frozen=True)
class TokenStreamDoc:
    token_ids: Tuple[int, ...]
    leaf_token_lists: Tuple[Tuple[int, ...], ...]
    true_cardinality: int


def _build_zipf_probability_bank(
    universe_size: int,
    alphas: Sequence[float],
) -> Dict[float, np.ndarray]:
    if universe_size <= 0:
        raise ValueError("universe_size must be positive")
    if len(alphas) == 0:
        raise ValueError("alphas must be non-empty")
    if any(float(a) <= 0.0 for a in alphas):
        raise ValueError("zipf alphas must be > 0")

    ranks = np.arange(1, int(universe_size) + 1, dtype=np.float64)
    bank: Dict[float, np.ndarray] = {}
    for a in alphas:
        weights = np.power(ranks, -float(a))
        probs = weights / float(weights.sum())
        bank[float(a)] = probs.astype(np.float64, copy=False)
    return bank


def generate_token_stream_docs(
    n_docs: int,
    *,
    universe_size: int,
    min_tokens: int,
    max_tokens: int,
    leaf_size: int,
    zipf_alphas: Sequence[float],
    seed: int,
) -> Tuple[TokenStreamDoc, ...]:
    if n_docs <= 0:
        return tuple()
    if max_tokens <= 0 or min_tokens <= 0 or max_tokens < min_tokens:
        raise ValueError("require 0 < min_tokens <= max_tokens")
    if universe_size <= 1:
        raise ValueError("universe_size must be >= 2")
    if leaf_size <= 0:
        raise ValueError("leaf_size must be positive")
    if len(zipf_alphas) == 0:
        raise ValueError("zipf_alphas must be non-empty")

    rng = np.random.default_rng(int(seed))
    alphas = tuple(float(a) for a in zipf_alphas)
    bank = _build_zipf_probability_bank(int(universe_size), alphas)
    alpha_keys = tuple(bank.keys())
    docs: List[TokenStreamDoc] = []

    for _ in range(int(n_docs)):
        alpha = float(alpha_keys[int(rng.integers(0, len(alpha_keys)))])
        probs = bank[alpha]
        n_tok = int(rng.integers(int(min_tokens), int(max_tokens) + 1))
        token_ids = rng.choice(
            int(universe_size),
            size=int(n_tok),
            replace=True,
            p=probs,
        ).astype(np.int64, copy=False)
        leaf_tokens: List[Tuple[int, ...]] = []
        for i in range(0, int(token_ids.shape[0]), int(leaf_size)):
            leaf = token_ids[i : i + int(leaf_size)]
            leaf_tokens.append(tuple(int(x) for x in leaf.tolist()))
        true_card = int(np.unique(token_ids).shape[0])
        docs.append(
            TokenStreamDoc(
                token_ids=tuple(int(x) for x in token_ids.tolist()),
                leaf_token_lists=tuple(leaf_tokens),
                true_cardinality=true_card,
            )
        )
    return tuple(docs)


def leaf_hll_registers(
    doc: TokenStreamDoc,
    *,
    precision: int,
    hash_bits: int = 64,
) -> Tuple[np.ndarray, ...]:
    cfg = HLLConfig(precision=int(precision), hash_bits=int(hash_bits))
    regs: List[np.ndarray] = []
    for leaf in doc.leaf_token_lists:
        sk = HyperLogLogSketch.from_tokens(cfg, leaf)
        regs.append(np.array(sk.registers, copy=True))
    return tuple(regs)


def precompute_leaf_hll_registers(
    docs: Sequence[TokenStreamDoc],
    *,
    precision: int,
    hash_bits: int = 64,
) -> Tuple[Tuple[np.ndarray, ...], ...]:
    """Precompute per-document leaf registers to avoid re-hashing each epoch."""
    return tuple(
        leaf_hll_registers(doc, precision=int(precision), hash_bits=int(hash_bits))
        for doc in docs
    )


class LearnedHLLMerger(nn.Module):
    """Elementwise MLP that merges two HLL register vectors."""

    def __init__(self, *, precision: int, hash_bits: int = 64, hidden_dim: int = 16):
        super().__init__()
        self.precision = int(precision)
        self.hash_bits = int(hash_bits)
        self.hidden_dim = int(hidden_dim)
        self.max_rho = float(_max_rho(self.hash_bits, self.precision))

        self.net = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        x = torch.stack([left, right], dim=-1)  # (..., 2)
        out = self.net(x).squeeze(-1)
        return torch.clamp(out, 0.0, self.max_rho)


class ExactMaxMerger(nn.Module):
    """Reference merger: elementwise max (merge-safe)."""

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.maximum(left, right)


class MeanMerger(nn.Module):
    """A deliberately wrong, non-associative merge (for negative controls)."""

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return 0.5 * (left + right)


def merge_leaf_states(
    merger: nn.Module,
    leaf_states: Sequence[torch.Tensor],
    *,
    schedule: ScheduleName,
) -> torch.Tensor:
    """Merge leaf states under a schedule (no supervision, inference-only)."""
    root, _, _ = _merge_states_torch(
        merger,
        leaf_states,
        leaf_states,
        schedule=str(schedule),
        audit_indices=None,
        collect_losses=False,
        idem_weight=0.0,
        comm_weight=0.0,
    )
    return root


def _merge_states_torch(
    merger: nn.Module,
    leaf_states: Sequence[torch.Tensor],
    leaf_oracle: Sequence[torch.Tensor],
    *,
    schedule: ScheduleName,
    audit_indices: Optional[set[int]],
    collect_losses: bool,
    idem_weight: float,
    comm_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if len(leaf_states) == 0:
        raise ValueError("leaf_states must be non-empty")
    if len(leaf_states) != len(leaf_oracle):
        raise ValueError("leaf_states and leaf_oracle must align")

    # Leaves are exact inputs (we do not learn the leaf encoder here).
    pred = list(leaf_states)
    oracle = list(leaf_oracle)

    c3_loss = torch.zeros((), device=pred[0].device, dtype=torch.float32)
    c3_count = 0
    merge_idx = 0

    def _maybe_supervise(
        merged_pred: torch.Tensor,
        left_pred: torch.Tensor,
        right_pred: torch.Tensor,
        merged_oracle: torch.Tensor,
    ) -> None:
        nonlocal c3_loss, c3_count
        if not collect_losses:
            return
        if audit_indices is not None and merge_idx not in audit_indices:
            return

        loss = F.mse_loss(merged_pred, merged_oracle, reduction="mean")
        if comm_weight > 0.0:
            merged_rev = merger.merge(right_pred, left_pred)
            loss = loss + float(comm_weight) * F.mse_loss(merged_pred, merged_rev, reduction="mean")
        if idem_weight > 0.0:
            idem_l = merger.merge(left_pred, left_pred)
            idem_r = merger.merge(right_pred, right_pred)
            loss = loss + float(idem_weight) * (
                F.mse_loss(idem_l, left_pred, reduction="mean")
                + F.mse_loss(idem_r, right_pred, reduction="mean")
            )
        c3_loss = c3_loss + loss
        c3_count += 1

    if schedule == "balanced":
        while len(pred) > 1:
            nxt_pred: List[torch.Tensor] = []
            nxt_oracle: List[torch.Tensor] = []
            i = 0
            while i < len(pred):
                if i + 1 >= len(pred):
                    nxt_pred.append(pred[i])
                    nxt_oracle.append(oracle[i])
                    i += 1
                    continue
                merged_pred = merger.merge(pred[i], pred[i + 1])
                merged_oracle = torch.maximum(oracle[i], oracle[i + 1])
                _maybe_supervise(merged_pred, pred[i], pred[i + 1], merged_oracle)
                merge_idx += 1
                nxt_pred.append(merged_pred)
                nxt_oracle.append(merged_oracle)
                i += 2
            pred = nxt_pred
            oracle = nxt_oracle
        return pred[0], c3_loss, c3_count

    if schedule in ("left_to_right", "right_to_left"):
        if schedule == "right_to_left":
            pred = list(reversed(pred))
            oracle = list(reversed(oracle))
        acc_pred = pred[0]
        acc_oracle = oracle[0]
        for st, st_or in zip(pred[1:], oracle[1:]):
            merged_pred = merger.merge(acc_pred, st)
            merged_oracle = torch.maximum(acc_oracle, st_or)
            _maybe_supervise(merged_pred, acc_pred, st, merged_oracle)
            merge_idx += 1
            acc_pred = merged_pred
            acc_oracle = merged_oracle
        return acc_pred, c3_loss, c3_count

    raise ValueError(f"unsupported schedule: {schedule!r}")


@dataclass(frozen=True)
class MergeEvalMetrics:
    relative_rmse: float
    mean_abs_rel_error: float
    schedule_spread_mean: float
    schedule_spread_p95: float


@dataclass(frozen=True)
class HLLBaselineMetrics:
    precision: int
    registers: int
    memory_bits: int
    memory_bytes: float
    rse_theory: float
    metrics: MergeEvalMetrics


@dataclass(frozen=True)
class LearnedMergerMetrics:
    metrics: MergeEvalMetrics
    merge_mse_mean: float


@dataclass(frozen=True)
class HLLMergeLearningRun:
    seed: int
    precision: int
    registers: int
    memory_bits: int
    train_docs: int
    n_test: int
    audit_policy: AuditPolicyName
    audit_fixed_nodes: int
    audit_fraction: float
    audit_scale: float
    n_epochs: int
    hidden_dim: int
    lr: float
    weight_decay: float
    idem_weight: float
    comm_weight: float
    train_mean_internal_nodes: float
    train_audit_nodes_mean: float
    train_audit_coverage_mean: float
    train_total_queries_estimate: int
    hll_baseline: HLLBaselineMetrics
    learned: LearnedMergerMetrics
    distance_to_hll_floor_rel_rmse: float
    ratio_to_hll_floor_rel_rmse: float


@dataclass(frozen=True)
class HLLMergeLearningConfig:
    universe_size: int = 65_536
    min_tokens: int = 4096
    max_tokens: int = 16384
    leaf_size: int = 512
    zipf_alphas: Tuple[float, ...] = (0.8, 1.0, 1.2)
    precisions: Tuple[int, ...] = (6, 7, 8, 9, 10, 11, 12)
    train_docs_grid: Tuple[int, ...] = (25, 50, 100, 200, 500, 1000)
    audit_policies: Tuple[AuditPolicyName, ...] = ("all", "sqrt", "log2", "fraction")
    audit_fixed_nodes: int = 0
    audit_fraction: float = 0.25
    audit_scale: float = 1.0
    n_test: int = 256
    hidden_dim: int = 16
    n_epochs: int = 6
    batch_docs: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    idem_weight: float = 0.10
    comm_weight: float = 0.10
    use_cuda: bool = True
    cuda_device: Optional[int] = None
    torch_threads: int = 0
    seed: int = 0


def _summarize_audit_geometry(
    leaf_regs_by_doc: Sequence[Tuple[np.ndarray, ...]],
    *,
    audit_policy: AuditPolicyName,
    audit_fixed_nodes: int,
    audit_fraction: float,
    audit_scale: float,
) -> Dict[str, float | int]:
    if len(leaf_regs_by_doc) == 0:
        return {
            "mean_internal_nodes": 0.0,
            "audit_nodes_mean": 0.0,
            "audit_coverage_mean": 0.0,
            "total_queries_estimate": 0,
        }
    internal: List[float] = []
    audits: List[float] = []
    covers: List[float] = []
    audit_nodes_total = 0
    for leaf_regs in leaf_regs_by_doc:
        n_leaves = int(len(leaf_regs))
        n_internal = int(max(0, n_leaves - 1))
        q = audit_sample_count(
            n_internal,
            policy=str(audit_policy),
            fixed_nodes=int(audit_fixed_nodes),
            fraction=float(audit_fraction),
            scale=float(audit_scale),
        )
        internal.append(float(n_internal))
        audits.append(float(q))
        covers.append(float(q) / float(n_internal) if n_internal > 0 else 1.0)
        audit_nodes_total += int(q)
    n = float(len(leaf_regs_by_doc))
    return {
        "mean_internal_nodes": float(sum(internal) / n),
        "audit_nodes_mean": float(sum(audits) / n),
        "audit_coverage_mean": float(sum(covers) / n),
        "total_queries_estimate": int(audit_nodes_total),
    }


def evaluate_merger_on_docs(
    docs: Sequence[TokenStreamDoc],
    *,
    merger: nn.Module,
    precision: int,
    hash_bits: int = 64,
    leaf_regs_by_doc: Optional[Sequence[Tuple[np.ndarray, ...]]] = None,
    device: torch.device,
) -> MergeEvalMetrics:
    if len(docs) == 0:
        return MergeEvalMetrics(
            relative_rmse=0.0,
            mean_abs_rel_error=0.0,
            schedule_spread_mean=0.0,
            schedule_spread_p95=0.0,
        )
    rel_sq: List[float] = []
    abs_rel: List[float] = []
    spreads: List[float] = []

    merger.eval()
    with torch.no_grad():
        if leaf_regs_by_doc is None:
            leaf_regs_by_doc = precompute_leaf_hll_registers(
                docs, precision=int(precision), hash_bits=int(hash_bits)
            )
        if len(leaf_regs_by_doc) != len(docs):
            raise ValueError("leaf_regs_by_doc must have same length as docs")

        for doc, leaf_regs in zip(docs, leaf_regs_by_doc):
            leaf = [
                torch.tensor(x, dtype=torch.float32, device=device) for x in leaf_regs
            ]
            ests: Dict[str, float] = {}
            for sched in VALID_SCHEDULES:
                root, _, _ = _merge_states_torch(
                    merger,
                    leaf,
                    leaf,
                    schedule=sched,
                    audit_indices=None,
                    collect_losses=False,
                    idem_weight=0.0,
                    comm_weight=0.0,
                )
                root_np = root.detach().cpu().numpy()
                root_uint = np.rint(
                    np.clip(root_np, 0.0, float(_max_rho(hash_bits, precision)))
                ).astype(np.uint8)
                ests[sched] = _hll_estimate_from_registers(
                    root_uint,
                    precision=int(precision),
                    hash_bits=int(hash_bits),
                )
            pred = float(ests["balanced"])
            truth = float(doc.true_cardinality)
            rel = _safe_rel_error(pred, truth)
            rel_sq.append(rel * rel)
            abs_rel.append(abs(rel))
            spreads.append(max(ests.values()) - min(ests.values()))

    rel_rmse = float(math.sqrt(float(np.mean(np.array(rel_sq, dtype=np.float64)))))
    return MergeEvalMetrics(
        relative_rmse=rel_rmse,
        mean_abs_rel_error=float(np.mean(np.array(abs_rel, dtype=np.float64))),
        schedule_spread_mean=float(np.mean(np.array(spreads, dtype=np.float64))),
        schedule_spread_p95=float(np.percentile(np.array(spreads, dtype=np.float64), 95.0)),
    )


def evaluate_hll_baseline(
    docs: Sequence[TokenStreamDoc],
    *,
    precision: int,
    hash_bits: int = 64,
    leaf_regs_by_doc: Optional[Sequence[Tuple[np.ndarray, ...]]] = None,
) -> HLLBaselineMetrics:
    cfg = HLLConfig(precision=int(precision), hash_bits=int(hash_bits))
    proto = HyperLogLogSketch(cfg)
    rse = float(hll_relative_standard_error(int(precision)))
    if leaf_regs_by_doc is None:
        leaf_regs_by_doc = precompute_leaf_hll_registers(
            docs, precision=int(precision), hash_bits=int(hash_bits)
        )
    if len(leaf_regs_by_doc) != len(docs):
        raise ValueError("leaf_regs_by_doc must have same length as docs")
    metrics = evaluate_merger_on_docs(
        docs,
        merger=ExactMaxMerger(),
        precision=int(precision),
        hash_bits=int(hash_bits),
        leaf_regs_by_doc=leaf_regs_by_doc,
        device=torch.device("cpu"),
    )
    return HLLBaselineMetrics(
        precision=int(precision),
        registers=int(proto.m),
        memory_bits=int(proto.memory_bits),
        memory_bytes=float(proto.memory_bits) / 8.0,
        rse_theory=rse,
        metrics=metrics,
    )


def train_learned_merger(
    model: LearnedHLLMerger,
    train_docs: Sequence[TokenStreamDoc],
    *,
    train_leaf_regs_by_doc: Sequence[Tuple[np.ndarray, ...]],
    precision: int,
    audit_policy: AuditPolicyName,
    audit_fixed_nodes: int,
    audit_fraction: float,
    audit_scale: float,
    n_epochs: int,
    batch_docs: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    idem_weight: float,
    comm_weight: float,
    device: torch.device,
    seed: int,
) -> float:
    if len(train_docs) == 0:
        raise ValueError("train_docs must be non-empty")
    if batch_docs <= 0:
        raise ValueError("batch_docs must be positive")
    if len(train_leaf_regs_by_doc) != len(train_docs):
        raise ValueError("train_leaf_regs_by_doc must align with train_docs")

    model.to(device)
    model.train()
    opt = torch.optim.Adam(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    rng = random.Random(int(seed))

    merge_mse_terms: List[float] = []
    idxs = list(range(len(train_docs)))
    for _ in range(int(n_epochs)):
        rng.shuffle(idxs)
        for b0 in range(0, len(idxs), int(batch_docs)):
            batch_idx = idxs[b0 : b0 + int(batch_docs)]
            opt.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device, dtype=torch.float32)
            n_losses = 0
            for i in batch_idx:
                doc = train_docs[i]
                leaf_regs = train_leaf_regs_by_doc[i]
                leaf = [
                    torch.tensor(x, dtype=torch.float32, device=device) for x in leaf_regs
                ]
                n_internal = int(max(0, len(leaf) - 1))
                n_audit = audit_sample_count(
                    n_internal,
                    policy=str(audit_policy),
                    fixed_nodes=int(audit_fixed_nodes),
                    fraction=float(audit_fraction),
                    scale=float(audit_scale),
                )
                if n_audit <= 0:
                    audit_indices: Optional[set[int]] = set()
                elif n_audit >= n_internal:
                    audit_indices = None
                else:
                    audit_indices = set(rng.sample(range(n_internal), k=int(n_audit)))

                root, c3_loss, c3_count = _merge_states_torch(
                    model,
                    leaf,
                    leaf,
                    schedule="balanced",
                    audit_indices=audit_indices,
                    collect_losses=True,
                    idem_weight=float(idem_weight),
                    comm_weight=float(comm_weight),
                )
                if int(c3_count) > 0:
                    doc_loss = c3_loss / float(c3_count)
                    loss = loss + doc_loss
                    n_losses += 1

                    # Track raw merge MSE (without regularizers) for reporting.
                    with torch.no_grad():
                        oracle_root = torch.max(torch.stack(leaf, dim=0), dim=0).values
                        mse = F.mse_loss(root, oracle_root, reduction="mean")
                        merge_mse_terms.append(float(mse.detach().cpu()))
            if n_losses <= 0:
                continue
            loss = loss / float(n_losses)
            loss.backward()
            if float(grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            opt.step()

    if len(merge_mse_terms) == 0:
        return float("nan")
    return float(np.mean(np.array(merge_mse_terms, dtype=np.float64)))


def run_hll_merge_learning_experiment(config: HLLMergeLearningConfig) -> Tuple[HLLMergeLearningRun, ...]:
    _set_global_seed(int(config.seed))

    if config.use_cuda and torch.cuda.is_available():
        if config.cuda_device is not None:
            cuda_idx = int(config.cuda_device)
            n_cuda = int(torch.cuda.device_count())
            if cuda_idx < 0 or cuda_idx >= n_cuda:
                raise ValueError(
                    f"cuda_device={cuda_idx} out of range; available devices: 0..{n_cuda - 1}"
                )
            torch.cuda.set_device(cuda_idx)
            device = torch.device(f"cuda:{cuda_idx}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    max_train = max(int(x) for x in config.train_docs_grid) if config.train_docs_grid else 0
    total_docs = int(max_train + config.n_test)
    docs = generate_token_stream_docs(
        total_docs,
        universe_size=int(config.universe_size),
        min_tokens=int(config.min_tokens),
        max_tokens=int(config.max_tokens),
        leaf_size=int(config.leaf_size),
        zipf_alphas=_parse_csv_floats(config.zipf_alphas),
        seed=int(config.seed),
    )
    train_pool = docs[:max_train]
    test_docs = docs[max_train:]

    results: List[HLLMergeLearningRun] = []
    baseline_cache: Dict[int, HLLBaselineMetrics] = {}

    for p in (int(x) for x in config.precisions):
        # Precompute leaf registers for this precision once; reuse for all train/policy slices.
        leaf_regs_all = precompute_leaf_hll_registers(
            docs,
            precision=int(p),
            hash_bits=64,
        )
        train_leaf_regs_all = leaf_regs_all[:max_train]
        test_leaf_regs = leaf_regs_all[max_train:]

        if p not in baseline_cache:
            baseline_cache[p] = evaluate_hll_baseline(
                test_docs, precision=p, hash_bits=64, leaf_regs_by_doc=test_leaf_regs
            )
        baseline = baseline_cache[p]

        for audit_policy in config.audit_policies:
            for train_docs in (int(x) for x in config.train_docs_grid):
                if train_docs <= 0:
                    continue
                model = LearnedHLLMerger(
                    precision=int(p),
                    hash_bits=64,
                    hidden_dim=int(config.hidden_dim),
                )
                train_subset = train_pool[:train_docs]
                train_leaf_regs = train_leaf_regs_all[:train_docs]
                geom = _summarize_audit_geometry(
                    train_leaf_regs,
                    audit_policy=str(audit_policy),
                    audit_fixed_nodes=int(config.audit_fixed_nodes),
                    audit_fraction=float(config.audit_fraction),
                    audit_scale=float(config.audit_scale),
                )

                merge_mse_mean = train_learned_merger(
                    model,
                    train_subset,
                    train_leaf_regs_by_doc=train_leaf_regs,
                    precision=int(p),
                    audit_policy=str(audit_policy),
                    audit_fixed_nodes=int(config.audit_fixed_nodes),
                    audit_fraction=float(config.audit_fraction),
                    audit_scale=float(config.audit_scale),
                    n_epochs=int(config.n_epochs),
                    batch_docs=int(config.batch_docs),
                    lr=float(config.lr),
                    weight_decay=float(config.weight_decay),
                    grad_clip_norm=float(config.grad_clip_norm),
                    idem_weight=float(config.idem_weight),
                    comm_weight=float(config.comm_weight),
                    device=device,
                    seed=int(config.seed + 7919 + p + train_docs),
                )

                learned_metrics = evaluate_merger_on_docs(
                    test_docs,
                    merger=model,
                    precision=int(p),
                    hash_bits=64,
                    leaf_regs_by_doc=test_leaf_regs,
                    device=device,
                )
                dist = float(learned_metrics.relative_rmse - baseline.rse_theory)
                ratio = float(learned_metrics.relative_rmse / max(1e-12, baseline.rse_theory))

                results.append(
                    HLLMergeLearningRun(
                        seed=int(config.seed),
                        precision=int(p),
                        registers=int(baseline.registers),
                        memory_bits=int(baseline.memory_bits),
                        train_docs=int(train_docs),
                        n_test=int(config.n_test),
                        audit_policy=str(audit_policy),
                        audit_fixed_nodes=int(config.audit_fixed_nodes),
                        audit_fraction=float(config.audit_fraction),
                        audit_scale=float(config.audit_scale),
                        n_epochs=int(config.n_epochs),
                        hidden_dim=int(config.hidden_dim),
                        lr=float(config.lr),
                        weight_decay=float(config.weight_decay),
                        idem_weight=float(config.idem_weight),
                        comm_weight=float(config.comm_weight),
                        train_mean_internal_nodes=float(geom["mean_internal_nodes"]),
                        train_audit_nodes_mean=float(geom["audit_nodes_mean"]),
                        train_audit_coverage_mean=float(geom["audit_coverage_mean"]),
                        train_total_queries_estimate=int(geom["total_queries_estimate"]),
                        hll_baseline=baseline,
                        learned=LearnedMergerMetrics(metrics=learned_metrics, merge_mse_mean=merge_mse_mean),
                        distance_to_hll_floor_rel_rmse=dist,
                        ratio_to_hll_floor_rel_rmse=ratio,
                    )
                )

    return tuple(results)


def experiment_rows(results: Sequence[HLLMergeLearningRun]) -> List[dict]:
    rows: List[dict] = []
    for r in results:
        hb = r.hll_baseline
        lm = r.learned.metrics
        bm = hb.metrics
        rows.append(
            {
                "seed": int(r.seed),
                "precision": int(r.precision),
                "registers": int(r.registers),
                "memory_bits": int(r.memory_bits),
                "memory_bytes": float(r.memory_bits) / 8.0,
                "train_docs": int(r.train_docs),
                "n_test": int(r.n_test),
                "audit_policy": str(r.audit_policy),
                "audit_fixed_nodes": int(r.audit_fixed_nodes),
                "audit_fraction": float(r.audit_fraction),
                "audit_scale": float(r.audit_scale),
                "n_epochs": int(r.n_epochs),
                "hidden_dim": int(r.hidden_dim),
                "lr": float(r.lr),
                "weight_decay": float(r.weight_decay),
                "idem_weight": float(r.idem_weight),
                "comm_weight": float(r.comm_weight),
                "train_mean_internal_nodes": float(r.train_mean_internal_nodes),
                "train_audit_nodes_mean": float(r.train_audit_nodes_mean),
                "train_audit_coverage_mean": float(r.train_audit_coverage_mean),
                "train_total_queries_estimate": int(r.train_total_queries_estimate),
                "hll_rse_theory": float(hb.rse_theory),
                "hll_relative_rmse": float(bm.relative_rmse),
                "hll_schedule_spread_mean": float(bm.schedule_spread_mean),
                "learned_relative_rmse": float(lm.relative_rmse),
                "learned_mean_abs_rel_error": float(lm.mean_abs_rel_error),
                "learned_schedule_spread_mean": float(lm.schedule_spread_mean),
                "learned_schedule_spread_p95": float(lm.schedule_spread_p95),
                "merge_mse_mean": float(r.learned.merge_mse_mean),
                "distance_to_hll_floor_rel_rmse": float(r.distance_to_hll_floor_rel_rmse),
                "ratio_to_hll_floor_rel_rmse": float(r.ratio_to_hll_floor_rel_rmse),
            }
        )
    return rows


def experiment_summary_json(config: HLLMergeLearningConfig, results: Sequence[HLLMergeLearningRun]) -> str:
    payload = {
        "config": asdict(config),
        "rows": experiment_rows(results),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


__all__ = [
    "VALID_SCHEDULES",
    "ExactMaxMerger",
    "HLLMergeLearningConfig",
    "HLLMergeLearningRun",
    "LearnedHLLMerger",
    "MeanMerger",
    "MergeEvalMetrics",
    "TokenStreamDoc",
    "evaluate_hll_baseline",
    "evaluate_merger_on_docs",
    "experiment_rows",
    "experiment_summary_json",
    "generate_token_stream_docs",
    "leaf_hll_registers",
    "merge_leaf_states",
    "run_hll_merge_learning_experiment",
    "_merge_schedule_registers_max",
]
