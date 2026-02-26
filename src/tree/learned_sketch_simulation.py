"""
Learned mergeable-sketch simulation with an HLL baseline.

This module provides a fully worked numeric simulation for the claim:
under sufficient state budget and oracle-query supervision, a learned
tree sketch approaches the performance of a classical mergeable sketch.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


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


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _splitmix64(x: int) -> int:
    """Deterministic 64-bit hash for integer tokens."""
    z = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z & 0xFFFFFFFFFFFFFFFF


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _safe_rel_error(pred: float, truth: float) -> float:
    denom = max(1.0, float(truth))
    return (float(pred) - float(truth)) / denom


@dataclass(frozen=True)
class HLLConfig:
    precision: int
    hash_bits: int = 64


@dataclass(frozen=True)
class SimulationConfig:
    universe_size: int = 2048
    min_tokens: int = 128
    max_tokens: int = 512
    leaf_size: int = 32
    zipf_alphas: Tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4)
    state_dims: Tuple[int, ...] = (16, 32, 64)
    train_sizes: Tuple[int, ...] = (128, 256, 512)
    n_val: int = 256
    n_test: int = 512
    hidden_dim: int = 128
    n_epochs: int = 14
    batch_size: int = 24
    lr: float = 3e-4
    weight_decay: float = 1e-5
    c3_weight: float = 0.20
    leaf_weight: float = 0.05
    grad_clip_norm: float = 1.0
    audit_policy: AuditPolicyName = "all"
    audit_fixed_nodes: int = 0
    audit_fraction: float = 1.0
    audit_scale: float = 1.0
    audit_include_root_query: bool = True
    use_cuda: bool = True
    cuda_device: Optional[int] = None
    seed: int = 0


@dataclass(frozen=True)
class CardinalityDocument:
    token_ids: Tuple[int, ...]
    leaf_vectors: Tuple[torch.Tensor, ...]
    leaf_cardinalities: Tuple[float, ...]
    true_cardinality: float


@dataclass(frozen=True)
class ModelEvalMetrics:
    mae: float
    rmse: float
    relative_rmse: float
    mean_rel_error: float
    mean_abs_rel_error: float
    schedule_spread_mean: float
    schedule_spread_p95: float
    c3_state_mse: float


@dataclass(frozen=True)
class HLLMetrics:
    precision: int
    registers: int
    register_bits: int
    memory_bits: int
    memory_bytes: float
    mae: float
    rmse: float
    relative_rmse: float
    mean_rel_error: float
    mean_abs_rel_error: float
    schedule_spread_mean: float
    schedule_spread_p95: float


@dataclass(frozen=True)
class LearningRunSummary:
    state_dim: int
    learned_memory_bits: int
    train_size: int
    val_loss_final: float
    train_loss_final: float
    learned_metrics: ModelEvalMetrics
    hll_metrics: HLLMetrics
    hll_rse_theory: float
    distance_to_hll_floor_rel_rmse: float
    distance_to_hll_empirical_rel_rmse: float
    train_mean_tokens: float
    train_mean_leaves: float
    train_mean_internal_nodes: float
    train_audit_nodes_mean: float
    train_audit_coverage_mean: float
    train_root_queries_total: int
    train_audit_nodes_total: int
    train_total_queries_estimate: int
    rmse_gap_vs_hll: float
    abs_rel_error_gap_vs_hll: float
    # Distance-to-floor metrics (absolute RMSE domain).
    theoretical_floor_rmse: float
    excess_rmse: float
    ratio_to_floor_rmse: float
    ratio_to_floor_rel_rmse: float
    hll_empirical_excess_rmse: float
    hll_empirical_excess_rel_rmse: float
    test_cardinality_rms: float
    test_cardinality_mean: float


@dataclass(frozen=True)
class ExperimentSummary:
    config: Dict[str, object]
    results: Tuple[LearningRunSummary, ...]

    def to_json(self) -> str:
        payload = {
            "config": self.config,
            "results": [_serialize_learning_run(x) for x in self.results],
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def _serialize_learning_run(run: LearningRunSummary) -> Dict[str, object]:
    return {
        "state_dim": run.state_dim,
        "learned_memory_bits": run.learned_memory_bits,
        "train_size": run.train_size,
        "val_loss_final": run.val_loss_final,
        "train_loss_final": run.train_loss_final,
        "hll_rse_theory": run.hll_rse_theory,
        "distance_to_hll_floor_rel_rmse": run.distance_to_hll_floor_rel_rmse,
        "distance_to_hll_empirical_rel_rmse": run.distance_to_hll_empirical_rel_rmse,
        "train_mean_tokens": run.train_mean_tokens,
        "train_mean_leaves": run.train_mean_leaves,
        "train_mean_internal_nodes": run.train_mean_internal_nodes,
        "train_audit_nodes_mean": run.train_audit_nodes_mean,
        "train_audit_coverage_mean": run.train_audit_coverage_mean,
        "train_root_queries_total": run.train_root_queries_total,
        "train_audit_nodes_total": run.train_audit_nodes_total,
        "train_total_queries_estimate": run.train_total_queries_estimate,
        "rmse_gap_vs_hll": run.rmse_gap_vs_hll,
        "abs_rel_error_gap_vs_hll": run.abs_rel_error_gap_vs_hll,
        "theoretical_floor_rmse": run.theoretical_floor_rmse,
        "excess_rmse": run.excess_rmse,
        "ratio_to_floor_rmse": run.ratio_to_floor_rmse,
        "ratio_to_floor_rel_rmse": run.ratio_to_floor_rel_rmse,
        "hll_empirical_excess_rmse": run.hll_empirical_excess_rmse,
        "hll_empirical_excess_rel_rmse": run.hll_empirical_excess_rel_rmse,
        "test_cardinality_rms": run.test_cardinality_rms,
        "test_cardinality_mean": run.test_cardinality_mean,
        "learned_metrics": asdict(run.learned_metrics),
        "hll_metrics": asdict(run.hll_metrics),
    }


def audit_sample_count(
    internal_nodes: int,
    *,
    policy: AuditPolicyName,
    fixed_nodes: int = 0,
    fraction: float = 1.0,
    scale: float = 1.0,
) -> int:
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


def _summarize_audit_geometry(
    docs: Sequence[CardinalityDocument],
    *,
    policy: AuditPolicyName,
    fixed_nodes: int,
    fraction: float,
    scale: float,
    include_root_query: bool,
) -> Dict[str, float | int]:
    if len(docs) == 0:
        return {
            "mean_tokens": 0.0,
            "mean_leaves": 0.0,
            "mean_internal_nodes": 0.0,
            "audit_nodes_mean": 0.0,
            "audit_coverage_mean": 0.0,
            "root_queries_total": 0,
            "audit_nodes_total": 0,
            "total_queries_estimate": 0,
        }

    n_docs = int(len(docs))
    toks: List[float] = []
    leaves: List[float] = []
    internals: List[float] = []
    audits: List[float] = []
    covers: List[float] = []
    audit_nodes_total = 0

    for doc in docs:
        n_tok = int(len(doc.token_ids))
        n_leaves = int(len(doc.leaf_vectors))
        n_internal = int(max(0, n_leaves - 1))
        q = audit_sample_count(
            n_internal,
            policy=policy,
            fixed_nodes=int(fixed_nodes),
            fraction=float(fraction),
            scale=float(scale),
        )
        toks.append(float(n_tok))
        leaves.append(float(n_leaves))
        internals.append(float(n_internal))
        audits.append(float(q))
        covers.append(float(q) / float(n_internal) if n_internal > 0 else 1.0)
        audit_nodes_total += int(q)

    root_queries_total = int(n_docs if include_root_query else 0)
    return {
        "mean_tokens": float(np.mean(np.array(toks, dtype=np.float64))),
        "mean_leaves": float(np.mean(np.array(leaves, dtype=np.float64))),
        "mean_internal_nodes": float(np.mean(np.array(internals, dtype=np.float64))),
        "audit_nodes_mean": float(np.mean(np.array(audits, dtype=np.float64))),
        "audit_coverage_mean": float(np.mean(np.array(covers, dtype=np.float64))),
        "root_queries_total": int(root_queries_total),
        "audit_nodes_total": int(audit_nodes_total),
        "total_queries_estimate": int(root_queries_total + audit_nodes_total),
    }


class HyperLogLogSketch:
    def __init__(self, config: HLLConfig):
        if not (4 <= int(config.precision) <= int(config.hash_bits) - 2):
            raise ValueError("precision must be in [4, hash_bits-2]")
        self.config = config
        self.m = 1 << int(config.precision)
        self.registers = np.zeros(self.m, dtype=np.uint8)
        self._remaining_bits = int(config.hash_bits) - int(config.precision)
        self._remaining_mask = (1 << self._remaining_bits) - 1

    @property
    def register_bits(self) -> int:
        return int(math.ceil(math.log2(self._remaining_bits + 1)))

    @property
    def memory_bits(self) -> int:
        return int(self.m * self.register_bits)

    def add(self, token_id: int) -> None:
        h = _splitmix64(int(token_id))
        p = int(self.config.precision)
        idx = h >> (self.config.hash_bits - p)
        w = h & self._remaining_mask
        if w == 0:
            rho = self._remaining_bits + 1
        else:
            rho = self._remaining_bits - int(w.bit_length()) + 1
        if rho > int(self.registers[idx]):
            self.registers[idx] = min(rho, 255)

    def merge(self, other: "HyperLogLogSketch") -> None:
        if self.config != other.config:
            raise ValueError("cannot merge HLL sketches with different configs")
        np.maximum(self.registers, other.registers, out=self.registers)

    def estimate(self) -> float:
        m = float(self.m)
        z = np.power(2.0, -self.registers.astype(np.float64)).sum()
        alpha = _hll_alpha(self.m)
        raw = alpha * (m * m) / max(z, 1e-12)
        n_zeros = int((self.registers == 0).sum())
        if raw <= 2.5 * m and n_zeros > 0:
            return m * math.log(m / float(n_zeros))
        return raw

    @staticmethod
    def from_tokens(config: HLLConfig, token_ids: Sequence[int]) -> "HyperLogLogSketch":
        sk = HyperLogLogSketch(config)
        for tok in token_ids:
            sk.add(int(tok))
        return sk


def _hll_alpha(m: int) -> float:
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / float(m))


def hll_relative_standard_error(precision: int) -> float:
    m = 1 << int(precision)
    return 1.04 / math.sqrt(float(m))


def compute_theoretical_floor_rmse(
    hll_rse_theory: float,
    test_cardinalities: Sequence[float],
) -> float:
    """Absolute RMSE floor from HLL asymptotic theory.

    HLL RSE = 1.04/sqrt(m) is the *relative* standard error: for a
    single stream with true cardinality n, RMSE ≈ RSE × n.  Over a test
    set with varying cardinalities {n_i}, the population RMSE floor is:

        floor_rmse = RSE × sqrt(mean_i(n_i²))

    For cardinalities in the linear-counting regime (n < 2.5m), HLL
    uses a bias-corrected estimator, so empirical HLL RMSE may slightly
    exceed this asymptotic formula.  This is expected and visible as a
    small positive ``hll_empirical_excess_rmse``.
    """
    if len(test_cardinalities) == 0:
        return 0.0
    cards_sq = np.array(
        [float(n) ** 2 for n in test_cardinalities], dtype=np.float64
    )
    return float(hll_rse_theory) * math.sqrt(float(np.mean(cards_sq)))


def _hll_memory_bits(precision: int, hash_bits: int = 64) -> int:
    m = 1 << int(precision)
    reg_bits = int(math.ceil(math.log2((hash_bits - int(precision)) + 1)))
    return int(m * reg_bits)


def match_hll_precision_for_bits(
    target_bits: int,
    *,
    hash_bits: int = 64,
    p_min: int = 4,
    p_max: int = 16,
) -> int:
    target = int(max(1, target_bits))
    best_p = p_min
    best_gap = float("inf")
    for p in range(int(p_min), int(p_max) + 1):
        bits = _hll_memory_bits(p, hash_bits=hash_bits)
        gap = abs(bits - target)
        if gap < best_gap:
            best_gap = gap
            best_p = p
    return best_p


def _build_zipf_probability_bank(
    universe_size: int,
    alphas: Sequence[float],
) -> Dict[float, np.ndarray]:
    bank: Dict[float, np.ndarray] = {}
    ranks = np.arange(1, int(universe_size) + 1, dtype=np.float64)
    for a in alphas:
        weights = np.power(ranks, -float(a))
        probs = weights / weights.sum()
        bank[float(a)] = probs.astype(np.float64)
    return bank


def _tokens_to_leaf_multihots(
    token_ids: np.ndarray,
    *,
    leaf_size: int,
    universe_size: int,
) -> Tuple[torch.Tensor, ...]:
    out: List[torch.Tensor] = []
    n = int(token_ids.shape[0])
    for start in range(0, n, int(leaf_size)):
        chunk = token_ids[start : start + int(leaf_size)]
        vec = np.zeros(int(universe_size), dtype=np.float32)
        vec[np.unique(chunk)] = 1.0
        out.append(torch.from_numpy(vec))
    return tuple(out)


def generate_cardinality_documents(
    n_docs: int,
    *,
    universe_size: int,
    min_tokens: int,
    max_tokens: int,
    leaf_size: int,
    zipf_alphas: Sequence[float],
    seed: int,
) -> Tuple[CardinalityDocument, ...]:
    if n_docs <= 0:
        return tuple()
    if min_tokens <= 0 or max_tokens < min_tokens:
        raise ValueError("require 0 < min_tokens <= max_tokens")
    if leaf_size <= 0:
        raise ValueError("leaf_size must be positive")
    if len(zipf_alphas) == 0:
        raise ValueError("zipf_alphas must be non-empty")

    rng = np.random.default_rng(int(seed))
    bank = _build_zipf_probability_bank(int(universe_size), tuple(float(a) for a in zipf_alphas))
    alphas = tuple(bank.keys())
    docs: List[CardinalityDocument] = []

    for _ in range(int(n_docs)):
        alpha = float(alphas[int(rng.integers(0, len(alphas)))])
        probs = bank[alpha]
        n_tok = int(rng.integers(int(min_tokens), int(max_tokens) + 1))
        token_ids = rng.choice(int(universe_size), size=n_tok, replace=True, p=probs).astype(np.int64)
        leaf_vectors = _tokens_to_leaf_multihots(
            token_ids,
            leaf_size=int(leaf_size),
            universe_size=int(universe_size),
        )
        leaf_cards = tuple(float(v.sum()) for v in leaf_vectors)
        true_card = float(np.unique(token_ids).shape[0])
        docs.append(
            CardinalityDocument(
                token_ids=tuple(int(x) for x in token_ids.tolist()),
                leaf_vectors=leaf_vectors,
                leaf_cardinalities=leaf_cards,
                true_cardinality=true_card,
            )
        )
    return tuple(docs)


class LearnedMergeableSketch(nn.Module):
    def __init__(self, input_dim: int, state_dim: int, hidden_dim: int, target_scale: float):
        super().__init__()
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.target_scale = float(target_scale)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.state_dim),
        )
        self.merger = nn.Sequential(
            nn.Linear(2 * self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.state_dim),
        )
        self.readout = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def predict_norm_from_state(self, state: torch.Tensor) -> torch.Tensor:
        logit = self.readout(state)
        return torch.sigmoid(logit).squeeze(-1)

    def _merge_states(
        self,
        states: Sequence[torch.Tensor],
        unions: Sequence[torch.Tensor],
        *,
        schedule: ScheduleName,
        c3_collect: bool,
        c3_audit_indices: Optional[set[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if len(states) != len(unions):
            raise ValueError("states and unions must align")
        if len(states) == 0:
            raise ValueError("need at least one state")
        if len(states) == 1:
            zero = torch.zeros((), device=states[0].device, dtype=states[0].dtype)
            return states[0], zero, 0

        if schedule == "balanced":
            cur_states = list(states)
            cur_unions = list(unions)
            c3_loss = torch.zeros((), device=states[0].device, dtype=states[0].dtype)
            c3_count = 0
            merge_idx = 0
            while len(cur_states) > 1:
                nxt_states: List[torch.Tensor] = []
                nxt_unions: List[torch.Tensor] = []
                i = 0
                while i < len(cur_states):
                    if i + 1 >= len(cur_states):
                        nxt_states.append(cur_states[i])
                        nxt_unions.append(cur_unions[i])
                        i += 1
                        continue
                    merged = self.merger(torch.cat([cur_states[i], cur_states[i + 1]], dim=-1))
                    union = torch.clamp(cur_unions[i] + cur_unions[i + 1], min=0.0, max=1.0)
                    if c3_collect and (
                        c3_audit_indices is None or merge_idx in c3_audit_indices
                    ):
                        joint = self.encoder(union)
                        c3_loss = c3_loss + F.mse_loss(merged, joint, reduction="mean")
                        c3_count += 1
                    merge_idx += 1
                    nxt_states.append(merged)
                    nxt_unions.append(union)
                    i += 2
                cur_states = nxt_states
                cur_unions = nxt_unions
            return cur_states[0], c3_loss, c3_count

        if schedule in ("left_to_right", "right_to_left"):
            if schedule == "left_to_right":
                it_states = list(states)
                it_unions = list(unions)
            else:
                it_states = list(reversed(states))
                it_unions = list(reversed(unions))
            acc_state = it_states[0]
            acc_union = it_unions[0]
            c3_loss = torch.zeros((), device=acc_state.device, dtype=acc_state.dtype)
            c3_count = 0
            merge_idx = 0
            for st, un in zip(it_states[1:], it_unions[1:]):
                merged = self.merger(torch.cat([acc_state, st], dim=-1))
                acc_union = torch.clamp(acc_union + un, min=0.0, max=1.0)
                if c3_collect and (c3_audit_indices is None or merge_idx in c3_audit_indices):
                    joint = self.encoder(acc_union)
                    c3_loss = c3_loss + F.mse_loss(merged, joint, reduction="mean")
                    c3_count += 1
                acc_state = merged
                merge_idx += 1
            return acc_state, c3_loss, c3_count

        raise ValueError(f"unsupported schedule: {schedule!r}")

    def forward_doc(
        self,
        leaf_vectors: Sequence[torch.Tensor],
        leaf_cardinalities: Sequence[float],
        *,
        schedule: ScheduleName,
        collect_c3: bool = True,
        collect_leaf: bool = True,
        c3_audit_indices: Optional[set[int]] = None,
    ) -> Dict[str, torch.Tensor | float]:
        if len(leaf_vectors) == 0:
            raise ValueError("leaf_vectors must be non-empty")
        if len(leaf_vectors) != len(leaf_cardinalities):
            raise ValueError("leaf vectors and cardinalities must have same length")
        states = [self.encoder(v) for v in leaf_vectors]
        unions = list(leaf_vectors)
        root_state, c3_loss, c3_count = self._merge_states(
            states,
            unions,
            schedule=schedule,
            c3_collect=collect_c3,
            c3_audit_indices=c3_audit_indices,
        )
        pred_norm = self.predict_norm_from_state(root_state)
        out: Dict[str, torch.Tensor | float] = {
            "pred_norm": pred_norm,
            "pred_count": pred_norm * self.target_scale,
        }

        if collect_c3:
            out["c3_loss"] = c3_loss / max(1, c3_count)
            out["c3_count"] = float(c3_count)
        else:
            out["c3_loss"] = torch.zeros((), device=root_state.device, dtype=root_state.dtype)
            out["c3_count"] = 0.0

        if collect_leaf:
            leaf_loss = torch.zeros((), device=root_state.device, dtype=root_state.dtype)
            for state, true_leaf in zip(states, leaf_cardinalities):
                pred_leaf_norm = self.predict_norm_from_state(state)
                true_leaf_norm = torch.tensor(
                    float(true_leaf) / self.target_scale,
                    device=root_state.device,
                    dtype=pred_leaf_norm.dtype,
                )
                leaf_loss = leaf_loss + F.mse_loss(pred_leaf_norm, true_leaf_norm, reduction="mean")
            out["leaf_loss"] = leaf_loss / float(len(states))
        else:
            out["leaf_loss"] = torch.zeros((), device=root_state.device, dtype=root_state.dtype)
        return out


@dataclass(frozen=True)
class TrainDiagnostics:
    train_loss_final: float
    val_loss_final: float


def _to_device_leaf_tensors(
    doc: CardinalityDocument,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[List[torch.Tensor], List[float]]:
    leaf_vecs: List[torch.Tensor] = []
    for v in doc.leaf_vectors:
        if v.device == device and v.dtype == dtype:
            leaf_vecs.append(v)
        else:
            leaf_vecs.append(v.to(device=device, dtype=dtype, non_blocking=True))
    leaf_cards = [float(x) for x in doc.leaf_cardinalities]
    return leaf_vecs, leaf_cards


def train_learned_model(
    model: LearnedMergeableSketch,
    train_docs: Sequence[CardinalityDocument],
    val_docs: Sequence[CardinalityDocument],
    *,
    n_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    c3_weight: float,
    leaf_weight: float,
    grad_clip_norm: float,
    audit_policy: AuditPolicyName,
    audit_fixed_nodes: int,
    audit_fraction: float,
    audit_scale: float,
    audit_include_root_query: bool,
    device: torch.device,
    seed: int,
) -> TrainDiagnostics:
    if len(train_docs) == 0:
        raise ValueError("train_docs must be non-empty")

    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    rng = random.Random(int(seed))

    train_loss_final = float("nan")
    val_loss_final = float("nan")

    idxs = list(range(len(train_docs)))
    for _ in range(int(n_epochs)):
        rng.shuffle(idxs)
        model.train()
        batch_losses: List[float] = []
        for b0 in range(0, len(idxs), int(batch_size)):
            batch_idx = idxs[b0 : b0 + int(batch_size)]
            opt.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=device, dtype=torch.float32)
            for i in batch_idx:
                doc = train_docs[i]
                leaf_vecs, leaf_cards = _to_device_leaf_tensors(doc, device=device)
                n_internal = int(max(0, len(leaf_vecs) - 1))
                n_audit = audit_sample_count(
                    n_internal,
                    policy=audit_policy,
                    fixed_nodes=int(audit_fixed_nodes),
                    fraction=float(audit_fraction),
                    scale=float(audit_scale),
                )
                if n_audit <= 0:
                    c3_audit_indices: Optional[set[int]] = set()
                elif n_audit >= n_internal:
                    c3_audit_indices = None
                else:
                    c3_audit_indices = set(rng.sample(range(n_internal), k=n_audit))
                out = model.forward_doc(
                    leaf_vecs,
                    leaf_cards,
                    schedule="balanced",
                    collect_c3=True,
                    collect_leaf=True,
                    c3_audit_indices=c3_audit_indices,
                )
                pred_norm = out["pred_norm"]
                true_norm = torch.tensor(
                    float(doc.true_cardinality) / model.target_scale,
                    device=device,
                    dtype=pred_norm.dtype,
                )
                if audit_include_root_query:
                    task_loss = F.mse_loss(pred_norm, true_norm, reduction="mean")
                else:
                    task_loss = torch.zeros((), device=device, dtype=pred_norm.dtype)
                doc_loss = (
                    task_loss
                    + float(c3_weight) * out["c3_loss"]
                    + float(leaf_weight) * out["leaf_loss"]
                )
                batch_loss = batch_loss + doc_loss
            batch_loss = batch_loss / float(len(batch_idx))
            batch_loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            opt.step()
            batch_losses.append(float(batch_loss.detach().cpu()))
        train_loss_final = float(np.mean(batch_losses))
        val_loss_final = evaluate_model_loss(
            model,
            val_docs,
            device=device,
            c3_weight=c3_weight,
            leaf_weight=leaf_weight,
            include_root_query=audit_include_root_query,
        )
    return TrainDiagnostics(
        train_loss_final=float(train_loss_final),
        val_loss_final=float(val_loss_final),
    )


@torch.no_grad()
def evaluate_model_loss(
    model: LearnedMergeableSketch,
    docs: Sequence[CardinalityDocument],
    *,
    device: torch.device,
    c3_weight: float,
    leaf_weight: float,
    include_root_query: bool,
) -> float:
    if len(docs) == 0:
        return 0.0
    model.eval()
    losses: List[float] = []
    for doc in docs:
        leaf_vecs, leaf_cards = _to_device_leaf_tensors(doc, device=device)
        out = model.forward_doc(
            leaf_vecs,
            leaf_cards,
            schedule="balanced",
            collect_c3=True,
            collect_leaf=True,
        )
        pred_norm = out["pred_norm"]
        true_norm = torch.tensor(
            float(doc.true_cardinality) / model.target_scale,
            device=device,
            dtype=pred_norm.dtype,
        )
        if include_root_query:
            task_loss = F.mse_loss(pred_norm, true_norm, reduction="mean")
        else:
            task_loss = torch.zeros((), device=device, dtype=pred_norm.dtype)
        total = (
            task_loss
            + float(c3_weight) * out["c3_loss"]
            + float(leaf_weight) * out["leaf_loss"]
        )
        losses.append(float(total.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_learned_model(
    model: LearnedMergeableSketch,
    docs: Sequence[CardinalityDocument],
    *,
    device: torch.device,
) -> ModelEvalMetrics:
    if len(docs) == 0:
        return ModelEvalMetrics(
            mae=0.0,
            rmse=0.0,
            relative_rmse=0.0,
            mean_rel_error=0.0,
            mean_abs_rel_error=0.0,
            schedule_spread_mean=0.0,
            schedule_spread_p95=0.0,
            c3_state_mse=0.0,
        )
    model.eval()

    abs_errs: List[float] = []
    sq_errs: List[float] = []
    rel_errs: List[float] = []
    rel_sq_errs: List[float] = []
    abs_rel_errs: List[float] = []
    spreads: List[float] = []
    c3_vals: List[float] = []

    for doc in docs:
        leaf_vecs, leaf_cards = _to_device_leaf_tensors(doc, device=device)
        preds: Dict[str, float] = {}
        c3_for_doc = 0.0
        for sched in VALID_SCHEDULES:
            out = model.forward_doc(
                leaf_vecs,
                leaf_cards,
                schedule=sched,
                collect_c3=True,
                collect_leaf=False,
            )
            pred_count = float(out["pred_count"].detach().cpu())
            preds[sched] = pred_count
            if sched == "balanced":
                c3_for_doc = float(out["c3_loss"].detach().cpu())

        pred = preds["balanced"]
        truth = float(doc.true_cardinality)
        err = pred - truth
        abs_errs.append(abs(err))
        sq_errs.append(err * err)
        rel = _safe_rel_error(pred, truth)
        rel_errs.append(rel)
        rel_sq_errs.append(rel * rel)
        abs_rel_errs.append(abs(rel))
        spread = max(preds.values()) - min(preds.values())
        spreads.append(spread)
        c3_vals.append(c3_for_doc)

    return ModelEvalMetrics(
        mae=float(np.mean(abs_errs)),
        rmse=float(math.sqrt(np.mean(sq_errs))),
        relative_rmse=float(math.sqrt(np.mean(rel_sq_errs))),
        mean_rel_error=float(np.mean(rel_errs)),
        mean_abs_rel_error=float(np.mean(abs_rel_errs)),
        schedule_spread_mean=float(np.mean(spreads)),
        schedule_spread_p95=float(np.percentile(np.array(spreads), 95.0)),
        c3_state_mse=float(np.mean(c3_vals)),
    )


def _hll_from_leaves(config: HLLConfig, leaf_token_lists: Sequence[Sequence[int]], schedule: ScheduleName) -> HyperLogLogSketch:
    if len(leaf_token_lists) == 0:
        return HyperLogLogSketch(config)
    leaf_sketches = [HyperLogLogSketch.from_tokens(config, toks) for toks in leaf_token_lists]

    if schedule == "balanced":
        cur = leaf_sketches
        while len(cur) > 1:
            nxt: List[HyperLogLogSketch] = []
            i = 0
            while i < len(cur):
                if i + 1 >= len(cur):
                    nxt.append(cur[i])
                    i += 1
                    continue
                merged = HyperLogLogSketch(config)
                merged.registers[:] = cur[i].registers
                merged.merge(cur[i + 1])
                nxt.append(merged)
                i += 2
            cur = nxt
        return cur[0]

    if schedule in ("left_to_right", "right_to_left"):
        if schedule == "left_to_right":
            order = leaf_sketches
        else:
            order = list(reversed(leaf_sketches))
        acc = HyperLogLogSketch(config)
        acc.registers[:] = order[0].registers
        for sk in order[1:]:
            acc.merge(sk)
        return acc

    raise ValueError(f"unsupported schedule: {schedule!r}")


def evaluate_hll_baseline(
    docs: Sequence[CardinalityDocument],
    *,
    precision: int,
    leaf_size: int,
    hash_bits: int = 64,
) -> HLLMetrics:
    if len(docs) == 0:
        cfg = HLLConfig(precision=int(precision), hash_bits=int(hash_bits))
        empty = HyperLogLogSketch(cfg)
        return HLLMetrics(
            precision=int(precision),
            registers=int(empty.m),
            register_bits=int(empty.register_bits),
            memory_bits=int(empty.memory_bits),
            memory_bytes=float(empty.memory_bits) / 8.0,
            mae=0.0,
            rmse=0.0,
            relative_rmse=0.0,
            mean_rel_error=0.0,
            mean_abs_rel_error=0.0,
            schedule_spread_mean=0.0,
            schedule_spread_p95=0.0,
        )

    cfg = HLLConfig(precision=int(precision), hash_bits=int(hash_bits))
    err_abs: List[float] = []
    err_sq: List[float] = []
    err_rel: List[float] = []
    err_rel_sq: List[float] = []
    err_abs_rel: List[float] = []
    spreads: List[float] = []

    for doc in docs:
        token_ids = tuple(int(x) for x in doc.token_ids)
        leaf_tokens = [
            token_ids[i : i + int(leaf_size)] for i in range(0, len(token_ids), int(leaf_size))
        ]
        ests: Dict[str, float] = {}
        for sched in VALID_SCHEDULES:
            sk = _hll_from_leaves(cfg, leaf_tokens, schedule=sched)
            ests[sched] = float(sk.estimate())

        pred = ests["balanced"]
        truth = float(doc.true_cardinality)
        diff = pred - truth
        err_abs.append(abs(diff))
        err_sq.append(diff * diff)
        rel = _safe_rel_error(pred, truth)
        err_rel.append(rel)
        err_rel_sq.append(rel * rel)
        err_abs_rel.append(abs(rel))
        spreads.append(max(ests.values()) - min(ests.values()))

    proto = HyperLogLogSketch(cfg)
    return HLLMetrics(
        precision=int(precision),
        registers=int(proto.m),
        register_bits=int(proto.register_bits),
        memory_bits=int(proto.memory_bits),
        memory_bytes=float(proto.memory_bits) / 8.0,
        mae=float(np.mean(err_abs)),
        rmse=float(math.sqrt(np.mean(err_sq))),
        relative_rmse=float(math.sqrt(np.mean(err_rel_sq))),
        mean_rel_error=float(np.mean(err_rel)),
        mean_abs_rel_error=float(np.mean(err_abs_rel)),
        schedule_spread_mean=float(np.mean(spreads)),
        schedule_spread_p95=float(np.percentile(np.array(spreads), 95.0)),
    )


def run_learning_vs_hll_experiment(config: SimulationConfig) -> ExperimentSummary:
    if len(config.state_dims) == 0:
        raise ValueError("state_dims must be non-empty")
    if len(config.train_sizes) == 0:
        raise ValueError("train_sizes must be non-empty")
    if config.max_tokens <= config.min_tokens:
        raise ValueError("max_tokens must exceed min_tokens")
    if config.leaf_size <= 0:
        raise ValueError("leaf_size must be positive")
    if str(config.audit_policy) not in VALID_AUDIT_POLICIES:
        raise ValueError(
            f"audit_policy={config.audit_policy!r} unsupported; "
            f"expected one of {VALID_AUDIT_POLICIES}"
        )
    if float(config.audit_fraction) <= 0.0:
        raise ValueError("audit_fraction must be positive")
    if float(config.audit_scale) <= 0.0:
        raise ValueError("audit_scale must be positive")
    if int(config.audit_fixed_nodes) < 0:
        raise ValueError("audit_fixed_nodes must be non-negative")

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

    max_train = max(int(x) for x in config.train_sizes)
    n_total = int(max_train + config.n_val + config.n_test)
    docs = generate_cardinality_documents(
        n_docs=n_total,
        universe_size=int(config.universe_size),
        min_tokens=int(config.min_tokens),
        max_tokens=int(config.max_tokens),
        leaf_size=int(config.leaf_size),
        zipf_alphas=config.zipf_alphas,
        seed=int(config.seed),
    )
    train_pool = docs[:max_train]
    val_docs = docs[max_train : max_train + int(config.n_val)]
    test_docs = docs[max_train + int(config.n_val) :]

    results: List[LearningRunSummary] = []
    hll_cache: Dict[int, HLLMetrics] = {}

    for state_dim in (int(x) for x in config.state_dims):
        learned_bits = int(state_dim * 32)
        p = match_hll_precision_for_bits(
            learned_bits,
            hash_bits=64,
            p_min=4,
            p_max=16,
        )
        if p not in hll_cache:
            hll_cache[p] = evaluate_hll_baseline(
                test_docs,
                precision=p,
                leaf_size=int(config.leaf_size),
                hash_bits=64,
            )
        hll_metrics = hll_cache[p]
        hll_rse_theory = float(hll_relative_standard_error(p))

        # Test-set cardinality stats (constant across train_sizes).
        _test_cards = np.array(
            [float(d.true_cardinality) for d in test_docs], dtype=np.float64
        )
        _test_card_mean = float(np.mean(_test_cards))
        _test_card_rms = float(math.sqrt(np.mean(_test_cards ** 2)))
        _floor_rmse = float(hll_rse_theory * _test_card_rms)
        _hll_emp_excess_rmse = float(hll_metrics.rmse - _floor_rmse)
        _hll_emp_excess_rel = float(hll_metrics.relative_rmse - hll_rse_theory)

        for train_size in (int(x) for x in config.train_sizes):
            model = LearnedMergeableSketch(
                input_dim=int(config.universe_size),
                state_dim=state_dim,
                hidden_dim=int(config.hidden_dim),
                target_scale=float(config.max_tokens),
            )
            train_docs = train_pool[:train_size]
            diag = train_learned_model(
                model,
                train_docs,
                val_docs,
                n_epochs=int(config.n_epochs),
                batch_size=int(config.batch_size),
                lr=float(config.lr),
                weight_decay=float(config.weight_decay),
                c3_weight=float(config.c3_weight),
                leaf_weight=float(config.leaf_weight),
                grad_clip_norm=float(config.grad_clip_norm),
                audit_policy=str(config.audit_policy),
                audit_fixed_nodes=int(config.audit_fixed_nodes),
                audit_fraction=float(config.audit_fraction),
                audit_scale=float(config.audit_scale),
                audit_include_root_query=bool(config.audit_include_root_query),
                device=device,
                seed=int(config.seed + 7919 + state_dim + train_size),
            )
            learned_metrics = evaluate_learned_model(
                model,
                test_docs,
                device=device,
            )
            geom = _summarize_audit_geometry(
                train_docs,
                policy=str(config.audit_policy),
                fixed_nodes=int(config.audit_fixed_nodes),
                fraction=float(config.audit_fraction),
                scale=float(config.audit_scale),
                include_root_query=bool(config.audit_include_root_query),
            )
            dist_floor = float(learned_metrics.relative_rmse - hll_rse_theory)
            dist_emp = float(learned_metrics.relative_rmse - hll_metrics.relative_rmse)
            results.append(
                LearningRunSummary(
                    state_dim=state_dim,
                    learned_memory_bits=learned_bits,
                    train_size=train_size,
                    val_loss_final=float(diag.val_loss_final),
                    train_loss_final=float(diag.train_loss_final),
                    learned_metrics=learned_metrics,
                    hll_metrics=hll_metrics,
                    hll_rse_theory=hll_rse_theory,
                    distance_to_hll_floor_rel_rmse=dist_floor,
                    distance_to_hll_empirical_rel_rmse=dist_emp,
                    train_mean_tokens=float(geom["mean_tokens"]),
                    train_mean_leaves=float(geom["mean_leaves"]),
                    train_mean_internal_nodes=float(geom["mean_internal_nodes"]),
                    train_audit_nodes_mean=float(geom["audit_nodes_mean"]),
                    train_audit_coverage_mean=float(geom["audit_coverage_mean"]),
                    train_root_queries_total=int(geom["root_queries_total"]),
                    train_audit_nodes_total=int(geom["audit_nodes_total"]),
                    train_total_queries_estimate=int(geom["total_queries_estimate"]),
                    rmse_gap_vs_hll=float(learned_metrics.rmse - hll_metrics.rmse),
                    abs_rel_error_gap_vs_hll=float(
                        learned_metrics.mean_abs_rel_error - hll_metrics.mean_abs_rel_error
                    ),
                    theoretical_floor_rmse=_floor_rmse,
                    excess_rmse=float(learned_metrics.rmse - _floor_rmse),
                    ratio_to_floor_rmse=float(
                        learned_metrics.rmse / max(1e-12, _floor_rmse)
                    ),
                    ratio_to_floor_rel_rmse=float(
                        learned_metrics.relative_rmse / max(1e-12, hll_rse_theory)
                    ),
                    hll_empirical_excess_rmse=_hll_emp_excess_rmse,
                    hll_empirical_excess_rel_rmse=_hll_emp_excess_rel,
                    test_cardinality_rms=_test_card_rms,
                    test_cardinality_mean=_test_card_mean,
                )
            )

    cfg_dict = asdict(config)
    cfg_dict["device_used"] = str(device)
    if device.type == "cuda":
        cfg_dict["cuda_current_device"] = int(torch.cuda.current_device())
        cfg_dict["cuda_device_name"] = str(
            torch.cuda.get_device_name(torch.cuda.current_device())
        )
    return ExperimentSummary(config=cfg_dict, results=tuple(results))


__all__ = [
    "VALID_AUDIT_POLICIES",
    "CardinalityDocument",
    "ExperimentSummary",
    "HLLConfig",
    "HLLMetrics",
    "LearningRunSummary",
    "LearnedMergeableSketch",
    "ModelEvalMetrics",
    "SimulationConfig",
    "VALID_SCHEDULES",
    "audit_sample_count",
    "compute_theoretical_floor_rmse",
    "evaluate_hll_baseline",
    "evaluate_learned_model",
    "generate_cardinality_documents",
    "hll_relative_standard_error",
    "match_hll_precision_for_bits",
    "run_learning_vs_hll_experiment",
    "train_learned_model",
]
