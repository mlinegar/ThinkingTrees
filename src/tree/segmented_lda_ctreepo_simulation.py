"""
Segmented-LDA end-to-end simulation for ThinkingTrees / C-TreePO.

This module is the "full decomposition" benchmark:

1) Upstream topic recovery / topic-word estimation (Tensor-LDA-inspired options):
   - `topic_phi_estimator="true"` uses the true topic-word matrix `φ`.
   - `topic_phi_estimator="noisy_theory"` perturbs `φ` with magnitude calibrated to a
     Lean-mirrored Thm-5.1-shaped `O(1/√N)` bound (simulation proxy for TLDA rates).
   - `topic_phi_estimator="tensor_lda"` estimates `φ̂` from unlabeled books via centered moments
     + whitening + tensor power method + recentering (batch baseline).
   - `topic_phi_estimator="online_tensor_lda"` estimates `φ̂` via burn-in whitening + STGD-style
     mini-batch factor updates (online baseline).
   - `topic_phi_estimator="spectral_numpy"` runs a lightweight spectral proxy on training leaves
     (center + SVD projection + k-means in spectral space).

2) Midstream summary-learning/calibration error:
   - Learn an affine calibration from queried leaves on training books.

3) Downstream merge/audit error:
   - Tree aggregation over leaf summaries with optional eval-time leaf/internal oracle guidance.

The simulation reports per-policy OPS-style local discrepancy metrics (C1/C3 proxies),
query accounting, selection-bias audit summaries, and an end-to-end triangle decomposition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import itertools
from statistics import fmean
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


from src.tree.segment_lda_ops_weight_recovery_simulation import (  # noqa: E402
    VALID_TOPIC_PHI_ESTIMATORS as _OPS_VALID_TOPIC_PHI_ESTIMATORS,
    estimate_topic_distributions,
)


TopicPhiEstimatorName = str
VALID_TOPIC_PHI_ESTIMATORS: Tuple[TopicPhiEstimatorName, ...] = tuple(_OPS_VALID_TOPIC_PHI_ESTIMATORS) + (
    "spectral_numpy",
)
VALID_CALIBRATION_POLICIES: Tuple[str, ...] = ("uniform", "entropy")
VALID_INTERNAL_QUERY_DESIGNS: Tuple[str, ...] = ("none", "uniform", "risk")


@dataclass(frozen=True)
class SegmentedLDACtreePOConfig:
    # Core LDA parameters.
    n_topics: int = 5
    vocab_size: int = 600
    alpha_topic: float = 0.20
    beta_word: float = 0.10

    # Segmentation DGP.
    n_books_train: int = 256
    n_books_test: int = 256
    min_segments: int = 8
    max_segments: int = 20
    min_seg_tokens: int = 24
    max_seg_tokens: int = 64
    segment_concentration: float = 80.0
    segment_background: float = 2.0

    # Leaf partition for C-TreePO aggregation.
    fixed_leaf_tokens: int = 32

    # Topic-word estimation (Tensor-LDA-inspired upstream step).
    topic_phi_estimator: TopicPhiEstimatorName = "noisy_theory"
    topic_phi_docs: int = 0  # if <=0, defaults to n_books_train for the estimator's effective N
    tlda_delta: float = 0.10
    tlda_rate_constant: float = 1.0
    tlda_sigmaK_floor: float = 1e-6
    topic_phi_permute: bool = True  # simulate topic unidentifiability (up to permutation)

    # Online Tensor-LDA knobs (used only when topic_phi_estimator="online_tensor_lda").
    online_tensor_lda_burn_in_docs: int = 0  # 0 => auto
    online_tensor_lda_batch_docs: int = 32
    online_tensor_lda_passes: int = 1
    online_tensor_lda_lr: float = 0.1
    online_tensor_lda_grad_clip_norm: float = 1.0

    # Lightweight spectral proxy knobs (used only when topic_phi_estimator="spectral_numpy").
    spectral_svd_dim_extra: int = 2
    spectral_max_leaves: int = 4000
    spectral_kmeans_inits: int = 6
    spectral_kmeans_max_iter: int = 60

    # Calibration from queried training leaves.
    calibration_leaf_query_rate: float = 0.10
    calibration_policy: str = "uniform"  # uniform|entropy
    calibration_ridge: float = 1e-4
    calibration_pi_min: float = 0.01

    # Evaluation-time oracle query budgets.
    eval_leaf_query_rate: float = 0.00
    eval_internal_query_rate: float = 0.00
    eval_internal_query_design: str = "none"  # none|uniform|risk

    # OPS discrepancy thresholds.
    c1_threshold: float = 0.20
    c3_threshold: float = 0.20

    # Optional selection-bias audit over internal-node discrepancy population.
    selection_audit_trials: int = 0
    selection_audit_sample_rate: float = 0.10
    selection_audit_pi_min: float = 0.01

    seed: int = 0


@dataclass(frozen=True)
class SegmentedBook:
    token_words: np.ndarray  # [T]
    token_topics: np.ndarray  # [T]
    boundaries: np.ndarray  # [B], cut-after indices
    book_topic_weights: np.ndarray  # [K]


@dataclass(frozen=True)
class SegmentedCorpus:
    topic_word_true: np.ndarray  # [K, V]
    books: Tuple[SegmentedBook, ...]


@dataclass(frozen=True)
class PolicyMetrics:
    n_books: int
    root_l1_mean: float
    root_l1_median: float
    root_l1_p95: float
    root_l2_mean: float
    c1_violation_rate: float
    c3_violation_rate: float
    mean_leaf_queries: float
    mean_internal_queries: float
    mean_total_queries: float


@dataclass(frozen=True)
class EndToEndDecompositionMetrics:
    n_books: int
    total_root_l1_mean: float
    topic_component_mean: float
    calibration_component_mean: float
    guidance_component_mean: float
    oracle_proxy_component_mean: float
    upper_bound_mean: float
    slack_mean: float


@dataclass(frozen=True)
class EstimatorStats:
    mean: float
    bias: float
    variance: float
    rmse: float


@dataclass(frozen=True)
class SelectionAuditSummary:
    n_units: int
    true_mean_discrepancy: float
    true_violation_rate: float
    trials: int
    target_sample_rate: float
    pi_min: float
    mean_sample_size: float
    mean_effective_sample_size: float
    naive_mean_discrepancy: EstimatorStats
    ipw_mean_discrepancy: EstimatorStats
    dsl0_mean_discrepancy: EstimatorStats
    dsl_oracle_mean_discrepancy: EstimatorStats
    naive_violation_rate: EstimatorStats
    ipw_violation_rate: EstimatorStats
    dsl0_violation_rate: EstimatorStats
    dsl_oracle_violation_rate: EstimatorStats
    ipw_violation_ci_coverage: float
    ipw_violation_ci_mean_radius: float


@dataclass(frozen=True)
class SegmentedLDACtreePOSummary:
    config: Dict[str, object]
    topic_meta: Dict[str, object]
    calibration_samples: int
    metrics: Dict[str, PolicyMetrics]
    decomposition: EndToEndDecompositionMetrics
    selection_audit: Optional[SelectionAuditSummary]

    def to_json(self) -> str:
        payload = {
            "config": self.config,
            "topic_meta": self.topic_meta,
            "calibration_samples": int(self.calibration_samples),
            "metrics": {k: asdict(v) for k, v in self.metrics.items()},
            "decomposition": asdict(self.decomposition),
            "selection_audit": asdict(self.selection_audit) if self.selection_audit is not None else None,
        }
        return json.dumps(payload, indent=2, sort_keys=True)


@dataclass
class _TreeNode:
    est: np.ndarray
    truth: np.ndarray
    leaves: int


def _safe_mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(fmean(vals))


def _median(xs: Sequence[float]) -> float:
    vals = np.asarray([float(x) for x in xs if math.isfinite(float(x))], dtype=np.float64)
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def _p95(xs: Sequence[float]) -> float:
    vals = np.asarray([float(x) for x in xs if math.isfinite(float(x))], dtype=np.float64)
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, 0.95))


def _l1(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(u, dtype=np.float64) - np.asarray(v, dtype=np.float64))))


def _l2(u: np.ndarray, v: np.ndarray) -> float:
    d = np.asarray(u, dtype=np.float64) - np.asarray(v, dtype=np.float64)
    return float(np.sqrt(np.sum(d * d)))


def _normalize_simplex_vec(x: np.ndarray) -> np.ndarray:
    y = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    s = float(np.sum(y))
    if not math.isfinite(s) or s <= 0.0:
        return np.full_like(y, 1.0 / float(y.size), dtype=np.float64)
    return y / s


def _normalize_simplex_rows(x: np.ndarray) -> np.ndarray:
    y = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    s = np.sum(y, axis=1, keepdims=True)
    out = np.zeros_like(y, dtype=np.float64)
    good = (s[:, 0] > 0.0) & np.isfinite(s[:, 0])
    if np.any(good):
        out[good] = y[good] / s[good]
    if np.any(~good):
        out[~good] = 1.0 / float(y.shape[1])
    return out


def _inclusion_probs_from_scores(scores: np.ndarray, *, target_rate: float, pi_min: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    target_rate = float(max(pi_min, min(1.0, target_rate)))
    pi_min = float(max(1e-9, min(1.0, pi_min)))
    if scores.size == 0:
        return np.zeros((0,), dtype=np.float64)
    s = scores.copy()
    s -= np.min(s)
    if float(np.max(s)) > 0.0:
        s /= float(np.max(s))
    c = float(max(0.0, target_rate - pi_min))
    pi = pi_min + c * s
    if float(np.mean(pi)) > target_rate:
        lo = 0.0
        hi = max(c, 1.0)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            cur = np.clip(pi_min + mid * s, pi_min, 1.0)
            if float(np.mean(cur)) > target_rate:
                hi = mid
            else:
                lo = mid
        pi = np.clip(pi_min + lo * s, pi_min, 1.0)
    return np.asarray(pi, dtype=np.float64)


def _bernoulli_sample(pi: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    pi = np.asarray(pi, dtype=np.float64)
    return rng.random(size=pi.shape) < pi


def _effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    if s2 <= 0.0:
        return 0.0
    return float((s1 * s1) / s2)


def _estimator_stats(estimates: Sequence[float], *, truth: float) -> EstimatorStats:
    vals = np.asarray([float(x) for x in estimates if math.isfinite(float(x))], dtype=np.float64)
    if vals.size == 0:
        return EstimatorStats(mean=float("nan"), bias=float("nan"), variance=float("nan"), rmse=float("nan"))
    mean = float(np.mean(vals))
    bias = float(mean - truth)
    var = float(np.var(vals))
    rmse = float(np.sqrt(np.mean((vals - truth) ** 2)))
    return EstimatorStats(mean=mean, bias=bias, variance=var, rmse=rmse)


def _validate_config(config: SegmentedLDACtreePOConfig) -> None:
    if config.n_topics < 2:
        raise ValueError("n_topics must be >= 2")
    if config.vocab_size < config.n_topics:
        raise ValueError("vocab_size must be >= n_topics")
    if config.alpha_topic <= 0 or config.beta_word <= 0:
        raise ValueError("alpha_topic and beta_word must be > 0")
    if config.n_books_train < 1 or config.n_books_test < 1:
        raise ValueError("n_books_train and n_books_test must be >= 1")
    if config.min_segments < 1 or config.max_segments < config.min_segments:
        raise ValueError("invalid segment bounds")
    if config.min_seg_tokens < 2 or config.max_seg_tokens < config.min_seg_tokens:
        raise ValueError("invalid segment token bounds")
    if config.segment_concentration <= 0 or config.segment_background <= 0:
        raise ValueError("segment_concentration and segment_background must be > 0")
    if config.fixed_leaf_tokens < 2:
        raise ValueError("fixed_leaf_tokens must be >= 2")
    if str(config.topic_phi_estimator) not in VALID_TOPIC_PHI_ESTIMATORS:
        raise ValueError(f"topic_phi_estimator must be one of {VALID_TOPIC_PHI_ESTIMATORS}")
    if config.topic_phi_docs < 0:
        raise ValueError("topic_phi_docs must be >= 0")
    if not (0.0 < float(config.tlda_delta) < 1.0):
        raise ValueError("tlda_delta must be in (0, 1)")
    if float(config.tlda_rate_constant) <= 0:
        raise ValueError("tlda_rate_constant must be > 0")
    if float(config.tlda_sigmaK_floor) <= 0:
        raise ValueError("tlda_sigmaK_floor must be > 0")
    if config.online_tensor_lda_burn_in_docs < 0:
        raise ValueError("online_tensor_lda_burn_in_docs must be >= 0")
    if config.online_tensor_lda_batch_docs < 1:
        raise ValueError("online_tensor_lda_batch_docs must be >= 1")
    if config.online_tensor_lda_passes < 1:
        raise ValueError("online_tensor_lda_passes must be >= 1")
    if float(config.online_tensor_lda_lr) <= 0:
        raise ValueError("online_tensor_lda_lr must be > 0")
    if float(config.online_tensor_lda_grad_clip_norm) <= 0:
        raise ValueError("online_tensor_lda_grad_clip_norm must be > 0")
    if config.spectral_svd_dim_extra < 0:
        raise ValueError("spectral_svd_dim_extra must be >= 0")
    if config.spectral_max_leaves < 1:
        raise ValueError("spectral_max_leaves must be >= 1")
    if config.spectral_kmeans_inits < 1:
        raise ValueError("spectral_kmeans_inits must be >= 1")
    if config.spectral_kmeans_max_iter < 1:
        raise ValueError("spectral_kmeans_max_iter must be >= 1")
    if not (0.0 <= config.calibration_leaf_query_rate <= 1.0):
        raise ValueError("calibration_leaf_query_rate must be in [0, 1]")
    if config.calibration_policy not in VALID_CALIBRATION_POLICIES:
        raise ValueError(f"calibration_policy must be one of {VALID_CALIBRATION_POLICIES}")
    if config.calibration_ridge < 0:
        raise ValueError("calibration_ridge must be >= 0")
    if not (0.0 < config.calibration_pi_min <= 1.0):
        raise ValueError("calibration_pi_min must be in (0, 1]")
    if not (0.0 <= config.eval_leaf_query_rate <= 1.0):
        raise ValueError("eval_leaf_query_rate must be in [0, 1]")
    if not (0.0 <= config.eval_internal_query_rate <= 1.0):
        raise ValueError("eval_internal_query_rate must be in [0, 1]")
    if config.eval_internal_query_design not in VALID_INTERNAL_QUERY_DESIGNS:
        raise ValueError(f"eval_internal_query_design must be one of {VALID_INTERNAL_QUERY_DESIGNS}")
    if config.c1_threshold < 0 or config.c3_threshold < 0:
        raise ValueError("c1_threshold and c3_threshold must be >= 0")
    if config.selection_audit_trials < 0:
        raise ValueError("selection_audit_trials must be >= 0")
    if not (0.0 <= config.selection_audit_sample_rate <= 1.0):
        raise ValueError("selection_audit_sample_rate must be in [0, 1]")
    if not (0.0 < config.selection_audit_pi_min <= 1.0):
        raise ValueError("selection_audit_pi_min must be in (0, 1]")


def _sample_topic_word_matrix(config: SegmentedLDACtreePOConfig, *, rng: np.random.Generator) -> np.ndarray:
    beta = np.full((int(config.vocab_size),), float(config.beta_word), dtype=np.float64)
    return np.asarray(rng.dirichlet(beta, size=int(config.n_topics)), dtype=np.float64)


def _sample_segmented_book(
    config: SegmentedLDACtreePOConfig,
    *,
    topic_word_true: np.ndarray,
    rng: np.random.Generator,
) -> SegmentedBook:
    k = int(config.n_topics)
    alpha = np.full((k,), float(config.alpha_topic), dtype=np.float64)
    w_book = np.asarray(rng.dirichlet(alpha), dtype=np.float64)

    n_seg = int(rng.integers(int(config.min_segments), int(config.max_segments) + 1))
    seg_lens = rng.integers(int(config.min_seg_tokens), int(config.max_seg_tokens) + 1, size=n_seg, dtype=np.int64)
    seg_lens = [int(x) for x in seg_lens]

    token_words: List[int] = []
    token_topics: List[int] = []
    boundaries: List[int] = []

    for seg_idx, seg_len in enumerate(seg_lens):
        dominant = int(rng.choice(np.arange(k), p=w_book))
        dir_param = (
            float(config.segment_background) * w_book
            + float(config.segment_concentration) * np.eye(k, dtype=np.float64)[dominant]
            + 1e-9
        )
        theta_seg = np.asarray(rng.dirichlet(dir_param), dtype=np.float64)

        z = np.asarray(rng.choice(np.arange(k), size=seg_len, p=theta_seg), dtype=np.int64)
        token_topics.extend(int(t) for t in z)
        for t in z:
            w = int(rng.choice(np.arange(topic_word_true.shape[1]), p=topic_word_true[int(t)]))
            token_words.append(w)

        if seg_idx < n_seg - 1:
            boundaries.append(len(token_words) - 1)

    return SegmentedBook(
        token_words=np.asarray(token_words, dtype=np.int64),
        token_topics=np.asarray(token_topics, dtype=np.int64),
        boundaries=np.asarray(boundaries, dtype=np.int64),
        book_topic_weights=np.asarray(w_book, dtype=np.float64),
    )


def _generate_segmented_corpus(
    config: SegmentedLDACtreePOConfig,
    *,
    topic_word_true: np.ndarray,
    n_books: int,
    rng: np.random.Generator,
) -> SegmentedCorpus:
    books = tuple(_sample_segmented_book(config, topic_word_true=topic_word_true, rng=rng) for _ in range(int(n_books)))
    return SegmentedCorpus(topic_word_true=np.asarray(topic_word_true, dtype=np.float64), books=books)


def _leaf_spans(n_tokens: int, *, leaf_tokens: int) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    i = 0
    while i < int(n_tokens):
        j = min(int(n_tokens), i + int(leaf_tokens))
        spans.append((int(i), int(j)))
        i = j
    if not spans:
        spans = [(0, 0)]
    return spans


def _span_topic_theta(token_topics: np.ndarray, *, start: int, end: int, n_topics: int) -> np.ndarray:
    if int(end) <= int(start):
        return np.full((int(n_topics),), 1.0 / float(n_topics), dtype=np.float64)
    z = np.asarray(token_topics[int(start) : int(end)], dtype=np.int64)
    c = np.bincount(z, minlength=int(n_topics)).astype(np.float64)
    return _normalize_simplex_vec(c)


def _span_word_counts(token_words: np.ndarray, *, start: int, end: int, vocab_size: int) -> np.ndarray:
    if int(end) <= int(start):
        return np.zeros((int(vocab_size),), dtype=np.float64)
    w = np.asarray(token_words[int(start) : int(end)], dtype=np.int64)
    c = np.bincount(w, minlength=int(vocab_size)).astype(np.float64)
    return np.asarray(c, dtype=np.float64)


def _estimate_theta_from_counts(counts: np.ndarray, *, topic_word_est: np.ndarray) -> np.ndarray:
    x = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(x))
    k = int(topic_word_est.shape[0])
    if total <= 0.0:
        return np.full((k,), 1.0 / float(k), dtype=np.float64)
    freq = x / total
    raw, *_ = np.linalg.lstsq(topic_word_est.T, freq, rcond=None)
    return _normalize_simplex_vec(np.asarray(raw, dtype=np.float64))


def _collect_train_leaf_count_matrix(
    books: Sequence[SegmentedBook],
    *,
    vocab_size: int,
    leaf_tokens: int,
    max_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    rows: List[np.ndarray] = []
    for book in books:
        spans = _leaf_spans(len(book.token_words), leaf_tokens=leaf_tokens)
        for (s, e) in spans:
            rows.append(_span_word_counts(book.token_words, start=s, end=e, vocab_size=vocab_size))
    if not rows:
        return np.zeros((0, int(vocab_size)), dtype=np.float64)
    x = np.asarray(rows, dtype=np.float64)
    n = int(x.shape[0])
    if n > int(max_rows):
        idx = rng.choice(np.arange(n, dtype=np.int64), size=int(max_rows), replace=False)
        x = np.asarray(x[idx], dtype=np.float64)
    return x


def _kmeans_lloyd(
    x: np.ndarray,
    *,
    k: int,
    n_init: int,
    max_iter: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    if n == 0:
        return np.zeros((int(k), int(d)), dtype=np.float64), np.zeros((0,), dtype=np.int64)

    best_inertia = float("inf")
    best_centers = np.zeros((int(k), int(d)), dtype=np.float64)
    best_labels = np.zeros((n,), dtype=np.int64)

    for _ in range(int(max(1, n_init))):
        if n >= int(k):
            init_ids = rng.choice(np.arange(n, dtype=np.int64), size=int(k), replace=False)
        else:
            init_ids = rng.choice(np.arange(n, dtype=np.int64), size=int(k), replace=True)
        centers = np.asarray(x[init_ids], dtype=np.float64).copy()
        labels_prev: Optional[np.ndarray] = None

        for _it in range(int(max(1, max_iter))):
            # Squared Euclidean distances.
            dist2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dist2, axis=1).astype(np.int64)
            if labels_prev is not None and np.array_equal(labels, labels_prev):
                break
            labels_prev = labels

            for j in range(int(k)):
                idx = np.where(labels == j)[0]
                if idx.size == 0:
                    centers[j] = x[int(rng.integers(0, n))]
                else:
                    centers[j] = np.mean(x[idx], axis=0)

        final_dist2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        inertia = float(np.sum(np.min(final_dist2, axis=1)))
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = np.asarray(centers, dtype=np.float64).copy()
            best_labels = np.argmin(final_dist2, axis=1).astype(np.int64)

    return best_centers, best_labels


def _estimate_topic_word_matrix_spectral_numpy(
    config: SegmentedLDACtreePOConfig,
    *,
    train_books: Sequence[SegmentedBook],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, object]]:
    k = int(config.n_topics)
    v = int(config.vocab_size)
    x_counts = _collect_train_leaf_count_matrix(
        train_books,
        vocab_size=v,
        leaf_tokens=int(config.fixed_leaf_tokens),
        max_rows=int(config.spectral_max_leaves),
        rng=rng,
    )
    spectral_meta: Dict[str, object] = {
        "topic_phi_estimator": "spectral_numpy",
        "spectral_numpy_leaf_rows": int(x_counts.shape[0]),
        "spectral_numpy_svd_dim_extra": int(config.spectral_svd_dim_extra),
        "spectral_numpy_max_leaves": int(config.spectral_max_leaves),
        "spectral_numpy_kmeans_inits": int(config.spectral_kmeans_inits),
        "spectral_numpy_kmeans_max_iter": int(config.spectral_kmeans_max_iter),
    }
    if x_counts.shape[0] == 0:
        return np.full((k, v), 1.0 / float(v), dtype=np.float64), spectral_meta

    row_sum = np.sum(x_counts, axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, 1.0)
    x = x_counts / row_sum
    m1 = np.mean(x, axis=0)
    xc = x - m1[None, :]

    if float(np.linalg.norm(xc)) < 1e-12:
        noisy = np.maximum(m1[None, :] + rng.normal(0.0, 1e-6, size=(k, v)), 1e-12)
        return _normalize_simplex_rows(noisy), spectral_meta

    d = int(
        min(
            max(1, k + int(config.spectral_svd_dim_extra)),
            xc.shape[0],
            xc.shape[1],
        )
    )
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    del u
    sd = np.asarray(s[:d], dtype=np.float64)
    vd = np.asarray(vt[:d, :], dtype=np.float64)
    eps = 1e-8

    x_proj = xc @ vd.T
    x_white = x_proj / np.maximum(sd[None, :], eps)
    centers_w, _labels = _kmeans_lloyd(
        x_white,
        k=k,
        n_init=int(config.spectral_kmeans_inits),
        max_iter=int(config.spectral_kmeans_max_iter),
        rng=rng,
    )
    centers_proj = centers_w * np.maximum(sd[None, :], eps)
    topics = centers_proj @ vd + m1[None, :]
    topics = np.maximum(topics, 1e-12)
    spectral_meta["spectral_numpy_svd_dim"] = int(d)
    return _normalize_simplex_rows(topics), spectral_meta


def _best_topic_permutation_l2(
    topics_est: Sequence[np.ndarray],
    topics_true: Sequence[np.ndarray],
) -> Tuple[Tuple[int, ...], np.ndarray]:
    """
    Find σ : est_index ↦ true_index minimizing Σ_i ||φ̂_i - φ_{σ(i)}||₂.

    Returns (perm, cost_matrix) where perm[i]=σ(i).
    """

    k = int(len(topics_true))
    if int(len(topics_est)) != k:
        raise ValueError("topics_est and topics_true must have same length")
    if k <= 0:
        return (tuple(), np.zeros((0, 0), dtype=np.float64))

    est = np.stack([np.asarray(t, dtype=np.float64).reshape(-1) for t in topics_est], axis=0)
    tru = np.stack([np.asarray(t, dtype=np.float64).reshape(-1) for t in topics_true], axis=0)
    if est.shape != tru.shape:
        raise ValueError("topics_est and topics_true must have aligned shapes")

    cost = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        diff = tru[None, :, :] - est[i : i + 1, None, :]
        cost[i] = np.linalg.norm(diff.reshape(k, -1), axis=1)

    # Exact assignment via brute force for small K.
    if k <= 9:
        best_perm: Tuple[int, ...] = tuple(range(k))
        best = float("inf")
        for perm in itertools.permutations(range(k)):
            total = 0.0
            for i, j in enumerate(perm):
                total += float(cost[i, j])
                if total >= best:
                    break
            if total < best:
                best = total
                best_perm = tuple(int(x) for x in perm)
        return best_perm, cost

    # Greedy fallback for large K (kept simple; this is a simulation helper).
    remaining = set(range(k))
    perm_out: List[int] = [-1 for _ in range(k)]
    for i in range(k):
        j = min(remaining, key=lambda jj: float(cost[i, jj]))
        perm_out[i] = int(j)
        remaining.remove(j)
    return (tuple(perm_out), cost)


def _invert_perm(perm: Sequence[int]) -> Tuple[int, ...]:
    k = int(len(perm))
    inv = [-1 for _ in range(k)]
    for i, j in enumerate(perm):
        jj = int(j)
        if jj < 0 or jj >= k:
            raise ValueError("perm must be a bijection on [0,K)")
        inv[jj] = int(i)
    if any(x < 0 for x in inv):
        raise ValueError("perm must be a bijection on [0,K)")
    return tuple(int(x) for x in inv)


def _estimate_topic_word_matrix(
    config: SegmentedLDACtreePOConfig,
    *,
    topic_word_true: np.ndarray,
    train_books: Sequence[SegmentedBook],
    n_train_docs: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, object]]:
    est = str(config.topic_phi_estimator).strip().lower()
    topics_true = [np.asarray(row, dtype=np.float64).reshape(-1) for row in np.asarray(topic_word_true, dtype=np.float64)]
    k = int(len(topics_true))
    if k <= 0:
        raise ValueError("need at least one topic")

    # Spectral proxy (leaf-count SVD + kmeans).
    if est == "spectral_numpy":
        topic_word_est, spectral_meta = _estimate_topic_word_matrix_spectral_numpy(config, train_books=train_books, rng=rng)
        topics_est = [np.asarray(row, dtype=np.float64).reshape(-1) for row in np.asarray(topic_word_est, dtype=np.float64)]
        perm_est_to_true, cost = _best_topic_permutation_l2(topics_est, topics_true)
        aligned_err = np.asarray([float(cost[i, perm_est_to_true[i]]) for i in range(k)], dtype=np.float64)
        meta: Dict[str, object] = {
            **spectral_meta,
            "topic_phi_perm_est_to_true": [int(x) for x in perm_est_to_true],
            "topic_phi_l2_error_mean": float(np.mean(aligned_err)) if aligned_err.size else 0.0,
            "topic_phi_l2_error_p95": float(np.percentile(aligned_err, 95.0)) if aligned_err.size else 0.0,
            "topic_phi_l2_error_max": float(np.max(aligned_err)) if aligned_err.size else 0.0,
        }
        inv = _invert_perm(perm_est_to_true)
        aligned = np.asarray(topic_word_est, dtype=np.float64)[np.asarray(inv, dtype=np.int64)]
        return aligned, meta

    # Tensor-LDA / noisy-theory / oracle baselines via shared estimator.
    phi_docs_effective = int(config.topic_phi_docs) if int(config.topic_phi_docs) > 0 else int(n_train_docs)
    phi_docs_effective = int(max(0, phi_docs_effective))

    docs_phi: List[np.ndarray] = [np.asarray(b.token_words, dtype=np.int64) for b in train_books]
    phi_extra = int(max(0, phi_docs_effective - len(docs_phi)))
    for _ in range(phi_extra):
        extra = _sample_segmented_book(config, topic_word_true=np.asarray(topic_word_true, dtype=np.float64), rng=rng)
        docs_phi.append(np.asarray(extra.token_words, dtype=np.int64))
    docs_phi = docs_phi[:phi_docs_effective]

    topics_est, meta_raw, perm_est_to_true = estimate_topic_distributions(
        topics_true,
        estimator=est,
        n_docs=int(max(1, phi_docs_effective)) if est != "true" else int(phi_docs_effective),
        doc_topic_concentration=float(config.alpha_topic),
        tlda_delta=float(config.tlda_delta),
        tlda_rate_constant=float(config.tlda_rate_constant),
        sigmaK_floor=float(config.tlda_sigmaK_floor),
        permute=bool(config.topic_phi_permute),
        seed=int(rng.integers(0, 2**31 - 1)),
        docs_tokens=[d.tolist() for d in docs_phi] if docs_phi else None,
        online_burn_in_docs=int(config.online_tensor_lda_burn_in_docs),
        online_batch_docs=int(config.online_tensor_lda_batch_docs),
        online_passes=int(config.online_tensor_lda_passes),
        online_lr=float(config.online_tensor_lda_lr),
        online_grad_clip_norm=float(config.online_tensor_lda_grad_clip_norm),
    )
    meta: Dict[str, object] = dict(meta_raw)
    meta["topic_phi_perm_est_to_true"] = [int(x) for x in perm_est_to_true]

    inv = _invert_perm(perm_est_to_true)
    aligned_topics = tuple(np.asarray(topics_est[int(i)], dtype=np.float64).reshape(-1) for i in inv)
    topic_word_est = np.stack(aligned_topics, axis=0).astype(np.float64, copy=False)
    return topic_word_est, meta


def _sample_leaf_query_mask(
    proxy_leaf_thetas: np.ndarray,
    *,
    rate: float,
    policy: str,
    pi_min: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    b, c, _ = proxy_leaf_thetas.shape
    rate = float(max(0.0, min(1.0, rate)))
    pi_min = float(max(1e-9, min(1.0, pi_min)))
    if rate <= 0.0:
        return np.zeros((b, c), dtype=bool), np.zeros((b, c), dtype=np.float64)
    if policy == "uniform":
        pi = np.full((b, c), float(max(rate, pi_min)), dtype=np.float64)
    elif policy == "entropy":
        p = np.clip(np.asarray(proxy_leaf_thetas, dtype=np.float64), 1e-12, 1.0)
        entropy = -np.sum(p * np.log(p), axis=2)
        pi = _inclusion_probs_from_scores(entropy.reshape(-1), target_rate=rate, pi_min=pi_min).reshape(b, c)
    else:
        raise ValueError(f"unknown calibration policy: {policy}")
    return np.asarray(_bernoulli_sample(pi, rng=rng), dtype=bool), np.asarray(pi, dtype=np.float64)


def _fit_affine_calibration(
    proxy_leaf_thetas: np.ndarray,
    true_leaf_thetas: np.ndarray,
    queried_mask: np.ndarray,
    *,
    ridge: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    x = np.asarray(proxy_leaf_thetas, dtype=np.float64)[queried_mask]
    y = np.asarray(true_leaf_thetas, dtype=np.float64)[queried_mask]
    k = int(proxy_leaf_thetas.shape[2])
    n = int(x.shape[0])
    if n <= 0:
        return np.eye(k, dtype=np.float64), np.zeros((k,), dtype=np.float64), 0

    x1 = np.concatenate([x, np.ones((n, 1), dtype=np.float64)], axis=1)
    gram = x1.T @ x1
    lam = float(max(0.0, ridge))
    if lam > 0.0:
        reg = lam * np.eye(k + 1, dtype=np.float64)
        reg[-1, -1] = 0.0
        gram = gram + reg
    rhs = x1.T @ y
    coef, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
    w = np.asarray(coef[:k, :], dtype=np.float64)
    b = np.asarray(coef[k, :], dtype=np.float64)
    return w, b, n


def _apply_affine_calibration(theta: np.ndarray, *, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    z = np.asarray(theta, dtype=np.float64)
    flat = z.reshape(-1, z.shape[2])
    mapped = flat @ np.asarray(w, dtype=np.float64) + np.asarray(b, dtype=np.float64)
    mapped = _normalize_simplex_rows(mapped)
    return mapped.reshape(z.shape)


def _reduce_balanced_tree_with_guidance(
    leaf_est: np.ndarray,
    leaf_truth: np.ndarray,
    *,
    leaf_query_rate: float,
    internal_query_rate: float,
    internal_query_design: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[float], List[float], int, int, List[float], List[float]]:
    c, _k = leaf_est.shape
    leaf_query_rate = float(max(0.0, min(1.0, leaf_query_rate)))
    internal_query_rate = float(max(0.0, min(1.0, internal_query_rate)))

    leaf_mask = rng.random(size=(c,)) < leaf_query_rate
    est = np.asarray(leaf_est, dtype=np.float64).copy()
    est[leaf_mask] = np.asarray(leaf_truth, dtype=np.float64)[leaf_mask]

    c1_errors = [_l1(est[i], leaf_truth[i]) for i in range(c)]
    leaf_queries = int(np.sum(leaf_mask))
    internal_queries = 0
    c3_errors: List[float] = []
    internal_population_errors: List[float] = []
    internal_population_scores: List[float] = []

    nodes: List[_TreeNode] = [
        _TreeNode(est=est[i].copy(), truth=np.asarray(leaf_truth[i], dtype=np.float64).copy(), leaves=1) for i in range(c)
    ]

    while len(nodes) > 1:
        next_nodes: List[_TreeNode] = []
        merged_flags: List[bool] = []
        merge_scores: List[float] = []

        i = 0
        while i < len(nodes):
            if i + 1 >= len(nodes):
                next_nodes.append(nodes[i])
                merged_flags.append(False)
                merge_scores.append(float("-inf"))
                i += 1
                continue

            left = nodes[i]
            right = nodes[i + 1]
            n = int(left.leaves + right.leaves)
            est_merge = (left.est * float(left.leaves) + right.est * float(right.leaves)) / float(n)
            truth_merge = (left.truth * float(left.leaves) + right.truth * float(right.leaves)) / float(n)
            pre_err = _l1(est_merge, truth_merge)
            score = _l1(left.est, right.est)
            internal_population_errors.append(pre_err)
            internal_population_scores.append(score)

            next_nodes.append(_TreeNode(est=est_merge, truth=truth_merge, leaves=n))
            merged_flags.append(True)
            merge_scores.append(score)
            i += 2

        candidate_ids = [idx for idx, flag in enumerate(merged_flags) if flag]
        n_candidates = len(candidate_ids)
        selected: set[int] = set()
        if n_candidates > 0 and internal_query_rate > 0.0 and internal_query_design != "none":
            q = int(round(internal_query_rate * float(n_candidates)))
            q = max(0, min(n_candidates, q))
            if q > 0:
                if internal_query_design == "uniform":
                    chosen = rng.choice(np.asarray(candidate_ids, dtype=np.int64), size=q, replace=False)
                    selected = {int(x) for x in np.asarray(chosen, dtype=np.int64)}
                elif internal_query_design == "risk":
                    ranked = sorted(candidate_ids, key=lambda idx: float(merge_scores[idx]), reverse=True)
                    selected = set(ranked[:q])
                else:
                    raise ValueError(f"unknown internal_query_design: {internal_query_design}")

        for idx in candidate_ids:
            node = next_nodes[idx]
            if idx in selected:
                node.est = node.truth.copy()
                internal_queries += 1
            c3_errors.append(_l1(node.est, node.truth))

        nodes = next_nodes

    root_est = _normalize_simplex_vec(nodes[0].est)
    return (
        root_est,
        c1_errors,
        c3_errors,
        leaf_queries,
        internal_queries,
        internal_population_errors,
        internal_population_scores,
    )


def _violation_rate(errs: Sequence[float], *, threshold: float) -> float:
    vals = [float(x) for x in errs if math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(np.mean(np.asarray(vals, dtype=np.float64) > float(threshold)))


def _extract_leaf_arrays(
    books: Sequence[SegmentedBook],
    *,
    n_topics: int,
    vocab_size: int,
    leaf_tokens: int,
    topic_word_est: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    # Returns per-book arrays:
    # - leaf_truth_theta[b]: [L_b, K]
    # - leaf_est_theta[b]: [L_b, K]
    # - leaf_counts[b]: [L_b, V]
    all_truth: List[np.ndarray] = []
    all_est: List[np.ndarray] = []
    all_counts: List[np.ndarray] = []
    for book in books:
        spans = _leaf_spans(len(book.token_words), leaf_tokens=leaf_tokens)
        truth_list: List[np.ndarray] = []
        est_list: List[np.ndarray] = []
        counts_list: List[np.ndarray] = []
        for (s, e) in spans:
            theta_truth = _span_topic_theta(book.token_topics, start=s, end=e, n_topics=n_topics)
            wc = _span_word_counts(book.token_words, start=s, end=e, vocab_size=vocab_size)
            theta_est = _estimate_theta_from_counts(wc, topic_word_est=topic_word_est)
            truth_list.append(theta_truth)
            est_list.append(theta_est)
            counts_list.append(wc)
        all_truth.append(np.asarray(truth_list, dtype=np.float64))
        all_est.append(np.asarray(est_list, dtype=np.float64))
        all_counts.append(np.asarray(counts_list, dtype=np.float64))
    return all_truth, all_est, all_counts


def _aggregate_root_truth(book: SegmentedBook, *, n_topics: int) -> np.ndarray:
    return _span_topic_theta(book.token_topics, start=0, end=len(book.token_topics), n_topics=n_topics)


def _build_policy_metrics(
    *,
    root_l1: Sequence[float],
    root_l2: Sequence[float],
    c1_errors: Sequence[float],
    c3_errors: Sequence[float],
    leaf_queries: Sequence[float],
    internal_queries: Sequence[float],
    c1_threshold: float,
    c3_threshold: float,
) -> PolicyMetrics:
    tot = [float(a + b) for a, b in zip(leaf_queries, internal_queries)]
    return PolicyMetrics(
        n_books=len(root_l1),
        root_l1_mean=_safe_mean(root_l1),
        root_l1_median=_median(root_l1),
        root_l1_p95=_p95(root_l1),
        root_l2_mean=_safe_mean(root_l2),
        c1_violation_rate=_violation_rate(c1_errors, threshold=float(c1_threshold)),
        c3_violation_rate=_violation_rate(c3_errors, threshold=float(c3_threshold)),
        mean_leaf_queries=_safe_mean(leaf_queries),
        mean_internal_queries=_safe_mean(internal_queries),
        mean_total_queries=_safe_mean(tot),
    )


def _run_selection_bias_audit(
    *,
    discrepancies: np.ndarray,
    violations: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    trials: int,
    sample_rate: float,
    pi_min: float,
    seed: int,
) -> SelectionAuditSummary:
    rng = np.random.default_rng(int(seed))
    disc = np.asarray(discrepancies, dtype=np.float64)
    viol = np.asarray(violations, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n = int(disc.size)
    if n == 0:
        nan_stats = EstimatorStats(mean=float("nan"), bias=float("nan"), variance=float("nan"), rmse=float("nan"))
        return SelectionAuditSummary(
            n_units=0,
            true_mean_discrepancy=float("nan"),
            true_violation_rate=float("nan"),
            trials=int(trials),
            target_sample_rate=float(sample_rate),
            pi_min=float(pi_min),
            mean_sample_size=float("nan"),
            mean_effective_sample_size=float("nan"),
            naive_mean_discrepancy=nan_stats,
            ipw_mean_discrepancy=nan_stats,
            dsl0_mean_discrepancy=nan_stats,
            dsl_oracle_mean_discrepancy=nan_stats,
            naive_violation_rate=nan_stats,
            ipw_violation_rate=nan_stats,
            dsl0_violation_rate=nan_stats,
            dsl_oracle_violation_rate=nan_stats,
            ipw_violation_ci_coverage=float("nan"),
            ipw_violation_ci_mean_radius=float("nan"),
        )

    pi = _inclusion_probs_from_scores(scores, target_rate=float(sample_rate), pi_min=float(pi_min))
    truth_mu = float(np.mean(disc))
    truth_p = float(np.mean(viol > float(threshold)))
    viol01 = (viol > float(threshold)).astype(np.float64)

    naive_mu: List[float] = []
    ipw_mu: List[float] = []
    dsl0_mu: List[float] = []
    dsl_oracle_mu: List[float] = []

    naive_p: List[float] = []
    ipw_p: List[float] = []
    dsl0_p: List[float] = []
    dsl_oracle_p: List[float] = []

    sample_sizes: List[float] = []
    ess_vals: List[float] = []
    ci_covered: List[float] = []
    ci_radius: List[float] = []

    pred0_mu = np.zeros_like(disc)
    pred0_p = np.zeros_like(viol01)
    pred_oracle_mu = disc.copy()
    pred_oracle_p = viol01.copy()

    for _ in range(int(trials)):
        idx = _bernoulli_sample(pi, rng=rng)
        m = int(np.sum(idx))
        if m <= 0:
            naive_mu.append(float("nan"))
            ipw_mu.append(float("nan"))
            dsl0_mu.append(float("nan"))
            dsl_oracle_mu.append(float("nan"))
            naive_p.append(float("nan"))
            ipw_p.append(float("nan"))
            dsl0_p.append(float("nan"))
            dsl_oracle_p.append(float("nan"))
            sample_sizes.append(0.0)
            ess_vals.append(0.0)
            ci_covered.append(float("nan"))
            ci_radius.append(float("nan"))
            continue

        w = idx.astype(np.float64) / pi
        ess = _effective_sample_size(w)
        sample_sizes.append(float(m))
        ess_vals.append(float(ess))

        naive_mu_t = float(np.mean(disc[idx]))
        naive_p_t = float(np.mean(viol01[idx]))
        ipw_mu_t = float(np.sum(w * disc) / float(n))
        ipw_p_t = float(np.sum(w * viol01) / float(n))
        dsl0_mu_t = float(np.mean(pred0_mu) + np.sum(w * (disc - pred0_mu)) / float(n))
        dsl0_p_t = float(np.mean(pred0_p) + np.sum(w * (viol01 - pred0_p)) / float(n))
        dsl_oracle_mu_t = float(np.mean(pred_oracle_mu) + np.sum(w * (disc - pred_oracle_mu)) / float(n))
        dsl_oracle_p_t = float(np.mean(pred_oracle_p) + np.sum(w * (viol01 - pred_oracle_p)) / float(n))

        naive_mu.append(naive_mu_t)
        ipw_mu.append(ipw_mu_t)
        dsl0_mu.append(dsl0_mu_t)
        dsl_oracle_mu.append(dsl_oracle_mu_t)
        naive_p.append(naive_p_t)
        ipw_p.append(ipw_p_t)
        dsl0_p.append(dsl0_p_t)
        dsl_oracle_p.append(dsl_oracle_p_t)

        rad = float(1.96 * math.sqrt(max(ipw_p_t * (1.0 - ipw_p_t), 1e-9) / max(ess, 1e-9)))
        ci_radius.append(rad)
        ci_covered.append(float(abs(ipw_p_t - truth_p) <= rad))

    return SelectionAuditSummary(
        n_units=int(n),
        true_mean_discrepancy=truth_mu,
        true_violation_rate=truth_p,
        trials=int(trials),
        target_sample_rate=float(sample_rate),
        pi_min=float(pi_min),
        mean_sample_size=_safe_mean(sample_sizes),
        mean_effective_sample_size=_safe_mean(ess_vals),
        naive_mean_discrepancy=_estimator_stats(naive_mu, truth=truth_mu),
        ipw_mean_discrepancy=_estimator_stats(ipw_mu, truth=truth_mu),
        dsl0_mean_discrepancy=_estimator_stats(dsl0_mu, truth=truth_mu),
        dsl_oracle_mean_discrepancy=_estimator_stats(dsl_oracle_mu, truth=truth_mu),
        naive_violation_rate=_estimator_stats(naive_p, truth=truth_p),
        ipw_violation_rate=_estimator_stats(ipw_p, truth=truth_p),
        dsl0_violation_rate=_estimator_stats(dsl0_p, truth=truth_p),
        dsl_oracle_violation_rate=_estimator_stats(dsl_oracle_p, truth=truth_p),
        ipw_violation_ci_coverage=_safe_mean(ci_covered),
        ipw_violation_ci_mean_radius=_safe_mean(ci_radius),
    )


def run_segmented_lda_ctreepo_simulation(
    config: SegmentedLDACtreePOConfig,
) -> SegmentedLDACtreePOSummary:
    _validate_config(config)
    rng = np.random.default_rng(int(config.seed))

    topic_word_true = _sample_topic_word_matrix(config, rng=rng)
    train = _generate_segmented_corpus(config, topic_word_true=topic_word_true, n_books=int(config.n_books_train), rng=rng)
    test = _generate_segmented_corpus(config, topic_word_true=topic_word_true, n_books=int(config.n_books_test), rng=rng)

    topic_word_est, topic_meta = _estimate_topic_word_matrix(
        config,
        topic_word_true=topic_word_true,
        train_books=train.books,
        n_train_docs=int(config.n_books_train),
        rng=rng,
    )

    # Leaf arrays for train/test under estimated topics and oracle topics.
    train_truth, train_est, _train_counts = _extract_leaf_arrays(
        train.books,
        n_topics=int(config.n_topics),
        vocab_size=int(config.vocab_size),
        leaf_tokens=int(config.fixed_leaf_tokens),
        topic_word_est=topic_word_est,
    )
    test_truth, test_est, _test_counts = _extract_leaf_arrays(
        test.books,
        n_topics=int(config.n_topics),
        vocab_size=int(config.vocab_size),
        leaf_tokens=int(config.fixed_leaf_tokens),
        topic_word_est=topic_word_est,
    )
    _test_truth2, test_oracle_proxy, _test_counts2 = _extract_leaf_arrays(
        test.books,
        n_topics=int(config.n_topics),
        vocab_size=int(config.vocab_size),
        leaf_tokens=int(config.fixed_leaf_tokens),
        topic_word_est=topic_word_true,
    )

    # Build train tensors (ragged -> padded stack for query sampling).
    max_train_leaves = max(arr.shape[0] for arr in train_est)
    k = int(config.n_topics)
    train_proxy_pad = np.zeros((len(train_est), max_train_leaves, k), dtype=np.float64)
    train_truth_pad = np.zeros((len(train_truth), max_train_leaves, k), dtype=np.float64)
    train_mask = np.zeros((len(train_est), max_train_leaves), dtype=bool)
    for i, (a, b) in enumerate(zip(train_est, train_truth)):
        l = a.shape[0]
        train_proxy_pad[i, :l] = a
        train_truth_pad[i, :l] = b
        train_mask[i, :l] = True

    query_mask_pad, _pi_train = _sample_leaf_query_mask(
        train_proxy_pad,
        rate=float(config.calibration_leaf_query_rate),
        policy=str(config.calibration_policy),
        pi_min=float(config.calibration_pi_min),
        rng=rng,
    )
    query_mask_pad = query_mask_pad & train_mask

    w_cal, b_cal, n_calib = _fit_affine_calibration(
        train_proxy_pad,
        train_truth_pad,
        query_mask_pad,
        ridge=float(config.calibration_ridge),
    )

    # Policy accumulators.
    policy_names = (
        "oracle_proxy",
        "estimated_uncalibrated",
        "estimated_calibrated",
        "estimated_calibrated_budgeted",
        "oracle_tree",
    )
    root_l1: Dict[str, List[float]] = {p: [] for p in policy_names}
    root_l2: Dict[str, List[float]] = {p: [] for p in policy_names}
    c1_err: Dict[str, List[float]] = {p: [] for p in policy_names}
    c3_err: Dict[str, List[float]] = {p: [] for p in policy_names}
    q_leaf: Dict[str, List[float]] = {p: [] for p in policy_names}
    q_internal: Dict[str, List[float]] = {p: [] for p in policy_names}

    # Decomposition components per book (L1 metric).
    decomp_total: List[float] = []
    decomp_topic: List[float] = []
    decomp_calib: List[float] = []
    decomp_guidance: List[float] = []
    decomp_oracle_proxy: List[float] = []
    decomp_upper: List[float] = []
    decomp_slack: List[float] = []

    audit_disc_population: List[float] = []
    audit_score_population: List[float] = []

    for i, book in enumerate(test.books):
        truth_root = _aggregate_root_truth(book, n_topics=int(config.n_topics))

        leaf_truth = np.asarray(test_truth[i], dtype=np.float64)
        leaf_est = np.asarray(test_est[i], dtype=np.float64)
        leaf_oracle_proxy = np.asarray(test_oracle_proxy[i], dtype=np.float64)

        # Policy A: oracle topics but still projected from words (oracle proxy baseline).
        root_op, c1_op, c3_op, lq_op, iq_op, _e0, _s0 = _reduce_balanced_tree_with_guidance(
            leaf_oracle_proxy,
            leaf_truth,
            leaf_query_rate=0.0,
            internal_query_rate=0.0,
            internal_query_design="none",
            rng=rng,
        )
        root_l1["oracle_proxy"].append(_l1(root_op, truth_root))
        root_l2["oracle_proxy"].append(_l2(root_op, truth_root))
        c1_err["oracle_proxy"].extend(c1_op)
        c3_err["oracle_proxy"].extend(c3_op)
        q_leaf["oracle_proxy"].append(float(lq_op))
        q_internal["oracle_proxy"].append(float(iq_op))

        # Policy B: estimated topics, uncalibrated.
        root_est_u, c1_u, c3_u, lq_u, iq_u, _e1, _s1 = _reduce_balanced_tree_with_guidance(
            leaf_est,
            leaf_truth,
            leaf_query_rate=0.0,
            internal_query_rate=0.0,
            internal_query_design="none",
            rng=rng,
        )
        root_l1["estimated_uncalibrated"].append(_l1(root_est_u, truth_root))
        root_l2["estimated_uncalibrated"].append(_l2(root_est_u, truth_root))
        c1_err["estimated_uncalibrated"].extend(c1_u)
        c3_err["estimated_uncalibrated"].extend(c3_u)
        q_leaf["estimated_uncalibrated"].append(float(lq_u))
        q_internal["estimated_uncalibrated"].append(float(iq_u))

        # Policy C: estimated topics + calibration.
        leaf_cal = _apply_affine_calibration(leaf_est[np.newaxis, :, :], w=w_cal, b=b_cal)[0]
        root_est_c, c1_c, c3_c, lq_c, iq_c, pop_e, pop_s = _reduce_balanced_tree_with_guidance(
            leaf_cal,
            leaf_truth,
            leaf_query_rate=0.0,
            internal_query_rate=0.0,
            internal_query_design="none",
            rng=rng,
        )
        root_l1["estimated_calibrated"].append(_l1(root_est_c, truth_root))
        root_l2["estimated_calibrated"].append(_l2(root_est_c, truth_root))
        c1_err["estimated_calibrated"].extend(c1_c)
        c3_err["estimated_calibrated"].extend(c3_c)
        q_leaf["estimated_calibrated"].append(float(lq_c))
        q_internal["estimated_calibrated"].append(float(iq_c))
        audit_disc_population.extend(float(x) for x in pop_e)
        audit_score_population.extend(float(x) for x in pop_s)

        # Policy D: estimated topics + calibration + eval-time oracle budget.
        root_est_b, c1_b, c3_b, lq_b, iq_b, _e2, _s2 = _reduce_balanced_tree_with_guidance(
            leaf_cal,
            leaf_truth,
            leaf_query_rate=float(config.eval_leaf_query_rate),
            internal_query_rate=float(config.eval_internal_query_rate),
            internal_query_design=str(config.eval_internal_query_design),
            rng=rng,
        )
        root_l1["estimated_calibrated_budgeted"].append(_l1(root_est_b, truth_root))
        root_l2["estimated_calibrated_budgeted"].append(_l2(root_est_b, truth_root))
        c1_err["estimated_calibrated_budgeted"].extend(c1_b)
        c3_err["estimated_calibrated_budgeted"].extend(c3_b)
        q_leaf["estimated_calibrated_budgeted"].append(float(lq_b))
        q_internal["estimated_calibrated_budgeted"].append(float(iq_b))

        # Policy E: oracle tree (true leaf summaries).
        root_l1["oracle_tree"].append(0.0)
        root_l2["oracle_tree"].append(0.0)
        c1_err["oracle_tree"].extend([0.0] * leaf_truth.shape[0])
        c3_err["oracle_tree"].extend([0.0] * max(0, leaf_truth.shape[0] - 1))
        q_leaf["oracle_tree"].append(float(leaf_truth.shape[0]))
        q_internal["oracle_tree"].append(float(max(0, leaf_truth.shape[0] - 1)))

        # End-to-end decomposition chain:
        # truth -> oracle_proxy -> estimated_uncalibrated -> estimated_calibrated -> estimated_calibrated_budgeted
        total = _l1(root_est_b, truth_root)
        comp_topic = _l1(root_est_u, root_op)
        comp_calib = _l1(root_est_c, root_est_u)
        comp_guidance = _l1(root_est_b, root_est_c)
        comp_oracle_proxy = _l1(root_op, truth_root)
        upper = comp_topic + comp_calib + comp_guidance + comp_oracle_proxy
        slack = upper - total

        decomp_total.append(total)
        decomp_topic.append(comp_topic)
        decomp_calib.append(comp_calib)
        decomp_guidance.append(comp_guidance)
        decomp_oracle_proxy.append(comp_oracle_proxy)
        decomp_upper.append(upper)
        decomp_slack.append(slack)

    metrics: Dict[str, PolicyMetrics] = {}
    for p in policy_names:
        metrics[p] = _build_policy_metrics(
            root_l1=root_l1[p],
            root_l2=root_l2[p],
            c1_errors=c1_err[p],
            c3_errors=c3_err[p],
            leaf_queries=q_leaf[p],
            internal_queries=q_internal[p],
            c1_threshold=float(config.c1_threshold),
            c3_threshold=float(config.c3_threshold),
        )

    decomposition = EndToEndDecompositionMetrics(
        n_books=int(config.n_books_test),
        total_root_l1_mean=_safe_mean(decomp_total),
        topic_component_mean=_safe_mean(decomp_topic),
        calibration_component_mean=_safe_mean(decomp_calib),
        guidance_component_mean=_safe_mean(decomp_guidance),
        oracle_proxy_component_mean=_safe_mean(decomp_oracle_proxy),
        upper_bound_mean=_safe_mean(decomp_upper),
        slack_mean=_safe_mean(decomp_slack),
    )

    selection_audit: Optional[SelectionAuditSummary] = None
    if int(config.selection_audit_trials) > 0 and len(audit_disc_population) > 0:
        disc = np.asarray(audit_disc_population, dtype=np.float64)
        viol = (disc > float(config.c3_threshold)).astype(np.float64)
        score = np.asarray(audit_score_population, dtype=np.float64)
        selection_audit = _run_selection_bias_audit(
            discrepancies=disc,
            violations=viol,
            scores=score,
            threshold=float(config.c3_threshold),
            trials=int(config.selection_audit_trials),
            sample_rate=float(config.selection_audit_sample_rate),
            pi_min=float(config.selection_audit_pi_min),
            seed=int(config.seed),
        )

    return SegmentedLDACtreePOSummary(
        config=asdict(config),
        topic_meta=topic_meta,
        calibration_samples=int(n_calib),
        metrics=metrics,
        decomposition=decomposition,
        selection_audit=selection_audit,
    )


__all__ = [
    "SegmentedLDACtreePOConfig",
    "SegmentedBook",
    "SegmentedCorpus",
    "PolicyMetrics",
    "EndToEndDecompositionMetrics",
    "EstimatorStats",
    "SelectionAuditSummary",
    "SegmentedLDACtreePOSummary",
    "VALID_TOPIC_PHI_ESTIMATORS",
    "run_segmented_lda_ctreepo_simulation",
]
