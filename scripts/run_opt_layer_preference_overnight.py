#!/usr/bin/env python3
"""Adaptive overnight preference-recovery optimization loop for ctreepo opt layer.

This runner stress-tests the generalized `g / f* / f_hat / pi` bridge in:
- Markov changepoint-count setting
- Mergeable spike-count setting
- Segment-LDA OPS setting

For each round it:
1) samples proxy-model hyperparameter trials per setting,
2) trains proxy `f_hat` via `collect_proxy_training_data`,
3) induces oracle preferences via `collect_pairwise_preferences`,
4) evaluates held-out preference recovery from proxy-implied utilities,
5) adapts the next round's training budget/model complexity if recovery lags.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Keep BLAS/OpenMP single-threaded per worker; parallelism comes from multi-process trials.
for _env_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.ops_checks import EvidenceStatus
from src.ctreepo.opt import (
    SklearnProxyOracle,
    TorchMSEProxyOracle,
    collect_pairwise_preferences,
    collect_proxy_training_data,
    derive_preference_from_utilities,
    to_training_preference_dataset,
)
from src.ctreepo.sim.core.segment_lda_ops_weight_recovery import (
    _oracle_from_prefix,
    _prefix_counts,
    _span_features_from_prefix,
    generate_segment_lda_docs,
    sample_sparse_oracle_weights,
    sample_topic_distributions,
)
from src.tree.markov_boundary_honesty_simulation import _make_transition_matrices
from src.tree.markov_changepoint_honesty_simulation import MarkovChangepointConfig, generate_changepoint_docs
from src.tree.mergeable_ablation import generate_exact_spike_count_document, true_spike_count


SETTINGS: Tuple[str, ...] = ("markov", "mergeable", "segment_lda")


@dataclass(frozen=True)
class RoundPlan:
    train_examples: int
    val_examples: int
    test_examples: int
    n_pairs_per_example: int
    tie_margin: float
    trials_per_setting: int
    model_budget: int


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": str(self.kind), "params": dict(self.params)}


@dataclass(frozen=True)
class TrialSpec:
    round_index: int
    setting: str
    model_spec: ModelSpec
    trial_seed: int
    dataset_seed: int
    preference_seed: int


@dataclass(frozen=True)
class MarkovExample:
    example_id: str
    y_true: float
    z: Tuple[float, ...]
    n_tokens: int
    token_change_count: int


@dataclass(frozen=True)
class MergeableExample:
    example_id: str
    y_true: float
    z: Tuple[float, ...]
    proxy_scores: Tuple[float, ...]


@dataclass(frozen=True)
class LDAExample:
    example_id: str
    y_true: float
    z: Tuple[float, ...]
    feat_true: Tuple[float, ...]


@dataclass(frozen=True)
class SettingBundle:
    setting: str
    train_examples: Sequence[Any]
    val_examples: Sequence[Any]
    test_examples: Sequence[Any]
    compressor: Any
    oracle_fn: Callable[[Any], float]
    utility_fn: Callable[[Any, Any], float]
    candidate_generator: Any
    candidate_score_fn: Callable[[Any], float]
    rubric: str
    evidence_status: str = EvidenceStatus.PROXY_ONLY.value


class _TupleCompressor:
    """Compressor adapter for examples that carry a tuple-valued sketch in `.z`."""

    def compress(self, x: Any) -> np.ndarray:
        return np.asarray(getattr(x, "z"), dtype=np.float64)


@dataclass(frozen=True)
class _MarkovCandidateGenerator:
    sigma_a: float
    sigma_b: float
    bias_b: float
    max_count: int

    def generate(self, x: MarkovExample, *, n: int, seed: Optional[int] = None) -> Sequence[int]:
        del n
        rng = np.random.default_rng(_seed_or_zero(seed))
        center = _markov_candidate_center(x.n_tokens, x.token_change_count)
        a = int(np.clip(np.rint(center + rng.normal(0.0, float(self.sigma_a))), 0, int(self.max_count)))
        b = int(
            np.clip(
                np.rint(center + float(self.bias_b) + rng.normal(0.0, float(self.sigma_b))),
                0,
                int(self.max_count),
            )
        )
        return (a, b)


@dataclass(frozen=True)
class _MergeableCandidateGenerator:
    spike_threshold: float
    sigma_a: float
    sigma_b: float
    bias_b: float

    def generate(self, x: MergeableExample, *, n: int, seed: Optional[int] = None) -> Sequence[int]:
        del n
        rng = np.random.default_rng(_seed_or_zero(seed))
        proxy = np.asarray(x.proxy_scores, dtype=np.float64)
        mid = proxy.size // 2
        base_a = float(np.sum(proxy >= float(self.spike_threshold)))
        base_b = float(np.sum(proxy[:mid] >= float(self.spike_threshold)))
        a = int(np.clip(np.rint(base_a + rng.normal(0.0, float(self.sigma_a))), 0, proxy.size))
        b = int(
            np.clip(
                np.rint(base_b + float(self.bias_b) + rng.normal(0.0, float(self.sigma_b))),
                0,
                proxy.size,
            )
        )
        return (a, b)


@dataclass(frozen=True)
class _LDACandidateGenerator:
    beta_unit: Tuple[float, ...]
    sigma_a: float
    sigma_b: float
    bias_b: float

    def generate(self, x: LDAExample, *, n: int, seed: Optional[int] = None) -> Sequence[np.ndarray]:
        del n
        rng = np.random.default_rng(_seed_or_zero(seed))
        feat = np.asarray(x.feat_true, dtype=np.float64)
        unit = np.asarray(self.beta_unit, dtype=np.float64)
        shift_a = float(rng.normal(0.0, float(self.sigma_a)))
        shift_b = float(self.bias_b + rng.normal(0.0, float(self.sigma_b)))
        cand_a = feat + (shift_a * unit)
        cand_b = feat + (shift_b * unit)
        return (cand_a, cand_b)


def _seed_or_zero(seed: Optional[int]) -> int:
    return int(0 if seed is None else seed)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive overnight preference-recovery optimization loop.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--duration-hours", type=float, default=10.0)
    p.add_argument("--jobs", type=int, default=64)
    p.add_argument("--cpu-set", type=str, default="64-127")
    p.add_argument("--target-pref-accuracy", type=float, default=0.90)
    p.add_argument("--max-rounds", type=int, default=0, help="0 means unlimited (until duration).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--initial-train-examples", type=int, default=384)
    p.add_argument("--initial-val-examples", type=int, default=160)
    p.add_argument("--initial-test-examples", type=int, default=160)
    p.add_argument("--initial-trials-per-setting", type=int, default=24)
    p.add_argument("--initial-pairs-per-example", type=int, default=2)
    p.add_argument("--initial-tie-margin", type=float, default=0.0)
    p.add_argument("--enable-torch", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _parse_cpu_set(cpu_set: str) -> List[int]:
    cpus: List[int] = []
    for chunk in str(cpu_set).split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if lo > hi:
                raise ValueError(f"Invalid cpu range: {part!r}")
            cpus.extend(list(range(lo, hi + 1)))
        else:
            cpus.append(int(part))
    uniq = sorted(set(cpus))
    if not uniq:
        raise ValueError("cpu set resolved to empty list")
    return uniq


def _set_affinity(cpus: Sequence[int]) -> None:
    try:
        os.sched_setaffinity(0, set(int(x) for x in cpus))
    except Exception:
        # Non-fatal on platforms without sched_setaffinity support.
        return


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    _ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def _markov_candidate_center(n_tokens: int, token_change_count: int) -> float:
    # A lightweight proxy from observable token dynamics -> changepoint-count guess.
    n = float(max(1, int(n_tokens)))
    change = float(max(0, int(token_change_count)))
    center = (0.055 * change) + (0.015 * n) - 0.9
    return float(max(0.0, min(16.0, center)))


def _markov_features(tokens: Sequence[int], *, vocab_size: int) -> Tuple[float, ...]:
    toks = np.asarray(tokens, dtype=np.int64)
    n = int(toks.size)
    if n <= 1:
        change_count = 0.0
        change_rate = 0.0
        run_mean = float(n)
        run_std = 0.0
    else:
        changes = (toks[1:] != toks[:-1]).astype(np.int64, copy=False)
        change_count = float(np.sum(changes))
        change_rate = float(change_count / float(max(1, n - 1)))
        cut_idx = np.where(changes > 0)[0] + 1
        cuts = np.concatenate([np.asarray([0]), cut_idx, np.asarray([n])], axis=0)
        runs = np.diff(cuts).astype(np.float64, copy=False)
        run_mean = float(np.mean(runs)) if runs.size > 0 else float(n)
        run_std = float(np.std(runs)) if runs.size > 0 else 0.0

    hist = np.bincount(toks, minlength=int(vocab_size)).astype(np.float64)
    denom = float(max(1, n))
    hist = hist / denom
    # Keep a compact fixed-width histogram.
    if hist.size > 16:
        parts = np.array_split(hist, 16)
        hist_compact = np.asarray([float(np.sum(p)) for p in parts], dtype=np.float64)
    elif hist.size < 16:
        hist_compact = np.pad(hist, (0, 16 - hist.size), mode="constant")
    else:
        hist_compact = hist
    nonzero = hist[hist > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero))) if nonzero.size > 0 else 0.0
    out = np.concatenate(
        [
            np.asarray(
                [
                    float(n),
                    float(change_count),
                    float(change_rate),
                    float(run_mean),
                    float(run_std),
                    float(entropy),
                ],
                dtype=np.float64,
            ),
            hist_compact.astype(np.float64, copy=False),
        ],
        axis=0,
    )
    return tuple(float(v) for v in out.tolist())


def _mergeable_features(proxy_scores: Sequence[float], *, spike_threshold: float) -> Tuple[float, ...]:
    arr = np.asarray(proxy_scores, dtype=np.float64)
    n = int(arr.size)
    mid = n // 2
    first = float(np.sum(arr[:mid] >= float(spike_threshold)))
    second = float(np.sum(arr[mid:] >= float(spike_threshold)))
    boundary = max(1, min(4, n))
    boundary_vals = np.concatenate([arr[:boundary], arr[n - boundary :]], axis=0)
    out = np.asarray(
        [
            float(n),
            float(first),
            float(second),
            float(first + second),
            float(np.mean(arr)) if n > 0 else 0.0,
            float(np.std(arr)) if n > 0 else 0.0,
            float(np.max(arr)) if n > 0 else 0.0,
            float(np.mean(boundary_vals)) if boundary_vals.size > 0 else 0.0,
        ],
        dtype=np.float64,
    )
    return tuple(float(v) for v in out.tolist())


def _lda_features(tokens: Sequence[int], *, vocab_size: int, bigram_hash_dim: int) -> Tuple[float, ...]:
    toks = np.asarray(tokens, dtype=np.int64)
    n = int(toks.size)
    hist = np.bincount(toks, minlength=int(vocab_size)).astype(np.float64)
    hist = hist / float(max(1, n))
    if n >= 2:
        pair_idx = (toks[:-1] * int(vocab_size) + toks[1:]).astype(np.int64, copy=False)
        hashed = np.bincount(pair_idx % int(bigram_hash_dim), minlength=int(bigram_hash_dim)).astype(
            np.float64
        )
        hashed = hashed / float(max(1, n - 1))
    else:
        hashed = np.zeros((int(bigram_hash_dim),), dtype=np.float64)

    stats = np.asarray(
        [
            float(n),
            float(np.mean(toks)) if n > 0 else 0.0,
            float(np.std(toks)) if n > 0 else 0.0,
        ],
        dtype=np.float64,
    )
    out = np.concatenate([stats, hist, hashed], axis=0).astype(np.float64, copy=False)
    return tuple(float(v) for v in out.tolist())


def _build_markov_examples(
    *,
    n_examples: int,
    dataset_seed: int,
    split_tag: str,
) -> List[MarkovExample]:
    config = MarkovChangepointConfig(
        n_regimes=4,
        vocab_size=24,
        min_tokens=56,
        max_tokens=88,
        min_segments=2,
        max_segments=6,
        min_seg_len=6,
        max_seg_len=24,
        train_docs=int(n_examples),
        test_docs=0,
        sinkhorn_iters=10,
        transition_log_std=1.0,
        seed=int(dataset_seed),
        use_cuda=False,
        torch_threads=1,
    )
    rng = np.random.default_rng(int(dataset_seed) + 17)
    transitions = _make_transition_matrices(
        n_classes=int(config.n_regimes),
        vocab_size=int(config.vocab_size),
        log_std=float(config.transition_log_std),
        sinkhorn_iters=int(config.sinkhorn_iters),
        rng=rng,
    )
    docs = generate_changepoint_docs(config, transitions=transitions)
    out: List[MarkovExample] = []
    for i, doc in enumerate(docs):
        y_true = float(len(doc.true_boundaries))
        token_change_count = int(np.sum(np.asarray(doc.tokens[1:]) != np.asarray(doc.tokens[:-1])))
        z = _markov_features(doc.tokens, vocab_size=int(config.vocab_size))
        out.append(
            MarkovExample(
                example_id=f"markov/{split_tag}/{i}",
                y_true=y_true,
                z=z,
                n_tokens=int(len(doc.tokens)),
                token_change_count=int(token_change_count),
            )
        )
    return out


def _build_mergeable_examples(
    *,
    n_examples: int,
    dataset_seed: int,
    split_tag: str,
    spike_threshold: float,
) -> List[MergeableExample]:
    rng = np.random.default_rng(int(dataset_seed))
    out: List[MergeableExample] = []
    for i in range(int(n_examples)):
        n_spikes = int(min(10, max(0, rng.poisson(2.2))))
        doc = generate_exact_spike_count_document(
            n_spikes=int(n_spikes),
            n_tokens=32,
            proxy_noise=0.10,
            boundary_span_tokens=4,
            force_boundary_spike=False,
            seed=int(dataset_seed) + 1000 + i,
        )
        y_true = float(true_spike_count(doc.token_scores))
        z = _mergeable_features(doc.proxy_scores, spike_threshold=float(spike_threshold))
        out.append(
            MergeableExample(
                example_id=f"mergeable/{split_tag}/{i}",
                y_true=y_true,
                z=z,
                proxy_scores=tuple(float(v) for v in doc.proxy_scores),
            )
        )
    return out


def _build_segment_lda_examples(
    *,
    n_examples: int,
    dataset_seed: int,
    split_tag: str,
) -> Tuple[List[LDAExample], np.ndarray]:
    n_topics = 4
    vocab_size = 48
    topics, _topic_meta = sample_topic_distributions(
        vocab_size=int(vocab_size),
        n_topics=int(n_topics),
        topic_concentration=0.35,
        emission_mode="disjoint",
        anchor_words_per_topic=0,
        anchor_multiplier=1.0,
        seed=int(dataset_seed) + 11,
    )
    _relevant, theta, w_mat = sample_sparse_oracle_weights(
        n_topics=int(n_topics),
        relevant_topics=2,
        theta_scale=1.0,
        zero_diagonal=False,
        seed=int(dataset_seed) + 23,
    )
    lambda_bigram = 0.65
    w_big = (float(lambda_bigram) * np.asarray(w_mat, dtype=np.float64).reshape(-1)).astype(
        np.float64, copy=False
    )
    beta = np.concatenate([np.asarray(theta, dtype=np.float64), w_big], axis=0).astype(np.float64, copy=False)

    docs, _stats = generate_segment_lda_docs(
        int(n_examples),
        topics=topics,
        min_tokens=56,
        max_tokens=88,
        min_segments=2,
        max_segments=6,
        min_seg_len=6,
        max_seg_len=24,
        leaf_tokens=8,
        align_segments_to_leaves=True,
        doc_topic_concentration=0.40,
        topic_process="segments",
        boundary_profile="middle",
        boundary_profile_strength=0.7,
        boundary_profile_seed=int(dataset_seed) + 31,
        segment_length_power=0.25,
        seed=int(dataset_seed) + 37,
    )

    out: List[LDAExample] = []
    for i, doc in enumerate(docs):
        span = (0, len(doc.topics))
        topic_prefix, bigram_prefix = _prefix_counts(doc.topics, n_topics=int(n_topics))
        feat_true, _first, _last = _span_features_from_prefix(
            topic_prefix,
            bigram_prefix,
            doc.topics,
            span,
            n_topics=int(n_topics),
        )
        y_true = _oracle_from_prefix(
            np.asarray(theta, dtype=np.float64),
            np.asarray(w_big, dtype=np.float64),
            topic_prefix,
            bigram_prefix,
            doc.topics,
            span,
        )
        z = _lda_features(doc.tokens, vocab_size=int(vocab_size), bigram_hash_dim=64)
        out.append(
            LDAExample(
                example_id=f"segment_lda/{split_tag}/{i}",
                y_true=float(y_true),
                z=z,
                feat_true=tuple(float(v) for v in np.asarray(feat_true, dtype=np.float64).tolist()),
            )
        )
    return out, beta


def _build_setting_bundle(setting: str, *, plan: RoundPlan, dataset_seed: int) -> SettingBundle:
    compressor = _TupleCompressor()
    oracle_fn = lambda ex: float(getattr(ex, "y_true"))

    if setting == "markov":
        train = _build_markov_examples(
            n_examples=int(plan.train_examples),
            dataset_seed=int(dataset_seed) + 101,
            split_tag="train",
        )
        val = _build_markov_examples(
            n_examples=int(plan.val_examples),
            dataset_seed=int(dataset_seed) + 103,
            split_tag="val",
        )
        test = _build_markov_examples(
            n_examples=int(plan.test_examples),
            dataset_seed=int(dataset_seed) + 107,
            split_tag="test",
        )
        candidate_generator = _MarkovCandidateGenerator(
            sigma_a=1.0,
            sigma_b=2.0,
            bias_b=1.6,
            max_count=16,
        )
        utility_fn = lambda ex, cand: float(-abs(float(cand) - float(ex.y_true)))
        score_fn = lambda cand: float(cand)
        return SettingBundle(
            setting=setting,
            train_examples=train,
            val_examples=val,
            test_examples=test,
            compressor=compressor,
            oracle_fn=oracle_fn,
            utility_fn=utility_fn,
            candidate_generator=candidate_generator,
            candidate_score_fn=score_fn,
            rubric="markov-changepoint-count",
        )

    if setting == "mergeable":
        threshold = 0.90
        train = _build_mergeable_examples(
            n_examples=int(plan.train_examples),
            dataset_seed=int(dataset_seed) + 211,
            split_tag="train",
            spike_threshold=threshold,
        )
        val = _build_mergeable_examples(
            n_examples=int(plan.val_examples),
            dataset_seed=int(dataset_seed) + 223,
            split_tag="val",
            spike_threshold=threshold,
        )
        test = _build_mergeable_examples(
            n_examples=int(plan.test_examples),
            dataset_seed=int(dataset_seed) + 227,
            split_tag="test",
            spike_threshold=threshold,
        )
        candidate_generator = _MergeableCandidateGenerator(
            spike_threshold=threshold,
            sigma_a=0.4,
            sigma_b=1.0,
            bias_b=1.2,
        )
        utility_fn = lambda ex, cand: float(-abs(float(cand) - float(ex.y_true)))
        score_fn = lambda cand: float(cand)
        return SettingBundle(
            setting=setting,
            train_examples=train,
            val_examples=val,
            test_examples=test,
            compressor=compressor,
            oracle_fn=oracle_fn,
            utility_fn=utility_fn,
            candidate_generator=candidate_generator,
            candidate_score_fn=score_fn,
            rubric="mergeable-spike-count",
        )

    if setting == "segment_lda":
        train, beta = _build_segment_lda_examples(
            n_examples=int(plan.train_examples),
            dataset_seed=int(dataset_seed) + 311,
            split_tag="train",
        )
        val, _beta_val = _build_segment_lda_examples(
            n_examples=int(plan.val_examples),
            dataset_seed=int(dataset_seed) + 313,
            split_tag="val",
        )
        test, _beta_test = _build_segment_lda_examples(
            n_examples=int(plan.test_examples),
            dataset_seed=int(dataset_seed) + 317,
            split_tag="test",
        )
        beta = np.asarray(beta, dtype=np.float64)
        denom = float(np.dot(beta, beta))
        unit = beta / float(max(1e-8, denom))
        beta_unit = tuple(float(v) for v in unit.tolist())
        candidate_generator = _LDACandidateGenerator(
            beta_unit=beta_unit,
            sigma_a=0.60,
            sigma_b=1.20,
            bias_b=0.85,
        )
        beta_local = np.asarray(beta, dtype=np.float64)

        def _cand_score(cand: Any) -> float:
            vec = np.asarray(cand, dtype=np.float64)
            return float(np.dot(beta_local, vec))

        def _utility(ex: LDAExample, cand: Any) -> float:
            score = _cand_score(cand)
            return float(-abs(score - float(ex.y_true)))

        return SettingBundle(
            setting=setting,
            train_examples=train,
            val_examples=val,
            test_examples=test,
            compressor=compressor,
            oracle_fn=oracle_fn,
            utility_fn=_utility,
            candidate_generator=candidate_generator,
            candidate_score_fn=_cand_score,
            rubric="segment-lda-ops-linear-functional",
        )

    raise ValueError(f"unsupported setting: {setting!r}")


def _sample_model_specs(
    *,
    rng: np.random.Generator,
    n_specs: int,
    model_budget: int,
    enable_torch: bool,
) -> List[ModelSpec]:
    budget = int(max(0, min(2, int(model_budget))))
    kinds: List[str] = ["ridge", "rf", "mlp"]
    if budget >= 1:
        kinds.append("extra_trees")
    if budget >= 2 and bool(enable_torch):
        kinds.append("torch")

    specs: List[ModelSpec] = []
    for _ in range(int(max(1, n_specs))):
        kind = str(rng.choice(kinds))
        if kind == "ridge":
            alpha = float(10 ** float(rng.uniform(-3.0, 2.0)))
            specs.append(ModelSpec(kind=kind, params={"alpha": alpha}))
            continue
        if kind == "rf":
            n_estimators = int(rng.choice([96, 128, 160, 224, 320]))
            max_depth = int(rng.choice([6, 8, 10, 14, 18, 0]))
            min_samples_leaf = int(rng.choice([1, 2, 4, 8]))
            specs.append(
                ModelSpec(
                    kind=kind,
                    params={
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                    },
                )
            )
            continue
        if kind == "extra_trees":
            n_estimators = int(rng.choice([96, 128, 192, 256, 320]))
            max_depth = int(rng.choice([6, 8, 10, 14, 0]))
            min_samples_leaf = int(rng.choice([1, 2, 4]))
            specs.append(
                ModelSpec(
                    kind=kind,
                    params={
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                    },
                )
            )
            continue
        if kind == "mlp":
            hidden = int(rng.choice([32, 48, 64, 96, 128, 160]))
            alpha = float(10 ** float(rng.uniform(-6.0, -2.0)))
            lr = float(10 ** float(rng.uniform(-4.0, -2.2)))
            max_iter = int(rng.choice([220, 280, 340]))
            specs.append(
                ModelSpec(
                    kind=kind,
                    params={
                        "hidden": hidden,
                        "alpha": alpha,
                        "learning_rate_init": lr,
                        "max_iter": max_iter,
                    },
                )
            )
            continue
        if kind == "torch":
            hidden = int(rng.choice([32, 64, 96, 128, 160]))
            lr = float(10 ** float(rng.uniform(-4.2, -2.4)))
            n_epochs = int(rng.choice([18, 26, 34, 48]))
            batch_size = int(rng.choice([32, 48, 64, 96]))
            specs.append(
                ModelSpec(
                    kind=kind,
                    params={
                        "hidden": hidden,
                        "lr": lr,
                        "n_epochs": n_epochs,
                        "batch_size": batch_size,
                    },
                )
            )
            continue
        raise ValueError(f"unreachable model kind: {kind!r}")
    return specs


def _build_proxy(model_spec: ModelSpec, *, input_dim: int, seed: int) -> Any:
    kind = str(model_spec.kind).strip().lower()
    params = dict(model_spec.params)
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        estimator = make_pipeline(
            StandardScaler(),
            Ridge(
                alpha=float(params.get("alpha", 1.0)),
                random_state=int(seed),
            ),
        )
        return SklearnProxyOracle(estimator=estimator)

    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor

        max_depth = int(params.get("max_depth", 0))
        estimator = RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 128)),
            max_depth=None if max_depth <= 0 else max_depth,
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=int(seed),
            n_jobs=1,
        )
        return SklearnProxyOracle(estimator=estimator)

    if kind == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        max_depth = int(params.get("max_depth", 0))
        estimator = ExtraTreesRegressor(
            n_estimators=int(params.get("n_estimators", 128)),
            max_depth=None if max_depth <= 0 else max_depth,
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=int(seed),
            n_jobs=1,
        )
        return SklearnProxyOracle(estimator=estimator)

    if kind == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        estimator = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(int(params.get("hidden", 64)),),
                activation="relu",
                alpha=float(params.get("alpha", 1e-4)),
                learning_rate_init=float(params.get("learning_rate_init", 1e-3)),
                max_iter=int(params.get("max_iter", 260)),
                random_state=int(seed),
            ),
        )
        return SklearnProxyOracle(estimator=estimator)

    if kind == "torch":
        import torch
        from torch import nn

        torch.manual_seed(int(seed))
        hidden = int(params.get("hidden", 64))
        model = nn.Sequential(
            nn.Linear(int(input_dim), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        return TorchMSEProxyOracle(
            model=model,
            lr=float(params.get("lr", 1e-3)),
            n_epochs=int(params.get("n_epochs", 20)),
            batch_size=int(params.get("batch_size", 64)),
            device="cpu",
        )

    raise ValueError(f"unsupported model kind: {model_spec.kind!r}")


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    abs_err = np.abs(y_true - y_pred)
    sq_err = (y_true - y_pred) ** 2
    mae = float(np.mean(abs_err)) if abs_err.size > 0 else float("nan")
    rmse = float(math.sqrt(float(np.mean(sq_err)))) if sq_err.size > 0 else float("nan")
    y_mean = float(np.mean(y_true)) if y_true.size > 0 else 0.0
    ss_res = float(np.sum(sq_err))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot <= 1e-12:
        r2 = 0.0
    else:
        r2 = float(1.0 - (ss_res / ss_tot))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _predict_pref_from_proxy(
    *,
    y_hat: float,
    candidate_a: Any,
    candidate_b: Any,
    candidate_score_fn: Callable[[Any], float],
    tie_margin: float,
) -> str:
    score_a = float(candidate_score_fn(candidate_a))
    score_b = float(candidate_score_fn(candidate_b))
    util_a = float(-abs(score_a - float(y_hat)))
    util_b = float(-abs(score_b - float(y_hat)))
    pred = derive_preference_from_utilities(util_a, util_b, tie_margin=float(tie_margin))
    return str(pred.preferred)


def _preference_metrics(
    *,
    records: Sequence[Any],
    y_hat_by_example: Dict[str, float],
    candidate_score_fn: Callable[[Any], float],
    tie_margin: float,
) -> Dict[str, float]:
    total = 0
    correct = 0
    non_tie_total = 0
    non_tie_correct = 0
    true_ties = 0
    pred_ties = 0

    for rec in records:
        ex_id = str(rec.example_id)
        if ex_id not in y_hat_by_example:
            continue
        y_hat = float(y_hat_by_example[ex_id])
        pred_pref = _predict_pref_from_proxy(
            y_hat=y_hat,
            candidate_a=rec.candidate_a,
            candidate_b=rec.candidate_b,
            candidate_score_fn=candidate_score_fn,
            tie_margin=float(tie_margin),
        )
        true_pref = str(rec.preferred)
        total += 1
        if pred_pref == true_pref:
            correct += 1
        if true_pref != "tie":
            non_tie_total += 1
            if pred_pref == true_pref:
                non_tie_correct += 1
        else:
            true_ties += 1
        if pred_pref == "tie":
            pred_ties += 1

    acc = (float(correct) / float(total)) if total > 0 else 0.0
    non_tie_acc = (float(non_tie_correct) / float(non_tie_total)) if non_tie_total > 0 else 0.0
    tie_rate = (float(true_ties) / float(total)) if total > 0 else 0.0
    pred_tie_rate = (float(pred_ties) / float(total)) if total > 0 else 0.0
    return {
        "n_pairs": float(total),
        "n_non_tie_pairs": float(non_tie_total),
        "pref_accuracy": float(acc),
        "pref_non_tie_accuracy": float(non_tie_acc),
        "true_tie_rate": float(tie_rate),
        "pred_tie_rate": float(pred_tie_rate),
    }


def _collect_split_records(
    *,
    examples: Sequence[Any],
    bundle: SettingBundle,
    seed: int,
    plan: RoundPlan,
) -> List[Any]:
    return collect_pairwise_preferences(
        examples,
        candidate_generator=bundle.candidate_generator,
        utility_fn=bundle.utility_fn,
        example_id_fn=lambda ex, _i: str(getattr(ex, "example_id")),
        rubric=bundle.rubric,
        tie_margin=float(plan.tie_margin),
        n_pairs_per_example=int(plan.n_pairs_per_example),
        seed=int(seed),
    )


def _fit_and_predict(
    *,
    model_spec: ModelSpec,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    val_inputs: np.ndarray,
    test_inputs: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    proxy = _build_proxy(model_spec, input_dim=int(train_inputs.shape[1]), seed=int(seed))
    proxy.fit(train_inputs, train_targets)
    y_val = np.asarray(proxy.predict(val_inputs), dtype=np.float64).reshape(-1)
    y_test = np.asarray(proxy.predict(test_inputs), dtype=np.float64).reshape(-1)
    return y_val, y_test


def _worker_init(cpu_ids: Sequence[int]) -> None:
    _set_affinity(cpu_ids)


def _trial_task(spec: TrialSpec, plan: RoundPlan, enable_torch: bool) -> Dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        bundle = _build_setting_bundle(
            spec.setting,
            plan=plan,
            dataset_seed=int(spec.dataset_seed),
        )
        if str(bundle.evidence_status) != EvidenceStatus.PROXY_ONLY.value:
            raise ValueError(
                "opt-layer overnight runner only supports proxy-only compressors in this pass"
            )
        train_inputs_raw, train_targets_raw, _weights = collect_proxy_training_data(
            bundle.train_examples,
            compressor=bundle.compressor,
            oracle=bundle.oracle_fn,
            ipw_fn=None,
        )
        train_inputs = np.asarray(train_inputs_raw, dtype=np.float64)
        train_targets = np.asarray(train_targets_raw, dtype=np.float64).reshape(-1)
        val_inputs = np.asarray([bundle.compressor.compress(ex) for ex in bundle.val_examples], dtype=np.float64)
        val_targets = np.asarray([bundle.oracle_fn(ex) for ex in bundle.val_examples], dtype=np.float64)
        test_inputs = np.asarray([bundle.compressor.compress(ex) for ex in bundle.test_examples], dtype=np.float64)
        test_targets = np.asarray([bundle.oracle_fn(ex) for ex in bundle.test_examples], dtype=np.float64)

        if train_inputs.ndim != 2 or train_inputs.shape[0] == 0:
            raise ValueError(f"bad training matrix shape: {train_inputs.shape!r}")

        model_kind = str(spec.model_spec.kind).strip().lower()
        if model_kind == "torch" and not bool(enable_torch):
            raise RuntimeError("torch model sampled while --enable-torch is false")

        y_val_hat, y_test_hat = _fit_and_predict(
            model_spec=spec.model_spec,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            test_inputs=test_inputs,
            seed=int(spec.trial_seed),
        )
        val_reg = _regression_metrics(val_targets, y_val_hat)
        test_reg = _regression_metrics(test_targets, y_test_hat)

        val_records = _collect_split_records(
            examples=bundle.val_examples,
            bundle=bundle,
            seed=int(spec.preference_seed) + 1,
            plan=plan,
        )
        test_records = _collect_split_records(
            examples=bundle.test_examples,
            bundle=bundle,
            seed=int(spec.preference_seed) + 2,
            plan=plan,
        )
        # Ensure adapter path remains healthy (records -> canonical preference dataset).
        val_ds = to_training_preference_dataset(val_records)
        test_ds = to_training_preference_dataset(test_records)

        val_hat_map = {
            str(ex.example_id): float(pred)
            for ex, pred in zip(bundle.val_examples, y_val_hat.tolist())
        }
        test_hat_map = {
            str(ex.example_id): float(pred)
            for ex, pred in zip(bundle.test_examples, y_test_hat.tolist())
        }
        val_pref = _preference_metrics(
            records=val_records,
            y_hat_by_example=val_hat_map,
            candidate_score_fn=bundle.candidate_score_fn,
            tie_margin=float(plan.tie_margin),
        )
        test_pref = _preference_metrics(
            records=test_records,
            y_hat_by_example=test_hat_map,
            candidate_score_fn=bundle.candidate_score_fn,
            tie_margin=float(plan.tie_margin),
        )

        serialization_ok = 1.0
        if spec.setting == "segment_lda" and len(test_ds) > 0:
            ok = 0
            for pair in test_ds:
                a_ok = str(pair.summary_a).strip().startswith("[")
                b_ok = str(pair.summary_b).strip().startswith("[")
                ok += int(a_ok and b_ok)
            serialization_ok = float(ok) / float(len(test_ds))

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        return {
            "ok": True,
            "setting": spec.setting,
            "evidence_status": str(bundle.evidence_status),
            "round_index": int(spec.round_index),
            "model_spec": spec.model_spec.to_dict(),
            "trial_seed": int(spec.trial_seed),
            "dataset_seed": int(spec.dataset_seed),
            "preference_seed": int(spec.preference_seed),
            "n_train_examples": int(len(bundle.train_examples)),
            "n_val_examples": int(len(bundle.val_examples)),
            "n_test_examples": int(len(bundle.test_examples)),
            "n_val_pairs": int(len(val_records)),
            "n_test_pairs": int(len(test_records)),
            "adapter_val_pairs": int(len(val_ds)),
            "adapter_test_pairs": int(len(test_ds)),
            "segment_lda_json_array_serialization_rate": float(serialization_ok),
            "val": {**val_reg, **val_pref},
            "test": {**test_reg, **test_pref},
            "elapsed_sec": float(elapsed),
        }
    except Exception as exc:
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        return {
            "ok": False,
            "setting": spec.setting,
            "evidence_status": EvidenceStatus.PROXY_ONLY.value,
            "round_index": int(spec.round_index),
            "model_spec": spec.model_spec.to_dict(),
            "trial_seed": int(spec.trial_seed),
            "dataset_seed": int(spec.dataset_seed),
            "preference_seed": int(spec.preference_seed),
            "elapsed_sec": float(elapsed),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=20),
        }


def _select_best_trial(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid = [r for r in rows if bool(r.get("ok"))]
    if not valid:
        return None
    return max(
        valid,
        key=lambda r: (
            float((r.get("val", {}) or {}).get("pref_non_tie_accuracy", -1.0)),
            float((r.get("val", {}) or {}).get("pref_accuracy", -1.0)),
            -float((r.get("val", {}) or {}).get("mae", 1e9)),
        ),
    )


def _trial_rank_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
    val = row.get("val", {}) if isinstance(row.get("val"), dict) else {}
    return (
        float(val.get("pref_non_tie_accuracy", -1.0)),
        float(val.get("pref_accuracy", -1.0)),
        -float(val.get("mae", 1e9)),
    )


def _round_score(best_by_setting: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, float]:
    vals: List[float] = []
    for setting in SETTINGS:
        row = best_by_setting.get(setting)
        if not row:
            continue
        test = row.get("test", {}) if isinstance(row.get("test"), dict) else {}
        vals.append(float(test.get("pref_non_tie_accuracy", 0.0)))
    if not vals:
        return {"mean_pref_non_tie_accuracy": 0.0, "min_pref_non_tie_accuracy": 0.0}
    return {
        "mean_pref_non_tie_accuracy": float(np.mean(np.asarray(vals, dtype=np.float64))),
        "min_pref_non_tie_accuracy": float(np.min(np.asarray(vals, dtype=np.float64))),
    }


def _choose_next_plan(
    *,
    current: RoundPlan,
    best_by_setting: Dict[str, Optional[Dict[str, Any]]],
    target_pref_accuracy: float,
) -> Tuple[RoundPlan, str]:
    below: List[str] = []
    for setting in SETTINGS:
        row = best_by_setting.get(setting)
        if not row:
            below.append(setting)
            continue
        score = float((row.get("test", {}) or {}).get("pref_non_tie_accuracy", 0.0))
        if score < float(target_pref_accuracy):
            below.append(setting)

    if below:
        nxt = replace(
            current,
            train_examples=min(6144, int(math.ceil(float(current.train_examples) * 1.45))),
            val_examples=min(2048, int(math.ceil(float(current.val_examples) * 1.25))),
            n_pairs_per_example=min(4, int(current.n_pairs_per_example) + 1),
            trials_per_setting=min(48, int(current.trials_per_setting) + 6),
            model_budget=min(2, int(current.model_budget) + 1),
            tie_margin=max(0.0, float(current.tie_margin) * 0.85),
        )
        return nxt, f"escalate_training_for={','.join(below)}"

    strong = True
    for setting in SETTINGS:
        row = best_by_setting.get(setting)
        if not row:
            strong = False
            break
        score = float((row.get("test", {}) or {}).get("pref_non_tie_accuracy", 0.0))
        if score < float(target_pref_accuracy) + 0.05:
            strong = False
            break
    if strong:
        nxt = replace(
            current,
            tie_margin=max(0.0, float(current.tie_margin) * 0.9),
            trials_per_setting=max(16, int(current.trials_per_setting) - 3),
        )
        return nxt, "stabilize_after_strong_recovery"

    return current, "hold_configuration"


def _render_latest_report(
    *,
    out_path: Path,
    output_dir: Path,
    history: Sequence[Dict[str, Any]],
    best_overall: Optional[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# Opt Layer Overnight Report")
    lines.append("")
    lines.append(f"Generated: {_now_iso()}")
    lines.append(f"Output root: `{output_dir}`")
    lines.append("")

    if best_overall is not None:
        lines.append("## Best Round")
        lines.append("")
        lines.append(f"- round: {best_overall.get('round_index')}")
        score = best_overall.get("aggregate", {})
        if isinstance(score, dict):
            lines.append(
                "- mean test non-tie preference accuracy: "
                f"{float(score.get('mean_pref_non_tie_accuracy', 0.0)):.4f}"
            )
            lines.append(
                "- min test non-tie preference accuracy: "
                f"{float(score.get('min_pref_non_tie_accuracy', 0.0)):.4f}"
            )
        lines.append("")
        lines.append("- settings:")
        best_settings = best_overall.get("best_by_setting", {})
        if isinstance(best_settings, dict):
            for setting in SETTINGS:
                row = best_settings.get(setting)
                if not isinstance(row, dict):
                    lines.append(f"  - {setting}: no valid trial")
                    continue
                test = row.get("test", {}) if isinstance(row.get("test"), dict) else {}
                mspec = row.get("model_spec", {})
                lines.append(
                    "  - "
                    f"{setting}: pref_non_tie={float(test.get('pref_non_tie_accuracy', 0.0)):.4f}, "
                    f"pref_all={float(test.get('pref_accuracy', 0.0)):.4f}, "
                    f"mae={float(test.get('mae', float('nan'))):.4f}, "
                    f"model={json.dumps(mspec, sort_keys=True)}"
                )
        lines.append("")

    lines.append("## Round Timeline")
    lines.append("")
    for row in history:
        aggregate = row.get("aggregate", {}) if isinstance(row.get("aggregate"), dict) else {}
        lines.append(
            "- "
            f"round={row.get('round_index')} "
            f"mean_non_tie={float(aggregate.get('mean_pref_non_tie_accuracy', 0.0)):.4f} "
            f"min_non_tie={float(aggregate.get('min_pref_non_tie_accuracy', 0.0)):.4f} "
            f"next_action={row.get('next_action')}"
        )

    _write_text(out_path, "\n".join(lines) + "\n")


def _progress_line(*, round_index: int, done: int, total: int, ok: int, failed: int) -> str:
    return (
        f"[{_now_iso()}] round={round_index} progress={done}/{total} "
        f"ok={ok} failed={failed}"
    )


def main() -> int:
    args = _parse_args()
    cpus = _parse_cpu_set(str(args.cpu_set))
    _set_affinity(cpus)

    start_utc = datetime.now(timezone.utc)
    deadline = start_utc + timedelta(hours=float(max(0.01, args.duration_hours)))
    stamp = start_utc.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("outputs") / f"opt_layer_preference_overnight_{stamp}"
    )
    rounds_dir = output_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)

    history_jsonl = output_dir / "history.jsonl"
    latest_json = output_dir / "latest_summary.json"
    latest_md = output_dir / "latest_report.md"
    run_manifest = output_dir / "run_manifest.json"

    initial_plan = RoundPlan(
        train_examples=int(args.initial_train_examples),
        val_examples=int(args.initial_val_examples),
        test_examples=int(args.initial_test_examples),
        n_pairs_per_example=int(args.initial_pairs_per_example),
        tie_margin=float(args.initial_tie_margin),
        trials_per_setting=int(args.initial_trials_per_setting),
        model_budget=0,
    )

    manifest = {
        "created_at": _now_iso(),
        "seed": int(args.seed),
        "duration_hours": float(args.duration_hours),
        "deadline_utc": deadline.isoformat(),
        "jobs": int(args.jobs),
        "cpu_set": str(args.cpu_set),
        "cpu_ids": [int(c) for c in cpus],
        "target_pref_accuracy": float(args.target_pref_accuracy),
        "max_rounds": int(args.max_rounds),
        "enable_torch": bool(args.enable_torch),
        "initial_plan": asdict(initial_plan),
    }
    _write_json(run_manifest, manifest)

    print(
        f"[{_now_iso()}] start | output_dir={output_dir} | jobs={int(args.jobs)} | "
        f"cpu_set={args.cpu_set} | deadline={deadline.isoformat()}",
        flush=True,
    )

    plan = initial_plan
    round_index = 0
    history_rows: List[Dict[str, Any]] = []
    best_overall: Optional[Dict[str, Any]] = None

    while datetime.now(timezone.utc) < deadline:
        if int(args.max_rounds) > 0 and int(round_index) >= int(args.max_rounds):
            print(f"[{_now_iso()}] reached max_rounds={int(args.max_rounds)}", flush=True)
            break

        round_started = datetime.now(timezone.utc)
        print(
            f"[{_now_iso()}] round={round_index} plan={json.dumps(asdict(plan), sort_keys=True)}",
            flush=True,
        )
        rng = np.random.default_rng(int(args.seed) + 10007 * int(round_index) + 17)
        trial_specs: List[TrialSpec] = []
        for setting_idx, setting in enumerate(SETTINGS):
            model_specs = _sample_model_specs(
                rng=rng,
                n_specs=int(plan.trials_per_setting),
                model_budget=int(plan.model_budget),
                enable_torch=bool(args.enable_torch),
            )
            ds_seed = int(args.seed) + 1_000_000 * int(round_index) + 10_000 * int(setting_idx) + 311
            pref_seed_base = int(args.seed) + 2_000_000 * int(round_index) + 5_000 * int(setting_idx) + 701
            for trial_idx, mspec in enumerate(model_specs):
                trial_seed = int(rng.integers(0, 2**31 - 1))
                trial_specs.append(
                    TrialSpec(
                        round_index=int(round_index),
                        setting=str(setting),
                        model_spec=mspec,
                        trial_seed=int(trial_seed),
                        dataset_seed=int(ds_seed),
                        preference_seed=int(pref_seed_base + trial_idx),
                    )
                )

        total_trials = len(trial_specs)
        print(f"[{_now_iso()}] round={round_index} launching_trials={total_trials}", flush=True)

        round_results: List[Dict[str, Any]] = []
        ok_count = 0
        fail_count = 0
        with ProcessPoolExecutor(
            max_workers=int(max(1, args.jobs)),
            initializer=_worker_init,
            initargs=(tuple(cpus),),
        ) as pool:
            fut_to_spec = {
                pool.submit(_trial_task, spec, plan, bool(args.enable_torch)): spec for spec in trial_specs
            }
            for done_idx, fut in enumerate(as_completed(fut_to_spec), start=1):
                result = fut.result()
                round_results.append(result)
                if bool(result.get("ok")):
                    ok_count += 1
                else:
                    fail_count += 1
                if done_idx == total_trials or done_idx % 8 == 0:
                    print(
                        _progress_line(
                            round_index=int(round_index),
                            done=int(done_idx),
                            total=int(total_trials),
                            ok=int(ok_count),
                            failed=int(fail_count),
                        ),
                        flush=True,
                    )

        by_setting: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SETTINGS}
        for row in round_results:
            setting = str(row.get("setting", ""))
            if setting in by_setting:
                by_setting[setting].append(row)

        best_by_setting: Dict[str, Optional[Dict[str, Any]]] = {}
        top_by_setting: Dict[str, List[Dict[str, Any]]] = {}
        for setting in SETTINGS:
            best = _select_best_trial(by_setting.get(setting, []))
            best_by_setting[setting] = best
            top_rows = sorted(
                [r for r in by_setting.get(setting, []) if bool(r.get("ok"))],
                key=_trial_rank_key,
                reverse=True,
            )[:5]
            top_by_setting[setting] = top_rows

        aggregate = _round_score(best_by_setting)
        next_plan, next_action = _choose_next_plan(
            current=plan,
            best_by_setting=best_by_setting,
            target_pref_accuracy=float(args.target_pref_accuracy),
        )

        round_elapsed = (datetime.now(timezone.utc) - round_started).total_seconds()
        round_summary = {
            "created_at": _now_iso(),
            "round_index": int(round_index),
            "plan": asdict(plan),
            "n_trials_total": int(total_trials),
            "n_trials_ok": int(ok_count),
            "n_trials_failed": int(fail_count),
            "aggregate": aggregate,
            "best_by_setting": best_by_setting,
            "top_by_setting": top_by_setting,
            "next_plan": asdict(next_plan),
            "next_action": str(next_action),
            "elapsed_sec": float(round_elapsed),
            "deadline_utc": deadline.isoformat(),
        }
        round_path = rounds_dir / f"round_{round_index:04d}.json"
        _write_json(round_path, round_summary)
        _append_jsonl(history_jsonl, round_summary)
        history_rows.append(round_summary)

        if best_overall is None:
            best_overall = round_summary
        else:
            prev = best_overall.get("aggregate", {}) if isinstance(best_overall.get("aggregate"), dict) else {}
            prev_score = float(prev.get("mean_pref_non_tie_accuracy", 0.0))
            cur_score = float(aggregate.get("mean_pref_non_tie_accuracy", 0.0))
            if cur_score >= prev_score:
                best_overall = round_summary

        latest_payload = {
            "updated_at": _now_iso(),
            "output_dir": str(output_dir),
            "round_index": int(round_index),
            "aggregate": aggregate,
            "next_action": str(next_action),
            "next_plan": asdict(next_plan),
            "best_overall_round_index": (
                int(best_overall.get("round_index")) if isinstance(best_overall, dict) else None
            ),
            "latest_round_path": str(round_path),
        }
        _write_json(latest_json, latest_payload)
        _render_latest_report(
            out_path=latest_md,
            output_dir=output_dir,
            history=history_rows,
            best_overall=best_overall,
        )

        print(
            f"[{_now_iso()}] round={round_index} done | mean_non_tie="
            f"{float(aggregate.get('mean_pref_non_tie_accuracy', 0.0)):.4f} | "
            f"min_non_tie={float(aggregate.get('min_pref_non_tie_accuracy', 0.0)):.4f} | "
            f"next_action={next_action}",
            flush=True,
        )

        plan = next_plan
        round_index += 1

        now = datetime.now(timezone.utc)
        remaining = (deadline - now).total_seconds()
        if remaining <= 0:
            break
        # Avoid starting another large round when very little time remains.
        est_next = max(300.0, float(round_elapsed) * 0.70)
        if remaining < est_next:
            print(
                f"[{_now_iso()}] stopping: remaining_sec={remaining:.1f} < est_next_round_sec={est_next:.1f}",
                flush=True,
            )
            break

    final_summary = {
        "finished_at": _now_iso(),
        "output_dir": str(output_dir),
        "n_rounds_completed": int(round_index),
        "best_overall_round_index": (
            int(best_overall.get("round_index")) if isinstance(best_overall, dict) else None
        ),
        "best_overall_aggregate": (
            (best_overall.get("aggregate") if isinstance(best_overall, dict) else {})
        ),
        "history_jsonl": str(history_jsonl),
        "latest_summary_json": str(latest_json),
        "latest_report_md": str(latest_md),
    }
    _write_json(output_dir / "final_summary.json", final_summary)
    print(
        f"[{_now_iso()}] finished | rounds={int(round_index)} | "
        f"best_round={final_summary['best_overall_round_index']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
