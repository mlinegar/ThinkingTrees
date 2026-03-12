#!/usr/bin/env python3
"""Large CPU-only optimizer for LDA topic estimators.

This script performs long-running random search over embedding and
neural+embedding estimator hyperparameters across both simulation pipelines:
- segmented_lda_ctreepo_simulation
- segment_lda_ops_weight_recovery_simulation

It writes incremental artifacts so long runs are auditable and resumable:
- config.json
- candidates.jsonl
- holdout_checks.jsonl
- best.json
- summary.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.segment_lda_ops_weight_recovery_simulation import (
    SegmentLDAOpsWeightRecoveryConfig,
    run_segment_lda_ops_weight_recovery_experiment,
)
from src.tree.segmented_lda_ctreepo_simulation import (
    SegmentedLDACtreePOConfig,
    run_segmented_lda_ctreepo_simulation,
)


def _parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for raw in str(text).replace(" ", ",").split(","):
        x = raw.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _json_dump(path: Path, obj: object) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _jsonl_append(path: Path, row: Dict[str, object]) -> None:
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _stats(xs: Sequence[float]) -> Dict[str, float]:
    ys = [float(x) for x in xs if math.isfinite(float(x))]
    if not ys:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "std": float("nan"),
        }
    ys_s = sorted(ys)
    p90_idx = min(len(ys_s) - 1, max(0, int(math.ceil(0.9 * len(ys_s)) - 1)))
    p90 = ys_s[p90_idx]
    return {
        "mean": float(sum(ys) / len(ys)),
        "median": float(statistics.median(ys)),
        "p90": float(p90),
        "std": float(statistics.pstdev(ys)) if len(ys) > 1 else 0.0,
    }


@dataclass(frozen=True)
class EmbeddingParams:
    svd_extra: int
    kmeans_inits: int
    kmeans_max_iter: int
    temperature: float
    ppmi_shift: float


@dataclass(frozen=True)
class NeuralParams:
    base_estimator: str
    seed_fraction: float
    hidden_dim: int
    steps: int
    lr: float
    weight_decay: float
    mix_samples: int
    mix_temperature: float
    operator_boost: float
    seed_w_min: float
    seed_w_max: float
    sim_temperature: float
    ridge: float


@dataclass(frozen=True)
class Candidate:
    estimator: str
    embedding: EmbeddingParams
    neural: NeuralParams

    def key(self) -> str:
        return json.dumps(
            {
                "estimator": self.estimator,
                "embedding": asdict(self.embedding),
                "neural": asdict(self.neural),
            },
            sort_keys=True,
        )


class HugeOptimizer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = random.Random(int(args.random_seed))

        self.train_docs_grid = _parse_int_list(args.train_docs_grid)
        self.seed_grid = _parse_int_list(args.seed_grid)
        self.holdout_train_docs_grid = _parse_int_list(args.holdout_train_docs_grid)
        self.holdout_seed_grid = _parse_int_list(args.holdout_seed_grid)

        if not self.train_docs_grid:
            raise ValueError("train-docs-grid must be non-empty")
        if not self.seed_grid:
            raise ValueError("seed-grid must be non-empty")
        if not self.holdout_train_docs_grid:
            raise ValueError("holdout-train-docs-grid must be non-empty")
        if not self.holdout_seed_grid:
            raise ValueError("holdout-seed-grid must be non-empty")

        run_name = (
            f"huge_cpu_opt_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
            f"_seed{int(args.random_seed)}"
        )
        self.run_dir = Path(args.out_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.candidates_jsonl = self.run_dir / "candidates.jsonl"
        self.holdout_jsonl = self.run_dir / "holdout_checks.jsonl"
        self.best_json = self.run_dir / "best.json"
        self.summary_json = self.run_dir / "summary.json"

        self.seen: set[str] = set()
        self.evaluated: List[Dict[str, object]] = []
        self.best: Dict[str, object] | None = None

        self.common = {
            "topic_phi_docs": int(args.topic_phi_docs),
            "topic_phi_permute": bool(args.topic_phi_permute),
            "tlda_delta": float(args.tlda_delta),
            "tlda_rate_constant": float(args.tlda_rate_constant),
            "tlda_sigmaK_floor": float(args.tlda_sigmaK_floor),
            "online_tensor_lda_burn_in_docs": int(args.online_tensor_lda_burn_in_docs),
            "online_tensor_lda_batch_docs": int(args.online_tensor_lda_batch_docs),
            "online_tensor_lda_passes": int(args.online_tensor_lda_passes),
            "online_tensor_lda_lr": float(args.online_tensor_lda_lr),
            "online_tensor_lda_grad_clip_norm": float(args.online_tensor_lda_grad_clip_norm),
        }

        self.seg_fixed = {
            "n_topics": int(args.n_topics),
            "vocab_size": int(args.vocab_size),
            "alpha_topic": float(args.alpha_topic),
            "beta_word": float(args.beta_word),
            "n_books_test": int(args.segmented_test_docs),
            "min_segments": int(args.min_segments),
            "max_segments": int(args.max_segments),
            "min_seg_tokens": int(args.min_seg_tokens),
            "max_seg_tokens": int(args.max_seg_tokens),
            "segment_concentration": float(args.segment_concentration),
            "segment_background": float(args.segment_background),
            "fixed_leaf_tokens": int(args.fixed_leaf_tokens),
            "calibration_leaf_query_rate": float(args.calibration_leaf_query_rate),
            "eval_leaf_query_rate": float(args.eval_leaf_query_rate),
            "eval_internal_query_rate": float(args.eval_internal_query_rate),
            "eval_internal_query_design": str(args.eval_internal_query_design),
        }

        self.ops_fixed = {
            "n_topics": int(args.n_topics),
            "vocab_size": int(args.vocab_size),
            "min_tokens": int(args.ops_min_tokens),
            "max_tokens": int(args.ops_max_tokens),
            "min_segments": int(args.min_segments),
            "max_segments": int(args.max_segments),
            "min_seg_len": int(args.ops_min_seg_len),
            "max_seg_len": int(args.ops_max_seg_len),
            "leaf_tokens": int(args.ops_leaf_tokens),
            "align_segments_to_leaves": bool(args.ops_align_segments_to_leaves),
            "doc_topic_concentration": float(args.doc_topic_concentration),
            "topic_process": str(args.ops_topic_process),
            "boundary_profile": str(args.ops_boundary_profile),
            "boundary_profile_strength": float(args.ops_boundary_profile_strength),
            "boundary_profile_seed": int(args.ops_boundary_profile_seed),
            "segment_length_power": float(args.ops_segment_length_power),
            "topic_concentration": float(args.topic_concentration),
            "emission_mode": str(args.ops_emission_mode),
            "anchor_words_per_topic": int(args.ops_anchor_words_per_topic),
            "anchor_multiplier": float(args.ops_anchor_multiplier),
            "relevant_topics": int(args.ops_relevant_topics),
            "theta_scale": float(args.ops_theta_scale),
            "zero_diagonal": bool(args.ops_zero_diagonal),
            "lambda_multiplier": float(args.ops_lambda_multiplier),
            "oracle_noise_std": float(args.ops_oracle_noise_std),
            "audit_policy": str(args.ops_audit_policy),
            "audit_fraction": float(args.ops_audit_fraction),
            "audit_strategy": str(args.ops_audit_strategy),
            "ridge_lambda": float(args.ops_ridge_lambda),
            "topic_source": str(args.ops_topic_source),
            "feature_inference": str(args.ops_feature_inference),
            "run_all_feature_modes": bool(args.ops_run_all_feature_modes),
            "violation_tau": float(args.ops_violation_tau),
            "test_docs": int(args.ops_test_docs),
        }

        _json_dump(
            self.run_dir / "config.json",
            {
                "args": vars(args),
                "run_dir": str(self.run_dir),
                "train_docs_grid": self.train_docs_grid,
                "seed_grid": self.seed_grid,
                "holdout_train_docs_grid": self.holdout_train_docs_grid,
                "holdout_seed_grid": self.holdout_seed_grid,
            },
        )

    def _log(self, text: str) -> None:
        if not bool(self.args.quiet):
            print(text, flush=True)

    def _rand_loguniform(self, lo: float, hi: float) -> float:
        a = math.log(float(lo))
        b = math.log(float(hi))
        return float(math.exp(self.rng.uniform(a, b)))

    def sample_candidate(self) -> Candidate:
        # Bias toward embedding_spectral; include neural_embedding_hybrid regularly.
        estimator = "embedding_spectral" if self.rng.random() < 0.60 else "neural_embedding_hybrid"

        emb = EmbeddingParams(
            svd_extra=self.rng.randint(1, 12),
            kmeans_inits=self.rng.choice([6, 8, 10, 12]),
            kmeans_max_iter=self.rng.choice([60, 80, 100, 120]),
            temperature=self.rng.uniform(0.15, 0.80),
            ppmi_shift=self.rng.uniform(0.80, 1.40),
        )

        if estimator == "neural_embedding_hybrid":
            seed_frac = self.rng.uniform(0.20, 0.80)
            seed_min = 0.15
            seed_max = max(seed_min, min(0.75, seed_min + self.rng.uniform(0.15, 0.45)))
            neu = NeuralParams(
                base_estimator="embedding_spectral",
                seed_fraction=seed_frac,
                hidden_dim=self.rng.choice([16, 24, 32, 48, 64, 96]),
                steps=self.rng.choice([20, 30, 40, 60, 80, 120]),
                lr=self._rand_loguniform(1e-3, 1e-2),
                weight_decay=self._rand_loguniform(1e-5, 1e-3),
                mix_samples=self.rng.choice([32, 64, 96, 128, 192, 256]),
                mix_temperature=self.rng.choice([0.75, 1.0, 1.25, 1.5]),
                operator_boost=self.rng.choice([0.8, 1.0, 1.2, 1.4, 1.8, 2.4]),
                seed_w_min=seed_min,
                seed_w_max=seed_max,
                sim_temperature=self.rng.choice([0.08, 0.12, 0.15, 0.2, 0.3]),
                ridge=self.rng.choice([1e-4, 3e-4, 5e-4, 1e-3, 2e-3]),
            )
        else:
            neu = NeuralParams(
                base_estimator="embedding_spectral",
                seed_fraction=0.35,
                hidden_dim=48,
                steps=60,
                lr=3e-3,
                weight_decay=1e-4,
                mix_samples=128,
                mix_temperature=1.0,
                operator_boost=1.4,
                seed_w_min=0.2,
                seed_w_max=0.55,
                sim_temperature=0.15,
                ridge=1e-3,
            )

        return Candidate(estimator=estimator, embedding=emb, neural=neu)

    def _score(self, seg_root: Dict[str, float], ops_root: Dict[str, float], seg_l2: Dict[str, float], ops_l2: Dict[str, float]) -> float:
        # Honest objective with variance penalty to discourage brittle candidates.
        return float(
            float(seg_root["mean"]) + 0.35 * float(seg_root["std"]) +
            float(ops_root["mean"]) + 0.15 * float(ops_root["std"]) +
            0.15 * float(seg_l2["mean"]) +
            0.10 * float(ops_l2["mean"])
        )

    def evaluate(self, candidate: Candidate, *, train_docs_grid: Sequence[int], seed_grid: Sequence[int]) -> Dict[str, object]:
        rows: List[Dict[str, float]] = []
        t0 = time.perf_counter()

        for td in train_docs_grid:
            for seed in seed_grid:
                seg_cfg = SegmentedLDACtreePOConfig(
                    **self.seg_fixed,
                    n_books_train=int(td),
                    seed=int(seed),
                    topic_phi_estimator=str(candidate.estimator),
                    topic_phi_docs=int(self.common["topic_phi_docs"]),
                    topic_phi_permute=bool(self.common["topic_phi_permute"]),
                    tlda_delta=float(self.common["tlda_delta"]),
                    tlda_rate_constant=float(self.common["tlda_rate_constant"]),
                    tlda_sigmaK_floor=float(self.common["tlda_sigmaK_floor"]),
                    online_tensor_lda_burn_in_docs=int(self.common["online_tensor_lda_burn_in_docs"]),
                    online_tensor_lda_batch_docs=int(self.common["online_tensor_lda_batch_docs"]),
                    online_tensor_lda_passes=int(self.common["online_tensor_lda_passes"]),
                    online_tensor_lda_lr=float(self.common["online_tensor_lda_lr"]),
                    online_tensor_lda_grad_clip_norm=float(self.common["online_tensor_lda_grad_clip_norm"]),
                    embedding_topic_svd_dim_extra=int(candidate.embedding.svd_extra),
                    embedding_topic_kmeans_inits=int(candidate.embedding.kmeans_inits),
                    embedding_topic_kmeans_max_iter=int(candidate.embedding.kmeans_max_iter),
                    embedding_topic_assignment_temperature=float(candidate.embedding.temperature),
                    embedding_topic_ppmi_shift=float(candidate.embedding.ppmi_shift),
                    neural_topic_base_estimator=str(candidate.neural.base_estimator),
                    neural_topic_seed_fraction=float(candidate.neural.seed_fraction),
                    neural_topic_hidden_dim=int(candidate.neural.hidden_dim),
                    neural_topic_steps=int(candidate.neural.steps),
                    neural_topic_lr=float(candidate.neural.lr),
                    neural_topic_weight_decay=float(candidate.neural.weight_decay),
                    neural_topic_mix_samples=int(candidate.neural.mix_samples),
                    neural_topic_mix_temperature=float(candidate.neural.mix_temperature),
                    neural_topic_operator_boost=float(candidate.neural.operator_boost),
                    neural_topic_seed_llm_min_weight=float(candidate.neural.seed_w_min),
                    neural_topic_seed_llm_max_weight=float(candidate.neural.seed_w_max),
                    neural_topic_similarity_temperature=float(candidate.neural.sim_temperature),
                    neural_topic_ridge=float(candidate.neural.ridge),
                )
                seg_out = run_segmented_lda_ctreepo_simulation(seg_cfg)
                seg_budgeted = seg_out.metrics["estimated_calibrated_budgeted"]
                rows.append(
                    {
                        "pipeline": 0.0,
                        "train_docs": float(td),
                        "seed": float(seed),
                        "root": float(seg_budgeted.root_l1_mean),
                        "l2": float(seg_out.topic_meta.get("topic_phi_l2_error_mean", float("nan"))),
                    }
                )

                ops_cfg = SegmentLDAOpsWeightRecoveryConfig(
                    **self.ops_fixed,
                    train_docs=int(td),
                    seed=int(seed),
                    topic_phi_estimator=str(candidate.estimator),
                    topic_phi_docs=int(self.common["topic_phi_docs"]),
                    topic_phi_permute=bool(self.common["topic_phi_permute"]),
                    tlda_delta=float(self.common["tlda_delta"]),
                    tlda_rate_constant=float(self.common["tlda_rate_constant"]),
                    tlda_sigmaK_floor=float(self.common["tlda_sigmaK_floor"]),
                    online_tensor_lda_burn_in_docs=int(self.common["online_tensor_lda_burn_in_docs"]),
                    online_tensor_lda_batch_docs=int(self.common["online_tensor_lda_batch_docs"]),
                    online_tensor_lda_passes=int(self.common["online_tensor_lda_passes"]),
                    online_tensor_lda_lr=float(self.common["online_tensor_lda_lr"]),
                    online_tensor_lda_grad_clip_norm=float(self.common["online_tensor_lda_grad_clip_norm"]),
                    embedding_topic_svd_dim_extra=int(candidate.embedding.svd_extra),
                    embedding_topic_kmeans_inits=int(candidate.embedding.kmeans_inits),
                    embedding_topic_kmeans_max_iter=int(candidate.embedding.kmeans_max_iter),
                    embedding_topic_assignment_temperature=float(candidate.embedding.temperature),
                    embedding_topic_ppmi_shift=float(candidate.embedding.ppmi_shift),
                    neural_topic_base_estimator=str(candidate.neural.base_estimator),
                    neural_topic_seed_fraction=float(candidate.neural.seed_fraction),
                    neural_topic_hidden_dim=int(candidate.neural.hidden_dim),
                    neural_topic_steps=int(candidate.neural.steps),
                    neural_topic_lr=float(candidate.neural.lr),
                    neural_topic_weight_decay=float(candidate.neural.weight_decay),
                    neural_topic_mix_samples=int(candidate.neural.mix_samples),
                    neural_topic_mix_temperature=float(candidate.neural.mix_temperature),
                    neural_topic_operator_boost=float(candidate.neural.operator_boost),
                    neural_topic_seed_llm_min_weight=float(candidate.neural.seed_w_min),
                    neural_topic_seed_llm_max_weight=float(candidate.neural.seed_w_max),
                    neural_topic_similarity_temperature=float(candidate.neural.sim_temperature),
                    neural_topic_ridge=float(candidate.neural.ridge),
                )
                ops_out = run_segment_lda_ops_weight_recovery_experiment(ops_cfg)
                ridge = ops_out.metrics.get("ridge", {}) if isinstance(ops_out.metrics, dict) else {}
                rows.append(
                    {
                        "pipeline": 1.0,
                        "train_docs": float(td),
                        "seed": float(seed),
                        "root": float(ridge.get("root_mae", float("nan"))),
                        "l2": float(ops_out.topic_meta.get("topic_phi_l2_error_mean", float("nan"))),
                    }
                )

        seg_rows = [r for r in rows if int(r["pipeline"]) == 0]
        ops_rows = [r for r in rows if int(r["pipeline"]) == 1]
        seg_root = _stats([r["root"] for r in seg_rows])
        seg_l2 = _stats([r["l2"] for r in seg_rows])
        ops_root = _stats([r["root"] for r in ops_rows])
        ops_l2 = _stats([r["l2"] for r in ops_rows])

        score = self._score(seg_root, ops_root, seg_l2, ops_l2)

        return {
            "candidate": {
                "estimator": candidate.estimator,
                "embedding": asdict(candidate.embedding),
                "neural": asdict(candidate.neural),
            },
            "score": float(score),
            "segmented": {"root": seg_root, "l2": seg_l2},
            "ops": {"root": ops_root, "l2": ops_l2},
            "n_runs": int(len(rows)),
            "runtime_sec": float(time.perf_counter() - t0),
        }

    def _update_best(self, result: Dict[str, object], idx: int, elapsed_sec: float) -> bool:
        improved = False
        if self.best is None or float(result["score"]) < float(self.best["score"]):
            improved = True
            self.best = {
                **result,
                "idx": int(idx),
                "elapsed_sec": float(elapsed_sec),
            }
            _json_dump(self.best_json, self.best)
        return improved

    def run(self) -> None:
        t0 = time.perf_counter()
        budget_sec = float(self.args.time_budget_hours) * 3600.0
        max_candidates = int(self.args.max_candidates)
        holdout_every = int(self.args.holdout_every)

        idx = 0
        while idx < max_candidates:
            elapsed = float(time.perf_counter() - t0)
            if elapsed >= budget_sec:
                break

            # Deduplicate samples.
            cand = None
            for _ in range(200):
                c = self.sample_candidate()
                key = c.key()
                if key in self.seen:
                    continue
                self.seen.add(key)
                cand = c
                break
            if cand is None:
                self._log("no new unique candidates after retries; stopping")
                break

            idx += 1
            c_t0 = time.perf_counter()
            result = self.evaluate(cand, train_docs_grid=self.train_docs_grid, seed_grid=self.seed_grid)
            c_elapsed = float(time.perf_counter() - c_t0)
            elapsed = float(time.perf_counter() - t0)

            row = {
                "idx": int(idx),
                "elapsed_sec": float(elapsed),
                **result,
            }
            self.evaluated.append(row)
            _jsonl_append(self.candidates_jsonl, row)

            improved = self._update_best(result, idx=idx, elapsed_sec=elapsed)
            self._log(
                "[{}/{} | {:.2f}h/{:.2f}h] est={} score={:.6g} seg_root={:.6g} ops_root={:.6g} runtime={:.2f}s{}".format(
                    idx,
                    max_candidates,
                    elapsed / 3600.0,
                    budget_sec / 3600.0,
                    str(result["candidate"]["estimator"]),
                    float(result["score"]),
                    float(result["segmented"]["root"]["mean"]),
                    float(result["ops"]["root"]["mean"]),
                    c_elapsed,
                    " *best" if improved else "",
                )
            )

            if holdout_every > 0 and (idx % holdout_every == 0) and self.best is not None:
                best_cand = self.best["candidate"]
                cand_obj = Candidate(
                    estimator=str(best_cand["estimator"]),
                    embedding=EmbeddingParams(**best_cand["embedding"]),
                    neural=NeuralParams(**best_cand["neural"]),
                )
                h_t0 = time.perf_counter()
                holdout = self.evaluate(
                    cand_obj,
                    train_docs_grid=self.holdout_train_docs_grid,
                    seed_grid=self.holdout_seed_grid,
                )
                h_elapsed = float(time.perf_counter() - h_t0)
                holdout_row = {
                    "idx": int(idx),
                    "elapsed_sec": float(time.perf_counter() - t0),
                    "holdout_runtime_sec": float(h_elapsed),
                    "best_idx": int(self.best["idx"]),
                    "best_score_train": float(self.best["score"]),
                    "holdout": holdout,
                }
                _jsonl_append(self.holdout_jsonl, holdout_row)
                self._log(
                    "  holdout@{}: score={:.6g} seg_root={:.6g} ops_root={:.6g}".format(
                        idx,
                        float(holdout["score"]),
                        float(holdout["segmented"]["root"]["mean"]),
                        float(holdout["ops"]["root"]["mean"]),
                    )
                )

        elapsed_total = float(time.perf_counter() - t0)
        top = sorted(self.evaluated, key=lambda r: float(r["score"]))[: int(max(1, self.args.top_k))]
        summary = {
            "run_dir": str(self.run_dir),
            "elapsed_sec": float(elapsed_total),
            "n_candidates_evaluated": int(len(self.evaluated)),
            "best": self.best,
            "top": top,
        }
        _json_dump(self.summary_json, summary)
        self._log(f"done | run_dir={self.run_dir}")
        if self.best is not None:
            self._log(
                "best | idx={} score={:.6g} est={} seg_root={:.6g} ops_root={:.6g}".format(
                    int(self.best["idx"]),
                    float(self.best["score"]),
                    str(self.best["candidate"]["estimator"]),
                    float(self.best["segmented"]["root"]["mean"]),
                    float(self.best["ops"]["root"]["mean"]),
                )
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Huge CPU optimization for LDA topic estimators.")

    p.add_argument("--out-dir", type=str, default="outputs/lda_topic_estimator_cpu_sweep")
    p.add_argument("--time-budget-hours", type=float, default=8.0)
    p.add_argument("--max-candidates", type=int, default=2000)
    p.add_argument("--holdout-every", type=int, default=20)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--random-seed", type=int, default=20260301)
    p.add_argument("--quiet", action="store_true")

    p.add_argument("--train-docs-grid", type=str, default="64,128,256,512")
    p.add_argument("--seed-grid", type=str, default="0,1,2")
    p.add_argument("--holdout-train-docs-grid", type=str, default="96,192,384")
    p.add_argument("--holdout-seed-grid", type=str, default="3,4")

    p.add_argument("--n-topics", type=int, default=5)
    p.add_argument("--vocab-size", type=int, default=160)
    p.add_argument("--topic-phi-docs", type=int, default=0)
    p.add_argument("--topic-phi-permute", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--alpha-topic", type=float, default=0.2)
    p.add_argument("--beta-word", type=float, default=0.1)
    p.add_argument("--doc-topic-concentration", type=float, default=0.6)
    p.add_argument("--topic-concentration", type=float, default=0.2)

    p.add_argument("--tlda-delta", type=float, default=0.10)
    p.add_argument("--tlda-rate-constant", type=float, default=1.0)
    p.add_argument("--tlda-sigmaK-floor", type=float, default=1e-6)
    p.add_argument("--online-tensor-lda-burn-in-docs", type=int, default=0)
    p.add_argument("--online-tensor-lda-batch-docs", type=int, default=32)
    p.add_argument("--online-tensor-lda-passes", type=int, default=1)
    p.add_argument("--online-tensor-lda-lr", type=float, default=0.1)
    p.add_argument("--online-tensor-lda-grad-clip-norm", type=float, default=1.0)

    p.add_argument("--min-segments", type=int, default=6)
    p.add_argument("--max-segments", type=int, default=10)
    p.add_argument("--segment-concentration", type=float, default=80.0)
    p.add_argument("--segment-background", type=float, default=2.0)

    p.add_argument("--segmented-test-docs", type=int, default=128)
    p.add_argument("--min-seg-tokens", type=int, default=14)
    p.add_argument("--max-seg-tokens", type=int, default=26)
    p.add_argument("--fixed-leaf-tokens", type=int, default=16)
    p.add_argument("--calibration-leaf-query-rate", type=float, default=0.10)
    p.add_argument("--eval-leaf-query-rate", type=float, default=0.0)
    p.add_argument("--eval-internal-query-rate", type=float, default=0.15)
    p.add_argument("--eval-internal-query-design", choices=["none", "uniform", "risk"], default="risk")

    p.add_argument("--ops-test-docs", type=int, default=128)
    p.add_argument("--ops-min-tokens", type=int, default=224)
    p.add_argument("--ops-max-tokens", type=int, default=224)
    p.add_argument("--ops-min-seg-len", type=int, default=24)
    p.add_argument("--ops-max-seg-len", type=int, default=80)
    p.add_argument("--ops-leaf-tokens", type=int, default=16)
    p.add_argument("--ops-align-segments-to-leaves", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ops-topic-process", choices=["segments", "bag_of_words"], default="segments")
    p.add_argument(
        "--ops-boundary-profile",
        choices=["uniform", "start", "middle", "end", "bimodal", "random"],
        default="uniform",
    )
    p.add_argument("--ops-boundary-profile-strength", type=float, default=0.0)
    p.add_argument("--ops-boundary-profile-seed", type=int, default=-1)
    p.add_argument("--ops-segment-length-power", type=float, default=1.0)
    p.add_argument("--ops-emission-mode", choices=["anchored", "disjoint"], default="anchored")
    p.add_argument("--ops-anchor-words-per-topic", type=int, default=8)
    p.add_argument("--ops-anchor-multiplier", type=float, default=12.0)
    p.add_argument("--ops-relevant-topics", type=int, default=2)
    p.add_argument("--ops-theta-scale", type=float, default=1.0)
    p.add_argument("--ops-zero-diagonal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ops-lambda-multiplier", type=float, default=1.0)
    p.add_argument("--ops-oracle-noise-std", type=float, default=0.0)
    p.add_argument("--ops-audit-policy", choices=["all", "fixed", "fraction", "sqrt", "log2"], default="fraction")
    p.add_argument("--ops-audit-fraction", type=float, default=0.2)
    p.add_argument("--ops-audit-strategy", choices=["random", "active_small", "profile"], default="random")
    p.add_argument("--ops-ridge-lambda", type=float, default=1e-3)
    p.add_argument("--ops-topic-source", choices=["true", "infer"], default="infer")
    p.add_argument("--ops-feature-inference", choices=["hard", "soft"], default="hard")
    p.add_argument("--ops-run-all-feature-modes", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ops-violation-tau", type=float, default=0.0)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    opt = HugeOptimizer(args)
    opt.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
