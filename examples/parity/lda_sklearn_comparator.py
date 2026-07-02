#!/usr/bin/env python3
"""Compare the LDA example world to scikit-learn LDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import LatentDirichletAllocation

from src.ctreepo.sim.core import lda_tree_recovery as base


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("smoke", "confirmation", "1k", "2k"), default="smoke")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/parity_memos/lda_sklearn_baseline.json"),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = _config_for_preset(str(args.preset))
    world = base.sample_lda_tree_recovery_world(cfg)
    x_train = _counts(world.docs_train, cfg.vocab_size)
    x_test = _counts(world.docs_test, cfg.vocab_size)
    true_pi = np.stack(
        [np.asarray(doc.topic_weights, dtype=np.float64) for doc in world.docs_test],
        axis=0,
    )
    true_phi = np.stack([np.asarray(t, dtype=np.float64) for t in world.topics_phi], axis=0)

    model = LatentDirichletAllocation(
        n_components=cfg.n_topics,
        doc_topic_prior=cfg.doc_topic_concentration / cfg.n_topics,
        topic_word_prior=cfg.topic_concentration / cfg.vocab_size,
        learning_method="batch",
        max_iter=80 if args.preset == "confirmation" else 50,
        random_state=cfg.seed,
    )
    theta_hat = model.fit(x_train).transform(x_test)
    phi_hat = model.components_ / np.maximum(model.components_.sum(axis=1, keepdims=True), 1e-12)

    similarity = _normalize(phi_hat) @ _normalize(true_phi).T
    rows, cols = linear_sum_assignment(-similarity)
    aligned_theta = np.zeros_like(theta_hat)
    for estimated_topic, true_topic in zip(rows, cols):
        aligned_theta[:, true_topic] = theta_hat[:, estimated_topic]

    pi_l1 = np.sum(np.abs(aligned_theta - true_pi), axis=1)
    true_utility = np.asarray([
        base._utility_from_pi(
            np.asarray(doc.topic_weights, dtype=np.float64),
            theta=world.theta_true,
            W_base=world.W_base,
            lambda_multiplier=cfg.lambda_multiplier,
        )
        for doc in world.docs_test
    ])
    estimated_utility = np.asarray([
        base._utility_from_pi(
            row,
            theta=world.theta_true,
            W_base=world.W_base,
            lambda_multiplier=cfg.lambda_multiplier,
        )
        for row in aligned_theta
    ])

    payload = {
        "experiment": "lda_sklearn_comparator",
        "preset": str(args.preset),
        "comparator": "sklearn.decomposition.LatentDirichletAllocation",
        "config": dict(cfg.__dict__),
        "metrics": {
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "topic_cosine_mean_after_alignment": float(np.mean(similarity[rows, cols])),
            "topic_cosine_min_after_alignment": float(np.min(similarity[rows, cols])),
            "pi_l1_to_true_mean": float(np.mean(pi_l1)),
            "pi_l1_to_true_median": float(np.median(pi_l1)),
            "pi_l1_to_true_p95": float(np.percentile(pi_l1, 95)),
            "utility_abs_to_true_mean": float(np.mean(np.abs(estimated_utility - true_utility))),
            "utility_abs_to_true_median": float(np.median(np.abs(estimated_utility - true_utility))),
        },
        "alignment": {
            "estimated_topic_to_true_topic": {
                str(int(row)): int(col) for row, col in zip(rows, cols)
            }
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))
    return 0


def _config_for_preset(preset: str) -> base.LDATreeRecoveryConfig:

    if preset == "1k":
        return base.LDATreeRecoveryConfig(
            n_topics=6,
            vocab_size=128,
            min_tokens=128,
            max_tokens=128,
            doc_topic_concentration=0.6,
            topic_concentration=0.2,
            emission_mode="anchored",
            anchor_words_per_topic=8,
            anchor_multiplier=20.0,
            relevant_topics=3,
            theta_scale=1.0,
            zero_diagonal=False,
            lambda_multiplier=1.0,
            leaf_tokens=16,
            train_docs=1024,
            test_docs=256,
            inference_prior_mass=0.25,
            inference_max_iter=80,
            inference_tol=1e-6,
            seed=109,
        )
    if preset == "2k":
        return base.LDATreeRecoveryConfig(
            n_topics=6,
            vocab_size=128,
            min_tokens=128,
            max_tokens=128,
            doc_topic_concentration=0.6,
            topic_concentration=0.2,
            emission_mode="anchored",
            anchor_words_per_topic=8,
            anchor_multiplier=20.0,
            relevant_topics=3,
            theta_scale=1.0,
            zero_diagonal=False,
            lambda_multiplier=1.0,
            leaf_tokens=16,
            train_docs=2048,
            test_docs=512,
            inference_prior_mass=0.25,
            inference_max_iter=80,
            inference_tol=1e-6,
            seed=113,
        )
    if preset == "confirmation":
        return base.LDATreeRecoveryConfig(
            n_topics=6,
            vocab_size=128,
            min_tokens=128,
            max_tokens=128,
            doc_topic_concentration=0.6,
            topic_concentration=0.2,
            emission_mode="anchored",
            anchor_words_per_topic=8,
            anchor_multiplier=20.0,
            relevant_topics=3,
            theta_scale=1.0,
            zero_diagonal=False,
            lambda_multiplier=1.0,
            leaf_tokens=16,
            train_docs=128,
            test_docs=64,
            inference_prior_mass=0.25,
            inference_max_iter=80,
            inference_tol=1e-6,
            seed=29,
        )
    return base.LDATreeRecoveryConfig(
        n_topics=4,
        vocab_size=64,
        min_tokens=64,
        max_tokens=64,
        doc_topic_concentration=0.6,
        topic_concentration=0.2,
        emission_mode="anchored",
        anchor_words_per_topic=4,
        anchor_multiplier=20.0,
        relevant_topics=2,
        theta_scale=1.0,
        zero_diagonal=False,
        lambda_multiplier=1.0,
        leaf_tokens=8,
        train_docs=16,
        test_docs=8,
        inference_prior_mass=0.25,
        inference_max_iter=40,
        inference_tol=1e-6,
        seed=0,
    )


def _counts(docs, vocab_size: int) -> np.ndarray:
    return np.stack(
        [base._counts_from_tokens(doc.tokens, vocab_size=vocab_size) for doc in docs],
        axis=0,
    ).astype(np.float64)


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


if __name__ == "__main__":
    raise SystemExit(main())
