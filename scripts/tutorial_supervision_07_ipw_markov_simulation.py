#!/usr/bin/env python3
"""CPU-only simulation: naive vs IPW under biased local sampling in a Markov setting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from src.training.supervision import (
    DenseScalarRidgeModelConfig,
    DenseScalarRidgeTrainingConfig,
    DenseSupervisionExample,
    build_dense_full_document_supervision_dataset,
    build_dense_sampled_substructure_supervision_dataset,
    fit_dense_scalar_ridge_regressor,
    predict_dense_scalar_ridge_regressor,
)


SEQUENCE_LENGTH = 48
BLOCK_LENGTH = 6
N_TRAIN_DOCS = 80
N_EVAL_DOCS = 20
N_TRIALS = 160
TRUE_DOCUMENT_TARGET = 1.0


def _generate_sequence(
    rng: np.random.Generator,
    *,
    length: int,
    p00: float,
    p11: float,
) -> list[int]:
    state = int(rng.integers(0, 2))
    sequence = [state]
    for _ in range(length - 1):
        if state == 0:
            state = 0 if float(rng.random()) < p00 else 1
        else:
            state = 1 if float(rng.random()) < p11 else 0
        sequence.append(state)
    return sequence


def _transition_features(sequence: list[int]) -> list[float]:
    counts = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    for left, right in zip(sequence[:-1], sequence[1:]):
        counts[(left, right)] += 1
    total = max(1, len(sequence) - 1)
    return [
        counts[(0, 0)] / total,
        counts[(0, 1)] / total,
        counts[(1, 0)] / total,
        counts[(1, 1)] / total,
    ]


def _block_scores(sequence: list[int], block_length: int) -> np.ndarray:
    blocks = [
        sequence[start : start + block_length]
        for start in range(0, len(sequence), block_length)
    ]
    return np.asarray(
        [float(sum(block) / len(block)) for block in blocks if block],
        dtype=np.float64,
    )


def _sampling_probabilities(block_scores: np.ndarray) -> np.ndarray:
    raw = 0.15 + block_scores
    return np.asarray(raw / np.sum(raw), dtype=np.float64)


def _make_documents(
    rng: np.random.Generator,
    n_docs: int,
) -> list[dict[str, object]]:
    documents = []
    for idx in range(n_docs):
        p00 = float(rng.uniform(0.55, 0.95))
        p11 = float(rng.uniform(0.20, 0.90))
        sequence = _generate_sequence(rng, length=SEQUENCE_LENGTH, p00=p00, p11=p11)
        features = _transition_features(sequence)
        raw_document_fraction = float(sum(sequence) / len(sequence))
        raw_block_scores = _block_scores(sequence, BLOCK_LENGTH)
        block_scores = np.asarray(
            TRUE_DOCUMENT_TARGET + (raw_block_scores - raw_document_fraction),
            dtype=np.float64,
        )
        block_probabilities = _sampling_probabilities(raw_block_scores)
        documents.append(
            {
                "doc_id": f"markov_doc_{idx}",
                "features": features,
                "doc_target": TRUE_DOCUMENT_TARGET,
                "block_scores": block_scores,
                "block_probabilities": block_probabilities,
                "raw_document_fraction": raw_document_fraction,
            }
        )
    return documents


def _build_full_document_supervision(documents: list[dict[str, object]]):
    rows = [
        DenseSupervisionExample(
            example_id=str(doc["doc_id"]),
            source_doc_id=str(doc["doc_id"]),
            features=list(doc["features"]),
            scalar_target=float(doc["doc_target"]),
            original_text=f"markov sequence {doc['doc_id']}",
            rubric="Predict the fraction of state-1 tokens from transition features.",
            response="document_score",
            response_id=f"{doc['doc_id']}:document_score",
            truth_label_source="oracle",
        )
        for doc in documents
    ]
    return build_dense_full_document_supervision_dataset(
        rows,
        application_name="tutorial_ipw_markov_simulation",
        supervision_signal_name="document_level_target",
        response_signal_name="fraction_state_1",
        law_type="document_level_target",
        split="train",
        response_signal_min=0.0,
        response_signal_max=1.0,
    )


def _build_sampled_supervision(
    rng: np.random.Generator,
    documents: list[dict[str, object]],
):
    rows: list[DenseSupervisionExample] = []
    preview = []
    for index, doc in enumerate(documents):
        block_scores = np.asarray(doc["block_scores"], dtype=np.float64)
        block_probabilities = np.asarray(doc["block_probabilities"], dtype=np.float64)
        sampled_index = int(rng.choice(len(block_scores), p=block_probabilities))
        sampled_score = float(block_scores[sampled_index])
        propensity = float(block_probabilities[sampled_index])
        row = DenseSupervisionExample(
            example_id=f"{doc['doc_id']}:block_{sampled_index}",
            source_doc_id=str(doc["doc_id"]),
            features=list(doc["features"]),
            scalar_target=sampled_score,
            original_text=f"sampled block {sampled_index} from {doc['doc_id']}",
            rubric="Predict document-level fraction of state-1 tokens from transition features.",
            response="block_score",
            response_id=f"{doc['doc_id']}:block_{sampled_index}",
            unit_kind=ObservationUnitKind.LEAF,
            truth_label_source="oracle",
            sampling=SamplingMetadata(
                document_propensity=1.0,
                unit_propensity=propensity,
                label_propensity=1.0,
                sampling_scheme="markov_block_sampling",
                policy_name="favor_high_state1_blocks",
                unit_kind=ObservationUnitKind.LEAF,
                supports_ipw_estimation=True,
                metadata={"block_index": sampled_index},
            ),
            metadata={"block_index": sampled_index},
        )
        rows.append(row)
        if index < 4:
            preview.append(
                {
                    "document_id": str(doc["doc_id"]),
                    "doc_target": round(float(doc["doc_target"]), 4),
                    "raw_document_fraction": round(float(doc["raw_document_fraction"]), 4),
                    "sampled_block_score": round(sampled_score, 4),
                    "logged_propensity": round(propensity, 4),
                }
            )
    supervision = build_dense_sampled_substructure_supervision_dataset(
        rows,
        application_name="tutorial_ipw_markov_simulation",
        supervision_signal_name="substructure_level_target",
        response_signal_name="fraction_state_1",
        law_type="substructure_level_target",
        split="train",
        response_signal_min=0.0,
        response_signal_max=1.0,
    )
    return supervision, preview


def _fit_summary(
    supervision,
    eval_features: np.ndarray,
    eval_targets: np.ndarray,
    *,
    sample_weights=None,
):
    model, _ = fit_dense_scalar_ridge_regressor(
        supervision,
        config=DenseScalarRidgeTrainingConfig(model=DenseScalarRidgeModelConfig(ridge_alpha=1e-3)),
        sample_weights=sample_weights,
    )
    predictions = predict_dense_scalar_ridge_regressor(model, eval_features)
    return {
        "eval_mae": float(np.mean(np.abs(predictions - eval_targets))),
        "eval_prediction_mean": float(np.mean(predictions)),
    }


def run_example() -> dict[str, object]:
    train_rng = np.random.default_rng(17)
    eval_rng = np.random.default_rng(23)
    sample_rng = np.random.default_rng(31)

    train_documents = _make_documents(train_rng, N_TRAIN_DOCS)
    eval_documents = _make_documents(eval_rng, N_EVAL_DOCS)
    full_document_supervision = _build_full_document_supervision(train_documents)

    eval_features = np.asarray([doc["features"] for doc in eval_documents], dtype=np.float64)
    eval_targets = np.asarray([doc["doc_target"] for doc in eval_documents], dtype=np.float64)
    reference_fit = _fit_summary(full_document_supervision, eval_features, eval_targets)

    naive_maes = []
    ipw_maes = []
    naive_prediction_means = []
    ipw_prediction_means = []
    preview = []

    for trial in range(N_TRIALS):
        sampled_supervision, trial_preview = _build_sampled_supervision(sample_rng, train_documents)
        naive_fit = _fit_summary(
            sampled_supervision,
            eval_features,
            eval_targets,
            sample_weights=np.ones(len(sampled_supervision.response_judgments), dtype=np.float64),
        )
        ipw_fit = _fit_summary(sampled_supervision, eval_features, eval_targets)
        if trial == 0:
            preview = trial_preview
        naive_maes.append(naive_fit["eval_mae"])
        ipw_maes.append(ipw_fit["eval_mae"])
        naive_prediction_means.append(naive_fit["eval_prediction_mean"])
        ipw_prediction_means.append(ipw_fit["eval_prediction_mean"])

    return {
        "example": "ipw_markov_simulation",
        "true_document_target": TRUE_DOCUMENT_TARGET,
        "sequence_length": SEQUENCE_LENGTH,
        "block_length": BLOCK_LENGTH,
        "n_train_docs": N_TRAIN_DOCS,
        "n_eval_docs": N_EVAL_DOCS,
        "n_trials": N_TRIALS,
        "sampling_rule": (
            "one block per document, sampled with probability increasing in raw local "
            "state-1 fraction; centered block labels still average to 1.0"
        ),
        "sample_trial_preview": preview,
        "reference_full_document_fit": {
            "eval_mae": round(reference_fit["eval_mae"], 4),
            "eval_prediction_mean": round(reference_fit["eval_prediction_mean"], 4),
            "eval_target_mean": round(float(np.mean(eval_targets)), 4),
        },
        "naive_summary": {
            "mean_eval_mae": round(float(np.mean(naive_maes)), 4),
            "std_eval_mae": round(float(np.std(naive_maes)), 4),
            "mean_eval_prediction": round(float(np.mean(naive_prediction_means)), 4),
        },
        "ipw_summary": {
            "mean_eval_mae": round(float(np.mean(ipw_maes)), 4),
            "std_eval_mae": round(float(np.std(ipw_maes)), 4),
            "mean_eval_prediction": round(float(np.mean(ipw_prediction_means)), 4),
        },
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
