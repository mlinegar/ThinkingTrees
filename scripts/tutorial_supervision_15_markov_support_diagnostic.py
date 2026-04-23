#!/usr/bin/env python3
"""CPU-only simulation: Markov support concentration, ESS, and calibration."""

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


TRUE_DOCUMENT_TARGET = 1.0
SEQUENCE_LENGTH = 48
BLOCK_LENGTH = 6
N_TRAIN_DOCS = 60
N_EVAL_DOCS = 20
N_TRIALS = 120
CONCENTRATIONS = (0.0, 4.0, 8.0, 16.0)


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
    return np.asarray([float(sum(block) / len(block)) for block in blocks if block], dtype=np.float64)


def _sampling_probabilities(raw_block_scores: np.ndarray, concentration: float) -> np.ndarray:
    logits = float(concentration) * raw_block_scores
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    return np.asarray(weights / np.sum(weights), dtype=np.float64)


def _make_documents(rng: np.random.Generator, n_docs: int) -> list[dict[str, object]]:
    documents = []
    for idx in range(n_docs):
        p00 = float(rng.uniform(0.55, 0.95))
        p11 = float(rng.uniform(0.20, 0.90))
        sequence = _generate_sequence(rng, length=SEQUENCE_LENGTH, p00=p00, p11=p11)
        features = _transition_features(sequence)
        raw_document_fraction = float(sum(sequence) / len(sequence))
        raw_block_scores = _block_scores(sequence, BLOCK_LENGTH)
        centered_block_scores = np.asarray(
            TRUE_DOCUMENT_TARGET + (raw_block_scores - raw_document_fraction),
            dtype=np.float64,
        )
        documents.append(
            {
                "doc_id": f"markov_support_{idx}",
                "features": features,
                "doc_target": TRUE_DOCUMENT_TARGET,
                "raw_document_fraction": raw_document_fraction,
                "raw_block_scores": raw_block_scores,
                "centered_block_scores": centered_block_scores,
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
            original_text=f"markov support doc {doc['doc_id']}",
            response="document_score",
            response_id=f"{doc['doc_id']}:document_score",
        )
        for doc in documents
    ]
    return build_dense_full_document_supervision_dataset(
        rows,
        application_name="tutorial_markov_support_diagnostic",
        supervision_signal_name="document_level_target",
        response_signal_name="fraction_state_1",
        law_type="document_level_target",
        split="train",
        response_signal_min=0.0,
        response_signal_max=1.5,
    )


def _build_sampled_supervision(
    rng: np.random.Generator,
    documents: list[dict[str, object]],
    *,
    concentration: float,
):
    rows: list[DenseSupervisionExample] = []
    for doc in documents:
        probabilities = _sampling_probabilities(
            np.asarray(doc["raw_block_scores"], dtype=np.float64),
            concentration,
        )
        sampled_index = int(rng.choice(len(probabilities), p=probabilities))
        propensity = float(probabilities[sampled_index])
        target = float(np.asarray(doc["centered_block_scores"], dtype=np.float64)[sampled_index])
        rows.append(
            DenseSupervisionExample(
                example_id=f"{doc['doc_id']}:block_{sampled_index}",
                source_doc_id=str(doc["doc_id"]),
                features=list(doc["features"]),
                scalar_target=target,
                original_text=f"sampled block {sampled_index} for {doc['doc_id']}",
                response="block_score",
                response_id=f"{doc['doc_id']}:block_{sampled_index}",
                unit_kind=ObservationUnitKind.LEAF,
                sampling=SamplingMetadata(
                    document_propensity=1.0,
                    unit_propensity=propensity,
                    label_propensity=1.0,
                    sampling_scheme="markov_support_softmax",
                    policy_name=f"softmax_concentration_{concentration}",
                    unit_kind=ObservationUnitKind.LEAF,
                    supports_ipw_estimation=True,
                ),
            )
        )
    return build_dense_sampled_substructure_supervision_dataset(
        rows,
        application_name="tutorial_markov_support_diagnostic",
        supervision_signal_name="substructure_level_target",
        response_signal_name="fraction_state_1",
        law_type="substructure_level_target",
        split="train",
        response_signal_min=0.0,
        response_signal_max=1.5,
    )


def _fit_summary(supervision, eval_features, eval_targets, *, sample_weights=None):
    model, _ = fit_dense_scalar_ridge_regressor(
        supervision,
        config=DenseScalarRidgeTrainingConfig(model=DenseScalarRidgeModelConfig(ridge_alpha=1e-3)),
        sample_weights=sample_weights,
    )
    predictions = predict_dense_scalar_ridge_regressor(model, eval_features)
    return {
        "mean_eval_prediction": float(np.mean(predictions)),
        "mean_eval_mae": float(np.mean(np.abs(predictions - eval_targets))),
    }


def run_example() -> dict[str, object]:
    train_rng = np.random.default_rng(509)
    eval_rng = np.random.default_rng(607)
    sample_rng = np.random.default_rng(701)

    train_documents = _make_documents(train_rng, N_TRAIN_DOCS)
    eval_documents = _make_documents(eval_rng, N_EVAL_DOCS)
    full_document_supervision = _build_full_document_supervision(train_documents)

    eval_features = np.asarray([doc["features"] for doc in eval_documents], dtype=np.float64)
    eval_targets = np.asarray([doc["doc_target"] for doc in eval_documents], dtype=np.float64)
    concentration_results = {}

    for concentration in CONCENTRATIONS:
        naive_predictions = []
        naive_maes = []
        ipw_predictions = []
        ipw_maes = []
        ess_values = []
        for _ in range(N_TRIALS):
            sampled_supervision = _build_sampled_supervision(
                sample_rng,
                train_documents,
                concentration=concentration,
            )
            weights = np.asarray(
                [judgment.ipw_weight() for judgment in sampled_supervision.response_judgments],
                dtype=np.float64,
            )
            ess = float((weights.sum() ** 2) / max(np.sum(weights ** 2), 1e-12))
            ess_values.append(ess)

            naive_fit = _fit_summary(
                sampled_supervision,
                eval_features,
                eval_targets,
                sample_weights=np.ones(len(sampled_supervision.response_judgments), dtype=np.float64),
            )
            ipw_fit = _fit_summary(sampled_supervision, eval_features, eval_targets)
            naive_predictions.append(naive_fit["mean_eval_prediction"])
            naive_maes.append(naive_fit["mean_eval_mae"])
            ipw_predictions.append(ipw_fit["mean_eval_prediction"])
            ipw_maes.append(ipw_fit["mean_eval_mae"])

        concentration_results[str(concentration)] = {
            "mean_ess": round(float(np.mean(ess_values)), 4),
            "min_ess": round(float(np.min(ess_values)), 4),
            "naive_mean_eval_prediction": round(float(np.mean(naive_predictions)), 4),
            "ipw_mean_eval_prediction": round(float(np.mean(ipw_predictions)), 4),
            "naive_mean_eval_mae": round(float(np.mean(naive_maes)), 4),
            "ipw_mean_eval_mae": round(float(np.mean(ipw_maes)), 4),
        }

    return {
        "example": "markov_support_diagnostic",
        "true_document_target": TRUE_DOCUMENT_TARGET,
        "n_train_docs": N_TRAIN_DOCS,
        "n_eval_docs": N_EVAL_DOCS,
        "n_trials": N_TRIALS,
        "concentrations": concentration_results,
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
