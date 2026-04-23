from __future__ import annotations

import pytest

from src.training.config_sections import OptimizerConfig, RunConfig, RuntimeConfig, TrainConfig
from src.training.supervision import (
    DenseSimplexModelConfig,
    DenseSimplexTrainingConfig,
    DenseSupervisionExample,
    build_dense_full_document_supervision_dataset,
    fit_dense_simplex_regressor,
    predict_dense_simplex_regressor,
)


def test_dense_simplex_regressor_fits_simple_topic_targets() -> None:
    train = build_dense_full_document_supervision_dataset(
        [
            DenseSupervisionExample(
                example_id="a",
                features=[1.0, 0.0],
                vector_target=[1.0, 0.0],
            ),
            DenseSupervisionExample(
                example_id="b",
                features=[0.0, 1.0],
                vector_target=[0.0, 1.0],
            ),
            DenseSupervisionExample(
                example_id="c",
                features=[0.8, 0.2],
                vector_target=[0.9, 0.1],
            ),
            DenseSupervisionExample(
                example_id="d",
                features=[0.2, 0.8],
                vector_target=[0.1, 0.9],
            ),
        ],
        application_name="dense_simplex_test",
        supervision_signal_name="document_level_target",
        response_signal_name="topic_mixture",
        law_type="document_level_target",
        split="train",
        response_signal_min=0.0,
        response_signal_max=1.0,
    )
    test = build_dense_full_document_supervision_dataset(
        [
            DenseSupervisionExample(
                example_id="e",
                features=[0.9, 0.1],
                vector_target=[0.95, 0.05],
            ),
            DenseSupervisionExample(
                example_id="f",
                features=[0.1, 0.9],
                vector_target=[0.05, 0.95],
            ),
        ],
        application_name="dense_simplex_test",
        supervision_signal_name="document_level_target",
        response_signal_name="topic_mixture",
        law_type="document_level_target",
        split="test",
        response_signal_min=0.0,
        response_signal_max=1.0,
    )

    model, fit = fit_dense_simplex_regressor(
        train,
        config=DenseSimplexTrainingConfig(
            model=DenseSimplexModelConfig(hidden_dims=(8,)),
            optimizer=OptimizerConfig(learning_rate=5e-2, weight_decay=0.0),
            train=TrainConfig(batch_size=2, epochs=150),
            runtime=RuntimeConfig(device="cpu"),
            run=RunConfig(seed=11),
        ),
    )
    pred = predict_dense_simplex_regressor(model, supervision=test, device="cpu")

    assert fit.input_dim == 2
    assert fit.output_dim == 2
    assert pred.shape == (2, 2)
    assert pred[0][0] > pred[0][1]
    assert pred[1][1] > pred[1][0]
    assert pred.sum(axis=1) == pytest.approx([1.0, 1.0], abs=1e-6)
