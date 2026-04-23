import numpy as np
import pytest

from src.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from src.training.config_sections import RunConfig
from src.training.supervision import (
    AffineSimplexCalibrationConfig,
    DenseScalarRidgeModelConfig,
    DenseScalarRidgeTrainingConfig,
    DenseSimplexForestModelConfig,
    DenseSimplexForestTrainingConfig,
    DenseSupervisionExample,
    apply_dense_affine_simplex_calibrator,
    build_dense_full_document_supervision_dataset,
    build_dense_sampled_substructure_supervision_dataset,
    fit_dense_scalar_ridge_regressor,
    fit_dense_affine_simplex_calibrator,
    fit_dense_simplex_forest_regressor,
    predict_dense_scalar_ridge_regressor,
    predict_dense_simplex_forest_regressor,
)


def test_affine_simplex_calibrator_fits_weighted_supervision() -> None:
    rows = [
        DenseSupervisionExample(
            example_id=f"leaf_{idx}",
            features=features,
            vector_target=target,
            unit_kind=ObservationUnitKind.LEAF,
            sampling=SamplingMetadata(
                document_propensity=1.0,
                unit_propensity=prob,
                label_propensity=1.0,
                joint_propensity=prob,
                sampling_scheme="sampled_substructure_supervision",
                policy_name="unit_test",
                unit_kind=ObservationUnitKind.LEAF,
                supports_ipw_estimation=True,
            ),
        )
        for idx, (features, target, prob) in enumerate(
            [
                ([0.80, 0.20], [0.90, 0.10], 0.4),
                ([0.25, 0.75], [0.15, 0.85], 0.7),
                ([0.55, 0.45], [0.60, 0.40], 0.9),
            ]
        )
    ]
    supervision = build_dense_sampled_substructure_supervision_dataset(
        rows,
        application_name="unit_test",
        supervision_signal_name="substructure_level_target",
        response_signal_name="leaf_topic_mixture",
        law_type="sufficiency",
        split="train",
        response_signal_min=0.0,
        response_signal_max=1.0,
    )
    calibrator, result = fit_dense_affine_simplex_calibrator(
        supervision,
        config=AffineSimplexCalibrationConfig(ridge=1e-6, use_sample_weights=True),
    )
    pred = apply_dense_affine_simplex_calibrator(
        calibrator,
        np.asarray([[0.8, 0.2], [0.25, 0.75]], dtype=np.float64),
    )
    assert pred.shape == (2, 2)
    assert np.all(np.isfinite(pred))
    assert np.allclose(np.sum(pred, axis=1), 1.0)
    assert result.n_train_rows == 3
    assert result.uses_sample_weights is True


def test_dense_simplex_forest_regressor_smoke() -> None:
    pytest.importorskip("sklearn.ensemble")
    rows = [
        DenseSupervisionExample(
            example_id=f"doc_{idx}",
            features=features,
            vector_target=target,
        )
        for idx, (features, target) in enumerate(
            [
                ([0.0, 1.0], [0.05, 0.95]),
                ([0.2, 0.8], [0.15, 0.85]),
                ([0.8, 0.2], [0.85, 0.15]),
                ([1.0, 0.0], [0.95, 0.05]),
            ]
        )
    ]
    supervision = build_dense_full_document_supervision_dataset(
        rows,
        application_name="unit_test",
        supervision_signal_name="document_level_target",
        response_signal_name="document_topic_mixture",
        law_type="document_level_target",
        split="train",
        response_signal_min=0.0,
        response_signal_max=1.0,
    )
    model, result = fit_dense_simplex_forest_regressor(
        supervision,
        config=DenseSimplexForestTrainingConfig(
            model=DenseSimplexForestModelConfig(
                n_estimators=16,
                max_depth=4,
                min_samples_leaf=1,
            ),
            run=RunConfig(seed=7),
        ),
    )
    pred = predict_dense_simplex_forest_regressor(
        model,
        np.asarray([[0.1, 0.9], [0.9, 0.1]], dtype=np.float64),
    )
    assert pred.shape == (2, 2)
    assert np.all(np.isfinite(pred))
    assert np.allclose(np.sum(pred, axis=1), 1.0)
    assert result.n_train_rows == 4
    assert result.input_dim == 2
    assert result.output_dim == 2


def test_dense_scalar_ridge_regressor_smoke() -> None:
    rows = [
        DenseSupervisionExample(
            example_id=f"doc_{idx}",
            features=features,
            scalar_target=target,
        )
        for idx, (features, target) in enumerate(
            [
                ([0.0, 0.0], 0.0),
                ([1.0, 0.0], 1.0),
                ([0.0, 1.0], 2.0),
                ([1.0, 1.0], 3.0),
            ]
        )
    ]
    supervision = build_dense_full_document_supervision_dataset(
        rows,
        application_name="unit_test",
        supervision_signal_name="document_level_target",
        response_signal_name="document_score",
        law_type="document_level_target",
        split="train",
    )
    model, result = fit_dense_scalar_ridge_regressor(
        supervision,
        config=DenseScalarRidgeTrainingConfig(
            model=DenseScalarRidgeModelConfig(ridge_alpha=1e-8)
        ),
    )
    pred = predict_dense_scalar_ridge_regressor(
        model,
        np.asarray([[0.5, 0.5], [1.0, 0.0]], dtype=np.float64),
    )
    assert pred.shape == (2,)
    assert np.all(np.isfinite(pred))
    assert pred[0] == pytest.approx(1.5, abs=1e-4)
    assert pred[1] == pytest.approx(1.0, abs=1e-4)
    assert result.n_train_rows == 4
    assert result.input_dim == 2
