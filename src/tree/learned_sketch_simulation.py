"""Compatibility wrapper for the cardinality recovery module.

The cardinality simulation now lives in ThinkingTrees
(``src.tree.cardinality_recovery``), mirroring how the Markov generators are
owned here rather than in the minimal ``treepo`` package. The HLL primitives
come from ``treepo.common`` and the vendored ``src.tree.hll``.
"""

from __future__ import annotations

from treepo.common import VALID_AUDIT_POLICIES, VALID_SCHEDULES, audit_sample_count
from src.tree.hll import (
    HLLConfig,
    HyperLogLogSketch,
    hll_relative_standard_error,
    match_hll_precision_for_bits,
    reduce_hll_sketches,
)
from src.tree.cardinality_recovery import (  # noqa: F401,F403
    APPROX_AUDITED_EVIDENCE,
    DEFAULT_REGULARIZER_WEIGHT,
    DEFAULT_SUMMARY_SHARE,
    DEFAULT_LAW_STRENGTH,
    DEFAULT_LAW_COMPONENT_SHARE,
    PROXY_ONLY_EVIDENCE,
    CardinalityBaselineMetrics,
    CardinalityDocument,
    CardinalityRecoveryConfig,
    CardinalityRecoveryRun,
    CardinalityRecoverySummary,
    ExperimentSummary,
    HLLMetrics,
    LearningRunSummary,
    LearnedMergeableSketch,
    ModelEvalMetrics,
    RegularizedObjectiveMetrics,
    SimulationConfig,
    VALID_SIMULATION_MODES,
    compute_regularized_objective_metrics,
    compute_theoretical_floor_rmse,
    evaluate_exact_set_baseline,
    evaluate_hll_baseline,
    evaluate_learned_model,
    evaluate_model_loss,
    evaluate_sum_leaf_uniques_baseline,
    experiment_rows,
    generate_cardinality_documents,
    run_cardinality_recovery_experiment,
    run_learning_vs_hll_experiment,
    train_learned_model,
)

__all__ = [
    "APPROX_AUDITED_EVIDENCE",
    "DEFAULT_REGULARIZER_WEIGHT",
    "DEFAULT_SUMMARY_SHARE",
    "DEFAULT_LAW_STRENGTH",
    "DEFAULT_LAW_COMPONENT_SHARE",
    "CardinalityBaselineMetrics",
    "CardinalityDocument",
    "CardinalityRecoveryConfig",
    "CardinalityRecoveryRun",
    "CardinalityRecoverySummary",
    "ExperimentSummary",
    "HLLConfig",
    "HLLMetrics",
    "HyperLogLogSketch",
    "LearningRunSummary",
    "LearnedMergeableSketch",
    "ModelEvalMetrics",
    "PROXY_ONLY_EVIDENCE",
    "RegularizedObjectiveMetrics",
    "SimulationConfig",
    "VALID_AUDIT_POLICIES",
    "VALID_SCHEDULES",
    "VALID_SIMULATION_MODES",
    "audit_sample_count",
    "compute_regularized_objective_metrics",
    "compute_theoretical_floor_rmse",
    "evaluate_exact_set_baseline",
    "evaluate_hll_baseline",
    "evaluate_learned_model",
    "evaluate_model_loss",
    "evaluate_sum_leaf_uniques_baseline",
    "experiment_rows",
    "generate_cardinality_documents",
    "hll_relative_standard_error",
    "match_hll_precision_for_bits",
    "reduce_hll_sketches",
    "run_cardinality_recovery_experiment",
    "run_learning_vs_hll_experiment",
    "train_learned_model",
]
