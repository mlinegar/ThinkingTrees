"""treepo: simulations + benchmarks for TreePO / C-TreePO."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from treepo.bench.cardinality_recovery import (
    CardinalityRecoveryConfig,
    CardinalityRecoverySummary,
    run_cardinality_recovery_experiment,
)
from treepo.bench.hll_merge_learning import (
    HLLMergeLearningConfig,
    HLLMergeLearningSummary,
    run_hll_merge_learning_experiment,
)
from treepo.hll import (
    HLLConfig,
    HyperLogLogSketch,
    hll_relative_standard_error,
    match_hll_precision_for_bits,
    reduce_hll_sketches,
)

try:
    __version__ = version("treepo")
except (PackageNotFoundError, TypeError, KeyError):  # pragma: no cover
    __version__ = "0.1.0"


__all__ = [
    "__version__",
    "CardinalityRecoveryConfig",
    "CardinalityRecoverySummary",
    "HLLConfig",
    "HLLMergeLearningConfig",
    "HLLMergeLearningSummary",
    "HyperLogLogSketch",
    "hll_relative_standard_error",
    "match_hll_precision_for_bits",
    "reduce_hll_sketches",
    "run_cardinality_recovery_experiment",
    "run_hll_merge_learning_experiment",
]
