from __future__ import annotations

from src.tree.learned_sketch_simulation import (
    HLLConfig,
    HyperLogLogSketch,
    SimulationConfig,
    match_hll_precision_for_bits,
)
from src.tree.hll_merge_learning_simulation import (
    ExactMaxMerger,
    HLLMergeLearningConfig,
)


def test_compat_wrapper_exports_public_treepo_symbols() -> None:
    cfg = HLLConfig(precision=6, hash_bits=64)
    sk = HyperLogLogSketch(cfg).update([1, 2, 3, 3, 4])
    assert sk.estimate() > 0.0
    assert match_hll_precision_for_bits(256) >= 4
    assert isinstance(SimulationConfig(), SimulationConfig)
    assert isinstance(HLLMergeLearningConfig(), HLLMergeLearningConfig)
    assert isinstance(ExactMaxMerger(), ExactMaxMerger)
