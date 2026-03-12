from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from src.ctreepo.opt import (
    IPWMetadata,
    collect_pairwise_preferences,
    collect_proxy_training_data,
    to_training_preference_dataset,
)
from src.tree.mergeable_ablation import ToyTokenDocument, true_spike_count


@dataclass(frozen=True)
class _SpikeCountMergeableSketch:
    """Exact mergeable sketch for spike count (leaf count + additive merge)."""

    spike_threshold: float = 0.90

    def leaf(self, scores: Sequence[float]) -> int:
        return sum(1 for v in scores if float(v) >= float(self.spike_threshold))

    def merge(self, left: int, right: int) -> int:
        return int(left) + int(right)

    def compress(self, doc: ToyTokenDocument) -> int:
        # Two-leaf toy tree: first half + second half.
        scores = list(doc.token_scores)
        mid = len(scores) // 2
        return self.merge(self.leaf(scores[:mid]), self.leaf(scores[mid:]))


class _TwoSketchCandidateGenerator:
    def __init__(self, exact: _SpikeCountMergeableSketch):
        self.exact = exact

    def generate(self, doc: ToyTokenDocument, *, n: int, seed: Optional[int] = None):
        del n, seed
        a = self.exact.compress(doc)
        # Misspecified sketch: "count spikes only in first half" (non-mergeable failure mode).
        scores = list(doc.token_scores)
        mid = len(scores) // 2
        b = self.exact.leaf(scores[:mid])
        return (a, b)


def test_opt_layer_handles_mergeable_spike_sketch_setting() -> None:
    scores = [0.1] * 8 + [0.99, 0.95, 0.93] + [0.1] * 5
    doc = ToyTokenDocument(token_scores=tuple(scores), proxy_scores=tuple(scores))
    oracle = true_spike_count(doc.token_scores)
    assert oracle == 3

    sketch = _SpikeCountMergeableSketch(spike_threshold=0.90)

    def utility(d: ToyTokenDocument, count: int) -> float:
        return -abs(int(count) - int(true_spike_count(d.token_scores)))

    records = collect_pairwise_preferences(
        [doc],
        candidate_generator=_TwoSketchCandidateGenerator(sketch),
        utility_fn=utility,
        rubric="spike-count",
        seed=0,
    )
    assert records[0].preferred == "A"

    dataset = to_training_preference_dataset(records)
    assert dataset[0].preferred == "A"


def test_opt_layer_collects_proxy_training_data_for_sketches() -> None:
    docs = []
    for n_spikes in range(4):
        scores = [0.1] * 8 + [0.99] * n_spikes + [0.1] * (8 - n_spikes)
        docs.append(ToyTokenDocument(token_scores=tuple(scores), proxy_scores=tuple(scores)))

    sketch = _SpikeCountMergeableSketch(spike_threshold=0.90)
    xs, ys, weights = collect_proxy_training_data(
        docs,
        compressor=sketch,
        oracle=lambda d: true_spike_count(d.token_scores),
        ipw_fn=lambda _d: IPWMetadata(doc_propensity=0.5),
    )
    assert weights is not None
    assert len(xs) == len(ys) == len(weights) == 4
    assert xs == ys  # exact sketch should match oracle count in this setting.
