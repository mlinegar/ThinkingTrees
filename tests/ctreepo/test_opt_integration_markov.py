from __future__ import annotations

import numpy as np

from src.ctreepo.opt import collect_pairwise_preferences, to_training_preference_dataset
from src.tree.markov_boundary_honesty_simulation import _make_transition_matrices
from src.tree.markov_changepoint_honesty_simulation import MarkovChangepointConfig, generate_changepoint_docs


class _ChangepointCountGenerator:
    def generate(self, doc, *, n: int, seed: int | None = None):
        del n, seed
        true_count = len(doc.true_boundaries)
        return (true_count, true_count + 1)


def test_opt_layer_handles_markov_changepoint_count_setting() -> None:
    config = MarkovChangepointConfig(
        n_regimes=3,
        vocab_size=16,
        min_tokens=32,
        max_tokens=32,
        min_segments=2,
        max_segments=4,
        min_seg_len=4,
        max_seg_len=16,
        train_docs=1,
        test_docs=0,
        sinkhorn_iters=3,
        transition_log_std=0.8,
        seed=0,
    )
    rng = np.random.default_rng(0)
    transitions = _make_transition_matrices(
        n_classes=int(config.n_regimes),
        vocab_size=int(config.vocab_size),
        log_std=float(config.transition_log_std),
        sinkhorn_iters=int(config.sinkhorn_iters),
        rng=rng,
    )
    docs = generate_changepoint_docs(config, transitions=transitions)
    assert len(docs) == 1

    def utility(doc, count: int) -> float:
        truth = len(doc.true_boundaries)
        return -abs(int(count) - int(truth))

    records = collect_pairwise_preferences(
        docs,
        candidate_generator=_ChangepointCountGenerator(),
        utility_fn=utility,
        rubric="changepoint-count",
        seed=123,
    )
    assert len(records) == 1
    assert records[0].preferred == "A"

    dataset = to_training_preference_dataset(records)
    assert len(dataset) == 1
    assert dataset[0].preferred == "A"

