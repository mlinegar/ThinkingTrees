from src.tree.learned_sketch import (
    DEFAULT_DISTRIBUTION,
    LearnedSketchModel,
    TrainingConfig,
    evaluate,
    sample_spike_count_mixture_documents,
)


def test_learned_sketch_reports_separate_local_law_metrics():
    config = TrainingConfig(
        state_dim=4,
        target_k=4,
        chunk_size=4,
        hidden_dim=8,
        eval_docs=2,
        seed=0,
    )
    docs = sample_spike_count_mixture_documents(
        spec=DEFAULT_DISTRIBUTION,
        n_docs=2,
        seed=123,
    )
    model = LearnedSketchModel(
        n_indicators=config.chunk_size,
        state_dim=config.state_dim,
        n_types=config.target_k,
        hidden_dim=config.hidden_dim,
    )
    metrics = evaluate(model, list(docs), config)

    assert metrics.root_oracle_mse >= 0.0
    assert metrics.l1_leaf_error >= 0.0
    assert metrics.l2_merge_error >= 0.0
    assert metrics.l3_idemp_error >= 0.0
    assert metrics.mean_node_oracle_mse >= 0.0
