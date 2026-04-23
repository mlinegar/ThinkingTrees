from src.tree.learned_sketch import (
    DEFAULT_DISTRIBUTION,
    LearnedSketchDataConfig,
    LearnedSketchEvaluationConfig,
    LearnedSketchModel,
    LearnedSketchModelConfig,
    LearnedSketchTrainingConfig,
    evaluate,
    sample_spike_count_mixture_documents,
)


def test_learned_sketch_reports_separate_local_law_metrics():
    config = LearnedSketchTrainingConfig(
        model=LearnedSketchModelConfig(state_dim=4, target_k=4, hidden_dim=8),
        data=LearnedSketchDataConfig(chunk_size=4),
        evaluation=LearnedSketchEvaluationConfig(eval_docs=2),
    )
    docs = sample_spike_count_mixture_documents(
        spec=DEFAULT_DISTRIBUTION,
        n_docs=2,
        seed=123,
    )
    model = LearnedSketchModel(
        n_indicators=config.data.chunk_size,
        state_dim=config.model.state_dim,
        n_types=config.model.target_k,
        hidden_dim=config.model.hidden_dim,
    )
    metrics = evaluate(model, list(docs), config)

    assert metrics.root_oracle_mse >= 0.0
    assert metrics.l1_leaf_error >= 0.0
    assert metrics.l2_merge_error >= 0.0
    assert metrics.l3_idemp_error >= 0.0
    assert metrics.mean_node_oracle_mse >= 0.0
