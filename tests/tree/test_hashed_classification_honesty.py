from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from src.tree.hashed_classification_honesty import (  # noqa: E402
    HashedClassificationConfig,
    run_hashed_classification_experiment,
)


def test_hashed_classification_full_training_is_honest_and_no_c3_breaks_merges():
    base = HashedClassificationConfig(
        n_classes=6,
        vocab_size=4000,
        hash_size=512,
        dirichlet_alpha=1.0,
        min_tokens=256,
        max_tokens=512,
        min_leaf_tokens=8,
        max_leaf_tokens=16,
        train_docs=50,
        test_docs=15,
        state_dim=64,
        hidden_dim=128,
        merger_hidden_dim=64,
        n_epochs=8,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip_norm=1.0,
        leaf_weight=1.0,
        c2_weight=0.1,
        c3_weight=1.0,
        c3_state_weight=0.5,
        audit_policy="all",
        audit_fixed_nodes=0,
        audit_fraction=1.0,
        audit_scale=1.0,
        use_log1p=True,
        normalize_counts=False,
        discrepancy_threshold=0.1,
        seed=0,
        use_cuda=False,
        cuda_device=None,
        torch_threads=0,
    )

    full = run_hashed_classification_experiment(base)
    no_c3 = run_hashed_classification_experiment(
        replace(base, c3_weight=0.0, c3_state_weight=0.0)
    )

    assert full.root_accuracy > 0.90

    assert full.c3.mean_discrepancy < no_c3.c3.mean_discrepancy - 0.03
    assert full.c3.violation_rate < no_c3.c3.violation_rate - 0.20
