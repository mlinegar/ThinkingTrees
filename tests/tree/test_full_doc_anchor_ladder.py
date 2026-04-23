from __future__ import annotations

from pathlib import Path

import pytest

from src.ctreepo.sim.core.full_doc_anchor_ladder import (
    MarkovFullDocAnchorStageSpec,
    render_full_doc_anchor_ladder_markdown,
    resolve_markov_full_doc_anchor_ladder,
    run_markov_full_doc_anchor_ladder,
)
from src.ctreepo.sim.core.markov_neural_operator_baselines import HAS_NEURAL_OPERATOR


def test_resolve_markov_full_doc_anchor_ladder_reproduction_preset() -> None:
    stages = resolve_markov_full_doc_anchor_ladder(
        preset="recoverable_reproduction_ladder"
    )
    names = [stage.name for stage in stages]
    assert names == [
        "recoverable_current_public",
        "recoverable_medium_operator_public",
        "recoverable_wider_operator_public",
        "recoverable_official_fno_reproduction",
        "recoverable_official_fno_reference",
    ]


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_run_markov_full_doc_anchor_ladder_tiny_smoke(tmp_path: Path) -> None:
    stage = MarkovFullDocAnchorStageSpec(
        name="tiny_smoke",
        description="Tiny smoke test stage.",
        observed_token_profile="smoke",
        config_overrides={
            "model_family": "neural",
            "feature_mode": "token_full",
            "train_docs": 6,
            "val_docs": 2,
            "test_docs": 4,
            "state_dim": 8,
            "hidden_dim": 16,
            "n_epochs": 1,
            "batch_size": 2,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "fno_width": 8,
            "fno_n_modes": 4,
            "fno_n_layers": 1,
            "seed": 7,
        },
    )
    payload = run_markov_full_doc_anchor_ladder(
        stage_specs=[stage],
        output_dir=tmp_path,
        use_cuda=False,
        torch_threads=1,
        preset="test",
    )
    assert payload["simulation"] == "markov_full_doc_anchor_ladder"
    assert payload["preset"] == "test"
    assert len(payload["stages"]) == 1
    result = payload["stages"][0]
    assert result["stage_name"] == "tiny_smoke"
    assert result["source"] == "fresh_run"
    assert result["doc_sequence_test_root_mae"] >= 0.0
    assert result["doc_level_ridge_test_root_mae"] >= 0.0
    assert result["doc_sequence_backend_package"] == "neuraloperator"
    assert result["doc_sequence_operator_class"] == "neuralop.models.FNO"
    assert result["doc_sequence_operator_evidence_status"] == "PROXY_ONLY"
    assert (tmp_path / "stages" / "tiny_smoke.json").exists()
    markdown = render_full_doc_anchor_ladder_markdown(payload)
    assert "Full-Doc Anchor Buildout" in markdown
    assert "tiny_smoke" in markdown
