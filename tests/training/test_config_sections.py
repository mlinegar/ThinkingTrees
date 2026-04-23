from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.ctreepo.distillation import DistillationTrainConfig
from src.training.config_sections import RunConfig, TrainConfig, config_to_dict
from src.training.ctreepo_trainer import CTreePOTrainingConfig
from src.training.supervision import DenseScalarTrainingConfig
from src.training.trl_training import TRLTrainingConfig


def test_shared_config_sections_are_frozen_and_json_safe(tmp_path: Path) -> None:
    run = RunConfig(output_dir=tmp_path / "run", dry_run=True, metadata={"phase": "smoke"})
    train = TrainConfig(train_splits=["train", "extra"], batch_size=4, epochs=3)

    payload = config_to_dict({"run": run, "train": train})

    assert payload["run"]["output_dir"] == str(tmp_path / "run")
    assert payload["run"]["metadata"] == {"phase": "smoke"}
    assert payload["train"]["train_splits"] == ["train", "extra"]

    with pytest.raises(FrozenInstanceError):
        run.seed = 99  # type: ignore[misc]


def test_old_flat_training_config_fields_are_rejected() -> None:
    with pytest.raises(TypeError):
        DistillationTrainConfig(dry_run=True)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        CTreePOTrainingConfig(n_epochs=1)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        TRLTrainingConfig(use_propensity_weighting=False)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        DenseScalarTrainingConfig(hidden_dims=tuple())  # type: ignore[call-arg]
