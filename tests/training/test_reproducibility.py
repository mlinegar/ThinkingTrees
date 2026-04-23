from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from src.training.reproducibility import (
    configure_reproducibility,
    write_reproducibility_manifest,
)


def test_configure_reproducibility_reseeds_python_numpy_and_torch() -> None:
    first = configure_reproducibility(123)
    sample_a = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1).item()),
    )

    second = configure_reproducibility(123)
    sample_b = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1).item()),
    )

    assert sample_a == sample_b
    assert first["seed"] == 123
    assert second["seed"] == 123
    assert second["numpy_seed_applied"] is True
    assert second["torch_seed_applied"] is True
    assert second["deterministic_torch_requested"] is True


def test_write_reproducibility_manifest_captures_run_context(tmp_path: Path) -> None:
    applied = configure_reproducibility(77)
    path = write_reproducibility_manifest(
        tmp_path,
        seed=77,
        cli_args={"epochs": 3, "output_dir": tmp_path},
        config={"lr": 1e-3, "batch_size": 4},
        applied=applied,
        extra={"task": "manifesto_rile"},
        command=["python", "scripts/train_ctreepo.py", "--seed", "77"],
    )

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "reproducibility_manifest.json"
    assert payload["seed"] == 77
    assert payload["command"] == ["python", "scripts/train_ctreepo.py", "--seed", "77"]
    assert payload["cli_args"]["epochs"] == 3
    assert payload["config"]["batch_size"] == 4
    assert payload["extra"]["task"] == "manifesto_rile"
    assert payload["applied_reproducibility"]["seed"] == 77
    assert "git" in payload
    assert "torch_runtime" in payload
