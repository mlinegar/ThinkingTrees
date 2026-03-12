from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[2]
    mod_path = root / "scripts" / "train_ctreepo.py"
    spec = importlib.util.spec_from_file_location("train_ctreepo", str(mod_path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_ctreepo_blocks_model_based_local_law_scoring_by_default(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "train_ctreepo.py",
            "--pilot",
            "--local-law-teacher-port",
            "8001",
        ],
    )

    rc = int(mod.main())

    assert rc == 2


def test_train_ctreepo_blocks_conflicting_local_law_sources(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "train_ctreepo.py",
            "--pilot",
            "--local-law-oracle-module",
            "tests.training.fake_oracle:score_span",
            "--local-law-teacher-port",
            "8001",
        ],
    )

    rc = int(mod.main())

    assert rc == 2


def test_train_ctreepo_blocks_task_model_based_local_law_oracle_by_default(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "train_ctreepo.py",
            "--pilot",
            "--local-law-oracle",
            "task",
            "--local-law-teacher-port",
            "8001",
        ],
    )

    rc = int(mod.main())

    assert rc == 2
