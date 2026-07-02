from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from treepo import Candidate, PreferenceDataset, PreferenceRecord

from src.training.finetune_adapters import (
    list_thinkingtrees_finetune_adapters,
    prepare_finetune_adapter,
    train_finetune_adapter,
)


def _dataset() -> PreferenceDataset:
    return PreferenceDataset(
        [
            PreferenceRecord(
                record_id="root",
                unit_id="doc:root",
                unit_type="root",
                target="f",
                context="Score this document.",
                tree_id="doc",
                doc_id="doc",
                node_id="root",
                level=1,
                position=0,
                candidates=(
                    Candidate(id="good", value="score: 0.8", score=0.8, preferred=True),
                    Candidate(id="bad", value="score: -0.2", score=-0.2),
                ),
            ),
            PreferenceRecord(
                record_id="leaf",
                unit_id="doc:leaf",
                unit_type="qsentence",
                target="g",
                context="Encode this qsentence.",
                tree_id="doc",
                doc_id="doc",
                node_id="leaf",
                level=0,
                position=0,
                parent_id="root",
                candidates=(Candidate(id="gold", value="policy state", score=1.0, preferred=True),),
            ),
        ]
    )


def test_thinkingtrees_adapter_listing_and_prepare(tmp_path: Path) -> None:
    names = {adapter.name for adapter in list_thinkingtrees_finetune_adapters()}

    assert {
        "thinkingtrees_trl_sft",
        "thinkingtrees_trl_dpo",
        "thinkingtrees_trl_reward",
        "thinkingtrees_trl_scalar_reward",
        "thinkingtrees_trl_grpo",
        "thinkingtrees_dspy",
    } <= names

    result = prepare_finetune_adapter(
        "thinkingtrees_trl_dpo",
        _dataset(),
        tmp_path / "prepared",
        save_hf=False,
    )

    assert result["adapter"] == "thinkingtrees_trl_dpo"
    assert result["core_adapter"] == "trl_dpo"
    row = json.loads(Path(result["files"]["dpo"]).read_text(encoding="utf-8").splitlines()[0])
    assert set(row) >= {"prompt", "chosen", "rejected", "metadata"}


def test_trl_sft_adapter_routes_to_existing_wrapper(monkeypatch: Any, tmp_path: Path) -> None:
    calls: dict[str, Any] = {}

    def fake_train_sft(**kwargs: Any) -> str:
        calls.update(kwargs)
        return str(tmp_path / "sft_model")

    monkeypatch.setitem(
        sys.modules,
        "src.training.trl_training",
        SimpleNamespace(train_sft=fake_train_sft),
    )

    result = train_finetune_adapter(
        "thinkingtrees_trl_sft",
        _dataset(),
        tmp_path / "model",
        model_name="small-gemma",
        dry_run=False,
    )

    assert result["artifact"] == str(tmp_path / "sft_model")
    assert calls["model_name"] == "small-gemma"
    assert calls["output_dir"] == tmp_path / "model"
    assert calls["records"][0]["prompt"] == "Score this document."
    assert calls["records"][0]["completion"] == "score: 0.8"


def test_trl_dpo_adapter_routes_dataset_to_existing_wrapper(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}

    def fake_train_dpo(**kwargs: Any) -> str:
        calls.update(kwargs)
        return str(tmp_path / "dpo_model")

    monkeypatch.setitem(
        sys.modules,
        "src.training.trl_training",
        SimpleNamespace(train_dpo=fake_train_dpo),
    )
    dataset = _dataset()

    result = train_finetune_adapter(
        "thinkingtrees_trl_dpo",
        dataset,
        tmp_path / "model",
        model_name="small-gemma",
        ref_model_name="small-gemma-ref",
        dry_run=False,
    )

    assert result["artifact"] == str(tmp_path / "dpo_model")
    assert calls["dataset"] is dataset
    assert calls["ref_model_name"] == "small-gemma-ref"


def test_trl_scalar_reward_adapter_uses_supervised_scores(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}

    def fake_train_scalar_reward_records(**kwargs: Any) -> str:
        calls.update(kwargs)
        return str(tmp_path / "reward_model")

    monkeypatch.setitem(
        sys.modules,
        "src.training.trl_training",
        SimpleNamespace(train_scalar_reward_records=fake_train_scalar_reward_records),
    )

    result = train_finetune_adapter(
        "thinkingtrees_trl_scalar_reward",
        _dataset(),
        tmp_path / "model",
        model_name="small-gemma",
        dry_run=False,
    )

    assert result["artifact"] == str(tmp_path / "reward_model")
    assert calls["records"][0]["response"] == "score: 0.8"
    assert calls["records"][0]["score"] == 0.8


def test_trl_grpo_adapter_requires_reward_funcs_for_training(tmp_path: Path) -> None:
    result = train_finetune_adapter(
        "thinkingtrees_trl_grpo",
        _dataset(),
        tmp_path / "model",
        model_name="small-gemma",
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert "reward_funcs" in result["missing"]


def test_dspy_adapter_can_call_existing_family_runtime(tmp_path: Path) -> None:
    class FakeFamily:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def train_g(self, **kwargs: Any) -> str:
            self.calls.append(kwargs)
            return str(tmp_path / "compiled_g.json")

    family = FakeFamily()
    traces = [object()]

    result = train_finetune_adapter(
        "thinkingtrees_dspy",
        _dataset(),
        tmp_path / "dspy",
        dry_run=False,
        family_runtime=family,
        kind="g",
        traces=traces,
        g_init="raw_concat",
        f="compiled_f",
        iteration=2,
    )

    assert result["artifact"] == str(tmp_path / "compiled_g.json")
    assert family.calls[0]["traces"] is traces
    assert family.calls[0]["g_init"] == "raw_concat"
    assert family.calls[0]["f"] == "compiled_f"
    assert family.calls[0]["iteration"] == 2


def test_importing_thinkingtrees_finetune_adapters_is_lazy() -> None:
    code = """
import json, sys
import src.training.finetune_adapters
mods = ["trl", "peft", "accelerate", "dspy", "sentence_transformers"]
print(json.dumps({name: name in sys.modules for name in mods}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded == {
        "accelerate": False,
        "dspy": False,
        "peft": False,
        "sentence_transformers": False,
        "trl": False,
    }
