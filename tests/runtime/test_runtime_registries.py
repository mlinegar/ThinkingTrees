from __future__ import annotations

import json
from pathlib import Path

from src.runtime.adapters.longbench import LongBenchV2Adapter
from src.runtime.adapters.registry import build_benchmark_adapter
from src.runtime.adapters.ruler import RulerSyntheticAdapter
from src.runtime.contracts import RunUnit
from src.runtime.methods import available_methods, discover_method


def test_benchmark_registry_resolves_longbench_without_network(tmp_path: Path) -> None:
    fixture = tmp_path / "lb.jsonl"
    fixture.write_text(
        json.dumps(
            {
                "_id": "x",
                "domain": "code",
                "sub_domain": "repo",
                "difficulty": "easy",
                "length": "short",
                "question": "Which option is correct?",
                "choice_A": "A1",
                "choice_B": "B1",
                "choice_C": "C1",
                "choice_D": "D1",
                "answer": "A",
                "context": "The context says A1.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    unit = RunUnit(
        run_id="r",
        unit_id="u000001",
        phase_id="P0",
        benchmark="longbench_v2",
        task_id="all",
        split="test",
        max_seq_length=8192,
        seed=0,
        num_samples=1,
        method="llm_direct_official",
    )

    adapter = build_benchmark_adapter(
        spec={"benchmark": {"name": "longbench_v2", "dataset_path": str(fixture)}},
        run_dir=tmp_path,
        unit=unit,
    )

    assert isinstance(adapter, LongBenchV2Adapter)
    assert adapter.primary_metric() == "longbench_v2_accuracy"
    assert len(list(adapter.load_split("test"))) == 1


def test_benchmark_registry_keeps_ruler_resolution_available(tmp_path: Path) -> None:
    unit = RunUnit(
        run_id="r",
        unit_id="u000001",
        phase_id="P0",
        benchmark="ruler_synthetic",
        task_id="vt",
        split="validation",
        max_seq_length=1024,
        seed=0,
        num_samples=1,
        method="runtime_full",
    )

    adapter = build_benchmark_adapter(
        spec={"benchmark": {"name": "ruler_synthetic", "ensure_prepared": False}},
        run_dir=tmp_path,
        unit=unit,
    )

    assert isinstance(adapter, RulerSyntheticAdapter)
    assert adapter.primary_metric() == "ruler_score"


def test_method_registry_exposes_runtime_and_longbench_methods() -> None:
    methods = set(available_methods())

    assert "runtime_full" in methods
    assert "llm_direct_official" in methods
    assert "llm_tree_memory" in methods
    assert "embedding_retrieval_llm" in methods
    assert "treepo_text_compressor_llm" in methods
    assert "neural_tree_selector_llm" in methods
    assert "full_context" in methods
    assert "retrieval" in methods
    assert "summary_tree" in methods
    assert "state_tree" in methods
    assert "neural_operator" in methods
    assert "baseline_llm_raw" in methods
    assert "embedding_proxy_ridge_trained" in methods
    assert "neural_operator_hybrid_raw" in methods
    assert "generator_lora_dpo_trained" in methods


def test_discover_method_maps_method_compare_profile_to_runtime_alias(tmp_path: Path) -> None:
    run_dir = tmp_path / "embedding_proxy_ridge"
    run_dir.mkdir()
    (run_dir / "final_stats.json").write_text(
        json.dumps({"profile": "embedding_proxy_ridge"}), encoding="utf-8"
    )

    trained = discover_method(run_dir, trained=True)
    raw = discover_method(run_dir, trained=False)

    assert trained.name == "embedding_proxy_ridge_trained"
    assert trained.runner_id == "embedding_retrieval_llm"
    assert trained.family == "embedding_proxy_ridge"
    assert trained.trained is True
    assert raw.name == "embedding_proxy_ridge_raw"
    assert raw.trained is False
