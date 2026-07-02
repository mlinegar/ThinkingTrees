from __future__ import annotations

import json
from pathlib import Path

from src.runtime.adapters.longbench import LongBenchV2Adapter, LongBenchV2Spec
from src.runtime.backbone import BackboneAdapter, BackboneConfig
from src.runtime.contracts import OperatorInput, OperatorOutput, ProblemSpec, RunUnit, RuntimeConfig
from src.runtime.inference_context import RuntimeInferenceContext
from src.runtime.memory import TokenCounter
from src.runtime.methods import (
    HashingEmbeddingClient,
    MethodResources,
    PAPER_METHOD_ALIASES,
    run_runtime_method,
)
from src.runtime.repair import SimpleRepairPolicy
from src.runtime.verifier import DeterministicVerifier
from src.core.inference_engine import NativeOperatorRegistry


def _problem_and_adapter(tmp_path: Path) -> tuple[ProblemSpec, LongBenchV2Adapter]:
    fixture = tmp_path / "lb.jsonl"
    row = {
        "_id": "method-1",
        "domain": "law",
        "sub_domain": "contracts",
        "difficulty": "easy",
        "length": "short",
        "question": "Which option names the delivery party?",
        "choice_A": "Alpha",
        "choice_B": "Beta",
        "choice_C": "Gamma",
        "choice_D": "Delta",
        "answer": "C",
        "context": "Alpha signs the preface. Gamma is the delivery party. Delta receives the report.",
    }
    fixture.write_text(json.dumps(row) + "\n", encoding="utf-8")
    adapter = LongBenchV2Adapter(
        spec=LongBenchV2Spec(
            task_id="all",
            split="test",
            max_seq_length=4096,
            num_samples=1,
            seed=0,
        ),
        dataset_path=fixture,
    )
    return list(adapter.load_split("test"))[0], adapter


def _unit(method: str) -> RunUnit:
    return RunUnit(
        run_id="r",
        unit_id="u000001",
        phase_id="P0",
        benchmark="longbench_v2",
        task_id="all",
        split="test",
        max_seq_length=4096,
        seed=0,
        num_samples=1,
        method=method,
    )


def test_longbench_method_runners_return_uniform_envelopes(tmp_path: Path) -> None:
    problem, adapter = _problem_and_adapter(tmp_path)
    backbone = BackboneAdapter(
        config=BackboneConfig(base_url="http://localhost:8000/v1", model="mock-model"),
        mock=True,
        enable_cache=False,
    )
    resources = MethodResources(
        backbone=backbone,
        embedding_client=HashingEmbeddingClient(dim=16),
        mock=True,
    )
    runtime = RuntimeConfig(
        cap_tokens=512,
        safety_tokens=16,
        max_output_tokens=16,
        chunk_tokens=12,
        overlap_tokens=0,
        leaf_memory_tokens=16,
        merge_memory_tokens=16,
        retrieval_top_k=1,
        retrieval_chunk_tokens=12,
        retrieval_overlap_tokens=0,
        verifier_enabled=False,
        repair_enabled=False,
    )
    counter = TokenCounter()
    verifier = DeterministicVerifier(counter)
    repair = SimpleRepairPolicy()

    for method in (
        "llm_direct_official",
        "llm_tree_memory",
        "embedding_retrieval_llm",
        "treepo_text_compressor_llm",
        "neural_tree_selector_llm",
    ):
        result = run_runtime_method(
            unit=_unit(method),
            problem=problem,
            adapter=adapter,
            runtime=runtime,
            resources=resources,
            counter=counter,
            verifier=verifier,
            repair=repair,
        )

        assert isinstance(result.prediction, str)
        assert isinstance(result.cost, dict)
        assert isinstance(result.steps, list)
        assert isinstance(result.artifacts, dict)
        assert result.artifacts["method_id"] == method


def test_paper_method_aliases_return_uniform_envelopes(tmp_path: Path) -> None:
    problem, adapter = _problem_and_adapter(tmp_path)
    backbone = BackboneAdapter(
        config=BackboneConfig(base_url="http://localhost:8000/v1", model="mock-model"),
        mock=True,
        enable_cache=False,
    )
    resources = MethodResources(
        backbone=backbone,
        embedding_client=HashingEmbeddingClient(dim=16),
        mock=True,
    )
    runtime = RuntimeConfig(
        cap_tokens=512,
        safety_tokens=16,
        max_output_tokens=16,
        chunk_tokens=12,
        overlap_tokens=0,
        leaf_memory_tokens=16,
        merge_memory_tokens=16,
        retrieval_top_k=1,
        retrieval_chunk_tokens=12,
        retrieval_overlap_tokens=0,
        verifier_enabled=False,
        repair_enabled=False,
    )
    counter = TokenCounter()
    verifier = DeterministicVerifier(counter)
    repair = SimpleRepairPolicy()

    for method, runner_id in PAPER_METHOD_ALIASES.items():
        result = run_runtime_method(
            unit=_unit(method),
            problem=problem,
            adapter=adapter,
            runtime=runtime,
            resources=resources,
            counter=counter,
            verifier=verifier,
            repair=repair,
        )

        assert isinstance(result.prediction, str)
        assert isinstance(result.cost, dict)
        assert isinstance(result.steps, list)
        assert isinstance(result.artifacts, dict)
        assert result.artifacts["method_id"] == method
        assert result.artifacts["runner_id"] == runner_id


def test_neural_tree_selector_can_route_through_operator_surface(tmp_path: Path) -> None:
    problem, adapter = _problem_and_adapter(tmp_path)
    NativeOperatorRegistry.clear()

    def _select(payload: OperatorInput) -> OperatorOutput:
        chunks = list(payload.inputs.get("chunks") or [])
        selected_idx = next((idx for idx, chunk in enumerate(chunks) if "Gamma" in str(chunk)), 0)
        return OperatorOutput(
            data={"selected_indices": [selected_idx], "selected_scores": [1.0]},
            artifacts={"handler": "fixture"},
        )

    NativeOperatorRegistry.register("select_evidence", _select)
    try:
        ctx = RuntimeInferenceContext(
            {
                "surfaces": {
                    "chat_openai": {
                        "engine": "vllm",
                        "base_url": "http://localhost:8000/v1",
                        "model": "mock-chat",
                    },
                    "operator": {
                        "engine": "native_operator",
                        "model": "fixture-selector",
                    },
                }
            },
            mock=True,
        )
        resources = MethodResources(inference_context=ctx, mock=True)
        runtime = RuntimeConfig(
            cap_tokens=512,
            safety_tokens=16,
            max_output_tokens=16,
            retrieval_top_k=1,
            retrieval_chunk_tokens=12,
            retrieval_overlap_tokens=0,
            verifier_enabled=False,
            repair_enabled=False,
        )
        result = run_runtime_method(
            unit=_unit("neural_tree_selector_llm"),
            problem=problem,
            adapter=adapter,
            runtime=runtime,
            resources=resources,
            counter=TokenCounter(),
            verifier=DeterministicVerifier(TokenCounter()),
            repair=SimpleRepairPolicy(),
        )
    finally:
        NativeOperatorRegistry.clear()

    assert isinstance(result.prediction, str)
    assert result.artifacts["selector_backend"] == "state_model"
    assert result.artifacts["operator_model_id"] == "fixture-selector"
    assert result.artifacts["operator_artifacts"]["handler"] == "fixture"
