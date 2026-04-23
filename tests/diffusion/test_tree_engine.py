from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.diffusion.backends import DiffusionBatchResponse, DiffusionGeneration
from src.diffusion.tree_engine import FixedBinaryDiffusionTreeEngine


class _QueueClient:
    def __init__(self, queued_outputs: Sequence[Sequence[str]]) -> None:
        self.backend_name = "mock_backend"
        self._queued_outputs = [list(batch) for batch in queued_outputs]
        self.calls: List[Dict[str, Any]] = []

    def generate(
        self,
        texts: Sequence[str] | str,
        sampling_params: Optional[Mapping[str, Any]] = None,
        engine_options: Optional[Mapping[str, Any]] = None,
    ) -> DiffusionBatchResponse:
        prompts = [texts] if isinstance(texts, str) else list(texts)
        outputs = self._queued_outputs.pop(0)
        assert len(outputs) == len(prompts)
        self.calls.append(
            {
                "texts": prompts,
                "sampling_params": dict(sampling_params or {}),
                "engine_options": dict(engine_options or {}),
            }
        )
        return DiffusionBatchResponse(
            generations=[
                DiffusionGeneration(input_text=prompt, output_text=output)
                for prompt, output in zip(prompts, outputs)
            ],
            latency_seconds=0.01,
            request_payload={"text": prompts},
            raw_response={"text": outputs},
        )


def test_run_fixed_tree_builds_binary_tree_and_tracks_refinements() -> None:
    client = _QueueClient(
        [
            ["leaf-1", "leaf-2", "leaf-3"],
            ["merge-12"],
            ["merge-root"],
            ["refine-1"],
            ["refine-2"],
        ]
    )
    engine = FixedBinaryDiffusionTreeEngine(client)

    result = engine.run_fixed_tree(
        ["alpha", "beta", "gamma"],
        rubric="Keep exact theorem content.",
        refine_rounds=2,
        sampling_params={"temperature": 0.0},
        engine_options={"dllm_algorithm": "LowConfidence", "dllm_algorithm_config": {"threshold": 0.2}},
    )

    assert result.tree.final_summary == "refine-2"
    assert result.tree.node_count == 5
    assert len(result.operations) == 5
    assert result.operations[1].carried_node_ids
    assert result.operations[-1].round_index == 2
    assert result.backend_name == "mock_backend"
    assert result.operations[0].backend_name == "mock_backend"
    assert result.operations[0].engine_options["dllm_algorithm"] == "LowConfidence"
    assert "dllm_algorithm" not in result.to_dict()
    assert "dllm_algorithm_config" not in result.to_dict()
    assert "dllm_algorithm" not in result.to_dict()["operations"][0]
    assert "dllm_algorithm" not in result.tree.metadata
    assert result.tree.metadata["engine_options"]["dllm_algorithm"] == "LowConfidence"


def test_run_fixed_tree_refine_uses_refine_prompt_template() -> None:
    client = _QueueClient([["leaf-1"], ["refine-1"]])
    engine = FixedBinaryDiffusionTreeEngine(client)

    result = engine.run_fixed_tree(
        ["alpha"],
        rubric="Keep exact theorem content.",
        refine_rounds=1,
    )

    assert result.tree.final_summary == "refine-1"
    assert len(client.calls) == 2
    refine_prompt = client.calls[-1]["texts"][0]
    assert "Refinement round: 1" in refine_prompt
    assert "Current summary:" in refine_prompt
    assert "leaf-1" in refine_prompt
