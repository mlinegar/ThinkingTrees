from __future__ import annotations

from typing import Any, Dict, List

from src.diffusion.backends import SGLangDiffusionBackend, VLLMOmniDiffusionBackend, build_diffusion_backend
from src.diffusion.sglang_client import SGLangDiffusionClient


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class _FakeSession:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: List[Dict[str, Any]] = []

    def post(self, url: str, json: Dict[str, Any], timeout: float) -> _FakeResponse:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse(self.payload)


def test_generate_posts_to_generate_and_parses_choices() -> None:
    session = _FakeSession({"choices": [{"text": "leaf-a"}, {"text": "leaf-b"}]})
    client = SGLangDiffusionClient(
        base_url="http://localhost:30000",
        model="demo-model",
        session=session,
    )

    result = client.generate(
        ["prompt-a", "prompt-b"],
        sampling_params={"temperature": 0.0, "max_new_tokens": 32},
        engine_options={
            "dllm_algorithm": "LowConfidence",
            "dllm_algorithm_config": {"threshold": 0.15},
        },
    )

    assert result.texts == ["leaf-a", "leaf-b"]
    assert session.calls[0]["url"] == "http://localhost:30000/generate"
    assert session.calls[0]["json"]["model"] == "demo-model"
    assert session.calls[0]["json"]["temperature"] == 0.0
    assert session.calls[0]["json"]["max_new_tokens"] == 32
    assert session.calls[0]["json"]["dllm_algorithm"] == "LowConfidence"
    assert session.calls[0]["json"]["dllm_algorithm_config"] == {"threshold": 0.15}
    assert result.telemetry["engine_options"]["dllm_algorithm"] == "LowConfidence"
    assert "dllm_algorithm" not in result.telemetry
    assert "dllm_algorithm_config" not in result.telemetry


def test_generate_accepts_json_algorithm_config_string() -> None:
    session = _FakeSession({"text": "single-output"})
    client = SGLangDiffusionClient(base_url="http://localhost:30000", session=session)

    result = client.generate(
        "prompt",
        sampling_params={"temperature": 0.3},
        dllm_algorithm="JointThreshold",
        dllm_algorithm_config='{"joint_threshold": 0.7}',
    )

    assert result.texts == ["single-output"]
    assert session.calls[0]["json"]["dllm_algorithm_config"] == {"joint_threshold": 0.7}


def test_engine_options_and_aliases_merge_without_overwriting() -> None:
    session = _FakeSession({"text": "single-output"})
    client = SGLangDiffusionClient(base_url="http://localhost:30000", session=session)

    client.generate(
        "prompt",
        engine_options={"dllm_algorithm": "JointThreshold"},
        dllm_algorithm="LowConfidence",
        dllm_algorithm_config={"threshold": 0.15},
    )

    assert session.calls[0]["json"]["dllm_algorithm"] == "JointThreshold"
    assert session.calls[0]["json"]["dllm_algorithm_config"] == {"threshold": 0.15}


def test_backend_factory_builds_engine_specific_adapters() -> None:
    sglang = build_diffusion_backend("sglang", base_url="http://localhost:30000")
    vllm_omni = build_diffusion_backend("vllm-omni", base_url="http://localhost:8000")

    assert isinstance(sglang, SGLangDiffusionBackend)
    assert isinstance(vllm_omni, VLLMOmniDiffusionBackend)
    assert sglang.backend_name == "sglang"
    assert vllm_omni.backend_name == "vllm_omni"
