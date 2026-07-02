from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from src.config.local_inference import add_local_inference_args, resolve_local_inference_config
from src.core.engines import (
    EngineRegistry,
    EngineSurface,
    EngineType,
    default_engine_port,
    normalize_engine_name,
    normalize_fallback_engine_name,
    resolve_engine_base_url,
    resolve_engine_for_usage,
    resolve_local_chat_endpoints,
)
from src.core.llm_client import LLMConfig, ServerType, engine_client


def test_llm_config_from_engine_uses_engine_defaults() -> None:
    vllm = LLMConfig.from_engine("vllm", model="demo")
    sglang = LLMConfig.from_engine(ServerType.SGLANG, model="demo")

    assert vllm.server_type is ServerType.VLLM
    assert vllm.base_url == "http://localhost:8000/v1"
    assert sglang.server_type is ServerType.SGLANG
    assert sglang.base_url == "http://localhost:30000/v1"


def test_llm_config_from_engine_supports_custom_base_url() -> None:
    custom = LLMConfig.from_engine("custom", model="demo", base_url="http://localhost:9999/v1")

    assert custom.server_type is ServerType.CUSTOM
    assert custom.base_url == "http://localhost:9999/v1"


def test_engine_client_preserves_selected_engine() -> None:
    client = engine_client("sglang", model="demo")

    assert client.config.server_type is ServerType.SGLANG
    assert client.config.base_url == "http://localhost:30000/v1"


def test_engine_registry_resolves_active_surfaces_and_hides_generate_transport() -> None:
    vllm = EngineRegistry.resolve(EngineType.VLLM)
    sglang = EngineRegistry.resolve("sglang")
    vllm_omni = EngineRegistry.resolve("vllm_omni")
    openai = EngineRegistry.resolve("openai")
    custom_http = EngineRegistry.resolve("custom_http")
    native_operator = EngineRegistry.resolve("native_operator")
    symbolic = EngineRegistry.resolve("symbolic_local")

    assert vllm.supports_surface(EngineSurface.CHAT_OPENAI) is True
    assert sglang.supports_surface(EngineSurface.CHAT_OPENAI) is True
    assert sglang.supports_surface(EngineSurface.EMBEDDING) is True
    assert sglang.supports_surface(EngineSurface.DIFFUSION_GENERATE) is False
    assert vllm_omni.supports_surface(EngineSurface.CHAT_OPENAI) is True
    assert vllm_omni.supports_surface(EngineSurface.DIFFUSION_GENERATE) is False
    assert openai.supports_surface(EngineSurface.CHAT_OPENAI) is True
    assert custom_http.supports_surface(EngineSurface.CHAT_OPENAI) is True
    assert custom_http.supports_surface(EngineSurface.DIFFUSION_GENERATE) is False
    assert custom_http.supports_surface(EngineSurface.OPERATOR) is True
    assert native_operator.supports_surface(EngineSurface.OPERATOR) is True
    assert native_operator.default_base_url(surface=EngineSurface.OPERATOR) is None
    assert symbolic.supports_surface(EngineSurface.SYMBOLIC_EXACT) is True
    assert symbolic.default_base_url(surface=EngineSurface.SYMBOLIC_EXACT) is None


def test_shared_engine_helpers_normalize_ports_and_base_urls() -> None:
    settings = {
        "vllm": {"port": 18000},
        "sglang": {"port": 31000, "genrm_port": 31001},
    }

    assert normalize_engine_name("custom") == "custom_http"
    assert normalize_fallback_engine_name("disabled", default=None) == "none"
    assert default_engine_port("vllm", role="task", settings=settings) == 18000
    assert default_engine_port("vllm", role="genrm", settings=settings) == 18001
    assert default_engine_port("sglang", role="task", settings=settings) == 31000
    assert default_engine_port("sglang", role="genrm", settings=settings) == 31001
    assert resolve_engine_base_url("vllm", settings=settings) == "http://localhost:18000/v1"
    assert (
        resolve_engine_base_url("sglang", role="genrm", settings=settings)
        == "http://localhost:31001/v1"
    )


def test_resolve_local_chat_endpoints_normalizes_endpoint_contract() -> None:
    settings = {"sglang": {"host": "0.0.0.0", "port": 31000}}

    endpoints = resolve_local_chat_endpoints(
        "sglang",
        ports=[31000, 31000, 31001],
        settings=settings,
    )

    assert endpoints.engine is EngineType.SGLANG
    assert endpoints.ports == (31000, 31001)
    assert endpoints.base_urls == (
        "http://localhost:31000/v1",
        "http://localhost:31001/v1",
    )
    assert endpoints.primary_port == 31000
    assert endpoints.primary_base_url == "http://localhost:31000/v1"
    assert endpoints.pipeline_base_urls == [
        "http://localhost:31000/v1",
        "http://localhost:31001/v1",
    ]


def test_resolve_local_chat_endpoints_filters_unreachable_alternates() -> None:
    endpoints = resolve_local_chat_endpoints(
        "vllm",
        ports=[18000, 18001],
        filter_unreachable=True,
        endpoint_ready=lambda url: url.endswith(":18001/v1"),
    )

    assert endpoints.ports == (18001,)
    assert endpoints.pipeline_base_urls is None

    with pytest.raises(RuntimeError, match="None of the provided local chat endpoints"):
        resolve_local_chat_endpoints(
            "vllm",
            ports=[18000, 18001],
            filter_unreachable=True,
            endpoint_ready=lambda _url: False,
        )


def test_local_inference_config_uses_settings_defaults_and_maps_kwargs() -> None:
    parser = argparse.ArgumentParser()
    add_local_inference_args(
        parser,
        include_generation=True,
        default_concurrent_requests=111,
        default_batch_size=17,
        default_batch_timeout=0.03,
        default_temperature=0.2,
        default_max_tokens=512,
    )
    args = parser.parse_args([])
    settings = {
        "inference": {"backend": {"task_backend": "sglang", "routing_policy": "document_affinity"}},
        "sglang": {"port": 31000},
    }

    config = resolve_local_inference_config(args, settings=settings)

    assert config.engine == "sglang"
    assert config.primary_port == 31000
    assert config.routing_policy == "document_affinity"
    assert config.max_concurrent_requests == 111
    assert config.batch_size == 17
    assert config.batch_timeout == 0.03
    assert config.temperature == 0.2
    assert config.max_tokens == 512
    assert config.dspy_kwargs()["batch_max_concurrent"] == 111
    assert config.dspy_kwargs()["max_tokens"] == 512
    assert config.pipeline_kwargs(max_concurrent_documents=9)["max_concurrent_documents"] == 9
    assert config.to_dict()["base_urls"] == ["http://localhost:31000/v1"]


def test_local_inference_config_normalizes_explicit_ports_and_timeouts() -> None:
    parser = argparse.ArgumentParser()
    add_local_inference_args(
        parser,
        include_generation=True,
        default_request_timeout_seconds=300.0,
        default_await_response_timeout_seconds=600.0,
    )
    args = parser.parse_args(
        [
            "--engine",
            "vllm",
            "--ports",
            "18000",
            "18000",
            "18001",
            "--batch-size",
            "7",
            "--batch-timeout",
            "0.05",
            "--max-tokens",
            "64",
        ]
    )

    config = resolve_local_inference_config(args, settings={"vllm": {"port": 18000}})

    assert config.ports == (18000, 18001)
    assert config.batch_size == 7
    assert config.batch_timeout == 0.05
    assert config.request_timeout_seconds == 300.0
    assert config.await_response_timeout_seconds == 600.0
    dspy_kwargs = config.dspy_kwargs()
    assert dspy_kwargs["batch_request_timeout"] == 300.0
    assert dspy_kwargs["batch_await_response_timeout"] == 600.0
    assert config.to_dict()["ports"] == [18000, 18001]


def test_engine_usage_validation_distinguishes_wrong_surface_and_unmanaged() -> None:
    with pytest.raises(ValueError, match="does not expose the diffusion_generate surface"):
        resolve_engine_for_usage(
            "vllm",
            surface=EngineSurface.DIFFUSION_GENERATE,
            usage="unit test",
        )

    with pytest.raises(ValueError, match="does not provide a managed local server"):
        resolve_engine_for_usage(
            "openai",
            surface=EngineSurface.CHAT_OPENAI,
            usage="unit test",
            require_managed=True,
        )


def test_lightweight_core_and_config_imports_do_not_require_heavy_optional_deps() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.core.engines; import src.config.settings; import src.core; import src.config; print('ok')",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_start_engine_print_spec_works_without_heavy_optional_deps() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/start_engine.py", "--engine", "vllm", "--print-spec"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "'engine': 'vllm'" in result.stdout
