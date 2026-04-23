from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from src.core.engines import (
    EngineRegistry,
    EngineSurface,
    EngineType,
    default_engine_port,
    normalize_engine_name,
    normalize_fallback_engine_name,
    resolve_engine_base_url,
    resolve_engine_for_usage,
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


def test_engine_registry_resolves_chat_and_diffusion_surfaces() -> None:
    vllm = EngineRegistry.resolve(EngineType.VLLM)
    sglang = EngineRegistry.resolve("sglang")
    vllm_omni = EngineRegistry.resolve("vllm_omni")
    openai = EngineRegistry.resolve("openai")
    custom_http = EngineRegistry.resolve("custom_http")
    symbolic = EngineRegistry.resolve("symbolic_local")

    assert vllm.supports_surface(EngineSurface.CHAT_OPENAI) is True
    assert sglang.supports_surface(EngineSurface.DIFFUSION_GENERATE) is True
    assert vllm_omni.supports_surface(EngineSurface.DIFFUSION_GENERATE) is True
    assert openai.supports_surface(EngineSurface.CHAT_OPENAI) is True
    assert custom_http.supports_surface(EngineSurface.DIFFUSION_GENERATE) is True
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
    assert resolve_engine_base_url("sglang", role="genrm", settings=settings) == "http://localhost:31001/v1"


def test_engine_usage_validation_distinguishes_wrong_surface_and_unmanaged() -> None:
    with pytest.raises(ValueError, match="does not expose the chat_openai surface"):
        resolve_engine_for_usage(
            "vllm_omni",
            surface=EngineSurface.CHAT_OPENAI,
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
