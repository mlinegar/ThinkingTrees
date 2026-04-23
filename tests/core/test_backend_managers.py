import pytest
from pathlib import Path
from tempfile import gettempdir

from src.core.engines import EngineRegistry, EngineType, build_server_manager
from src.benchmark.throughput import (
    _configure_nvfp4_runtime_env,
    SGLangServerManager,
    ServerManager,
    VLLMServerManager,
    load_sglang_config,
)


def test_backend_managers_implement_server_manager_protocol():
    vllm = VLLMServerManager(profile="nemotron-30b-nvfp4", port=18000)
    sglang = SGLangServerManager(profile="nemotron-30b-nvfp4", port=38000)

    assert isinstance(vllm, ServerManager)
    assert isinstance(sglang, ServerManager)
    assert vllm.capabilities.backend == "vllm"
    assert sglang.capabilities.backend == "sglang"
    assert vllm.capabilities.supports_sleep_mode is True
    assert sglang.capabilities.supports_sleep_mode is False


def test_build_server_manager_uses_shared_engine_registry() -> None:
    vllm = build_server_manager(EngineType.VLLM, profile="nemotron-30b-nvfp4", port=18000)
    sglang = build_server_manager("sglang", profile="nemotron-30b-nvfp4", port=38000)

    assert isinstance(vllm, VLLMServerManager)
    assert isinstance(sglang, SGLangServerManager)
    assert EngineRegistry.resolve("vllm").manager_kind == "vllm"
    assert EngineRegistry.resolve("sglang").manager_kind == "sglang"


@pytest.mark.anyio
async def test_backend_manager_health_returns_bool():
    vllm = VLLMServerManager(profile="nemotron-30b-nvfp4", port=65531)
    sglang = SGLangServerManager(profile="nemotron-30b-nvfp4", port=65530)

    vllm_health = await vllm.health(timeout=0.1)
    sglang_health = await sglang.health(timeout=0.1)

    assert isinstance(vllm_health, bool)
    assert isinstance(sglang_health, bool)


def test_sglang_profile_override_applied_for_qwen80b():
    cfg = load_sglang_config("qwen-80b")

    assert cfg["attention_backend"] == "triton"
    assert cfg["disable_cuda_graph"] is True
    assert cfg["cuda_toolkit_venv_path"] == "/home/mlinegar/vllm-env"


def test_sglang_profile_defaults_remain_for_nemotron():
    cfg = load_sglang_config("nemotron-30b-nvfp4")

    assert cfg["attention_backend"] == ""
    assert cfg["disable_cuda_graph"] is False


def _write_fake_nvcc(path: Path, exit_code: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/usr/bin/env bash\nexit {exit_code}\n")
    path.chmod(0o755)


def test_nvfp4_env_prefers_working_cu13_nvcc(tmp_path: Path):
    bad_cuda_home = tmp_path / "bad_cuda"
    _write_fake_nvcc(bad_cuda_home / "bin" / "nvcc", exit_code=127)

    venv_path = tmp_path / "vllm-env"
    cu13_root = venv_path / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13"
    good_nvcc = cu13_root / "bin" / "nvcc"
    _write_fake_nvcc(good_nvcc, exit_code=0)
    (venv_path / "lib" / "python3.12" / "site-packages" / "nvidia" / "curand" / "include").mkdir(
        parents=True,
        exist_ok=True,
    )

    env = {"CUDA_HOME": str(bad_cuda_home)}
    _configure_nvfp4_runtime_env(env, venv_path=str(venv_path), profile="nemotron-30b-nvfp4")

    assert env["CUDA_HOME"] == str(cu13_root)
    assert env["CUDA_PATH"] == str(cu13_root)
    assert env["FLASHINFER_NVCC"] == str(good_nvcc)
    assert env["CUDACXX"] == str(good_nvcc)
    assert env["VLLM_USE_FLASHINFER_MOE_FP4"] == "1"
    assert env["VLLM_FLASHINFER_MOE_BACKEND"] == "throughput"
    assert env["FLASHINFER_WORKSPACE_BASE"].startswith(str(Path(gettempdir()) / "thinkingtrees" / "flashinfer"))


def test_nvfp4_env_clears_broken_cuda_home_when_no_replacement(tmp_path: Path):
    bad_cuda_home = tmp_path / "bad_cuda"
    _write_fake_nvcc(bad_cuda_home / "bin" / "nvcc", exit_code=127)
    venv_path = tmp_path / "vllm-env"

    env = {"CUDA_HOME": str(bad_cuda_home), "CUDA_PATH": str(bad_cuda_home)}
    _configure_nvfp4_runtime_env(env, venv_path=str(venv_path), profile="genrm-nvfp4")

    assert "CUDA_HOME" not in env
    assert "CUDA_PATH" not in env
    assert env["FLASHINFER_WORKSPACE_BASE"].startswith(str(Path(gettempdir()) / "thinkingtrees" / "flashinfer"))
