from pathlib import Path

import pytest

from src.core import gpu_orchestrator as go


def _settings_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def test_orchestrator_uses_backend_specific_venvs():
    cfg = go.OrchestratorConfig(
        task_primary=go.ServerConfig(
            profile="nemotron-30b-nvfp4",
            port=31000,
            cuda_devices="0,1",
            tensor_parallel=2,
            backend="sglang",
            supports_sleep_mode=False,
            enable_sleep_mode=False,
        ),
        task_replica=go.ServerConfig(
            profile="nemotron-30b-nvfp4",
            port=31002,
            cuda_devices="2,3",
            tensor_parallel=2,
            backend="sglang",
            supports_sleep_mode=False,
            enable_sleep_mode=False,
        ),
        genrm=go.ServerConfig(
            profile="genrm-nvfp4",
            port=31001,
            cuda_devices="2,3",
            tensor_parallel=2,
            backend="vllm",
        ),
        venv_path="/tmp/vllm-env",
        sglang_venv_path="/tmp/sglang-env",
        config_path=_settings_path(),
    )

    orchestrator = go.GPUOrchestrator(config=cfg)
    assert orchestrator._task_primary.venv_path == "/tmp/sglang-env"
    assert orchestrator._task_replica.venv_path == "/tmp/sglang-env"
    assert orchestrator._genrm.venv_path == "/tmp/vllm-env"


@pytest.mark.anyio
async def test_managed_server_starts_sglang_via_launcher(monkeypatch):
    captured = {}

    class _DummyProc:
        pid = 424242

        def poll(self):
            return None

    def _fake_popen(cmd, stdout=None, stderr=None, preexec_fn=None, env=None):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(env or {})
        return _DummyProc()

    async def _fake_wait_for_ready(self):
        return None

    monkeypatch.setattr(go.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(go.ManagedServer, "_wait_for_ready", _fake_wait_for_ready)
    monkeypatch.setattr(go, "_listener_pids_on_port", lambda port: [])
    monkeypatch.setattr(go, "kill_process_on_port", lambda port: False)

    server = go.ManagedServer(
        config=go.ServerConfig(
            profile="nemotron-30b-nvfp4",
            port=39000,
            cuda_devices="2,3",
            tensor_parallel=2,
            backend="sglang",
            supports_sleep_mode=False,
            enable_sleep_mode=False,
        ),
        venv_path="/tmp/sglang-env",
        model_path="/tmp/fake-model",
    )

    await server.start()

    cmd = captured["cmd"]
    assert cmd[0] == "/bin/bash"
    assert cmd[1].endswith("scripts/start_sglang.sh")
    assert "nemotron-30b-nvfp4" in cmd
    assert "--port" in cmd and "39000" in cmd
    assert "--cuda-devices" in cmd and "2,3" in cmd
    assert "--sglang-venv-path" in cmd and "/tmp/sglang-env" in cmd
    assert server.is_sleeping is False
    if server._log_file is not None:
        server._log_file.close()
