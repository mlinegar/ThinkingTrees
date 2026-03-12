import sys
from pathlib import Path

from src.core import gpu_orchestrator as go


def _touch_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\n")
    path.chmod(0o755)


def test_resolve_vllm_python_prefers_env_override_when_probe_passes(monkeypatch, tmp_path: Path) -> None:
    override_py = tmp_path / "override" / "bin" / "python"
    preferred_py = tmp_path / "preferred" / "bin" / "python"
    _touch_executable(override_py)
    _touch_executable(preferred_py)

    monkeypatch.setenv("TT_VLLM_PYTHON", str(override_py))
    monkeypatch.setattr(go.sys, "executable", str(preferred_py))

    def _fake_probe(path: str, cuda_devices: str) -> bool:
        return Path(path) == override_py

    monkeypatch.setattr(go, "_python_supports_vllm_with_cuda", _fake_probe)

    resolved, use_isolation = go._resolve_vllm_python_interpreter(str(preferred_py.parent.parent), "0,1")
    assert Path(resolved) == override_py
    assert use_isolation is True


def test_resolve_vllm_python_falls_back_to_existing_candidate(monkeypatch, tmp_path: Path) -> None:
    preferred_py = tmp_path / "vllm-env" / "bin" / "python"
    _touch_executable(preferred_py)
    monkeypatch.delenv("TT_VLLM_PYTHON", raising=False)
    monkeypatch.delenv("VLLM_PYTHON", raising=False)
    monkeypatch.setattr(go.sys, "executable", sys.executable)

    monkeypatch.setattr(go, "_python_supports_vllm_with_cuda", lambda path, cuda_devices: False)

    resolved, use_isolation = go._resolve_vllm_python_interpreter(str(preferred_py.parent.parent), "0,1")
    assert Path(resolved) == preferred_py
    assert use_isolation is True


def test_resolve_vllm_python_uses_sys_executable_when_preferred_fails_probe(
    monkeypatch,
    tmp_path: Path,
) -> None:
    preferred_py = tmp_path / "vllm-env" / "bin" / "python"
    _touch_executable(preferred_py)
    monkeypatch.delenv("TT_VLLM_PYTHON", raising=False)
    monkeypatch.delenv("VLLM_PYTHON", raising=False)
    monkeypatch.setattr(go.sys, "executable", sys.executable)

    def _fake_probe(path: str, cuda_devices: str) -> bool:
        return Path(path).resolve() == Path(sys.executable).resolve()

    monkeypatch.setattr(go, "_python_supports_vllm_with_cuda", _fake_probe)

    resolved, use_isolation = go._resolve_vllm_python_interpreter(str(preferred_py.parent.parent), "0,1")
    assert Path(resolved).resolve() == Path(sys.executable).resolve()
    assert use_isolation is True


def test_resolve_vllm_python_disables_isolation_when_only_unmasked_probe_passes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    preferred_py = tmp_path / "vllm-env" / "bin" / "python"
    _touch_executable(preferred_py)
    monkeypatch.delenv("TT_VLLM_PYTHON", raising=False)
    monkeypatch.delenv("VLLM_PYTHON", raising=False)
    monkeypatch.setattr(go.sys, "executable", str(preferred_py))

    def _fake_probe(path: str, cuda_devices):
        return cuda_devices is None

    monkeypatch.setattr(go, "_python_supports_vllm_with_cuda", _fake_probe)

    resolved, use_isolation = go._resolve_vllm_python_interpreter(str(preferred_py.parent.parent), "0,1")
    assert Path(resolved).exists()
    assert use_isolation is False
