from __future__ import annotations

import importlib
import os
from pathlib import Path

import src
import src.config.dspy_config as dspy_config


def test_src_sets_writable_default_dspy_cache(monkeypatch) -> None:
    monkeypatch.delenv("DSPY_CACHEDIR", raising=False)

    importlib.reload(src)

    cache_dir = str(os.getenv("DSPY_CACHEDIR", "") or "").strip()
    assert cache_dir
    cache_path = Path(cache_dir)
    assert cache_path.exists()
    assert cache_path.is_dir()

    probe = cache_path / ".pytest_write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)


def test_src_does_not_override_explicit_dspy_cache(monkeypatch, tmp_path: Path) -> None:
    explicit = tmp_path / "explicit_cache_dir"
    monkeypatch.setenv("DSPY_CACHEDIR", str(explicit))

    importlib.reload(src)

    assert os.environ.get("DSPY_CACHEDIR") == str(explicit)


class _DummyDiskCache:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _DummyDSPYCache:
    def __init__(self) -> None:
        self.disk_cache = _DummyDiskCache()
        self.memory_reset = False

    def reset_memory_cache(self) -> None:
        self.memory_reset = True


def test_configure_dspy_cache_from_env_reconfigures_and_closes_old_cache(monkeypatch, tmp_path: Path) -> None:
    old_cache = _DummyDSPYCache()
    new_cache = _DummyDSPYCache()
    recorded: dict[str, object] = {}

    def _fake_configure_cache(**kwargs) -> None:
        recorded.update(kwargs)
        dspy_config.dspy.cache = new_cache

    monkeypatch.setattr(dspy_config.dspy, "cache", old_cache, raising=False)
    monkeypatch.setattr(dspy_config.dspy, "configure_cache", _fake_configure_cache)
    monkeypatch.setenv("TT_DSPY_ENABLE_DISK_CACHE", "0")
    monkeypatch.setenv("TT_DSPY_ENABLE_MEMORY_CACHE", "0")
    monkeypatch.setenv("DSPY_CACHEDIR", str(tmp_path / "dspy_cache"))

    dspy_config._dspy_cache_runtime_signature = None
    enable_disk, enable_memory = dspy_config.configure_dspy_cache_from_env(force=True)

    assert enable_disk is False
    assert enable_memory is False
    assert recorded["enable_disk_cache"] is False
    assert recorded["enable_memory_cache"] is False
    assert str(recorded["disk_cache_dir"]) == str(tmp_path / "dspy_cache")
    assert old_cache.disk_cache.closed is True


def test_close_dspy_cache_closes_current_cache_and_resets_signature(monkeypatch) -> None:
    cache = _DummyDSPYCache()
    monkeypatch.setattr(dspy_config.dspy, "cache", cache, raising=False)
    dspy_config._dspy_cache_runtime_signature = (True, True, "/tmp/x", 1, 1)

    dspy_config.close_dspy_cache(reset_memory_cache=True)

    assert cache.disk_cache.closed is True
    assert cache.memory_reset is True
    assert dspy_config._dspy_cache_runtime_signature is None
