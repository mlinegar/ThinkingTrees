from __future__ import annotations

import importlib
import os
from pathlib import Path

import src


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
