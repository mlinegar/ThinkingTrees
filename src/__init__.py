"""ThinkingTrees: Oracle-Preserving Summarization."""

from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"


def _is_writable_dir(path: Path) -> bool:
    """Return True when `path` can be created/written by the current process."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".tt_dspy_cache_probe"
        with open(probe, "w", encoding="utf-8"):
            pass
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _configure_default_dspy_cache_dir() -> None:
    """Set a writable default for DSPy disk cache when not explicitly configured."""
    existing = str(os.getenv("DSPY_CACHEDIR", "") or "").strip()
    if existing:
        return

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "outputs" / "dspy_cache",
        Path("/tmp/thinkingtrees_dspy_cache"),
    ]
    for candidate in candidates:
        if _is_writable_dir(candidate):
            os.environ["DSPY_CACHEDIR"] = str(candidate)
            return


_configure_default_dspy_cache_dir()
