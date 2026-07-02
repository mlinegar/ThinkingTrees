"""Project-wide test path setup.

Inserts ``src/`` at the front of ``sys.path`` so ``from treepo.X import Y``
always resolves to the in-tree package, not whatever may be installed in
site-packages. Lets tests run cleanly from a uv-managed checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TREEPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _TREEPO_SRC.is_dir() and str(_TREEPO_SRC) not in sys.path:
    sys.path.insert(0, str(_TREEPO_SRC))


def pytest_ignore_collect(collection_path: Path, config) -> bool:
    """Skip tests that belong to the externalized research archive.

    The publishable v0.1 package no longer bundles ``treepo._research``.
    Tests that import it are still useful in the research workspace, but they
    are not part of this package's default test surface.
    """

    del config
    path = Path(collection_path)
    if path.suffix != ".py" or not path.name.startswith("test_"):
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "from treepo._research" in text or "import treepo._research" in text
