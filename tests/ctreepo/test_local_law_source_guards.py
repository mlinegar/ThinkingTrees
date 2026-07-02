from __future__ import annotations

from pathlib import Path


def test_thinkingtrees_local_law_shims_delegate_to_treepo() -> None:
    shim_paths = (
        Path("src/core/local_law_adjustment.py"),
        Path("src/training/supervision/local_law_torch.py"),
    )
    banned_fragments = (
        "proxy +",
        "(oracle - proxy) /",
        "torch.pow",
        "gamma_depth **",
        "gamma**",
    )

    for path in shim_paths:
        source = path.read_text(encoding="utf-8")
        assert "treepo.training.local_law" in source
        for fragment in banned_fragments:
            assert fragment not in source


def test_new_row_adapter_does_not_import_archived_paths() -> None:
    source = Path("src/ctreepo/local_law_rows.py").read_text(encoding="utf-8")

    assert "treepo_cdx" not in source
    assert "OLD_" not in source
    assert "._research" not in source
    assert "treepo.training.local_law" in source


def test_live_source_uses_treepo_local_law_not_thinkingtrees_shims() -> None:
    banned_fragments = (
        "from src.core.local_law_adjustment import",
        "import src.core.local_law_adjustment",
        "from src.training.supervision.local_law_torch import",
        "import src.training.supervision.local_law_torch",
    )
    allowed_paths = {
        Path("src/core/__init__.py"),
        Path("src/core/local_law_adjustment.py"),
        Path("src/training/supervision/local_law_torch.py"),
    }

    offenders: list[str] = []
    for root in (Path("src"), Path("scripts")):
        for path in root.rglob("*.py"):
            if path in allowed_paths:
                continue
            source = path.read_text(encoding="utf-8")
            for fragment in banned_fragments:
                if fragment in source:
                    offenders.append(f"{path}: {fragment}")
    assert offenders == []
