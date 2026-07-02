from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_hygiene_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "check_repo_release_hygiene.py"
    spec = importlib.util.spec_from_file_location("check_repo_release_hygiene", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_release_hygiene_flags_tracked_generated_artifacts() -> None:
    hygiene = _load_hygiene_module()

    failures = hygiene._tracked_generated(
        [
            "src/experiments/context.py",
            "outputs/example/metrics.json",
            "logs/server.out",
            "src/runtime/__pycache__/calls.cpython-311.pyc",
            "models/state_model.pt",
            "paper/ctreepo/main.aux",
        ]
    )

    assert failures == [
        "logs/server.out",
        "models/state_model.pt",
        "outputs/example/metrics.json",
        "paper/ctreepo/main.aux",
        "src/runtime/__pycache__/calls.cpython-311.pyc",
    ]


def test_release_hygiene_scans_public_text_markers(tmp_path: Path, monkeypatch) -> None:
    hygiene = _load_hygiene_module()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "public.md").write_text("local path: /home/mlinegar/project\n", encoding="utf-8")

    monkeypatch.setattr(hygiene, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(hygiene, "PUBLIC_SCAN_PATHS", ("docs",))

    failures = hygiene._scan_text(("/home/mlinegar",))

    assert failures == [{"path": "docs/public.md", "line": 1, "marker": "/home/mlinegar"}]
