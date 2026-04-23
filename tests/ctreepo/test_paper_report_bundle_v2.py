from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from src.ctreepo.sim.suite.registry import iter_canonical_suite_targets


def _load_bundle_module():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts" / "generate_paper_simulation_report_bundle.py"
    spec = importlib.util.spec_from_file_location("generate_paper_simulation_report_bundle", str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_paper_report_bundle_generator_uses_suite_registry(tmp_path: Path, monkeypatch) -> None:
    module = _load_bundle_module()
    calls: list[list[str]] = []

    def _fake_run(cmd, cwd, capture_output, text):  # noqa: ANN001
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    for rel in ("cpu_megasweep", "identifiable_zero_longrun_clean"):
        root = tmp_path / rel
        root.mkdir(parents=True, exist_ok=True)
        (root / "dummy.json").write_text("{}", encoding="utf-8")

    rc = int(module.main(["--formal-root", str(tmp_path), "--python-bin", sys.executable]))
    assert rc == 0

    manifest = json.loads((tmp_path / "paper_reports" / "paper_report_bundle_manifest.json").read_text(encoding="utf-8"))
    expected = [target.key for target in iter_canonical_suite_targets(bundle_roles=("paper", "appendix", "diagnostic"))]
    assert [row["name"] for row in manifest["results"]] == expected
    assert any(row["bundle_role"] == "appendix" for row in manifest["results"])
    assert all("scripts/report_" not in json.dumps(row) for row in manifest["results"])
    assert calls
    assert any(
        call[:7] == [sys.executable, "-m", "src.ctreepo.cli", "sim", "suite", "cpu-megasweep", "report"]
        for call in calls
    )
    assert any(
        call[:7] == [sys.executable, "-m", "src.ctreepo.cli", "sim", "suite", "identifiable-zero-publication", "report"]
        for call in calls
    )
