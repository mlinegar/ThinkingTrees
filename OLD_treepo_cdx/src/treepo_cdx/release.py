from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PACKAGE_ROOT / "src" / "treepo_cdx"
HEAVY_IMPORT_ROOTS = {"dspy", "openai", "pandas", "torch", "transformers", "vllm"}


def audit_public_imports() -> dict[str, Any]:
    before = set(sys.modules)
    importlib.import_module("treepo_cdx")
    loaded = {name: name in sys.modules and name not in before for name in HEAVY_IMPORT_ROOTS}
    failures = [
        {"reason": "heavy_import_loaded_by_public_import", "module": name}
        for name, is_loaded in sorted(loaded.items())
        if bool(is_loaded)
    ]
    return {"ok": not failures, "loaded": loaded, "failures": failures}


def audit_static_imports() -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    py_files = sorted(SRC_ROOT.rglob("*.py"))
    for path in py_files:
        rel = str(path.relative_to(PACKAGE_ROOT))
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            failures.append({"path": rel, "reason": "syntax_error", "error": str(exc)})
            continue
        for node in ast.walk(tree):
            roots: list[str] = []
            if isinstance(node, ast.Import):
                roots.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.append(node.module.split(".")[0])
            for root in roots:
                if root in HEAVY_IMPORT_ROOTS:
                    failures.append({"path": rel, "reason": "heavy_static_import", "import": root})
    return {"ok": not failures, "checked_files": len(py_files), "failures": failures}


def audit_release() -> dict[str, Any]:
    checks = {
        "public_imports": audit_public_imports(),
        "static_imports": audit_static_imports(),
    }
    failures: list[dict[str, Any]] = []
    for name, report in checks.items():
        failures.extend({"check": name, **item} for item in report.get("failures", []))
    return {"ok": not failures, "checks": checks, "failures": failures}


__all__ = ["audit_public_imports", "audit_release", "audit_static_imports"]
