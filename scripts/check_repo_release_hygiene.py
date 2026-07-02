#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

GENERATED_PREFIXES = (
    "outputs/",
    "logs/",
    ".cache/",
    ".pytest_cache/",
    "tmp_search_debug/",
)
GENERATED_PARTS = (
    "/__pycache__/",
    "/hf_runtime_cache/",
)
GENERATED_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".pt",
    ".pth",
    ".safetensors",
    ".bin",
    ".ckpt",
    ".checkpoint",
    ".onnx",
    ".npy",
    ".npz",
    ".aux",
    ".fls",
    ".fdb_latexmk",
    ".out",
    ".toc",
    ".bbl",
    ".blg",
    ".synctex.gz",
)
PUBLIC_SCAN_PATHS = (
    "README.md",
    "docs/experiment_method_api.md",
    "docs/runtime_v1_launch.md",
    "docs/runtime_v1_paper_appendix.md",
    "docs/runtime_backbone_handoff_spec.md",
    "config/runtime_eval",
    "config/perf/perf_matrix.yaml",
    "config/runtime_umbrella_entrypoints.yaml",
)
LOCAL_PATH_MARKERS = (
    "/home/mlinegar",
    "/mnt/data",
)
STALE_RUNTIME_FLAGS = (
    "--run-id",
    "--run-dir",
)


def _git_ls_files() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _tracked_generated(paths: Iterable[str]) -> list[str]:
    failures: list[str] = []
    for path in paths:
        normalized = path.replace("\\", "/")
        if normalized.startswith(GENERATED_PREFIXES):
            failures.append(path)
            continue
        with_slashes = f"/{normalized}"
        if any(part in with_slashes for part in GENERATED_PARTS):
            failures.append(path)
            continue
        if normalized.endswith(GENERATED_SUFFIXES):
            failures.append(path)
    return sorted(set(failures))


def _iter_public_files() -> Iterable[Path]:
    for raw in PUBLIC_SCAN_PATHS:
        path = REPO_ROOT / raw
        if path.is_dir():
            yield from sorted(
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in {".md", ".yaml", ".yml"}
            )
        elif path.exists():
            yield path


def _scan_text(markers: Sequence[str]) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for path in _iter_public_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = str(path.relative_to(REPO_ROOT))
        for lineno, line in enumerate(text.splitlines(), start=1):
            for marker in markers:
                if marker in line:
                    failures.append({"path": rel, "line": lineno, "marker": marker})
    return failures


def build_report() -> dict[str, object]:
    tracked = _git_ls_files()
    generated = _tracked_generated(tracked)
    local_paths = _scan_text(LOCAL_PATH_MARKERS)
    stale_flags = _scan_text(STALE_RUNTIME_FLAGS)
    failures = {
        "tracked_generated": generated,
        "local_absolute_paths": local_paths,
        "stale_runtime_flags": stale_flags,
    }
    return {
        "ok": not any(failures.values()),
        "failures": failures,
        "checked_public_paths": list(PUBLIC_SCAN_PATHS),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check public-repo release hygiene.")
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    args = parser.parse_args(argv)
    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("release hygiene: " + ("ok" if report["ok"] else "FAILED"))
        for group, items in dict(report["failures"]).items():
            if items:
                print(f"- {group}: {len(items)}")
    return 0 if bool(report["ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
