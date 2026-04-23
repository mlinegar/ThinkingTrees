#!/usr/bin/env python3
"""Run the broad classical-sketch paper bundle and stage paper assets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    except Exception:
        return None
    return out.decode("utf-8", errors="replace").strip() or None


def _run(cmd: list[str], *, env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def _copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _count_rows(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    learned = sum(1 for row in rows if row.get("implementation_status") == "learned_empirical")
    return len(rows), learned


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def _stage_assets(report_dir: Path) -> dict[str, Any]:
    assets_root = REPO_ROOT / "paper" / "ctreepo" / "assets" / "sketches"
    figures_dir = assets_root / "figures"
    tables_dir = assets_root / "tables"
    staged: dict[str, Any] = {"figures": [], "tables": []}
    for src in sorted(report_dir.glob("classical_sketches_*.png")):
        dst = figures_dir / src.name
        copied = _copy_if_exists(src, dst)
        if copied:
            staged["figures"].append(copied)
    for name in (
        "classical_sketches_grid.md",
        "classical_sketches_grid.tex",
        "classical_sketches_compact.md",
        "classical_sketches_compact.tex",
        "classical_sketches_aggregate.csv",
        "classical_sketches_aggregate.json",
        "classical_sketches_report.md",
    ):
        copied = _copy_if_exists(report_dir / name, tables_dir / name)
        if copied:
            staged["tables"].append(copied)
    return staged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--capacities", type=str, default="small,medium,large")
    parser.add_argument("--leaf-counts", type=str, default="1,2,4,8,16")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-learned", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learned-targets", type=str, default="all")
    parser.add_argument("--learned-variants", type=str, default="f,g,fg,gf")
    parser.add_argument("--learned-epochs", type=int, default=150)
    parser.add_argument("--learned-n-train", type=int, default=128)
    parser.add_argument("--learned-n-val", type=int, default=48)
    args = parser.parse_args(argv)

    out_root = Path(args.out_root) if args.out_root is not None else REPO_ROOT / "outputs" / f"classical_sketches_paper_all_cpu_{_utc_stamp()}"
    out_root = out_root.resolve()
    env = dict(os.environ)
    py_path = [str(REPO_ROOT / "treepo" / "src"), str(REPO_ROOT / "parallel" / "unified_g_v1" / "src"), str(REPO_ROOT)]
    env["PYTHONPATH"] = os.pathsep.join(py_path + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))

    suite_cmd = [
        sys.executable,
        "-m",
        "treepo.bench.cli",
        "suite",
        "classical-sketches",
        "--out-root",
        str(out_root),
        "--jobs",
        str(int(args.jobs)),
        "--seeds",
        str(args.seeds),
        "--capacities",
        str(args.capacities),
        "--leaf-counts",
        str(args.leaf_counts),
    ]
    if bool(args.skip_existing):
        suite_cmd.append("--skip-existing")
    if bool(args.include_learned):
        suite_cmd.extend(
            [
                "--include-learned",
                "--learned-targets",
                str(args.learned_targets),
                "--learned-variants",
                str(args.learned_variants),
                "--learned-epochs",
                str(int(args.learned_epochs)),
                "--learned-n-train",
                str(int(args.learned_n_train)),
                "--learned-n-val",
                str(int(args.learned_n_val)),
            ]
        )
    else:
        suite_cmd.append("--no-include-learned")
    _run(suite_cmd, env=env)

    report_cmd = [
        sys.executable,
        "-m",
        "treepo.bench.cli",
        "report",
        "classical-sketches",
        "--output-root",
        str(out_root),
        "--tables-dir",
        str(REPO_ROOT / "paper" / "ctreepo" / "tables"),
    ]
    _run(report_cmd, env=env)

    report_dir = out_root / "reports" / "classical_sketches"
    staged = _stage_assets(report_dir)
    aggregate_csv = report_dir / "classical_sketches_aggregate.csv"
    aggregate_rows, learned_rows = _count_rows(aggregate_csv)
    manifest = {
        "out_root": str(out_root),
        "report_dir": str(report_dir),
        "aggregate_csv": str(aggregate_csv),
        "aggregate_rows": int(aggregate_rows),
        "learned_rows": int(learned_rows),
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": _jsonable_args(args),
        "staged_assets": staged,
    }
    manifest_path = out_root / "paper_bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
