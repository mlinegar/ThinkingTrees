#!/usr/bin/env python3
"""General C-TreePO run launcher.

This runner provides one thin surface for paper and non-paper work.  It writes
RunManifest v1 for every planned or executed target; individual target commands
remain owned by their existing scripts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ctreepo.contracts import run_manifest_metadata  # noqa: E402
from src.ctreepo.run_registry import RunTargetRecord, get_run_target, iter_run_targets  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-targets", action="store_true")
    parser.add_argument("--target", action="append", default=[], help="Run target name.")
    parser.add_argument("--suite", action="append", default=[], help="Suite tag to select targets.")
    parser.add_argument("--plan-only", action="store_true", help="Write manifests without executing commands.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ctreepo_runs"))
    parser.add_argument("--allow-legacy", action="store_true")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--json", action="store_true", help="Print target listing as JSON.")
    return parser.parse_args(argv)


def _render_command(record: RunTargetRecord, *, output_root: Path, python_bin: str) -> list[str]:
    rendered: list[str] = []
    for part in record.command:
        rendered.append(
            str(part)
            .replace("{output_root}", str(output_root))
            .replace("{python}", str(python_bin))
        )
    return rendered


def _targets_from_args(args: argparse.Namespace) -> list[RunTargetRecord]:
    records: list[RunTargetRecord] = []
    seen: set[str] = set()
    for suite in args.suite:
        for record in iter_run_targets(suites=(suite,)):
            if record.target not in seen:
                records.append(record)
                seen.add(record.target)
    for target in args.target:
        record = get_run_target(target)
        if record.target not in seen:
            records.append(record)
            seen.add(record.target)
    return sorted(records, key=lambda row: row.target)


def _run_dir_for(record: RunTargetRecord, *, output_root: Path, multi: bool) -> Path:
    if not multi:
        return output_root
    return output_root / record.target.replace(".", "/")


def _build_manifest(
    record: RunTargetRecord,
    *,
    run_dir: Path,
    command: list[str],
    status: str,
    allow_legacy: bool,
    elapsed_seconds: float | None = None,
    returncode: int | None = None,
) -> dict[str, object]:
    metadata = {
        "target": record.target,
        "target_status": record.status,
        "expected_input_contract": record.expected_input_contract,
        "output_contract": record.output_contract,
        "audit_policy": record.audit_policy,
        "target_publication_facing": bool(record.publication_ready),
        "suites": list(record.suites),
        "notes": record.notes,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if elapsed_seconds is not None:
        metadata["elapsed_seconds"] = float(elapsed_seconds)
    if returncode is not None:
        metadata["returncode"] = int(returncode)
    return run_manifest_metadata(
        run_id=record.target,
        domain=record.domain,
        role=record.role,
        backend=record.backend,
        status=status,
        f_init="",
        g_init="",
        command=command,
        allow_legacy=allow_legacy,
        publication_ready=False,
        metadata=metadata,
        output_artifacts=[
            {
                "kind": "run_directory",
                "uri": str(run_dir),
            }
        ],
    )


def _write_manifest(run_dir: Path, payload: dict[str, object]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _execute_one(
    record: RunTargetRecord,
    *,
    run_dir: Path,
    command: list[str],
    allow_legacy: bool,
) -> tuple[int, Path]:
    if record.status in {"legacy_compat", "not_publication_safe"} and not allow_legacy:
        payload = _build_manifest(
            record,
            run_dir=run_dir,
            command=command,
            status="failed",
            allow_legacy=allow_legacy,
            returncode=2,
        )
        path = _write_manifest(run_dir, payload)
        print(f"{record.target}: refused unsafe target without --allow-legacy", file=sys.stderr)
        return 2, path
    if not command:
        payload = _build_manifest(
            record,
            run_dir=run_dir,
            command=command,
            status="failed",
            allow_legacy=allow_legacy,
            returncode=2,
        )
        path = _write_manifest(run_dir, payload)
        print(f"{record.target}: no command registered; use --plan-only", file=sys.stderr)
        return 2, path

    logs = run_dir / "launcher"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / "stdout.log"
    stderr_path = logs / "stderr.log"
    started = time.time()
    proc = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    elapsed = time.time() - started
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    payload = _build_manifest(
        record,
        run_dir=run_dir,
        command=command,
        status="completed" if proc.returncode == 0 else "failed",
        allow_legacy=allow_legacy,
        elapsed_seconds=elapsed,
        returncode=int(proc.returncode),
    )
    payload["output_artifacts"] = [
        {"kind": "run_directory", "uri": str(run_dir)},
        {"kind": "stdout_log", "uri": str(stdout_path)},
        {"kind": "stderr_log", "uri": str(stderr_path)},
    ]
    path = _write_manifest(run_dir, payload)
    return int(proc.returncode), path


def _print_targets(records: Iterable[RunTargetRecord], *, as_json: bool) -> None:
    rows = list(records)
    if as_json:
        print(json.dumps([row.to_dict() for row in rows], indent=2, sort_keys=True))
        return
    for row in rows:
        pub = "publication" if row.publication_ready else "general"
        print(f"{row.target}\t{row.domain}\t{row.backend}\t{row.status}\t{pub}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list_targets:
        _print_targets(iter_run_targets(suites=args.suite), as_json=bool(args.json))
        return 0

    records = _targets_from_args(args)
    if not records:
        print("No targets selected. Use --target, --suite, or --list-targets.", file=sys.stderr)
        return 2

    multi = len(records) > 1
    manifest_paths: list[str] = []
    exit_code = 0
    for record in records:
        run_dir = _run_dir_for(record, output_root=args.output_root, multi=multi)
        command = _render_command(record, output_root=run_dir, python_bin=str(args.python_bin))
        if args.plan_only:
            payload = _build_manifest(
                record,
                run_dir=run_dir,
                command=command,
                status="planned",
                allow_legacy=bool(args.allow_legacy),
            )
            manifest_paths.append(str(_write_manifest(run_dir, payload)))
            continue
        rc, manifest_path = _execute_one(
            record,
            run_dir=run_dir,
            command=command,
            allow_legacy=bool(args.allow_legacy),
        )
        manifest_paths.append(str(manifest_path))
        if rc != 0:
            exit_code = rc

    print(json.dumps({"run_manifests": manifest_paths}, indent=2, sort_keys=True))
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
