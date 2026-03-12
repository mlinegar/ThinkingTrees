#!/usr/bin/env python3
"""Monitor the tree-relevant LDA production queue and emit timed report checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor a tree-relevant LDA production queue.")
    p.add_argument("--output-root", type=str, required=True)
    p.add_argument("--poll-seconds", type=int, default=900, help="Polling interval. Default: 900s (15 min).")
    p.add_argument(
        "--report-after-seconds",
        type=int,
        default=7200,
        help="Emit the first timed checkpoint report after this many seconds. Default: 7200s (2 hours).",
    )
    p.add_argument(
        "--report-every-seconds",
        type=int,
        default=7200,
        help="Emit subsequent timed checkpoint reports at this cadence. Default: 7200s.",
    )
    return p.parse_args()


def _parse_spec(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _count_outputs(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.json"))


def _pid_running(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _status_payload(output_root: Path, spec: Dict[str, str]) -> Dict[str, object]:
    stage1_root = output_root / "stage1"
    stage2_root = output_root / "stage2"
    stage1_results = stage1_root / "results"
    stage2_results = stage2_root / "results"
    stage1_done = _count_outputs(stage1_results)
    stage2_done = _count_outputs(stage2_results)
    stage1_total = int(spec.get("stage1_total_commands", "0") or 0)
    stage2_total = int(spec.get("stage2_total_commands", "0") or 0)
    return {
        "timestamp_utc": _utc_now().isoformat(),
        "output_root": str(output_root),
        "stage1_done": int(stage1_done),
        "stage1_total": int(stage1_total),
        "stage2_done": int(stage2_done),
        "stage2_total": int(stage2_total),
        "stage1_running": _pid_running(stage1_root / "sweep.pid"),
        "stage2_running": _pid_running(stage2_root / "sweep.pid"),
    }


def _write_markdown(path: Path, payload: Dict[str, object]) -> None:
    lines = [
        "# Tree-Relevant LDA Queue Monitor",
        "",
        f"- Timestamp UTC: `{payload['timestamp_utc']}`",
        f"- Output root: `{payload['output_root']}`",
        f"- Stage 1: `{payload['stage1_done']}/{payload['stage1_total']}` complete",
        f"- Stage 2: `{payload['stage2_done']}/{payload['stage2_total']}` complete",
        f"- Stage 1 running: `{payload['stage1_running']}`",
        f"- Stage 2 running: `{payload['stage2_running']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _emit_report(output_root: Path, *, suffix: str) -> None:
    report_dir = output_root / "monitor" / suffix
    report_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(REPO_ROOT / "venv/bin/python"),
        str(REPO_ROOT / "scripts/report_lda_tree_methods_paper.py"),
        "--stage1-root",
        str(output_root / "stage1" / "results"),
        "--stage2-root",
        str(output_root / "stage2" / "results"),
        "--output-dir",
        str(report_dir),
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    monitor_root = output_root / "monitor"
    monitor_root.mkdir(parents=True, exist_ok=True)
    spec = _parse_spec(output_root / "sweep_spec.txt")

    status_json = monitor_root / "latest_status.json"
    status_md = monitor_root / "latest_status.md"
    history_jsonl = monitor_root / "history.jsonl"
    start_time = _utc_now()
    next_report_time = start_time + timedelta(seconds=int(args.report_after_seconds))
    report_every = timedelta(seconds=int(args.report_every_seconds))

    while True:
        payload = _status_payload(output_root, spec)
        status_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_markdown(status_md, payload)
        with history_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

        now = _utc_now()
        if now >= next_report_time:
            suffix = f"checkpoint_{now.strftime('%Y%m%d_%H%M%S')}"
            _emit_report(output_root, suffix=suffix)
            next_report_time = now + report_every

        stage1_done = int(payload["stage1_done"])
        stage1_total = int(payload["stage1_total"])
        stage2_done = int(payload["stage2_done"])
        stage2_total = int(payload["stage2_total"])
        if (
            not bool(payload["stage1_running"])
            and not bool(payload["stage2_running"])
            and stage1_done >= stage1_total
            and stage2_done >= stage2_total
        ):
            _emit_report(output_root, suffix="final")
            break

        time.sleep(max(30, int(args.poll_seconds)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
