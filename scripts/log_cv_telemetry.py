#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


NEUTRAL_FALLBACK_PATTERN = re.compile(
    r"RILEScorer prediction failed; defaulting to neutral"
)
LM_TIMEOUT_PATTERN = re.compile(r"LM timeout")
LM_INTERNAL_PATTERN = re.compile(r"InternalServerError|Connection error")
FOLD_PATTERN = re.compile(r"/folds/(fold_\d+)\b")
BATCH_STATS_PATTERN = re.compile(
    r"BatchStats\(reqs=(\d+)/(\d+),\s*tokens=([\d,]+),\s*tok/s=([\d.]+)\s*\[r:([\d.]+),\s*w:([\d.]+)\]\)"
)


def _run_command(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.stdout or ""


def _tail_text(path: Path, max_bytes: int = 800_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        start = max(0, size - max_bytes)
        f.seek(start)
        chunk = f.read().decode("utf-8", errors="ignore")
    return chunk.replace("\r", "\n")


def _parse_gpu_snapshot() -> List[Dict[str, Any]]:
    text = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,pstate",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        try:
            rows.append(
                {
                    "gpu": int(parts[0]),
                    "utilization_gpu_pct": float(parts[1]),
                    "utilization_mem_pct": float(parts[2]),
                    "memory_used_mib": float(parts[3]),
                    "memory_total_mib": float(parts[4]),
                    "power_w": float(parts[5]),
                    "temp_c": float(parts[6]),
                    "pstate": parts[7],
                }
            )
        except ValueError:
            continue
    return rows


def _find_active_fold(cv_output_dir: Path) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    pgrep_out = _run_command(["pgrep", "-af", "src.training.run_pipeline"])
    cv_hint = str(cv_output_dir)
    cv_name = cv_output_dir.name
    for line in pgrep_out.splitlines():
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1]
        if "/folds/fold_" not in cmd:
            continue
        if cv_hint not in cmd and cv_name not in cmd:
            continue
        fold_match = FOLD_PATTERN.search(cmd)
        if not fold_match:
            continue
        return fold_match.group(1), pid, cmd
    return None, None, None


def _parse_batch_stats_from_text(text: str) -> Optional[Dict[str, Any]]:
    matches = list(BATCH_STATS_PATTERN.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return {
        "requests_done": int(match.group(1)),
        "requests_total": int(match.group(2)),
        "tokens_total": int(match.group(3).replace(",", "")),
        "tokens_per_sec": float(match.group(4)),
        "read_tokens_per_sec": float(match.group(5)),
        "write_tokens_per_sec": float(match.group(6)),
    }


def _count_recent_warnings(text: str) -> Dict[str, int]:
    return {
        "neutral_fallbacks": len(NEUTRAL_FALLBACK_PATTERN.findall(text)),
        "lm_timeouts": len(LM_TIMEOUT_PATTERN.findall(text)),
        "lm_internal_errors": len(LM_INTERNAL_PATTERN.findall(text)),
    }


def _last_nonempty_line(text: str) -> Optional[str]:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _completed_fold_count(cv_output_dir: Path) -> int:
    return len(list((cv_output_dir / "folds").glob("fold_*/final_stats.json")))


def _snapshot(cv_output_dir: Path, active_log_tail_bytes: int = 800_000) -> Dict[str, Any]:
    active_fold, active_pid, active_cmd = _find_active_fold(cv_output_dir)
    active_log_path: Optional[Path] = None
    active_tail = ""
    if active_fold is not None:
        candidate = cv_output_dir / "folds" / active_fold / "cv_run.log"
        if candidate.exists():
            active_log_path = candidate
            active_tail = _tail_text(candidate, max_bytes=active_log_tail_bytes)

    gpu = _parse_gpu_snapshot()
    gpu_utils = [row["utilization_gpu_pct"] for row in gpu]

    return {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "cv_output_dir": str(cv_output_dir),
        "active_fold": active_fold,
        "active_pid": active_pid,
        "completed_folds": _completed_fold_count(cv_output_dir),
        "gpu": gpu,
        "gpu_util_mean_pct": (sum(gpu_utils) / len(gpu_utils)) if gpu_utils else None,
        "gpu_util_max_pct": max(gpu_utils) if gpu_utils else None,
        "active_log_path": str(active_log_path) if active_log_path else None,
        "active_log_last_line": _last_nonempty_line(active_tail) if active_tail else None,
        "active_log_recent_warnings": _count_recent_warnings(active_tail) if active_tail else None,
        "batch_stats_latest": _parse_batch_stats_from_text(active_tail) if active_tail else None,
        "active_command": active_cmd,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Periodic CV telemetry logger (JSONL)."
    )
    parser.add_argument(
        "--cv-output-dir",
        type=Path,
        required=True,
        help="CV output directory (for example: outputs/manifesto_cv_20260225_0304).",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=None,
        help="JSONL output path (default: <cv-output-dir>/telemetry/cv_telemetry.jsonl).",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=30.0,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--tail-bytes",
        type=int,
        default=800_000,
        help="How many bytes to tail from active fold log each sample.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Write a single snapshot and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cv_output_dir = args.cv_output_dir.resolve()
    out_path = (
        args.out_jsonl.resolve()
        if args.out_jsonl is not None
        else (cv_output_dir / "telemetry" / "cv_telemetry.jsonl")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def write_snapshot() -> None:
        rec = _snapshot(cv_output_dir=cv_output_dir, active_log_tail_bytes=int(args.tail_bytes))
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.once:
        write_snapshot()
        return 0

    try:
        while True:
            try:
                write_snapshot()
            except Exception as exc:
                err = {
                    "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                    "cv_output_dir": str(cv_output_dir),
                    "error": str(exc),
                }
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(err, ensure_ascii=False) + "\n")
            time.sleep(max(1.0, float(args.interval_sec)))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
