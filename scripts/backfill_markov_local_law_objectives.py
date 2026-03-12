#!/usr/bin/env python3
"""Replay completed Markov local-law runs that are missing exact held-out objective fields."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REPLAY_SCRIPT = Path(__file__).resolve().with_name("replay_markov_changepoint_ops_count_summary.py")


@dataclass(frozen=True)
class ReplayTarget:
    summary_json: Path
    csv_summary: Path
    age_seconds: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill missing Markov local-law objective fields.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--require-field",
        type=str,
        default="test_objective_full_labels",
        help="Field required inside metrics.learned for a summary to count as already backfilled.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1)),
        help="Number of subprocess workers to run in parallel.",
    )
    parser.add_argument(
        "--device",
        choices=["inherit", "cpu", "cuda"],
        default="cpu",
        help="Replay device override.",
    )
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch intra/inter-op thread override passed to the replay script.",
    )
    parser.add_argument(
        "--min-age-seconds",
        type=float,
        default=0.0,
        help="Only replay summaries older than this age to avoid racing live writers.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional cap on the number of summaries to replay (0 means all eligible).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for per-run logs. Defaults to <input-root>/backfill_logs.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional JSON manifest output path.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="Polling interval for queue progress.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _iter_summary_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("seed_*.json")
        if "local_law_report" not in str(path)
    )


def _is_missing_required_field(summary_path: Path, field_name: str) -> bool:
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    learned = ((data.get("metrics") or {}).get("learned") or {})
    return field_name not in learned


def _eligible_targets(root: Path, *, field_name: str, min_age_seconds: float) -> list[ReplayTarget]:
    now = time.time()
    targets: list[ReplayTarget] = []
    for summary_path in _iter_summary_paths(root):
        age_seconds = float(now - summary_path.stat().st_mtime)
        if age_seconds < float(min_age_seconds):
            continue
        if not _is_missing_required_field(summary_path, field_name):
            continue
        targets.append(
            ReplayTarget(
                summary_json=summary_path,
                csv_summary=summary_path.with_suffix(".csv"),
                age_seconds=age_seconds,
            )
        )
    targets.sort(key=lambda item: (-item.age_seconds, str(item.summary_json)))
    return targets


def _run_one(
    idx: int,
    target: ReplayTarget,
    *,
    log_dir: Path,
    device: str,
    cuda_device: int | None,
    torch_threads: int | None,
) -> dict[str, Any]:
    log_path = log_dir / f"run_{idx:04d}.log"
    cmd = [
        sys.executable,
        str(REPLAY_SCRIPT),
        "--summary-json",
        str(target.summary_json),
        "--csv-summary",
        str(target.csv_summary),
        "--device",
        str(device),
    ]
    if cuda_device is not None:
        cmd.extend(["--cuda-device", str(int(cuda_device))])
    if torch_threads is not None:
        cmd.extend(["--torch-threads", str(int(torch_threads))])
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"summary_json={target.summary_json}\n")
        handle.write(f"csv_summary={target.csv_summary}\n")
        handle.write(f"age_seconds={target.age_seconds:.1f}\n")
        handle.write(f"cmd={' '.join(cmd)}\n")
        handle.flush()
        proc = subprocess.run(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "idx": int(idx),
        "ok": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "seconds": round(time.time() - start, 1),
        "summary_json": str(target.summary_json),
        "log": str(log_path),
    }


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _target_to_json(target: ReplayTarget) -> dict[str, Any]:
    return {
        "summary_json": str(target.summary_json),
        "csv_summary": str(target.csv_summary),
        "age_seconds": float(target.age_seconds),
    }


def main() -> int:
    args = _parse_args()
    input_root = Path(args.input_root)
    if not input_root.exists():
        raise SystemExit(f"input root does not exist: {input_root}")
    log_dir = Path(args.log_dir) if args.log_dir is not None else (input_root / "backfill_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    targets = _eligible_targets(
        input_root,
        field_name=str(args.require_field),
        min_age_seconds=float(args.min_age_seconds),
    )
    if int(args.max_runs) > 0:
        targets = targets[: int(args.max_runs)]

    manifest_path = (
        Path(args.manifest_path)
        if args.manifest_path is not None
        else (log_dir / "backfill_manifest.json")
    )
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "input_root": str(input_root),
        "require_field": str(args.require_field),
        "device": str(args.device),
        "torch_threads": int(args.torch_threads) if args.torch_threads is not None else None,
        "min_age_seconds": float(args.min_age_seconds),
        "workers": int(args.workers),
        "target_count": int(len(targets)),
        "completed": 0,
        "failed": 0,
        "targets": [_target_to_json(t) for t in targets],
        "results": [],
        "dry_run": bool(args.dry_run),
    }
    _write_manifest(manifest_path, manifest)
    if bool(args.dry_run):
        print(json.dumps({"target_count": len(targets), "manifest": str(manifest_path)}, indent=2, sort_keys=True))
        return 0
    if len(targets) == 0:
        print(json.dumps({"target_count": 0, "manifest": str(manifest_path)}, indent=2, sort_keys=True))
        return 0

    inflight: dict[Future, int] = {}
    next_idx = 0
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        while next_idx < len(targets) or inflight:
            while next_idx < len(targets) and len(inflight) < max(1, int(args.workers)):
                future = pool.submit(
                    _run_one,
                    next_idx,
                    targets[next_idx],
                    log_dir=log_dir,
                    device=str(args.device),
                    cuda_device=(int(args.cuda_device) if args.cuda_device is not None else None),
                    torch_threads=(int(args.torch_threads) if args.torch_threads is not None else None),
                )
                inflight[future] = next_idx
                next_idx += 1

            if not inflight:
                break

            done, _ = wait(inflight.keys(), timeout=float(args.poll_seconds), return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                inflight.pop(future, None)
                result = future.result()
                results.append(result)
                manifest["completed"] = int(len(results))
                manifest["failed"] = int(sum(1 for row in results if not bool(row["ok"])))
                manifest["results"] = results
                _write_manifest(manifest_path, manifest)
                status = "ok" if bool(result["ok"]) else "fail"
                print(
                    f"[{int(result['idx']) + 1}/{len(targets)}] {status} {result['seconds']}s "
                    f"summary={result['summary_json']} log={result['log']}",
                    flush=True,
                )

    failures = [row for row in results if not bool(row["ok"])]
    manifest["finished_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    manifest["failed"] = int(len(failures))
    manifest["results"] = results
    _write_manifest(manifest_path, manifest)
    if failures:
        print(json.dumps({"failed": len(failures), "manifest": str(manifest_path)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps({"completed": len(results), "manifest": str(manifest_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
