#!/usr/bin/env python3
"""Probe vLLM /metrics endpoints for prefix-cache metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ports(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def _to_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fetch_metrics(port: int, timeout_seconds: float) -> Dict[str, Any]:
    url = f"http://127.0.0.1:{int(port)}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8", errors="replace")
            status = int(getattr(response, "status", 200) or 200)
    except urllib.error.HTTPError as exc:
        return {
            "port": int(port),
            "ok": False,
            "status_code": int(exc.code),
            "error": f"HTTPError: {exc}",
            "url": url,
        }
    except Exception as exc:
        return {
            "port": int(port),
            "ok": False,
            "status_code": None,
            "error": f"{type(exc).__name__}: {exc}",
            "url": url,
        }

    metric_names: List[str] = []
    prefix_lines: List[str] = []
    hit_rate_values: Dict[str, float] = {}

    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split(" ", 1)[0].strip()
        metric_names.append(name)
        if "prefix" not in name.lower() and "prefix" not in line.lower():
            continue
        prefix_lines.append(line)
        if "hit" in name.lower() and "rate" in name.lower():
            parts = line.split()
            if len(parts) >= 2:
                val = _to_float(parts[-1])
                if val is not None:
                    hit_rate_values[name] = val

    prefix_metric_names = sorted({ln.split(" ", 1)[0].strip() for ln in prefix_lines})

    return {
        "port": int(port),
        "ok": True,
        "status_code": status,
        "url": url,
        "metrics_total": len(metric_names),
        "prefix_lines_count": len(prefix_lines),
        "prefix_metric_names": prefix_metric_names,
        "hit_rate_values": hit_rate_values,
    }


def _start_servers(
    *,
    start_server: bool,
    start_genrm: bool,
    timeout_seconds: float,
    startup_log: Path,
) -> Dict[str, Any]:
    if not start_server and not start_genrm:
        return {
            "invoked": False,
            "returncode": 0,
            "command": None,
            "log_path": str(startup_log),
        }

    cmd = ["./scripts/start_dual_servers.sh"]
    if start_server and not start_genrm:
        cmd.append("--small-only")
    if start_genrm and not start_server:
        cmd.append("--large-only")

    startup_log.parent.mkdir(parents=True, exist_ok=True)
    with startup_log.open("w", encoding="utf-8") as handle:
        handle.write("Command:\n")
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds if timeout_seconds > 0 else None,
            check=False,
            text=True,
        )

    return {
        "invoked": True,
        "returncode": int(proc.returncode),
        "command": cmd,
        "log_path": str(startup_log),
    }


def _wait_for_metrics(
    *,
    ports: List[int],
    timeout_seconds: float,
    probe_timeout_seconds: float,
) -> Dict[int, bool]:
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    ready = {int(port): False for port in ports}
    while time.monotonic() <= deadline:
        all_ready = True
        for port in ports:
            if ready[int(port)]:
                continue
            result = _fetch_metrics(int(port), timeout_seconds=probe_timeout_seconds)
            is_ready = bool(result.get("ok"))
            ready[int(port)] = is_ready
            all_ready = all_ready and is_ready
        if all_ready:
            return ready
        time.sleep(1.0)
    return ready


def _stop_servers(stop_log: Path, timeout_seconds: float) -> Dict[str, Any]:
    cmd = ["./scripts/stop_small_servers.sh", "--all"]
    stop_log.parent.mkdir(parents=True, exist_ok=True)
    with stop_log.open("w", encoding="utf-8") as handle:
        handle.write("Command:\n")
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds if timeout_seconds > 0 else None,
            check=False,
            text=True,
        )
    return {
        "invoked": True,
        "returncode": int(proc.returncode),
        "command": cmd,
        "log_path": str(stop_log),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe vLLM metrics for prefix-cache metric visibility.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ports",
        type=str,
        default="8000,8001",
        help="Comma-separated metrics ports to probe.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        required=True,
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--probe-timeout-seconds",
        type=float,
        default=5.0,
        help="HTTP timeout per metrics fetch.",
    )
    parser.add_argument(
        "--start-timeout-seconds",
        type=float,
        default=900.0,
        help="Timeout for optional server startup command.",
    )
    parser.add_argument(
        "--wait-after-start-seconds",
        type=float,
        default=120.0,
        help="Max time to wait for /metrics after startup.",
    )
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Start task-model server before probing.",
    )
    parser.add_argument(
        "--start-genrm",
        action="store_true",
        help="Start GenRM server before probing.",
    )
    parser.add_argument(
        "--stop-servers-after-probe",
        action="store_true",
        help="Stop servers after probe via stop_small_servers.sh --all.",
    )
    parser.add_argument(
        "--require-prefix-metrics",
        action="store_true",
        help="Fail if no prefix-related metric lines are visible on any port.",
    )
    parser.add_argument(
        "--require-hit-rate-metric",
        action="store_true",
        help="Fail if no prefix hit-rate metric is detected on any port.",
    )
    args = parser.parse_args()

    started_at = _utc_now_iso()
    t0 = time.perf_counter()

    out_path = args.json_out
    if not out_path.is_absolute():
        out_path = (PROJECT_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ports = _parse_ports(args.ports)
    start_meta = _start_servers(
        start_server=bool(args.start_server),
        start_genrm=bool(args.start_genrm),
        timeout_seconds=float(args.start_timeout_seconds),
        startup_log=out_path.parent / "startup.log",
    )

    wait_ready: Dict[int, bool] = {}
    if bool(start_meta.get("invoked")) and int(start_meta.get("returncode", 1)) == 0:
        wait_ready = _wait_for_metrics(
            ports=ports,
            timeout_seconds=float(args.wait_after_start_seconds),
            probe_timeout_seconds=float(args.probe_timeout_seconds),
        )

    port_results: Dict[str, Any] = {}
    all_prefix_metric_names: List[str] = []
    ports_with_prefix = 0
    total_prefix_lines = 0
    hit_rate_metric_detected = False
    for port in ports:
        result = _fetch_metrics(int(port), timeout_seconds=float(args.probe_timeout_seconds))
        port_results[str(port)] = result
        if bool(result.get("ok")):
            prefix_names = result.get("prefix_metric_names", [])
            if isinstance(prefix_names, list):
                all_prefix_metric_names.extend(str(v) for v in prefix_names)
            total_prefix_lines += int(result.get("prefix_lines_count", 0) or 0)
            if int(result.get("prefix_lines_count", 0) or 0) > 0:
                ports_with_prefix += 1
            hit_vals = result.get("hit_rate_values", {})
            if isinstance(hit_vals, dict) and hit_vals:
                hit_rate_metric_detected = True

    stop_meta: Dict[str, Any] = {"invoked": False, "returncode": 0}
    if bool(args.stop_servers_after_probe):
        stop_meta = _stop_servers(
            stop_log=out_path.parent / "shutdown.log",
            timeout_seconds=float(args.start_timeout_seconds),
        )

    payload: Dict[str, Any] = {
        "created_at": started_at,
        "completed_at": _utc_now_iso(),
        "duration_seconds": float(time.perf_counter() - t0),
        "ports": port_results,
        "startup": start_meta,
        "wait_ready": {str(k): bool(v) for k, v in wait_ready.items()},
        "shutdown": stop_meta,
        "summary": {
            "ports_checked": len(ports),
            "ports_ok": sum(1 for result in port_results.values() if bool(result.get("ok"))),
            "ports_with_prefix_metrics": int(ports_with_prefix),
            "total_prefix_lines": int(total_prefix_lines),
            "all_prefix_metric_names": sorted(set(all_prefix_metric_names)),
            "hit_rate_metric_detected": bool(hit_rate_metric_detected),
            "hit_rate_metric_detected_numeric": 1 if hit_rate_metric_detected else 0,
        },
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved prefix metrics probe JSON: {out_path}")

    if int(start_meta.get("returncode", 0)) != 0:
        print("Probe status: failed (startup command returned non-zero)")
        return 2
    if bool(args.require_prefix_metrics) and int(ports_with_prefix) <= 0:
        print("Probe status: failed (no prefix metrics detected)")
        return 3
    if bool(args.require_hit_rate_metric) and not hit_rate_metric_detected:
        print("Probe status: failed (no prefix hit-rate metric detected)")
        return 4
    if int(stop_meta.get("returncode", 0)) != 0:
        print("Probe status: failed (shutdown command returned non-zero)")
        return 5

    print("Probe status: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
