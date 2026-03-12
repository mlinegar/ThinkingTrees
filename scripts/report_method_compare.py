#!/usr/bin/env python3
"""
Aggregate method-comparison runs into JSON and Markdown summaries.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _method_cell(method_status: Dict[str, Any], key: str) -> str:
    row = method_status.get(key)
    if not isinstance(row, dict):
        return "n/a"
    if row.get("skipped"):
        return "skip"
    if row.get("completed"):
        return "ok"
    if row.get("error"):
        return "err"
    if row.get("attempted"):
        return "run"
    return "n/a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize run_method_compare outputs.")
    parser.add_argument("--manifest", required=True, help="Path to method_compare_manifest.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    output_root = manifest_path.parent
    rows: List[Dict[str, Any]] = []
    for entry in manifest.get("entries", []):
        if not isinstance(entry, dict):
            continue
        profile = str(entry.get("profile", "unknown"))
        run_dir = Path(str(entry.get("run_dir", output_root / profile)))
        stats_path = run_dir / "final_stats.json"
        stats: Dict[str, Any] = {}
        if stats_path.exists():
            try:
                with open(stats_path, "r", encoding="utf-8") as handle:
                    stats = json.load(handle)
            except Exception:
                stats = {}
        method_status = stats.get("method_status", {}) if isinstance(stats.get("method_status"), dict) else {}
        row = {
            "profile": profile,
            "run_status": str(entry.get("status", "unknown")),
            "exit_code": entry.get("exit_code"),
            "duration_seconds": _safe_float(entry.get("duration_seconds")),
            "pipeline_success": bool(stats.get("success")) if stats else None,
            "train_mae": _safe_float((stats.get("train") or {}).get("mae")) if isinstance(stats.get("train"), dict) else None,
            "test_mae": _safe_float((stats.get("test") or {}).get("mae")) if isinstance(stats.get("test"), dict) else None,
            "method_status": method_status,
            "run_dir": str(run_dir),
            "final_stats_path": str(stats_path),
        }
        rows.append(row)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "mode": manifest.get("mode"),
        "task": manifest.get("task"),
        "dataset": manifest.get("dataset"),
        "rows": rows,
    }

    summary_json_path = output_root / "comparison_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    md_lines: List[str] = []
    md_lines.append("# Method Comparison Summary")
    md_lines.append("")
    md_lines.append(f"- Generated: {summary['generated_at']}")
    md_lines.append(f"- Manifest: `{manifest_path}`")
    md_lines.append(f"- Task/Dataset: `{summary.get('task')}` / `{summary.get('dataset')}`")
    md_lines.append("")
    md_lines.append("| Profile | Run | Pipeline | Train MAE | Test MAE | Duration (s) | LLM Opt | Embed | Neural | Generator |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        method_status = row.get("method_status", {})
        md_lines.append(
            "| "
            f"{row['profile']} | "
            f"{row['run_status']} | "
            f"{row['pipeline_success']} | "
            f"{_fmt(row['train_mae'])} | "
            f"{_fmt(row['test_mae'])} | "
            f"{_fmt(row['duration_seconds'], 1)} | "
            f"{_method_cell(method_status, 'llm_prompt_optimization')} | "
            f"{_method_cell(method_status, 'embedding_proxy')} | "
            f"{_method_cell(method_status, 'neural_operators')} | "
            f"{_method_cell(method_status, 'generator_finetune')} |"
        )
    md_lines.append("")
    md_lines.append("## Runs")
    for row in rows:
        md_lines.append(f"- `{row['profile']}`: `{row['run_dir']}`")

    summary_md_path = output_root / "comparison_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(md_lines) + "\n")

    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary MD:   {summary_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
