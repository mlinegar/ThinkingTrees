#!/usr/bin/env python3
"""
Standard pipeline run report.

Reads final_stats.json (and optionally run.log) from a training pipeline
output directory and prints a concise summary covering quality, efficiency,
and Phase 1-6 component status.

Usage:
    python scripts/report_pipeline_run.py <run_dir>
    python scripts/report_pipeline_run.py --latest
    python scripts/report_pipeline_run.py --latest --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Defaults ──────────────────────────────────────────────────────────────────

RESULTS_BASE = PROJECT_ROOT / "data" / "results"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(val: Any, digits: int = 3) -> str:
    if val is None:
        return "n/a"
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return str(val)


def _pct(val: Any, digits: int = 1) -> str:
    if val is None:
        return "n/a"
    try:
        return f"{float(val):.{digits}f}%"
    except (TypeError, ValueError):
        return str(val)


def _duration(started: Optional[str], completed: Optional[str]) -> Optional[float]:
    if not started or not completed:
        return None
    try:
        t0 = datetime.fromisoformat(str(started))
        t1 = datetime.fromisoformat(str(completed))
        return (t1 - t0).total_seconds()
    except Exception:
        return None


def _find_latest_run() -> Optional[Path]:
    """Find the most recently modified run directory under RESULTS_BASE."""
    candidates: List[Tuple[float, Path]] = []
    for task_dir in RESULTS_BASE.iterdir():
        pipeline_dir = task_dir / "training_pipeline"
        if not pipeline_dir.is_dir():
            continue
        for run_dir in pipeline_dir.iterdir():
            stats = run_dir / "final_stats.json"
            if stats.exists():
                candidates.append((stats.stat().st_mtime, run_dir))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ── Log parsing ───────────────────────────────────────────────────────────────

_PHASE_RE = re.compile(r"PHASE (\S+):")
_RECOVERY_RE = re.compile(r"Recovering (\S+) server on port (\d+)")
_TRANSITION_RE = re.compile(r"Transitioned to (\S+) mode in ([0-9.]+)s")
_METRICS_RE = re.compile(r"Inference metrics: (.+)")
_BATCH_STATS_RE = re.compile(
    r"Cascading progress:.*rate=([0-9.]+) items/s.*tokens=([0-9,]+).*tok/s=(\d+)"
)
_CASCADING_DONE_RE = re.compile(r"Cascading build complete: (\d+)/(\d+) documents")
_WARNING_RE = re.compile(r"\| WARNING \|")
_ERROR_RE = re.compile(r"\| ERROR \|")


def parse_log(log_path: Path) -> Dict[str, Any]:
    """Extract timing, transitions, warnings, and throughput from run.log."""
    info: Dict[str, Any] = {
        "phases_seen": [],
        "recoveries": [],
        "transitions": [],
        "warnings": 0,
        "errors": 0,
        "peak_items_per_sec": 0.0,
        "peak_tok_per_sec": 0,
        "total_tokens": 0,
        "cascade_docs_built": 0,
        "cascade_docs_total": 0,
        "prefix_cache_nonzero": False,
    }
    if not log_path.exists():
        return info

    for line in log_path.read_text(errors="replace").splitlines():
        m = _PHASE_RE.search(line)
        if m:
            phase = m.group(1)
            if phase not in info["phases_seen"]:
                info["phases_seen"].append(phase)

        m = _RECOVERY_RE.search(line)
        if m:
            info["recoveries"].append({"server": m.group(1), "port": int(m.group(2))})

        m = _TRANSITION_RE.search(line)
        if m:
            info["transitions"].append({"mode": m.group(1), "seconds": float(m.group(2))})

        m = _BATCH_STATS_RE.search(line)
        if m:
            rate = float(m.group(1))
            tok_s = int(m.group(3))
            if rate > info["peak_items_per_sec"]:
                info["peak_items_per_sec"] = rate
            if tok_s > info["peak_tok_per_sec"]:
                info["peak_tok_per_sec"] = tok_s
            info["total_tokens"] = max(info["total_tokens"], int(m.group(2).replace(",", "")))

        m = _CASCADING_DONE_RE.search(line)
        if m:
            info["cascade_docs_built"] += int(m.group(1))
            info["cascade_docs_total"] += int(m.group(2))

        m = _METRICS_RE.search(line)
        if m and "pfx=" in line:
            # Check if any prefix cache is nonzero
            for pfx_match in re.finditer(r"pfx=(\d+)", line):
                if int(pfx_match.group(1)) > 0:
                    info["prefix_cache_nonzero"] = True

        if _WARNING_RE.search(line):
            info["warnings"] += 1
        if _ERROR_RE.search(line):
            info["errors"] += 1

    return info


# ── Report rendering ──────────────────────────────────────────────────────────

def render_report(stats: Dict[str, Any], log_info: Dict[str, Any], run_dir: Path) -> str:
    """Render the standard pipeline report as a string."""
    lines: List[str] = []
    cfg = stats.get("config", {})

    # ── Header ────────────────────────────────────────────────────────────
    lines.append("=" * 72)
    lines.append("  PIPELINE RUN REPORT")
    lines.append("=" * 72)
    lines.append("")

    # ── Run metadata ──────────────────────────────────────────────────────
    started = stats.get("started_at", "")
    completed = stats.get("completed_at", "")
    duration = _duration(started, completed)
    success = stats.get("success", False)

    lines.append(f"  Task:       {cfg.get('task', 'unknown')}")
    lines.append(f"  Dataset:    {cfg.get('dataset', 'unknown')}")
    lines.append(f"  Samples:    train={cfg.get('train_samples', '?')} val={cfg.get('val_samples', '?')} test={cfg.get('test_samples', '?')}")
    lines.append(f"  Optimizer:  {cfg.get('optimizer', '?')} ({cfg.get('optimizer_budget', '?')})")
    lines.append(f"  Iterations: {cfg.get('n_iterations', '?')}")
    lines.append(f"  Started:    {started}")
    lines.append(f"  Duration:   {duration:.0f}s" if duration else "  Duration:   n/a")
    lines.append(f"  Status:     {'OK' if success else 'FAILED'}")
    lines.append(f"  Output:     {run_dir}")
    lines.append("")

    # ── Component status ──────────────────────────────────────────────────
    lines.append("-" * 72)
    lines.append("  COMPONENT STATUS (Phases 1-6)")
    lines.append("-" * 72)

    def _on_off(key: str, *, true_val: str = "ON", false_val: str = "off") -> str:
        v = cfg.get(key)
        if v is None:
            return false_val
        if isinstance(v, bool):
            return true_val if v else false_val
        return str(v)

    routing = cfg.get("routing_policy", "round_robin")
    cm_mode = cfg.get("conditional_memory_mode", "off")
    engram = cfg.get("engram_memory", False)
    genrm = cfg.get("enable_genrm", False)
    dynamic = cfg.get("dynamic_gpu", False)

    lines.append(f"  [1.2] Routing policy:        {routing}")
    lines.append(f"  [3.0] ConditionalMemory:      {cm_mode}")
    lines.append(f"  [3.4] Engram memory:          {'ON' if engram else 'off'}")
    lines.append(f"  [4.0] GenRM enabled:          {'ON' if genrm else 'off'}")
    lines.append(f"  [4.1] Dynamic GPU:            {'ON' if dynamic else 'off'}")

    # Backend info
    lines.append(f"  [--]  Task backend:           {cfg.get('task_backend', 'vllm')}")
    lines.append(f"  [--]  GenRM backend:          {cfg.get('genrm_backend', 'vllm')}")
    lines.append("")

    # ── Method status ─────────────────────────────────────────────────────
    method_status = stats.get("method_status", {})
    if isinstance(method_status, dict) and method_status:
        lines.append("-" * 72)
        lines.append("  METHOD STATUS")
        lines.append("-" * 72)
        ordered = [
            "llm_prompt_optimization",
            "embedding_proxy",
            "neural_operators",
            "generator_finetune",
        ]
        for key in ordered:
            row = method_status.get(key)
            if not isinstance(row, dict):
                continue
            enabled = bool(row.get("enabled", False))
            attempted = bool(row.get("attempted", False))
            completed = bool(row.get("completed", False))
            skipped = bool(row.get("skipped", False))
            error = str(row.get("error", "") or "").strip()
            duration = row.get("duration_seconds")
            artifacts = row.get("artifact_paths", [])
            n_artifacts = len(artifacts) if isinstance(artifacts, list) else 0
            status = "pending"
            if skipped:
                status = "skipped"
            elif completed:
                status = "ok"
            elif error:
                status = "error"
            elif attempted:
                status = "running"
            lines.append(
                f"  {key:24s} enabled={str(enabled):5s} status={status:8s} "
                f"attempted={str(attempted):5s} duration={_fmt(duration, 1)}s artifacts={n_artifacts}"
            )
            if error:
                lines.append(f"    error: {error}")
        lines.append("")

    # ── Quality metrics ───────────────────────────────────────────────────
    lines.append("-" * 72)
    lines.append("  QUALITY METRICS")
    lines.append("-" * 72)

    for split in ("train", "test"):
        s = stats.get(split, {})
        if not s:
            continue
        lines.append(f"  {split.upper():>5}:  MAE={_fmt(s.get('mae'))}  "
                      f"Pearson={_fmt(s.get('pearson_r'))}  "
                      f"Spearman={_fmt(s.get('spearman_r'))}  "
                      f"in5%={_pct(s.get('within_5pct'))}  "
                      f"in10%={_pct(s.get('within_10pct'))}  "
                      f"n={s.get('n_evaluated', '?')}")
    lines.append("")

    # Prediction distribution diagnostics
    for split in ("train", "test"):
        s = stats.get(split, {})
        pd = s.get("prediction_distribution", {})
        if pd:
            frac_neutral = pd.get("frac_neutral")
            n_unique = pd.get("n_unique_rounded_4dp")
            note = ""
            if frac_neutral is not None and float(frac_neutral) > 0.5:
                note = "  ** >50% neutral predictions **"
            neutral_pct = float(frac_neutral) * 100 if frac_neutral is not None else None
            lines.append(f"  {split.upper():>5} dist: mean={_fmt(pd.get('mean'))} "
                          f"std={_fmt(pd.get('std'))} "
                          f"unique={n_unique} "
                          f"neutral={_pct(neutral_pct, 0)}"
                          f"{note}")
    lines.append("")

    # ── Optimization rounds ───────────────────────────────────────────────
    rounds = stats.get("rounds", [])
    if rounds:
        lines.append("-" * 72)
        lines.append("  OPTIMIZATION ROUNDS")
        lines.append("-" * 72)
        for r in rounds:
            rn = r.get("round", "?")
            before = _fmt(r.get("metric_before"))
            after = _fmt(r.get("metric_after"))
            opt = r.get("optimizer_used", "?")
            lines.append(f"  Round {rn}: {before} -> {after} ({opt})")
            summ = r.get("summarizer", {})
            if summ:
                leaf = summ.get("leaf", {})
                merge = summ.get("merge", {})
                lines.append(f"    Leaf:  {_fmt(leaf.get('metric_before'))} -> {_fmt(leaf.get('metric_after'))}  "
                              f"(examples: {summ.get('leaf_train_examples', '?')})")
                lines.append(f"    Merge: {_fmt(merge.get('metric_before'))} -> {_fmt(merge.get('metric_after'))}  "
                              f"(examples: {summ.get('merge_train_examples', '?')})")
        lines.append("")

    # ── ConditionalMemory ─────────────────────────────────────────────────
    cm = stats.get("conditional_memory")
    if cm:
        lines.append("-" * 72)
        lines.append("  CONDITIONAL MEMORY")
        lines.append("-" * 72)
        lines.append(f"  Mode:             {cm.get('mode', '?')}")
        lines.append(f"  Namespace:        {cm.get('namespace_version', '?')}")
        lines.append(f"  L1 entries:       {cm.get('l1_entries', 0)}")
        lines.append(f"  L2 entries:       {cm.get('l2_entries', 0)}")
        lines.append(f"  Hits (L1/L2):     {cm.get('l1_hits', 0)} / {cm.get('l2_hits', 0)}")
        lines.append(f"  Misses:           {cm.get('misses', 0)}")
        lines.append(f"  Writes:           {cm.get('writes', 0)} ({cm.get('bytes_written', 0):,} bytes)")
        lines.append(f"  Hit rate:         {_pct(cm.get('hit_rate', 0) * 100 if cm.get('hit_rate') is not None else None, 1)}")
        lines.append(f"  L2 path:          {cm.get('l2_path', 'n/a')}")
        lines.append("")

    # ── Throughput & efficiency (from log) ────────────────────────────────
    if log_info.get("peak_tok_per_sec", 0) > 0 or log_info.get("transitions"):
        lines.append("-" * 72)
        lines.append("  THROUGHPUT & EFFICIENCY")
        lines.append("-" * 72)

        if log_info.get("peak_tok_per_sec", 0) > 0:
            lines.append(f"  Peak tok/s:       {log_info['peak_tok_per_sec']:,}")
            lines.append(f"  Peak items/s:     {log_info['peak_items_per_sec']:.2f}")
            lines.append(f"  Total tokens:     {log_info.get('total_tokens', 0):,}")

        if log_info.get("cascade_docs_total", 0) > 0:
            lines.append(f"  Trees built:      {log_info['cascade_docs_built']}/{log_info['cascade_docs_total']}")

        lines.append(f"  Prefix cache hit: {'yes' if log_info.get('prefix_cache_nonzero') else 'no (0% observed)'}")

        if log_info.get("transitions"):
            for t in log_info["transitions"]:
                lines.append(f"  GPU transition:   -> {t['mode']} in {t['seconds']:.1f}s")

        lines.append("")

    # ── Issues ────────────────────────────────────────────────────────────
    if log_info.get("warnings", 0) > 0 or log_info.get("errors", 0) > 0 or log_info.get("recoveries"):
        lines.append("-" * 72)
        lines.append("  ISSUES")
        lines.append("-" * 72)
        lines.append(f"  Warnings:         {log_info.get('warnings', 0)}")
        lines.append(f"  Errors:           {log_info.get('errors', 0)}")
        if log_info.get("recoveries"):
            for rec in log_info["recoveries"]:
                lines.append(f"  Recovery:         {rec['server']} on port {rec['port']}")
        lines.append("")

    # ── Phases seen ───────────────────────────────────────────────────────
    if log_info.get("phases_seen"):
        lines.append("-" * 72)
        lines.append(f"  Phases completed: {', '.join(log_info['phases_seen'])}")
        lines.append("-" * 72)
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standard pipeline run report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dir", nargs="?", default=None,
        help="Path to run directory (containing final_stats.json).",
    )
    parser.add_argument(
        "--latest", action="store_true",
        help="Automatically find the most recent run.",
    )
    parser.add_argument(
        "--json", action="store_true", dest="emit_json",
        help="Emit raw JSON instead of formatted report.",
    )
    args = parser.parse_args()

    # Resolve run directory
    if args.latest:
        run_dir = _find_latest_run()
        if run_dir is None:
            print("ERROR: No runs found under", RESULTS_BASE, file=sys.stderr)
            return 1
    elif args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        parser.print_help()
        return 1

    stats_path = run_dir / "final_stats.json"
    if not stats_path.exists():
        print(f"ERROR: {stats_path} not found", file=sys.stderr)
        return 1

    stats = json.loads(stats_path.read_text())

    if args.emit_json:
        # Compact JSON summary for programmatic consumption
        cfg = stats.get("config", {})
        cm = stats.get("conditional_memory", {})
        summary = {
            "run_dir": str(run_dir),
            "task": cfg.get("task"),
            "success": stats.get("success"),
            "duration_s": _duration(stats.get("started_at"), stats.get("completed_at")),
            "samples": {
                "train": cfg.get("train_samples"),
                "val": cfg.get("val_samples"),
                "test": cfg.get("test_samples"),
            },
            "components": {
                "routing_policy": cfg.get("routing_policy"),
                "conditional_memory": cfg.get("conditional_memory_mode"),
                "engram_memory": cfg.get("engram_memory"),
                "genrm": cfg.get("enable_genrm"),
                "dynamic_gpu": cfg.get("dynamic_gpu"),
            },
            "quality": {},
            "conditional_memory": cm if cm else None,
        }
        for split in ("train", "test"):
            s = stats.get(split, {})
            if s:
                summary["quality"][split] = {
                    "mae": s.get("mae"),
                    "pearson_r": s.get("pearson_r"),
                    "spearman_r": s.get("spearman_r"),
                    "within_10pct": s.get("within_10pct"),
                    "n": s.get("n_evaluated"),
                }
        print(json.dumps(summary, indent=2))
        return 0

    log_path = run_dir / "run.log"
    log_info = parse_log(log_path)

    report = render_report(stats, log_info, run_dir)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
