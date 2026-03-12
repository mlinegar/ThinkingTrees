#!/usr/bin/env python3
"""Unattended overnight optimization loop for teacher-trace generation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_MODEL = "/mnt/data/models/nvidia/Qwen3.5-397B-A17B-NVFP4"


@dataclass(frozen=True)
class RoundConfig:
    score_tolerance_raw: float
    max_attempts: int
    dspy_guidance_temperature: float
    dspy_guidance_max_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_tolerance_raw": float(self.score_tolerance_raw),
            "max_attempts": int(self.max_attempts),
            "dspy_guidance_temperature": float(self.dspy_guidance_temperature),
            "dspy_guidance_max_tokens": int(self.dspy_guidance_max_tokens),
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _append_jsonl_rows(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    _ensure_parent(path)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _quality_score(manifest: Dict[str, Any]) -> float:
    requested = max(1, int(manifest.get("requested_docs", 0) or 0))
    accepted = int(manifest.get("accepted_docs", 0) or 0)
    metrics = manifest.get("metrics", {}) if isinstance(manifest.get("metrics"), dict) else {}
    accept_rate = accepted / float(requested)
    same1 = float(metrics.get("same_side_summary1_pct", 0.0) or 0.0)
    same2 = float(metrics.get("same_side_summary2_pct", 0.0) or 0.0)
    c1 = float(metrics.get("c1_pass_pct", 0.0) or 0.0)
    c2 = float(metrics.get("c2_pass_pct", 0.0) or 0.0)
    expanded_mae = float(metrics.get("expanded_mae_raw", 100.0) or 100.0)
    hop_drift = float(metrics.get("summary2_vs_summary1_mae_raw", 100.0) or 100.0)
    return (
        accept_rate * 120.0
        + (same1 * 0.25)
        + (same2 * 0.25)
        + (c1 * 0.35)
        + (c2 * 0.35)
        - (expanded_mae * 0.7)
        - (hop_drift * 0.2)
    )


def _choose_round_config(
    *,
    round_index: int,
    history: Sequence[Dict[str, Any]],
    explore_configs: Sequence[RoundConfig],
) -> RoundConfig:
    if round_index < len(explore_configs):
        return explore_configs[round_index]
    if not history:
        return explore_configs[-1]

    best = max(history, key=lambda row: float(row.get("quality_score", float("-inf"))))
    best_cfg = best.get("config", {}) if isinstance(best.get("config"), dict) else {}
    cfg = RoundConfig(
        score_tolerance_raw=float(best_cfg.get("score_tolerance_raw", 25.0) or 25.0),
        max_attempts=int(best_cfg.get("max_attempts", 2) or 2),
        dspy_guidance_temperature=float(best_cfg.get("dspy_guidance_temperature", 0.15) or 0.15),
        dspy_guidance_max_tokens=max(
            1800,
            int(best_cfg.get("dspy_guidance_max_tokens", 1800) or 1800),
        ),
    )

    accepted = int(best.get("accepted_docs", 0) or 0)
    requested = max(1, int(best.get("requested_docs", 0) or 0))
    accept_rate = accepted / float(requested)
    expanded_mae = float(best.get("expanded_mae_raw", 999.0) or 999.0)

    if accept_rate >= 0.75 and expanded_mae <= 20.0 and cfg.score_tolerance_raw > 20.0:
        return RoundConfig(
            score_tolerance_raw=max(20.0, cfg.score_tolerance_raw - 5.0),
            max_attempts=min(4, cfg.max_attempts + 1),
            dspy_guidance_temperature=max(0.1, cfg.dspy_guidance_temperature - 0.02),
            dspy_guidance_max_tokens=min(3200, max(1800, cfg.dspy_guidance_max_tokens + 200)),
        )
    if accept_rate < 0.4:
        return RoundConfig(
            score_tolerance_raw=min(35.0, cfg.score_tolerance_raw + 5.0),
            max_attempts=min(4, cfg.max_attempts + 1),
            dspy_guidance_temperature=min(0.25, cfg.dspy_guidance_temperature + 0.02),
            dspy_guidance_max_tokens=min(3200, max(1800, cfg.dspy_guidance_max_tokens + 300)),
        )
    return cfg


def _merge_round_outputs(
    *,
    run_dir: Path,
    aggregate_dir: Path,
    seen_source_ids: set[str],
) -> Dict[str, int]:
    merged_counts = {
        "records_added": 0,
        "benchmark_added": 0,
        "summary_pairs_added": 0,
        "trace_artifacts_added": 0,
    }

    round_records_path = run_dir / "teacher_trace_records.jsonl"
    round_benchmark_path = run_dir / "benchmark_docs.jsonl"
    round_pairs_path = run_dir / "summary_training_pairs.jsonl"
    round_trace_path = run_dir / "trace_artifacts.jsonl"

    agg_records_path = aggregate_dir / "teacher_trace_records.jsonl"
    agg_benchmark_path = aggregate_dir / "benchmark_docs.jsonl"
    agg_pairs_path = aggregate_dir / "summary_training_pairs.jsonl"
    agg_trace_path = aggregate_dir / "trace_artifacts.jsonl"

    round_records = list(_iter_jsonl(round_records_path))
    new_records: List[Dict[str, Any]] = []
    new_example_ids: set[str] = set()

    for row in round_records:
        source_id = str(row.get("source_manifesto_id", "") or "")
        example_id = str(row.get("example_id", "") or "")
        if not source_id or not example_id:
            continue
        if source_id in seen_source_ids:
            continue
        seen_source_ids.add(source_id)
        new_records.append(row)
        new_example_ids.add(example_id)

    merged_counts["records_added"] = _append_jsonl_rows(agg_records_path, new_records)

    benchmark_rows = [
        row
        for row in _iter_jsonl(round_benchmark_path)
        if str(row.get("id", "") or "") in new_example_ids
    ]
    merged_counts["benchmark_added"] = _append_jsonl_rows(agg_benchmark_path, benchmark_rows)

    pair_rows = [
        row
        for row in _iter_jsonl(round_pairs_path)
        if str(row.get("example_id", "") or "") in new_example_ids
    ]
    merged_counts["summary_pairs_added"] = _append_jsonl_rows(agg_pairs_path, pair_rows)

    trace_rows = [
        row
        for row in _iter_jsonl(round_trace_path)
        if str(row.get("example_id", "") or "") in new_example_ids
    ]
    merged_counts["trace_artifacts_added"] = _append_jsonl_rows(agg_trace_path, trace_rows)
    return merged_counts


def _render_report(
    *,
    report_path: Path,
    history: Sequence[Dict[str, Any]],
    aggregate_dir: Path,
    best_row: Optional[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# Teacher Trace Overnight Report")
    lines.append("")
    lines.append(f"Generated at: {_now_iso()}")
    lines.append("")
    if best_row:
        lines.append("## Best Round")
        lines.append("")
        lines.append(f"- Round: {best_row.get('round')}")
        lines.append(f"- Quality score: {best_row.get('quality_score'):.3f}")
        lines.append(f"- Accepted / Requested: {best_row.get('accepted_docs')} / {best_row.get('requested_docs')}")
        lines.append(f"- Expanded MAE raw: {best_row.get('expanded_mae_raw')}")
        lines.append(f"- C1 pass pct: {best_row.get('c1_pass_pct')}")
        lines.append(f"- C2 pass pct: {best_row.get('c2_pass_pct')}")
        lines.append(f"- Config: `{json.dumps(best_row.get('config', {}), sort_keys=True)}`")
        lines.append("")

    lines.append("## Aggregate Artifacts")
    lines.append("")
    lines.append(f"- records: {_line_count(aggregate_dir / 'teacher_trace_records.jsonl')}")
    lines.append(f"- benchmark_docs: {_line_count(aggregate_dir / 'benchmark_docs.jsonl')}")
    lines.append(f"- summary_training_pairs: {_line_count(aggregate_dir / 'summary_training_pairs.jsonl')}")
    lines.append(f"- trace_artifacts: {_line_count(aggregate_dir / 'trace_artifacts.jsonl')}")
    lines.append("")

    lines.append("## Rounds")
    lines.append("")
    for row in history:
        lines.append(
            "- "
            f"round={row.get('round')} "
            f"accepted={row.get('accepted_docs')}/{row.get('requested_docs')} "
            f"expanded_mae={row.get('expanded_mae_raw')} "
            f"same_side1={row.get('same_side_summary1_pct')} "
            f"same_side2={row.get('same_side_summary2_pct')} "
            f"c1={row.get('c1_pass_pct')} "
            f"c2={row.get('c2_pass_pct')} "
            f"score={row.get('quality_score'):.3f} "
            f"config={json.dumps(row.get('config', {}), sort_keys=True)}"
        )

    _ensure_parent(report_path)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run teacher-trace generation overnight with adaptive tuning")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--docs-per-round", type=int, default=12)
    parser.add_argument("--sleep-seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--teacher-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--scorer-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--teacher-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--scorer-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--teacher-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--scorer-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--max-source-chars", type=int, default=0)
    parser.add_argument("--expand-max-tokens", type=int, default=900)
    parser.add_argument("--summary-max-tokens", type=int, default=260)
    parser.add_argument("--trace-max-tokens", type=int, default=360)
    parser.add_argument("--score-max-tokens", type=int, default=96)
    parser.add_argument("--min-source-chars", type=int, default=1200)
    parser.add_argument("--dspy-guidance-source-max-chars", type=int, default=262144)
    parser.add_argument("--dspy-guidance-expansion-max-chars", type=int, default=262144)
    parser.add_argument("--previous-expansion-max-chars", type=int, default=262144)
    parser.add_argument("--revision-guidance-max-chars", type=int, default=65536)
    parser.add_argument("--trace-source-max-chars", type=int, default=262144)
    parser.add_argument("--trace-expanded-max-chars", type=int, default=262144)
    parser.add_argument(
        "--allow-source-shrink-on-error",
        action="store_true",
        help="Pass through fallback that may shrink source prompts after request failures.",
    )

    parser.add_argument("--manifesto-ids", nargs="*", default=None)
    parser.add_argument("--no-enable-thinking", action="store_true", default=True)

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.docs_per_round <= 0:
        raise ValueError("--docs-per-round must be positive")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or (Path("outputs") / f"teacher_trace_overnight_{stamp}")
    runs_dir = output_root / "runs"
    logs_dir = output_root / "logs"
    aggregate_dir = output_root / "aggregate"
    output_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    status_path = output_root / "overnight_status.json"
    report_path = output_root / "overnight_report.md"

    explore_configs: List[RoundConfig] = [
        RoundConfig(20.0, 3, 0.10, 2600),
        RoundConfig(25.0, 3, 0.10, 2200),
        RoundConfig(25.0, 2, 0.15, 2000),
        RoundConfig(30.0, 2, 0.15, 1800),
    ]

    seen_source_ids: set[str] = set()
    history: List[Dict[str, Any]] = []

    env = dict(os.environ)
    env["PYTHONPATH"] = f".{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)

    for round_idx in range(int(args.rounds)):
        config = _choose_round_config(
            round_index=round_idx,
            history=history,
            explore_configs=explore_configs,
        )
        run_name = f"round_{round_idx + 1:02d}"
        run_dir = runs_dir / run_name
        run_log = logs_dir / f"{run_name}.log"

        cmd: List[str] = [
            sys.executable,
            "scripts/generate_manifesto_teacher_traces.py",
            "--output-dir",
            str(run_dir),
            "--train-size",
            str(args.docs_per_round),
            "--val-size",
            "0",
            "--test-size",
            "0",
            "--seed",
            str(int(args.seed) + round_idx),
            "--min-source-chars",
            str(args.min_source_chars),
            "--max-source-chars",
            str(args.max_source_chars),
            "--teacher-base-url",
            str(args.teacher_base_url),
            "--scorer-base-url",
            str(args.scorer_base_url),
            "--teacher-model",
            str(args.teacher_model),
            "--scorer-model",
            str(args.scorer_model),
            "--teacher-timeout-seconds",
            str(args.teacher_timeout_seconds),
            "--scorer-timeout-seconds",
            str(args.scorer_timeout_seconds),
            "--expand-max-tokens",
            str(args.expand_max_tokens),
            "--summary-max-tokens",
            str(args.summary_max_tokens),
            "--trace-max-tokens",
            str(args.trace_max_tokens),
            "--score-max-tokens",
            str(args.score_max_tokens),
            "--score-tolerance-raw",
            str(config.score_tolerance_raw),
            "--max-attempts",
            str(config.max_attempts),
            "--use-dspy-guidance",
            "--dspy-guidance-temperature",
            str(config.dspy_guidance_temperature),
            "--dspy-guidance-max-tokens",
            str(config.dspy_guidance_max_tokens),
            "--dspy-guidance-source-max-chars",
            str(args.dspy_guidance_source_max_chars),
            "--dspy-guidance-expansion-max-chars",
            str(args.dspy_guidance_expansion_max_chars),
            "--previous-expansion-max-chars",
            str(args.previous_expansion_max_chars),
            "--revision-guidance-max-chars",
            str(args.revision_guidance_max_chars),
            "--trace-source-max-chars",
            str(args.trace_source_max_chars),
            "--trace-expanded-max-chars",
            str(args.trace_expanded_max_chars),
        ]
        if bool(args.no_enable_thinking):
            cmd.append("--no-enable-thinking")
        if bool(args.allow_source_shrink_on_error):
            cmd.append("--allow-source-shrink-on-error")
        if args.manifesto_ids:
            cmd.append("--manifesto-ids")
            cmd.extend([str(v) for v in args.manifesto_ids])

        start = time.time()
        _ensure_parent(run_log)
        with run_log.open("w", encoding="utf-8") as log_handle:
            log_handle.write(f"# started_utc={_now_iso()}\n")
            log_handle.write("# cmd=" + " ".join(cmd) + "\n\n")
            log_handle.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(Path(__file__).resolve().parents[1]),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = float(time.time() - start)

        manifest_path = run_dir / "manifest.json"
        manifest: Dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}

        requested_docs = int(manifest.get("requested_docs", args.docs_per_round) or args.docs_per_round)
        accepted_docs = int(manifest.get("accepted_docs", 0) or 0)
        metrics = manifest.get("metrics", {}) if isinstance(manifest.get("metrics"), dict) else {}

        merged_counts = _merge_round_outputs(
            run_dir=run_dir,
            aggregate_dir=aggregate_dir,
            seen_source_ids=seen_source_ids,
        )
        quality = _quality_score(manifest) if manifest else float("-inf")

        round_row: Dict[str, Any] = {
            "round": round_idx + 1,
            "started_utc": _now_iso(),
            "elapsed_seconds": elapsed,
            "return_code": int(proc.returncode),
            "run_dir": str(run_dir),
            "run_log": str(run_log),
            "requested_docs": requested_docs,
            "accepted_docs": accepted_docs,
            "expanded_mae_raw": float(metrics.get("expanded_mae_raw", 999.0) or 999.0),
            "summary1_mae_raw": float(metrics.get("summary1_mae_raw", 999.0) or 999.0),
            "summary2_mae_raw": float(metrics.get("summary2_mae_raw", 999.0) or 999.0),
            "same_side_summary1_pct": float(metrics.get("same_side_summary1_pct", 0.0) or 0.0),
            "same_side_summary2_pct": float(metrics.get("same_side_summary2_pct", 0.0) or 0.0),
            "c1_pass_pct": float(metrics.get("c1_pass_pct", 0.0) or 0.0),
            "c2_pass_pct": float(metrics.get("c2_pass_pct", 0.0) or 0.0),
            "quality_score": float(quality),
            "config": config.to_dict(),
            "merged_counts": merged_counts,
        }
        history.append(round_row)

        best_row = None if not history else max(history, key=lambda row: float(row.get("quality_score", float("-inf"))))
        status_payload = {
            "updated_utc": _now_iso(),
            "output_root": str(output_root),
            "rounds_completed": len(history),
            "best_round": best_row,
            "history": history,
            "aggregate_counts": {
                "records": _line_count(aggregate_dir / "teacher_trace_records.jsonl"),
                "benchmark_docs": _line_count(aggregate_dir / "benchmark_docs.jsonl"),
                "summary_pairs": _line_count(aggregate_dir / "summary_training_pairs.jsonl"),
                "trace_artifacts": _line_count(aggregate_dir / "trace_artifacts.jsonl"),
            },
        }
        _write_json(status_path, status_payload)
        _render_report(
            report_path=report_path,
            history=history,
            aggregate_dir=aggregate_dir,
            best_row=best_row,
        )

        if proc.returncode != 0:
            # Keep going after failures to avoid losing the whole night.
            time.sleep(max(0.0, float(args.sleep_seconds)))
            continue
        time.sleep(max(0.0, float(args.sleep_seconds)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
