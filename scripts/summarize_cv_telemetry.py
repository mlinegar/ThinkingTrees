#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


WARNING_KEYS = ("neutral_fallbacks", "lm_timeouts", "lm_internal_errors")
TIMESTAMP_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \|")
GEPA_FINISH_PATTERN = re.compile(
    r"(Scorer|Leaf summarizer|Merge summarizer) optimization \(gepa\) finished in ([0-9]+(?:\.[0-9]+)?)s"
)
NEUTRAL_PATTERN = re.compile(r"RILEScorer prediction failed; defaulting to neutral")
TIMEOUT_PATTERN = re.compile(r"LM timeout")
INTERNAL_PATTERN = re.compile(r"InternalServerError|Connection error")


@dataclass
class PhaseAggregate:
    samples: int = 0
    util_sum: float = 0.0
    util_max_peak: float = 0.0
    warning_deltas: Dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.warning_deltas is None:
            self.warning_deltas = {k: 0 for k in WARNING_KEYS}

    def add(self, *, util_mean: Optional[float], util_max: Optional[float], deltas: Dict[str, int]) -> None:
        self.samples += 1
        if util_mean is not None:
            self.util_sum += float(util_mean)
        if util_max is not None:
            self.util_max_peak = max(self.util_max_peak, float(util_max))
        for key in WARNING_KEYS:
            self.warning_deltas[key] += int(deltas.get(key, 0) or 0)

    def as_dict(self) -> Dict[str, Any]:
        avg = self.util_sum / self.samples if self.samples > 0 else None
        return {
            "samples": self.samples,
            "avg_gpu_util_mean_pct": avg,
            "peak_gpu_util_max_pct": self.util_max_peak if self.samples > 0 else None,
            "warning_deltas": dict(self.warning_deltas),
        }


def _parse_iso(ts: str) -> Optional[datetime]:
    value = str(ts or "").strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def _find_latest_throughput_json(repo_root: Path) -> Optional[Path]:
    candidates = sorted(repo_root.glob("outputs/throughput_limits*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _phase_from_line(last_line: Optional[str]) -> str:
    text = (last_line or "").lower()
    if "leaf summarizer optimization" in text:
        return "gepa_leaf"
    if "merge summarizer optimization" in text:
        return "gepa_merge"
    if "scorer optimization" in text or "score predictor" in text:
        return "gepa_scorer"
    if "completed " in text and "/" in text:
        return "batch_docs"
    if "global pipelined processing complete" in text:
        return "batch_docs"
    if "evaluating on test set" in text or "test results" in text:
        return "eval_test"
    if "leaf-score export" in text:
        return "leaf_export"
    if "phase 1.5" in text and "checkpoint" in text:
        return "phase1_5_resume"
    if "loading from checkpoint" in text:
        return "resume_checkpoint"
    return "other"


def _extract_int_flag(cmd: Optional[str], flag: str) -> Optional[int]:
    if not cmd:
        return None
    try:
        toks = shlex.split(cmd)
    except Exception:
        toks = str(cmd).split()
    for idx, tok in enumerate(toks):
        if tok == flag and idx + 1 < len(toks):
            try:
                return int(toks[idx + 1])
            except Exception:
                return None
        if tok.startswith(flag + "="):
            try:
                return int(tok.split("=", 1)[1])
            except Exception:
                return None
    return None


def _round_step(value: int, step: int = 16, min_value: int = 32, max_value: int = 1024) -> int:
    value = max(min_value, min(max_value, int(value)))
    return int(round(value / step) * step)


def _load_throughput_recommendations(path: Optional[Path]) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {
        "task_dp2": None,
        "task_merge": None,
        "task_score": None,
        "genrm_batch_fast": None,
        "genrm_batch_think": None,
        "genrm_raw_fast": None,
        "genrm_raw_think": None,
    }
    if path is None or not path.exists():
        return out
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return out
    steps = payload.get("steps", {})
    if not isinstance(steps, dict):
        return out
    for key in list(out.keys()):
        summary = steps.get(key, {}).get("summary", {})
        value = summary.get("recommended_concurrency")
        if isinstance(value, (int, float)) and value > 0:
            out[key] = int(value)
    return out


def _compute_warning_deltas(samples: List[Dict[str, Any]]) -> Dict[str, int]:
    totals = {k: 0 for k in WARNING_KEYS}
    prev_by_fold: Dict[str, Dict[str, int]] = {}
    for sample in samples:
        fold_key = str(sample.get("active_fold") or "__none__")
        cur_raw = sample.get("active_log_recent_warnings") or {}
        cur = {k: int(cur_raw.get(k, 0) or 0) for k in WARNING_KEYS}
        prev = prev_by_fold.get(fold_key)
        if prev is not None:
            for key in WARNING_KEYS:
                delta = cur[key] - prev.get(key, 0)
                if delta > 0:
                    totals[key] += delta
        prev_by_fold[fold_key] = cur
    return totals


def _aggregate_phases(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    phases: Dict[str, PhaseAggregate] = defaultdict(PhaseAggregate)
    prev_by_fold: Dict[str, Dict[str, int]] = {}

    for sample in samples:
        phase = _phase_from_line(sample.get("active_log_last_line"))
        fold_key = str(sample.get("active_fold") or "__none__")
        cur_raw = sample.get("active_log_recent_warnings") or {}
        cur = {k: int(cur_raw.get(k, 0) or 0) for k in WARNING_KEYS}
        prev = prev_by_fold.get(fold_key)
        deltas = {k: 0 for k in WARNING_KEYS}
        if prev is not None:
            for key in WARNING_KEYS:
                d = cur[key] - prev.get(key, 0)
                deltas[key] = d if d > 0 else 0
        prev_by_fold[fold_key] = cur
        phases[phase].add(
            util_mean=sample.get("gpu_util_mean_pct"),
            util_max=sample.get("gpu_util_max_pct"),
            deltas=deltas,
        )
    return {name: agg.as_dict() for name, agg in sorted(phases.items())}


def _parse_fold_timing(log_text: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    lines = [ln for ln in log_text.splitlines() if ln.strip()]
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    for line in lines:
        if "Starting training pipeline" in line:
            m = TIMESTAMP_LINE.match(line)
            if m:
                start = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                break
    for line in reversed(lines):
        if "Results saved to " in line or "Generating PDF report:" in line or "Pipeline run complete" in line:
            m = TIMESTAMP_LINE.match(line)
            if m:
                end = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                break
    return start, end


def _summarize_folds(cv_output_dir: Path) -> Dict[str, Any]:
    folds_dir = cv_output_dir / "folds"
    fold_summaries: Dict[str, Dict[str, Any]] = {}
    for fold_dir in sorted(folds_dir.glob("fold_*")):
        log_path = fold_dir / "cv_run.log"
        if not log_path.exists():
            continue
        text = log_path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
        start, end = _parse_fold_timing(text)

        first_nonzero_gepa: Dict[str, float] = {}
        for m in GEPA_FINISH_PATTERN.finditer(text):
            comp = m.group(1)
            secs = float(m.group(2))
            if secs > 0 and comp not in first_nonzero_gepa:
                first_nonzero_gepa[comp] = secs

        fold_summaries[fold_dir.name] = {
            "has_final_stats": (fold_dir / "final_stats.json").exists(),
            "start_utc": start.isoformat().replace("+00:00", "Z") if start else None,
            "end_utc": end.isoformat().replace("+00:00", "Z") if end else None,
            "duration_sec": int((end - start).total_seconds()) if start and end else None,
            "warnings": {
                "neutral_fallbacks_total": len(NEUTRAL_PATTERN.findall(text)),
                "lm_timeouts_total": len(TIMEOUT_PATTERN.findall(text)),
                "lm_internal_errors_total": len(INTERNAL_PATTERN.findall(text)),
            },
            "gepa_first_nonzero_sec": {
                "scorer": first_nonzero_gepa.get("Scorer"),
                "leaf": first_nonzero_gepa.get("Leaf summarizer"),
                "merge": first_nonzero_gepa.get("Merge summarizer"),
                "total": sum(first_nonzero_gepa.values()) if first_nonzero_gepa else 0.0,
            },
        }
    return fold_summaries


def _estimate_from_telemetry(
    *,
    samples: List[Dict[str, Any]],
    current_task_conc: Optional[int],
    current_genrm_conc: Optional[int],
) -> Dict[str, Any]:
    util_vals = [float(s.get("gpu_util_mean_pct")) for s in samples if s.get("gpu_util_mean_pct") is not None]
    util_avg = (sum(util_vals) / len(util_vals)) if util_vals else None
    deltas = _compute_warning_deltas(samples)
    infra_delta = int(deltas["lm_timeouts"] + deltas["lm_internal_errors"])

    task = current_task_conc
    if task is not None:
        suggested = task
        if infra_delta >= 20:
            suggested = int(task * 0.85)
        elif infra_delta >= 8:
            suggested = int(task * 0.90)
        elif util_avg is not None and util_avg < 55 and infra_delta == 0:
            suggested = int(task * 1.20)
        elif util_avg is not None and util_avg < 70 and infra_delta <= 2:
            suggested = int(task * 1.10)
        task = _round_step(suggested, step=16, min_value=64, max_value=640)

    genrm = current_genrm_conc
    if genrm is not None:
        suggested = genrm
        if infra_delta >= 20:
            suggested = int(genrm * 0.85)
        elif infra_delta >= 8:
            suggested = int(genrm * 0.90)
        elif util_avg is not None and util_avg < 65 and infra_delta == 0:
            suggested = int(genrm * 1.20)
        genrm = _round_step(suggested, step=1, min_value=2, max_value=64)

    return {
        "mean_gpu_util_pct": util_avg,
        "warning_deltas": deltas,
        "suggested_task_concurrency_from_telemetry": task,
        "suggested_genrm_concurrency_from_telemetry": genrm,
    }


def _pick_task_recommendation(throughput: Dict[str, Optional[int]]) -> Optional[int]:
    vals = [throughput.get("task_dp2"), throughput.get("task_merge"), throughput.get("task_score")]
    vals = [int(v) for v in vals if isinstance(v, int) and v > 0]
    return min(vals) if vals else None


def _pick_genrm_recommendation(throughput: Dict[str, Optional[int]]) -> Optional[int]:
    vals = [
        throughput.get("genrm_batch_fast"),
        throughput.get("genrm_raw_fast"),
        throughput.get("genrm_batch_think"),
        throughput.get("genrm_raw_think"),
    ]
    vals = [int(v) for v in vals if isinstance(v, int) and v > 0]
    return min(vals) if vals else None


def _choose_suite_or_telemetry(
    *,
    current: Optional[int],
    suite_value: Optional[int],
    telemetry_value: Optional[int],
    min_floor: int,
    max_ceiling: int,
    max_relative_swing: float = 0.50,
) -> Optional[int]:
    if current is None:
        if suite_value is not None:
            return max(min_floor, min(max_ceiling, int(suite_value)))
        if telemetry_value is not None:
            return max(min_floor, min(max_ceiling, int(telemetry_value)))
        return None

    lower = max(min_floor, int(current * (1.0 - max_relative_swing)))
    upper = min(max_ceiling, int(current * (1.0 + max_relative_swing)))
    if suite_value is not None and lower <= int(suite_value) <= upper:
        return int(suite_value)
    if telemetry_value is not None:
        return max(min_floor, min(max_ceiling, int(telemetry_value)))
    if suite_value is not None:
        return max(min_floor, min(max_ceiling, int(suite_value)))
    return int(current)


def _human_hms(seconds: Optional[int]) -> Optional[str]:
    if seconds is None:
        return None
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    r = s % 60
    return f"{h:02d}:{m:02d}:{r:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CV telemetry JSONL and emit tuning recommendations.")
    parser.add_argument("--cv-output-dir", type=Path, required=True)
    parser.add_argument(
        "--telemetry-jsonl",
        type=Path,
        default=None,
        help="Path to telemetry JSONL (default: <cv-output-dir>/telemetry/cv_telemetry.jsonl).",
    )
    parser.add_argument(
        "--throughput-json",
        type=Path,
        default=None,
        help="Optional throughput-limit JSON from scripts/run_pipeline_throughput_limits.py.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output summary JSON path (default: <cv-output-dir>/telemetry/cv_telemetry_summary.json).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cv_output_dir = args.cv_output_dir.resolve()
    repo_root = Path(__file__).resolve().parent.parent

    telemetry_path = (
        args.telemetry_jsonl.resolve()
        if args.telemetry_jsonl is not None
        else (cv_output_dir / "telemetry" / "cv_telemetry.jsonl")
    )
    throughput_path = args.throughput_json.resolve() if args.throughput_json is not None else _find_latest_throughput_json(repo_root)
    out_json = (
        args.out_json.resolve()
        if args.out_json is not None
        else (cv_output_dir / "telemetry" / "cv_telemetry_summary.json")
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)

    samples = _load_jsonl(telemetry_path)
    sample_times = [_parse_iso(s.get("ts_utc")) for s in samples]
    sample_times = [t for t in sample_times if t is not None]

    latest_sample = samples[-1] if samples else {}
    current_cmd = latest_sample.get("active_command")
    current_task_conc = _extract_int_flag(current_cmd, "--concurrent-requests")
    current_genrm_conc = _extract_int_flag(current_cmd, "--genrm-max-concurrent")
    current_genrm_tree_conc = _extract_int_flag(current_cmd, "--genrm-tree-concurrency")

    throughput_recs = _load_throughput_recommendations(throughput_path)
    telemetry_estimate = _estimate_from_telemetry(
        samples=samples,
        current_task_conc=current_task_conc,
        current_genrm_conc=current_genrm_conc,
    )
    phase_agg = _aggregate_phases(samples)
    fold_summaries = _summarize_folds(cv_output_dir)

    task_from_suite = _pick_task_recommendation(throughput_recs)
    genrm_from_suite = _pick_genrm_recommendation(throughput_recs)
    telemetry_task = telemetry_estimate.get("suggested_task_concurrency_from_telemetry")
    telemetry_genrm = telemetry_estimate.get("suggested_genrm_concurrency_from_telemetry")
    recommended_task_conc = _choose_suite_or_telemetry(
        current=current_task_conc,
        suite_value=task_from_suite,
        telemetry_value=telemetry_task,
        min_floor=64,
        max_ceiling=640,
    )
    recommended_genrm_conc = _choose_suite_or_telemetry(
        current=current_genrm_conc,
        suite_value=genrm_from_suite,
        telemetry_value=telemetry_genrm,
        min_floor=2,
        max_ceiling=64,
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "cv_output_dir": str(cv_output_dir),
        "inputs": {
            "telemetry_jsonl": str(telemetry_path),
            "throughput_json": (str(throughput_path) if throughput_path is not None else None),
        },
        "sample_window": {
            "n_samples": len(samples),
            "first_ts_utc": sample_times[0].isoformat().replace("+00:00", "Z") if sample_times else None,
            "last_ts_utc": sample_times[-1].isoformat().replace("+00:00", "Z") if sample_times else None,
        },
        "current_flags": {
            "concurrent_requests": current_task_conc,
            "genrm_max_concurrent": current_genrm_conc,
            "genrm_tree_concurrency": current_genrm_tree_conc,
        },
        "phase_aggregates": phase_agg,
        "telemetry_estimate": telemetry_estimate,
        "throughput_suite_recommendations": throughput_recs,
        "recommendations": {
            "concurrent_requests": recommended_task_conc,
            "genrm_max_concurrent": recommended_genrm_conc,
            "genrm_tree_concurrency": recommended_genrm_conc,
        },
        "folds": fold_summaries,
    }

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Telemetry samples: {len(samples)}")
    print(f"Summary JSON: {out_json}")
    if sample_times:
        print(
            "Window:",
            sample_times[0].isoformat().replace("+00:00", "Z"),
            "->",
            sample_times[-1].isoformat().replace("+00:00", "Z"),
        )
    if throughput_path is not None:
        print(f"Throughput suite JSON: {throughput_path}")
    print(
        "Recommended flags:",
        f"--concurrent-requests {payload['recommendations']['concurrent_requests']}",
        f"--genrm-max-concurrent {payload['recommendations']['genrm_max_concurrent']}",
        f"--genrm-tree-concurrency {payload['recommendations']['genrm_tree_concurrency']}",
    )

    for fold_name, info in sorted(fold_summaries.items()):
        dur = _human_hms(info.get("duration_sec"))
        gepa_total = info.get("gepa_first_nonzero_sec", {}).get("total")
        print(
            f"{fold_name}: done={info.get('has_final_stats')} duration={dur} "
            f"gepa_nonzero={gepa_total:.1f}s neutral_fallbacks={info.get('warnings', {}).get('neutral_fallbacks_total')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
