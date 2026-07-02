#!/usr/bin/env python3
"""Run a small local-law tuning sweep on one manifesto anchor."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence


from src.tasks.manifesto.openai_chat import OpenAIChatClient
from src.tasks.manifesto.teacher_trace_generator import (
    TeacherTraceRecord,
    load_teacher_trace_records_jsonl,
    strict_same_side_raw,
)


DEFAULT_MODEL = "/mnt/data/models/nvidia/Qwen3.5-397B-A17B-NVFP4"
_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    use_dspy_guidance: bool
    score_tolerance_raw: float
    max_attempts: int
    dspy_guidance_temperature: float
    dspy_guidance_max_tokens: int
    summary_temperature: float


@dataclass
class CandidateResult:
    config: CandidateConfig
    run_dir: str
    run_log: str
    return_code: int
    accepted: bool
    c1_pass: bool
    c2_pass: bool
    c3_pass: bool
    c1_abs_delta_raw: float
    c2_abs_drift_raw: float
    c3_abs_delta_raw: float
    source_manifesto_id: str
    objective: float
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "run_dir": self.run_dir,
            "run_log": self.run_log,
            "return_code": self.return_code,
            "accepted": self.accepted,
            "c1_pass": self.c1_pass,
            "c2_pass": self.c2_pass,
            "c3_pass": self.c3_pass,
            "c1_abs_delta_raw": self.c1_abs_delta_raw,
            "c2_abs_drift_raw": self.c2_abs_drift_raw,
            "c3_abs_delta_raw": self.c3_abs_delta_raw,
            "source_manifesto_id": self.source_manifesto_id,
            "objective": self.objective,
            "details": self.details,
        }


def _parse_score(text: str) -> Optional[float]:
    rendered = str(text or "").strip()
    if not rendered:
        return None
    matches = _NUMERIC_RE.findall(rendered)
    values: List[float] = []
    for token in matches:
        try:
            values.append(float(token))
        except Exception:
            continue
    in_range = [value for value in values if -100.0 <= value <= 100.0]
    if not in_range:
        return None
    non_boundary = [value for value in in_range if abs(value) < 99.999]
    return float(non_boundary[0] if non_boundary else in_range[0])


def _parse_last_number(text: str) -> Optional[float]:
    rendered = str(text or "").strip()
    if not rendered:
        return None
    matches = _NUMERIC_RE.findall(rendered)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except (TypeError, ValueError):
        return None


def _build_score_fn(client: OpenAIChatClient, *, max_tokens: int) -> Any:
    def _score(text: str) -> float:
        response = client.chat(
            system=(
                "You are a strict directional coder for information extraction. "
                "Return exactly one numeric RILE score in [-100, 100]."
            ),
            user=(
                "Score this text on a RILE-style directional scale. "
                "Return only one number.\n\n"
                f"TEXT:\n{text}"
            ),
            temperature=0.0,
            max_tokens=max_tokens,
        )
        parsed = _parse_score(response)
        retry = None
        if parsed is None:
            retry = client.chat(
                system=(
                    "Output exactly one numeric RILE score in [-100,100]. "
                    "No words, no explanation, no JSON."
                ),
                user=(
                    "Extract and return only the numeric RILE score.\n"
                    "Output format example: -12.50\n\n"
                    f"TEXT:\n{text}"
                ),
                temperature=0.0,
                max_tokens=max(8, int(max_tokens)),
            )
            parsed = _parse_score(retry)
        if parsed is None:
            salvage = _parse_last_number(f"{response}\n{retry or ''}")
            if salvage is not None:
                parsed = max(-100.0, min(100.0, float(salvage)))
        if parsed is None:
            raise ValueError(f"Could not parse score responses: first={response!r} retry={retry!r}")
        return float(parsed)

    return _score


def _split_segments(text: str) -> tuple[str, str]:
    rendered = str(text or "").strip()
    if not rendered:
        return "", ""
    if len(rendered) < 300:
        mid = len(rendered) // 2
        return rendered[:mid].strip(), rendered[mid:].strip()

    mid = len(rendered) // 2
    left = rendered.rfind("\n\n", 0, mid)
    right = rendered.find("\n\n", mid)
    candidates = [pos for pos in (left, right) if pos >= 0]
    if not candidates:
        cut = mid
    else:
        cut = min(candidates, key=lambda pos: abs(pos - mid))
    a = rendered[:cut].strip()
    b = rendered[cut:].strip()
    if not a or not b:
        cut = mid
        a = rendered[:cut].strip()
        b = rendered[cut:].strip()
    return a, b


def _summarize_text(
    *,
    client: OpenAIChatClient,
    text: str,
    source_rile_raw: float,
    hop: int,
    temperature: float,
    max_tokens: int,
) -> str:
    return client.chat(
        system=(
            "Summarize for information extraction while preserving directional stance, "
            "factual commitments, and qualifying caveats."
        ),
        user=(
            f"Target directional score to preserve: {source_rile_raw:.2f}\n"
            f"Resummary hop: {hop}\n"
            "Return only summary text.\n\n"
            f"TEXT:\n{text}"
        ),
        temperature=temperature,
        max_tokens=max_tokens,
    ).strip()


def _merge_summaries(
    *,
    client: OpenAIChatClient,
    summary_a: str,
    summary_b: str,
    source_rile_raw: float,
    temperature: float,
    max_tokens: int,
) -> str:
    return client.chat(
        system=(
            "Merge two summaries into one concise, faithful summary for information extraction. "
            "Preserve directional stance, core commitments, entities, and caveats."
        ),
        user=(
            f"Target directional score to preserve: {source_rile_raw:.2f}\n\n"
            f"SUMMARY_A:\n{summary_a}\n\n"
            f"SUMMARY_B:\n{summary_b}\n"
        ),
        temperature=temperature,
        max_tokens=max_tokens,
    ).strip()


def _eval_c3(
    *,
    record: TeacherTraceRecord,
    teacher_client: OpenAIChatClient,
    score_fn: Any,
    summary_temperature: float,
    summary_max_tokens: int,
    c3_threshold_raw: float,
) -> Dict[str, Any]:
    seg_a, seg_b = _split_segments(record.expanded_text)
    if not seg_a or not seg_b:
        return {
            "c3_pass": False,
            "c3_abs_delta_raw": 999.0,
            "error": "Could not split expanded text into non-empty segments",
        }
    y_a = float(score_fn(seg_a))
    y_b = float(score_fn(seg_b))
    sum_a = _summarize_text(
        client=teacher_client,
        text=seg_a,
        source_rile_raw=record.source_rile_raw,
        hop=1,
        temperature=summary_temperature,
        max_tokens=summary_max_tokens,
    )
    sum_b = _summarize_text(
        client=teacher_client,
        text=seg_b,
        source_rile_raw=record.source_rile_raw,
        hop=1,
        temperature=summary_temperature,
        max_tokens=summary_max_tokens,
    )
    merged = _merge_summaries(
        client=teacher_client,
        summary_a=sum_a,
        summary_b=sum_b,
        source_rile_raw=record.source_rile_raw,
        temperature=summary_temperature,
        max_tokens=summary_max_tokens,
    )
    y_merge = float(score_fn(merged))
    expected = (
        (len(seg_a) * y_a + len(seg_b) * y_b)
        / max(1, (len(seg_a) + len(seg_b)))
    )
    abs_delta = abs(float(y_merge - expected))
    return {
        "c3_pass": abs_delta <= float(c3_threshold_raw),
        "c3_abs_delta_raw": abs_delta,
        "y_a_raw": y_a,
        "y_b_raw": y_b,
        "y_merge_raw": y_merge,
        "y_merge_expected_raw": expected,
    }


def _default_candidates() -> List[CandidateConfig]:
    return [
        CandidateConfig(
            name="no_dspy_law_focus",
            use_dspy_guidance=False,
            score_tolerance_raw=20.0,
            max_attempts=4,
            dspy_guidance_temperature=0.1,
            dspy_guidance_max_tokens=1200,
            summary_temperature=0.08,
        ),
        CandidateConfig(
            name="dspy_law_focus",
            use_dspy_guidance=True,
            score_tolerance_raw=20.0,
            max_attempts=3,
            dspy_guidance_temperature=0.1,
            dspy_guidance_max_tokens=1600,
            summary_temperature=0.08,
        ),
    ]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-doc local-law tuning sweep.")
    parser.add_argument("--manifesto-id", type=str, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--teacher-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--scorer-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--teacher-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--scorer-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--teacher-timeout-seconds", type=float, default=420.0)
    parser.add_argument("--scorer-timeout-seconds", type=float, default=420.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-source-chars", type=int, default=1200)
    parser.add_argument("--max-source-chars", type=int, default=0)
    parser.add_argument("--expand-max-tokens", type=int, default=1200)
    parser.add_argument("--summary-max-tokens", type=int, default=320)
    parser.add_argument("--trace-max-tokens", type=int, default=480)
    parser.add_argument("--score-max-tokens", type=int, default=120)
    parser.add_argument("--c1-threshold-raw", type=float, default=10.0)
    parser.add_argument("--c2-threshold-raw", type=float, default=6.0)
    parser.add_argument("--c3-threshold-raw", type=float, default=8.0)
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--no-enable-thinking", action="store_true", default=True)
    parser.add_argument("--dspy-guidance-source-max-chars", type=int, default=262144)
    parser.add_argument("--dspy-guidance-expansion-max-chars", type=int, default=262144)
    parser.add_argument("--previous-expansion-max-chars", type=int, default=262144)
    parser.add_argument("--revision-guidance-max-chars", type=int, default=65536)
    parser.add_argument("--trace-source-max-chars", type=int, default=262144)
    parser.add_argument("--trace-expanded-max-chars", type=int, default=262144)
    return parser.parse_args(argv)


def _render_report(path: Path, best: Optional[CandidateResult], results: Sequence[CandidateResult]) -> None:
    lines: List[str] = []
    lines.append("# Single-Doc Local-Law Tuning Report")
    lines.append("")
    lines.append(f"Generated at: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    if best is not None:
        lines.append("## Best Candidate")
        lines.append("")
        lines.append(f"- name: {best.config.name}")
        lines.append(f"- objective: {best.objective:.3f}")
        lines.append(f"- accepted: {best.accepted}")
        lines.append(f"- C1/C2/C3: {best.c1_pass} / {best.c2_pass} / {best.c3_pass}")
        lines.append(f"- c1_abs_delta_raw: {best.c1_abs_delta_raw:.3f}")
        lines.append(f"- c2_abs_drift_raw: {best.c2_abs_drift_raw:.3f}")
        lines.append(f"- c3_abs_delta_raw: {best.c3_abs_delta_raw:.3f}")
        lines.append(f"- run_dir: {best.run_dir}")
        lines.append("")
    lines.append("## All Candidates")
    lines.append("")
    for row in results:
        lines.append(
            "- "
            f"name={row.config.name} accepted={row.accepted} "
            f"c1={row.c1_pass} c2={row.c2_pass} c3={row.c3_pass} "
            f"c1_abs={row.c1_abs_delta_raw:.3f} "
            f"c2_abs={row.c2_abs_drift_raw:.3f} "
            f"c3_abs={row.c3_abs_delta_raw:.3f} "
            f"objective={row.objective:.3f} "
            f"return_code={row.return_code}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or (Path("outputs") / f"single_local_law_tune_{stamp}")
    output_root.mkdir(parents=True, exist_ok=True)
    runs_dir = output_root / "candidates"
    runs_dir.mkdir(parents=True, exist_ok=True)

    teacher_client = OpenAIChatClient(
        base_url=args.teacher_base_url,
        model=args.teacher_model,
        timeout_seconds=float(args.teacher_timeout_seconds),
        enable_thinking=not bool(args.no_enable_thinking),
    )
    scorer_client = OpenAIChatClient(
        base_url=args.scorer_base_url,
        model=args.scorer_model,
        timeout_seconds=float(args.scorer_timeout_seconds),
        enable_thinking=not bool(args.no_enable_thinking),
    )
    score_fn = _build_score_fn(scorer_client, max_tokens=int(args.score_max_tokens))

    candidates = _default_candidates()[: max(1, int(args.max_candidates))]
    results: List[CandidateResult] = []

    for idx, config in enumerate(candidates):
        run_dir = runs_dir / f"{idx + 1:02d}_{config.name}"
        run_log = run_dir / "run.log"
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd: List[str] = [
            sys.executable,
            "scripts/generate_manifesto_teacher_traces.py",
            "--output-dir",
            str(run_dir),
            "--train-size",
            "1",
            "--val-size",
            "0",
            "--test-size",
            "0",
            "--seed",
            str(int(args.seed) + idx),
            "--manifesto-ids",
            str(args.manifesto_id),
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
            "--summary-temperature",
            str(config.summary_temperature),
            "--score-tolerance-raw",
            str(config.score_tolerance_raw),
            "--max-attempts",
            str(config.max_attempts),
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
            "--no-allow-source-shrink-on-error",
        ]
        if bool(config.use_dspy_guidance):
            cmd.append("--use-dspy-guidance")
        else:
            cmd.append("--no-use-dspy-guidance")
        if bool(args.no_enable_thinking):
            cmd.append("--no-enable-thinking")

        with run_log.open("w", encoding="utf-8") as log:
            log.write("# cmd=" + " ".join(cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(Path(__file__).resolve().parents[1]),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )

        records_path = run_dir / "teacher_trace_records.jsonl"
        records: List[TeacherTraceRecord] = []
        if records_path.exists():
            records = load_teacher_trace_records_jsonl(records_path)

        if proc.returncode != 0 or not records:
            result = CandidateResult(
                config=config,
                run_dir=str(run_dir),
                run_log=str(run_log),
                return_code=int(proc.returncode),
                accepted=False,
                c1_pass=False,
                c2_pass=False,
                c3_pass=False,
                c1_abs_delta_raw=999.0,
                c2_abs_drift_raw=999.0,
                c3_abs_delta_raw=999.0,
                source_manifesto_id=str(args.manifesto_id),
                objective=-1e9,
                details={"error": "candidate did not yield accepted records"},
            )
            results.append(result)
            continue

        record = records[0]
        c1_abs = abs(float(record.summary1_delta_raw))
        c2_abs = abs(float(record.summary2_vs_summary1_delta_raw))
        c1_pass = bool(c1_abs <= float(args.c1_threshold_raw))
        c2_pass = bool(
            c2_abs <= float(args.c2_threshold_raw)
            and strict_same_side_raw(record.summary2_score_raw, record.source_rile_raw)
        )
        c3_details = _eval_c3(
            record=record,
            teacher_client=teacher_client,
            score_fn=score_fn,
            summary_temperature=float(config.summary_temperature),
            summary_max_tokens=int(args.summary_max_tokens),
            c3_threshold_raw=float(args.c3_threshold_raw),
        )
        c3_abs = float(c3_details.get("c3_abs_delta_raw", 999.0))
        c3_pass = bool(c3_details.get("c3_pass", False))
        objective = (
            (100.0 if c1_pass else 0.0)
            + (100.0 if c2_pass else 0.0)
            + (100.0 if c3_pass else 0.0)
            - c1_abs
            - c2_abs
            - c3_abs
        )

        result = CandidateResult(
            config=config,
            run_dir=str(run_dir),
            run_log=str(run_log),
            return_code=int(proc.returncode),
            accepted=True,
            c1_pass=c1_pass,
            c2_pass=c2_pass,
            c3_pass=c3_pass,
            c1_abs_delta_raw=c1_abs,
            c2_abs_drift_raw=c2_abs,
            c3_abs_delta_raw=c3_abs,
            source_manifesto_id=record.source_manifesto_id,
            objective=float(objective),
            details={
                "record_example_id": record.example_id,
                "source_rile_raw": record.source_rile_raw,
                "expanded_score_raw": record.expanded_score_raw,
                "summary1_score_raw": record.summary1_score_raw,
                "summary2_score_raw": record.summary2_score_raw,
                "c3": c3_details,
            },
        )
        results.append(result)

    best = None
    accepted_rows = [row for row in results if row.accepted]
    if accepted_rows:
        best = max(accepted_rows, key=lambda row: row.objective)
    elif results:
        best = max(results, key=lambda row: row.objective)

    results_path = output_root / "candidate_results.json"
    results_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "manifesto_id": str(args.manifesto_id),
                "best": None if best is None else best.to_dict(),
                "results": [row.to_dict() for row in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path = output_root / "candidate_report.md"
    _render_report(report_path, best, results)
    print(str(output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
