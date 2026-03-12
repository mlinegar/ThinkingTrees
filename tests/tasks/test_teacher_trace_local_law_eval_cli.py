from __future__ import annotations

import importlib
import json
from pathlib import Path

from src.tasks.manifesto.teacher_trace_generator import (
    TeacherTraceRecord,
    write_teacher_trace_records_jsonl,
)


def _record(example_id: str, split: str, rile: float) -> TeacherTraceRecord:
    return TeacherTraceRecord(
        example_id=example_id,
        split=split,
        source_manifesto_id=f"src_{example_id}",
        source_party_abbrev="P",
        source_country_name="Country",
        source_year=2000,
        source_rile_raw=rile,
        source_bin_name="center",
        source_text=f"source::{example_id}",
        expanded_text=(f"expanded::{example_id}\n\n" + ("policy text " * 50)).strip(),
        expanded_score_raw=rile,
        expanded_delta_raw=0.0,
        summary1=f"summary1::{example_id}",
        summary1_score_raw=rile,
        summary1_delta_raw=0.0,
        summary2=f"summary2::{example_id}",
        summary2_score_raw=rile,
        summary2_delta_raw=0.0,
        summary2_vs_summary1_delta_raw=0.0,
        same_side_summary1=True,
        same_side_summary2=True,
        trace_critical_points=[],
        trace_entities=[],
        trace_qualifiers=[],
        trace_invariants=[],
        trace_notes="",
        attempts_used=1,
    )


def test_teacher_trace_local_law_eval_cli_smoke(tmp_path: Path, monkeypatch) -> None:
    cli = importlib.import_module("scripts.eval_manifesto_teacher_trace_local_laws")

    def fake_build_summarize_fn(  # noqa: ARG001
        client,
        *,
        temperature: float,
        max_tokens: int,
        include_score_conditioning: bool,
    ):
        def _summarize(text: str, rubric: str, source_rile_raw: float, hop: int) -> str:  # noqa: ARG001
            return f"summary_h{hop}::{text[:24]}"

        return _summarize

    def fake_build_merge_fn(  # noqa: ARG001
        client,
        *,
        temperature: float,
        max_tokens: int,
        include_score_conditioning: bool,
    ):
        def _merge(left: str, right: str, rubric: str, source_rile_raw: float) -> str:  # noqa: ARG001
            return f"merge::{left[:12]}::{right[:12]}"

        return _merge

    def fake_build_score_fn(client, *, temperature: float, max_tokens: int):  # noqa: ARG001
        def _score(text: str) -> float:
            if text.startswith("summary_h1"):
                return 5.0
            if text.startswith("summary_h2"):
                return 4.5
            if text.startswith("merge::"):
                return 5.0
            return 5.0

        return _score

    monkeypatch.setattr(cli, "_build_summarize_fn", fake_build_summarize_fn)
    monkeypatch.setattr(cli, "_build_merge_fn", fake_build_merge_fn)
    monkeypatch.setattr(cli, "_build_score_fn", fake_build_score_fn)

    records_path = tmp_path / "teacher_trace_records.jsonl"
    records = [
        _record("doc1", "test", 5.0),
        _record("doc2", "test", 5.0),
    ]
    write_teacher_trace_records_jsonl(records_path, records)

    output_dir = tmp_path / "eval"
    rc = cli.main(
        [
            "--records",
            str(records_path),
            "--output-dir",
            str(output_dir),
            "--mode",
            "full",
            "--splits",
            "test",
        ]
    )
    assert rc == 0

    assert (output_dir / "predictions.jsonl").exists()
    assert (output_dir / "eval_results.jsonl").exists()
    assert (output_dir / "eval_metrics.json").exists()
    assert (output_dir / "eval_report.md").exists()

    metrics = json.loads((output_dir / "eval_metrics.json").read_text(encoding="utf-8"))
    assert metrics["overall"]["n"] == 2


def test_teacher_trace_local_law_eval_cli_with_dspy_module(tmp_path: Path, monkeypatch) -> None:
    cli = importlib.import_module("scripts.eval_manifesto_teacher_trace_local_laws")

    class _FakeUnifiedG:
        def load(self, path: str) -> None:  # noqa: ARG002
            return None

        def __call__(self, *, content: str, rubric: str) -> str:  # noqa: ARG002
            return f"dspy::{content[:20]}"

    monkeypatch.setattr(cli, "create_vllm_lm", lambda **kwargs: object())
    monkeypatch.setattr(cli, "configure_dspy", lambda **kwargs: None)
    monkeypatch.setattr(cli, "UnifiedG", _FakeUnifiedG)
    monkeypatch.setattr(cli, "_build_score_fn", lambda *args, **kwargs: (lambda text: 5.0))

    records_path = tmp_path / "teacher_trace_records.jsonl"
    records = [
        _record("doc1", "test", 5.0),
    ]
    write_teacher_trace_records_jsonl(records_path, records)

    module_path = tmp_path / "unified_g_final.json"
    module_path.write_text("{}", encoding="utf-8")

    output_dir = tmp_path / "eval_dspy"
    rc = cli.main(
        [
            "--records",
            str(records_path),
            "--output-dir",
            str(output_dir),
            "--mode",
            "full",
            "--splits",
            "test",
            "--dspy-module",
            str(module_path),
        ]
    )
    assert rc == 0
    metrics = json.loads((output_dir / "eval_metrics.json").read_text(encoding="utf-8"))
    assert metrics["overall"]["n"] == 1
