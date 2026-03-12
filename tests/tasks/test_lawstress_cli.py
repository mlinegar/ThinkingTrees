from __future__ import annotations

import importlib
from pathlib import Path


def _fake_chat(self, *, system: str, user: str, temperature: float, max_tokens: int) -> str:  # noqa: ARG001
    if "numeric RILE score" in system:
        # Deterministic pseudo-score from prompt text.
        value = (sum(ord(ch) for ch in user) % 201) - 100
        return str(value)
    if "Merged Summary" in user:
        return "Merged summary preserving directional stance."
    return "Summary preserving key ideological commitments."


def test_generate_and_eval_cli_smoke(tmp_path: Path, monkeypatch) -> None:
    gen_cli = importlib.import_module("scripts.generate_manifesto_lawstress")
    eval_cli = importlib.import_module("scripts.eval_manifesto_lawstress")

    gen_out = tmp_path / "gen"
    rc = gen_cli.main(
        [
            "--output-dir",
            str(gen_out),
            "--train-size",
            "6",
            "--val-size",
            "3",
            "--test-size",
            "3",
            "--seed",
            "123",
            "--real-anchor-ratio",
            "0.0",
            "--disable-teacher-gates",
            "--disable-teacher-rewrite",
            "--disable-reference-summary",
            "--skip-counterexample-pairs",
        ]
    )
    assert rc == 0

    records_path = gen_out / "lawstress_records.jsonl"
    assert records_path.exists()
    assert (gen_out / "benchmark_docs.jsonl").exists()
    assert (gen_out / "reference_summaries.jsonl").exists()

    monkeypatch.setattr(eval_cli.OpenAIChatClient, "chat", _fake_chat)
    monkeypatch.setattr(
        eval_cli,
        "_build_genrm_judge",
        lambda **kwargs: (
            lambda context, original_text, summary_a, summary_b, law_type: {
                "preferred": "A",
                "confidence": 0.8,
            }
        ),
    )

    eval_out = tmp_path / "eval"
    predictions_path = eval_out / "predictions.jsonl"

    rc = eval_cli.main(
        [
            "--records",
            str(records_path),
            "--output-dir",
            str(eval_out),
            "--mode",
            "summarize_only",
            "--predictions-path",
            str(predictions_path),
        ]
    )
    assert rc == 0
    assert predictions_path.exists()

    rc = eval_cli.main(
        [
            "--records",
            str(records_path),
            "--output-dir",
            str(eval_out),
            "--mode",
            "score_and_judge_only",
            "--predictions-path",
            str(predictions_path),
        ]
    )
    assert rc == 0

    assert (eval_out / "eval_results.jsonl").exists()
    assert (eval_out / "eval_metrics.json").exists()
    assert (eval_out / "eval_by_group.json").exists()
    assert (eval_out / "eval_report.md").exists()
