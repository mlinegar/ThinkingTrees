from __future__ import annotations

import json
from pathlib import Path

from src.runtime.adapters.longbench import LongBenchV2Adapter, LongBenchV2Spec


def _write_fixture(path: Path) -> None:
    rows = [
        {
            "_id": "lb-1",
            "domain": "law",
            "sub_domain": "contracts",
            "difficulty": "easy",
            "length": "short",
            "question": "Which party must deliver the report?",
            "choice_A": "Alpha LLC",
            "choice_B": "Beta Inc",
            "choice_C": "Gamma Co",
            "choice_D": "Delta PLC",
            "answer": "C",
            "context": "The agreement says Gamma Co must deliver the report by Friday.",
        },
        {
            "_id": "lb-2",
            "domain": "finance",
            "sub_domain": "filings",
            "difficulty": "hard",
            "length": "long",
            "question": "What was the stated reserve change?",
            "choice_A": "Up",
            "choice_B": "Down",
            "choice_C": "Unchanged",
            "choice_D": "Not disclosed",
            "answer": "B",
            "context": "The filing says the reserve went down after the audit.",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _adapter(path: Path, *, task_id: str = "all") -> LongBenchV2Adapter:
    return LongBenchV2Adapter(
        spec=LongBenchV2Spec(
            task_id=task_id,
            split="test",
            max_seq_length=8192,
            num_samples=2,
            seed=0,
        ),
        dataset_path=path,
    )


def test_longbench_v2_loads_local_rows_and_builds_task_view(tmp_path: Path) -> None:
    fixture = tmp_path / "longbench.jsonl"
    _write_fixture(fixture)

    problems = list(_adapter(fixture).load_split("test"))

    assert [p.problem_id for p in problems] == ["longbench_v2:lb-1", "longbench_v2:lb-2"]
    assert problems[0].references == ["C"]
    assert problems[0].metadata["domain"] == "law"
    assert "Context:" in problems[0].input_text
    assert "Return only one letter" in problems[0].input_text

    view = _adapter(fixture).task_view(problems[0])
    assert view.context == "The agreement says Gamma Co must deliver the report by Friday."
    assert view.question == "Which party must deliver the report?"
    assert view.choices["C"] == "Gamma Co"
    assert view.answer_instruction == "Return only one letter: A, B, C, or D."
    assert view.official_prompt == problems[0].input_text


def test_longbench_v2_filters_parses_and_scores_grouped_metrics(tmp_path: Path) -> None:
    fixture = tmp_path / "longbench.jsonl"
    _write_fixture(fixture)

    adapter = _adapter(fixture, task_id="law")
    problems = list(adapter.load_split("test"))

    assert len(problems) == 1
    problem = problems[0]
    assert adapter.parse_prediction(problem, "The answer is C.") == "C"
    assert adapter.parse_prediction(problem, "Answer: C because Gamma is named.") == "C"

    metrics = adapter.score(problem, {"prediction": "Answer: C"})
    assert metrics["longbench_v2_accuracy"] == 1.0
    assert metrics["exact_match"] == 1.0
    assert metrics["longbench_v2_accuracy_domain_law"] == 1.0
    assert metrics["longbench_v2_accuracy_difficulty_easy"] == 1.0
    assert metrics["longbench_v2_accuracy_length_short"] == 1.0

    miss = adapter.score(problem, {"prediction": "A"})
    assert miss["longbench_v2_accuracy"] == 0.0

