from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.core.strategy import tournament_doc_id
from src.runtime import longbench_batched_example as longbench_example


def _write_fixture(path: Path) -> None:
    rows = [
        {
            "_id": "batched-1",
            "domain": "law",
            "sub_domain": "contracts",
            "difficulty": "easy",
            "length": "short",
            "question": "Which option names the delivery party?",
            "choice_A": "Alpha",
            "choice_B": "Beta",
            "choice_C": "Gamma",
            "choice_D": "Delta",
            "answer": "C",
            "context": "Gamma is named as the delivery party in the final clause.",
        },
        {
            "_id": "batched-2",
            "domain": "finance",
            "sub_domain": "filings",
            "difficulty": "hard",
            "length": "medium",
            "question": "What happened to reserves?",
            "choice_A": "They rose",
            "choice_B": "They fell",
            "choice_C": "They were unchanged",
            "choice_D": "They were omitted",
            "answer": "B",
            "context": "The filing says reserves fell after the audit.",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_longbench_batched_prompt_helpers_use_doc_metadata() -> None:
    assert longbench_example.parse_choice_score("Answer: C because the clause says so.") == 2.0
    assert longbench_example.score_to_choice(1.0) == "B"

    longbench_example._PROMPT_METADATA_BY_DOC_ID["doc-1"] = {
        "question": "Which option names the delivery party?",
        "choices": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
    }
    token = tournament_doc_id.set("doc-1")
    try:
        messages = longbench_example.build_longbench_summarize_prompt("Gamma appears here.", "rubric")
    finally:
        tournament_doc_id.reset(token)

    rendered = "\n".join(msg["content"] for msg in messages)
    assert "Which option names the delivery party?" in rendered
    assert "C. Gamma" in rendered
    assert "Do not answer yet" in rendered


def test_longbench_batched_cli_dry_run_fixture(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixture = tmp_path / "longbench.jsonl"
    output = tmp_path / "longbench_batched.json"
    _write_fixture(fixture)

    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_longbench_batched_example.py",
            "--dataset-path",
            str(fixture),
            "--limit",
            "2",
            "--dry-run",
            "--output",
            str(output),
            "--chunk-size",
            "32",
        ],
        cwd=repo_root,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["mode"] == "dry_run"
    assert payload["method_ref"]["method_id"] == "longbench_batched_tree"
    assert payload["method_ref"]["adapter"] == "batched_doc_pipeline"
    assert len(payload["chunk_analysis"]) == 2
    assert payload["chunk_analysis"][0]["_id"] == "batched-1"
    assert payload["chunk_analysis"][0]["chunk_count"] >= 1
    assert payload["results"] == []
    assert payload["config"]["runtime_mode"] == "unified_v2"
    sidecar_root = output.parent / f"{output.stem}_experiment"
    assert (sidecar_root / "experiment_manifest.json").exists()
    assert (sidecar_root / "artifacts.json").exists()
    assert (sidecar_root / "results.jsonl").exists()
