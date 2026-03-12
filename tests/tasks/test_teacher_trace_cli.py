from __future__ import annotations

import importlib
import json
from pathlib import Path

from src.tasks.manifesto.data_loader import ManifestoSample


class _FakeDataset:
    def __init__(self) -> None:
        self._samples = {
            f"m{i}": ManifestoSample(
                manifesto_id=f"m{i}",
                party_id=1,
                party_name="Party",
                party_abbrev="P",
                country_code=1,
                country_name="Country",
                election_date="2000-01-01",
                date_code=200001,
                text=("Policy text. " * 80).strip(),
                rile=0.0,
                vote_share=None,
                party_family=None,
            )
            for i in range(1, 7)
        }

    def get_all_ids(self):
        return list(self._samples.keys())

    def get_sample(self, manifesto_id: str):
        return self._samples.get(manifesto_id)


def _fake_chat(self, *, system: str, user: str, temperature: float, max_tokens: int) -> str:  # noqa: ARG001
    if "numeric RILE score" in system:
        return "0"
    if "Return a JSON object with keys" in user:
        return json.dumps(
            {
                "critical_points": ["cp1", "cp2"],
                "entities": ["entity1"],
                "qualifiers": ["qualifier1"],
                "invariants": ["keep direction"],
                "notes": "trace ok",
            }
        )
    if "Resummary hop: 2" in user:
        return "Second-hop summary."
    if "Resummary hop: 1" in user:
        return "First-hop summary."
    return "Expanded synthetic policy document."


def test_teacher_trace_cli_smoke(tmp_path: Path, monkeypatch) -> None:
    cli = importlib.import_module("scripts.generate_manifesto_teacher_traces")
    monkeypatch.setattr(cli, "ManifestoDataset", lambda require_text=True: _FakeDataset())  # noqa: ARG005
    monkeypatch.setattr(cli.OpenAIChatClient, "chat", _fake_chat)

    output_dir = tmp_path / "teacher_trace"
    rc = cli.main(
        [
            "--output-dir",
            str(output_dir),
            "--train-size",
            "2",
            "--val-size",
            "1",
            "--test-size",
            "1",
            "--min-source-chars",
            "10",
            "--max-attempts",
            "1",
        ]
    )
    assert rc == 0

    assert (output_dir / "teacher_trace_records.jsonl").exists()
    assert (output_dir / "benchmark_docs.jsonl").exists()
    assert (output_dir / "summary_training_pairs.jsonl").exists()
    assert (output_dir / "trace_artifacts.jsonl").exists()
    assert (output_dir / "manifest.json").exists()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["accepted_docs"] == 4
    assert manifest["rejected_docs"] == 0

