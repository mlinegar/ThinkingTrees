from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def test_eval_manifesto_lawstress_rejects_genrm_mode(tmp_path: Path) -> None:
    cli = importlib.import_module("scripts.eval_manifesto_lawstress")
    output_dir = tmp_path / "out"
    records_path = tmp_path / "dummy_records.jsonl"
    with pytest.raises(ValueError, match="no GenRM"):
        cli.main(
            [
                "--records",
                str(records_path),
                "--output-dir",
                str(output_dir),
                "--no-disable-genrm",
            ]
        )
