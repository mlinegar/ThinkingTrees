from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_manifesto_fg_baselines as baselines
from src.tasks.manifesto.data_loader import ManifestoDataset
from src.tasks.manifesto.dimensions import PolicyDimension


def test_safe_party_mask_skips_one_letter_aliases_inside_words() -> None:
    sample = SimpleNamespace(party_name="Liberal Party", party_abbrev="V")
    text = (
        "V supports environmental investment over growth. "
        "The Liberal Party's plan names the Liberal Party directly."
    )

    masked = baselines._mask_party_names(text, sample, mode="safe_boundary")

    assert "environmental" in masked
    assert "investment" in masked
    assert "over" in masked
    assert "V supports" in masked
    assert "<PARTY>'s plan" in masked
    assert "names the <PARTY> directly" in masked


def test_legacy_party_mask_preserves_old_unrestricted_substitution() -> None:
    sample = SimpleNamespace(party_name="Liberal Party", party_abbrev="V")
    text = "V supports environmental investment over growth."

    masked = baselines._mask_party_names(text, sample, mode="legacy")

    assert masked.startswith("<PARTY> supports")
    assert "en<PARTY>ironmental" in masked
    assert "in<PARTY>estment" in masked
    assert "o<PARTY>er" in masked


def test_safe_party_mask_skips_invalid_aliases_and_uses_boundaries() -> None:
    sample = SimpleNamespace(party_name=float("nan"), party_abbrev="EL")
    text = "Green finance for EL without fuel corruption."

    masked = baselines._mask_party_names(text, sample, mode="safe_boundary")

    assert "finance" in masked
    assert "fuel" in masked
    assert "for <PARTY> without" in masked
    assert "fi<PARTY>ce" not in masked
    assert "fu<PARTY>" not in masked


def test_f0g_benoit_combo_is_raw_prompt_on_benoit_summaries() -> None:
    spec = baselines.COMBO_SPECS["f0g_benoit"]

    assert spec["f_kind"] == "raw_benoit_prompt"
    assert spec["g_kind"] == "benoit_masked_summary"


def test_environment_benoit_rows_load_on_current_split() -> None:
    split_path = Path(
        "outputs/manifesto_fg_alternating/"
        "environment_teacher_benoit_rubric_norm1_7_20260427_185453/"
        "teacher/split_ids.json"
    )
    data_dir = Path("data/raw/manifesto_corpus_benoit")
    dataverse_dir = Path("data/examples/benoit_dataverse")
    if not split_path.exists() or not data_dir.exists() or not dataverse_dir.exists():
        pytest.skip("local Benoit split/corpus fixtures are not available")

    split_map, _ = baselines._load_split_map(split_path)
    dataset = ManifestoDataset(data_dir=data_dir, require_text=True)
    rows = baselines._load_benoit_rows(
        dimension=PolicyDimension.ENVIRONMENT,
        dataset=dataset,
        split_map=split_map,
    )

    assert len(rows) == 185
    assert Counter(row["split"] for row in rows) == {"train": 107, "val": 30, "test": 48}
    assert {row["summary_source"] for row in rows} == {"benoit_masked_summary"}
    assert all(row["summary"] == row["masked_summary"] for row in rows)
