from __future__ import annotations

import inspect
from pathlib import Path

from src.datasets.jsonl import JSONLDataset
from src.tasks.manifesto import generate_data as manifesto_generate_data
from src.tasks.manifesto.lawstress_generator import (
    LawStressSpec,
    PolicyAtom,
    compute_raw_rile_from_atoms,
    generate_lawstress_records,
    generate_lawstress_specs,
    get_rile_bin,
    summarize_spec_balance,
    write_benchmark_docs_jsonl,
)
from src.training import synthetic_data as synthetic_data_compat


def _max_count_spread(counts: dict[str, int]) -> int:
    if not counts:
        return 0
    values = list(counts.values())
    return int(max(values) - min(values))


def test_compute_raw_rile_from_atoms_matches_formula() -> None:
    atoms = [
        PolicyAtom("a1", direction=1, strength=1.0, weight=1.0, segment="A", topic="x"),
        PolicyAtom("a2", direction=-1, strength=0.25, weight=2.0, segment="B", topic="y"),
    ]
    # numerator = (1*1*1) + (-1*0.25*2) = 0.5
    # denominator = (1*1) + (0.25*2) = 1.5
    # y = 100 * (0.5 / 1.5) = 33.333...
    value = compute_raw_rile_from_atoms(atoms)
    assert abs(value - 33.3333333) < 1e-6


def test_get_rile_bin_boundaries() -> None:
    assert get_rile_bin(-100.0).name == "extreme_left"
    assert get_rile_bin(-75.0).name == "far_left"
    assert get_rile_bin(-10.0).name == "center"
    assert get_rile_bin(10.0).name == "center"
    assert get_rile_bin(10.0001).name == "center_right"
    assert get_rile_bin(100.0).name == "extreme_right"


def test_generate_specs_are_balanced_within_split() -> None:
    specs = generate_lawstress_specs(
        split_sizes={"train": 37, "val": 23, "test": 19},
        hard_ratio=0.8,
        real_anchor_ratio=0.3,
        seed=123,
    )
    balance = summarize_spec_balance(specs)

    for split in ("train", "val", "test"):
        split_stats = balance["splits"][split]

        assert _max_count_spread(split_stats["bins"]) <= 1
        assert _max_count_spread(split_stats["laws"]) <= 1
        assert _max_count_spread(split_stats["families"]) <= 1

        n_split = split_stats["n"]
        expected_hard = round(0.8 * n_split)
        assert split_stats["difficulty"]["hard"] == expected_hard
        assert split_stats["difficulty"]["control"] == n_split - expected_hard


def test_generate_records_respects_difficulty_drift_and_bin() -> None:
    specs = generate_lawstress_specs(
        split_sizes={"train": 24, "val": 0, "test": 0},
        hard_ratio=0.8,
        real_anchor_ratio=0.0,
        seed=7,
    )
    records = generate_lawstress_records(
        specs,
        seed=7,
        max_attempts=4,
        teacher_score_fn=None,
        teacher_rewrite_fn=None,
        reference_summary_fn=None,
    )

    assert len(records) == len(specs)
    for row in records:
        assert get_rile_bin(row.y_raw).name == row.bin_name
        if row.difficulty == "hard":
            assert row.naive_drift_norm > 0.20
        else:
            assert row.naive_drift_norm < 0.08


def test_benchmark_jsonl_is_dataset_compatible(tmp_path: Path) -> None:
    specs = [
        LawStressSpec(
            example_id=f"lawstress_train_{idx:04d}",
            split="train",
            bin_name="center",
            law_target="c1_sufficiency",
            family="polarity_cancellation",
            difficulty="control",
            anchor_source="synthetic",
        )
        for idx in range(4)
    ]
    records = generate_lawstress_records(specs, seed=99)

    jsonl_path = tmp_path / "benchmark_docs.jsonl"
    write_benchmark_docs_jsonl(jsonl_path, records)

    dataset = JSONLDataset(path=str(jsonl_path))
    samples = dataset.load_samples(shuffle=False)
    assert len(samples) == len(records)
    assert samples[0].doc_id.startswith("lawstress_train_")
    assert isinstance(samples[0].reference_score, float)


def test_synthetic_import_compatibility_and_generate_data_path() -> None:
    # Compatibility shim is importable and exposes expected symbols.
    assert hasattr(synthetic_data_compat, "SyntheticDataGenerator")

    # Regression guard for stale import path in generate_data synthetic entrypoint.
    source = inspect.getsource(manifesto_generate_data.generate_synthetic_data)
    assert "from src.training.synthetic import" in source
