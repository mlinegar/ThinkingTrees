from __future__ import annotations

from src.tasks.manifesto.data_loader import ManifestoSample
from src.tasks.manifesto.teacher_trace_generator import (
    TeacherTraceRecord,
    build_split_labels,
    select_seed_manifestos,
    summarize_teacher_trace_records,
)


class _FakeDataset:
    def __init__(self, samples: list[ManifestoSample]):
        self._samples = {sample.manifesto_id: sample for sample in samples}

    def get_all_ids(self) -> list[str]:
        return list(self._samples.keys())

    def get_sample(self, manifesto_id: str):
        return self._samples.get(manifesto_id)


def _sample(manifesto_id: str, rile: float, text: str) -> ManifestoSample:
    return ManifestoSample(
        manifesto_id=manifesto_id,
        party_id=1,
        party_name="Party",
        party_abbrev="P",
        country_code=1,
        country_name="Country",
        election_date="2000-01-01",
        date_code=200001,
        text=text,
        rile=rile,
        vote_share=None,
        party_family=None,
    )


def test_select_seed_manifestos_balances_bins_when_requested() -> None:
    text = "A" * 200
    dataset = _FakeDataset(
        [
            _sample("m1", -80.0, text),
            _sample("m2", -70.0, text),
            _sample("m3", 0.0, text),
            _sample("m4", 5.0, text),
            _sample("m5", 80.0, text),
            _sample("m6", 85.0, text),
        ]
    )

    selected = select_seed_manifestos(
        dataset,  # type: ignore[arg-type]
        n_docs=6,
        seed=1,
        min_source_chars=100,
        balanced_bins=True,
    )
    assert len(selected) == 6
    counts: dict[str, int] = {}
    for row in selected:
        counts[row.source_bin_name] = counts.get(row.source_bin_name, 0) + 1
    values = list(counts.values())
    assert max(values) - min(values) <= 1


def test_build_split_labels_uses_exact_counts() -> None:
    labels = build_split_labels(
        total_docs=7,
        train_size=4,
        val_size=2,
        test_size=1,
        seed=123,
    )
    assert len(labels) == 7
    assert labels.count("train") == 4
    assert labels.count("val") == 2
    assert labels.count("test") == 1


def test_teacher_trace_record_and_summary_metrics() -> None:
    record = TeacherTraceRecord(
        example_id="teacher_trace_train_0001",
        split="train",
        source_manifesto_id="m1",
        source_party_abbrev="P",
        source_country_name="Country",
        source_year=2000,
        source_rile_raw=10.0,
        source_bin_name="center",
        source_text="source text",
        expanded_text="expanded text",
        expanded_score_raw=11.0,
        expanded_delta_raw=1.0,
        summary1="s1",
        summary1_score_raw=9.5,
        summary1_delta_raw=-0.5,
        summary2="s2",
        summary2_score_raw=9.0,
        summary2_delta_raw=-1.0,
        summary2_vs_summary1_delta_raw=-0.5,
        same_side_summary1=True,
        same_side_summary2=True,
        trace_critical_points=["c1", "c2"],
        trace_entities=["entity"],
        trace_qualifiers=["qualifier"],
        trace_invariants=["inv"],
        trace_notes="ok",
        attempts_used=1,
    )

    benchmark = record.to_benchmark_doc()
    assert benchmark["reference_score"] == 10.0
    assert benchmark["metadata"]["source_manifesto_id"] == "m1"

    pair_rows = record.to_summary_pair_rows()
    assert len(pair_rows) == 2
    assert pair_rows[0]["hop"] == 1
    assert pair_rows[1]["hop"] == 2

    stats = summarize_teacher_trace_records([record])
    assert stats["n"] == 1
    assert stats["same_side_summary1_pct"] == 100.0

