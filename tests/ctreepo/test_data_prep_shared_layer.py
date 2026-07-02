"""Focused tests for the shared data-prep layer and the per-family entrypoints.

Covers:
* ``src.ctreepo.data.splits`` id-based split round-trip + count-slice agreement.
* ``prepare_markov_corpus`` / ``prepare_lda_corpus`` materialize loadable,
  split-consistent corpora under a temp dir.
"""

from __future__ import annotations

import json

import pytest

from src.ctreepo.data.splits import (
    CorpusSplit,
    SPLIT_SCHEMA_VERSION,
    split_from_count_slices,
    splits_agree,
    validate_split,
)
from src.ctreepo.data.prep_common import (
    MANIFEST_FILENAME,
    load_corpus_manifest,
)


def test_split_roundtrip_and_count_slice_agreement(tmp_path):
    split = split_from_count_slices(train=6, val=2, test=3, id_prefix="doc")
    assert split.counts() == {"train": 6, "val": 2, "test": 3}
    validate_split(split)

    split.save(tmp_path)
    loaded = CorpusSplit.load(tmp_path)
    assert splits_agree(loaded, split)
    assert loaded.schema_version == SPLIT_SCHEMA_VERSION

    # A second count-slice with the same counts/prefix must agree id-for-id.
    again = split_from_count_slices(train=6, val=2, test=3, id_prefix="doc")
    assert splits_agree(again, split)


def test_validate_split_rejects_overlap():
    bad = CorpusSplit(split_ids={"train": ["a", "b"], "val": [], "test": ["b"]})
    with pytest.raises(ValueError):
        validate_split(bad)


def test_prepare_markov_corpus_materializes_and_roundtrips(tmp_path):
    from src.ctreepo.sim.core.markov_changepoint_ops_count import OPSCountConfig
    from src.ctreepo.sim.core.markov_data_prep import prepare_markov_corpus

    config = OPSCountConfig(
        train_docs=6,
        val_docs=2,
        test_docs=3,
        min_tokens=64,
        max_tokens=64,
        min_segments=2,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=32,
        fixed_leaf_tokens=16,
    )
    prepared = prepare_markov_corpus(config, out_dir=tmp_path / "markov", name="test")

    # Split round-trips and validates.
    loaded = CorpusSplit.load(prepared.out_dir)
    assert splits_agree(loaded, prepared.split)
    validate_split(loaded)
    assert prepared.split.counts() == {"train": 6, "val": 2, "test": 3}

    # Count-slice agreement with the id-based split.
    slice_split = split_from_count_slices(train=6, val=2, test=3, id_prefix="markov_doc")
    assert splits_agree(slice_split, prepared.split)

    # Manifest is loadable with the shared schema.
    manifest = load_corpus_manifest(prepared.out_dir)
    assert manifest["family"] == "markov"
    assert manifest["schema_version"].startswith("ctreepo.corpus_manifest")
    assert manifest["split_summary"] == {"train": 6, "val": 2, "test": 3}

    # Docs jsonl has one line per doc across all splits, each parseable.
    assert prepared.docs_path is not None
    lines = prepared.docs_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 11
    first = json.loads(lines[0])
    assert set(first) >= {"tokens", "token_regimes", "true_boundaries"}


def test_prepare_lda_corpus_materializes_and_roundtrips(tmp_path):
    from src.ctreepo.sim.core.lda_data_prep import LDAPrepConfig, prepare_lda_corpus

    config = LDAPrepConfig(
        vocab_size=48,
        n_topics=3,
        anchor_words_per_topic=4,
        min_tokens=64,
        max_tokens=64,
        leaf_tokens=16,
        train_docs=8,
        val_docs=2,
        test_docs=4,
        seed=0,
    )
    prepared = prepare_lda_corpus(config, out_dir=tmp_path / "lda", name="test")

    loaded = CorpusSplit.load(prepared.out_dir)
    assert splits_agree(loaded, prepared.split)
    validate_split(loaded)
    assert prepared.split.counts() == {"train": 8, "val": 2, "test": 4}

    manifest = load_corpus_manifest(prepared.out_dir)
    assert manifest["family"] == "lda"
    assert set(manifest["signatures"]) == {"train_corpus", "val_corpus", "test_corpus"}

    assert prepared.docs_path is not None
    lines = prepared.docs_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 14
    first = json.loads(lines[0])
    assert set(first) == {"tokens", "topics"}
    assert len(first["tokens"]) == len(first["topics"])
