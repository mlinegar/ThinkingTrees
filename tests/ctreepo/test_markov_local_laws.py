"""Tests for shared Markov local-law sketch helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.ops_checks import LawKind
from src.ctreepo.sim.core.contextual_sbijax import (
    build_contextual_response_dataset,
    make_synthetic_markov_docs,
    markov_exact_sketch_targets_for_dataset,
    palette_block_map,
)
from src.ctreepo.sim.core.markov_local_laws import (
    MARKOV_COUNT_SKETCH_LAW_SET_ID,
    markov_canonical_project_np,
    markov_exact_merge_np,
    markov_exact_sketch_from_tokens_np,
    markov_local_law_observation_rows,
)


def test_exact_sketch_encoding_matches_contextual_targets() -> None:
    block_by_token = palette_block_map(vocab_size=8, n_regimes=2)
    docs = make_synthetic_markov_docs(
        n_docs=3,
        doc_tokens=16,
        vocab_size=8,
        n_regimes=2,
        expected_boundaries=3.0,
        seed=11,
    )
    dataset = build_contextual_response_dataset(
        docs,
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=2,
        seed=12,
    )
    expected = markov_exact_sketch_targets_for_dataset(
        dataset,
        block_by_token=block_by_token,
        target_scale=32.0,
        n_regimes=2,
    )
    actual = markov_exact_sketch_from_tokens_np(
        dataset.item_tokens,
        block_by_token=block_by_token,
        pad_id=dataset.pad_id,
        target_scale=32.0,
        n_regimes=2,
    )
    assert np.allclose(actual, expected)


def test_exact_merge_matches_direct_encoding_of_concatenated_span() -> None:
    block_by_token = [0, 0, 1, 1]
    pad_id = 4
    left = np.asarray([[0, 0, 2, pad_id], [2, 2, pad_id, pad_id]], dtype=np.int64)
    right = np.asarray([[2, 1, 1, pad_id], [0, 3, 3, pad_id]], dtype=np.int64)
    concat = np.asarray([[0, 0, 2, 2, 1, 1], [2, 2, 0, 3, 3, pad_id]], dtype=np.int64)
    left_state = markov_exact_sketch_from_tokens_np(
        left,
        block_by_token=block_by_token,
        pad_id=pad_id,
        target_scale=16.0,
        n_regimes=2,
    )
    right_state = markov_exact_sketch_from_tokens_np(
        right,
        block_by_token=block_by_token,
        pad_id=pad_id,
        target_scale=16.0,
        n_regimes=2,
    )
    direct = markov_exact_sketch_from_tokens_np(
        concat,
        block_by_token=block_by_token,
        pad_id=pad_id,
        target_scale=16.0,
        n_regimes=2,
    )
    merged = markov_exact_merge_np(
        left_state,
        right_state,
        target_scale=16.0,
        n_regimes=2,
    )
    assert np.allclose(merged, direct)


def test_canonical_projection_detects_count_shifted_state_against_target() -> None:
    exact = np.asarray([[1.0 / 16.0, 1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    projected = markov_canonical_project_np(exact, target_scale=16.0, n_regimes=2)
    assert np.allclose(projected, exact)

    shifted = exact.copy()
    shifted[:, 0] += 1.0 / 16.0
    shifted_projected = markov_canonical_project_np(
        shifted,
        target_scale=16.0,
        n_regimes=2,
    )
    assert np.mean(np.abs(shifted_projected - exact)) > 0.0


def test_law_rows_are_keyed_by_lawkind_with_valid_sparse_metadata() -> None:
    rows = markov_local_law_observation_rows(
        leaf_losses=[0.0, 1.0],
        merge_losses=[0.5],
        idempotence_losses=[0.25, 0.0],
        supervision_mode="sparse_ipw",
        leaf_rate=0.5,
        merge_rate=1.0,
        idempotence_rate=0.25,
        seed=22,
    )
    assert rows.law_set_id == MARKOV_COUNT_SKETCH_LAW_SET_ID
    assert set(rows.rows_by_law) == {
        LawKind.L1_LEAF,
        LawKind.L2_MERGE,
        LawKind.L3_IDEMPOTENCE,
    }
    metadata = rows.to_metadata()
    assert metadata["supervision_mode"] == "sparse_ipw"
    assert metadata["row_counts"]["leaf_preservation"] == 2
    assert metadata["row_counts"]["merge_preservation"] == 1
    assert 0 <= metadata["observed_counts"]["leaf_preservation"] <= 2
    assert metadata["propensity_means"]["merge_preservation"] == pytest.approx(1.0)
