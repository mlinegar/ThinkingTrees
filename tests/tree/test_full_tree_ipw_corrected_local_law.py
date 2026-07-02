from __future__ import annotations

import pytest

from src.tree.full_tree_ipw import (
    FullTreeNodeRecord,
    corrected_local_law_node_summary,
    summarize_full_tree_ipw,
)
from src.tree.ipw import NodeType


def _record(
    node_id: str,
    *,
    proxy_loss: float,
    oracle_loss: float,
    sampled: bool,
    propensity: float,
) -> FullTreeNodeRecord:
    return FullTreeNodeRecord(
        doc_id="doc",
        node_id=node_id,
        depth=0,
        node_type=NodeType.LEAF,
        is_root=False,
        prediction=0.0,
        target=0.0,
        sampled=sampled,
        propensity=propensity,
        proxy_loss=proxy_loss,
        oracle_loss=oracle_loss,
    )


def test_corrected_full_tree_summary_proxy_only_endpoint() -> None:
    summary = corrected_local_law_node_summary(
        [
            _record("a", proxy_loss=0.4, oracle_loss=0.1, sampled=False, propensity=0.0),
            _record("b", proxy_loss=0.2, oracle_loss=0.7, sampled=False, propensity=0.0),
        ]
    )

    assert summary["sampled_count"] == 0
    assert summary["corrected_mean"] == pytest.approx(0.3)


def test_corrected_full_tree_summary_full_oracle_endpoint() -> None:
    summary = corrected_local_law_node_summary(
        [
            _record("a", proxy_loss=0.4, oracle_loss=0.1, sampled=True, propensity=1.0),
            _record("b", proxy_loss=0.2, oracle_loss=0.7, sampled=True, propensity=1.0),
        ]
    )

    assert summary["sampled_count"] == 2
    assert summary["corrected_mean"] == pytest.approx(0.4)


def test_summarize_full_tree_ipw_emits_corrected_diagnostics() -> None:
    summary = summarize_full_tree_ipw(
        [
            _record("a", proxy_loss=0.4, oracle_loss=0.1, sampled=True, propensity=1.0),
            _record("b", proxy_loss=0.2, oracle_loss=0.7, sampled=False, propensity=0.0),
        ],
        [],
    )

    assert "corrected_local_law" in summary
    assert summary["corrected_local_law"]["population_count"] == 2


def test_sampled_full_tree_record_requires_explicit_oracle_loss() -> None:
    record = FullTreeNodeRecord(
        doc_id="doc",
        node_id="sampled",
        depth=0,
        node_type=NodeType.LEAF,
        is_root=False,
        prediction=0.0,
        target=1.0,
        sampled=True,
        propensity=0.5,
        proxy_loss=0.25,
    )

    with pytest.raises(ValueError, match="requires explicit oracle_loss"):
        corrected_local_law_node_summary([record])


def test_unsampled_full_tree_record_without_oracle_loss_stays_proxy_only() -> None:
    record = FullTreeNodeRecord(
        doc_id="doc",
        node_id="unsampled",
        depth=0,
        node_type=NodeType.LEAF,
        is_root=False,
        prediction=0.0,
        target=1.0,
        sampled=False,
        propensity=0.0,
        proxy_loss=0.25,
    )

    summary = corrected_local_law_node_summary([record])

    assert summary["sampled_count"] == 0
    assert summary["corrected_mean"] == pytest.approx(0.25)
