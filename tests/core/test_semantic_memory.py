from __future__ import annotations

import numpy as np

from src.core.documents import DocumentResult
from src.core.semantic_memory import (
    SemanticMemoryConfig,
    SemanticMemoryIndex,
    temporal_delta_targets,
)
from src.pipelines.batched import BatchedDocPipeline, BatchedPipelineConfig


def _vec(x: float, y: float) -> np.ndarray:
    return np.asarray([x, y], dtype=np.float32)


def test_semantic_query_respects_scope_order_and_future_exclusion(tmp_path):
    index = SemanticMemoryIndex(
        SemanticMemoryConfig(
            enabled=True,
            index_dir=tmp_path / "semantic",
            top_k=5,
            lambda_year=0.08,
            temporal_mode=True,
        )
    )

    common = {
        "country_code": 11,
        "party_id": 100,
        "party_family": 30,
        "rile": -10.0,
    }
    index.add_document(doc_id="d_same_old", vector=_vec(1.0, 0.0), metadata={**common, "year": 2000, "date_code": 200001})
    index.add_document(
        doc_id="d_future",
        vector=_vec(1.0, 0.0),
        metadata={**common, "year": 2005, "date_code": 200501},
    )
    index.add_document(
        doc_id="d_family",
        vector=_vec(1.0, 0.0),
        metadata={"country_code": 11, "party_id": 200, "party_family": 30, "year": 1999, "date_code": 199901, "rile": -8.0},
    )
    index.add_document(
        doc_id="d_global",
        vector=_vec(1.0, 0.0),
        metadata={"country_code": 41, "party_id": 300, "party_family": 60, "year": 1998, "date_code": 199801, "rile": 12.0},
    )

    neighbors = index.query(
        query_vector=_vec(1.0, 0.0),
        query_meta={"country_code": 11, "party_id": 100, "party_family": 30, "year": 2001, "date_code": 200106},
        top_k=5,
    )
    ids = [n.doc_id for n in neighbors]
    assert "d_future" not in ids
    assert ids[:3] == ["d_same_old", "d_family", "d_global"]


def test_temporal_delta_targets_use_nearest_prior_same_party_country(tmp_path):
    index = SemanticMemoryIndex(
        SemanticMemoryConfig(
            enabled=True,
            index_dir=tmp_path / "semantic_delta",
            top_k=5,
            temporal_mode=True,
        )
    )
    index.add_document(
        doc_id="older",
        vector=_vec(1.0, 0.0),
        metadata={"party_id": 10, "country_code": 11, "party_family": 30, "year": 2000, "date_code": 200006, "rile": -20.0},
    )
    index.add_document(
        doc_id="newer",
        vector=_vec(1.0, 0.0),
        metadata={"party_id": 10, "country_code": 11, "party_family": 30, "year": 2002, "date_code": 200201, "rile": -5.0},
    )

    rows = [
        {
            "manifesto_id": "target1",
            "party_id": 10,
            "country_code": 11,
            "party_family": 30,
            "year": 2003,
            "date_code": 200303,
            "true_rile": 15.0,
        },
        {
            "manifesto_id": "target2",
            "party_id": 999,
            "country_code": 99,
            "party_family": 99,
            "year": 2003,
            "date_code": 200303,
            "true_rile": 1.0,
        },
    ]
    deltas = temporal_delta_targets(rows=rows, index=index, rile_key="true_rile")
    assert len(deltas) == 2
    assert deltas[0] == ((15.0 - (-5.0)) / 200.0)
    assert deltas[1] is None


def test_post_score_write_policy_writes_once(tmp_path):
    sem_cfg = SemanticMemoryConfig(
        enabled=True,
        index_dir=tmp_path / "semantic_policy",
        update_policy="post_score",
        index_granularity="doc",
    )
    pipeline = BatchedDocPipeline(
        BatchedPipelineConfig(
            semantic_memory=sem_cfg,
            show_progress=False,
        )
    )
    index = SemanticMemoryIndex(sem_cfg)
    pipeline._semantic_index = index
    pipeline._semantic_doc_embedder = object()

    result = DocumentResult(
        doc_id="d_policy",
        original_content="Texto breve",
        metadata={
            "party_id": 10,
            "country_code": 11,
            "party_family": 30,
            "year": 2001,
            "date_code": 200101,
        },
    )
    vec = _vec(1.0, 0.0)

    # No score yet: post-score policy must not write.
    pipeline._semantic_write_after_score(result=result, query_vector=vec)
    assert index.report()["doc_entries"] == 0

    # First scored write succeeds exactly once.
    result.estimated_score = -12.5
    pipeline._semantic_write_after_score(result=result, query_vector=vec)
    assert index.report()["doc_entries"] == 1

    # Repeated calls for the same doc_id should not duplicate entries.
    pipeline._semantic_write_after_score(result=result, query_vector=vec)
    assert index.report()["doc_entries"] == 1
