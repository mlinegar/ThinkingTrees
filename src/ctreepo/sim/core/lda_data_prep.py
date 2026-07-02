"""Single data-prep entrypoint for the segmented-LDA example family.

LDA data was previously generated only *inside* ``run_*`` experiments. This
module gives LDA the same materialize-to-disk entrypoint the Markov and
manifesto families have, by wrapping the existing generators
:func:`sample_topic_distributions` and :func:`generate_segment_lda_docs`
(producing :class:`SegmentLDADoc`).

It is additive: the generators are unchanged. :class:`LDAPrepConfig` is a small
readable surface over their explicit keyword arguments; generated docs are split
into id-based train/val/test via distinct seeds and written under the shared
``data/processed/<family>/<name>/`` convention with the shared manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from src.ctreepo.data.prep_common import (
    CorpusManifest,
    PreparedCorpus,
    processed_corpus_dir,
    write_corpus_manifest,
)
from src.ctreepo.data.splits import (
    CorpusSplit,
    positional_ids,
    validate_split,
)
from src.ctreepo.sim.core.segment_lda_ops_weight_recovery import (
    SegmentLDADoc,
    generate_segment_lda_docs,
    sample_topic_distributions,
)


DOCS_FILENAME = "corpus_docs.jsonl"


@dataclass(frozen=True)
class LDAPrepConfig:
    """Config surface for :func:`prepare_lda_corpus`.

    Fields map one-to-one onto the arguments of
    :func:`sample_topic_distributions` and :func:`generate_segment_lda_docs`;
    defaults match a small, fast, self-consistent segmented-LDA corpus.
    """

    # Topic-word distributions.
    vocab_size: int = 96
    n_topics: int = 4
    topic_concentration: float = 0.2
    emission_mode: str = "anchored"
    anchor_words_per_topic: int = 8
    anchor_multiplier: float = 25.0

    # Document generation.
    min_tokens: int = 128
    max_tokens: int = 128
    min_segments: int = 2
    max_segments: int = 6
    min_seg_len: int = 8
    max_seg_len: int = 32
    leaf_tokens: int = 16
    align_segments_to_leaves: bool = True
    doc_topic_concentration: float = 0.6
    topic_process: str = "segments"
    boundary_profile: str = "uniform"
    boundary_profile_strength: float = 0.0
    boundary_profile_seed: int = 0
    segment_length_power: float = 0.0

    # Split sizes / seeding.
    train_docs: int = 64
    val_docs: int = 0
    test_docs: int = 32
    seed: int = 0
    val_seed_offset: int = 5_000
    test_seed_offset: int = 10_000


def _doc_to_dict(doc: SegmentLDADoc) -> Dict[str, Any]:
    return {
        "tokens": [int(t) for t in doc.tokens],
        "topics": [int(t) for t in doc.topics],
    }


def _corpus_signature(docs: Sequence[SegmentLDADoc]) -> str:
    h = hashlib.sha256()
    for doc in docs:
        h.update(json.dumps(_doc_to_dict(doc), sort_keys=True).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _generate_split(
    config: LDAPrepConfig,
    *,
    topics: Sequence[np.ndarray],
    n_docs: int,
    seed: int,
) -> Tuple[Tuple[SegmentLDADoc, ...], Dict[str, float]]:
    if int(n_docs) <= 0:
        return tuple(), {}
    return generate_segment_lda_docs(
        int(n_docs),
        topics=topics,
        min_tokens=int(config.min_tokens),
        max_tokens=int(config.max_tokens),
        min_segments=int(config.min_segments),
        max_segments=int(config.max_segments),
        min_seg_len=int(config.min_seg_len),
        max_seg_len=int(config.max_seg_len),
        leaf_tokens=int(config.leaf_tokens),
        align_segments_to_leaves=bool(config.align_segments_to_leaves),
        doc_topic_concentration=float(config.doc_topic_concentration),
        topic_process=str(config.topic_process),
        boundary_profile=str(config.boundary_profile),
        boundary_profile_strength=float(config.boundary_profile_strength),
        boundary_profile_seed=int(config.boundary_profile_seed),
        segment_length_power=float(config.segment_length_power),
        seed=int(seed),
    )


def prepare_lda_corpus(
    config: LDAPrepConfig,
    *,
    out_dir: Optional[Path] = None,
    name: str = "default",
    write_docs: bool = True,
) -> PreparedCorpus:
    """Generate a segmented-LDA corpus and materialize it to disk.

    Sample topic-word distributions once (shared across splits), then generate
    train / val / test with distinct seeds derived from ``config.seed`` and the
    seed offsets (mirroring the Markov family's val/test seed offsets). Docs are
    written to ``corpus_docs.jsonl`` in train -> val -> test order, agreeing with
    the id-based split.
    """
    topics, topic_meta = sample_topic_distributions(
        vocab_size=int(config.vocab_size),
        n_topics=int(config.n_topics),
        topic_concentration=float(config.topic_concentration),
        emission_mode=str(config.emission_mode),
        anchor_words_per_topic=int(config.anchor_words_per_topic),
        anchor_multiplier=float(config.anchor_multiplier),
        seed=int(config.seed),
    )

    train_docs, train_stats = _generate_split(
        config, topics=topics, n_docs=int(config.train_docs), seed=int(config.seed)
    )
    val_docs, _ = _generate_split(
        config,
        topics=topics,
        n_docs=int(config.val_docs),
        seed=int(config.seed) + int(config.val_seed_offset),
    )
    test_docs, _ = _generate_split(
        config,
        topics=topics,
        n_docs=int(config.test_docs),
        seed=int(config.seed) + int(config.test_seed_offset),
    )

    order = ("train", "val", "test")
    docs_by_split = {"train": train_docs, "val": val_docs, "test": test_docs}
    signatures = {
        f"{split}_corpus": _corpus_signature(docs_by_split[split]) for split in order
    }

    # Id-based split: positional ids assigned across the concatenated corpus,
    # so ids agree with the train -> val -> test doc write order.
    total = sum(len(docs_by_split[s]) for s in order)
    ids = positional_ids(total, prefix="lda_doc")
    split_ids: Dict[str, list] = {}
    cursor = 0
    for split in order:
        n = len(docs_by_split[split])
        split_ids[split] = ids[cursor : cursor + n]
        cursor += n
    split = CorpusSplit(
        split_ids=split_ids,
        metadata={"family": "lda", "signatures": signatures},
    )
    validate_split(split, allow_empty_val=True)

    target = Path(out_dir) if out_dir is not None else processed_corpus_dir("lda", name)
    target.mkdir(parents=True, exist_ok=True)
    split.save(target)

    docs_path: Optional[Path] = None
    if write_docs:
        docs_path = target / DOCS_FILENAME
        with open(docs_path, "w", encoding="utf-8") as handle:
            for split_name in order:
                for doc in docs_by_split[split_name]:
                    handle.write(json.dumps(_doc_to_dict(doc), sort_keys=True) + "\n")

    generator_summary = dict(asdict(config))
    generator_summary["topic_meta"] = {
        k: v for k, v in dict(topic_meta or {}).items() if k != "anchors"
    }

    manifest = CorpusManifest(
        family="lda",
        name=str(name),
        generator=generator_summary,
        split_summary=split.counts(),
        signatures=signatures,
        docs_path=DOCS_FILENAME if docs_path is not None else None,
        split_dir=".",
        stats={
            "n_train": len(train_docs),
            "n_val": len(val_docs),
            "n_test": len(test_docs),
            "train_gen_stats": {k: float(v) for k, v in dict(train_stats or {}).items()},
        },
    )
    manifest_path = write_corpus_manifest(target, manifest)

    return PreparedCorpus(
        out_dir=target,
        manifest_path=manifest_path,
        docs_path=docs_path,
        split=split,
        manifest=manifest.to_dict(),
    )


__all__ = [
    "DOCS_FILENAME",
    "LDAPrepConfig",
    "prepare_lda_corpus",
]
