"""Shared data-prep layer for C-TreePO example families.

Provides a single, family-agnostic split representation
(:mod:`src.ctreepo.data.splits`) and shared corpus-manifest / bootstrap plumbing
(:mod:`src.ctreepo.data.prep_common`) used by the Markov and LDA per-family prep
entrypoints (``markov_data_prep.prepare_markov_corpus`` and
``lda_data_prep.prepare_lda_corpus``), mirroring the manifesto family.
"""

from __future__ import annotations

from src.ctreepo.data.splits import (
    CorpusSplit,
    SPLIT_SCHEMA_VERSION,
    split_from_count_slices,
    split_from_id_lists,
    splits_agree,
    validate_split,
)
from src.ctreepo.data.prep_common import (
    CorpusManifest,
    MANIFEST_SCHEMA_VERSION,
    PreparedCorpus,
    default_processed_root,
    ensure_repo_on_path,
    load_corpus_manifest,
    processed_corpus_dir,
    write_corpus_manifest,
)

__all__ = [
    "CorpusSplit",
    "SPLIT_SCHEMA_VERSION",
    "split_from_count_slices",
    "split_from_id_lists",
    "splits_agree",
    "validate_split",
    "CorpusManifest",
    "MANIFEST_SCHEMA_VERSION",
    "PreparedCorpus",
    "default_processed_root",
    "ensure_repo_on_path",
    "load_corpus_manifest",
    "processed_corpus_dir",
    "write_corpus_manifest",
]
