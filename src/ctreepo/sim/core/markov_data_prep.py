"""Single data-prep entrypoint for the Markov changepoint example family.

Wraps the existing in-core builder
:func:`build_markov_changepoint_ops_count_data_bundle` and materializes the
result to disk under the shared ``data/processed/<family>/<name>/`` convention
with an id-based split (:class:`src.ctreepo.data.CorpusSplit`) and the shared
``corpus_manifest.json``.

This is additive: it does not change the generator. The Markov bundle's native
split is an integer count-slice (``train_docs`` / ``val_docs`` / ``test_docs``
prefix counts of one generated corpus); :func:`split_from_count_slices` promotes
that into the family-agnostic id-based form so it agrees with the manifesto
representation.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from src.ctreepo.data.prep_common import (
    CorpusManifest,
    PreparedCorpus,
    processed_corpus_dir,
    write_corpus_manifest,
)
from src.ctreepo.data.splits import (
    CorpusSplit,
    positional_ids,
    split_from_count_slices,
    validate_split,
)
from src.ctreepo.sim.core.markov_changepoint_ops_count import (
    MarkovOPSDataBundle,
    OPSCountConfig,
    build_markov_changepoint_ops_count_data_bundle,
)


DOCS_FILENAME = "corpus_docs.jsonl"


def _doc_to_dict(doc: Any) -> Dict[str, Any]:
    return {
        "tokens": [int(t) for t in doc.tokens],
        "token_regimes": [int(r) for r in doc.token_regimes],
        "transition_regimes": [int(r) for r in doc.transition_regimes],
        "true_boundaries": [int(b) for b in doc.true_boundaries],
    }


def _generator_summary(config: OPSCountConfig) -> Dict[str, Any]:
    keys = (
        "problem_id",
        "n_regimes",
        "vocab_size",
        "generator_profile",
        "min_tokens",
        "max_tokens",
        "min_segments",
        "max_segments",
        "min_seg_len",
        "max_seg_len",
        "fixed_leaf_tokens",
        "train_docs",
        "val_docs",
        "test_docs",
        "data_seed",
        "seed",
    )
    out: Dict[str, Any] = {}
    for key in keys:
        if hasattr(config, key):
            value = getattr(config, key)
            out[key] = value
    return out


def _write_docs_jsonl(path: Path, bundle: MarkovOPSDataBundle, *, order: Sequence[str]) -> int:
    import json as _json

    n = 0
    with open(path, "w", encoding="utf-8") as handle:
        for split in order:
            docs = getattr(bundle, f"{split}_docs")
            for doc in docs:
                handle.write(_json.dumps(_doc_to_dict(doc), sort_keys=True) + "\n")
                n += 1
    return n


def prepare_markov_corpus(
    config: OPSCountConfig,
    *,
    out_dir: Optional[Path] = None,
    name: str = "default",
    write_docs: bool = True,
) -> PreparedCorpus:
    """Generate a Markov corpus and materialize it under the shared convention.

    Parameters
    ----------
    config:
        The :class:`OPSCountConfig` driving generation. Its ``train_docs`` /
        ``val_docs`` / ``test_docs`` counts define the split slices.
    out_dir:
        Destination directory. Defaults to
        ``data/processed/markov/<name>/``.
    name:
        Corpus instance name (used for the default ``out_dir`` and the manifest).
    write_docs:
        When True (default), write ``corpus_docs.jsonl`` (docs in
        train -> val -> test order, agreeing with the id-based split).

    Returns
    -------
    PreparedCorpus
        Paths + the id-based :class:`CorpusSplit` that round-trips through
        ``splits.py``.
    """
    bundle = build_markov_changepoint_ops_count_data_bundle(config)

    order = ("train", "val", "test")
    split = split_from_count_slices(
        train=len(bundle.train_docs),
        val=len(bundle.val_docs),
        test=len(bundle.test_docs),
        order=order,
        id_prefix="markov_doc",
        metadata={
            "family": "markov",
            "signatures": {
                "train": str(bundle.train_corpus_signature),
                "val": str(bundle.val_corpus_signature),
                "test": str(bundle.test_corpus_signature),
            },
        },
    )
    validate_split(split, allow_empty_val=True)

    target = Path(out_dir) if out_dir is not None else processed_corpus_dir("markov", name)
    target.mkdir(parents=True, exist_ok=True)

    # Splits land in the corpus dir (split_ids.json), same layout as manifesto.
    split.save(target)

    docs_path: Optional[Path] = None
    if write_docs:
        docs_path = target / DOCS_FILENAME
        _write_docs_jsonl(docs_path, bundle, order=order)

    manifest = CorpusManifest(
        family="markov",
        name=str(name),
        generator=_generator_summary(config),
        split_summary=split.counts(),
        signatures={
            "train_corpus": str(bundle.train_corpus_signature),
            "val_corpus": str(bundle.val_corpus_signature),
            "test_corpus": str(bundle.test_corpus_signature),
        },
        docs_path=DOCS_FILENAME if docs_path is not None else None,
        split_dir=".",
        stats={
            "n_train": len(bundle.train_docs),
            "n_val": len(bundle.val_docs),
            "n_test": len(bundle.test_docs),
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
    "prepare_markov_corpus",
]
