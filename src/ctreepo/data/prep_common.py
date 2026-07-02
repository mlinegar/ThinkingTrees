"""Shared corpus-prep plumbing for example data families.

This module centralizes three things that were previously copy-pasted across the
one-off ``scripts/prepare_*`` scripts:

* :func:`ensure_repo_on_path` — the ``REPO_ROOT`` / ``sys.path`` bootstrap.
* :func:`processed_corpus_dir` — the standard output convention
  ``data/processed/<family>/<name>/``.
* :class:`CorpusManifest` / :func:`write_corpus_manifest` — the shared
  ``corpus_manifest.json`` schema so every family emits a manifest with the same
  top-level keys (family, name, generator, split summary, signatures, docs path).

It deliberately does not touch the generators. Family prep entrypoints
(``markov_data_prep.prepare_markov_corpus``, ``lda_data_prep.prepare_lda_corpus``)
call these helpers to wrap the existing builders.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.ctreepo.data.splits import CorpusSplit


MANIFEST_SCHEMA_VERSION = "ctreepo.corpus_manifest.v1"
MANIFEST_FILENAME = "corpus_manifest.json"


def repo_root() -> Path:
    """Return the ThinkingTrees repo root (four levels above this file)."""
    return Path(__file__).resolve().parents[3]


def ensure_repo_on_path() -> Path:
    """Insert the repo root onto ``sys.path`` if missing; return it.

    Replaces the ``REPO_ROOT = Path(__file__).resolve().parents[1]`` blocks that
    each ``scripts/prepare_*`` script copy-pasted.
    """
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def default_processed_root() -> Path:
    """Standard root for materialized corpora: ``<repo>/data/processed``."""
    return repo_root() / "data" / "processed"


def _safe_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .replace("::", "__")
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def processed_corpus_dir(
    family: str,
    name: str,
    *,
    processed_root: Optional[Path] = None,
) -> Path:
    """Return ``data/processed/<family>/<name>/`` (not yet created)."""
    root = Path(processed_root) if processed_root is not None else default_processed_root()
    return root / _safe_name(family) / _safe_name(name)


@dataclass
class CorpusManifest:
    """Shared ``corpus_manifest.json`` payload.

    ``family`` is e.g. ``"markov"`` / ``"lda"`` / ``"manifesto"``; ``name`` is the
    corpus/benchmark instance name. ``docs_path`` / ``split_dir`` are recorded
    relative to the manifest directory when they live under it, else absolute.
    """

    family: str
    name: str
    generator: Mapping[str, Any]
    split_summary: Mapping[str, Any]
    signatures: Mapping[str, Any] = field(default_factory=dict)
    docs_path: Optional[str] = None
    split_dir: Optional[str] = None
    stats: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": str(self.schema_version),
            "family": str(self.family),
            "name": str(self.name),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": dict(self.generator or {}),
            "split_summary": dict(self.split_summary or {}),
            "signatures": dict(self.signatures or {}),
            "stats": dict(self.stats or {}),
        }
        if self.docs_path is not None:
            payload["docs_path"] = str(self.docs_path)
        if self.split_dir is not None:
            payload["split_dir"] = str(self.split_dir)
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


def write_corpus_manifest(out_dir: str | Path, manifest: CorpusManifest) -> Path:
    """Write ``corpus_manifest.json`` under ``out_dir`` and return its path."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_FILENAME
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_corpus_manifest(out_dir: str | Path) -> Dict[str, Any]:
    """Load ``corpus_manifest.json`` from ``out_dir``."""
    path = Path(out_dir) / MANIFEST_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class PreparedCorpus:
    """Return value of a family prep entrypoint: where everything landed."""

    out_dir: Path
    manifest_path: Path
    docs_path: Optional[Path]
    split: CorpusSplit
    manifest: Dict[str, Any]


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "repo_root",
    "ensure_repo_on_path",
    "default_processed_root",
    "processed_corpus_dir",
    "CorpusManifest",
    "write_corpus_manifest",
    "load_corpus_manifest",
    "PreparedCorpus",
]
