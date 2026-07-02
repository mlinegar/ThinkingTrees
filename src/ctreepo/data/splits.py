"""Family-agnostic id-based train/val/test split representation.

Promoted from the manifesto coverage-split schema (``split_ids.json`` with a
versioned manifest) so the Markov and LDA example families can share one split
representation with the manifesto family.

Two representations are supported and made to agree:

* **id-based** — an explicit ``{split: [id, ...]}`` mapping, the canonical form
  written to ``split_ids.json``. This mirrors the manifesto family
  (``src/tasks/manifesto/coverage_split.py``).
* **count-slice** — the Markov family's historical form, where the split is
  simply integer ``train/val/test`` counts consumed as nested prefix slices of a
  single generated corpus. :func:`split_from_count_slices` converts a count-slice
  into the id-based form (using positional string ids ``doc_00000`` ...), so both
  representations round-trip to the same ``CorpusSplit``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


SPLIT_SCHEMA_VERSION = "ctreepo.corpus_split.v1"
SPLIT_IDS_FILENAME = "split_ids.json"
DEFAULT_SPLIT_ORDER: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class CorpusSplit:
    """An id-based train/val/test split.

    ``split_ids`` maps a split name to an ordered list of document ids. The ids
    are opaque strings; they only need to be unique within the corpus and stable
    across a materialize/load round-trip.
    """

    split_ids: Mapping[str, Sequence[str]]
    schema_version: str = SPLIT_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def ids(self, split: str) -> List[str]:
        return [str(x) for x in list(self.split_ids.get(str(split), []) or [])]

    def counts(self) -> Dict[str, int]:
        return {str(k): len(list(v or [])) for k, v in dict(self.split_ids).items()}

    @property
    def all_ids(self) -> List[str]:
        out: List[str] = []
        for split in self.split_ids:
            out.extend(str(x) for x in list(self.split_ids[split] or []))
        return out

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_split_ids_payload(self) -> Dict[str, List[str]]:
        return {str(k): [str(x) for x in list(v or [])] for k, v in dict(self.split_ids).items()}

    def save(self, split_dir: str | Path) -> Path:
        """Write ``split_ids.json`` under ``split_dir`` and return its path."""
        directory = Path(split_dir)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": str(self.schema_version),
            "split_ids": self.to_split_ids_payload(),
            "counts": self.counts(),
            "metadata": dict(self.metadata or {}),
        }
        path = directory / SPLIT_IDS_FILENAME
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, split_dir: str | Path) -> "CorpusSplit":
        path = Path(split_dir) / SPLIT_IDS_FILENAME
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_payload(payload)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CorpusSplit":
        payload = dict(payload or {})
        # Accept both the versioned wrapper and a bare {split: [ids]} mapping
        # (the manifesto family writes the latter historically).
        if "split_ids" in payload:
            raw = payload.get("split_ids") or {}
            schema_version = str(payload.get("schema_version", SPLIT_SCHEMA_VERSION))
            metadata = dict(payload.get("metadata") or {})
        else:
            raw = payload
            schema_version = SPLIT_SCHEMA_VERSION
            metadata = {}
        split_ids: Dict[str, List[str]] = {}
        for split, values in dict(raw).items():
            if isinstance(values, Mapping):
                split_ids[str(split)] = [str(k) for k in values.keys()]
            else:
                split_ids[str(split)] = [str(x) for x in list(values or [])]
        return cls(split_ids=split_ids, schema_version=schema_version, metadata=metadata)


def positional_ids(n: int, *, prefix: str = "doc", width: int = 6) -> List[str]:
    """Return stable positional ids ``{prefix}_00000`` ... for ``n`` docs."""
    return [f"{prefix}_{i:0{int(width)}d}" for i in range(int(n))]


def split_from_count_slices(
    *,
    train: int,
    val: int,
    test: int,
    order: Sequence[str] = DEFAULT_SPLIT_ORDER,
    id_prefix: str = "doc",
    id_width: int = 6,
    metadata: Mapping[str, Any] | None = None,
) -> CorpusSplit:
    """Convert Markov-style integer count slices into an id-based split.

    The counts are laid out as *nested prefix slices* of one corpus, in ``order``
    (default ``train`` then ``val`` then ``test``), matching how
    ``build_markov_changepoint_ops_count_data_bundle`` slices its generated docs.
    Positional ids are assigned across the whole corpus so the id-based split and
    the count-slice describe exactly the same document positions.
    """
    counts = {"train": int(train), "val": int(val), "test": int(test)}
    total = sum(counts[str(s)] for s in order)
    ids = positional_ids(total, prefix=id_prefix, width=id_width)
    split_ids: Dict[str, List[str]] = {}
    cursor = 0
    for split in order:
        n = counts[str(split)]
        split_ids[str(split)] = ids[cursor : cursor + n]
        cursor += n
    meta = dict(metadata or {})
    meta.setdefault("representation", "count_slice")
    meta.setdefault("counts", {str(s): int(counts[str(s)]) for s in order})
    return CorpusSplit(split_ids=split_ids, metadata=meta)


def split_from_id_lists(
    split_ids: Mapping[str, Sequence[str]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> CorpusSplit:
    """Build a :class:`CorpusSplit` directly from explicit id lists."""
    return CorpusSplit(
        split_ids={str(k): [str(x) for x in list(v or [])] for k, v in dict(split_ids).items()},
        metadata=dict(metadata or {}),
    )


def validate_split(split: CorpusSplit, *, allow_empty_val: bool = True) -> Dict[str, Any]:
    """Validate a split for id uniqueness and non-overlap across splits.

    Returns a summary dict. Raises ``ValueError`` on a hard inconsistency
    (duplicate ids within a split, overlap across splits, empty required split).
    """
    seen: Dict[str, str] = {}
    for split_name in split.split_ids:
        ids = split.ids(split_name)
        within = set()
        for doc_id in ids:
            if doc_id in within:
                raise ValueError(f"duplicate id {doc_id!r} within split {split_name!r}")
            within.add(doc_id)
            if doc_id in seen:
                raise ValueError(
                    f"id {doc_id!r} appears in both {seen[doc_id]!r} and {split_name!r}"
                )
            seen[doc_id] = split_name
    counts = split.counts()
    if counts.get("train", 0) <= 0:
        raise ValueError("train split must be non-empty")
    if not allow_empty_val and counts.get("val", 0) <= 0:
        raise ValueError("val split must be non-empty")
    if counts.get("test", 0) <= 0:
        raise ValueError("test split must be non-empty")
    return {
        "schema_version": split.schema_version,
        "counts": counts,
        "total": len(seen),
    }


def splits_agree(a: CorpusSplit, b: CorpusSplit) -> bool:
    """True iff two splits assign the same id lists (in order) per split."""
    if set(a.split_ids) != set(b.split_ids):
        return False
    for split in a.split_ids:
        if a.ids(split) != b.ids(split):
            return False
    return True


__all__ = [
    "SPLIT_SCHEMA_VERSION",
    "SPLIT_IDS_FILENAME",
    "DEFAULT_SPLIT_ORDER",
    "CorpusSplit",
    "positional_ids",
    "split_from_count_slices",
    "split_from_id_lists",
    "validate_split",
    "splits_agree",
]
