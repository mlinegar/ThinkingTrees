from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from treepo_cdx._json import stable_digest


@dataclass(frozen=True)
class FoldSpec:
    n_folds: int = 5
    seed: int = 0
    eval_fold: int = 0
    namespace: str = "default"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.n_folds) <= 1:
            raise ValueError("n_folds must be > 1")
        eval_fold = int(self.eval_fold)
        if eval_fold < 0 or eval_fold >= int(self.n_folds):
            raise ValueError("eval_fold must be in [0, n_folds)")
        object.__setattr__(self, "n_folds", int(self.n_folds))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "eval_fold", eval_fold)
        object.__setattr__(self, "namespace", str(self.namespace or "default"))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def artifact_id(self) -> str:
        return "folds:" + stable_digest(self.to_dict())[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FoldAssignment:
    unit_id: str
    fold_id: int
    split: str

    def __post_init__(self) -> None:
        if not str(self.unit_id):
            raise ValueError("unit_id is required")
        if int(self.fold_id) < 0:
            raise ValueError("fold_id must be non-negative")
        if str(self.split) not in {"train", "eval"}:
            raise ValueError("split must be 'train' or 'eval'")
        object.__setattr__(self, "unit_id", str(self.unit_id))
        object.__setattr__(self, "fold_id", int(self.fold_id))
        object.__setattr__(self, "split", str(self.split))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_fold_id(unit_id: str, spec: FoldSpec) -> int:
    payload = f"{spec.seed}:{spec.namespace}:{unit_id}".encode("utf-8", errors="ignore")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % int(spec.n_folds)


def assign_folds(unit_ids: Iterable[str], spec: FoldSpec) -> tuple[FoldAssignment, ...]:
    seen: set[str] = set()
    out: list[FoldAssignment] = []
    for raw_unit_id in unit_ids:
        unit_id = str(raw_unit_id)
        if unit_id in seen:
            raise ValueError(f"duplicate unit_id in fold assignment: {unit_id}")
        seen.add(unit_id)
        fold_id = stable_fold_id(unit_id, spec)
        out.append(
            FoldAssignment(
                unit_id=unit_id,
                fold_id=fold_id,
                split="eval" if fold_id == int(spec.eval_fold) else "train",
            )
        )
    return tuple(out)


def fold_view(assignments: Sequence[FoldAssignment], *, eval_fold: int) -> tuple[FoldAssignment, ...]:
    return tuple(
        FoldAssignment(
            unit_id=item.unit_id,
            fold_id=item.fold_id,
            split="eval" if int(item.fold_id) == int(eval_fold) else "train",
        )
        for item in assignments
    )


def validate_fold_disjointness(assignments: Sequence[FoldAssignment]) -> None:
    train = {item.unit_id for item in assignments if item.split == "train"}
    eval_ = {item.unit_id for item in assignments if item.split == "eval"}
    overlap = sorted(train & eval_)
    if overlap:
        raise ValueError(f"fold train/eval overlap: {overlap}")


def split_unit_ids(assignments: Sequence[FoldAssignment]) -> dict[str, tuple[str, ...]]:
    validate_fold_disjointness(assignments)
    return {
        "train": tuple(item.unit_id for item in assignments if item.split == "train"),
        "eval": tuple(item.unit_id for item in assignments if item.split == "eval"),
    }


__all__ = [
    "FoldAssignment",
    "FoldSpec",
    "assign_folds",
    "fold_view",
    "split_unit_ids",
    "stable_fold_id",
    "validate_fold_disjointness",
]
