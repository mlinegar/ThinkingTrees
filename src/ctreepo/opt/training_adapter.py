from __future__ import annotations

from typing import Any, Iterable

from .records import PairwisePreference


def to_training_preference_dataset(records: Iterable[PairwisePreference]) -> Any:
    """Convert opt-layer records into the repo's PreferenceDataset (lazy import)."""
    from src.training.preference.types import PreferenceDataset

    pairs = [record.to_training_preference_pair() for record in records]
    return PreferenceDataset(pairs)
