"""Canonical Benoit-replication reference values used by the paper figures.

The numbers here are cited in the paper body; the figure generators import
them so a single edit propagates through F1/F2/F3 and the auto-regenerated
LaTeX table. Keep this module small and grep-able.
"""

from __future__ import annotations

from typing import Final


BENOIT_SPLIT_EXPERT_R: Final[float] = 0.880
"""Economic split-expert Pearson (Benoit 2025, Table 3). Cited in App. H."""

PARITY_GAP: Final[float] = 0.0
"""Internal-external Pearson gap at which internal agreement matches external."""

ECONOMIC_PEARSON_AXIS: Final[tuple[float, float]] = (0.75, 0.92)
"""External-Pearson color/axis bounds pinned across F1/F2/F3."""

ECONOMIC_GAP_AXIS: Final[tuple[float, float]] = (-0.05, 0.20)
"""Signed internal-external gap axis. RdBu_r centers on PARITY_GAP=0."""

HEADLINE_CELL: Final[dict] = {
    "lane": "raw_init",
    "leaf_size_tokens": 256,
    "stage_name": "f1g1",
    "external_expert_pearson": 0.909,
    "f_star_gap": 0.056,
    "tolerance": 0.001,
}
"""The main-text-cited cell; renderers hard-fail if the CSV disagrees."""


BENOIT_INIT_LANE: Final[str] = "benoit_init"
RAW_INIT_LANE: Final[str] = "raw_init"
