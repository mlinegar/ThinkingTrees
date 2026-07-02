"""Shared data loader for manifesto f/g ladder publication figures.

The canonical per-cell CSV is produced by
``scripts/plot_manifesto_fg_ladder_grid.py`` and lives under
``outputs/manifesto_fg_alternating/*/plots/manifesto_fg_ladder_grid_rows.csv``.
This module lifts those rows into a small dataclass that the figure and
table generators share, and implements the inherited-cell collapse rule
described in ``paper/ctreepo/appendix/H_benoit_replication.tex``.

Lane assignment is explicit (caller passes ``{lane: [csv_paths]}``) rather
than derived from column values. Run roots move around; CSVs are the
stable surface.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Iterable


POWER_STAGE_RE = re.compile(r"^f(\d+)g(\d+)$")
LABEL_STAGE_RE = re.compile(r"^f\^\{?(\d+)\}?\s*g\^\{?(\d+)\}?$")
LEGACY_STAGE_MAP = {
    "f0": (0, 0),
    "f0g0": (0, 0),
    "f1g_benoit": (1, 0),
    "f1g0": (1, 0),
    "fg": (1, 1),
    "f1g1": (1, 1),
    "fgf": (2, 1),
    "f2g1": (2, 1),
    "fgfg": (2, 2),
    "f2g2": (2, 2),
    "fgfgf": (3, 2),
    "f3g2": (3, 2),
    "fgfgfg": (3, 3),
    "f3g3": (3, 3),
}


@dataclass(frozen=True)
class LadderCell:
    """One evaluated (lane, leaf, stage) cell."""

    lane: str
    family: str
    leaf_size_tokens: int
    iteration: int
    stage_name: str
    stage_label: str
    trained: str
    n_eval: int | None
    internal_f_pearson: float | None
    external_expert_pearson: float | None
    f_star_gap: float | None
    internal_f_mae_1_7: float | None
    external_expert_mae_1_7: float | None
    mean_prediction_1_7: float | None
    mean_teacher_1_7: float | None
    mean_expert_1_7: float | None
    source_type: str
    source_root: str
    source_path: str
    source_created_at: str

    @property
    def stage_ab(self) -> tuple[int, int] | None:
        """(scorer_updates, summarizer_updates) or None for unparseable stages."""
        if self.stage_name in LEGACY_STAGE_MAP:
            return LEGACY_STAGE_MAP[self.stage_name]
        match = POWER_STAGE_RE.match(self.stage_name)
        if match is not None:
            return int(match.group(1)), int(match.group(2))
        label = (self.stage_label or "").replace(" ", "")
        match = LABEL_STAGE_RE.match(label)
        if match is not None:
            return int(match.group(1)), int(match.group(2))
        return None

    @property
    def stage_key(self) -> tuple[int, int]:
        """Sort key for ladder stages: (total updates, scorer updates)."""
        ab = self.stage_ab
        if ab is None:
            return (99, 99)
        a, b = ab
        return (a + b, a)


def _safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if out != out:  # NaN
        return None
    return out


def _safe_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _row_to_cell(row: dict[str, str], lane: str) -> LadderCell:
    return LadderCell(
        lane=lane,
        family=row.get("family", ""),
        leaf_size_tokens=_safe_int(row.get("leaf_size_tokens")) or 0,
        iteration=_safe_int(row.get("iteration")) or 0,
        stage_name=row.get("stage_name", ""),
        stage_label=row.get("stage_label", ""),
        trained=row.get("trained", ""),
        n_eval=_safe_int(row.get("n_eval")),
        internal_f_pearson=_safe_float(row.get("internal_f_pearson")),
        external_expert_pearson=_safe_float(row.get("external_expert_pearson")),
        f_star_gap=_safe_float(row.get("f_star_gap")),
        internal_f_mae_1_7=_safe_float(row.get("internal_f_mae_1_7")),
        external_expert_mae_1_7=_safe_float(row.get("external_expert_mae_1_7")),
        mean_prediction_1_7=_safe_float(row.get("mean_prediction_1_7")),
        mean_teacher_1_7=_safe_float(row.get("mean_teacher_1_7")),
        mean_expert_1_7=_safe_float(row.get("mean_expert_1_7")),
        source_type=row.get("source_type", ""),
        source_root=row.get("source_root", ""),
        source_path=row.get("source_path", ""),
        source_created_at=row.get("source_created_at", ""),
    )


def load_cells(
    csv_paths: Iterable[Path | str],
    *,
    lane: str,
    family: str = "dspy",
    leaf_axis_only: bool = True,
) -> list[LadderCell]:
    """Load all rows from ``csv_paths`` and tag them with ``lane``.

    ``leaf_axis_only`` keeps rows whose ``axis_kind`` is
    ``leaf_size_tokens`` (the publication axis). ``family`` is matched
    against the ``family`` column.
    """
    cells: list[LadderCell] = []
    for csv_path in csv_paths:
        path = Path(csv_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if family and row.get("family") != family:
                    continue
                if leaf_axis_only and row.get("axis_kind") != "leaf_size_tokens":
                    continue
                cells.append(_row_to_cell(row, lane=lane))
    return cells


def dedupe(cells: Iterable[LadderCell]) -> list[LadderCell]:
    """Keep one cell per (lane, leaf_size_tokens, stage_name).

    When multiple rows hit the same key (e.g. the same stage appeared in
    two runs), prefer the one with the newest ``source_created_at``.
    """
    by_key: dict[tuple[str, int, str], LadderCell] = {}
    for cell in cells:
        key = (cell.lane, cell.leaf_size_tokens, cell.stage_name)
        prev = by_key.get(key)
        if prev is None or cell.source_created_at > prev.source_created_at:
            by_key[key] = cell
    return list(by_key.values())


def collapse_inherited(cells: Iterable[LadderCell]) -> list[LadderCell]:
    """Drop rows whose evaluation was inherited from the preceding stage.

    The iteration plotter repeats the same (external_expert_pearson,
    internal_f_pearson, source_path) triple across ladder stages whenever a
    later stage reused the preceding evaluation. We keep only the first
    stage at which a distinct triple appears, per (lane, leaf).
    """
    out: list[LadderCell] = []
    groups: dict[tuple[str, int], list[LadderCell]] = {}
    for cell in cells:
        groups.setdefault((cell.lane, cell.leaf_size_tokens), []).append(cell)
    for key in sorted(groups.keys()):
        group = sorted(groups[key], key=lambda c: c.stage_key)
        last_ext = None
        last_int = None
        last_source = None
        for cell in group:
            sig = (cell.external_expert_pearson, cell.internal_f_pearson, cell.source_path)
            if (
                last_ext is not None
                and sig == (last_ext, last_int, last_source)
            ):
                continue
            out.append(cell)
            last_ext, last_int, last_source = sig
    return out


def per_leaf_best(
    cells: Iterable[LadderCell], *, metric: str = "external_expert_pearson"
) -> list[LadderCell]:
    """Pick the best cell per (lane, leaf_size_tokens) by ``metric``.

    ``metric`` must be an attribute storing ``float | None``. Higher is
    better for Pearson; pass ``metric="f_star_gap"`` with ``reverse=True``
    elsewhere if you ever want the min. This function always maximizes.
    """
    groups: dict[tuple[str, int], list[LadderCell]] = {}
    for cell in cells:
        groups.setdefault((cell.lane, cell.leaf_size_tokens), []).append(cell)
    out: list[LadderCell] = []
    for key in sorted(groups.keys()):
        finite = [c for c in groups[key] if getattr(c, metric) is not None]
        if not finite:
            continue
        best = max(finite, key=lambda c: getattr(c, metric))
        out.append(best)
    return out


def headline_cell(
    cells: Iterable[LadderCell],
    *,
    lane: str,
    leaf_size_tokens: int,
    stage_name: str,
) -> LadderCell | None:
    """Find the single (lane, leaf, stage) cell or return None."""
    for cell in cells:
        if (
            cell.lane == lane
            and cell.leaf_size_tokens == leaf_size_tokens
            and cell.stage_name == stage_name
        ):
            return cell
    return None


def assert_headline(
    cells: Iterable[LadderCell],
    *,
    expected: dict,
) -> LadderCell:
    """Hard-fail if the headline cell's numerics drift from ``expected``.

    Guards the paper's cited numbers (App. H line 451, §9.4) from silent
    drift when the CSV is regenerated.
    """
    tol = expected.get("tolerance", 1e-3)
    cell = headline_cell(
        cells,
        lane=expected["lane"],
        leaf_size_tokens=expected["leaf_size_tokens"],
        stage_name=expected["stage_name"],
    )
    if cell is None:
        raise RuntimeError(
            f"headline cell not found: lane={expected['lane']}, "
            f"leaf={expected['leaf_size_tokens']}, stage={expected['stage_name']}"
        )
    for field in ("external_expert_pearson", "f_star_gap"):
        expected_val = expected.get(field)
        actual = getattr(cell, field)
        if expected_val is None or actual is None:
            continue
        if abs(actual - expected_val) > tol:
            raise RuntimeError(
                f"headline cell drift on {field}: "
                f"expected {expected_val} ± {tol}, got {actual}"
            )
    return cell


def sorted_stages(cells: Iterable[LadderCell]) -> list[str]:
    """Return unique stage_name values in canonical ladder order."""
    seen: dict[str, tuple[int, int]] = {}
    for cell in cells:
        if cell.stage_name not in seen:
            seen[cell.stage_name] = cell.stage_key
    return [name for name, _ in sorted(seen.items(), key=lambda kv: kv[1])]


def sorted_leaves(cells: Iterable[LadderCell]) -> list[int]:
    """Return unique leaf_size_tokens values ascending."""
    return sorted({c.leaf_size_tokens for c in cells})


def as_matrix(
    cells: Iterable[LadderCell],
    *,
    metric: str,
    stages: list[str] | None = None,
    leaves: list[int] | None = None,
) -> tuple["object", list[str], list[int]]:
    """Reshape cells into a (stages × leaves) matrix for heatmap rendering."""
    import numpy as _np

    cells = list(cells)
    if stages is None:
        stages = sorted_stages(cells)
    if leaves is None:
        leaves = sorted_leaves(cells)
    matrix = _np.full((len(stages), len(leaves)), _np.nan, dtype=float)
    for cell in cells:
        try:
            r = stages.index(cell.stage_name)
            c = leaves.index(cell.leaf_size_tokens)
        except ValueError:
            continue
        value = getattr(cell, metric)
        if value is not None:
            matrix[r, c] = value
    return matrix, stages, leaves


def inherited_mask(
    cells: Iterable[LadderCell],
    *,
    stages: list[str],
    leaves: list[int],
) -> "object":
    """Boolean mask marking cells whose evaluation was inherited.

    A cell is "inherited" if an earlier-stage cell at the same (lane, leaf)
    shares the same source_path and the same pair of Pearson values. Used
    to overlay a ``=`` glyph instead of repeating identical numbers.
    """
    import numpy as _np

    out = _np.zeros((len(stages), len(leaves)), dtype=bool)
    groups: dict[tuple[str, int], list[LadderCell]] = {}
    for cell in cells:
        groups.setdefault((cell.lane, cell.leaf_size_tokens), []).append(cell)
    for (lane, leaf), group in groups.items():
        group_sorted = sorted(group, key=lambda c: c.stage_key)
        last = None
        for cell in group_sorted:
            sig = (
                cell.external_expert_pearson,
                cell.internal_f_pearson,
                cell.source_path,
            )
            if last is not None and sig == last:
                try:
                    r = stages.index(cell.stage_name)
                    c = leaves.index(cell.leaf_size_tokens)
                except ValueError:
                    continue
                out[r, c] = True
            last = sig
    return out


__all__ = [
    "LadderCell",
    "load_cells",
    "dedupe",
    "collapse_inherited",
    "per_leaf_best",
    "headline_cell",
    "assert_headline",
    "sorted_stages",
    "sorted_leaves",
    "as_matrix",
    "inherited_mask",
]
