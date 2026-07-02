#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.report.pdf_utils import write_image_page, write_text_page
from src.ctreepo.sim.util import safe_float
from src.ctreepo.sim.core.markov_parity_grid_io import (
    ASSUMED_DOC_TOKENS,
    CANONICAL_TRAIN_LADDER,
    CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
    CLAIM_LEVEL_EXACT_COLLAPSE,
    ONE_LEAF_TARGET_FIXED_LEAF_TOKENS,
    RECIPE_DISPLAY_NAMES,
    load_parity_grid_root,
)


ROOT_PACKAGE_RE = re.compile(r"^full(?P<root>\d+)")
LEAF_COUNT_RE = re.compile(r"_leaf_count(?P<leaf>\d+)")
LEAF_FULL_RE = re.compile(r"_leaf_full(?P<leaf>\d+)")
INTERNAL_COUNT_RE = re.compile(r"_internal(?:_depth\d+)?_count(?P<count>\d+)")
LOCAL_ORDER = ("none", "LcIa10", "LcIa20", "LcIa50", "LcIa100")
FAMILY_COLOR_MAP: Dict[str, str] = {
    "tree_neural": "#1d4ed8",
    "official_fno": "#0f766e",
    "official_fno_sumlen": "#b91c1c",
    "fair_fno": "#d17c00",
}
DEFAULT_FAMILY_COLOR = "#64748b"
PARITY_RECIPE_COLOR_MAP: Dict[str, str] = {
    "historical_replay": "#6b7280",
    "optimization_fairness": "#d97706",
    "capacity_fairness": "#7c3aed",
    "matched_root": "#1d4ed8",
    "fairfno_matched_root": "#0f766e",
}
DEFAULT_PARITY_RECIPE_COLOR = "#475569"


@dataclass(frozen=True)
class CohortSpec:
    label: str
    root: Path


@dataclass
class SummaryRow:
    cohort: str
    scope: str
    package_name: str
    train_doc_count: int
    test_root_mae_mean: float
    val_root_mae_mean: float
    winner_family: str
    winner_config_label: str
    source_path: str
    source_kind: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cohort": self.cohort,
            "scope": self.scope,
            "package_name": self.package_name,
            "train_doc_count": self.train_doc_count,
            "test_root_mae_mean": self.test_root_mae_mean,
            "val_root_mae_mean": self.val_root_mae_mean,
            "winner_family": self.winner_family,
            "winner_config_label": self.winner_config_label,
            "source_path": self.source_path,
            "source_kind": self.source_kind,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an explicit directory-to-directory Markov cohort comparison report."
    )
    parser.add_argument(
        "--cohort",
        action="append",
        required=True,
        help="Comparison cohort in LABEL=PATH form. Repeat for multiple cohorts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs")
        / f"markov_cohort_compare_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--template-report-dir",
        type=Path,
        default=None,
        help="Optional existing family-grids report dir to use as a base for non-ladder figures/sections.",
    )
    parser.add_argument(
        "--historical-summary",
        type=Path,
        default=None,
        help="Optional historical supervision_recovery summary.json used for the parity explainer section.",
    )
    parser.add_argument(
        "--parity-grid-root",
        action="append",
        default=[],
        help="Optional parity-grid study root to merge as a separate geometry/parity panel.",
    )
    return parser.parse_args()


def _parse_cohort_arg(raw: str) -> CohortSpec:
    label, sep, root_text = str(raw).partition("=")
    if not sep or not label.strip() or not root_text.strip():
        raise SystemExit(f"invalid --cohort value: {raw!r}; expected LABEL=PATH")
    return CohortSpec(label=label.strip(), root=Path(root_text).expanduser().resolve())


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


_safe_float = safe_float


def _is_finite(value: float) -> bool:
    return math.isfinite(float(value))


def _infer_scope(raw_scope: str) -> str:
    scope = str(raw_scope or "").strip()
    if scope == "recoverable_v4":
        return "recoverable"
    if scope == "r12_seg10to12" or scope == "structural_core_v1::r12_seg10to12":
        return "structural"
    if scope == "recoverable":
        return "recoverable"
    if scope == "structural":
        return "structural"
    return scope


def _infer_winner_family(config_label: str) -> str:
    label = str(config_label or "").strip().lower()
    if "fair_fno" in label:
        return "fair_fno"
    if "official_fno_sumlen" in label:
        return "official_fno_sumlen"
    if "official_fno" in label:
        return "official_fno"
    if "fno" in label:
        return "fno"
    if "tree" in label:
        return "tree_neural"
    return str(config_label or "")


def _package_meta(package_name: str) -> Dict[str, Any]:
    raw_name = str(package_name or "").strip()
    match = ROOT_PACKAGE_RE.match(raw_name)
    if not match:
        return {
            "root_pct": None,
            "local_pct": None,
            "local_label": raw_name,
            "display_label": raw_name,
        }
    root_pct = int(match.group("root"))
    local_candidates: List[int] = []
    leaf_count_match = LEAF_COUNT_RE.search(raw_name)
    if leaf_count_match is not None:
        local_candidates.append(int(leaf_count_match.group("leaf")))
    leaf_full_match = LEAF_FULL_RE.search(raw_name)
    if leaf_full_match is not None:
        local_candidates.append(int(leaf_full_match.group("leaf")))
    internal_count_match = INTERNAL_COUNT_RE.search(raw_name)
    if internal_count_match is not None:
        local_candidates.append(int(internal_count_match.group("count")))
    local_pct = max(local_candidates) if local_candidates else 0
    local_label = "none" if local_pct == 0 else (
        f"LcIa{local_pct}" if local_pct in {10, 20, 50, 100} else f"Local{local_pct}"
    )
    display_label = f"R{root_pct}" if local_pct == 0 else f"R{root_pct}+{local_label}"
    return {
        "root_pct": root_pct,
        "local_pct": local_pct,
        "local_label": local_label,
        "display_label": display_label,
    }


def _sort_key_for_row(row: SummaryRow) -> tuple[int, int, str]:
    meta = _package_meta(row.package_name)
    root_pct = int(meta["root_pct"] or 10**9)
    local_pct = int(meta["local_pct"] or 0)
    return (root_pct, local_pct, row.package_name)


def _load_legacy_supervision_summary(cohort: CohortSpec) -> List[SummaryRow]:
    summary_path = cohort.root / "supervision_recovery" / "summary.json"
    if not summary_path.exists():
        return []
    payload = dict(_load_json(summary_path))
    rows = list(payload.get("family_rows") or [])
    best_by_key: Dict[tuple[str, str, int], SummaryRow] = {}
    for raw_row in rows:
        scope = _infer_scope(str(raw_row.get("scope_key") or ""))
        if scope not in {"recoverable", "structural"}:
            continue
        package_name = str(raw_row.get("package_name") or "").strip()
        train_doc_count = int(raw_row.get("train_doc_count") or 0)
        test_root_mae_mean = _safe_float(raw_row.get("test_root_mae_mean"))
        val_root_mae_mean = _safe_float(raw_row.get("val_root_mae_mean"))
        if not package_name or train_doc_count <= 0 or not _is_finite(test_root_mae_mean):
            continue
        winner_family = str(raw_row.get("baseline_family") or "").strip()
        candidate = SummaryRow(
            cohort=cohort.label,
            scope=scope,
            package_name=package_name,
            train_doc_count=train_doc_count,
            test_root_mae_mean=test_root_mae_mean,
            val_root_mae_mean=val_root_mae_mean,
            winner_family=winner_family,
            winner_config_label=winner_family,
            source_path=str(summary_path),
            source_kind="legacy_supervision_summary",
        )
        key = (scope, package_name, train_doc_count)
        current = best_by_key.get(key)
        if current is None or candidate.test_root_mae_mean < current.test_root_mae_mean:
            best_by_key[key] = candidate
    return sorted(best_by_key.values(), key=_sort_key_for_row)


def _load_package_capacity_rows(cohort: CohortSpec) -> List[SummaryRow]:
    package_root = cohort.root / "package_capacity"
    if not package_root.exists():
        return []
    rows: List[SummaryRow] = []
    for summary_path in sorted(package_root.rglob("tree_fno_capacity_locked_summary.json")):
        if not summary_path.is_file():
            continue
        package_dir = summary_path.parent
        scope = _infer_scope(package_dir.parent.name)
        package_name = package_dir.name
        payload = dict(_load_json(summary_path))
        winning = dict(payload.get("winning_config") or {})
        winner_config_label = str(payload.get("winning_config_label") or "")
        rows.append(
            SummaryRow(
                cohort=cohort.label,
                scope=scope,
                package_name=package_name,
                train_doc_count=int(payload.get("train_doc_count") or 0),
                test_root_mae_mean=_safe_float(winning.get("test_root_mae_mean")),
                val_root_mae_mean=_safe_float(winning.get("val_root_mae_mean")),
                winner_family=_infer_winner_family(winner_config_label),
                winner_config_label=winner_config_label,
                source_path=str(summary_path),
                source_kind="package_capacity_locked_summary",
            )
        )
    return sorted(rows, key=_sort_key_for_row)


def _load_rows_for_cohort(cohort: CohortSpec) -> List[SummaryRow]:
    package_rows = _load_package_capacity_rows(cohort)
    if package_rows:
        return package_rows
    legacy_rows = _load_legacy_supervision_summary(cohort)
    if legacy_rows:
        return legacy_rows
    raise SystemExit(f"no supported Markov results found under cohort root: {cohort.root}")


def _load_legacy_supervision_summary_from_path(path: Path) -> List[SummaryRow]:
    payload = dict(_load_json(path))
    rows = list(payload.get("family_rows") or [])
    out: List[SummaryRow] = []
    for raw_row in rows:
        scope = _infer_scope(str(raw_row.get("scope_key") or ""))
        if scope not in {"recoverable", "structural"}:
            continue
        package_name = str(raw_row.get("package_name") or "").strip()
        train_doc_count = int(raw_row.get("train_doc_count") or 0)
        test_root_mae_mean = _safe_float(raw_row.get("test_root_mae_mean"))
        val_root_mae_mean = _safe_float(raw_row.get("val_root_mae_mean"))
        if not package_name or train_doc_count <= 0 or not _is_finite(test_root_mae_mean):
            continue
        winner_family = str(raw_row.get("baseline_family") or "").strip()
        out.append(
            SummaryRow(
                cohort="historical",
                scope=scope,
                package_name=package_name,
                train_doc_count=train_doc_count,
                test_root_mae_mean=test_root_mae_mean,
                val_root_mae_mean=val_root_mae_mean,
                winner_family=winner_family,
                winner_config_label=winner_family,
                source_path=str(path),
                source_kind="legacy_supervision_summary",
            )
        )
    return out


def _load_phase_progress(root: Path) -> Mapping[str, Any]:
    combined_path = root / "combined_scheduler_status.json"
    if not combined_path.exists():
        return {}
    payload = dict(_load_json(combined_path))
    return dict(payload.get("phase_progress") or {})


def _scope_pending_rows(cohort: CohortSpec, scope: str) -> List[Dict[str, Any]]:
    phase_progress = _load_phase_progress(cohort.root)
    out: List[Dict[str, Any]] = []
    for raw_key, raw_state in sorted(phase_progress.items()):
        key = str(raw_key)
        if not key.startswith(f"{scope}/"):
            continue
        package_name = key.split("/", 1)[1]
        state = dict(raw_state or {})
        if str(state.get("state") or "") == "completed":
            continue
        out.append(
            {
                "package_name": package_name,
                "display_label": _package_meta(package_name)["display_label"],
                "state": str(state.get("state") or ""),
                "percent_complete": float(state.get("percent_complete") or 0.0),
                "completed_items": int(state.get("completed_items") or 0),
                "items_total": int(state.get("items_total") or 0),
            }
        )
    return out


def _overlapping_train_docs(rows: Sequence[SummaryRow], cohorts: Sequence[CohortSpec]) -> List[int]:
    by_cohort: Dict[str, set[int]] = defaultdict(set)
    for row in rows:
        by_cohort[row.cohort].add(int(row.train_doc_count))
    if not cohorts:
        return []
    overlap = None
    for cohort in cohorts:
        values = by_cohort.get(cohort.label, set())
        overlap = set(values) if overlap is None else overlap & set(values)
    return sorted(overlap or [])


def _train_docs_by_cohort(rows: Sequence[SummaryRow], cohorts: Sequence[CohortSpec]) -> Dict[str, List[int]]:
    by_cohort: Dict[str, set[int]] = defaultdict(set)
    for row in rows:
        by_cohort[row.cohort].add(int(row.train_doc_count))
    return {
        cohort.label: sorted(by_cohort.get(cohort.label, set()))
        for cohort in cohorts
    }


def _group_rows(
    rows: Sequence[SummaryRow],
    *,
    train_doc_count: int,
) -> Dict[str, Dict[str, Dict[str, SummaryRow]]]:
    grouped: Dict[str, Dict[str, Dict[str, SummaryRow]]] = {
        "recoverable": defaultdict(dict),
        "structural": defaultdict(dict),
    }
    for row in rows:
        if int(row.train_doc_count) != int(train_doc_count):
            continue
        meta = _package_meta(row.package_name)
        local_label = str(meta["local_label"])
        grouped[row.scope][local_label][row.cohort] = row
    return grouped


def _family_color(family: str) -> str:
    return str(FAMILY_COLOR_MAP.get(str(family or "").strip(), DEFAULT_FAMILY_COLOR))


def _cohort_hatch(cohort_label: str, cohort_order: Sequence[str]) -> str:
    ordered = [str(item) for item in cohort_order]
    try:
        idx = ordered.index(str(cohort_label))
    except ValueError:
        idx = len(ordered)
    hatches = ("", "//", "xx", "..", "\\\\")
    return hatches[idx % len(hatches)]


def _plot_scope_comparison(
    grouped_scope: Mapping[str, Mapping[str, SummaryRow]],
    *,
    train_doc_count: int,
    scope: str,
    cohorts: Sequence[CohortSpec],
    output_path: Path,
) -> str:
    ordered_local_labels = [label for label in LOCAL_ORDER if label in grouped_scope]
    if not ordered_local_labels:
        return ""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n = len(cohorts)
    width = 0.35 if n <= 2 else max(0.2, 0.8 / max(n, 1))
    xs = list(range(len(ordered_local_labels)))
    family_legend_keys: set[str] = set()
    for idx, cohort in enumerate(cohorts):
        offset = (idx - (n - 1) / 2.0) * width
        bar_xs: List[float] = []
        values: List[float] = []
        labels: List[str] = []
        colors: List[str] = []
        for x, local_label in zip(xs, ordered_local_labels):
            row = dict(grouped_scope.get(local_label) or {}).get(cohort.label)
            if row is None or not _is_finite(row.test_root_mae_mean):
                continue
            bar_xs.append(x + offset)
            values.append(float(row.test_root_mae_mean))
            labels.append(str(row.winner_family))
            colors.append(_family_color(str(row.winner_family)))
            family_legend_keys.add(str(row.winner_family))
        if not values:
            continue
        bars = ax.bar(
            bar_xs,
            values,
            width=width * 0.95,
            color=colors,
            edgecolor="#111827",
            linewidth=0.9,
            hatch=_cohort_hatch(cohort.label, [item.label for item in cohorts]),
            alpha=0.9,
        )
        for bar, value, family in zip(bars, values, labels):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value,
                f"{value:.4f}\n{family}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(xs, ordered_local_labels)
    ax.set_ylabel("test_root_mae")
    ax.set_title(f"{scope.title()} @ train_docs={train_doc_count}")
    ax.text(
        0.01,
        0.99,
        "Fill color = model family. Hatch = cohort. Missing bars mean that cohort/package is not completed yet.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    family_handles = [
        Patch(facecolor=_family_color(family), edgecolor="#111827", label=family)
        for family in sorted(family_legend_keys)
    ]
    cohort_handles = [
        Patch(facecolor="#ffffff", edgecolor="#111827", hatch=_cohort_hatch(cohort.label, [item.label for item in cohorts]), label=cohort.label)
        for cohort in cohorts
    ]
    family_legend = ax.legend(handles=family_handles, title="Model family", loc="upper right")
    ax.add_artist(family_legend)
    ax.legend(handles=cohort_handles, title="Cohort", loc="upper center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path)


def _comparison_lines(
    grouped_scope: Mapping[str, Mapping[str, SummaryRow]],
    *,
    train_doc_count: int,
    scope: str,
    cohorts: Sequence[CohortSpec],
) -> List[str]:
    cohort_order = [cohort.label for cohort in cohorts]
    lines: List[str] = [
        f"**{scope.title()} @ train_docs={train_doc_count}**",
        "",
        "Budget note: `R10` means 10% directly root-labeled docs; `R20` means 20% directly root-labeled docs.",
        "",
        "| Local label | " + " | ".join(
            f"{label} test_root_mae / family" for label in cohort_order
        ) + " |",
        "|" + "---|" + "".join("---:|" for _ in cohort_order),
    ]
    ordered_local_labels = [label for label in LOCAL_ORDER if label in grouped_scope]
    for local_label in ordered_local_labels:
        parts = [local_label]
        per_cohort = dict(grouped_scope.get(local_label) or {})
        for cohort_label in cohort_order:
            row = per_cohort.get(cohort_label)
            if row is None:
                parts.append("pending")
            else:
                parts.append(f"{row.test_root_mae_mean:.6f} / `{row.winner_family}`")
        lines.append("| " + " | ".join(parts) + " |")
    return lines


def _single_cohort_lines(
    grouped_scope: Mapping[str, Mapping[str, SummaryRow]],
    *,
    train_doc_count: int,
    scope: str,
    cohort_label: str,
) -> List[str]:
    ordered_local_labels = [label for label in LOCAL_ORDER if label in grouped_scope]
    if not ordered_local_labels:
        return []
    lines: List[str] = [
        f"**{cohort_label} {scope.title()} @ train_docs={train_doc_count}**",
        "",
        "| Local label | test_root_mae / family |",
        "|---|---:|",
    ]
    wrote_any = False
    for local_label in ordered_local_labels:
        row = dict(grouped_scope.get(local_label) or {}).get(cohort_label)
        if row is None:
            continue
        wrote_any = True
        lines.append(f"| {local_label} | {row.test_root_mae_mean:.6f} / `{row.winner_family}` |")
    return lines if wrote_any else []


def _pending_lines(cohort: CohortSpec, scope: str) -> List[str]:
    rows = _scope_pending_rows(cohort, scope)
    if not rows:
        return []
    lines = [f"**Pending {cohort.label} {scope.title()} Roots**", "", "| Package | State | Progress |", "|---|---|---:|"]
    for row in rows:
        lines.append(
            f"| `{row['display_label']}` | `{row['state']}` | {row['percent_complete']:.1f}% ({row['completed_items']}/{row['items_total']}) |"
        )
    return lines


def _split_markdown_sections(text: str) -> tuple[List[str], Dict[str, List[str]]]:
    preamble: List[str] = []
    sections: Dict[str, List[str]] = {}
    current_title = ""
    current_lines: List[str] = []
    for raw_line in text.splitlines():
        line = str(raw_line)
        if line.startswith("## "):
            if current_title:
                sections[current_title] = list(current_lines)
            else:
                preamble = list(current_lines)
            current_title = line[3:].strip()
            current_lines = []
            continue
        current_lines.append(line)
    if current_title:
        sections[current_title] = list(current_lines)
    elif current_lines:
        preamble = list(current_lines)
    return preamble, sections


def _template_markdown_sections(template_report_dir: Path | None) -> tuple[List[str], Dict[str, List[str]]]:
    if template_report_dir is None:
        return [], {}
    template_md = template_report_dir / "report.md"
    if not template_md.exists():
        return [], {}
    return _split_markdown_sections(template_md.read_text(encoding="utf-8"))


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _strip_existing_cohort_addendum(lines: Sequence[str]) -> List[str]:
    out: List[str] = []
    for line in lines:
        if str(line).strip() == "**Cohort Comparison Addendum**":
            break
        out.append(str(line))
    while out and not out[-1].strip():
        out.pop()
    return out


def _default_historical_summary_path(
    template_report_dir: Path | None,
    explicit_path: Path | None,
) -> Path | None:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        return path if path.exists() else None
    if template_report_dir is None:
        return None
    candidate = template_report_dir.parent / "supervision_recovery" / "summary.json"
    return candidate if candidate.exists() else None


def _best_row(
    rows: Sequence[SummaryRow],
    *,
    train_doc_count: int,
    scope: str,
    family: str | None = None,
    require_local: bool | None = None,
) -> SummaryRow | None:
    candidates: List[SummaryRow] = []
    for row in rows:
        if int(row.train_doc_count) != int(train_doc_count):
            continue
        if str(row.scope) != str(scope):
            continue
        if family is not None and str(row.winner_family) != str(family):
            continue
        local_pct = int(_package_meta(row.package_name)["local_pct"] or 0)
        if require_local is True and local_pct <= 0:
            continue
        if require_local is False and local_pct != 0:
            continue
        if not _is_finite(row.test_root_mae_mean):
            continue
        candidates.append(row)
    if not candidates:
        return None
    return min(candidates, key=lambda item: float(item.test_root_mae_mean))


def _parity_explainer_10240(
    historical_rows: Sequence[SummaryRow],
    *,
    train_doc_count: int,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for scope in ("recoverable", "structural"):
        fno_candidates = [
            _best_row(
                historical_rows,
                train_doc_count=train_doc_count,
                scope=scope,
                family=family,
                require_local=False,
            )
            for family in ("official_fno", "official_fno_sumlen")
        ]
        best_fno = min(
            [row for row in fno_candidates if row is not None],
            key=lambda item: float(item.test_root_mae_mean),
            default=None,
        )
        root_tree = _best_row(
            historical_rows,
            train_doc_count=train_doc_count,
            scope=scope,
            family="tree_neural",
            require_local=False,
        )
        local_tree = _best_row(
            historical_rows,
            train_doc_count=train_doc_count,
            scope=scope,
            family="tree_neural",
            require_local=True,
        )
        if best_fno is None or root_tree is None or local_tree is None:
            continue
        out[scope] = {
            "train_doc_count": int(train_doc_count),
            "best_root_only_fno": {
                "family": best_fno.winner_family,
                "package_name": best_fno.package_name,
                "display_label": _package_meta(best_fno.package_name)["display_label"],
                "test_root_mae_mean": float(best_fno.test_root_mae_mean),
            },
            "best_root_only_tree": {
                "family": root_tree.winner_family,
                "package_name": root_tree.package_name,
                "display_label": _package_meta(root_tree.package_name)["display_label"],
                "test_root_mae_mean": float(root_tree.test_root_mae_mean),
            },
            "best_local_tree": {
                "family": local_tree.winner_family,
                "package_name": local_tree.package_name,
                "display_label": _package_meta(local_tree.package_name)["display_label"],
                "test_root_mae_mean": float(local_tree.test_root_mae_mean),
            },
            "root_tree_gap_vs_best_fno": float(root_tree.test_root_mae_mean - best_fno.test_root_mae_mean),
            "local_tree_gap_vs_best_fno": float(local_tree.test_root_mae_mean - best_fno.test_root_mae_mean),
        }
    return out


def _scope_title(scope: str) -> str:
    return "Recoverable" if str(scope) == "recoverable" else "Structural"


def _parity_section_lines(parity: Mapping[str, Mapping[str, Any]]) -> List[str]:
    if not parity:
        return []
    out: List[str] = [
        "This section resolves the 10240-doc parity confusion using the full historical supervision-recovery sweep, not just the newer cohort-comparison addendum.",
        "",
        "Gap convention: `tree - best_FNO`, so positive means the tree point is worse than the best root-only FNO reference and negative means it is better.",
    ]
    for scope in ("recoverable", "structural"):
        entry = dict(parity.get(scope) or {})
        if not entry:
            continue
        best_fno = dict(entry["best_root_only_fno"])
        root_tree = dict(entry["best_root_only_tree"])
        local_tree = dict(entry["best_local_tree"])
        out.extend(
            [
                "",
                f"### {_scope_title(scope)} @ train_docs={entry['train_doc_count']}",
                "",
                "| Row | Family | Package | test_root_mae |",
                "|---|---|---|---:|",
                f"| Best root-only FNO reference | `{best_fno['family']}` | `{best_fno['package_name']}` | {best_fno['test_root_mae_mean']:.6f} |",
                f"| Best root-only tree point | `{root_tree['family']}` | `{root_tree['package_name']}` | {root_tree['test_root_mae_mean']:.6f} |",
                f"| Best tree point with local supervision | `{local_tree['family']}` | `{local_tree['package_name']}` | {local_tree['test_root_mae_mean']:.6f} |",
                f"| Gap: root-only tree vs best FNO | `tree - FNO` | - | {entry['root_tree_gap_vs_best_fno']:+.6f} |",
                f"| Gap: best local tree vs best FNO | `tree - FNO` | - | {entry['local_tree_gap_vs_best_fno']:+.6f} |",
                "",
            ]
        )
        if scope == "recoverable":
            out.extend(
                [
                    "Interpretation: the recoverable 10240 near-parity/win does not come from root-only `tree_neural`.",
                    (
                        f"Root-only tree at `{root_tree['package_name']}` is {root_tree['test_root_mae_mean']:.6f}, "
                        f"which trails the best root-only FNO `{best_fno['family']} {best_fno['package_name']}` "
                        f"at {best_fno['test_root_mae_mean']:.6f} by {entry['root_tree_gap_vs_best_fno']:+.6f}."
                    ),
                    (
                        f"The tree parity point comes from the locally supervised package `{local_tree['package_name']}` "
                        f"at {local_tree['test_root_mae_mean']:.6f}, which moves the tree to {entry['local_tree_gap_vs_best_fno']:+.6f} "
                        "against the best root-only FNO reference."
                    ),
                ]
            )
        else:
            out.extend(
                [
                    "Interpretation: structural behaves differently at 10240.",
                    (
                        f"Root-only tree at `{root_tree['package_name']}` is {root_tree['test_root_mae_mean']:.6f}, "
                        f"well above the best root-only FNO reference `{best_fno['family']} {best_fno['package_name']}` "
                        f"at {best_fno['test_root_mae_mean']:.6f}."
                    ),
                    (
                        f"Even the best locally supervised tree point `{local_tree['package_name']}` at {local_tree['test_root_mae_mean']:.6f} "
                        f"still sits {entry['local_tree_gap_vs_best_fno']:+.6f} above the best FNO, so the structural panel does not support "
                        "the same parity claim as recoverable."
                    ),
                ]
            )
    return out


def _load_parity_grid_payloads(
    roots: Sequence[Path],
) -> Dict[str, Any]:
    merged_rows: Dict[tuple[str, str, str, int, int], Dict[str, Any]] = {}
    sources: List[str] = []
    indexed_roots: List[Dict[str, Any]] = []
    assumed_doc_tokens = int(ASSUMED_DOC_TOKENS)
    one_leaf_target_fixed_leaf_tokens = int(ONE_LEAF_TARGET_FIXED_LEAF_TOKENS)
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.exists():
            continue
        payload = load_parity_grid_root(root)
        indexed_roots.append(
            {
                "root": str(root),
                "state": str(payload.get("state", "") or ""),
                "evidence_status": str(payload.get("evidence_status", "") or ""),
            }
        )
        if str(payload.get("evidence_status", "") or "") == "stopped":
            continue
        sources.append(str(root))
        payload_assumed_doc_tokens = int(payload.get("assumed_doc_tokens", 0) or 0)
        if payload_assumed_doc_tokens > 0:
            assumed_doc_tokens = payload_assumed_doc_tokens
        payload_one_leaf_target = int(payload.get("one_leaf_target_fixed_leaf_tokens", 0) or 0)
        if payload_one_leaf_target > 0:
            one_leaf_target_fixed_leaf_tokens = payload_one_leaf_target
        for raw_row in list(payload.get("rows") or []):
            row = dict(raw_row or {})
            key = (
                str(row.get("scope_label", "")),
                str(row.get("claim_level", "")),
                str(row.get("recipe_id", "")),
                int(row.get("fixed_leaf_tokens", 0) or 0),
                int(row.get("seed", 0) or 0),
            )
            merged_rows[key] = row
    return {
        "sources": sources,
        "indexed_roots": indexed_roots,
        "rows": list(merged_rows.values()),
        "assumed_doc_tokens": assumed_doc_tokens,
        "one_leaf_target_fixed_leaf_tokens": one_leaf_target_fixed_leaf_tokens,
        "canonical_train_ladder": [int(value) for value in CANONICAL_TRAIN_LADDER],
    }


def _parity_recipe_color(recipe_id: str) -> str:
    return str(
        PARITY_RECIPE_COLOR_MAP.get(str(recipe_id or "").strip(), DEFAULT_PARITY_RECIPE_COLOR)
    )


def _historical_family_reference(
    rows: Sequence[SummaryRow],
    *,
    scope: str,
    family: str,
    train_doc_count: int,
) -> SummaryRow | None:
    return _best_row(
        rows,
        train_doc_count=int(train_doc_count),
        scope=str(scope),
        family=str(family),
        require_local=False,
    )


def _parity_grid_rows_for_scope(
    payload: Mapping[str, Any],
    *,
    scope: str,
    train_doc_count: int | None = None,
    claim_level: str = CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw_row in list(payload.get("rows") or []):
        row = dict(raw_row or {})
        if str(row.get("scope_label", "")) != str(scope):
            continue
        if str(row.get("claim_level", "")) != str(claim_level):
            continue
        if train_doc_count is not None and int(row.get("train_doc_count", 0) or 0) != int(train_doc_count):
            continue
        rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row.get("recipe_id", "")),
            int(row.get("fixed_leaf_tokens", 0) or 0),
        )
    )
    return rows


def _parity_grid_train_doc_counts(
    payload: Mapping[str, Any],
    *,
    scope: str | None = None,
    claim_level: str | None = None,
) -> List[int]:
    train_doc_counts: set[int] = set()
    for raw_row in list(payload.get("rows") or []):
        row = dict(raw_row or {})
        if scope is not None and str(row.get("scope_label", "")) != str(scope):
            continue
        if claim_level is not None and str(row.get("claim_level", "")) != str(claim_level):
            continue
        train_doc_count = int(row.get("train_doc_count", 0) or 0)
        if train_doc_count > 0:
            train_doc_counts.add(train_doc_count)
    return sorted(train_doc_counts)


def _plot_geometry_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    historical_rows: Sequence[SummaryRow],
    scope: str,
    train_doc_count: int,
    assumed_doc_tokens: int,
    one_leaf_target_fixed_leaf_tokens: int,
    output_path: Path,
    title: str,
) -> str:
    completed = [
        dict(row)
        for row in rows
        if str((row or {}).get("state", "")) == "completed"
        and _is_finite(_safe_float((row or {}).get("test_root_mae_mean")))
    ]
    if not completed:
        return ""

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    recipes = sorted({str(row.get("recipe_id", "")) for row in completed})
    for recipe_id in recipes:
        sub = sorted(
            [row for row in completed if str(row.get("recipe_id", "")) == recipe_id],
            key=lambda row: int(row.get("fixed_leaf_tokens", 0) or 0),
        )
        xs = [int(row.get("fixed_leaf_tokens", 0) or 0) for row in sub]
        ys = [_safe_float(row.get("test_root_mae_mean")) for row in sub]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.0,
            color=_parity_recipe_color(recipe_id),
            label=str(RECIPE_DISPLAY_NAMES.get(recipe_id, recipe_id)),
        )

    official_fno = _historical_family_reference(
        historical_rows,
        scope=scope,
        family="official_fno",
        train_doc_count=train_doc_count,
    )
    official_fno_sumlen = _historical_family_reference(
        historical_rows,
        scope=scope,
        family="official_fno_sumlen",
        train_doc_count=train_doc_count,
    )
    if official_fno is not None and _is_finite(official_fno.test_root_mae_mean):
        ax.axhline(
            official_fno.test_root_mae_mean,
            color=_family_color("official_fno"),
            linestyle="--",
            linewidth=2.0,
            label=f"historical official_fno {official_fno.package_name}",
        )
    if official_fno_sumlen is not None and _is_finite(official_fno_sumlen.test_root_mae_mean):
        ax.axhline(
            official_fno_sumlen.test_root_mae_mean,
            color=_family_color("official_fno_sumlen"),
            linestyle=":",
            linewidth=2.0,
            label=f"historical official_fno_sumlen {official_fno_sumlen.package_name}",
        )

    ax.axvline(
        one_leaf_target_fixed_leaf_tokens,
        color="#991b1b",
        linestyle=":",
        linewidth=1.8,
    )
    ax.text(
        float(one_leaf_target_fixed_leaf_tokens) + 1.0,
        max(_safe_float(row.get("test_root_mae_mean")) for row in completed),
        f"one-leaf target ({one_leaf_target_fixed_leaf_tokens} tokens)",
        color="#991b1b",
        fontsize=9,
        va="bottom",
    )
    pending = [
        f"{RECIPE_DISPLAY_NAMES.get(str(row.get('recipe_id', '')), str(row.get('recipe_id', '')))}@{int(row.get('fixed_leaf_tokens', 0) or 0)}"
        for row in rows
        if str((row or {}).get("state", "")) != "completed"
    ]
    if pending:
        ax.text(
            0.01,
            0.01,
            "Pending: " + ", ".join(pending[:6]) + (" ..." if len(pending) > 6 else ""),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
        )
    ax.set_title(title)
    ax.set_xlabel("fixed_leaf_tokens")
    ax.set_ylabel("test_root_mae")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path)


def _geometry_section_lines(
    rows: Sequence[Mapping[str, Any]],
    *,
    historical_rows: Sequence[SummaryRow],
    scope: str,
    train_doc_count: int,
    assumed_doc_tokens: int,
    one_leaf_target_fixed_leaf_tokens: int,
) -> List[str]:
    if not rows:
        return []
    best_fno = _historical_family_reference(
        historical_rows,
        scope=scope,
        family="official_fno",
        train_doc_count=train_doc_count,
    )
    sumlen_fno = _historical_family_reference(
        historical_rows,
        scope=scope,
        family="official_fno_sumlen",
        train_doc_count=train_doc_count,
    )
    completed = [
        dict(row)
        for row in rows
        if str((row or {}).get("state", "")) == "completed"
        and _is_finite(_safe_float((row or {}).get("test_root_mae_mean")))
    ]
    pending = [
        dict(row)
        for row in rows
        if str((row or {}).get("state", "")) != "completed"
    ]
    lines: List[str] = [
        (
            f"This panel keeps the historical package ladders intact and adds a separate root-only "
            f"geometry study for `{_scope_title(scope).lower()} full100 @ train_docs={train_doc_count}`."
        ),
        "",
        (
            f"`fixed_leaf_tokens={one_leaf_target_fixed_leaf_tokens}` is the one-leaf target under the current "
            f"`assumed_doc_tokens={assumed_doc_tokens}` reporting convention."
        ),
        (
            "Interpretation rule: this is an empirical one-leaf parity check against the historical full-root FNO "
            "references. It is not a claim of strict architectural identity."
        ),
        "",
    ]
    if best_fno is not None:
        lines.append(
            f"- Historical `official_fno/full100`: `{best_fno.test_root_mae_mean:.6f}`"
        )
    if sumlen_fno is not None:
        lines.append(
            f"- Historical `official_fno_sumlen/full100`: `{sumlen_fno.test_root_mae_mean:.6f}`"
        )
    lines.extend(
        [
            "",
            "| Recipe | fixed_leaf_tokens | State | test_root_mae | Gap vs official_fno |",
            "|---|---:|---|---:|---:|",
        ]
    )
    fno_value = (
        float(best_fno.test_root_mae_mean)
        if best_fno is not None and _is_finite(best_fno.test_root_mae_mean)
        else float("nan")
    )
    for row in sorted(
        rows,
        key=lambda raw: (
            str(raw.get("recipe_id", "")),
            int(raw.get("fixed_leaf_tokens", 0) or 0),
        ),
    ):
        value = _safe_float(row.get("test_root_mae_mean"))
        gap = value - fno_value if _is_finite(value) and math.isfinite(fno_value) else float("nan")
        value_text = f"{value:.6f}" if _is_finite(value) else "pending"
        gap_text = f"{gap:+.6f}" if math.isfinite(gap) else "pending"
        lines.append(
            f"| `{RECIPE_DISPLAY_NAMES.get(str(row.get('recipe_id', '')), str(row.get('recipe_id', '')) )}` | "
            f"{int(row.get('fixed_leaf_tokens', 0) or 0)} | `{row.get('state', '')}` | {value_text} | {gap_text} |"
        )
    one_leaf_completed = [
        row
        for row in completed
        if int(row.get("fixed_leaf_tokens", 0) or 0) == int(one_leaf_target_fixed_leaf_tokens)
    ]
    if one_leaf_completed and math.isfinite(fno_value):
        best_one_leaf = min(
            one_leaf_completed,
            key=lambda row: float(row.get("test_root_mae_mean", float("inf"))),
        )
        gap = float(best_one_leaf.get("test_root_mae_mean")) - fno_value
        lines.extend(["", f"Best one-leaf row: `{RECIPE_DISPLAY_NAMES.get(str(best_one_leaf.get('recipe_id', '')), str(best_one_leaf.get('recipe_id', '')) )}` "
                       f"at `{float(best_one_leaf.get('test_root_mae_mean')):.6f}` ({gap:+.6f} vs historical `official_fno/full100`)."])
        if scope == "recoverable":
            if gap <= 0.0:
                lines.append("Current interpretation: the one-leaf regime has reached empirical parity or better on recoverable.")
            else:
                lines.append("Current interpretation: the one-leaf regime is still above the historical `official_fno/full100` reference on recoverable.")
        else:
            lines.append("Current interpretation: structural remains a confirmation panel only; do not generalize the recoverable parity claim unless this gap closes.")
    elif pending:
        lines.extend(
            [
                "",
                f"One-leaf interpretation is still pending because the relevant `fixed_leaf_tokens={one_leaf_target_fixed_leaf_tokens}` row is not completed yet.",
            ]
        )
    return lines


def _exact_collapse_rows(
    payload: Mapping[str, Any],
    *,
    train_doc_count: int | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw_row in list(payload.get("rows") or []):
        row = dict(raw_row or {})
        if str(row.get("claim_level", "")) != CLAIM_LEVEL_EXACT_COLLAPSE:
            continue
        if train_doc_count is not None and int(row.get("train_doc_count", 0) or 0) != int(train_doc_count):
            continue
        rows.append(row)
    rows.sort(
        key=lambda row: (
            int(row.get("train_doc_count", 0) or 0),
            str(row.get("scope_label", "")),
            str(row.get("recipe_id", "")),
            int(row.get("fixed_leaf_tokens", 0) or 0),
        )
    )
    return rows


def _exact_collapse_section_lines(
    rows: Sequence[Mapping[str, Any]],
) -> List[str]:
    if not rows:
        return []
    lines: List[str] = [
        "This table is the exact-collapse readiness check, separate from the empirical geometry panel.",
        "",
        (
            "Interpretation rule: `strict_collapse_pass=true` only when the one-leaf tree row has "
            "no disallowed config differences against the production-FNO comparison surface and also "
            "carries the required bundle and prefix metadata."
        ),
        "",
        "| train_docs | Scope | Recipe | State | strict_collapse_pass | Bundle | Prefix counts | Diff fields |",
        "|---:|---|---|---|---|---|---|---:|",
    ]
    for raw_row in rows:
        row = dict(raw_row or {})
        diff = dict(row.get("config_diff_vs_official_fno") or {})
        bundle = str(row.get("reference_bundle_source", "") or "") or "pending"
        prefix_counts = list(row.get("train_prefix_counts") or [])
        prefix_text = (
            ", ".join(str(int(value)) for value in prefix_counts)
            if prefix_counts
            else "pending"
        )
        lines.append(
            f"| {int(row.get('train_doc_count', 0) or 0)} | `{row.get('scope_label', '')}` | "
            f"`{RECIPE_DISPLAY_NAMES.get(str(row.get('recipe_id', '')), str(row.get('recipe_id', '')) )}` | "
            f"`{row.get('state', '')}` | `{bool(row.get('strict_collapse_pass', False))}` | "
            f"`{bundle}` | `{prefix_text}` | {len(diff)} |"
        )
        if diff:
            sample_fields = ", ".join(sorted(diff)[:6])
            lines.append(
                f"Current diff fields for `{row.get('scope_label', '')}` @ train_docs={int(row.get('train_doc_count', 0) or 0)}: `{sample_fields}`"
                + (" ..." if len(diff) > 6 else "")
            )
    lines.extend(
        [
            "",
            "Policy: the draft may cite the geometry panel as empirical evidence, but it should not claim strict tree-to-FNO identity unless an exact-collapse row clears this table.",
        ]
    )
    return lines


def _hybrid_markdown_lines(
    *,
    template_report_dir: Path,
    cohort_intro_lines: Sequence[str],
    recoverable_compare_lines: Sequence[str],
    pending_recoverable: Sequence[str],
    structural_compare_lines: Sequence[str],
    pending_structural: Sequence[str],
    supplemental_single_cohort_lines: Sequence[str],
    parity_lines: Sequence[str],
    exact_collapse_lines: Sequence[str],
    recoverable_geometry_lines: Sequence[str],
    structural_geometry_lines: Sequence[str],
) -> List[str]:
    preamble, sections = _template_markdown_sections(template_report_dir)
    if not sections:
        return list(cohort_intro_lines) + [""] + list(recoverable_compare_lines) + [""] + list(structural_compare_lines)
    out: List[str] = [str(line) for line in preamble if not str(line).startswith("Generated: ")]
    protocol_lines = _strip_existing_cohort_addendum(list(sections.get("Protocol / Setup") or []))
    protocol_lines.extend(["", "**Cohort Comparison Addendum**", ""])
    protocol_lines.extend(list(cohort_intro_lines))
    ordered_titles = [
        "Protocol / Setup",
        "Dense Full-Doc Anchor",
        "Recoverable Package Ladder",
        "Structural Package Ladder",
        "Recoverable Ordered Families",
        "Structural Ordered Families",
        "Best Tree Summary",
        "Runtime Appendix",
    ]
    for title in ordered_titles:
        section_lines = list(sections.get(title) or [])
        if title == "Protocol / Setup":
            section_lines = protocol_lines
        if not section_lines:
            continue
        out.extend(["", f"## {title}"])
        out.extend(section_lines)
        if title == "Recoverable Package Ladder":
            out.extend(["", "## Recoverable R10/R20 Cohort Comparison"])
            out.extend(list(recoverable_compare_lines))
            if pending_recoverable:
                out.extend([""] + list(pending_recoverable))
        elif title == "Structural Package Ladder":
            out.extend(["", "## Structural R10/R20 Cohort Comparison"])
            out.extend(list(structural_compare_lines))
            if pending_structural:
                out.extend([""] + list(pending_structural))
            if supplemental_single_cohort_lines:
                out.extend(["", "## Supplemental Single-Cohort Coverage"])
                out.extend(list(supplemental_single_cohort_lines))
        elif title == "Structural Ordered Families" and (
            parity_lines or exact_collapse_lines or recoverable_geometry_lines or structural_geometry_lines
        ):
            if parity_lines:
                out.extend(["", "## 10240 Parity Disambiguation"])
                out.extend(list(parity_lines))
            if exact_collapse_lines:
                out.extend(["", "## Exact-Collapse Readiness"])
                out.extend(list(exact_collapse_lines))
            if recoverable_geometry_lines:
                out.extend(["", "## Recoverable Full100 Geometry / Parity"])
                out.extend(list(recoverable_geometry_lines))
            if structural_geometry_lines:
                out.extend(["", "## Structural Full100 Geometry Confirmation"])
                out.extend(list(structural_geometry_lines))
    for title, section_lines in sections.items():
        if title in ordered_titles:
            continue
        out.extend(["", f"## {title}"])
        out.extend(section_lines)
    return out


def main() -> int:
    from scripts._markov_report_archive import archived_report_exit

    return archived_report_exit(
        legacy_script="scripts/report_markov_cohort_compare.py",
        replacements=(
            "python3 scripts/report_markov_optimization_tradeoffs.py --summary-json <tradeoff_pipeline/tradeoff_report/summary.json>",
        ),
        note=(
            "The cohort-compare report is archived. Cross-run comparison should now be "
            "done through v3 tradeoff/publication summaries with explicit provenance."
        ),
    )

    args = _parse_args()
    cohorts = [_parse_cohort_arg(raw) for raw in list(args.cohort or ())]
    if len(cohorts) < 2:
        raise SystemExit("at least two --cohort arguments are required")
    template_report_dir = (
        Path(args.template_report_dir).expanduser().resolve()
        if args.template_report_dir is not None
        else None
    )
    historical_summary_path = _default_historical_summary_path(
        template_report_dir,
        args.historical_summary,
    )
    parity_grid_roots = [
        Path(raw).expanduser().resolve() for raw in list(args.parity_grid_root or [])
    ]

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for stale_png in figure_dir.glob("*.png"):
        stale_png.unlink()
    generated_at = datetime.now(timezone.utc).isoformat()

    rows: List[SummaryRow] = []
    cohort_rows: Dict[str, List[SummaryRow]] = {}
    for cohort in cohorts:
        loaded = _load_rows_for_cohort(cohort)
        cohort_rows[cohort.label] = loaded
        rows.extend(loaded)

    overlapping_train_docs = _overlapping_train_docs(rows, cohorts)
    if not overlapping_train_docs:
        raise SystemExit("no overlapping train_doc_count values across the requested cohorts")
    train_docs_by_cohort = _train_docs_by_cohort(rows, cohorts)
    target_train_docs = overlapping_train_docs[-1]
    grouped_by_train_docs = {
        int(train_doc_count): _group_rows(rows, train_doc_count=int(train_doc_count))
        for train_doc_count in overlapping_train_docs
    }
    supplemental_single_cohort_lines: List[str] = []
    overlapping_set = {int(value) for value in overlapping_train_docs}
    for cohort in cohorts:
        supplemental_train_docs = [
            int(value)
            for value in train_docs_by_cohort.get(cohort.label, [])
            if int(value) not in overlapping_set
        ]
        for train_doc_count in supplemental_train_docs:
            grouped = _group_rows(rows, train_doc_count=int(train_doc_count))
            recoverable_lines = _single_cohort_lines(
                grouped["recoverable"],
                train_doc_count=int(train_doc_count),
                scope="recoverable",
                cohort_label=cohort.label,
            )
            structural_lines = _single_cohort_lines(
                grouped["structural"],
                train_doc_count=int(train_doc_count),
                scope="structural",
                cohort_label=cohort.label,
            )
            if not recoverable_lines and not structural_lines:
                continue
            if supplemental_single_cohort_lines:
                supplemental_single_cohort_lines.extend([""])
            supplemental_single_cohort_lines.extend(
                [
                    f"### {cohort.label} @ train_docs={int(train_doc_count)}",
                    "",
                    "This rung is available for one cohort but not all requested cohorts, so it is reported here as standalone evidence rather than a paired cohort comparison.",
                ]
            )
            if recoverable_lines:
                supplemental_single_cohort_lines.extend([""] + recoverable_lines)
            if structural_lines:
                supplemental_single_cohort_lines.extend([""] + structural_lines)
    parity_payload: Dict[str, Dict[str, Any]] = {}
    parity_lines: List[str] = []
    historical_rows: List[SummaryRow] = []
    if historical_summary_path is not None:
        historical_rows = _load_legacy_supervision_summary_from_path(historical_summary_path)
        parity_payload = _parity_explainer_10240(historical_rows, train_doc_count=target_train_docs)
        parity_lines = _parity_section_lines(parity_payload)
    parity_grid_payload = _load_parity_grid_payloads(parity_grid_roots)
    parity_assumed_doc_tokens = int(
        parity_grid_payload.get("assumed_doc_tokens", 0) or ASSUMED_DOC_TOKENS
    )
    parity_one_leaf_target_fixed_leaf_tokens = int(
        parity_grid_payload.get("one_leaf_target_fixed_leaf_tokens", 0)
        or ONE_LEAF_TARGET_FIXED_LEAF_TOKENS
    )
    recoverable_geometry_train_docs = _parity_grid_train_doc_counts(
        parity_grid_payload,
        scope="recoverable",
        claim_level=CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
    )
    structural_geometry_train_docs = _parity_grid_train_doc_counts(
        parity_grid_payload,
        scope="structural",
        claim_level=CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
    )
    exact_collapse_rows = _exact_collapse_rows(parity_grid_payload)
    exact_collapse_train_doc_counts = sorted(
        {
            int(dict(row).get("train_doc_count", 0) or 0)
            for row in exact_collapse_rows
            if int(dict(row).get("train_doc_count", 0) or 0) > 0
        }
    )
    exact_collapse_lines = _exact_collapse_section_lines(exact_collapse_rows)
    recoverable_geometry_lines: List[str] = []
    for train_doc_count in recoverable_geometry_train_docs:
        recoverable_geometry_rows = _parity_grid_rows_for_scope(
            parity_grid_payload,
            scope="recoverable",
            train_doc_count=train_doc_count,
            claim_level=CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
        )
        section_lines = _geometry_section_lines(
            recoverable_geometry_rows,
            historical_rows=historical_rows,
            scope="recoverable",
            train_doc_count=train_doc_count,
            assumed_doc_tokens=parity_assumed_doc_tokens,
            one_leaf_target_fixed_leaf_tokens=parity_one_leaf_target_fixed_leaf_tokens,
        )
        if not section_lines:
            continue
        if recoverable_geometry_lines:
            recoverable_geometry_lines.extend([""])
        recoverable_geometry_lines.extend([f"### train_docs={train_doc_count}", ""])
        recoverable_geometry_lines.extend(section_lines)
    structural_geometry_lines: List[str] = []
    for train_doc_count in structural_geometry_train_docs:
        structural_geometry_rows = _parity_grid_rows_for_scope(
            parity_grid_payload,
            scope="structural",
            train_doc_count=train_doc_count,
            claim_level=CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
        )
        section_lines = _geometry_section_lines(
            structural_geometry_rows,
            historical_rows=historical_rows,
            scope="structural",
            train_doc_count=train_doc_count,
            assumed_doc_tokens=parity_assumed_doc_tokens,
            one_leaf_target_fixed_leaf_tokens=parity_one_leaf_target_fixed_leaf_tokens,
        )
        if not section_lines:
            continue
        if structural_geometry_lines:
            structural_geometry_lines.extend([""])
        structural_geometry_lines.extend([f"### train_docs={train_doc_count}", ""])
        structural_geometry_lines.extend(section_lines)

    recoverable_figures_by_train_docs: Dict[int, str] = {}
    structural_figures_by_train_docs: Dict[int, str] = {}
    for train_doc_count in overlapping_train_docs:
        grouped = grouped_by_train_docs[int(train_doc_count)]
        recoverable_fig = _plot_scope_comparison(
            grouped["recoverable"],
            train_doc_count=int(train_doc_count),
            scope="recoverable",
            cohorts=cohorts,
            output_path=figure_dir / f"recoverable_cohort_compare_train_docs_{int(train_doc_count)}.png",
        )
        if recoverable_fig:
            recoverable_figures_by_train_docs[int(train_doc_count)] = recoverable_fig
        structural_fig = _plot_scope_comparison(
            grouped["structural"],
            train_doc_count=int(train_doc_count),
            scope="structural",
            cohorts=cohorts,
            output_path=figure_dir / f"structural_cohort_compare_train_docs_{int(train_doc_count)}.png",
        )
        if structural_fig:
            structural_figures_by_train_docs[int(train_doc_count)] = structural_fig

    recoverable_geometry_figures_by_train_docs: Dict[int, str] = {}
    for train_doc_count in recoverable_geometry_train_docs:
        recoverable_geometry_rows = _parity_grid_rows_for_scope(
            parity_grid_payload,
            scope="recoverable",
            train_doc_count=train_doc_count,
            claim_level=CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
        )
        recoverable_geometry_fig = _plot_geometry_panel(
            recoverable_geometry_rows,
            historical_rows=historical_rows,
            scope="recoverable",
            train_doc_count=train_doc_count,
            assumed_doc_tokens=parity_assumed_doc_tokens,
            one_leaf_target_fixed_leaf_tokens=parity_one_leaf_target_fixed_leaf_tokens,
            output_path=figure_dir
            / f"recoverable_full100_geometry_parity_train_docs_{int(train_doc_count)}.png",
            title=f"Recoverable Full100 Geometry / Parity @ train_docs={train_doc_count}",
        )
        if recoverable_geometry_fig:
            recoverable_geometry_figures_by_train_docs[int(train_doc_count)] = recoverable_geometry_fig

    structural_geometry_figures_by_train_docs: Dict[int, str] = {}
    for train_doc_count in structural_geometry_train_docs:
        structural_geometry_rows = _parity_grid_rows_for_scope(
            parity_grid_payload,
            scope="structural",
            train_doc_count=train_doc_count,
            claim_level=CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
        )
        structural_geometry_fig = _plot_geometry_panel(
            structural_geometry_rows,
            historical_rows=historical_rows,
            scope="structural",
            train_doc_count=train_doc_count,
            assumed_doc_tokens=parity_assumed_doc_tokens,
            one_leaf_target_fixed_leaf_tokens=parity_one_leaf_target_fixed_leaf_tokens,
            output_path=figure_dir
            / f"structural_full100_geometry_confirmation_train_docs_{int(train_doc_count)}.png",
            title=f"Structural Full100 Geometry Confirmation @ train_docs={train_doc_count}",
        )
        if structural_geometry_fig:
            structural_geometry_figures_by_train_docs[int(train_doc_count)] = structural_geometry_fig

    recoverable_fig = recoverable_figures_by_train_docs.get(int(target_train_docs), "")
    structural_fig = structural_figures_by_train_docs.get(int(target_train_docs), "")
    recoverable_geometry_primary_train_docs = (
        recoverable_geometry_train_docs[-1] if recoverable_geometry_train_docs else None
    )
    structural_geometry_primary_train_docs = (
        structural_geometry_train_docs[-1] if structural_geometry_train_docs else None
    )
    recoverable_geometry_fig = (
        recoverable_geometry_figures_by_train_docs.get(int(recoverable_geometry_primary_train_docs), "")
        if recoverable_geometry_primary_train_docs is not None
        else ""
    )
    structural_geometry_fig = (
        structural_geometry_figures_by_train_docs.get(int(structural_geometry_primary_train_docs), "")
        if structural_geometry_primary_train_docs is not None
        else ""
    )

    cohort_intro_lines: List[str] = [
        "This report compares explicit cohort directories only. It does not mix in older rendered reports or pre-aggregated PDFs.",
        "",
        "**Cohorts**",
        "",
    ]
    for cohort in cohorts:
        cohort_intro_lines.append(f"- `{cohort.label}`: `{cohort.root}`")
    cohort_intro_lines.extend(
        [
            "",
            "**Comparison Policy**",
            "",
            f"- `train_docs covered`: `{', '.join(str(int(value)) for value in overlapping_train_docs)}` (all overlapping train-doc counts across the requested cohorts)",
            f"- `primary train_docs`: `{target_train_docs}` (largest overlapping train-doc count; used for legacy/generic figure slots)",
            "- `scope matching`: recoverable is compared to recoverable, structural to structural",
            "- `row provenance`: each plotted value comes directly from the cohort directory that produced it",
            "- `partial ladders`: missing bars mean the corresponding cohort/package is still pending or unavailable",
        ]
    )
    recoverable_compare_lines: List[str] = []
    for train_doc_count in overlapping_train_docs:
        grouped = grouped_by_train_docs[int(train_doc_count)]["recoverable"]
        if not grouped:
            continue
        if recoverable_compare_lines:
            recoverable_compare_lines.extend([""])
        recoverable_compare_lines.extend(
            _comparison_lines(
                grouped,
                train_doc_count=int(train_doc_count),
                scope="recoverable",
                cohorts=cohorts,
            )
        )
    pending_recoverable: List[str] = []
    for cohort in cohorts:
        pending_recoverable.extend(_pending_lines(cohort, "recoverable"))
    structural_compare_lines: List[str] = []
    for train_doc_count in overlapping_train_docs:
        grouped = grouped_by_train_docs[int(train_doc_count)]["structural"]
        if not grouped:
            continue
        if structural_compare_lines:
            structural_compare_lines.extend([""])
        structural_compare_lines.extend(
            _comparison_lines(
                grouped,
                train_doc_count=int(train_doc_count),
                scope="structural",
                cohorts=cohorts,
            )
        )
    pending_structural: List[str] = []
    for cohort in cohorts:
        pending_structural.extend(_pending_lines(cohort, "structural"))
    md_lines: List[str] = ["# Markov Supervision-Recovery Report"]
    if template_report_dir is not None:
        md_lines = _hybrid_markdown_lines(
            template_report_dir=template_report_dir,
            cohort_intro_lines=cohort_intro_lines,
            recoverable_compare_lines=recoverable_compare_lines,
            pending_recoverable=pending_recoverable,
            structural_compare_lines=structural_compare_lines,
            pending_structural=pending_structural,
            supplemental_single_cohort_lines=supplemental_single_cohort_lines,
            parity_lines=parity_lines,
            exact_collapse_lines=exact_collapse_lines,
            recoverable_geometry_lines=recoverable_geometry_lines,
            structural_geometry_lines=structural_geometry_lines,
        )
    else:
        md_lines.extend([""] + cohort_intro_lines)
        md_lines.extend(["", "## Recoverable R10/R20 Cohort Comparison"])
        md_lines.extend(recoverable_compare_lines)
        if pending_recoverable:
            md_lines.extend([""] + pending_recoverable)
        md_lines.extend(["", "## Structural R10/R20 Cohort Comparison"])
        md_lines.extend(structural_compare_lines)
        if pending_structural:
            md_lines.extend([""] + pending_structural)
        if supplemental_single_cohort_lines:
            md_lines.extend(["", "## Supplemental Single-Cohort Coverage"])
            md_lines.extend(supplemental_single_cohort_lines)
        if parity_lines:
            md_lines.extend(["", "## 10240 Parity Disambiguation"])
            md_lines.extend(parity_lines)
        if exact_collapse_lines:
            md_lines.extend(["", "## Exact-Collapse Readiness"])
            md_lines.extend(exact_collapse_lines)
        if recoverable_geometry_lines:
            md_lines.extend(["", "## Recoverable Full100 Geometry / Parity"])
            md_lines.extend(recoverable_geometry_lines)
        if structural_geometry_lines:
            md_lines.extend(["", "## Structural Full100 Geometry Confirmation"])
            md_lines.extend(structural_geometry_lines)

    md_lines.extend(["", "## Source Rows", ""])
    md_lines.extend(
        [
            "| Cohort | Scope | train_docs | Package | Winner | test_root_mae_mean | Source |",
            "|---|---|---:|---|---|---:|---|",
        ]
    )
    for row in sorted(
        [row for row in rows if int(row.train_doc_count) in {int(value) for value in overlapping_train_docs}],
        key=lambda item: (int(item.train_doc_count), item.scope, _sort_key_for_row(item), item.cohort),
    ):
        md_lines.append(
            f"| `{row.cohort}` | `{row.scope}` | {int(row.train_doc_count)} | `{_package_meta(row.package_name)['display_label']}` | `{row.winner_config_label}` | {row.test_root_mae_mean:.6f} | `{row.source_path}` |"
        )

    normalized_md_lines: List[str] = []
    title_written = False
    generated_written = False
    for idx, line in enumerate(md_lines):
        if idx == 0 and str(line).startswith("# "):
            normalized_md_lines.extend([str(line), "", f"Generated: `{generated_at}`"])
            title_written = True
            generated_written = True
            continue
        if str(line).startswith("Generated: "):
            if generated_written:
                continue
            normalized_md_lines.append(f"Generated: `{generated_at}`")
            generated_written = True
            continue
        normalized_md_lines.append(str(line))
    if not title_written:
        normalized_md_lines = ["# Markov Supervision-Recovery Report", "", f"Generated: `{generated_at}`"] + normalized_md_lines
    md_lines = normalized_md_lines

    summary = {
        "generated_at": generated_at,
        "report_kind": "markov_cohort_compare",
        "comparison_policy": {
            "explicit_directory_cohorts_only": True,
            "largest_overlapping_train_docs_only": False,
            "train_doc_count": target_train_docs,
            "train_doc_counts": [int(value) for value in overlapping_train_docs],
            "primary_train_doc_count": target_train_docs,
        },
        "canonical_train_ladder": [int(value) for value in CANONICAL_TRAIN_LADDER],
        "family_palette": dict(FAMILY_COLOR_MAP),
        "cohort_styles": {
            cohort.label: {"hatch": _cohort_hatch(cohort.label, [item.label for item in cohorts])}
            for cohort in cohorts
        },
        "cohorts": [{"label": cohort.label, "root": str(cohort.root)} for cohort in cohorts],
        "rows": [row.to_dict() for row in rows],
        "overlapping_train_doc_counts": overlapping_train_docs,
        "figures": {},
        "pending": {
            cohort.label: {
                "recoverable": _scope_pending_rows(cohort, "recoverable"),
                "structural": _scope_pending_rows(cohort, "structural"),
            }
            for cohort in cohorts
        },
        "single_cohort_train_doc_counts": {
            cohort.label: [
                int(value)
                for value in train_docs_by_cohort.get(cohort.label, [])
                if int(value) not in {int(item) for item in overlapping_train_docs}
            ]
            for cohort in cohorts
        },
    }
    if recoverable_fig:
        summary["figures"]["Recoverable R10/R20 Cohort Comparison"] = recoverable_fig
    if structural_fig:
        summary["figures"]["Structural R10/R20 Cohort Comparison"] = structural_fig
    for train_doc_count, path in sorted(recoverable_figures_by_train_docs.items()):
        summary["figures"][
            f"Recoverable R10/R20 Cohort Comparison @ train_docs={int(train_doc_count)}"
        ] = path
    for train_doc_count, path in sorted(structural_figures_by_train_docs.items()):
        summary["figures"][
            f"Structural R10/R20 Cohort Comparison @ train_docs={int(train_doc_count)}"
        ] = path
    if recoverable_geometry_fig:
        summary["figures"]["Recoverable Full100 Geometry / Parity"] = recoverable_geometry_fig
    if structural_geometry_fig:
        summary["figures"]["Structural Full100 Geometry Confirmation"] = structural_geometry_fig
    for train_doc_count, path in sorted(recoverable_geometry_figures_by_train_docs.items()):
        summary["figures"][
            f"Recoverable Full100 Geometry / Parity @ train_docs={int(train_doc_count)}"
        ] = path
    for train_doc_count, path in sorted(structural_geometry_figures_by_train_docs.items()):
        summary["figures"][
            f"Structural Full100 Geometry Confirmation @ train_docs={int(train_doc_count)}"
        ] = path
    if parity_grid_payload["sources"]:
        summary.update(
            {
                "geometry_parity_present": True,
                "geometry_parity_sources": list(parity_grid_payload["sources"]),
                "geometry_claim_level": CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
                "geometry_train_doc_counts": {
                    "recoverable": [int(value) for value in recoverable_geometry_train_docs],
                    "structural": [int(value) for value in structural_geometry_train_docs],
                },
                "canonical_train_ladder": [int(value) for value in CANONICAL_TRAIN_LADDER],
                "one_leaf_target_fixed_leaf_tokens": int(
                    parity_one_leaf_target_fixed_leaf_tokens
                ),
                "assumed_doc_tokens": int(parity_assumed_doc_tokens),
                "supervision_recovery_parity_grid": parity_grid_payload,
                "exact_collapse_rows": exact_collapse_rows,
                "exact_collapse_train_doc_counts": [int(value) for value in exact_collapse_train_doc_counts],
            }
        )
    if template_report_dir is not None:
        template_summary_path = template_report_dir / "summary.json"
        if template_summary_path.exists():
            template_summary = dict(_load_json(template_summary_path))
            merged_figures = dict(template_summary.get("figures") or {})
            merged_figures.update(summary["figures"])
            template_summary.update(
                {
                    "generated_at": summary["generated_at"],
                    "report_kind": (
                        "markov_family_grids_with_cohort_compare_v3"
                        if parity_grid_payload["sources"]
                        else "markov_family_grids_with_cohort_compare_v2"
                    ),
                    "historical_grid_present": True,
                    "cohort_compare_present": True,
                    "parity_explainer_10240": parity_payload,
                    "comparison_policy": summary["comparison_policy"],
                    "cohorts": summary["cohorts"],
                    "rows": summary["rows"],
                    "pending": summary["pending"],
                    "single_cohort_train_doc_counts": summary["single_cohort_train_doc_counts"],
                    "family_palette": summary["family_palette"],
                    "cohort_styles": summary["cohort_styles"],
                    "figures": merged_figures,
                    "canonical_train_ladder": [int(value) for value in CANONICAL_TRAIN_LADDER],
                }
            )
            if parity_grid_payload["sources"]:
                template_summary.update(
                    {
                        "geometry_parity_present": True,
                        "geometry_parity_sources": list(parity_grid_payload["sources"]),
                        "geometry_claim_level": CLAIM_LEVEL_EMPIRICAL_GEOMETRY,
                        "geometry_train_doc_counts": {
                            "recoverable": [int(value) for value in recoverable_geometry_train_docs],
                            "structural": [int(value) for value in structural_geometry_train_docs],
                        },
                        "one_leaf_target_fixed_leaf_tokens": int(
                            parity_one_leaf_target_fixed_leaf_tokens
                        ),
                        "assumed_doc_tokens": int(parity_assumed_doc_tokens),
                        "supervision_recovery_parity_grid": parity_grid_payload,
                        "exact_collapse_rows": exact_collapse_rows,
                        "exact_collapse_train_doc_counts": [
                            int(value) for value in exact_collapse_train_doc_counts
                        ],
                    }
                )
            summary = template_summary

    report_md = output_dir / "report.md"
    report_pdf = output_dir / "report.pdf"
    summary_json = output_dir / "summary.json"
    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if template_report_dir is not None:
        template_figure_dir = template_report_dir / "figures"
        for name in (
            "dense_full_doc_anchor.png",
            "recoverable_package_ladder.png",
            "structural_package_ladder.png",
            "recoverable_ordered_families.png",
            "structural_ordered_families.png",
        ):
            _copy_if_exists(template_figure_dir / name, figure_dir / name)

    with PdfPages(report_pdf) as pdf:
        template_sections = _template_markdown_sections(template_report_dir)[1]
        if template_report_dir is not None:
            protocol_lines = _strip_existing_cohort_addendum(
                list(template_sections.get("Protocol / Setup") or [])
            )
            protocol_lines.extend(["", "**Cohort Comparison Addendum**", ""])
            protocol_lines.extend(cohort_intro_lines)
            write_text_page(pdf, title="Protocol / Setup", lines=protocol_lines)
            dense_anchor = figure_dir / "dense_full_doc_anchor.png"
            dense_anchor_lines = list(template_sections.get("Dense Full-Doc Anchor") or [])
            if dense_anchor_lines:
                write_text_page(pdf, title="Dense Full-Doc Anchor", lines=dense_anchor_lines)
            if dense_anchor.exists():
                write_image_page(pdf, image_path=dense_anchor, title="Dense Full-Doc Anchor")
            recoverable_pkg_fig = figure_dir / "recoverable_package_ladder.png"
            structural_pkg_fig = figure_dir / "structural_package_ladder.png"
            recoverable_pkg_lines = list(template_sections.get("Recoverable Package Ladder") or [])
            structural_pkg_lines = list(template_sections.get("Structural Package Ladder") or [])
            if recoverable_pkg_lines:
                write_text_page(pdf, title="Recoverable Package Ladder", lines=recoverable_pkg_lines)
            if recoverable_pkg_fig.exists():
                write_image_page(pdf, image_path=recoverable_pkg_fig, title="Recoverable Package Ladder")
        else:
            write_text_page(pdf, title="Markov Supervision-Recovery Report", lines=cohort_intro_lines)
        write_text_page(pdf, title="Recoverable R10/R20 Cohort Comparison", lines=recoverable_compare_lines)
        if pending_recoverable:
            write_text_page(pdf, title="Pending Recoverable", lines=pending_recoverable)
        for train_doc_count, figure_path in sorted(recoverable_figures_by_train_docs.items()):
            write_image_page(
                pdf,
                image_path=Path(figure_path),
                title=f"Recoverable R10/R20 Cohort Comparison @ train_docs={int(train_doc_count)}",
            )
        if template_report_dir is not None:
            if structural_pkg_lines:
                write_text_page(pdf, title="Structural Package Ladder", lines=structural_pkg_lines)
            if structural_pkg_fig.exists():
                write_image_page(pdf, image_path=structural_pkg_fig, title="Structural Package Ladder")
        write_text_page(pdf, title="Structural R10/R20 Cohort Comparison", lines=structural_compare_lines)
        if pending_structural:
            write_text_page(pdf, title="Pending Structural", lines=pending_structural)
        if supplemental_single_cohort_lines:
            write_text_page(pdf, title="Supplemental Single-Cohort Coverage", lines=supplemental_single_cohort_lines)
        for train_doc_count, figure_path in sorted(structural_figures_by_train_docs.items()):
            write_image_page(
                pdf,
                image_path=Path(figure_path),
                title=f"Structural R10/R20 Cohort Comparison @ train_docs={int(train_doc_count)}",
            )
        if template_report_dir is None:
            if parity_lines:
                write_text_page(pdf, title="10240 Parity Disambiguation", lines=parity_lines)
            if exact_collapse_lines:
                write_text_page(pdf, title="Exact-Collapse Readiness", lines=exact_collapse_lines)
            if recoverable_geometry_lines:
                write_text_page(pdf, title="Recoverable Full100 Geometry / Parity", lines=recoverable_geometry_lines)
            for train_doc_count, figure_path in sorted(recoverable_geometry_figures_by_train_docs.items()):
                write_image_page(
                    pdf,
                    image_path=Path(figure_path),
                    title=f"Recoverable Full100 Geometry / Parity @ train_docs={int(train_doc_count)}",
                )
            if structural_geometry_lines:
                write_text_page(pdf, title="Structural Full100 Geometry Confirmation", lines=structural_geometry_lines)
            for train_doc_count, figure_path in sorted(structural_geometry_figures_by_train_docs.items()):
                write_image_page(
                    pdf,
                    image_path=Path(figure_path),
                    title=f"Structural Full100 Geometry Confirmation @ train_docs={int(train_doc_count)}",
                )
        if template_report_dir is not None:
            recoverable_ordered = figure_dir / "recoverable_ordered_families.png"
            structural_ordered = figure_dir / "structural_ordered_families.png"
            recoverable_ordered_lines = list(template_sections.get("Recoverable Ordered Families") or [])
            structural_ordered_lines = list(template_sections.get("Structural Ordered Families") or [])
            if recoverable_ordered_lines:
                write_text_page(pdf, title="Recoverable Ordered Families", lines=recoverable_ordered_lines)
            if recoverable_ordered.exists():
                write_image_page(pdf, image_path=recoverable_ordered, title="Recoverable Ordered Families")
            if structural_ordered_lines:
                write_text_page(pdf, title="Structural Ordered Families", lines=structural_ordered_lines)
            if structural_ordered.exists():
                write_image_page(pdf, image_path=structural_ordered, title="Structural Ordered Families")
            if parity_lines:
                write_text_page(pdf, title="10240 Parity Disambiguation", lines=parity_lines)
            if exact_collapse_lines:
                write_text_page(pdf, title="Exact-Collapse Readiness", lines=exact_collapse_lines)
            if recoverable_geometry_lines:
                write_text_page(pdf, title="Recoverable Full100 Geometry / Parity", lines=recoverable_geometry_lines)
            for train_doc_count, figure_path in sorted(recoverable_geometry_figures_by_train_docs.items()):
                write_image_page(
                    pdf,
                    image_path=Path(figure_path),
                    title=f"Recoverable Full100 Geometry / Parity @ train_docs={int(train_doc_count)}",
                )
            if structural_geometry_lines:
                write_text_page(pdf, title="Structural Full100 Geometry Confirmation", lines=structural_geometry_lines)
            for train_doc_count, figure_path in sorted(structural_geometry_figures_by_train_docs.items()):
                write_image_page(
                    pdf,
                    image_path=Path(figure_path),
                    title=f"Structural Full100 Geometry Confirmation @ train_docs={int(train_doc_count)}",
                )
            for title in ("Best Tree Summary", "Runtime Appendix", "Appendix"):
                section_lines = list(template_sections.get(title) or [])
                if section_lines:
                    write_text_page(pdf, title=title, lines=section_lines)

    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
