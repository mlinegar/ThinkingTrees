#!/usr/bin/env python3
"""Render a root-only tree/FNO parity diagnosis report."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.report.pdf_utils import write_image_page, write_text_page
from src.ctreepo.sim.util import safe_float


ROOT_STAGE_NAMES = (
    "historical_replay",
    "optimization_fairness",
    "capacity_fairness",
    "matched_root",
    "capacity_sweep",
    "representation_sweep",
)
STAGE_TITLES = {
    "historical_replay": "Historical Replay",
    "optimization_fairness": "Optimization-Fairness Fix",
    "capacity_fairness": "Capacity-Fairness Fix",
    "matched_root": "Combined Matched-Root Recipe",
    "capacity_sweep": "Matched-Root Capacity Sweep",
    "representation_sweep": "Representation Bottleneck Sweep",
    "structural_confirmation": "Structural Confirmation",
}
ROOT_PACKAGE_RE = re.compile(r"^full(?P<root>\d+)")
LEAF_COUNT_RE = re.compile(r"_leaf_count(?P<leaf>\d+)")
LEAF_FULL_RE = re.compile(r"_leaf_full(?P<leaf>\d+)")
INTERNAL_COUNT_RE = re.compile(r"_internal(?:_depth\d+)?_count(?P<count>\d+)")


@dataclass(frozen=True)
class HistoricalReference:
    scope: str
    train_doc_count: int
    best_fno_family: str
    best_fno_package: str
    best_fno_test_root_mae: float
    root_only_tree_package: str
    root_only_tree_test_root_mae: float
    best_local_tree_package: str
    best_local_tree_test_root_mae: float


@dataclass(frozen=True)
class StageResult:
    stage_name: str
    stage_title: str
    root: str
    summary_path: str
    config_label: str
    selection_metric: str
    val_root_mae: float
    test_root_mae: float
    test_leaf_mae: float
    test_merge_mae: float
    best_epoch: float
    elapsed_s: float
    state_dim: int
    hidden_dim: int
    n_epochs: int
    slot_count: int
    fixed_leaf_tokens: int | None
    tree_training_schedule: str
    tree_checkpoint_metric: str
    tree_stage1_checkpoint_metric: str
    tree_stage1_root_weight: float
    tree_leaf_fno_width: int
    tree_leaf_fno_n_modes: int
    tree_leaf_fno_n_layers: int
    gap_vs_best_fno: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a root-only tree/FNO parity diagnosis report."
    )
    parser.add_argument("--historical-summary", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs")
        / f"tree_root_only_parity_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--train-doc-count", type=int, default=10240)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--historical-replay-root", type=Path, default=None)
    parser.add_argument("--optimization-fairness-root", type=Path, default=None)
    parser.add_argument("--capacity-fairness-root", type=Path, default=None)
    parser.add_argument("--matched-root-root", type=Path, default=None)
    parser.add_argument("--capacity-sweep-root", type=Path, default=None)
    parser.add_argument("--representation-sweep-root", type=Path, default=None)
    parser.add_argument("--structural-confirm-root", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


_safe_float = safe_float


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _infer_scope(raw_scope: str) -> str:
    scope = str(raw_scope or "").strip()
    if scope == "recoverable_v4":
        return "recoverable"
    if scope in {"structural", "r12_seg10to12", "structural_core_v1::r12_seg10to12"}:
        return "structural"
    return scope


def _package_meta(package_name: str) -> Dict[str, int]:
    name = str(package_name or "").strip()
    match = ROOT_PACKAGE_RE.match(name)
    root_pct = int(match.group("root")) if match else 0
    local_candidates: list[int] = []
    for pattern in (LEAF_COUNT_RE, LEAF_FULL_RE, INTERNAL_COUNT_RE):
        found = pattern.search(name)
        if found is not None:
            local_candidates.append(int(next(iter(found.groupdict().values()))))
    return {
        "root_pct": root_pct,
        "local_pct": max(local_candidates) if local_candidates else 0,
    }


def _package_has_local_supervision(package_name: str) -> bool:
    return int(_package_meta(package_name).get("local_pct", 0)) > 0


def load_historical_reference(
    summary_path: Path,
    *,
    scope: str,
    train_doc_count: int,
) -> HistoricalReference:
    payload = dict(_load_json(summary_path))
    rows = list(payload.get("family_rows") or [])
    relevant = [
        dict(row)
        for row in rows
        if _infer_scope(str(row.get("scope_key") or "")) == str(scope)
        and int(row.get("train_doc_count") or 0) == int(train_doc_count)
    ]
    if not relevant:
        raise ValueError(
            f"no historical rows found for scope={scope!r} train_doc_count={train_doc_count}"
        )
    fno_rows = [
        row
        for row in relevant
        if str(row.get("package_name") or "") == "full100"
        and str(row.get("baseline_family") or "").startswith("official_fno")
        and _is_finite(row.get("test_root_mae_mean"))
    ]
    root_only_tree_rows = [
        row
        for row in relevant
        if str(row.get("package_name") or "") == "full100"
        and str(row.get("baseline_family") or "") == "tree_neural"
        and _is_finite(row.get("test_root_mae_mean"))
    ]
    local_tree_rows = [
        row
        for row in relevant
        if str(row.get("baseline_family") or "") == "tree_neural"
        and _package_has_local_supervision(str(row.get("package_name") or ""))
        and _is_finite(row.get("test_root_mae_mean"))
    ]
    if not fno_rows or not root_only_tree_rows:
        raise ValueError(
            f"historical summary {summary_path} is missing full100 root-only/FNO rows for scope={scope}"
        )
    best_fno = min(fno_rows, key=lambda row: float(row.get("test_root_mae_mean")))
    root_only_tree = min(
        root_only_tree_rows,
        key=lambda row: float(row.get("test_root_mae_mean")),
    )
    best_local_tree = (
        min(local_tree_rows, key=lambda row: float(row.get("test_root_mae_mean")))
        if local_tree_rows
        else {
            "package_name": "",
            "test_root_mae_mean": float("nan"),
        }
    )
    return HistoricalReference(
        scope=str(scope),
        train_doc_count=int(train_doc_count),
        best_fno_family=str(best_fno.get("baseline_family") or ""),
        best_fno_package=str(best_fno.get("package_name") or ""),
        best_fno_test_root_mae=float(best_fno.get("test_root_mae_mean")),
        root_only_tree_package=str(root_only_tree.get("package_name") or ""),
        root_only_tree_test_root_mae=float(root_only_tree.get("test_root_mae_mean")),
        best_local_tree_package=str(best_local_tree.get("package_name") or ""),
        best_local_tree_test_root_mae=_safe_float(
            best_local_tree.get("test_root_mae_mean")
        ),
    )


def load_stage_result(
    stage_name: str,
    root: Path | None,
    *,
    best_fno_mae: float,
) -> StageResult | None:
    if root is None:
        return None
    summary_path = Path(root) / "tree_fno_capacity_locked_summary.json"
    if not summary_path.exists():
        return None
    payload = dict(_load_json(summary_path))
    winning = dict(payload.get("winning_config") or {})
    if not winning:
        rankings = list(payload.get("locked_rankings") or [])
        if rankings:
            winning = dict(rankings[0] or {})
    config_spec = dict(payload.get("winning_config_spec") or {})
    merged = {**config_spec, **winning}
    test_root_mae = _safe_float(winning.get("test_root_mae_mean"))
    return StageResult(
        stage_name=str(stage_name),
        stage_title=str(STAGE_TITLES.get(stage_name, stage_name.replace("_", " ").title())),
        root=str(Path(root)),
        summary_path=str(summary_path),
        config_label=str(payload.get("winning_config_label") or winning.get("config_label") or ""),
        selection_metric=str(payload.get("selection_metric") or "val_root_mae_mean"),
        val_root_mae=_safe_float(winning.get("val_root_mae_mean")),
        test_root_mae=test_root_mae,
        test_leaf_mae=_safe_float(winning.get("test_leaf_mae_mean")),
        test_merge_mae=_safe_float(winning.get("test_merge_mae_mean")),
        best_epoch=_safe_float(
            winning.get("best_epoch_mean", winning.get("best_epoch", float("nan")))
        ),
        elapsed_s=_safe_float(winning.get("elapsed_s_mean")),
        state_dim=int(merged.get("state_dim", 0) or 0),
        hidden_dim=int(merged.get("hidden_dim", 0) or 0),
        n_epochs=int(merged.get("n_epochs", 0) or 0),
        slot_count=int(merged.get("slot_count", 0) or 0),
        fixed_leaf_tokens=(
            None
            if merged.get("fixed_leaf_tokens") is None
            else int(merged.get("fixed_leaf_tokens", 0))
        ),
        tree_training_schedule=str(merged.get("tree_training_schedule") or ""),
        tree_checkpoint_metric=str(merged.get("tree_checkpoint_metric") or ""),
        tree_stage1_checkpoint_metric=str(
            merged.get("tree_stage1_checkpoint_metric") or ""
        ),
        tree_stage1_root_weight=float(merged.get("tree_stage1_root_weight", 0.0) or 0.0),
        tree_leaf_fno_width=int(merged.get("tree_leaf_fno_width", 0) or 0),
        tree_leaf_fno_n_modes=int(merged.get("tree_leaf_fno_n_modes", 0) or 0),
        tree_leaf_fno_n_layers=int(merged.get("tree_leaf_fno_n_layers", 0) or 0),
        gap_vs_best_fno=(
            float(test_root_mae - best_fno_mae)
            if _is_finite(test_root_mae) and _is_finite(best_fno_mae)
            else float("nan")
        ),
    )


def _best_stage(results: Sequence[StageResult]) -> StageResult | None:
    finite = [result for result in results if _is_finite(result.test_root_mae)]
    if not finite:
        return None
    return min(finite, key=lambda result: float(result.test_root_mae))


def classify_root_only_diagnosis(
    recoverable_results: Mapping[str, StageResult | None],
    *,
    threshold: float,
) -> str:
    historical = recoverable_results.get("historical_replay")
    optimization = recoverable_results.get("optimization_fairness")
    capacity = recoverable_results.get("capacity_fairness")
    best = _best_stage(
        [
            result
            for stage_name, result in recoverable_results.items()
            if stage_name != "historical_replay" and result is not None
        ]
    )
    if best is None or not _is_finite(best.gap_vs_best_fno) or best.gap_vs_best_fno > float(threshold):
        return "root_only_architecture_gap_persists"
    if optimization is not None and _is_finite(optimization.gap_vs_best_fno):
        if optimization.gap_vs_best_fno <= float(threshold):
            return "recipe_fairness_fixed"
    if capacity is not None and _is_finite(capacity.gap_vs_best_fno):
        if capacity.gap_vs_best_fno <= float(threshold):
            return "capacity_fixed"
    historical_gap = (
        float(historical.gap_vs_best_fno)
        if historical is not None and _is_finite(historical.gap_vs_best_fno)
        else float("nan")
    )
    optimization_gap = (
        float(optimization.gap_vs_best_fno)
        if optimization is not None and _is_finite(optimization.gap_vs_best_fno)
        else float("nan")
    )
    capacity_gap = (
        float(capacity.gap_vs_best_fno)
        if capacity is not None and _is_finite(capacity.gap_vs_best_fno)
        else float("nan")
    )
    optimization_improvement = (
        historical_gap - optimization_gap
        if _is_finite(historical_gap) and _is_finite(optimization_gap)
        else float("-inf")
    )
    capacity_improvement = (
        historical_gap - capacity_gap
        if _is_finite(historical_gap) and _is_finite(capacity_gap)
        else float("-inf")
    )
    return (
        "recipe_fairness_fixed"
        if optimization_improvement >= capacity_improvement
        else "capacity_fixed"
    )


def _conclusion_lines(classification: str, *, best_stage: StageResult | None) -> list[str]:
    if classification == "recipe_fairness_fixed":
        return [
            "The root-only gap is primarily a recipe-fairness issue.",
            "The historical tree recipe was optimized for sketch selection rather than direct root recovery, and the corrected root-focused recipe materially closes the gap.",
            (
                f"Best recoverable stage: `{best_stage.stage_title}`"
                if best_stage is not None
                else "Best recoverable stage: unavailable"
            ),
        ]
    if classification == "capacity_fixed":
        return [
            "The root-only gap is primarily a capacity-fairness issue.",
            "Matching the tree recipe to the FNO-scale hidden/state dimensions is necessary to approach the full-root FNO reference.",
            (
                f"Best recoverable stage: `{best_stage.stage_title}`"
                if best_stage is not None
                else "Best recoverable stage: unavailable"
            ),
        ]
    return [
        "The root-only architecture gap still persists after the fair recipe/capacity follow-ups.",
        "Local supervision remains the mechanism that closes the gap in the historical recoverable sweep, rather than a cosmetic bonus.",
        (
            f"Best recoverable stage: `{best_stage.stage_title}`"
            if best_stage is not None
            else "Best recoverable stage: unavailable"
        ),
    ]


def _render_ladder_plot(
    output_path: Path,
    *,
    historical: HistoricalReference,
    recoverable_results: Sequence[StageResult],
) -> str:
    if not recoverable_results:
        return ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xs = list(range(len(recoverable_results)))
    ys = [float(result.test_root_mae) for result in recoverable_results]
    labels = [str(result.stage_title) for result in recoverable_results]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(xs, ys, color="#1d4ed8", edgecolor="#111827", linewidth=1.0)
    ax.axhline(
        float(historical.best_fno_test_root_mae),
        color="#b91c1c",
        linestyle="--",
        linewidth=2.0,
        label=f"Best full-root FNO ({historical.best_fno_family})",
    )
    if _is_finite(historical.best_local_tree_test_root_mae):
        ax.axhline(
            float(historical.best_local_tree_test_root_mae),
            color="#0f766e",
            linestyle=":",
            linewidth=2.0,
            label="Best locally supervised tree",
        )
    for bar, value in zip(bars, ys):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{float(value):.4g}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(xs, labels, rotation=20, ha="right")
    ax.set_ylabel("test_root_mae")
    ax.set_title(
        f"Recoverable Root-Only Diagnosis Ladder @ train_docs={historical.train_doc_count}"
    )
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path)


def generate_root_only_parity_report(
    *,
    historical_summary: Path,
    output_dir: Path,
    train_doc_count: int,
    threshold: float,
    stage_roots: Mapping[str, Path | None],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    historical_recoverable = load_historical_reference(
        historical_summary,
        scope="recoverable",
        train_doc_count=int(train_doc_count),
    )
    try:
        historical_structural = load_historical_reference(
            historical_summary,
            scope="structural",
            train_doc_count=int(train_doc_count),
        )
    except Exception:
        historical_structural = None

    recoverable_results = {
        stage_name: load_stage_result(
            stage_name,
            stage_roots.get(stage_name),
            best_fno_mae=float(historical_recoverable.best_fno_test_root_mae),
        )
        for stage_name in ROOT_STAGE_NAMES
    }
    structural_result = load_stage_result(
        "structural_confirmation",
        stage_roots.get("structural_confirmation"),
        best_fno_mae=(
            float(historical_structural.best_fno_test_root_mae)
            if historical_structural is not None
            else float("nan")
        ),
    )
    best_recoverable_stage = _best_stage(
        [result for result in recoverable_results.values() if result is not None]
    )
    classification = classify_root_only_diagnosis(
        recoverable_results,
        threshold=float(threshold),
    )

    figure_path = Path(output_dir) / "figures" / "recoverable_root_only_diagnosis_ladder.png"
    recoverable_stage_rows = [
        result
        for stage_name, result in recoverable_results.items()
        if result is not None and stage_name in ROOT_STAGE_NAMES
    ]
    ladder_figure = _render_ladder_plot(
        figure_path,
        historical=historical_recoverable,
        recoverable_results=recoverable_stage_rows,
    )

    historical_lines = [
        "**Historical full100 root-only comparison**",
        "",
        "| row | package | test_root_mae | gap vs best FNO |",
        "|---|---|---:|---:|",
        (
            f"| best full-root FNO | `{historical_recoverable.best_fno_package}` / "
            f"`{historical_recoverable.best_fno_family}` | "
            f"{historical_recoverable.best_fno_test_root_mae:.6f} | 0.000000 |"
        ),
        (
            f"| historical root-only tree | `{historical_recoverable.root_only_tree_package}` | "
            f"{historical_recoverable.root_only_tree_test_root_mae:.6f} | "
            f"{historical_recoverable.root_only_tree_test_root_mae - historical_recoverable.best_fno_test_root_mae:.6f} |"
        ),
    ]
    if _is_finite(historical_recoverable.best_local_tree_test_root_mae):
        historical_lines.append(
            "| historical locally supervised tree | "
            f"`{historical_recoverable.best_local_tree_package}` | "
            f"{historical_recoverable.best_local_tree_test_root_mae:.6f} | "
            f"{historical_recoverable.best_local_tree_test_root_mae - historical_recoverable.best_fno_test_root_mae:.6f} |"
        )

    ladder_lines = [
        "**Root-only diagnosis ladder**",
        "",
        "| stage | config | test_root_mae | test_leaf_mae | test_merge_mae | gap vs best FNO | selection_metric | best_epoch | elapsed_s | state_dim | hidden_dim | slot_count | fixed_leaf_tokens |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in recoverable_stage_rows:
        ladder_lines.append(
            "| "
            f"{result.stage_title} | "
            f"`{result.config_label}` | "
            f"{result.test_root_mae:.6f} | "
            f"{result.test_leaf_mae:.6f} | "
            f"{result.test_merge_mae:.6f} | "
            f"{result.gap_vs_best_fno:.6f} | "
            f"`{result.selection_metric}` | "
            f"{result.best_epoch:.3g} | "
            f"{result.elapsed_s:.6g} | "
            f"{result.state_dim} | "
            f"{result.hidden_dim} | "
            f"{result.slot_count} | "
            f"{'none' if result.fixed_leaf_tokens is None else result.fixed_leaf_tokens} |"
        )

    conclusion_lines = [
        "**Conclusion**",
        "",
        f"- classification: `{classification}`",
        f"- success threshold: `{float(threshold):.6f}` absolute MAE over the best full-root FNO",
        *[f"- {line}" for line in _conclusion_lines(classification, best_stage=best_recoverable_stage)],
    ]

    structural_lines: list[str] = []
    if historical_structural is not None:
        structural_lines.extend(
            [
                "**Structural confirmation**",
                "",
                "| row | package | test_root_mae | gap vs best structural FNO |",
                "|---|---|---:|---:|",
                (
                    f"| historical full-root FNO | `{historical_structural.best_fno_package}` / "
                    f"`{historical_structural.best_fno_family}` | "
                    f"{historical_structural.best_fno_test_root_mae:.6f} | 0.000000 |"
                ),
                (
                    f"| historical root-only tree | `{historical_structural.root_only_tree_package}` | "
                    f"{historical_structural.root_only_tree_test_root_mae:.6f} | "
                    f"{historical_structural.root_only_tree_test_root_mae - historical_structural.best_fno_test_root_mae:.6f} |"
                ),
            ]
        )
        if structural_result is not None:
            structural_lines.append(
                "| structural confirmation winner | "
                f"`{structural_result.config_label}` | "
                f"{structural_result.test_root_mae:.6f} | "
                f"{structural_result.gap_vs_best_fno:.6f} |"
            )

    md_lines = [
        "# Root-Only Tree/FNO Parity Diagnosis",
        "",
        f"- report_kind: `tree_root_only_parity_diagnosis_v1`",
        f"- train_doc_count: `{int(train_doc_count)}`",
        f"- historical_summary: `{historical_summary}`",
        "",
        "## Historical full100 root-only comparison",
        "",
        *historical_lines,
        "",
        "## Root-only diagnosis ladder",
        "",
        *ladder_lines,
        "",
        "## Conclusion",
        "",
        *conclusion_lines,
    ]
    if structural_lines:
        md_lines.extend(["", "## Structural confirmation", "", *structural_lines])
    md_lines.append("")

    report_md = output_dir / "report.md"
    report_md.write_text("\n".join(md_lines), encoding="utf-8")

    summary = {
        "report_kind": "tree_root_only_parity_diagnosis_v1",
        "train_doc_count": int(train_doc_count),
        "threshold": float(threshold),
        "classification": str(classification),
        "historical_recoverable_reference": asdict(historical_recoverable),
        "historical_structural_reference": (
            asdict(historical_structural) if historical_structural is not None else {}
        ),
        "recoverable_stage_results": {
            stage_name: (
                asdict(result) if result is not None else {}
            )
            for stage_name, result in recoverable_results.items()
        },
        "structural_confirmation_result": (
            asdict(structural_result) if structural_result is not None else {}
        ),
        "best_recoverable_stage": (
            asdict(best_recoverable_stage) if best_recoverable_stage is not None else {}
        ),
        "figures": {
            "recoverable_root_only_diagnosis_ladder": str(ladder_figure),
        },
    }
    summary_json = output_dir / "summary.json"
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    report_pdf = output_dir / "report.pdf"
    with PdfPages(report_pdf) as pdf:
        write_text_page(
            pdf,
            title="Root-Only Tree/FNO Parity Diagnosis",
            lines=[
                f"train_doc_count = {int(train_doc_count)}",
                f"classification = {classification}",
                f"threshold = {float(threshold):.6f}",
                *historical_lines,
            ],
        )
        if ladder_figure:
            write_image_page(
                pdf,
                image_path=Path(ladder_figure),
                title=f"Recoverable Root-Only Diagnosis Ladder @ train_docs={int(train_doc_count)}",
            )
        write_text_page(pdf, title="Root-Only Diagnosis Ladder", lines=ladder_lines)
        write_text_page(pdf, title="Conclusion", lines=conclusion_lines)
        if structural_lines:
            write_text_page(pdf, title="Structural Confirmation", lines=structural_lines)

    return {
        "report_md": str(report_md),
        "report_pdf": str(report_pdf),
        "summary_json": str(summary_json),
        "classification": str(classification),
        "best_recoverable_stage_name": (
            best_recoverable_stage.stage_name if best_recoverable_stage is not None else ""
        ),
    }


def main() -> int:
    from scripts._markov_report_archive import archived_report_exit

    return archived_report_exit(
        legacy_script="scripts/report_tree_root_only_parity_pdf.py",
        replacements=(
            "python3 scripts/report_markov_optimization_tradeoffs.py --summary-json <tradeoff_pipeline/tradeoff_report/summary.json>",
            "python3 scripts/report_markov_parity_self_contained.py --simulation-root <parity_root>",
        ),
        note=(
            "The root-only parity diagnosis PDF is archived. Use the v3 tradeoff report "
            "for headline evidence, or the self-contained parity appendix for a "
            "parity-grid-only view."
        ),
    )

    args = parse_args()
    result = generate_root_only_parity_report(
        historical_summary=Path(args.historical_summary),
        output_dir=Path(args.output_dir),
        train_doc_count=int(args.train_doc_count),
        threshold=float(args.threshold),
        stage_roots={
            "historical_replay": args.historical_replay_root,
            "optimization_fairness": args.optimization_fairness_root,
            "capacity_fairness": args.capacity_fairness_root,
            "matched_root": args.matched_root_root,
            "capacity_sweep": args.capacity_sweep_root,
            "representation_sweep": args.representation_sweep_root,
            "structural_confirmation": args.structural_confirm_root,
        },
    )
    print(str(result["report_pdf"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
