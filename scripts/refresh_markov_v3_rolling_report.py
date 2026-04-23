#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = REPO_ROOT / "outputs"
ROLLING_CURRENT_PATH = OUTPUTS_ROOT / "markov_v3_rolling_partial_report_current"
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "run_markov_optimization_tradeoff_pipeline.py"
REPORT_SCRIPT = REPO_ROOT / "scripts" / "report_markov_optimization_tradeoffs.py"
EXCLUDED_ROOT_PREFIXES = (
    "markov_v3_depth_equal_optimized_shadow_",
    "markov_v3_rolling_partial_report_",
    "markov_v3_completed_redistribution_report",
    "markov_v3_fresh_completed_exploratory_report",
)


@dataclass(frozen=True)
class PanelSpec:
    panel_key: str
    title: str
    figure_name: str
    required_bundle_groups: tuple[str, ...]
    placeholder_message: str


def _panel_specs() -> List[PanelSpec]:
    specs: List[PanelSpec] = []
    for leaf in (128, 64, 32, 16, 8):
        leaf_tag = f"leaf{int(leaf):03d}"
        specs.extend(
            [
                PanelSpec(
                    panel_key=f"recoverable_ordered_families_{leaf_tag}",
                    title=f"Recoverable Ordered Families ({leaf_tag})",
                    figure_name=f"recoverable_ordered_families_{leaf_tag}.png",
                    required_bundle_groups=_panel_groups_for_leaf(leaf, ordered=True),
                    placeholder_message="Awaiting bundle summary for recoverable ordered-family history.",
                ),
                PanelSpec(
                    panel_key=f"structural_ordered_families_{leaf_tag}",
                    title=f"Structural Ordered Families ({leaf_tag})",
                    figure_name=f"structural_ordered_families_{leaf_tag}.png",
                    required_bundle_groups=_panel_groups_for_leaf(leaf, ordered=True),
                    placeholder_message="Awaiting bundle summary for structural ordered-family history.",
                ),
                PanelSpec(
                    panel_key=f"recoverable_package_ladder_{leaf_tag}",
                    title=f"Recoverable Package Ladder ({leaf_tag})",
                    figure_name=f"recoverable_package_ladder_{leaf_tag}.png",
                    required_bundle_groups=_panel_groups_for_leaf(leaf, package_ladder=True),
                    placeholder_message="Awaiting bundle summary for recoverable package-ladder coverage.",
                ),
                PanelSpec(
                    panel_key=f"structural_package_ladder_{leaf_tag}",
                    title=f"Structural Package Ladder ({leaf_tag})",
                    figure_name=f"structural_package_ladder_{leaf_tag}.png",
                    required_bundle_groups=_panel_groups_for_leaf(leaf, package_ladder=True),
                    placeholder_message="Awaiting bundle summary for structural package-ladder coverage.",
                ),
                PanelSpec(
                    panel_key=f"recoverable_tree_constant_density_root_ladders_{leaf_tag}",
                    title=f"Recoverable Tree Constant-Density Root Ladders ({leaf_tag})",
                    figure_name=f"recoverable_tree_constant_density_root_ladders_{leaf_tag}.png",
                    required_bundle_groups=_panel_groups_for_leaf(leaf, root_ladder=True),
                    placeholder_message="Awaiting bundle summary for recoverable constant-density ladders.",
                ),
                PanelSpec(
                    panel_key=f"structural_tree_constant_density_root_ladders_{leaf_tag}",
                    title=f"Structural Tree Constant-Density Root Ladders ({leaf_tag})",
                    figure_name=f"structural_tree_constant_density_root_ladders_{leaf_tag}.png",
                    required_bundle_groups=_panel_groups_for_leaf(leaf, root_ladder=True),
                    placeholder_message="Awaiting bundle summary for structural constant-density ladders.",
                ),
                PanelSpec(
                    panel_key=f"recoverable_dense_local_root_ladder_{leaf_tag}",
                    title=f"Recoverable Dense Local Root Ladder ({leaf_tag})",
                    figure_name=f"recoverable_dense_local_root_ladder_{leaf_tag}.png",
                    required_bundle_groups=_panel_groups_for_leaf(leaf, dense_local=True),
                    placeholder_message="Awaiting bundle summary for recoverable dense local-law ladders.",
                ),
                PanelSpec(
                    panel_key=f"structural_dense_local_root_ladder_{leaf_tag}",
                    title=f"Structural Dense Local Root Ladder ({leaf_tag})",
                    figure_name=f"structural_dense_local_root_ladder_{leaf_tag}.png",
                    required_bundle_groups=_panel_groups_for_leaf(leaf, dense_local=True),
                    placeholder_message="Awaiting bundle summary for structural dense local-law ladders.",
                ),
                PanelSpec(
                    panel_key=f"recoverable_tree_diagnostics_{leaf_tag}",
                    title=f"Recoverable Tree Diagnostics ({leaf_tag})",
                    figure_name=f"recoverable_tree_diagnostics_{leaf_tag}.png",
                    required_bundle_groups=_panel_groups_for_leaf(leaf, diagnostics=True),
                    placeholder_message="Awaiting bundle summary for recoverable tree diagnostics.",
                ),
                PanelSpec(
                    panel_key=f"structural_tree_diagnostics_{leaf_tag}",
                    title=f"Structural Tree Diagnostics ({leaf_tag})",
                    figure_name=f"structural_tree_diagnostics_{leaf_tag}.png",
                    required_bundle_groups=_panel_groups_for_leaf(leaf, diagnostics=True),
                    placeholder_message="Awaiting bundle summary for structural tree diagnostics.",
                ),
                PanelSpec(
                    panel_key=f"recoverable_r10_local_ablations_{leaf_tag}",
                    title=f"Recoverable R10 Local Ablations ({leaf_tag})",
                    figure_name=f"recoverable_r10_local_ablations_{leaf_tag}.png",
                    required_bundle_groups=_panel_groups_for_leaf(leaf, ablations=True),
                    placeholder_message="Awaiting bundle summary for recoverable local-ablation coverage.",
                ),
                PanelSpec(
                    panel_key=f"structural_r10_local_ablations_{leaf_tag}",
                    title=f"Structural R10 Local Ablations ({leaf_tag})",
                    figure_name=f"structural_r10_local_ablations_{leaf_tag}.png",
                    required_bundle_groups=_panel_groups_for_leaf(leaf, ablations=True),
                    placeholder_message="Awaiting bundle summary for structural local-ablation coverage.",
                ),
            ]
        )
    specs.extend(
        [
            PanelSpec(
                panel_key="recoverable_r10_leaf_geometry",
                title="Recoverable R10 Leaf Geometry",
                figure_name="recoverable_r10_leaf_geometry.png",
                required_bundle_groups=("main_grid_followup", "leaf_only_redistribution", "depth_equal_redistribution"),
                placeholder_message="Awaiting bundle summary for recoverable leaf-geometry coverage.",
            ),
            PanelSpec(
                panel_key="recoverable_r20_leaf_geometry",
                title="Recoverable R20 Leaf Geometry",
                figure_name="recoverable_r20_leaf_geometry.png",
                required_bundle_groups=("main_grid_followup", "leaf_only_redistribution", "depth_equal_redistribution"),
                placeholder_message="Awaiting bundle summary for recoverable leaf-geometry coverage.",
            ),
            PanelSpec(
                panel_key="recoverable_r80_leaf_geometry",
                title="Recoverable R80 Leaf Geometry",
                figure_name="recoverable_r80_leaf_geometry.png",
                required_bundle_groups=("main_grid_followup", "leaf_only_redistribution", "depth_equal_redistribution"),
                placeholder_message="Awaiting bundle summary for recoverable leaf-geometry coverage.",
            ),
            PanelSpec(
                panel_key="recoverable_r90_leaf_geometry",
                title="Recoverable R90 Leaf Geometry",
                figure_name="recoverable_r90_leaf_geometry.png",
                required_bundle_groups=("main_grid_followup", "leaf_only_redistribution", "depth_equal_redistribution"),
                placeholder_message="Awaiting bundle summary for recoverable leaf-geometry coverage.",
            ),
            PanelSpec(
                panel_key="recoverable_r100_leaf_geometry",
                title="Recoverable R100 Leaf Geometry",
                figure_name="recoverable_r100_leaf_geometry.png",
                required_bundle_groups=("main_grid_followup", "leaf_only_redistribution", "depth_equal_redistribution"),
                placeholder_message="Awaiting bundle summary for recoverable leaf-geometry coverage.",
            ),
            PanelSpec(
                panel_key="structural_r100_leaf_geometry",
                title="Structural R100 Leaf Geometry",
                figure_name="structural_r100_leaf_geometry.png",
                required_bundle_groups=("main_grid_followup", "leaf_only_redistribution", "depth_equal_redistribution"),
                placeholder_message="Awaiting bundle summary for structural leaf-geometry coverage.",
            ),
            PanelSpec(
                panel_key="structural_r80_leaf_geometry",
                title="Structural R80 Leaf Geometry",
                figure_name="structural_r80_leaf_geometry.png",
                required_bundle_groups=("main_grid_followup", "leaf_only_redistribution", "depth_equal_redistribution"),
                placeholder_message="Awaiting bundle summary for structural leaf-geometry coverage.",
            ),
            PanelSpec(
                panel_key="structural_r90_leaf_geometry",
                title="Structural R90 Leaf Geometry",
                figure_name="structural_r90_leaf_geometry.png",
                required_bundle_groups=("main_grid_followup", "leaf_only_redistribution", "depth_equal_redistribution"),
                placeholder_message="Awaiting bundle summary for structural leaf-geometry coverage.",
            ),
        ]
    )
    return specs


def _panel_groups_for_leaf(
    leaf_tokens: int,
    *,
    ordered: bool = False,
    package_ladder: bool = False,
    root_ladder: bool = False,
    dense_local: bool = False,
    diagnostics: bool = False,
    ablations: bool = False,
) -> tuple[str, ...]:
    if leaf_tokens >= 128:
        return ("oneleaf_root_budget", "structural_oneleaf_rescue")
    if leaf_tokens >= 64:
        groups = ["multileaf_root_budget", "leaf_only_redistribution", "depth_equal_redistribution"]
        if dense_local or diagnostics or ablations:
            groups.extend(("local_law_publication", "r100_superset_local10", "main_grid_followup"))
        return tuple(dict.fromkeys(groups))
    groups = [
        "multileaf_root_budget",
        "leaf_only_redistribution",
        "depth_equal_redistribution",
        "local_law_publication",
        "r100_superset_local10",
        "main_grid_followup",
        "gamma_protocol_ablation",
        "structural_oneleaf_rescue",
    ]
    if dense_local or diagnostics:
        groups.extend(("quickcheck_local_law",))
    return tuple(dict.fromkeys(groups))


PANEL_SPECS: tuple[PanelSpec, ...] = tuple(_panel_specs())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the full Markov v3 rolling report from all discovered bundle summaries."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_ROOT
        / f"markov_v3_full_rolling_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--refresh-completed-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refresh completed bundles that have raw worker outputs but no supervision_recovery summary.",
    )
    parser.add_argument(
        "--plan-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _root_prefix_from_root_name(root_name: str) -> str:
    normalized = str(root_name or "").strip()
    match = re.match(r"^(.*_)\d{8}_\d{4,6}$", normalized)
    if match:
        return str(match.group(1))
    return normalized


def _source_tier_for_root(root_name: str, bundle_name: str) -> str:
    root_text = str(root_name or "").lower()
    bundle_text = str(bundle_name or "").lower()
    if "publication_fullval_v3" in root_text:
        return "publication_fullval_v3"
    if "overnight_xlarge" in root_text or "publication_xlarge" in bundle_text:
        return "publication_xlarge"
    if "publication_fullval" in root_text:
        return "publication_fullval"
    if any(token in root_text or token in bundle_text for token in ("ablation", "check_basics", "one_leaf_parity", "one_leaf_canary")):
        return "protocol_ablation"
    return "exploratory"


def _bundle_group(bundle_name: str) -> str:
    name = str(bundle_name or "").strip()
    if name.startswith("oneleaf_root_budget"):
        return "oneleaf_root_budget"
    if name.startswith("root_budget"):
        return "multileaf_root_budget"
    if "mass_preserving_leaf_only" in name or "leaf_only_publication" in name:
        return "leaf_only_redistribution"
    if "mass_preserving_depth_equal" in name or "depth_equal_publication" in name:
        return "depth_equal_redistribution"
    if "local_law" in name or "multileaf_full_laws" in name or "root_only" in name:
        return "local_law_publication"
    if "superset_local10" in name or "r100_superset" in name:
        return "r100_superset_local10"
    if "structural_oneleaf" in name:
        return "structural_oneleaf_rescue"
    if any(token in name for token in ("gamma", "preset_ablation", "check_basics")):
        return "gamma_protocol_ablation"
    if any(token in name for token in ("quick", "duplicate_local")):
        return "quickcheck_local_law"
    return "main_grid_followup"


def _affected_panels_for_group(bundle_group: str) -> List[str]:
    return [
        spec.panel_key
        for spec in PANEL_SPECS
        if str(bundle_group) in set(spec.required_bundle_groups)
    ]


def _bundle_state(bundle_root: Path) -> Dict[str, Any]:
    status_path = bundle_root / "experiment_status.json"
    if not status_path.exists():
        return {"state": "missing", "active_items": 0}
    payload = _load_json(status_path)
    return {
        "state": str(payload.get("state", "") or ""),
        "completed_items": int(payload.get("completed_items", 0) or 0),
        "failed_items": int(payload.get("failed_items", 0) or 0),
        "active_items": int(payload.get("active_items", 0) or 0),
        "pending_items": int(payload.get("pending_items", 0) or 0),
    }


def _has_raw_outputs(bundle_root: Path) -> bool:
    attempts_root = bundle_root / "supervision_recovery" / "attempts"
    if not attempts_root.exists():
        return False
    return any(attempts_root.rglob("raw/*/summary.json"))


def _refresh_bundle(bundle_root: Path) -> subprocess.CompletedProcess[str]:
    argv = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--output-root",
        str(bundle_root),
        "--phases",
        "supervision_recovery,report",
        "--refresh-existing-output-root",
        "--device-mode",
        "cpu",
    ]
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        argv,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _replace_current_symlink(target: Path) -> None:
    if ROLLING_CURRENT_PATH.is_symlink() or ROLLING_CURRENT_PATH.exists():
        if ROLLING_CURRENT_PATH.is_dir() and not ROLLING_CURRENT_PATH.is_symlink():
            raise RuntimeError(
                f"refusing to replace non-symlink directory {ROLLING_CURRENT_PATH}"
            )
        ROLLING_CURRENT_PATH.unlink()
    rel_target = os.path.relpath(str(target), start=str(ROLLING_CURRENT_PATH.parent))
    ROLLING_CURRENT_PATH.symlink_to(rel_target)


def _is_excluded_root(root_name: str) -> bool:
    return any(str(root_name).startswith(prefix) for prefix in EXCLUDED_ROOT_PREFIXES)


def _discover_v3_bundle_records(outputs_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for status_path in sorted(outputs_root.glob("markov_v3_*/*/experiment_status.json")):
        bundle_root = status_path.parent
        root = bundle_root.parent
        root_name = str(root.name)
        bundle_name = str(bundle_root.name)
        if _is_excluded_root(root_name):
            continue
        key = (root_name, bundle_name)
        if key in seen:
            continue
        seen.add(key)
        state = _bundle_state(bundle_root)
        root_prefix = _root_prefix_from_root_name(root_name)
        bundle_group = _bundle_group(bundle_name)
        summary_path = bundle_root / "supervision_recovery" / "summary.json"
        records.append(
            {
                "bundle_label": f"{root_name}/{bundle_name}",
                "bundle_name": bundle_name,
                "bundle_root": str(bundle_root),
                "root_name": root_name,
                "root_prefix": root_prefix,
                "bundle_group": bundle_group,
                "source_tier": _source_tier_for_root(root_name, bundle_name),
                "attempt_lineage": f"{root_name}/{bundle_name}",
                "summary_path": str(summary_path),
                "summary_ready": bool(summary_path.exists()),
                "affected_panels": _affected_panels_for_group(bundle_group),
                **state,
            }
        )
    return records


def _bundle_record_sort_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return (str(record.get("root_name", "")), str(record.get("bundle_name", "")))


def _figure_path_by_basename(summary: Mapping[str, Any]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for raw_path in dict(summary.get("figures") or {}).values():
        figure_path = Path(str(raw_path)).expanduser()
        mapping[figure_path.name] = figure_path
    return mapping


def _write_placeholder_panel(
    output_path: Path,
    *,
    title: str,
    message: str,
    missing_bundles: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.axis("off")
    lines = [str(message).strip()]
    if missing_bundles:
        lines.append("")
        lines.append("Missing contributing bundles:")
        lines.extend(f"- {bundle}" for bundle in missing_bundles[:10])
        if len(missing_bundles) > 10:
            lines.append(f"- ... and {len(missing_bundles) - 10} more")
    fig.suptitle(str(title), fontsize=12)
    ax.text(
        0.02,
        0.92,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        wrap=True,
    )
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _materialize_panel_slots(
    output_dir: Path,
    *,
    summary: Mapping[str, Any],
    bundle_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    figures_dir = output_dir / "figures"
    panel_dir = figures_dir / "panels"
    figure_by_name = _figure_path_by_basename(summary)
    coverage: Dict[str, Dict[str, Any]] = {}
    for spec in PANEL_SPECS:
        required_groups = set(spec.required_bundle_groups)
        contributing = [
            dict(record)
            for record in bundle_records
            if str(record.get("bundle_group", "")) in required_groups
        ]
        included = [
            record["bundle_label"]
            for record in contributing
            if bool(record.get("summary_ready"))
        ]
        missing = [
            record["bundle_label"]
            for record in contributing
            if not bool(record.get("summary_ready"))
        ]
        target_path = panel_dir / spec.figure_name
        source_figure = figure_by_name.get(spec.figure_name)
        if source_figure is not None and source_figure.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_figure, target_path)
            status = "ready"
        else:
            _write_placeholder_panel(
                target_path,
                title=spec.title,
                message=spec.placeholder_message,
                missing_bundles=missing,
            )
            status = "placeholder"
        coverage[spec.panel_key] = {
            "panel_key": spec.panel_key,
            "title": spec.title,
            "figure_name": spec.figure_name,
            "panel_path": str(target_path),
            "status": status,
            "required_bundle_groups": list(spec.required_bundle_groups),
            "included_bundles": included,
            "missing_bundles": missing,
        }
    return coverage


def _append_completeness_markdown(
    markdown_path: Path,
    *,
    completeness: Mapping[str, Any],
    panel_coverage: Mapping[str, Mapping[str, Any]],
    hidden_invalid_row_count: int,
    hidden_invalid_sources: Sequence[str],
    hidden_invalid_reasons: Sequence[str],
) -> None:
    lines = markdown_path.read_text(encoding="utf-8").rstrip().splitlines()
    lines.extend(["", "## Rolling Completeness"])
    lines.append(
        f"- Included bundle summaries: {int(completeness.get('included_bundle_count', 0))}."
    )
    lines.append(
        f"- Running bundles without summary: {int(completeness.get('running_bundle_count', 0))}."
    )
    lines.append(
        f"- Completed bundles still missing summary: {int(completeness.get('completed_missing_bundle_count', 0))}."
    )
    if hidden_invalid_row_count > 0:
        lines.append(
            f"- Hidden invalid or diagnostic-only rows excluded from plots: {hidden_invalid_row_count}."
        )
        if hidden_invalid_sources:
            lines.append(
                f"- Hidden invalid sources: {', '.join(str(source) for source in hidden_invalid_sources)}."
            )
        if hidden_invalid_reasons:
            lines.append(
                f"- Hidden invalid reasons: {', '.join(str(reason) for reason in hidden_invalid_reasons)}."
            )
    for item in list(completeness.get("running_bundles_without_summary") or []):
        lines.append(
            f"- Running bundle awaiting summary: `{item['bundle_label']}`"
            + (
                f" (active={int(item.get('active_items', 0))}, pending={int(item.get('pending_items', 0))})."
            )
        )
    for item in list(completeness.get("completed_bundles_missing_summary") or []):
        lines.append(
            f"- Completed bundle missing summary: `{item['bundle_label']}`"
            + (f" — {item['reason']}." if str(item.get("reason", "")).strip() else ".")
        )
    lines.extend(["", "## Panel Coverage"])
    for panel_key, item in sorted(panel_coverage.items()):
        lines.append(
            f"- `{panel_key}`: {item['status']} @ `{item['panel_path']}`."
        )
        if list(item.get("missing_bundles") or []):
            lines.append(
                f"  Missing contributors: {', '.join(str(bundle) for bundle in item['missing_bundles'])}."
            )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    bundle_records = _discover_v3_bundle_records(OUTPUTS_ROOT)
    refresh_attempts: List[Dict[str, Any]] = []
    summary_paths: List[Path] = []

    for record in bundle_records:
        bundle_root = Path(str(record["bundle_root"]))
        summary_path = Path(str(record["summary_path"]))
        if (
            not bool(record.get("summary_ready"))
            and args.refresh_completed_missing
            and int(record.get("active_items", 0)) <= 0
            and _has_raw_outputs(bundle_root)
        ):
            result = _refresh_bundle(bundle_root)
            refresh_attempts.append(
                {
                    "bundle_root": str(bundle_root),
                    "returncode": int(result.returncode),
                    "stdout": str(result.stdout or ""),
                    "stderr": str(result.stderr or ""),
                }
            )
            record["summary_ready"] = bool(summary_path.exists())
            if not bool(record["summary_ready"]) and result.returncode != 0:
                record["reason"] = "refresh_existing_output_root failed"
        elif not bool(record.get("summary_ready")):
            record["reason"] = (
                "bundle is still running"
                if int(record.get("active_items", 0)) > 0
                else "bundle summary missing"
            )
        if bool(record.get("summary_ready")) and summary_path.exists():
            if summary_path not in summary_paths:
                summary_paths.append(summary_path)

    completeness = {
        "included_bundle_count": int(sum(1 for record in bundle_records if bool(record.get("summary_ready")))),
        "missing_bundle_count": int(sum(1 for record in bundle_records if not bool(record.get("summary_ready")))),
        "running_bundle_count": int(
            sum(
                1
                for record in bundle_records
                if not bool(record.get("summary_ready"))
                and int(record.get("active_items", 0)) > 0
            )
        ),
        "completed_missing_bundle_count": int(
            sum(
                1
                for record in bundle_records
                if not bool(record.get("summary_ready"))
                and int(record.get("active_items", 0)) <= 0
            )
        ),
        "included_bundle_summaries": [
            dict(record)
            for record in sorted(bundle_records, key=_bundle_record_sort_key)
            if bool(record.get("summary_ready"))
        ],
        "missing_bundle_summaries": [
            dict(record)
            for record in sorted(bundle_records, key=_bundle_record_sort_key)
            if not bool(record.get("summary_ready"))
        ],
        "running_bundles_without_summary": [
            dict(record)
            for record in sorted(bundle_records, key=_bundle_record_sort_key)
            if not bool(record.get("summary_ready"))
            and int(record.get("active_items", 0)) > 0
        ],
        "completed_bundles_missing_summary": [
            dict(record)
            for record in sorted(bundle_records, key=_bundle_record_sort_key)
            if not bool(record.get("summary_ready"))
            and int(record.get("active_items", 0)) <= 0
        ],
        "refresh_attempts": refresh_attempts,
        "potentially_incomplete_panels": sorted(
            {
                panel
                for record in bundle_records
                if not bool(record.get("summary_ready"))
                for panel in list(record.get("affected_panels") or [])
            }
        ),
    }

    if args.plan_only:
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "bundle_records": bundle_records,
                    "summary_paths": [str(path) for path in summary_paths],
                    "completeness": completeness,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not summary_paths:
        raise SystemExit("no completed supervision_recovery summaries available for the rolling report")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, str(REPORT_SCRIPT), "--output-dir", str(args.output_dir)]
    for summary_path in summary_paths:
        argv.extend(["--supervision-recovery-summary", str(summary_path)])
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    subprocess.run(argv, cwd=REPO_ROOT, env=env, check=True)

    summary_path = args.output_dir / "summary.json"
    markdown_path = args.output_dir / "report.md"
    summary = _load_json(summary_path)

    panel_coverage = _materialize_panel_slots(
        args.output_dir,
        summary=summary,
        bundle_records=bundle_records,
    )

    hidden_invalid_row_count = int(summary.get("hidden_invalid_row_count", 0) or 0)
    hidden_invalid_sources = list(summary.get("hidden_invalid_sources") or [])
    hidden_invalid_reasons = list(summary.get("hidden_invalid_reasons") or [])
    summary["alignment_contract_gate_status"] = str(
        summary.get(
            "alignment_contract_gate_status",
            summary.get("contract_gate_status", "pass"),
        )
        or "pass"
    )
    summary["contract_gate_status"] = str(
        dict(summary.get("supervision_recovery") or {}).get(
            "contract_gate_status",
            summary.get("contract_gate_status", "pass"),
        )
        or "pass"
    )

    summary["included_bundle_count"] = completeness["included_bundle_count"]
    summary["missing_bundle_count"] = completeness["missing_bundle_count"]
    summary["running_bundle_count"] = completeness["running_bundle_count"]
    summary["completed_missing_bundle_count"] = completeness["completed_missing_bundle_count"]
    summary["included_bundle_summaries"] = completeness["included_bundle_summaries"]
    summary["missing_bundle_summaries"] = completeness["missing_bundle_summaries"]
    summary["running_bundles_without_summary"] = completeness["running_bundles_without_summary"]
    summary["completed_bundles_missing_summary"] = completeness["completed_bundles_missing_summary"]
    summary["panel_coverage"] = panel_coverage
    summary["rolling_report_note"] = (
        "This full rolling report is bundle-summary driven. Live worker cells only appear after their parent bundle publishes supervision_recovery/summary.json."
    )
    summary["rolling_report_completeness"] = completeness
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest_path = args.output_dir / "rolling_report_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "discovered_bundles": sorted(bundle_records, key=_bundle_record_sort_key),
                "included_bundles": completeness["included_bundle_summaries"],
                "missing_bundles": completeness["missing_bundle_summaries"],
                "hidden_invalid_row_count": hidden_invalid_row_count,
                "hidden_invalid_sources": hidden_invalid_sources,
                "hidden_invalid_reasons": hidden_invalid_reasons,
                "panel_coverage": panel_coverage,
                "lineage_labels": list(summary.get("lineage_labels") or []),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _append_completeness_markdown(
        markdown_path,
        completeness=completeness,
        panel_coverage=panel_coverage,
        hidden_invalid_row_count=hidden_invalid_row_count,
        hidden_invalid_sources=hidden_invalid_sources,
        hidden_invalid_reasons=hidden_invalid_reasons,
    )
    _replace_current_symlink(args.output_dir)

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "summary_json": str(summary_path),
                "rolling_report_manifest": str(manifest_path),
                "markdown": str(markdown_path),
                "current_symlink": str(ROLLING_CURRENT_PATH),
                "included_bundle_count": completeness["included_bundle_count"],
                "missing_bundle_count": completeness["missing_bundle_count"],
                "panel_count": len(panel_coverage),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
