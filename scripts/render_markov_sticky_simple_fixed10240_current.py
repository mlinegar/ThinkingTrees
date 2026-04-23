#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PYTHON_BIN = REPO_ROOT / "venv" / "bin" / "python"
PLOTTER_SCRIPT = REPO_ROOT / "scripts" / "plot_markov_v3_fixed_train_leaf_size_publication.py"

from scripts.launch_markov_sticky_simple_fixed10240_quick import (  # noqa: E402
    DEFAULT_SEED,
    FULL_GRID_ROOT_SHARES,
    RECOVERABLE_SCOPE_KEY,
    STRUCTURAL_GRID_KEY,
    STRUCTURAL_SCOPE_KEY,
    _known_visible_job_keys,
    _leaf_mass_package,
    _local_law_package,
    _root_only_package,
)
from scripts.run_markov_optimization_tradeoff_pipeline import (  # noqa: E402
    _aggregate_supervision_recovery_from_payloads,
    _load_ops_payloads,
)


def _normalize_rows_by_train_docs(scope_payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw_rows_by_train_docs = scope_payload.get("rows_by_train_docs") or {}
    if isinstance(raw_rows_by_train_docs, Mapping):
        return {
            str(key): dict(value or {})
            for key, value in dict(raw_rows_by_train_docs).items()
            if isinstance(value, Mapping)
        }
    normalized: Dict[str, Dict[str, Any]] = {}
    for item in list(raw_rows_by_train_docs):
        payload = dict(item or {})
        train_doc_count = int(payload.get("train_doc_count", 0) or 0)
        if train_doc_count <= 0:
            continue
        normalized[str(train_doc_count)] = payload
    return normalized


def _iter_visible_job_output_roots(output_root: Path) -> list[tuple[str, Path]]:
    visible: list[tuple[str, Path]] = []
    for key in _known_visible_job_keys(output_root):
        job_output_root = output_root / key
        if job_output_root.exists():
            visible.append((str(key), job_output_root))
    return visible


def _task_name_from_source_summary_json(path_text: str) -> str:
    normalized = str(path_text or "").strip()
    if not normalized:
        return ""
    try:
        return str(Path(normalized).parent.name)
    except Exception:
        return ""


def _patch_single_run_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    payload_map = dict(payload or {})
    aggregate_rows = [dict(item or {}) for item in list(payload_map.get("aggregate_rows") or [])]
    family_rows = [dict(item or {}) for item in list(payload_map.get("family_rows") or [])]
    runs = [dict(item or {}) for item in list(payload_map.get("runs") or [])]
    if len(runs) != 1:
        return payload_map
    run = dict(runs[0] or {})
    representative = {}
    if aggregate_rows:
        representative = dict(aggregate_rows[0] or {})
    elif family_rows:
        representative = dict(family_rows[0] or {})
    if not representative:
        return payload_map

    patched = dict(representative)
    if not str(patched.get("package_name", "") or "").strip():
        patched["package_name"] = str(
            representative.get(
                "package_name",
                run.get("package_name", ""),
            )
            or ""
        ).strip()
    if not str(patched.get("scope_key", "") or "").strip():
        patched["scope_key"] = str(
            representative.get(
                "scope_key",
                run.get("benchmark", ""),
            )
            or ""
        ).strip()
    if not str(patched.get("baseline_family", "") or "").strip():
        patched["baseline_family"] = str(
            representative.get(
                "baseline_family",
                run.get("baseline_family", ""),
            )
            or ""
        ).strip()
    if int(patched.get("train_doc_count", 0) or 0) <= 0:
        patched["train_doc_count"] = int(run.get("train_doc_count", 0) or 0)
    if int(patched.get("fixed_leaf_tokens", 0) or 0) <= 0:
        patched["fixed_leaf_tokens"] = int(run.get("fixed_leaf_tokens", 0) or 0)

    metric_fallbacks = {
        "test_root_mae_mean": run.get("test_root_mae"),
        "val_root_mae_mean": run.get("val_root_mae"),
        "train_root_mae_mean": run.get("train_root_mae"),
        "tree_test_root_mae": run.get("test_root_mae"),
        "tree_val_root_mae": run.get("val_root_mae"),
        "tree_train_root_mae": run.get("train_root_mae"),
    }
    for key, value in metric_fallbacks.items():
        if patched.get(key) in (None, ""):
            patched[key] = value

    payload_map["aggregate_rows"] = [patched]
    return payload_map


def _load_current_supervision_recovery_payloads(output_root: Path) -> list[Mapping[str, Any]]:
    return _load_current_supervision_recovery_payloads_with_overlays(
        output_root,
        overlay_output_roots=(),
    )


def _load_current_supervision_recovery_payloads_with_overlays(
    output_root: Path,
    *,
    overlay_output_roots: Sequence[Path],
) -> list[Mapping[str, Any]]:
    selected_payloads: dict[str, Mapping[str, Any]] = {}
    source_roots = [Path(output_root)] + [Path(path) for path in overlay_output_roots]
    for root_rank, current_root in enumerate(source_roots):
        for job_key, job_output_root in _iter_visible_job_output_roots(current_root):
            attempts_root = job_output_root / "supervision_recovery" / "attempts"
            if not attempts_root.exists():
                continue
            for attempt_dir in sorted(
                [path for path in attempts_root.iterdir() if path.is_dir()],
                key=lambda path: str(path.name),
            ):
                raw_root = attempt_dir / "raw"
                if not raw_root.exists():
                    continue
                for payload in _load_ops_payloads(raw_root):
                    payload_map = _patch_single_run_payload(payload)
                    payload_map["sticky_current_view_job_key"] = str(job_key)
                    payload_map["sticky_current_view_output_root"] = str(current_root)
                    payload_map["sticky_current_view_root_rank"] = int(root_rank)
                    task_name = _task_name_from_source_summary_json(
                        str(payload_map.get("source_summary_json", "") or "")
                    )
                    dedupe_key = str(task_name or payload_map.get("source_summary_json", "") or "")
                    if not dedupe_key:
                        continue
                    selected_payloads[dedupe_key] = payload_map
    return [dict(payload) for payload in selected_payloads.values()]


def _build_current_supervision_recovery_summary(
    output_root: Path,
    *,
    overlay_output_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    payloads = _load_current_supervision_recovery_payloads_with_overlays(
        output_root,
        overlay_output_roots=overlay_output_roots,
    )
    aggregate = _aggregate_supervision_recovery_from_payloads(
        payloads,
        tree_family="tree_neural",
        recoverable_benchmark=RECOVERABLE_SCOPE_KEY,
        structural_grid=STRUCTURAL_GRID_KEY,
        structural_cell=STRUCTURAL_SCOPE_KEY,
    )
    scope_payloads: dict[str, dict[str, Any]] = {}
    grouped_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    aggregate_scopes = dict(aggregate.get("scopes") or {})
    for raw_row in list(aggregate.get("family_rows") or []):
        row = dict(raw_row or {})
        scope_key = str(row.get("scope_key", "") or "").strip()
        train_doc_count = int(row.get("train_doc_count", 0) or 0)
        if not scope_key or train_doc_count <= 0:
            continue
        grouped_rows.setdefault((scope_key, train_doc_count), []).append(row)
        scope_payloads.setdefault(
            scope_key,
            {
                "scope_key": str(scope_key),
                "scope_label": str(
                    (dict(aggregate_scopes.get(scope_key) or {})).get(
                        "scope_label",
                        scope_key,
                    )
                    or scope_key
                ),
                "rows_by_train_docs": {},
                "available_train_docs": [],
                "dense_anchor_rows": [],
                "best_tree_by_train_docs": {},
            },
        )
    for (scope_key, train_doc_count), rows in sorted(grouped_rows.items()):
        sorted_rows = sorted(
            [dict(row) for row in rows],
            key=lambda row: (
                str(row.get("package_name", "") or ""),
                str(row.get("baseline_family", "") or ""),
                int(row.get("fixed_leaf_tokens", 0) or 0),
            ),
        )
        scope_payload = scope_payloads[str(scope_key)]
        scope_payload["rows_by_train_docs"][str(int(train_doc_count))] = {
            "train_doc_count": int(train_doc_count),
            "rows": sorted_rows,
        }
        scope_payload["available_train_docs"] = sorted(
            {
                *[int(value) for value in list(scope_payload.get("available_train_docs") or [])],
                int(train_doc_count),
            }
        )
        scope_payload["dense_anchor_rows"].extend(
            [
                dict(row)
                for row in sorted_rows
                if str(row.get("package_name", "") or "") == "full100"
            ]
        )
    plotter_recovery = {
        "status": "ready",
        "reason": "",
        "tree_family": str(aggregate.get("tree_family", "") or ""),
        "recoverable_scope_key": str(aggregate.get("recoverable_scope_key", "") or ""),
        "structural_scope_key": str(aggregate.get("structural_scope_key", "") or ""),
        "structural_hardness_grid": str(
            aggregate.get("structural_hardness_grid", "") or ""
        ),
        "package_order": list(aggregate.get("package_order") or []),
        "family_rows": [dict(row) for row in list(aggregate.get("family_rows") or [])],
        "scopes": scope_payloads,
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "overlay_output_roots": [str(path) for path in overlay_output_roots],
        "source_job_keys": [job_key for job_key, _ in _iter_visible_job_output_roots(output_root)],
        "payload_count": len(payloads),
        "supervision_recovery": plotter_recovery,
        "supervision_recovery_aggregate": aggregate,
    }


def _sorted_leaf_tokens(
    rows: Iterable[Mapping[str, Any]],
    *,
    package_name: str,
    baseline_family: str,
) -> list[int]:
    values = {
        int(row.get("fixed_leaf_tokens", 0) or 0)
        for row in rows
        if str(row.get("package_name", "") or "") == str(package_name)
        and str(row.get("baseline_family", "") or "") == str(baseline_family)
        and int(row.get("fixed_leaf_tokens", 0) or 0) > 0
    }
    return sorted(values, reverse=True)


def _has_leaf_token(
    rows: Iterable[Mapping[str, Any]],
    *,
    package_name: str,
    baseline_family: str,
    leaf_tokens: int,
) -> bool:
    target_leaf_tokens = int(leaf_tokens)
    for row in rows:
        if (
            str(row.get("package_name", "") or "") == str(package_name)
            and str(row.get("baseline_family", "") or "") == str(baseline_family)
            and int(row.get("fixed_leaf_tokens", 0) or 0) == target_leaf_tokens
        ):
            return True
    return False


def _build_coverage_summary(
    merged_summary: Mapping[str, Any],
    *,
    train_doc_count: int,
    root_shares: Sequence[int],
) -> dict[str, Any]:
    recovery = dict(merged_summary.get("supervision_recovery") or {})
    scopes = dict(recovery.get("scopes") or {})
    coverage: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_doc_count": int(train_doc_count),
        "root_shares": [int(value) for value in root_shares],
        "scopes": {},
    }
    for scope_key, scope_payload in scopes.items():
        rows_by_train_docs = _normalize_rows_by_train_docs(scope_payload)
        row_group = dict(rows_by_train_docs.get(str(int(train_doc_count))) or {})
        rows = [dict(row or {}) for row in list(row_group.get("rows") or [])]
        root_share_payloads: dict[str, Any] = {}
        for root_share in root_shares:
            root_share_int = int(root_share)
            root_package = _root_only_package(root_share_int)
            local_law_package = _local_law_package(root_share_int)
            mass_package = (
                _leaf_mass_package(root_share_int)
                if root_share_int < 100
                else ""
            )
            root_share_payloads[str(root_share_int)] = {
                "root_only_tree_leaf_tokens": _sorted_leaf_tokens(
                    rows,
                    package_name=root_package,
                    baseline_family="tree_neural",
                ),
                "root_only_fno_leaf128_present": _has_leaf_token(
                    rows,
                    package_name=root_package,
                    baseline_family="fno",
                    leaf_tokens=128,
                )
                or _has_leaf_token(
                    rows,
                    package_name=root_package,
                    baseline_family="official_fno",
                    leaf_tokens=128,
                ),
                "duplicate_local_leaf_tokens": _sorted_leaf_tokens(
                    rows,
                    package_name=local_law_package,
                    baseline_family="tree_neural",
                ),
                "leaf_mass_eq_leaf_tokens": (
                    _sorted_leaf_tokens(
                        rows,
                        package_name=mass_package,
                        baseline_family="tree_neural",
                    )
                    if mass_package
                    else []
                ),
            }
        coverage["scopes"][str(scope_key)] = {
            "scope_label": str(scope_payload.get("scope_label", "") or ""),
            "available_train_docs": list(scope_payload.get("available_train_docs") or []),
            "root_shares": root_share_payloads,
        }
    return coverage


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the current sticky-simple 10240 fixed-doc plots directly from "
            "landed raw supervision-recovery rows on an existing output root."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "markov_v5_simple_fixed10240_quick_20260414_utc",
    )
    parser.add_argument(
        "--overlay-output-root",
        type=Path,
        nargs="*",
        default=[],
        help=(
            "Optional later roots whose landed raw summaries should override duplicate "
            "task names from --output-root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / f"markov_v5_simple_current_plots_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--train-doc-count", type=int, default=10240)
    parser.add_argument(
        "--root-shares",
        type=int,
        nargs="+",
        default=list(FULL_GRID_ROOT_SHARES),
    )
    parser.add_argument(
        "--secondary-tree-series",
        type=str,
        default="leaf_mass_eq",
    )
    parser.add_argument(
        "--structural-scope-key",
        type=str,
        default=STRUCTURAL_SCOPE_KEY,
    )
    parser.add_argument(
        "--empirical-bayes",
        choices=("off", "collapsed_hmm"),
        default="collapsed_hmm",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else (REPO_ROOT / args.output_root)
    )
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else (REPO_ROOT / args.output_dir)
    )
    python_bin = (
        args.python_bin
        if args.python_bin.is_absolute()
        else (REPO_ROOT / args.python_bin)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay_output_roots = [
        path if path.is_absolute() else (REPO_ROOT / path)
        for path in list(args.overlay_output_root or [])
    ]
    merged_summary = _build_current_supervision_recovery_summary(
        output_root,
        overlay_output_roots=overlay_output_roots,
    )
    coverage = _build_coverage_summary(
        merged_summary,
        train_doc_count=int(args.train_doc_count),
        root_shares=list(args.root_shares),
    )

    merged_summary_path = output_dir / "merged_current_report_summary.json"
    coverage_path = output_dir / "merged_current_report_coverage.json"
    _write_json(merged_summary_path, merged_summary)
    _write_json(coverage_path, coverage)

    combined_progress_path = output_root / "combined_progress.json"
    if combined_progress_path.exists():
        snapshot_path = output_dir / "combined_progress_snapshot.json"
        snapshot_path.write_text(
            combined_progress_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    cmd = [
        str(python_bin),
        str(PLOTTER_SCRIPT),
        "--report-summary",
        str(merged_summary_path),
        "--output-dir",
        str(output_dir),
        "--train-doc-counts",
        str(int(args.train_doc_count)),
        "--root-shares",
        *[str(int(value)) for value in list(args.root_shares)],
        "--structural-scope-key",
        str(args.structural_scope_key),
    ]
    if str(args.secondary_tree_series or "").strip():
        cmd.extend(["--secondary-tree-series", str(args.secondary_tree_series)])
    if str(args.empirical_bayes or "").strip():
        cmd.extend(["--empirical-bayes", str(args.empirical_bayes)])
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
