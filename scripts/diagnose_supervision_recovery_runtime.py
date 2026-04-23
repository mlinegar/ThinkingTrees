#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_markov_optimization_tradeoff_pipeline import (
    CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES,
    SUPERVISION_RECOVERY_PACKAGE_ORDER,
    _safe_float,
    _safe_int,
    _summarize_supervision_recovery_runtime_diagnosis,
    _supervision_recovery_runtime_row_from_payload,
)


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / f"supervision_recovery_runtime_diagnosis_{stamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan completed supervision-recovery worker summaries and emit a live runtime-speed diagnosis."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Run output root or direct supervision_recovery directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scope", nargs="*", default=())
    parser.add_argument("--package", nargs="*", default=())
    parser.add_argument("--train-docs", nargs="*", type=int, default=())
    parser.add_argument("--model-family", nargs="*", default=())
    return parser.parse_args()


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_supervision_root(input_root: Path) -> Path:
    root = input_root.expanduser().resolve()
    if root.name == "supervision_recovery":
        return root
    return root / "supervision_recovery"


def _normalize_filter(values: Sequence[Any]) -> set[str]:
    return {str(value).strip() for value in values if str(value).strip()}


def _collect_completed_runtime_rows(
    supervision_root: Path,
    *,
    scope_filter: set[str],
    package_filter: set[str],
    train_doc_filter: set[int],
    model_family_filter: set[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw_dir in sorted((supervision_root / "attempts").glob("*/raw/*")):
        summary_path = raw_dir / "summary.json"
        if not summary_path.exists():
            continue
        progress_path = raw_dir / "progress.json"
        progress = dict(_read_json(progress_path)) if progress_path.exists() else {}
        if str(progress.get("state", "")).strip().lower() != "completed":
            continue
        payload = dict(_read_json(summary_path))
        task_config = dict(payload.get("config") or {})
        package_name = str(
            task_config.get("pipeline_supervision_recovery_package", "") or ""
        ).strip()
        scope_key = str(
            task_config.get("pipeline_supervision_recovery_scope", "") or ""
        ).strip()
        train_docs_task = int(_safe_int(task_config.get("train_docs"), 0))
        if scope_filter and scope_key not in scope_filter:
            continue
        if package_filter and package_name not in package_filter:
            continue
        if train_doc_filter and train_docs_task not in train_doc_filter:
            continue
        for run in list(payload.get("runs") or []):
            if not isinstance(run, Mapping):
                continue
            row = _supervision_recovery_runtime_row_from_payload(
                payload,
                run,
                progress=progress,
            )
            baseline_family = str(row.get("baseline_family", "") or "").strip()
            if model_family_filter and baseline_family not in model_family_filter:
                continue
            if scope_filter and str(row.get("scope_key", "")) not in scope_filter:
                continue
            if package_filter and str(row.get("package_name", "")) not in package_filter:
                continue
            if train_doc_filter and int(_safe_int(row.get("train_doc_count"), 0)) not in train_doc_filter:
                continue
            rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row.get("scope_key", "")),
            int(_safe_int(row.get("train_doc_count"), 0)),
            SUPERVISION_RECOVERY_PACKAGE_ORDER.index(str(row.get("package_name", "")))
            if str(row.get("package_name", "")) in SUPERVISION_RECOVERY_PACKAGE_ORDER
            else len(SUPERVISION_RECOVERY_PACKAGE_ORDER),
            str(row.get("package_name", "")),
            str(row.get("baseline_family", "")),
            int(_safe_int(row.get("seed"), 0)),
        )
    )
    return rows


def _build_fno_context_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    fno_rows = [
        dict(row)
        for row in rows
        if str(row.get("baseline_family", "") or "").strip()
        in set(CANONICAL_FULL_DOC_FNO_BASELINE_FAMILIES)
    ]
    configs = {}
    for row in fno_rows:
        family = str(row.get("baseline_family", "") or "")
        configs.setdefault(family, 0)
        configs[family] += 1
    return {
        "completed_fno_rows": int(len(fno_rows)),
        "families": sorted(configs),
        "family_counts": configs,
    }


def _write_markdown(payload: Mapping[str, Any], output_path: Path) -> None:
    tree_rows = list(payload.get("tree_rows") or [])
    grouped_rows = list(payload.get("grouped_rows") or [])
    fno_context = dict(payload.get("fno_context") or {})
    lines: List[str] = [
        "# Supervision-Recovery Runtime Diagnosis",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"Input root: `{payload.get('input_root', '')}`",
        f"Supervision root: `{payload.get('supervision_root', '')}`",
        "",
        "## Live Diagnosis",
        (
            f"- Tree fast-path confirmed rows: "
            f"`{_safe_int(payload.get('tree_fast_path_confirmed_runs'))}` / "
            f"`{_safe_int(payload.get('tree_fast_path_confirmed_runs')) + _safe_int(payload.get('tree_partial_or_fallback_runs'))}` "
            f"({_safe_float(payload.get('tree_fast_path_completion_rate'), 0.0) * 100.0:.1f}%)."
        ),
        (
            f"- Zero steady-state H2D rate across completed tree rows: "
            f"`{_safe_float(payload.get('tree_zero_h2d_rate'), 0.0) * 100.0:.1f}%`."
        ),
        (
            f"- Median tree train-loop time per epoch: "
            f"`{_safe_float(payload.get('tree_median_train_loop_s_per_epoch'), float('nan')):.4f}s`; "
            f"per epoch per 1k docs: "
            f"`{_safe_float(payload.get('tree_median_train_loop_s_per_epoch_per_1k_docs'), float('nan')):.4f}s`."
        ),
        (
            f"- Median resident hits / dense-bucket hits / fused batches: "
            f"`{_safe_float(payload.get('tree_median_resident_store_hits'), float('nan')):.2f}` / "
            f"`{_safe_float(payload.get('tree_median_dense_bucket_hits'), float('nan')):.2f}` / "
            f"`{_safe_float(payload.get('tree_median_auto_queue_fused_batches'), float('nan')):.2f}`."
        ),
        (
            f"- Current evidence: `{payload.get('current_evidence_status', 'strict_causal_ab_proof_pending')}`. "
            "This can show the fast path is engaged and likely helping, but it is not a matched before/after proof."
        ),
        "",
        "## Tree Rows",
        "| scope | train_docs | package | seed | class | zero_h2d | resident_hits | dense_hits | fused_batches | train_s_per_epoch | train_s_per_epoch_per_1k_docs |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in tree_rows:
        if str(row.get("baseline_family", "")) != str(payload.get("tree_family", "")):
            continue
        zero_h2d = (
            _safe_float(row.get("steady_state_h2d_bytes"), 0.0) <= 0.0
            and _safe_float(row.get("steady_state_h2d_events"), 0.0) <= 0.0
        )
        lines.append(
            f"| {row.get('scope_label')} | {_safe_int(row.get('train_doc_count'))} | `{row.get('package_name')}` | "
            f"{_safe_int(row.get('seed'))} | `{row.get('fast_path_classification')}` | "
            f"{'yes' if zero_h2d else 'no'} | "
            f"{_safe_float(row.get('resident_store_hits'), 0.0):.2f} | "
            f"{_safe_float(row.get('fixed_shape_dense_bucket_store_hits'), 0.0):.2f} | "
            f"{_safe_float(row.get('auto_queue_fused_batches'), 0.0):.2f} | "
            f"{_safe_float(row.get('train_loop_s_per_epoch'), float('nan')):.4f}s | "
            f"{_safe_float(row.get('train_loop_s_per_epoch_per_1k_docs'), float('nan')):.4f}s |"
        )
    lines.extend(
        [
            "",
            "## Grouped Tree Summary",
            "| scope | train_docs | package | seeds | class | zero_h2d | train_s_per_epoch | train_s_per_epoch_per_1k_docs |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in grouped_rows:
        lines.append(
            f"| {row.get('scope_label')} | {_safe_int(row.get('train_doc_count'))} | `{row.get('package_name')}` | "
            f"{_safe_int(row.get('n_seeds_completed'))} | `{row.get('fast_path_classification')}` | "
            f"{_safe_float(row.get('zero_h2d_rate'), 0.0) * 100.0:.1f}% | "
            f"{_safe_float(row.get('train_loop_s_per_epoch_median'), float('nan')):.4f}s | "
            f"{_safe_float(row.get('train_loop_s_per_epoch_per_1k_docs_median'), float('nan')):.4f}s |"
        )
    lines.extend(
        [
            "",
            "## FNO Context",
            (
                f"- Completed FNO rows: `{_safe_int(fno_context.get('completed_fno_rows'))}`; "
                f"families: `{', '.join(str(item) for item in fno_context.get('families', []))}`."
            ),
            "- FNO rows are listed for context only and are excluded from the tree batching speed claims.",
            "",
            "## A/B Proof Template",
            "- Use `scripts/benchmark_supervision_recovery_runtime_ablation.py` for the strict matched before/after proof.",
            "- Compare the same scope/package/train_docs/seed slices under baseline `structure_bucket + exact_then_bucketed` and optimized `resident + fixed_fused + leaf_count_auto_queue`.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    supervision_root = _resolve_supervision_root(args.input_root)
    if not supervision_root.exists():
        raise SystemExit(f"supervision_recovery root not found: {supervision_root}")
    output_dir = args.output_dir or _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _collect_completed_runtime_rows(
        supervision_root,
        scope_filter=_normalize_filter(args.scope),
        package_filter=_normalize_filter(args.package),
        train_doc_filter={int(value) for value in list(args.train_docs or []) if int(value) > 0},
        model_family_filter=_normalize_filter(args.model_family),
    )
    payload = _summarize_supervision_recovery_runtime_diagnosis(rows)
    payload.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_root": str(args.input_root.expanduser().resolve()),
            "supervision_root": str(supervision_root),
            "filters": {
                "scope": sorted(_normalize_filter(args.scope)),
                "package": sorted(_normalize_filter(args.package)),
                "train_docs": sorted(
                    int(value) for value in list(args.train_docs or []) if int(value) > 0
                ),
                "model_family": sorted(_normalize_filter(args.model_family)),
            },
            "fno_context": _build_fno_context_summary(rows),
        }
    )
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "report.md"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(payload, markdown_path)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary_json": str(summary_path),
                "markdown": str(markdown_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
