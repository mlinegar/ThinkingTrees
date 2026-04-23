#!/usr/bin/env python3
"""Audit legacy-vs-corrected weighting views across core mergeable/HLL artifacts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ctreepo.sim.util import safe_float


DEFAULT_ARTIFACTS: Tuple[str, ...] = (
    "outputs/hll_merge_learning_summary.json",
    "outputs/hll_merge_learning_raw.csv",
    "outputs/hll_merge_learning_agg.csv",
    "outputs/mergeable_k_m_phase_summary.json",
    "outputs/mergeable_chunk_quality_sweep_summary.json",
    "outputs/mergeable_nonlanguage_suite_summary.json",
    "outputs/mergeable_nonlanguage_coverage_summary.json",
    "outputs/mergeable_complexity_ladder_summary.json",
)

VALID_MODES = ("doc", "leaf", "token")


_safe_float = safe_float


def _is_finite(x: float) -> bool:
    return bool(math.isfinite(float(x)))


def _sign(x: float, eps: float = 1e-12) -> int:
    if not _is_finite(x):
        return 0
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _is_row_like(d: dict) -> bool:
    if "weighting_views" in d:
        return True
    if "method_name" in d:
        return True
    if any(k.startswith("learned_relative_rmse_") for k in d.keys()):
        return True
    if any(k.startswith("hll_relative_rmse_") for k in d.keys()):
        return True
    return False


def _flatten_rows(obj: object, context: Optional[dict] = None) -> List[dict]:
    ctx = dict(context or {})
    out: List[dict] = []
    if isinstance(obj, dict):
        if _is_row_like(obj):
            row = dict(ctx)
            row.update(obj)
            out.append(row)
            return out
        for k, v in obj.items():
            next_ctx = dict(ctx)
            if str(k).startswith("stage"):
                next_ctx["stage"] = str(k)
            if isinstance(v, dict):
                if "method_name" not in next_ctx and "target_ks" in v and "rows" in v:
                    next_ctx["method_name"] = str(k)
            out.extend(_flatten_rows(v, next_ctx))
        return out
    if isinstance(obj, list):
        for item in obj:
            out.extend(_flatten_rows(item, ctx))
        return out
    return out


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _detect_denominator_keys(keys: Iterable[str]) -> List[str]:
    needles = (
        "n_tokens",
        "token_count",
        "token_counts",
        "n_chunks",
        "chunk_count",
        "chunks_total",
        "leaf_count",
        "leaf_counts",
    )
    out = []
    for k in keys:
        kk = str(k).lower()
        if any(n in kk for n in needles):
            out.append(str(k))
    return sorted(set(out))


def _parse_weighting_views_block(
    row: dict,
    *,
    legacy_fallback_mode: str,
) -> List[dict]:
    views = row.get("weighting_views")
    if not isinstance(views, dict):
        return []
    legacy_mode = str(row.get("legacy_weighting_mode", legacy_fallback_mode)).strip().lower()
    if legacy_mode not in VALID_MODES:
        legacy_mode = legacy_fallback_mode
    method_name = str(row.get("method_name", row.get("method", "unknown_method")))
    context_parts = []
    for k in (
        "stage",
        "scenario_name",
        "target_k",
        "sketch_order",
        "chunk_budget",
        "precision",
        "train_docs",
        "audit_policy",
    ):
        if k in row:
            context_parts.append(f"{k}={row[k]}")
    context_id = "|".join(context_parts) if context_parts else "global"
    out: List[dict] = []

    def _metric_from_mode_stats(metric_name: str, by_mode: Dict[str, dict]) -> Optional[dict]:
        if not all(m in by_mode for m in VALID_MODES):
            return None
        metric_row = {
            "metric": metric_name,
            "method_name": method_name,
            "context_id": context_id,
            "legacy_mode": legacy_mode,
        }
        for mode in VALID_MODES:
            stats = by_mode.get(mode, {})
            if not isinstance(stats, dict):
                return None
            metric_row[f"{mode}_mean_hat"] = _safe_float(stats.get("mean_hat"))
            metric_row[f"{mode}_bias"] = _safe_float(stats.get("bias"))
            metric_row[f"{mode}_rmse"] = _safe_float(stats.get("rmse"))
            metric_row[f"{mode}_mean_abs_bias"] = _safe_float(stats.get("mean_abs_bias"))
        return metric_row

    mode_keys = [k for k in views.keys() if k in VALID_MODES]
    if len(mode_keys) >= 2:
        sample_mode = views.get(mode_keys[0], {})
        if isinstance(sample_mode, dict) and "mean_hat" in sample_mode:
            metric = _metric_from_mode_stats("scalar", views)  # type: ignore[arg-type]
            if metric is not None:
                out.append(metric)
        else:
            metric_names = set()
            for m in mode_keys:
                mvals = views.get(m, {})
                if isinstance(mvals, dict):
                    metric_names.update(str(k) for k in mvals.keys())
            for metric_name in sorted(metric_names):
                by_mode: Dict[str, dict] = {}
                for m in mode_keys:
                    sub = views.get(m, {})
                    if isinstance(sub, dict) and isinstance(sub.get(metric_name), dict):
                        by_mode[m] = sub[metric_name]
                metric = _metric_from_mode_stats(metric_name, by_mode)
                if metric is not None:
                    out.append(metric)

    # HLL flattened mode columns fallback.
    for prefix in ("hll", "learned"):
        for metric_name in (
            "relative_rmse",
            "mean_abs_rel_error",
            "schedule_spread_mean",
            "schedule_spread_p95",
        ):
            row_candidate = {
                "metric": f"{prefix}_{metric_name}",
                "method_name": method_name,
                "context_id": context_id,
                "legacy_mode": legacy_mode,
            }
            ok = True
            for mode in VALID_MODES:
                key = f"{prefix}_{metric_name}_{mode}"
                if key not in row:
                    ok = False
                    break
                row_candidate[f"{mode}_mean_hat"] = _safe_float(row.get(key))
                row_candidate[f"{mode}_bias"] = _safe_float("nan")
                row_candidate[f"{mode}_rmse"] = _safe_float("nan")
                row_candidate[f"{mode}_mean_abs_bias"] = _safe_float("nan")
            if ok:
                out.append(row_candidate)

    return out


def _rank_score(entry: dict, mode: str) -> float:
    rmse = _safe_float(entry.get(f"{mode}_rmse"))
    if _is_finite(rmse):
        return rmse
    mab = _safe_float(entry.get(f"{mode}_mean_abs_bias"))
    if _is_finite(mab):
        return mab
    bias = _safe_float(entry.get(f"{mode}_bias"))
    if _is_finite(bias):
        return abs(bias)
    val = _safe_float(entry.get(f"{mode}_mean_hat"))
    if _is_finite(val):
        return abs(val)
    return float("inf")


def _check_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build weighting audit report from core legacy artifacts.")
    p.add_argument(
        "--artifacts",
        type=str,
        default=",".join(DEFAULT_ARTIFACTS),
        help="Comma-separated artifact paths (json/csv).",
    )
    p.add_argument("--legacy-fallback-mode", type=str, default="doc", choices=VALID_MODES)
    p.add_argument("--delta-threshold", type=float, default=0.05)
    p.add_argument("--active-run-pid", type=int, default=533082)
    p.add_argument("--active-run-start", type=str, default="2026-03-03T02:30:15Z")
    p.add_argument(
        "--active-run-log",
        type=str,
        default="identifiable_zero_suite_20260303_identifiable_zero_expand_v2_detached.log",
    )
    p.add_argument(
        "--active-run-output-root",
        type=str,
        default="identifiable_zero_suite_20260303_identifiable_zero_expand_v2",
    )
    p.add_argument(
        "--json-output",
        type=str,
        default="outputs/weighting_audit_report.json",
    )
    p.add_argument(
        "--markdown-output",
        type=str,
        default="outputs/weighting_audit_report.md",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = [Path(x.strip()) for x in args.artifacts.split(",") if x.strip()]
    legacy_fallback_mode = str(args.legacy_fallback_mode).strip().lower()
    delta_threshold = float(args.delta_threshold)

    manifest_rows: List[dict] = []
    metric_rows: List[dict] = []

    for path in artifacts:
        item = {
            "artifact": str(path),
            "exists": bool(path.exists()),
            "format": path.suffix.lower().lstrip("."),
            "reweightable": False,
            "needs_rerun": True,
            "row_count": 0,
            "has_weighting_views": False,
            "denominator_keys": [],
            "reason": "",
        }
        if not path.exists():
            item["reason"] = "missing"
            manifest_rows.append(item)
            continue

        if path.suffix.lower() == ".json":
            payload = _load_json(path)
            rows = _flatten_rows(payload)
            item["row_count"] = len(rows)
            all_keys = set()
            for row in rows:
                all_keys.update(str(k) for k in row.keys())
                if isinstance(row.get("weighting_views"), dict):
                    item["has_weighting_views"] = True
                metric_rows.extend(
                    {
                        **mr,
                        "artifact": str(path),
                        "family": path.stem,
                    }
                    for mr in _parse_weighting_views_block(
                        row,
                        legacy_fallback_mode=legacy_fallback_mode,
                    )
                )
            denom_keys = _detect_denominator_keys(all_keys)
            item["denominator_keys"] = denom_keys
            item["reweightable"] = bool(item["has_weighting_views"] or len(denom_keys) > 0)
            item["needs_rerun"] = not bool(item["reweightable"])
            item["reason"] = (
                "contains weighting views"
                if item["has_weighting_views"]
                else ("contains denominator-like fields" if denom_keys else "no weighting views or denominators")
            )
        elif path.suffix.lower() == ".csv":
            rows = _load_csv_rows(path)
            item["row_count"] = len(rows)
            keys = rows[0].keys() if rows else []
            denom_keys = _detect_denominator_keys(keys)
            item["denominator_keys"] = denom_keys
            item["reweightable"] = bool(len(denom_keys) > 0)
            item["needs_rerun"] = not bool(item["reweightable"])
            item["reason"] = (
                "contains denominator-like fields" if denom_keys else "no denominator-like fields"
            )
        else:
            item["reason"] = f"unsupported format: {path.suffix}"

        manifest_rows.append(item)

    delta_rows: List[dict] = []
    flagged_rows: List[dict] = []
    sign_flips: List[dict] = []
    for row in metric_rows:
        legacy_mode = str(row["legacy_mode"])
        legacy_val = _safe_float(row.get(f"{legacy_mode}_mean_hat"))
        doc_val = _safe_float(row.get("doc_mean_hat"))
        leaf_val = _safe_float(row.get("leaf_mean_hat"))
        token_val = _safe_float(row.get("token_mean_hat"))
        token_delta = token_val - legacy_val if (_is_finite(token_val) and _is_finite(legacy_val)) else float("nan")
        leaf_delta = leaf_val - legacy_val if (_is_finite(leaf_val) and _is_finite(legacy_val)) else float("nan")
        doc_delta = doc_val - legacy_val if (_is_finite(doc_val) and _is_finite(legacy_val)) else float("nan")
        out_row = {
            **row,
            "legacy_mean_hat": legacy_val,
            "doc_minus_legacy": doc_delta,
            "leaf_minus_legacy": leaf_delta,
            "token_minus_legacy": token_delta,
        }
        delta_rows.append(out_row)
        if _is_finite(token_delta) and abs(token_delta) >= delta_threshold:
            flagged_rows.append({**out_row, "flag_reason": f"|token-legacy|>={delta_threshold}"})
        legacy_bias = _safe_float(row.get(f"{legacy_mode}_bias"))
        token_bias = _safe_float(row.get("token_bias"))
        if _sign(legacy_bias) != 0 and _sign(token_bias) != 0 and _sign(legacy_bias) != _sign(token_bias):
            sign_flips.append(
                {
                    "artifact": row["artifact"],
                    "family": row["family"],
                    "context_id": row["context_id"],
                    "method_name": row["method_name"],
                    "metric": row["metric"],
                    "legacy_mode": legacy_mode,
                    "legacy_bias": legacy_bias,
                    "token_bias": token_bias,
                }
            )

    ranking_rows: List[dict] = []
    grouped: Dict[Tuple[str, str, str], List[dict]] = {}
    for row in metric_rows:
        gkey = (str(row["family"]), str(row["metric"]), str(row["context_id"]))
        grouped.setdefault(gkey, []).append(row)
    for (family, metric, context_id), group_rows in sorted(grouped.items()):
        by_method = {}
        for g in group_rows:
            by_method[str(g["method_name"])] = g
        if len(by_method) < 2:
            continue
        methods = sorted(by_method.keys())
        legacy_mode = str(group_rows[0]["legacy_mode"])
        ranked_legacy = sorted(methods, key=lambda m: _rank_score(by_method[m], legacy_mode))
        ranked_token = sorted(methods, key=lambda m: _rank_score(by_method[m], "token"))
        top2_legacy = tuple(ranked_legacy[:2])
        top2_token = tuple(ranked_token[:2])
        changed = top2_legacy != top2_token
        ranking_rows.append(
            {
                "family": family,
                "metric": metric,
                "context_id": context_id,
                "legacy_mode": legacy_mode,
                "top2_legacy": list(top2_legacy),
                "top2_token": list(top2_token),
                "top2_changed": bool(changed),
            }
        )
        if changed:
            flagged_rows.append(
                {
                    "artifact": family,
                    "family": family,
                    "context_id": context_id,
                    "metric": metric,
                    "flag_reason": "top2_ranking_changed_legacy_vs_token",
                }
            )

    active_status = {
        "pid": int(args.active_run_pid),
        "running": bool(_check_pid_alive(int(args.active_run_pid))),
        "start_time_utc": str(args.active_run_start),
        "log": str(args.active_run_log),
        "output_root": str(args.active_run_output_root),
    }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "active_background_run": active_status,
        "delta_threshold": delta_threshold,
        "legacy_weighting_mode": legacy_fallback_mode,
        "legacy_fallback_mode": legacy_fallback_mode,
        "artifact_manifest": manifest_rows,
        "delta_rows": delta_rows,
        "sign_flips": sign_flips,
        "ranking_stability": ranking_rows,
        "flagged_cells": flagged_rows,
        "counts": {
            "n_artifacts": len(manifest_rows),
            "n_reweightable": sum(1 for r in manifest_rows if r["reweightable"]),
            "n_needs_rerun": sum(1 for r in manifest_rows if r["needs_rerun"]),
            "n_metric_rows": len(metric_rows),
            "n_delta_rows": len(delta_rows),
            "n_sign_flips": len(sign_flips),
            "n_ranking_groups": len(ranking_rows),
            "n_ranking_changes": sum(1 for r in ranking_rows if r["top2_changed"]),
            "n_flagged_cells": len(flagged_rows),
        },
    }

    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Weighting Audit Report",
        "",
        "## Active Run Status",
        "",
        f"- PID: `{active_status['pid']}`",
        f"- Running: `{int(bool(active_status['running']))}`",
        f"- Start time (UTC): `{active_status['start_time_utc']}`",
        f"- Log: `{active_status['log']}`",
        f"- Output root: `{active_status['output_root']}`",
        "",
        "## Artifact Manifest",
        "",
        "| Artifact | Exists | Reweightable | Needs rerun | Rows | Reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in manifest_rows:
        lines.append(
            f"| `{r['artifact']}` | {int(bool(r['exists']))} | {int(bool(r['reweightable']))} | "
            f"{int(bool(r['needs_rerun']))} | {r['row_count']} | {r['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Summary Counts",
            "",
            f"- Artifacts: `{summary['counts']['n_artifacts']}`",
            f"- Reweightable: `{summary['counts']['n_reweightable']}`",
            f"- Needs rerun: `{summary['counts']['n_needs_rerun']}`",
            f"- Delta rows: `{summary['counts']['n_delta_rows']}`",
            f"- Sign flips: `{summary['counts']['n_sign_flips']}`",
            f"- Ranking groups: `{summary['counts']['n_ranking_groups']}`",
            f"- Ranking changes: `{summary['counts']['n_ranking_changes']}`",
            f"- Flagged cells: `{summary['counts']['n_flagged_cells']}`",
            "",
            "## Acceptance Checks",
            "",
            f"- `|token - legacy| >= {delta_threshold}` flags are included in `flagged_cells`.",
            "- Top-2 ranking changes (legacy vs token) are included in `flagged_cells`.",
            "",
            f"- JSON report: `{json_path}`",
        ]
    )
    md_path = Path(args.markdown_output)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"json_report": str(json_path), "markdown_report": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
