#!/usr/bin/env python3
"""Build a side-by-side embedding-vs-full-LLM manifesto performance report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.memory_defaults import recommend_manifesto_memory_defaults


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        rendered = float(value)
    except (TypeError, ValueError):
        return None
    if rendered != rendered:
        return None
    return float(rendered)


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _find_latest_full_llm_stats() -> Optional[Path]:
    results_base = PROJECT_ROOT / "data" / "results"
    if not results_base.exists():
        return None

    def _scan(task_dirs: List[Path]) -> Optional[Path]:
        latest: Optional[Tuple[float, Path]] = None
        for task_dir in task_dirs:
            pipeline_dir = task_dir / "training_pipeline"
            if not pipeline_dir.is_dir():
                continue
            for run_dir in pipeline_dir.iterdir():
                stats_path = run_dir / "final_stats.json"
                if not stats_path.exists():
                    continue
                mtime = float(stats_path.stat().st_mtime)
                if latest is None or mtime > latest[0]:
                    latest = (mtime, stats_path)
        if latest is None:
            return None
        return latest[1]

    manifesto_dir = results_base / "manifesto_rile"
    preferred = [manifesto_dir] if manifesto_dir.is_dir() else []
    fallback = [p for p in results_base.iterdir() if p.is_dir() and p != manifesto_dir]

    selected = _scan(preferred)
    if selected is not None:
        return selected
    return _scan(fallback)


def _artifact_row_by_id(artifact: Mapping[str, Any], scenario_id: str) -> Optional[Dict[str, Any]]:
    results = artifact.get("results")
    if not isinstance(results, list):
        return None
    for row in results:
        if not isinstance(row, dict):
            continue
        if str(row.get("id", "")).strip() == scenario_id:
            return row
    return None


def _extract_embedding_summary(
    artifact: Mapping[str, Any],
    recommendation_payload: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if recommendation_payload is None:
        recommendation_payload = recommend_manifesto_memory_defaults(artifact)

    selected_id = str(recommendation_payload.get("selected_scenario_id", "")).strip()
    selected_row = _artifact_row_by_id(artifact, selected_id) if selected_id else None
    selected_metrics = (
        dict(selected_row.get("metrics") or {})
        if isinstance(selected_row, dict)
        else {}
    )

    crosslang_row = _artifact_row_by_id(artifact, "micro_crosslang_embedding_reference")
    crosslang_metrics = (
        dict(crosslang_row.get("metrics") or {})
        if isinstance(crosslang_row, dict)
        else {}
    )

    test_mae = _safe_float(selected_metrics.get("test_rile_mae"))
    test_mae_rile = (float(test_mae) * 200.0) if test_mae is not None else None

    return {
        "selected_scenario_id": selected_id or None,
        "selection_reason": recommendation_payload.get("selection_reason"),
        "status": selected_row.get("status") if isinstance(selected_row, dict) else None,
        "wall_seconds": (
            _safe_float(selected_row.get("wall_seconds"))
            if isinstance(selected_row, dict)
            else None
        ),
        "metrics": {
            "test_rile_mae_normalized": test_mae,
            "test_rile_mae_rile_points": test_mae_rile,
            "test_delta_count": _safe_float(selected_metrics.get("test_delta_count")),
            "test_delta_improvement": _safe_float(selected_metrics.get("test_delta_improvement")),
            "val_delta_improvement": _safe_float(selected_metrics.get("val_delta_improvement")),
        },
        "crosslang_reference": {
            "status": crosslang_row.get("status") if isinstance(crosslang_row, dict) else None,
            "precision_at_1": _safe_float(crosslang_metrics.get("precision_at_1")),
            "separation": _safe_float(crosslang_metrics.get("separation")),
        },
    }


def _extract_full_llm_summary(stats: Mapping[str, Any], *, stats_path: Path) -> Dict[str, Any]:
    cfg = stats.get("config")
    if not isinstance(cfg, Mapping):
        cfg = {}
    test = stats.get("test")
    if not isinstance(test, Mapping):
        test = {}
    train = stats.get("train")
    if not isinstance(train, Mapping):
        train = {}
    inference = stats.get("inference_telemetry")
    if not isinstance(inference, Mapping):
        inference = {}

    test_mae = _safe_float(test.get("mae"))
    test_mae_rile = (float(test_mae) * 200.0) if test_mae is not None else None

    return {
        "stats_path": str(stats_path),
        "task": cfg.get("task"),
        "dataset": cfg.get("dataset"),
        "started_at": stats.get("started_at"),
        "completed_at": stats.get("completed_at"),
        "success": bool(stats.get("success", False)),
        "metrics": {
            "test_mae_normalized": test_mae,
            "test_mae_rile_points": test_mae_rile,
            "test_within_10pct": _safe_float(test.get("within_10pct")),
            "test_pearson_r": _safe_float(test.get("pearson_r")),
            "train_mae_normalized": _safe_float(train.get("mae")),
        },
        "performance": {
            "docs_per_second": _safe_float(inference.get("docs_per_second")),
            "tokens_per_second": _safe_float((inference.get("llm") or {}).get("tokens_per_second")),
        },
    }


def _build_comparison(
    embedding_summary: Mapping[str, Any],
    full_llm_summary: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "available": bool(full_llm_summary),
        "note": (
            "Comparison assumes both systems were evaluated on comparable normalized [0,1] RILE test sets."
        ),
    }
    if not full_llm_summary:
        out["reason"] = "missing_full_llm_stats"
        return out

    emb_mae = _safe_float(
        ((embedding_summary.get("metrics") or {}) if isinstance(embedding_summary, Mapping) else {}).get(
            "test_rile_mae_normalized"
        )
    )
    llm_mae = _safe_float(
        ((full_llm_summary.get("metrics") or {}) if isinstance(full_llm_summary, Mapping) else {}).get(
            "test_mae_normalized"
        )
    )
    if emb_mae is None or llm_mae is None:
        out["reason"] = "missing_test_mae"
        out["embedding_test_mae_normalized"] = emb_mae
        out["full_llm_test_mae_normalized"] = llm_mae
        return out

    mae_gap = float(emb_mae - llm_mae)
    out.update(
        {
            "reason": "ok",
            "embedding_test_mae_normalized": emb_mae,
            "full_llm_test_mae_normalized": llm_mae,
            "mae_gap_embedding_minus_full_llm": mae_gap,
            "mae_gap_rile_points": mae_gap * 200.0,
            "winner": "embedding" if mae_gap < 0 else ("full_llm" if mae_gap > 0 else "tie"),
        }
    )
    return out


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Manifesto System Comparison")
    lines.append("")
    lines.append(f"- Created UTC: `{payload.get('created_utc')}`")
    lines.append(f"- Perf artifact: `{payload.get('perf_artifact')}`")
    lines.append("")

    emb = payload.get("embedding_system")
    if isinstance(emb, Mapping):
        emb_metrics = emb.get("metrics") if isinstance(emb.get("metrics"), Mapping) else {}
        cross = emb.get("crosslang_reference") if isinstance(emb.get("crosslang_reference"), Mapping) else {}
        lines.append("## Embedding System")
        lines.append(f"- Selected scenario: `{emb.get('selected_scenario_id')}` ({emb.get('status')})")
        lines.append(
            "- Test MAE: "
            f"{emb_metrics.get('test_rile_mae_normalized')} normalized "
            f"({emb_metrics.get('test_rile_mae_rile_points')} RILE points)"
        )
        lines.append(
            "- Delta head: "
            f"count={emb_metrics.get('test_delta_count')}, "
            f"test_improvement={emb_metrics.get('test_delta_improvement')}, "
            f"val_improvement={emb_metrics.get('val_delta_improvement')}"
        )
        lines.append(
            "- Cross-lang gate: "
            f"precision@1={cross.get('precision_at_1')}, separation={cross.get('separation')}, status={cross.get('status')}"
        )
        lines.append("")

    llm = payload.get("full_llm_system")
    if isinstance(llm, Mapping):
        llm_metrics = llm.get("metrics") if isinstance(llm.get("metrics"), Mapping) else {}
        perf = llm.get("performance") if isinstance(llm.get("performance"), Mapping) else {}
        lines.append("## Full LLM System")
        lines.append(f"- Stats path: `{llm.get('stats_path')}`")
        lines.append(f"- Task/Dataset: `{llm.get('task')}` / `{llm.get('dataset')}`")
        lines.append(
            "- Test MAE: "
            f"{llm_metrics.get('test_mae_normalized')} normalized "
            f"({llm_metrics.get('test_mae_rile_points')} RILE points)"
        )
        lines.append(
            "- Test quality: "
            f"within_10pct={llm_metrics.get('test_within_10pct')}, "
            f"pearson_r={llm_metrics.get('test_pearson_r')}"
        )
        lines.append(
            "- Throughput: "
            f"docs/s={perf.get('docs_per_second')}, tokens/s={perf.get('tokens_per_second')}"
        )
        lines.append("")
    else:
        lines.append("## Full LLM System")
        lines.append("- No `final_stats.json` was provided or auto-discovered.")
        lines.append("")

    comp = payload.get("comparison")
    if isinstance(comp, Mapping):
        lines.append("## Comparison")
        lines.append(f"- Available: `{comp.get('available')}`")
        lines.append(f"- Result: `{comp.get('reason')}`")
        if comp.get("reason") == "ok":
            lines.append(
                "- MAE gap (embedding - full_llm): "
                f"{comp.get('mae_gap_embedding_minus_full_llm')} normalized "
                f"({comp.get('mae_gap_rile_points')} RILE points)"
            )
            lines.append(f"- Winner (lower MAE): `{comp.get('winner')}`")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a combined embedding-vs-full-LLM manifesto performance report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifact", type=Path, required=True, help="Perf harness result.json path.")
    parser.add_argument(
        "--recommended-defaults",
        type=Path,
        default=None,
        help="Optional recommended_defaults.json path (defaults to sibling file if present).",
    )
    parser.add_argument(
        "--full-llm-stats",
        type=Path,
        default=None,
        help="Optional full LLM final_stats.json path.",
    )
    parser.add_argument(
        "--auto-find-full-llm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-discover latest data/results/*/training_pipeline/*/final_stats.json when --full-llm-stats is absent.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path (default: sibling system_comparison.json).",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=None,
        help="Markdown output path (default: sibling system_comparison.md).",
    )
    args = parser.parse_args()

    artifact_path = args.artifact if args.artifact.is_absolute() else (Path.cwd() / args.artifact).resolve()
    if not artifact_path.exists():
        raise SystemExit(f"Artifact not found: {artifact_path}")

    defaults_path = args.recommended_defaults
    if defaults_path is None:
        candidate = artifact_path.parent / "recommended_defaults.json"
        defaults_path = candidate if candidate.exists() else None
    elif not defaults_path.is_absolute():
        defaults_path = (Path.cwd() / defaults_path).resolve()

    stats_path = args.full_llm_stats
    if stats_path is not None and not stats_path.is_absolute():
        stats_path = (Path.cwd() / stats_path).resolve()
    if stats_path is None and bool(args.auto_find_full_llm):
        stats_path = _find_latest_full_llm_stats()

    out_path = args.output
    if out_path is None:
        out_path = artifact_path.parent / "system_comparison.json"
    elif not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()

    md_path = args.markdown_out
    if md_path is None:
        md_path = artifact_path.parent / "system_comparison.md"
    elif not md_path.is_absolute():
        md_path = (Path.cwd() / md_path).resolve()

    artifact_payload = _load_json(artifact_path)
    recommendation_payload = _load_json(defaults_path) if defaults_path and defaults_path.exists() else None
    embedding_summary = _extract_embedding_summary(artifact_payload, recommendation_payload)

    full_llm_summary: Optional[Dict[str, Any]] = None
    if stats_path is not None and stats_path.exists():
        try:
            full_llm_summary = _extract_full_llm_summary(_load_json(stats_path), stats_path=stats_path)
        except Exception:
            full_llm_summary = None

    output_payload: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "perf_artifact": str(artifact_path),
        "recommended_defaults_path": str(defaults_path) if defaults_path is not None else None,
        "full_llm_stats_path": str(stats_path) if stats_path is not None else None,
        "harness_summary": dict(artifact_payload.get("summary") or {}),
        "embedding_system": embedding_summary,
        "full_llm_system": full_llm_summary,
    }
    output_payload["comparison"] = _build_comparison(embedding_summary, full_llm_summary)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_render_markdown(output_payload), encoding="utf-8")

    print(f"json={out_path}")
    print(f"markdown={md_path}")
    if full_llm_summary is None:
        print("full_llm_stats=missing")
    else:
        print(f"full_llm_stats={full_llm_summary.get('stats_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
