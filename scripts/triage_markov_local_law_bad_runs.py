#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Mapping, Optional


MERGE_ANCHOR_TITLE = "Markov high-support anchor: learned merge_mae vs undersupported"
ROOT_ANCHOR_TITLE = "Markov high-support anchor: learned root_mae beats undersupported"


@dataclass(frozen=True)
class BadRunRecord:
    category: str
    scenario: str
    source_path: str
    relative_source_path: Optional[str]
    train_docs: int
    val_docs: int
    test_docs: int
    audit_fraction: float
    local_law_weight: Optional[float]
    weighting_scheme: str
    parameterization: str
    model_family: str
    transition_log_std: Optional[float]
    min_segments: Optional[int]
    max_segments: Optional[int]
    min_seg_len: Optional[int]
    max_seg_len: Optional[int]
    root_anchor_status: str
    learned_root_mae: float
    undersupported_root_mae: float
    root_gain: float
    learned_merge_mae: float
    undersupported_merge_mae: float
    merge_excess: float
    learned_schedule_spread: float
    notes: List[str]


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _scenario_value(scenario: str, key: str) -> Optional[str]:
    prefix = f"{key}="
    for part in str(scenario).split("|"):
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def _scenario_float(scenario: str, key: str) -> Optional[float]:
    return _as_float(_scenario_value(scenario, key))


def _safe_relative(path: Path, root: Path) -> Optional[str]:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return None


def _classify_bad_run(
    *,
    val_docs: int,
    audit_fraction: float,
    merge_excess: float,
    learned_schedule_spread: float,
) -> str:
    if val_docs == 0 and audit_fraction <= 0.05 and merge_excess >= 2.0:
        return "sparse_c3_collapse_no_validation"
    if val_docs == 0 and audit_fraction <= 0.05:
        return "sparse_c3_underperformance_no_validation"
    if val_docs == 0 and learned_schedule_spread >= 10.0:
        return "schedule_instability_no_validation"
    return "merge_underperformance_other"


def _notes_for_record(
    *,
    val_docs: int,
    audit_fraction: float,
    merge_excess: float,
    learned_schedule_spread: float,
    root_gain: float,
) -> List[str]:
    notes: List[str] = []
    if val_docs == 0:
        notes.append("no validation split")
    if audit_fraction <= 0.05:
        notes.append("very sparse internal-label budget")
    if merge_excess >= 2.0:
        notes.append("severe merge/C3 collapse")
    if learned_schedule_spread >= 10.0:
        notes.append("large schedule instability")
    if root_gain > 0.0:
        notes.append("root task still improves over undersupported baseline")
    return notes


def collect_bad_run_records(expectation_json: Path) -> tuple[Path, List[BadRunRecord], Dict[str, Any]]:
    payload = json.loads(expectation_json.read_text(encoding="utf-8"))
    formal_root = Path(payload.get("input_root") or expectation_json.parent.parent).resolve()
    expectations = list(payload.get("expectations", []) or [])

    root_anchor_status_by_scenario: Dict[str, str] = {}
    for finding in expectations:
        if str(finding.get("title", "")) != ROOT_ANCHOR_TITLE:
            continue
        scenario = str(finding.get("scenario", ""))
        root_anchor_status_by_scenario[scenario] = str(finding.get("status", ""))

    scenario_to_rows: Dict[str, List[str]] = defaultdict(list)
    merge_bad_scenarios: List[Mapping[str, Any]] = []
    for finding in expectations:
        if str(finding.get("title", "")) != MERGE_ANCHOR_TITLE:
            continue
        scenario = str(finding.get("scenario", ""))
        status = str(finding.get("status", ""))
        if status not in {"warn", "fail"}:
            continue
        if "theorem_relevant=True" not in scenario or "model_family=neural" not in scenario:
            continue
        merge_bad_scenarios.append(finding)
        for row in list(finding.get("supporting_rows", []) or []):
            source_path = row.get("source_path")
            if isinstance(source_path, str) and source_path:
                scenario_to_rows[scenario].append(source_path)

    records: List[BadRunRecord] = []
    seen: set[tuple[str, str]] = set()
    for finding in merge_bad_scenarios:
        scenario = str(finding.get("scenario", ""))
        root_anchor_status = root_anchor_status_by_scenario.get(scenario, "missing")
        unique_paths = sorted(set(scenario_to_rows.get(scenario, [])))
        for source_path_str in unique_paths:
            key = (scenario, source_path_str)
            if key in seen:
                continue
            seen.add(key)
            source_path = Path(source_path_str)
            run = json.loads(source_path.read_text(encoding="utf-8"))
            cfg = dict(run.get("config", {}) or {})
            metrics = dict(run.get("metrics", {}) or {})
            learned = dict(metrics.get("learned", {}) or {})
            undersupported = dict(metrics.get("undersupported", {}) or {})

            learned_root = float(_as_float(learned.get("root_mae")) or 0.0)
            unders_root = float(_as_float(undersupported.get("root_mae")) or 0.0)
            learned_merge = float(_as_float(learned.get("merge_mae")) or 0.0)
            unders_merge = float(_as_float(undersupported.get("merge_mae")) or 0.0)
            learned_spread = float(_as_float(learned.get("schedule_spread_mean")) or 0.0)
            root_gain = float(unders_root - learned_root)
            merge_excess = float(learned_merge - unders_merge)
            val_docs = int(_as_float(cfg.get("val_docs")) or 0)
            audit_fraction = float(_as_float(cfg.get("audit_fraction")) or 0.0)

            category = _classify_bad_run(
                val_docs=val_docs,
                audit_fraction=audit_fraction,
                merge_excess=merge_excess,
                learned_schedule_spread=learned_spread,
            )
            notes = _notes_for_record(
                val_docs=val_docs,
                audit_fraction=audit_fraction,
                merge_excess=merge_excess,
                learned_schedule_spread=learned_spread,
                root_gain=root_gain,
            )
            record = BadRunRecord(
                category=category,
                scenario=scenario,
                source_path=str(source_path.resolve()),
                relative_source_path=_safe_relative(source_path, formal_root),
                train_docs=int(_as_float(cfg.get("train_docs")) or 0),
                val_docs=val_docs,
                test_docs=int(_as_float(cfg.get("test_docs")) or 0),
                audit_fraction=audit_fraction,
                local_law_weight=_scenario_float(scenario, "objective_local_law_weight"),
                weighting_scheme=str(_scenario_value(scenario, "objective_weighting_scheme") or "unknown"),
                parameterization=str(_scenario_value(scenario, "objective_parameterization") or "unknown"),
                model_family=str(cfg.get("model_family", "unknown")),
                transition_log_std=_as_float(cfg.get("transition_log_std")),
                min_segments=int(_as_float(cfg.get("min_segments")) or 0) or None,
                max_segments=int(_as_float(cfg.get("max_segments")) or 0) or None,
                min_seg_len=int(_as_float(cfg.get("min_seg_len")) or 0) or None,
                max_seg_len=int(_as_float(cfg.get("max_seg_len")) or 0) or None,
                root_anchor_status=root_anchor_status,
                learned_root_mae=learned_root,
                undersupported_root_mae=unders_root,
                root_gain=root_gain,
                learned_merge_mae=learned_merge,
                undersupported_merge_mae=unders_merge,
                merge_excess=merge_excess,
                learned_schedule_spread=learned_spread,
                notes=notes,
            )
            records.append(record)

    records.sort(
        key=lambda r: (
            str(r.category),
            -float(r.merge_excess),
            -float(r.root_gain),
            float(r.audit_fraction),
            int(r.train_docs),
            str(r.source_path),
        )
    )

    counts_by_category = Counter(r.category for r in records)
    counts_by_val_docs = Counter(r.val_docs for r in records)
    counts_by_audit_fraction = Counter(r.audit_fraction for r in records)
    summary = {
        "formal_root": str(formal_root),
        "expectation_json": str(expectation_json.resolve()),
        "n_bad_scenarios": int(len(merge_bad_scenarios)),
        "n_bad_runs": int(len(records)),
        "counts_by_category": dict(counts_by_category),
        "counts_by_val_docs": {str(k): int(v) for k, v in sorted(counts_by_val_docs.items())},
        "counts_by_audit_fraction": {str(k): int(v) for k, v in sorted(counts_by_audit_fraction.items())},
        "mean_root_gain": (
            sum(float(r.root_gain) for r in records) / float(len(records)) if records else 0.0
        ),
        "mean_merge_excess": (
            sum(float(r.merge_excess) for r in records) / float(len(records)) if records else 0.0
        ),
        "mean_schedule_spread": (
            sum(float(r.learned_schedule_spread) for r in records) / float(len(records))
            if records
            else 0.0
        ),
    }
    return formal_root, records, summary


def _write_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        dst.symlink_to(os.path.relpath(src.resolve(), start=dst.parent.resolve()))
    except OSError:
        shutil.copy2(src, dst)


def _write_readme(output_dir: Path, *, summary: Mapping[str, Any], records: List[BadRunRecord]) -> None:
    top_records = sorted(records, key=lambda r: float(r.merge_excess), reverse=True)[:10]
    counts_by_category = dict(summary.get("counts_by_category", {}) or {})
    lines: List[str] = []
    lines.append("# Markov Local-Law Optimization Triage")
    lines.append("")
    lines.append("This folder isolates the theorem-relevant Markov neural runs where the learned")
    lines.append("operator improves the root task but still underperforms the count-only baseline")
    lines.append("on the merge/C3 metric.")
    lines.append("")
    lines.append("The originals remain in place in the main rerun tree. This folder contains")
    lines.append("grouped links so the diagnostic subset is easy to inspect without confusing it")
    lines.append("with exact controls or root-only baselines.")
    lines.append("")
    lines.append("## Problem")
    lines.append("")
    lines.append("- Exact Markov path theory says the endpoint+count sketch is theorem-backed.")
    lines.append("- The count-only baseline is expected to fail compositionally.")
    lines.append("- In the learned neural lane, the root metric usually improves, but C3/merge")
    lines.append("  supervision is unstable and often worse than the count-only baseline.")
    lines.append("")
    lines.append("## Current summary")
    lines.append("")
    lines.append(f"- Bad scenarios: `{int(summary.get('n_bad_scenarios', 0))}`")
    lines.append(f"- Linked runs: `{int(summary.get('n_bad_runs', 0))}`")
    lines.append(f"- Mean root gain over undersupported: `{float(summary.get('mean_root_gain', 0.0)):.4f}`")
    lines.append(f"- Mean merge excess over undersupported: `{float(summary.get('mean_merge_excess', 0.0)):.4f}`")
    lines.append(f"- Mean learned schedule spread: `{float(summary.get('mean_schedule_spread', 0.0)):.4f}`")
    lines.append("")
    lines.append("Counts by category:")
    for category, count in sorted(counts_by_category.items()):
        lines.append(f"- `{category}`: `{int(count)}`")
    lines.append("")
    lines.append("## Most likely cause")
    lines.append("")
    lines.append("- Many of the worst runs have `val_docs = 0`, so there is no validation-based")
    lines.append("  checkpoint selection.")
    lines.append("- The neural trainer uses fixed-epoch optimization, which makes sparse C3")
    lines.append("  supervision fragile when `audit_fraction` is very small.")
    lines.append("- Large learned schedule spread often appears alongside the merge underperformance.")
    lines.append("")
    lines.append("Relevant code:")
    lines.append("- `lean3/FormalProofs/OPT/MarkovPathDGP.lean`")
    lines.append("- `src/ctreepo/sim/core/markov_changepoint_ops_count.py`")
    lines.append("- `docs/markov_theory_alignment_diagnosis_20260311.md`")
    lines.append("")
    lines.append("## Representative runs")
    lines.append("")
    for idx, record in enumerate(top_records, start=1):
        rel = record.relative_source_path or record.source_path
        lines.append(
            f"{idx}. `{record.category}` | merge excess `{record.merge_excess:.4f}` | "
            f"root gain `{record.root_gain:.4f}` | audit `{record.audit_fraction:.2f}` | "
            f"llw `{record.local_law_weight}` | `{rel}`"
        )
    lines.append("")
    lines.append("## Folder layout")
    lines.append("")
    lines.append("- `by_category/`: grouped links to all suspicious runs")
    lines.append("- `representatives/`: top merge-collapse examples")
    lines.append("- `bad_run_manifest.json`: structured manifest for all linked runs")
    lines.append("- `summary.json`: aggregate counts and means")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize_triage_folder(
    *,
    formal_root: Path,
    records: List[BadRunRecord],
    summary: Mapping[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "bad_run_manifest.json"
    summary_path = output_dir / "summary.json"
    _write_readme(output_dir, summary=summary, records=records)
    manifest_path.write_text(
        json.dumps([asdict(record) for record in records], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(dict(summary), indent=2, sort_keys=True), encoding="utf-8")

    by_category = output_dir / "by_category"
    representatives = output_dir / "representatives"
    by_category.mkdir(parents=True, exist_ok=True)
    representatives.mkdir(parents=True, exist_ok=True)

    for record in records:
        src = Path(record.source_path)
        rel = Path(record.relative_source_path or src.name)
        dst = by_category / record.category / rel
        _write_link(src, dst)

    for idx, record in enumerate(sorted(records, key=lambda r: float(r.merge_excess), reverse=True)[:20], start=1):
        src = Path(record.source_path)
        rel_name = record.relative_source_path or src.name
        safe_name = rel_name.replace("/", "__")
        dst = representatives / f"{idx:02d}__{safe_name}"
        _write_link(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a Markov local-law bad-run triage folder.")
    parser.add_argument(
        "--expectation-json",
        type=Path,
        required=True,
        help="Path to a Markov expectation report JSON, typically markov_only_expectations_v2.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the triage folder. Defaults to <expectation-json-dir>/markov_local_law_optimization_triage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expectation_json = args.expectation_json.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (expectation_json.parent / "markov_local_law_optimization_triage").resolve()
    )
    formal_root, records, summary = collect_bad_run_records(expectation_json)
    materialize_triage_folder(
        formal_root=formal_root,
        records=records,
        summary=summary,
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {
                "formal_root": str(formal_root),
                "output_dir": str(output_dir),
                "n_bad_runs": len(records),
                "counts_by_category": dict(summary.get("counts_by_category", {}) or {}),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
