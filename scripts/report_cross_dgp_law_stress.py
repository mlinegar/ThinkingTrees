#!/usr/bin/env python3
"""Cross-DGP law-stress summary: reads Markov and LDA report outputs to produce
a single comparison table showing that local laws → learnable g across DGPs.

Usage:
    python scripts/report_cross_dgp_law_stress.py \
        --markov-dir outputs/markov_law_stress_report \
        --lda-dir outputs/lda_law_stress_report \
        --output-dir outputs/cross_dgp_law_stress_report
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
import sys
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DGPSummaryRow:
    dgp: str
    law_package: str
    n_runs: int
    primary_pass_rate: float
    c1_pass_rate: float
    c2_pass_rate: float
    c3_pass_rate: float
    mean_laws_improved: float
    mean_primary_gain: float
    # Kept for backward compat
    bundle_success_rate: float = 0.0
    root_pass_rate: float = 0.0
    downstream_correlation: float = float("nan")


def _load_markov_rows(markov_dir: Path) -> List[DGPSummaryRow]:
    """Load assessed rows from Markov law-stress report."""
    # Try aggregated CSV first (has per-package summary)
    agg_csv = markov_dir / "markov_law_stress_aggregated_rows.csv"
    assessed_csv = markov_dir / "markov_law_stress_assessed_rows.csv"

    if agg_csv.exists():
        return _load_markov_from_aggregated(agg_csv)
    if assessed_csv.exists():
        return _load_markov_from_assessed(assessed_csv)

    # Fall back to summary JSON
    summary_json = markov_dir / "markov_law_stress_summary.json"
    if summary_json.exists():
        return _load_markov_from_summary(summary_json)

    return []


def _load_markov_from_aggregated(path: Path) -> List[DGPSummaryRow]:
    """Load and re-aggregate by law_package.

    The aggregated CSV from report_markov_law_stress.py contains one row per
    (config, law_package) combination.  We need to collapse across configs so
    that each law_package appears exactly once in the cross-DGP table.
    """
    by_pkg: Dict[str, List[Dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            pkg = str(rec.get("law_package", "unknown"))
            by_pkg.setdefault(pkg, []).append(rec)

    rows: List[DGPSummaryRow] = []
    for pkg, recs in sorted(by_pkg.items()):
        # Weight each sub-row by its n_runs when aggregating
        total_n = sum(int(float(r.get("n_runs", r.get("count", 1)))) for r in recs)

        def _wmean(key: str, fallback_key: str = "") -> float:
            vals = []
            for r in recs:
                v = r.get(key) if r.get(key) is not None else r.get(fallback_key)
                if v is not None:
                    n = int(float(r.get("n_runs", r.get("count", 1))))
                    vals.append((float(v), n))
            if not vals:
                return 0.0
            return sum(v * n for v, n in vals) / sum(n for _, n in vals)

        # Derive primary_gain from root_ratio when mean_primary_gain is absent.
        # root_ratio = learned_root_mae / baseline_root_mae, so
        # primary_gain_frac = 1 - root_ratio (positive = improvement).
        primary_gain = _wmean("mean_primary_gain")
        if primary_gain == 0.0 and any(r.get("root_ratio") for r in recs):
            primary_gain = 1.0 - _wmean("root_ratio")

        # Derive laws_improved from per-law pass rates when absent.
        laws_improved = _wmean("mean_laws_improved")
        if laws_improved == 0.0:
            laws_improved = (
                _wmean("c1_pass_rate") + _wmean("c2_pass_rate") + _wmean("c3_pass_rate")
            )

        rows.append(DGPSummaryRow(
            dgp="markov_ops_count",
            law_package=pkg,
            n_runs=total_n,
            primary_pass_rate=_wmean("primary_pass_rate", "bundle_full_success_rate"),
            c1_pass_rate=_wmean("c1_pass_rate"),
            c2_pass_rate=_wmean("c2_pass_rate"),
            c3_pass_rate=_wmean("c3_pass_rate"),
            mean_laws_improved=laws_improved,
            mean_primary_gain=primary_gain,
        ))
    return rows


def _load_markov_from_assessed(path: Path) -> List[DGPSummaryRow]:
    """Aggregate per-run assessed rows by law_package."""
    by_pkg: Dict[str, List[Dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            pkg = str(rec.get("law_package", "unknown"))
            by_pkg.setdefault(pkg, []).append(rec)

    rows: List[DGPSummaryRow] = []
    for pkg, recs in sorted(by_pkg.items()):
        n = len(recs)
        rows.append(DGPSummaryRow(
            dgp="markov",
            law_package=pkg,
            n_runs=n,
            primary_pass_rate=_mean_bool_field(recs, "primary_pass") or _mean_bool_field(recs, "bundle_full_success"),
            c1_pass_rate=_mean_bool_field(recs, "c1_pass"),
            c2_pass_rate=_mean_bool_field(recs, "c2_pass"),
            c3_pass_rate=_mean_bool_field(recs, "c3_pass"),
            mean_laws_improved=_mean_float_field(recs, "laws_improved"),
            mean_primary_gain=_mean_float_field(recs, "primary_gain_frac"),
        ))
    return rows


def _load_markov_from_summary(path: Path) -> List[DGPSummaryRow]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: List[DGPSummaryRow] = []
    if isinstance(data, dict):
        for pkg, info in data.items():
            if isinstance(info, dict):
                rows.append(DGPSummaryRow(
                    dgp="markov",
                    law_package=pkg,
                    n_runs=int(info.get("n_runs", info.get("count", 1))),
                    primary_pass_rate=float(info.get("primary_pass_rate", info.get("bundle_full_success_rate", 0))),
                    c1_pass_rate=float(info.get("c1_pass_rate", 0)),
                    c2_pass_rate=float(info.get("c2_pass_rate", 0)),
                    c3_pass_rate=float(info.get("c3_pass_rate", 0)),
                    mean_laws_improved=float(info.get("mean_laws_improved", 0)),
                    mean_primary_gain=float(info.get("mean_primary_gain", 0)),
                    bundle_success_rate=float(info.get("bundle_full_success_rate", 0)),
                    root_pass_rate=float(info.get("root_pass_rate", 0)),
                    downstream_correlation=float(info.get("downstream_correlation", float("nan"))),
                ))
    return rows


def _load_lda_rows(lda_dir: Path) -> List[DGPSummaryRow]:
    """Load LDA law-stress records from JSON files in output directory."""
    rows_by_pkg: Dict[str, List[Dict[str, Any]]] = {}

    json_files = sorted(lda_dir.rglob("*.json"))
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        law_stress = data.get("law_stress") or data.get("local_law", {}).get("law_stress")
        if not law_stress:
            continue

        cfg = dict(data.get("config", {}) or {})
        local_law = dict(data.get("local_law", {}) or {})
        local_law_cfg = dict(local_law.get("config", {}) or {})
        objective = dict(local_law.get("objective", {}) or data.get("objective", {}) or {})
        pkg = str(
            local_law_cfg.get("law_package", "")
            or objective.get("law_package", "")
            or cfg.get("law_package", "")
            or "unknown"
        ).strip()

        selected_candidate = str(
            data.get("local_law", {}).get("selection", {}).get("selected_candidate", "")
            or data.get("selection", {}).get("selected_candidate", "")
            or ""
        ).strip()

        best_stress: Optional[Dict[str, Any]] = None
        if selected_candidate and isinstance(law_stress.get(selected_candidate), dict):
            best_stress = dict(law_stress[selected_candidate])
        elif isinstance(law_stress.get("learned_g"), dict):
            best_stress = dict(law_stress["learned_g"])
        else:
            for stress in law_stress.values():
                if isinstance(stress, dict) and "bundle_status" in stress:
                    best_stress = dict(stress)
                    break

        if best_stress is None:
            continue

        rec = {
            "law_package": pkg,
            "c1_pass": bool(best_stress.get("c1_pass", False)),
            "c2_pass": bool(best_stress.get("c2_pass", False)),
            "c3_pass": bool(best_stress.get("c3_pass", False)),
            "bundle_full_success": bool(best_stress.get("bundle_full_success", False)),
            "root_pass": bool(best_stress.get("root_pass", False)),
        }
        rows_by_pkg.setdefault(pkg, []).append(rec)

    result: List[DGPSummaryRow] = []
    for pkg, recs in sorted(rows_by_pkg.items()):
        n = len(recs)
        result.append(DGPSummaryRow(
            dgp="lda",
            law_package=pkg,
            n_runs=n,
            primary_pass_rate=_mean_bool_field(recs, "primary_pass") or _mean_bool_field(recs, "bundle_full_success"),
            c1_pass_rate=_mean_bool_field(recs, "c1_pass"),
            c2_pass_rate=_mean_bool_field(recs, "c2_pass"),
            c3_pass_rate=_mean_bool_field(recs, "c3_pass"),
            mean_laws_improved=_mean_float_field(recs, "laws_improved"),
            mean_primary_gain=_mean_float_field(recs, "primary_gain_frac"),
        ))
    return result


def _mean_bool_field(recs: Sequence[Dict[str, Any]], key: str) -> float:
    vals = [1.0 if _to_bool(r.get(key, False)) else 0.0 for r in recs]
    return float(fmean(vals)) if vals else 0.0


def _mean_float_field(recs: Sequence[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0)) for r in recs if r.get(key) is not None]
    return float(fmean(vals)) if vals else 0.0


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v)


def _expected_main_package_for_dgp(dgp: str, packages: Sequence[str]) -> Optional[str]:
    available = {str(x).strip() for x in packages if str(x).strip()}
    dgp = str(dgp).strip()
    if dgp in {"markov", "markov_ops_count"}:
        if "all_laws_plus_sched" in available:
            return "all_laws_plus_sched"
        if "all_laws" in available:
            return "all_laws"
    if dgp in {"lda", "tree_relevant_lda_local_law"}:
        if "all_laws" in available:
            return "all_laws"
    for candidate in ("all_laws_plus_sched", "all_laws"):
        if candidate in available:
            return candidate
    return None


def _claim_role(row: DGPSummaryRow, rows: Sequence[DGPSummaryRow]) -> str:
    expected = _expected_main_package_for_dgp(
        row.dgp,
        [r.law_package for r in rows if str(r.dgp) == str(row.dgp)],
    )
    return "main" if expected is not None and str(row.law_package) == str(expected) else "ablation"


def _dgp_guidance(rows: Sequence[DGPSummaryRow]) -> List[str]:
    notes: List[str] = [
        "Only the expected full package should be read as the paper claim. Single-law and partial-law packages are ablations/mechanism diagnostics.",
        "PrimGain is downstream fractional improvement versus matched baseline; positive is better, negative is worse.",
    ]
    by_dgp: Dict[str, List[DGPSummaryRow]] = {}
    for row in rows:
        by_dgp.setdefault(str(row.dgp), []).append(row)
    for dgp, dgp_rows in sorted(by_dgp.items()):
        expected = _expected_main_package_for_dgp(dgp, [r.law_package for r in dgp_rows])
        main_row = next((r for r in dgp_rows if str(r.law_package) == str(expected)), None)
        ablations = [r for r in dgp_rows if str(r.law_package) != str(expected)]
        strongest_ablation = None
        if ablations:
            strongest_ablation = max(
                ablations,
                key=lambda r: (float(r.primary_pass_rate), float(r.mean_primary_gain), float(r.mean_laws_improved)),
            )
        if expected is None:
            notes.append(f"{dgp}: no canonical full-package row is present in the current comparison.")
            continue
        if main_row is None:
            notes.append(f"{dgp}: expected full-package row `{expected}` is missing from the current comparison.")
            continue
        if strongest_ablation is None:
            notes.append(f"{dgp}: only the expected full-package row `{expected}` is present here; the ablation story is not covered in this comparison.")
            continue
        if (
            float(strongest_ablation.primary_pass_rate) > float(main_row.primary_pass_rate)
            or float(strongest_ablation.mean_primary_gain) > float(main_row.mean_primary_gain)
        ):
            notes.append(
                f"{dgp}: expected claim row is `{expected}`. `{strongest_ablation.law_package}` is stronger on downstream metrics in this sweep, "
                "but it is an ablation and should be read as a mechanism diagnostic rather than a replacement claim."
            )
        else:
            notes.append(
                f"{dgp}: `{expected}` remains the expected claim row. Ablations are present as mechanism checks and should not be ranked as alternative success criteria."
            )
    return notes


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _build_text_table(rows: List[DGPSummaryRow]) -> str:
    """Build a monospace text table for the cross-DGP comparison."""
    notes = _dgp_guidance(rows)
    header = f"{'DGP':<35} {'Package':<18} {'Role':<8} {'N':>5} {'Prim%':>6} {'C1%':>5} {'C2%':>5} {'C3%':>5} {'Laws':>5} {'PrimGain':>9}"
    sep = "-" * len(header)
    lines = notes + ["", header, sep]
    for r in rows:
        lines.append(
            f"{r.dgp:<35} {r.law_package:<18} {_claim_role(r, rows):<8} {r.n_runs:>5} "
            f"{r.primary_pass_rate:>5.1%} "
            f"{r.c1_pass_rate:>4.0%} {r.c2_pass_rate:>4.0%} {r.c3_pass_rate:>4.0%} "
            f"{r.mean_laws_improved:>5.1f} {r.mean_primary_gain:>8.1%}"
        )
    return "\n".join(lines)


def _build_comparison_figure(rows: List[DGPSummaryRow]) -> plt.Figure:
    """Grouped bar chart: C1/C2/C3/Bundle pass rates by DGP x law_package."""
    # Gather unique packages present in both DGPs
    all_packages = sorted({r.law_package for r in rows})
    dgps = sorted({r.dgp for r in rows})

    metric_names = ["Primary", "C1", "C2", "C3"]
    metric_keys = ["primary_pass_rate", "c1_pass_rate", "c2_pass_rate", "c3_pass_rate"]
    colors = ["#2d2d2d", "#4c72b0", "#55a868", "#c44e52"]

    n_groups = len(all_packages)
    n_dgps = len(dgps)
    n_metrics = len(metric_names)
    bar_width = 0.8 / max(n_dgps * n_metrics, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 2.5), 5))

    lookup = {(r.dgp, r.law_package): r for r in rows}

    x = np.arange(n_groups)
    for d_idx, dgp in enumerate(dgps):
        for m_idx, (mname, mkey) in enumerate(zip(metric_names, metric_keys)):
            offset = (d_idx * n_metrics + m_idx - n_dgps * n_metrics / 2 + 0.5) * bar_width
            vals = [getattr(lookup.get((dgp, pkg)), mkey, 0.0) if (dgp, pkg) in lookup else 0.0
                    for pkg in all_packages]
            hatch = "//" if d_idx == 1 else None
            ax.bar(x + offset, vals, bar_width * 0.9, label=f"{dgp} {mname}" if d_idx == 0 or mname == "C1" else "",
                   color=colors[m_idx], alpha=0.7 + 0.3 * (d_idx == 0), hatch=hatch, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(all_packages, rotation=30, ha="right")
    ax.set_ylabel("Pass Rate")
    ax.set_title("Cross-DGP Law-Stress Comparison\n(full-package rows are the claim rows; ablations are diagnostic only)")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)

    # Build legend
    handles = []
    for m_idx, mname in enumerate(metric_names):
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=colors[m_idx], alpha=0.85, label=mname))
    for d_idx, dgp in enumerate(dgps):
        h = "//" if d_idx == 1 else None
        handles.append(plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.5, hatch=h, label=f"({dgp})"))
    ax.legend(handles=handles, loc="upper left", ncol=3, fontsize=8)

    fig.tight_layout()
    return fig


def _collect_source_paths(
    unified_root: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> List[Path]:
    """Collect JSON source paths from a manifest JSONL and/or a directory tree."""
    paths: List[Path] = []
    seen: set = set()

    # Read manifest JSONL: each line has config.source_path
    if manifest_path is not None and manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sp = str(rec.get("config", {}).get("source_path", "") or "").strip()
            if sp and sp not in seen:
                p = Path(sp)
                if p.exists():
                    seen.add(sp)
                    paths.append(p)

    # Also rglob from unified_root if given
    if unified_root is not None and unified_root.exists():
        for jf in sorted(unified_root.rglob("*.json")):
            sp = str(jf)
            if sp not in seen:
                seen.add(sp)
                paths.append(jf)

    return paths


def _load_unified_rows(
    unified_root: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> List[DGPSummaryRow]:
    """Load rows from a manifest JSONL and/or directory using the backfill pipeline."""
    from src.ctreepo.sim.local_law_backfill import (
        collect_law_stress_assessments,
        load_or_backfill_local_law_payload,
    )

    source_paths = _collect_source_paths(unified_root, manifest_path)
    if not source_paths:
        return []

    by_group: Dict[str, List[Dict[str, Any]]] = {}
    loaded_count = 0
    loaded_records: List[tuple[str, object, Dict[str, Any]]] = []

    for jf in source_paths:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        loaded_result = load_or_backfill_local_law_payload(data, source_path=str(jf))
        if loaded_result is None:
            continue
        loaded_count += 1
        summary, augmented = loaded_result
        loaded_records.append((str(jf), summary, augmented))

    classified_records = collect_law_stress_assessments(loaded_records)
    for record in classified_records:
        family = str(record.get("family", ""))
        law_package = str(record.get("law_package", "") or "").strip() or "unknown"
        stress = dict(record.get("assessment", {}) or {})
        if not stress:
            continue
        key = f"{family}|{law_package}"
        by_group.setdefault(key, []).append(stress)

    print(f"Loaded {loaded_count} runs, classified {len(classified_records)} with law-stress.")

    rows: List[DGPSummaryRow] = []
    for key, assessments in sorted(by_group.items()):
        parts = key.split("|", 1)
        family = parts[0]
        law_package = parts[1] if len(parts) > 1 else ""
        n = len(assessments)
        rows.append(DGPSummaryRow(
            dgp=family,
            law_package=law_package,
            n_runs=n,
            primary_pass_rate=_mean_bool_field(assessments, "primary_pass") or _mean_bool_field(assessments, "bundle_full_success"),
            c1_pass_rate=_mean_bool_field(assessments, "c1_pass"),
            c2_pass_rate=_mean_bool_field(assessments, "c2_pass"),
            c3_pass_rate=_mean_bool_field(assessments, "c3_pass"),
            mean_laws_improved=_mean_float_field(assessments, "laws_improved"),
            mean_primary_gain=_mean_float_field(assessments, "primary_gain_frac"),
            bundle_success_rate=_mean_bool_field(assessments, "bundle_full_success"),
            root_pass_rate=_mean_bool_field(assessments, "root_pass"),
        ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-DGP law-stress comparison report")
    parser.add_argument("--markov-dir", type=str, default=None,
                        help="Directory with Markov law-stress report outputs (CSV/JSON)")
    parser.add_argument("--lda-dir", type=str, default=None,
                        help="Directory with LDA law-stress report outputs (JSON files)")
    parser.add_argument("--unified-root", type=str, default=None,
                        help="Unified inventory root directory (rglobs for JSON files).")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Manifest JSONL with source_path entries (fastest path).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for cross-DGP report")
    parser.add_argument("--pdf-path", type=str, default=None,
                        help="Explicit PDF output path")
    args = parser.parse_args()

    all_rows: List[DGPSummaryRow] = []

    if args.unified_root or args.manifest:
        all_rows.extend(_load_unified_rows(
            unified_root=Path(args.unified_root) if args.unified_root else None,
            manifest_path=Path(args.manifest) if args.manifest else None,
        ))

    if args.markov_dir:
        all_rows.extend(_load_markov_rows(Path(args.markov_dir)))

    if args.lda_dir:
        all_rows.extend(_load_lda_rows(Path(args.lda_dir)))

    if not all_rows:
        print("No law-stress data found in either directory. Exiting.")
        return

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.manifest:
        output_dir = Path(args.manifest).parent / "cross_dgp_report"
    elif args.unified_root:
        output_dir = Path(args.unified_root) / "cross_dgp_report"
    elif args.markov_dir:
        output_dir = Path(args.markov_dir).parent / "cross_dgp_report"
    else:
        output_dir = Path("outputs/cross_dgp_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Text table
    table = _build_text_table(all_rows)
    print(table)
    (output_dir / "cross_dgp_law_stress_table.txt").write_text(table + "\n", encoding="utf-8")

    # JSON summary
    summary = {
        "generated": datetime.now(tz=timezone.utc).isoformat(),
        "sources": {
            "unified_root": str(args.unified_root or ""),
            "manifest": str(args.manifest or ""),
            "markov_dir": str(args.markov_dir or ""),
            "lda_dir": str(args.lda_dir or ""),
        },
        "methodological_note": (
            "Only the expected full package should be read as the paper claim. "
            "Single-law and partial-law packages are ablations/mechanism diagnostics."
        ),
        "expected_main_package_by_dgp": {
            str(dgp): _expected_main_package_for_dgp(str(dgp), [r.law_package for r in all_rows if str(r.dgp) == str(dgp)])
            for dgp in sorted({str(r.dgp) for r in all_rows})
        },
        "rows": [
            {
                "dgp": r.dgp,
                "law_package": r.law_package,
                "claim_role": _claim_role(r, all_rows),
                "n_runs": r.n_runs,
                "primary_pass_rate": r.primary_pass_rate,
                "c1_pass_rate": r.c1_pass_rate,
                "c2_pass_rate": r.c2_pass_rate,
                "c3_pass_rate": r.c3_pass_rate,
                "mean_laws_improved": r.mean_laws_improved,
                "mean_primary_gain": r.mean_primary_gain,
            }
            for r in all_rows
        ],
    }
    summary_path = output_dir / "cross_dgp_law_stress_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # PDF report
    pdf_path = Path(args.pdf_path) if args.pdf_path else (output_dir / "cross_dgp_law_stress_report.pdf")

    with PdfPages(str(pdf_path)) as pdf:
        fig_note, ax_note = plt.subplots(figsize=(11, 4.2))
        ax_note.axis("off")
        ax_note.text(
            0.02,
            0.98,
            "\n".join(_dgp_guidance(all_rows)),
            transform=ax_note.transAxes,
            fontsize=10,
            verticalalignment="top",
            wrap=True,
        )
        ax_note.set_title("Cross-DGP Readout", fontsize=12, fontweight="bold")
        fig_note.tight_layout()
        pdf.savefig(fig_note)
        plt.close(fig_note)

        # Page 1: comparison bar chart
        fig = _build_comparison_figure(all_rows)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: text table as figure
        fig_table, ax_table = plt.subplots(figsize=(11, max(4, len(all_rows) * 0.4 + 2)))
        ax_table.axis("off")
        ax_table.text(0.02, 0.98, table, transform=ax_table.transAxes,
                      fontsize=8, verticalalignment="top", fontfamily="monospace")
        ax_table.set_title("Cross-DGP Law-Stress Summary", fontsize=12, fontweight="bold")
        fig_table.tight_layout()
        pdf.savefig(fig_table)
        plt.close(fig_table)

    print(f"\nReport written to: {output_dir}")
    print(f"  PDF: {pdf_path}")
    print(f"  JSON: {summary_path}")
    print(f"  Table: {output_dir / 'cross_dgp_law_stress_table.txt'}")


if __name__ == "__main__":
    main()
