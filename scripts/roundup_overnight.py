#!/usr/bin/env python3
"""
Aggregate overnight Benoit-comparison results into a single side-by-side table.

Looks for:
  outputs/overnight_benoit/scorer_only/{dim}/report.json
  outputs/overnight_benoit/full_pipeline/{dim}/report.json
  outputs/overnight_benoit/optimizer_bootstrap/{dim}/report.json
  outputs/phase0_full_pipeline_economic_229/report.json   (the prior Economic full-pipeline run)

Prints a unified Pearson-r table with Benoit Figure 1 / Table 3 / Table 6
reference columns. Writes the aggregated table to
outputs/overnight_benoit/roundup.json and a human-readable markdown table at
outputs/overnight_benoit/roundup.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


_ORDER = ["economic", "social", "immigration", "eu", "environment", "decentralization"]
_LABEL = {
    "economic": "Economic",
    "social": "Social",
    "immigration": "Immigration",
    "eu": "European Union",
    "environment": "Environment",
    "decentralization": "Decentralization",
}

# Benoit published reference values
_BENOIT_FIG1 = {
    "economic": 0.87, "social": 0.92, "immigration": 0.89,
    "eu": 0.91, "environment": 0.82, "decentralization": 0.49,
}
_BENOIT_TABLE3_UPPER = {
    "economic": 0.88, "social": 0.91, "immigration": 0.88,
    "eu": 0.95, "environment": 0.84, "decentralization": 0.78,
}
# Open-weight per-LLM correlations from Benoit Table 6 (Llama / DeepSeek / Gemma)
_BENOIT_TABLE6_OW = {
    "economic": (0.84, 0.84, 0.86),
    "social": (0.87, 0.87, 0.86),
    "immigration": (0.86, 0.89, 0.89),
    "eu": (0.86, 0.86, 0.84),
    "environment": (0.68, 0.79, 0.86),
    "decentralization": (0.40, 0.45, 0.45),
}


def _load_json(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _r(report: Optional[dict], key_chain: list[str]) -> Optional[dict]:
    if report is None:
        return None
    cur = report
    for k in key_chain:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _load_latest_phase2_report(root: Path, stem: str) -> Optional[dict]:
    candidates = []
    direct = root / stem
    if direct.is_dir():
        candidates.append(direct)
    candidates.extend(sorted(p for p in root.glob(f"{stem}_*") if p.is_dir()))

    newest_report = None
    newest_key = None
    for candidate in candidates:
        report_path = candidate / "report.json"
        report = _load_json(report_path)
        if report is None:
            continue
        key = (report_path.stat().st_mtime, str(candidate))
        if newest_key is None or key > newest_key:
            newest_key = key
            newest_report = report
    return newest_report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=project_root / "outputs" / "overnight_benoit")
    p.add_argument("--economic-old", type=Path,
                   default=project_root / "outputs" / "phase0_full_pipeline_economic_229" / "report.json")
    p.add_argument("--phase2-root", type=Path, default=project_root / "outputs" / "phase2")
    p.add_argument("--out-dir", type=Path,
                   default=project_root / "outputs" / "overnight_benoit")
    args = p.parse_args()

    # Phase 2 reports are single files covering all 6 dims; load once and index by dim
    phase2_joint = _load_latest_phase2_report(args.phase2_root, "joint_optimize")
    phase2_combined = _load_latest_phase2_report(args.phase2_root, "combined_pipeline")

    aggregated = {"dimensions": {}}

    for dim in _ORDER:
        scorer_only = _load_json(args.root / "scorer_only" / dim / "report.json")
        full_pipeline = _load_json(args.root / "full_pipeline" / dim / "report.json")
        if dim == "economic" and full_pipeline is None:
            full_pipeline = _load_json(args.economic_old)
        optimizer = _load_json(args.root / "optimizer_bootstrap" / dim / "report.json")

        joint_baseline = _r(phase2_joint, ["baseline", "per_dim", dim])
        joint_optimized = _r(phase2_joint, ["optimized", "per_dim", dim])
        combined = _r(phase2_combined, ["per_dim", dim])

        aggregated["dimensions"][dim] = {
            "label": _LABEL[dim],
            "benoit": {
                "figure1_proprietary_ensemble": _BENOIT_FIG1[dim],
                "table3_expert_upper_bound": _BENOIT_TABLE3_UPPER[dim],
                "table6_openweight_per_llm": list(_BENOIT_TABLE6_OW[dim]),
            },
            "scorer_only": _r(scorer_only, ["ours_vs_expert"]),
            "full_pipeline": _r(full_pipeline, ["report"]),
            "optimizer_baseline": _r(optimizer, ["baseline_test"]),
            "optimizer_optimized": _r(optimizer, ["optimized_test"]),
            "joint_baseline": joint_baseline,
            "joint_optimized": joint_optimized,
            "combined_pipeline": combined,
        }
    # Macro from phase 2 reports
    aggregated["macro"] = {
        "joint_baseline": _r(phase2_joint, ["baseline", "macro_pearson_r"]),
        "joint_optimized": _r(phase2_joint, ["optimized", "macro_pearson_r"]),
        "combined_pipeline": _r(phase2_combined, ["macro_pearson_r"]),
    }

    out_json = args.out_dir / "roundup.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(aggregated, indent=2))

    def _cell(rep: Optional[dict]) -> str:
        if rep is None:
            return "—"
        r = rep.get("pearson_r")
        n = rep.get("n")
        ci_low = rep.get("pearson_ci_low")
        ci_high = rep.get("pearson_ci_high")
        if r is None:
            return "—"
        return f"{r:+.3f}<br>[{ci_low:+.2f}, {ci_high:+.2f}], n={n}"

    lines = []
    lines.append(f"# Overnight Benoit comparison roundup\n")
    lines.append("Pearson r per dimension. Benoit reference is from Figure 1 (proprietary 18-score ensemble), Table 3 (expert upper bound), and Table 6 (open-weight per-LLM).\n")
    lines.append("## Per-dimension phase 0/1 results (per-dim summarizer + scorer)\n")
    lines.append("|Dimension|Scorer-only<br>(ours, on Benoit GPT-4o)|Full pipeline<br>(per-dim)|Optimizer baseline|Optimizer optimized|Benoit Fig 1|Benoit Table 3|Benoit Table 6 open-weight|")
    lines.append("|---|---|---|---|---|---|---|---|")
    for dim in _ORDER:
        d = aggregated["dimensions"][dim]
        ow = d["benoit"]["table6_openweight_per_llm"]
        lines.append(
            f"|{d['label']}"
            f"|{_cell(d['scorer_only'])}"
            f"|{_cell(d['full_pipeline'])}"
            f"|{_cell(d['optimizer_baseline'])}"
            f"|{_cell(d['optimizer_optimized'])}"
            f"|{d['benoit']['figure1_proprietary_ensemble']:.2f}"
            f"|{d['benoit']['table3_expert_upper_bound']:.2f}"
            f"|{ow[0]:.2f} / {ow[1]:.2f} / {ow[2]:.2f}|"
        )

    lines.append("\n## Phase 2 joint / combined results (shared g and f across dims)\n")
    lines.append("|Dimension|Joint baseline<br>(shared scorer, unoptimized)|Joint optimized<br>(BootstrapFewShot on pooled train)|Combined pipeline<br>(one summary w/ JOINT_RUBRIC, all 6 scored)|Benoit Fig 1|")
    lines.append("|---|---|---|---|---|")
    for dim in _ORDER:
        d = aggregated["dimensions"][dim]
        lines.append(
            f"|{d['label']}"
            f"|{_cell(d.get('joint_baseline'))}"
            f"|{_cell(d.get('joint_optimized'))}"
            f"|{_cell(d.get('combined_pipeline'))}"
            f"|{d['benoit']['figure1_proprietary_ensemble']:.2f}|"
        )
    macro = aggregated.get("macro", {})
    lines.append("")
    lines.append("**Macro avg Pearson r across 6 dims:**")
    def _macro(v):
        return f"{v:+.3f}" if v is not None else "—"
    lines.append(f"- joint baseline: {_macro(macro.get('joint_baseline'))}")
    lines.append(f"- joint optimized: {_macro(macro.get('joint_optimized'))}")
    lines.append(f"- combined pipeline: {_macro(macro.get('combined_pipeline'))}")
    out_md = args.out_dir / "roundup.md"
    out_md.write_text("\n".join(lines) + "\n")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print()
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
