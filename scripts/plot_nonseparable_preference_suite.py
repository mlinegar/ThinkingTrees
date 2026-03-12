#!/usr/bin/env python3
"""Plot non-separable preference suite summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


ARM_ORDER = [
    "oracle",
    "supported_merge_safe",
    "undersupported_sketch",
    "right_rule_wrong_chunker",
    "naive_non_merge_safe",
]
ARM_LABEL = {
    "oracle": "oracle",
    "supported_merge_safe": "supported",
    "undersupported_sketch": "undersupported",
    "right_rule_wrong_chunker": "wrong chunker",
    "naive_non_merge_safe": "naive",
}
ARM_COLOR = {
    "oracle": "#1f77b4",
    "supported_merge_safe": "#2ca02c",
    "undersupported_sketch": "#ff7f0e",
    "right_rule_wrong_chunker": "#d62728",
    "naive_non_merge_safe": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot non-separable preference suite results.")
    p.add_argument(
        "--json-summary",
        type=str,
        default="outputs/nonseparable_preference_suite_summary.json",
    )
    p.add_argument(
        "--output",
        type=str,
        default="outputs/nonseparable_preference_suite.png",
    )
    p.add_argument(
        "--report-json",
        type=str,
        default="outputs/nonseparable_preference_suite_plot_report.json",
    )
    return p.parse_args()


def _arm_map(dgp: dict) -> Dict[str, dict]:
    return {str(a["arm"]): a for a in dgp.get("arms", [])}


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.json_summary).read_text(encoding="utf-8"))
    dgps = payload.get("dgps", [])
    if len(dgps) == 0:
        raise ValueError("summary has no DGP results")

    n_rows = len(dgps)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.8 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.array([axes])  # type: ignore[assignment]

    report_rows: List[dict] = []
    for i, dgp in enumerate(dgps):
        name = str(dgp["name"])
        by_arm = _arm_map(dgp)
        arms = [a for a in ARM_ORDER if a in by_arm]

        ax0 = axes[i, 0]
        xs = np.arange(len(arms))
        means = [float(by_arm[a]["mean_gap_to_oracle_loss"]) for a in arms]
        lo = [float(by_arm[a]["mean_gap_to_oracle_loss_ci95_low"]) for a in arms]
        hi = [float(by_arm[a]["mean_gap_to_oracle_loss_ci95_high"]) for a in arms]
        err = np.array([means[j] - lo[j] for j in range(len(arms))], dtype=np.float64)
        err_hi = np.array([hi[j] - means[j] for j in range(len(arms))], dtype=np.float64)
        ax0.bar(xs, means, color=[ARM_COLOR[a] for a in arms], alpha=0.88)
        ax0.errorbar(xs, means, yerr=[err, err_hi], fmt="none", ecolor="#222222", capsize=3)
        ax0.set_xticks(xs)
        ax0.set_xticklabels([ARM_LABEL[a] for a in arms], rotation=20, ha="right")
        ax0.set_ylabel("Mean gap-to-oracle loss")
        ax0.set_title(f"{name}: Primary Separation Metric")
        ax0.grid(axis="y", alpha=0.2)

        ax1 = axes[i, 1]
        reg = [float(by_arm[a]["mean_utility_regret"]) for a in arms]
        bnd = [float(by_arm[a]["mean_bound_envelope"]) for a in arms]
        width = 0.38
        ax1.bar(xs - width / 2.0, reg, width=width, color="#ff9896", label="observed regret")
        ax1.bar(xs + width / 2.0, bnd, width=width, color="#98df8a", label="bound envelope")
        ax1.set_xticks(xs)
        ax1.set_xticklabels([ARM_LABEL[a] for a in arms], rotation=20, ha="right")
        ax1.set_ylabel("Mean value")
        ax1.set_title(f"{name}: Regret vs DGP-Implied Bound")
        ax1.grid(axis="y", alpha=0.2)
        ax1.legend(frameon=False, fontsize=8)

        ax2 = axes[i, 2]
        checks = dgp.get("separation_checks", [])
        if len(checks) > 0:
            c_arms = [str(c["arm"]) for c in checks]
            x2 = np.arange(len(c_arms))
            c_mean = [float(c["mean_delta_supported_vs_arm"]) for c in checks]
            c_lo = [float(c["ci95_low"]) for c in checks]
            c_hi = [float(c["ci95_high"]) for c in checks]
            c_err = np.array([c_mean[j] - c_lo[j] for j in range(len(c_arms))], dtype=np.float64)
            c_err_hi = np.array([c_hi[j] - c_mean[j] for j in range(len(c_arms))], dtype=np.float64)
            ax2.bar(x2, c_mean, color=[ARM_COLOR.get(a, "#444444") for a in c_arms], alpha=0.88)
            ax2.errorbar(x2, c_mean, yerr=[c_err, c_err_hi], fmt="none", ecolor="#222222", capsize=3)
            ax2.axhline(0.0, color="#444444", linewidth=1)
            ax2.axhline(float(payload["config"]["effect_gate"]), color="#444444", linestyle="--", linewidth=1)
            ax2.axhline(
                float(payload["config"]["strong_effect_gate"]),
                color="#666666",
                linestyle=":",
                linewidth=1,
            )
            ax2.set_xticks(x2)
            ax2.set_xticklabels([ARM_LABEL.get(a, a) for a in c_arms], rotation=20, ha="right")
            ax2.set_ylabel("Delta gap (arm - supported)")
            ax2.set_title(f"{name}: Gate Deltas (95% CI)")
            ax2.grid(axis="y", alpha=0.2)
        else:
            ax2.axis("off")

        report_rows.append(
            {
                "dgp": name,
                "strong_separation_pass": bool(dgp.get("strong_separation_pass", False)),
                "n_flagged_cells": len(dgp.get("flagged_cells", [])),
                "n_separation_checks": len(checks),
                "n_gate_passes": sum(1 for c in checks if bool(c.get("passes_gate", False))),
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    report = {
        "summary_path": str(args.json_summary),
        "figure_path": str(out_path),
        "rows": report_rows,
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote_figure | {out_path}")
    print(f"wrote_report | {report_path}")
    for row in report_rows:
        print(
            f"dgp={row['dgp']} | strong_separation={int(bool(row['strong_separation_pass']))} "
            f"| gate_passes={row['n_gate_passes']}/{row['n_separation_checks']} "
            f"| flagged_cells={row['n_flagged_cells']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

