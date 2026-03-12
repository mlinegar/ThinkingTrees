#!/usr/bin/env python3
"""Analyze the C1/C2/C3 weight ablation results.

Reads JSON outputs from the weight ablation suite, extracts root_error from
local_law_learnability policies, and produces matched comparisons.

Usage:
    python scripts/analyze_weight_ablation.py \
        --input-root outputs/markov_weight_ablation_20260309/weight_ablation_suite \
        --output-dir outputs/markov_weight_ablation_20260309/analysis
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


WEIGHT_PROFILE_ORDER = [
    "pure_c2",
    "c2_trace_c1c3",
    "c2_light_c1c3",
    "c2_mild_c1c3",
    "c2_moderate_c1c3",
    "c2_very_dominant",
    "c2_dominant",
    "c2_heavy",
    "equal",
    "c1c3_heavy",
    "c3_dominant",
    "no_c2",
]

WEIGHT_PROFILES: Dict[Tuple[float, float, float], str] = {}

def _register_profiles():
    exact = {
        (0.0, 1.0, 0.0): "pure_c2",
        (1.0, 0.0, 4.0): "no_c2",
        (0.05, 1.0, 0.05): "c2_trace_c1c3",
        (0.1, 1.0, 0.1): "c2_light_c1c3",
        (0.25, 1.0, 0.25): "c2_mild_c1c3",
        (0.5, 1.0, 0.5): "c2_moderate_c1c3",
        (1.0, 8.0, 1.0): "c2_very_dominant",
        (1.0, 4.0, 1.0): "c2_dominant",
        (1.0, 2.0, 1.0): "c2_heavy",
        (1.0, 1.0, 1.0): "equal",
        (2.0, 1.0, 2.0): "c1c3_heavy",
        (1.0, 1.0, 4.0): "c3_dominant",
    }
    WEIGHT_PROFILES.update(exact)

_register_profiles()


def _classify_profile(c1r: float, c2r: float, c3r: float, pkg: str) -> str:
    if pkg == "root_only":
        return "root_only"
    key = (round(c1r, 2), round(c2r, 2), round(c3r, 2))
    if key in WEIGHT_PROFILES:
        return WEIGHT_PROFILES[key]
    key2 = (round(c1r, 1), round(c2r, 1), round(c3r, 1))
    if key2 in WEIGHT_PROFILES:
        return WEIGHT_PROFILES[key2]
    return f"c1={c1r}_c2={c2r}_c3={c3r}"


def _scenario_key(config: Dict[str, Any]) -> str:
    return (
        f"train_{config.get('train_docs', '?')}"
        f"_audit_{config.get('audit_fraction', '?')}"
        f"_dseed_{config.get('data_seed', '?')}"
        f"_sd_{config.get('state_dim', '?')}"
        f"_hd_{config.get('hidden_dim', '?')}"
    )


def _extract_root_error(payload: Dict[str, Any]) -> Optional[float]:
    """Extract test root_error from JSON payload."""
    ll = payload.get("local_law_learnability", {})
    policies = ll.get("policies", {})
    for pname in ["learned_g", "root_only"]:
        pdata = policies.get(pname, {})
        test_dm = pdata.get("split_metrics", {}).get("test", {}).get("downstream_metrics", {})
        root_err = test_dm.get("root_error")
        if root_err is not None:
            try:
                v = float(root_err)
                if math.isfinite(v):
                    return v
            except (ValueError, TypeError):
                pass
        test_llm = pdata.get("split_metrics", {}).get("test", {}).get("local_law_metrics", {})
        root_err = test_llm.get("root_error")
        if root_err is not None:
            try:
                v = float(root_err)
                if math.isfinite(v):
                    return v
            except (ValueError, TypeError):
                pass
    return None


def _extract_law_metrics(payload: Dict[str, Any]) -> Dict[str, float]:
    ll = payload.get("local_law_learnability", {})
    policies = ll.get("policies", {})
    for pname in ["learned_g", "root_only"]:
        pdata = policies.get(pname, {})
        test_llm = pdata.get("split_metrics", {}).get("test", {}).get("local_law_metrics", {})
        if test_llm:
            return {
                "c1": float(test_llm.get("c1", float("nan"))),
                "c2": float(test_llm.get("c2", float("nan"))),
                "c3": float(test_llm.get("c3", float("nan"))),
            }
    return {"c1": float("nan"), "c2": float("nan"), "c3": float("nan")}


def load_jsons(input_root: Path) -> List[Dict[str, Any]]:
    records = []
    for json_path in sorted(input_root.rglob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            config = payload.get("config", {})
            if not config:
                continue
            c1r = float(config.get("c1_relative_weight", 0))
            c2r = float(config.get("c2_relative_weight", 0))
            c3r = float(config.get("c3_relative_weight", 0))
            pkg = str(config.get("law_package", "") or "").strip()
            profile = _classify_profile(c1r, c2r, c3r, pkg)
            root_error = _extract_root_error(payload)
            laws = _extract_law_metrics(payload)
            scenario = _scenario_key(config)
            records.append({
                "path": str(json_path),
                "profile": profile,
                "scenario": scenario,
                "root_error": root_error,
                "c1": laws["c1"],
                "c2": laws["c2"],
                "c3": laws["c3"],
            })
        except Exception:
            continue
    return records


def analyze(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_profile: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_scenario: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for rec in records:
        by_profile[rec["profile"]].append(rec)
        by_scenario[rec["scenario"]][rec["profile"]].append(rec)

    profile_summaries = []
    for profile in ["root_only"] + WEIGHT_PROFILE_ORDER:
        recs = by_profile.get(profile, [])
        if not recs:
            continue
        root_errors = [r["root_error"] for r in recs if r["root_error"] is not None]
        profile_summaries.append({
            "profile": profile,
            "n": len(recs),
            "n_with_root": len(root_errors),
            "mean_root_error": float(np.mean(root_errors)) if root_errors else float("nan"),
            "median_root_error": float(np.median(root_errors)) if root_errors else float("nan"),
        })

    matched_pairs = []
    for scenario, profiles in sorted(by_scenario.items()):
        baseline_recs = profiles.get("root_only", [])
        if not baseline_recs:
            continue
        baseline_errors = [r["root_error"] for r in baseline_recs if r["root_error"] is not None]
        if not baseline_errors:
            continue
        baseline_mae = float(np.mean(baseline_errors))
        if baseline_mae <= 0:
            continue

        for profile in WEIGHT_PROFILE_ORDER:
            treatment_recs = profiles.get(profile, [])
            if not treatment_recs:
                continue
            treatment_errors = [r["root_error"] for r in treatment_recs if r["root_error"] is not None]
            if not treatment_errors:
                continue
            treatment_mae = float(np.mean(treatment_errors))
            matched_pairs.append({
                "scenario": scenario,
                "profile": profile,
                "baseline_mae": baseline_mae,
                "treatment_mae": treatment_mae,
                "ratio": treatment_mae / baseline_mae,
                "gain_pct": (1.0 - treatment_mae / baseline_mae) * 100,
            })

    matched_summaries = []
    for profile in WEIGHT_PROFILE_ORDER:
        pairs = [p for p in matched_pairs if p["profile"] == profile]
        if not pairs:
            continue
        ratios = [p["ratio"] for p in pairs]
        gains = [p["gain_pct"] for p in pairs]
        matched_summaries.append({
            "profile": profile,
            "n_matched": len(pairs),
            "mean_ratio": float(np.mean(ratios)),
            "median_ratio": float(np.median(ratios)),
            "min_ratio": float(np.min(ratios)),
            "max_ratio": float(np.max(ratios)),
            "mean_gain_pct": float(np.mean(gains)),
            "primary_pass_rate": len([r for r in ratios if r < 1.0]) / len(ratios),
        })

    return {
        "profile_summaries": profile_summaries,
        "matched_summaries": matched_summaries,
        "matched_pairs": matched_pairs,
        "n_total": len(records),
        "n_scenarios": len(by_scenario),
    }


def _format_table(summaries: List[Dict[str, Any]], title: str) -> str:
    lines = [title, "=" * len(title), ""]
    header = f"{'Profile':<20s} {'N':>4s} {'MeanRatio':>10s} {'MedRatio':>10s} {'MinR':>7s} {'MaxR':>7s} {'Pass%':>7s} {'Gain%':>8s}"
    lines.append(header)
    lines.append("-" * len(header))
    for s in summaries:
        lines.append(
            f"{s['profile']:<20s} "
            f"{s.get('n_matched', s.get('n', 0)):>4d} "
            f"{s.get('mean_ratio', float('nan')):>10.4f} "
            f"{s.get('median_ratio', float('nan')):>10.4f} "
            f"{s.get('min_ratio', float('nan')):>7.3f} "
            f"{s.get('max_ratio', float('nan')):>7.3f} "
            f"{s.get('primary_pass_rate', 0)*100:>6.1f}% "
            f"{s.get('mean_gain_pct', 0):>+7.1f}%"
        )
    lines.append("")
    return "\n".join(lines)


def plot_profile_comparison(matched_summaries: List[Dict[str, Any]], output_path: Path) -> None:
    if not matched_summaries:
        return

    profiles = [s["profile"] for s in matched_summaries]
    gains = [s["mean_gain_pct"] for s in matched_summaries]
    pass_rates = [s["primary_pass_rate"] * 100 for s in matched_summaries]
    n_matched = [s["n_matched"] for s in matched_summaries]

    pretty_labels = {
        "pure_c2": "C₂ only\n(0,1,0)",
        "c2_trace_c1c3": "C₂+trace\n(.05,1,.05)",
        "c2_light_c1c3": "C₂+light\n(.1,1,.1)",
        "c2_mild_c1c3": "C₂+mild\n(.25,1,.25)",
        "c2_moderate_c1c3": "C₂+mod\n(.5,1,.5)",
        "c2_very_dominant": "C₂ v.dom\n(1,8,1)",
        "c2_dominant": "C₂ dom\n(1,4,1)",
        "c2_heavy": "C₂ heavy\n(1,2,1)",
        "equal": "Equal\n(1,1,1)",
        "c1c3_heavy": "C₁C₃ hvy\n(2,1,2)",
        "c3_dominant": "C₃ dom\n(1,1,4)",
        "no_c2": "No C₂\n(1,0,4)",
    }
    labels = [pretty_labels.get(p, p) for p in profiles]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2ca02c" if g > 0 else "#d62728" for g in gains]
    bars = ax1.bar(range(len(profiles)), gains, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5, width=0.7)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(range(len(profiles)))
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("Mean Primary Gain (%) = 1 - ratio")
    ax1.set_title("(a) Matched Primary Gain vs Root-Only Baseline\n(Pure test MAE ratio)")
    for bar, g, n in zip(bars, gains, n_matched):
        y = bar.get_height()
        offset = 0.5 if g >= 0 else -0.5
        ax1.text(bar.get_x() + bar.get_width()/2, y + offset,
                f"{g:+.1f}%\nn={n}", ha="center", va="bottom" if g >= 0 else "top", fontsize=6)

    bars2 = ax2.bar(range(len(profiles)), pass_rates, color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.5, width=0.7)
    ax2.set_xticks(range(len(profiles)))
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("Primary Pass Rate (%)")
    ax2.set_title("(b) % Configs Where Learned Beats Root-Only\n(ratio < 1.0)")
    ax2.set_ylim(0, 110)
    for bar, pr in zip(bars2, pass_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{pr:.0f}%", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Markov C₁/C₂/C₃ Weight Ablation: Impact on Root Prediction Quality", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(matched_pairs: List[Dict[str, Any]], output_path: Path) -> None:
    if not matched_pairs:
        return

    available = [p for p in WEIGHT_PROFILE_ORDER if any(mp["profile"] == p for mp in matched_pairs)]
    scenarios = sorted(set(p["scenario"] for p in matched_pairs))

    data = np.full((len(available), len(scenarios)), float("nan"))
    for p in matched_pairs:
        if p["profile"] in available:
            pi = available.index(p["profile"])
            ci = scenarios.index(p["scenario"])
            data[pi, ci] = p["ratio"]

    fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 0.6), max(4, len(available) * 0.5)))

    vmin = max(0.5, float(np.nanmin(data)) - 0.05) if np.any(np.isfinite(data)) else 0.5
    vmax = min(2.0, float(np.nanmax(data)) + 0.05) if np.any(np.isfinite(data)) else 2.0
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", norm=norm, interpolation="nearest")
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=5, rotation=45, ha="right")
    ax.set_yticks(range(len(available)))
    ax.set_yticklabels([p.replace("_", " ") for p in available], fontsize=7)

    for i in range(len(available)):
        for j in range(len(scenarios)):
            val = data[i, j]
            if math.isfinite(val):
                color = "white" if abs(val - 1.0) > 0.15 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=4.5, color=color)

    ax.set_title("Matched Root MAE Ratio by Weight Profile × Config\n(< 1.0 green = learned beats baseline)", fontsize=10)
    fig.colorbar(im, ax=ax, label="ratio = treatment_MAE / baseline_MAE")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze weight ablation results")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading JSONs from {input_root}...")
    records = load_jsons(input_root)
    print(f"Loaded {len(records)} records ({sum(1 for r in records if r['root_error'] is not None)} with root_error)")

    results = analyze(records)

    summary_path = output_dir / "weight_ablation_summary.json"
    summary_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    table_path = output_dir / "weight_ablation_table.txt"
    text = _format_table(results["matched_summaries"], "Matched-Pair Summary: treatment_MAE / baseline_MAE")
    table_path.write_text(text, encoding="utf-8")
    print(f"\n{text}")

    heatmap_path = output_dir / "weight_ablation_heatmap.png"
    plot_heatmap(results["matched_pairs"], heatmap_path)

    comparison_path = output_dir / "weight_ablation_comparison.png"
    plot_profile_comparison(results["matched_summaries"], comparison_path)

    print(f"All outputs in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
