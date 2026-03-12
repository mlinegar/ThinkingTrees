#!/usr/bin/env python3
"""Run variable-length/adversarial generalization sweep for mergeable ablations."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.mergeable_ablation import run_three_parameter_generalization_sweep
from src.ctreepo.sim.objective_semantics import mergeable_parameter_vector_objective_semantics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stress-test method robustness across variable-length and adversarial "
            "non-additive spike-mixture DGP shifts."
        )
    )
    parser.add_argument("--n-replicates", type=int, default=120)
    parser.add_argument("--docs-per-replicate", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline-scenario-name", type=str, default=None)
    parser.add_argument(
        "--align-boundary-span",
        action="store_true",
        help="Retune method boundary-span statistic per scenario's true boundary span.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path.")
    return parser.parse_args()


def _write_csv(path: Path, rows: List[dict]) -> None:
    if len(rows) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    summaries = run_three_parameter_generalization_sweep(
        n_replicates=args.n_replicates,
        docs_per_replicate=args.docs_per_replicate,
        seed=args.seed,
        baseline_scenario_name=args.baseline_scenario_name,
        align_boundary_span_to_distribution=args.align_boundary_span,
    )
    rows = [asdict(s) for s in summaries]

    if args.csv:
        _write_csv(Path(args.csv), rows)

    if args.json:
        payload = {
            "n_replicates": args.n_replicates,
            "docs_per_replicate": args.docs_per_replicate,
            "seed": args.seed,
            "baseline_scenario_name": args.baseline_scenario_name,
            "align_boundary_span": args.align_boundary_span,
            "objective": mergeable_parameter_vector_objective_semantics(
                name="mergeable_generalization_target",
                parameter_names=("p_spike_doc", "p_two_given_spike", "p_boundary_given_spike"),
                optimized_against="three_parameter_generalization_gap",
                metadata={"family": "mergeable_generalization_sweep"},
            ),
            "summaries": rows,
        }
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    scenario_names = sorted({r["scenario_name"] for r in rows})
    method_names = sorted({r["method_name"] for r in rows})
    print(
        f"scenarios={len(scenario_names)} methods={len(method_names)} "
        f"reps={args.n_replicates} docs_per_rep={args.docs_per_replicate} seed={args.seed} "
        f"align_boundary_span={int(bool(args.align_boundary_span))}"
    )

    by_scenario: Dict[str, List[dict]] = {name: [] for name in scenario_names}
    for r in rows:
        by_scenario[r["scenario_name"]].append(r)

    for scenario_name in scenario_names:
        srows = sorted(by_scenario[scenario_name], key=lambda r: r["aggregate_mean_abs_bias"])
        header = srows[0]
        print("")
        print(
            f"[{scenario_name}] true=(p_spike={header['true_p_spike']:.3f}, "
            f"p_two|spike={header['true_p_two_given_spike']:.3f}, "
            f"p_boundary|spike={header['true_p_boundary_given_spike']:.3f})"
        )
        print("method | agg_abs_bias | gap_vs_baseline | abs_bias_spike | abs_bias_two|spike | abs_bias_boundary|spike")
        for r in srows:
            print(
                f"{r['method_name']} | {r['aggregate_mean_abs_bias']:.4f} | "
                f"{r['generalization_gap_vs_baseline']:+.4f} | "
                f"{r['mean_abs_bias_p_spike']:.4f} | {r['mean_abs_bias_p_two_given_spike']:.4f} | "
                f"{r['mean_abs_bias_p_boundary_given_spike']:.4f}"
            )

    by_method: Dict[str, List[dict]] = {name: [] for name in method_names}
    for r in rows:
        by_method[r["method_name"]].append(r)

    print("")
    print("method_summary | mean_agg_abs_bias | worst_agg_abs_bias | mean_gap_vs_baseline")
    method_summary = []
    for method_name in method_names:
        mrows = by_method[method_name]
        mean_agg = sum(r["aggregate_mean_abs_bias"] for r in mrows) / float(len(mrows))
        worst_agg = max(r["aggregate_mean_abs_bias"] for r in mrows)
        mean_gap = sum(r["generalization_gap_vs_baseline"] for r in mrows) / float(len(mrows))
        method_summary.append((method_name, mean_agg, worst_agg, mean_gap))
    for method_name, mean_agg, worst_agg, mean_gap in sorted(method_summary, key=lambda t: t[1]):
        print(f"{method_name} | {mean_agg:.4f} | {worst_agg:.4f} | {mean_gap:+.4f}")

    if args.csv:
        print(f"\nWrote CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
