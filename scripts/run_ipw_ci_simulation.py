#!/usr/bin/env python3
"""Run TreeIPW empirical-Bernstein CI simulations with known ground truth."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.ipw_simulation import (
    ChunkScenario,
    SamplingDesign,
    evaluate_empirical_bernstein_coverage,
    generate_chunk_population,
)
from src.tree.ipw_toy_problems import (
    ChunkGranularity,
    ChunkPattern,
    ImbalanceProfile,
    LengthProfile,
    OraclePreferenceProfile,
    run_mergeable_sketch_examples,
    run_toy_coverage_suite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate IPW empirical-Bernstein CIs on synthetic chunk populations."
    )
    parser.add_argument(
        "--population-model",
        choices=["synthetic", "toy"],
        default="synthetic",
        help="Simulation family to run (default: synthetic).",
    )
    parser.add_argument(
        "--scenario",
        choices=["separable", "nonseparable", "doc-nonseparable", "both", "all"],
        default="all",
        help="Population type to simulate (default: all).",
    )
    parser.add_argument(
        "--design",
        choices=["bernoulli", "wor", "compare"],
        default="compare",
        help="Sampling design for logged labels (default: compare).",
    )
    parser.add_argument("--n-docs", type=int, default=80, help="Number of simulated documents.")
    parser.add_argument(
        "--chunks-per-doc",
        type=int,
        default=10,
        help="Fixed number of chunks per document.",
    )
    parser.add_argument(
        "--granularity",
        choices=["word", "char", "both"],
        default="both",
        help="Toy chunk granularity (default: both).",
    )
    parser.add_argument(
        "--pattern",
        choices=["uniform", "front-loaded", "back-loaded", "alternating", "spike", "boundary", "all"],
        default="all",
        help="Toy chunk-importance pattern (default: all).",
    )
    parser.add_argument(
        "--imbalance",
        choices=["balanced", "moderate", "severe", "adversarial", "all"],
        default="all",
        help="Toy propensity imbalance profile (default: all).",
    )
    parser.add_argument(
        "--toy-matrix",
        action="store_true",
        help=(
            "Run curated worst-case toy matrix "
            "(word+char, front/back/spike/boundary, moderate/severe/adversarial)."
        ),
    )
    parser.add_argument(
        "--toy-mergeable-examples",
        action="store_true",
        help="Run curated positive/negative mergeable-sketch examples with non-additive oracle preferences.",
    )
    parser.add_argument(
        "--length-profile",
        choices=["fixed", "uniform", "bimodal", "long-tail", "all"],
        default="all",
        help="Toy per-document chunk-count profile (default: all).",
    )
    parser.add_argument(
        "--oracle-preference",
        choices=["legacy-smooth", "additive-mean", "topk-spike", "quorum-gate", "hybrid-extreme", "all"],
        default="legacy-smooth",
        help="Toy doc-level oracle preference functional (default: legacy-smooth).",
    )
    parser.add_argument(
        "--min-chunks-per-doc",
        type=int,
        default=None,
        help="Toy minimum chunks per doc for variable-length profiles.",
    )
    parser.add_argument(
        "--max-chunks-per-doc",
        type=int,
        default=None,
        help="Toy maximum chunks per doc for variable-length profiles.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=300,
        help="Monte Carlo trials per scenario.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.10,
        help="Delta used for two-sided empirical-Bernstein intervals.",
    )
    parser.add_argument(
        "--wor-docs",
        type=int,
        default=None,
        help="Fixed docs sampled under WOR design (default: derived from average propensity).",
    )
    parser.add_argument(
        "--wor-chunks-per-doc",
        type=int,
        default=None,
        help="Fixed chunks per selected doc under WOR design (default: derived from average propensity).",
    )
    parser.add_argument(
        "--population-seed",
        type=int,
        default=17,
        help="Base RNG seed used for synthetic population generation.",
    )
    parser.add_argument(
        "--trial-seed",
        type=int,
        default=23,
        help="Base RNG seed used for trial sampling.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )
    parser.add_argument(
        "--enforce-target",
        action="store_true",
        help="Exit non-zero if empirical coverage falls below target by tolerance.",
    )
    parser.add_argument(
        "--coverage-tolerance",
        type=float,
        default=0.03,
        help="Allowed coverage shortfall from target when --enforce-target is set.",
    )
    return parser.parse_args()


def _scenarios_from_arg(arg: str) -> List[ChunkScenario]:
    if arg == "both":
        return [ChunkScenario.SEPARABLE, ChunkScenario.NONSEPARABLE]
    if arg == "all":
        return [ChunkScenario.SEPARABLE, ChunkScenario.NONSEPARABLE, ChunkScenario.DOC_NONSEPARABLE]
    return [ChunkScenario(arg)]


def _designs_from_arg(arg: str) -> List[SamplingDesign]:
    if arg == "compare":
        return [SamplingDesign.BERNOULLI, SamplingDesign.WOR]
    return [SamplingDesign(arg)]


def _granularities_from_arg(arg: str) -> List[ChunkGranularity]:
    if arg == "both":
        return [ChunkGranularity.WORD, ChunkGranularity.CHAR]
    return [ChunkGranularity(arg)]


def _patterns_from_arg(arg: str) -> List[ChunkPattern]:
    if arg == "all":
        return [
            ChunkPattern.UNIFORM,
            ChunkPattern.FRONT_LOADED,
            ChunkPattern.BACK_LOADED,
            ChunkPattern.ALTERNATING,
            ChunkPattern.SPIKE,
            ChunkPattern.BOUNDARY,
        ]
    return [ChunkPattern(arg)]


def _imbalances_from_arg(arg: str) -> List[ImbalanceProfile]:
    if arg == "all":
        return [
            ImbalanceProfile.BALANCED,
            ImbalanceProfile.MODERATE,
            ImbalanceProfile.SEVERE,
            ImbalanceProfile.ADVERSARIAL,
        ]
    return [ImbalanceProfile(arg)]


def _toy_matrix_defaults() -> tuple[List[ChunkGranularity], List[ChunkPattern], List[ImbalanceProfile]]:
    return (
        [ChunkGranularity.WORD, ChunkGranularity.CHAR],
        [
            ChunkPattern.FRONT_LOADED,
            ChunkPattern.BACK_LOADED,
            ChunkPattern.SPIKE,
            ChunkPattern.BOUNDARY,
        ],
        [ImbalanceProfile.MODERATE, ImbalanceProfile.SEVERE, ImbalanceProfile.ADVERSARIAL],
    )


def _length_profiles_from_arg(arg: str) -> List[LengthProfile]:
    if arg == "all":
        return [
            LengthProfile.FIXED,
            LengthProfile.UNIFORM,
            LengthProfile.BIMODAL,
            LengthProfile.LONG_TAIL,
        ]
    return [LengthProfile(arg)]


def _oracle_preferences_from_arg(arg: str) -> List[OraclePreferenceProfile]:
    if arg == "all":
        return [
            OraclePreferenceProfile.LEGACY_SMOOTH,
            OraclePreferenceProfile.ADDITIVE_MEAN,
            OraclePreferenceProfile.TOPK_SPIKE,
            OraclePreferenceProfile.QUORUM_GATE,
            OraclePreferenceProfile.HYBRID_EXTREME,
        ]
    return [OraclePreferenceProfile(arg)]


def _toy_matrix_length_defaults() -> List[LengthProfile]:
    return [LengthProfile.BIMODAL, LengthProfile.LONG_TAIL]


def main() -> int:
    args = parse_args()
    scenarios = _scenarios_from_arg(args.scenario)
    designs = _designs_from_arg(args.design)

    records = []
    if args.population_model == "synthetic":
        run_idx = 0
        for scenario in scenarios:
            population = generate_chunk_population(
                n_docs=args.n_docs,
                chunks_per_doc=args.chunks_per_doc,
                scenario=scenario,
                seed=args.population_seed + run_idx,
            )
            for design in designs:
                result = evaluate_empirical_bernstein_coverage(
                    population,
                    n_trials=args.trials,
                    delta=args.delta,
                    seed=args.trial_seed + run_idx,
                    sampling_design=design,
                    wor_docs_sample=args.wor_docs,
                    wor_chunks_per_doc_sample=args.wor_chunks_per_doc,
                )
                record = asdict(result)
                record.update(
                    {
                        "population_model": "synthetic",
                        "example_name": "synthetic",
                        "expectation": "n/a",
                        "description": "",
                        "granularity": "synthetic",
                        "pattern": "synthetic",
                        "imbalance": "synthetic",
                        "length_profile": "synthetic",
                        "oracle_preference": "synthetic",
                        "min_joint_propensity": None,
                        "p10_joint_propensity": None,
                        "median_joint_propensity": None,
                        "max_joint_weight": None,
                        "high_signal_low_propensity_overlap": None,
                        "min_doc_length": None,
                        "p50_doc_length": None,
                        "p90_doc_length": None,
                        "max_doc_length": None,
                        "ipw_violation_bias": result.ipw_violation_bias,
                        "ipw_preference_bias": result.ipw_preference_bias,
                        "naive_violation_coverage": result.naive_violation_coverage,
                        "naive_preference_coverage": result.naive_preference_coverage,
                        "naive_violation_mean_width": result.naive_violation_mean_width,
                        "naive_preference_mean_width": result.naive_preference_mean_width,
                        "naive_violation_bias": result.naive_violation_bias,
                        "naive_preference_bias": result.naive_preference_bias,
                    }
                )
                records.append(record)
                run_idx += 1
    else:
        if args.toy_mergeable_examples:
            mergeable_runs = run_mergeable_sketch_examples(
                designs=designs,
                n_docs=args.n_docs,
                chunks_per_doc=args.chunks_per_doc,
                min_chunks_per_doc=args.min_chunks_per_doc,
                max_chunks_per_doc=args.max_chunks_per_doc,
                n_trials=args.trials,
                delta=args.delta,
                population_seed=args.population_seed,
                trial_seed=args.trial_seed,
                wor_docs_sample=args.wor_docs,
                wor_chunks_per_doc_sample=args.wor_chunks_per_doc,
            )
            for run in mergeable_runs:
                records.append(
                    {
                        "population_model": "toy",
                        "example_name": run.example_name,
                        "expectation": run.expectation.value,
                        "description": run.description,
                        "scenario": run.scenario.value,
                        "sampling_design": run.sampling_design.value,
                        "granularity": run.granularity.value,
                        "pattern": run.pattern.value,
                        "imbalance": run.imbalance.value,
                        "length_profile": run.length_profile.value,
                        "oracle_preference": run.oracle_preference.value,
                        "delta": run.coverage["delta"],
                        "n_trials": run.coverage["n_trials"],
                        "true_violation_rate": run.coverage["true_violation_rate"],
                        "true_preference_loss": run.coverage["true_preference_loss"],
                        "violation_coverage": run.coverage["violation_coverage"],
                        "preference_coverage": run.coverage["preference_coverage"],
                        "violation_mean_width": run.coverage["violation_mean_width"],
                        "preference_mean_width": run.coverage["preference_mean_width"],
                        "mean_sample_count": run.coverage["mean_sample_count"],
                        "mean_effective_sample_size": run.coverage["mean_effective_sample_size"],
                        "empty_sample_rate": run.coverage["empty_sample_rate"],
                        "ipw_violation_bias": run.coverage["ipw_violation_bias"],
                        "ipw_preference_bias": run.coverage["ipw_preference_bias"],
                        "naive_violation_coverage": run.coverage["naive_violation_coverage"],
                        "naive_preference_coverage": run.coverage["naive_preference_coverage"],
                        "naive_violation_mean_width": run.coverage["naive_violation_mean_width"],
                        "naive_preference_mean_width": run.coverage["naive_preference_mean_width"],
                        "naive_violation_bias": run.coverage["naive_violation_bias"],
                        "naive_preference_bias": run.coverage["naive_preference_bias"],
                        "min_joint_propensity": run.diagnostics.min_joint_propensity,
                        "p10_joint_propensity": run.diagnostics.p10_joint_propensity,
                        "median_joint_propensity": run.diagnostics.median_joint_propensity,
                        "max_joint_weight": run.diagnostics.max_joint_weight,
                        "high_signal_low_propensity_overlap": run.diagnostics.high_signal_low_propensity_overlap,
                        "min_doc_length": run.diagnostics.min_doc_length,
                        "p50_doc_length": run.diagnostics.p50_doc_length,
                        "p90_doc_length": run.diagnostics.p90_doc_length,
                        "max_doc_length": run.diagnostics.max_doc_length,
                    }
                )
        else:
            if args.toy_matrix:
                granularities, patterns, imbalances = _toy_matrix_defaults()
                length_profiles = _toy_matrix_length_defaults()
            else:
                granularities = _granularities_from_arg(args.granularity)
                patterns = _patterns_from_arg(args.pattern)
                imbalances = _imbalances_from_arg(args.imbalance)
                length_profiles = _length_profiles_from_arg(args.length_profile)

            oracle_preferences = _oracle_preferences_from_arg(args.oracle_preference)
            toy_runs = run_toy_coverage_suite(
                scenarios=scenarios,
                designs=designs,
                granularities=granularities,
                patterns=patterns,
                imbalances=imbalances,
                length_profiles=length_profiles,
                oracle_preferences=oracle_preferences,
                n_docs=args.n_docs,
                chunks_per_doc=args.chunks_per_doc,
                min_chunks_per_doc=args.min_chunks_per_doc,
                max_chunks_per_doc=args.max_chunks_per_doc,
                n_trials=args.trials,
                delta=args.delta,
                population_seed=args.population_seed,
                trial_seed=args.trial_seed,
                wor_docs_sample=args.wor_docs,
                wor_chunks_per_doc_sample=args.wor_chunks_per_doc,
            )
            for run in toy_runs:
                records.append(
                    {
                        "population_model": "toy",
                        "example_name": "grid",
                        "expectation": "n/a",
                        "description": "",
                        "scenario": run.scenario.value,
                        "sampling_design": run.sampling_design.value,
                        "granularity": run.granularity.value,
                        "pattern": run.pattern.value,
                        "imbalance": run.imbalance.value,
                        "length_profile": run.length_profile.value,
                        "oracle_preference": run.oracle_preference.value,
                        "delta": run.coverage["delta"],
                        "n_trials": run.coverage["n_trials"],
                        "true_violation_rate": run.coverage["true_violation_rate"],
                        "true_preference_loss": run.coverage["true_preference_loss"],
                        "violation_coverage": run.coverage["violation_coverage"],
                        "preference_coverage": run.coverage["preference_coverage"],
                        "violation_mean_width": run.coverage["violation_mean_width"],
                        "preference_mean_width": run.coverage["preference_mean_width"],
                        "mean_sample_count": run.coverage["mean_sample_count"],
                        "mean_effective_sample_size": run.coverage["mean_effective_sample_size"],
                        "empty_sample_rate": run.coverage["empty_sample_rate"],
                        "ipw_violation_bias": run.coverage["ipw_violation_bias"],
                        "ipw_preference_bias": run.coverage["ipw_preference_bias"],
                        "naive_violation_coverage": run.coverage["naive_violation_coverage"],
                        "naive_preference_coverage": run.coverage["naive_preference_coverage"],
                        "naive_violation_mean_width": run.coverage["naive_violation_mean_width"],
                        "naive_preference_mean_width": run.coverage["naive_preference_mean_width"],
                        "naive_violation_bias": run.coverage["naive_violation_bias"],
                        "naive_preference_bias": run.coverage["naive_preference_bias"],
                        "min_joint_propensity": run.diagnostics.min_joint_propensity,
                        "p10_joint_propensity": run.diagnostics.p10_joint_propensity,
                        "median_joint_propensity": run.diagnostics.median_joint_propensity,
                        "max_joint_weight": run.diagnostics.max_joint_weight,
                        "high_signal_low_propensity_overlap": run.diagnostics.high_signal_low_propensity_overlap,
                        "min_doc_length": run.diagnostics.min_doc_length,
                        "p50_doc_length": run.diagnostics.p50_doc_length,
                        "p90_doc_length": run.diagnostics.p90_doc_length,
                        "max_doc_length": run.diagnostics.max_doc_length,
                    }
                )

    if args.json:
        json.dump(records, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        target = 1.0 - args.delta
        print(
            f"model={args.population_model}  target_coverage={target:.3f}  trials={args.trials}  "
            f"n_docs={args.n_docs}  chunks_per_doc={args.chunks_per_doc}"
        )
        for record in records:
            prefix = ""
            if record.get("example_name") and record["example_name"] not in ("grid", "synthetic"):
                prefix = f"{record['example_name']} ({record.get('expectation', 'n/a')}) "
            print(
                f"[{prefix}{record['scenario']} | {record['sampling_design']} | "
                f"{record['granularity']} | {record['pattern']} | {record['imbalance']} | "
                f"{record['length_profile']} | {record['oracle_preference']}]"
            )
            print(
                "  true_violation={:.4f}  true_preference={:.4f}".format(
                    record["true_violation_rate"],
                    record["true_preference_loss"],
                )
            )
            print(
                "  coverage(violation={:.3f}, preference={:.3f})".format(
                    record["violation_coverage"],
                    record["preference_coverage"],
                )
            )
            print(
                "  naive_coverage(violation={:.3f}, preference={:.3f})".format(
                    record["naive_violation_coverage"],
                    record["naive_preference_coverage"],
                )
            )
            print(
                "  mean_width(violation={:.3f}, preference={:.3f})".format(
                    record["violation_mean_width"],
                    record["preference_mean_width"],
                )
            )
            print(
                "  naive_width(violation={:.3f}, preference={:.3f})".format(
                    record["naive_violation_mean_width"],
                    record["naive_preference_mean_width"],
                )
            )
            print(
                "  bias_ipw(violation={:+.4f}, preference={:+.4f})  "
                "bias_naive(violation={:+.4f}, preference={:+.4f})".format(
                    record["ipw_violation_bias"],
                    record["ipw_preference_bias"],
                    record["naive_violation_bias"],
                    record["naive_preference_bias"],
                )
            )
            print(
                "  mean_sample_count={:.1f}  mean_n_eff={:.1f}  empty_rate={:.3f}".format(
                    record["mean_sample_count"],
                    record["mean_effective_sample_size"],
                    record["empty_sample_rate"],
                )
            )
            if record["min_joint_propensity"] is not None:
                print(
                    "  joint_propensity(min={:.4f}, p10={:.4f}, median={:.4f})  "
                    "max_weight={:.1f}  high_signal_low_prop_overlap={:.3f}".format(
                        record["min_joint_propensity"],
                        record["p10_joint_propensity"],
                        record["median_joint_propensity"],
                        record["max_joint_weight"],
                        record["high_signal_low_propensity_overlap"],
                    )
                )
                print(
                    "  doc_lengths(min={}, p50={:.1f}, p90={:.1f}, max={})".format(
                        int(record["min_doc_length"]),
                        float(record["p50_doc_length"]),
                        float(record["p90_doc_length"]),
                        int(record["max_doc_length"]),
                    )
                )

    if args.enforce_target:
        target = 1.0 - args.delta
        threshold = target - max(0.0, float(args.coverage_tolerance))
        failures = []
        for record in records:
            case_tag = f"{record.get('example_name', 'case')}:{record['scenario']}|{record['sampling_design']}"
            if record["violation_coverage"] < threshold:
                failures.append(
                    f"{case_tag}: "
                    f"violation_coverage={record['violation_coverage']:.3f} < {threshold:.3f}"
                )
            if record["preference_coverage"] < threshold:
                failures.append(
                    f"{case_tag}: "
                    f"preference_coverage={record['preference_coverage']:.3f} < {threshold:.3f}"
                )
        if failures:
            for msg in failures:
                print(f"coverage_gate_failed: {msg}", file=sys.stderr)
            return 2
        print(
            f"coverage_gate_passed: target={target:.3f}, tolerance={args.coverage_tolerance:.3f}, "
            f"threshold={threshold:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
