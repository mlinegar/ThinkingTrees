#!/usr/bin/env python3
"""Run large IPW stress-ladder simulations (IPW vs naive-unweighted baseline)."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from typing import Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.ipw_simulation import (  # noqa: E402
    ChunkScenario,
    SamplingDesign,
    evaluate_empirical_bernstein_coverage,
)
from src.tree.ipw_toy_problems import (  # noqa: E402
    ChunkGranularity,
    ChunkPattern,
    ImbalanceProfile,
    LengthProfile,
    OraclePreferenceProfile,
    generate_toy_chunk_population,
    toy_population_diagnostics,
)


def _parse_int_csv(s: str) -> Tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in s.split(",") if x.strip())
    if len(vals) == 0:
        raise ValueError("expected non-empty comma-separated int list")
    return vals


@dataclass(frozen=True)
class StressCase:
    name: str
    scenario: ChunkScenario
    granularity: ChunkGranularity
    pattern: ChunkPattern
    imbalance: ImbalanceProfile
    length_profile: LengthProfile
    oracle_preference: OraclePreferenceProfile
    design: SamplingDesign


@dataclass(frozen=True)
class StressTask:
    case: StressCase
    n_docs: int
    chunks_per_doc: int
    min_chunks_per_doc: int
    max_chunks_per_doc: int
    n_trials: int
    delta: float
    population_seed: int
    trial_seed: int


def _base_cases() -> Tuple[StressCase, ...]:
    return (
        StressCase(
            name="control_balanced_bernoulli",
            scenario=ChunkScenario.SEPARABLE,
            granularity=ChunkGranularity.CHAR,
            pattern=ChunkPattern.ALTERNATING,
            imbalance=ImbalanceProfile.BALANCED,
            length_profile=LengthProfile.BIMODAL,
            oracle_preference=OraclePreferenceProfile.HYBRID_EXTREME,
            design=SamplingDesign.BERNOULLI,
        ),
        StressCase(
            name="stress_adversarial_bernoulli",
            scenario=ChunkScenario.SEPARABLE,
            granularity=ChunkGranularity.CHAR,
            pattern=ChunkPattern.ALTERNATING,
            imbalance=ImbalanceProfile.ADVERSARIAL,
            length_profile=LengthProfile.BIMODAL,
            oracle_preference=OraclePreferenceProfile.HYBRID_EXTREME,
            design=SamplingDesign.BERNOULLI,
        ),
        StressCase(
            name="stress_adversarial_wor",
            scenario=ChunkScenario.SEPARABLE,
            granularity=ChunkGranularity.CHAR,
            pattern=ChunkPattern.ALTERNATING,
            imbalance=ImbalanceProfile.ADVERSARIAL,
            length_profile=LengthProfile.BIMODAL,
            oracle_preference=OraclePreferenceProfile.HYBRID_EXTREME,
            design=SamplingDesign.WOR,
        ),
    )


def _hard_cases() -> Tuple[StressCase, ...]:
    return (
        StressCase(
            name="hard_control_additive_balanced",
            scenario=ChunkScenario.DOC_NONSEPARABLE,
            granularity=ChunkGranularity.WORD,
            pattern=ChunkPattern.UNIFORM,
            imbalance=ImbalanceProfile.BALANCED,
            length_profile=LengthProfile.FIXED,
            oracle_preference=OraclePreferenceProfile.ADDITIVE_MEAN,
            design=SamplingDesign.BERNOULLI,
        ),
        StressCase(
            name="hard_topk_spike_adversarial",
            scenario=ChunkScenario.DOC_NONSEPARABLE,
            granularity=ChunkGranularity.CHAR,
            pattern=ChunkPattern.SPIKE,
            imbalance=ImbalanceProfile.ADVERSARIAL,
            length_profile=LengthProfile.LONG_TAIL,
            oracle_preference=OraclePreferenceProfile.TOPK_SPIKE,
            design=SamplingDesign.BERNOULLI,
        ),
        StressCase(
            name="hard_hybrid_boundary_adversarial",
            scenario=ChunkScenario.DOC_NONSEPARABLE,
            granularity=ChunkGranularity.WORD,
            pattern=ChunkPattern.BOUNDARY,
            imbalance=ImbalanceProfile.ADVERSARIAL,
            length_profile=LengthProfile.BIMODAL,
            oracle_preference=OraclePreferenceProfile.HYBRID_EXTREME,
            design=SamplingDesign.WOR,
        ),
        StressCase(
            name="hard_quorum_backloaded_severe",
            scenario=ChunkScenario.DOC_NONSEPARABLE,
            granularity=ChunkGranularity.CHAR,
            pattern=ChunkPattern.BACK_LOADED,
            imbalance=ImbalanceProfile.SEVERE,
            length_profile=LengthProfile.LONG_TAIL,
            oracle_preference=OraclePreferenceProfile.QUORUM_GATE,
            design=SamplingDesign.WOR,
        ),
    )


def _resolve_cases(case_set: str) -> Tuple[StressCase, ...]:
    if case_set == "base":
        return _base_cases()
    if case_set == "hard":
        return _hard_cases()
    if case_set == "both":
        return _base_cases() + _hard_cases()
    raise ValueError(f"Unsupported case_set: {case_set!r}")


def _build_tasks(
    *,
    cases: Sequence[StressCase],
    n_docs_values: Sequence[int],
    chunks_per_doc: int,
    min_chunks_per_doc: int,
    max_chunks_per_doc: int,
    n_trials: int,
    delta: float,
    n_population_seeds: int,
    population_seed_base: int,
    trial_seed_base: int,
) -> List[StressTask]:
    tasks: List[StressTask] = []
    for case_idx, case in enumerate(cases):
        for n_idx, n_docs in enumerate(n_docs_values):
            for s_idx in range(n_population_seeds):
                # Deterministic seed mapping with large offsets to avoid collisions.
                seed_offset = (case_idx * 1_000_000) + (n_idx * 10_000) + s_idx
                tasks.append(
                    StressTask(
                        case=case,
                        n_docs=int(n_docs),
                        chunks_per_doc=int(chunks_per_doc),
                        min_chunks_per_doc=int(min_chunks_per_doc),
                        max_chunks_per_doc=int(max_chunks_per_doc),
                        n_trials=int(n_trials),
                        delta=float(delta),
                        population_seed=int(population_seed_base + seed_offset),
                        trial_seed=int(trial_seed_base + seed_offset),
                    )
                )
    return tasks


def _worker(task: StressTask) -> Dict[str, float | int | str]:
    case = task.case
    population = generate_toy_chunk_population(
        n_docs=int(task.n_docs),
        chunks_per_doc=int(task.chunks_per_doc),
        scenario=case.scenario,
        granularity=case.granularity,
        pattern=case.pattern,
        imbalance=case.imbalance,
        length_profile=case.length_profile,
        oracle_preference=case.oracle_preference,
        min_chunks_per_doc=int(task.min_chunks_per_doc),
        max_chunks_per_doc=int(task.max_chunks_per_doc),
        seed=int(task.population_seed),
    )
    diag = toy_population_diagnostics(population)
    result = evaluate_empirical_bernstein_coverage(
        population,
        n_trials=int(task.n_trials),
        delta=float(task.delta),
        seed=int(task.trial_seed),
        sampling_design=case.design,
    )

    row: Dict[str, float | int | str] = {
        "case": case.name,
        "scenario": case.scenario.value,
        "sampling_design": case.design.value,
        "granularity": case.granularity.value,
        "pattern": case.pattern.value,
        "imbalance": case.imbalance.value,
        "length_profile": case.length_profile.value,
        "oracle_preference": case.oracle_preference.value,
        "n_docs": int(task.n_docs),
        "chunks_per_doc": int(task.chunks_per_doc),
        "min_chunks_per_doc": int(task.min_chunks_per_doc),
        "max_chunks_per_doc": int(task.max_chunks_per_doc),
        "population_seed": int(task.population_seed),
        "trial_seed": int(task.trial_seed),
        "delta": float(result.delta),
        "n_trials": int(result.n_trials),
        "true_violation_rate": float(result.true_violation_rate),
        "true_preference_loss": float(result.true_preference_loss),
        "violation_coverage": float(result.violation_coverage),
        "preference_coverage": float(result.preference_coverage),
        "violation_mean_width": float(result.violation_mean_width),
        "preference_mean_width": float(result.preference_mean_width),
        "mean_sample_count": float(result.mean_sample_count),
        "mean_effective_sample_size": float(result.mean_effective_sample_size),
        "empty_sample_rate": float(result.empty_sample_rate),
        "ipw_violation_bias": float(result.ipw_violation_bias),
        "ipw_preference_bias": float(result.ipw_preference_bias),
        "naive_violation_coverage": float(result.naive_violation_coverage),
        "naive_preference_coverage": float(result.naive_preference_coverage),
        "naive_violation_mean_width": float(result.naive_violation_mean_width),
        "naive_preference_mean_width": float(result.naive_preference_mean_width),
        "naive_violation_bias": float(result.naive_violation_bias),
        "naive_preference_bias": float(result.naive_preference_bias),
        "min_joint_propensity": float(diag.min_joint_propensity),
        "p10_joint_propensity": float(diag.p10_joint_propensity),
        "median_joint_propensity": float(diag.median_joint_propensity),
        "max_joint_weight": float(diag.max_joint_weight),
        "high_signal_low_propensity_overlap": float(diag.high_signal_low_propensity_overlap),
    }
    return row


def _mean(xs: Sequence[float]) -> float:
    if len(xs) == 0:
        return float("nan")
    return float(sum(xs) / float(len(xs)))


def _std(xs: Sequence[float]) -> float:
    n = len(xs)
    if n <= 1:
        return 0.0
    mu = _mean(xs)
    return float((sum((x - mu) ** 2 for x in xs) / float(n - 1)) ** 0.5)


def _aggregate_rows(rows: Sequence[Dict[str, float | int | str]]) -> List[Dict[str, float | int | str]]:
    metrics = (
        "violation_coverage",
        "naive_violation_coverage",
        "preference_coverage",
        "naive_preference_coverage",
        "violation_mean_width",
        "naive_violation_mean_width",
        "preference_mean_width",
        "naive_preference_mean_width",
        "ipw_violation_bias",
        "naive_violation_bias",
        "ipw_preference_bias",
        "naive_preference_bias",
        "mean_effective_sample_size",
        "mean_sample_count",
        "max_joint_weight",
        "min_joint_propensity",
        "high_signal_low_propensity_overlap",
    )
    by_key: Dict[Tuple[str, int], List[Dict[str, float | int | str]]] = {}
    for row in rows:
        key = (str(row["case"]), int(row["n_docs"]))
        by_key.setdefault(key, []).append(row)

    out: List[Dict[str, float | int | str]] = []
    for (case, n_docs), grp in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        base = grp[0]
        rec: Dict[str, float | int | str] = {
            "case": case,
            "n_docs": n_docs,
            "n_population_seeds": len(grp),
            "scenario": str(base["scenario"]),
            "sampling_design": str(base["sampling_design"]),
            "granularity": str(base["granularity"]),
            "pattern": str(base["pattern"]),
            "imbalance": str(base["imbalance"]),
            "length_profile": str(base["length_profile"]),
            "oracle_preference": str(base["oracle_preference"]),
            "delta": float(base["delta"]),
            "n_trials": int(base["n_trials"]),
            "true_violation_rate_mean": _mean([float(r["true_violation_rate"]) for r in grp]),
            "true_preference_loss_mean": _mean([float(r["true_preference_loss"]) for r in grp]),
        }
        for m in metrics:
            vals = [float(r[m]) for r in grp]
            rec[f"{m}_mean"] = _mean(vals)
            rec[f"{m}_std"] = _std(vals)
        out.append(rec)
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, float | int | str]]) -> None:
    if len(rows) == 0:
        return
    keys: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k in seen:
                continue
            keys.append(k)
            seen.add(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})


def _format_duration(seconds: float) -> str:
    s = max(0, int(round(float(seconds))))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run large IPW stress-ladder simulations.")
    p.add_argument(
        "--case-set",
        choices=["base", "hard", "both"],
        default="base",
        help="Stress-case bundle to run (default: base).",
    )
    p.add_argument("--n-docs-values", type=str, default="60,120,240,480,960")
    p.add_argument("--chunks-per-doc", type=int, default=20)
    p.add_argument("--min-chunks-per-doc", type=int, default=4)
    p.add_argument("--max-chunks-per-doc", type=int, default=50)
    p.add_argument("--trials", type=int, default=1200)
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--n-population-seeds", type=int, default=10)
    p.add_argument("--population-seed-base", type=int, default=17_000)
    p.add_argument("--trial-seed-base", type=int, default=23_000)
    p.add_argument("--jobs", type=int, default=0, help="Worker processes (<=0 => cpu_count).")
    p.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Progress print frequency in completed tasks (default: 10).",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Checkpoint frequency for partial raw/summary CSV writes (default: 10).",
    )
    p.add_argument(
        "--output-csv",
        type=str,
        default="outputs/ipw_stress_ladder/raw_rows.csv",
        help="Per-task raw rows CSV.",
    )
    p.add_argument(
        "--output-summary-csv",
        type=str,
        default="outputs/ipw_stress_ladder/summary_rows.csv",
        help="Aggregated summary CSV.",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="outputs/ipw_stress_ladder/summary.json",
        help="JSON summary path.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    n_docs_values = _parse_int_csv(str(args.n_docs_values))
    cases = _resolve_cases(str(args.case_set))
    tasks = _build_tasks(
        cases=cases,
        n_docs_values=n_docs_values,
        chunks_per_doc=int(args.chunks_per_doc),
        min_chunks_per_doc=int(args.min_chunks_per_doc),
        max_chunks_per_doc=int(args.max_chunks_per_doc),
        n_trials=int(args.trials),
        delta=float(args.delta),
        n_population_seeds=int(args.n_population_seeds),
        population_seed_base=int(args.population_seed_base),
        trial_seed_base=int(args.trial_seed_base),
    )
    if len(tasks) == 0:
        raise ValueError("no tasks generated")

    max_workers = int(args.jobs)
    if max_workers <= 0:
        import os

        max_workers = max(1, (os.cpu_count() or 1))

    progress_every = max(1, int(args.progress_every))
    checkpoint_every = max(1, int(args.checkpoint_every))
    out_raw = Path(args.output_csv)
    out_summary_csv = Path(args.output_summary_csv)
    out_json = Path(args.output_json)
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)
    clock_start = time.monotonic()
    rows: List[Dict[str, float | int | str]] = []
    total = len(tasks)

    with out_raw.open("w", encoding="utf-8", newline="") as raw_file:
        raw_writer: csv.DictWriter | None = None
        raw_keys: List[str] = []

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_worker, task) for task in tasks]
            done = 0
            for fut in as_completed(futs):
                row = fut.result()
                rows.append(row)
                done += 1

                if raw_writer is None:
                    raw_keys = list(row.keys())
                    raw_writer = csv.DictWriter(raw_file, fieldnames=raw_keys)
                    raw_writer.writeheader()
                raw_writer.writerow({k: row.get(k, "") for k in raw_keys})

                checkpoint_due = (done % checkpoint_every == 0) or (done == total)
                progress_due = (done % progress_every == 0) or (done == total)

                if checkpoint_due:
                    raw_file.flush()
                    _write_csv(out_summary_csv, _aggregate_rows(rows))

                if progress_due:
                    elapsed = time.monotonic() - clock_start
                    rate = float(done) / max(1e-9, elapsed)
                    eta_seconds = float(total - done) / max(1e-9, rate)
                    print(
                        "progress | "
                        f"{done}/{total} | "
                        f"elapsed={_format_duration(elapsed)} | "
                        f"eta={_format_duration(eta_seconds)} | "
                        f"rows={len(rows)}",
                        flush=True,
                    )

    summary_rows = _aggregate_rows(rows)
    _write_csv(out_summary_csv, summary_rows)

    finished = datetime.now(timezone.utc)
    payload = {
        "generated_at_utc": finished.strftime("%Y-%m-%d %H:%M:%SZ"),
        "started_at_utc": started.strftime("%Y-%m-%d %H:%M:%SZ"),
        "duration_seconds": (finished - started).total_seconds(),
        "config": {
            "case_set": str(args.case_set),
            "n_docs_values": list(n_docs_values),
            "chunks_per_doc": int(args.chunks_per_doc),
            "min_chunks_per_doc": int(args.min_chunks_per_doc),
            "max_chunks_per_doc": int(args.max_chunks_per_doc),
            "trials": int(args.trials),
            "delta": float(args.delta),
            "n_population_seeds": int(args.n_population_seeds),
            "population_seed_base": int(args.population_seed_base),
            "trial_seed_base": int(args.trial_seed_base),
            "jobs": int(max_workers),
            "progress_every": int(progress_every),
            "checkpoint_every": int(checkpoint_every),
            "cases": [
                {
                    "name": c.name,
                    "scenario": c.scenario.value,
                    "granularity": c.granularity.value,
                    "pattern": c.pattern.value,
                    "imbalance": c.imbalance.value,
                    "length_profile": c.length_profile.value,
                    "oracle_preference": c.oracle_preference.value,
                    "design": c.design.value,
                }
                for c in cases
            ],
        },
        "n_raw_rows": len(rows),
        "n_summary_rows": len(summary_rows),
        "output_raw_csv": str(out_raw),
        "output_summary_csv": str(out_summary_csv),
        "summary_rows": summary_rows,
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({"raw_rows": len(rows), "summary_rows": len(summary_rows), "output_json": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
