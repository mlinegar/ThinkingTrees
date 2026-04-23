#!/usr/bin/env python3
"""Adaptive overnight optimizer for learned mergeable-g LDA benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
TREEPO_SRC = REPO_ROOT / "treepo" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TREEPO_SRC) not in sys.path:
    sys.path.insert(0, str(TREEPO_SRC))

from src.ctreepo.sim.util import safe_float
from treepo.bench.lda.learned_segment_lda_ops_g import LearnedSegmentLDAOpsGConfig
from treepo.bench.lda.learned_segmented_lda_theta_g import LearnedSegmentedLDATopicThetaGConfig
from treepo.bench.runner import (
    EXPERIMENT_LEARNED_OPS_G,
    EXPERIMENT_LEARNED_SEGMENTED_THETA_G,
    RunSpec,
    run_specs,
    validate_config_dict,
)


ExperimentName = str


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _set_cpu_thread_env_one() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
            n += 1
    return n


_safe_float = safe_float


def _parse_seeds(text: str) -> List[int]:
    items = [p.strip() for p in str(text).replace(",", " ").split(" ") if p.strip()]
    out = sorted({int(x) for x in items})
    if not out:
        raise ValueError("at least one seed is required")
    return out


def _loss_from_metrics(metrics: Mapping[str, Any]) -> float:
    root = _safe_float(metrics.get("root_mae"), default=1e9)
    merge = _safe_float(metrics.get("merge_mae"), default=1e9)
    leaf = _safe_float(metrics.get("leaf_mae"), default=1e9)
    spread_mean = _safe_float(metrics.get("schedule_spread_mean"), default=1e9)
    spread_p95 = _safe_float(metrics.get("schedule_spread_p95"), default=1e9)
    leaf_v = _safe_float(metrics.get("leaf_violation_rate"), default=1.0)
    merge_v = _safe_float(metrics.get("merge_violation_rate"), default=1.0)
    return float(
        root
        + 0.45 * merge
        + 0.20 * leaf
        + 0.30 * spread_mean
        + 0.10 * spread_p95
        + 0.25 * (leaf_v + merge_v)
    )


def _mean_dict(dicts: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in keys:
        vals = [_safe_float(d.get(key), default=float("nan")) for d in dicts]
        vals = [v for v in vals if math.isfinite(v)]
        out[key] = float(sum(vals) / len(vals)) if vals else float("nan")
    return out


def _clamp_int(value: object, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(float(value))))))


def _clamp_float(value: object, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(value))))


def _scaled_int(value: object, factor: float, *, lo: int, hi: int) -> int:
    return _clamp_int(float(value) * float(factor), lo=lo, hi=hi)


def _to_dataclass_config(experiment: ExperimentName, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    if experiment == EXPERIMENT_LEARNED_OPS_G:
        return asdict(LearnedSegmentLDAOpsGConfig(**dict(cfg)))
    if experiment == EXPERIMENT_LEARNED_SEGMENTED_THETA_G:
        return asdict(LearnedSegmentedLDATopicThetaGConfig(**dict(cfg)))
    raise ValueError(f"unsupported experiment: {experiment}")


def _base_config(experiment: ExperimentName, *, profile: str) -> Dict[str, Any]:
    if experiment == EXPERIMENT_LEARNED_OPS_G:
        cfg = asdict(LearnedSegmentLDAOpsGConfig())
        if profile == "smoke":
            cfg.update(
                {
                    "n_topics": 4,
                    "vocab_size": 64,
                    "anchor_words_per_topic": 8,
                    "min_tokens": 64,
                    "max_tokens": 64,
                    "min_segments": 2,
                    "max_segments": 4,
                    "min_seg_len": 16,
                    "max_seg_len": 32,
                    "leaf_tokens": 8,
                    "train_docs": 48,
                    "test_docs": 48,
                    "state_dim": 16,
                    "hidden_dim": 48,
                    "n_epochs": 2,
                    "batch_docs": 8,
                    "audit_fraction": 0.25,
                    "schedule_consistency_weight": 0.05,
                    "idempotence_weight": 0.02,
                }
            )
        else:
            cfg.update(
                {
                    "train_docs": 384,
                    "test_docs": 384,
                    "state_dim": 48,
                    "hidden_dim": 160,
                    "n_epochs": 10,
                    "batch_docs": 16,
                    "audit_fraction": 0.25,
                    "schedule_consistency_weight": 0.05,
                    "idempotence_weight": 0.02,
                }
            )
    elif experiment == EXPERIMENT_LEARNED_SEGMENTED_THETA_G:
        cfg = asdict(LearnedSegmentedLDATopicThetaGConfig())
        if profile == "smoke":
            cfg.update(
                {
                    "n_topics": 4,
                    "vocab_size": 64,
                    "n_books_train": 48,
                    "n_books_test": 48,
                    "min_segments": 2,
                    "max_segments": 4,
                    "min_seg_tokens": 16,
                    "max_seg_tokens": 32,
                    "fixed_leaf_tokens": 16,
                    "state_dim": 24,
                    "hidden_dim": 64,
                    "n_epochs": 2,
                    "batch_docs": 8,
                    "audit_fraction": 0.25,
                    "schedule_consistency_weight": 0.05,
                    "idempotence_weight": 0.02,
                }
            )
        else:
            cfg.update(
                {
                    "n_books_train": 384,
                    "n_books_test": 384,
                    "state_dim": 64,
                    "hidden_dim": 192,
                    "n_epochs": 10,
                    "batch_docs": 16,
                    "audit_fraction": 0.25,
                    "schedule_consistency_weight": 0.05,
                    "idempotence_weight": 0.02,
                }
            )
    else:
        raise ValueError(f"unsupported experiment: {experiment}")
    cfg["torch_threads"] = 1
    return _to_dataclass_config(experiment, cfg)


def _mutated_candidates(
    experiment: ExperimentName,
    *,
    pivot_config: Mapping[str, Any],
    last_metrics: Optional[Mapping[str, Any]],
    rounds_without_recovery: int,
    n_candidates: int,
) -> List[Dict[str, Any]]:
    base = dict(pivot_config)
    root = _safe_float((last_metrics or {}).get("root_mae"), default=float("nan"))
    merge = _safe_float((last_metrics or {}).get("merge_mae"), default=float("nan"))
    leaf = _safe_float((last_metrics or {}).get("leaf_mae"), default=float("nan"))
    spread = _safe_float((last_metrics or {}).get("schedule_spread_mean"), default=float("nan"))
    pressure = int(max(0, rounds_without_recovery))

    high_spread = math.isfinite(spread) and math.isfinite(root) and spread > max(0.02, 0.30 * root)
    weak_merge = math.isfinite(merge) and math.isfinite(root) and merge > (1.10 * root)
    weak_leaf = math.isfinite(leaf) and math.isfinite(root) and leaf > (1.25 * root)

    train_key = "train_docs" if experiment == EXPERIMENT_LEARNED_OPS_G else "n_books_train"
    max_train = 4096 if experiment == EXPERIMENT_LEARNED_OPS_G else 4096

    def with_updates(cfg: Mapping[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
        out = dict(cfg)
        out.update(dict(updates))
        out["state_dim"] = _clamp_int(out.get("state_dim", 32), 8, 256)
        out["hidden_dim"] = _clamp_int(out.get("hidden_dim", 128), 16, 512)
        out["n_epochs"] = _clamp_int(out.get("n_epochs", 10), 1, 60)
        out["batch_docs"] = _clamp_int(out.get("batch_docs", 16), 1, 128)
        out["lr"] = _clamp_float(out.get("lr", 3e-4), 1e-5, 5e-3)
        out["weight_decay"] = _clamp_float(out.get("weight_decay", 1e-5), 0.0, 1e-2)
        out["leaf_query_rate"] = _clamp_float(out.get("leaf_query_rate", 1.0), 0.0, 1.0)
        out["audit_fraction"] = _clamp_float(out.get("audit_fraction", 0.2), 0.0, 1.0)
        out["audit_scale"] = _clamp_float(out.get("audit_scale", 1.0), 0.1, 8.0)
        out["root_weight"] = _clamp_float(out.get("root_weight", 1.0), 0.0, 8.0)
        out["leaf_weight"] = _clamp_float(out.get("leaf_weight", 0.05), 0.0, 4.0)
        out["c3_weight"] = _clamp_float(out.get("c3_weight", 0.2), 0.0, 8.0)
        out["schedule_consistency_weight"] = _clamp_float(out.get("schedule_consistency_weight", 0.0), 0.0, 4.0)
        out["idempotence_weight"] = _clamp_float(out.get("idempotence_weight", 0.0), 0.0, 4.0)
        out[train_key] = _clamp_int(out.get(train_key, 256), 16, max_train)
        out["torch_threads"] = 1
        return _to_dataclass_config(experiment, out)

    candidates: List[Dict[str, Any]] = []
    candidates.append(with_updates(base, {}))
    candidates.append(
        with_updates(
            base,
            {
                "schedule_consistency_weight": float(base.get("schedule_consistency_weight", 0.0)) * (2.0 + 0.3 * pressure) + 0.01,
                "idempotence_weight": max(float(base.get("idempotence_weight", 0.0)), 0.02),
            },
        )
    )
    candidates.append(
        with_updates(
            base,
            {
                "audit_policy": "all" if pressure >= 2 else "fraction",
                "audit_fraction": float(base.get("audit_fraction", 0.2)) * (1.5 + 0.2 * pressure),
                "c3_weight": float(base.get("c3_weight", 0.2)) * (1.4 + 0.2 * pressure),
            },
        )
    )
    candidates.append(
        with_updates(
            base,
            {
                "leaf_weight": float(base.get("leaf_weight", 0.05)) * (1.5 + 0.1 * pressure),
                "leaf_query_rate": max(float(base.get("leaf_query_rate", 1.0)), 0.95),
                "batch_docs": _scaled_int(base.get("batch_docs", 16), 0.8, lo=1, hi=128),
            },
        )
    )
    candidates.append(
        with_updates(
            base,
            {
                "state_dim": _scaled_int(base.get("state_dim", 32), 1.6 + 0.1 * pressure, lo=8, hi=256),
                "hidden_dim": _scaled_int(base.get("hidden_dim", 128), 1.7 + 0.1 * pressure, lo=16, hi=512),
                "lr": float(base.get("lr", 3e-4)) * 0.85,
            },
        )
    )
    candidates.append(
        with_updates(
            base,
            {
                "n_epochs": _scaled_int(base.get("n_epochs", 10), 1.4 + 0.1 * pressure, lo=1, hi=60),
                "lr": float(base.get("lr", 3e-4)) * 0.7,
            },
        )
    )
    candidates.append(
        with_updates(
            base,
            {
                train_key: _scaled_int(base.get(train_key, 256), 1.5 + 0.2 * pressure, lo=16, hi=max_train),
                "n_epochs": _scaled_int(base.get("n_epochs", 10), 1.2, lo=1, hi=60),
            },
        )
    )
    candidates.append(
        with_updates(
            base,
            {
                "audit_policy": "sqrt" if weak_merge else str(base.get("audit_policy", "fraction")),
                "audit_scale": 1.5 if weak_merge else float(base.get("audit_scale", 1.0)),
                "schedule_consistency_weight": float(base.get("schedule_consistency_weight", 0.0)) * (2.8 if high_spread else 1.2)
                + (0.03 if high_spread else 0.0),
                "c3_weight": float(base.get("c3_weight", 0.2)) * (1.6 if weak_merge else 1.2),
                "leaf_weight": float(base.get("leaf_weight", 0.05)) * (1.6 if weak_leaf else 1.1),
            },
        )
    )

    dedup: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for cfg in candidates:
        key = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(cfg)
    return dedup[: max(1, int(n_candidates))]


def _aggregate_round(
    *,
    run_rows: Sequence[Mapping[str, Any]],
    metric_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in run_rows:
        key = (str(row.get("experiment")), str(row.get("candidate_id")))
        grouped.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for (experiment, candidate_id), rows in grouped.items():
        seed_rows = [r for r in rows if r.get("status") == "ok" and isinstance(r.get("metrics"), dict)]
        metric_maps = [dict(r["metrics"]) for r in seed_rows]  # type: ignore[index]
        mean_metrics = _mean_dict(metric_maps, metric_keys) if metric_maps else {}
        score = _loss_from_metrics(mean_metrics) if metric_maps else float("inf")
        out.append(
            {
                "experiment": experiment,
                "candidate_id": candidate_id,
                "n_runs": int(len(rows)),
                "n_ok": int(len(seed_rows)),
                "score": float(score),
                "mean_metrics": mean_metrics,
                "candidate_config": dict(rows[0].get("candidate_config", {})) if rows else {},
                "seed_details": [
                    {
                        "seed": int(r.get("seed", -1)),
                        "status": r.get("status"),
                        "score": _safe_float(r.get("score"), default=float("nan")),
                        "metrics": dict(r.get("metrics", {})) if isinstance(r.get("metrics"), dict) else None,
                        "json_out": r.get("json_out"),
                    }
                    for r in rows
                ],
            }
        )
    out.sort(key=lambda r: (str(r["experiment"]), float(r["score"])))
    return out


def _init_experiment_state(
    *,
    experiment: ExperimentName,
    base_config: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "experiment": str(experiment),
        "pivot_config": dict(base_config),
        "best_config": dict(base_config),
        "best_score": float("inf"),
        "best_metrics": {},
        "baseline_score": None,
        "baseline_spread": None,
        "last_metrics": None,
        "recovered": False,
        "rounds_without_recovery": 0,
    }


def _update_experiment_state(
    *,
    state: Dict[str, Any],
    ranked_rows: Sequence[Mapping[str, Any]],
    recovery_ratio: float,
    spread_ratio: float,
) -> Dict[str, Any]:
    if not ranked_rows:
        state["rounds_without_recovery"] = int(state.get("rounds_without_recovery", 0)) + 1
        return state

    round_scores = [float(r.get("score", float("inf"))) for r in ranked_rows if math.isfinite(float(r.get("score", float("inf"))))]
    round_spreads = [
        _safe_float((r.get("mean_metrics", {}) or {}).get("schedule_spread_mean"), default=float("nan"))
        for r in ranked_rows
    ]
    round_spreads = [v for v in round_spreads if math.isfinite(v)]

    if state.get("baseline_score") is None and round_scores:
        state["baseline_score"] = float(statistics.median(round_scores))
    if state.get("baseline_spread") is None and round_spreads:
        state["baseline_spread"] = float(statistics.median(round_spreads))

    best = dict(ranked_rows[0])
    best_score = float(best.get("score", float("inf")))
    best_metrics = dict(best.get("mean_metrics", {}))
    best_cfg = dict(best.get("candidate_config", {}))

    if best_score < float(state.get("best_score", float("inf"))):
        state["best_score"] = best_score
        state["best_metrics"] = best_metrics
        state["best_config"] = best_cfg

    state["pivot_config"] = best_cfg
    state["last_metrics"] = best_metrics

    baseline_score = _safe_float(state.get("baseline_score"), default=float("inf"))
    baseline_spread = _safe_float(state.get("baseline_spread"), default=float("inf"))
    best_root = _safe_float(best_metrics.get("root_mae"), default=float("inf"))
    best_merge = _safe_float(best_metrics.get("merge_mae"), default=float("inf"))
    best_spread = _safe_float(best_metrics.get("schedule_spread_mean"), default=float("inf"))

    score_ok = math.isfinite(best_score) and math.isfinite(baseline_score) and best_score <= baseline_score * float(recovery_ratio)
    spread_target = max(1e-3, baseline_spread * float(spread_ratio))
    spread_relative_cap = max(0.25, 2.0 * max(1e-6, best_root))
    merge_relative_cap = max(0.25, 2.5 * max(1e-6, best_root))
    spread_ok = (
        math.isfinite(best_spread)
        and math.isfinite(spread_target)
        and best_spread <= spread_target
        and best_spread <= spread_relative_cap
    )
    merge_ok = math.isfinite(best_merge) and best_merge <= merge_relative_cap
    recovered = bool(score_ok and spread_ok and merge_ok)

    state["recovered"] = recovered
    if recovered:
        state["rounds_without_recovery"] = 0
    else:
        state["rounds_without_recovery"] = int(state.get("rounds_without_recovery", 0)) + 1
    return state


def _render_report(
    *,
    out_path: Path,
    started_at: str,
    finished_at: str,
    history: Sequence[Mapping[str, Any]],
    state_by_experiment: Mapping[str, Mapping[str, Any]],
    output_root: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Learned-g Overnight Report")
    lines.append("")
    lines.append(f"- started_at_utc: `{started_at}`")
    lines.append(f"- finished_at_utc: `{finished_at}`")
    lines.append(f"- output_root: `{output_root}`")
    lines.append("")
    lines.append("## Best Per Experiment")
    lines.append("")
    for exp in (EXPERIMENT_LEARNED_OPS_G, EXPERIMENT_LEARNED_SEGMENTED_THETA_G):
        st = dict(state_by_experiment.get(exp, {}))
        lines.append(f"### `{exp}`")
        lines.append(f"- recovered: `{bool(st.get('recovered', False))}`")
        lines.append(f"- best_score: `{_safe_float(st.get('best_score'), default=float('nan')):.6g}`")
        lines.append(f"- baseline_score: `{_safe_float(st.get('baseline_score'), default=float('nan')):.6g}`")
        lines.append(f"- rounds_without_recovery: `{int(st.get('rounds_without_recovery', 0))}`")
        lines.append(f"- best_metrics: `{json.dumps(st.get('best_metrics', {}), sort_keys=True)}`")
        lines.append(f"- best_config: `{json.dumps(st.get('best_config', {}), sort_keys=True)}`")
        lines.append("")

    lines.append("## Round Snapshots")
    lines.append("")
    for row in history:
        ridx = int(row.get("round", -1))
        duration_s = _safe_float(row.get("duration_seconds"), default=float("nan"))
        lines.append(f"- round={ridx} duration_s={duration_s:.1f} n_specs={int(row.get('n_specs', 0))}")
        exp_summ = row.get("experiments", {})
        if isinstance(exp_summ, dict):
            for exp, info in exp_summ.items():
                if not isinstance(info, dict):
                    continue
                lines.append(
                    f"  - {exp}: score={_safe_float(info.get('round_best_score'), default=float('nan')):.6g} "
                    f"root={_safe_float(info.get('root_mae'), default=float('nan')):.6g} "
                    f"merge={_safe_float(info.get('merge_mae'), default=float('nan')):.6g} "
                    f"spread={_safe_float(info.get('schedule_spread_mean'), default=float('nan')):.6g} "
                    f"recovered={bool(info.get('recovered', False))}"
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive overnight learned-g benchmark optimizer")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=64)
    parser.add_argument("--max-hours", type=float, default=10.0)
    parser.add_argument("--max-rounds", type=int, default=24)
    parser.add_argument("--candidates-per-experiment", type=int, default=8)
    parser.add_argument("--seeds", type=str, default="0,1,2,3")
    parser.add_argument("--profile", choices=["overnight", "smoke"], default="overnight")
    parser.add_argument("--recovery-ratio", type=float, default=0.70)
    parser.add_argument("--spread-ratio", type=float, default=0.70)
    parser.add_argument("--stop-when-recovered", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    _set_cpu_thread_env_one()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root is not None else (REPO_ROOT / "outputs" / f"learned_g_overnight_{timestamp}")
    output_root.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(str(args.seeds))
    jobs = int(max(1, args.jobs))
    max_rounds = int(max(1, args.max_rounds))
    max_seconds = float(max(0.25, args.max_hours)) * 3600.0
    started_at = _utc_now()
    started_ts = time.time()

    metric_keys = (
        "root_mae",
        "root_median_abs_error",
        "root_p95_abs_error",
        "schedule_spread_mean",
        "schedule_spread_p95",
        "leaf_mae",
        "leaf_violation_rate",
        "merge_mae",
        "merge_violation_rate",
        "n_docs",
    )

    experiments = (EXPERIMENT_LEARNED_OPS_G, EXPERIMENT_LEARNED_SEGMENTED_THETA_G)
    state: Dict[str, Dict[str, Any]] = {
        exp: _init_experiment_state(experiment=exp, base_config=_base_config(exp, profile=str(args.profile)))
        for exp in experiments
    }

    status_path = output_root / "overnight_status.json"
    history_path = output_root / "round_history.jsonl"
    report_path = output_root / "overnight_report.md"

    history: List[Dict[str, Any]] = []
    round_index = 0

    while round_index < max_rounds and (time.time() - started_ts) < max_seconds:
        round_started = time.time()
        round_dir = output_root / f"round_{round_index:03d}"

        specs: List[RunSpec] = []
        spec_meta: Dict[str, Dict[str, Any]] = {}
        per_round_candidates: Dict[str, List[Dict[str, Any]]] = {}

        for exp in experiments:
            st = state[exp]
            cand_cfgs = _mutated_candidates(
                exp,
                pivot_config=dict(st["pivot_config"]),
                last_metrics=st.get("last_metrics"),
                rounds_without_recovery=int(st.get("rounds_without_recovery", 0)),
                n_candidates=int(args.candidates_per_experiment),
            )
            per_round_candidates[exp] = cand_cfgs
            for cidx, cfg in enumerate(cand_cfgs):
                validate_config_dict(exp, cfg)
                candidate_id = f"c{cidx:02d}"
                for seed in seeds:
                    run_cfg = dict(cfg)
                    run_cfg["seed"] = int(seed)
                    run_cfg["torch_threads"] = 1
                    run_cfg = _to_dataclass_config(exp, run_cfg)
                    validate_config_dict(exp, run_cfg)

                    run_out = round_dir / exp / candidate_id / f"seed_{seed}"
                    json_out = run_out / "summary.json"
                    csv_out = run_out / "summary.csv"
                    spec = RunSpec(experiment=exp, config=run_cfg, json_out=json_out, csv_out=csv_out)
                    specs.append(spec)
                    spec_meta[spec.key] = {
                        "round": int(round_index),
                        "experiment": exp,
                        "candidate_id": candidate_id,
                        "seed": int(seed),
                        "candidate_config": cfg,
                        "config": run_cfg,
                        "json_out": str(json_out),
                        "csv_out": str(csv_out),
                    }

        print(
            f"[{_utc_now()}] round={round_index} launching {len(specs)} runs "
            f"(jobs={jobs}, seeds={len(seeds)}, candidates/exp={int(args.candidates_per_experiment)})",
            flush=True,
        )
        raw_results = run_specs(specs, jobs=jobs, skip_existing=False)

        run_rows: List[Dict[str, Any]] = []
        for res in raw_results:
            spec_key = str(res.get("spec", "")) if "spec" in res else ""
            if not spec_key:
                json_out = str(res.get("json_out", ""))
                exp = ""
                for k in spec_meta.keys():
                    if k.endswith(json_out):
                        spec_key = k
                        break
            meta = dict(spec_meta.get(spec_key, {}))
            row: Dict[str, Any] = dict(meta)
            row["status"] = str(res.get("status", "unknown"))
            row["error"] = str(res.get("error", "")) if "error" in res else ""
            row["json_out"] = str(res.get("json_out", meta.get("json_out", "")))
            row["csv_out"] = str(res.get("csv_out", meta.get("csv_out", "")))
            row["score"] = float("nan")
            row["metrics"] = {}

            if row["status"] == "ok":
                payload: Dict[str, Any] = {}
                try:
                    payload = json.loads(Path(str(row["json_out"])).read_text(encoding="utf-8"))
                except Exception as e:
                    row["status"] = "error"
                    row["error"] = f"failed to load output json: {e}"
                metrics = dict(payload.get("metrics", {})) if isinstance(payload.get("metrics"), dict) else {}
                row["metrics"] = metrics
                row["score"] = _loss_from_metrics(metrics) if metrics else float("inf")
            run_rows.append(row)

        agg_rows = _aggregate_round(run_rows=run_rows, metric_keys=metric_keys)
        _append_jsonl(history_path, run_rows)

        round_info: Dict[str, Any] = {
            "round": int(round_index),
            "started_at_utc": _utc_now(),
            "n_specs": int(len(specs)),
            "n_status_ok": int(sum(1 for r in run_rows if r.get("status") == "ok")),
            "n_status_error": int(sum(1 for r in run_rows if r.get("status") == "error")),
            "results": agg_rows,
            "experiments": {},
        }

        for exp in experiments:
            exp_ranked = [r for r in agg_rows if str(r.get("experiment")) == exp]
            exp_ranked.sort(key=lambda r: float(r.get("score", float("inf"))))
            state[exp] = _update_experiment_state(
                state=state[exp],
                ranked_rows=exp_ranked,
                recovery_ratio=float(args.recovery_ratio),
                spread_ratio=float(args.spread_ratio),
            )
            round_best = dict(exp_ranked[0]) if exp_ranked else {}
            round_best_metrics = dict(round_best.get("mean_metrics", {}))
            round_info["experiments"][exp] = {
                "round_best_score": _safe_float(round_best.get("score"), default=float("nan")),
                "candidate_id": str(round_best.get("candidate_id", "")),
                "root_mae": _safe_float(round_best_metrics.get("root_mae"), default=float("nan")),
                "merge_mae": _safe_float(round_best_metrics.get("merge_mae"), default=float("nan")),
                "schedule_spread_mean": _safe_float(round_best_metrics.get("schedule_spread_mean"), default=float("nan")),
                "recovered": bool(state[exp].get("recovered", False)),
                "rounds_without_recovery": int(state[exp].get("rounds_without_recovery", 0)),
            }

        round_duration = float(time.time() - round_started)
        round_info["duration_seconds"] = round_duration
        round_info["finished_at_utc"] = _utc_now()
        history.append(round_info)

        elapsed = float(time.time() - started_ts)
        snapshot = {
            "status": "running",
            "started_at_utc": started_at,
            "updated_at_utc": _utc_now(),
            "elapsed_seconds": elapsed,
            "elapsed_hours": elapsed / 3600.0,
            "max_hours": float(args.max_hours),
            "max_rounds": int(max_rounds),
            "current_round": int(round_index),
            "history_length": int(len(history)),
            "state_by_experiment": state,
            "last_round": round_info,
            "output_root": str(output_root),
            "history_path": str(history_path),
            "report_path": str(report_path),
        }
        _write_json(status_path, snapshot)

        msg = (
            f"[{_utc_now()}] round={round_index} done in {round_duration:.1f}s "
            f"ok={round_info['n_status_ok']}/{round_info['n_specs']} | "
            f"ops_score={_safe_float(round_info['experiments'][EXPERIMENT_LEARNED_OPS_G]['round_best_score']):.5g} "
            f"ops_recovered={bool(state[EXPERIMENT_LEARNED_OPS_G]['recovered'])} | "
            f"theta_score={_safe_float(round_info['experiments'][EXPERIMENT_LEARNED_SEGMENTED_THETA_G]['round_best_score']):.5g} "
            f"theta_recovered={bool(state[EXPERIMENT_LEARNED_SEGMENTED_THETA_G]['recovered'])}"
        )
        print(msg, flush=True)

        all_recovered = all(bool(state[exp].get("recovered", False)) for exp in experiments)
        if bool(args.stop_when_recovered) and all_recovered:
            print(f"[{_utc_now()}] stopping early: all experiments met recovery criteria", flush=True)
            break

        round_index += 1

    finished_at = _utc_now()
    final_status = {
        "status": "completed",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "elapsed_seconds": float(time.time() - started_ts),
        "max_hours": float(args.max_hours),
        "max_rounds": int(max_rounds),
        "rounds_completed": int(len(history)),
        "output_root": str(output_root),
        "history_path": str(history_path),
        "report_path": str(report_path),
        "state_by_experiment": state,
        "history": history,
    }
    _write_json(status_path, final_status)
    _render_report(
        out_path=report_path,
        started_at=started_at,
        finished_at=finished_at,
        history=history,
        state_by_experiment=state,
        output_root=output_root,
    )
    print(f"[{_utc_now()}] completed rounds={len(history)} report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
