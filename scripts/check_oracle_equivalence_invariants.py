#!/usr/bin/env python3
"""Light-formal invariant checks for identifiable oracle-equivalence suites.

Gates:
1) Exact/oracle ceilings are near zero.
2) q_infer=1 guided points hit the ceiling threshold.
3) Monotonicity diagnostics for q_train and q_infer (warnings by default).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check oracle-equivalence invariants on a suite output root.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--ceiling-threshold", type=float, default=1e-8)
    p.add_argument("--hard-guided-threshold", type=float, default=1e-12)
    p.add_argument("--monotonicity-tol", type=float, default=1e-10)
    p.add_argument("--warn-only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(x: object) -> Optional[float]:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _median(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if not xs:
        return float("nan")
    return float(statistics.median(xs))


def _max(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if not xs:
        return float("nan")
    return float(max(xs))


def _monotonic_increase_violations(
    xs: List[float],
    ys: List[float],
    *,
    tol: float,
) -> List[Tuple[float, float, float, float]]:
    if not xs or len(xs) != len(ys):
        return []
    order = sorted(range(len(xs)), key=lambda i: float(xs[i]))
    vv: List[Tuple[float, float, float, float]] = []
    for i in range(len(order) - 1):
        a = order[i]
        b = order[i + 1]
        x0 = float(xs[a])
        x1 = float(xs[b])
        y0 = float(ys[a])
        y1 = float(ys[b])
        if not (math.isfinite(y0) and math.isfinite(y1)):
            continue
        if y1 > y0 + float(tol):
            vv.append((x0, y0, x1, y1))
    return vv


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    details: Dict[str, object]


def _scan_segment(root: Path, *, ceiling: float, monotonicity_tol: float) -> Tuple[GateResult, Dict[str, object]]:
    files = sorted(glob.glob(str(root / "segment_lda_ops_weight_recovery" / "**" / "*seed_*.json"), recursive=True))
    if not files:
        return (
            GateResult("segment_exact_ceiling", True, {"present": False, "reason": "no files"}),
            {"warnings": []},
        )

    exact_vals: List[float] = []
    rows: List[dict] = []
    for fp in files:
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        m = payload.get("metrics", {}) or {}
        ex = _finite(((m.get("exact") or {}).get("root_mae")))
        if ex is not None:
            exact_vals.append(float(ex))
        rt = _finite(((m.get("ridge_true_topics") or {}).get("root_mae")))
        if rt is not None:
            rows.append(
                {
                    "train_docs": int(cfg.get("train_docs", -1)),
                    "audit_fraction": float(cfg.get("audit_fraction", float("nan"))),
                    "lambda_multiplier": float(cfg.get("lambda_multiplier", float("nan"))),
                    "topic_phi_estimator": str(cfg.get("topic_phi_estimator", "")),
                    "ridge_true_topics_root_mae": float(rt),
                }
            )

    max_exact = _max(exact_vals)
    g1 = GateResult(
        "segment_exact_ceiling",
        bool(math.isfinite(max_exact) and max_exact <= float(ceiling)),
        {
            "present": True,
            "n_files": int(len(files)),
            "exact_root_mae_max": float(max_exact),
            "ceiling_threshold": float(ceiling),
        },
    )

    warnings: List[Dict[str, object]] = []
    # q_train monotonic diagnostic in primary panel slice (phi=true, lambda=1, max train_docs).
    train_candidates = sorted({int(r["train_docs"]) for r in rows if int(r["train_docs"]) > 0})
    max_train = int(max(train_candidates)) if train_candidates else -1
    slice_rows = [
        r
        for r in rows
        if int(r["train_docs"]) == max_train
        and str(r["topic_phi_estimator"]) == "true"
        and abs(float(r["lambda_multiplier"]) - 1.0) <= 1e-12
    ]
    if slice_rows:
        afs = sorted({float(r["audit_fraction"]) for r in slice_rows})
        ys = [
            _median(
                float(r["ridge_true_topics_root_mae"])
                for r in slice_rows
                if abs(float(r["audit_fraction"]) - float(af)) <= 1e-12
            )
            for af in afs
        ]
        viol = _monotonic_increase_violations(afs, ys, tol=float(monotonicity_tol))
        if viol:
            warnings.append(
                {
                    "family": "segment",
                    "type": "q_train_monotonicity",
                    "n_violations": int(len(viol)),
                    "worst_examples": [
                        {"from_q": v[0], "from_err": v[1], "to_q": v[2], "to_err": v[3]} for v in viol[:8]
                    ],
                }
            )
    return g1, {"warnings": warnings}


def _scan_ctree(root: Path, *, ceiling: float, monotonicity_tol: float) -> Tuple[List[GateResult], Dict[str, object]]:
    files = sorted(glob.glob(str(root / "segmented_lda_ctreepo" / "**" / "*.json"), recursive=True))
    if not files:
        return (
            [
                GateResult("ctree_oracle_ceiling", True, {"present": False, "reason": "no files"}),
                GateResult("ctree_qinfer_q1", True, {"present": False, "reason": "no files"}),
            ],
            {"warnings": []},
        )

    oracle_vals: List[float] = []
    q1_vals: List[float] = []
    rows: List[dict] = []
    for fp in files:
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        m = payload.get("metrics", {}) or {}
        oracle = _finite(((m.get("oracle_tree") or {}).get("root_l1_mean")))
        bud = _finite(((m.get("estimated_calibrated_budgeted") or {}).get("root_l1_mean")))
        ql = _finite(cfg.get("eval_leaf_query_rate"))
        qi = _finite(cfg.get("eval_internal_query_rate"))
        if oracle is not None:
            oracle_vals.append(float(oracle))
        if bud is None or ql is None or qi is None:
            continue
        rows.append(
            {
                "train_docs": int(cfg.get("n_books_train", -1)),
                "cal_rate": float(cfg.get("calibration_leaf_query_rate", float("nan"))),
                "q_leaf": float(ql),
                "q_internal": float(qi),
                "root_l1": float(bud),
            }
        )
        if abs(float(ql) - 1.0) <= 1e-12 and abs(float(qi) - 1.0) <= 1e-12:
            q1_vals.append(float(bud))

    max_oracle = _max(oracle_vals)
    max_q1 = _max(q1_vals)
    gates = [
        GateResult(
            "ctree_oracle_ceiling",
            bool(math.isfinite(max_oracle) and max_oracle <= float(ceiling)),
            {
                "present": True,
                "n_files": int(len(files)),
                "oracle_tree_root_l1_max": float(max_oracle),
                "ceiling_threshold": float(ceiling),
            },
        ),
        GateResult(
            "ctree_qinfer_q1",
            bool(math.isfinite(max_q1) and max_q1 <= float(ceiling)),
            {
                "present": bool(q1_vals),
                "q1_budgeted_root_l1_max": float(max_q1),
                "ceiling_threshold": float(ceiling),
            },
        ),
    ]

    warnings: List[Dict[str, object]] = []
    # q_infer monotonic diagnostic on coupled slice.
    train_candidates = sorted({int(r["train_docs"]) for r in rows if int(r["train_docs"]) > 0})
    max_train = int(max(train_candidates)) if train_candidates else -1
    cal_candidates = sorted(
        {
            float(r["cal_rate"])
            for r in rows
            if int(r["train_docs"]) == max_train and math.isfinite(float(r["cal_rate"]))
        }
    )
    cal_rate = 0.05 if any(abs(c - 0.05) <= 1e-12 for c in cal_candidates) else (cal_candidates[0] if cal_candidates else float("nan"))
    coupled = [
        r
        for r in rows
        if int(r["train_docs"]) == max_train
        and math.isfinite(float(r["cal_rate"]))
        and abs(float(r["cal_rate"]) - float(cal_rate)) <= 1e-12
        and abs(float(r["q_leaf"]) - float(r["q_internal"])) <= 1e-12
    ]
    if coupled:
        qvals = sorted({float(r["q_leaf"]) for r in coupled})
        ys = [
            _median(
                float(r["root_l1"])
                for r in coupled
                if abs(float(r["q_leaf"]) - float(q)) <= 1e-12
            )
            for q in qvals
        ]
        viol = _monotonic_increase_violations(qvals, ys, tol=float(monotonicity_tol))
        if viol:
            warnings.append(
                {
                    "family": "ctree",
                    "type": "q_infer_monotonicity",
                    "n_violations": int(len(viol)),
                    "worst_examples": [
                        {"from_q": v[0], "from_err": v[1], "to_q": v[2], "to_err": v[3]} for v in viol[:8]
                    ],
                }
            )
    return gates, {"warnings": warnings}


def _scan_markov(
    root: Path,
    *,
    ceiling: float,
    hard_guided_threshold: float,
    monotonicity_tol: float,
) -> Tuple[List[GateResult], Dict[str, object]]:
    files = sorted(glob.glob(str(root / "markov_changepoint_ops_count" / "**" / "*seed_*.json"), recursive=True))
    if not files:
        return (
            [
                GateResult("markov_exact_ceiling", True, {"present": False, "reason": "no files"}),
                GateResult("markov_qinfer_q1", True, {"present": False, "reason": "no files"}),
            ],
            {"warnings": []},
        )

    exact_vals: List[float] = []
    guided_q1_vals: List[float] = []
    guided_q1_hard_vals: List[float] = []
    train_rows: List[dict] = []
    infer_rows: List[dict] = []

    for fp in files:
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        m = payload.get("metrics", {}) or {}
        ex = _finite(((m.get("exact") or {}).get("root_mae")))
        if ex is not None:
            exact_vals.append(float(ex))
        learned = _finite(((m.get("learned") or {}).get("root_mae")))
        af = _finite(cfg.get("audit_fraction"))
        if learned is not None and af is not None:
            train_rows.append(
                {
                    "model_family": str(cfg.get("model_family", "")),
                    "include_root_query": bool(cfg.get("include_root_query", True)),
                    "leaf_query_rate": float(cfg.get("leaf_query_rate", float("nan"))),
                    "audit_fraction": float(af),
                    "root_mae": float(learned),
                }
            )

        g = m.get("guided_eval_curve", {}) or {}
        if not isinstance(g, dict):
            continue
        points = g.get("points") or []
        if not isinstance(points, list):
            continue
        include_root = bool(g.get("include_root", True))
        for p in points:
            if not isinstance(p, dict):
                continue
            q = _finite(p.get("q"))
            rm = _finite(p.get("root_mae"))
            if q is None or rm is None:
                continue
            infer_rows.append(
                {
                    "model_family": str(cfg.get("model_family", "")),
                    "q": float(q),
                    "root_mae": float(rm),
                    "include_root": bool(include_root),
                }
            )
            if abs(float(q) - 1.0) <= 1e-12:
                guided_q1_vals.append(float(rm))
                if include_root:
                    guided_q1_hard_vals.append(float(rm))

    max_exact = _max(exact_vals)
    max_q1 = _max(guided_q1_vals)
    max_q1_hard = _max(guided_q1_hard_vals)
    gates = [
        GateResult(
            "markov_exact_ceiling",
            bool(math.isfinite(max_exact) and max_exact <= float(ceiling)),
            {
                "present": True,
                "n_files": int(len(files)),
                "exact_root_mae_max": float(max_exact),
                "ceiling_threshold": float(ceiling),
            },
        ),
        GateResult(
            "markov_qinfer_q1",
            bool(math.isfinite(max_q1) and max_q1 <= float(ceiling)),
            {
                "present": bool(guided_q1_vals),
                "guided_q1_root_mae_max": float(max_q1),
                "ceiling_threshold": float(ceiling),
            },
        ),
        GateResult(
            "markov_qinfer_q1_hard",
            bool(math.isfinite(max_q1_hard) and max_q1_hard <= float(hard_guided_threshold)),
            {
                "present": bool(guided_q1_hard_vals),
                "guided_q1_root_mae_max_include_root": float(max_q1_hard),
                "hard_guided_threshold": float(hard_guided_threshold),
            },
        ),
    ]

    warnings: List[Dict[str, object]] = []
    # q_train monotonicity on additive positive-control slice.
    add_slice = [
        r
        for r in train_rows
        if str(r["model_family"]) == "additive"
        and bool(r["include_root_query"]) is True
        and math.isfinite(float(r["leaf_query_rate"]))
        and abs(float(r["leaf_query_rate"]) - 1.0) <= 1e-12
    ]
    if add_slice:
        qvals = sorted({float(r["audit_fraction"]) for r in add_slice})
        ys = [
            _median(
                float(r["root_mae"])
                for r in add_slice
                if abs(float(r["audit_fraction"]) - float(q)) <= 1e-12
            )
            for q in qvals
        ]
        viol = _monotonic_increase_violations(qvals, ys, tol=float(monotonicity_tol))
        if viol:
            warnings.append(
                {
                    "family": "markov",
                    "type": "q_train_monotonicity",
                    "n_violations": int(len(viol)),
                    "worst_examples": [
                        {"from_q": v[0], "from_err": v[1], "to_q": v[2], "to_err": v[3]} for v in viol[:8]
                    ],
                }
            )

    # q_infer monotonicity per model family.
    for fam in sorted({str(r["model_family"]) for r in infer_rows if str(r["model_family"])}):
        fam_rows = [
            r for r in infer_rows if str(r["model_family"]) == fam and bool(r["include_root"]) is True
        ]
        if not fam_rows:
            continue
        qvals = sorted({float(r["q"]) for r in fam_rows})
        ys = [
            _median(
                float(r["root_mae"])
                for r in fam_rows
                if abs(float(r["q"]) - float(q)) <= 1e-12
            )
            for q in qvals
        ]
        viol = _monotonic_increase_violations(qvals, ys, tol=float(monotonicity_tol))
        if viol:
            warnings.append(
                {
                    "family": "markov",
                    "model_family": fam,
                    "type": "q_infer_monotonicity",
                    "n_violations": int(len(viol)),
                    "worst_examples": [
                        {"from_q": v[0], "from_err": v[1], "to_q": v[2], "to_err": v[3]} for v in viol[:8]
                    ],
                }
            )

    return gates, {"warnings": warnings}


def main() -> int:
    args = _parse_args()
    root = args.output_root.resolve()
    ceiling = float(args.ceiling_threshold)
    hard_guided = float(args.hard_guided_threshold)
    mono_tol = float(args.monotonicity_tol)

    segment_gate, segment_aux = _scan_segment(root, ceiling=ceiling, monotonicity_tol=mono_tol)
    ctree_gates, ctree_aux = _scan_ctree(root, ceiling=ceiling, monotonicity_tol=mono_tol)
    markov_gates, markov_aux = _scan_markov(
        root,
        ceiling=ceiling,
        hard_guided_threshold=hard_guided,
        monotonicity_tol=mono_tol,
    )

    gates = [segment_gate, *ctree_gates, *markov_gates]
    failed = [g for g in gates if not bool(g.passed)]
    warnings = [*segment_aux.get("warnings", []), *ctree_aux.get("warnings", []), *markov_aux.get("warnings", [])]

    payload = {
        "output_root": str(root),
        "ceiling_threshold": float(ceiling),
        "hard_guided_threshold": float(hard_guided),
        "monotonicity_tol": float(mono_tol),
        "gates": [
            {"name": g.name, "passed": bool(g.passed), "details": dict(g.details)}
            for g in gates
        ],
        "n_failed_gates": int(len(failed)),
        "failed_gate_names": [g.name for g in failed],
        "warnings": warnings,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)

    if args.output_json is not None:
        out_path = args.output_json.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")

    if failed and not bool(args.warn_only):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
