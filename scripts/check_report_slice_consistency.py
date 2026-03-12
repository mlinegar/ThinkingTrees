#!/usr/bin/env python3
"""Validate that report scalar summaries align with declared fixed slices.

This checker is non-mutating: it reads raw outputs + report diagnostics and
compares fixed-slice scalar endpoints and coverage counts.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

FIXED_SEG_TRAIN_DOCS = 12000
FIXED_SEG_LAMBDA = 1.0
FIXED_CTREE_TRAIN_DOCS = 4096
FIXED_CTREE_MIN_CAL_SAMPLES = 50
FIXED_MARKOV_TRAIN_DOCS = 8000
FIXED_MARKOV_LEAF_QUERY_RATE = 1.0
FIXED_MARKOV_INCLUDE_ROOT_QUERY = True
CEILING_THRESHOLD = 1e-8


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check report slice consistency against raw outputs.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--report-diagnostics-json", type=Path, required=True)
    p.add_argument("--tolerance", type=float, default=1e-12)
    return p.parse_args()


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(x: object) -> Optional[float]:
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


def _close(a: float, b: float, tol: float) -> bool:
    if not (math.isfinite(float(a)) and math.isfinite(float(b))):
        return False
    return abs(float(a) - float(b)) <= float(tol)


def _lane_available(lane: object) -> bool:
    if not isinstance(lane, dict):
        return False
    if bool(lane.get("available", False)):
        return True
    if any(_as_float(v) is not None for v in (lane.get("q_train") or [])):
        return True
    if any(_as_float(v) is not None for v in (lane.get("q_infer") or [])):
        return True
    if any(_as_float(v) is not None for v in (lane.get("train_curve_raw") or [])):
        return True
    for row in (lane.get("matrix_raw") or []):
        if any(_as_float(v) is not None for v in row):
            return True
    return False


def _segment_q1(output_root: Path) -> Tuple[float, float]:
    files = sorted(glob.glob(str(output_root / "segment_lda_ops_weight_recovery" / "**" / "*seed_*.json"), recursive=True))
    true_vals: List[float] = []
    emb_vals: List[float] = []
    for fp in files:
        p = _load_json(Path(fp))
        c = p.get("config", {}) or {}
        m = p.get("metrics", {}) or {}
        if int(c.get("train_docs", -1)) != FIXED_SEG_TRAIN_DOCS:
            continue
        lam = _as_float(c.get("lambda_multiplier"))
        if lam is None or abs(float(lam) - FIXED_SEG_LAMBDA) > 1e-12:
            continue
        q = _as_float(c.get("audit_fraction"))
        if q is None or abs(float(q) - 1.0) > 1e-12:
            continue
        phi = str(c.get("topic_phi_estimator", ""))
        if phi == "true":
            v = _as_float(((m.get("ridge_true_topics") or {}).get("root_mae")))
            if v is not None:
                true_vals.append(float(v))
        if phi == "embedding_spectral":
            v = _as_float(((m.get("ridge") or {}).get("root_mae")))
            if v is not None:
                emb_vals.append(float(v))
    return (_median(true_vals), _median(emb_vals))


def _ctree_infer_full(output_root: Path) -> Tuple[float, int, int, int]:
    files = sorted(glob.glob(str(output_root / "segmented_lda_ctreepo" / "**" / "*.json"), recursive=True))
    rows: List[Tuple[float, float, float]] = []
    counts = 0
    n_qtr = set()
    n_qinf = set()
    for fp in files:
        p = _load_json(Path(fp))
        c = p.get("config", {}) or {}
        m = p.get("metrics", {}) or {}
        if int(c.get("n_books_train", -1)) != FIXED_CTREE_TRAIN_DOCS:
            continue
        if int(p.get("calibration_samples", 0) or 0) < FIXED_CTREE_MIN_CAL_SAMPLES:
            continue
        qtr = _as_float(c.get("calibration_leaf_query_rate"))
        ql = _as_float(c.get("eval_leaf_query_rate"))
        qi = _as_float(c.get("eval_internal_query_rate"))
        v = _as_float(((m.get("estimated_calibrated_budgeted") or {}).get("root_l1_mean")))
        if qtr is None or ql is None or qi is None or v is None:
            continue
        if abs(float(ql) - float(qi)) > 1e-12:
            continue
        n_qtr.add(float(qtr))
        n_qinf.add(float(ql))
        rows.append((float(qtr), float(ql), float(v)))
        counts += 1
    qtr_max = max((r[0] for r in rows), default=float("nan"))
    vals = [float(v) for (qtr, qinf, v) in rows if abs(float(qinf) - 1.0) <= 1e-12 and abs(float(qtr) - qtr_max) <= 1e-12]
    return (_median(vals), int(counts), len(n_qtr), len(n_qinf))


def _markov_endpoints(output_root: Path, family: str) -> Tuple[float, float, int, int, int]:
    files = sorted(glob.glob(str(output_root / "markov_changepoint_ops_count" / "**" / "*seed_*.json"), recursive=True))
    train_vals: List[float] = []
    infer_vals: List[float] = []
    qtr = set()
    qinf = set()
    guided_rows = 0
    for fp in files:
        p = _load_json(Path(fp))
        c = p.get("config", {}) or {}
        m = p.get("metrics", {}) or {}
        if str(c.get("model_family", "")) != family:
            continue
        if int(c.get("train_docs", -1)) != FIXED_MARKOV_TRAIN_DOCS:
            continue
        leaf = _as_float(c.get("leaf_query_rate"))
        if leaf is None or abs(float(leaf) - FIXED_MARKOV_LEAF_QUERY_RATE) > 1e-12:
            continue
        if bool(c.get("include_root_query", True)) is not bool(FIXED_MARKOV_INCLUDE_ROOT_QUERY):
            continue
        q = _as_float(c.get("audit_fraction"))
        if q is None:
            continue
        qtr.add(float(q))
        learned = _as_float(((m.get("learned") or {}).get("root_mae")))
        if learned is not None and abs(float(q) - 1.0) <= 1e-12:
            train_vals.append(float(learned))

        pts = (m.get("guided_eval_curve") or {}).get("points") or []
        for pt in pts:
            if not isinstance(pt, dict):
                continue
            qi = _as_float(pt.get("q"))
            y = _as_float(pt.get("root_mae"))
            if qi is None or y is None:
                continue
            qinf.add(float(qi))
            guided_rows += 1
            if abs(float(q) - 1.0) <= 1e-12 and abs(float(qi) - 1.0) <= 1e-12:
                infer_vals.append(float(y))

    return (_median(train_vals), _median(infer_vals), len(qtr), len(qinf), int(guided_rows))


def main() -> int:
    args = _parse_args()
    report = _load_json(args.report_diagnostics_json)
    diag = (report.get("diagnostics") or {})
    fig_a = (diag.get("figure_a") or {})
    endpoint_table = fig_a.get("endpoint_table") or []

    expected: Dict[str, float] = {}
    expected_norm_valid: Dict[str, bool] = {}
    expected_norm_display: Dict[str, str] = {}
    expected_status: Dict[str, str] = {}
    for ep in endpoint_table:
        eid = str(ep.get("endpoint_id", ""))
        expected[eid] = float(ep.get("raw", float("nan")))
        expected_norm_valid[eid] = bool(ep.get("norm_valid", False))
        expected_norm_display[eid] = str(ep.get("norm_display", ""))
        expected_status[eid] = str(ep.get("status", ""))

    seg_true, seg_embed = _segment_q1(args.output_root)
    ct_infer, ct_rows, ct_nqtr, ct_nqinf = _ctree_infer_full(args.output_root)
    mk_add_train, mk_add_infer, mk_add_nqtr, mk_add_nqinf, mk_add_grows = _markov_endpoints(args.output_root, "additive")
    mk_neu_train, mk_neu_infer, mk_neu_nqtr, mk_neu_nqinf, mk_neu_grows = _markov_endpoints(args.output_root, "neural")

    checks: List[Dict[str, object]] = []

    def add(name: str, got: float, endpoint_id: str) -> None:
        if endpoint_id not in expected:
            return
        exp = float(expected.get(endpoint_id, float("nan")))
        checks.append(
            {
                "name": name,
                "endpoint_id": endpoint_id,
                "expected": exp,
                "got": float(got),
                "passed": bool(_close(float(got), exp, float(args.tolerance))),
            }
        )

    add("segment_phi_true_train_q1", seg_true, "segment_phi_true_learn_full")
    add("segment_phi_embed_train_q1", seg_embed, "segment_phi_embedding_learn_full")
    add("ctree_infer_full_qtrain_max_qinfer1", ct_infer, "ctree_decision_full")
    add("markov_add_train_q1", mk_add_train, "markov_additive_learn_full")
    add("markov_add_infer_q1_q1", mk_add_infer, "markov_additive_decision_full")
    add("markov_neural_train_q1", mk_neu_train, "markov_neural_learn_full")
    add("markov_neural_infer_q1_q1", mk_neu_infer, "markov_neural_decision_full")

    normalization_validity = (diag.get("normalization_validity") or {}).get("lanes") or {}
    norm_flag_checks: List[Dict[str, object]] = []
    for lane_name, lane in normalization_validity.items():
        if not isinstance(lane, dict):
            continue
        has_fields = ("norm_valid" in lane) and ("norm_den" in lane)
        norm_flag_checks.append({"lane": lane_name, "has_required_fields": bool(has_fields)})
    norm_flags_pass = bool(norm_flag_checks) and all(bool(x.get("has_required_fields", False)) for x in norm_flag_checks)

    endpoint_norm_display_checks: List[Dict[str, object]] = []
    for eid, is_valid in expected_norm_valid.items():
        disp = expected_norm_display.get(eid, "")
        passed = (bool(is_valid) and str(disp).strip().upper() != "N/A") or ((not bool(is_valid)) and str(disp).strip().upper() == "N/A")
        endpoint_norm_display_checks.append(
            {
                "endpoint_id": eid,
                "norm_valid": bool(is_valid),
                "norm_display": disp,
                "passed": bool(passed),
            }
        )
    endpoint_norm_display_pass = bool(endpoint_norm_display_checks) and all(
        bool(x.get("passed", False)) for x in endpoint_norm_display_checks
    )

    status_checks: List[Dict[str, object]] = []
    for eid, raw in expected.items():
        expected_stat = "PASS" if (math.isfinite(float(raw)) and float(raw) <= CEILING_THRESHOLD) else "FAIL"
        got_stat = expected_status.get(eid, "")
        status_checks.append(
            {
                "endpoint_id": eid,
                "raw": float(raw),
                "expected_status": expected_stat,
                "display_status": got_stat,
                "passed": bool(expected_stat == got_stat),
            }
        )
    stakeholder_table_status_pass = bool(status_checks) and all(bool(x.get("passed", False)) for x in status_checks)

    coverage_checks = {
        "ctree_rows": ct_rows,
        "ctree_q_train_count": ct_nqtr,
        "ctree_q_infer_count": ct_nqinf,
        "markov_add_q_train_count": mk_add_nqtr,
        "markov_add_q_infer_count": mk_add_nqinf,
        "markov_add_guided_rows": mk_add_grows,
        "markov_neural_q_train_count": mk_neu_nqtr,
        "markov_neural_q_infer_count": mk_neu_nqinf,
        "markov_neural_guided_rows": mk_neu_grows,
    }

    mixed_tradeoff = (diag.get("mixed_tradeoff") or {})
    report_markov_fams = (((mixed_tradeoff.get("markov") or {}).get("families")) or {})
    coverage_expected: Dict[str, int] = {
        "ctree_q_train_count": 4,
        "ctree_q_infer_count": 8,
    }
    if _lane_available(report_markov_fams.get("additive") or {}):
        coverage_expected["markov_add_q_train_count"] = 8
        coverage_expected["markov_add_q_infer_count"] = 6
    if _lane_available(report_markov_fams.get("neural") or {}):
        coverage_expected["markov_neural_q_train_count"] = 8
        coverage_expected["markov_neural_q_infer_count"] = 6
    coverage_pass = all(int(coverage_checks.get(key, -1)) == int(value) for key, value in coverage_expected.items())

    all_pass = bool(
        all(bool(c.get("passed", False)) for c in checks)
        and coverage_pass
        and norm_flags_pass
        and endpoint_norm_display_pass
        and stakeholder_table_status_pass
    )
    payload = {
        "output_root": str(args.output_root.resolve()),
        "report_diagnostics_json": str(args.report_diagnostics_json.resolve()),
        "tolerance": float(args.tolerance),
        "passed": all_pass,
        "checks": checks,
        "coverage": coverage_checks,
        "coverage_expected": coverage_expected,
        "coverage_passed": coverage_pass,
        "normalization_flag_checks": norm_flag_checks,
        "normalization_flags_passed": norm_flags_pass,
        "endpoint_norm_display_checks": endpoint_norm_display_checks,
        "endpoint_norm_display_passed": endpoint_norm_display_pass,
        "stakeholder_table_status_checks": status_checks,
        "stakeholder_table_status_passed": stakeholder_table_status_pass,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
