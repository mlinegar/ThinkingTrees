#!/usr/bin/env python3
"""Quick sanity checks for identifiable-zero suite output roots.

This script is designed to be run while a large CPU sweep is still generating
outputs. It reads raw per-run JSON summaries and reports a few high-signal
diagnostics:

- Do we have explicit ceilings (exact / oracle_tree) at ~0 error?
- Under full budget, do learned/budgeted methods approach those ceilings?
- Are there obvious pathologies (e.g., calibration underdetermined)?
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity-check identifiable-zero suite outputs (raw JSON scan).")
    p.add_argument("--output-root", type=Path, required=True, help="Suite output root (contains family subdirs).")
    p.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on total JSON files scanned (0 = no cap).",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the emitted JSON payload (default: none).",
    )
    return p.parse_args()


def _finite(x: object) -> Optional[float]:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _safe_median(vals: Iterable[object]) -> float:
    xs = [v for v in (_finite(x) for x in vals) if v is not None]
    return float(statistics.median(xs)) if xs else float("nan")


def _safe_mean(vals: Iterable[object]) -> float:
    xs = [v for v in (_finite(x) for x in vals) if v is not None]
    return float(statistics.fmean(xs)) if xs else float("nan")


def _safe_max(vals: Iterable[object]) -> float:
    xs = [v for v in (_finite(x) for x in vals) if v is not None]
    return float(max(xs)) if xs else float("nan")


def _iter_json_files(root: Path, *, max_files: int) -> List[Path]:
    files = sorted(root.rglob("*.json"))
    if max_files and max_files > 0:
        return files[: int(max_files)]
    return files


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class SegmentAnchor:
    phi_estimator: str
    max_train_docs: int
    lambda_multiplier: float
    full_audit_ridge_median: float
    full_audit_ridge_true_topics_median: float
    full_audit_exact_max: float
    full_audit_leaf_acc_test_median: float
    low_audit_train100_ridge_median: float


def _scan_segment_ops(seg_root: Path, *, max_files: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"present": False}
    if not seg_root.exists():
        return out

    files = _iter_json_files(seg_root, max_files=max_files)
    if not files:
        return out

    out["present"] = True

    # Group by topic_phi_estimator and extract a few anchor slices.
    by_phi: Dict[str, List[Tuple[Path, Dict[str, Any]]]] = {}
    for p in files:
        payload = _load_json(p)
        cfg = payload.get("config", {}) or {}
        phi = str(cfg.get("topic_phi_estimator", "")) or "unknown"
        by_phi.setdefault(phi, []).append((p, payload))

    anchors: List[Dict[str, Any]] = []
    for phi, rows in sorted(by_phi.items()):
        # Determine max train_docs available in this output root for this phi.
        train_docs_vals = []
        for _p, payload in rows:
            cfg = payload.get("config", {}) or {}
            td = _finite(cfg.get("train_docs"))
            if td is not None:
                train_docs_vals.append(int(td))
        if not train_docs_vals:
            continue
        max_train = int(max(train_docs_vals))

        # Use lambda=1.0 when present; otherwise fall back to the maximum.
        lam_vals = []
        for _p, payload in rows:
            cfg = payload.get("config", {}) or {}
            lam = _finite(cfg.get("lambda_multiplier"))
            if lam is not None:
                lam_vals.append(float(lam))
        if not lam_vals:
            continue
        lam_target = 1.0 if any(abs(float(x) - 1.0) <= 1e-12 for x in lam_vals) else float(max(lam_vals))

        def _select(
            *,
            train_docs: int,
            audit_fraction: float,
            lambda_multiplier: float,
        ) -> List[Dict[str, Any]]:
            picked: List[Dict[str, Any]] = []
            for _p, payload in rows:
                cfg = payload.get("config", {}) or {}
                if int(cfg.get("train_docs", -1)) != int(train_docs):
                    continue
                af = _finite(cfg.get("audit_fraction"))
                if af is None or abs(float(af) - float(audit_fraction)) > 1e-12:
                    continue
                lam = _finite(cfg.get("lambda_multiplier"))
                if lam is None or abs(float(lam) - float(lambda_multiplier)) > 1e-12:
                    continue
                picked.append(payload)
            return picked

        full = _select(train_docs=max_train, audit_fraction=1.0, lambda_multiplier=lam_target)
        low = _select(train_docs=100, audit_fraction=0.01, lambda_multiplier=lam_target)
        if not full or not low:
            continue

        def _metric(payload: Dict[str, Any], family: str, key: str) -> Optional[float]:
            metrics = payload.get("metrics", {}) or {}
            block = metrics.get(family, {}) or {}
            if not isinstance(block, dict):
                return None
            return _finite(block.get(key))

        anchors.append(
            SegmentAnchor(
                phi_estimator=str(phi),
                max_train_docs=int(max_train),
                lambda_multiplier=float(lam_target),
                full_audit_ridge_median=_safe_median(_metric(p, "ridge", "root_mae") for p in full),
                full_audit_ridge_true_topics_median=_safe_median(_metric(p, "ridge_true_topics", "root_mae") for p in full),
                full_audit_exact_max=_safe_max(_metric(p, "exact", "root_mae") for p in full),
                full_audit_leaf_acc_test_median=_safe_median(_metric(p, "ridge", "leaf_accuracy_test") for p in full),
                low_audit_train100_ridge_median=_safe_median(_metric(p, "ridge", "root_mae") for p in low),
            ).__dict__
        )

    out["anchors"] = anchors
    return out


def _scan_ctreepo(ctree_root: Path, *, max_files: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"present": False}
    if not ctree_root.exists():
        return out
    files = _iter_json_files(ctree_root, max_files=max_files)
    if not files:
        return out
    out["present"] = True

    oracle_vals: List[float] = []
    full_guidance_budgeted: List[float] = []
    bound_violations = 0
    underdetermined_calibration = 0
    worst_calibration: List[Tuple[float, str]] = []

    for p in files:
        payload = _load_json(p)
        cfg = payload.get("config", {}) or {}
        metrics = payload.get("metrics", {}) or {}
        decomp = payload.get("decomposition", {}) or {}
        cal_samples = int(payload.get("calibration_samples", 0) or 0)

        oracle = metrics.get("oracle_tree", {}) or {}
        bud = metrics.get("estimated_calibrated_budgeted", {}) or {}
        cal = metrics.get("estimated_calibrated", {}) or {}
        unc = metrics.get("estimated_uncalibrated", {}) or {}

        oracle_l1 = _finite(oracle.get("root_l1_mean"))
        if oracle_l1 is not None:
            oracle_vals.append(float(oracle_l1))

        lr = _finite(cfg.get("eval_leaf_query_rate"))
        ir = _finite(cfg.get("eval_internal_query_rate"))
        bud_l1 = _finite(bud.get("root_l1_mean"))
        if lr is not None and ir is not None and abs(float(lr) - 1.0) <= 1e-12 and abs(float(ir) - 1.0) <= 1e-12:
            if bud_l1 is not None:
                full_guidance_budgeted.append(float(bud_l1))

        total = _finite(decomp.get("total_root_l1_mean"))
        upper = _finite(decomp.get("upper_bound_mean"))
        if total is not None and upper is not None and float(total) > float(upper) + 1e-9:
            bound_violations += 1

        # Flag likely-underconstrained affine calibration: n_calib <= k+1 (noisy / underdetermined).
        k = int(cfg.get("n_topics", 0) or 0)
        cal_rate = _finite(cfg.get("calibration_leaf_query_rate"))
        if cal_rate is not None and cal_rate > 0.0 and k > 0 and cal_samples > 0 and cal_samples <= (k + 1):
            underdetermined_calibration += 1

        # Track worst calibrations where calibrated is much worse than uncalibrated.
        cal_l1 = _finite(cal.get("root_l1_mean"))
        unc_l1 = _finite(unc.get("root_l1_mean"))
        if cal_l1 is not None and unc_l1 is not None and float(cal_l1) - float(unc_l1) > 0.25:
            worst_calibration.append((float(cal_l1) - float(unc_l1), str(p)))

    worst_calibration.sort(reverse=True)
    worst_calibration = worst_calibration[:10]

    out.update(
        {
            "oracle_tree_root_l1_max": _safe_max(oracle_vals),
            "oracle_tree_root_l1_median": _safe_median(oracle_vals),
            "budgeted_full_guidance_root_l1_max": _safe_max(full_guidance_budgeted),
            "budgeted_full_guidance_root_l1_median": _safe_median(full_guidance_budgeted),
            "bound_violations_total_gt_upper": int(bound_violations),
            "underdetermined_calibration_runs": int(underdetermined_calibration),
            "worst_calibration_regressions": [{"delta": float(d), "path": path} for d, path in worst_calibration],
        }
    )
    return out


def _scan_markov(markov_root: Path, *, max_files: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"present": False}
    if not markov_root.exists():
        return out
    files = _iter_json_files(markov_root, max_files=max_files)
    if not files:
        return out
    out["present"] = True

    exact_full: List[float] = []
    learned_full: List[float] = []
    learned_low: List[float] = []
    for p in files:
        payload = _load_json(p)
        cfg = payload.get("config", {}) or {}
        af = _finite(cfg.get("audit_fraction"))
        td = _finite(cfg.get("train_docs"))
        metrics = payload.get("metrics", {}) or {}
        exact = metrics.get("exact", {}) or {}
        learned = metrics.get("learned", {}) or {}
        ex = _finite(exact.get("root_mae"))
        le = _finite(learned.get("root_mae"))
        if af is not None and abs(float(af) - 1.0) <= 1e-12:
            if ex is not None:
                exact_full.append(float(ex))
            if le is not None:
                learned_full.append(float(le))
        if af is not None and abs(float(af) - 0.01) <= 1e-12 and td is not None and float(td) <= 200:
            if le is not None:
                learned_low.append(float(le))

    out.update(
        {
            "full_audit_exact_root_mae_median": _safe_median(exact_full),
            "full_audit_learned_root_mae_median": _safe_median(learned_full),
            "low_audit_learned_root_mae_median": _safe_median(learned_low),
        }
    )
    return out


def main() -> int:
    args = _parse_args()
    output_root = args.output_root.resolve()
    max_files = int(args.max_files)

    payload: Dict[str, Any] = {
        "output_root": str(output_root),
        "segment_lda_ops_weight_recovery": _scan_segment_ops(output_root / "segment_lda_ops_weight_recovery", max_files=max_files),
        "segmented_lda_ctreepo": _scan_ctreepo(output_root / "segmented_lda_ctreepo", max_files=max_files),
        "markov_changepoint_ops_count": _scan_markov(output_root / "markov_changepoint_ops_count", max_files=max_files),
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)

    if args.output_json is not None:
        out_path = args.output_json.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

