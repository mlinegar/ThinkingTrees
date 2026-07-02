#!/usr/bin/env python3
"""Compare per-dimension Pearson of a CONTROL run against one or more TEST runs.

Purpose: stop re-running the gold-children control every A/B. Point this at a
pre-generated control's ``iter_*_post_eval.jsonl`` (e.g. the FULL218 leafgrid, or
a saved control arm) plus any new test runs, and get the per-dim Pearson table
with deltas vs control. Pearson is reported per CMP dimension (rile + domain_1..7)
because pooling across dimensions inflates the number (between-dim mean
separation), so pooled Pearson is NOT a faithful composition signal.

Each input is either:
  - a path to an ``iter_<N>_post_eval.jsonl`` file, or
  - a run/ladder output dir, in which case
    ``<dir>/dspy/leafq<LEAF>/prediction_records/iter_<ITER>_post_eval.jsonl``
    is resolved from --leaf / --iter.

Records carry one row per (dimension, doc) with ``prediction`` and
``teacher_score``. We compute Pearson(prediction, teacher) per dimension. When
control and test share (dimension, doc_id) keys we also report a PAIRED delta on
the common docs, so the comparison is not confounded by different eval samples.

Usage:
  ./venv/bin/python scripts/compare_qsentence_per_dim_pearson.py \
      --control outputs/manifesto_qsentence_diffusiongemma_FULL218_leafgrid \
      --test outputs/sched_sampling_ab_leaf8_XXXX/sched_rate1 \
      --leaf 8 --iter 2 --labels control,sched
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from src.experiments.metrics import pearson
from src.experiments.script_parse import safe_float as pearson_safe_float
from src.experiments.script_io import read_jsonl as _read_jsonl, write_json as _write_json

DIMS = ["rile", "domain_1", "domain_2", "domain_3", "domain_4", "domain_5", "domain_6", "domain_7"]


def _resolve_eval_path(spec: str, *, leaf: int, it: int) -> Path:
    p = Path(spec)
    if p.is_file():
        return p
    candidate = p / "dspy" / f"leafq{int(leaf):03d}" / "prediction_records" / f"iter_{int(it):02d}_post_eval.jsonl"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"could not resolve eval jsonl for {spec!r} (tried {candidate}); pass a "
        f"direct iter_*_post_eval.jsonl path or a run dir with --leaf/--iter"
    )


def _load(path: Path) -> List[Mapping[str, object]]:
    return _read_jsonl(path)


def _row_key(row: Mapping[str, object]) -> Tuple[str, str]:
    return (str(row.get("dimension")), str(row.get("doc_id") or row.get("index")))


def _per_dim_pred_truth(rows: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """dim -> {row_key_doc: (prediction, teacher)} for paired alignment."""
    out: Dict[str, Dict[str, Tuple[float, float]]] = {d: {} for d in DIMS}
    for row in rows:
        dim = str(row.get("dimension"))
        if dim not in out:
            continue
        p = pearson_safe_float(row.get("prediction"))
        t = pearson_safe_float(row.get("teacher_score"))
        if p is None or t is None:
            continue
        out[dim][str(row.get("doc_id") or row.get("index"))] = (p, t)
    return out


def _pearson_table(per_dim: Mapping[str, Mapping[str, Tuple[float, float]]]) -> Dict[str, Optional[float]]:
    table: Dict[str, Optional[float]] = {}
    for dim in DIMS:
        pairs = list(per_dim.get(dim, {}).values())
        if len(pairs) < 2:
            table[dim] = None
            continue
        table[dim] = pearson([p for p, _ in pairs], [t for _, t in pairs])
    return table


def _fmt(v: Optional[float]) -> str:
    return f"{v:+.3f}" if v is not None else "   n/a"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True, help="control run dir or iter_*_post_eval.jsonl")
    ap.add_argument("--test", action="append", required=True, help="test run dir/jsonl (repeatable)")
    ap.add_argument("--leaf", type=int, default=8)
    ap.add_argument("--iter", type=int, default=2)
    ap.add_argument("--labels", default=None, help="comma list: control,test1[,test2...]")
    ap.add_argument("--json-out", default=None, help="optional path to write the comparison JSON")
    args = ap.parse_args(argv)

    control_path = _resolve_eval_path(args.control, leaf=args.leaf, it=args.iter)
    test_paths = [_resolve_eval_path(t, leaf=args.leaf, it=args.iter) for t in args.test]

    labels = (args.labels.split(",") if args.labels else None)
    ctl_label = labels[0] if labels else "control"
    test_labels = (
        labels[1:] if labels and len(labels) > 1
        else [f"test{i+1}" for i in range(len(test_paths))]
    )
    while len(test_labels) < len(test_paths):
        test_labels.append(f"test{len(test_labels)+1}")

    ctl_pd = _per_dim_pred_truth(_load(control_path))
    ctl_tbl = _pearson_table(ctl_pd)

    print(f"Per-dimension Pearson (prediction vs teacher), leaf={args.leaf} iter={args.iter}")
    print(f"  control = {ctl_label}: {control_path}")
    for tl, tp in zip(test_labels, test_paths):
        print(f"  test    = {tl}: {tp}")
    print()

    header = f'{"dim":10s} {ctl_label:>10s}'
    for tl in test_labels:
        header += f' {tl:>10s} {("Δ"+tl)[:9]:>10s} {"Δpaired":>9s}'
    print(header)

    payload: Dict[str, object] = {
        "leaf": args.leaf, "iter": args.iter,
        "control": {"label": ctl_label, "path": str(control_path), "per_dim": {}},
        "tests": [],
    }
    test_pds = [_per_dim_pred_truth(_load(tp)) for tp in test_paths]
    test_tbls = [_pearson_table(pd) for pd in test_pds]

    for dim in DIMS:
        line = f"{dim:10s} {_fmt(ctl_tbl[dim]):>10s}"
        payload["control"]["per_dim"][dim] = ctl_tbl[dim]  # type: ignore[index]
        for tl, t_tbl, t_pd in zip(test_labels, test_tbls, test_pds):
            tv = t_tbl[dim]
            delta = (tv - ctl_tbl[dim]) if (tv is not None and ctl_tbl[dim] is not None) else None
            # paired delta on common docs only
            common = set(ctl_pd.get(dim, {})) & set(t_pd.get(dim, {}))
            paired = None
            if len(common) >= 2:
                cp = pearson([ctl_pd[dim][k][0] for k in common], [ctl_pd[dim][k][1] for k in common])
                tpr = pearson([t_pd[dim][k][0] for k in common], [t_pd[dim][k][1] for k in common])
                if cp is not None and tpr is not None:
                    paired = tpr - cp
            line += f" {_fmt(tv):>10s} {_fmt(delta):>10s} {_fmt(paired):>9s}"
        print(line)

    # verdict: mean delta across dims with both defined
    for tl, t_tbl in zip(test_labels, test_tbls):
        deltas = [
            t_tbl[d] - ctl_tbl[d]
            for d in DIMS
            if t_tbl[d] is not None and ctl_tbl[d] is not None
        ]
        mean_delta = sum(deltas) / len(deltas) if deltas else None
        improved = sum(1 for x in deltas if x > 0)
        print(
            f"\n[{tl}] mean Δ Pearson vs {ctl_label} = "
            f"{(f'{mean_delta:+.3f}' if mean_delta is not None else 'n/a')} "
            f"({improved}/{len(deltas)} dims improved)"
        )
        payload["tests"].append({  # type: ignore[union-attr]
            "label": tl, "per_dim": {d: t_tbl[d] for d in DIMS},
            "mean_delta_vs_control": mean_delta, "dims_improved": improved, "dims_compared": len(deltas),
        })

    if args.json_out:
        _write_json(args.json_out, payload)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
