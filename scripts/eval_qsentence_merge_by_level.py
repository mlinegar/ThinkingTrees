#!/usr/bin/env python3
"""Per-level, lopsidedness-weighted merge evaluation for q-sentence trees.

The manifesto dim/RILE targets are RATIOS with a per-node ``total_non_header``
denominator (``src/tasks/manifesto/span_targets.py``). So the correct merge of
two child states is the MASS-WEIGHTED mean of their ratios, NOT the equal
average. Most merge nodes (level-1 leaf pairs) have near-balanced sibling
masses, so equal-averaging looks "fine" pooled -- but it is provably wrong, and
increasingly so, at the deep LOPSIDED merges where one subtree dwarfs its
sibling.

This evaluator quantifies that gap with NO LM calls. For every internal node it
compares two reconstructions of the gold parent ratio from the gold CHILD
ratios:

* ``equal_avg``   = 0.5*(left + right)                 (what a mass-blind g does)
* ``mass_wtd``    = w_l*left + w_r*right, w = mass/sum  (the exact merge; the
                    ceiling a real learned merge should reach)

and reports MAE vs the gold parent per tree LEVEL, both unweighted and
LOPSIDEDNESS-WEIGHTED (weight = 1 + strength * |m_l-m_r|/(m_l+m_r)). The
weighted view is the headline: a merge that nails balanced leaves but mishandles
lopsided deep merges scores worse, surfacing the real non-additive signal.

Optionally, pass ``--g-states-jsonl`` mapping (doc_id, node_id) -> compact state
to score a LEARNED g against the same per-level, weighted yardstick: the bar to
beat is ``equal_avg``; the ceiling is ``mass_wtd``.

Usage:
    python scripts/eval_qsentence_merge_by_level.py \
        --labeled-trees outputs/manifesto_qsentence_dspy_labeled_grid/leafq008/labeled_trees.jsonl \
        --lopsidedness-strength 4.0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

DIMS = ["rile"] + [f"domain_{i}" for i in range(1, 8)]


def _parse_first_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _node_mass(node: Mapping[str, Any]) -> Optional[float]:
    """Subtree mass (total_non_header) from a labeled-tree node."""
    meta = node.get("metadata") or {}
    for key in ("total_non_header", "n_non_header", "node_mass"):
        v = _parse_first_float(meta.get(key))
        if v is not None and v > 0:
            return v
    for key in ("teacher_summary", "target_summary", "summary"):
        raw = meta.get(key)
        if not raw:
            continue
        try:
            payload = json.loads(str(raw))
        except (TypeError, ValueError):
            continue
        for container in (payload, payload.get("cmp_state") if isinstance(payload, Mapping) else None):
            if isinstance(container, Mapping):
                v = _parse_first_float(container.get("total_non_header"))
                if v is not None and v > 0:
                    return v
    return None


def _node_scores(node: Mapping[str, Any]) -> Dict[str, float]:
    raw = node.get("dimension_scores") or {}
    out: Dict[str, float] = {}
    if isinstance(raw, Mapping):
        for dim in DIMS:
            v = _parse_first_float(raw.get(dim))
            if v is not None:
                out[dim] = max(0.0, min(1.0, v))
    return out


def _lop_weight(lopsidedness: float, strength: float) -> float:
    lop = min(1.0, max(0.0, float(lopsidedness)))
    return 1.0 + max(0.0, float(strength)) * lop


def _clamp_scores(raw: Any) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    if isinstance(raw, Mapping):
        for dim in DIMS:
            v = _parse_first_float(raw.get(dim))
            if v is not None:
                scores[dim] = max(0.0, min(1.0, v))
    return scores


def _load_g_states(
    path: Optional[str],
) -> Tuple[Dict[Tuple[str, str], Dict[str, float]], Dict[Tuple[str, str], Dict[str, float]]]:
    """Load learned-g per-node states: (direct_parse_map, f_readout_map).

    Each JSONL row: {"doc_id":.., "node_id":.., "compact_targets":{dim:val}|null,
    "f_readout":{dim:val}?}. ``compact_targets`` is g's state DIRECT-PARSED;
    ``f_readout`` is g's state read THROUGH f (the way g is actually used — f
    rescues off-schema states). Returns both maps so the eval can score either.
    """
    direct: Dict[Tuple[str, str], Dict[str, float]] = {}
    via_f: Dict[Tuple[str, str], Dict[str, float]] = {}
    if not path:
        return direct, via_f
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (str(row.get("doc_id")), str(row.get("node_id")))
            ct = row.get("compact_targets")
            if ct is None and isinstance(row.get("cmp_state"), Mapping):
                ct = row["cmp_state"].get("compact_targets")
            ds = _clamp_scores(ct)
            if ds:
                direct[key] = ds
            fs = _clamp_scores(row.get("f_readout"))
            if fs:
                via_f[key] = fs
    return direct, via_f


def evaluate(
    trees_path: str,
    *,
    strength: float,
    g_states_path: Optional[str] = None,
    split: Optional[str] = None,
) -> Dict[str, Any]:
    g_direct, g_via_f = _load_g_states(g_states_path)
    has_via_f = bool(g_via_f)
    # accumulators: level -> method -> [(abs_err, weight)] flattened over dims
    acc: Dict[int, Dict[str, List[Tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    lop_by_level: Dict[int, List[float]] = defaultdict(list)
    n_merges = 0
    g_nodes_found = 0
    g_nodes_missing = 0
    g_via_f_found = 0

    with open(trees_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if split and str(rec.get("metadata", {}).get("split") or rec.get("split") or "") not in ("", split):
                # only filter when the record actually carries a split
                rec_split = rec.get("metadata", {}).get("split") or rec.get("split")
                if rec_split and rec_split != split:
                    continue
            doc_id = str(rec.get("doc_id"))
            nodes = rec.get("nodes") or {}
            for nid, node in nodes.items():
                l_id, r_id = node.get("left_child_id"), node.get("right_child_id")
                if not l_id or not r_id or l_id == r_id:
                    continue
                left, right = nodes.get(l_id), nodes.get(r_id)
                if left is None or right is None:
                    continue
                ml, mr = _node_mass(left), _node_mass(right)
                if ml is None or mr is None or (ml + mr) <= 0:
                    continue
                parent = _node_scores(node)
                ls, rs = _node_scores(left), _node_scores(right)
                if not parent or not ls or not rs:
                    continue
                level = int(node.get("level", 0))
                wl = ml / (ml + mr)
                wr = 1.0 - wl
                lop = abs(ml - mr) / (ml + mr)
                weight = _lop_weight(lop, strength)
                lop_by_level[level].append(lop)
                n_merges += 1

                g_scores = g_direct.get((doc_id, str(nid)))
                gf_scores = g_via_f.get((doc_id, str(nid)))
                if g_states_path:
                    if g_scores:
                        g_nodes_found += 1
                    else:
                        g_nodes_missing += 1
                    if gf_scores:
                        g_via_f_found += 1

                for dim in DIMS:
                    if dim not in parent or dim not in ls or dim not in rs:
                        continue
                    tgt = parent[dim]
                    eq = 0.5 * (ls[dim] + rs[dim])
                    mw = wl * ls[dim] + wr * rs[dim]
                    acc[level]["equal_avg"].append((abs(eq - tgt), weight))
                    acc[level]["mass_wtd"].append((abs(mw - tgt), weight))
                    if g_scores and dim in g_scores:
                        acc[level]["learned_g"].append((abs(g_scores[dim] - tgt), weight))
                    if gf_scores and dim in gf_scores:
                        acc[level]["learned_g_via_f"].append((abs(gf_scores[dim] - tgt), weight))

    def wmae(pairs: Sequence[Tuple[float, float]]) -> Optional[float]:
        if not pairs:
            return None
        num = sum(e * w for e, w in pairs)
        den = sum(w for _, w in pairs)
        return num / den if den > 0 else None

    def umae(pairs: Sequence[Tuple[float, float]]) -> Optional[float]:
        if not pairs:
            return None
        return sum(e for e, _ in pairs) / len(pairs)

    methods = ["equal_avg", "mass_wtd"]
    if g_states_path:
        methods.append("learned_g")
        if has_via_f:
            methods.append("learned_g_via_f")
    by_level: List[Dict[str, Any]] = []
    for level in sorted(acc):
        lops = lop_by_level[level]
        row: Dict[str, Any] = {
            "level": level,
            "n_merges": len(lops),
            "lop_median": sorted(lops)[len(lops) // 2] if lops else None,
            "lop_p90": sorted(lops)[int(0.9 * (len(lops) - 1))] if lops else None,
        }
        for m in methods:
            row[f"{m}_mae"] = umae(acc[level][m])
            row[f"{m}_wmae"] = wmae(acc[level][m])
        by_level.append(row)

    # pooled (lopsidedness-weighted) across all levels
    pooled: Dict[str, Any] = {"n_merges": n_merges}
    for m in methods:
        flat = [p for level in acc for p in acc[level][m]]
        pooled[f"{m}_mae"] = umae(flat)
        pooled[f"{m}_wmae"] = wmae(flat)

    out = {
        "labeled_trees": trees_path,
        "lopsidedness_strength": float(strength),
        "n_merges": n_merges,
        "by_level": by_level,
        "pooled_weighted": pooled,
    }
    if g_states_path:
        out["g_states"] = {
            "path": g_states_path,
            "nodes_found": g_nodes_found,
            "nodes_missing": g_nodes_missing,
            "via_f_found": g_via_f_found,
            "has_via_f": has_via_f,
        }
    return out


def _fmt(v: Optional[float]) -> str:
    return f"{v:.5f}" if v is not None else "   n/a "


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labeled-trees", required=True)
    ap.add_argument(
        "--lopsidedness-strength",
        type=float,
        default=4.0,
        help="weight = 1 + strength * lopsidedness (0 = unweighted)",
    )
    ap.add_argument(
        "--g-states-jsonl",
        default=None,
        help="optional per-node learned-g states to score vs the same yardstick",
    )
    ap.add_argument("--split", default=None, help="filter to this split if records carry one")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    result = evaluate(
        args.labeled_trees,
        strength=args.lopsidedness_strength,
        g_states_path=args.g_states_jsonl,
        split=args.split,
    )

    methods = ["equal_avg", "mass_wtd"]
    if args.g_states_jsonl:
        methods.append("learned_g")
        if result.get("g_states", {}).get("has_via_f"):
            methods.append("learned_g_via_f")
    print(f"\n=== per-level merge MAE (lopsidedness strength={args.lopsidedness_strength}) ===")
    print(f"  trees: {args.labeled_trees}  merges: {result['n_merges']}")
    hdr = f'{"lvl":>3} {"n":>6} {"lop_p90":>8}'
    for m in methods:
        hdr += f' {m+"_wmae":>16}'
    print(hdr)
    for r in result["by_level"]:
        line = f'{r["level"]:>3} {r["n_merges"]:>6} {_fmt(r["lop_p90"]):>8}'
        for m in methods:
            line += f' {_fmt(r.get(f"{m}_wmae")):>16}'
        print(line)
    p = result["pooled_weighted"]
    print("\n=== pooled (lopsidedness-weighted) ===")
    for m in methods:
        print(f'  {m:>10}: wmae={_fmt(p.get(f"{m}_wmae"))}  (unweighted mae={_fmt(p.get(f"{m}_mae"))})')
    if args.g_states_jsonl:
        gs = result["g_states"]
        print(f'  g states: {gs["nodes_found"]} direct-parse found / {gs["nodes_missing"]} missing'
              + (f'  | {gs.get("via_f_found")} via-f scorable' if gs.get("has_via_f") else ""))
        eq, mw = p.get("equal_avg_wmae"), p.get("mass_wtd_wmae")
        lg, lgf = p.get("learned_g_wmae"), p.get("learned_g_via_f_wmae")
        if None not in (eq, mw, lg):
            print(f'\n  VERDICT (direct-parse): learned_g wmae={lg:.5f}  | beats equal_avg ({eq:.5f})? '
                  f'{"YES" if lg < eq else "no"}  | gap-to-ceiling: {lg-mw:+.5f}')
        if lgf is not None and None not in (eq, mw):
            print(f'  VERDICT (through f): learned_g_via_f wmae={lgf:.5f}  | beats equal_avg ({eq:.5f})? '
                  f'{"YES" if lgf < eq else "no"}  | gap-to-ceiling: {lgf-mw:+.5f}')

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
