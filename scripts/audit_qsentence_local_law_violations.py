#!/usr/bin/env python3
"""Audit per-node local-law violations (paper sec:min-framework) for the
q-sentence ladder, broken out by tree level and leaf size.

Laws (all read THROUGH the f readout: violation = mean_dim |f(a)-f(b)|):
  C1  Sufficiency        (leaf)   d(f(g(b)), f(b))
  C2  Idempotence        (leaf s) d(f(g(s)), f(s))
  C3a Joint faithfulness (merge)  d(f(u@v),  f(g(u@v)))
  C3b Compositionality   (merge)  d(f(g(u@v)), f(g(g(u)@g(v))))

Hypothesis under test: smaller leaves -> deeper trees -> more merges -> if the
overall composition is bad, the per-law violation (esp C3b) should GROW with
tree depth/level. Runs on BOTH gold states (sanity ~0) and learned-g states.

Usage:
  ./venv/bin/python scripts/audit_qsentence_local_law_violations.py \
      --fg-grid-dir outputs/manifesto_qsentence_dspy_labeled_grid \
      --f-artifact <f.json/dir> \
      --g-artifact <g.json>            # omit for gold-only \
      --leaf-qsentences 2,4,8 --n-docs 10 \
      --dspy-model openai/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4 \
      --dspy-api-base http://localhost:8004/v1 \
      --output-dir outputs/law_audit_<tag>
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ctreepo.distillation import load_labeled_trees  # noqa: E402
from src.ctreepo.manifesto_qsentence_dspy_family import (  # noqa: E402
    ManifestoQSentenceDSPyFamily,
    ManifestoQSentenceDSPyFamilyConfig,
)

LAWS = ["C1_sufficiency", "C2_idempotence", "C3a_joint_faithfulness", "C3b_compositionality"]


def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None]
    return float(statistics.fmean(xs)) if xs else None


def _row_scalar(row: Dict[str, Any]) -> Optional[float]:
    """Mean over dimensions of one node's violation dict."""
    vals = [v for v in (row.get("violation") or {}).values() if v is not None]
    return float(statistics.fmean(vals)) if vals else None


def _build_family(args) -> ManifestoQSentenceDSPyFamily:
    return ManifestoQSentenceDSPyFamily(
        config=ManifestoQSentenceDSPyFamilyConfig(
            optimizer="gepa",
            lm_config={
                "model": str(args.dspy_model),
                "api_base": str(args.dspy_api_base),
                "api_key": "EMPTY",
                "max_tokens": int(args.dspy_max_tokens),
            },
            lm_transport=str(args.dspy_lm_transport),
            num_threads=int(args.dspy_num_threads),
            batch_max_concurrent=int(args.dspy_num_threads),
            # CRITICAL: batch_processor caps in-flight batches to
            # ceil(max_concurrent/batch_size). The default batch_size(64) >
            # max_concurrent(48) => only 1 batch in flight => the whole 4-GPU
            # fleet gets fed ~1-4 requests total (one worker's worth) and 3 GPUs
            # starve. A SMALL batch_size lets many batches fly: ceil(48/2)=24
            # in-flight, enough to feed all 4 workers their ~4-concurrent sweet
            # spot (single-worker saturates ~10 req/s at N=4; fleet target ~40).
            batch_size=int(args.dspy_batch_size),
            batch_routing_policy=str(args.dspy_batch_routing_policy),
            # The per-call HF tokenization guard holds the GIL and serializes the
            # ~16-way doc fan-out; truncation is not a concern for a diagnostic
            # audit, so skip it to let LM calls actually run concurrently.
            skip_lm_input_budget_check=True,
            # Bound hung calls so a malformed-JSON retry loop can't park a worker
            # thread forever (default request_timeout=300s, await=None -> deadlock
            # observed: 86 threads sleeping, fleet idle). Fail fast -> the resilient
            # auditor records the node as a failure and moves on.
            batch_request_timeout=float(args.dspy_batch_request_timeout),
            batch_await_response_timeout=float(args.dspy_batch_await_response_timeout),
            leaf_size_tokens=int(args.leaf_size_tokens),
            lm_context_window_tokens=int(args.dspy_lm_context_tokens),
            max_completion_tokens=int(args.dspy_max_tokens),
            strict_optimizer_errors=False,
        )
    )


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean violation by (state_source, leaf, law, level) + law shares."""
    by_key: Dict[tuple, List[float]] = collections.defaultdict(list)
    by_law_total: Dict[tuple, List[float]] = collections.defaultdict(list)
    for r in rows:
        s = _row_scalar(r)
        if s is None:
            continue
        k = (r["state_source"], r["leaf_qsentences"], r["law"], r["level"])
        by_key[k].append(s)
        by_law_total[(r["state_source"], r["leaf_qsentences"], r["law"])].append(s)

    by_level = [
        {"state_source": ss, "leaf": leaf, "law": law, "level": lvl,
         "mean_violation": _mean(v), "n": len(v)}
        for (ss, leaf, law, lvl), v in sorted(by_key.items())
    ]
    # law share of total violation within (state_source, leaf)
    totals: Dict[tuple, float] = {}
    for (ss, leaf, law), v in by_law_total.items():
        totals[(ss, leaf)] = totals.get((ss, leaf), 0.0) + (sum(v) / len(v) if v else 0.0)
    law_share = []
    for (ss, leaf, law), v in sorted(by_law_total.items()):
        mean_v = (sum(v) / len(v)) if v else 0.0
        denom = totals.get((ss, leaf), 0.0)
        law_share.append({
            "state_source": ss, "leaf": leaf, "law": law,
            "mean_violation": mean_v,
            "share_of_total": (mean_v / denom) if denom > 0 else None,
            "n": len(v),
        })
    return {"by_level": by_level, "law_share": law_share}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fg-grid-dir", default="outputs/manifesto_qsentence_dspy_labeled_grid")
    ap.add_argument("--leaf-qsentences", default="2,4,8")
    ap.add_argument("--leaf-size-tokens", type=int, default=512)
    ap.add_argument("--n-docs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260623)
    ap.add_argument("--f-artifact", required=True)
    ap.add_argument("--g-artifact", default=None, help="omit => gold-states audit only")
    ap.add_argument("--gold-only", action="store_true")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--dspy-model", default="openai/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4")
    ap.add_argument("--dspy-api-base", default="http://localhost:8004/v1")
    ap.add_argument("--dspy-lm-transport", choices=["batch", "litellm"], default="batch")
    ap.add_argument("--dspy-num-threads", type=int, default=48)
    ap.add_argument(
        "--dspy-batch-routing-policy",
        default="round_robin",
        help="round_robin spreads every request across the fleet; affinity pins to one server",
    )
    ap.add_argument(
        "--doc-concurrency",
        type=int,
        default=16,
        help="docs audited concurrently so f AND g calls fan out across all fleet GPUs",
    )
    ap.add_argument(
        "--doc-timeout",
        type=float,
        default=180.0,
        help="hard wall-clock timeout per doc (s); a hung doc is abandoned so the leaf still completes",
    )
    ap.add_argument("--dspy-lm-context-tokens", type=int, default=32768)
    ap.add_argument("--dspy-max-tokens", type=int, default=1024)
    ap.add_argument("--dspy-batch-size", type=int, default=2,
                    help="SMALL => many in-flight batches (ceil(max_concurrent/batch_size)) to feed all fleet GPUs")

    ap.add_argument("--dspy-batch-request-timeout", type=float, default=90.0,
                    help="per-request hard timeout (s); bounds hung malformed-JSON retries")
    ap.add_argument("--dspy-batch-await-response-timeout", type=float, default=120.0,
                    help="max wait for a batched response (s); prevents thread deadlock")

    args = ap.parse_args()

    out_dir = Path(args.output_dir or "outputs/law_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    leaves = [int(x) for x in str(args.leaf_qsentences).replace(";", ",").split(",") if x.strip()]

    family = _build_family(args)
    f_program = family._load_f_program(str(args.f_artifact))
    g_program = (
        family._load_g_program(str(args.g_artifact))
        if (args.g_artifact and not args.gold_only)
        else None
    )

    all_rows: List[Dict[str, Any]] = []
    for leaf in leaves:
        grid = Path(args.fg_grid_dir) / f"leafq{leaf:03d}" / "labeled_trees.jsonl"
        if not grid.exists():
            print(f"[skip] no grid for leaf={leaf}: {grid}")
            continue
        trees = load_labeled_trees(grid)
        rng = random.Random(int(args.seed) + leaf)
        if len(trees) > int(args.n_docs):
            trees = [trees[i] for i in sorted(rng.sample(range(len(trees)), int(args.n_docs)))]

        def audit_one(tree: Any) -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            gold_rows = family.audit_local_law_violations(
                f_program=f_program, g_program=family.TEACHER_PASSTHROUGH,
                tree=tree, use_gold_states=True,
            )
            for r in gold_rows:
                r["state_source"] = "gold"
            rows.extend(gold_rows)
            if g_program is not None:
                g_rows = family.audit_local_law_violations(
                    f_program=f_program, g_program=g_program,
                    tree=tree, use_gold_states=False,
                )
                for r in g_rows:
                    r["state_source"] = "learned_g"
                rows.extend(g_rows)
            return rows

        # Process docs CONCURRENTLY so their LM calls fan out across the fleet.
        # CRITICAL: each doc gets a HARD per-doc wall-clock timeout. A doc whose
        # bottom-up chain hangs (DSPy internal retry loop on malformed dgemma
        # JSON, transport stall, etc.) is ABANDONED and recorded as a failure,
        # so the leaf always completes. pool.map blocks forever on a single hung
        # doc; submit + per-future timeout does not.
        leaf_rows: List[Dict[str, Any]] = []
        doc_workers = max(1, min(int(args.doc_concurrency), len(trees)))
        doc_timeout = float(args.doc_timeout)
        completed = 0
        abandoned = 0
        # Per-doc timeout, applied correctly: docs run in WAVES of doc_workers,
        # so the whole-leaf budget = doc_timeout * number_of_waves. A doc that
        # genuinely hangs is abandoned once the budget passes; docs that are just
        # working (most of them) finish well inside it. (A flat as_completed
        # timeout would wrongly abandon every doc that didn't finish in the first
        # doc_timeout window, even healthy ones.)
        import math as _math
        waves = max(1, _math.ceil(len(trees) / doc_workers))
        leaf_budget = doc_timeout * waves
        with ThreadPoolExecutor(max_workers=doc_workers) as pool:
            futs = {pool.submit(audit_one, tree): i for i, tree in enumerate(trees)}
            pending = set(futs)
            try:
                for fut in as_completed(list(futs), timeout=leaf_budget):
                    pending.discard(fut)
                    try:
                        leaf_rows.extend(fut.result())
                        completed += 1
                    except Exception as exc:
                        abandoned += 1
                        print(f"  [leaf={leaf}] doc {futs[fut]} failed: "
                              f"{type(exc).__name__}", flush=True)
            except TimeoutError:
                pass
            # Anything still pending past the whole-leaf budget is genuinely hung.
            for fut in pending:
                abandoned += 1
                fut.cancel()
                print(f"  [leaf={leaf}] doc {futs[fut]} abandoned at {leaf_budget:.0f}s leaf budget",
                      flush=True)
        all_rows.extend(leaf_rows)
        # Write a per-leaf summary INCREMENTALLY so partial progress survives a
        # later hang and is inspectable as soon as this leaf finishes.
        leaf_dir = out_dir / f"leaf{leaf}"
        leaf_dir.mkdir(parents=True, exist_ok=True)
        (leaf_dir / "law_violation_summary.json").write_text(
            json.dumps(_aggregate(leaf_rows), indent=2, sort_keys=True)
        )
        print(f"[leaf={leaf}] audited {completed}/{len(trees)} docs "
              f"({abandoned} abandoned, {doc_workers}-way, {doc_timeout:.0f}s/doc timeout)",
              flush=True)

    summary = _aggregate(all_rows)
    (out_dir / "law_violations_nodes.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in all_rows) + "\n"
    )
    (out_dir / "law_violation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    # human-readable: violation by level, per (state_source, leaf, law)
    print("\n=== mean law violation by level (smaller leaf = deeper tree) ===")
    print(f'{"src":9s} {"leaf":>4s} {"law":24s} {"level":>5s} {"viol":>8s} {"n":>5s}')
    for r in summary["by_level"]:
        mv = r["mean_violation"]
        print(f'{r["state_source"]:9s} {r["leaf"]:>4d} {r["law"]:24s} {r["level"]:>5d} '
              f'{(mv if mv is not None else float("nan")):>8.4f} {r["n"]:>5d}')
    print("\n=== law share of total violation (per state_source, leaf) ===")
    for r in summary["law_share"]:
        sh = r["share_of_total"]
        print(f'  {r["state_source"]:9s} leaf={r["leaf"]:<3d} {r["law"]:24s} '
              f'viol={r["mean_violation"]:.4f} share={(f"{sh:.1%}" if sh is not None else "n/a")}')
    print(f"\nwrote {out_dir}/law_violation_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
