#!/usr/bin/env python3
"""
Phase 3: DSPy-optimize the full pipeline with unified g plus scorer f.

Previous work only optimized the scorer's `dspy.Predict`. For dimensions like
Decentralization where the gating factor is how much relevant content survives
summarization, scorer-only optimization can't help. This script compiles the
whole `DimensionFullPipeline` (chunk -> unified-g leaves -> unified-g pairwise
merge -> score root) via DSPy. Optimizer scopes are exactly `f`, `g`, or `gf`;
`gf` means the unified g plus scorer f, never separate leaf/merge prompts.

Train pool: Benoit's open-weight LLM ensemble mean on non-test manifestos
(same Pool C as phase1 / phase2). Test: held-out Benoit expert-benchmark
manifestos.

Usage:
    python scripts/phase3_full_pipeline_optimize.py \\
        --ports 8010 8011 8012 8013 \\
        --dimension decentralization \\
        --optimizer bootstrap \\
        --train-n 30 --test-n 120 \\
        --chunk-chars 24000 \\
        --output-dir outputs/phase3/decentralization_bfs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlopen

import dspy
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config.dspy_config import configure_dspy, create_vllm_lm, create_vllm_lm_multi
from src.core.protocols import format_merge_input
from src.preprocessing.chunker import chunk_for_ops
from src.tasks.manifesto import ManifestoDataset
from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r
from src.tasks.manifesto.dimensions import (
    BENOIT_DIMENSIONS,
    PolicyDimension,
    get_preservation_rubric,
)
from src.tasks.manifesto.dimension_scorer import DimensionScoreSignature
from src.tasks.manifesto.expert_benchmarks import (
    benoit_ensemble_mean,
    load_benoit_expert_means,
    load_benoit_llm_scores,
    load_benoit_mp_crosswalk,
)
from src.tasks.manifesto.pipeline import UnifiedManifestoG
from src.tasks.manifesto.pipeline_config import DEFAULT_SCORER_MAX_TOKENS
from src.tasks.manifesto.resume_utils import load_resume_rows
from src.tasks.manifesto.scoring_contexts import get_scoring_context
from src.core.prompting import parse_numeric_score

logger = logging.getLogger(__name__)
_DIM_FROM_NAME = {d.value: d for d in PolicyDimension}


class _FrozenCallable:
    """Callable wrapper that keeps a module out of DSPy's optimizer traversal."""

    def __init__(self, module):
        self._module = module

    def __call__(self, **kwargs):
        return self._module(**kwargs)


class TraceableUnifiedManifestoG(UnifiedManifestoG):
    """Unified g variant that leaves a Prediction object in GEPA traces."""

    def forward(self, content: str, rubric: str) -> dspy.Prediction:
        summary = super().forward(content=content, rubric=rubric)
        return dspy.Prediction(summary=summary)


def _summary_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return str(result.get("summary", ""))
    return str(getattr(result, "summary", result))


def _json_fingerprint(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _module_fingerprint(module) -> str:
    inner = getattr(module, "_module", module)
    try:
        state = inner.dump_state()
    except Exception:  # noqa: BLE001
        state = repr(inner)
    return _json_fingerprint(state)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


class DimensionFullPipeline(dspy.Module):
    """Full pipeline for one policy dimension: chunk -> unified g -> score.

    The optimizer-visible scope is exactly one of:
    - f: scorer only
    - g: unified tree summarizer only
    - gf: unified g plus scorer f
    """

    def __init__(
        self,
        dim: PolicyDimension,
        *,
        chunk_chars: int = 24000,
        max_workers: int = 8,
        optimize_scope: str = "gf",
        enable_node_cache: bool = True,
    ):
        super().__init__()
        if optimize_scope not in {"f", "g", "gf"}:
            raise ValueError(f"optimize_scope must be one of f/g/gf, got {optimize_scope!r}")
        self.dim = dim
        self.spec = BENOIT_DIMENSIONS[dim]
        self.rubric = get_preservation_rubric(dim)
        self.task_context = get_scoring_context(dim)
        self.chunk_chars = chunk_chars
        self.max_workers = max_workers
        self.optimize_scope = optimize_scope
        self.enable_node_cache = bool(enable_node_cache)
        self.scorer_max_tokens = int(DEFAULT_SCORER_MAX_TOKENS)
        self._node_cache: dict[str, str] = {}
        self._node_cache_hits = 0
        self._node_cache_misses = 0

        g_module = TraceableUnifiedManifestoG(use_cot=False)
        scorer_module = dspy.Predict(DimensionScoreSignature)
        self.g = g_module if optimize_scope in {"g", "gf"} else _FrozenCallable(g_module)
        self.scorer = scorer_module if optimize_scope in {"f", "gf"} else _FrozenCallable(scorer_module)

    def _g(self, content: str) -> str:
        if not self.enable_node_cache:
            return _summary_text(self.g(content=content, rubric=self.rubric))
        key = _json_fingerprint(
            {
                "candidate": _module_fingerprint(self.g),
                "content": _hash_text(content),
                "rubric": _hash_text(self.rubric),
            }
        )
        cached = self._node_cache.get(key)
        if cached is not None:
            self._node_cache_hits += 1
            return cached
        self._node_cache_misses += 1
        summary = _summary_text(self.g(content=content, rubric=self.rubric))
        self._node_cache[key] = summary
        return summary

    def cache_stats(self) -> dict[str, int]:
        return {
            "node_cache_hits": int(self._node_cache_hits),
            "node_cache_misses": int(self._node_cache_misses),
            "node_cache_size": len(self._node_cache),
        }

    def _map_nodes(self, fn, items):
        if self.max_workers <= 1:
            return [fn(item) for item in items]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(fn, items))

    def forward(self, text: str) -> dspy.Prediction:
        chunks = chunk_for_ops(text, max_chars=self.chunk_chars, strategy="axis")
        if not chunks:
            return dspy.Prediction(score=None, summary="", reasoning="no chunks")

        summaries = self._map_nodes(lambda c: self._g(c.text), chunks)
        while len(summaries) > 1:
            pairs, carry = [], None
            for i in range(0, len(summaries), 2):
                if i + 1 < len(summaries):
                    pairs.append((summaries[i], summaries[i + 1]))
                else:
                    carry = summaries[i]
            merged = self._map_nodes(lambda p: self._g(format_merge_input(p[0], p[1])), pairs)
            if carry is not None:
                merged.append(carry)
            summaries = merged

        final_summary = summaries[0]
        scored = self.scorer(
            task_context=self.task_context,
            summary=final_summary,
            config={"max_tokens": self.scorer_max_tokens},
        )
        raw_str = str(getattr(scored, "score", ""))
        if raw_str.strip().lower() in {"na", "n/a", "none", ""}:
            return dspy.Prediction(score=None, summary=final_summary,
                                   reasoning=getattr(scored, "reasoning", ""))
        raw = parse_numeric_score(
            raw_str, min_value=self.spec.scale.min_value,
            max_value=self.spec.scale.max_value, allow_llm_fallback=True,
        )
        if raw is None:
            return dspy.Prediction(score=None, summary=final_summary,
                                   reasoning=getattr(scored, "reasoning", ""))
        return dspy.Prediction(
            score=self.spec.scale.clamp(float(raw)),
            summary=final_summary,
            reasoning=getattr(scored, "reasoning", ""),
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--ports", type=int, nargs="+", default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--dimension", type=str, default="decentralization", choices=sorted(_DIM_FROM_NAME))
    p.add_argument("--optimizer", choices=["bootstrap", "miprov2", "gepa", "none"], default="bootstrap")
    p.add_argument("--train-pool", default="expert-split",
                   choices=["openweight", "expert", "expert-split"],
                   help="'expert-split' = split the 235 Benoit expert manifestos into train/test "
                        "(only option when we need raw text since crosswalk covers 233).")
    p.add_argument("--split-strategy", choices=["random", "label-stratified"], default="random",
                   help="How to split expert-split examples. label-stratified keeps small dev/test "
                        "sets from collapsing to one side of the 1-7 scale.")
    p.add_argument("--train-n", type=int, default=30)
    p.add_argument("--dev-n", type=int, default=10)
    p.add_argument("--test-n", type=int, default=120)
    p.add_argument("--chunk-chars", type=int, default=24000)
    p.add_argument("--max-demos", type=int, default=4)
    p.add_argument("--optimize-scope", choices=["f", "g", "gf"], default="gf",
                   help="Optimizer-visible modules: scorer f, unified g, or both.")
    p.add_argument("--metric-mode", choices=["mae", "rank"], default="rank",
                   help="GEPA/optimizer metric: MAE-style score or rank-side-aware score.")
    p.add_argument("--feedback-mode", choices=["scalar", "rich"], default="rich",
                   help="GEPA metric return mode. rich returns ScoreWithFeedback.")
    p.add_argument("--gepa-auto", choices=["light", "medium", "heavy"], default="light")
    p.add_argument("--gepa-threads", type=int, default=4)
    p.add_argument("--gepa-valset-cap", type=int, default=4,
                   help="Cap GEPA validation examples; <=0 uses the full dev set.")
    p.add_argument("--gepa-max-metric-calls", type=int, default=0,
                   help="Explicit GEPA metric-call budget; overrides --gepa-auto when >0.")
    p.add_argument("--selection-guard", choices=["none", "dev"], default="none",
                   help="Select baseline vs optimized by dev performance before test reporting.")
    p.add_argument("--reflection-max-tokens", type=int, default=2048,
                   help="Max tokens for GEPA reflection LM calls.")
    p.add_argument("--init-program", type=Path, default=None,
                   help="Optional saved DimensionFullPipeline program JSON to warm-start before optimization.")
    p.add_argument("--init-dir", type=Path, default=None,
                   help="Optional artifact directory; loads compatible optimized_program/optimized_scorer/unified_g files if present.")
    p.add_argument("--init-artifact-kind", choices=["final", "optimized"], default="final",
                   help="When --init-dir is given, prefer final artifacts for standalone reporting "
                        "or optimized artifacts for staged f/g continuation.")
    p.add_argument("--init-components-only", action="store_true",
                   help="With --init-dir, load scorer/unified-g component artifacts without "
                        "inferring a full-program artifact. This is the safe mode for staged "
                        "f/g continuation across optimizer scopes.")
    p.add_argument("--init-scorer", type=Path, default=None,
                   help="Optional scorer-only artifact JSON to warm-start f.")
    p.add_argument("--init-g", type=Path, default=None,
                   help="Optional unified-g artifact JSON to warm-start g.")
    p.add_argument("--init-g-legacy-leaf", type=Path, default=None,
                   help="Optional legacy LeafSummarizer artifact; transplants only "
                        "the learned instruction text into unified g.")
    p.add_argument("--cheat-train-on-test", action="store_true",
                   help="Diagnostic only: use held-out test examples as GEPA train/dev to test in-sample overfit.")
    p.add_argument("--max-workers", type=int, default=8, help="Threads inside one pipeline call")
    p.add_argument("--seed", type=int, default=0)
    env_cap = os.environ.get("MANIFESTO_MAX_TOKENS")
    p.add_argument("--max-tokens", type=int, default=int(env_cap) if env_cap else None)
    p.add_argument("--mp-data-dir", type=Path,
                   default=project_root / "data" / "raw" / "manifesto_corpus_benoit")
    p.add_argument("--output-dir", type=Path,
                   default=project_root / "outputs" / "phase3" /
                   f"full_pipeline_optimize_{datetime.now(timezone.utc):%Y%m%d_%H%M}")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _stratified_take(records: list[dict[str, Any]], n: int, rng) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Take n roughly quantile-spaced records by expert label."""
    if n <= 0:
        return [], list(records)
    if n > len(records):
        raise ValueError(f"Cannot take {n} records from pool of {len(records)}")
    ordered = sorted(records, key=lambda r: (float(r["label"]), str(r["manifesto_id"])))
    selected_ids: set[int] = set()
    selected: list[dict[str, Any]] = []
    for i in range(n):
        start = i * len(ordered) // n
        end = (i + 1) * len(ordered) // n
        bucket = ordered[start:end] or ordered
        choices = [r for r in bucket if id(r) not in selected_ids]
        if not choices:
            choices = [r for r in ordered if id(r) not in selected_ids]
        rec = rng.choice(choices)
        selected.append(rec)
        selected_ids.add(id(rec))
    remaining = [r for r in records if id(r) not in selected_ids]
    rng.shuffle(selected)
    rng.shuffle(remaining)
    return selected, remaining


def _split_records(
    records: list[dict[str, Any]],
    *,
    train_n: int,
    dev_n: int,
    test_n: int,
    split_strategy: str,
    rng,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    needed = train_n + dev_n + test_n
    if len(records) < needed:
        raise SystemExit(
            f"Not enough records: have {len(records)}, need train={train_n}+"
            f"dev={dev_n}+test={test_n}={needed}"
        )
    pool = list(records)
    if split_strategy == "random":
        rng.shuffle(pool)
        return pool[:train_n], pool[train_n:train_n + dev_n], pool[train_n + dev_n:needed]
    if split_strategy != "label-stratified":
        raise ValueError(f"unknown split strategy: {split_strategy}")
    train, pool = _stratified_take(pool, train_n, rng)
    dev, pool = _stratified_take(pool, dev_n, rng)
    test, pool = _stratified_take(pool, test_n, rng)
    return train, dev, test


def _build_examples(
    dim: PolicyDimension,
    train_pool: str,
    mp_data_dir: Path,
    train_n: int,
    dev_n: int,
    test_n: int,
    seed: int,
    split_strategy: str = "random",
) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
    """Build train, dev, test examples with (text, expert_mean)-like pairs.

    Train labels = Benoit open-weight LLM ensemble mean (on non-test manifestos).
    Test labels = Benoit expert ensemble mean (on held-out test manifestos).

    Joins via (party, year) between our local text dataset and Benoit's manifesto
    strings; skips any manifestos without local text.
    """
    ds = ManifestoDataset(data_dir=mp_data_dir, require_text=True)
    crosswalk = load_benoit_mp_crosswalk()
    benoit_to_py = {
        row.manifesto: (int(row.party), int(row.year))
        for row in crosswalk.itertuples()
    }
    py_to_mid: dict[tuple[int, int], str] = {}
    for mid in ds.get_all_ids():
        s = ds.get_sample(mid)
        if s is None:
            continue
        py_to_mid[(int(s.party_id), int(s.year))] = mid

    # Test: expert means
    experts = load_benoit_expert_means(dim)
    test_records = []
    for row in experts.itertuples():
        key = benoit_to_py.get(str(row.manifesto))
        if key is None:
            continue
        mid = py_to_mid.get(key)
        if mid is None:
            continue
        sample = ds.get_sample(mid)
        if sample is None or not sample.text:
            continue
        test_records.append({
            "manifesto_id": mid,
            "benoit_key": str(row.manifesto),
            "text": sample.text,
            "label": float(row.expert_mean),
            "party": key[0],
            "year": key[1],
        })

    import random
    rng = random.Random(seed)

    if train_pool == "expert-split":
        # Split the 235 expert-benchmarked manifestos into train / dev / test.
        # Train uses a BENOIT-DISJOINT label source (openweight) when available
        # to avoid circularity, but falls back to expert labels for manifestos
        # without openweight scores.
        # The crucial property: test manifestos have NEVER been seen by the
        # optimizer; train manifestos carry labels that may correlate with
        # (but are distinct from) the evaluation labels.
        ow_scores = load_benoit_llm_scores(kind="openweight", dimension=dim)
        ow_ens = benoit_ensemble_mean(ow_scores)
        ow_lookup = {row.manifesto: float(row.score_llm_mean) for row in ow_ens.itertuples()}

        all_records = list(test_records)  # all 233 with local text
        train_records, dev_records, test_records = _split_records(
            all_records,
            train_n=train_n,
            dev_n=dev_n,
            test_n=test_n,
            split_strategy=split_strategy,
            rng=rng,
        )

        # Swap the train labels to openweight-mean where available, so the
        # training signal is disjoint from expert-label leakage.
        train = []
        dev = []
        for rec in train_records:
            ow_label = ow_lookup.get(rec["benoit_key"])
            label = ow_label if ow_label is not None else rec["label"]
            ex = dspy.Example(text=rec["text"], expert_mean=label,
                              manifesto_id=rec["manifesto_id"]).with_inputs("text")
            train.append(ex)
        for rec in dev_records:
            ow_label = ow_lookup.get(rec["benoit_key"])
            label = ow_label if ow_label is not None else rec["label"]
            dev.append(
                dspy.Example(text=rec["text"], expert_mean=label,
                             manifesto_id=rec["manifesto_id"]).with_inputs("text")
            )
        test = [
            dspy.Example(text=r["text"], expert_mean=r["label"],
                         manifesto_id=r["manifesto_id"]).with_inputs("text")
            for r in test_records
        ]
        return train, dev, test

    # Legacy paths: disjoint pools (require non-test manifestos with local text,
    # which doesn't work for raw-text path since crosswalk is 233).
    if train_pool == "openweight":
        scores = load_benoit_llm_scores(kind="openweight", dimension=dim)
        ensemble = benoit_ensemble_mean(scores)
        train_lookup = {row.manifesto: float(row.score_llm_mean) for row in ensemble.itertuples()}
    elif train_pool == "expert":
        train_lookup = {row.manifesto: float(row.expert_mean) for row in experts.itertuples()}
    else:
        raise ValueError(train_pool)

    test_keys_s = set(row["benoit_key"] for row in test_records)
    train_records = []
    for bkey, label in train_lookup.items():
        if bkey in test_keys_s:
            continue
        key = benoit_to_py.get(bkey)
        if key is None:
            continue
        mid = py_to_mid.get(key)
        if mid is None:
            continue
        sample = ds.get_sample(mid)
        if sample is None or not sample.text:
            continue
        train_records.append({
            "manifesto_id": mid, "benoit_key": bkey, "text": sample.text,
            "label": float(label), "party": key[0], "year": key[1],
        })

    train, dev, _unused = _split_records(
        train_records,
        train_n=train_n,
        dev_n=dev_n,
        test_n=0,
        split_strategy=split_strategy,
        rng=rng,
    )
    test, _remaining_test = _stratified_take(test_records, test_n, rng) if split_strategy == "label-stratified" else (test_records[:test_n], test_records[test_n:])

    def _ex(r):
        return dspy.Example(text=r["text"], expert_mean=r["label"],
                            manifesto_id=r["manifesto_id"]).with_inputs("text")
    return [_ex(r) for r in train], [_ex(r) for r in dev], [_ex(r) for r in test]


def _score_from_prediction(prediction) -> Optional[float]:
    score = getattr(prediction, "score", None)
    if score is None or score == "":
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def _metric(example, prediction, trace=None, *, mode: str = "mae"):
    del trace
    pred_score = _score_from_prediction(prediction)
    if pred_score is None:
        return 0.0
    target = float(example.expert_mean)
    mae_score = max(0.0, 1.0 - abs(pred_score - target) / 6.0)
    if mode == "mae":
        return mae_score
    if mode == "rank":
        center = 4.0
        target_side = "above" if target >= center else "below"
        pred_side = "above" if pred_score >= center else "below"
        side_penalty = 0.25 if target_side != pred_side else 0.0
        return max(0.0, mae_score - side_penalty)
    raise ValueError(f"unknown metric mode: {mode}")


def _make_metric(mode: str):
    def metric(example, prediction, trace=None):
        return _metric(example, prediction, trace=trace, mode=mode)
    return metric


def _make_gepa_metric(spec, *, mode: str, feedback_mode: str):
    scalar_metric = _make_metric(mode)
    if feedback_mode == "scalar":
        def scalar_gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            del pred_name, pred_trace
            return scalar_metric(gold, pred, trace=trace)
        return scalar_gepa_metric

    from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

    def rich_gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        del trace, pred_name, pred_trace
        score = float(scalar_metric(gold, pred))
        pred_score = _score_from_prediction(pred)
        target = float(gold.expert_mean)
        reasoning = str(getattr(pred, "reasoning", "") or "")
        summary = str(getattr(pred, "summary", "") or "")
        if pred_score is None:
            feedback = (
                "Parse/NA failure: prediction did not contain a numeric score. "
                f"Target was {target:.3f} on {spec.dimension.value}. "
                "Return a valid numeric score on the 1-7 scale."
            )
        else:
            error = pred_score - target
            direction = "lower" if error > 0 else "higher" if error < 0 else "unchanged"
            center = 4.0
            pred_side = "above-neutral" if pred_score >= center else "below-neutral"
            target_side = "above-neutral" if target >= center else "below-neutral"
            anchors = f"1={spec.anchor_low}; 7={spec.anchor_high}; 4=neutral/mixed"
            feedback = (
                f"Predicted score: {pred_score:.3f}. Target: {target:.3f}. "
                f"Absolute error: {abs(error):.3f}. Direction of correction: {direction}. "
                f"Rank-side check: prediction is {pred_side}; target is {target_side}. "
                f"Scale anchors: {anchors}. "
                "Use the summary evidence and dimension rubric; avoid generic party stereotypes.\n"
                f"Model reasoning (truncated): {reasoning[:800] or '(empty)'}\n"
                f"Final summary excerpt (truncated): {summary[:800] or '(empty)'}"
            )
        return ScoreWithFeedback(score=score, feedback=feedback)

    return rich_gepa_metric


def _predict(program: dspy.Module, ex: dspy.Example) -> Optional[float]:
    try:
        pred = program(text=ex.text)
    except Exception as e:  # noqa: BLE001
        logger.warning("prediction failed: %s", e)
        return None
    return _score_from_prediction(pred)


def _evaluate(program: dspy.Module, examples: list[dspy.Example], label: str,
              output_dir: Path, dim_value: str) -> dict:
    t0 = time.time()
    out_path = output_dir / f"per_mfesto_{label}.jsonl"
    already, resuming = load_resume_rows(out_path, log_label=f"{dim_value}/{label}")
    rows: list[dict] = list(already.values())
    preds: list[float | None] = [r.get("pred") for r in rows]
    truths: list[float] = [float(r["expert_mean"]) for r in rows]
    with out_path.open("a" if resuming else "w") as fp:
        for i, ex in enumerate(examples):
            mid = getattr(ex, "manifesto_id", None)
            if mid is not None and str(mid) in already:
                continue
            p = _predict(program, ex)
            row = {
                "phase": label, "dimension": dim_value,
                "manifesto_id": mid,
                "pred": p, "expert_mean": float(ex.expert_mean),
            }
            rows.append(row)
            preds.append(p)
            truths.append(float(ex.expert_mean))
            fp.write(json.dumps(row) + "\n")
            fp.flush()
            if (len(rows) % 10) == 0:
                logger.info("[%s] scored %d/%d (%.1fs)", label, len(rows), len(examples),
                            time.time() - t0)
    report = compute_corpus_pearson_r(preds, truths)
    logger.info(
        "[%s] r=%+.3f n=%d CI[%+.3f,%+.3f] elapsed=%.1fs",
        label, report.pearson_r, report.n, report.pearson_ci_low, report.pearson_ci_high,
        time.time() - t0,
    )
    payload = report.as_dict()
    payload["elapsed_seconds"] = round(time.time() - t0, 1)
    payload["prediction_path"] = str(out_path)
    payload["n_examples_requested"] = len(examples)
    return payload


def _fetch_vllm_prefix_metrics(ports: list[int]) -> dict[str, Any]:
    """Best-effort scrape of vLLM /metrics prefix-cache counters."""
    metrics: dict[str, Any] = {}
    for port in ports:
        port_metrics: dict[str, float] = {}
        try:
            with urlopen(f"http://localhost:{int(port)}/metrics", timeout=2.0) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            metrics[str(port)] = {"error": str(exc)}
            continue
        for line in text.splitlines():
            if not line or line.startswith("#") or "prefix" not in line.lower():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0]
            try:
                value = float(parts[-1])
            except ValueError:
                continue
            port_metrics[name] = value
        queries = sum(
            value for name, value in port_metrics.items()
            if name.startswith("vllm:prefix_cache_queries_total")
        )
        hits = sum(
            value for name, value in port_metrics.items()
            if name.startswith("vllm:prefix_cache_hits_total")
        )
        if queries > 0:
            port_metrics["prefix_cache_hit_rate"] = hits / queries
        metrics[str(port)] = port_metrics
    return metrics


def _existing(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path)
    return path if path.exists() else None


def _first_existing(base: Path, names: tuple[str, ...]) -> Optional[Path]:
    for name in names:
        path = _existing(base / name)
        if path is not None:
            return path
    return None


def _resolve_init_paths(args: argparse.Namespace) -> dict[str, Optional[Path]]:
    init_dir = Path(args.init_dir) if args.init_dir is not None else None
    artifact_kind = getattr(args, "init_artifact_kind", "final")
    components_only = bool(getattr(args, "init_components_only", False))
    if artifact_kind not in {"final", "optimized"}:
        raise ValueError(f"unknown init artifact kind: {artifact_kind!r}")
    paths: dict[str, Optional[Path]] = {
        "program": args.init_program,
        "scorer": args.init_scorer,
        "g": args.init_g,
        "g_legacy_leaf": args.init_g_legacy_leaf,
    }
    if init_dir is not None:
        if artifact_kind == "optimized":
            program_candidates = ("optimized_program.json", "final_program.json")
            scorer_candidates = ("optimized_scorer.json", "scorer_final.json")
            g_candidates = ("optimized_unified_g.json", "unified_g_final.json", "g_final.json")
        else:
            program_candidates = ("final_program.json", "optimized_program.json")
            scorer_candidates = ("scorer_final.json", "optimized_scorer.json")
            g_candidates = ("unified_g_final.json", "g_final.json", "optimized_unified_g.json")
        if not components_only:
            paths["program"] = paths["program"] or _first_existing(init_dir, program_candidates)
        paths["scorer"] = paths["scorer"] or _first_existing(init_dir, scorer_candidates)
        paths["g"] = paths["g"] or _first_existing(init_dir, g_candidates)
        if paths["g"] is None:
            paths["g_legacy_leaf"] = (
                paths["g_legacy_leaf"]
                or _existing(init_dir / "leaf_summarizer_final.json")
            )
    return paths


def _load_scorer_component(pipeline: DimensionFullPipeline, path: Path) -> None:
    data = json.loads(Path(path).read_text())
    target = getattr(pipeline.scorer, "_module", pipeline.scorer)
    if "scorer" in data:
        target.load_state(data["scorer"])
        return
    if "score" in data:
        from src.tasks.manifesto.dimension_scorer import DimensionScorer
        temp = DimensionScorer(pipeline.spec)
        temp.score.load_state(data["score"])
        if hasattr(pipeline.scorer, "_module"):
            pipeline.scorer._module = temp.score
        else:
            pipeline.scorer = temp.score
        return
    if "scorer.score" in data:
        target.load_state(data["scorer.score"])
        return
    target.load(str(path))


def _load_g_component(pipeline: DimensionFullPipeline, path: Path) -> None:
    data = json.loads(Path(path).read_text())
    target = getattr(pipeline.g, "_module", pipeline.g)
    if "g.summarize" in data:
        target.summarize.load_state(data["g.summarize"])
        return
    if "summarize" in data:
        target.summarize.load_state(data["summarize"])
        return
    target.load(str(path))


def _signature_instruction(state: dict[str, Any]) -> Optional[str]:
    signature = state.get("signature") if isinstance(state, dict) else None
    if isinstance(signature, dict):
        instruction = signature.get("instructions")
        if isinstance(instruction, str) and instruction.strip():
            return instruction
    return None


def _extract_legacy_g_instruction(data: dict[str, Any]) -> str:
    """Find a learned summarizer instruction without importing old signatures."""
    for key in (
        "summarize",
        "g.summarize",
        "leaf.summarize",
        "leaf_summarizer",
        "leaf_summarizer.summarize",
    ):
        value = data.get(key)
        if isinstance(value, dict):
            instruction = _signature_instruction(value)
            if instruction:
                return instruction
    instruction = _signature_instruction(data)
    if instruction:
        return instruction

    def walk(obj: Any) -> Optional[str]:
        if isinstance(obj, dict):
            found = _signature_instruction(obj)
            if found:
                return found
            for child in obj.values():
                found = walk(child)
                if found:
                    return found
        elif isinstance(obj, list):
            for child in obj:
                found = walk(child)
                if found:
                    return found
        return None

    instruction = walk(data)
    if instruction:
        return instruction
    raise ValueError("No signature.instructions field found in legacy g artifact")


def _unwrap_module(module):
    return getattr(module, "_module", module)


def _load_g_legacy_leaf_instruction(pipeline: DimensionFullPipeline, path: Path) -> None:
    data = json.loads(Path(path).read_text())
    instruction = _extract_legacy_g_instruction(data)
    target = _unwrap_module(pipeline.g)
    if not hasattr(target, "summarize"):
        raise TypeError("Pipeline g does not expose a summarize predictor")
    state = target.summarize.dump_state()
    state.setdefault("signature", {})["instructions"] = instruction
    target.summarize.load_state(state)


def _warm_start_pipeline(
    pipeline: DimensionFullPipeline,
    *,
    init_program: Optional[Path],
    init_scorer: Optional[Path],
    init_g: Optional[Path],
    init_g_legacy_leaf: Optional[Path],
) -> dict[str, str]:
    loaded: dict[str, str] = {}
    if init_program is not None:
        if not init_program.exists():
            raise FileNotFoundError(f"--init-program not found: {init_program}")
        pipeline.load(str(init_program))
        loaded["program"] = str(init_program)
    if init_scorer is not None:
        if not init_scorer.exists():
            raise FileNotFoundError(f"--init-scorer not found: {init_scorer}")
        _load_scorer_component(pipeline, init_scorer)
        loaded["scorer"] = str(init_scorer)
    if init_g is not None:
        if not init_g.exists():
            raise FileNotFoundError(f"--init-g not found: {init_g}")
        _load_g_component(pipeline, init_g)
        loaded["g"] = str(init_g)
    if init_g_legacy_leaf is not None:
        if not init_g_legacy_leaf.exists():
            raise FileNotFoundError(f"--init-g-legacy-leaf not found: {init_g_legacy_leaf}")
        _load_g_legacy_leaf_instruction(pipeline, init_g_legacy_leaf)
        loaded["g_legacy_leaf_instruction"] = str(init_g_legacy_leaf)
    return loaded


def _save_component_artifacts(
    pipeline: DimensionFullPipeline,
    output_dir: Path,
    *,
    kind: str,
) -> dict[str, str]:
    """Persist full program plus separately reusable f and unified-g artifacts."""
    if kind == "optimized":
        names = {
            "program": "optimized_program.json",
            "scorer": "optimized_scorer.json",
            "g": "optimized_unified_g.json",
        }
    elif kind == "final":
        names = {
            "program": "final_program.json",
            "scorer": "scorer_final.json",
            "g": "unified_g_final.json",
        }
    else:
        raise ValueError(f"unknown artifact kind: {kind}")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for label, module in (
        ("program", pipeline),
        ("scorer", _unwrap_module(pipeline.scorer)),
        ("g", _unwrap_module(pipeline.g)),
    ):
        path = output_dir / names[label]
        try:
            module.save(str(path))
            saved[label] = str(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save %s %s artifact to %s: %s", kind, label, path, exc)
    return saved


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dim = _DIM_FROM_NAME[args.dimension]
    lm_kwargs = {"model": args.model, "temperature": 0.0, "cache": True}
    if args.max_tokens is not None:
        lm_kwargs["max_tokens"] = args.max_tokens
    if args.ports:
        logger.info("LM: load-balanced across %s", args.ports)
        lm = create_vllm_lm_multi(ports=args.ports, **lm_kwargs)
    else:
        lm = create_vllm_lm(port=args.port, **lm_kwargs)
    configure_dspy(lm=lm)

    logger.info("Building examples for dimension=%s", dim.value)
    trainset, devset, testset = _build_examples(
        dim, args.train_pool, args.mp_data_dir,
        args.train_n, args.dev_n, args.test_n, args.seed,
        args.split_strategy,
    )
    logger.info("Split: train=%d dev=%d test=%d", len(trainset), len(devset), len(testset))
    if args.cheat_train_on_test:
        if not testset:
            raise SystemExit("--cheat-train-on-test requires non-empty testset")
        trainset = testset[: max(1, min(len(testset), int(args.train_n)))]
        devset = testset[: max(1, min(len(testset), int(args.dev_n)))]
        logger.warning(
            "CHEAT DIAGNOSTIC: using test examples as GEPA train/dev "
            "(train=%d dev=%d test=%d). Do not use for paper claims.",
            len(trainset),
            len(devset),
            len(testset),
        )

    baseline_pipeline = DimensionFullPipeline(
        dim,
        chunk_chars=args.chunk_chars,
        max_workers=args.max_workers,
        optimize_scope=args.optimize_scope,
    )
    gepa_optimizes_g = args.optimizer == "gepa" and args.optimize_scope in {"g", "gf"}
    student_node_cache_enabled = not (
        args.optimizer == "gepa" and args.optimize_scope in {"g", "gf"}
    )
    if not student_node_cache_enabled:
        logger.info(
            "Disabling student node cache during GEPA because optimize_scope=%s includes g; "
            "GEPA needs visible g predictor traces for reflection.",
            args.optimize_scope,
        )
    student_max_workers = 1 if gepa_optimizes_g else args.max_workers
    if student_max_workers != args.max_workers:
        logger.info(
            "Using student max_workers=1 during GEPA because g calls must run "
            "on the tracing thread."
        )
    student_pipeline = DimensionFullPipeline(
        dim,
        chunk_chars=args.chunk_chars,
        max_workers=student_max_workers,
        optimize_scope=args.optimize_scope,
        enable_node_cache=student_node_cache_enabled,
    )
    init_paths = _resolve_init_paths(args)
    baseline_loaded = _warm_start_pipeline(
        baseline_pipeline,
        init_program=init_paths["program"],
        init_scorer=init_paths["scorer"],
        init_g=init_paths["g"],
        init_g_legacy_leaf=init_paths["g_legacy_leaf"],
    )
    student_loaded = _warm_start_pipeline(
        student_pipeline,
        init_program=init_paths["program"],
        init_scorer=init_paths["scorer"],
        init_g=init_paths["g"],
        init_g_legacy_leaf=init_paths["g_legacy_leaf"],
    )
    if baseline_loaded or student_loaded:
        logger.info("Warm-started baseline/student with artifacts: %s", baseline_loaded or student_loaded)

    metric = _make_metric(args.metric_mode)

    logger.info("Evaluating baseline on dev (n=%d)", len(devset))
    baseline_dev = _evaluate(baseline_pipeline, devset, "baseline_dev", args.output_dir, dim.value)

    out = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": getattr(lm, "model", "unknown"),
            "dimension": dim.value,
            "optimizer": args.optimizer,
            "optimize_scope": args.optimize_scope,
            "metric_mode": args.metric_mode,
            "feedback_mode": args.feedback_mode,
            "selection_guard": args.selection_guard,
            "init_dir": str(args.init_dir) if args.init_dir else None,
            "init_artifact_kind": args.init_artifact_kind,
            "init_paths_loaded": baseline_loaded,
            "cheat_train_on_test": bool(args.cheat_train_on_test),
            "gepa_auto": args.gepa_auto,
            "gepa_threads": args.gepa_threads,
            "gepa_valset_cap": args.gepa_valset_cap,
            "gepa_max_metric_calls": args.gepa_max_metric_calls,
            "reflection_max_tokens": args.reflection_max_tokens,
            "chunk_chars": args.chunk_chars,
            "train_pool": args.train_pool,
            "split_strategy": args.split_strategy,
            "n_train": len(trainset), "n_dev": len(devset), "n_test": len(testset),
            "max_demos": args.max_demos,
            "max_tokens": args.max_tokens,
            "student_node_cache_enabled": student_node_cache_enabled,
            "student_max_workers": student_max_workers,
            "seed": args.seed,
        },
        "baseline_dev": baseline_dev,
        "dev_selection_guard_triggered": False,
        "baseline_guard_triggered": False,
    }

    selected_program: dspy.Module = baseline_pipeline
    selected_label = "baseline"
    compile_seconds = 0.0

    if args.optimizer != "none":
        if args.optimizer == "bootstrap":
            compiler = dspy.BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=args.max_demos,
                max_labeled_demos=args.max_demos,
            )
            logger.info("Compiling BootstrapFewShot(max_demos=%d)", args.max_demos)
            t0 = time.time()
            optimized = compiler.compile(student_pipeline, trainset=trainset)
        elif args.optimizer == "miprov2":
            compiler = dspy.MIPROv2(metric=metric, auto="light", num_threads=4)
            logger.info("Compiling MIPROv2(auto=light)")
            t0 = time.time()
            optimized = compiler.compile(student_pipeline, trainset=trainset, valset=devset)
        elif args.optimizer == "gepa":
            reflection_kwargs = {
                "model": args.model,
                "temperature": 0.7,
                "cache": True,
                "max_tokens": int(args.reflection_max_tokens),
            }
            reflection_lm = (
                create_vllm_lm_multi(ports=args.ports, **reflection_kwargs)
                if args.ports
                else create_vllm_lm(port=args.port, **reflection_kwargs)
            )
            gepa_kwargs: dict[str, Any] = {
                "metric": _make_gepa_metric(
                    BENOIT_DIMENSIONS[dim],
                    mode=args.metric_mode,
                    feedback_mode=args.feedback_mode,
                ),
                "reflection_lm": reflection_lm,
                "num_threads": args.gepa_threads,
                "track_stats": True,
            }
            if args.gepa_max_metric_calls > 0:
                gepa_kwargs["max_metric_calls"] = int(args.gepa_max_metric_calls)
            else:
                gepa_kwargs["auto"] = args.gepa_auto
            compiler = dspy.GEPA(**gepa_kwargs)
            gepa_valset = devset[: int(args.gepa_valset_cap)] if args.gepa_valset_cap > 0 else devset
            logger.info(
                "Compiling GEPA(scope=%s, valset=%d, max_metric_calls=%s, auto=%s)",
                args.optimize_scope,
                len(gepa_valset),
                args.gepa_max_metric_calls if args.gepa_max_metric_calls > 0 else None,
                None if args.gepa_max_metric_calls > 0 else args.gepa_auto,
            )
            t0 = time.time()
            optimized = compiler.compile(student=student_pipeline, trainset=trainset, valset=gepa_valset)
        else:
            raise ValueError(args.optimizer)
        compile_seconds = round(time.time() - t0, 1)
        logger.info("Compile done in %.1fs", compile_seconds)

        logger.info("Evaluating optimized on dev")
        optimized_dev = _evaluate(optimized, devset, "optimized_dev", args.output_dir, dim.value)
        out["optimized_dev"] = optimized_dev
        base_dev_r = baseline_dev.get("pearson_r")
        opt_dev_r = optimized_dev.get("pearson_r")
        base_dev_score = float(base_dev_r) if base_dev_r is not None else float("-inf")
        opt_dev_score = float(opt_dev_r) if opt_dev_r is not None else float("-inf")
        if args.selection_guard == "dev" and opt_dev_score < base_dev_score:
            selected_program = baseline_pipeline
            selected_label = "baseline"
            out["dev_selection_guard_triggered"] = True
            logger.info(
                "Dev selection kept baseline: optimized_dev=%s baseline_dev=%s",
                opt_dev_r,
                base_dev_r,
            )
        else:
            selected_program = optimized
            selected_label = "optimized"
            logger.info(
                "Dev selection chose optimized: optimized_dev=%s baseline_dev=%s",
                opt_dev_r,
                base_dev_r,
            )
        out["compile_time_seconds"] = compile_seconds
        out["optimized_artifacts"] = _save_component_artifacts(
            optimized,
            args.output_dir,
            kind="optimized",
        )

    logger.info("Evaluating dev-selected %s program on test (n=%d)", selected_label, len(testset))
    final_test = _evaluate(selected_program, testset, "final_test", args.output_dir, dim.value)
    out["selection"] = {"selected": selected_label, "criterion": args.selection_guard}
    out["final_test"] = final_test
    out["final_artifacts"] = _save_component_artifacts(
        selected_program,
        args.output_dir,
        kind="final",
    )
    prediction_paths = {
        key: value.get("prediction_path")
        for key, value in (
            ("baseline_dev", out.get("baseline_dev", {})),
            ("optimized_dev", out.get("optimized_dev", {})),
            ("final_test", out.get("final_test", {})),
        )
        if isinstance(value, dict) and value.get("prediction_path")
    }
    out["prediction_paths"] = prediction_paths
    out["canonical_outputs"] = {
        "program": out.get("final_artifacts", {}).get("program"),
        "scorer_f": out.get("final_artifacts", {}).get("scorer"),
        "unified_g": out.get("final_artifacts", {}).get("g"),
        "dev_predictions": (
            prediction_paths.get("optimized_dev")
            if selected_label == "optimized"
            else prediction_paths.get("baseline_dev")
        ),
        "test_predictions": prediction_paths.get("final_test"),
        "optimized_program": out.get("optimized_artifacts", {}).get("program"),
        "optimized_scorer_f": out.get("optimized_artifacts", {}).get("scorer"),
        "optimized_unified_g": out.get("optimized_artifacts", {}).get("g"),
        "optimized_dev_predictions": prediction_paths.get("optimized_dev"),
    }
    if hasattr(selected_program, "cache_stats"):
        out["selected_program_cache_stats"] = selected_program.cache_stats()
    if hasattr(baseline_pipeline, "cache_stats"):
        out["baseline_cache_stats"] = baseline_pipeline.cache_stats()
    ports_for_metrics = args.ports or [args.port]
    out["vllm_prefix_metrics"] = _fetch_vllm_prefix_metrics([int(p) for p in ports_for_metrics])

    (args.output_dir / "report.json").write_text(json.dumps(out, indent=2))
    logger.info("Outputs in %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
