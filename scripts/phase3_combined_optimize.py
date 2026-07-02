#!/usr/bin/env python3
"""
Phase 3 combined: full-pipeline GEPA / BFS with ONE shared unified g
(JOINT_RUBRIC) and ONE shared scorer f switching task_context per dimension.

Trains the whole tree + score module on pooled (text, dim, label)
examples across all 6 dims. Evaluates per-dim on held-out Benoit
expert-benchmark manifestos.

Usage:
    python scripts/phase3_combined_optimize.py \\
        --ports 8010 8011 8012 8013 \\
        --optimizer gepa --gepa-auto light \\
        --train-n 18 --dev-n 6 --test-n 30 \\
        --chunk-chars 24000 \\
        --output-dir outputs/phase3/combined_gepa
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlopen

import dspy

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config.dspy_config import configure_dspy, create_local_engine_lm
from src.config.local_inference import resolve_local_inference_config
from src.core.protocols import format_merge_input
from src.preprocessing.chunker import chunk_for_ops
from src.tasks.manifesto import ManifestoDataset
from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r
from src.tasks.manifesto.dimensions import (
    BENOIT_DIMENSIONS, DimensionSpec, PolicyDimension, get_joint_rubric,
)
from src.tasks.manifesto.dimension_scorer import DimensionScoreSignature
from src.tasks.manifesto.expert_benchmarks import (
    load_benoit_expert_means, load_benoit_mp_crosswalk,
)
from src.tasks.manifesto.pipeline import UnifiedManifestoG
from src.tasks.manifesto.pipeline_config import DEFAULT_SCORER_MAX_TOKENS
from src.tasks.manifesto.resume_utils import load_resume_rows
from src.tasks.manifesto.scoring_contexts import get_scoring_context
from src.core.prompting import parse_numeric_score

logger = logging.getLogger(__name__)


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


class CombinedFullPipeline(dspy.Module):
    """Full tree pipeline with JOINT_RUBRIC unified g + dim-switching scorer.

    Optimizer scope is exactly one of f, g, or gf. The scorer's task_context
    is supplied per-call from the `dimension_spec`.
    """

    def __init__(
        self,
        *,
        chunk_chars: int = 24000,
        max_workers: int = 8,
        optimize_scope: str = "gf",
        enable_node_cache: bool = True,
    ):
        super().__init__()
        if optimize_scope not in {"f", "g", "gf"}:
            raise ValueError(f"optimize_scope must be one of f/g/gf, got {optimize_scope!r}")
        self.chunk_chars = chunk_chars
        self.max_workers = max_workers
        self.optimize_scope = optimize_scope
        self.rubric = get_joint_rubric()
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

    def forward(self, text: str, dimension_spec: DimensionSpec) -> dspy.Prediction:
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
        ctx = get_scoring_context(dimension_spec.dimension)
        scored = self.scorer(
            task_context=ctx,
            summary=final_summary,
            config={"max_tokens": self.scorer_max_tokens},
        )

        raw_str = str(getattr(scored, "score", ""))
        if raw_str.strip().lower() in {"na", "n/a", "none", ""}:
            return dspy.Prediction(score=None, summary=final_summary,
                                   reasoning=getattr(scored, "reasoning", ""))
        raw = parse_numeric_score(
            raw_str, min_value=dimension_spec.scale.min_value,
            max_value=dimension_spec.scale.max_value, allow_llm_fallback=True,
        )
        if raw is None:
            return dspy.Prediction(score=None, summary=final_summary,
                                   reasoning=getattr(scored, "reasoning", ""))
        return dspy.Prediction(
            score=dimension_spec.scale.clamp(float(raw)),
            summary=final_summary,
            reasoning=getattr(scored, "reasoning", ""),
        )


_DIMS = [
    PolicyDimension.ECONOMIC, PolicyDimension.SOCIAL, PolicyDimension.IMMIGRATION,
    PolicyDimension.EU, PolicyDimension.ENVIRONMENT, PolicyDimension.DECENTRALIZATION,
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--ports", type=int, nargs="+", default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--optimizer", choices=["bootstrap", "gepa", "none"], default="gepa")
    p.add_argument("--optimize-scope", choices=["f", "g", "gf"], default="gf")
    p.add_argument("--metric-mode", choices=["mae", "rank"], default="rank")
    p.add_argument("--feedback-mode", choices=["scalar", "rich"], default="rich")
    p.add_argument("--gepa-auto", choices=["light", "medium", "heavy"], default="light")
    p.add_argument("--gepa-threads", type=int, default=4)
    p.add_argument("--gepa-valset-cap", type=int, default=4)
    p.add_argument("--gepa-max-metric-calls", type=int, default=0)
    p.add_argument("--selection-guard", choices=["none", "dev"], default="none")
    p.add_argument("--reflection-max-tokens", type=int, default=2048)
    p.add_argument("--init-program", type=Path, default=None,
                   help="Optional saved CombinedFullPipeline program JSON to warm-start before optimization.")
    p.add_argument("--init-dir", type=Path, default=None,
                   help="Optional artifact directory; loads compatible optimized_program/optimized_scorer/unified_g files if present.")
    p.add_argument("--init-scorer", type=Path, default=None,
                   help="Optional scorer-only artifact JSON to warm-start f.")
    p.add_argument("--init-g", type=Path, default=None,
                   help="Optional unified-g artifact JSON to warm-start g.")
    p.add_argument("--init-g-legacy-leaf", type=Path, default=None,
                   help="Optional legacy LeafSummarizer artifact; transplants only "
                        "the learned instruction text into unified g.")
    p.add_argument("--cheat-train-on-test", action="store_true",
                   help="Diagnostic only: use held-out test examples as GEPA train/dev to test in-sample overfit.")
    p.add_argument("--train-n", type=int, default=18)
    p.add_argument("--dev-n", type=int, default=6)
    p.add_argument("--test-n", type=int, default=30)
    p.add_argument("--chunk-chars", type=int, default=24000)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--max-demos", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    env_cap = os.environ.get("MANIFESTO_MAX_TOKENS")
    p.add_argument("--max-tokens", type=int, default=int(env_cap) if env_cap else None)
    p.add_argument("--mp-data-dir", type=Path,
                   default=project_root / "data" / "raw" / "manifesto_corpus_benoit")
    p.add_argument("--output-dir", type=Path,
                   default=project_root / "outputs" / "phase3" /
                   f"combined_gepa_{datetime.now(timezone.utc):%Y%m%d_%H%M}")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _build_pooled(mp_data_dir: Path, train_n: int, dev_n: int, test_n: int, seed: int):
    """Pooled (text, dim, label) examples across 6 dims.

    Train labels: Benoit's expert ensemble mean from data_experts (the same
    signal that will be used for test — but with strict manifesto-level
    disjointness). Train and test manifestos do not overlap (global holdout).
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

    rng = random.Random(seed)
    # Build the set of manifestos with ANY expert label (across all 6 dims)
    records_by_mfesto: dict[str, dict] = {}
    for dim in _DIMS:
        experts = load_benoit_expert_means(dim)
        for row in experts.itertuples():
            bkey = str(row.manifesto)
            key = benoit_to_py.get(bkey)
            if key is None:
                continue
            mid = py_to_mid.get(key)
            if mid is None:
                continue
            s = ds.get_sample(mid)
            if s is None or not s.text:
                continue
            r = records_by_mfesto.setdefault(
                bkey, {"manifesto_id": mid, "benoit_key": bkey, "text": s.text, "labels": {}}
            )
            r["labels"][dim.value] = float(row.expert_mean_1_7)

    all_recs = list(records_by_mfesto.values())
    rng.shuffle(all_recs)
    needed = train_n + dev_n
    train_dev = all_recs[:needed]
    test_recs = all_recs[needed : needed + test_n]

    def expand(mfesto_recs):
        out = []
        for r in mfesto_recs:
            for dim in _DIMS:
                lab = r["labels"].get(dim.value)
                if lab is None:
                    continue
                out.append(dspy.Example(
                    text=r["text"],
                    dimension_spec=BENOIT_DIMENSIONS[dim],
                    dim_value=dim.value,
                    expert_mean=lab,
                    manifesto_id=r["manifesto_id"],
                ).with_inputs("text", "dimension_spec"))
        return out

    train_exs = expand(train_dev[:train_n])
    dev_exs = expand(train_dev[train_n : train_n + dev_n])
    test_exs = expand(test_recs)
    return train_exs, dev_exs, test_exs


def _score_from_prediction(prediction) -> Optional[float]:
    raw = getattr(prediction, "score", None)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
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
        side_penalty = 0.25 if (pred_score >= center) != (target >= center) else 0.0
        return max(0.0, mae_score - side_penalty)
    raise ValueError(f"unknown metric mode: {mode}")


def _make_metric(mode: str):
    def metric(example, prediction, trace=None):
        return _metric(example, prediction, trace=trace, mode=mode)
    return metric


def _make_gepa_metric(*, mode: str, feedback_mode: str):
    scalar_metric = _make_metric(mode)
    if feedback_mode == "scalar":
        def scalar_gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            del pred_name, pred_trace
            return scalar_metric(gold, pred, trace=trace)
        return scalar_gepa_metric

    from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

    def rich_gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        del trace, pred_name, pred_trace
        spec = gold.dimension_spec
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
                f"Dimension: {spec.dimension.value}. Predicted score: {pred_score:.3f}. "
                f"Target: {target:.3f}. Absolute error: {abs(error):.3f}. "
                f"Direction of correction: {direction}. Rank-side check: prediction is "
                f"{pred_side}; target is {target_side}. Scale anchors: {anchors}. "
                "Use the summary evidence and dimension rubric; avoid generic party stereotypes.\n"
                f"Model reasoning (truncated): {reasoning[:800] or '(empty)'}\n"
                f"Final summary excerpt (truncated): {summary[:800] or '(empty)'}"
            )
        return ScoreWithFeedback(score=score, feedback=feedback)

    return rich_gepa_metric


def _predict(program, ex) -> Optional[float]:
    try:
        pred = program(text=ex.text, dimension_spec=ex.dimension_spec)
    except Exception as e:  # noqa: BLE001
        logger.warning("pred failed: %s", e)
        return None
    return _score_from_prediction(pred)


def _per_dim_eval(program, examples, label: str, output_dir: Path) -> dict:
    t0 = time.time()
    out_path = output_dir / f"per_dim_{label}.jsonl"
    # key = manifesto_id + dimension (one example per (mfesto, dim) pair)
    already_raw, resuming = load_resume_rows(out_path, log_label=label)
    # Reindex by compound (mid|dim) key
    already: dict[str, dict] = {}
    for row in already_raw.values():
        mid = row.get("manifesto_id")
        dv = row.get("dimension")
        if mid is not None and dv is not None:
            already[f"{mid}|{dv}"] = row

    rows: list[dict] = list(already.values())
    by_dim: dict[str, dict] = {d.value: {"preds": [], "truths": []} for d in _DIMS}
    for row in rows:
        dv = row.get("dimension")
        if dv in by_dim:
            by_dim[dv]["preds"].append(row.get("pred"))
            by_dim[dv]["truths"].append(float(row["expert_mean"]))
    with out_path.open("a" if resuming else "w") as fp:
        for i, ex in enumerate(examples):
            mid = getattr(ex, "manifesto_id", None)
            dv = ex.dim_value
            ck = f"{mid}|{dv}"
            if mid is not None and ck in already:
                continue
            p = _predict(program, ex)
            row = {"phase": label, "dimension": dv,
                   "manifesto_id": mid,
                   "pred": p, "expert_mean": float(ex.expert_mean)}
            rows.append(row)
            by_dim[dv]["preds"].append(p)
            by_dim[dv]["truths"].append(float(ex.expert_mean))
            fp.write(json.dumps(row) + "\n")
            fp.flush()
            if (len(rows) % 20) == 0:
                logger.info("[%s] %d/%d (%.0fs)", label, len(rows), len(examples),
                            time.time() - t0)

    per_dim = {}
    for dim_v, bundle in by_dim.items():
        if len(bundle["preds"]) < 3:
            per_dim[dim_v] = {"n": len(bundle["preds"]), "pearson_r": None}
            continue
        rep = compute_corpus_pearson_r(bundle["preds"], bundle["truths"])
        per_dim[dim_v] = rep.as_dict()
        logger.info("[%s] %-18s r=%+.3f n=%d", label, dim_v, rep.pearson_r, rep.n)

    macros = [v["pearson_r"] for v in per_dim.values() if v.get("pearson_r") is not None]
    macro = sum(macros) / len(macros) if macros else None
    return {"per_dim": per_dim, "macro_pearson_r": macro, "elapsed_seconds": round(time.time() - t0, 1)}


def _fetch_vllm_prefix_metrics(ports: list[int]) -> dict[str, Any]:
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
            try:
                value = float(parts[-1])
            except ValueError:
                continue
            port_metrics[parts[0]] = value
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


def _resolve_init_paths(args: argparse.Namespace) -> dict[str, Optional[Path]]:
    init_dir = Path(args.init_dir) if args.init_dir is not None else None
    paths: dict[str, Optional[Path]] = {
        "program": args.init_program,
        "scorer": args.init_scorer,
        "g": args.init_g,
        "g_legacy_leaf": args.init_g_legacy_leaf,
    }
    if init_dir is not None:
        paths["program"] = (
            paths["program"]
            or _existing(init_dir / "final_program.json")
            or _existing(init_dir / "optimized_program.json")
        )
        paths["scorer"] = (
            paths["scorer"]
            or _existing(init_dir / "scorer_final.json")
            or _existing(init_dir / "optimized_scorer.json")
        )
        paths["g"] = (
            paths["g"]
            or _existing(init_dir / "unified_g_final.json")
            or _existing(init_dir / "g_final.json")
            or _existing(init_dir / "optimized_unified_g.json")
        )
        if paths["g"] is None:
            paths["g_legacy_leaf"] = (
                paths["g_legacy_leaf"]
                or _existing(init_dir / "leaf_summarizer_final.json")
            )
    return paths


def _load_scorer_component(pipeline: CombinedFullPipeline, path: Path) -> None:
    data = json.loads(Path(path).read_text())
    target = getattr(pipeline.scorer, "_module", pipeline.scorer)
    if "scorer" in data:
        target.load_state(data["scorer"])
        return
    if "score" in data:
        from src.tasks.manifesto.dimension_scorer import DimensionScorer
        temp = DimensionScorer(BENOIT_DIMENSIONS[PolicyDimension.ECONOMIC])
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


def _load_g_component(pipeline: CombinedFullPipeline, path: Path) -> None:
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


def _load_g_legacy_leaf_instruction(pipeline: CombinedFullPipeline, path: Path) -> None:
    data = json.loads(Path(path).read_text())
    instruction = _extract_legacy_g_instruction(data)
    target = _unwrap_module(pipeline.g)
    if not hasattr(target, "summarize"):
        raise TypeError("Pipeline g does not expose a summarize predictor")
    state = target.summarize.dump_state()
    state.setdefault("signature", {})["instructions"] = instruction
    target.summarize.load_state(state)


def _warm_start_pipeline(
    pipeline: CombinedFullPipeline,
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
    pipeline: CombinedFullPipeline,
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

    local_inference = resolve_local_inference_config({**vars(args), "temperature": 0.0})
    lm = create_local_engine_lm(**local_inference.dspy_kwargs(cache=True))
    configure_dspy(lm=lm)

    logger.info("Building pooled examples (6 dims)")
    trainset, devset, testset = _build_pooled(
        args.mp_data_dir, args.train_n, args.dev_n, args.test_n, args.seed,
    )
    logger.info("Sizes: train=%d dev=%d test=%d", len(trainset), len(devset), len(testset))
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

    baseline_pipeline = CombinedFullPipeline(
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
    student_pipeline = CombinedFullPipeline(
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

    logger.info("Evaluating baseline on dev")
    baseline_dev = _per_dim_eval(baseline_pipeline, devset, "baseline_dev", args.output_dir)
    logger.info("Baseline dev macro r=%+.3f", baseline_dev["macro_pearson_r"] or float("nan"))

    out = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": getattr(lm, "model", "unknown"),
            "optimizer": args.optimizer,
            "optimize_scope": args.optimize_scope,
            "metric_mode": args.metric_mode,
            "feedback_mode": args.feedback_mode,
            "selection_guard": args.selection_guard,
            "init_dir": str(args.init_dir) if args.init_dir else None,
            "init_paths_loaded": baseline_loaded,
            "cheat_train_on_test": bool(args.cheat_train_on_test),
            "chunk_chars": args.chunk_chars,
            "n_train": len(trainset), "n_dev": len(devset), "n_test": len(testset),
            "gepa_auto": args.gepa_auto,
            "gepa_threads": args.gepa_threads,
            "gepa_valset_cap": args.gepa_valset_cap,
            "gepa_max_metric_calls": args.gepa_max_metric_calls,
            "reflection_max_tokens": args.reflection_max_tokens,
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
            logger.info("Compiling BootstrapFewShot")
            t0 = time.time()
            optimized = compiler.compile(student_pipeline, trainset=trainset)
        elif args.optimizer == "gepa":
            reflection_kwargs = {
                "model": args.model,
                "temperature": 0.7,
                "cache": True,
                "max_tokens": int(args.reflection_max_tokens),
            }
            reflection_lm = create_local_engine_lm(
                engine=local_inference.engine,
                endpoints=local_inference.endpoints,
                **reflection_kwargs,
            )
            gepa_kwargs: dict[str, Any] = {
                "metric": _make_gepa_metric(mode=args.metric_mode, feedback_mode=args.feedback_mode),
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

        optimized_dev = _per_dim_eval(optimized, devset, "optimized_dev", args.output_dir)
        logger.info("Optimized dev macro r=%+.3f", optimized_dev["macro_pearson_r"] or float("nan"))
        out["optimized_dev"] = optimized_dev
        base_dev_r = baseline_dev.get("macro_pearson_r")
        opt_dev_r = optimized_dev.get("macro_pearson_r")
        base_dev_score = float(base_dev_r) if base_dev_r is not None else float("-inf")
        opt_dev_score = float(opt_dev_r) if opt_dev_r is not None else float("-inf")
        if args.selection_guard == "dev" and opt_dev_score < base_dev_score:
            selected_program = baseline_pipeline
            selected_label = "baseline"
            out["dev_selection_guard_triggered"] = True
            logger.info("Dev selection kept baseline: optimized=%s baseline=%s", opt_dev_r, base_dev_r)
        else:
            selected_program = optimized
            selected_label = "optimized"
            logger.info("Dev selection chose optimized: optimized=%s baseline=%s", opt_dev_r, base_dev_r)
        out["compile_time_seconds"] = compile_seconds
        out["optimized_artifacts"] = _save_component_artifacts(
            optimized,
            args.output_dir,
            kind="optimized",
        )

    logger.info("Evaluating dev-selected %s program on test", selected_label)
    final_test = _per_dim_eval(selected_program, testset, "final_test", args.output_dir)
    logger.info("Final macro r=%+.3f", final_test["macro_pearson_r"] or float("nan"))
    out["selection"] = {"selected": selected_label, "criterion": args.selection_guard}
    out["final_test"] = final_test
    out["final_artifacts"] = _save_component_artifacts(
        selected_program,
        args.output_dir,
        kind="final",
    )
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
