#!/usr/bin/env python3
"""Train/evaluate sampled-label q-sentence g from a fixed trained f.

This runner answers a different question than the alternating q-sentence
ladder: given a random sample of *gold-labeled* leaf-window states from each manifesto,
can a learned g produce a compact full-document state such that a fixed learned
f matches the full-document gold compact targets? This is a sampled-root
prediction runner, not the uniform all-node IPW local-law objective.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import importlib.metadata as _importlib_metadata
import json
import logging
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


def _safe_packages_distributions():
    pkg_to_dist = collections.defaultdict(list)
    for dist in _importlib_metadata.distributions():
        try:
            md = dist.metadata
            if md is None:
                continue
            name = md["Name"]
        except Exception:
            continue
        for top in (
            _importlib_metadata._top_level_declared(dist)
            or _importlib_metadata._top_level_inferred(dist)
        ):
            pkg_to_dist[top].append(name)
    return dict(pkg_to_dist)


_importlib_metadata.packages_distributions = _safe_packages_distributions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.batch_transport import (  # noqa: E402
    DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_BATCH_ROUTING_POLICY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_TIMEOUT_SECONDS,
)
from src.ctreepo.distillation import load_labeled_trees  # noqa: E402
from src.ctreepo.manifesto_qsentence_runner import leafq_dir  # noqa: E402
from src.ctreepo.manifesto_qsentence_dspy_family import (  # noqa: E402
    ManifestoQSentenceDSPyFamily,
    ManifestoQSentenceDSPyFamilyConfig,
    _node_target_scores,
    _prediction_scores,
    _summary_target,
    _tree_target_scores,
)
from src.tasks.manifesto.script_utils import (  # noqa: E402
    append_jsonl,
    now_iso as _now_iso,
    now_stamp as _now_stamp,
    parse_compact_dimensions as _parse_dimensions,
    write_json as _write_json,
)
from src.tasks.manifesto.span_targets import (  # noqa: E402
    COMPACT_TARGET_DIMENSIONS,
    parse_compact_scores_json,
)
from src.tree.labeled import LabeledNode, LabeledTree  # noqa: E402


LOGGER = logging.getLogger("manifesto_qsentence_sampled_supervision")



def _split_trees(
    trees: Sequence[LabeledTree],
    *,
    train_split: str,
    eval_split: str,
) -> Tuple[List[LabeledTree], List[LabeledTree]]:
    train_label = str(train_split).strip().lower()
    eval_label = str(eval_split).strip().lower()
    train = [
        tree
        for tree in trees
        if str((tree.metadata or {}).get("split", "")).strip().lower() == train_label
    ]
    eval_trees = [
        tree
        for tree in trees
        if str((tree.metadata or {}).get("split", "")).strip().lower() == eval_label
    ]
    return train, eval_trees


def _cap(items: Sequence[LabeledTree], n: int, *, seed: int) -> List[LabeledTree]:
    values = list(items)
    if n <= 0 or len(values) <= n:
        return values
    rng = random.Random(seed)
    return [tree for _idx, tree in sorted(rng.sample(list(enumerate(values)), n))]


def _compact_state_json(scores: Mapping[str, Any], *, total_non_header: int) -> str:
    payload = {
        "cmp_state": {
            "compact_targets": {
                dim: float(scores[dim])
                for dim in COMPACT_TARGET_DIMENSIONS
                if dim in scores and scores[dim] is not None
            },
            "total_non_header": int(max(0, total_non_header)),
        }
    }
    return json.dumps(payload, sort_keys=True)


def _leaf_total(node: LabeledNode) -> int:
    meta = dict(node.metadata or {})
    for key in ("total_non_header_qsentences", "total_non_header", "total_qsentences"):
        raw = meta.get(key)
        if raw is not None:
            try:
                return max(1, int(raw))
            except (TypeError, ValueError):
                pass
    return 1


def _weighted_leaf_estimate(
    leaves: Sequence[LabeledNode],
    *,
    dims: Sequence[str],
) -> Dict[str, float]:
    totals = [_leaf_total(node) for node in leaves]
    denom = float(sum(totals))
    if denom <= 0.0:
        return {}
    out: Dict[str, float] = {}
    for dim in dims:
        num = 0.0
        seen = False
        for node, weight in zip(leaves, totals):
            scores = _node_target_scores(node)
            if dim not in scores:
                continue
            num += float(scores[dim]) * float(weight)
            seen = True
        if seen:
            out[dim] = max(0.0, min(1.0, num / denom))
    return out


@dataclass
class SampledExample:
    tree: LabeledTree
    prompt: str
    completion: str
    target_scores: Dict[str, float]
    sample_scores: Dict[str, float]
    sample_node_ids: List[str]
    sample_leaf_count: int
    total_leaf_count: int
    sampled_qsentences: int
    total_qsentences: int
    sample_unit: str
    leaf_inclusion_probability: float
    # Source of the per-leaf summary statistics placed in the prompt:
    #   "gold_stats"  -> gold CMP weighted means of the sampled leaves (legacy;
    #                    this regresses gold-stats -> gold-target and does NOT
    #                    exercise the learned f or any text merge).
    #   "f_states"    -> the learned f applied to each sampled leaf's summary,
    #                    i.e. the model's own predicted leaf states. This makes
    #                    g compose real f outputs into a root prediction.
    sample_state_source: str = "gold_stats"
    # Per-leaf predicted f-states (f_states mode only); empty otherwise.
    leaf_f_states: List[Dict[str, float]] = field(default_factory=list)
    # FAIR no-merge baseline: the qsentence-weighted mean of the per-leaf states
    # actually placed in the prompt. In f_states mode these are the LEARNED f
    # states g was given, so "g beats input_baseline" = g learned a merge better
    # than naively averaging the same inputs (vs sample_baseline, which is the
    # GOLD-leaf weighted mean = a near-oracle g cannot fairly beat).
    input_baseline: Dict[str, float] = field(default_factory=dict)


def _root_summary(tree: LabeledTree, target_scores: Mapping[str, Any]) -> str:
    if tree.levels:
        root_id = tree.levels[-1][0]
        root = tree.get_node(str(root_id))
        if root is not None:
            summary = _summary_target(root, include_identity_targets=True)
            if summary:
                return summary
    return _compact_state_json(
        target_scores,
        total_non_header=int((tree.metadata or {}).get("qsents_per_doc") or 0),
    )


def _weighted_state_estimate(
    states: Sequence[Mapping[str, float]],
    weights: Sequence[int],
    *,
    dims: Sequence[str],
) -> Dict[str, float]:
    """Qsentence-weighted aggregate over a list of per-leaf state dicts."""
    denom = float(sum(max(0, int(w)) for w in weights))
    if denom <= 0.0:
        return {}
    out: Dict[str, float] = {}
    for dim in dims:
        num = 0.0
        seen = False
        for state, weight in zip(states, weights):
            if dim not in state or state[dim] is None:
                continue
            num += float(state[dim]) * float(weight)
            seen = True
        if seen:
            out[dim] = max(0.0, min(1.0, num / denom))
    return out


def _sample_example(
    tree: LabeledTree,
    *,
    sample_leaf_count: int,
    dims: Sequence[str],
    rng: random.Random,
    leaf_state_fn: Optional[Callable[[LabeledNode], Dict[str, float]]] = None,
) -> Optional[SampledExample]:
    leaves = list(tree.get_leaves())
    if not leaves:
        return None
    k = min(max(1, int(sample_leaf_count)), len(leaves))
    sampled = [node for _idx, node in sorted(rng.sample(list(enumerate(leaves)), k))]
    target_scores = {
        dim: float(value)
        for dim, value in _tree_target_scores(tree).items()
        if dim in dims and value is not None
    }
    if not target_scores:
        return None
    # sample_scores stays the GOLD weighted estimate so the sample_baseline
    # method (oracle reference) is unchanged regardless of prompt source.
    sample_scores = _weighted_leaf_estimate(sampled, dims=dims)
    total_q = int((tree.metadata or {}).get("qsents_per_doc") or sum(_leaf_total(n) for n in leaves))
    sampled_q = sum(_leaf_total(node) for node in sampled)
    leaf_weights = [_leaf_total(node) for node in sampled]
    leaf_inclusion_probability = float(k) / float(len(leaves))

    state_source = "f_states" if leaf_state_fn is not None else "gold_stats"
    leaf_f_states: List[Dict[str, float]] = []
    if leaf_state_fn is not None:
        # Predicted per-leaf states from the learned f. g must compose THESE,
        # so gold labels never enter the prompt.
        for node in sampled:
            predicted = leaf_state_fn(node) or {}
            leaf_f_states.append(
                {dim: float(predicted[dim]) for dim in dims if dim in predicted and predicted[dim] is not None}
            )
        prompt_states = leaf_f_states
        prompt_aggregate = _weighted_state_estimate(leaf_f_states, leaf_weights, dims=dims)
        stat_note = (
            "Per-leaf states below are the LEARNED f model's own predictions for "
            "each sampled leaf window (no gold labels). Compose them into a "
            "full-document CMP compact state."
        )
    else:
        # Legacy oracle path: gold weighted means. Bypasses f and any text merge.
        prompt_states = [
            {dim: float(_node_target_scores(node)[dim]) for dim in dims if dim in _node_target_scores(node)}
            for node in sampled
        ]
        prompt_aggregate = {
            dim: round(float(sample_scores[dim]), 6)
            for dim in dims
            if dim in sample_scores and sample_scores[dim] is not None
        }
        stat_note = (
            "Per-leaf states below are GOLD CMP weighted means of the sampled "
            "leaf windows. Estimate the full-document CMP compact state."
        )

    per_leaf_payload = [
        {
            "node_id": str(node.node_id),
            "qsentences": int(weight),
            "state": {dim: round(float(state[dim]), 6) for dim in dims if dim in state and state[dim] is not None},
        }
        for node, weight, state in zip(sampled, leaf_weights, prompt_states)
    ]
    aggregate_means = {
        dim: round(float(value), 6)
        for dim, value in prompt_aggregate.items()
        if value is not None
    }
    aggregate_sums = {
        dim: round(float(value) * float(max(1, sampled_q)), 6)
        for dim, value in aggregate_means.items()
    }
    prompt_payload = {
        "task": (
            "Estimate the full-document CMP compact state from a uniform random "
            "sample of q-sentence leaf-window states. Compose the per-leaf "
            "states into one document-level compact state. Do not reclassify raw text."
        ),
        "input_type": (
            "sample_f_states_v1" if leaf_state_fn is not None else "sample_sufficient_stats_v1"
        ),
        "state_source": state_source,
        "state_note": stat_note,
        "sampling": {
            "scheme": "uniform_without_replacement_over_leaf_windows",
            "sample_unit": "leaf_window",
            "doc_id": str(tree.doc_id),
            "total_leaf_count": int(len(leaves)),
            "sample_leaf_count": int(k),
            "leaf_inclusion_probability": float(leaf_inclusion_probability),
            "total_qsentences": int(total_q),
            "sampled_qsentences": int(sampled_q),
            "qsentence_inclusion_note": (
                "Qsentences are included through their sampled leaf window; this is "
                "not individual-qsentence sampling unless leaf_qsentences=1."
            ),
        },
        "per_leaf_states": per_leaf_payload,
        "sample_statistics": {
            "denominator_qsentences": int(max(1, sampled_q)),
            "weighted_mean": aggregate_means,
            "weighted_sum": aggregate_sums,
        },
        "required_output": {
            "format": "strict_json",
            "path": "cmp_state.compact_targets",
            "dimensions": list(dims),
        },
    }
    return SampledExample(
        tree=tree,
        prompt=json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True),
        completion=_root_summary(tree, target_scores),
        target_scores=target_scores,
        sample_scores=sample_scores,
        sample_node_ids=[str(node.node_id) for node in sampled],
        sample_leaf_count=int(k),
        total_leaf_count=int(len(leaves)),
        sampled_qsentences=int(sampled_q),
        total_qsentences=int(total_q),
        sample_unit="leaf_window",
        leaf_inclusion_probability=float(leaf_inclusion_probability),
        sample_state_source=state_source,
        leaf_f_states=leaf_f_states,
        input_baseline={
            dim: float(prompt_aggregate[dim])
            for dim in dims
            if dim in prompt_aggregate and prompt_aggregate[dim] is not None
        },
    )


def _build_examples(
    trees: Sequence[LabeledTree],
    *,
    sample_leaf_count: int,
    samples_per_doc: int,
    seed: int,
    dims: Sequence[str],
    leaf_state_fn: Optional[Callable[[LabeledNode], Dict[str, float]]] = None,
) -> List[SampledExample]:
    out: List[SampledExample] = []
    for tree_idx, tree in enumerate(trees):
        for repeat in range(max(1, int(samples_per_doc))):
            rng = random.Random(int(seed) + 1000003 * tree_idx + 9176 * repeat)
            example = _sample_example(
                tree,
                sample_leaf_count=int(sample_leaf_count),
                dims=dims,
                rng=rng,
                leaf_state_fn=leaf_state_fn,
            )
            if example is not None:
                out.append(example)
    return out


def _sample_mask_row(example: SampledExample, *, split: str, index: int) -> Dict[str, Any]:
    return {
        "split": str(split),
        "index": int(index),
        "doc_id": str(example.tree.doc_id),
        "sample_unit": str(example.sample_unit),
        "sample_state_source": str(example.sample_state_source),
        "sampling_scheme": "uniform_without_replacement_over_leaf_windows",
        "sample_node_ids": list(example.sample_node_ids),
        "sample_leaf_count": int(example.sample_leaf_count),
        "total_leaf_count": int(example.total_leaf_count),
        "leaf_inclusion_probability": float(example.leaf_inclusion_probability),
        "sampled_qsentences": int(example.sampled_qsentences),
        "total_qsentences": int(example.total_qsentences),
    }


def _mae(pred: Sequence[Optional[float]], truth: Sequence[Optional[float]]) -> Optional[float]:
    diffs = [
        abs(float(p) - float(t))
        for p, t in zip(pred, truth)
        if p is not None and t is not None and math.isfinite(float(p)) and math.isfinite(float(t))
    ]
    return float(sum(diffs) / len(diffs)) if diffs else None


def _pearson(pred: Sequence[Optional[float]], truth: Sequence[Optional[float]]) -> Optional[float]:
    pairs = [
        (float(p), float(t))
        for p, t in zip(pred, truth)
        if p is not None and t is not None and math.isfinite(float(p)) and math.isfinite(float(t))
    ]
    if len(pairs) < 2:
        return None
    xs = [p for p, _ in pairs]
    ys = [t for _, t in pairs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return float(cov / math.sqrt(vx * vy))


def _metric_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    dims: Sequence[str],
    target_key: str = "target_scores",
    methods: Sequence[str] = ("sample_baseline", "input_baseline", "g_direct", "f_on_g"),
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(rows), "methods": {}}
    for method in methods:
        method_payload: Dict[str, Any] = {}
        all_pred: List[Optional[float]] = []
        all_truth: List[Optional[float]] = []
        for dim in dims:
            pred = [
                (row.get(method) or {}).get(dim)
                for row in rows
            ]
            truth = [
                (row.get(target_key) or {}).get(dim)
                for row in rows
            ]
            method_payload[dim] = {
                "mae": _mae(pred, truth),
                "pearson": _pearson(pred, truth),
            }
            all_pred.extend(pred)
            all_truth.extend(truth)
        method_payload["all_dims"] = {
            "mae": _mae(all_pred, all_truth),
            "pearson": _pearson(all_pred, all_truth),
        }
        out["methods"][method] = method_payload
    return out



def _build_family(args: argparse.Namespace, dims: Sequence[str]) -> ManifestoQSentenceDSPyFamily:
    lm_config: Dict[str, Any] = {
        "model": str(args.dspy_model),
        "api_base": str(args.dspy_api_base),
        "api_key": str(args.dspy_api_key),
        "max_tokens": int(args.dspy_max_tokens),
    }
    config_kwargs: Dict[str, Any] = {
        "optimizer": str(args.dspy_optimizer),
        "budget": str(args.dspy_budget),
        "num_threads": int(args.dspy_num_threads),
        "lm_config": lm_config,
        "lm_transport": str(args.dspy_lm_transport),
        "batch_max_concurrent": int(args.dspy_batch_max_concurrent),
        "batch_size": int(args.dspy_batch_size),
        "batch_timeout": float(args.dspy_batch_timeout),
        "batch_request_timeout": float(args.dspy_batch_request_timeout),
        "batch_await_response_timeout": args.dspy_batch_await_response_timeout,
        "batch_routing_policy": str(args.dspy_batch_routing_policy),
        "max_train_records": int(args.max_train_examples),
        "leaf_size_tokens": int(args.leaf_size_tokens),
        "lm_context_window_tokens": int(args.dspy_lm_context_tokens),
        "max_completion_tokens": int(args.dspy_max_tokens),
        "target_dimensions": tuple(dims),
        "g_direct_parse_reward_weight": float(args.g_direct_parse_reward_weight),
        "g_f_proxy_reward_weight": float(args.g_f_proxy_reward_weight),
        "strict_optimizer_errors": True,
    }
    if args.dspy_reflection_minibatch_size is not None:
        from src.training.optimization.gepa import GEPA_STRONG_DEFAULT_KWARGS

        gepa_kwargs = dict(GEPA_STRONG_DEFAULT_KWARGS)
        gepa_kwargs["reflection_minibatch_size"] = int(args.dspy_reflection_minibatch_size)
        config_kwargs["gepa_kwargs"] = gepa_kwargs
    return ManifestoQSentenceDSPyFamily(
        config=ManifestoQSentenceDSPyFamilyConfig(**config_kwargs)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fg-grid-dir", default="outputs/manifesto_qsentence_dspy_labeled_grid")
    parser.add_argument("--leaf-qsentences", type=int, default=16)
    parser.add_argument("--leaf-size-tokens", type=int, default=512)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--f-artifact", required=True)
    parser.add_argument("--sample-leaf-count", type=int, default=2)
    parser.add_argument("--samples-per-doc", type=int, default=1)
    parser.add_argument(
        "--sample-state-source",
        choices=["f_states", "gold_stats"],
        default="f_states",
        help=(
            "What per-leaf states go in the g prompt. 'f_states' (default) feeds "
            "the learned f's own per-leaf predictions so g composes real model "
            "outputs into a root prediction. 'gold_stats' is the legacy oracle "
            "path that injects gold CMP means (bypasses f and any text merge)."
        ),
    )
    parser.add_argument("--train-docs", type=int, default=32)
    parser.add_argument("--eval-docs", type=int, default=12)
    parser.add_argument("--max-train-examples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--target-dimensions", default="all")
    parser.add_argument("--dspy-model", default="openai/nvidia/Gemma-4-31B-IT-NVFP4")
    parser.add_argument("--dspy-api-base", default="http://localhost:8000/v1")
    parser.add_argument("--dspy-api-key", default="EMPTY")
    parser.add_argument("--dspy-lm-transport", choices=["batch", "litellm"], default="batch")
    parser.add_argument("--dspy-optimizer", default="gepa")
    parser.add_argument("--dspy-budget", default="light")
    parser.add_argument("--dspy-num-threads", type=int, default=64)
    parser.add_argument("--eval-num-threads", type=int, default=0)
    parser.add_argument(
        "--f-prewarm-threads",
        type=int,
        default=0,
        help=(
            "Concurrency for pre-warming per-leaf f-states in f_states mode "
            "(0 = use --dspy-num-threads). Higher saturates the LM server."
        ),
    )
    parser.add_argument("--dspy-batch-max-concurrent", type=int, default=64)
    parser.add_argument("--dspy-batch-size", type=int, default=32)
    parser.add_argument("--dspy-batch-timeout", type=float, default=DEFAULT_BATCH_TIMEOUT_SECONDS)
    parser.add_argument("--dspy-batch-request-timeout", type=float, default=DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS)
    parser.add_argument("--dspy-batch-await-response-timeout", type=float, default=None)
    parser.add_argument("--dspy-batch-routing-policy", default=DEFAULT_BATCH_ROUTING_POLICY)
    parser.add_argument("--dspy-lm-context-tokens", type=int, default=32768)
    parser.add_argument("--dspy-max-tokens", type=int, default=1024)
    parser.add_argument("--dspy-gepa-val-examples", type=int, default=0)
    parser.add_argument("--dspy-reflection-minibatch-size", type=int, default=None)
    parser.add_argument("--skip-gepa-if-base-score-at-least", type=float, default=0.0)
    parser.add_argument("--g-direct-parse-reward-weight", type=float, default=0.75)
    parser.add_argument("--g-f-proxy-reward-weight", type=float, default=0.25)
    parser.add_argument("--align-to-sample-baseline", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--g-artifact", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir or f"outputs/manifesto_qsentence_sampled_supervision_{_now_stamp()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dims = _parse_dimensions(str(args.target_dimensions))
    grid_path = leafq_dir(args.fg_grid_dir, int(args.leaf_qsentences)) / "labeled_trees.jsonl"
    trees = load_labeled_trees(grid_path)
    train_trees, eval_trees = _split_trees(
        trees,
        train_split=str(args.train_split),
        eval_split=str(args.eval_split),
    )
    train_trees = _cap(train_trees, int(args.train_docs), seed=int(args.seed) + 11)
    eval_trees = _cap(eval_trees or trees, int(args.eval_docs), seed=int(args.seed) + 29)

    import dspy

    family = _build_family(args, dims)
    # Load the tokenizer before DSPy/GEPA starts worker threads. Lazy HF imports
    # from metric workers can trip Python shutdown/atexit registration races.
    from src.preprocessing.leaf_size_utils import get_gemma_tokenizer

    get_gemma_tokenizer()
    f_program = family._load_f_program(str(args.f_artifact))

    # Build the per-leaf state source. In f_states mode (default), each sampled
    # leaf's state is the LEARNED f applied to that leaf's summary, memoized by
    # node id so repeated samples of the same leaf cost one f call. In gold_stats
    # mode, _sample_example falls back to gold CMP means (legacy oracle path).
    leaf_state_fn: Optional[Callable[[LabeledNode], Dict[str, float]]] = None
    _leaf_state_cache: Dict[str, Dict[str, float]] = {}
    if str(args.sample_state_source) == "f_states":
        lm = family._ensure_lm()

        def _eval_leaf_f(node: LabeledNode) -> Dict[str, float]:
            summary = _summary_target(node, include_identity_targets=True) or str(node.text or "")
            with dspy.context(lm=lm, max_errors=1, provide_traceback=True):
                scores = family._apply_f_scores(f_program, response=summary)
            return {dim: float(scores[dim]) for dim in dims if dim in scores and scores[dim] is not None}

        def _leaf_state(node: LabeledNode) -> Dict[str, float]:
            key = str(node.node_id)
            cached = _leaf_state_cache.get(key)
            if cached is not None:
                return cached
            scores = _eval_leaf_f(node)
            _leaf_state_cache[key] = scores
            return scores

        # Throughput: a single dry build pass records exactly which leaves the
        # sampler will draw (deterministic seeds), then we run f over those
        # UNIQUE leaves CONCURRENTLY to saturate the LM server. The real build
        # passes below then hit a fully warm cache (zero extra f calls).
        _recorded: Dict[str, LabeledNode] = {}

        def _record(node: LabeledNode) -> Dict[str, float]:
            _recorded.setdefault(str(node.node_id), node)
            return {}

        for build_seed, src_trees, spd in (
            (int(args.seed) + 101, train_trees, int(args.samples_per_doc)),
            (int(args.seed) + 503, eval_trees, 1),
        ):
            _build_examples(
                src_trees,
                sample_leaf_count=int(args.sample_leaf_count),
                samples_per_doc=spd,
                seed=build_seed,
                dims=dims,
                leaf_state_fn=_record,
            )
        leaves_to_eval = list(_recorded.values())
        prewarm_threads = int(args.f_prewarm_threads) if int(args.f_prewarm_threads) > 0 else int(args.dspy_num_threads)
        prewarm_threads = max(1, min(prewarm_threads, len(leaves_to_eval) or 1))
        LOGGER.info(
            "Pre-warming %d unique leaf f-states with %d concurrent workers",
            len(leaves_to_eval),
            prewarm_threads,
        )
        if leaves_to_eval:
            def _warm(node: LabeledNode) -> Tuple[str, Dict[str, float]]:
                return str(node.node_id), _eval_leaf_f(node)

            if prewarm_threads == 1:
                warmed = [_warm(node) for node in leaves_to_eval]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=prewarm_threads) as pool:
                    warmed = list(pool.map(_warm, leaves_to_eval))
            for key, scores in warmed:
                _leaf_state_cache[key] = scores

        leaf_state_fn = _leaf_state
        LOGGER.info(
            "Sampled-g prompt source = f_states (learned f per-leaf predictions; "
            "cache warmed for %d leaves)",
            len(_leaf_state_cache),
        )
    else:
        LOGGER.info("Sampled-g prompt source = gold_stats (legacy oracle; bypasses f)")

    train_examples_payload = _build_examples(
        train_trees,
        sample_leaf_count=int(args.sample_leaf_count),
        samples_per_doc=int(args.samples_per_doc),
        seed=int(args.seed) + 101,
        dims=dims,
        leaf_state_fn=leaf_state_fn,
    )
    eval_examples_payload = _build_examples(
        eval_trees,
        sample_leaf_count=int(args.sample_leaf_count),
        samples_per_doc=1,
        seed=int(args.seed) + 503,
        dims=dims,
        leaf_state_fn=leaf_state_fn,
    )
    if not train_examples_payload:
        raise RuntimeError("no sampled training examples built")
    if not eval_examples_payload:
        raise RuntimeError("no sampled eval examples built")

    append_jsonl(
        output_dir / "train_sample_masks.jsonl",
        (
            _sample_mask_row(example, split=str(args.train_split), index=idx)
            for idx, example in enumerate(train_examples_payload)
        ),
        append=False,
    )
    append_jsonl(
        output_dir / "eval_sample_masks.jsonl",
        (
            _sample_mask_row(example, split=str(args.eval_split), index=idx)
            for idx, example in enumerate(eval_examples_payload)
        ),
        append=False,
    )

    g_program: Any
    g_artifact: Optional[str] = str(args.g_artifact) if args.g_artifact else None
    g_training_strategy = "loaded" if args.skip_train else "compiled"
    base_g_validation_score: Optional[float] = None
    skipped_gepa_due_to_base_score = False
    if args.skip_train:
        if not g_artifact:
            raise ValueError("--skip-train requires --g-artifact")
        g_program = family._load_g_program(g_artifact)
    else:
        train_payload = train_examples_payload[: int(args.max_train_examples)]
        train_completions = [
            (
                _compact_state_json(
                    ex.sample_scores,
                    total_non_header=ex.sampled_qsentences,
                )
                if args.align_to_sample_baseline
                else ex.completion
            )
            for ex in train_payload
        ]
        train_examples = [
            dspy.Example(
                prompt=ex.prompt,
                completion=completion,
                target_scores_json=json.dumps(
                    ex.sample_scores if args.align_to_sample_baseline else ex.target_scores,
                    sort_keys=True,
                ),
            ).with_inputs("prompt")
            for ex, completion in zip(train_payload, train_completions)
        ]
        base_g = dspy.Predict(family._g_signature())

        def metric(gold: Any, pred: Any, trace: Any = None, *unused: Any, **kwargs: Any) -> float:
            summary = str(getattr(pred, "completion", "") or "")
            if not summary:
                return 0.0
            try:
                target = json.loads(str(getattr(gold, "target_scores_json", "{}") or "{}"))
            except json.JSONDecodeError:
                target = {}
            return family._score_g_candidate_state(
                summary=summary,
                target=target,
                f_program=f_program,
            )

        optimizer_val_examples = list(train_examples)
        if 0 < int(args.dspy_gepa_val_examples) < len(train_examples):
            rng = random.Random(int(args.seed) + 1009)
            chosen = sorted(rng.sample(range(len(train_examples)), int(args.dspy_gepa_val_examples)))
            optimizer_val_examples = [train_examples[idx] for idx in chosen]
        lm = family._ensure_lm()
        base_threshold = float(args.skip_gepa_if_base_score_at_least)
        if base_threshold > 0.0:
            base_eval_threads = max(1, min(int(args.dspy_num_threads), len(optimizer_val_examples)))
            LOGGER.info(
                "Evaluating base sampled-stat g on %d validation examples with %d threads",
                len(optimizer_val_examples),
                base_eval_threads,
            )

            def score_base(example: Any) -> float:
                with dspy.context(lm=lm, max_errors=1, provide_traceback=True):
                    pred = base_g(prompt=example.prompt, config={"max_tokens": int(args.dspy_max_tokens)})
                return float(metric(example, pred))

            if base_eval_threads == 1:
                base_scores = [score_base(example) for example in optimizer_val_examples]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=base_eval_threads) as pool:
                    base_scores = list(pool.map(score_base, optimizer_val_examples))
            base_g_validation_score = (
                float(sum(base_scores) / len(base_scores)) if base_scores else 0.0
            )
            LOGGER.info(
                "Base sampled-stat g validation score %.9f; skip threshold %.9f",
                base_g_validation_score,
                base_threshold,
            )
            if base_g_validation_score >= base_threshold:
                g_program = base_g
                g_training_strategy = "base_g_skipped_gepa"
                skipped_gepa_due_to_base_score = True
            else:
                with dspy.context(lm=lm, max_errors=1, provide_traceback=True):
                    g_program = family._compile(
                        program=base_g,
                        metric=metric,
                        trainset=train_examples,
                        valset=optimizer_val_examples,
                    )
        else:
            with dspy.context(lm=lm, max_errors=1, provide_traceback=True):
                g_program = family._compile(
                    program=base_g,
                    metric=metric,
                    trainset=train_examples,
                    valset=optimizer_val_examples,
                )
        g_artifact_path = output_dir / "sampled_g_program.json"
        g_program.save(str(g_artifact_path))
        g_artifact = str(g_artifact_path)

    lm = family._ensure_lm()
    eval_threads = int(args.eval_num_threads)
    if eval_threads <= 0:
        eval_threads = int(args.dspy_num_threads)
    eval_threads = max(1, min(eval_threads, len(eval_examples_payload)))
    LOGGER.info(
        "Running sampled-g eval for %d examples with %d worker threads",
        len(eval_examples_payload),
        eval_threads,
    )

    def eval_one(item: Tuple[int, SampledExample]) -> Tuple[int, Dict[str, Any]]:
        idx, ex = item
        with dspy.context(lm=lm, max_errors=1, provide_traceback=True):
            pred = g_program(prompt=ex.prompt, config={"max_tokens": int(args.dspy_max_tokens)})
        completion = str(getattr(pred, "completion", "") or "")
        g_scores = parse_compact_scores_json(completion)
        f_scores = family._apply_f_scores(f_program, response=completion) if completion else {}
        return idx, {
            "doc_id": ex.tree.doc_id,
            "sample_node_ids": ex.sample_node_ids,
            "sample_leaf_count": ex.sample_leaf_count,
            "total_leaf_count": ex.total_leaf_count,
            "sample_unit": ex.sample_unit,
            "leaf_inclusion_probability": ex.leaf_inclusion_probability,
            "sampled_qsentences": ex.sampled_qsentences,
            "total_qsentences": ex.total_qsentences,
            "target_scores": ex.target_scores,
            "sample_baseline": ex.sample_scores,
            "input_baseline": ex.input_baseline,
            "g_direct": g_scores,
            "f_on_g": f_scores,
            "g_completion": completion,
        }

    indexed_eval = list(enumerate(eval_examples_payload))
    if eval_threads == 1:
        indexed_rows = [eval_one(item) for item in indexed_eval]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=eval_threads) as pool:
            indexed_rows = list(pool.map(eval_one, indexed_eval))
    rows = [row for _, row in sorted(indexed_rows, key=lambda item: item[0])]

    metrics = _metric_table(rows, dims=dims)
    metrics["alignment_to_sample_baseline"] = _metric_table(
        rows,
        dims=dims,
        target_key="sample_baseline",
        methods=("g_direct", "f_on_g"),
    )
    # Honest headline (two distinct questions):
    #  (1) DOES g COMPOSE: g_direct vs the TARGET (Pearson + MAE). Positive
    #      correlation means g reconstructs the document score from its inputs.
    #  (2) DID g BEAT NAIVE AVERAGING: g_direct vs input_baseline = the weighted
    #      mean of the SAME per-leaf states g was given. This is the FAIR no-merge
    #      baseline. (sample_baseline is the GOLD-leaf weighted mean = a near-oracle
    #      g cannot fairly beat when g works from noisy f-states, so it is reported
    #      only as a reference ceiling, NOT the pass/fail.)
    def _all_dims(method: str, key: str) -> Optional[float]:
        return ((metrics.get("methods") or {}).get(method) or {}).get("all_dims", {}).get(key)

    g_mae = _all_dims("g_direct", "mae")
    g_r = _all_dims("g_direct", "pearson")
    fg_mae = _all_dims("f_on_g", "mae")
    input_mae = _all_dims("input_baseline", "mae")
    oracle_mae = _all_dims("sample_baseline", "mae")
    metrics["composition_headline"] = {
        "sample_state_source": str(args.sample_state_source),
        # (1) does g compose at all (primary):
        "g_direct_vs_target_pearson": g_r,
        "g_direct_all_dims_mae": g_mae,
        "f_on_g_all_dims_mae": fg_mae,
        # (2) fair no-merge baseline = mean of the f-states g was given:
        "input_baseline_all_dims_mae": input_mae,
        "g_beats_input_baseline": (
            bool(g_mae is not None and input_mae is not None and g_mae < input_mae)
        ),
        "g_improvement_over_input_baseline_mae": (
            float(input_mae - g_mae) if (g_mae is not None and input_mae is not None) else None
        ),
        # reference ceiling only (gold-leaf oracle), NOT pass/fail:
        "oracle_sample_baseline_all_dims_mae": oracle_mae,
        "note": (
            "PRIMARY: g_direct_vs_target_pearson > 0 means g composes (reconstructs "
            "the doc score). FAIR TEST: g_beats_input_baseline = g_direct MAE < the "
            "weighted-mean of the SAME f-states g was given (did the learned merge "
            "beat naive averaging of identical inputs). oracle_sample_baseline "
            "(gold-leaf weighted mean) is a reference CEILING, not a target g can "
            "fairly beat from noisy f-states."
        ),
    }
    manifest = {
        "runner": Path(__file__).name,
        "created_at": _now_iso(),
        "grid_path": str(grid_path),
        "leaf_qsentences": int(args.leaf_qsentences),
        "leaf_size_tokens": int(args.leaf_size_tokens),
        "sample_leaf_count": int(args.sample_leaf_count),
        "samples_per_doc": int(args.samples_per_doc),
        "align_to_sample_baseline": bool(args.align_to_sample_baseline),
        "sample_state_source": str(args.sample_state_source),
        "sample_prompt_mode": (
            "sample_f_states_v1"
            if str(args.sample_state_source) == "f_states"
            else "sample_sufficient_stats_v1"
        ),
        "sampling_policy": {
            "estimand": "sampled_leaf_window_to_root_prediction",
            "candidate_population": "leaf_windows",
            "sample_unit": "leaf_window",
            "scheme": "fixed_size_uniform_without_replacement_per_document",
            "requested_sample_leaf_count": int(args.sample_leaf_count),
            "per_example_inclusion_probability": "sample_leaf_count / total_leaf_count",
            "masks_persisted": True,
            "train_sample_masks": "train_sample_masks.jsonl",
            "eval_sample_masks": "eval_sample_masks.jsonl",
            "not_ipw_all_node_local_law_objective": True,
            "qsentence_note": (
                "Qsentences are sampled through leaf windows; use leaf_qsentences=1 "
                "for individual-qsentence sampling."
            ),
        },
        "seed": int(args.seed),
        "train_docs": len(train_trees),
        "eval_docs": len(eval_trees),
        "train_examples": len(train_examples_payload),
        "used_train_examples": min(len(train_examples_payload), int(args.max_train_examples)),
        "optimizer_val_examples": (
            min(len(train_examples_payload), int(args.max_train_examples))
            if int(args.dspy_gepa_val_examples) <= 0
            else min(int(args.dspy_gepa_val_examples), min(len(train_examples_payload), int(args.max_train_examples)))
        ),
        "skip_gepa_if_base_score_at_least": float(args.skip_gepa_if_base_score_at_least),
        "base_g_validation_score": base_g_validation_score,
        "g_training_strategy": g_training_strategy,
        "skipped_gepa_due_to_base_score": skipped_gepa_due_to_base_score,
        "dspy_reflection_minibatch_size": args.dspy_reflection_minibatch_size,
        "eval_num_threads": int(args.eval_num_threads) if int(args.eval_num_threads) > 0 else int(args.dspy_num_threads),
        "eval_examples": len(eval_examples_payload),
        "f_artifact": str(args.f_artifact),
        "g_artifact": g_artifact,
        "target_dimensions": list(dims),
        "metrics": metrics,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "metrics.json", metrics)
    append_jsonl(output_dir / "eval_predictions.jsonl", rows, append=False)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
