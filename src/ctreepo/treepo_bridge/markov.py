"""Markov task benchmark adapter for the standalone :mod:`treepo` package.

ThinkingTrees owns the Markov generators. The package owns the
small benchmark/method contract. This module connects them without moving the
generator into ``treepo``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# NOTE: ``src.tree.*`` imports are deferred into the functions that use them.
# Importing them at module load pulls in ``src.tree`` -> ``src.preprocessing``
# -> ``langextract``, which would force the whole preprocessing/LLM stack just
# to import this bridge (see treepo_bridge/__init__.py for the lazy rationale).
# ``ChangepointMarkovDoc``/``MarkovChangepointConfig`` are referenced only in
# type hints under ``from __future__ import annotations`` (strings at runtime),
# so they do not need a module-level import.


MARKOV_BENCHMARK = "markov"


@dataclass(frozen=True)
class TreepoMarkovLeaf:
    tokens: tuple[int, ...]
    regimes: tuple[int, ...]
    token_regimes: tuple[int, ...]


@dataclass(frozen=True)
class TreepoMarkovTree:
    leaves: tuple[TreepoMarkovLeaf, ...]
    tokens: tuple[int, ...]
    regimes: tuple[int, ...]
    token_regimes: tuple[int, ...]
    true_boundaries: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def make_markov_trees(
    *,
    n_trees: int = 8,
    n_regimes: int = 4,
    vocab_size: int = 96,
    min_tokens: int = 96,
    max_tokens: int = 96,
    min_segments: int = 2,
    max_segments: int = 5,
    min_seg_len: int = 8,
    max_seg_len: int = 32,
    leaf_token_count: int = 16,
    sinkhorn_iters: int = 30,
    transition_log_std: float = 1.25,
    seed: int = 0,
    split: str = "test",
) -> list[TreepoMarkovTree]:
    """Generate ThinkingTrees Markov docs and wrap them for ``treepo``."""

    if int(n_trees) <= 0:
        raise ValueError("n_trees must be positive")
    if int(leaf_token_count) <= 0:
        raise ValueError("leaf_token_count must be positive")

    from src.tree.markov_boundary_honesty_simulation import _make_transition_matrices
    from src.tree.markov_changepoint_honesty_simulation import (
        MarkovChangepointConfig,
        generate_changepoint_docs,
    )

    config = MarkovChangepointConfig(
        n_regimes=int(n_regimes),
        vocab_size=int(vocab_size),
        min_tokens=int(min_tokens),
        max_tokens=int(max_tokens),
        min_segments=int(min_segments),
        max_segments=int(max_segments),
        min_seg_len=int(min_seg_len),
        max_seg_len=int(max_seg_len),
        train_docs=0,
        test_docs=int(n_trees),
        sinkhorn_iters=int(sinkhorn_iters),
        transition_log_std=float(transition_log_std),
        seed=int(seed),
        use_cuda=False,
    )
    rng = np.random.default_rng(int(seed))
    transitions = _make_transition_matrices(
        n_classes=int(config.n_regimes),
        vocab_size=int(config.vocab_size),
        log_std=float(config.transition_log_std),
        sinkhorn_iters=int(config.sinkhorn_iters),
        rng=rng,
    )
    docs = generate_changepoint_docs(config, transitions=transitions)
    return [
        _wrap_markov_doc(
            doc,
            index=i,
            split=str(split),
            seed=int(seed),
            leaf_token_count=int(leaf_token_count),
            config=config,
        )
        for i, doc in enumerate(docs)
    ]


def register_markov_benchmark() -> None:
    """Register the ThinkingTrees Markov benchmark with ``treepo``."""

    from treepo.bench.tasks import (
        TaskBenchmarkSpec,
        list_task_benchmarks,
        register_task_benchmark,
    )

    register_task_benchmark(
        TaskBenchmarkSpec(
            name=MARKOV_BENCHMARK,
            default_method="oracle",
            default_scorer="markov_changepoint_count",
            supported_scorers=("markov_changepoint_count",),
            default_task_config={
                "n_regimes": 4,
                "vocab_size": 96,
                "min_tokens": 96,
                "max_tokens": 96,
                "min_segments": 2,
                "max_segments": 5,
                "min_seg_len": 8,
                "max_seg_len": 32,
                "leaf_token_count": 16,
                "sinkhorn_iters": 30,
                "transition_log_std": 1.25,
            },
            allowed_task_config_keys=(
                "n_regimes",
                "vocab_size",
                "min_tokens",
                "max_tokens",
                "min_segments",
                "max_segments",
                "min_seg_len",
                "max_seg_len",
                "leaf_token_count",
                "sinkhorn_iters",
                "transition_log_std",
            ),
            build_method_config=_build_markov_method_config,
        ),
        replace=True,
    )


def run_markov_benchmark(
    *,
    config: Mapping[str, Any],
    json_out: str | Path,
    csv_out: str | Path,
    print_json: bool = False,
) -> dict[str, object]:
    """Run the registered benchmark through ``treepo.bench.runner``."""

    register_markov_benchmark()
    from treepo.bench.runner import run_single

    return run_single(
        experiment=MARKOV_BENCHMARK,
        config=dict(config),
        json_out=Path(json_out),
        csv_out=Path(csv_out),
        print_json=bool(print_json),
    )


def _build_markov_method_config(
    config: Any,
    task_config: Mapping[str, Any],
    scorer: str,
    output_dir: Path | None,
) -> dict[str, Any]:
    trees = make_markov_trees(
        n_trees=int(config.n_trees),
        seed=int(config.seed),
        split=str(config.split),
        **dict(task_config),
    )
    method_config = dict(getattr(config, "method_config", {}) or {})
    method_config.update(
        {
            "oracle_name": str(scorer),
            "eval_data": trees,
        }
    )
    if output_dir is not None:
        method_config["output_dir"] = str(output_dir)
    return method_config


def _wrap_markov_doc(
    doc: ChangepointMarkovDoc,
    *,
    index: int,
    split: str,
    seed: int,
    leaf_token_count: int,
    config: MarkovChangepointConfig,
) -> TreepoMarkovTree:
    leaves: list[TreepoMarkovLeaf] = []
    tokens = tuple(int(x) for x in doc.tokens)
    regimes = tuple(int(x) for x in doc.token_regimes)
    for start in range(0, len(tokens), int(leaf_token_count)):
        end = min(len(tokens), start + int(leaf_token_count))
        leaf_tokens = tokens[start:end]
        leaf_regimes = regimes[start:end]
        leaves.append(
            TreepoMarkovLeaf(
                tokens=leaf_tokens,
                regimes=leaf_regimes,
                token_regimes=leaf_regimes,
            )
        )
    truth = float(len(tuple(doc.true_boundaries)))
    return TreepoMarkovTree(
        leaves=tuple(leaves),
        tokens=tokens,
        regimes=regimes,
        token_regimes=regimes,
        true_boundaries=tuple(int(x) for x in doc.true_boundaries),
        metadata={
            "tree_id": f"thinkingtrees_markov_{int(seed)}_{int(index)}",
            "split": str(split),
            "teacher_score_1_7": truth,
            "teacher_score_native": truth,
            "expert_score_1_7": truth,
            "expert_score_native": truth,
            "expert_target_scale": "raw",
            "expert_score_for_objective": truth,
            "source": "ThinkingTrees.markov_changepoint_honesty_simulation",
            "n_regimes": int(config.n_regimes),
            "vocab_size": int(config.vocab_size),
            "min_tokens": int(config.min_tokens),
            "max_tokens": int(config.max_tokens),
            "leaf_token_count": int(leaf_token_count),
        },
    )


__all__ = [
    "MARKOV_BENCHMARK",
    "TreepoMarkovLeaf",
    "TreepoMarkovTree",
    "make_markov_trees",
    "register_markov_benchmark",
    "run_markov_benchmark",
]
