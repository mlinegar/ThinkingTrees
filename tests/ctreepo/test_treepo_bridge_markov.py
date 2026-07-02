from __future__ import annotations

import json
from pathlib import Path

from src.ctreepo.treepo_bridge.markov import (
    MARKOV_BENCHMARK,
    make_markov_trees,
    register_markov_benchmark,
    run_markov_benchmark,
)


def test_markov_trees_expose_treepo_oracle_shape() -> None:
    trees = make_markov_trees(
        n_trees=3,
        n_regimes=3,
        vocab_size=24,
        min_tokens=32,
        max_tokens=32,
        min_segments=2,
        max_segments=4,
        min_seg_len=4,
        max_seg_len=16,
        leaf_token_count=8,
        sinkhorn_iters=3,
        transition_log_std=0.8,
        seed=7,
    )
    assert len(trees) == 3
    for tree in trees:
        assert tree.regimes == tree.token_regimes
        assert sum(len(leaf.regimes) for leaf in tree.leaves) == len(tree.regimes)
        truth = sum(
            int(left) != int(right)
            for left, right in zip(tree.regimes[:-1], tree.regimes[1:])
        )
        assert tree.metadata["teacher_score_native"] == float(truth)


def test_markov_registers_with_treepo_before_runner_import() -> None:
    register_markov_benchmark()
    from treepo.bench.runner import VALID_EXPERIMENTS

    assert MARKOV_BENCHMARK in VALID_EXPERIMENTS


def test_markov_run_single_works_after_runner_preimport(tmp_path: Path) -> None:
    from treepo.bench.runner import run_single

    register_markov_benchmark()
    run_single(
        experiment=MARKOV_BENCHMARK,
        config={
            "method": "oracle",
            "scorer": "markov_changepoint_count",
            "seed": 4,
            "split": "test",
            "n_trees": 2,
            "task_config": {
                "n_regimes": 3,
                "vocab_size": 24,
                "min_tokens": 32,
                "max_tokens": 32,
                "min_segments": 2,
                "max_segments": 4,
                "min_seg_len": 4,
                "max_seg_len": 16,
                "leaf_token_count": 8,
                "sinkhorn_iters": 3,
                "transition_log_std": 0.8,
            },
        },
        json_out=tmp_path / "markov_preimport.json",
        csv_out=tmp_path / "markov_preimport.csv",
    )
    assert (tmp_path / "markov_preimport.json").exists()


def test_markov_runs_through_treepo_benchmark(tmp_path: Path) -> None:
    run_markov_benchmark(
        config={
            "method": "oracle",
            "scorer": "markov_changepoint_count",
            "seed": 3,
            "split": "test",
            "n_trees": 4,
            "task_config": {
                "n_regimes": 3,
                "vocab_size": 24,
                "min_tokens": 32,
                "max_tokens": 32,
                "min_segments": 2,
                "max_segments": 4,
                "min_seg_len": 4,
                "max_seg_len": 16,
                "leaf_token_count": 8,
                "sinkhorn_iters": 3,
                "transition_log_std": 0.8,
            },
        },
        json_out=tmp_path / "markov.json",
        csv_out=tmp_path / "markov.csv",
    )
    payload = json.loads((tmp_path / "markov.json").read_text(encoding="utf-8"))
    row = payload["rows"][0]
    assert row["experiment"] == MARKOV_BENCHMARK
    assert row["method"] == "oracle"
    assert row["scorer"] == "markov_changepoint_count"
    assert row["n"] == 4
    assert row["internal_f_mae"] == 0.0
    assert row["external_expert_mae"] == 0.0
    assert (tmp_path / "markov.csv").exists()
