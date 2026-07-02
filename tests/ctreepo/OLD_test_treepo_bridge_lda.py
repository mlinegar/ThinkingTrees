# OLD_: archived 2026-07-02; tests the archived treepo_bridge LDA benchmark (OLD_lda.py). Kept for reference; do not import or run.
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.ctreepo.treepo_bridge.lda import (
    LDA_BENCHMARK,
    LDA_METHOD,
    LDA_SCORER,
    register_lda_benchmark,
    register_lda_method,
    run_lda_benchmark,
)


def _tiny_config(*, device: str = "cpu", cuda_device: int | None = None) -> dict[str, object]:
    return {
        "n_topics": 3,
        "vocab_size": 48,
        "min_tokens": 48,
        "max_tokens": 48,
        "doc_topic_concentration": 0.6,
        "topic_concentration": 0.2,
        "emission_mode": "anchored",
        "anchor_words_per_topic": 4,
        "anchor_multiplier": 15.0,
        "relevant_topics": 2,
        "theta_scale": 1.0,
        "zero_diagonal": False,
        "quadratic_utility_weight": 1.0,
        "leaf_tokens": 8,
        "train_docs": 8,
        "test_docs": 4,
        "inference_prior_mass": 0.25,
        "inference_max_iter": 25,
        "inference_tol": 1e-6,
        "full_hidden_dim": 16,
        "full_n_layers": 1,
        "state_dim": 8,
        "supervise_all_balanced_nodes": True,
        "n_epochs": 1,
        "batch_size": 4,
        "lr": 3e-3,
        "weight_decay": 1e-5,
        "device": device,
        "cuda_device": cuda_device,
        "torch_threads": 1,
        "seed": 0,
    }


def test_lda_registers_method_and_benchmark() -> None:
    register_lda_benchmark()
    from treepo.bench.tasks import list_task_benchmarks
    from treepo.methods import list_methods

    assert LDA_METHOD in set(list_methods())
    assert LDA_BENCHMARK in set(list_task_benchmarks())


def test_lda_method_runs_on_cpu(tmp_path: Path) -> None:
    register_lda_method()
    from treepo.methods import run

    result = run(
        LDA_METHOD,
        {
            **_tiny_config(device="cpu"),
            "output_dir": str(tmp_path / "lda_method"),
        },
    )
    payload = dict(result)
    assert payload["status"] == "success"
    assert Path(str(payload["manifest_path"])).exists()
    metrics = dict(payload["metrics"])
    assert metrics["device_is_cuda"] == 0.0
    assert metrics["n_train"] == 8.0
    assert metrics["n_test"] == 4.0
    assert math.isfinite(float(metrics["full_doc_operator_pi_l1_to_full_mean"]))


def test_lda_benchmark_runs_through_treepo(tmp_path: Path) -> None:
    run_lda_benchmark(
        config={
            "method": LDA_METHOD,
            "scorer": LDA_SCORER,
            "seed": 0,
            "split": "test",
            "n_trees": 0,
            "task_config": _tiny_config(device="cpu"),
        },
        json_out=tmp_path / "lda.json",
        csv_out=tmp_path / "lda.csv",
    )
    payload = json.loads((tmp_path / "lda.json").read_text(encoding="utf-8"))
    row = payload["rows"][0]
    assert row["experiment"] == LDA_BENCHMARK
    assert row["method"] == LDA_METHOD
    assert row["scorer"] == LDA_SCORER
    assert row["device_is_cuda"] == 0.0
    assert (tmp_path / "lda.csv").exists()


def test_lda_gpu_smoke_if_cuda_available(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    cuda_device = 1 if torch.cuda.device_count() > 1 else 0
    register_lda_method()
    from treepo.methods import run

    result = run(
        LDA_METHOD,
        {
            **_tiny_config(device="cuda", cuda_device=cuda_device),
            "output_dir": str(tmp_path / "lda_gpu_method"),
        },
    )
    payload = dict(result)
    metrics = dict(payload["metrics"])
    assert metrics["device_is_cuda"] == 1.0
    assert str(payload["summary"]["full_doc_operator"]["device"]).startswith("cuda")
