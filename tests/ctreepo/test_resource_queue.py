from __future__ import annotations

from src.ctreepo.sim.manifest import RunSpec
from src.ctreepo.sim.resource_queue import (
    assign_job_lane,
    detect_gpu_tokens,
    job_from_command,
    rewrite_command_for_lane,
)


def test_runspec_populates_resource_metadata_for_markov_auto() -> None:
    run = RunSpec.create(
        family="markov-ops-count",
        config={"device": "auto", "model_family": "neural", "torch_threads": 2},
        outputs={"json_summary": "out.json", "csv_summary": "out.csv"},
        command=(
            "python scripts/run_markov_changepoint_ops_count_simulation.py "
            "--device auto --model-family neural --torch-threads 2"
        ),
        requires=["torch"],
    )
    assert run.resources["accelerator"] == "auto"
    assert run.resources["gpu_eligible"] is True
    assert run.resources["gpu_preferred"] is False
    assert run.resources["cpu_threads"] == 2


def test_markov_auto_job_is_cpu_preferred() -> None:
    job = job_from_command(
        (
            "python scripts/run_markov_changepoint_ops_count_simulation.py "
            "--device auto --model-family neural --torch-threads 1"
        ),
        idx=2,
    )
    assert job.resources["accelerator"] == "auto"
    assert job.resources["gpu_preferred"] is False
    assert assign_job_lane(job, gpu_tokens=["MIG-0"]) == "cpu"


def test_ctree_sklearn_job_stays_cpu_even_with_device_auto() -> None:
    job = job_from_command(
        (
            "python scripts/run_segmented_lda_ctreepo_simulation.py "
            "--topic-phi-estimator sklearn_lda --leaf-theta-estimator sklearn_lda "
            "--device auto --torch-threads 1"
        ),
        idx=0,
    )
    assert job.resources["accelerator"] == "cpu"
    assert assign_job_lane(job, gpu_tokens=["MIG-0"]) == "cpu"


def test_ctree_neural_job_prefers_gpu() -> None:
    job = job_from_command(
        (
            "python scripts/run_segmented_lda_ctreepo_simulation.py "
            "--topic-phi-estimator neural_ctreepo --leaf-theta-estimator lstsq "
            "--device auto --torch-threads 1"
        ),
        idx=1,
    )
    assert job.resources["accelerator"] == "auto"
    assert job.resources["gpu_preferred"] is True
    assert assign_job_lane(job, gpu_tokens=["MIG-0"]) == "gpu"


def test_rewrite_command_for_lane_normalizes_cuda_device() -> None:
    command = "python foo.py --device auto --cuda-device 3 --torch-threads 1"
    assert "--cuda-device 0" in rewrite_command_for_lane(command, lane="gpu")
    cpu_cmd = rewrite_command_for_lane(command, lane="cpu")
    assert "--cuda-device" not in cpu_cmd
    assert "--device cpu" in cpu_cmd


def test_rewrite_command_for_lane_leaves_no_device_command_unchanged() -> None:
    command = "python scripts/run_lda_tree_recovery_simulation.py --seed 0 --json-summary out.json"
    assert rewrite_command_for_lane(command, lane="cpu") == command
    assert rewrite_command_for_lane(command, lane="gpu") == command


def test_detect_gpu_tokens_expands_gpu_indices(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.ctreepo.sim.resource_queue._query_gpu_layout",
        lambda: {0: ["MIG-a", "MIG-b"], 1: ["MIG-c"], 2: []},
    )
    assert detect_gpu_tokens("0 2") == ["MIG-a", "MIG-b", "2"]
