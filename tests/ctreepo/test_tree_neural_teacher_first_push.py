from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from scripts import run_tree_neural_full_doc_mig as mig
from scripts import run_tree_neural_teacher_first_push as tfpush


def test_candidate_summary_prefers_lower_teacher_first_bound() -> None:
    runs = [
        {
            "config_label": "teacherfirst_shared_feature_phi128__leaf_dense__judge_t256",
            "seed": 0,
            "teacher_first_total_bound": 0.9,
            "stage1_substitution_cost": 0.6,
            "test_root_mae": 0.2,
            "stage2_transport_budget": 0.1,
        },
        {
            "config_label": "teacherfirst_shared_feature_phi128__internal_full_dense__judge_t256",
            "seed": 0,
            "teacher_first_total_bound": 0.7,
            "stage1_substitution_cost": 0.5,
            "test_root_mae": 0.25,
            "stage2_transport_budget": 0.1,
        },
        {
            "config_label": "teacherfirst_fiber_primary_phi128__leaf_dense__judge_t256",
            "seed": 0,
            "teacher_first_total_bound": 0.55,
            "stage1_substitution_cost": 0.4,
            "test_root_mae": 0.35,
            "stage2_transport_budget": 0.08,
        },
    ]

    summary = tfpush._aggregate_candidate_summary(runs)

    assert summary[0]["candidate_label"] == "teacherfirst_fiber_primary_phi128"
    assert summary[1]["candidate_label"] == "teacherfirst_shared_feature_phi128"
    assert summary[0]["on_pareto_frontier"] is True


def test_build_stage2_jobs_uses_loaded_stage1_artifact_dir() -> None:
    args = tfpush._parser().parse_args(
        [
            "--benchmark",
            "smoke",
            "--phase1-train-docs",
            "8",
            "--phase2-train-docs",
            "16",
            "--phase1-seeds",
            "0",
            "--phase2-seeds",
            "0",
            "1",
            "--no-use-cuda",
        ]
    )
    variant = tfpush.SURROGATE_VARIANTS[0]
    base_config = tfpush._make_stage1_config(
        args,
        train_doc_count=int(args.phase1_train_docs),
        variant=variant,
    )
    jobs = tfpush._build_stage2_jobs(
        args=args,
        stage1_runs=[
            {
                "config_label": str(variant["label"]),
                "seed": 0,
                "tree_stage1_artifact_dir": "/tmp/stage1_artifact",
            }
        ],
        base_configs={str(variant["label"]): base_config},
    )

    assert len(jobs) == len(tfpush.STAGE2_JUDGE_CONDITIONS) * len(args.phase2_seeds)
    assert {job.config.tree_stage1_artifact_dir for job in jobs} == {
        "/tmp/stage1_artifact"
    }
    assert {job.config.tree_stage1_epochs for job in jobs} == {0}
    assert {job.config.tree_stage2_epochs for job in jobs} == {int(args.stage2_epochs)}


def test_build_surrogate_variants_expands_root_search_grid() -> None:
    args = tfpush._parser().parse_args(
        [
            "--benchmark",
            "smoke",
            "--surrogate-labels",
            "teacherfirst_shared_feature_adapters_phi128",
            "--root-search-labels",
            "teacherfirst_shared_feature_adapters_phi128",
            "--stage1-root-weight-grid",
            "0.25",
            "0.5",
            "--no-use-cuda",
        ]
    )
    variants = tfpush._build_surrogate_variants(args)
    labels = {variant["label"] for variant in variants}

    assert "teacherfirst_shared_feature_adapters_phi128" in labels
    assert "teacherfirst_shared_feature_adapters_phi128_root0p25" in labels
    assert "teacherfirst_shared_feature_adapters_phi128_root0p50" in labels
    assert "teacherfirst_shared_feature_phi128" not in labels
    assert "teacherfirst_shared_feature_phi192_root0p50" not in labels


def test_make_stage1_config_applies_variant_root_weight_and_checkpoint_metric() -> None:
    args = tfpush._parser().parse_args(
        [
            "--benchmark",
            "smoke",
            "--phase1-train-docs",
            "8",
            "--tree-stage1-eval-mode",
            "end_only",
            "--tree-stage1-screen-doc-limit",
            "16",
            "--root-search-labels",
            "teacherfirst_shared_feature_adapters_phi128",
            "--stage1-root-weight-grid",
            "0.5",
            "--no-use-cuda",
        ]
    )
    variant = next(
        item
        for item in tfpush._build_surrogate_variants(args)
        if item["label"] == "teacherfirst_shared_feature_adapters_phi128_root0p50"
    )
    config = tfpush._make_stage1_config(
        args,
        train_doc_count=int(args.phase1_train_docs),
        variant=variant,
    )

    assert config.tree_stage1_root_weight == 0.5
    assert config.tree_stage1_checkpoint_metric == "val_root_mae"
    assert config.tree_stage1_eval_mode == "end_only"
    assert config.tree_stage1_screen_doc_limit == 16


def test_base_args_propagates_batch_controls() -> None:
    args = tfpush._parser().parse_args(
        [
            "--benchmark",
            "smoke",
            "--batch-token-budget",
            "4096",
            "--batch-node-budget",
            "512",
            "--no-batch-autotune",
            "--eval-workers-per-mig",
            "2",
            "--no-use-cuda",
        ]
    )

    base_args = tfpush._base_args(args)

    assert base_args.tree_batch_pack_mode == "structure_bucket"
    assert base_args.tree_batch_token_budget == 4096
    assert base_args.tree_batch_node_budget == 512
    assert base_args.tree_batch_autotune is False
    assert base_args.tree_eval_workers_per_mig == 2


def test_base_args_defaults_to_fixed_fused_for_recoverable_v4() -> None:
    args = tfpush._parser().parse_args(
        [
            "--benchmark",
            "recoverable_v4",
            "--no-use-cuda",
        ]
    )

    base_args = tfpush._base_args(args)

    assert base_args.tree_batch_pack_mode == "fixed_fused"


def test_build_grouped_stage2_jobs_collapses_conditions_per_candidate_seed() -> None:
    args = tfpush._parser().parse_args(
        [
            "--benchmark",
            "smoke",
            "--phase1-train-docs",
            "8",
            "--phase2-train-docs",
            "16",
            "--phase1-seeds",
            "0",
            "--phase2-seeds",
            "0",
            "1",
            "--no-use-cuda",
        ]
    )
    variant = tfpush.SURROGATE_VARIANTS[0]
    base_config = tfpush._make_stage1_config(
        args,
        train_doc_count=int(args.phase1_train_docs),
        variant=variant,
    )
    flat_jobs = tfpush._build_stage2_jobs(
        args=args,
        stage1_runs=[
            {
                "config_label": str(variant["label"]),
                "seed": 0,
                "tree_stage1_artifact_dir": "/tmp/stage1_artifact",
            }
        ],
        base_configs={str(variant["label"]): base_config},
    )

    grouped = tfpush._build_grouped_stage2_jobs(flat_jobs)

    assert len(grouped) == len(args.phase2_seeds)
    assert {
        len(list(item.get("jobs", ())))
        for item in grouped
    } == {len(tfpush.STAGE2_JUDGE_CONDITIONS)}
    assert {
        str(item.get("candidate_label", ""))
        for item in grouped
    } == {str(variant["label"])}
    reconstructed = [tfpush._job_from_mapping(mapping) for mapping in grouped[0]["jobs"]]
    assert {
        str(job.config.label).split("__", 1)[0]
        for job in reconstructed
    } == {str(variant["label"])}


def test_grouped_stage2_worker_main_emits_small_completion_record(
    tmp_path, monkeypatch, capsys
) -> None:
    def _fake_worker(**_: object) -> dict[str, object]:
        return {
            "job_name": "grouped_job",
            "condition_results": [{"payload": "x" * 50_000}],
        }

    monkeypatch.setattr(tfpush, "_run_grouped_stage2_worker", _fake_worker)
    manifest_path = tmp_path / "group_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "grouped"

    monkeypatch.setattr(
        tfpush.sys,
        "argv",
        [
            "run_tree_neural_teacher_first_push.py",
            "--grouped-stage2-worker-manifest",
            str(manifest_path),
            "--grouped-stage2-worker-output-dir",
            str(output_dir),
            "--grouped-stage2-worker-job-name",
            "grouped_job",
            "--no-use-cuda",
        ],
    )

    rc = tfpush.main()

    assert rc == 0
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["job_name"] == "grouped_job"
    assert payload["grouped_stage2_summary"].endswith(
        tfpush.GROUPED_STAGE2_SUMMARY_FILENAME
    )
    assert "condition_results" not in out
    assert len(out) < 512


def test_run_grouped_stage2_worker_records_elapsed_times(tmp_path, monkeypatch) -> None:
    job_mapping = {
        "family": "tree_neural",
        "train_doc_count": 128,
        "benchmark": "smoke",
        "hardness_grid": "",
        "grid_cell_ids": [],
        "seeds": [0],
        "tuning_stage": "stage2_judge",
        "study_name": "teacher_first_tournament",
        "study_axis": "stage2_judge_config",
        "axis_value": "candidate__leaf_dense__judge_t128",
        "selection_metric": "teacher_first_total_bound",
        "config": {
            "label": "candidate__leaf_dense__judge_t128",
            "state_dim": 8,
            "hidden_dim": 16,
            "n_epochs": 1,
            "batch_size": 1,
            "lr": 1e-3,
            "weight_decay": 0.0,
        },
    }
    manifest_path = tmp_path / "group_manifest.json"
    manifest_path.write_text(json.dumps({"jobs": [job_mapping]}), encoding="utf-8")
    output_dir = tmp_path / "grouped"

    monkeypatch.setattr(
        tfpush.mig,
        "_worker_payload",
        lambda _args: {"test_root_mae": 0.25},
    )

    payload = tfpush._run_grouped_stage2_worker(
        manifest_path=manifest_path,
        output_dir=output_dir,
        job_name="grouped_job",
        use_cuda=False,
        torch_threads=1,
    )
    summary = json.loads(
        tfpush._grouped_stage2_summary_path(output_dir).read_text(encoding="utf-8")
    )

    assert payload["elapsed_s"] >= 0.0
    assert payload["condition_results"][0]["elapsed_s"] >= 0.0
    assert summary["condition_results"][0]["payload"]["test_root_mae"] == 0.25


def test_run_grouped_stage2_phase_terminates_lingering_worker_after_summary(
    tmp_path, monkeypatch
) -> None:
    class _FakePopen:
        instances: list["_FakePopen"] = []

        def __init__(self, cmd, stdout=None, stderr=None, cwd=None, env=None):
            self.cmd = list(cmd)
            self.stdout = stdout
            self.stderr = stderr
            self.cwd = cwd
            self.env = env
            self.pid = 4242 + len(self.instances)
            self.returncode = None
            self.terminated = False
            output_dir = Path(
                self.cmd[self.cmd.index("--grouped-stage2-worker-output-dir") + 1]
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            tfpush._grouped_stage2_summary_path(output_dir).write_text(
                json.dumps(
                    {
                        "job_name": "grouped_job",
                        "manifest": {},
                        "condition_results": [],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            self.instances.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = 0

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired(self.cmd, timeout)
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(tfpush.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(tfpush, "GROUPED_STAGE2_COMPLETION_GRACE_S", 0.0)
    monkeypatch.setattr(
        tfpush.mig,
        "_write_summary_outputs",
        lambda _output_root: {
            "runs": [
                {
                    "config_label": "candidate__leaf_dense__judge_t128",
                    "seed": 0,
                    "train_doc_count": 128,
                    "teacher_first_total_bound": 1.0,
                    "stage1_substitution_cost": 0.5,
                    "test_root_mae": 0.25,
                    "stage2_transport_budget": 0.1,
                }
            ]
        },
    )
    args = tfpush._parser().parse_args(["--benchmark", "smoke", "--no-use-cuda"])

    result = tfpush._run_grouped_stage2_phase(
        output_root=tmp_path / "phase2",
        grouped_jobs=[
            {
                "job_name": "grouped_job",
                "candidate_label": "candidate",
                "jobs": [],
            }
        ],
        args=args,
        manifest_payload={"jobs": []},
    )

    assert result["result"]["failed_jobs"] == []
    assert result["result"]["completed_jobs"][0]["job_name"] == "grouped_job"
    assert _FakePopen.instances[0].terminated is True
