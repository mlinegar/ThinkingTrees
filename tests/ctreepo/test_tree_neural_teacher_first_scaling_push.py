from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess

from scripts import run_tree_neural_full_doc_mig as mig
from scripts import run_tree_neural_teacher_first_push as tfpush
from scripts import run_tree_neural_teacher_first_scaling_push as scaling


def test_frontier_variants_cover_expected_root_weights() -> None:
    variants = scaling._frontier_variants()
    labels = {str(variant["label"]) for variant in variants}

    assert "teacherfirst_shared_feature_adapters_phi128" in labels
    assert "teacherfirst_shared_feature_adapters_phi128_root0p50" in labels
    assert "teacherfirst_shared_feature_adapters_phi128_root1p00" in labels
    assert "teacherfirst_shared_feature_phi192" in labels
    assert "teacherfirst_shared_feature_phi192_root0p50" in labels
    assert "teacherfirst_scorefiber_s1_f15" in labels
    assert "teacherfirst_scorefiber_s1_f15_root0p50" in labels
    assert "teacherfirst_scorefiber_s1_f31" in labels
    assert "teacherfirst_scorefiber_s1_f31_root0p50" in labels
    assert len(variants) == 9


def test_count_args_scales_phase2_docs_with_multiplier() -> None:
    args = scaling._parser().parse_args(
        [
            "--phase2-train-multiplier",
            "1.5",
            "--phase1-seeds",
            "0",
            "--phase2-seeds",
            "1",
        ]
    )

    count_args = scaling._count_args(args, train_doc_count=128)

    assert count_args.phase1_train_docs == 128
    assert count_args.phase2_train_docs == 192
    assert count_args.phase1_seeds == (0,)
    assert count_args.phase2_seeds == (1,)
    assert count_args.tree_stage1_eval_mode == "end_only"
    assert count_args.tree_stage1_screen_doc_limit == 16
    assert count_args.stage2_epochs == 4
    assert count_args.group_stage2_conditions is True


def test_count_args_propagates_batch_controls() -> None:
    args = scaling._parser().parse_args(
        [
            "--train-doc-counts",
            "128",
            "--batch-token-budget",
            "4096",
            "--batch-node-budget",
            "512",
            "--no-batch-autotune",
            "--eval-workers-per-mig",
            "2",
        ]
    )

    count_args = scaling._count_args(args, train_doc_count=128)

    assert count_args.tree_batch_pack_mode == "fixed_fused"
    assert count_args.batch_token_budget == 4096
    assert count_args.batch_node_budget == 512
    assert count_args.batch_autotune is False
    assert count_args.eval_workers_per_mig == 2


def test_resolved_stage2_policy_maps_use_defaults_and_overrides() -> None:
    args = scaling._parser().parse_args(
        [
            "--train-doc-counts",
            "128",
            "512",
            "1024",
            "--stage2-epochs-by-count",
            "512:9",
            "--stage2-survivors-by-count",
            "1024:3",
        ]
    )

    epochs_by_count = scaling._resolved_stage2_epochs_by_count(args)
    survivors_by_count = scaling._resolved_stage2_survivors_by_count(args)

    assert epochs_by_count == {128: 4, 512: 9, 1024: 8}
    assert survivors_by_count == {128: 1, 512: 1, 1024: 3}


def test_runs_for_train_doc_count_filters_exact_matches() -> None:
    runs = [
        {"train_doc_count": 128, "config_label": "a"},
        {"train_doc_count": 512, "config_label": "b"},
        {"train_doc_count": 128, "config_label": "c"},
    ]

    filtered = scaling._runs_for_train_doc_count(runs, train_doc_count=128)

    assert [run["config_label"] for run in filtered] == ["a", "c"]


def test_resolved_stage1_rungs_default_halving_schedule() -> None:
    args = scaling._parser().parse_args([])

    rungs = scaling._resolved_stage1_rungs(args, variant_count=5)

    assert [(rung.index, rung.total_epochs, rung.promote_k) for rung in rungs] == [
        (1, 2, 3),
        (2, 6, 2),
        (3, 12, None),
    ]


def test_resolved_stage1_rungs_single_rung_matches_exhaustive_mode() -> None:
    args = scaling._parser().parse_args(["--stage1-rung-epochs", "12"])

    rungs = scaling._resolved_stage1_rungs(args, variant_count=5)

    assert [(rung.index, rung.total_epochs, rung.promote_k) for rung in rungs] == [
        (1, 12, None),
    ]


def test_aggregate_stage1_rung_candidate_summary_breaks_ties_by_existing_rank_key() -> None:
    runs = [
        {
            "train_doc_count": 128,
            "config_label": "candidate_a",
            "seed": 0,
            "val_root_mae": 0.5,
            "teacher_first_total_bound": 1.2,
            "stage1_substitution_cost": 1.0,
            "test_root_mae": 0.8,
            "stage2_transport_budget": 0.2,
            "tree_stage1_root_weight": 0.0,
            "tree_stage1_artifact_dir": "artifacts/a",
        },
        {
            "train_doc_count": 128,
            "config_label": "candidate_b",
            "seed": 0,
            "val_root_mae": 0.5,
            "teacher_first_total_bound": 0.9,
            "stage1_substitution_cost": 0.7,
            "test_root_mae": 0.6,
            "stage2_transport_budget": 0.1,
            "tree_stage1_root_weight": 0.5,
            "tree_stage1_artifact_dir": "artifacts/b",
        },
    ]

    summary = scaling._aggregate_stage1_rung_candidate_summary(
        runs,
        screen_metric_name="val_root_mae",
    )

    assert [row["candidate_label"] for row in summary] == ["candidate_b", "candidate_a"]


def test_build_stage1_rung_jobs_spans_all_requested_counts() -> None:
    args = scaling._parser().parse_args(["--train-doc-counts", "128", "512"])
    train_doc_counts = [128, 512]
    count_args_by_train = {
        count: scaling._count_args(args, train_doc_count=count)
        for count in train_doc_counts
    }
    variants = scaling._frontier_variants()
    stage1_configs_by_train = {
        count: {
            str(variant["label"]): tfpush._make_stage1_config(
                count_args_by_train[count],
                train_doc_count=count,
                variant=variant,
            )
            for variant in variants
        }
        for count in train_doc_counts
    }
    active_labels_by_count = {
        count: [str(variant["label"]) for variant in variants]
        for count in train_doc_counts
    }

    jobs = scaling._build_stage1_rung_jobs(
        args=args,
        train_doc_counts=train_doc_counts,
        count_args_by_train=count_args_by_train,
        stage1_configs_by_train=stage1_configs_by_train,
        active_labels_by_count=active_labels_by_count,
        rung=scaling._Stage1RungSpec(index=1, total_epochs=2, promote_k=3),
        screen_metric_name="val_root_mae",
    )

    assert len(jobs) == len(variants) * len(train_doc_counts)
    assert {int(job.train_doc_count) for job in jobs} == {128, 512}
    assert {int(job.config.n_epochs) for job in jobs} == {2}
    assert {int(job.config.tree_stage1_epochs) for job in jobs} == {2}


def test_build_stage2_jobs_for_counts_uses_only_final_survivors() -> None:
    args = scaling._parser().parse_args(["--train-doc-counts", "128"])
    count_args_by_train = {128: scaling._count_args(args, train_doc_count=128)}
    variants = scaling._frontier_variants()
    stage1_configs_by_train = {
        128: {
            str(variant["label"]): tfpush._make_stage1_config(
                count_args_by_train[128],
                train_doc_count=128,
                variant=variant,
            )
            for variant in variants
        }
    }
    active_labels_by_count = {
        128: [
            "teacherfirst_shared_feature_adapters_phi128",
            "teacherfirst_shared_feature_phi192",
        ]
    }
    final_stage1_runs = [
        {
            "train_doc_count": 128,
            "config_label": "teacherfirst_shared_feature_adapters_phi128",
            "seed": 0,
            "tree_stage1_artifact_dir": "artifacts/a",
        },
        {
            "train_doc_count": 128,
            "config_label": "teacherfirst_shared_feature_phi192",
            "seed": 0,
            "tree_stage1_artifact_dir": "artifacts/b",
        },
        {
            "train_doc_count": 128,
            "config_label": "teacherfirst_shared_feature_adapters_phi128_root0p50",
            "seed": 0,
            "tree_stage1_artifact_dir": "artifacts/c",
        },
    ]

    jobs = scaling._build_stage2_jobs_for_counts(
        train_doc_counts=(128,),
        count_args_by_train=count_args_by_train,
        stage1_configs_by_train=stage1_configs_by_train,
        active_labels_by_count=active_labels_by_count,
        final_stage1_runs=final_stage1_runs,
        stage2_survivors_by_count={128: 1},
    )

    expected_labels = {"teacherfirst_shared_feature_adapters_phi128"}
    assert len(jobs) == len(expected_labels) * len(tfpush.STAGE2_JUDGE_CONDITIONS)
    assert {
        str(job.config.label).split("__", 1)[0]
        for job in jobs
    } == expected_labels
    assert {int(job.config.tree_stage2_epochs) for job in jobs} == {4}


def test_write_stage1_rung_summary_contains_rung_history(tmp_path) -> None:
    args = scaling._parser().parse_args([])
    count_args = scaling._count_args(args, train_doc_count=128)
    rung_specs = scaling._resolved_stage1_rungs(args, variant_count=5)
    count_root = tmp_path / "train_128"
    count_root.mkdir(parents=True, exist_ok=True)

    scaling._write_stage1_rung_summary(
        count_root,
        args=count_args,
        stage1_screen_metric="val_root_mae",
        rung_specs=rung_specs,
        rung_history=[
            {
                "rung_index": 1,
                "total_epochs": 2,
                "active_candidates": ["a", "b", "c"],
                "promoted_candidates": ["a", "b"],
                "candidate_summary": [
                    {"candidate_label": "a", "mean_screen_metric": 0.3, "mean_teacher_first_total_bound": 0.8},
                    {"candidate_label": "b", "mean_screen_metric": 0.4, "mean_teacher_first_total_bound": 0.9},
                ],
            }
        ],
        final_survivors=["a", "b"],
    )

    payload = json.loads((count_root / "stage1_rung_summary.json").read_text())

    assert payload["final_survivors"] == ["a", "b"]
    assert payload["rungs"][0]["promoted_candidates"] == ["a", "b"]


def test_make_stage1_config_populates_factorized_scorefiber_fields() -> None:
    args = tfpush._parser().parse_args(
        [
            "--benchmark",
            "smoke",
            "--phase1-train-docs",
            "8",
            "--no-use-cuda",
        ]
    )
    variant = next(
        item
        for item in tfpush.SURROGATE_VARIANTS
        if item["label"] == "teacherfirst_scorefiber_s1_f15"
    )

    config = tfpush._make_stage1_config(
        args,
        train_doc_count=int(args.phase1_train_docs),
        variant=variant,
    )

    assert config.tree_theorem_surface_mode == "factorized_score_fiber"
    assert config.theorem_feature_adapter == "markov_score_endpoints"
    assert config.tree_theorem_score_dim == 1
    assert config.tree_theorem_fiber_dim == 15
    assert config.tree_theorem_aux_dim == 0
    assert config.tree_score_merge_mode == "gated_affine"


def test_main_dispatches_to_async_scheduler(tmp_path, monkeypatch) -> None:
    called: dict[str, object] = {}

    def _fake_async(**kwargs):
        called.update(kwargs)
        output_root = kwargs["output_root"]
        (output_root / "teacher_first_scaling_summary.json").write_text(
            json.dumps({"scaling_rows": []}),
            encoding="utf-8",
        )

    monkeypatch.setattr(scaling, "_run_async_scaling", _fake_async)

    rc = scaling.main(
        [
            "--output-root",
            str(tmp_path / "async"),
            "--train-doc-counts",
            "128",
            "--no-use-cuda",
        ]
    )

    assert rc == 0
    assert int(called["rung_specs"][0].total_epochs) == 2
    assert str(called["screen_metric_name"]) == "val_root_mae"
    assert str(called["output_root"]).endswith("async")


def test_async_scaling_finalizes_grouped_worker_from_summary_file(
    tmp_path, monkeypatch
) -> None:
    args = scaling._parser().parse_args(
        [
            "--output-root",
            str(tmp_path / "async"),
            "--benchmark",
            "smoke",
            "--train-doc-counts",
            "128",
            "--phase1-seeds",
            "0",
            "--phase2-seeds",
            "0",
            "--stage1-rung-epochs",
            "1",
            "--stage2-survivors-by-count",
            "128:1",
            "--no-use-cuda",
        ]
    )

    def _fake_make_stage1_config(_count_args, *, train_doc_count, variant):
        return mig._RunConfigSpec(
            label=str(variant["label"]),
            state_dim=8,
            hidden_dim=16,
            n_epochs=1,
            batch_size=1,
            lr=1e-3,
            weight_decay=0.0,
        )

    def _fake_stage1_job(label: str) -> mig._JobSpec:
        return mig._JobSpec(
            family="tree_neural",
            train_doc_count=128,
            benchmark="smoke",
            hardness_grid="",
            grid_cell_ids=(),
            seeds=(0,),
            config=mig._RunConfigSpec(
                label=label,
                state_dim=8,
                hidden_dim=16,
                n_epochs=1,
                batch_size=1,
                lr=1e-3,
                weight_decay=0.0,
            ),
            tuning_stage="stage1_surrogate",
            study_name="teacher_first_tournament",
            study_axis="stage1_surrogate",
            axis_value=label,
            selection_metric="teacher_first_total_bound",
        )

    def _fake_stage2_job(label: str) -> mig._JobSpec:
        return mig._JobSpec(
            family="tree_neural",
            train_doc_count=128,
            benchmark="smoke",
            hardness_grid="",
            grid_cell_ids=(),
            seeds=(0,),
            config=mig._RunConfigSpec(
                label=f"{label}__leaf_dense__judge_t128",
                state_dim=8,
                hidden_dim=16,
                n_epochs=1,
                batch_size=1,
                lr=1e-3,
                weight_decay=0.0,
            ),
            tuning_stage="stage2_judge",
            study_name="teacher_first_tournament",
            study_axis="stage2_judge_config",
            axis_value=f"{label}__leaf_dense__judge_t128",
            selection_metric="teacher_first_total_bound",
        )

    stage2_stdout_targets: list[object] = []

    class _FakePopen:
        def __init__(self, cmd, stdout=None, stderr=None, cwd=None, env=None, text=None):
            self.cmd = list(cmd)
            self.stdout = stdout
            self.stderr = stderr
            self.cwd = cwd
            self.env = env
            self.text = text
            self.pid = 5000
            self.returncode = None
            self._stage2 = "--grouped-stage2-worker-manifest" in self.cmd
            if self._stage2:
                stage2_stdout_targets.append(stdout)
                output_dir = tmp_path / "async" / "train_128" / "phase2" / "jobs" / "grouped_job"
                if "--grouped-stage2-worker-output-dir" in self.cmd:
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
                        }
                    ),
                    encoding="utf-8",
                )
            else:
                self.stdout = io.StringIO(json.dumps({"test_root_mae": 0.4}) + "\n")
                self.returncode = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired(self.cmd, timeout)
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(tfpush, "_make_stage1_config", _fake_make_stage1_config)
    monkeypatch.setattr(
        scaling,
        "_build_stage1_rung_jobs",
        lambda **_: [_fake_stage1_job("candidate")],
    )
    monkeypatch.setattr(
        scaling,
        "_build_stage2_jobs_for_counts",
        lambda **_: [_fake_stage2_job("candidate")],
    )
    monkeypatch.setattr(
        tfpush,
        "_build_grouped_stage2_jobs",
        lambda jobs: [
            {
                "job_name": "grouped_job",
                "candidate_label": "candidate",
                "jobs": [],
            }
        ],
    )
    monkeypatch.setattr(mig, "_load_completed_run_keys", lambda _root: set())
    monkeypatch.setattr(mig, "_worker_command_for_job", lambda *_, **__: ["fake-stage1"])
    monkeypatch.setattr(mig, "_worker_env_for_token", lambda _token, **__: {})
    monkeypatch.setattr(scaling.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(scaling.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(tfpush, "GROUPED_STAGE2_COMPLETION_GRACE_S", 0.0)

    def _fake_write_summary_outputs(output_root):
        root_str = str(output_root)
        if "phase1" in root_str:
            return {
                "runs": [
                    {
                        "train_doc_count": 128,
                        "config_label": "candidate",
                        "seed": 0,
                        "val_root_mae": 0.2,
                        "teacher_first_total_bound": 0.8,
                        "stage1_substitution_cost": 0.4,
                        "test_root_mae": 0.3,
                        "stage2_transport_budget": 0.1,
                        "tree_stage1_root_weight": 0.0,
                        "tree_stage1_artifact_dir": "/tmp/artifact",
                    }
                ]
            }
        return {
            "runs": [
                {
                    "train_doc_count": 128,
                    "config_label": "candidate__leaf_dense__judge_t128",
                    "seed": 0,
                    "teacher_first_total_bound": 0.7,
                    "stage1_substitution_cost": 0.4,
                    "test_root_mae": 0.25,
                    "stage2_transport_budget": 0.1,
                    "tree_stage1_root_weight": 0.0,
                    "tree_stage1_checkpoint_metric": "val_root_mae",
                }
            ]
        }

    monkeypatch.setattr(mig, "_write_summary_outputs", _fake_write_summary_outputs)

    result = scaling._run_async_scaling(
        args=args,
        output_root=tmp_path / "async",
        variants=[{"label": "candidate"}],
        rung_specs=[scaling._Stage1RungSpec(index=1, total_epochs=1, promote_k=None)],
        screen_metric_name="val_root_mae",
        mig_uuids=("cpu0",),
    )

    assert 128 in result["phase2_output_roots_by_count"]
    assert stage2_stdout_targets
    assert all(target is not subprocess.PIPE for target in stage2_stdout_targets)
    assert (tmp_path / "async" / "teacher_first_scaling_summary.json").exists()
