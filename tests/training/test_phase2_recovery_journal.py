import argparse
import json
import pickle
from pathlib import Path

from src.training.run_pipeline import (
    _build_phase2_runtime_signature,
    _export_gepa_state_artifacts,
)


class _DummyResult:
    def __init__(self, doc_id: str, reference_score: float, truth_label_source: str = "dataset") -> None:
        self.doc_id = doc_id
        self.reference_score = reference_score
        self.truth_label_source = truth_label_source


def _make_args() -> argparse.Namespace:
    return argparse.Namespace(
        optimizer="gepa",
        optimizer_budget="light",
        max_metric_calls=0,
        num_threads=2,
        data_seed=7,
        n_iterations=3,
        skip_oracle_opt=False,
        skip_summarizer_opt=False,
        gepa_reflection_minibatch_size=3,
        gepa_leaf_merge_sampling_design="two_stage_pps_bernoulli",
        gepa_ipw_estimator="hajek",
        gepa_ipw_min_propensity=1e-6,
        convergence_threshold=0.001,
        convergence_patience=2,
    )


def test_phase2_runtime_signature_stable_and_sensitive() -> None:
    args = _make_args()
    train = [_DummyResult("doc_a", 0.2), _DummyResult("doc_b", 0.7)]
    val = [_DummyResult("doc_c", 0.4)]

    sig_1, id_1 = _build_phase2_runtime_signature(
        args=args,
        task_name="manifesto_rile",
        train_results=train,
        val_results=val,
    )
    sig_2, id_2 = _build_phase2_runtime_signature(
        args=args,
        task_name="manifesto_rile",
        train_results=train,
        val_results=val,
    )
    assert sig_1 == sig_2
    assert id_1 == id_2

    changed_train = [_DummyResult("doc_a", 0.21), _DummyResult("doc_b", 0.7)]
    _, id_3 = _build_phase2_runtime_signature(
        args=args,
        task_name="manifesto_rile",
        train_results=changed_train,
        val_results=val,
    )
    assert id_3 != id_1

    args_design_changed = _make_args()
    args_design_changed.gepa_leaf_merge_sampling_design = "srswor"
    _, id_4 = _build_phase2_runtime_signature(
        args=args_design_changed,
        task_name="manifesto_rile",
        train_results=train,
        val_results=val,
    )
    assert id_4 != id_1

    args_estimator_changed = _make_args()
    args_estimator_changed.gepa_ipw_estimator = "horvitz_thompson"
    _, id_5 = _build_phase2_runtime_signature(
        args=args_estimator_changed,
        task_name="manifesto_rile",
        train_results=train,
        val_results=val,
    )
    assert id_5 != id_1


def test_export_gepa_state_artifacts_writes_snapshot_and_prompt_trajectory(tmp_path: Path) -> None:
    log_dir = tmp_path / "gepa" / "scorer"
    log_dir.mkdir(parents=True, exist_ok=True)
    phase2_runtime_dir = tmp_path / "runtime_exports"

    fake_state = {
        "program_candidates": [
            {"predictor_a": "base instruction"},
            {"predictor_a": "improved instruction"},
        ],
        "program_full_scores_val_set": [0.5, 0.75],
        "parent_program_for_candidate": [[None], [0]],
        "num_metric_calls_by_discovery": [0, 18],
        "full_program_trace": [{"i": 0}, {"i": 1, "new_program_idx": 1}],
        "i": 1,
        "total_num_evals": 42,
        "num_full_ds_evals": 4,
    }
    with open(log_dir / "gepa_state.bin", "wb") as f:
        pickle.dump(fake_state, f)

    exported = _export_gepa_state_artifacts(
        log_dir=log_dir,
        component="scorer",
        phase2_runtime_dir=phase2_runtime_dir,
    )

    assert exported["available"] is True
    assert exported["num_candidates"] == 2
    assert Path(exported["snapshot_path"]).exists()
    assert Path(exported["prompt_trajectory_path"]).exists()
    assert Path(exported["runtime_snapshot_path"]).exists()
    assert Path(exported["runtime_prompt_trajectory_path"]).exists()

    with open(exported["snapshot_path"], "r") as f:
        snapshot = json.load(f)
    assert snapshot["best_candidate_idx"] == 1
    assert snapshot["best_candidate_score"] == 0.75
    assert snapshot["total_metric_calls"] == 42
    assert len(snapshot["full_program_trace"]) == 2

    with open(exported["prompt_trajectory_path"], "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 2
    assert rows[1]["candidate_idx"] == 1
    assert rows[1]["score"] == 0.75
    assert rows[1]["instructions"]["predictor_a"] == "improved instruction"
