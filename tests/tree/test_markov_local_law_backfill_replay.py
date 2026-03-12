from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from src.tree.markov_changepoint_ops_count_simulation import (
    OPSCountConfig,
    run_markov_changepoint_ops_count_experiment,
)


def test_markov_local_law_replay_smoke(tmp_path: Path) -> None:
    out_path = tmp_path / "seed_0.json"
    cfg = OPSCountConfig(
        n_regimes=3,
        vocab_size=32,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=4,
        test_docs=4,
        model_family="neural",
        feature_mode="full",
        state_dim=8,
        hidden_dim=16,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=1.0,
        include_root_query=True,
        local_law_weight=0.5,
        data_seed=1,
        model_seed=2,
        seed=2,
        use_cuda=False,
        torch_threads=1,
    )
    summary = run_markov_changepoint_ops_count_experiment(cfg)
    out_path.write_text(summary.to_json(), encoding="utf-8")

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    learned = payload["metrics"]["learned"]
    learned.pop("test_objective_full_labels", None)
    learned.pop("test_objective_root_term", None)
    learned.pop("test_objective_leaf_term", None)
    learned.pop("test_objective_merge_term", None)
    learned.pop("test_objective_schedule_consistency_term", None)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    subprocess.check_call(
        [
            sys.executable,
            "scripts/replay_markov_changepoint_ops_count_summary.py",
            "--summary-json",
            str(out_path),
            "--device",
            "cpu",
            "--torch-threads",
            "1",
        ],
        cwd=repo_root,
    )

    replayed = json.loads(out_path.read_text(encoding="utf-8"))
    learned_replayed = replayed["metrics"]["learned"]
    assert "test_objective_full_labels" in learned_replayed
    assert "train_objective_full_labels" in learned_replayed
