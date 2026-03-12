from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def _markov_payload(*, family: str, q_train: float, hidden_dim: int = 128) -> dict:
    return {
        "config": {
            "model_family": family,
            "audit_fraction": q_train,
            "train_docs": 8000,
            "feature_mode": "full",
            "state_dim": 32,
            "hidden_dim": hidden_dim,
            "local_law_weight": 0.2,
            "schedule_consistency_weight": 0.0,
            "guidance_override_mode": "reset",
            "seed": 0,
        },
        "objective": {"local_law_weight": 0.2},
        "metrics": {
            "learned": {
                "root_mae": 1.0 if family == "neural" else 1e-5,
                "merge_mae": 0.5 if family == "neural" else 1e-6,
                "schedule_spread_mean": 5.0 if family == "neural" else 1e-5,
            },
            "guided_eval_curve": {
                "points": [
                    {
                        "q": 0.0,
                        "root_mae": 1.0 if family == "neural" else 1e-5,
                        "merge_mae": 0.5 if family == "neural" else 1e-6,
                        "effective_q_mean": 0.0,
                        "guided_internal_nodes_mean": 0.0,
                    },
                    {
                        "q": 0.5,
                        "root_mae": 0.25 if family == "neural" else 2e-6,
                        "merge_mae": 0.15 if family == "neural" else 2e-7,
                        "effective_q_mean": 0.5,
                        "guided_internal_nodes_mean": 12.0,
                    },
                    {
                        "q": 1.0,
                        "root_mae": 0.0,
                        "merge_mae": 0.0,
                        "effective_q_mean": 1.0,
                        "guided_internal_nodes_mean": 23.0,
                    },
                ]
            },
        },
    }


def _ctree_payload(*, estimator: str, seed_topics: int, q_infer: float, root_l1: float) -> dict:
    return {
        "config": {
            "topic_phi_docs": 4096,
            "topic_phi_estimator": estimator,
            "eval_leaf_query_rate": q_infer,
            "eval_internal_query_rate": q_infer,
            "seed": seed_topics,
        },
        "metrics": {
            "estimated_calibrated_budgeted": {
                "root_l1_mean": root_l1,
            }
        },
        "topic_meta": {
            "topic_phi_neural_seed_count": seed_topics,
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_fake_roots(tmp_path: Path, *, mismatched_publication_hidden_dim: bool = False) -> tuple[Path, Path, Path]:
    learn_root = tmp_path / "learnability"
    pub_root = tmp_path / "publication_clean"
    overnight_root = tmp_path / "neural_overnight"

    for q in (0.02, 0.1, 1.0):
        _write_json(
            learn_root / "markov_changepoint_ops_count" / "family_neural" / f"q_{str(q).replace('.', 'p')}.json",
            _markov_payload(family="neural", q_train=q),
        )
        _write_json(
            learn_root / "markov_changepoint_ops_count" / "family_additive" / f"q_{str(q).replace('.', 'p')}.json",
            _markov_payload(family="additive", q_train=q),
        )

    pub_hidden_dim = 64 if mismatched_publication_hidden_dim else 128
    for family in ("neural", "additive"):
        _write_json(
            pub_root / "markov_changepoint_ops_count" / f"family_{family}" / "seed_0.json",
            _markov_payload(family=family, q_train=1.0, hidden_dim=pub_hidden_dim),
        )

    for estimator, base in (
        ("neural_ctreepo", 0.6),
        ("neural_hybrid", 0.4),
        ("neural_mergeable_sketch", 0.08),
    ):
        for q, val in ((0.0, base), (0.5, base / 10.0), (1.0, 0.0)):
            _write_json(
                overnight_root
                / "segmented_lda_ctreepo"
                / estimator
                / f"q_{str(q).replace('.', 'p')}.json",
                _ctree_payload(estimator=estimator, seed_topics=4, q_infer=q, root_l1=val),
            )

    diag = {
        "diagnostics": {
            "neural_lag_evidence": {
                "ctree_reference": {
                    "series": {
                        "0": {"root_l1_mean": 0.6},
                        "0.5": {"root_l1_mean": 0.06},
                        "1": {"root_l1_mean": 0.0},
                    }
                },
                "markov_additive": {
                    "0": {"root_mae": 1e-5},
                    "0.5": {"root_mae": 2e-6},
                    "1": {"root_mae": 0.0},
                },
                "fixed_slice": {
                    "ctree": {"learn_time_oracle_visibility": 0.1, "train_docs": 4096},
                    "markov": {"learn_time_oracle_visibility": 1.0, "train_docs": 8000},
                },
            }
        }
    }
    _write_json(pub_root / "figures" / "identifiable_zero_publication_report_latest_diagnostics.json", diag)
    return learn_root, pub_root, overnight_root


def test_operator_learning_slide_deck_smoke(tmp_path: Path) -> None:
    learn_root, pub_root, overnight_root = _build_fake_roots(tmp_path)
    out_dir = tmp_path / "report"
    asset_dir = tmp_path / "assets"

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_operator_learning_slide_deck.py",
            "--learnability-root",
            str(learn_root),
            "--publication-clean-root",
            str(pub_root),
            "--neural-overnight-root",
            str(overnight_root),
            "--out-dir",
            str(out_dir),
            "--figure-asset-dir",
            str(asset_dir),
            "--no-emit-pdf",
        ],
        cwd=REPO_ROOT,
        env=env,
    )

    summary = json.loads((out_dir / "operator_learning_slide_deck_summary.json").read_text(encoding="utf-8"))
    deck_tex = (out_dir / "operator_learning_slide_deck.tex").read_text(encoding="utf-8")

    assert summary["canonical_markov_slice"]["train_docs"] == 8000
    assert summary["bridge_series"]["ctree_norm"] == [1.0, 0.1, 0.0]
    assert (asset_dir / "operator_info_axes_slide.tex").exists()
    assert (asset_dir / "markov_neural_operator_slide.tex").exists()
    assert (asset_dir / "ctree_operator_bridge_slide.tex").exists()
    assert "markov_changepoint_exact_merge_slide.pdf" in deck_tex
    assert "ctree_operator_bridge_slide.pdf" in deck_tex


def test_operator_learning_slide_deck_rejects_mismatched_publication_slice(tmp_path: Path) -> None:
    learn_root, pub_root, overnight_root = _build_fake_roots(
        tmp_path, mismatched_publication_hidden_dim=True
    )
    out_dir = tmp_path / "report_bad"
    asset_dir = tmp_path / "assets_bad"

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/report_operator_learning_slide_deck.py",
            "--learnability-root",
            str(learn_root),
            "--publication-clean-root",
            str(pub_root),
            "--neural-overnight-root",
            str(overnight_root),
            "--out-dir",
            str(out_dir),
            "--figure-asset-dir",
            str(asset_dir),
            "--no-emit-pdf",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "publication-clean neural canonical slice" in (proc.stderr + proc.stdout)
