from __future__ import annotations

import json
from pathlib import Path

from src.ctreepo.sim.cli.report.publication_ctreepo_progress import main as report_main


def test_publication_ctreepo_progress_report_smoke(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    row_path = (
        out_root
        / "segmented_lda_ctreepo"
        / "equivalence"
        / "lda"
        / "k8_v512"
        / "lane_test"
        / "phi_true"
        / "train_128"
        / "seed_0.json"
    )
    row_path.parent.mkdir(parents=True, exist_ok=True)
    row_path.write_text(
        json.dumps(
            {
                "config": {
                    "n_books_train": 128,
                    "fixed_leaf_tokens": 16,
                    "calibration_leaf_query_rate": 0.1,
                    "eval_leaf_query_rate": 0.0,
                    "eval_internal_query_rate": 0.0,
                    "seed": 0,
                },
                "metrics": {"estimated_calibrated_budgeted": {"root_l1_mean": 0.123}},
                "topic_meta": {"topic_phi_l2_error_mean": 0.5},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rc = int(report_main(["--output-root", str(out_root), "--no-emit-pdf"]))
    assert rc == 0

    figs = out_root / "figures" / "publication_progress"
    assert (figs / "publication_ctreepo_progress_diagnostics.json").exists()
    assert (figs / "publication_ctreepo_progress_latest.md").exists()
    assert (figs / "pages" / "progress_counts_by_lane.png").exists()

