from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_tiny_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(0.4, 0.4), dpi=40)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow([[(0.29, 0.50, 0.71)]])
    ax.set_axis_off()
    fig.savefig(path, dpi=40)
    plt.close(fig)


def _fake_diag_payload() -> dict:
    return {
        "diagnostics": {
            "neural_lag_evidence": {
                "markov_additive": {
                    "0": {"root_mae": 6.359815597534179e-06},
                    "0.5": {"root_mae": 2.0623207092285156e-07},
                    "1": {"root_mae": 0.0},
                },
                "ctree_reference": {
                    "series": {
                        "0": {"root_l1_mean": 0.06390876538240056},
                        "0.5": {"root_l1_mean": 0.005873102528096362},
                        "1": {"root_l1_mean": 0.0},
                    }
                },
                "fixed_slice": {
                    "markov": {"learn_time_oracle_visibility": 1.0, "train_docs": 8000},
                    "ctree": {"learn_time_oracle_visibility": 0.1, "train_docs": 4096},
                },
                "observations": [
                    {
                        "evidence": {
                            "partial_gain_share_to_q05": {
                                "markov_additive": 0.9675726335520151,
                                "ctree": 0.9081017683105842,
                            }
                        }
                    }
                ],
            }
        }
    }


def _build_fake_publication_root(tmp_path: Path) -> Path:
    output_root = tmp_path / "identifiable_zero_longrun_clean"
    _write_json(
        output_root / "figures" / "identifiable_zero_publication_report_latest_diagnostics.json",
        _fake_diag_payload(),
    )
    _write_tiny_png(output_root / "figures" / "pub_clean" / "main_figure_B_gap_decomposition.png")

    _write_json(
        output_root / "markov_changepoint_ops_count" / "family_additive" / "seed_0.json",
        {
            "config": {
                "model_family": "additive",
                "train_docs": 8000,
                "audit_fraction": 1.0,
                "seed": 0,
            }
        },
    )
    _write_json(
        output_root / "segmented_lda_ctreepo" / "neural_ctreepo" / "seed_0.json",
        {
            "config": {
                "n_books_train": 4096,
                "calibration_leaf_query_rate": 0.1,
                "eval_leaf_query_rate": 1.0,
                "seed": 0,
            }
        },
    )
    return output_root


def _tex_class_available(name: str) -> bool:
    if shutil.which("kpsewhich") is None:
        return False
    proc = subprocess.run(
        ["kpsewhich", name],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def test_appendix_walkthrough_narrative_deck_smoke(tmp_path: Path) -> None:
    output_root = _build_fake_publication_root(tmp_path)
    out_dir = tmp_path / "deck_out"

    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_appendix_walkthrough_narrative_deck.py",
            "--output-root",
            str(output_root),
            "--out-dir",
            str(out_dir),
            "--no-emit-pdf",
        ],
        cwd=REPO_ROOT,
    )

    summary = json.loads((out_dir / "appendix_narrative_slide_deck_summary.json").read_text(encoding="utf-8"))
    deck_tex = (out_dir / "appendix_narrative_slide_deck.tex").read_text(encoding="utf-8")

    assert summary["slide_count"] == 18
    assert summary["frame_titles"][0] == "Appendix-to-Slides Reading Rule"
    assert "Markov DGP I: Here Are the Colors" in summary["frame_titles"]
    assert "C-TreePO II: Where the Unknowns Live" in summary["frame_titles"]
    assert r"\node<2->" in deck_tex
    assert "latent state IDs" in deck_tex
    assert "no explicit merge rule" in deck_tex
    assert "main_figure_B_gap_decomposition" in deck_tex


def test_appendix_walkthrough_narrative_deck_compile_smoke(tmp_path: Path) -> None:
    if shutil.which("latexmk") is None and shutil.which("pdflatex") is None:
        pytest.skip("LaTeX engine not available")
    if not _tex_class_available("beamer.cls"):
        pytest.skip("beamer.cls not available")

    output_root = _build_fake_publication_root(tmp_path)
    out_dir = tmp_path / "deck_pdf_out"

    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_appendix_walkthrough_narrative_deck.py",
            "--output-root",
            str(output_root),
            "--out-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
    )

    assert (out_dir / "appendix_narrative_slide_deck.pdf").exists()


def test_existing_appendix_script_is_archived() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/report_publication_clean_markov_ctreepo_appendix.py",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 2
    assert "archived" in result.stderr.lower()
    assert "report_appendix_walkthrough_narrative_deck.py" in result.stderr
