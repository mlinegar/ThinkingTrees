import argparse
import csv
import json
from pathlib import Path

from scripts import report_hll_fno_progress as report
from scripts import run_hll_canonical_observation_grid as grid


def _grid_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "python_bin": "./venv/bin/python",
        "sample_cache_dir": None,
        "exact_leaves": "16",
        "rollout_leaves": "16",
        "budget_leaves": "16",
        "include_baseline": True,
        "include_known_f": True,
        "include_sampled1": False,
        "include_budgeted_mass": False,
        "known_f_exact_schedules": "gfgf",
        "canonical_exact_schedules": "fgfg",
        "known_f_rollout_schedules": "gf",
        "canonical_rollout_schedules": "fgfg",
        "reuse_from_root": [],
        "root_label_shares": "0,50,100",
        "mass_target_per_doc": 1.0,
        "local_label_pool": "nonroot",
        "n_train": 8,
        "n_val": 2,
        "min_tokens": 1024,
        "max_tokens": 1024,
        "universe_size": 512,
        "zipf_alphas": "0.8,1.0,1.2",
        "precision": 8,
        "epochs": 20,
        "g_exact_epochs": 30,
        "gfgf_exact_epochs": 20,
        "fgfg_exact_epochs": 20,
        "rollout_epochs": 20,
        "batch_size": 8192,
        "rollout_min_docs_per_batch": 16,
        "rollout_max_docs_per_batch": 0,
        "eval_batch_size": 65536,
        "grad_accum_steps": 1,
        "hidden_channels": 512,
        "head_hidden_dim": 256,
        "n_modes": 32,
        "n_layers": 2,
        "f_learning_rate": "1e-4",
        "g_learning_rate": "1e-4",
        "local_law_weight": 0.5,
        "state_loss_weight": 1.0,
        "exact_state_anchor_weight": 0.1,
        "objective_loss_weight": 1.0,
        "eval_every_epochs": 1,
        "progress_every_epochs": 1,
        "progress_every_batches": 20,
        "seed": 0,
        "precompute_only": False,
        "gpu_ids": "0",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_canonical_grid_builds_1024_token_paper_cells(tmp_path: Path) -> None:
    args = _grid_args(sample_cache_dir=tmp_path / "sample_cache")
    cells = grid.build_cells(args, tmp_path / "grid")
    by_id = {cell.cell_id: cell for cell in cells}

    assert {
        "fstar_gstar_L16",
        "known_f_gfgf_exact_L16",
        "exact_formula_noid_L16",
        "rollout_root_formula_noid_L16",
        "rollout_dense_formula_noid_L16",
        "known_f_gf_rollout_root_L16",
        "known_f_gf_rollout_dense_L16",
    }.issubset(by_id)

    assert by_id["fstar_gstar_L16"].command == []
    assert by_id["known_f_gfgf_exact_L16"].command[by_id["known_f_gfgf_exact_L16"].command.index("--schedule") + 1] == "gfgf"
    assert by_id["exact_formula_noid_L16"].command[by_id["exact_formula_noid_L16"].command.index("--schedule") + 1] == "fgfg"
    assert by_id["known_f_gf_rollout_dense_L16"].command[
        by_id["known_f_gf_rollout_dense_L16"].command.index("--oracle-observation-design") + 1
    ] == "dense_oracle"

    for cell in cells:
        if not cell.command:
            continue
        assert cell.command[cell.command.index("--min-tokens") + 1] == "1024"
        assert cell.command[cell.command.index("--max-tokens") + 1] == "1024"
        assert cell.command[cell.command.index("--rollout-min-docs-per-batch") + 1] == "16"
        assert cell.command[cell.command.index("--rollout-max-docs-per-batch") + 1] == "0"
        assert cell.command[cell.command.index("--eval-batch-size") + 1] == "65536"
        assert cell.command[cell.command.index("--sample-cache-dir") + 1] == str(tmp_path / "sample_cache")

    manifest = grid.write_manifest_and_runners(args, tmp_path / "grid", cells)
    assert manifest["dgp"]["min_tokens"] == 1024
    assert manifest["dgp"]["max_tokens"] == 1024
    assert manifest["training"]["rollout_min_docs_per_batch"] == 16
    assert manifest["sample_cache_dir"] == str(tmp_path / "sample_cache")


def test_canonical_grid_precompute_only_uses_shared_1024_cache(tmp_path: Path) -> None:
    args = _grid_args(
        sample_cache_dir=tmp_path / "sample_cache",
        exact_leaves="16,64",
        rollout_leaves="64",
        precompute_only=True,
    )
    cells = grid.build_cells(args, tmp_path / "grid")

    assert {cell.cell_id for cell in cells} == {"precompute_L16", "precompute_L64"}
    for cell in cells:
        assert "--precompute-samples-only" in cell.command
        assert cell.command[cell.command.index("--device") + 1] == "cpu"
        assert cell.command[cell.command.index("--min-tokens") + 1] == "1024"
        assert cell.command[cell.command.index("--max-tokens") + 1] == "1024"
        assert cell.command[cell.command.index("--sample-cache-dir") + 1] == str(tmp_path / "sample_cache")


def test_canonical_grid_reuses_longest_schedule_prefix_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source_cell = source / "known_f_gfgf_exact_L16"
    checkpoint = source_cell / "hll_register_space" / "stage_04_f_model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake checkpoint")
    _write_report_summary(source_cell / "summary.csv", min_tokens=1024, root_mae=2.0)
    source_command = [
        "./venv/bin/python",
        "scripts/run_fno_mergeable_sketch_diagnostic.py",
        "--targets",
        "hll_register_space",
        "--n-train",
        "8",
        "--n-val",
        "2",
        "--n-leaves",
        "16",
        "--min-tokens",
        "1024",
        "--max-tokens",
        "1024",
        "--universe-size",
        "512",
        "--precision",
        "8",
        "--zipf-alphas",
        "0.8,1.0,1.2",
        "--target-transform",
        "linear01",
        "--state-normalization",
        "register_div64",
        "--hidden-channels",
        "512",
        "--head-hidden-dim",
        "256",
        "--n-modes",
        "32",
        "--n-layers",
        "2",
        "--readout-arch",
        "hll_formula",
        "--seed",
        "0",
        "--schedule",
        "gfgf",
        "--objective-mode",
        "exact_rows",
        "--oracle-observation-design",
        "root_only",
    ]
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_id": "known_f_gfgf_exact_L16",
                        "family": "fixed_f_exact_gfgf",
                        "n_leaves": 16,
                        "output_dir": str(source_cell),
                        "command": source_command,
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    args = _grid_args(
        sample_cache_dir=tmp_path / "sample_cache",
        known_f_exact_schedules="gfgfgf",
        canonical_exact_schedules="fgfg",
        exact_leaves="16",
        rollout_leaves="16",
        budget_leaves="16",
    )
    cells = grid.build_cells(args, tmp_path / "next")
    reused = grid.apply_reuse_sources(
        cells,
        output_root=tmp_path / "next",
        reuse_roots=[source],
        gpu_ids=[0],
    )
    target = next(cell for cell in reused if cell.cell_id == "known_f_gfgfgf_exact_L16")

    assert target.command[target.command.index("--schedule") + 1] == "gf"
    assert target.command[target.command.index("--schedule-prefix") + 1] == "gfgf"
    assert target.command[target.command.index("--stage-index-offset") + 1] == "4"
    assert target.command[target.command.index("--init-checkpoint") + 1] == str(checkpoint)


def test_fstar_gstar_baseline_summary_checks_tree_equals_flat(tmp_path: Path) -> None:
    args = _grid_args(
        n_train=2,
        n_val=2,
        min_tokens=8,
        max_tokens=8,
        universe_size=32,
        zipf_alphas="1.0",
        precision=4,
        exact_leaves="2",
        rollout_leaves="2",
        budget_leaves="2",
        include_known_f=False,
    )
    cells = grid.build_cells(args, tmp_path / "grid")
    grid.write_baseline_summaries(args, cells)

    baseline = next(cell for cell in cells if cell.cell_id == "fstar_gstar_L2")
    with (Path(baseline.output_dir) / "summary.csv").open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))

    assert row["objective_mode"] == "exact_package"
    assert float(row["hll_tree_flat_max_abs_diff"]) == 0.0
    assert float(row["hll_tree_flat_register_max_diff"]) == 0.0
    assert float(row["fstar_gstar_root_analytic_mae"]) >= 0.0


def _write_report_summary(path: Path, *, min_tokens: int, root_mae: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_kind",
        "n_leaves",
        "n_train",
        "n_val",
        "precision",
        "universe_size",
        "min_tokens",
        "max_tokens",
        "zipf_alphas",
        "seed",
        "root_mae",
        "root_rel_mae",
        "official_f_on_learned_root_mae",
        "merge_state_mae",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "target_kind": "hll_register_space",
                "n_leaves": 2,
                "n_train": 1,
                "n_val": 1,
                "precision": 4,
                "universe_size": 64,
                "min_tokens": min_tokens,
                "max_tokens": min_tokens,
                "zipf_alphas": "1.0",
                "seed": 0,
                "root_mae": root_mae,
                "root_rel_mae": root_mae / 10.0,
                "official_f_on_learned_root_mae": root_mae,
                "merge_state_mae": root_mae / 2.0,
            }
        )


def test_report_token_count_filter_keeps_1024_rows_first_class(tmp_path: Path) -> None:
    root = tmp_path / "hll_canonical_observation_grid_test"
    cells = []
    for min_tokens, root_mae in ((128, 8.0), (1024, 1.0)):
        cell_id = f"exact_formula_noid_T{min_tokens}_L2"
        output_dir = root / cell_id
        command = [
            "./venv/bin/python",
            "scripts/run_fno_mergeable_sketch_diagnostic.py",
            "--targets",
            "hll_register_space",
            "--n-leaves",
            "2",
            "--n-train",
            "1",
            "--n-val",
            "1",
            "--precision",
            "4",
            "--universe-size",
            "64",
            "--min-tokens",
            str(min_tokens),
            "--max-tokens",
            str(min_tokens),
            "--zipf-alphas",
            "1.0",
            "--schedule",
            "fgfg",
            "--objective-mode",
            "exact_rows",
            "--readout-arch",
            "hll_formula",
            "--oracle-observation-design",
            "root_only",
        ]
        _write_report_summary(output_dir / "summary.csv", min_tokens=min_tokens, root_mae=root_mae)
        cells.append(
            {
                "cell_id": cell_id,
                "family": "canonical_exact",
                "n_leaves": 2,
                "output_dir": str(output_dir),
                "command": command,
            }
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({"cells": cells}) + "\n", encoding="utf-8")

    out = tmp_path / "report"
    assert report.main(["--roots", str(root), "--output-dir", str(out), "--token-count", "1024"]) == 0

    with (out / "hll_fno_progress_rows.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["min_tokens"] == "1024"
    assert rows[0]["cell"] == "exact_formula_noid_T1024_L2"

    report_text = (out / "report.md").read_text(encoding="utf-8")
    assert "## Token Regime" in report_text
    assert "Filtered to `1024` to `1024` tokens/document" in report_text


def test_report_discovers_treepo_hll_adjusted_local_law_rows(tmp_path: Path) -> None:
    root = tmp_path / "treepo_hll_grid"
    run_dir = root / "treepo_induced"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "model_kind": "induced_projection",
                        "objective_mode": "corrected_local_law",
                        "proxy_mode": "frozen_rollout",
                        "lean_adjusted_loss": "proxy + R / pi * (oracle - proxy)",
                        "lean_merge_adapter": "merge(a,b)=g_theta(a+b); encode_leaf(x)=g_theta(x)",
                        "precision": 4,
                        "train_docs": 8,
                        "audit_policy": "all",
                        "learned_relative_rmse": 0.25,
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = report._discover_rows([root])

    assert len(rows) == 1
    assert rows[0]["source_family"] == "treepo_hll_merge_learning"
    assert rows[0]["target_kind"] == "hll_register_space"
    assert rows[0]["model_kind"] == "induced_projection"
    assert rows[0]["root_rel_mae"] == 0.25
