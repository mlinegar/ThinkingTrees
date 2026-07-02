import argparse
import csv
import json
from pathlib import Path

from scripts import report_hll_fno_progress as report
from scripts import run_hll_sampled_node_rate_grid as grid


def _grid_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "python_bin": "./venv/bin/python",
        "leaves": "16",
        "root_label_shares": "1.0",
        "sample_rates": "0,0.01,1.0",
        "skip_zero_label_cell": False,
        "gpu_ids": "0",
        "n_train": 8,
        "n_val": 2,
        "min_tokens": 8,
        "max_tokens": 8,
        "universe_size": 32,
        "zipf_alphas": "1.0",
        "precision": 4,
        "schedule": "g",
        "epochs": 1,
        "batch_size": 4,
        "rollout_min_docs_per_batch": 2,
        "rollout_max_docs_per_batch": 0,
        "eval_batch_size": 64,
        "grad_accum_steps": 1,
        "target_transform": "log1p_zscore",
        "hidden_channels": 16,
        "head_hidden_dim": 16,
        "n_modes": 4,
        "n_layers": 1,
        "f_learning_rate": "1e-4",
        "g_learning_rate": "1e-4",
        "local_law_weight": 0.5,
        "local_law_leaf_discount_gamma": 1.0,
        "merge_output_constraint": "none",
        "objective_loss_weight": 1.0,
        "state_loss_weight": 0.0,
        "exact_state_anchor_weight": 0.0,
        "allow_dense_regularizers": False,
        "eval_every_epochs": 1,
        "progress_every_epochs": 1,
        "progress_every_batches": 2,
        "seed": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_sampled_node_rate_grid_builds_root_and_sampled_commands(tmp_path: Path) -> None:
    cells = grid.build_cells(_grid_args(), tmp_path / "grid")
    by_rate = {cell.sampled_node_rate: cell for cell in cells}

    assert sorted(by_rate) == [0.0, 0.01, 1.0]

    root_cmd = by_rate[0.0].command
    assert "--schedule" in root_cmd
    assert root_cmd[root_cmd.index("--schedule") + 1] == "g"
    assert root_cmd[root_cmd.index("--oracle-observation-design") + 1] == "root_only"
    assert "--sampled-node-rate" not in root_cmd

    sampled_cmd = by_rate[0.01].command
    assert sampled_cmd[sampled_cmd.index("--oracle-observation-design") + 1] == "sampled_nodes"
    assert sampled_cmd[sampled_cmd.index("--sampled-node-rate") + 1] == "0.01"

    dense_cmd = by_rate[1.0].command
    assert dense_cmd[dense_cmd.index("--oracle-observation-design") + 1] == "sampled_nodes"
    assert dense_cmd[dense_cmd.index("--sampled-node-rate") + 1] == "1"

    for cell in cells:
        assert cell.root_label_share == 1.0
        assert cell.command[cell.command.index("--readout-arch") + 1] == "hll_formula"
        assert cell.command[cell.command.index("--rollout-min-docs-per-batch") + 1] == "2"
        assert cell.command[cell.command.index("--rollout-max-docs-per-batch") + 1] == "0"
        assert cell.command[cell.command.index("--eval-batch-size") + 1] == "64"
        assert cell.command[cell.command.index("--progress-every-batches") + 1] == "2"
        assert cell.command[cell.command.index("--target-transform") + 1] == "log1p_zscore"
        assert cell.command[cell.command.index("--state-loss-weight") + 1] == "0.0"
        assert cell.command[cell.command.index("--exact-state-anchor-weight") + 1] == "0.0"
        assert cell.command[cell.command.index("--local-law-leaf-discount-gamma") + 1] == "1.0"
        assert "--merge-base" not in cell.command
        assert cell.command[cell.command.index("--merge-output-constraint") + 1] == "none"
        assert "--no-identity-residual-init" not in cell.command


def test_sampled_node_rate_grid_threads_local_law_leaf_discount_gamma(tmp_path: Path) -> None:
    cells = grid.build_cells(_grid_args(local_law_leaf_discount_gamma=0.8), tmp_path / "grid")

    for cell in cells:
        assert cell.command[cell.command.index("--local-law-leaf-discount-gamma") + 1] == "0.8"


def test_sampled_node_rate_grid_threads_merge_output_constraint(tmp_path: Path) -> None:
    cells = grid.build_cells(_grid_args(merge_output_constraint="unit_clamp"), tmp_path / "grid")

    for cell in cells:
        assert cell.command[cell.command.index("--merge-output-constraint") + 1] == "unit_clamp"


def test_sampled_node_rate_grid_threads_target_transform(tmp_path: Path) -> None:
    cells = grid.build_cells(_grid_args(target_transform="linear01"), tmp_path / "grid")

    for cell in cells:
        assert cell.command[cell.command.index("--target-transform") + 1] == "linear01"


def test_sampled_node_rate_grid_has_no_merge_base_knob(tmp_path: Path) -> None:
    cells = grid.build_cells(_grid_args(), tmp_path / "grid")

    for cell in cells:
        assert "--merge-base" not in cell.command


def test_sampled_node_rate_grid_builds_root_rate_ablation_commands(tmp_path: Path) -> None:
    cells = grid.build_cells(
        _grid_args(root_label_shares="1.0,0.1", sample_rates="0,0.03"),
        tmp_path / "grid",
    )
    by_design = {(cell.root_label_share, cell.sampled_node_rate): cell for cell in cells}

    assert sorted(by_design) == [(0.1, 0.0), (0.1, 0.03), (1.0, 0.0), (1.0, 0.03)]

    root_subsample = by_design[(0.1, 0.0)].command
    assert root_subsample[root_subsample.index("--oracle-observation-design") + 1] == "sampled_root_nodes"
    assert root_subsample[root_subsample.index("--root-label-share") + 1] == "0.1"
    assert root_subsample[root_subsample.index("--sampled-node-rate") + 1] == "0"

    root_plus_nodes = by_design[(0.1, 0.03)].command
    assert root_plus_nodes[root_plus_nodes.index("--oracle-observation-design") + 1] == "sampled_root_nodes"
    assert root_plus_nodes[root_plus_nodes.index("--root-label-share") + 1] == "0.1"
    assert root_plus_nodes[root_plus_nodes.index("--sampled-node-rate") + 1] == "0.03"


def test_sampled_node_rate_grid_orders_full_root_share_first(tmp_path: Path) -> None:
    cells = grid.build_cells(
        _grid_args(root_label_shares="0.1,1.0,0.5", sample_rates="0.1,0,1.0"),
        tmp_path / "grid",
    )

    assert [cell.cell_id for cell in cells] == [
        "sampled_R100_r000_L16",
        "sampled_R100_r100_L16",
        "sampled_R100_r1000_L16",
        "sampled_R050_r000_L16",
        "sampled_R050_r100_L16",
        "sampled_R050_r1000_L16",
        "sampled_R010_r000_L16",
        "sampled_R010_r100_L16",
        "sampled_R010_r1000_L16",
    ]


def test_sampled_node_rate_grid_supports_fgfgfg_schedule(tmp_path: Path) -> None:
    cells = grid.build_cells(
        _grid_args(
            schedule="fgfgfg",
            root_label_shares="1.0",
            sample_rates="0,1.0",
            state_loss_weight=1.0,
            exact_state_anchor_weight=0.1,
            allow_dense_regularizers=True,
        ),
        tmp_path / "grid",
    )
    assert {cell.cell_id for cell in cells} == {
        "sampled_fgfgfg_R100_r000_L16",
        "sampled_fgfgfg_R100_r1000_L16",
    }
    for cell in cells:
        assert cell.command[cell.command.index("--schedule") + 1] == "fgfgfg"
        assert cell.command[cell.command.index("--state-loss-weight") + 1] == "1.0"
        assert cell.command[cell.command.index("--exact-state-anchor-weight") + 1] == "0.1"
        assert cell.estimated_row_work == 6 * 1 * 8 * (2 * 16 - 1)


def test_sampled_node_rate_grid_allows_scalar_only_override(tmp_path: Path) -> None:
    cells = grid.build_cells(
        _grid_args(
            state_loss_weight=0.0,
            exact_state_anchor_weight=0.0,
            sample_rates="0.1",
        ),
        tmp_path / "grid",
    )
    assert len(cells) == 1
    command = cells[0].command
    assert command[command.index("--state-loss-weight") + 1] == "0.0"
    assert command[command.index("--exact-state-anchor-weight") + 1] == "0.0"


def test_sampled_node_rate_grid_rejects_dense_regularizers_without_opt_in(tmp_path: Path) -> None:
    args = _grid_args(state_loss_weight=1.0, exact_state_anchor_weight=0.1)

    try:
        grid.build_cells(args, tmp_path / "grid")
    except ValueError as exc:
        assert "--allow-dense-regularizers" in str(exc)
    else:  # pragma: no cover - explicit guard assertion
        raise AssertionError("expected dense regularizers to require explicit opt-in")


def test_sampled_node_rate_grid_can_skip_zero_label_cell(tmp_path: Path) -> None:
    cells = grid.build_cells(
        _grid_args(
            root_label_shares="0,0.1",
            sample_rates="0,0.01",
            skip_zero_label_cell=True,
        ),
        tmp_path / "grid",
    )

    assert {cell.cell_id for cell in cells} == {
        "sampled_R000_r010_L16",
        "sampled_R010_r000_L16",
        "sampled_R010_r010_L16",
    }


def _write_summary(path: Path, *, official_mae: float, merge_root_mae: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
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
                "official_f_on_learned_root_mae",
                "merge_state_root_mae",
                "merge_state_mae",
                "train_observed_rows_per_doc_end",
                "train_root_observed_rows_per_doc_end",
                "train_nonroot_observed_rows_per_doc_end",
                "train_max_ipw_weight_end",
                "train_effective_sample_size_end",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "target_kind": "hll_register_space",
                "n_leaves": 16,
                "n_train": 1,
                "n_val": 1,
                "precision": 4,
                "universe_size": 32,
                "min_tokens": 8,
                "max_tokens": 8,
                "zipf_alphas": "1.0",
                "seed": 0,
                "root_mae": official_mae,
                "official_f_on_learned_root_mae": official_mae,
                "merge_state_root_mae": merge_root_mae,
                "merge_state_mae": merge_root_mae,
                "train_observed_rows_per_doc_end": 2.0,
                "train_root_observed_rows_per_doc_end": 1.0,
                "train_nonroot_observed_rows_per_doc_end": 1.0,
                "train_max_ipw_weight_end": 10.0,
                "train_effective_sample_size_end": 1.2,
            }
        )


def test_report_emits_sampled_rate_curve_and_primary_metric(tmp_path: Path) -> None:
    root = tmp_path / "hll_sampled_node_rate_grid_test"
    cells = []
    for rate, official, merge_root in ((0.0, 4.0, 0.7), (0.1, 0.4, 0.2)):
        cell_id = f"sampled_{int(rate * 1000):03d}_L16"
        output_dir = root / cell_id
        command = [
            "./venv/bin/python",
            "scripts/run_fno_mergeable_sketch_diagnostic.py",
            "--targets",
            "hll_register_space",
            "--n-leaves",
            "16",
            "--n-train",
            "1",
            "--n-val",
            "1",
            "--precision",
            "4",
            "--universe-size",
            "32",
            "--min-tokens",
            "8",
            "--max-tokens",
            "8",
            "--zipf-alphas",
            "1.0",
            "--schedule",
            "g",
            "--objective-mode",
            "rollout_local_law",
            "--readout-arch",
            "hll_formula",
            "--state-loss-weight",
            "0.0",
            "--exact-state-anchor-weight",
            "0.0",
            "--oracle-observation-design",
            "root_only" if rate == 0.0 else "sampled_nodes",
        ]
        if rate > 0.0:
            command.extend(["--sampled-node-rate", str(rate)])
        _write_summary(output_dir / "summary.csv", official_mae=official, merge_root_mae=merge_root)
        cells.append(
            {
                "cell_id": cell_id,
                "family": "sampled",
                "n_leaves": 16,
                "sampled_node_rate": rate,
                "output_dir": str(output_dir),
                "command": command,
            }
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({"cells": cells}) + "\n", encoding="utf-8")

    out = tmp_path / "report"
    assert report.main(["--roots", str(root), "--output-dir", str(out)]) == 0

    report_text = (out / "report.md").read_text(encoding="utf-8")
    assert "Root Plus Random Non-Root Sampling" in report_text
    assert "official_f_on_learned_root_mae" in report_text
    assert "primary root MAE f*(g_theta)" in report_text
    assert "fstar_gtheta_sampled_rate_by_leaves.png" in report_text
    assert (out / "figures" / "fstar_gtheta_sampled_rate_by_leaves.png").exists()
