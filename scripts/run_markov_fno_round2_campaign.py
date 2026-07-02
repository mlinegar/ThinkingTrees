#!/usr/bin/env python3
"""Round 2 follow-up to `run_markov_fno_bridge_budget_campaign.py`.

After the 8h bridge campaign confirmed the FNO bridge is not solved at the
campaign's capacity envelope, this runner tests two general-method
hypotheses on the PyTorch CleanUnifiedNO surface:

  Phase A: wider FNO at the best-cell zone (capacity hypothesis).
           Anchor leaf=32, ch=128, gm=16, ep=24, root_mae=1.94 from the
           campaign; sweep up channels, modes, and epochs to see if pure
           capacity closes the gap to JAX.

  Phase B: long-train best-cell with seed sweep.
           Tests whether the campaign's best cell was just under-trained
           or has actually plateaued.

JAX phases (C, D) are launched separately by run_markov_fno_round2_jax.sh.

Reuses the dispatch infrastructure (worker queue, summary/report writing)
from run_markov_fno_bridge_budget_campaign.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Reuse infrastructure from the bridge campaign.
from run_markov_fno_bridge_budget_campaign import (
    CampaignCell,
    _batch_for_leaf,
    _parse_gpus,
    _run_cells_parallel,
    _utc_stamp,
    _write_rows,
)


REPO = Path(__file__).resolve().parents[1]


def _train_docs_for_leaf_round2(leaf_tokens: int, default_train_docs: int) -> int:
    if leaf_tokens <= 4:
        return min(default_train_docs, 2048)
    if leaf_tokens <= 16:
        return min(default_train_docs, 4096)
    return default_train_docs


def _round2_cell(
    *,
    phase: str,
    leaf_tokens: int,
    channels: int,
    g_n_modes: int,
    epochs: int,
    train_docs: int,
    eval_docs: int,
    seed: int,
    load_data_bundle: str,
    label: str = "",
) -> CampaignCell:
    return CampaignCell(
        phase=phase,
        objective="markov_local_laws_fno",
        leaf_tokens=leaf_tokens,
        channels=channels,
        g_n_modes=g_n_modes,
        epochs=epochs,
        batch_size=_batch_for_leaf(leaf_tokens),
        train_docs=_train_docs_for_leaf_round2(leaf_tokens, train_docs),
        eval_docs=eval_docs,
        seed=seed,
        load_data_bundle=load_data_bundle,
        label=label,
    )


def _phase_a_wider_fno_cells(args: argparse.Namespace) -> list[CampaignCell]:
    cells: list[CampaignCell] = []
    for leaf in [32, 64, 128]:
        for channels in [128, 256, 512]:
            for modes in [16, 32, 64]:
                cells.append(
                    _round2_cell(
                        phase="r2_phase_a_wider_fno",
                        leaf_tokens=leaf,
                        channels=channels,
                        g_n_modes=modes,
                        epochs=args.phase_a_epochs,
                        train_docs=args.train_docs,
                        eval_docs=args.eval_docs,
                        seed=args.seed,
                        load_data_bundle=args.load_data_bundle,
                    )
                )
    return cells


def _phase_b_long_train_cells(args: argparse.Namespace) -> list[CampaignCell]:
    cells: list[CampaignCell] = []
    for epochs in [48, 96, 192]:
        for seed in [0, 1, 2]:
            cells.append(
                _round2_cell(
                    phase="r2_phase_b_long_train",
                    leaf_tokens=32,
                    channels=128,
                    g_n_modes=16,
                    epochs=epochs,
                    train_docs=args.train_docs,
                    eval_docs=args.eval_docs,
                    seed=seed,
                    load_data_bundle=args.load_data_bundle,
                    label=f"ep{epochs}",
                )
            )
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Round 2 PyTorch FNO bridge campaign: wider-FNO sweep + "
            "long-train best-cell seed sweep."
        )
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO / "outputs" / f"markov_fno_round2_{_utc_stamp()}"),
    )
    parser.add_argument("--budget-hours", type=float, default=10.0)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument(
        "--load-data-bundle",
        default=(
            "outputs/_bundles/markov_hazard_panels/"
            "paper_hazard_panel_v1_t128/seed_0/base_bundle.json"
        ),
    )
    parser.add_argument("--train-docs", type=int, default=10240)
    parser.add_argument("--eval-docs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phase-a-epochs", type=int, default=48)
    parser.add_argument("--skip-phase-a", action="store_true")
    parser.add_argument("--skip-phase-b", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    gpus = _parse_gpus(args.gpus)
    deadline = time.time() + float(args.budget_hours) * 3600.0
    rows: list[dict[str, Any]] = []
    rows_lock = threading.Lock()
    manifest = {
        "schema_version": "markov_fno_round2_campaign.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "output_root": str(output_root),
        "budget_hours": args.budget_hours,
        "deadline_utc": datetime.fromtimestamp(deadline, UTC).isoformat(),
        "gpus": gpus,
        "args": vars(args),
    }
    import json as _json

    with open(output_root / "campaign_manifest.json", "w") as fh:
        _json.dump(manifest, fh, indent=2)

    if not args.skip_phase_a:
        phase_a = _phase_a_wider_fno_cells(args)
        _run_cells_parallel(
            phase_name="phase A wider FNO",
            cells=phase_a,
            gpus=gpus,
            output_root=output_root,
            deadline=deadline,
            rows=rows,
            rows_lock=rows_lock,
        )

    if not args.skip_phase_b:
        phase_b = _phase_b_long_train_cells(args)
        _run_cells_parallel(
            phase_name="phase B long-train + seed sweep",
            cells=phase_b,
            gpus=gpus,
            output_root=output_root,
            deadline=deadline,
            rows=rows,
            rows_lock=rows_lock,
        )

    _write_rows(output_root, rows)
    print(f"summary: {output_root / 'campaign_summary.csv'}", flush=True)
    print(f"report:  {output_root / 'campaign_report.md'}", flush=True)


if __name__ == "__main__":
    main()
