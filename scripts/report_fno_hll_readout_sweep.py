#!/usr/bin/env python3
"""Aggregate HLL FNO readout sweep runs into tables and learning-curve plots."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.contracts import assert_public_contract_clean


def _read_csv_one(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} has no rows")
    return dict(rows[0])


def _float(row: dict[str, object], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _label(row: dict[str, object]) -> str:
    return (
        f"{row.get('schedule', '')} L={row.get('n_leaves', '')} "
        f"{row.get('readout_arch', '')}/{row.get('target_transform', '')}/"
        f"{row.get('state_normalization', '')}"
    )


def _collect_epoch_rows(run_root: Path, summary_row: dict[str, object]) -> list[dict[str, object]]:
    target_dir = run_root / "hll_register_space"
    out: list[dict[str, object]] = []
    offset = 0
    for path in sorted(target_dir.glob("stage_*_losses.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = list(data.get("epoch_metrics") or [])
        component = str(data.get("component") or path.stem.split("_")[-2])
        for item in metrics:
            if not item:
                continue
            epoch = int(float(item.get("epoch", 0)))
            row: dict[str, object] = {
                "run_root": str(run_root),
                "run_label": _label(summary_row),
                "schedule": summary_row.get("schedule", ""),
                "n_leaves": summary_row.get("n_leaves", ""),
                "readout_arch": summary_row.get("readout_arch", ""),
                "target_transform": summary_row.get("target_transform", ""),
                "state_normalization": summary_row.get("state_normalization", ""),
                "component": component,
                "stage_epoch": epoch,
                "global_epoch": offset + epoch,
            }
            row.update(item)
            out.append(row)
        raw_losses = list(data.get("epoch_mean_losses") or [])
        offset += len(raw_losses)
    return out


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_public_contract_clean(rows, surface=str(path))
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: Sequence[dict[str, object]]) -> None:
    ordered = sorted(rows, key=lambda r: _float(r, "root_rel_mae"))
    lines = [
        "# HLL FNO Readout Sweep",
        "",
        "| run | schedule | L | readout | transform | state norm | root rel MAE | f exact-root rel MAE | official-on-learned rel MAE | seconds |",
        "|---|---|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            "| {run} | {schedule} | {n_leaves} | {readout_arch} | {target_transform} | "
            "{state_normalization} | {root_rel_mae:.6g} | {learned_f_on_exact_root_rel_mae:.6g} | "
            "{official_f_on_learned_root_rel_mae:.6g} | {wall_seconds:.1f} |".format(
                run=Path(str(row["run_root"])).name,
                schedule=row.get("schedule", ""),
                n_leaves=row.get("n_leaves", ""),
                readout_arch=row.get("readout_arch", ""),
                target_transform=row.get("target_transform", ""),
                state_normalization=row.get("state_normalization", ""),
                root_rel_mae=_float(row, "root_rel_mae"),
                learned_f_on_exact_root_rel_mae=_float(row, "learned_f_on_exact_root_rel_mae"),
                official_f_on_learned_root_rel_mae=_float(row, "official_f_on_learned_root_rel_mae"),
                wall_seconds=_float(row, "wall_seconds"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(path_stem: Path, epoch_rows: Sequence[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path_stem.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict[str, object]]] = {}
    for row in epoch_rows:
        groups.setdefault(str(row["run_label"]), []).append(row)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for label, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: float(r["global_epoch"]))
        xs = [float(r["global_epoch"]) for r in rows]
        ys = [_float(r, "root_rel_mae") for r in rows]
        ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.2, label=label)
    ax.set_xlabel("global epoch across f/g stages")
    ax.set_ylabel("root relative MAE")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(path_stem.with_suffix(f".{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    summary_rows: list[dict[str, object]] = []
    epoch_rows: list[dict[str, object]] = []
    for root in args.run_roots:
        row = _read_csv_one(root / "summary.csv")
        row["run_root"] = str(root)
        row["run_label"] = _label(row)
        summary_rows.append(row)
        epoch_rows.extend(_collect_epoch_rows(root, row))

    output_dir = Path(args.output_dir)
    assert_public_contract_clean(
        summary_rows,
        surface="HLL/FNO readout sweep summary rows",
    )
    assert_public_contract_clean(
        epoch_rows,
        surface="HLL/FNO readout sweep epoch rows",
    )
    _write_csv(output_dir / "hll_fno_readout_sweep_summary.csv", summary_rows)
    _write_csv(output_dir / "hll_fno_readout_sweep_epoch_curves.csv", epoch_rows)
    _write_md(output_dir / "hll_fno_readout_sweep_summary.md", summary_rows)
    if epoch_rows:
        _plot(output_dir / "hll_fno_readout_learning_curves", epoch_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
