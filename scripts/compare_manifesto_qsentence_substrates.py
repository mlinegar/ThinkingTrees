#!/usr/bin/env python3
"""Compare substrate families on the Manifesto q-sentence reconstruction task.

Joins ``grid_summary.json`` files from any number of ladder runs (DSPy/LLM,
DSPy/DiffusionGemma, FNO/embeddings — anything emitted by
``run_manifesto_qsentence_dspy_ladder.py`` or the legacy alternating ladders)
into one tidy table keyed by (run label, family, leaf axis, iteration/stage).

Runs are only comparable when they share the grid bundle and split; the tool
records ``fg_grid_dir``/``eval_split`` per run and warns when they differ.

Usage:
    ./venv/bin/python scripts/compare_manifesto_qsentence_substrates.py \
        gemma4=outputs/manifesto_qsentence_gemma4_run \
        dgemma=outputs/manifesto_qsentence_diffusiongemma_small \
        fno=outputs/manifesto_qsentence_fno_run \
        --output-dir outputs/manifesto_qsentence_substrate_comparison

Each positional arg is ``label=run_dir`` (a dir containing grid_summary.json,
or a direct path to one). Without ``label=``, the directory name is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

METRIC_COLUMNS = (
    "n_eval",
    "external_expert_pearson",
    "external_expert_mae_1_7",
    "internal_f_pearson",
    "internal_f_mae_1_7",
    "f_star_gap",
    "mean_prediction_1_7",
    "mean_teacher_1_7",
    "mean_expert_1_7",
)

KEY_COLUMNS = (
    "run",
    "family",
    "substrate_model",
    "leaf_axis",
    "leaf_value",
    "iteration",
    "stage_label",
    "trained",
)


def _resolve_summary_path(spec: str) -> Tuple[str, Path]:
    label: Optional[str] = None
    raw = spec
    if "=" in spec:
        label, raw = spec.split("=", 1)
    path = Path(raw).expanduser()
    if path.is_dir():
        candidate = path / "grid_summary.json"
        if not candidate.exists():
            matches = sorted(path.glob("**/grid_summary.json"))
            if len(matches) == 1:
                candidate = matches[0]
            elif not matches:
                raise FileNotFoundError(f"no grid_summary.json under {path}")
            else:
                raise ValueError(
                    f"multiple grid_summary.json under {path}; pass one explicitly: "
                    + ", ".join(str(m) for m in matches)
                )
        path = candidate
    if not path.exists():
        raise FileNotFoundError(path)
    if label is None:
        label = path.parent.name
    return str(label), path


def _rows_from_summary(label: str, summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    family_default = summary.get("family")
    for row in summary.get("rows", []) or []:
        leaf_axis = row.get("axis_kind") or summary.get("topology_axis") or "leaf"
        out: Dict[str, Any] = {
            "run": label,
            "family": row.get("family") or family_default,
            "substrate_model": row.get("substrate_model") or summary.get("substrate_model"),
            "leaf_axis": leaf_axis,
            "leaf_value": row.get("axis_value"),
            "iteration": row.get("iteration"),
            "stage_label": row.get("stage_label") or row.get("stage_name"),
            "trained": row.get("trained"),
        }
        for key in METRIC_COLUMNS:
            out[key] = row.get(key)
        rows.append(out)
    return rows


def _select_best(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Keep, per (run, family, leaf), the best iteration by external pearson
    (falling back to the highest iteration when pearson is absent)."""
    best: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        key = (row.get("run"), row.get("family"), row.get("leaf_value"))
        current = best.get(key)

        def rank(r: Mapping[str, Any]) -> Tuple[int, float, int]:
            pearson = r.get("external_expert_pearson")
            has = 1 if isinstance(pearson, (int, float)) else 0
            return (has, float(pearson) if has else 0.0, int(r.get("iteration") or 0))

        if current is None or rank(row) > rank(current):
            best[key] = dict(row)
    return [best[key] for key in sorted(best, key=lambda k: tuple(str(x) for x in k))]


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    provenance: Sequence[Mapping[str, Any]],
    best_rows: Sequence[Mapping[str, Any]],
) -> None:
    columns = list(KEY_COLUMNS) + list(METRIC_COLUMNS)
    lines = [
        "# Manifesto q-sentence substrate comparison",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
        "",
        "## Runs",
        "",
        "| run | family | grid bundle | eval split | summary |",
        "|---|---|---|---|---|",
    ]
    for item in provenance:
        lines.append(
            f"| {item['run']} | {item.get('family') or '?'} | "
            f"{item.get('fg_grid_dir') or '?'} | {item.get('eval_split') or '?'} | "
            f"{item['path']} |"
        )
    lines += [
        "",
        "## Best iteration per (run, leaf) — ranked by external expert Pearson",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in best_rows:
        lines.append("| " + " | ".join(_fmt(row.get(col)) for col in columns) + " |")
    lines += [
        "",
        "## All iterations",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col)) for col in columns) + " |")
    lines += [
        "",
        "Metrics are on the normalized [0,1] CMP aggregate scale; `_1_7` field "
        "names are retained for schema compatibility. FNO rows target the "
        "scalar `rile`; DSPy rows target the full compact dimension vector.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        help="label=path pairs (run dir containing grid_summary.json, or the file).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    all_rows: List[Dict[str, Any]] = []
    provenance: List[Dict[str, Any]] = []
    for spec in args.runs:
        label, path = _resolve_summary_path(str(spec))
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows = _rows_from_summary(label, summary)
        if not rows:
            LOGGER.warning("run %s (%s) has no rows", label, path)
        all_rows.extend(rows)
        provenance.append(
            {
                "run": label,
                "path": str(path),
                "family": summary.get("family"),
                "fg_grid_dir": summary.get("fg_grid_dir") or summary.get("tree_bundle"),
                "eval_split": summary.get("eval_split"),
            }
        )

    bundles = {item.get("fg_grid_dir") for item in provenance}
    if len(bundles) > 1:
        LOGGER.warning(
            "runs use DIFFERENT grid bundles %s — comparison is not apples-to-apples",
            sorted(str(b) for b in bundles),
        )
    splits = {item.get("eval_split") for item in provenance}
    if len(splits) > 1:
        LOGGER.warning("runs use different eval splits: %s", sorted(str(s) for s in splits))

    all_rows.sort(
        key=lambda r: (
            str(r.get("run")),
            str(r.get("family")),
            int(r.get("leaf_value") or 0),
            int(r.get("iteration") or 0),
        )
    )
    best_rows = _select_best(all_rows)

    output_dir = args.output_dir or Path("outputs") / (
        "manifesto_qsentence_substrate_comparison_"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = list(KEY_COLUMNS) + list(METRIC_COLUMNS)
    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    md_path = output_dir / "comparison.md"
    _write_markdown(all_rows, md_path, provenance=provenance, best_rows=best_rows)
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(md_path.read_text(encoding="utf-8"))
    LOGGER.info("Wrote %s and %s", csv_path, md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
