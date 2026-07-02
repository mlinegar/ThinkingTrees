#!/usr/bin/env python3
"""Per-dimension substrate comparison for the Manifesto q-sentence ladder.

The grid summary emitted by ``run_manifesto_qsentence_dspy_ladder.py`` collapses
the compact target vector to a single pooled metric. Pooling across dimensions
that sit at very different means inflates Pearson (the model only has to rank
dimensions, not discriminate within one), so the pooled number is not a faithful
reconstruction score. This tool reads the per-leaf ``iteration_history.json``
files instead, where ``split_metrics[split].per_dimension[dim]`` carries the
honest within-dimension metrics, and joins any number of runs into one tidy
per-(run, leaf, stage, dimension) table.

External expert codes exist for ``rile`` only (domains have ``n_external=0``),
so ``internal_f_*`` (prediction vs teacher rollup) is the cross-dimension /
cross-substrate comparable metric; ``external_expert_*`` is rile-only.

Usage:
    ./venv/bin/python scripts/summarize_manifesto_qsentence_per_dimension.py \
        dgemma=outputs/manifesto_qsentence_diffusiongemma_full_leafgrid \
        gemma4=outputs/manifesto_qsentence_gemma4_small \
        fno=outputs/manifesto_qsentence_fno_perdim \
        --split test --output-dir outputs/manifesto_qsentence_perdim_comparison

Each positional arg is ``label=run_dir`` (a dir containing one or more
``iteration_history.json`` under it). Without ``label=`` the dir name is used.
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

# Canonical dimension order (rile first, then the 7 CMP domains).
DIMENSION_ORDER = ("rile",) + tuple(f"domain_{i}" for i in range(1, 8))

METRIC_COLUMNS = (
    "internal_f_pearson",
    "internal_f_mae_1_7",
    "external_expert_pearson",
    "external_expert_mae_1_7",
    "f_star_gap",
    "mean_prediction_1_7",
    "mean_teacher_1_7",
    "mean_expert_1_7",
    "n_internal",
    "n_external",
)

KEY_COLUMNS = (
    "run",
    "family",
    "substrate_model",
    "leaf_value",
    "iteration",
    "stage_label",
    "trained",
    "dimension",
)


def _resolve_run(spec: str) -> Tuple[str, Path]:
    label: Optional[str] = None
    raw = spec
    if "=" in spec:
        label, raw = spec.split("=", 1)
    path = Path(raw).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if label is None:
        label = path.name
    return str(label), path


def _iter_history_files(run_dir: Path) -> List[Path]:
    if run_dir.is_file() and run_dir.name == "iteration_history.json":
        return [run_dir]
    return sorted(run_dir.glob("**/iteration_history.json"))


def _rows_from_history(label: str, hist: Mapping[str, Any], split: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    family = hist.get("family")
    substrate = hist.get("substrate_model")
    leaf = hist.get("axis_value")
    if leaf is None:
        leaf = hist.get("leaf_qsentences")
    # FNO histories carry a single scalar head (target_dimensions == [dim]) and
    # report top-level metrics with no per_dimension block; treat that as the
    # one dimension it was retargeted to.
    scalar_dims = [str(d) for d in (hist.get("target_dimensions") or [])]
    scalar_dim = scalar_dims[0] if len(scalar_dims) == 1 else None
    for it in hist.get("iterations", []) or []:
        split_metrics = it.get("split_metrics", {}) or {}
        sm = split_metrics.get(split) or split_metrics.get("all") or {}
        per_dim = sm.get("per_dimension") or {}
        if not per_dim and scalar_dim is not None:
            per_dim = {scalar_dim: sm}
        if not per_dim:
            continue
        for dim, metrics in per_dim.items():
            metrics = metrics or {}
            out: Dict[str, Any] = {
                "run": label,
                "family": family,
                "substrate_model": substrate,
                "leaf_value": leaf,
                "iteration": it.get("iteration"),
                "stage_label": it.get("stage_label") or it.get("stage_name"),
                "trained": it.get("trained"),
                "dimension": dim,
            }
            for key in METRIC_COLUMNS:
                out[key] = metrics.get(key)
            rows.append(out)
    return rows


def _dim_rank(dim: Any) -> Tuple[int, str]:
    s = str(dim)
    return (DIMENSION_ORDER.index(s) if s in DIMENSION_ORDER else len(DIMENSION_ORDER), s)


def _select_final(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Final composed stage (max iteration) per (run, leaf, dimension).

    We deliberately do NOT rank by internal_f_pearson: at the f-stage the leaf
    reader echoes the teacher labels exactly (prediction == teacher, MAE 0.000,
    Pearson 1.000), so a best-Pearson rule just selects that trivial echo and
    hides the learned-merge behavior. The ladder's actual output is the highest
    iteration (fully composed f.g), so that is the honest comparison row."""
    final: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        key = (row.get("run"), row.get("leaf_value"), row.get("dimension"))
        cur = final.get(key)
        if cur is None or int(row.get("iteration") or 0) > int(cur.get("iteration") or 0):
            final[key] = dict(row)
    return sorted(
        final.values(),
        key=lambda r: (str(r.get("run")), int(r.get("leaf_value") or 0), _dim_rank(r.get("dimension"))),
    )


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
    all_rows: Sequence[Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    provenance: Sequence[Mapping[str, Any]],
    split: str,
) -> None:
    cols = list(KEY_COLUMNS) + list(METRIC_COLUMNS)
    lines = [
        "# Manifesto q-sentence per-dimension substrate comparison",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}; eval split = `{split}`.",
        "",
        "`internal_f_*` = prediction vs teacher rollup (defined for all 8 dims). "
        "`external_expert_*` = vs human expert codes (rile only; domains have n_external=0). "
        "Metrics are on the normalized [0,1] CMP scale.",
        "",
        "## Runs",
        "",
        "| run | family | substrate | iteration_history files |",
        "|---|---|---|---:|",
    ]
    for item in provenance:
        lines.append(
            f"| {item['run']} | {item.get('family') or '?'} | "
            f"{item.get('substrate_model') or '?'} | {item.get('n_files')} |"
        )

    # Compact pivots at the final composed g-stage (the ladder's real output).
    # The f-stage echoes teacher labels (Pearson 1.000), so it is excluded here.
    pivot: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    for r in final_rows:
        pivot.setdefault((r.get("run"), r.get("leaf_value")), {})[str(r.get("dimension"))] = r

    def _pivot_block(metric: str, title: str) -> List[str]:
        block = [
            "",
            f"## {title}",
            "",
            "| run | leaf | " + " | ".join(DIMENSION_ORDER) + " |",
            "|---|---:|" + "|".join("---:" for _ in DIMENSION_ORDER) + "|",
        ]
        for (run, leaf) in sorted(pivot, key=lambda k: (str(k[0]), int(k[1] or 0))):
            cells = []
            for dim in DIMENSION_ORDER:
                r = pivot[(run, leaf)].get(dim)
                cells.append(_fmt(r.get(metric)) if r else "n/a")
            block.append(f"| {run} | {leaf} | " + " | ".join(cells) + " |")
        return block

    lines += _pivot_block(
        "internal_f_pearson",
        "Final g-stage internal_f Pearson (prediction vs teacher rollup) per (run, leaf) x dimension",
    )
    lines += _pivot_block(
        "internal_f_mae_1_7",
        "Final g-stage internal_f MAE (lower = better) per (run, leaf) x dimension",
    )

    lines += [
        "",
        "## Final composed g-stage per (run, leaf, dimension)",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for row in final_rows:
        lines.append("| " + " | ".join(_fmt(row.get(c)) for c in cols) + " |")

    lines += [
        "",
        "## All iterations",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for row in all_rows:
        lines.append("| " + " | ".join(_fmt(row.get(c)) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="label=run_dir pairs.")
    parser.add_argument("--split", default="test")
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
        label, run_dir = _resolve_run(str(spec))
        files = _iter_history_files(run_dir)
        if not files:
            LOGGER.warning("run %s (%s): no iteration_history.json found", label, run_dir)
        family = substrate = None
        for f in files:
            hist = json.loads(f.read_text(encoding="utf-8"))
            family = family or hist.get("family")
            substrate = substrate or hist.get("substrate_model")
            rows = _rows_from_history(label, hist, args.split)
            if not rows:
                LOGGER.warning("%s: no per_dimension for split %s", f, args.split)
            all_rows.extend(rows)
        provenance.append(
            {"run": label, "family": family, "substrate_model": substrate, "n_files": len(files)}
        )

    all_rows.sort(
        key=lambda r: (
            str(r.get("run")),
            int(r.get("leaf_value") or 0),
            int(r.get("iteration") or 0),
            _dim_rank(r.get("dimension")),
        )
    )
    final_rows = _select_final(all_rows)

    output_dir = args.output_dir or Path("outputs") / (
        "manifesto_qsentence_perdim_comparison_"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cols = list(KEY_COLUMNS) + list(METRIC_COLUMNS)
    csv_path = output_dir / "per_dimension.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    md_path = output_dir / "per_dimension.md"
    _write_markdown(all_rows, final_rows, md_path, provenance=provenance, split=args.split)
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(md_path.read_text(encoding="utf-8"))
    LOGGER.info("Wrote %s and %s", csv_path, md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
