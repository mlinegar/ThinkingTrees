#!/usr/bin/env python3
"""
Produce the Benoit-vs-ours comparison table for the paper.

Rows = method / pipeline / model configuration (Benoit-published literals +
our run-result JSONs). Columns = 6 policy dimensions + Macro. Default metric
is Pearson r; --metric selects other fields from our report.jsons (mae,
rmse, spearman, ci_width, n).

Extensible:
  * Add new Benoit-literal rows by appending to `LITERAL_ROWS`.
  * Add new result-file rows by adding a resolver to `FILE_ROWS`.
  * `--metric` maps to a field name across all report-based cells.

Usage:
    python scripts/comparison_table.py                     # pearson r
    python scripts/comparison_table.py --metric mae
    python scripts/comparison_table.py --out-md outputs/comparison_table.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Optional

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

_DIM_ORDER = ["economic", "social", "immigration", "eu", "environment", "decentralization"]
_DIM_LABEL = {
    "economic": "Economic",
    "social": "Social",
    "immigration": "Immigration",
    "eu": "EU",
    "environment": "Environment",
    "decentralization": "Decentral.",
}
_ROOT = project_root

# Rescore prefix: when set, all file-based resolvers prepend this subdirectory
# under outputs/. Used to render tables at (T, N) configs other than T=0, N=1.
# Set via --rescore-key flag; formatted as "T{T}_N{N}" (e.g. "T0.2_N3").
_RESCORE_PREFIX: Optional[str] = None


def _outputs_root() -> Path:
    """Base path for all file-based resolvers. Swaps to the rescored tree when
    _RESCORE_PREFIX is set."""
    if _RESCORE_PREFIX:
        return _ROOT / "outputs" / "rescore" / _RESCORE_PREFIX
    return _ROOT / "outputs"


# ------------------------------------------------------------------
# Benoit-published literals (pearson r; other metrics not applicable)
# ------------------------------------------------------------------
LITERAL_ROWS = [
    # (label, {dim: value, ...})
    ("Benoit Fig 1 proprietary ensemble (GPT-4o + Claude + Gemini, 18 scores)", {
        "economic": 0.87, "social": 0.92, "immigration": 0.89,
        "eu": 0.91, "environment": 0.82, "decentralization": 0.49,
    }),
    ("Benoit Table 3 expert upper bound", {
        "economic": 0.88, "social": 0.91, "immigration": 0.88,
        "eu": 0.95, "environment": 0.84, "decentralization": 0.78,
    }),
    ("Benoit Table 6 LLaMA-3.3-70B", {
        "economic": 0.84, "social": 0.87, "immigration": 0.86,
        "eu": 0.86, "environment": 0.68, "decentralization": 0.40,
    }),
    ("Benoit Table 6 DeepSeek-V3", {
        "economic": 0.84, "social": 0.87, "immigration": 0.89,
        "eu": 0.86, "environment": 0.79, "decentralization": 0.45,
    }),
    ("Benoit Table 6 Gemma-3-27B-IT", {
        "economic": 0.86, "social": 0.86, "immigration": 0.89,
        "eu": 0.84, "environment": 0.86, "decentralization": 0.45,
    }),
]


def _read(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _field(rep: Optional[dict], metric: str) -> Optional[float]:
    """Pull the requested metric from a pearson-report dict."""
    if rep is None:
        return None
    key = {
        "pearson": "pearson_r",
        "mae": "mae_rescaled",
        "rmse": "rmse_rescaled",
        "spearman": "spearman_r",
        "n": "n",
    }.get(metric)
    if key is None:
        raise ValueError(f"unknown metric {metric!r}")
    v = rep.get(key)
    if v is None:
        return None
    return float(v) if metric != "n" else int(v)


def _ci_width(rep: Optional[dict]) -> Optional[float]:
    if rep is None:
        return None
    lo, hi = rep.get("pearson_ci_low"), rep.get("pearson_ci_high")
    if lo is None or hi is None:
        return None
    return float(hi) - float(lo)


# ------------------------------------------------------------------
# File-row resolvers: each returns {dim: float|None} for a given metric.
# ------------------------------------------------------------------

def _phase0_scorer_only(metric: str) -> dict[str, Optional[float]]:
    """phase0/scorer_only: per-dim report.json with `ours_vs_expert` sub-dict."""
    out = {}
    for dim in _DIM_ORDER:
        rep = _read(_outputs_root() / "overnight_benoit" / "scorer_only" / dim / "report.json")
        ours = rep.get("ours_vs_expert") if rep else None
        out[dim] = _field(ours, metric) if metric != "ci_width" else _ci_width(ours)
    return out


def _phase0_full_pipeline(metric: str) -> dict[str, Optional[float]]:
    """phase0/full_pipeline (per-dim at chunk=24K)."""
    out = {}
    for dim in _DIM_ORDER:
        rep = _read(_outputs_root() / "overnight_benoit" / "full_pipeline" / dim / "report.json")
        sub = rep.get("report") if rep else None
        out[dim] = _field(sub, metric) if metric != "ci_width" else _ci_width(sub)
    return out


def _phase0_optimizer(metric: str, kind: str) -> dict[str, Optional[float]]:
    """phase0/optimizer_bootstrap: kind='baseline_test' or 'optimized_test'."""
    out = {}
    for dim in _DIM_ORDER:
        rep = _read(_outputs_root() / "overnight_benoit" / "optimizer_bootstrap" / dim / "report.json")
        sub = rep.get(kind) if rep else None
        out[dim] = _field(sub, metric) if metric != "ci_width" else _ci_width(sub)
    return out


def _phase2_combined(metric: str) -> dict[str, Optional[float]]:
    rep = _read(_outputs_root() / "phase2" / "combined_pipeline" / "report.json")
    per_dim = rep.get("per_dim") if rep else None
    out = {}
    for dim in _DIM_ORDER:
        sub = per_dim.get(dim) if per_dim else None
        out[dim] = _field(sub, metric) if metric != "ci_width" else _ci_width(sub)
    return out


def _phase2_joint(metric: str, kind: str, report_name: str) -> dict[str, Optional[float]]:
    """kind='baseline' or 'optimized'. report_name='joint_optimize' or 'joint_gepa'."""
    rep = _read(_outputs_root() / "phase2" / report_name / "report.json")
    branch = rep.get(kind) if rep else None
    per_dim = branch.get("per_dim") if branch else None
    out = {}
    for dim in _DIM_ORDER:
        sub = per_dim.get(dim) if per_dim else None
        out[dim] = _field(sub, metric) if metric != "ci_width" else _ci_width(sub)
    return out


def _chunk_sweep_per_dim(chunk_chars: int, metric: str) -> dict[str, Optional[float]]:
    out = {}
    for dim in _DIM_ORDER:
        rep = _read(_outputs_root() / "chunk_sweep" / f"{dim}_c{chunk_chars}" / "report.json")
        sub = rep.get("report") if rep else None
        out[dim] = _field(sub, metric) if metric != "ci_width" else _ci_width(sub)
    return out


def _phase3_combined(chunk_chars: int, metric: str) -> dict[str, Optional[float]]:
    rep = _read(_outputs_root() / "phase3" / f"combined_c{chunk_chars}" / "report.json")
    per_dim = rep.get("per_dim") if rep else None
    out = {}
    for dim in _DIM_ORDER:
        sub = per_dim.get(dim) if per_dim else None
        out[dim] = _field(sub, metric) if metric != "ci_width" else _ci_width(sub)
    return out


def _gemma3_scorer_only(metric: str) -> dict[str, Optional[float]]:
    """Gemma-3-27B scorer-only run (direct Benoit apples-to-apples).
    Expected output path: outputs/gemma3/scorer_only/{dim}/report.json."""
    out = {}
    for dim in _DIM_ORDER:
        rep = _read(_outputs_root() / "gemma3" / "scorer_only" / dim / "report.json")
        ours = rep.get("ours_vs_expert") if rep else None
        out[dim] = _field(ours, metric) if metric != "ci_width" else _ci_width(ours)
    return out


def _gemma3_scorer_benoit_rubric(metric: str) -> dict[str, Optional[float]]:
    """Gemma-3-27B scorer-only using Benoit's exact scoring rubric (extracted
    from his data_masked.csv SystemMessages)."""
    out = {}
    for dim in _DIM_ORDER:
        rep = _read(_outputs_root() / "gemma3" / "scorer_only_benoit_rubric" / dim / "report.json")
        ours = rep.get("ours_vs_expert") if rep else None
        out[dim] = _field(ours, metric) if metric != "ci_width" else _ci_width(ours)
    return out


def _gemma3_scorer_raw_benoit(metric: str) -> dict[str, Optional[float]]:
    """Gemma-3-27B scored via raw chat completions (no DSPy): Benoit's exact
    SystemMessage rubric + HumanMessage 'Analyze the following political text:\\n\\n...',
    bare-integer response. This is the closest possible replication of
    Benoit's native LangChain inference path."""
    out = {}
    for dim in _DIM_ORDER:
        rep = _read(_outputs_root() / "gemma3" / "scorer_raw_benoit" / dim / "report.json")
        ours = rep.get("ours_vs_expert") if rep else None
        out[dim] = _field(ours, metric) if metric != "ci_width" else _ci_width(ours)
    return out


def _ablation(subdir: str, metric: str) -> dict[str, Optional[float]]:
    """Load outputs/ablations/{subdir}/{dim}/report.json (same schema as phase0 full_pipeline)."""
    out = {}
    for dim in _DIM_ORDER:
        rep = _read(_outputs_root() / "ablations" / subdir / dim / "report.json")
        sub = rep.get("report") if rep else None
        out[dim] = _field(sub, metric) if metric != "ci_width" else _ci_width(sub)
    return out


def _flat_cap(cap: int, metric: str) -> dict[str, Optional[float]]:
    return _ablation(f"flat_t{cap}", metric)


def _concat_chunk(chunk: int, metric: str) -> dict[str, Optional[float]]:
    return _ablation(f"concat_c{chunk}", metric)


def _phase3_per_dim_gepa(chunk: int, kind: str, metric: str) -> dict[str, Optional[float]]:
    """phase3 per-dim full-pipeline GEPA. kind='baseline' or 'optimized'.
    chunk=24000 uses outputs/phase3/gepa_{dim}/; others gepa_{dim}_c{chunk}/."""
    out = {}
    for dim in _DIM_ORDER:
        if chunk == 24000:
            path = _outputs_root() / "phase3" / f"gepa_{dim}" / "report.json"
        else:
            path = _outputs_root() / "phase3" / f"gepa_{dim}_c{chunk}" / "report.json"
        rep = _read(path)
        sub = rep.get(kind) if rep else None
        out[dim] = _field(sub, metric) if metric != "ci_width" else _ci_width(sub)
    return out


def _phase3_combined_gepa(chunk: int, kind: str, metric: str) -> dict[str, Optional[float]]:
    """Combined full-pipeline GEPA. kind='baseline' or 'optimized'."""
    rep = _read(_outputs_root() / "phase3" / f"combined_gepa_c{chunk}" / "report.json")
    branch = rep.get(kind) if rep else None
    per_dim = branch.get("per_dim") if branch else None
    out = {}
    for dim in _DIM_ORDER:
        sub = per_dim.get(dim) if per_dim else None
        out[dim] = _field(sub, metric) if metric != "ci_width" else _ci_width(sub)
    return out


def _gemma3_full_pipeline(chunk_chars: int, metric: str) -> dict[str, Optional[float]]:
    """Gemma-3-27B full-pipeline by dim (if we run it). Path like
    outputs/gemma3/full_pipeline/{dim}_c{chunk}/report.json."""
    out = {}
    for dim in _DIM_ORDER:
        rep = _read(_outputs_root() / "gemma3" / "full_pipeline" / f"{dim}_c{chunk_chars}" / "report.json")
        sub = rep.get("report") if rep else None
        out[dim] = _field(sub, metric) if metric != "ci_width" else _ci_width(sub)
    return out


# Rows are grouped into *sections* so the table reads as a method-ablation
# story. Each section has a header + rows. When a row's data isn't available
# it renders as "—" but still shows up (unless --hide-missing-rows).
SECTIONS: list[dict] = [
    {
        "title": "Benoit (2026 AJPS) reference",
        "rows": [
            ("Proprietary ensemble, 18 scores (Fig 1)",
             {"economic": 0.87, "social": 0.92, "immigration": 0.89,
              "eu": 0.91, "environment": 0.82, "decentralization": 0.49}),
            ("Expert upper bound (Table 3)",
             {"economic": 0.88, "social": 0.91, "immigration": 0.88,
              "eu": 0.95, "environment": 0.84, "decentralization": 0.78}),
            ("LLaMA-3.3-70B (Table 6)",
             {"economic": 0.84, "social": 0.87, "immigration": 0.86,
              "eu": 0.86, "environment": 0.68, "decentralization": 0.40}),
            ("DeepSeek-V3 (Table 6)",
             {"economic": 0.84, "social": 0.87, "immigration": 0.89,
              "eu": 0.86, "environment": 0.79, "decentralization": 0.45}),
            ("Gemma-3-27B-IT (Table 6)",
             {"economic": 0.86, "social": 0.86, "immigration": 0.89,
              "eu": 0.84, "environment": 0.86, "decentralization": 0.45}),
        ],
    },
    {
        "title": "Ours: per-dim pipeline × leaf size (Gemma-4-31B-NVFP4, 1 summarizer + 1 scorer per dim)",
        "rows": [
            ("leaf = 64 K chars (≈16K tokens)",
             lambda m: _chunk_sweep_per_dim(64000, m)),
            ("leaf = 32 K chars (≈8K tokens)",
             lambda m: _chunk_sweep_per_dim(32000, m)),
            ("leaf = 24 K chars (≈6K tokens) — full test n≈215",
             _phase0_full_pipeline),
            ("leaf = 16 K chars (≈4K tokens)",
             lambda m: _chunk_sweep_per_dim(16000, m)),
            ("leaf =  8 K chars (≈2K tokens)",
             lambda m: _chunk_sweep_per_dim(8000, m)),
        ],
    },
    {
        "title": "Ours: combined pipeline × leaf size (one shared summarizer w/ JOINT_RUBRIC → 6 scores)",
        "rows": [
            ("leaf = 64 K chars",
             lambda m: _phase3_combined(64000, m)),
            ("leaf = 32 K chars",
             lambda m: _phase3_combined(32000, m)),
            ("leaf = 24 K chars (full test n=229)",
             _phase2_combined),
            ("leaf = 16 K chars",
             lambda m: _phase3_combined(16000, m)),
            ("leaf =  8 K chars",
             lambda m: _phase3_combined(8000, m)),
        ],
    },
    {
        "title": "Ours: tiny-leaf extensions of the chunk sweep (Gemma-4)",
        "rows": [
            ("tree, leaf = 4 K chars",
             lambda m: _chunk_sweep_per_dim(4000, m)),
            ("tree, leaf = 2 K chars",
             lambda m: _chunk_sweep_per_dim(2000, m)),
            ("tree, leaf = 1 K chars (stress test, depth ≥6)",
             lambda m: _chunk_sweep_per_dim(1000, m)),
        ],
    },
    {
        "title": "Ours: concat-no-merge (chunks summarized independently, joined, scored — tests whether the merge step carries signal)",
        "rows": [
            ("concat, leaf = 32 K chars",
             lambda m: _concat_chunk(32000, m)),
            ("concat, leaf = 16 K chars (default)",
             lambda m: _ablation("concat", m)),
            ("concat, leaf =  8 K chars",
             lambda m: _concat_chunk(8000, m)),
        ],
    },
    {
        "title": "Ours: flat baseline (no chunk, no summary; truncate text and score) — tests whether tree is needed at all",
        "rows": [
            ("flat, truncation = 48 K chars",
             lambda m: _flat_cap(48000, m)),
            ("flat, truncation = 24 K chars (default)",
             lambda m: _ablation("flat", m)),
            ("flat, truncation = 12 K chars",
             lambda m: _flat_cap(12000, m)),
            ("flat, truncation =  6 K chars",
             lambda m: _flat_cap(6000, m)),
        ],
    },
    {
        "title": "Ours: full-pipeline GEPA per-dim joint optimization on pooled train set",
        "rows": [
            ("GEPA per-dim, leaf = 24 K (baseline before optimization)",
             lambda m: _phase3_per_dim_gepa(24000, "baseline", m)),
            ("GEPA per-dim, leaf = 24 K (optimized)",
             lambda m: _phase3_per_dim_gepa(24000, "optimized", m)),
            ("GEPA per-dim, leaf = 16 K (optimized)",
             lambda m: _phase3_per_dim_gepa(16000, "optimized", m)),
            ("GEPA per-dim, leaf =  8 K (optimized)",
             lambda m: _phase3_per_dim_gepa(8000, "optimized", m)),
        ],
    },
    {
        "title": "Ours: full-pipeline GEPA combined joint optimization (one shared program across 6 dims, JOINT_RUBRIC)",
        "rows": [
            ("Combined GEPA, leaf = 24 K (baseline)",
             lambda m: _phase3_combined_gepa(24000, "baseline", m)),
            ("Combined GEPA, leaf = 24 K (optimized)",
             lambda m: _phase3_combined_gepa(24000, "optimized", m)),
            ("Combined GEPA, leaf = 16 K (optimized)",
             lambda m: _phase3_combined_gepa(16000, "optimized", m)),
            ("Combined GEPA, leaf =  8 K (optimized)",
             lambda m: _phase3_combined_gepa(8000, "optimized", m)),
        ],
    },
    {
        "title": "Ours: scorer ablations (Gemma-4, Benoit's GPT-4o summaries held fixed)",
        "rows": [
            ("1. zero-shot scorer (phase0 baseline)",
             _phase0_scorer_only),
            ("2. BootstrapFewShot-optimized scorer (phase0 optimized)",
             lambda m: _phase0_optimizer(m, "optimized_test")),
            ("3. Joint scorer baseline (shared Predict, 6 dims) — phase2",
             lambda m: _phase2_joint(m, "baseline", "joint_optimize")),
            ("4. Joint scorer BFS-optimized — phase2",
             lambda m: _phase2_joint(m, "optimized", "joint_optimize")),
            ("5. Joint scorer GEPA-optimized — phase2",
             lambda m: _phase2_joint(m, "optimized", "joint_gepa")),
        ],
    },
    {
        "title": "Ours: exact-model replication (Benoit's Gemma-3-27B-IT BF16)",
        "rows": [
            ("Gemma-3-27B scorer-only, OUR scoring rubric",
             _gemma3_scorer_only),
            ("Gemma-3-27B scorer-only, BENOIT's exact rubric (from data_masked.csv)",
             _gemma3_scorer_benoit_rubric),
            ("Gemma-3-27B, raw-prompt Benoit format (no DSPy, bare-integer response)",
             _gemma3_scorer_raw_benoit),
            ("Gemma-3-27B, per-dim pipeline (chunk=24K)",
             lambda m: _gemma3_full_pipeline(24000, m)),
            ("Gemma-3-27B, per-dim pipeline (chunk= 8K)",
             lambda m: _gemma3_full_pipeline(8000, m)),
        ],
    },
]


def _macro(values: dict[str, Optional[float]]) -> Optional[float]:
    present = [v for v in values.values() if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _cell(v: Optional[float], metric: str) -> str:
    if v is None:
        return "—"
    if metric == "n":
        return f"{v:.0f}"
    return f"{v:+.3f}" if metric in {"pearson", "spearman"} else f"{v:.3f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metric", default="pearson",
                   choices=["pearson", "mae", "rmse", "spearman", "n", "ci_width"],
                   help="Which metric to fill each cell with (default: pearson r).")
    p.add_argument("--rescore-key", default=None,
                   help="Load cells from outputs/rescore/<key>/ instead of outputs/. "
                        "Format: 'T{T}_N{N}', e.g. 'T0.2_N3'.")
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument("--out-tex", type=Path, default=None,
                   help="Write a LaTeX booktabs table in addition to (or instead of) markdown.")
    p.add_argument("--hide-missing-rows", action="store_true",
                   help="Skip rows where all dimensions are '—'.")
    args = p.parse_args()

    # Activate the rescored tree if asked.
    global _RESCORE_PREFIX
    _RESCORE_PREFIX = args.rescore_key

    # Render markdown, one section at a time
    lines: list[str] = []
    metric_title = {
        "pearson": "Pearson r (higher better)",
        "mae": "Mean absolute error on 1-7 scale (lower better)",
        "rmse": "RMSE on 1-7 scale (lower better)",
        "spearman": "Spearman ρ",
        "n": "n (sample size)",
        "ci_width": "95% CI width (lower = tighter)",
    }[args.metric]
    lines.append(f"# Benoit-vs-ours comparison — {metric_title}\n")
    lines.append("Columns: 6 policy dimensions + Macro (unweighted mean of available cells).\n")

    for section in SECTIONS:
        lines.append(f"## {section['title']}\n")
        header = "|Method|" + "|".join(_DIM_LABEL[d] for d in _DIM_ORDER) + "|Macro|coverage|"
        sep = "|---|" + "|".join("---:" for _ in _DIM_ORDER) + "|---:|---:|"
        lines.append(header)
        lines.append(sep)
        for label, source in section["rows"]:
            if callable(source):
                values = source(args.metric)
            else:
                # Literal dict — only meaningful for pearson
                values = source if args.metric == "pearson" else {d: None for d in _DIM_ORDER}
            if args.hide_missing_rows and not any(v is not None for v in values.values()):
                continue
            cells = [_cell(values.get(d), args.metric) for d in _DIM_ORDER]
            macro = _macro(values)
            macro_cell = _cell(macro, args.metric)
            coverage = sum(1 for d in _DIM_ORDER if values.get(d) is not None)
            lines.append(f"|{label}|" + "|".join(cells) + f"|{macro_cell}|{coverage}/6|")
        lines.append("")  # blank line between sections

    out = "\n".join(lines) + "\n"
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(out)
    if args.out_tex:
        args.out_tex.parent.mkdir(parents=True, exist_ok=True)
        args.out_tex.write_text(_as_latex(args.metric, metric_title))
    print(out)
    return 0


def _as_latex(metric: str, metric_title: str) -> str:
    """Emit a single booktabs LaTeX table body covering all sections (one
    tabular per section, grouped via `\\midrule[heavy]` and `\\multicolumn`
    headers). Designed to paste into paper/ctreepo/sections/XX_*.tex inside
    a `table` environment.
    """
    dim_tex = {
        "economic": "Econ.", "social": "Social", "immigration": "Immig.",
        "eu": "EU", "environment": "Env.", "decentralization": "Decent.",
    }
    parts: list[str] = []
    parts.append("% Auto-generated by scripts/comparison_table.py — do not edit by hand.")
    parts.append(f"% Metric: {metric_title}")
    parts.append(r"\begin{tabular}{lrrrrrrrr}")
    parts.append(r"\toprule")
    parts.append("Method & " + " & ".join(dim_tex[d] for d in _DIM_ORDER) + r" & Macro & \#/6 \\")
    parts.append(r"\midrule")
    for section in SECTIONS:
        parts.append(r"\multicolumn{9}{l}{\textit{" + _tex_escape(section["title"]) + r"}} \\")
        for label, source in section["rows"]:
            if callable(source):
                values = source(metric)
            else:
                values = source if metric == "pearson" else {d: None for d in _DIM_ORDER}
            if not any(v is not None for v in values.values()):
                # still include row with em-dashes, for clarity about what's pending
                pass
            cells = [_latex_cell(values.get(d), metric) for d in _DIM_ORDER]
            macro = _macro(values)
            macro_cell = _latex_cell(macro, metric)
            coverage = sum(1 for d in _DIM_ORDER if values.get(d) is not None)
            parts.append(
                _tex_escape(label) + " & " + " & ".join(cells)
                + f" & {macro_cell} & {coverage}/6 \\\\"
            )
        parts.append(r"\midrule")
    # replace last midrule with bottomrule
    parts[-1] = r"\bottomrule"
    parts.append(r"\end{tabular}")
    return "\n".join(parts) + "\n"


def _latex_cell(v: Optional[float], metric: str) -> str:
    if v is None:
        return "—"
    if metric == "n":
        return f"{v:.0f}"
    if metric in {"pearson", "spearman"}:
        return f"${v:+.3f}$"
    return f"${v:.3f}$"


def _tex_escape(s: str) -> str:
    return (s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
              .replace("#", r"\#")
              .replace("≈", r"$\approx$").replace("≥", r"$\geq$").replace("≤", r"$\leq$")
              .replace("—", r"---").replace("×", r"$\times$"))


if __name__ == "__main__":
    sys.exit(main())
