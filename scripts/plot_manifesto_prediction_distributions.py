#!/usr/bin/env python3
"""Plot Benoit expert means against manifesto prediction scores.

The source pipeline gives one direct LLM 1-7 prediction per manifesto. Recent
alternating-ladder runs may also persist per-document C-TreePO predictions under
``ladder/*/leafXXXXtok/prediction_records``. This script combines those sources
into jittered expert-vs-prediction panels with a 45-degree agreement line.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_DIMENSIONS = (
    "economic",
    "social",
    "immigration",
    "eu",
    "environment",
    "decentralization",
)

MODEL_STYLES = (
    {"color": "#d55e00", "marker": "o"},
    {"color": "#2878b5", "marker": "s"},
    {"color": "#009e73", "marker": "^"},
    {"color": "#cc79a7", "marker": "D"},
    {"color": "#7f7f7f", "marker": "P"},
)


def _style_for_model(model: str, fallback_index: int) -> Dict[str, str]:
    label = str(model).lower()
    if "tree" in label or "ctreepo" in label or "c-treepo" in label:
        return {"color": "#009e73", "marker": "^"}
    if "benoit" in label or "manifesto" in label:
        return {"color": "#2878b5", "marker": "s"}
    if "direct" in label:
        return {"color": "#d55e00", "marker": "o"}
    return MODEL_STYLES[fallback_index % len(MODEL_STYLES)]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    mean_x = sum(x for x, _ in pairs) / len(pairs)
    mean_y = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x, _ in pairs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for _, y in pairs))
    if den_x <= 0 or den_y <= 0:
        return None
    return float(num / (den_x * den_y))


def _mae(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if not pairs:
        return None
    return float(sum(abs(x - y) for x, y in pairs) / len(pairs))


def _ols_fit(xs: Sequence[float], ys: Sequence[float]) -> Optional[Tuple[float, float]]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    mean_x = sum(x for x, _ in pairs) / len(pairs)
    mean_y = sum(y for _, y in pairs) / len(pairs)
    var_x = sum((x - mean_x) ** 2 for x, _ in pairs)
    if var_x <= 0:
        return None
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    return float(intercept), float(slope)


def _fmt_metric(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _record_doc_id(row: Mapping[str, Any]) -> Optional[str]:
    for key in ("doc_id", "manifesto_id", "benoit_manifesto_key"):
        value = row.get(key)
        if value is not None and str(value):
            return str(value)
    return None


def _load_direct_records(source_root: Path, dimension: str, label: str) -> List[Dict[str, Any]]:
    path = Path(source_root) / dimension / "per_manifesto.jsonl"
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    for row in _read_jsonl(path):
        expert = _safe_float(row.get("benoit_expert_mean"))
        pred = _safe_float(row.get("llm_score_1_7"))
        if expert is None or pred is None:
            continue
        doc_id = _record_doc_id(row)
        records.append(
            {
                "dimension": dimension,
                "model": label,
                "doc_id": doc_id,
                "manifesto_id": row.get("manifesto_id"),
                "benoit_manifesto_key": row.get("benoit_manifesto_key"),
                "party_abbrev": row.get("party_abbrev"),
                "country_name": row.get("country_name"),
                "year": row.get("year"),
                "expert_mean_1_7": expert,
                "prediction_1_7": pred,
                "source": str(path),
                "split": "all",
            }
        )
    return records


def _load_benoit_reference_records(
    dimension: str,
    *,
    kind: str,
    label: str,
    dataverse_dir: Optional[Path],
    restrict_manifestos: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    from src.tasks.manifesto.dimensions import PolicyDimension
    from src.tasks.manifesto.expert_benchmarks import (
        benoit_ensemble_mean,
        load_benoit_expert_means,
        load_benoit_llm_scores,
    )

    dim = PolicyDimension(str(dimension))
    llm = load_benoit_llm_scores(kind=kind, dimension=dim, dataverse_dir=dataverse_dir)
    ensemble = benoit_ensemble_mean(llm)
    experts = load_benoit_expert_means(dim, dataverse_dir=dataverse_dir)
    merged = ensemble.merge(
        experts[["manifesto", "expert_mean"]], on="manifesto", how="left"
    )
    if restrict_manifestos:
        merged = merged[merged["manifesto"].astype(str).isin(restrict_manifestos)]

    records: List[Dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        expert = _safe_float(getattr(row, "expert_mean", None))
        pred = _safe_float(getattr(row, "score_llm_mean", None))
        manifesto = str(getattr(row, "manifesto", ""))
        if expert is None or pred is None or not manifesto:
            continue
        records.append(
            {
                "dimension": dimension,
                "model": label,
                "doc_id": manifesto,
                "manifesto_id": None,
                "benoit_manifesto_key": manifesto,
                "party_abbrev": None,
                "country_name": None,
                "year": None,
                "expert_mean_1_7": expert,
                "prediction_1_7": pred,
                "source": f"benoit_dataverse:data_llms_all_{kind}.rds",
                "split": "all",
            }
        )
    return records


def _parse_mapping_args(values: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise SystemExit(f"expected DIM=PATH mapping, got {raw!r}")
        dim, path = raw.split("=", 1)
        dim = dim.strip().lower()
        if not dim:
            raise SystemExit(f"empty dimension in mapping {raw!r}")
        out[dim] = Path(path).expanduser()
    return out


def _split_metrics_for_iteration(iteration: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    metrics = iteration.get("split_metrics") or {}
    if not isinstance(metrics, Mapping):
        return {}
    selected = metrics.get(split) or metrics.get("all") or {}
    return selected if isinstance(selected, Mapping) else {}


def _best_prediction_record_path(
    ladder_root: Path,
    *,
    split: str,
    dimension: Optional[str] = None,
) -> Tuple[Optional[Path], Optional[Mapping[str, Any]], Optional[Path]]:
    best_path: Optional[Path] = None
    best_iteration: Optional[Mapping[str, Any]] = None
    best_manifest: Optional[Path] = None
    best_score = -math.inf

    for manifest_path in sorted(Path(ladder_root).glob("ladder/*/leaf*tok/iteration_history.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for iteration in manifest.get("iterations", []) or []:
            if not isinstance(iteration, Mapping):
                continue
            iter_num = iteration.get("iteration")
            if iter_num is None:
                continue
            record_path = (
                manifest_path.parent
                / "prediction_records"
                / f"iter_{int(iter_num):02d}_post_eval.jsonl"
            )
            if not record_path.exists():
                continue
            metrics = _split_metrics_for_iteration(iteration, split)
            if dimension is not None:
                per_dimension = metrics.get("per_dimension")
                if isinstance(per_dimension, Mapping):
                    dim_metrics = per_dimension.get(str(dimension))
                    if isinstance(dim_metrics, Mapping):
                        metrics = dim_metrics
            score = _safe_float(metrics.get("external_expert_pearson"))
            if score is None:
                score = -math.inf
            if score > best_score:
                best_score = score
                best_path = record_path
                best_iteration = iteration
                best_manifest = manifest_path
    return best_path, best_iteration, best_manifest


def _load_ctreepo_records(
    ladder_root: Path,
    dimension: str,
    *,
    split: str,
    label: str,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    record_path, iteration, manifest_path = _best_prediction_record_path(
        ladder_root, split=split, dimension=dimension
    )
    if record_path is None or iteration is None:
        return [], None

    rows: List[Dict[str, Any]] = []
    for row in _read_jsonl(record_path):
        row_dim = row.get("dimension")
        if row_dim is not None and str(row_dim) != dimension:
            continue
        row_split = str(row.get("split") or "").lower()
        if split != "all" and row_split and row_split != split.lower():
            continue
        expert = _safe_float(row.get("expert_score_1_7") or row.get("expert_mean_1_7"))
        pred = _safe_float(row.get("prediction_1_7"))
        if expert is None or pred is None:
            continue
        doc_id = _record_doc_id(row)
        rows.append(
            {
                "dimension": dimension,
                "model": label,
                "doc_id": doc_id,
                "manifesto_id": row.get("manifesto_id") or doc_id,
                "party_abbrev": row.get("party_abbrev"),
                "country_name": row.get("country_name"),
                "year": row.get("year"),
                "expert_mean_1_7": expert,
                "prediction_1_7": pred,
                "source": str(record_path),
                "split": row.get("split") or split,
            }
        )

    metrics = _split_metrics_for_iteration(iteration, split)
    per_dimension = metrics.get("per_dimension")
    if isinstance(per_dimension, Mapping):
        dim_metrics = per_dimension.get(str(dimension))
        if isinstance(dim_metrics, Mapping):
            metrics = dim_metrics
    meta = {
        "dimension": dimension,
        "record_path": str(record_path),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "iteration": iteration.get("iteration"),
        "stage_label": iteration.get("stage_label"),
        "f_degree": iteration.get("f_degree"),
        "g_degree": iteration.get("g_degree"),
        "external_expert_pearson": metrics.get("external_expert_pearson"),
        "external_expert_mae_1_7": metrics.get("external_expert_mae_1_7"),
    }
    return rows, meta


def _stable_jitter_seed(*parts: str) -> int:
    text = "::".join(parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _jitter(values: Sequence[float], *, seed: int, scale: float = 0.045) -> List[float]:
    # Tiny deterministic LCG-style jitter avoids importing numpy just for scatter offsets.
    state = seed & 0xFFFFFFFF
    out: List[float] = []
    for value in values:
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        u1 = max((state + 1) / 4294967297.0, 1e-9)
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        u2 = (state + 1) / 4294967297.0
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        out.append(float(value) + scale * z)
    return out


def _summaries(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in records:
        grouped.setdefault((str(row["dimension"]), str(row["model"])), []).append(row)
    out: List[Dict[str, Any]] = []
    for (dim, model), rows in sorted(grouped.items()):
        xs = [float(r["expert_mean_1_7"]) for r in rows]
        ys = [float(r["prediction_1_7"]) for r in rows]
        fit = _ols_fit(xs, ys)
        out.append(
            {
                "dimension": dim,
                "model": model,
                "n": len(rows),
                "pearson_r": _pearson(xs, ys),
                "mae_1_7": _mae(xs, ys),
                "ols_intercept": fit[0] if fit is not None else None,
                "ols_slope": fit[1] if fit is not None else None,
                "mean_expert_1_7": sum(xs) / len(xs) if xs else None,
                "mean_prediction_1_7": sum(ys) / len(ys) if ys else None,
            }
        )
    return out


def _plot_grid(
    *,
    records: Sequence[Mapping[str, Any]],
    dimensions: Sequence[str],
    models: Sequence[str],
    output_path: Path,
    title: str,
    overlay_models: bool = True,
) -> None:
    nrows = max(1, len(dimensions))
    ncols = 1 if overlay_models else max(1, len(models))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.1 * ncols, 2.85 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    summary_by_key = {(row["dimension"], row["model"]): row for row in _summaries(records)}
    for r, dim in enumerate(dimensions):
        for c in range(ncols):
            ax = axes[r][c]
            panel_models = list(models) if overlay_models else [models[c]]
            ax.plot([1, 7], [1, 7], linestyle="--", linewidth=1.0, color="#737373", zorder=1)
            ax.set_xlim(0.75, 7.25)
            ax.set_ylim(0.75, 7.25)
            ax.set_xticks(range(1, 8))
            ax.set_yticks(range(1, 8))
            ax.grid(True, linewidth=0.35, color="#dddddd", alpha=0.75)
            plotted_any = False
            legend_labels: List[str] = []
            for model_idx, model in enumerate(panel_models):
                panel = [
                    row
                    for row in records
                    if row.get("dimension") == dim and row.get("model") == model
                ]
                if not panel:
                    continue
                plotted_any = True
                style = _style_for_model(str(model), model_idx)
                xs = [float(row["expert_mean_1_7"]) for row in panel]
                ys = [float(row["prediction_1_7"]) for row in panel]
                seed = _stable_jitter_seed(dim, model)
                ax.scatter(
                    _jitter(xs, seed=seed),
                    _jitter(ys, seed=seed ^ 0xA5A5A5A5),
                    s=23,
                    alpha=0.58 if overlay_models else 0.68,
                    linewidths=0.25,
                    edgecolors="white",
                    color=style["color"],
                    marker=style["marker"],
                    zorder=2,
                )
                fit = _ols_fit(xs, ys)
                if fit is not None:
                    intercept, slope = fit
                    line_x = [1.0, 7.0]
                    line_y = [intercept + slope * x for x in line_x]
                    ax.plot(
                        line_x,
                        line_y,
                        linewidth=1.6,
                        color=style["color"],
                        alpha=0.9,
                        zorder=3,
                    )
                summary = summary_by_key.get((dim, model), {})
                legend_labels.append(
                    f"r={_fmt_metric(_safe_float(summary.get('pearson_r')))}, "
                    f"slope={_fmt_metric(_safe_float(summary.get('ols_slope')), digits=2)}, "
                    f"MAE={_fmt_metric(_safe_float(summary.get('mae_1_7')), digits=2)}, "
                    f"n={summary.get('n', len(panel))}"
                )
                ax.scatter(
                    [],
                    [],
                    s=26,
                    color=style["color"],
                    marker=style["marker"],
                    label=f"{model}: {legend_labels[-1]}",
                )
            if plotted_any:
                ax.legend(
                    loc="upper left",
                    fontsize=7,
                    frameon=True,
                    framealpha=0.88,
                    borderpad=0.35,
                    handlelength=1.1,
                )
            else:
                ax.text(
                    4,
                    4,
                    "no cached\npredictions",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#777777",
                )
            if overlay_models:
                ax.set_title(str(dim), fontsize=10)
            elif r == 0:
                model = panel_models[0]
                summary = summary_by_key.get((dim, model), {})
                subtitle = (
                    f"r={_fmt_metric(_safe_float(summary.get('pearson_r')))}, "
                    f"slope={_fmt_metric(_safe_float(summary.get('ols_slope')), digits=2)}, "
                    f"MAE={_fmt_metric(_safe_float(summary.get('mae_1_7')), digits=2)}, "
                    f"n={summary.get('n', '')}"
                )
                ax.set_title(f"{model}\n{subtitle}", fontsize=10)
            else:
                model = panel_models[0]
                summary = summary_by_key.get((dim, model), {})
                subtitle = (
                    f"r={_fmt_metric(_safe_float(summary.get('pearson_r')))}, "
                    f"slope={_fmt_metric(_safe_float(summary.get('ols_slope')), digits=2)}, "
                    f"MAE={_fmt_metric(_safe_float(summary.get('mae_1_7')), digits=2)}, "
                    f"n={summary.get('n', '')}"
                )
                ax.set_title(subtitle, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{dim}\nPrediction", fontsize=10)
            if r == nrows - 1:
                ax.set_xlabel("Avg expert prediction", fontsize=10)
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    summaries = _summaries(rows)
    _write_csv(path.with_suffix(".csv"), summaries)
    path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="outputs/overnight_benoit/full_pipeline",
        help="Root containing per-dimension per_manifesto.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/manifesto_prediction_distributions",
        help="Directory for plots and CSV summaries.",
    )
    parser.add_argument(
        "--dimension",
        action="append",
        dest="dimensions",
        help="Dimension to plot. May be repeated. Defaults to all six dimensions.",
    )
    parser.add_argument(
        "--ladder-root",
        action="append",
        default=[],
        help="Optional DIM=PATH root for cached C-TreePO prediction records.",
    )
    parser.add_argument("--split", default="test", help="Split to use for cached C-TreePO records.")
    parser.add_argument("--direct-label", default="Direct LLM", help="Label for source llm_score_1_7.")
    parser.add_argument("--us-label", default="C-TreePO", help="Label for cached ladder predictions.")
    parser.add_argument(
        "--no-direct",
        action="store_true",
        help="Use the direct source predictions only for matching/restriction, not as a plotted series.",
    )
    parser.add_argument(
        "--benoit-kind",
        choices=["reported", "openweight", "replication"],
        default="reported",
        help="Benoit Dataverse LLM score file to overlay.",
    )
    parser.add_argument("--benoit-label", default="Benoit 2025", help="Legend label for Benoit scores.")
    parser.add_argument("--dataverse-dir", type=Path, default=None)
    parser.add_argument("--no-benoit-reference", action="store_true")
    parser.add_argument(
        "--no-restrict-benoit-to-source",
        action="store_true",
        help="Plot all Benoit Dataverse manifestos instead of the source-root subset.",
    )
    parser.add_argument(
        "--separate-model-panels",
        action="store_true",
        help="Use one column per model instead of overlaying models by color.",
    )
    parser.add_argument(
        "--no-restrict-direct",
        action="store_true",
        help="Do not restrict direct baseline points to the same docs as cached C-TreePO records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    dimensions = tuple(args.dimensions or DEFAULT_DIMENSIONS)
    ladder_roots = _parse_mapping_args(args.ladder_root)

    all_records: List[Dict[str, Any]] = []
    ctreepo_meta: List[Dict[str, Any]] = []
    models = [] if args.no_direct else [str(args.direct_label)]
    if not args.no_benoit_reference:
        models.append(str(args.benoit_label))
    any_us = False

    for dim in dimensions:
        direct = _load_direct_records(source_root, dim, str(args.direct_label))
        if not args.no_benoit_reference:
            source_manifestos = None
            if not args.no_restrict_benoit_to_source:
                source_manifestos = {
                    str(row["benoit_manifesto_key"])
                    for row in direct
                    if row.get("benoit_manifesto_key")
                }
            try:
                all_records.extend(
                    _load_benoit_reference_records(
                        dim,
                        kind=str(args.benoit_kind),
                        label=str(args.benoit_label),
                        dataverse_dir=args.dataverse_dir,
                        restrict_manifestos=source_manifestos,
                    )
                )
            except Exception as exc:
                print(
                    f"warning: could not load Benoit reference for {dim}: {exc}",
                    file=sys.stderr,
                )
        ctreepo_records: List[Dict[str, Any]] = []
        root = ladder_roots.get(dim)
        if root is not None:
            ctreepo_records, meta = _load_ctreepo_records(
                root, dim, split=str(args.split), label=str(args.us_label)
            )
            if meta is not None:
                ctreepo_meta.append(meta)
        if ctreepo_records:
            any_us = True
            if not args.no_restrict_direct:
                doc_ids = {row["doc_id"] for row in ctreepo_records if row.get("doc_id")}
                matched = [row for row in direct if row.get("doc_id") in doc_ids]
                if len(matched) >= max(1, int(0.5 * len(ctreepo_records))):
                    direct = matched
        if not args.no_direct:
            all_records.extend(direct)
        all_records.extend(ctreepo_records)

    if any_us:
        models.append(str(args.us_label))
    # Preserve order while dropping models that have no records.
    present_models = {str(row.get("model")) for row in all_records}
    models = [model for model in dict.fromkeys(models) if model in present_models]

    _write_csv(output_dir / "manifesto_prediction_distribution_points.csv", all_records)
    _write_summary(output_dir / "manifesto_prediction_distribution_summary.json", all_records)
    if ctreepo_meta:
        _write_csv(output_dir / "ctreepo_prediction_sources.csv", ctreepo_meta)
        (output_dir / "ctreepo_prediction_sources.json").write_text(
            json.dumps(ctreepo_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    title = "Manifesto Expert Mean vs Prediction"
    _plot_grid(
        records=all_records,
        dimensions=dimensions,
        models=models,
        output_path=output_dir / "manifesto_prediction_distribution_grid.png",
        title=title,
        overlay_models=not args.separate_model_panels,
    )
    _plot_grid(
        records=all_records,
        dimensions=dimensions,
        models=models,
        output_path=output_dir / "manifesto_prediction_distribution_grid.pdf",
        title=title,
        overlay_models=not args.separate_model_panels,
    )

    for dim in dimensions:
        dim_records = [row for row in all_records if row.get("dimension") == dim]
        if not dim_records:
            continue
        _plot_grid(
            records=dim_records,
            dimensions=(dim,),
            models=models,
            output_path=output_dir / f"{dim}_prediction_distribution.png",
            title=f"{dim}: Expert Mean vs Prediction",
            overlay_models=not args.separate_model_panels,
        )
        _plot_grid(
            records=dim_records,
            dimensions=(dim,),
            models=models,
            output_path=output_dir / f"{dim}_prediction_distribution.pdf",
            title=f"{dim}: Expert Mean vs Prediction",
            overlay_models=not args.separate_model_panels,
        )

    print(json.dumps({"output_dir": str(output_dir), "n_points": len(all_records)}, indent=2))


if __name__ == "__main__":
    main()
