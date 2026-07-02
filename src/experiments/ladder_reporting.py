from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_LADDER_METRIC_FIELDS = (
    "internal_f_pearson",
    "external_expert_pearson",
    "f_star_gap",
    "internal_f_mae",
    "external_expert_mae",
    "mean_prediction",
    "mean_teacher",
    "mean_expert",
    "internal_f_mae_1_7",
    "external_expert_mae_1_7",
    "mean_prediction_1_7",
    "mean_teacher_1_7",
    "mean_expert_1_7",
    "metrics_scale",
)


DEFAULT_ROW_FIELDS = (
    "family",
    "axis_kind",
    "axis_value",
    "leaf_count",
    "leaf_size_tokens",
)


def summarize_ladder_grid(
    grid_rows: Sequence[Mapping[str, Any]],
    *,
    eval_split: str,
    row_fields: Sequence[str] = DEFAULT_ROW_FIELDS,
    metric_fields: Sequence[str] = DEFAULT_LADDER_METRIC_FIELDS,
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for history in grid_rows:
        for iteration in history.get("iterations", []) or []:
            split_metrics = iteration.get("split_metrics", {}) or {}
            metrics = split_metrics.get(eval_split) or split_metrics.get("all") or {}
            row = {field: history.get(field) for field in row_fields}
            row.update(
                {
                    "iteration": iteration.get("iteration"),
                    "stage_name": iteration.get("stage_name"),
                    "stage_label": iteration.get("stage_label") or iteration.get("stage_name"),
                    "f_degree": iteration.get("f_degree"),
                    "g_degree": iteration.get("g_degree"),
                    "trained": iteration.get("trained"),
                    "n_eval": metrics.get("n"),
                }
            )
            for field in metric_fields:
                row[field] = metrics.get(field)
            table.append(row)
    return table


def format_metric(value: Any, *, width: int = 8, digits: int = 3) -> str:
    if value is None:
        return "n/a".rjust(width)
    if isinstance(value, int):
        return f"{value:>{width}d}"
    return f"{float(value):>{width}.{digits}f}"


def metric_or_fallback(row: Mapping[str, Any], primary: str, fallback: str) -> Any:
    value = row.get(primary)
    return row.get(fallback) if value is None else value


def ladder_axis_label(row: Mapping[str, Any]) -> str:
    leaf_size = row.get("leaf_size_tokens")
    if leaf_size is not None:
        return f"leaf{int(leaf_size):04d}tok"
    return f"leaf_{int(row.get('leaf_count') or row.get('axis_value') or 0):03d}"


def write_alternating_markdown_summary(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    eval_split: str,
    title: str = "Alternating ladder grid summary",
) -> None:
    header = (
        "| family | axis | k | stage | trained | n | int_p | ext_p | f_star_gap | "
        "int_mae | ext_mae | mean_p | mean_t | mean_e |"
    )
    sep = "|" + "|".join("-" * (len(seg) + 2) for seg in header.strip("|").split("|")) + "|"
    lines = [f"# {title} ({eval_split} split)", "", header, sep]
    for row in rows:
        lines.append(
            "| {family} | {axis} | {k} | {stage} | {trained} | {n} | {ip} | {ep} | {gap} | "
            "{im} | {em} | {mp} | {mt} | {me} |".format(
                family=row.get("family"),
                axis=ladder_axis_label(row),
                k=row.get("iteration"),
                stage=row.get("stage_label") or row.get("stage_name"),
                trained=row.get("trained"),
                n=format_metric(row.get("n_eval"), width=4, digits=0),
                ip=format_metric(row.get("internal_f_pearson")),
                ep=format_metric(row.get("external_expert_pearson")),
                gap=format_metric(row.get("f_star_gap")),
                im=format_metric(metric_or_fallback(row, "internal_f_mae", "internal_f_mae_1_7")),
                em=format_metric(metric_or_fallback(row, "external_expert_mae", "external_expert_mae_1_7")),
                mp=format_metric(metric_or_fallback(row, "mean_prediction", "mean_prediction_1_7")),
                mt=format_metric(metric_or_fallback(row, "mean_teacher", "mean_teacher_1_7")),
                me=format_metric(metric_or_fallback(row, "mean_expert", "mean_expert_1_7")),
            )
        )
    lines.append("")
    lines.append(
        "Columns: `int_p` = internal Pearson (our f vs teacher f at root); "
        "`ext_p` = external Pearson (our f vs gold expert); "
        "`f_star_gap` = int_p - ext_p (positive = reward-hacking warning)."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_qsentence_markdown_summary(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    eval_split: str,
) -> None:
    lines = [
        f"# Manifesto q-sentence DSPy ladder summary ({eval_split} split)",
        "",
        (
            "| leaf_q | k | stage | trained | n | int_p | ext_p | f_star_gap | "
            "int_mae | ext_mae | mean_p | mean_t | mean_e |"
        ),
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            (
                "| {leaf} | {k} | {stage} | {trained} | {n} | {ip} | {ep} | "
                "{gap} | {im} | {em} | {mp} | {mt} | {me} |"
            ).format(
                leaf=row.get("leaf_qsentences") or row.get("axis_value"),
                k=row.get("iteration"),
                stage=row.get("stage_label") or row.get("stage_name"),
                trained=row.get("trained"),
                n=format_metric(row.get("n_eval"), width=4, digits=0),
                ip=format_metric(row.get("internal_f_pearson")),
                ep=format_metric(row.get("external_expert_pearson")),
                gap=format_metric(row.get("f_star_gap")),
                im=format_metric(row.get("internal_f_mae_1_7")),
                em=format_metric(row.get("external_expert_mae_1_7")),
                mp=format_metric(row.get("mean_prediction_1_7")),
                mt=format_metric(row.get("mean_teacher_1_7")),
                me=format_metric(row.get("mean_expert_1_7")),
            )
        )
    lines.extend(
        [
            "",
            "All compact target metrics are on the normalized [0,1] CMP aggregate scale. "
            "The legacy `_1_7` field names are retained for compatibility with the "
            "shared alternating summary schema.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_corrected_scale_markdown_summary(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    eval_split: str,
) -> None:
    header = (
        "| family | axis | k | stage | trained | n | int_p | ext_p | f_star_gap | "
        "int_mae | ext_mae | mean_p | mean_t | mean_e | scale |"
    )
    sep = "|" + "|".join("-" * (len(seg) + 2) for seg in header.strip("|").split("|")) + "|"
    lines = [f"# Corrected-scale ladder grid summary ({eval_split} split)", "", header, sep]
    for row in rows:
        lines.append(
            "| {family} | {axis} | {k} | {stage} | {trained} | {n} | {ip} | {ep} | {gap} | "
            "{im} | {em} | {mp} | {mt} | {me} | {scale} |".format(
                family=row.get("family"),
                axis=ladder_axis_label(row) if row.get("leaf_size_tokens") is not None else str(row.get("axis_value")),
                k=row.get("iteration"),
                stage=row.get("stage_label") or row.get("stage_name"),
                trained=row.get("trained"),
                n=format_metric(row.get("n_eval"), width=4, digits=0),
                ip=format_metric(row.get("internal_f_pearson")),
                ep=format_metric(row.get("external_expert_pearson")),
                gap=format_metric(row.get("f_star_gap")),
                im=format_metric(metric_or_fallback(row, "internal_f_mae", "internal_f_mae_1_7")),
                em=format_metric(metric_or_fallback(row, "external_expert_mae", "external_expert_mae_1_7")),
                mp=format_metric(metric_or_fallback(row, "mean_prediction", "mean_prediction_1_7")),
                mt=format_metric(metric_or_fallback(row, "mean_teacher", "mean_teacher_1_7")),
                me=format_metric(metric_or_fallback(row, "mean_expert", "mean_expert_1_7")),
                scale=row.get("metrics_scale") or "",
            )
        )
    lines.append("")
    lines.append(
        "External metrics are recomputed from stored prediction records and source "
        "Benoit rows; model outputs are not rerun."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_LADDER_METRIC_FIELDS",
    "DEFAULT_ROW_FIELDS",
    "format_metric",
    "ladder_axis_label",
    "metric_or_fallback",
    "summarize_ladder_grid",
    "write_alternating_markdown_summary",
    "write_corrected_scale_markdown_summary",
    "write_qsentence_markdown_summary",
]
