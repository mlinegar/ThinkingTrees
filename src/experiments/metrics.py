from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

from src.experiments.script_parse import safe_float


def pearson(xs: Sequence[Any], ys: Sequence[Any]) -> Optional[float]:
    pairs: list[tuple[float, float]] = []
    for x_raw, y_raw in zip(xs, ys):
        x = safe_float(x_raw)
        y = safe_float(y_raw)
        if x is not None and y is not None:
            pairs.append((float(x), float(y)))
    if len(pairs) < 2:
        return None
    x_values = [pair[0] for pair in pairs]
    y_values = [pair[1] for pair in pairs]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    x_var = sum((x - x_mean) ** 2 for x in x_values)
    y_var = sum((y - y_mean) ** 2 for y in y_values)
    if x_var <= 0.0 or y_var <= 0.0:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    return float(cov / math.sqrt(x_var * y_var))


def rankdata(values: Sequence[Any]) -> list[float]:
    sortable = [(float(value), idx) for idx, value in enumerate(values)]
    sortable.sort()
    ranks = [0.0] * len(sortable)
    i = 0
    while i < len(sortable):
        j = i + 1
        while j < len(sortable) and sortable[j][0] == sortable[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for _value, idx in sortable[i:j]:
            ranks[idx] = avg
        i = j
    return ranks


def spearman(xs: Sequence[Any], ys: Sequence[Any]) -> Optional[float]:
    pairs: list[tuple[float, float]] = []
    for x_raw, y_raw in zip(xs, ys):
        x = safe_float(x_raw)
        y = safe_float(y_raw)
        if x is not None and y is not None:
            pairs.append((float(x), float(y)))
    if len(pairs) < 2:
        return None
    return pearson(
        rankdata([pair[0] for pair in pairs]),
        rankdata([pair[1] for pair in pairs]),
    )


def regression_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    pred_key: str,
    truth_key: str,
) -> dict[str, Any]:
    preds: list[float] = []
    truths: list[float] = []
    for row in rows:
        pred = safe_float(row.get(pred_key))
        truth = safe_float(row.get(truth_key))
        if pred is None or truth is None:
            continue
        preds.append(float(pred))
        truths.append(float(truth))
    if not preds:
        return {
            "n": 0,
            "pearson": None,
            "pearson_r": None,
            "spearman": None,
            "spearman_r": None,
            "mae": None,
            "mse": None,
            "rmse": None,
            "mean_prediction": None,
            "mean_truth": None,
        }
    errors = [pred - truth for pred, truth in zip(preds, truths)]
    sq_errors = [err * err for err in errors]
    mse = float(sum(sq_errors) / len(sq_errors))
    pearson_r = pearson(preds, truths)
    spearman_r = spearman(preds, truths)
    return {
        "n": len(preds),
        "pearson": pearson_r,
        "pearson_r": pearson_r,
        "spearman": spearman_r,
        "spearman_r": spearman_r,
        "mae": float(sum(abs(err) for err in errors) / len(errors)),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mean_prediction": float(sum(preds) / len(preds)),
        "mean_truth": float(sum(truths) / len(truths)),
    }


__all__ = ["pearson", "rankdata", "regression_metrics", "spearman"]
