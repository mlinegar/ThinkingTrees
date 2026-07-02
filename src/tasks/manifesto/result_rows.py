from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.experiments.script_io import read_json, read_jsonl
from src.experiments.script_parse import safe_float
from src.tasks.manifesto import ManifestoDataset
from src.tasks.manifesto.dimensions import PolicyDimension
from src.tasks.manifesto.expert_scale import (
    EXPERT_SCALE_NORMALIZED_1_7,
    resolve_benoit_expert_target,
)
from src.tasks.manifesto.phase3_splits import build_phase3_records


DIMENSION_BY_NAME = {dim.value: dim for dim in PolicyDimension}


def normalize_score_1_7(value: float) -> float:
    return max(0.0, min(1.0, (float(value) - 1.0) / 6.0))


def denormalize_score_1_7(value: float) -> float:
    return 1.0 + 6.0 * max(0.0, min(1.0, float(value)))


def row_manifesto_id(row: Mapping[str, Any]) -> str:
    return str(row.get("manifesto_id") or row.get("doc_id") or row.get("id") or "").strip()


def row_summary(row: Mapping[str, Any]) -> str:
    return str(row.get("summary") or row.get("root_summary") or "").strip()


def row_teacher_score(row: Mapping[str, Any], *, dimension: str) -> Optional[float]:
    direct = safe_float(row.get("llm_score_1_7"))
    if direct is not None:
        return direct
    direct = safe_float(row.get("teacher_score_1_7"))
    if direct is not None:
        return direct
    predictions = row.get("predictions")
    if isinstance(predictions, Mapping):
        return safe_float(predictions.get(dimension))
    return safe_float(row.get("pred"))


def row_expert_score(
    row: Mapping[str, Any],
    *,
    dimension: str,
    expert_scale: str = EXPERT_SCALE_NORMALIZED_1_7,
) -> Optional[float]:
    return resolve_benoit_expert_target(row, dimension=dimension, scale=expert_scale)


def row_target_score(
    row: Mapping[str, Any],
    *,
    dimension: str,
    target_source: str,
    expert_scale: str = EXPERT_SCALE_NORMALIZED_1_7,
) -> Optional[float]:
    if target_source == "teacher":
        return row_teacher_score(row, dimension=dimension)
    if target_source == "expert":
        return row_expert_score(row, dimension=dimension, expert_scale=expert_scale)
    raise ValueError(f"Unsupported target_source={target_source!r}")


def load_run_metadata(report_path: Optional[Path]) -> dict[str, Any]:
    if report_path is None or not report_path.exists():
        return {}
    payload = read_json(report_path)
    run = payload.get("run") if isinstance(payload, Mapping) else None
    return dict(run) if isinstance(run, Mapping) else {}


def load_rows_by_id(path: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        doc_id = row_manifesto_id(row)
        if doc_id:
            out[doc_id] = dict(row)
    return out


def load_rows_by_dimension(
    source_root: str | Path,
    dimensions: Sequence[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    root = Path(source_root)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for dim in dimensions:
        path = root / str(dim) / "per_manifesto.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing per-dimension results: {path}")
        out[str(dim)] = load_rows_by_id(path)
    return out


def phase3_split_examples(
    *,
    dimension: PolicyDimension,
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
    split_strategy: str,
    train_pool: str,
    mp_data_dir: Optional[Path],
) -> dict[str, dict[str, str]]:
    train, val, test = build_phase3_records(
        dimension,
        train_pool,
        Path(mp_data_dir) if mp_data_dir is not None else Path("data/raw/manifesto_corpus_benoit"),
        int(train_n),
        int(val_n),
        int(test_n),
        int(seed),
        split_strategy=split_strategy,
    )
    out: dict[str, dict[str, str]] = {"train": {}, "val": {}, "test": {}}
    for split, records in (("train", train), ("val", val), ("test", test)):
        for record in records:
            mid = str(record.get("manifesto_id", "") or "").strip()
            text = str(record.get("text", "") or "")
            if mid and text:
                out[split][mid] = text
    return out


def order_split_rows(
    rows_by_id: Mapping[str, Mapping[str, Any]],
    *,
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
) -> dict[str, dict[str, str]]:
    ids = sorted(rows_by_id)
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    selected = ids[: int(train_n) + int(val_n) + int(test_n)]
    selected_texts = {
        mid: str(rows_by_id[mid].get("text") or rows_by_id[mid].get("document_text") or "")
        for mid in selected
    }
    return {
        "train": {mid: selected_texts[mid] for mid in selected[: int(train_n)]},
        "val": {mid: selected_texts[mid] for mid in selected[int(train_n) : int(train_n) + int(val_n)]},
        "test": {mid: selected_texts[mid] for mid in selected[int(train_n) + int(val_n) :]},
    }


def get_text_for_row(
    *,
    row: Mapping[str, Any],
    split_texts: Mapping[str, str],
    dataset: Optional[ManifestoDataset],
) -> str:
    text = str(row.get("text") or row.get("document_text") or "").strip()
    if text:
        return text
    mid = row_manifesto_id(row)
    text = str(split_texts.get(mid) or "").strip()
    if text:
        return text
    if dataset is not None and mid:
        sample = dataset.get_sample(mid)
        if sample is not None and getattr(sample, "text", None):
            return str(sample.text)
    return ""


__all__ = [
    "DIMENSION_BY_NAME",
    "denormalize_score_1_7",
    "get_text_for_row",
    "load_rows_by_dimension",
    "load_rows_by_id",
    "load_run_metadata",
    "normalize_score_1_7",
    "order_split_rows",
    "phase3_split_examples",
    "row_expert_score",
    "row_manifesto_id",
    "row_summary",
    "row_target_score",
    "row_teacher_score",
]
