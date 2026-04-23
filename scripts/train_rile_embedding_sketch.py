#!/usr/bin/env python3
"""
Train a mergeable embedding sketch model for Manifesto RILE.

This script:
  1) Loads Manifesto Project samples with multilingual text.
  2) Embeds each document as a small set of deterministic char windows.
  3) Trains a strictly-mergeable DeepSets sketch over window embeddings to
     predict normalized RILE in [0, 1].

The intent is to validate that multilingual embeddings + a mergeable sketch can
learn cross-country/cross-language "position" signals without translating all
documents into English.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.reproducibility import (
    configure_reproducibility,
    write_reproducibility_manifest,
)

logger = logging.getLogger(__name__)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_rile(raw_rile: float) -> float:
    from src.tasks.manifesto.constants import RILE_MIN, RILE_RANGE

    return _clamp01((float(raw_rile) - float(RILE_MIN)) / float(RILE_RANGE))


def _denormalize_rile(score01: float) -> float:
    from src.tasks.manifesto.constants import RILE_MIN, RILE_RANGE

    return float(RILE_MIN) + float(RILE_RANGE) * float(score01)


def _resolve_out_dir(arg: Optional[Path]) -> Path:
    if arg is None:
        run_id = datetime.now().strftime("rile_embedding_sketch_%Y%m%d_%H%M%S")
        return (PROJECT_ROOT / "outputs" / "rile_embedding_sketch" / run_id).resolve()
    path = Path(arg)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _set_conditional_memory_env(args: argparse.Namespace) -> None:
    mapping = {
        "conditional_memory_dir": "TT_CONDITIONAL_MEMORY_DIR",
        "conditional_memory_mode": "TT_CONDITIONAL_MEMORY_MODE",
        "conditional_memory_l1_cap": "TT_CONDITIONAL_MEMORY_L1_CAP",
        "conditional_memory_max_l2_entries": "TT_CONDITIONAL_MEMORY_MAX_L2_ENTRIES",
        "conditional_memory_l2_path": "TT_CONDITIONAL_MEMORY_L2_PATH",
        "conditional_memory_l2_shards": "TT_CONDITIONAL_MEMORY_L2_SHARDS",
        "conditional_memory_namespace_version": "TT_CONDITIONAL_MEMORY_NAMESPACE_VERSION",
    }
    for attr, env_key in mapping.items():
        value = getattr(args, attr, None)
        if value is None:
            continue
        os.environ[env_key] = str(value)


def _seed_everything(seed: int) -> Dict[str, Any]:
    return dict(configure_reproducibility(int(seed)))


def _maybe_subsample(ids: Sequence[str], *, max_items: Optional[int], seed: int) -> List[str]:
    cleaned = [str(x).strip() for x in (ids or []) if str(x).strip()]
    if max_items is None:
        return cleaned
    max_items = int(max_items)
    if max_items <= 0 or max_items >= len(cleaned):
        return cleaned
    rng = random.Random(int(seed))
    rng.shuffle(cleaned)
    return cleaned[:max_items]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _sanitize_payload(payload: Any) -> Any:
    """Convert non-primitive types (e.g., Path) into safe JSON/pickle primitives."""
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {str(k): _sanitize_payload(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_sanitize_payload(v) for v in payload]
    return payload


def _pad_windows(
    windows: List[np.ndarray],
    *,
    pad_windows: int,
    embedding_dim: int,
) -> np.ndarray:
    out = np.zeros((len(windows), int(pad_windows), int(embedding_dim)), dtype=np.float32)
    for idx, arr in enumerate(windows):
        if arr.size == 0:
            continue
        usable = min(int(pad_windows), int(arr.shape[0]))
        out[idx, :usable, :] = arr[:usable, :].astype(np.float32, copy=False)
    return out


def _subsample_text_windows(window_texts: List[str], *, max_windows: int) -> List[str]:
    """
    Deterministically downsample windows while keeping tail coverage.

    This mirrors the "tail coverage" logic used elsewhere so long docs don't
    lose conclusions when max_windows is set.
    """
    max_windows = int(max_windows)
    if max_windows <= 0 or len(window_texts) <= max_windows:
        return list(window_texts)
    stride = max(1, int(np.ceil(len(window_texts) / float(max_windows))))
    reduced = list(window_texts[::stride])
    if reduced and reduced[-1] != window_texts[-1]:
        if len(reduced) >= max_windows:
            reduced[-1] = window_texts[-1]
        else:
            reduced.append(window_texts[-1])
    return reduced[:max_windows]


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    if y_true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan")}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    return {"mae": mae, "rmse": rmse}


def _compute_masked_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
    if y_true.size == 0 or mask_arr.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "count": 0}
    valid = mask_arr[: min(mask_arr.size, y_true.size, y_pred.size)]
    if not np.any(valid):
        return {"mae": float("nan"), "rmse": float("nan"), "count": 0}
    return {
        **_compute_metrics(y_true[valid], y_pred[valid]),
        "count": int(np.sum(valid)),
    }


def _delta_vs_zero_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Any]:
    head = _compute_masked_metrics(y_true, y_pred, mask)
    zero_pred = np.zeros_like(np.asarray(y_true, dtype=np.float32).reshape(-1))
    zero = _compute_masked_metrics(y_true, zero_pred, mask)
    head_mae = head.get("mae")
    zero_mae = zero.get("mae")
    if head_mae is None or zero_mae is None or not np.isfinite(head_mae) or not np.isfinite(zero_mae):
        rel = None
    else:
        rel = float((float(zero_mae) - float(head_mae)) / max(abs(float(zero_mae)), 1e-12))
    return {
        "count": int(head.get("count", 0) or 0),
        "mae_delta_head": (None if not np.isfinite(head.get("mae", np.nan)) else float(head["mae"])),
        "mae_delta_zero": (None if not np.isfinite(zero.get("mae", np.nan)) else float(zero["mae"])),
        "relative_improvement": rel,
    }


def _device_from_arg(device: str):
    import torch

    requested = str(device or "").strip().lower()
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        # Some environments report CUDA availability but still fail on first allocation.
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception:
            logger.warning("CUDA auto-detected but not usable; falling back to CPU.")
    return torch.device("cpu")


def _embed_split(
    *,
    samples: Sequence[Any],
    split_name: str,
    doc_embedder: Any,
    include_meta: bool,
    embedding_dim_hint: Optional[int],
    log_every: int,
    windowing_mode: str = "uniform",
) -> Tuple[
    List[np.ndarray],
    List[int],
    List[np.ndarray],
    List[np.ndarray],
    List[float],
    List[Dict[str, Any]],
    Optional[int],
]:
    from src.core.doc_metadata import DocMetadata
    from src.preprocessing.chunker import chunk_for_ops

    window_arrays: List[np.ndarray] = []
    counts: List[int] = []
    meta_vectors: List[np.ndarray] = []
    query_vectors: List[np.ndarray] = []
    targets: List[float] = []
    rows: List[Dict[str, Any]] = []

    embedding_dim: Optional[int] = int(embedding_dim_hint) if embedding_dim_hint is not None else None

    for idx, sample in enumerate(samples):
        text = str(getattr(sample, "text", "") or "")
        if not text.strip():
            continue

        try:
            meta = DocMetadata.from_manifesto_sample(sample)
        except Exception:
            meta = None

        mode = str(windowing_mode or "uniform").strip().lower()
        if mode == "chunker":
            chunks = chunk_for_ops(
                text,
                max_chars=int(getattr(doc_embedder.config, "window_chars", 6000) or 6000),
                strategy="axis",
            )
            window_texts = [c.text for c in chunks]
            window_texts = _subsample_text_windows(
                window_texts,
                max_windows=int(getattr(doc_embedder.config, "max_windows", 0) or 0),
            )
            window_embeddings = doc_embedder.client.embed_texts(window_texts) if window_texts else []
        else:
            windows, _window_texts, window_embeddings = doc_embedder.embed_text(text)
        if not window_embeddings:
            continue

        if embedding_dim is None:
            embedding_dim = int(len(window_embeddings[0]))
        if any(len(vec) != embedding_dim for vec in window_embeddings):
            raise RuntimeError("Embedding dimension mismatch within a single document")

        win_arr = np.asarray(window_embeddings, dtype=np.float32)
        window_arrays.append(win_arr)
        counts.append(int(win_arr.shape[0]))

        meta_vec = np.zeros((embedding_dim,), dtype=np.float32)
        meta_text = None
        if include_meta and meta is not None:
            meta_text, embedded = doc_embedder.embed_metadata_vector(meta)
            if embedded is not None:
                if int(embedded.shape[0]) != embedding_dim:
                    raise RuntimeError("Metadata embedding dimension mismatch")
                meta_vec = embedded.astype(np.float32, copy=False)
        meta_vectors.append(meta_vec)

        text_vec = win_arr.mean(axis=0).astype(np.float32, copy=False)
        text_norm = float(np.linalg.norm(text_vec) + 1e-12)
        text_vec = (text_vec / text_norm).astype(np.float32, copy=False)
        combined_vec = doc_embedder.combine_vectors(
            text_vector=text_vec,
            meta_vector=meta_vec if include_meta else None,
        )
        if combined_vec is None:
            combined_vec = text_vec
        query_vectors.append(np.asarray(combined_vec, dtype=np.float32))

        raw_rile = float(getattr(sample, "rile", 0.0))
        targets.append(float(_normalize_rile(raw_rile)))

        rows.append(
            {
                "split": split_name,
                "manifesto_id": str(getattr(sample, "manifesto_id", "") or ""),
                "country_name": str(getattr(sample, "country_name", "") or ""),
                "country_code": int(getattr(sample, "country_code", 0) or 0),
                "party_name": str(getattr(sample, "party_name", "") or ""),
                "party_id": int(getattr(sample, "party_id", 0) or 0),
                "party_abbrev": str(getattr(sample, "party_abbrev", "") or ""),
                "year": int(getattr(sample, "year", 0) or 0),
                "date_code": int(getattr(sample, "date_code", 0) or 0),
                "party_family": getattr(sample, "party_family", None),
                "text_chars": int(len(text)),
                "true_rile": raw_rile,
                "true_score01": float(_normalize_rile(raw_rile)),
                "meta_text": meta_text or "",
                "window_count": int(win_arr.shape[0]),
            }
        )

        if log_every > 0 and (idx + 1) % int(log_every) == 0:
            logger.info("[%s] embedded %d/%d docs", split_name, idx + 1, len(samples))

    return window_arrays, counts, meta_vectors, query_vectors, targets, rows, embedding_dim


def _build_temporal_semantic_signals(
    *,
    split_rows: Dict[str, List[Dict[str, Any]]],
    split_query_vectors: Dict[str, List[np.ndarray]],
    index_dir: Path,
    top_k: int,
    lambda_year: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    from src.core.semantic_memory import (
        SemanticMemoryConfig,
        SemanticMemoryIndex,
        normalize_rile_delta,
    )

    import shutil

    index_path = Path(index_dir)
    if index_path.exists():
        shutil.rmtree(index_path, ignore_errors=True)

    cfg = SemanticMemoryConfig(
        enabled=True,
        index_dir=index_path,
        top_k=max(1, int(top_k)),
        lambda_year=max(0.0, float(lambda_year)),
        index_granularity="doc",
        update_policy="post_score",
        inject_prompts=False,
        model_features=True,
        temporal_mode=True,
    )
    index = SemanticMemoryIndex(cfg)

    split_features: Dict[str, np.ndarray] = {}
    split_delta_targets: Dict[str, np.ndarray] = {}
    split_delta_mask: Dict[str, np.ndarray] = {}
    for split_name, rows in split_rows.items():
        split_features[split_name] = np.zeros((len(rows), 6), dtype=np.float32)
        split_delta_targets[split_name] = np.zeros((len(rows),), dtype=np.float32)
        split_delta_mask[split_name] = np.zeros((len(rows),), dtype=bool)

    ordered: List[Tuple[int, int, str, str, int]] = []
    for split_name, rows in split_rows.items():
        for idx, row in enumerate(rows):
            year = int(row.get("year", 0) or 0)
            date_code = int(row.get("date_code", 0) or 0)
            manifesto_id = str(row.get("manifesto_id", "") or "")
            ordered.append((year, date_code, manifesto_id, split_name, idx))
    ordered.sort(key=lambda item: (item[0], item[1], item[2]))

    for _year, _date_code, _doc, split_name, idx in ordered:
        rows = split_rows[split_name]
        query_vectors = split_query_vectors[split_name]
        if idx >= len(rows) or idx >= len(query_vectors):
            continue
        row = rows[idx]
        query_vec = np.asarray(query_vectors[idx], dtype=np.float32).reshape(-1)
        if query_vec.size <= 0:
            continue
        norm = float(np.linalg.norm(query_vec) + 1e-12)
        query_vec = (query_vec / norm).astype(np.float32, copy=False)

        query_meta = {
            "party_id": int(row.get("party_id", 0) or 0),
            "country_code": int(row.get("country_code", 0) or 0),
            "party_family": int(row.get("party_family", 0) or 0) if row.get("party_family") is not None else None,
            "year": int(row.get("year", 0) or 0),
            "date_code": int(row.get("date_code", 0) or 0),
        }
        doc_id = str(row.get("manifesto_id", "") or "").strip()
        if not doc_id:
            continue

        neighbors = index.query(
            query_vector=query_vec,
            query_meta=query_meta,
            top_k=max(1, int(top_k)),
            exclude_doc_id=doc_id,
        )
        split_features[split_name][idx, :] = index.retrieval_features(neighbors)

        predecessor = index.get_temporal_predecessor(
            party_id=query_meta.get("party_id"),
            country_code=query_meta.get("country_code"),
            year=query_meta.get("year"),
            date_code=query_meta.get("date_code"),
            exclude_doc_id=doc_id,
        )
        if predecessor is not None:
            delta = normalize_rile_delta(
                current_rile=float(row.get("true_rile", 0.0) or 0.0),
                previous_rile=predecessor.rile,
                rile_range=200.0,
            )
            if delta is not None:
                split_delta_targets[split_name][idx] = float(delta)
                split_delta_mask[split_name][idx] = True

        index.add_document(
            doc_id=doc_id,
            vector=query_vec,
            metadata={
                "party_id": query_meta.get("party_id"),
                "country_code": query_meta.get("country_code"),
                "party_family": query_meta.get("party_family"),
                "year": query_meta.get("year"),
                "date_code": query_meta.get("date_code"),
                "rile": float(row.get("true_rile", 0.0) or 0.0),
                "provenance": {"split": split_name, "source": "train_rile_embedding_sketch"},
            },
        )

    diagnostics = {
        "index_report": index.report(),
        "eligible_delta_samples": {k: int(v.sum()) for k, v in split_delta_mask.items()},
    }
    return split_features, split_delta_targets, split_delta_mask, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a mergeable embedding sketch baseline for Manifesto RILE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: timestamped under outputs/)")

    parser.add_argument("--embedding-url", type=str, default=None, help="Embedding endpoint base URL (OpenAI-compatible)")
    parser.add_argument("--embedding-model", type=str, default=None, help="Embedding model id served by the endpoint")
    parser.add_argument("--embedding-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--embedding-batch-size", type=int, default=32)

    parser.add_argument("--window-chars", type=int, default=6000, help="Chars per embedding window")
    parser.add_argument("--overlap-chars", type=int, default=0, help="Overlap between windows (chars)")
    parser.add_argument("--max-windows", type=int, default=0, help="Max windows per document (0 = unlimited)")
    parser.add_argument(
        "--windowing-mode",
        type=str,
        choices=["uniform", "chunker"],
        default="uniform",
        help="How to choose windows before embedding (uniform char windows vs OPS chunker windows).",
    )
    parser.add_argument("--no-metadata", action="store_true", help="Disable metadata embeddings/conditioning")
    parser.add_argument("--meta-weight", type=float, default=0.25, help="(For doc vectors) relative weight of meta embedding")

    parser.add_argument("--train-end-year", type=int, default=1995)
    parser.add_argument("--val-end-year", type=int, default=2005)
    parser.add_argument("--countries", type=str, default=None, help="Optional comma-separated CMP country codes")
    parser.add_argument("--min-year", type=int, default=None)
    parser.add_argument("--max-year", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=400)
    parser.add_argument("--val-samples", type=int, default=120)
    parser.add_argument("--test-samples", type=int, default=120)

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--phi-hidden-dim", type=int, default=256)
    parser.add_argument("--readout-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-count-feature", action="store_true", help="Disable log1p(count) feature")
    parser.add_argument(
        "--delta-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable multitask delta-RILE head with tanh output in [-1,1].",
    )
    parser.add_argument(
        "--learn-loss-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Learn multitask loss weights (uncertainty weighting) instead of fixed constants.",
    )
    parser.add_argument(
        "--fixed-rile-loss-weight",
        type=float,
        default=0.5,
        help="Fallback fixed RILE weight when --no-learn-loss-weights.",
    )
    parser.add_argument(
        "--fixed-delta-loss-weight",
        type=float,
        default=0.5,
        help="Fallback fixed delta weight when --no-learn-loss-weights.",
    )
    parser.add_argument(
        "--semantic-memory-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute chronological semantic retrieval features from prior docs only.",
    )
    parser.add_argument(
        "--semantic-memory-top-k",
        type=int,
        default=5,
        help="Top-k prior neighbors used for semantic retrieval features.",
    )
    parser.add_argument(
        "--semantic-memory-lambda-year",
        type=float,
        default=0.08,
        help="Temporal decay lambda_year for semantic retrieval features.",
    )
    parser.add_argument(
        "--semantic-memory-index-dir",
        type=Path,
        default=None,
        help="Index directory for temporal semantic feature construction (default: <output-dir>/semantic_memory_index).",
    )

    parser.add_argument("--log-every", type=int, default=25, help="Progress log cadence while embedding")
    parser.add_argument("--save-embeddings", action="store_true", help="Save embedded tensors to output dir")

    # ConditionalMemory controls (optional convenience; can also be set via env vars).
    parser.add_argument("--conditional-memory-dir", type=str, default=None)
    parser.add_argument("--conditional-memory-mode", type=str, choices=["off", "read", "write", "readwrite"], default=None)
    parser.add_argument("--conditional-memory-l1-cap", type=int, default=None)
    parser.add_argument("--conditional-memory-max-l2-entries", type=int, default=None)
    parser.add_argument("--conditional-memory-l2-path", type=str, default=None)
    parser.add_argument("--conditional-memory-l2-shards", type=int, default=None)
    parser.add_argument("--conditional-memory-namespace-version", type=str, default=None)

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _set_conditional_memory_env(args)
    applied_repro = _seed_everything(int(args.seed))

    out_dir = _resolve_out_dir(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", out_dir)

    from src.config.settings import get_embedding_model, get_embedding_url, load_settings
    from src.embeddings.document_embedder import DocumentEmbedder, DocumentEmbeddingConfig
    from src.tasks.manifesto.data_loader import ManifestoDataset
    from src.training.embedding_proxy import VLLMEmbeddingClient

    settings = load_settings()
    api_base = (args.embedding_url or get_embedding_url(settings) or "").rstrip("/")
    model = (args.embedding_model or get_embedding_model(settings) or None)

    client = VLLMEmbeddingClient(
        api_base=api_base,
        model=model,
        timeout_seconds=float(args.embedding_timeout_seconds),
        batch_size=int(args.embedding_batch_size),
    )
    try:
        resolved_model = client.resolve_model()
    except Exception as e:
        logger.error("Embedding endpoint not reachable or misconfigured (%s).", e)
        logger.error("Start it with: ./scripts/start_embedding_server.sh")
        return 1

    include_meta = not bool(args.no_metadata)
    embed_cfg = DocumentEmbeddingConfig(
        window_chars=int(args.window_chars),
        overlap_chars=int(args.overlap_chars),
        max_windows=int(args.max_windows),
        l2_normalize=True,
        embed_metadata=include_meta,
        meta_weight=float(args.meta_weight),
    )
    doc_embedder = DocumentEmbedder(client, config=embed_cfg)

    countries: Optional[List[int]] = None
    if args.countries:
        parts = [p.strip() for p in str(args.countries).split(",") if p.strip()]
        parsed: List[int] = []
        for p in parts:
            try:
                parsed.append(int(p))
            except ValueError:
                raise SystemExit(f"Invalid country code: {p}")
        countries = parsed or None

    dataset = ManifestoDataset(
        countries=countries,
        min_year=int(args.min_year) if args.min_year is not None else None,
        max_year=int(args.max_year) if args.max_year is not None else None,
        require_text=True,
    )
    train_ids, val_ids, test_ids = dataset.create_temporal_split(
        train_end_year=int(args.train_end_year),
        val_end_year=int(args.val_end_year),
    )

    train_ids = _maybe_subsample(train_ids, max_items=args.train_samples, seed=int(args.seed) + 1)
    val_ids = _maybe_subsample(val_ids, max_items=args.val_samples, seed=int(args.seed) + 2)
    test_ids = _maybe_subsample(test_ids, max_items=args.test_samples, seed=int(args.seed) + 3)

    train_samples = [s for s in (dataset.get_sample(x) for x in train_ids) if s is not None]
    val_samples = [s for s in (dataset.get_sample(x) for x in val_ids) if s is not None]
    test_samples = [s for s in (dataset.get_sample(x) for x in test_ids) if s is not None]

    logger.info("Split sizes (samples): train=%d val=%d test=%d", len(train_samples), len(val_samples), len(test_samples))
    logger.info("Embedding endpoint: %s", api_base)
    logger.info("Resolved embedding model id: %s", resolved_model)
    logger.info(
        "Windowing: window_chars=%d overlap_chars=%d max_windows=%d",
        int(embed_cfg.window_chars),
        int(embed_cfg.overlap_chars),
        int(embed_cfg.max_windows),
    )
    logger.info("Metadata conditioning: %s", "on" if include_meta else "off")

    safe_args = _sanitize_payload(vars(args))
    _write_json(
        out_dir / "run_config.json",
        {
            "created_at": datetime.now().isoformat(),
            "embedding_url": api_base,
            "embedding_model_resolved": resolved_model,
            "args": safe_args,
            "document_embedding_config": asdict(embed_cfg),
        },
    )
    repro_manifest_path = write_reproducibility_manifest(
        out_dir,
        seed=int(args.seed),
        cli_args=vars(args),
        config={
            "document_embedding_config": asdict(embed_cfg),
            "windowing_mode": str(args.windowing_mode),
        },
        applied=applied_repro,
        extra={
            "embedding_url": str(api_base),
            "embedding_model_resolved": str(resolved_model),
            "split_ids": {
                "train": list(train_ids),
                "val": list(val_ids),
                "test": list(test_ids),
            },
        },
    )
    logger.info("Reproducibility manifest: %s", repro_manifest_path)

    embedding_dim: Optional[int] = None
    train_windows, train_counts, train_meta, train_query, train_targets, train_rows, embedding_dim = _embed_split(
        samples=train_samples,
        split_name="train",
        doc_embedder=doc_embedder,
        include_meta=include_meta,
        embedding_dim_hint=embedding_dim,
        log_every=int(args.log_every),
        windowing_mode=str(args.windowing_mode),
    )
    val_windows, val_counts, val_meta, val_query, val_targets, val_rows, embedding_dim = _embed_split(
        samples=val_samples,
        split_name="val",
        doc_embedder=doc_embedder,
        include_meta=include_meta,
        embedding_dim_hint=embedding_dim,
        log_every=int(args.log_every),
        windowing_mode=str(args.windowing_mode),
    )
    test_windows, test_counts, test_meta, test_query, test_targets, test_rows, embedding_dim = _embed_split(
        samples=test_samples,
        split_name="test",
        doc_embedder=doc_embedder,
        include_meta=include_meta,
        embedding_dim_hint=embedding_dim,
        log_every=int(args.log_every),
        windowing_mode=str(args.windowing_mode),
    )

    if embedding_dim is None:
        logger.error("No embeddings produced (empty dataset or missing texts).")
        return 2

    pad_windows = max([0] + train_counts + val_counts + test_counts)
    if int(args.max_windows) > 0:
        pad_windows = min(int(args.max_windows), pad_windows)
    if pad_windows <= 0:
        logger.error("No windows produced.")
        return 2

    # If we cap/pad windows, ensure counts are consistent with the tensor width.
    train_counts = [min(int(c), int(pad_windows)) for c in train_counts]
    val_counts = [min(int(c), int(pad_windows)) for c in val_counts]
    test_counts = [min(int(c), int(pad_windows)) for c in test_counts]
    for row in train_rows:
        row["window_count"] = min(int(row.get("window_count", 0) or 0), int(pad_windows))
    for row in val_rows:
        row["window_count"] = min(int(row.get("window_count", 0) or 0), int(pad_windows))
    for row in test_rows:
        row["window_count"] = min(int(row.get("window_count", 0) or 0), int(pad_windows))

    x_train = _pad_windows(train_windows, pad_windows=pad_windows, embedding_dim=embedding_dim)
    x_val = _pad_windows(val_windows, pad_windows=pad_windows, embedding_dim=embedding_dim)
    x_test = _pad_windows(test_windows, pad_windows=pad_windows, embedding_dim=embedding_dim)

    m_train = np.stack(train_meta, axis=0).astype(np.float32, copy=False) if train_meta else np.zeros((0, embedding_dim), dtype=np.float32)
    m_val = np.stack(val_meta, axis=0).astype(np.float32, copy=False) if val_meta else np.zeros((0, embedding_dim), dtype=np.float32)
    m_test = np.stack(test_meta, axis=0).astype(np.float32, copy=False) if test_meta else np.zeros((0, embedding_dim), dtype=np.float32)

    y_train = np.asarray(train_targets, dtype=np.float32)
    y_val = np.asarray(val_targets, dtype=np.float32)
    y_test = np.asarray(test_targets, dtype=np.float32)

    r_train = np.zeros((len(train_rows), 6), dtype=np.float32)
    r_val = np.zeros((len(val_rows), 6), dtype=np.float32)
    r_test = np.zeros((len(test_rows), 6), dtype=np.float32)
    d_train = np.zeros((len(train_rows),), dtype=np.float32)
    d_val = np.zeros((len(val_rows),), dtype=np.float32)
    d_test = np.zeros((len(test_rows),), dtype=np.float32)
    dmask_train = np.zeros((len(train_rows),), dtype=bool)
    dmask_val = np.zeros((len(val_rows),), dtype=bool)
    dmask_test = np.zeros((len(test_rows),), dtype=bool)
    semantic_diagnostics: Dict[str, Any] = {}
    if bool(args.semantic_memory_features) or bool(args.delta_head):
        sem_index_dir = args.semantic_memory_index_dir
        if sem_index_dir is None:
            sem_index_dir = out_dir / "semantic_memory_index"
        split_features, split_delta_targets, split_delta_masks, semantic_diagnostics = _build_temporal_semantic_signals(
            split_rows={
                "train": train_rows,
                "val": val_rows,
                "test": test_rows,
            },
            split_query_vectors={
                "train": train_query,
                "val": val_query,
                "test": test_query,
            },
            index_dir=Path(sem_index_dir),
            top_k=int(args.semantic_memory_top_k),
            lambda_year=float(args.semantic_memory_lambda_year),
        )
        d_train = split_delta_targets["train"]
        d_val = split_delta_targets["val"]
        d_test = split_delta_targets["test"]
        dmask_train = split_delta_masks["train"]
        dmask_val = split_delta_masks["val"]
        dmask_test = split_delta_masks["test"]
        if bool(args.semantic_memory_features):
            r_train = split_features["train"]
            r_val = split_features["val"]
            r_test = split_features["test"]

        logger.info(
            "Temporal semantic signals ready: retrieval_features=%s delta_eligible(train=%d,val=%d,test=%d)",
            "on" if bool(args.semantic_memory_features) else "off",
            int(np.sum(dmask_train)),
            int(np.sum(dmask_val)),
            int(np.sum(dmask_test)),
        )

    if args.save_embeddings:
        np.savez_compressed(
            out_dir / "embedded_splits.npz",
            x_train=x_train,
            c_train=np.asarray(train_counts, dtype=np.int64),
            m_train=m_train,
            y_train=y_train,
            x_val=x_val,
            c_val=np.asarray(val_counts, dtype=np.int64),
            m_val=m_val,
            y_val=y_val,
            x_test=x_test,
            c_test=np.asarray(test_counts, dtype=np.int64),
            m_test=m_test,
            y_test=y_test,
            r_train=r_train,
            r_val=r_val,
            r_test=r_test,
            d_train=d_train,
            d_val=d_val,
            d_test=d_test,
            dmask_train=dmask_train.astype(np.int8),
            dmask_val=dmask_val.astype(np.int8),
            dmask_test=dmask_test.astype(np.int8),
        )

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.training.embedding_sketch import (
        EmbeddingSketchConfig,
        MergeableEmbeddingSketch,
        merge_prediction_consistency,
    )

    device = _device_from_arg(args.device)
    logger.info("Training device: %s", device)

    sketch_cfg = EmbeddingSketchConfig(
        embedding_dim=int(embedding_dim),
        state_dim=int(args.state_dim),
        phi_hidden_dim=int(args.phi_hidden_dim),
        readout_hidden_dim=int(args.readout_hidden_dim),
        dropout=float(args.dropout),
        include_meta=bool(include_meta),
        use_count_feature=not bool(args.no_count_feature),
        include_retrieval_features=bool(args.semantic_memory_features),
        retrieval_feature_dim=6,
        include_delta_head=bool(args.delta_head),
    )
    model = MergeableEmbeddingSketch(sketch_cfg).to(device)
    training_started_at = time.time()

    learn_loss_weights = bool(args.learn_loss_weights) and bool(args.delta_head)
    fixed_rile_weight = max(0.0, float(args.fixed_rile_loss_weight))
    fixed_delta_weight = max(0.0, float(args.fixed_delta_loss_weight))
    if (fixed_rile_weight + fixed_delta_weight) <= 0.0:
        fixed_rile_weight = 1.0
        fixed_delta_weight = 1.0

    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(np.asarray(train_counts, dtype=np.int64)),
        torch.from_numpy(m_train),
        torch.from_numpy(r_train),
        torch.from_numpy(d_train),
        torch.from_numpy(dmask_train.astype(np.float32)),
        torch.from_numpy(y_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val),
        torch.from_numpy(np.asarray(val_counts, dtype=np.int64)),
        torch.from_numpy(m_val),
        torch.from_numpy(r_val),
        torch.from_numpy(d_val),
        torch.from_numpy(dmask_val.astype(np.float32)),
        torch.from_numpy(y_val),
    )
    test_ds = TensorDataset(
        torch.from_numpy(x_test),
        torch.from_numpy(np.asarray(test_counts, dtype=np.int64)),
        torch.from_numpy(m_test),
        torch.from_numpy(r_test),
        torch.from_numpy(d_test),
        torch.from_numpy(dmask_test.astype(np.float32)),
        torch.from_numpy(y_test),
    )

    train_loader_seed = torch.Generator()
    train_loader_seed.manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
        generator=train_loader_seed,
    )
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, drop_last=False)

    if learn_loss_weights:
        log_var_rile = torch.nn.Parameter(torch.zeros((), dtype=torch.float32, device=device))
        log_var_delta = torch.nn.Parameter(torch.zeros((), dtype=torch.float32, device=device))
        opt = torch.optim.AdamW(
            [
                {
                    "params": model.parameters(),
                    "lr": float(args.learning_rate),
                    "weight_decay": float(args.weight_decay),
                },
                {
                    "params": [log_var_rile, log_var_delta],
                    "lr": float(args.learning_rate),
                    "weight_decay": 0.0,
                },
            ]
        )
    else:
        log_var_rile = None
        log_var_delta = None
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
    loss_fn = torch.nn.MSELoss()

    baseline_mean = float(y_train.mean()) if y_train.size else 0.5
    baseline_val = _compute_metrics(y_val, np.full_like(y_val, baseline_mean))
    logger.info("Baseline (mean predictor) val_mae=%.4f val_rmse=%.4f", baseline_val["mae"], baseline_val["rmse"])

    best_val_mae = float("inf")
    best_path = out_dir / "checkpoint_best.pt"
    history: List[Dict[str, Any]] = []

    def _predict(
        loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        preds: List[np.ndarray] = []
        trues: List[np.ndarray] = []
        delta_preds: List[np.ndarray] = []
        delta_trues: List[np.ndarray] = []
        delta_masks: List[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for xb, cb, mb, rb, db, dmb, yb in loader:
                xb = xb.to(device=device, dtype=torch.float32)
                cb = cb.to(device=device)
                mb = mb.to(device=device, dtype=torch.float32)
                rb = rb.to(device=device, dtype=torch.float32)
                db = db.to(device=device, dtype=torch.float32)
                dmb = dmb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.float32)
                out = model(
                    xb,
                    counts=cb,
                    meta_embeddings=mb if include_meta else None,
                    retrieval_features=rb if bool(args.semantic_memory_features) else None,
                    return_dict=bool(args.delta_head),
                )
                if isinstance(out, dict):
                    rile_pred = out.get("rile")
                    delta_pred = out.get("delta")
                else:
                    rile_pred = out
                    delta_pred = None
                if rile_pred is None:
                    continue
                preds.append(rile_pred.detach().cpu().numpy().astype(np.float32, copy=False))
                trues.append(yb.detach().cpu().numpy().astype(np.float32, copy=False))
                if delta_pred is None:
                    delta_pred_np = np.zeros_like(db.detach().cpu().numpy().astype(np.float32, copy=False))
                else:
                    delta_pred_np = delta_pred.detach().cpu().numpy().astype(np.float32, copy=False)
                delta_preds.append(delta_pred_np)
                delta_trues.append(db.detach().cpu().numpy().astype(np.float32, copy=False))
                delta_masks.append((dmb.detach().cpu().numpy() > 0.5).astype(np.bool_))
        if not trues:
            empty = np.zeros((0,), dtype=np.float32)
            empty_mask = np.zeros((0,), dtype=np.bool_)
            return empty, empty, empty, empty, empty_mask
        return (
            np.concatenate(trues, axis=0),
            np.concatenate(preds, axis=0),
            np.concatenate(delta_trues, axis=0),
            np.concatenate(delta_preds, axis=0),
            np.concatenate(delta_masks, axis=0),
        )

    def _merge_consistency(loader: DataLoader) -> Dict[str, Any]:
        model.eval()
        maes: List[float] = []
        max_abs: List[float] = []
        n_batches = 0
        with torch.no_grad():
            for xb, cb, mb, rb, _db, _dmb, _yb in loader:
                xb = xb.to(device=device, dtype=torch.float32)
                cb = cb.to(device=device, dtype=torch.int64)
                mb = mb.to(device=device, dtype=torch.float32)
                rb = rb.to(device=device, dtype=torch.float32)
                stats = merge_prediction_consistency(
                    model,
                    xb,
                    counts=cb,
                    meta_embeddings=mb if include_meta else None,
                    retrieval_features=rb if bool(args.semantic_memory_features) else None,
                )
                maes.append(float(stats["prediction_mae"]))
                max_abs.append(float(stats["prediction_max_abs"]))
                n_batches += 1
        if not maes:
            return {
                "prediction_mae": float("nan"),
                "prediction_max_abs": float("nan"),
                "n_batches": 0,
            }
        return {
            "prediction_mae": float(np.mean(np.asarray(maes, dtype=np.float64))),
            "prediction_max_abs": float(np.max(np.asarray(max_abs, dtype=np.float64))),
            "n_batches": int(n_batches),
        }

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: List[float] = []
        rile_losses: List[float] = []
        delta_losses: List[float] = []
        for xb, cb, mb, rb, db, dmb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            cb = cb.to(device=device)
            mb = mb.to(device=device, dtype=torch.float32)
            rb = rb.to(device=device, dtype=torch.float32)
            db = db.to(device=device, dtype=torch.float32)
            dmb = dmb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)

            out = model(
                xb,
                counts=cb,
                meta_embeddings=mb if include_meta else None,
                retrieval_features=rb if bool(args.semantic_memory_features) else None,
                return_dict=bool(args.delta_head),
            )
            if isinstance(out, dict):
                pred_rile = out.get("rile")
                pred_delta = out.get("delta")
            else:
                pred_rile = out
                pred_delta = None
            if pred_rile is None:
                continue

            loss_rile = loss_fn(pred_rile, yb)
            loss_delta = torch.zeros((), dtype=torch.float32, device=device)
            has_delta_targets = False
            if bool(args.delta_head) and pred_delta is not None:
                delta_mask = dmb > 0.5
                if bool(torch.any(delta_mask)):
                    loss_delta = loss_fn(pred_delta[delta_mask], db[delta_mask])
                    has_delta_targets = True

            if learn_loss_weights and log_var_rile is not None:
                loss = torch.exp(-log_var_rile) * loss_rile + log_var_rile
                if has_delta_targets and log_var_delta is not None:
                    loss = loss + torch.exp(-log_var_delta) * loss_delta + log_var_delta
            else:
                if has_delta_targets:
                    denom = max(1e-12, fixed_rile_weight + fixed_delta_weight)
                    wr = fixed_rile_weight / denom
                    wd = fixed_delta_weight / denom
                    loss = (wr * loss_rile) + (wd * loss_delta)
                else:
                    loss = loss_rile

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.max_grad_norm))
            opt.step()

            losses.append(float(loss.detach().cpu().item()))
            rile_losses.append(float(loss_rile.detach().cpu().item()))
            delta_losses.append(float(loss_delta.detach().cpu().item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        train_rile_mse = float(np.mean(rile_losses)) if rile_losses else float("nan")
        train_delta_mse = float(np.mean(delta_losses)) if delta_losses else float("nan")
        if learn_loss_weights and log_var_rile is not None:
            rile_weight_display = float(torch.exp(-log_var_rile.detach()).cpu().item())
            delta_weight_display = (
                float(torch.exp(-log_var_delta.detach()).cpu().item())
                if log_var_delta is not None
                else 0.0
            )
        else:
            denom = max(1e-12, fixed_rile_weight + fixed_delta_weight)
            rile_weight_display = float(fixed_rile_weight / denom)
            delta_weight_display = float(fixed_delta_weight / denom)

        val_true, val_pred, val_delta_true, val_delta_pred, val_delta_mask = _predict(val_loader)
        val_metrics = _compute_metrics(val_true, val_pred)
        val_delta_metrics = _compute_masked_metrics(val_delta_true, val_delta_pred, val_delta_mask)

        row = {
            "epoch": int(epoch),
            "train_mse": train_loss,
            "train_rile_mse": train_rile_mse,
            "train_delta_mse": train_delta_mse,
            "val_mae": float(val_metrics["mae"]),
            "val_rmse": float(val_metrics["rmse"]),
            "val_delta_mae": float(val_delta_metrics["mae"]),
            "val_delta_rmse": float(val_delta_metrics["rmse"]),
            "val_delta_count": int(val_delta_metrics["count"]),
            "loss_weight_rile": rile_weight_display,
            "loss_weight_delta": delta_weight_display,
        }
        history.append(row)

        logger.info(
            "epoch=%d train_mse=%.6f val_mae=%.4f val_rmse=%.4f val_delta_mae=%s w_rile=%.4f w_delta=%.4f",
            int(epoch),
            float(train_loss),
            float(val_metrics["mae"]),
            float(val_metrics["rmse"]),
            (
                f"{float(val_delta_metrics['mae']):.4f}"
                if np.isfinite(val_delta_metrics["mae"])
                else "nan"
            ),
            float(rile_weight_display),
            float(delta_weight_display),
        )

        if float(val_metrics["mae"]) < best_val_mae:
            best_val_mae = float(val_metrics["mae"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "sketch_config": asdict(sketch_cfg),
                    "embedding_url": api_base,
                    "embedding_model_resolved": resolved_model,
                    "pad_windows": int(pad_windows),
                    "embedding_dim": int(embedding_dim),
                    "baseline_mean": float(baseline_mean),
                    "history": list(history),
                    "args": safe_args,
                    "semantic_diagnostics": semantic_diagnostics,
                    "multitask": {
                        "delta_head": bool(args.delta_head),
                        "semantic_memory_features": bool(args.semantic_memory_features),
                        "learn_loss_weights": bool(learn_loss_weights),
                        "fixed_rile_loss_weight": float(fixed_rile_weight),
                        "fixed_delta_loss_weight": float(fixed_delta_weight),
                    },
                    "loss_weighting": {
                        "rile_log_var": float(log_var_rile.detach().cpu().item()) if log_var_rile is not None else None,
                        "delta_log_var": float(log_var_delta.detach().cpu().item()) if log_var_delta is not None else None,
                        "rile_weight": float(rile_weight_display),
                        "delta_weight": float(delta_weight_display),
                    },
                    "local_law": model.local_law_capabilities(),
                },
                best_path,
            )

    # Load best checkpoint and evaluate.
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt.get("model_state_dict", {}))

    val_true, val_pred, val_delta_true, val_delta_pred, val_delta_mask = _predict(val_loader)
    test_true, test_pred, test_delta_true, test_delta_pred, test_delta_mask = _predict(test_loader)
    val_metrics = _compute_metrics(val_true, val_pred)
    test_metrics = _compute_metrics(test_true, test_pred)
    val_delta_metrics = _compute_masked_metrics(val_delta_true, val_delta_pred, val_delta_mask)
    test_delta_metrics = _compute_masked_metrics(test_delta_true, test_delta_pred, test_delta_mask)
    val_merge_consistency = _merge_consistency(val_loader)
    test_merge_consistency = _merge_consistency(test_loader)
    delta_vs_zero = {
        "val": _delta_vs_zero_summary(val_delta_true, val_delta_pred, val_delta_mask),
        "test": _delta_vs_zero_summary(test_delta_true, test_delta_pred, test_delta_mask),
    }
    local_law = {
        **model.local_law_capabilities(),
        "merge_prediction_consistency": {
            "val": val_merge_consistency,
            "test": test_merge_consistency,
        },
        "notes": [
            "Exact latent-state mergeability is enforced by construction and measured above.",
            "This script does not attach span-level oracle labels or a theorem-domain decode/resummary path, so C1/C2/C3-style objective terms remain inactive.",
        ],
    }

    summary = {
        "created_at": datetime.now().isoformat(),
        "training_time_seconds": float(time.time() - training_started_at),
        "embedding_url": api_base,
        "embedding_model_resolved": resolved_model,
        "embedding_dim": int(embedding_dim),
        "pad_windows": int(pad_windows),
        "split_sizes": {"train": int(len(train_ds)), "val": int(len(val_ds)), "test": int(len(test_ds))},
        "baseline": {"mean": float(baseline_mean), "val": baseline_val},
        "best_val_mae": float(best_val_mae),
        "semantic_diagnostics": semantic_diagnostics,
        "multitask": {
            "delta_head": bool(args.delta_head),
            "semantic_memory_features": bool(args.semantic_memory_features),
            "learn_loss_weights": bool(learn_loss_weights),
            "fixed_rile_loss_weight": float(fixed_rile_weight),
            "fixed_delta_loss_weight": float(fixed_delta_weight),
            "rile_log_var": float(log_var_rile.detach().cpu().item()) if log_var_rile is not None else None,
            "delta_log_var": float(log_var_delta.detach().cpu().item()) if log_var_delta is not None else None,
            "rile_weight": float(
                torch.exp(-log_var_rile.detach()).cpu().item()
                if log_var_rile is not None
                else (fixed_rile_weight / max(1e-12, fixed_rile_weight + fixed_delta_weight))
            ),
            "delta_weight": float(
                torch.exp(-log_var_delta.detach()).cpu().item()
                if log_var_delta is not None
                else (fixed_delta_weight / max(1e-12, fixed_rile_weight + fixed_delta_weight))
            ),
        },
        "final": {
            "val": {**val_metrics, "delta": val_delta_metrics},
            "test": {**test_metrics, "delta": test_delta_metrics},
        },
        "delta_vs_zero": delta_vs_zero,
        "local_law": local_law,
        "history": history,
    }
    _write_json(out_dir / "metrics.json", summary)

    # Write predictions (denormalized RILE) for quick inspection.
    all_rows = train_rows + val_rows + test_rows
    split_to_preds = {
        "train": None,
        "val": val_pred,
        "test": test_pred,
    }
    split_to_delta_preds = {
        "train": None,
        "val": val_delta_pred,
        "test": test_delta_pred,
    }
    split_to_delta_targets = {
        "train": None,
        "val": val_delta_true,
        "test": test_delta_true,
    }
    split_to_delta_masks = {
        "train": None,
        "val": val_delta_mask,
        "test": test_delta_mask,
    }
    # Also produce train preds (use loader).
    train_eval_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=False, drop_last=False)
    _train_true, train_pred, train_delta_true, train_delta_pred, train_delta_mask = _predict(train_eval_loader)
    split_to_preds["train"] = train_pred
    split_to_delta_preds["train"] = train_delta_pred
    split_to_delta_targets["train"] = train_delta_true
    split_to_delta_masks["train"] = train_delta_mask

    # Stitch predictions back to rows in-order within each split.
    pred_by_split: Dict[str, Iterable[float]] = {
        "train": iter([float(x) for x in split_to_preds["train"].reshape(-1)]),
        "val": iter([float(x) for x in split_to_preds["val"].reshape(-1)]),
        "test": iter([float(x) for x in split_to_preds["test"].reshape(-1)]),
    }
    delta_pred_by_split: Dict[str, Iterable[float]] = {
        "train": iter([float(x) for x in split_to_delta_preds["train"].reshape(-1)]),
        "val": iter([float(x) for x in split_to_delta_preds["val"].reshape(-1)]),
        "test": iter([float(x) for x in split_to_delta_preds["test"].reshape(-1)]),
    }
    delta_true_by_split: Dict[str, Iterable[float]] = {
        "train": iter([float(x) for x in split_to_delta_targets["train"].reshape(-1)]),
        "val": iter([float(x) for x in split_to_delta_targets["val"].reshape(-1)]),
        "test": iter([float(x) for x in split_to_delta_targets["test"].reshape(-1)]),
    }
    delta_mask_by_split: Dict[str, Iterable[bool]] = {
        "train": iter([bool(x) for x in split_to_delta_masks["train"].reshape(-1)]),
        "val": iter([bool(x) for x in split_to_delta_masks["val"].reshape(-1)]),
        "test": iter([bool(x) for x in split_to_delta_masks["test"].reshape(-1)]),
    }

    out_csv = out_dir / "predictions.csv"
    fieldnames = [
        "split",
        "manifesto_id",
        "country_name",
        "party_name",
        "party_abbrev",
        "year",
        "date_code",
        "party_family",
        "text_chars",
        "window_count",
        "true_rile",
        "pred_rile",
        "true_score01",
        "pred_score01",
        "true_delta_rile",
        "pred_delta_rile",
        "has_delta_target",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            split = str(row.get("split", "") or "")
            pred01 = float(next(pred_by_split[split]))
            pred_delta = float(next(delta_pred_by_split[split]))
            true_delta = float(next(delta_true_by_split[split]))
            has_delta = bool(next(delta_mask_by_split[split]))
            row_out = {k: row.get(k, "") for k in fieldnames}
            row_out["pred_score01"] = pred01
            row_out["pred_rile"] = float(_denormalize_rile(pred01))
            row_out["true_delta_rile"] = true_delta if has_delta else ""
            row_out["pred_delta_rile"] = pred_delta
            row_out["has_delta_target"] = int(has_delta)
            writer.writerow(row_out)

    logger.info("Saved: %s", out_csv)
    logger.info("Saved: %s", out_dir / "metrics.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
