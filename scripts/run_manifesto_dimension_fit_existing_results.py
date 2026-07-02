#!/usr/bin/env python3
"""Replay existing Gemma-4 manifesto dimension results through TreePO fit.

This is the apples-to-apples smoke path for the modern contract runner:

1. Reuse existing phase-0/phase-2/phase-3 Gemma result files.
2. Reuse the existing phase-3 split builder for train/dev/test IDs.
3. Reconstruct the same fixed chunking topology from the source run's
   ``chunk_chars`` setting.
4. Convert the rows into partial ``LabeledTree`` artifacts.
5. Fit/export ``f`` and optional ``g`` targets through ``fit_treepo_contract``.

The existing result rows usually only contain root summaries and root scores.
That is still useful for ``f`` regression/proxy tests and is recorded as a
partial artifact; node-level ``g`` supervision remains limited to whatever
teacher summaries are present in the source rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ctreepo.distillation import (
    DistillationContractConfig,
    DistillationTrainConfig,
    FEmbeddingConfig,
    FLMConfig,
    GLMConfig,
    ScoreTargetConfig,
    SummaryTargetConfig,
    TRAIN_TARGET_F,
    TRAIN_TARGET_G,
    STUDENT_MODEL_EMBEDDING_RIDGE_PROXY,
    STUDENT_MODEL_LM_SCALAR_REGRESSION,
    STUDENT_MODEL_LM_SFT,
    build_f_lm_regression_records,
    build_g_sft_records,
    write_labeled_trees_jsonl,
)
from src.ctreepo.treepo_bridge.manifesto_finetune import (
    add_manifesto_finetune_args,
    export_manifesto_finetune_bundle_from_args,
    finetune_export_config,
)
from src.tasks.manifesto.script_utils import (
    now_iso as _now_iso,
    now_stamp as _now_stamp,
    read_jsonl as _read_jsonl,
    safe_float as _safe_float,
    write_json as _write_json,
)
from src.preprocessing.chunker import chunk_for_ops
from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r
from src.tasks.manifesto.dimensions import PolicyDimension, get_preservation_rubric
from src.tasks.manifesto.expert_scale import (
    EXPERT_SCALE_CHOICES,
    EXPERT_SCALE_NORMALIZED_1_7,
    expert_scale_metadata,
    raw_benoit_expert_from_row,
    resolve_benoit_expert_target,
)
from src.tasks.manifesto import ManifestoDataset
from src.training.config_sections import (
    OptimizerConfig,
    RunConfig,
    RuntimeConfig,
    TestConfig,
    TrainConfig,
    ValidationConfig,
)
from src.tree.contract_runner import RESOURCE_EMBEDDING, TreePOResourceSpec, fit_treepo_contract
from src.tree.labeled import LabeledNode, LabeledTree
from src.tree.treepo_stack import TreePOContractSpec, TreePOModelSpec


LOGGER = logging.getLogger(__name__)
_DIM_FROM_NAME = {dim.value: dim for dim in PolicyDimension}



def _normalize_score_1_7(value: float) -> float:
    return max(0.0, min(1.0, (float(value) - 1.0) / 6.0))


def _denormalize_score_1_7(value: float) -> float:
    return 1.0 + 6.0 * max(0.0, min(1.0, float(value)))


def _row_manifesto_id(row: Mapping[str, Any]) -> str:
    return str(row.get("manifesto_id") or row.get("doc_id") or row.get("id") or "").strip()


def _row_summary(row: Mapping[str, Any]) -> str:
    return str(row.get("summary") or row.get("root_summary") or "").strip()


def _row_teacher_score(row: Mapping[str, Any], *, dimension: str) -> Optional[float]:
    direct = _safe_float(row.get("llm_score_1_7"))
    if direct is not None:
        return direct
    direct = _safe_float(row.get("teacher_score_1_7"))
    if direct is not None:
        return direct
    predictions = row.get("predictions")
    if isinstance(predictions, Mapping):
        return _safe_float(predictions.get(dimension))
    return _safe_float(row.get("pred"))


def _row_expert_score(
    row: Mapping[str, Any],
    *,
    dimension: str,
    expert_scale: str = EXPERT_SCALE_NORMALIZED_1_7,
) -> Optional[float]:
    return resolve_benoit_expert_target(row, dimension=dimension, scale=expert_scale)


def _row_target_score(
    row: Mapping[str, Any],
    *,
    dimension: str,
    target_source: str,
    expert_scale: str = EXPERT_SCALE_NORMALIZED_1_7,
) -> Optional[float]:
    if target_source == "teacher":
        return _row_teacher_score(row, dimension=dimension)
    if target_source == "expert":
        return _row_expert_score(row, dimension=dimension, expert_scale=expert_scale)
    raise ValueError(f"Unsupported target_source={target_source!r}")


class HashingEmbeddingClient:
    """Deterministic offline embedding client for fast smoke tests."""

    def __init__(self, *, dim: int = 256, model: str = "hashing_embedding"):
        self.dim = int(max(8, dim))
        self.model = str(model)

    def resolve_model(self) -> str:
        return f"{self.model}:{self.dim}"

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        outputs: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in str(text or "").lower().split():
                digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = -1.0 if (digest[4] & 1) else 1.0
                vec[bucket] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            outputs.append([float(v / norm) for v in vec])
        return outputs


class LocalHFEmbeddingClient:
    """Small local HuggingFace embedding client with mean pooling."""

    def __init__(
        self,
        *,
        model: str,
        batch_size: int = 8,
        max_length: int = 1024,
        device: str = "auto",
        normalize: bool = True,
        allow_truncation: bool = False,
    ):
        self.model = str(model)
        self.batch_size = int(max(1, batch_size))
        self.max_length = int(max(8, max_length))
        self.device = str(device or "auto")
        self.normalize = bool(normalize)
        self.allow_truncation = bool(allow_truncation)
        self._tokenizer = None
        self._model = None
        self._resolved_device = None

    def resolve_model(self) -> str:
        return self.model

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model, self._resolved_device
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device
        # Use fp32 by default. Some embedding models (notably EmbeddingGemma-300m
        # via plain AutoModel) produce NaN outputs in fp16 due to numerical
        # instability in the post-encoder pooling. fp32 is the safe baseline;
        # callers that want fp16 can override via the FORCE_FP16 env var.
        import os as _os
        if _os.environ.get("CTREEPO_EMBEDDING_FP16", "").lower() in {"1", "true", "yes"}:
            dtype = torch.float16 if device.startswith("cuda") else torch.float32
        else:
            dtype = torch.float32
        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            self.model,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        model.to(device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._resolved_device = device
        return tokenizer, model, device

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        import torch

        tokenizer, model, device = self._load()
        outputs: List[List[float]] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = [str(text or "") for text in texts[start : start + self.batch_size]]
                if not self.allow_truncation:
                    lengths = [
                        len(tokenizer.encode(text, add_special_tokens=False))
                        for text in batch
                    ]
                    too_long = [
                        (idx, length)
                        for idx, length in enumerate(lengths)
                        if int(length) > int(self.max_length)
                    ]
                    if too_long:
                        idx, length = too_long[0]
                        raise RuntimeError(
                            "LocalHFEmbeddingClient no-truncation guard: "
                            f"batch item {start + idx} has {length} tokens but "
                            f"max_length={self.max_length}. Split the text before embedding."
                        )
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=bool(self.allow_truncation),
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                if self.normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                outputs.extend(pooled.detach().cpu().float().tolist())
        return outputs


def _make_embedding_client(args: argparse.Namespace):
    if args.embedding_backend == "hashing":
        return HashingEmbeddingClient(dim=args.hashing_embedding_dim)
    if args.embedding_backend == "local-hf":
        return LocalHFEmbeddingClient(
            model=str(args.embedding_model),
            batch_size=int(args.embedding_batch_size),
            max_length=int(args.embedding_max_length),
            device=str(args.embedding_device),
        )
    if args.embedding_backend == "vllm":
        from src.training.embedding_proxy import VLLMEmbeddingClient

        if not args.embedding_url:
            raise ValueError("--embedding-backend vllm requires --embedding-url")
        return VLLMEmbeddingClient(
            api_base=str(args.embedding_url),
            model=args.embedding_model,
            api_key=str(args.embedding_api_key),
            timeout_seconds=float(args.embedding_timeout_seconds),
            batch_size=int(args.embedding_batch_size),
        )
    raise ValueError(f"Unsupported embedding backend: {args.embedding_backend!r}")


def _preload_transformers_for_local_embedding(args: argparse.Namespace) -> None:
    """Import transformers before phase-3 DSPy imports alter metadata scanning.

    In this repo environment, importing the phase-3 DSPy script first can expose
    a broken package metadata entry to ``transformers`` import-time dependency
    scanning.  Preloading transformers keeps local-HF embedding runs stable
    without changing the global training stack.
    """

    if getattr(args, "embedding_backend", None) != "local-hf":
        return
    try:
        import transformers  # noqa: F401
        return
    except TypeError as exc:
        if "packages_distributions" not in str(exc) and "'NoneType' object is not subscriptable" not in str(exc):
            raise

    import importlib.metadata as importlib_metadata
    from collections import defaultdict

    def _safe_packages_distributions() -> Dict[str, List[str]]:
        pkg_to_dist: Dict[str, List[str]] = defaultdict(list)
        for dist in importlib_metadata.distributions():
            try:
                name = dist.metadata["Name"]
            except Exception:
                continue
            top_level = dist.read_text("top_level.txt") or ""
            for package in top_level.splitlines():
                package = package.strip()
                if package:
                    pkg_to_dist[package].append(name)
        return dict(pkg_to_dist)

    importlib_metadata.packages_distributions = _safe_packages_distributions  # type: ignore[assignment]
    import transformers  # noqa: F401


def _load_run_metadata(report_path: Optional[Path]) -> Dict[str, Any]:
    if report_path is None or not report_path.exists():
        return {}
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    run = payload.get("run")
    return dict(run) if isinstance(run, Mapping) else {}


def _phase3_split_examples(
    *,
    dimension: PolicyDimension,
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
    split_strategy: str,
    train_pool: str,
    mp_data_dir: Optional[Path],
) -> Dict[str, Dict[str, str]]:
    """Reuse the phase-3 script's split construction."""

    from scripts.phase3_full_pipeline_optimize import _build_examples  # type: ignore

    train, val, test = _build_examples(
        dimension,
        train_pool,
        mp_data_dir,
        int(train_n),
        int(val_n),
        int(test_n),
        int(seed),
        split_strategy,
    )
    out: Dict[str, Dict[str, str]] = {"train": {}, "val": {}, "test": {}}
    for split, examples in (("train", train), ("val", val), ("test", test)):
        for ex in examples:
            mid = str(getattr(ex, "manifesto_id", "") or "").strip()
            text = str(getattr(ex, "text", "") or "")
            if mid and text:
                out[split][mid] = text
    return out


def _order_split_rows(
    rows_by_id: Mapping[str, Mapping[str, Any]],
    *,
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
) -> Dict[str, Dict[str, str]]:
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


def _get_text_for_row(
    *,
    row: Mapping[str, Any],
    split_texts: Mapping[str, str],
    dataset: Optional[ManifestoDataset],
) -> str:
    text = str(row.get("text") or row.get("document_text") or "").strip()
    if text:
        return text
    mid = _row_manifesto_id(row)
    text = str(split_texts.get(mid) or "").strip()
    if text:
        return text
    if dataset is not None and mid:
        sample = dataset.get_sample(mid)
        if sample is not None and getattr(sample, "text", None):
            return str(sample.text)
    return ""


def _add_labeled_node(
    tree: LabeledTree,
    *,
    node_id: str,
    level: int,
    text: str,
    score: float,
    char_start: int,
    char_end: int,
    is_leaf: bool,
    label_source: str,
    left_child_id: Optional[str] = None,
    right_child_id: Optional[str] = None,
    teacher_summary: Optional[str] = None,
    summary_source: Optional[str] = None,
) -> None:
    metadata: Dict[str, Any] = {
        "char_start": int(char_start),
        "char_end": int(char_end),
        "node_id": str(node_id),
        "is_leaf": bool(is_leaf),
        "label_source": str(label_source),
        "g_training_role": "leaf" if is_leaf else "merge",
        "f_input_kind": "summary_embedding",
    }
    if teacher_summary:
        metadata["teacher_summary"] = str(teacher_summary)
        metadata["target_summary"] = str(teacher_summary)
        metadata["teacher_summary_source"] = str(summary_source or "existing_gemma_result_root")
        if is_leaf:
            metadata["teacher_leaf_summary"] = str(teacher_summary)
        else:
            metadata["teacher_merge_summary"] = str(teacher_summary)
    else:
        metadata["missing_teacher_summary"] = True
    tree.add_node(
        LabeledNode(
            node_id=str(node_id),
            doc_id=tree.doc_id,
            level=int(level),
            text=str(text),
            score=float(score),
            left_child_id=left_child_id,
            right_child_id=right_child_id,
            metadata=metadata,
        )
    )


def _build_partial_labeled_tree(
    *,
    row: Mapping[str, Any],
    text: str,
    split: str,
    dimension: str,
    target_source: str,
    expert_target_scale: str,
    chunk_chars: int,
    source_results_path: Path,
) -> Optional[LabeledTree]:
    mid = _row_manifesto_id(row)
    summary = _row_summary(row)
    target = _row_target_score(
        row,
        dimension=dimension,
        target_source=target_source,
        expert_scale=expert_target_scale,
    )
    teacher_score = _row_teacher_score(row, dimension=dimension)
    expert_score = _row_expert_score(row, dimension=dimension, expert_scale=expert_target_scale)
    expert_raw = raw_benoit_expert_from_row(row, dimension=dimension)
    if not mid or not text.strip() or not summary or target is None:
        return None

    chunks = chunk_for_ops(text, max_chars=int(chunk_chars), strategy="axis")
    if not chunks:
        return None
    tree = LabeledTree(
        doc_id=str(mid),
        document_text=str(text),
        document_score=float(target),
        label_source=f"existing_gemma4_{target_source}",
        metadata={
            "artifact_version": "manifesto_dimension_existing_results_v1",
            "split": str(split),
            "dimension": str(dimension),
            "target_source": str(target_source),
            "teacher_score_1_7": teacher_score,
            "expert_score_1_7": expert_score,
            "expert_score_raw_benoit": expert_raw,
            **expert_scale_metadata(dimension=dimension, scale=expert_target_scale),
            "leaf_size_chars": int(chunk_chars),
            "chunking_source": "src.preprocessing.chunker.chunk_for_ops(strategy='axis')",
            "topology_policy": {
                "kind": "existing_phase_fixed_char_windows",
                "leaf_size_chars": int(chunk_chars),
                "actual_leaves": int(len(chunks)),
            },
            "topology_replay": "exact_artifact_spans",
            "source_results_path": str(source_results_path),
            "partial_artifact": True,
            "partial_artifact_reason": "existing result rows contain root summaries/scores but not all node summaries",
            "paper_to_lean_local_law_mapping": {
                "leaf": "C1_sufficiency",
                "idempotence": "C2_idempotence",
                "merge": "C3_associativity",
            },
        },
    )
    current: List[Tuple[str, int, int, int]] = []
    tree.levels = []
    for idx, chunk in enumerate(chunks):
        node_id = f"node_l0_{idx:05d}"
        start = int(getattr(chunk, "start_char", 0))
        end = int(getattr(chunk, "end_char", start + len(str(chunk.text))))
        _add_labeled_node(
            tree,
            node_id=node_id,
            level=0,
            text=str(chunk.text),
            score=float(target),
            char_start=start,
            char_end=end,
            is_leaf=True,
            label_source=tree.label_source,
        )
        current.append((node_id, start, end, 0))

    level = 1
    sibling_triples: List[Dict[str, str]] = []
    while len(current) > 1:
        next_level: List[Tuple[str, int, int, int]] = []
        for pair_idx in range(0, len(current), 2):
            left = current[pair_idx]
            right = current[pair_idx + 1] if pair_idx + 1 < len(current) else left
            node_id = f"node_l{level}_{len(next_level):05d}"
            start = int(left[1])
            end = int(right[2])
            parent_text = text[start:end]
            is_root = len(current) <= 2
            _add_labeled_node(
                tree,
                node_id=node_id,
                level=level,
                text=parent_text,
                score=float(target),
                char_start=start,
                char_end=end,
                is_leaf=False,
                label_source=tree.label_source,
                left_child_id=left[0],
                right_child_id=right[0],
                teacher_summary=summary if is_root else None,
                summary_source="existing_gemma_result_root" if is_root else None,
            )
            sibling_triples.append(
                {
                    "left_node_id": str(left[0]),
                    "right_node_id": str(right[0]),
                    "parent_node_id": str(node_id),
                }
            )
            next_level.append((node_id, start, end, level))
        current = next_level
        level += 1

    if len(chunks) == 1:
        only_id = "node_l0_00000"
        node = tree.get_node(only_id)
        if node is not None:
            node.metadata["teacher_summary"] = summary
            node.metadata["target_summary"] = summary
            node.metadata["teacher_summary_source"] = "existing_gemma_result_root"
            node.metadata.pop("missing_teacher_summary", None)

    tree.metadata["sibling_triples"] = sibling_triples
    tree.metadata["idempotence_pairs"] = []
    return tree


def _build_labeled_trees(
    *,
    rows: Sequence[Mapping[str, Any]],
    split_ids: Mapping[str, Mapping[str, str]],
    dimension: str,
    target_source: str,
    expert_target_scale: str,
    chunk_chars: int,
    source_results_path: Path,
    mp_data_dir: Optional[Path],
) -> Tuple[List[LabeledTree], Dict[str, Any]]:
    rows_by_id = {_row_manifesto_id(row): row for row in rows if _row_manifesto_id(row)}
    dataset: Optional[ManifestoDataset] = None
    trees: List[LabeledTree] = []
    skipped: Dict[str, int] = {"missing_row": 0, "missing_text": 0, "missing_target_or_summary": 0}

    if any(not text for split in split_ids.values() for text in split.values()):
        dataset = ManifestoDataset(data_dir=mp_data_dir, require_text=True)

    for split, id_to_text in split_ids.items():
        for mid, split_text in id_to_text.items():
            row = rows_by_id.get(str(mid))
            if row is None:
                skipped["missing_row"] += 1
                continue
            text = _get_text_for_row(row=row, split_texts={str(mid): split_text}, dataset=dataset)
            if not text.strip():
                skipped["missing_text"] += 1
                continue
            tree = _build_partial_labeled_tree(
                row=row,
                text=text,
                split=str(split),
                dimension=dimension,
                target_source=target_source,
                expert_target_scale=expert_target_scale,
                chunk_chars=int(chunk_chars),
                source_results_path=source_results_path,
            )
            if tree is None:
                skipped["missing_target_or_summary"] += 1
                continue
            trees.append(tree)
    counts = {
        "total": len(trees),
        "train": sum(1 for tree in trees if tree.metadata.get("split") == "train"),
        "val": sum(1 for tree in trees if tree.metadata.get("split") == "val"),
        "test": sum(1 for tree in trees if tree.metadata.get("split") == "test"),
        "skipped": skipped,
    }
    return trees, counts


def _write_prediction_report(
    *,
    trees: Sequence[LabeledTree],
    model_path: Path,
    embedding_client: Any,
    output_dir: Path,
    dimension: str,
    target_source: str,
) -> Dict[str, Any]:
    from src.training.embedding_proxy import load_embedding_proxy_model

    model = load_embedding_proxy_model(model_path)
    rows: List[Dict[str, Any]] = []
    for tree in trees:
        if tree.metadata.get("split") != "test":
            continue
        root_summary = ""
        for level_ids in reversed(tree.levels):
            if not level_ids:
                continue
            node = tree.get_node(str(level_ids[0]))
            if node is not None:
                root_summary = str((node.metadata or {}).get("teacher_summary") or "")
                break
        if not root_summary:
            continue
        predicted_norm = model.predict_from_embedding(embedding_client.embed_texts([root_summary])[0])
        predicted = _denormalize_score_1_7(predicted_norm)
        teacher = _safe_float((tree.metadata or {}).get("teacher_score_1_7"))
        expert = _safe_float((tree.metadata or {}).get("expert_score_1_7"))
        target = teacher if target_source == "teacher" else expert
        rows.append(
            {
                "manifesto_id": tree.doc_id,
                "dimension": dimension,
                "split": "test",
                "prediction_1_7": float(predicted),
                "target_1_7": target,
                "teacher_score_1_7": teacher,
                "expert_score_1_7": expert,
            }
        )

    pred_path = output_dir / "fit_predictions_test.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _metric(target_key: str) -> Dict[str, Any]:
        preds: List[float] = []
        truths: List[float] = []
        for row in rows:
            truth = _safe_float(row.get(target_key))
            pred = _safe_float(row.get("prediction_1_7"))
            if truth is None or pred is None:
                continue
            preds.append(pred)
            truths.append(truth)
        if len(preds) >= 4:
            payload = compute_corpus_pearson_r(preds, truths).as_dict()
        else:
            payload = {"n": len(preds), "pearson_r": None}
        if preds:
            payload["mae_1_7"] = float(sum(abs(p - t) for p, t in zip(preds, truths)) / len(preds))
        else:
            payload["mae_1_7"] = None
        return payload

    report = {
        "prediction_path": str(pred_path),
        "target_source": target_source,
        "teacher_fit": _metric("teacher_score_1_7"),
        "expert_eval": _metric("expert_score_1_7"),
    }
    return report


def _build_trl_config(args: argparse.Namespace):
    from src.training.trl_training import (
        TRLLoraConfig,
        TRLPropensityWeightingConfig,
        TRLQuantizationConfig,
        TRLSequenceConfig,
        TRLTrainingConfig,
    )

    return TRLTrainingConfig(
        train=TrainConfig(
            epochs=int(args.trl_epochs),
            batch_size=int(args.trl_batch_size),
            gradient_accumulation_steps=int(args.trl_grad_accumulation_steps),
            logging_steps=int(args.trl_logging_steps),
            save_steps=int(args.trl_save_steps),
        ),
        optimizer=OptimizerConfig(
            learning_rate=float(args.trl_learning_rate),
            warmup_ratio=float(args.trl_warmup_ratio),
        ),
        runtime=RuntimeConfig(
            device=str(args.trl_device),
            bf16=not bool(args.no_bf16),
            gradient_checkpointing=not bool(args.no_gradient_checkpointing),
        ),
        lora=TRLLoraConfig(use_lora=not bool(args.no_lora), lora_r=int(args.lora_rank)),
        quantization=TRLQuantizationConfig(load_in_4bit=not bool(args.no_4bit)),
        sequence=TRLSequenceConfig(max_length=int(args.trl_max_length)),
        propensity_weighting=TRLPropensityWeightingConfig(use_propensity_weighting=False),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dimension", choices=sorted(_DIM_FROM_NAME), default="economic")
    parser.add_argument(
        "--source-results",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "overnight_benoit" / "full_pipeline" / "economic" / "per_manifesto.jsonl",
        help="Existing Gemma/phase result per_manifesto JSONL.",
    )
    parser.add_argument(
        "--source-report",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "overnight_benoit" / "full_pipeline" / "economic" / "report.json",
        help="Optional report.json used to recover chunk_chars/model metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/manifesto_dimension_fit_existing/<timestamp>.",
    )
    parser.add_argument("--target-source", choices=["teacher", "expert"], default="teacher")
    parser.add_argument(
        "--expert-target-scale",
        choices=EXPERT_SCALE_CHOICES,
        default=EXPERT_SCALE_NORMALIZED_1_7,
        help=(
            "Scale used when materializing expert_score_1_7. "
            "normalized_1_7 derives a 1-7 target from Benoit's released expert_mean; "
            "raw_benoit preserves the older/raw behavior."
        ),
    )
    parser.add_argument("--split-source", choices=["phase3", "results-order"], default="phase3")
    parser.add_argument("--train-pool", choices=["expert-split", "openweight", "expert"], default="expert-split")
    parser.add_argument("--split-strategy", choices=["random", "label-stratified"], default="label-stratified")
    parser.add_argument("--train-n", type=int, default=80)
    parser.add_argument("--val-n", type=int, default=20)
    parser.add_argument("--test-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-chars", type=int, default=0, help="0 = read from --source-report.")
    parser.add_argument("--mp-data-dir", type=Path, default=None)

    parser.add_argument("--skip-g-export", action="store_true")
    parser.add_argument("--skip-f-lm-export", action="store_true")
    parser.add_argument("--skip-f-embedding-fit", action="store_true")
    parser.add_argument("--run-g-sft", action="store_true")
    parser.add_argument("--g-model-name", type=str, default=None)
    parser.add_argument("--run-f-lm-regression", action="store_true")
    parser.add_argument("--f-lm-model-name", type=str, default=None)
    parser.add_argument("--include-identity-targets", action=argparse.BooleanOptionalAction, default=False)
    add_manifesto_finetune_args(
        parser,
        kind="generic",
        help_text="Write treepo PreferenceDataset/fine-tune adapter bundle next to labeled_trees.jsonl.",
    )

    parser.add_argument("--embedding-backend", choices=["local-hf", "vllm", "hashing"], default="local-hf")
    parser.add_argument("--embedding-model", type=str, default="/mnt/data/models/Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-url", type=str, default=None)
    parser.add_argument("--embedding-api-key", type=str, default="EMPTY")
    parser.add_argument("--embedding-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-max-length", type=int, default=1024)
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--hashing-embedding-dim", type=int, default=256)
    parser.add_argument("--f-method", choices=["ridge", "linear_sgd"], default="ridge")
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--f-epochs", type=int, default=25)
    parser.add_argument("--f-learning-rate", type=float, default=5e-3)
    parser.add_argument("--f-weight-decay", type=float, default=1e-4)

    parser.add_argument("--trl-learning-rate", type=float, default=1e-5)
    parser.add_argument("--trl-warmup-ratio", type=float, default=0.03)
    parser.add_argument("--trl-epochs", type=int, default=1)
    parser.add_argument("--trl-batch-size", type=int, default=1)
    parser.add_argument("--trl-grad-accumulation-steps", type=int, default=4)
    parser.add_argument("--trl-max-length", type=int, default=2048)
    parser.add_argument("--trl-logging-steps", type=int, default=5)
    parser.add_argument("--trl-save-steps", type=int, default=100)
    parser.add_argument("--trl-device", default="auto")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "manifesto_dimension_fit_existing" / _now_stamp()
    output_dir.mkdir(parents=True, exist_ok=True)

    dim = _DIM_FROM_NAME[args.dimension]
    rows = _read_jsonl(args.source_results)
    rows_by_id = {_row_manifesto_id(row): row for row in rows if _row_manifesto_id(row)}
    run_metadata = _load_run_metadata(args.source_report)
    chunk_chars = int(args.chunk_chars or run_metadata.get("chunk_chars") or 24000)
    _preload_transformers_for_local_embedding(args)

    if args.split_source == "phase3":
        split_ids = _phase3_split_examples(
            dimension=dim,
            train_n=int(args.train_n),
            val_n=int(args.val_n),
            test_n=int(args.test_n),
            seed=int(args.seed),
            split_strategy=str(args.split_strategy),
            train_pool=str(args.train_pool),
            mp_data_dir=args.mp_data_dir,
        )
    else:
        split_ids = _order_split_rows(
            rows_by_id,
            train_n=int(args.train_n),
            val_n=int(args.val_n),
            test_n=int(args.test_n),
            seed=int(args.seed),
        )

    trees, tree_counts = _build_labeled_trees(
        rows=rows,
        split_ids=split_ids,
        dimension=dim.value,
        target_source=str(args.target_source),
        expert_target_scale=str(args.expert_target_scale),
        chunk_chars=chunk_chars,
        source_results_path=args.source_results,
        mp_data_dir=args.mp_data_dir,
    )
    if not trees:
        raise SystemExit("No labeled trees could be built from the selected result rows.")

    labeled_tree_path = write_labeled_trees_jsonl(output_dir / "labeled_trees.jsonl", trees)
    split_manifest = {
        split: sorted(mapping)
        for split, mapping in split_ids.items()
    }
    _write_json(output_dir / "split_ids.json", split_manifest)

    train_cfg = TrainConfig(train_splits=("train",), epochs=1)
    val_cfg = ValidationConfig(val_splits=("val",), enabled=True)
    test_cfg = TestConfig(test_splits=("test",), enabled=True)
    base_contract = TreePOContractSpec(
        contract_id=f"manifesto_{dim.value}_existing_gemma_fit",
        objective_kind="labeled_tree_distillation",
        state_semantics="natural_language_summary",
        adapter_preference="labeled_tree_distillation",
        rubric=get_preservation_rubric(dim),
        oracle_scale_min=1.0,
        oracle_scale_max=7.0,
        metadata={
            "dimension": dim.value,
            "source_results": str(args.source_results),
            "source_report": str(args.source_report) if args.source_report else None,
            "target_source": str(args.target_source),
            "expert_target_scale": str(args.expert_target_scale),
            "chunk_chars": int(chunk_chars),
            "split_source": str(args.split_source),
        },
    )
    model = TreePOModelSpec(kind="artifact_distillation", model="existing_gemma4_results")
    teacher_spec = {
        "kind": "existing_gemma4_results",
        "source_results": str(args.source_results),
        "source_report": str(args.source_report) if args.source_report else None,
        "model": run_metadata.get("model"),
        "dimension": dim.value,
        "target_source": str(args.target_source),
    }

    artifacts: Dict[str, Any] = {
        "labeled_trees": str(labeled_tree_path),
        "split_ids": str(output_dir / "split_ids.json"),
    }
    finetune_bundle = export_manifesto_finetune_bundle_from_args(
        args=args,
        trees=trees,
        output_dir=output_dir / "treepo_finetune",
        kind="generic",
    )
    if finetune_bundle:
        artifacts["finetune_bundle"] = str(output_dir / "treepo_finetune")
    results: Dict[str, Any] = {}

    if not args.skip_g_export:
        g_config = DistillationTrainConfig(
            contract=DistillationContractConfig(
                train_targets=(TRAIN_TARGET_G,),
                student_model_class=STUDENT_MODEL_LM_SFT,
                supervision_source="labeled_tree_artifact",
                teacher_model_spec=teacher_spec,
            ),
            run=RunConfig(output_dir=output_dir / "g_fit"),
            train=train_cfg,
            validation=val_cfg,
            test=test_cfg,
            summary_targets=SummaryTargetConfig(
                include_identity_targets=bool(args.include_identity_targets),
            ),
            g_lm=GLMConfig(
                run_trl_sft=bool(args.run_g_sft),
                model_name=args.g_model_name,
                trl_config=_build_trl_config(args) if args.run_g_sft else None,
            ),
        )
        if args.run_g_sft and not args.g_model_name:
            raise ValueError("--run-g-sft requires --g-model-name")
        g_result = fit_treepo_contract(
            contract=base_contract,
            model=model,
            run=replace(g_config.run, output_dir=output_dir / "g_fit"),
            train=train_cfg,
            validation=val_cfg,
            test=test_cfg,
            data={"labeled_trees": trees, "distillation_config": g_config},
            output_dir=output_dir / "g_contract_fit",
        )
        artifacts["g_contract_summary"] = g_result.artifacts.get("summary")
        artifacts["g_sft_train"] = str(output_dir / "g_fit" / "g_sft_train.jsonl")
        results["g"] = g_result.to_dict()

    if not args.skip_f_lm_export:
        f_lm_config = DistillationTrainConfig(
            contract=DistillationContractConfig(
                train_targets=(TRAIN_TARGET_F,),
                student_model_class=STUDENT_MODEL_LM_SCALAR_REGRESSION,
                supervision_source="labeled_tree_artifact",
                teacher_model_spec=teacher_spec,
            ),
            run=RunConfig(output_dir=output_dir / "f_lm_fit"),
            train=train_cfg,
            validation=val_cfg,
            test=test_cfg,
            score_targets=ScoreTargetConfig(
                include_identity_targets=bool(args.include_identity_targets),
                target_min=1.0,
                target_max=7.0,
            ),
            f_lm=FLMConfig(
                run_trl_scalar_reward=bool(args.run_f_lm_regression),
                model_name=args.f_lm_model_name,
                trl_config=_build_trl_config(args) if args.run_f_lm_regression else None,
            ),
        )
        if args.run_f_lm_regression and not args.f_lm_model_name:
            raise ValueError("--run-f-lm-regression requires --f-lm-model-name")
        f_lm_result = fit_treepo_contract(
            contract=base_contract,
            model=model,
            run=replace(f_lm_config.run, output_dir=output_dir / "f_lm_fit"),
            train=train_cfg,
            validation=val_cfg,
            test=test_cfg,
            data={"labeled_trees": trees, "distillation_config": f_lm_config},
            output_dir=output_dir / "f_lm_contract_fit",
        )
        artifacts["f_lm_contract_summary"] = f_lm_result.artifacts.get("summary")
        artifacts["f_lm_train"] = str(output_dir / "f_lm_fit" / "f_lm_regression_train.jsonl")
        results["f_lm"] = f_lm_result.to_dict()

    if not args.skip_f_embedding_fit:
        embedding_client = _make_embedding_client(args)
        f_embedding_config = DistillationTrainConfig(
            contract=DistillationContractConfig(
                train_targets=(TRAIN_TARGET_F,),
                student_model_class=STUDENT_MODEL_EMBEDDING_RIDGE_PROXY,
                supervision_source="labeled_tree_artifact",
                teacher_model_spec=teacher_spec,
            ),
            run=RunConfig(output_dir=output_dir / "f_embedding_fit"),
            train=train_cfg,
            validation=val_cfg,
            test=test_cfg,
            score_targets=ScoreTargetConfig(
                include_identity_targets=bool(args.include_identity_targets),
                target_min=1.0,
                target_max=7.0,
            ),
            f_embedding=FEmbeddingConfig(
                method=str(args.f_method),
                ridge_lambda=float(args.ridge_lambda),
                epochs=int(args.f_epochs),
                learning_rate=float(args.f_learning_rate),
                weight_decay=float(args.f_weight_decay),
                model_id=f"manifesto_{dim.value}_existing_gemma_f_embedding",
            ),
        )
        f_embedding_result = fit_treepo_contract(
            contract=base_contract,
            model=model,
            run=replace(f_embedding_config.run, output_dir=output_dir / "f_embedding_fit"),
            train=train_cfg,
            validation=val_cfg,
            test=test_cfg,
            data={"labeled_trees": trees, "distillation_config": f_embedding_config},
            output_dir=output_dir / "f_embedding_contract_fit",
            resources={
                RESOURCE_EMBEDDING: TreePOResourceSpec(
                    kind="object",
                    value=embedding_client,
                )
            },
        )
        artifacts["f_embedding_contract_summary"] = f_embedding_result.artifacts.get("summary")
        proxy_path = output_dir / "f_embedding_fit" / "f_embedding_proxy.json"
        artifacts["f_embedding_proxy"] = str(proxy_path)
        results["f_embedding"] = f_embedding_result.to_dict()
        if proxy_path.exists():
            results["f_embedding_test_report"] = _write_prediction_report(
                trees=trees,
                model_path=proxy_path,
                embedding_client=embedding_client,
                output_dir=output_dir,
                dimension=dim.value,
                target_source=str(args.target_source),
            )

    manifest = {
        "created_at": _now_iso(),
        "status": "completed",
        "dimension": dim.value,
        "source_results": str(args.source_results),
        "source_report": str(args.source_report) if args.source_report else None,
        "run_metadata": run_metadata,
        "config": {
            "target_source": str(args.target_source),
            "split_source": str(args.split_source),
            "train_pool": str(args.train_pool),
            "split_strategy": str(args.split_strategy),
            "train_n": int(args.train_n),
            "val_n": int(args.val_n),
            "test_n": int(args.test_n),
            "seed": int(args.seed),
            "chunk_chars": int(chunk_chars),
            "embedding_backend": str(args.embedding_backend),
            "embedding_model": str(args.embedding_model),
            "include_identity_targets": bool(args.include_identity_targets),
            "run_g_sft": bool(args.run_g_sft),
            "run_f_lm_regression": bool(args.run_f_lm_regression),
            "finetune_export": finetune_export_config(args),
        },
        "tree_counts": tree_counts,
        "dataset_counts": {
            "g_sft_records": len(build_g_sft_records(trees, include_identity_targets=bool(args.include_identity_targets))),
            "f_lm_records": len(
                build_f_lm_regression_records(
                    trees,
                    include_identity_targets=bool(args.include_identity_targets),
                    target_min=1.0,
                    target_max=7.0,
                )
            ),
        },
        "contract": {
            "train_targets": ["f", "g"],
            "supervision_source": "labeled_tree_artifact",
            "teacher_model_spec": teacher_spec,
            "target_scale": [1.0, 7.0],
        },
        "artifacts": artifacts,
        "finetune": finetune_bundle,
        "results": results,
    }
    _write_json(output_dir / "manifest.json", manifest)
    LOGGER.info("Wrote manifesto dimension fit smoke to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
