from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Dict, List, Sequence


class HashingEmbeddingClient:
    """Deterministic offline embedding client for fast smoke tests."""

    def __init__(self, *, dim: int = 256, model: str = "hashing_embedding"):
        self.dim = int(max(8, dim))
        self.model = str(model)

    def resolve_model(self) -> str:
        return f"{self.model}:{self.dim}"

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        outputs: list[list[float]] = []
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
        import os as _os

        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device
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

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        import torch

        tokenizer, model, device = self._load()
        outputs: list[list[float]] = []
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


def make_embedding_client_from_args(args: Any):
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

        embedding_url = getattr(args, "embedding_url", None) or getattr(args, "embedding_api_base", None)
        if not embedding_url:
            raise ValueError("--embedding-backend vllm requires an embedding URL/API base")
        return VLLMEmbeddingClient(
            api_base=str(embedding_url),
            model=args.embedding_model,
            api_key=str(getattr(args, "embedding_api_key", "EMPTY")),
            timeout_seconds=float(getattr(args, "embedding_timeout_seconds", 60.0)),
            batch_size=int(args.embedding_batch_size),
        )
    raise ValueError(f"Unsupported embedding backend: {args.embedding_backend!r}")


def preload_transformers_for_local_embedding(args: Any) -> None:
    """Import transformers before script imports that can perturb metadata scanning."""

    if getattr(args, "embedding_backend", None) != "local-hf":
        return
    try:
        import transformers  # noqa: F401
        return
    except TypeError as exc:
        if "packages_distributions" not in str(exc) and "'NoneType' object is not subscriptable" not in str(exc):
            raise

    import importlib.metadata as importlib_metadata

    def safe_packages_distributions() -> Dict[str, List[str]]:
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

    importlib_metadata.packages_distributions = safe_packages_distributions  # type: ignore[assignment]
    import transformers  # noqa: F401


__all__ = [
    "HashingEmbeddingClient",
    "LocalHFEmbeddingClient",
    "make_embedding_client_from_args",
    "preload_transformers_for_local_embedding",
]
