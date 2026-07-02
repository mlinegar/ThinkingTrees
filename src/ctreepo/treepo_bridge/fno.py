from __future__ import annotations

import hashlib
from dataclasses import is_dataclass
from typing import Any, Mapping, Sequence


class HashingEmbeddingClient:
    """Deterministic offline embedding client for FNO smoke runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = int(dim)

    def resolve_model(self) -> str:
        return f"hashing_embedding:{self.dim}"

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in str(text or "").lower().split():
                digest = hashlib.blake2b(
                    token.encode("utf-8", errors="ignore"), digest_size=8
                ).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                vec[bucket] += -1.0 if (digest[4] & 1) else 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([float(v / norm) for v in vec])
        return out


THINKINGTREES_FNO_FAMILY = "thinkingtrees_fno"


def register_fno_family(name: str | None = None) -> str:
    """Register the ThinkingTrees FNO family and return its family name.

    Latest ``treepo`` ships a generic built-in ``family='fno'``. In that case
    this bridge keeps the package default intact and registers the richer
    ThinkingTrees implementation as ``family='thinkingtrees_fno'``. With older
    ``treepo`` installs that do not have built-in FNO, the bridge still fills
    ``family='fno'`` for compatibility.
    """

    from treepo.methods.families import list_families, register_family

    existing = set(list_families())
    family_name = str(name or (THINKINGTREES_FNO_FAMILY if "fno" in existing else "fno"))
    if family_name in existing:
        return family_name
    register_family(family_name, build_fno_family)
    return family_name


def build_fno_family(backend_config: Mapping[str, Any]) -> Any:
    """Build a ThinkingTrees ``FNOFamily`` from a treepo backend config.

    Expected keys:
    - ``fno_config``: optional ``src.ctreepo.fno_family.FNOFamilyConfig`` or mapping.
    - ``embedding_client``: optional client with ``embed_texts``.
    - ``embedding_dim`` / ``hashing_embedding_dim``: dimension for the default
      deterministic hashing client.
    - ``device``: optional torch device string/object.
    """

    from src.ctreepo.fno_family import FNOFamily, FNOFamilyConfig

    config_dict = dict(backend_config or {})
    embedding_client = config_dict.get("embedding_client")
    embedding_dim = int(
        config_dict.get(
            "embedding_dim",
            config_dict.get("hashing_embedding_dim", 64),
        )
    )
    if embedding_client is None:
        embedding_client = HashingEmbeddingClient(dim=embedding_dim)

    fno_config = _coerce_fno_config(
        config_dict.get("fno_config", config_dict.get("config")),
        default_embedding_dim=embedding_dim,
    )
    return FNOFamily(
        config=fno_config,
        embedding_client=embedding_client,
        device=config_dict.get("device"),
    )


def _coerce_fno_config(raw: Any, *, default_embedding_dim: int) -> Any:
    from src.ctreepo.fno_family import FNOFamilyConfig

    if isinstance(raw, FNOFamilyConfig):
        return raw
    if raw is None:
        return FNOFamilyConfig(
            hidden_channels=8,
            n_modes=4,
            n_layers=1,
            head_hidden_dim=16,
            epochs_per_iteration=1,
            batch_size=2,
            effective_embedding_dim=int(default_embedding_dim),
            embedding_max_length_tokens=None,
            identity_init=True,
        )
    if isinstance(raw, Mapping):
        payload = dict(raw)
    elif is_dataclass(raw):
        payload = dict(getattr(raw, "__dict__", {}) or {})
    else:
        raise TypeError(
            "backend_config['fno_config'] must be a ThinkingTrees FNOFamilyConfig "
            f"or mapping; got {type(raw).__name__}"
        )
    payload.pop("metadata", None)
    return FNOFamilyConfig(**payload)


__all__ = [
    "HashingEmbeddingClient",
    "THINKINGTREES_FNO_FAMILY",
    "build_fno_family",
    "register_fno_family",
]
