from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


DEFAULT_DIMENSIONS = (
    "economic",
    "social",
    "decentralization",
    "environment",
    "eu",
    "immigration",
)
SPLIT_SCHEMA_VERSION = "ctreepo.manifesto_coverage_split.v1"
DEFAULT_FULL_DOC_DATA_DIR = Path("data") / "raw" / "manifesto_corpus_benoit"


def load_split_ids(split_dir: str | Path) -> Dict[str, List[str]]:
    path = Path(split_dir) / "split_ids.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, List[str]] = {}
    for split, values in dict(payload).items():
        if isinstance(values, Mapping):
            out[str(split)] = [str(key) for key in values.keys()]
        else:
            out[str(split)] = [str(item) for item in list(values or [])]
    return out


def load_split_summary(split_dir: str | Path) -> Dict[str, Any]:
    path = Path(split_dir) / "coverage_split_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def resolve_full_doc_data_dir(path: Optional[Path], *, project_root: Optional[Path] = None) -> Optional[Path]:
    if path is not None:
        return Path(path)
    candidate = DEFAULT_FULL_DOC_DATA_DIR
    if project_root is not None and not candidate.is_absolute():
        candidate = Path(project_root) / candidate
    return candidate if candidate.exists() else None


__all__ = [
    "DEFAULT_DIMENSIONS",
    "DEFAULT_FULL_DOC_DATA_DIR",
    "SPLIT_SCHEMA_VERSION",
    "load_split_ids",
    "load_split_summary",
    "resolve_full_doc_data_dir",
]
