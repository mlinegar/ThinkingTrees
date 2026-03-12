#!/usr/bin/env python3
"""
Multilingual embedding smoke test for ThinkingTrees.

This script embeds a few manifesto documents (potentially in different
languages) using the configured embedding endpoint and prints pairwise cosine
similarities. It is a quick sanity check that multilingual embeddings are
working and broadly align similar-topic documents across languages.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


DEFAULT_IDS = [
    # Sweden: Social Democratic Labour Party (Swedish)
    "11320_199809",
    # Spain: PSOE (Spanish)
    "33320_199603",
]


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_manifesto_text(manifesto_id: str) -> str:
    path = PROJECT_ROOT / "data" / "raw" / "manifesto_project_full" / "texts" / f"{manifesto_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifesto text file: {path}")
    return _read_text_file(path)


def _load_manifesto_metadata(ids: List[str]) -> Dict[str, Dict[str, Any]]:
    csv_path = PROJECT_ROOT / "data" / "raw" / "manifesto_project_full" / "manifesto_maindataset.csv"
    if not csv_path.exists():
        return {}
    try:
        import pandas as pd
    except Exception:
        return {}

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return {}
    try:
        df["manifesto_id"] = df["party"].astype(str) + "_" + df["date"].astype(str)
    except Exception:
        return {}

    wanted = set(str(x) for x in ids)
    sub = df[df["manifesto_id"].astype(str).isin(wanted)]
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in sub.iterrows():
        manifesto_id = str(row.get("manifesto_id", "")).strip()
        if not manifesto_id:
            continue
        out[manifesto_id] = {
            "partyname": row.get("partyname"),
            "countryname": row.get("countryname"),
            "date": row.get("date"),
            "rile": row.get("rile"),
            "parfam": row.get("parfam"),
        }
    return out


def _window_starts(total_len: int, window_len: int, max_windows: int) -> List[int]:
    if total_len <= 0 or window_len <= 0 or max_windows <= 0:
        return [0]
    if total_len <= window_len:
        return [0]
    max_windows = max(1, int(max_windows))
    if max_windows == 1:
        return [0]
    span = total_len - window_len
    return [int(round(i * span / float(max_windows - 1))) for i in range(max_windows)]


def _build_windows(text: str, *, window_chars: int, max_windows: int) -> List[str]:
    raw = str(text or "")
    if window_chars <= 0:
        return [raw]
    starts = _window_starts(len(raw), int(window_chars), int(max_windows))
    out: List[str] = []
    for s in starts:
        out.append(raw[s : s + int(window_chars)])
    return out


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(vec) + 1e-12)
    return vec / denom


def _embed_doc(
    *,
    client: Any,
    text: str,
    window_chars: int,
    max_windows: int,
) -> np.ndarray:
    windows = _build_windows(text, window_chars=int(window_chars), max_windows=int(max_windows))
    vectors = client.embed_texts(windows)
    mat = np.asarray(vectors, dtype=np.float32)
    pooled = mat.mean(axis=0)
    return _l2_normalize(pooled)


def _pairwise_cosine(embs: np.ndarray) -> np.ndarray:
    embs = np.asarray(embs, dtype=np.float32)
    # If embeddings are L2-normalized, cosine is just dot product.
    return embs @ embs.T


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multilingual embedding smoke test (pairwise cosine similarity).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ids", nargs="+", default=list(DEFAULT_IDS), help="Manifesto IDs to embed")
    parser.add_argument("--embedding-url", type=str, default=None, help="Embedding endpoint base (OpenAI-compatible)")
    parser.add_argument("--embedding-model", type=str, default=None, help="Embedding model id served by the endpoint")
    parser.add_argument("--window-chars", type=int, default=6000, help="Chars per embedding window (0 = full text)")
    parser.add_argument("--max-windows", type=int, default=8, help="Max windows per document when pooling")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ids = [str(x).strip() for x in (args.ids or []) if str(x).strip()]
    if not ids:
        logger.error("No ids provided")
        return 2

    from src.config.settings import get_embedding_model, get_embedding_url, load_settings
    from src.training.embedding_proxy import VLLMEmbeddingClient

    settings = load_settings()
    api_base = (args.embedding_url or get_embedding_url(settings)).rstrip("/")
    model = (args.embedding_model or get_embedding_model(settings) or None)

    logger.info("Embedding endpoint: %s", api_base)
    if model:
        logger.info("Embedding model: %s", model)
    logger.info("Pooling: window_chars=%d max_windows=%d", int(args.window_chars), int(args.max_windows))

    meta = _load_manifesto_metadata(ids)

    client = VLLMEmbeddingClient(
        api_base=api_base,
        model=model,
        timeout_seconds=60.0,
        batch_size=32,
    )
    try:
        resolved = client.resolve_model()
        logger.info("Resolved embedding model id: %s", resolved)
    except Exception as e:
        logger.error("Embedding endpoint not reachable or misconfigured (%s).", e)
        logger.error("Start it with: ./scripts/start_embedding_server.sh")
        return 1

    doc_vectors: List[np.ndarray] = []
    docs: List[Dict[str, Any]] = []
    for manifesto_id in ids:
        text = _load_manifesto_text(manifesto_id)
        vec = _embed_doc(
            client=client,
            text=text,
            window_chars=int(args.window_chars),
            max_windows=int(args.max_windows),
        )
        doc_vectors.append(vec)
        docs.append(
            {
                "manifesto_id": manifesto_id,
                "meta": meta.get(manifesto_id, {}),
                "text_chars": len(text),
            }
        )

    mat = np.stack(doc_vectors, axis=0)
    sims = _pairwise_cosine(mat)

    print("")
    print("Multilingual embedding smoke summary")
    print(f"  embedding_url={api_base} model={model or 'auto'} window_chars={int(args.window_chars)} max_windows={int(args.max_windows)}")
    print("")
    for idx, doc in enumerate(docs):
        label = doc["manifesto_id"]
        info = doc.get("meta", {}) or {}
        if info.get("countryname") and info.get("partyname"):
            label += f" | {info.get('countryname')} | {info.get('partyname')}"
        print(f"- [{idx}] {label} (chars={doc.get('text_chars')})")
    print("")
    print("Pairwise cosine similarities:")
    for i in range(len(ids)):
        row = " ".join(f"{float(sims[i, j]):.3f}" for j in range(len(ids)))
        print(f"  {i}: {row}")

    payload = {
        "embedding_url": api_base,
        "embedding_model": model,
        "window_chars": int(args.window_chars),
        "max_windows": int(args.max_windows),
        "docs": docs,
        "cosine": sims.tolist(),
    }

    if args.json_out:
        out_path = args.json_out if args.json_out.is_absolute() else (PROJECT_ROOT / args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved JSON: %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
