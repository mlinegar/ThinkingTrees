#!/usr/bin/env python3
"""
Cross-language embedding validation for C-TreePO.

Embeds a set of manifestos from different countries/languages and evaluates
whether ideologically similar parties cluster together in embedding space
regardless of language. This validates the Qwen3-Embedding-8B foundation
before building learned sketches on top.

Usage:
    ./venv/bin/python scripts/eval_crosslang_embeddings.py
    ./venv/bin/python scripts/eval_crosslang_embeddings.py --json-out outputs/crosslang_eval.json
    ./venv/bin/python scripts/eval_crosslang_embeddings.py --embedding-url http://localhost:8003/v1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test set: late-1990s socialist vs conservative parties
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TestDoc:
    manifesto_id: str
    label: str         # human-readable
    family: str        # "socialist" or "conservative"
    family_code: int   # parfam code


TEST_DOCS = [
    # Socialist / Social Democratic (parfam=30)
    TestDoc("11320_199809", "SAP (Sweden)", "socialist", 30),
    TestDoc("33320_199603", "PSOE (Spain)", "socialist", 30),
    TestDoc("31320_199705", "PS (France)", "socialist", 30),
    # Conservative / Christian Democratic (parfam=50/60)
    TestDoc("51620_199705", "Conservatives (UK)", "conservative", 60),
    TestDoc("41521_199410", "CDU/CSU (Germany)", "conservative", 50),
]


# ---------------------------------------------------------------------------
# Embedding helpers (mirrors run_multilang_embedding_smoke.py)
# ---------------------------------------------------------------------------

def _load_manifesto_text(manifesto_id: str) -> str:
    path = PROJECT_ROOT / "data" / "raw" / "manifesto_project_full" / "texts" / f"{manifesto_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifesto text: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def _build_windows(text: str, window_chars: int, max_windows: int) -> List[str]:
    if window_chars <= 0 or len(text) <= window_chars:
        return [text]
    n_windows = min(max_windows, max(1, len(text) // window_chars))
    if n_windows == 1:
        return [text[:window_chars]]
    span = len(text) - window_chars
    starts = [int(round(i * span / (n_windows - 1))) for i in range(n_windows)]
    return [text[s : s + window_chars] for s in starts]


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec) + 1e-12)
    return vec / norm


def _embed_doc(client: Any, text: str, window_chars: int, max_windows: int) -> np.ndarray:
    windows = _build_windows(text, window_chars, max_windows)
    vectors = client.embed_texts(windows)
    mat = np.asarray(vectors, dtype=np.float32)
    pooled = mat.mean(axis=0)
    return _l2_normalize(pooled)


def _pairwise_cosine(embs: np.ndarray) -> np.ndarray:
    return embs @ embs.T


# ---------------------------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------------------------

def compute_cluster_separation(
    sims: np.ndarray,
    families: List[str],
) -> Dict[str, Any]:
    """Compute intra-family vs inter-family similarity statistics."""
    n = len(families)
    intra_sims: List[float] = []
    inter_sims: List[float] = []

    for i in range(n):
        for j in range(i + 1, n):
            s = float(sims[i, j])
            if families[i] == families[j]:
                intra_sims.append(s)
            else:
                inter_sims.append(s)

    avg_intra = float(np.mean(intra_sims)) if intra_sims else 0.0
    avg_inter = float(np.mean(inter_sims)) if inter_sims else 0.0
    separation = avg_intra - avg_inter

    return {
        "avg_intra_family_similarity": round(avg_intra, 4),
        "avg_inter_family_similarity": round(avg_inter, 4),
        "separation": round(separation, 4),
        "intra_pairs": len(intra_sims),
        "inter_pairs": len(inter_sims),
        "intra_sims": [round(s, 4) for s in intra_sims],
        "inter_sims": [round(s, 4) for s in inter_sims],
        "pass": separation > 0,
    }


def compute_retrieval_precision(
    sims: np.ndarray,
    families: List[str],
) -> Dict[str, Any]:
    """For each document, check if the nearest neighbor is same family (precision@1)."""
    n = len(families)
    correct = 0
    details: List[Dict[str, Any]] = []

    for i in range(n):
        # Mask self-similarity
        row = sims[i].copy()
        row[i] = -1.0
        best_j = int(np.argmax(row))
        is_correct = families[i] == families[best_j]
        if is_correct:
            correct += 1
        details.append({
            "query_idx": i,
            "best_match_idx": best_j,
            "similarity": round(float(row[best_j]), 4),
            "same_family": is_correct,
        })

    precision = correct / n if n > 0 else 0.0
    return {
        "precision_at_1": round(precision, 4),
        "correct": correct,
        "total": n,
        "details": details,
    }


def compute_pca_2d(embs: np.ndarray) -> List[List[float]]:
    """Simple 2D PCA for visualization (no sklearn dependency)."""
    centered = embs - embs.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, len(embs) - 1)
    # Power iteration for top 2 eigenvectors
    d = cov.shape[0]
    coords = []
    for _ in range(2):
        v = np.random.randn(d).astype(np.float32)
        for _ in range(100):
            v = cov @ v
            v = v / (np.linalg.norm(v) + 1e-12)
        coords.append(centered @ v)
        # Deflate
        cov = cov - np.outer(v, v) * (v @ cov @ v)
    return [[round(float(coords[0][i]), 4), round(float(coords[1][i]), 4)] for i in range(len(embs))]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-language embedding validation for C-TreePO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--embedding-url", type=str, default=None,
                        help="Embedding endpoint base URL")
    parser.add_argument("--embedding-model", type=str, default=None,
                        help="Embedding model id")
    parser.add_argument("--window-chars", type=int, default=6000,
                        help="Chars per embedding window")
    parser.add_argument("--max-windows", type=int, default=8,
                        help="Max windows per document")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Optional JSON output path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ------------------------------------------------------------------
    # Load settings + embedding client
    # ------------------------------------------------------------------
    from src.config.settings import get_embedding_model, get_embedding_url, load_settings
    from src.training.embedding_proxy import VLLMEmbeddingClient

    settings = load_settings()
    api_base = (args.embedding_url or get_embedding_url(settings)).rstrip("/")
    model = args.embedding_model or get_embedding_model(settings) or None

    logger.info("Embedding endpoint: %s  model: %s", api_base, model or "auto")

    client = VLLMEmbeddingClient(
        api_base=api_base,
        model=model,
        timeout_seconds=60.0,
        batch_size=32,
    )
    try:
        resolved = client.resolve_model()
        logger.info("Resolved model: %s", resolved)
    except Exception as e:
        logger.error("Embedding endpoint not reachable (%s). Start with: ./scripts/start_embedding_server.sh", e)
        return 1

    # ------------------------------------------------------------------
    # Load RILE ground truth
    # ------------------------------------------------------------------
    rile_scores: Dict[str, float] = {}
    try:
        import pandas as pd
        csv_path = PROJECT_ROOT / "data" / "raw" / "manifesto_project_full" / "manifesto_maindataset.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, low_memory=False)
            df["manifesto_id"] = df["party"].astype(str) + "_" + df["date"].astype(str)
            for doc in TEST_DOCS:
                row = df[df["manifesto_id"] == doc.manifesto_id]
                if len(row) > 0:
                    rile_scores[doc.manifesto_id] = float(row.iloc[0]["rile"])
    except Exception as e:
        logger.warning("Could not load RILE scores: %s", e)

    # ------------------------------------------------------------------
    # Embed all test documents
    # ------------------------------------------------------------------
    embeddings: List[np.ndarray] = []
    doc_info: List[Dict[str, Any]] = []

    for doc in TEST_DOCS:
        text = _load_manifesto_text(doc.manifesto_id)
        vec = _embed_doc(client, text, args.window_chars, args.max_windows)
        embeddings.append(vec)
        doc_info.append({
            "manifesto_id": doc.manifesto_id,
            "label": doc.label,
            "family": doc.family,
            "family_code": doc.family_code,
            "text_chars": len(text),
            "rile": rile_scores.get(doc.manifesto_id),
        })
        logger.info("Embedded %s (%d chars, RILE=%.1f)",
                     doc.label, len(text),
                     rile_scores.get(doc.manifesto_id, float("nan")))

    mat = np.stack(embeddings, axis=0)
    sims = _pairwise_cosine(mat)
    families = [d.family for d in TEST_DOCS]

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    cluster_metrics = compute_cluster_separation(sims, families)
    retrieval_metrics = compute_retrieval_precision(sims, families)
    pca_coords = compute_pca_2d(mat)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("")
    print("=" * 70)
    print("  Cross-Language Embedding Validation")
    print("=" * 70)
    print(f"  Endpoint: {api_base}  Model: {model or 'auto'}")
    print(f"  Window: {args.window_chars} chars x {args.max_windows} max")
    print("")

    print("Documents:")
    for i, doc in enumerate(doc_info):
        rile_str = f"RILE={doc['rile']:+.1f}" if doc['rile'] is not None else "RILE=?"
        print(f"  [{i}] {doc['label']:30s} {doc['family']:12s} {rile_str}  ({doc['text_chars']} chars)")
    print("")

    print("Pairwise cosine similarities:")
    header = "     " + "  ".join(f"[{i}]  " for i in range(len(TEST_DOCS)))
    print(header)
    for i in range(len(TEST_DOCS)):
        row_str = "  ".join(f"{float(sims[i, j]):.3f}" for j in range(len(TEST_DOCS)))
        print(f"  [{i}] {row_str}")
    print("")

    print("Cluster separation:")
    print(f"  Avg intra-family similarity: {cluster_metrics['avg_intra_family_similarity']:.4f}")
    print(f"  Avg inter-family similarity: {cluster_metrics['avg_inter_family_similarity']:.4f}")
    print(f"  Separation (intra - inter):  {cluster_metrics['separation']:+.4f}")
    status = "PASS" if cluster_metrics["pass"] else "FAIL"
    print(f"  Status: {status}")
    print("")

    print("Retrieval precision@1 (nearest neighbor same family):")
    print(f"  {retrieval_metrics['correct']}/{retrieval_metrics['total']} = {retrieval_metrics['precision_at_1']:.0%}")
    for detail in retrieval_metrics["details"]:
        q = detail["query_idx"]
        m = detail["best_match_idx"]
        ok = "OK" if detail["same_family"] else "MISS"
        print(f"    [{q}] {doc_info[q]['label']:30s} -> [{m}] {doc_info[m]['label']:30s} (sim={detail['similarity']:.3f}) {ok}")
    print("")

    print("PCA 2D coordinates (for plotting):")
    for i, (x, y) in enumerate(pca_coords):
        print(f"  [{i}] {doc_info[i]['label']:30s} ({x:+.3f}, {y:+.3f})")
    print("")

    overall_pass = cluster_metrics["pass"] and retrieval_metrics["precision_at_1"] >= 0.8
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    payload = {
        "config": {
            "embedding_url": api_base,
            "embedding_model": model,
            "window_chars": args.window_chars,
            "max_windows": args.max_windows,
        },
        "docs": doc_info,
        "cosine_matrix": sims.tolist(),
        "cluster_metrics": cluster_metrics,
        "retrieval_metrics": retrieval_metrics,
        "pca_2d": pca_coords,
        "overall_pass": overall_pass,
    }

    if args.json_out:
        out_path = args.json_out if args.json_out.is_absolute() else (PROJECT_ROOT / args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved JSON: %s", out_path)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
