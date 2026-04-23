#!/usr/bin/env python3
"""
Evaluate a trained CTreePO model on cross-language retrieval.

Loads a trained model checkpoint, embeds test documents, computes root
sketches, and evaluates:
  1. RILE prediction accuracy (MAE on [-100, +100] scale)
  2. Cross-language retrieval (nearest-neighbor by sketch cosine similarity)
  3. Sketch-space clustering (intra- vs inter-family separation)
  4. Comparison: sketch similarity vs raw embedding similarity

Usage:
    ./venv/bin/python scripts/eval_ctreepo_crosslang.py --model outputs/ctreepo/<run>/best.pt
    ./venv/bin/python scripts/eval_ctreepo_crosslang.py --model outputs/ctreepo/<run>/best.pt --pilot
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


# Same test set as eval_crosslang_embeddings.py for direct comparison
PILOT_IDS = [
    ("11320_199809", "SAP (Sweden)", "socialist"),
    ("33320_199603", "PSOE (Spain)", "socialist"),
    ("31320_199705", "PS (France)", "socialist"),
    ("51620_199705", "Conservatives (UK)", "conservative"),
    ("41521_199410", "CDU/CSU (Germany)", "conservative"),
]


def _pairwise_cosine(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normed = vecs / np.maximum(norms, 1e-12)
    return normed @ normed.T


def _cluster_separation(sims: np.ndarray, families: List[str]) -> Dict[str, Any]:
    n = len(families)
    intra, inter = [], []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sims[i, j])
            (intra if families[i] == families[j] else inter).append(s)
    avg_intra = float(np.mean(intra)) if intra else 0.0
    avg_inter = float(np.mean(inter)) if inter else 0.0
    return {
        "avg_intra": round(avg_intra, 4),
        "avg_inter": round(avg_inter, 4),
        "separation": round(avg_intra - avg_inter, 4),
        "pass": avg_intra > avg_inter,
    }


def _retrieval_precision(sims: np.ndarray, families: List[str]) -> float:
    n = len(families)
    correct = 0
    for i in range(n):
        row = sims[i].copy()
        row[i] = -1.0
        best_j = int(np.argmax(row))
        if families[i] == families[best_j]:
            correct += 1
    return correct / max(n, 1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate CTreePO model on cross-language retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--pilot", action="store_true", help="Use pilot test set")
    parser.add_argument("--ids", nargs="+", default=None, help="Custom manifesto IDs")
    parser.add_argument("--sketch-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--merge-type", default="gated")
    parser.add_argument("--tree-model-version", choices=["legacy", "v2"], default=None)
    parser.add_argument("--window-size", type=int, default=1200)
    parser.add_argument("--embedding-url", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from src.config.settings import get_embedding_model, get_embedding_url, load_settings
    from src.tasks.manifesto.data_loader import ManifestoDataset
    from src.training.embedding_proxy import VLLMEmbeddingClient
    from src.tree.ctreepo_model import load_ctreepo_model_checkpoint
    from src.tree.embedding_tree import build_tree_from_text, forward_ctreepo, get_root_sketch

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model, config = load_ctreepo_model_checkpoint(
        args.model,
        config_overrides={
            "sketch_dim": int(args.sketch_dim),
            "hidden_dim": int(args.hidden_dim),
            "merge_type": str(args.merge_type),
            "head_names": ("rile",),
        },
        tree_model_version=args.tree_model_version,
        map_location="cpu",
    )
    logger.info("Loaded model from %s", args.model)

    # ------------------------------------------------------------------
    # Set up embedding client
    # ------------------------------------------------------------------
    settings = load_settings()
    api_base = (args.embedding_url or get_embedding_url(settings)).rstrip("/")
    model_name = args.embedding_model or get_embedding_model(settings) or None

    client = VLLMEmbeddingClient(api_base=api_base, model=model_name, timeout_seconds=60.0, batch_size=32)
    try:
        client.resolve_model()
    except Exception as e:
        logger.error("Embedding server not reachable (%s)", e)
        return 1

    # ------------------------------------------------------------------
    # Load test documents
    # ------------------------------------------------------------------
    test_specs = []
    if args.pilot:
        test_specs = list(PILOT_IDS)
    elif args.ids:
        test_specs = [(mid, mid, "unknown") for mid in args.ids]
    else:
        test_specs = list(PILOT_IDS)

    ds = ManifestoDataset()
    samples = []
    families = []
    labels = []
    for mid, label, family in test_specs:
        sample = ds.get_sample(mid)
        if sample is None:
            logger.warning("Could not load %s, skipping", mid)
            continue
        samples.append(sample)
        families.append(family)
        labels.append(label)

    if not samples:
        logger.error("No samples loaded")
        return 2

    # ------------------------------------------------------------------
    # Compute sketches + raw embeddings
    # ------------------------------------------------------------------
    sketch_vecs: List[np.ndarray] = []
    raw_emb_vecs: List[np.ndarray] = []
    rile_preds: List[float] = []
    rile_trues: List[float] = []

    for sample in samples:
        # Build tree and compute sketches
        nodes = build_tree_from_text(
            sample.text, client,
            window_size=args.window_size, window_overlap=150,
        )
        forward_ctreepo(model, nodes)
        root_sketch = get_root_sketch(nodes)
        rile_pred = model.predict(root_sketch, "rile").item()

        sketch_vecs.append(root_sketch.detach().numpy())
        rile_preds.append(rile_pred)
        rile_trues.append(sample.rile)

        # Also compute raw embedding (average-pool) for comparison
        leaf_embs = [np.array(n.embedding) for n in nodes if n.is_leaf and n.embedding]
        if leaf_embs:
            raw_pooled = np.mean(leaf_embs, axis=0)
            raw_emb_vecs.append(raw_pooled)

        logger.info("  %s: true=%.1f pred=%.1f err=%.1f",
                     sample.manifesto_id, sample.rile, rile_pred, abs(rile_pred - sample.rile))

    sketch_mat = np.stack(sketch_vecs)
    raw_emb_mat = np.stack(raw_emb_vecs) if len(raw_emb_vecs) == len(samples) else None

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    # RILE prediction
    errors = [abs(p - t) for p, t in zip(rile_preds, rile_trues)]
    mae = float(np.mean(errors))

    # Sketch-space clustering
    sketch_sims = _pairwise_cosine(sketch_mat)
    sketch_cluster = _cluster_separation(sketch_sims, families)
    sketch_retrieval_p1 = _retrieval_precision(sketch_sims, families)

    # Raw embedding clustering (comparison)
    raw_cluster = None
    raw_retrieval_p1 = None
    if raw_emb_mat is not None:
        raw_sims = _pairwise_cosine(raw_emb_mat)
        raw_cluster = _cluster_separation(raw_sims, families)
        raw_retrieval_p1 = _retrieval_precision(raw_sims, families)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("")
    print("=" * 70)
    print("  CTreePO Cross-Language Evaluation")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print("")

    print("Per-document RILE predictions:")
    for i, sample in enumerate(samples):
        mark = "OK" if errors[i] < 15 else "MISS"
        print(f"  {labels[i]:30s} true={rile_trues[i]:+6.1f}  pred={rile_preds[i]:+6.1f}  err={errors[i]:.1f} {mark}")
    print(f"  MAE = {mae:.2f}  (target: < 15)")
    print("")

    print("Sketch-space clustering:")
    print(f"  Avg intra-family similarity: {sketch_cluster['avg_intra']:.4f}")
    print(f"  Avg inter-family similarity: {sketch_cluster['avg_inter']:.4f}")
    print(f"  Separation:                  {sketch_cluster['separation']:+.4f}")
    print(f"  Retrieval precision@1:       {sketch_retrieval_p1:.0%}")
    print(f"  Status: {'PASS' if sketch_cluster['pass'] else 'FAIL'}")
    print("")

    if raw_cluster is not None:
        print("Raw embedding clustering (comparison):")
        print(f"  Avg intra-family similarity: {raw_cluster['avg_intra']:.4f}")
        print(f"  Avg inter-family similarity: {raw_cluster['avg_inter']:.4f}")
        print(f"  Separation:                  {raw_cluster['separation']:+.4f}")
        print(f"  Retrieval precision@1:       {raw_retrieval_p1:.0%}")
        print("")
        improvement = (sketch_cluster["separation"] - raw_cluster["separation"])
        print(f"  Sketch vs raw separation improvement: {improvement:+.4f}")
        print("")

    overall = sketch_cluster["pass"] and mae < 15 and sketch_retrieval_p1 >= 0.8
    print(f"Overall: {'PASS' if overall else 'FAIL'}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    if args.json_out:
        payload = {
            "model_path": str(args.model),
            "mae": round(mae, 2),
            "sketch_cluster": sketch_cluster,
            "sketch_retrieval_p1": sketch_retrieval_p1,
            "raw_cluster": raw_cluster,
            "raw_retrieval_p1": raw_retrieval_p1,
            "per_doc": [
                {
                    "id": samples[i].manifesto_id,
                    "label": labels[i],
                    "family": families[i],
                    "rile_true": rile_trues[i],
                    "rile_pred": round(rile_preds[i], 2),
                    "error": round(errors[i], 2),
                }
                for i in range(len(samples))
            ],
            "overall_pass": overall,
        }
        out_path = args.json_out if args.json_out.is_absolute() else (PROJECT_ROOT / args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved JSON: %s", out_path)

    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
