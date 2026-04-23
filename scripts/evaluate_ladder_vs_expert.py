#!/usr/bin/env python3
"""Evaluate trained fg-ladder students against expert scores at the manifesto root.

Walks a manifesto_fg_ladder output directory, loads each trained embedding-ridge
proxy student, scores the root node of every tree in the corresponding labeled-tree
artifact, and reports Pearson r + MAE vs `expert_score_1_7` per (stage, leaf, split).

This fills the gap the ladder manifest leaves open: per-node train/val MAE is
recorded during training, but no per-manifesto root Pearson vs expert. That root
comparison is the central chart of the distillation experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ctreepo.distillation import load_labeled_trees  # noqa: E402
from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r  # noqa: E402
from src.training.embedding_proxy import EmbeddingRidgeProxyModel  # noqa: E402
from src.tree.labeled import LabeledNode, LabeledTree  # noqa: E402

LOGGER = logging.getLogger(__name__)


def hashing_embed(text: str, *, dim: int) -> List[float]:
    """Deterministic hashing embedding, bit-for-bit matching HashingEmbeddingClient."""
    vec = [0.0] * dim
    for token in str(text or "").lower().split():
        digest = hashlib.blake2b(
            token.encode("utf-8", errors="ignore"), digest_size=8
        ).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = -1.0 if (digest[4] & 1) else 1.0
        vec[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [float(v / norm) for v in vec]


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def _root_node(tree: LabeledTree) -> Optional[LabeledNode]:
    for level_ids in reversed(tree.levels or []):
        for node_id in level_ids:
            node = tree.get_node(str(node_id))
            if node is not None:
                return node
    return None


def _root_text(tree: LabeledTree) -> str:
    root = _root_node(tree)
    if root is None:
        return ""
    meta = root.metadata or {}
    return str(
        meta.get("teacher_summary")
        or meta.get("target_summary")
        or root.text
        or tree.document_text
        or ""
    )


def _evaluate_trees(
    *,
    model: EmbeddingRidgeProxyModel,
    trees: Sequence[LabeledTree],
    split: str,
    target_min: float,
    target_max: float,
) -> Dict[str, Any]:
    span = float(target_max - target_min)
    preds: List[float] = []
    experts: List[float] = []
    skipped: Dict[str, int] = {"missing_root": 0, "missing_text": 0, "missing_expert": 0}
    for tree in trees:
        tree_split = str((tree.metadata or {}).get("split") or "").lower()
        if split != "all" and tree_split != split.lower():
            continue
        text = _root_text(tree)
        if not text.strip():
            skipped["missing_text"] += 1
            continue
        expert = _safe_float((tree.metadata or {}).get("expert_score_1_7"))
        if expert is None:
            skipped["missing_expert"] += 1
            continue
        emb = hashing_embed(text, dim=model.embedding_dim)
        pred_01 = model.predict_from_embedding(emb)
        pred_scaled = target_min + span * float(pred_01)
        preds.append(pred_scaled)
        experts.append(expert)
    payload: Dict[str, Any] = {
        "split": split,
        "n_manifestos": len(preds),
        "skipped": skipped,
    }
    if len(preds) < 4:
        payload["pearson_r"] = None
        payload["mae_1_7"] = None
        return payload
    corr = compute_corpus_pearson_r(preds, experts).as_dict()
    payload.update({
        "pearson_r": corr.get("pearson_r"),
        "pearson_ci_low": corr.get("pearson_ci_low"),
        "pearson_ci_high": corr.get("pearson_ci_high"),
        "spearman_r": corr.get("spearman_r"),
        "mae_1_7": float(
            sum(abs(p - t) for p, t in zip(preds, experts)) / max(1, len(preds))
        ),
        "mean_prediction_1_7": float(sum(preds) / len(preds)),
        "mean_expert_1_7": float(sum(experts) / len(experts)),
    })
    return payload


def evaluate_fit(
    *,
    model_path: Path,
    trees_path: Path,
    target_min: float = 1.0,
    target_max: float = 7.0,
    splits: Sequence[str] = ("all", "train", "val", "test"),
) -> Dict[str, Any]:
    with open(model_path) as fh:
        model_data = json.load(fh)
    model = EmbeddingRidgeProxyModel.from_dict(model_data)
    trees = load_labeled_trees(trees_path)
    return {
        "model_path": str(model_path),
        "trees_path": str(trees_path),
        "model_id": model.model_id,
        "embedding_dim": model.embedding_dim,
        "embedding_model": model.embedding_model,
        "splits": {
            split: _evaluate_trees(
                model=model,
                trees=trees,
                split=split,
                target_min=target_min,
                target_max=target_max,
            )
            for split in splits
        },
    }


def _collect_stage_specs(
    manifest: Mapping[str, Any], *, ladder_dir: Path
) -> List[Dict[str, Any]]:
    """Enumerate (stage, leaf, fit_name, model_path, trees_path) for every fit."""
    specs: List[Dict[str, Any]] = []
    config = manifest.get("config") or {}

    def _rel(p: Optional[str]) -> Optional[Path]:
        if not p:
            return None
        path = Path(p)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    baseline_trees = _rel(config.get("f_baseline_labeled_trees"))
    f_doc_trees = _rel(config.get("f_doc_labeled_trees"))
    fg_grid_dir = _rel(config.get("fg_grid_dir"))

    for stage_key, trees_path in (("f", baseline_trees), ("f_doc", f_doc_trees)):
        stage_payload = manifest.get(stage_key) or {}
        fits = (stage_payload.get("fits") or {}) if isinstance(stage_payload, dict) else {}
        if not trees_path or not trees_path.exists():
            continue
        for fit_name, fit_payload in fits.items():
            if fit_name != "f_embedding_proxy":
                continue
            model_path = _rel(
                (fit_payload or {}).get("metadata", {})
                .get("distillation_result", {})
                .get("model_path")
            )
            if model_path is None:
                continue
            specs.append({
                "stage": stage_key,
                "leaf_count": None,
                "fit_name": fit_name,
                "model_path": model_path,
                "trees_path": trees_path,
            })

    leaves = manifest.get("leaves") or {}
    for leaf_key in sorted(leaves.keys()):
        leaf_payload = leaves[leaf_key] or {}
        leaf_count = int(leaf_payload.get("leaf_count") or int(leaf_key.split("_")[-1]))
        artifact = _rel(leaf_payload.get("artifact"))
        if artifact is None and fg_grid_dir is not None:
            artifact = fg_grid_dir / f"leaf_{leaf_count:03d}" / "labeled_trees.jsonl"
        if artifact is None or not artifact.exists():
            LOGGER.warning("Missing labeled_trees for %s; skipping", leaf_key)
            continue
        fgf_payload = leaf_payload.get("fgf") or {}
        fits = fgf_payload.get("fits") or {}
        for fit_name, fit_payload in fits.items():
            if fit_name != "f_embedding_proxy":
                continue
            model_path = _rel(
                (fit_payload or {}).get("metadata", {})
                .get("distillation_result", {})
                .get("model_path")
            )
            if model_path is None:
                continue
            specs.append({
                "stage": f"{leaf_key}_fgf",
                "leaf_count": leaf_count,
                "fit_name": fit_name,
                "model_path": model_path,
                "trees_path": artifact,
            })
    return specs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fg-ladder embedding-ridge students vs expert scores."
    )
    parser.add_argument("--ladder-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <ladder-dir>/eval_vs_expert.json",
    )
    parser.add_argument("--target-min", type=float, default=1.0)
    parser.add_argument("--target-max", type=float, default=7.0)
    parser.add_argument(
        "--splits",
        type=str,
        default="all,train,val,test",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ladder_dir = Path(args.ladder_dir)
    if not ladder_dir.exists():
        raise SystemExit(f"ladder dir not found: {ladder_dir}")

    manifest_path = ladder_dir / "fg_ladder_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    with open(manifest_path) as fh:
        manifest = json.load(fh)

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    specs = _collect_stage_specs(manifest, ladder_dir=ladder_dir)
    LOGGER.info("Found %d fits to evaluate", len(specs))

    results: Dict[str, Any] = {
        "ladder_dir": str(ladder_dir),
        "manifest": str(manifest_path),
        "dimension": manifest.get("dimension"),
        "target_min": float(args.target_min),
        "target_max": float(args.target_max),
        "stages": {},
    }

    for spec in specs:
        stage_key = spec["stage"]
        LOGGER.info(
            "Evaluating %s (leaf=%s, model=%s)",
            stage_key,
            spec["leaf_count"],
            spec["model_path"],
        )
        try:
            payload = evaluate_fit(
                model_path=spec["model_path"],
                trees_path=spec["trees_path"],
                target_min=float(args.target_min),
                target_max=float(args.target_max),
                splits=splits,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            LOGGER.exception("Evaluation failed for %s", stage_key)
            payload = {
                "model_path": str(spec["model_path"]),
                "trees_path": str(spec["trees_path"]),
                "error": str(exc),
            }
        payload.update({
            "leaf_count": spec["leaf_count"],
            "fit_name": spec["fit_name"],
        })
        results["stages"][stage_key] = payload

    output_path = args.output or (ladder_dir / "eval_vs_expert.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
