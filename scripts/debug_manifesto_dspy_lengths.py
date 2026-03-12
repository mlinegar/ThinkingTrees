#!/usr/bin/env python3
"""
Debug summary-length blowups when using DSPy leaf/merge modules.

This builds a single OPS tree for a specific manifesto ID using the provided
optimized modules, then reports per-level compression ratios and the largest
nodes. It is useful for diagnosing why merges can grow until they hit
max_tokens and get truncated.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.config.settings import load_settings
from src.config.dspy_config import configure_dspy, create_vllm_lm, create_vllm_lm_multi
from src.core.documents import DocumentSample
from src.core.strategy import DSPyStrategy
from src.preprocessing.chunker import chunk_for_ops
from src.tasks import get_task
from src.tasks.manifesto.data_loader import ManifestoDataset
from src.tasks.manifesto.rubrics import RILE_PRESERVATION_RUBRIC
from src.tree.builder import BuildConfig, TreeBuilder

logger = logging.getLogger(__name__)


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    p = max(0.0, min(100.0, float(p)))
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    idx = int(round((p / 100.0) * (len(xs) - 1)))
    idx = max(0, min(len(xs) - 1, idx))
    return float(xs[idx])


def _iter_nodes(root: Any) -> Iterable[Any]:
    if root is None:
        return
    queue = deque([root])
    while queue:
        node = queue.popleft()
        yield node
        left = getattr(node, "left_child", None)
        right = getattr(node, "right_child", None)
        if left is not None:
            queue.append(left)
        if right is not None:
            queue.append(right)


def _resolve_default_generation() -> Tuple[float, int]:
    settings = load_settings()
    summarizer_cfg = (
        (settings.get("generation", {}) or {}).get("summarizer", {}) if isinstance(settings, dict) else {}
    )
    temperature = float(summarizer_cfg.get("temperature", 0.5))
    max_tokens = int(summarizer_cfg.get("max_tokens", 4096))
    return temperature, max_tokens


def _load_module_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    root = Path(args.modules_dir) if args.modules_dir else Path("outputs/latest/manifesto_rile/trained_modules")
    leaf_path = Path(args.leaf_module_path) if args.leaf_module_path else root / "leaf_summarizer_final.json"
    merge_path = Path(args.merge_module_path) if args.merge_module_path else root / "merge_summarizer_final.json"
    if not leaf_path.exists():
        raise FileNotFoundError(f"Missing leaf module: {leaf_path}")
    if not merge_path.exists():
        raise FileNotFoundError(f"Missing merge module: {merge_path}")
    return leaf_path, merge_path


def _make_doc_sample(sample: Any) -> DocumentSample:
    return DocumentSample(
        doc_id=str(getattr(sample, "manifesto_id", "")),
        text=str(getattr(sample, "text", "")),
        reference_score=float(getattr(sample, "rile", 0.0)),
        metadata={
            "party_abbrev": getattr(sample, "party_abbrev", None),
            "country_code": getattr(sample, "country_code", None),
            "year": getattr(sample, "year", None),
            "rile_raw": getattr(sample, "rile", None),
        },
    )


def main() -> int:
    default_temp, default_max_tokens = _resolve_default_generation()

    parser = argparse.ArgumentParser(
        description="Debug per-level summary lengths for DSPy leaf/merge modules.",
    )
    parser.add_argument("--id", required=True, help="Manifesto ID (e.g., 51320_198306)")
    parser.add_argument("--chunk-size", type=int, default=8192, help="Max chunk chars (default: 8192)")
    parser.add_argument("--port", type=int, default=8000, help="Task model port (default: 8000)")
    parser.add_argument(
        "--ports",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of task model ports for load balancing (overrides --port)",
    )
    parser.add_argument(
        "--modules-dir",
        type=str,
        default=None,
        help="Directory containing leaf_summarizer_final.json and merge_summarizer_final.json (optional).",
    )
    parser.add_argument("--leaf-module-path", type=str, default=None, help="Explicit leaf module path (optional).")
    parser.add_argument("--merge-module-path", type=str, default=None, help="Explicit merge module path (optional).")
    parser.add_argument(
        "--dspy-temperature",
        type=float,
        default=default_temp,
        help="DSPy temperature for leaf/merge summaries (default: generation.summarizer.temperature)",
    )
    parser.add_argument(
        "--dspy-max-tokens",
        type=int,
        default=default_max_tokens,
        help="DSPy max_tokens for leaf/merge summaries (default: generation.summarizer.max_tokens)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for stats.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    leaf_path, merge_path = _load_module_paths(args)
    logger.info("Using leaf module: %s", leaf_path)
    logger.info("Using merge module: %s", merge_path)

    ports = list(dict.fromkeys(args.ports)) if args.ports else [int(args.port)]
    if len(ports) > 1:
        lm = create_vllm_lm_multi(
            ports=ports,
            temperature=float(args.dspy_temperature),
            max_tokens=int(args.dspy_max_tokens),
        )
    else:
        lm = create_vllm_lm(
            port=int(ports[0]),
            temperature=float(args.dspy_temperature),
            max_tokens=int(args.dspy_max_tokens),
        )
    configure_dspy(lm=lm)

    dataset = ManifestoDataset(countries=None, min_year=1900, require_text=True)
    raw_sample = dataset.get_sample(str(args.id))
    if raw_sample is None:
        raise SystemExit(f"Manifesto ID not found: {args.id}")
    sample = _make_doc_sample(raw_sample)

    chunks = chunk_for_ops(sample.text, max_chars=int(args.chunk_size), strategy="axis")
    logger.info(
        "Doc %s: chars=%d chunks=%d (chunk_size=%d)",
        sample.doc_id,
        len(sample.text),
        len(chunks),
        int(args.chunk_size),
    )

    task = get_task("manifesto_rile")
    leaf_module = task.create_summarizer()
    merge_module = task.create_merge_summarizer()
    leaf_module.load(str(leaf_path))
    merge_module.load(str(merge_path))

    strategy = DSPyStrategy(
        leaf_module=leaf_module,
        merge_module=merge_module,
        default_temperature=float(args.dspy_temperature),
        max_tokens=int(args.dspy_max_tokens),
    )
    builder = TreeBuilder(
        strategy=strategy,
        config=BuildConfig(max_chunk_chars=int(args.chunk_size)),
    )
    build_result = builder.build_sync(sample.text, rubric=RILE_PRESERVATION_RUBRIC)
    tree = build_result.tree

    nodes = list(_iter_nodes(tree.root))
    by_level: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for node in nodes:
        level = int(getattr(node, "level", 0) or 0)
        summary = getattr(node, "summary", "") or ""
        is_leaf = bool(getattr(node, "is_leaf", False))
        if is_leaf:
            raw_text = getattr(node, "raw_text_span", "") or ""
            input_chars = len(raw_text)
        else:
            left = getattr(node, "left_child", None)
            right = getattr(node, "right_child", None)
            input_chars = len(getattr(left, "summary", "") or "") + len(getattr(right, "summary", "") or "")
        output_chars = len(summary)
        ratio = (output_chars / input_chars) if input_chars else None
        by_level[level].append(
            {
                "is_leaf": is_leaf,
                "input_chars": input_chars,
                "output_chars": output_chars,
                "ratio": ratio,
                "preview": summary[:200],
            }
        )

    final_summary = tree.final_summary or ""
    logger.info(
        "Tree: leaves=%d height=%d nodes=%d final_summary_chars=%d",
        tree.leaf_count,
        tree.height,
        tree.node_count,
        len(final_summary),
    )

    report: Dict[str, Any] = {
        "manifesto_id": sample.doc_id,
        "chunk_size": int(args.chunk_size),
        "chunks": len(chunks),
        "tree": {
            "leaves": tree.leaf_count,
            "height": tree.height,
            "nodes": tree.node_count,
            "final_summary_chars": len(final_summary),
        },
        "generation": {
            "ports": ports,
            "dspy_temperature": float(args.dspy_temperature),
            "dspy_max_tokens": int(args.dspy_max_tokens),
        },
        "levels": {},
    }

    print("")
    print("Per-level summary stats (chars + compression ratios):")
    for level in sorted(by_level.keys()):
        rows = by_level[level]
        out_chars = [float(r["output_chars"]) for r in rows]
        in_chars = [float(r["input_chars"]) for r in rows]
        ratios = [float(r["ratio"]) for r in rows if r["ratio"] is not None]

        level_stats = {
            "n_nodes": len(rows),
            "out_chars_p50": _percentile(out_chars, 50),
            "out_chars_p95": _percentile(out_chars, 95),
            "out_chars_max": max(out_chars) if out_chars else None,
            "in_chars_p50": _percentile(in_chars, 50),
            "in_chars_p95": _percentile(in_chars, 95),
            "in_chars_max": max(in_chars) if in_chars else None,
            "ratio_p50": _percentile(ratios, 50),
            "ratio_p95": _percentile(ratios, 95),
            "ratio_max": max(ratios) if ratios else None,
        }
        report["levels"][str(level)] = level_stats

        print(
            f"  level {level:>2}: n={level_stats['n_nodes']:<3d} "
            f"out(p50/p95/max)={level_stats['out_chars_p50']:.0f}/{level_stats['out_chars_p95']:.0f}/{level_stats['out_chars_max']:.0f} "
            f"in(p50/p95/max)={level_stats['in_chars_p50']:.0f}/{level_stats['in_chars_p95']:.0f}/{level_stats['in_chars_max']:.0f} "
            f"ratio(p50/p95/max)={level_stats['ratio_p50']:.3f}/{level_stats['ratio_p95']:.3f}/{level_stats['ratio_max']:.3f}"
        )

    # Biggest nodes by output size
    flat = []
    for level, rows in by_level.items():
        for row in rows:
            flat.append((level, row))
    flat.sort(key=lambda item: item[1]["output_chars"], reverse=True)
    print("")
    print("Top 5 largest nodes by output_chars:")
    for level, row in flat[:5]:
        ratio = row["ratio"]
        ratio_str = f"{ratio:.3f}" if ratio is not None else "n/a"
        kind = "leaf" if row["is_leaf"] else "merge"
        print(
            f"  level {level:>2} ({kind}): out={row['output_chars']} in={row['input_chars']} ratio={ratio_str} preview={row['preview']!r}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote stats to %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

