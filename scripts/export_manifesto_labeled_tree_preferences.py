#!/usr/bin/env python3
"""Export Manifesto labeled trees as treepo fine-tuning views.

Input is any ThinkingTrees ``labeled_trees.jsonl`` artifact. The exporter can
use the qsentence-specific prompts or the generic Manifesto f/g labeled-tree
projection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ctreepo.distillation import load_labeled_trees
from src.ctreepo.treepo_bridge.manifesto_finetune import (
    default_finetune_adapters,
    default_learning_adapters,
    export_manifesto_finetune_bundle_from_args,
    parse_name_grid,
    resolve_manifesto_finetune_kind,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-trees", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("auto", "generic", "qsentence"), default="auto")
    parser.add_argument("--mode", choices=("scores", "pairwise", "ranked"), default="ranked")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--leaf-unit-type", default="leaf")
    parser.add_argument(
        "--finetune-adapters",
        default=None,
        help=(
            "Comma-separated treepo.finetune adapter exports. "
            "Defaults depend on --kind; use an empty value to disable exports."
        ),
    )
    parser.add_argument(
        "--learning-adapters",
        default=None,
        help=(
            "Comma-separated ThinkingTrees dry-run training adapters. "
            "Defaults depend on --kind; use an empty value to disable them."
        ),
    )
    parser.add_argument("--save-finetune-hf", action="store_true")
    args = parser.parse_args(argv)

    trees = load_labeled_trees(args.labeled_trees)
    kind = resolve_manifesto_finetune_kind(str(args.kind), trees)
    args.finetune_mode = str(args.mode)
    args.finetune_max_records = args.max_records
    args.finetune_adapters = ",".join(
        parse_name_grid(args.finetune_adapters, default=default_finetune_adapters(kind))
    )
    args.learning_adapters = ",".join(
        parse_name_grid(args.learning_adapters, default=default_learning_adapters(kind))
    )
    result = export_manifesto_finetune_bundle_from_args(
        args=args,
        trees=trees,
        output_dir=args.output_dir,
        kind=kind,
        leaf_unit_type=str(args.leaf_unit_type),
        respect_enabled=False,
    )
    if result is None:
        raise RuntimeError("Manifesto fine-tune export unexpectedly returned no bundle")

    result = {
        "labeled_trees": str(args.labeled_trees),
        "requested_kind": str(args.kind),
        **result,
    }
    result_path = args.output_dir / "manifesto_labeled_tree_preferences_result.json"
    result["result_file"] = str(result_path)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    summary = result.get("summary", {}) or {}
    counts = result.get("counts", {}) or {}
    print(
        "status=success "
        f"kind={result.get('bundle_kind')} trees={summary.get('n_trees', len(trees))} "
        f"units={counts.get('dataset', counts.get('units', 0))} "
        f"candidates={counts.get('candidates', 0)} "
        f"output={result_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
