#!/usr/bin/env python3
"""Export Manifesto qsentence labeled trees as treepo preferences.

Input is a ThinkingTrees ``labeled_trees.jsonl`` artifact, usually produced by
``build_manifesto_qsentence_dspy_labeled_grid.py``. Output is the canonical
treepo ``PreferenceDataset`` plus supervised, DPO, reward, and GRPO projection
files.
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
    DEFAULT_QSENTENCE_FINETUNE_ADAPTERS,
    DEFAULT_QSENTENCE_LEARNING_ADAPTERS,
    parse_name_grid,
)
from src.ctreepo.treepo_bridge.manifesto_preferences import (
    build_manifesto_qsentence_preferences,
    build_manifesto_qsentence_tree_records,
    export_manifesto_qsentence_finetune_adapters,
)
from treepo.methods.preference import export_preference_records
from treepo.tree import write_tree_records_jsonl


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-trees", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("scores", "pairwise", "ranked"), default="ranked")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--finetune-adapters",
        default=",".join(DEFAULT_QSENTENCE_FINETUNE_ADAPTERS),
        help=(
            "Comma-separated treepo.finetune adapter exports to write. "
            "Use an empty value to disable adapter exports."
        ),
    )
    parser.add_argument(
        "--learning-adapters",
        default=",".join(DEFAULT_QSENTENCE_LEARNING_ADAPTERS),
        help=(
            "Comma-separated ThinkingTrees training adapter dry-runs to write. "
            "Default is thinkingtrees_dspy."
        ),
    )
    parser.add_argument(
        "--save-finetune-hf",
        action="store_true",
        help="Also save HF DatasetDict directories for adapter exports.",
    )
    args = parser.parse_args(argv)

    trees = load_labeled_trees(args.labeled_trees)
    tree_records = build_manifesto_qsentence_tree_records(trees)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tree_records_path = write_tree_records_jsonl(args.output_dir / "tree_records.jsonl", tree_records)
    preferences = build_manifesto_qsentence_preferences(
        trees,
        mode=str(args.mode),
        max_records=args.max_records,
    )
    artifacts = export_preference_records(preferences, args.output_dir)
    finetune_adapters = export_manifesto_qsentence_finetune_adapters(
        preferences,
        args.output_dir / "finetune_adapters",
        adapters=parse_name_grid(args.finetune_adapters),
        learning_adapters=parse_name_grid(args.learning_adapters),
        save_hf=bool(args.save_finetune_hf),
    )
    files = dict(artifacts.get("files") or {})
    files["tree_records"] = str(tree_records_path)
    result = {
        "labeled_trees": str(args.labeled_trees),
        "mode": str(args.mode),
        "n_trees": len(trees),
        **artifacts,
        "files": files,
        "finetune_adapters": finetune_adapters,
    }
    result_path = args.output_dir / "manifesto_qsentence_preferences_result.json"
    result["result_file"] = str(result_path)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    counts = result.get("counts", {}) or {}
    print(
        "status=success "
        f"trees={len(trees)} mode={args.mode} "
        f"units={counts.get('units', 0)} candidates={counts.get('candidates', 0)} "
        f"dpo={counts.get('dpo', 0)} reward={counts.get('reward', 0)} "
        f"grpo={counts.get('grpo', 0)} "
        f"finetune_adapters={finetune_adapters['summary']['n_adapters']} "
        f"learning_adapters={finetune_adapters['summary']['n_learning_adapters']} "
        f"output={result_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
