#!/usr/bin/env python3
"""
ThinkingTrees: Oracle-Preserving Summarization

Entry point for orchestrating training, inference, and auditing workflows.
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Optional

import yaml

from src.core.data_models import OPSTree
from src.ops_engine.builder import (
    BuildConfig,
    BuildResult,
    OPSTreeBuilder,
    TruncatingSummarizer,
)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def set_rng_seed(seed: Optional[int]) -> None:
    """Seed Python's RNG for reproducibility when provided."""

    if seed is None:
        return

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        logging.getLogger(__name__).debug("NumPy not available; skipping NumPy seeding")


def load_config(config_path: Optional[Path], default_filename: str) -> Dict:
    """Load configuration from YAML file, falling back to a default location."""

    path = config_path or Path(__file__).parent / "config" / default_filename
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}

    logging.getLogger(__name__).warning("No configuration found at %s", path)
    return {}


def build_tree(
    input_path: Path,
    rubric: str = "",
    output_path: Optional[Path] = None,
    config: Optional[dict] = None,
) -> BuildResult:
    """
    Build an OPS tree from a document.

    Args:
        input_path: Path to input document
        rubric: Information preservation criteria
        output_path: Optional path to save tree
        config: Configuration dictionary

    Returns:
        BuildResult containing the tree and statistics
    """

    config = config or {}
    chunking_config = config.get("chunking", {})

    build_config = BuildConfig(
        max_chunk_chars=chunking_config.get("max_chars", 2000),
        min_chunk_chars=chunking_config.get("min_chars", 100),
        chunk_strategy=chunking_config.get("strategy", "sentence"),
        verbose=config.get("tree", {}).get("verbose", False),
    )

    summarizer = TruncatingSummarizer(max_length=500)
    builder = OPSTreeBuilder(summarizer=summarizer, config=build_config)
    result = builder.build_from_file(input_path, rubric)

    if output_path:
        save_tree(result.tree, output_path)

    return result


def save_tree(tree: OPSTree, output_path: Path) -> None:
    """Save tree to file (placeholder - implement serialization)."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("OPS Tree Summary\n")
        f.write("================\n\n")
        f.write(f"Height: {tree.height}\n")
        f.write(f"Nodes: {tree.node_count}\n")
        f.write(f"Leaves: {tree.leaf_count}\n")
        f.write(f"Rubric: {tree.rubric}\n\n")
        f.write("Final Summary:\n")
        f.write("--------------\n")
        f.write(tree.final_summary)


def print_tree_info(result: BuildResult) -> None:
    """Print tree information to console."""

    tree = result.tree
    print(f"\n{'=' * 50}")
    print("OPS Tree Built Successfully")
    print(f"{'=' * 50}")
    print(f"Chunks created:  {result.chunks_created}")
    print(f"Nodes created:   {result.nodes_created}")
    print(f"Tree height:     {tree.height}")
    print(f"Leaf count:      {tree.leaf_count}")
    print(f"{'=' * 50}")
    print("\nFinal Summary (first 500 chars):")
    print(f"{'-' * 50}")
    print(tree.final_summary[:500])
    if len(tree.final_summary) > 500:
        print("...")
    print(f"{'-' * 50}\n")


def run_training_mode(args: argparse.Namespace) -> None:
    """Handle training subcommand."""

    logger = logging.getLogger(__name__)
    config = load_config(args.config, "training.yaml")
    set_rng_seed(config.get("seed"))

    artifacts = config.get("artifacts", {})
    dataset = args.dataset or config.get("data", {}).get("preference_dataset")
    checkpoints_dir = Path(args.output or artifacts.get("checkpoint_dir", "experiments/checkpoints"))
    distilled_dir = Path(config.get("data", {}).get("distilled_summaries_dir", "data/distilled_summaries"))
    logs_dir = Path(artifacts.get("logs_dir", "experiments/logs"))

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    distilled_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training with model: %s", config.get("model", {}).get("path", "<unspecified>"))
    logger.info("Preference dataset: %s", dataset)
    logger.info("Checkpoints will be saved to: %s", checkpoints_dir)
    logger.info("Distilled summaries will be stored at: %s", distilled_dir)
    logger.info("Logs directory: %s", logs_dir)

    if config.get("evaluation", {}).get("run_eval", False):
        logger.info("Evaluation is enabled on split: %s", config.get("evaluation", {}).get("eval_split", "validation"))
    else:
        logger.info("Evaluation pass is disabled; enable evaluation.run_eval to score checkpoints.")

    metadata_path = checkpoints_dir / "training_run.yaml"
    with open(metadata_path, "w") as f:
        yaml.safe_dump(
            {
                "seed": config.get("seed"),
                "model": config.get("model", {}),
                "data": config.get("data", {}),
                "evaluation": config.get("evaluation", {}),
                "artifacts": {
                    "checkpoints": str(checkpoints_dir),
                    "distilled_summaries": str(distilled_dir),
                    "logs": str(logs_dir),
                },
            },
            f,
            sort_keys=False,
        )

    logger.info("Wrote training metadata to %s", metadata_path)
    logger.info("Training stub complete. Plug in trainer implementation to begin optimization.")


def run_inference_mode(args: argparse.Namespace) -> None:
    """Handle inference subcommand that builds an OPS tree."""

    logger = logging.getLogger(__name__)
    config = load_config(args.config, "inference.yaml")
    set_rng_seed(config.get("seed"))

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    logger.info("Building OPS tree from: %s", args.input)
    try:
        result = build_tree(
            input_path=args.input,
            rubric=args.rubric,
            output_path=args.output,
            config=config,
        )
        print_tree_info(result)

        if args.output:
            logger.info("Tree saved to: %s", args.output)

    except Exception as exc:
        logger.error("Failed to build tree: %s", exc)
        if args.verbose:
            raise
        sys.exit(1)


def run_audit_mode(args: argparse.Namespace) -> None:
    """Handle auditing subcommand."""

    logger = logging.getLogger(__name__)
    config = load_config(args.config, "audit.yaml")
    set_rng_seed(config.get("seed"))

    report_path = Path(args.output or config.get("artifacts", {}).get("report_path", "experiments/audit/report.yaml"))
    report_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Auditing tree: %s", args.input)
    logger.info(
        "Audit settings -> sample_budget: %s, discrepancy_threshold: %s",
        config.get("audit", {}).get("sample_budget"),
        config.get("audit", {}).get("discrepancy_threshold"),
    )

    summary = {
        "seed": config.get("seed"),
        "tree_path": str(args.input),
        "discrepancy_threshold": config.get("audit", {}).get("discrepancy_threshold"),
        "sample_budget": config.get("audit", {}).get("sample_budget"),
        "prioritize_high_levels": config.get("audit", {}).get("prioritize_high_levels", True),
        "evaluation": config.get("evaluation", {}),
    }

    with open(report_path, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    logger.info("Wrote audit stub report to %s", report_path)
    logger.info("Connect real auditing logic here to populate discrepancy findings.")


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI parser with subcommands."""

    parser = argparse.ArgumentParser(
        description="ThinkingTrees: Build, audit, and train OPS summarization trees",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Optimize and checkpoint summarization models")
    train_parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Training configuration YAML (defaults to config/training.yaml)",
    )
    train_parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional override for preference dataset path",
    )
    train_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional override for checkpoint output directory",
    )

    infer_parser = subparsers.add_parser("infer", help="Build an OPS tree for a document")
    infer_parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input document path",
    )
    infer_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for serialized tree",
    )
    infer_parser.add_argument(
        "--rubric",
        "-r",
        type=str,
        default="Preserve key information, entities, and conclusions.",
        help="Information preservation rubric",
    )
    infer_parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Inference configuration YAML (defaults to config/inference.yaml)",
    )

    audit_parser = subparsers.add_parser("audit", help="Audit previously built OPS trees")
    audit_parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to a serialized tree or tree metadata",
    )
    audit_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Destination for audit report",
    )
    audit_parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Audit configuration YAML (defaults to config/audit.yaml)",
    )

    return parser


def main() -> None:
    """Main entry point."""

    parser = build_parser()
    args = parser.parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)

    if args.command == "train":
        run_training_mode(args)
    elif args.command == "infer":
        run_inference_mode(args)
    elif args.command == "audit":
        run_audit_mode(args)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
