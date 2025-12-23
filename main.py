#!/usr/bin/env python3
"""
ThinkingTrees: Oracle-Preserving Summarization

Entry point for building and analyzing OPS trees from documents.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import yaml

from src.core.data_models import OPSTree
from src.ops_engine.builder import (
    OPSTreeBuilder, BuildConfig, BuildResult,
    IdentitySummarizer, TruncatingSummarizer
)
from src.preprocessing.chunker import DocumentChunker


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "settings.yaml"

    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)

    return {}


def build_tree(
    input_path: Path,
    rubric: str = "",
    output_path: Optional[Path] = None,
    config: Optional[dict] = None,
    teacher_guided: Optional[bool] = None,
    student_only: Optional[bool] = None,
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
    chunking_config = config.get('chunking', {})
    tree_config = config.get('tree', {})

    default_build_config = BuildConfig()

    teacher_guided = (
        teacher_guided
        if teacher_guided is not None
        else tree_config.get('teacher_guided', default_build_config.teacher_guided)
    )

    student_only = (
        student_only
        if student_only is not None
        else tree_config.get('student_only', default_build_config.student_only)
    )

    # Create build configuration
    build_config = BuildConfig(
        max_chunk_chars=chunking_config.get('max_chars', 2000),
        min_chunk_chars=chunking_config.get('min_chars', 100),
        chunk_strategy=chunking_config.get('strategy', 'sentence'),
        teacher_guided=teacher_guided,
        student_only=student_only,
        verbose=tree_config.get('verbose', False)
    )

    # For now, use a simple summarizer
    # TODO: Integrate with LLM client
    summarizer = TruncatingSummarizer(max_length=500)

    # Build tree
    builder = OPSTreeBuilder(summarizer=summarizer, config=build_config)
    result = builder.build_from_file(input_path, rubric)

    # Save if output path specified
    if output_path:
        save_tree(result.tree, output_path)

    return result


def save_tree(tree: OPSTree, output_path: Path) -> None:
    """Save tree to file (placeholder - implement serialization)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # For now, save basic info as text
    with open(output_path, 'w') as f:
        f.write(f"OPS Tree Summary\n")
        f.write(f"================\n\n")
        f.write(f"Height: {tree.height}\n")
        f.write(f"Nodes: {tree.node_count}\n")
        f.write(f"Leaves: {tree.leaf_count}\n")
        f.write(f"Rubric: {tree.rubric}\n\n")
        f.write(f"Final Summary:\n")
        f.write(f"--------------\n")
        f.write(tree.final_summary)


def print_tree_info(result: BuildResult) -> None:
    """Print tree information to console."""
    tree = result.tree
    print(f"\n{'='*50}")
    print(f"OPS Tree Built Successfully")
    print(f"{'='*50}")
    print(f"Chunks created:  {result.chunks_created}")
    print(f"Nodes created:   {result.nodes_created}")
    print(f"Tree height:     {tree.height}")
    print(f"Leaf count:      {tree.leaf_count}")
    print(f"{'='*50}")
    print(f"\nFinal Summary (first 500 chars):")
    print(f"{'-'*50}")
    print(tree.final_summary[:500])
    if len(tree.final_summary) > 500:
        print("...")
    print(f"{'-'*50}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ThinkingTrees: Build OPS summarization trees"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input document path"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output path for tree"
    )
    parser.add_argument(
        "--rubric", "-r",
        type=str,
        default="Preserve key information, entities, and conclusions.",
        help="Information preservation rubric"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Configuration file path",
    )
    parser.add_argument(
        "--teacher-guided",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable teacher-guided merges using a preference scorer",
    )
    parser.add_argument(
        "--student-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Disable teacher guidance and rely solely on student summarization",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Setup
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    # Load config
    config = load_config(args.config)
    if args.verbose:
        config.setdefault('tree', {})['verbose'] = True

    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Build tree
    logger.info(f"Building OPS tree from: {args.input}")
    try:
        result = build_tree(
            input_path=args.input,
            rubric=args.rubric,
            output_path=args.output,
            config=config,
            teacher_guided=args.teacher_guided,
            student_only=args.student_only,
        )
        print_tree_info(result)

        if args.output:
            logger.info(f"Tree saved to: {args.output}")

    except Exception as e:
        logger.error(f"Failed to build tree: {e}")
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
