#!/usr/bin/env python3
"""
ThinkingTrees: Oracle-Preserving Summarization

Entry point for building and analyzing OPS trees from documents.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import yaml

from src.core.data_models import Tree
from src.ops_engine.builder import (
    BuildConfig, BuildResult, build,
    IdentitySummarizer
)
from src.preprocessing.chunker import DocumentChunker
from src.core.llm_client import LLMConfig, LLMClient, create_summarizer, create_client
from src.config.logging import setup_logging


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
    llm_port: Optional[int] = None,
) -> BuildResult:
    """
    Build an OPS tree from a document.

    Args:
        input_path: Path to input document
        rubric: Information preservation criteria
        output_path: Optional path to save tree
        config: Configuration dictionary
        llm_port: Optional port for vLLM server (uses identity summarizer if not available)

    Returns:
        BuildResult containing the tree and statistics
    """
    config = config or {}
    chunking_config = config.get('chunking', {})
    logger = logging.getLogger(__name__)

    # Create build configuration
    build_config = BuildConfig(
        max_chunk_chars=chunking_config.get('max_chars', 2000),
        min_chunk_chars=chunking_config.get('min_chars', 100),
        chunk_strategy=chunking_config.get('strategy', 'sentence'),
        verbose=config.get('tree', {}).get('verbose', False)
    )

    # Try to create LLM-based summarizer if port specified
    summarizer = None
    if llm_port is not None:
        try:
            llm_config = LLMConfig.vllm(port=llm_port)
            client = create_client(llm_config)
            # Test connection
            import requests
            response = requests.get(f"http://localhost:{llm_port}/v1/models", timeout=5)
            if response.ok:
                summarizer = create_summarizer(client)
                logger.info(f"Using LLM summarizer (port {llm_port})")
        except Exception as e:
            logger.warning(f"Could not connect to LLM server on port {llm_port}: {e}")

    # Fallback to identity summarizer
    if summarizer is None:
        logger.info("Using identity summarizer (no LLM server)")
        summarizer = IdentitySummarizer()

    # Read file content
    text = Path(input_path).read_text(encoding='utf-8')

    # Build tree using the build() helper (handles sync summarizer wrapping)
    tree = build(
        text=text,
        rubric=rubric,
        summarizer=summarizer,
        max_chars=build_config.max_chunk_chars
    )

    # Wrap in BuildResult for consistent API
    result = BuildResult(
        tree=tree,
        chunks_created=tree.leaf_count,
        nodes_created=tree.node_count,
        levels_created=tree.height + 1,
        errors=[],
        preferences=[],
    )

    # Save if output path specified
    if output_path:
        save_tree(result.tree, output_path)

    return result


def save_tree(tree: Tree, output_path: Path) -> None:
    """
    Save tree to file.

    Supports two formats based on file extension:
    - .json: Full tree serialization (can be loaded back)
    - .txt or other: Human-readable summary
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == '.json':
        # Full serialization
        tree.save(output_path)
    else:
        # Human-readable summary
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
    print(f"\nFinal Summary:")
    print(f"{'-'*50}")
    print(tree.final_summary)
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
        help="Configuration file path"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="vLLM server port for LLM-based summarization (omit to use test summarizer)"
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
            llm_port=args.port
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
