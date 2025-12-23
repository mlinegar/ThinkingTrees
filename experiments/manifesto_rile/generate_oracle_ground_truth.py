#!/usr/bin/env python3
"""
Generate Oracle Ground Truth Trees for Manifesto Documents.

This script:
1. Loads manifesto documents with ground truth RILE scores
2. Builds hierarchical tree structure (chunks → merge levels)
3. Scores all nodes with oracle model to create ground truth at every level
4. Saves ground truth trees for later preference pair collection

The resulting trees enable testing all three OPS laws:
- Sufficiency: Each chunk has oracle score
- Idempotence: Re-summarize and compare to original chunk oracle score
- Merge: Parent nodes have oracle scores for merged content

Usage:
    # Generate ground truth for small sample (server must be running)
    python generate_oracle_ground_truth.py \
        --oracle-port 8001 \
        --max-documents 10 \
        --output-dir data/ground_truth

    # Start server automatically and generate
    python generate_oracle_ground_truth.py \
        --start-server \
        --max-documents 10 \
        --output-dir data/ground_truth

    # Full dataset
    python generate_oracle_ground_truth.py \
        --oracle-port 8001 \
        --output-dir data/ground_truth
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_python_path():
    """Add project root to Python path."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


setup_python_path()


def start_oracle_server(port: int, model: str = "qwen3-nemotron-genrm") -> subprocess.Popen:
    """Start the oracle server as a background process."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    server_script = project_root / "scripts" / "start_oracle_server.sh"

    logger.info(f"Starting oracle server on port {port}...")

    # Start server in background
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["MODEL"] = model

    proc = subprocess.Popen(
        [str(server_script)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # Create new process group for cleanup
    )

    return proc


def wait_for_server(port: int, timeout: int = 300, interval: int = 5) -> bool:
    """Wait for the server to be ready."""
    url = f"http://localhost:{port}/v1/models"
    start_time = time.time()

    logger.info(f"Waiting for server on port {port} (timeout: {timeout}s)...")

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"Server ready after {time.time() - start_time:.1f}s")
                return True
        except requests.exceptions.RequestException:
            pass

        elapsed = int(time.time() - start_time)
        logger.info(f"  Still waiting... ({elapsed}s)")
        time.sleep(interval)

    logger.error(f"Server failed to start within {timeout}s")
    return False


def stop_server(proc: subprocess.Popen):
    """Stop the oracle server process."""
    if proc and proc.poll() is None:
        logger.info("Stopping oracle server...")
        try:
            # Kill the entire process group
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception as e:
            logger.warning(f"Error stopping server: {e}")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass


def build_tree_levels(chunks: list[str]) -> list[list[str]]:
    """
    Build hierarchical merge levels from chunks.

    Args:
        chunks: List of text chunks (level 0)

    Returns:
        List of levels, where levels[i] contains text for level i nodes
    """
    levels = [chunks]  # Level 0 = leaves

    current_level = chunks
    while len(current_level) > 1:
        next_level = []

        # Pair adjacent nodes and merge
        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                # Merge two nodes
                merged = f"{current_level[i]}\n\n{current_level[i+1]}"
            else:
                # Odd one out - promote as-is
                merged = current_level[i]
            next_level.append(merged)

        levels.append(next_level)
        current_level = next_level

    return levels


def main():
    parser = argparse.ArgumentParser(
        description="Generate oracle ground truth trees for manifesto documents"
    )

    # Oracle configuration
    parser.add_argument(
        "--oracle-port", type=int, default=8001,
        help="Port for oracle model server (default: 8001)"
    )
    parser.add_argument(
        "--oracle-model", type=str, default="openai/qwen-30b-thinking",
        help="Model name for oracle LM"
    )

    # Chunking configuration
    parser.add_argument(
        "--chunk-size", type=int, default=4000,
        help="Maximum characters per chunk (default: 4000)"
    )

    # Data configuration
    parser.add_argument(
        "--max-documents", type=int, default=None,
        help="Maximum documents to process (default: all)"
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Only process training split"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/ground_truth"),
        help="Output directory for ground truth trees"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to settings.yaml (default: config/settings.yaml)"
    )

    # Server management
    parser.add_argument(
        "--start-server", action="store_true",
        help="Start oracle server automatically before generation"
    )
    parser.add_argument(
        "--server-model", type=str, default="qwen3-nemotron-genrm-gguf",
        help="Model profile to use when starting server (default: qwen3-nemotron-genrm-gguf)"
    )
    parser.add_argument(
        "--server-timeout", type=int, default=300,
        help="Timeout in seconds for server startup (default: 300)"
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Banner
    print()
    print("=" * 70)
    print("  ORACLE GROUND TRUTH TREE GENERATION")
    print("=" * 70)
    print(f"  Oracle Port:       {args.oracle_port}")
    print(f"  Oracle Model:      {args.oracle_model}")
    print(f"  Chunk Size:        {args.chunk_size}")
    print(f"  Output Directory:  {args.output_dir}")
    if args.start_server:
        print(f"  Start Server:      Yes ({args.server_model})")
    print("=" * 70)
    print()

    # Start server if requested
    server_proc = None
    if args.start_server:
        server_proc = start_oracle_server(args.oracle_port, args.server_model)
        if not wait_for_server(args.oracle_port, timeout=args.server_timeout):
            stop_server(server_proc)
            sys.exit(1)

    try:
        _run_generation(args)
    finally:
        if server_proc:
            stop_server(server_proc)


def _run_generation(args):
    """Run the actual generation logic."""
    # Import dependencies (avoid circular imports)
    import dspy
    from src.config.settings import load_settings

    # Import oracle_ground_truth directly to avoid circular import
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "oracle_ground_truth",
        Path(__file__).parent.parent.parent / "src/ops_engine/training_framework/oracle_ground_truth.py"
    )
    oracle_gt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle_gt_module)
    ChunkGroundTruth = oracle_gt_module.ChunkGroundTruth
    ManifestoGroundTruthTree = oracle_gt_module.ManifestoGroundTruthTree
    GroundTruthDataset = oracle_gt_module.GroundTruthDataset

    from src.manifesto.data_loader import ManifestoDataset
    from src.manifesto.batched_pipeline import chunk_text
    from src.manifesto.position_oracle import create_rile_scorer

    settings = load_settings(args.config)
    generation_cfg = settings.get("generation", {})
    ground_truth_cfg = generation_cfg.get("ground_truth", {})
    oracle_temperature = ground_truth_cfg.get("temperature", 0.3)
    oracle_max_tokens = ground_truth_cfg.get("max_tokens", 2048)

    # Configure oracle LM
    logger.info(f"Configuring oracle on port {args.oracle_port}...")
    oracle_lm = dspy.LM(
        model=args.oracle_model,
        api_base=f"http://localhost:{args.oracle_port}/v1",
        api_key="not-needed",
        temperature=oracle_temperature,
        max_tokens=oracle_max_tokens,
    )
    dspy.configure(lm=oracle_lm)

    # Create oracle scorer
    rile_scorer = create_rile_scorer()

    def oracle_predict(text: str) -> dict:
        """Score text with oracle, returning score and reasoning."""
        try:
            result = rile_scorer.value_extractor(text)
            score = float(result)

            # Try to get reasoning if available
            reasoning = ""
            left_indicators = ""
            right_indicators = ""

            # The RILEScorer might have reasoning in the result
            if hasattr(result, 'reasoning'):
                reasoning = str(result.reasoning)
            if hasattr(result, 'left_indicators'):
                left_indicators = str(result.left_indicators)
            if hasattr(result, 'right_indicators'):
                right_indicators = str(result.right_indicators)

            return {
                "score": score,
                "reasoning": reasoning,
                "left_indicators": left_indicators,
                "right_indicators": right_indicators,
            }
        except Exception as exc:
            logger.warning(f"Oracle scoring failed: {exc}")
            return {
                "score": 0.0,
                "reasoning": f"Error: {exc}",
                "left_indicators": "",
                "right_indicators": "",
            }

    # Load manifesto data
    logger.info("Loading manifesto data...")
    dataset = ManifestoDataset()
    train_ids, val_ids, test_ids = dataset.create_temporal_split()

    if args.train_only:
        sample_ids = train_ids
    else:
        sample_ids = train_ids + val_ids

    if args.max_documents:
        sample_ids = sample_ids[:args.max_documents]

    # Convert IDs to samples
    samples = list(dataset.get_split_samples(sample_ids))

    logger.info(f"Processing {len(samples)} documents")

    # Create dataset to store all trees
    dataset = GroundTruthDataset()

    # Process each document
    for i, sample in enumerate(samples):
        # ManifestoSample is a dataclass with attributes
        manifesto_id = sample.manifesto_id
        doc_text = sample.text
        ground_truth_rile = sample.rile

        if not doc_text:
            logger.warning(f"Skipping document {manifesto_id}: no text")
            continue

        logger.info(f"[{i+1}/{len(samples)}] Processing {manifesto_id}...")

        # Build tree structure
        logger.info(f"  Chunking text ({len(doc_text)} chars)...")
        chunks = chunk_text(doc_text, args.chunk_size)
        logger.info(f"  Created {len(chunks)} chunks")

        logger.info(f"  Building merge levels...")
        levels = build_tree_levels(chunks)
        logger.info(f"  Built {len(levels)} levels")

        # Create ground truth tree
        tree = ManifestoGroundTruthTree(
            manifesto_id=manifesto_id,
            document_text=doc_text,
            document_rile=ground_truth_rile,
            oracle_model=args.oracle_model,
        )

        # Score all nodes in the tree
        total_nodes = sum(len(level) for level in levels)
        node_count = 0

        for level_idx, level_texts in enumerate(levels):
            logger.info(f"  Scoring level {level_idx} ({len(level_texts)} nodes)...")

            for node_idx, text in enumerate(level_texts):
                node_count += 1
                chunk_id = f"{manifesto_id}_L{level_idx}_N{node_idx}"

                # Score with oracle
                oracle_result = oracle_predict(text)

                # Determine children (for merge nodes)
                left_child_id = None
                right_child_id = None
                if level_idx > 0:
                    # This is a merge node - find its children in previous level
                    child_idx_base = node_idx * 2
                    left_child_id = f"{manifesto_id}_L{level_idx-1}_N{child_idx_base}"
                    if child_idx_base + 1 < len(levels[level_idx - 1]):
                        right_child_id = f"{manifesto_id}_L{level_idx-1}_N{child_idx_base+1}"

                # Create ground truth node
                node = ChunkGroundTruth(
                    chunk_id=chunk_id,
                    manifesto_id=manifesto_id,
                    level=level_idx,
                    text=text,
                    rile_score=oracle_result["score"],
                    reasoning=oracle_result["reasoning"],
                    left_indicators=oracle_result["left_indicators"],
                    right_indicators=oracle_result["right_indicators"],
                    confidence=1.0,
                    left_child_id=left_child_id,
                    right_child_id=right_child_id,
                )

                tree.add_node(node)

                if node_count % 10 == 0:
                    logger.info(f"    Scored {node_count}/{total_nodes} nodes...")

        logger.info(f"  Completed: {tree.num_chunks} nodes, {tree.num_levels} levels")

        # Add to dataset
        dataset.add_tree(tree)

        # Save individual tree
        tree_file = args.output_dir / f"{manifesto_id}_ground_truth.json"
        tree.save(tree_file)

        # Print statistics
        stats = tree.get_statistics()
        logger.info(f"  RILE range: [{stats['rile_min']:.1f}, {stats['rile_max']:.1f}], mean: {stats['rile_mean']:.1f}")

    # Save dataset index
    logger.info("Saving dataset index...")
    dataset.save(args.output_dir)

    # Final statistics
    dataset_stats = dataset.get_statistics()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file = args.output_dir / f"generation_stats_{timestamp}.json"

    stats = {
        "created_at": datetime.now().isoformat(),
        "config": {
            "oracle_port": args.oracle_port,
            "oracle_model": args.oracle_model,
            "chunk_size": args.chunk_size,
            "max_documents": args.max_documents,
            "train_only": args.train_only,
        },
        "dataset_stats": dataset_stats,
    }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # Summary
    print()
    print("=" * 70)
    print("  GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Documents processed:    {len(dataset)}")
    print(f"  Total chunks:           {dataset_stats['total_chunks']}")
    print(f"  Total leaf nodes:       {dataset_stats['total_leaves']}")
    print(f"  Total merge nodes:      {dataset_stats['total_merge_nodes']}")
    print(f"  Avg chunks per tree:    {dataset_stats['avg_chunks_per_tree']:.1f}")
    print(f"  Avg levels:             {dataset_stats['avg_levels']:.1f}")
    print()
    print(f"  Output directory:       {args.output_dir}")
    print(f"  Statistics file:        {stats_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
