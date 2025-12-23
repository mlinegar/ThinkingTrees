#!/usr/bin/env python3
"""
Generate Synthetic Training Data for OPS Summarization.

Uses a large oracle model (e.g., Nemotron-Ultra-253B) to generate
high-quality training data for smaller summarization models.

Workflow (inspired by Nemotron-3 training):
1. Load manifesto documents with ground truth RILE scores
2. Use large model to identify critical information for preservation
3. Generate reference summaries that preserve critical info
4. Validate quality and filter by score
5. Save as training dataset

Usage:
    # Generate data using Nemotron-253B oracle
    python generate_synthetic_data.py --oracle-port 8001 --output-dir data/synthetic

    # Use existing Qwen-235B instead
    python generate_synthetic_data.py --oracle-port 8000 --oracle-model qwen-235b

    # Limit to N documents for testing
    python generate_synthetic_data.py --max-documents 10

Requirements:
    - vLLM server running with the oracle model
    - Manifesto data available
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_python_path():
    """Add project root to Python path."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


setup_python_path()


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for OPS summarization"
    )

    # Oracle model configuration
    parser.add_argument(
        "--oracle-port", type=int, default=8001,
        help="Port for oracle model vLLM server (default: 8001)"
    )
    parser.add_argument(
        "--oracle-model", type=str, default="nemotron-253b-fp8",
        help="Oracle model name for metadata"
    )
    parser.add_argument(
        "--reasoning-mode", type=str, default="on",
        choices=["on", "off"],
        help="Enable/disable reasoning mode for Nemotron (default: on)"
    )

    # Data configuration
    parser.add_argument(
        "--max-documents", type=int, default=None,
        help="Maximum number of documents to process (default: all)"
    )
    parser.add_argument(
        "--min-quality-score", type=float, default=70.0,
        help="Minimum quality score to accept (default: 70.0)"
    )
    parser.add_argument(
        "--target-compression", type=float, default=0.2,
        help="Target compression ratio (default: 0.2)"
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Max retries per document for quality (default: 3)"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/synthetic"),
        help="Output directory for generated data"
    )
    parser.add_argument(
        "--save-intermediate", action="store_true",
        help="Save intermediate challenges and references"
    )

    # Generation parameters
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Temperature for generation (default: from config)"
    )
    parser.add_argument(
        "--top-p", type=float, default=None,
        help="Top-p for generation (default: from config)"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to settings.yaml (default: config/settings.yaml)"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Banner
    print()
    print("=" * 70)
    print("  SYNTHETIC DATA GENERATION FOR OPS SUMMARIZATION")
    print("=" * 70)
    print(f"  Oracle Model:      {args.oracle_model}")
    print(f"  Oracle Port:       {args.oracle_port}")
    print(f"  Reasoning Mode:    {args.reasoning_mode}")
    print(f"  Min Quality:       {args.min_quality_score}")
    print(f"  Target Compression:{args.target_compression}")
    print(f"  Output Directory:  {args.output_dir}")
    print("=" * 70)
    print()

    # Import dependencies after path setup
    import dspy
    from src.config.settings import load_settings
    from src.ops_engine.training_framework.synthetic_data import (
        SyntheticDataGenerator,
        ChallengeGenerator,
        ReferenceGenerator,
        SyntheticDataset,
    )

    settings = load_settings(args.config)
    generation_cfg = settings.get("generation", {})
    synthetic_cfg = generation_cfg.get("synthetic_data", {})

    if args.temperature is None:
        args.temperature = synthetic_cfg.get("temperature", 0.6)
    if args.top_p is None:
        args.top_p = synthetic_cfg.get("top_p", 0.95)
    max_tokens = synthetic_cfg.get("max_tokens", 4096)

    # Configure DSPy with oracle model
    logger.info(f"Configuring DSPy with oracle model on port {args.oracle_port}...")

    # System prompt for Nemotron reasoning mode
    if args.reasoning_mode == "on":
        system_prompt = "detailed thinking on"
    else:
        system_prompt = "detailed thinking off"

    oracle_lm = dspy.LM(
        model=f"openai/nvidia/{args.oracle_model}",
        api_base=f"http://localhost:{args.oracle_port}/v1",
        api_key="not-needed",
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )
    dspy.configure(lm=oracle_lm)

    logger.info("DSPy configured successfully")

    # Load manifesto data
    logger.info("Loading manifesto data...")
    from src.manifesto.data_loader import ManifestoDataLoader

    loader = ManifestoDataLoader()
    train_samples, val_samples, test_samples = loader.get_temporal_split()

    # Combine for synthetic data generation
    all_samples = train_samples + val_samples
    if args.max_documents:
        all_samples = all_samples[:args.max_documents]

    logger.info(f"Loaded {len(all_samples)} documents for processing")

    # Define the RILE rubric
    rile_rubric = """Preserve the political positioning (left-right stance) of the content.

Key information to preserve:
- Left-wing indicators: social welfare, equality, international cooperation, environmental protection
- Right-wing indicators: traditional values, free enterprise, national strength, law and order
- Overall political stance and intensity
- Key policy positions and their framing

The RILE score ranges from -100 (far left) to +100 (far right)."""

    # Create generators
    logger.info("Initializing synthetic data generators...")
    challenge_gen = ChallengeGenerator(use_cot=True)
    reference_gen = ReferenceGenerator(use_cot=True)

    generator = SyntheticDataGenerator(
        challenge_generator=challenge_gen,
        reference_generator=reference_gen,
        min_quality_score=args.min_quality_score,
        target_compression=args.target_compression,
        oracle_model_name=args.oracle_model,
    )

    # Generate synthetic data
    logger.info("Starting synthetic data generation...")
    print()

    successful_examples = []
    failed_count = 0

    for i, sample in enumerate(all_samples):
        logger.info(f"Processing document {i + 1}/{len(all_samples)}")

        # Get document text
        doc_text = sample.get('text', '') or sample.get('content', '')
        if not doc_text:
            logger.warning(f"Skipping document {i + 1}: no text content")
            failed_count += 1
            continue

        try:
            example = generator.generate_example(
                document=doc_text,
                rubric=rile_rubric,
                max_retries=args.max_retries,
            )

            if example is not None:
                successful_examples.append(example)
                logger.info(
                    f"  Generated example: score={example.preservation_score:.1f}, "
                    f"compression={example.compression_ratio:.2f}"
                )
            else:
                failed_count += 1
                logger.warning(f"  Failed to generate quality example")

        except Exception as e:
            failed_count += 1
            logger.error(f"  Error processing document: {e}")
            continue

        # Save intermediate results every 10 documents
        if (i + 1) % 10 == 0:
            stats = generator.get_statistics()
            logger.info(
                f"Progress: {stats['total_examples']}/{i + 1} successful, "
                f"avg_score={stats.get('avg_quality_score', 0):.1f}"
            )

    # Create dataset
    dataset = SyntheticDataset(successful_examples)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output_dir / f"synthetic_data_{timestamp}.json"
    dataset.save(output_file)

    # Save statistics
    stats = generator.get_statistics()
    stats["generation_config"] = {
        "oracle_model": args.oracle_model,
        "oracle_port": args.oracle_port,
        "reasoning_mode": args.reasoning_mode,
        "min_quality_score": args.min_quality_score,
        "target_compression": args.target_compression,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_retries": args.max_retries,
    }
    stats["total_documents"] = len(all_samples)
    stats["failed_count"] = failed_count

    stats_file = args.output_dir / f"generation_stats_{timestamp}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # Summary
    print()
    print("=" * 70)
    print("  GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Documents processed: {len(all_samples)}")
    print(f"  Successful examples: {stats['total_examples']}")
    print(f"  Failed:              {failed_count}")
    print(f"  Average quality:     {stats.get('avg_quality_score', 0):.1f}")
    print(f"  High quality (>=80): {stats.get('high_quality_count', 0)}")
    print()
    print(f"  Output file:         {output_file}")
    print(f"  Statistics:          {stats_file}")
    print("=" * 70)

    # Also save in training formats
    logger.info("Saving training format outputs...")

    # DSPy examples
    dspy_examples = dataset.to_dspy_examples()
    dspy_file = args.output_dir / f"dspy_examples_{timestamp}.json"
    with open(dspy_file, 'w') as f:
        json.dump([e.toDict() for e in dspy_examples], f, indent=2)

    # SFT format
    sft_data = dataset.to_sft_format()
    sft_file = args.output_dir / f"sft_data_{timestamp}.json"
    with open(sft_file, 'w') as f:
        json.dump(sft_data, f, indent=2)

    logger.info(f"Saved {len(dspy_examples)} DSPy examples to {dspy_file}")
    logger.info(f"Saved {len(sft_data)} SFT examples to {sft_file}")


if __name__ == "__main__":
    main()
