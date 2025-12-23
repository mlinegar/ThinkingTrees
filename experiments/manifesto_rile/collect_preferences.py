#!/usr/bin/env python3
"""
Collect Preference Data using GenRM for OPS Summarization.

Uses NVIDIA's Qwen3-Nemotron-235B-A22B-GenRM to compare candidate summaries
and build a preference dataset for training smaller models.

Workflow:
1. Load manifesto documents with ground truth RILE scores
2. Generate k candidate summaries using the small model (Nemotron-nano/Qwen-30B)
3. Use GenRM to compare all pairs and determine preferences
4. Save preference pairs for training

Usage:
    # Default: GenRM on port 8001, summarizer on port 8000
    python collect_preferences.py --output-dir data/preferences

    # Custom ports
    python collect_preferences.py --genrm-port 8001 --summarizer-port 8000

    # Limit documents for testing
    python collect_preferences.py --max-documents 10

Requirements:
    - vLLM server running with GenRM on port 8001
    - vLLM server running with summarizer (Qwen-30B) on port 8000
    - Manifesto data available
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

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
        description="Collect preference data using GenRM"
    )

    # Server configuration
    parser.add_argument(
        "--genrm-port", type=int, default=8001,
        help="Port for GenRM server (default: 8001)"
    )
    parser.add_argument(
        "--summarizer-port", type=int, default=8000,
        help="Port for summarizer model server (default: 8000)"
    )

    # Generation configuration
    parser.add_argument(
        "--k-candidates", type=int, default=4,
        help="Number of candidate summaries per document (default: 4)"
    )
    parser.add_argument(
        "--temperatures", type=float, nargs="+",
        default=None,
        help="Temperatures for diverse generation (default: from config)"
    )

    # Data configuration
    parser.add_argument(
        "--max-documents", type=int, default=None,
        help="Maximum documents to process (default: all)"
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Only use training split"
    )
    parser.add_argument(
        "--law-type", type=str, default="sufficiency",
        choices=["sufficiency", "idempotence", "merge"],
        help="OPS law type for comparison (default: sufficiency)"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/preferences"),
        help="Output directory for preference data"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to settings.yaml (default: config/settings.yaml)"
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Import dependencies
    import dspy
    from src.config.settings import load_settings
    from src.ops_engine.training_framework.genrm_preference import (
        GenRMJudge,
        GenRMPreferenceCollector,
    )
    from src.ops_engine.training_framework.preference import PreferenceDataset
    from src.manifesto.dspy_summarizer import LeafSummarizer

    settings = load_settings(args.config)
    generation_cfg = settings.get("generation", {})
    summarizer_cfg = generation_cfg.get("summarizer", {})
    judge_cfg = generation_cfg.get("genrm_judge", {})

    if args.temperatures is None:
        args.temperatures = summarizer_cfg.get(
            "candidate_temperatures", [0.3, 0.5, 0.7, 0.9]
        )
    summarizer_temperature = summarizer_cfg.get("temperature", 0.5)
    summarizer_max_tokens = summarizer_cfg.get("max_tokens", 2048)
    judge_temperature = judge_cfg.get("temperature", 0.6)
    judge_top_p = judge_cfg.get("top_p", 0.95)
    judge_max_tokens = judge_cfg.get("max_tokens", 2048)

    # Banner
    print()
    print("=" * 70)
    print("  PREFERENCE DATA COLLECTION WITH GenRM")
    print("=" * 70)
    print(f"  GenRM Port:        {args.genrm_port}")
    print(f"  Summarizer Port:   {args.summarizer_port}")
    print(f"  K Candidates:      {args.k_candidates}")
    print(f"  Temperatures:      {args.temperatures}")
    print(f"  Output Directory:  {args.output_dir}")
    print("=" * 70)
    print()

    # Configure summarizer LM
    logger.info(f"Configuring summarizer on port {args.summarizer_port}...")
    summarizer_lm = dspy.LM(
        model="openai/qwen-30b-thinking",
        api_base=f"http://localhost:{args.summarizer_port}/v1",
        api_key="not-needed",
        temperature=summarizer_temperature,  # Will be overridden per candidate
        max_tokens=summarizer_max_tokens,
    )
    dspy.configure(lm=summarizer_lm)

    # Create summarizer module
    summarizer = LeafSummarizer(use_cot=True)

    # Create GenRM judge
    logger.info(f"Configuring GenRM judge on port {args.genrm_port}...")
    judge = GenRMJudge(
        base_url=f"http://localhost:{args.genrm_port}/v1",
        model_name="nvidia/Qwen3-Nemotron-235B-A22B-GenRM",
        temperature=judge_temperature,
        top_p=judge_top_p,
        max_tokens=judge_max_tokens,
    )

    # Create collector
    collector = GenRMPreferenceCollector(
        summarizer=summarizer,
        judge=judge,
        k_candidates=args.k_candidates,
        temperatures=args.temperatures,
    )

    # Load manifesto data
    logger.info("Loading manifesto data...")
    from src.manifesto.data_loader import ManifestoDataLoader

    loader = ManifestoDataLoader()
    train_samples, val_samples, test_samples = loader.get_temporal_split()

    if args.train_only:
        samples = train_samples
    else:
        samples = train_samples + val_samples

    if args.max_documents:
        samples = samples[:args.max_documents]

    logger.info(f"Processing {len(samples)} documents")

    # RILE rubric
    rile_rubric = """Preserve the political positioning (left-right stance) of the content.

Key information to preserve:
- Left-wing indicators: social welfare, equality, international cooperation, environmental protection
- Right-wing indicators: traditional values, free enterprise, national strength, law and order
- Overall political stance and intensity
- Key policy positions and their framing"""

    # Collect preferences
    logger.info("Starting preference collection...")
    print()

    for i, sample in enumerate(samples):
        doc_id = sample.get('id', f'doc_{i}')
        doc_text = sample.get('text', '') or sample.get('content', '')
        ground_truth_rile = sample.get('rile', 0.0)

        if not doc_text:
            logger.warning(f"Skipping document {doc_id}: no text")
            continue

        logger.info(f"[{i+1}/{len(samples)}] Processing {doc_id}...")

        try:
            pairs = collector.collect_pairs_for_example(
                example_id=doc_id,
                original_text=doc_text[:8000],  # Truncate for context limits
                rubric=rile_rubric,
                ground_truth_score=ground_truth_rile,
                law_type=args.law_type,
            )
            logger.info(f"  Generated {len(pairs)} preference pairs")

        except Exception as e:
            logger.error(f"  Error: {e}")
            continue

        # Save intermediate results every 10 documents
        if (i + 1) % 10 == 0:
            stats = collector.get_statistics()
            logger.info(
                f"Progress: {stats['total_pairs']} pairs, "
                f"avg_confidence={stats['avg_confidence']:.2f}"
            )

    # Get final dataset
    dataset = collector.get_dataset()
    stats = collector.get_statistics()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output_dir / f"preferences_{timestamp}.json"
    dataset.save(output_file)

    # Save statistics
    stats["collection_config"] = {
        "genrm_port": args.genrm_port,
        "summarizer_port": args.summarizer_port,
        "k_candidates": args.k_candidates,
        "temperatures": args.temperatures,
        "max_documents": args.max_documents,
    }
    stats["total_documents"] = len(samples)

    stats_file = args.output_dir / f"collection_stats_{timestamp}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # Also save DPO format
    dpo_file = None
    if args.law_type == "sufficiency":
        dpo_data = dataset.to_dpo_format(law_type="sufficiency")
        dpo_file = args.output_dir / f"dpo_data_{timestamp}.json"
        with open(dpo_file, 'w') as f:
            json.dump(dpo_data, f, indent=2)
    else:
        logger.info("Skipping DPO export for non-sufficiency preferences.")

    # Summary
    print()
    print("=" * 70)
    print("  COLLECTION COMPLETE")
    print("=" * 70)
    print(f"  Documents processed: {len(samples)}")
    print(f"  Total pairs:         {stats['total_pairs']}")
    print(f"  Prefer A:            {stats['prefer_a']}")
    print(f"  Prefer B:            {stats['prefer_b']}")
    print(f"  Ties:                {stats['ties']}")
    print(f"  Avg confidence:      {stats['avg_confidence']:.2f}")
    print()
    print(f"  Preference file:     {output_file}")
    if dpo_file:
        print(f"  DPO format:          {dpo_file}")
    print(f"  Statistics:          {stats_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
