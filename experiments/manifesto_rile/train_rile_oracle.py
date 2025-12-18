#!/usr/bin/env python3
"""
Train RILE Oracle Classifier from Manifesto Results.

This script demonstrates the full training workflow:
1. Load existing manifesto results (or process new ones)
2. Extract training data using the training framework
3. Train a RILE oracle classifier with DSPy optimization
4. Evaluate the trained classifier
5. Use the classifier for OPS law verification

Usage:
    # Train from existing results
    python train_rile_oracle.py --results-dir data/results/manifesto_rile/overnight_run_1

    # Process and train
    python train_rile_oracle.py --process --samples 10 --port 8000
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import dspy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_existing_results(results_dir: Path):
    """Load ManifestoResults from a results directory."""
    from src.manifesto.evaluation import load_results

    # Find the latest results file
    result_files = list(results_dir.glob("**/results.json")) + \
                   list(results_dir.glob("**/iteration_*_results.json"))

    if not result_files:
        raise FileNotFoundError(f"No results files found in {results_dir}")

    # Use the most recent
    latest = max(result_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading results from {latest}")

    return load_results(latest)


def setup_dspy(port: int):
    """Configure DSPy with vLLM."""
    lm = dspy.LM(
        "openai/default",
        api_base=f"http://localhost:{port}/v1",
        api_key="EMPTY"
    )
    dspy.configure(lm=lm)
    logger.info(f"DSPy configured with vLLM on port {port}")


def main():
    parser = argparse.ArgumentParser(description="Train RILE Oracle Classifier")

    # Data source options
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory containing existing manifesto results"
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Process new manifestos instead of loading results"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples to process (if --process)"
    )

    # Model options
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port"
    )

    # Training options
    parser.add_argument(
        "--bin-size",
        type=float,
        default=10.0,
        help="RILE bin size for discretization"
    )
    parser.add_argument(
        "--error-threshold-high",
        type=float,
        default=20.0,
        help="Error above this = violation (training positive)"
    )
    parser.add_argument(
        "--error-threshold-low",
        type=float,
        default=10.0,
        help="Error below this = good (training negative)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=50,
        help="Maximum training examples"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/rile_oracle_training"),
        help="Output directory for training results"
    )

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Configure DSPy
    setup_dspy(args.port)

    # Get manifesto results
    if args.process:
        # Process new manifestos
        logger.info("Processing new manifestos...")
        from src.manifesto.data_loader import ManifestoDataset
        from src.manifesto.training_integration import (
            create_rile_training_pipeline,
            PipelineConfig,
        )

        # Load dataset
        dataset = ManifestoDataset(
            countries=[51, 41],  # UK, Germany for pilot
            min_year=1990,
            require_text=True,
        )

        # Get samples
        sample_ids = dataset.get_all_ids()[:args.samples]
        samples = [dataset.get_sample(sid) for sid in sample_ids]
        samples = [s for s in samples if s is not None]

        logger.info(f"Processing {len(samples)} manifestos...")

        # Create trainable pipeline
        config = PipelineConfig(
            task_model_port=args.port,
            run_baseline=False,  # Skip baseline for speed
        )
        pipeline = create_rile_training_pipeline(pipeline_config=config)

        # Process
        for i, sample in enumerate(samples):
            logger.info(f"Processing {i+1}/{len(samples)}: {sample.manifesto_id}")
            try:
                pipeline.process_manifesto(sample)
            except Exception as e:
                logger.error(f"Error processing {sample.manifesto_id}: {e}")

        results = pipeline.get_results()
        collector = pipeline.get_training_collector()

    else:
        # Load existing results
        if args.results_dir is None:
            # Try to find the most recent results
            results_base = Path("data/results/manifesto_rile")
            if results_base.exists():
                subdirs = [d for d in results_base.iterdir() if d.is_dir()]
                if subdirs:
                    args.results_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
                    logger.info(f"Using most recent results: {args.results_dir}")

        if args.results_dir is None:
            raise ValueError("No results directory specified and none found")

        results = load_existing_results(args.results_dir)
        logger.info(f"Loaded {len(results)} results")

        # Create training collector from results
        from src.manifesto.training_integration import (
            create_rile_training_collector,
        )

        collector = create_rile_training_collector(
            results,
            bin_size=args.bin_size,
            error_threshold_high=args.error_threshold_high,
            error_threshold_low=args.error_threshold_low,
        )

    # Get training statistics
    stats = collector.get_statistics()
    logger.info(f"Training data statistics: {stats}")

    # Get training examples
    trainset = collector.get_dspy_trainset(
        max_examples=args.max_examples,
        balanced=True,
    )
    logger.info(f"Training set size: {len(trainset)}")

    if len(trainset) < 4:
        logger.error("Not enough training examples (need at least 4)")
        return

    # Save training data
    training_data = []
    for ex in trainset:
        training_data.append({
            'original_content': ex.original_content[:200],
            'summary': ex.summary[:200] if hasattr(ex, 'summary') else '',
            'label': str(getattr(ex, 'label', '')),
        })

    with open(output_dir / "training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)

    # Train the classifier
    logger.info("Training RILE oracle classifier...")
    from src.manifesto.training_integration import train_rile_oracle
    from src.ops_engine.training_framework import FrameworkConfig, OptimizationConfig

    config = FrameworkConfig()
    config.optimization.max_examples = args.max_examples
    config.optimization.save_checkpoints = True
    config.optimization.checkpoint_dir = output_dir / "checkpoints"

    classifier, eval_result = train_rile_oracle(
        collector,
        bin_size=args.bin_size,
        config=config,
    )

    # Save results
    output_data = {
        'timestamp': timestamp,
        'args': vars(args),
        'training_stats': stats,
        'trainset_size': len(trainset),
    }

    if eval_result:
        output_data['evaluation'] = eval_result.to_dict()
        logger.info(f"Evaluation results:")
        logger.info(f"  Accuracy: {eval_result.accuracy:.3f}")
        logger.info(f"  Weighted Accuracy: {eval_result.weighted_accuracy:.3f}")
        logger.info(f"  MAE: {eval_result.mae:.2f}")

    with open(output_dir / "training_results.json", "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Training complete! Results saved to {output_dir}")

    # Demo: Use the trained classifier
    logger.info("\n--- Demo: Using trained classifier ---")
    demo_texts = [
        "We will expand public healthcare, increase minimum wage, and strengthen workers' rights through union support.",
        "We support free enterprise, lower taxes for businesses, and reducing government regulation of the market.",
    ]

    for text in demo_texts:
        try:
            rile, confidence, reasoning = classifier.predict_rile(text)
            logger.info(f"\nText: {text[:60]}...")
            logger.info(f"  Predicted RILE: {rile:.1f}")
            logger.info(f"  Confidence: {confidence:.2f}")
        except Exception as e:
            logger.error(f"Prediction error: {e}")


if __name__ == "__main__":
    main()
