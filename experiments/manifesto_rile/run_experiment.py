#!/usr/bin/env python3
"""
Main experiment runner for Manifesto RILE scoring evaluation.

Usage:
    # Test with simple pipeline (no LLM)
    python run_experiment.py --simple --max-samples 5

    # Run full pipeline with LLM
    python run_experiment.py --task-port 8000 --max-samples 10

    # Full evaluation on test set
    python run_experiment.py --task-port 8000 --split test
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.manifesto.data_loader import create_pilot_dataset, ManifestoDataset
from src.manifesto.ops_pipeline import ManifestoOPSPipeline, SimplePipeline, PipelineConfig
from src.manifesto.evaluation import ManifestoEvaluator, save_results

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Run Manifesto RILE scoring experiment')

    # Data options
    parser.add_argument('--data-dir', type=Path, default=None,
                        help='Path to manifesto data directory')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'],
                        default='test', help='Data split to evaluate')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to process (for quick testing)')

    # Model options
    parser.add_argument('--task-port', type=int, default=8000,
                        help='Port for task model (30b)')
    parser.add_argument('--auditor-port', type=int, default=8001,
                        help='Port for auditor model (80b)')
    parser.add_argument('--use-llm-oracle', action='store_true',
                        help='Use LLM-based oracle for auditing')

    # Pipeline options
    parser.add_argument('--simple', action='store_true',
                        help='Use simple pipeline (no LLM) for testing')
    parser.add_argument('--no-baseline', action='store_true',
                        help='Skip baseline (full text) scoring')
    parser.add_argument('--chunk-size', type=int, default=2000,
                        help='Maximum chunk size in characters')
    parser.add_argument('--audit-budget', type=int, default=10,
                        help='Number of nodes to sample for auditing')

    # Output options
    parser.add_argument('--output-dir', type=Path,
                        default=project_root / 'data' / 'results' / 'manifesto_rile',
                        help='Directory for output files')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Name for this experiment run')

    return parser.parse_args()


def main():
    args = parse_args()

    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"run_{timestamp}"

    logger.info(f"Starting experiment: {args.experiment_name}")
    logger.info(f"Split: {args.split}, Max samples: {args.max_samples}")

    # Create output directory
    output_dir = args.output_dir / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load dataset
    logger.info("Loading pilot dataset...")
    pilot_data = create_pilot_dataset(
        data_dir=args.data_dir,
        output_dir=output_dir / 'splits'
    )

    # Get sample IDs for the selected split
    if args.split == 'train':
        sample_ids = pilot_data['train_ids']
    elif args.split == 'val':
        sample_ids = pilot_data['val_ids']
    elif args.split == 'test':
        sample_ids = pilot_data['test_ids']
    else:  # all
        sample_ids = pilot_data['train_ids'] + pilot_data['val_ids'] + pilot_data['test_ids']

    # Limit samples if requested
    if args.max_samples is not None:
        sample_ids = sample_ids[:args.max_samples]

    logger.info(f"Processing {len(sample_ids)} samples from {args.split} split")

    # Load samples
    dataset = pilot_data['dataset']
    samples = [dataset.get_sample(sid) for sid in sample_ids]
    samples = [s for s in samples if s is not None]

    if not samples:
        logger.error("No samples to process!")
        return 1

    # Initialize pipeline
    if args.simple:
        logger.info("Using simple pipeline (no LLM)")
        pipeline = SimplePipeline(max_chunk_chars=args.chunk_size)
    else:
        logger.info(f"Using LLM pipeline (task port: {args.task_port})")
        config = PipelineConfig(
            task_model_port=args.task_port,
            auditor_model_port=args.auditor_port,
            max_chunk_chars=args.chunk_size,
            audit_budget=args.audit_budget,
            run_baseline=not args.no_baseline,
            use_llm_oracle=args.use_llm_oracle
        )
        pipeline = ManifestoOPSPipeline(config)

    # Process samples
    logger.info("Processing manifestos...")
    results = []
    for i, sample in enumerate(samples):
        logger.info(f"[{i+1}/{len(samples)}] Processing {sample.manifesto_id} ({sample.party_name})")
        try:
            result = pipeline.process_manifesto(sample)
            results.append(result)

            if result.predicted_rile is not None:
                error = abs(result.predicted_rile - result.ground_truth_rile)
                logger.info(f"  Predicted: {result.predicted_rile:.1f}, "
                          f"Actual: {result.ground_truth_rile:.1f}, "
                          f"Error: {error:.1f}")
            if result.error:
                logger.warning(f"  Error: {result.error}")
        except Exception as e:
            logger.error(f"  Failed: {e}")

    # Evaluate results
    logger.info("Evaluating results...")
    evaluator = ManifestoEvaluator()
    metrics = evaluator.evaluate(results, split_name=args.split)

    # Generate report
    report = evaluator.generate_report(
        results,
        split_name=args.split,
        output_path=output_dir / 'report.txt'
    )
    print("\n" + report)

    # Save results
    save_results(results, output_dir / 'results.json')
    logger.info(f"Results saved to {output_dir}")

    # Save metrics
    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)

    logger.info("Experiment complete!")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Samples processed: {len(results)}")
    print(f"MAE:  {metrics.mae:.2f} RILE points")
    print(f"RMSE: {metrics.rmse:.2f} RILE points")
    print(f"Correlation: {metrics.correlation:.3f}")
    print(f"Within 10 points: {metrics.within_10_points:.1f}%")

    return 0


if __name__ == '__main__':
    sys.exit(main())
