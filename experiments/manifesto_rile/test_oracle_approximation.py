#!/usr/bin/env python3
"""
Test oracle function approximation with RILE prediction results.

This script demonstrates using the oracle approximation to learn which
RILE predictions are likely to be wrong (violations) vs correct.
"""

import sys
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import dspy
from openai import OpenAI

from src.config.dspy_config import configure_dspy
from src.ops_engine.oracle_func_approximation import (
    OracleFuncTrainingCollector,
    OracleFuncTrainingExample,
    LearnedOracleFunc,
    OracleFuncConfig,
    ExampleLabel,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_rile_results(results_dir: Path) -> list:
    """Load RILE prediction results from a run directory."""
    results = []
    for f in sorted(results_dir.glob("iteration_*_results.json")):
        with open(f) as fp:
            data = json.load(fp)
            for item in data:
                item['iteration'] = int(f.stem.split('_')[1])
            results.extend(data)
    return results


def create_oracle_training_examples(
    rile_results: list,
    error_threshold_high: float = 30.0,  # Errors > this are "violations"
    error_threshold_low: float = 10.0,   # Errors < this are "good"
) -> OracleFuncTrainingCollector:
    """
    Create training examples from RILE prediction results.

    - High-error predictions are labeled as POSITIVE (true violations)
    - Low-error predictions are labeled as NEGATIVE (false positives / good predictions)
    """
    collector = OracleFuncTrainingCollector()

    for r in rile_results:
        if r.get('estimated_score') is None:
            continue

        error = abs(r['estimated_score'] - r['reference_score'])

        # Create example
        if error > error_threshold_high:
            # High error = violation (prediction was bad)
            example = OracleFuncTrainingExample(
                original_content=f"Party: {r['party_name']}, Year: {r['year']}, Country: {r['country']}",
                summary=r.get('final_summary', ''),
                rubric="RILE scoring: preserve left-right political indicators",
                check_type="rile_prediction",
                approx_discrepancy=error / 100.0,  # Normalize to 0-1
                label=ExampleLabel.POSITIVE,
                corrected_summary=f"Ground truth RILE: {r['reference_score']:.1f}",
                human_reasoning=f"Predicted {r['estimated_score']:.1f}, actual {r['reference_score']:.1f}, error {error:.1f}"
            )
            collector.add_example(example)

        elif error < error_threshold_low:
            # Low error = good prediction (not a violation)
            example = OracleFuncTrainingExample(
                original_content=f"Party: {r['party_name']}, Year: {r['year']}, Country: {r['country']}",
                summary=r.get('final_summary', ''),
                rubric="RILE scoring: preserve left-right political indicators",
                check_type="rile_prediction",
                approx_discrepancy=error / 100.0,
                label=ExampleLabel.NEGATIVE,
                human_reasoning=f"Predicted {r['estimated_score']:.1f}, actual {r['reference_score']:.1f}, error {error:.1f} - acceptable"
            )
            collector.add_example(example)

    return collector


def setup_dspy_lm(port: int = 8000):
    """Set up DSPy with local vLLM model."""
    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
    models = client.models.list()
    model_id = models.data[0].id if models.data else "default"

    lm = dspy.LM(
        model=f"openai/{model_id}",
        api_base=f"http://localhost:{port}/v1",
        api_key="EMPTY",
        max_tokens=4000,
    )
    configure_dspy(lm=lm)
    logger.info(f"Configured DSPy with model: {model_id}")
    return lm


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test oracle approximation with RILE results')
    parser.add_argument('--results-dir', type=Path,
                       default=project_root / 'data/results/manifesto_rile/overnight_20251216_175823',
                       help='Directory with RILE prediction results')
    parser.add_argument('--port', type=int, default=8000, help='vLLM port')
    parser.add_argument('--train', action='store_true', help='Train the oracle function')
    args = parser.parse_args()

    # Load results
    logger.info(f"Loading results from {args.results_dir}")
    results = load_rile_results(args.results_dir)
    logger.info(f"Loaded {len(results)} prediction results")

    # Create training examples
    collector = create_oracle_training_examples(results)
    stats = collector.get_statistics()

    print("\n" + "="*60)
    print("ORACLE TRAINING DATA STATISTICS")
    print("="*60)
    print(f"Total examples: {stats['total_examples']}")
    print(f"  Positive (high error): {stats['positive_examples']}")
    print(f"  Negative (low error): {stats['negative_examples']}")
    print(f"  Balance ratio: {stats['balance_ratio']:.2f}")
    print(f"  Avg discrepancy (positive): {stats['avg_discrepancy_positive']:.3f}")
    print(f"  Avg discrepancy (negative): {stats['avg_discrepancy_negative']:.3f}")

    # Show sample examples
    print("\n" + "-"*60)
    print("SAMPLE POSITIVE EXAMPLES (High Error):")
    for ex in collector.positive_examples[:3]:
        print(f"  - {ex.original_content}")
        print(f"    Reasoning: {ex.human_reasoning}")
        print()

    print("SAMPLE NEGATIVE EXAMPLES (Low Error):")
    for ex in collector.negative_examples[:3]:
        print(f"  - {ex.original_content}")
        print(f"    Reasoning: {ex.human_reasoning}")
        print()

    if args.train:
        # Set up DSPy
        logger.info("Setting up DSPy...")
        setup_dspy_lm(args.port)

        # Train oracle function
        logger.info("Training oracle function approximation...")
        config = OracleFuncConfig(
            min_training_examples=4,
            max_bootstrapped_demos=2,
            max_labeled_demos=4,
        )
        oracle = LearnedOracleFunc(config=config)
        oracle.training_collector = collector

        success = oracle.train()

        if success:
            print("\n" + "="*60)
            print("ORACLE FUNCTION TRAINED SUCCESSFULLY")
            print("="*60)
            print(f"Statistics: {oracle.get_statistics()}")

            # Save training data
            save_path = args.results_dir / "oracle_training_data.json"
            collector.save(save_path)
            print(f"Training data saved to: {save_path}")
        else:
            print("\nTraining failed!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
