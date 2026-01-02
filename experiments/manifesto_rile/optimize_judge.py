#!/usr/bin/env python3
"""
Optimize GenRM Judge Prompts - Tournament of Tournaments.

This script implements meta-optimization of the GenRM judge:
1. Creates a GenRMComparisonModule with optimizable DSPy prompts
2. Trains the judge on preference pairs with ground truth labels
3. Uses GEPA to optimize comparison prompts
4. Produces an improved judge for use in TournamentStrategy

The "tournament of tournaments" works by:
- Running tournaments to collect preference pairs (with oracle scores as ground truth)
- Training the judge to predict which summary better preserves information
- Iterating to improve judge discrimination quality

Usage:
    # Train judge from existing preference data
    python optimize_judge.py --preferences-file data/preferences.json --output-dir outputs/judge_training

    # Train judge with specific budget
    python optimize_judge.py --preferences-file data/preferences.json --budget medium --threads 8

    # Use trained judge in pipeline
    from optimize_judge import load_optimized_judge
    judge = load_optimized_judge("outputs/judge_training/optimized_judge.json")
    pipeline = ManifestoPipelineWithStrategy(judge=judge)
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import dspy

from src.config.dspy_config import configure_dspy
from src.ops_engine.training_framework.preference import PreferencePair
from src.training.judge_optimization import (
    JudgeOptimizer,
    JudgeOptimizationConfig,
    make_preference_labeler,
    load_optimized_judge as _load_optimized_judge,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Training Data Preparation
# =============================================================================

def load_preference_pairs(filepath: Path) -> List[PreferencePair]:
    """Load preference pairs from JSON file (list or PreferenceDataset format)."""
    with open(filepath) as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("pairs"), list):
        items = data["pairs"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("Unsupported preference file format")

    defaults = {
        "pair_id": "",
        "source_example_id": "",
        "original_text": "",
        "rubric": "",
        "ground_truth_score": None,
        "summary_a": "",
        "summary_b": "",
        "preferred": "tie",
        "reasoning": "",
        "confidence": 0.5,
        "law_type": "sufficiency",
        "score_estimate_a": None,
        "score_estimate_b": None,
        "oracle_error_a": None,
        "oracle_error_b": None,
        "judge_model": "",
    }
    pairs = []
    for item in items:
        data = {**defaults, **item}
        pairs.append(PreferencePair.from_dict(data))
    logger.info(f"Loaded {len(pairs)} preference pairs from {filepath}")
    return pairs


def save_optimized_judge(judge, output_path: Path) -> None:
    """Save optimized judge to file."""
    optimizer = JudgeOptimizer()
    optimizer.save(judge, output_path)


def load_optimized_judge(filepath: Path):
    """Load optimized judge from file."""
    return _load_optimized_judge(filepath)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimize GenRM Judge Prompts (Tournament of Tournaments)"
    )

    # Data options
    parser.add_argument(
        "--preferences-file",
        type=Path,
        required=True,
        help="Path to JSON file with preference pairs"
    )
    parser.add_argument(
        "--use-existing-preferences",
        action="store_true",
        help="Use existing 'preferred' field instead of deriving from oracle scores"
    )
    parser.add_argument(
        "--tie-margin",
        type=float,
        default=0.05,
        help="Normalized score difference below this is a tie (default: 0.05)"
    )
    parser.add_argument(
        "--scale-range",
        type=float,
        default=None,
        help="Optional scale range to normalize raw tie margin values"
    )
    parser.add_argument(
        "--labeler-metric",
        type=str,
        default=None,
        help="Metric base name for labeling (e.g., oracle_error, score_estimate)"
    )
    parser.add_argument(
        "--prefer-lower",
        action="store_true",
        help="Prefer lower values for the labeler metric"
    )

    # Training options
    parser.add_argument(
        "--budget",
        type=str,
        default="light",
        choices=["light", "medium", "heavy", "superheavy"],
        help="GEPA optimization budget (default: light)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of parallel evaluation threads (default: 4)"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port (default: 8000)"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/judge_training"),
        help="Output directory for results (default: outputs/judge_training)"
    )

    args = parser.parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Configure DSPy
    lm = dspy.LM(
        "openai/default",
        api_base=f"http://localhost:{args.port}/v1",
        api_key="EMPTY"
    )
    configure_dspy(lm=lm)
    logger.info(f"DSPy configured with vLLM on port {args.port}")

    # Load preference pairs
    pairs = load_preference_pairs(args.preferences_file)

    preference_labeler = None
    if args.labeler_metric:
        preference_labeler = make_preference_labeler(
            args.labeler_metric,
            prefer_lower=args.prefer_lower,
        )

    tie_margin = args.tie_margin
    if args.scale_range:
        tie_margin = args.tie_margin / args.scale_range
        logger.info(f"Normalized tie margin: {tie_margin:.4f} (scale_range={args.scale_range})")

    # Optimize judge
    config = JudgeOptimizationConfig(
        budget=args.budget,
        num_threads=args.threads,
        tie_margin=tie_margin,
        test_split=args.test_split,
        preference_labeler=preference_labeler,
    )
    optimizer = JudgeOptimizer(config=config)
    optimized_judge, results = optimizer.optimize(
        pairs,
        use_oracle_as_ground_truth=not args.use_existing_preferences,
    )

    # Save results
    results_path = output_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': {
                'preferences_file': str(args.preferences_file),
                'budget': args.budget,
                'threads': args.threads,
                'tie_margin': args.tie_margin,
                'test_split': args.test_split,
                'labeler_metric': args.labeler_metric,
                'prefer_lower': args.prefer_lower,
                'use_existing_preferences': args.use_existing_preferences,
            },
            'data': {
                'total_pairs': len(pairs),
            },
            'results': results,
        }, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Save optimized judge
    judge_path = output_dir / "optimized_judge.json"
    optimizer.save(optimized_judge, judge_path)

    # Summary
    baseline = results.get('baseline', {})
    optimized = results.get('optimized', {})
    logger.info("\n" + "="*60)
    logger.info("JUDGE OPTIMIZATION COMPLETE")
    logger.info("="*60)
    if baseline:
        logger.info(f"Baseline accuracy: {baseline.get('accuracy', 0.0):.3f}")
    if optimized:
        logger.info(f"Optimized accuracy: {optimized.get('accuracy', 0.0):.3f}")
    if baseline and optimized:
        logger.info(f"Improvement: {optimized.get('accuracy', 0.0) - baseline.get('accuracy', 0.0):+.3f}")
    if results.get('error'):
        logger.info(f"Optimization warning: {results['error']}")
    logger.info(f"Optimized judge saved to: {judge_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
