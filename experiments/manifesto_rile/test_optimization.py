#!/usr/bin/env python3
"""
Quick test script for DSPy optimization debugging.

Loads saved results and tests the optimization pipeline without
re-processing manifestos.

Usage:
    python test_optimization.py --results-dir data/results/manifesto_rile/training_pipeline/run_XXXX
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))


def setup_dspy(port: int):
    """Configure DSPy with vLLM."""
    import dspy
    import requests

    try:
        resp = requests.get(f"http://localhost:{port}/v1/models")
        model_id = resp.json()["data"][0]["id"]
        logger.info(f"Detected model: {model_id}")
    except Exception as e:
        logger.error(f"Could not detect model: {e}")
        return False

    lm = dspy.LM(
        f"openai/{model_id}",
        api_base=f"http://localhost:{port}/v1",
        api_key="EMPTY",
    )
    dspy.configure(lm=lm)
    logger.info("DSPy configured")
    return True


def load_results(results_dir: Path):
    """Load saved manifesto results."""
    from src.manifesto.evaluation import load_results as load_manifesto_results

    train_file = results_dir / "train_results.json"
    val_file = results_dir / "val_results.json"

    train_results = load_manifesto_results(train_file) if train_file.exists() else []
    val_results = load_manifesto_results(val_file) if val_file.exists() else []

    return train_results, val_results


def create_training_examples(results, bin_size=10.0, error_high=20.0, error_low=10.0):
    """Create training examples from results."""
    from src.manifesto.training_integration import create_rile_training_collector

    collector = create_rile_training_collector(
        results,
        bin_size=bin_size,
        error_threshold_high=error_high,
        error_threshold_low=error_low,
    )

    trainset = collector.get_dspy_trainset(max_examples=50, balanced=True)

    logger.info(f"Created {len(trainset)} training examples")

    # Show a few examples
    for i, ex in enumerate(trainset[:3]):
        logger.info(f"  Example {i}: label={ex.label}, violation_type={ex.violation_type}")

    return collector, trainset


def test_classifier_forward(trainset, port):
    """Test that classifier forward() works."""
    from src.manifesto.training_integration import RILEOracleClassifier

    logger.info("Testing classifier forward()...")

    classifier = RILEOracleClassifier(bin_size=10.0)

    ex = trainset[0]
    logger.info(f"Input: original_content={ex.original_content[:50]}...")
    logger.info(f"Input: summary={ex.summary[:50]}...")
    logger.info(f"Expected label: {ex.label}")

    try:
        pred = classifier(
            original_content=ex.original_content,
            summary=ex.summary,
            rubric=ex.rubric,
        )
        logger.info(f"Prediction: label={pred.label}, confidence={pred.confidence}")
        logger.info(f"Reasoning: {pred.reasoning[:100]}...")
        return True
    except Exception as e:
        logger.error(f"Forward failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metric(trainset):
    """Test that the metric function works."""
    from src.ops_engine.training_framework import create_classification_metric
    from src.manifesto.training_integration import create_rile_label_space

    logger.info("Testing metric function...")

    label_space = create_rile_label_space(bin_size=10.0)
    metric = create_classification_metric(label_space, weighted=True)

    ex = trainset[0]

    # Create a mock prediction
    class MockPred:
        def __init__(self, label):
            self.label = label
            self.confidence = 0.8
            self.reasoning = "test"

    # Test exact match
    pred_exact = MockPred(ex.label)
    score_exact = metric(ex, pred_exact)
    logger.info(f"Exact match (label={ex.label}): score={score_exact}")

    # Test near match
    try:
        near_label = str(int(float(ex.label)) + 10)
        pred_near = MockPred(near_label)
        score_near = metric(ex, pred_near)
        logger.info(f"Near match (label={near_label}): score={score_near}")
    except:
        pass

    # Test far match
    pred_far = MockPred("50")
    score_far = metric(ex, pred_far)
    logger.info(f"Far match (label=50): score={score_far}")

    return True


def test_optimization(trainset, port):
    """Test DSPy optimization."""
    from src.manifesto.training_integration import RILEOracleClassifier, create_rile_label_space
    from src.ops_engine.training_framework import OracleOptimizer, create_classification_metric, OptimizationConfig

    logger.info("Testing optimization...")

    classifier = RILEOracleClassifier(bin_size=10.0)
    label_space = create_rile_label_space(bin_size=10.0)
    metric = create_classification_metric(label_space, weighted=True)

    config = OptimizationConfig(
        max_bootstrapped_demos=2,
        max_labeled_demos=4,
        max_rounds=1,
        num_candidate_programs=2,  # Minimal for testing
    )

    optimizer = OracleOptimizer(config)

    # Use small trainset for test
    small_trainset = trainset[:6]
    logger.info(f"Optimizing with {len(small_trainset)} examples...")

    try:
        compiled = optimizer.optimize(classifier, small_trainset, metric=metric)

        if optimizer.optimization_history:
            result = optimizer.optimization_history[-1]
            logger.info(f"Optimization result: {result.metric_before} -> {result.metric_after}")
        else:
            logger.warning("No optimization history")

        return True
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, help="Directory with saved results")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--step", type=str, default="all",
                       choices=["dspy", "metric", "forward", "optimize", "all"],
                       help="Which step to test")
    args = parser.parse_args()

    # Find latest results if not specified
    if args.results_dir is None:
        results_base = Path("data/results/manifesto_rile/training_pipeline")
        if results_base.exists():
            subdirs = [d for d in results_base.iterdir() if d.is_dir()]
            if subdirs:
                args.results_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
                logger.info(f"Using latest results: {args.results_dir}")

    if args.results_dir is None or not args.results_dir.exists():
        logger.error("No results directory found. Run the pipeline first or specify --results-dir")
        return 1

    # Always setup DSPy first
    if not setup_dspy(args.port):
        logger.error("DSPy setup failed")
        return 1

    # Test steps
    steps_to_run = ["metric", "forward", "optimize"] if args.step == "all" else [args.step]
    if args.step == "dspy":
        steps_to_run = []  # Already done

    trainset = None

    for step in steps_to_run:
        logger.info(f"\n{'='*50}")
        logger.info(f"STEP: {step.upper()}")
        logger.info(f"{'='*50}")

        if step == "metric":
            if trainset is None:
                train_results, _ = load_results(args.results_dir)
                if not train_results:
                    logger.error("No training results found")
                    return 1
                _, trainset = create_training_examples(train_results)

            if not test_metric(trainset):
                return 1

        elif step == "forward":
            if trainset is None:
                train_results, _ = load_results(args.results_dir)
                _, trainset = create_training_examples(train_results)

            if not test_classifier_forward(trainset, args.port):
                return 1

        elif step == "optimize":
            if trainset is None:
                train_results, _ = load_results(args.results_dir)
                _, trainset = create_training_examples(train_results)

            if not test_optimization(trainset, args.port):
                return 1

    logger.info("\n" + "="*50)
    logger.info("ALL TESTS PASSED")
    logger.info("="*50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
