#!/usr/bin/env python3
"""
Unified Training Pipeline - Full Learning Flow.

This script implements the complete training loop:

Phase 1: Initial Tournament with Default Judge
    - Uses DSPyStrategy + TournamentStrategy
    - Collects preferences with oracle scores (ground truth)

Phase 2: Judge Optimization (Tournament of Tournaments)
    - Creates GenRMComparisonModule with optimizable DSPy prompts
    - Trains judge on collected preferences
    - GEPA optimizes comparison prompts

Phase 3: Summary Module Optimization
    - Uses improved judge in TournamentStrategy
    - Collects better preferences (with improved discrimination)
    - Optimizes LeafSummarizer/MergeSummarizer via GEPA

Phase 4: Production Ready
    - Extracts learned prompts from optimized DSPy modules
    - Can switch to BatchedStrategy for efficient inference

Usage:
    # Full training pipeline
    python unified_training_pipeline.py --port 8000 --genrm-port 8001 --samples 50

    # Skip judge optimization (use default GenRM)
    python unified_training_pipeline.py --port 8000 --genrm-port 8001 --skip-judge-optimization

    # Resume from Phase 2
    python unified_training_pipeline.py --resume-from phase2 --preferences-file outputs/phase1/preferences.json
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import dspy

from src.config.dspy_config import configure_dspy
from src.manifesto.data_loader import create_pilot_dataset
from src.tasks.manifesto import RILE_PRESERVATION_RUBRIC, RILE_TASK_CONTEXT, RILE_SCALE
from src.ops_engine.training_framework.genrm_preference import GenRMJudge
from src.ops_engine.training_framework.genrm_dspy import GenRMComparisonModule
from src.ops_engine.training_framework.preference import PreferencePair
from src.training.judge_optimization import JudgeOptimizer, JudgeOptimizationConfig

# Import our components from canonical location
from src.tasks.manifesto import (
    ManifestoPipelineWithStrategy,
    ManifestoScorer,
    create_training_examples,
    rile_metric,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Phase 1: Initial Data Collection
# =============================================================================

def phase1_collect_preferences(
    samples: list,
    genrm_url: str,
    rubric: str = RILE_PRESERVATION_RUBRIC,
    tournament_k: int = 4,
    output_dir: Optional[Path] = None,
) -> List[PreferencePair]:
    """
    Phase 1: Collect initial preferences using default GenRM judge.

    Processes samples through ManifestoPipelineWithStrategy with TournamentStrategy,
    collecting preferences as a free byproduct of tournament selection.

    Args:
        samples: List of manifesto samples with ground truth RILE scores
        genrm_url: URL of GenRM vLLM server
        rubric: Information preservation rubric
        tournament_k: Number of candidates per tournament
        output_dir: Optional directory to save intermediate results

    Returns:
        List of PreferencePair from tournament selection
    """
    logger.info("="*60)
    logger.info("PHASE 1: Initial Preference Collection")
    logger.info("="*60)

    # Create GenRM judge
    judge = GenRMJudge(base_url=genrm_url)
    logger.info(f"Created GenRMJudge at {genrm_url}")

    # Create pipeline with tournament strategy
    pipeline = ManifestoPipelineWithStrategy(
        judge=judge,
        tournament_k=tournament_k,
    )
    logger.info(f"Created pipeline with tournament_k={tournament_k}")

    all_preferences = []
    results = []

    for i, sample in enumerate(samples):
        logger.info(f"Processing sample {i+1}/{len(samples)}: {sample.party_name} ({sample.country_name})")

        try:
            # Process through pipeline
            prediction = pipeline(text=sample.text, rubric=rubric)

            # Get preferences from tournament
            prefs = pipeline.get_preferences()
            logger.info(f"  Collected {len(prefs)} preference pairs")

            # Augment preferences with oracle score (for ground truth)
            for pref in prefs:
                pref.ground_truth_score = sample.rile  # RILE score as oracle

            all_preferences.extend(prefs)

            # Track results
            results.append({
                'manifesto_id': sample.manifesto_id,
                'party_name': sample.party_name,
                'country': sample.country_name,
                'reference_score': sample.rile,
                'estimated_score': prediction.rile_score,
                'error': abs(prediction.rile_score - sample.rile),
                'preferences_collected': len(prefs),
            })

            # Reset preferences for next document
            pipeline.reset_preferences()

        except Exception as e:
            logger.error(f"  Error processing sample: {e}")
            results.append({
                'manifesto_id': sample.manifesto_id,
                'error': str(e),
            })

    logger.info(f"Phase 1 complete: {len(all_preferences)} total preferences collected")

    # Save intermediate results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save preferences
        prefs_data = [_pref_to_dict(p) for p in all_preferences]
        with open(output_dir / "phase1_preferences.json", 'w') as f:
            json.dump(prefs_data, f, indent=2)

        # Save results
        with open(output_dir / "phase1_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved Phase 1 results to {output_dir}")

    return all_preferences


def _pref_to_dict(pref: PreferencePair) -> dict:
    """Convert PreferencePair to dictionary."""
    return {
        'pair_id': pref.pair_id,
        'source_example_id': pref.source_example_id,
        'original_text': pref.original_text,
        'rubric': pref.rubric,
        'ground_truth_score': pref.ground_truth_score,
        'summary_a': pref.summary_a,
        'summary_b': pref.summary_b,
        'preferred': pref.preferred,
        'reasoning': pref.reasoning,
        'confidence': pref.confidence,
        'law_type': pref.law_type,
        'score_estimate_a': pref.score_estimate_a,
        'score_estimate_b': pref.score_estimate_b,
        'judge_model': pref.judge_model,
    }


# =============================================================================
# Phase 2: Judge Optimization
# =============================================================================

def phase2_optimize_judge(
    preferences: List[PreferencePair],
    budget: str = 'light',
    num_threads: int = 4,
    test_split: float = 0.2,
    output_dir: Optional[Path] = None,
) -> GenRMComparisonModule:
    """
    Phase 2: Optimize the GenRM judge prompts.

    Creates an optimizable GenRMComparisonModule and trains it on the collected
    preferences using GEPA optimization.

    Args:
        preferences: List of PreferencePair from Phase 1
        budget: GEPA optimization budget
        num_threads: Parallel evaluation threads
        test_split: Fraction of data for testing
        output_dir: Optional directory for results

    Returns:
        Optimized GenRMComparisonModule
    """
    logger.info("="*60)
    logger.info("PHASE 2: Judge Optimization (Tournament of Tournaments)")
    logger.info("="*60)

    # Enrich preferences with oracle errors (normalized)
    try:
        from src.tasks.manifesto.task import ManifestoTask
        oracle_predict = ManifestoTask().create_oracle_scorer()
        scale_range = RILE_SCALE.range
        for pref in preferences:
            if pref.ground_truth_score is None:
                continue
            score_a = oracle_predict(pref.summary_a)
            score_b = oracle_predict(pref.summary_b)
            pref.score_estimate_a = score_a
            pref.score_estimate_b = score_b
            error_a = abs(score_a - pref.ground_truth_score)
            error_b = abs(score_b - pref.ground_truth_score)
            if scale_range > 0:
                pref.oracle_error_a = min(1.0, max(0.0, error_a / scale_range))
                pref.oracle_error_b = min(1.0, max(0.0, error_b / scale_range))
            else:
                pref.oracle_error_a = error_a
                pref.oracle_error_b = error_b
    except Exception as e:
        logger.warning(f"Oracle enrichment skipped: {e}")

    config = JudgeOptimizationConfig(
        budget=budget,
        num_threads=num_threads,
        tie_margin=min(0.05, 5.0 / RILE_SCALE.range),
        test_split=test_split,
    )
    optimizer = JudgeOptimizer(config=config)
    optimized_judge, results = optimizer.optimize(
        preferences,
        use_oracle_as_ground_truth=True,
    )

    baseline_results = results.get('baseline', {})
    optimized_results = results.get('optimized', {})

    if baseline_results and optimized_results:
        logger.info(f"Baseline accuracy: {baseline_results.get('accuracy', 0.0):.3f}")
        logger.info(f"Optimized accuracy: {optimized_results.get('accuracy', 0.0):.3f}")
        logger.info(f"Improvement: {optimized_results.get('accuracy', 0.0) - baseline_results.get('accuracy', 0.0):+.3f}")
    elif results.get('error'):
        logger.warning(f"Judge optimization error: {results['error']}")

    if output_dir:
        with open(output_dir / "phase2_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        optimizer.save(optimized_judge, output_dir / "optimized_judge.json")

    return optimized_judge


# =============================================================================
# Phase 3: Summary Module Optimization
# =============================================================================

def phase3_optimize_summarizer(
    samples: list,
    optimized_judge: GenRMComparisonModule,
    rubric: str = RILE_PRESERVATION_RUBRIC,
    budget: str = 'light',
    num_threads: int = 4,
    output_dir: Optional[Path] = None,
) -> ManifestoPipelineWithStrategy:
    """
    Phase 3: Optimize the summarization modules using the improved judge.

    Uses the optimized judge in TournamentStrategy to collect better preferences,
    then optimizes LeafSummarizer and MergeSummarizer via GEPA.

    Args:
        samples: Training samples with ground truth
        optimized_judge: Optimized judge from Phase 2
        rubric: Information preservation rubric
        budget: GEPA optimization budget
        num_threads: Parallel evaluation threads
        output_dir: Optional directory for results

    Returns:
        Optimized ManifestoPipelineWithStrategy
    """
    logger.info("="*60)
    logger.info("PHASE 3: Summary Module Optimization")
    logger.info("="*60)

    # Create pipeline with optimized judge
    pipeline = ManifestoPipelineWithStrategy(
        judge=optimized_judge,
        tournament_k=4,
    )

    # Create training examples (imported at top of file)
    trainset = create_training_examples(samples)

    logger.info(f"Training set size: {len(trainset)}")

    # Optimize the pipeline
    optimizer = dspy.GEPA(
        metric=rile_metric,
        auto=budget,
        num_threads=num_threads,
    )

    optimized_pipeline = optimizer.compile(
        pipeline,
        trainset=trainset,
    )

    logger.info("Phase 3 optimization complete")

    # Save optimized pipeline
    if output_dir:
        optimized_pipeline.save(str(output_dir / "optimized_pipeline.json"))
        logger.info(f"Saved optimized pipeline to {output_dir}")

    return optimized_pipeline


# =============================================================================
# Phase 4: Evaluate and Export
# =============================================================================

def phase4_evaluate_and_export(
    optimized_pipeline: ManifestoPipelineWithStrategy,
    test_samples: list,
    rubric: str = RILE_PRESERVATION_RUBRIC,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Phase 4: Evaluate optimized pipeline and export for production.

    Args:
        optimized_pipeline: Optimized pipeline from Phase 3
        test_samples: Held-out test samples
        rubric: Information preservation rubric
        output_dir: Optional directory for results

    Returns:
        Evaluation results dictionary
    """
    logger.info("="*60)
    logger.info("PHASE 4: Evaluation and Export")
    logger.info("="*60)

    results = []
    total_error = 0.0

    for i, sample in enumerate(test_samples):
        logger.info(f"Testing {i+1}/{len(test_samples)}: {sample.party_name}")

        try:
            prediction = optimized_pipeline(text=sample.text, rubric=rubric)
            error = abs(prediction.rile_score - sample.rile)

            results.append({
                'manifesto_id': sample.manifesto_id,
                'party_name': sample.party_name,
                'ground_truth': sample.rile,
                'predicted': prediction.rile_score,
                'error': error,
            })
            total_error += error

        except Exception as e:
            logger.error(f"  Error: {e}")
            results.append({
                'manifesto_id': sample.manifesto_id,
                'error': str(e),
            })

    avg_error = total_error / len(results) if results else 0.0
    logger.info(f"Average RILE error: {avg_error:.2f}")

    # Extract learned prompts for BatchedStrategy
    learned_prompts = {}
    try:
        if hasattr(optimized_pipeline.leaf_module, '_inner'):
            inner = optimized_pipeline.leaf_module._inner
            if hasattr(inner, 'summarize') and hasattr(inner.summarize, 'extended_signature'):
                learned_prompts['summarize'] = inner.summarize.extended_signature.instructions
        if hasattr(optimized_pipeline.merge_module, '_inner'):
            inner = optimized_pipeline.merge_module._inner
            if hasattr(inner, 'merge') and hasattr(inner.merge, 'extended_signature'):
                learned_prompts['merge'] = inner.merge.extended_signature.instructions
    except Exception as e:
        logger.warning(f"Could not extract learned prompts: {e}")

    summary = {
        'avg_error': avg_error,
        'num_samples': len(results),
        'results': results,
        'learned_prompts': learned_prompts,
    }

    if output_dir:
        with open(output_dir / "phase4_results.json", 'w') as f:
            json.dump(summary, f, indent=2)

        # Save learned prompts separately for easy access
        if learned_prompts:
            with open(output_dir / "learned_prompts.json", 'w') as f:
                json.dump(learned_prompts, f, indent=2)
            logger.info("Exported learned prompts for BatchedStrategy")

    return summary


# =============================================================================
# Main Pipeline
# =============================================================================

def run_unified_pipeline(
    port: int = 8000,
    genrm_port: int = 8001,
    num_samples: int = 50,
    budget: str = 'light',
    num_threads: int = 4,
    skip_judge_optimization: bool = False,
    output_dir: Path = Path("outputs/unified_training"),
) -> Dict[str, Any]:
    """
    Run the complete unified training pipeline.

    Args:
        port: vLLM server port for main LM
        genrm_port: vLLM server port for GenRM
        num_samples: Number of manifesto samples to use
        budget: GEPA optimization budget
        num_threads: Parallel threads
        skip_judge_optimization: If True, skip Phase 2
        output_dir: Output directory for all results

    Returns:
        Summary of all phases
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting unified training pipeline")
    logger.info(f"Output directory: {run_dir}")

    # Configure DSPy
    lm = dspy.LM(
        "openai/default",
        api_base=f"http://localhost:{port}/v1",
        api_key="EMPTY"
    )
    configure_dspy(lm=lm)

    genrm_url = f"http://localhost:{genrm_port}/v1"

    # Load data
    logger.info(f"Loading {num_samples} manifesto samples...")
    samples = create_pilot_dataset(max_samples=num_samples)

    # Split train/test
    train_samples = samples[:int(len(samples) * 0.8)]
    test_samples = samples[int(len(samples) * 0.8):]
    logger.info(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    # Save config
    config = {
        'timestamp': timestamp,
        'port': port,
        'genrm_port': genrm_port,
        'num_samples': num_samples,
        'budget': budget,
        'num_threads': num_threads,
        'skip_judge_optimization': skip_judge_optimization,
        'train_size': len(train_samples),
        'test_size': len(test_samples),
    }
    with open(run_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Phase 1: Collect preferences
    preferences = phase1_collect_preferences(
        samples=train_samples,
        genrm_url=genrm_url,
        output_dir=run_dir,
    )

    # Phase 2: Optimize judge (optional)
    if skip_judge_optimization:
        logger.info("Skipping Phase 2 (judge optimization)")
        optimized_judge = GenRMComparisonModule(use_dspy_predictor=False)
    else:
        optimized_judge = phase2_optimize_judge(
            preferences=preferences,
            budget=budget,
            num_threads=num_threads,
            output_dir=run_dir,
        )

    # Phase 3: Optimize summarizer
    optimized_pipeline = phase3_optimize_summarizer(
        samples=train_samples,
        optimized_judge=optimized_judge,
        budget=budget,
        num_threads=num_threads,
        output_dir=run_dir,
    )

    # Phase 4: Evaluate and export
    final_results = phase4_evaluate_and_export(
        optimized_pipeline=optimized_pipeline,
        test_samples=test_samples,
        output_dir=run_dir,
    )

    # Summary
    logger.info("\n" + "="*60)
    logger.info("UNIFIED TRAINING PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"Total preferences collected: {len(preferences)}")
    logger.info(f"Test set average error: {final_results['avg_error']:.2f}")
    logger.info(f"Results saved to: {run_dir}")
    logger.info("="*60)

    return {
        'config': config,
        'preferences_collected': len(preferences),
        'final_results': final_results,
        'output_dir': str(run_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Unified Training Pipeline - Full Learning Flow"
    )

    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--genrm-port", type=int, default=8001, help="GenRM server port")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--budget", type=str, default="light",
                       choices=["light", "medium", "heavy", "superheavy"])
    parser.add_argument("--threads", type=int, default=4, help="Parallel threads")
    parser.add_argument("--skip-judge-optimization", action="store_true",
                       help="Skip Phase 2 (judge optimization)")
    parser.add_argument("--output-dir", type=Path,
                       default=Path("outputs/unified_training"))

    args = parser.parse_args()

    run_unified_pipeline(
        port=args.port,
        genrm_port=args.genrm_port,
        num_samples=args.samples,
        budget=args.budget,
        num_threads=args.threads,
        skip_judge_optimization=args.skip_judge_optimization,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
