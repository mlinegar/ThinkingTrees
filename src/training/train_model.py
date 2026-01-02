#!/usr/bin/env python3
"""
Unified Model Training for OPS Framework.

Consolidates training workflows into a single entry point:
- rile-oracle: Train RILE oracle classifier from manifesto results
- ops-comparison: Train OPS comparison module from preference pairs

Usage Examples:
    # Train OPS comparison module from preference data
    python -m src.training.train_model --type ops-comparison \
        --preference-data data/preferences/pairs.json \
        --output-dir models/comparison

    # Train RILE oracle classifier from results
    python -m src.training.train_model --type rile-oracle \
        --results-dir data/results/manifesto_rile/run_xxx \
        --output-dir models/oracle
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.config.logging import setup_logging, get_logger

logger = get_logger(__name__)


# =============================================================================
# Common Utilities
# =============================================================================

def preference_metric(example, prediction, trace=None) -> float:
    """Metric for preference prediction accuracy."""
    predicted = str(getattr(prediction, "preferred", "")).upper().strip()
    actual = str(getattr(example, "preferred", "")).upper().strip()

    if predicted == actual:
        return 1.0
    if predicted == "TIE" or actual == "TIE":
        return 0.5
    return 0.0


def print_banner(title: str, config: dict) -> None:
    """Print configuration banner."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    for key, value in config.items():
        print(f"  {key:20s} {value}")
    print("=" * 70)
    print()


# =============================================================================
# Type: OPS Comparison Training
# =============================================================================

def train_ops_comparison(args) -> None:
    """Train OPS comparison module from preference pairs."""
    import dspy
    from src.config.dspy_config import configure_dspy
    from src.config.settings import load_settings
    from src.ops_engine.training_framework.ops_comparison_module import OPSComparisonModule
    from src.ops_engine.training_framework.preference import PreferenceDataset

    if not args.preference_data or not args.preference_data.exists():
        raise ValueError(f"Preference data required: {args.preference_data}")

    print_banner("OPS COMPARISON MODULE TRAINING", {
        "Preference Data:": str(args.preference_data),
        "Law Type:": args.law_type,
        "Model:": args.model,
        "Port:": str(args.port),
        "Budget:": args.budget,
        "Output Directory:": str(args.output_dir),
    })

    # Load and filter dataset
    logger.info("Loading preference dataset...")
    dataset = PreferenceDataset.load(args.preference_data)

    if args.law_type != "all":
        pairs = [p for p in dataset.pairs if p.law_type == args.law_type]
        dataset = PreferenceDataset(pairs)

    if args.max_pairs:
        dataset = PreferenceDataset(dataset.pairs[:args.max_pairs])

    if len(dataset) == 0:
        raise ValueError("No preference pairs available after filtering")

    logger.info(f"Training on {len(dataset)} pairs (law_type={args.law_type})")

    # Configure LM
    settings = load_settings(args.config)
    gen_cfg = settings.get("generation", {})
    judge_cfg = gen_cfg.get("comparison_judge", {})

    temperature = args.temperature or judge_cfg.get("temperature", 0.3)
    max_tokens = args.max_tokens or judge_cfg.get("max_tokens", 2048)

    lm = dspy.LM(
        model=args.model,
        api_base=f"http://localhost:{args.port}/v1",
        api_key="not-needed",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    configure_dspy(lm=lm)

    # Split and convert to examples
    train_set, val_set = dataset.split(train_ratio=args.train_ratio, shuffle=True)
    train_examples = train_set.to_dspy_examples()
    val_examples = val_set.to_dspy_examples()

    logger.info(f"Train: {len(train_examples)} | Val: {len(val_examples)}")

    # Create and optimize module
    comparison_module = OPSComparisonModule(use_cot=not args.no_cot)
    optimizer = dspy.GEPA(
        metric=preference_metric,
        auto=args.budget,
        num_threads=args.num_threads,
    )

    compile_kwargs = {
        "student": comparison_module,
        "trainset": train_examples,
    }
    if val_examples:
        compile_kwargs["valset"] = val_examples

    logger.info("Starting GEPA optimization...")
    trained_module = optimizer.compile(**compile_kwargs)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = args.output_dir / f"ops_comparison_{timestamp}.json"
    trained_module.save(str(model_path))

    stats = {
        "created_at": datetime.now().isoformat(),
        "type": "ops-comparison",
        "model_path": str(model_path),
        "num_pairs": len(dataset),
        "law_type": args.law_type,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "config": {
            "model": args.model,
            "port": args.port,
            "budget": args.budget,
            "num_threads": args.num_threads,
            "use_cot": not args.no_cot,
            "train_ratio": args.train_ratio,
        },
    }

    stats_path = args.output_dir / f"ops_comparison_{timestamp}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print()
    print("=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Model saved to:    {model_path}")
    print(f"  Stats saved to:    {stats_path}")
    print("=" * 70)


# =============================================================================
# Type: RILE Oracle Training
# =============================================================================

def train_rile_oracle(args) -> None:
    """Train RILE oracle classifier from manifesto results."""
    import dspy
    from dspy.teleprompt import BootstrapFewShot
    from src.config.dspy_config import configure_dspy
    from src.config.settings import load_settings
    from src.tasks.manifesto import (
        ManifestoDataset,
        ManifestoPipeline,
        create_training_examples,
        rile_metric,
    )

    print_banner("RILE ORACLE CLASSIFIER TRAINING", {
        "Results Dir:": str(args.results_dir) if args.results_dir else "Process new",
        "Samples:": str(args.samples),
        "Port:": str(args.port),
        "Bin Size:": str(args.bin_size),
        "Output Directory:": str(args.output_dir),
    })

    # Configure DSPy
    settings = load_settings(args.config)
    gen_cfg = settings.get("generation", {})
    summarizer_cfg = gen_cfg.get("summarizer", {})

    lm = dspy.LM(
        "openai/default",
        api_base=f"http://localhost:{args.port}/v1",
        api_key="EMPTY",
        temperature=summarizer_cfg.get("temperature", 0.3),
        max_tokens=summarizer_cfg.get("max_tokens", 8192),
    )
    configure_dspy(lm=lm)
    logger.info(f"DSPy configured with vLLM on port {args.port}")

    # Get training data
    if args.results_dir and args.results_dir.exists():
        # Load existing results
        logger.info(f"Loading results from {args.results_dir}...")
        result_files = list(args.results_dir.glob("**/results.json"))
        if not result_files:
            raise FileNotFoundError(f"No results files found in {args.results_dir}")

        with open(result_files[0]) as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} results")

        # Create training examples from results
        training_examples = []
        for r in results:
            if r.get('estimated_score') is not None:
                training_examples.append(dspy.Example(
                    text=r.get('text', ''),  # Use full text - truncation corrupts training
                    rile_score=r.get('reference_score', 0.0),
                ).with_inputs('text'))
    else:
        # Process new manifestos
        logger.info("Processing new manifestos...")
        dataset = ManifestoDataset(
            countries=[51, 41],
            min_year=1990,
            require_text=True,
        )

        sample_ids = dataset.get_all_ids()[:args.samples]
        samples = [dataset.get_sample(sid) for sid in sample_ids if dataset.get_sample(sid)]

        training_examples = create_training_examples(samples)
        logger.info(f"Created {len(training_examples)} training examples")

    if len(training_examples) < 4:
        raise ValueError(f"Need at least 4 training examples, got {len(training_examples)}")

    # Limit training examples
    if args.max_examples and len(training_examples) > args.max_examples:
        training_examples = training_examples[:args.max_examples]

    logger.info(f"Using {len(training_examples)} training examples")

    # Create and train pipeline
    pipeline = ManifestoPipeline(chunk_size=2000)

    optimizer = BootstrapFewShot(
        metric=rile_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
    )

    logger.info("Starting BootstrapFewShot optimization...")
    trained_pipeline = optimizer.compile(pipeline, trainset=training_examples)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = args.output_dir / f"rile_oracle_{timestamp}.json"
    trained_pipeline.save(str(model_path))

    stats = {
        "created_at": datetime.now().isoformat(),
        "type": "rile-oracle",
        "model_path": str(model_path),
        "num_examples": len(training_examples),
        "bin_size": args.bin_size,
        "config": {
            "port": args.port,
            "samples": args.samples,
            "max_examples": args.max_examples,
        },
    }

    stats_path = args.output_dir / f"rile_oracle_{timestamp}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print()
    print("=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Model saved to:    {model_path}")
    print(f"  Stats saved to:    {stats_path}")
    print("=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified model training for OPS framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Type selection
    parser.add_argument(
        "--type", type=str, required=True,
        choices=["rile-oracle", "ops-comparison"],
        help="Training type"
    )

    # Common options
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--model", type=str, default="openai/qwen-30b-thinking")
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    # OPS comparison options
    parser.add_argument("--preference-data", type=Path, default=None)
    parser.add_argument("--law-type", type=str, default="all",
                       choices=["all", "sufficiency", "idempotence", "merge"])
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--budget", type=str, default="heavy",
                       choices=["light", "medium", "heavy", "superheavy"])
    parser.add_argument("--num-threads", type=int, default=64)
    parser.add_argument("--no-cot", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)

    # RILE oracle options
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--bin-size", type=float, default=10.0)
    parser.add_argument("--max-examples", type=int, default=50)

    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    setup_logging(verbose=args.verbose)

    if args.type == "ops-comparison":
        train_ops_comparison(args)
    elif args.type == "rile-oracle":
        train_rile_oracle(args)
    else:
        logger.error(f"Unknown type: {args.type}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
