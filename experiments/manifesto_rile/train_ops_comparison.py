#!/usr/bin/env python3
"""
Train OPS comparison module from preference data.

This script optimizes a DSPy module to predict pairwise preferences
(A/B/tie) using oracle-labeled or GenRM-labeled preference pairs.
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_python_path() -> None:
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


setup_python_path()


def preference_metric(example, prediction, trace=None) -> float:
    predicted = str(getattr(prediction, "preferred", "")).upper().strip()
    actual = str(getattr(example, "preferred", "")).upper().strip()

    if predicted == actual:
        return 1.0
    if predicted == "TIE" or actual == "TIE":
        return 0.5
    return 0.0


def filter_pairs(pairs, law_type: str) -> List:
    if law_type == "all":
        return pairs
    return [pair for pair in pairs if pair.law_type == law_type]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train OPS comparison module from preference pairs"
    )
    parser.add_argument(
        "--preference-data",
        type=Path,
        required=True,
        help="Path to preference dataset JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/ops_comparison"),
        help="Directory to save trained module",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on total pairs",
    )
    parser.add_argument(
        "--law-type",
        type=str,
        default="all",
        choices=["all", "sufficiency", "idempotence", "merge"],
        help="Filter to a specific OPS law (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/qwen-30b-thinking",
        help="Model name for DSPy LM",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port for DSPy LM",
    )
    parser.add_argument(
        "--budget",
        type=str,
        default="heavy",
        help="GEPA budget (light, medium, heavy, superheavy)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=64,
        help="Parallel threads for GEPA",
    )
    parser.add_argument(
        "--no-cot",
        action="store_true",
        help="Disable chain-of-thought in comparison module",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to settings.yaml (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override LM temperature (default: from config)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override LM max tokens (default: from config)",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    import dspy
    from src.config.settings import load_settings
    from src.ops_engine.training_framework.ops_comparison_module import OPSComparisonModule
    from src.ops_engine.training_framework.preference import PreferenceDataset

    logger.info("Loading preference dataset...")
    dataset = PreferenceDataset.load(args.preference_data)

    pairs = filter_pairs(dataset.pairs, args.law_type)
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]
    dataset = PreferenceDataset(pairs)

    if len(dataset) == 0:
        raise ValueError("No preference pairs available after filtering.")

    logger.info(f"Training on {len(dataset)} pairs (law_type={args.law_type})")

    settings = load_settings(args.config)
    generation_cfg = settings.get("generation", {})
    judge_cfg = generation_cfg.get("comparison_judge", {})
    temperature = args.temperature
    if temperature is None:
        temperature = judge_cfg.get("temperature", 0.3)
    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = judge_cfg.get("max_tokens", 2048)

    # Configure DSPy LM
    lm = dspy.LM(
        model=args.model,
        api_base=f"http://localhost:{args.port}/v1",
        api_key="not-needed",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    dspy.configure(lm=lm)

    train_set, val_set = dataset.split(train_ratio=args.train_ratio, shuffle=True)
    train_examples = train_set.to_dspy_examples()
    val_examples = val_set.to_dspy_examples()

    logger.info(f"Train examples: {len(train_examples)} | Val examples: {len(val_examples)}")

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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = args.output_dir / f"ops_comparison_{timestamp}.json"
    trained_module.save(str(model_path))

    stats = {
        "created_at": datetime.now().isoformat(),
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
            "max_pairs": args.max_pairs,
            "seed": args.seed,
        },
    }

    stats_path = args.output_dir / f"ops_comparison_{timestamp}_stats.json"
    with open(stats_path, "w") as handle:
        json.dump(stats, handle, indent=2)

    logger.info(f"Saved trained module to {model_path}")
    logger.info(f"Saved training stats to {stats_path}")


if __name__ == "__main__":
    main()
