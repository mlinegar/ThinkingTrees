#!/usr/bin/env python3
"""
Unified Experiment Runner for Manifesto RILE Scoring.

Consolidates three experiment modes into a single entry point:
- single: Basic evaluation with BatchedDocPipeline
- batched: High-throughput batched processing with concurrency tuning
- optimized: DSPy prompt optimization workflow

Usage Examples:
    # Basic evaluation
    python -m src.tasks.manifesto.run_experiment --mode single --max-samples 10

    # High-throughput batched processing
    python -m src.tasks.manifesto.run_experiment --mode batched --samples 100 --concurrent-docs 30

    # DSPy optimization
    python -m src.tasks.manifesto.run_experiment --mode optimized --iterations 3 --max-samples 20

    # Full options
    python -m src.tasks.manifesto.run_experiment \
        --mode batched \
        --task manifesto_rile \
        --port 8000 \
        --samples 50 \
        --output-dir data/results
"""

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config.logging import setup_logging, get_logger
from src.tasks.manifesto import RILE_SCALE

logger = get_logger(__name__)


# =============================================================================
# Common Utilities
# =============================================================================

def normalize_score(value: Optional[float], scale) -> Optional[float]:
    """Normalize a raw score to 0-1."""
    if value is None:
        return None
    normalized = scale.normalize(float(value))
    return max(0.0, min(1.0, normalized))


def compute_metrics(results: list) -> dict:
    """Compute evaluation metrics from results."""
    valid = [r for r in results if r.get('estimated_score') is not None and not r.get('error')]

    if not valid:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "correlation": 0.0,
            "within_0_05": 0.0,
            "n_samples": 0,
            "n_failed": len(results),
        }

    errors = [abs(r['estimated_score'] - r['reference_score']) for r in valid]
    mae = sum(errors) / len(errors)
    rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
    within_05 = sum(1 for e in errors if e <= 0.05) / len(errors) * 100

    # Pearson correlation
    pred = [r['estimated_score'] for r in valid]
    actual = [r['reference_score'] for r in valid]
    n = len(pred)
    mean_pred = sum(pred) / n
    mean_actual = sum(actual) / n
    num = sum((p - mean_pred) * (a - mean_actual) for p, a in zip(pred, actual))
    denom_pred = math.sqrt(sum((p - mean_pred)**2 for p in pred))
    denom_actual = math.sqrt(sum((a - mean_actual)**2 for a in actual))
    correlation = num / (denom_pred * denom_actual) if denom_pred and denom_actual else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "correlation": correlation,
        "within_0_05": within_05,
        "n_samples": len(valid),
        "n_failed": len(results) - len(valid),
    }


def save_results(output_dir: Path, results: list, metrics: dict, config: dict) -> None:
    """Save experiment results and metrics."""
    # Results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(f"Results saved to {output_dir}")


def print_summary(metrics: dict, mode: str) -> None:
    """Print experiment summary."""
    print("\n" + "=" * 60)
    print(f"  EXPERIMENT SUMMARY ({mode.upper()} MODE)")
    print("=" * 60)
    print(f"  Samples processed: {metrics['n_samples']}")
    print(f"  Failed:            {metrics['n_failed']}")
    print(f"  MAE (0-1):         {metrics['mae']:.3f}")
    print(f"  RMSE (0-1):        {metrics['rmse']:.3f}")
    print(f"  Correlation:       {metrics['correlation']:.3f}")
    print(f"  Within 0.05:       {metrics['within_0_05']:.1f}%")
    print("=" * 60)


# =============================================================================
# Mode: Single (Basic Evaluation)
# =============================================================================

def run_single_mode(args, samples, output_dir: Path) -> list:
    """Run basic single-threaded evaluation."""
    from src.pipelines.batched import BatchedDocPipeline, BatchedPipelineConfig

    logger.info("Running SINGLE mode evaluation...")

    config = BatchedPipelineConfig(
        task_model_url=f"http://localhost:{args.port}/v1",
        max_chunk_chars=args.chunk_size,
        run_baseline=not args.no_baseline,
    )
    pipeline = BatchedDocPipeline(config)

    results = []
    for i, sample in enumerate(samples):
        logger.info(f"[{i+1}/{len(samples)}] Processing {sample.manifesto_id}")
        reference_score = normalize_score(sample.rile, RILE_SCALE)
        try:
            result = pipeline.process_manifesto(sample)
            result_dict = {
                "manifesto_id": sample.manifesto_id,
                "estimated_score": result.estimated_score,
                "reference_score": reference_score,
                "error": str(result.error) if result.error else None,
            }
            results.append(result_dict)

            if result.estimated_score is not None:
                err = abs(result.estimated_score - reference_score)
                logger.info(f"  Pred: {result.estimated_score:.3f}, Actual: {reference_score:.3f}, Error: {err:.3f}")
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results.append({
                "manifesto_id": sample.manifesto_id,
                "estimated_score": None,
                "reference_score": reference_score,
                "error": str(e),
            })

    return results


# =============================================================================
# Mode: Batched (High Throughput)
# =============================================================================

def run_batched_mode(args, samples, output_dir: Path) -> list:
    """Run high-throughput batched processing."""
    from src.pipelines.batched import BatchedDocPipeline, BatchedPipelineConfig

    logger.info("Running BATCHED mode evaluation...")
    logger.info(f"  Concurrent docs: {args.concurrent_docs}")
    logger.info(f"  Concurrent requests: {args.concurrent_requests}")

    config = BatchedPipelineConfig(
        task_model_url=f"http://localhost:{args.port}/v1",
        max_concurrent_requests=args.concurrent_requests,
        max_concurrent_documents=args.concurrent_docs,
        batch_size=args.batch_size,
        batch_timeout=args.batch_timeout,
        max_chunk_chars=args.chunk_size,
        run_baseline=not args.no_baseline,
    )

    def progress_callback(completed, total):
        pct = completed / total * 100
        logger.info(f"Progress: {completed}/{total} ({pct:.1f}%)")

    pipeline = BatchedDocPipeline(config)

    start_time = time.time()
    raw_results = pipeline.process_batch(samples, progress_callback=progress_callback)
    elapsed = time.time() - start_time

    # Convert to dict format
    results = []
    for idx, r in enumerate(raw_results):
        sample = samples[idx] if idx < len(samples) else None
        reference_score = (
            normalize_score(sample.rile, RILE_SCALE)
            if sample
            else getattr(r, "reference_score", None)
        )
        results.append({
            "manifesto_id": getattr(r, 'doc_id', None),
            "estimated_score": r.estimated_score,
            "reference_score": reference_score,
            "error": str(r.error) if r.error else None,
        })

    throughput = len(samples) / elapsed
    logger.info(f"Completed in {elapsed:.1f}s ({throughput:.2f} samples/sec)")

    # Save throughput stats
    with open(output_dir / 'throughput.json', 'w') as f:
        json.dump({
            "elapsed_seconds": elapsed,
            "samples_per_second": throughput,
            "total_samples": len(samples),
            "concurrent_docs": args.concurrent_docs,
            "concurrent_requests": args.concurrent_requests,
        }, f, indent=2)

    return results


# =============================================================================
# Mode: Optimized (DSPy Optimization)
# =============================================================================

def run_optimized_mode(args, samples, output_dir: Path) -> list:
    """Run DSPy optimization workflow."""
    import dspy
    from openai import OpenAI
    from src.config.dspy_config import configure_dspy
    from src.config.settings import load_settings
    from src.tasks.manifesto import (
        ManifestoPipeline,
        create_training_examples,
        rile_metric,
    )

    logger.info("Running OPTIMIZED mode with DSPy...")
    logger.info(f"  Iterations: {args.iterations}")

    # Verify model connection
    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{args.port}/v1")
    try:
        models = client.models.list()
        model_id = models.data[0].id if models.data else "default"
        logger.info(f"  Model: {model_id}")
    except Exception as e:
        logger.error(f"Model not ready: {e}")
        return []

    # Configure DSPy
    settings = load_settings(args.config)
    gen_cfg = settings.get("generation", {})
    summarizer_cfg = gen_cfg.get("summarizer", {})

    lm = dspy.LM(
        f"openai/{model_id}",
        api_base=f"http://localhost:{args.port}/v1",
        api_key="EMPTY",
        temperature=summarizer_cfg.get("temperature", 0.3),
        max_tokens=16000,
    )
    configure_dspy(lm=lm)

    # Create pipeline
    pipeline = ManifestoPipeline(chunk_size=args.chunk_size)
    training_examples = create_training_examples(samples)

    all_results = []

    def process_sample(pipeline, sample, idx, total):
        """Process single sample through pipeline."""
        result = {
            'manifesto_id': sample.manifesto_id,
            'reference_score': normalize_score(sample.rile, RILE_SCALE),
            'text_length': len(sample.text),
        }
        try:
            pred = pipeline(text=sample.text)
            result['estimated_score'] = pred.rile_score
            result['summary_length'] = len(pred.final_summary)
            logger.info(
                f"[{idx+1}/{total}] {sample.manifesto_id}: "
                f"Pred={pred.rile_score:.3f}, Actual={result['reference_score']:.3f}"
            )
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"[{idx+1}/{total}] {sample.manifesto_id}: Error: {e}")
        return result

    # Run iterations
    for iteration in range(1, args.iterations + 1):
        logger.info(f"\n=== Iteration {iteration}/{args.iterations} ===")

        # Process samples in parallel
        results = [None] * len(samples)
        with ThreadPoolExecutor(max_workers=len(samples)) as executor:
            futures = {
                executor.submit(process_sample, pipeline, sample, idx, len(samples)): idx
                for idx, sample in enumerate(samples)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = {'error': str(e)}

        # Save iteration results
        with open(output_dir / f'iteration_{iteration}.json', 'w') as f:
            json.dump(results, f, indent=2)

        metrics = compute_metrics(results)
        logger.info(
            f"Iteration {iteration}: MAE={metrics['mae']:.3f}, Within 0.05={metrics['within_0_05']:.1f}%"
        )

        all_results.extend(results)

        # Optimize if not last iteration
        if iteration < args.iterations and len(training_examples) >= 4:
            logger.info("Optimizing pipeline with DSPy...")
            from dspy.teleprompt import BootstrapFewShot

            optimizer = BootstrapFewShot(
                metric=rile_metric,
                max_bootstrapped_demos=3,
                max_labeled_demos=3,
            )
            try:
                pipeline = optimizer.compile(pipeline, trainset=training_examples)
                pipeline.save(str(output_dir / f"pipeline_iter{iteration}.json"))
                logger.info("Optimization complete")
            except Exception as e:
                logger.error(f"Optimization failed: {e}")

    return all_results


# =============================================================================
# Main Entry Point
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified experiment runner for manifesto RILE scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selection
    parser.add_argument(
        "--mode", type=str, default="single",
        choices=["single", "batched", "optimized"],
        help="Experiment mode: single (basic), batched (high-throughput), optimized (DSPy)"
    )

    # Task selection
    parser.add_argument("--task", type=str, default="manifesto_rile", help="Task name")

    # Data options
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--max-samples", "--samples", type=int, default=None, dest="max_samples")
    parser.add_argument("--countries", type=int, nargs="+", default=[51, 41])
    parser.add_argument("--min-year", type=int, default=1990)

    # Server options
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")

    # Pipeline options
    parser.add_argument("--chunk-size", type=int, default=2000, help="Max chunk chars")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline scoring")

    # Batched mode options
    parser.add_argument("--concurrent-docs", type=int, default=30)
    parser.add_argument("--concurrent-requests", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--batch-timeout", type=float, default=0.02)

    # Optimized mode options
    parser.add_argument("--iterations", type=int, default=3, help="Optimization iterations")
    parser.add_argument("--config", type=Path, default=None, help="Path to settings.yaml")

    # Output options
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = args.experiment_name or f"{args.mode}_{timestamp}"

    if args.output_dir:
        output_dir = args.output_dir / exp_name
    else:
        output_dir = Path("data/results/manifesto_rile") / exp_name

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load dataset
    logger.info("Loading dataset...")
    from src.tasks.manifesto import ManifestoDataset

    dataset = ManifestoDataset(
        countries=args.countries,
        min_year=args.min_year,
        require_text=True,
    )

    # Get samples based on split
    if args.split == "all":
        sample_ids = dataset.get_all_ids()
    else:
        train_ids, val_ids, test_ids = dataset.create_temporal_split()
        sample_ids = {"train": train_ids, "val": val_ids, "test": test_ids}[args.split]

    if args.max_samples:
        sample_ids = sample_ids[:args.max_samples]

    samples = [dataset.get_sample(sid) for sid in sample_ids]
    samples = [s for s in samples if s is not None]

    logger.info(f"Loaded {len(samples)} samples from {args.split} split")

    if not samples:
        logger.error("No samples to process!")
        return 1

    # Save config
    config = vars(args).copy()
    config['n_samples'] = len(samples)
    config['timestamp'] = timestamp

    # Run appropriate mode
    if args.mode == "single":
        results = run_single_mode(args, samples, output_dir)
    elif args.mode == "batched":
        results = run_batched_mode(args, samples, output_dir)
    elif args.mode == "optimized":
        results = run_optimized_mode(args, samples, output_dir)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1

    # Compute and save metrics
    metrics = compute_metrics(results)
    save_results(output_dir, results, metrics, config)
    print_summary(metrics, args.mode)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
