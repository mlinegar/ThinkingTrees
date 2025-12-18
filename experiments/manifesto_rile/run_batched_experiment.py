#!/usr/bin/env python3
"""
Run Batched Manifesto Experiment.

This script demonstrates high-throughput batched processing:
- Processes N manifestos concurrently
- Pools all LLM requests for optimal GPU utilization
- Shows throughput statistics

Expected throughput improvement:
- Sequential: ~1 doc/min (60+ seconds per manifesto)
- Batched: ~5-20 docs/min depending on GPU and concurrency settings

Usage:
    # Basic run
    python run_batched_experiment.py --samples 100 --port 8000

    # High throughput settings
    python run_batched_experiment.py --samples 200 --concurrent-docs 100 --concurrent-requests 200

    # Compare with sequential
    python run_batched_experiment.py --samples 10 --compare-sequential
"""

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Batched Manifesto Experiment")

    # Sample selection
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to process")
    parser.add_argument("--countries", type=int, nargs="+", default=[51, 41],
                       help="Country codes (51=UK, 41=Germany)")
    parser.add_argument("--min-year", type=int, default=1990, help="Minimum year")

    # Server settings
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--model-url", type=str, default=None,
                       help="Full model URL (overrides --port)")

    # Batching settings
    parser.add_argument("--concurrent-docs", type=int, default=50,
                       help="Max concurrent documents")
    parser.add_argument("--concurrent-requests", type=int, default=100,
                       help="Max concurrent LLM requests")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Requests per batch")
    parser.add_argument("--batch-timeout", type=float, default=0.05,
                       help="Max wait to fill batch (seconds)")

    # Processing options
    parser.add_argument("--no-baseline", action="store_true",
                       help="Skip baseline scoring")
    parser.add_argument("--max-chunk-chars", type=int, default=2000,
                       help="Max characters per chunk")

    # Comparison
    parser.add_argument("--compare-sequential", action="store_true",
                       help="Also run sequential for comparison")

    # Output
    parser.add_argument("--output-dir", type=Path,
                       default=Path("data/results/manifesto_rile/batched"),
                       help="Output directory")

    args = parser.parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load dataset
    logger.info("Loading dataset...")
    from src.manifesto.data_loader import ManifestoDataset

    dataset = ManifestoDataset(
        countries=args.countries,
        min_year=args.min_year,
        require_text=True,
    )

    sample_ids = dataset.get_all_ids()[:args.samples]
    samples = [dataset.get_sample(sid) for sid in sample_ids]
    samples = [s for s in samples if s is not None]
    logger.info(f"Loaded {len(samples)} samples")

    # Save sample metadata
    sample_meta = [
        {"id": s.manifesto_id, "party": s.party_name, "country": s.country_name, "year": s.year}
        for s in samples
    ]
    with open(output_dir / "samples.json", "w") as f:
        json.dump(sample_meta, f, indent=2)

    # Configure batched pipeline
    from src.manifesto.batched_pipeline import (
        BatchedManifestoPipeline,
        BatchedPipelineConfig,
    )

    model_url = args.model_url or f"http://localhost:{args.port}/v1"

    config = BatchedPipelineConfig(
        task_model_url=model_url,
        max_concurrent_requests=args.concurrent_requests,
        max_concurrent_documents=args.concurrent_docs,
        batch_size=args.batch_size,
        batch_timeout=args.batch_timeout,
        max_chunk_chars=args.max_chunk_chars,
        run_baseline=not args.no_baseline,
    )

    # Progress callback
    def progress(completed, total):
        pct = completed / total * 100
        logger.info(f"Progress: {completed}/{total} ({pct:.1f}%)")

    # Run batched experiment
    logger.info("=" * 60)
    logger.info("BATCHED PROCESSING")
    logger.info("=" * 60)
    logger.info(f"Config: {args.concurrent_docs} concurrent docs, "
               f"{args.concurrent_requests} concurrent requests")

    pipeline = BatchedManifestoPipeline(config)

    start_time = time.time()
    results = pipeline.process_batch(samples, progress_callback=progress)
    batched_time = time.time() - start_time

    # Compute metrics
    valid_results = [r for r in results if r.error is None and r.predicted_rile is not None]
    errors = [abs(r.predicted_rile - r.ground_truth_rile) for r in valid_results]
    mae = sum(errors) / len(errors) if errors else 0

    batched_stats = {
        "method": "batched",
        "samples": len(samples),
        "successful": len(valid_results),
        "failed": len(samples) - len(valid_results),
        "total_time_sec": batched_time,
        "samples_per_sec": len(samples) / batched_time,
        "mae": mae,
        "config": {
            "concurrent_docs": args.concurrent_docs,
            "concurrent_requests": args.concurrent_requests,
            "batch_size": args.batch_size,
        }
    }

    logger.info(f"Batched Results:")
    logger.info(f"  Time: {batched_time:.1f}s")
    logger.info(f"  Throughput: {batched_stats['samples_per_sec']:.2f} samples/sec")
    logger.info(f"  MAE: {mae:.2f} RILE points")
    logger.info(f"  Success rate: {len(valid_results)}/{len(samples)}")

    # Save batched results
    from src.manifesto.evaluation import save_results as save_manifesto_results
    save_manifesto_results(results, output_dir / "batched_results.json")

    # Optional: Compare with sequential
    if args.compare_sequential:
        logger.info("")
        logger.info("=" * 60)
        logger.info("SEQUENTIAL PROCESSING (for comparison)")
        logger.info("=" * 60)

        # Use subset for sequential (it's slow!)
        seq_samples = samples[:min(10, len(samples))]
        logger.info(f"Running sequential on {len(seq_samples)} samples...")

        from src.manifesto.ops_pipeline import ManifestoOPSPipeline, PipelineConfig as SeqConfig

        seq_config = SeqConfig(
            task_model_port=args.port,
            run_baseline=not args.no_baseline,
            max_chunk_chars=args.max_chunk_chars,
        )

        seq_pipeline = ManifestoOPSPipeline(seq_config)

        start_time = time.time()
        seq_results = []
        for i, sample in enumerate(seq_samples):
            logger.info(f"Sequential: {i+1}/{len(seq_samples)}")
            result = seq_pipeline.process_manifesto(sample)
            seq_results.append(result)
        sequential_time = time.time() - start_time

        seq_valid = [r for r in seq_results if r.error is None and r.predicted_rile is not None]
        seq_errors = [abs(r.predicted_rile - r.ground_truth_rile) for r in seq_valid]
        seq_mae = sum(seq_errors) / len(seq_errors) if seq_errors else 0

        sequential_stats = {
            "method": "sequential",
            "samples": len(seq_samples),
            "successful": len(seq_valid),
            "total_time_sec": sequential_time,
            "samples_per_sec": len(seq_samples) / sequential_time,
            "mae": seq_mae,
        }

        # Compute speedup
        speedup = (sequential_stats["samples_per_sec"] / batched_stats["samples_per_sec"]
                  if batched_stats["samples_per_sec"] > 0 else 0)
        speedup = batched_stats["samples_per_sec"] / sequential_stats["samples_per_sec"]

        logger.info(f"Sequential Results:")
        logger.info(f"  Time: {sequential_time:.1f}s for {len(seq_samples)} samples")
        logger.info(f"  Throughput: {sequential_stats['samples_per_sec']:.2f} samples/sec")
        logger.info(f"  MAE: {seq_mae:.2f} RILE points")

        logger.info("")
        logger.info("=" * 60)
        logger.info("COMPARISON")
        logger.info("=" * 60)
        logger.info(f"Speedup: {speedup:.1f}x faster with batching")
        logger.info(f"  Sequential: {sequential_stats['samples_per_sec']:.2f} samples/sec")
        logger.info(f"  Batched: {batched_stats['samples_per_sec']:.2f} samples/sec")

        batched_stats["speedup_vs_sequential"] = speedup
        batched_stats["sequential_comparison"] = sequential_stats

    # Save final stats
    with open(output_dir / "experiment_stats.json", "w") as f:
        json.dump(batched_stats, f, indent=2)

    logger.info("")
    logger.info(f"Results saved to: {output_dir}")

    # Generate evaluation report
    from src.manifesto.evaluation import ManifestoEvaluator
    evaluator = ManifestoEvaluator()
    report = evaluator.generate_report(results, output_path=output_dir / "report.txt")
    print("\n" + report)


if __name__ == "__main__":
    main()
