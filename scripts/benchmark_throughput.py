#!/usr/bin/env python3
"""
Throughput Comparison Tool for vLLM Models.

Compare generation speed between different vLLM model deployments.
Supports two modes:
1. Auto mode (recommended): Automatically starts/stops vLLM servers
2. Manual mode: Uses pre-started servers at specified URLs

Usage (Auto Mode - Recommended):
    # Compare two models using profiles from config/settings.yaml
    python scripts/benchmark_throughput.py \
        --profile-a nemotron-30b-fp8 \
        --profile-b qwen-30b-thinking \
        --samples 50

Usage (Manual Mode):
    # Compare using pre-started vLLM servers
    python scripts/benchmark_throughput.py \
        --model-a "Nemotron-30B-FP8" --url-a http://localhost:8000/v1 \
        --model-b "Qwen-30B-Thinking" --url-b http://localhost:8002/v1 \
        --samples 50

Available model profiles (from config/settings.yaml):
    - qwen-80b
    - qwen-30b-thinking
    - qwen-235b
    - glm-4.6
    - olmo-32b-think
    - nemotron-30b-fp8
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.throughput import (
    ThroughputComparison,
    ThroughputBenchmark,
    VLLMServerManager,
    run_sequential_comparison,
    run_parallel_comparison,
    load_model_config,
    save_results,
)
from src.tasks.manifesto.data_loader import ManifestoDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare throughput between vLLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode selection
    mode_group = parser.add_argument_group("Mode Selection (choose one)")

    # Auto mode: use profiles
    mode_group.add_argument(
        "--profile-a",
        help="Model A profile from config/settings.yaml (auto-starts server)"
    )
    mode_group.add_argument(
        "--profile-b",
        help="Model B profile from config/settings.yaml (auto-starts server)"
    )

    # Manual mode: use URLs
    mode_group.add_argument(
        "--model-a",
        help="Model A name (for manual mode with --url-a)"
    )
    mode_group.add_argument(
        "--url-a",
        help="vLLM URL for model A (e.g., http://localhost:8000/v1)"
    )
    mode_group.add_argument(
        "--model-b",
        help="Model B name (for manual mode with --url-b)"
    )
    mode_group.add_argument(
        "--url-b",
        help="vLLM URL for model B (e.g., http://localhost:8002/v1)"
    )

    # Benchmark settings
    bench_group = parser.add_argument_group("Benchmark Settings")
    bench_group.add_argument(
        "--samples", type=int, default=50,
        help="Number of manifesto samples to use (default: 50)"
    )
    bench_group.add_argument(
        "--max-tokens", type=int, default=500,
        help="Maximum tokens per response (default: 500)"
    )
    bench_group.add_argument(
        "--concurrent", type=int, default=100,
        help="Maximum concurrent requests (default: 100)"
    )
    bench_group.add_argument(
        "--port", type=int, default=8000,
        help="Port for auto mode (default: 8000)"
    )
    bench_group.add_argument(
        "--chunk-size", type=int, default=2000,
        help="Characters per chunk - smaller = more concurrent requests (default: 2000)"
    )

    # Parallel mode settings
    parallel_group = parser.add_argument_group("Parallel Mode (run both models simultaneously)")
    parallel_group.add_argument(
        "--parallel", action="store_true",
        help="Run both models in parallel on separate GPU sets"
    )
    parallel_group.add_argument(
        "--gpus-a", default="0,1",
        help="CUDA devices for model A (default: 0,1)"
    )
    parallel_group.add_argument(
        "--gpus-b", default="2,3",
        help="CUDA devices for model B (default: 2,3)"
    )
    parallel_group.add_argument(
        "--port-b", type=int, default=8002,
        help="Port for model B in parallel mode (default: 8002)"
    )
    parallel_group.add_argument(
        "--tensor-parallel", type=int, default=2,
        help="Tensor parallel size for parallel mode (default: 2)"
    )

    # Data settings
    data_group = parser.add_argument_group("Data Settings")
    data_group.add_argument(
        "--countries", nargs="+", type=int, default=[51, 41],
        help="CMP country codes to filter (default: 51 41 = UK, Germany)"
    )
    data_group.add_argument(
        "--min-year", type=int, default=2000,
        help="Minimum year for manifesto samples (default: 2000)"
    )

    # Output
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Path to save JSON results (optional)"
    )

    # Single model mode
    parser.add_argument(
        "--single", action="store_true",
        help="Run benchmark on model A only (no comparison)"
    )

    # List profiles
    parser.add_argument(
        "--list-profiles", action="store_true",
        help="List available model profiles and exit"
    )

    return parser.parse_args()


def list_profiles():
    """List available model profiles from config."""
    import yaml
    config_path = PROJECT_ROOT / "config" / "settings.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    models = cfg.get("vllm", {}).get("models", {})
    default = cfg.get("vllm", {}).get("default", "")

    print("\nAvailable model profiles:")
    print("-" * 60)
    for name, config in models.items():
        marker = " (default)" if name == default else ""
        print(f"  {name}{marker}")
        print(f"    Path: {config.get('path', 'N/A')}")
        print(f"    Tensor Parallel: {config.get('tensor_parallel', 1)}")
        print()


def load_samples(args) -> list:
    """Load manifesto samples for benchmarking."""
    logger.info(f"Loading manifesto samples...")
    logger.info(f"  Countries: {args.countries}")
    logger.info(f"  Min year: {args.min_year}")
    logger.info(f"  Requested samples: {args.samples}")

    dataset = ManifestoDataset(
        countries=args.countries,
        min_year=args.min_year,
    )

    all_ids = dataset.get_all_ids()
    sample_ids = all_ids[:args.samples]

    samples = []
    for sid in sample_ids:
        sample = dataset.get_sample(sid)
        if sample is not None:
            samples.append(sample)

    logger.info(f"Loaded {len(samples)} samples")

    if len(samples) < args.samples:
        logger.warning(
            f"Only {len(samples)} samples available "
            f"(requested {args.samples})"
        )

    return samples


async def run_auto_comparison(args, samples):
    """Run comparison using automatic server management (sequential)."""
    from src.benchmark.throughput import ThroughputComparison

    result = await run_sequential_comparison(
        profile_a=args.profile_a,
        profile_b=args.profile_b,
        samples=samples,
        port=args.port,
        max_tokens=args.max_tokens,
        max_concurrent=args.concurrent,
        chunk_size=args.chunk_size,
        show_progress=True,
    )

    # Display comparison
    comparison = ThroughputComparison(
        model_a_name=args.profile_a,
        model_a_url=f"http://localhost:{args.port}/v1",
        model_b_name=args.profile_b,
        model_b_url=f"http://localhost:{args.port}/v1",
    )
    comparison.display_comparison(result)

    return result


async def run_auto_parallel_comparison(args, samples):
    """Run comparison with both models in parallel on separate GPU sets."""
    from src.benchmark.throughput import ThroughputComparison

    result = await run_parallel_comparison(
        profile_a=args.profile_a,
        profile_b=args.profile_b,
        samples=samples,
        port_a=args.port,
        port_b=args.port_b,
        cuda_devices_a=args.gpus_a,
        cuda_devices_b=args.gpus_b,
        tensor_parallel=args.tensor_parallel,
        max_tokens=args.max_tokens,
        max_concurrent=args.concurrent,
        chunk_size=args.chunk_size,
        show_progress=True,
    )

    # Display comparison
    comparison = ThroughputComparison(
        model_a_name=args.profile_a,
        model_a_url=f"http://localhost:{args.port}/v1",
        model_b_name=args.profile_b,
        model_b_url=f"http://localhost:{args.port_b}/v1",
    )
    comparison.display_comparison(result)

    return result


async def run_auto_single(args, samples):
    """Run single model benchmark with auto server management."""
    print(f"\n{'='*70}")
    print(f"  SINGLE MODEL BENCHMARK: {args.profile_a}")
    print(f"{'='*70}\n")

    async with VLLMServerManager(args.profile_a, port=args.port) as server:
        benchmark = ThroughputBenchmark(
            model_name=args.profile_a,
            server_url=server.url,
            max_concurrent_requests=args.concurrent,
            chunk_size=args.chunk_size,
        )
        result = await benchmark.run_benchmark(
            samples=samples,
            max_tokens=args.max_tokens,
            show_progress=True,
        )

    # Display results
    print(f"\n{'='*70}")
    print(f"  BENCHMARK RESULTS: {result.model_name}")
    print(f"{'='*70}")
    print(f"  Samples: {result.n_samples}")
    print(f"  Wall clock: {result.wall_clock_seconds:.1f}s")
    print(f"\n  Throughput:")
    print(f"    Total tok/s:  {result.tokens_per_second:,.0f}")
    print(f"    Read tok/s:   {result.read_tokens_per_second:,.0f}")
    print(f"    Write tok/s:  {result.write_tokens_per_second:,.0f}")
    print(f"\n  Latency:")
    print(f"    Avg: {result.avg_latency_ms:,.0f}ms")
    print(f"\n  Requests: {result.completed_requests}/{result.total_requests} "
          f"({result.failed_requests} failed)")
    print()

    return result


async def run_manual_comparison(args, samples):
    """Run comparison using pre-started servers."""
    comparison = ThroughputComparison(
        model_a_name=args.model_a,
        model_a_url=args.url_a,
        model_b_name=args.model_b,
        model_b_url=args.url_b,
        max_concurrent_requests=args.concurrent,
    )

    result = await comparison.compare(
        samples=samples,
        max_tokens=args.max_tokens,
        show_progress=True,
    )

    comparison.display_comparison(result)
    return result


async def run_manual_single(args, samples):
    """Run single model benchmark with pre-started server."""
    benchmark = ThroughputBenchmark(
        model_name=args.model_a,
        server_url=args.url_a,
        max_concurrent_requests=args.concurrent,
        chunk_size=args.chunk_size,
    )

    result = await benchmark.run_benchmark(
        samples=samples,
        max_tokens=args.max_tokens,
        show_progress=True,
    )

    # Display results
    print(f"\n{'='*70}")
    print(f"  BENCHMARK RESULTS: {result.model_name}")
    print(f"{'='*70}")
    print(f"  Server: {result.server_url}")
    print(f"  Samples: {result.n_samples}")
    print(f"  Wall clock: {result.wall_clock_seconds:.1f}s")
    print(f"\n  Throughput:")
    print(f"    Total tok/s:  {result.tokens_per_second:,.0f}")
    print(f"    Read tok/s:   {result.read_tokens_per_second:,.0f}")
    print(f"    Write tok/s:  {result.write_tokens_per_second:,.0f}")
    print(f"\n  Latency:")
    print(f"    Avg: {result.avg_latency_ms:,.0f}ms")
    print(f"\n  Requests: {result.completed_requests}/{result.total_requests} "
          f"({result.failed_requests} failed)")
    print()

    return result


async def main():
    args = parse_args()

    # List profiles and exit
    if args.list_profiles:
        list_profiles()
        sys.exit(0)

    # Determine mode
    auto_mode = args.profile_a is not None
    manual_mode = args.url_a is not None

    if not auto_mode and not manual_mode:
        print("Error: Must specify either --profile-a (auto mode) or --url-a (manual mode)")
        print("Use --list-profiles to see available model profiles")
        print("Use --help for full usage information")
        sys.exit(1)

    if auto_mode and manual_mode:
        print("Error: Cannot mix auto mode (--profile-*) and manual mode (--url-*)")
        sys.exit(1)

    # Validate arguments
    if auto_mode:
        if not args.single and not args.profile_b:
            print("Error: Must specify --profile-b for comparison, or use --single")
            sys.exit(1)
    else:
        if not args.model_a:
            print("Error: Must specify --model-a with --url-a")
            sys.exit(1)
        if not args.single and (not args.url_b or not args.model_b):
            print("Error: Must specify --model-b and --url-b for comparison, or use --single")
            sys.exit(1)

    # Load samples
    samples = load_samples(args)
    if not samples:
        logger.error("No samples loaded. Check your data directory and filters.")
        sys.exit(1)

    # Run benchmark
    if auto_mode:
        if args.single:
            result = await run_auto_single(args, samples)
            # Save single result
            if args.output:
                import json
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
                logger.info(f"Results saved to {args.output}")
        elif args.parallel:
            # Run both models in parallel on separate GPU sets
            result = await run_auto_parallel_comparison(args, samples)
            if args.output:
                save_results(result, args.output)
        else:
            result = await run_auto_comparison(args, samples)
            if args.output:
                save_results(result, args.output)
    else:
        if args.single:
            result = await run_manual_single(args, samples)
            if args.output:
                import json
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
                logger.info(f"Results saved to {args.output}")
        else:
            result = await run_manual_comparison(args, samples)
            if args.output:
                save_results(result, args.output)

    logger.info("Benchmark complete!")


if __name__ == "__main__":
    asyncio.run(main())
