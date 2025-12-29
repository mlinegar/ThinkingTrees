"""
Throughput Benchmark for vLLM Model Comparison.

Compares generation speed between different vLLM model deployments
using identical workloads.

Usage:
    from src.benchmark.throughput import ThroughputComparison

    comparison = ThroughputComparison(
        model_a_name="Nemotron-30B-FP8",
        model_a_url="http://localhost:8000/v1",
        model_b_name="Qwen-30B-Thinking",
        model_b_url="http://localhost:8002/v1",
    )

    result = await comparison.compare(samples, max_tokens=500)
    comparison.display_comparison(result)
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import time
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any

import aiohttp

from src.core.batch_processor import (
    AsyncBatchLLMClient,
    BatchRequest,
    BatchResponse,
    BatchStats,
)
from src.preprocessing.chunker import chunk_text

# Handle optional rich import gracefully
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Table = None
    Panel = None


logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ThroughputResult:
    """Results from a single model throughput test."""

    model_name: str
    server_url: str

    # Sample info
    n_samples: int
    total_input_chars: int
    total_output_tokens: int

    # Core throughput metrics
    wall_clock_seconds: float
    tokens_per_second: float
    read_tokens_per_second: float
    write_tokens_per_second: float

    # Latency metrics
    avg_latency_ms: float
    requests_per_second: float

    # Request stats
    total_requests: int
    completed_requests: int
    failed_requests: int

    # Token counts
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int

    # Per-sample breakdown
    per_sample_latencies: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "server_url": self.server_url,
            "n_samples": self.n_samples,
            "total_input_chars": self.total_input_chars,
            "total_output_tokens": self.total_output_tokens,
            "wall_clock_seconds": self.wall_clock_seconds,
            "tokens_per_second": self.tokens_per_second,
            "read_tokens_per_second": self.read_tokens_per_second,
            "write_tokens_per_second": self.write_tokens_per_second,
            "avg_latency_ms": self.avg_latency_ms,
            "requests_per_second": self.requests_per_second,
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "per_sample_latencies": self.per_sample_latencies,
        }


@dataclass
class ComparisonResult:
    """Side-by-side comparison of two models."""

    model_a: ThroughputResult
    model_b: ThroughputResult

    @property
    def speedup_factor(self) -> float:
        """Model B tokens/sec relative to Model A (>1 means B is faster)."""
        if self.model_a.tokens_per_second <= 0:
            return float('inf') if self.model_b.tokens_per_second > 0 else 1.0
        return self.model_b.tokens_per_second / self.model_a.tokens_per_second

    @property
    def write_speedup(self) -> float:
        """Model B generation (write) speed relative to Model A."""
        if self.model_a.write_tokens_per_second <= 0:
            return float('inf') if self.model_b.write_tokens_per_second > 0 else 1.0
        return self.model_b.write_tokens_per_second / self.model_a.write_tokens_per_second

    @property
    def latency_improvement(self) -> float:
        """Latency improvement (negative means B has lower latency)."""
        if self.model_a.avg_latency_ms <= 0:
            return 0.0
        return (self.model_b.avg_latency_ms - self.model_a.avg_latency_ms) / self.model_a.avg_latency_ms * 100

    @property
    def winner(self) -> str:
        """Determine which model is faster overall."""
        if self.speedup_factor > 1.05:  # 5% threshold
            return self.model_b.model_name
        elif self.speedup_factor < 0.95:
            return self.model_a.model_name
        else:
            return "Tie (within 5%)"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_a": self.model_a.to_dict(),
            "model_b": self.model_b.to_dict(),
            "comparison": {
                "speedup_factor": self.speedup_factor,
                "write_speedup": self.write_speedup,
                "latency_improvement_pct": self.latency_improvement,
                "winner": self.winner,
            }
        }


# =============================================================================
# Benchmark Classes
# =============================================================================

class ThroughputBenchmark:
    """Run throughput benchmarks against a vLLM server."""

    def __init__(
        self,
        model_name: str,
        server_url: str,
        max_concurrent_requests: int = 100,
        batch_timeout: float = 0.05,
        chunk_size: int = 2000,
    ):
        """
        Initialize benchmark runner.

        Args:
            model_name: Human-readable name for this model
            server_url: vLLM server URL (e.g., http://localhost:8000/v1)
            max_concurrent_requests: Maximum concurrent HTTP requests
            batch_timeout: Max time to wait for batch to fill (seconds)
            chunk_size: Characters per chunk (smaller = more concurrent requests)
        """
        self.model_name = model_name
        self.server_url = server_url
        self.max_concurrent = max_concurrent_requests
        self.batch_timeout = batch_timeout
        self.chunk_size = chunk_size

    async def run_benchmark(
        self,
        samples: List[Any],
        max_tokens: int = 500,
        show_progress: bool = True,
    ) -> ThroughputResult:
        """
        Run throughput benchmark on samples.

        Chunks each sample and sends ALL chunks concurrently to properly
        test batched inference throughput.

        Args:
            samples: List of samples with .text attribute (ManifestoSample or similar)
            max_tokens: Maximum tokens per response
            show_progress: Whether to show progress output

        Returns:
            ThroughputResult with all metrics
        """
        # Chunk all samples to create many concurrent requests
        all_chunks = []
        for i, sample in enumerate(samples):
            text = self._get_text(sample)
            chunks = chunk_text(text, self.chunk_size)
            for j, chunk in enumerate(chunks):
                all_chunks.append((f"doc{i}_chunk{j}", chunk))

        total_input_chars = sum(len(chunk) for _, chunk in all_chunks)

        if show_progress:
            logger.info(f"Starting benchmark for {self.model_name}")
            logger.info(f"  Server: {self.server_url}")
            logger.info(f"  Documents: {len(samples)}")
            logger.info(f"  Total chunks: {len(all_chunks)} (chunk_size={self.chunk_size})")
            logger.info(f"  Max tokens/response: {max_tokens}")
            logger.info(f"  Concurrent requests: {self.max_concurrent}")

        async with AsyncBatchLLMClient(
            base_url=self.server_url,
            max_concurrent=self.max_concurrent,
            batch_timeout=self.batch_timeout,
        ) as client:
            # Submit ALL chunk requests concurrently
            request_ids = []
            for request_id, chunk in all_chunks:
                request = BatchRequest(
                    request_id=request_id,
                    messages=self._build_chunk_prompt(chunk),
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                await client.submit(request)
                request_ids.append(request_id)

            if show_progress:
                logger.info(f"Submitted {len(request_ids)} requests, awaiting responses...")

            # Await all responses and collect latencies
            latencies = []
            for request_id in request_ids:
                response = await client.await_response(request_id)
                latencies.append(response.latency_ms)
                if response.error:
                    logger.warning(f"Request {request_id} failed: {response.error}")

            # Set wall clock end time before extracting stats
            # (normally set in stop(), but we need it before __aexit__)
            client.stats.wall_clock_end = time.time()

            # Extract stats
            stats = client.stats

            return ThroughputResult(
                model_name=self.model_name,
                server_url=self.server_url,
                n_samples=len(samples),
                total_input_chars=total_input_chars,
                total_output_tokens=stats.completion_tokens,
                wall_clock_seconds=stats.wall_clock_seconds,
                tokens_per_second=stats.tokens_per_second,
                read_tokens_per_second=stats.read_tokens_per_second,
                write_tokens_per_second=stats.write_tokens_per_second,
                avg_latency_ms=stats.avg_latency_ms,
                requests_per_second=stats.requests_per_second,
                total_requests=stats.total_requests,
                completed_requests=stats.completed_requests,
                failed_requests=stats.failed_requests,
                total_tokens=stats.total_tokens,
                prompt_tokens=stats.prompt_tokens,
                completion_tokens=stats.completion_tokens,
                per_sample_latencies=latencies,
            )

    def _get_text(self, sample: Any) -> str:
        """Extract text from a sample object."""
        if hasattr(sample, 'text'):
            return sample.text
        elif isinstance(sample, dict):
            return sample.get('text', str(sample))
        else:
            return str(sample)

    def _build_benchmark_prompt(self, sample: Any) -> List[Dict[str, str]]:
        """Build a standardized prompt for benchmarking (full document)."""
        text = self._get_text(sample)[:4000]
        return self._build_chunk_prompt(text)

    def _build_chunk_prompt(self, chunk: str) -> List[Dict[str, str]]:
        """Build a standardized prompt for a text chunk."""
        return [
            {
                "role": "system",
                "content": "You are a political text analyst. Summarize the political text provided, preserving key policy positions and left-right political positioning."
            },
            {
                "role": "user",
                "content": f"Summarize the following political text:\n\n{chunk}\n\nSUMMARY:"
            }
        ]


class ThroughputComparison:
    """Compare throughput between two models."""

    def __init__(
        self,
        model_a_name: str,
        model_a_url: str,
        model_b_name: str,
        model_b_url: str,
        max_concurrent_requests: int = 100,
        batch_timeout: float = 0.05,
    ):
        """
        Initialize comparison runner.

        Args:
            model_a_name: Name for model A
            model_a_url: vLLM URL for model A
            model_b_name: Name for model B
            model_b_url: vLLM URL for model B
            max_concurrent_requests: Max concurrent requests per model
            batch_timeout: Max time to wait for batch fill
        """
        self.benchmark_a = ThroughputBenchmark(
            model_a_name, model_a_url, max_concurrent_requests, batch_timeout
        )
        self.benchmark_b = ThroughputBenchmark(
            model_b_name, model_b_url, max_concurrent_requests, batch_timeout
        )

    async def compare(
        self,
        samples: List[Any],
        max_tokens: int = 500,
        show_progress: bool = True,
    ) -> ComparisonResult:
        """
        Run identical workload on both models and compare.

        Args:
            samples: List of samples to process
            max_tokens: Max tokens per response
            show_progress: Whether to show progress

        Returns:
            ComparisonResult with both results and comparison metrics
        """
        if show_progress:
            print(f"\n{'='*70}")
            print("  THROUGHPUT BENCHMARK")
            print(f"{'='*70}\n")

        # Run Model A
        if show_progress:
            print(f"Running benchmark for Model A: {self.benchmark_a.model_name}...")
        result_a = await self.benchmark_a.run_benchmark(
            samples, max_tokens, show_progress
        )
        if show_progress:
            print(f"  Completed in {result_a.wall_clock_seconds:.1f}s "
                  f"({result_a.tokens_per_second:.0f} tok/s)\n")

        # Run Model B
        if show_progress:
            print(f"Running benchmark for Model B: {self.benchmark_b.model_name}...")
        result_b = await self.benchmark_b.run_benchmark(
            samples, max_tokens, show_progress
        )
        if show_progress:
            print(f"  Completed in {result_b.wall_clock_seconds:.1f}s "
                  f"({result_b.tokens_per_second:.0f} tok/s)\n")

        return ComparisonResult(model_a=result_a, model_b=result_b)

    def display_comparison(self, result: ComparisonResult, console: Optional["Console"] = None):
        """Display formatted comparison results."""
        if RICH_AVAILABLE:
            self._display_rich(result, console)
        else:
            self._display_plain(result)

    def _display_rich(self, result: ComparisonResult, console: Optional["Console"] = None):
        """Display with rich formatting."""
        console = console or Console()

        # Model A Table
        table_a = Table(title=f"Model A: {result.model_a.model_name}",
                       show_header=True, header_style="bold blue")
        table_a.add_column("Metric", style="cyan", width=22)
        table_a.add_column("Value", style="green", justify="right", width=18)

        table_a.add_row("Server", result.model_a.server_url)
        table_a.add_row("Samples", str(result.model_a.n_samples))
        table_a.add_row("Wall Clock", f"{result.model_a.wall_clock_seconds:.1f}s")
        table_a.add_row("", "")
        table_a.add_row("Total tok/s", f"{result.model_a.tokens_per_second:,.0f}")
        table_a.add_row("Read tok/s", f"{result.model_a.read_tokens_per_second:,.0f}")
        table_a.add_row("Write tok/s", f"{result.model_a.write_tokens_per_second:,.0f}")
        table_a.add_row("", "")
        table_a.add_row("Avg Latency", f"{result.model_a.avg_latency_ms:,.0f}ms")
        table_a.add_row("Requests",
                       f"{result.model_a.completed_requests}/{result.model_a.total_requests}")

        console.print(table_a)
        console.print()

        # Model B Table
        table_b = Table(title=f"Model B: {result.model_b.model_name}",
                       show_header=True, header_style="bold blue")
        table_b.add_column("Metric", style="cyan", width=22)
        table_b.add_column("Value", style="green", justify="right", width=18)

        table_b.add_row("Server", result.model_b.server_url)
        table_b.add_row("Samples", str(result.model_b.n_samples))
        table_b.add_row("Wall Clock", f"{result.model_b.wall_clock_seconds:.1f}s")
        table_b.add_row("", "")
        table_b.add_row("Total tok/s", f"{result.model_b.tokens_per_second:,.0f}")
        table_b.add_row("Read tok/s", f"{result.model_b.read_tokens_per_second:,.0f}")
        table_b.add_row("Write tok/s", f"{result.model_b.write_tokens_per_second:,.0f}")
        table_b.add_row("", "")
        table_b.add_row("Avg Latency", f"{result.model_b.avg_latency_ms:,.0f}ms")
        table_b.add_row("Requests",
                       f"{result.model_b.completed_requests}/{result.model_b.total_requests}")

        console.print(table_b)
        console.print()

        # Comparison Summary
        speedup = result.speedup_factor
        write_speedup = result.write_speedup
        latency_diff = result.latency_improvement

        summary = Table(title="COMPARISON SUMMARY", show_header=True,
                       header_style="bold yellow")
        summary.add_column("Metric", style="cyan", width=22)
        summary.add_column("Value", style="bold", justify="right", width=18)

        # Format speedup with color
        speedup_color = "green" if speedup > 1 else "red" if speedup < 1 else "white"
        summary.add_row("Overall Speedup (B/A)",
                       f"[{speedup_color}]{speedup:.2f}x[/{speedup_color}]")

        write_color = "green" if write_speedup > 1 else "red" if write_speedup < 1 else "white"
        summary.add_row("Generation Speedup (B/A)",
                       f"[{write_color}]{write_speedup:.2f}x[/{write_color}]")

        latency_color = "green" if latency_diff < 0 else "red" if latency_diff > 0 else "white"
        summary.add_row("Latency Change",
                       f"[{latency_color}]{latency_diff:+.1f}%[/{latency_color}]")

        summary.add_row("", "")
        winner_color = "green bold"
        summary.add_row("Winner", f"[{winner_color}]{result.winner}[/{winner_color}]")

        console.print(summary)

    def _display_plain(self, result: ComparisonResult):
        """Display with plain text formatting."""
        print("\n" + "=" * 70)
        print("  THROUGHPUT COMPARISON RESULTS")
        print("=" * 70)

        # Model A
        print(f"\nModel A: {result.model_a.model_name}")
        print(f"  Server: {result.model_a.server_url}")
        print(f"  Samples: {result.model_a.n_samples}")
        print(f"  Wall clock: {result.model_a.wall_clock_seconds:.1f}s")
        print(f"  Throughput:")
        print(f"    Total tok/s:  {result.model_a.tokens_per_second:,.0f}")
        print(f"    Read tok/s:   {result.model_a.read_tokens_per_second:,.0f}")
        print(f"    Write tok/s:  {result.model_a.write_tokens_per_second:,.0f}")
        print(f"  Latency: {result.model_a.avg_latency_ms:,.0f}ms avg")
        print(f"  Requests: {result.model_a.completed_requests}/{result.model_a.total_requests}")

        # Model B
        print(f"\nModel B: {result.model_b.model_name}")
        print(f"  Server: {result.model_b.server_url}")
        print(f"  Samples: {result.model_b.n_samples}")
        print(f"  Wall clock: {result.model_b.wall_clock_seconds:.1f}s")
        print(f"  Throughput:")
        print(f"    Total tok/s:  {result.model_b.tokens_per_second:,.0f}")
        print(f"    Read tok/s:   {result.model_b.read_tokens_per_second:,.0f}")
        print(f"    Write tok/s:  {result.model_b.write_tokens_per_second:,.0f}")
        print(f"  Latency: {result.model_b.avg_latency_ms:,.0f}ms avg")
        print(f"  Requests: {result.model_b.completed_requests}/{result.model_b.total_requests}")

        # Summary
        print("\n" + "-" * 70)
        print("COMPARISON SUMMARY")
        print("-" * 70)
        print(f"  Overall Speedup (B/A):    {result.speedup_factor:.2f}x")
        print(f"  Generation Speedup (B/A): {result.write_speedup:.2f}x")
        print(f"  Latency Change:           {result.latency_improvement:+.1f}%")
        print(f"  Winner: {result.winner}")
        print()


# =============================================================================
# vLLM Server Management
# =============================================================================

def load_model_config(profile: str, config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load model configuration from settings.yaml.

    Args:
        profile: Model profile name (e.g., 'nemotron-30b-fp8')
        config_path: Path to settings.yaml (defaults to config/settings.yaml)

    Returns:
        Dict with model path, tensor_parallel, max_model_len, etc.
    """
    if config_path is None:
        # Find project root
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    vllm_cfg = cfg.get("vllm", {})
    models = vllm_cfg.get("models", {})

    if profile not in models:
        available = list(models.keys())
        raise ValueError(f"Profile '{profile}' not found. Available: {available}")

    model_cfg = models[profile]
    return {
        "profile": profile,
        "path": model_cfg["path"],
        "tensor_parallel": model_cfg.get("tensor_parallel", 1),
        "max_model_len": model_cfg.get("max_model_len", 8192),
        "host": vllm_cfg.get("host", "0.0.0.0"),
        "gpu_memory_utilization": vllm_cfg.get("gpu_memory_utilization", 0.90),
    }


class VLLMServerManager:
    """
    Manage vLLM server lifecycle for benchmarking.

    Handles starting, health checking, and stopping vLLM servers.
    Supports both sequential (one model at a time) and parallel (multiple GPUs) modes.

    Usage (sequential):
        async with VLLMServerManager("nemotron-30b-fp8", port=8000) as server:
            # Server is ready, run benchmark
            result = await benchmark.run_benchmark(samples)
        # Server is automatically stopped

    Usage (parallel on specific GPUs):
        async with VLLMServerManager("model-a", port=8000, cuda_devices="0,1", tensor_parallel=2) as server_a:
            async with VLLMServerManager("model-b", port=8002, cuda_devices="2,3", tensor_parallel=2) as server_b:
                # Both servers running on separate GPU sets
    """

    def __init__(
        self,
        profile: str,
        port: int = 8000,
        host: str = "0.0.0.0",
        venv_path: str = "/home/mlinegar/vllm-env",
        startup_timeout: float = 300.0,  # 5 minutes max for model loading
        health_check_interval: float = 2.0,
        cuda_devices: Optional[str] = None,  # e.g., "0,1" or "2,3"
        tensor_parallel: Optional[int] = None,  # Override config tensor_parallel
    ):
        """
        Initialize server manager.

        Args:
            profile: Model profile name from config/settings.yaml
            port: Port to run the server on
            host: Host to bind to
            venv_path: Path to vLLM virtual environment
            startup_timeout: Max seconds to wait for server startup
            health_check_interval: Seconds between health checks
            cuda_devices: CUDA_VISIBLE_DEVICES string (e.g., "0,1")
            tensor_parallel: Override tensor_parallel from config
        """
        self.profile = profile
        self.port = port
        self.host = host
        self.venv_path = venv_path
        self.startup_timeout = startup_timeout
        self.health_check_interval = health_check_interval
        self.cuda_devices = cuda_devices
        self.tensor_parallel_override = tensor_parallel

        self._process: Optional[subprocess.Popen] = None
        self._config: Optional[Dict[str, Any]] = None
        self._log_file = None

    @property
    def url(self) -> str:
        """Get the server URL."""
        return f"http://localhost:{self.port}/v1"

    @property
    def model_path(self) -> str:
        """Get the model path."""
        return self._config["path"] if self._config else ""

    async def start(self) -> None:
        """Start the vLLM server and wait for it to be ready."""
        # Load config
        self._config = load_model_config(self.profile)

        # Use tensor_parallel override if provided
        tensor_parallel = self.tensor_parallel_override or self._config["tensor_parallel"]

        logger.info(f"Starting vLLM server for {self.profile}")
        logger.info(f"  Model: {self._config['path']}")
        logger.info(f"  Port: {self.port}")
        logger.info(f"  Tensor Parallel: {tensor_parallel}")
        if self.cuda_devices:
            logger.info(f"  CUDA Devices: {self.cuda_devices}")

        # Build command (compatible with vLLM 0.12+)
        python_path = os.path.join(self.venv_path, "bin", "python")
        cmd = [
            python_path,
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", self._config["path"],
            "--host", self.host,
            "--port", str(self.port),
            "--tensor-parallel-size", str(tensor_parallel),
            "--max-model-len", str(self._config["max_model_len"]),
            "--gpu-memory-utilization", str(self._config["gpu_memory_utilization"]),
            "--trust-remote-code",
            "--enforce-eager",
        ]

        # Build environment with optional CUDA device isolation
        env = os.environ.copy()
        if self.cuda_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_devices

        # Create log file for server output to avoid pipe buffer deadlock
        import tempfile
        self._log_file = tempfile.NamedTemporaryFile(
            mode='w', prefix=f'vllm_{self.profile}_', suffix='.log', delete=False
        )
        logger.info(f"  Log file: {self._log_file.name}")

        # Start process with output to log file (avoids PIPE buffer deadlock)
        self._process = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # Create new process group for clean shutdown
            env=env,
        )

        logger.info(f"Server process started (PID: {self._process.pid})")

        # Wait for server to be ready
        await self._wait_for_ready()

    async def _wait_for_ready(self) -> None:
        """Wait for the server to be ready to accept requests."""
        start_time = time.time()
        last_log_time = start_time

        logger.info("Waiting for server to be ready...")

        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < self.startup_timeout:
                # Check if process died
                if self._process.poll() is not None:
                    # Read output from log file
                    try:
                        self._log_file.flush()
                        with open(self._log_file.name, 'r') as f:
                            output = f.read()
                    except Exception:
                        output = "Could not read log file"
                    raise RuntimeError(
                        f"vLLM server exited unexpectedly with code {self._process.returncode}. "
                        f"Output:\n{output[-2000:] if output else 'No output'}"
                    )

                # Try health check
                try:
                    async with session.get(
                        f"http://localhost:{self.port}/v1/models",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            elapsed = time.time() - start_time
                            logger.info(f"Server ready in {elapsed:.1f}s")
                            return
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass

                # Log progress every 30 seconds
                if time.time() - last_log_time > 30:
                    elapsed = time.time() - start_time
                    logger.info(f"Still waiting for server... ({elapsed:.0f}s elapsed)")
                    last_log_time = time.time()

                await asyncio.sleep(self.health_check_interval)

        # Timeout
        self.stop()
        raise TimeoutError(
            f"Server did not become ready within {self.startup_timeout}s"
        )

    def stop(self) -> None:
        """Stop the vLLM server."""
        if self._process is None:
            return

        logger.info(f"Stopping vLLM server (PID: {self._process.pid})")

        try:
            # Send SIGTERM to process group
            os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)

            # Wait for graceful shutdown
            try:
                self._process.wait(timeout=10)
                logger.info("Server stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill
                logger.warning("Server did not stop gracefully, forcing kill")
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                self._process.wait()
        except ProcessLookupError:
            pass  # Already dead
        except Exception as e:
            logger.warning(f"Error stopping server: {e}")

        self._process = None

        # Close log file
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    async def __aenter__(self) -> "VLLMServerManager":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


async def run_sequential_comparison(
    profile_a: str,
    profile_b: str,
    samples: List[Any],
    port: int = 8000,
    max_tokens: int = 500,
    max_concurrent: int = 100,
    chunk_size: int = 2000,
    show_progress: bool = True,
) -> ComparisonResult:
    """
    Run comparison by starting/stopping servers sequentially.

    Since models typically share GPU memory, this runs one model at a time:
    1. Start model A, benchmark, stop
    2. Start model B, benchmark, stop
    3. Compare results

    Args:
        profile_a: Model A profile name from settings.yaml
        profile_b: Model B profile name from settings.yaml
        samples: List of samples to benchmark
        port: Port to use (same for both models)
        max_tokens: Max tokens per response
        max_concurrent: Max concurrent requests
        show_progress: Whether to show progress

    Returns:
        ComparisonResult with both results
    """
    if show_progress:
        print(f"\n{'='*70}")
        print("  SEQUENTIAL THROUGHPUT BENCHMARK")
        print(f"{'='*70}\n")

    # Benchmark Model A
    if show_progress:
        print(f"[1/2] Benchmarking {profile_a}...")

    async with VLLMServerManager(profile_a, port=port) as server_a:
        benchmark_a = ThroughputBenchmark(
            model_name=profile_a,
            server_url=server_a.url,
            max_concurrent_requests=max_concurrent,
            chunk_size=chunk_size,
        )
        result_a = await benchmark_a.run_benchmark(samples, max_tokens, show_progress)

    if show_progress:
        print(f"  Completed in {result_a.wall_clock_seconds:.1f}s "
              f"({result_a.tokens_per_second:.0f} tok/s)\n")

    # Benchmark Model B
    if show_progress:
        print(f"[2/2] Benchmarking {profile_b}...")

    async with VLLMServerManager(profile_b, port=port) as server_b:
        benchmark_b = ThroughputBenchmark(
            model_name=profile_b,
            server_url=server_b.url,
            max_concurrent_requests=max_concurrent,
            chunk_size=chunk_size,
        )
        result_b = await benchmark_b.run_benchmark(samples, max_tokens, show_progress)

    if show_progress:
        print(f"  Completed in {result_b.wall_clock_seconds:.1f}s "
              f"({result_b.tokens_per_second:.0f} tok/s)\n")

    return ComparisonResult(model_a=result_a, model_b=result_b)


async def run_parallel_comparison(
    profile_a: str,
    profile_b: str,
    samples: List[Any],
    port_a: int = 8000,
    port_b: int = 8002,
    cuda_devices_a: str = "0,1",
    cuda_devices_b: str = "2,3",
    tensor_parallel: int = 2,
    max_tokens: int = 500,
    max_concurrent: int = 100,
    chunk_size: int = 2000,
    show_progress: bool = True,
) -> ComparisonResult:
    """
    Run comparison with both models in parallel on separate GPU sets.

    This starts both vLLM servers simultaneously on different GPUs and runs
    the benchmarks concurrently. More efficient than sequential but requires
    enough GPUs (typically 4+ GPUs for 2 models with tensor_parallel=2 each).

    Args:
        profile_a: Model A profile name from settings.yaml
        profile_b: Model B profile name from settings.yaml
        samples: List of samples to benchmark
        port_a: Port for model A server
        port_b: Port for model B server
        cuda_devices_a: CUDA devices for model A (e.g., "0,1")
        cuda_devices_b: CUDA devices for model B (e.g., "2,3")
        tensor_parallel: Tensor parallel size for both models
        max_tokens: Max tokens per response
        max_concurrent: Max concurrent requests per model
        chunk_size: Characters per chunk
        show_progress: Whether to show progress

    Returns:
        ComparisonResult with both results
    """
    if show_progress:
        print(f"\n{'='*70}")
        print("  PARALLEL THROUGHPUT BENCHMARK")
        print(f"{'='*70}")
        print(f"  Model A: {profile_a} on GPUs {cuda_devices_a} (port {port_a})")
        print(f"  Model B: {profile_b} on GPUs {cuda_devices_b} (port {port_b})")
        print(f"  Tensor Parallel: {tensor_parallel}")
        print(f"{'='*70}\n")

    # Start both servers concurrently
    server_a = VLLMServerManager(
        profile_a,
        port=port_a,
        cuda_devices=cuda_devices_a,
        tensor_parallel=tensor_parallel,
    )
    server_b = VLLMServerManager(
        profile_b,
        port=port_b,
        cuda_devices=cuda_devices_b,
        tensor_parallel=tensor_parallel,
    )

    if show_progress:
        print("Starting both vLLM servers in parallel...")

    # Start servers concurrently
    try:
        await asyncio.gather(server_a.start(), server_b.start())
    except Exception as e:
        logger.error(f"Failed to start servers: {e}")
        # Try to get any output from failed servers
        for name, server in [("A", server_a), ("B", server_b)]:
            if server._process and server._process.poll() is not None:
                try:
                    stdout = server._process.stdout.read() if server._process.stdout else ""
                    logger.error(f"Server {name} output:\n{stdout[-3000:] if stdout else 'No output'}")
                except:
                    pass
        server_a.stop()
        server_b.stop()
        raise

    if show_progress:
        print("Both servers ready. Running benchmarks in parallel...\n")

    try:
        # Create benchmarks
        benchmark_a = ThroughputBenchmark(
            model_name=profile_a,
            server_url=server_a.url,
            max_concurrent_requests=max_concurrent,
            chunk_size=chunk_size,
        )
        benchmark_b = ThroughputBenchmark(
            model_name=profile_b,
            server_url=server_b.url,
            max_concurrent_requests=max_concurrent,
            chunk_size=chunk_size,
        )

        # Run benchmarks concurrently
        result_a, result_b = await asyncio.gather(
            benchmark_a.run_benchmark(samples, max_tokens, show_progress),
            benchmark_b.run_benchmark(samples, max_tokens, show_progress),
        )

        if show_progress:
            print(f"\n{profile_a}: {result_a.wall_clock_seconds:.1f}s "
                  f"({result_a.tokens_per_second:.0f} tok/s)")
            print(f"{profile_b}: {result_b.wall_clock_seconds:.1f}s "
                  f"({result_b.tokens_per_second:.0f} tok/s)\n")

        return ComparisonResult(model_a=result_a, model_b=result_b)

    finally:
        # Always stop both servers
        if show_progress:
            print("Stopping servers...")
        server_a.stop()
        server_b.stop()


def save_results(result: ComparisonResult, output_path: Path):
    """Save comparison results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"Results saved to {output_path}")
