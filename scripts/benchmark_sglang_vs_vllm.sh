#!/bin/bash
# Benchmark SGLang vs vLLM for ThinkingTrees workloads.
#
# Compares throughput, prefix cache hit rate, and latency between backends
# on identical workloads. Results are saved to benchmarks/ directory.
#
# Prerequisites:
#   - vLLM server running on port 8000 (or specify --vllm-port)
#   - SGLang server running on port 30000 (or specify --sglang-port)
#
# Usage:
#   ./scripts/benchmark_sglang_vs_vllm.sh
#   ./scripts/benchmark_sglang_vs_vllm.sh --docs 50 --concurrency 100
#   ./scripts/benchmark_sglang_vs_vllm.sh --vllm-only
#   ./scripts/benchmark_sglang_vs_vllm.sh --sglang-only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="$PROJECT_ROOT/benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$BENCHMARK_DIR/$TIMESTAMP"

# Defaults
VLLM_PORT=8000
SGLANG_PORT=30000
NUM_DOCS=20
CONCURRENCY=50
RUN_VLLM=true
RUN_SGLANG=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vllm-port) VLLM_PORT="$2"; shift 2 ;;
        --sglang-port) SGLANG_PORT="$2"; shift 2 ;;
        --docs) NUM_DOCS="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --vllm-only) RUN_SGLANG=false; shift ;;
        --sglang-only) RUN_VLLM=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "ThinkingTrees Backend Benchmark"
echo "=========================================="
echo "Documents: $NUM_DOCS"
echo "Concurrency: $CONCURRENCY"
echo "vLLM port: $VLLM_PORT (run=$RUN_VLLM)"
echo "SGLang port: $SGLANG_PORT (run=$RUN_SGLANG)"
echo "Results: $RESULTS_DIR"
echo "=========================================="

# Check server health
check_server() {
    local port=$1
    local name=$2
    if curl -sf "http://localhost:$port/v1/models" > /dev/null 2>&1; then
        echo "[OK] $name server on port $port is healthy"
        return 0
    else
        echo "[SKIP] $name server on port $port is not reachable"
        return 1
    fi
}

# Collect Prometheus metrics from server
collect_metrics() {
    local port=$1
    local output=$2
    curl -sf "http://localhost:$port/metrics" > "$output" 2>/dev/null || true
}

# Run benchmark via Python
run_benchmark() {
    local backend=$1
    local port=$2
    local output_file=$3

    echo ""
    echo "--- Running $backend benchmark (port=$port, docs=$NUM_DOCS, concurrency=$CONCURRENCY) ---"

    # Collect pre-run metrics
    collect_metrics "$port" "$RESULTS_DIR/${backend}_metrics_pre.txt"

    python3 - "$port" "$NUM_DOCS" "$CONCURRENCY" "$output_file" "$backend" <<'PYEOF'
import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

port = int(sys.argv[1])
num_docs = int(sys.argv[2])
concurrency = int(sys.argv[3])
output_file = sys.argv[4]
backend_name = sys.argv[5]

async def run_benchmark():
    import aiohttp

    base_url = f"http://localhost:{port}/v1/chat/completions"
    rubric = "Preserve all named entities, dates, and numerical values exactly."

    # Generate synthetic documents of varying sizes
    docs = []
    for i in range(num_docs):
        size = 200 + (i * 50) % 800  # 200-1000 words
        text = f"Document {i}: " + " ".join(
            [f"word_{j}" for j in range(size)]
        )
        docs.append(text)

    results = {
        "backend": backend_name,
        "port": port,
        "num_docs": num_docs,
        "concurrency": concurrency,
        "requests": [],
    }

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    errors = 0

    async def send_request(session, doc_text, doc_idx):
        nonlocal completed, total_prompt_tokens, total_completion_tokens, errors
        async with sem:
            payload = {
                "model": "default",
                "messages": [
                    {"role": "system", "content": f"You are a careful text summarizer.\nPreservation rubric:\n{rubric}"},
                    {"role": "user", "content": f"{doc_text}\n\nReturn ONLY the summary text."},
                ],
                "max_tokens": 512,
                "temperature": 0.0,
            }
            start = time.monotonic()
            try:
                async with session.post(base_url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    data = await resp.json()
                    elapsed = time.monotonic() - start
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    completed += 1
                    results["requests"].append({
                        "doc_idx": doc_idx,
                        "latency_s": round(elapsed, 3),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "status": resp.status,
                    })
            except Exception as e:
                elapsed = time.monotonic() - start
                errors += 1
                results["requests"].append({
                    "doc_idx": doc_idx,
                    "latency_s": round(elapsed, 3),
                    "error": str(e),
                })

    # Run all requests
    wall_start = time.monotonic()
    async with aiohttp.ClientSession() as session:
        # Phase 1: Leaf summarization (all docs)
        tasks = [send_request(session, doc, i) for i, doc in enumerate(docs)]
        await asyncio.gather(*tasks)

        # Phase 2: Merge pairs (simulate merge layer)
        merge_tasks = []
        for i in range(0, len(docs) - 1, 2):
            merged = f"PART A:\n{docs[i][:200]}\n\nPART B:\n{docs[i+1][:200]}"
            merge_tasks.append(send_request(session, merged, f"merge_{i}_{i+1}"))
        await asyncio.gather(*merge_tasks)

    wall_elapsed = time.monotonic() - wall_start

    # Compute statistics
    latencies = [r["latency_s"] for r in results["requests"] if "error" not in r]
    latencies.sort()

    results["summary"] = {
        "wall_clock_s": round(wall_elapsed, 2),
        "total_requests": len(results["requests"]),
        "completed": completed,
        "errors": errors,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "tokens_per_second": round((total_prompt_tokens + total_completion_tokens) / wall_elapsed, 1) if wall_elapsed > 0 else 0,
        "p50_latency_s": round(latencies[len(latencies)//2], 3) if latencies else 0,
        "p99_latency_s": round(latencies[int(len(latencies)*0.99)], 3) if latencies else 0,
        "mean_latency_s": round(sum(latencies)/len(latencies), 3) if latencies else 0,
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    s = results["summary"]
    print(f"\n  {backend_name} Results:")
    print(f"    Wall clock: {s['wall_clock_s']}s")
    print(f"    Requests: {s['completed']}/{s['total_requests']} (errors={s['errors']})")
    print(f"    Throughput: {s['tokens_per_second']} tok/s")
    print(f"    Latency P50={s['p50_latency_s']}s P99={s['p99_latency_s']}s mean={s['mean_latency_s']}s")

asyncio.run(run_benchmark())
PYEOF

    # Collect post-run metrics
    collect_metrics "$port" "$RESULTS_DIR/${backend}_metrics_post.txt"
    echo "  Results saved to $output_file"
}

# Run benchmarks
if $RUN_VLLM; then
    if check_server "$VLLM_PORT" "vLLM"; then
        run_benchmark "vllm" "$VLLM_PORT" "$RESULTS_DIR/vllm_results.json"
    fi
fi

if $RUN_SGLANG; then
    if check_server "$SGLANG_PORT" "SGLang"; then
        run_benchmark "sglang" "$SGLANG_PORT" "$RESULTS_DIR/sglang_results.json"
    fi
fi

# Compare results if both ran
if [[ -f "$RESULTS_DIR/vllm_results.json" && -f "$RESULTS_DIR/sglang_results.json" ]]; then
    echo ""
    echo "=========================================="
    echo "Comparison"
    echo "=========================================="
    python3 - "$RESULTS_DIR" <<'PYEOF'
import json
import sys

results_dir = sys.argv[1]
with open(f"{results_dir}/vllm_results.json") as f:
    vllm = json.load(f)["summary"]
with open(f"{results_dir}/sglang_results.json") as f:
    sglang = json.load(f)["summary"]

print(f"{'Metric':<25} {'vLLM':>12} {'SGLang':>12} {'Ratio':>10}")
print("-" * 62)
for key in ["wall_clock_s", "tokens_per_second", "p50_latency_s", "p99_latency_s", "mean_latency_s"]:
    v = vllm.get(key, 0)
    s = sglang.get(key, 0)
    ratio = s / v if v > 0 else 0
    better = "SGLang" if (ratio < 1 and "latency" in key) or (ratio > 1 and "tokens" in key) else "vLLM"
    # For wall_clock, lower is better
    if key == "wall_clock_s":
        better = "SGLang" if ratio < 1 else "vLLM"
    print(f"  {key:<23} {v:>12} {s:>12} {ratio:>9.2f}x  ({better})")

print()
throughput_ratio = sglang["tokens_per_second"] / vllm["tokens_per_second"] if vllm["tokens_per_second"] > 0 else 0
if throughput_ratio > 1.2:
    print(f"RECOMMENDATION: SGLang is {throughput_ratio:.1f}x faster. Consider migrating.")
elif throughput_ratio < 0.8:
    print(f"RECOMMENDATION: vLLM is {1/throughput_ratio:.1f}x faster. Keep vLLM as primary.")
else:
    print("RECOMMENDATION: Performance is comparable. Keep vLLM (better GPU orchestration support).")
PYEOF
fi

echo ""
echo "Benchmark complete. Results in: $RESULTS_DIR"
