#!/usr/bin/env python3
"""
Simple test script for quantized models.

Tests that a model can be loaded and generate text with vLLM.

Usage:
    # Test the Nano model
    python scripts/test_quantized_model.py /mnt/data/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 --port 8001

    # Test GenRM
    python scripts/test_quantized_model.py /mnt/data/models/nvidia/Qwen3-Nemotron-235B-A22B-GenRM-NVFP4 --port 8001 --tp 4
"""

import argparse
import subprocess
import sys
import time
import signal
import json
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def wait_for_server(url: str, timeout: int = 300) -> bool:
    """Wait for vLLM server to be ready."""
    print(f"Waiting for server at {url}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = Request(f"{url}/health")
            with urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print("Server ready!")
                    return True
        except (URLError, HTTPError, TimeoutError):
            pass
        time.sleep(2)
        elapsed = int(time.time() - start)
        if elapsed % 30 == 0:
            print(f"  Still waiting... ({elapsed}s)")
    return False


def get_model_name(url: str) -> str:
    """Get the model name from the server."""
    req = Request(f"{url}/models")
    with urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
        if data.get("data"):
            return data["data"][0]["id"]
    raise RuntimeError("No models found on server")


def test_generation(url: str, model_name: str) -> dict:
    """Test basic text generation."""
    prompt = "What is 2 + 2? Answer with just the number:"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 32,
        "temperature": 0.0,
    }

    req = Request(
        f"{url}/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def test_chat(url: str, model_name: str) -> dict:
    """Test chat completion."""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Say 'Hello World' and nothing else."}
        ],
        "max_tokens": 32,
        "temperature": 0.0,
    }

    req = Request(
        f"{url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def main():
    parser = argparse.ArgumentParser(description="Test a quantized model with vLLM")
    parser.add_argument("model_path", help="Path to the quantized model")
    parser.add_argument("--port", type=int, default=8001, help="Port for vLLM server")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95, help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Max model length")
    parser.add_argument("--no-start-server", action="store_true", help="Skip starting server (use existing)")
    parser.add_argument("--keep-server", action="store_true", help="Keep server running after test")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}/v1"
    server_proc = None

    try:
        if not args.no_start_server:
            print(f"\n{'='*60}")
            print(f"Testing: {args.model_path}")
            print(f"{'='*60}\n")

            cmd = [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", args.model_path,
                "--port", str(args.port),
                "--tensor-parallel-size", str(args.tp),
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                "--max-model-len", str(args.max_model_len),
                "--trust-remote-code",
                "--dtype", "auto",
            ]

            print(f"Starting vLLM server...")
            print(f"Command: {' '.join(cmd)}\n")

            server_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Wait for server
            if not wait_for_server(url, timeout=300):
                print("\nERROR: Server failed to start!")
                # Print server output
                if server_proc.stdout:
                    print("\nServer output:")
                    for line in server_proc.stdout:
                        print(line, end="")
                        if server_proc.poll() is not None:
                            break
                return 1

        # Get model name
        print("\nFetching model info...")
        model_name = get_model_name(url)
        print(f"Model: {model_name}")

        # Test completion
        print("\n--- Testing Completion API ---")
        print("Prompt: 'What is 2 + 2? Answer with just the number:'")
        result = test_generation(url, model_name)
        text = result["choices"][0]["text"].strip()
        print(f"Response: {text}")

        if "4" in text:
            print("✓ Math test PASSED")
        else:
            print("✗ Math test FAILED (expected '4' in response)")

        # Test chat
        print("\n--- Testing Chat API ---")
        print("Message: 'Say Hello World and nothing else.'")
        result = test_chat(url, model_name)
        text = result["choices"][0]["message"]["content"].strip()
        print(f"Response: {text}")

        if "hello" in text.lower():
            print("✓ Chat test PASSED")
        else:
            print("✗ Chat test FAILED (expected 'hello' in response)")

        print(f"\n{'='*60}")
        print("All basic tests completed!")
        print(f"{'='*60}\n")

        if args.keep_server:
            print(f"Server running at {url}")
            print("Press Ctrl+C to stop...")
            try:
                server_proc.wait()
            except KeyboardInterrupt:
                pass

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        if server_proc and not args.keep_server:
            print("\nShutting down server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    sys.exit(main())
