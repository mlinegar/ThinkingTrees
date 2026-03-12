#!/usr/bin/env python3
"""
CLI entry point for the ThinkingTrees audit harness.

Usage (with running server):
    python scripts/run_harness.py \
        --documents data/corpus/*.txt \
        --oracle-module my_project.oracle:score_fn \
        --llm-endpoint http://localhost:8000/v1 \
        --context-limit 32768 \
        --delta 0.05 \
        --epsilon 0.10 \
        --output-dir outputs/audit_20260217

Usage (auto-start vLLM):
    python scripts/run_harness.py \
        --documents data/corpus/*.txt \
        --oracle-module my_project.oracle:score_fn \
        --start-server nemotron-30b-nvfp4 \
        --output-dir outputs/audit_20260217

Usage (auto-start SGLang):
    python scripts/run_harness.py \
        --documents data/corpus/*.txt \
        --oracle-module my_project.oracle:score_fn \
        --start-server nemotron-30b-nvfp4 --server-type sglang \
        --output-dir outputs/audit_20260217

The oracle module is loaded via Python import path: "module.path:function_name".
The function must accept a single string and return a float in [0, 1].
"""

import argparse
import asyncio
import importlib
import json
import logging
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.harness import TreeAudit, AuditBudget


def load_oracle(spec: str):
    """Load oracle function from 'module.path:function_name' spec."""
    if ":" not in spec:
        raise ValueError(
            f"Oracle spec must be 'module.path:function_name', got: {spec!r}"
        )
    module_path, func_name = spec.rsplit(":", 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, func_name)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn


def load_documents(paths: list[str]) -> list[str]:
    """Load document texts from file paths."""
    docs = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            logging.warning("File not found, skipping: %s", p)
            continue
        docs.append(path.read_text(encoding="utf-8"))
    return docs


def _run_with_server(audit, documents, args):
    """Run the harness with an auto-started vLLM or SGLang server."""
    server_type = getattr(args, "server_type", "vllm")

    async def _inner():
        if server_type == "sglang":
            from src.benchmark.throughput import SGLangServerManager
            manager = SGLangServerManager(
                profile=args.start_server,
                port=args.server_port,
                cuda_devices=args.cuda_devices,
            )
        else:
            from src.benchmark.throughput import VLLMServerManager
            manager = VLLMServerManager(
                profile=args.start_server,
                port=args.server_port,
                cuda_devices=args.cuda_devices,
            )

        async with manager as server:
            logging.info(
                "%s server started: %s on port %d",
                server_type.upper(), args.start_server, args.server_port,
            )
            return await audit.run(documents)

    return asyncio.run(_inner())


def main():
    parser = argparse.ArgumentParser(
        description="ThinkingTrees audit harness: build trees, audit, emit certificate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--documents", nargs="+", required=True,
        help="Paths to document text files (supports shell glob).",
    )
    parser.add_argument(
        "--oracle-module", required=True,
        help="Python import path for oracle: 'module.path:function_name'.",
    )
    parser.add_argument(
        "--llm-endpoint", default="http://localhost:8000/v1",
        help="OpenAI-compatible LLM endpoint URL (vLLM, SGLang, OpenAI, etc.).",
    )
    parser.add_argument(
        "--model", default="default",
        help="Model name (auto-detected from endpoint if 'default').",
    )
    parser.add_argument(
        "--api-key", default="EMPTY",
        help="API key for the LLM endpoint. 'EMPTY' for local vLLM/SGLang servers, "
             "or a real key for OpenAI/Anthropic. Also reads OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--context-limit", type=int, default=32768,
        help="Model context window in tokens.",
    )
    parser.add_argument(
        "--delta", type=float, default=0.05,
        help="Confidence parameter (certificate holds with prob >= 1 - delta).",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.10,
        help="Target violation bound (drives sample budget).",
    )
    parser.add_argument(
        "--sample-budget", type=int, default=20,
        help="Max audit nodes per tree (overridden by epsilon/delta if set).",
    )
    parser.add_argument(
        "--chunk-chars", type=int, default=2000,
        help="Target characters per leaf chunk.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--rubric", default="",
        help="Task rubric for summarization.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for output artifacts.",
    )
    parser.add_argument(
        "--no-idempotence", action="store_true",
        help="Skip idempotence checks (faster).",
    )
    parser.add_argument(
        "--no-substitution", action="store_true",
        help="Skip substitution checks (faster).",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=200,
        help="Max in-flight requests (200 for local vLLM, ~20 for APIs).",
    )
    parser.add_argument(
        "--start-server", metavar="PROFILE",
        help="Auto-start a server from a settings.yaml profile name "
             "(e.g. 'nemotron-30b-nvfp4'). Stopped on exit.",
    )
    parser.add_argument(
        "--server-type", choices=["vllm", "sglang"], default="vllm",
        help="Server backend for --start-server (default: vllm).",
    )
    parser.add_argument(
        "--server-port", type=int, default=None,
        help="Port for auto-started server (default: 8000 for vLLM, 30000 for SGLang).",
    )
    parser.add_argument(
        "--cuda-devices", default=None,
        help="CUDA_VISIBLE_DEVICES for auto-started server (e.g. '0,1').",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Resolve API key: explicit flag > env var > default
    if args.api_key == "EMPTY":
        env_key = os.environ.get("OPENAI_API_KEY", "")
        if env_key:
            args.api_key = env_key

    # Validate: warn about server flags without --start-server
    if not args.start_server:
        if args.server_type != "vllm":
            logging.warning("--server-type has no effect without --start-server")
        if args.server_port is not None:
            logging.warning("--server-port has no effect without --start-server")
        if args.cuda_devices is not None:
            logging.warning("--cuda-devices has no effect without --start-server")

    # Load oracle
    oracle_fn = load_oracle(args.oracle_module)
    logging.info("Loaded oracle: %s", args.oracle_module)

    # Load documents
    documents = load_documents(args.documents)
    if not documents:
        logging.error("No documents loaded.")
        sys.exit(1)
    logging.info("Loaded %d documents.", len(documents))

    # Determine endpoint
    if args.start_server:
        if args.server_port is None:
            args.server_port = 30000 if args.server_type == "sglang" else 8000
        llm_endpoint = f"http://localhost:{args.server_port}/v1"
    else:
        llm_endpoint = args.llm_endpoint

    # Create harness
    audit = TreeAudit(
        llm_endpoint=llm_endpoint,
        oracle=oracle_fn,
        context_limit=args.context_limit,
        budget=AuditBudget(
            delta=args.delta,
            epsilon=args.epsilon,
            sample_budget=args.sample_budget,
            audit_idempotence=not args.no_idempotence,
            audit_substitution=not args.no_substitution,
        ),
        rubric=args.rubric,
        model=args.model,
        api_key=args.api_key,
        chunk_chars=args.chunk_chars,
        max_concurrent=args.max_concurrent,
        seed=args.seed,
    )

    # Run (with optional auto-server-start)
    if args.start_server:
        result = _run_with_server(audit, documents, args)
    else:
        result = audit.run_sync(documents)

    # Save artifacts
    result.save(args.output_dir)
    logging.info("Artifacts saved to %s", args.output_dir)

    # Print certificate summary
    cert = result.certificate
    print("\n--- Audit Certificate ---")
    print(f"Guarantee level : {cert.guarantee_level}")
    print(f"Violation bound : {cert.violation_bound:.6f}")
    print(f"Confidence      : {cert.confidence:.2%}")
    print(f"CI              : [{cert.ci_low:.6f}, {cert.ci_high:.6f}]")
    print(f"Documents       : {cert.n_documents}")
    print(f"Nodes audited   : {cert.n_nodes_audited}")
    print(f"Effective N     : {cert.effective_sample_size:.1f}")
    print(f"Preferences     : {len(result.preferences)}")
    print(f"Output          : {args.output_dir}/certificate.json")


if __name__ == "__main__":
    main()
