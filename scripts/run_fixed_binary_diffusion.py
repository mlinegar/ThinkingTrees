#!/usr/bin/env python3
"""Run the standalone fixed-binary diffusion prototype against a selected backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diffusion.backends import build_diffusion_backend
from src.diffusion.tree_engine import FixedBinaryDiffusionTreeEngine


def _chunk_text(text: str, chunk_size: int) -> List[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return [stripped[index:index + chunk_size] for index in range(0, len(stripped), chunk_size)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fixed-binary diffusion prototype.")
    parser.add_argument("--engine", type=str, default="sglang", help="Engine adapter: sglang | vllm_omni | custom_http.")
    parser.add_argument("--backend", type=str, default="", help="Deprecated alias for --engine.")
    parser.add_argument("--text", type=str, default="", help="Inline document text.")
    parser.add_argument("--text-file", type=Path, help="Path to a text file to summarize.")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Leaf chunk size in characters.")
    parser.add_argument("--rounds", type=int, default=0, help="Number of root refinement rounds.")
    parser.add_argument("--rubric", type=str, default="Preserve theorem-relevant content.", help="Preservation rubric.")
    parser.add_argument("--base-url", type=str, default="", help="Optional backend base URL override.")
    parser.add_argument("--generate-path", type=str, default="/generate", help="Generation path for generic HTTP-style backends.")
    parser.add_argument("--model", type=str, default="", help="Optional served model name.")
    parser.add_argument("--engine-options", type=str, default="{}", help="JSON object forwarded as generic engine options.")
    parser.add_argument("--dllm-algorithm", type=str, default="LowConfidence", help="Optional backend algorithm selector.")
    parser.add_argument(
        "--dllm-algorithm-config",
        type=str,
        default="{}",
        help="JSON object passed through as backend algorithm config.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens per generation.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    text = args.text
    if args.text_file is not None:
        text = args.text_file.read_text(encoding="utf-8")
    if not text.strip():
        raise SystemExit("Provide --text or --text-file.")

    chunks = _chunk_text(text, args.chunk_size)
    if not chunks:
        raise SystemExit("Input text produced no chunks.")

    selected_engine = args.backend or args.engine or "sglang"
    engine_options = json.loads(args.engine_options)
    if args.dllm_algorithm:
        engine_options.setdefault("dllm_algorithm", args.dllm_algorithm)
    dllm_algorithm_config = json.loads(args.dllm_algorithm_config)
    if dllm_algorithm_config:
        engine_options.setdefault("dllm_algorithm_config", dllm_algorithm_config)

    backend = build_diffusion_backend(
        selected_engine,
        base_url=args.base_url or None,
        model=args.model or None,
        generate_path=args.generate_path,
    )
    engine = FixedBinaryDiffusionTreeEngine(backend)
    result = engine.run_fixed_tree(
        chunks,
        rubric=args.rubric,
        refine_rounds=args.rounds,
        sampling_params={
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        },
        engine_options=engine_options,
    )

    payload = result.to_dict()
    payload["input_chunk_count"] = len(chunks)
    payload["input_chunks"] = chunks

    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
