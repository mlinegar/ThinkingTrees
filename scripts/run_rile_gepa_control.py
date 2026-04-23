#!/usr/bin/env python3
"""Safely rerun the exact historical RILE GEPA control.

This refuses to run unless the OpenAI-compatible endpoint advertises the exact
model used by the historical control. It prevents accidental substitutions from
being recorded as a reproduction.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_DATASET = Path("outputs/manifesto_nested_20260417_045842/text_pairs_v1_200")
DEFAULT_MODEL = "google/gemma-4-31B-it"
DEFAULT_API_BASE = "http://localhost:8005/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--prepared-dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _models_url(api_base: str) -> str:
    return api_base.rstrip("/") + "/models"


def advertised_models(api_base: str, timeout_seconds: float) -> list[str]:
    try:
        with urllib.request.urlopen(_models_url(api_base), timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"Endpoint {api_base!r} is not available or did not return model JSON: {exc}"
        ) from exc
    return [str(item.get("id", "")) for item in payload.get("data", []) if item.get("id")]


def build_command(args: argparse.Namespace) -> list[str]:
    params = {
        "prepared_dataset_path": str(args.prepared_dataset_path),
        "api_base": str(args.api_base),
        "model_name": str(args.model_name),
        "temperature": 0.0,
        "max_tokens": 1024,
        "optimizer": "gepa",
        "gepa_auto": "light",
        "gepa_num_threads": 8,
        "gepa_reflection_minibatch_size": 3,
        "gepa_valset_cap": 48,
        "reflection_max_tokens": 16384,
        "max_train_examples": 0,
        "n_val": 0,
    }
    return [
        str(args.python),
        "parallel/unified_g_v1/scripts/run_unified_g_bundle.py",
        "run",
        "--approach",
        "dspy_rile",
        "--output-root",
        str(args.output_root),
        "--params-json",
        json.dumps(params),
    ]


def main() -> int:
    args = parse_args()
    if args.dry_run:
        try:
            models = advertised_models(args.api_base, args.timeout_seconds)
            endpoint_available = True
        except SystemExit as exc:
            models = []
            endpoint_available = False
            endpoint_error = str(exc)
        else:
            endpoint_error = ""
        print(
            json.dumps(
                {
                    "endpoint_available": endpoint_available,
                    "endpoint_error": endpoint_error,
                    "exact_model_available": args.model_name in models,
                    "models": models,
                    "command": build_command(args),
                },
                indent=2,
            )
        )
        return 0

    models = advertised_models(args.api_base, args.timeout_seconds)
    if args.model_name not in models:
        raise SystemExit(
            "Refusing to rerun RILE GEPA control: endpoint "
            f"{args.api_base!r} advertises {models}, not exact model "
            f"{args.model_name!r}."
        )
    command = build_command(args)
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
