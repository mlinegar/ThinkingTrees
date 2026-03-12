#!/usr/bin/env python3
"""Train an embedding ridge proxy (synthetic oracle) from LawStress records."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Add project root for direct script execution.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.tasks.manifesto.lawstress_generator import load_lawstress_records_jsonl
from src.tasks.manifesto.lawstress_proxy import (
    build_proxy_training_examples,
    evaluate_embedding_proxy,
)
from src.training.embedding_proxy import VLLMEmbeddingClient, fit_embedding_ridge_proxy


LOGGER = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LawStress embedding proxy (ridge head).")
    parser.add_argument("--records", type=Path, required=True, help="Path to lawstress_records.jsonl")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default timestamped path)")

    parser.add_argument("--embedding-url", type=str, default="http://localhost:8003/v1")
    parser.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--embedding-api-key", type=str, default="EMPTY")
    parser.add_argument("--embedding-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--embedding-batch-size", type=int, default=32)

    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--model-id", type=str, default="lawstress_embedding_ridge_proxy_v1")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _save_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / "lawstress_proxy" / f"run_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_lawstress_records_jsonl(args.records)
    train_records = [r for r in records if r.split == "train"]
    val_records = [r for r in records if r.split == "val"]

    if not train_records:
        raise SystemExit("No train split records found in provided file.")
    if not val_records:
        LOGGER.warning("No val split records found; evaluation will be empty.")

    train_examples = build_proxy_training_examples(train_records)
    val_examples = build_proxy_training_examples(val_records)

    embedding_client = VLLMEmbeddingClient(
        api_base=str(args.embedding_url),
        model=str(args.embedding_model),
        api_key=str(args.embedding_api_key),
        timeout_seconds=float(args.embedding_timeout_seconds),
        batch_size=int(args.embedding_batch_size),
    )

    LOGGER.info("Fitting ridge proxy: train_examples=%d ridge_lambda=%.3g", len(train_examples), float(args.ridge_lambda))
    model = fit_embedding_ridge_proxy(
        train_examples,
        embedding_client=embedding_client,
        ridge_lambda=float(args.ridge_lambda),
        model_id=str(args.model_id),
    )

    model_path = output_dir / "embedding_proxy.json"
    model.save_json(model_path)

    metrics = evaluate_embedding_proxy(
        model,
        embedding_client=embedding_client,
        eval_examples=val_examples,
    )

    metrics_path = output_dir / "proxy_metrics.json"
    _save_json(metrics_path, metrics)

    manifest = {
        "created_at": datetime.utcnow().isoformat(),
        "records_path": str(Path(args.records)),
        "output_dir": str(output_dir),
        "embedding_url": str(args.embedding_url),
        "embedding_model": str(args.embedding_model),
        "ridge_lambda": float(args.ridge_lambda),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "paths": {
            "model": str(model_path),
            "metrics": str(metrics_path),
        },
        "metrics_overall": metrics.get("overall"),
    }
    _save_json(output_dir / "manifest.json", manifest)

    LOGGER.info("Saved model: %s", model_path)
    LOGGER.info("Saved metrics: %s", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
