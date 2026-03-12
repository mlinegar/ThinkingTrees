#!/usr/bin/env python3
"""Diagnose local Qwen3.5 small-model downloads and env compatibility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


REQUIRED_MODELS = (
    "Qwen3.5-0.8B",
    "Qwen3.5-0.8B-Base",
    "Qwen3.5-2B",
    "Qwen3.5-2B-Base",
    "Qwen3.5-4B",
    "Qwen3.5-4B-Base",
)


def gib(n_bytes: int) -> float:
    return n_bytes / (1024 ** 3)


def iter_weight_files(model_dir: Path) -> Iterable[Path]:
    sharded = sorted(model_dir.glob("model.safetensors-*-of-*.safetensors"))
    if sharded:
        return sharded
    single = model_dir / "model.safetensors"
    return [single] if single.exists() else []


def check_downloads(model_root: Path, model_names: Iterable[str]) -> list[dict]:
    rows: list[dict] = []
    for name in model_names:
        model_dir = model_root / name
        exists = model_dir.exists()
        weight_files = list(iter_weight_files(model_dir)) if exists else []
        weight_bytes = sum(f.stat().st_size for f in weight_files)
        row = {
            "name": name,
            "exists": exists,
            "path": str(model_dir),
            "weight_file_count": len(weight_files),
            "weight_gib": round(gib(weight_bytes), 2),
            "has_config": (model_dir / "config.json").exists() if exists else False,
            "has_tokenizer": (model_dir / "tokenizer.json").exists() if exists else False,
            "has_index": (model_dir / "model.safetensors.index.json").exists() if exists else False,
            "incomplete_artifacts": len(list(model_dir.rglob("*.incomplete"))) if exists else 0,
            "model_type": None,
            "arch": None,
            "transformers_config_ok": None,
            "transformers_config_error": None,
        }
        if row["has_config"]:
            try:
                cfg = json.loads((model_dir / "config.json").read_text())
                row["model_type"] = cfg.get("model_type")
                archs = cfg.get("architectures") or []
                row["arch"] = archs[0] if archs else None
            except Exception as exc:  # pragma: no cover - best-effort diagnostics
                row["model_type"] = f"parse_error:{exc.__class__.__name__}"
        rows.append(row)
    return rows


def check_transformers_config_load(rows: list[dict]) -> tuple[str, str]:
    try:
        from transformers import AutoConfig  # type: ignore
        import transformers  # type: ignore
    except Exception as exc:
        version = "not-importable"
        err = f"{exc.__class__.__name__}: {exc}"
        for row in rows:
            row["transformers_config_ok"] = False
            row["transformers_config_error"] = err
        return version, err

    version = getattr(transformers, "__version__", "unknown")
    last_err = ""
    for row in rows:
        if not row["exists"]:
            row["transformers_config_ok"] = False
            row["transformers_config_error"] = "model directory missing"
            continue
        try:
            AutoConfig.from_pretrained(row["path"], trust_remote_code=True)
            row["transformers_config_ok"] = True
            row["transformers_config_error"] = None
        except Exception as exc:
            row["transformers_config_ok"] = False
            row["transformers_config_error"] = f"{exc.__class__.__name__}: {exc}"
            last_err = row["transformers_config_error"]
    return version, last_err


def print_report(rows: list[dict], transformers_version: str) -> int:
    print(f"transformers_version\t{transformers_version}")
    print(
        "model\texists\tweights\tweight_gib\tconfig\ttokenizer\tindex\tincomplete\tmodel_type\tcfg_load"
    )
    failed = 0
    for row in rows:
        ok = bool(
            row["exists"]
            and row["weight_file_count"] > 0
            and row["has_config"]
            and row["has_tokenizer"]
            and row["has_index"]
            and row["incomplete_artifacts"] == 0
        )
        cfg_ok = row["transformers_config_ok"]
        if not ok or not cfg_ok:
            failed += 1
        cfg_load = "ok" if cfg_ok else "fail"
        print(
            f"{row['name']}\t{row['exists']}\t{row['weight_file_count']}"
            f"\t{row['weight_gib']:.2f}\t{row['has_config']}"
            f"\t{row['has_tokenizer']}\t{row['has_index']}"
            f"\t{row['incomplete_artifacts']}\t{row['model_type'] or '-'}\t{cfg_load}"
        )
        if row["transformers_config_error"]:
            print(f"  error: {row['transformers_config_error']}")
    return failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        default="/mnt/data/models/Qwen",
        help="Local directory containing Qwen model folders.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(REQUIRED_MODELS),
        help="Model folder names relative to --model-root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = check_downloads(Path(args.model_root), args.models)
    transformers_version, _ = check_transformers_config_load(rows)
    failed = print_report(rows, transformers_version)
    if failed:
        print(f"diagnostic_status\tFAIL\t{failed} model entries need attention")
        return 1
    print("diagnostic_status\tPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
