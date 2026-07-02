#!/usr/bin/env python3
"""Run a small overtraining ladder for the Markov theorem-sketch base case.

This script is deliberately narrower than the publication pipeline.  It asks
whether the 16-token-leaf Markov DGP is learnable when the state is explicitly
the theorem sketch `(count, first, last)`, and records parameter counts versus
training examples for each leaf/merge parameterization.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from test_markov_explicit_sketch_learning import (  # noqa: E402
    ExplicitMarkovSketchModel,
    _eval,
    _jsonable,
    _load_docs,
    _shape_info,
    _target_scale,
    _train_phase,
)
from src.ctreepo.sim.core.markov_neural_operator_baselines import _set_global_seed  # noqa: E402


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(value).replace(",", " ").split())


def _parameter_count(params: Sequence[nn.Parameter]) -> int:
    return int(sum(int(p.numel()) for p in params))


def _variant_config(name: str) -> Dict[str, Any]:
    if name == "mlp_mlp":
        return {
            "leaf_encoder": "mlp",
            "merge_count_mode": "mlp",
            "embed_dim": 64,
            "hidden_dim": 384,
            "exact_logit": 8.0,
            "merge_epochs": 300,
            "endpoint_pretrain_epochs": 100,
            "leaf_epochs": 900,
            "merge_lr": 0.002,
            "endpoint_lr": 0.001,
            "leaf_lr": 0.0007,
            "freeze_merge": True,
        }
    if name == "transition_mlp":
        return {
            "leaf_encoder": "transition_table",
            "merge_count_mode": "mlp",
            "embed_dim": 48,
            "hidden_dim": 256,
            "exact_logit": 8.0,
            "merge_epochs": 1200,
            "endpoint_pretrain_epochs": 150,
            "leaf_epochs": 800,
            "merge_lr": 0.0005,
            "endpoint_lr": 0.01,
            "leaf_lr": 0.005,
            "freeze_merge": True,
        }
    if name == "transition_additive":
        return {
            "leaf_encoder": "transition_table",
            "merge_count_mode": "additive_join_table",
            "embed_dim": 48,
            "hidden_dim": 256,
            "exact_logit": 16.0,
            "merge_epochs": 300,
            "endpoint_pretrain_epochs": 150,
            "leaf_epochs": 600,
            "merge_lr": 0.05,
            "endpoint_lr": 0.01,
            "leaf_lr": 0.005,
            "freeze_merge": True,
        }
    raise ValueError(f"unknown variant {name!r}")


def _render_md(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Markov Base-Case Overtrain Ladder",
        "",
        f"- Benchmark: `{summary['run']['benchmark']}`",
        f"- Seed: `{summary['run']['seed']}`",
        f"- Device: `{summary['run']['device']}`",
        "",
        "| train docs | variant | params | leaf examples / leaf param | merge examples / merge param | learned root exact | learned leaf exact | learned merge exact | learned root MAE | exact root exact |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["results"]:
        counts = row["parameter_counts"]
        ratios = row["sample_ratios"]
        learned = row["splits"]["test"]["learned_leaves"]
        exact = row["splits"]["test"]["exact_leaves"]
        lines.append(
            f"| {row['train_docs']} | {row['variant']} | {counts['total']} | "
            f"{ratios['leaf_examples_per_leaf_param']:.3g} | "
            f"{ratios['merge_examples_per_merge_param']:.3g} | "
            f"{learned['root_exact_match']:.6g} | {learned['leaf_exact_match']:.6g} | "
            f"{learned['merge_exact_match']:.6g} | {learned['root_mae']:.6g} | "
            f"{exact['root_exact_match']:.6g} |"
        )
    lines.extend(["", "## Details", "", "```json"])
    lines.append(json.dumps(summary["results"], indent=2, sort_keys=True))
    lines.extend(["```", ""])
    return "\n".join(lines)


def run_one(
    *,
    variant: str,
    train_docs_count: int,
    args: argparse.Namespace,
    device: torch.device,
    output_root: Path,
) -> Dict[str, Any]:
    cfg = _variant_config(str(variant))
    train_docs, val_docs, test_docs, prepared = _load_docs(
        benchmark=str(args.benchmark),
        train_docs=int(train_docs_count),
        seed=int(args.seed),
    )
    all_docs = tuple(train_docs) + tuple(val_docs) + tuple(test_docs)
    n_leaves, leaf_tokens, n_regimes, vocab_size = _shape_info(all_docs)
    model = ExplicitMarkovSketchModel(
        vocab_size=vocab_size,
        leaf_tokens=leaf_tokens,
        n_regimes=n_regimes,
        target_scale=_target_scale(train_docs),
        embed_dim=int(cfg["embed_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        exact_logit=float(cfg["exact_logit"]),
        leaf_encoder=str(cfg["leaf_encoder"]),
        merge_count_mode=str(cfg["merge_count_mode"]),
    ).to(device)
    variant_root = output_root / f"train{int(train_docs_count):06d}_{variant}"
    variant_root.mkdir(parents=True, exist_ok=True)
    progress_path = variant_root / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    leaf_params = _parameter_count(model.leaf_parameters())
    merge_params = _parameter_count(model.merge_parameters())
    total_params = _parameter_count(tuple(model.parameters()))
    n_merge = int(n_leaves) - 1
    leaf_examples = int(train_docs_count) * int(n_leaves)
    merge_examples = int(train_docs_count) * int(n_merge)
    print(
        json.dumps(
            {
                "event": "start_variant",
                "variant": str(variant),
                "train_docs": int(train_docs_count),
                "total_params": int(total_params),
                "leaf_params": int(leaf_params),
                "merge_params": int(merge_params),
                "leaf_examples": int(leaf_examples),
                "merge_examples": int(merge_examples),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    merge_final = _train_phase(
        model,
        train_docs,
        device=device,
        batch_size=int(args.batch_size),
        epochs=int(cfg["merge_epochs"]),
        lr=float(cfg["merge_lr"]),
        exact_leaves=True,
        root_weight=0.2,
        leaf_weight=0.0,
        merge_weight=1.0,
        seed=int(args.seed) + 1000 + int(train_docs_count),
        progress_path=progress_path,
        phase="merge_oracle_sketch",
        train_leaf=False,
        train_merge=True,
    )
    endpoint_final = None
    if int(cfg["endpoint_pretrain_epochs"]) > 0:
        endpoint_final = _train_phase(
            model,
            train_docs,
            device=device,
            batch_size=int(args.batch_size),
            epochs=int(cfg["endpoint_pretrain_epochs"]),
            lr=float(cfg["endpoint_lr"]),
            exact_leaves=False,
            root_weight=0.0,
            leaf_weight=1.0,
            merge_weight=0.0,
            leaf_count_loss_weight=0.0,
            leaf_endpoint_loss_weight=1.0,
            seed=int(args.seed) + 2000 + int(train_docs_count),
            progress_path=progress_path,
            phase="token_endpoint_pretrain",
            train_leaf=True,
            train_merge=False,
        )
    leaf_final = _train_phase(
        model,
        train_docs,
        device=device,
        batch_size=int(args.batch_size),
        epochs=int(cfg["leaf_epochs"]),
        lr=float(cfg["leaf_lr"]),
        exact_leaves=False,
        root_weight=0.2,
        leaf_weight=1.0,
        merge_weight=1.0,
        seed=int(args.seed) + 3000 + int(train_docs_count),
        progress_path=progress_path,
        phase="token_leaf_to_sketch",
        train_leaf=True,
        train_merge=not bool(cfg["freeze_merge"]),
    )
    splits = {}
    for split_name, docs in (("train", train_docs), ("val", val_docs), ("test", test_docs)):
        splits[split_name] = {
            "exact_leaves": _eval(
                model,
                docs,
                device=device,
                batch_size=int(args.batch_size),
                exact_leaves=True,
            ),
            "learned_leaves": _eval(
                model,
                docs,
                device=device,
                batch_size=int(args.batch_size),
                exact_leaves=False,
            ),
        }
    torch.save(model.state_dict(), variant_root / "model_state.pt")
    row = {
        "variant": str(variant),
        "train_docs": int(train_docs_count),
        "shape": {
            "n_leaves": int(n_leaves),
            "leaf_tokens": int(leaf_tokens),
            "n_regimes": int(n_regimes),
            "vocab_size": int(vocab_size),
            "target_scale": float(_target_scale(train_docs)),
        },
        "config": cfg,
        "parameter_counts": {
            "leaf": int(leaf_params),
            "merge": int(merge_params),
            "total": int(total_params),
        },
        "sample_counts": {
            "leaf_examples": int(leaf_examples),
            "merge_examples": int(merge_examples),
        },
        "sample_ratios": {
            "leaf_examples_per_leaf_param": float(leaf_examples) / max(1.0, float(leaf_params)),
            "merge_examples_per_merge_param": float(merge_examples) / max(1.0, float(merge_params)),
            "docs_per_total_param": float(train_docs_count) / max(1.0, float(total_params)),
        },
        "prepared_data_root": str(prepared.get("prepared_data_root", "")),
        "train_final": {
            "merge_oracle_sketch": merge_final,
            "token_endpoint_pretrain": endpoint_final,
            "token_leaf_to_sketch": leaf_final,
        },
        "splits": splits,
        "model_state": str(variant_root / "model_state.pt"),
        "progress": str(progress_path),
    }
    (variant_root / "summary.json").write_text(
        json.dumps(_jsonable(row), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "finish_variant",
                "variant": str(variant),
                "train_docs": int(train_docs_count),
                "test": _jsonable(splits["test"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="recoverable_v5_t128")
    parser.add_argument("--train-docs", default="256,1024,4096")
    parser.add_argument(
        "--variants",
        default="mlp_mlp,transition_mlp,transition_additive",
        help="Comma/space-separated list: mlp_mlp, transition_mlp, transition_additive.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--output-root",
        default=str(REPO / "outputs" / f"markov_base_case_overtrain_{_timestamp()}"),
    )
    args = parser.parse_args()

    _set_global_seed(int(args.seed))
    device = (
        torch.device(f"cuda:{int(args.cuda_device)}")
        if bool(args.use_cuda) and torch.cuda.is_available()
        else torch.device("cpu")
    )
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    train_doc_counts = _parse_int_list(str(args.train_docs))
    variants = tuple(str(part) for part in str(args.variants).replace(",", " ").split())

    results = []
    for train_docs_count in train_doc_counts:
        for variant in variants:
            results.append(
                run_one(
                    variant=variant,
                    train_docs_count=int(train_docs_count),
                    args=args,
                    device=device,
                    output_root=output_root,
                )
            )
            summary = {
                "run": {
                    "benchmark": str(args.benchmark),
                    "seed": int(args.seed),
                    "device": str(device),
                    "train_doc_counts": [int(v) for v in train_doc_counts],
                    "variants": list(variants),
                },
                "results": results,
            }
            (output_root / "overtrain_summary.json").write_text(
                json.dumps(_jsonable(summary), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            (output_root / "overtrain_summary.md").write_text(
                _render_md(_jsonable(summary)),
                encoding="utf-8",
            )

    print(output_root / "overtrain_summary.json")
    print(output_root / "overtrain_summary.md")


if __name__ == "__main__":
    main()
