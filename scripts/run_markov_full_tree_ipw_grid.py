#!/usr/bin/env python3
"""Run a layered p_internal x p_leaf grid for shared-g full-tree IPW learning."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.core.full_tree_ipw_grid import (  # noqa: E402
    build_markov_full_tree_ipw_tradeoff_summary,
    grid_rows_from_payload,
    load_markov_full_tree_ipw_grid_from_output_dir,
    render_markov_full_tree_ipw_tradeoff_markdown,
    run_markov_full_tree_ipw_grid,
    write_grid_csv,
)
from src.ctreepo.sim.core.markov_changepoint_ops_count import (  # noqa: E402
    MarkovOPSDataBundle,
    OPSCountConfig,
    VALID_GENERATOR_PROFILES,
    build_markov_changepoint_ops_count_data_bundle,
)
from src.ctreepo.sim.suite.markov_observed_token_policy import (  # noqa: E402
    resolve_markov_observed_token_policy,
)
from src.tree.full_tree_ipw import (  # noqa: E402
    DEFAULT_LAYERED_RATE_GRID,
    DEFAULT_TRADEOFF_RATE_GRID,
)


OBSERVED_TOKEN_CANONICAL_BUNDLES = {
    "demo_v1": REPO_ROOT
    / "outputs/markov_observed_token_suite_demo_v1/markov_data/observed_token_bundle.json",
    "recoverable": REPO_ROOT
    / "outputs/markov_observed_token_recoverable_v4/markov_data/observed_token_bundle.json",
}

OBSERVED_TOKEN_CANONICAL_ROOT_ONLY_SUMMARIES = {
    "demo_v1": REPO_ROOT
    / "outputs/markov_observed_token_suite_demo_v1/markov_changepoint_ops_count/root_only/seed_0.json",
    "recoverable": REPO_ROOT
    / "outputs/markov_observed_token_recoverable_v4/markov_changepoint_ops_count/root_only/seed_0.json",
}


def _parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for raw in str(text).replace(",", " ").split():
        token = raw.strip()
        if token:
            values.append(float(token))
    return values


def _load_saved_reference_anchors(profile_name: str) -> dict[str, object]:
    path = OBSERVED_TOKEN_CANONICAL_ROOT_ONLY_SUMMARIES.get(str(profile_name).strip().lower())
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = dict(payload.get("metrics") or {})

    def _surface(key: str) -> dict[str, object]:
        return {
            "test": dict(metrics.get(key) or {}),
            "val": dict(metrics.get(f"{key}_val") or {}),
            "train": dict(metrics.get(f"{key}_train") or {}),
            "training": dict(metrics.get(f"{key}_training") or {}),
            "summary_json": str(path),
        }

    return {
        "root_only_tree_neural": _surface("learned"),
        "root_only_full_doc_dense": _surface("doc_level"),
        "root_only_full_doc_ridge": _surface("doc_level_ridge"),
        "root_only_rf_root": _surface("rf_root"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep layered internal/leaf sampling rates for the Markov shared-g IPW study."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/markov_full_tree_ipw_grid",
        help="Directory for aggregate outputs plus per-cell JSON summaries.",
    )
    parser.add_argument(
        "--json-summary",
        type=str,
        default="",
        help="Optional aggregate JSON output path. Defaults to <output-dir>/summary.json.",
    )
    parser.add_argument(
        "--csv-summary",
        type=str,
        default="",
        help="Optional aggregate CSV output path. Defaults to <output-dir>/summary.csv.",
    )
    parser.add_argument(
        "--tradeoff-summary",
        type=str,
        default="",
        help="Optional tradeoff JSON path. Defaults to <output-dir>/tradeoff_summary.json.",
    )
    parser.add_argument(
        "--tradeoff-markdown",
        type=str,
        default="",
        help="Optional tradeoff Markdown path. Defaults to <output-dir>/tradeoff_summary.md.",
    )
    parser.add_argument(
        "--grid-rates",
        type=str,
        default=",".join(f"{x:g}" for x in DEFAULT_LAYERED_RATE_GRID),
        help="Comma/space separated layered rates for both axes.",
    )
    parser.add_argument(
        "--internal-rates",
        type=str,
        default="",
        help="Optional comma/space separated internal-node rates. Defaults to --grid-rates.",
    )
    parser.add_argument(
        "--leaf-rates",
        type=str,
        default="",
        help="Optional comma/space separated leaf-node rates. Defaults to --grid-rates.",
    )
    parser.add_argument(
        "--grid-preset",
        type=str,
        choices=["coarse", "tradeoff"],
        default="coarse",
        help=(
            "Named rate-grid preset. `tradeoff` adds more intermediate rates; "
            "`coarse` keeps the original paper grid."
        ),
    )
    parser.add_argument(
        "--root-only-fractions",
        type=str,
        default="0",
        help=(
            "Comma/space separated fractions of training/validation docs collapsed to "
            "a one-node full-document view."
        ),
    )
    parser.add_argument(
        "--doc-sequence-train-fractions",
        type=str,
        default="0",
        help=(
            "Comma/space separated fractions of training docs routed through the "
            "in-model full-document doc-sequence objective."
        ),
    )
    parser.add_argument(
        "--bundle-output",
        type=str,
        default="",
        help=(
            "Optional path to save the fixed train/val/test bundle used across all cells. "
            "Defaults to <output-dir>/markov_bundle.json when generating a new bundle."
        ),
    )
    parser.add_argument(
        "--load-data-bundle",
        type=str,
        default="",
        help="Optional path to an existing MarkovOPSDataBundle JSON.",
    )
    parser.add_argument(
        "--observed-token-profile",
        type=str,
        choices=["custom", "demo", "demo_v1", "recoverable"],
        default="demo_v1",
        help=(
            "Apply a saved observed-token suite profile to reproduce a known regime. "
            "When a canonical bundle exists locally and --load-data-bundle is omitted, "
            "the script reuses that bundle automatically."
        ),
    )
    parser.add_argument("--n-regimes", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument(
        "--generator-profile",
        type=str,
        choices=list(VALID_GENERATOR_PROFILES),
        default="piecewise_markov",
    )
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--min-segments", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument("--min-seg-len", type=int, default=4)
    parser.add_argument("--max-seg-len", type=int, default=16)
    parser.add_argument("--fixed-leaf-tokens", type=int, default=16)
    parser.add_argument("--train-docs", type=int, default=128)
    parser.add_argument("--val-docs", type=int, default=32)
    parser.add_argument("--test-docs", type=int, default=64)
    parser.add_argument(
        "--model-family",
        type=str,
        choices=["neural", "fno"],
        default="neural",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=["full", "no_endpoints", "token_full", "token_bow"],
        default="full",
    )
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--fno-width", type=int, default=32)
    parser.add_argument("--fno-n-modes", type=int, default=8)
    parser.add_argument("--fno-n-layers", type=int, default=2)
    parser.add_argument(
        "--use-residual-decomposition",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--val-seed-offset", type=int, default=5_000)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument(
        "--include-full-doc-anchors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run one extra full-document anchor job and attach doc-level, ridge, "
            "doc-sequence/FNO, and RF root baselines to the grid payload."
        ),
    )
    parser.add_argument(
        "--skip-full-doc-anchors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable the extra full-document anchor pass, even for saved observed-token profiles.",
    )
    parser.add_argument(
        "--skip-existing-cells",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse existing per-cell JSON summaries in <output-dir>/cells instead of rerunning them.",
    )
    parser.add_argument(
        "--aggregate-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do not run any cells; rebuild the aggregate payload from existing cell JSON files.",
    )
    parser.add_argument(
        "--write-aggregate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write summary.json/csv/tradeoff outputs. Disable for worker shards sharing one output dir.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    json_summary = (
        Path(str(args.json_summary))
        if str(args.json_summary).strip()
        else output_dir / "summary.json"
    )
    csv_summary = (
        Path(str(args.csv_summary))
        if str(args.csv_summary).strip()
        else output_dir / "summary.csv"
    )
    tradeoff_summary = (
        Path(str(args.tradeoff_summary))
        if str(args.tradeoff_summary).strip()
        else output_dir / "tradeoff_summary.json"
    )
    tradeoff_markdown = (
        Path(str(args.tradeoff_markdown))
        if str(args.tradeoff_markdown).strip()
        else output_dir / "tradeoff_summary.md"
    )
    if args.cpu:
        args.device = "cpu"
    use_cuda = args.device in ("auto", "cuda")
    if args.device == "auto":
        use_cuda = torch.cuda.is_available()

    grid_rates = _parse_float_list(str(args.grid_rates))
    default_coarse_rates = ",".join(f"{x:g}" for x in DEFAULT_LAYERED_RATE_GRID)
    if not grid_rates:
        grid_rates = (
            list(DEFAULT_TRADEOFF_RATE_GRID)
            if str(args.grid_preset) == "tradeoff"
            else list(DEFAULT_LAYERED_RATE_GRID)
        )
    elif (
        str(args.grid_rates).strip() == default_coarse_rates
        and str(args.grid_preset) == "tradeoff"
    ):
        grid_rates = list(DEFAULT_TRADEOFF_RATE_GRID)
    internal_rates = _parse_float_list(str(args.internal_rates)) or list(grid_rates)
    leaf_rates = _parse_float_list(str(args.leaf_rates)) or list(grid_rates)
    root_only_fractions = _parse_float_list(str(args.root_only_fractions)) or [0.0]
    doc_sequence_train_fractions = (
        _parse_float_list(str(args.doc_sequence_train_fractions)) or [0.0]
    )
    observed_token_profile = str(args.observed_token_profile).strip().lower()
    if observed_token_profile == "custom":
        observed_token_profile = ""
    if observed_token_profile:
        policy = resolve_markov_observed_token_policy(profile_name=observed_token_profile)
        cfg = OPSCountConfig(
            n_regimes=int(policy.n_regimes),
            vocab_size=int(policy.vocab_size),
            generator_profile=str(policy.generator_profile),
            min_tokens=int(policy.min_tokens),
            max_tokens=int(policy.max_tokens),
            min_segments=int(policy.min_segments),
            max_segments=int(policy.max_segments),
            min_seg_len=int(policy.min_seg_len),
            max_seg_len=int(policy.max_seg_len),
            fixed_leaf_tokens=int(policy.fixed_leaf_tokens),
            train_docs=int(policy.train_docs),
            val_docs=int(policy.val_docs),
            test_docs=int(policy.test_docs),
            model_family=str(args.model_family),
            feature_mode="token_full",
            state_dim=int(policy.state_dim),
            hidden_dim=int(policy.hidden_dim),
            n_epochs=int(policy.n_epochs),
            batch_size=int(policy.batch_size),
            lr=float(policy.lr),
            weight_decay=float(policy.weight_decay),
            fno_width=int(args.fno_width),
            fno_n_modes=int(args.fno_n_modes),
            fno_n_layers=int(args.fno_n_layers),
            use_unified_ipw=True,
            ipw_leaf_sample_rate=1.0,
            ipw_internal_sample_rate=1.0,
            use_residual_decomposition=bool(args.use_residual_decomposition),
            root_only_train_fraction=0.0,
            doc_sequence_train_fraction=0.0,
            grad_clip_norm=float(args.grad_clip_norm),
            seed=int(policy.seed),
            data_seed=(int(args.data_seed) if args.data_seed is not None else None),
            model_seed=(int(args.model_seed) if args.model_seed is not None else None),
            val_seed_offset=int(args.val_seed_offset),
            use_cuda=bool(use_cuda),
            cuda_device=int(args.cuda_device) if args.cuda_device is not None else None,
            torch_threads=max(int(args.torch_threads), int(policy.torch_threads)),
            artifact_dir=str(output_dir / "artifacts"),
        )
    else:
        cfg = OPSCountConfig(
            n_regimes=int(args.n_regimes),
            vocab_size=int(args.vocab_size),
            generator_profile=str(args.generator_profile),
            min_tokens=int(args.min_tokens),
            max_tokens=int(args.max_tokens),
            min_segments=int(args.min_segments),
            max_segments=int(args.max_segments),
            min_seg_len=int(args.min_seg_len),
            max_seg_len=int(args.max_seg_len),
            fixed_leaf_tokens=int(args.fixed_leaf_tokens),
            train_docs=int(args.train_docs),
            val_docs=int(args.val_docs),
            test_docs=int(args.test_docs),
            model_family=str(args.model_family),
            feature_mode=str(args.feature_mode),
            state_dim=int(args.state_dim),
            hidden_dim=int(args.hidden_dim),
            n_epochs=int(args.n_epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            fno_width=int(args.fno_width),
            fno_n_modes=int(args.fno_n_modes),
            fno_n_layers=int(args.fno_n_layers),
            use_unified_ipw=True,
            ipw_leaf_sample_rate=1.0,
            ipw_internal_sample_rate=1.0,
            use_residual_decomposition=bool(args.use_residual_decomposition),
            root_only_train_fraction=0.0,
            doc_sequence_train_fraction=0.0,
            grad_clip_norm=float(args.grad_clip_norm),
            seed=int(args.seed),
            data_seed=(int(args.data_seed) if args.data_seed is not None else None),
            model_seed=(int(args.model_seed) if args.model_seed is not None else None),
            val_seed_offset=int(args.val_seed_offset),
            use_cuda=bool(use_cuda),
            cuda_device=int(args.cuda_device) if args.cuda_device is not None else None,
            torch_threads=int(args.torch_threads),
            artifact_dir=str(output_dir / "artifacts"),
        )

    bundle_path = (
        Path(str(args.bundle_output))
        if str(args.bundle_output).strip()
        else output_dir / "markov_bundle.json"
    )
    load_data_bundle = str(args.load_data_bundle).strip()
    if not load_data_bundle and observed_token_profile:
        canonical_bundle = OBSERVED_TOKEN_CANONICAL_BUNDLES.get(observed_token_profile)
        if canonical_bundle is not None and canonical_bundle.exists():
            load_data_bundle = str(canonical_bundle)
    if load_data_bundle:
        data_bundle = MarkovOPSDataBundle.load(Path(load_data_bundle))
        bundle_source = str(Path(load_data_bundle))
    else:
        data_bundle = build_markov_changepoint_ops_count_data_bundle(cfg)
        data_bundle.save(bundle_path)
        bundle_source = str(bundle_path)

    include_full_doc_anchors = bool(
        not args.skip_full_doc_anchors
        and (args.include_full_doc_anchors or observed_token_profile)
    )
    if bool(args.aggregate_only):
        payload = load_markov_full_tree_ipw_grid_from_output_dir(
            output_dir=output_dir,
            base_config=cfg,
            data_bundle=data_bundle,
            rate_axis=grid_rates,
            internal_rate_axis=internal_rates,
            leaf_rate_axis=leaf_rates,
            root_only_fraction_axis=root_only_fractions,
            doc_sequence_train_fraction_axis=doc_sequence_train_fractions,
            include_full_doc_anchors=include_full_doc_anchors,
        )
    else:
        payload = run_markov_full_tree_ipw_grid(
            base_config=cfg,
            data_bundle=data_bundle,
            rate_axis=grid_rates,
            internal_rate_axis=internal_rates,
            leaf_rate_axis=leaf_rates,
            root_only_fraction_axis=root_only_fractions,
            doc_sequence_train_fraction_axis=doc_sequence_train_fractions,
            include_full_doc_anchors=include_full_doc_anchors,
            output_dir=output_dir,
            skip_existing=bool(args.skip_existing_cells),
        )
    payload["bundle_source"] = str(bundle_source)
    payload["observed_token_profile"] = observed_token_profile
    payload["saved_reference_anchors"] = (
        _load_saved_reference_anchors(observed_token_profile)
        if observed_token_profile
        else {}
    )
    doc_sequence_root_mae = float(
        (
            payload.get("full_doc_anchors", {})
            .get("doc_sequence", {})
            .get("test", {})
            .get("root_mae", float("nan"))
        )
    )
    if observed_token_profile and doc_sequence_root_mae == doc_sequence_root_mae:
        payload["default_reference"] = {
            "name": "full_doc_doc_sequence",
            "family": "official_neuraloperator_fno",
            "root_mae": float(doc_sequence_root_mae),
            "source": "computed_full_doc_anchor",
        }
    payload["base_config"] = asdict(cfg)

    rows = grid_rows_from_payload(payload)
    tradeoff_payload = build_markov_full_tree_ipw_tradeoff_summary(payload)
    tradeoff_markdown_text = render_markov_full_tree_ipw_tradeoff_markdown(
        tradeoff_payload
    )
    if bool(args.write_aggregate):
        json_summary.parent.mkdir(parents=True, exist_ok=True)
        json_summary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_grid_csv(csv_summary, rows)
        tradeoff_summary.parent.mkdir(parents=True, exist_ok=True)
        tradeoff_summary.write_text(
            json.dumps(tradeoff_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tradeoff_markdown.parent.mkdir(parents=True, exist_ok=True)
        tradeoff_markdown.write_text(tradeoff_markdown_text, encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "json_summary": str(json_summary) if args.write_aggregate else "",
                    "csv_summary": str(csv_summary) if args.write_aggregate else "",
                    "tradeoff_summary": str(tradeoff_summary) if args.write_aggregate else "",
                    "tradeoff_markdown": str(tradeoff_markdown) if args.write_aggregate else "",
                    "write_aggregate": bool(args.write_aggregate),
                    "cells": len(list(payload.get("cells") or []))
                    or sum(len(list(plane.get("cells") or [])) for plane in list(payload.get("planes") or [])),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
