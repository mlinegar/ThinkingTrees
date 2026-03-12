#!/usr/bin/env python3
"""Run the Stage-1 bag-of-words LDA utility-vector tree-recovery simulation."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
from pathlib import Path
import sys
from typing import List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.core.lda_tree_utility_vector import (  # noqa: E402
    LDATreeUtilityVectorConfig,
    VALID_DEVICE_MODES,
    VALID_EMISSION_MODES,
    VALID_UTILITY_DESIGNS,
    run_lda_tree_utility_vector_experiment,
)


def _parse_fraction(text: str) -> float:
    raw = str(text).strip()
    if not raw:
        raise ValueError("leaf fraction must be non-empty")
    if "/" in raw:
        return float(Fraction(raw))
    return float(raw)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Stage-1 LDA utility-vector tree simulation.")

    p.add_argument("--n-topics", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--doc-tokens", type=int, default=384)
    p.add_argument("--doc-topic-concentration", type=float, default=0.6)

    p.add_argument("--topic-concentration", type=float, default=0.2)
    p.add_argument("--emission-mode", type=str, choices=list(VALID_EMISSION_MODES), default="anchored")
    p.add_argument("--anchor-words-per-topic", type=int, default=20)
    p.add_argument("--anchor-multiplier", type=float, default=25.0)

    p.add_argument("--utility-dim", type=int, default=16)
    p.add_argument("--utility-design", type=str, choices=list(VALID_UTILITY_DESIGNS), default="topic_anchored_sparse")
    p.add_argument(
        "--leaf-fraction",
        type=str,
        default="1/24",
        help="Leaf size as a fraction of document length, e.g. 1, 1/2, 1/4, 1/24.",
    )

    p.add_argument("--train-docs", type=int, default=512)
    p.add_argument("--test-docs", type=int, default=256)
    p.add_argument("--state-dim", type=int, default=64)

    p.add_argument(
        "--run-full-doc-mlp-diag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to fit the appendix-only full-document MLP diagnostic.",
    )
    p.add_argument("--full-hidden-dim", type=int, default=128)
    p.add_argument("--full-n-layers", type=int, default=2)
    p.add_argument("--n-epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--device", type=str, choices=list(VALID_DEVICE_MODES), default="auto")
    p.add_argument("--cuda-device", type=int, default=None)
    p.add_argument("--torch-threads", type=int, default=0)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-summary", type=str, required=True)
    p.add_argument("--csv-summary", type=str, required=True)
    p.add_argument("--json", action="store_true", help="Emit JSON to stdout as well.")
    return p.parse_args(list(argv) if argv is not None else None)


def _rows_from_summary(summary) -> List[dict]:
    cfg = dict(summary.config)
    world_stats = dict(summary.world_stats)
    exact = dict(summary.exact_recovery)
    rows: List[dict] = []
    methods = summary.methods if isinstance(summary.methods, dict) else {}
    for method, metrics in methods.items():
        if not isinstance(metrics, dict):
            continue
        row = {
            "family": str(summary.family),
            "target_kind": str(summary.target_kind),
            "method": str(method),
            "is_stale_generation": bool(summary.is_stale_generation),
            **{f"cfg_{k}": v for k, v in cfg.items()},
            **{f"world_{k}": v for k, v in world_stats.items()},
            **{f"exact_{k}": v for k, v in exact.items()},
        }
        row.update(metrics)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = LDATreeUtilityVectorConfig(
        n_topics=int(args.n_topics),
        vocab_size=int(args.vocab_size),
        doc_tokens=int(args.doc_tokens),
        doc_topic_concentration=float(args.doc_topic_concentration),
        topic_concentration=float(args.topic_concentration),
        emission_mode=str(args.emission_mode),
        anchor_words_per_topic=int(args.anchor_words_per_topic),
        anchor_multiplier=float(args.anchor_multiplier),
        utility_dim=int(args.utility_dim),
        utility_design=str(args.utility_design),
        leaf_fraction=float(_parse_fraction(args.leaf_fraction)),
        train_docs=int(args.train_docs),
        test_docs=int(args.test_docs),
        state_dim=int(args.state_dim),
        run_full_doc_mlp_diag=bool(args.run_full_doc_mlp_diag),
        full_hidden_dim=int(args.full_hidden_dim),
        full_n_layers=int(args.full_n_layers),
        n_epochs=int(args.n_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=str(args.device),
        cuda_device=args.cuda_device,
        torch_threads=int(args.torch_threads),
        seed=int(args.seed),
    )
    summary = run_lda_tree_utility_vector_experiment(cfg)

    json_path = Path(args.json_summary)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(summary.to_json(), encoding="utf-8")

    csv_path = Path(args.csv_summary)
    _write_csv(csv_path, _rows_from_summary(summary))

    exact = dict(summary.exact_recovery)
    methods = dict(summary.methods)
    tree = dict(methods.get("tree_exact_utility", {}))
    count = dict(methods.get("count_svd_ceiling", {}))
    utility = dict(methods.get("utility_pca_practical", {}))
    print(f"wrote_json | {json_path}")
    print(f"wrote_csv | {csv_path}")
    print(
        "exact_recovery | root_u_l1={:.3e} | root_scalar_abs={:.3e}".format(
            float(exact.get("root_utility_l1_mean", float("nan"))),
            float(exact.get("root_scalar_abs_mean", float("nan"))),
        )
    )
    print(
        "tree_exact_utility | u_l1_to_full={:.3e} | scalar_abs_to_full={:.3e}".format(
            float(tree.get("utility_l1_to_full_mean", float("nan"))),
            float(tree.get("scalar_abs_to_full_mean", float("nan"))),
        )
    )
    print(
        "count_svd_ceiling | u_l1_to_full={:.4f} | count_l1_to_full={:.4f}".format(
            float(count.get("utility_l1_to_full_mean", float("nan"))),
            float(count.get("count_l1_to_full_mean", float("nan"))),
        )
    )
    print(
        "utility_pca_practical | u_l1_to_full={:.4f} | scalar_abs_to_full={:.4f}".format(
            float(utility.get("utility_l1_to_full_mean", float("nan"))),
            float(utility.get("scalar_abs_to_full_mean", float("nan"))),
        )
    )
    if bool(args.json):
        print(summary.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
