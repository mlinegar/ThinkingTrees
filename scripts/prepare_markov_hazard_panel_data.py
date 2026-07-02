#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ctreepo.data.prep_common import ensure_repo_on_path  # noqa: E402

REPO_ROOT = ensure_repo_on_path()

from src.ctreepo.data.splits import (  # noqa: E402
    SPLIT_SCHEMA_VERSION,
    split_from_count_slices,
)
from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (  # noqa: E402
    FullDocDiagnosticBenchmarkSpec,
    _ensure_prepared_markov_tree_data,
)
from src.ctreepo.sim.core.markov_changepoint_ops_count import (  # noqa: E402
    _root_count_diagnostics,
)
from src.ctreepo.sim.core.markov_hazard_panels import (  # noqa: E402
    build_markov_hazard_panel_data_bundle,
    panel_to_ops_overrides,
    resolve_markov_hazard_panel,
)


DEFAULT_PANEL_IDS = (
    "paper_hazard_panel_v1_t128",
    "paper_hazard_panel_v1_t2048",
)
DEFAULT_TRAIN_PREFIX_COUNTS = (1024, 4096, 10240)


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / f"markov_hazard_panel_data_{timestamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize paper-facing Markov hazard panel bundles and prepared "
            "tree/FNO caches with condition-balanced train prefixes."
        )
    )
    parser.add_argument("--panel-ids", nargs="*", default=list(DEFAULT_PANEL_IDS))
    parser.add_argument("--train-docs", type=int, default=10240)
    parser.add_argument("--val-docs", type=int, default=1024)
    parser.add_argument("--test-docs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-prefix-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRAIN_PREFIX_COUNTS),
    )
    parser.add_argument(
        "--prepared-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds for prepared leaf/internal supervision orderings. Defaults to --seed.",
    )
    parser.add_argument("--fixed-leaf-tokens", type=int, default=16)
    parser.add_argument("--max-internal-depth", type=int, default=0)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "_bundles" / "markov_hazard_panels",
    )
    parser.add_argument(
        "--prepared-data-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "_prepared_data" / "markov_hazard_panels",
    )
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument(
        "--skip-prepared-cache",
        action="store_true",
        help="Write bundles/reports only. Intended for very fast local smoke checks.",
    )
    return parser.parse_args()


def _sorted_positive_prefix_counts(
    values: Sequence[int],
    *,
    train_docs: int,
) -> tuple[int, ...]:
    seen: set[int] = set()
    out: list[int] = []
    for value in sorted(int(v) for v in values):
        if int(value) <= 0 or int(value) > int(train_docs):
            continue
        if int(value) in seen:
            continue
        seen.add(int(value))
        out.append(int(value))
    if int(train_docs) > 0 and int(train_docs) not in seen:
        out.append(int(train_docs))
    return tuple(out)


def _condition_counts_for_prefix(
    condition_ids: Sequence[str],
    prefix_count: int,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for condition_id in list(condition_ids)[: int(prefix_count)]:
        key = str(condition_id)
        counts[key] = int(counts.get(key, 0)) + 1
    return counts


def _split_diagnostics(bundle: Any, split: str) -> Dict[str, Any]:
    metadata = dict(getattr(bundle, "metadata", {}) or {})
    condition_ids = dict(metadata.get("condition_ids") or {}).get(str(split), [])
    docs = tuple(getattr(bundle, f"{split}_docs"))
    return dict(_root_count_diagnostics(docs, condition_ids=condition_ids))


def _panel_benchmark(
    *,
    panel_id: str,
    bundle_path: Path,
    train_docs: int,
    val_docs: int,
    test_docs: int,
) -> FullDocDiagnosticBenchmarkSpec:
    panel = resolve_markov_hazard_panel(panel_id)
    overrides = dict(panel_to_ops_overrides(panel))
    overrides.update(
        {
            "train_docs": int(train_docs),
            "val_docs": int(val_docs),
            "test_docs": int(test_docs),
        }
    )
    doc_tokens = int(max(condition.doc_tokens for condition in panel.conditions))
    regime_count = int(max(condition.n_regimes for condition in panel.conditions))
    return FullDocDiagnosticBenchmarkSpec(
        name=str(panel.panel_id),
        description=str(panel.display_name),
        observed_token_profile="hazard_panel",
        canonical_bundle_path=str(bundle_path),
        canonical_train_docs_capacity=int(train_docs),
        degenerate=False,
        cell_id=str(panel.panel_id),
        grid_name="markov_hazard_panel",
        regime_count=int(regime_count),
        segment_density_band="mixed",
        segment_min=0,
        segment_max=0,
        hazard_switch_prob=float("nan"),
        config_overrides=overrides
        | {
            "min_tokens": int(doc_tokens),
            "max_tokens": int(doc_tokens),
        },
        default_train_doc_counts=tuple(DEFAULT_TRAIN_PREFIX_COUNTS),
    )


def _prepare_panel(args: argparse.Namespace, panel_id: str) -> Dict[str, Any]:
    panel = resolve_markov_hazard_panel(panel_id)
    bundle = build_markov_hazard_panel_data_bundle(
        str(panel.panel_id),
        train_docs=int(args.train_docs),
        val_docs=int(args.val_docs),
        test_docs=int(args.test_docs),
        seed=int(args.seed),
    )
    bundle_dir = (
        Path(args.bundle_root).expanduser()
        / str(panel.panel_id)
        / f"seed_{int(args.seed)}"
    )
    bundle_path = bundle_dir / "base_bundle.json"
    bundle.save(bundle_path)

    # Additive: emit the shared id-based split beside the bundle so downstream
    # can round-trip through src.ctreepo.data.splits. Slices mirror the bundle's
    # train/val/test counts; no existing output is altered.
    shared_split = split_from_count_slices(
        train=int(len(bundle.train_docs)),
        val=int(len(bundle.val_docs)),
        test=int(len(bundle.test_docs)),
        id_prefix="markov_doc",
        metadata={"family": "markov", "panel_id": str(panel.panel_id)},
    )
    shared_split.save(bundle_dir)

    train_prefix_counts = _sorted_positive_prefix_counts(
        args.train_prefix_counts,
        train_docs=int(args.train_docs),
    )
    prepared_seeds = tuple(
        int(seed)
        for seed in (
            list(args.prepared_seeds)
            if args.prepared_seeds is not None
            else [int(args.seed)]
        )
    )
    prepared_payload: Dict[str, Any] = {}
    if not bool(args.skip_prepared_cache):
        benchmark = _panel_benchmark(
            panel_id=str(panel.panel_id),
            bundle_path=bundle_path,
            train_docs=int(args.train_docs),
            val_docs=int(args.val_docs),
            test_docs=int(args.test_docs),
        )
        prepared = _ensure_prepared_markov_tree_data(
            benchmark=benchmark,
            base_bundle=bundle,
            required_train_docs=int(args.train_docs),
            train_prefix_counts=train_prefix_counts,
            fixed_leaf_tokens=int(args.fixed_leaf_tokens),
            max_internal_depth=int(args.max_internal_depth),
            seeds=prepared_seeds,
            prepared_data_root=str(Path(args.prepared_data_root).expanduser()),
            allow_create=True,
        )
        prepared_payload = {
            "prepared_data_root": str(prepared.root),
            "prepared_data_signature": str(prepared.signature),
            "metadata_json": str(prepared.root / "metadata.json"),
            "base_bundle_json": str(prepared.root / "base_bundle.json"),
            "train_fno_docs_json": str(prepared.root / "train_fno_docs.json"),
            "val_fno_docs_json": str(prepared.root / "val_fno_docs.json"),
            "test_fno_docs_json": str(prepared.root / "test_fno_docs.json"),
            "leaf_orderings_json": str(prepared.root / "leaf_orderings.json"),
            "internal_orderings_json": str(prepared.root / "internal_orderings.json"),
        }

    metadata = dict(getattr(bundle, "metadata", {}) or {})
    train_condition_ids = list(dict(metadata.get("condition_ids") or {}).get("train", []))
    return {
        "panel_id": str(panel.panel_id),
        "display_name": str(panel.display_name),
        "bundle_path": str(bundle_path),
        "shared_split": {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "split_ids_path": str(bundle_dir / "split_ids.json"),
            "counts": shared_split.counts(),
        },
        "prepared": prepared_payload,
        "split_sizes": {
            "train": int(len(bundle.train_docs)),
            "val": int(len(bundle.val_docs)),
            "test": int(len(bundle.test_docs)),
        },
        "corpus_signatures": {
            "train": str(bundle.train_corpus_signature),
            "val": str(bundle.val_corpus_signature),
            "test": str(bundle.test_corpus_signature),
        },
        "condition_counts": dict(metadata.get("condition_counts") or {}),
        "train_prefix_counts": [int(value) for value in train_prefix_counts],
        "train_prefix_condition_counts": {
            str(int(prefix)): _condition_counts_for_prefix(train_condition_ids, int(prefix))
            for prefix in train_prefix_counts
        },
        "target_diagnostics": {
            "train": _split_diagnostics(bundle, "train"),
            "val": _split_diagnostics(bundle, "val"),
            "test": _split_diagnostics(bundle, "test"),
        },
        "panel": panel.to_dict(),
    }


def _write_report(path: Path, manifest: Mapping[str, Any]) -> None:
    lines = [
        "# Markov Hazard Panel Data",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        f"- Train/val/test docs: `{manifest['train_docs']} / {manifest['val_docs']} / {manifest['test_docs']}`",
        f"- Seed: `{manifest['seed']}`",
        f"- Train prefixes: `{', '.join(str(v) for v in manifest['train_prefix_counts'])}`",
        "",
        "## Panels",
        "",
        "| Panel | Bundle | Train Counts | Val Counts | Test Counts | Global Mean MAE | Condition Mean MAE | Gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for panel_payload in list(manifest.get("panels") or []):
        target = dict(panel_payload.get("target_diagnostics") or {}).get("test", {})
        lines.append(
            "| "
            + f"`{panel_payload.get('panel_id')}` | "
            + f"`{panel_payload.get('bundle_path')}` | "
            + f"`{json.dumps(panel_payload.get('condition_counts', {}).get('train', {}), sort_keys=True)}` | "
            + f"`{json.dumps(panel_payload.get('condition_counts', {}).get('val', {}), sort_keys=True)}` | "
            + f"`{json.dumps(panel_payload.get('condition_counts', {}).get('test', {}), sort_keys=True)}` | "
            + f"{float(target.get('global_mean_baseline_mae', float('nan'))):.4g} | "
            + f"{float(target.get('condition_mean_baseline_mae', float('nan'))):.4g} | "
            + f"{float(target.get('mean_guess_gap', float('nan'))):.4g} |"
        )
    lines.extend(["", "## Prepared Caches", ""])
    for panel_payload in list(manifest.get("panels") or []):
        prepared = dict(panel_payload.get("prepared") or {})
        lines.append(
            "- "
            + f"`{panel_payload.get('panel_id')}`: "
            + f"`{prepared.get('prepared_data_root', '(skipped)')}`"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels = [_prepare_panel(args, str(panel_id)) for panel_id in list(args.panel_ids)]
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "panel_ids": [str(panel_id) for panel_id in list(args.panel_ids)],
        "train_docs": int(args.train_docs),
        "val_docs": int(args.val_docs),
        "test_docs": int(args.test_docs),
        "seed": int(args.seed),
        "train_prefix_counts": [
            int(value)
            for value in _sorted_positive_prefix_counts(
                args.train_prefix_counts,
                train_docs=int(args.train_docs),
            )
        ],
        "fixed_leaf_tokens": int(args.fixed_leaf_tokens),
        "max_internal_depth": int(args.max_internal_depth),
        "bundle_root": str(Path(args.bundle_root).expanduser()),
        "prepared_data_root": str(Path(args.prepared_data_root).expanduser()),
        "panels": panels,
    }
    manifest_path = Path(args.output_dir) / "manifest.json"
    report_path = Path(args.output_dir) / "report.md"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(report_path, manifest)
    print(f"Wrote {manifest_path}")
    print(f"Wrote {report_path}")
    for panel_payload in panels:
        print(f"Wrote {panel_payload['bundle_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
