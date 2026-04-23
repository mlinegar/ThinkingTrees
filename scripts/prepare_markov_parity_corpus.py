#!/usr/bin/env python3
"""Prepare a single large Markov corpus for parity-grid experiments.

Generates one corpus of ``--total-docs`` documents (default 50000) from a
single RNG stream, then partitions it into:

  * **Training prefixes** (nested): e.g. first 1024, 2048, 4096, 10240, 20480
  * **Test set**: ``--test-docs`` documents sampled from the remainder
  * **Validation set**: everything left after train + test

All experiments loading from this prepared root will share the same test set
regardless of which training prefix they use.

Output layout::

    <output-root>/
        corpus_manifest.json      # metadata + prefix signatures
        corpus_docs.jsonl         # all docs, one JSON object per line
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.core.markov_changepoint_ops_count import (  # noqa: E402
    MarkovOPSDataBundle,
    OPSCountConfig,
    _build_generator_transitions,
    _generate_ops_count_docs,
    _markov_corpus_signature as _bundle_corpus_signature,
)
from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (  # noqa: E402
    resolve_full_doc_diagnostic_benchmark,
)
from src.ctreepo.sim.suite.markov_observed_token_policy import (  # noqa: E402
    resolve_markov_observed_token_policy,
)


def _parse_int_list(raw: str) -> List[int]:
    return sorted({int(v) for v in raw.split() if v.strip()})


def _parse_name_list(raw: str) -> List[str]:
    return [str(v).strip() for v in str(raw).split() if str(v).strip()]


def _safe_benchmark_dir_name(benchmark_name: str) -> str:
    return (
        str(benchmark_name)
        .strip()
        .replace("::", "__")
        .replace("/", "_")
        .replace(":", "_")
    )


def _oracle_root_count(doc: Any) -> int:
    """Exact changepoint count from ground truth boundaries."""
    return len(doc.true_boundaries)


def _oracle_leaf_counts(doc: Any, *, leaf_tokens: int) -> List[int]:
    """Exact changepoint count per leaf span."""
    n_tokens = len(doc.tokens)
    regimes = doc.token_regimes
    counts: List[int] = []
    i = 0
    while i < n_tokens:
        j = min(n_tokens, i + leaf_tokens)
        span_regimes = regimes[i:j]
        count = sum(
            1 for a, b in zip(span_regimes[:-1], span_regimes[1:])
            if int(a) != int(b)
        )
        counts.append(count)
        i = j
    return counts


def _doc_to_dict(doc: Any) -> Dict[str, Any]:
    return {
        "tokens": [int(t) for t in doc.tokens],
        "token_regimes": [int(r) for r in doc.token_regimes],
        "transition_regimes": [int(r) for r in doc.transition_regimes],
        "true_boundaries": [int(b) for b in doc.true_boundaries],
        "oracle_root_count": _oracle_root_count(doc),
    }


def _corpus_signature(docs: Sequence[Any]) -> str:
    h = hashlib.sha256()
    for doc in docs:
        h.update(
            json.dumps(_doc_to_dict(doc), sort_keys=True).encode("utf-8")
        )
        h.update(b"\n")
    return h.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a single large Markov corpus for parity-grid experiments."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs")
        / f"markov_parity_corpus_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--benchmark", default="recoverable_v4")
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="",
        help=(
            "Optional space-separated benchmark list to materialize under one "
            "corpus root. Defaults to --benchmark only."
        ),
    )
    parser.add_argument("--total-docs", type=int, default=50000)
    parser.add_argument(
        "--train-doc-counts",
        type=str,
        default="1024 2048 4096 10240 20480",
        help="Space-separated nested training prefix sizes.",
    )
    parser.add_argument("--test-docs", type=int, default=1024)
    parser.add_argument("--val-docs", type=int, default=2048)
    parser.add_argument(
        "--leaf-token-sizes",
        type=str,
        default="16 32 64 128",
        help="Space-separated leaf token sizes to precompute FNO docs for.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _prepare_single_benchmark_corpus(
    *,
    output_root: Path,
    benchmark_name: str,
    total_docs: int,
    train_prefix_counts: Sequence[int],
    test_docs_count: int,
    val_docs_count: int,
    leaf_token_sizes: Sequence[int],
    seed: int,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    max_train = max(int(value) for value in train_prefix_counts)

    spec = resolve_full_doc_diagnostic_benchmark(str(benchmark_name))
    policy = resolve_markov_observed_token_policy(
        profile_name=str(spec.observed_token_profile),
    )
    config_overrides = dict(getattr(spec, "config_overrides", {}) or {})

    config = OPSCountConfig(
        n_regimes=int(config_overrides.get("n_regimes", policy.n_regimes)),
        vocab_size=int(config_overrides.get("vocab_size", policy.vocab_size)),
        generator_profile=str(
            config_overrides.get("generator_profile", policy.generator_profile)
        ),
        min_tokens=int(config_overrides.get("min_tokens", policy.min_tokens)),
        max_tokens=int(config_overrides.get("max_tokens", policy.max_tokens)),
        min_segments=int(config_overrides.get("min_segments", policy.min_segments)),
        max_segments=int(config_overrides.get("max_segments", policy.max_segments)),
        fixed_leaf_tokens=int(
            config_overrides.get("fixed_leaf_tokens", policy.fixed_leaf_tokens)
        ),
        train_docs=int(total_docs),
        seed=int(seed),
    )

    print(f"Generating {total_docs} documents...")
    print(f"  benchmark:  {benchmark_name}")
    print(f"  profile:    {spec.observed_token_profile}")
    print(f"  doc_tokens: {policy.min_tokens}-{policy.max_tokens}")
    print(f"  generator:  {config.generator_profile}")
    print(f"  seed:       {seed}")

    transitions = _build_generator_transitions(config, seed=int(seed))
    all_docs = _generate_ops_count_docs(
        config,
        n_docs=int(total_docs),
        seed=int(seed),
        transitions=transitions,
    )

    train_pool = all_docs[:max_train]
    test_set = all_docs[max_train : max_train + test_docs_count]
    val_set = all_docs[max_train + test_docs_count : max_train + test_docs_count + val_docs_count]

    print(f"  train pool: {len(train_pool)} docs (prefixes: {list(train_prefix_counts)})")
    print(f"  test set:   {len(test_set)} docs")
    print(f"  val set:    {len(val_set)} docs")

    prefix_signatures: Dict[str, str] = {}
    for count in train_prefix_counts:
        sig = _corpus_signature(train_pool[: int(count)])
        prefix_signatures[str(int(count))] = sig
        print(f"  prefix {int(count):>6d}: {sig[:16]}...")

    test_signature = _corpus_signature(test_set)
    val_signature = _corpus_signature(val_set)
    full_signature = _corpus_signature(all_docs)

    print(f"  test sig:   {test_signature[:16]}...")
    print(f"  val sig:    {val_signature[:16]}...")
    print(f"  full sig:   {full_signature[:16]}...")

    corpus_path = output_root / "corpus_docs.jsonl"
    print(f"Writing {total_docs} docs to {corpus_path}...")
    with open(corpus_path, "w", encoding="utf-8") as handle:
        for doc in all_docs:
            handle.write(json.dumps(_doc_to_dict(doc), sort_keys=True) + "\n")

    bundles_dir = output_root / "bundles"
    bundles_dir.mkdir(parents=True, exist_ok=True)
    bundle_paths: Dict[str, str] = {}
    for count in train_prefix_counts:
        train_prefix = train_pool[: int(count)]
        bundle = MarkovOPSDataBundle(
            train_docs=train_prefix,
            val_docs=val_set,
            test_docs=test_set,
            train_corpus_signature=_bundle_corpus_signature(train_prefix),
            val_corpus_signature=_bundle_corpus_signature(val_set),
            test_corpus_signature=_bundle_corpus_signature(test_set),
        )
        bundle_path = bundles_dir / f"bundle_train{int(count)}.pkl"
        bundle.save(bundle_path)
        bundle_paths[str(int(count))] = str(bundle_path)
        print(f"  bundle train={int(count):>6d}: {bundle_path.name}")

    import numpy as np

    test_oracle_counts = [_oracle_root_count(doc) for doc in test_set]
    oracle_stats: Dict[str, Any] = {
        "test_set": {
            "mean": float(np.mean(test_oracle_counts)),
            "std": float(np.std(test_oracle_counts)),
            "min": int(np.min(test_oracle_counts)),
            "max": int(np.max(test_oracle_counts)),
            "n_docs": len(test_set),
        },
    }
    doc_tokens = int(policy.min_tokens)
    for leaf_tokens in (16, 32, 64, doc_tokens):
        if leaf_tokens > doc_tokens:
            continue
        leaf_counts_flat = []
        for doc in test_set:
            leaf_counts_flat.extend(_oracle_leaf_counts(doc, leaf_tokens=leaf_tokens))
        n_leaves = doc_tokens // leaf_tokens if leaf_tokens <= doc_tokens else 1
        oracle_stats[f"leaf_{leaf_tokens}"] = {
            "n_leaves_per_doc": n_leaves,
            "leaf_count_mean": float(np.mean(leaf_counts_flat)),
            "leaf_count_std": float(np.std(leaf_counts_flat)),
        }
    print(
        f"  oracle test mean: {oracle_stats['test_set']['mean']:.2f} "
        f"(std={oracle_stats['test_set']['std']:.2f})"
    )

    from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (
        _ensure_prepared_markov_tree_data,
    )

    prepared_data_root = str(output_root / "prepared_tree_data")
    precomputed_paths: Dict[str, Dict[str, str]] = {}
    largest_bundle = MarkovOPSDataBundle(
        train_docs=train_pool,
        val_docs=val_set,
        test_docs=test_set,
        train_corpus_signature=_bundle_corpus_signature(train_pool),
        val_corpus_signature=_bundle_corpus_signature(val_set),
        test_corpus_signature=_bundle_corpus_signature(test_set),
    )
    for leaf_tokens in sorted({int(v) for v in leaf_token_sizes if int(v) > 0}):
        print(f"  precomputing tree data for leaf={leaf_tokens}...")
        prepared = _ensure_prepared_markov_tree_data(
            benchmark=spec,
            base_bundle=largest_bundle,
            required_train_docs=int(max_train),
            train_prefix_counts=train_prefix_counts,
            fixed_leaf_tokens=int(leaf_tokens),
            max_internal_depth=0,
            seeds=(int(seed),),
            prepared_data_root=prepared_data_root,
            allow_create=True,
        )
        precomputed_paths[str(int(leaf_tokens))] = {
            "root": str(prepared.root),
            "signature": str(prepared.signature),
        }
        print(f"    -> {prepared.root}")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(benchmark_name),
        "observed_token_profile": str(spec.observed_token_profile),
        "total_docs": int(total_docs),
        "seed": int(seed),
        "doc_tokens": int(policy.min_tokens),
        "generator_profile": str(config.generator_profile),
        "n_regimes": int(config.n_regimes),
        "vocab_size": int(config.vocab_size),
        "partition": {
            "train_pool_size": int(max_train),
            "test_size": int(len(test_set)),
            "val_size": int(len(val_set)),
            "train_prefix_counts": [int(value) for value in train_prefix_counts],
        },
        "signatures": {
            "full_corpus": str(full_signature),
            "test": str(test_signature),
            "val": str(val_signature),
            "train_prefixes": dict(prefix_signatures),
        },
        "bundle_paths": dict(bundle_paths),
        "prepared_data_root": str(prepared_data_root),
        "precomputed_tree_data": dict(precomputed_paths),
        "oracle_stats": dict(oracle_stats),
    }
    manifest_path = output_root / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest written to {manifest_path}")
    return manifest


def main() -> int:
    args = _parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_prefix_counts = _parse_int_list(str(args.train_doc_counts))
    max_train = max(train_prefix_counts)
    test_docs_count = int(args.test_docs)
    total_needed = max_train + test_docs_count
    total_docs = max(int(args.total_docs), total_needed)
    val_docs_count = int(args.val_docs)
    leaf_token_sizes = sorted(
        {int(v) for v in str(args.leaf_token_sizes).split() if v.strip()}
    )
    benchmark_names = _parse_name_list(str(args.benchmarks))
    if not benchmark_names:
        benchmark_names = [str(args.benchmark)]

    benchmark_payloads: Dict[str, Dict[str, Any]] = {}
    if len(benchmark_names) == 1:
        benchmark_name = str(benchmark_names[0])
        benchmark_payloads[benchmark_name] = _prepare_single_benchmark_corpus(
            output_root=output_root,
            benchmark_name=benchmark_name,
            total_docs=int(total_docs),
            train_prefix_counts=train_prefix_counts,
            test_docs_count=int(test_docs_count),
            val_docs_count=int(val_docs_count),
            leaf_token_sizes=leaf_token_sizes,
            seed=int(args.seed),
        )
    else:
        benchmark_roots_dir = output_root / "benchmark_corpora"
        for benchmark_name in benchmark_names:
            benchmark_root = benchmark_roots_dir / _safe_benchmark_dir_name(benchmark_name)
            benchmark_payloads[str(benchmark_name)] = _prepare_single_benchmark_corpus(
                output_root=benchmark_root,
                benchmark_name=str(benchmark_name),
                total_docs=int(total_docs),
                train_prefix_counts=train_prefix_counts,
                test_docs_count=int(test_docs_count),
                val_docs_count=int(val_docs_count),
                leaf_token_sizes=leaf_token_sizes,
                seed=int(args.seed),
            )

    primary_benchmark = str(args.benchmark)
    primary_manifest = dict(benchmark_payloads[primary_benchmark])
    manifest = {
        **primary_manifest,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": primary_benchmark,
        "requested_benchmarks": [str(name) for name in benchmark_names],
        "benchmarks": {
            str(name): {
                **dict(payload),
                "root": str(
                    output_root
                    if len(benchmark_names) == 1
                    else output_root / "benchmark_corpora" / _safe_benchmark_dir_name(str(name))
                ),
                "manifest_path": str(
                    (
                        output_root
                        if len(benchmark_names) == 1
                        else output_root / "benchmark_corpora" / _safe_benchmark_dir_name(str(name))
                    )
                    / "corpus_manifest.json"
                ),
            }
            for name, payload in benchmark_payloads.items()
        },
        "precomputed_tree_data_by_benchmark": {
            str(name): dict(payload.get("precomputed_tree_data") or {})
            for name, payload in benchmark_payloads.items()
        },
    }
    (output_root / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
