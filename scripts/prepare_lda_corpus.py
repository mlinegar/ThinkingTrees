#!/usr/bin/env python3
"""Thin CLI over :func:`prepare_lda_corpus`.

Materializes a segmented-LDA corpus under the shared
``data/processed/lda/<name>/`` convention (id-based split + shared manifest).

Example::

    python scripts/prepare_lda_corpus.py --name smoke \
        --train-docs 64 --test-docs 32
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.ctreepo.data.prep_common import ensure_repo_on_path

ensure_repo_on_path()

from src.ctreepo.sim.core.lda_data_prep import (  # noqa: E402
    LDAPrepConfig,
    prepare_lda_corpus,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="default")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--vocab-size", type=int, default=LDAPrepConfig.vocab_size)
    parser.add_argument("--n-topics", type=int, default=LDAPrepConfig.n_topics)
    parser.add_argument("--min-tokens", type=int, default=LDAPrepConfig.min_tokens)
    parser.add_argument("--max-tokens", type=int, default=LDAPrepConfig.max_tokens)
    parser.add_argument("--leaf-tokens", type=int, default=LDAPrepConfig.leaf_tokens)
    parser.add_argument("--train-docs", type=int, default=LDAPrepConfig.train_docs)
    parser.add_argument("--val-docs", type=int, default=LDAPrepConfig.val_docs)
    parser.add_argument("--test-docs", type=int, default=LDAPrepConfig.test_docs)
    parser.add_argument("--seed", type=int, default=LDAPrepConfig.seed)
    parser.add_argument("--no-write-docs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = LDAPrepConfig(
        vocab_size=int(args.vocab_size),
        n_topics=int(args.n_topics),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        leaf_tokens=int(args.leaf_tokens),
        train_docs=int(args.train_docs),
        val_docs=int(args.val_docs),
        test_docs=int(args.test_docs),
        seed=int(args.seed),
    )
    prepared = prepare_lda_corpus(
        config,
        out_dir=args.out_dir,
        name=str(args.name),
        write_docs=not bool(args.no_write_docs),
    )
    print(f"Wrote {prepared.manifest_path}")
    if prepared.docs_path is not None:
        print(f"Wrote {prepared.docs_path}")
    print(f"Splits: {prepared.split.counts()}")
    print(prepared.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
