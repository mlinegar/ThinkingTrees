#!/usr/bin/env python3
"""Build xargs-friendly command lists for the Stage-1 LDA utility-vector family."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
from typing import Iterable, List


def _parse_items(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text).replace(",", " ").split():
        item = raw.strip()
        if item:
            out.append(item)
    return out


def _parse_ints(text: str) -> List[int]:
    return [int(x) for x in _parse_items(text)]


def _parse_floats(text: str) -> List[float]:
    return [float(x) for x in _parse_items(text)]


def _parse_fraction_text(text: str) -> List[str]:
    return _parse_items(text)


def _fraction_dir_label(text: str) -> str:
    frac = Fraction(str(text))
    if frac.denominator == 1:
        return "doc_100pct"
    pct = 100.0 * float(frac)
    if abs(pct - round(pct)) <= 1e-9:
        return f"doc_{int(round(pct))}pct"
    return f"doc_{str(text).replace('/', 'of').replace('.', 'p')}"


def _iter_commands(
    *,
    python_bin: str,
    output_root: Path,
    leaf_fractions: Iterable[str],
    doc_topic_concentrations: Iterable[float],
    state_dims: Iterable[int],
    seeds: Iterable[int],
    utility_dim: int,
    doc_tokens: int,
    train_docs: int,
    test_docs: int,
    n_topics: int,
    vocab_size: int,
    run_full_doc_mlp_diag: bool,
    skip_existing: bool,
) -> List[str]:
    script = "scripts/run_lda_tree_utility_vector_simulation.py"
    cmds: List[str] = []
    for leaf_frac in leaf_fractions:
        leaf_label = _fraction_dir_label(leaf_frac)
        for alpha in doc_topic_concentrations:
            for state_dim in state_dims:
                for seed in seeds:
                    base = (
                        output_root
                        / leaf_label
                        / f"dtc_{alpha:g}"
                        / f"state_{int(state_dim)}"
                        / f"seed_{int(seed)}"
                    )
                    out_json = base.with_suffix(".json")
                    out_csv = base.with_suffix(".csv")
                    if skip_existing and out_json.exists() and out_csv.exists():
                        continue
                    cmd = (
                        f"{python_bin} -u {script} "
                        f"--n-topics {int(n_topics)} --vocab-size {int(vocab_size)} "
                        f"--doc-tokens {int(doc_tokens)} "
                        f"--doc-topic-concentration {float(alpha)} "
                        f"--utility-dim {int(utility_dim)} "
                        f"--leaf-fraction {leaf_frac} "
                        f"--train-docs {int(train_docs)} --test-docs {int(test_docs)} "
                        f"--state-dim {int(state_dim)} "
                        f"--seed {int(seed)} "
                        f"--json-summary {out_json} --csv-summary {out_csv}"
                    )
                    if not bool(run_full_doc_mlp_diag):
                        cmd += " --no-run-full-doc-mlp-diag"
                    cmds.append(cmd)
    return cmds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build command lists for Stage-1 LDA utility-vector sweeps.")
    p.add_argument("--python-bin", type=str, default="venv/bin/python")
    p.add_argument("--out-cmds", type=str, default="logs/lda_tree_utility_vector_cmds.txt")
    p.add_argument("--output-root", type=str, default="outputs/lda_tree_utility_vector")
    p.add_argument("--leaf-fractions", type=str, default="1 1/2 1/4 1/24")
    p.add_argument("--doc-topic-concentrations", type=str, default="0.2 0.6 1.5")
    p.add_argument("--state-dims", type=str, default="4 8 16 32 64 128 256 512")
    p.add_argument("--seeds", type=str, default="0 1 2 3 4")
    p.add_argument("--utility-dim", type=int, default=16)
    p.add_argument("--doc-tokens", type=int, default=384)
    p.add_argument("--train-docs", type=int, default=512)
    p.add_argument("--test-docs", type=int, default=256)
    p.add_argument("--n-topics", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument(
        "--run-full-doc-mlp-diag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include the appendix-only full-document MLP diagnostic in each run.",
    )
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_cmds = Path(args.out_cmds)
    out_cmds.parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    cmds = _iter_commands(
        python_bin=str(args.python_bin),
        output_root=Path(args.output_root),
        leaf_fractions=_parse_fraction_text(args.leaf_fractions),
        doc_topic_concentrations=_parse_floats(args.doc_topic_concentrations),
        state_dims=_parse_ints(args.state_dims),
        seeds=_parse_ints(args.seeds),
        utility_dim=int(args.utility_dim),
        doc_tokens=int(args.doc_tokens),
        train_docs=int(args.train_docs),
        test_docs=int(args.test_docs),
        n_topics=int(args.n_topics),
        vocab_size=int(args.vocab_size),
        run_full_doc_mlp_diag=bool(args.run_full_doc_mlp_diag),
        skip_existing=bool(args.skip_existing),
    )
    out_cmds.write_text("\n".join(cmds) + ("\n" if cmds else ""), encoding="utf-8")
    print(f"wrote_cmds | {out_cmds} | n_commands={len(cmds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
