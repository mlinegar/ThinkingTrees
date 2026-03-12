#!/usr/bin/env python3
"""Build a compact, named-regime Markov supervision sweep for report figures."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class Regime:
    name: str
    audit_fraction: float
    leaf_query_rate: float
    include_root_query: bool


REGIMES: tuple[Regime, ...] = (
    Regime("none", audit_fraction=0.0, leaf_query_rate=0.0, include_root_query=False),
    Regime("sparse_merge", audit_fraction=0.05, leaf_query_rate=0.0, include_root_query=False),
    Regime("full_merge", audit_fraction=1.0, leaf_query_rate=0.0, include_root_query=False),
    Regime("root_only", audit_fraction=0.0, leaf_query_rate=0.0, include_root_query=True),
    Regime("full_local", audit_fraction=1.0, leaf_query_rate=1.0, include_root_query=False),
    Regime("full_direct", audit_fraction=1.0, leaf_query_rate=1.0, include_root_query=True),
)


def _parse_items(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text).replace(",", " ").split():
        item = raw.strip()
        if item:
            out.append(item)
    return out


def _parse_ints(text: str) -> List[int]:
    return [int(item) for item in _parse_items(text)]


def _selected_regimes(text: str) -> List[Regime]:
    wanted = set(_parse_items(text))
    if not wanted:
        return list(REGIMES)
    by_name = {regime.name: regime for regime in REGIMES}
    missing = sorted(wanted.difference(by_name))
    if missing:
        raise ValueError(f"unknown regimes: {', '.join(missing)}")
    return [by_name[name] for name in by_name if name in wanted]


def _iter_cmds(
    *,
    python_bin: str,
    output_root: Path,
    train_docs: Iterable[int],
    test_docs: int,
    fixed_leaf_tokens: Iterable[int],
    model_families: Iterable[str],
    regimes: Iterable[Regime],
    feature_mode: str,
    n_epochs: int,
    local_law_weight: float,
    c1_relative_weight: float,
    c3_relative_weight: float,
    device: str,
    torch_threads: int,
    seeds: Iterable[int],
    skip_existing: bool,
) -> List[str]:
    script = "scripts/run_markov_changepoint_ops_count_simulation.py"
    cmds: List[str] = []
    for leaf_tokens in fixed_leaf_tokens:
        for td in train_docs:
            for family in model_families:
                for regime in regimes:
                    for seed in seeds:
                        out_base = (
                            output_root
                            / f"leaf_{int(leaf_tokens)}"
                            / f"train_{int(td)}"
                            / f"model_{family}"
                            / regime.name
                            / f"seed_{int(seed)}"
                        )
                        out_json = out_base.with_suffix(".json")
                        out_csv = out_base.with_suffix(".csv")
                        if skip_existing and out_json.exists() and out_csv.exists():
                            continue
                        parts = [
                            f"{python_bin} -u {script}",
                            f"--train-docs {int(td)}",
                            f"--test-docs {int(test_docs)}",
                            f"--fixed-leaf-tokens {int(leaf_tokens)}",
                            f"--model-family {family}",
                            f"--feature-mode {feature_mode}",
                            "--audit-policy fraction",
                            f"--audit-fraction {float(regime.audit_fraction)}",
                            "--c3-audit-strategy uniform",
                            f"--leaf-query-rate {float(regime.leaf_query_rate)}",
                            f"--local-law-weight {float(local_law_weight)}",
                            f"--c1-relative-weight {float(c1_relative_weight)}",
                            f"--c3-relative-weight {float(c3_relative_weight)}",
                            "--root-weight 1.0",
                            "--schedule-consistency-weight 0.0",
                            f"--n-epochs {int(n_epochs)}",
                            f"--device {device}",
                            f"--torch-threads {int(torch_threads)}",
                            "--violation-tau 0.0",
                            f"--seed {int(seed)}",
                            f"--json-summary {out_json}",
                            f"--csv-summary {out_csv}",
                        ]
                        if not regime.include_root_query:
                            parts.append("--no-root-query")
                        cmds.append(" ".join(parts))
    return cmds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build named-regime Markov supervision commands.")
    parser.add_argument("--python-bin", type=str, default="venv/bin/python")
    parser.add_argument("--out-cmds", type=Path, default=Path("logs/markov_supervision_narrative_cmds.txt"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/markov_supervision_narrative"),
    )
    parser.add_argument("--train-docs", type=str, default="8000")
    parser.add_argument("--test-docs", type=int, default=256)
    parser.add_argument("--fixed-leaf-tokens", type=str, default="16 32")
    parser.add_argument("--model-families", type=str, default="neural additive")
    parser.add_argument(
        "--regimes",
        type=str,
        default="none sparse_merge full_merge root_only full_local full_direct",
    )
    parser.add_argument("--feature-mode", type=str, default="full")
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--local-law-weight", type=float, default=0.0)
    parser.add_argument("--c1-relative-weight", type=float, default=1.0)
    parser.add_argument("--c3-relative-weight", type=float, default=4.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seeds", type=str, default="0 1")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.out_cmds.parent.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    cmds = _iter_cmds(
        python_bin=str(args.python_bin),
        output_root=args.output_root,
        train_docs=_parse_ints(args.train_docs),
        test_docs=int(args.test_docs),
        fixed_leaf_tokens=_parse_ints(args.fixed_leaf_tokens),
        model_families=_parse_items(args.model_families),
        regimes=_selected_regimes(args.regimes),
        feature_mode=str(args.feature_mode),
        n_epochs=int(args.n_epochs),
        local_law_weight=float(args.local_law_weight),
        c1_relative_weight=float(args.c1_relative_weight),
        c3_relative_weight=float(args.c3_relative_weight),
        device=str(args.device),
        torch_threads=int(args.torch_threads),
        seeds=_parse_ints(args.seeds),
        skip_existing=bool(args.skip_existing),
    )
    args.out_cmds.write_text("\n".join(cmds) + ("\n" if cmds else ""), encoding="utf-8")
    print(f"wrote_cmds | {args.out_cmds} | n_commands={len(cmds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
