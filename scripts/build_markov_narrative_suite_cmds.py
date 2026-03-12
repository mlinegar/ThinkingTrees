#!/usr/bin/env python3
"""Build xargs-friendly command lists for a compact Markov narrative sweep.

This sweep is meant to answer a specific paper-facing question:
when does the learned sketch *fail* vs *begin to work* as we add supervision?

We therefore include:
- leaf_query_rate sweeps (leaf supervision),
- internal-node audit_fraction sweeps (C3 supervision),
- (optional) root supervision on/off,
- model_family sweeps (neural vs additive).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List


def _parse_items(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text).replace(",", " ").split():
        x = raw.strip()
        if x:
            out.append(x)
    return out


def _parse_ints(text: str) -> List[int]:
    return [int(x) for x in _parse_items(text)]


def _parse_floats(text: str) -> List[float]:
    return [float(x) for x in _parse_items(text)]


def _fmt_float(x: float) -> str:
    s = f"{float(x):.6g}"
    return s.replace("-", "m").replace(".", "p")


def _iter_cmds(
    *,
    python_bin: str,
    output_root: Path,
    train_docs: Iterable[int],
    test_docs: int,
    fixed_leaf_tokens: Iterable[int],
    model_families: Iterable[str],
    audit_fractions: Iterable[float],
    leaf_query_rates: Iterable[float],
    include_root_queries: Iterable[bool],
    c3_audit_strategy: str,
    feature_mode: str,
    n_epochs: int,
    local_law_weight: float,
    c1_relative_weight: float,
    c3_relative_weight: float,
    schedule_consistency_weight: float,
    seeds: Iterable[int],
    device: str,
    torch_threads: int,
    skip_existing: bool,
) -> List[str]:
    script = "scripts/run_markov_changepoint_ops_count_simulation.py"
    cmds: List[str] = []
    for leaf_tokens in fixed_leaf_tokens:
        for td in train_docs:
            for fam in model_families:
                for rootq in include_root_queries:
                    for frac in audit_fractions:
                        for lqr in leaf_query_rates:
                            for seed in seeds:
                                sub = (
                                    f"leaf_{int(leaf_tokens)}/train_{int(td)}/model_{str(fam)}"
                                    f"/rootq_{1 if rootq else 0}"
                                    f"/budget_{_fmt_float(frac)}/lqr_{_fmt_float(lqr)}"
                                )
                                base = output_root / sub / f"seed_{int(seed)}"
                                out_json = base.with_suffix(".json")
                                out_csv = base.with_suffix(".csv")
                                if skip_existing and out_json.exists() and out_csv.exists():
                                    continue
                                parts: List[str] = [
                                    f"{python_bin} -u {script}",
                                    f"--train-docs {int(td)}",
                                    f"--test-docs {int(test_docs)}",
                                    f"--fixed-leaf-tokens {int(leaf_tokens)}",
                                    f"--model-family {str(fam)}",
                                    "--audit-policy fraction",
                                    f"--audit-fraction {float(frac)}",
                                    f"--c3-audit-strategy {str(c3_audit_strategy)}",
                                    f"--leaf-query-rate {float(lqr)}",
                                    f"--local-law-weight {float(local_law_weight)}",
                                    f"--c1-relative-weight {float(c1_relative_weight)}",
                                    f"--c3-relative-weight {float(c3_relative_weight)}",
                                    "--root-weight 1.0",
                                    f"--schedule-consistency-weight {float(schedule_consistency_weight)}",
                                    f"--feature-mode {str(feature_mode)}",
                                    f"--n-epochs {int(n_epochs)}",
                                    f"--device {str(device)}",
                                    f"--torch-threads {int(torch_threads)}",
                                    "--violation-tau 0.0",
                                    f"--seed {int(seed)}",
                                ]
                                if not rootq:
                                    parts.append("--no-root-query")
                                parts.extend([f"--json-summary {out_json}", f"--csv-summary {out_csv}"])
                                cmds.append(" ".join(parts))
    return cmds


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Markov narrative sweep command list.")
    p.add_argument("--python-bin", type=str, default="venv/bin/python")
    p.add_argument("--out-cmds", type=Path, default=Path("logs/markov_narrative_suite_cmds.txt"))
    p.add_argument("--output-root", type=Path, default=Path("outputs/markov_narrative_suite"))
    p.add_argument("--train-docs", type=str, default="200")
    p.add_argument("--test-docs", type=int, default=1000)
    p.add_argument("--fixed-leaf-tokens", type=str, default="16")
    p.add_argument("--model-families", type=str, default="neural additive")
    p.add_argument("--audit-fractions", type=str, default="0 0.01 0.1 1.0")
    p.add_argument("--leaf-query-rates", type=str, default="0 0.25 1.0")
    p.add_argument(
        "--include-root-query",
        type=str,
        default="true false",
        help="Space/comma list of booleans; include both to visualize root-supervision effects.",
    )
    p.add_argument("--c3-audit-strategy", type=str, default="uniform")
    p.add_argument("--feature-mode", type=str, default="full")
    p.add_argument("--n-epochs", type=int, default=12)
    p.add_argument("--local-law-weight", type=float, default=0.0)
    p.add_argument("--c1-relative-weight", type=float, default=1.0)
    p.add_argument("--c3-relative-weight", type=float, default=4.0)
    p.add_argument("--schedule-consistency-weight", type=float, default=0.0)
    p.add_argument("--seeds", type=str, default="0 1 2 3")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _parse_bools(text: str) -> List[bool]:
    out: List[bool] = []
    for raw in _parse_items(text):
        x = raw.strip().lower()
        if x in ("1", "true", "t", "yes", "y"):
            out.append(True)
        elif x in ("0", "false", "f", "no", "n"):
            out.append(False)
        else:
            raise ValueError(f"could not parse boolean: {raw!r}")
    return out


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
        audit_fractions=_parse_floats(args.audit_fractions),
        leaf_query_rates=_parse_floats(args.leaf_query_rates),
        include_root_queries=_parse_bools(args.include_root_query),
        c3_audit_strategy=str(args.c3_audit_strategy),
        feature_mode=str(args.feature_mode),
        n_epochs=int(args.n_epochs),
        local_law_weight=float(args.local_law_weight),
        c1_relative_weight=float(args.c1_relative_weight),
        c3_relative_weight=float(args.c3_relative_weight),
        schedule_consistency_weight=float(args.schedule_consistency_weight),
        seeds=_parse_ints(args.seeds),
        device=str(args.device),
        torch_threads=int(args.torch_threads),
        skip_existing=bool(args.skip_existing),
    )

    args.out_cmds.write_text("\n".join(cmds) + ("\n" if cmds else ""), encoding="utf-8")
    print(f"wrote_cmds | {args.out_cmds} | n_commands={len(cmds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
