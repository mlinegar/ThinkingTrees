#!/usr/bin/env python3
"""Build focused Stage-2 follow-up command lists for the tree-relevant LDA report."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_leaf_local_mixture_utility_cmds import _iter_commands


@dataclass(frozen=True)
class FollowupSuite:
    name: str
    purpose: str
    latent_leaf_tokens: str
    leaf_fractions: str
    doc_topic_concentrations: str
    local_mixture_concentrations: str
    lambda_grid: str
    seeds: str
    budget_regimes: str = "all_leaves_labeled"
    leaf_label_budgets: str = "8"


@dataclass(frozen=True)
class FollowupCommand:
    suite: str
    purpose: str
    cmd: str
    json_summary: str


def _parse_items(text: str) -> List[str]:
    return [item for item in str(text).replace(",", " ").split() if item.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(x) for x in _parse_items(text)]


def _parse_floats(text: str) -> List[float]:
    return [float(x) for x in _parse_items(text)]


def _extract_flag(cmd: str, flag: str) -> str:
    needle = f"{flag} "
    if needle not in cmd:
        return ""
    return cmd.split(needle, 1)[1].strip().split()[0]


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build focused tree-relevant LDA follow-up command lists.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--cmd-file", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--matrix-md", type=Path, required=True)
    p.add_argument("--python-bin", type=str, default="venv/bin/python")
    p.add_argument("--train-docs", type=int, default=512)
    p.add_argument("--test-docs", type=int, default=512)
    p.add_argument("--doc-tokens", type=int, default=384)
    p.add_argument("--crossover-seeds", type=str, default="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23")
    p.add_argument("--lambda-seeds", type=str, default="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23")
    p.add_argument("--robustness-seeds", type=str, default="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15")
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    suites = [
        FollowupSuite(
            name="tau_crossover_dense",
            purpose="Densify tau around the observed crossover to pin down where leaf inference stops beating pooling.",
            latent_leaf_tokens="16 32 64 96",
            leaf_fractions="1",
            doc_topic_concentrations="0.6",
            local_mixture_concentrations="0.25 0.5 1 2 4 8 16 32 64",
            lambda_grid="2",
            seeds=str(args.crossover_seeds),
        ),
        FollowupSuite(
            name="lambda_onset_dense",
            purpose="Measure how quickly the pooled-vs-leaf gap turns on as lambda moves away from zero.",
            latent_leaf_tokens="64 96",
            leaf_fractions="1",
            doc_topic_concentrations="0.6",
            local_mixture_concentrations="0.25 1 8",
            lambda_grid="0 0.25 0.5 1 1.5 2 3",
            seeds=str(args.lambda_seeds),
        ),
        FollowupSuite(
            name="doc_topic_concentration_robustness",
            purpose="Check that the main crossover story survives changes in the document-level topic concentration alpha.",
            latent_leaf_tokens="64 96",
            leaf_fractions="1",
            doc_topic_concentrations="0.2 0.6 1.5",
            local_mixture_concentrations="0.25 1 8 64",
            lambda_grid="2",
            seeds=str(args.robustness_seeds),
        ),
    ]

    commands: List[FollowupCommand] = []
    suite_counts: List[tuple[str, int]] = []
    for suite in suites:
        suite_root = args.output_root / suite.name
        suite_cmds = _iter_commands(
            python_bin=str(args.python_bin),
            output_root=suite_root,
            leaf_fractions=_parse_items(suite.leaf_fractions),
            doc_topic_concentrations=_parse_floats(suite.doc_topic_concentrations),
            taus=_parse_floats(suite.local_mixture_concentrations),
            lambda_grid=_parse_floats(suite.lambda_grid),
            budgets=_parse_floats(suite.leaf_label_budgets),
            budget_regimes=_parse_items(suite.budget_regimes),
            latent_leaf_tokens_list=_parse_ints(suite.latent_leaf_tokens),
            seeds=_parse_ints(suite.seeds),
            skip_existing=bool(args.skip_existing),
            doc_tokens=int(args.doc_tokens),
            train_docs=int(args.train_docs),
            test_docs=int(args.test_docs),
        )
        suite_counts.append((suite.name, len(suite_cmds)))
        for cmd in suite_cmds:
            commands.append(
                FollowupCommand(
                    suite=suite.name,
                    purpose=suite.purpose,
                    cmd=cmd,
                    json_summary=_extract_flag(cmd, "--json-summary"),
                )
            )

    _write_lines(args.cmd_file, (item.cmd for item in commands))

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as handle:
        for item in commands:
            handle.write(json.dumps(asdict(item), sort_keys=True) + "\n")

    matrix_lines = [
        "# Tree-Relevant LDA Follow-up Matrix",
        "",
        "These follow-up suites are targeted at the exact claims in the updated report.",
        "",
        f"- Train docs per run: `{int(args.train_docs)}`",
        f"- Test docs per run: `{int(args.test_docs)}`",
        f"- Doc tokens: `{int(args.doc_tokens)}`",
        f"- Total commands: `{len(commands)}`",
        "",
        "## Suites",
        "",
    ]
    for suite in suites:
        count = dict(suite_counts).get(suite.name, 0)
        matrix_lines.extend(
            [
                f"### `{suite.name}`",
                "",
                f"- Purpose: {suite.purpose}",
                f"- Latent leaf tokens: `{suite.latent_leaf_tokens}`",
                f"- Document topic concentrations: `{suite.doc_topic_concentrations}`",
                f"- Tau grid: `{suite.local_mixture_concentrations}`",
                f"- Lambda grid: `{suite.lambda_grid}`",
                f"- Seeds: `{suite.seeds}`",
                f"- Commands: `{count}`",
                "",
            ]
        )
    args.matrix_md.parent.mkdir(parents=True, exist_ok=True)
    args.matrix_md.write_text("\n".join(matrix_lines), encoding="utf-8")

    print(f"wrote_cmds | {args.cmd_file} | n={len(commands)}")
    print(f"wrote_manifest | {args.manifest}")
    print(f"wrote_matrix_md | {args.matrix_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
