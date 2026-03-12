#!/usr/bin/env python3
"""Build command files for the LDA local-law stress suites.

Mirrors the Markov law-stress suite structure for cross-DGP consistency:
- sanity_suite:     law packages + exact families on a small grid
- transition_map:   tau x lambda x law_package heatmap data
- mechanism_suite:  boundary cells from transition map
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CLI_SCRIPT = "scripts/run_leaf_local_mixture_utility_simulation.py"


@dataclass(frozen=True)
class LDALawStressCommand:
    command: str
    json_path: Path
    csv_path: Path
    artifact_dir: Path


def _build_cmd(
    *,
    python_bin: str,
    output_root: Path,
    tau: float,
    lam: float,
    train_docs: int,
    test_docs: int,
    law_package: str,
    exact_family: str,
    local_law_mode: str,
    law_leaf_query_rate: float,
    law_internal_query_rate: float,
    analysis_partition_mode: str,
    seed: int,
    suite_role: str,
) -> LDALawStressCommand:
    slug = f"tau{tau:g}_lam{lam:g}_pkg_{law_package}_mode_{analysis_partition_mode}_s{seed}"
    if exact_family:
        slug = f"fam_{exact_family}_{slug}"
    json_path = output_root / "results" / suite_role / f"{slug}.json"
    csv_path = output_root / "results" / suite_role / f"{slug}.csv"
    artifact_dir = output_root / "results" / suite_role / f"{slug}_artifacts"
    parts = [
        python_bin, CLI_SCRIPT,
        f"--local-mixture-concentration {tau}",
        f"--lambda-multiplier {lam}",
        f"--train-docs {train_docs}",
        f"--test-docs {test_docs}",
        f"--local-law-mode {local_law_mode}",
        f"--law-package {law_package}",
        f"--law-leaf-query-rate {law_leaf_query_rate}",
        f"--law-internal-query-rate {law_internal_query_rate}",
        f"--analysis-partition-mode {analysis_partition_mode}",
        f"--seed {seed}",
        f"--json-summary {json_path}",
        f"--csv-summary {csv_path}",
        f"--artifact-dir {artifact_dir}",
    ]
    if exact_family:
        parts.append(f"--exact-family {exact_family}")
    return LDALawStressCommand(
        command=" ".join(parts),
        json_path=json_path,
        csv_path=csv_path,
        artifact_dir=artifact_dir,
    )


def _write_cmd_file(path: Path, cmds: Sequence[LDALawStressCommand]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(str(item.command) for item in cmds) + ("\n" if cmds else ""),
        encoding="utf-8",
    )


def _append_if_needed(
    cmds: List[LDALawStressCommand],
    item: LDALawStressCommand,
    *,
    skip_existing: bool,
) -> None:
    if bool(skip_existing) and item.json_path.exists() and item.csv_path.exists():
        return
    cmds.append(item)


def _build_sanity_suite(
    *,
    python_bin: str,
    suite_root: Path,
    smoke: bool,
    skip_existing: bool,
) -> List[LDALawStressCommand]:
    """Small grid confirming ablation signal + exact families."""
    taus = [1.0, 8.0] if smoke else [1.0, 4.0, 16.0]
    lams = [0.0, 1.5] if smoke else [0.0, 0.5, 1.5]
    seeds = [0] if smoke else [0, 1, 2]
    train_docs = 32 if smoke else 128
    test_docs = 16 if smoke else 64

    learned_cmds: List[LDALawStressCommand] = []
    for tau, lam, pkg, seed in itertools.product(
        taus, lams,
        ["root_only", "c1_only", "c3_only", "all_laws"],
        seeds,
    ):
        _append_if_needed(learned_cmds, _build_cmd(
            python_bin=python_bin,
            output_root=suite_root,
            tau=tau, lam=lam,
            train_docs=train_docs, test_docs=test_docs,
            law_package=pkg,
            exact_family="",
            local_law_mode="diagnostics_and_learned",
            law_leaf_query_rate=0.25,
            law_internal_query_rate=0.25,
            analysis_partition_mode="aligned",
            seed=seed,
            suite_role="sanity_learned",
        ), skip_existing=skip_existing)

    exact_cmds: List[LDALawStressCommand] = []
    for tau, lam, fam, seed in itertools.product(
        taus, lams,
        ["oracle", "scrambled_topics", "uniform_prior", "adversarial_merge"],
        seeds,
    ):
        _append_if_needed(exact_cmds, _build_cmd(
            python_bin=python_bin,
            output_root=suite_root,
            tau=tau, lam=lam,
            train_docs=train_docs, test_docs=test_docs,
            law_package="all_laws",
            exact_family=fam,
            local_law_mode="diagnostics",
            law_leaf_query_rate=0.25,
            law_internal_query_rate=0.25,
            analysis_partition_mode="aligned",
            seed=seed,
            suite_role="sanity_exact",
        ), skip_existing=skip_existing)

    return learned_cmds + exact_cmds


def _build_transition_map_suite(
    *,
    python_bin: str,
    suite_root: Path,
    smoke: bool,
    skip_existing: bool,
) -> List[LDALawStressCommand]:
    """tau x lambda x law_package grid for heatmap data."""
    taus = [1.0, 8.0] if smoke else [1.0, 2.0, 4.0, 8.0, 16.0]
    lams = [0.0, 1.5] if smoke else [0.0, 0.1, 0.5, 1.0, 1.5, 3.0]
    packages = ["root_only", "all_laws"] if smoke else ["root_only", "c1_only", "c3_only", "c1c3", "all_laws"]
    seeds = [0] if smoke else [0, 1, 2, 3]
    train_docs = 32 if smoke else 256
    test_docs = 16 if smoke else 128

    cmds: List[LDALawStressCommand] = []
    for tau, lam, pkg, seed in itertools.product(taus, lams, packages, seeds):
        _append_if_needed(cmds, _build_cmd(
            python_bin=python_bin,
            output_root=suite_root,
            tau=tau, lam=lam,
            train_docs=train_docs, test_docs=test_docs,
            law_package=pkg,
            exact_family="",
            local_law_mode="diagnostics_and_learned",
            law_leaf_query_rate=0.10,
            law_internal_query_rate=0.10,
            analysis_partition_mode="aligned",
            seed=seed,
            suite_role="transition_map",
        ), skip_existing=skip_existing)
    return cmds


def _build_mechanism_suite(
    *,
    python_bin: str,
    suite_root: Path,
    smoke: bool,
    skip_existing: bool,
) -> List[LDALawStressCommand]:
    """Mismatch x law_package grid — shows when mismatch breaks laws."""
    taus = [1.0, 8.0] if smoke else [1.0, 4.0, 16.0]
    lams = [1.5] if smoke else [0.5, 1.5]
    modes = ["aligned", "shift_half"] if smoke else ["aligned", "coarsen_2x", "shift_half", "random_same_count"]
    packages = ["all_laws"] if smoke else ["root_only", "all_laws"]
    seeds = [0] if smoke else [0, 1, 2, 3]
    train_docs = 32 if smoke else 256
    test_docs = 16 if smoke else 128

    cmds: List[LDALawStressCommand] = []
    for tau, lam, mode, pkg, seed in itertools.product(taus, lams, modes, packages, seeds):
        _append_if_needed(cmds, _build_cmd(
            python_bin=python_bin,
            output_root=suite_root,
            tau=tau, lam=lam,
            train_docs=train_docs, test_docs=test_docs,
            law_package=pkg,
            exact_family="",
            local_law_mode="diagnostics_and_learned",
            law_leaf_query_rate=0.10,
            law_internal_query_rate=0.10,
            analysis_partition_mode=mode,
            seed=seed,
            suite_role="mechanism",
        ), skip_existing=skip_existing)
    return cmds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build LDA law-stress suite command files.")
    p.add_argument("--suite", type=str, required=True,
                   choices=["sanity_suite", "transition_map_suite", "mechanism_suite", "all"])
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--cmd-dir", type=Path, required=True)
    p.add_argument("--python-bin", type=str, default="venv/bin/python")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    cmd_dir = Path(args.cmd_dir)
    cmd_dir.mkdir(parents=True, exist_ok=True)

    suites_to_build = (
        ["sanity_suite", "transition_map_suite", "mechanism_suite"]
        if args.suite == "all"
        else [args.suite]
    )

    total = 0
    meta: Dict[str, object] = {"suites": {}}
    for suite in suites_to_build:
        suite_root = output_root / suite
        if suite == "sanity_suite":
            cmds = _build_sanity_suite(
                python_bin=args.python_bin,
                suite_root=suite_root,
                smoke=args.smoke,
                skip_existing=bool(args.skip_existing),
            )
        elif suite == "transition_map_suite":
            cmds = _build_transition_map_suite(
                python_bin=args.python_bin,
                suite_root=suite_root,
                smoke=args.smoke,
                skip_existing=bool(args.skip_existing),
            )
        elif suite == "mechanism_suite":
            cmds = _build_mechanism_suite(
                python_bin=args.python_bin,
                suite_root=suite_root,
                smoke=args.smoke,
                skip_existing=bool(args.skip_existing),
            )
        else:
            raise ValueError(f"Unknown suite: {suite}")

        cmd_file = cmd_dir / f"lda_law_stress_{suite}_cmds.txt"
        _write_cmd_file(cmd_file, cmds)
        meta["suites"][suite] = {"n_commands": len(cmds), "cmd_file": str(cmd_file)}  # type: ignore[index]
        total += len(cmds)
        print(f"{suite}: {len(cmds)} commands -> {cmd_file}")

    meta["total_commands"] = total
    meta_path = cmd_dir / "lda_law_stress_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"total: {total} commands")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
