#!/usr/bin/env python3
"""Build Stage-3 tree-relevant LDA command lists."""

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


@dataclass(frozen=True)
class Stage3Command:
    suite: str
    purpose: str
    latent_partition_mode: str
    latent_length_profile: str
    analysis_partition_mode: str
    query_design: str
    tau: float
    lam: float
    seed: int
    cmd: str
    json_summary: str


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _json_csv_paths(base: Path) -> tuple[Path, Path]:
    return base.with_suffix(".json"), base.with_suffix(".csv")


def _run_cmd(
    *,
    python_bin: str,
    out_base: Path,
    doc_tokens: int,
    train_docs: int,
    test_docs: int,
    latent_leaf_tokens: int,
    latent_partition_mode: str,
    latent_length_profile: str,
    analysis_partition_mode: str,
    analysis_leaf_tokens: int,
    tau: float,
    lam: float,
    seed: int,
    query_design: str = "uniform",
    target_query_budget_per_doc: float = 0.0,
    heldout_doc_sample_rate: float = 0.5,
    doc_topic_concentration: float = 0.6,
    anchor_multiplier: float = 25.0,
    topic_concentration: float = 0.2,
) -> str:
    json_path, csv_path = _json_csv_paths(out_base)
    leaf_fraction = f"{latent_leaf_tokens}/{doc_tokens}"
    return (
        f"{python_bin} -u scripts/run_leaf_local_mixture_utility_simulation.py "
        f"--doc-tokens {int(doc_tokens)} "
        f"--train-docs {int(train_docs)} --test-docs {int(test_docs)} "
        f"--latent-leaf-tokens {int(latent_leaf_tokens)} "
        f"--latent-partition-mode {latent_partition_mode} "
        f"--latent-length-profile {latent_length_profile} "
        f"--analysis-partition-mode {analysis_partition_mode} "
        f"--analysis-leaf-tokens {int(analysis_leaf_tokens)} "
        f"--leaf-fraction {leaf_fraction} "
        f"--doc-topic-concentration {float(doc_topic_concentration)} "
        f"--local-mixture-concentration {float(tau)} "
        f"--lambda-multiplier {float(lam)} "
        f"--query-design {query_design} "
        f"--target-query-budget-per-doc {float(target_query_budget_per_doc)} "
        f"--heldout-doc-sample-rate {float(heldout_doc_sample_rate)} "
        f"--anchor-multiplier {float(anchor_multiplier)} "
        f"--topic-concentration {float(topic_concentration)} "
        f"--seed {int(seed)} "
        f"--json-summary {json_path} --csv-summary {csv_path}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Stage-3 tree-relevant LDA command lists.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--cmd-file", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--matrix-md", type=Path, required=True)
    p.add_argument("--python-bin", type=str, default="venv/bin/python")
    p.add_argument("--doc-tokens", type=int, default=384)
    p.add_argument("--train-docs", type=int, default=512)
    p.add_argument("--test-docs", type=int, default=512)
    p.add_argument("--latent-leaf-tokens", type=int, default=64)
    p.add_argument("--weighted-seeds", type=str, default="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23")
    p.add_argument("--mismatch-seeds", type=str, default="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23")
    p.add_argument("--ipw-seeds", type=str, default="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47")
    p.add_argument("--hardness-seeds", type=str, default="0 1 2 3 4 5 6 7 8 9 10 11")
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _parse_ints(text: str) -> List[int]:
    return [int(x) for x in str(text).replace(",", " ").split() if x.strip()]


def main() -> int:
    args = parse_args()
    doc_tokens = int(args.doc_tokens)
    train_docs = int(args.train_docs)
    test_docs = int(args.test_docs)
    nominal_llt = int(args.latent_leaf_tokens)
    commands: List[Stage3Command] = []

    def add(item: Stage3Command) -> None:
        json_path = Path(item.json_summary)
        csv_path = json_path.with_suffix(".csv")
        if bool(args.skip_existing) and json_path.exists() and csv_path.exists():
            return
        commands.append(item)

    weighted_purpose = "Weighted-length control: unequal latent section lengths, aligned analysis, weighted vs unweighted aggregation."
    for profile in ("equal", "bimodal", "long_tail"):
        for tau in (0.25, 1.0, 8.0):
            for lam in (0.0, 2.0):
                for seed in _parse_ints(args.weighted_seeds):
                    out_base = (
                        args.output_root
                        / "suite_a_weighted_length"
                        / f"profile_{profile}"
                        / f"tau_{tau:g}"
                        / f"lam_{lam:g}"
                        / f"seed_{seed}"
                    )
                    cmd = _run_cmd(
                        python_bin=str(args.python_bin),
                        out_base=out_base,
                        doc_tokens=doc_tokens,
                        train_docs=train_docs,
                        test_docs=test_docs,
                        latent_leaf_tokens=nominal_llt,
                        latent_partition_mode="variable",
                        latent_length_profile=profile,
                        analysis_partition_mode="aligned",
                        analysis_leaf_tokens=nominal_llt,
                        tau=float(tau),
                        lam=float(lam),
                        seed=int(seed),
                    )
                    add(
                        Stage3Command(
                            suite="suite_a_weighted_length",
                            purpose=weighted_purpose,
                            latent_partition_mode="variable",
                            latent_length_profile=profile,
                            analysis_partition_mode="aligned",
                            query_design="uniform",
                            tau=float(tau),
                            lam=float(lam),
                            seed=int(seed),
                            cmd=cmd,
                            json_summary=str(out_base.with_suffix(".json")),
                        )
                    )

    mismatch_purpose = "Partition mismatch oracle decomposition: aligned vs coarsened vs refined vs shifted vs random boundaries."
    for mode in ("aligned", "coarsen_2x", "refine_2x", "shift_half", "random_same_count"):
        for tau in (0.25, 1.0, 4.0, 8.0, 16.0, 64.0):
            for lam in (0.0, 0.5, 2.0):
                for seed in _parse_ints(args.mismatch_seeds):
                    out_base = (
                        args.output_root
                        / "suite_b_partition_mismatch"
                        / f"mode_{mode}"
                        / f"tau_{tau:g}"
                        / f"lam_{lam:g}"
                        / f"seed_{seed}"
                    )
                    cmd = _run_cmd(
                        python_bin=str(args.python_bin),
                        out_base=out_base,
                        doc_tokens=doc_tokens,
                        train_docs=train_docs,
                        test_docs=test_docs,
                        latent_leaf_tokens=nominal_llt,
                        latent_partition_mode="equal",
                        latent_length_profile="equal",
                        analysis_partition_mode=mode,
                        analysis_leaf_tokens=nominal_llt,
                        tau=float(tau),
                        lam=float(lam),
                        seed=int(seed),
                    )
                    add(
                        Stage3Command(
                            suite="suite_b_partition_mismatch",
                            purpose=mismatch_purpose,
                            latent_partition_mode="equal",
                            latent_length_profile="equal",
                            analysis_partition_mode=mode,
                            query_design="uniform",
                            tau=float(tau),
                            lam=float(lam),
                            seed=int(seed),
                            cmd=cmd,
                            json_summary=str(out_base.with_suffix(".json")),
                        )
                    )

    ipw_purpose = "Budgeted analysis-section supervision with adaptive querying and IPW-style training/evaluation."
    for mode in ("aligned", "shift_half"):
        for tau in (1.0, 8.0, 16.0):
            for lam in (0.0, 1.5, 3.0):
                for design in ("uniform", "proxy_priority", "proxy_adversarial"):
                    for budget in (1.0, 2.0):
                        for seed in _parse_ints(args.ipw_seeds):
                            out_base = (
                                args.output_root
                                / "suite_c_ipw_budgeted"
                                / f"mode_{mode}"
                                / f"design_{design}"
                                / f"budget_{budget:g}"
                                / f"tau_{tau:g}"
                                / f"lam_{lam:g}"
                                / f"seed_{seed}"
                            )
                            cmd = _run_cmd(
                                python_bin=str(args.python_bin),
                                out_base=out_base,
                                doc_tokens=doc_tokens,
                                train_docs=train_docs,
                                test_docs=test_docs,
                                latent_leaf_tokens=nominal_llt,
                                latent_partition_mode="equal",
                                latent_length_profile="equal",
                                analysis_partition_mode=mode,
                                analysis_leaf_tokens=nominal_llt,
                                tau=float(tau),
                                lam=float(lam),
                                seed=int(seed),
                                query_design=design,
                                target_query_budget_per_doc=float(budget),
                                heldout_doc_sample_rate=0.5,
                            )
                            add(
                                Stage3Command(
                                    suite="suite_c_ipw_budgeted",
                                    purpose=ipw_purpose,
                                    latent_partition_mode="equal",
                                    latent_length_profile="equal",
                                    analysis_partition_mode=mode,
                                    query_design=design,
                                    tau=float(tau),
                                    lam=float(lam),
                                    seed=int(seed),
                                    cmd=cmd,
                                    json_summary=str(out_base.with_suffix(".json")),
                                )
                            )

    hardness_purpose = "Hardness appendix: repeat boundary/winning cells under harder topic recovery."
    for anchor_multiplier in (25.0, 10.0):
        for topic_concentration in (0.2, 1.0):
            for mode in ("aligned", "shift_half"):
                for tau in (1.0, 8.0):
                    for lam in (1.5, 3.0):
                        for seed in _parse_ints(args.hardness_seeds):
                            out_base = (
                                args.output_root
                                / "suite_d_hardness"
                                / f"anchor_{anchor_multiplier:g}"
                                / f"topicconc_{topic_concentration:g}"
                                / f"mode_{mode}"
                                / f"tau_{tau:g}"
                                / f"lam_{lam:g}"
                                / f"seed_{seed}"
                            )
                            cmd = _run_cmd(
                                python_bin=str(args.python_bin),
                                out_base=out_base,
                                doc_tokens=doc_tokens,
                                train_docs=train_docs,
                                test_docs=test_docs,
                                latent_leaf_tokens=nominal_llt,
                                latent_partition_mode="equal",
                                latent_length_profile="equal",
                                analysis_partition_mode=mode,
                                analysis_leaf_tokens=nominal_llt,
                                tau=float(tau),
                                lam=float(lam),
                                seed=int(seed),
                                query_design="proxy_priority",
                                target_query_budget_per_doc=2.0,
                                heldout_doc_sample_rate=0.5,
                                anchor_multiplier=float(anchor_multiplier),
                                topic_concentration=float(topic_concentration),
                            )
                            add(
                                Stage3Command(
                                    suite="suite_d_hardness",
                                    purpose=hardness_purpose,
                                    latent_partition_mode="equal",
                                    latent_length_profile="equal",
                                    analysis_partition_mode=mode,
                                    query_design="proxy_priority",
                                    tau=float(tau),
                                    lam=float(lam),
                                    seed=int(seed),
                                    cmd=cmd,
                                    json_summary=str(out_base.with_suffix(".json")),
                                )
                            )

    _write_lines(args.cmd_file, (item.cmd for item in commands))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as handle:
        for item in commands:
            handle.write(json.dumps(asdict(item), sort_keys=True) + "\n")

    counts: dict[str, int] = {}
    for item in commands:
        counts[item.suite] = counts.get(item.suite, 0) + 1
    matrix_lines = [
        "# Tree-Relevant LDA Stage 3 Matrix",
        "",
        "This queue targets weighting, mismatch, and IPW-budgeted supervision inside the tree-relevant LDA ladder.",
        "",
        f"- Doc tokens: `{doc_tokens}`",
        f"- Nominal latent section tokens: `{nominal_llt}`",
        f"- Train docs per run: `{train_docs}`",
        f"- Test docs per run: `{test_docs}`",
        f"- Total commands: `{len(commands)}`",
        "",
        "## Suites",
        "",
        f"- `suite_a_weighted_length`: `{counts.get('suite_a_weighted_length', 0)}` commands. Variable latent lengths with aligned analysis.",
        f"- `suite_b_partition_mismatch`: `{counts.get('suite_b_partition_mismatch', 0)}` commands. Boundary mismatch decomposition.",
        f"- `suite_c_ipw_budgeted`: `{counts.get('suite_c_ipw_budgeted', 0)}` commands. Adaptive querying plus IPW-inspired training/evaluation.",
        f"- `suite_d_hardness`: `{counts.get('suite_d_hardness', 0)}` commands. Topic-recovery difficulty appendix.",
        "",
    ]
    args.matrix_md.parent.mkdir(parents=True, exist_ok=True)
    args.matrix_md.write_text("\n".join(matrix_lines), encoding="utf-8")

    print(f"wrote_cmds | {args.cmd_file} | n={len(commands)}")
    print(f"wrote_manifest | {args.manifest}")
    print(f"wrote_matrix_md | {args.matrix_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
