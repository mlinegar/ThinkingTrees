#!/usr/bin/env python3
"""Build tree-relevant LDA local-law companion command lists."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.contracts import (
    LAW_ID_LEAF_PRESERVATION,
    LAW_ID_MERGE_PRESERVATION,
    LAW_ID_ON_RANGE_IDEMPOTENCE,
    LAW_SET_ALL,
    LAW_SET_LEAF_AND_MERGE_PRESERVATION,
    LAW_SET_LEAF_PRESERVATION_ONLY,
    LAW_SET_MERGE_PRESERVATION_ONLY,
    LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY,
    assert_public_contract_clean,
    canonical_law_set_id,
)
from src.ctreepo.sim.composite_objective import resolve_root_local_objective_weights
from src.ctreepo.sim.manifest import RunSpec, write_manifest_jsonl

DEFAULT_LAW_WEIGHT = 1.0 / 3.0

_LAW_SET_ACTIVE_LAWS = {
    LAW_SET_ALL: (
        LAW_ID_LEAF_PRESERVATION,
        LAW_ID_ON_RANGE_IDEMPOTENCE,
        LAW_ID_MERGE_PRESERVATION,
    ),
    LAW_SET_LEAF_PRESERVATION_ONLY: (LAW_ID_LEAF_PRESERVATION,),
    LAW_SET_MERGE_PRESERVATION_ONLY: (LAW_ID_MERGE_PRESERVATION,),
    LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY: (LAW_ID_ON_RANGE_IDEMPOTENCE,),
    LAW_SET_LEAF_AND_MERGE_PRESERVATION: (
        LAW_ID_LEAF_PRESERVATION,
        LAW_ID_MERGE_PRESERVATION,
    ),
}


def _canonical_objective_fields(
    *,
    law_set_id: str,
    local_law_weight: float,
) -> dict[str, object]:
    law_set = canonical_law_set_id(str(law_set_id), allow_aliases=False)
    resolved = resolve_root_local_objective_weights(
        local_law_weight=float(local_law_weight),
        active_laws=_LAW_SET_ACTIVE_LAWS.get(law_set, _LAW_SET_ACTIVE_LAWS[LAW_SET_ALL]),
        objective_context="LDA command builder",
    )
    return {
        "problem_id": "leaf_local_mixture_utility",
        "method_id": "tree_relevant_lda_local_law",
        "law_set_id": law_set,
        "root_share": float(resolved.root_share),
        "local_law_weight": float(resolved.local_law_weight),
        "local_law_component_weights": {
            str(k): float(v) for k, v in resolved.local_law_shares.items()
        },
    }


@dataclass(frozen=True)
class LocalLawCommand:
    suite: str
    suite_role: str
    purpose: str
    analysis_partition_mode: str
    tau: float
    qweight: float
    seed: int
    cmd: str
    json_summary: str
    csv_summary: str
    artifact_dir: str
    extras: dict


SUITE_ROLE_BY_NAME = {
    "suite_a_exact_controls": "positive_controls",
    "suite_b_local_law_learnability": "support_scaling",
    "suite_c_mismatch_mediation": "relevance_mediation",
    "suite_d_ipw_sparse_labels": "failure_modes",
    "suite_e_hardness": "hardness",
}


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _json_csv_paths(base: Path) -> tuple[Path, Path]:
    return base.with_suffix(".json"), base.with_suffix(".csv")


def _apply_law_objective_path_labels(
    base: Path,
    *,
    law_set_id: str,
    local_law_weight: float,
) -> Path:
    if str(law_set_id).strip() and str(law_set_id).strip() != "all":
        base = base / f"lawset_{str(law_set_id).strip()}"
    if abs(float(local_law_weight) - 0.5) > 1e-12:
        base = base / f"llw_{str(float(local_law_weight)).replace('.', 'p')}"
    return base


def _run_cmd(
    *,
    python_bin: str,
    out_base: Path,
    doc_tokens: int,
    train_docs: int,
    val_docs: int,
    test_docs: int,
    latent_leaf_tokens: int,
    latent_partition_mode: str,
    latent_length_profile: str,
    analysis_partition_mode: str,
    analysis_leaf_tokens: int,
    tau: float,
    qweight: float,
    seed: int,
    local_law_mode: str,
    law_leaf_query_rate: float,
    law_internal_query_rate: float,
    law_leaf_query_design: str,
    law_internal_query_design: str,
    suite_role: str,
    anchor_multiplier: float = 25.0,
    topic_concentration: float = 0.2,
    law_set_id: str = "all",
    local_law_weight: float = 0.5,
) -> tuple[str, Path, Path, Path]:
    out_base = _apply_law_objective_path_labels(
        out_base,
        law_set_id=str(law_set_id).strip() or "all",
        local_law_weight=float(local_law_weight),
    )
    json_path, csv_path = _json_csv_paths(out_base)
    artifact_dir = out_base.parent / f"{out_base.name}_artifacts"
    leaf_fraction = f"{latent_leaf_tokens}/{doc_tokens}"
    cmd = (
        f"{python_bin} -u scripts/run_leaf_local_mixture_utility_simulation.py "
        f"--doc-tokens {int(doc_tokens)} "
        f"--train-docs {int(train_docs)} --val-docs {int(val_docs)} --test-docs {int(test_docs)} "
        f"--latent-leaf-tokens {int(latent_leaf_tokens)} "
        f"--latent-partition-mode {latent_partition_mode} "
        f"--latent-length-profile {latent_length_profile} "
        f"--analysis-partition-mode {analysis_partition_mode} "
        f"--analysis-leaf-tokens {int(analysis_leaf_tokens)} "
        f"--leaf-fraction {leaf_fraction} "
        f"--local-mixture-concentration {float(tau)} "
        f"--quadratic-utility-weight {float(qweight)} "
        f"--anchor-multiplier {float(anchor_multiplier)} "
        f"--topic-concentration {float(topic_concentration)} "
        f"--local-law-mode {local_law_mode} "
        f"--law-leaf-query-rate {float(law_leaf_query_rate)} "
        f"--law-internal-query-rate {float(law_internal_query_rate)} "
        f"--law-leaf-query-design {law_leaf_query_design} "
        f"--law-internal-query-design {law_internal_query_design} "
        f"--law-set-id {str(law_set_id).strip() or 'all'} "
        f"--local-law-weight {float(local_law_weight)} "
        f"--suite-role {suite_role} "
        f"--artifact-dir {artifact_dir} "
        f"--seed {int(seed)} "
        f"--json-summary {json_path} --csv-summary {csv_path}"
    )
    return cmd, json_path, csv_path, artifact_dir


def _run_spec_from_command(item: LocalLawCommand) -> RunSpec:
    objective_fields = _canonical_objective_fields(
        law_set_id=str(item.extras.get("law_set_id", "all")),
        local_law_weight=float(item.extras.get("local_law_weight", 0.5)),
    )
    config = {
        **objective_fields,
        "suite": str(item.suite),
        "suite_role": str(item.suite_role),
        "analysis_partition_mode": str(item.analysis_partition_mode),
        "tau": float(item.tau),
        "quadratic_utility_weight": float(item.qweight),
        "seed": int(item.seed),
        **dict(item.extras),
    }
    assert_public_contract_clean(config, surface="LDA command manifest config")
    return RunSpec.create(
        family="leaf_local_mixture_utility",
        config=config,
        outputs={
            "json_summary": str(item.json_summary),
            "csv_summary": str(item.csv_summary),
            "artifact_dir": str(item.artifact_dir),
        },
        command=str(item.cmd),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build tree-relevant LDA local-law companion command lists."
    )
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--cmd-file", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--matrix-md", type=Path, required=True)
    p.add_argument("--python-bin", type=str, default="venv/bin/python")
    p.add_argument("--doc-tokens", type=int, default=384)
    p.add_argument("--train-docs", type=int, default=512)
    p.add_argument("--val-docs", type=int, default=128)
    p.add_argument("--test-docs", type=int, default=512)
    p.add_argument("--latent-leaf-tokens", type=int, default=64)
    p.add_argument("--suite-a-seeds", type=str, default="0 1 2 3 4 5 6 7 8 9 10 11")
    p.add_argument(
        "--suite-b-seeds",
        type=str,
        default="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23",
    )
    p.add_argument(
        "--suite-c-seeds",
        type=str,
        default="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23",
    )
    p.add_argument(
        "--suite-d-seeds",
        type=str,
        default="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23",
    )
    p.add_argument("--suite-e-seeds", type=str, default="0 1 2 3 4 5 6 7 8 9 10 11")
    p.add_argument("--law-set-id", type=str, default="all")
    p.add_argument("--local-law-weight", type=float, default=0.5)
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _parse_ints(text: str) -> List[int]:
    return [int(x) for x in str(text).replace(",", " ").split() if x.strip()]


def main() -> int:
    args = parse_args()
    commands: List[LocalLawCommand] = []
    run_specs: List[RunSpec] = []
    doc_tokens = int(args.doc_tokens)
    nominal_llt = int(args.latent_leaf_tokens)

    def add(item: LocalLawCommand) -> None:
        json_path = Path(item.json_summary)
        csv_path = json_path.with_suffix(".csv")
        if bool(args.skip_existing) and json_path.exists() and csv_path.exists():
            return
        commands.append(item)
        run_specs.append(_run_spec_from_command(item))

    exact_purpose = "Exact controls for analysis-summary local laws under aligned and shifted analysis boundaries."
    for mode in ("aligned", "shift_half"):
        for tau in (1.0, 8.0, 16.0):
            for qweight in (0.0, 1.5, 3.0):
                for seed in _parse_ints(args.suite_a_seeds):
                    out_base = (
                        args.output_root
                        / "suite_a_exact_controls"
                        / f"mode_{mode}"
                        / f"tau_{tau:g}"
                        / f"qweight_{qweight:g}"
                        / f"seed_{seed}"
                    )
                    suite_name = "suite_a_exact_controls"
                    cmd, json_path, csv_path, artifact_dir = _run_cmd(
                        python_bin=str(args.python_bin),
                        out_base=out_base,
                        doc_tokens=doc_tokens,
                        train_docs=int(args.train_docs),
                        val_docs=int(args.val_docs),
                        test_docs=int(args.test_docs),
                        latent_leaf_tokens=nominal_llt,
                        latent_partition_mode="equal",
                        latent_length_profile="equal",
                        analysis_partition_mode=mode,
                        analysis_leaf_tokens=nominal_llt,
                        tau=float(tau),
                        qweight=float(qweight),
                        seed=int(seed),
                        local_law_mode="diagnostics",
                        law_leaf_query_rate=0.10,
                        law_internal_query_rate=0.10,
                        law_leaf_query_design="uniform",
                        law_internal_query_design="uniform",
                        suite_role=SUITE_ROLE_BY_NAME[suite_name],
                        law_set_id=str(args.law_set_id).strip() or "all",
                        local_law_weight=float(args.local_law_weight),
                    )
                    add(
                        LocalLawCommand(
                            suite=suite_name,
                            suite_role=SUITE_ROLE_BY_NAME[suite_name],
                            purpose=exact_purpose,
                            analysis_partition_mode=mode,
                            tau=float(tau),
                            qweight=float(qweight),
                            seed=int(seed),
                            cmd=cmd,
                            json_summary=str(json_path),
                            csv_summary=str(csv_path),
                            artifact_dir=str(artifact_dir),
                            extras={
                                "local_law_mode": "diagnostics",
                                "law_set_id": str(args.law_set_id).strip() or "all",
                                "local_law_weight": float(args.local_law_weight),
                            },
                        )
                    )

    learnability_purpose = (
        "Local-law learnability with more training documents and more leaf/internal law labels."
    )
    for tau in (1.0, 8.0, 16.0):
        for qweight in (0.0, 1.5, 3.0):
            for train_docs in (64, 128, 256, 512, 1024):
                for leaf_rate in (0.05, 0.10, 0.20):
                    for internal_rate in (0.05, 0.10, 0.20):
                        for seed in _parse_ints(args.suite_b_seeds):
                            out_base = (
                                args.output_root
                                / "suite_b_local_law_learnability"
                                / f"train_{train_docs}"
                                / f"leafrate_{leaf_rate:g}"
                                / f"internalrate_{internal_rate:g}"
                                / f"tau_{tau:g}"
                                / f"qweight_{qweight:g}"
                                / f"seed_{seed}"
                            )
                            suite_name = "suite_b_local_law_learnability"
                            cmd, json_path, csv_path, artifact_dir = _run_cmd(
                                python_bin=str(args.python_bin),
                                out_base=out_base,
                                doc_tokens=doc_tokens,
                                train_docs=int(train_docs),
                                val_docs=int(args.val_docs),
                                test_docs=int(args.test_docs),
                                latent_leaf_tokens=nominal_llt,
                                latent_partition_mode="equal",
                                latent_length_profile="equal",
                                analysis_partition_mode="aligned",
                                analysis_leaf_tokens=nominal_llt,
                                tau=float(tau),
                                qweight=float(qweight),
                                seed=int(seed),
                                local_law_mode="diagnostics_and_learned",
                                law_leaf_query_rate=float(leaf_rate),
                                law_internal_query_rate=float(internal_rate),
                                law_leaf_query_design="uniform",
                                law_internal_query_design="uniform",
                                suite_role=SUITE_ROLE_BY_NAME[suite_name],
                                law_set_id=str(args.law_set_id).strip() or "all",
                                local_law_weight=float(args.local_law_weight),
                            )
                            add(
                                LocalLawCommand(
                                    suite=suite_name,
                                    suite_role=SUITE_ROLE_BY_NAME[suite_name],
                                    purpose=learnability_purpose,
                                    analysis_partition_mode="aligned",
                                    tau=float(tau),
                                    qweight=float(qweight),
                                    seed=int(seed),
                                    cmd=cmd,
                                    json_summary=str(json_path),
                                    csv_summary=str(csv_path),
                                    artifact_dir=str(artifact_dir),
                                    extras={
                                        "train_docs": int(train_docs),
                                        "val_docs": int(args.val_docs),
                                        "law_leaf_query_rate": float(leaf_rate),
                                        "law_internal_query_rate": float(internal_rate),
                                        "law_set_id": str(args.law_set_id).strip() or "all",
                                        "local_law_weight": float(args.local_law_weight),
                                    },
                                )
                            )

    mismatch_purpose = "Boundary mismatch mediation: local-law error by analysis mode and how it maps into downstream Delta."
    for mode in ("aligned", "coarsen_2x", "shift_half", "random_same_count"):
        for tau in (1.0, 8.0, 16.0):
            for qweight in (0.0, 1.5, 3.0):
                for seed in _parse_ints(args.suite_c_seeds):
                    out_base = (
                        args.output_root
                        / "suite_c_mismatch_mediation"
                        / f"mode_{mode}"
                        / f"tau_{tau:g}"
                        / f"qweight_{qweight:g}"
                        / f"seed_{seed}"
                    )
                    suite_name = "suite_c_mismatch_mediation"
                    cmd, json_path, csv_path, artifact_dir = _run_cmd(
                        python_bin=str(args.python_bin),
                        out_base=out_base,
                        doc_tokens=doc_tokens,
                        train_docs=512,
                        val_docs=int(args.val_docs),
                        test_docs=int(args.test_docs),
                        latent_leaf_tokens=nominal_llt,
                        latent_partition_mode="equal",
                        latent_length_profile="equal",
                        analysis_partition_mode=mode,
                        analysis_leaf_tokens=nominal_llt,
                        tau=float(tau),
                        qweight=float(qweight),
                        seed=int(seed),
                        local_law_mode="diagnostics_and_learned",
                        law_leaf_query_rate=0.10,
                        law_internal_query_rate=0.10,
                        law_leaf_query_design="uniform",
                        law_internal_query_design="uniform",
                        suite_role=SUITE_ROLE_BY_NAME[suite_name],
                        law_set_id=str(args.law_set_id).strip() or "all",
                        local_law_weight=float(args.local_law_weight),
                    )
                    add(
                        LocalLawCommand(
                            suite=suite_name,
                            suite_role=SUITE_ROLE_BY_NAME[suite_name],
                            purpose=mismatch_purpose,
                            analysis_partition_mode=mode,
                            tau=float(tau),
                            qweight=float(qweight),
                            seed=int(seed),
                            cmd=cmd,
                            json_summary=str(json_path),
                            csv_summary=str(csv_path),
                            artifact_dir=str(artifact_dir),
                            extras={
                                "val_docs": int(args.val_docs),
                                "law_leaf_query_rate": 0.10,
                                "law_internal_query_rate": 0.10,
                                "law_set_id": str(args.law_set_id).strip() or "all",
                                "local_law_weight": float(args.local_law_weight),
                            },
                        )
                    )

    ipw_purpose = (
        "Adaptive local-law labeling with naive vs IPW vs stabilized-IPW summary calibration."
    )
    for mode in ("aligned", "shift_half"):
        for tau in (1.0, 8.0, 16.0):
            for qweight in (0.0, 1.5, 3.0):
                for leaf_design in ("uniform", "proxy_priority", "proxy_adversarial"):
                    for internal_design in ("uniform", "risk"):
                        for leaf_rate in (0.05, 0.10):
                            for internal_rate in (0.05, 0.10):
                                for seed in _parse_ints(args.suite_d_seeds):
                                    out_base = (
                                        args.output_root
                                        / "suite_d_ipw_sparse_labels"
                                        / f"mode_{mode}"
                                        / f"leafdesign_{leaf_design}"
                                        / f"internaldesign_{internal_design}"
                                        / f"leafrate_{leaf_rate:g}"
                                        / f"internalrate_{internal_rate:g}"
                                        / f"tau_{tau:g}"
                                        / f"qweight_{qweight:g}"
                                        / f"seed_{seed}"
                                    )
                                    suite_name = "suite_d_ipw_sparse_labels"
                                    cmd, json_path, csv_path, artifact_dir = _run_cmd(
                                        python_bin=str(args.python_bin),
                                        out_base=out_base,
                                        doc_tokens=doc_tokens,
                                        train_docs=512,
                                        val_docs=int(args.val_docs),
                                        test_docs=512,
                                        latent_leaf_tokens=nominal_llt,
                                        latent_partition_mode="equal",
                                        latent_length_profile="equal",
                                        analysis_partition_mode=mode,
                                        analysis_leaf_tokens=nominal_llt,
                                        tau=float(tau),
                                        qweight=float(qweight),
                                        seed=int(seed),
                                        local_law_mode="diagnostics_and_learned",
                                        law_leaf_query_rate=float(leaf_rate),
                                        law_internal_query_rate=float(internal_rate),
                                        law_leaf_query_design=leaf_design,
                                        law_internal_query_design=internal_design,
                                        suite_role=SUITE_ROLE_BY_NAME[suite_name],
                                        law_set_id=str(args.law_set_id).strip() or "all",
                                        local_law_weight=float(args.local_law_weight),
                                    )
                                    add(
                                        LocalLawCommand(
                                            suite=suite_name,
                                            suite_role=SUITE_ROLE_BY_NAME[suite_name],
                                            purpose=ipw_purpose,
                                            analysis_partition_mode=mode,
                                            tau=float(tau),
                                            qweight=float(qweight),
                                            seed=int(seed),
                                            cmd=cmd,
                                            json_summary=str(json_path),
                                            csv_summary=str(csv_path),
                                            artifact_dir=str(artifact_dir),
                                            extras={
                                                "val_docs": int(args.val_docs),
                                                "law_leaf_query_design": leaf_design,
                                                "law_internal_query_design": internal_design,
                                                "law_leaf_query_rate": float(leaf_rate),
                                                "law_internal_query_rate": float(internal_rate),
                                                "law_set_id": str(args.law_set_id).strip() or "all",
                                                "local_law_weight": float(args.local_law_weight),
                                            },
                                        )
                                    )

    hardness_purpose = "Harder topic recovery appendix slice for the local-law companion."
    for mode in ("aligned", "shift_half"):
        for tau in (8.0, 16.0):
            for qweight in (1.5, 3.0):
                for anchor_multiplier in (25.0, 10.0):
                    for topic_concentration in (0.2, 1.0):
                        for seed in _parse_ints(args.suite_e_seeds):
                            out_base = (
                                args.output_root
                                / "suite_e_hardness"
                                / f"mode_{mode}"
                                / f"anchor_{anchor_multiplier:g}"
                                / f"topicconc_{topic_concentration:g}"
                                / f"tau_{tau:g}"
                                / f"qweight_{qweight:g}"
                                / f"seed_{seed}"
                            )
                            suite_name = "suite_e_hardness"
                            cmd, json_path, csv_path, artifact_dir = _run_cmd(
                                python_bin=str(args.python_bin),
                                out_base=out_base,
                                doc_tokens=doc_tokens,
                                train_docs=512,
                                val_docs=int(args.val_docs),
                                test_docs=int(args.test_docs),
                                latent_leaf_tokens=nominal_llt,
                                latent_partition_mode="equal",
                                latent_length_profile="equal",
                                analysis_partition_mode=mode,
                                analysis_leaf_tokens=nominal_llt,
                                tau=float(tau),
                                qweight=float(qweight),
                                seed=int(seed),
                                local_law_mode="diagnostics_and_learned",
                                law_leaf_query_rate=0.10,
                                law_internal_query_rate=0.10,
                                law_leaf_query_design="uniform",
                                law_internal_query_design="uniform",
                                anchor_multiplier=float(anchor_multiplier),
                                topic_concentration=float(topic_concentration),
                                suite_role=SUITE_ROLE_BY_NAME[suite_name],
                                law_set_id=str(args.law_set_id).strip() or "all",
                                local_law_weight=float(args.local_law_weight),
                            )
                            add(
                                LocalLawCommand(
                                    suite=suite_name,
                                    suite_role=SUITE_ROLE_BY_NAME[suite_name],
                                    purpose=hardness_purpose,
                                    analysis_partition_mode=mode,
                                    tau=float(tau),
                                    qweight=float(qweight),
                                    seed=int(seed),
                                    cmd=cmd,
                                    json_summary=str(json_path),
                                    csv_summary=str(csv_path),
                                    artifact_dir=str(artifact_dir),
                                    extras={
                                        "val_docs": int(args.val_docs),
                                        "anchor_multiplier": float(anchor_multiplier),
                                        "topic_concentration": float(topic_concentration),
                                        "law_set_id": str(args.law_set_id).strip() or "all",
                                        "local_law_weight": float(args.local_law_weight),
                                    },
                                )
                            )

    _write_lines(args.cmd_file, [item.cmd for item in commands])
    write_manifest_jsonl(args.manifest, run_specs)

    suite_counts: dict[str, int] = {}
    for item in commands:
        suite_counts[item.suite] = suite_counts.get(item.suite, 0) + 1
    matrix_lines = [
        "# Tree-Relevant LDA Local-Law Companion Matrix",
        "",
        "| Suite | Commands | Purpose |",
        "| --- | ---: | --- |",
    ]
    for suite, count in sorted(suite_counts.items()):
        purpose = next(item.purpose for item in commands if item.suite == suite)
        matrix_lines.append(f"| `{suite}` | {count} | {purpose} |")
    matrix_lines.append("")
    matrix_lines.append(f"Total commands: **{len(commands)}**")
    _write_lines(args.matrix_md, matrix_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
