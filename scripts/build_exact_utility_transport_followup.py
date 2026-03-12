#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FollowupCommand:
    kind: str
    cmd: str
    json_summary: str


def _cmd(
    *,
    runner: str,
    json_summary: Path,
    oracle_profile: str,
    objective_family: str,
    structural_arm: str,
    train_docs: int,
    test_docs: int,
    seed: int,
    n_epochs: int,
    batch_size: int,
    hidden_dim: int,
    fixed_leaf_tokens: int = 1,
    n_binary_leaves: int | None = None,
    n_leaves: int | None = None,
    leaf_label_rate: float = 0.0,
    internal_label_rate: float = 0.0,
    root_query_rate: float = 0.0,
    pairwise_prefs_per_doc: int = 0,
    group_pref_groups_per_doc: int = 0,
    group_size: int = 4,
    ppo_rollouts_per_doc: int = 0,
) -> str:
    parts = [
        "source venv/bin/activate &&",
        "python",
        runner,
        "--oracle-profile",
        str(oracle_profile),
        "--objective-family",
        str(objective_family),
        "--structural-arm",
        str(structural_arm),
        "--train-docs",
        str(int(train_docs)),
        "--test-docs",
        str(int(test_docs)),
        "--seed",
        str(int(seed)),
        "--n-epochs",
        str(int(n_epochs)),
        "--batch-size",
        str(int(batch_size)),
        "--hidden-dim",
        str(int(hidden_dim)),
        "--fixed-leaf-tokens",
        str(int(fixed_leaf_tokens)),
        "--leaf-label-rate",
        str(float(leaf_label_rate)),
        "--internal-label-rate",
        str(float(internal_label_rate)),
        "--root-query-rate",
        str(float(root_query_rate)),
        "--pairwise-prefs-per-doc",
        str(int(pairwise_prefs_per_doc)),
        "--group-pref-groups-per-doc",
        str(int(group_pref_groups_per_doc)),
        "--group-size",
        str(int(group_size)),
        "--ppo-rollouts-per-doc",
        str(int(ppo_rollouts_per_doc)),
    ]
    if n_binary_leaves is not None:
        parts.extend(["--n-binary-leaves", str(int(n_binary_leaves))])
    if n_leaves is not None:
        parts.extend(["--n-leaves", str(int(n_leaves))])
    parts.extend(["--json-summary", str(json_summary)])
    return " ".join(parts)


def _build_boundary_reruns(root: Path, overnight_root: Path) -> List[FollowupCommand]:
    overnight_logs = overnight_root / "_launcher_logs" / "cpu"
    cmds: List[FollowupCommand] = []
    for path in sorted(overnight_logs.glob("run_*.log")):
        text = path.read_text(encoding="utf-8")
        if "RuntimeError: mat1 and mat2 shapes cannot be multiplied" not in text:
            continue
        first = text.splitlines()[0].strip()
        if not first.startswith("cmd="):
            continue
        cmd = first[len("cmd=") :]
        json_summary = ""
        if "--json-summary " in cmd:
            json_summary = cmd.split("--json-summary ", 1)[1].strip().split()[0]
        cmds.append(FollowupCommand(kind="boundary_topic_bugfix_rerun", cmd=cmd, json_summary=json_summary))
    return cmds


def _build_nonseparable_ppo_focus(root: Path, seeds: Iterable[int]) -> List[FollowupCommand]:
    cmds: List[FollowupCommand] = []
    runner = "scripts/run_nonseparable_treepo_preference.py"
    objective_bundles = [
        ("ppo", dict(leaf_label_rate=0.0, internal_label_rate=0.0, root_query_rate=0.0)),
        ("hybrid_supervised_plus_ppo", dict(leaf_label_rate=1.0, internal_label_rate=1.0, root_query_rate=0.0)),
    ]
    arms = ["tree_neural_supported", "flat_equal_info", "tree_undersupported", "one_leaf_control"]
    for objective_family, support in objective_bundles:
        for train_docs in (2048, 4096, 8192):
            for n_binary_leaves in (4, 8):
                for ppo_rollouts in (16, 64, 128, 256):
                    for hidden_dim in (64, 128):
                        for seed in seeds:
                            for arm in arms:
                                out = (
                                    root
                                    / "nonseparable"
                                    / "dgp2_boundary_interaction"
                                    / "ppo_focus"
                                    / objective_family
                                    / arm
                                    / f"train_{train_docs}__nleaves_{n_binary_leaves}__ppo_{ppo_rollouts}__hd_{hidden_dim}"
                                    / f"seed_{seed}.json"
                                )
                                cmd = _cmd(
                                    runner=runner,
                                    json_summary=out,
                                    oracle_profile="dgp2_boundary_interaction",
                                    objective_family=objective_family,
                                    structural_arm=arm,
                                    train_docs=train_docs,
                                    test_docs=512,
                                    seed=seed,
                                    n_epochs=40,
                                    batch_size=64,
                                    hidden_dim=hidden_dim,
                                    fixed_leaf_tokens=1,
                                    n_binary_leaves=n_binary_leaves,
                                    leaf_label_rate=float(support["leaf_label_rate"]),
                                    internal_label_rate=float(support["internal_label_rate"]),
                                    root_query_rate=float(support["root_query_rate"]),
                                    pairwise_prefs_per_doc=0,
                                    group_pref_groups_per_doc=0,
                                    group_size=4,
                                    ppo_rollouts_per_doc=ppo_rollouts,
                                )
                                cmds.append(FollowupCommand(kind="nonseparable_ppo_focus", cmd=cmd, json_summary=str(out)))
    for objective_family, support in objective_bundles:
        for train_docs in (2048, 8192):
            for n_binary_leaves in (4, 8):
                for seed in seeds:
                    for arm in ("oracle_exact", "tree_exact_supported"):
                        out = (
                            root
                            / "nonseparable"
                            / "dgp2_boundary_interaction"
                            / "ppo_focus"
                            / objective_family
                            / arm
                            / f"train_{train_docs}__nleaves_{n_binary_leaves}"
                            / f"seed_{seed}.json"
                        )
                        cmd = _cmd(
                            runner=runner,
                            json_summary=out,
                            oracle_profile="dgp2_boundary_interaction",
                            objective_family=objective_family,
                            structural_arm=arm,
                            train_docs=train_docs,
                            test_docs=512,
                            seed=seed,
                            n_epochs=1,
                            batch_size=64,
                            hidden_dim=64,
                            fixed_leaf_tokens=1,
                            n_binary_leaves=n_binary_leaves,
                            leaf_label_rate=float(support["leaf_label_rate"]),
                            internal_label_rate=float(support["internal_label_rate"]),
                            root_query_rate=float(support["root_query_rate"]),
                            pairwise_prefs_per_doc=0,
                            group_pref_groups_per_doc=0,
                            group_size=4,
                            ppo_rollouts_per_doc=256,
                        )
                        cmds.append(FollowupCommand(kind="nonseparable_ppo_focus", cmd=cmd, json_summary=str(out)))
    return cmds


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build focused exact-utility follow-up simulations.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--overnight-root", type=Path, required=True)
    p.add_argument("--cmd-file", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--matrix-md", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    reruns = _build_boundary_reruns(args.output_root, args.overnight_root)
    ppo_focus = _build_nonseparable_ppo_focus(args.output_root, args.seeds)
    cmds = reruns + ppo_focus
    _write_lines(args.cmd_file, (item.cmd for item in cmds))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as handle:
        for item in cmds:
            handle.write(json.dumps(asdict(item), sort_keys=True) + "\n")
    args.matrix_md.parent.mkdir(parents=True, exist_ok=True)
    args.matrix_md.write_text(
        "\n".join(
            [
                "# Exact Utility Follow-up Matrix",
                "",
                f"- Boundary-topic bugfix reruns: `{len(reruns)}`",
                f"- Nonseparable PPO focus jobs: `{len(ppo_focus)}`",
                f"- Total jobs: `{len(cmds)}`",
                "",
                "## Purpose",
                "",
                "- Replay boundary-topic rows that failed before the undersupported observation-dimension fix.",
                "- Expand the exact nonseparable PPO lane where the tree-aware learner still lags the flat baseline.",
                "- Test whether more train docs, more PPO rollouts, and more hidden capacity close that gap.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote_cmds | {args.cmd_file} | n={len(cmds)}")
    print(f"wrote_manifest | {args.manifest}")
    print(f"wrote_matrix_md | {args.matrix_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
