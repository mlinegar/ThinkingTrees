#!/usr/bin/env python3
"""Launch the authoritative depth-discount sweep via the tradeoff pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "run_markov_optimization_tradeoff_pipeline.py"

DEFAULT_SCOPES = ("recoverable_v4", "r12_seg10to12")
DEFAULT_TRAIN_DOCS = "10240"
DEFAULT_SEEDS = "0,1"
DEFAULT_PACKAGES = "full100,r100_mass_local_eq_15p0"
DEFAULT_LEAF_TOKENS = "32,16,8"
DEFAULT_GAMMAS = "1.0,0.9,0.75"
DEFAULT_RECOVERABLE_BENCHMARK = "recoverable_v4"
DEFAULT_STRUCTURAL_GRID = "structural_core_v1"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Thin wrapper over scripts/run_markov_optimization_tradeoff_pipeline.py "
            "for the authoritative depth-discount supervision-recovery sweep."
        )
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--preset", type=str, default="standard")
    parser.add_argument(
        "--phases",
        type=str,
        default="supervision_recovery,report",
    )
    parser.add_argument(
        "--scopes",
        type=str,
        default=",".join(DEFAULT_SCOPES),
        help=(
            "Comma-separated supervision-recovery scopes. "
            "Use recoverable plus at most one structural cell; benchmark identity "
            "is controlled separately via --recoverable-benchmark and "
            "--structural-grid."
        ),
    )
    parser.add_argument(
        "--recoverable-benchmark",
        type=str,
        default=DEFAULT_RECOVERABLE_BENCHMARK,
        help="Recoverable benchmark name passed through to the pipeline.",
    )
    parser.add_argument(
        "--structural-grid",
        type=str,
        default=DEFAULT_STRUCTURAL_GRID,
        help="Structural grid name passed through to the pipeline.",
    )
    parser.add_argument(
        "--train-docs",
        type=str,
        default=DEFAULT_TRAIN_DOCS,
        help="Comma-separated supervision-recovery train-doc counts.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=DEFAULT_SEEDS,
        help="Comma-separated supervision-recovery seeds.",
    )
    parser.add_argument(
        "--packages",
        type=str,
        default=DEFAULT_PACKAGES,
        help="Comma-separated supervision-recovery package list.",
    )
    parser.add_argument(
        "--leaf-token-ladder",
        type=str,
        default=DEFAULT_LEAF_TOKENS,
        help="Comma-separated leaf-token ladder.",
    )
    parser.add_argument(
        "--gammas",
        type=str,
        default=DEFAULT_GAMMAS,
        help="Comma-separated depth-discount gamma values.",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--no-reuse-existing", action="store_true")
    return parser.parse_known_args()


def main() -> int:
    args, passthrough = _parse_args()
    raw_scopes = [
        str(item).strip()
        for item in str(args.scopes).replace(",", " ").split()
        if str(item).strip()
    ]
    structural_scopes = [scope for scope in raw_scopes if scope != "recoverable_v4"]
    if not raw_scopes:
        raise SystemExit("at least one scope is required")
    if len(structural_scopes) > 1:
        raise SystemExit(
            "the wrapper supports at most one structural scope; "
            f"got {structural_scopes}"
        )
    structural_cell = structural_scopes[0] if structural_scopes else "r12_seg10to12"

    cmd = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--preset",
        str(args.preset),
        "--phases",
        str(args.phases),
        "--supervision-recovery-train-docs",
        str(args.train_docs),
        "--supervision-recovery-seeds",
        str(args.seeds),
        "--supervision-recovery-packages",
        str(args.packages),
        "--supervision-recovery-leaf-token-ladder",
        str(args.leaf_token_ladder),
        "--supervision-recovery-depth-discount-gammas",
        str(args.gammas),
        "--supervision-recovery-recoverable-benchmark",
        str(args.recoverable_benchmark),
        "--supervision-recovery-structural-grid",
        str(args.structural_grid),
        "--supervision-recovery-structural-cell",
        str(structural_cell),
    ]
    if args.config is not None:
        cmd.extend(["--config", str(args.config)])
    if args.output_root is not None:
        cmd.extend(["--output-root", str(args.output_root)])
    if bool(args.plan_only):
        cmd.append("--plan-only")
    if bool(args.detach):
        cmd.append("--detach")
    if bool(args.no_reuse_existing):
        cmd.append("--no-reuse-existing")
    cmd.extend(list(passthrough))
    return int(subprocess.call(cmd, cwd=str(REPO_ROOT)))


if __name__ == "__main__":
    raise SystemExit(main())
