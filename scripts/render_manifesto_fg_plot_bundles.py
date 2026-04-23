#!/usr/bin/env python3
"""Render reproducible manifesto f/g plot bundles.

This script codifies the current publication-facing bundle recipes so they can
be re-rendered with one command instead of reconstructing long plotter
invocations by hand.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "manifesto_fg_alternating"
PLOTTER = REPO_ROOT / "scripts" / "plot_manifesto_fg_ladder_grid.py"

ECONOMIC_EXTERNAL_PEARSON_MIN = 0.75
ECONOMIC_EXTERNAL_PEARSON_MAX = 0.88


@dataclass(frozen=True)
class BundleSpec:
    name: str
    title: str
    subtitle: str
    output_dir: Path
    input_roots: tuple[Path | str, ...]
    stage_labels: tuple[str, ...] = ()


# Plot titles and subtitles stay short on purpose; the reference-line
# interpretation (expert-expert ceiling at r=0.880, internal-external gap floor at 0),
# apples-to-apples baselines, and inherited-cell caveats live in the
# LaTeX figure captions in paper/ctreepo/appendix/H_benoit_replication.tex.
COMMON_SUBTITLE = (
    r"Gemma-4-31B-NVFP4, DSPy-medium; $n = 48$ held-out manifestos per cell."
)


def _bundle_specs(raw_run_root: Path) -> dict[str, BundleSpec]:
    return {
        "plain_benoit": BundleSpec(
            name="plain_benoit",
            title=(
                r"Alternating $f/g$ ladder, economic dimension "
                r"-- Benoit init (no anchor row)"
            ),
            subtitle=COMMON_SUBTITLE,
            output_dir=OUTPUT_ROOT / "benoit_grid_plots",
            input_roots=(
                OUTPUT_ROOT / "economic_benoit_moreleaves_dspy_medium_20260422_192229",
                OUTPUT_ROOT / "economic_benoit_largeleaves_dspy_medium_20260423_001200",
                OUTPUT_ROOT / "economic_benoit_largeleaves40k_dspy_medium_20260423",
            ),
        ),
        "benoit_init": BundleSpec(
            name="benoit_init",
            title=(
                r"Alternating $f/g$ ladder, economic dimension "
                r"-- Benoit init ($g^0 = g^{\mathrm{Benoit}}$)"
            ),
            subtitle=COMMON_SUBTITLE,
            output_dir=OUTPUT_ROOT / "benoit_grid_plots_benoit_init",
            input_roots=(
                OUTPUT_ROOT / "economic_benoit_f1g_benoit_anchor_20260423",
                OUTPUT_ROOT / "economic_benoit_moreleaves_dspy_medium_20260422_192229",
                OUTPUT_ROOT / "economic_benoit_largeleaves_dspy_medium_20260423_001200",
                OUTPUT_ROOT / "economic_benoit_largeleaves40k_dspy_medium_20260423",
            ),
            stage_labels=(
                "f1g_benoit=f^1g^0",
                "fg=f^1g^1",
                "fgf=f^2g^1",
                "fgfg=f^2g^2",
                "fgfgf=f^3g^2",
                "fgfgfg=f^3g^3",
            ),
        ),
        "raw_init": BundleSpec(
            name="raw_init",
            title=(
                r"Alternating $f/g$ ladder, economic dimension "
                r"-- raw init ($g^0$ = own-Gemma baseline)"
            ),
            subtitle=COMMON_SUBTITLE,
            output_dir=OUTPUT_ROOT / "benoit_grid_plots_raw_init",
            input_roots=(
                OUTPUT_ROOT / "economic_raw_f0g0_anchor_20260423",
                raw_run_root,
            ),
            stage_labels=(
                "f0g0=f^0g^0",
                "f1g0=f^1g^0",
                "f1g1=f^1g^1",
                "f2g1=f^2g^1",
                "f2g2=f^2g^2",
                "f3g2=f^3g^2",
                "f3g3=f^3g^3",
            ),
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render reproducible manifesto f/g plot bundle presets."
    )
    parser.add_argument(
        "--bundle",
        action="append",
        choices=("plain_benoit", "benoit_init", "raw_init"),
        default=[],
        help="Bundle preset to render. May be repeated.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render all bundle presets.",
    )
    parser.add_argument(
        "--raw-run-root",
        type=Path,
        default=OUTPUT_ROOT / "economic_benoit_g0init_f3g3_dspy_20260423_172036",
        help="Run root used as the dynamic input for the raw-init bundle.",
    )
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--family", default="dspy")
    return parser.parse_args()


def _render_bundle(spec: BundleSpec, *, eval_split: str, family: str) -> None:
    cmd = [
        sys.executable,
        str(PLOTTER),
        "--output-dir",
        str(spec.output_dir),
        "--eval-split",
        str(eval_split),
        "--family",
        str(family),
        "--figure-title",
        str(spec.title),
        "--figure-subtitle",
        str(spec.subtitle),
        "--external-pearson-min",
        str(ECONOMIC_EXTERNAL_PEARSON_MIN),
        "--external-pearson-max",
        str(ECONOMIC_EXTERNAL_PEARSON_MAX),
    ]
    for root in spec.input_roots:
        cmd.extend(["--input-root", str(root)])
    for label in spec.stage_labels:
        cmd.extend(["--stage-label", label])
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> int:
    args = _parse_args()
    bundles = list(args.bundle)
    if args.all:
        bundles = ["plain_benoit", "benoit_init", "raw_init"]
    if not bundles:
        raise SystemExit("pass --bundle ... or --all")

    specs = _bundle_specs(args.raw_run_root)
    for bundle in bundles:
        _render_bundle(specs[bundle], eval_split=str(args.eval_split), family=str(args.family))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
