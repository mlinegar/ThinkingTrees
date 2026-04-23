#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill some or all missing scalar labels in a SupervisionDataset by calling a ScoringOracle. "
            "Optionally subsample which rows get labeled and record label propensities for IPW."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Input SupervisionDataset JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: overwrite --input).",
    )
    parser.add_argument(
        "--oracle-import-path",
        type=str,
        default="src.tree.auditor:SimpleScorer",
        help="Python import path to a ScoringOracle factory/value.",
    )
    parser.add_argument(
        "--oracle-kwargs",
        type=str,
        default="{}",
        help="JSON kwargs passed to the oracle factory (if callable).",
    )
    parser.add_argument(
        "--rubric-fallback",
        type=str,
        default="",
        help="Fallback rubric when rows have an empty rubric.",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=None,
        help="If set, label at most this many unlabeled rows (uniform without replacement).",
    )
    parser.add_argument(
        "--label-probability",
        type=float,
        default=None,
        help="If set, Bernoulli-label each unlabeled row with this probability p in (0,1].",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed used for subsampling unlabeled rows (if applicable).",
    )
    parser.add_argument(
        "--policy-name",
        type=str,
        default="treepo_supervision_labeler_v1",
        help="Policy name recorded into SamplingMetadata.metadata for labeled rows.",
    )
    return parser.parse_args()


def _import_from_path(import_path: str) -> Any:
    import importlib

    text = str(import_path or "").strip()
    if not text:
        raise ValueError("Empty import_path")
    module_path: str
    attr_path: str
    if ":" in text:
        module_path, attr_path = text.split(":", 1)
    else:
        parts = text.split(".")
        if len(parts) < 2:
            raise ValueError(
                "Import path must be 'module:attr' or 'module.attr', "
                f"received {import_path!r}."
            )
        module_path, attr_path = ".".join(parts[:-1]), parts[-1]
    module = importlib.import_module(module_path)
    value: Any = module
    for part in attr_path.split("."):
        value = getattr(value, part)
    return value


def main() -> int:
    args = _parse_args()

    from src.training.supervision.types import SupervisionDataset
    from src.tree.treepo_supervision import label_supervision_dataset

    try:
        oracle_kwargs = json.loads(str(args.oracle_kwargs or "{}"))
        if not isinstance(oracle_kwargs, dict):
            raise ValueError("--oracle-kwargs must be a JSON object.")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --oracle-kwargs: {args.oracle_kwargs!r}") from exc

    oracle_target = _import_from_path(str(args.oracle_import_path))
    oracle = oracle_target(**dict(oracle_kwargs)) if callable(oracle_target) else oracle_target

    dataset = SupervisionDataset.load(args.input)
    label_supervision_dataset(
        dataset,
        oracle=oracle,
        rubric_fallback=str(args.rubric_fallback or ""),
        max_labels=args.max_labels,
        label_probability=args.label_probability,
        random_seed=args.random_seed,
        policy_name=str(args.policy_name or "treepo_supervision_labeler_v1"),
    )

    output = args.output if args.output is not None else args.input
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(output)
    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
