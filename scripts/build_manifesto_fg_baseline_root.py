#!/usr/bin/env python3
"""Build a synthetic manifesto f/g ladder root from a baseline report.

This is useful when a baseline such as ``f^0 g^0`` is not a true per-leaf
alternating run but should still occupy its own row/slot in downstream plots.
The script writes one synthetic ``iteration_history.json`` per requested leaf
size under a standard ``ladder/dspy/leafTTTTtok`` layout so the regular plot
aggregator can ingest it alongside live ladder checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_int_grid(value: str) -> list[int]:
    values = [part.strip() for part in str(value or "").replace(";", ",").split(",")]
    out = [int(part) for part in values if part]
    if not out:
        raise ValueError("leaf size grid must contain at least one integer")
    return out


def _metric_row(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "n": int(raw.get("n", 0) or 0),
        "internal_f_pearson": None,
        "internal_f_mae_1_7": None,
        "external_expert_pearson": raw.get("pearson_r"),
        "external_expert_mae_1_7": raw.get("mae_1_7"),
        "f_star_gap": None,
        "mean_prediction_1_7": raw.get("mean_prediction_1_7"),
        "mean_teacher_1_7": None,
        "mean_expert_1_7": raw.get("mean_expert_1_7"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a synthetic ladder root from a baseline report.json."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--leaf-size-tokens", required=True)
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--stage-label", required=True)
    parser.add_argument("--family", default="dspy")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--trained", default="none")
    args = parser.parse_args(argv)

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    metrics = dict(report.get("metrics") or {})
    split_metrics = {
        split: _metric_row(raw)
        for split, raw in metrics.items()
        if split in {"all", "train", "val", "test"} and isinstance(raw, dict)
    }
    generated_at = str(report.get("generated_at") or "")
    leaf_sizes = _parse_int_grid(str(args.leaf_size_tokens))

    output_root = Path(args.output_root)
    ladder_root = output_root / "ladder" / str(args.family)
    ladder_root.mkdir(parents=True, exist_ok=True)

    per_row_paths: list[str] = []
    summary_rows: list[Dict[str, Any]] = []
    for leaf_size in leaf_sizes:
        row_dir = ladder_root / f"leaf{int(leaf_size):04d}tok"
        row_dir.mkdir(parents=True, exist_ok=True)
        row_payload = {
            "family": str(args.family),
            "axis_kind": "leaf_size_tokens",
            "axis_value": int(leaf_size),
            "leaf_count": None,
            "leaf_size_tokens": int(leaf_size),
            "row_label": f"leaf{int(leaf_size):04d}tok",
            "max_iterations": int(args.iteration),
            "eval_split": "test",
            "train_split": "train",
            "n_train_trees": None,
            "n_eval_trees": split_metrics.get("test", {}).get("n"),
            "iterations": [
                {
                    "iteration": int(args.iteration),
                    "stage_name": str(args.stage_name),
                    "stage_label": str(args.stage_label),
                    "family": str(args.family),
                    "trained": str(args.trained),
                    "f_degree": None,
                    "g_degree": None,
                    "axis_kind": "leaf_size_tokens",
                    "axis_value": int(leaf_size),
                    "leaf_count": None,
                    "leaf_size_tokens": int(leaf_size),
                    "f_artifact": None,
                    "g_artifact": None,
                    "split_metrics": split_metrics,
                    "extra": {
                        "synthetic_baseline": True,
                        "source_report": str(Path(args.report)),
                    },
                }
            ],
        }
        path = row_dir / "iteration_history.json"
        path.write_text(json.dumps(row_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        per_row_paths.append(str(path.relative_to(output_root)))
        test_metrics = split_metrics.get("test") or split_metrics.get("all") or {}
        summary_rows.append(
            {
                "family": str(args.family),
                "axis_kind": "leaf_size_tokens",
                "axis_value": int(leaf_size),
                "leaf_count": None,
                "leaf_size_tokens": int(leaf_size),
                "iteration": int(args.iteration),
                "stage_name": str(args.stage_name),
                "stage_label": str(args.stage_label),
                "trained": str(args.trained),
                "n_eval": test_metrics.get("n"),
                "internal_f_pearson": None,
                "external_expert_pearson": test_metrics.get("external_expert_pearson"),
                "f_star_gap": None,
                "internal_f_mae_1_7": None,
                "external_expert_mae_1_7": test_metrics.get("external_expert_mae_1_7"),
                "mean_prediction_1_7": test_metrics.get("mean_prediction_1_7"),
                "mean_teacher_1_7": None,
                "mean_expert_1_7": test_metrics.get("mean_expert_1_7"),
            }
        )

    grid_summary = {
        "created_at": generated_at,
        "dimension": str(report.get("dimension") or "economic"),
        "families": [str(args.family)],
        "topology_axis": "leaf_size_tokens",
        "leaf_grid": None,
        "leaf_size_tokens": leaf_sizes,
        "max_iterations": int(args.iteration),
        "eval_split": "test",
        "rows": summary_rows,
        "per_row_paths": per_row_paths,
        "synthetic_baseline": True,
        "source_report": str(Path(args.report)),
    }
    summary_path = output_root / "ladder" / "grid_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(grid_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "generated_at": generated_at,
        "report": str(Path(args.report)),
        "output_root": str(output_root),
        "stage_name": str(args.stage_name),
        "stage_label": str(args.stage_label),
        "leaf_size_tokens": leaf_sizes,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
