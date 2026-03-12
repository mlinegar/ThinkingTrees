#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.memory_defaults import recommend_manifesto_memory_defaults


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recommend sensible manifesto-memory defaults from a perf harness artifact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifact", type=Path, required=True, help="Path to run_perf_harness artifact JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: sibling of artifact as recommended_defaults.json).",
    )
    parser.add_argument("--min-delta-count", type=int, default=20, help="Minimum eligible temporal samples.")
    parser.add_argument("--max-rile-mae", type=float, default=0.20, help="Maximum eligible RILE MAE.")
    parser.add_argument(
        "--fallback-scenario-id",
        type=str,
        default="temporal_main_sem_on_learned_chunker",
        help="Fallback temporal scenario id when no valid run candidates exist.",
    )
    args = parser.parse_args()

    artifact_path = args.artifact
    if not artifact_path.is_absolute():
        artifact_path = (Path.cwd() / artifact_path).resolve()
    if not artifact_path.exists():
        raise SystemExit(f"Artifact not found: {artifact_path}")

    output_path = args.output
    if output_path is None:
        output_path = artifact_path.parent / "recommended_defaults.json"
    elif not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    recommendation = recommend_manifesto_memory_defaults(
        payload,
        min_delta_count=max(1, int(args.min_delta_count)),
        max_rile_mae=max(0.0, float(args.max_rile_mae)),
        fallback_scenario_id=str(args.fallback_scenario_id),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")

    selected = recommendation.get("selected_scenario_id")
    defaults = recommendation.get("recommended_defaults", {})
    training_flags = recommendation.get("training_flags", [])
    print(f"selected_scenario_id={selected}")
    print(
        "defaults="
        + json.dumps(
            {
                "semantic_memory_features": defaults.get("semantic_memory_features"),
                "learn_loss_weights": defaults.get("learn_loss_weights"),
                "windowing_mode": defaults.get("windowing_mode"),
            },
            sort_keys=True,
        )
    )
    if isinstance(training_flags, list):
        print("training_flags=" + " ".join(str(v) for v in training_flags))
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
