#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments import (
    ResultRow,
    benchmark_ref_from_parts,
    metadata_with_roles,
    method_ref_from_parts,
    oracle_ref,
    sidecar_root_for_output_file,
    state_model_role_ref,
    write_canonical_sidecars,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small demo of the unified TreePO stack on the exact Markov toy lane.")
    parser.add_argument(
        "--path",
        type=str,
        default="a a b b a c c",
        help="Space-separated Markov states (e.g. 'a a b b a').",
    )
    parser.add_argument("--chunk-size", type=int, default=2, help="Chunk size for leaf spans.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write run JSON.")
    return parser.parse_args()


def _chunk(items: Sequence[str], chunk_size: int) -> List[List[str]]:
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    values = list(items)
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def main() -> int:
    args = _parse_args()

    from src.diffusion.markov_toy import encode_markov_path
    from src.tree import OracleLaneSpec, TreePOContractSpec, TreePOModelSpec, build_treepo_stack

    path = [token for token in str(args.path or "").split() if token]
    if not path:
        raise ValueError("--path must contain at least one state token.")

    leaf_spans = _chunk(path, int(args.chunk_size))
    stack = build_treepo_stack(
        TreePOModelSpec(kind="markov_toy_exact"),
        TreePOContractSpec(
            rubric="(unused for Markov toy lane)",
            oracle_lane=OracleLaneSpec(kind="markov_exact"),
        ),
    )

    result = stack.run_fixed_binary(leaf_spans, refine_rounds=0)
    full = encode_markov_path(path)
    root_state = result.tree.root.state

    summary = {
        "path": path,
        "chunk_size": int(args.chunk_size),
        "leaf_spans": leaf_spans,
        "root_state": result.tree.root.to_dict().get("state"),
        "full_path_state": {
            "changepoints": full.changepoints,
            "start_state": full.start_state,
            "end_state": full.end_state,
            "length": full.length,
        },
        "root_matches_full_path": bool(root_state == full),
        "root_law_checks": result.tree.root.audit.get("law_checks"),
        "operations": [op.to_dict() for op in result.operations],
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        benchmark_ref = benchmark_ref_from_parts(
            family="treepo_stack_demo",
            scope="markov_toy",
            name="treepo_stack_markov_demo",
            dataset_id="inline_path",
            metadata={"path_length": len(path), "chunk_size": int(args.chunk_size)},
        )
        method_ref = method_ref_from_parts(
            family="treepo_stack_markov_exact",
            variant="markov_toy_exact",
            adapter="treepo_stack_demo",
            metadata=metadata_with_roles(
                {"chunk_size": int(args.chunk_size)},
                roles={
                    "state_model": state_model_role_ref(
                        engine="markov_toy_exact",
                        model="markov_toy_exact",
                    )
                },
                oracle=oracle_ref(kind="markov_exact", source="markov_toy"),
            ),
        )
        write_canonical_sidecars(
            sidecar_root_for_output_file(args.output_json),
            title="treepo_stack_markov_demo",
            adapter_id="treepo_stack_demo",
            benchmark_refs=(benchmark_ref,),
            method_refs=(method_ref,),
            phases=("demo",),
            artifacts={"output_json": str(args.output_json)},
            result_rows=(
                ResultRow(
                    experiment_id="",
                    phase="demo",
                    benchmark_ref=benchmark_ref,
                    method_ref=method_ref,
                    metric_name="root_matches_full_path",
                    metric_value=bool(root_state == full),
                    artifact_refs=("output_json",),
                ),
            ),
            state="completed",
            metadata={"chunk_size": int(args.chunk_size)},
            launch_command=sys.argv,
            report_profiles=("runtime_eval_summary",),
        )
        print(f"\nWrote: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
