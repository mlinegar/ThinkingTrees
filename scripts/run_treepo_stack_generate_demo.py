#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments import (
    ResultRow,
    benchmark_ref_from_parts,
    chat_role_ref,
    metadata_with_roles,
    method_ref_from_parts,
    oracle_ref,
    sidecar_root_for_output_file,
    state_model_role_ref,
    write_canonical_sidecars,
)


def _parse_kv(items: List[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Empty key in {item!r}.")
        # Small convenience: try JSON first so users can pass numbers/bools/lists.
        try:
            parsed[key] = json.loads(value)
        except Exception:
            parsed[key] = value
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo of the unified TreePO stack over /generate (preferred) with chat fallback."
    )
    parser.add_argument("--engine", type=str, default="sglang", help="Engine kind (sglang, vllm, openai, custom_http).")
    parser.add_argument("--model", type=str, default="default", help="Model identifier for the engine.")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL (host:port root; /generate suffix is OK).")
    parser.add_argument("--surface", type=str, default="generate", choices=("generate", "chat"), help="Requested surface.")
    parser.add_argument("--generate-path", type=str, default="/generate", help="Generate endpoint path.")

    parser.add_argument("--rubric", type=str, default="Preserve named entities, numbers, and causal claims.")
    parser.add_argument("--refine-rounds", type=int, default=1)
    parser.add_argument("--max-concurrent", type=int, default=64)

    parser.add_argument("--sampling-param", action="append", default=[], help="Sampling param KEY=VALUE (JSON allowed).")
    parser.add_argument("--engine-option", action="append", default=[], help="Engine option KEY=VALUE (JSON allowed).")

    parser.add_argument(
        "--oracle-import-path",
        type=str,
        default="src.tree.auditor:SimpleScorer",
        help="Python import path to a ScoringOracle factory (default: simple word overlap scorer).",
    )
    parser.add_argument(
        "--oracle-kwargs",
        type=str,
        default="{}",
        help="JSON dict of kwargs passed to the oracle factory (default: {}).",
    )

    parser.add_argument(
        "--leaf",
        action="append",
        default=[],
        help="Leaf text span. Repeat to provide multiple leaves. If omitted, uses a small built-in example.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write full run JSON.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    from src.tree import OracleLaneSpec, TreePOContractSpec, TreePOModelSpec, build_treepo_stack

    leaves = list(args.leaf or [])
    if not leaves:
        leaves = [
            "Alice paid Bob $10 on Tuesday.",
            "Later, Bob refunded Alice $5.",
            "Alice says the refund was incomplete.",
        ]

    sampling_params: Optional[Mapping[str, Any]] = None
    engine_options: Optional[Mapping[str, Any]] = None
    if args.sampling_param:
        sampling_params = _parse_kv(list(args.sampling_param))
    if args.engine_option:
        engine_options = _parse_kv(list(args.engine_option))

    try:
        oracle_kwargs = json.loads(str(args.oracle_kwargs or "{}"))
        if not isinstance(oracle_kwargs, dict):
            raise ValueError("--oracle-kwargs must be a JSON object.")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --oracle-kwargs: {args.oracle_kwargs!r}") from exc

    stack = build_treepo_stack(
        TreePOModelSpec(
            kind="inference_engine",
            engine=str(args.engine),
            model=str(args.model),
            base_url=args.base_url,
            surface=str(args.surface),
            generate_path=str(args.generate_path),
        ),
        TreePOContractSpec(
            rubric=str(args.rubric),
            oracle_lane=OracleLaneSpec(
                kind="provided_scoring_oracle",
                import_path=str(args.oracle_import_path),
                kwargs=dict(oracle_kwargs),
            ),
        ),
    )

    result = stack.run_fixed_binary(
        leaves,
        refine_rounds=int(args.refine_rounds),
        sampling_params=sampling_params,
        engine_options=engine_options,
        max_concurrent=int(args.max_concurrent),
    )

    treepo_meta = result.tree.metadata.get("treepo_stack", {})
    print("\nFinal summary:\n")
    print(result.tree.final_rendered)
    print("\nChosen surface:\n")
    print(json.dumps(treepo_meta, indent=2, sort_keys=True))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        benchmark_ref = benchmark_ref_from_parts(
            family="treepo_stack_demo",
            scope="generate",
            name="treepo_stack_generate_demo",
            dataset_id="inline_leaves",
            metadata={"leaf_count": len(leaves)},
        )
        method_ref = method_ref_from_parts(
            family="treepo_stack_generate",
            variant=str(args.surface),
            adapter="treepo_stack_demo",
            metadata=metadata_with_roles(
                {"engine": str(args.engine), "surface": str(args.surface)},
                roles={
                    "summarizer": chat_role_ref(
                        role="summarizer",
                        engine=str(args.engine),
                        model=str(args.model),
                        base_url=str(args.base_url or ""),
                    ),
                    "state_model": state_model_role_ref(
                        engine="treepo_stack",
                        model=str(args.surface),
                    ),
                },
                oracle=oracle_ref(kind="provided_scoring_oracle", source=str(args.oracle_import_path)),
            ),
        )
        write_canonical_sidecars(
            sidecar_root_for_output_file(args.output_json),
            title="treepo_stack_generate_demo",
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
                    metric_name="operations",
                    metric_value=len(result.operations),
                    artifact_refs=("output_json",),
                ),
            ),
            state="completed",
            metadata={"surface": str(args.surface)},
            launch_command=sys.argv,
            report_profiles=("runtime_eval_summary",),
        )
        print(f"\nWrote: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
