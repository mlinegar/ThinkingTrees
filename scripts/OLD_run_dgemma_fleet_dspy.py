# OLD_: archived 2026-07-02; depends on treepo._research, removed in the treepo 2026-06 minimization. Kept for reference; do not import or run.
#!/usr/bin/env python
"""Live dspy scoring on dgemma, fanned across a multi-GPU vLLM fleet.

dgemma is treated as a normal LLM: the DSPy family consumes it via the standard
OpenAI transport with `api_bases` round-robin across the fleet (all GPUs), exactly
as it would a regular LLM — the only dgemma-specific bit is dropping
`response_format` (TT_DSPY_DROP_RESPONSE_FORMAT). Writes a JSON summary.

Run with:
  TT_DSPY_DROP_RESPONSE_FORMAT=1 run_dgemma_fleet_dspy.py \
      --ports 8004 8005 8006 8007 \
      --model RedHatAI/diffusiongemma-26B-A4B-it-NVFP4 --out OUTDIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import treepo.methods as M

# Reuse the existing economic pretuned scorer + smoke labeled trees (real data).
TREEPO_ROOT = Path("/home/mlinegar/treepo")
PRETUNED = TREEPO_ROOT / "outputs/phase1_gepa_v2_rank/economic/optimized_scorer.json"
LABELED = TREEPO_ROOT / "outputs/manifesto_dimension_fit_existing/smoke_qwen_embedding_economic/labeled_trees.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", nargs="+", type=int, default=[8004, 8005, 8006, 8007])
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--model", default="RedHatAI/diffusiongemma-26B-A4B-it-NVFP4")
    ap.add_argument("--max-trees", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    api_bases = [f"http://{args.host}:{p}/v1" for p in args.ports]
    print(f"dspy fleet endpoints: {api_bases}")

    from treepo._research.ctreepo.distillation import load_labeled_trees
    from treepo._research.ctreepo.dspy_family import DSPyFamilyConfig

    trees = [t for t in load_labeled_trees(LABELED) if t.document_text and len(t.document_text) < 6000]
    trees = trees[: args.max_trees]
    print(f"loaded {len(trees)} eval trees")

    cfg = DSPyFamilyConfig(
        optimizer="bootstrap_fewshot",
        budget="light",
        num_threads=16,
        target_min=1.0, target_max=7.0,
        scorer_output_min=1.0, scorer_output_max=7.0,
        lm_transport="batch",
        batch_size=8, batch_max_concurrent=32, batch_timeout=0.05,
        batch_request_timeout=180.0,
        batch_routing_policy="round_robin",  # fan across the whole fleet -> all GPUs
        leaf_size_tokens=1024,
        lm_context_window_tokens=8192,
        max_completion_tokens=2048,
        prompt_template_overhead_tokens=512,
        lm_config={
            "model": f"openai/{args.model}",
            "api_bases": api_bases,
            "api_key": "EMPTY",
            "temperature": 0.0,
            "max_tokens": 2048,
            "cache": False,
        },
        problem_id="manifesto_benoit",
        dimension="economic",
        f_init_mode="pretuned_scorer",
        f_init_path=str(PRETUNED),
    )

    result = M.run("fit", {
        "family": "dspy",
        "train_data": [],
        "eval_data": trees,
        "backend_config": {"dspy_config": cfg, "output_dir": str(out / "dspy_fit")},
        "axis": {"max_iterations": 0, "axis_value": 0},
        "initial_artifacts": {"f": str(PRETUNED), "g": "teacher_passthrough"},
    })

    metrics = dict(getattr(result, "metrics", {}) or {})
    summary = {
        "model": args.model,
        "fleet": api_bases,
        "n_eval_trees": len(trees),
        "status": getattr(result, "status", None),
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
    }
    (out / "dspy_fleet_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[dspy fleet] status={summary['status']} "
          f"n={metrics.get('n')} pearson={metrics.get('external_expert_pearson')}")
    print(f"DONE — summary at {out / 'dspy_fleet_summary.json'}")


if __name__ == "__main__":
    main()
