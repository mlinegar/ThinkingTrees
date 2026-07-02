#!/usr/bin/env python3
# OLD/ARCHIVED (2026-06-11): imports the archived OLD_treepo_cld workspace.
# Superseded by ~/treepo/examples/research/methods/run_manifesto_fg_compile.py
"""Full f,g alternating compile on real Benoit manifestos through treepo_cld.

Loads canonical defaults from ``treepo_cld/configs/manifesto_fg_compile.toml``
(three sections: ``[family]`` → ``DSPyFamilyConfig``, ``[lm]``, ``[scenario]``).
CLI flags override individual fields. See ``treepo_cld/docs/training_defaults.md``.

Schedule (max_iterations=2, first_train_side='g'):
    k=0  eval (identity init), k=1 train_g, k=2 train_f

Train pool: ALL multi-leaf split=train trees (g training is per-node).
Eval pool: multi-leaf trees fitting under ``scenario.max_input_chars``
(k=0 raw_concat baseline needs the full document_text to fit f's context).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "treepo_cld" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

ART_DEFAULT = REPO_ROOT / "outputs/manifesto_dimension_fit_existing/economic_qwen_embedding_80_20_50/labeled_trees.jsonl"
SCORER_DEFAULT = REPO_ROOT / "outputs/phase1_gepa_v2_rank/economic/optimized_scorer.json"
CONFIG_DEFAULT = REPO_ROOT / "treepo_cld/configs/manifesto_fg_compile.toml"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Manifesto f,g compile — loads canonical defaults from TOML; CLI flags override.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, default=ART_DEFAULT)
    ap.add_argument("--scorer", type=Path, default=SCORER_DEFAULT)
    ap.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    ap.add_argument("--problem-id", default="manifesto_benoit")
    ap.add_argument("--dimension", default="economic")
    # Overrides — None means "use TOML value".
    ap.add_argument("--optimizer", default=None)
    ap.add_argument("--budget", default=None)
    ap.add_argument("--num-threads", type=int, default=None)
    ap.add_argument("--leaf-size-tokens", type=int, default=None)
    ap.add_argument("--lm-context-tokens", type=int, default=None,
                    dest="lm_context_window_tokens")
    ap.add_argument("--max-completion-tokens", type=int, default=None)
    ap.add_argument("--max-input-chars", type=int, default=None)
    ap.add_argument("--max-iterations", type=int, default=2)
    ap.add_argument("--first-train-side", default="g")
    ap.add_argument("--vllm-urls", default=None,
                    help="comma-separated endpoint list (overrides [lm].endpoints)")
    ap.add_argument("--vllm-model", default=None,
                    help="overrides [lm].model")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "run.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_path.open("a").write(line + "\n")

    log(f"output_dir={args.output_dir}  config={args.config}")

    from treepo_cld.canonical_defaults import build_lm_config_dict, load_dataclass, LmSection
    from src.ctreepo.dspy_family import DSPyFamilyConfig
    from src.ctreepo.distillation import load_labeled_trees
    import treepo_cld

    # Load two sections; CLI overrides go in via the loader.
    family = load_dataclass(args.config, DSPyFamilyConfig, section="family", overrides={
        "optimizer": args.optimizer,
        "budget": args.budget,
        "num_threads": args.num_threads,
        "leaf_size_tokens": args.leaf_size_tokens,
        "lm_context_window_tokens": args.lm_context_window_tokens,
        "max_completion_tokens": args.max_completion_tokens,
        "max_input_chars": args.max_input_chars,
    })
    lm = load_dataclass(args.config, LmSection, section="lm", overrides={
        "model": args.vllm_model,
        "endpoints": ([u.strip() for u in args.vllm_urls.split(",") if u.strip()]
                      if args.vllm_urls else None),
    })

    log(f"family: optimizer={family.optimizer} budget={family.budget} "
        f"leaf={family.leaf_size_tokens} ctx={family.lm_context_window_tokens} "
        f"max_completion={family.max_completion_tokens} "
        f"include_identity_targets={family.include_identity_targets}")
    log(f"canonical batch: size={family.batch_size} max_concurrent={family.batch_max_concurrent} "
        f"timeout={family.batch_timeout}s routing={family.batch_routing_policy}")

    # Validate endpoints.
    import urllib.request
    for u in lm.endpoints:
        try:
            urllib.request.urlopen(f"{u}/models", timeout=10).read()
        except Exception as exc:
            log(f"FATAL: vLLM endpoint not reachable: {u}: {exc}")
            return 2
    log(f"vLLM endpoints ({len(lm.endpoints)}): {lm.endpoints}")

    # Strong GEPA defaults live on family.gepa_kwargs (DSPyFamilyConfig field default).
    if family.optimizer.strip().lower() == "gepa":
        log(f"GEPA kwargs: {family.gepa_kwargs}")
    family.lm_config = build_lm_config_dict(lm, max_tokens=family.max_completion_tokens)
    family.problem_id = args.problem_id
    family.dimension = args.dimension
    family.f_init_path = str(args.scorer)

    # Tree pools.
    all_trees = load_labeled_trees(args.artifact)
    multi = [t for t in all_trees if t.num_chunks > 1]
    train_trees = [t for t in multi if (t.metadata or {}).get("split") == "train"]
    eval_trees = [t for t in multi
                  if family.max_input_chars is None
                  or len(t.document_text or "") < family.max_input_chars]
    train_splits: dict = {}
    for t in train_trees:
        s = (t.metadata or {}).get("split", "?")
        train_splits[s] = train_splits.get(s, 0) + 1
    eval_splits: dict = {}
    for t in eval_trees:
        s = (t.metadata or {}).get("split", "?")
        eval_splits[s] = eval_splits.get(s, 0) + 1

    log(f"loaded {len(all_trees)} trees, {len(multi)} multi-leaf")
    log(f"TRAIN pool: {len(train_trees)} trees, "
        f"num_chunks={sorted(t.num_chunks for t in train_trees)}")
    log(f"EVAL pool (<{family.max_input_chars} chars for k=0 raw_concat): "
        f"{len(eval_trees)} trees, splits={eval_splits}")
    log(f"  per-tree node total (≈ g records w/ identity targets): "
        f"~{sum(2*t.num_chunks - 1 for t in train_trees)}")

    # Dispatch.
    log("dispatching treepo_cld.run('fit', ...)")
    t0 = time.perf_counter()
    result = treepo_cld.run("fit", {
        "family": "dspy",
        "train_data": train_trees, "eval_data": eval_trees,
        "backend_config": {"dspy_config": family,
                           "output_dir": str(args.output_dir / "fit"),
                           "first_train_side": args.first_train_side},
        "axis": {"max_iterations": int(args.max_iterations), "axis_value": 0},
        "initial_artifacts": {"f": str(args.scorer), "g": "raw_concat"},
    })
    wall = time.perf_counter() - t0
    log(f"fit complete in {wall:.1f}s ({wall/60:.1f} min), status={result.status}")

    summary = {
        "wall_seconds": wall, "status": result.status,
        "n_train_trees": len(train_trees), "n_eval_trees": len(eval_trees),
        "splits": eval_splits, "train_splits": train_splits,
        "max_iterations": args.max_iterations, "first_train_side": args.first_train_side,
        "config": {
            "leaf_size_tokens": family.leaf_size_tokens,
            "lm_context_window_tokens": family.lm_context_window_tokens,
            "max_completion_tokens": family.max_completion_tokens,
            "include_identity_targets": family.include_identity_targets,
            "optimizer": family.optimizer, "budget": family.budget,
            "num_threads": family.num_threads,
            "batch_size": family.batch_size,
            "batch_max_concurrent": family.batch_max_concurrent,
            "batch_routing_policy": family.batch_routing_policy,
            "vllm_model": lm.model, "config_file": str(args.config),
        },
        "metrics": {k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in dict(result.metrics).items()},
        "history": [
            {"iteration": h.get("iteration"), "trained": h.get("trained"),
             "stage_name": h.get("stage_name"), "stage_label": h.get("stage_label"),
             "f_artifact": h.get("f_artifact"), "g_artifact": h.get("g_artifact"),
             "split_metrics": h.get("split_metrics"), "extra": h.get("extra")}
            for h in (result.history or [])
        ],
        "artifacts": {"f": result.artifacts.get("f"), "g": result.artifacts.get("g")},
        "manifest_path": result.manifest_path,
        "prediction_records": result.artifacts.get("prediction_records") or [],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    log(f"summary at {args.output_dir / 'summary.json'}")
    log("per-iter trained: " + " | ".join(
        f"k={h['iteration']}={h['trained']}" for h in summary["history"]
    ))
    log(f"final external_expert_pearson={summary['metrics'].get('external_expert_pearson')}, "
        f"external_expert_mae={summary['metrics'].get('external_expert_mae')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
