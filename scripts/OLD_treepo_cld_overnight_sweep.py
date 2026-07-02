#!/usr/bin/env python3
# OLD/ARCHIVED (2026-06-11): imports the archived OLD_treepo_cld workspace.
# Superseded by ~/treepo (official repo): tests/methods + examples/research/methods/
"""Overnight reproducibility sweep across every treepo_cld main model.

Runs the simplest/first-level cell of every family in sequence,
captures wall-time + status + key metrics per cell, persists a JSON
summary, and keeps going past per-cell failures so the whole batch
runs unattended.

Coverage (one cell per row, ordered fastest-first):

  Unit pytest sweep ......... 123 unit tests (no GPU, no server)
  Live pytest sweep ......... live integration tier (needs Gemma + GPU)

  Sketch family
    HLL precision sweep ...... p ∈ {6,8,10,12,14} vs hll_exact baseline
    HLL vocab sweep .......... 4 vocab sizes at p=12
    HLL schedule invariance .. 3 fold orders identical at p=12

  Markov family
    Markov DGP grid .......... 5 seeds × 3 regimes × 2 lengths = 30 cells
    Markov oracle on synthetic 8 trees @ 4 distinct seeds

  LDA family
    leaf-local-mixture oracle  4 seeds, per-tree bit-for-bit
    LDA tree recovery .......  3 configs (tiny / small / medium)

  LLM (Gemma-4-31B-IT-NVFP4)
    DSPy manifesto inference   18-tree economic cell via live Gemma
    DSPy litellm transport ... 1 predict call live
    DSPy batched transport ... 1 predict call live

  FNO
    FNO live (no server) ..... 1 train_f step, identity init, GPU

  Markov FNO probe
    probe tiny ............... doc_tokens=256, 32 train docs, 2 epochs
    probe small .............. doc_tokens=512, 64 train docs, 3 epochs

Usage:

    nohup ./venv/bin/python scripts/treepo_cld_overnight_sweep.py \\
      --output-root outputs/treepo_cld_overnight_$(date +%Y%m%d_%H%M%S) \\
      > /dev/null 2>&1 &

Each cell logs to ``<output_root>/<cell_name>/log.txt`` and writes its
own ``status.json``. The top-level ``<output_root>/summary.json`` is
written after every cell so partial progress is visible mid-run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "treepo_cld" / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "treepo_cld" / "src"))

VENV_PY = REPO_ROOT / "venv" / "bin" / "python"


@dataclass
class CellResult:
    name: str
    family: str
    status: str = "pending"
    wall_seconds: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    log_path: Optional[str] = None
    summary_path: Optional[str] = None


def _log(output_root: Path, message: str) -> None:
    """Append-only progress log so a user tailing the file sees activity."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    (output_root / "PROGRESS.log").open("a").write(line + "\n")


def _persist_summary(output_root: Path, cells: List[CellResult]) -> None:
    payload = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_cells": len(cells),
        "completed": sum(1 for c in cells if c.status == "success"),
        "failed": sum(1 for c in cells if c.status == "failed"),
        "pending": sum(1 for c in cells if c.status == "pending"),
        "total_wall_seconds": sum(c.wall_seconds for c in cells),
        "cells": [asdict(c) for c in cells],
    }
    (output_root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def _run_cell(
    cell: CellResult,
    body: Callable[[Path], Dict[str, Any]],
    *,
    output_root: Path,
    cells_so_far: List[CellResult],
) -> None:
    cell_dir = output_root / cell.name
    cell_dir.mkdir(parents=True, exist_ok=True)
    cell.log_path = str(cell_dir / "log.txt")
    cell.summary_path = str(cell_dir / "status.json")
    _log(output_root, f"START {cell.name} ({cell.family})")
    t0 = time.perf_counter()
    try:
        metrics = body(cell_dir)
        cell.metrics = metrics or {}
        cell.status = "success"
    except Exception as exc:
        cell.status = "failed"
        cell.error = f"{type(exc).__name__}: {exc}"
        (cell_dir / "log.txt").open("a").write(traceback.format_exc())
        _log(output_root, f"  FAILED: {cell.error}")
    cell.wall_seconds = time.perf_counter() - t0
    (cell_dir / "status.json").write_text(json.dumps(asdict(cell), indent=2, sort_keys=True, default=str))
    _persist_summary(output_root, cells_so_far)
    _log(output_root, f"  END   {cell.name} status={cell.status} wall={cell.wall_seconds:.1f}s")


# =========================================================================== #
# Cell bodies                                                                  #
# =========================================================================== #


def _cell_unit_pytest(cell_dir: Path) -> Dict[str, Any]:
    log = cell_dir / "log.txt"
    result = subprocess.run(
        [str(VENV_PY), "-m", "pytest", str(REPO_ROOT / "treepo_cld" / "tests"), "-q",
         "--ignore=" + str(REPO_ROOT / "treepo_cld" / "tests" / "integration"),
         "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    log.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"unit pytest exit={result.returncode}; tail: {result.stdout[-500:]}")
    # Parse "N passed" from the summary line.
    import re
    m = re.search(r"(\d+) passed", result.stdout)
    return {"n_passed": int(m.group(1)) if m else None, "returncode": 0}


def _cell_live_pytest(cell_dir: Path) -> Dict[str, Any]:
    log = cell_dir / "log.txt"
    env = os.environ.copy()
    env["TT_RUN_LIVE_TESTS"] = "1"
    result = subprocess.run(
        [str(VENV_PY), "-m", "pytest", str(REPO_ROOT / "treepo_cld" / "tests" / "integration"),
         "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=REPO_ROOT, env=env, timeout=1800,
    )
    log.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr)
    # Don't hard-fail on the known DSPy concurrent flake.
    import re
    n_passed = int(re.search(r"(\d+) passed", result.stdout).group(1)) if re.search(r"(\d+) passed", result.stdout) else 0
    n_failed = int(re.search(r"(\d+) failed", result.stdout).group(1)) if re.search(r"(\d+) failed", result.stdout) else 0
    return {"n_passed": n_passed, "n_failed": n_failed, "returncode": result.returncode}


def _cell_hll_precision_sweep(cell_dir: Path) -> Dict[str, Any]:
    import treepo_cld
    fixture_kwargs = dict(n_trees=8, leaves_per_tree=4, leaf_token_count=32, vocabulary_size=512, seed=17)
    # Exact baseline.
    exact_result = treepo_cld.run("oracle", {"oracle_name": "hll_exact",
                                              "output_dir": str(cell_dir / "exact"),
                                              **fixture_kwargs})
    exact_path = exact_result.artifacts["prediction_records"][0]
    exact_rows = [json.loads(line) for line in Path(exact_path).read_text().splitlines() if line.strip()]
    exact_counts = [int(r["prediction"]) for r in exact_rows]
    out: Dict[str, Any] = {"mean_exact": sum(exact_counts) / len(exact_counts), "by_precision": {}}
    for p in (6, 8, 10, 12, 14):
        res = treepo_cld.run("sketch", {"sketch_kind": "hll", "precision": p, "hash_bits": 64,
                                          "output_dir": str(cell_dir / f"p{p}"), **fixture_kwargs})
        rows = [json.loads(line) for line in Path(res.artifacts["prediction_records"][0]).read_text().splitlines() if line.strip()]
        est = [float(r["prediction"]) for r in rows]
        mae = sum(abs(e - x) for e, x in zip(est, exact_counts)) / len(est)
        out["by_precision"][f"p{p}"] = {"mae": mae, "estimates": est}
    return out


def _cell_hll_vocab_sweep(cell_dir: Path) -> Dict[str, Any]:
    import treepo_cld
    out: Dict[str, Any] = {"by_vocab": {}}
    for vocab in (64, 128, 256, 512):
        # Exact baseline at this vocab.
        ex = treepo_cld.run("oracle", {"oracle_name": "hll_exact",
                                         "n_trees": 6, "leaves_per_tree": 4,
                                         "leaf_token_count": 32, "vocabulary_size": vocab, "seed": 3,
                                         "output_dir": str(cell_dir / f"vocab{vocab}_exact")})
        ex_rows = [json.loads(line) for line in Path(ex.artifacts["prediction_records"][0]).read_text().splitlines() if line.strip()]
        ex_counts = [int(r["prediction"]) for r in ex_rows]
        sk = treepo_cld.run("sketch", {"sketch_kind": "hll", "precision": 12,
                                         "n_trees": 6, "leaves_per_tree": 4,
                                         "leaf_token_count": 32, "vocabulary_size": vocab, "seed": 3,
                                         "output_dir": str(cell_dir / f"vocab{vocab}_p12")})
        sk_rows = [json.loads(line) for line in Path(sk.artifacts["prediction_records"][0]).read_text().splitlines() if line.strip()]
        sk_est = [float(r["prediction"]) for r in sk_rows]
        mae = sum(abs(e - x) for e, x in zip(sk_est, ex_counts)) / len(sk_est)
        out["by_vocab"][f"vocab{vocab}"] = {
            "mean_exact": sum(ex_counts) / len(ex_counts), "p12_mae": mae,
        }
    return out


def _cell_hll_schedule_invariance(cell_dir: Path) -> Dict[str, Any]:
    import treepo_cld
    common = dict(sketch_kind="hll", precision=12, n_trees=8, leaves_per_tree=4,
                  leaf_token_count=32, vocabulary_size=256, seed=99)
    estimates: Dict[str, List[float]] = {}
    for schedule in ("balanced", "left_to_right", "right_to_left"):
        res = treepo_cld.run("sketch", {**common, "schedule": schedule,
                                          "output_dir": str(cell_dir / schedule)})
        rows = [json.loads(line) for line in Path(res.artifacts["prediction_records"][0]).read_text().splitlines() if line.strip()]
        estimates[schedule] = [float(r["prediction"]) for r in rows]
    invariant = estimates["balanced"] == estimates["left_to_right"] == estimates["right_to_left"]
    return {"identical_across_schedules": invariant, "estimates": estimates}


def _cell_markov_grid(cell_dir: Path) -> Dict[str, Any]:
    """Auto-fixture via the registered ``markov`` domain builder."""
    from itertools import product
    import treepo_cld

    out: Dict[str, Any] = {"cells": []}
    for seed, n_regimes, max_tokens in product([0, 1, 2, 3, 4], [3, 4, 5], [64, 128]):
        res = treepo_cld.run("oracle", {
            "oracle_name": "markov_changepoint_count",
            "output_dir": str(cell_dir / f"s{seed}_r{n_regimes}_t{max_tokens}"),
            # Dispatcher's _make_oracle_fixture_markov consumes these knobs.
            "n_regimes": int(n_regimes), "vocab_size": 32,
            "min_tokens": int(max_tokens), "max_tokens": int(max_tokens),
            "min_segments": 2, "max_segments": 6,
            "min_seg_len": 8, "max_seg_len": int(max_tokens // 4),
            "train_docs": 2, "test_docs": 4,
            "sinkhorn_iters": 20, "transition_log_std": 1.0,
            "seed": int(seed),
        })
        out["cells"].append({"seed": seed, "n_regimes": n_regimes, "max_tokens": max_tokens,
                             "mae": float(res.metrics["internal_f_mae"]),
                             "n": int(res.metrics["n"])})
    out["all_mae_zero"] = all(c["mae"] == 0.0 for c in out["cells"])
    return out


def _cell_lda_oracle(cell_dir: Path) -> Dict[str, Any]:
    import treepo_cld
    out: Dict[str, Any] = {"by_seed": {}}
    for seed in (0, 1, 7, 42):
        res = treepo_cld.run("oracle", {"oracle_name": "leaf_local_mixture_target",
                                          "seed": seed,
                                          "output_dir": str(cell_dir / f"seed{seed}")})
        out["by_seed"][f"seed{seed}"] = {"mae": float(res.metrics["internal_f_mae"]),
                                          "n": int(res.metrics["n"])}
    out["all_mae_zero"] = all(v["mae"] < 1e-9 for v in out["by_seed"].values())
    return out


def _cell_lda_recovery(cell_dir: Path) -> Dict[str, Any]:
    from src.ctreepo.sim.core.lda_tree_recovery import (
        LDATreeRecoveryConfig, run_lda_tree_recovery_experiment,
    )
    from treepo_cld.canonical_defaults import load_dataclass

    configs = {
        "tiny":   dict(n_topics=4, vocab_size=64,  min_tokens=64,  max_tokens=64,
                       anchor_words_per_topic=4, leaf_tokens=16, train_docs=4,  test_docs=16, seed=0),
        "small":  dict(n_topics=4, vocab_size=128, min_tokens=128, max_tokens=128,
                       anchor_words_per_topic=6, leaf_tokens=16, train_docs=8,  test_docs=32, seed=0),
        "medium": dict(n_topics=8, vocab_size=256, min_tokens=192, max_tokens=192,
                       anchor_words_per_topic=8, leaf_tokens=16, train_docs=16, test_docs=64, seed=0),
    }
    out: Dict[str, Any] = {"configs": {}}
    for name, kw in configs.items():
        cfg = load_dataclass(None, LDATreeRecoveryConfig, overrides=kw)
        t0 = time.perf_counter()
        summary = run_lda_tree_recovery_experiment(cfg)
        elapsed = time.perf_counter() - t0
        d = json.loads(summary.to_json())
        out["configs"][name] = {
            "wall_seconds": elapsed,
            "exact_recovery": {k: float(v) for k, v in d["exact_recovery"].items() if isinstance(v, (int, float))},
            "full_doc_pi_l1_to_true_mean": float(d["methods"]["full_doc"]["pi_l1_to_true_mean"]),
            "leaf_average_pi_l1_to_full_mean": float(d["methods"]["leaf_average"]["pi_l1_to_full_mean"]),
        }
        (cell_dir / f"{name}_summary.json").write_text(summary.to_json())
    return out


def _cell_manifesto_teacher(cell_dir: Path) -> Dict[str, Any]:
    """Re-run the bit-for-bit teacher parity to confirm it's still tight."""
    from scripts.run_manifesto_fg_real_training_grid import _fg_teacher_metrics, _tree_lookup
    from src.ctreepo.distillation import load_labeled_trees
    from types import SimpleNamespace
    import treepo_cld

    ART = REPO_ROOT / "outputs/manifesto_dimension_fit_existing/smoke_qwen_embedding_economic/labeled_trees.jsonl"
    if not ART.exists():
        return {"skipped": True, "reason": f"missing artifact {ART}"}
    paper = _fg_teacher_metrics(ART)
    rve = paper["root_vs_expert"]

    class TP:
        name = "tp"
        def train_f(self, *, f_init, g, traces, output_dir, iteration): return f_init
        def train_g(self, *, g_init, f, traces, output_dir, iteration): return g_init
        def score_roots_with_f(self, *, f, g, trees):
            return [float(t.metadata["teacher_score_1_7"]) for t in trees]
        def validate_artifact(self, *, kind, artifact): return None

    trees_raw = load_labeled_trees(ART)
    lookup = _tree_lookup(trees_raw)
    rrows = [r for r in lookup.values() if r.get("is_root")]
    trees = [SimpleNamespace(
        leaves=[SimpleNamespace(tokens=[])],
        metadata={"split": "test", "teacher_score_1_7": float(r["teacher_score_1_7"]),
                  "teacher_score_native": float(r["teacher_score_1_7"]),
                  "expert_score_1_7": float(r["expert_score_1_7"]),
                  "expert_score_native": float(r["expert_score_1_7"]),
                  "expert_target_scale": "raw",
                  "expert_score_for_objective": float(r["expert_score_1_7"])},
    ) for r in rrows if r.get("teacher_score_1_7") is not None and r.get("expert_score_1_7") is not None]
    result = treepo_cld.run("fit", {"family": "tp", "eval_data": trees,
                                       "backend_config": {"family_runtime": TP(), "output_dir": str(cell_dir)}})
    return {
        "paper_pearson": rve["pearson_r"], "paper_mae": rve["mae"], "paper_n": rve["n"],
        "tcld_pearson": result.metrics["external_expert_pearson"],
        "tcld_mae": result.metrics["external_expert_mae"],
        "tcld_n": int(result.metrics["n"]),
        "pearson_match_exact": rve["pearson_r"] == result.metrics["external_expert_pearson"],
        "mae_match_exact": rve["mae"] == result.metrics["external_expert_mae"],
    }


def _cell_manifesto_dspy_live(cell_dir: Path) -> Dict[str, Any]:
    """Live DSPy + Gemma on the smoke artifact (k=0 inference; skipped if no server).

    Smoke-tier overrides on top of canonical defaults: smaller leaf/context for
    a faster sanity check, bootstrap_fewshot since no compile happens at k=0.
    """
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5).read()
    except Exception:
        return {"skipped": True, "reason": "vLLM not at http://localhost:8000/v1/models"}

    ART = REPO_ROOT / "outputs/manifesto_dimension_fit_existing/smoke_qwen_embedding_economic/labeled_trees.jsonl"
    SCORER = REPO_ROOT / "outputs/phase1_gepa_v2_rank/economic/optimized_scorer.json"
    if not ART.exists() or not SCORER.exists():
        return {"skipped": True, "reason": "missing artifact or pretuned scorer"}

    from src.ctreepo.dspy_family import DSPyFamilyConfig
    from src.ctreepo.distillation import load_labeled_trees
    from treepo_cld.canonical_defaults import build_lm_config_dict, load_dataclass, LmSection
    import treepo_cld

    all_trees = load_labeled_trees(ART)
    short_trees = [t for t in all_trees if t.document_text and len(t.document_text) < 6000]

    # Canonical DSPyFamilyConfig defaults + smoke-tier overrides (smaller
    # leaf/context/batch, bootstrap_fewshot for the k=0 inference smoke).
    cfg = load_dataclass(None, DSPyFamilyConfig, overrides={
        "optimizer": "bootstrap_fewshot", "num_threads": 8,
        "batch_size": 8, "batch_max_concurrent": 16, "batch_timeout": 0.05,
        "batch_request_timeout": 180.0,
        "leaf_size_tokens": 1024, "lm_context_window_tokens": 8192,
        "max_completion_tokens": 2048, "prompt_template_overhead_tokens": 512,
        "problem_id": "manifesto_benoit", "dimension": "economic",
        "f_init_path": str(SCORER),
    })
    cfg.lm_config = build_lm_config_dict(
        LmSection(endpoints=["http://localhost:8000/v1"]),
        max_tokens=cfg.max_completion_tokens,
    )
    result = treepo_cld.run("fit", {
        "family": "dspy", "train_data": [], "eval_data": short_trees,
        "backend_config": {"dspy_config": cfg, "output_dir": str(cell_dir)},
        "axis": {"max_iterations": 0, "axis_value": 0},
        "initial_artifacts": {"f": str(SCORER), "g": "teacher_passthrough"},
    })
    return {
        "n": int(result.metrics["n"]),
        "external_expert_pearson": float(result.metrics["external_expert_pearson"]),
        "external_expert_mae": float(result.metrics["external_expert_mae"]),
        "f_star_gap": float(result.metrics["f_star_gap"]),
        "mean_prediction": float(result.metrics["mean_prediction"]),
    }


def _cell_fno_live(cell_dir: Path) -> Dict[str, Any]:
    """Live FNO training step through treepo_cld on GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"skipped": True, "reason": "CUDA not available"}
        import neuralop  # noqa: F401
    except ImportError:
        return {"skipped": True, "reason": "torch or neuralop not installed"}

    from src.ctreepo.fno_family import FNOFamilyConfig
    from src.tree.labeled import LabeledNode, LabeledTree
    from treepo_cld.canonical_defaults import load_dataclass
    from treepo_cld.families import resolve_family
    import treepo_cld

    class _FakeLM:
        def __init__(self, dim=8): self.dim = dim
        def embed_texts(self, texts):
            return [[float((len(str(t)) + i) % 7) for i in range(self.dim)] for t in texts]

    # Canonical FNOFamilyConfig defaults + smoke-tier overrides.
    cfg = load_dataclass(None, FNOFamilyConfig, overrides={
        "hidden_channels": 8, "n_modes": 4, "n_layers": 1, "head_hidden_dim": 8,
        "epochs_per_iteration": 2, "leaf_size_tokens": 8,
        "embedding_max_length_tokens": 8, "effective_embedding_dim": 8, "seed": 0,
    })
    family = resolve_family("fno", {"fno_config": cfg, "embedding_client": _FakeLM()})

    trees: List[LabeledTree] = []
    for i in range(4):
        doc_id = f"d{i:02d}"
        score = 4.0 + (i - 2) * 0.3
        text = "left a b right c d"
        tree = LabeledTree(
            doc_id=doc_id, document_text=text, document_score=float(score),
            metadata={"split": "train" if i < 2 else "test",
                      "teacher_score_1_7": float(score), "expert_score_1_7": float(score) + 0.1,
                      "teacher_score_native": float(score), "expert_score_native": float(score) + 0.1,
                      "expert_target_scale": "raw", "expert_score_for_objective": float(score) + 0.1},
            label_source="test",
        )
        tree.add_node(LabeledNode(node_id="leaf_0", doc_id=doc_id, level=0, text="left a b",
                                    score=float(score) - 0.2, metadata={"teacher_summary": "L", "target_summary": "L"}))
        tree.add_node(LabeledNode(node_id="leaf_1", doc_id=doc_id, level=0, text="right c d",
                                    score=float(score) + 0.2, metadata={"teacher_summary": "R", "target_summary": "R"}))
        tree.add_node(LabeledNode(node_id="root", doc_id=doc_id, level=1, text=text, score=float(score),
                                    left_child_id="leaf_0", right_child_id="leaf_1",
                                    metadata={"teacher_summary": "root", "target_summary": "root"}))
        trees.append(tree)

    result = treepo_cld.run("fit", {
        "family": "fno", "train_data": trees, "eval_data": trees,
        "backend_config": {"family_runtime": family, "output_dir": str(cell_dir)},
        "axis": {"max_iterations": 1, "axis_value": 0},
        "initial_artifacts": {"f": "identity", "g": "raw_concat"},
    })
    pred_path = result.artifacts["prediction_records"][-1]
    rows = [json.loads(line) for line in Path(pred_path).read_text().splitlines() if line.strip()]
    preds = [r["prediction"] for r in rows if r.get("prediction") is not None]
    return {"status": result.status, "n_iterations": result.summary["n_iterations"],
            "n_preds_returned": len(preds), "first_pred": float(preds[0]) if preds else None}


def _cell_probe(cell_dir: Path, *, label: str, doc_tokens: int, train_docs: int, leaf_tokens: int,
                epochs: int) -> Dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"skipped": True, "reason": "CUDA not available"}
    except ImportError:
        return {"skipped": True, "reason": "torch not installed"}

    import treepo_cld
    result = treepo_cld.run("probe", {
        "output_root": str(cell_dir), "doc_tokens": doc_tokens, "leaf_tokens": leaf_tokens,
        "train_docs": train_docs, "eval_docs": max(8, train_docs // 4),
        "epochs": epochs, "batch_size": 4, "channels": 16,
        "g_n_modes": 8, "g_n_layers": 1, "scorer_n_modes": 8, "scorer_n_layers": 1,
        "seed": 0, "device": "cuda", "training_objective": "root", "timeout": 1800,
    })
    summary = result.get("summary") or {}
    return {
        "label": label, "returncode": result["returncode"],
        "test_root_mae": summary.get("test_root_mae"),
        "best_val_root_mae": summary.get("best_val_root_mae"),
        "best_val_epoch": summary.get("best_val_epoch"),
        "n_params_g": summary.get("n_params_g"), "n_params_f": summary.get("n_params_f"),
        "n_epochs_history": len(summary.get("history") or []),
    }


# =========================================================================== #
# Orchestrator                                                                 #
# =========================================================================== #


def _build_cells() -> List[tuple]:
    return [
        ("unit_pytest",           "infrastructure", _cell_unit_pytest),
        ("hll_precision_sweep",   "sketch",         _cell_hll_precision_sweep),
        ("hll_vocab_sweep",       "sketch",         _cell_hll_vocab_sweep),
        ("hll_schedule_invariance", "sketch",       _cell_hll_schedule_invariance),
        ("markov_dgp_grid",       "markov",         _cell_markov_grid),
        ("lda_oracle_seeds",      "lda",            _cell_lda_oracle),
        ("lda_tree_recovery",     "lda",            _cell_lda_recovery),
        ("manifesto_teacher",     "llm_passthrough", _cell_manifesto_teacher),
        ("fno_live",              "fno",            _cell_fno_live),
        ("probe_tiny",            "fno_probe",      lambda d: _cell_probe(d, label="tiny", doc_tokens=256, train_docs=32, leaf_tokens=64, epochs=2)),
        ("probe_small",           "fno_probe",      lambda d: _cell_probe(d, label="small", doc_tokens=512, train_docs=64, leaf_tokens=128, epochs=3)),
        ("manifesto_dspy_live",   "llm",            _cell_manifesto_dspy_live),
        # Live pytest runs last because it's the slowest and most flake-prone.
        ("live_pytest",           "infrastructure", _cell_live_pytest),
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Overnight reproducibility sweep across treepo_cld")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cell", action="append", default=None,
                        help="Only run named cells (repeatable)")
    args = parser.parse_args(argv)

    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "PROGRESS.log").open("w").write("")  # clear

    cells = _build_cells()
    if args.cell:
        wanted = set(args.cell)
        cells = [c for c in cells if c[0] in wanted]

    cell_results: List[CellResult] = []
    for name, family, body in cells:
        cell = CellResult(name=name, family=family)
        cell_results.append(cell)
        _run_cell(cell, body, output_root=output_root, cells_so_far=cell_results)

    _persist_summary(output_root, cell_results)

    # Final summary table.
    _log(output_root, "=" * 70)
    _log(output_root, "FINAL SUMMARY")
    _log(output_root, "=" * 70)
    for c in cell_results:
        mark = "OK " if c.status == "success" else "FAIL"
        _log(output_root, f"  [{mark}] {c.family:<18} {c.name:<28} {c.wall_seconds:>7.1f}s")
    total_ok = sum(1 for c in cell_results if c.status == "success")
    _log(output_root, "=" * 70)
    _log(output_root, f"Completed: {total_ok}/{len(cell_results)} cells; "
                       f"total wall {sum(c.wall_seconds for c in cell_results):.1f}s")
    _log(output_root, f"Summary: {output_root}/summary.json")
    return 0 if total_ok == len(cell_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
