#!/usr/bin/env python3
"""Optuna HPO for the Benoit chunk-FNO (econ pilot) — tune on VAL, 4 GPUs.

Each trial runs the q-sentence FNO ladder (LLM-span node supervision grid) with
sampled hyperparameters, evaluates on the VAL split, and returns the root-prediction
vs Benoit-expert Pearson (maximize). Trials run concurrently, one per GPU (a queue
hands out GPUs). The winning config is then re-checked on TEST separately.

Search space is informed by the manual ablation (moderate capacity beat the 51M
"bump a TON" model, which overfit 100 train docs; root-weight ~10 beat 30), and adds
weight-decay / epochs to curb overfitting.

    ./venv/bin/python scripts/hpo_econ_chunkfno.py --n-trials 48 --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import queue
import statistics as s
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERT = json.loads(
    (PROJECT_ROOT / "outputs/benoit_qsentence_targets/expert_means_raw.json").read_text()
)
DIM = os.environ.get("HPO_DIM", "economic")
GRID = os.environ.get("HPO_GRID", f"outputs/benoit_chunkgrid_forced_{DIM}_llmspan")


def _root_vs_expert(run_dir: str, split: str) -> Optional[float]:
    prs = sorted(glob.glob(f"{run_dir}/fno/leafq016/prediction_records/*post_eval*"))
    if not prs:
        return None
    recs = [json.loads(l) for l in open(prs[-1])]
    bydoc = {}
    for r in recs:
        d = r.get("doc_id")
        lvl = r.get("level", 0) or 0
        if d is None:
            continue
        if d not in bydoc or lvl > bydoc[d][0]:
            bydoc[d] = (lvl, r.get("prediction"))
    xs, ys = [], []
    for d, (lvl, p) in bydoc.items():
        if p is None or d not in EXPERT or DIM not in EXPERT[d]:
            continue
        xs.append(float(p))
        ys.append(float(EXPERT[d][DIM]))
    if len(xs) < 3:
        return None
    mx, my = s.mean(xs), s.mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / len(xs)
    sx, sy = s.pstdev(xs), s.pstdev(ys)
    return cov / (sx * sy) if sx > 0 and sy > 0 else None


def _run_trial(params: dict, gpu: int, eval_split: str, out_dir: str) -> Optional[float]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TT_EXPORT_FULL_TREE_TRACES"] = "0"  # metrics-only (fast)
    cmd = [
        "./venv/bin/python", "scripts/run_manifesto_qsentence_dspy_ladder.py",
        "--family", "fno", "--embedding-backend", "local-hf",
        "--embedding-model", "/mnt/data/models/google/embeddinggemma-300m",
        "--embedding-device", "cuda", "--embedding-batch-size", "64",
        "--fno-device", "cuda",
        "--fg-grid-dir", GRID, "--leaf-qsentences", "16",
        "--max-iterations", "2", "--fno-target-dimension", DIM,
        "--eval-split", eval_split,
        "--fno-n-modes", str(params["n_modes"]),
        "--fno-hidden-channels", str(params["hidden"]),
        "--fno-n-layers", str(params["layers"]),
        "--fno-head-hidden-dim", str(params["head"]),
        "--fno-epochs", str(params["epochs"]),
        "--fno-learning-rate", f"{params['lr']:.5f}",
        "--fno-weight-decay", f"{params['wd']:.6f}",
        "--fno-leaf-weight", f"{params['lw']:.3f}",
        "--fno-merge-weight", f"{params['mw']:.3f}",
        "--fno-root-weight", f"{params['rw']:.3f}",
        "--output-dir", out_dir,
    ]
    with open(f"{out_dir}.log", "w") as lf:
        subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT), stdout=lf, stderr=lf, timeout=1800)
    return _root_vs_expert(out_dir, eval_split)


def main(argv=None) -> int:
    import optuna

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-trials", type=int, default=48)
    p.add_argument("--gpus", default="0,1,2,3")
    # Per-dim isolation: econ keeps its historical path; other dims get their own dir/study.
    default_root = "outputs/hpo_econ_chunkfno" if DIM == "economic" else f"outputs/hpo_{DIM}_chunkfno"
    p.add_argument("--out-root", default=default_root)
    p.add_argument("--study", default=None)
    p.add_argument("--eval-split", default="val")
    args = p.parse_args(argv)

    gpus = [int(g) for g in str(args.gpus).split(",") if g.strip()]
    gpu_q: "queue.Queue[int]" = queue.Queue()
    for g in gpus:
        gpu_q.put(g)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.study is None:
        args.study = str(out_root / "study.db")

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_modes": trial.suggest_categorical("n_modes", [128, 192, 256, 384, 512]),
            "hidden": trial.suggest_categorical("hidden", [32, 48, 64, 96]),
            "layers": trial.suggest_categorical("layers", [2, 3, 4]),
            "head": trial.suggest_categorical("head", [64, 128, 256]),
            "epochs": trial.suggest_categorical("epochs", [4, 8, 12, 16]),
            "lr": trial.suggest_float("lr", 1e-3, 6e-3, log=True),
            "wd": trial.suggest_float("wd", 1e-5, 3e-2, log=True),
            "lw": trial.suggest_float("lw", 0.5, 2.0),
            "mw": trial.suggest_float("mw", 0.1, 1.0),
            "rw": trial.suggest_float("rw", 2.0, 20.0),
        }
        gpu = gpu_q.get()
        try:
            out_dir = str(out_root / f"trial_{trial.number:04d}")
            val = _run_trial(params, gpu, str(args.eval_split), out_dir)
        finally:
            gpu_q.put(gpu)
        if val is None:
            return -1.0
        trial.set_user_attr("val_pearson", val)
        return val

    storage = f"sqlite:///{args.study}"
    study = optuna.create_study(
        direction="maximize", study_name=f"{DIM}_chunkfno",
        storage=storage, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=12),
    )
    study.optimize(objective, n_trials=int(args.n_trials), n_jobs=len(gpus))

    print("\n=== HPO complete ===")
    print(f"best val Pearson = {study.best_value:+.3f}")
    print("best params:", json.dumps(study.best_params, indent=2))
    (out_root / "best.json").write_text(
        json.dumps({"value": study.best_value, "params": study.best_params}, indent=2)
    )
    # top 5
    trials = sorted([t for t in study.trials if t.value is not None], key=lambda t: -t.value)[:5]
    print("\ntop 5:")
    for t in trials:
        print(f"  {t.value:+.3f}  {t.params}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
