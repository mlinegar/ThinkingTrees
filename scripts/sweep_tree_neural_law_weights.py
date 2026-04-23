#!/usr/bin/env python3
"""Sweep tree neural FNO law weights to diagnose C1/C3 interference.

Runs multiple weight configurations in parallel across all 16 MIG slices.
Each subprocess gets CUDA_VISIBLE_DEVICES set to a specific MIG UUID.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Weight configurations to sweep: (name, root_w, c1_w, c2_w, c3_w)
WEIGHT_CONFIGS = [
    # Pure baselines
    ("root_only",        1.0, 0.0, 0.0, 0.0),
    ("c2_0.2",           1.0, 0.0, 0.2, 0.0),
    ("c2_0.5",           1.0, 0.0, 0.5, 0.0),
    ("c2_1.0",           1.0, 0.0, 1.0, 0.0),
    # C1 alone
    ("c1_0.1",           1.0, 0.1, 0.0, 0.0),
    ("c1_0.5",           1.0, 0.5, 0.0, 0.0),
    # C3 alone
    ("c3_0.1",           1.0, 0.0, 0.0, 0.1),
    ("c3_0.5",           1.0, 0.0, 0.0, 0.5),
    # C2 + C1 (does C1 hurt?)
    ("c2_0.2_c1_0.01",   1.0, 0.01, 0.2, 0.0),
    ("c2_0.2_c1_0.05",   1.0, 0.05, 0.2, 0.0),
    ("c2_0.2_c1_0.1",    1.0, 0.1,  0.2, 0.0),
    # C2 + C3 (does C3 hurt?)
    ("c2_0.2_c3_0.01",   1.0, 0.0, 0.2, 0.01),
    ("c2_0.2_c3_0.05",   1.0, 0.0, 0.2, 0.05),
    ("c2_0.2_c3_0.1",    1.0, 0.0, 0.2, 0.1),
    # All three at different scales
    ("all_0.01",          1.0, 0.01, 0.01, 0.01),
    ("all_0.1",           1.0, 0.1,  0.1,  0.1),
    # C2-dominant mix
    ("c2_dom_mix",        1.0, 0.01, 0.2, 0.01),
]


def _get_mig_uuids() -> list[str]:
    """Parse MIG UUIDs from nvidia-smi -L output."""
    result = subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True, check=True
    )
    uuids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if "MIG" in line and "UUID:" in line:
            uuid = line.split("UUID: ")[1].rstrip(")")
            uuids.append(uuid)
    return uuids


WORKER_TEMPLATE = r'''#!/usr/bin/env python3
import json, sys, time, os
import numpy as np
import torch

sys.path.insert(0, "{repo_root}")

from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (
    _fit_tree_neural_baseline_with_predictions,
    _materialize_base_bundle,
    _bundle_with_fixed_eval_splits,
    resolve_full_doc_diagnostic_benchmark,
    _base_config_for_benchmark,
)
from src.ctreepo.sim.core.markov_changepoint_ops_count import OPSCountConfig

name = "{name}"
root_w, c1_w, c2_w, c3_w = {root_w}, {c1_w}, {c2_w}, {c3_w}
train_docs_count = {train_docs}
n_epochs = {n_epochs}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Worker {{name}} using device: {{device}}", file=sys.stderr)

benchmark = resolve_full_doc_diagnostic_benchmark("recoverable_v4")

# Materialize the data bundle (reuses cache if available).
base_bundle, base_source = _materialize_base_bundle(
    benchmark=benchmark,
    required_train_docs=train_docs_count,
    output_dir=None,
)
bundle, _ = _bundle_with_fixed_eval_splits(
    base_bundle=base_bundle,
    base_source=base_source,
    train_doc_count=train_docs_count,
)

config = _base_config_for_benchmark(
    benchmark=benchmark,
    train_docs=train_docs_count,
    use_cuda=torch.cuda.is_available(),
    cuda_device=0,
    torch_threads=1,
    seed=0,
)
# Override model config
config = OPSCountConfig(
    n_regimes=config.n_regimes,
    vocab_size=config.vocab_size,
    fixed_leaf_tokens=config.fixed_leaf_tokens,
    train_docs=train_docs_count,
    val_docs=len(bundle.val_docs),
    test_docs=len(bundle.test_docs),
    state_dim=128,
    hidden_dim=512,
    n_epochs=n_epochs,
    batch_size=64,
    lr=5e-4,
    weight_decay=0.0,
    violation_tau=getattr(config, 'violation_tau', 0.0),
)

t0 = time.monotonic()
result = _fit_tree_neural_baseline_with_predictions(
    config=config,
    seeds={{"effective_model_seed": 42}},
    device=device,
    train_docs=bundle.train_docs,
    val_docs=bundle.val_docs,
    test_docs=bundle.test_docs,
    root_weight=root_w,
    c1_weight=c1_w,
    c2_weight=c2_w,
    c3_weight=c3_w,
)
elapsed = time.monotonic() - t0

test_preds = np.asarray(result["test_preds"])
test_truths = np.asarray(result["test_truths"])
test_mae = float(np.mean(np.abs(test_preds - test_truths)))
tm = result.get("test_metrics")
fit_diag = result.get("fit_diag")

# SketchMetrics is a dataclass, use getattr
row = {{
    "name": name,
    "root_weight": root_w,
    "c1_weight": c1_w,
    "c2_weight": c2_w,
    "c3_weight": c3_w,
    "train_docs": train_docs_count,
    "n_epochs": n_epochs,
    "test_root_mae": test_mae,
    "test_leaf_mae": float(getattr(tm, "leaf_mae", float("nan"))),
    "test_c2_count_drift_r1_mae": float(getattr(tm, "c2_count_drift_r1_mae", float("nan"))),
    "test_c2_idempotence_mae": float(getattr(tm, "c2_idempotence_mae", float("nan"))),
    "test_merge_mae": float(getattr(tm, "merge_mae", float("nan"))),
    "test_leaf_violation_rate": float(getattr(tm, "leaf_violation_rate", float("nan"))),
    "test_merge_violation_rate": float(getattr(tm, "merge_violation_rate", float("nan"))),
    "elapsed_s": elapsed,
    "device": str(device),
}}
if hasattr(fit_diag, "best_epoch"):
    row["best_epoch"] = int(fit_diag.best_epoch)
    row["train_loss_final"] = float(fit_diag.train_loss_final)
    row["selection_metric_value"] = float(fit_diag.selection_metric_value)

print(json.dumps(row))
'''


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-docs", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=32)
    parser.add_argument("--output-dir", type=str,
                        default="outputs/tree_neural_law_weight_sweep")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mig_uuids = _get_mig_uuids()
    n_migs = len(mig_uuids)
    print(f"Found {n_migs} MIG slices")

    configs = WEIGHT_CONFIGS
    print(f"Running {len(configs)} weight configs across {n_migs} MIG slices")
    print(f"  train_docs={args.train_docs}, n_epochs={args.n_epochs}")
    print()

    t0 = time.monotonic()

    # Launch all as subprocesses, each with its own MIG UUID
    procs: list[tuple[str, subprocess.Popen, Path]] = []
    for i, (name, rw, c1w, c2w, c3w) in enumerate(configs):
        mig_uuid = mig_uuids[i % n_migs]
        script = WORKER_TEMPLATE.format(
            repo_root=str(REPO_ROOT),
            name=name,
            root_w=rw, c1_w=c1w, c2_w=c2w, c3_w=c3w,
            train_docs=args.train_docs,
            n_epochs=args.n_epochs,
        )

        script_path = out / f"worker_{name}.py"
        script_path.write_text(script)

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = mig_uuid

        log_path = out / f"worker_{name}.log"
        log_f = open(log_path, "w")

        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=log_f,
            env=env,
            cwd=str(REPO_ROOT),
        )
        procs.append((name, proc, log_path))
        print(f"  Launched {name:25s} on MIG {mig_uuid[:12]}... (PID {proc.pid})")

    print(f"\nAll {len(procs)} jobs launched. Waiting for results...\n")

    # Collect results as they finish
    results = []
    for name, proc, log_path in procs:
        stdout_bytes, _ = proc.communicate()
        stdout = stdout_bytes.decode().strip()
        if proc.returncode != 0:
            print(f"  {name:25s}  FAILED (exit {proc.returncode}), see {log_path}")
            continue
        try:
            # Last line of stdout is the JSON result
            row = json.loads(stdout.split("\n")[-1])
            results.append(row)
            print(f"  {name:25s}  root_mae={row['test_root_mae']:.4f}  "
                  f"leaf_mae={row['test_leaf_mae']:.4f}  "
                  f"c2_mae={row.get('test_c2_count_drift_r1_mae', row['test_c2_idempotence_mae']):.4f}  "
                  f"merge_mae={row['test_merge_mae']:.4f}  "
                  f"({row['elapsed_s']:.0f}s)")
        except (json.JSONDecodeError, IndexError, KeyError) as exc:
            print(f"  {name:25s}  PARSE ERROR: {exc}, stdout={stdout[:200]}")

    total = time.monotonic() - t0
    print(f"\nTotal wall time: {total:.0f}s")

    # Sort by test_root_mae
    results.sort(key=lambda r: r["test_root_mae"])

    print("\n" + "=" * 110)
    print("RESULTS (sorted by test_root_mae)")
    print("=" * 110)
    print(f"{'Name':25s} {'root_w':>6s} {'c1_w':>6s} {'c2_w':>6s} {'c3_w':>6s}  "
          f"{'root_MAE':>9s} {'leaf_MAE':>9s} {'c2_MAE':>9s} {'merge_MAE':>9s}")
    print("-" * 110)
    for r in results:
        print(f"{r['name']:25s} {r['root_weight']:6.2f} {r['c1_weight']:6.3f} "
              f"{r['c2_weight']:6.3f} {r['c3_weight']:6.3f}  "
              f"{r['test_root_mae']:9.4f} {r['test_leaf_mae']:9.4f} "
              f"{r.get('test_c2_count_drift_r1_mae', r['test_c2_idempotence_mae']):9.4f} "
              f"{r['test_merge_mae']:9.4f}")

    # Save
    Path(out / "sweep_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    print(f"\nSaved to {out / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
