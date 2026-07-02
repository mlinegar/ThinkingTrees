#!/usr/bin/env python3
"""Throughput probe for the Markov t2048 composition-stress sweep.

Sweeps (fixed_leaf_tokens, supervision_batch_size) to find the largest batch
size per leaf-tokens rung that fits in GPU memory and sustains the highest
leaves/second throughput. Uses the real tradeoff_pipeline runner with a
small training budget (1 package, train_docs=512, supervision_epochs=2).

Output: outputs/markov_t2048_throughput_<timestamp>/probe_results.jsonl plus
a recommended per-rung batch-size map printed at the end and saved as
recommended_batch_sizes.json.

Example:
  ./venv/bin/python scripts/probe_markov_t2048_throughput.py \
    --gpus 0,1,2,3 \
    --output-root outputs/markov_t2048_throughput_$(date +%Y%m%d_%H%M%S)
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


# (leaf_tokens, leaves_per_doc, candidate batch sizes in docs)
# leaves_per_doc = 2048 / leaf_tokens
DEFAULT_GRID: List[Tuple[int, int, List[int]]] = [
    (2048, 1, [256, 512, 1024, 2048]),
    (1024, 2, [128, 256, 512, 1024]),
    (512, 4, [64, 128, 256, 512]),
    (256, 8, [32, 64, 128, 256]),
    (128, 16, [16, 32, 64, 128]),
    (64, 32, [8, 16, 32, 64]),
    (32, 64, [4, 8, 16, 32]),
    (16, 128, [2, 4, 8, 16]),
]


def _emit_config(
    *,
    output_dir: Path,
    leaf_tokens: int,
    batch_size: int,
    train_docs: int,
    epochs: int,
) -> Path:
    """Write a tiny TOML config for one (leaf_tokens, batch_size) probe cell."""
    config_text = f"""# Throughput probe: leaf_tokens={leaf_tokens}, batch_size={batch_size}
[tradeoff_pipeline]
preset = "standard"
phases = ["supervision_recovery"]
device_mode = "cuda"
train_docs = {train_docs}
val_docs = 32
test_docs = 32
supervision_recovery_train_docs = [{train_docs}]
supervision_recovery_seeds = [0]
supervision_recovery_packages = ["full10"]
supervision_recovery_method_id = "tree_neural"
supervision_recovery_recoverable_benchmark = "recoverable_v5_t2048"
supervision_recovery_structural_cell = "r12_p079"
supervision_recovery_scope_keys = ["recoverable_v5_t2048"]
supervision_fixed_leaf_tokens = {leaf_tokens}
supervision_recovery_leaf_token_ladder = [{leaf_tokens}]
supervision_min_tokens = 2048
supervision_max_tokens = 2048
supervision_epochs = {epochs}
supervision_batch_size = {batch_size}
exact_metric_final_doc_limit = 32
tree_posttrain_train_doc_limit = 32
tree_stage1_artifact_root = "outputs/_stage1_artifacts/throughput_probe_t2048_lt{leaf_tokens}_bs{batch_size}"

[tradeoff_pipeline.tree_reference]
mode = "preset"
preset = "unified_g_full_local_laws_v1"

[tradeoff_pipeline.structural_tree_reference]
mode = "preset"
preset = "unified_g_full_local_laws_v1"

[tradeoff_pipeline.runtime]
data_mode = "resident"
bucket_mode = "leaf_count_auto_queue"
tree_batch_structural_pad_limit = 0.5
tree_batch_auto_queue_min_docs = 8
tree_batch_auto_queue_min_fill_ratio = 0.5
preload_splits = ["train", "val", "test"]
preload_targets = true
workers_per_mig = 1

[tradeoff_pipeline.scheduler]
mode = "global_per_run"
default_job_granularity = "family_train_seed"
cleanup_stale_children = true
max_gpu_items_per_mig = 1
"""
    config_path = output_dir / f"probe_lt{leaf_tokens}_bs{batch_size}.toml"
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


def _gpu_uuid_for_index(index: int) -> str:
    """Return the GPU UUID for nvidia-smi index (e.g. 0)."""
    out = subprocess.run(
        ["nvidia-smi", "-i", str(int(index)), "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    uuid = out.stdout.strip().splitlines()[0].strip()
    if not uuid:
        raise RuntimeError(f"could not resolve UUID for GPU index {index}")
    return uuid


def _run_probe_cell(
    *,
    output_root: Path,
    leaf_tokens: int,
    batch_size: int,
    train_docs: int,
    epochs: int,
    gpu_id: int,
    timeout_s: int,
) -> Dict[str, Any]:
    cell_dir = output_root / f"lt{leaf_tokens}_bs{batch_size}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    config_path = _emit_config(
        output_dir=cell_dir,
        leaf_tokens=leaf_tokens,
        batch_size=batch_size,
        train_docs=train_docs,
        epochs=epochs,
    )
    log_path = cell_dir / "run.log"
    gpu_uuid = _gpu_uuid_for_index(int(gpu_id))
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_markov_optimization_tradeoff_pipeline.py"),
        "--config", str(config_path),
        "--output-root", str(cell_dir / "run"),
        "--device-mode", "cuda",
        "--max-workers", "1",
        "--migs", gpu_uuid,
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    started = time.perf_counter()
    status = "ok"
    error_excerpt = ""
    try:
        with log_path.open("w") as logf:
            logf.write(f"# command: {shlex.join(cmd)}\n")
            logf.write(f"# gpu_uuid={gpu_uuid}\n")
            logf.flush()
            proc = subprocess.run(
                cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
        rc = int(proc.returncode)
        if rc != 0:
            status = f"exit_{rc}"
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            error_excerpt = "\n".join(tail)
    except subprocess.TimeoutExpired:
        rc = -1
        status = "timeout"
    wall_s = time.perf_counter() - started
    leaves_per_doc = 2048 // int(leaf_tokens)
    leaves_per_batch = int(batch_size) * int(leaves_per_doc)
    total_leaves = int(train_docs) * int(leaves_per_doc) * int(epochs)
    leaves_per_sec = float(total_leaves) / float(wall_s) if wall_s > 0 and status == "ok" else 0.0
    return {
        "leaf_tokens": int(leaf_tokens),
        "batch_size": int(batch_size),
        "leaves_per_doc": int(leaves_per_doc),
        "leaves_per_batch": int(leaves_per_batch),
        "train_docs": int(train_docs),
        "epochs": int(epochs),
        "wall_s": float(wall_s),
        "leaves_per_sec": float(leaves_per_sec),
        "status": status,
        "rc": int(rc),
        "gpu_id": int(gpu_id),
        "log_path": str(log_path),
        "error_excerpt": error_excerpt[:2000],
    }


def _run_gpu_queue_module(
    gpu_id: int,
    queue: List[Tuple[int, int, int]],
    output_root_str: str,
    train_docs: int,
    epochs: int,
    timeout_s: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    output_root = Path(output_root_str)
    for cell_idx, leaf_tokens, batch_size in queue:
        result = _run_probe_cell(
            output_root=output_root,
            leaf_tokens=int(leaf_tokens),
            batch_size=int(batch_size),
            train_docs=int(train_docs),
            epochs=int(epochs),
            gpu_id=int(gpu_id),
            timeout_s=int(timeout_s),
        )
        result["cell_idx"] = int(cell_idx)
        out.append(result)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3",
                        help="Comma-separated GPU ids to use in parallel")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--train-docs", type=int, default=512,
                        help="Train docs per probe cell (small for speed)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Supervision epochs per probe cell")
    parser.add_argument("--timeout-s", type=int, default=900,
                        help="Per-cell timeout in seconds")
    parser.add_argument("--leaf-tokens", type=str, default=None,
                        help="Comma-separated subset of leaf_tokens rungs to probe")
    parser.add_argument("--max-batch-only", action="store_true",
                        help="Only probe the largest batch-size candidate per rung")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in str(args.gpus).split(",") if x.strip()]
    if not gpu_ids:
        raise SystemExit("--gpus must list at least one GPU id")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or (REPO_ROOT / "outputs" / f"markov_t2048_throughput_{timestamp}")
    output_root.mkdir(parents=True, exist_ok=True)

    leaf_tokens_filter = (
        {int(x) for x in str(args.leaf_tokens).split(",") if x.strip()}
        if args.leaf_tokens else None
    )

    # Build cell list
    cells: List[Tuple[int, int]] = []
    for leaf_tokens, _leaves_per_doc, bs_candidates in DEFAULT_GRID:
        if leaf_tokens_filter and int(leaf_tokens) not in leaf_tokens_filter:
            continue
        candidates = [max(bs_candidates)] if args.max_batch_only else list(bs_candidates)
        for bs in candidates:
            cells.append((int(leaf_tokens), int(bs)))

    print(f"[probe] {len(cells)} cells across {len(gpu_ids)} GPUs", flush=True)
    print(f"[probe] output: {output_root}", flush=True)
    results_path = output_root / "probe_results.jsonl"

    # GPU affinity by leaf_tokens: same-leaf_tokens cells share a GPU so the
    # prepared-data cache built by the first cell is reused by subsequent ones.
    # Each GPU runs its assigned cells serially via a per-GPU queue.
    unique_leaf_tokens = sorted({lt for lt, _ in cells}, reverse=True)
    leaf_to_gpu = {lt: gpu_ids[idx % len(gpu_ids)] for idx, lt in enumerate(unique_leaf_tokens)}
    queues_by_gpu: Dict[int, List[Tuple[int, int, int]]] = {gid: [] for gid in gpu_ids}
    for cell_idx, (leaf_tokens, batch_size) in enumerate(cells):
        queues_by_gpu[int(leaf_to_gpu[leaf_tokens])].append(
            (int(cell_idx), int(leaf_tokens), int(batch_size))
        )

    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = {
            pool.submit(
                _run_gpu_queue_module,
                int(gid),
                queue,
                str(output_root),
                int(args.train_docs),
                int(args.epochs),
                int(args.timeout_s),
            ): int(gid)
            for gid, queue in queues_by_gpu.items()
            if queue
        }
        with results_path.open("w") as fh:
            for future in as_completed(futures):
                gid = futures[future]
                try:
                    queue_results = future.result()
                except Exception as exc:
                    queue_results = [{
                        "leaf_tokens": -1,
                        "batch_size": -1,
                        "gpu_id": int(gid),
                        "status": f"queue_exception: {type(exc).__name__}: {exc}",
                        "rc": -1,
                        "wall_s": 0.0,
                        "leaves_per_sec": 0.0,
                    }]
                for result in queue_results:
                    results.append(result)
                    fh.write(json.dumps(result) + "\n")
                    fh.flush()
                    tag = (
                        f"[{int(result.get('cell_idx', -1))+1}/{len(cells)}] "
                        f"lt={int(result['leaf_tokens']):4d} "
                        f"bs={int(result['batch_size']):4d} "
                        f"gpu={int(result['gpu_id'])} "
                        f"status={str(result.get('status', '?')):>10s} "
                        f"wall={float(result.get('wall_s', 0)):6.1f}s "
                        f"leaves/s={float(result.get('leaves_per_sec', 0)):.0f}"
                    )
                    print(tag, flush=True)

    # Build recommendation: max-leaves/sec OK cell per leaf_tokens
    by_lt: Dict[int, List[Dict[str, Any]]] = {}
    for r in results:
        by_lt.setdefault(int(r["leaf_tokens"]), []).append(r)

    recommended: Dict[int, int] = {}
    print("\n=== Throughput summary ===", flush=True)
    print(f"{'leaf_tokens':>11} {'batch_size':>10} {'leaves/sec':>11} {'wall_s':>8} {'status':>12}")
    for lt in sorted(by_lt.keys(), reverse=True):
        cells_lt = sorted(by_lt[lt], key=lambda r: int(r["batch_size"]))
        ok_cells = [c for c in cells_lt if c.get("status") == "ok"]
        for c in cells_lt:
            print(
                f"{lt:>11d} {int(c['batch_size']):>10d} "
                f"{c.get('leaves_per_sec', 0):>11.0f} "
                f"{c.get('wall_s', 0):>8.1f} "
                f"{c.get('status', '?'):>12s}"
            )
        if ok_cells:
            best = max(ok_cells, key=lambda r: float(r.get("leaves_per_sec", 0.0)))
            recommended[int(lt)] = int(best["batch_size"])

    print("\n=== Recommended per-rung batch sizes ===", flush=True)
    print(json.dumps({str(k): v for k, v in sorted(recommended.items(), reverse=True)}, indent=2))
    rec_path = output_root / "recommended_batch_sizes.json"
    rec_path.write_text(json.dumps(
        {str(k): int(v) for k, v in sorted(recommended.items(), reverse=True)},
        indent=2,
    ), encoding="utf-8")
    print(f"\nWrote {rec_path}", flush=True)
    print(f"Wrote {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
