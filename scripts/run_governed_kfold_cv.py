#!/usr/bin/env python3
"""
Governed CV runner with a frozen outer split (dev/practical/lockbox).

Default workflow:
1) Build (or reuse) an outer split manifest:
   - dev_pool: used for inner k-fold CV only
   - practical_val: held out during tuning
   - lockbox_test: untouched until final reporting
2) Run either:
   - k>=2: `scripts/run_kfold_cv.py` on dev_pool only.
          (per-fold test on one fold; val carved from the remaining train pool)
   - k=1: a single `src.training.run_pipeline` run with train/val from dev_pool,
          and test from practical_val (default) or lockbox_test.
3) Emit split JSON artifacts for later final evaluation.

Example:
  ./venv/bin/python scripts/run_governed_kfold_cv.py \
    --task manifesto_rile --dataset manifesto \
    --cv-output-dir outputs/manifesto_governed_cv_$(date +%Y%m%d_%H%M) \
    --n-samples 400 --k 5 --stratify dist --bins 10 --seed 42 \
    --phase1-cache --phase1-cache-dynamic-gpu \
    --max-parallel-folds 1 \
    -- \
    --dynamic-gpu --dynamic-gpu-hard-quiesce \
    --optimizer gepa --optimizer-budget medium --max-metric-calls 300
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_kfold_cv import _assign_bin, _load_samples_for_cv, _quantile_edges


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _normalize_fracs(dev: float, practical: float, lockbox: float) -> Dict[str, float]:
    vals = {
        "dev_pool": float(dev),
        "practical_val": float(practical),
        "lockbox_test": float(lockbox),
    }
    if any(v < 0.0 for v in vals.values()):
        raise ValueError("Outer split fractions must be non-negative.")
    total = sum(vals.values())
    if total <= 0.0:
        raise ValueError("At least one outer split fraction must be > 0.")
    return {k: v / total for k, v in vals.items()}


def _counts_from_fracs(total: int, fracs: Dict[str, float]) -> Dict[str, int]:
    raw = {name: float(total) * float(frac) for name, frac in fracs.items()}
    base = {name: int(math.floor(value)) for name, value in raw.items()}
    assigned = sum(base.values())
    remainder = int(total) - int(assigned)
    if remainder > 0:
        ranked = sorted(raw.keys(), key=lambda name: (raw[name] - base[name], raw[name]), reverse=True)
        for idx in range(remainder):
            base[ranked[idx % len(ranked)]] += 1
    return base


def _make_outer_split_indices(
    strat_values: Sequence[float],
    *,
    bins: int,
    seed: int,
    target_counts: Dict[str, int],
) -> Dict[str, List[int]]:
    edges = _quantile_edges(strat_values, bins=max(2, int(bins)))
    by_bin: Dict[int, List[int]] = defaultdict(list)
    for idx, value in enumerate(strat_values):
        by_bin[_assign_bin(float(value), edges)].append(idx)

    rng = random.Random(int(seed))
    for bucket in by_bin.values():
        rng.shuffle(bucket)

    groups = ["dev_pool", "practical_val", "lockbox_test"]
    remaining = {name: int(target_counts.get(name, 0)) for name in groups}
    targets = {name: max(1, int(target_counts.get(name, 0))) for name in groups}
    assigned: Dict[str, List[int]] = {name: [] for name in groups}
    order = {name: idx for idx, name in enumerate(groups)}

    for bin_idx in sorted(by_bin.keys()):
        for row_idx in by_bin[bin_idx]:
            candidates = [name for name in groups if remaining[name] > 0]
            if not candidates:
                break
            choice = max(
                candidates,
                key=lambda name: (
                    remaining[name] / targets[name],
                    remaining[name],
                    -order[name],
                ),
            )
            assigned[choice].append(int(row_idx))
            remaining[choice] -= 1

    if any(v != 0 for v in remaining.values()):
        raise RuntimeError(f"Failed to assign all rows to outer splits. Remaining: {remaining}")

    for values in assigned.values():
        values.sort()
    return assigned


def _coerce_doc_ids(raw: Any, *, label: str) -> List[str]:
    if not isinstance(raw, list):
        raise ValueError(f"Outer split key '{label}' must be a JSON list")
    out: List[str] = []
    seen = set()
    for item in raw:
        if item is None:
            continue
        doc_id = str(item).strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
    return out


def _load_outer_split(path: Path) -> Dict[str, List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    splits = payload.get("splits") if isinstance(payload, dict) else None
    block = splits if isinstance(splits, dict) else payload
    if not isinstance(block, dict):
        raise ValueError(f"Outer split payload must be a JSON object: {path}")

    out = {
        "dev_pool": _coerce_doc_ids(block.get("dev_pool", []), label="dev_pool"),
        "practical_val": _coerce_doc_ids(block.get("practical_val", []), label="practical_val"),
        "lockbox_test": _coerce_doc_ids(block.get("lockbox_test", []), label="lockbox_test"),
    }

    overlap = (
        set(out["dev_pool"]).intersection(out["practical_val"])
        | set(out["dev_pool"]).intersection(out["lockbox_test"])
        | set(out["practical_val"]).intersection(out["lockbox_test"])
    )
    if overlap:
        raise ValueError(f"Outer split has overlapping doc_ids ({len(overlap)} overlaps): {path}")
    return out


def _build_single_run_split(
    outer_splits: Dict[str, List[str]],
    *,
    seed: int,
    dev_val_frac: float,
    test_pool: str,
) -> Dict[str, List[str]]:
    dev_pool = list(outer_splits.get("dev_pool", []))
    practical = list(outer_splits.get("practical_val", []))
    lockbox = list(outer_splits.get("lockbox_test", []))

    if len(dev_pool) < 2:
        raise ValueError(f"k=1 mode needs at least 2 dev_pool docs, got {len(dev_pool)}")

    rng = random.Random(int(seed))
    rng.shuffle(dev_pool)
    val_frac = max(0.05, min(0.5, float(dev_val_frac)))
    n_val = int(round(float(len(dev_pool)) * val_frac))
    n_val = max(1, min(len(dev_pool) - 1, n_val))

    val_ids = sorted(dev_pool[:n_val])
    train_ids = sorted(dev_pool[n_val:])

    if str(test_pool) == "lockbox_test":
        test_ids = sorted(lockbox)
    else:
        test_ids = sorted(practical)
        if not test_ids and lockbox:
            test_ids = sorted(lockbox)
    if not test_ids:
        test_ids = list(val_ids)

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _build_inner_cv_command(args: argparse.Namespace, *, outer_split_path: Path, dev_pool_size: int) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/run_kfold_cv.py",
        "--task",
        str(args.task),
        "--dataset",
        str(args.dataset),
        "--cv-output-dir",
        str(Path(args.cv_output_dir) / str(args.inner_cv_subdir)),
        "--n-samples",
        str(int(dev_pool_size)),
        "--k",
        str(int(args.k)),
        "--bins",
        str(int(args.bins)),
        "--seed",
        str(int(args.seed)),
        "--stratify",
        str(args.stratify),
        "--neutral",
        str(float(args.neutral)),
        "--train-pool-val-frac",
        str(float(args.inner_val_frac)),
        "--max-parallel-folds",
        str(int(args.max_parallel_folds)),
        "--restrict-doc-ids-path",
        str(outer_split_path),
        "--restrict-doc-ids-key",
        "dev_pool",
    ]
    if args.dataset_path:
        cmd.extend(["--dataset-path", str(args.dataset_path)])
    if not bool(args.shuffle):
        cmd.append("--no-shuffle")
    if bool(args.phase1_cache):
        cmd.append("--phase1-cache")
    if int(args.phase1_cache_concurrent_docs) > 0:
        cmd.extend(["--phase1-cache-concurrent-docs", str(int(args.phase1_cache_concurrent_docs))])
    if int(args.phase1_cache_concurrent_requests) > 0:
        cmd.extend(["--phase1-cache-concurrent-requests", str(int(args.phase1_cache_concurrent_requests))])

    if args.phase1_cache_dynamic_gpu is True:
        cmd.append("--phase1-cache-dynamic-gpu")
    elif args.phase1_cache_dynamic_gpu is False:
        cmd.append("--no-phase1-cache-dynamic-gpu")

    if bool(args.reuse_dynamic_gpu_servers):
        cmd.append("--reuse-dynamic-gpu-servers")
    else:
        cmd.append("--no-reuse-dynamic-gpu-servers")

    if bool(args.shutdown_servers_after_cv):
        cmd.append("--shutdown-servers-after-cv")
    else:
        cmd.append("--no-shutdown-servers-after-cv")

    if bool(args.rebuild_phase1_cache):
        cmd.append("--rebuild-phase1-cache")
    if bool(args.make_fold_reports):
        cmd.append("--make-fold-reports")
    if bool(args.dry_run):
        cmd.append("--dry-run")

    forwarded = list(args.pipeline_args or [])
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if forwarded:
        cmd.extend(["--", *forwarded])
    return cmd


def _build_single_run_command(
    args: argparse.Namespace,
    *,
    split_path: Path,
    output_dir: Path,
    split_counts: Dict[str, int],
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "src.training.run_pipeline",
        "--task",
        str(args.task),
        "--dataset",
        str(args.dataset),
        "--output-dir",
        str(output_dir),
        "--split-ids-path",
        str(split_path),
        "--train-samples",
        str(int(split_counts.get("train", 0))),
        "--val-samples",
        str(int(split_counts.get("val", 0))),
        "--test-samples",
        str(int(split_counts.get("test", 0))),
    ]
    if args.dataset_path:
        cmd.extend(["--dataset-path", str(args.dataset_path)])

    forwarded = list(args.pipeline_args or [])
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if forwarded:
        cmd = [cmd[0], cmd[1], cmd[2], *forwarded, *cmd[3:]]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run k-fold CV with a frozen outer split (dev/practical/lockbox).")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--cv-output-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, required=True, help="Total sampled docs before outer splitting.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stratify", type=str, default="dist", choices=["dist", "actual"])
    parser.add_argument("--neutral", type=float, default=0.5)

    parser.add_argument("--dev-frac", type=float, default=0.70)
    parser.add_argument("--practical-val-frac", type=float, default=0.15)
    parser.add_argument("--lockbox-test-frac", type=float, default=0.15)
    parser.add_argument("--outer-split-path", type=Path, default=None)
    parser.add_argument("--rebuild-outer-split", action="store_true")
    parser.add_argument("--inner-cv-subdir", type=str, default="inner_cv")

    parser.add_argument("--max-parallel-folds", type=int, default=1)
    parser.add_argument(
        "--k1-dev-val-frac",
        type=float,
        default=0.15,
        help="When --k=1, hold out this fraction of dev_pool for validation (clipped to [0.05, 0.5]).",
    )
    parser.add_argument(
        "--k1-test-pool",
        type=str,
        default="practical_val",
        choices=["practical_val", "lockbox_test"],
        help="When --k=1, use this outer pool for the test split.",
    )
    parser.add_argument(
        "--inner-val-frac",
        type=float,
        default=0.15,
        help="When --k>=2, hold out this fraction of each fold's train pool for validation.",
    )
    parser.add_argument("--phase1-cache", action="store_true")
    parser.add_argument("--phase1-cache-concurrent-docs", type=int, default=0)
    parser.add_argument("--phase1-cache-concurrent-requests", type=int, default=0)
    parser.add_argument("--phase1-cache-dynamic-gpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reuse-dynamic-gpu-servers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shutdown-servers-after-cv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-phase1-cache", action="store_true")
    parser.add_argument("--make-fold-reports", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument(
        "pipeline_args",
        nargs=argparse.REMAINDER,
        help=(
            "Arguments forwarded after '--'. "
            "For k>=2: forwarded to scripts/run_kfold_cv.py and then run_pipeline. "
            "For k=1: forwarded directly to run_pipeline."
        ),
    )
    args = parser.parse_args()

    if int(args.n_samples) <= 0:
        raise SystemExit("--n-samples must be > 0")
    if int(args.k) < 1:
        raise SystemExit("--k must be >= 1")

    cv_dir = Path(args.cv_output_dir)
    cv_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = cv_dir / "splits"
    outer_split_path = Path(args.outer_split_path) if args.outer_split_path else (splits_dir / "outer_split.json")

    if outer_split_path.exists() and not bool(args.rebuild_outer_split):
        outer_splits = _load_outer_split(outer_split_path)
        print(f"Using existing outer split: {outer_split_path}")
    else:
        fracs = _normalize_fracs(args.dev_frac, args.practical_val_frac, args.lockbox_test_frac)
        samples = _load_samples_for_cv(
            task_name=str(args.task),
            dataset_name=str(args.dataset),
            dataset_path=args.dataset_path,
            n_samples=int(args.n_samples),
            shuffle=bool(args.shuffle),
            seed=int(args.seed),
        )
        if len(samples) < max(10, int(args.k) + 3):
            raise SystemExit(f"Not enough rows for governed splitting: got {len(samples)}")

        neutral = max(0.0, min(1.0, float(args.neutral)))
        strat_values = [
            abs(row.reference_norm - neutral) if args.stratify == "dist" else row.reference_norm
            for row in samples
        ]
        counts = _counts_from_fracs(len(samples), fracs)
        min_dev_pool = 2 if int(args.k) == 1 else max(6, int(args.k))
        if counts["dev_pool"] < min_dev_pool:
            raise SystemExit(
                f"dev_pool too small for k={args.k}: dev_pool={counts['dev_pool']} "
                f"(n_samples={len(samples)}, fractions={fracs})"
            )
        indices = _make_outer_split_indices(
            strat_values,
            bins=int(args.bins),
            seed=int(args.seed),
            target_counts=counts,
        )
        outer_splits = {
            name: [samples[idx].doc_id for idx in indices[name]]
            for name in ("dev_pool", "practical_val", "lockbox_test")
        }
        payload = {
            "generated_at": datetime.now().isoformat(),
            "task": str(args.task),
            "dataset": str(args.dataset),
            "dataset_path": str(args.dataset_path) if args.dataset_path else None,
            "n_samples_requested": int(args.n_samples),
            "n_samples_selected": int(len(samples)),
            "seed": int(args.seed),
            "shuffle": bool(args.shuffle),
            "stratify": str(args.stratify),
            "bins": int(args.bins),
            "neutral": float(neutral),
            "fractions": fracs,
            "counts": {name: int(len(ids)) for name, ids in outer_splits.items()},
            "splits": outer_splits,
        }
        _write_json(outer_split_path, payload)
        print(
            "Wrote outer split "
            f"(dev={len(outer_splits['dev_pool'])}, practical={len(outer_splits['practical_val'])}, "
            f"lockbox={len(outer_splits['lockbox_test'])}): {outer_split_path}"
        )

    min_dev_pool = 2 if int(args.k) == 1 else max(6, int(args.k))
    if len(outer_splits["dev_pool"]) < min_dev_pool:
        raise SystemExit(
            f"Outer split dev_pool has {len(outer_splits['dev_pool'])} docs, insufficient for --k={args.k}"
        )

    _write_json(splits_dir / "dev_pool_ids.json", outer_splits["dev_pool"])
    _write_json(splits_dir / "practical_val_ids.json", outer_splits["practical_val"])
    _write_json(splits_dir / "lockbox_test_ids.json", outer_splits["lockbox_test"])
    _write_json(
        splits_dir / "final_eval_split.json",
        {
            "train": outer_splits["dev_pool"],
            "val": outer_splits["practical_val"],
            "test": outer_splits["lockbox_test"],
        },
    )

    if int(args.k) == 1:
        single_split = _build_single_run_split(
            outer_splits,
            seed=int(args.seed),
            dev_val_frac=float(args.k1_dev_val_frac),
            test_pool=str(args.k1_test_pool),
        )
        split_path = splits_dir / "single_run_split.json"
        _write_json(split_path, single_split)
        split_counts = {name: len(ids) for name, ids in single_split.items()}

        output_dir = cv_dir / "single_run"
        run_cmd = _build_single_run_command(
            args,
            split_path=split_path,
            output_dir=output_dir,
            split_counts=split_counts,
        )
        cmd_path = cv_dir / "single_run_command.sh"
        cmd_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(run_cmd) + "\n", encoding="utf-8")
        cmd_path.chmod(0o755)

        if bool(args.phase1_cache):
            print("Note: --phase1-cache has no effect in k=1 mode (single run, no fold reuse).")
        if int(args.max_parallel_folds) != 1:
            print("Note: --max-parallel-folds is ignored in k=1 mode.")

        _write_json(
            cv_dir / "governed_mode_summary.json",
            {
                "mode": "single_run_k1",
                "task": str(args.task),
                "dataset": str(args.dataset),
                "outer_split_path": str(outer_split_path),
                "single_split_path": str(split_path),
                "single_split_counts": split_counts,
                "k1_test_pool": str(args.k1_test_pool),
                "k1_dev_val_frac": float(args.k1_dev_val_frac),
                "command_path": str(cmd_path),
                "created_at": datetime.now().isoformat(),
            },
        )

        print(f"k=1 mode: train={split_counts['train']} val={split_counts['val']} test={split_counts['test']}")
        print(f"Single-run command: {shlex.join(run_cmd)}")
        print(f"Saved reproducible command: {cmd_path}")
        print(f"Final-eval split template: {splits_dir / 'final_eval_split.json'}")
        if bool(args.dry_run):
            return 0

        completed = subprocess.run(run_cmd, check=False)
        if int(completed.returncode) != 0:
            raise SystemExit(int(completed.returncode))
        return 0

    inner_cmd = _build_inner_cv_command(
        args,
        outer_split_path=outer_split_path,
        dev_pool_size=len(outer_splits["dev_pool"]),
    )
    cmd_path = cv_dir / "inner_cv_command.sh"
    cmd_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(inner_cmd) + "\n", encoding="utf-8")
    cmd_path.chmod(0o755)

    print(f"Inner CV command: {shlex.join(inner_cmd)}")
    print(f"Saved reproducible command: {cmd_path}")
    print(f"Final-eval split template: {splits_dir / 'final_eval_split.json'}")
    if bool(args.dry_run):
        return 0

    completed = subprocess.run(inner_cmd, check=False)
    if int(completed.returncode) != 0:
        raise SystemExit(int(completed.returncode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
