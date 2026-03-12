#!/usr/bin/env python3
"""
True k-fold cross-validation runner for `src.training.run_pipeline`.

This script:
1) Samples documents from a dataset plugin.
2) Builds stratified folds (by score or by distance-from-neutral).
3) For each fold, creates an explicit {train,val,test} doc_id split JSON.
4) Invokes `python -m src.training.run_pipeline` once per fold with
   `--split-ids-path`, so each fold re-optimizes modules on its own train/val.

Notes
-----
- This is "true CV" in the sense that the scorer/summarizers are *re-trained*
  for each fold. It is expensive.
- Split construction (k>=2): per-fold test on one fold; validation carved from
  the remaining train pool.
- `--phase1-cache` is the biggest speedup: it processes the CV document set
  once (batched OPS), then seeds each fold with a Phase 1 checkpoint so train/val
  docs are not re-processed k times.
- Dynamic GPU orchestration can be used, but it is *not fold-parallel-safe*:
    - set `--max-parallel-folds 1`
    - keep servers running across folds (default: `--reuse-dynamic-gpu-servers`)
  Static mode is still an option if you prefer manual server control:
    - start servers once (e.g. `./scripts/start_dual_servers.sh`)
    - pass `--no-dynamic-gpu` in the forwarded pipeline args.
- `--max-parallel-folds` can overlap folds, but remember concurrency budgets are
  not coordinated across processes; lower `--concurrent-requests` if using >1.

Example
-------
./venv/bin/python scripts/run_kfold_cv.py \\
  --task manifesto_rile --dataset manifesto \\
  --cv-output-dir outputs/manifesto_cv_$(date +%Y%m%d_%H%M) \\
  --n-samples 400 --k 5 --stratify dist --bins 10 --seed 42 \\
  -- \\
  --no-dynamic-gpu --port 8000 --genrm-port 8001 \\
  --optimizer gepa --optimizer-budget heavy --max-metric-calls 800 \\
  --honest-chunking --three-layer-honesty
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import pickle
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _quantile_edges(values: Sequence[float], bins: int) -> List[float]:
    if not values:
        return [0.0, 1.0]
    bins = max(2, int(bins))
    xs = sorted(float(v) for v in values)
    n = len(xs)
    edges: List[float] = []
    for i in range(bins + 1):
        q = i / bins
        idx = int(round(q * (n - 1)))
        idx = max(0, min(n - 1, idx))
        edges.append(xs[idx])
    for i in range(1, len(edges)):
        if edges[i] < edges[i - 1]:
            edges[i] = edges[i - 1]
    return edges


def _assign_bin(value: float, edges: Sequence[float]) -> int:
    if not edges or len(edges) < 2:
        return 0
    lo = 0
    hi = len(edges) - 2
    while lo <= hi:
        mid = (lo + hi) // 2
        if value < edges[mid + 1]:
            hi = mid - 1
        else:
            lo = mid + 1
    return max(0, min(len(edges) - 2, lo))


def _make_stratified_folds(
    strat_values: Sequence[float],
    *,
    k: int,
    bins: int,
    seed: int,
) -> List[List[int]]:
    k = max(2, int(k))
    bins = max(2, int(bins))
    edges = _quantile_edges(strat_values, bins=bins)

    by_bin: Dict[int, List[int]] = defaultdict(list)
    for idx, val in enumerate(strat_values):
        by_bin[_assign_bin(float(val), edges)].append(idx)

    rng = random.Random(int(seed))
    for bucket in by_bin.values():
        rng.shuffle(bucket)

    folds: List[List[int]] = [[] for _ in range(k)]
    fold_offset = 0
    for bin_idx in sorted(by_bin.keys()):
        bucket = by_bin[bin_idx]
        if not bucket:
            continue
        for j, idx in enumerate(bucket):
            folds[(fold_offset + j) % k].append(idx)
        fold_offset = (fold_offset + len(bucket)) % k
    for fold in folds:
        fold.sort()
    return folds


def _split_train_val_from_pool(
    pool_indices: Sequence[int],
    *,
    val_frac: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    pool = [int(i) for i in pool_indices]
    if len(pool) < 2:
        raise RuntimeError(
            f"Need at least 2 docs in train pool to build train+val, got {len(pool)}"
        )
    rng = random.Random(int(seed))
    rng.shuffle(pool)
    clipped = max(0.05, min(0.5, float(val_frac)))
    n_val = int(round(float(len(pool)) * clipped))
    n_val = max(1, min(len(pool) - 1, n_val))
    val = sorted(pool[:n_val])
    train = sorted(pool[n_val:])
    return train, val


@dataclass(frozen=True)
class SampleRow:
    doc_id: str
    reference_norm: float
    sample: Any


def _load_samples_for_cv(
    *,
    task_name: str,
    dataset_name: str,
    dataset_path: Optional[str],
    n_samples: int,
    shuffle: bool,
    seed: int,
    task_kwargs: Optional[Dict[str, Any]] = None,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
) -> List[SampleRow]:
    from src.tasks import get_task
    from src.datasets import get_dataset

    task = get_task(task_name, **(task_kwargs or {}))
    dataset = get_dataset(dataset_name, **(dataset_kwargs or {}))

    raw_samples = dataset.load_samples(
        path=dataset_path,
        limit=int(n_samples) if n_samples > 0 else None,
        shuffle=bool(shuffle),
        seed=int(seed),
    )
    rows: List[SampleRow] = []
    for sample in raw_samples:
        doc_id = str(getattr(sample, "doc_id", "") or "").strip()
        ref = _as_float(getattr(sample, "reference_score", None))
        if not doc_id or ref is None:
            continue
        try:
            ref_norm = float(task.normalize_score(ref))
        except Exception:
            # Assume it's already normalized if task normalize fails.
            ref_norm = float(ref)
        ref_norm = max(0.0, min(1.0, ref_norm))
        rows.append(SampleRow(doc_id=doc_id, reference_norm=ref_norm, sample=sample))

    # De-dup by doc_id (keep first occurrence).
    seen: set[str] = set()
    deduped: List[SampleRow] = []
    for row in rows:
        if row.doc_id in seen:
            continue
        seen.add(row.doc_id)
        deduped.append(row)
    return deduped


def _write_split_json(path: Path, *, train: List[str], val: List[str], test: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"train": train, "val": val, "test": test}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_final_stats(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_metrics(final_stats: Dict[str, Any], split: str) -> Dict[str, Any]:
    block = final_stats.get(split, {}) if isinstance(final_stats, dict) else {}
    if not isinstance(block, dict):
        return {}
    out: Dict[str, Any] = {
        "n_evaluated": block.get("n_evaluated"),
        "mae": block.get("mae"),
        "pearson_r": block.get("pearson_r"),
        "spearman_r": block.get("spearman_r"),
        "within_5pct": block.get("within_5pct"),
        "within_10pct": block.get("within_10pct"),
    }
    out["honest_split_metrics"] = block.get("honest_split_metrics")
    out["three_layer_honesty_metrics"] = block.get("three_layer_honesty_metrics")
    return out


def _mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None, None
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    return float(mean), float(math.sqrt(var))


def _nested_get(obj: Any, keys: Sequence[str]) -> Any:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _load_phase1_cache(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "results" in payload and isinstance(payload["results"], dict):
        return payload
    if isinstance(payload, dict):
        return {"version": 1, "built_at": None, "results": payload}
    raise RuntimeError(f"Unexpected Phase 1 cache format at {path}")


def _write_phase1_cache(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(payload, handle)
    tmp_path.replace(path)


def _load_doc_ids_from_json(path: Path, *, key: Optional[str]) -> List[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to load doc-id restriction file {path}: {exc}") from exc

    raw_ids: Any
    if isinstance(payload, list):
        if key:
            raise RuntimeError(
                f"--restrict-doc-ids-key={key!r} was provided, but {path} contains a JSON list (not an object)."
            )
        raw_ids = payload
    elif isinstance(payload, dict):
        if key:
            if key in payload:
                raw_ids = payload.get(key, [])
            elif isinstance(payload.get("splits"), dict) and key in payload.get("splits", {}):
                raw_ids = payload["splits"].get(key, [])
            else:
                available = ", ".join(sorted(str(k) for k in payload.keys()))
                nested = payload.get("splits")
                nested_available = (
                    ", ".join(sorted(str(k) for k in nested.keys()))
                    if isinstance(nested, dict)
                    else ""
                )
                if nested_available:
                    available = f"{available}; splits.* keys: [{nested_available}]"
                raise RuntimeError(
                    f"--restrict-doc-ids-key={key!r} not found in {path}. Available keys: [{available}]"
                )
        elif "doc_ids" in payload:
            raw_ids = payload.get("doc_ids", [])
        elif isinstance(payload.get("splits"), dict) and "doc_ids" in payload.get("splits", {}):
            raw_ids = payload["splits"].get("doc_ids", [])
        elif {"train", "val", "test"}.issubset(set(payload.keys())):
            raw_ids = list(payload.get("train", [])) + list(payload.get("val", [])) + list(payload.get("test", []))
        elif isinstance(payload.get("splits"), dict) and {"train", "val", "test"}.issubset(
            set(payload.get("splits", {}).keys())
        ):
            split_block = payload.get("splits", {})
            raw_ids = list(split_block.get("train", [])) + list(split_block.get("val", [])) + list(
                split_block.get("test", [])
            )
        else:
            available = ", ".join(sorted(str(k) for k in payload.keys()))
            raise RuntimeError(
                f"Could not infer doc-id list from {path}. Provide --restrict-doc-ids-key. Available keys: [{available}]"
            )
    else:
        raise RuntimeError(
            f"Doc-id restriction file must contain a JSON list/object, got {type(payload).__name__}: {path}"
        )

    if not isinstance(raw_ids, list):
        raise RuntimeError(
            f"Doc-id restriction payload must be a JSON list, got {type(raw_ids).__name__} from {path}"
        )

    out: List[str] = []
    seen: Set[str] = set()
    for item in raw_ids:
        if item is None:
            continue
        doc_id = str(item).strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
    return out


def _probe_model_ids_on_port(port: int, *, timeout: float = 2.0) -> Optional[List[str]]:
    url = f"http://localhost:{int(port)}/v1/models"
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=float(timeout)) as resp:
            if int(getattr(resp, "status", 200)) != 200:
                return None
            payload = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError):
        return None
    except Exception:
        return None

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return None
    ids: List[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        if mid is None:
            continue
        ids.append(str(mid))
    return ids or None


def _task_ports_if_dp2_available(primary_port: int, replica_port: int) -> Optional[List[int]]:
    primary_ids = _probe_model_ids_on_port(primary_port, timeout=1.5)
    replica_ids = _probe_model_ids_on_port(replica_port, timeout=1.5)
    if not primary_ids or not replica_ids:
        return None
    primary = primary_ids[0].rstrip("/").split("/")[-1]
    replica = replica_ids[0].rstrip("/").split("/")[-1]
    if primary != replica:
        return None
    return [int(primary_port), int(replica_port)]


def _build_phase1_args_from_forwarded(forwarded: Sequence[str]) -> argparse.Namespace:
    """
    Parse a minimal subset of `src.training.run_pipeline` args needed for Phase 1 processing.

    Unknown args are ignored so this remains compatible with whatever is forwarded
    for the full pipeline.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrent-docs", type=int, default=20)
    parser.add_argument("--concurrent-requests", type=int, default=200)
    parser.add_argument("--max-chunk-chars", type=int, default=4000)
    parser.add_argument("--phase1-score-requests", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--phase1-run-baseline", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--phase1-max-tokens-summary", type=int, default=None)
    parser.add_argument("--phase1-max-tokens-score", type=int, default=None)
    args, _ = parser.parse_known_args(list(forwarded))
    return args


def _create_fold_phase1_checkpoint(
    fold_out: Path,
    *,
    train_ids: List[str],
    val_ids: List[str],
    cache_results: Dict[str, Any],
) -> None:
    """
    Seed fold_out/checkpoints/{phase1_data.pkl,phase1_complete.json} from a global cache.

    This allows per-fold runs to use `--resume` and skip Phase 1 doc processing.
    """
    checkpoints_dir = fold_out / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_results = [cache_results[doc_id] for doc_id in train_ids if doc_id in cache_results]
    val_results = [cache_results[doc_id] for doc_id in val_ids if doc_id in cache_results]

    phase1_payload = {
        "train_results": train_results,
        "val_results": val_results,
        "train_complete": len(train_results) == len(train_ids),
        "val_complete": len(val_results) == len(val_ids),
        "train_total": int(len(train_ids)),
        "val_total": int(len(val_ids)),
        "updated_at": datetime.now().isoformat(),
        "interleaved_last_optimized_count": 0,
    }

    with open(checkpoints_dir / "phase1_data.pkl", "wb") as handle:
        pickle.dump(phase1_payload, handle)

    with open(checkpoints_dir / "phase1_complete.json", "w", encoding="utf-8") as handle:
        json.dump({"train_count": len(train_results), "val_count": len(val_results)}, handle, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run k-fold CV for src.training.run_pipeline.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument(
        "--restrict-doc-ids-path",
        type=Path,
        default=None,
        help=(
            "Optional JSON file restricting which doc_ids are eligible for CV. "
            "Accepts a JSON list, {\"doc_ids\": [...]}, or an object key selected via --restrict-doc-ids-key."
        ),
    )
    parser.add_argument(
        "--restrict-doc-ids-key",
        type=str,
        default=None,
        help="Optional key inside --restrict-doc-ids-path when the file is a JSON object (example: dev_pool).",
    )

    parser.add_argument("--cv-output-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, required=True, help="Total docs to include in CV (sampled from dataset).")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-parallel-folds",
        type=int,
        default=1,
        help=(
            "Run up to N folds concurrently (each fold is a subprocess). "
            "Warning: concurrency limits are not coordinated across processes; "
            "if you set this >1, lower --concurrent-requests accordingly."
        ),
    )
    parser.add_argument(
        "--stratify",
        type=str,
        default="dist",
        choices=["dist", "actual"],
        help="Stratify folds by |actual-neutral| ('dist') or by actual score ('actual').",
    )
    parser.add_argument("--neutral", type=float, default=0.5)
    parser.add_argument(
        "--train-pool-val-frac",
        type=float,
        default=0.15,
        help=(
            "Hold out this fraction of the train pool for validation "
            "(clipped to [0.05, 0.5])."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print fold commands without running them.")
    parser.add_argument("--make-fold-reports", action="store_true", help="Run scripts/report_score_run.py for each fold.")
    parser.add_argument(
        "--make-cv-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate a consolidated CV PDF report with tables and figures across folds.",
    )
    parser.add_argument(
        "--cv-report-split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Which split report to use for the consolidated CV report figures/tables.",
    )
    parser.add_argument(
        "--cv-report-path",
        type=Path,
        default=None,
        help="Optional path for consolidated CV PDF (default: <cv-output-dir>/cv_report.pdf).",
    )
    parser.add_argument(
        "--phase1-cache",
        action="store_true",
        help=(
            "Precompute Phase 1 doc processing once for all CV docs, then seed each fold's "
            "output dir with a Phase 1 checkpoint so folds can skip re-processing train/val docs. "
            "This is the biggest speedup when k>3."
        ),
    )
    parser.add_argument(
        "--phase1-cache-concurrent-docs",
        type=int,
        default=0,
        help="Override forwarded --concurrent-docs for the Phase 1 cache build only (0 = use forwarded/default).",
    )
    parser.add_argument(
        "--phase1-cache-concurrent-requests",
        type=int,
        default=0,
        help="Override forwarded --concurrent-requests for the Phase 1 cache build only (0 = use forwarded/default).",
    )
    parser.add_argument(
        "--phase1-cache-dynamic-gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "When --phase1-cache is enabled and forwarded pipeline args do NOT include --no-dynamic-gpu, "
            "start the GPU orchestrator in the parent process so Phase 1 caching can use task_dp2 (ports 8000+8002). "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--reuse-dynamic-gpu-servers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When forwarded pipeline args use dynamic GPU (default), inject --keep-servers-running so vLLM servers "
            "stay up across folds (requires --max-parallel-folds=1)."
        ),
    )
    parser.add_argument(
        "--shutdown-servers-after-cv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If --reuse-dynamic-gpu-servers is enabled, shut down the orchestrated vLLM servers after CV completes.",
    )
    parser.add_argument(
        "--rebuild-phase1-cache",
        action="store_true",
        help="Rebuild the Phase 1 cache even if it already exists under cv-output-dir.",
    )
    parser.add_argument(
        "pipeline_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to `python -m src.training.run_pipeline` (prefix with '--').",
    )
    args = parser.parse_args()

    if args.k < 2:
        raise SystemExit("--k must be >= 2 for run_kfold_cv.py (use run_governed_kfold_cv.py --k 1 for single-run mode)")
    if int(args.n_samples) == 0:
        raise SystemExit("--n-samples cannot be 0 (use a positive value, or -1 to use all available rows).")
    val_carve_frac = float(args.train_pool_val_frac)

    cv_dir = Path(args.cv_output_dir)
    cv_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = cv_dir / "splits"
    folds_dir = cv_dir / "folds"

    neutral = max(0.0, min(1.0, float(args.neutral)))

    restricted_doc_ids: Optional[List[str]] = None
    restricted_doc_id_set: Optional[Set[str]] = None
    if args.restrict_doc_ids_path:
        restricted_doc_ids = _load_doc_ids_from_json(
            Path(args.restrict_doc_ids_path),
            key=str(args.restrict_doc_ids_key).strip() if args.restrict_doc_ids_key else None,
        )
        if not restricted_doc_ids:
            raise SystemExit(f"--restrict-doc-ids-path produced zero doc_ids: {args.restrict_doc_ids_path}")
        restricted_doc_id_set = set(restricted_doc_ids)

    load_limit = int(args.n_samples) if int(args.n_samples) > 0 and restricted_doc_id_set is None else -1
    samples = _load_samples_for_cv(
        task_name=str(args.task),
        dataset_name=str(args.dataset),
        dataset_path=args.dataset_path,
        n_samples=load_limit,
        shuffle=bool(args.shuffle),
        seed=int(args.seed),
    )

    if restricted_doc_id_set is not None:
        before = len(samples)
        filtered = [row for row in samples if row.doc_id in restricted_doc_id_set]
        present_ids = {row.doc_id for row in filtered}
        missing = len(restricted_doc_id_set.difference(present_ids))
        if int(args.n_samples) > 0:
            filtered = filtered[: int(args.n_samples)]
        samples = filtered
        print(
            f"Applied doc-id restriction ({len(restricted_doc_id_set)} requested): "
            f"{len(samples)}/{before} rows retained; missing_from_dataset={missing}"
        )

    if len(samples) < max(6, args.k):
        raise SystemExit(f"Not enough samples for CV: got {len(samples)} rows")

    strat_values = [
        abs(row.reference_norm - neutral) if args.stratify == "dist" else row.reference_norm
        for row in samples
    ]
    folds = _make_stratified_folds(strat_values, k=int(args.k), bins=int(args.bins), seed=int(args.seed))

    # Remove the leading '--' that argparse includes when forwarding args.
    forwarded = list(args.pipeline_args or [])
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    dynamic_enabled = "--no-dynamic-gpu" not in forwarded
    if dynamic_enabled and int(args.max_parallel_folds) != 1:
        raise SystemExit(
            "Forwarded pipeline args use dynamic GPU mode (default), which is not fold-parallel-safe. "
            "Set --max-parallel-folds=1 or add --no-dynamic-gpu to the forwarded pipeline args."
        )
    reuse_dynamic_servers = bool(dynamic_enabled) and bool(getattr(args, "reuse_dynamic_gpu_servers", True))

    if reuse_dynamic_servers and "--keep-servers-running" not in forwarded:
        forwarded = ["--keep-servers-running", *forwarded]

    parent_orchestrator = None

    # Optional Phase 1 cache build: process each document once (batched),
    # then reuse the resulting DocumentResults across folds.
    phase1_cache_path = cv_dir / "phase1_cache.pkl"
    phase1_cache_results: Optional[Dict[str, Any]] = None
    if args.phase1_cache:
        if phase1_cache_path.exists() and not args.rebuild_phase1_cache:
            cache_payload = _load_phase1_cache(phase1_cache_path)
            phase1_cache_results = cache_payload.get("results", {}) if isinstance(cache_payload, dict) else {}
            if not isinstance(phase1_cache_results, dict):
                phase1_cache_results = {}
            print(f"Loaded Phase 1 cache: {len(phase1_cache_results)}/{len(samples)} docs ({phase1_cache_path})")
        else:
            phase1_cache_results = {}

        missing_samples = [
            row.sample for row in samples if phase1_cache_results is not None and row.doc_id not in phase1_cache_results
        ]
        if missing_samples:
            from src.tasks import get_task
            from src.training.run_pipeline import process_docs

            if dynamic_enabled and args.phase1_cache_dynamic_gpu is not False and parent_orchestrator is None:
                try:
                    from src.core.gpu_orchestrator import GPUOrchestrator

                    parent_orchestrator = GPUOrchestrator()
                    if "--dynamic-gpu-soft-quiesce" in forwarded:
                        parent_orchestrator.config.shared_gpu_hard_quiesce = False
                    asyncio.run(parent_orchestrator.initialize())
                except Exception as exc:
                    parent_orchestrator = None
                    if bool(args.phase1_cache_dynamic_gpu):
                        raise SystemExit(
                            "Failed to start dynamic GPU orchestrator for Phase 1 cache. "
                            "You explicitly requested --phase1-cache-dynamic-gpu, so aborting "
                            f"instead of silently falling back to static mode. Error: {exc}"
                        )
                    print(
                        "WARNING: failed to start dynamic GPU orchestrator for Phase 1 cache; "
                        f"continuing in static mode: {exc}"
                    )

            phase1_args = _build_phase1_args_from_forwarded(forwarded)
            if int(getattr(args, "phase1_cache_concurrent_docs", 0) or 0) > 0:
                phase1_args.concurrent_docs = int(args.phase1_cache_concurrent_docs)
            if int(getattr(args, "phase1_cache_concurrent_requests", 0) or 0) > 0:
                phase1_args.concurrent_requests = int(args.phase1_cache_concurrent_requests)
            task = get_task(str(args.task))
            task_ports = None
            if parent_orchestrator is not None:
                try:
                    task_ports = parent_orchestrator.get_active_task_ports()
                except Exception:
                    task_ports = None
            if task_ports is None:
                task_ports = _task_ports_if_dp2_available(int(phase1_args.port), 8002)

            print(
                f"Building Phase 1 cache for {len(missing_samples)} docs "
                f"(port={phase1_args.port}, concurrent_docs={phase1_args.concurrent_docs}, "
                f"concurrent_requests={phase1_args.concurrent_requests}, max_chunk_chars={phase1_args.max_chunk_chars})"
            )
            results = process_docs(
                missing_samples,
                phase1_args,
                task,
                desc="Phase1 cache",
                task_ports=task_ports,
            )
            added = 0
            for result in results:
                if result is None:
                    continue
                doc_id = getattr(result, "doc_id", None)
                doc_id = str(doc_id).strip() if doc_id is not None else ""
                if not doc_id:
                    continue
                if doc_id in phase1_cache_results:
                    continue
                phase1_cache_results[doc_id] = result
                added += 1
            _write_phase1_cache(
                phase1_cache_path,
                {
                    "version": 1,
                    "built_at": datetime.now().isoformat(),
                    "task": str(args.task),
                    "dataset": str(args.dataset),
                    "results": phase1_cache_results,
                },
            )
            print(f"Wrote Phase 1 cache: added {added} docs ({phase1_cache_path})")

    fold_jobs: List[Dict[str, Any]] = []
    for test_fold in range(int(args.k)):
        train_pool_idx: List[int] = []
        for fold_idx in range(int(args.k)):
            if fold_idx == int(test_fold):
                continue
            train_pool_idx.extend(folds[fold_idx])
        train_idx, val_idx = _split_train_val_from_pool(
            train_pool_idx,
            val_frac=float(val_carve_frac),
            seed=int(args.seed) + (int(test_fold) * 1009),
        )
        train_ids = [samples[i].doc_id for i in train_idx]
        val_ids = [samples[i].doc_id for i in val_idx]
        test_ids = [samples[i].doc_id for i in folds[test_fold]]

        split_path = splits_dir / f"fold_{test_fold}.json"
        _write_split_json(split_path, train=train_ids, val=val_ids, test=test_ids)

        fold_out = folds_dir / f"fold_{test_fold}"
        fold_out.mkdir(parents=True, exist_ok=True)
        fold_log = fold_out / "cv_run.log"

        cmd_forwarded = list(forwarded)
        if phase1_cache_results is not None:
            _create_fold_phase1_checkpoint(
                fold_out,
                train_ids=train_ids,
                val_ids=val_ids,
                cache_results=phase1_cache_results,
            )
            if "--resume" not in cmd_forwarded:
                cmd_forwarded = ["--resume", *cmd_forwarded]

        cmd = [
            sys.executable,
            "-m",
            "src.training.run_pipeline",
            *cmd_forwarded,
            "--task",
            str(args.task),
            "--dataset",
            str(args.dataset),
        ]
        if args.dataset_path:
            cmd.extend(["--dataset-path", str(args.dataset_path)])
        # Fold-specific split args must come last so forwarded args cannot override them.
        cmd.extend(
            [
                "--output-dir",
                str(fold_out),
                "--split-ids-path",
                str(split_path),
                "--train-samples",
                str(len(train_ids)),
                "--val-samples",
                str(len(val_ids)),
                "--test-samples",
                str(len(test_ids)),
            ]
        )

        fold_jobs.append(
            {
                "fold": test_fold,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
                "split_path": split_path,
                "fold_out": fold_out,
                "fold_log": fold_log,
                "cmd": cmd,
            }
        )

    for job in fold_jobs:
        print(
            f"[fold {job['fold']}] train={len(job['train_ids'])} val={len(job['val_ids'])} "
            f"test={len(job['test_ids'])} split={job['split_path']}"
        )
        print("  " + " ".join(job["cmd"]))
    if args.dry_run:
        return 0

    fold_stats: List[Dict[str, Any]] = []
    max_parallel = max(1, int(args.max_parallel_folds))
    pending = list(fold_jobs)
    running: List[Tuple[Dict[str, Any], subprocess.Popen, Any]] = []

    while pending or running:
        while pending and len(running) < max_parallel:
            job = pending.pop(0)
            log_handle = open(job["fold_log"], "w", encoding="utf-8")
            proc = subprocess.Popen(job["cmd"], stdout=log_handle, stderr=subprocess.STDOUT)
            running.append((job, proc, log_handle))

        finished_idx: Optional[int] = None
        for idx, (job, proc, log_handle) in enumerate(running):
            ret = proc.poll()
            if ret is None:
                continue
            finished_idx = idx
            log_handle.close()
            if ret != 0:
                # Terminate remaining folds for faster feedback.
                for other_job, other_proc, other_handle in running:
                    if other_proc.poll() is None:
                        try:
                            other_proc.terminate()
                        except Exception:
                            pass
                    try:
                        other_handle.close()
                    except Exception:
                        pass
                raise SystemExit(f"Fold {job['fold']} failed (exit={ret}). See {job['fold_log']}")

            if args.make_fold_reports:
                report_cmd = [
                    sys.executable,
                    "scripts/report_score_run.py",
                    "--output-dir",
                    str(job["fold_out"]),
                    "--splits",
                    "train",
                    "test",
                ]
                with open(job["fold_out"] / "cv_report.log", "w", encoding="utf-8") as handle:
                    subprocess.run(report_cmd, stdout=handle, stderr=subprocess.STDOUT, check=False)

            final_stats = _load_final_stats(job["fold_out"] / "final_stats.json") or {}
            fold_stats.append(
                {
                    "fold": job["fold"],
                    "paths": {
                        "output_dir": str(job["fold_out"]),
                        "split_json": str(job["split_path"]),
                    },
                    "train": _extract_metrics(final_stats, "train"),
                    "test": _extract_metrics(final_stats, "test"),
                }
            )
            break

        if finished_idx is None:
            import time

            time.sleep(1.0)
            continue

        # Remove finished process entry.
        running.pop(finished_idx)

    summary_path = cv_dir / "cv_summary.json"
    # Aggregate key metrics across folds (mean/std).
    def _collect(keys: Sequence[str]) -> List[float]:
        out: List[float] = []
        for fold in fold_stats:
            val = _nested_get(fold, keys)
            if val is None:
                continue
            try:
                out.append(float(val))
            except Exception:
                continue
        return out

    aggregates = {
        "test": {
            "mae": {"mean_std": _mean_std(_collect(["test", "mae"]))},
            "within_10pct": {"mean_std": _mean_std(_collect(["test", "within_10pct"]))},
            "pearson_r": {"mean_std": _mean_std(_collect(["test", "pearson_r"]))},
            "honest_split": {
                "boundary_mae": {"mean_std": _mean_std(_collect(["test", "honest_split_metrics", "boundary", "mae"]))},
                "evaluation_mae": {"mean_std": _mean_std(_collect(["test", "honest_split_metrics", "evaluation", "mae"]))},
            },
            "three_layer": {
                "chunk_eval_mae": {"mean_std": _mean_std(_collect(["test", "three_layer_honesty_metrics", "chunk", "eval", "mae"]))},
                "summarizer_eval_mae": {"mean_std": _mean_std(_collect(["test", "three_layer_honesty_metrics", "summarizer", "eval", "mae"]))},
                "oracle_eval_mae": {"mean_std": _mean_std(_collect(["test", "three_layer_honesty_metrics", "oracle", "eval", "mae"]))},
                "joint_eval_mae": {"mean_std": _mean_std(_collect(["test", "three_layer_honesty_metrics", "joint_eval", "mae"]))},
            },
        }
    }

    report_pdfs: List[str] = []
    for fold in fold_stats:
        fold_paths = fold.get("paths", {}) if isinstance(fold, dict) else {}
        output_dir = fold_paths.get("output_dir")
        if not output_dir:
            continue
        pdf_path = Path(str(output_dir)) / "score_report.pdf"
        if pdf_path.exists():
            report_pdfs.append(str(pdf_path.resolve()))

    summary_payload = {
        "task": str(args.task),
        "dataset": str(args.dataset),
        "n_samples": len(samples),
        "n_samples_requested": int(args.n_samples),
        "k": int(args.k),
        "stratify": str(args.stratify),
        "bins": int(args.bins),
        "seed": int(args.seed),
        "neutral": neutral,
        "split_policy": "carve_val_from_train_pool",
        "train_pool_val_frac": float(val_carve_frac),
        "doc_id_restriction": (
            {
                "path": str(args.restrict_doc_ids_path),
                "key": str(args.restrict_doc_ids_key) if args.restrict_doc_ids_key else None,
                "n_requested": len(restricted_doc_ids or []),
            }
            if args.restrict_doc_ids_path
            else None
        ),
        "report_pdfs": report_pdfs,
        "aggregates": aggregates,
        "folds": fold_stats,
    }

    cv_report_pdf: Optional[str] = None
    cv_report_path = Path(args.cv_report_path) if args.cv_report_path else (cv_dir / "cv_report.pdf")
    if bool(getattr(args, "make_cv_report", True)):
        cv_report_cmd = [
            sys.executable,
            "scripts/report_cv_summary.py",
            "--cv-dir",
            str(cv_dir),
            "--split",
            str(args.cv_report_split),
            "--pdf-path",
            str(cv_report_path),
        ]
        cv_report_log = cv_dir / "cv_report.log"
        with open(cv_report_log, "w", encoding="utf-8") as handle:
            completed = subprocess.run(cv_report_cmd, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if completed.returncode == 0 and cv_report_path.exists():
            cv_report_pdf = str(cv_report_path.resolve())
            print(f"CV report PDF: {cv_report_pdf}")
        else:
            print(f"WARNING: failed to generate consolidated CV report (see {cv_report_log})")

    summary_payload["cv_report_pdf"] = cv_report_pdf
    summary_payload["cv_report_split"] = str(args.cv_report_split)
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"CV summary (absolute): {summary_path.resolve()}")
    if report_pdfs:
        print("Fold report PDFs:")
        for pdf_path in report_pdfs:
            print(f"  {pdf_path}")

    if bool(getattr(args, "shutdown_servers_after_cv", True)) and (
        parent_orchestrator is not None or reuse_dynamic_servers
    ):
        try:
            if parent_orchestrator is not None:
                asyncio.run(parent_orchestrator.shutdown())
            else:
                from src.core.gpu_orchestrator import GPUOrchestrator

                asyncio.run(GPUOrchestrator().shutdown())
        except Exception as exc:
            print(f"WARNING: failed to shutdown GPU orchestrator servers: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
