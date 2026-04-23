#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Sequence

for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_key, "1")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.autotune_probe_cache import AUTOTUNE_PROBE_CACHE_DIR_ENV
from src.ctreepo.sim.core.markov_neural_operator_baselines import (
    FNOCountSketch,
    HAS_NEURAL_OPERATOR,
    _fixed_fused_training_batch_forward,
    _pack_tree_work_items,
    _prepare_fno_count_docs,
    _tree_work_item_from_doc,
    train_fno_tree,
)
from src.tree.markov_boundary_honesty_simulation import _make_transition_matrices
from src.tree.markov_changepoint_honesty_simulation import (
    MarkovChangepointConfig,
    generate_changepoint_docs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile Markov fixed-fused autotune cold/warm runs with persisted probe cache."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-docs", type=int, default=1000)
    parser.add_argument("--val-docs", type=int, default=128)
    parser.add_argument("--min-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--min-segments", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=4)
    parser.add_argument("--leaf-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--root-weight", type=float, default=1.0)
    parser.add_argument("--c1-weight", type=float, default=0.0)
    parser.add_argument("--c2-weight", type=float, default=1.0)
    parser.add_argument("--c3-weight", type=float, default=0.0)
    parser.add_argument("--phi-compose-weight", type=float, default=1.0)
    parser.add_argument("--phi-contrastive-weight", type=float, default=0.25)
    parser.add_argument(
        "--leaf-supervision-kind",
        type=str,
        default="full_sketch",
        choices=("count_only", "full_sketch"),
    )
    parser.add_argument(
        "--internal-supervision-kind",
        type=str,
        default="count_only",
        choices=("none", "count_only", "full_sketch"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--n-regimes", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--fno-width", type=int, default=16)
    parser.add_argument("--fno-n-modes", type=int, default=8)
    parser.add_argument("--fno-n-layers", type=int, default=2)
    parser.add_argument("--theorem-feature-dim", type=int, default=16)
    parser.add_argument("--theorem-feature-hidden-dim", type=int, default=32)
    parser.add_argument("--screen-doc-limit", type=int, default=32)
    parser.add_argument("--exact-doc-limit", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--pack-mode",
        type=str,
        default="fixed_fused",
        choices=("fixed_fused", "structure_bucket"),
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=("cold_autotune", "warm_autotune", "no_autotune"),
        help="Subset of cases to run: cold_autotune warm_autotune no_autotune.",
    )
    parser.add_argument("--trace", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trace-docs", type=int, default=64)
    return parser.parse_args()


def _default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return Path("outputs") / f"markov_fixed_fused_autotune_profile_{stamp}"


def _resolve_device(raw: str) -> torch.device:
    requested = str(raw or "").strip().lower()
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_docs(args: argparse.Namespace) -> tuple:
    generator_config = MarkovChangepointConfig(
        n_regimes=int(args.n_regimes),
        vocab_size=int(args.vocab_size),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        min_segments=int(args.min_segments),
        max_segments=int(args.max_segments),
        min_seg_len=max(2, int(args.min_tokens) // max(1, int(args.max_segments))),
        max_seg_len=max(2, int(args.max_tokens)),
        fixed_leaf_tokens=int(args.leaf_tokens),
        train_docs=int(args.train_docs),
        test_docs=int(args.val_docs),
        seed=int(args.seed),
    )
    rng = np.random.default_rng(int(args.seed))
    transitions = _make_transition_matrices(
        n_classes=int(args.n_regimes),
        vocab_size=int(args.vocab_size),
        log_std=1.25,
        sinkhorn_iters=30,
        rng=rng,
    )
    docs = generate_changepoint_docs(generator_config, transitions=transitions)
    fno_docs = _prepare_fno_count_docs(docs, leaf_tokens=int(args.leaf_tokens))
    train_docs = tuple(fno_docs[: int(args.train_docs)])
    val_docs = tuple(fno_docs[int(args.train_docs) : int(args.train_docs) + int(args.val_docs)])
    return train_docs, val_docs


def _build_model(args: argparse.Namespace, device: torch.device) -> FNOCountSketch:
    fiber_dim = max(1, int(args.theorem_feature_dim) - 1)
    model = FNOCountSketch(
        vocab_size=int(args.vocab_size),
        leaf_tokens=int(args.leaf_tokens),
        state_dim=int(args.state_dim),
        hidden_dim=int(args.hidden_dim),
        target_scale=float(max(8, args.max_tokens)),
        n_regimes=int(args.n_regimes),
        fno_width=int(args.fno_width),
        fno_n_modes=int(args.fno_n_modes),
        fno_n_layers=int(args.fno_n_layers),
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_surface_mode="factorized_score_fiber",
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_feature_dim=int(args.theorem_feature_dim),
        theorem_feature_hidden_dim=int(args.theorem_feature_hidden_dim),
        theorem_score_dim=1,
        theorem_fiber_dim=int(fiber_dim),
        tree_model_version="v2",
    )
    return model.to(device=device)


def _read_proc_status_kb() -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                text = value.strip()
                if key == "Threads":
                    out["Threads"] = int(text or "0")
                    continue
                if text.endswith("kB"):
                    number = text[:-2].strip()
                    if number.isdigit():
                        out[str(key)] = int(number)
    except OSError:
        return out
    return out


def _gpu_memory_snapshot(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    try:
        return {
            "gpu_allocated_gb": float(
                torch.cuda.memory_allocated(device=device) / float(1024 ** 3)
            ),
            "gpu_reserved_gb": float(
                torch.cuda.memory_reserved(device=device) / float(1024 ** 3)
            ),
            "gpu_max_reserved_gb": float(
                torch.cuda.max_memory_reserved(device=device) / float(1024 ** 3)
            ),
        }
    except Exception:
        return {}


def _snapshot_memory(
    *,
    label: str,
    device: torch.device,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    status = _read_proc_status_kb()
    snapshot: Dict[str, Any] = {
        "label": str(label),
        "t_wall_s": float(time.perf_counter()),
        "rss_gb": float(status.get("VmRSS", 0) / float(1024 ** 2)),
        "rss_anon_gb": float(status.get("RssAnon", 0) / float(1024 ** 2)),
        "vm_data_gb": float(status.get("VmData", 0) / float(1024 ** 2)),
        "vm_hwm_gb": float(status.get("VmHWM", 0) / float(1024 ** 2)),
        "threads": int(status.get("Threads", 0)),
    }
    snapshot.update(_gpu_memory_snapshot(device))
    if extra:
        snapshot.update({str(key): value for key, value in dict(extra).items()})
    return snapshot


def _state_dict_tensor_megabytes(state: Mapping[str, Any] | None) -> float:
    total_bytes = 0
    for value in dict(state or {}).values():
        if hasattr(value, "numel") and hasattr(value, "element_size"):
            total_bytes += int(value.numel()) * int(value.element_size())
    return float(total_bytes / float(1024 ** 2))


def _cleanup_runtime_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    try:
        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except Exception:
        pass


def _summarize_train_result(result: Mapping[str, Any], wall_clock_s: float) -> Dict[str, Any]:
    return {
        "wall_clock_s": float(wall_clock_s),
        "train": dict(result.get("train", {}) or {}),
        "val": dict(result.get("val", {}) or {}),
        "best_epoch": int(result.get("best_epoch", 0) or 0),
        "epochs_completed": int(result.get("epochs_completed", 0) or 0),
        "selection_mode": str(result.get("selection_mode", "") or ""),
        "selection_metric_name": str(result.get("selection_metric_name", "") or ""),
        "best_val_mae": float(result.get("best_val_mae", float("nan"))),
        "timing_breakdown": dict(result.get("timing_breakdown", {}) or {}),
        "batching_metrics": dict(result.get("batching_metrics", {}) or {}),
        "autotuned_batch_budgets": dict(result.get("autotuned_batch_budgets", {}) or {}),
        "autotune_probe_profile": dict(result.get("autotune_probe_profile", {}) or {}),
        "best_model_state_mb": float(
            _state_dict_tensor_megabytes(result.get("best_model_state"))
        ),
    }


def _run_case(
    *,
    label: str,
    args: argparse.Namespace,
    device: torch.device,
    train_docs: Sequence[Any],
    val_docs: Sequence[Any],
    autotune: bool,
    memory_timeline: list[Dict[str, Any]],
) -> Dict[str, Any]:
    case_probe_events: list[Dict[str, Any]] = []
    memory_timeline.append(
        _snapshot_memory(
            label=f"{label}:start",
            device=device,
            extra={"tree_batch_autotune": bool(autotune)},
        )
    )
    model = _build_model(args, device)
    memory_timeline.append(
        _snapshot_memory(label=f"{label}:after_model", device=device)
    )

    def _case_memory_probe(event: str, payload: Mapping[str, Any]) -> None:
        case_probe_events.append(
            _snapshot_memory(
                label=f"{label}:probe:{str(event)}",
                device=device,
                extra=payload,
            )
        )

    start_s = time.perf_counter()
    result = train_fno_tree(
        model=model,
        train_docs=train_docs,
        val_docs=val_docs,
        device=device,
        n_epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        c1_weight=float(args.c1_weight),
        c2_weight=float(args.c2_weight),
        c3_weight=float(args.c3_weight),
        root_weight=float(args.root_weight),
        phi_compose_weight=float(args.phi_compose_weight),
        phi_contrastive_weight=float(args.phi_contrastive_weight),
        leaf_supervision_kind=str(args.leaf_supervision_kind),
        internal_supervision_kind=str(args.internal_supervision_kind),
        checkpoint_metric="val_root_mae",
        tree_training_schedule="single_stage",
        tree_batch_pack_mode=str(args.pack_mode),
        tree_batch_autotune=bool(autotune),
        tree_stage1_screen_doc_limit=int(args.screen_doc_limit),
        tree_stage1_final_exact_doc_limit=int(args.exact_doc_limit),
        exact_metric_selection_doc_limit=int(args.exact_doc_limit),
        exact_metric_final_doc_limit=int(args.exact_doc_limit),
        memory_probe=_case_memory_probe,
        seed=int(args.seed),
    )
    wall_clock_s = time.perf_counter() - start_s
    memory_timeline.append(
        _snapshot_memory(label=f"{label}:after_train", device=device)
    )
    summary = _summarize_train_result(result, wall_clock_s=wall_clock_s)
    summary["label"] = str(label)
    summary["tree_batch_autotune"] = bool(autotune)
    summary["tree_batch_pack_mode"] = str(args.pack_mode)
    summary["loss_weights"] = {
        "root_weight": float(args.root_weight),
        "c1_weight": float(args.c1_weight),
        "c2_weight": float(args.c2_weight),
        "c3_weight": float(args.c3_weight),
        "phi_compose_weight": float(args.phi_compose_weight),
        "phi_contrastive_weight": float(args.phi_contrastive_weight),
    }
    summary["supervision_kinds"] = {
        "leaf_supervision_kind": str(args.leaf_supervision_kind),
        "internal_supervision_kind": str(args.internal_supervision_kind),
    }
    summary["memory_probe_events"] = list(case_probe_events)
    summary["memory_probe_event_count"] = int(len(case_probe_events))
    summary["memory_after_train"] = _snapshot_memory(
        label=f"{label}:summary", device=device
    )
    del result
    del model
    _cleanup_runtime_memory(device)
    post_cleanup_snapshot = _snapshot_memory(
        label=f"{label}:post_cleanup",
        device=device,
    )
    summary["memory_after_cleanup"] = post_cleanup_snapshot
    memory_timeline.append(post_cleanup_snapshot)
    return summary


def _capture_trace(
    *,
    args: argparse.Namespace,
    device: torch.device,
    docs: Sequence[Any],
    output_dir: Path,
) -> Dict[str, Any]:
    if not bool(args.trace):
        return {}
    if str(args.pack_mode).strip().lower() != "fixed_fused":
        return {
            "enabled": True,
            "skipped": True,
            "reason": "trace_only_supported_for_fixed_fused",
            "pack_mode": str(args.pack_mode),
        }
    model = _build_model(args, device)
    trace_docs = list(docs[: max(1, int(args.trace_docs))])
    if not trace_docs:
        return {"enabled": True, "skipped": True, "reason": "no_docs"}
    packed_items = [
        _tree_work_item_from_doc(
            doc,
            doc_index=int(idx),
            work_kind="full_tree",
            collect_leaf=True,
            collect_c2=True,
            collect_c3=True,
            root_only_supervision=True,
        )
        for idx, doc in enumerate(trace_docs)
    ]
    packed_batch = _pack_tree_work_items(
        packed_items,
        max_docs=len(packed_items),
        max_total_leaf_tokens=0,
        max_total_nodes=0,
        max_total_merge_ops=0,
        bucket_docs_cap_by_n_leaves=None,
    )[0]
    work_lookup = {
        int(idx): {
            "doc": doc,
            "root_only_supervision": True,
            "doc_sequence_supervision": False,
            "doc_sequence_loss": torch.zeros((), device=device, dtype=torch.float32),
            "collect_leaf": True,
            "collect_c2": True,
            "collect_c3": True,
            "leaf_audit_indices": None,
            "c3_audit_indices": None,
        }
        for idx, doc in enumerate(trace_docs)
    }
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    trace_path = output_dir / "fixed_fused_batch_trace.json"
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        model.zero_grad(set_to_none=True)
        loss = _fixed_fused_training_batch_forward(
            model,
            packed_batch,
            work_lookup=work_lookup,
            device=device,
            root_weight=1.0,
            c1_weight=1.0,
            c2_weight=1.0,
            c3_weight=0.0,
            phi_compose_weight=1.0,
            leaf_supervision_kind="full_sketch",
            internal_supervision_kind="full_sketch",
            defer_contrastive=True,
        )["batch_loss"]
        if bool(getattr(loss, "requires_grad", False)):
            loss.backward()
    profiler.export_chrome_trace(str(trace_path))
    return {
        "enabled": True,
        "skipped": False,
        "trace_path": str(trace_path),
        "trace_docs": int(len(trace_docs)),
    }


def main() -> None:
    args = _parse_args()
    if not HAS_NEURAL_OPERATOR:
        raise SystemExit("neuraloperator is not installed; cannot profile FNO fixed-fused autotune.")

    torch_threads = max(1, int(args.torch_threads))
    torch_interop_threads = max(1, int(args.torch_interop_threads))
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(torch_interop_threads)
    except RuntimeError:
        pass

    output_dir = Path(args.output_dir or _default_output_dir()).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "probe_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for cache_file in cache_dir.glob("*.json"):
        cache_file.unlink()
    device = _resolve_device(args.device)
    memory_timeline: list[Dict[str, Any]] = [
        _snapshot_memory(label="main:start", device=device)
    ]
    train_docs, val_docs = _make_docs(args)
    memory_timeline.append(
        _snapshot_memory(
            label="main:after_docs",
            device=device,
            extra={
                "n_train_docs": int(len(train_docs)),
                "n_val_docs": int(len(val_docs)),
            },
        )
    )

    old_cache_env = os.environ.get(AUTOTUNE_PROBE_CACHE_DIR_ENV)
    os.environ[AUTOTUNE_PROBE_CACHE_DIR_ENV] = str(cache_dir)
    selected_cases = {
        str(name).strip().lower()
        for name in tuple(args.cases or ())
        if str(name).strip()
    }
    valid_cases = {"cold_autotune", "warm_autotune", "no_autotune"}
    invalid_cases = sorted(selected_cases - valid_cases)
    if invalid_cases:
        raise SystemExit(f"invalid --cases values: {invalid_cases}")
    try:
        runs: Dict[str, Dict[str, Any]] = {}
        if "cold_autotune" in selected_cases:
            runs["cold_autotune"] = _run_case(
                label="cold_autotune",
                args=args,
                device=device,
                train_docs=train_docs,
                val_docs=val_docs,
                autotune=True,
                memory_timeline=memory_timeline,
            )
        if "warm_autotune" in selected_cases:
            runs["warm_autotune"] = _run_case(
                label="warm_autotune",
                args=args,
                device=device,
                train_docs=train_docs,
                val_docs=val_docs,
                autotune=True,
                memory_timeline=memory_timeline,
            )
        if "no_autotune" in selected_cases:
            runs["no_autotune"] = _run_case(
                label="no_autotune",
                args=args,
                device=device,
                train_docs=train_docs,
                val_docs=val_docs,
                autotune=False,
                memory_timeline=memory_timeline,
            )
        trace_summary = _capture_trace(
            args=args,
            device=device,
            docs=train_docs,
            output_dir=output_dir,
        )
    finally:
        if old_cache_env is None:
            os.environ.pop(AUTOTUNE_PROBE_CACHE_DIR_ENV, None)
        else:
            os.environ[AUTOTUNE_PROBE_CACHE_DIR_ENV] = old_cache_env

    report = {
        "device": str(device),
        "cache_dir": str(cache_dir),
        "config": {
            "train_docs": int(args.train_docs),
            "val_docs": int(args.val_docs),
            "min_tokens": int(args.min_tokens),
            "max_tokens": int(args.max_tokens),
            "min_segments": int(args.min_segments),
            "max_segments": int(args.max_segments),
            "leaf_tokens": int(args.leaf_tokens),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "pack_mode": str(args.pack_mode),
            "torch_threads": int(torch_threads),
            "torch_interop_threads": int(torch_interop_threads),
            "omp_num_threads": str(os.getenv("OMP_NUM_THREADS", "")),
            "mkl_num_threads": str(os.getenv("MKL_NUM_THREADS", "")),
        },
        "runs": runs,
        "trace": trace_summary,
        "memory_timeline": memory_timeline,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary_json": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
