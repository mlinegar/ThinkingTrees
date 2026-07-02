#!/usr/bin/env python3
"""Aggregate current HLL FNO/register-recovery grids into one report."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
UNIFIED_SRC = REPO_ROOT / "parallel" / "unified_g_v1" / "src"
TREEPO_SRC = REPO_ROOT / "treepo" / "src"
DEFAULT_ROOTS = (
    Path("outputs/hll_recoverability_grid_20260429_174209"),
    Path("outputs/hll_many_gpu_f_g_probe_20260429_190821"),
    Path("outputs/hll_leaf_grid_fgfg_parallel_20260429_193019"),
    Path("outputs/hll_known_f_g_then_f_grid_20260429_204832"),
)
FIXED_MASS_SMOKE_ROOT = Path("outputs/hll_fixed_mass_cpu_smoke_20260429_101215")
FNO_ARG_DEFAULTS = {
    "precision": "8",
    "universe_size": "512",
    "min_tokens": "128",
    "max_tokens": "128",
    "zipf_alphas": "0.8,1.0,1.2",
    "seed": "0",
}
BASELINE_PRECISIONS = (4, 8, 9, 16)
CONTEXT_BASELINE_PRECISIONS = (8, 9, 16)
REQUESTED_BASELINE_PRECISIONS = (4, 8, 9, 16, 32)
MAX_MATERIALIZED_BASELINE_PRECISION = 16
_FSTAR_CACHE: dict[tuple[int, int, int, int, int, int, int, int, tuple[float, ...]], dict[str, float]] = {}
CANONICAL_SERIES_LABEL = r"learn $f_\theta,g_\theta$ (exact)"
PRIMARY_PLOT_SERIES = (
    r"classical HLL $f^\star \circ g^\star$",
    r"fixed $f^\star$: learn $g_\theta$ (exact)",
    r"fixed $f^\star$: learn $g_\theta/f_\theta$ (exact)",
    r"fixed $f^\star$: rollout $g_\theta/f_\theta$ (dense)",
    r"learn $f_\theta,g_\theta$ (exact)",
    r"learn $f_\theta,g_\theta$ (exact, id init)",
    r"learn $f_\theta,g_\theta$ + residual $f_\theta$",
    r"learn $f_\theta,g_\theta$ (rollout dense)",
    r"learn $f_\theta,g_\theta$ (rollout root)",
    r"learn $f_\theta,g_\theta$ (rollout sampled)",
    r"learn $f_\theta,g_\theta$ (rollout R0)",
    r"learn $f_\theta,g_\theta$ (rollout R50)",
    r"learn $f_\theta,g_\theta$ (rollout R100)",
    r"probe: $f/g/f/g$",
    r"probe: $g/f/g/f/g/f$",
    r"probe: $g$ state+scalar",
    r"probe: rollout $f/g/f/g$ root",
)
CONTEXT_RATIO_SERIES = (
    r"fixed $f^\star$: learn $g_\theta$ (exact)",
    r"fixed $f^\star$: rollout $g_\theta/f_\theta$ (dense)",
    r"learn $f_\theta,g_\theta$ (exact)",
    r"learn $f_\theta,g_\theta$ + residual $f_\theta$",
    r"learn $f_\theta,g_\theta$ (rollout dense)",
    r"learn $f_\theta,g_\theta$ (rollout root)",
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_roots() -> list[Path]:
    roots = list(DEFAULT_ROOTS)
    roots.extend(sorted((REPO_ROOT / "outputs").glob("hll_canonical_observation_grid_*")))
    return roots


def _read_one_csv(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return dict(rows[0]) if rows else None


def _read_treepo_hll_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, Mapping):
        return []
    raw_rows = payload.get("rows", [])
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in raw_rows:
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        if "model_kind" not in row and "lean_merge_adapter" not in row:
            continue
        row.setdefault("target_kind", "hll_register_space")
        row.setdefault("source_family", "treepo_hll_merge_learning")
        row.setdefault("status", "completed")
        row.setdefault("schedule", "treepo_balanced")
        row.setdefault("root_rel_mae", row.get("learned_relative_rmse", ""))
        rows.append(row)
    return rows


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _fmt(value: Any, digits: int = 4) -> str:
    out = _float(value)
    if not math.isfinite(out):
        return ""
    return f"{out:.{digits}g}"


def _int(value: Any, default: int | None = None) -> int | None:
    out = _float(value)
    if not math.isfinite(out):
        return default
    return int(round(out))


def _fmt_int(value: Any) -> str:
    out = _int(value)
    return "" if out is None else str(out)


def _hll_register_count(precision: int) -> int:
    return 1 << int(precision)


def _hll_register_bits(precision: int, *, hash_bits: int = 64) -> int:
    remaining_bits = int(hash_bits) - int(precision)
    return int(math.ceil(math.log2(remaining_bits + 1)))


def _hll_state_bits(precision: int, *, hash_bits: int = 64) -> int:
    return int(_hll_register_count(int(precision)) * _hll_register_bits(int(precision), hash_bits=hash_bits))


def _nearest_hll_precision_for_bits(target_bits: int, *, hash_bits: int = 64) -> int:
    candidates = range(4, int(hash_bits) - 1)
    return min(candidates, key=lambda p: abs(_hll_state_bits(p, hash_bits=hash_bits) - int(target_bits)))


def _flag_value(command: Sequence[str], flag: str, default: str = "") -> str:
    try:
        idx = list(command).index(flag)
    except ValueError:
        return default
    if idx + 1 >= len(command):
        return default
    return str(command[idx + 1])


def _csv_floats(value: Any) -> tuple[float, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    out: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            return ()
    return tuple(out)


def _bool_flag(command: Sequence[str], true_flag: str, false_flag: str, default: str = "") -> str:
    if true_flag in command:
        return "True"
    if false_flag in command:
        return "False"
    return default


def _parse_observation(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return str(value.get("design_id", ""))
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("{"):
        try:
            payload = ast.literal_eval(text)
        except Exception:
            return text
        if isinstance(payload, Mapping):
            return str(payload.get("design_id", text))
    return text


def _summary_path(output_dir: Path) -> Path:
    direct = output_dir / "summary.csv"
    if direct.exists():
        return direct
    nested = output_dir / "hll" / "summary.csv"
    if nested.exists():
        return nested
    return direct


def _read_launcher_command(output_dir: Path) -> list[str]:
    manifest = output_dir / "launcher" / "manifest.json"
    if not manifest.exists():
        return []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return []
    command = payload.get("command", []) if isinstance(payload, Mapping) else []
    return [str(item) for item in command] if isinstance(command, list) else []


def _load_manifest_cells(root: Path) -> list[dict[str, Any]]:
    manifest = root / "manifest.json"
    if not manifest.exists():
        return []
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        out: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, Mapping):
                out.append(dict(item))
        return out
    if isinstance(payload, Mapping):
        cells = payload.get("cells", [])
        if isinstance(cells, list):
            return [dict(item) for item in cells if isinstance(item, Mapping)]
    return []


def _single_target(command: Sequence[str]) -> str:
    targets = _flag_value(command, "--targets")
    parts = [part.strip() for part in targets.split(",") if part.strip()]
    return parts[0] if len(parts) == 1 else targets


def _seed_command_fields(command: Sequence[str]) -> dict[str, Any]:
    return {
        "target_kind": _single_target(command),
        "schedule": _flag_value(command, "--schedule"),
        "n_leaves": _flag_value(command, "--n-leaves"),
        "objective_mode": _flag_value(command, "--objective-mode"),
        "observation": _flag_value(command, "--oracle-observation-design"),
        "sampled_node_rate": _flag_value(command, "--sampled-node-rate"),
        "readout_arch": _flag_value(command, "--readout-arch"),
        "identity_residual_init": _bool_flag(command, "--identity-residual-init", "--no-identity-residual-init"),
        "local_law_weight": _flag_value(command, "--local-law-weight"),
        "objective_loss_weight": _flag_value(command, "--objective-loss-weight"),
        "state_loss_weight": _flag_value(command, "--state-loss-weight"),
        "exact_state_anchor_weight": _flag_value(command, "--exact-state-anchor-weight"),
        "epochs_per_stage": _flag_value(command, "--epochs"),
        "n_train": _flag_value(command, "--n-train"),
        "n_val": _flag_value(command, "--n-val"),
        "precision": _flag_value(command, "--precision", FNO_ARG_DEFAULTS["precision"]),
        "universe_size": _flag_value(command, "--universe-size", FNO_ARG_DEFAULTS["universe_size"]),
        "min_tokens": _flag_value(command, "--min-tokens", FNO_ARG_DEFAULTS["min_tokens"]),
        "max_tokens": _flag_value(command, "--max-tokens", FNO_ARG_DEFAULTS["max_tokens"]),
        "zipf_alphas": _flag_value(command, "--zipf-alphas", FNO_ARG_DEFAULTS["zipf_alphas"]),
        "seed": _flag_value(command, "--seed", FNO_ARG_DEFAULTS["seed"]),
    }


def _is_hll_row(row: Mapping[str, Any]) -> bool:
    target = str(row.get("target_kind", ""))
    return target == "hll_register_space" or "hll_register_space" in target.split(",")


def _import_hll_baseline_tools() -> tuple[Any, Any, Any, Any] | None:
    for path in (REPO_ROOT, TREEPO_SRC, UNIFIED_SRC):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        from src.tree.hll import HLLConfig, HyperLogLogSketch
        from unified_g_v1.sketch.classical_parity import ClassicalHLLParityConfig, generate_documents
    except Exception as exc:
        print(f"warning: could not import HLL baseline tools: {exc}", file=sys.stderr)
        return None
    return HLLConfig, HyperLogLogSketch, ClassicalHLLParityConfig, generate_documents


def _fstar_gstar_analytic_baseline(row: Mapping[str, Any]) -> dict[str, float]:
    if not _is_hll_row(row):
        return {}
    precision = _int(row.get("precision"), _int(FNO_ARG_DEFAULTS["precision"]))
    n_leaves = _int(row.get("n_leaves"))
    n_train = _int(row.get("n_train"))
    n_val = _int(row.get("n_val"))
    seed = _int(row.get("seed"), _int(FNO_ARG_DEFAULTS["seed"]))
    universe_size = _int(row.get("universe_size"), _int(FNO_ARG_DEFAULTS["universe_size"]))
    min_tokens = _int(row.get("min_tokens"), _int(FNO_ARG_DEFAULTS["min_tokens"]))
    max_tokens = _int(row.get("max_tokens"), _int(FNO_ARG_DEFAULTS["max_tokens"]))
    zipf_alphas = _csv_floats(row.get("zipf_alphas")) or _csv_floats(FNO_ARG_DEFAULTS["zipf_alphas"])
    if None in (precision, n_leaves, n_train, n_val, seed, universe_size, min_tokens, max_tokens):
        return {}
    assert precision is not None
    assert n_leaves is not None
    assert n_train is not None
    assert n_val is not None
    assert seed is not None
    assert universe_size is not None
    assert min_tokens is not None
    assert max_tokens is not None
    key = (
        int(precision),
        int(n_leaves),
        int(n_train),
        int(n_val),
        int(seed),
        int(universe_size),
        int(min_tokens),
        int(max_tokens),
        tuple(float(x) for x in zipf_alphas),
    )
    if key in _FSTAR_CACHE:
        return _FSTAR_CACHE[key]
    tools = _import_hll_baseline_tools()
    if tools is None:
        return {}
    HLLConfig, HyperLogLogSketch, ClassicalHLLParityConfig, generate_documents = tools
    cfg = ClassicalHLLParityConfig(
        precision=int(precision),
        n_leaves=int(n_leaves),
        leaf_size=None,
        schedule="balanced",
        backend="native",
        n_val=int(n_train) + int(n_val),
        seed=int(seed),
        universe_size=int(universe_size),
        min_tokens=int(min_tokens),
        max_tokens=int(max_tokens),
        zipf_alphas=tuple(float(x) for x in zipf_alphas),
        oracle_kind="analytic",
    )
    docs = generate_documents(cfg)
    val_docs = docs[int(n_train) : int(n_train) + int(n_val)]
    hll_cfg = HLLConfig(precision=int(precision), hash_bits=64)
    root_abs_err: list[float] = []
    root_rel_err: list[float] = []
    leaf_abs_err: list[float] = []
    leaf_rel_err: list[float] = []
    for leaves, truth, flat in val_docs:
        est = float(HyperLogLogSketch.from_tokens(hll_cfg, list(flat)).estimate())
        err = abs(est - float(truth))
        root_abs_err.append(err)
        root_rel_err.append(err / max(1.0, abs(float(truth))))
        for leaf in leaves:
            leaf_truth = float(len(set(leaf)))
            leaf_est = float(HyperLogLogSketch.from_tokens(hll_cfg, list(leaf)).estimate())
            leaf_err = abs(leaf_est - leaf_truth)
            leaf_abs_err.append(leaf_err)
            leaf_rel_err.append(leaf_err / max(1.0, abs(leaf_truth)))
    result = {
        "fstar_gstar_analytic_mae": float(sum(root_abs_err) / len(root_abs_err)) if root_abs_err else float("nan"),
        "fstar_gstar_analytic_rel_mae": float(sum(root_rel_err) / len(root_rel_err)) if root_rel_err else float("nan"),
        "fstar_gstar_root_analytic_mae": float(sum(root_abs_err) / len(root_abs_err)) if root_abs_err else float("nan"),
        "fstar_gstar_root_analytic_rel_mae": float(sum(root_rel_err) / len(root_rel_err)) if root_rel_err else float("nan"),
        "fstar_gstar_leaf_analytic_mae": float(sum(leaf_abs_err) / len(leaf_abs_err)) if leaf_abs_err else float("nan"),
        "fstar_gstar_leaf_analytic_rel_mae": float(sum(leaf_rel_err) / len(leaf_rel_err)) if leaf_rel_err else float("nan"),
    }
    _FSTAR_CACHE[key] = result
    return result


def _annotate_fstar_gstar(row: dict[str, Any]) -> dict[str, Any]:
    if not _is_hll_row(row):
        return row
    readout_kind = str(row.get("readout_kind", ""))
    if not readout_kind:
        row["readout_kind"] = "hll_reference"
        readout_kind = "hll_reference"
    if readout_kind == "hll_reference":
        row["fstar_gstar_target_mae"] = 0.0
        row["fstar_gstar_target_rel_mae"] = 0.0
        root_mae = _float(row.get("root_mae"))
        root_rel = _float(row.get("root_rel_mae"))
        if math.isfinite(root_mae):
            row["gap_to_fstar_gstar_target_mae"] = root_mae
        if math.isfinite(root_rel):
            row["gap_to_fstar_gstar_target_rel_mae"] = root_rel
    row.update(_fstar_gstar_analytic_baseline(row))
    precision = _int(row.get("precision"))
    if precision is not None:
        row["hll_register_count"] = _hll_register_count(int(precision))
        row["hll_state_bits"] = _hll_state_bits(int(precision))
    state_dim = _int(row.get("state_dim"))
    if state_dim is not None:
        row["neural_state_fp16_bits"] = int(state_dim) * 16
        row["neural_state_fp32_bits"] = int(state_dim) * 32
        if int(state_dim) > 0 and int(state_dim) & (int(state_dim) - 1) == 0:
            row["state_size_matched_hll_precision"] = int(math.log2(int(state_dim)))
        row["fp16_memory_matched_hll_precision"] = _nearest_hll_precision_for_bits(int(state_dim) * 16)
        row["fp32_memory_matched_hll_precision"] = _nearest_hll_precision_for_bits(int(state_dim) * 32)
    root_mae = _float(row.get("root_mae"))
    hll_truth_mae = _float(row.get("fstar_gstar_root_analytic_mae"))
    if math.isfinite(root_mae) and math.isfinite(hll_truth_mae) and hll_truth_mae > 0:
        row["root_mae_over_fstar_root_raw_mae"] = root_mae / hll_truth_mae
    return row


def _baseline_config_key(row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    if not _is_hll_row(row):
        return None
    n_leaves = _int(row.get("n_leaves"))
    n_train = _int(row.get("n_train"))
    n_val = _int(row.get("n_val"))
    seed = _int(row.get("seed"), _int(FNO_ARG_DEFAULTS["seed"]))
    universe_size = _int(row.get("universe_size"), _int(FNO_ARG_DEFAULTS["universe_size"]))
    min_tokens = _int(row.get("min_tokens"), _int(FNO_ARG_DEFAULTS["min_tokens"]))
    max_tokens = _int(row.get("max_tokens"), _int(FNO_ARG_DEFAULTS["max_tokens"]))
    zipf_alphas = _csv_floats(row.get("zipf_alphas")) or _csv_floats(FNO_ARG_DEFAULTS["zipf_alphas"])
    if None in (n_leaves, n_train, n_val, seed, universe_size, min_tokens, max_tokens):
        return None
    return (
        int(n_leaves),
        int(n_train),
        int(n_val),
        int(seed),
        int(universe_size),
        int(min_tokens),
        int(max_tokens),
        tuple(float(x) for x in zipf_alphas),
    )


def _baseline_precision_sweep_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    precisions: Sequence[int] = BASELINE_PRECISIONS,
) -> list[dict[str, Any]]:
    reps: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = _baseline_config_key(row)
        if key is None:
            continue
        reps.setdefault(key, row)

    out: list[dict[str, Any]] = []
    for key, row in sorted(reps.items(), key=lambda item: item[0]):
        n_leaves = int(key[0])
        n_train = int(key[1])
        n_val = int(key[2])
        seed = int(key[3])
        universe_size = int(key[4])
        min_tokens = int(key[5])
        max_tokens = int(key[6])
        zipf_alphas = ",".join(str(float(x)).rstrip("0").rstrip(".") for x in key[7])
        for precision in precisions:
            baseline: dict[str, Any] = {
                "grid": "fstar_gstar_baseline",
                "cell": f"package_hll_p{int(precision)}_L{n_leaves}",
                "status": "baseline",
                "target_kind": "hll_register_space",
                "schedule": "package_hll",
                "objective_mode": "exact_package",
                "observation": "analytic_truth",
                "readout_arch": "package_hll",
                "readout_kind": "hll_reference",
                "precision": str(int(precision)),
                "n_leaves": str(n_leaves),
                "n_train": str(n_train),
                "n_val": str(n_val),
                "seed": str(seed),
                "universe_size": str(universe_size),
                "min_tokens": str(min_tokens),
                "max_tokens": str(max_tokens),
                "zipf_alphas": zipf_alphas,
                "fstar_gstar_target_mae": 0.0,
                "fstar_gstar_target_rel_mae": 0.0,
                "baseline_kind": "fstar_gstar_precision_sweep",
            }
            baseline.update(_fstar_gstar_analytic_baseline(baseline))
            out.append(baseline)
    return out


def _active_output_dirs() -> set[str]:
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "run_fno_mergeable_sketch_diagnostic.py"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return set()
    active: set[str] = set()
    for line in out.splitlines():
        parts = line.split()
        for i, part in enumerate(parts):
            if part == "--output-dir" and i + 1 < len(parts):
                active.add(str(Path(parts[i + 1]).resolve()))
    return active


def _cell_name(output_dir: Path) -> str:
    return output_dir.name


def _row_from_cell(root: Path, cell: Mapping[str, Any], active_dirs: set[str]) -> dict[str, Any]:
    command = [str(x) for x in cell.get("command", [])] if isinstance(cell.get("command"), list) else []
    output_dir = Path(str(cell.get("output_dir", ""))).expanduser()
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    summary = _read_one_csv(_summary_path(output_dir)) or {}
    completed = bool(summary)
    status = "completed" if completed else ("running" if str(output_dir.resolve()) in active_dirs else "planned_no_summary")

    row: dict[str, Any] = {
        "grid": root.name,
        "cell": str(cell.get("cell_id") or _cell_name(output_dir)),
        "status": status,
        "family": str(cell.get("family", "")),
        "root_label_share": cell.get("root_label_share", ""),
        "sampled_node_rate": cell.get("sampled_node_rate", ""),
        "output_dir": str(output_dir),
        "summary_path": str(_summary_path(output_dir)),
        **_seed_command_fields(command),
    }
    if "sampled_node_rate" in cell:
        row["sampled_node_rate"] = cell.get("sampled_node_rate", "")
    if "root_label_share" in cell:
        row["root_label_share"] = cell.get("root_label_share", "")
    row.update(summary)
    row["grid"] = root.name
    row["cell"] = str(cell.get("cell_id") or _cell_name(output_dir))
    row["status"] = status
    row["output_dir"] = str(output_dir)
    row["summary_path"] = str(_summary_path(output_dir))
    row["observation"] = _parse_observation(row.get("oracle_observation_design") or row.get("observation"))
    return _annotate_fstar_gstar(row)


def _discover_rows(roots: Sequence[Path]) -> list[dict[str, Any]]:
    active_dirs = _active_output_dirs()
    rows: list[dict[str, Any]] = []
    seen_dirs: set[str] = set()
    for raw_root in roots:
        root = raw_root if raw_root.is_absolute() else REPO_ROOT / raw_root
        root = root.resolve()
        cells = _load_manifest_cells(root)
        for cell in cells:
            row = _row_from_cell(root, cell, active_dirs)
            rows.append(row)
            seen_dirs.add(str(Path(row["output_dir"]).resolve()))

        # Include completed ad hoc subdirs even if the root manifest was edited or absent.
        for summary_path in sorted(root.glob("*/summary.csv")):
            out_dir = summary_path.parent.resolve()
            if str(out_dir) in seen_dirs:
                continue
            summary = _read_one_csv(summary_path) or {}
            command = _read_launcher_command(out_dir)
            row: dict[str, Any] = {
                "grid": root.name,
                "cell": out_dir.name,
                "status": "completed",
                "output_dir": str(out_dir),
                "summary_path": str(summary_path),
                **_seed_command_fields(command),
            }
            row.update(summary)
            row["observation"] = _parse_observation(row.get("oracle_observation_design") or row.get("observation"))
            rows.append(_annotate_fstar_gstar(row))
            seen_dirs.add(str(out_dir))
        for summary_path in sorted(root.glob("*/summary.json")):
            out_dir = summary_path.parent.resolve()
            if str(out_dir) in seen_dirs:
                continue
            treepo_rows = _read_treepo_hll_rows(summary_path)
            if not treepo_rows:
                continue
            for idx, treepo_row in enumerate(treepo_rows):
                row: dict[str, Any] = {
                    "grid": root.name,
                    "cell": (
                        f"treepo_{treepo_row.get('model_kind', 'hll')}_"
                        f"p{treepo_row.get('precision', '')}_"
                        f"train{treepo_row.get('train_docs', '')}_"
                        f"{treepo_row.get('audit_policy', '')}"
                    ),
                    "status": "completed",
                    "output_dir": str(out_dir),
                    "summary_path": str(summary_path),
                    **treepo_row,
                }
                if len(treepo_rows) > 1:
                    row["cell"] = f"{row['cell']}_{idx:03d}"
                rows.append(_annotate_fstar_gstar(row))
            seen_dirs.add(str(out_dir))
    return rows


def _columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "grid",
        "cell",
        "status",
        "target_kind",
        "source_family",
        "model_kind",
        "schedule",
        "n_leaves",
        "objective_mode",
        "proxy_mode",
        "lean_adjusted_loss",
        "lean_merge_adapter",
        "observation",
        "root_label_share",
        "sampled_node_rate",
        "readout_arch",
        "identity_residual_init",
        "local_law_weight",
        "objective_loss_weight",
        "state_loss_weight",
        "exact_state_anchor_weight",
        "root_mae",
        "root_rel_mae",
        "learned_relative_rmse",
        "hll_relative_rmse",
        "hll_rse_theory",
        "corrected_local_law_loss_mean",
        "proxy_loss_mean",
        "oracle_ipw_loss_mean",
        "ipw_correction_mean",
        "observed_rows_mean",
        "fstar_gstar_target_mae",
        "gap_to_fstar_gstar_target_mae",
        "fstar_gstar_root_analytic_mae",
        "fstar_gstar_root_analytic_rel_mae",
        "root_mae_over_fstar_root_raw_mae",
        "fstar_gstar_leaf_analytic_mae",
        "fstar_gstar_leaf_analytic_rel_mae",
        "hll_register_count",
        "hll_state_bits",
        "neural_state_fp16_bits",
        "neural_state_fp32_bits",
        "leaf_readout_mae",
        "internal_readout_mae",
        "root_readout_mae",
        "all_node_readout_mae",
        "merge_state_internal_mae",
        "merge_state_root_mae",
        "merge_readout_internal_mae",
        "merge_readout_root_mae",
        "learned_f_on_exact_root_mae",
        "official_f_on_learned_root_mae",
        "merge_state_mae",
        "train_observed_rows_end",
        "train_population_rows_end",
        "train_root_observed_rows_end",
        "train_root_population_rows_end",
        "train_nonroot_observed_rows_end",
        "train_nonroot_population_rows_end",
        "train_observed_rows_per_doc_end",
        "train_root_observed_rows_per_doc_end",
        "train_nonroot_observed_rows_per_doc_end",
        "train_max_ipw_weight_end",
        "train_effective_sample_size_end",
        "wall_seconds",
        "output_dir",
    ]
    out = [key for key in preferred if any(key in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in out:
                out.append(key)
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = _columns(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in cols})


def _completed(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("status") == "completed"]


def _sorted_completed(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        _completed(rows),
        key=lambda row: (
            str(row.get("grid", "")),
            str(row.get("objective_mode", "")),
            str(row.get("schedule", "")),
            _float(row.get("n_leaves")),
            str(row.get("cell", "")),
        ),
    )


def _sampled_node_rate(row: Mapping[str, Any]) -> float:
    direct = _float(row.get("sampled_node_rate"))
    if math.isfinite(direct):
        return direct
    if str(row.get("observation", "")) == "root_only":
        return 0.0
    payload_raw = row.get("oracle_observation_design")
    if isinstance(payload_raw, Mapping):
        params = payload_raw.get("design_parameters", {})
        if isinstance(params, Mapping):
            value = _float(params.get("sampled_node_rate"))
            if math.isfinite(value):
                return value
    text = str(payload_raw or "").strip()
    if text.startswith("{"):
        try:
            payload = ast.literal_eval(text)
        except Exception:
            payload = None
        if isinstance(payload, Mapping):
            params = payload.get("design_parameters", {})
            if isinstance(params, Mapping):
                value = _float(params.get("sampled_node_rate"))
                if math.isfinite(value):
                    return value
    return float("nan")


def _is_known_f_row(row: Mapping[str, Any]) -> bool:
    return str(row.get("grid", "")).startswith("hll_known_f_g") or str(row.get("cell", "")).startswith("known_f_")


def _token_regime(row: Mapping[str, Any]) -> tuple[int | None, int | None]:
    return _int(row.get("min_tokens")), _int(row.get("max_tokens"))


def _filter_token_regime(
    rows: Sequence[dict[str, Any]],
    *,
    min_tokens: int | None,
    max_tokens: int | None,
) -> list[dict[str, Any]]:
    if min_tokens is None and max_tokens is None:
        return list(rows)
    if min_tokens is None or max_tokens is None:
        raise ValueError("token filtering requires both min_tokens and max_tokens")
    out: list[dict[str, Any]] = []
    for row in rows:
        row_min, row_max = _token_regime(row)
        if row_min == int(min_tokens) and row_max == int(max_tokens):
            out.append(row)
    return out


def _token_regime_lines(
    rows: Sequence[Mapping[str, Any]],
    *,
    token_filter: tuple[int, int] | None = None,
) -> list[str]:
    counts: dict[tuple[int | None, int | None], int] = {}
    for row in rows:
        counts[_token_regime(row)] = counts.get(_token_regime(row), 0) + 1
    lines = ["## Token Regime", ""]
    if token_filter is not None:
        lines.append(
            f"Filtered to `{int(token_filter[0])}` to `{int(token_filter[1])}` tokens/document before computing headline plots and tables."
        )
        lines.append("")
    elif len(counts) > 1:
        lines.append(
            "Multiple token regimes are present. Pass `--token-count 1024` or `--min-tokens ... --max-tokens ...` to prevent old 128-token rows and new 1024-token rows from sharing headline plots."
        )
        lines.append("")
    elif len(counts) == 1:
        (only_min, only_max), _count = next(iter(counts.items()))
        if only_min is not None and only_max is not None:
            lines.append(f"All discovered rows use `{only_min}` to `{only_max}` tokens/document.")
            lines.append("")
    if counts:
        lines.extend(
            [
                "| min tokens | max tokens | rows |",
                "|---:|---:|---:|",
            ]
        )
        for (min_tok, max_tok), count in sorted(counts.items(), key=lambda item: (item[0][0] is None, item[0][0] or -1, item[0][1] or -1)):
            lines.append(
                f"| {'' if min_tok is None else min_tok} | {'' if max_tok is None else max_tok} | {count} |"
            )
    else:
        lines.append("No rows matched the requested token regime.")
    return lines


def _series_label(row: Mapping[str, Any]) -> str:
    cell = str(row.get("cell", ""))
    grid = str(row.get("grid", ""))
    if cell.startswith("fstar_gstar") or str(row.get("objective_mode", "")) == "exact_package":
        return r"classical HLL $f^\star \circ g^\star$"
    if _is_known_f_row(row):
        if cell.startswith("known_f_g_exact"):
            return r"fixed $f^\star$: learn $g_\theta$ (exact)"
        if "gfgf_exact" in cell:
            return r"fixed $f^\star$: learn $g_\theta/f_\theta$ (exact)"
        if "rollout_dense" in cell:
            return r"fixed $f^\star$: rollout $g_\theta/f_\theta$ (dense)"
        if "rollout_root" in cell:
            return r"fixed $f^\star$: rollout $g_\theta/f_\theta$ (root)"
    if grid.startswith("hll_sampled_node_rate_grid"):
        rate = _sampled_node_rate(row)
        if math.isfinite(rate):
            return rf"fixed $f^\star$: root + {100.0 * rate:g}% non-root"
        return r"fixed $f^\star$: sampled non-root"
    if grid.startswith("hll_leaf_grid") or grid.startswith("hll_canonical_observation_grid"):
        obs = str(row.get("observation", ""))
        mode = str(row.get("objective_mode", ""))
        readout = str(row.get("readout_arch", ""))
        ident = str(row.get("identity_residual_init", ""))
        mode_label = "exact" if mode == "exact_rows" else "rollout"
        readout_label = readout.replace("hll_", "") or "readout"
        obs_label = {
            "root_only": "root",
            "dense_oracle": "dense",
            "sampled_nodes": "sampled",
        }.get(obs, obs or "root")
        if obs == "budgeted_mass":
            match = re.search(r"budget_R(\d+)", cell)
            obs_label = f"R{match.group(1)}" if match else "budget"
        init_label = "id-init" if ident == "True" else "no-id"
        if mode_label == "exact":
            if readout_label == "residual":
                base = r"learn $f_\theta,g_\theta$ + residual $f_\theta$"
            else:
                base = r"learn $f_\theta,g_\theta$ (exact, id init)" if init_label == "id-init" else r"learn $f_\theta,g_\theta$ (exact)"
            return f"{base}, id init" if readout_label == "residual" and init_label == "id-init" else base
        return rf"learn $f_\theta,g_\theta$ (rollout {obs_label})"
    if grid.startswith("hll_many"):
        label = re.sub(r"^hll_L\d+_", "", cell)
        label = re.sub(r"\d+$", "", label)
        label = label.replace("rollout_sampled1", "rollout sampled")
        label = label.replace("rollout_root", "rollout root")
        pretty = label.replace("_", " ")
        if pretty == "formula fgfg":
            return r"probe: $f/g/f/g$"
        if pretty == "formula gfgfgf":
            return r"probe: $g/f/g/f/g/f$"
        if pretty == "formula g both":
            return r"probe: $g$ state+scalar"
        if pretty == "formula rollout root fgfg":
            return r"probe: rollout $f/g/f/g$ root"
        return f"probe: {pretty}"
    if grid.startswith("hll_recoverability"):
        label = re.sub(r"_L\d+.*$", "", cell)
        return f"recoverability {label.replace('_', ' ')}"
    return cell


FIGURE_CAPTIONS = {
    "grid_status.png": "Current grid coverage, including running and planned/no-summary cells.",
    "root_rel_mae_by_leaves.png": "Relative approximation error to the HLL target. This is not HLL-vs-truth accuracy; L=1 is mostly a no-merge canary.",
    "root_mae_by_leaves.png": "Approximation MAE to the HLL target, in distinct-count units. This is not an estimator-accuracy claim against truth.",
    "root_mae_with_hll_line_by_leaves.png": "Approximation MAE over leaves, with the register-matched classical HLL p=8 estimator error as a count-unit reference line. Curves below the line are closer to HLL than HLL is to truth; they are not necessarily better estimators than HLL.",
    "canonical_root_mae_with_classical_hll_by_leaves.png": "Canonical learned f_theta,g_theta exact-row family, shown against the classical HLL p=8 raw estimator error scale. The red curve is approximation error to HLL; the black curve is HLL error to truth.",
    "canonical_error_components_by_leaves.png": "Canonical learned f_theta,g_theta component diagnostics from newly instrumented runs: root error, leaf/internal readout error, and internal/root merge-state residuals.",
    "merge_state_mae_by_leaves.png": "Register-state merge residual; this isolates the learned g_theta part from the scalar readout f_theta.",
    "root_mae_context_ratio_by_leaves.png": "Approximation-error context: learned root error to the package HLL target divided by package HLL's own raw root error to true distinct count. This is an error-budget scale, not an end-task accuracy plot.",
    "fstar_gstar_root_raw_mae_by_leaves.png": "Classical HLL f* after g* error against true distinct counts. Lines are flat because exact HLL merge is lossless with respect to leaf partition.",
    "known_f_g_residuals.png": "Known f* runs isolate whether g_theta can recover the HLL merge/root state when the readout f* is supplied.",
    "fstar_gtheta_rollout_dense_vs_root_by_leaves.png": "Supplied-readout rollout comparison: package f* applied to learned g_theta roots under dense all-node labels versus sparse root-only labels.",
    "fstar_gtheta_sampled_rate_by_leaves.png": "Scalar-only sampled-node-rate grid: package f* applied to learned g_theta roots, with roots always observed and curves varying the random non-root label rate.",
}


def _setup_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def _set_leaf_axis(ax: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    leaves = sorted({_int(row.get("n_leaves")) for row in rows if _int(row.get("n_leaves")) is not None})
    if not leaves:
        return
    ax.set_xscale("log", base=2)
    ax.set_xticks(leaves)
    ax.set_xticklabels([str(leaf) for leaf in leaves])
    ax.set_xlabel("leaves")


def _aggregate_series(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, tuple[list[int], list[float]]]:
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        leaf = _int(row.get("n_leaves"))
        value = _float(row.get(metric))
        if leaf is None or not math.isfinite(value):
            continue
        label = _series_label(row)
        grouped.setdefault(label, {}).setdefault(int(leaf), []).append(float(value))
    out: dict[str, tuple[list[int], list[float]]] = {}
    for label, by_leaf in grouped.items():
        xs = sorted(by_leaf)
        # If multiple cells collapse to the same visual series, show their
        # mean rather than the best run. The table keeps the per-cell detail.
        ys = [sum(values) / len(values) for leaf, values in sorted(by_leaf.items())]
        out[label] = (xs, ys)
    return out


def _plot_metric_by_leaves(
    rows: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    metric: str,
    title: str,
    ylabel: str,
    show_target_optimum: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_plot_style()
    plot_rows = [row for row in _completed(rows) if math.isfinite(_float(row.get(metric))) and math.isfinite(_float(row.get("n_leaves")))]
    if not plot_rows:
        return
    plot_rows = [row for row in plot_rows if _is_hll_row(row)]
    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    groups = _aggregate_series(plot_rows, metric)
    if len(groups) > len(PRIMARY_PLOT_SERIES):
        groups = {label: groups[label] for label in PRIMARY_PLOT_SERIES if label in groups}
    for label, (xs, ys) in groups.items():
        ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=4, label=label)
    if show_target_optimum and any(math.isfinite(_float(row.get("fstar_gstar_target_mae"))) for row in plot_rows):
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.2, label="classical HLL target (0)")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    _set_leaf_axis(ax, plot_rows)
    ax.set_yscale("symlog", linthresh=1e-5)
    if show_target_optimum:
        _bottom, top = ax.get_ylim()
        ax.set_ylim(0.0, top)
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_canonical_root_mae_with_classical_hll(
    rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    precision: int = 8,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_plot_style()
    canonical_rows = [
        row
        for row in _completed(rows)
        if _is_hll_row(row)
        and _series_label(row) == CANONICAL_SERIES_LABEL
        and (
            str(row.get("grid", "")).startswith("hll_leaf_grid")
            or str(row.get("grid", "")).startswith("hll_canonical_observation_grid")
        )
        and math.isfinite(_float(row.get("root_mae")))
        and math.isfinite(_float(row.get("n_leaves")))
    ]
    hll_rows = [
        row
        for row in baseline_rows
        if _int(row.get("precision")) == int(precision)
        and math.isfinite(_float(row.get("fstar_gstar_root_analytic_mae")))
        and math.isfinite(_float(row.get("n_leaves")))
    ]
    if not canonical_rows or not hll_rows:
        return

    def mean_by_leaf(source: Sequence[Mapping[str, Any]], key: str) -> tuple[list[int], list[float]]:
        grouped: dict[int, list[float]] = {}
        for row in source:
            leaf = _int(row.get("n_leaves"))
            if leaf is None:
                continue
            value = _float(row.get(key))
            if not math.isfinite(value):
                continue
            grouped.setdefault(int(leaf), []).append(float(value))
        xs = sorted(grouped)
        ys = [sum(grouped[x]) / len(grouped[x]) for x in xs]
        return xs, ys

    canonical_xs, canonical_ys = mean_by_leaf(canonical_rows, "root_mae")
    hll_xs, hll_ys = mean_by_leaf(hll_rows, "fstar_gstar_root_analytic_mae")
    if not canonical_xs or not hll_xs:
        return

    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    ax.plot(
        canonical_xs,
        canonical_ys,
        color="#d62728",
        marker="o",
        linewidth=2.0,
        markersize=5,
        label=r"canonical learn $f_\theta,g_\theta$ (to HLL)",
    )
    ax.plot(
        hll_xs,
        hll_ys,
        color="#222222",
        linestyle="--",
        marker="s",
        linewidth=1.8,
        markersize=4,
        label=f"classical HLL p={precision} (to truth)",
    )
    ax.set_title("Canonical learned operator with classical HLL scale")
    ax.set_ylabel("MAE in distinct-count units")
    _set_leaf_axis(ax, [*canonical_rows, *hll_rows])
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_canonical_error_components(rows: Sequence[Mapping[str, Any]], output: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_plot_style()
    if output.exists():
        output.unlink()
    plot_rows = [
        row
        for row in _completed(rows)
        if _is_hll_row(row)
        and _series_label(row) == CANONICAL_SERIES_LABEL
        and math.isfinite(_float(row.get("n_leaves")))
        and any(
            math.isfinite(_float(row.get(metric)))
            for metric in (
                "leaf_readout_mae",
                "internal_readout_mae",
                "merge_state_internal_mae",
                "merge_state_root_mae",
            )
        )
    ]
    if not plot_rows:
        return False
    metric_labels = [
        ("root_mae", "root end-to-end"),
        ("leaf_readout_mae", r"leaf $f_\theta$"),
        ("internal_readout_mae", r"internal $f_\theta$"),
        ("merge_state_internal_mae", r"internal $g_\theta$ state"),
        ("merge_state_root_mae", r"root $g_\theta$ state"),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    for metric, label in metric_labels:
        grouped: dict[int, list[float]] = {}
        for row in plot_rows:
            leaf = _int(row.get("n_leaves"))
            value = _float(row.get(metric))
            if leaf is None or not math.isfinite(value):
                continue
            grouped.setdefault(int(leaf), []).append(float(value))
        if not grouped:
            continue
        xs = sorted(grouped)
        ys = [sum(grouped[x]) / len(grouped[x]) for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=label)
    ax.set_title("Canonical learned operator component errors")
    ax.set_ylabel("MAE")
    _set_leaf_axis(ax, plot_rows)
    ax.set_yscale("symlog", linthresh=1e-6)
    _bottom, top = ax.get_ylim()
    ax.set_ylim(0.0, top)
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_root_mae_with_hll_line(
    rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    precision: int = 8,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_plot_style()
    plot_rows = [
        row
        for row in _completed(rows)
        if _is_hll_row(row)
        and math.isfinite(_float(row.get("root_mae")))
        and math.isfinite(_float(row.get("n_leaves")))
    ]
    hll_rows = [
        row
        for row in baseline_rows
        if _int(row.get("precision")) == int(precision)
        and math.isfinite(_float(row.get("fstar_gstar_root_analytic_mae")))
        and math.isfinite(_float(row.get("n_leaves")))
    ]
    if not plot_rows or not hll_rows:
        return

    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    groups = _aggregate_series(plot_rows, "root_mae")
    if len(groups) > len(PRIMARY_PLOT_SERIES):
        groups = {label: groups[label] for label in PRIMARY_PLOT_SERIES if label in groups}
    for label, (xs, ys) in groups.items():
        ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=4, label=label)

    by_leaf: dict[int, list[float]] = {}
    for row in hll_rows:
        leaf = _int(row.get("n_leaves"))
        if leaf is None:
            continue
        by_leaf.setdefault(int(leaf), []).append(_float(row.get("fstar_gstar_root_analytic_mae")))
    hll_xs = sorted(by_leaf)
    hll_ys = [sum(by_leaf[x]) / len(by_leaf[x]) for x in hll_xs]
    ax.plot(
        hll_xs,
        hll_ys,
        color="#222222",
        linestyle="--",
        marker="s",
        linewidth=1.8,
        markersize=4,
        label=f"classical HLL p={precision} to truth",
    )

    ax.set_title("Root approximation MAE with classical HLL scale")
    ax.set_ylabel("MAE in distinct-count units")
    _set_leaf_axis(ax, [*plot_rows, *hll_rows])
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_context_ratio_by_leaves(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_plot_style()
    plot_rows = [
        row
        for row in _completed(rows)
        if _is_hll_row(row)
        and math.isfinite(_float(row.get("root_mae_over_fstar_root_raw_mae")))
        and _float(row.get("root_mae_over_fstar_root_raw_mae")) > 0
        and math.isfinite(_float(row.get("n_leaves")))
    ]
    if not plot_rows:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    groups = _aggregate_series(plot_rows, "root_mae_over_fstar_root_raw_mae")
    groups = {label: groups[label] for label in CONTEXT_RATIO_SERIES if label in groups}
    for label, (xs, ys) in groups.items():
        ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=4, label=label)
    ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1.2, label="same as package HLL raw MAE")
    ax.set_title("Approximation error as fraction of classical HLL estimator error")
    ax.set_ylabel("approximation MAE / classical HLL estimator MAE")
    _set_leaf_axis(ax, plot_rows)
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_fstar_root_by_leaves(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_plot_style()
    plot_rows = [
        row
        for row in rows
        if _is_hll_row(row)
        and math.isfinite(_float(row.get("fstar_gstar_root_analytic_mae")))
        and math.isfinite(_float(row.get("precision")))
        and math.isfinite(_float(row.get("n_leaves")))
    ]
    if not plot_rows:
        return
    grouped: dict[int, dict[int, list[float]]] = {}
    for row in plot_rows:
        precision = _int(row.get("precision"))
        leaf = _int(row.get("n_leaves"))
        if precision is None or leaf is None:
            continue
        grouped.setdefault(int(precision), {}).setdefault(int(leaf), []).append(_float(row.get("fstar_gstar_root_analytic_mae")))
    if not grouped:
        return
    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    all_ys: list[float] = []
    for precision, by_leaf in sorted(grouped.items()):
        xs = sorted(by_leaf)
        ys = [sum(by_leaf[x]) / len(by_leaf[x]) for x in xs]
        all_ys.extend(ys)
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            label=f"p={precision} ({1 << precision:,} regs)",
        )
    ax.set_title("Classical HLL root MAE by leaves")
    ax.set_ylabel("root MAE in distinct-count units")
    _set_leaf_axis(ax, plot_rows)
    if all_ys:
        ymax = max(all_ys)
        ax.set_ylim(0.0, ymax * 1.12 if ymax > 0 else 1.0)
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(frameon=False, title="HLL precision")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_known_f_g(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_plot_style()
    known = [
        row
        for row in _completed(rows)
        if _is_known_f_row(row)
        and str(row.get("cell", "")).startswith("known_f_g_exact")
        and math.isfinite(_float(row.get("n_leaves")))
    ]
    if not known:
        return
    known = sorted(known, key=lambda row: _float(row.get("n_leaves")))
    xs = [_int(row.get("n_leaves")) or 0 for row in known]
    metrics = [
        ("root_mae", "formula/readout residual"),
        ("official_f_on_learned_root_mae", "official readout residual"),
        ("merge_state_mae", "register-state merge MAE"),
        ("learned_f_on_exact_root_mae", "f on exact root residual"),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    for metric, label in metrics:
        ys = [_float(row.get(metric)) for row in known]
        if any(math.isfinite(y) for y in ys):
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=label)
    if any(math.isfinite(_float(row.get("fstar_gstar_target_mae"))) for row in known):
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.2, label="classical HLL target (0)")
    ax.set_title("Fixed f*: learned g_theta residuals")
    ax.set_ylabel("MAE")
    _set_leaf_axis(ax, known)
    ax.set_yscale("symlog", linthresh=1e-6)
    _bottom, top = ax.get_ylim()
    ax.set_ylim(0.0, top)
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_fstar_gtheta_rollout_dense_vs_root(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_plot_style()
    plot_rows = [
        row
        for row in _completed(rows)
        if _is_known_f_row(row)
        and str(row.get("cell", "")).startswith("known_f_gf_rollout_")
        and str(row.get("observation", "")) in {"dense_oracle", "root_only"}
        and math.isfinite(_float(row.get("n_leaves")))
    ]
    if not plot_rows:
        return

    series = [
        ("dense_oracle", r"dense all-node labels"),
        ("root_only", r"sparse root-only labels"),
    ]
    metrics = [
        ("official_f_on_learned_root_mae", r"package $f^\star(g_\theta)$ root MAE"),
        ("merge_state_mae", r"$g_\theta$ state MAE"),
    ]
    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(7.8, 6.4),
        sharex=True,
        constrained_layout=True,
    )
    if len(metrics) == 1:
        axes = [axes]
    colors = {
        "dense_oracle": "#1f77b4",
        "root_only": "#d62728",
    }
    for ax, (metric, ylabel) in zip(axes, metrics):
        plotted_any = False
        all_ys: list[float] = []
        for observation, label in series:
            grouped: dict[int, list[float]] = {}
            for row in plot_rows:
                if str(row.get("observation", "")) != observation:
                    continue
                leaf = _int(row.get("n_leaves"))
                value = _float(row.get(metric))
                if leaf is None or not math.isfinite(value):
                    continue
                grouped.setdefault(int(leaf), []).append(float(value))
            if not grouped:
                continue
            xs = sorted(grouped)
            ys = [sum(grouped[x]) / len(grouped[x]) for x in xs]
            all_ys.extend(ys)
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.0,
                markersize=5,
                color=colors[observation],
                label=label,
            )
            plotted_any = True
        if metric == "official_f_on_learned_root_mae":
            ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0, label=r"classical HLL target")
        ymax = max(all_ys) if all_ys else 0.0
        ax.set_ylim(0.0, ymax * 1.15 if ymax > 0 else 1.0)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", alpha=0.25)
        if plotted_any:
            ax.legend(frameon=False, loc="upper left")
    axes[0].set_title(r"Supplied $f^\star$: learned $g_\theta$ rollout by observation design")
    _set_leaf_axis(axes[-1], plot_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _sampled_rate_label(rate: float) -> str:
    if math.isfinite(rate) and abs(float(rate)) <= 1e-12:
        return "root only"
    if math.isfinite(rate) and abs(float(rate) - 1.0) <= 1e-12:
        return "100% non-root"
    return f"{100.0 * float(rate):g}% non-root"


def _first_finite(row: Mapping[str, Any], keys: Sequence[str]) -> float:
    for key in keys:
        value = _float(row.get(key))
        if math.isfinite(value):
            return value
    return float("nan")


def _plot_fstar_gtheta_sampled_rate(rows: Sequence[Mapping[str, Any]], output: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_plot_style()
    if output.exists():
        output.unlink()
    plot_rows = [
        row
        for row in _completed(rows)
        if str(row.get("grid", "")).startswith("hll_sampled_node_rate_grid")
        and _is_hll_row(row)
        and math.isfinite(_float(row.get("n_leaves")))
        and math.isfinite(_sampled_node_rate(row))
    ]
    if not plot_rows:
        return False

    def grouped(metric_keys: Sequence[str]) -> dict[float, tuple[list[int], list[float]]]:
        by_rate: dict[float, dict[int, list[float]]] = {}
        for row in plot_rows:
            leaf = _int(row.get("n_leaves"))
            rate = _sampled_node_rate(row)
            value = _first_finite(row, metric_keys)
            if leaf is None or not math.isfinite(rate) or not math.isfinite(value):
                continue
            by_rate.setdefault(float(rate), {}).setdefault(int(leaf), []).append(float(value))
        out: dict[float, tuple[list[int], list[float]]] = {}
        for rate, by_leaf in sorted(by_rate.items()):
            xs = sorted(by_leaf)
            ys = [sum(by_leaf[x]) / len(by_leaf[x]) for x in xs]
            out[rate] = (xs, ys)
        return out

    metric_groups = [
        (
            ("official_f_on_learned_root_mae",),
            r"package $f^\star(g_\theta)$ root MAE",
            r"Package $f^\star$ on Learned Root",
        ),
        (
            ("merge_state_root_mae", "merge_state_mae"),
            r"$g_\theta$ root/state MAE",
            r"Merge-State Diagnostic",
        ),
    ]
    primary = grouped(metric_groups[0][0])
    if not primary:
        return False

    fig, axes = plt.subplots(2, 1, figsize=(7.8, 6.4), sharex=True, constrained_layout=True)
    palette = {
        0.0: "#d62728",
        0.01: "#ff7f0e",
        0.03: "#9467bd",
        0.1: "#1f77b4",
        1.0: "#2ca02c",
    }
    for ax, (keys, ylabel, title) in zip(axes, metric_groups):
        groups = grouped(keys)
        all_ys: list[float] = []
        for rate, (xs, ys) in groups.items():
            all_ys.extend(ys)
            color = palette.get(round(float(rate), 8))
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.0,
                markersize=5,
                color=color,
                label=_sampled_rate_label(rate),
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_yscale("symlog", linthresh=1e-4)
        if all_ys:
            ymax = max(all_ys)
            ax.set_ylim(0.0, ymax * 1.18 if ymax > 0 else 1.0)
        ax.grid(True, which="major", alpha=0.25)
        ax.legend(frameon=False, ncol=2, loc="upper left")
    axes[0].axhline(0.0, color="#222222", linestyle="--", linewidth=1.0, label=r"classical HLL target")
    axes[0].set_title(r"Supplied $f^\star$: scalar node-label sampling")
    _set_leaf_axis(axes[-1], plot_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_status(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_plot_style()
    grids = sorted({str(row.get("grid", "")) for row in rows})
    statuses = ("completed", "running", "planned_no_summary")
    counts = {grid: {status: 0 for status in statuses} for grid in grids}
    for row in rows:
        grid = str(row.get("grid", ""))
        status = str(row.get("status", "planned_no_summary"))
        counts.setdefault(grid, {item: 0 for item in statuses})
        counts[grid][status if status in statuses else "planned_no_summary"] += 1
    fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
    bottom = [0] * len(grids)
    colors = {"completed": "#4c78a8", "running": "#59a14f", "planned_no_summary": "#bab0ac"}
    for status in statuses:
        vals = [counts[grid][status] for grid in grids]
        ax.bar(grids, vals, bottom=bottom, label=status, color=colors[status])
        bottom = [a + b for a, b in zip(bottom, vals)]
    ax.set_title("HLL FNO grid status")
    ax.set_ylabel("cells")
    ax.tick_params(axis="x", labelrotation=18)
    ax.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _md_table(rows: Sequence[Mapping[str, Any]], *, limit: int | None = None) -> list[str]:
    lines = [
        "| grid | cell | status | L | schedule | objective | obs | readout | id init | root MAE | root rel MAE | HLL target MAE | gap to target | classical HLL leaf raw MAE | classical HLL root raw MAE | f exact MAE | official-on-learned MAE | merge MAE |",
        "|---|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    display = rows[:limit] if limit is not None else rows
    for row in display:
        lines.append(
            "| {grid} | {cell} | {status} | {leaves} | {schedule} | {objective} | {obs} | {readout} | {ident} | {root} | {rel} | {fstar} | {gap} | {fstar_leaf_raw} | {fstar_root_raw} | {fexact} | {official} | {merge} |".format(
                grid=row.get("grid", ""),
                cell=row.get("cell", ""),
                status=row.get("status", ""),
                leaves=_fmt_int(row.get("n_leaves")),
                schedule=row.get("schedule", ""),
                objective=row.get("objective_mode", ""),
                obs=row.get("observation", ""),
                readout=row.get("readout_arch", ""),
                ident=row.get("identity_residual_init", ""),
                root=_fmt(row.get("root_mae")),
                rel=_fmt(row.get("root_rel_mae")),
                fstar=_fmt(row.get("fstar_gstar_target_mae")),
                gap=_fmt(row.get("gap_to_fstar_gstar_target_mae")),
                fstar_leaf_raw=_fmt(row.get("fstar_gstar_leaf_analytic_mae")),
                fstar_root_raw=_fmt(row.get("fstar_gstar_root_analytic_mae") or row.get("fstar_gstar_analytic_mae")),
                fexact=_fmt(row.get("learned_f_on_exact_root_mae")),
                official=_fmt(row.get("official_f_on_learned_root_mae")),
                merge=_fmt(row.get("merge_state_mae")),
            )
        )
    return lines


def _fstar_summary_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        if not _is_hll_row(row):
            continue
        leaf = _int(row.get("n_leaves"))
        precision = _int(row.get("precision"))
        if leaf is None or precision is None:
            continue
        if not math.isfinite(_float(row.get("fstar_gstar_analytic_rel_mae"))):
            continue
        grouped.setdefault((int(precision), int(leaf)), []).append(row)
    lines = [
        "| precision | L | classical HLL target MAE | target rel MAE | leaf raw MAE vs truth | leaf rel MAE vs truth | root raw MAE vs truth | root rel MAE vs truth |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (precision, leaf), group in sorted(grouped.items()):
        def mean(key: str) -> float:
            vals = [_float(row.get(key)) for row in group if math.isfinite(_float(row.get(key)))]
            return float(sum(vals) / len(vals)) if vals else float("nan")

        lines.append(
            f"| {precision} | {leaf} | {_fmt(mean('fstar_gstar_target_mae'))} | "
            f"{_fmt(mean('fstar_gstar_target_rel_mae'))} | {_fmt(mean('fstar_gstar_leaf_analytic_mae'))} | "
            f"{_fmt(mean('fstar_gstar_leaf_analytic_rel_mae'))} | {_fmt(mean('fstar_gstar_root_analytic_mae'))} | "
            f"{_fmt(mean('fstar_gstar_root_analytic_rel_mae'))} |"
        )
    return lines


def _mode_int(rows: Sequence[Mapping[str, Any]], key: str) -> int | None:
    counts: dict[int, int] = {}
    for row in rows:
        value = _int(row.get(key))
        if value is None:
            continue
        counts[int(value)] = counts.get(int(value), 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _baseline_root_mae(rows: Sequence[Mapping[str, Any]], precision: int) -> float:
    vals = [
        _float(row.get("fstar_gstar_root_analytic_mae"))
        for row in rows
        if _int(row.get("precision")) == int(precision)
        and math.isfinite(_float(row.get("fstar_gstar_root_analytic_mae")))
    ]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _best_context_row(rows: Sequence[Mapping[str, Any]], predicate: Any) -> Mapping[str, Any] | None:
    candidates = [
        row
        for row in _completed(rows)
        if _is_hll_row(row)
        and _int(row.get("n_leaves")) not in (None, 1)
        and math.isfinite(_float(row.get("root_mae")))
        and predicate(row)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: _float(row.get("root_mae")))


def _terminology_lines() -> list[str]:
    return [
        "## Terminology",
        "",
        "`classical HLL` means the package/native HLL pipeline: exact HLL leaf sketches, exact registerwise-max merge `g*`, and exact package readout `f*`. It is accurate to call this `f* after g*`, but it is not a learned model and it is not a leaf-count limit. At fixed precision `p`, the root sketch has constant memory in the number of leaves; the raw HLL-vs-truth error is estimator error, not approximation error.",
        "",
        "For a single canonical learned comparison, use `learn f_theta,g_theta (exact)`: the no-identity-init, formula-readout, exact-row `fgfg` family (`exact_formula_noid_L*`). That is the cleanest current `f_theta,g_theta` neural-operator family because it learns both the merge state and readout while avoiding the rollout-observation confound.",
        "",
        "| report term | meaning |",
        "|---|---|",
        "| `exact_rows` | Supervised on rows generated from exact HLL states/readouts. This tests recoverability without rollout compounding. |",
        "| `rollout dense` | Rollout/local-law training with oracle observations exposed densely across the tree. This is a high-information ablation. |",
        "| `rollout root` | Rollout/local-law training with only root observations. Internal behavior must be inferred from the objective, so it is much harder. |",
        "| `sampled_nodes` | Rollout/local-law training with a sampled subset of non-root oracle node labels. |",
        "| `budgeted_mass` / `fixed_mass` | A fixed observation-mass design: the total supervised mass per document is capped and split between root and non-root labels. This is the HLL version of the constant-observation-mass experiment. |",
        "| `g state` | A probe that learns the merge/state map `g_theta` using state-level loss. |",
        "| `g scalar` | A probe that trains `g_theta` through scalar/readout loss only; this has been unstable in the current results. |",
        "| `g state+scalar` | A probe that combines state-level and scalar/readout losses for `g_theta`. |",
        "| `f/g/f/g` | A staged schedule alternating readout and merge training. These are probes, not the canonical comparison family. |",
    ]


def _canonical_summary_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    canonical = [
        row
        for row in _completed(rows)
        if _is_hll_row(row)
        and _series_label(row) == CANONICAL_SERIES_LABEL
        and (
            str(row.get("grid", "")).startswith("hll_leaf_grid")
            or str(row.get("grid", "")).startswith("hll_canonical_observation_grid")
        )
        and math.isfinite(_float(row.get("root_mae")))
    ]
    if not canonical:
        return []
    canonical = sorted(canonical, key=lambda row: _float(row.get("n_leaves")))
    lines = [
        "## Canonical Learned Family",
        "",
        "This is the current recommended one-line neural-operator comparison family: exact-row `fgfg`, formula readout, no identity-residual init. It asks whether a learned `f_theta,g_theta` can reproduce the classical HLL target as leaves increase.",
        "",
        "| L | cell | root MAE to HLL | fraction of p=8 HLL raw MAE | official f* on learned root | merge MAE |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in canonical:
        lines.append(
            f"| {_fmt_int(row.get('n_leaves'))} | {row.get('cell', '')} | {_fmt(row.get('root_mae'))} | "
            f"{_fmt(row.get('root_mae_over_fstar_root_raw_mae'))} | {_fmt(row.get('official_f_on_learned_root_mae'))} | "
            f"{_fmt(row.get('merge_state_mae'))} |"
        )
    return lines


def _context_summary_lines(
    rows: Sequence[Mapping[str, Any]],
    context_baseline_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    hll_rows = [row for row in rows if _is_hll_row(row)]
    state_dim = _mode_int(hll_rows, "state_dim")
    precision = _mode_int(hll_rows, "precision")
    p8_mae = _baseline_root_mae(context_baseline_rows, 8)
    p9_mae = _baseline_root_mae(context_baseline_rows, 9)
    p16_mae = _baseline_root_mae(context_baseline_rows, 16)
    lines = [
        "## HLL Context for NO/FNO Error",
        "",
        "There are two different questions here. For merge-law recovery, HLL is the gold standard because the exact HLL state and max-register merge define the target algebra. For cardinality accuracy, HLL is only an estimator, so HLL-vs-truth error is part of the task error rather than a gold-standard label.",
        "",
        "The current fair capacity comparison is register-count matched: HLL has `2^p` internal registers, and the neural operator has `state_dim` internal coordinates. If we raise HLL to p=9 as a cleaner teacher, the matched neural state should be 512 coordinates rather than the current 256.",
        "",
    ]
    if state_dim is not None and precision is not None:
        state_match = int(math.log2(int(state_dim))) if int(state_dim) > 0 and int(state_dim) & (int(state_dim) - 1) == 0 else None
        lines.extend(
            [
                "| role | value | package HLL root raw MAE vs true count |",
                "|---|---:|---:|",
                f"| current HLL target / register-matched baseline | p={precision} ({_hll_register_count(precision):,} registers) | {_fmt(_baseline_root_mae(context_baseline_rows, precision))} |",
                f"| current neural root state | {state_dim:,} coordinates |  |",
            ]
        )
        if state_match is not None:
            lines.append(
                f"| state-size matched HLL | p={state_match} ({_hll_register_count(state_match):,} registers) | {_fmt(_baseline_root_mae(context_baseline_rows, state_match))} |"
            )
        if math.isfinite(p9_mae):
            lines.append(f"| possible cleaner teacher | p=9 ({_hll_register_count(9):,} registers) | {_fmt(p9_mae)} |")
        if math.isfinite(p16_mae):
            lines.append(f"| near-exact reference teacher | p=16 ({_hll_register_count(16):,} registers) | {_fmt(p16_mae)} |")
        lines.append("")
        lines.extend(
            [
                "Interpretation: p=8/256 registers is the fair comparison for the current NO/FNO runs. p=9 is a reasonable next experiment if we want less HLL estimator noise, but then the matched neural operator should use `state_dim=512`. p=16 is useful as a teacher/reference, not as a fair capacity-matched baseline for these 256-coordinate runs.",
                "",
            ]
        )

    slices: list[tuple[str, Any]] = [
        ("fixed f*: best g_theta", lambda row: _is_known_f_row(row)),
        (
            "best exact-row f_theta,g_theta",
            lambda row: str(row.get("objective_mode", "")) == "exact_rows" and not _is_known_f_row(row),
        ),
        (
            "best rollout/local-law f_theta,g_theta",
            lambda row: str(row.get("objective_mode", "")) == "rollout_local_law" and not _is_known_f_row(row),
        ),
    ]
    lines.extend(
        [
            "| context slice | best completed nontrivial cell | L | root MAE vs HLL reference | fraction of same-p HLL raw MAE | official-on-learned MAE |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for label, predicate in slices:
        row = _best_context_row(rows, predicate)
        if row is None:
            continue
        lines.append(
            f"| {label} | {row.get('cell', '')} | {_fmt_int(row.get('n_leaves'))} | "
            f"{_fmt(row.get('root_mae'))} | {_fmt(row.get('root_mae_over_fstar_root_raw_mae'))} | "
            f"{_fmt(row.get('official_f_on_learned_root_mae'))} |"
        )
    if math.isfinite(p8_mae):
        lines.extend(
            [
                "",
                f"The ratio plot is only an error-budget scale: for the current p=8 target, package HLL's own raw root MAE is `{_fmt(p8_mae)}` distinct-count units. A learned root MAE of `0.03` is about 1% of that HLL noise scale, but it does not mean the learned estimator is 1% wrong against truth. To make truth-level claims we should add learned-vs-true distinct-count metrics to future summaries.",
            ]
        )
    return lines


def _row_by_cell(rows: Sequence[Mapping[str, Any]], cell: str) -> Mapping[str, Any] | None:
    for row in rows:
        if str(row.get("cell", "")) == cell:
            return row
    return None


def _current_read_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    interesting = [
        ("exact g-first L16", "known_f_gfgf_exact_L16"),
        ("rollout dense L8", "known_f_gf_rollout_dense_L8"),
        ("rollout root-only L8", "known_f_gf_rollout_root_L8"),
    ]
    found: list[tuple[str, Mapping[str, Any]]] = []
    for label, cell in interesting:
        row = _row_by_cell(rows, cell)
        if row is not None and str(row.get("status", "")) == "completed":
            found.append((label, row))
    if not found:
        return []
    lines = [
        "## Current Read",
        "",
        "| result | cell | L | root MAE | official f* on learned root | merge MAE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, row in found:
        lines.append(
            f"| {label} | {row.get('cell', '')} | {_fmt_int(row.get('n_leaves'))} | "
            f"{_fmt(row.get('root_mae'))} | {_fmt(row.get('official_f_on_learned_root_mae'))} | "
            f"{_fmt(row.get('merge_state_mae'))} |"
        )
    dense = _row_by_cell(rows, "known_f_gf_rollout_dense_L8")
    root = _row_by_cell(rows, "known_f_gf_rollout_root_L8")
    exact_l16 = _row_by_cell(rows, "known_f_gfgf_exact_L16")
    notes: list[str] = []
    if exact_l16 is not None and str(exact_l16.get("status", "")) == "completed":
        notes.append(
            f"The exact g-first path is strong at L=16: root MAE `{_fmt(exact_l16.get('root_mae'))}` with official f* on the learned root `{_fmt(exact_l16.get('official_f_on_learned_root_mae'))}`."
        )
    if (
        dense is not None
        and root is not None
        and str(dense.get("status", "")) == "completed"
        and str(root.get("status", "")) == "completed"
    ):
        notes.append(
            f"At L=8, dense rollout remains good (`{_fmt(dense.get('root_mae'))}` root MAE) while root-only rollout fails (`{_fmt(root.get('root_mae'))}`), pointing to the rollout observation/sampling objective rather than basic recoverability."
        )
    if notes:
        lines.extend(["", *notes])
    return lines


def _load_fixed_mass_smoke_rows(root: Path = FIXED_MASS_SMOKE_ROOT) -> list[dict[str, Any]]:
    root_path = root if root.is_absolute() else REPO_ROOT / root
    if not root_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root_path.glob("R*/summary.csv")):
        summary = _read_one_csv(summary_path)
        if not summary:
            continue
        row: dict[str, Any] = {
            "grid": root_path.name,
            "cell": summary_path.parent.name,
            "status": "completed",
            "output_dir": str(summary_path.parent.resolve()),
            "summary_path": str(summary_path.resolve()),
            "precision": FNO_ARG_DEFAULTS["precision"],
            "universe_size": FNO_ARG_DEFAULTS["universe_size"],
            "min_tokens": FNO_ARG_DEFAULTS["min_tokens"],
            "max_tokens": FNO_ARG_DEFAULTS["max_tokens"],
            "zipf_alphas": FNO_ARG_DEFAULTS["zipf_alphas"],
            "seed": FNO_ARG_DEFAULTS["seed"],
        }
        row.update(summary)
        row["observation"] = _parse_observation(row.get("oracle_observation_mode") or row.get("observation"))
        stage_path = summary_path.parent / "hll_register_space" / "stage_metrics.json"
        if stage_path.exists():
            try:
                stage_payload = json.loads(stage_path.read_text(encoding="utf-8"))
            except Exception:
                stage_payload = []
            if isinstance(stage_payload, list) and stage_payload:
                final_stage = stage_payload[-1]
                if isinstance(final_stage, Mapping):
                    for key in (
                        "train_observed_rows_end",
                        "train_population_rows_end",
                        "train_root_observed_rows_end",
                        "train_root_population_rows_end",
                        "train_observed_mass_end",
                        "train_population_mass_end",
                    ):
                        if key in final_stage:
                            row[key] = final_stage[key]
        rows.append(_annotate_fstar_gstar(row))
    return sorted(rows, key=lambda row: _float(row.get("root_label_share")))


def _constant_observation_mass_lines(
    rows: Sequence[Mapping[str, Any]],
    smoke_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    budget_rows = [
        row
        for row in rows
        if "budget" in str(row.get("cell", ""))
        or str(row.get("observation", "")) in {"budgeted_mass", "fixed_mass"}
    ]
    if not budget_rows and not smoke_rows:
        return []

    lines = [
        "## Constant Observation Mass",
        "",
        "The HLL code path is present as `budgeted_mass`/`fixed_mass`: it fixes the amount of oracle supervision per document and allocates it between root and non-root node labels. This is the experiment family we need for the constant-observation-mass argument.",
        "",
    ]
    if budget_rows:
        counts: dict[str, int] = {}
        for row in budget_rows:
            status = str(row.get("status", "planned_no_summary"))
            counts[status] = counts.get(status, 0) + 1
        lines.extend(
            [
                "| budgeted-mass grid status | cells |",
                "|---|---:|",
                f"| completed | {counts.get('completed', 0)} |",
                f"| running | {counts.get('running', 0)} |",
                f"| planned/no summary | {counts.get('planned_no_summary', 0)} |",
                "",
            ]
        )
    if smoke_rows:
        lines.extend(
            [
                "The only completed HLL fixed-mass outputs currently available are CPU smoke tests, not publication-grade runs (`n_train=32`, `n_val=8`, one epoch, L=4). They are useful for checking wiring but not for the claim.",
                "",
                "| root label share | root MAE | official f* on learned root | merge MAE | observed mass | population mass | n_train | n_val | epochs |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in smoke_rows:
            lines.append(
                f"| {_fmt(row.get('root_label_share'))} | {_fmt(row.get('root_mae'))} | "
                f"{_fmt(row.get('official_f_on_learned_root_mae'))} | {_fmt(row.get('merge_state_mae'))} | "
                f"{_fmt(row.get('train_observed_mass_end'))} | {_fmt(row.get('train_population_mass_end'))} | "
                f"{_fmt_int(row.get('n_train'))} | {_fmt_int(row.get('n_val'))} | {_fmt_int(row.get('epochs_per_stage'))} |"
            )
        lines.extend(
            [
                "",
                "Readout: the constant-mass mechanism is ready, but the actual HLL evidence is not. The recoverability grid still has the budgeted-mass cells in planned/no-summary state, so the next useful run is a real detached R-grid with the same scale as the main HLL jobs plus dense/root controls.",
            ]
        )
    return lines


def _sampled_node_rate_grid_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    sample_rows = [
        row
        for row in rows
        if str(row.get("grid", "")).startswith("hll_sampled_node_rate_grid")
    ]
    if not sample_rows:
        return []

    lines = [
        "## Root Plus Random Non-Root Sampling",
        "",
        "This grid keeps the supplied-answer semantics fixed: `g_theta` rolls out states, each observed node predicts with the current package/formula readout `f_latest(g_theta node)`, and the scalar target is `f*(g* exact node)`. Roots have propensity 1; sampled non-root rows have propensity equal to the configured rate, and corrected rows use observed-over-propensity weighting.",
        "",
        "The primary metric is `official_f_on_learned_root_mae`, i.e. package `f*` applied to the learned root state. The main grid forces `state_loss_weight=0`, so exact hidden HLL states are not supplied as labels.",
        "",
        "| L | root rate | non-root rate | status | obs design | primary root MAE f*(g_theta) | merge-state root MAE | merge-state MAE | observed rows/doc | root rows/doc | non-root rows/doc | max IPW weight | effective sample size | output |",
        "|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    ordered = sorted(
        sample_rows,
        key=lambda row: (
            _float(row.get("n_leaves")),
            _sampled_node_rate(row) if math.isfinite(_sampled_node_rate(row)) else float("inf"),
            str(row.get("cell", "")),
        ),
    )
    for row in ordered:
        rate = _sampled_node_rate(row)
        root_rate = _float(row.get("root_label_share"))
        if not math.isfinite(root_rate):
            root_rate = 1.0 if str(row.get("observation", "")) in {"root_only", "sampled_nodes", "dense_oracle"} else float("nan")
        output = str(row.get("output_dir", ""))
        output_cell = Path(output).name if output else ""
        lines.append(
            f"| {_fmt_int(row.get('n_leaves'))} | {_sampled_rate_label(root_rate) if math.isfinite(root_rate) else ''} | "
            f"{_sampled_rate_label(rate) if math.isfinite(rate) else ''} | "
            f"{row.get('status', '')} | {row.get('observation', '')} | "
            f"{_fmt(row.get('official_f_on_learned_root_mae'))} | {_fmt(row.get('merge_state_root_mae'))} | "
            f"{_fmt(row.get('merge_state_mae'))} | {_fmt(row.get('train_observed_rows_per_doc_end'))} | "
            f"{_fmt(row.get('train_root_observed_rows_per_doc_end'))} | {_fmt(row.get('train_nonroot_observed_rows_per_doc_end'))} | "
            f"{_fmt(row.get('train_max_ipw_weight_end'))} | {_fmt(row.get('train_effective_sample_size_end'))} | {output_cell} |"
        )
    return lines


def _write_report(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    figure_names: Sequence[str],
    *,
    baseline_rows: Sequence[Mapping[str, Any]] = (),
    context_baseline_rows: Sequence[Mapping[str, Any]] = (),
    fixed_mass_smoke_rows: Sequence[Mapping[str, Any]] = (),
    token_filter: tuple[int, int] | None = None,
) -> None:
    completed = _sorted_completed(rows)
    by_grid: dict[str, dict[str, int]] = {}
    for row in rows:
        grid = str(row.get("grid", ""))
        status = str(row.get("status", ""))
        by_grid.setdefault(grid, {})
        by_grid[grid][status] = by_grid[grid].get(status, 0) + 1

    known_rows = [
        row
        for row in completed
        if _is_known_f_row(row)
        and str(row.get("cell", "")).startswith("known_f_g_exact")
    ]
    exact_hll = [
        row
        for row in completed
        if str(row.get("target_kind", "hll_register_space")) == "hll_register_space"
        and str(row.get("objective_mode", "")) == "exact_rows"
    ]
    best_exact = sorted(exact_hll, key=lambda row: _float(row.get("root_rel_mae")))[:12]
    rollout = [row for row in completed if str(row.get("objective_mode", "")) == "rollout_local_law"]

    lines = [
        "# HLL FNO Progress Report",
        "",
        f"Generated: `{_utc_now()}`",
        "",
        "This is a partial live aggregation. Rows without summaries are kept so the running grids have visible coverage.",
        "",
        "`classical HLL` here means exact native HLL leaf states, the package HLL max-register merge `g*`, and the package HLL readout `f*`. "
        "For rows trained against `hll_reference`, the classical-HLL target error is exactly zero because the target is the package HLL reference itself. "
        "That zero is not a normalization artifact. The classical-HLL root/raw leaf-count figure shows the non-normalized package-HLL root MAE in distinct-count units, with one line per HLL precision.",
        "For learned rows, `root MAE` and `root rel MAE` are approximation error to the HLL target unless explicitly labeled `vs true count`; L=1 is a no-merge canary.",
        "`p` is HLL precision bits, so p=16 means 65,536 registers. p=32 would require 2^32 registers per sketch and is not materialized here; if you meant register counts 4/8/16/32, that is a different axis.",
        "The overview figures use the main HLL series for readability; the full row set is preserved in the tables and CSV.",
        "",
    ]
    lines.extend(_token_regime_lines(rows, token_filter=token_filter))
    lines.extend(
        [
            "",
            "## Grid Status",
            "",
            "| grid | completed | running | planned/no summary | total |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for grid, counts in sorted(by_grid.items()):
        completed_n = counts.get("completed", 0)
        running_n = counts.get("running", 0)
        planned_n = counts.get("planned_no_summary", 0)
        lines.append(f"| {grid} | {completed_n} | {running_n} | {planned_n} | {completed_n + running_n + planned_n} |")

    lines.extend(["", "## Figures", ""])
    for name in figure_names:
        lines.append(f"![{Path(name).stem}](figures/{name})")
        caption = FIGURE_CAPTIONS.get(name)
        if caption:
            lines.append("")
            lines.append(caption)
        lines.append("")

    lines.extend(_terminology_lines())
    lines.append("")

    canonical_lines = _canonical_summary_lines(rows)
    if canonical_lines:
        lines.extend(canonical_lines)
        lines.append("")

    lines.extend(_context_summary_lines(rows, context_baseline_rows or baseline_rows))
    lines.append("")

    current_read = _current_read_lines(rows)
    if current_read:
        lines.extend(current_read)
        lines.append("")

    sampled_rate_lines = _sampled_node_rate_grid_lines(rows)
    if sampled_rate_lines:
        lines.extend(sampled_rate_lines)
        lines.append("")

    fixed_mass_lines = _constant_observation_mass_lines(rows, fixed_mass_smoke_rows)
    if fixed_mass_lines:
        lines.extend(fixed_mass_lines)
        lines.append("")

    lines.extend(["## Classical HLL (f* after g*) Baseline", ""])
    fstar_table = _fstar_summary_table(baseline_rows or rows)
    if len(fstar_table) > 2:
        lines.extend(fstar_table)
    else:
        lines.append("No computable classical-HLL package baseline rows found.")
    lines.append("")

    lines.extend(["## Known f* G-Only", ""])
    if known_rows:
        lines.extend(_md_table(known_rows))
        lines.append("")
        lines.append(
            "For these rows, `official-on-learned MAE` is the exact/reference HLL readout on the learned root state. "
            "When that is zero and `f exact MAE` is near zero, the remaining root MAE is concentrated in the differentiable/formula readout path and the register-state merge residual."
        )
    else:
        lines.append("No completed known-f g-only rows found.")

    lines.extend(["", "## Best Exact-Row HLL Cells", ""])
    if best_exact:
        lines.extend(_md_table(best_exact))
    else:
        lines.append("No completed exact-row HLL rows found.")

    lines.extend(["", "## Rollout / Local-Law Cells", ""])
    if rollout:
        lines.extend(_md_table(sorted(rollout, key=lambda row: (_float(row.get("n_leaves")), str(row.get("cell", ""))))))
    else:
        lines.append("No completed rollout-local-law rows found.")

    lines.extend(["", "## All Cells", ""])
    all_rows = sorted(rows, key=lambda row: (str(row.get("grid", "")), str(row.get("status", "")), str(row.get("cell", ""))))
    lines.extend(_md_table(all_rows))
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="*",
        type=Path,
        default=None,
        help="Grid roots to aggregate. Defaults include historical roots plus existing hll_canonical_observation_grid_* roots.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--token-count", type=int, default=None, help="Filter to fixed min=max tokens/document.")
    parser.add_argument("--min-tokens", type=int, default=None, help="Filter report rows to this minimum token count.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Filter report rows to this maximum token count.")
    args = parser.parse_args(argv)

    token_filter: tuple[int, int] | None = None
    if args.token_count is not None:
        if int(args.token_count) <= 0:
            raise ValueError("--token-count must be positive")
        for name in ("min_tokens", "max_tokens"):
            value = getattr(args, name)
            if value is not None and int(value) != int(args.token_count):
                raise ValueError("--token-count conflicts with explicit --min-tokens/--max-tokens")
        token_filter = (int(args.token_count), int(args.token_count))
    elif args.min_tokens is not None or args.max_tokens is not None:
        if args.min_tokens is None or args.max_tokens is None:
            raise ValueError("pass both --min-tokens and --max-tokens, or use --token-count for fixed mass")
        if int(args.min_tokens) <= 0 or int(args.max_tokens) <= 0:
            raise ValueError("--min-tokens/--max-tokens must be positive")
        token_filter = (int(args.min_tokens), int(args.max_tokens))

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "outputs" / f"hll_fno_progress_report_{_utc_stamp()}"
    output_dir = output_dir.resolve()
    fig_dir = output_dir / "figures"
    roots = _default_roots() if args.roots is None else list(args.roots)
    rows = _discover_rows(roots)
    if token_filter is not None:
        rows = _filter_token_regime(rows, min_tokens=token_filter[0], max_tokens=token_filter[1])
    rows = sorted(rows, key=lambda row: (str(row.get("grid", "")), str(row.get("cell", ""))))
    baseline_rows = _baseline_precision_sweep_rows(rows)
    context_baseline_rows = _baseline_precision_sweep_rows(rows, precisions=CONTEXT_BASELINE_PRECISIONS)
    fixed_mass_smoke_rows = _load_fixed_mass_smoke_rows()
    if token_filter is not None:
        fixed_mass_smoke_rows = _filter_token_regime(
            fixed_mass_smoke_rows,
            min_tokens=token_filter[0],
            max_tokens=token_filter[1],
        )

    _write_csv(output_dir / "hll_fno_progress_rows.csv", rows)
    _write_csv(output_dir / "hll_fstar_gstar_precision_sweep.csv", baseline_rows)
    _write_csv(output_dir / "hll_capacity_context_baselines.csv", context_baseline_rows)
    if fixed_mass_smoke_rows:
        _write_csv(output_dir / "hll_fixed_mass_smoke_rows.csv", fixed_mass_smoke_rows)
    figures = [
        "grid_status.png",
        "root_rel_mae_by_leaves.png",
        "root_mae_by_leaves.png",
        "root_mae_with_hll_line_by_leaves.png",
        "canonical_root_mae_with_classical_hll_by_leaves.png",
    ]
    _plot_status(rows, fig_dir / figures[0])
    _plot_metric_by_leaves(
        rows,
        fig_dir / figures[1],
        metric="root_rel_mae",
        title="Relative approximation error to classical HLL target",
        ylabel="relative MAE to HLL target",
        show_target_optimum=True,
    )
    _plot_metric_by_leaves(
        rows,
        fig_dir / figures[2],
        metric="root_mae",
        title="Root approximation MAE to classical HLL target",
        ylabel="MAE to HLL target",
        show_target_optimum=True,
    )
    _plot_root_mae_with_hll_line(rows, baseline_rows, fig_dir / figures[3], precision=8)
    _plot_canonical_root_mae_with_classical_hll(rows, baseline_rows, fig_dir / figures[4], precision=8)
    expanded_components = "canonical_error_components_by_leaves.png"
    if _plot_canonical_error_components(rows, fig_dir / expanded_components):
        figures.append(expanded_components)
    figures.extend(
        [
            "merge_state_mae_by_leaves.png",
            "root_mae_context_ratio_by_leaves.png",
            "fstar_gstar_root_raw_mae_by_leaves.png",
            "known_f_g_residuals.png",
            "fstar_gtheta_rollout_dense_vs_root_by_leaves.png",
        ]
    )
    _plot_metric_by_leaves(rows, fig_dir / "merge_state_mae_by_leaves.png", metric="merge_state_mae", title="Register-state merge residual by leaves", ylabel="merge-state MAE")
    _plot_context_ratio_by_leaves(rows, fig_dir / "root_mae_context_ratio_by_leaves.png")
    _plot_fstar_root_by_leaves(baseline_rows, fig_dir / "fstar_gstar_root_raw_mae_by_leaves.png")
    _plot_known_f_g(rows, fig_dir / "known_f_g_residuals.png")
    _plot_fstar_gtheta_rollout_dense_vs_root(rows, fig_dir / "fstar_gtheta_rollout_dense_vs_root_by_leaves.png")
    sampled_rate_fig = "fstar_gtheta_sampled_rate_by_leaves.png"
    if _plot_fstar_gtheta_sampled_rate(rows, fig_dir / sampled_rate_fig):
        figures.append(sampled_rate_fig)
    _write_report(
        output_dir / "report.md",
        rows,
        figures,
        baseline_rows=baseline_rows,
        context_baseline_rows=context_baseline_rows,
        fixed_mass_smoke_rows=fixed_mass_smoke_rows,
        token_filter=token_filter,
    )
    print(f"wrote {output_dir / 'report.md'}")
    print(f"wrote {output_dir / 'hll_fno_progress_rows.csv'}")
    print(f"wrote {output_dir / 'hll_fstar_gstar_precision_sweep.csv'}")
    print(f"wrote {output_dir / 'hll_capacity_context_baselines.csv'}")
    if fixed_mass_smoke_rows:
        print(f"wrote {output_dir / 'hll_fixed_mass_smoke_rows.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
