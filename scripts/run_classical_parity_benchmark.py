"""Classical + learned HLL parity benchmark — routes every cell through `fit()`.

Usage:

    python scripts/run_classical_parity_benchmark.py --out outputs/classical_parity

Each cell issues one `fit(trainer_config=<preset>(...))` call and writes its
`FitResult` into a CSV row. Cells are independent, so the driver runs them in
a `ProcessPoolExecutor` (one worker per CPU core by default).

Three methods can participate, keyed by the `method` CSV column:

- `classical_native` / `classical_datasketches` — no optimization; classical
  `g` (register-wise max / HLL union) and classical `f` (HLL estimate formula).
- `learned_g` — learned leaf encoder + learned merge `g` with state dim =
  2^precision and classical HLL estimator as `f`.
- `learned_g_oracle_state` — official native-HLL leaf register states and
  fixed classical `f`; only the merge operator is learned.
- `learned_joint` — fully end-to-end learned; `g` and `f` are both MLPs.

Learned cells use a smaller sweep grid by default since each is a full training
run.

After the CSV is written, the companion report generator produces the figure
and LaTeX table for Appendix F.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    lane_src = repo_root / "parallel" / "unified_g_v1" / "src"
    for p in (str(lane_src), str(repo_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_repo_on_path()


# ---------------------------------------------------------------------------
# Shared CSV schema. All cells land in the same summary.csv.
# ---------------------------------------------------------------------------

METRIC_COLUMNS = (
    "count",
    "val_mae",
    "root_mae",
    "root_rmse",
    "root_rel_mae",
    "c1_mae",
    "c3_mae",
    "merge_state_mae",
    "flat_vs_tree_abs_mean",
    "flat_vs_tree_abs_max",
    "flat_vs_tree_rel_mean",
    "state_equal_rate",
    "state_bytes_equal_rate",
    "tree_wall_ms_mean",
    "flat_wall_ms_mean",
    "memory_bytes_mean",
    "hll_rse_theory",
    "total_wall_seconds",
)

COLUMNS = (
    "method",
    "backend",
    "precision",
    "n_leaves",
    "schedule",
    "seed",
    "oracle_kind",
    "n_val",
    "embedding_dim",
    "summary_dim",
    "state_dim",
    "hidden_dim",
    *METRIC_COLUMNS,
)


def _write_canonical_parity_sidecars(
    *,
    output_root: Path,
    summary_path: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    from src.experiments import (
        ARTIFACT_SUMMARY_CSV,
        ResultRow,
        benchmark_ref_from_parts,
        experiment_method_ref,
        oracle_ref,
        role_ref,
        state_model_role_ref,
        write_canonical_sidecars,
    )

    benchmark_ref = benchmark_ref_from_parts(
        family="hll_parity",
        scope="classical_parity",
        name="Classical and learned HLL parity",
        metadata={
            "precisions": list(args.precisions),
            "leaf_counts": list(args.leaf_counts),
            "learned_precisions": list(args.learned_precisions),
            "learned_leaf_counts": list(args.learned_leaf_counts),
        },
    )
    method_refs = {
        method_name: experiment_method_ref(
            family=str(method_name),
            variant="fit_sweep",
            adapter="classical_parity_benchmark",
            roles={
                "scorer": role_ref(
                    role="scorer",
                    surface="native",
                    engine="python",
                    model=str(method_name),
                ),
                **(
                    {
                        "state_model": state_model_role_ref(
                            engine="pytorch",
                            model=str(method_name),
                            execution_mode="fit",
                        )
                    }
                    if str(method_name).startswith("learned_")
                    else {}
                ),
            },
            oracle=oracle_ref(kind="analytic_or_reference", source="hll_parity"),
            metadata={"method": str(method_name)},
        )
        for method_name in sorted({str(row.get("method", "") or "") for row in rows})
        if method_name
    }
    result_rows: list[ResultRow] = []
    for row in rows:
        method_name = str(row.get("method", "") or "")
        method_ref = method_refs.get(method_name)
        if method_ref is None:
            continue
        seed = None
        try:
            seed = int(row.get("seed"))
        except Exception:
            seed = None
        base_metadata = {
            "backend": row.get("backend", ""),
            "precision": row.get("precision", ""),
            "n_leaves": row.get("n_leaves", ""),
            "schedule": row.get("schedule", ""),
            "oracle_kind": row.get("oracle_kind", ""),
        }
        for metric_name in METRIC_COLUMNS:
            value = row.get(metric_name, "")
            if value == "":
                continue
            result_rows.append(
                ResultRow(
                    experiment_id="",
                    phase="eval",
                    benchmark_ref=benchmark_ref,
                    method_ref=method_ref,
                    split="validation",
                    seed=seed,
                    metric_name=str(metric_name),
                    metric_value=value,
                    artifact_refs=(ARTIFACT_SUMMARY_CSV,),
                    metadata=base_metadata,
                )
            )
    write_canonical_sidecars(
        output_root,
        title="classical_parity_benchmark",
        adapter_id="classical_parity_benchmark",
        benchmark_refs=(benchmark_ref,),
        method_refs=tuple(method_refs.values()),
        phases=("fit", "eval", "report"),
        artifacts={ARTIFACT_SUMMARY_CSV: str(summary_path)},
        result_rows=tuple(result_rows),
        state="completed",
        metadata={"total_cells": len(rows)},
        launch_command=tuple(sys.argv),
        report_profiles=("runtime_eval_summary",),
    )


@dataclass(frozen=True)
class ClassicalCell:
    backend: str
    oracle_kind: str
    precision: int
    n_leaves: int
    schedule: str
    seed: int
    n_val: int
    min_tokens: int
    max_tokens: int
    universe_size: int
    out_root: str

    @property
    def method(self) -> str:
        return f"classical_{self.backend}"

    def cell_dir(self) -> Path:
        return Path(self.out_root) / (
            f"{self.method}_{self.oracle_kind}_p{self.precision}_"
            f"L{self.n_leaves}_{self.schedule}_seed{self.seed}"
        )


@dataclass(frozen=True)
class LearnedCell:
    method: str  # "learned_g", "learned_g_oracle_state", or "learned_joint"
    oracle_kind: str
    precision: int
    n_leaves: int
    seed: int
    n_train: int
    n_val: int
    min_tokens: int
    max_tokens: int
    universe_size: int
    n_epochs: int
    train_batch_size: int
    learning_rate: float
    local_law_weight: float
    merge_state_relative_weight: float
    embedding_dim: int | None
    summary_dim: int | None
    state_dim: int | None
    hidden_dim: int | None
    use_cuda: bool
    cuda_device: int | None
    out_root: str

    def cell_dir(self) -> Path:
        return Path(self.out_root) / (
            f"{self.method}_{self.oracle_kind}_p{self.precision}_"
            f"L{self.n_leaves}_seed{self.seed}"
        )


# ---------------------------------------------------------------------------
# Workers — one per method family.
# ---------------------------------------------------------------------------


def _worker_env_init() -> None:
    # Pin BLAS threads so N workers scale linearly on CPU.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _run_classical_cell(cell: ClassicalCell) -> dict[str, Any]:
    _ensure_repo_on_path()
    _worker_env_init()
    from unified_g_v1.sketch.classical_parity import classical_hll_parity_task
    from unified_g_v1.training.fit import fit

    cfg = classical_hll_parity_task(
        precision=int(cell.precision),
        n_leaves=int(cell.n_leaves),
        schedule=cell.schedule,
        backend=cell.backend,
        n_val=int(cell.n_val),
        seed=int(cell.seed),
        universe_size=int(cell.universe_size),
        min_tokens=int(cell.min_tokens),
        max_tokens=int(cell.max_tokens),
        oracle_kind=cell.oracle_kind,
    )
    result = fit(trainer_config=cfg, output_dir=cell.cell_dir())
    row: dict[str, Any] = {
        "method": cell.method,
        "backend": cell.backend,
        "precision": int(cell.precision),
        "n_leaves": int(cell.n_leaves),
        "schedule": cell.schedule,
        "seed": int(cell.seed),
        "oracle_kind": cell.oracle_kind,
        "n_val": int(cell.n_val),
    }
    for k in METRIC_COLUMNS:
        row[k] = result.metrics.get(k, "")
    return row


def _run_learned_cell(cell: LearnedCell) -> dict[str, Any]:
    _ensure_repo_on_path()
    _worker_env_init()
    from unified_g_v1.sketch.learned_hll_parity import learned_hll_parity_task
    from unified_g_v1.training.fit import fit

    t0 = time.perf_counter()
    cfg = learned_hll_parity_task(
        method=cell.method,
        precision=int(cell.precision),
        n_leaves=int(cell.n_leaves),
        n_train=int(cell.n_train),
        n_val=int(cell.n_val),
        seed=int(cell.seed),
        universe_size=int(cell.universe_size),
        min_tokens=int(cell.min_tokens),
        max_tokens=int(cell.max_tokens),
        oracle_kind=cell.oracle_kind,
        n_epochs=int(cell.n_epochs),
        train_batch_size=int(cell.train_batch_size),
        learning_rate=float(cell.learning_rate),
        local_law_weight=float(cell.local_law_weight),
        merge_state_relative_weight=float(cell.merge_state_relative_weight),
        embedding_dim=cell.embedding_dim,
        summary_dim=cell.summary_dim,
        state_dim=cell.state_dim,
        hidden_dim=cell.hidden_dim,
        use_cuda=bool(cell.use_cuda),
        cuda_device=cell.cuda_device,
    )
    result = fit(trainer_config=cfg, output_dir=cell.cell_dir())
    total_wall = float(time.perf_counter() - t0)

    # The learned path's metrics live in the last history entry; top-level
    # FitResult.metrics only surfaces val_mae_raw / best_metric_value.
    last = result.history[-1] if result.history else {}

    def _num(key: str, default: float = 0.0) -> float:
        val = last.get(key, result.metrics.get(key, default))
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    val_mae = _num("val_mae_raw")
    rel_mae = _num("root_rel_mae")
    c1 = _num("c1_mae")
    c3 = _num("c3_mae")
    merge_state = _num("merge_state_mae", default="")
    root_rmse = _num("root_rmse")

    row: dict[str, Any] = {
        "method": cell.method,
        "backend": cell.method,  # carry as backend for CSV column parity
        "precision": int(cell.precision),
        "n_leaves": int(cell.n_leaves),
        "schedule": "balanced",
        "seed": int(cell.seed),
        "oracle_kind": cell.oracle_kind,
        "n_val": int(cell.n_val),
        "embedding_dim": int(cfg.extra.get("embedding_dim", 0)),
        "summary_dim": int(cfg.extra.get("summary_dim", 0)),
        "state_dim": int(cfg.extra.get("state_dim", 0)),
        "hidden_dim": int(cfg.extra.get("hidden_dim", 0)),
        "count": int(cell.n_val),
        "val_mae": val_mae,
        "root_mae": val_mae,
        "root_rmse": root_rmse,
        "root_rel_mae": rel_mae,
        "c1_mae": c1,
        "c3_mae": c3,
        "merge_state_mae": merge_state,
        # The learned path doesn't compute flat_vs_tree deltas directly; we
        # leave those blank so the report can distinguish learned from
        # classical rows by column population.
        "flat_vs_tree_abs_mean": "",
        "flat_vs_tree_abs_max": "",
        "flat_vs_tree_rel_mean": "",
        "state_equal_rate": "",
        "state_bytes_equal_rate": "",
        "tree_wall_ms_mean": "",
        "flat_wall_ms_mean": "",
        "memory_bytes_mean": "",
        "hll_rse_theory": 1.04 / (2 ** (int(cell.precision) * 0.5)),
        "total_wall_seconds": total_wall,
    }
    return row


# ---------------------------------------------------------------------------
# Grid assembly.
# ---------------------------------------------------------------------------


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _build_classical_cells(args, out_hll: Path) -> list[ClassicalCell]:
    backends: list[str] = []
    for backend in args.backends:
        if backend == "datasketches":
            try:
                import datasketches  # noqa: F401
            except ImportError:
                print("skipping backend=datasketches: not installed")
                continue
        backends.append(backend)
    cells: list[ClassicalCell] = []
    for backend in backends:
        for oracle_kind in args.oracle_kinds:
            for precision in args.precisions:
                for n_leaves in args.leaf_counts:
                    for schedule in args.schedules:
                        for seed in args.seeds:
                            cells.append(
                                ClassicalCell(
                                    backend=backend,
                                    oracle_kind=oracle_kind,
                                    precision=int(precision),
                                    n_leaves=int(n_leaves),
                                    schedule=schedule,
                                    seed=int(seed),
                                    n_val=int(args.n_val),
                                    min_tokens=int(args.min_tokens),
                                    max_tokens=int(args.max_tokens),
                                    universe_size=int(args.universe_size),
                                    out_root=str(out_hll),
                                )
                            )
    return cells


def _build_learned_cells(args, out_hll: Path) -> list[LearnedCell]:
    cells: list[LearnedCell] = []
    methods_requested: list[str] = []
    for method in args.methods:
        canonical = "learned_joint" if method == "learned_fg" else str(method)
        if canonical in ("learned_g", "learned_g_oracle_state", "learned_joint") and canonical not in methods_requested:
            methods_requested.append(canonical)
    if not methods_requested:
        return cells
    for method in methods_requested:
        for oracle_kind in args.learned_oracle_kinds:
            for precision in args.learned_precisions:
                for n_leaves in args.learned_leaf_counts:
                    for seed in args.learned_seeds:
                        cells.append(
                            LearnedCell(
                                method=method,
                                oracle_kind=oracle_kind,
                                precision=int(precision),
                                n_leaves=int(n_leaves),
                                seed=int(seed),
                                n_train=int(args.learned_n_train),
                                n_val=int(args.learned_n_val),
                                min_tokens=int(args.learned_min_tokens),
                                max_tokens=int(args.learned_max_tokens),
                                universe_size=int(args.learned_universe_size),
                                n_epochs=int(args.learned_n_epochs),
                                train_batch_size=int(args.learned_batch_size),
                                learning_rate=float(args.learned_lr),
                                local_law_weight=float(args.learned_local_law_weight),
                                merge_state_relative_weight=float(args.learned_merge_state_weight),
                                embedding_dim=args.learned_embedding_dim,
                                summary_dim=args.learned_summary_dim,
                                state_dim=args.learned_state_dim,
                                hidden_dim=args.learned_hidden_dim,
                                use_cuda=bool(args.learned_use_cuda),
                                cuda_device=args.learned_cuda_device,
                                out_root=str(out_hll),
                            )
                        )
    return cells


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Classical + learned HLL parity sweep via fit()")
    ap.add_argument("--out", type=Path, default=Path("outputs/classical_parity"))
    ap.add_argument(
        "--methods",
        type=_parse_str_list,
        default="classical,learned_g,learned_g_oracle_state,learned_joint",
        help="Methods to sweep: any of 'classical', 'learned_g', "
             "'learned_g_oracle_state', 'learned_joint'. 'learned_fg' is accepted "
             "as a legacy alias.",
    )
    # Classical grid (fast, big).
    ap.add_argument("--precisions", type=_parse_int_list, default="7,9,11,13")
    ap.add_argument("--leaf-counts", type=_parse_int_list, default="1,2,4,8,16")
    ap.add_argument("--schedules", type=_parse_str_list, default="balanced")
    ap.add_argument("--backends", type=_parse_str_list, default="native,datasketches")
    ap.add_argument("--seeds", type=_parse_int_list, default="0,1,2")
    ap.add_argument("--oracle-kinds", type=_parse_str_list, default="analytic,hll_reference")
    ap.add_argument("--n-val", type=int, default=48)
    ap.add_argument("--min-tokens", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--universe-size", type=int, default=100_000)
    # Learned grid — defaults now match the classical grid (precision and
    # leaf-count axes) so paper-figure panels align across all methods.
    ap.add_argument("--learned-precisions", type=_parse_int_list, default="7,9,11,13")
    ap.add_argument("--learned-leaf-counts", type=_parse_int_list, default="1,2,4,8,16")
    ap.add_argument("--learned-seeds", type=_parse_int_list, default="0,1,2")
    ap.add_argument("--learned-oracle-kinds", type=_parse_str_list, default="hll_reference")
    ap.add_argument("--learned-n-train", type=int, default=128)
    ap.add_argument("--learned-n-val", type=int, default=48)
    ap.add_argument("--learned-min-tokens", type=int, default=128)
    ap.add_argument("--learned-max-tokens", type=int, default=512)
    ap.add_argument("--learned-universe-size", type=int, default=10_000)
    ap.add_argument("--learned-n-epochs", type=int, default=150)
    ap.add_argument("--learned-batch-size", type=int, default=16)
    ap.add_argument("--learned-lr", type=float, default=1e-3)
    ap.add_argument(
        "--learned-embedding-dim",
        type=int,
        default=None,
        help="Token embedding width for learned HLL cells. Default: 2 * max leaf tokens for the cell.",
    )
    ap.add_argument(
        "--learned-summary-dim",
        type=int,
        default=None,
        help="Leaf summary width for learned HLL cells. Default: learned embedding width.",
    )
    ap.add_argument(
        "--learned-state-dim",
        type=int,
        default=None,
        help="State width for learned f+g cells. Default: 2 * summary_dim.",
    )
    ap.add_argument(
        "--learned-hidden-dim",
        type=int,
        default=None,
        help="Hidden width for learned HLL cells. Default: max(128, 2 * summary_dim).",
    )
    ap.add_argument(
        "--learned-local-law-weight",
        type=float,
        default=0.9,
        help="Weight on local-law losses for learned HLL methods.",
    )
    ap.add_argument(
        "--learned-merge-state-weight",
        type=float,
        default=100.0,
        help="Extra register-state target weight for learned_g_oracle_state.",
    )
    ap.add_argument("--learned-use-cuda", action="store_true", help="Train learned HLL cells on CUDA.")
    ap.add_argument("--learned-cuda-device", type=int, default=None, help="CUDA device index for learned HLL cells.")
    ap.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("paper/ctreepo/tables"),
        help="Directory for generated report tables.",
    )
    # Parallelism.
    ap.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Parallel workers (ProcessPoolExecutor). 0 = os.cpu_count(); 1 = sequential.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print the grid without running.")
    args = ap.parse_args(argv)

    out_hll = args.out / "hll"
    out_hll.mkdir(parents=True, exist_ok=True)
    summary_path = out_hll / "summary.csv"

    # Assemble cells.
    classical_cells: list[ClassicalCell] = []
    if "classical" in args.methods:
        classical_cells = _build_classical_cells(args, out_hll)
    learned_cells = _build_learned_cells(args, out_hll)
    print(
        f"classical-HLL parity sweep: {len(classical_cells)} classical + "
        f"{len(learned_cells)} learned = {len(classical_cells) + len(learned_cells)} cells → "
        f"{summary_path}"
    )

    if args.dry_run:
        for c in classical_cells:
            print(f"  [classical] {asdict(c)}")
        for c in learned_cells:
            print(f"  [learned]   {asdict(c)}")
        return 0

    total = len(classical_cells) + len(learned_cells)
    if total == 0:
        print("no cells to run")
        return 0
    jobs = int(args.jobs) if int(args.jobs) > 0 else (os.cpu_count() or 1)
    jobs = min(jobs, max(1, total))
    if bool(args.learned_use_cuda) and jobs > 1 and learned_cells:
        print("warning: --learned-use-cuda with --jobs > 1 launches multiple CUDA workers")
    print(f"running {total} cells with {jobs} parallel workers")

    rows: list[dict[str, Any]] = []
    completed = 0
    progress_step = max(1, total // 20)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futures: dict[Any, str] = {}
        for cell in classical_cells:
            futures[ex.submit(_run_classical_cell, cell)] = f"classical::{cell.cell_dir().name}"
        for cell in learned_cells:
            futures[ex.submit(_run_learned_cell, cell)] = f"learned::{cell.cell_dir().name}"
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as exc:  # pragma: no cover
                print(f"  FAILED {futures[fut]}: {exc!r}")
                raise
            completed += 1
            if completed % progress_step == 0 or completed == total:
                print(f"  {completed}/{total} done")

    rows.sort(
        key=lambda r: (
            r["method"],
            r["oracle_kind"],
            int(r["precision"]),
            int(r["n_leaves"]),
            r.get("schedule", ""),
            int(r["seed"]),
        )
    )

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {summary_path} ({len(rows)} rows)")
    _write_canonical_parity_sidecars(
        output_root=out_hll,
        summary_path=summary_path,
        rows=rows,
        args=args,
    )

    try:
        from unified_g_v1.sketch.classical_parity_report import main as report_main
    except ImportError:
        print("skipping report generation (matplotlib missing?)")
        return 0

    report_main(
        [
            "--summary",
            str(summary_path),
            "--out-dir",
            str(out_hll),
            "--tables-dir",
            str(args.tables_dir),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
