#!/usr/bin/env python3
"""Run the broad classical-sketch paper bundle and stage paper assets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.contracts import (  # noqa: E402
    LOCAL_LAW_ESTIMATOR_ORACLE_STATE,
    objective_metadata,
    run_manifest_metadata,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    except Exception:
        return None
    return out.decode("utf-8", errors="replace").strip() or None


def _run(cmd: list[str], *, env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def _audit_tree_bundle_output(out_root: Path, *, env: dict[str, str]) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/audit_tree_bundle_contracts.py",
        str(out_root),
        "--require-tree-bundle",
        "--expected-domain",
        "classical_sketch",
        "--expected-leaf-unit",
        "stream_item",
    ]
    _run(cmd, env=env)
    return {
        "status": "passed",
        "expected_domain": "classical_sketch",
        "expected_leaf_unit": "stream_item",
    }


def _copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _count_rows(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    learned = sum(1 for row in rows if row.get("implementation_status") == "learned_empirical")
    return len(rows), learned


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def _uses_target_parallelism(value: object) -> bool:
    text = str(value).strip().lower()
    if text == "auto":
        return True
    try:
        return int(text) > 1
    except ValueError:
        return False


def _stage_assets(report_dir: Path) -> dict[str, Any]:
    assets_root = REPO_ROOT / "paper" / "ctreepo" / "assets" / "sketches"
    figures_dir = assets_root / "figures"
    tables_dir = assets_root / "tables"
    staged: dict[str, Any] = {"figures": [], "tables": []}
    paper_stems = {
        "classical_sketches_summary",
        "classical_sketches_hll_leaf_size",
        "classical_sketches_distinct_detail",
        "classical_sketches_frequency_detail",
        "classical_sketches_quantile_detail",
        "classical_sketches_sampling_detail",
        "classical_sketches_set_detail",
        "learned_sketch_leaf_size_diagnostic",
    }
    archive_dir = figures_dir / "archive_legacy"
    if figures_dir.exists():
        for old in sorted(figures_dir.glob("*")):
            if not old.is_file() or old.suffix.lower() not in {".png", ".pdf"}:
                continue
            if old.stem in paper_stems:
                continue
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(archive_dir / old.name))
    for src in sorted(
        list(report_dir.glob("classical_sketches_*.png"))
        + list(report_dir.glob("classical_sketches_*.pdf"))
        + list(report_dir.glob("learned_sketch_*.png"))
        + list(report_dir.glob("learned_sketch_*.pdf"))
    ):
        dst = figures_dir / src.name
        copied = _copy_if_exists(src, dst)
        if copied:
            staged["figures"].append(copied)
    for name in (
        "classical_sketches_grid.md",
        "classical_sketches_grid.tex",
        "classical_sketches_compact.md",
        "classical_sketches_compact.tex",
        "classical_sketches_aggregate.csv",
        "classical_sketches_aggregate.json",
        "classical_sketches_report.md",
        "classical_sketches_figure_manifest.json",
    ):
        copied = _copy_if_exists(report_dir / name, tables_dir / name)
        if copied:
            staged["tables"].append(copied)
    copied = _copy_if_exists(
        report_dir / "classical_sketches_figure_manifest.json",
        figures_dir / "classical_sketches_figure_manifest.json",
    )
    if copied:
        staged["figures"].append(copied)
    return staged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--capacities", type=str, default="small,medium,large")
    parser.add_argument("--leaf-counts", type=str, default=None)
    parser.add_argument("--leaf-sizes", type=str, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-learned", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learned-targets", type=str, default="all")
    parser.add_argument("--learned-variants", type=str, default="g,fg")
    parser.add_argument("--learned-readout-archs", type=str, default="structured")
    parser.add_argument("--learned-epochs", type=int, default=150)
    parser.add_argument("--learned-n-train", type=int, default=8192)
    parser.add_argument("--learned-n-val", type=int, default=1024)
    parser.add_argument("--learned-batch-size", type=int, default=1024)
    parser.add_argument("--learned-target-jobs", type=str, default="auto")
    parser.add_argument("--learned-gpu-ids", type=str, default="auto")
    parser.add_argument("--learned-batch-reference-leaf-size", type=int, default=128)
    parser.add_argument("--learned-max-batch-size", type=int, default=8192)
    parser.add_argument("--learned-eval-every", type=int, default=25)
    parser.add_argument(
        "--stage-assets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy generated report assets into paper/ctreepo/assets/sketches.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help=(
            "Directory for report table side outputs. Defaults to the paper tables "
            "directory when staging assets, otherwise to <out-root>/reports/classical_sketches/tables."
        ),
    )
    parser.add_argument(
        "--learned-local-label-rates",
        type=str,
        default=None,
        help="Comma-separated R-grid rates applied to both learned leaf and internal local labels.",
    )
    parser.add_argument(
        "--learned-leaf-query-rates",
        type=str,
        default=None,
        help="Comma-separated learned leaf-label rates; overrides --learned-local-label-rates for leaves.",
    )
    parser.add_argument(
        "--learned-root-query-rates",
        type=str,
        default=None,
        help=(
            "Comma-separated learned document-root label rates. Defaults to 1.0 "
            "for separate axes and to the node R for uniform_all_nodes."
        ),
    )
    parser.add_argument(
        "--learned-internal-query-rates",
        type=str,
        default=None,
        help="Comma-separated learned internal-node label rates; overrides --learned-local-label-rates for internal nodes.",
    )
    parser.add_argument(
        "--learned-supervision-sampling-policy",
        choices=("separate_axes", "uniform_all_nodes"),
        default="separate_axes",
        help="How learned local labels are sampled: separate axes or one uniform root+leaf+internal node pool.",
    )
    args = parser.parse_args(argv)
    if args.leaf_counts is not None and args.leaf_sizes is not None:
        raise SystemExit("use either --leaf-counts or --leaf-sizes, not both")
    if args.leaf_counts is None and args.leaf_sizes is None:
        args.leaf_sizes = "16,32,64,128,256"

    out_root = Path(args.out_root) if args.out_root is not None else REPO_ROOT / "outputs" / f"classical_sketches_paper_all_cpu_{_utc_stamp()}"
    out_root = out_root.resolve()
    env = dict(os.environ)
    py_path = [str(REPO_ROOT / "treepo" / "src"), str(REPO_ROOT / "parallel" / "unified_g_v1" / "src"), str(REPO_ROOT)]
    env["PYTHONPATH"] = os.pathsep.join(py_path + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))

    suite_jobs = int(args.jobs)
    if bool(args.include_learned) and _uses_target_parallelism(args.learned_target_jobs):
        suite_jobs = 1

    suite_cmd = [
        sys.executable,
        "-m",
        "treepo.bench.cli",
        "suite",
        "classical-sketches",
        "--out-root",
        str(out_root),
        "--jobs",
        str(int(suite_jobs)),
        "--seeds",
        str(args.seeds),
        "--capacities",
        str(args.capacities),
    ]
    if args.leaf_counts is not None:
        suite_cmd.extend(["--leaf-counts", str(args.leaf_counts)])
    if args.leaf_sizes is not None:
        suite_cmd.extend(["--leaf-sizes", str(args.leaf_sizes)])
    if bool(args.skip_existing):
        suite_cmd.append("--skip-existing")
    if bool(args.include_learned):
        suite_cmd.extend(
            [
                "--include-learned",
                "--learned-targets",
                str(args.learned_targets),
                "--learned-variants",
                str(args.learned_variants),
                "--learned-readout-archs",
                str(args.learned_readout_archs),
                "--learned-epochs",
                str(int(args.learned_epochs)),
                "--learned-n-train",
                str(int(args.learned_n_train)),
                "--learned-n-val",
                str(int(args.learned_n_val)),
                "--learned-batch-size",
                str(int(args.learned_batch_size)),
                "--learned-target-jobs",
                str(args.learned_target_jobs),
                "--learned-gpu-ids",
                str(args.learned_gpu_ids),
                "--learned-batch-reference-leaf-size",
                str(int(args.learned_batch_reference_leaf_size)),
                "--learned-max-batch-size",
                str(int(args.learned_max_batch_size)),
                "--learned-eval-every",
                str(int(args.learned_eval_every)),
            ]
        )
        if args.learned_local_label_rates is not None:
            suite_cmd.extend(["--learned-local-label-rates", str(args.learned_local_label_rates)])
        if args.learned_leaf_query_rates is not None:
            suite_cmd.extend(["--learned-leaf-query-rates", str(args.learned_leaf_query_rates)])
        if args.learned_root_query_rates is not None:
            suite_cmd.extend(["--learned-root-query-rates", str(args.learned_root_query_rates)])
        if args.learned_internal_query_rates is not None:
            suite_cmd.extend(["--learned-internal-query-rates", str(args.learned_internal_query_rates)])
        suite_cmd.extend(
            [
                "--learned-supervision-sampling-policy",
                str(args.learned_supervision_sampling_policy),
            ]
        )
    else:
        suite_cmd.append("--no-include-learned")
    _run(suite_cmd, env=env)

    report_cmd = [
        sys.executable,
        "-m",
        "treepo.bench.cli",
        "report",
        "classical-sketches",
        "--output-root",
        str(out_root),
        "--tables-dir",
        str(
            args.tables_dir
            if args.tables_dir is not None
            else (
                REPO_ROOT / "paper" / "ctreepo" / "tables"
                if bool(args.stage_assets)
                else out_root / "reports" / "classical_sketches" / "tables"
            )
        ),
    ]
    _run(report_cmd, env=env)

    tree_bundle_audit = _audit_tree_bundle_output(out_root, env=env)
    report_dir = out_root / "reports" / "classical_sketches"
    staged = _stage_assets(report_dir) if bool(args.stage_assets) else {"figures": [], "tables": [], "skipped": True}
    aggregate_csv = report_dir / "classical_sketches_aggregate.csv"
    aggregate_rows, learned_rows = _count_rows(aggregate_csv)
    manifest = {
        "out_root": str(out_root),
        "report_dir": str(report_dir),
        "aggregate_csv": str(aggregate_csv),
        "aggregate_rows": int(aggregate_rows),
        "learned_rows": int(learned_rows),
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": _jsonable_args(args),
        "tree_bundle_audit": tree_bundle_audit,
        "staged_assets": staged,
    }
    manifest["run_manifest"] = run_manifest_metadata(
        run_id="classical_sketch.paper_bundle",
        domain="classical_sketch",
        role="paper_bundle",
        backend="treepo",
        status="completed",
        input_contracts=[
            {
                "kind": "tree_bundle_collection",
                "schema_version": "ctreepo.tree_bundle.v1",
                "uri": str(out_root),
                "audit": tree_bundle_audit,
            }
        ],
        f_init="official_oracle",
        g_init="raw_concat",
        f_lineage={"init": "official_oracle", "artifact": "datasketches_or_native"},
        g_lineage={"init": "raw_concat", "artifact": "raw_concat"},
        objective=objective_metadata(
            objective_family="classical_sketch_oracle_state",
            local_law_estimator=LOCAL_LAW_ESTIMATOR_ORACLE_STATE,
            root_share=1.0,
            local_law_component_weights={"merge_preservation": 1.0},
            metadata={
                "oracle_implementation": "datasketches_or_native",
                "aggregate_rows": int(aggregate_rows),
                "learned_rows": int(learned_rows),
            },
        ),
        audit_results={"ok": True, "tree_bundle_audit": tree_bundle_audit},
        publication_ready=True,
        command=sys.argv,
        metadata={"runner": "scripts/run_classical_sketches_paper_bundle.py"},
        output_artifacts=[
            {"kind": "run_directory", "uri": str(out_root)},
            {"kind": "aggregate_csv", "uri": str(aggregate_csv)},
        ],
    )
    manifest_path = out_root / "paper_bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
