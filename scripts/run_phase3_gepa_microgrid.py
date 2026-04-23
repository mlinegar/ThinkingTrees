#!/usr/bin/env python3
"""Run a bounded phase-3 GEPA microgrid.

The intended use is an apples-to-apples sweep over the same chunk-size axis as
the phase-3 combined grid, while keeping GEPA cheap:

1. Evaluate a fresh unified-g + fresh scorer baseline.
2. Run a staged component optimization plan such as f, fg, or fgf.
3. Warm-start every stage from the previous stage's saved program components.

This keeps exact schedule syntax (fg, gf, fgf, ...), but reports any
multi-component schedule under the simpler "joint" codename. In this repo's
current DSPy version the official API is still optimizer.compile(...), but this
runner uses "stage plan" terminology so the experiment maps cleanly onto
fit-style component schedules like fgf and fgfgf.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE3_SCRIPT = REPO_ROOT / "scripts" / "phase3_full_pipeline_optimize.py"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "phase3"
    / f"gepa_microgrid_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _stage_plan_codename(stage_plan: str | None) -> str:
    plan = str(stage_plan or "").strip().lower()
    if not plan:
        return ""
    if plan in {"f", "g"}:
        return plan
    return "joint"


def _load_report(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _pearson(report: dict[str, Any] | None, section: str) -> float | None:
    if not report:
        return None
    value = report.get(section, {}).get("pearson_r")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _section(report: dict[str, Any] | None, section: str) -> dict[str, Any]:
    if not report:
        return {}
    value = report.get(section)
    return value if isinstance(value, dict) else {}


def _prediction_path(report: dict[str, Any] | None, section: str) -> str | None:
    value = _section(report, section).get("prediction_path")
    return str(value) if value else None


def _artifacts_for_run(report: dict[str, Any] | None, stage: str) -> dict[str, Any]:
    if not report:
        return {}
    key = "final_artifacts" if stage == "baseline" else "optimized_artifacts"
    artifacts = report.get(key)
    if isinstance(artifacts, dict) and artifacts:
        return dict(artifacts)
    fallback = report.get("final_artifacts")
    return dict(fallback) if isinstance(fallback, dict) else {}


def _stage_entry(run: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(run["output_dir"])
    report = _load_report(output_dir)
    stage = str(run["stage"])
    dev_section = "baseline_dev" if stage == "baseline" else "optimized_dev"
    if stage != "baseline" and not _section(report, dev_section):
        dev_section = "baseline_dev"
    test_section = "final_test"
    artifacts = _artifacts_for_run(report, stage)
    dev_report = _section(report, dev_section)
    test_report = _section(report, test_section)
    return {
        "condition_key": run.get("condition_key"),
        "stage": stage,
        "stage_index": run.get("stage_index"),
        "stage_component": run.get("stage_component"),
        "stage_plan": run.get("stage_plan"),
        "stage_plan_codename": run.get("stage_plan_codename")
        or _stage_plan_codename(run.get("stage_plan")),
        "output_dir": str(output_dir),
        "status": run.get("status"),
        "return_code": run.get("return_code"),
        "init_dir": run.get("init_dir"),
        "init_artifact_kind": run.get("init_artifact_kind"),
        "chunk_chars": run.get("chunk_chars"),
        "train_n": run.get("train_n"),
        "dev_n": run.get("dev_n"),
        "test_n": run.get("test_n"),
        "dev_metric_section": dev_section,
        "test_metric_section": test_section,
        "dev_pearson_r": _pearson(report, dev_section),
        "test_pearson_r": _pearson(report, test_section),
        "dev_pearson_defined": dev_report.get("pearson_defined"),
        "test_pearson_defined": test_report.get("pearson_defined"),
        "prediction_paths": {
            "dev": _prediction_path(report, dev_section),
            "test": _prediction_path(report, test_section),
            "baseline_dev": _prediction_path(report, "baseline_dev"),
            "optimized_dev": _prediction_path(report, "optimized_dev"),
            "final_test": _prediction_path(report, "final_test"),
        },
        "artifacts": {
            "program": artifacts.get("program"),
            "scorer_f": artifacts.get("scorer"),
            "unified_g": artifacts.get("g"),
        },
        "compile_time_seconds": (
            report.get("compile_time_seconds")
            if isinstance(report, dict)
            else None
        ),
        "elapsed_seconds": run.get("elapsed_seconds"),
        "report_path": run.get("report_path"),
        "log_path": run.get("log_path"),
    }


def _best_by_dev(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        entry for entry in entries
        if entry.get("dev_pearson_r") is not None
        and entry.get("dev_pearson_defined") is not False
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda entry: (
            float(entry["dev_pearson_r"]),
            -1 if entry.get("stage") == "baseline" else int(entry.get("stage_index") or 0),
        ),
    )


def _refresh_stage_manifest(manifest: dict[str, Any]) -> None:
    entries = [_stage_entry(run) for run in manifest.get("runs", [])]
    manifest["stage_history"] = entries
    condition_summaries: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = str(entry.get("condition_key") or "default")
        condition = condition_summaries.setdefault(key, {"stage_history": []})
        condition["stage_history"].append(entry)
        condition["final_stage"] = entry
        condition["dev_best_stage"] = _best_by_dev(condition["stage_history"])
    manifest["condition_summaries"] = condition_summaries
    manifest["final_stage"] = entries[-1] if entries else None
    manifest["dev_best_stage"] = _best_by_dev(entries)


def _fmt(value: float | None) -> str:
    return f"{value:+.3f}" if value is not None else "-"


def _run_command(
    *,
    cmd: list[str],
    output_dir: Path,
    env: dict[str, str],
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    log_path = output_dir / "run.log"
    rendered = " ".join(shlex.quote(item) for item in cmd)
    record: dict[str, Any] = {
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "log_path": str(log_path),
        "command": cmd,
        "command_rendered": rendered,
        "started_at": _utc_now(),
    }
    if report_path.exists() and not force:
        record.update({"status": "skipped_existing", "return_code": 0, "finished_at": _utc_now()})
        return record
    if dry_run:
        record.update({"status": "dry_run", "return_code": None, "finished_at": _utc_now()})
        return record

    with log_path.open("w") as log:
        log.write(f"$ {rendered}\n\n")
        log.flush()
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    record.update(
        {
            "status": "completed" if proc.returncode == 0 else "failed",
            "return_code": int(proc.returncode),
            "elapsed_seconds": round(time.time() - t0, 1),
            "finished_at": _utc_now(),
        }
    )
    if proc.returncode != 0:
        tail = ""
        try:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
        except OSError:
            pass
        record["log_tail"] = tail
    return record


def _append_initial_init_args(cmd: list[str], args: argparse.Namespace) -> None:
    if args.init_dir is not None:
        cmd.extend(["--init-dir", str(args.init_dir)])
        cmd.extend(["--init-artifact-kind", args.init_artifact_kind])
    if args.init_program is not None:
        cmd.extend(["--init-program", str(args.init_program)])
    if args.init_scorer is not None:
        cmd.extend(["--init-scorer", str(args.init_scorer)])
    if args.init_g is not None:
        cmd.extend(["--init-g", str(args.init_g)])
    if args.init_g_legacy_leaf is not None:
        cmd.extend(["--init-g-legacy-leaf", str(args.init_g_legacy_leaf)])


def _phase3_base_cmd(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    optimizer: str,
    scope: str,
    init_dir: Path | None = None,
    use_initial_init: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        str(PHASE3_SCRIPT),
        "--dimension",
        args.dimension,
        "--optimizer",
        optimizer,
        "--optimize-scope",
        scope,
        "--metric-mode",
        args.metric_mode,
        "--feedback-mode",
        args.feedback_mode,
        "--selection-guard",
        args.selection_guard,
        "--split-strategy",
        args.split_strategy,
        "--train-n",
        str(args.current_train_n),
        "--dev-n",
        str(args.dev_n),
        "--test-n",
        str(args.test_n),
        "--chunk-chars",
        str(args.current_chunk_chars),
        "--max-workers",
        str(args.max_workers),
        "--seed",
        str(args.seed),
        "--max-tokens",
        str(args.max_tokens),
        "--reflection-max-tokens",
        str(args.reflection_max_tokens),
        "--output-dir",
        str(output_dir),
        "--log-level",
        args.log_level,
    ]
    if init_dir is not None:
        cmd.extend(["--init-dir", str(init_dir)])
        cmd.extend(["--init-artifact-kind", "optimized"])
        cmd.append("--init-components-only")
    elif use_initial_init:
        _append_initial_init_args(cmd, args)
    if args.model:
        cmd.extend(["--model", args.model])
    if args.ports:
        cmd.append("--ports")
        cmd.extend(str(port) for port in args.ports)
    else:
        cmd.extend(["--port", str(args.port)])
    return cmd


def _baseline_cmd(args: argparse.Namespace, output_dir: Path) -> list[str]:
    return _phase3_base_cmd(
        args,
        output_dir,
        optimizer="none",
        scope="gf",
        use_initial_init=True,
    )


def _f_gepa_cmd(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    init_dir: Path | None = None,
    use_initial_init: bool = False,
) -> list[str]:
    cmd = _phase3_base_cmd(
        args,
        output_dir,
        optimizer="gepa",
        scope="f",
        init_dir=init_dir,
        use_initial_init=use_initial_init,
    )
    cmd.extend(
        [
            "--gepa-max-metric-calls",
            str(args.gepa_f_calls),
            "--gepa-valset-cap",
            str(args.gepa_valset_cap),
            "--gepa-threads",
            str(args.gepa_threads),
        ]
    )
    return cmd


def _g_gepa_cmd(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    init_dir: Path | None = None,
    use_initial_init: bool = False,
) -> list[str]:
    cmd = _phase3_base_cmd(
        args,
        output_dir,
        optimizer="gepa",
        scope="g",
        init_dir=init_dir,
        use_initial_init=use_initial_init,
    )
    cmd.extend(
        [
            "--gepa-max-metric-calls",
            str(args.gepa_g_calls),
            "--gepa-valset-cap",
            str(args.gepa_g_valset_cap),
            "--gepa-threads",
            str(args.gepa_g_threads),
        ]
    )
    return cmd


def _stage_cmd(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    component: str,
    init_dir: Path | None,
    use_initial_init: bool = False,
) -> list[str]:
    if component == "f":
        return _f_gepa_cmd(
            args,
            output_dir,
            init_dir=init_dir,
            use_initial_init=use_initial_init,
        )
    if component == "g":
        return _g_gepa_cmd(
            args,
            output_dir,
            init_dir=init_dir,
            use_initial_init=use_initial_init,
        )
    raise ValueError(f"unknown GEPA component: {component!r}")


def _grid_reference(chunk_chars: int, dimension: str) -> dict[str, Any] | None:
    path = REPO_ROOT / "outputs" / "phase3" / f"combined_c{chunk_chars}" / "report.json"
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    per_dim = report.get("per_dim", {}).get(dimension)
    return {
        "path": str(path),
        "macro_pearson_r": report.get("macro_pearson_r"),
        "dimension_pearson_r": per_dim.get("pearson_r") if isinstance(per_dim, dict) else None,
        "n_scored": report.get("run", {}).get("n_scored"),
    }


def _summarize(manifest: dict[str, Any], out_md: Path) -> None:
    _refresh_stage_manifest(manifest)
    final_stage = manifest.get("final_stage") or {}
    dev_best_stage = manifest.get("dev_best_stage") or {}
    lines = [
        "# Phase 3 GEPA Microgrid",
        "",
        f"Started: `{manifest['started_at']}`",
        f"Dimension: `{manifest['args']['dimension']}`",
        f"Selection guard: `{manifest['args']['selection_guard']}`",
        f"Best model policy: `{manifest['args']['best_model_policy']}`",
        f"Split strategy: `{manifest['args']['split_strategy']}`",
        f"Stage plan: `{manifest['args']['stage_plan']}`",
        f"Codename: `{manifest['args'].get('stage_plan_codename', _stage_plan_codename(manifest['args'].get('stage_plan')))}`",
        "",
        "## Stage Selection",
        "",
        "|role|stage|dev r|test r|output|",
        "|---|---|---:|---:|---|",
        "|final|"
        + "|".join(
            [
                str(final_stage.get("stage", "-")),
                _fmt(final_stage.get("dev_pearson_r")),
                _fmt(final_stage.get("test_pearson_r")),
                f"`{Path(str(final_stage.get('output_dir', '-'))).name}`",
            ]
        )
        + "|",
        "|dev-best|"
        + "|".join(
            [
                str(dev_best_stage.get("stage", "-")),
                _fmt(dev_best_stage.get("dev_pearson_r")),
                _fmt(dev_best_stage.get("test_pearson_r")),
                f"`{Path(str(dev_best_stage.get('output_dir', '-'))).name}`",
            ]
        )
        + "|",
        "",
        "## Runs",
        "",
        "|stage|chunk|train|dev|test|grid dim r|baseline dev|optimized dev|final test|seconds|output|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in manifest.get("runs", []):
        report = _load_report(Path(run["output_dir"]))
        ref = _grid_reference(int(run["chunk_chars"]), manifest["args"]["dimension"])
        output = Path(run["output_dir"]).name
        lines.append(
            "|"
            + "|".join(
                [
                    str(run["stage"]),
                    str(run["chunk_chars"]),
                    str(run["train_n"]),
                    str(run["dev_n"]),
                    str(run["test_n"]),
                    _fmt(ref.get("dimension_pearson_r") if ref else None),
                    _fmt(_pearson(report, "baseline_dev")),
                    _fmt(_pearson(report, "optimized_dev")),
                    _fmt(_pearson(report, "final_test")),
                    str(run.get("elapsed_seconds", "-")),
                    f"`{output}`",
                ]
            )
            + "|"
        )
    lines.append("")
    lines.append("Existing grid references are read from `outputs/phase3/combined_c*/report.json`.")
    out_md.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dimension", default="decentralization")
    p.add_argument("--chunk-chars", type=int, nargs="+", default=[16000])
    p.add_argument("--train-ns", type=int, nargs="+", default=[8])
    p.add_argument("--dev-n", type=int, default=8)
    p.add_argument("--test-n", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--ports", type=int, nargs="+", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--context-window", type=int, default=12000)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--metric-mode", choices=["mae", "rank"], default="rank")
    p.add_argument("--feedback-mode", choices=["scalar", "rich"], default="rich")
    p.add_argument("--selection-guard", choices=["none", "dev"], default="none")
    p.add_argument("--best-model-policy", choices=["report-both", "final-only", "dev-best-only"],
                   default="report-both",
                   help="Reporting policy only. Staged learning always continues from the previous "
                        "optimized state; dev-best does not rewrite the stage trajectory.")
    p.add_argument("--split-strategy", choices=["random", "label-stratified"], default="random")
    p.add_argument("--stage-plan", default="f",
                   help="Sequential GEPA component schedule, e.g. f, fg, fgf, fgfgf. "
                        "Each stage warm-starts from the previous stage output.")
    p.add_argument("--init-dir", type=Path, default=None,
                   help="Optional initial artifact directory for baseline and first stage.")
    p.add_argument("--init-artifact-kind", choices=["final", "optimized"], default="final",
                   help="Artifact preference for the optional initial --init-dir. Stage-to-stage "
                        "continuation always uses optimized artifacts.")
    p.add_argument("--init-program", type=Path, default=None)
    p.add_argument("--init-scorer", type=Path, default=None)
    p.add_argument("--init-g", type=Path, default=None)
    p.add_argument("--init-g-legacy-leaf", type=Path, default=None)
    p.add_argument("--reflection-max-tokens", type=int, default=2048)
    p.add_argument("--gepa-f-calls", type=int, default=32)
    p.add_argument("--gepa-g-calls", type=int, default=8)
    p.add_argument("--gepa-valset-cap", type=int, default=8)
    p.add_argument("--gepa-g-valset-cap", type=int, default=4)
    p.add_argument("--gepa-threads", type=int, default=4)
    p.add_argument("--gepa-g-threads", type=int, default=1)
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-f-stage", action="store_true")
    p.add_argument("--run-g-stage", action="store_true",
                   help="Deprecated alias: with the default stage plan, run fg.")
    p.add_argument("--allow-g-after-undefined-dev", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    stage_plan = str(args.stage_plan or "").strip().lower()
    if args.run_g_stage and stage_plan == "f":
        stage_plan = "fg"
    if args.skip_f_stage:
        stage_plan = stage_plan.replace("f", "")
    if any(ch not in {"f", "g"} for ch in stage_plan):
        raise SystemExit(f"--stage-plan may contain only 'f' and 'g', got {args.stage_plan!r}")
    args.stage_plan = stage_plan
    stage_plan_codename = _stage_plan_codename(stage_plan)
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["MANIFESTO_CONTEXT_WINDOW"] = str(args.context_window)
    env["MANIFESTO_MAX_TOKENS"] = str(args.max_tokens)

    manifest: dict[str, Any] = {
        "started_at": _utc_now(),
        "output_root": str(args.output_root),
        "best_model_policy": args.best_model_policy,
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key not in {"current_train_n", "current_chunk_chars"}
        },
        "runs": [],
        "stage_history": [],
        "final_stage": None,
        "dev_best_stage": None,
    }
    manifest["args"]["stage_plan_codename"] = stage_plan_codename
    manifest_path = args.output_root / "microgrid_manifest.json"
    summary_path = args.output_root / "summary.md"

    for chunk in args.chunk_chars:
        for train_n in args.train_ns:
            args.current_chunk_chars = int(chunk)
            args.current_train_n = int(train_n)
            suffix = f"{args.dimension}_c{chunk}_tr{train_n}_dev{args.dev_n}_test{args.test_n}"
            condition_key = suffix

            if not args.skip_baseline:
                out_dir = args.output_root / f"baseline_{suffix}"
                record = _run_command(
                    cmd=_baseline_cmd(args, out_dir),
                    output_dir=out_dir,
                    env=env,
                    force=args.force,
                    dry_run=args.dry_run,
                )
                record.update(
                    {
                        "stage": "baseline",
                        "condition_key": condition_key,
                        "stage_plan": stage_plan,
                        "stage_plan_codename": stage_plan_codename,
                        "init_dir": str(args.init_dir) if args.init_dir is not None else None,
                        "init_artifact_kind": args.init_artifact_kind,
                        "chunk_chars": chunk,
                        "train_n": train_n,
                        "dev_n": args.dev_n,
                        "test_n": args.test_n,
                    }
                )
                manifest["runs"].append(record)
                report = _load_report(out_dir)
                record["baseline_dev_pearson_r"] = _pearson(report, "baseline_dev")
                record["final_test_pearson_r"] = _pearson(report, "final_test")
                if report:
                    record["artifacts"] = {
                        "optimized": report.get("optimized_artifacts"),
                        "final": report.get("final_artifacts"),
                    }
                    record["prediction_paths"] = report.get("prediction_paths")
                    record["canonical_outputs"] = report.get("canonical_outputs")
                    record["cache_stats"] = {
                        "selected": report.get("selected_program_cache_stats"),
                        "baseline": report.get("baseline_cache_stats"),
                    }
                    record["vllm_prefix_metrics"] = report.get("vllm_prefix_metrics")
                _refresh_stage_manifest(manifest)
                _write_json(manifest_path, manifest)
                _summarize(manifest, summary_path)
                if record.get("return_code") not in (0, None):
                    return int(record["return_code"])

            init_dir: Path | None = None
            stop_reason: str | None = None
            for stage_index, component in enumerate(stage_plan, start=1):
                out_dir = args.output_root / f"stage{stage_index}_{component}_{suffix}"
                use_initial_init = init_dir is None
                record = _run_command(
                    cmd=_stage_cmd(
                        args,
                        out_dir,
                        component=component,
                        init_dir=init_dir,
                        use_initial_init=use_initial_init,
                    ),
                    output_dir=out_dir,
                    env=env,
                    force=args.force,
                    dry_run=args.dry_run,
                )
                record.update(
                    {
                        "stage": f"stage{stage_index}_{component}",
                        "condition_key": condition_key,
                        "stage_index": stage_index,
                        "stage_component": component,
                        "stage_plan": stage_plan,
                        "stage_plan_codename": stage_plan_codename,
                        "init_dir": str(init_dir) if init_dir is not None else None,
                        "init_artifact_kind": "optimized" if init_dir is not None else args.init_artifact_kind,
                        "used_initial_init": use_initial_init,
                        "chunk_chars": chunk,
                        "train_n": train_n,
                        "dev_n": args.dev_n,
                        "test_n": args.test_n,
                    }
                )
                manifest["runs"].append(record)
                report = _load_report(out_dir)
                record["optimized_dev_pearson_r"] = _pearson(report, "optimized_dev")
                record["final_test_pearson_r"] = _pearson(report, "final_test")
                opt_dev = report.get("optimized_dev", {}) if report else {}
                record["optimized_dev_pearson_defined"] = opt_dev.get("pearson_defined")
                if report:
                    record["artifacts"] = {
                        "optimized": report.get("optimized_artifacts"),
                        "final": report.get("final_artifacts"),
                    }
                    record["prediction_paths"] = report.get("prediction_paths")
                    record["canonical_outputs"] = report.get("canonical_outputs")
                    record["compile_time_seconds"] = report.get("compile_time_seconds")
                    record["cache_stats"] = {
                        "selected": report.get("selected_program_cache_stats"),
                        "baseline": report.get("baseline_cache_stats"),
                    }
                    record["vllm_prefix_metrics"] = report.get("vllm_prefix_metrics")
                _refresh_stage_manifest(manifest)
                _write_json(manifest_path, manifest)
                _summarize(manifest, summary_path)
                if record.get("return_code") not in (0, None):
                    return int(record["return_code"])
                if not args.dry_run and report is None:
                    stop_reason = f"stage {stage_index} did not write report.json"
                    break
                if (
                    not args.dry_run
                    and
                    component == "f"
                    and stage_index < len(stage_plan)
                    and not bool(record["optimized_dev_pearson_defined"])
                    and not args.allow_g_after_undefined_dev
                ):
                    stop_reason = (
                        f"stage {stage_index} f optimized-dev Pearson undefined; "
                        "skipping remaining stages for this condition"
                    )
                    break
                init_dir = out_dir
            if stop_reason is not None:
                manifest.setdefault("skipped_stage_suffixes", {})[suffix] = stop_reason
                _refresh_stage_manifest(manifest)
                _write_json(manifest_path, manifest)
                _summarize(manifest, summary_path)

    manifest["finished_at"] = _utc_now()
    _refresh_stage_manifest(manifest)
    _write_json(manifest_path, manifest)
    _summarize(manifest, summary_path)
    print(f"Wrote {manifest_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
