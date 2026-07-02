#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.contracts import MethodRef, ResultRow, benchmark_ref_from_parts, method_ref_from_parts
from src.experiments.roles import (
    ROLE_SCORER,
    ROLE_STATE_MODEL,
    chat_role_ref,
    embedder_role_ref,
    metadata_with_roles,
    oracle_ref,
    state_model_role_ref,
)
from src.experiments.sidecars import write_canonical_sidecars
from src.runtime.methods import METHOD_COMPARE_RUNNER_ALIASES, discover_method


PROFILE_ORDER = (
    "baseline_llm",
    "embedding_proxy_ridge",
    "neural_operator_hybrid",
    "generator_lora_dpo",
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _method_entries(method_compare_dir: Path, profiles: Iterable[str]) -> List[Dict[str, Any]]:
    manifest_path = method_compare_dir / "method_compare_manifest.json"
    by_profile: Dict[str, Dict[str, Any]] = {}
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        for entry in list(manifest.get("entries", []) or []):
            profile = str(entry.get("profile") or "").strip()
            run_dir = Path(str(entry.get("run_dir") or method_compare_dir / profile)).expanduser()
            if profile:
                by_profile[profile] = {"profile": profile, "run_dir": str(run_dir)}

    entries: List[Dict[str, Any]] = []
    for profile in profiles:
        if profile in by_profile:
            entries.append(by_profile[profile])
            continue
        run_dir = method_compare_dir / profile
        if run_dir.exists():
            entries.append({"profile": profile, "run_dir": str(run_dir)})
    return entries


def _variant_config(
    *,
    base_cfg: Dict[str, Any],
    method: str,
    method_dir: Path,
    method_family: str,
    trained: bool,
    variant: str,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    runtime_defaults = dict(cfg.get("runtime_defaults", {}) or {})
    runtime_defaults.update(
        {
            "method_dir": str(method_dir),
            "method_family": method_family,
            "method_trained": bool(trained),
            "method_variant": variant,
        }
    )
    cfg["runtime_defaults"] = runtime_defaults
    cfg["methods"] = [method]
    phases = []
    for phase in list(cfg.get("phases", []) or []):
        ph = dict(phase)
        ph["phase_id"] = f"{ph.get('phase_id', 'P0')}_{method}"
        ph["methods"] = [method]
        phases.append(ph)
    cfg["phases"] = phases
    cfg.setdefault("run", {})["name"] = f"longbench_v2_{method}"
    return cfg


def _commands_for_variant(
    *,
    python_exe: str,
    config_path: Path,
    output_root: Path,
    run_id: str,
    mock_llm: bool,
    max_problems: int | None,
    max_units: int | None,
) -> List[List[str]]:
    run_dir = output_root / run_id
    init_cmd = [
        python_exe,
        "scripts/run_runtime_eval.py",
        "init",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_root),
        "--run-id",
        run_id,
    ]
    run_cmd = [
        python_exe,
        "scripts/run_runtime_eval.py",
        "run",
        "--run-dir",
        str(run_dir),
    ]
    if mock_llm:
        run_cmd.append("--mock-llm")
    if max_problems is not None:
        run_cmd.extend(["--max-problems", str(max_problems)])
    if max_units is not None:
        run_cmd.extend(["--max-units", str(max_units)])
    agg_cmd = [
        python_exe,
        "scripts/run_runtime_eval.py",
        "aggregate",
        "--run-dir",
        str(run_dir),
    ]
    return [init_cmd, run_cmd, agg_cmd]


def _write_summary(output_root: Path, rows: List[Dict[str, Any]]) -> None:
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    _write_json(output_root / "method_compare_lbv2_summary.json", summary)
    lines = [
        "# LongBench v2 Method Compare",
        "",
        "| method | variant | primary_mean | predictions |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        metrics = dict(row.get("metrics") or {})
        lines.append(
            "| {method} | {variant} | {primary:.4f} | {n} |".format(
                method=row.get("method", ""),
                variant=row.get("variant", ""),
                primary=float(metrics.get("primary_mean", 0.0) or 0.0),
                n=int(metrics.get("n_predictions", 0) or 0),
            )
        )
    (output_root / "method_compare_lbv2_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _lbv2_method_ref(method: str, *, profile: str, variant: str) -> MethodRef:
    roles: Dict[str, Any] = {ROLE_SCORER: chat_role_ref(role=ROLE_SCORER)}
    if "embedding" in profile or "retrieval" in method:
        roles["embedder"] = embedder_role_ref(engine="configured")
    if "neural_operator" in profile or "neural" in method:
        roles[ROLE_STATE_MODEL] = state_model_role_ref(
            engine="configured",
            model=profile,
            execution_mode="frozen_or_trained",
        )
    return method_ref_from_parts(
        family=str(method),
        variant=str(variant),
        adapter="method_compare_lbv2",
        metadata=metadata_with_roles(
            {"profile": profile, "variant": variant},
            roles=roles,
            oracle=oracle_ref(kind="benchmark_labels", source="longbench_v2"),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate method_compare profiles on LongBench v2."
    )
    parser.add_argument(
        "--method-compare-dir", required=True, help="Existing outputs/method_compare_* directory."
    )
    parser.add_argument(
        "--lbv2-config",
        default=str(REPO_ROOT / "config" / "runtime_eval" / "longbench_v2_full_stack.yaml"),
        help="Base LongBench v2 runtime-eval config.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "outputs" / f"method_compare_lbv2_{_timestamp()}"),
    )
    parser.add_argument(
        "--profiles", nargs="+", default=list(PROFILE_ORDER), choices=list(PROFILE_ORDER)
    )
    parser.add_argument("--include-raw-variants", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=None, help="Run at most this many problems per unit."
    )
    parser.add_argument(
        "--max-units", type=int, default=None, help="Run at most this many units per variant."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    method_compare_dir = Path(args.method_compare_dir).expanduser().resolve()
    lbv2_config = Path(args.lbv2_config).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base_cfg = yaml.safe_load(lbv2_config.read_text(encoding="utf-8"))
    entries = _method_entries(method_compare_dir, args.profiles)
    if not entries:
        raise SystemExit(f"No method profiles found under {method_compare_dir}")

    manifest_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for entry in entries:
        profile = str(entry["profile"])
        profile_dir = Path(str(entry["run_dir"])).expanduser().resolve()
        variants = ["trained"]
        if args.include_raw_variants:
            variants.insert(0, "raw")
        for variant in variants:
            trained = variant == "trained"
            method_spec = discover_method(profile_dir, trained=trained)
            method = f"{profile}_{variant}"
            if method not in METHOD_COMPARE_RUNNER_ALIASES:
                raise SystemExit(f"Unsupported method_compare LongBench method: {method}")
            config_payload = _variant_config(
                base_cfg=base_cfg,
                method=method,
                method_dir=profile_dir,
                method_family=method_spec.family,
                trained=trained,
                variant=variant,
            )
            config_path = output_root / "configs" / f"{method}.yaml"
            _write_yaml(config_path, config_payload)
            commands = _commands_for_variant(
                python_exe=str(args.python),
                config_path=config_path,
                output_root=output_root,
                run_id=method,
                mock_llm=bool(args.mock_llm),
                max_problems=args.limit,
                max_units=args.max_units,
            )
            row = {
                "profile": profile,
                "variant": variant,
                "method": method,
                "run_dir": str(output_root / method),
                "config_path": str(config_path),
                "commands": commands,
            }
            manifest_rows.append(row)
            if args.dry_run:
                print(f"[dry-run] {method}")
                for command in commands:
                    print("  " + " ".join(command))
                continue
            for command in commands:
                subprocess.check_call(command, cwd=str(REPO_ROOT))
            metrics_path = output_root / method / "metrics.json"
            metrics = _load_json(metrics_path) if metrics_path.exists() else {}
            summary_rows.append({**row, "metrics": metrics})

    _write_json(
        output_root / "method_compare_lbv2_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "method_compare_dir": str(method_compare_dir),
            "lbv2_config": str(lbv2_config),
            "dry_run": bool(args.dry_run),
            "entries": manifest_rows,
        },
    )
    _write_summary(output_root, summary_rows)
    benchmark_ref = benchmark_ref_from_parts(
        family="longbench_v2",
        scope="method_compare",
        dataset_id=str(base_cfg.get("benchmark", {}).get("dataset_path") or base_cfg.get("benchmark", {}).get("hf_dataset") or ""),
        name="longbench_v2",
        metadata=dict(base_cfg.get("benchmark") or {}),
    )
    method_refs = tuple(
        _lbv2_method_ref(
            str(row.get("method") or ""),
            profile=str(row.get("profile") or ""),
            variant=str(row.get("variant") or ""),
        )
        for row in manifest_rows
    )
    method_by_name = {method.family: method for method in method_refs}
    result_rows = []
    for row in summary_rows:
        metrics = dict(row.get("metrics") or {})
        method_name = str(row.get("method") or "")
        method_ref = method_by_name.get(method_name)
        if method_ref is None:
            continue
        result_rows.append(
            ResultRow(
                experiment_id="",
                phase="method_compare_lbv2",
                benchmark_ref=benchmark_ref,
                method_ref=method_ref,
                split="test",
                metric_name="primary_mean",
                metric_value=metrics.get("primary_mean"),
                artifact_refs=("method_compare_lbv2_manifest_json",),
                metadata={"metrics": metrics, **row},
            )
        )
    artifacts = {
        "method_compare_lbv2_manifest_json": str(output_root / "method_compare_lbv2_manifest.json"),
        "method_compare_lbv2_summary_json": str(output_root / "method_compare_lbv2_summary.json"),
        "method_compare_lbv2_summary_md": str(output_root / "method_compare_lbv2_summary.md"),
    }
    write_canonical_sidecars(
        output_root,
        title="method_compare_lbv2",
        adapter_id="method_compare_lbv2",
        benchmark_refs=(benchmark_ref,),
        method_refs=method_refs,
        phases=("dry_run" if args.dry_run else "run",),
        artifacts=artifacts,
        result_rows=result_rows,
        state="dry_run" if args.dry_run else "completed",
        metadata={"method_compare_dir": str(method_compare_dir), "lbv2_config": str(lbv2_config)},
        launch_command=sys.argv,
        report_profiles=("runtime_eval_summary",),
    )
    print(f"Wrote LongBench v2 method-compare outputs to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
