#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.error import URLError
from urllib.request import urlopen

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runtime.adapters.ruler import RulerDatasetSpec, RulerSyntheticAdapter
from src.runtime.backbone import BackboneAdapter, BackboneConfig
from src.runtime.contracts import RunPhaseSpec, RunSpec, RunUnit, RuntimeConfig, expand_units, units_digest
from src.runtime.loop import run_unit
from src.runtime.trace import JsonlWriter, TraceWriter


def _normalize_backend_name(raw: Any, default: str = "vllm") -> str:
    rendered = str(raw or "").strip().lower()
    if rendered in {"vllm", "sglang"}:
        return rendered
    return default


def _load_inference_backend_defaults() -> Dict[str, Any]:
    """Best-effort backend defaults from settings, with safe fallbacks."""
    defaults = {
        "task_backend": "vllm",
        "fallback_backend": "none",
        "sglang_venv_path": "/home/mlinegar/sglang-env",
        "vllm_venv_path": "/home/mlinegar/vllm-env",
        "sglang_port": 30000,
        "vllm_port": 8000,
    }
    try:
        from src.config.settings import get_inference_backend_config, load_settings

        settings = load_settings()
        backend_cfg = get_inference_backend_config(settings)
        sglang_cfg = settings.get("sglang", {}) if isinstance(settings, dict) else {}
        vllm_cfg = settings.get("vllm", {}) if isinstance(settings, dict) else {}
        defaults.update(
            {
                "task_backend": _normalize_backend_name(
                    backend_cfg.get("task_backend"),
                    default="vllm",
                ),
                "fallback_backend": _normalize_backend_name(
                    backend_cfg.get("fallback_backend"),
                    default="vllm",
                ),
                "sglang_venv_path": str(
                    backend_cfg.get("sglang_venv_path") or defaults["sglang_venv_path"]
                ),
                "vllm_venv_path": str(
                    backend_cfg.get("vllm_venv_path") or defaults["vllm_venv_path"]
                ),
                "sglang_port": int(sglang_cfg.get("port", defaults["sglang_port"])),
                "vllm_port": int(vllm_cfg.get("port", defaults["vllm_port"])),
            }
        )
    except Exception:
        pass
    return defaults


def _default_backend_port(backend: str, defaults: Optional[Dict[str, Any]] = None) -> int:
    cfg = defaults or _load_inference_backend_defaults()
    backend_name = _normalize_backend_name(backend, default="vllm")
    if backend_name == "sglang":
        return int(cfg.get("sglang_port", 30000))
    return int(cfg.get("vllm_port", 8000))


def _default_backend_base_url(backend: str, defaults: Optional[Dict[str, Any]] = None) -> str:
    return f"http://localhost:{_default_backend_port(backend, defaults)}/v1"


def _endpoint_ready(base_url: str, timeout_seconds: float = 2.0) -> bool:
    target = str(base_url).rstrip("/") + "/models"
    try:
        with urlopen(target, timeout=max(0.25, float(timeout_seconds))) as resp:
            return int(getattr(resp, "status", 0)) == 200
    except (URLError, OSError, ValueError):
        return False


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _load_run_spec(config_path: Path, *, output_dir: Path, run_id: Optional[str]) -> RunSpec:
    cfg = yaml.safe_load(config_path.read_text())

    run_id_final = run_id or cfg.get("run", {}).get("run_id") or f"runtime_{_utc_ts()}"

    benchmark_cfg = dict(cfg.get("benchmark", {}))
    model_cfg = dict(cfg.get("model", {}))
    runtime_defaults = dict(cfg.get("runtime_defaults", {}))

    phases: List[RunPhaseSpec] = []
    for ph in cfg.get("phases", []):
        phases.append(
            RunPhaseSpec(
                phase_id=str(ph["phase_id"]),
                tasks=list(ph["tasks"]),
                lengths=[int(x) for x in ph["lengths"]],
                seeds=[int(x) for x in ph["seeds"]],
                num_samples=int(ph["num_samples"]),
                split=str(ph.get("split", "validation")),
                modes=list(ph["modes"]),
                runtime_overrides=dict(ph.get("runtime_overrides", {})),
                benchmark_overrides=dict(ph.get("benchmark_overrides", {})),
                runtime_grid=dict(ph.get("runtime_grid", {})),
                benchmark_grid=dict(ph.get("benchmark_grid", {})),
            )
        )

    return RunSpec(
        run_id=run_id_final,
        created_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        output_dir=str(output_dir),
        benchmark=benchmark_cfg,
        model=model_cfg,
        runtime_defaults=runtime_defaults,
        phases=phases,
    )


def _run_dir(output_dir: Path, run_id: str) -> Path:
    return output_dir / run_id


def cmd_init(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = _load_run_spec(config_path, output_dir=output_dir, run_id=args.run_id)
    units = expand_units(spec)
    digest = units_digest(units)

    run_dir = _run_dir(output_dir, spec.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_yaml(run_dir / "resolved_run.yaml", spec.to_dict())
    _write_json(run_dir / "config.json", spec.to_dict())

    units_path = run_dir / "units.jsonl"
    units_writer = JsonlWriter(units_path)
    for u in units:
        units_writer.write(u.to_dict())

    (run_dir / "units_digest.txt").write_text(digest + "\n")

    print(f"Initialized run {spec.run_id}")
    print(f"- Run dir: {run_dir}")
    print(f"- Units: {len(units)} ({units_path})")
    print(f"- Digest: {digest[:12]}… ({run_dir / 'units_digest.txt'})")


def _iter_units(run_dir: Path) -> Iterable[RunUnit]:
    units_path = run_dir / "units.jsonl"
    if not units_path.exists():
        raise FileNotFoundError(f"units.jsonl not found in {run_dir}. Run init first.")
    with units_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            yield RunUnit(**d)


def _load_config(run_dir: Path) -> Dict[str, Any]:
    config_json = run_dir / "config.json"
    if not config_json.exists():
        raise FileNotFoundError(f"config.json not found in {run_dir}. Run init first.")
    return json.loads(config_json.read_text())


def _make_runtime_config(spec: Dict[str, Any], unit: RunUnit) -> RuntimeConfig:
    base = dict(spec.get("runtime_defaults", {}))
    base.update(dict(unit.runtime_overrides or {}))
    base["mode"] = unit.mode
    return RuntimeConfig(**base)


def _make_backbone(spec: Dict[str, Any], *, mock_llm: bool) -> Optional[BackboneAdapter]:
    model_cfg = dict(spec.get("model", {}))
    if not model_cfg:
        return None

    bb_cfg = BackboneConfig(
        base_url=str(model_cfg.get("base_url", "http://localhost:8000/v1")),
        model=str(model_cfg.get("model", "default")),
        api_key=str(model_cfg.get("api_key", "EMPTY")),
        temperature=float(model_cfg.get("temperature", 0.0)),
        timeout=float(model_cfg.get("timeout", 120.0)),
    )
    return BackboneAdapter(config=bb_cfg, mock=mock_llm, enable_cache=bool(model_cfg.get("enable_cache", True)))


def _make_adapter(spec: Dict[str, Any], run_dir: Path, unit: RunUnit) -> RulerSyntheticAdapter:
    bench_cfg = dict(spec.get("benchmark", {}))
    bench_cfg.update(dict(unit.benchmark_overrides or {}))
    name = str(unit.benchmark)
    if name != "ruler_synthetic":
        raise ValueError(f"Only benchmark=ruler_synthetic is implemented (got {name!r})")

    ruler_dir = Path(bench_cfg.get("ruler_dir", "outside_data/RULER")).resolve()
    dataset_root = run_dir / "datasets"

    ds_spec = RulerDatasetSpec(
        task_id=unit.task_id,
        split=unit.split,
        max_seq_length=unit.max_seq_length,
        num_samples=unit.num_samples,
        seed=unit.seed,
    )

    return RulerSyntheticAdapter(
        ruler_dir=ruler_dir,
        dataset_root=dataset_root,
        spec=ds_spec,
        benchmark_name=str(bench_cfg.get("benchmark_name", "synthetic")),
        tokenizer_type=str(bench_cfg.get("tokenizer_type", "openai")),
        tokenizer_path=str(bench_cfg.get("tokenizer_path", "cl100k_base")),
        model_template_type=str(bench_cfg.get("model_template_type", "base")),
        ensure_prepared=bool(bench_cfg.get("ensure_prepared", True)),
    )


def _select_units(run_dir: Path, args: argparse.Namespace) -> List[RunUnit]:
    shard_index = args.shard_index
    shard_count = args.shard_count
    if (shard_index is None) != (shard_count is None):
        raise SystemExit("Provide both --shard-index and --shard-count (or neither).")
    if shard_count is not None:
        if shard_count <= 0:
            raise SystemExit("--shard-count must be > 0")
        if shard_index < 0 or shard_index >= shard_count:
            raise SystemExit("--shard-index must be in [0, --shard-count)")

    max_units = args.max_units
    if max_units is not None and max_units <= 0:
        raise SystemExit("--max-units must be > 0")

    max_problems = args.max_problems
    if max_problems is not None and max_problems <= 0:
        raise SystemExit("--max-problems must be > 0")

    selected: List[RunUnit] = []
    for unit in _iter_units(run_dir):
        if args.unit_id and unit.unit_id != args.unit_id:
            continue
        if args.phase_id and unit.phase_id != args.phase_id:
            continue
        if args.mode and unit.mode != args.mode:
            continue
        if args.task_id and unit.task_id != args.task_id:
            continue
        if args.max_seq_length is not None and unit.max_seq_length != args.max_seq_length:
            continue
        if args.seed is not None and unit.seed != args.seed:
            continue
        if args.split and unit.split != args.split:
            continue

        if shard_count is not None:
            unit_num = int(unit.unit_id[1:])  # u000001 -> 1
            if (unit_num - 1) % shard_count != shard_index:
                continue

        if args.skip_done:
            if (run_dir / "units" / unit.unit_id / "metrics_partial.json").exists():
                continue

        selected.append(unit)
        if max_units is not None and len(selected) >= max_units:
            break

    if not selected:
        raise SystemExit("No units selected (check --unit-id/--phase-id/--mode filters).")
    return selected


def _prepare_spec_for_run(
    spec: Dict[str, Any],
    *,
    model_base_url: Optional[str],
) -> Dict[str, Any]:
    """Create per-run spec copy with optional model endpoint override."""
    prepared = dict(spec)
    model_cfg = dict(prepared.get("model", {}))
    if model_base_url:
        model_cfg["base_url"] = str(model_base_url)
    prepared["model"] = model_cfg
    return prepared


def _resolve_model_base_url(
    spec: Dict[str, Any],
    *,
    args: argparse.Namespace,
    defaults: Dict[str, Any],
) -> str:
    explicit_backend = (
        _normalize_backend_name(args.backend, default="")
        if getattr(args, "backend", None)
        else ""
    )
    base_url = str(
        args.model_base_url
        or (spec.get("model", {}) or {}).get("base_url")
        or _default_backend_base_url(
            explicit_backend or defaults.get("task_backend", "vllm"),
            defaults,
        )
    )

    if explicit_backend and not args.model_base_url:
        base_url = _default_backend_base_url(explicit_backend, defaults)

    if bool(args.mock_llm):
        return base_url

    if _endpoint_ready(base_url):
        return base_url

    fallback_backend = str(getattr(args, "backend_fallback", "none") or "none").strip().lower()
    if fallback_backend in {"vllm", "sglang"}:
        fallback_url = _default_backend_base_url(fallback_backend, defaults)
        if fallback_url != base_url and _endpoint_ready(fallback_url):
            print(
                f"Primary endpoint {base_url} unavailable; falling back to {fallback_url}",
                file=sys.stderr,
            )
            return fallback_url
    return base_url


def _build_server_manager(
    backend: str,
    *,
    args: argparse.Namespace,
    defaults: Dict[str, Any],
):
    from src.benchmark.throughput import SGLangServerManager, VLLMServerManager

    profile = str(args.start_server)
    port = (
        int(args.server_port)
        if args.server_port is not None
        else _default_backend_port(backend, defaults)
    )
    if _normalize_backend_name(backend) == "sglang":
        return SGLangServerManager(
            profile=profile,
            port=port,
            cuda_devices=args.cuda_devices,
            venv_path=str(args.sglang_venv_path or defaults.get("sglang_venv_path")),
        )
    return VLLMServerManager(
        profile=profile,
        port=port,
        cuda_devices=args.cuda_devices,
        venv_path=str(args.vllm_venv_path or defaults.get("vllm_venv_path")),
    )


def _run_selected_units(
    *,
    run_dir: Path,
    spec: Dict[str, Any],
    selected: List[RunUnit],
    args: argparse.Namespace,
    model_base_url: Optional[str] = None,
) -> None:
    trace = TraceWriter(run_dir)
    prepared_spec = _prepare_spec_for_run(spec, model_base_url=model_base_url)
    backbone = _make_backbone(prepared_spec, mock_llm=bool(args.mock_llm))

    for i, unit in enumerate(selected, start=1):
        print(
            f"[{i}/{len(selected)}] unit={unit.unit_id} phase={unit.phase_id} "
            f"task={unit.task_id} len={unit.max_seq_length} seed={unit.seed} mode={unit.mode}"
        )
        adapter = _make_adapter(prepared_spec, run_dir, unit)
        runtime = _make_runtime_config(prepared_spec, unit)
        run_unit(
            unit=unit,
            run_dir=run_dir,
            adapter=adapter,
            runtime=runtime,
            trace=trace,
            backbone=backbone,
            limit_problems=args.max_problems,
        )


def cmd_run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    spec = _load_config(run_dir)
    defaults = _load_inference_backend_defaults()
    selected = _select_units(run_dir, args)

    if args.dry_run:
        for u in selected:
            print(
                f"unit={u.unit_id} phase={u.phase_id} task={u.task_id} len={u.max_seq_length} seed={u.seed} mode={u.mode}"
            )
        return

    # Optional auto-started backend manager path.
    if args.start_server:
        primary_backend = _normalize_backend_name(
            args.backend or defaults.get("task_backend", "vllm"),
            default="vllm",
        )
        fallback_backend = str(getattr(args, "backend_fallback", "none") or "none").strip().lower()
        backends_to_try = [primary_backend]
        if fallback_backend in {"vllm", "sglang"} and fallback_backend != primary_backend:
            backends_to_try.append(fallback_backend)

        last_error: Optional[Exception] = None
        for idx, backend_name in enumerate(backends_to_try):
            manager = _build_server_manager(backend_name, args=args, defaults=defaults)
            try:
                print(
                    f"Starting managed backend={backend_name} profile={args.start_server} "
                    f"port={getattr(manager, 'port', 'auto')}",
                    file=sys.stderr,
                )

                async def _run_managed() -> None:
                    async with manager as server:
                        _run_selected_units(
                            run_dir=run_dir,
                            spec=spec,
                            selected=selected,
                            args=args,
                            model_base_url=str(server.url),
                        )

                asyncio.run(_run_managed())
                return
            except Exception as exc:
                last_error = exc
                if idx < len(backends_to_try) - 1:
                    print(
                        f"Managed backend {backend_name} failed ({exc}); trying fallback backend {backends_to_try[idx + 1]}",
                        file=sys.stderr,
                    )
                    continue
                raise

        if last_error is not None:
            raise last_error

    model_base_url = _resolve_model_base_url(spec, args=args, defaults=defaults)
    _run_selected_units(
        run_dir=run_dir,
        spec=spec,
        selected=selected,
        args=args,
        model_base_url=model_base_url,
    )


def cmd_aggregate(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()

    units_dir = run_dir / "units"
    if not units_dir.exists():
        raise SystemExit(f"No units dir found at {units_dir}; run some units first.")

    # Merge steps + predictions.
    merged_steps = run_dir / "steps.jsonl"
    merged_preds = run_dir / "predictions.jsonl"
    merged_steps.unlink(missing_ok=True)
    merged_preds.unlink(missing_ok=True)

    steps_writer = JsonlWriter(merged_steps)
    preds_writer = JsonlWriter(merged_preds)

    partial_metrics: List[Dict[str, Any]] = []

    for unit_dir in sorted(units_dir.iterdir()):
        if not unit_dir.is_dir():
            continue

        mp = unit_dir / "metrics_partial.json"
        if mp.exists():
            partial_metrics.append(json.loads(mp.read_text()))

        steps = unit_dir / "steps.jsonl"
        if steps.exists():
            for line in steps.read_text().splitlines():
                if line.strip():
                    steps_writer.write(json.loads(line))

        preds = unit_dir / "predictions.jsonl"
        if preds.exists():
            for line in preds.read_text().splitlines():
                if line.strip():
                    preds_writer.write(json.loads(line))

    # Aggregate metrics from merged predictions (preferred: consistent across adapters).
    primary_scores: List[float] = []
    by_phase: Dict[str, List[float]] = {}
    by_task_len_mode: Dict[str, List[float]] = {}

    if merged_preds.exists():
        with merged_preds.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                metrics = r.get("metrics", {})
                primary_name = str(r.get("primary_metric") or "")
                if not primary_name:
                    primary_name = next(iter(metrics.keys())) if metrics else "score"

                score = float(metrics.get(primary_name, 0.0)) if metrics else 0.0

                primary_scores.append(score)

                phase = str(r.get("phase_id", ""))
                by_phase.setdefault(phase, []).append(score)

                key = f"{r.get('task_id')}|{r.get('max_seq_length')}|{r.get('mode')}"
                by_task_len_mode.setdefault(key, []).append(score)

    metrics_out: Dict[str, Any] = {
        "n_units": len(list(units_dir.iterdir())),
        "n_predictions": len(primary_scores),
        "primary_mean": sum(primary_scores) / max(1, len(primary_scores)),
        "by_phase": {k: sum(v) / max(1, len(v)) for k, v in by_phase.items()},
        "by_task_len_mode": {k: sum(v) / max(1, len(v)) for k, v in by_task_len_mode.items()},
        "partial_units": partial_metrics,
    }

    _write_json(run_dir / "metrics.json", metrics_out)

    print(f"Wrote merged predictions: {merged_preds}")
    print(f"Wrote merged steps: {merged_steps}")
    print(f"Wrote metrics: {run_dir / 'metrics.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Runtime benchmark evaluation harness.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    backend_defaults = _load_inference_backend_defaults()
    fallback_default = str(backend_defaults.get("fallback_backend", "none")).strip().lower()
    if fallback_default not in {"vllm", "sglang"}:
        fallback_default = "none"

    p_init = sub.add_parser("init", help="Initialize a run: expand phases into units.jsonl")
    p_init.add_argument("--config", required=True, help="Path to run config YAML")
    p_init.add_argument("--output-dir", default="outputs/evals", help="Root output dir")
    p_init.add_argument("--run-id", default=None, help="Override run_id")
    p_init.set_defaults(fn=cmd_init)

    p_run = sub.add_parser("run", help="Run one or more units from an initialized run dir")
    p_run.add_argument("--run-dir", required=True, help="Run directory created by init")
    p_run.add_argument("--unit-id", default=None, help="Run only one unit (e.g., u000001)")
    p_run.add_argument("--phase-id", default=None, help="Run only units from one phase")
    p_run.add_argument("--mode", default=None, help="Run only units with this mode")
    p_run.add_argument("--task-id", default=None, help="Run only units for one task_id (e.g., vt)")
    p_run.add_argument("--max-seq-length", type=int, default=None, help="Run only units with this max_seq_length")
    p_run.add_argument("--seed", type=int, default=None, help="Run only units with this seed")
    p_run.add_argument("--split", default=None, help="Run only units with this split")
    p_run.add_argument("--shard-index", type=int, default=None, help="Shard index (0-based)")
    p_run.add_argument("--shard-count", type=int, default=None, help="Total shards")
    p_run.add_argument("--max-units", type=int, default=None, help="Run at most N selected units")
    p_run.add_argument("--max-problems", type=int, default=None, help="Run only first N problems per unit")
    p_run.add_argument("--skip-done", action="store_true", help="Skip units with existing metrics_partial.json")
    p_run.add_argument("--dry-run", action="store_true", help="Print selected units and exit")
    p_run.add_argument("--mock-llm", action="store_true", help="Use MockLLMClient (no server required)")
    p_run.add_argument("--model-base-url", default=None, help="Override model.base_url for this run")
    p_run.add_argument(
        "--backend",
        choices=["vllm", "sglang"],
        default=None,
        help="Backend hint for default endpoint selection when --model-base-url is omitted.",
    )
    p_run.add_argument(
        "--backend-fallback",
        choices=["none", "vllm", "sglang"],
        default=fallback_default,
        help="Fallback backend when primary endpoint/manager is unavailable.",
    )
    p_run.add_argument(
        "--start-server",
        default=None,
        help="Auto-start a managed backend server from profile name (stopped on exit).",
    )
    p_run.add_argument(
        "--server-port",
        type=int,
        default=None,
        help="Port override for --start-server (default uses backend defaults).",
    )
    p_run.add_argument("--cuda-devices", default=None, help="CUDA_VISIBLE_DEVICES for --start-server")
    p_run.add_argument(
        "--vllm-venv-path",
        default=str(backend_defaults.get("vllm_venv_path", "/home/mlinegar/vllm-env")),
        help="vLLM virtualenv path for managed startup.",
    )
    p_run.add_argument(
        "--sglang-venv-path",
        default=str(backend_defaults.get("sglang_venv_path", "/home/mlinegar/sglang-env")),
        help="SGLang virtualenv path for managed startup.",
    )
    p_run.set_defaults(fn=cmd_run)

    p_agg = sub.add_parser("aggregate", help="Aggregate unit artifacts into run-level files")
    p_agg.add_argument("--run-dir", required=True, help="Run directory")
    p_agg.set_defaults(fn=cmd_aggregate)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
