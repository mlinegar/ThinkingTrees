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

from src.core.engines import (
    EngineSurface,
    EngineType,
    build_server_manager,
    default_engine_port,
    normalize_engine_name,
    normalize_fallback_engine_name,
    resolve_engine_base_url,
    resolve_engine_for_usage,
)
from src.experiments import (
    ArtifactRef,
    ProgressSnapshot,
    ResultRow,
    append_result_rows,
    canonical_artifact_refs_from_paths,
    merge_artifacts,
    write_experiment_manifest,
    write_experiment_status,
)
from src.experiments.legacy import runtime_run_spec_to_experiment
from src.runtime.adapters.registry import build_benchmark_adapter
from src.runtime.contracts import (
    RunPhaseSpec,
    RunSpec,
    RunUnit,
    RuntimeConfig,
    expand_units,
    units_digest,
)
from src.runtime.inference_context import (
    RuntimeInferenceContext,
    normalize_runtime_inference_spec,
    normalize_role_config,
    normalize_oracle_config,
    normalize_surface_config,
    validate_public_runtime_config,
)
from src.runtime.loop import run_unit
from src.runtime.methods import METHOD_COMPARE_RUNNER_ALIASES, PAPER_METHOD_ALIASES, MethodResources
from src.runtime.results import runtime_method_ref, runtime_role_metadata, runtime_surface_metadata
from src.runtime.trace import JsonlWriter, TraceWriter


def _load_inference_backend_defaults() -> Dict[str, Any]:
    """Best-effort backend defaults from settings, with safe fallbacks."""
    defaults = {
        "task_backend": "vllm",
        "fallback_backend": "none",
        "sglang_venv_path": "/home/mlinegar/sglang-env",
        "vllm_venv_path": "/home/mlinegar/vllm-env",
        "settings": {},
    }
    try:
        from src.config.settings import get_inference_backend_config, load_settings

        settings = load_settings()
        backend_cfg = get_inference_backend_config(settings)
        defaults.update(
            {
                "task_backend": str(backend_cfg.get("task_backend", "vllm")),
                "fallback_backend": str(backend_cfg.get("fallback_backend", "none")),
                "sglang_venv_path": str(
                    backend_cfg.get("sglang_venv_path") or defaults["sglang_venv_path"]
                ),
                "vllm_venv_path": str(
                    backend_cfg.get("vllm_venv_path") or defaults["vllm_venv_path"]
                ),
                "settings": settings,
            }
        )
    except Exception:
        pass
    return defaults


def _default_backend_port(backend: str, defaults: Optional[Dict[str, Any]] = None) -> int:
    cfg = defaults or _load_inference_backend_defaults()
    backend_name = normalize_engine_name(backend, default="vllm") or "vllm"
    return int(default_engine_port(backend_name, role="task", settings=cfg.get("settings")) or 0)


def _default_backend_base_url(backend: str, defaults: Optional[Dict[str, Any]] = None) -> str:
    cfg = defaults or _load_inference_backend_defaults()
    resolved = resolve_engine_base_url(
        normalize_engine_name(backend, default="vllm") or "vllm",
        surface=EngineSurface.CHAT_OPENAI,
        role="task",
        settings=cfg.get("settings"),
        host="localhost",
        port=_default_backend_port(backend, defaults),
    )
    if resolved:
        return resolved
    raise ValueError(
        f"Engine '{backend}' requires an explicit scorer endpoint for runtime evaluation."
    )


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


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _load_run_spec(
    config_path: Path,
    *,
    output_dir: Path,
    experiment_id: Optional[str],
) -> RunSpec:
    cfg = yaml.safe_load(config_path.read_text()) or {}
    validate_public_runtime_config(cfg)

    experiment_cfg = dict(cfg.get("experiment", {}) or {})
    experiment_id_final = (
        experiment_id
        or cfg.get("experiment_id")
        or experiment_cfg.get("experiment_id")
        or f"runtime_{_utc_ts()}"
    )

    benchmark_cfg = dict(cfg.get("benchmark", {}))
    runtime_defaults = dict(cfg.get("runtime_defaults", {}))
    roles_cfg = normalize_role_config(cfg)
    surfaces_cfg = normalize_surface_config(cfg)
    oracle_cfg = normalize_oracle_config(cfg)
    default_methods = _as_list(cfg.get("methods"))

    phases: List[RunPhaseSpec] = []
    for ph in cfg.get("phases", []):
        phase_methods = _as_list(ph.get("methods"))
        if not phase_methods:
            phase_methods = list(default_methods)
        if not phase_methods:
            raise ValueError(
                f"Phase {ph.get('phase_id', '<unknown>')!r} must define methods "
                "or use top-level methods."
            )
        phases.append(
            RunPhaseSpec(
                phase_id=str(ph["phase_id"]),
                tasks=_as_list(ph["tasks"]),
                lengths=[int(x) for x in ph["lengths"]],
                seeds=[int(x) for x in ph["seeds"]],
                num_samples=int(ph["num_samples"]),
                split=str(ph.get("split", "validation")),
                methods=[str(item) for item in phase_methods],
                runtime_overrides=dict(ph.get("runtime_overrides", {})),
                benchmark_overrides=dict(ph.get("benchmark_overrides", {})),
                runtime_grid=dict(ph.get("runtime_grid", {})),
                benchmark_grid=dict(ph.get("benchmark_grid", {})),
            )
        )

    configured_roles = set(roles_cfg)
    missing_by_method: Dict[str, List[str]] = {}
    for method in sorted({m for phase in phases for m in phase.methods}):
        missing = [
            role
            for role in _required_roles_for_method(method)
            if role != "state_model_or_embedder" and role not in configured_roles
        ]
        if "state_model_or_embedder" in _required_roles_for_method(method):
            if "state_model" not in configured_roles and "embedder" not in configured_roles:
                missing.append("state_model_or_embedder")
        if missing:
            missing_by_method[method] = missing
    if missing_by_method:
        rendered = "; ".join(
            f"{method}: {', '.join(missing)}"
            for method, missing in sorted(missing_by_method.items())
        )
        raise ValueError(f"Runtime config is missing required role(s): {rendered}")

    return RunSpec(
        run_id=str(experiment_id_final),
        created_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        output_dir=str(output_dir),
        benchmark=benchmark_cfg,
        runtime_defaults=runtime_defaults,
        phases=phases,
        surfaces=surfaces_cfg,
        roles=roles_cfg,
        oracle=oracle_cfg,
    )


def _runner_for_method(method: str) -> str:
    method = str(method)
    if method in PAPER_METHOD_ALIASES:
        return str(PAPER_METHOD_ALIASES[method])
    if method in METHOD_COMPARE_RUNNER_ALIASES:
        return str(METHOD_COMPARE_RUNNER_ALIASES[method][0])
    return method


def _required_roles_for_method(method: str) -> List[str]:
    runner = _runner_for_method(method)
    if runner in {"ground_truth", "empty"}:
        return []
    required = ["scorer"]
    if runner == "embedding_retrieval_llm":
        required.append("embedder")
    if runner in {
        "chunk_concat_baseline",
        "runtime_no_repair",
        "runtime_no_verifier",
        "runtime_full",
        "llm_tree_memory",
        "treepo_text_compressor_llm",
    }:
        required.append("summarizer")
    if runner == "neural_tree_selector_llm":
        required.append("state_model_or_embedder")
    return required


def _runtime_plan_payload(
    *,
    config_path: Path,
    output_dir: Path,
    experiment_id: Optional[str],
    check_endpoints: bool = False,
) -> Dict[str, Any]:
    spec = _load_run_spec(config_path, output_dir=output_dir, experiment_id=experiment_id)
    units = expand_units(spec)
    methods = sorted({unit.method for unit in units})
    ctx = RuntimeInferenceContext(spec.to_dict(), mock=False)
    roles = {
        key: {
            "config": value,
            "capabilities": ctx.role_capabilities(key).to_dict(),
            "ready": (
                _endpoint_ready(str(value.get("base_url", "") or ""))
                if check_endpoints and str(value.get("base_url", "") or "")
                else None
            ),
        }
        for key, value in ctx.roles.items()
    }
    surfaces = {
        key: {
            "config": value,
            "capabilities": ctx.capabilities(key).to_dict(),
            "ready": (
                _endpoint_ready(str(value.get("base_url", "") or ""))
                if check_endpoints and str(value.get("base_url", "") or "")
                else None
            ),
        }
        for key, value in ctx.surfaces.items()
    }
    missing: Dict[str, List[str]] = {}
    call_groups: Dict[str, List[str]] = {
        "scorer": [],
        "summarizer": [],
        "embedder": [],
        "state_model": [],
    }
    configured = set(ctx.roles)
    for method in methods:
        required = _required_roles_for_method(method)
        missing_for_method: List[str] = []
        for role in required:
            if role == "state_model_or_embedder":
                if "state_model" in configured:
                    call_groups["state_model"].append(method)
                elif "embedder" in configured:
                    call_groups["embedder"].append(method)
                else:
                    missing_for_method.append("state_model_or_embedder")
                continue
            call_groups.setdefault(role, []).append(method)
            if role not in configured:
                missing_for_method.append(role)
        if missing_for_method:
            missing[method] = missing_for_method
    return {
        "experiment_id": spec.run_id,
        "config_path": str(config_path),
        "unit_count": len(units),
        "methods": [
            {
                "method_id": method,
                "runner_id": _runner_for_method(method),
                "required_roles": _required_roles_for_method(method),
            }
            for method in methods
        ],
        "phases": [phase.to_dict() for phase in spec.phases],
        "roles": roles,
        "oracle": spec.oracle,
        "surfaces": surfaces,
        "missing_roles": missing,
        "call_groups": {key: sorted(set(value)) for key, value in call_groups.items() if value},
    }


def cmd_plan(args: argparse.Namespace) -> None:
    payload = _runtime_plan_payload(
        config_path=Path(args.config).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        experiment_id=args.experiment_id,
        check_endpoints=bool(args.check_endpoints),
    )
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return
    print(f"Runtime experiment: {payload['experiment_id']}")
    print(f"- Config: {payload['config_path']}")
    print(f"- Units: {payload['unit_count']}")
    print("- Methods:")
    for method in payload["methods"]:
        required = ", ".join(method["required_roles"]) or "none"
        print(f"  - {method['method_id']} -> {method['runner_id']} ({required})")
    print("- Roles:")
    for key, info in payload["roles"].items():
        caps = dict(info.get("capabilities") or {})
        ready = info.get("ready")
        suffix = "" if ready is None else f" ready={ready}"
        print(
            f"  - {key}: surface={caps.get('surface')} engine={caps.get('engine')} "
            f"model={caps.get('model')} batching={caps.get('supports_batching')}{suffix}"
        )
    print(f"- Oracle: {payload['oracle']}")
    if payload["missing_roles"]:
        print("- Missing roles:")
        for method, roles in payload["missing_roles"].items():
            print(f"  - {method}: {', '.join(roles)}")
    print("- Call groups:")
    for surface, methods in payload["call_groups"].items():
        print(f"  - {surface}: {', '.join(methods)}")


def _experiment_dir(output_dir: Path, experiment_id: str) -> Path:
    return output_dir / experiment_id


def _runtime_experiment_status(
    *,
    spec: RunSpec,
    run_dir: Path,
    state: str,
    active_phase: str = "",
    completed_items: int = 0,
    active_items: int = 0,
    pending_items: int = 0,
    failed_items: int = 0,
) -> ProgressSnapshot:
    total_units = len(expand_units(spec))
    finished = int(completed_items) + int(failed_items)
    percent_complete = 100.0 * float(finished) / float(total_units) if total_units > 0 else 100.0
    return ProgressSnapshot(
        experiment_id=str(spec.run_id),
        state=str(state),
        active_phase=str(active_phase),
        items_total=int(total_units),
        completed_items=int(completed_items),
        failed_items=int(failed_items),
        active_items=int(active_items),
        pending_items=int(pending_items),
        percent_complete=percent_complete,
        artifact_targets=("metrics_json", "merged_predictions_jsonl", "merged_calls_jsonl"),
        metadata={"adapter": "runtime_eval", "experiment_root": str(run_dir)},
    )


def cmd_init(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = _load_run_spec(config_path, output_dir=output_dir, experiment_id=args.experiment_id)
    units = expand_units(spec)
    digest = units_digest(units)

    run_dir = _experiment_dir(output_dir, spec.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_yaml(run_dir / "resolved_experiment.yaml", spec.to_dict())
    _write_json(run_dir / "config.json", spec.to_dict())

    units_path = run_dir / "units.jsonl"
    units_writer = JsonlWriter(units_path)
    for u in units:
        units_writer.write(u.to_dict())

    (run_dir / "units_digest.txt").write_text(digest + "\n")
    experiment_spec = runtime_run_spec_to_experiment(
        spec,
        launch_command=[
            sys.executable,
            "scripts/run_runtime_eval.py",
            "run",
            "--experiment-dir",
            str(run_dir),
        ],
    )
    write_experiment_manifest(run_dir, experiment_spec)
    write_experiment_status(
        run_dir,
        _runtime_experiment_status(
            spec=spec,
            run_dir=run_dir,
            state="initialized",
            pending_items=len(units),
        ),
    )
    merge_artifacts(
        run_dir,
        canonical_artifact_refs_from_paths(
            {
                "resolved_experiment_yaml": str(run_dir / "resolved_experiment.yaml"),
                "config_json": str(run_dir / "config.json"),
                "units_jsonl": str(units_path),
                "units_digest_txt": str(run_dir / "units_digest.txt"),
            },
            phase_id="init",
            required=True,
        ),
    )

    print(f"Initialized experiment {spec.run_id}")
    print(f"- Experiment dir: {run_dir}")
    print(f"- Units: {len(units)} ({units_path})")
    print(f"- Digest: {digest[:12]}… ({run_dir / 'units_digest.txt'})")


def _iter_units(run_dir: Path) -> Iterable[RunUnit]:
    units_path = run_dir / "units.jsonl"
    if not units_path.exists():
        raise FileNotFoundError(f"units.jsonl not found in {run_dir}. Initialize the experiment first.")
    with units_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            yield RunUnit(**d)


def _load_config(run_dir: Path) -> Dict[str, Any]:
    config_json = run_dir / "config.json"
    if not config_json.exists():
        raise FileNotFoundError(f"config.json not found in {run_dir}. Initialize the experiment first.")
    return json.loads(config_json.read_text())


def _make_runtime_config(spec: Dict[str, Any], unit: RunUnit) -> RuntimeConfig:
    base = dict(spec.get("runtime_defaults", {}))
    base.update(dict(unit.runtime_overrides or {}))
    base["method"] = unit.method
    return RuntimeConfig(**base)


def _make_adapter(spec: Dict[str, Any], run_dir: Path, unit: RunUnit):
    return build_benchmark_adapter(spec=spec, run_dir=run_dir, unit=unit)


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
        if args.method and unit.method != args.method:
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
        raise SystemExit("No units selected (check --unit-id/--phase-id/--method filters).")
    return selected


def _prepare_spec_for_run(
    spec: Dict[str, Any],
    *,
    scorer_endpoint: Optional[str],
) -> Dict[str, Any]:
    """Create per-run spec copy with optional scorer endpoint override."""
    prepared = dict(spec)
    roles_cfg = dict(prepared.get("roles", {}) or {})
    surfaces_cfg = dict(prepared.get("surfaces", {}) or {})
    if scorer_endpoint:
        scorer_cfg = dict(roles_cfg.get("scorer") or {})
        scorer_cfg["base_url"] = str(scorer_endpoint)
        roles_cfg["scorer"] = scorer_cfg
        summarizer_cfg = dict(roles_cfg.get("summarizer") or {})
        if summarizer_cfg.get("defaulted_from") == "scorer":
            summarizer_cfg["base_url"] = str(scorer_endpoint)
            roles_cfg["summarizer"] = summarizer_cfg
        chat_cfg = dict(surfaces_cfg.get("chat_openai") or {})
        chat_cfg["base_url"] = str(scorer_endpoint)
        surfaces_cfg["chat_openai"] = chat_cfg
    prepared["roles"] = roles_cfg
    prepared["surfaces"] = surfaces_cfg
    return prepared


def _resolve_scorer_endpoint(
    spec: Dict[str, Any],
    *,
    args: argparse.Namespace,
    defaults: Dict[str, Any],
) -> str:
    explicit_backend = None
    if getattr(args, "backend", None):
        explicit_backend = resolve_engine_for_usage(
            args.backend,
            surface=EngineSurface.CHAT_OPENAI,
            usage="runtime evaluation backend selection",
        ).engine.value
    base_url = str(
        args.scorer_endpoint
        or ((spec.get("roles", {}) or {}).get("scorer", {}) or {}).get("base_url")
        or ((spec.get("surfaces", {}) or {}).get("chat_openai", {}) or {}).get("base_url")
        or _default_backend_base_url(
            explicit_backend or str(defaults.get("task_backend", "vllm")),
            defaults,
        )
    )

    if explicit_backend and not args.scorer_endpoint:
        base_url = _default_backend_base_url(explicit_backend, defaults)

    if bool(args.mock_llm):
        return base_url

    if _endpoint_ready(base_url):
        return base_url

    raw_fallback_backend = getattr(
        args, "backend_fallback", defaults.get("fallback_backend", "none")
    )
    if str(raw_fallback_backend or "").strip().lower().replace("-", "_") not in {
        "",
        "none",
        "off",
        "disabled",
    }:
        fallback_backend = resolve_engine_for_usage(
            raw_fallback_backend,
            surface=EngineSurface.CHAT_OPENAI,
            usage="runtime evaluation fallback endpoint selection",
        ).engine.value
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
    profile = str(args.start_server)
    normalized = normalize_engine_name(backend, default="vllm") or "vllm"
    spec = resolve_engine_for_usage(
        normalized,
        surface=EngineSurface.CHAT_OPENAI,
        usage="runtime evaluation managed startup",
        require_managed=True,
    )
    port = (
        int(args.server_port)
        if args.server_port is not None
        else int(
            default_engine_port(spec.engine, role="task", settings=defaults.get("settings")) or 0
        )
    )
    venv_path = None
    if spec.engine is EngineType.SGLANG:
        venv_path = str(args.sglang_venv_path or defaults.get("sglang_venv_path"))
    elif spec.engine is EngineType.VLLM:
        venv_path = str(args.vllm_venv_path or defaults.get("vllm_venv_path"))
    return build_server_manager(
        spec.engine,
        profile=profile,
        port=port,
        cuda_devices=args.cuda_devices,
        venv_path=venv_path,
    )


def _run_selected_units(
    *,
    run_dir: Path,
    spec: Dict[str, Any],
    selected: List[RunUnit],
    args: argparse.Namespace,
    scorer_endpoint: Optional[str] = None,
) -> None:
    trace = TraceWriter(run_dir)
    prepared_spec = _prepare_spec_for_run(spec, scorer_endpoint=scorer_endpoint)
    prepared_spec = normalize_runtime_inference_spec(prepared_spec)
    inference_context = RuntimeInferenceContext(
        prepared_spec,
        mock=bool(args.mock_llm),
        call_sink=trace.write_call_record,
    )
    backbone = inference_context.backbone("scorer") if inference_context.has_role("scorer") else None
    summarizer_backbone = (
        inference_context.backbone("summarizer")
        if inference_context.has_role("summarizer")
        else None
    )
    embedding_client = (
        inference_context.embedding_client() if inference_context.has_role("embedder") else None
    )
    resources = MethodResources(
        backbone=backbone,
        summarizer_backbone=summarizer_backbone,
        embedding_client=embedding_client,
        inference_context=inference_context,
        run_config=prepared_spec,
        mock=bool(args.mock_llm),
    )

    for i, unit in enumerate(selected, start=1):
        print(
            f"[{i}/{len(selected)}] unit={unit.unit_id} phase={unit.phase_id} "
            f"task={unit.task_id} len={unit.max_seq_length} seed={unit.seed} method={unit.method}"
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
            resources=resources,
            limit_problems=args.max_problems,
        )


def cmd_run(args: argparse.Namespace) -> None:
    if not args.experiment_dir:
        raise SystemExit("missing --experiment-dir")
    run_dir = Path(args.experiment_dir).resolve()
    spec = _load_config(run_dir)
    defaults = _load_inference_backend_defaults()
    selected = _select_units(run_dir, args)
    run_spec = RunSpec(
        run_id=str(spec.get("experiment_id") or spec.get("run_id", "")),
        created_utc=str(spec.get("created_utc", "")),
        output_dir=str(spec.get("output_dir", "")),
        benchmark=dict(spec.get("benchmark", {}) or {}),
        runtime_defaults=dict(spec.get("runtime_defaults", {}) or {}),
        surfaces=dict(spec.get("surfaces", {}) or {}),
        roles=dict(spec.get("roles", {}) or {}),
        oracle=dict(spec.get("oracle", {}) or {}),
        phases=[
            RunPhaseSpec(
                phase_id=str(phase.get("phase_id", "")),
                tasks=list(phase.get("tasks", []) or []),
                lengths=[int(item) for item in list(phase.get("lengths", []) or [])],
                seeds=[int(item) for item in list(phase.get("seeds", []) or [])],
                num_samples=int(phase.get("num_samples", 0) or 0),
                split=str(phase.get("split", "validation") or "validation"),
                methods=list(phase.get("methods", []) or []),
                runtime_overrides=dict(phase.get("runtime_overrides", {}) or {}),
                benchmark_overrides=dict(phase.get("benchmark_overrides", {}) or {}),
                runtime_grid=dict(phase.get("runtime_grid", {}) or {}),
                benchmark_grid=dict(phase.get("benchmark_grid", {}) or {}),
            )
            for phase in list(spec.get("phases", []) or [])
        ],
    )
    write_experiment_status(
        run_dir,
        _runtime_experiment_status(
            spec=run_spec,
            run_dir=run_dir,
            state="running",
            active_phase=(selected[0].phase_id if selected else ""),
            active_items=len(selected),
            pending_items=max(0, len(expand_units(run_spec)) - len(selected)),
        ),
    )

    if args.dry_run:
        for u in selected:
            print(
                f"unit={u.unit_id} phase={u.phase_id} task={u.task_id} len={u.max_seq_length} seed={u.seed} method={u.method}"
            )
        return

    # Optional auto-started backend manager path.
    if args.start_server:
        primary_backend = resolve_engine_for_usage(
            args.backend or defaults.get("task_backend", "vllm"),
            surface=EngineSurface.CHAT_OPENAI,
            usage="runtime evaluation managed startup",
            require_managed=True,
        ).engine.value
        raw_fallback_backend = getattr(
            args, "backend_fallback", defaults.get("fallback_backend", "none")
        )
        fallback_backend = "none"
        if str(raw_fallback_backend or "").strip().lower().replace("-", "_") not in {
            "",
            "none",
            "off",
            "disabled",
        }:
            fallback_backend = resolve_engine_for_usage(
                raw_fallback_backend,
                surface=EngineSurface.CHAT_OPENAI,
                usage="runtime evaluation managed fallback startup",
                require_managed=True,
            ).engine.value
        backends_to_try = [primary_backend]
        if fallback_backend != "none" and fallback_backend != primary_backend:
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
                            scorer_endpoint=str(server.url),
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

    scorer_endpoint = _resolve_scorer_endpoint(spec, args=args, defaults=defaults)
    _run_selected_units(
        run_dir=run_dir,
        spec=spec,
        selected=selected,
        args=args,
        scorer_endpoint=scorer_endpoint,
    )
    completed_units = sum(
        1
        for unit in _iter_units(run_dir)
        if (run_dir / "units" / unit.unit_id / "metrics_partial.json").exists()
    )
    write_experiment_status(
        run_dir,
        _runtime_experiment_status(
            spec=run_spec,
            run_dir=run_dir,
            state="units_completed",
            completed_items=completed_units,
            pending_items=max(0, len(expand_units(run_spec)) - completed_units),
        ),
    )


def cmd_aggregate(args: argparse.Namespace) -> None:
    if not args.experiment_dir:
        raise SystemExit("missing --experiment-dir")
    run_dir = Path(args.experiment_dir).resolve()

    units_dir = run_dir / "units"
    if not units_dir.exists():
        raise SystemExit(f"No units dir found at {units_dir}; run some units first.")

    # Merge calls + steps + predictions.
    merged_calls = run_dir / "calls.jsonl"
    merged_steps = run_dir / "steps.jsonl"
    merged_preds = run_dir / "predictions.jsonl"
    merged_calls.unlink(missing_ok=True)
    merged_steps.unlink(missing_ok=True)
    merged_preds.unlink(missing_ok=True)

    calls_writer = JsonlWriter(merged_calls)
    steps_writer = JsonlWriter(merged_steps)
    preds_writer = JsonlWriter(merged_preds)

    partial_metrics: List[Dict[str, Any]] = []
    call_count = 0

    for unit_dir in sorted(units_dir.iterdir()):
        if not unit_dir.is_dir():
            continue

        mp = unit_dir / "metrics_partial.json"
        if mp.exists():
            partial_metrics.append(json.loads(mp.read_text()))

        calls = unit_dir / "calls.jsonl"
        if calls.exists():
            for line in calls.read_text().splitlines():
                if line.strip():
                    calls_writer.write(json.loads(line))
                    call_count += 1

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

    merged_calls.touch(exist_ok=True)
    merged_steps.touch(exist_ok=True)
    merged_preds.touch(exist_ok=True)

    # Aggregate metrics from merged predictions (preferred: consistent across adapters).
    primary_scores: List[float] = []
    by_phase: Dict[str, List[float]] = {}
    by_task_len_method: Dict[str, List[float]] = {}
    by_domain: Dict[str, List[float]] = {}
    by_difficulty: Dict[str, List[float]] = {}
    by_length_bucket: Dict[str, List[float]] = {}

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

                key = f"{r.get('task_id')}|{r.get('max_seq_length')}|{r.get('method')}"
                by_task_len_method.setdefault(key, []).append(score)
                problem_meta = dict((r.get("metadata") or {}).get("problem") or {})
                if problem_meta.get("domain") is not None:
                    by_domain.setdefault(str(problem_meta.get("domain")), []).append(score)
                if problem_meta.get("difficulty") is not None:
                    by_difficulty.setdefault(str(problem_meta.get("difficulty")), []).append(score)
                if problem_meta.get("length") is not None:
                    by_length_bucket.setdefault(str(problem_meta.get("length")), []).append(score)

    metrics_out: Dict[str, Any] = {
        "n_units": len(list(units_dir.iterdir())),
        "n_predictions": len(primary_scores),
        "n_surface_calls": call_count,
        "primary_mean": sum(primary_scores) / max(1, len(primary_scores)),
        "by_phase": {k: sum(v) / max(1, len(v)) for k, v in by_phase.items()},
        "by_task_len_method": {k: sum(v) / max(1, len(v)) for k, v in by_task_len_method.items()},
        "by_domain": {k: sum(v) / max(1, len(v)) for k, v in by_domain.items()},
        "by_difficulty": {k: sum(v) / max(1, len(v)) for k, v in by_difficulty.items()},
        "by_length_bucket": {k: sum(v) / max(1, len(v)) for k, v in by_length_bucket.items()},
        "partial_units": partial_metrics,
    }

    _write_json(run_dir / "metrics.json", metrics_out)
    merge_artifacts(
        run_dir,
        canonical_artifact_refs_from_paths(
            {
                "metrics_json": str(run_dir / "metrics.json"),
                "merged_calls_jsonl": str(merged_calls),
                "merged_steps_jsonl": str(merged_steps),
                "merged_predictions_jsonl": str(merged_preds),
            },
            phase_id="aggregate",
            required=True,
        ),
    )
    result_rows: list[ResultRow] = []
    config_json = _load_config(run_dir)
    benchmark_name = str(dict(config_json.get("benchmark", {}) or {}).get("name", "") or "")
    surfaces_json = dict(config_json.get("surfaces", {}) or {})
    roles_json = dict(config_json.get("roles", {}) or {})
    oracle_json = dict(config_json.get("oracle", {}) or {})
    surfaces_meta = runtime_surface_metadata(surfaces_json)
    roles_meta = runtime_role_metadata(roles_json)
    for raw_line in merged_preds.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        metrics = dict(payload.get("metrics", {}) or {})
        artifacts = dict((payload.get("metadata") or {}).get("artifacts") or {})
        method_id = str(payload.get("method", "") or "")
        runner_id = str(artifacts.get("runner_id", "") or "")
        for metric_name, metric_value in metrics.items():
            result_rows.append(
                ResultRow.from_dict(
                    {
                        "experiment_id": str(run_dir.name),
                        "phase": str(payload.get("phase_id", "") or ""),
                        "benchmark_ref": {
                            "benchmark_id": benchmark_name or "runtime_benchmark",
                            "family": "runtime_benchmark",
                            "scope": benchmark_name,
                            "name": benchmark_name,
                        },
                        "method_ref": runtime_method_ref(
                            method_id=method_id,
                            runner_id=runner_id,
                            surfaces=surfaces_json,
                            roles=roles_json,
                            oracle=oracle_json,
                            adapter="runtime_eval",
                        ),
                        "split": str(payload.get("split", "") or ""),
                        "seed": payload.get("seed"),
                        "train_docs": None,
                        "metric_name": str(metric_name),
                        "metric_value": metric_value,
                        "artifact_refs": ["merged_predictions_jsonl", "metrics_json"],
                        "metadata": {
                            "task_id": payload.get("task_id"),
                            "problem_id": payload.get("problem_id"),
                            "max_seq_length": payload.get("max_seq_length"),
                            "primary_metric": payload.get("primary_metric"),
                            "runner_id": runner_id,
                            "roles": roles_meta,
                            "oracle": oracle_json,
                            "surfaces": surfaces_meta,
                            **dict((payload.get("metadata") or {}).get("problem") or {}),
                        },
                    }
                )
            )
    append_result_rows(run_dir, result_rows)
    run_spec = RunSpec(
        run_id=str(config_json.get("experiment_id") or config_json.get("run_id", "")),
        created_utc=str(config_json.get("created_utc", "")),
        output_dir=str(config_json.get("output_dir", "")),
        benchmark=dict(config_json.get("benchmark", {}) or {}),
        runtime_defaults=dict(config_json.get("runtime_defaults", {}) or {}),
        surfaces=dict(config_json.get("surfaces", {}) or {}),
        roles=dict(config_json.get("roles", {}) or {}),
        oracle=dict(config_json.get("oracle", {}) or {}),
        phases=[
            RunPhaseSpec(
                phase_id=str(phase.get("phase_id", "")),
                tasks=list(phase.get("tasks", []) or []),
                lengths=[int(item) for item in list(phase.get("lengths", []) or [])],
                seeds=[int(item) for item in list(phase.get("seeds", []) or [])],
                num_samples=int(phase.get("num_samples", 0) or 0),
                split=str(phase.get("split", "validation") or "validation"),
                methods=list(phase.get("methods", []) or []),
                runtime_overrides=dict(phase.get("runtime_overrides", {}) or {}),
                benchmark_overrides=dict(phase.get("benchmark_overrides", {}) or {}),
                runtime_grid=dict(phase.get("runtime_grid", {}) or {}),
                benchmark_grid=dict(phase.get("benchmark_grid", {}) or {}),
            )
            for phase in list(config_json.get("phases", []) or [])
        ],
    )
    write_experiment_status(
        run_dir,
        _runtime_experiment_status(
            spec=run_spec,
            run_dir=run_dir,
            state="completed",
            completed_items=len(expand_units(run_spec)),
        ),
    )

    print(f"Wrote merged predictions: {merged_preds}")
    print(f"Wrote merged calls: {merged_calls}")
    print(f"Wrote merged steps: {merged_steps}")
    print(f"Wrote metrics: {run_dir / 'metrics.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Runtime benchmark evaluation harness.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    backend_defaults = _load_inference_backend_defaults()
    fallback_default = normalize_fallback_engine_name(
        backend_defaults.get("fallback_backend", "none"),
        default=None,
    )

    p_plan = sub.add_parser("plan", help="Resolve config without running units")
    p_plan.add_argument("--config", required=True, help="Path to experiment config YAML")
    p_plan.add_argument("--output-dir", default="outputs/evals", help="Root output dir")
    p_plan.add_argument("--experiment-id", default=None, help="Override experiment_id")
    p_plan.add_argument(
        "--check-endpoints",
        action="store_true",
        help="Probe configured /models endpoints where available.",
    )
    p_plan.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    p_plan.set_defaults(fn=cmd_plan)

    p_init = sub.add_parser("init", help="Initialize an experiment: expand phases into units.jsonl")
    p_init.add_argument("--config", required=True, help="Path to experiment config YAML")
    p_init.add_argument("--output-dir", default="outputs/evals", help="Root output dir")
    p_init.add_argument("--experiment-id", default=None, help="Override experiment_id")
    p_init.set_defaults(fn=cmd_init)

    p_run = sub.add_parser("run", help="Run one or more units from an initialized experiment")
    p_run.add_argument("--experiment-dir", default=None, help="Experiment directory created by init")
    p_run.add_argument("--unit-id", default=None, help="Run only one unit (e.g., u000001)")
    p_run.add_argument("--phase-id", default=None, help="Run only units from one phase")
    p_run.add_argument("--method", default=None, help="Run only units with this method")
    p_run.add_argument("--task-id", default=None, help="Run only units for one task_id (e.g., vt)")
    p_run.add_argument(
        "--max-seq-length", type=int, default=None, help="Run only units with this max_seq_length"
    )
    p_run.add_argument("--seed", type=int, default=None, help="Run only units with this seed")
    p_run.add_argument("--split", default=None, help="Run only units with this split")
    p_run.add_argument("--shard-index", type=int, default=None, help="Shard index (0-based)")
    p_run.add_argument("--shard-count", type=int, default=None, help="Total shards")
    p_run.add_argument("--max-units", type=int, default=None, help="Run at most N selected units")
    p_run.add_argument(
        "--max-problems", type=int, default=None, help="Run only first N problems per unit"
    )
    p_run.add_argument(
        "--skip-done", action="store_true", help="Skip units with existing metrics_partial.json"
    )
    p_run.add_argument("--dry-run", action="store_true", help="Print selected units and exit")
    p_run.add_argument(
        "--mock-llm", action="store_true", help="Use MockLLMClient (no server required)"
    )
    p_run.add_argument("--scorer-endpoint", default=None, help="Override scorer endpoint for this run")
    p_run.add_argument(
        "--backend",
        default=None,
        help="Engine hint for default endpoint selection when --scorer-endpoint is omitted.",
    )
    p_run.add_argument(
        "--backend-fallback",
        default=fallback_default,
        help="Fallback engine when the primary endpoint/manager is unavailable.",
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
    p_run.add_argument(
        "--cuda-devices", default=None, help="CUDA_VISIBLE_DEVICES for --start-server"
    )
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

    p_agg = sub.add_parser("aggregate", help="Aggregate unit artifacts into experiment-level files")
    p_agg.add_argument("--experiment-dir", default=None, help="Experiment directory")
    p_agg.set_defaults(fn=cmd_aggregate)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
