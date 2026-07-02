from __future__ import annotations

import importlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from treepo_cdx._json import jsonable
from treepo_cdx.adapters import local_law_rows_from_manifest, local_law_rows_from_mappings
from treepo_cdx.audit import compute_influence_weighted_overlap
from treepo_cdx.local_law import local_law_objective_summary
from treepo_cdx.manifest import (
    ArtifactLineage,
    ArtifactRef,
    ManifestRow,
    RoleTuple,
    RunManifestContract,
    Span,
    TopLevelUnit,
)
from treepo_cdx.sketches import hll_fit_summary


@dataclass(frozen=True)
class FitConfig:
    mode: str = ""
    experiment: str = ""
    config: Mapping[str, Any] = field(default_factory=dict)
    output_dir: str | Path = "outputs/treepo_cdx_fit"
    json_out: str | Path | None = None
    csv_out: str | Path | None = None
    spec: Mapping[str, Any] = field(default_factory=dict)
    train_data: Any = None
    eval_data: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FitConfig":
        data = dict(payload or {})
        nested = data.get("fit")
        if isinstance(nested, Mapping):
            merged = dict(data)
            merged.update(dict(nested))
            data = merged
        config = data.get("config")
        spec = data.get("spec")
        if config is None and not spec:
            config = {
                k: v
                for k, v in data.items()
                if k
                not in {
                    "mode",
                    "kind",
                    "experiment",
                    "output_dir",
                    "json_out",
                    "csv_out",
                    "metadata",
                    "fit",
                }
            }
        return cls(
            mode=str(data.get("mode") or data.get("kind") or ""),
            experiment=str(data.get("experiment") or ""),
            config=dict(config or {}),
            output_dir=data.get("output_dir") or "outputs/treepo_cdx_fit",
            json_out=data.get("json_out"),
            csv_out=data.get("csv_out"),
            spec=dict(spec or {}),
            train_data=data.get("train_data"),
            eval_data=data.get("eval_data"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class FitResult:
    status: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    history: tuple[Mapping[str, Any], ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)
    manifest_path: str | None = None
    mode: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "mode": str(self.mode),
            "metrics": dict(self.metrics or {}),
            "artifacts": dict(self.artifacts or {}),
            "history": [dict(item) for item in self.history],
            "summary": jsonable(dict(self.summary or {})),
            "manifest_path": self.manifest_path,
        }


def _ensure_monorepo_paths() -> None:
    """Make sibling monorepo packages importable when treepo_cdx is run in place."""

    for parent in Path(__file__).resolve().parents:
        repo_root = parent
        treepo_src = repo_root / "treepo" / "src"
        if treepo_src.exists():
            for path in (str(repo_root), str(treepo_src)):
                if path not in sys.path:
                    sys.path.insert(0, path)
            return


def _infer_mode(cfg: FitConfig, *, task: str | None = None, backend: str | None = None) -> str:
    if cfg.mode:
        return str(cfg.mode).strip().lower()
    if task:
        return "paper_experiment"
    payload = dict(cfg.config or cfg.spec or {})
    if "methods" in payload and "benchmark" in payload:
        return "runtime"
    if "family" in payload and ("schedule" in payload or "backend_config" in payload):
        return "learning"
    if "local_law_rows" in payload:
        return "local_law"
    if "leaf_token_lists" in payload or "token_ids" in payload:
        return "hll_sketch"
    if cfg.experiment or "experiment" in payload:
        return "paper_experiment"
    if backend in {"runtime", "longbench"}:
        return "runtime"
    raise ValueError("could not infer fit mode; provide mode='paper_experiment', 'runtime', or 'learning'")


def _as_fit_config(config: FitConfig | Mapping[str, Any] | None, **kwargs: Any) -> FitConfig:
    payload: dict[str, Any] = {}
    if isinstance(config, FitConfig):
        payload = asdict(config)
    elif isinstance(config, Mapping):
        payload = dict(config)
    elif config is not None:
        raise TypeError(f"fit config must be a mapping or FitConfig, got {type(config).__name__}")
    payload.update({k: v for k, v in kwargs.items() if v is not None})
    return FitConfig.from_mapping(payload)


def fit(
    config: FitConfig | Mapping[str, Any] | None = None,
    *,
    task: str | None = None,
    backend: str | None = None,
    output_dir: str | Path | None = None,
    train_data: Any = None,
    eval_data: Any = None,
    **kwargs: Any,
) -> FitResult:
    """Run a TreePO exercise through the CDX package facade.

    The facade follows proven repo shapes:

    - `{"experiment": ..., "config": ...}` -> `treepo.bench.runner`
    - runtime configs with `benchmark` and `methods` -> `treepo.runtime`
    - f/g learning specs with `family` and `schedule` -> `src.ctreepo.learning`
    """

    cfg = _as_fit_config(
        config,
        output_dir=output_dir,
        train_data=train_data,
        eval_data=eval_data,
        **kwargs,
    )
    mode = _infer_mode(cfg, task=task, backend=backend)
    if mode in {"paper", "paper_experiment", "bench", "suite"}:
        return _fit_paper_experiment(cfg, task=task, mode=mode)
    if mode in {"runtime", "longbench", "runtime_eval"}:
        return _fit_runtime(cfg, mode=mode)
    if mode in {"learning", "ladder", "family_runtime", "fg"}:
        return _fit_learning(cfg, mode=mode)
    if mode in {"local_law", "law_objective", "audit"}:
        return _fit_local_law(cfg, mode=mode)
    if mode in {"hll", "hll_sketch", "sketch"}:
        return _fit_hll_sketch(cfg, mode=mode)
    raise ValueError(f"unsupported fit mode: {mode!r}")


def _fit_paper_experiment(cfg: FitConfig, *, task: str | None, mode: str) -> FitResult:
    _ensure_monorepo_paths()
    runner = importlib.import_module("treepo.bench.runner")
    experiment = str(task or cfg.experiment or cfg.config.get("experiment") or "")
    if not experiment:
        raise ValueError("paper_experiment fit requires an experiment/task name")
    run_config = dict(cfg.config.get("config") or cfg.config)
    run_config.pop("experiment", None)
    output_root = Path(cfg.output_dir)
    json_out = Path(cfg.json_out) if cfg.json_out is not None else output_root / experiment / "summary.json"
    csv_out = Path(cfg.csv_out) if cfg.csv_out is not None else output_root / experiment / "summary.csv"
    result = runner.run_single(
        experiment=experiment,
        config=run_config,
        json_out=json_out,
        csv_out=csv_out,
    )
    result_out = FitResult(
        status=str(result.get("status", "ok")),
        artifacts={"json_out": str(json_out), "csv_out": str(csv_out)},
        summary=_read_json(json_out),
        mode=mode,
    )
    return _write_fit_sidecars(result_out, output_root=output_root / experiment)


def _fit_runtime(cfg: FitConfig, *, mode: str) -> FitResult:
    _ensure_monorepo_paths()
    runtime = importlib.import_module("treepo.runtime")
    run_config = dict(cfg.config or cfg.spec or {})
    output_root = Path(cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = runtime.run_runtime_eval(run_config)
    payload = summary.to_dict()
    json_out = Path(cfg.json_out) if cfg.json_out is not None else output_root / "runtime_summary.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = {str(k): float(v) for k, v in dict(summary.metrics).items() if isinstance(v, (int, float))}
    manifest = _manifest_from_runtime_summary(payload, run_config=run_config)
    result = FitResult(
        status="ok",
        metrics=metrics,
        artifacts={"json_out": str(json_out)},
        summary=payload,
        mode=mode,
    )
    return _write_fit_sidecars(result, output_root=output_root, manifest=manifest)


def _fit_learning(cfg: FitConfig, *, mode: str) -> FitResult:
    _ensure_monorepo_paths()
    learning = importlib.import_module("src.ctreepo.learning")
    spec = dict(cfg.spec or cfg.config or {})
    if cfg.train_data is not None:
        spec["train_data"] = cfg.train_data
    if cfg.eval_data is not None:
        spec["eval_data"] = cfg.eval_data
    result = learning.fit(spec, output_dir=cfg.output_dir)
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    result_out = FitResult(
        status=str(payload.get("status", "ok")),
        metrics=dict(payload.get("metrics") or {}),
        artifacts=dict(payload.get("artifacts") or {}),
        history=tuple(dict(item) for item in list(payload.get("history") or [])),
        summary=dict(payload.get("summary") or {}),
        manifest_path=payload.get("manifest_path"),
        mode=mode,
    )
    return _write_fit_sidecars(result_out, output_root=Path(cfg.output_dir))


def _fit_local_law(cfg: FitConfig, *, mode: str) -> FitResult:
    run_config = dict(cfg.config or cfg.spec or {})
    rows_payload = run_config.get("local_law_rows", run_config.get("rows"))
    if rows_payload is None:
        manifest_payload = run_config.get("manifest")
        manifest_path = run_config.get("manifest_path")
        if manifest_payload is None and manifest_path:
            manifest_payload = _read_json(Path(manifest_path))
        if isinstance(manifest_payload, Mapping):
            manifest = RunManifestContract.from_dict(manifest_payload)
            rows = local_law_rows_from_manifest(
                manifest.rows,
                strict=bool(run_config.get("strict", True)),
            )
        else:
            raise ValueError("local_law fit requires local_law_rows, rows, manifest, or manifest_path")
    else:
        if not isinstance(rows_payload, list | tuple):
            raise TypeError("local_law rows must be a sequence of mappings")
        rows = local_law_rows_from_mappings(tuple(dict(item) for item in rows_payload))

    objective = local_law_objective_summary(
        rows,
        gamma_depth=float(run_config.get("gamma_depth", 1.0)),
        objective_mode=str(run_config.get("objective_mode") or "corrected_local_law"),
    )
    overlap = compute_influence_weighted_overlap(rows)
    metrics = {
        "objective": float(objective.objective),
        "row_count": float(objective.row_count),
        "observed_count": float(objective.observed_count),
        "D_lambda": float(overlap.D_lambda),
        "W_lambda": float(overlap.W_lambda),
    }
    result_out = FitResult(
        status="ok",
        metrics=metrics,
        artifacts={},
        summary={
            "local_law_objective": objective.to_dict(),
            "influence_weighted_overlap": overlap.to_dict(),
        },
        mode=mode,
    )
    return _write_fit_sidecars(result_out, output_root=Path(cfg.output_dir))


def _fit_hll_sketch(cfg: FitConfig, *, mode: str) -> FitResult:
    run_config = dict(cfg.config or cfg.spec or {})
    leaves = run_config.get("leaf_token_lists")
    if leaves is None:
        leaves = (tuple(int(token) for token in list(run_config.get("token_ids") or ())),)
    if not isinstance(leaves, list | tuple):
        raise TypeError("hll_sketch fit requires leaf_token_lists as a sequence of token sequences")
    summary = hll_fit_summary(
        tuple(tuple(int(token) for token in leaf) for leaf in leaves),
        precision=int(run_config.get("precision", 10)),
        hash_bits=int(run_config.get("hash_bits", 64)),
        schedule=str(run_config.get("schedule") or "balanced"),
    )
    result_out = FitResult(
        status="ok",
        metrics={
            "estimate": float(summary["estimate"]),
            "true_cardinality": float(summary["true_cardinality"]),
            "abs_error": float(summary["abs_error"]),
            "rel_error": float(summary["rel_error"]),
            "memory_bytes": float(summary["memory_bytes"]),
        },
        artifacts={},
        summary=summary,
        mode=mode,
    )
    return _write_fit_sidecars(result_out, output_root=Path(cfg.output_dir))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return payload if isinstance(payload, dict) else {"value": payload}


def _write_fit_sidecars(
    result: FitResult,
    *,
    output_root: Path,
    manifest: RunManifestContract | None = None,
) -> FitResult:
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = dict(result.artifacts or {})
    manifest_path = result.manifest_path
    if manifest is not None:
        manifest_file = output_root / "run_manifest.json"
        report = manifest.validate()
        if not report.ok:
            raise ValueError(f"generated run manifest failed validation: {report.to_dict()}")
        manifest_file.write_text(
            json.dumps(jsonable(manifest.to_dict()), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_path = str(manifest_file)
        artifacts["run_manifest_json"] = str(manifest_file)
        artifacts["manifest_digest"] = manifest.digest

    result_file = output_root / "fit_result.json"
    sidecar = FitResult(
        status=result.status,
        metrics=dict(result.metrics or {}),
        artifacts={**artifacts, "fit_result_json": str(result_file)},
        history=tuple(dict(item) for item in result.history),
        summary=dict(result.summary or {}),
        manifest_path=manifest_path,
        mode=result.mode,
    )
    result_file.write_text(
        json.dumps(jsonable(sidecar.to_dict()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sidecar


def _manifest_from_runtime_summary(
    payload: Mapping[str, Any],
    *,
    run_config: Mapping[str, Any],
) -> RunManifestContract:
    predictions = [dict(item) for item in list(payload.get("predictions") or []) if isinstance(item, Mapping)]
    benchmark = dict(run_config.get("benchmark") or {})
    oracle = dict(run_config.get("oracle") or {})
    defaults = dict(run_config.get("runtime_defaults") or {})
    experiment_id = str(payload.get("experiment_id") or run_config.get("experiment_id") or "runtime_eval")
    split = str(benchmark.get("split") or "runtime")

    lengths: dict[str, int] = {}
    metadata_by_problem: dict[str, dict[str, Any]] = {}
    for pred in predictions:
        problem_id = str(pred.get("problem_id") or "")
        if not problem_id:
            continue
        artifacts = dict(pred.get("artifacts") or {})
        n_chars = _positive_int(artifacts.get("n_context_chars"), default=1)
        lengths[problem_id] = max(lengths.get(problem_id, 1), n_chars)
        metadata_by_problem.setdefault(
            problem_id,
            {
                "domain": pred.get("domain", ""),
                "difficulty": pred.get("difficulty", ""),
                "length": pred.get("length", ""),
            },
        )

    units = tuple(
        TopLevelUnit(
            unit_id=problem_id,
            length=length,
            source_ref=str(benchmark.get("dataset") or ""),
            metadata=metadata_by_problem.get(problem_id, {}),
        )
        for problem_id, length in sorted(lengths.items())
    )

    artifacts = (
        ArtifactRef("chunker:runtime", kind="chunker", metadata={"benchmark": benchmark}),
        ArtifactRef("g:runtime", kind="g", metadata={"methods": list(run_config.get("methods") or [])}),
        ArtifactRef("f:runtime", kind="f", metadata={"scorer": dict(run_config.get("scorer") or {})}),
        ArtifactRef("oracle:runtime", kind="oracle", metadata=oracle),
        ArtifactRef("query_policy:runtime", kind="query_policy", metadata=defaults),
    )
    lineage = ArtifactLineage(
        chunker="chunker:runtime",
        g="g:runtime",
        f="f:runtime",
        oracle_online="oracle:runtime",
        oracle_eval="oracle:runtime",
        query_policy="query_policy:runtime",
    )

    rows: list[ManifestRow] = []
    for pred in predictions:
        problem_id = str(pred.get("problem_id") or "")
        if not problem_id:
            continue
        method_id = str(pred.get("method_id") or "method")
        length = lengths.get(problem_id, 1)
        rows.append(
            ManifestRow(
                row_id=f"{method_id}:{problem_id}",
                top_level_unit_id=problem_id,
                fold_id=split,
                split_seed=0,
                roles=RoleTuple(chunker="eval", g="eval", oracle="eval"),
                artifacts=lineage,
                law_kind="runtime_prediction",
                support=Span(0, length, unit="char"),
                observed=True,
                propensity=1.0,
                truth_source=str(oracle.get("kind") or "benchmark_labels"),
                approx_source=method_id,
                metadata={
                    "answer": pred.get("answer", ""),
                    "correct": bool(pred.get("correct", False)),
                    "method_id": method_id,
                    "prediction": pred.get("prediction", ""),
                    "runtime_artifacts": dict(pred.get("artifacts") or {}),
                },
            )
        )

    return RunManifestContract(
        run_id=experiment_id,
        top_level_units=units,
        rows=tuple(rows),
        artifacts=artifacts,
        metadata={
            "mode": "runtime",
            "experiment_id": experiment_id,
            "benchmark": benchmark,
            "runtime_defaults": defaults,
            "method_metrics": list(payload.get("method_metrics") or []),
            "metrics": dict(payload.get("metrics") or {}),
        },
    )


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


__all__ = ["FitConfig", "FitResult", "fit"]
