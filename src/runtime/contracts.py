from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import product
from uuid import uuid4
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.core.engines import EngineSurface, EngineType


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_hash(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProblemSpec:
    """A single benchmark/problem instance independent of model choice."""

    problem_id: str
    input_text: str
    query: str = ""

    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    success_metric: str = ""
    success_target: Any = None

    constraints: Dict[str, Any] = field(default_factory=dict)
    allowed_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NodeContract:
    """Defines what each tree node must preserve/produce."""

    objective: str
    must_preserve: List[str] = field(default_factory=list)
    output_schema: Dict[str, str] = field(default_factory=dict)
    acceptance_checks: List[str] = field(default_factory=list)
    max_input_tokens: int = 8192
    max_output_tokens: int = 256

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerifierResult:
    pass_: bool
    score: float = 0.0
    failures: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelResponse:
    text: str
    model_id: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    raw: Any = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Avoid serializing large raw objects by default.
        d["raw"] = None
        return d


@dataclass(frozen=True)
class ChatInput:
    """Typed chat input for the universal inference engine."""

    messages: List[Dict[str, str]]
    max_tokens: int = 256
    temperature: float = 0.0
    stop: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiffusionInput:
    """Typed diffusion input for `/generate`-style engines."""

    texts: List[str]
    sampling_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SymbolicInput:
    """Typed symbolic execution input for exact local engines."""

    operation: str
    inputs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TextOutput:
    """Single text completion."""

    text: str
    finish_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TextListOutput:
    """Ordered batch of generated texts."""

    texts: List[str]
    finish_reasons: List[Optional[str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructuredOutput:
    """Structured symbolic or tool-style output."""

    data: Any
    schema_name: str = ""
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": self.data,
            "schema_name": self.schema_name,
            "artifacts": dict(self.artifacts),
        }


InferenceInput = ChatInput | DiffusionInput | SymbolicInput
InferenceOutput = TextOutput | TextListOutput | StructuredOutput


@dataclass(frozen=True)
class InferenceRequest:
    """Universal typed inference request across chat, diffusion, and symbolic surfaces."""

    surface: EngineSurface
    input: InferenceInput
    engine_options: Dict[str, Any] = field(default_factory=dict)
    request_id: str = ""
    document_id: str = ""
    routing_key: str = ""
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def resolved_request_id(self) -> str:
        return self.request_id or uuid4().hex

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface": self.surface.value,
            "input": self.input.to_dict(),
            "engine_options": dict(self.engine_options),
            "request_id": self.request_id,
            "document_id": self.document_id,
            "routing_key": self.routing_key,
            "priority": int(self.priority),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class InferenceResponse:
    """Universal inference response envelope."""

    surface: EngineSurface
    engine: EngineType
    model_id: str = ""
    output: Optional[InferenceOutput] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    telemetry: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    request_id: str = ""

    def to_model_response(self) -> ModelResponse:
        if isinstance(self.output, TextOutput):
            text = self.output.text
        elif isinstance(self.output, TextListOutput):
            text = self.output.texts[0] if self.output.texts else ""
        else:
            raise TypeError(
                "InferenceResponse.to_model_response() requires a text output. "
                f"Received {type(self.output).__name__}."
            )
        return ModelResponse(
            text=text,
            model_id=self.model_id,
            prompt_tokens=int(self.usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(self.usage.get("completion_tokens", 0) or 0),
            latency_ms=float(self.latency_ms or 0.0),
            raw=None,
        )

    def to_dict(self) -> Dict[str, Any]:
        if hasattr(self.output, "to_dict"):
            output = self.output.to_dict()
        else:
            output = self.output
        return {
            "surface": self.surface.value,
            "engine": self.engine.value,
            "model_id": self.model_id,
            "output": output,
            "usage": dict(self.usage),
            "latency_ms": float(self.latency_ms),
            "telemetry": dict(self.telemetry),
            "artifacts": dict(self.artifacts),
            "request_id": self.request_id,
            "raw": None,
        }


@dataclass(frozen=True)
class PackedPrompt:
    messages: List[Dict[str, str]]
    max_output_tokens: int
    input_tokens_est: int
    packed_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeConfig:
    """Runtime settings shared across modes."""

    mode: str = "runtime_full"
    cap_tokens: int = 8192
    safety_tokens: int = 256
    max_output_tokens: int = 256

    # Compression/tree settings (free-form memory).
    chunk_tokens: int = 6144
    overlap_tokens: int = 256
    leaf_memory_tokens: int = 256
    merge_memory_tokens: int = 256
    tree_shape: str = "balanced"

    # Optional Engram-style static memory injection into leaf/merge prompts.
    engram_memory: bool = False
    engram_memory_max_items: int = 32
    engram_memory_max_chars: int = 800

    # Execution policy.
    verifier_enabled: bool = True
    repair_enabled: bool = True
    max_retries_per_step: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunUnit:
    """A single independent job in a run (suitable for job arrays)."""

    run_id: str
    unit_id: str
    phase_id: str

    benchmark: str
    task_id: str
    split: str
    max_seq_length: int
    seed: int
    num_samples: int

    mode: str
    runtime_overrides: Dict[str, Any] = field(default_factory=dict)
    benchmark_overrides: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunPhaseSpec:
    phase_id: str
    tasks: List[str]
    lengths: List[int]
    seeds: List[int]
    num_samples: int
    split: str
    modes: List[str]

    runtime_overrides: Dict[str, Any] = field(default_factory=dict)
    benchmark_overrides: Dict[str, Any] = field(default_factory=dict)
    runtime_grid: Dict[str, List[Any]] = field(default_factory=dict)
    benchmark_grid: Dict[str, List[Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunSpec:
    """Multi-phase run spec expanded into units.jsonl."""

    run_id: str
    created_utc: str
    output_dir: str

    benchmark: Dict[str, Any]
    model: Dict[str, Any]
    runtime_defaults: Dict[str, Any]
    phases: List[RunPhaseSpec]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["phases"] = [p.to_dict() for p in self.phases]
        return d


def expand_units(spec: RunSpec) -> List[RunUnit]:
    def _expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        if not grid:
            return [{}]
        keys = sorted(grid.keys())
        values = [grid[k] for k in keys]
        expanded: List[Dict[str, Any]] = []
        for combo in product(*values):
            expanded.append({k: v for k, v in zip(keys, combo)})
        return expanded

    units: List[RunUnit] = []
    unit_idx = 0
    for phase in spec.phases:
        runtime_variants = _expand_grid(phase.runtime_grid)
        benchmark_variants = _expand_grid(phase.benchmark_grid)
        for task_id in phase.tasks:
            for length in phase.lengths:
                for seed in phase.seeds:
                    for mode in phase.modes:
                        for r_var in runtime_variants:
                            for b_var in benchmark_variants:
                                unit_idx += 1
                                runtime_overrides = dict(phase.runtime_overrides)
                                runtime_overrides.update(r_var)
                                benchmark_overrides = dict(phase.benchmark_overrides)
                                benchmark_overrides.update(b_var)
                                units.append(
                                    RunUnit(
                                        run_id=spec.run_id,
                                        unit_id=f"u{unit_idx:06d}",
                                        phase_id=phase.phase_id,
                                        benchmark=spec.benchmark.get("name", ""),
                                        task_id=task_id,
                                        split=phase.split,
                                        max_seq_length=int(length),
                                        seed=int(seed),
                                        num_samples=int(phase.num_samples),
                                        mode=mode,
                                        runtime_overrides=runtime_overrides,
                                        benchmark_overrides=benchmark_overrides,
                                    )
                                )
    return units


def units_digest(units: Iterable[RunUnit]) -> str:
    payload = [u.to_dict() for u in units]
    return stable_hash({"units": payload})
