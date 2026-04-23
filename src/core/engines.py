"""Shared engine registry and factories for inference and symbolic runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple


class EngineSurface(Enum):
    """Abstract serving or execution surface exposed by an engine."""

    CHAT_OPENAI = "chat_openai"
    DIFFUSION_GENERATE = "diffusion_generate"
    EMBEDDING = "embedding"
    SYMBOLIC_EXACT = "symbolic_exact"


class EngineType(Enum):
    """Canonical engine identifiers used across AR, diffusion, and symbolic paths."""

    VLLM = "vllm"
    SGLANG = "sglang"
    VLLM_OMNI = "vllm_omni"
    OPENAI = "openai"
    CUSTOM_HTTP = "custom_http"
    SYMBOLIC_LOCAL = "symbolic_local"

    @classmethod
    def normalize(cls, value: str | "EngineType") -> "EngineType":
        if isinstance(value, cls):
            return value
        if isinstance(value, Enum):
            value = str(value.value)
        normalized = str(value).strip().lower().replace("-", "_")
        aliases = {
            "vllm": cls.VLLM,
            "sglang": cls.SGLANG,
            "vllm_omni": cls.VLLM_OMNI,
            "vllmomni": cls.VLLM_OMNI,
            "vllm_omni_generate": cls.VLLM_OMNI,
            "openai": cls.OPENAI,
            "custom": cls.CUSTOM_HTTP,
            "custom_http": cls.CUSTOM_HTTP,
            "http": cls.CUSTOM_HTTP,
            "http_generate": cls.CUSTOM_HTTP,
            "symbolic": cls.SYMBOLIC_LOCAL,
            "symbolic_local": cls.SYMBOLIC_LOCAL,
            "local_symbolic": cls.SYMBOLIC_LOCAL,
            "mock": cls.SYMBOLIC_LOCAL,
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported engine '{value}'.")
        return aliases[normalized]

    @property
    def default_port(self) -> Optional[int]:
        return EngineRegistry.resolve(self).default_port

    @property
    def launchable(self) -> bool:
        return EngineRegistry.resolve(self).launchable


LOCAL_CHAT_MANAGED_ENGINES: Tuple[EngineType, ...] = (
    EngineType.VLLM,
    EngineType.SGLANG,
)
_NONE_ENGINE_ALIASES = {"none", "off", "disabled"}
_ENGINE_ROLE_ALIASES = {
    "task": "task",
    "chat": "task",
    "task_model": "task",
    "genrm": "genrm",
    "judge": "genrm",
    "reward": "genrm",
}


@dataclass(frozen=True)
class EngineSpec:
    """Resolved engine metadata derived from static defaults plus repo config."""

    engine: EngineType
    surfaces: Tuple[EngineSurface, ...]
    default_port: Optional[int] = None
    default_host: Optional[str] = "localhost"
    openai_base_path: str = "/v1"
    diffusion_generate_path: str = "/generate"
    launchable: bool = False
    openai_compatible: bool = False
    supports_profiles: bool = False
    supports_diffusion: bool = False
    supports_embeddings: bool = False
    symbolic: bool = False
    config_section: Optional[str] = None
    manager_kind: Optional[str] = None
    launch_script: Optional[str] = None
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def supports_surface(self, surface: EngineSurface | str) -> bool:
        target = (
            surface
            if isinstance(surface, EngineSurface)
            else EngineSurface(str(surface).strip().lower())
        )
        return target in self.surfaces

    def default_base_url(
        self,
        *,
        surface: EngineSurface = EngineSurface.CHAT_OPENAI,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> Optional[str]:
        resolved_host = host or self.default_host
        if resolved_host in {"0.0.0.0", "::"}:
            resolved_host = "localhost"
        resolved_port = port if port is not None else self.default_port
        if self.engine is EngineType.OPENAI:
            return "https://api.openai.com/v1"
        if self.engine is EngineType.SYMBOLIC_LOCAL:
            return None
        if not resolved_host or resolved_port is None:
            return None
        suffix = (
            self.diffusion_generate_path
            if surface is EngineSurface.DIFFUSION_GENERATE
            else self.openai_base_path
        )
        return f"http://{resolved_host}:{resolved_port}{suffix}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine.value,
            "surfaces": tuple(surface.value for surface in self.surfaces),
            "default_port": self.default_port,
            "default_host": self.default_host,
            "openai_base_path": self.openai_base_path,
            "diffusion_generate_path": self.diffusion_generate_path,
            "launchable": self.launchable,
            "openai_compatible": self.openai_compatible,
            "supports_profiles": self.supports_profiles,
            "supports_diffusion": self.supports_diffusion,
            "supports_embeddings": self.supports_embeddings,
            "symbolic": self.symbolic,
            "config_section": self.config_section,
            "manager_kind": self.manager_kind,
            "launch_script": self.launch_script,
            "notes": tuple(self.notes),
        }


class EngineRegistry:
    """Single source of truth for engine metadata and factories."""

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _base_specs(cls) -> dict[EngineType, EngineSpec]:
        scripts_dir = cls._repo_root() / "scripts"
        return {
            EngineType.VLLM: EngineSpec(
                engine=EngineType.VLLM,
                surfaces=(EngineSurface.CHAT_OPENAI, EngineSurface.EMBEDDING),
                default_port=8000,
                default_host="localhost",
                launchable=True,
                openai_compatible=True,
                supports_profiles=True,
                supports_embeddings=True,
                config_section="vllm",
                manager_kind="vllm",
                launch_script=str(scripts_dir / "start_vllm.sh"),
            ),
            EngineType.SGLANG: EngineSpec(
                engine=EngineType.SGLANG,
                surfaces=(EngineSurface.CHAT_OPENAI, EngineSurface.DIFFUSION_GENERATE),
                default_port=30000,
                default_host="localhost",
                launchable=True,
                openai_compatible=True,
                supports_profiles=True,
                supports_diffusion=True,
                config_section="sglang",
                manager_kind="sglang",
                launch_script=str(scripts_dir / "start_sglang.sh"),
            ),
            EngineType.VLLM_OMNI: EngineSpec(
                engine=EngineType.VLLM_OMNI,
                surfaces=(EngineSurface.DIFFUSION_GENERATE,),
                default_port=8004,
                default_host="localhost",
                launchable=False,
                openai_compatible=False,
                supports_profiles=True,
                supports_diffusion=True,
                config_section="vllm_omni",
                notes=(
                    "vLLM-Omni is treated as a diffusion-oriented engine surface in the first pass.",
                ),
            ),
            EngineType.OPENAI: EngineSpec(
                engine=EngineType.OPENAI,
                surfaces=(EngineSurface.CHAT_OPENAI, EngineSurface.EMBEDDING),
                default_port=None,
                default_host=None,
                launchable=False,
                openai_compatible=True,
                supports_embeddings=True,
            ),
            EngineType.CUSTOM_HTTP: EngineSpec(
                engine=EngineType.CUSTOM_HTTP,
                surfaces=(
                    EngineSurface.CHAT_OPENAI,
                    EngineSurface.DIFFUSION_GENERATE,
                    EngineSurface.EMBEDDING,
                ),
                default_port=None,
                default_host=None,
                launchable=False,
                openai_compatible=True,
                supports_diffusion=True,
                supports_embeddings=True,
            ),
            EngineType.SYMBOLIC_LOCAL: EngineSpec(
                engine=EngineType.SYMBOLIC_LOCAL,
                surfaces=(EngineSurface.SYMBOLIC_EXACT,),
                default_port=None,
                default_host=None,
                launchable=False,
                openai_compatible=False,
                symbolic=True,
                notes=("Exact local theorem-family execution with no external server.",),
            ),
        }

    @classmethod
    def available(cls) -> Tuple[EngineType, ...]:
        return tuple(cls._base_specs().keys())

    @classmethod
    def resolve(
        cls,
        engine: str | EngineType,
        *,
        settings: Optional[Mapping[str, Any]] = None,
    ) -> EngineSpec:
        resolved_engine = EngineType.normalize(engine)
        base = cls._base_specs()[resolved_engine]
        if settings is None:
            from src.config.settings import load_settings

            settings = load_settings()

        cfg = (
            dict(settings.get(base.config_section or "", {}) or {})
            if isinstance(settings, Mapping) and base.config_section
            else {}
        )
        default_host = str(cfg.get("host", base.default_host) or base.default_host or "") or None
        default_port = base.default_port
        if cfg.get("port") is not None:
            try:
                default_port = int(cfg["port"])
            except (TypeError, ValueError):
                pass

        return EngineSpec(
            engine=base.engine,
            surfaces=base.surfaces,
            default_port=default_port,
            default_host=default_host,
            openai_base_path=base.openai_base_path,
            diffusion_generate_path=str(
                cfg.get("generate_path", base.diffusion_generate_path)
                or base.diffusion_generate_path
            ),
            launchable=base.launchable,
            openai_compatible=base.openai_compatible,
            supports_profiles=base.supports_profiles,
            supports_diffusion=base.supports_diffusion,
            supports_embeddings=base.supports_embeddings,
            symbolic=base.symbolic,
            config_section=base.config_section,
            manager_kind=base.manager_kind,
            launch_script=base.launch_script,
            notes=base.notes,
        )


def normalize_engine_name(
    value: Any,
    *,
    default: str | EngineType | None = EngineType.VLLM,
) -> Optional[str]:
    """Best-effort normalization for config/CLI engine values."""

    rendered = str(value or "").strip()
    if not rendered:
        return EngineType.normalize(default).value if default is not None else None
    try:
        return EngineType.normalize(rendered).value
    except ValueError:
        return EngineType.normalize(default).value if default is not None else None


def normalize_fallback_engine_name(
    value: Any,
    *,
    default: str | EngineType | None = EngineType.VLLM,
) -> str:
    """Normalize fallback engine values, preserving disabled aliases as `none`."""

    rendered = str(value or "").strip().lower().replace("-", "_")
    if not rendered:
        if default is None:
            return "none"
        return EngineType.normalize(default).value
    if rendered in _NONE_ENGINE_ALIASES:
        return "none"
    return normalize_engine_name(rendered, default=default) or "none"


def normalize_engine_role(role: str = "task") -> str:
    """Normalize logical engine-service roles used for default port selection."""

    normalized = str(role or "task").strip().lower().replace("-", "_")
    if normalized not in _ENGINE_ROLE_ALIASES:
        raise ValueError(f"Unsupported engine role '{role}'.")
    return _ENGINE_ROLE_ALIASES[normalized]


def default_engine_port(
    engine: str | EngineType,
    *,
    role: str = "task",
    settings: Optional[Mapping[str, Any]] = None,
) -> Optional[int]:
    """Resolve the default local port for an engine/service role."""

    resolved_role = normalize_engine_role(role)
    if settings is None:
        from src.config.settings import load_settings

        settings = load_settings()
    spec = EngineRegistry.resolve(engine, settings=settings)
    if resolved_role == "task":
        return spec.default_port

    settings_dict = dict(settings or {})
    if spec.engine is EngineType.SGLANG:
        sglang_cfg = settings_dict.get("sglang", {})
        if isinstance(sglang_cfg, Mapping) and sglang_cfg.get("genrm_port") is not None:
            try:
                return int(sglang_cfg["genrm_port"])
            except (TypeError, ValueError):
                pass
        return (spec.default_port + 1) if spec.default_port is not None else None

    if spec.engine is EngineType.VLLM:
        vllm_cfg = settings_dict.get("vllm", {})
        if isinstance(vllm_cfg, Mapping) and vllm_cfg.get("genrm_port") is not None:
            try:
                return int(vllm_cfg["genrm_port"])
            except (TypeError, ValueError):
                pass
        return (spec.default_port + 1) if spec.default_port is not None else 8001

    return spec.default_port


def resolve_engine_base_url(
    engine: str | EngineType,
    *,
    surface: EngineSurface = EngineSurface.CHAT_OPENAI,
    role: str = "task",
    settings: Optional[Mapping[str, Any]] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> Optional[str]:
    """Resolve the default base URL for an engine/surface pair."""

    spec = EngineRegistry.resolve(engine, settings=settings)
    resolved_port = port
    if resolved_port is None and surface in {
        EngineSurface.CHAT_OPENAI,
        EngineSurface.EMBEDDING,
    }:
        resolved_port = default_engine_port(spec.engine, role=role, settings=settings)
    return spec.default_base_url(surface=surface, host=host, port=resolved_port)


def resolve_engine_for_usage(
    engine: str | EngineType,
    *,
    surface: EngineSurface,
    usage: str,
    settings: Optional[Mapping[str, Any]] = None,
    require_managed: bool = False,
    allowed_engines: Optional[Sequence[str | EngineType]] = None,
) -> EngineSpec:
    """Resolve and validate an engine for a concrete runtime surface/path."""

    spec = EngineRegistry.resolve(engine, settings=settings)
    if not spec.supports_surface(surface):
        raise ValueError(
            f"Engine '{spec.engine.value}' is registered but does not expose the "
            f"{surface.value} surface required by {usage}."
        )
    if allowed_engines is not None:
        allowed = tuple(EngineType.normalize(candidate) for candidate in allowed_engines)
        if spec.engine not in allowed:
            supported = ", ".join(candidate.value for candidate in allowed)
            raise ValueError(
                f"Engine '{spec.engine.value}' is registered for {surface.value} but is not "
                f"supported in {usage}. Supported engines here: {supported}."
            )
    if require_managed and not spec.launchable:
        raise ValueError(
            f"Engine '{spec.engine.value}' is registered for {surface.value} but does not "
            f"provide a managed local server for {usage}."
        )
    return spec


def build_server_manager(
    engine: str | EngineType,
    *,
    profile: str,
    port: Optional[int] = None,
    host: Optional[str] = None,
    cuda_devices: Optional[str] = None,
    tensor_parallel: Optional[int] = None,
    venv_path: Optional[str] = None,
    startup_timeout: float = 300.0,
    health_check_interval: float = 2.0,
    extra_args: Optional[Sequence[str]] = None,
) -> Any:
    """Construct a managed local server from the shared engine registry."""

    spec = resolve_engine_for_usage(
        engine,
        surface=EngineSurface.CHAT_OPENAI,
        usage="managed local server startup",
        require_managed=True,
    )
    if spec.manager_kind not in {"vllm", "sglang"}:
        raise ValueError(
            f"Engine '{spec.engine.value}' is registered for chat_openai but does not "
            f"provide a managed local server manager in this pass."
        )

    from src.benchmark.throughput import SGLangServerManager, VLLMServerManager

    manager_kwargs = {
        "profile": profile,
        "port": int(port if port is not None else spec.default_port or 0),
        "host": host or spec.default_host or "0.0.0.0",
        "startup_timeout": float(startup_timeout),
        "health_check_interval": float(health_check_interval),
        "cuda_devices": cuda_devices,
        "tensor_parallel": tensor_parallel,
        "extra_args": list(extra_args or []),
    }
    if venv_path:
        manager_kwargs["venv_path"] = str(venv_path)

    if spec.manager_kind == "sglang":
        return SGLangServerManager(**manager_kwargs)
    return VLLMServerManager(**manager_kwargs)


__all__ = [
    "EngineRegistry",
    "EngineSpec",
    "EngineSurface",
    "EngineType",
    "LOCAL_CHAT_MANAGED_ENGINES",
    "build_server_manager",
    "default_engine_port",
    "normalize_engine_name",
    "normalize_engine_role",
    "normalize_fallback_engine_name",
    "resolve_engine_base_url",
    "resolve_engine_for_usage",
]
