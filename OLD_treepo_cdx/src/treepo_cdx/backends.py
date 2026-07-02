from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from treepo_cdx._json import jsonable


@dataclass(frozen=True)
class StateShapeContract:
    state_family: str
    shape: tuple[int, ...] = ()
    dtype: str = ""
    variable_length: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.state_family):
            raise ValueError("state_family is required")
        shape = tuple(int(item) for item in self.shape)
        if any(item < 0 for item in shape):
            raise ValueError("state shape dimensions must be non-negative")
        object.__setattr__(self, "state_family", str(self.state_family))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", str(self.dtype or ""))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass(frozen=True)
class SupervisionSpec:
    name: str
    requires_oracle: bool = False
    supports_ipw: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("supervision name is required")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


class BackendRuntime(Protocol):
    def state_shape_contract(self) -> StateShapeContract | Mapping[str, Any]:
        ...

    def supported_supervisions(self) -> Sequence[SupervisionSpec | Mapping[str, Any] | str]:
        ...


def backend_capabilities(backend: BackendRuntime) -> dict[str, Any]:
    shape = backend.state_shape_contract()
    if not isinstance(shape, StateShapeContract):
        shape = StateShapeContract(**dict(shape))
    supervisions = tuple(_coerce_supervision(item) for item in backend.supported_supervisions())
    return {
        "state_shape_contract": shape.to_dict(),
        "supported_supervisions": [item.to_dict() for item in supervisions],
    }


def _coerce_supervision(item: SupervisionSpec | Mapping[str, Any] | str) -> SupervisionSpec:
    if isinstance(item, SupervisionSpec):
        return item
    if isinstance(item, str):
        return SupervisionSpec(name=item)
    return SupervisionSpec(**dict(item))


__all__ = [
    "BackendRuntime",
    "StateShapeContract",
    "SupervisionSpec",
    "backend_capabilities",
]
