from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TrainingSelectionMetadata:
    mode: str
    split: str
    metric_name: str
    metric_value: float
    best_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selection_mode": str(self.mode),
            "selection_split": str(self.split),
            "selection_metric_name": str(self.metric_name),
            "selection_metric_value": float(self.metric_value),
            "best_epoch": int(self.best_epoch),
        }


def clone_module_state(module: Any) -> Dict[str, Any]:
    return {
        str(name): value.detach().cpu().clone()
        for name, value in dict(module.state_dict()).items()
    }


def restore_module_state(module: Any, state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    module.load_state_dict(state)


def improved_metric(candidate: float, incumbent: float, *, atol: float = 1e-12) -> bool:
    if not math.isfinite(float(candidate)):
        return False
    if not math.isfinite(float(incumbent)):
        return True
    return float(candidate) < (float(incumbent) - float(atol))
