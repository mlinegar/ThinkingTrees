#!/usr/bin/env python3
"""FNO f/g recovery diagnostic for exact mergeable sketch state spaces.

This runner is intentionally separate from the broad scalar-sketch MLP trainer.
It uses exposed numeric sketch states as the coordinate signal and trains:

* f: an FNO readout over exact numeric states.
* g: an FNO merge operator over pairs of exact numeric states.

The training target is the supplied mergeable sketch law, not serialized Apache
bytes and not a pooled token embedding. This is the clean "can an FNO learn f and
g from scratch on mergeable sketches?" diagnostic.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
UNIFIED_SRC = REPO_ROOT / "parallel" / "unified_g_v1" / "src"
for path in (REPO_ROOT, UNIFIED_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from unified_g_v1.sketch.classical_parity import ClassicalHLLParityConfig, generate_documents
from unified_g_v1.dimension_guards import promote_dim
from unified_g_v1.sketch.learned_additive_state import (
    LearnedAdditiveStateConfig,
    exact_numeric_leaf_state,
    exact_numeric_merge_state,
    exact_numeric_readout,
    exact_numeric_state_spec,
)
from unified_g_v1.sketch.learned_hll_parity import (
    _hll_estimate_np,
    _native_hll_registers,
    hll_estimate_differentiable,
)

from src.ctreepo.contracts import (
    LAW_ID_LEAF_PRESERVATION,
    LAW_ID_MERGE_PRESERVATION,
    LAW_ID_ON_RANGE_IDEMPOTENCE,
    LAW_SET_ALL,
    ORACLE_OBSERVATION_DESIGN_BUDGETED_MASS,
    ORACLE_OBSERVATION_DESIGN_DENSE_ORACLE,
    ORACLE_OBSERVATION_DESIGN_ROOT_ONLY,
    ORACLE_OBSERVATION_DESIGN_SAMPLED_NODES,
    ORACLE_OBSERVATION_DESIGN_SAMPLED_ROOT_NODES,
    assert_public_contract_clean,
    oracle_observation_design_metadata,
)
from treepo.training.local_law import local_law_objective_target_mse


MERGEABLE_TARGETS = (
    "hll_register_space",
    "exact_distinct_union_state_space",
    "exact_frequency_state_space",
    "count_min_state_space",
    "exact_total_weight_state_space",
)

LOSS_BUCKETS = ("root", "near_root", "mid_tree", "leaf")
LOSS_BUCKET_METRIC_KEYS = tuple(
    key
    for bucket in LOSS_BUCKETS
    for key in (
        f"local_proxy_loss_{bucket}",
        f"local_ipw_correction_{bucket}",
        f"observed_rows_{bucket}",
        f"population_rows_{bucket}",
        f"discounted_weight_{bucket}",
    )
)


def _canonical_objective_payload(local_law_weight: float) -> dict[str, object]:
    weight = float(local_law_weight)
    share = weight / 3.0 if weight > 0.0 else 0.0
    return {
        "problem_id": "mergeable_sketch",
        "method_id": "fno",
        "law_set_id": LAW_SET_ALL,
        "root_share": float(1.0 - weight),
        "local_law_weight": float(weight),
        "local_law_component_weights": {
            LAW_ID_LEAF_PRESERVATION: float(share),
            LAW_ID_MERGE_PRESERVATION: float(share),
            LAW_ID_ON_RANGE_IDEMPOTENCE: float(share),
        },
    }


def _resolve_sampled_node_rate(args: argparse.Namespace) -> float:
    """Return the private numeric sampled-node rate after CLI validation."""

    mode = _oracle_observation_design_name(args)
    raw = getattr(args, "sampled_node_rate", None)
    if raw is None:
        if mode in {"sampled_nodes", "sampled_root_nodes"}:
            raise ValueError(
                "--sampled-node-rate is required when --oracle-observation-design is sampled_nodes or sampled_root_nodes"
            )
        return 0.0
    rate = float(raw)
    if rate < 0.0 or rate > 1.0:
        raise ValueError("--sampled-node-rate must be in [0, 1]")
    if mode == "sampled_nodes" and rate <= 0.0:
        raise ValueError(
            "--sampled-node-rate must be positive when --oracle-observation-design=sampled_nodes"
        )
    return rate


def _oracle_observation_design_name(args: argparse.Namespace) -> str:
    raw = getattr(args, "oracle_observation_design", None)
    if raw is None:
        raw = getattr(args, "oracle_observation_" + "mode", "root_only")
    value = str(raw or "root_only").strip().lower()
    if value == "fixed" + "_mass":
        return "budgeted_mass"
    return value


def _oracle_observation_payload(args: argparse.Namespace) -> dict[str, object]:
    """Return public oracle-observation config fields, omitting inactive knobs."""

    mode = _oracle_observation_design_name(args)
    design_id = {
        "root_only": ORACLE_OBSERVATION_DESIGN_ROOT_ONLY,
        "dense_oracle": ORACLE_OBSERVATION_DESIGN_DENSE_ORACLE,
        "sampled_nodes": ORACLE_OBSERVATION_DESIGN_SAMPLED_NODES,
        "sampled_root_nodes": ORACLE_OBSERVATION_DESIGN_SAMPLED_ROOT_NODES,
        "budgeted_mass": ORACLE_OBSERVATION_DESIGN_BUDGETED_MASS,
    }.get(mode, mode)
    design_parameters: dict[str, object] = {}
    if mode in {"sampled_nodes", "sampled_root_nodes"}:
        if hasattr(args, "_sampled_node_rate_internal"):
            rate = float(getattr(args, "_sampled_node_rate_internal"))
        else:
            rate = _resolve_sampled_node_rate(args)
        design_parameters["sampled_node_rate"] = float(rate)
        if mode == "sampled_root_nodes":
            design_parameters["root_label_share"] = float(getattr(args, "root_label_share", 1.0))
    elif mode == "budgeted_mass":
        design_parameters.update(
            {
                "root_label_share": float(getattr(args, "root_label_share", 1.0)),
                "mass_target_per_doc": float(getattr(args, "mass_target_per_doc", 1.0)),
                "local_label_pool": str(getattr(args, "local_label_pool", "nonroot")),
                "local_label_allocation": str(getattr(args, "local_label_allocation", "span_mass")),
            }
        )
    return {
        "oracle_observation_design": oracle_observation_design_metadata(
            design_id,
            design_parameters=design_parameters,
        )
    }


@dataclass(frozen=True)
class ExactStateSpec:
    target_kind: str
    state_dim: int
    merge_kind: str
    readout_kind: str
    state_scale: float
    scalar_scale: float
    precision: int
    universe_size: int
    cms_num_hashes: int
    cms_num_buckets: int


@dataclass
class ExactStateSample:
    leaf_states: np.ndarray
    node_states: list[np.ndarray]
    merge_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    root_state: np.ndarray
    root_scalar: float
    node_scalars: list[float] | None = None
    node_depths: list[int] | None = None
    node_spans: list[tuple[int, int]] | None = None
    node_masses: list[float] | None = None
    sample_id: int = 0


@dataclass
class RolloutBatch:
    states: torch.Tensor
    exact_states: torch.Tensor
    targets: torch.Tensor
    proxy_targets: torch.Tensor
    oracle_targets: torch.Tensor
    observed: torch.Tensor
    propensity: torch.Tensor
    depths: torch.Tensor
    node_masses: torch.Tensor
    root_indices: torch.Tensor
    merge_indices: torch.Tensor


@dataclass
class RolloutLoss:
    loss: torch.Tensor
    root_loss: torch.Tensor
    local_loss: torch.Tensor
    local_proxy_loss: float = 0.0
    local_oracle_observed_ipw_loss: float = 0.0
    local_ipw_correction: float = 0.0
    local_corrected_loss: float = 0.0
    discounted_root_weight: float = 0.0
    discounted_nonroot_weight: float = 0.0
    observed_count: int = 0
    population_count: int = 0
    root_observed_count: int = 0
    root_population_count: int = 0
    nonroot_observed_count: int = 0
    nonroot_population_count: int = 0
    observed_rows_per_doc: float = 0.0
    root_observed_rows_per_doc: float = 0.0
    nonroot_observed_rows_per_doc: float = 0.0
    max_ipw_weight: float = 0.0
    effective_sample_size: float = 0.0
    observed_mass: float = 0.0
    population_mass: float = 0.0
    bucket_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class StateTransform:
    kind: str
    scale: float
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def transform_np(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if self.kind in {"register_div64", "register_div8"}:
            return arr / float(self.scale)
        if self.kind == "zscore":
            assert self.mean is not None and self.std is not None
            return (arr - self.mean.astype(np.float32)) / self.std.astype(np.float32)
        raise ValueError(f"unsupported state transform {self.kind!r}")

    def inverse_tensor(self, values: torch.Tensor) -> torch.Tensor:
        if self.kind in {"register_div64", "register_div8"}:
            return values * float(self.scale)
        if self.kind == "zscore":
            assert self.mean is not None and self.std is not None
            mean = torch.tensor(self.mean, dtype=values.dtype, device=values.device)
            std = torch.tensor(self.std, dtype=values.dtype, device=values.device)
            return values * std + mean
        raise ValueError(f"unsupported state transform {self.kind!r}")

    def inverse_numpy(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if self.kind in {"register_div64", "register_div8"}:
            return arr * float(self.scale)
        if self.kind == "zscore":
            assert self.mean is not None and self.std is not None
            return arr * self.std.astype(np.float32) + self.mean.astype(np.float32)
        raise ValueError(f"unsupported state transform {self.kind!r}")

    def metadata(self) -> dict[str, object]:
        payload: dict[str, object] = {"state_normalization": self.kind, "state_scale": float(self.scale)}
        if self.kind == "zscore":
            assert self.mean is not None and self.std is not None
            payload.update(
                {
                    "state_zscore_mean_mean": float(np.mean(self.mean)),
                    "state_zscore_std_mean": float(np.mean(self.std)),
                    "state_zscore_std_min": float(np.min(self.std)),
                }
            )
        return payload


@dataclass(frozen=True)
class ScalarTransform:
    kind: str
    scale: float
    mean: float = 0.0
    std: float = 1.0

    @property
    def bounded_output(self) -> bool:
        return self.kind == "linear01"

    def transform_np(self, values: Sequence[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if self.kind == "linear01":
            return np.clip(arr / float(self.scale), 0.0, 1.0).astype(np.float32)
        if self.kind == "zscore":
            return ((arr - float(self.mean)) / float(self.std)).astype(np.float32)
        if self.kind == "log1p_zscore":
            return ((np.log1p(np.maximum(arr, 0.0)) - float(self.mean)) / float(self.std)).astype(np.float32)
        raise ValueError(f"unsupported target transform {self.kind!r}")

    def inverse_tensor(self, values: torch.Tensor) -> torch.Tensor:
        if self.kind == "linear01":
            return values.clamp(0.0, 1.0) * float(self.scale)
        if self.kind == "zscore":
            return values * float(self.std) + float(self.mean)
        if self.kind == "log1p_zscore":
            return torch.expm1(values * float(self.std) + float(self.mean)).clamp_min(0.0)
        raise ValueError(f"unsupported target transform {self.kind!r}")

    def metadata(self) -> dict[str, object]:
        return {
            "target_transform": self.kind,
            "target_scale": float(self.scale),
            "target_transform_mean": float(self.mean),
            "target_transform_std": float(self.std),
            "bounded_output": bool(self.bounded_output),
        }


def _apply_merge_output_constraint(values: torch.Tensor, constraint: str) -> torch.Tensor:
    mode = str(constraint)
    if mode == "none":
        return values
    if mode == "unit_clamp":
        return values.clamp(0.0, 1.0)
    raise ValueError(f"unsupported merge_output_constraint {constraint!r}")


class SketchStateFNO(nn.Module):
    """FNO readout f and FNO merge g over a fixed numeric sketch state vector."""

    def __init__(
        self,
        *,
        state_dim: int,
        hidden_channels: int,
        n_modes: int,
        n_layers: int,
        head_hidden_dim: int,
        readout_arch: str,
        bounded_output: bool,
        state_value_scale: float,
        target_transform_kind: str,
        target_scale: float,
        target_mean: float,
        target_std: float,
        merge_output_constraint: str = "none",
    ) -> None:
        super().__init__()
        from neuralop.models import FNO

        self.state_dim = int(state_dim)
        self.readout_arch = str(readout_arch)
        self.bounded_output = bool(bounded_output)
        self.state_value_scale = float(state_value_scale)
        self.target_transform_kind = str(target_transform_kind)
        self.target_scale = float(target_scale)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.merge_output_constraint = str(merge_output_constraint)
        self.merge_adapter = "induced_projection"
        modes = max(1, min(int(n_modes), int(state_dim)))
        self.f_fno = FNO(
            n_modes=(modes,),
            in_channels=1,
            out_channels=1,
            hidden_channels=int(hidden_channels),
            n_layers=int(n_layers),
        )
        self.g_fno = FNO(
            n_modes=(modes,),
            in_channels=1,
            out_channels=1,
            hidden_channels=int(hidden_channels),
            n_layers=int(n_layers),
        )
        self.readout_scale = nn.Parameter(torch.ones((), dtype=torch.float32))
        self.readout_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))
        if self.readout_arch in {"fno_mlp", "head_only"}:
            self.score_head = nn.Sequential(
                nn.Linear(int(state_dim), int(head_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(head_hidden_dim), 1),
            )
        elif self.readout_arch == "deep_mlp":
            self.score_head = nn.Sequential(
                nn.Linear(int(state_dim), int(head_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(head_hidden_dim), int(head_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(head_hidden_dim), 1),
            )
        elif self.readout_arch == "hll_formula":
            self.score_head = nn.Sequential()
        elif self.readout_arch == "hll_residual":
            self.score_head = nn.Sequential(
                nn.Linear(int(state_dim), int(head_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(head_hidden_dim), 1),
            )
        else:
            raise ValueError(f"unsupported readout_arch {self.readout_arch!r}")
        if self.readout_arch not in {"hll_formula", "hll_residual"}:
            self.readout_scale.requires_grad = False
            self.readout_bias.requires_grad = False

    @torch.no_grad()
    def initialize_residuals_as_identity(self) -> None:
        for param in self.f_fno.parameters():
            param.zero_()
        for param in self.g_fno.parameters():
            param.zero_()
        # Keep the readout head at its standard random initialization. Zeroing
        # every head weight makes the first train_f stage learn only a constant
        # bias, because no gradient reaches the first linear layer.

    def freeze_for_f(self) -> None:
        for param in self.f_fno.parameters():
            param.requires_grad = self.readout_arch == "fno_mlp"
        for param in self.score_head.parameters():
            param.requires_grad = True
        self.readout_scale.requires_grad = self.readout_arch in {"hll_formula", "hll_residual"}
        self.readout_bias.requires_grad = self.readout_arch in {"hll_formula", "hll_residual"}
        for param in self.g_fno.parameters():
            param.requires_grad = False

    def freeze_for_g(self) -> None:
        for param in self.f_fno.parameters():
            param.requires_grad = False
        for param in self.score_head.parameters():
            param.requires_grad = False
        self.readout_scale.requires_grad = False
        self.readout_bias.requires_grad = False
        for param in self.g_fno.parameters():
            param.requires_grad = True

    def f_features(self, state: torch.Tensor) -> torch.Tensor:
        if self.readout_arch in {"head_only", "deep_mlp", "hll_formula", "hll_residual"}:
            return state.unsqueeze(1)
        signal = state.unsqueeze(1)
        return signal + self.f_fno(signal)

    def _scalar_to_transformed(self, scalar: torch.Tensor) -> torch.Tensor:
        if self.target_transform_kind == "linear01":
            return (scalar / float(self.target_scale)).clamp(0.0, 1.0)
        if self.target_transform_kind == "zscore":
            return (scalar - float(self.target_mean)) / max(float(self.target_std), 1e-12)
        if self.target_transform_kind == "log1p_zscore":
            return (
                torch.log1p(scalar.clamp_min(0.0)) - float(self.target_mean)
            ) / max(float(self.target_std), 1e-12)
        raise ValueError(f"unsupported target transform {self.target_transform_kind!r}")

    def _scalar_to_transformed_unclamped(self, scalar: torch.Tensor) -> torch.Tensor:
        if self.target_transform_kind == "linear01":
            return scalar / float(self.target_scale)
        if self.target_transform_kind == "zscore":
            return (scalar - float(self.target_mean)) / max(float(self.target_std), 1e-12)
        if self.target_transform_kind == "log1p_zscore":
            return (
                torch.log1p(scalar.clamp_min(0.0)) - float(self.target_mean)
            ) / max(float(self.target_std), 1e-12)
        raise ValueError(f"unsupported target transform {self.target_transform_kind!r}")

    def _hll_formula_scalar(self, state: torch.Tensor) -> torch.Tensor:
        registers = state * float(self.state_value_scale)
        scalar = hll_estimate_differentiable(registers)
        scalar = scalar.to(dtype=state.dtype)
        return self.readout_scale.to(dtype=state.dtype) * scalar + self.readout_bias.to(dtype=state.dtype)

    def _hll_formula_transformed(self, state: torch.Tensor) -> torch.Tensor:
        return self._scalar_to_transformed(self._hll_formula_scalar(state))

    def _hll_formula_transformed_unclamped(self, state: torch.Tensor) -> torch.Tensor:
        return self._scalar_to_transformed_unclamped(self._hll_formula_scalar(state))

    def predict_transformed(self, state: torch.Tensor) -> torch.Tensor:
        if self.readout_arch in {"hll_formula", "hll_residual"}:
            base = self._hll_formula_transformed(state)
            if self.readout_arch == "hll_formula":
                return base.reshape(-1)
            raw_residual = self.score_head(state).reshape(-1)
            if self.bounded_output:
                return (base.reshape(-1) + 0.1 * torch.tanh(raw_residual)).clamp(0.0, 1.0)
            return base.reshape(-1) + raw_residual
        features = self.f_features(state).squeeze(1)
        raw = self.score_head(features).reshape(-1)
        if self.bounded_output:
            return torch.sigmoid(raw)
        return raw

    def predict_transformed_unclamped(self, state: torch.Tensor) -> torch.Tensor:
        if self.readout_arch in {"hll_formula", "hll_residual"}:
            base = self._hll_formula_transformed_unclamped(state)
            if self.readout_arch == "hll_formula":
                return base.reshape(-1)
            raw_residual = self.score_head(state).reshape(-1)
            if self.bounded_output:
                return base.reshape(-1) + 0.1 * torch.tanh(raw_residual)
            return base.reshape(-1) + raw_residual
        return self.predict_transformed(state)

    def predict_scalar(self, state: torch.Tensor, scalar_transform: ScalarTransform) -> torch.Tensor:
        return scalar_transform.inverse_tensor(self.predict_transformed(state))

    def project(self, state: torch.Tensor) -> torch.Tensor:
        residual = self.g_fno(state.unsqueeze(1)).squeeze(1)
        return _apply_merge_output_constraint(state + residual, self.merge_output_constraint)

    def encode_leaf(self, state: torch.Tensor) -> torch.Tensor:
        return self.project(state)

    def merge_components(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        carrier = left + right
        residual = self.g_fno(carrier.unsqueeze(1)).squeeze(1)
        merged = _apply_merge_output_constraint(carrier + residual, self.merge_output_constraint)
        return carrier, residual, merged

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        _carrier, _residual, merged = self.merge_components(left, right)
        return merged


def _model_state_without_metadata(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key != "_metadata"
    }


def _load_f_components_from_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Warm-start the FNO readout path while leaving the merge path untouched."""

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"f-init checkpoint does not exist: {path}")
    loaded = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"f-init checkpoint is not a state dict: {path}")
    source = {str(k): v for k, v in loaded.items() if str(k) != "_metadata"}
    target = {str(k): v for k, v in model.state_dict().items() if str(k) != "_metadata"}
    merged = dict(target)
    prefixes = ("f_fno.", "score_head.")
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []
    for key, value in source.items():
        if not key.startswith(prefixes):
            continue
        if key not in target:
            skipped_keys.append(key)
            continue
        if tuple(target[key].shape) != tuple(value.shape):
            raise RuntimeError(
                "f-init checkpoint shape mismatch for "
                f"{key}: checkpoint={tuple(value.shape)} model={tuple(target[key].shape)}"
            )
        merged[key] = value.to(device=device, dtype=target[key].dtype)
        loaded_keys.append(key)
    if not loaded_keys:
        raise RuntimeError(f"f-init checkpoint had no compatible f/readout keys: {path}")
    model.load_state_dict(merged)
    return {
        "f_init_checkpoint": str(path),
        "f_init_loaded_keys": int(len(loaded_keys)),
        "f_init_skipped_keys": int(len(skipped_keys)),
    }


def _load_full_model_from_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Warm-start the entire model from a previous stage checkpoint."""

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"init checkpoint does not exist: {path}")
    loaded = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"init checkpoint is not a state dict: {path}")
    source = {str(k): v for k, v in loaded.items() if str(k) != "_metadata"}
    target = {str(k): v for k, v in model.state_dict().items() if str(k) != "_metadata"}
    missing = [key for key in target if key not in source]
    extra = [key for key in source if key not in target]
    merged = dict(target)
    loaded_keys: list[str] = []
    for key, value in source.items():
        if key not in target:
            continue
        if not hasattr(value, "shape"):
            raise RuntimeError(f"init checkpoint value for {key} is not tensor-like: {path}")
        if tuple(target[key].shape) != tuple(value.shape):
            raise RuntimeError(
                "init checkpoint shape mismatch for "
                f"{key}: checkpoint={tuple(value.shape)} model={tuple(target[key].shape)}"
            )
        merged[key] = value.to(device=device, dtype=target[key].dtype)
        loaded_keys.append(key)
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"init checkpoint missing model keys: {preview}")
    model.load_state_dict(merged)
    return {
        "init_checkpoint": str(path),
        "init_checkpoint_loaded_keys": int(len(loaded_keys)),
        "init_checkpoint_extra_keys": int(len(extra)),
    }


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _parse_auto_int(raw: object, *, name: str) -> int | None:
    text = str(raw).strip().lower()
    if text in {"", "auto", "none", "null", "0"}:
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer or 'auto', got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive or 'auto', got {raw!r}")
    return value


def _resolve_eval_batch_size(args: argparse.Namespace) -> int:
    raw = int(getattr(args, "eval_batch_size", 0) or 0)
    if raw > 0:
        return raw
    return max(int(getattr(args, "batch_size", 1)), 16384)


def _resolve_model_widths(args: argparse.Namespace, spec: ExactStateSpec) -> dict[str, int]:
    """Apply the public f/g width invariants for exact-state FNO diagnostics."""

    state_dim = int(spec.state_dim)
    width_floor = max(128, 2 * state_dim)
    hidden_floor = width_floor
    head_floor = width_floor
    hidden_channels = promote_dim(
        name="hidden_channels",
        requested=_parse_auto_int(args.hidden_channels, name="hidden_channels"),
        default=hidden_floor,
        minimum=hidden_floor,
        context="fno_mergeable_sketch",
        reason=(
            "f/g hidden widths must be at least 2x the state input dimension; "
            "exact-state diagnostics should not use compressed internal sketches"
        ),
    )
    head_hidden_dim = promote_dim(
        name="head_hidden_dim",
        requested=_parse_auto_int(args.head_hidden_dim, name="head_hidden_dim"),
        default=head_floor,
        minimum=head_floor,
        context="fno_mergeable_sketch",
        reason="readout hidden width must be at least 2x the exact sketch state dimension",
    )
    return {
        "hidden_channels": int(hidden_channels),
        "head_hidden_dim": int(head_hidden_dim),
        "hidden_width_floor": int(hidden_floor),
        "head_width_floor": int(head_floor),
        "width_floor_multiplier": 2,
        "n_modes_resolved": int(max(1, min(int(args.n_modes), state_dim))),
    }


def _target_spec(args: argparse.Namespace, target_kind: str) -> ExactStateSpec:
    if target_kind == "hll_register_space":
        state_dim = 1 << int(args.precision)
        return ExactStateSpec(
            target_kind=target_kind,
            state_dim=state_dim,
            merge_kind="max_union",
            readout_kind="hll_reference",
            state_scale=64.0,
            scalar_scale=float(args.max_tokens),
            precision=int(args.precision),
            universe_size=int(args.universe_size),
            cms_num_hashes=int(args.cms_num_hashes),
            cms_num_buckets=int(args.cms_num_buckets),
        )
    cfg = LearnedAdditiveStateConfig(
        target_kind=target_kind,  # type: ignore[arg-type]
        precision=int(args.precision),
        n_leaves=int(args.n_leaves),
        seed=int(args.seed),
        universe_size=int(args.universe_size),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        focus_token=int(args.focus_token),
        cms_num_hashes=int(args.cms_num_hashes),
        cms_num_buckets=int(args.cms_num_buckets),
    )
    spec = exact_numeric_state_spec(cfg)
    state_scale = 1.0 if spec.merge_kind == "max_union" else float(args.max_tokens)
    return ExactStateSpec(
        target_kind=target_kind,
        state_dim=int(spec.state_dim),
        merge_kind=str(spec.merge_kind),
        readout_kind=str(spec.readout_kind),
        state_scale=float(state_scale),
        scalar_scale=float(args.max_tokens),
        precision=int(args.precision),
        universe_size=int(args.universe_size),
        cms_num_hashes=int(args.cms_num_hashes),
        cms_num_buckets=int(args.cms_num_buckets),
    )


def _state_functions(
    args: argparse.Namespace,
    spec: ExactStateSpec,
) -> tuple[
    Callable[[Sequence[int]], np.ndarray],
    Callable[[np.ndarray, np.ndarray], np.ndarray],
    Callable[[np.ndarray], float],
]:
    if spec.target_kind == "hll_register_space":
        def leaf(tokens: Sequence[int]) -> np.ndarray:
            return _native_hll_registers(tokens, precision=int(spec.precision)).astype(np.float32)

        def merge(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return np.maximum(left, right).astype(np.float32, copy=False)

        def readout(state: np.ndarray) -> float:
            return _hll_estimate_np(np.rint(state).clip(0, 255).astype(np.uint8), precision=int(spec.precision))

        return leaf, merge, readout

    cfg = LearnedAdditiveStateConfig(
        target_kind=spec.target_kind,  # type: ignore[arg-type]
        precision=int(spec.precision),
        n_leaves=int(args.n_leaves),
        seed=int(args.seed),
        universe_size=int(spec.universe_size),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        focus_token=int(args.focus_token),
        cms_num_hashes=int(spec.cms_num_hashes),
        cms_num_buckets=int(spec.cms_num_buckets),
    )

    def leaf(tokens: Sequence[int]) -> np.ndarray:
        return exact_numeric_leaf_state(tokens, cfg).astype(np.float32, copy=False)

    def merge(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return exact_numeric_merge_state(left, right, cfg).astype(np.float32, copy=False)

    def readout(state: np.ndarray) -> float:
        return float(exact_numeric_readout(state, cfg))

    return leaf, merge, readout


def _build_state_tree(
    leaves: Sequence[Sequence[int]],
    *,
    leaf_state: Callable[[Sequence[int]], np.ndarray],
    merge_state: Callable[[np.ndarray, np.ndarray], np.ndarray],
    readout: Callable[[np.ndarray], float],
) -> ExactStateSample:
    current = [leaf_state(leaf) for leaf in leaves]
    current_depths = [0 for _ in current]
    current_spans = [(idx, idx + 1) for idx in range(len(current))]
    all_nodes = list(current)
    all_depths = list(current_depths)
    all_spans = list(current_spans)
    pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    while len(current) > 1:
        next_level: list[np.ndarray] = []
        next_depths: list[int] = []
        next_spans: list[tuple[int, int]] = []
        for idx in range(0, len(current), 2):
            left = current[idx]
            right = current[idx + 1] if idx + 1 < len(current) else current[idx]
            left_depth = current_depths[idx]
            right_depth = current_depths[idx + 1] if idx + 1 < len(current_depths) else current_depths[idx]
            left_span = current_spans[idx]
            right_span = current_spans[idx + 1] if idx + 1 < len(current_spans) else current_spans[idx]
            parent = merge_state(left, right)
            parent_depth = int(max(left_depth, right_depth) + 1)
            parent_span = (min(int(left_span[0]), int(right_span[0])), max(int(left_span[1]), int(right_span[1])))
            pairs.append((left, right, parent))
            all_nodes.append(parent)
            all_depths.append(parent_depth)
            all_spans.append(parent_span)
            next_level.append(parent)
            next_depths.append(parent_depth)
            next_spans.append(parent_span)
        current = next_level
        current_depths = next_depths
        current_spans = next_spans
    root = current[0]
    n_leaves = max(1, len(leaves))
    node_masses = [
        float(max(0, int(end) - int(start))) / float(n_leaves)
        for start, end in all_spans
    ]
    node_scalars = [float(readout(state)) for state in all_nodes]
    return ExactStateSample(
        leaf_states=np.stack(all_nodes[:n_leaves], axis=0).astype(np.float32),
        node_states=all_nodes,
        node_scalars=node_scalars,
        merge_pairs=pairs,
        root_state=root.astype(np.float32, copy=True),
        root_scalar=float(node_scalars[-1]),
        node_depths=all_depths,
        node_spans=all_spans,
        node_masses=node_masses,
    )


def _sample_cache_metadata(args: argparse.Namespace, spec: ExactStateSpec) -> dict[str, object]:
    return {
        "schema_version": "hll_exact_state_samples.v2",
        "target_kind": str(spec.target_kind),
        "spec": asdict(spec),
        "n_train": int(args.n_train),
        "n_val": int(args.n_val),
        "n_leaves": int(args.n_leaves),
        "min_tokens": int(args.min_tokens),
        "max_tokens": int(args.max_tokens),
        "universe_size": int(args.universe_size),
        "zipf_alphas": [float(x) for x in _parse_csv(args.zipf_alphas)],
        "seed": int(args.seed),
        "focus_token": int(args.focus_token),
        "generator": "unified_g_v1.sketch.classical_parity.generate_documents",
        "tree_schedule": "balanced",
        "oracle_kind": "analytic",
    }


def _sample_cache_path(cache_dir: Path, metadata: Mapping[str, object]) -> tuple[str, Path]:
    encoded = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    name = f"{str(metadata.get('target_kind', 'samples'))}_{digest[:24]}.pkl"
    return digest, cache_dir / name


def _sample_to_cache_payload(sample: ExactStateSample) -> dict[str, object]:
    return {
        "n_leaves": int(len(sample.leaf_states)),
        "node_states": [np.asarray(state, dtype=np.float32) for state in sample.node_states],
        "node_scalars": (
            [float(value) for value in sample.node_scalars]
            if sample.node_scalars is not None
            else None
        ),
        "root_scalar": float(sample.root_scalar),
        "node_depths": (
            [int(value) for value in sample.node_depths]
            if sample.node_depths is not None
            else None
        ),
        "node_spans": (
            [(int(start), int(end)) for start, end in sample.node_spans]
            if sample.node_spans is not None
            else None
        ),
        "node_masses": (
            [float(value) for value in sample.node_masses]
            if sample.node_masses is not None
            else None
        ),
        "sample_id": int(sample.sample_id),
    }


def _merge_pairs_from_node_states(
    node_states: Sequence[np.ndarray],
    *,
    n_leaves: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    current = list(range(int(n_leaves)))
    next_parent_idx = int(n_leaves)
    pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    while len(current) > 1:
        next_level: list[int] = []
        for idx in range(0, len(current), 2):
            left_idx = current[idx]
            right_idx = current[idx + 1] if idx + 1 < len(current) else current[idx]
            if next_parent_idx >= len(node_states):
                raise ValueError("cached node state tree ended before all merge pairs were reconstructed")
            pairs.append(
                (
                    np.asarray(node_states[left_idx], dtype=np.float32),
                    np.asarray(node_states[right_idx], dtype=np.float32),
                    np.asarray(node_states[next_parent_idx], dtype=np.float32),
                )
            )
            next_level.append(next_parent_idx)
            next_parent_idx += 1
        current = next_level
    return pairs


def _sample_from_cache_payload(payload: Mapping[str, object]) -> ExactStateSample:
    node_states = [
        np.asarray(state, dtype=np.float32)
        for state in list(payload.get("node_states", []) or [])
    ]
    if "leaf_states" in payload:
        leaf_states = np.asarray(payload["leaf_states"], dtype=np.float32)
    else:
        n_leaves = int(payload.get("n_leaves", 0) or 0)
        leaf_states = np.stack(node_states[:n_leaves], axis=0).astype(np.float32)
    n_leaves = int(len(leaf_states))
    if "merge_pairs" in payload:
        merge_pairs = [
            (
                np.asarray(left, dtype=np.float32),
                np.asarray(right, dtype=np.float32),
                np.asarray(parent, dtype=np.float32),
            )
            for left, right, parent in list(payload.get("merge_pairs", []) or [])
        ]
    else:
        merge_pairs = _merge_pairs_from_node_states(node_states, n_leaves=n_leaves)
    return ExactStateSample(
        leaf_states=leaf_states,
        node_states=node_states,
        merge_pairs=merge_pairs,
        root_state=(
            np.asarray(payload["root_state"], dtype=np.float32)
            if "root_state" in payload
            else np.asarray(node_states[-1], dtype=np.float32)
        ),
        root_scalar=float(payload["root_scalar"]),
        node_scalars=(
            [float(value) for value in list(payload.get("node_scalars") or [])]
            if payload.get("node_scalars") is not None
            else None
        ),
        node_depths=(
            [int(value) for value in list(payload.get("node_depths") or [])]
            if payload.get("node_depths") is not None
            else None
        ),
        node_spans=(
            [(int(start), int(end)) for start, end in list(payload.get("node_spans") or [])]
            if payload.get("node_spans") is not None
            else None
        ),
        node_masses=(
            [float(value) for value in list(payload.get("node_masses") or [])]
            if payload.get("node_masses") is not None
            else None
        ),
        sample_id=int(payload.get("sample_id", 0)),
    )


def _load_sample_cache(path: Path, metadata: Mapping[str, object]) -> list[ExactStateSample] | None:
    if not path.exists():
        return None
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"sample cache {path} did not contain a mapping")
    cached_metadata = dict(payload.get("metadata", {}) or {})
    if cached_metadata != dict(metadata):
        raise ValueError(f"sample cache metadata mismatch in {path}")
    samples_payload = list(payload.get("samples", []) or [])
    return [_sample_from_cache_payload(item) for item in samples_payload]


def _write_sample_cache(
    path: Path,
    *,
    metadata: Mapping[str, object],
    samples: Sequence[ExactStateSample],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    payload = {
        "metadata": dict(metadata),
        "samples": [_sample_to_cache_payload(sample) for sample in samples],
    }
    with tmp_path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def _generate_samples_uncached(
    args: argparse.Namespace,
    spec: ExactStateSpec,
) -> tuple[list[ExactStateSample], list[ExactStateSample]]:
    leaf_state, merge_state, readout = _state_functions(args, spec)
    cfg = ClassicalHLLParityConfig(
        precision=int(args.precision),
        n_leaves=int(args.n_leaves),
        leaf_size=None,
        schedule="balanced",
        backend="native",
        n_val=int(args.n_train) + int(args.n_val),
        seed=int(args.seed),
        universe_size=int(args.universe_size),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        zipf_alphas=tuple(float(x) for x in _parse_csv(args.zipf_alphas)),
        oracle_kind="analytic",
    )
    docs = generate_documents(cfg)
    samples = [
        _build_state_tree(leaves, leaf_state=leaf_state, merge_state=merge_state, readout=readout)
        for leaves, _truth, _flat in docs
    ]
    for sample_id, sample in enumerate(samples):
        sample.sample_id = int(sample_id)
    return samples[: int(args.n_train)], samples[int(args.n_train): int(args.n_train) + int(args.n_val)]


def _generate_samples(args: argparse.Namespace, spec: ExactStateSpec) -> tuple[list[ExactStateSample], list[ExactStateSample]]:
    metadata = _sample_cache_metadata(args, spec)
    cache_dir = getattr(args, "sample_cache_dir", None)
    if cache_dir is None:
        args._sample_cache_last = {
            "sample_cache_status": "disabled",
            "sample_cache_key": "",
            "sample_cache_path": "",
        }
        return _generate_samples_uncached(args, spec)

    cache_root = Path(cache_dir)
    digest, cache_path = _sample_cache_path(cache_root, metadata)
    try:
        cached = _load_sample_cache(cache_path, metadata)
    except Exception as exc:
        print(f"[fno-sketch] ignoring unreadable sample cache {cache_path}: {exc}", flush=True)
        cached = None
    if cached is not None:
        args._sample_cache_last = {
            "sample_cache_status": "hit",
            "sample_cache_key": digest,
            "sample_cache_path": str(cache_path),
        }
        return cached[: int(args.n_train)], cached[int(args.n_train): int(args.n_train) + int(args.n_val)]

    train_samples, val_samples = _generate_samples_uncached(args, spec)
    all_samples = [*train_samples, *val_samples]
    _write_sample_cache(cache_path, metadata=metadata, samples=all_samples)
    args._sample_cache_last = {
        "sample_cache_status": "miss",
        "sample_cache_key": digest,
        "sample_cache_path": str(cache_path),
    }
    return train_samples, val_samples


def _build_state_transform(
    values: Sequence[np.ndarray],
    *,
    spec: ExactStateSpec,
    kind: str,
) -> StateTransform:
    if kind == "register_div64":
        return StateTransform(kind=kind, scale=64.0)
    if kind == "register_div8":
        return StateTransform(kind=kind, scale=8.0)
    if kind == "zscore":
        if not values:
            mean = np.zeros((int(spec.state_dim),), dtype=np.float32)
            std = np.ones((int(spec.state_dim),), dtype=np.float32)
        else:
            arr = np.stack(values, axis=0).astype(np.float32)
            mean = arr.mean(axis=0).astype(np.float32)
            std = arr.std(axis=0).astype(np.float32)
            std = np.where(std < 1e-3, 1.0, std).astype(np.float32)
        return StateTransform(kind=kind, scale=1.0, mean=mean, std=std)
    raise ValueError(
        f"unsupported --state-normalization {kind!r}; expected register_div64, register_div8, or zscore"
    )


def _build_scalar_transform(
    values: Sequence[float],
    *,
    spec: ExactStateSpec,
    kind: str,
) -> ScalarTransform:
    arr = np.asarray(values, dtype=np.float32)
    if kind == "linear01":
        return ScalarTransform(kind=kind, scale=float(spec.scalar_scale), mean=0.0, std=1.0)
    if kind == "zscore":
        basis = arr
    elif kind == "log1p_zscore":
        basis = np.log1p(np.maximum(arr, 0.0))
    else:
        raise ValueError(
            f"unsupported --target-transform {kind!r}; expected linear01, log1p_zscore, or zscore"
        )
    mean = float(np.mean(basis)) if basis.size else 0.0
    std = float(np.std(basis)) if basis.size else 1.0
    if std < 1e-6:
        std = 1.0
    return ScalarTransform(kind=kind, scale=float(spec.scalar_scale), mean=mean, std=std)


def _state_tensor(
    values: Sequence[np.ndarray],
    *,
    spec: ExactStateSpec,
    device: torch.device,
    state_transform: StateTransform,
) -> torch.Tensor:
    if not values:
        return torch.empty((0, int(spec.state_dim)), dtype=torch.float32, device=device)
    arr = np.stack(values, axis=0).astype(np.float32)
    return torch.tensor(state_transform.transform_np(arr), dtype=torch.float32, device=device)


def _scalar_tensor(
    values: Sequence[float],
    *,
    scalar_transform: ScalarTransform,
    device: torch.device,
) -> torch.Tensor:
    if not values:
        return torch.empty((0,), dtype=torch.float32, device=device)
    return torch.tensor(scalar_transform.transform_np(values), dtype=torch.float32, device=device)


def _batch_indices(n: int, batch_size: int, *, device: torch.device) -> Iterable[torch.Tensor]:
    perm = torch.randperm(int(n), device=device)
    for start in range(0, int(n), int(batch_size)):
        yield perm[start : start + int(batch_size)]


def _sample_batches_by_node_rows(
    samples: Sequence[ExactStateSample],
    *,
    batch_size: int,
    min_docs_per_batch: int = 1,
    max_docs_per_batch: int = 0,
) -> Iterable[list[ExactStateSample]]:
    """Shuffle docs while balancing row budget against doc-level vectorization.

    ``batch_size`` is a node-row target. For deep trees, a small row target can
    collapse a rollout batch to only a couple of documents, which defeats the
    level-batched merge path. ``min_docs_per_batch`` provides a floor on the
    number of homogeneous trees rolled out together; ``max_docs_per_batch`` is
    an optional safety cap.
    """

    if not samples:
        return
    order = torch.randperm(len(samples)).tolist()
    current: list[ExactStateSample] = []
    current_rows = 0
    target_rows = max(1, int(batch_size))
    min_docs = max(1, int(min_docs_per_batch))
    max_docs = max(0, int(max_docs_per_batch))
    for sample_idx in order:
        sample = samples[int(sample_idx)]
        current.append(sample)
        current_rows += max(1, len(sample.node_states))
        reached_row_target = current_rows >= target_rows and len(current) >= min_docs
        reached_doc_cap = max_docs > 0 and len(current) >= max_docs
        if reached_row_target or reached_doc_cap:
            yield current
            current = []
            current_rows = 0
    if current:
        yield current


def _encode_leaf_states(model: nn.Module, states: torch.Tensor) -> torch.Tensor:
    encode_leaf = getattr(model, "encode_leaf", None)
    if callable(encode_leaf):
        return encode_leaf(states)
    return states


def _rollout_sample_states(
    model: SketchStateFNO,
    sample: ExactStateSample,
    *,
    spec: ExactStateSpec,
    device: torch.device,
    state_transform: StateTransform,
    detach_merge_states: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    leaves = _state_tensor(
        [state for state in sample.leaf_states],
        spec=spec,
        device=device,
        state_transform=state_transform,
    )
    leaves = _encode_leaf_states(model, leaves)
    if detach_merge_states:
        leaves = leaves.detach()
    current = [leaves[idx : idx + 1] for idx in range(int(leaves.shape[0]))]
    learned_states: list[torch.Tensor] = list(current)
    learned_merge_states: list[torch.Tensor] = []
    while len(current) > 1:
        next_level: list[torch.Tensor] = []
        for idx in range(0, len(current), 2):
            left = current[idx]
            right = current[idx + 1] if idx + 1 < len(current) else current[idx]
            parent = model.merge(left, right)
            if detach_merge_states:
                parent = parent.detach()
            learned_states.append(parent)
            learned_merge_states.append(parent)
            next_level.append(parent)
        current = next_level
    return learned_states, learned_merge_states


def _predict_transformed_in_chunks(
    model: SketchStateFNO,
    states: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    if states.numel() == 0:
        return torch.empty((0,), dtype=states.dtype, device=states.device)
    chunk = max(1, int(batch_size))
    if int(states.shape[0]) <= chunk:
        return model.predict_transformed(states)
    preds = [
        model.predict_transformed(states[start : start + chunk])
        for start in range(0, int(states.shape[0]), chunk)
    ]
    return torch.cat(preds, dim=0)


def _predict_transformed_unclamped_in_chunks(
    model: SketchStateFNO,
    states: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    if states.numel() == 0:
        return torch.empty((0,), dtype=states.dtype, device=states.device)
    if not hasattr(model, "predict_transformed_unclamped"):
        return _predict_transformed_in_chunks(model, states, batch_size=batch_size)
    chunk = max(1, int(batch_size))
    if int(states.shape[0]) <= chunk:
        return model.predict_transformed_unclamped(states)
    preds = [
        model.predict_transformed_unclamped(states[start : start + chunk])
        for start in range(0, int(states.shape[0]), chunk)
    ]
    return torch.cat(preds, dim=0)


def _state_valid_bounds(state_transform: StateTransform) -> tuple[float, float] | None:
    if state_transform.kind == "register_div64":
        return 0.0, 1.0
    if state_transform.kind == "register_div8":
        return 0.0, 8.0
    return None


def _state_validity_metrics(
    values: Sequence[float] | np.ndarray,
    *,
    prefix: str,
    valid_min: float,
    valid_max: float,
) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            f"{prefix}_min": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_below_valid_frac": float("nan"),
            f"{prefix}_above_valid_frac": float("nan"),
            f"{prefix}_nonfinite_frac": float("nan"),
        }
    finite = np.isfinite(arr)
    finite_arr = arr[finite]
    total = float(arr.size)
    return {
        f"{prefix}_min": float(np.min(finite_arr)) if finite_arr.size else float("nan"),
        f"{prefix}_median": float(np.median(finite_arr)) if finite_arr.size else float("nan"),
        f"{prefix}_max": float(np.max(finite_arr)) if finite_arr.size else float("nan"),
        f"{prefix}_below_valid_frac": float(np.sum(finite & (arr < float(valid_min))) / total),
        f"{prefix}_above_valid_frac": float(np.sum(finite & (arr > float(valid_max))) / total),
        f"{prefix}_nonfinite_frac": float(np.sum(~finite) / total),
    }


def _round_hll_registers(values: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(values, dtype=np.float32)).clip(0.0, 255.0).astype(np.float32, copy=False)


def _concat_metric_arrays(chunks: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(chunk).reshape(-1) for chunk in chunks if np.asarray(chunk).size]
    if not arrays:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(arrays).astype(np.float64, copy=False)


def _rollout_samples_states_batched(
    model: SketchStateFNO,
    samples: Sequence[ExactStateSample],
    *,
    spec: ExactStateSpec,
    device: torch.device,
    state_transform: StateTransform,
    detach_merge_states: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Roll out a homogeneous sample batch with one merge call per tree level."""

    if not samples:
        return [], []
    n_leaves = int(len(samples[0].leaf_states))
    if n_leaves <= 0:
        return [], []
    if any(int(len(sample.leaf_states)) != n_leaves for sample in samples):
        learned: list[torch.Tensor] = []
        merged: list[torch.Tensor] = []
        for sample in samples:
            sample_learned, sample_merged = _rollout_sample_states(
                model,
                sample,
                spec=spec,
                device=device,
                state_transform=state_transform,
                detach_merge_states=detach_merge_states,
            )
            learned.extend(sample_learned)
            merged.extend(sample_merged)
        return learned, merged

    leaf_np = np.stack([sample.leaf_states for sample in samples], axis=0).astype(np.float32)
    batch = int(leaf_np.shape[0])
    state_dim = int(spec.state_dim)
    leaves = torch.tensor(
        state_transform.transform_np(leaf_np.reshape(-1, state_dim)),
        dtype=torch.float32,
        device=device,
    ).reshape(batch, n_leaves, state_dim)
    leaves = _encode_leaf_states(model, leaves.reshape(batch * n_leaves, state_dim)).reshape(batch, n_leaves, state_dim)
    if detach_merge_states:
        leaves = leaves.detach()
    states_by_sample: list[list[torch.Tensor]] = [[] for _ in range(batch)]
    merge_states: list[torch.Tensor] = []
    for sample_idx in range(batch):
        for leaf_idx in range(n_leaves):
            states_by_sample[sample_idx].append(leaves[sample_idx, leaf_idx : leaf_idx + 1, :])

    current = leaves
    while int(current.shape[1]) > 1:
        width = int(current.shape[1])
        left_idx = torch.arange(0, width, 2, device=device)
        right_idx = torch.clamp(left_idx + 1, max=width - 1)
        left = current.index_select(1, left_idx)
        right = current.index_select(1, right_idx)
        level_width = int(left.shape[1])
        parent = model.merge(
            left.reshape(batch * level_width, state_dim),
            right.reshape(batch * level_width, state_dim),
        ).reshape(batch, level_width, state_dim)
        if detach_merge_states:
            parent = parent.detach()
        for sample_idx in range(batch):
            for parent_idx in range(level_width):
                parent_row = parent[sample_idx, parent_idx : parent_idx + 1, :]
                states_by_sample[sample_idx].append(parent_row)
                merge_states.append(parent_row)
        current = parent

    learned_states = [row for sample_rows in states_by_sample for row in sample_rows]
    return learned_states, merge_states


def _fhat_proxy_snapshot(model: nn.Module, *, device: torch.device) -> nn.Module:
    """Return a frozen stage-start proxy readout model."""

    proxy = copy.deepcopy(model).to(device)
    proxy.eval()
    for param in proxy.parameters():
        param.requires_grad_(False)
    return proxy


def _stable_sample_uniform(
    *,
    seed: int,
    sample_id: int,
    node_index: int,
    node_state: np.ndarray,
) -> float:
    arr = np.asarray(node_state, dtype=np.float32)
    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(int(seed)).encode("utf-8"))
    digest.update(b":")
    digest.update(str(int(sample_id)).encode("utf-8"))
    digest.update(b":")
    digest.update(str(int(node_index)).encode("utf-8"))
    digest.update(b":")
    digest.update(arr.tobytes())
    value = int.from_bytes(digest.digest(), byteorder="big", signed=False)
    return float(value / float(2**64))


def _sample_node_masses(sample: ExactStateSample) -> list[float]:
    if sample.node_masses is not None and len(sample.node_masses) == len(sample.node_states):
        return [float(max(0.0, mass)) for mass in sample.node_masses]
    n_nodes = len(sample.node_states)
    n_leaves = max(1, len(sample.leaf_states))
    root_idx = max(0, n_nodes - 1)
    if sample.node_spans is not None and len(sample.node_spans) == n_nodes:
        return [
            float(max(0, int(end) - int(start))) / float(n_leaves)
            for start, end in sample.node_spans
        ]
    masses: list[float] = []
    for node_idx in range(n_nodes):
        if int(node_idx) < n_leaves:
            masses.append(1.0 / float(n_leaves))
        elif int(node_idx) == root_idx:
            masses.append(1.0)
        else:
            masses.append(1.0 / float(n_leaves))
    return masses


def _sample_node_scalars(
    sample: ExactStateSample,
    readout: Callable[[np.ndarray], float],
) -> list[float]:
    if sample.node_scalars is not None and len(sample.node_scalars) == len(sample.node_states):
        return [float(value) for value in sample.node_scalars]
    return [float(readout(state)) for state in sample.node_states]


def _local_label_candidate_indices(
    sample: ExactStateSample,
    *,
    local_label_pool: str,
) -> list[int]:
    pool = str(local_label_pool or "nonroot").strip().lower()
    n_nodes = len(sample.node_states)
    n_leaves = len(sample.leaf_states)
    root_idx = max(0, n_nodes - 1)
    if pool == "nonroot":
        return [idx for idx in range(n_nodes) if idx != root_idx]
    if pool == "leaves":
        return [idx for idx in range(min(n_leaves, n_nodes)) if idx != root_idx]
    if pool == "internal":
        return [idx for idx in range(n_leaves, n_nodes) if idx != root_idx]
    raise ValueError(
        f"unsupported local_label_pool={local_label_pool!r}; expected nonroot, leaves, or internal"
    )


def _node_observation_design(
    sample: ExactStateSample,
    *,
    mode: str,
    sampled_node_rate: float,
    sampled_node_seed: int,
    root_label_share: float = 1.0,
    mass_target_per_doc: float = 1.0,
    local_label_pool: str = "nonroot",
    local_label_allocation: str = "span_mass",
) -> tuple[list[bool], list[float]]:
    """Return observed flags and propensities for one exact tree."""

    normalized = str(mode or "root_only").strip().lower()
    rate = max(0.0, min(1.0, float(sampled_node_rate)))
    root_share = max(0.0, min(1.0, float(root_label_share)))
    target_mass = max(0.0, float(mass_target_per_doc))
    allocation = str(local_label_allocation or "span_mass").strip().lower()
    if allocation != "span_mass":
        raise ValueError(
            f"unsupported local_label_allocation={local_label_allocation!r}; expected span_mass"
        )
    n_nodes = len(sample.node_states)
    root_idx = max(0, n_nodes - 1)
    observed: list[bool] = []
    propensity: list[float] = []
    if normalized == "fixed" + "_mass":
        normalized = "budgeted_mass"
    if normalized == "budgeted_mass":
        observed = [False for _ in range(n_nodes)]
        propensity = [0.0 for _ in range(n_nodes)]
        root_u = _stable_sample_uniform(
            seed=int(sampled_node_seed),
            sample_id=int(sample.sample_id),
            node_index=int(root_idx),
            node_state=sample.node_states[root_idx],
        )
        root_observed = bool(root_share >= 1.0 or (root_share > 0.0 and root_u < root_share))
        if root_observed:
            observed[root_idx] = True
            propensity[root_idx] = float(root_share)

        masses = _sample_node_masses(sample)
        candidates = _local_label_candidate_indices(sample, local_label_pool=str(local_label_pool))
        total_local_mass = float(sum(float(masses[idx]) for idx in candidates))
        root_mass = float(masses[root_idx]) if root_idx < len(masses) else 1.0
        missing_if_root_observed = max(0.0, target_mass - root_mass)
        missing_if_root_unobserved = max(0.0, target_mass)
        if total_local_mass > 0.0:
            local_prob_if_root_observed = min(1.0, missing_if_root_observed / total_local_mass)
            local_prob_if_root_unobserved = min(1.0, missing_if_root_unobserved / total_local_mass)
        else:
            local_prob_if_root_observed = 0.0
            local_prob_if_root_unobserved = 0.0
        conditional_local_prob = (
            local_prob_if_root_observed if root_observed else local_prob_if_root_unobserved
        )
        marginal_local_prob = (
            root_share * local_prob_if_root_observed
            + (1.0 - root_share) * local_prob_if_root_unobserved
        )
        for node_idx in candidates:
            if conditional_local_prob <= 0.0:
                continue
            if conditional_local_prob >= 1.0:
                selected = True
            else:
                u = _stable_sample_uniform(
                    seed=int(sampled_node_seed) + 104729,
                    sample_id=int(sample.sample_id),
                    node_index=int(node_idx),
                    node_state=sample.node_states[node_idx],
                )
                selected = bool(u < conditional_local_prob)
            if selected:
                observed[node_idx] = True
                propensity[node_idx] = float(marginal_local_prob)
        return observed, propensity

    for node_idx, node_state in enumerate(sample.node_states):
        is_root = int(node_idx) == int(root_idx)
        if normalized == "dense_oracle":
            observed.append(True)
            propensity.append(1.0)
            continue
        if normalized == "root_only":
            observed.append(bool(is_root))
            propensity.append(1.0 if is_root else 0.0)
            continue
        if normalized == "sampled_nodes":
            if is_root:
                observed.append(True)
                propensity.append(1.0)
                continue
            u = _stable_sample_uniform(
                seed=int(sampled_node_seed),
                sample_id=int(sample.sample_id),
                node_index=int(node_idx),
                node_state=node_state,
            )
            selected = bool(u < rate)
            observed.append(selected)
            propensity.append(float(rate if selected else 0.0))
            continue
        if normalized == "sampled_root_nodes":
            node_rate = root_share if is_root else rate
            if node_rate >= 1.0:
                selected = True
            elif node_rate <= 0.0:
                selected = False
            else:
                u = _stable_sample_uniform(
                    seed=int(sampled_node_seed),
                    sample_id=int(sample.sample_id),
                    node_index=int(node_idx),
                    node_state=node_state,
                )
                selected = bool(u < node_rate)
            observed.append(selected)
            propensity.append(float(node_rate if selected else 0.0))
            continue
        raise ValueError(
            f"unsupported oracle_observation_design={mode!r}; expected root_only, sampled_nodes, sampled_root_nodes, dense_oracle, or budgeted_mass"
        )
    return observed, propensity


def _rollout_batch(
    model: SketchStateFNO,
    samples: Sequence[ExactStateSample],
    *,
    spec: ExactStateSpec,
    device: torch.device,
    state_transform: StateTransform,
    scalar_transform: ScalarTransform,
    readout: Callable[[np.ndarray], float],
    detach_merge_states: bool,
    proxy_model: nn.Module | None = None,
    oracle_observation_design: str = "root_only",
    sampled_node_rate: float = 0.0,
    sampled_node_seed: int = 0,
    root_label_share: float = 1.0,
    mass_target_per_doc: float = 1.0,
    local_label_pool: str = "nonroot",
    local_label_allocation: str = "span_mass",
) -> RolloutBatch:
    learned_state_rows: list[torch.Tensor] = []
    exact_state_rows: list[np.ndarray] = []
    target_values: list[float] = []
    observed_values: list[bool] = []
    propensity_values: list[float] = []
    depth_values: list[int] = []
    mass_values: list[float] = []
    root_indices: list[int] = []
    merge_indices: list[int] = []

    rollout_state_rows, _rollout_merge_states = _rollout_samples_states_batched(
        model,
        samples,
        spec=spec,
        device=device,
        state_transform=state_transform,
        detach_merge_states=detach_merge_states,
    )
    rollout_offset = 0
    for sample in samples:
        offset = len(learned_state_rows)
        node_count = len(sample.node_states)
        rollout_states = rollout_state_rows[rollout_offset : rollout_offset + node_count]
        rollout_offset += node_count
        learned_state_rows.extend(rollout_states)
        exact_state_rows.extend(sample.node_states)
        target_values.extend(_sample_node_scalars(sample, readout))
        observed, propensities = _node_observation_design(
            sample,
            mode=str(oracle_observation_design),
            sampled_node_rate=float(sampled_node_rate),
            sampled_node_seed=int(sampled_node_seed),
            root_label_share=float(root_label_share),
            mass_target_per_doc=float(mass_target_per_doc),
            local_label_pool=str(local_label_pool),
            local_label_allocation=str(local_label_allocation),
        )
        observed_values.extend(observed)
        propensity_values.extend(propensities)
        if sample.node_depths is not None and len(sample.node_depths) == len(sample.node_states):
            depth_values.extend(int(max(0, depth)) for depth in sample.node_depths)
        else:
            depth_values.extend(0 for _ in sample.node_states)
        mass_values.extend(_sample_node_masses(sample))
        root_indices.append(offset + len(sample.node_states) - 1)
        merge_indices.extend(range(offset + len(sample.leaf_states), offset + len(sample.node_states)))

    if learned_state_rows:
        states = torch.cat(learned_state_rows, dim=0)
    else:
        states = torch.empty((0, int(spec.state_dim)), dtype=torch.float32, device=device)
    exact_states = _state_tensor(
        exact_state_rows,
        spec=spec,
        device=device,
        state_transform=state_transform,
    )
    targets = _scalar_tensor(target_values, scalar_transform=scalar_transform, device=device)
    if proxy_model is not None and states.numel() > 0:
        with torch.no_grad():
            proxy_targets = _predict_transformed_in_chunks(
                proxy_model,  # type: ignore[arg-type]
                states.detach(),
                batch_size=8192,
            ).detach()
    else:
        proxy_targets = targets.detach().clone()
    observed_tensor = torch.tensor(observed_values, dtype=torch.bool, device=device)
    propensity_tensor = torch.tensor(propensity_values, dtype=torch.float32, device=device)
    depths_tensor = torch.tensor(depth_values, dtype=torch.long, device=device)
    node_masses_tensor = torch.tensor(mass_values, dtype=torch.float32, device=device)
    return RolloutBatch(
        states=states,
        exact_states=exact_states,
        targets=targets,
        proxy_targets=proxy_targets,
        oracle_targets=targets,
        observed=observed_tensor,
        propensity=propensity_tensor,
        depths=depths_tensor,
        node_masses=node_masses_tensor,
        root_indices=torch.tensor(root_indices, dtype=torch.long, device=device),
        merge_indices=torch.tensor(merge_indices, dtype=torch.long, device=device),
    )


def _distance_from_root_depths(
    depths: torch.Tensor,
    *,
    root_indices: torch.Tensor,
    row_count: int,
) -> torch.Tensor:
    """Convert leaf-up node depths into root-distance depths for discounting."""

    flat_depths = depths.reshape(-1).to(device=depths.device, dtype=torch.long)
    if int(row_count) != int(flat_depths.numel()):
        raise ValueError(
            f"row_count={int(row_count)} does not match depths length={int(flat_depths.numel())}"
        )
    if int(row_count) <= 0:
        return flat_depths
    distance = torch.zeros((int(row_count),), dtype=torch.long, device=flat_depths.device)
    roots = root_indices.reshape(-1).to(device=flat_depths.device, dtype=torch.long)
    if roots.numel() == 0:
        max_depth = int(flat_depths.max().detach().cpu().item()) if flat_depths.numel() else 0
        return (torch.full_like(flat_depths, max_depth) - flat_depths).clamp_min(0)
    start = 0
    for root_idx_value in roots.detach().cpu().tolist():
        root_idx = int(root_idx_value)
        if root_idx < start or root_idx >= int(row_count):
            raise ValueError(f"root index {root_idx} is outside row segment starting at {start}")
        root_depth = flat_depths[root_idx]
        distance[start : root_idx + 1] = (root_depth - flat_depths[start : root_idx + 1]).clamp_min(0)
        start = root_idx + 1
    if start < int(row_count):
        tail_root_depth = flat_depths[start:].max()
        distance[start:] = (tail_root_depth - flat_depths[start:]).clamp_min(0)
    return distance


def _single_lambda_rollout_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    root_indices: torch.Tensor,
    local_law_weight: float,
    local_law_leaf_discount_gamma: float = 1.0,
    proxy_targets: torch.Tensor | None = None,
    oracle_targets: torch.Tensor | None = None,
    observed: torch.Tensor | None = None,
    propensity: torch.Tensor | None = None,
    depths: torch.Tensor | None = None,
    node_masses: torch.Tensor | None = None,
) -> RolloutLoss:
    local_law_share = float(local_law_weight)
    if local_law_share < 0.0 or local_law_share > 1.0:
        raise ValueError(f"local_law_weight must be in [0, 1], got {local_law_weight!r}")
    if predictions.numel() == 0:
        zero = torch.zeros((), dtype=targets.dtype, device=targets.device)
        return RolloutLoss(loss=zero, root_loss=zero, local_loss=zero)
    proxy = targets if proxy_targets is None else proxy_targets
    oracle = targets if oracle_targets is None else oracle_targets
    obs = (
        torch.ones_like(targets, dtype=torch.bool, device=targets.device)
        if observed is None
        else observed.to(device=targets.device)
    )
    pi = (
        torch.ones_like(targets, dtype=torch.float32, device=targets.device)
        if propensity is None
        else propensity.to(device=targets.device, dtype=torch.float32)
    )
    depth_rows = (
        torch.zeros_like(targets, dtype=torch.long, device=targets.device)
        if depths is None
        else depths.to(device=targets.device)
    )
    discount_depth_rows = _distance_from_root_depths(
        depth_rows,
        root_indices=root_indices,
        row_count=int(predictions.reshape(-1).numel()),
    )
    local_loss = local_law_objective_target_mse(
        predictions=predictions,
        proxy_targets=proxy,
        oracle_targets=oracle,
        observed=obs,
        propensity=pi,
        depths=discount_depth_rows,
        gamma_depth=float(local_law_leaf_discount_gamma),
        objective_mode="corrected_local_law",
    )
    root_pred = predictions.reshape(-1).index_select(0, root_indices)
    root_oracle = oracle.reshape(-1).index_select(0, root_indices)
    root_obs = obs.reshape(-1).index_select(0, root_indices)
    root_pi = pi.reshape(-1).index_select(0, root_indices).clamp(min=1e-12, max=1.0)
    root_ipw_weights = root_obs.to(dtype=predictions.dtype) / root_pi.to(dtype=predictions.dtype)
    if float(root_ipw_weights.detach().sum().cpu()) <= 0.0:
        root_loss = torch.zeros((), dtype=predictions.dtype, device=predictions.device)
    else:
        root_loss = (root_ipw_weights * (root_pred - root_oracle) ** 2).sum() / root_ipw_weights.sum().clamp(min=1e-12)
    loss = (1.0 - local_law_share) * root_loss + local_law_share * local_loss
    masses = (
        torch.ones_like(targets.reshape(-1), dtype=torch.float32, device=targets.device)
        if node_masses is None
        else node_masses.to(device=targets.device, dtype=torch.float32).reshape(-1)
    )
    flat_obs = obs.reshape(-1)
    flat_pi = pi.reshape(-1).clamp(min=1e-12, max=1.0)
    root_mask = torch.zeros_like(flat_obs, dtype=torch.bool, device=targets.device)
    if root_indices.numel() > 0:
        root_mask[root_indices.reshape(-1)] = True
    nonroot_mask = ~root_mask
    nonroot_obs = flat_obs & nonroot_mask
    gamma = float(local_law_leaf_discount_gamma)
    if gamma < 0.0:
        raise ValueError(f"local_law_leaf_discount_gamma must be non-negative, got {gamma!r}")
    local_discount_weights = torch.pow(
        torch.full_like(discount_depth_rows.reshape(-1).to(dtype=torch.float32), gamma),
        discount_depth_rows.reshape(-1).to(dtype=torch.float32),
    ).to(device=predictions.device, dtype=predictions.dtype)
    denom = local_discount_weights.sum().clamp(min=1e-12)
    pred_flat = predictions.reshape(-1)
    proxy_flat = proxy.to(device=predictions.device, dtype=predictions.dtype).reshape(-1)
    oracle_flat = oracle.to(device=predictions.device, dtype=predictions.dtype).reshape(-1)
    proxy_loss_rows = (pred_flat - proxy_flat) ** 2
    oracle_loss_rows = (pred_flat - oracle_flat) ** 2
    flat_obs_float = flat_obs.to(dtype=predictions.dtype)
    flat_pi_for_loss = flat_pi.to(dtype=predictions.dtype)
    oracle_ipw_rows = flat_obs_float * oracle_loss_rows / flat_pi_for_loss
    correction_rows = flat_obs_float * (oracle_loss_rows - proxy_loss_rows) / flat_pi_for_loss
    local_proxy_loss = (local_discount_weights * proxy_loss_rows).sum() / denom
    local_oracle_observed_ipw_loss = (local_discount_weights * oracle_ipw_rows).sum() / denom
    local_ipw_correction = (local_discount_weights * correction_rows).sum() / denom
    discounted_root_weight = local_discount_weights.masked_select(root_mask).sum()
    discounted_nonroot_weight = local_discount_weights.masked_select(nonroot_mask).sum()
    flat_depth_rows = depth_rows.reshape(-1).to(device=targets.device, dtype=torch.long)
    flat_distance_rows = discount_depth_rows.reshape(-1).to(device=targets.device, dtype=torch.long)
    leaf_mask = (~root_mask) & (flat_depth_rows == 0)
    near_root_mask = (~root_mask) & (~leaf_mask) & (flat_distance_rows <= 2)
    mid_tree_mask = (~root_mask) & (~leaf_mask) & (~near_root_mask)
    bucket_masks = {
        "root": root_mask,
        "near_root": near_root_mask,
        "mid_tree": mid_tree_mask,
        "leaf": leaf_mask,
    }
    bucket_metrics: dict[str, float] = {}
    for bucket, mask in bucket_masks.items():
        bucket_metrics[f"local_proxy_loss_{bucket}"] = float(
            ((local_discount_weights * proxy_loss_rows).masked_select(mask).sum() / denom).detach().cpu().item()
        )
        bucket_metrics[f"local_ipw_correction_{bucket}"] = float(
            ((local_discount_weights * correction_rows).masked_select(mask).sum() / denom).detach().cpu().item()
        )
        bucket_metrics[f"observed_rows_{bucket}"] = float(
            (flat_obs & mask).detach().to(dtype=torch.float32).sum().cpu().item()
        )
        bucket_metrics[f"population_rows_{bucket}"] = float(mask.detach().to(dtype=torch.float32).sum().cpu().item())
        bucket_metrics[f"discounted_weight_{bucket}"] = float(
            local_discount_weights.masked_select(mask).sum().detach().cpu().item()
        )
    observed_ipw = flat_obs.to(dtype=predictions.dtype) / flat_pi.to(dtype=predictions.dtype)
    observed_ipw_values = observed_ipw.masked_select(flat_obs)
    ipw_sum = observed_ipw_values.sum()
    ipw_sq_sum = (observed_ipw_values * observed_ipw_values).sum()
    if float(ipw_sq_sum.detach().cpu().item()) > 0.0:
        effective_sample_size = float(((ipw_sum * ipw_sum) / ipw_sq_sum).detach().cpu().item())
    else:
        effective_sample_size = 0.0
    sample_count = max(1, int(root_indices.reshape(-1).numel()))
    obs_mass = (masses * flat_obs.to(dtype=masses.dtype)).sum()
    return RolloutLoss(
        loss=loss,
        root_loss=root_loss,
        local_loss=local_loss,
        local_proxy_loss=float(local_proxy_loss.detach().cpu().item()),
        local_oracle_observed_ipw_loss=float(local_oracle_observed_ipw_loss.detach().cpu().item()),
        local_ipw_correction=float(local_ipw_correction.detach().cpu().item()),
        local_corrected_loss=float(local_loss.detach().cpu().item()),
        discounted_root_weight=float(discounted_root_weight.detach().cpu().item()),
        discounted_nonroot_weight=float(discounted_nonroot_weight.detach().cpu().item()),
        observed_count=int(flat_obs.detach().to(dtype=torch.int64).sum().cpu().item()),
        population_count=int(predictions.reshape(-1).numel()),
        root_observed_count=int(root_obs.detach().to(dtype=torch.int64).sum().cpu().item()),
        root_population_count=int(root_indices.reshape(-1).numel()),
        nonroot_observed_count=int(nonroot_obs.detach().to(dtype=torch.int64).sum().cpu().item()),
        nonroot_population_count=int(nonroot_mask.detach().to(dtype=torch.int64).sum().cpu().item()),
        observed_rows_per_doc=float(flat_obs.detach().to(dtype=torch.float32).sum().cpu().item()) / float(sample_count),
        root_observed_rows_per_doc=float(root_obs.detach().to(dtype=torch.float32).sum().cpu().item()) / float(sample_count),
        nonroot_observed_rows_per_doc=float(nonroot_obs.detach().to(dtype=torch.float32).sum().cpu().item()) / float(sample_count),
        max_ipw_weight=float(observed_ipw_values.max().detach().cpu().item()) if observed_ipw_values.numel() else 0.0,
        effective_sample_size=effective_sample_size,
        observed_mass=float(obs_mass.detach().cpu().item()),
        population_mass=float(masses.detach().sum().cpu().item()),
        bucket_metrics=bucket_metrics,
    )


def _zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), dtype=reference.dtype, device=reference.device)


def _first_finite_metric(row: Mapping[str, object], keys: Sequence[str]) -> float | None:
    for key in keys:
        try:
            value = float(row[key])
        except Exception:
            continue
        if math.isfinite(value):
            return value
    return None


def _assert_finite_scalar(name: str, value: object, *, context: str) -> float:
    try:
        numeric = float(value)
    except Exception as exc:
        raise RuntimeError(f"{context}: {name} is not numeric: {value!r}") from exc
    if not math.isfinite(numeric):
        raise RuntimeError(f"{context}: non-finite {name}={numeric!r}")
    return numeric


def _assert_nonempty_batches(count: int, *, context: str) -> None:
    if int(count) <= 0:
        raise RuntimeError(f"{context}: no training batches were produced")


def _assert_finite_metrics(row: Mapping[str, object], keys: Sequence[str], *, context: str) -> None:
    for key in keys:
        if key in row and row[key] is not None:
            _assert_finite_scalar(key, row[key], context=context)


def _fmt_progress_value(value: object) -> str:
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(numeric):
        return str(numeric)
    if abs(numeric) >= 1000.0 or (0.0 < abs(numeric) < 1e-3):
        return f"{numeric:.4e}"
    return f"{numeric:.6g}"


def _emit_epoch_progress(
    label: str,
    *,
    epoch: int,
    epochs: int,
    train_loss: float,
    metrics: Mapping[str, object],
    started_at: float,
    progress_every_epochs: int,
) -> None:
    if not label or int(progress_every_epochs) <= 0:
        return
    if epoch != 1 and epoch != int(epochs) and epoch % int(progress_every_epochs) != 0:
        return
    elapsed = time.perf_counter() - float(started_at)
    parts = [
        f"[fno-sketch] {label}",
        f"epoch={int(epoch)}/{int(epochs)}",
        f"train_loss={_fmt_progress_value(train_loss)}",
    ]
    for key in (
        "root_mae",
        "official_f_on_learned_root_mae",
        "learned_f_on_exact_root_mae",
        "merge_state_mae",
        "train_observed_rows_per_doc",
        "train_nonroot_observed_rows_per_doc",
        "train_max_ipw_weight",
    ):
        if key in metrics:
            parts.append(f"{key}={_fmt_progress_value(metrics[key])}")
    parts.append(f"elapsed_s={elapsed:.1f}")
    print(" ".join(parts), flush=True)


def _emit_batch_progress(
    label: str,
    *,
    epoch: int,
    epochs: int,
    batch_index: int,
    batches: int,
    loss: float,
    started_at: float,
    progress_every_batches: int,
) -> None:
    if not label or int(progress_every_batches) <= 0:
        return
    if (
        int(batch_index) != 1
        and int(batch_index) != int(batches)
        and int(batch_index) % int(progress_every_batches) != 0
    ):
        return
    elapsed = time.perf_counter() - float(started_at)
    print(
        "[fno-sketch] "
        f"{label} epoch={int(epoch)}/{int(epochs)} "
        f"batch={int(batch_index)}/{int(batches)} "
        f"loss={_fmt_progress_value(loss)} elapsed_s={elapsed:.1f}",
        flush=True,
    )


def _train_f_stage(
    model: SketchStateFNO,
    *,
    states: torch.Tensor,
    targets: torch.Tensor,
    eval_callback: Callable[[int, float], dict[str, float]] | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    grad_accum_steps: int,
    progress_label: str = "",
    progress_every_epochs: int = 1,
    progress_every_batches: int = 0,
) -> tuple[list[float], list[dict[str, float]]]:
    model.freeze_for_f()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    losses: list[float] = []
    losses_eval: list[dict[str, float]] = []
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    progress_started_at = time.perf_counter()
    for _epoch in range(int(epochs)):
        epoch_losses: list[float] = []
        accum = max(1, int(grad_accum_steps))
        batches = list(_batch_indices(states.shape[0], int(batch_size), device=states.device))
        _assert_nonempty_batches(len(batches), context=f"{progress_label or 'f-stage'} epoch={_epoch + 1}")
        optimizer.zero_grad(set_to_none=True)
        for batch_index, idx in enumerate(batches, start=1):
            pred = model.predict_transformed(states[idx])
            loss = F.mse_loss(pred, targets[idx])
            loss_value = _assert_finite_scalar(
                "loss",
                loss.detach().cpu().item(),
                context=f"{progress_label or 'f-stage'} epoch={_epoch + 1} batch={batch_index}",
            )
            _emit_batch_progress(
                progress_label,
                epoch=_epoch + 1,
                epochs=int(epochs),
                batch_index=batch_index,
                batches=len(batches),
                loss=loss_value,
                started_at=progress_started_at,
                progress_every_batches=int(progress_every_batches),
            )
            epoch_losses.append(loss_value)
            (loss / float(accum)).backward()
            if batch_index % accum == 0 or batch_index == len(batches):
                if float(grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, float(grad_clip_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        mean_loss = _assert_finite_scalar(
            "epoch_mean_loss",
            sum(epoch_losses) / max(1, len(epoch_losses)),
            context=f"{progress_label or 'f-stage'} epoch={_epoch + 1}",
        )
        losses.append(mean_loss)
        selection_score = float(mean_loss)
        eval_row: dict[str, float] = {}
        if eval_callback is not None:
            eval_row = eval_callback(_epoch + 1, mean_loss)
            eval_row["epoch"] = float(_epoch + 1)
            eval_row["train_loss"] = float(mean_loss)
            _assert_finite_metrics(
                eval_row,
                ("root_mae", "learned_f_on_exact_root_mae"),
                context=f"{progress_label or 'f-stage'} eval epoch={_epoch + 1}",
            )
            losses_eval.append(eval_row)
            metric_score = _first_finite_metric(
                eval_row,
                ("learned_f_on_exact_root_mae", "root_mae"),
            )
            if metric_score is not None:
                selection_score = metric_score
        _emit_epoch_progress(
            progress_label,
            epoch=_epoch + 1,
            epochs=int(epochs),
            train_loss=mean_loss,
            metrics=eval_row,
            started_at=progress_started_at,
            progress_every_epochs=int(progress_every_epochs),
        )
        if selection_score < best_score:
            best_score = float(selection_score)
            best_state = _model_state_without_metadata(model)
    if best_state is not None:
        model.load_state_dict(best_state)
    return losses, losses_eval


def _train_g_stage(
    model: SketchStateFNO,
    *,
    left: torch.Tensor,
    right: torch.Tensor,
    target_state: torch.Tensor,
    target_scalar: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    grad_accum_steps: int,
    state_loss_weight: float,
    scalar_loss_weight: float,
    eval_callback: Callable[[int, float], dict[str, float]] | None,
    progress_label: str = "",
    progress_every_epochs: int = 1,
    progress_every_batches: int = 0,
) -> tuple[list[float], list[dict[str, float]]]:
    model.freeze_for_g()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    losses: list[float] = []
    losses_eval: list[dict[str, float]] = []
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    progress_started_at = time.perf_counter()
    for _epoch in range(int(epochs)):
        epoch_losses: list[float] = []
        accum = max(1, int(grad_accum_steps))
        batches = list(_batch_indices(left.shape[0], int(batch_size), device=left.device))
        _assert_nonempty_batches(len(batches), context=f"{progress_label or 'g-stage'} epoch={_epoch + 1}")
        optimizer.zero_grad(set_to_none=True)
        for batch_index, idx in enumerate(batches, start=1):
            pred_state = model.merge(left[idx], right[idx])
            pred_scalar = model.predict_transformed(pred_state)
            state_loss = F.mse_loss(pred_state, target_state[idx])
            scalar_loss = F.mse_loss(pred_scalar, target_scalar[idx])
            loss = float(state_loss_weight) * state_loss + float(scalar_loss_weight) * scalar_loss
            loss_value = _assert_finite_scalar(
                "loss",
                loss.detach().cpu().item(),
                context=f"{progress_label or 'g-stage'} epoch={_epoch + 1} batch={batch_index}",
            )
            _emit_batch_progress(
                progress_label,
                epoch=_epoch + 1,
                epochs=int(epochs),
                batch_index=batch_index,
                batches=len(batches),
                loss=loss_value,
                started_at=progress_started_at,
                progress_every_batches=int(progress_every_batches),
            )
            epoch_losses.append(loss_value)
            (loss / float(accum)).backward()
            if batch_index % accum == 0 or batch_index == len(batches):
                if float(grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, float(grad_clip_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        mean_loss = _assert_finite_scalar(
            "epoch_mean_loss",
            sum(epoch_losses) / max(1, len(epoch_losses)),
            context=f"{progress_label or 'g-stage'} epoch={_epoch + 1}",
        )
        losses.append(mean_loss)
        selection_score = float(mean_loss)
        eval_row: dict[str, float] = {}
        if eval_callback is not None:
            eval_row = eval_callback(_epoch + 1, mean_loss)
            eval_row["epoch"] = float(_epoch + 1)
            eval_row["train_loss"] = float(mean_loss)
            _assert_finite_metrics(
                eval_row,
                ("root_mae", "official_f_on_learned_root_mae"),
                context=f"{progress_label or 'g-stage'} eval epoch={_epoch + 1}",
            )
            losses_eval.append(eval_row)
            metric_score = _first_finite_metric(
                eval_row,
                ("merge_state_mae", "official_f_on_learned_root_mae", "root_mae"),
            )
            if metric_score is not None:
                selection_score = metric_score
        _emit_epoch_progress(
            progress_label,
            epoch=_epoch + 1,
            epochs=int(epochs),
            train_loss=mean_loss,
            metrics=eval_row,
            started_at=progress_started_at,
            progress_every_epochs=int(progress_every_epochs),
        )
        if selection_score < best_score:
            best_score = float(selection_score)
            best_state = _model_state_without_metadata(model)
    if best_state is not None:
        model.load_state_dict(best_state)
    return losses, losses_eval


def _mean_metric(rows: Sequence[dict[str, float]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
    return float(sum(values) / len(values)) if values else float("nan")


def _max_metric(rows: Sequence[dict[str, float]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
    return float(max(values)) if values else float("nan")


def _train_f_stage_rollout(
    model: SketchStateFNO,
    *,
    samples: Sequence[ExactStateSample],
    spec: ExactStateSpec,
    device: torch.device,
    state_transform: StateTransform,
    scalar_transform: ScalarTransform,
    readout: Callable[[np.ndarray], float],
    eval_callback: Callable[[int, float], dict[str, float]] | None,
    epochs: int,
    batch_size: int,
    rollout_min_docs_per_batch: int,
    rollout_max_docs_per_batch: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    grad_accum_steps: int,
    local_law_weight: float,
    local_law_leaf_discount_gamma: float,
    objective_loss_weight: float,
    exact_state_anchor_weight: float,
    oracle_observation_design: str = "root_only",
    sampled_node_rate: float = 0.0,
    sampled_node_seed: int = 0,
    root_label_share: float = 1.0,
    mass_target_per_doc: float = 1.0,
    local_label_pool: str = "nonroot",
    local_label_allocation: str = "span_mass",
    progress_label: str = "",
    progress_every_epochs: int = 1,
    progress_every_batches: int = 0,
) -> tuple[list[float], list[dict[str, float]], list[dict[str, float]]]:
    model.freeze_for_f()
    proxy_model = _fhat_proxy_snapshot(model, device=device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    losses: list[float] = []
    losses_eval: list[dict[str, float]] = []
    train_components: list[dict[str, float]] = []
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    progress_started_at = time.perf_counter()
    for _epoch in range(int(epochs)):
        epoch_losses: list[float] = []
        epoch_components: list[dict[str, float]] = []
        accum = max(1, int(grad_accum_steps))
        batches = list(
            _sample_batches_by_node_rows(
                samples,
                batch_size=int(batch_size),
                min_docs_per_batch=int(rollout_min_docs_per_batch),
                max_docs_per_batch=int(rollout_max_docs_per_batch),
            )
        )
        _assert_nonempty_batches(len(batches), context=f"{progress_label or 'rollout-f-stage'} epoch={_epoch + 1}")
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch_samples in enumerate(batches, start=1):
            with torch.no_grad():
                rollout = _rollout_batch(
                    model,
                    batch_samples,
                    spec=spec,
                    device=device,
                    state_transform=state_transform,
                    scalar_transform=scalar_transform,
                    readout=readout,
                    detach_merge_states=True,
                    proxy_model=proxy_model,
                    oracle_observation_design=str(oracle_observation_design),
                    sampled_node_rate=float(sampled_node_rate),
                    sampled_node_seed=int(sampled_node_seed),
                    root_label_share=float(root_label_share),
                    mass_target_per_doc=float(mass_target_per_doc),
                    local_label_pool=str(local_label_pool),
                    local_label_allocation=str(local_label_allocation),
                )
            pred = model.predict_transformed(rollout.states)
            rollout_loss = _single_lambda_rollout_loss(
                pred,
                rollout.targets,
                root_indices=rollout.root_indices,
                local_law_weight=float(local_law_weight),
                local_law_leaf_discount_gamma=float(local_law_leaf_discount_gamma),
                proxy_targets=rollout.proxy_targets,
                oracle_targets=rollout.oracle_targets,
                observed=rollout.observed,
                propensity=rollout.propensity,
                depths=rollout.depths,
                node_masses=rollout.node_masses,
            )
            if float(exact_state_anchor_weight) > 0.0 and rollout.exact_states.numel() > 0:
                exact_anchor_loss = F.mse_loss(
                    model.predict_transformed(rollout.exact_states),
                    rollout.targets,
                )
            else:
                exact_anchor_loss = _zero_like_loss(rollout_loss.loss)
            state_regularizer_loss = _zero_like_loss(rollout_loss.loss)
            loss = (
                float(objective_loss_weight) * rollout_loss.loss
                + float(exact_state_anchor_weight) * exact_anchor_loss
            )
            batch_context = f"{progress_label or 'rollout-f-stage'} epoch={_epoch + 1} batch={batch_index}"
            loss_value = _assert_finite_scalar("loss", loss.detach().cpu().item(), context=batch_context)
            _emit_batch_progress(
                progress_label,
                epoch=_epoch + 1,
                epochs=int(epochs),
                batch_index=batch_index,
                batches=len(batches),
                loss=loss_value,
                started_at=progress_started_at,
                progress_every_batches=int(progress_every_batches),
            )
            epoch_losses.append(loss_value)
            epoch_components.append(
                {
                    "total_loss": loss_value,
                    "objective_loss": _assert_finite_scalar(
                        "objective_loss", rollout_loss.loss.detach().cpu().item(), context=batch_context
                    ),
                    "root_loss": _assert_finite_scalar(
                        "root_loss", rollout_loss.root_loss.detach().cpu().item(), context=batch_context
                    ),
                    "local_loss": _assert_finite_scalar(
                        "local_loss", rollout_loss.local_loss.detach().cpu().item(), context=batch_context
                    ),
                    "local_proxy_loss": float(rollout_loss.local_proxy_loss),
                    "local_oracle_observed_ipw_loss": float(rollout_loss.local_oracle_observed_ipw_loss),
                    "local_ipw_correction": float(rollout_loss.local_ipw_correction),
                    "local_corrected_loss": float(rollout_loss.local_corrected_loss),
                    "discounted_root_weight": float(rollout_loss.discounted_root_weight),
                    "discounted_nonroot_weight": float(rollout_loss.discounted_nonroot_weight),
                    "exact_anchor_loss": _assert_finite_scalar(
                        "exact_anchor_loss", exact_anchor_loss.detach().cpu().item(), context=batch_context
                    ),
                    "state_regularizer_loss": _assert_finite_scalar(
                        "state_regularizer_loss",
                        state_regularizer_loss.detach().cpu().item(),
                        context=batch_context,
                    ),
                    "observed_rows": float(rollout_loss.observed_count),
                    "population_rows": float(rollout_loss.population_count),
                    "root_observed_rows": float(rollout_loss.root_observed_count),
                    "root_population_rows": float(rollout_loss.root_population_count),
                    "nonroot_observed_rows": float(rollout_loss.nonroot_observed_count),
                    "nonroot_population_rows": float(rollout_loss.nonroot_population_count),
                    "observed_rows_per_doc": float(rollout_loss.observed_rows_per_doc),
                    "root_observed_rows_per_doc": float(rollout_loss.root_observed_rows_per_doc),
                    "nonroot_observed_rows_per_doc": float(rollout_loss.nonroot_observed_rows_per_doc),
                    "max_ipw_weight": float(rollout_loss.max_ipw_weight),
                    "effective_sample_size": float(rollout_loss.effective_sample_size),
                    "observed_mass": float(rollout_loss.observed_mass),
                    "population_mass": float(rollout_loss.population_mass),
                    **rollout_loss.bucket_metrics,
                }
            )
            (loss / float(accum)).backward()
            if batch_index % accum == 0 or batch_index == len(batches):
                if float(grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, float(grad_clip_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        mean_loss = _assert_finite_scalar(
            "epoch_mean_loss",
            sum(epoch_losses) / max(1, len(epoch_losses)),
            context=f"{progress_label or 'rollout-f-stage'} epoch={_epoch + 1}",
        )
        losses.append(mean_loss)
        row = {
            "train_total_loss": float(mean_loss),
            "train_objective_loss": _mean_metric(epoch_components, "objective_loss"),
            "train_root_loss": _mean_metric(epoch_components, "root_loss"),
            "train_local_loss": _mean_metric(epoch_components, "local_loss"),
            "train_local_proxy_loss": _mean_metric(epoch_components, "local_proxy_loss"),
            "train_local_oracle_observed_ipw_loss": _mean_metric(epoch_components, "local_oracle_observed_ipw_loss"),
            "train_local_ipw_correction": _mean_metric(epoch_components, "local_ipw_correction"),
            "train_local_corrected_loss": _mean_metric(epoch_components, "local_corrected_loss"),
            "train_discounted_root_weight": _mean_metric(epoch_components, "discounted_root_weight"),
            "train_discounted_nonroot_weight": _mean_metric(epoch_components, "discounted_nonroot_weight"),
            "train_exact_anchor_loss": _mean_metric(epoch_components, "exact_anchor_loss"),
            "train_state_regularizer_loss": _mean_metric(epoch_components, "state_regularizer_loss"),
            "train_observed_rows": _mean_metric(epoch_components, "observed_rows"),
            "train_population_rows": _mean_metric(epoch_components, "population_rows"),
            "train_root_observed_rows": _mean_metric(epoch_components, "root_observed_rows"),
            "train_root_population_rows": _mean_metric(epoch_components, "root_population_rows"),
            "train_nonroot_observed_rows": _mean_metric(epoch_components, "nonroot_observed_rows"),
            "train_nonroot_population_rows": _mean_metric(epoch_components, "nonroot_population_rows"),
            "train_observed_rows_per_doc": _mean_metric(epoch_components, "observed_rows_per_doc"),
            "train_root_observed_rows_per_doc": _mean_metric(epoch_components, "root_observed_rows_per_doc"),
            "train_nonroot_observed_rows_per_doc": _mean_metric(epoch_components, "nonroot_observed_rows_per_doc"),
            "train_max_ipw_weight": _max_metric(epoch_components, "max_ipw_weight"),
            "train_effective_sample_size": _mean_metric(epoch_components, "effective_sample_size"),
            "train_observed_mass": _mean_metric(epoch_components, "observed_mass"),
            "train_population_mass": _mean_metric(epoch_components, "population_mass"),
        }
        for key in LOSS_BUCKET_METRIC_KEYS:
            row[f"train_{key}"] = _mean_metric(epoch_components, key)
        _assert_finite_metrics(
            row,
            (
                "train_total_loss",
                "train_objective_loss",
                "train_root_loss",
                "train_local_loss",
                "train_observed_rows_per_doc",
                "train_max_ipw_weight",
            ),
            context=f"{progress_label or 'rollout-f-stage'} train metrics epoch={_epoch + 1}",
        )
        train_components.append(row)
        selection_score = float(mean_loss)
        eval_row: dict[str, float] = {}
        if eval_callback is not None:
            eval_row = eval_callback(_epoch + 1, mean_loss)
            eval_row["epoch"] = float(_epoch + 1)
            eval_row["train_loss"] = float(mean_loss)
            eval_row.update(row)
            _assert_finite_metrics(
                eval_row,
                ("root_mae", "learned_f_on_exact_root_mae"),
                context=f"{progress_label or 'rollout-f-stage'} eval epoch={_epoch + 1}",
            )
            losses_eval.append(eval_row)
            metric_score = _first_finite_metric(
                eval_row,
                ("learned_f_on_exact_root_mae", "root_mae"),
            )
            if metric_score is not None:
                selection_score = metric_score
        _emit_epoch_progress(
            progress_label,
            epoch=_epoch + 1,
            epochs=int(epochs),
            train_loss=mean_loss,
            metrics=eval_row or row,
            started_at=progress_started_at,
            progress_every_epochs=int(progress_every_epochs),
        )
        if selection_score < best_score:
            best_score = float(selection_score)
            best_state = _model_state_without_metadata(model)
    if best_state is not None:
        model.load_state_dict(best_state)
    return losses, losses_eval, train_components


def _train_g_stage_rollout(
    model: SketchStateFNO,
    *,
    samples: Sequence[ExactStateSample],
    spec: ExactStateSpec,
    device: torch.device,
    state_transform: StateTransform,
    scalar_transform: ScalarTransform,
    readout: Callable[[np.ndarray], float],
    epochs: int,
    batch_size: int,
    rollout_min_docs_per_batch: int,
    rollout_max_docs_per_batch: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    grad_accum_steps: int,
    state_loss_weight: float,
    local_law_weight: float,
    local_law_leaf_discount_gamma: float,
    objective_loss_weight: float,
    oracle_observation_design: str = "root_only",
    sampled_node_rate: float = 0.0,
    sampled_node_seed: int = 0,
    root_label_share: float = 1.0,
    mass_target_per_doc: float = 1.0,
    local_label_pool: str = "nonroot",
    local_label_allocation: str = "span_mass",
    eval_callback: Callable[[int, float], dict[str, float]] | None,
    progress_label: str = "",
    progress_every_epochs: int = 1,
    progress_every_batches: int = 0,
) -> tuple[list[float], list[dict[str, float]], list[dict[str, float]]]:
    model.freeze_for_g()
    proxy_model = _fhat_proxy_snapshot(model, device=device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    losses: list[float] = []
    losses_eval: list[dict[str, float]] = []
    train_components: list[dict[str, float]] = []
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    progress_started_at = time.perf_counter()
    for _epoch in range(int(epochs)):
        epoch_losses: list[float] = []
        epoch_components: list[dict[str, float]] = []
        accum = max(1, int(grad_accum_steps))
        batches = list(
            _sample_batches_by_node_rows(
                samples,
                batch_size=int(batch_size),
                min_docs_per_batch=int(rollout_min_docs_per_batch),
                max_docs_per_batch=int(rollout_max_docs_per_batch),
            )
        )
        _assert_nonempty_batches(len(batches), context=f"{progress_label or 'rollout-g-stage'} epoch={_epoch + 1}")
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch_samples in enumerate(batches, start=1):
            rollout = _rollout_batch(
                model,
                batch_samples,
                spec=spec,
                device=device,
                state_transform=state_transform,
                scalar_transform=scalar_transform,
                readout=readout,
                detach_merge_states=False,
                proxy_model=proxy_model,
                oracle_observation_design=str(oracle_observation_design),
                sampled_node_rate=float(sampled_node_rate),
                sampled_node_seed=int(sampled_node_seed),
                root_label_share=float(root_label_share),
                mass_target_per_doc=float(mass_target_per_doc),
                local_label_pool=str(local_label_pool),
                local_label_allocation=str(local_label_allocation),
            )
            pred = model.predict_transformed(rollout.states)
            rollout_loss = _single_lambda_rollout_loss(
                pred,
                rollout.targets,
                root_indices=rollout.root_indices,
                local_law_weight=float(local_law_weight),
                local_law_leaf_discount_gamma=float(local_law_leaf_discount_gamma),
                proxy_targets=rollout.proxy_targets,
                oracle_targets=rollout.oracle_targets,
                observed=rollout.observed,
                propensity=rollout.propensity,
                depths=rollout.depths,
                node_masses=rollout.node_masses,
            )
            exact_anchor_loss = _zero_like_loss(rollout_loss.loss)
            if rollout.merge_indices.numel() > 0:
                state_regularizer_loss = F.mse_loss(
                    rollout.states.index_select(0, rollout.merge_indices),
                    rollout.exact_states.index_select(0, rollout.merge_indices),
                )
            else:
                state_regularizer_loss = _zero_like_loss(rollout_loss.loss)
            loss = (
                float(objective_loss_weight) * rollout_loss.loss
                + float(state_loss_weight) * state_regularizer_loss
            )
            batch_context = f"{progress_label or 'rollout-g-stage'} epoch={_epoch + 1} batch={batch_index}"
            loss_value = _assert_finite_scalar("loss", loss.detach().cpu().item(), context=batch_context)
            _emit_batch_progress(
                progress_label,
                epoch=_epoch + 1,
                epochs=int(epochs),
                batch_index=batch_index,
                batches=len(batches),
                loss=loss_value,
                started_at=progress_started_at,
                progress_every_batches=int(progress_every_batches),
            )
            epoch_losses.append(loss_value)
            epoch_components.append(
                {
                    "total_loss": loss_value,
                    "objective_loss": _assert_finite_scalar(
                        "objective_loss", rollout_loss.loss.detach().cpu().item(), context=batch_context
                    ),
                    "root_loss": _assert_finite_scalar(
                        "root_loss", rollout_loss.root_loss.detach().cpu().item(), context=batch_context
                    ),
                    "local_loss": _assert_finite_scalar(
                        "local_loss", rollout_loss.local_loss.detach().cpu().item(), context=batch_context
                    ),
                    "local_proxy_loss": float(rollout_loss.local_proxy_loss),
                    "local_oracle_observed_ipw_loss": float(rollout_loss.local_oracle_observed_ipw_loss),
                    "local_ipw_correction": float(rollout_loss.local_ipw_correction),
                    "local_corrected_loss": float(rollout_loss.local_corrected_loss),
                    "discounted_root_weight": float(rollout_loss.discounted_root_weight),
                    "discounted_nonroot_weight": float(rollout_loss.discounted_nonroot_weight),
                    "exact_anchor_loss": _assert_finite_scalar(
                        "exact_anchor_loss", exact_anchor_loss.detach().cpu().item(), context=batch_context
                    ),
                    "state_regularizer_loss": _assert_finite_scalar(
                        "state_regularizer_loss",
                        state_regularizer_loss.detach().cpu().item(),
                        context=batch_context,
                    ),
                    "observed_rows": float(rollout_loss.observed_count),
                    "population_rows": float(rollout_loss.population_count),
                    "root_observed_rows": float(rollout_loss.root_observed_count),
                    "root_population_rows": float(rollout_loss.root_population_count),
                    "nonroot_observed_rows": float(rollout_loss.nonroot_observed_count),
                    "nonroot_population_rows": float(rollout_loss.nonroot_population_count),
                    "observed_rows_per_doc": float(rollout_loss.observed_rows_per_doc),
                    "root_observed_rows_per_doc": float(rollout_loss.root_observed_rows_per_doc),
                    "nonroot_observed_rows_per_doc": float(rollout_loss.nonroot_observed_rows_per_doc),
                    "max_ipw_weight": float(rollout_loss.max_ipw_weight),
                    "effective_sample_size": float(rollout_loss.effective_sample_size),
                    "observed_mass": float(rollout_loss.observed_mass),
                    "population_mass": float(rollout_loss.population_mass),
                    **rollout_loss.bucket_metrics,
                }
            )
            if bool(getattr(loss, "requires_grad", False)):
                (loss / float(accum)).backward()
            if batch_index % accum == 0 or batch_index == len(batches):
                if any(p.grad is not None for p in trainable):
                    if float(grad_clip_norm) > 0:
                        torch.nn.utils.clip_grad_norm_(trainable, float(grad_clip_norm))
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        mean_loss = _assert_finite_scalar(
            "epoch_mean_loss",
            sum(epoch_losses) / max(1, len(epoch_losses)),
            context=f"{progress_label or 'rollout-g-stage'} epoch={_epoch + 1}",
        )
        losses.append(mean_loss)
        row = {
            "train_total_loss": float(mean_loss),
            "train_objective_loss": _mean_metric(epoch_components, "objective_loss"),
            "train_root_loss": _mean_metric(epoch_components, "root_loss"),
            "train_local_loss": _mean_metric(epoch_components, "local_loss"),
            "train_local_proxy_loss": _mean_metric(epoch_components, "local_proxy_loss"),
            "train_local_oracle_observed_ipw_loss": _mean_metric(epoch_components, "local_oracle_observed_ipw_loss"),
            "train_local_ipw_correction": _mean_metric(epoch_components, "local_ipw_correction"),
            "train_local_corrected_loss": _mean_metric(epoch_components, "local_corrected_loss"),
            "train_discounted_root_weight": _mean_metric(epoch_components, "discounted_root_weight"),
            "train_discounted_nonroot_weight": _mean_metric(epoch_components, "discounted_nonroot_weight"),
            "train_exact_anchor_loss": _mean_metric(epoch_components, "exact_anchor_loss"),
            "train_state_regularizer_loss": _mean_metric(epoch_components, "state_regularizer_loss"),
            "train_observed_rows": _mean_metric(epoch_components, "observed_rows"),
            "train_population_rows": _mean_metric(epoch_components, "population_rows"),
            "train_root_observed_rows": _mean_metric(epoch_components, "root_observed_rows"),
            "train_root_population_rows": _mean_metric(epoch_components, "root_population_rows"),
            "train_nonroot_observed_rows": _mean_metric(epoch_components, "nonroot_observed_rows"),
            "train_nonroot_population_rows": _mean_metric(epoch_components, "nonroot_population_rows"),
            "train_observed_rows_per_doc": _mean_metric(epoch_components, "observed_rows_per_doc"),
            "train_root_observed_rows_per_doc": _mean_metric(epoch_components, "root_observed_rows_per_doc"),
            "train_nonroot_observed_rows_per_doc": _mean_metric(epoch_components, "nonroot_observed_rows_per_doc"),
            "train_max_ipw_weight": _max_metric(epoch_components, "max_ipw_weight"),
            "train_effective_sample_size": _mean_metric(epoch_components, "effective_sample_size"),
            "train_observed_mass": _mean_metric(epoch_components, "observed_mass"),
            "train_population_mass": _mean_metric(epoch_components, "population_mass"),
        }
        for key in LOSS_BUCKET_METRIC_KEYS:
            row[f"train_{key}"] = _mean_metric(epoch_components, key)
        _assert_finite_metrics(
            row,
            (
                "train_total_loss",
                "train_objective_loss",
                "train_root_loss",
                "train_local_loss",
                "train_observed_rows_per_doc",
                "train_max_ipw_weight",
            ),
            context=f"{progress_label or 'rollout-g-stage'} train metrics epoch={_epoch + 1}",
        )
        train_components.append(row)
        selection_score = float(mean_loss)
        eval_row: dict[str, float] = {}
        if eval_callback is not None:
            eval_row = eval_callback(_epoch + 1, mean_loss)
            eval_row["epoch"] = float(_epoch + 1)
            eval_row["train_loss"] = float(mean_loss)
            eval_row.update(row)
            _assert_finite_metrics(
                eval_row,
                ("root_mae", "official_f_on_learned_root_mae"),
                context=f"{progress_label or 'rollout-g-stage'} eval epoch={_epoch + 1}",
            )
            losses_eval.append(eval_row)
            metric_score = _first_finite_metric(
                eval_row,
                ("merge_state_mae", "official_f_on_learned_root_mae", "root_mae"),
            )
            if metric_score is not None:
                selection_score = metric_score
        _emit_epoch_progress(
            progress_label,
            epoch=_epoch + 1,
            epochs=int(epochs),
            train_loss=mean_loss,
            metrics=eval_row or row,
            started_at=progress_started_at,
            progress_every_epochs=int(progress_every_epochs),
        )
        if selection_score < best_score:
            best_score = float(selection_score)
            best_state = _model_state_without_metadata(model)
    if best_state is not None:
        model.load_state_dict(best_state)
    return losses, losses_eval, train_components


def _mean_or_nan(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


@torch.no_grad()
def _evaluate(
    model: SketchStateFNO,
    samples: Sequence[ExactStateSample],
    *,
    spec: ExactStateSpec,
    device: torch.device,
    state_transform: StateTransform,
    scalar_transform: ScalarTransform,
    readout: Callable[[np.ndarray], float],
    batch_size: int = 16384,
) -> dict[str, float]:
    root_abs: list[float] = []
    root_rel: list[float] = []
    exact_root_abs: list[float] = []
    exact_root_rel: list[float] = []
    official_on_learned_abs: list[float] = []
    official_on_learned_rel: list[float] = []
    state_mae: list[float] = []
    all_node_readout_abs: list[float] = []
    leaf_readout_abs: list[float] = []
    internal_readout_abs: list[float] = []
    root_readout_abs: list[float] = []
    merge_state_internal_abs: list[float] = []
    merge_state_root_abs: list[float] = []
    merge_readout_abs: list[float] = []
    merge_readout_internal_abs: list[float] = []
    merge_readout_root_abs: list[float] = []
    official_merge_abs: list[float] = []
    official_merge_internal_abs: list[float] = []
    official_merge_root_abs: list[float] = []
    hll_register_fractional_abs: list[float] = []
    hll_merge_register_rounded_mae: list[float] = []
    hll_merge_register_linf: list[float] = []
    hll_merge_register_exact: list[float] = []
    hll_zero_scalar_bad_state: list[float] = []
    hll_within_tol_bad_state: list[float] = []
    hll_future_context_abs: list[float] = []
    hll_zero_scalar_bad_future: list[float] = []
    hll_within_tol_bad_future: list[float] = []
    exact_node_readout_by_depth: dict[int, list[float]] = {}
    merge_state_by_depth: dict[int, list[float]] = {}
    merge_readout_by_depth: dict[int, list[float]] = {}
    official_merge_by_depth: dict[int, list[float]] = {}
    pred_scalars: list[float] = []
    pred_transformed: list[float] = []
    pred_transformed_unclamped: list[float] = []
    learned_state_chunks: list[np.ndarray] = []
    learned_root_state_chunks: list[np.ndarray] = []
    learned_nonroot_state_chunks: list[np.ndarray] = []
    learned_state_chunks_by_depth: dict[int, list[np.ndarray]] = {}
    merge_carrier_norms: list[float] = []
    merge_projection_delta_norms: list[float] = []
    merge_projection_delta_to_carrier_norms: list[float] = []
    merge_projection_delta_to_carrier_norms_root: list[float] = []
    merge_projection_delta_to_carrier_norms_nonroot: list[float] = []
    eval_batch_size = max(1, int(batch_size))
    valid_bounds = _state_valid_bounds(state_transform)
    hll_alignment_enabled = str(spec.target_kind) == "hll_register_space"
    hll_scalar_tol = max(1e-6, 1e-3 * float(scalar_transform.scale))

    def record_state_values(state_tensor: torch.Tensor, *, depth: int, is_root: bool) -> None:
        values = state_tensor.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        learned_state_chunks.append(values)
        learned_state_chunks_by_depth.setdefault(int(depth), []).append(values)
        if bool(is_root):
            learned_root_state_chunks.append(values)
        else:
            learned_nonroot_state_chunks.append(values)

    def record_merge_components(
        carrier_tensor: torch.Tensor,
        residual_tensor: torch.Tensor,
        *,
        is_root: bool,
    ) -> None:
        carrier_norm = carrier_tensor.detach().norm(dim=1).cpu().numpy().reshape(-1).astype(np.float64)
        residual_norm = residual_tensor.detach().norm(dim=1).cpu().numpy().reshape(-1).astype(np.float64)
        ratio = residual_norm / np.maximum(carrier_norm, 1e-12)
        merge_carrier_norms.extend(float(value) for value in carrier_norm)
        merge_projection_delta_norms.extend(float(value) for value in residual_norm)
        merge_projection_delta_to_carrier_norms.extend(float(value) for value in ratio)
        if bool(is_root):
            merge_projection_delta_to_carrier_norms_root.extend(float(value) for value in ratio)
        else:
            merge_projection_delta_to_carrier_norms_nonroot.extend(float(value) for value in ratio)
    for batch_samples in _sample_batches_by_node_rows(samples, batch_size=eval_batch_size):
        exact_node_rows: list[np.ndarray] = []
        exact_node_targets: list[float] = []
        exact_node_depths: list[int] = []
        exact_node_root_depths: list[int] = []
        for sample in batch_samples:
            sample_depths = list(sample.node_depths or [])
            root_depth = (
                max(sample_depths)
                if sample_depths
                else int(math.ceil(math.log2(max(1, len(sample.leaf_states)))))
            )
            if len(sample_depths) != len(sample.node_states):
                sample_depths = [0 for _ in sample.leaf_states] + [
                    0 for _ in sample.node_states[len(sample.leaf_states) :]
                ]
                if sample_depths:
                    sample_depths[-1] = root_depth
            node_targets = _sample_node_scalars(sample, readout)
            exact_node_rows.extend(sample.node_states)
            exact_node_targets.extend(node_targets)
            exact_node_depths.extend(int(depth) for depth in sample_depths)
            exact_node_root_depths.extend(int(root_depth) for _ in sample.node_states)

        if exact_node_rows:
            exact_node_states = _state_tensor(
                exact_node_rows,
                spec=spec,
                device=device,
                state_transform=state_transform,
            )
            exact_node_pred_trans = _predict_transformed_in_chunks(
                model,
                exact_node_states,
                batch_size=eval_batch_size,
            )
            exact_node_pred_scalars = (
                scalar_transform.inverse_tensor(exact_node_pred_trans)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float64)
            )
            for pred_value, target_value, depth, root_depth in zip(
                exact_node_pred_scalars,
                exact_node_targets,
                exact_node_depths,
                exact_node_root_depths,
            ):
                err = abs(float(pred_value) - float(target_value))
                all_node_readout_abs.append(err)
                exact_node_readout_by_depth.setdefault(int(depth), []).append(err)
                if int(depth) == 0:
                    leaf_readout_abs.append(err)
                elif int(depth) == int(root_depth):
                    root_readout_abs.append(err)
                else:
                    internal_readout_abs.append(err)

        groups: dict[int, list[ExactStateSample]] = {}
        for sample in batch_samples:
            groups.setdefault(int(len(sample.leaf_states)), []).append(sample)
        for n_leaves, group_samples in groups.items():
            if n_leaves <= 0:
                continue
            leaf_np = np.stack([sample.leaf_states for sample in group_samples], axis=0).astype(np.float32)
            group_size = int(leaf_np.shape[0])
            state_dim = int(spec.state_dim)
            current = torch.tensor(
                state_transform.transform_np(leaf_np.reshape(-1, state_dim)),
                dtype=torch.float32,
                device=device,
            ).reshape(group_size, int(n_leaves), state_dim)
            current = _encode_leaf_states(model, current.reshape(group_size * int(n_leaves), state_dim)).reshape(
                group_size,
                int(n_leaves),
                state_dim,
            )
            current_depths = [0 for _ in range(int(n_leaves))]
            level_start = 0
            root_depths = [
                max(sample.node_depths or [int(math.ceil(math.log2(max(1, len(sample.leaf_states)))) )])
                for sample in group_samples
            ]
            leaf_is_root = int(n_leaves) == 1
            record_state_values(
                current.reshape(group_size * int(n_leaves), state_dim),
                depth=0,
                is_root=leaf_is_root,
            )
            while int(current.shape[1]) > 1:
                width = int(current.shape[1])
                left_idx = torch.arange(0, width, 2, device=device)
                right_idx = torch.clamp(left_idx + 1, max=width - 1)
                left = current.index_select(1, left_idx)
                right = current.index_select(1, right_idx)
                level_width = int(left.shape[1])
                level_depths = [
                    int(max(current_depths[int(left_i)], current_depths[int(right_i)]) + 1)
                    for left_i, right_i in zip(left_idx.detach().cpu().tolist(), right_idx.detach().cpu().tolist())
                ]
                left_rows = left.reshape(group_size * level_width, state_dim)
                right_rows = right.reshape(group_size * level_width, state_dim)
                level_is_root = int(level_width) == 1
                if hasattr(model, "merge_components"):
                    carrier_rows, residual_rows, pred = model.merge_components(left_rows, right_rows)
                    record_merge_components(carrier_rows, residual_rows, is_root=level_is_root)
                else:
                    pred = model.merge(left_rows, right_rows)
                pred_by_parent = pred.reshape(group_size, level_width, state_dim)
                for local_idx, parent_depth in enumerate(level_depths):
                    record_state_values(
                        pred_by_parent[:, int(local_idx), :],
                        depth=int(parent_depth),
                        is_root=level_is_root,
                    )
                parent_exact_rows: list[np.ndarray] = []
                parent_target_scalars: list[float] = []
                parent_depth_rows: list[int] = []
                parent_root_depth_rows: list[int] = []
                future_context_rows: list[np.ndarray | None] = []
                future_target_scalars: list[float] = []
                for sample, root_depth in zip(group_samples, root_depths):
                    sample_scalars = _sample_node_scalars(sample, readout)
                    first_merge_idx = len(sample.leaf_states)
                    for local_idx, parent_depth in enumerate(level_depths):
                        merge_idx = level_start + int(local_idx)
                        _left_exact, _right_exact, parent_exact = sample.merge_pairs[merge_idx]
                        parent_exact_rows.append(parent_exact)
                        scalar_idx = first_merge_idx + merge_idx
                        if scalar_idx < len(sample_scalars):
                            parent_target_scalars.append(float(sample_scalars[scalar_idx]))
                        else:
                            parent_target_scalars.append(float(readout(parent_exact)))
                        parent_depth_rows.append(int(parent_depth))
                        parent_root_depth_rows.append(int(root_depth))
                        sibling_local_idx = int(local_idx) + (1 if int(local_idx) % 2 == 0 else -1)
                        if hll_alignment_enabled and 0 <= sibling_local_idx < int(level_width):
                            sibling_node_idx = first_merge_idx + level_start + sibling_local_idx
                            if sibling_node_idx < len(sample.node_states):
                                sibling_exact = np.asarray(sample.node_states[sibling_node_idx], dtype=np.float32)
                                future_context_rows.append(sibling_exact)
                                future_target_scalars.append(float(readout(np.maximum(parent_exact, sibling_exact))))
                            else:
                                future_context_rows.append(None)
                                future_target_scalars.append(float("nan"))
                        else:
                            future_context_rows.append(None)
                            future_target_scalars.append(float("nan"))
                target = _state_tensor(
                    parent_exact_rows,
                    spec=spec,
                    device=device,
                    state_transform=state_transform,
                )
                pred_raw_state = state_transform.inverse_tensor(pred)
                target_raw_state = state_transform.inverse_tensor(target)
                state_errs = (
                    (pred_raw_state - target_raw_state)
                    .abs()
                    .mean(dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(np.float64)
                )
                parent_pred_trans = _predict_transformed_in_chunks(
                    model,
                    pred,
                    batch_size=eval_batch_size,
                )
                parent_pred_scalars = (
                    scalar_transform.inverse_tensor(parent_pred_trans)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(np.float64)
                )
                parent_learned_raw = pred_raw_state.detach().cpu().numpy().astype(np.float32)
                parent_official_scalars = [float(readout(row)) for row in parent_learned_raw]
                for (
                    row_idx,
                    state_err,
                    parent_pred_scalar,
                    parent_target_scalar,
                    parent_official_scalar,
                    parent_depth,
                    root_depth,
                ) in zip(
                    range(len(parent_target_scalars)),
                    state_errs,
                    parent_pred_scalars,
                    parent_target_scalars,
                    parent_official_scalars,
                    parent_depth_rows,
                    parent_root_depth_rows,
                ):
                    state_err_value = float(state_err)
                    learned_readout_err = abs(float(parent_pred_scalar) - float(parent_target_scalar))
                    official_readout_err = abs(float(parent_official_scalar) - float(parent_target_scalar))
                    state_mae.append(state_err_value)
                    merge_state_by_depth.setdefault(int(parent_depth), []).append(state_err_value)
                    merge_readout_abs.append(learned_readout_err)
                    official_merge_abs.append(official_readout_err)
                    merge_readout_by_depth.setdefault(int(parent_depth), []).append(learned_readout_err)
                    official_merge_by_depth.setdefault(int(parent_depth), []).append(official_readout_err)
                    if int(parent_depth) == int(root_depth):
                        merge_state_root_abs.append(state_err_value)
                        merge_readout_root_abs.append(learned_readout_err)
                        official_merge_root_abs.append(official_readout_err)
                    else:
                        merge_state_internal_abs.append(state_err_value)
                        merge_readout_internal_abs.append(learned_readout_err)
                        official_merge_internal_abs.append(official_readout_err)
                    if hll_alignment_enabled:
                        learned_registers = parent_learned_raw[int(row_idx)]
                        exact_registers = np.asarray(parent_exact_rows[int(row_idx)], dtype=np.float32)
                        rounded_registers = _round_hll_registers(learned_registers)
                        rounded_abs = np.abs(rounded_registers - exact_registers)
                        rounded_mae = float(np.mean(rounded_abs))
                        rounded_linf = float(np.max(rounded_abs)) if rounded_abs.size else 0.0
                        state_bad = rounded_linf > 1e-6
                        hll_register_fractional_abs.append(
                            float(np.mean(np.abs(learned_registers - np.rint(learned_registers))))
                        )
                        hll_merge_register_rounded_mae.append(rounded_mae)
                        hll_merge_register_linf.append(rounded_linf)
                        hll_merge_register_exact.append(0.0 if state_bad else 1.0)
                        hll_zero_scalar_bad_state.append(
                            1.0 if official_readout_err <= 1e-6 and state_bad else 0.0
                        )
                        hll_within_tol_bad_state.append(
                            1.0 if official_readout_err <= hll_scalar_tol and state_bad else 0.0
                        )
                        context = future_context_rows[int(row_idx)]
                        if context is not None and np.isfinite(future_target_scalars[int(row_idx)]):
                            future_state = np.maximum(rounded_registers, np.asarray(context, dtype=np.float32))
                            future_err = abs(float(readout(future_state)) - float(future_target_scalars[int(row_idx)]))
                            future_bad = future_err > 1e-6
                            hll_future_context_abs.append(future_err)
                            hll_zero_scalar_bad_future.append(
                                1.0 if official_readout_err <= 1e-6 and future_bad else 0.0
                            )
                            hll_within_tol_bad_future.append(
                                1.0 if official_readout_err <= hll_scalar_tol and future_bad else 0.0
                            )
                current = pred.reshape(group_size, level_width, state_dim)
                current_depths = level_depths
                level_start += level_width

            root_states = current.reshape(group_size, state_dim)
            root_pred_trans = _predict_transformed_in_chunks(
                model,
                root_states,
                batch_size=eval_batch_size,
            )
            root_pred_unclamped = _predict_transformed_unclamped_in_chunks(
                model,
                root_states,
                batch_size=eval_batch_size,
            )
            root_pred_scalars = (
                scalar_transform.inverse_tensor(root_pred_trans)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float64)
            )
            learned_root_raw = state_transform.inverse_tensor(root_states).detach().cpu().numpy().astype(np.float32)
            official_learned_scalars = [float(readout(row)) for row in learned_root_raw]
            exact_root_states = _state_tensor(
                [sample.root_state for sample in group_samples],
                spec=spec,
                device=device,
                state_transform=state_transform,
            )
            exact_root_pred_trans = _predict_transformed_in_chunks(
                model,
                exact_root_states,
                batch_size=eval_batch_size,
            )
            exact_root_scalars = (
                scalar_transform.inverse_tensor(exact_root_pred_trans)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float64)
            )
            for sample, pred_scalar, pred_trans_value, exact_root_scalar, official_scalar in zip(
                group_samples,
                root_pred_scalars,
                root_pred_trans.detach().cpu().numpy().reshape(-1).astype(np.float64),
                exact_root_scalars,
                official_learned_scalars,
            ):
                target_scalar = float(sample.root_scalar)
                abs_err = abs(float(pred_scalar) - target_scalar)
                root_abs.append(abs_err)
                root_rel.append(abs_err / max(1.0, abs(target_scalar)))
                exact_err = abs(float(exact_root_scalar) - target_scalar)
                exact_root_abs.append(exact_err)
                exact_root_rel.append(exact_err / max(1.0, abs(target_scalar)))
                official_err = abs(float(official_scalar) - target_scalar)
                official_on_learned_abs.append(official_err)
                official_on_learned_rel.append(official_err / max(1.0, abs(target_scalar)))
                pred_scalars.append(float(pred_scalar))
                pred_transformed.append(float(pred_trans_value))
            pred_transformed_unclamped.extend(
                float(value)
                for value in root_pred_unclamped.detach().cpu().numpy().reshape(-1).astype(np.float64)
            )
    transformed_arr = np.asarray(pred_transformed, dtype=np.float64)
    transformed_unclamped_arr = np.asarray(pred_transformed_unclamped, dtype=np.float64)
    scalar_arr = np.asarray(pred_scalars, dtype=np.float64)
    learned_state_arr = _concat_metric_arrays(learned_state_chunks)
    learned_root_state_arr = _concat_metric_arrays(learned_root_state_chunks)
    learned_nonroot_state_arr = _concat_metric_arrays(learned_nonroot_state_chunks)
    valid_min, valid_max = valid_bounds if valid_bounds is not None else (float("nan"), float("nan"))
    bounded = bool(scalar_transform.bounded_output)
    metrics = {
        "root_mae": _mean_or_nan(root_abs),
        "root_rel_mae": _mean_or_nan(root_rel),
        "learned_f_on_exact_root_mae": _mean_or_nan(exact_root_abs),
        "learned_f_on_exact_root_rel_mae": _mean_or_nan(exact_root_rel),
        "official_f_on_learned_root_mae": _mean_or_nan(official_on_learned_abs),
        "official_f_on_learned_root_rel_mae": _mean_or_nan(official_on_learned_rel),
        "learned_f_on_learned_root_rel_mae": _mean_or_nan(root_rel),
        "leaf_readout_mae": _mean_or_nan(leaf_readout_abs),
        "internal_readout_mae": _mean_or_nan(internal_readout_abs),
        "root_readout_mae": _mean_or_nan(root_readout_abs),
        "all_node_readout_mae": _mean_or_nan(all_node_readout_abs),
        "merge_state_mae": _mean_or_nan(state_mae),
        "merge_state_internal_mae": _mean_or_nan(merge_state_internal_abs),
        "merge_state_root_mae": _mean_or_nan(merge_state_root_abs),
        "merge_readout_mae": _mean_or_nan(merge_readout_abs),
        "merge_readout_internal_mae": _mean_or_nan(merge_readout_internal_abs),
        "merge_readout_root_mae": _mean_or_nan(merge_readout_root_abs),
        "official_merge_readout_mae": _mean_or_nan(official_merge_abs),
        "official_merge_readout_internal_mae": _mean_or_nan(official_merge_internal_abs),
        "official_merge_readout_root_mae": _mean_or_nan(official_merge_root_abs),
        "hll_scalar_alignment_tol": float(hll_scalar_tol) if hll_alignment_enabled else float("nan"),
        "hll_register_fractional_abs_mean": _mean_or_nan(hll_register_fractional_abs),
        "hll_merge_register_rounded_mae": _mean_or_nan(hll_merge_register_rounded_mae),
        "hll_merge_register_linf_mean": _mean_or_nan(hll_merge_register_linf),
        "hll_merge_register_exact_frac": _mean_or_nan(hll_merge_register_exact),
        "hll_zero_scalar_bad_state_frac": _mean_or_nan(hll_zero_scalar_bad_state),
        "hll_within_tol_bad_state_frac": _mean_or_nan(hll_within_tol_bad_state),
        "hll_future_context_readout_mae": _mean_or_nan(hll_future_context_abs),
        "hll_zero_scalar_bad_future_frac": _mean_or_nan(hll_zero_scalar_bad_future),
        "hll_within_tol_bad_future_frac": _mean_or_nan(hll_within_tol_bad_future),
        "pred_scalar_min": float(np.min(scalar_arr)) if scalar_arr.size else float("nan"),
        "pred_scalar_mean": float(np.mean(scalar_arr)) if scalar_arr.size else float("nan"),
        "pred_scalar_max": float(np.max(scalar_arr)) if scalar_arr.size else float("nan"),
        "pred_transformed_min": float(np.min(transformed_arr)) if transformed_arr.size else float("nan"),
        "pred_transformed_mean": float(np.mean(transformed_arr)) if transformed_arr.size else float("nan"),
        "pred_transformed_max": float(np.max(transformed_arr)) if transformed_arr.size else float("nan"),
        "pred_transformed_unclamped_min": (
            float(np.min(transformed_unclamped_arr)) if transformed_unclamped_arr.size else float("nan")
        ),
        "pred_transformed_unclamped_mean": (
            float(np.mean(transformed_unclamped_arr)) if transformed_unclamped_arr.size else float("nan")
        ),
        "pred_transformed_unclamped_max": (
            float(np.max(transformed_unclamped_arr)) if transformed_unclamped_arr.size else float("nan")
        ),
        "hll_readout_preclamp_below_zero_frac": (
            float(np.mean(transformed_unclamped_arr < 0.0))
            if bounded and transformed_unclamped_arr.size
            else float("nan")
        ),
        "hll_readout_preclamp_above_one_frac": (
            float(np.mean(transformed_unclamped_arr > 1.0))
            if bounded and transformed_unclamped_arr.size
            else float("nan")
        ),
        "hll_readout_postclamp_near_zero_frac": (
            float(np.mean(transformed_arr <= 0.01)) if bounded and transformed_arr.size else float("nan")
        ),
        "hll_readout_postclamp_near_one_frac": (
            float(np.mean(transformed_arr >= 0.99)) if bounded and transformed_arr.size else float("nan")
        ),
        "bounded_pred_near_zero_frac": (
            float(np.mean(transformed_arr <= 0.01)) if bounded and transformed_arr.size else float("nan")
        ),
        "bounded_pred_near_one_frac": (
            float(np.mean(transformed_arr >= 0.99)) if bounded and transformed_arr.size else float("nan")
        ),
        "learned_state_valid_min_bound": float(valid_min),
        "learned_state_valid_max_bound": float(valid_max),
        "merge_carrier_state_norm_mean": _mean_or_nan(merge_carrier_norms),
        "merge_projection_delta_norm_mean": _mean_or_nan(merge_projection_delta_norms),
        "merge_projection_delta_to_carrier_norm_mean": _mean_or_nan(merge_projection_delta_to_carrier_norms),
        "merge_projection_delta_to_carrier_norm_root_mean": _mean_or_nan(
            merge_projection_delta_to_carrier_norms_root
        ),
        "merge_projection_delta_to_carrier_norm_nonroot_mean": _mean_or_nan(
            merge_projection_delta_to_carrier_norms_nonroot
        ),
    }
    if valid_bounds is not None:
        metrics.update(
            _state_validity_metrics(
                learned_state_arr,
                prefix="learned_state",
                valid_min=valid_min,
                valid_max=valid_max,
            )
        )
        metrics.update(
            _state_validity_metrics(
                learned_root_state_arr,
                prefix="learned_root_state",
                valid_min=valid_min,
                valid_max=valid_max,
            )
        )
        metrics.update(
            _state_validity_metrics(
                learned_nonroot_state_arr,
                prefix="learned_nonroot_state",
                valid_min=valid_min,
                valid_max=valid_max,
            )
        )
    else:
        for prefix in ("learned_state", "learned_root_state", "learned_nonroot_state"):
            metrics.update(
                {
                    f"{prefix}_min": float("nan"),
                    f"{prefix}_median": float("nan"),
                    f"{prefix}_max": float("nan"),
                    f"{prefix}_below_valid_frac": float("nan"),
                    f"{prefix}_above_valid_frac": float("nan"),
                    f"{prefix}_nonfinite_frac": float("nan"),
                }
            )
    for depth, values in sorted(exact_node_readout_by_depth.items()):
        metrics[f"exact_node_readout_mae_depth_{depth}"] = _mean_or_nan(values)
    if valid_bounds is not None:
        for depth, chunks in sorted(learned_state_chunks_by_depth.items()):
            metrics.update(
                _state_validity_metrics(
                    _concat_metric_arrays(chunks),
                    prefix=f"learned_state_depth_{depth}",
                    valid_min=valid_min,
                    valid_max=valid_max,
                )
            )
    for depth, values in sorted(merge_state_by_depth.items()):
        metrics[f"merge_state_mae_depth_{depth}"] = _mean_or_nan(values)
    for depth, values in sorted(merge_readout_by_depth.items()):
        metrics[f"merge_readout_mae_depth_{depth}"] = _mean_or_nan(values)
    for depth, values in sorted(official_merge_by_depth.items()):
        metrics[f"official_merge_readout_mae_depth_{depth}"] = _mean_or_nan(values)
    return metrics


def _materialize_training_tensors(
    samples: Sequence[ExactStateSample],
    *,
    spec: ExactStateSpec,
    device: torch.device,
    readout: Callable[[np.ndarray], float],
    state_normalization: str,
    target_transform: str,
) -> tuple[dict[str, torch.Tensor], StateTransform, ScalarTransform]:
    f_states: list[np.ndarray] = []
    f_targets: list[float] = []
    g_left: list[np.ndarray] = []
    g_right: list[np.ndarray] = []
    g_target_state: list[np.ndarray] = []
    g_target_scalar: list[float] = []
    for sample in samples:
        node_scalars = _sample_node_scalars(sample, readout)
        for node_idx, state in enumerate(sample.node_states):
            f_states.append(state)
            f_targets.append(float(node_scalars[node_idx]))
        first_merge_idx = len(sample.leaf_states)
        for merge_idx, (left, right, parent) in enumerate(sample.merge_pairs):
            g_left.append(left)
            g_right.append(right)
            g_target_state.append(parent)
            scalar_idx = first_merge_idx + merge_idx
            if scalar_idx < len(node_scalars):
                g_target_scalar.append(float(node_scalars[scalar_idx]))
            else:
                g_target_scalar.append(float(readout(parent)))
    state_transform = _build_state_transform(
        f_states,
        spec=spec,
        kind=str(state_normalization),
    )
    scalar_transform = _build_scalar_transform(
        f_targets,
        spec=spec,
        kind=str(target_transform),
    )
    return {
        "f_states": _state_tensor(f_states, spec=spec, device=device, state_transform=state_transform),
        "f_targets": _scalar_tensor(f_targets, scalar_transform=scalar_transform, device=device),
        "g_left": _state_tensor(g_left, spec=spec, device=device, state_transform=state_transform),
        "g_right": _state_tensor(g_right, spec=spec, device=device, state_transform=state_transform),
        "g_target_state": _state_tensor(g_target_state, spec=spec, device=device, state_transform=state_transform),
        "g_target_scalar": _scalar_tensor(g_target_scalar, scalar_transform=scalar_transform, device=device),
    }, state_transform, scalar_transform


def _run_target(args: argparse.Namespace, target_kind: str, output_dir: Path) -> dict[str, object]:
    target_start = time.perf_counter()
    spec = _target_spec(args, target_kind)
    device = torch.device(str(args.device))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    train_samples, val_samples = _generate_samples(args, spec)
    sample_cache_info = dict(getattr(args, "_sample_cache_last", {}) or {})
    _leaf_state, _merge_state, readout = _state_functions(args, spec)
    tensors, state_transform, scalar_transform = _materialize_training_tensors(
        train_samples,
        spec=spec,
        device=device,
        readout=readout,
        state_normalization=str(args.state_normalization),
        target_transform=str(args.target_transform),
    )
    widths = _resolve_model_widths(args, spec)
    base_lr = 1e-3 if args.learning_rate is None else float(args.learning_rate)
    default_f_lr = 3e-4 if target_kind == "hll_register_space" and args.learning_rate is None else base_lr
    default_g_lr = 1e-4 if target_kind == "hll_register_space" and args.learning_rate is None else base_lr
    f_learning_rate = float(default_f_lr if args.f_learning_rate is None else args.f_learning_rate)
    g_learning_rate = float(default_g_lr if args.g_learning_rate is None else args.g_learning_rate)
    model = SketchStateFNO(
        state_dim=int(spec.state_dim),
        hidden_channels=int(widths["hidden_channels"]),
        n_modes=int(args.n_modes),
        n_layers=int(args.n_layers),
        head_hidden_dim=int(widths["head_hidden_dim"]),
        readout_arch=str(args.readout_arch),
        bounded_output=bool(scalar_transform.bounded_output),
        state_value_scale=float(state_transform.scale),
        target_transform_kind=str(scalar_transform.kind),
        target_scale=float(scalar_transform.scale),
        target_mean=float(scalar_transform.mean),
        target_std=float(scalar_transform.std),
        merge_output_constraint=str(args.merge_output_constraint),
    ).to(device)
    if bool(args.identity_residual_init):
        model.initialize_residuals_as_identity()
    init_payload: dict[str, object] = {}
    if args.init_checkpoint is not None:
        init_payload.update(
            _load_full_model_from_checkpoint(
                model,
                Path(args.init_checkpoint),
                device=device,
            )
        )
    if args.f_init_checkpoint is not None:
        init_payload.update(
            _load_f_components_from_checkpoint(
                model,
                Path(args.f_init_checkpoint),
                device=device,
            )
        )
    target_dir = output_dir / target_kind
    target_dir.mkdir(parents=True, exist_ok=True)
    stage_rows: list[dict[str, object]] = []
    schedule = str(args.schedule)
    schedule_prefix = str(getattr(args, "schedule_prefix", "") or "")
    stage_index_offset = int(getattr(args, "stage_index_offset", 0) or 0)
    if schedule_prefix and stage_index_offset == 0:
        stage_index_offset = len(schedule_prefix)
    display_schedule = f"{schedule_prefix}{schedule}"
    display_stage_total = max(len(display_schedule), stage_index_offset + len(schedule))
    observation_payload = _oracle_observation_payload(args)
    sampled_node_rate_internal = float(getattr(args, "_sampled_node_rate_internal", 0.0))
    eval_batch_size = _resolve_eval_batch_size(args)
    print(
        "[fno-sketch] prepared "
        f"target={target_kind} train_docs={len(train_samples)} val_docs={len(val_samples)} "
        f"leaves={int(args.n_leaves)} objective={args.objective_mode} "
        f"obs={_oracle_observation_design_name(args)} batch_size={int(args.batch_size)} "
        f"eval_batch_size={int(eval_batch_size)} "
        f"rollout_min_docs_per_batch={int(args.rollout_min_docs_per_batch)} "
        f"rollout_max_docs_per_batch={int(args.rollout_max_docs_per_batch)} "
        f"local_law_leaf_discount_gamma={float(args.local_law_leaf_discount_gamma):g} "
        "merge_adapter=induced_projection "
        f"merge_output_constraint={args.merge_output_constraint} "
        f"cache={sample_cache_info.get('sample_cache_status', 'none')}",
        flush=True,
    )
    for local_stage_index, component in enumerate(schedule, start=1):
        stage_index = stage_index_offset + local_stage_index
        progress_label = (
            f"target={target_kind} stage={stage_index}/{display_stage_total} "
            f"component={component} objective={args.objective_mode} "
            f"obs={_oracle_observation_design_name(args)}"
        )
        print(
            f"[fno-sketch] start {progress_label} epochs={int(args.epochs)}",
            flush=True,
        )

        def eval_callback(epoch: int, train_loss: float) -> dict[str, float]:
            if int(args.eval_every_epochs) <= 0:
                return {}
            if epoch % int(args.eval_every_epochs) != 0 and epoch != int(args.epochs):
                return {}
            row = _evaluate(
                model,
                val_samples,
                spec=spec,
                device=device,
                state_transform=state_transform,
                scalar_transform=scalar_transform,
                readout=readout,
                batch_size=eval_batch_size,
            )
            row["stage_index"] = float(stage_index)
            row["component_is_f"] = float(component == "f")
            row["component_is_g"] = float(component == "g")
            return row

        if component == "f":
            if str(args.objective_mode) == "rollout_local_law":
                losses, epoch_metrics, train_components = _train_f_stage_rollout(
                    model,
                    samples=train_samples,
                    spec=spec,
                    device=device,
                    state_transform=state_transform,
                    scalar_transform=scalar_transform,
                    readout=readout,
                    eval_callback=eval_callback,
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    rollout_min_docs_per_batch=int(args.rollout_min_docs_per_batch),
                    rollout_max_docs_per_batch=int(args.rollout_max_docs_per_batch),
                    learning_rate=float(f_learning_rate),
                    weight_decay=float(args.weight_decay),
                    grad_clip_norm=float(args.grad_clip_norm),
                    grad_accum_steps=int(args.grad_accum_steps),
                    local_law_weight=float(args.local_law_weight),
                    local_law_leaf_discount_gamma=float(args.local_law_leaf_discount_gamma),
                    objective_loss_weight=float(args.objective_loss_weight),
                    exact_state_anchor_weight=float(args.exact_state_anchor_weight),
                    oracle_observation_design=_oracle_observation_design_name(args),
                    sampled_node_rate=sampled_node_rate_internal,
                    sampled_node_seed=int(args.seed),
                    root_label_share=float(args.root_label_share),
                    mass_target_per_doc=float(args.mass_target_per_doc),
                    local_label_pool=str(args.local_label_pool),
                    local_label_allocation=str(args.local_label_allocation),
                    progress_label=progress_label,
                    progress_every_epochs=int(args.progress_every_epochs),
                    progress_every_batches=int(args.progress_every_batches),
                )
            else:
                losses, epoch_metrics = _train_f_stage(
                    model,
                    states=tensors["f_states"],
                    targets=tensors["f_targets"],
                    eval_callback=eval_callback,
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    learning_rate=float(f_learning_rate),
                    weight_decay=float(args.weight_decay),
                    grad_clip_norm=float(args.grad_clip_norm),
                    grad_accum_steps=int(args.grad_accum_steps),
                    progress_label=progress_label,
                    progress_every_epochs=int(args.progress_every_epochs),
                    progress_every_batches=int(args.progress_every_batches),
                )
                train_components = []
        elif component == "g":
            if str(args.objective_mode) == "rollout_local_law":
                losses, epoch_metrics, train_components = _train_g_stage_rollout(
                    model,
                    samples=train_samples,
                    spec=spec,
                    device=device,
                    state_transform=state_transform,
                    scalar_transform=scalar_transform,
                    readout=readout,
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    rollout_min_docs_per_batch=int(args.rollout_min_docs_per_batch),
                    rollout_max_docs_per_batch=int(args.rollout_max_docs_per_batch),
                    learning_rate=float(g_learning_rate),
                    weight_decay=float(args.weight_decay),
                    grad_clip_norm=float(args.grad_clip_norm),
                    grad_accum_steps=int(args.grad_accum_steps),
                    state_loss_weight=float(args.state_loss_weight),
                    local_law_weight=float(args.local_law_weight),
                    local_law_leaf_discount_gamma=float(args.local_law_leaf_discount_gamma),
                    objective_loss_weight=float(args.objective_loss_weight),
                    oracle_observation_design=_oracle_observation_design_name(args),
                    sampled_node_rate=sampled_node_rate_internal,
                    sampled_node_seed=int(args.seed),
                    root_label_share=float(args.root_label_share),
                    mass_target_per_doc=float(args.mass_target_per_doc),
                    local_label_pool=str(args.local_label_pool),
                    local_label_allocation=str(args.local_label_allocation),
                    eval_callback=eval_callback,
                    progress_label=progress_label,
                    progress_every_epochs=int(args.progress_every_epochs),
                    progress_every_batches=int(args.progress_every_batches),
                )
            else:
                losses, epoch_metrics = _train_g_stage(
                    model,
                    left=tensors["g_left"],
                    right=tensors["g_right"],
                    target_state=tensors["g_target_state"],
                    target_scalar=tensors["g_target_scalar"],
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    learning_rate=float(g_learning_rate),
                    weight_decay=float(args.weight_decay),
                    grad_clip_norm=float(args.grad_clip_norm),
                    grad_accum_steps=int(args.grad_accum_steps),
                    state_loss_weight=float(args.state_loss_weight),
                    scalar_loss_weight=float(args.scalar_loss_weight),
                    eval_callback=eval_callback,
                    progress_label=progress_label,
                    progress_every_epochs=int(args.progress_every_epochs),
                    progress_every_batches=int(args.progress_every_batches),
                )
                train_components = []
        else:
            raise ValueError(f"schedule contains unsupported component {component!r}")
        metrics = _evaluate(
            model,
            val_samples,
            spec=spec,
            device=device,
            state_transform=state_transform,
            scalar_transform=scalar_transform,
            readout=readout,
            batch_size=eval_batch_size,
        )
        _assert_finite_metrics(
            metrics,
            ("root_mae", "official_f_on_learned_root_mae", "learned_f_on_exact_root_mae"),
            context=f"{progress_label} final eval",
        )
        print(
            f"[fno-sketch] finish {progress_label} "
            f"root_mae={_fmt_progress_value(metrics.get('root_mae'))} "
            f"official_f_on_learned_root_mae={_fmt_progress_value(metrics.get('official_f_on_learned_root_mae'))} "
            f"merge_state_mae={_fmt_progress_value(metrics.get('merge_state_mae'))}",
            flush=True,
        )
        objective_payload = _canonical_objective_payload(float(args.local_law_weight))
        stage_row = {
            **objective_payload,
            "target_kind": target_kind,
            "stage_index": stage_index,
            "local_stage_index": local_stage_index,
            "component": component,
            "schedule": display_schedule,
            "trained_schedule": schedule,
            "schedule_prefix": schedule_prefix,
            "stage_index_offset": stage_index_offset,
            "objective_mode": str(args.objective_mode),
            "objective_ablation": bool(str(args.objective_mode) == "exact_rows"),
            **observation_payload,
            "local_law_weight": float(args.local_law_weight),
            "local_law_leaf_discount_gamma": float(args.local_law_leaf_discount_gamma),
            "objective_loss_weight": float(args.objective_loss_weight),
            "exact_state_anchor_weight": float(args.exact_state_anchor_weight),
            "state_loss_weight": float(args.state_loss_weight),
            "merge_adapter": "induced_projection",
            "lean_merge_adapter": "merge(a,b)=g_theta(a+b); encode_leaf(x)=g_theta(x)",
            "lean_projection_target": "f*(x+y)=f*(g*(g*(x)+g*(y)))",
            "merge_output_constraint": str(args.merge_output_constraint),
            "learning_rate": float(f_learning_rate if component == "f" else g_learning_rate),
            "loss_start": losses[0] if losses else float("nan"),
            "loss_end": losses[-1] if losses else float("nan"),
            "loss_min": min(losses) if losses else float("nan"),
            "train_objective_loss_end": (
                train_components[-1].get("train_objective_loss", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_root_loss_end": (
                train_components[-1].get("train_root_loss", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_local_loss_end": (
                train_components[-1].get("train_local_loss", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_local_proxy_loss_end": (
                train_components[-1].get("train_local_proxy_loss", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_local_oracle_observed_ipw_loss_end": (
                train_components[-1].get("train_local_oracle_observed_ipw_loss", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_local_ipw_correction_end": (
                train_components[-1].get("train_local_ipw_correction", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_local_corrected_loss_end": (
                train_components[-1].get("train_local_corrected_loss", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_discounted_root_weight_end": (
                train_components[-1].get("train_discounted_root_weight", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_discounted_nonroot_weight_end": (
                train_components[-1].get("train_discounted_nonroot_weight", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_exact_anchor_loss_end": (
                train_components[-1].get("train_exact_anchor_loss", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_state_regularizer_loss_end": (
                train_components[-1].get("train_state_regularizer_loss", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_observed_rows_end": (
                train_components[-1].get("train_observed_rows", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_population_rows_end": (
                train_components[-1].get("train_population_rows", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_root_observed_rows_end": (
                train_components[-1].get("train_root_observed_rows", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_root_population_rows_end": (
                train_components[-1].get("train_root_population_rows", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_nonroot_observed_rows_end": (
                train_components[-1].get("train_nonroot_observed_rows", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_nonroot_population_rows_end": (
                train_components[-1].get("train_nonroot_population_rows", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_observed_rows_per_doc_end": (
                train_components[-1].get("train_observed_rows_per_doc", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_root_observed_rows_per_doc_end": (
                train_components[-1].get("train_root_observed_rows_per_doc", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_nonroot_observed_rows_per_doc_end": (
                train_components[-1].get("train_nonroot_observed_rows_per_doc", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_max_ipw_weight_end": (
                train_components[-1].get("train_max_ipw_weight", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_effective_sample_size_end": (
                train_components[-1].get("train_effective_sample_size", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_observed_mass_end": (
                train_components[-1].get("train_observed_mass", float("nan"))
                if train_components
                else float("nan")
            ),
            "train_population_mass_end": (
                train_components[-1].get("train_population_mass", float("nan"))
                if train_components
                else float("nan")
            ),
            **metrics,
        }
        for key in LOSS_BUCKET_METRIC_KEYS:
            stage_row[f"train_{key}_end"] = (
                train_components[-1].get(f"train_{key}", float("nan"))
                if train_components
                else float("nan")
            )
        stage_rows.append(stage_row)
        stage_loss_payload = {
            **objective_payload,
            "component": component,
            "stage_index": stage_index,
            "local_stage_index": local_stage_index,
            "schedule": display_schedule,
            "trained_schedule": schedule,
            "schedule_prefix": schedule_prefix,
            "stage_index_offset": stage_index_offset,
            "objective_mode": str(args.objective_mode),
            "objective_ablation": bool(str(args.objective_mode) == "exact_rows"),
            **observation_payload,
            "local_law_leaf_discount_gamma": float(args.local_law_leaf_discount_gamma),
            "merge_adapter": "induced_projection",
            "lean_merge_adapter": "merge(a,b)=g_theta(a+b); encode_leaf(x)=g_theta(x)",
            "lean_projection_target": "f*(x+y)=f*(g*(g*(x)+g*(y)))",
            "merge_output_constraint": str(args.merge_output_constraint),
            "epoch_mean_losses": losses,
            "epoch_metrics": epoch_metrics,
            "epoch_train_components": train_components,
        }
        assert_public_contract_clean(
            stage_loss_payload,
            surface=f"{target_kind} stage {stage_index} loss payload",
        )
        (target_dir / f"stage_{stage_index:02d}_{component}_losses.json").write_text(
            json.dumps(
                stage_loss_payload,
                indent=2,
                allow_nan=True,
            )
            + "\n",
            encoding="utf-8",
        )
        torch.save(model.state_dict(), target_dir / f"stage_{stage_index:02d}_{component}_model.pt")
    final_metrics = _evaluate(
        model,
        val_samples,
        spec=spec,
        device=device,
        state_transform=state_transform,
        scalar_transform=scalar_transform,
        readout=readout,
        batch_size=eval_batch_size,
    )
    spec_payload = asdict(spec)
    objective_payload = _canonical_objective_payload(float(args.local_law_weight))
    final_stage = stage_rows[-1] if stage_rows else {}
    final_train_accounting = {
        key: final_stage.get(key, float("nan"))
        for key in (
            "train_objective_loss_end",
            "train_root_loss_end",
            "train_local_loss_end",
            "train_local_proxy_loss_end",
            "train_local_oracle_observed_ipw_loss_end",
            "train_local_ipw_correction_end",
            "train_local_corrected_loss_end",
            "train_discounted_root_weight_end",
            "train_discounted_nonroot_weight_end",
            "train_exact_anchor_loss_end",
            "train_state_regularizer_loss_end",
            "train_observed_rows_end",
            "train_population_rows_end",
            "train_root_observed_rows_end",
            "train_root_population_rows_end",
            "train_nonroot_observed_rows_end",
            "train_nonroot_population_rows_end",
            "train_observed_rows_per_doc_end",
            "train_root_observed_rows_per_doc_end",
            "train_nonroot_observed_rows_per_doc_end",
            "train_max_ipw_weight_end",
            "train_effective_sample_size_end",
            "train_observed_mass_end",
            "train_population_mass_end",
        )
    }
    for key in LOSS_BUCKET_METRIC_KEYS:
        final_train_accounting[f"train_{key}_end"] = final_stage.get(f"train_{key}_end", float("nan"))
    row: dict[str, object] = {
        **objective_payload,
        "target_kind": target_kind,
        "schedule": display_schedule,
        "trained_schedule": schedule,
        "schedule_prefix": schedule_prefix,
        "stage_index_offset": stage_index_offset,
        "state_dim": int(spec.state_dim),
        "g_input_dim": int(spec.state_dim),
        "raw_input_kind": "synthetic_token_ids",
        "leaf_state_kind": "exact_numeric_state_from_raw_tokens",
        "hidden_channels": int(widths["hidden_channels"]),
        "head_hidden_dim": int(widths["head_hidden_dim"]),
        "hidden_width_floor": int(widths["hidden_width_floor"]),
        "head_width_floor": int(widths["head_width_floor"]),
        "width_floor_multiplier": int(widths["width_floor_multiplier"]),
        "n_modes_resolved": int(widths["n_modes_resolved"]),
        "identity_residual_init": bool(args.identity_residual_init),
        "merge_adapter": "induced_projection",
        "lean_merge_adapter": "merge(a,b)=g_theta(a+b); encode_leaf(x)=g_theta(x)",
        "lean_projection_target": "f*(x+y)=f*(g*(g*(x)+g*(y)))",
        "merge_output_constraint": str(args.merge_output_constraint),
        "readout_arch": str(args.readout_arch),
        "f_learning_rate": float(f_learning_rate),
        "g_learning_rate": float(g_learning_rate),
        "objective_mode": str(args.objective_mode),
        "objective_ablation": bool(str(args.objective_mode) == "exact_rows"),
        **observation_payload,
        "local_law_weight": float(args.local_law_weight),
        "local_law_leaf_discount_gamma": float(args.local_law_leaf_discount_gamma),
        "objective_loss_weight": float(args.objective_loss_weight),
        "exact_state_anchor_weight": float(args.exact_state_anchor_weight),
        "state_loss_weight": float(args.state_loss_weight),
        "scalar_loss_weight": float(args.scalar_loss_weight),
        "eval_every_epochs": int(args.eval_every_epochs),
        "merge_kind": spec.merge_kind,
        "readout_kind": spec.readout_kind,
        "n_train": int(args.n_train),
        "n_val": int(args.n_val),
        "n_leaves": int(args.n_leaves),
        "epochs_per_stage": int(args.epochs),
        "batch_size": int(args.batch_size),
        "rollout_min_docs_per_batch": int(args.rollout_min_docs_per_batch),
        "rollout_max_docs_per_batch": int(args.rollout_max_docs_per_batch),
        "eval_batch_size_rows": int(eval_batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "effective_batch_size_rows": int(args.batch_size) * max(1, int(args.grad_accum_steps)),
        "batching_unit": "local_state_rows",
        "f_training_rows": int(tensors["f_states"].shape[0]),
        "g_training_rows": int(tensors["g_left"].shape[0]),
        "f_batches_per_epoch": int(math.ceil(float(tensors["f_states"].shape[0]) / max(1, int(args.batch_size)))),
        "g_batches_per_epoch": int(math.ceil(float(tensors["g_left"].shape[0]) / max(1, int(args.batch_size)))),
        "device": str(device),
        "wall_seconds": float(time.perf_counter() - target_start),
        "sample_cache_status": str(sample_cache_info.get("sample_cache_status", "")),
        "sample_cache_key": str(sample_cache_info.get("sample_cache_key", "")),
        "sample_cache_path": str(sample_cache_info.get("sample_cache_path", "")),
        **final_train_accounting,
        **final_metrics,
        **state_transform.metadata(),
        **scalar_transform.metadata(),
        **init_payload,
    }
    transform_payload = {
        **state_transform.metadata(),
        **scalar_transform.metadata(),
        "state_zscore_mean": state_transform.mean.tolist() if state_transform.mean is not None else None,
        "state_zscore_std": state_transform.std.tolist() if state_transform.std is not None else None,
    }
    assert_public_contract_clean(
        stage_rows,
        surface=f"{target_kind} stage metrics",
    )
    assert_public_contract_clean(
        row,
        surface=f"{target_kind} summary row",
    )
    (target_dir / "stage_metrics.json").write_text(
        json.dumps(stage_rows, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    (target_dir / "spec.json").write_text(json.dumps(spec_payload, indent=2) + "\n", encoding="utf-8")
    (target_dir / "transforms.json").write_text(json.dumps(transform_payload, indent=2) + "\n", encoding="utf-8")
    return row


def _write_report(rows: Sequence[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_public_contract_clean(rows, surface="FNO mergeable sketch diagnostic summary")
    if rows:
        columns = list(rows[0].keys())
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# FNO Mergeable Sketch Diagnostic",
        "",
        "| target | state_dim | schedule | objective | root_mae | root_rel_mae | merge_state_mae | seconds |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        objective = str(row.get("objective_mode", ""))
        if bool(row.get("objective_ablation", False)):
            objective = f"{objective} (ablation)"
        lines.append(
            "| {target_kind} | {state_dim} | {schedule} | {objective} | {root_mae:.6g} | "
            "{root_rel_mae:.6g} | {merge_state_mae:.6g} | {wall_seconds:.1f} |".format(
                objective=objective,
                **row,
            )
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(list(rows), indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default="hll_register_space,exact_frequency_state_space,count_min_state_space,exact_total_weight_state_space")
    parser.add_argument("--schedule", default="fg")
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-val", type=int, default=512)
    parser.add_argument("--n-leaves", type=int, default=4)
    parser.add_argument("--min-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--universe-size", type=int, default=512)
    parser.add_argument("--zipf-alphas", default="0.8,1.0,1.2")
    parser.add_argument("--focus-token", type=int, default=0)
    parser.add_argument("--precision", type=int, default=8)
    parser.add_argument("--cms-num-hashes", type=int, default=4)
    parser.add_argument("--cms-num-buckets", type=int, default=128)
    parser.add_argument(
        "--readout-arch",
        choices=("fno_mlp", "head_only", "deep_mlp", "hll_formula", "hll_residual"),
        default="fno_mlp",
    )
    parser.add_argument(
        "--target-transform",
        choices=("linear01", "log1p_zscore", "zscore"),
        default="linear01",
    )
    parser.add_argument(
        "--state-normalization",
        choices=("register_div64", "register_div8", "zscore"),
        default="register_div64",
    )
    parser.add_argument(
        "--hidden-channels",
        default="auto",
        help="FNO lift width, integer or 'auto'. Auto uses max(128, 2*state_dim).",
    )
    parser.add_argument("--n-modes", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument(
        "--head-hidden-dim",
        default="auto",
        help="Readout MLP hidden width, integer or 'auto'. Auto uses max(128, 2*state_dim).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--rollout-min-docs-per-batch",
        type=int,
        default=1,
        help=(
            "Minimum number of documents per rollout training batch. "
            "This prevents deep trees from degenerating into tiny level-batched merge calls "
            "when --batch-size is a small node-row budget."
        ),
    )
    parser.add_argument(
        "--rollout-max-docs-per-batch",
        type=int,
        default=0,
        help="Optional maximum documents per rollout training batch; 0 disables the cap.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="Validation/eval row batch target. Default 0 uses max(batch-size, 16384).",
    )
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Legacy shared LR. If omitted, HLL defaults to f=3e-4 and g=1e-4.",
    )
    parser.add_argument("--f-learning-rate", type=float, default=None)
    parser.add_argument("--g-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--state-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--objective-mode",
        choices=("rollout_local_law", "exact_rows"),
        default="rollout_local_law",
        help="rollout_local_law trains on current learned g rollouts; exact_rows keeps the legacy exact-state row ablation.",
    )
    parser.add_argument(
        "--oracle-observation-design",
        choices=("root_only", "sampled_nodes", "sampled_root_nodes", "dense_oracle", "budgeted_mass"),
        default="root_only",
        help="Oracle labels exposed to corrected local-law rows. sampled_root_nodes independently samples roots and non-root nodes.",
    )
    parser.add_argument(
        "--sampled-node-rate",
        type=float,
        default=None,
        help=(
            "Non-root node inclusion probability for --oracle-observation-design sampled_nodes. "
            "Required in sampled_nodes mode; inactive and omitted from artifacts otherwise."
        ),
    )
    parser.add_argument(
        "--root-label-share",
        type=float,
        default=1.0,
        help="Root label probability for --oracle-observation-design=budgeted_mass, e.g. 0.1 for R10.",
    )
    parser.add_argument(
        "--mass-target-per-doc",
        type=float,
        default=1.0,
        help="Target expected oracle supervision mass per doc for budgeted_mass.",
    )
    parser.add_argument(
        "--local-label-pool",
        choices=("nonroot", "leaves", "internal"),
        default="nonroot",
        help="Node pool used to reallocate missing root mass in budgeted_mass mode.",
    )
    parser.add_argument(
        "--local-label-allocation",
        choices=("span_mass",),
        default="span_mass",
        help="Budgeted-mass local allocation rule.",
    )
    parser.add_argument("--local-law-weight", type=float, default=0.5)
    parser.add_argument(
        "--local-law-leaf-discount-gamma",
        type=float,
        default=1.0,
        help=(
            "Discount local-law rows by gamma^(root_depth - node_depth), so leaves "
            "are downweighted relative to the root. 1.0 disables discounting."
        ),
    )
    parser.add_argument(
        "--merge-output-constraint",
        choices=("none", "unit_clamp"),
        default="none",
        help=(
            "Diagnostic constraint applied to learned g outputs. unit_clamp keeps "
            "register_div64 HLL states in the normalized [0, 1] register range."
        ),
    )
    parser.add_argument("--objective-loss-weight", type=float, default=None)
    parser.add_argument("--exact-state-anchor-weight", type=float, default=0.1)
    parser.add_argument(
        "--scalar-loss-weight",
        type=float,
        default=None,
        help="Backward-compatible alias for --objective-loss-weight in this diagnostic.",
    )
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument(
        "--progress-every-epochs",
        type=int,
        default=1,
        help="Print flushed stage progress every N epochs. Use 0 to suppress epoch progress.",
    )
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=0,
        help="Print flushed in-epoch progress every N batches. Use 0 to suppress batch progress.",
    )
    parser.add_argument(
        "--f-init-checkpoint",
        type=Path,
        default=None,
        help="Optional state_dict checkpoint used to warm-start f_fno + score_head only.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional full model state_dict checkpoint used to continue a schedule suffix.",
    )
    parser.add_argument(
        "--schedule-prefix",
        default="",
        help=(
            "Previously completed schedule prefix when --init-checkpoint starts from an "
            "intermediate stage, e.g. gfgf while running suffix gf."
        ),
    )
    parser.add_argument(
        "--stage-index-offset",
        type=int,
        default=0,
        help="Number of already-completed stages; defaults to len(--schedule-prefix) when a prefix is supplied.",
    )
    parser.add_argument(
        "--sample-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for persisted exact state trees and precomputed oracle "
            "node scores. Cells with matching data-generation config reuse the same cache."
        ),
    )
    parser.add_argument(
        "--precompute-samples-only",
        action="store_true",
        help="Populate --sample-cache-dir for the requested targets and exit before model training.",
    )
    parser.add_argument(
        "--identity-residual-init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize f/g FNO residual operators to zero so the state path starts as identity/average.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    targets = _parse_csv(args.targets)
    bad_targets = [target for target in targets if target not in MERGEABLE_TARGETS]
    if bad_targets:
        raise ValueError(f"unsupported target(s): {bad_targets}; expected one of {MERGEABLE_TARGETS}")
    if str(args.readout_arch) in {"hll_formula", "hll_residual"}:
        if any(target != "hll_register_space" for target in targets):
            raise ValueError("--readout-arch hll_formula/hll_residual is only valid for hll_register_space")
        if str(args.state_normalization) == "zscore":
            raise ValueError("--readout-arch hll_formula/hll_residual requires register_div64 or register_div8 state normalization")
    if any(ch not in {"f", "g"} for ch in str(args.schedule)):
        raise ValueError("--schedule must be a string over {'f','g'}, e.g. fg or fgfg")
    if any(ch not in {"f", "g"} for ch in str(args.schedule_prefix)):
        raise ValueError("--schedule-prefix must be a string over {'f','g'}, e.g. gfgf")
    if int(args.stage_index_offset) < 0:
        raise ValueError("--stage-index-offset must be non-negative")
    if args.init_checkpoint is not None and args.f_init_checkpoint is not None:
        raise ValueError("--init-checkpoint and --f-init-checkpoint are mutually exclusive")
    if (str(args.schedule_prefix) or int(args.stage_index_offset) > 0) and args.init_checkpoint is None:
        raise ValueError("--schedule-prefix/--stage-index-offset require --init-checkpoint")
    if int(args.epochs) <= 0:
        raise ValueError("--epochs must be positive")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if int(args.rollout_min_docs_per_batch) <= 0:
        raise ValueError("--rollout-min-docs-per-batch must be positive")
    if int(args.rollout_max_docs_per_batch) < 0:
        raise ValueError("--rollout-max-docs-per-batch must be non-negative")
    if (
        int(args.rollout_max_docs_per_batch) > 0
        and int(args.rollout_max_docs_per_batch) < int(args.rollout_min_docs_per_batch)
    ):
        raise ValueError("--rollout-max-docs-per-batch must be 0 or at least --rollout-min-docs-per-batch")
    if int(args.grad_accum_steps) <= 0:
        raise ValueError("--grad-accum-steps must be positive")
    if int(args.eval_every_epochs) < 0:
        raise ValueError("--eval-every-epochs must be non-negative")
    if int(args.progress_every_epochs) < 0:
        raise ValueError("--progress-every-epochs must be non-negative")
    if int(args.progress_every_batches) < 0:
        raise ValueError("--progress-every-batches must be non-negative")
    if int(args.n_train) <= 0 or int(args.n_val) <= 0:
        raise ValueError("--n-train and --n-val must be positive")
    if int(args.n_leaves) <= 0:
        raise ValueError("--n-leaves must be positive")
    if float(args.local_law_weight) < 0.0 or float(args.local_law_weight) > 1.0:
        raise ValueError("--local-law-weight must be in [0, 1]")
    if float(args.local_law_leaf_discount_gamma) < 0.0:
        raise ValueError("--local-law-leaf-discount-gamma must be non-negative")
    if str(args.merge_output_constraint) != "none" and str(args.state_normalization) != "register_div64":
        raise ValueError("--merge-output-constraint=unit_clamp is only supported with --state-normalization=register_div64")
    args._sampled_node_rate_internal = _resolve_sampled_node_rate(args)
    if float(args.root_label_share) < 0.0 or float(args.root_label_share) > 1.0:
        raise ValueError("--root-label-share must be in [0, 1]")
    if float(args.mass_target_per_doc) < 0.0:
        raise ValueError("--mass-target-per-doc must be non-negative")
    if args.objective_loss_weight is None:
        args.objective_loss_weight = (
            1.0 if args.scalar_loss_weight is None else float(args.scalar_loss_weight)
        )
    elif args.scalar_loss_weight is not None and not math.isclose(
        float(args.objective_loss_weight),
        float(args.scalar_loss_weight),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("--scalar-loss-weight is an alias for --objective-loss-weight; pass only one value")
    args.objective_loss_weight = float(args.objective_loss_weight)
    args.scalar_loss_weight = float(args.objective_loss_weight)
    if float(args.objective_loss_weight) < 0.0:
        raise ValueError("--objective-loss-weight must be non-negative")
    if float(args.exact_state_anchor_weight) < 0.0:
        raise ValueError("--exact-state-anchor-weight must be non-negative")
    if float(args.state_loss_weight) < 0.0:
        raise ValueError("--state-loss-weight must be non-negative")
    if int(args.eval_batch_size) < 0:
        raise ValueError("--eval-batch-size must be non-negative; use 0 for auto")
    if bool(args.precompute_samples_only) and args.sample_cache_dir is None:
        raise ValueError("--precompute-samples-only requires --sample-cache-dir")

    if bool(args.precompute_samples_only):
        for target in targets:
            spec = _target_spec(args, target)
            train_samples, val_samples = _generate_samples(args, spec)
            cache_info = dict(getattr(args, "_sample_cache_last", {}) or {})
            print(
                "[fno-sketch] precomputed "
                f"target={target} train={len(train_samples)} val={len(val_samples)} "
                f"cache_status={cache_info.get('sample_cache_status', '')} "
                f"cache_path={cache_info.get('sample_cache_path', '')}",
                flush=True,
            )
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for target in targets:
        print(f"[fno-sketch] target={target} schedule={args.schedule} device={args.device}", flush=True)
        rows.append(_run_target(args, target, output_dir))
        _write_report(rows, output_dir)
    _write_report(rows, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
